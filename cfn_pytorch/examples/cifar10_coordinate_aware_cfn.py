import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, ParallelCompositionLayer
from cfn_pytorch.trainer import Trainer

class StabilizedHierarchicalCoordinateAwareCFN(nn.Module):
    """
    A stabilized hierarchical CFN with numerical safeguards to prevent NaN issues.
    """
    def __init__(self, image_size=(32, 32), n_channels=3, n_classes=10, patch_size=8, stride=4):
        super().__init__()
        if stride is None:
            stride = patch_size // 2
        self.image_size = image_size
        self.n_channels = n_channels

        # Patch-level parameters
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches_per_dim = (image_size[0] - patch_size) // stride + 1
        self.num_patches = self.num_patches_per_dim * self.num_patches_per_dim
        self.pixels_per_patch = patch_size * patch_size

        # --- Coordinate Grids ---
        # For pixels within a patch
        x_coords_patch = torch.linspace(-1, 1, patch_size)
        y_coords_patch = torch.linspace(-1, 1, patch_size)
        grid_x_patch, grid_y_patch = torch.meshgrid(x_coords_patch, y_coords_patch, indexing='ij')
        self.register_buffer('coordinate_grid_patch', torch.stack([grid_x_patch.flatten(), grid_y_patch.flatten()], dim=1))

        # For patches within the image
        x_coords_image = torch.linspace(-1, 1, self.num_patches_per_dim)
        y_coords_image = torch.linspace(-1, 1, self.num_patches_per_dim)
        grid_x_image, grid_y_image = torch.meshgrid(x_coords_image, y_coords_image, indexing='ij')
        self.register_buffer('coordinate_grid_image', torch.stack([grid_x_image.flatten(), grid_y_image.flatten()], dim=1))

        # --- Define the CFN Architecture ---
        node_factory = FunctionNodeFactory()
        
        # --- Layer 1: Simplified Patch-level CFN ---
        pixel_input_dim = 2 + n_channels
        
        # Create a simpler, more stable parallel layer
        self.layer1_pixel_cfn = ParallelCompositionLayer(
            function_nodes=[
                # Simple linear and ReLU block for stability
                node_factory.create('Linear', input_dim=pixel_input_dim, output_dim=64),
                node_factory.create('ReLU', input_dim=pixel_input_dim),
                # Simple polynomial for nonlinearity
                node_factory.create('Polynomial', input_dim=pixel_input_dim, degree=2),
                # Gaussian for capturing local patterns
                node_factory.create('Gaussian', input_dim=pixel_input_dim),
            ],
            combination='concat'
        )
        
        # Simplified patch operations - use only Gabor for now
        patch_input_dim = patch_size * patch_size * n_channels
        self.use_gabor = True
        
        if self.use_gabor:
            self.gabor_node = node_factory.create('Gabor', 
                                                 input_dim=patch_input_dim, 
                                                 image_size=(patch_size, patch_size), 
                                                 n_channels=n_channels)
        
        layer1_pixel_output_dim = self.layer1_pixel_cfn.output_dim
        gabor_output_dim = self.gabor_node.output_dim if self.use_gabor else 0
        
        # Use only mean and max aggregation for stability
        layer1_output_dim_per_patch = layer1_pixel_output_dim * 2 + gabor_output_dim

        # --- Layer 2: Simplified Inter-Patch CFN ---
        patch_input_dim = 2 + layer1_output_dim_per_patch
        self.layer2_inter_patch_cfn = ParallelCompositionLayer(
            function_nodes=[
                # Simple linear and ReLU block for stability
                node_factory.create('Linear', input_dim=patch_input_dim, output_dim=128),
                node_factory.create('ReLU', input_dim=patch_input_dim),
                # Add one polynomial for nonlinearity
                node_factory.create('Polynomial', input_dim=patch_input_dim, degree=2),
                # Add one sinusoidal for capturing periodic patterns
                node_factory.create('Sinusoidal', input_dim=patch_input_dim),
            ],
            combination='concat'
        )
        layer2_output_dim_per_patch = self.layer2_inter_patch_cfn.output_dim
        
        # --- Single Attention Head for Stability ---
        self.patch_attention = node_factory.create('MathematicalAttention', input_dim=layer2_output_dim_per_patch)
        
        # Simpler aggregation with just mean and max
        layer2_output_dim_image = layer2_output_dim_per_patch * 2

        # --- Simplified Classifier Head ---
        self.classifier_head = SequentialCompositionLayer(
            function_nodes=[
                node_factory.create('Linear', input_dim=layer2_output_dim_image, output_dim=256),
                node_factory.create('ReLU', input_dim=256),
                node_factory.create('Dropout', input_dim=256, p=0.2),  # Reduced dropout
                node_factory.create('Linear', input_dim=256, output_dim=n_classes),
            ]
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_img = x.view(batch_size, self.n_channels, self.image_size[0], self.image_size[1])
        
        # --- Layer 1 Processing ---
        all_patch_features_layer1 = []
        patch_coords_for_layer2 = self.coordinate_grid_image.unsqueeze(0).expand(batch_size, -1, -1)

        for i in range(self.num_patches):
            # Extract patch slice indices
            row_idx = i // self.num_patches_per_dim
            col_idx = i % self.num_patches_per_dim
            start_y = row_idx * self.stride
            end_y = start_y + self.patch_size
            start_x = col_idx * self.stride
            end_x = start_x + self.patch_size

            # Extract patch data
            patch = x_img[:, :, start_y:end_y, start_x:end_x]
            patch_pixel_values = patch.flatten(start_dim=2).transpose(1, 2)

            # 1. Process with pixel-based operations
            pixel_coords = self.coordinate_grid_patch.unsqueeze(0).expand(batch_size, -1, -1)
            pixel_points = torch.cat([pixel_coords, patch_pixel_values], dim=2)
            pixel_points_flat = pixel_points.view(-1, 2 + self.n_channels)

            # Apply pixel-based Layer 1
            pixel_features = self.layer1_pixel_cfn(pixel_points_flat)
            pixel_features = pixel_features.view(batch_size, self.pixels_per_patch, -1)

            # Simpler aggregation with just mean and max
            mean_pixel_features = torch.mean(pixel_features, dim=1)
            max_pixel_features = torch.max(pixel_features, dim=1)[0]
            
            # 2. Process with Gabor
            patch_features = []
            if self.use_gabor:
                # Reshape for Gabor with error handling
                try:
                    patch_contiguous = patch.contiguous()
                    patch_flat = patch_contiguous.reshape(batch_size, -1)
                    
                    # Apply Gabor function (skip Fourier for now)
                    gabor_features = self.gabor_node(patch_flat)
                    patch_features.append(gabor_features)
                except Exception as e:
                    print(f"Error in Gabor processing: {e}")
                    # Create a zero tensor as fallback
                    gabor_features = torch.zeros((batch_size, self.gabor_node.output_dim), 
                                                device=x.device)
                    patch_features.append(gabor_features)
            
            # Combine features
            patch_features = [mean_pixel_features, max_pixel_features] + patch_features
            aggregated_patch_feature = torch.cat(patch_features, dim=1)
            
            # Check for NaN and replace with zeros if needed
            if torch.isnan(aggregated_patch_feature).any():
                print(f"NaN detected in patch {i} features, replacing with zeros")
                aggregated_patch_feature = torch.where(
                    torch.isnan(aggregated_patch_feature),
                    torch.zeros_like(aggregated_patch_feature),
                    aggregated_patch_feature
                )
                
            all_patch_features_layer1.append(aggregated_patch_feature)

        # Stack all patch features from Layer 1
        stacked_patch_features = torch.stack(all_patch_features_layer1, dim=1)

        # --- Layer 2 Processing ---
        patch_points_for_layer2 = torch.cat([patch_coords_for_layer2, stacked_patch_features], dim=2)
        patch_points_for_layer2_flat = patch_points_for_layer2.view(-1, 2 + stacked_patch_features.shape[2])

        # Apply Layer 2
        context_aware_patch_features = self.layer2_inter_patch_cfn(patch_points_for_layer2_flat)
        
        # Check for NaN before attention
        if torch.isnan(context_aware_patch_features).any():
            print("NaN detected before attention, replacing with zeros")
            context_aware_patch_features = torch.where(
                torch.isnan(context_aware_patch_features),
                torch.zeros_like(context_aware_patch_features),
                context_aware_patch_features
            )
        
        # Apply single attention head for stability
        context_shape = (batch_size, self.num_patches)
        context_aware_patch_features = self.patch_attention(context_aware_patch_features, context_shape)
        
        # Reshape back to per-image structure
        context_aware_patch_features = context_aware_patch_features.view(batch_size, self.num_patches, -1)
        
        # Simpler aggregation with mean and max only
        mean_image_features = torch.mean(context_aware_patch_features, dim=1)
        max_image_features = torch.max(context_aware_patch_features, dim=1)[0]
        final_image_feature_vector = torch.cat([mean_image_features, max_image_features], dim=1)

        # Check for NaN before classification
        if torch.isnan(final_image_feature_vector).any():
            print("NaN detected before classification, replacing with zeros")
            final_image_feature_vector = torch.where(
                torch.isnan(final_image_feature_vector),
                torch.zeros_like(final_image_feature_vector),
                final_image_feature_vector
            )

        # --- Classifier ---
        logits = self.classifier_head(final_image_feature_vector)
        
        return logits

def run():
    """Runs the CIFAR-10 classification example using the stabilized hierarchical CFN."""
    print("--- Running PyTorch CIFAR-10 Stabilized Hierarchical Coordinate-Aware CFN ---")

    # 1. Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load CIFAR-10 dataset with standard augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Batch size for training - reduced for stability
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Build the stabilized CFN
    network = StabilizedHierarchicalCoordinateAwareCFN().to(device)

    # Helper function for accuracy calculation
    def accuracy_metric_fn(outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == targets).sum().item()

    # 4. Train the model
    print("Stabilized Model Architecture:")
    print("--- Layer 1: Pixel-based CFN ---")
    print(network.layer1_pixel_cfn.describe())
    if network.use_gabor:
        print("--- Layer 1: Gabor Node ---")
        print(network.gabor_node.describe())
    print("--- Layer 2: Inter-Patch CFN ---")
    print(network.layer2_inter_patch_cfn.describe())
    print("--- Attention ---")
    print(network.patch_attention.describe())
    print("--- Classifier Head ---")
    print(network.classifier_head.describe())
    print(f"Patch size: {network.patch_size}, Stride: {network.stride}, Number of patches: {network.num_patches}")

    # Stability-focused optimizer with lower learning rate
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-5)

    # Simple step LR scheduler for stability
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Enable gradient clipping for stability
    trainer = Trainer(
        network,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip_norm=0.5,  # More aggressive gradient clipping
        device=device
    )
    
    start_time = time.time()
    
    # Add error handling during training
    try:
        trainer.train(
            train_loader, 
            val_loader=test_loader, 
            epochs=150,  
            loss_fn=nn.CrossEntropyLoss(), 
            early_stopping_patience=20,
            metric_fn=accuracy_metric_fn
        )
    except Exception as e:
        print(f"Training error: {e}")
        
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # 6. Plot loss and accuracy
    trainer.plot_loss('pytorch_cifar10_stabilized_hierarchical_loss.png')
    trainer.plot_accuracy('pytorch_cifar10_stabilized_hierarchical_accuracy.png')
    print("-------------------------------------------")

if __name__ == '__main__':
    run()