import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cfn_pytorch.function_nodes import (
    FourierFunctionNode, 
    SinusoidalFunctionNode, 
    GaussianFunctionNode, 
    PolynomialFunctionNode,
    LinearFunctionNode
)
from cfn_pytorch.composition_layers import ParallelCompositionLayer

# 1. Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Loading and Preprocessing with Augmentation
print("Loading and preprocessing CIFAR-10 data...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# 3. Pure Fourier-based CFN Model
class FourierDrivenCFN(nn.Module):
    def __init__(self, input_dim, image_size, n_channels, hidden_dim, output_dim):
        super(FourierDrivenCFN, self).__init__()
        
        # First layer: Multiple Fourier nodes capturing different frequency bands
        # This provides a frequency-domain representation of the image
        self.fourier_extraction = ParallelCompositionLayer(
            [
                # Low frequency features (broad patterns)
                FourierFunctionNode(input_dim, image_size, n_channels, n_features=8),
                # Mid frequency features (medium details)
                FourierFunctionNode(input_dim, image_size, n_channels, n_features=16),
                # High frequency features (fine details)
                FourierFunctionNode(input_dim, image_size, n_channels, n_features=32),
                # Very high frequency features (edges, textures)
                FourierFunctionNode(input_dim, image_size, n_channels, n_features=64)
            ],
            combination='concat'
        )
        
        # Total output dimension from all Fourier nodes
        fourier_output_dim = (8 + 16 + 32 + 64) * 2  # *2 because of complex output
        
        # Add complementary non-Fourier processing for the raw image
        self.complementary_extraction = ParallelCompositionLayer(
            [
                GaussianFunctionNode(input_dim),
                PolynomialFunctionNode(input_dim, degree=2)
            ],
            combination='concat'
        )
        
        complementary_output_dim = 1 + 1  # One output each from Gaussian and Polynomial
        
        # Combine Fourier and complementary features
        combined_dim = fourier_output_dim + complementary_output_dim
        self.bn1 = nn.BatchNorm1d(combined_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # Intermediate processing layer
        self.hidden_layer = ParallelCompositionLayer(
            [
                LinearFunctionNode(combined_dim, hidden_dim),
                SinusoidalFunctionNode(combined_dim)
            ],
            combination='concat'
        )
        
        hidden_output_dim = hidden_dim + 1  # Linear + Sinusoidal
        self.bn2 = nn.BatchNorm1d(hidden_output_dim)
        self.dropout2 = nn.Dropout(0.4)
        
        # Final classification layer
        self.classifier = LinearFunctionNode(hidden_output_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Extract frequency domain features via Fourier nodes
        fourier_features = self.fourier_extraction(x_flat)
        
        # Extract complementary features
        complementary_features = self.complementary_extraction(x_flat)
        
        # Combine all features
        combined = torch.cat([fourier_features, complementary_features], dim=1)
        combined = self.bn1(combined)
        combined = self.dropout1(combined)
        
        # Process through hidden layer
        hidden = self.hidden_layer(combined)
        hidden = self.bn2(hidden)
        hidden = self.dropout2(hidden)
        
        # Final classification
        output = self.classifier(hidden)
        
        return output

# 4. Model Initialization, Optimizer and Loss Function
print("Initializing Fourier-Driven CFN model...")
image_size = (32, 32)
n_channels = 3
input_dim = image_size[0] * image_size[1] * n_channels
hidden_dim = 256
output_dim = 10

model = FourierDrivenCFN(
    input_dim=input_dim,
    image_size=image_size,
    n_channels=n_channels,
    hidden_dim=hidden_dim,
    output_dim=output_dim
).to(device)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")

# Optimizer with weight decay for regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,  # Restart every 10 epochs
    T_mult=2  # Double the restart interval after each restart
)

# 5. Training Loop
print("Starting training...")
num_epochs = 100
best_acc = 0.0
save_path = 'best_fourier_driven_cfn_model.pth'

# For mixed precision training if using CUDA
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        # Use mixed precision for faster training if on GPU
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}: Loss: {running_loss/100:.3f}, Acc: {100*correct/total:.2f}%')
            running_loss = 0.0
    
    train_accuracy = 100 * correct / total
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}/{num_epochs} - LR: {current_lr:.6f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    # Save best model
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save(model.state_dict(), save_path)
        print(f'New best model saved with accuracy: {test_accuracy:.2f}%')

print(f'Finished Training. Highest accuracy: {best_acc:.2f}% (saved to {save_path})')