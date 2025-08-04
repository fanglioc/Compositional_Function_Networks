
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import copy
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cfn_pytorch.function_nodes import FunctionNode, FunctionNodeFactory
from cfn_pytorch.composition_layers import (
    CompositionFunctionNetwork, ParallelCompositionLayer, SequentialCompositionLayer, 
    PatchwiseCompositionLayer, ReassembleToGridLayer, FlattenAndConcatenateLayer, ResidualCompositionLayer, SELayer
)
from cfn_pytorch.trainer import Trainer

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_math_cnn(input_image_shape, num_classes=10):
    """
    Builds an even more powerful 'Mathematician's CNN' with multiple downsampling stages.
    """
    # Layer 1: Wider Diverse Mathematical Filter Bank (Convolutional)
    l1_patch_size = (5, 5)
    l1_stride = 1
    l1_padding = 2
    l1_sub_nodes = []
    for _ in range(8):
        l1_sub_nodes.extend([
            FunctionNodeFactory.create('Gabor', input_dim=3*l1_patch_size[0]*l1_patch_size[1], image_size=l1_patch_size, n_channels=3),
            FunctionNodeFactory.create('Polynomial', input_dim=3*l1_patch_size[0]*l1_patch_size[1], degree=2),
            FunctionNodeFactory.create('Sinusoidal', input_dim=3*l1_patch_size[0]*l1_patch_size[1], 
                                       frequency=np.random.uniform(0.5, 5.0), 
                                       amplitude=np.random.uniform(0.5, 2.0), 
                                       phase=np.random.uniform(0.0, 2*np.pi)),
        ])
    l1_parallel_node = ParallelCompositionLayer(l1_sub_nodes, combination='concat')
    l1_conv_layer = PatchwiseCompositionLayer(
        sub_node=l1_parallel_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=input_image_shape,
        patch_size=l1_patch_size,
        stride=l1_stride,
        padding=l1_padding,
    )

    # Layer 2: Residual Block 1
    l2_input_shape = l1_conv_layer.output_shape
    l2_sub_node = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Linear', input_dim=l2_input_shape[0]*3*3, output_dim=l2_input_shape[0]),
        FunctionNodeFactory.create('ReLU', input_dim=l2_input_shape[0]),
    ])
    l2_conv_layer = PatchwiseCompositionLayer(
        sub_node=l2_sub_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=l2_input_shape,
        patch_size=(3, 3),
        padding=1,
    )
    res_block_1 = ResidualCompositionLayer(main_path_layers=[l2_conv_layer], input_shape=l2_input_shape, output_shape=l2_conv_layer.output_shape)
    se_layer_1 = SELayer(l2_input_shape[0])

    # Layer 3: Downsampling Layer 1
    l3_input_shape = res_block_1.output_shape
    l3_output_channels = l3_input_shape[0] * 2
    downsampling_layer_1 = PatchwiseCompositionLayer(
        sub_node=FunctionNodeFactory.create('Linear', input_dim=l3_input_shape[0]*3*3, output_dim=l3_output_channels),
        combination_layer=ReassembleToGridLayer(),
        input_shape=l3_input_shape,
        patch_size=(3, 3),
        stride=2, # Downsample
        padding=1,
    )

    # Layer 4: Residual Block 2
    l4_input_shape = downsampling_layer_1.output_shape
    l4_sub_node = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Linear', input_dim=l4_input_shape[0]*3*3, output_dim=l4_input_shape[0]),
        FunctionNodeFactory.create('ReLU', input_dim=l4_input_shape[0]),
    ])
    l4_conv_layer = PatchwiseCompositionLayer(
        sub_node=l4_sub_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=l4_input_shape,
        patch_size=(3, 3),
        padding=1,
    )
    res_block_2 = ResidualCompositionLayer(main_path_layers=[l4_conv_layer], input_shape=l4_input_shape, output_shape=l4_conv_layer.output_shape)
    se_layer_2 = SELayer(l4_input_shape[0])

    # Layer 5: Downsampling Layer 2
    l5_input_shape = res_block_2.output_shape
    l5_output_channels = l5_input_shape[0] * 2
    downsampling_layer_2 = PatchwiseCompositionLayer(
        sub_node=FunctionNodeFactory.create('Linear', input_dim=l5_input_shape[0]*3*3, output_dim=l5_output_channels),
        combination_layer=ReassembleToGridLayer(),
        input_shape=l5_input_shape,
        patch_size=(3, 3),
        stride=2, # Downsample
        padding=1,
    )

    # Layer 6: Residual Block 3
    l6_input_shape = downsampling_layer_2.output_shape
    l6_sub_node = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Linear', input_dim=l6_input_shape[0]*3*3, output_dim=l6_input_shape[0]),
        FunctionNodeFactory.create('ReLU', input_dim=l6_input_shape[0]),
    ])
    l6_conv_layer = PatchwiseCompositionLayer(
        sub_node=l6_sub_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=l6_input_shape,
        patch_size=(3, 3),
        padding=1,
    )
    res_block_3 = ResidualCompositionLayer(main_path_layers=[l6_conv_layer], input_shape=l6_input_shape, output_shape=l6_conv_layer.output_shape)
    se_layer_3 = SELayer(l6_input_shape[0])

    # Layer 7: Residual Block 4
    l7_input_shape = res_block_3.output_shape
    l7_sub_node = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Linear', input_dim=l7_input_shape[0]*3*3, output_dim=l7_input_shape[0]),
        FunctionNodeFactory.create('ReLU', input_dim=l7_input_shape[0]),
    ])
    l7_conv_layer = PatchwiseCompositionLayer(
        sub_node=l7_sub_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=l7_input_shape,
        patch_size=(3, 3),
        padding=1,
    )
    res_block_4 = ResidualCompositionLayer(main_path_layers=[l7_conv_layer], input_shape=l7_input_shape, output_shape=l7_conv_layer.output_shape)
    se_layer_4 = SELayer(l7_input_shape[0])

    # Layer 8: Residual Block 5
    l8_input_shape = res_block_4.output_shape
    l8_sub_node = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Linear', input_dim=l8_input_shape[0]*3*3, output_dim=l8_input_shape[0]),
        FunctionNodeFactory.create('ReLU', input_dim=l8_input_shape[0]),
    ])
    l8_conv_layer = PatchwiseCompositionLayer(
        sub_node=l8_sub_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=l8_input_shape,
        patch_size=(3, 3),
        padding=1,
    )
    res_block_5 = ResidualCompositionLayer(main_path_layers=[l8_conv_layer], input_shape=l8_input_shape, output_shape=l8_conv_layer.output_shape)
    se_layer_5 = SELayer(l8_input_shape[0])

    # Layer 9: Aggregation (Global Pooling)
    l9_input_shape = res_block_5.output_shape
    l9_patch_size = (l9_input_shape[1], l9_input_shape[2]) # Global pooling
    l9_sub_node = FunctionNodeFactory.create('Linear', input_dim=l9_input_shape[0]*l9_patch_size[0]*l9_patch_size[1], output_dim=256)
    aggregation_layer = PatchwiseCompositionLayer(
        sub_node=l9_sub_node,
        combination_layer=FlattenAndConcatenateLayer(),
        input_shape=l9_input_shape,
        patch_size=l9_patch_size,
    )

    # Layer 10: Final Classifier
    classifier = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Dropout', input_dim=256, p=0.6),
        FunctionNodeFactory.create('Linear', input_dim=256, output_dim=num_classes),
    ])

    network = CompositionFunctionNetwork(
        layers=[
            l1_conv_layer,
            res_block_1,
            se_layer_1,
            downsampling_layer_1,
            res_block_2,
            se_layer_2,
            downsampling_layer_2,
            res_block_3,
            se_layer_3,
            res_block_4,
            se_layer_4,
            res_block_5,
            se_layer_5,
            aggregation_layer,
            classifier,
        ]
    )
    return network


def main():
    set_seed(42) # For reproducibility
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading and transformation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Re-added ColorJitter
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False), # Cutout
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_trainset = copy.deepcopy(trainset)
    val_trainset.transform = val_transform

    # Split training data into training and validation sets
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    # The validation dataset should not have augmentation
    val_dataset.dataset = val_trainset 

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    # Network setup
    input_shape = (3, 32, 32)
    patch_size = (7, 7)
    stride = (2, 2)
    
    # Build the CFN
    model = build_math_cnn(input_shape, num_classes=10).to(device)

    print("--- Network Architecture ---")
    print(model.describe())
    print("--------------------------")

    # Add torch.compile for performance optimization
    if torch.__version__ >= "2.0":
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, backend='aot_eager')

    # Trainer setup with modern optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, device=device, log_dir='runs/cifar10_math_cnn', use_amp=True, grad_clip_norm=1.0)
    
    # Define the accuracy metric for validation
    def accuracy_metric(outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == targets).sum().item()

    # Train the model with Label Smoothing
    print("Starting training with early stopping, LR decay, and Label Smoothing...")
    trainer.train(trainloader, val_loader=valloader, epochs=150, loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1), early_stopping_patience=20, metric_fn=accuracy_metric, warmup_epochs=5)
    print("Training finished.")

    # Evaluate the model
    print("Evaluating on test set...")
    accuracy = trainer.evaluate(testloader)
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
