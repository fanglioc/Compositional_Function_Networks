
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import copy

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cfn_pytorch.function_nodes import FunctionNode, FunctionNodeFactory
from cfn_pytorch.composition_layers import (
    CompositionFunctionNetwork, ParallelCompositionLayer, SequentialCompositionLayer, 
    PatchwiseCompositionLayer, ReassembleToGridLayer, FlattenAndConcatenateLayer, ResidualCompositionLayer
)
from cfn_pytorch.trainer import Trainer


def build_math_cnn(input_image_shape, num_classes=10):
    """
    Builds the true 'Mathematician's CNN' using the new, generic framework.
    """
    # Layer 1: Diverse Mathematical Filter Bank (Convolutional)
    l1_patch_size = (5, 5)
    l1_stride = 1
    l1_padding = 2
    l1_sub_nodes = [
        FunctionNodeFactory.create('Gabor', input_dim=3*l1_patch_size[0]*l1_patch_size[1], image_size=l1_patch_size, n_channels=3),
        FunctionNodeFactory.create('Polynomial', input_dim=3*l1_patch_size[0]*l1_patch_size[1], degree=2),
        FunctionNodeFactory.create('Sinusoidal', input_dim=3*l1_patch_size[0]*l1_patch_size[1]),
    ]
    l1_parallel_node = ParallelCompositionLayer(l1_sub_nodes, combination='concat')
    l1_conv_layer = PatchwiseCompositionLayer(
        sub_node=l1_parallel_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=input_image_shape,
        patch_size=l1_patch_size,
        stride=l1_stride,
        padding=l1_padding,
    )

    # Layer 2: Residual Block 1 (Convolutional)
    l2_input_shape = l1_conv_layer.output_shape
    l2_patch_size = (3, 3)
    l2_stride = 1
    l2_padding = 1
    l2_sub_node = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Linear', input_dim=l2_input_shape[0]*l2_patch_size[0]*l2_patch_size[1], output_dim=l2_input_shape[0]),
        FunctionNodeFactory.create('ReLU', input_dim=l2_input_shape[0]),
    ])
    l2_conv_layer = PatchwiseCompositionLayer(
        sub_node=l2_sub_node,
        combination_layer=ReassembleToGridLayer(),
        input_shape=l2_input_shape,
        patch_size=l2_patch_size,
        stride=l2_stride,
        padding=l2_padding,
    )
    res_block_1 = ResidualCompositionLayer(main_path_layers=[l2_conv_layer], input_shape=l2_input_shape, output_shape=l2_conv_layer.output_shape)

    # Layer 3: Aggregation (Flattening)
    l3_input_shape = res_block_1.output_shape
    l3_patch_size = (l3_input_shape[1], l3_input_shape[2]) # Global pooling
    l3_sub_node = FunctionNodeFactory.create('Linear', input_dim=l3_input_shape[0]*l3_patch_size[0]*l3_patch_size[1], output_dim=128)
    aggregation_layer = PatchwiseCompositionLayer(
        sub_node=l3_sub_node,
        combination_layer=FlattenAndConcatenateLayer(),
        input_shape=l3_input_shape,
        patch_size=l3_patch_size,
    )

    # Layer 4: Final Classifier
    classifier = FunctionNodeFactory.create('SequentialWrapper', function_nodes=[
        FunctionNodeFactory.create('Dropout', input_dim=128, p=0.5),
        FunctionNodeFactory.create('Linear', input_dim=128, output_dim=num_classes),
    ])

    network = CompositionFunctionNetwork(
        layers=[
            l1_conv_layer,
            res_block_1,
            aggregation_layer,
            classifier,
        ]
    )
    return network


def main():
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading and transformation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Network setup
    input_shape = (3, 32, 32)
    patch_size = (7, 7)
    stride = (2, 2)
    
    # Flatten the input for the network
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    # Build the CFN
    cfn_model = build_math_cnn(input_shape, num_classes=10)

    # The CFN works on flattened data, so we wrap it
    model = nn.Sequential(
        Flatten(),
        cfn_model
    ).to(device) # Move the entire model to the device

    print("--- Network Architecture ---")
    print(cfn_model.describe())
    print("--------------------------")

    # Trainer setup with learning rate decay and early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, device=device, log_dir='runs/cifar10_math_cnn')
    
    # Define the accuracy metric for validation
    def accuracy_metric(outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == targets).sum().item()

    # Train the model
    print("Starting training with early stopping and LR decay...")
    trainer.train(trainloader, val_loader=valloader, epochs=100, loss_fn=nn.CrossEntropyLoss(), early_stopping_patience=5, metric_fn=accuracy_metric)
    print("Training finished.")

    # Evaluate the model
    print("Evaluating on test set...")
    accuracy = trainer.evaluate(testloader)
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
