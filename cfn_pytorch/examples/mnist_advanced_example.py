import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, ParallelCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer

def run():
    """Runs the MNIST classification example with an advanced CFN architecture."""
    print("--- Running PyTorch MNIST Advanced CFN Example ---")

    # 1. Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load MNIST dataset
    train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Build the CFN
    node_factory = FunctionNodeFactory()

    # Define the input dimension for MNIST (28*28 = 784)
    mnist_input_dim = 28 * 28
    image_size = (28, 28)
    n_channels = 1 # Grayscale image

    # First layer: Parallel composition of various basis functions for feature extraction
    feature_layer = ParallelCompositionLayer(
        function_nodes=[
            # Fourier features for global frequency patterns
            node_factory.create('Fourier', input_dim=mnist_input_dim, image_size=image_size, n_channels=n_channels, n_features=10),
            
            # Gabor features for localized edge/texture detection
            node_factory.create('Gabor', input_dim=mnist_input_dim, image_size=image_size, n_channels=n_channels),
            
            # Polynomial features for non-linear interactions
            node_factory.create('Polynomial', input_dim=mnist_input_dim, degree=2),
            
            # A standard linear feature extractor
            node_factory.create('Linear', input_dim=mnist_input_dim, output_dim=32)
        ],
        combination='concat'
    )
    
    # Calculate output_dim of feature_layer
    # Fourier (10 features * 2) + Gabor (1) + Polynomial (1) + Linear (32) = 20 + 1 + 1 + 32 = 54
    output_dim_feature_layer = 54

    # Second layer: Sequential composition for classification
    classification_layer = SequentialCompositionLayer(
        function_nodes=[
            node_factory.create('Linear', input_dim=output_dim_feature_layer, output_dim=64),
            node_factory.create('ReLU', input_dim=64),
            node_factory.create('Linear', input_dim=64, output_dim=10) # 10 classes for MNIST
        ]
    )

    network = CompositionFunctionNetwork(layers=[
        feature_layer,
        classification_layer
    ])

    # 4. Train the model
    print("Initial Model:")
    print(network.describe())
    trainer = Trainer(network, learning_rate=0.01, device=device)
    # Use Cross Entropy Loss for multi-class classification
    trainer.train(train_loader, epochs=10, loss_fn=nn.CrossEntropyLoss())

    # 5. Evaluate the model
    print("\n--- Evaluating on test set ---")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    network.eval() # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 28 * 28)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    # 6. Visualize some predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in np.arange(0, 10):
        axes[i].imshow(test_dataset.data[i], cmap='gray')
        axes[i].set_title(f"Pred: {all_preds[i]}\nTrue: {all_labels[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('pytorch_mnist_advanced_example.png')
    print("\nPlot saved to pytorch_mnist_advanced_example.png")
    print("-------------------------------------------")

if __name__ == '__main__':
    run()
