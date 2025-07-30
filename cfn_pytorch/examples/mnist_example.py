import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer

def run():
    """Runs the MNIST classification example."""
    print("--- Running PyTorch MNIST Classification Example ---")

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
    network = CompositionFunctionNetwork(layers=[
        SequentialCompositionLayer([
            node_factory.create('Linear', input_dim=784, output_dim=128),
            node_factory.create('ReLU', input_dim=128),
            node_factory.create('Linear', input_dim=128, output_dim=64),
            node_factory.create('ReLU', input_dim=64),
            node_factory.create('Linear', input_dim=64, output_dim=10)
        ])
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
    plt.savefig('pytorch_mnist_example.png')
    print("\nPlot saved to pytorch_mnist_example.png")
    print("-------------------------------------------")

if __name__ == '__main__':
    run()
