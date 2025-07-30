

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer

def run():
    """Runs a simple regression example."""
    print("--- Running PyTorch Regression Example ---")
    # 1. Generate synthetic data for y = sin(2*pi*x) + 0.5*x^2
    X = torch.linspace(0, 1, 100).unsqueeze(1)
    y = torch.sin(2 * torch.pi * X) + 0.5 * X**2 + 0.1 * torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Build the CFN
    node_factory = FunctionNodeFactory()
    network = CompositionFunctionNetwork(layers=[
        ParallelCompositionLayer(
            function_nodes=[
                node_factory.create('Sinusoidal', input_dim=1),
                node_factory.create('Polynomial', input_dim=1, degree=2)
            ],
            combination='sum'
        )
    ])

    # 3. Train the model
    print("Initial Model:")
    print(network.describe())
    trainer = Trainer(network, learning_rate=0.01)
    trainer.train(loader, epochs=150, loss_fn=nn.MSELoss())

    # 4. Print results and plot
    print("\nTrained Model:")
    print(network.describe())
    trainer.plot_loss('pytorch_regression_loss.png')
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(X.numpy(), y.numpy(), 'ro', label='Actual')
    with torch.no_grad():
        predictions = network(X).numpy()
    plt.plot(X.numpy(), predictions, 'b-', label='Predicted')
    plt.title('PyTorch Regression Example')
    plt.legend()
    plt.grid(True)
    plt.savefig('pytorch_regression_visualization.png')
    print("\nPlot saved to pytorch_regression_visualization.png")
    print("-----------------------------------------")

