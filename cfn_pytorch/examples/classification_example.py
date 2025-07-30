

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer

def run():
    """Runs a simple binary classification example."""
    print("--- Running PyTorch Classification Example ---")
    # 1. Generate synthetic data (two moons)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Build the CFN
    node_factory = FunctionNodeFactory()
    network = CompositionFunctionNetwork(layers=[
        SequentialCompositionLayer([
            node_factory.create('Linear', input_dim=2, output_dim=8),
            node_factory.create('Sigmoid', input_dim=8), # Element-wise sigmoid
            node_factory.create('Linear', input_dim=8, output_dim=1),
            node_factory.create('Sigmoid', input_dim=1) # Final output activation
        ])
    ])

    # 3. Train the model
    print("Initial Model:")
    print(network.describe())
    trainer = Trainer(network, learning_rate=0.05)
    # Use Binary Cross Entropy Loss for classification
    trainer.train(loader, epochs=100, loss_fn=nn.BCELoss())

    # 4. Print results and plot decision boundary
    print("\nTrained Model:")
    print(network.describe())
    trainer.plot_loss('pytorch_classification_loss.png')

    # Create a meshgrid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.02), torch.arange(y_min, y_max, 0.02))
    grid = torch.cat([xx.ravel().unsqueeze(1), yy.ravel().unsqueeze(1)], dim=1)

    with torch.no_grad():
        Z = network(grid).reshape(xx.shape)
        Z = (Z > 0.5).float() # Get class predictions

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.RdBu, edgecolors='k')
    plt.title('PyTorch Classification Example - Decision Boundary')
    plt.savefig('pytorch_classification_visualization.png')
    print("\nPlot saved to pytorch_classification_visualization.png")
    print("-------------------------------------------")

