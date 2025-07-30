

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, ParallelCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def generate_spiral_data(n_samples_per_class=300, n_classes=3, noise=0.2):
    X = torch.zeros(n_samples_per_class * n_classes, 2)
    y = torch.zeros(n_samples_per_class * n_classes, dtype=torch.long) # Use long for CrossEntropyLoss
    
    for class_idx in range(n_classes):
        ix = torch.arange(n_samples_per_class * class_idx, n_samples_per_class * (class_idx + 1))
        r = torch.linspace(0.0, 1, n_samples_per_class)  # radius
        t = torch.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples_per_class) + torch.randn(n_samples_per_class) * noise  # theta
        X[ix] = torch.stack([r * torch.sin(t), r * torch.cos(t)], dim=1)
        y[ix] = class_idx
    
    return X, y

def run():
    """Runs a multi-class classification example on the spiral dataset."""
    print("--- Running PyTorch Multi-Class Classification Example: Spiral Dataset ---")
    
    # 1. Generate spiral dataset
    torch.manual_seed(42)
    n_classes = 3
    X, y = generate_spiral_data(n_samples_per_class=300, n_classes=n_classes, noise=0.2)
    
    # Split into train and validation sets
    indices = torch.randperm(X.shape[0])
    split_idx = int(0.8 * X.shape[0])
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 2. Create CFN for classification
    node_factory = FunctionNodeFactory()
    
    network = CompositionFunctionNetwork(
        name="ClassificationCFN",
        layers=[
        # First layer: Feature extraction
        ParallelCompositionLayer(
            name="FeatureExtractionLayer",
            function_nodes=[
                # Gaussian RBFs at different locations
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([0.3, 0.3]), width=0.3),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([-0.3, 0.3]), width=0.3),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([0.3, -0.3]), width=0.3),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([-0.3, -0.3]), width=0.3),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([0.0, 0.0]), width=0.5),
                
                # Sigmoid functions for directional splits
                node_factory.create("Sigmoid", input_dim=2, direction=torch.tensor([1.0, 0.0])),
                node_factory.create("Sigmoid", input_dim=2, direction=torch.tensor([0.0, 1.0])),
                node_factory.create("Sigmoid", input_dim=2, direction=torch.tensor([1.0, 1.0])),
                node_factory.create("Sigmoid", input_dim=2, direction=torch.tensor([1.0, -1.0])),
                
                # Sinusoidal functions for circular patterns
                node_factory.create("Sinusoidal", input_dim=2, frequency=3.0),
                node_factory.create("Sinusoidal", input_dim=2, frequency=5.0),
                
                # Distance from origin (useful for spiral)
                node_factory.create("Polynomial", input_dim=2, degree=2,
                                    coefficients=torch.tensor([0.0, 0.0, 1.0])),
            ],
            combination='concat'
        ),
        
        # Second layer: Hidden layer
        SequentialCompositionLayer(
            name="HiddenLayer",
            function_nodes=[
            node_factory.create("Linear", input_dim=12, output_dim=8),
            node_factory.create("Sigmoid", input_dim=8)  # Element-wise sigmoid
        ]),
        
        # Third layer: Output layer (multiclass)
        SequentialCompositionLayer(
            name="OutputLayer",
            function_nodes=[
            node_factory.create("Linear", input_dim=8, output_dim=n_classes),
            # Softmax is implicitly handled by nn.CrossEntropyLoss
        ])
    ])
    
    print("Initial Network Structure:")
    print(network.describe())
    
    # 3. Train the network
    trainer = Trainer(network, learning_rate=0.02)
    
    trainer.train(
        train_loader, val_loader=val_loader, epochs=300, 
        loss_fn=nn.CrossEntropyLoss(), # For multi-class classification
        early_stopping_patience=30
    )
    
    # Plot loss
    trainer.plot_loss('pytorch_spiral_loss.png')
    
    # 4. Visualize and Evaluate
    visualize_classification_results(network, X, y, n_classes)
    
    # 5. Interpret the model
    print("\n--- Final Model Interpretation ---")
    interpret_model(network)
    print("--------------------------------------------------")
    return network

def visualize_classification_results(network, X, y, n_classes):
    """Visualize classification results with decision boundaries."""
    # Create a grid of points
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h), indexing='ij')
    
    # Evaluate network on grid
    grid_points = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    network.eval()
    with torch.no_grad():
        Z = network(grid_points)
    
    # Get predicted class (apply softmax and argmax)
    Z_class = torch.argmax(Z, dim=1).reshape(xx.shape)
    
    # Plot decision boundaries
    plt.figure(figsize=(10, 8))
    plt.contourf(xx.numpy(), yy.numpy(), Z_class.numpy(), cmap=plt.cm.Spectral, alpha=0.8)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap=plt.cm.Spectral, 
                         edgecolors='k', s=40)
    
    plt.xlim(xx.min().numpy(), xx.max().numpy())
    plt.ylim(yy.min().numpy(), yy.max().numpy())
    plt.title("Decision Boundaries")
    
    # Add colorbar
    plt.colorbar(scatter, ticks=range(n_classes))
    plt.savefig('pytorch_spiral_decision_boundary.png')
    print("\nPlot saved to pytorch_spiral_decision_boundary.png")
    plt.close()
    
    # Evaluate accuracy
    network.eval()
    with torch.no_grad():
        y_pred_raw = network(X)
    y_pred = torch.argmax(y_pred_raw, dim=1)
    accuracy = (y_pred == y).float().mean().item()
    
    print(f"\nClassification Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y)):
        conf_matrix[y[i].item(), y_pred[i].item()] += 1
    
    print("Confusion Matrix:")
    print(conf_matrix)
    return network

if __name__ == '__main__':
    model = run()
    print("\n--- Final Model Interpretation from Runner---")
    interpret_model(model)

