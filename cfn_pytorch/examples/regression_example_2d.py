

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def run():
    """Regression example: Approximating a complex 2D function."""
    print("--- Running PyTorch Regression Example: 2D Function Approximation ---")
    
    # Generate synthetic data
    def target_function(x1, x2):
        return np.sin(x1 * 3) * np.cos(x2 * 2) + 0.2 * x1**2 - 0.3 * x2
    
    # Generate random data points
    np.random.seed(42)
    n_samples = 1000
    X_np = np.random.uniform(-2, 2, (n_samples, 2))
    y_np = np.array([target_function(x[0], x[1]) for x in X_np]).reshape(-1, 1)
    
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create function node factory
    node_factory = FunctionNodeFactory()
    
    # Create network for regression
    network = CompositionFunctionNetwork(
        name="RegressionCFN",
        layers=[
        # First layer: Parallel composition of different basis functions
        ParallelCompositionLayer(
            name="BasisFunctionsLayer",
            function_nodes=[
                # Gaussian RBFs with different centers and widths
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([0.0, 0.0]), width=0.8),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([1.0, 1.0]), width=0.8),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([-1.0, -1.0]), width=0.8),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([1.0, -1.0]), width=0.8),
                node_factory.create("Gaussian", input_dim=2, center=torch.tensor([-1.0, 1.0]), width=0.8),
                
                # Sinusoidal functions with different frequencies and directions
                node_factory.create("Sinusoidal", input_dim=2, frequency=1.0),
                node_factory.create("Sinusoidal", input_dim=2, frequency=2.0),
                node_factory.create("Sinusoidal", input_dim=2, frequency=3.0),
                
                # Polynomial functions
                node_factory.create("Polynomial", input_dim=2, degree=2),
                node_factory.create("Polynomial", input_dim=2, degree=3),
                
            ],
            combination='concat'
        ),
        
        # Second layer: Linear combination of basis functions
        SequentialCompositionLayer(
            name="CombinationLayer",
            function_nodes=[
            node_factory.create("Linear", input_dim=10, output_dim=1)
        ])
    ])
    
    # Print network description
    print("Initial Network Structure:")
    print(network.describe())
    
    # Initialize trainer
    trainer = Trainer(network, learning_rate=0.01)
    
    # Train the network
    trainer.train(
        train_loader, val_loader=val_loader, epochs=200, 
        loss_fn=nn.MSELoss(), early_stopping_patience=20
    )
    
    # Plot loss
    trainer.plot_loss('pytorch_regression_2d_loss.png')
    
    # Visualize the learned function
    visualize_regression_results(network, target_function)
    
    # Interpret the final model
    print("\n--- Final Model Interpretation ---")
    interpret_model(network)
    print("--------------------------------------------------")

def visualize_regression_results(network, target_function):
    """Visualize regression results."""
    # Create a grid of points
    x1_grid = np.linspace(-2, 2, 50)
    x2_grid = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Evaluate target function on grid
    Z_true = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z_true[i, j] = target_function(X1[i, j], X2[i, j])
    
    # Evaluate network on grid
    grid_points = torch.tensor(np.column_stack((X1.flatten(), X2.flatten())), dtype=torch.float32)
    network.eval()
    with torch.no_grad():
        Z_pred = network(grid_points).numpy().reshape(X1.shape)
    
    # Calculate error
    Z_error = np.abs(Z_pred - Z_true)
    
    # Plot results
    fig = plt.figure(figsize=(18, 6))
    
    # Plot true function
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.8)
    ax1.set_title('True Function')
    
    # Plot predicted function
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X1, X2, Z_pred, cmap='viridis', alpha=0.8)
    ax2.set_title('CFN Prediction')
    
    # Plot error
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X1, X2, Z_error, cmap='hot', alpha=0.8)
    ax3.set_title('Absolute Error')
    
    # Calculate error statistics
    mse = np.mean((Z_pred - Z_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(Z_pred - Z_true))
    max_error = np.max(np.abs(Z_pred - Z_true))

    # Add metrics to the plot title
    fig.suptitle(f"Regression Metrics: MSE={mse:.4f}, MAE={mae:.4f}, Max Error={max_error:.4f}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
    plt.savefig('pytorch_regression_2d_results.png')
    print("\nPlot saved to pytorch_regression_2d_results.png")
    plt.close(fig)
    
    # --- Performance Analysis ---
    y_range = np.max(Z_true) - np.min(Z_true)
    y_std = np.std(Z_true)
    nrmse = rmse / y_std if y_std > 0 else 0

    print("\n----------------------------------------")
    print("         Regression Performance         ")
    print("----------------------------------------")
    print(f"Mean Squared Error (MSE):      {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.6f}")
    print(f"Mean Absolute Error (MAE):       {mae:.6f}")
    print(f"Maximum Error:                   {max_error:.6f}")
    print("\n--- Context for Interpretation ---")
    print(f"Target Function Value Range:     {y_range:.4f}")
    print(f"Target Function Std. Dev.:      {y_std:.4f}")
    print(f"Normalized RMSE (RMSE / StdDev): {nrmse:.4f}")
    print("----------------------------------------")
    return network

if __name__ == '__main__':
    model = run()
    


