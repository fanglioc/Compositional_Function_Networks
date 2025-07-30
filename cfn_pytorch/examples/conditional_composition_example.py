

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, ConditionalCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def run():
    """
    Advanced example: Combining different composition types
    for approximating a complex function with local behavior.
    This PyTorch version is designed to replicate the logic of the NumPy version.
    """
    print("--- Running PyTorch Advanced Example: Local Expert Functions ---")
    
    # Generate synthetic data
    def target_function(x1, x2):
        # Different behavior in different regions
        r = np.sqrt(x1**2 + x2**2)
        theta = np.arctan2(x2, x1)
        
        # Region 1: Sinusoidal pattern
        if r < 0.5:
            return 0.5 * np.sin(theta * 5) * r
        # Region 2: Polynomial
        elif r < 1.0:
            return 0.2 * r**2 - 0.1 * x1 * x2
        # Region 3: Exponential decay
        elif r < 1.5:
            return 0.3 * np.exp(-2 * (r - 1.0))
        # Region 4: Linear
        else:
            return 0.1 * x1 - 0.2 * x2 + 0.05
    
    # Generate random data points
    np.random.seed(42)
    n_samples = 2000
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Create factories
    node_factory = FunctionNodeFactory()
    
    # Define the network architecture
    input_dim = 2
    feature_dim = 5  # [x1, x2, r^2, sin(theta), cos(theta)]

    network = CompositionFunctionNetwork(
        name="AdvancedCFN",
        layers=[
        # Layer 1: Feature Extraction
        ParallelCompositionLayer(
            name="FeatureExtractionLayer",
            function_nodes=[
                # Raw coordinates (x1, x2)
                node_factory.create("Linear", input_dim=input_dim, output_dim=2,
                                    weights=torch.eye(2), bias=torch.zeros(2)),
                
                # Radius squared (r^2)
                node_factory.create("Polynomial", input_dim=input_dim, degree=2,
                                    coefficients=torch.tensor([0.0, 0.0, 1.0]),
                                    direction=torch.tensor([1.0, 1.0])), # direction doesn't matter for r^2
                
                # Sine and Cosine of angle (approximated)
                node_factory.create("Sinusoidal", input_dim=input_dim, 
                                    direction=torch.tensor([1.0, 0.0])),
                node_factory.create("Sinusoidal", input_dim=input_dim, 
                                    direction=torch.tensor([0.0, 1.0])),
            ],
            combination="concat"
        ),
        
        # Layer 2: Region-specific Experts
        ConditionalCompositionLayer(
            name="RegionalExpertsLayer",
            condition_nodes=[
                # Region 1: Inner circle (r < 0.5) -> r^2 < 0.25
                node_factory.create("Step", input_dim=feature_dim, bias=0.25, 
                                    direction=torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0]), smoothing=0.1),
                
                # Region 2: Middle ring (0.5 <= r < 1.0) -> 0.25 <= r^2 < 1.0
                node_factory.create("Step", input_dim=feature_dim, bias=0.25, 
                                    direction=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), smoothing=0.1),
                
                # Region 3: Outer ring (1.0 <= r < 1.5) -> 1.0 <= r^2 < 2.25
                node_factory.create("Step", input_dim=feature_dim, bias=1.0, 
                                    direction=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), smoothing=0.1),
                
                # Region 4: Far field (r >= 1.5) -> r^2 >= 2.25
                node_factory.create("Step", input_dim=feature_dim, bias=2.25, 
                                    direction=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), smoothing=0.1),
            ],
            function_nodes=[
                # Expert for Region 1: Sinusoidal pattern
                ParallelCompositionLayer(
                    name="Region1Expert",
                    function_nodes=[
                        node_factory.create("Sinusoidal", input_dim=feature_dim, 
                                            direction=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]), frequency=5.0),
                        node_factory.create("Linear", input_dim=feature_dim, output_dim=1,
                                            weights=torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]]).T, bias=torch.zeros(1))
                    ],
                    combination="product"
                ),
                
                # Expert for Region 2: Polynomial
                node_factory.create("Polynomial", input_dim=feature_dim, degree=2,
                                    direction=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])),
                
                # Expert for Region 3: Exponential decay
                node_factory.create("Exponential", input_dim=feature_dim, 
                                    direction=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), rate=-2.0, shift=-1.0, scale=0.3),
                
                # Expert for Region 4: Linear
                node_factory.create("Linear", input_dim=feature_dim, output_dim=1,
                                    weights=torch.tensor([[0.1, -0.2, 0.0, 0.0, 0.0]]).T, bias=torch.tensor([0.05])),
            ]
        )
    ])
    
    # Print network description
    print("Initial Network Structure:")
    print(network.describe())
    
    # Initialize trainer
    trainer = Trainer(network, learning_rate=0.001)
    
    # Train the network
    trainer.train(
        train_loader, val_loader=val_loader, epochs=300, 
        loss_fn=nn.MSELoss(), early_stopping_patience=30
    )
    
    # Plot loss
    trainer.plot_loss('pytorch_advanced_loss.png')
    
    # Create visualization function
    def visualize_advanced_results(network, target_function):
        """Visualize results for the advanced example."""
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
        fig = plt.figure(figsize=(18, 12))
        font_size = 16 # Larger font size for titles
        label_size = 14 # Font size for labels

        # Plot true function
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.8)
        ax1.set_title('True Function', fontsize=font_size)
        ax1.set_xlabel('x1', fontsize=label_size)
        ax1.set_ylabel('x2', fontsize=label_size)

        # Plot predicted function
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.plot_surface(X1, X2, Z_pred, cmap='viridis', alpha=0.8)
        ax2.set_title('CFN Prediction', fontsize=font_size)
        ax2.set_xlabel('x1', fontsize=label_size)
        ax2.set_ylabel('x2', fontsize=label_size)

        # Plot error
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.plot_surface(X1, X2, Z_error, cmap='hot', alpha=0.8)
        ax3.set_title('Absolute Error', fontsize=font_size)
        ax3.set_xlabel('x1', fontsize=label_size)
        ax3.set_ylabel('x2', fontsize=label_size)

        # Plot true function contour
        ax4 = fig.add_subplot(234)
        contour1 = ax4.contourf(X1, X2, Z_true, 20, cmap='viridis')
        plt.colorbar(contour1, ax=ax4)
        ax4.set_title('True Function (Contour)', fontsize=font_size)
        ax4.set_xlabel('x1', fontsize=label_size)
        ax4.set_ylabel('x2', fontsize=label_size)

        # Plot predicted function contour
        ax5 = fig.add_subplot(235)
        contour2 = ax5.contourf(X1, X2, Z_pred, 20, cmap='viridis')
        plt.colorbar(contour2, ax=ax5)
        ax5.set_title('CFN Prediction (Contour)', fontsize=font_size)
        ax5.set_xlabel('x1', fontsize=label_size)
        ax5.set_ylabel('x2', fontsize=label_size)

        # Plot error contour
        ax6 = fig.add_subplot(236)
        contour3 = ax6.contourf(X1, X2, Z_error, 20, cmap='hot')
        plt.colorbar(contour3, ax=ax6)
        ax6.set_title('Absolute Error (Contour)', fontsize=font_size)
        ax6.set_xlabel('x1', fontsize=label_size)
        ax6.set_ylabel('x2', fontsize=label_size)
        
        plt.tight_layout()
        plt.savefig('pytorch_advanced_results.png')
        print("Plot saved to pytorch_advanced_results.png")
        plt.close(fig)
        
        # Print error statistics
        mse = np.mean(Z_error**2)
        mae = np.mean(np.abs(Z_error))
        max_error = np.max(Z_error)
        
        print("\nAdvanced Example Metrics:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
        print(f"Maximum Error: {max_error:.6f}")
    
    # Visualize results
    visualize_advanced_results(network, target_function)
    
    print("--------------------------------------------------")
    return network

if __name__ == '__main__':
    model = run()
    print("\n--- Interpreting the Trained Model from Runner ---")
    interpret_model(model)

