

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def run():
    """
    Symbolic Regression Example: Learning Koza-1 function (f(x) = x^3 + x^2 + x).
    This example demonstrates learning a cubic polynomial from data.
    """
    print("--- Running PyTorch Symbolic Regression Example: Koza-1 ---")

    # 1. Generate synthetic Koza-1 data
    # True coefficients for x^3 + x^2 + x are [0, 1, 1, 1] for [constant, x, x^2, x^3]
    true_coeffs = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    n_samples = 1000
    # Generate x values in a reasonable range, e.g., -1 to 1
    x_data = torch.linspace(-1, 1, n_samples).unsqueeze(1) # Add a dimension for features
    
    # Calculate true y values
    y_true = true_coeffs[0] + true_coeffs[1]*x_data + true_coeffs[2]*(x_data**2) + true_coeffs[3]*(x_data**3)
    y_true += 0.05 * torch.randn(n_samples, 1) # Add some noise

    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_true[:split_idx], y_true[split_idx:]

    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Create CFN for Koza-1
    node_factory = FunctionNodeFactory()

    network = CompositionFunctionNetwork(
        name="Koza1CFN",
        layers=[
        # Use a single PolynomialFunctionNode of degree 3
        # The direction is fixed to [1.0] as it's a 1D input
        SequentialCompositionLayer(
            name="Koza1PolynomialLayer",
            function_nodes=[
            node_factory.create("Polynomial", input_dim=1, degree=3,
                                coefficients=torch.zeros(4), # Initialize coefficients to zeros
                                direction=torch.tensor([1.0])) # Fixed direction for 1D input
        ])
    ])

    print("Initial Network Structure:")
    print(network.describe())

    # 3. Train the network
    trainer = Trainer(network, learning_rate=0.005, grad_clip_norm=1.0)

    trainer.train(
        train_loader, val_loader=val_loader, epochs=2000,
        loss_fn=nn.MSELoss(), early_stopping_patience=500
    )

    # Plot loss
    trainer.plot_loss('pytorch_koza1_loss.png')

    # 4. Visualize learned function vs true function
    plt.figure(figsize=(10, 6))
    plt.plot(x_data.numpy(), y_true.numpy(), label='True Koza-1 Function', linestyle='--', color='gray')
    network.eval()
    with torch.no_grad():
        y_pred = network(x_data).numpy()
    plt.plot(x_data.numpy(), y_pred, label='Learned Koza-1 Function', color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('PyTorch Koza-1: True vs. Learned Function')
    plt.legend()
    plt.grid(True)
    plt.savefig('pytorch_koza1_visualization.png')
    print("Plot saved to pytorch_koza1_visualization.png")
    plt.close()

    # 5. Interpret the learned parameters
    print("\n--- Learned Koza-1 Parameters ---")
    learned_poly_node = network.layers[0].function_nodes[0]
    learned_coeffs = learned_poly_node.coefficients.detach().numpy()

    print(f"\nTrue Coefficients (for [1, x, x^2, x^3]): {true_coeffs.numpy()}")
    print(f"Learned Coefficients (for [1, x, x^2, x^3]): {learned_coeffs}")
    print(f"Difference from True Coefficients: {np.abs(learned_coeffs - true_coeffs.numpy())}")
    
    print("--------------------------------------------------")
    return network

if __name__ == '__main__':
    model = run()
    print("\n--- Full Model Interpretation from Runner ---")
    interpret_model(model)

