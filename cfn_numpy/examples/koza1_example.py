import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports from CompositionLayerStructure, FunctionNodes, Framework


from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory
from cfn_numpy.FunctionNodes import FunctionNodeFactory, PolynomialFunctionNode
from cfn_numpy.Framework import Trainer, mse_loss
from cfn_numpy.interpretability import interpret_model

def koza1_example():
    """
    Symbolic Regression Example: Learning Koza-1 function (f(x) = x^3 + x^2 + x).
    This example demonstrates learning a cubic polynomial from data.
    """
    print("Running Symbolic Regression Example: Koza-1 (x^3 + x^2 + x)")

    # 1. Generate synthetic Koza-1 data
    # True coefficients for x^3 + x^2 + x are [0, 1, 1, 1] for [constant, x, x^2, x^3]
    true_coeffs = np.array([0.0, 1.0, 1.0, 1.0])

    n_samples = 1000
    # Generate x values in a reasonable range, e.g., -1 to 1
    x_data = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    
    # Calculate true y values
    y_true = true_coeffs[0] + true_coeffs[1]*x_data + true_coeffs[2]*(x_data**2) + true_coeffs[3]*(x_data**3)
    y_true += np.random.normal(0, 0.05, (n_samples, 1)) # Add some noise

    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_true[:split_idx], y_true[split_idx:]

    # 2. Create CFN for Koza-1
    node_factory = FunctionNodeFactory()
    layer_factory = CompositionLayerFactory(node_factory)

    network = CompositionFunctionNetwork(name="Koza1CFN")

    # Use a single PolynomialFunctionNode of degree 3
    # The direction is fixed to [1.0] as it's a 1D input
    degree = 3 # Define degree here
    polynomial_layer = layer_factory.create_sequential([
        ("PolynomialFunctionNode", {
            "input_dim": 1,
            "degree": degree,
            "coefficients": np.zeros(degree + 1), # Initialize coefficients to zeros
            "direction": np.array([1.0]) # Fixed direction for 1D input
        })
    ], name="Koza1PolynomialLayer")
    network.add_layer(polynomial_layer)

    print(network.describe())

    # 3. Train the network
    trainer = Trainer(network, mse_loss, learning_rate=0.005, batch_size=32, grad_clip_norm=1.0)

    train_losses, val_losses = trainer.train(
        x_train, y_train, x_val, y_val, epochs=2000,
        verbose=True, early_stopping=True, patience=500
    )

    # Plot loss
    trainer.plot_loss()

    # 4. Visualize learned function vs true function
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_true, label='True Koza-1 Function', linestyle='--', color='gray')
    y_pred = network.forward(x_data)
    plt.plot(x_data, y_pred, label='Learned Koza-1 Function', color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('True vs. Learned Koza-1 Function')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('plot.png') 

    # 5. Interpret the learned parameters
    print("\n--- Learned Koza-1 Parameters ---")
    interpret_model(network)

    # Extract and compare learned coefficients
    learned_poly_node = network.layers[0].function_nodes[0]
    learned_coeffs = learned_poly_node.parameters["coefficients"]

    print(f"\nTrue Coefficients (for [1, x, x^2, x^3]): {true_coeffs}")
    print(f"Learned Coefficients (for [1, x, x^2, x^3]): {learned_coeffs}")
    print(f"Difference from True Coefficients: {np.abs(learned_coeffs - true_coeffs)}")

    return network