from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory
from cfn_numpy.FunctionNodes import FunctionNodeFactory
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cfn_numpy.Framework import Trainer, mse_loss
from cfn_numpy.interpretability import interpret_model

def regression_example():
    """Regression example: Approximating a complex 2D function."""
    print("Running Regression Example: 2D Function Approximation")
    
    # Generate synthetic data
    def target_function(x1, x2):
        return np.sin(x1 * 3) * np.cos(x2 * 2) + 0.2 * x1**2 - 0.3 * x2
    
    # Generate random data points
    np.random.seed(42)
    n_samples = 1000
    X = np.random.uniform(-2, 2, (n_samples, 2))
    y = np.array([target_function(x[0], x[1]) for x in X]).reshape(-1, 1)
    
    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create function node factory
    node_factory = FunctionNodeFactory()
    
    # Create layer factory
    layer_factory = CompositionLayerFactory(node_factory)
    
    # Create network for regression
    network = CompositionFunctionNetwork(name="RegressionCFN")
    
    # First layer: Parallel composition of different basis functions
    input_dim = 2
    parallel_layer = layer_factory.create_parallel([
        # Gaussian RBFs with different centers and widths
        ("GaussianFunctionNode", {"input_dim": input_dim, "center": np.array([0.0, 0.0]), "width": 0.8}),
        ("GaussianFunctionNode", {"input_dim": input_dim, "center": np.array([1.0, 1.0]), "width": 0.8}),
        ("GaussianFunctionNode", {"input_dim": input_dim, "center": np.array([-1.0, -1.0]), "width": 0.8}),
        ("GaussianFunctionNode", {"input_dim": input_dim, "center": np.array([1.0, -1.0]), "width": 0.8}),
        ("GaussianFunctionNode", {"input_dim": input_dim, "center": np.array([-1.0, 1.0]), "width": 0.8}),
        
        # Sinusoidal functions with different frequencies and directions
        ("SinusoidalFunctionNode", {"input_dim": input_dim, "frequency": 1.0}),
        ("SinusoidalFunctionNode", {"input_dim": input_dim, "frequency": 2.0}),
        ("SinusoidalFunctionNode", {"input_dim": input_dim, "frequency": 3.0}),
        
        # Polynomial functions
        ("PolynomialFunctionNode", {"input_dim": input_dim, "degree": 2}),
        ("PolynomialFunctionNode", {"input_dim": input_dim, "degree": 3}),
        
    ], combination='concat', name="BasisFunctionsLayer")
    
    network.add_layer(parallel_layer)
    
    # Second layer: Linear combination of basis functions
    linear_layer = layer_factory.create_sequential([
        ("LinearFunctionNode", {"input_dim": 10, "output_dim": 1})
    ], name="CombinationLayer")
    
    network.add_layer(linear_layer)
    
    # Print network description
    print(network.describe())
    
    # Initialize trainer
    trainer = Trainer(network, mse_loss, learning_rate=0.01, batch_size=64)
    
    # Train the network
    train_losses, val_losses = trainer.train(
        X_train, y_train, X_val, y_val, epochs=200, 
        verbose=True, early_stopping=True, patience=20
    )
    
    # Plot loss
    trainer.plot_loss()
    
    # Visualize the learned function
    visualize_regression_results(network, target_function)
    
    return network

def visualize_regression_results(network, target_function):
    """Visualize regression results."""
    # Create a grid of points
    x1_grid = np.linspace(-2, 2, 100)
    x2_grid = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Evaluate target function on grid
    Z_true = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z_true[i, j] = target_function(X1[i, j], X2[i, j])
    
    # Evaluate network on grid
    grid_points = np.column_stack((X1.flatten(), X2.flatten()))
    Z_pred = network.forward(grid_points).reshape(X1.shape)
    
    # Calculate error
    Z_error = np.abs(Z_pred - Z_true)
    
    # Plot results
    fig = plt.figure(figsize=(18, 6))
    
    # Plot true function
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')
    ax1.set_title('True Function')
    
    # Plot predicted function
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X1, X2, Z_pred, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Y')
    ax2.set_title('CFN Prediction')
    
    # Plot error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X1, X2, Z_error, cmap='hot', alpha=0.8)
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_zlabel('Error')
    ax3.set_title('Absolute Error')
    
    # Calculate error statistics
    mse = np.mean((Z_pred - Z_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(Z_pred - Z_true))
    max_error = np.max(np.abs(Z_pred - Z_true))

    # Add metrics to the plot title
    fig.suptitle(f"Regression Metrics: MSE={mse:.4f}, MAE={mae:.4f}, Max Error={max_error:.4f}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
    #plt.show()
    plt.savefig('plot.png') 
    
    # --- Performance Analysis ---
    # Calculate target data statistics for context
    y_range = np.max(Z_true) - np.min(Z_true)
    y_std = np.std(Z_true)
    nrmse = rmse / y_std if y_std > 0 else 0

    print("----------------------------------------")
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
    
    if nrmse < 0.1:
        print("\nInterpretation: Excellent performance. The model's error is very small compared to the data's variance.")
    elif nrmse < 0.5:
        print("\nInterpretation: Good performance. The model provides a reasonable approximation of the target function.")
    elif nrmse < 1.0:
        print("\nInterpretation: Moderate performance. The model has learned the basic shape but has significant errors.")
    else:
        print("\nInterpretation: Poor performance. The model's predictions are not much better than simply guessing the mean value.")

def main():
    regression_example()