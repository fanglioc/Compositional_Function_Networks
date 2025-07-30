import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os


from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory, SequentialCompositionLayer, ParallelCompositionLayer, ConditionalCompositionLayer
from cfn_numpy.FunctionNodes import FunctionNodeFactory, LinearFunctionNode, GaussianFunctionNode, SigmoidFunctionNode, PolynomialFunctionNode, SinusoidalFunctionNode, ExponentialFunctionNode, StepFunctionNode
from cfn_numpy.Framework import Trainer, mse_loss
from cfn_numpy.interpretability import interpret_model

def advanced_example():
    """
    Advanced example: Combining different composition types
    for approximating a complex function with local behavior.
    """
    print("Running Advanced Example: Local Expert Functions")
    
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
    X = np.random.uniform(-2, 2, (n_samples, 2))
    y = np.array([target_function(x[0], x[1]) for x in X]).reshape(-1, 1)
    
    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create factories
    node_factory = FunctionNodeFactory()
    layer_factory = CompositionLayerFactory(node_factory)
    
    # Create network
    network = CompositionFunctionNetwork(name="AdvancedCFN")
    
    # First layer: Feature extraction
    input_dim = 2
    feature_layer = layer_factory.create_parallel([
        # Raw coordinates
        ("LinearFunctionNode", {"input_dim": input_dim, "output_dim": 2, 
                               "weights": np.eye(2), "bias": np.zeros(2)}),
        
        # Radius (distance from origin)
        ("PolynomialFunctionNode", {"input_dim": input_dim, "degree": 2, 
                                   "coefficients": np.array([0.0, 0.0, 1.0])}),
        
        # Sine and cosine of angle
        ("SinusoidalFunctionNode", {"input_dim": input_dim, 
                                   "direction": np.array([1.0, 0.0])}),
        ("SinusoidalFunctionNode", {"input_dim": input_dim, 
                                   "direction": np.array([0.0, 1.0])}),
    ], combination="concat", name="FeatureExtractionLayer")
    
    network.add_layer(feature_layer)
    
    # Second layer: Region-specific experts
    expert_layer = layer_factory.create_conditional(
        # Condition nodes (region detectors)
        [
            # Region 1: Inner circle (r < 0.5)
            ("StepFunctionNode", {"input_dim": 5, "bias": 0.5, 
                               "direction": np.array([0, 0, -1, 0, 0]), "smoothing": 0.1}),
            
            # Region 2: Middle ring (0.5 < r < 1.0)
            ("StepFunctionNode", {"input_dim": 5, "bias": 0.5, 
                               "direction": np.array([0, 0, 1, 0, 0]), "smoothing": 0.1}),
            
            # Region 3: Outer ring (1.0 < r < 1.5)
            ("StepFunctionNode", {"input_dim": 5, "bias": 1.0, 
                               "direction": np.array([0, 0, 1, 0, 0]), "smoothing": 0.1}),
            
            # Region 4: Far field (r > 1.5)
            ("StepFunctionNode", {"input_dim": 5, "bias": 1.5, 
                               "direction": np.array([0, 0, 1, 0, 0]), "smoothing": 0.1}),
        ],
        # Function nodes (region-specific experts)
        [
            # Expert for Region 1: Sinusoidal pattern
            ("ParallelCompositionLayer", {
                "function_node_specs": [
                    ("SinusoidalFunctionNode", {"input_dim": 5, 
                                   "direction": np.array([0, 0, 0, 1, 1]), "frequency": 5.0}),
                    ("LinearFunctionNode", {"input_dim": 5, "output_dim": 1,
                                   "weights": np.array([[0, 0, 0.5, 0, 0]]).T, "bias": np.zeros(1)})
                ],
                "combination": "product"
            }),
            
            # Expert for Region 2: Polynomial
            ("PolynomialFunctionNode", {"input_dim": 5, "degree": 2,
                               "direction": np.array([1, 1, 0, 0, 0])}),
            
            # Expert for Region 3: Exponential decay
            ("ExponentialFunctionNode", {"input_dim": 5, 
                               "direction": np.array([0, 0, 1, 0, 0]), "rate": -2.0, "shift": -1.0, "scale": 0.3}),
            
            # Expert for Region 4: Linear
            ("LinearFunctionNode", {"input_dim": 5, "output_dim": 1,
                               "weights": np.array([[0.1, -0.2, 0, 0, 0]]).T, "bias": np.array([0.05])}),
        ],
        name="RegionalExpertsLayer"
    )
    
    network.add_layer(expert_layer)
    
    # Print network description
    print(network.describe())
    
    # Initialize trainer
    trainer = Trainer(network, mse_loss, learning_rate=0.001, batch_size=128)
    
    # Train the network
    train_losses, val_losses = trainer.train(
        X_train, y_train, X_val, y_val, epochs=300, 
        verbose=True, early_stopping=True, patience=30
    )
    
    # Plot loss
    trainer.plot_loss()
    
    # Create visualization function
    def visualize_advanced_results(network, target_function):
        """Visualize results for the advanced example."""
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
        fig = plt.figure(figsize=(18, 12))
        
        # Plot true function
        ax1 = fig.add_subplot(231, projection='3d')
        surf1 = ax1.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('Y')
        ax1.set_title('True Function')
        
        # Plot predicted function
        ax2 = fig.add_subplot(232, projection='3d')
        surf2 = ax2.plot_surface(X1, X2, Z_pred, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('Y')
        ax2.set_title('CFN Prediction')
        
        # Plot error
        ax3 = fig.add_subplot(233, projection='3d')
        surf3 = ax3.plot_surface(X1, X2, Z_error, cmap='hot', alpha=0.8)
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X2')
        ax3.set_zlabel('Error')
        ax3.set_title('Absolute Error')
        
        # Plot true function contour
        ax4 = fig.add_subplot(234)
        contour1 = ax4.contourf(X1, X2, Z_true, 20, cmap='viridis')
        ax4.set_xlabel('X1')
        ax4.set_ylabel('X2')
        ax4.set_title('True Function (Contour)')
        plt.colorbar(contour1, ax=ax4)
        
        # Plot predicted function contour
        ax5 = fig.add_subplot(235)
        contour2 = ax5.contourf(X1, X2, Z_pred, 20, cmap='viridis')
        ax5.set_xlabel('X1')
        ax5.set_ylabel('X2')
        ax5.set_title('CFN Prediction (Contour)')
        plt.colorbar(contour2, ax=ax5)
        
        # Plot error contour
        ax6 = fig.add_subplot(236)
        contour3 = ax6.contourf(X1, X2, Z_error, 20, cmap='hot')
        ax6.set_xlabel('X1')
        ax6.set_ylabel('X2')
        ax6.set_title('Absolute Error (Contour)')
        plt.colorbar(contour3, ax=ax6)
        
        plt.tight_layout()
        #plt.show()
        plt.savefig('plot.png') 
        
        # Print error statistics
        mse = np.mean(Z_error**2)
        mae = np.mean(np.abs(Z_error))
        max_error = np.max(Z_error)
        
        print(f"Advanced Example Metrics:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
        print(f"Maximum Error: {max_error:.6f}")
    
    # Visualize results
    visualize_advanced_results(network, target_function)
    
    # Interpret the model
    interpret_model(network)
    
    return network