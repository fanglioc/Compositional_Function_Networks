import numpy as np
import matplotlib.pyplot as plt
from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory
from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.Framework import Trainer, softmax_cross_entropy_loss
from cfn_numpy.interpretability import interpret_model

def classification_example():
    """Classification example: Spiral dataset."""
    print("Running Classification Example: Spiral Dataset")
    
    # Generate spiral dataset
    def generate_spiral_data(n_samples_per_class=300, n_classes=3, noise=0.2):
        X = np.zeros((n_samples_per_class * n_classes, 2))
        y = np.zeros(n_samples_per_class * n_classes, dtype='uint8')
        
        for class_idx in range(n_classes):
            ix = range(n_samples_per_class * class_idx, n_samples_per_class * (class_idx + 1))
            r = np.linspace(0.0, 1, n_samples_per_class)  # radius
            t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples_per_class) + np.random.randn(n_samples_per_class) * noise  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = class_idx
        
        return X, y
    
    # Generate data
    np.random.seed(42)
    n_classes = 3
    X, y = generate_spiral_data(n_samples_per_class=300, n_classes=n_classes, noise=0.2)
    
    # Split into train and validation sets
    indices = np.random.permutation(X.shape[0])
    split_idx = int(0.8 * X.shape[0])
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    y_train_one_hot = np.eye(n_classes)[y_train]
    y_val_one_hot = np.eye(n_classes)[y_val]
    
    # Create function node factory
    node_factory = FunctionNodeFactory()
    
    # Create layer factory
    layer_factory = CompositionLayerFactory(node_factory)
    
    # Create network for classification
    network = CompositionFunctionNetwork(name="ClassificationCFN")
    
    # First layer: Radial basis functions and directional features
    input_dim = 2
    feature_layer = layer_factory.create_parallel(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": input_dim, "output_dim": input_dim}),
            ("PolynomialFunctionNode", {"input_dim": input_dim, "degree": 2}),
            ("GaussianFunctionNode", {"input_dim": input_dim}),
            ("SigmoidFunctionNode", {"input_dim": input_dim}),
        ],
        combination='concat',
        name="FeatureExtractionLayer"
    )
    output_dim_feature_layer = feature_layer.output_dim
    
    # Output layer
    output_layer = layer_factory.create_sequential(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": output_dim_feature_layer, "output_dim": n_classes})
        ],
        name="OutputLayer"
    )
    
    network.add_layer(feature_layer)
    network.add_layer(output_layer)
    
    # Print network description
    print(network.describe())
    
    # Initialize trainer
    trainer = Trainer(network, softmax_cross_entropy_loss, learning_rate=0.02, batch_size=64)
    
    # Train the network
    trainer.train(
        X_train, y_train_one_hot, X_val, y_val_one_hot, epochs=300, 
        verbose=True, early_stopping=True, patience=30
    )
    
    # Plot loss
    trainer.plot_loss()
    
    # Visualize the decision boundaries
    visualize_classification_results(network, X, y, n_classes)
    
    interpret_model(network)
    return network

def visualize_classification_results(network, X, y, n_classes):
    """Visualize classification results with decision boundaries."""
    # Create a grid of points
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Evaluate network on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = network.forward(grid_points)
    
    # Apply softmax to get probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    Z = softmax(Z)
    
    # Get predicted class
    Z_class = np.argmax(Z, axis=1).reshape(xx.shape)
    
    # Plot decision boundaries
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_class, cmap=plt.cm.Spectral, alpha=0.8)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, 
                         edgecolors='k', s=40)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundaries")
    
    # Add colorbar
    plt.colorbar(scatter, ticks=range(n_classes))
    
    #plt.show()
    plt.savefig('plot.png') 
