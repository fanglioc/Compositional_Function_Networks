import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Add parent directory to path to allow imports


from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory
from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.Framework import Trainer
from cfn_numpy.interpretability import interpret_model

def softmax_cross_entropy_loss(predictions, targets):
    """
    Softmax activation followed by cross-entropy loss.
    """
    # Apply softmax
    exp_x = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    n_samples = predictions.shape[0]
    log_likelihood = -np.sum(targets * np.log(probs + 1e-10)) / n_samples
    
    # Compute gradient
    grad_output = (probs - targets) / n_samples
    
    return log_likelihood, grad_output

def mnist_example():
    """
    MNIST classification example.
    """
    print("Running MNIST Classification Example")

    # 1. Load MNIST dataset
    print("Loading MNIST dataset...")
    try:
        # Using fetch_openml to get MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist["data"], mnist["target"].astype(np.uint8)
    except Exception as e:
        print(f"Could not download MNIST dataset. Error: {e}")
        print("Please check your internet connection.")
        return

    # Use a smaller subset for faster training
    n_samples = 5000
    X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=42)
    
    # Reshape and normalize
    X = X / 255.0
    
    # One-hot encode labels
    n_classes = 10
    y_one_hot = np.eye(n_classes)[y]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    
    input_dim = X_train.shape[1]

    # 2. Create CFN for MNIST
    node_factory = FunctionNodeFactory()
    layer_factory = CompositionLayerFactory(node_factory)
    network = CompositionFunctionNetwork(name="MNIST_CFN")

    # Layer 1: Feature Extraction
    network.add_layer(layer_factory.create_sequential([
        ("LinearFunctionNode", {"input_dim": input_dim, "output_dim": 128}),
        ("ReLUFunctionNode", {"input_dim": 128}),
    ], name="FeatureExtractionLayer"))

    # Layer 2: Hidden Layer 1
    network.add_layer(layer_factory.create_sequential([
        ("LinearFunctionNode", {"input_dim": 128, "output_dim": 64}),
        ("ReLUFunctionNode", {"input_dim": 64}),
    ], name="HiddenLayer1"))

    # Layer 3: Hidden Layer 2
    network.add_layer(layer_factory.create_sequential([
        ("LinearFunctionNode", {"input_dim": 64, "output_dim": 32}),
        ("ReLUFunctionNode", {"input_dim": 32}),
    ], name="HiddenLayer2"))

    # Layer 4: Output Layer
    network.add_layer(layer_factory.create_sequential([
        ("LinearFunctionNode", {"input_dim": 32, "output_dim": n_classes})
    ], name="OutputLayer"))

    print(network.describe())

    # 3. Train the network
    trainer = Trainer(network, softmax_cross_entropy_loss, learning_rate=0.001, batch_size=64, grad_clip_norm=1.0, l2_lambda=1e-5)
    
    train_losses, val_losses = trainer.train(
        X_train, y_train, X_test, y_test, epochs=200,
        verbose=True, early_stopping=True, patience=30
    )

    # Plot loss
    trainer.plot_loss()

    # 4. Evaluate the model
    y_pred_raw = network.forward(X_test)
    y_pred_class = np.argmax(y_pred_raw, axis=1)
    y_true_class = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_class, y_pred_class)
    conf_matrix = confusion_matrix(y_true_class, y_pred_class)

    print(f"\n--- MNIST Classification Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # 5. Visualize some predictions
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {y_pred_class[i]}\nTrue: {y_true_class[i]}")
        plt.axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('plot.png') 

    return network