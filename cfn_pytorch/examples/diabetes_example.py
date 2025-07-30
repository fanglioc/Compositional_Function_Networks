

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def run():
    """
    Real-World Regression Example: Diabetes Dataset.
    This example demonstrates CFN's ability to handle tabular data and learn complex relationships.
    """
    print("--- Running PyTorch Regression Example: Diabetes Dataset ---")

    # 1. Load and preprocess Diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target.reshape(-1, 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize target (optional, but often helps with MSE loss)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train_scaled.shape[1] # 10 features for diabetes dataset

    # 2. Create CFN for Diabetes Regression
    node_factory = FunctionNodeFactory()

    # First layer: Parallel composition of various basis functions for feature extraction
    feature_layer = ParallelCompositionLayer(
        function_nodes=[
            # Linear features (direct input) - now trainable
            node_factory.create("Linear", input_dim=input_dim, output_dim=input_dim),
            
            # Polynomial features (degree 2) for non-linear interactions
            node_factory.create("Polynomial", input_dim=input_dim, degree=2),
            
            # Gaussian RBFs for localized features
            node_factory.create("Gaussian", input_dim=input_dim, center=torch.zeros(input_dim), width=1.0),
            node_factory.create("Gaussian", input_dim=input_dim, center=torch.ones(input_dim) * 0.5, width=0.8),
            node_factory.create("Gaussian", input_dim=input_dim, center=torch.ones(input_dim) * -0.5, width=0.8),

            # Sigmoid functions for thresholding/activation
            node_factory.create("Sigmoid", input_dim=input_dim, direction=torch.randn(input_dim)),
            node_factory.create("Sigmoid", input_dim=input_dim, direction=torch.randn(input_dim)),

        ],
        combination='concat'
    )
    
    # Calculate output_dim of feature_layer
    # Linear (10) + Polynomial (1) + Gaussian (3*1) + Sigmoid (2*1) = 10 + 1 + 3 + 2 = 16
    output_dim_feature_layer = input_dim + 1 + 3 + 2

    # Second layer: Linear combination of the extracted features to a single output
    output_layer = SequentialCompositionLayer(
        name="OutputLayer",
        function_nodes=[
        node_factory.create("Linear", input_dim=output_dim_feature_layer, output_dim=1)
    ])

    network = CompositionFunctionNetwork(layers=[
        feature_layer,
        output_layer
    ])

    print("Initial Network Structure:")
    print(network.describe())

    # 3. Train the network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trainer = Trainer(network, learning_rate=0.005, grad_clip_norm=1.0, device=device)

    trainer.train(
        train_loader, val_loader=test_loader, epochs=500,
        loss_fn=nn.MSELoss(), early_stopping_patience=50
    )

    # Plot loss
    trainer.plot_loss('pytorch_diabetes_loss.png')

    # 4. Evaluate and Interpret
    network.eval()
    with torch.no_grad():
        # Move test tensor to the correct device for evaluation
        X_test_tensor = X_test_tensor.to(device)
        y_pred_scaled = network(X_test_tensor)
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())
    y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())

    mse = np.mean((y_pred - y_test_original)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test_original))

    print("\n--- Diabetes Regression Metrics (on original scale) ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Visualize predictions vs true values for a subset
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred, alpha=0.6)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], '--r', label='Ideal Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Diabetes: True vs. Predicted Values')
    plt.grid(True)
    plt.legend()
    plt.savefig('pytorch_diabetes_visualization.png')
    print("\nPlot saved to pytorch_diabetes_visualization.png")
    plt.close()
    
    print("--------------------------------------------------")
    return network

if __name__ == '__main__':
    model = run()
    # 5. Interpret the model
    print("\n--- Final Model Interpretation from Runner ---")
    interpret_model(model)


