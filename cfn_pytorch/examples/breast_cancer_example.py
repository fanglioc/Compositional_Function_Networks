

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def run():
    """
    Real-World Classification Example: Breast Cancer Wisconsin (Diagnostic) Dataset.
    This example demonstrates CFN's ability to handle tabular data and binary classification.
    """
    print("--- Running PyTorch Classification Example: Breast Cancer Wisconsin ---")

    # 1. Load and preprocess Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train_scaled.shape[1] # 30 features for Breast Cancer dataset

    # 2. Create CFN for Breast Cancer Classification
    node_factory = FunctionNodeFactory()

    # Create a non-trainable linear node to act as a passthrough
    passthrough_node = node_factory.create("Linear", input_dim=input_dim, output_dim=input_dim,
                                           weights=torch.eye(input_dim), bias=torch.zeros(input_dim))
    passthrough_node.set_trainable(False)

    # First layer: Parallel composition of various basis functions for feature extraction
    feature_layer = ParallelCompositionLayer(
        function_nodes=[
            # Non-trainable Linear features (direct input passthrough)
            passthrough_node,
            
            # Polynomial features (degree 2)
            node_factory.create("Polynomial", input_dim=input_dim, degree=2),
            
            # Gaussian RBFs
            node_factory.create("Gaussian", input_dim=input_dim, center=torch.zeros(input_dim), width=1.0),
            node_factory.create("Gaussian", input_dim=input_dim, center=torch.ones(input_dim) * 0.5, width=0.8),
            node_factory.create("Gaussian", input_dim=input_dim, center=torch.ones(input_dim) * -0.5, width=0.8),

            # Sigmoid functions
            node_factory.create("Sigmoid", input_dim=input_dim, direction=torch.randn(input_dim)),
            node_factory.create("Sigmoid", input_dim=input_dim, direction=torch.randn(input_dim)),

        ],
        combination='concat'
    )
    
    # Calculate output_dim of feature_layer
    # Linear (30) + Polynomial (1) + Gaussian (3*1) + Sigmoid (2*1) = 30 + 1 + 3 + 2 = 36
    output_dim_feature_layer = input_dim + 1 + 3 + 2

    # Second layer: Linear combination of the extracted features to a single logit output
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
        loss_fn=nn.BCEWithLogitsLoss(), # Use BCEWithLogitsLoss for numerical stability
        early_stopping_patience=50
    )

    # Plot loss
    trainer.plot_loss('pytorch_breast_cancer_loss.png')

    # 4. Evaluate and Interpret
    network.eval()
    with torch.no_grad():
        # Move test tensor to the correct device for evaluation
        X_test_tensor = X_test_tensor.to(device)
        y_pred_logits = network(X_test_tensor)
        y_pred_probs = torch.sigmoid(y_pred_logits) # Apply sigmoid to logits to get probabilities
    
    y_pred_class = (y_pred_probs.cpu() > 0.5).float()

    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_class.numpy())
    conf_matrix = confusion_matrix(y_test_tensor.numpy(), y_pred_class.numpy())

    print("\n--- Breast Cancer Classification Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malignant'], rotation=45)
    plt.yticks(tick_marks, ['Benign', 'Malignant'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    plt.savefig('pytorch_breast_cancer_visualization.png')
    print("Plot saved to pytorch_breast_cancer_visualization.png")
    plt.close()
    
    print("--------------------------------------------------------")
    return network

if __name__ == '__main__':
    model = run()
    print("\n--- Breast Cancer Model Interpretation from Runner ---")
    interpret_model(model)
