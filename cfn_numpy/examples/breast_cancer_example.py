import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.CompositionLayerStructure import CompositionLayerFactory, CompositionFunctionNetwork
from cfn_numpy.Framework import Trainer, binary_cross_entropy_loss
from cfn_numpy.interpretability import interpret_model

def breast_cancer_example():
    """
    Real-World Classification Example: Breast Cancer Wisconsin (Diagnostic) Dataset.
    This example demonstrates CFN's ability to handle tabular data and binary classification,
    following the robust structure from the benchmark runner.
    """
    print("Running Real-World Classification Example: Breast Cancer Wisconsin")

    # 1. Load and preprocess Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape y for the loss function
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)

    # 2. Create the CFN using the benchmark architecture
    input_dim = X_train.shape[1]
    node_factory = FunctionNodeFactory()
    layer_factory = CompositionLayerFactory(node_factory)

    # Standardized feature layer from the benchmark
    feature_layer = layer_factory.create_parallel(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": input_dim, "output_dim": input_dim}),
            ("PolynomialFunctionNode", {"input_dim": input_dim, "degree": 2}),
            ("GaussianFunctionNode", {"input_dim": input_dim}),
            ("SigmoidFunctionNode", {"input_dim": input_dim}), # Note: No direction, so it's element-wise
        ],
        combination='concat',
        name="FeatureExtractionLayer"
    )
    output_dim_feature_layer = feature_layer.output_dim # This will be 30+1+1+30 = 62

    # Output layer with integrated sigmoid activation
    output_layer = layer_factory.create_sequential(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": output_dim_feature_layer, "output_dim": 1}),
            ("SigmoidFunctionNode", {"input_dim": 1, "direction": None}) # Element-wise sigmoid on the single output
        ],
        name="OutputLayer"
    )

    network = CompositionFunctionNetwork(layers=[feature_layer, output_layer], name="BreastCancerCFN_BenchmarkStyle")
    print(network.describe())

    # 3. Train the network using benchmark parameters
    trainer = Trainer(network, binary_cross_entropy_loss, learning_rate=0.005, grad_clip_norm=1.0, l2_lambda=1e-4)
    trainer.train(X_train, y_train_reshaped, X_test, y_test_reshaped, epochs=150, lr_decay_step=50, early_stopping=True, patience=20)

    # Plot the training and validation loss
    trainer.plot_loss()

    # 4. Evaluate and Interpret
    print("\n--- Model Evaluation ---")
    preds = network.forward(X_test)
    
    # Get class predictions
    pred_classes = (preds > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, pred_classes)
    auc = roc_auc_score(y_test, preds)
    conf_matrix = confusion_matrix(y_test, pred_classes)

    print(f"CFN Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Interpret the final model
    print("\n--- Model Interpretation ---")
    interpret_model(network)
    
    return network