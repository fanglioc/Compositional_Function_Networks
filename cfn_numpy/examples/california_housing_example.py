import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.CompositionLayerStructure import CompositionLayerFactory, CompositionFunctionNetwork
from cfn_numpy.Framework import Trainer, mse_loss
from cfn_numpy.interpretability import interpret_model

def california_housing_example():
    """
    Real-World Regression Example: California Housing Dataset.
    This example demonstrates CFN's ability to handle tabular data,
    following the robust structure from the benchmark runner.
    """
    print("Running Real-World Regression Example: California Housing")

    # 1. Load and preprocess California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features and target
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_reshaped = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_reshaped = scaler_y.transform(y_test.reshape(-1, 1))

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
            ("SigmoidFunctionNode", {"input_dim": input_dim}),  # Element-wise, preserves dim
        ],
        combination='concat',
        name="FeatureExtractionLayer"
    )
    output_dim_feature_layer = feature_layer.output_dim

    # Output layer for regression
    output_layer = layer_factory.create_sequential(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": output_dim_feature_layer, "output_dim": 1})
        ],
        name="OutputLayer"
    )

    network = CompositionFunctionNetwork(layers=[feature_layer, output_layer], name="CaliforniaHousingCFN_BenchmarkStyle")
    print(network.describe())

    # 3. Train the network using benchmark parameters
    trainer = Trainer(network, mse_loss, learning_rate=0.005, grad_clip_norm=1.0, l2_lambda=1e-4)
    trainer.train(X_train, y_train_reshaped, X_test, y_test_reshaped, epochs=150, lr_decay_step=50, early_stopping=True, patience=20)

    # Plot the training and validation loss
    trainer.plot_loss()

    # 4. Evaluate and Interpret
    print("\n--- Model Evaluation ---")
    y_pred_scaled = network.forward(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"CFN Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Visualize predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('California Housing: True vs. Predicted Values')
    plt.grid(True)
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

    # Interpret the final model
    print("\n--- Model Interpretation ---")
    interpret_model(network)
    
    return network
