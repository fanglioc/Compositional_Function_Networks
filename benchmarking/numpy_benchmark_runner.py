import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork
from cfn_numpy.Framework import Trainer, mse_loss, softmax_cross_entropy_loss, binary_cross_entropy_loss

def get_dataset(name):
    """Loads a specified dataset."""
    if name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        return X, y, 'classification'
    elif name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        return X, y, 'classification'
    elif name == 'diabetes':
        data = load_diabetes()
        X, y = data.data, data.target
        return X, y, 'regression'
    else:
        raise ValueError(f"Unknown dataset: {name}")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def run_cfn_benchmark(X_train, y_train, X_test, y_test, task):
    """Trains and evaluates the CFN model."""
    print(f"--- Running CFN Benchmark ({task}) ---")
    start_time = time.time()

    input_dim = X_train.shape[1]
    node_factory = FunctionNodeFactory()
    
    # Create a factory for layers
    from cfn_numpy.CompositionLayerStructure import CompositionLayerFactory
    layer_factory = CompositionLayerFactory(node_factory)

    feature_layer = layer_factory.create_parallel(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": input_dim, "output_dim": input_dim}),
            ("PolynomialFunctionNode", {"input_dim": input_dim, "degree": 2}),
            ("GaussianFunctionNode", {"input_dim": input_dim}),
            ("SigmoidFunctionNode", {"input_dim": input_dim}),
            ("SinusoidalFunctionNode", {"input_dim": input_dim}),
            ("ExponentialFunctionNode", {"input_dim": input_dim}),
        ],
        combination='concat'
    )
    output_dim_feature_layer = feature_layer.output_dim

    hidden_layer = layer_factory.create_sequential(
        function_node_specs=[
            ("LinearFunctionNode", {"input_dim": output_dim_feature_layer, "output_dim": 64}),
            ("ReLUFunctionNode", {"input_dim": 64}),
        ]
    )
    output_dim_hidden_layer = hidden_layer.output_dim

    if task == 'classification':
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            y_train_reshaped = y_train.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            output_layer = layer_factory.create_sequential(
                function_node_specs=[
                    ("LinearFunctionNode", {"input_dim": output_dim_hidden_layer, "output_dim": 1}),
                    ("SigmoidFunctionNode", {"input_dim": 1})
                ]
            )
            loss_fn = binary_cross_entropy_loss
        else:
            y_train_reshaped = np.eye(n_classes)[y_train]
            y_test_reshaped = np.eye(n_classes)[y_test]
            output_layer = layer_factory.create_sequential(
                function_node_specs=[
                    ("LinearFunctionNode", {"input_dim": output_dim_hidden_layer, "output_dim": n_classes})
                ]
            )
            loss_fn = softmax_cross_entropy_loss
    else:  # regression
        y_train_reshaped = y_train.reshape(-1, 1)
        y_test_reshaped = y_test.reshape(-1, 1)
        output_layer = layer_factory.create_sequential(
            function_node_specs=[
                ("LinearFunctionNode", {"input_dim": output_dim_hidden_layer, "output_dim": 1})
            ]
        )
        loss_fn = mse_loss

    network = CompositionFunctionNetwork(layers=[feature_layer, hidden_layer, output_layer])
    trainer = Trainer(network, loss_fn, learning_rate=0.01, grad_clip_norm=1.0, l2_lambda=1e-4)
    trainer.train(X_train, y_train_reshaped, X_test, y_test_reshaped, epochs=150, lr_decay_step=50, early_stopping=True, patience=20)

    preds = network.forward(X_test)
    end_time = time.time()
    duration = end_time - start_time
    print(f"CFN Training Time: {duration:.4f}s")

    if task == 'classification':
        if n_classes == 2:
            acc = accuracy_score(y_test, (preds > 0.5).astype(int))
            auc = roc_auc_score(y_test, preds)
            print(f"CFN Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            return {'accuracy': acc, 'auc': auc, 'time': duration}
        else:
            pred_classes = np.argmax(preds, axis=1)
            pred_probs = softmax(preds)
            acc = accuracy_score(y_test, pred_classes)
            auc = roc_auc_score(y_test, pred_probs, multi_class='ovr')
            print(f"CFN Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            return {'accuracy': acc, 'auc': auc, 'time': duration}

    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"CFN RMSE: {rmse:.4f}")
        return {'rmse': rmse, 'time': duration}

def run_xgboost_benchmark(X_train, y_train, X_test, y_test, task):
    """Trains and evaluates an XGBoost model."""
    print(f"--- Running XGBoost Benchmark ({task}) ---")
    start_time = time.time()
    if task == 'classification':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        model = xgb.XGBRegressor(eval_metric='rmse')
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    end_time = time.time()
    duration = end_time - start_time
    print(f"XGBoost Training Time: {duration:.4f}s")

    if task == 'classification':
        y_pred_probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        if len(np.unique(y_train)) > 2:
            auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_pred_probs[:, 1])
        print(f"XGBoost Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return {'accuracy': acc, 'auc': auc, 'time': duration}
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"XGBoost RMSE: {rmse:.4f}")
        return {'rmse': rmse, 'time': duration}

def run_ebm_benchmark(X_train, y_train, X_test, y_test, task):
    """Trains and evaluates an Explainable Boosting Machine."""
    print(f"--- Running EBM Benchmark ({task}) ---")
    start_time = time.time()
    if task == 'classification':
        model = ExplainableBoostingClassifier()
    else:
        model = ExplainableBoostingRegressor()

    model.fit(X_train, y_train)
    end_time = time.time()
    duration = end_time - start_time
    print(f"EBM Training Time: {duration:.4f}s")
    
    if task == 'classification':
        y_pred_probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if len(np.unique(y_train)) > 2:
            auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_pred_probs[:, 1])
        print(f"EBM Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return {'accuracy': acc, 'auc': auc, 'time': duration}
    else:
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"EBM RMSE: {rmse:.4f}")
        return {'rmse': rmse, 'time': duration}

def main():
    """Main function to run all benchmarks."""
    datasets = ['breast_cancer', 'wine', 'diabetes']
    results = []

    for ds_name in datasets:
        print(f"\n===== Running Benchmarks on {ds_name.upper()} =====")
        X, y, task = get_dataset(ds_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        cfn_metrics = run_cfn_benchmark(X_train, y_train, X_test, y_test, task)
        xgb_metrics = run_xgboost_benchmark(X_train, y_train, X_test, y_test, task)
        ebm_metrics = run_ebm_benchmark(X_train, y_train, X_test, y_test, task)

        results.append({
            'dataset': ds_name,
            'task': task,
            'CFN': cfn_metrics,
            'XGBoost': xgb_metrics,
            'EBM': ebm_metrics
        })

    print("\n\n--- Overall Benchmark Summary ---")
    # Display results in a formatted table
    for result in results:
        print(f"\n--- {result['dataset'].upper()} ({result['task']}) ---")
        if result['task'] == 'classification':
            df = pd.DataFrame({
                'CFN': result['CFN'],
                'XGBoost': result['XGBoost'],
                'EBM': result['EBM']
            }).T
            print(df.to_markdown())
        else: # regression
            df = pd.DataFrame({
                'CFN': result['CFN'],
                'XGBoost': result['XGBoost'],
                'EBM': result['EBM']
            }, index=['rmse', 'time']).T
            print(df.to_markdown())

if __name__ == "__main__":
    main()
