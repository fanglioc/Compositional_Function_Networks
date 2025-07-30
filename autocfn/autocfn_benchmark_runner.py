import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from autocfn.autocfn_search import AutoCFN, genome_to_network

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
        y = y.reshape(-1, 1) # Reshape y for regression
        return X, y, 'regression'
    else:
        raise ValueError(f"Unknown dataset: {name}")

def run_cfn_benchmark(X_train, y_train, X_test, y_test, task):
    """Trains and evaluates the manually defined CFN model."""
    print(f"--- Running Manually Defined CFN Benchmark ({task}) ---")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    input_dim = X_train.shape[1]
    node_factory = FunctionNodeFactory()
    feature_layer = ParallelCompositionLayer(
        function_nodes=[
            node_factory.create("Linear", input_dim=input_dim, output_dim=input_dim),
            node_factory.create("Polynomial", input_dim=input_dim, degree=2),
            node_factory.create("Gaussian", input_dim=input_dim),
            node_factory.create("Sigmoid", input_dim=input_dim),
        ],
        combination='concat'
    )
    output_dim_feature_layer = feature_layer.output_dim

    if task == 'classification':
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            output_layer = SequentialCompositionLayer([
                node_factory.create("Linear", input_dim=output_dim_feature_layer, output_dim=1),
                node_factory.create("Sigmoid", input_dim=1)
            ])
            loss_fn = nn.BCELoss()
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            output_layer = SequentialCompositionLayer([
                node_factory.create("Linear", input_dim=output_dim_feature_layer, output_dim=n_classes)
            ])
            loss_fn = nn.CrossEntropyLoss()
    else: # regression
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        output_layer = SequentialCompositionLayer([
            node_factory.create("Linear", input_dim=output_dim_feature_layer, output_dim=1)
        ])
        loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    network = CompositionFunctionNetwork(layers=[feature_layer, output_layer])
    trainer = Trainer(network, learning_rate=0.005, grad_clip_norm=1.0)
    trainer.train(train_loader, val_loader=test_loader, epochs=100, loss_fn=loss_fn)

    network.eval()
    with torch.no_grad():
        preds = network(X_test_tensor).numpy()

    if task == 'classification':
        if n_classes == 2:
            acc = accuracy_score(y_test, (preds > 0.5).astype(int))
            auc = roc_auc_score(y_test, preds)
            print(f"CFN Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            return {'accuracy': acc, 'auc': auc}
        else:
            pred_classes = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test, pred_classes)
            print(f"CFN Accuracy: {acc:.4f}")
            return {'accuracy': acc}

    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"CFN RMSE: {rmse:.4f}")
        return {'rmse': rmse}

def run_autocfn_benchmark(X_train, y_train, X_test, y_test, task):
    """Runs AutoCFN to find an architecture and evaluates it."""
    print(f"--- Running AutoCFN Benchmark ({task}) ---")
    
    # Run AutoCFN search
    autocfn = AutoCFN(X_train, y_train, task=task, population_size=50, generations=20) # Increased for better search
    best_genome = autocfn.run_search()

    # Train and evaluate the best discovered architecture
    input_dim = X_train.shape[1]
    if task == 'classification':
        n_classes = len(np.unique(y_train))
        output_dim = 1 if n_classes == 2 else n_classes
        loss_fn = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1) if n_classes == 2 else torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1) if n_classes == 2 else torch.tensor(y_test, dtype=torch.long)
    else:
        output_dim = 1
        loss_fn = nn.MSELoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    network = genome_to_network(best_genome, input_dim, output_dim, task)
    trainer = Trainer(network, learning_rate=0.005, grad_clip_norm=1.0)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    trainer.train(train_loader, val_loader=test_loader, epochs=500, loss_fn=loss_fn) # Increased epochs for final training

    network.eval()
    with torch.no_grad():
        preds = network(torch.tensor(X_test, dtype=torch.float32)).numpy()

    if task == 'classification':
        if n_classes == 2:
            acc = accuracy_score(y_test, (preds > 0.5).astype(int))
            auc = roc_auc_score(y_test, preds)
            print(f"AutoCFN Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            return {'accuracy': acc, 'auc': auc}
        else:
            pred_classes = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test, pred_classes)
            print(f"AutoCFN Accuracy: {acc:.4f}")
            return {'accuracy': acc}
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"AutoCFN RMSE: {rmse:.4f}")
        return {'rmse': rmse}

def run_xgboost_benchmark(X_train, y_train, X_test, y_test, task):
    """Trains and evaluates an XGBoost model."""
    print(f"--- Running XGBoost Benchmark ({task}) ---")
    if task == 'classification':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        model = xgb.XGBRegressor(eval_metric='rmse')
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task == 'classification':
        y_pred_probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        if len(np.unique(y_train)) > 2:
            auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_pred_probs[:, 1])
        print(f"XGBoost Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return {'accuracy': acc, 'auc': auc}
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"XGBoost RMSE: {rmse:.4f}")
        return {'rmse': rmse}

def run_ebm_benchmark(X_train, y_train, X_test, y_test, task):
    """Trains and evaluates an Explainable Boosting Machine."""
    print(f"--- Running EBM Benchmark ({task}) ---")
    if task == 'classification':
        model = ExplainableBoostingClassifier()
    else:
        model = ExplainableBoostingRegressor()

    model.fit(X_train, y_train)
    
    if task == 'classification':
        y_pred_probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if len(np.unique(y_train)) > 2:
            auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_pred_probs[:, 1])
        print(f"EBM Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return {'accuracy': acc, 'auc': auc}
    else:
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"EBM RMSE: {rmse:.4f}")
        return {'rmse': rmse}

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
        autocfn_metrics = run_autocfn_benchmark(X_train, y_train, X_test, y_test, task)
        xgb_metrics = run_xgboost_benchmark(X_train, y_train, X_test, y_test, task)
        ebm_metrics = run_ebm_benchmark(X_train, y_train, X_test, y_test, task)

        results.append({
            'dataset': ds_name,
            'task': task,
            'CFN': cfn_metrics,
            'AutoCFN': autocfn_metrics,
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
                'AutoCFN': result['AutoCFN'],
                'XGBoost': result['XGBoost'],
                'EBM': result['EBM']
            }).T
            print(df.to_markdown())
        else: # regression
            df = pd.DataFrame({
                'CFN': result['CFN'],
                'AutoCFN': result['AutoCFN'],
                'XGBoost': result['XGBoost'],
                'EBM': result['EBM']
            }, index=['rmse']).T
            print(df.to_markdown())

if __name__ == "__main__":
    main()
