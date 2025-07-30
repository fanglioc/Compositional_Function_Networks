import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from autocfn.autocfn_search_numpy import AutoCFN_Numpy, genome_to_network
from cfn_numpy.Framework import Trainer, softmax_cross_entropy_loss

def main():
    """Main function to run the AutoCFN NumPy benchmark on MNIST."""
    print("--- Running AutoCFN NumPy Example on MNIST Dataset ---")
    
    # 1. Load and preprocess data
    print("Loading MNIST dataset...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist["data"], mnist["target"].astype(np.uint8)
    except Exception as e:
        print(f"Could not download MNIST dataset. Error: {e}")
        return

    # Use a smaller subset for faster search and training
    n_samples_search = 2000
    n_samples_train = 5000
    
    X_search, _, y_search, _ = train_test_split(X, y, train_size=n_samples_search, stratify=y, random_state=42)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, train_size=n_samples_train, stratify=y, random_state=42)

    # Normalize data
    X_search = X_search / 255.0
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    # 2. Run AutoCFN search on a smaller subset
    print(f"\n--- Starting AutoCFN Search on {n_samples_search} samples ---")
    autocfn = AutoCFN_Numpy(X_search, y_search, task='classification', population_size=30, generations=10, max_layers=4, max_nodes_per_layer=3)
    # For MNIST, restrict the search to more appropriate node types
    autocfn.node_pool = ['LinearFunctionNode', 'ReLUFunctionNode']
    best_genome = autocfn.run_search()

    # 3. Train the best discovered network from scratch on a larger dataset
    print(f"\n--- Training the Best Discovered Architecture on {n_samples_train} samples ---")
    input_dim = X_train_full.shape[1]
    n_classes = len(np.unique(y_train_full))
    
    best_network = genome_to_network(best_genome, input_dim, n_classes, 'classification')
    
    print("Best Network Architecture:")
    print(best_network.describe())
    
    # One-hot encode labels for training
    y_train_one_hot = np.eye(n_classes)[y_train_full]
    y_test_one_hot = np.eye(n_classes)[y_test]
    
    # Use the improved trainer for final training
    trainer = Trainer(best_network, softmax_cross_entropy_loss, learning_rate=0.001, l2_lambda=1e-5)
    trainer.train(X_train_full, y_train_one_hot, X_test, y_test_one_hot, epochs=150, early_stopping=True, patience=25, lr_decay_step=50)

    # 4. Evaluate the final model
    final_preds_raw = best_network.forward(X_test)
    final_preds = np.argmax(final_preds_raw, axis=1)
    final_acc = accuracy_score(y_test, final_preds)
    
    print("\n--- Final Evaluation ---")
    print(f"Accuracy of the best discovered model on MNIST: {final_acc:.4f}")

if __name__ == "__main__":
    main()
