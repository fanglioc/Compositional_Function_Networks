import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from autocfn.autocfn_search_numpy import AutoCFN_Numpy, genome_to_network
from cfn_numpy.Framework import Trainer, binary_cross_entropy_loss

def main():
    """Main function to run the AutoCFN NumPy benchmark."""
    print("--- Running AutoCFN NumPy Example on Breast Cancer Dataset ---")
    
    # 1. Load and preprocess data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Run AutoCFN search
    autocfn = AutoCFN_Numpy(X_train, y_train, task='classification', population_size=30, generations=10)
    best_genome = autocfn.run_search()

    # 3. Train the best discovered network from scratch
    print("\n--- Training the Best Discovered Architecture ---")
    input_dim = X_train.shape[1]
    output_dim = 1 # Binary classification
    
    best_network = genome_to_network(best_genome, input_dim, output_dim, 'classification')
    
    print("Best Network Architecture:")
    print(best_network.describe())
    
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    # Use the improved trainer for final training
    trainer = Trainer(best_network, binary_cross_entropy_loss, learning_rate=0.001, l2_lambda=1e-5)
    trainer.train(X_train, y_train_reshaped, X_test, y_test_reshaped, epochs=200, early_stopping=True, patience=30, lr_decay_step=50)

    # 4. Evaluate the final model
    final_preds = best_network.forward(X_test)
    final_acc = accuracy_score(y_test, (final_preds > 0.5).astype(int))
    
    print("\n--- Final Evaluation ---")
    print(f"Accuracy of the best discovered model: {final_acc:.4f}")

if __name__ == "__main__":
    main()
