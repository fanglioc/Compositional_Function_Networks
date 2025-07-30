import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_diabetes

from autocfn.autocfn_search import AutoCFN

def main():
    """Runs the AutoCFN search on a dataset."""
    # Load a dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # Initialize and run the search
    autocfn = AutoCFN(X, y, task='regression', population_size=5, generations=2)
    best_architecture = autocfn.run_search()

    print("\n--- Best Discovered Architecture ---")
    print(best_architecture)

if __name__ == "__main__":
    main()

