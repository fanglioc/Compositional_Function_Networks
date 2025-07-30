from cfn_numpy.examples.regression import regression_example
from cfn_numpy.examples.classification import classification_example
from cfn_numpy.examples.conditional_composition_example import advanced_example
from cfn_numpy.examples.physics_example import physics_example
from cfn_numpy.examples.koza1_example import koza1_example
from cfn_numpy.examples.california_housing_example import california_housing_example
from cfn_numpy.examples.breast_cancer_example import breast_cancer_example
from cfn_numpy.examples.mnist_example import mnist_example
from cfn_numpy.interpretability import interpret_model

def main():
    """Run example applications of the Compositional Function Network."""
    examples = {
        "1": ("Regression Example", regression_example),
        "2": ("Classification Example", classification_example),
        "3": ("Advanced Example", advanced_example),
        "4": ("Physics Example", physics_example),
        "5": ("Koza-1 Symbolic Regression Example", koza1_example),
        "6": ("California Housing Regression Example", california_housing_example),
        "7": ("Breast Cancer Classification Example", breast_cancer_example),
        "8": ("MNIST Example", mnist_example),
    }

    print("=== Compositional Function Network (CFN) Examples ===\n")
    print("Select an example to run:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  A. Run All Examples")
    print("  Q. Quit")

    while True:
        choice = input("\nEnter your choice: ").strip().upper()

        if choice == 'Q':
            print("Exiting examples.")
            break
        elif choice == 'A':
            for key in sorted(examples.keys(), key=int):
                name, func = examples[key]
                print(f"\n--- Running {name} ---")
                model = func()
                interpret_model(model)
                print("\n" + "="*50 + "\n")
            break
        elif choice in examples:
            name, func = examples[choice]
            print(f"\n--- Running {name} ---")
            model = func()
            interpret_model(model)
            print("\n" + "="*50 + "\n")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()