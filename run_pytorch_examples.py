import os
import importlib
from cfn_numpy.interpretability import interpret_model

def main():
    """Dynamically finds and runs PyTorch examples."""
    example_dir = "cfn_pytorch/examples"
    # Get a list of Python files in the examples directory, sorted for consistent order
    example_files = sorted([f for f in os.listdir(example_dir) if f.endswith('.py') and not f.startswith('__')])
    
    # Create a mapping from number to module name
    example_map = {i+1: f.replace('.py', '') for i, f in enumerate(example_files)}

    if not example_map:
        print("No examples found.")
        return

    while True:
        print("\nAvailable PyTorch Examples:")
        for i, name in example_map.items():
            print(f"  {i}. {name.replace('_', ' ').title()}")
        print("  A. Run All Examples")
        print("  Q. Quit")

        choice = input("Enter your choice: ").strip().upper()

        if choice == 'Q':
            break
        elif choice == 'A':
            for name in example_map.values():
                print(f"\n{'='*20} Running: {name} {'='*20}")
                try:
                    module_path = f"cfn_pytorch.examples.{name}"
                    module = importlib.import_module(module_path)
                    model = module.run()
                    if model:
                        print(f"\n--- Final Model Interpretation for {name} ---")
                        interpret_model(model)
                except Exception as e:
                    print(f"\nError running {name}: {e}")
                print(f"{'='*50}\n")
            continue

        try:
            choice_num = int(choice)
            if choice_num in example_map:
                name = example_map[choice_num]
                print(f"\n{'='*20} Running: {name} {'='*20}")
                try:
                    module_path = f"cfn_pytorch.examples.{name}"
                    module = importlib.import_module(module_path)
                    model = module.run()
                    if model:
                        print(f"\n--- Final Model Interpretation for {name} ---")
                        interpret_model(model)
                except Exception as e:
                    print(f"\nError running {name}: {e}")
                print(f"{'='*50}\n")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number, 'A', or 'Q'.")

if __name__ == "__main__":
    main()
