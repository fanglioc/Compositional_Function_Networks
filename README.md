# Compositional Function Networks (CFN)

<p align="left">
  <img src="docs/cfn_logo_new.png" alt="CFN Logo Placeholder" width="150">
</p>

## Introduction

The **Compositional Function Network (CFN)** is an innovative approach to machine learning, designed as a powerful and interpretable alternative to traditional deep neural networks. Unlike black-box models, CFNs build complex functions by composing simpler, interpretable function nodes in a structured, hierarchical manner. This allows for a deeper understanding of the learned relationships within the data.

## Key Features

-   **Interpretability:** Each component (function node and layer) has a clear mathematical and conceptual meaning, making it easier to understand *why* the model makes certain predictions.
-   **Modularity:** Complex functions are constructed from a library of basic, well-understood function nodes (e.g., Gaussian, Sigmoid, Linear, Polynomial, Sinusoidal).
-   **Flexibility:** Supports various composition patterns, including parallel, sequential, and conditional compositions, enabling the creation of highly adaptable models.
-   **Alternative to Deep Learning:** Offers a different paradigm for learning complex patterns, potentially providing advantages in scenarios requiring transparency and domain-specific knowledge integration.

## Project Structure

```
Compositional_Function_Network/
├── cfn_pytorch/                  # Core PyTorch implementation of CFN
├── cfn_numpy/                    # Core NumPy implementation of CFN
├── autocfn/                      # Tools for automatic architecture search (Under development)
├── benchmarking/                 # Scripts for performance benchmarking
├── data/                         # Datasets for examples
├── docs/                         # Project documentation
├── run_pytorch_examples.py       # Script to run PyTorch examples
├── run_numpy_examples.py         # Script to run NumPy examples
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

It is recommended to use a Python virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Compositional_Function_Networks.git
    cd Compositional_Function_Networks
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install the package:**
    ```bash
    pip install .
    ```
    For development, use editable mode:
    ```bash
    pip install -e .
    ```

## Quick Start (PyTorch)

Here's a minimal example of how to define and train a CFN using the PyTorch implementation to learn a simple function: `y = sin(2*pi*x) + 0.5*x^2`.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer

# 1. Generate synthetic data
X = torch.linspace(0, 1, 100).unsqueeze(1)
y = torch.sin(2 * torch.pi * X) + 0.5 * X**2 + 0.1 * torch.randn(100, 1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Build the CFN
# We know the underlying function is a sum of a sinusoidal and a quadratic function,
# so we can construct a CFN with these two function nodes in parallel.
node_factory = FunctionNodeFactory()
network = CompositionFunctionNetwork(layers=[
    ParallelCompositionLayer(
        function_nodes=[
            node_factory.create('Sinusoidal', input_dim=1),
            node_factory.create('Polynomial', input_dim=1, degree=2)
        ],
        combination='sum'
    )
])

# 3. Train the model
print("Initial Model:")
print(network.describe())

trainer = Trainer(network, learning_rate=0.01)
trainer.train(loader, epochs=150, loss_fn=nn.MSELoss())

print("\nTrained Model:")
print(network.describe())

# 4. Make predictions
with torch.no_grad():
    predictions = network(X)
    print(f"\nPredictions on the first 5 data points:\n{predictions[:5]}")
```


## Running Examples

You can run the demonstration examples for both the PyTorch and NumPy versions.

**PyTorch Examples:**
```bash
python run_pytorch_examples.py
```

**NumPy Examples:**
```bash
python run_numpy_examples.py
```
Upon running, you will be presented with an interactive menu to select which example to run.

## Model Interpretability

The `cfn_numpy/interpretability.py` module provides functions to analyze trained CFN models. After training a NumPy model, the `interpret_model(network)` function can be called to print a detailed breakdown of the model's structure and learned parameters.

## Citation

If you use this project in your research, please cite our paper:

```bibtex
@misc{li2025compositionalfunctionnetworkshighperformance,
      title={Compositional Function Networks: A High-Performance Alternative to Deep Neural Networks with Built-in Interpretability}, 
      author={Fang Li},
      year={2025},
      eprint={2507.21004},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.21004}, 
}
```

[https://arxiv.org/abs/2507.21004](https://arxiv.org/abs/2507.21004)

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.