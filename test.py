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