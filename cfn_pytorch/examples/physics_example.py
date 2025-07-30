

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer
from cfn_numpy.interpretability import interpret_model

def run():
    """Physics example: Learning Simple Harmonic Motion (SHM)."""
    print("--- Running PyTorch Physics Example: Simple Harmonic Motion ---")
    
    # 1. Generate synthetic SHM data
    true_amplitude = 2.0
    true_frequency = 1.5  # angular frequency omega
    true_phase = np.pi / 4
    
    n_samples = 1000
    # Scale time to a more normalized range, e.g., 0 to 2*pi for one cycle
    # Adjust true_frequency accordingly if the original data spanned multiple cycles
    original_t_range = 8 * np.pi
    scaled_t_range = 2 * np.pi
    t_scale_factor = scaled_t_range / original_t_range

    t = torch.linspace(0, scaled_t_range, n_samples).unsqueeze(1) # Scaled time as input
    # Adjust true frequency for the scaled time input
    adjusted_true_frequency = true_frequency / t_scale_factor

    y_true = true_amplitude * torch.sin(adjusted_true_frequency * t + true_phase) + 0.1 * torch.randn(n_samples, 1) # Position as output + noise
    
    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    t_train, t_val = t[:split_idx], t[split_idx:]
    y_train, y_val = y_true[:split_idx], y_true[split_idx:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(t_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(t_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Create CFN for SHM
    node_factory = FunctionNodeFactory()
    
    network = CompositionFunctionNetwork(
        name="SHMCFN",
        layers=[
        # A single SinusoidalFunctionNode to learn A, omega, phi
        # input_dim is 1 because time (t) is a scalar input
        # direction is [1] because it's a scalar input
        SequentialCompositionLayer(
            name="SHMLayer",
            function_nodes=[
            node_factory.create("Sinusoidal", input_dim=1,
                                amplitude=true_amplitude, # Initial guess
                                frequency=adjusted_true_frequency, # Initial guess
                                phase=true_phase,     # Initial guess
                                direction=torch.tensor([1.0]))
        ])
    ])
    
    print("Initial Network Structure:")
    print(network.describe())
    
    # 3. Train the network
    trainer = Trainer(network, learning_rate=0.001)
    
    trainer.train(
        train_loader, val_loader=val_loader, epochs=5000,
        loss_fn=nn.MSELoss(), early_stopping_patience=500
    )
    
    # Plot loss
    trainer.plot_loss('pytorch_physics_loss.png')
    
    # 4. Visualize learned function vs true function
    plt.figure(figsize=(10, 6))
    plt.plot(t.numpy(), y_true.numpy(), label='True SHM', linestyle='--', color='gray')
    network.eval()
    with torch.no_grad():
        y_pred = network(t).numpy()
    plt.plot(t.numpy(), y_pred, label='Learned SHM', color='blue')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.title('PyTorch SHM: True vs. Learned')
    plt.legend()
    plt.grid(True)
    plt.savefig('pytorch_physics_visualization.png')
    print("Plot saved to pytorch_physics_visualization.png")
    plt.close()
    
    # 5. Interpret the learned parameters
    print("\n--- Learned SHM Parameters ---")
    learned_node = network.layers[0].function_nodes[0]
    learned_amplitude = learned_node.amplitude.item()
    learned_frequency = learned_node.frequency.item()
    learned_phase = learned_node.phase.item()
    
    print(f"\nTrue Amplitude: {true_amplitude:.4f}, Learned Amplitude: {learned_amplitude:.4f}")
    print(f"True Original Frequency: {true_frequency:.4f}, Learned Original Frequency: {learned_frequency * t_scale_factor:.4f}")
    print(f"True Phase: {true_phase:.4f}, Learned Phase: {learned_phase:.4f}")

    print("--------------------------------------------------")
    return network

if __name__ == '__main__':
    model = run()
    print("\n--- Full Model Interpretation from Runner ---")
    interpret_model(model)

