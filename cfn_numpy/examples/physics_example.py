import sys
import os


import numpy as np
import matplotlib.pyplot as plt

from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory
from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.Framework import Trainer, mse_loss
from cfn_numpy.interpretability import interpret_model

def physics_example():
    """Physics example: Learning Simple Harmonic Motion (SHM)."""
    print("Running Physics Example: Simple Harmonic Motion")
    
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

    t = np.linspace(0, scaled_t_range, n_samples).reshape(-1, 1) # Scaled time as input
    # Adjust true frequency for the scaled time input
    adjusted_true_frequency = true_frequency / t_scale_factor

    y_true = true_amplitude * np.sin(adjusted_true_frequency * t + true_phase) + np.random.normal(0, 0.1, (n_samples, 1)) # Position as output + noise
    
    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    t_train, t_val = t[:split_idx], t[split_idx:]
    y_train, y_val = y_true[:split_idx], y_true[split_idx:]
    
    # 2. Create CFN for SHM
    node_factory = FunctionNodeFactory()
    layer_factory = CompositionLayerFactory(node_factory)
    
    network = CompositionFunctionNetwork(name="SHMCFN")
    
    # A single SinusoidalFunctionNode to learn A, omega, phi
    # input_dim is 1 because time (t) is a scalar input
    # direction is [1] because it's a scalar input
    network.add_layer(layer_factory.create_sequential([("SinusoidalFunctionNode", {
        "input_dim": 1,
        "amplitude": true_amplitude, # Initial guess, exact true_amplitude
        "frequency": adjusted_true_frequency, # Initial guess, exact true_frequency
        "phase": true_phase,     # Initial guess, exact true_phase
        "direction": np.array([1.0])
    })], name="SHMLayer"))
    
    print(network.describe())
    
    # 3. Train the network
    trainer = Trainer(network, mse_loss, learning_rate=0.001, batch_size=32)
    
    train_losses, val_losses = trainer.train(
        t_train, y_train, t_val, y_val, epochs=5000, 
        verbose=True, early_stopping=True, patience=500
    )
    
    # Plot loss
    trainer.plot_loss()
    
    # 4. Visualize learned function vs true function
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_true, label='True SHM', linestyle='--', color='gray')
    y_pred = network.forward(t)
    plt.plot(t, y_pred, label='Learned SHM', color='blue')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.title('True vs. Learned Simple Harmonic Motion')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('plot.png') 
    
    # 5. Interpret the learned parameters
    print("\n--- Learned SHM Parameters ---")
    interpret_model(network)
    
    # Compare with true parameters
    learned_amplitude = network.layers[0].function_nodes[0].parameters["amplitude"]
    learned_frequency = network.layers[0].function_nodes[0].parameters["frequency"]
    learned_phase = network.layers[0].function_nodes[0].parameters["phase"]
    
    print(f"\nTrue Amplitude: {true_amplitude:.4f}, Learned Amplitude: {learned_amplitude:.4f}")
    print(f"True Frequency: {true_frequency:.4f}, Learned Frequency: {learned_frequency:.4f}")
    print(f"True Phase: {true_phase:.4f}, Learned Phase: {learned_phase:.4f}")
    
    return network
