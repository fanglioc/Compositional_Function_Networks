import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import copy
from cfn_numpy.CompositionLayerStructure import CompositionLayerFactory
from cfn_numpy.FunctionNodes import FunctionNodeFactory

class Trainer:
    """Training utilities for Compositional Function Networks."""
    
    def __init__(self, network, loss_fn, learning_rate=0.01, batch_size=32, grad_clip_norm=None, optimizer='adam', beta1=0.9, beta2=0.999, epsilon=1e-8, l2_lambda=0.0):
        """
        Initialize trainer.
        
        Args:
            network: CompositionFunctionNetwork to train
            loss_fn: Loss function to use
            learning_rate: Learning rate for parameter updates
            batch_size: Batch size for mini-batch training
            grad_clip_norm: Optional. If provided, gradients will be clipped to this L2 norm.
            optimizer: 'sgd' or 'adam'.
            beta1, beta2, epsilon: Adam parameters.
            l2_lambda: L2 regularization strength.
        """
        self.network = network
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []
        self.grad_clip_norm = grad_clip_norm
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        
        if self.optimizer == 'adam':
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = None
            self.v = None
            self.t = 0

    def _init_adam_states(self):
        """Initializes the moment vectors for Adam."""
        self.m = copy.deepcopy(self.network.serialize()['layers'])
        self.v = copy.deepcopy(self.network.serialize()['layers'])
        for layer_state in self.m:
            nodes_to_process = []
            if 'nodes' in layer_state:
                nodes_to_process.extend(layer_state['nodes'])
            if 'condition_nodes' in layer_state:
                nodes_to_process.extend(layer_state['condition_nodes'])
            if 'function_nodes' in layer_state:
                nodes_to_process.extend(layer_state['function_nodes'])

            for node in nodes_to_process:
                if 'parameters' in node:
                    for p_name in node['parameters']:
                        node['parameters'][p_name] = np.zeros_like(node['parameters'][p_name])

        for layer_state in self.v:
            nodes_to_process = []
            if 'nodes' in layer_state:
                nodes_to_process.extend(layer_state['nodes'])
            if 'condition_nodes' in layer_state:
                nodes_to_process.extend(layer_state['condition_nodes'])
            if 'function_nodes' in layer_state:
                nodes_to_process.extend(layer_state['function_nodes'])
                
            for node in nodes_to_process:
                if 'parameters' in node:
                    for p_name in node['parameters']:
                        node['parameters'][p_name] = np.zeros_like(node['parameters'][p_name])

    def _clip_gradients(self, gradients):
        """Clips gradients by their L2 norm."""
        if self.grad_clip_norm is None:
            return gradients
        total_norm = 0
        for layer_grads in gradients.values():
            for node_grads in layer_grads.values():
                for grad in node_grads.values():
                    if isinstance(grad, np.ndarray):
                        total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        clip_factor = self.grad_clip_norm / (total_norm + 1e-6)
        if clip_factor < 1:
            for layer_name, layer_grads in gradients.items():
                for node_name, node_grads in layer_grads.items():
                    for param_name, grad in node_grads.items():
                        if isinstance(grad, np.ndarray):
                            gradients[layer_name][node_name][param_name] = grad * clip_factor
        return gradients

    def _update_parameters(self, gradients):
        """Update parameters based on the chosen optimizer."""
        if self.optimizer == 'sgd':
            self.network.update_parameters(gradients, self.learning_rate)
        elif self.optimizer == 'adam':
            if self.m is None or self.v is None:
                self._init_adam_states()
            
            self.t += 1
            
            for i, layer in enumerate(self.network.layers):
                layer_name = f"{layer.name}_{i}"
                if layer_name not in gradients: continue

                nodes_and_grads = []
                if hasattr(layer, 'function_nodes'):
                    for j, node in enumerate(layer.function_nodes):
                        node_name = f"{node.__class__.__name__}_{j}"
                        if node_name in gradients[layer_name]:
                            nodes_and_grads.append((node, gradients[layer_name][node_name], self.m[i]['nodes'][j], self.v[i]['nodes'][j]))
                
                if hasattr(layer, 'condition_nodes'):
                    for j, node in enumerate(layer.condition_nodes):
                        node_name = f"condition_{node.__class__.__name__}_{j}"
                        if node_name in gradients[layer_name]:
                            nodes_and_grads.append((node, gradients[layer_name][node_name], self.m[i]['condition_nodes'][j], self.v[i]['condition_nodes'][j]))

                for node, node_grads, m_node, v_node in nodes_and_grads:
                    for p_name, grad in node_grads.items():
                        if p_name not in node.parameters: continue
                        
                        # Adam update
                        m_node['parameters'][p_name] = self.beta1 * m_node['parameters'][p_name] + (1 - self.beta1) * grad
                        v_node['parameters'][p_name] = self.beta2 * v_node['parameters'][p_name] + (1 - self.beta2) * (grad**2)
                        
                        m_hat = m_node['parameters'][p_name] / (1 - self.beta1**self.t)
                        v_hat = v_node['parameters'][p_name] / (1 - self.beta2**self.t)
                        
                        update_val = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        node.parameters[p_name] -= update_val

    def train_epoch(self, X_train, y_train):
        """Train for one epoch."""
        n_samples = X_train.shape[0]
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]
            
            y_pred = self.network.forward(X_batch)
            loss, grad_output = self.loss_fn(y_pred, y_batch)
            
            # L2 Regularization
            if self.l2_lambda > 0:
                l2_penalty = 0
                for layer in self.network.layers:
                    for node in layer.function_nodes:
                        for param in node.parameters.values():
                            l2_penalty += np.sum(param**2)
                loss += self.l2_lambda * l2_penalty

            gradients = self.network.backward(grad_output)

            # Add L2 gradient penalty
            if self.l2_lambda > 0:
                for i, layer in enumerate(self.network.layers):
                    layer_name = f"{layer.name}_{i}"
                    for j, node in enumerate(layer.function_nodes):
                        node_name = f"{node.__class__.__name__}_{j}"
                        for p_name, param in node.parameters.items():
                            if layer_name in gradients and node_name in gradients[layer_name] and p_name in gradients[layer_name][node_name]:
                                gradients[layer_name][node_name][p_name] += self.l2_lambda * 2 * param

            gradients = self._clip_gradients(gradients)
            self._update_parameters(gradients)
            
            total_loss += loss
        
        return total_loss / (n_samples / self.batch_size)
    
    def evaluate(self, X, y):
        """Evaluate the network."""
        n_samples = X.shape[0]
        total_loss = 0.0
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_batch, y_batch = X[start_idx:end_idx], y[start_idx:end_idx]
            y_pred = self.network.forward(X_batch)
            loss, _ = self.loss_fn(y_pred, y_batch)
            total_loss += loss
            
        return total_loss / (n_samples / self.batch_size)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, verbose=True, early_stopping=False, patience=10,
              lr_decay_factor=0.1, lr_decay_step=50):
        """
        Train the network.
        
        Args:
            lr_decay_factor: Factor to multiply learning rate by.
            lr_decay_step: Number of epochs between learning rate decays.
        """
        best_val_loss = float('inf')
        best_epoch = 0
        no_improvement = 0
        initial_lr = self.learning_rate
        best_model_state = None
        
        for epoch in range(epochs):
            # Learning rate decay
            if (epoch > 0) and (epoch % lr_decay_step == 0):
                self.learning_rate *= lr_decay_factor
                print(f"\nLearning rate decayed to {self.learning_rate:.6f}")

            start_time = time()
            train_loss = self.train_epoch(X_train, y_train)
            self.train_losses.append(train_loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_improvement = 0
                    best_model_state = copy.deepcopy(self.network.serialize())
                else:
                    no_improvement += 1
            
            if verbose:
                epoch_time = time() - start_time
                log_msg = f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.4f}"
                if X_val is not None and y_val is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                print(log_msg)
            
            if early_stopping and no_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                break
        
        if best_model_state is not None:
            print("Restoring best model weights.")
            factory = CompositionLayerFactory(FunctionNodeFactory())
            self.network.deserialize(best_model_state, factory)

        self.learning_rate = initial_lr # Reset for subsequent runs
        return self.train_losses, self.val_losses
    
    def plot_loss(self, filename='loss_plot.png'):
        """Plot the training and validation loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        print(f"Loss plot saved to {filename}")
        plt.close()

def mse_loss(y_pred, y_true):
    """Mean squared error loss."""
    loss = np.mean((y_pred - y_true)**2)
    grad = 2 * (y_pred - y_true) / y_true.size
    return loss, grad

def softmax_cross_entropy_loss(predictions, targets):
    """Softmax activation followed by cross-entropy loss."""
    exp_x = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    n_samples = predictions.shape[0]
    log_likelihood = -np.sum(targets * np.log(probs + 1e-10)) / n_samples
    grad_output = (probs - targets) / n_samples
    return log_likelihood, grad_output

def binary_cross_entropy_loss(predictions, targets):
    """Binary cross entropy loss function."""
    eps = 1e-10
    predictions = np.clip(predictions, eps, 1 - eps)
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    grad_output = -(targets / predictions - (1 - targets) / (1 - predictions)) / predictions.shape[0]
    return loss, grad_output
