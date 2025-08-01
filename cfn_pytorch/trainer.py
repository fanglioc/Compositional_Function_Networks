
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

class Trainer:
    """Training utility for PyTorch-based Compositional Function Networks."""

    def __init__(self, network, optimizer=None, scheduler=None, learning_rate=0.01, grad_clip_norm=None, weight_decay=0.0, device='cpu'):
        self.network = network
        self.learning_rate = learning_rate
        if optimizer is None:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = [] # Added for validation accuracy
        self.device = device
        self.network.to(self.device)

    def train(self, train_loader, val_loader=None, epochs=100, loss_fn=nn.MSELoss(), early_stopping_patience=None, lr_decay_step=None, lr_decay_gamma=0.1, metric_fn=None):
        """
        Train the network.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set (optional).
            epochs: Number of epochs to train.
            loss_fn: The loss function to use.
            early_stopping_patience: Number of epochs to wait before stopping if validation loss doesn't improve.
                                     If None, early stopping is disabled.
            metric_fn: Optional function to calculate a metric (e.g., accuracy) on the validation set.
        """
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        # Initialize learning rate scheduler if decay step is provided
        scheduler = None
        if lr_decay_step is not None:
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

        for epoch in range(epochs):
            self.network.train() # Set the model to training mode
            running_loss = 0.0
            
            # Training loop
            for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                
                # Flatten the input if it's a 4D tensor (e.g., images)
                if inputs.dim() == 4:
                    inputs = inputs.view(inputs.size(0), -1)

                outputs = self.network(inputs)
                loss = loss_fn(outputs, targets)
                
                loss.backward()
                
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
                
                self.optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            self.train_losses.append(epoch_loss)
            
            # Step the scheduler after each epoch
            if scheduler:
                scheduler.step()

            # Validation loop
            if val_loader:
                self.network.eval() # Set the model to evaluation mode
                val_loss = 0.0
                val_metric = 0.0
                total_samples = 0
                
                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        # Flatten the input if it's a 4D tensor (e.g., images)
                        if inputs.dim() == 4:
                            inputs = inputs.view(inputs.size(0), -1)
                        outputs = self.network(inputs)
                        loss = loss_fn(outputs, targets)
                        val_loss += loss.item()

                        if metric_fn:
                            val_metric += metric_fn(outputs, targets)
                            total_samples += targets.size(0)
                
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

                log_string = f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                if metric_fn and total_samples > 0:
                    val_metric_avg = (val_metric / total_samples) * 100 # Assuming metric_fn returns sum of correct predictions
                    self.val_accuracies.append(val_metric_avg)
                    log_string += f", Val Acc: {val_metric_avg:.2f} %"
                print(log_string)

                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        # Save the state of the best model
                        best_model_state = copy.deepcopy(self.network.state_dict())
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {early_stopping_patience} epochs.")
                            break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

        # Restore the best model weights if early stopping was used
        if best_model_state is not None:
            print("Restoring best model weights.")
            self.network.load_state_dict(best_model_state)

        return self.train_losses, self.val_losses

    def plot_loss(self, filename='loss_plot.png'):
        """
        Plot the training and validation loss curves.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        # Save the plot instead of showing it to avoid blocking in non-interactive environments
        plt.savefig(filename)
        print(f"Loss plot saved to {filename}")
        plt.close()

    def plot_accuracy(self, filename='accuracy_plot.png'):
        """
        Plot the validation accuracy curve.
        """
        if self.val_accuracies:
            plt.figure(figsize=(10, 6))
            plt.plot(self.val_accuracies, label='Validation Accuracy', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(filename)
            print(f"Accuracy plot saved to {filename}")
            plt.close()
        else:
            print("No validation accuracy data to plot.")
