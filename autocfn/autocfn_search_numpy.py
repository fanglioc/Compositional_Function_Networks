import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
from sklearn.model_selection import train_test_split

from cfn_numpy.FunctionNodes import FunctionNodeFactory
from cfn_numpy.CompositionLayerStructure import CompositionFunctionNetwork, CompositionLayerFactory
from cfn_numpy.Framework import Trainer, mse_loss, binary_cross_entropy_loss, softmax_cross_entropy_loss

# --- Genetic Algorithm Components ---

def genome_to_network(genome, input_dim, output_dim, task):
    """Converts a genome into a NumPy-based CompositionFunctionNetwork."""
    node_factory = FunctionNodeFactory()
    layer_factory = CompositionLayerFactory(node_factory)
    network = CompositionFunctionNetwork()
    current_input_dim = input_dim

    # Define a pool of sensible hidden layer sizes
    hidden_dims = [256, 128, 64, 32]

    for i, (layer_type, node_types, combination_type) in enumerate(genome):
        # This function now assumes MLP-style sequential layers
        if layer_type == 'sequential':
            # The genome specifies ['LinearFunctionNode', 'ReLUFunctionNode']
            # We now define the output dimension of the Linear layer.
            hidden_dim = random.choice(hidden_dims)
            node_specs = [
                ("LinearFunctionNode", {"input_dim": current_input_dim, "output_dim": hidden_dim}),
                ("ReLUFunctionNode", {"input_dim": hidden_dim})
            ]
            layer = layer_factory.create_sequential(node_specs)
            current_input_dim = layer.output_dim
            network.add_layer(layer)
        else:
            # For now, we ignore parallel layers for this structured search
            pass

    # Add a final linear layer to map to the correct output dimension
    final_layer_specs = [("LinearFunctionNode", {"input_dim": current_input_dim, "output_dim": output_dim})]
    # For multi-class, softmax is handled by the loss function, so no final activation here
    
    final_layer = layer_factory.create_sequential(final_layer_specs)
    network.add_layer(final_layer)
    
    return network

def evaluate_fitness(genome, X_train, y_train, X_val, y_val, task, n_classes=None, complexity_penalty_factor=0.01):
    """Evaluates the fitness of a genome using the NumPy trainer."""
    input_dim = X_train.shape[1]
    
    if task == 'classification':
        if n_classes == 2:
            output_dim = 1
            y_train_reshaped = y_train.reshape(-1, 1)
            y_val_reshaped = y_val.reshape(-1, 1)
            loss_fn = binary_cross_entropy_loss
        else:
            output_dim = n_classes
            y_train_reshaped = np.eye(n_classes)[y_train.astype(int)]
            y_val_reshaped = np.eye(n_classes)[y_val.astype(int)]
            loss_fn = softmax_cross_entropy_loss
    else: # regression
        output_dim = 1
        y_train_reshaped = y_train.reshape(-1, 1)
        y_val_reshaped = y_val.reshape(-1, 1)
        loss_fn = mse_loss

    try:
        network = genome_to_network(genome, input_dim, output_dim, task)
    except Exception:
        return float('inf')

    trainer = Trainer(network, loss_fn, learning_rate=0.01, optimizer='adam', grad_clip_norm=1.0)
    train_losses, val_losses = trainer.train(X_train, y_train_reshaped, X_val, y_val_reshaped, epochs=10, verbose=False, early_stopping=True, patience=5)

    complexity_score = sum(1 + len(nodes) for _, nodes, _ in genome)
    
    best_val_loss = min(val_losses) if val_losses else float('inf')
    if np.isnan(best_val_loss) or np.isinf(best_val_loss):
        return float('inf')

    fitness = best_val_loss + complexity_penalty_factor * complexity_score
    return fitness

class AutoCFN_Numpy:
    def __init__(self, X, y, task, population_size=20, generations=10, max_layers=3, max_nodes_per_layer=5):
        self.X = X
        self.y = y
        self.task = task
        self.population_size = population_size
        self.generations = generations
        self.max_layers = max_layers
        self.max_nodes_per_layer = max_nodes_per_layer
        self.population = []
        self.node_pool = list(FunctionNodeFactory._node_types.keys())
        if self.task == 'classification':
            self.n_classes = len(np.unique(self.y))
        else:
            self.n_classes = None

    def _create_random_genome(self):
        """Creates a random genome representing a multi-layer perceptron-style architecture."""
        genome = []
        # Force a sequential structure of Linear -> ReLU layers
        for _ in range(random.randint(1, self.max_layers)):
            # Each "block" in the genome is a Sequential layer with a Linear and a ReLU node
            # The number of nodes (neurons) in the linear layer is randomized.
            # Note: For simplicity in the genome, we don't specify output dims here;
            # the genome_to_network function will handle wiring them up.
            # We are essentially just deciding the number of layers.
            genome.append(('sequential', ['LinearFunctionNode', 'ReLUFunctionNode'], None))
        return genome

    def _mutate(self, genome, mutation_rate=0.2):
        """Mutates a genome by adding or removing a (Linear -> ReLU) layer."""
        mutated_genome = list(genome)
        
        # Add a new (Linear -> ReLU) layer
        if random.random() < mutation_rate and len(mutated_genome) < self.max_layers:
            mutated_genome.insert(random.randint(0, len(mutated_genome)), ('sequential', ['LinearFunctionNode', 'ReLUFunctionNode'], None))

        # Remove a layer
        if random.random() < mutation_rate and len(mutated_genome) > 1:
            mutated_genome.pop(random.randint(0, len(mutated_genome) - 1))
            
        return mutated_genome

    def _select_parents(self, fitness_scores):
        inverted_fitness = [1.0 / (f + 1e-6) for f in fitness_scores]
        total_inverted_fitness = sum(inverted_fitness)
        probabilities = [f / total_inverted_fitness for f in inverted_fitness]
        parents_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
        return [self.population[i] for i in parents_indices]

    def _crossover(self, parent1, parent2):
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1, parent2
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def run_search(self):
        print("--- Starting NumPy AutoCFN Search ---")
        self.population = [self._create_random_genome() for _ in range(self.population_size)]
        best_overall_fitness = float('inf')
        best_overall_genome = None

        for gen in range(self.generations):
            print(f"\n--- Generation {gen + 1}/{self.generations} ---")
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            fitness_scores = [evaluate_fitness(g, X_train, y_train, X_val, y_val, self.task, self.n_classes) for g in self.population]
            
            current_best_idx = np.argmin(fitness_scores)
            if fitness_scores[current_best_idx] < best_overall_fitness:
                best_overall_fitness = fitness_scores[current_best_idx]
                best_overall_genome = self.population[current_best_idx]

            print(f"Best Fitness of Gen {gen + 1}: {fitness_scores[current_best_idx]:.4f} | Overall Best Fitness: {best_overall_fitness:.4f}")
            
            new_population = [best_overall_genome] # Elitism
            parents = self._select_parents(fitness_scores)
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self._crossover(p1, p2)
                new_population.extend([self._mutate(c1), self._mutate(c2)])
            self.population = new_population[:self.population_size]

        print("\n--- AutoCFN Search Finished ---")
        print(f"Overall Best Architecture: {best_overall_genome}")
        print(f"Overall Best Fitness: {best_overall_fitness:.4f}")
        return best_overall_genome
