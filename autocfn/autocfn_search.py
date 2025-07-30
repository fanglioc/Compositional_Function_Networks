import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from cfn_pytorch.function_nodes import FunctionNodeFactory
from cfn_pytorch.composition_layers import ParallelCompositionLayer, SequentialCompositionLayer, CompositionFunctionNetwork
from cfn_pytorch.trainer import Trainer

# --- Genetic Algorithm Components ---

def genome_to_network(genome, input_dim, output_dim, task):
    """Converts a genome into a CompositionFunctionNetwork."""
    layers = []
    current_input_dim = input_dim
    node_factory = FunctionNodeFactory()

    for i, (layer_type, node_types, combination_type) in enumerate(genome):
        if layer_type == 'parallel':
            nodes_in_layer = []
            for node_type_name in node_types:
                # All nodes in a parallel layer receive the same input_dim
                node = node_factory.create(node_type_name, input_dim=current_input_dim)
                nodes_in_layer.append(node)
            layer = ParallelCompositionLayer(nodes_in_layer, combination=combination_type)
            # Output dimension of a concat parallel layer is the sum of its nodes' output dims
            current_input_dim = layer.output_dim
        
        elif layer_type == 'sequential':
            nodes_in_layer = []
            temp_input_dim = current_input_dim # Input for the first node in sequence
            for node_type_name in node_types:
                node = node_factory.create(node_type_name, input_dim=temp_input_dim)
                nodes_in_layer.append(node)
                temp_input_dim = node.output_dim # Output of current node is input for next
            layer = SequentialCompositionLayer(nodes_in_layer)
            current_input_dim = layer.output_dim # Output of sequential layer is output of its last node

        layers.append(layer)

    # Add a final linear layer to map to the correct output dimension
    # This layer's input_dim is the output_dim of the last layer in the genome
    final_linear_node = node_factory.create('Linear', input_dim=current_input_dim, output_dim=output_dim)
    final_layer_nodes = [final_linear_node]

    if task == 'classification':
        # For binary classification, ensure the final output is sigmoid activated
        if output_dim == 1: 
            final_layer_nodes.append(node_factory.create('Sigmoid', input_dim=output_dim))
        # For multi-class, CrossEntropyLoss expects raw logits, so no final activation here

    layers.append(SequentialCompositionLayer(final_layer_nodes))
    
    return CompositionFunctionNetwork(layers)

def evaluate_fitness(genome, X_train, y_train, X_val, y_val, task, n_classes=None, complexity_penalty_factor=0.01):
    """Evaluates the fitness of a genome (lower is better)."""
    input_dim = X_train.shape[1]
    
    if task == 'classification':
        if n_classes == 2:
            output_dim = 1
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            loss_fn = nn.BCELoss()
        else:
            output_dim = n_classes
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
    else: # regression
        output_dim = 1
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        loss_fn = nn.MSELoss()

    try:
        network = genome_to_network(genome, input_dim, output_dim, task)
    except Exception:
        # Penalize architectures that are invalid
        # print(f"Failed to create network from genome: {e}")
        return float('inf')

    # Simplified training for fitness evaluation
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Debugging: Check initial network output before training
    network.eval()
    with torch.no_grad():
        initial_outputs = network(X_train_tensor)
        if task == 'classification' and output_dim == 1:
            initial_outputs = torch.sigmoid(initial_outputs)
        print(f"DEBUG: Initial network output (min/max/mean): {initial_outputs.min().item():.4f}/{initial_outputs.max().item():.4f}/{initial_outputs.mean().item():.4f}")
        print(f"DEBUG: Initial network output shape: {initial_outputs.shape}")
    network.train() # Set back to train mode

    trainer = Trainer(network, learning_rate=0.01)
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=10, loss_fn=loss_fn)

    # Calculate complexity score
    complexity_score = 0
    for layer_type, node_types, _ in genome:
        complexity_score += 1 # Penalty for each layer
        complexity_score += len(node_types) # Penalty for each node

    # Fitness is the best validation loss + complexity penalty
    fitness = (min(val_losses) if val_losses else float('inf')) + complexity_penalty_factor * complexity_score
    return fitness


class AutoCFN:
    def __init__(self, X, y, task, population_size=20, generations=10, max_layers=3, max_nodes_per_layer=5):
        self.X = X
        self.y = y
        self.task = task
        self.population_size = population_size
        self.generations = generations
        self.max_layers = max_layers
        self.max_nodes_per_layer = max_nodes_per_layer
        self.population = []
        if self.task == 'classification':
            self.n_classes = len(np.unique(self.y))
        else:
            self.n_classes = None

    def _create_random_genome(self):
        """Creates a random genome representing a CFN architecture."""
        genome = []
        for _ in range(random.randint(1, self.max_layers)):
            layer_type = random.choice(['parallel', 'sequential'])
            nodes = []
            for _ in range(random.randint(1, self.max_nodes_per_layer)):
                node_type = random.choice(list(FunctionNodeFactory._node_types.keys()))
                nodes.append(node_type)
            
            combination_type = None
            if layer_type == 'parallel':
                combination_type = random.choice(['sum', 'product', 'concat', 'weighted_sum'])
            
            genome.append((layer_type, nodes, combination_type))
        return genome

    def _select_parents(self, fitness_scores):
        # Simple roulette wheel selection (fitness-proportionate selection)
        # Invert fitness scores because lower loss is better
        inverted_fitness = [1.0 / (f + 1e-6) for f in fitness_scores]
        total_inverted_fitness = sum(inverted_fitness)
        probabilities = [f / total_inverted_fitness for f in inverted_fitness]

        parents_indices = np.random.choice(
            len(self.population), 
            size=self.population_size, # Select as many parents as population size
            p=probabilities
        )
        return [self.population[i] for i in parents_indices]

    def _crossover(self, parent1_genome, parent2_genome):
        # Simple one-point crossover at the layer level
        if len(parent1_genome) < 2 or len(parent2_genome) < 2:
            # If crossover is not possible, return the parents as children
            return parent1_genome, parent2_genome

        crossover_point = random.randint(1, min(len(parent1_genome), len(parent2_genome)) - 1)
        child1_genome = parent1_genome[:crossover_point] + parent2_genome[crossover_point:]
        child2_genome = parent2_genome[:crossover_point] + parent1_genome[crossover_point:]
        return child1_genome, child2_genome

    def _mutate(self, genome, mutation_rate=0.1, param_mutation_rate=0.05, param_mutation_strength=0.1):
        mutated_genome = []
        for layer_type, node_types, combination_type in genome:
            mutated_genome.append([layer_type, list(node_types), combination_type]) # Convert node_types tuple to list

        # Mutate layers (add/remove/change type)
        if random.random() < mutation_rate and len(mutated_genome) < self.max_layers:
            # Add a new random layer
            new_layer_type = random.choice(['parallel', 'sequential'])
            new_nodes = [random.choice(list(FunctionNodeFactory._node_types.keys())) for _ in range(random.randint(1, self.max_nodes_per_layer))]
            new_combination_type = None
            if new_layer_type == 'parallel':
                new_combination_type = random.choice(['sum', 'product', 'concat', 'weighted_sum'])
            mutated_genome.insert(random.randint(0, len(mutated_genome)), [new_layer_type, new_nodes, new_combination_type])
        
        if random.random() < mutation_rate and len(mutated_genome) > 1:
            # Remove a random layer
            mutated_genome.pop(random.randint(0, len(mutated_genome) - 1))

        for i in range(len(mutated_genome)):
            # Mutate layer type
            if random.random() < mutation_rate:
                mutated_genome[i][0] = random.choice(['parallel', 'sequential'])

                # If layer type changed, update combination_type accordingly
                if new_layer_type == 'parallel':
                    mutated_genome[i][2] = random.choice(['sum', 'product', 'concat', 'weighted_sum'])
                else:
                    mutated_genome[i][2] = None
            
            # Mutate combination type for parallel layers (if not already set by layer type mutation)
            elif mutated_genome[i][0] == 'parallel' and random.random() < mutation_rate:
                mutated_genome[i][2] = random.choice(['sum', 'product', 'concat', 'weighted_sum'])

            # Mutate nodes within a layer (add/remove/change type)
            if random.random() < mutation_rate and len(mutated_genome[i][1]) < self.max_nodes_per_layer:
                # Add a new random node
                new_node_type = random.choice(list(FunctionNodeFactory._node_types.keys()))
                mutated_genome[i][1].insert(random.randint(0, len(mutated_genome[i][1])), new_node_type)

            if random.random() < mutation_rate and len(mutated_genome[i][1]) > 1:
                # Remove a random node
                mutated_genome[i][1].pop(random.randint(0, len(mutated_genome[i][1]) - 1))

            for j in range(len(mutated_genome[i][1])):
                if random.random() < mutation_rate:
                    # Mutate node type
                    mutated_genome[i][1][j] = random.choice(list(FunctionNodeFactory._node_types.keys()))

        return [tuple(layer) for layer in mutated_genome]

    def run_search(self):
        print("--- Starting AutoCFN Search ---")
        # 1. Initialize population
        for _ in range(self.population_size):
            self.population.append(self._create_random_genome())

        # 2. Evolutionary loop
        best_overall_fitness = float('inf')
        best_overall_genome = None

        for gen in range(self.generations):
            print(f"\n--- Generation {gen + 1}/{self.generations} ---")
            
            # Evaluate fitness of the population
            fitness_scores = []
            for i, genome in enumerate(self.population):
                # print(f"Evaluating genome {i+1}/{self.population_size}...")
                X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                fitness = evaluate_fitness(genome, X_train, y_train, X_val, y_val, self.task, self.n_classes)
                fitness_scores.append(fitness)
                # print(f"Genome {i+1} Fitness (Val Loss): {fitness:.4f}")

            # Update best overall genome
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_genome = self.population[current_best_idx]

            if current_best_fitness < best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_genome = current_best_genome

            print(f"Best Genome of Generation {gen + 1}: {current_best_genome}")
            print(f"Best Fitness of Generation {gen + 1}: {current_best_fitness:.4f}")
            print(f"Overall Best Fitness: {best_overall_fitness:.4f}")

            # Create next generation
            new_population = []
            # Elitism: Keep the best genome
            new_population.append(current_best_genome)

            # Selection, Crossover, Mutation
            parents = self._select_parents(fitness_scores)
            for i in range(0, self.population_size - 1, 2):
                if i + 1 < len(parents):
                    p1, p2 = parents[i], parents[i+1]
                    child1, child2 = self._crossover(p1, p2)
                    new_population.append(self._mutate(child1))
                    new_population.append(self._mutate(child2))
                else:
                    new_population.append(self._mutate(parents[i]))
            
            self.population = new_population[:self.population_size] # Trim to population size

        print("\n--- AutoCFN Search Finished ---")
        print(f"Overall Best Architecture: {best_overall_genome}")
        print(f"Overall Best Fitness: {best_overall_fitness:.4f}")
        return best_overall_genome