import numpy as np
import inspect


class CompositionLayer:
    """Base class for all composition layers in the CFN."""
    
    def __init__(self, name=None):
        """
        Initialize a composition layer.
        
        Args:
            name: Optional name for this layer
        """
        self.name = name or self.__class__.__name__
        self.trainable = True
    
    def forward(self, x):
        """
        Apply the composition to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of the loss with respect to this layer's output
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient with respect to the input
            - grad_params: Dictionary mapping parameter names to their gradients
        """
        raise NotImplementedError("Subclasses must implement backward method")
    
    def get_parameters(self):
        """
        Return all trainable parameters of this layer.
        
        Returns:
            Dictionary mapping parameter names to their current values
        """
        raise NotImplementedError("Subclasses must implement get_parameters method")
    
    def update_parameters(self, updates):
        """
        Update parameters using provided gradients.
        
        Args:
            updates: Dictionary mapping parameter names to update values
        """
        raise NotImplementedError("Subclasses must implement update_parameters method")
    
    def describe(self):
        """
        Return a human-readable description of this layer.
        
        Returns:
            String describing the layer and its components
        """
        raise NotImplementedError("Subclasses must implement describe method")
    
    def serialize(self):
        """
        Serialize the layer for saving/loading.
        
        Returns:
            Dictionary containing all information needed to reconstruct this layer
        """
        raise NotImplementedError("Subclasses must implement serialize method")
    
class SequentialCompositionLayer(CompositionLayer):
    """
    Layer that composes functions sequentially: output = f_n(...f_2(f_1(x))...)
    """
    
    def __init__(self, function_nodes, name=None):
        """
        Initialize a sequential composition layer.
        
        Args:
            function_nodes: List of function nodes to compose sequentially
            name: Optional name for this layer
        """
        super().__init__(name)
        
        # Validate function nodes dimensions
        for i in range(1, len(function_nodes)):
            if function_nodes[i].input_dim != function_nodes[i-1].output_dim:
                raise ValueError(f"Dimension mismatch: Node {i-1} outputs {function_nodes[i-1].output_dim} " +
                                f"dimensions but Node {i} expects {function_nodes[i].input_dim}")
        
        self.function_nodes = function_nodes
        self.input_dim = function_nodes[0].input_dim if function_nodes else 0
        self.output_dim = function_nodes[-1].output_dim if function_nodes else 0
    
    def forward(self, x):
        """
        Apply sequential composition to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Store intermediate activations for backward pass
        self.activations = [x]
        
        # Apply each function sequentially
        current = x
        for node in self.function_nodes:
            current = node.forward(current)
            self.activations.append(current)
        
        return current
    
    def backward(self, grad_output, x=None):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, output_dim)
            x: Input (not needed as we store activations)
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for all nodes
        """
        # Initialize gradients dictionary
        all_gradients = {}
        
        # Backpropagate through each function in reverse order
        current_grad = grad_output
        for i in range(len(self.function_nodes) - 1, -1, -1):
            node = self.function_nodes[i]
            activation = self.activations[i]
            
            # Compute gradients for this node
            grad_input, grad_params = node.backward(current_grad, activation)
            
            # Store gradients for this node
            node_name = f"{node.__class__.__name__}_{i}"
            all_gradients[node_name] = grad_params
            
            # Update gradient for next iteration
            current_grad = grad_input
        
        return current_grad, all_gradients
    
    def get_parameters(self):
        """Return all trainable parameters of this layer."""
        all_params = {}
        
        for i, node in enumerate(self.function_nodes):
            node_name = f"{node.__class__.__name__}_{i}"
            all_params[node_name] = node.get_parameters()
        
        return all_params
    
    def update_parameters(self, updates, learning_rate=0.01):
        """
        Update parameters using provided gradients.
        
        Args:
            updates: Dictionary mapping parameter names to update values
            learning_rate: Learning rate for parameter updates
        """
        if not self.trainable:
            return
            
        for i, node in enumerate(self.function_nodes):
            node_name = f"{node.__class__.__name__}_{i}"
            if node_name in updates:
                node.update_parameters(updates[node_name], learning_rate)
    
    def describe(self):
        """Return human-readable description of this layer."""
        description = f"Sequential Composition ({self.name}):\n"
        for i, node in enumerate(self.function_nodes):
            description += f"  {i+1}. {node.describe()}\n"
        return description
    
    def serialize(self):
        """Serialize the layer for saving/loading."""
        return {
            "type": "SequentialCompositionLayer",
            "name": self.name,
            "trainable": self.trainable,
            "nodes": [node.serialize() for node in self.function_nodes],
            "output_dim": self.output_dim
        }
    
class ParallelCompositionLayer(CompositionLayer):
    """
    Layer that applies multiple functions in parallel and combines their outputs.
    Combination methods: 'sum', 'product', 'concat', 'weighted_sum'
    """
    
    def __init__(self, function_nodes, combination='sum', weights=None, name=None):
        """
        Initialize a parallel composition layer.
        
        Args:
            function_nodes: List of function nodes to apply in parallel
            combination: How to combine outputs ('sum', 'product', 'concat', 'weighted_sum')
            weights: Weights for weighted_sum combination (default: equal weights)
            name: Optional name for this layer
        """
        super().__init__(name)
        
        # Validate function nodes dimensions
        input_dim = function_nodes[0].input_dim
        for i, node in enumerate(function_nodes):
            if node.input_dim != input_dim:
                raise ValueError(f"Input dimension mismatch: Node 0 has {input_dim} " +
                                f"dimensions but Node {i} has {node.input_dim}")
        
        self.function_nodes = function_nodes
        self.combination = combination
        self.input_dim = input_dim
        
        # Determine output dimension based on combination method
        if combination == 'concat':
            self.output_dim = sum(node.output_dim for node in function_nodes)
        else:
            # For sum, product, and weighted_sum, all outputs must have the same dimension
            output_dims = [node.output_dim for node in function_nodes]
            if len(set(output_dims)) > 1:
                raise ValueError(f"For {combination} combination, all nodes must have the same output dimension")
            self.output_dim = output_dims[0]
        
        # Initialize weights for weighted sum
        if combination == 'weighted_sum':
            if weights is None:
                # Equal weights by default
                weights = np.ones(len(function_nodes), dtype=np.float32) / len(function_nodes)
            elif len(weights) != len(function_nodes):
                raise ValueError(f"Number of weights ({len(weights)}) doesn't match " +
                                f"number of nodes ({len(function_nodes)})")
            
            self.parameters = {"weights": np.array(weights, dtype=np.float32)}
        else:
            self.parameters = {}
    
    def forward(self, x):
        """
        Apply parallel composition to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Store input for backward pass
        self.input = x
        
        # Apply each function in parallel
        self.node_outputs = [node.forward(x) for node in self.function_nodes]
        
        # Combine outputs based on specified method
        if self.combination == 'sum':
            return sum(self.node_outputs)
        
        elif self.combination == 'product':
            result = self.node_outputs[0].copy()
            for output in self.node_outputs[1:]:
                result *= output
            return result
        
        elif self.combination == 'concat':
            return np.concatenate(self.node_outputs, axis=1)
        
        elif self.combination == 'weighted_sum':
            weights = self.parameters["weights"]
            return sum(w * output for w, output in zip(weights, self.node_outputs))
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")
    
    def backward(self, grad_output, x=None):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, output_dim)
            x: Input (not needed as we store it)
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for all nodes and combination weights
        """
        x = self.input  # Use stored input
        all_gradients = {}
        
        # Handle different combination methods
        if self.combination == 'sum':
            # For sum, each node gets the same gradient
            node_grads = [grad_output for _ in self.function_nodes]
            
        elif self.combination == 'product':
            # For product, use product rule of differentiation
            node_grads = []
            for i, _ in enumerate(self.function_nodes):
                # Compute product of all outputs except the current one
                other_outputs = self.node_outputs[:i] + self.node_outputs[i+1:]
                product = np.ones_like(grad_output)
                for output in other_outputs:
                    product *= output
                # Gradient for this node
                node_grads.append(grad_output * product)
        
        elif self.combination == 'concat':
            # For concat, split the gradient along the feature dimension
            node_grads = []
            start_idx = 0
            for node in self.function_nodes:
                end_idx = start_idx + node.output_dim
                node_grads.append(grad_output[:, start_idx:end_idx])
                start_idx = end_idx
        
        elif self.combination == 'weighted_sum':
            # For weighted sum, scale gradients by weights
            weights = self.parameters["weights"]
            node_grads = [grad_output * w for w in weights]
            
            # Compute gradients for weights
            weight_grads = np.array([np.sum(grad_output * output) for output in self.node_outputs])
            all_gradients["combination"] = {"weights": weight_grads}
        
        # Compute gradients for each node
        grad_inputs = []
        for i, (node, node_grad) in enumerate(zip(self.function_nodes, node_grads)):
            grad_input, grad_params = node.backward(node_grad, x)
            grad_inputs.append(grad_input)
            
            # Store gradients for this node
            node_name = f"{node.__class__.__name__}_{i}"
            all_gradients[node_name] = grad_params
        
        # Combine input gradients based on combination method
        if self.combination in ['sum', 'weighted_sum', 'concat']:
            final_grad_input = sum(grad_inputs)
        elif self.combination == 'product':
            # For product, add all gradients
            final_grad_input = sum(grad_inputs)
        
        return final_grad_input, all_gradients
    
    def get_parameters(self):
        """Return all trainable parameters of this layer."""
        all_params = {"combination": self.parameters}
        
        for i, node in enumerate(self.function_nodes):
            node_name = f"{node.__class__.__name__}_{i}"
            all_params[node_name] = node.get_parameters()
        
        return all_params
    
    def update_parameters(self, updates, learning_rate=0.01):
        """
        Update parameters using provided gradients.
        
        Args:
            updates: Dictionary mapping parameter names to update values
            learning_rate: Learning rate for parameter updates
        """
        if not self.trainable:
            return
            
        # Update combination parameters (if any)
        if "combination" in updates:
            for param_name, grad in updates["combination"].items():
                if param_name in self.parameters:
                    self.parameters[param_name] -= learning_rate * grad
        
        # Update function node parameters
        for i, node in enumerate(self.function_nodes):
            node_name = f"{node.__class__.__name__}_{i}"
            if node_name in updates:
                node.update_parameters(updates[node_name], learning_rate)
    
    def describe(self):
        """Return human-readable description of this layer."""
        description = f"Parallel Composition ({self.name}, {self.combination}):\n"
        for i, node in enumerate(self.function_nodes):
            if self.combination == 'weighted_sum':
                weight = self.parameters["weights"][i]
                description += f"  {i+1}. [{weight:.3f}] {node.describe()}\n"
            else:
                description += f"  {i+1}. {node.describe()}\n"
        return description
    
    def serialize(self):
        """Serialize the layer for saving/loading."""
        return {
            "type": "ParallelCompositionLayer",
            "name": self.name,
            "trainable": self.trainable,
            "combination": self.combination,
            "parameters": self.parameters,
            "nodes": [node.serialize() for node in self.function_nodes],
            "output_dim": self.output_dim
        }
    
class ConditionalCompositionLayer(CompositionLayer):
    """
    Layer that applies different functions based on conditions.
    f(x) = condition_1(x) * function_1(x) + condition_2(x) * function_2(x) + ...
    """
    
    def __init__(self, condition_nodes, function_nodes, name=None):
        """
        Initialize a conditional composition layer.
        
        Args:
            condition_nodes: List of function nodes that output conditions
            function_nodes: List of function nodes to apply conditionally
            name: Optional name for this layer
        """
        super().__init__(name)
        
        if len(condition_nodes) != len(function_nodes):
            raise ValueError(f"Number of condition nodes ({len(condition_nodes)}) must match " +
                           f"number of function nodes ({len(function_nodes)})")
        
        # Validate dimensions
        input_dim = condition_nodes[0].input_dim
        for i, node in enumerate(condition_nodes + function_nodes):
            if node.input_dim != input_dim:
                raise ValueError(f"Input dimension mismatch: Node 0 has {input_dim} " +
                                f"dimensions but Node {i} has {node.input_dim}")
        
        # All function nodes should have the same output dimension
        output_dims = [node.output_dim for node in function_nodes]
        if len(set(output_dims)) > 1:
            raise ValueError("All function nodes must have the same output dimension")
        
        # All condition nodes should output scalar values
        for i, node in enumerate(condition_nodes):
            if node.output_dim != 1:
                raise ValueError(f"Condition node {i} must have output_dim=1, got {node.output_dim}")
        
        self.condition_nodes = condition_nodes
        self.function_nodes = function_nodes
        self.input_dim = input_dim
        self.output_dim = output_dims[0]
    
    def forward(self, x):
        """
        Apply conditional composition to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Store input for backward pass
        self.input = x
        
        # Compute conditions
        self.condition_outputs = [node.forward(x) for node in self.condition_nodes]
        
        # Normalize conditions to sum to 1 along the sample dimension
        condition_sum = sum(self.condition_outputs)
        self.normalized_conditions = [cond / (condition_sum + 1e-10) for cond in self.condition_outputs]
        
        # Compute function outputs
        self.function_outputs = [node.forward(x) for node in self.function_nodes]
        
        # Weighted combination based on conditions
        result = np.zeros((x.shape[0], self.output_dim), dtype=np.float32)
        for cond, func_out in zip(self.normalized_conditions, self.function_outputs):
            result += cond * func_out
        
        return result
    
    def backward(self, grad_output, x=None):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, output_dim)
            x: Input (not needed as we store it)
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for all nodes
        """
        x = self.input  # Use stored input
        all_gradients = {}
        
        # Gradient w.r.t function outputs
        function_grads = [cond * grad_output for cond in self.normalized_conditions]
        
        # Gradient w.r.t conditions (more complex due to normalization)
        condition_sum = sum(self.condition_outputs)
        
        condition_grads = []
        for i, func_out in enumerate(self.function_outputs):
            # Derivative of the weighted sum w.r.t. each condition
            direct_effect = func_out * grad_output
            
            # Sum over output dimension for each sample
            direct_effect_sum = np.sum(direct_effect, axis=1, keepdims=True)
            
            # Effect due to normalization (simplified calculation)
            norm_effect = -1.0 / (condition_sum + 1e-10)**2 * np.sum(
                [cond * np.sum(func_out * grad_output, axis=1, keepdims=True) 
                 for cond, func_out in zip(self.condition_outputs, self.function_outputs)], axis=0)
            
            condition_grads.append(direct_effect_sum + norm_effect)
        
        # Compute gradients for each node
        grad_inputs_condition = []
        grad_inputs_function = []
        
        # Process condition nodes
        for i, (node, node_grad) in enumerate(zip(self.condition_nodes, condition_grads)):
            grad_input, grad_params = node.backward(node_grad, x)
            grad_inputs_condition.append(grad_input)
            
            # Store gradients for this node
            node_name = f"condition_{node.__class__.__name__}_{i}"
            all_gradients[node_name] = grad_params
        
        # Process function nodes
        for i, (node, node_grad) in enumerate(zip(self.function_nodes, function_grads)):
            grad_input, grad_params = node.backward(node_grad, x)
            grad_inputs_function.append(grad_input)
            
            # Store gradients for this node
            node_name = f"function_{node.__class__.__name__}_{i}"
            all_gradients[node_name] = grad_params
        
        # Combine input gradients
        final_grad_input = sum(grad_inputs_condition + grad_inputs_function)
        
        return final_grad_input, all_gradients
    
    def get_parameters(self):
        """Return all trainable parameters of this layer."""
        all_params = {}
        
        # Parameters from condition nodes
        for i, node in enumerate(self.condition_nodes):
            node_name = f"condition_{node.__class__.__name__}_{i}"
            all_params[node_name] = node.get_parameters()
        
        # Parameters from function nodes
        for i, node in enumerate(self.function_nodes):
            node_name = f"function_{node.__class__.__name__}_{i}"
            all_params[node_name] = node.get_parameters()
        
        return all_params
    
    def update_parameters(self, updates, learning_rate=0.01):
        """
        Update parameters using provided gradients.
        
        Args:
            updates: Dictionary mapping parameter names to update values
            learning_rate: Learning rate for parameter updates
        """
        if not self.trainable:
            return
            
        # Update condition node parameters
        for i, node in enumerate(self.condition_nodes):
            node_name = f"condition_{node.__class__.__name__}_{i}"
            if node_name in updates:
                node.update_parameters(updates[node_name], learning_rate)
        
        # Update function node parameters
        for i, node in enumerate(self.function_nodes):
            node_name = f"function_{node.__class__.__name__}_{i}"
            if node_name in updates:
                # Check if the update is for a nested layer or a function node
                if isinstance(updates[node_name], dict) and "combination" in updates[node_name]: # Heuristic for nested layer
                    # It's a nested layer, pass the learning rate down
                    node.update_parameters(updates[node_name], learning_rate)
                else:
                    # It's a function node, apply learning rate directly
                    node.update_parameters(updates[node_name], learning_rate)
    
    def describe(self):
        """Return human-readable description of this layer."""
        description = f"Conditional Composition ({self.name}):\n"
        for i, (cond_node, func_node) in enumerate(zip(self.condition_nodes, self.function_nodes)):
            description += f"  Region {i+1}:\n"
            description += f"    Condition: {cond_node.describe()}\n"
            description += f"    Function: {func_node.describe()}\n"
        return description
    
    def serialize(self):
        """Serialize the layer for saving/loading."""
        
        def serialize_item(item):
            if isinstance(item, CompositionLayer):
                return {"type": "layer", "data": item.serialize()}
            else:
                return item.serialize()

        return {
            "type": "ConditionalCompositionLayer",
            "name": self.name,
            "trainable": self.trainable,
            "condition_nodes": [node.serialize() for node in self.condition_nodes],
            "function_nodes": [serialize_item(node) for node in self.function_nodes]
        }
    
class CompositionFunctionNetwork:
    """
    Complete function network combining multiple composition layers.
    """
    
    def __init__(self, layers=None, name="CFN"):
        """
        Initialize a compositional function network.
        
        Args:
            layers: List of composition layers
            name: Name for this network
        """
        self.layers = layers or []
        self.name = name
        
        # Validate layer dimensions if any layers are provided
        if layers:
            for i in range(1, len(layers)):
                if layers[i].input_dim != layers[i-1].output_dim:
                    raise ValueError(f"Dimension mismatch: Layer {i-1} outputs {layers[i-1].output_dim} " +
                                   f"dimensions but Layer {i} expects {layers[i].input_dim}")
    
    def add_layer(self, layer):
        """
        Add a layer to the network.
        
        Args:
            layer: CompositionLayer to add
        """
        if self.layers and layer.input_dim != self.layers[-1].output_dim:
            raise ValueError(f"Dimension mismatch: Last layer outputs {self.layers[-1].output_dim} " +
                           f"dimensions but new layer expects {layer.input_dim}")
        
        self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Store activations for backward pass
        self.activations = [x]
        
        # Forward through each layer
        current = x
        for layer in self.layers:
            current = layer.forward(current)
            self.activations.append(current)
        
        return current
    
    def backward(self, grad_output):
        """
        Backward pass through the network.
        
        Args:
            grad_output: Gradient of loss with respect to network output
            
        Returns:
            Dictionary of gradients for all parameters
        """
        all_gradients = {}
        
        # Backpropagate through each layer in reverse order
        current_grad = grad_output
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            activation = self.activations[i]
            
            # Compute gradients for this layer
            if len(inspect.signature(layer.backward).parameters) == 2:
                grad_input, layer_grads = layer.backward(current_grad)
            else:
                grad_input, layer_grads = layer.backward(current_grad, activation)
            
            # Store gradients for this layer
            layer_name = f"{layer.name}_{i}"
            all_gradients[layer_name] = layer_grads
            
            # Update gradient for next iteration
            current_grad = grad_input
        
        return all_gradients
    
    def update_parameters(self, gradients, learning_rate=0.01):
        """
        Update all network parameters using computed gradients.
        
        Args:
            gradients: Dictionary of gradients returned by backward method
            learning_rate: Learning rate for parameter updates
        """
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.name}_{i}"
            if layer_name in gradients:
                layer.update_parameters(gradients[layer_name], learning_rate)
    
    def train_step(self, x, y, loss_fn, learning_rate=0.01):
        """
        Perform a single training step.
        
        Args:
            x: Input data batch
            y: Target data batch
            loss_fn: Loss function that returns loss and gradient
            learning_rate: Learning rate for parameter updates
            
        Returns:
            Loss value
        """
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss and initial gradient
        loss, grad_output = loss_fn(y_pred, y)
        
        # Backward pass
        gradients = self.backward(grad_output)
        
        # Update parameters
        self.update_parameters(gradients, learning_rate)
        
        return loss
    
    def describe(self):
        """Return human-readable description of the network."""
        description = f"Compositional Function Network ({self.name}):\n"
        for i, layer in enumerate(self.layers):
            description += f"Layer {i+1}: {layer.describe()}\n"
        return description
    
    def serialize(self):
        """Serialize the network for saving/loading."""
        return {
            "name": self.name,
            "layers": [layer.serialize() for layer in self.layers]
        }
    
    def deserialize(self, data, factory=None):
        """
        Create a network from serialized data.
        
        Args:
            data: Dictionary containing serialized network data
            factory: Factory instance for creating function nodes
            
        Returns:
            Instantiated CompositionFunctionNetwork
        """
        self.name = data["name"]
        self.layers = []
        if 'layers' in data:
            if not factory:
                raise ValueError("A CompositionLayerFactory is required to deserialize a network.")
            for layer_data in data['layers']:
                self.add_layer(factory.deserialize(layer_data))

class CompositionLayerFactory:
    """Factory class for creating and deserializing composition layers."""
    
    # Registry of layer types
    _layer_types = {
        "SequentialCompositionLayer": SequentialCompositionLayer,
        "ParallelCompositionLayer": ParallelCompositionLayer,
        "ConditionalCompositionLayer": ConditionalCompositionLayer,
    }
    
    def __init__(self, function_node_factory):
        """
        Initialize the layer factory.
        
        Args:
            function_node_factory: FunctionNodeFactory instance for creating nodes
        """
        self.function_node_factory = function_node_factory
    
    def create_sequential(self, function_node_specs, name=None):
        """
        Create a sequential composition layer.
        
        Args:
            function_node_specs: List of (node_type, params_dict) tuples
            name: Optional name for the layer
            
        Returns:
            Instantiated SequentialCompositionLayer
        """
        nodes = []
        for node_type, params in function_node_specs:
            nodes.append(self.function_node_factory.create(node_type, **params))
        
        return SequentialCompositionLayer(nodes, name=name)
    
    def create_parallel(self, function_node_specs, combination='sum', weights=None, name=None):
        """
        Create a parallel composition layer.
        
        Args:
            function_node_specs: List of (node_type, params_dict) tuples
            combination: How to combine outputs ('sum', 'product', 'concat', 'weighted_sum')
            weights: Weights for weighted_sum combination
            name: Optional name for the layer
            
        Returns:
            Instantiated ParallelCompositionLayer
        """
        nodes = []
        for node_type, params in function_node_specs:
            nodes.append(self.function_node_factory.create(node_type, **params))
        
        return ParallelCompositionLayer(nodes, combination=combination, weights=weights, name=name)
    
    def create_conditional(self, condition_node_specs, function_node_specs, name=None):
        """
        Create a conditional composition layer.
        
        Args:
            condition_node_specs: List of (node_type, params_dict) tuples for conditions
            function_node_specs: List of (node_type, params_dict) tuples for functions
            name: Optional name for the layer
            
        Returns:
            Instantiated ConditionalCompositionLayer
        """
        condition_nodes = []
        for node_type, params in condition_node_specs:
            condition_nodes.append(self.function_node_factory.create(node_type, **params))
        
        function_nodes = []
        for spec in function_node_specs:
            # Check if the spec is for a FunctionNode or a CompositionLayer
            if isinstance(spec, tuple) and len(spec) == 2:
                node_or_layer_type, params = spec
                if node_or_layer_type in self.function_node_factory._node_types:
                    # It's a FunctionNode
                    function_nodes.append(self.function_node_factory.create(node_or_layer_type, **params))
                elif node_or_layer_type in self._layer_types:
                    # It's a CompositionLayer, so we need to create it using the layer factory itself
                    if node_or_layer_type == "SequentialCompositionLayer":
                        function_nodes.append(self.create_sequential(**params))
                    elif node_or_layer_type == "ParallelCompositionLayer":
                        function_nodes.append(self.create_parallel(**params))
                    elif node_or_layer_type == "ConditionalCompositionLayer":
                        raise ValueError("Nested ConditionalCompositionLayer not supported directly in function_node_specs for now.")
                    else:
                        raise ValueError(f"Unknown layer type in function_node_specs: {node_or_layer_type}")
                else:
                    raise ValueError(f"Unknown type in function_node_specs: {node_or_layer_type}")
            else:
                raise ValueError("Invalid spec format in function_node_specs. Expected (type, params_dict) tuple.")
        
        return ConditionalCompositionLayer(condition_nodes, function_nodes, name=name)
    
    def deserialize(self, data):
        """
        Create a composition layer from serialized data.
        
        Args:
            data: Dictionary containing serialized layer data
            
        Returns:
            Instantiated composition layer
        """
        layer_type_name = data["type"]
        if layer_type_name not in self._layer_types:
            raise ValueError(f"Unknown layer type: {layer_type_name}")
        
        layer_class = self._layer_types[layer_type_name]
        
        # Deserialize nodes
        nodes = [self.function_node_factory.deserialize(node_data) for node_data in data.get("nodes", [])]
        
        if layer_class == SequentialCompositionLayer:
            layer = SequentialCompositionLayer(nodes, name=data["name"])
            if "output_dim" in data:
                layer.output_dim = data["output_dim"]
        elif layer_class == ParallelCompositionLayer:
            layer = ParallelCompositionLayer(nodes, combination=data["combination"], name=data["name"])
            if "parameters" in data:
                layer.parameters = data["parameters"]
            if "output_dim" in data:
                layer.output_dim = data["output_dim"]
        elif layer_class == ConditionalCompositionLayer:
            condition_nodes = [self.function_node_factory.deserialize(node_data) for node_data in data.get("condition_nodes", [])]
            
            function_nodes = []
            for item_data in data.get("function_nodes", []):
                if item_data['type'] == 'layer':
                    function_nodes.append(self.deserialize(item_data['data']))
                else:
                    function_nodes.append(self.function_node_factory.deserialize(item_data))

            layer = ConditionalCompositionLayer(condition_nodes, function_nodes, name=data["name"])
        else:
            # Fallback for simple layers if any are added in the future
            layer = layer_class(name=data["name"])

        layer.trainable = data.get("trainable", True)
        return layer

def mse_loss(predictions, targets):
    """
    Mean squared error loss function.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        - loss: Scalar loss value
        - grad_output: Gradient of loss with respect to predictions
    """
    # Compute loss
    diff = predictions - targets
    loss = np.mean(np.sum(diff**2, axis=1))
    
    # Compute gradient
    grad_output = 2 * diff / (diff.shape[0] * diff.shape[1])
    
    return loss, grad_output

def binary_cross_entropy_loss(predictions, targets):
    """
    Binary cross entropy loss function.
    
    Args:
        predictions: Predicted probabilities
        targets: Binary target values
        
    Returns:
        - loss: Scalar loss value
        - grad_output: Gradient of loss with respect to predictions
    """
    # Clip predictions to avoid numerical issues
    eps = 1e-10
    predictions = np.clip(predictions, eps, 1 - eps)
    
    # Compute loss
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # Compute gradient
    grad_output = -(targets / predictions - (1 - targets) / (1 - predictions)) / predictions.shape[0]
    
    return loss, grad_output

def create_example_network():
    """Create an example network for demonstration."""
    # Create function node factory
    node_factory = FunctionNodeFactory()
    
    # Create layer factory
    layer_factory = CompositionLayerFactory(node_factory)
    
    # Create a simple network for a regression task
    network = CompositionFunctionNetwork(name="ExampleCFN")
    
    # First layer: Parallel composition of different function types
    input_dim = 2
    parallel_layer = layer_factory.create_parallel([
        ("GaussianFunctionNode", {"input_dim": input_dim, "width": 0.5}),
        ("SigmoidFunctionNode", {"input_dim": input_dim}),
        ("SinusoidalFunctionNode", {"input_dim": input_dim, "frequency": 2.0})
    ], combination='weighted_sum', name="InputLayer")
    
    network.add_layer(parallel_layer)
    
    # Second layer: Sequential composition for transformation
    sequential_layer = layer_factory.create_sequential([
        ("LinearFunctionNode", {"input_dim": 1, "output_dim": 3}),
        ("PolynomialFunctionNode", {"input_dim": 3, "degree": 2})
    ], name="TransformationLayer")
    
    network.add_layer(sequential_layer)
    
    # Third layer: Conditional composition for final output
    conditional_layer = layer_factory.create_conditional(
        [("StepFunctionNode", {"input_dim": 1, "smoothing": 0.1}),
         ("StepFunctionNode", {"input_dim": 1, "bias": 0.5, "smoothing": 0.1})],
        [("LinearFunctionNode", {"input_dim": 1, "output_dim": 1}),
         ("ExponentialFunctionNode", {"input_dim": 1, "rate": 0.5})],
        name="OutputLayer"
    )
    
    network.add_layer(conditional_layer)
    
    return network
