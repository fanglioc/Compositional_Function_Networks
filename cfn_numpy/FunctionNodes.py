import numpy as np

class FunctionNode:
    """Base class for all function nodes in our compositional network."""
    
    def __init__(self, input_dim, trainable=True):
        """
        Initialize a function node.
        
        Args:
            input_dim: Dimensionality of the input to this function
            trainable: Whether the function parameters can be trained
        """
        self.input_dim = input_dim
        self.trainable = trainable
        self.parameters = {}  # Dictionary to store all trainable parameters
        
    def forward(self, x):
        """
        Apply the function to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def backward(self, grad_output):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of the loss with respect to this node's output
            
        Returns:
            - grad_input: Gradient with respect to the input
            - grad_params: Dictionary mapping parameter names to their gradients
        """
        raise NotImplementedError("Subclasses must implement backward method")
    
    def get_parameters(self):
        """
        Return all trainable parameters of this function node.
        
        Returns:
            Dictionary mapping parameter names to their current values
        """
        return self.parameters
    
    def update_parameters(self, updates, learning_rate=0.01):
        """
        Update parameters using provided gradients.
        
        Args:
            updates: Dictionary mapping parameter names to update values
            learning_rate: Learning rate for parameter updates
        """
        if not self.trainable:
            return
            
        for name, update in updates.items():
            if name in self.parameters:
                self.parameters[name] -= learning_rate * update
                # Enforce positive amplitude and frequency
                if name == "amplitude":
                    self.parameters[name] = np.maximum(1e-6, self.parameters[name])
                if name == "frequency":
                    self.parameters[name] = np.maximum(1e-6, self.parameters[name])
    
    def describe(self):
        """
        Return a human-readable description of this function.
        
        Returns:
            String describing the function and its parameters
        """
        raise NotImplementedError("Subclasses must implement describe method")
    
    def serialize(self):
        """
        Serialize the function node for saving/loading.
        
        Returns:
            Dictionary containing all information needed to reconstruct this node
        """
        return {
            "type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "trainable": self.trainable,
            "parameters": self.parameters
        }
    
    @staticmethod
    def deserialize(data):
        """
        Create a function node from serialized data.
        
        Args:
            data: Dictionary containing serialized node data
            
        Returns:
            Instantiated FunctionNode object
        """
        raise NotImplementedError("This should be implemented by a factory")
    
    
class GaussianFunctionNode(FunctionNode):
    """Gaussian radial basis function node: f(x) = exp(-||x - center||²/(2*width²))"""
    
    def __init__(self, input_dim, center=None, width=None, trainable=True):
        """
        Initialize a Gaussian function node.
        
        Args:
            input_dim: Dimensionality of the input
            center: Center of the Gaussian (default: random initialization)
            width: Width parameter controlling spread (default: random initialization)
            trainable: Whether parameters can be updated during training
        """
        super().__init__(input_dim, trainable)
        
        if center is None:
            center = np.random.randn(input_dim)
        if width is None:
            width = np.random.uniform(0.5, 1.5)
        
        self.parameters["center"] = np.array(center, dtype=np.float32)
        self.parameters["width"] = np.float32(width)
        self.output_dim = 1
    
    def forward(self, x):
        """
        Compute Gaussian function value for input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Calculate squared distance from center
        center = self.parameters["center"]
        width = self.parameters["width"]
        
        # Reshape for broadcasting
        center_reshaped = center.reshape(1, -1)
        
        # Calculate squared Euclidean distance
        squared_dist = np.sum(np.square(x - center_reshaped), axis=1, keepdims=True)
        
        # Apply Gaussian function
        return np.exp(-squared_dist / (2 * width * width))
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, 1)
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient w.r.t input, shape (batch_size, input_dim)
            - grad_params: Dictionary with gradients for center and width
        """
        # Get cached forward pass outputs and parameters
        center = self.parameters["center"]
        width = self.parameters["width"]
        
        # Reshape center for broadcasting
        center_reshaped = center.reshape(1, -1)
        
        # Compute intermediate values
        center = self.parameters["center"]
        width = self.parameters["width"]
        
        # Reshape center for broadcasting
        center_reshaped = center.reshape(1, -1)
        
        # Compute intermediate values
        diff = x - center_reshaped
        squared_dist = np.sum(np.square(diff), axis=1, keepdims=True)
        gaussian_output = np.exp(-squared_dist / (2 * width * width))
        
        # Gradient w.r.t input
        grad_input = grad_output * gaussian_output * (-diff / (width * width))
        
        # Gradient w.r.t center
        grad_center = np.sum(grad_output * gaussian_output * diff / (width * width), axis=0)
        
        # Gradient w.r.t width
        grad_width = np.sum(grad_output * gaussian_output * squared_dist / (width * width * width))
        
        return grad_input, {"center": grad_center, "width": grad_width}
    def describe(self):
        """Return human-readable description of this function."""
        return f"Gaussian RBF with center at {self.parameters['center']} and width {self.parameters['width']}"
    


class SigmoidFunctionNode(FunctionNode):
    """
    Sigmoid function node.
    - If direction is provided, it computes: f(x) = 1 / (1 + exp(-s * (x·d + o)))
    - If direction is None, it computes element-wise: f(x) = 1 / (1 + exp(-s * (x + o)))
    """
    
    def __init__(self, input_dim, direction=None, offset=None, steepness=None, trainable=True):
        """
        Initialize a sigmoid function node.
        
        Args:
            input_dim: Dimensionality of the input
            direction: Direction vector for projection. If None, applies element-wise.
            offset: Offset parameter (default: random)
            steepness: Steepness parameter (default: random)
            trainable: Whether parameters can be updated
        """
        super().__init__(input_dim, trainable)
        
        self.is_elementwise = direction is None
        
        if self.is_elementwise:
            if offset is None:
                offset = np.random.uniform(-0.1, 0.1, input_dim)
            if steepness is None:
                steepness = np.random.uniform(0.8, 1.2, input_dim)
            self.parameters["offset"] = np.array(offset, dtype=np.float32)
            self.parameters["steepness"] = np.array(steepness, dtype=np.float32)
            self.output_dim = input_dim
        else:
            if direction is None:
                direction = np.random.randn(input_dim)
            if offset is None:
                offset = np.random.uniform(-0.1, 0.1)
            if steepness is None:
                steepness = np.random.uniform(0.8, 1.2)
            self.parameters["direction"] = np.array(direction, dtype=np.float32) / np.linalg.norm(direction)
            self.parameters["offset"] = np.float32(offset)
            self.parameters["steepness"] = np.float32(steepness)
            self.output_dim = 1
            
    def forward(self, x):
        """Compute sigmoid function value for input x."""
        offset = self.parameters["offset"]
        steepness = self.parameters["steepness"]
        
        if self.is_elementwise:
            # Apply element-wise sigmoid
            return 1.0 / (1.0 + np.exp(-steepness * (x + offset)))
        else:
            # Apply directional sigmoid
            direction = self.parameters["direction"]
            projection = np.dot(x, direction) + offset
            return 1.0 / (1.0 + np.exp(-steepness * projection)).reshape(-1, 1)

    def backward(self, grad_output, x):
        """Compute gradients for backpropagation."""
        offset = self.parameters["offset"]
        steepness = self.parameters["steepness"]
        
        # Compute sigmoid output (forward pass again, or could be cached)
        y = self.forward(x)
        
        # Derivative of sigmoid: y * (1 - y)
        sigmoid_derivative = y * (1 - y)
        
        grad_params = {}
        
        if self.is_elementwise:
            # Element-wise backpropagation
            grad_common = grad_output * sigmoid_derivative * steepness
            
            # Gradient w.r.t input
            grad_input = grad_common
            
            # Gradients for parameters
            grad_params["offset"] = np.sum(grad_common, axis=0)
            grad_params["steepness"] = np.sum(grad_output * sigmoid_derivative * (x + offset), axis=0)

        else:
            # Directional backpropagation
            direction = self.parameters["direction"]
            projection = np.dot(x, direction) + offset
            
            grad_common = (grad_output * sigmoid_derivative * steepness).flatten()

            # Gradient w.r.t input
            grad_input = np.outer(grad_common, direction)
            
            # Gradients for parameters
            grad_params["direction"] = np.dot(grad_common, x)
            grad_params["offset"] = np.sum(grad_common)
            grad_params["steepness"] = np.sum(grad_common * projection)

        return grad_input, grad_params
    
    def describe(self):
        """Return human-readable description of this function."""
        if self.is_elementwise:
            return f"Element-wise Sigmoid (input_dim={self.input_dim})"
        else:
            return (f"Directional Sigmoid with direction={self.parameters['direction']}, "
                    f"offset={self.parameters['offset']}, steepness={self.parameters['steepness']}")
    
class LinearFunctionNode(FunctionNode):
    """Linear function node: f(x) = W·x + b"""
    
    def __init__(self, input_dim, output_dim=1, weights=None, bias=None, trainable=True):
        """
        Initialize a linear function node.
        
        Args:
            input_dim: Dimensionality of the input
            output_dim: Dimensionality of the output (default: 1)
            weights: Weight matrix of shape (input_dim, output_dim) (default: random)
            bias: Bias vector of shape (output_dim,) (default: zeros)
            trainable: Whether parameters can be updated during training
        """
        super().__init__(input_dim, trainable)
        self.output_dim = output_dim
        
        # Initialize weights
        if weights is None:
            # Use He initialization for better training dynamics
            scale = np.sqrt(2.0 / input_dim)
            weights = np.random.randn(input_dim, output_dim) * scale
        
        # Initialize bias
        if bias is None:
            bias = np.zeros(output_dim)
        
        # Store parameters
        self.parameters["weights"] = np.array(weights, dtype=np.float32)
        self.parameters["bias"] = np.array(bias, dtype=np.float32)
    
    def forward(self, x):
        """
        Apply linear transformation to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        weights = self.parameters["weights"]
        bias = self.parameters["bias"]
        
        # Linear transformation
        return np.dot(x, weights) + bias
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, output_dim)
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient w.r.t input, shape (batch_size, input_dim)
            - grad_params: Dictionary with gradients for weights and bias
        """
        weights = self.parameters["weights"]
        
        # Gradient w.r.t input
        grad_input = np.dot(grad_output, weights.T)
        
        # Gradient w.r.t weights
        grad_weights = np.dot(x.T, grad_output)
        
        # Gradient w.r.t bias
        grad_bias = np.sum(grad_output, axis=0)
        
        return grad_input, {"weights": grad_weights, "bias": grad_bias}
    
    def describe(self):
        """Return human-readable description of this function."""
        return f"Linear transformation with shape ({self.input_dim}, {self.output_dim})"
    
class PolynomialFunctionNode(FunctionNode):
    """Polynomial function node: f(x) = a₀ + a₁(x·v) + a₂(x·v)² + ... + aₙ(x·v)ⁿ"""
    
    def __init__(self, input_dim, degree=3, coefficients=None, direction=None, trainable=True):
        """
        Initialize a polynomial function node.
        
        Args:
            input_dim: Dimensionality of the input
            degree: Degree of the polynomial (default: 3)
            coefficients: List of polynomial coefficients [a₀, a₁, ..., aₙ] (default: random)
            direction: Direction vector for projection (default: random unit vector)
            trainable: Whether parameters can be updated during training
        """
        super().__init__(input_dim, trainable)
        
        # Initialize coefficients
        if coefficients is None:
            # Start with reasonable random coefficients that decrease with degree
            coefficients = np.random.randn(degree + 1) / (np.arange(degree + 1) + 1.0)
        
        # Initialize direction
        if direction is None:
            direction = np.random.randn(input_dim)
            direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Store parameters
        self.parameters["coefficients"] = np.array(coefficients, dtype=np.float32)
        self.parameters["direction"] = np.array(direction, dtype=np.float32)
        self.degree = degree
        self.output_dim = 1
    
    def forward(self, x):
        """
        Compute polynomial function value for input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        coefficients = self.parameters["coefficients"]
        direction = self.parameters["direction"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction)
        
        # Compute powers of projection
        powers = np.column_stack([projection ** i for i in range(len(coefficients))])
        
        # Compute polynomial value
        return np.dot(powers, coefficients).reshape(-1, 1)
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, 1)
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for coefficients and direction
        """
        coefficients = self.parameters["coefficients"]
        direction = self.parameters["direction"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction)
        
        # Compute powers of projection for derivative
        powers = np.column_stack([i * projection ** (i-1) if i > 0 else np.zeros_like(projection) 
                                for i in range(len(coefficients))])
        
        # Gradient w.r.t projection
        grad_projection = np.dot(powers, coefficients)
        
        # Gradient w.r.t input
        grad_input = grad_output * grad_projection.reshape(-1, 1) * direction.reshape(1, -1)
        
        # Gradient w.r.t coefficients
        powers = np.column_stack([projection ** i for i in range(len(coefficients))])
        grad_coefficients = np.dot(grad_output.T, powers)[0]
        
        # Gradient w.r.t direction
        # This is complex because changing direction changes the projection
        polynomial_derivative = np.dot(powers[:, 1:], 
                                     np.arange(1, len(coefficients)) * coefficients[1:])
        grad_direction = np.sum(grad_output * polynomial_derivative.reshape(-1, 1) * x, axis=0)
        
        return grad_input, {"coefficients": grad_coefficients, "direction": grad_direction}
    
    def describe(self):
        """Return human-readable description of this function."""
        coeff_str = ", ".join([f"{c:.3f}" for c in self.parameters["coefficients"]])
        return f"Polynomial of degree {self.degree} with coefficients [{coeff_str}]"
    
class SinusoidalFunctionNode(FunctionNode):
    """Sinusoidal function node: f(x) = amplitude * sin(frequency * (x·direction) + phase)"""
    
    def __init__(self, input_dim, frequency=None, amplitude=None, phase=None, 
                 direction=None, trainable=True):
        """
        Initialize a sinusoidal function node.
        
        Args:
            frequency, amplitude, phase: Parameters of the sinusoid (default: random)
            direction: Direction vector for projection (default: random unit vector)
        """
        super().__init__(input_dim, trainable)
        
        if direction is None:
            direction = np.random.randn(input_dim)
            direction = direction / np.linalg.norm(direction)
        if frequency is None:
            frequency = np.random.uniform(0.5, 2.0)
        if amplitude is None:
            amplitude = np.random.uniform(0.5, 1.5)
        if phase is None:
            phase = np.random.uniform(0, 2 * np.pi)
        
        self.parameters["frequency"] = np.float64(frequency)
        self.parameters["amplitude"] = np.float64(amplitude)
        self.parameters["phase"] = np.float64(phase)
        self.parameters["direction"] = np.array(direction, dtype=np.float64)
        self.output_dim = 1
    
    def forward(self, x):
        """
        Compute sinusoidal function value for input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        frequency = self.parameters["frequency"]
        amplitude = self.parameters["amplitude"]
        phase = self.parameters["phase"]
        direction = self.parameters["direction"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction)
        
        # Apply sinusoidal function
        return amplitude * np.sin(frequency * projection + phase).reshape(-1, 1)
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, 1)
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for sinusoid parameters
        """
        frequency = self.parameters["frequency"]
        amplitude = self.parameters["amplitude"]
        phase = self.parameters["phase"]
        direction = self.parameters["direction"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction)
        
        # Compute intermediate values
        inner_term = frequency * projection + phase
        sin_term = np.sin(inner_term)
        cos_term = np.cos(inner_term)
        
        # Gradient w.r.t input
        grad_input = grad_output * amplitude * frequency * cos_term.reshape(-1, 1) * direction.reshape(1, -1)
        
        # Gradient w.r.t amplitude
        grad_amplitude = np.sum(grad_output * sin_term)
        
        # Gradient w.r.t frequency
        grad_frequency = np.sum(grad_output * amplitude * projection * cos_term)
        
        # Gradient w.r.t phase
        grad_phase = np.sum(grad_output * amplitude * cos_term)
        
        # Gradient w.r.t direction
        grad_direction = np.sum(grad_output * amplitude * frequency * cos_term.reshape(-1, 1) * x, axis=0)
        
        return grad_input, {
            "amplitude": grad_amplitude,
            "frequency": grad_frequency,
            "phase": grad_phase,
            "direction": grad_direction
        }
    
    def describe(self):
        """Return human-readable description of this function."""
        return (f"Sinusoid with amplitude={self.parameters['amplitude']:.3f}, "
                f"frequency={self.parameters['frequency']:.3f}, phase={self.parameters['phase']:.3f}")
    
class ExponentialFunctionNode(FunctionNode):
    """Exponential function node: f(x) = scale * exp(rate * (x·direction + shift))"""
    
    def __init__(self, input_dim, rate=None, scale=None, shift=None, 
                 direction=None, trainable=True):
        """
        Initialize an exponential function node.
        
        Args:
            rate, scale, shift: Parameters of the exponential (default: random)
            direction: Direction vector for projection (default: random unit vector)
        """
        super().__init__(input_dim, trainable)
        
        if direction is None:
            direction = np.random.randn(input_dim)
            direction = direction / np.linalg.norm(direction)
        if rate is None:
            rate = np.random.uniform(0.05, 0.2)
        if scale is None:
            scale = np.random.uniform(0.8, 1.2)
        if shift is None:
            shift = np.random.uniform(-0.1, 0.1)
        
        self.parameters["rate"] = np.float32(rate)
        self.parameters["scale"] = np.float32(scale)
        self.parameters["shift"] = np.float32(shift)
        self.parameters["direction"] = np.array(direction, dtype=np.float32)
        self.output_dim = 1
    
    def forward(self, x):
        """
        Compute exponential function value for input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        rate = self.parameters["rate"]
        scale = self.parameters["scale"]
        shift = self.parameters["shift"]
        direction = self.parameters["direction"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction) + shift
        
        # Apply exponential function
        return scale * np.exp(rate * projection).reshape(-1, 1)
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, 1)
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for exponential parameters
        """
        rate = self.parameters["rate"]
        scale = self.parameters["scale"]
        shift = self.parameters["shift"]
        direction = self.parameters["direction"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction) + shift
        
        # Compute exponential values
        exp_values = scale * np.exp(rate * projection)
        
        # Gradient w.r.t input
        grad_input = grad_output * exp_values.reshape(-1, 1) * rate * direction.reshape(1, -1)
        
        # Gradient w.r.t scale
        grad_scale = np.sum(grad_output * exp_values / scale)
        
        # Gradient w.r.t rate
        grad_rate = np.sum(grad_output * exp_values * projection)
        
        # Gradient w.r.t shift
        grad_shift = np.sum(grad_output * exp_values * rate)
        
        # Gradient w.r.t direction
        grad_direction = np.sum(grad_output * exp_values.reshape(-1, 1) * rate * x, axis=0)
        
        return grad_input, {
            "scale": grad_scale,
            "rate": grad_rate,
            "shift": grad_shift,
            "direction": grad_direction
        }
    
    def describe(self):
        """Return human-readable description of this function."""
        return (f"Exponential with scale={self.parameters['scale']:.3f}, "
                f"rate={self.parameters['rate']:.3f}, shift={self.parameters['shift']:.3f}")
    
class StepFunctionNode(FunctionNode):
    """Step function node: f(x) = height if (x·direction + bias) > 0 else 0"""
    
    def __init__(self, input_dim, height=None, bias=None, direction=None, 
                 smoothing=0.01, trainable=True):
        """
        Initialize a step function node with optional smoothing.
        
        Args:
            height, bias: Parameters of the step function (default: random)
            direction: Direction vector for projection (default: random unit vector)
        """
        super().__init__(input_dim, trainable)
        
        if direction is None:
            direction = np.random.randn(input_dim)
            direction = direction / np.linalg.norm(direction)
        if height is None:
            height = np.random.uniform(0.8, 1.2)
        if bias is None:
            bias = np.random.uniform(-0.1, 0.1)
        
        self.parameters["height"] = np.float32(height)
        self.parameters["bias"] = np.float32(bias)
        self.parameters["direction"] = np.array(direction, dtype=np.float32)
        self.parameters["smoothing"] = np.float32(smoothing)
        self.output_dim = 1
    
    def forward(self, x):
        """
        Compute step function value for input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        height = self.parameters["height"]
        bias = self.parameters["bias"]
        direction = self.parameters["direction"]
        smoothing = self.parameters["smoothing"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction) + bias
        
        # Apply smoothed step function
        if smoothing > 0:
            # Use sigmoid as a smooth approximation of step function
            return height / (1.0 + np.exp(-projection / smoothing)).reshape(-1, 1)
        else:
            # Pure step function (not differentiable)
            return height * (projection > 0).astype(np.float32).reshape(-1, 1)
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of shape (batch_size, 1)
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient w.r.t input
            - grad_params: Dictionary with gradients for step parameters
        """
        height = self.parameters["height"]
        bias = self.parameters["bias"]
        direction = self.parameters["direction"]
        smoothing = self.parameters["smoothing"]
        
        # Project input onto direction vector
        projection = np.dot(x, direction) + bias
        
        # Compute sigmoid and its derivative
        sigmoid = 1.0 / (1.0 + np.exp(-projection / smoothing))
        sigmoid_derivative = sigmoid * (1 - sigmoid) / smoothing
        
        # Gradient w.r.t input
        grad_input = grad_output * height * sigmoid_derivative.reshape(-1, 1) * direction.reshape(1, -1)
        
        # Gradient w.r.t height
        grad_height = np.sum(grad_output * sigmoid)
        
        # Gradient w.r.t bias
        grad_bias = np.sum(grad_output * height * sigmoid_derivative)
        
        # Gradient w.r.t direction
        grad_direction = np.sum(grad_output * height * sigmoid_derivative.reshape(-1, 1) * x, axis=0)
        
        # Gradient w.r.t smoothing
        temp = projection / (smoothing * smoothing)
        grad_smoothing = np.sum(grad_output * height * sigmoid_derivative * temp)
        
        return grad_input, {
            "height": grad_height,
            "bias": grad_bias,
            "direction": grad_direction,
            "smoothing": grad_smoothing
        }
    
    def describe(self):
        """Return human-readable description of this function."""
        return (f"Step function with height={self.parameters['height']:.3f}, "
                f"bias={self.parameters['bias']:.3f}, "
                f"smoothing={self.parameters['smoothing']:.3f}")

class ReLUFunctionNode(FunctionNode):
    """Rectified Linear Unit (ReLU) function node: f(x) = max(0, x)"""
    
    def __init__(self, input_dim, trainable=False):
        """
        Initialize a ReLU function node.
        
        Args:
            input_dim: Dimensionality of the input
            trainable: ReLU has no parameters, so this is always False
        """
        super().__init__(input_dim, trainable=False)
        self.output_dim = input_dim
    
    def forward(self, x):
        """
        Apply ReLU function to input x.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, input_dim)
        """
        return np.maximum(0, x)
    
    def backward(self, grad_output, x):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient of the loss with respect to this node's output
            x: Input that was used in the forward pass
            
        Returns:
            - grad_input: Gradient with respect to the input
            - grad_params: Empty dictionary (ReLU has no parameters)
        """
        # Gradient is 1 for positive inputs, 0 otherwise
        grad_input = grad_output * (x > 0)
        return grad_input, {}
    
    def describe(self):
        """Return human-readable description of this function."""
        return f"ReLU activation (input_dim={self.input_dim})"
    

class FunctionNodeFactory:
    """Factory class for creating and deserializing function nodes."""
    
    # Registry of function node types
    _node_types = {
        "GaussianFunctionNode": GaussianFunctionNode,
        "SigmoidFunctionNode": SigmoidFunctionNode,
        "LinearFunctionNode": LinearFunctionNode,
        "PolynomialFunctionNode": PolynomialFunctionNode,
        "SinusoidalFunctionNode": SinusoidalFunctionNode,
        "ExponentialFunctionNode": ExponentialFunctionNode,
        "StepFunctionNode": StepFunctionNode,
        "ReLUFunctionNode": ReLUFunctionNode,
        # Add more as needed
    }
    
    @classmethod
    def create(cls, node_type, **kwargs):
        """
        Create a new function node of the specified type.
        
        Args:
            node_type: String identifier of the node type
            **kwargs: Arguments to pass to the node constructor
            
        Returns:
            Instantiated function node
        """
        if node_type not in cls._node_types:
            raise ValueError(f"Unknown function node type: {node_type}")
        
        return cls._node_types[node_type](**kwargs)
    
    @classmethod
    def deserialize(cls, data):
        """
        Create a function node from serialized data.
        
        Args:
            data: Dictionary containing serialized node data
            
        Returns:
            Instantiated FunctionNode object
        """
        node_type = data["type"]
        
        if node_type not in cls._node_types:
            raise ValueError(f"Unknown function node type: {node_type}")
        
        # Create node with basic parameters
        node = cls._node_types[node_type](
            input_dim=data["input_dim"]
        )
        
        # Restore parameters and output_dim
        node.parameters = data["parameters"]
        if "output_dim" in data:
            node.output_dim = data["output_dim"]
        
        return node