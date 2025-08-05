
from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionNode(nn.Module):
    """
    Base class for all function nodes in the compositional network.

    Each function node represents a mathematical operation.

    Attributes:
        input_dim (int): The dimensionality of the input.
        output_dim (int): The dimensionality of the output.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim: Optional[int] = None  # Must be set by subclasses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the function node.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def describe(self) -> str:
        """
        Returns a string description of the function node and its parameters.

        Returns:
            str: The description of the function node.
        """
        params = {
            name: f"{param.detach().cpu().numpy().round(3)}"
            for name, param in self.named_parameters()
        }
        return f"{self.__class__.__name__} with params: {params}"

    def set_trainable(self, trainable: bool):
        """
        Sets the `requires_grad` attribute of the node's parameters.

        Args:
            trainable (bool): Whether the parameters should be trainable.
        """
        for param in self.parameters():
            param.requires_grad = trainable


# region --- Standard Function Nodes ---


class LinearFunctionNode(FunctionNode):
    """
    A linear function node: `f(x) = xW + b`.

    Args:
        input_dim (int): The dimensionality of the input.
        output_dim (int): The dimensionality of the output. Defaults to 1.
        weights (Optional[torch.Tensor]): The initial weights. If None,
            Kaiming uniform initialization is used. Defaults to None.
        bias (Optional[torch.Tensor]): The initial bias. If None, initialized
            to zeros. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        weights: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__(input_dim)
        self.output_dim = output_dim

        if weights is None:
            scale = np.sqrt(2.0 / input_dim) if input_dim > 0 else 1.0
            weights = torch.randn(input_dim, output_dim) * scale
        self.weights = nn.Parameter(torch.as_tensor(weights, dtype=torch.float32))

        if bias is None:
            bias = torch.zeros(output_dim)
        self.bias = nn.Parameter(torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights) + self.bias


class ReLUFunctionNode(FunctionNode):
    """
    A Rectified Linear Unit (ReLU) function node: `f(x) = max(0, x)`.

    Args:
        input_dim (int): The dimensionality of the input.
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.output_dim = input_dim
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


class BatchNormNode(FunctionNode):
    """
    A Batch Normalization function node implemented from first principles.

    Args:
        input_dim (int): The number of features in the input.
        eps (float): A value added to the denominator for numerical stability. Defaults to 1e-5.
        momentum (float): The value used for the running_mean and running_var computation. Defaults to 0.1.
    """
    def __init__(self, input_dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(input_dim)
        self.output_dim = input_dim
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

        # Buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()

            # Normalize using batch statistics
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Normalize using running statistics
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        return self.gamma * x_hat + self.beta

    def describe(self) -> str:
        return f"BatchNormNode(eps={self.eps}, momentum={self.momentum})"


class SinusoidalFunctionNode(FunctionNode):
    """
    A sinusoidal function node: `f(x) = amplitude * sin(frequency * (x·direction) + phase)`.

    Args:
        input_dim (int): The dimensionality of the input.
        frequency (float): The frequency of the sinusoid. Defaults to 1.0.
        amplitude (float): The amplitude of the sinusoid. Defaults to 1.0.
        phase (float): The phase of the sinusoid. Defaults to 0.0.
        direction (Optional[torch.Tensor]): The direction vector. If None,
            initialized randomly. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        direction: Optional[torch.Tensor] = None,
    ):
        super().__init__(input_dim)
        self.output_dim = 1
        if direction is None:
            direction = torch.randn(input_dim)
            direction /= torch.norm(direction) if torch.norm(direction) > 0 else 1.0
        self.direction = nn.Parameter(torch.as_tensor(direction, dtype=torch.float32))
        self.frequency = nn.Parameter(torch.tensor(float(frequency), dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(float(amplitude), dtype=torch.float32))
        self.phase = nn.Parameter(torch.tensor(float(phase), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = torch.clamp(self.frequency, min=1e-6)
        amplitude = torch.clamp(self.amplitude, min=1e-6)
        projection = torch.matmul(x, self.direction)
        return (amplitude * torch.sin(frequency * projection + self.phase)).unsqueeze(-1)


class PolynomialFunctionNode(FunctionNode):
    """
    A polynomial function node: `f(x) = c_0 + c_1*(x·d) + c_2*(x·d)^2 + ...`

    Args:
        input_dim (int): The dimensionality of the input.
        degree (int): The degree of the polynomial. Defaults to 3.
        coefficients (Optional[torch.Tensor]): The polynomial coefficients.
            If None, initialized randomly. Defaults to None.
        direction (Optional[torch.Tensor]): The direction vector. If None,
            initialized randomly. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        degree: int = 3,
        coefficients: Optional[torch.Tensor] = None,
        direction: Optional[torch.Tensor] = None,
    ):
        super().__init__(input_dim)
        self.output_dim = 1
        self.degree = degree
        if direction is None:
            direction = torch.randn(input_dim)
            direction /= torch.norm(direction) if torch.norm(direction) > 0 else 1.0
        self.direction = nn.Parameter(torch.as_tensor(direction, dtype=torch.float32))
        if coefficients is None:
            coefficients = torch.randn(degree + 1) / torch.arange(
                1, degree + 2, dtype=torch.float32
            )
        self.coefficients = nn.Parameter(
            torch.as_tensor(coefficients, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.direction)
        powers = torch.stack([projection**i for i in range(self.degree + 1)], dim=1)
        return torch.matmul(powers, self.coefficients).unsqueeze(-1)


class GaussianFunctionNode(FunctionNode):
    """
    A Gaussian function node: `f(x) = exp(-||x - center||^2 / (2 * width^2))`.

    Args:
        input_dim (int): The dimensionality of the input.
        center (Optional[torch.Tensor]): The center of the Gaussian. If None,
            initialized randomly. Defaults to None.
        width (float): The width of the Gaussian. Defaults to 1.0.
    """

    def __init__(
        self, input_dim: int, center: Optional[torch.Tensor] = None, width: float = 1.0
    ):
        super().__init__(input_dim)
        self.output_dim = 1
        if center is None:
            center = torch.randn(input_dim)
        self.center = nn.Parameter(torch.as_tensor(center, dtype=torch.float32))
        self.width = nn.Parameter(torch.tensor(float(width), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = torch.clamp(self.width, min=1e-6)
        squared_dist = torch.sum((x - self.center) ** 2, dim=1, keepdim=True)
        return torch.exp(-squared_dist / (2 * width**2))


class SigmoidFunctionNode(FunctionNode):
    """
    A sigmoid function node.

    - If `direction` is provided, it computes: `f(x) = 1 / (1 + exp(-s * (x·d + o)))`
    - If `direction` is None, it computes element-wise: `f(x) = 1 / (1 + exp(-s * (x + o)))`

    Args:
        input_dim (int): The dimensionality of the input.
        direction (Optional[torch.Tensor]): The direction vector. If None,
            the sigmoid is applied element-wise. Defaults to None.
        offset (float): The offset. Defaults to 0.0.
        steepness (float): The steepness of the sigmoid. Defaults to 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        direction: Optional[torch.Tensor] = None,
        offset: float = 0.0,
        steepness: float = 1.0,
    ):
        super().__init__(input_dim)

        self.is_elementwise = direction is None

        if self.is_elementwise:
            self.output_dim = input_dim
            self.offset = nn.Parameter(
                torch.full((input_dim,), float(offset), dtype=torch.float32)
            )
            self.steepness = nn.Parameter(
                torch.full((input_dim,), float(steepness), dtype=torch.float32)
            )
        else:
            self.output_dim = 1
            if direction is None:
                direction = torch.randn(input_dim)
                direction = direction / torch.norm(direction)
            self.direction = nn.Parameter(torch.as_tensor(direction, dtype=torch.float32))
            self.offset = nn.Parameter(torch.tensor(float(offset), dtype=torch.float32))
            self.steepness = nn.Parameter(
                torch.tensor(float(steepness), dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_elementwise:
            return torch.sigmoid(self.steepness * (x + self.offset))
        else:
            projection = torch.matmul(x, self.direction) + self.offset
            return torch.sigmoid(self.steepness * projection).unsqueeze(-1)


class StepFunctionNode(FunctionNode):
    """
    A smoothed, differentiable step function.

    `f(x) = 1 / (1 + exp(-smoothing * (x·d - bias)))`

    Args:
        input_dim (int): The dimensionality of the input.
        direction (torch.Tensor): The direction vector.
        bias (float): The bias (threshold) of the step.
        smoothing (float): The smoothing factor. Defaults to 10.0.
    """

    def __init__(
        self, input_dim: int, direction: torch.Tensor, bias: float, smoothing: float = 10.0
    ):
        super().__init__(input_dim)
        self.output_dim = 1
        self.direction = nn.Parameter(torch.as_tensor(direction, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))
        self.smoothing = nn.Parameter(torch.tensor(float(smoothing), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.direction)
        return torch.sigmoid(self.smoothing * (projection - self.bias)).unsqueeze(-1)


class ExponentialFunctionNode(FunctionNode):
    """
    An exponential function node.

    `f(x) = scale * exp(rate * (x·d - shift))`

    Args:
        input_dim (int): The dimensionality of the input.
        direction (torch.Tensor): The direction vector.
        rate (float): The rate of the exponential.
        shift (float): The shift of the exponential.
        scale (float): The scale of the exponential.
    """

    def __init__(
        self,
        input_dim: int,
        direction: torch.Tensor,
        rate: float,
        shift: float,
        scale: float,
    ):
        super().__init__(input_dim)
        self.output_dim = 1
        self.direction = nn.Parameter(torch.as_tensor(direction, dtype=torch.float32))
        self.rate = nn.Parameter(torch.tensor(float(rate), dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(float(shift), dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.direction)
        return (self.scale * torch.exp(self.rate * (projection - self.shift))).unsqueeze(
            -1
        )


# endregion


# region --- Image-Specific Function Nodes ---


class FourierFunctionNode(FunctionNode):
    """
    A function node that computes Fourier features from an image.

    Args:
        input_dim (int): The dimensionality of the flattened input image.
        image_size (Tuple[int, int]): The (height, width) of the image.
            Defaults to (32, 32).
        n_channels (int): The number of channels in the image. Defaults to 3.
        n_features (int): The number of Fourier features to compute.
            Defaults to 1.
    """

    def __init__(
        self,
        input_dim: int,
        image_size: Tuple[int, int] = (32, 32),
        n_channels: int = 3,
        n_features: int = 1,
    ):
        super().__init__(input_dim)
        assert input_dim == image_size[0] * image_size[1] * n_channels
        self.output_dim = n_features * 2
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_features = n_features
        self.freqs = nn.Parameter(torch.randn(n_features, 2))
        self.phases = nn.Parameter(torch.randn(n_features))
        self.channel_weights = nn.Parameter(torch.randn(n_features, n_channels))
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, image_size[0]),
            torch.linspace(0, 1, image_size[1]),
            indexing="ij",
        )
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_img = x.view(
            batch_size, self.n_channels, self.image_size[0], self.image_size[1]
        )
        basis_arg = (
            2
            * np.pi
            *
            (
                self.grid[..., 0] * self.freqs[:, 0].view(-1, 1, 1)
                + self.grid[..., 1] * self.freqs[:, 1].view(-1, 1, 1)
            )
            + self.phases.view(-1, 1, 1)
        )
        cos_basis = torch.cos(basis_arg)
        sin_basis = torch.sin(basis_arg)
        cos_weighted_channels = torch.einsum("bchw,fhw->bfc", x_img, cos_basis)
        sin_weighted_channels = torch.einsum("bchw,fhw->bfc", x_img, sin_basis)
        cos_output = torch.einsum(
            "bfc,fc->bf", cos_weighted_channels, self.channel_weights
        )
        sin_output = torch.einsum(
            "bfc,fc->bf", sin_weighted_channels, self.channel_weights
        )
        return torch.cat([cos_output, sin_output], dim=1)


class GaborFunctionNode(FunctionNode):
    """
    A function node that applies a Gabor filter to an image.

    Args:
        input_dim (int): The dimensionality of the flattened input image.
        image_size (Tuple[int, int]): The (height, width) of the image.
            Defaults to (32, 32).
        n_channels (int): The number of channels in the image. Defaults to 3.
    """

    def __init__(
        self, input_dim: int, image_size: Tuple[int, int] = (32, 32), n_channels: int = 3
    ):
        super().__init__(input_dim)
        assert input_dim == image_size[0] * image_size[1] * n_channels
        self.output_dim = 1
        self.image_size = image_size
        self.n_channels = n_channels
        self.frequency = nn.Parameter(torch.randn(1))
        self.theta = nn.Parameter(torch.randn(1))
        self.sigma_x = nn.Parameter(torch.randn(1))
        self.sigma_y = nn.Parameter(torch.randn(1))
        self.psi = nn.Parameter(torch.randn(1))
        self.channel_weights = nn.Parameter(torch.randn(n_channels))
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-image_size[0] // 2, image_size[0] // 2, image_size[0]),
            torch.linspace(-image_size[1] // 2, image_size[1] // 2, image_size[1]),
            indexing="ij",
        )
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1))

    def _get_gabor_kernel(self) -> torch.Tensor:
        sigma_x = torch.clamp(self.sigma_x, min=1e-6)
        sigma_y = torch.clamp(self.sigma_y, min=1e-6)
        x_prime = (
            self.grid[..., 0] * torch.cos(self.theta)
            + self.grid[..., 1] * torch.sin(self.theta)
        )
        y_prime = (
            -self.grid[..., 0] * torch.sin(self.theta)
            + self.grid[..., 1] * torch.cos(self.theta)
        )
        envelope = torch.exp(-0.5 * (x_prime**2 / sigma_x**2 + y_prime**2 / sigma_y**2))
        carrier = torch.cos(2 * np.pi * self.frequency * x_prime + self.psi)
        return envelope * carrier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_img = x.view(
            batch_size, self.n_channels, self.image_size[0], self.image_size[1]
        )
        gabor_kernel = self._get_gabor_kernel()
        response = torch.einsum("bchw,hw->bc", x_img, gabor_kernel)
        output = torch.einsum("bc,c->b", response, self.channel_weights).unsqueeze(-1)
        return output


# endregion


# region --- DeepCFN Nodes ---


class GenericConvNode(FunctionNode):
    """
    A generic 2D convolutional node.

    Args:
        input_shape (Tuple[int, int]): The (height, width) of the input.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        padding (int): The padding to apply. Defaults to 0.
        stride (int): The stride of the convolution. Defaults to 1.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        self.input_shape_2d = (in_channels, input_shape[0], input_shape[1])
        input_dim = in_channels * input_shape[0] * input_shape[1]
        super().__init__(input_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        output_h = (input_shape[0] + 2 * padding - kernel_size) // stride + 1
        output_w = (input_shape[1] + 2 * padding - kernel_size) // stride + 1
        self.output_shape_2d = (out_channels, output_h, output_w)
        self.output_dim = out_channels * output_h * output_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_2d = x.view(
            batch_size,
            self.in_channels,
            self.input_shape_2d[1],
            self.input_shape_2d[2],
        )
        conv_out = nn.functional.conv2d(
            x_2d, self.weight, self.bias, self.stride, self.padding
        )
        return conv_out.view(batch_size, -1)


class PoolingNode(FunctionNode):
    """
    An average pooling node.

    Args:
        input_shape (Tuple[int, int]): The (height, width) of the input.
        in_channels (int): The number of input channels.
        kernel_size (int): The size of the pooling kernel. Defaults to 2.
        stride (int): The stride of the pooling. Defaults to 2.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        in_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        self.input_shape_2d = (in_channels, input_shape[0], input_shape[1])
        input_dim = in_channels * input_shape[0] * input_shape[1]
        super().__init__(input_dim)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        output_h = (input_shape[0] - kernel_size) // stride + 1
        output_w = (input_shape[1] - kernel_size) // stride + 1
        self.output_shape_2d = (in_channels, output_h, output_w)
        self.output_dim = in_channels * output_h * output_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_2d = x.view(
            batch_size,
            self.in_channels,
            self.input_shape_2d[1],
            self.input_shape_2d[2],
        )
        pool_out = nn.functional.avg_pool2d(x_2d, self.kernel_size, self.stride)
        return pool_out.view(batch_size, -1)


# endregion


# region --- Regularization Nodes ---


class DropoutFunctionNode(FunctionNode):
    """
    A dropout function node.

    Args:
        input_dim (int): The dimensionality of the input.
        p (float): The probability of an element to be zeroed. Defaults to 0.5.
    """

    def __init__(self, input_dim: int, p: float = 0.5):
        super().__init__(input_dim)
        self.output_dim = input_dim  # Dropout doesn't change dimension
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


# endregion


# region --- Wrapper Nodes ---


class SequentialWrapperNode(FunctionNode):
    """
    A node that wraps a sequence of function nodes.

    Args:
        function_nodes (list[FunctionNode]): The list of function nodes to wrap.
        input_dim (Optional[int]): The input dimension. If provided, it must
            match the input dimension of the first node. Defaults to None.
    """

    def __init__(self, function_nodes, input_dim: Optional[int] = None):
        # Added input_dim parameter
        # Validate that the provided input_dim (if any) matches the first node's input_dim
        if input_dim is not None and input_dim != function_nodes[0].input_dim:
            raise ValueError(
                "Provided input_dim does not match the input_dim of the first function node."
            )
        super().__init__(function_nodes[0].input_dim)
        self.function_nodes = nn.ModuleList(function_nodes)

        # Validate dimensions and set output_dim
        for i in range(1, len(function_nodes)):
            if function_nodes[i].input_dim != function_nodes[i - 1].output_dim:
                raise ValueError(
                    f"Dimension mismatch at index {i} in SequentialWrapperNode"
                )

        self.output_dim = function_nodes[-1].output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for node in self.function_nodes:
            x = node(x)
        return x

    def describe(self) -> str:
        lines = [f"{self.__class__.__name__} (Sequential):"]
        for i, node in enumerate(self.function_nodes):
            lines.append(f"  - Step {i + 1}: {node.describe()}")
        return "\n".join(lines)


class IdentityFunctionNode(FunctionNode):
    """
    An identity function node: `f(x) = x`.

    Args:
        input_dim (int): The dimensionality of the input.
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.output_dim = input_dim  # Output dimension is same as input dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def describe(self) -> str:
        return f"{self.__class__.__name__} with params: {{}}"


class AttentionFunctionNode(FunctionNode):
    """
    A mathematical function node that computes attention as a weighted sum operation.

    The attention operation is defined as:
    1. Compute similarity between elements: sim(x_i, x_j) = (x_i·x_j)/sqrt(d)
    2. Normalize similarities with softmax: w_ij = exp(sim_ij)/sum_k(exp(sim_ik))
    3. Compute weighted sum: y_i = sum_j(w_ij * x_j)

    Args:
        input_dim (int): The dimensionality of the input.
        scale_factor (Optional[float]): The scale factor for the dot product.
            If None, defaults to `1 / sqrt(input_dim)`. Defaults to None.
    """

    def __init__(self, input_dim: int, scale_factor: Optional[float] = None):
        super().__init__(input_dim)
        self.input_dim = input_dim
        # Scale factor for dot product (typically 1/sqrt(dimension))
        self.scale_factor = scale_factor or (input_dim**-0.5)
        self.output_dim = input_dim

    def forward(
        self, x: torch.Tensor, context_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Apply attention as a mathematical operation.

        Args:
            x (torch.Tensor): The input tensor of shape
                `[batch_size * num_patches, input_dim]`.
            context_shape (Optional[Tuple[int, int]]): The (batch_size, num_patches)
                to reshape the input. If None, it is inferred. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        if context_shape is None:
            # Default reshape assumes x is [batch_size * num_patches, input_dim]
            batch_size = x.shape[0] // 225  # Default to 225 patches
            num_patches = 225
        else:
            batch_size, num_patches = context_shape

        # Reshape to [batch_size, num_patches, input_dim]
        x_reshaped = x.view(batch_size, num_patches, self.input_dim)

        # Compute similarity matrix - mathematical dot product operation
        # [batch_size, num_patches, num_patches]
        similarity = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2)) * self.scale_factor

        # Apply softmax to get normalized weights - mathematical exponential and normalization
        attention_weights = F.softmax(similarity, dim=-1)

        # Apply attention weights - mathematical weighted sum
        attended_features = torch.bmm(attention_weights, x_reshaped)

        # Reshape back to original format
        return attended_features.view(batch_size * num_patches, self.input_dim)

    def describe(self) -> str:
        return f"AttentionFunctionNode: Mathematical attention operation with scale={self.scale_factor}"


# endregion


class FunctionNodeFactory:
    """
    A factory for creating function nodes.
    """

    _node_types: Dict[str, Type[FunctionNode]] = {
        "Linear": LinearFunctionNode,
        "ReLU": ReLUFunctionNode,
        "BatchNorm": BatchNormNode,
        "Sinusoidal": SinusoidalFunctionNode,
        "Polynomial": PolynomialFunctionNode,
        "Gaussian": GaussianFunctionNode,
        "Sigmoid": SigmoidFunctionNode,
        "Fourier": FourierFunctionNode,
        "Gabor": GaborFunctionNode,
        "Step": StepFunctionNode,
        "Exponential": ExponentialFunctionNode,
        "GenericConv": GenericConvNode,
        "Pooling": PoolingNode,
        "Dropout": DropoutFunctionNode,
        "SequentialWrapper": SequentialWrapperNode,
        "Identity": IdentityFunctionNode,
        "MathematicalAttention": AttentionFunctionNode,
    }

    @classmethod
    def create(cls, node_type: str, **kwargs) -> FunctionNode:
        """
        Creates a function node of the specified type.

        Args:
            node_type (str): The type of the function node to create.
            **kwargs: The arguments for the function node's constructor.

        Returns:
            FunctionNode: The created function node.
        """
        if node_type not in cls._node_types:
            raise ValueError(f"Unknown function node type: {node_type}")
        return cls._node_types[node_type](**kwargs)

