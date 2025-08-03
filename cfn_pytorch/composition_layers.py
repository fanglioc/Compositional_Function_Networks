

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .function_nodes import FunctionNode


class CompositionLayer(nn.Module):
    """
    Base class for all composition layers in the CFN.

    Attributes:
        name (str): The name of the layer.
        input_dim (int): The dimensionality of the input.
        output_dim (int): The dimensionality of the output.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name if name else self.__class__.__name__
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def describe(self) -> str:
        """
        Returns a string description of the layer.

        Returns:
            str: The description of the layer.
        """
        return self.name


class SequentialCompositionLayer(CompositionLayer):
    """
    A layer that composes a sequence of function nodes sequentially.

    The output of one node is the input to the next, i.e.,
    `output = f_n(...f_2(f_1(x)))...

    Args:
        function_nodes (List[FunctionNode]): A list of function nodes to be
            composed sequentially.
        name (Optional[str]): The name of the layer.
    """

    def __init__(self, function_nodes: List[FunctionNode], name: Optional[str] = None):
        super().__init__(name)
        self.function_nodes = nn.ModuleList(function_nodes)

        # Validate dimensions
        for i in range(1, len(function_nodes)):
            if function_nodes[i].input_dim != function_nodes[i - 1].output_dim:
                raise ValueError(f"Dimension mismatch at index {i}")

        self.input_dim = function_nodes[0].input_dim if function_nodes else 0
        self.output_dim = function_nodes[-1].output_dim if function_nodes else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the sequential composition to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after sequential composition.
        """
        for node in self.function_nodes:
            x = node(x)
        return x

    def describe(self) -> str:
        """
        Returns a string description of the sequential layer.

        Returns:
            str: The description of the layer.
        """
        description = f"{self.name} (Sequential):\n"
        for i, node in enumerate(self.function_nodes):
            description += f"  - Step {i + 1}: {node.describe()}\n"
        return description


class ParallelCompositionLayer(CompositionLayer):
    """
    A layer that applies multiple function nodes to the same input in parallel
    and combines their outputs.

    Args:
        function_nodes (List[FunctionNode]): A list of function nodes to be
            applied in parallel.
        combination (str): The method for combining the outputs of the function
            nodes. One of 'sum', 'add', 'product', 'concat', or 'weighted_sum'.
            Defaults to 'sum'.
        weights (Optional[Union[torch.Tensor, List[float]]]): The weights for
            the 'weighted_sum' combination method. If None, equal weights are
            used. Defaults to None.
        name (Optional[str]): The name of the layer.
    """

    def __init__(
        self,
        function_nodes: List[FunctionNode],
        combination: str = "sum",
        weights: Optional[Union[torch.Tensor, List[float]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.function_nodes = nn.ModuleList(function_nodes)
        self.combination = combination

        # Validate input dimensions
        input_dim = function_nodes[0].input_dim
        for node in function_nodes:
            if node.input_dim != input_dim:
                raise ValueError(
                    "All nodes in a parallel layer must have the same input dimension."
                )
        self.input_dim = input_dim

        # Determine output dimension
        if self.combination == "concat":
            self.output_dim = sum(node.output_dim for node in function_nodes)
        else:
            output_dim = function_nodes[0].output_dim
            for node in function_nodes:
                if node.output_dim != output_dim:
                    raise ValueError(
                        f"For '{self.combination}', all nodes must have the same output dimension."
                    )
            self.output_dim = output_dim

        if self.combination == "weighted_sum":
            if weights is None:
                weights = torch.ones(len(function_nodes)) / len(function_nodes)
            self.weights = nn.Parameter(
                weights.clone().detach()
                if isinstance(weights, torch.Tensor)
                else torch.tensor(weights, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the parallel composition to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The combined output tensor.
        """
        node_outputs = [node(x) for node in self.function_nodes]

        if self.combination in ("sum", "add"):
            return torch.sum(torch.stack(node_outputs), dim=0)
        elif self.combination == "product":
            return torch.prod(torch.stack(node_outputs), dim=0)
        elif self.combination == "concat":
            return torch.cat(node_outputs, dim=1)
        elif self.combination == "weighted_sum":
            return torch.sum(
                torch.stack(node_outputs) * self.weights.view(-1, 1, 1), dim=0
            )
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")

    def describe(self) -> str:
        """
        Returns a string description of the parallel layer.

        Returns:
            str: The description of the layer.
        """
        description = f"{self.name} (Parallel, combination={self.combination}):\n"
        for i, node in enumerate(self.function_nodes):
            description += f"  - Node {i}: {node.describe()}\n"
        if self.combination == "weighted_sum":
            description += f"  - Weights: {self.weights.detach().numpy().round(3)}\n"
        return description


class ConditionalCompositionLayer(CompositionLayer):
    """
    A layer that applies different functions based on conditions.

    The output is a weighted sum of the function outputs, where the weights
    are determined by the condition nodes:
    `f(x) = condition_1(x) * function_1(x) + condition_2(x) * function_2(x) + ...`

    Args:
        condition_nodes (List[FunctionNode]): A list of nodes that determine
            the weights for each function.
        function_nodes (List[FunctionNode]): A list of functions to be applied.
        name (Optional[str]): The name of the layer.
    """

    def __init__(
        self,
        condition_nodes: List[FunctionNode],
        function_nodes: List[FunctionNode],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        if len(condition_nodes) != len(function_nodes):
            raise ValueError("Number of condition nodes must match number of function nodes.")

        self.condition_nodes = nn.ModuleList(condition_nodes)
        self.function_nodes = nn.ModuleList(function_nodes)

        # Validate dimensions
        input_dim = condition_nodes[0].input_dim
        for node in self.condition_nodes:
            if node.input_dim != input_dim:
                raise ValueError("All condition nodes must have the same input dimension.")
            if node.output_dim != 1:
                raise ValueError("All condition nodes must have output_dim=1.")
        for node in self.function_nodes:
            if node.input_dim != input_dim:
                raise ValueError("All function nodes must have the same input dimension.")

        self.input_dim = input_dim
        self.output_dim = function_nodes[0].output_dim
        for node in self.function_nodes:
            if node.output_dim != self.output_dim:
                raise ValueError("All function nodes must have the same output dimension.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the conditional composition to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        condition_outputs = [node(x) for node in self.condition_nodes]
        function_outputs = [node(x) for node in self.function_nodes]

        # Normalize conditions to sum to 1 (using a softmax-like approach)
        condition_sum = torch.sum(torch.cat(condition_outputs, dim=1), dim=1, keepdim=True) + 1e-10
        normalized_conditions = [cond / condition_sum for cond in condition_outputs]

        result = torch.zeros_like(function_outputs[0])
        for cond, func_out in zip(normalized_conditions, function_outputs):
            result += cond * func_out

        return result

    def describe(self) -> str:
        """
        Returns a string description of the conditional layer.

        Returns:
            str: The description of the layer.
        """
        description = f"{self.name} (Conditional):\n"
        for i, (cond_node, func_node) in enumerate(
            zip(self.condition_nodes, self.function_nodes)
        ):
            description += f"  - Region {i + 1}:\n"
            description += f"    Condition: {cond_node.describe()}\n"
            description += f"    Function: {func_node.describe()}\n"
        return description


class CompositionFunctionNetwork(nn.Module):
    """
    A complete function network that combines multiple composition layers.

    Args:
        layers (List[CompositionLayer]): A list of composition layers.
        name (str): The name of the network.
    """

    def __init__(
        self,
        layers: List[CompositionLayer],
        name: str = "Compositional Function Network",
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the network to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def describe(self) -> str:
        """
        Returns a string description of the entire network.

        Returns:
            str: The description of the network.
        """
        description = f"{self.name}:\n"
        for i, layer in enumerate(self.layers):
            description += f"Layer {i + 1}: {layer.describe()}"
        return description


class ResidualCompositionLayer(CompositionLayer):
    """
    A composition layer that adds a residual connection around a main path
    composed of other CompositionLayers.
    """
    def __init__(self, main_path_layers: List[CompositionLayer], input_shape: tuple, output_shape: tuple, name: Optional[str] = None):
        super().__init__(name)
        self.main_path = nn.ModuleList(main_path_layers)

        if not main_path_layers:
            raise ValueError("main_path_layers cannot be empty.")

        # Get input and output shapes from the main path
        self.input_shape = input_shape # (C, H, W) or (D,)
        self.output_shape = output_shape # (C_out, H_out, W_out) or (D_out,)

        # Determine if we are dealing with 4D (image) or 2D (flattened) data
        is_4d_data = len(self.input_shape) == 3

        # Create shortcut connection
        if self.input_shape != self.output_shape:
            if is_4d_data:
                # For 4D data, use a 1x1 convolution to match channels and spatial dimensions
                # Calculate stride to match spatial dimensions if they change
                stride_h = self.input_shape[1] // self.output_shape[1] if self.output_shape[1] > 0 else 1
                stride_w = self.input_shape[2] // self.output_shape[2] if self.output_shape[2] > 0 else 1
                stride_h = max(1, stride_h) # Ensure stride is at least 1
                stride_w = max(1, stride_w)

                self.shortcut = nn.Conv2d(
                    in_channels=self.input_shape[0],
                    out_channels=self.output_shape[0],
                    kernel_size=1,
                    stride=(stride_h, stride_w),
                    bias=False # Often no bias in shortcut conv
                )
            else:
                # For 2D data, use a Linear layer to match dimensions
                self.shortcut = nn.Linear(np.prod(self.input_shape), np.prod(self.output_shape))
        else:
            self.shortcut = nn.Identity()

        # Set input_dim and output_dim for the base CompositionLayer
        self.input_dim = np.prod(self.input_shape)
        self.output_dim = np.prod(self.output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input x is in the expected shape (B, C, H, W) or (B, D)
        # If it's flattened (B, D) but expected 4D, reshape it
        if len(x.shape) == 2 and len(self.input_shape) == 3:
            x_reshaped_for_main_path = x.view(-1, *self.input_shape)
        else:
            x_reshaped_for_main_path = x

        # Pass input through the main path
        main_path_output = x_reshaped_for_main_path
        for layer in self.main_path:
            main_path_output = layer(main_path_output)

        # Pass original input through the shortcut
        # Ensure x is in the correct shape for the shortcut (B, C, H, W) or (B, D)
        if len(x.shape) == 2 and len(self.input_shape) == 3:
            x_reshaped_for_shortcut = x.view(-1, *self.input_shape)
        else:
            x_reshaped_for_shortcut = x

        shortcut_output = self.shortcut(x_reshaped_for_shortcut)

        # Ensure both outputs have the same shape before addition
        # This is crucial. If main_path_output is 4D and shortcut_output is 4D, they must match.
        # If main_path_output is 2D and shortcut_output is 2D, they must match.
        if main_path_output.shape != shortcut_output.shape:
            # Attempt to reshape shortcut_output to match main_path_output if possible
            if main_path_output.numel() == shortcut_output.numel() and main_path_output.shape[0] == shortcut_output.shape[0]:
                shortcut_output = shortcut_output.view_as(main_path_output)
            else:
                raise RuntimeError(f"Shape mismatch for residual addition: main_path_output {main_path_output.shape} vs shortcut_output {shortcut_output.shape}. Cannot reshape to match.")

        return main_path_output + shortcut_output

    def describe(self) -> str:
        main_path_desc = " -> ".join([layer.describe() for layer in self.main_path])
        shortcut_desc = "Conv2d" if isinstance(self.shortcut, nn.Conv2d) else "Linear" if isinstance(self.shortcut, nn.Linear) else "Identity"
        return f"ResidualCompositionLayer(Main Path: {main_path_desc}, Shortcut: {shortcut_desc})"


class PatchwiseCompositionLayer(CompositionLayer):
    """
    A layer that applies a function node to patches of an input tensor.
    This is a generic layer that can be used for both convolutional and
    fully-connected-like operations, depending on the combination_layer.
    """

    def __init__(
        self,
        sub_node: FunctionNode,
        combination_layer: CompositionLayer,
        input_shape: tuple, # (C, H, W)
        patch_size: tuple,
        stride: int = 1,
        padding: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.sub_node = sub_node
        self.combination_layer = combination_layer
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding

        # Calculate output shape and set input_dim/output_dim
        self.output_shape = self._calculate_output_shape()
        self.input_dim = np.prod(self.input_shape)
        self.output_dim = np.prod(self.output_shape)

    def _calculate_output_shape(self):
        c, h, w = self.input_shape
        kh, kw = self.patch_size
        out_h = (h + 2 * self.padding - kh) // self.stride + 1
        out_w = (w + 2 * self.padding - kw) // self.stride + 1
        # The number of output channels is determined by the sub_node
        out_c = self.sub_node.output_dim
        return (out_c, out_h, out_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape flattened input to image format (B, C, H, W)
        x_img = x.view(-1, *self.input_shape)

        # Add padding if specified
        if self.padding > 0:
            x_img = F.pad(x_img, (self.padding, self.padding, self.padding, self.padding))

        # Extract patches using unfold
        # Output shape: (B, C * patch_h * patch_w, num_patches_h * num_patches_w)
        patches = F.unfold(x_img, kernel_size=self.patch_size, stride=self.stride)

        # Reshape patches for the sub_node: (B * num_patches, C * patch_h * patch_w)
        # First, reshape to (B, C*P_h*P_w, N_h*N_w) -> (B, N_h*N_w, C*P_h*P_w)
        patches_reshaped_for_sub_node = patches.transpose(1, 2).contiguous()
        # Then flatten to (B * N_h*N_w, C*P_h*P_w)
        patches_reshaped_for_sub_node = patches_reshaped_for_sub_node.view(-1, self.sub_node.input_dim)

        # Apply sub_node
        sub_node_outputs = self.sub_node(patches_reshaped_for_sub_node)

        # Reshape for combination_layer
        # sub_node_outputs is (B * num_patches, sub_node_output_dim)
        # We need to get back to (B, sub_node_output_dim, num_patches_h, num_patches_w)
        batch_size = x.shape[0]
        num_patches_h = (self.input_shape[1] + 2 * self.padding - self.patch_size[0]) // self.stride + 1
        num_patches_w = (self.input_shape[2] + 2 * self.padding - self.patch_size[1]) // self.stride + 1

        sub_node_outputs_reshaped = sub_node_outputs.view(
            batch_size, num_patches_h, num_patches_w, self.sub_node.output_dim
        )
        # Permute to (B, sub_node_output_dim, num_patches_h, num_patches_w)
        sub_node_outputs_reshaped = sub_node_outputs_reshaped.permute(0, 3, 1, 2).contiguous()

        # Apply combination layer
        return self.combination_layer(sub_node_outputs_reshaped)

    def describe(self) -> str:
        return f"{self.name} (Patchwise): sub_node={self.sub_node.describe()}, combination={self.combination_layer.describe()}"


class FlattenAndConcatenateLayer(CompositionLayer):
    """
    A combination layer that flattens the output of a PatchwiseCompositionLayer.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ReassembleToGridLayer(CompositionLayer):
    """
    A combination layer that reassembles the output of a PatchwiseCompositionLayer
    into a new grid, preserving spatial information.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


