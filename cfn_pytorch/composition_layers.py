

from typing import List, Optional, Union

import torch
import torch.nn as nn

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


