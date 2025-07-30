import numpy as np
import torch

# It's important to handle both NumPy and PyTorch versions of the classes.
# We use try-except blocks to import classes from both frameworks.

try:
    # NumPy implementation imports
    from .CompositionLayerStructure import (
        SequentialCompositionLayer as NumPySequential,
        ParallelCompositionLayer as NumPyParallel,
        ConditionalCompositionLayer as NumPyConditional
    )
    from .FunctionNodes import (
        LinearFunctionNode as NumPyLinear,
        GaussianFunctionNode as NumPyGaussian,
        SigmoidFunctionNode as NumPySigmoid,
        PolynomialFunctionNode as NumPyPolynomial,
        SinusoidalFunctionNode as NumPySinusoidal
    )
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    # PyTorch implementation imports
    from cfn_pytorch.composition_layers import (
        SequentialCompositionLayer as TorchSequential,
        ParallelCompositionLayer as TorchParallel,
        ConditionalCompositionLayer as TorchConditional
    )
    from cfn_pytorch.function_nodes import (
        LinearFunctionNode as TorchLinear,
        GaussianFunctionNode as TorchGaussian,
        SigmoidFunctionNode as TorchSigmoid,
        PolynomialFunctionNode as TorchPolynomial,
        SinusoidalFunctionNode as TorchSinusoidal
    )
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

def interpret_model(network):
    """Interpret the learned functions in the network, supporting both NumPy and PyTorch models."""
    print("\n===== Model Interpretation =====")
    
    is_torch_model = PYTORCH_AVAILABLE and isinstance(network, torch.nn.Module)

    for i, layer in enumerate(network.layers):
        print(f"\nLayer {i+1}: {layer.name}")
        
        if (is_torch_model and isinstance(layer, TorchSequential)) or \
           (NUMPY_AVAILABLE and isinstance(layer, NumPySequential)):
            interpret_sequential_layer(layer, is_torch_model)
        elif (is_torch_model and isinstance(layer, TorchParallel)) or \
             (NUMPY_AVAILABLE and isinstance(layer, NumPyParallel)):
            interpret_parallel_layer(layer, is_torch_model)
        elif (is_torch_model and isinstance(layer, TorchConditional)) or \
             (NUMPY_AVAILABLE and isinstance(layer, NumPyConditional)):
            interpret_conditional_layer(layer, is_torch_model)
    
    print("\n===== End of Interpretation =====")

def interpret_sequential_layer(layer, is_torch_model):
    """Interpret a sequential composition layer."""
    print("  Sequential composition of functions:")
    for i, node in enumerate(layer.function_nodes):
        print(f"  Step {i+1}: {node.describe()}")
        print_node_parameters(node, is_torch_model)

def interpret_parallel_layer(layer, is_torch_model):
    """Interpret a parallel composition layer."""
    print(f"  Parallel composition with {layer.combination} combination:")
    
    if layer.combination == 'weighted_sum':
        weights = layer.weights.detach().cpu().numpy() if is_torch_model else layer.parameters["weights"]
        sorted_indices = np.argsort(-np.abs(weights))
        
        for i, idx in enumerate(sorted_indices):
            node = layer.function_nodes[idx]
            weight = weights[idx]
            print(f"  Component {i+1}: Weight = {weight:.4f}, {node.describe()}")
            if abs(weight) > 0.05:
                print_node_parameters(node, is_torch_model)
    else:
        for i, node in enumerate(layer.function_nodes):
            print(f"  Component {i+1}: {node.describe()}")
            print_node_parameters(node, is_torch_model)

def interpret_conditional_layer(layer, is_torch_model):
    """Interpret a conditional composition layer."""
    print("  Conditional composition (mixture of experts):")
    
    for i, (cond_node, func_node) in enumerate(zip(layer.condition_nodes, layer.function_nodes)):
        print(f"  Region {i+1}:")
        print(f"    Condition: {cond_node.describe()}")
        print(f"    Function: {func_node.describe()}")
        
        print("    Condition parameters:")
        print_node_parameters(cond_node, is_torch_model, indent="      ")
        
        print("    Function parameters:")
        print_node_parameters(func_node, is_torch_model, indent="      ")

def get_param_value(node, param_name, is_torch_model):
    """Helper to get parameter value from either NumPy or PyTorch node."""
    if is_torch_model:
        param = getattr(node, param_name)
        return param.detach().cpu().numpy()
    else:
        return node.parameters[param_name]

def print_node_parameters(node, is_torch_model, indent="    "):
    """Print the important parameters of a function node."""
    node_type = type(node)

    # Check for Linear Node
    if (is_torch_model and node_type is TorchLinear) or (NUMPY_AVAILABLE and node_type is NumPyLinear):
        weights = get_param_value(node, 'weights', is_torch_model)
        bias = get_param_value(node, 'bias', is_torch_model)
        
        if weights.size <= 10:
            print(f"{indent}Weights: {weights.round(3)}")
            print(f"{indent}Bias: {bias.round(3)}")
        else:
            print(f"{indent}Weight stats: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}, "
                  f"min={np.min(weights):.4f}, max={np.max(weights):.4f}")
            flat_weights = weights.flatten()
            indices = np.argsort(-np.abs(flat_weights))[:5]
            print(f"{indent}Top 5 weights by magnitude: {flat_weights[indices].round(3)}")

    # Check for Gaussian Node
    elif (is_torch_model and node_type is TorchGaussian) or (NUMPY_AVAILABLE and node_type is NumPyGaussian):
        center = get_param_value(node, 'center', is_torch_model)
        width = get_param_value(node, 'width', is_torch_model)
        print(f"{indent}Center: {center.round(3)}")
        print(f"{indent}Width: {width.item():.4f}" if is_torch_model else f"{indent}Width: {width:.4f}")

    # Check for Sigmoid Node
    elif (is_torch_model and node_type is TorchSigmoid) or (NUMPY_AVAILABLE and node_type is NumPySigmoid):
        if node.is_elementwise:
            print(f"{indent}Element-wise sigmoid")
            offset = get_param_value(node, 'offset', is_torch_model)
            steepness = get_param_value(node, 'steepness', is_torch_model)
            if offset.size > 10:
                print(f"{indent}  Offset stats: mean={np.mean(offset):.4f}, std={np.std(offset):.4f}")
                print(f"{indent}  Steepness stats: mean={np.mean(steepness):.4f}, std={np.std(steepness):.4f}")
            else:
                print(f"{indent}  Offset: {offset.round(3)}")
                print(f"{indent}  Steepness: {steepness.round(3)}")
        else:
            direction = get_param_value(node, 'direction', is_torch_model)
            offset = get_param_value(node, 'offset', is_torch_model)
            steepness = get_param_value(node, 'steepness', is_torch_model)
            print(f"{indent}Direction: {direction.round(3)}")
            print(f"{indent}Offset: {offset.item():.4f}" if is_torch_model else f"{indent}Offset: {offset:.4f}")
            print(f"{indent}Steepness: {steepness.item():.4f}" if is_torch_model else f"{indent}Steepness: {steepness:.4f}")

    # Check for Polynomial Node
    elif (is_torch_model and node_type is TorchPolynomial) or (NUMPY_AVAILABLE and node_type is NumPyPolynomial):
        coefficients = get_param_value(node, 'coefficients', is_torch_model)
        direction = get_param_value(node, 'direction', is_torch_model)
        print(f"{indent}Coefficients: {coefficients.round(3)}")
        print(f"{indent}Direction: {direction.round(3)}")

    # Check for Sinusoidal Node
    elif (is_torch_model and node_type is TorchSinusoidal) or (NUMPY_AVAILABLE and node_type is NumPySinusoidal):
        frequency = get_param_value(node, 'frequency', is_torch_model)
        amplitude = get_param_value(node, 'amplitude', is_torch_model)
        phase = get_param_value(node, 'phase', is_torch_model)
        direction = get_param_value(node, 'direction', is_torch_model)
        print(f"{indent}Frequency: {frequency.item():.4f}" if is_torch_model else f"{indent}Frequency: {frequency:.4f}")
        print(f"{indent}Amplitude: {amplitude.item():.4f}" if is_torch_model else f"{indent}Amplitude: {amplitude:.4f}")
        print(f"{indent}Phase: {phase.item():.4f}" if is_torch_model else f"{indent}Phase: {phase:.4f}")
        print(f"{indent}Direction: {direction.round(3)}")
