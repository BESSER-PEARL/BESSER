"""
This module provides helper functions to convert BUML code
to PyTorch or TensorFlow code.
"""

import os
import random
from typing import TYPE_CHECKING
from PIL import Image
import numpy as np
from torch import nn


from besser.BUML.metamodel.nn import TensorOp, Layer

if TYPE_CHECKING:
    from besser.generators.nn.nn_code_generator import NNCodeGenerator


def get_previous_out_var(modules_details: dict, prev_module: str, inputs_outputs: dict = None):
    """
    It retrieves the output variable of the previous module in order to
    use it as the input variable of the current module.

    Arguments:
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        prev_module (str): The name of the previous module.
        inputs_outputs (dict): Optional dictionary mapping module names to [input, output] variables.

    Returns:
        The previous output variable.

    """
    if isinstance(modules_details[prev_module], dict):
        return modules_details[prev_module]["in_out_variable"]
    else:
        # Check if this is an RNN with return_type="both" that uses hidden as output
        module_data = modules_details[prev_module]
        if len(module_data) > 4:  # Has hidden variable (element [4])
            layer_obj = module_data[3]
            if (hasattr(layer_obj, 'return_type') and layer_obj.return_type == "both" and
                hasattr(layer_obj, 'use_hidden_as_output') and layer_obj.use_hidden_as_output):
                # Use hidden variable (element [4]) instead of output variable (element [1])
                return module_data[4]
            # Check if there's a __hidden subscript that created a different output variable
            # For example: x = h[-1] creates rnn__hidden entry with output var 'x'
            if (inputs_outputs and hasattr(layer_obj, 'return_type') and
                layer_obj.return_type == "hidden"):
                # Remove the '_layer' suffix properly
                layer_name = prev_module[:-6] if prev_module.endswith('_layer') else prev_module
                hidden_key = layer_name + "__hidden"
                if hidden_key in inputs_outputs and inputs_outputs[hidden_key][1]:
                    # Use the variable from the subscript operation (e.g., 'x' from 'x = h[-1]')
                    return inputs_outputs[hidden_key][1]
        return module_data[1]

def _get_bidirectional_concat_var(lyr_input, modules_details):
    """Get bidirectional concat variable from layer input marker."""
    base_module = lyr_input.replace("bidirectional_concat_", "")
    modules_names = list(modules_details.keys())
    if f"{base_module}_layer" in modules_names:
        layer_details = modules_details[f"{base_module}_layer"]
        if len(layer_details) > 4 and layer_details[4]:
            return layer_details[4]
        return layer_details[1]
    return None


def _get_split_output_var(lyr_input, modules_details):
    """Get specific split output variable from split tensorop."""
    base_op, idx_str = lyr_input.rsplit("__split_", 1)
    idx = int(idx_str)
    modules_names = list(modules_details.keys())
    if f"{base_op}_op" in modules_names:
        op_details = modules_details[f"{base_op}_op"]
        var_list = [v.strip() for v in op_details[1].split(',')]
        if idx < len(var_list):
            return var_list[idx]
    return None


def _get_rnn_state_component_var(lyr_input, suffix_len, index, modules_details, is_cell=False):
    """Get RNN hidden or cell state component for bidirectional RNNs."""
    base_module = lyr_input[:-suffix_len]
    modules_names = list(modules_details.keys())
    if f"{base_module}_layer" in modules_names:
        layer_details = modules_details[f"{base_module}_layer"]
        if len(layer_details) > 4:
            hidden_var = layer_details[4]
            if is_cell:
                cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
                return f"{cell_var}[{index}]"
            return f"{hidden_var}[{index}]"
        return layer_details[1]
    return None


def _get_regular_module_var(lyr_input, layer, modules_details, prev_out_var, inputs_outputs):
    """Get output variable from regular module reference (_layer, _nn, _activ, _op)."""
    modules_names = list(modules_details.keys())

    if f"{lyr_input}_layer" in modules_details:
        layer_details = modules_details[f"{lyr_input}_layer"]
        if (len(layer_details) > 4 and hasattr(layer_details[3], 'return_type') and
            layer_details[3].return_type == "hidden"):
            if (inputs_outputs and f"{lyr_input}__hidden" in inputs_outputs and
                prev_out_var == inputs_outputs[f"{lyr_input}__hidden"][1]):
                return prev_out_var
        if hasattr(layer, 'use_rnn_hidden') and layer.use_rnn_hidden:
            if len(layer_details) > 4:
                return layer_details[4]
        return layer_details[1]

    if f"{lyr_input}_nn" in modules_names:
        nn_details = modules_details[f"{lyr_input}_nn"]
        # Use subnn_output if available (for parallel branches), otherwise in_out_variable
        if "subnn_output" in nn_details:
            return nn_details["subnn_output"]
        return nn_details["in_out_variable"]
    if f"{lyr_input}_activ" in modules_names:
        return modules_details[f"{lyr_input}_activ"][1]
    if f"{lyr_input}_op" in modules_names:
        return modules_details[f"{lyr_input}_op"][1]

    return None


def get_input_var(layer: Layer, modules_details: dict, prev_out_var: str, inputs_outputs: dict = None):
    """
    It determines the input variable of the current layer. It is either
    the output variable of the module in `name_module_input` attribute
    (if it is given), or simply the output of the previous module given
    by `get_previous_out_variable` function.

    Arguments:
        layer (Layer): The layer BUML object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        prev_out_var (str): The previous output variable.
        inputs_outputs (dict): Optional dictionary mapping module names to [input, output] variables.

    Returns:
        The input variable.
    """
    lyr_input = layer.name_module_input
    if lyr_input is None or lyr_input is False:
        return prev_out_var

    if lyr_input == 'INPUT':
        return 'x'

    if not isinstance(lyr_input, str):
        return prev_out_var

    if lyr_input.startswith("bidirectional_concat_"):
        result = _get_bidirectional_concat_var(lyr_input, modules_details)
        if result:
            return result

    if "__split_" in lyr_input:
        result = _get_split_output_var(lyr_input, modules_details)
        if result:
            return result

    if lyr_input.endswith("__hidden_forward"):
        result = _get_rnn_state_component_var(lyr_input, 16, -2, modules_details)
        if result:
            return result
    elif lyr_input.endswith("__hidden_backward"):
        result = _get_rnn_state_component_var(lyr_input, 17, -1, modules_details)
        if result:
            return result
    elif lyr_input.endswith("__cell_forward"):
        result = _get_rnn_state_component_var(lyr_input, 14, -2, modules_details, is_cell=True)
        if result:
            return result
    elif lyr_input.endswith("__cell_backward"):
        result = _get_rnn_state_component_var(lyr_input, 15, -1, modules_details, is_cell=True)
        if result:
            return result
    elif lyr_input.endswith("__hidden"):
        base_module = lyr_input[:-8]
        if inputs_outputs and lyr_input in inputs_outputs and inputs_outputs[lyr_input][1]:
            return inputs_outputs[lyr_input][1]
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return layer_details[4]
            return layer_details[1]
    elif lyr_input.endswith("__cell"):
        base_module = lyr_input[:-6]
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return layer_details[4]
            return layer_details[1]

    result = _get_regular_module_var(lyr_input, layer, modules_details, prev_out_var, inputs_outputs)
    if result:
        return result

    return prev_out_var

def add_in_out_var_to_subnn(modules_details: dict, inputs_outputs: dict | None = None):
    """
    It sets the in_out_variable of subnns, which refers to the input
    and output variable of the subnn.

    Arguments:
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        inputs_outputs (dict | None): Optional dictionary mapping module names to [input, output] variables.

    Returns:
        None, but stores the in_out_var in modules_details dict.

    """
    last_module = list(modules_details.keys())[-1]

    # Extract the base module name (remove _nn suffix)
    # e.g., "encoder_nn" -> "encoder"
    if last_module.endswith("_nn"):
        base_module_name = last_module[:-3]  # Remove "_nn"
        # Check inputs_outputs for the original input/output variables
        if inputs_outputs and base_module_name in inputs_outputs:
            input_var = inputs_outputs[base_module_name][0]
            output_var = inputs_outputs[base_module_name][1]
            # Store both input and output separately
            modules_details[last_module]["subnn_input"] = input_var
            modules_details[last_module]["subnn_output"] = output_var
            # Keep in_out_variable for backward compatibility (sequential case where input==output)
            modules_details[last_module]["in_out_variable"] = output_var
            return

    if len(modules_details) == 1:
        in_out_var = "x"
    else:
        prev_module_details = list(modules_details.values())[-2]
        if isinstance(prev_module_details, dict):
            in_out_var = prev_module_details["in_out_variable"]
        else:
            # get the output variable of the layer or tns before the sub_nn
            in_out_var = prev_module_details[1]
    modules_details[last_module]["in_out_variable"] = in_out_var


def _handle_inline_call(layer_name):
    """Handle inline layer calls (INLINE_CALL:layer_name:input_var)."""
    parts = layer_name.split(":")
    inline_layer_name = parts[1]
    inline_input_var = parts[2]
    return f"self.{inline_layer_name}({inline_input_var})"


def _handle_rnn_hidden_suffix(layer_name, modules_details):
    """Handle RNN __hidden suffix."""
    base_layer = layer_name[:-8]  # Remove '__hidden' suffix
    my_keys = list(modules_details.keys())
    if base_layer + "_layer" in my_keys:
        layer_details = modules_details[base_layer + "_layer"]
        if len(layer_details) > 4:
            return layer_details[4]
        else:
            return layer_details[1]
    else:
        return "x"


def _handle_rnn_cell_suffix(layer_name, modules_details):
    """Handle RNN __cell suffix."""
    base_layer = layer_name[:-6]
    my_keys = list(modules_details.keys())
    if base_layer + "_layer" in my_keys:
        layer_details = modules_details[base_layer + "_layer"]
        if len(layer_details) > 4:
            return layer_details[4]
        else:
            return layer_details[1]
    else:
        return "x"


def _handle_rnn_hidden_forward_suffix(layer_name, modules_details):
    """Handle RNN __hidden_forward suffix for bidirectional RNNs."""
    base_layer = layer_name[:-16]  # Remove "__hidden_forward"
    my_keys = list(modules_details.keys())
    if base_layer + "_layer" in my_keys:
        layer_details = modules_details[base_layer + "_layer"]
        if len(layer_details) > 4:
            return f"{layer_details[4]}[-2]"  # Forward component
        else:
            return layer_details[1]
    else:
        return "x"


def _handle_rnn_hidden_backward_suffix(layer_name, modules_details):
    """Handle RNN __hidden_backward suffix for bidirectional RNNs."""
    base_layer = layer_name[:-17]  # Remove "__hidden_backward"
    my_keys = list(modules_details.keys())
    if base_layer + "_layer" in my_keys:
        layer_details = modules_details[base_layer + "_layer"]
        if len(layer_details) > 4:
            return f"{layer_details[4]}[-1]"  # Backward component
        else:
            return layer_details[1]
    else:
        return "x"


def _handle_rnn_cell_forward_suffix(layer_name, modules_details):
    """Handle RNN __cell_forward suffix for bidirectional LSTMs."""
    base_layer = layer_name[:-14]  # Remove "__cell_forward"
    my_keys = list(modules_details.keys())
    if base_layer + "_layer" in my_keys:
        layer_details = modules_details[base_layer + "_layer"]
        if len(layer_details) > 4:
            hidden_var = layer_details[4]
            # Cell variable follows hidden variable naming
            cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
            return f"{cell_var}[-2]"  # Forward component
        else:
            return layer_details[1]
    else:
        return "x"


def _handle_rnn_cell_backward_suffix(layer_name, modules_details):
    """Handle RNN __cell_backward suffix for bidirectional LSTMs."""
    base_layer = layer_name[:-15]  # Remove "__cell_backward"
    my_keys = list(modules_details.keys())
    if base_layer + "_layer" in my_keys:
        layer_details = modules_details[base_layer + "_layer"]
        if len(layer_details) > 4:
            hidden_var = layer_details[4]
            # Cell variable follows hidden variable naming
            cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
            return f"{cell_var}[-1]"  # Backward component
        else:
            return layer_details[1]
    else:
        return "x"


def _handle_bidirectional_forward(layer_name):
    """Handle bidirectional RNN __forward suffix."""
    base_name = layer_name.rsplit("__forward", 1)[0]
    return f"{base_name}_forward_h"


def _handle_bidirectional_backward(layer_name):
    """Handle bidirectional RNN __backward suffix."""
    base_name = layer_name.rsplit("__backward", 1)[0]
    return f"{base_name}_backward_h"


def _handle_split_output(layer_name, modules_details):
    """Handle split tensorop output with __split_N suffix."""
    # Extract base op name and index: "op_1__split_0" -> ("op_1", 0)
    base_name, idx_str = layer_name.rsplit("__split_", 1)
    idx = int(idx_str)

    # Get the split tensorop from modules_details
    if base_name + "_op" in modules_details:
        # The out_var is a tuple like "x_2_0, x_2_1, x_2_2"
        # Extract the specific variable at index idx
        out_var_tuple = modules_details[base_name + "_op"][1]
        var_list = [v.strip() for v in out_var_tuple.split(',')]
        if idx < len(var_list):
            return var_list[idx]
        else:
            return f"{base_name}_{idx}"
    else:
        # Fallback if tensorop not found
        return f"{base_name}_{idx}"


def _handle_regular_layer(layer_name, modules_details, i, actual_vars):
    """Handle regular layer with optional RNN actual_vars."""
    layer_details = modules_details[layer_name + "_layer"]
    if len(layer_details) > 4 and actual_vars and i < len(actual_vars):
        if actual_vars[i] == "hidden":
            return layer_details[4]
        else:
            out_var = layer_details[1]
            return f"{out_var}[:, -1, :]"
    else:
        return layer_details[1]


def _handle_activation_layer(layer_name, modules_details):
    """Handle standalone activation layer."""
    activ_details = modules_details[layer_name + "_activ"]
    return activ_details[1]


def _handle_subnetwork(layer_name, modules_details):
    """Handle sub-network (Sequential)."""
    my_keys = list(modules_details.keys())
    nn_key = next(k for k in my_keys if k.startswith(layer_name + "_") and k.endswith("_nn"))
    nn_details = modules_details[nn_key]
    return nn_details["in_out_variable"]


def _handle_bidirectional_concat(layer_name, modules_details):
    """Handle bidirectional RNN concat marker."""
    base_layer = layer_name.replace("bidirectional_concat_", "")
    layer_key = base_layer + "_layer"

    if layer_key in modules_details:
        layer_details = modules_details[layer_key]
        # If module_details[4] exists, it's the hidden state variable
        # which should already be set to the original concat variable name (e.g., "gru_concat")
        if len(layer_details) > 4 and layer_details[4]:
            return layer_details[4]
        layer_obj = layer_details[3] if len(layer_details) > 3 else None
        if layer_obj and hasattr(layer_obj, 'return_type'):
            if layer_obj.return_type == "hidden":
                return layer_details[1]
        if len(layer_details) > 1:
            return layer_details[1]
    return "x"


def _handle_tensorop(layer_name, modules_details):
    """Handle tensorop output."""
    return modules_details[layer_name + "_op"][1]


def get_layers_output_for_tensorops(layers_names: list, modules_details: dict,
                                     actual_vars: list = None):
    """
    It retrieves the output variables of the layers in `layers_name`
    list to use them as input of the tensorop.

    Arguments:
        layers_names (list): Names of layers on which the tensorop is applied.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        actual_vars (list): Component types ("output" or "hidden") for RNN layers.

    Returns:
        The output variables of the layers in 'layers_names'.

    """
    my_keys = list(modules_details.keys())
    out_vars = []

    for i, layer_name in enumerate(layers_names):
        # Handle inline layer calls
        if isinstance(layer_name, str) and layer_name.startswith("INLINE_CALL:"):
            out_vars.append(_handle_inline_call(layer_name))
        # Handle INPUT marker
        elif layer_name == 'INPUT':
            out_vars.append('inp')
        # Handle numeric constants
        elif isinstance(layer_name, (int, float)):
            out_vars.append(str(layer_name))
        # Handle RNN hidden/cell state suffixes
        elif isinstance(layer_name, str) and layer_name.endswith("__hidden_forward"):
            out_vars.append(_handle_rnn_hidden_forward_suffix(layer_name, modules_details))
        elif isinstance(layer_name, str) and layer_name.endswith("__hidden_backward"):
            out_vars.append(_handle_rnn_hidden_backward_suffix(layer_name, modules_details))
        elif isinstance(layer_name, str) and layer_name.endswith("__cell_forward"):
            out_vars.append(_handle_rnn_cell_forward_suffix(layer_name, modules_details))
        elif isinstance(layer_name, str) and layer_name.endswith("__cell_backward"):
            out_vars.append(_handle_rnn_cell_backward_suffix(layer_name, modules_details))
        elif isinstance(layer_name, str) and layer_name.endswith("__hidden"):
            out_vars.append(_handle_rnn_hidden_suffix(layer_name, modules_details))
        elif isinstance(layer_name, str) and layer_name.endswith("__cell"):
            out_vars.append(_handle_rnn_cell_suffix(layer_name, modules_details))
        # Handle split tensorop output suffixes
        elif isinstance(layer_name, str) and "__split_" in layer_name:
            out_vars.append(_handle_split_output(layer_name, modules_details))
        # Handle bidirectional RNN forward/backward suffixes
        elif isinstance(layer_name, str) and layer_name.endswith("__forward"):
            out_vars.append(_handle_bidirectional_forward(layer_name))
        elif isinstance(layer_name, str) and layer_name.endswith("__backward"):
            out_vars.append(_handle_bidirectional_backward(layer_name))
        # Handle regular layer
        elif layer_name + "_layer" in my_keys:
            out_vars.append(_handle_regular_layer(layer_name, modules_details, i, actual_vars))
        # Handle standalone activation layer
        elif layer_name + "_activ" in my_keys:
            out_vars.append(_handle_activation_layer(layer_name, modules_details))
        # Handle sub-network (Sequential)
        elif any(k.startswith(layer_name + "_") and k.endswith("_nn") for k in my_keys):
            out_vars.append(_handle_subnetwork(layer_name, modules_details))
        # Handle bidirectional concat marker
        elif layer_name.startswith("bidirectional_concat_"):
            out_vars.append(_handle_bidirectional_concat(layer_name, modules_details))
        # Handle tensorop
        else:
            out_vars.append(_handle_tensorop(layer_name, modules_details))

    return out_vars

def initialize_tensorop_var(tensorop: TensorOp):
    """
    It sets the output variable of the tensorop in the case it is the
    first module in the neural network.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.

    Returns:
        The output variable of the tensorop.

    """
    # Identity tensorops should use their name as the output variable
    if tensorop.tns_type == "identity":
        return tensorop.name
    elif tensorop.input_reused is True:
        out_var = "x_1"
    else:
        out_var = "x"
    return out_var


def _collect_used_x_numbers(modules_details):
    """Collect all used x_N variable numbers from modules_details."""
    used_nums = set()
    for module_details in modules_details.values():
        if isinstance(module_details, list):
            for idx in [1, 2]:  # Check both output (1) and input (2) variables
                if len(module_details) > idx:
                    var = module_details[idx]
                    if isinstance(var, str) and var.startswith('x_'):
                        try:
                            num = int(var.split('_')[1])
                            used_nums.add(num)
                        except (ValueError, IndexError):
                            pass
    return used_nums


def _find_next_available_x_number(used_nums):
    """Find the first available x_N number."""
    num = 1
    while num in used_nums:
        num += 1
    return num


def get_out_var_input_reused(prev_out_var: str, modules_details: dict = None):
    """
    It sets the output variable of the module in the case the output
    of the previous module is reused (therefore, they need to be
    different).

    Arguments:
        prev_out_var (str): The previous output variable.
        modules_details (dict): Dict of existing modules to find next available var.
    Returns:
        The current output variable.

    """
    if prev_out_var == "x":
        return "x_1"

    if modules_details:
        used_nums = _collect_used_x_numbers(modules_details)
        num = _find_next_available_x_number(used_nums)
        return f"x_{num}"

    parts = prev_out_var.split('_')
    try:
        num = int(parts[-1])
        return f"x_{num + 1}"
    except ValueError:
        return "x_1"


def get_layer_vars(layer: Layer, prev_out_var: str, modules_details: dict, inputs_outputs: dict | None = None):
    """
    It sets the input and output variables of the layer.

    Arguments:
        layer (Layer): The BUML layer object.
        prev_out_var (str): The previous output variable.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        inputs_outputs (dict | None): Dictionary mapping module names to [input_var, output_var].

    Returns:
        - The input variable and output variables of both the layer and
          its activation function.

    """
    out_var_actv, in_var_actv = None, None
    in_var_layer = get_input_var(layer, modules_details, prev_out_var, inputs_outputs)

    # Check if inputs_outputs has the original output variable from the code
    if inputs_outputs and layer.name in inputs_outputs:
        out_var_layer = inputs_outputs[layer.name][1]
    elif layer.input_reused:
        out_var_layer = get_out_var_input_reused(prev_out_var, modules_details)
    else:
        # If input != prev_out_var (e.g., input='x' from INPUT, prev='residual'),
        # use input var as output to preserve original code pattern (x = lstm(x))
        if in_var_layer != prev_out_var:
            out_var_layer = in_var_layer
        else:
            out_var_layer = prev_out_var
    if layer.actv_func is not None:
        out_var_actv, in_var_actv = out_var_layer, out_var_layer
    return out_var_layer, in_var_layer, out_var_actv, in_var_actv

def initialize_layer_vars(layer: Layer, inputs_outputs: dict | None = None):
    """
    It sets the input and output variables of layer (and activation
    function for PyTorch) in the case it is the first module in
    the neural network.

    Arguments:
        layer (Layer): The BUML layer object.
        inputs_outputs (dict | None): Dictionary mapping module names to [input_var, output_var].

    Returns:
        - The input variable and output variables of both the layer and
          its activation function.
    """
    out_var_actv, in_var_actv = None, None

    # Check if inputs_outputs has the original output variable from the code
    if inputs_outputs and layer.name in inputs_outputs:
        out_var_layer = inputs_outputs[layer.name][1]
        in_var_layer = inputs_outputs[layer.name][0] if inputs_outputs[layer.name][0] else "x"
        # BUGFIX: Set activation variables when layer has activation function
        if layer.actv_func is not None:
            out_var_actv, in_var_actv = out_var_layer, out_var_layer
    elif layer.input_reused is True:
        out_var_layer, in_var_layer = "x_1", "x"
        if layer.actv_func is not None:
            out_var_actv, in_var_actv = "x_1", "x_1"
    else:
        out_var_layer, in_var_layer = "x", "x"
        if layer.actv_func is not None:
            out_var_actv, in_var_actv = "x", "x"
    return out_var_layer, in_var_layer, out_var_actv, in_var_actv



def get_layer_syntax(setup_layer_cls: 'NNCodeGenerator',
                     layer: Layer, modules_details: dict,
                     actv_func_synt: str | bool,
                     out_var: str = None, in_var: str = None,
                     is_subnn: bool = False):
    """
    It retrieves the syntax of the layer (and the activation
    function in the case of PyTorch) from the ´setup_layer_cls´ class.

    Arguments:
        setup_layer_cls (NNCodeGenerator): The class that
        constructs the syntax of layers.
        layer (Layer): The BUML layer object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        actv_func_synt (str | bool): Whether to get the syntax of
            the actvation function.
        out_var (str): Output variable for standalone activations.
        in_var (str): Input variable for standalone activations.
        is_subnn (bool): if the layer is inside a subnn model.

    Returns:
        The syntax of the layer and its activation function (if relevant) and
        the class instance.

    """
    setup = setup_layer_cls(layer, modules_details, is_subnn=is_subnn)
    parent_class = layer.__class__.mro()[1].__name__
    cls_name = layer.__class__.__name__

    if (parent_class == "ConvolutionalLayer" or parent_class == "CNN"):
        layer_synt = setup.setup_cnn()
    elif parent_class == "RNN":
        layer_synt = setup.setup_rnn()
    elif parent_class == "GeneralLayer":
        layer_synt = setup.setup_general_layer()
    elif cls_name == "GeneralLayer":
        # Standalone activation layer - don't call setup_actv_func since the layer itself IS the activation
        layer_synt = setup.setup_standalone_activation(out_var, in_var)
        actv_func_synt = False
    else: #(parent_class == "LayerModifier" or
           #parent_class == "NormalizationLayer")
        layer_synt = setup.setup_layer_modifier()

    if actv_func_synt:
        actv_func_synt = setup.setup_actv_func()

    return layer_synt, actv_func_synt, setup

def _is_shape_dim_operation(module_name: str, modules_details: dict) -> bool:
    """
    Check if a module is a shape_dim operation (side operation that extracts
    shape dimensions but doesn't produce tensors for the main flow).

    Args:
        module_name: Name of the module to check
        modules_details: Dictionary of module details

    Returns:
        True if the module is a shape_dim operation
    """
    if not module_name.endswith("_op"):
        return False

    # Get the tensorop syntax from modules_details
    if module_name in modules_details:
        syntax = modules_details[module_name][0]
        # Check if it's a shape extraction operation
        # TensorFlow: "tf.shape(x)[0]"
        # PyTorch: "x.size(0)"
        return "tf.shape(" in syntax or ".size(" in syntax
    return False

def _find_previous_non_shape_dim_module(modules_details: dict) -> str:
    """
    Find the previous module that is not a shape_dim operation.

    Args:
        modules_details: Dictionary of module details

    Returns:
        Name of the previous non-shape_dim module, or None if not found
    """
    module_keys = list(modules_details.keys())
    # Start from the end and work backwards
    for i in range(len(module_keys) - 1, -1, -1):
        module_name = module_keys[i]
        if not _is_shape_dim_operation(module_name, modules_details):
            return module_name
    return None

def _initialize_layer_variables(layer, modules_details, inputs_outputs=None):
    """Initialize layer input/output variables based on modules_details."""
    if len(modules_details) == 0:
        return initialize_layer_vars(layer, inputs_outputs), None

    prev_module = list(modules_details.keys())[-1]

    # Skip shape_dim operations to find actual previous tensor-producing module
    if _is_shape_dim_operation(prev_module, modules_details):
        actual_prev = _find_previous_non_shape_dim_module(modules_details)
        if actual_prev:
            prev_module = actual_prev
        else:
            return initialize_layer_vars(layer), None

    if prev_module is not None:
        prev_out_var = get_previous_out_var(modules_details, prev_module, inputs_outputs)
        return get_layer_vars(layer, prev_out_var, modules_details, inputs_outputs), prev_module

    return initialize_layer_vars(layer, inputs_outputs), None


def _add_input_permute_if_needed(setup, channel_last, layer, in_layer, is_seq, is_subnn):
    """Add input permute for PyTorch generation if needed."""
    if setup.permute_in and hasattr(setup, 'add_permute') and channel_last:
        dim = setup.dim
        setup.add_permute(
            layer.name, dim, in_layer, permute_in=True,
            sequential=is_seq, is_subnn=is_subnn
        )


def _get_unique_layer_key(layer_name, modules_details):
    """Generate unique key for layer reuse."""
    base_key = layer_name + "_layer"
    unique_key = base_key
    use_counter = 1
    while unique_key in modules_details:
        unique_key = f"{base_key}_use_{use_counter}"
        use_counter += 1
    return unique_key


def _get_rnn_hidden_var_name(layer, out_layer, model):
    """Get RNN hidden state variable name from model or generate default."""
    concat_var_names = getattr(model, 'bidirectional_concat_var_names', {}) if model else {}
    inputs_outputs = getattr(model, 'inputs_outputs', {}) if model else {}

    if layer.name in concat_var_names:
        return concat_var_names[layer.name]

    if inputs_outputs and (layer.name + "__hidden") in inputs_outputs:
        hidden_var = inputs_outputs[layer.name + "__hidden"][0]
        if hidden_var is None:
            return f"{out_layer}_h" if out_layer != "x" else "h"
        return hidden_var

    return f"{out_layer}_h" if out_layer != "x" else "h"


def _get_lstm_cell_var_name(layer, hidden_var, model):
    """Get LSTM cell state variable name from model or generate default."""
    inputs_outputs = getattr(model, 'inputs_outputs', {}) if model else {}

    if inputs_outputs and (layer.name + "__cell") in inputs_outputs:
        cell_var = inputs_outputs[layer.name + "__cell"][0]
        if cell_var is None or cell_var == "_":
            return f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
        return cell_var

    return f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var


def _store_layer_in_modules_details(layer, layer_synt, out_layer, in_layer, modules_details, model=None):
    """Store layer syntax and variables in modules_details."""
    if layer_synt is None:
        return

    unique_key = _get_unique_layer_key(layer.name, modules_details)

    if hasattr(layer, 'return_type') and layer.return_type in ("both", "hidden"):
        hidden_var = _get_rnn_hidden_var_name(layer, out_layer, model)

        if layer.__class__.__name__ == "LSTMLayer":
            cell_var = _get_lstm_cell_var_name(layer, hidden_var, model)
            modules_details[unique_key] = [layer_synt, out_layer, in_layer, layer, hidden_var, cell_var]
        else:
            modules_details[unique_key] = [layer_synt, out_layer, in_layer, layer, hidden_var]
    else:
        modules_details[unique_key] = [layer_synt, out_layer, in_layer, layer]


def _add_output_permute_if_needed(setup, channel_last, layer, out_layer, is_seq, is_subnn):
    """Add output permute for PyTorch generation if needed."""
    if setup.permute_out and hasattr(setup, 'add_permute') and channel_last:
        dim = setup.dim
        setup.add_permute(layer.name, dim, out_layer, permute_in=False,
                          sequential=is_seq, is_subnn=is_subnn)


def handle_layer(layer: Layer, setup_layer: 'NNCodeGenerator',
                 modules_details: dict, channel_last: bool | None,
                 actv_func_syntax: str | bool = False, is_seq: bool = False,
                 is_subnn: bool = False, model=None):
    """
    It populates the `modules_details` dictionary with layer's
    information: Its syntax, input and output variables, and the
    layer class.
    In the case of PyTorch, the activation function is treated as
    a layer.

    Arguments:
        setup_layer_cls (NNCodeGenerator): The class that
        constructs the syntax of layers.
        layer (Layer): The BUML layer object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        actv_func_synt (str | bool): Whether to get the syntax of
            the actvation function.
        is_seq (bool): Whether the model is sequential.
        channel_last (bool, optional): If true, PyTorch conv layers will
            have their input and output permuted to match TF convention.
        is_subnn (bool): if the layer is inside a subnn model.

    Returns:
        None, but stores the layer details in the modules_details dict.

    """
    inputs_outputs = getattr(model, 'inputs_outputs', {}) if model else {}
    (out_layer, in_layer, out_actv, in_actv), prev_module = _initialize_layer_variables(
        layer, modules_details, inputs_outputs
    )

    layer_synt, actv_func_syntax, setup = get_layer_syntax(
        setup_layer, layer, modules_details, actv_func_syntax, out_layer, in_layer,
        is_subnn=is_subnn
    )

    _add_input_permute_if_needed(setup, channel_last, layer, in_layer, is_seq, is_subnn)

    _store_layer_in_modules_details(layer, layer_synt, out_layer, in_layer, modules_details, model)

    if actv_func_syntax:
        modules_details[layer.name + "_activ"] = [actv_func_syntax, out_actv, in_actv]

    # TF-specific: Add separate activation for BatchNorm/LayerNorm
    if hasattr(setup, 'add_separate_activation_if_needed'):
        setup.add_separate_activation_if_needed(out_actv, in_actv)

    _add_output_permute_if_needed(setup, channel_last, layer, out_layer, is_seq, is_subnn)


# ============================================================================
# Helper functions for get_tensorop_params (refactored for maintainability)
# ============================================================================

def _get_bidirectional_hidden_component(source_layer, suffix_len, index, modules_details):
    """Get bidirectional RNN hidden state component (forward or backward)."""
    base_module = source_layer[:-suffix_len]
    modules_names = list(modules_details.keys())
    if f"{base_module}_layer" in modules_names:
        layer_details = modules_details[f"{base_module}_layer"]
        if len(layer_details) > 4:
            return f"{layer_details[4]}[{index}]"
        return layer_details[1]
    return None


def _get_bidirectional_cell_component(source_layer, suffix_len, index, modules_details):
    """Get bidirectional LSTM cell state component (forward or backward)."""
    base_module = source_layer[:-suffix_len]
    modules_names = list(modules_details.keys())
    if f"{base_module}_layer" in modules_names:
        layer_details = modules_details[f"{base_module}_layer"]
        if len(layer_details) > 4:
            hidden_var = layer_details[4]
            cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
            return f"{cell_var}[{index}]"
        return layer_details[1]
    return None


def _get_rnn_hidden_output(source_layer, modules_details):
    """Get RNN hidden state output variable based on return_type."""
    base_module = source_layer[:-8]
    modules_names = list(modules_details.keys())
    if f"{base_module}_layer" in modules_names:
        layer_details = modules_details[f"{base_module}_layer"]
        layer_obj = layer_details[3] if len(layer_details) > 3 else None
        if layer_obj and hasattr(layer_obj, 'return_type'):
            if layer_obj.return_type == "hidden":
                if len(layer_details) > 4:
                    return layer_details[4]
                return layer_details[1]
            elif layer_obj.return_type in ("both", "full") and len(layer_details) > 4:
                return layer_details[4]
        return layer_details[1]
    return None


def _get_lstm_cell_output(source_layer, modules_details):
    """Get LSTM cell state output variable."""
    base_module = source_layer[:-6]
    modules_names = list(modules_details.keys())
    if f"{base_module}_layer" in modules_names:
        layer_details = modules_details[f"{base_module}_layer"]
        if len(layer_details) > 5:
            return layer_details[5]
        elif len(layer_details) > 4:
            return f"{layer_details[4]}_cell"
        return layer_details[1]
    return None


def _get_split_indexed_output(source_layer, modules_details):
    """Get specific indexed output from split operation."""
    base_op, idx_str = source_layer.rsplit("__split_", 1)
    idx = int(idx_str)
    if f"{base_op}_op" in modules_details:
        out_var_tuple = modules_details[f"{base_op}_op"][1]
        var_list = [v.strip() for v in out_var_tuple.split(',')]
        if idx < len(var_list):
            return var_list[idx]
        return f"{base_op}_{idx}"
    return None


def _get_bidirectional_concat_output(source_layer, modules_details):
    """Get bidirectional RNN concatenated output variable."""
    base_layer = source_layer.replace("bidirectional_concat_", "")
    layer_key = f"{base_layer}_layer"
    if layer_key in modules_details:
        layer_details = modules_details[layer_key]
        if len(layer_details) > 4 and layer_details[4]:
            return layer_details[4]
        return layer_details[1]
    return None


def _resolve_source_layer_var(source_layer, modules_details, inputs_outputs=None, tensorop=None):
    """Resolve a source layer name to its output variable."""
    if source_layer == 'INPUT':
        if inputs_outputs and tensorop and hasattr(tensorop, 'name') and tensorop.name in inputs_outputs:
            actual_input_var = inputs_outputs[tensorop.name][0]
            if actual_input_var:
                return actual_input_var
        return 'inp'

    if not isinstance(source_layer, str):
        return source_layer

    if source_layer.endswith("__hidden_forward"):
        result = _get_bidirectional_hidden_component(source_layer, 16, -2, modules_details)
        if result:
            return result
    elif source_layer.endswith("__hidden_backward"):
        result = _get_bidirectional_hidden_component(source_layer, 17, -1, modules_details)
        if result:
            return result
    elif source_layer.endswith("__cell_forward"):
        result = _get_bidirectional_cell_component(source_layer, 14, -2, modules_details)
        if result:
            return result
    elif source_layer.endswith("__cell_backward"):
        result = _get_bidirectional_cell_component(source_layer, 15, -1, modules_details)
        if result:
            return result
    elif source_layer.endswith("__hidden"):
        result = _get_rnn_hidden_output(source_layer, modules_details)
        if result:
            return result
    elif source_layer.endswith("__cell"):
        result = _get_lstm_cell_output(source_layer, modules_details)
        if result:
            return result

    if "__split_" in source_layer:
        result = _get_split_indexed_output(source_layer, modules_details)
        if result:
            return result

    if source_layer.startswith("bidirectional_concat_"):
        result = _get_bidirectional_concat_output(source_layer, modules_details)
        if result:
            return result

    modules_names = list(modules_details.keys())
    for suffix in ['_layer', '_op', '_activ', '_nn']:
        key = f"{source_layer}{suffix}"
        if key in modules_names:
            if suffix == '_nn':
                return modules_details[key]["in_out_variable"]
            return modules_details[key][1]

    return 'inp' if source_layer == 'x' else source_layer


def _get_rnn_state_var(base_module, modules_details, get_rnn_hidden_var_fn):
    """Get RNN hidden/cell state variable."""
    modules_names = list(modules_details.keys())
    layer_key = f"{base_module}_layer"

    if layer_key in modules_names:
        layer_details = modules_details[layer_key]
        if get_rnn_hidden_var_fn:
            return get_rnn_hidden_var_fn(layer_details, base_module)
        if len(layer_details) > 4:
            return layer_details[4]
        return layer_details[1]
    return "x"


def _resolve_prev_out_var_from_module_input(
    module_input, modules_details, get_rnn_hidden_var_fn, inputs_outputs=None, tensorop=None
):
    """Resolve prev_out_var from tensorop's name_module_input or layers_of_tensors."""
    if module_input == 'INPUT':
        if inputs_outputs and tensorop and hasattr(tensorop, 'name') and tensorop.name in inputs_outputs:
            actual_input_var = inputs_outputs[tensorop.name][0]
            if actual_input_var:
                return actual_input_var
        return 'inp'

    if module_input.endswith("__hidden_forward"):
        result = _get_bidirectional_hidden_component(module_input, 16, -2, modules_details)
        return result if result else "x"

    if module_input.endswith("__hidden_backward"):
        result = _get_bidirectional_hidden_component(module_input, 17, -1, modules_details)
        return result if result else "x"

    if module_input.endswith("__cell_forward"):
        result = _get_bidirectional_cell_component(module_input, 14, -2, modules_details)
        return result if result else "x"

    if module_input.endswith("__cell_backward"):
        result = _get_bidirectional_cell_component(module_input, 15, -1, modules_details)
        return result if result else "x"

    if module_input.endswith("__hidden"):
        base_module = module_input[:-8]
        return _get_rnn_state_var(base_module, modules_details, get_rnn_hidden_var_fn)

    if module_input.endswith("__cell"):
        base_module = module_input[:-6]
        return _get_rnn_state_var(base_module, modules_details, get_rnn_hidden_var_fn)

    if module_input.startswith("bidirectional_concat_"):
        result = _get_bidirectional_concat_output(module_input, modules_details)
        return result if result else "x"

    return _resolve_source_layer_var(module_input, modules_details)


def _get_initial_prev_out_var(modules_details, tensorop, get_rnn_hidden_var_fn, inputs_outputs=None):
    """Get the initial prev_out_var before tensorop-specific handling."""
    if len(list(modules_details.keys())) == 0:
        # Check if we have actual input variable in inputs_outputs even when modules_details is empty
        if inputs_outputs and tensorop.name in inputs_outputs:
            actual_input_var = inputs_outputs[tensorop.name][0]
            if actual_input_var:
                return actual_input_var
        return "x"

    prev_module = list(modules_details.keys())[-1]

    # Skip shape_dim operations to find actual previous tensor-producing module
    if _is_shape_dim_operation(prev_module, modules_details):
        actual_prev = _find_previous_non_shape_dim_module(modules_details)
        if actual_prev:
            prev_module = actual_prev

    prev_out_var = get_previous_out_var(modules_details, prev_module)

    # Override with name_module_input or layers_of_tensors if available
    module_input = None
    if hasattr(tensorop, 'name_module_input') and tensorop.name_module_input is not None:
        module_input = tensorop.name_module_input
    elif hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
        if len(tensorop.layers_of_tensors) == 1:
            module_input = tensorop.layers_of_tensors[0]
        elif 'INPUT' in tensorop.layers_of_tensors:
            module_input = 'INPUT'

    if module_input is not None:
        prev_out_var = _resolve_prev_out_var_from_module_input(
            module_input, modules_details, get_rnn_hidden_var_fn, inputs_outputs, tensorop
        )

    return prev_out_var


def _override_prev_out_var_if_needed(
    tensorop, modules_details, current_prev_out_var, check_name_module_input=True, inputs_outputs=None
):
    """Override prev_out_var from layers_of_tensors if present."""
    # Skip if name_module_input is already set (unless told not to check)
    if (check_name_module_input and
        hasattr(tensorop, 'name_module_input') and
        tensorop.name_module_input is not None):
        return current_prev_out_var

    if (hasattr(tensorop, 'layers_of_tensors') and
        tensorop.layers_of_tensors and
        isinstance(tensorop.layers_of_tensors[0], str)):
        source_layer = tensorop.layers_of_tensors[0]
        return _resolve_source_layer_var(source_layer, modules_details, inputs_outputs, tensorop)

    return current_prev_out_var


def _handle_reshape_params(tensorop, modules_details):
    """Handle reshape tensorop parameters."""
    prev_out_var = None

    # Override prev_out_var if layers_of_tensors is set and not INPUT
    if (hasattr(tensorop, 'layers_of_tensors') and
        tensorop.layers_of_tensors and
        tensorop.layers_of_tensors[0] != 'INPUT'):
        source_layer = tensorop.layers_of_tensors[0]
        prev_out_var = _resolve_source_layer_var(source_layer, modules_details)

    # Resolve operation names to their actual variable names
    resolved_dims = []
    for dim in tensorop.reshape_dim:
        if isinstance(dim, str) and f"{dim}_op" in modules_details:
            resolved_dims.append(modules_details[f"{dim}_op"][1])
        else:
            resolved_dims.append(str(dim))

    params = ', '.join(resolved_dims)
    return prev_out_var, params


def _handle_concatenate_params(tensorop, modules_details):
    """Handle concatenate tensorop parameters."""
    actual_vars = getattr(tensorop, 'actual_vars', None)
    tensors = get_layers_output_for_tensorops(
        tensorop.layers_of_tensors, modules_details, actual_vars
    )
    # Filter out None values to prevent join errors (should not happen with proper layer handling)
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        raise ValueError(f"Concatenate operation '{tensorop.name}' has no valid input tensors")
    params = ', '.join(tensors)
    return None, params


def _handle_transpose_params(tensorop, modules_details):
    """Handle transpose tensorop parameters."""
    prev_out_var = _override_prev_out_var_if_needed(
        tensorop, modules_details, None, check_name_module_input=False
    )
    params = ", ".join([str(i) for i in tensorop.transpose_dim])
    return prev_out_var, params


def _handle_permute_params(tensorop, modules_details):
    """Handle permute tensorop parameters."""
    params = ", ".join([str(i) for i in tensorop.permute_dim])
    return None, params


def _handle_simple_op_params(tensorop, modules_details, inputs_outputs=None):
    """Handle simple ops (mean, max, squeeze, unsqueeze, normalize, shape_dim, subscript)."""
    prev_out_var = _override_prev_out_var_if_needed(
        tensorop, modules_details, None, check_name_module_input=False, inputs_outputs=inputs_outputs
    )
    return prev_out_var, ""


def _handle_repeat_params(tensorop, modules_details, inputs_outputs=None):
    """Handle repeat tensorop parameters."""
    prev_out_var = _override_prev_out_var_if_needed(
        tensorop, modules_details, None, check_name_module_input=False, inputs_outputs=inputs_outputs
    )

    # Resolve operation names in repeat_dim (similar to reshape_dim)
    resolved_multiples = []
    for mult in tensorop.repeat_dim:
        if isinstance(mult, str) and f"{mult}_op" in modules_details:
            resolved_multiples.append(modules_details[f"{mult}_op"][1])
        else:
            resolved_multiples.append(str(mult))

    params = ', '.join(resolved_multiples)
    return prev_out_var, params


def _handle_generic_params(tensorop, modules_details):
    """Handle generic tensorop parameters using layers_of_tensors."""
    tensors = tensorop.layers_of_tensors
    if any(isinstance(t, str) for t in tensors):
        actual_vars = getattr(tensorop, 'actual_vars', None)
        tensors = get_layers_output_for_tensorops(tensors, modules_details, actual_vars)

    params = ', '.join([str(i) for i in tensors])
    return None, params


# Dispatch table for tensorop types
_TENSOROP_HANDLERS = {
    "reshape": _handle_reshape_params,
    "concatenate": _handle_concatenate_params,
    "transpose": _handle_transpose_params,
    "permute": _handle_permute_params,
    "mean": _handle_simple_op_params,
    "max": _handle_simple_op_params,
    "squeeze": _handle_simple_op_params,
    "unsqueeze": _handle_simple_op_params,
    "subscript": _handle_simple_op_params,
    "shape_dim": _handle_simple_op_params,
    "normalize": _handle_simple_op_params,
    "repeat": _handle_repeat_params,
    "split": _handle_simple_op_params,
    "interpolate": lambda t, m: (None, None),
    "pad": lambda t, m: (None, None),
    "dropout": lambda t, m: (None, None),
}


def get_tensorop_params(tensorop: TensorOp, modules_details: dict, get_rnn_hidden_var_fn=None, inputs_outputs=None):
    """
    It retrieves tensorops parameters that are used by
    `get_tensorop_syntax` function defined in PyTorch and
    TensorFlow `utils.py` files.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        get_rnn_hidden_var_fn (callable): Optional framework-specific function
            to get RNN hidden variable name. Should accept (layer_details, base_module).
        inputs_outputs (dict): Optional dict mapping tensorop names to [input_var, output_var].

    Returns:
        - previous output variable and the parameters of the tensorop.

    """
    prev_out_var = _get_initial_prev_out_var(modules_details, tensorop, get_rnn_hidden_var_fn, inputs_outputs)

    tns_type = tensorop.tns_type
    handler = _TENSOROP_HANDLERS.get(tns_type, _handle_generic_params)

    if tns_type in ["mean", "max", "squeeze", "unsqueeze", "subscript", "shape_dim", "normalize", "split", "repeat"]:
        override_prev_out_var, params = handler(tensorop, modules_details, inputs_outputs)
    else:
        override_prev_out_var, params = handler(tensorop, modules_details)

    if override_prev_out_var is not None:
        prev_out_var = override_prev_out_var

    return prev_out_var, params


def get_tensorop_out_var(tensorop: TensorOp, prev_out_var: str, modules_details: dict = None):
    """
    It sets the output variable of tensorop.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.
        prev_out_var (str): previous output variable.
        modules_details (dict): Dict of existing modules to find next available var.

    Returns:
        - The current output variable.

    """
    if tensorop.input_reused is True:
        out_var  = get_out_var_input_reused(prev_out_var, modules_details)
    else:
        # If prev_out_var doesn't follow the standard x or x_N pattern (e.g., it's 'b', 't', 'last'),
        # don't reuse it - generate a new x_N variable instead
        if prev_out_var != "x" and not (prev_out_var.startswith("x_") and prev_out_var.split('_')[-1].isdigit()):
            # Non-standard variable, generate new x_N
            out_var = get_out_var_input_reused(prev_out_var, modules_details)
        else:
            out_var = prev_out_var
    return out_var

def _determine_tensorop_out_var(tensorop, modules_details, out_var, referenced_tensorops, inputs_outputs):
    """Determine the output variable name for a tensorop."""
    if inputs_outputs and tensorop.name in inputs_outputs and inputs_outputs[tensorop.name][1] is not None:
        return inputs_outputs[tensorop.name][1]

    if out_var is not None:
        return out_var

    if tensorop.tns_type == "shape_dim":
        return tensorop.name

    if len(modules_details) == 0:
        return initialize_tensorop_var(tensorop)

    prev_module = list(modules_details.keys())[-1]
    prev_out_var = get_previous_out_var(modules_details, prev_module)

    if referenced_tensorops and tensorop.name in referenced_tensorops:
        return tensorop.name
    if inputs_outputs and tensorop.name in inputs_outputs and inputs_outputs[tensorop.name][1] is not None:
        return inputs_outputs[tensorop.name][1]
    if tensorop.tns_type in ("split", "identity"):
        return tensorop.name
    if hasattr(tensorop, 'input_reused') and tensorop.input_reused:
        return tensorop.name

    return get_tensorop_out_var(tensorop, prev_out_var, modules_details)


def _handle_skip_syntax(ts_op_synt, out_var, inputs_outputs, tensorop):
    """Handle SKIP: syntax and extract actual variable name."""
    if not (isinstance(ts_op_synt, str) and ts_op_synt.startswith("SKIP:")):
        return out_var

    skip_var = ts_op_synt[5:]
    if not (inputs_outputs and tensorop.name in inputs_outputs and inputs_outputs[tensorop.name][1] is not None):
        return skip_var
    return out_var


def _handle_split_tuple(tensorop, out_var):
    """Generate tuple variable names for split operations."""
    if tensorop.tns_type == "split" and hasattr(tensorop, 'split_sizes'):
        num_splits = tensorop.split_sizes
        split_vars = [f"{out_var}_{i}" for i in range(num_splits)]
        return ", ".join(split_vars)
    return out_var


def _add_tensorop_input_permute(tensorop, modules_details, get_tensorop_syntax):
    """Add input permute for spatial tensorops (NHWC → NCHW)."""
    if hasattr(tensorop, 'permute_in') and tensorop.permute_in and tensorop.tns_type in ("interpolate", "pad"):
        from besser.BUML.metamodel.nn import TensorOp as TensorOpClass

        in_permute_name = f"{tensorop.name}_in_op"
        in_permute = TensorOpClass(name=in_permute_name, tns_type="permute", permute_dim=[0, 3, 1, 2])
        prev_module = list(modules_details.keys())[-1] if modules_details else None
        in_var = get_previous_out_var(modules_details, prev_module) if prev_module else "x"
        handle_tensorop(in_permute, modules_details, get_tensorop_syntax, out_var=in_var, channel_last=False)


def _update_skip_source_layer(ts_op_synt, out_var, inputs_outputs, tensorop, modules_details):
    """Update source layer output variable for SKIP operations with user variable names."""
    if not (isinstance(ts_op_synt, str) and ts_op_synt.startswith("SKIP:")):
        return

    if (inputs_outputs and tensorop.name in inputs_outputs and
        inputs_outputs[tensorop.name][1] is not None and out_var != ts_op_synt[5:]):
        if hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
            source_layer_name = tensorop.layers_of_tensors[0]
            layer_key = source_layer_name + "_layer"
            if layer_key in modules_details:
                modules_details[layer_key][1] = out_var


def _add_tensorop_output_permute(tensorop, modules_details, get_tensorop_syntax, out_var):
    """Add output permute for spatial tensorops (NCHW → NHWC)."""
    if hasattr(tensorop, 'permute_out') and tensorop.permute_out and tensorop.tns_type in ("interpolate", "pad"):
        from besser.BUML.metamodel.nn import TensorOp as TensorOpClass

        out_permute_name = f"{tensorop.name}_out_op"
        out_permute = TensorOpClass(name=out_permute_name, tns_type="permute", permute_dim=[0, 2, 3, 1])
        handle_tensorop(out_permute, modules_details, get_tensorop_syntax, out_var=out_var, channel_last=False)


def handle_tensorop(tensorop: TensorOp, modules_details: dict,
                    get_tensorop_syntax: callable, out_var: str | None = None,
                    referenced_tensorops: set | None = None,
                    inputs_outputs: dict | None = None,
                    channel_last: bool = False):
    """
    It populates the `modules_details` dictionary with tensorop's
    information: Its syntax and output variable.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        out_var (str | None): The output variable of the tensorop.
        referenced_tensorops (set | None): Set of tensorop names that are
            referenced by other tensorops and need unique variable names.
        channel_last (bool): If True, add permutes for spatial tensorops.

    Returns:
        None, but stores the tensorop details in the modules_details dict.

    """
    ts_op_synt = get_tensorop_syntax(tensorop, modules_details, out_var, inputs_outputs)

    out_var = _determine_tensorop_out_var(tensorop, modules_details, out_var, referenced_tensorops, inputs_outputs)
    out_var = _handle_skip_syntax(ts_op_synt, out_var, inputs_outputs, tensorop)
    out_var = _handle_split_tuple(tensorop, out_var)

    _add_tensorop_input_permute(tensorop, modules_details, get_tensorop_syntax)

    modules_details[tensorop.name + "_op"] = [ts_op_synt, out_var, tensorop]

    _update_skip_source_layer(ts_op_synt, out_var, inputs_outputs, tensorop, modules_details)
    _add_tensorop_output_permute(tensorop, modules_details, get_tensorop_syntax, out_var)


def preprocess_image(image_path: str, target_size: tuple):
    """
    It resizes and returns the images as np arrays.

    Arguments:
        image_path (str): The path to the images.
        target_size (tuple): The desired size of the images

    Returns:
        - The resized image as an np array

    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    np_image = np.array(image.convert('RGB'), dtype=np.float32)
    return np_image



def compute_mean_std(image_dir: str, num_samples: int = 100,
                     target_size: tuple = (256, 256)):
    """
    It computes the mean and standard deviation of images and checks
    whether scaling is needed.

    Arguments:
        image_dir (str): The directory where the images are stored.
        num_samples (int): Number of samples to use in the calculation.
        target_size (tuple): The desired size of the iamges.

    Returns:
        - The mean and std of the samples.

    """
    image_files = [os.path.join(root, file)
                   for root, _, files in os.walk(image_dir)
                   for file in files]
    sampled_files = random.sample(image_files,
                                  min(num_samples, len(image_files)))
    all_pixels = []
    for file in sampled_files:
        np_image = preprocess_image(file, target_size)
        all_pixels.append(np_image.reshape(-1, 3))

    all_pixels = np.concatenate(all_pixels)

    # Rescale if necessary
    if all_pixels.max() >= 1:
        scale = True
        all_pixels /= 255.0
    else:
        scale=False

    return (scale, np.mean(all_pixels, axis=0).tolist(),
            np.std(all_pixels, axis=0).tolist())


def format_value(elem: list):
    """
    It formats BUML list of int. If it contains one element, it is
    returned as `int`. Otherwise, it converts the list to a tuple.

    Arguments:
        elem (list): a list of int values


    Returns:
        - The formated elements either as int or tuple.

    """
    if len(elem) == 1:
        return elem[0]
    return tuple(elem)


class Permute(nn.Module):
    """A custom permute module for the sequential architecture"""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x


def _collect_op_and_temp_vars(modules_details):
    """Collect op_X and temp variable names from modules_details."""
    import re

    op_names_in_use = set()
    temp_vars_by_prefix = {
        '_binop_temp_': set(),
        '_nested_temp_': set(),
        '_subscript_temp_': set(),
        '_chain_temp_': set(),
    }

    for module_name, module_data in modules_details.items():
        out_var = None

        if module_name.endswith("_op"):
            out_var = module_data[1]
        elif module_name.endswith("_layer") or module_name.endswith("_activ"):
            out_var = module_data[1]

        if out_var:
            if isinstance(out_var, str) and re.match(r'^op_\d+$', out_var):
                op_names_in_use.add(out_var)
            elif isinstance(out_var, str):
                for prefix in temp_vars_by_prefix.keys():
                    if prefix == '_chain_temp_':
                        if re.match(rf'^{re.escape(prefix)}\d+_\d+$', out_var):
                            temp_vars_by_prefix[prefix].add(out_var)
                            break
                    elif re.match(rf'^{re.escape(prefix)}\d+$', out_var):
                        temp_vars_by_prefix[prefix].add(out_var)
                        break
            if isinstance(out_var, str) and ', ' in out_var:
                for var in out_var.split(', '):
                    var = var.strip()
                    if re.match(r'^op_\d+(_\d+)?$', var):
                        op_names_in_use.add(var)

    return op_names_in_use, temp_vars_by_prefix


def _create_op_renaming_map(op_names_in_use):
    """Create mapping from old op_X names to new sequential _op_Y names."""
    import re

    def get_base_op_num(op_name):
        match = re.match(r'^op_(\d+)', op_name)
        return int(match.group(1)) if match else 0

    sorted_op_names = sorted(op_names_in_use, key=get_base_op_num)

    old_to_new = {}
    new_counter = 1
    last_base_op = None

    for old_name in sorted_op_names:
        match = re.match(r'^(op_)(\d+)(_\d+)?$', old_name)
        if match:
            prefix, old_num, suffix = match.groups()
            base_old_name = f"op_{old_num}"

            if base_old_name != last_base_op:
                last_base_op = base_old_name
                current_counter = new_counter
                new_counter += 1

            if suffix:
                old_to_new[old_name] = f"_op_{current_counter}{suffix}"
            else:
                old_to_new[old_name] = f"_op_{current_counter}"

    return old_to_new


def _create_temp_renaming_map(temp_vars_by_prefix):
    """Create mapping from old temp variable names to new sequential names."""
    import re

    old_to_new = {}

    for prefix, vars_set in temp_vars_by_prefix.items():
        if not vars_set:
            continue

        def get_temp_num(temp_name):
            if prefix == '_chain_temp_':
                match = re.match(rf'^{re.escape(prefix)}(\d+)_\d+$', temp_name)
            else:
                match = re.match(rf'^{re.escape(prefix)}(\d+)$', temp_name)
            return int(match.group(1)) if match else 0

        sorted_temp_names = sorted(vars_set, key=get_temp_num)

        temp_counter = 1
        for old_temp_name in sorted_temp_names:
            new_temp_name = f"{prefix}{temp_counter}"
            old_to_new[old_temp_name] = new_temp_name
            temp_counter += 1

    return old_to_new


def _apply_variable_renaming(modules_details, old_to_new):
    """Apply variable renaming to all module entries in modules_details."""
    import re

    for module_name, module_data in modules_details.items():
        if module_name.endswith("_op"):
            syntax = module_data[0]
            if isinstance(syntax, str):
                for old_name, new_name in old_to_new.items():
                    syntax = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, syntax)
                module_data[0] = syntax

            out_var = module_data[1]
            if isinstance(out_var, str):
                if ', ' in out_var:
                    new_parts = []
                    for part in out_var.split(', '):
                        part = part.strip()
                        new_parts.append(old_to_new.get(part, part))
                    module_data[1] = ', '.join(new_parts)
                else:
                    module_data[1] = old_to_new.get(out_var, out_var)

        elif module_name.endswith("_layer") or module_name.endswith("_activ"):
            syntax = module_data[0]
            if isinstance(syntax, str):
                for old_name, new_name in old_to_new.items():
                    syntax = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, syntax)
                module_data[0] = syntax

            out_var = module_data[1]
            if isinstance(out_var, str) and out_var in old_to_new:
                module_data[1] = old_to_new[out_var]

            if len(module_data) > 2:
                in_var = module_data[2]
                if isinstance(in_var, str) and in_var in old_to_new:
                    module_data[2] = old_to_new[in_var]


def renumber_tensorop_variables(modules_details):
    """
    Renumber op_X and temp variables sequentially to avoid gaps when some operations
    use their original variable names from the source code.

    This function:
    1. Finds all tensorop entries that use op_X naming (not original names)
    2. Creates a mapping from old op_X names to new sequential _op_1, _op_2, _op_3...
    3. Finds all temp variables (_binop_temp_X, _nested_temp_X, _subscript_temp_X)
    4. Renumbers them sequentially within their prefix category
    5. Updates all references in modules_details (syntax, input/output vars)

    Args:
        modules_details (dict): The modules details dictionary to update in-place
    """
    op_names_in_use, temp_vars_by_prefix = _collect_op_and_temp_vars(modules_details)

    if not op_names_in_use and not any(temp_vars_by_prefix.values()):
        return

    old_to_new = _create_op_renaming_map(op_names_in_use)
    old_to_new.update(_create_temp_renaming_map(temp_vars_by_prefix))

    _apply_variable_renaming(modules_details, old_to_new)
    

