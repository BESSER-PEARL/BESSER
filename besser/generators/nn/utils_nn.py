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
    modules_names = list(modules_details.keys())
    lyr_input = layer.name_module_input
    # Handle None and False (False means input not reused) - use previous output
    if lyr_input is not None and lyr_input is not False:
        # Special case: INPUT marker for network input
        # Return 'x' (the original input variable)
        if lyr_input == 'INPUT':
            return 'x'
        # Handle bidirectional concat marker (from torch2tf bidirectional RNN migration)
        if isinstance(lyr_input, str) and lyr_input.startswith("bidirectional_concat_"):
            base_module = lyr_input.replace("bidirectional_concat_", "")
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                # The concat variable name is now stored in index 4 for all bidirectional RNNs
                if len(layer_details) > 4 and layer_details[4]:
                    return layer_details[4]  # Concat variable (e.g., "gru_concat")
                # Fallback: return output variable
                return layer_details[1]
        # Check for split tensorop output suffixes
        if isinstance(lyr_input, str) and "__split_" in lyr_input:
            # e.g., "op_1__split_0" -> get specific split output variable
            base_op, idx_str = lyr_input.rsplit("__split_", 1)
            idx = int(idx_str)
            if f"{base_op}_op" in modules_names:
                op_details = modules_details[f"{base_op}_op"]
                # op_details[1] is the tuple like "x_2_0, x_2_1, x_2_2"
                var_list = [v.strip() for v in op_details[1].split(',')]
                if idx < len(var_list):
                    return var_list[idx]
        # Check for RNN hidden/cell state suffixes before checking layer names
        elif isinstance(lyr_input, str) and lyr_input.endswith("__hidden_forward"):
            base_module = lyr_input[:-16]  # Remove "__hidden_forward"
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                if len(layer_details) > 4:
                    # For bidirectional, extract forward component: hidden[-2]
                    return f"{layer_details[4]}[-2]"
                return layer_details[1]
        elif isinstance(lyr_input, str) and lyr_input.endswith("__hidden_backward"):
            base_module = lyr_input[:-17]  # Remove "__hidden_backward"
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                if len(layer_details) > 4:
                    # For bidirectional, extract backward component: hidden[-1]
                    return f"{layer_details[4]}[-1]"
                    return f"{layer_details[4]}[-1]"
                return layer_details[1]
        elif isinstance(lyr_input, str) and lyr_input.endswith("__cell_forward"):
            base_module = lyr_input[:-14]  # Remove "__cell_forward"
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                if len(layer_details) > 4:
                    # For bidirectional LSTM, extract forward cell: cell[-2]
                    return f"{layer_details[4]}_cell[-2]" if "_cell" not in layer_details[4] else f"{layer_details[4]}[-2]"
                return layer_details[1]
        elif isinstance(lyr_input, str) and lyr_input.endswith("__cell_backward"):
            base_module = lyr_input[:-15]  # Remove "__cell_backward"
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                if len(layer_details) > 4:
                    # For bidirectional LSTM, extract backward cell: cell[-1]
                    return f"{layer_details[4]}_cell[-1]" if "_cell" not in layer_details[4] else f"{layer_details[4]}[-1]"
                return layer_details[1]
        elif isinstance(lyr_input, str) and lyr_input.endswith("__hidden"):
            base_module = lyr_input[:-8]  # Remove "__hidden"
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                if len(layer_details) > 4:
                    return layer_details[4]  # Hidden variable
                return layer_details[1]  # Fallback
        elif isinstance(lyr_input, str) and lyr_input.endswith("__cell"):
            base_module = lyr_input[:-6]  # Remove "__cell"
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                if len(layer_details) > 4:
                    return layer_details[4]  # For now use hidden (cell state needs separate tracking)
                return layer_details[1]
        elif f"{lyr_input}_layer" in modules_details:
            layer_details = modules_details[f"{lyr_input}_layer"]
            # Check if this is an RNN with return_type="hidden" that had a subscript creating different var
            # Example: x = h[-1] where h is the hidden state, x is the subscript result
            if (len(layer_details) > 4 and hasattr(layer_details[3], 'return_type') and
                layer_details[3].return_type == "hidden"):
                # Check if prev_out_var matches the subscript output from inputs_outputs
                if (inputs_outputs and f"{lyr_input}__hidden" in inputs_outputs and
                    prev_out_var == inputs_outputs[f"{lyr_input}__hidden"][1]):
                    # Use prev_out_var which is the subscript result variable
                    return prev_out_var
            # Check if this layer should use RNN hidden state instead of sequence output
            if hasattr(layer, 'use_rnn_hidden') and layer.use_rnn_hidden:
                # Return hidden state variable (index 4) instead of sequence output (index 1)
                if len(layer_details) > 4:
                    return layer_details[4]
            return layer_details[1]
        if f"{lyr_input}_nn" in modules_names:
            return  modules_details[f"{lyr_input}_nn"]["in_out_variable"]
        if f"{lyr_input}_activ" in modules_names:
            return modules_details[f"{lyr_input}_activ"][1]
        #module.name_module_input+"_op" in my_keys
        if f"{lyr_input}_op" in modules_names:
            return modules_details[f"{lyr_input}_op"][1]
        # Fallback: TensorOp not processed yet, use prev_out_var
        # This can happen when layers reference TensorOps in parallel operations
        return prev_out_var
    return prev_out_var

def add_in_out_var_to_subnn(modules_details: dict):
    """
    It sets the in_out_variable of subnns, which refers to the input
    and output variable of the subnn.

    Arguments:
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.

    Returns:
        None, but stores the in_out_var in modules_details dict.

    """
    last_module = list(modules_details.keys())[-1]
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
        out_var = "x_1"
    else:
        # Find the next available x_N by scanning existing variables
        if modules_details:
            used_nums = set()
            for module_details in modules_details.values():
                if isinstance(module_details, list) and len(module_details) > 1:
                    var = module_details[1]  # Output variable
                    if isinstance(var, str) and var.startswith('x_'):
                        try:
                            num = int(var.split('_')[1])
                            used_nums.add(num)
                        except (ValueError, IndexError):
                            pass
            # Also check input variables (module_details[2]) for nested/reused vars
            for module_details in modules_details.values():
                if isinstance(module_details, list) and len(module_details) > 2:
                    var = module_details[2]  # Input variable
                    if isinstance(var, str) and var.startswith('x_'):
                        try:
                            num = int(var.split('_')[1])
                            used_nums.add(num)
                        except (ValueError, IndexError):
                            pass

            # Find the first available number starting from 1
            num = 1
            while num in used_nums:
                num += 1
            out_var = f"x_{num}"
        else:
            # Fallback: try to extract number from prev_out_var
            parts = prev_out_var.split('_')
            try:
                num = int(parts[-1])
                out_var = f"x_{num + 1}"
            except ValueError:
                out_var = "x_1"
    return out_var


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
        # Standalone activation layer created from functional API
        layer_synt = setup.setup_standalone_activation(out_var, in_var)
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


def _store_layer_in_modules_details(layer, layer_synt, out_layer, in_layer, modules_details, model=None):
    """Store layer syntax and variables in modules_details."""
    if layer_synt is None:
        return

    print(f"[DEBUG _store_layer] layer.name={layer.name}, out_layer={out_layer}, in_layer={in_layer}")

    if hasattr(layer, 'return_type') and layer.return_type in ("both", "hidden"):
        # Check if there's a stored concat variable name for bidirectional RNNs
        concat_var_names = getattr(model, 'bidirectional_concat_var_names', {}) if model else {}
        inputs_outputs = getattr(model, 'inputs_outputs', {}) if model else {}

        if layer.name in concat_var_names:
            hidden_var = concat_var_names[layer.name]
        elif inputs_outputs and (layer.name + "__hidden") in inputs_outputs:
            # Use original hidden state variable name from tuple unpacking (first element [0])
            # NOT the subscript result (second element [1]) to avoid name collisions
            hidden_var = inputs_outputs[layer.name + "__hidden"][0]
            # Safety check: if hidden_var is None or "_", handle appropriately
            if hidden_var is None:
                hidden_var = f"{out_layer}_h" if out_layer != "x" else "h"
            # If hidden_var is "_", keep it as underscore (don't auto-generate)
            # The generator will use this to preserve the underscore pattern
        else:
            hidden_var = f"{out_layer}_h" if out_layer != "x" else "h"
        modules_details[layer.name + "_layer"] = [layer_synt, out_layer,
                                                  in_layer, layer, hidden_var]
    else:
        modules_details[layer.name + "_layer"] = [layer_synt, out_layer,
                                                  in_layer, layer]


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

def _resolve_source_layer_var(source_layer, modules_details, inputs_outputs=None, tensorop=None):
    """Resolve a source layer name to its output variable."""
    if source_layer == 'INPUT':
        # Check if we have actual input variable in inputs_outputs first
        if inputs_outputs and tensorop and hasattr(tensorop, 'name') and tensorop.name in inputs_outputs:
            actual_input_var = inputs_outputs[tensorop.name][0]
            if actual_input_var:
                print(f"[DEBUG _resolve_source_layer_var] Using '{actual_input_var}' from inputs_outputs")
                return actual_input_var
        # Fallback to 'inp' for backward compatibility
        print(f"[DEBUG _resolve_source_layer_var] Falling back to 'inp' for INPUT")
        return 'inp'

    # Handle RNN hidden/cell state suffixes before split suffixes
    if isinstance(source_layer, str) and source_layer.endswith("__hidden_forward"):
        base_module = source_layer[:-16]  # Remove "__hidden_forward"
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return f"{layer_details[4]}[-2]"  # Forward component
            return layer_details[1]
    elif isinstance(source_layer, str) and source_layer.endswith("__hidden_backward"):
        base_module = source_layer[:-17]  # Remove "__hidden_backward"
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return f"{layer_details[4]}[-1]"  # Backward component
            return layer_details[1]
    elif isinstance(source_layer, str) and source_layer.endswith("__cell_forward"):
        base_module = source_layer[:-14]  # Remove "__cell_forward"
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                hidden_var = layer_details[4]
                cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
                return f"{cell_var}[-2]"  # Forward component
            return layer_details[1]
    elif isinstance(source_layer, str) and source_layer.endswith("__cell_backward"):
        base_module = source_layer[:-15]  # Remove "__cell_backward"
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                hidden_var = layer_details[4]
                cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
                return f"{cell_var}[-1]"  # Backward component
            return layer_details[1]
    elif isinstance(source_layer, str) and source_layer.endswith("__hidden"):
        base_module = source_layer[:-8]  # Remove "__hidden"
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            # Check return_type to determine where hidden variable is
            layer_obj = layer_details[3] if len(layer_details) > 3 else None
            if layer_obj and hasattr(layer_obj, 'return_type'):
                if layer_obj.return_type == "hidden":
                    # For return_type="hidden", hidden is the main output (index 1)
                    return layer_details[1]
                elif layer_obj.return_type in ("both", "full") and len(layer_details) > 4:
                    # For return_type="both", hidden is at index 4
                    return layer_details[4]
            # Fallback
            return layer_details[1]
    elif isinstance(source_layer, str) and source_layer.endswith("__cell"):
        base_module = source_layer[:-6]  # Remove "__cell"
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return f"{layer_details[4]}_cell"  # Cell variable
            return layer_details[1]  # Fallback

    # Handle split output suffixes: "op_2__split_0" -> get specific split variable
    if isinstance(source_layer, str) and "__split_" in source_layer:
        base_op, idx_str = source_layer.rsplit("__split_", 1)
        idx = int(idx_str)
        if f"{base_op}_op" in modules_details:
            # The out_var is a tuple like "x_2_0, x_2_1, x_2_2"
            out_var_tuple = modules_details[f"{base_op}_op"][1]
            var_list = [v.strip() for v in out_var_tuple.split(',')]
            if idx < len(var_list):
                return var_list[idx]
            # Fallback if index out of range
            return f"{base_op}_{idx}"

    # Handle bidirectional concat marker
    if isinstance(source_layer, str) and source_layer.startswith("bidirectional_concat_"):
        base_layer = source_layer.replace("bidirectional_concat_", "")
        layer_key = f"{base_layer}_layer"
        if layer_key in modules_details:
            layer_details = modules_details[layer_key]
            # The concat variable name is now stored in index 4 for all bidirectional RNNs
            if len(layer_details) > 4 and layer_details[4]:
                return layer_details[4]
            return layer_details[1]

    modules_names = list(modules_details.keys())
    for suffix in ['_layer', '_op', '_activ', '_nn']:
        key = f"{source_layer}{suffix}"
        if key in modules_names:
            if suffix == '_nn':
                return modules_details[key]["in_out_variable"]
            return modules_details[key][1]

    # Fallback: treat as input variable if it's 'x', otherwise use as-is
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
        # Check if we have actual input variable in inputs_outputs first
        print(f"[DEBUG _resolve_prev_out_var] module_input=INPUT, inputs_outputs={inputs_outputs is not None}, tensorop={tensorop}, has_name={hasattr(tensorop, 'name') if tensorop else False}")
        if tensorop and hasattr(tensorop, 'name'):
            print(f"[DEBUG _resolve_prev_out_var] tensorop.name={tensorop.name}, in_inputs_outputs={tensorop.name in inputs_outputs if inputs_outputs else False}")
        if inputs_outputs and tensorop and hasattr(tensorop, 'name') and tensorop.name in inputs_outputs:
            actual_input_var = inputs_outputs[tensorop.name][0]
            print(f"[DEBUG _resolve_prev_out_var] actual_input_var={actual_input_var}")
            if actual_input_var:
                print(f"[DEBUG _resolve_prev_out_var] Using actual input var '{actual_input_var}' from inputs_outputs instead of 'inp'")
                return actual_input_var
        print(f"[DEBUG _resolve_prev_out_var] Falling back to 'inp'")
        return 'inp'

    # Bidirectional RNN forward/backward hidden states
    if module_input.endswith("__hidden_forward"):
        base_module = module_input[:-16]
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return f"{layer_details[4]}[-2]"
            return layer_details[1]
        return "x"

    if module_input.endswith("__hidden_backward"):
        base_module = module_input[:-17]
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                return f"{layer_details[4]}[-1]"
            return layer_details[1]
        return "x"

    # Bidirectional LSTM forward/backward cell states
    if module_input.endswith("__cell_forward"):
        base_module = module_input[:-14]
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                hidden_var = layer_details[4]
                cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
                return f"{cell_var}[-2]"
            return layer_details[1]
        return "x"

    if module_input.endswith("__cell_backward"):
        base_module = module_input[:-15]
        modules_names = list(modules_details.keys())
        if f"{base_module}_layer" in modules_names:
            layer_details = modules_details[f"{base_module}_layer"]
            if len(layer_details) > 4:
                hidden_var = layer_details[4]
                cell_var = f"{hidden_var}_cell" if "_cell" not in hidden_var else hidden_var
                return f"{cell_var}[-1]"
            return layer_details[1]
        return "x"

    # RNN hidden state
    if module_input.endswith("__hidden"):
        base_module = module_input[:-8]
        return _get_rnn_state_var(base_module, modules_details, get_rnn_hidden_var_fn)

    # LSTM cell state
    if module_input.endswith("__cell"):
        base_module = module_input[:-6]
        return _get_rnn_state_var(base_module, modules_details, get_rnn_hidden_var_fn)

    # Bidirectional RNN concat
    if module_input.startswith("bidirectional_concat_"):
        base_layer = module_input.replace("bidirectional_concat_", "")
        layer_key = f"{base_layer}_layer"
        if layer_key in modules_details:
            layer_details = modules_details[layer_key]
            # The concat variable name is now stored in index 4 for all bidirectional RNNs
            if len(layer_details) > 4 and layer_details[4]:
                return layer_details[4]
            return layer_details[1]
        return "x"

    # Regular module
    return _resolve_source_layer_var(module_input, modules_details)


def _get_initial_prev_out_var(modules_details, tensorop, get_rnn_hidden_var_fn, inputs_outputs=None):
    """Get the initial prev_out_var before tensorop-specific handling."""
    print(f"[DEBUG _get_initial_prev_out_var ENTRY] tensorop.name={tensorop.name}, modules_details empty={len(list(modules_details.keys())) == 0}")
    if len(list(modules_details.keys())) == 0:
        print(f"[DEBUG _get_initial_prev_out_var] modules_details empty, but checking inputs_outputs for {tensorop.name}")
        # Check if we have actual input variable in inputs_outputs even when modules_details is empty
        if inputs_outputs and tensorop.name in inputs_outputs:
            actual_input_var = inputs_outputs[tensorop.name][0]
            if actual_input_var:
                print(f"[DEBUG _get_initial_prev_out_var] Using '{actual_input_var}' from inputs_outputs for first tensorop")
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
    print(f"[DEBUG _get_initial_prev_out_var] tensorop.name={tensorop.name}, has_name_module_input={hasattr(tensorop, 'name_module_input')}, has_layers_of_tensors={hasattr(tensorop, 'layers_of_tensors')}")
    if hasattr(tensorop, 'layers_of_tensors'):
        print(f"[DEBUG _get_initial_prev_out_var] layers_of_tensors={tensorop.layers_of_tensors}")
    if hasattr(tensorop, 'name_module_input') and tensorop.name_module_input is not None:
        module_input = tensorop.name_module_input
        print(f"[DEBUG _get_initial_prev_out_var] Set module_input from name_module_input: {module_input}")
    elif hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
        if len(tensorop.layers_of_tensors) == 1:
            module_input = tensorop.layers_of_tensors[0]
            print(f"[DEBUG _get_initial_prev_out_var] Set module_input from layers_of_tensors[0]: {module_input}")
        elif 'INPUT' in tensorop.layers_of_tensors:
            module_input = 'INPUT'
            print(f"[DEBUG _get_initial_prev_out_var] Set module_input to INPUT")

    if module_input is not None:
        print(f"[DEBUG _get_initial_prev_out_var] module_input={module_input}, calling _resolve_prev_out_var_from_module_input")
        prev_out_var = _resolve_prev_out_var_from_module_input(
            module_input, modules_details, get_rnn_hidden_var_fn, inputs_outputs, tensorop
        )

    print(f"[DEBUG _get_initial_prev_out_var RETURN] prev_out_var={prev_out_var}")
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
    print(f"[DEBUG get_tensorop_params] tensorop.name={tensorop.name}, tns_type={tensorop.tns_type}, inputs_outputs={inputs_outputs is not None}")
    if inputs_outputs and tensorop.name in inputs_outputs:
        print(f"[DEBUG get_tensorop_params] inputs_outputs[{tensorop.name}] = {inputs_outputs[tensorop.name]}")
    # Get initial prev_out_var
    prev_out_var = _get_initial_prev_out_var(modules_details, tensorop, get_rnn_hidden_var_fn, inputs_outputs)

    # Get tensorop-specific parameters and potentially override prev_out_var
    tns_type = tensorop.tns_type
    handler = _TENSOROP_HANDLERS.get(tns_type, _handle_generic_params)

    # Pass inputs_outputs to handlers that support it
    if tns_type in ["mean", "max", "squeeze", "unsqueeze", "subscript", "shape_dim", "normalize", "split", "repeat"]:
        override_prev_out_var, params = handler(tensorop, modules_details, inputs_outputs)
    else:
        override_prev_out_var, params = handler(tensorop, modules_details)

    # Use override if provided
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

    # If inputs_outputs has the actual output variable from original code, use it for ALL tensorop types
    # This preserves original variable names (e.g., x = tf.add(x, y) uses 'x', not 'op_1')
    if out_var is None and inputs_outputs and tensorop.name in inputs_outputs:
        out_var = inputs_outputs[tensorop.name][1]

    if out_var is None:
        # For shape_dim TensorOps, use the TensorOp name as the output variable
        # (e.g., 'b' for extracting batch size, 't' for sequence length)
        if tensorop.tns_type == "shape_dim":
            out_var = tensorop.name
        elif len(modules_details) == 0:
            out_var  = initialize_tensorop_var(tensorop)
        else:
            prev_module = list(modules_details.keys())[-1]
            prev_out_var = get_previous_out_var(modules_details, prev_module)

            # If this tensorop is referenced by other operations, give it a unique variable
            if referenced_tensorops and tensorop.name in referenced_tensorops:
                # Create a unique intermediate variable name
                out_var = tensorop.name
            # Operations with output variables in inputs_outputs: use actual target variable from original code
            # This applies to binop, concatenate, subscript, and other operations that have explicit assignments
            elif inputs_outputs and tensorop.name in inputs_outputs and inputs_outputs[tensorop.name][1] is not None:
                out_var = inputs_outputs[tensorop.name][1]  # Use actual target variable
            # Split operations always use their own name to ensure unique tuple variables
            elif tensorop.tns_type == "split":
                out_var = tensorop.name
            # Identity operations use their name to preserve variable assignments (e.g., residual = x)
            elif tensorop.tns_type == "identity":
                out_var = tensorop.name
            # Tensorops with input_reused should also use their op_N name to avoid x_N conflicts
            elif hasattr(tensorop, 'input_reused') and tensorop.input_reused:
                out_var = tensorop.name
            else:
                out_var = get_tensorop_out_var(tensorop, prev_out_var, modules_details)

    # If the tensorop syntax is SKIP:variable, extract the actual variable name
    # This ensures downstream layers can correctly reference the skipped operation
    if isinstance(ts_op_synt, str) and ts_op_synt.startswith("SKIP:"):
        out_var = ts_op_synt[5:]  # Extract variable after "SKIP:"

    # Special handling for split: generate tuple unpacking with indexed variable names
    if tensorop.tns_type == "split" and hasattr(tensorop, 'split_sizes'):
        num_splits = tensorop.split_sizes
        # Generate variable names: out_var_0, out_var_1, out_var_2, ...
        split_vars = [f"{out_var}_{i}" for i in range(num_splits)]
        out_var = ", ".join(split_vars)

    # Add input permute for spatial tensorops if needed (PyTorch TF→PyTorch migration)
    if hasattr(tensorop, 'permute_in') and tensorop.permute_in and tensorop.tns_type in ("interpolate", "pad"):
        from besser.BUML.metamodel.nn import TensorOp as TensorOpClass

        # Add input permute: NHWC → NCHW
        in_permute_name = f"{tensorop.name}_in_op"
        in_permute = TensorOpClass(name=in_permute_name, tns_type="permute",
                                     permute_dim=[0, 3, 1, 2])
        # Get the current input variable
        prev_module = list(modules_details.keys())[-1] if modules_details else None
        in_var = get_previous_out_var(modules_details, prev_module) if prev_module else "x"
        # Store input permute
        handle_tensorop(in_permute, modules_details, get_tensorop_syntax,
                        out_var=in_var, channel_last=False)

    modules_details[tensorop.name + "_op"] = [ts_op_synt, out_var, tensorop]

    # Add output permute for spatial tensorops if needed
    if hasattr(tensorop, 'permute_out') and tensorop.permute_out and tensorop.tns_type in ("interpolate", "pad"):
        from besser.BUML.metamodel.nn import TensorOp as TensorOpClass

        out_permute_name = f"{tensorop.name}_out_op"
        out_permute = TensorOpClass(name=out_permute_name, tns_type="permute",
                                      permute_dim=[0, 2, 3, 1])
        # Use the current tensorop's output as input to the permute
        handle_tensorop(out_permute, modules_details, get_tensorop_syntax,
                        out_var=out_var, channel_last=False)


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
    import re

    print("[DEBUG renumber_tensorop_variables] Starting renumbering...")

    # Step 1: Collect all op_X names that are actually used in output variables
    op_names_in_use = set()
    # Also collect temp variables by prefix type
    temp_vars_by_prefix = {
        '_binop_temp_': set(),
        '_nested_temp_': set(),
        '_subscript_temp_': set(),
    }

    for module_name, module_data in modules_details.items():
        if module_name.endswith("_op"):
            # module_data format: [syntax, out_var, tensorop_obj]
            out_var = module_data[1]
            # Check if output variable is op_X (not an original variable name)
            if isinstance(out_var, str) and re.match(r'^op_\d+$', out_var):
                op_names_in_use.add(out_var)
            # Check for temp variables
            elif isinstance(out_var, str):
                for prefix in temp_vars_by_prefix.keys():
                    if re.match(rf'^{re.escape(prefix)}\d+$', out_var):
                        temp_vars_by_prefix[prefix].add(out_var)
                        break
            # Also check for tuple outputs like "op_5_0, op_5_1, op_5_2"
            if isinstance(out_var, str) and ', ' in out_var:
                for var in out_var.split(', '):
                    var = var.strip()
                    if re.match(r'^op_\d+(_\d+)?$', var):
                        op_names_in_use.add(var)

    print(f"[DEBUG renumber_tensorop_variables] Found {len(op_names_in_use)} op_X variables in use: {sorted(op_names_in_use)}")
    for prefix, vars_set in temp_vars_by_prefix.items():
        if vars_set:
            print(f"[DEBUG renumber_tensorop_variables] Found {len(vars_set)} {prefix}X variables: {sorted(vars_set)}")

    if not op_names_in_use and not any(temp_vars_by_prefix.values()):
        print("[DEBUG renumber_tensorop_variables] No op_X or temp variables to renumber")
        return
    
    # Step 2: Sort by original number and create sequential mapping
    # Extract base op names (op_5_0 -> op_5) for sorting
    def get_base_op_num(op_name):
        match = re.match(r'^op_(\d+)', op_name)
        return int(match.group(1)) if match else 0
    
    sorted_op_names = sorted(op_names_in_use, key=get_base_op_num)
    
    # Create mapping: old op_X -> new op_Y
    # Handle both simple (op_7) and indexed (op_7_0, op_7_1) names
    old_to_new = {}
    new_counter = 1
    last_base_op = None
    
    for old_name in sorted_op_names:
        match = re.match(r'^(op_)(\d+)(_\d+)?$', old_name)
        if match:
            prefix, old_num, suffix = match.groups()
            base_old_name = f"op_{old_num}"
            
            # If this is a new base operation, increment counter
            if base_old_name != last_base_op:
                last_base_op = base_old_name
                current_counter = new_counter
                new_counter += 1
            
            # Map old to new (use _op_X prefix to avoid conflicts with user variables)
            if suffix:  # op_7_0 -> _op_1_0
                old_to_new[old_name] = f"_op_{current_counter}{suffix}"
            else:  # op_7 -> _op_1
                old_to_new[old_name] = f"_op_{current_counter}"
    
    print(f"[DEBUG renumber_tensorop_variables] op_X mapping: {old_to_new}")

    # Step 2b: Renumber temp variables (_binop_temp_X, _nested_temp_X, _subscript_temp_X)
    for prefix, vars_set in temp_vars_by_prefix.items():
        if not vars_set:
            continue

        # Sort by number
        def get_temp_num(temp_name):
            match = re.match(rf'^{re.escape(prefix)}(\d+)$', temp_name)
            return int(match.group(1)) if match else 0

        sorted_temp_names = sorted(vars_set, key=get_temp_num)

        # Create sequential mapping: _binop_temp_7 -> _binop_temp_1, etc.
        temp_counter = 1
        for old_temp_name in sorted_temp_names:
            new_temp_name = f"{prefix}{temp_counter}"
            old_to_new[old_temp_name] = new_temp_name
            temp_counter += 1

        print(f"[DEBUG renumber_tensorop_variables] {prefix}X mapping: {dict((k, v) for k, v in old_to_new.items() if k.startswith(prefix))}")

    # Step 3: Update all references in modules_details
    for module_name, module_data in modules_details.items():
        if module_name.endswith("_op"):
            # Update syntax (index 0)
            syntax = module_data[0]
            if isinstance(syntax, str):
                for old_name, new_name in old_to_new.items():
                    # Use word boundary to avoid partial replacements
                    syntax = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, syntax)
                module_data[0] = syntax
            
            # Update out_var (index 1)
            out_var = module_data[1]
            if isinstance(out_var, str):
                # Handle tuple outputs
                if ', ' in out_var:
                    new_parts = []
                    for part in out_var.split(', '):
                        part = part.strip()
                        new_parts.append(old_to_new.get(part, part))
                    module_data[1] = ', '.join(new_parts)
                else:
                    module_data[1] = old_to_new.get(out_var, out_var)
        
        elif module_name.endswith("_layer"):
            # Update in_var (index 2) for layers
            if len(module_data) > 2:
                in_var = module_data[2]
                if isinstance(in_var, str) and in_var in old_to_new:
                    module_data[2] = old_to_new[in_var]
    
    print("[DEBUG renumber_tensorop_variables] Renumbering complete")

