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


def get_previous_out_var(modules_details: dict, prev_module: str):
    """
    It retrieves the output variable of the previous module in order to
    use it as the input variable of the current module.

    Arguments:
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        prev_module (str): The name of the previous module.

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
        return module_data[1]

def get_input_var(layer: Layer, modules_details: dict, prev_out_var: str):
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

    Returns:
        The input variable.
    """
    modules_names = list(modules_details.keys())
    lyr_input = layer.name_module_input
    # Handle None and False (False means input not reused) - use previous output
    if lyr_input is not None and lyr_input is not False:
        # Special case: INPUT marker for network input
        if lyr_input == 'INPUT':
            return 'x'
        # Handle bidirectional concat marker (from torch2tf bidirectional RNN migration)
        if isinstance(lyr_input, str) and lyr_input.startswith("bidirectional_concat_"):
            base_module = lyr_input.replace("bidirectional_concat_", "")
            if f"{base_module}_layer" in modules_names:
                layer_details = modules_details[f"{base_module}_layer"]
                # Return the hidden variable (index 4) which contains the concatenated hidden states
                if len(layer_details) > 4:
                    return layer_details[4]
                return layer_details[1]  # Fallback
        # Check for RNN hidden/cell state suffixes before checking layer names
        if isinstance(lyr_input, str) and lyr_input.endswith("__hidden"):
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
        elif f"{lyr_input}_layer" in modules_names:
            # Check if this layer should use RNN hidden state instead of sequence output
            if hasattr(layer, 'use_rnn_hidden') and layer.use_rnn_hidden:
                # Return hidden state variable (index 4) instead of sequence output (index 1)
                if len(modules_details[f"{lyr_input}_layer"]) > 4:
                    return modules_details[f"{lyr_input}_layer"][4]
            return modules_details[f"{lyr_input}_layer"][1]
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
        # Handle inline layer calls (from tf2torch)
        if isinstance(layer_name, str) and layer_name.startswith("INLINE_CALL:"):
            # Format: "INLINE_CALL:layer_name:input_var"
            parts = layer_name.split(":")
            inline_layer_name = parts[1]
            inline_input_var = parts[2]
            out_vars.append(f"self.{inline_layer_name}({inline_input_var})")
            continue
        # Handle numeric constants directly
        if isinstance(layer_name, (int, float)):
            out_vars.append(str(layer_name))
        # Handle RNN hidden/cell state suffixes
        elif isinstance(layer_name, str) and layer_name.endswith("__hidden"):
            base_layer = layer_name[:-8]  # Remove "__hidden"
            if base_layer + "_layer" in my_keys:
                layer_details = modules_details[base_layer + "_layer"]
                if len(layer_details) > 4:
                    out_vars.append(layer_details[4])  # Hidden variable
                else:
                    out_vars.append(layer_details[1])  # Fallback to output
            else:
                out_vars.append("x")  # Fallback
        elif isinstance(layer_name, str) and layer_name.endswith("__cell"):
            base_layer = layer_name[:-6]  # Remove "__cell"
            if base_layer + "_layer" in my_keys:
                layer_details = modules_details[base_layer + "_layer"]
                if len(layer_details) > 4:
                    # For LSTM, cell state would need separate tracking
                    # For now, use hidden var as placeholder
                    out_vars.append(layer_details[4])
                else:
                    out_vars.append(layer_details[1])
            else:
                out_vars.append("x")
        elif layer_name+"_layer" in my_keys:
            layer_details = modules_details[layer_name + "_layer"]
            # Check if this layer has return_type="both" and we have actual_vars info
            if (len(layer_details) > 4 and actual_vars and i < len(actual_vars)):
                # actual_vars contains "output" or "hidden" flags
                if actual_vars[i] == "hidden":
                    # Use hidden variable (element [4])
                    out_vars.append(layer_details[4])
                else:
                    # Use output variable (element [1]), but slice to get last timestep
                    # For RNN with return_sequences=True, we need [:, -1, :] to get last timestep
                    out_var = layer_details[1]
                    out_vars.append(f"{out_var}[:, -1, :]")
            else:
                # Normal case: use output variable
                out_vars.append(layer_details[1])
        elif layer_name+"_activ" in my_keys:
            # Standalone activation layer (GeneralLayer)
            activ_details = modules_details[layer_name + "_activ"]
            out_vars.append(activ_details[1])
        elif any(k.startswith(layer_name + "_") and k.endswith("_nn") for k in my_keys):
            # Sub-network (Sequential) - has format: name_N_nn where N is a counter
            nn_key = next(k for k in my_keys if k.startswith(layer_name + "_") and k.endswith("_nn"))
            nn_details = modules_details[nn_key]
            out_vars.append(nn_details["in_out_variable"])
        elif layer_name == "INPUT":
            # Special marker for network input
            out_vars.append("x")
        elif layer_name.startswith("bidirectional_concat_"):
            # Special marker for bidirectional RNN hidden state concat (from torch2tf)
            # Extract the actual layer name and get its hidden variable
            base_layer = layer_name.replace("bidirectional_concat_", "")
            layer_details = modules_details.get(base_layer + "_layer")
            if layer_details and len(layer_details) > 4:
                out_vars.append(layer_details[4])  # Hidden variable at index 4
            else:
                # Fallback to looking for the marker itself as an op
                out_vars.append(modules_details[layer_name + "_op"][1])
        else:
            out_vars.append(modules_details[layer_name + "_op"][1])
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
    if tensorop.input_reused is True:
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
        # Try to extract the number from x_N pattern
        parts = prev_out_var.split('_')
        try:
            # If the last part is a number, increment it
            num = int(parts[-1])
            out_var = f"x_{num + 1}"
        except ValueError:
            # If prev_out_var doesn't follow x_N pattern (e.g., 'b', 't', 'last')
            # Find the next available x_N by scanning existing variables
            if modules_details:
                max_num = 0
                for module_details in modules_details.values():
                    if isinstance(module_details, list) and len(module_details) > 1:
                        var = module_details[1]  # Output variable
                        if isinstance(var, str) and var.startswith('x_'):
                            try:
                                num = int(var.split('_')[1])
                                max_num = max(max_num, num)
                            except (ValueError, IndexError):
                                pass
                out_var = f"x_{max_num + 1}"
            else:
                # Fallback if modules_details not provided
                out_var = "x_1"
    return out_var


def get_layer_vars(layer: Layer, prev_out_var: str, modules_details: dict):
    """
    It sets the input and output variables of the layer.

    Arguments:
        layer (Layer): The BUML layer object.
        prev_out_var (str): The previous output variable.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.

    Returns:
        - The input variable and output variables of both the layer and
          its activation function.

    """
    out_var_actv, in_var_actv = None, None
    if layer.input_reused:
        out_var_layer = get_out_var_input_reused(prev_out_var, modules_details)
    else:
        out_var_layer = prev_out_var
    in_var_layer = get_input_var(layer, modules_details,
                                           prev_out_var)
    if layer.actv_func is not None:
        out_var_actv, in_var_actv = out_var_layer, out_var_layer
    return out_var_layer, in_var_layer, out_var_actv, in_var_actv

def initialize_layer_vars(layer: Layer):
    """
    It sets the input and output variables of layer (and activation
    function for PyTorch) in the case it is the first module in
    the neural network.

    Arguments:
        layer (Layer): The BUML layer object.

    Returns:
        - The input variable and output variables of both the layer and
          its activation function.
    """
    out_var_actv, in_var_actv = None, None
    if layer.input_reused is True:
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

def handle_layer(layer: Layer, setup_layer: 'NNCodeGenerator',
                 modules_details: dict, channel_last: bool | None,
                 actv_func_syntax: str | bool = False, is_seq: bool = False,
                 is_subnn: bool = False):
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

    if len(modules_details) == 0:
        out_layer, in_layer, out_actv, in_actv = initialize_layer_vars(layer)
    else:
        prev_module = list(modules_details.keys())[-1]

        # Skip shape_dim operations to find the actual previous tensor-producing module
        if _is_shape_dim_operation(prev_module, modules_details):
            actual_prev = _find_previous_non_shape_dim_module(modules_details)
            if actual_prev:
                prev_module = actual_prev
            else:
                # No previous non-shape_dim module, treat as first layer
                out_layer, in_layer, out_actv, in_actv = initialize_layer_vars(layer)
                prev_module = None

        if prev_module is not None:
            prev_out_var = get_previous_out_var(modules_details, prev_module)
            out_layer, in_layer, out_actv, in_actv = get_layer_vars(
                layer, prev_out_var, modules_details
            )

    layer_synt, actv_func_syntax, setup = get_layer_syntax(
        setup_layer, layer, modules_details, actv_func_syntax, out_layer, in_layer,
        is_subnn=is_subnn
    )

    if setup.permute_in and hasattr(setup, 'add_permute'):
        if channel_last is None or channel_last:
            dim = setup.dim
            setup.add_permute(
                layer.name, dim, in_layer, permute_in=True,
                sequential=is_seq, is_subnn=is_subnn
            )

    # Only add _layer entry if layer_synt is not None (standalone activations return None)
    if layer_synt is not None:
        # For RNNs with return_type="both", add hidden variable as 5th element
        if (hasattr(layer, 'return_type') and layer.return_type == "both"):
            # Generate hidden variable name
            hidden_var = f"{out_layer}_h" if out_layer != "x" else "h"
            modules_details[layer.name + "_layer"] = [layer_synt, out_layer,
                                                      in_layer, layer, hidden_var]
        else:
            modules_details[layer.name + "_layer"] = [layer_synt, out_layer,
                                                      in_layer, layer]
    if actv_func_syntax:
        modules_details[layer.name + "_activ"] = [actv_func_syntax, out_actv,
                                                  in_actv]

    # TF-specific: Add separate activation for BatchNorm/LayerNorm
    if hasattr(setup, 'add_separate_activation_if_needed'):
        setup.add_separate_activation_if_needed(out_actv, in_actv)

    if setup.permute_out and hasattr(setup, 'add_permute'):
        if channel_last is None or channel_last:
            dim = setup.dim
            setup.add_permute(layer.name, dim, out_layer, permute_in = False,
                              sequential=is_seq, is_subnn=is_subnn)



def get_tensorop_params(tensorop: TensorOp, modules_details: dict):
    """
    It retrieves tensorops parameters that are used by
    `get_tensorop_syntax` function defined in PyTorch and
    TensorFlow `utils.py` files.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.

    Returns:
        - previous output variable and the parameters of the tensorop.

    """
    if len(list(modules_details.keys())) == 0:
        prev_out_var = "x"
    else:
        prev_module = list(modules_details.keys())[-1]

        # Skip shape_dim operations to find the actual previous tensor-producing module
        if _is_shape_dim_operation(prev_module, modules_details):
            actual_prev = _find_previous_non_shape_dim_module(modules_details)
            if actual_prev:
                prev_module = actual_prev

        prev_out_var = get_previous_out_var(modules_details, prev_module)

    # Check if tensorop has name_module_input set (similar to layers)
    # If not, try to derive it from layers_of_tensors for single-input tensorops
    module_input = None
    if hasattr(tensorop, 'name_module_input') and tensorop.name_module_input is not None:
        module_input = tensorop.name_module_input
    elif hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
        # For single-input tensorops, use the first (and only) input
        # For multi-input tensorops, check if any input is INPUT
        if len(tensorop.layers_of_tensors) == 1:
            module_input = tensorop.layers_of_tensors[0]
        elif 'INPUT' in tensorop.layers_of_tensors:
            module_input = 'INPUT'

    if module_input is not None:
        if module_input == 'INPUT':
            # Use dedicated variable for preserved input
            # Generator will need to add: inp = x at the start
            prev_out_var = 'inp'
        else:
            # Check for RNN hidden/cell state suffixes
            if module_input.endswith("__hidden"):
                # Extract base module name and get hidden variable
                base_module = module_input[:-8]  # Remove "__hidden"
                modules_names = list(modules_details.keys())
                if f"{base_module}_layer" in modules_names:
                    # RNN layers with return_type="both" have hidden_var at index 4
                    layer_details = modules_details[f"{base_module}_layer"]
                    if len(layer_details) > 4:
                        prev_out_var = layer_details[4]
                    else:
                        # Fallback to regular output if hidden var not available
                        prev_out_var = layer_details[1]
                else:
                    prev_out_var = "x"
            elif module_input.endswith("__cell"):
                # LSTM cell state - for now use same logic as hidden
                # (may need separate tracking in the future)
                base_module = module_input[:-6]  # Remove "__cell"
                modules_names = list(modules_details.keys())
                if f"{base_module}_layer" in modules_names:
                    layer_details = modules_details[f"{base_module}_layer"]
                    if len(layer_details) > 4:
                        # For LSTM, cell state would need separate tracking
                        # For now, use hidden var as placeholder
                        prev_out_var = layer_details[4]
                    else:
                        prev_out_var = layer_details[1]
                else:
                    prev_out_var = "x"
            else:
                # Use get_input_var logic for tensorops too
                modules_names = list(modules_details.keys())
                if f"{module_input}_layer" in modules_names:
                    prev_out_var = modules_details[f"{module_input}_layer"][1]
                elif f"{module_input}_nn" in modules_names:
                    prev_out_var = modules_details[f"{module_input}_nn"]["in_out_variable"]
                elif f"{module_input}_activ" in modules_names:
                    prev_out_var = modules_details[f"{module_input}_activ"][1]
                elif f"{module_input}_op" in modules_names:
                    prev_out_var = modules_details[f"{module_input}_op"][1]

    tns_type = tensorop.tns_type
    if tns_type == "reshape":
        # If reshape has layers_of_tensors set and it's not INPUT, use it to determine source variable
        # (INPUT case is handled by name_module_input logic above)
        if (hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors and
            tensorop.layers_of_tensors[0] != 'INPUT'):
            source_layer = tensorop.layers_of_tensors[0]
            modules_names = list(modules_details.keys())
            if f"{source_layer}_layer" in modules_names:
                prev_out_var = modules_details[f"{source_layer}_layer"][1]
            elif f"{source_layer}_op" in modules_names:
                prev_out_var = modules_details[f"{source_layer}_op"][1]
        # Resolve operation names to their actual variable names
        resolved_dims = []
        for dim in tensorop.reshape_dim:
            if isinstance(dim, str) and f"{dim}_op" in modules_details:
                # This is an operation name - use its output variable
                resolved_dims.append(modules_details[f"{dim}_op"][1])
            else:
                resolved_dims.append(str(dim))
        params = ', '.join(resolved_dims)
    elif tns_type == "concatenate":
        actual_vars = getattr(tensorop, 'actual_vars', None)
        tensors = get_layers_output_for_tensorops(tensorop.layers_of_tensors,
                                                  modules_details,
                                                  actual_vars)
        params = ', '.join(tensors)
    elif tns_type == "transpose":
        # Transpose may have layers_of_tensors set
        if hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
        params = ", ".join([str(i) for i in tensorop.transpose_dim])
    elif tns_type == "permute":
        params = ", ".join([str(i) for i in tensorop.permute_dim])
    elif tns_type == "mean":
        # Mean operates on prev_out_var, but may have layers_of_tensors set
        if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
        params = ""
    elif tns_type == "max":
        # Max operates on prev_out_var, but may have layers_of_tensors set
        if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
        params = ""
    elif tns_type == "squeeze" or tns_type == "unsqueeze":
        # Squeeze/unsqueeze operate on prev_out_var, but may have layers_of_tensors set
        if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
                elif f"{source_layer}_activ" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_activ"][1]
        params = ""
    elif tns_type == "subscript":
        # Subscript operates on prev_out_var, pattern is in subscript_indices
        # Only override prev_out_var if name_module_input wasn't already set
        if not (hasattr(tensorop, 'name_module_input') and tensorop.name_module_input is not None):
            if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
                source_layer = tensorop.layers_of_tensors[0]
                if source_layer == 'INPUT':
                    prev_out_var = 'inp'
                elif (source_layer + "_layer" in modules_details or
                      source_layer + "_op" in modules_details or
                      source_layer + "_activ" in modules_details):
                    prev_out_var = get_layers_output_for_tensorops([source_layer], modules_details)[0]
                else:
                    prev_out_var = source_layer
        params = ""
    elif tns_type == "shape_dim":
        # Shape extraction: layers_of_tensors[0] contains the source to extract shape from
        if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
                elif f"{source_layer}_activ" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_activ"][1]
                else:
                    # Source not found in modules - use it directly (e.g., variable name 'x')
                    # Treat as input variable
                    prev_out_var = 'inp' if source_layer == 'x' else source_layer
        params = ""
    elif tns_type == "normalize":
        # Normalize operates on prev_out_var, resolve from layers_of_tensors
        if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
                elif f"{source_layer}_activ" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_activ"][1]
        params = ""
    elif tns_type == "repeat":
        # Repeat operates on prev_out_var, resolve from layers_of_tensors
        if tensorop.layers_of_tensors and isinstance(tensorop.layers_of_tensors[0], str):
            source_layer = tensorop.layers_of_tensors[0]
            if source_layer == 'INPUT':
                prev_out_var = 'inp'
            else:
                modules_names = list(modules_details.keys())
                if f"{source_layer}_layer" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_layer"][1]
                elif f"{source_layer}_op" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_op"][1]
                elif f"{source_layer}_activ" in modules_names:
                    prev_out_var = modules_details[f"{source_layer}_activ"][1]
        # Resolve operation names in repeat_dim (similar to reshape_dim)
        resolved_multiples = []
        for mult in tensorop.repeat_dim:
            if isinstance(mult, str) and f"{mult}_op" in modules_details:
                # This is an operation name - use its output variable
                resolved_multiples.append(modules_details[f"{mult}_op"][1])
            else:
                resolved_multiples.append(str(mult))
        params = ', '.join(resolved_multiples)
    elif tns_type in ["interpolate", "pad", "dropout"]:
        # These operations don't use layers_of_tensors or params
        params = None
    else:
        tensors = tensorop.layers_of_tensors
        if isinstance(tensors[0], str):
            # Check if we need to use actual_vars for output vs hidden selection
            actual_vars = getattr(tensorop, 'actual_vars', None)
            tensors = get_layers_output_for_tensorops(tensors,
                                                      modules_details,
                                                      actual_vars)

        params = ', '.join([str(i) for i in tensors])
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
                    referenced_tensorops: set | None = None):
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

    Returns:
        None, but stores the tensorop details in the modules_details dict.

    """
    ts_op_synt = get_tensorop_syntax(tensorop, modules_details, out_var)
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
            # Binary operations always get unique op_ prefix to avoid conflicts
            elif tensorop.tns_type.startswith("binop_"):
                out_var = tensorop.name
            else:
                out_var = get_tensorop_out_var(tensorop, prev_out_var, modules_details)

    modules_details[tensorop.name + "_op"] = [ts_op_synt, out_var, tensorop]


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
