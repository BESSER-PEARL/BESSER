"""
This module provides helper functions to convert BUML code
to PyTorch or TensorFlow code.
"""

import os
import random

from PIL import Image
import numpy as np


def get_previous_out_var(modules_details, prev_module):
    """
    It retrieves the output variable of the previous module in order to
    use it as the input variable of the current module.
    """
    if isinstance(modules_details[prev_module], dict):
        return modules_details[prev_module]["in_out_variable"]
    else:
        return modules_details[prev_module][1]

def get_input_var(layer, modules_details, prev_out_var):
    """
    It determines the input variable of the current layer. It is either
    the output variable of the module in `name_module_input` attribute 
    (if it is given), or simply the output of the previous module given
    by `get_previous_out_variable` function.
    """
    modules_names = list(modules_details.keys())
    lyr_input = layer.name_module_input
    if lyr_input is not None:
        if f"{lyr_input}_layer" in modules_names:
            return modules_details[f"{lyr_input}_layer"][1]
        if f"{lyr_input}_nn" in modules_names:
            return  modules_details[f"{lyr_input}_nn"]["in_out_variable"]
        #module.name_module_input+"_op" in my_keys
        return modules_details[f"{lyr_input}_op"][1]
    return prev_out_var

def add_in_out_var_to_subnn(modules_details):
    """
    It sets the in_out_variable of subnns, which refers to the input
    and output variable of the subnn.
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
    return modules_details

def get_layers_output_for_tensorops(layers_names, modules_details):
    """
    It retrieves the output variables of the layers in `layers_name`
    list to use them as input of the tensorop.
    """
    my_keys = list(modules_details.keys())
    out_vars = []
    for layer_name in layers_names:
        if layer_name+"_layer" in my_keys:
            out_vars.append(modules_details[layer_name + "_layer"][1])
        else:
            out_vars.append(modules_details[layer_name + "_op"][1])
    return out_vars

def initialize_tensorop_var(tensorop):
    """
    It sets the output variable of the tensorop in the case it is the
    first module in the neural network.
    """
    if tensorop.input_reused is True:
        out_var = "x_1"
    else:
        out_var = "x"
    return out_var


def get_out_var_input_reused(prev_out_var):
    """
    It sets the output variable of the module in the case the output
    of the previous module is reused (therefore, they need to be 
    different).
    """
    if prev_out_var == "x":
        out_var = "x_1"
    else:
        out_var = f"x_{int(prev_out_var.split('_')[-1])+1}"
    return out_var


def get_layer_vars(layer, prev_out_var, modules_details):
    """
    It sets the input and output variables of the layer.
    """
    out_var_actv, in_var_actv = None, None
    if layer.input_reused:
        out_var_layer = get_out_var_input_reused(prev_out_var)
    else:
        out_var_layer = prev_out_var
    in_var_layer = get_input_var(layer, modules_details,
                                           prev_out_var)
    if layer.actv_func is not None:
        out_var_actv, in_var_actv = out_var_layer, out_var_layer
    return out_var_layer, in_var_layer, out_var_actv, in_var_actv

def initialize_layer_vars(layer):
    """
    It sets the input and output variables of layer (and activation 
    function for PyTorch) in the case it is the first module in 
    the neural network.
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

def get_layer_syntax(setup_layer_cls, layer, modules_details,
                     in_layer, actv_func_synt):
    """
    It retrieves the syntax of the layer (and the activation 
    function in the case of PyTorch) from the ´setup_layer_cls´ 
    class.
    """
    setup = setup_layer_cls(layer, modules_details)
    parent_class = layer.__class__.mro()[1].__name__
    if (parent_class == "ConvolutionalLayer" or parent_class == "CNN"):
        layer_synt, modules_details = setup.setup_cnn()
    elif parent_class == "RNN":
        layer_synt, modules_details = setup.setup_rnn()
    elif parent_class == "GeneralLayer":
        layer_synt = setup.setup_general_layer()
    else: #(parent_class == "LayerModifier" or
           #parent_class == "NormalizationLayer")
        layer_synt = setup.setup_layer_modifier()

    if actv_func_synt is not None:
        actv_func_synt = setup.setup_actv_func()
    return layer_synt, actv_func_synt, modules_details, setup

def handle_layer(layer, setup_layer, modules_details, actv_func_syntax=None):
    """
    It populates the `modules_details` dictionary with layer's 
    information: Its syntax, input and output variables, and the 
    layer class.
    In the case of PyTorch, the activation function is treated as 
    a layer.
    """

    if len(modules_details) == 0:
        out_layer, in_layer, out_actv, in_actv = initialize_layer_vars(layer)
    else:
        prev_module = list(modules_details.keys())[-1]
        prev_out_var = get_previous_out_var(modules_details, prev_module)
        out_layer, in_layer, out_actv, in_actv = get_layer_vars(
            layer, prev_out_var, modules_details)

    layer_synt, actv_func_syntax, modules_details, setup = get_layer_syntax(
        setup_layer, layer, modules_details, in_layer, actv_func_syntax)

    if setup.permute_in:
        dim = setup.dim
        setup.add_permute(layer.name, dim, in_layer)

    modules_details[layer.name + "_layer"] = [layer_synt, out_layer,
                                              in_layer, layer]
    if actv_func_syntax is not None:
        modules_details[layer.name + "_activ"] = [actv_func_syntax, out_actv,
                                                  in_actv]
    if setup.permute_out:
        dim = setup.dim
        setup.add_permute(layer.name, dim, out_layer)
    return modules_details


def get_tensorop_params(tensorop, modules_details):
    """
    It retrieves tensorops parameters that are used by 
    `get_tensorop_syntax` function defined in PyTorch and 
    TensorFlow `utils.py` files
    """
    if len(list(modules_details.keys())) == 0:
        prev_out_var = "x"
    else:
        prev_module = list(modules_details.keys())[-1]
        prev_out_var = get_previous_out_var(modules_details, prev_module)
    tns_type = tensorop.tns_type
    if tns_type == "reshape":
        params = ', '.join([str(i) for i in tensorop.reshape_dim])
    elif tns_type == "concatenate":
        tensors = get_layers_output_for_tensorops(tensorop.layers_of_tensors,
                                                  modules_details)
        params = ', '.join(tensors)
    elif tns_type == "transpose":
        params = ", ".join([str(i) for i in tensorop.transpose_dim])
    elif tns_type == "permute":
        params = ", ".join([str(i) for i in tensorop.permute_dim])
    else:
        tensors = tensorop.layers_of_tensors
        if isinstance(tensors[0], str):
            tensors = get_layers_output_for_tensorops(tensorop,
                                                      modules_details)

        params = ', '.join([str(i) for i in tensors])
    return prev_out_var, params


def get_tensorop_out_var(tensorop, prev_out_var):
    """
    It sets the output variable of tensorop.
    """
    if tensorop.input_reused is True:
        out_var  = get_out_var_input_reused(prev_out_var)
    else:
        out_var = prev_out_var
    return out_var

def handle_tensorop(tensorop, modules_details,
                    get_tensorop_syntax, out_var=None):
    """
    It populates the `modules_details` dictionary with tensorop's 
    information: Its syntax and output variable.
    """
    ts_op_synt = get_tensorop_syntax(tensorop, modules_details, out_var)
    if out_var is None:
        if len(modules_details) == 0:
            out_var  = initialize_tensorop_var(tensorop)
        else:
            prev_module = list(modules_details.keys())[-1]
            prev_out_var = get_previous_out_var(modules_details, prev_module)
            out_var = get_tensorop_out_var(tensorop, prev_out_var)

    modules_details[tensorop.name + "_op"] = [ts_op_synt, out_var]
    return modules_details


def preprocess_image(image_path, target_size):
    """
    It resizes and returns the images as np arrays.
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    np_image = np.array(image.convert('RGB'), dtype=np.float32)
    return np_image



def compute_mean_std(image_dir, num_samples=100, target_size=(256, 256)):
    """
    It computes the mean and standard deviation of images and checks
    whether scaling is needed.
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


def format_value(elem):
    """
    It formats BUML list of int. If it contains one element, it is
    returned as `int`. Otherwise, it converts the list to a tuple.
    """
    if len(elem) == 1:
        return elem[0]
    return tuple(elem)
