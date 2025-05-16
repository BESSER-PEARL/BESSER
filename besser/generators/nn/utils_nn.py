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
        return modules_details[prev_module][1]

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
    if lyr_input is not None:
        if f"{lyr_input}_layer" in modules_names:
            return modules_details[f"{lyr_input}_layer"][1]
        if f"{lyr_input}_nn" in modules_names:
            return  modules_details[f"{lyr_input}_nn"]["in_out_variable"]
        #module.name_module_input+"_op" in my_keys
        return modules_details[f"{lyr_input}_op"][1]
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


def get_layers_output_for_tensorops(layers_names: list, modules_details: dict):
    """
    It retrieves the output variables of the layers in `layers_name`
    list to use them as input of the tensorop.

    Arguments:
        layers_names (list): Names of layers on which the tensorop is applied.
        modules_details (dict): A dict storing the NN modules syntax and 
            attributes.

    Returns:
        The output variables of the layers in 'layers_names'.
        
    """
    my_keys = list(modules_details.keys())
    out_vars = []
    for layer_name in layers_names:
        if layer_name+"_layer" in my_keys:
            out_vars.append(modules_details[layer_name + "_layer"][1])
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


def get_out_var_input_reused(prev_out_var: str):
    """
    It sets the output variable of the module in the case the output
    of the previous module is reused (therefore, they need to be 
    different).

    Arguments:
        prev_out_var (str): The previous output variable.
    Returns:
        The current output variable.
        
    """
    if prev_out_var == "x":
        out_var = "x_1"
    else:
        out_var = f"x_{int(prev_out_var.split('_')[-1])+1}"
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
        out_var_layer = get_out_var_input_reused(prev_out_var)
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
                     actv_func_synt: str | bool ):
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

    Returns:
        The syntax of the layer and its activation function (if relevant) and
        the class instance.

    """
    setup = setup_layer_cls(layer, modules_details)
    parent_class = layer.__class__.mro()[1].__name__
    if (parent_class == "ConvolutionalLayer" or parent_class == "CNN"):
        layer_synt = setup.setup_cnn()
    elif parent_class == "RNN":
        layer_synt = setup.setup_rnn()
    elif parent_class == "GeneralLayer":
        layer_synt = setup.setup_general_layer()
    else: #(parent_class == "LayerModifier" or
           #parent_class == "NormalizationLayer")
        layer_synt = setup.setup_layer_modifier()

    if actv_func_synt:
        actv_func_synt = setup.setup_actv_func()

    return layer_synt, actv_func_synt, setup

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
        prev_out_var = get_previous_out_var(modules_details, prev_module)
        out_layer, in_layer, out_actv, in_actv = get_layer_vars(
            layer, prev_out_var, modules_details
        )

    layer_synt, actv_func_syntax, setup = get_layer_syntax(
        setup_layer, layer, modules_details, actv_func_syntax
    )

    if setup.permute_in and channel_last:
        dim = setup.dim
        setup.add_permute(
            layer.name, dim, in_layer, permute_in=True,
            sequential=is_seq, is_subnn=is_subnn
        )

    modules_details[layer.name + "_layer"] = [layer_synt, out_layer,
                                              in_layer, layer]
    if actv_func_syntax:
        modules_details[layer.name + "_activ"] = [actv_func_syntax, out_actv,
                                                  in_actv]
    if setup.permute_out and channel_last:
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
            tensors = get_layers_output_for_tensorops(tensors,
                                                      modules_details)

        params = ', '.join([str(i) for i in tensors])
    return prev_out_var, params


def get_tensorop_out_var(tensorop: TensorOp, prev_out_var: str):
    """
    It sets the output variable of tensorop.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.
        prev_out_var (str): previous output variable.

    Returns:
        - The current output variable.
        
    """
    if tensorop.input_reused is True:
        out_var  = get_out_var_input_reused(prev_out_var)
    else:
        out_var = prev_out_var
    return out_var

def handle_tensorop(tensorop: TensorOp, modules_details: dict,
                    get_tensorop_syntax: callable, out_var: str | None = None):
    """
    It populates the `modules_details` dictionary with tensorop's 
    information: Its syntax and output variable.

    Arguments:
        tensorop (TensorOp): The BUML tensorop object.
        modules_details (dict): A dict storing the NN modules syntax and 
            attributes.
        out_var (str | None): The output variable of the tensorop.

    Returns:
        None, but stores the tensorop details in the modules_details dict.
        
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
