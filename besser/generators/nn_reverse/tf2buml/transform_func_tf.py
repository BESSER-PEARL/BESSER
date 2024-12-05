"""
Helper functions to transform NN TensorFlow code to BUML code.
"""

from besser.generators.nn_reverse.tf2buml.definitions import lookup_layers, \
    lookup_layers_params, layers_fixed_params, rnn_layers, \
    layers_specific_params
from besser.generators.nn_reverse.code2buml.utils_code2buml import (
    handle_remaining_params
)

def transform_layers(layers, inputs_outputs, layer_of_output,
                     modules, subnns, is_layer):
    """
    It processes the information related to layers and transforms the layers'
    code from TensorFlow to BUML.
    """
    padding_amount = None
    layers = param_int_to_list(layers)
    layers_copy = layers.copy()
    for layer_name, layer_elems in layers_copy.items():
        layer_elems, padding_amount = handle_conv_padding(layer_elems,
                                                          padding_amount)
        if layer_elems[0].startswith("ZeroPadding"):
            del layers[layer_name]
        else:
            layer_type = lookup_layers[layer_elems[0]]
            layer_params = handle_params(layer_elems, layer_name,
                                         inputs_outputs, layer_of_output,
                                         is_layer)
            layer_params = set_default_rnn_return_type(layer_type,
                                                       layer_params)
            layers[layer_name] = [layer_type, layer_params]
    if not subnns:
        layers = add_permute_dim(modules, layers)
    return layers


def add_permute_dim(modules, layers):
    """
    Check whether input and output of cnn layers needs to be permuted 
    to make pytorch and tensorflow equivalent.
    """
    modules_names = list(modules.keys())
    prev_module = None
    next_module = None
    cnns = ["Conv1D", "Conv2D", "Conv3D", "PoolingLayer"]
    permuted = []

    for i, name in enumerate(modules_names[:-1]):
        next_module = modules_names[i+1]
        current_cnn, prev_cnn, next_cnn = False, False, False

        if next_module in layers:
            if layers[next_module][0] in cnns:
                next_cnn = True

        if name in layers:
            if layers[name][0] in cnns:
                current_cnn = True
                if "name_module_input" in layers[name][1]:
                    prev_module = layers[name][1]["name_module_input"]
                if next_module in layers:
                    if "name_module_input" in layers[next_module][1]:
                        if layers[next_module][1]["name_module_input"] != name:
                            next_cnn = False

        if prev_module in layers:
            if layers[prev_module][0] in cnns:
                prev_cnn = True
        elif prev_module is None:
            prev_cnn = False

        if current_cnn:
            if not prev_cnn:
                if prev_module not in permuted:
                    layers[name][1]["permute_in"] = True
                    permuted.append(prev_module)
            if not next_cnn:
                layers[name][1]["permute_out"] = True

        prev_module = name

    return layers


def wrap_transform_layers(layers, inputs_outputs, layer_of_output,
                          modules, subnns=None, is_layer=True):
    """
    It wraps the 'transform_layers' function so that it returns two 
    outputs instead of one. It is used to unify the outputs of 
    'transform_layers' functions from both Torch and TensorFlow.
    """
    return (transform_layers(layers, inputs_outputs, layer_of_output,
                             modules, subnns, is_layer),
            None)

def handle_params(layer_elems, layer_name, inputs_outputs, layer_of_output,
                  is_layer=True):
    """It processes and transforms the layers' parameters"""
    layer_params = {}
    #the layer_elems[1] dict changes size during iteration, a copy is needed.
    layer_elems_cp = layer_elems[1].copy()
    layer_type = layer_elems[0]
    for param in layer_elems_cp:
        if layer_type in layers_specific_params:
            if param in layers_specific_params[layer_type]:
                param_name = layers_specific_params[layer_type][param]
                layer_params[param_name] = layer_elems[1][param]
        if param == "activation":
            layer_params["actv_func"] = layer_elems[1][param]
        elif param == "return_sequences" and layer_elems[1][param] is True:
            layer_params["return_type"] = "full"
        elif param == "return_state" and layer_elems[1][param] is True:
            layer_params["return_type"] = "hidden"
        if param in ["activation", "return_sequences", "return_state",
                     "units"]:
            del layer_elems[1][param]
        elif param in lookup_layers_params:
            param_name = lookup_layers_params[param]
            layer_params[param_name] = layer_elems[1][param]
        else:
            print(f"parameter {param} of layer {layer_name} is not found!")

    layer_params = handle_remaining_params(layer_params, layer_type,
                                           layer_name, inputs_outputs,
                                           layer_of_output,
                                           layers_fixed_params, is_layer)
    return layer_params


def param_int_to_list(layers):
    """It converts some int parameters to list format for BUML"""
    params_to_convert = ["kernel_size", "strides", "pool_size",
                         "output_size", "axis"]
    layers_of_params = ["Conv1D", "MaxPool1D", "AveragePooling1D",
                        "AdaptiveAveragePooling1D", "AdaptiveMaxPooling1D", 
                        "LayerNormalization"]

    for layer_elems in layers.values():
        if layer_elems[0] in layers_of_params:
            for param in layer_elems[1]:
                if (param in params_to_convert and
                    isinstance(layer_elems[1][param], int)):
                    layer_elems[1][param] = [layer_elems[1][param]]
    return layers


def set_default_rnn_return_type(layer_type, layer_params):
    """
    It sets a default value for the 'return_type' RNN attribute if
    it could not be infered from the code.
    """
    #check if the layer is an rnn layer
    if layer_type[:-5] in rnn_layers[:3]:
        if "return_type" not in layer_params:
            layer_params["return_type"] = "last"
    return layer_params


def handle_conv_padding(layer_elems, padding_amount):
    """
    If padding is used before conv layers, its amount is stored in
    the 'padding_amount' attribute.
    """
    layer_type = layer_elems[0]
    if layer_type.startswith("ZeroPadding"):
        padding_amount = layer_elems[1]["padding"]
    elif layer_type.startswith("Conv"):
        if padding_amount is not None:
            layer_elems[1]["padding_amount"] = padding_amount
            padding_amount = None
    return layer_elems, padding_amount
