"""
Helper functions to transform NN TensorFlow code to BUML code.
"""

from besser.generators.nn_reverse.tf2buml.definitions import lookup_layers, \
    lookup_layers_params, layers_fixed_params, rnn_cnn_layers, \
    layers_specific_params
from besser.generators.nn_reverse.code2buml.utils import (
    handle_remaining_params
)

def transform_layers(layers, inputs_outputs, layer_of_output, is_layer):
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
    return layers

def wrap_transform_layers(layers, inputs_outputs, layer_of_output,
                          _modules=None, is_layer=True):
    """
    It wraps the 'transform_layers' function so that it returns two 
    outputs instead of one. It is used to unify the outputs of 
    'transform_layers' functions from both Torch and TensorFlow.
    """
    return (transform_layers(layers, inputs_outputs, layer_of_output,
                             is_layer),
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
        #"permute_dim" not needed in tf
        if param in ["activation", "return_sequences", "return_state",
                     "units", "permute_dim"]:
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
    if layer_type[:-5] in rnn_cnn_layers[:3]:
        if "return_type" not in layer_params:
            layer_params["return_type"] = "last"
    return layer_params


def handle_conv_padding(layer_elems, padding_amount):
    """
    If padding is used before conv layers, it is amount is stored in
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
