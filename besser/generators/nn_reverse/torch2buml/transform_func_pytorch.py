"""
Helper functions to transform NN Torch code to BUML code.
"""

from besser.generators.nn_reverse.torch2buml.definitions import (
    lookup_layers, lookup_layers_params, layers_fixed_params, lookup_actv_fun,
    pos_params
)
from besser.generators.nn_reverse.code2buml.utils_code2buml import (
    handle_remaining_params, handle_positional_params
)

def transform_layers(layers, inputs_outputs, layer_of_output,
                     order, is_layer):
    """
    It processes the information related to layers and transforms the layers'
    code from Torch to BUML.
    """
    layers = param_to_list(layers)
    layers_copy = layers.copy()
    prev_layer = None
    for layer_name, layer_elems in layers_copy.items():
        if layer_elems[0] in lookup_actv_fun:
            activ_func = lookup_actv_fun[layer_elems[0]]
            layers[prev_layer][1]["actv_func"] = activ_func
            del layers[layer_name]
            if is_layer:
                del order[layer_name]
        else:
            layer_type = lookup_layers[layer_elems[0]]
            layer_params = handle_params(layer_elems, layer_name,
                                         inputs_outputs, layer_of_output,
                                         is_layer)
            layers[layer_name] = [layer_type, layer_params]
            prev_layer = layer_name
    return layers, order


def handle_params(layer_elems, layer_name, inputs_outputs,
                  layer_of_output, is_layer=True):
    """It processes and transforms the layers' parameters"""
    layer_params = {}
    lyr_type = layer_elems[0]

    layer_elems = handle_positional_params(lyr_type, layer_elems, pos_params)

    for param in layer_elems[1]:
        if param in lookup_layers_params:
            layer_params[lookup_layers_params[param]] = layer_elems[1][param]
        elif param == "actv_func":
            layer_params[param] = layer_elems[1][param]
        else:
            print(f"parameter {param} of layer {layer_name} is not found!")

    layer_params = handle_remaining_params(layer_params, lyr_type,
                                           layer_name, inputs_outputs,
                                           layer_of_output,
                                           layers_fixed_params, is_layer)
    return layer_params



def param_to_list(layers):
    """It converts some int parameters to list format for BUML"""
    params_to_convert = ["kernel_size", "stride", "output_size",
                         "normalized_shape"]
    layers_of_params = ["Conv1d", "MaxPool1d", "AvgPool1d",
                        "AdaptiveAvgPool1d", "AdaptiveMaxPool1d", 
                        "LayerNorm"]

    for layer_elems in layers.values():
        if layer_elems[0] in layers_of_params:
            for param in layer_elems[1]:
                if (param in params_to_convert and
                    isinstance(layer_elems[1][param], int)):
                    layer_elems[1][param] = [layer_elems[1][param]]
    return layers


def transform_actv_func(activ_func, modules, activation_functions,
                        sub_nn = None, in_forward=True):
    """
    It adds the activation function as parameter to the previous 
    layer.
    """

    if in_forward:
        layers = modules["layers"]
        modules = modules["order"]
        activ_func = activation_functions[activ_func]
        previous_layer = list(modules.keys())[-1]
        previous_layer_param = layers[previous_layer][1]
        previous_layer_param["actv_func"] = lookup_actv_fun[activ_func]
    else:
        last_layer_name = list(sub_nn.keys())[-1]
        #In sub_nns, the activation function does not have a name.
        #We get the name of its layer and add it as param.
        activ_func_buml = lookup_actv_fun[activ_func]
        sub_nn[last_layer_name][1]["actv_func"] = activ_func_buml
