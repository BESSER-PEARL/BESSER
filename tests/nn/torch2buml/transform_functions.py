from tests.nn.torch_to_buml.definitions import lookup_layers, lookup_layers_params, \
    layers_fixed_params, rnn_cnn_layers


def set_default_rnn_return_type(layer_type, layer_params):
    #check if the layer is an rnn layer
    if layer_type[:-5] in rnn_cnn_layers[:3]:
        if "return_type" not in layer_params:
            layer_params["return_type"] = "full"
    return layer_params

def param_int_to_list(layers):
    params_to_convert = ["kernel_size", "stride", "output_size", "normalized_shape"]
    layers_of_params = ["Conv1d", "MaxPool1d", "AvgPool1d", 
                        "AdaptiveAvgPool1d", "AdaptiveMaxPool1d", "LayerNorm"]
    
    for layer_elems in layers.values():
        if layer_elems[0] in layers_of_params:
            for param in layer_elems[1]:
                if param in params_to_convert and isinstance(layer_elems[1][param], int):
                    layer_elems[1][param] = [layer_elems[1][param]]
    return layers
    





def handle_params(layer_elems, layer_name, layer_type, inputs_outputs, layer_of_output, is_layer=True):
    layer_params = {}
    layer_elems_cp = layer_elems[1].copy() #the dict changes size during iteration, a copy is needed.
    for param in layer_elems_cp:
        if layer_type in layers_specific_params:
            if param in layers_specific_params[layer_type]:
                layer_params[layers_specific_params[layer_type][param]] = layer_elems[1][param]
        if param == "activation":
            layer_params["actv_func"] = layer_elems[1][param]
        if param == "return_sequences" and layer_elems[1][param] == True:
            layer_params["return_type"]="full"
        if param == "return_state" and layer_elems[1][param] == True:
            layer_params["return_type"]="hidden"
        #"permute_dim" not needed in tf    
        if param in ["activation", "return_sequences", "return_state", "units", "permute_dim"]:
            del layer_elems[1][param]
        elif param in lookup_layers_params:
            layer_params[lookup_layers_params[param]] = layer_elems[1][param] 
        else:
            print(f"parameter {param} of layer {layer_name} is not found!")
    if layer_elems[0] in layers_fixed_params:
        layer_params.update(layers_fixed_params[layer_elems[0]])
    if is_layer:            
        if inputs_outputs[layer_name][0] != inputs_outputs[layer_name][1]:
            layer_params["input_reused"] = True
            layer_params["name_layer_input"] = layer_of_output[inputs_outputs[layer_name][0]] 
    return layer_params




def transform_layers(layers, inputs_outputs, layer_of_output, 
                     is_layer=True):
    layers = param_int_to_list(layers)
    layers_copy = layers.copy()
    for layer_name, layer_elems in layers_copy.items():
        layer_type = lookup_layers[layer_elems[0]]
        layer_params = {}
        for param in layer_elems[1]:
            if param in lookup_layers_params:
                layer_params[lookup_layers_params[param]] = layer_elems[1][param]
            elif param == "actv_func":
                layer_params[param] = layer_elems[1][param]
            else:
                print(f"parameter {param} of layer {layer_name} is not found!")
        if layer_elems[0] in layers_fixed_params:
            layer_params.update(layers_fixed_params[layer_elems[0]])
        if is_layer:            
            if inputs_outputs[layer_name][0] != inputs_outputs[layer_name][1]:
                layer_params["input_reused"] = True
                layer_params["name_layer_input"] = layer_of_output[inputs_outputs[layer_name][0]] 
        layer_params = set_default_rnn_return_type(layer_type, layer_params)
        layers[layer_name] = [layer_type, layer_params]
    return layers













def adjust_layers_ids(layers):
    layers_new = {}
    counter = 1
    for layer_elems in layers.values():
        layers_new[counter] = layer_elems
        counter+=1
    return layers_new


def format_params(layer_params):
    formatted_params = []
    for key, value in layer_params.items():
        if isinstance(value, str):
            if key == "path_data":
                formatted_params.append(f"{key}=r'{value}'")
            else:
                formatted_params.append(f"{key}='{value}'")
        else:
            formatted_params.append(f"{key}={value}")
    layer_params_str = ', '.join(formatted_params)
    return layer_params_str

def get_imports(extractor):
    cls_to_import = set()
    modules = extractor.modules
    configuration = extractor.data_config["config"]
    train_data = extractor.data_config["train_data"]
    test_data =  extractor.data_config["test_data"]
    for layer_elem in modules["layers"].values():
        cls_to_import.add(layer_elem[0])
    if modules["sub_nns"]:
        for subnn_elem in modules["sub_nns"].values():
            for layer_elem in subnn_elem.values():
                cls_to_import.add(layer_elem[0])
    if modules["tensorops"]:
        cls_to_import.add("TensorOp")
    if configuration:
        cls_to_import.add("Configuration")
    if len(train_data)>1 and len(test_data)>1:
        cls_to_import.add("Dataset")
        if train_data["input_format"] == "images":
            cls_to_import.add("Image")
    cls_to_import = ["NN"] + list(cls_to_import)
    return cls_to_import