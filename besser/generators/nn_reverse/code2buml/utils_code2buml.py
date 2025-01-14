"""
Helper functions to transform NN code to BUML code.
"""

import argparse
import ast
from besser.generators.nn_reverse.tf2buml.missing_attr import get_attributes


def parse_arguments_code2buml():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Read a NN file and convert it to buml code."
    )
    parser.add_argument(
        "filename", type=str, help="Path to the file to read"
    )
    parser.add_argument(
        "--datashape", type=parse_tuple, default=None, 
        help=(
            "The shape of the input data (optional)."
            "It is needed when transforming tf code to pytorch code"
            "It is used to recover some layer attributes dynamically"
        ),
    )

    return parser.parse_args()

def parse_tuple(value):
    """Parse a tuple from a string input"""
    return (1,) + tuple(map(int, value.strip("()").split(",")))

def code2buml(args, ast_parser_class, framework, transform_layers,
              config_list, train_param_list, test_param_list,
              lookup_loss_func):
    """
    Tt gets the AST, processes the data and prints 
    the BUML transformed code.
    """

    # Read the NN code from an external file
    with open(args.filename, "r", encoding="utf-8") as file:
        code = file.read()

    shape = args.datashape

    # Parse the code and use the extractor
    tree = ast.parse(code)
    extractor = ast_parser_class()
    extractor.visit(tree)

    nn_name = extractor.nn_name
    if framework == "TF":
        extractor = get_attributes(extractor, args.filename, shape)
    if extractor.input_nn_type == "sequential":
        if extractor.modules["sub_nns"]:
            extractor.modules["sub_nns"].pop(nn_name)

    extractor = transform_modules(extractor, transform_layers)

    print_imports(extractor, framework)
    print_subnn_def(extractor.modules["sub_nns"])
    print_nn_def(extractor)

    if (extractor.data_config["config"] and
        extractor.data_config["train_data"] and
        extractor.data_config["test_data"]):
        print_config(extractor.data_config["config"], nn_name,
                     lookup_loss_func, config_list)
        print_data(extractor.data_config["train_data"],
                   extractor.data_config["test_data"], nn_name,
                   train_param_list, test_param_list)
    return nn_name, extractor.output_nn_type


def transform_modules(extractor, transform_layers):
    """
    It transforms names and parameters of modules to BUML.
    """

    for sub_nn_name, sub_nn_elems in extractor.modules["sub_nns"].items():
        sub_nn_elems, _ = transform_layers(sub_nn_elems,
                                           extractor.inputs_outputs,
                                           extractor.layer_of_output,
                                           extractor.modules["order"],
                                           is_layer=False)
        extractor.modules["sub_nns"][sub_nn_name] = sub_nn_elems

    is_layer = True
    if (extractor.output_nn_type == "sequential" or
        extractor.input_nn_type == "sequential"):
        # if the main model is seq, only that model can remain in modules
        # so that its layers are not defined or called separately.
        is_layer = False

    layers, order = transform_layers(extractor.modules["layers"],
                                     extractor.inputs_outputs,
                                     extractor.layer_of_output,
                                     extractor.modules["order"],
                                     is_layer=is_layer)

    extractor.modules["layers"] = layers
    if order:
        extractor.modules["order"] = order
    return extractor


def handle_positional_params(layer_type, layer_elems, pos_params):
    """
    It handles the positional parameters to convert them
    to keyword parameters.
    """
    lyr_pos = next((e for e in pos_params if layer_type.startswith(e)), None)
    if lyr_pos and layer_elems[1]["positional_params"]:
        counter = 0
        params = layer_elems[1]["positional_params"]
        new_dict = {}
        for pos_arg in params:
            new_dict[pos_params[lyr_pos][counter]] = pos_arg
            counter+=1
        layer_elems[1] = {**new_dict, **layer_elems[1]}
    layer_elems[1].pop("positional_params")
    return layer_elems


def handle_remaining_params(params, layer_type, layer_name, inputs_outputs,
                            layer_of_output, layers_fixed_params,
                            is_layer=True):
    """
    It handles the parameters that are processed in the same way
    for TF and Torch.
    """
    if layer_type in layers_fixed_params:
        params.update(layers_fixed_params[layer_type])
    if is_layer:
        if (not isinstance(inputs_outputs[layer_name][1], list) and
            inputs_outputs[layer_name][0] != inputs_outputs[layer_name][1]):
            lyr_in_out = layer_of_output[inputs_outputs[layer_name][0]]
            params["input_reused"] = True
            params["name_module_input"] = lyr_in_out

    return params


def format_params(layer_params):
    """It formats the layers' parameters for printing."""
    formatted_params = []
    for key, value in layer_params.items():
        if str(value).startswith("self"):
            formatted_params.append(f"{key}='{value}'")
        elif isinstance(value, str):
            if key == "path_data":
                formatted_params.append(f"{key}=r'{value}'")
            else:
                formatted_params.append(f"{key}='{value}'")
        else:
            formatted_params.append(f"{key}={value}")
    layer_params_str = ', '.join(formatted_params)
    return layer_params_str


def get_imports(extractor):
    """It collects the list of modules to import in BUML."""
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


def print_nn_def(extractor):
    """It generates code for the definition of the NN modules"""
    nn_name = extractor.nn_name
    layers = extractor.modules["layers"]
    tensorops = extractor.modules["tensorops"]
    print(f"{nn_name}: NN = NN(name='{nn_name}')")
    for module_name, module_type in extractor.modules["order"].items():
        if module_type == "layer":
            layer_type = layers[module_name][0]
            layer_params = layers[module_name][1]
            if layer_params:
                params_str = format_params(layer_params)
                print(f"{nn_name}.add_layer({layer_type}("
                      f"name='{module_name}', {params_str}))")
            else:
                print(f"{nn_name}.add_layer({layer_type}("
                      f"name='{module_name}'))")

        elif module_type == "sub_nn":
            print(f"{nn_name}.add_sub_nn({module_name})")
        else:
            params_str = format_params(tensorops[module_name])
            print(f"{nn_name}.add_tensor_op(TensorOp("
                f"name='{module_name}', {params_str}))")

def print_subnn_def(sub_nns):
    """It generates code for the definition of sub NNs"""
    for sub_nn_name, sub_nn_elems in sub_nns.items():
        print(f"{sub_nn_name}: NN = NN(name='{sub_nn_name}')")
        for layer_id, layer_elems in sub_nn_elems.items():
            layer_type = layer_elems[0]
            layer_params = layer_elems[1]
            params_str = format_params(layer_params)
            print(f"{sub_nn_name}.add_layer({layer_type}("
                  f"name='{sub_nn_name}_{layer_id}', {params_str}))")
        print("\n")


def print_imports(extractor, framework):
    """It generates code for the BUML imports"""
    cls_to_import =  get_imports(extractor)
    frwo_lwr = framework.lower()
    print(f"from besser.BUML.metamodel.nn import "
        f"{', '.join(map(str, cls_to_import[:2]))}, \\")
    for i in range(2, len(cls_to_import), 5):
        if len(cls_to_import[i:]) > 5:
            print(f"    {', '.join(map(str, cls_to_import[i:i+5]))}, \\")

        else:
            print(f"    {', '.join(map(str, cls_to_import[i:i+5]))}")
    print(f"from besser.generators.nn.{frwo_lwr}.{frwo_lwr}_code_generator " \
          f"import {framework}Generator\n\n\n")


def print_config(configuration, nn_name, lookup_loss_functions, config_list):
    """It generates code for the parameters of the NN"""
    loss_func = lookup_loss_functions[configuration["loss_function"]]
    configuration["loss_function"] = loss_func
    config_to_print = {k: configuration[k] for k in config_list}
    config_str = format_params(config_to_print)
    print(f"\nconfiguration: Configuration = Configuration({config_str})")
    print(f"{nn_name}.add_configuration(configuration)\n")


def print_data(train_data, test_data, nn_name,
               train_param_lst, test_param_lst):
    """It generates code for training and test datasets"""
    train_to_print = {k: train_data[k] for k in train_param_lst}
    train_str = format_params(train_to_print)
    test_to_print = {k: test_data[k] for k in test_param_lst}
    test_str = format_params(test_to_print)

    if train_data["input_format"] == "images":
        if "normalize_images" not in train_data:
            train_data["normalize_images"] = False
        print(f"image = Image(shape={train_data['images_size']}, "
              f"normalize={train_data['normalize_images']})")

        print(f"train_data = Dataset({train_str}, image=image)")
    else:
        print(f"train_data = Dataset({train_str})")

    print(f"test_data = Dataset({test_str})")
    print(f"\n{nn_name}.add_train_data(train_data)")
    print(f"{nn_name}.add_test_data(test_data)\n")
