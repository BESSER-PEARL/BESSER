"""
Helper functions to transform PyTotrch and TensorFlow nn code to BUML model.
"""

import argparse
import ast
from typing import TYPE_CHECKING


from besser.generators.nn_migration.tf2torch.input_shape_retriever import (
    update_model
)
from besser.BUML.metamodel.nn import Configuration, Dataset, Image
from besser.BUML.metamodel.nn import Layer, NN

if TYPE_CHECKING:
    from besser.generators.nn_migration.ast_parser_nn import ASTParser

def parse_arguments_transform():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Read a NN file and convert it to buml code."
    )
    parser.add_argument(
        "filename", type=str, help="Path to the file to read"
    )
    parser.add_argument(
        "typeinput", type=str, default="subclassing",
        help="Type of the input NN architecture."
    )
    parser.add_argument(
        "typeoutput", type=str, default="subclassing",
        help="Type of the output NN architecture."
    )
    parser.add_argument(
        "--onlynn", type=str2bool, help="Whether the file contains only \
            NN def or also the code for training and evaluation.",
            default=True, const=True, nargs='?'
    )
    parser.add_argument(
        "--datashape", type=parse_tuple, default=None, 
        help=(
            "The shape of the input data (optional)."
            "It is needed when transforming tf code to pytorch code"
            "It is used to recover some layer attributes dynamically"
            "If the migrated script defines a dataset, it can be skipped"
        ),
    )

    return parser.parse_args()

def str2bool(v):
    """
    Parse bool from a string input
    
    Parameters:
    ----------
    value(str): The bool in a string format.

    Returns:
    -------
    The bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_tuple(value: str):
    """
    Parse a tuple from a string input
    
    Parameters:
    ----------
    value(str): The tuple in a string format.

    Returns:
    -------
    The tuple
    """
    return (1,) + tuple(map(int, value.strip("()").split(",")))

def transform(args: argparse.Namespace, framework: str,
              ast_parser_class: 'ASTParser'):
    """
    Tt gets the AST and transforms it to a buml model.

    Parameters:
        args(argparse.Namespace): An object with arg attributes.
            framework (str): "TF" or "PyTorch".
        ast_parser_class ('ASTParser'): The class to use to 
            parse the AST

    Returns:
        The buml model and its output architecture type (i.e., sequential
            or subclassing). It is specified in the config file.
    """

    # Read the nn code to migrate
    with open(args.filename, "r", encoding="utf-8") as file:
        code = file.read()

    # shape of the data input by the user
    shape = args.datashape
    input_nn_type = args.typeinput
    output_nn_type = args.typeoutput
    only_nn = args.onlynn

    # Parse the code and get the buml model
    tree = ast.parse(code)
    extractor = ast_parser_class(input_nn_type, only_nn)
    extractor.visit(tree)
    buml_model = extractor.buml_model

    # If the input_nn_type is sequential, it is processed as a sub_nn.
    # It needs to be removed from sub_nns and the order of modules needs
    # to be corrected.
    if input_nn_type == "sequential":
        nn_obj = next((obj for obj in buml_model.sub_nns if
                        obj.name == buml_model.name), None)
        buml_model.modules.clear()
        buml_model.layers.clear()
        for module in nn_obj.modules:
            if isinstance(module, Layer):
                buml_model.add_layer(module)
            #automatically appends to modules
            elif isinstance(module, NN):
                buml_model.add_sub_nn(module)
            else:
                buml_model.modules.append(module)
        buml_model.sub_nns.remove(nn_obj)

    if framework == "TF":
        update_model(input_nn_type, buml_model, args.filename, shape)

    config = extractor.data_config["config"]
    if config:
        if "classification" in config:
            del config["classification"]
        cnf = Configuration(**config)
        buml_model.add_configuration(cnf)

    dt_tr = extractor.data_config["train_data"]
    dt_ts = extractor.data_config["test_data"]
    if dt_tr:
        train_data = Dataset(name="train_data", path_data=dt_tr["path_data"],
                             task_type=dt_tr["task_type"],
                             input_format=dt_tr["input_format"])
        test_data = Dataset(name="test_data", path_data=dt_ts["path_data"])

        if dt_tr["input_format"] == "images":
            if "normalize_images" not in dt_tr:
                dt_tr["normalize_images"] = False
            img = Image(shape=dt_tr["images_size"],
                        normalize=dt_tr["normalize_images"])
            train_data.add_image(img)

        buml_model.add_train_data(train_data)
        buml_model.add_test_data(test_data)

    return buml_model, output_nn_type


def param_to_list(lyr_type: str, lyr_params: dict, params_to_convert: list,
                  layers_of_params: list):
    """
    It converts int parameters to list format for buml model
    
    Parameters:
        lyr_type (str): The type of the layer.
        lyr_params (dict): A dictionnary of all the layer parameters and their
            values.
        params_to_convert (list): The list of parameters to be converted.
        layers_of_params (list): The list of layers that need their params
            to be converted.

    Returns:
        None
    """

    if lyr_type in layers_of_params:
        for param in lyr_params:
            if (param in params_to_convert and
                isinstance(lyr_params[param], int)):
                lyr_params[param] = [lyr_params[param]]


def process_positional_params(lyr_type: str, lyr_params: dict,
                              pos_params: dict):
    """
    It processes the positional parameters to convert them
    to keyword parameters.

    Parameters:
        lyr_type (str): The type of the layer.
        lyr_params (dict): A dictionnary of all the layer parameters
            and their values.
        pos_params (dict): A dictionary storing the as keys layers 
            types and as values the names of their positional params.

    Returns:
        None
    """
    lyr_of_pos_parm = next(
        (e for e in pos_params if lyr_type.startswith(e)), None
    )
    if lyr_of_pos_parm and lyr_params["positional_params"]:
        counter = 0
        pos_params_with_values = {}
        for pos_arg in lyr_params["positional_params"]:
            par = pos_params[lyr_of_pos_parm][counter]
            pos_params_with_values[par] = pos_arg
            counter+=1
        lyr_params = {**pos_params_with_values, **lyr_params}
    lyr_params.pop("positional_params")


def set_static_params(lyr_type: str, lyr_params: dict,
                      static_params: dict):
    """
    It handles the parameters that are retreived from the layer type.
    Ex: For a MaxPool1D layer, 'pooling_type' ('max) and 'dimension ('1D')
    are retrieved.

    Parameters:
        lyr_type (str): The type of the layer.
        lyr_params (dict): A dictionnary of all the layer parameters
            and their values.
        fixed_params (dict): A dictionary storing as keys the layers 
            types and as values their fixed params.

    Returns:
        Nones
    """
    if lyr_type in static_params:
        lyr_params.update(static_params[lyr_type])


def set_remaining_params(lyr_obj: Layer, inputs_outputs: dict,
                         layer_of_output: dict):
    """
    It sets the 'input_reused' and 'name_module_input' layer parameters.

    Parameters:
        lyr_obj (Layer): The buml layer object.
        inputs_outputs (dict): It stores input and output variables 
            of layers.
        layer_of_output (dict): It stores name of layers given their
            output var.
    
    Returns:
        None
    """

    layer_name = lyr_obj.name
    if (not isinstance(inputs_outputs[layer_name][1], list) and
        inputs_outputs[layer_name][0] != inputs_outputs[layer_name][1]):
        lyr_in_out = layer_of_output[inputs_outputs[layer_name][0]]
        lyr_obj.input_reused = True
        lyr_obj.name_module_input = lyr_in_out
