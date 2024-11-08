"""Script to convert PyTorch code to BUML code."""

import ast
import sys
import argparse
from ast_parser import ASTParser
from transform_functions import transform_layers, adjust_layers_ids, \
    format_params, get_imports
from tests.nn.torch_to_buml.definitions import config_list, train_param_list, \
    test_param_list, lookup_loss_functions

def parse_arguments():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Read a PyTorch file and convert it to buml code."
    )
    parser.add_argument(
        "filename", type=str, help="Path to the PyTorch file to read"
    )
    return parser.parse_args()

def main():
    """Tt transforms PyTorch code to BUML code"""
    args = parse_arguments()

    # Read tf code from an external file
    with open(args.filename, "r", encoding="utf-8") as file:
        code = file.read()

    f = open("torch_transformed.py", "w", encoding="utf-8")
    sys.stdout = f

    # Parse the code and use the extractor
    tree = ast.parse(code)
    extractor = ASTParser()
    extractor.visit(tree)
    #print(ast.dump(tree, indent=4))
    #print("layers", layers)
    #print("subnn", sub_nns)

    nn_name = extractor.nn_name
    extractor = transform_modules(extractor)
    print_imports(extractor)
    print_subnn_def(extractor.modules["sub_nns"])
    print_nn_def(extractor)

    if (extractor.data_config["config"] and
        extractor.data_config["train_data"] and
        extractor.data_config["test_data"]):
        print_config(extractor.data_config["config"], nn_name)
        print_data(extractor.data_config["train_data"],
                   extractor.data_config["test_data"], nn_name)

    print(f"pytorch_model = PytorchGenerator(model={nn_name}, "
          f"output_dir='output/{nn_name}')")
    print("pytorch_model.generate()")

    sys.stdout = sys.__stdout__
    f.close()

def transform_modules(extractor):
    """
    It transforms names of parameters of modules from TF to BUML.
    """
    for sub_nn_name, sub_nn_elems in extractor.modules["sub_nns"].items():
        sub_nn_elems = transform_layers(sub_nn_elems,
                                        extractor.inputs_outputs,
                                        extractor.layer_of_output,
                                        is_layer=False)
        sub_nn_elems = adjust_layers_ids(sub_nn_elems)
        extractor.modules["sub_nns"][sub_nn_name] = sub_nn_elems
    layers = transform_layers(extractor.modules["layers"],
                              extractor.inputs_outputs,
                              extractor.layer_of_output)
    extractor.modules["layers"] = layers
    return extractor

def print_imports(extractor):
    """It generates code for the TF imports"""
    cls_to_import =  get_imports(extractor)
    print(f"from besser.BUML.metamodel.nn import "
        f"{', '.join(map(str, cls_to_import[:2]))}, \\")
    for i in range(2, len(cls_to_import), 5):
        if len(cls_to_import[i:]) > 5:
            print(f"    {', '.join(map(str, cls_to_import[i:i+5]))}, \\")

        else:
            print(f"    {', '.join(map(str, cls_to_import[i:i+5]))}")
    print("from besser.generators.pytorch.pytorch_code_generator " \
      "import PytorchGenerator\n\n\n")

def print_subnn_def(sub_nns):
    """It generates code for the definition of the sub NNs"""
    for sub_nn_name, sub_nn_elems in sub_nns.items():
        print(f"{sub_nn_name}: NN = NN(name='{sub_nn_name}')")
        for layer_id, layer_elems in sub_nn_elems.items():
            layer_type = layer_elems[0]
            layer_params = layer_elems[1]
            params_str = format_params(layer_params)
            print(f"{sub_nn_name}.add_layer({layer_type}("
                f"name='{sub_nn_name}_{layer_id}', {params_str}))")
        print("\n")

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

def print_config(configuration, nn_name):
    """It generates code for the parameters of the NN"""
    loss_func = lookup_loss_functions[configuration["loss_function"]]
    configuration["loss_function"] = loss_func
    config_to_print = {k: configuration[k] for k in config_list}
    config_str = format_params(config_to_print)
    print(f"\nconfiguration: Configuration = Configuration({config_str})")
    print(f"{nn_name}.add_configuration(configuration)\n")

def print_data(train_data, test_data, nn_name):
    """It generates code for training and test datasets"""
    train_to_print = {k: train_data[k] for k in train_param_list}
    train_str = format_params(train_to_print)
    test_to_print = {k: test_data[k] for k in test_param_list}
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

if __name__ == "__main__":
    main()
