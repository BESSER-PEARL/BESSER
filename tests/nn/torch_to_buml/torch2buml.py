import ast
from ast_parser import ASTParser
import sys
import argparse
from tests.nn.torch_to_buml.definitions import config_list, train_param_list, \
    test_param_list, lookup_loss_functions
from transform_functions import transform_layers, adjust_layers_ids, \
    format_params, get_imports



f = open("torch_transformed.py", "w")
sys.stdout = f

# Set up argument parser
parser = argparse.ArgumentParser(description="Read a Pytorch file and convert it to buml code.")
parser.add_argument("filename", type=str, help="Path to the Pytorch file to read")
args = parser.parse_args()

# Read pytorch code from an external file
with open(args.filename, "r") as file:
    code = file.read()


# Parse the code and use the extractor
tree = ast.parse(code)
extractor = ASTParser()
extractor.visit(tree)
#print(ast.dump(tree, indent=4))

layers = extractor.layers
sub_nns = extractor.sub_nns
tensorops = extractor.tensorops
modules = extractor.modules
inputs_outputs = extractor.inputs_outputs
layer_of_output = extractor.layer_of_output
train_data = extractor.train_data
test_data = extractor.test_data
configuration = extractor.configuration

layers = transform_layers(layers, inputs_outputs, layer_of_output)
for sub_nn_name, sub_nn_elems in sub_nns.items():
    sub_nn_elems = transform_layers(sub_nn_elems, inputs_outputs, layer_of_output, is_layer=False)
    sub_nn_elems = adjust_layers_ids(sub_nn_elems)
    sub_nns[sub_nn_name] = sub_nn_elems

#imports        
cls_to_import =  get_imports(layers, tensorops, sub_nns, configuration, train_data, test_data)
print(f"from besser.BUML.metamodel.nn import {', '.join(map(str, cls_to_import[:2]))}, \\")
for i in range(2, len(cls_to_import), 5):
    if len(cls_to_import[i:i+5]) > 5:
        print(f"    {', '.join(map(str, cls_to_import[i:i+5]))}, \\")
    else:
        print(f"    {', '.join(map(str, cls_to_import[i:i+5]))}")
print("from besser.generators.pytorch.pytorch_code_generator import PytorchGenerator")
print("\n\n\n")

for sub_nn_name, sub_nn_elems in sub_nns.items():
    print(f"{sub_nn_name}: NN = NN(name='{sub_nn_name}')")
    for layer_id, layer_elems in sub_nn_elems.items():
        layer_type = layer_elems[0]
        layer_params = layer_elems[1]
        layer_params_str = format_params(layer_params)
        print(f"{sub_nn_name}.add_layer({layer_type}(name='{sub_nn_name}_{layer_id}', {layer_params_str}))")
    print("\n")
nn_name = extractor.nn_name

print(f"{nn_name}: NN = NN(name='{nn_name}')")
for module_name, module_type in modules.items():
    if module_type == "layer":
        layer_type = layers[module_name][0]
        layer_params = layers[module_name][1]
        layer_params_str = format_params(layer_params)
        print(f"{nn_name}.add_layer({layer_type}(name='{module_name}', {layer_params_str}))")
    elif module_type == "sub_nn":
        print(f"{nn_name}.add_sub_nn({module_name})")
    else:
        tensorop_params_str = format_params(tensorops[module_name])
        print(f"{nn_name}.add_tensor_op(TensorOp(name='{module_name}', {tensorop_params_str}))")

if configuration and train_data and test_data:
    configuration["loss_function"] = lookup_loss_functions[configuration["loss_function"]]
    config_to_print = {k: configuration[k] for k in config_list}
    config_to_print = format_params(config_to_print)
    print(f"\nconfiguration: Configuration = Configuration({config_to_print})")
    print(f"{nn_name}.add_configuration(configuration)\n")

    train_to_print = {k: train_data[k] for k in train_param_list}
    train_to_print = format_params(train_to_print)
    test_to_print = {k: test_data[k] for k in test_param_list}
    test_to_print = format_params(test_to_print)

    if train_data["input_format"] == "images":
        if "normalize_images" not in train_data:
            train_data["normalize_images"] = False
        print(f"image = Image(shape={train_data['images_size']}, normalize={train_data['normalize_images']})")

        print(f"train_data = Dataset({train_to_print}, image=image)")
    else:
        print(f"train_data = Dataset({train_to_print})")
        
    print(f"test_data = Dataset({test_to_print})") 
    print(f"\n{nn_name}.add_train_data(train_data)")
    print(f"{nn_name}.add_test_data(test_data)\n")

print(f"pytorch_model = PytorchGenerator(model={nn_name}, output_dir='output/{nn_name}')")
print("pytorch_model.generate()") 

sys.stdout = sys.__stdout__
f.close()