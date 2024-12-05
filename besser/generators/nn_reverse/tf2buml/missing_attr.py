"""
Extract number of input features from tensorflow dynamically. 
This is specific to conv and dense layers.
These layer attributes are needed when transforming tensorflow
code to pytorch code.
The steps are: 
- Extract the code needed from tf script and add a function that 
retrieves the desired attributes from tf code.
- Save that code to `tf_code_extracted.py` file.
- Execute the `tf_code_extracted.py` file to retrieve the attributes.
- Add the attributes to the dictionaries in `extractor`.

Args:
    file_path (str): Path to the input file.
    word (str): The word to search for.
    output_path (str): Path to the output file.
"""
import subprocess
import re
from ast import literal_eval
import tensorflow as tf
from keras import layers


def extract_nn_code(file_path):
    """
    Extracts the code on the neural network from the given file.
    It returns:
        `init_code` that contains the code defined in the init method.
        `call_code` containing the statements in the call method.
        `loader_code`: containing the code for the data loader
    """
    nn_name, loader_name = None, None
    in_call_method = False
    call_code = []
    init_code = []
    in_init = True
    loader_code = []

    init_code.append("from tf2buml.missing_attr import get_data\n")
    init_code.append("from tf2buml.missing_attr import get_input_shape_all\n")
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            if "class " in line:
                nn_name = line.split(" ")[1].split("(")[0]

            if in_init:
                init_code.append(line)
                if "super().__init__()" in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    init_code.append(f"{indent}self.dict_details = {{}}\n")

            if "def call" in line:
                in_call_method = True
                in_init = False

            if "return " in line:
                in_call_method = False

            if in_call_method and line.strip():
                call_code.append(line)

            if not (in_init or in_call_method):
                loader_code.append(line)
                # Check for the first occurrence of `loader` and `train`
                if "loader" in line and "train" in line:
                    loader_name = line.split(" ")[0]
                    break

    return nn_name, loader_name, init_code, call_code, loader_code[1:]


def get_data(loader, input_shape):
    """
    Get the data that will be passed to the NN to infer the layers 
    input shape.
    If the data loader is not defined in the code to be transformed,
    an input_shape can also be provided to generate random data.
    """
    if loader is not None:
        for batch in loader:
            input_data, _ = batch
            break
    elif input_shape is not None:
        input_data = tf.random.uniform(input_shape)
    else:
        input_data = None
    return input_data

def get_modules_names(model):
    """Maps modules objects to their names"""
    modules_names = {}
    for name, value in model.__dict__.items():
        if isinstance(value, (tf.keras.layers.Layer, tf.keras.Model)):
            modules_names[value] = name
    return modules_names



def get_shape(lyr_details, layer, index, x, attribute_name=None):
    """Get the layer type and its input shape"""
    lyrs = (layers.Conv1D, layers.Conv2D, layers.Conv3D, layers.Dense,
            layers.LSTM, layers.GRU, layers.RNN)

    if isinstance(layer, lyrs) or isinstance(layer, layers.Bidirectional):
        if isinstance(layer, layers.Bidirectional):
            layer = layer.layer
        cls_name = layer.__class__.__name__
        # Record class type and shape before passing through layer
        lyr_details[index] = [cls_name, x.shape[-1], attribute_name]

    x = layer(x)
    return lyr_details, x

def get_input_shape_all(x, model):
    """
    Get input shape by iteration through layers. This method is less
    accurate as it supposes that the input of a layer comes from its
    preceeding layer, which is not always correct.
    It is used because it enables to cover all the layers including
    the ones defined in a sub neural network. Its output is further
    corrected but the second method that gets the shapes from the
    call method of the NN.
    """
    input_shape_all = {}

    modules_names = get_modules_names(model)
    for index, module in enumerate(model.layers):
        if isinstance(module, tf.keras.models.Sequential):
            subnn_name = modules_names.get(module, "Unknown")
            subnn_details = {}
            for sub_index, sub_layer in enumerate(module.layers):
                subnn_details, x = get_shape(subnn_details, sub_layer,
                                             sub_index, x)
            input_shape_all[index] = [module.__class__.__name__,
                                      subnn_details, subnn_name]
        else:
            lyr_name = modules_names.get(module, "Unknown")
            input_shape_all, x = get_shape(input_shape_all, module,
                                           index, x, lyr_name)

    return input_shape_all

def modify_nn_call(call_code, init_code):
    """
    Add prints to the call method of the NN to get the input shapes.
    This is more accurate than iterating through layers. 
    Shapes retrieved using this method are used to update the values 
    in the `input_shape_all` dict.
    """
    call_code = call_code[1:]
    indent = call_code[0][:len(call_code[0]) - len(call_code[0].lstrip())]
    for line in call_code:
        if "self." in line:
            _input = line.split("(")[1].split(")")[0].strip()
            _output =  line.split("=")[0].strip()
            name_module = line.split(".")[1].split("(")[0].strip()
            shape_line = (f'{indent}self.dict_details["{name_module}"] = '
                          f'["{_input}", "{_output}", {_input}.shape]\n')
            init_code.append(shape_line)
        init_code.append(line)

    init_code.append(f'{indent}return self.dict_details')
    model_code = "".join(init_code)
    return model_code


def extract_code(file_path, shape):
    """
    Extract code from the beginning of the file up to and including
    the line containing the first occurrence of `loader`, which is used
    to infer the shape of the model input data.
    The extracted code is saved it to `tf_code_extracted.py` file.

    Arg:
        file_path (str): Path to the input file.
    """

    nn_name, loader_name, init_code, call_code, loader_code = (
        extract_nn_code(file_path)
    )

    model_code = modify_nn_call(call_code, init_code)
    loader_code = "".join(loader_code)

    function_code = f"""\
tf_model = {nn_name}()
input_data = get_data({loader_name}, {shape})
x = input_data

input_shape_all = get_input_shape_all(x, tf_model)

tf_model = {nn_name}()
input_shape_from_call = tf_model(input_data)

for mdl_name, mdl_details in input_shape_all.items():
    if input_shape_from_call.get(mdl_details[-1], None):
        if not isinstance(mdl_details[-2], dict):
            mdl_details[1] = input_shape_from_call[mdl_details[-1]][-1][-1]

print(input_shape_all)
"""

    code = f"{''.join(model_code)}\n{loader_code}\n{function_code}"

    return code


def extract_dict_from_output(output: str):
    """Extract the Python dictionary from the output string."""
    # Regular expression to match the first dictionary-like structure
    # Match everything starting with `{` and ending with `}`, considering
    # nested braces
    match = re.search(r'\{.*\}', output.strip())

    if match:
        dict_str = match.group(0)  # Get the dictionary part as a string
        try:
            # Safely evaluate the string to a Python dictionary using
            # ast.literal_eval
            return literal_eval(dict_str)
        except (ValueError, SyntaxError) as e:
            print("Error parsing dictionary:", e)
            return None
    return None



def increment_counter(cnt_lyr_missing, in_feat_dict):
    """
    Increment the counter that represents layers id in the `in_feat_dict`
    """
    if cnt_lyr_missing != next(reversed(in_feat_dict)):
        cnt_lyr_missing+=1
    while (cnt_lyr_missing not in list(in_feat_dict.keys()) and
            cnt_lyr_missing < next(reversed(in_feat_dict))):
        cnt_lyr_missing+=1
    return cnt_lyr_missing


def execute_code(code):
    """
    Executes the code and returns the dict containing
    the missing attributes.
    """

    file_path = "temp.py"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(code)

    result = subprocess.run(["python", file_path], capture_output=True,
                            text=True, check=True)

    modules_details = extract_dict_from_output(result.stdout)

    return modules_details


def update_layers_attr(shape_attr, cnt_lyr_missing,
                       cnt_lyr, lyr_details):
    """
    Return the missing attributes of a layer along with 
    its original attributes.
    """
    new_details = lyr_details[1]
    if lyr_details[0] == shape_attr[cnt_lyr_missing][0]:
        in_feat = shape_attr[cnt_lyr_missing][1]
        if lyr_details[0] == "Dense":
            new_item = {'in_features': in_feat}
        elif lyr_details[0] in ["LSTM", "RNN", "GRU"]:
            new_item = {'input_size': in_feat}
        else:
            new_item = {'in_channels': in_feat}
        new_details = {**new_item, **lyr_details[1]}
        cnt_lyr_missing = increment_counter(cnt_lyr_missing,
                                            shape_attr)

    return new_details, cnt_lyr_missing, cnt_lyr


def get_attributes(extractor, filename, shape=None):
    """
    Retrieve the in_features and in_channels missig attributes and 
    store them to `extractor`.
    """

    code = extract_code(filename, shape)
    shape_attr = execute_code(code)
    if shape_attr is None:
        return extractor

    _layers = extractor.modules["layers"]
    subnns = extractor.modules["sub_nns"]
    cnt_lyr_missing = next(iter(shape_attr))
    cnt_mdl = 0

    for mdl_name, mdl_type in extractor.modules["order"].items():
        if (mdl_type == "sub_nn" and
            shape_attr[cnt_lyr_missing][-1] == mdl_name):
            subnn_attr_dict = shape_attr[cnt_lyr_missing][1]
            cnt_ms_sub = next(iter(subnn_attr_dict))
            cnt_lyr_sub = 0
            for lyr_name, lyr_details in subnns[mdl_name].items():
                new_details, cnt_ms_sub, cnt_lyr_sub = update_layers_attr(
                    subnn_attr_dict, cnt_ms_sub, cnt_lyr_sub, lyr_details)
                subnns[mdl_name][lyr_name][1] = new_details
                cnt_lyr_sub += 1
            cnt_lyr_missing = increment_counter(cnt_lyr_missing, shape_attr)

        elif (mdl_type == "layer" and
              shape_attr[cnt_lyr_missing][-1] == mdl_name):
            lyr_details = _layers[mdl_name]
            new_details, cnt_lyr_missing, cnt_mdl = update_layers_attr(
                shape_attr, cnt_lyr_missing, cnt_mdl, lyr_details)
            _layers[mdl_name][1] = new_details
        cnt_mdl += 1

    return extractor
