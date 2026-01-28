"""
Extracts input_shape from tensorflow conv and dense layers dynamically. 
This is specific to conv and dense layers.
These layer attributes are needed when transforming tensorflow
code to pytorch code.
"""
import sys
import tempfile
import importlib.util
from pathlib import Path
import tensorflow as tf
from keras import layers
from besser.BUML.metamodel.nn import NN, Layer
from besser.generators.nn_migration.tf2torch.definitions import layers_buml2tf


def extract_nn_code(file_path: str, nn_type: str):
    """
    Extracts the code of the neural network from the input file.

    Parameters:
        file_path (str): The path to the tf nn file.
        nn_type (str): The type of the nn architecture (i.e., 'sequential'
            or 'subclassing')
    
    Returns:
        nn_name (str): The name of the nn model.
        loader_name (str): The name of the data loader in the nn file
            if defined.
        init_code(list): The lines of the code in the init method of
            of the neural network. The list also contains a line of code
            defining a dict that can stores the input_shpae of conv and
            dense layers.
        call_code (list): The lines of code in the call method.
        loader_code (list): The lines of code defining the data loader.
    """
    nn_name = None
    in_call_method = False
    call_code = []
    init_code = []
    in_init = True

    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            if "class " in line and not nn_name:
                nn_name = line.split(" ")[1].split("(")[0]

            if nn_type == "sequential":
                if "Sequential" in line:
                    nn_name = line.split(" = ")[0]
                    return nn_name, None, None

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

    return nn_name, init_code, call_code


def get_modules_names(model: tf.keras.Model):
    """
    Maps layer or model objects to their attribute names.
    

    Parameters:
        model (tf.keras.Model): The nn model.
    
    Returns:
        dict: A dictionary where keys are model/layer objects, 
            and values are their attribute names.
    """
    modules_names = {}
    for name, value in model.__dict__.items():
        if isinstance(value, (tf.keras.layers.Layer, tf.keras.Model)):
            modules_names[value] = name
    return modules_names



def get_shape(lyr_details: dict, layer: tf.keras.layers.Layer, index: int,
              x: tf.Tensor | None, lyr_name: str | None = None):
    """
    Given a layer, it updates the lyr_details dict with its class name,
    shape of its input data and its name when applicable.

    Parameters:
        lyr_details (dict): The dict that holds the class name,
            shape of its input data and its name when applicable.
        layer (tf.keras.layers.Layer): The layer obejct.
        index (int): The index of the layer in the model.
        x (tf.Tensor | None): The input data to the layer.
        lyr_name (str | None): The name of the layer.

    Returns:
        The updated input data (the output of the layer) and lyr_details
            dict gets updated with the layer info.
    """

    lyrs = (layers.Conv1D, layers.Conv2D, layers.Conv3D, layers.Dense,
            layers.LSTM, layers.GRU, layers.RNN)

    if isinstance(layer, lyrs) or isinstance(layer, layers.Bidirectional):
        if isinstance(layer, layers.Bidirectional):
            layer = layer.layer
        cls_name = layer.__class__.__name__
        # Record class type and shape before passing through layer
        lyr_details[index] = [cls_name, x.shape[-1], lyr_name]

    x = layer(x)
    return x

def get_input_shape_all(x: tf.keras.layers.Layer | None,
                        model: tf.keras.Model):
    """
    Gets input shape of layers by iteration through the model layers. 
    This method is less accurate as it supposes that the input of a 
    layer comes from its preceeding layer, which is not always correct.
    It is used because it enables to cover all the layers including
    the ones defined in a sub neural network. Its output is further
    corrected by the input_shape_from_call function that gets 
    the shapes from the call method of the NN dynamically.
    For the case of a sequential architecture, the output of this function
    alone is accurate as the layers have a sequnetial order.

    Parameters:
        x (tf.keras.layers.Layer | None): The input data.
        model (tf.keras.Model): The nn model.

    Returns:
        A dictionary containing index of layers/subnns as keys and their
        class name, input shape and name as values.
    """
    input_shape_all = {}

    modules_names = get_modules_names(model)
    for index, module in enumerate(model.layers):
        if isinstance(module, tf.keras.models.Sequential):
            subnn_name = modules_names.get(module, "Unknown")
            subnn_details = {}
            for sub_index, sub_layer in enumerate(module.layers):
                x = get_shape(subnn_details, sub_layer, sub_index, x)
            input_shape_all[index] = [module.__class__.__name__,
                                      subnn_details, subnn_name]
        else:
            lyr_name = modules_names.get(module, "Unknown")
            x = get_shape(input_shape_all, module, index, x, lyr_name)
    return input_shape_all




def load_modified_nn(modified_code: str, class_name: str):
    # Write modified code to temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(modified_code)
        tmp_path = Path(tmp.name)

    # Load as a module
    spec = importlib.util.spec_from_file_location("temp_module", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["temp_module"] = mod
    spec.loader.exec_module(mod)

    # Get the class
    return getattr(mod, class_name)

def modify_nn_call(init_code: list, call_code: list):
    """
    Adds prints to the call method of the NN to get the input shapes.
    This is more accurate than iterating through layers. 
    Shapes retrieved using this method are used to update the values 
    in the dictionary returned by get_input_shape_all method.

    Parameters:
        init_code(list): The lines of the code in the init method of
            of the neural network. The list also contains a line of code
            defining a dict that can stores the input_shpae of conv and
            dense layers.
        call_code (list): The lines of code in the call method.
        
    Returns:
        The full code in str format that has the init method and the 
            modified call method so that it records the input shape
            of layers in a dict dynamically.

    """
    call_code = call_code[1:]  # skip 'def call...' line
    indent = call_code[0][:len(call_code[0]) - len(call_code[0].lstrip())]

    for line in call_code:
        if "self." in line:
            # extract input/output names as strings
            _input_name = line.split("(")[1].split(")")[0].strip()
            layer_name = line.split(".")[1].split("(")[0].strip()

            # append a line that stores names as strings + symbolic shape tuple
            shape_line = (
                f'{indent}self.dict_details["{layer_name}"] = '
                f'tf.shape({_input_name})\n'
            )
            init_code.append(shape_line)

        init_code.append(line)

    init_code.append(f'{indent}return self.dict_details')
    return "".join(init_code)

def extract_in_size_from_dict(layer_dict: dict):
    """
    Extracts desired input size from KerasTensor shapes in ListWrapper.

    Parameters:
        lyr_input_shapes (dict): Dictionary where values are ListWrapper containing KerasTensors.

    Returns:
        dict: Dictionary with same keys but values are desired input size.
    """
    result = {}
    for layer_name, ktensor in layer_dict.items():
        # Attempt to use inferred_value if present
        inferred = getattr(ktensor, '_inferred_value', None)
        if inferred is not None:
            shape_list = inferred.tolist() if hasattr(inferred, 'tolist') else list(inferred)
            result[layer_name] = shape_list[-1]  # last element
        else:
            # fallback to static shape
            shape_list = ktensor.shape.as_list() if hasattr(ktensor, 'shape') else None
            result[layer_name] = shape_list[-1] if shape_list else None
    return result



def increment_counter(counter: int, dict_shapes: dict):
    """
    Increments the counter that represents layers id in
    the `dict_shapes` dictionary.

    Parameters:
        counter (int): The counter to increment.
        dict_shapes (dict): The dictionary containing the input 
        shapes of layers

    Returns:
        The incremented counter.

    """
    if counter != next(reversed(dict_shapes)):
        counter+=1
    while (counter not in list(dict_shapes.keys()) and
            counter < next(reversed(dict_shapes))):
        counter+=1
    return counter



def update_layers_attr(lyr_input_shapes: dict, counter: int,
                       module_obj):
    """
    Updates buml layers with the input shape if
    they are of type conv or dense.

    Parameters:
        lyr_input_shapes (dict): The dictionary containing the input 
        shapes of layers.
        counter (int): It is used to keep track of order of layers in
        the buml model.
        module_obj: A buml module object. It can be either a Layer,
            a subNN or a TensorOp.

    Returns:
        The updated counter and modifies the module_obj if it is
            a layer of type conv or dense to have their missing 
            input shapes.

    """
    mdl_type = module_obj.__class__.__name__
    if mdl_type in layers_buml2tf:
        mdl_type_tf = layers_buml2tf[mdl_type]
    else:
        mdl_type_tf = mdl_type
    if mdl_type_tf == lyr_input_shapes[counter][0]:
        in_feat = lyr_input_shapes[counter][1]
        if mdl_type_tf == "Dense":
            module_obj.in_features = in_feat
        elif mdl_type_tf in ["LSTM", "RNN", "GRU"]:
            module_obj.input_size = in_feat
        else:
            module_obj.in_channels = in_feat
        counter = increment_counter(counter, lyr_input_shapes)

    return counter

def load_nn_module(file_path: str, module_name: str):
    """
    Load a Python file as a module dynamically.
    
    Parameters:
        file_path (str): Path to the Python file.
        module_name (str): Name to assign to the module.
        
    Returns:
        The loaded module object.
    """
    file_path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def update_model(input_nn_type: str, buml_model: NN, file_path: str,
                 shape: tuple | None = None):
    """
    Updates conv and dense layers with their missing input shapes.
    Supports both sequential and subclassed models.
    """
    
    nn_name, init_code, call_code = extract_nn_code(file_path, input_nn_type)
    
    input_data = tf.keras.Input(shape=tuple(shape))
    nn_module = load_nn_module(file_path, "user_module")

    # Subclassing: modify call() to populate dict_details
    if input_nn_type == "subclassing":
        modified_code = modify_nn_call(init_code, call_code)
        print("fffff", modified_code)
        NNClass = load_modified_nn(modified_code, nn_name)
        lyr_input_shapes = NNClass()(input_data)  # call returns dict_details
        input_shape_from_call = extract_in_size_from_dict(lyr_input_shapes)
        
        tf_model = NNClass()
        lyr_input_shapes = get_input_shape_all(input_data, tf_model)

        for mdl_name, mdl_details in lyr_input_shapes.items():
            if input_shape_from_call.get(mdl_details[-1], None):
                if not isinstance(mdl_details[-2], dict):
                    mdl_details[1] = input_shape_from_call[mdl_details[-1]]

    # Sequential: symbolic tracing via get_input_shape_all
    else:
        tf_model = getattr(nn_module, nn_name)  # here nn_name is the variable name of the Sequential
        lyr_input_shapes = get_input_shape_all(input_data, tf_model)

    if lyr_input_shapes is None:
        return

    # Update buml model layers
    cnt_lyr_shape = next(iter(lyr_input_shapes))

    for mdl_obj in buml_model.modules:
        if isinstance(mdl_obj, NN):
            subnn_attr_dict = lyr_input_shapes[cnt_lyr_shape][1]
            cnt_ms_sub = next(iter(subnn_attr_dict))
            for subnn_mdl_obj in mdl_obj.modules:
                cnt_ms_sub = update_layers_attr(subnn_attr_dict, cnt_ms_sub, subnn_mdl_obj)
            cnt_lyr_shape = increment_counter(cnt_lyr_shape, lyr_input_shapes)

        elif isinstance(mdl_obj, Layer):
            cnt_lyr_shape = update_layers_attr(lyr_input_shapes, cnt_lyr_shape, mdl_obj)
