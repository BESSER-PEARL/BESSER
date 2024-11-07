from typing import Dict
from besser.BUML.metamodel.nn import Layer, TensorOp
import tensorflow as tf
import tensorflow_addons as tfa
import os
import random
from PIL import Image
import numpy as np

modules_details = dict()

"""
A module can be a layer, a sub_nn or a tensorop.
This dict is created to keep track of the syntax of the modules,
their tensor input variable, and their tensor output variable 
in the forward method. It has this structure:
{"name_module": [syntax, out_var, in_var]}
syntax: The syntax of calling the module.
out_var: the output tensor variable of the module.
in_var: the input tensor variable of module.
Example: {"l2": ["self.l2 = layers.Dense(units=40, activation='relu')", "x_1", "x"]}
PS: for the case of layers, an additional element is added to the list, representing the layer object.
"""



def setup_general_layer(layer: Layer) -> str:
    actv_func = setup_activation_function(layer)
    if layer.__class__.__name__ == "LinearLayer":
        my_layer = f"self.{layer.name} = layers.Dense(units={layer.out_features}, activation={actv_func})"
    elif layer.__class__.__name__ == "FlattenLayer":
        my_layer = f"self.{layer.name} = layers.Flatten()"
    elif layer.__class__.__name__ == "EmbeddingLayer":
        my_layer = f"self.{layer.name} = layers.Embedding(input_dim={layer.num_embeddings}, output_dim={layer.embedding_dim})"
    return my_layer
    

def setup_layer_modifier(layer: Layer) -> str:
    if layer.__class__.mro()[1].__name__ == "NormalizationLayer":
        if layer.__class__.__name__ == "BatchNormLayer":
            my_layer = f"self.{layer.name} = layers.BatchNormalization()"
        elif layer.__class__.__name__ == "LayerNormLayer":
            if layer.normalized_shape[1] != None:
                if layer.normalized_shape[2] != None:
                    normalized_shape = [layer.normalized_shape[0], layer.normalized_shape[1], layer.normalized_shape[2]]
                else:
                    normalized_shape = [layer.normalized_shape[0], layer.normalized_shape[1]]
            else:
                normalized_shape = [layer.normalized_shape[0]]
    
            my_layer = f"self.{layer.name} = layers.LayerNormalization(axis={normalized_shape})"
    elif layer.__class__.__name__ == "DropoutLayer":
        my_layer = f"self.{layer.name} = layers.Dropout(rate={layer.rate})"
    return my_layer       


def setup_rnn(layer: Layer) -> str:
    actv_func = setup_activation_function(layer)
    layer_type = layer.__class__.__name__[:-5]
    my_layer = f"layers.{layer_type}(units={layer.hidden_size}, activation={actv_func}, dropout={layer.dropout}"
    if layer.return_type == "full":
        my_layer = f"{my_layer}, return_sequences=True)"
    elif layer.return_type == "hidden":
        my_layer = f"{my_layer}, return_state=True)"
    else:
        my_layer = f"{my_layer})"

    if layer.bidirectional == True:
        my_layer = f"self.{layer.name} = layers.Bidirectional({my_layer})"
    else:
        my_layer = f"self.{layer.name} = {my_layer}"
    return my_layer

def setup_activation_function(layer: Layer) -> str:
    if hasattr(layer, 'actv_func'):
        if layer.actv_func != None:
            return f"'{layer.actv_func}'"
        else:
            return None
        

def setup_cnn(layer: Layer) -> str:
    actv_func = setup_activation_function(layer)
    if layer.__class__.__name__ == "Conv1D" and layer.padding_amount != 0:
        my_layer = f"self.{layer.name}_pad = layers.ZeroPadding1D(padding={layer.padding_amount})#self.{layer.name} = layers.Conv1D(filters={layer.out_channels}, kernel_size={layer.kernel_dim[0]}, strides={layer.stride_dim[0]}, padding='{layer.padding_type}', activation={actv_func})"
    elif layer.__class__.__name__ == "Conv1D":
        my_layer = f"self.{layer.name} = layers.Conv1D(filters={layer.out_channels}, kernel_size={layer.kernel_dim[0]}, strides={layer.stride_dim[0]}, padding='{layer.padding_type}', activation={actv_func})"
    elif layer.__class__.__name__ == "Conv2D" and layer.padding_amount != 0:
        my_layer = f"self.{layer.name}_pad = layers.ZeroPadding2D(padding={layer.padding_amount})#self.{layer.name} = layers.Conv2D(filters={layer.out_channels}, kernel_size=({layer.kernel_dim[0]}, {layer.kernel_dim[1]}), strides=({layer.stride_dim[0]}, {layer.stride_dim[1]}), padding='{layer.padding_type}', activation={actv_func})"
    elif layer.__class__.__name__ == "Conv2D":
        my_layer = f"self.{layer.name} = layers.Conv2D(filters={layer.out_channels}, kernel_size=({layer.kernel_dim[0]}, {layer.kernel_dim[1]}), strides=({layer.stride_dim[0]}, {layer.stride_dim[1]}), padding='{layer.padding_type}', activation={actv_func})"
    elif layer.__class__.__name__ == "Conv3D" and layer.padding_amount != 0:
        my_layer = f"self.{layer.name}_pad = layers.ZeroPadding3D(padding={layer.padding_amount})#self.{layer.name} = layers.Conv3D(filters={layer.out_channels}, kernel_size=({layer.kernel_dim[0]}, {layer.kernel_dim[1]}, {layer.kernel_dim[2]}), strides=({layer.stride_dim[0]}, {layer.stride_dim[1]}, {layer.stride_dim[2]}), padding='{layer.padding_type}', activation={actv_func})"
    elif layer.__class__.__name__ == "Conv3D":    
        my_layer = f"self.{layer.name} = layers.Conv3D(filters={layer.out_channels}, kernel_size=({layer.kernel_dim[0]}, {layer.kernel_dim[1]}, {layer.kernel_dim[2]}), strides=({layer.stride_dim[0]}, {layer.stride_dim[1]}, {layer.stride_dim[2]}), padding='{layer.padding_type}', activation={actv_func})"
    elif layer.__class__.__name__ == "PoolingLayer":
        if layer.pooling_type == "max" or layer.pooling_type == "average":
            pl = "MaxPool" if layer.pooling_type == "max" else "AveragePooling"        
            if layer.dimension == "1D":
                my_layer = f"self.{layer.name} = layers.{pl}1D(pool_size={layer.kernel_dim[0]}, strides={layer.stride_dim[0]}, padding='{layer.padding_type}')"
            elif layer.dimension == "2D":
                my_layer = f"self.{layer.name} = layers.{pl}2D(pool_size=({layer.kernel_dim[0]}, {layer.kernel_dim[1]}), strides=({layer.stride_dim[0]}, {layer.stride_dim[1]}), padding='{layer.padding_type}')"
            else:
                my_layer = f"self.{layer.name} = layers.{pl}3D(pool_size=({layer.kernel_dim[0]}, {layer.kernel_dim[1]}, {layer.kernel_dim[2]}), strides=({layer.stride_dim[0]}, {layer.stride_dim[1]}, {layer.stride_dim[2]}), padding='{layer.padding_type}')"     
        else:
            pl = "AdaptiveAveragePooling" if layer.pooling_type == "adaptive_average" else "AdaptiveMaxPooling"
            if layer.dimension == "1D":
                my_layer = f"self.{layer.name} = tfa.layers.{pl}1D(output_size={layer.output_dim[0]})"
            elif layer.dimension == "2D":
                my_layer = f"self.{layer.name} = tfa.layers.{pl}2D(output_size=({layer.output_dim[0]}, {layer.output_dim[1]}))"
            else:
                my_layer = f"self.{layer.name} = tfa.layers.{pl}3D(output_size=({layer.output_dim[0]}, {layer.output_dim[1]}, {layer.output_dim[2]}))"
    return my_layer




def get_input_variable(layer, modules_details, prev_out_variable):
    my_keys = list(modules_details.keys())
    if layer.name_layer_input != None:
        if layer.name_layer_input+"_layer" in my_keys:
            return modules_details[layer.name_layer_input + "_layer"][1]
        elif layer.name_layer_input+"_nn" in my_keys:
            return  modules_details[layer.name_layer_input + "_nn"]["in_out_variable"]
        #module.name_layer_input+"_op" in my_keys:
        else:
            return modules_details[layer.name_layer_input + "_op"][1]
    else:
        return prev_out_variable
        
def get_layers_output_for_tensorops(layers_names, modules_details):
    my_keys = list(modules_details.keys())
    out_variables = []
    for layer_name in layers_names:
        if layer_name+"_layer" in my_keys:
            out_variables.append(modules_details[layer_name + "_layer"][1])
        else:
            out_variables.append(modules_details[layer_name + "_op"][1])
    return out_variables

def get_previous_out_variable(modules_details, previous_module):
    if isinstance(modules_details[previous_module], dict):
        return modules_details[previous_module]["in_out_variable"]
    else:
        return modules_details[previous_module][1]
    
def get_layer_syntax(layer, modules_details):
    parent_class = layer.__class__.mro()[1].__name__
    if (parent_class == "ConvolutionalLayer" or parent_class == "CNN"):    
        layer_synt = setup_cnn(layer)
    elif parent_class == "RNN":
        layer_synt = setup_rnn(layer)
    elif parent_class == "GeneralLayer":
        layer_synt = setup_general_layer(layer)
    elif (parent_class == "LayerModifier" or parent_class == "NormalizationLayer"):
        layer_synt = setup_layer_modifier(layer)

    return layer_synt, modules_details
  

def add_in_out_variable_to_subnn(modules_details):
    previous_module = list(modules_details.keys())[-1]
    if previous_module.endswith("nn"):
        if len(modules_details) == 1:
            in_out_variable = "x"
        elif isinstance(list(modules_details.values())[-2], dict):
            in_out_variable = list(modules_details.values())[-2]["in_out_variable"]
        else:
            # get the output variable of the layer before the sub_nn
            in_out_variable = list(modules_details.values())[-2][1]
        modules_details[previous_module]["in_out_variable"] = in_out_variable
    return modules_details


def get_tensorop_output_empty_dict(tensorOp):
    if tensorOp.input_reused == True:
        out_variable = "x_1"
    else:
        out_variable = "x"
    return out_variable


def get_layer_variable_empty_dict(layer):
    if layer.input_reused == True:
        out_variable_layer, in_variable_layer = "x_1", "x"
    else:
        out_variable_layer, in_variable_layer = "x", "x"
    return out_variable_layer, in_variable_layer

def get_layer_variable_input_reused(layer, prev_out_variable, modules_details):
    if prev_out_variable == "x":
        out_variable_layer = "x_1"
    else:
        out_variable_layer = f"x_{int(prev_out_variable.split('_')[-1])+1}"
    in_variable_layer = get_input_variable(layer, modules_details, prev_out_variable)
    return out_variable_layer, in_variable_layer

def get_layer_variable_input_not_reused(layer, prev_out_variable, modules_details):   
    out_variable_layer = prev_out_variable
    in_variable_layer = get_input_variable(layer, modules_details, prev_out_variable)
    return out_variable_layer, in_variable_layer

def get_tensorop_out_var_input_reused(prev_out_variable):
    if prev_out_variable == "x":
        out_variable = "x_1"
    else:
        out_variable = f"x_{int(prev_out_variable.split('_')[-1])+1}"
    return out_variable


def get_rnn_output_variable(modules_details):
    for module_def in modules_details.values():
        if len(module_def) == 4:
            if module_def[-1].__class__.mro()[1].__name__ == "RNN" and module_def[-1].return_hidden:
                module_def[1] = "_, " + module_def[1]
            elif module_def[-1].__class__.mro()[1].__name__ == "RNN":
                module_def[1] = module_def[1] + ", _"      
    return modules_details

def get_layer_in_out_variables(layer: Layer, modules_details: Dict) -> Dict:
    layer_synt, modules_details = get_layer_syntax(layer, modules_details)
    if (len(modules_details) == 0):
        out_variable_layer, in_variable_layer = get_layer_variable_empty_dict(layer)
    else:
        previous_module = list(modules_details.keys())[-1]
        prev_out_variable = get_previous_out_variable(modules_details, previous_module)
        if layer.input_reused == True:
            out_variable_layer, in_variable_layer = get_layer_variable_input_reused(layer, prev_out_variable, modules_details)
        else:
            out_variable_layer, in_variable_layer = get_layer_variable_input_not_reused(layer, prev_out_variable, modules_details)
    
    modules_details[layer.name + "_layer"] = [layer_synt, out_variable_layer, in_variable_layer, layer]
    return modules_details

"""if the output of a layer is expected to be reused as input to more than one layer,
the first layer that receives that output should have the parameter input_reused set to True"""


def get_tensorop_syntax(tensorOp, modules_details):
    if len(list(modules_details.keys())) == 0:
        prev_out_variable = "x"
    else:
        previous_module = list(modules_details.keys())[-1]
        prev_out_variable = get_previous_out_variable(modules_details, previous_module)
    if tensorOp.type == "reshape":
        reshape_dim = ', '.join([str(i) for i in tensorOp.reshape_dim])
        ts_op_synt = f"tf.reshape({prev_out_variable}, {reshape_dim})"
    elif tensorOp.type == "concatenate":
        tensors = get_layers_output_for_tensorops(tensorOp.layers_of_tensors, modules_details)
        tensors = ', '.join(tensors)
        ts_op_synt = f"tf.concat([{tensors}], axis={tensorOp.concatenate_dim})"
    elif tensorOp.type == "transpose":
        transpose_dim = ", ".join([str(i) for i in tensorOp.transpose_dim])
        ts_op_synt = f"tf.transpose({prev_out_variable}, perm=[{transpose_dim}])"
    elif tensorOp.type == "permute":
        permute_dim = ", ".join([str(i) for i in tensorOp.permute_dim])
        ts_op_synt = f"tf.transpose({prev_out_variable}, perm=[{permute_dim}])"
    else:
        tensors = []
        for elem in tensorOp.layers_of_tensors:
            if type(elem) == str:
                out_variable_layer = get_layers_output_for_tensorops([elem], modules_details)[0]
                tensors.append(out_variable_layer)
            else:
                tensors.append(elem)
        tensors = ', '.join([str(i) for i in tensors])

        if tensorOp.type == "multiply":
            ts_op_synt = f"tf.math.multiply({tensors})"
        else:
            ts_op_synt = f"tf.matmul({tensors})"
    return ts_op_synt

def get_tensorop_out_variable(tensorOp: TensorOp, modules_details: Dict) -> Dict:
    ts_op_synt = get_tensorop_syntax(tensorOp, modules_details)
    if len(modules_details) == 0:
        out_variable  = get_tensorop_output_empty_dict(tensorOp)
    else:
        previous_module = list(modules_details.keys())[-1]
        prev_out_variable = get_previous_out_variable(modules_details, previous_module)
        if tensorOp.input_reused == True:
            out_variable  = get_tensorop_out_var_input_reused(prev_out_variable)
        else:   
            out_variable = prev_out_variable
    
    modules_details[tensorOp.name + "_op"] = [ts_op_synt, out_variable]
    return modules_details



def loss_with_weight_decay(base_loss_fn, weight_decay):
    def loss_fn(y_true, y_pred, model):
        # Calculate the base loss
        base_loss = base_loss_fn(y_true, y_pred)
        # Calculate the L2 regularization term
        l2_loss = 0
        for weight in model.trainable_variables:
            l2_loss += tf.reduce_sum(tf.square(weight))
        # Combine the base loss with the L2 regularization term
        total_loss = base_loss + weight_decay * l2_loss
        return total_loss
    return loss_fn


def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)
    np_image = np.array(image.convert('RGB'), dtype=np.float32)
    return np_image

def compute_mean_std(image_dir, num_samples=100, target_size=(256, 256)):
    image_files = [os.path.join(root, file)
                   for root, _, files in os.walk(image_dir)
                   for file in files]
    sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    all_pixels = []
    for file in sampled_files:
        np_image = preprocess_image(file, target_size)
        all_pixels.append(np_image.reshape(-1, 3))
    
    all_pixels = np.concatenate(all_pixels)

    # Rescale if necessary
    if all_pixels.max() >= 1:
        scale = True
        all_pixels /= 255.0
    else:
        scale=False
    
    return scale, np.mean(all_pixels, axis=0).tolist(), np.std(all_pixels, axis=0).tolist()


