"""
Module to extract information from the AST of a neural network
written in TensorFlow and transforms it to a BUML model. 
It also extracts data and model configration attributes. 
"""


import ast
from besser.generators.nn_migration.ast_parser_nn import ASTParser

from besser.generators.nn_migration.tf2torch.definitions import (
    layers_mapping, params_mapping, static_params, rnn_layers,
    pos_params, int2list_params, lyrs_of_int2list_params, loss_func_mapping
)
from besser.generators.nn_migration.transform_code import (
    process_positional_params, param_to_list, set_static_params
)

import besser.BUML.metamodel.nn as mm_classes
from besser.BUML.metamodel.nn import NN, Layer

class ASTParserTF(ASTParser):
    """
    Class visiting and parsing TensorFlow code AST
    
    Attributes:
        input_nn_type (str): The type of the nn input architecture.
        only_nn (str): Whether to process only the model definition or also
            its configuration and dataset.
        padding_amount (int | None): It  keeps track of padding in 
            ZeroPadding layer. In TF, padding is added to conv layers 
            using a separate layer, but in PyTorch it is defined as an 
            attribute of the conv layer.
    """

    def __init__(self, input_nn_type: str, only_nn:bool):
        super().__init__(input_nn_type, only_nn)

        self.padding_amount: int | None = None

    def handle_init(self, node: ast.Assign):
        """
        It retrieves the sub_nn layers and stores them in the 'sub_nn' 
        dict. It also retreives the layers and their parameters and 
        stores them in the 'layers' dict.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name)):
            module_type = node.value.func.id
            if module_type == "Sequential":
                self.handle_sequential_layers(node, module_name)

        #simple calls to layers
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            lyr_type, lyr_params = self.extract_layer(node.value)

            if len(node.value.args)>0: #rnn bidirectional
                if (isinstance(node.value.args[0], ast.Call) and
                    node.value.func.attr == "Bidirectional"):
                    lyr = node.value.args[0]
                    lyr_type, lyr_params = self.extract_layer(lyr)
                    lyr_params["bidirectional"] = True

            lyr_type, lyr_params, padding_amount = transform_layer(
                lyr_type, lyr_params, self.padding_amount, module_name
            )

            self.padding_amount = padding_amount
            if not lyr_type.startswith("ZeroPadding"):
                buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
                self.buml_model.add_layer(buml_layer)

        #to get the proper order from forward the method
        self.buml_model.modules.clear()


    def handle_sequential_layers(self, node: ast.Assign, seq_name: str):
        """
        It retrieves layers of a sequential model.
        It can be used for the main sequential nn and sub-nns.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.
            seq_name (str): The name of the sequential model.

        Returns:
            None, but populates the BUML model.

        """
        subnn: NN = NN(name=seq_name)
        layer_id = 1
        # Extract layers within Sequential
        for elt in node.value.args[0].elts:
            if isinstance(elt, ast.Call):
                lyr_type, lyr_params = self.extract_layer(elt)
                if len(elt.args)>0: #rnn bidirectional
                    if (isinstance(elt.args[0], ast.Call) and
                        isinstance(elt.func, ast.Attribute)):
                        if  elt.func.attr == "Bidirectional":
                            lyr = elt.args[0]
                            lyr_type, lyr_params = self.extract_layer(lyr)
                            lyr_params["bidirectional"] = True

                lyr_type, lyr_params, padding_amount = transform_layer(
                    lyr_type, lyr_params, self.padding_amount
                )
                self.padding_amount = padding_amount
                if not lyr_type.startswith("ZeroPadding"):
                    lyr_params["name"] = f"layer_{layer_id}"

                    subnn_layer = getattr(mm_classes, lyr_type)(**lyr_params)
                    subnn.add_layer(subnn_layer)
                    layer_id+=1

            elif isinstance(elt, ast.Name):
                subnn_obj = next((obj for obj in self.buml_model.sub_nns if
                                  obj.name == elt.id), None)
                subnn.add_sub_nn(subnn_obj)

        self.buml_model.add_sub_nn(subnn)


    def handle_forward_simple_call(self, node: ast.Assign):
        """
        This method: 
        - retrieves the input and output variables of modules 
        and populates 'inputs_outputs' and 'layer_of_output' dictionaries.
        - sets the order of modules in buml model and processes tensorops.  

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model. 
        """
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "self":
                module_name = node.value.func.attr
                #populate inputs_outputs and layer_of_output
                self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                    node.targets[0].id]
                self.layer_of_output[node.targets[0].id] = module_name
                module_obj = next((obj for obj in self.buml_model.layers if
                                   obj.name == module_name), None)
                if not module_obj:
                    subnns = self.buml_model.sub_nns
                    module_obj = next((obj for obj in subnns if
                                       obj.name == module_name), None)

                self.buml_model.modules.append(module_obj)
            else:
                #tensorops
                self.extract_tensorop(node)
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.value.id == "tf":
                self.extract_tensorop(node)
        self.previous_assign = node


    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """
        op_type = node.value.func.attr
        op_args = node.value.args
        tensorop_param = None
        if op_type == "concat":
            op_args = node.value.args[0].elts
            if (op_args[0].id in self.layer_of_output and
                op_args[1].id in self.layer_of_output):
                lyr1 = self.layer_of_output[op_args[0].id]
                lyr2 = self.layer_of_output[op_args[1].id]
                if lyr1 != lyr2:
                    layers_of_tensors = [lyr1, lyr2]
                    cat_dim = self.param_value(node.value.keywords[0].value)
                    tensorop_param = {"tns_type": "concatenate",
                                      "layers_of_tensors": layers_of_tensors,
                                      "concatenate_dim": cat_dim}
        elif op_type == "matmul" or op_type == "multiply":
            op_type = "matmultiply" if op_type == "matmul" else "multiply"
            layers_of_tensors = [self.layer_of_output[op_args[0].id],
                                 self.layer_of_output[op_args[1].id]]
            tensorop_param = {"tns_type": op_type,
                              "layers_of_tensors": layers_of_tensors}
        elif op_type == "transpose":
            op_args = node.value.keywords[0].value.elts
            transpose_dim = [op_args[0].value, op_args[1].value,
                             op_args[2].value]
            tensorop_param = {"tns_type": op_type,
                              "transpose_dim": transpose_dim}
        elif op_type == "reshape":
            reshape_dim = [op_args[i].value for i in range(1, len(op_args))]
            tensorop_param = {"tns_type": op_type,
                              "reshape_dim": reshape_dim}
        else:
            print(f"{op_type} is not recognized!")

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1

    def handle_outer_attribute_assignment(self, node: ast.Assign):
        """
        It visits and extracts information from assignment statements
        (node attributes) called outside the NN class or the sequentail model.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the data_config dictionary.
        """
        config = self.data_config["config"]
        if isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.attr == "optimizers":
                self.get_params_from_optimizer(node)
            elif node.value.func.value.attr == "losses":
                loss = node.value.func.attr
                config["loss_function"] = loss_func_mapping[loss]


    def handle_outer_assignments(self, node: ast.Assign):
        """
        It visits and extracts information from assignment statements
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config and data in 
                data_config dict.
        """
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            self.handle_outer_attribute_assignment(node)

        elif (isinstance(node.value, ast.Call) and
              isinstance(node.value.func, ast.Name)):
            self.handle_outer_simple_assignment(node)
        elif (isinstance(node.value, ast.List) and
              isinstance(node.targets[0], ast.Name)):
            if node.targets[0].id == "metrics":
                elts = node.value.elts
                config = self.data_config["config"]
                config["metrics"] = [elt.value for elt in elts]
        elif isinstance(node.value, ast.Constant):
            self.handle_outer_constant_assignment(node)
        elif isinstance(node.value, ast.Tuple):
            self.handle_outer_tuple_assignment(node)
        else:
            self.unprocessed_nodes.append(node)


    def get_images_attr(self, node: ast.Assign):
        """
        It extracts information related to images.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for data in data_config dict.
        """

        self.data_config["train_data"]["input_format"] = "images"
        if node.targets[0].elts[1].id != "_":
            self.data_config["train_data"]["normalize_images"] = True
        else:
            self.data_config["train_data"]["normalize_images"] = False


    def add_permute_dim(self):
        """
        It permutes input and output of cnn layers if needed
        to make pytorch and tensorflow equivalent.
        """
        cnns = ["Conv1D", "Conv2D", "Conv3D", "PoolingLayer"]
        bml_modules = self.buml_model.modules

        def iterate_and_permute(modules):
            prev_module = None
            lyr_out_permuted = []
            for i, module in enumerate(modules):
                if i == len(modules)-1:
                    next_module = None
                else:
                    next_module = modules[i+1]

                prev_module = self.permute(
                    module, bml_modules, prev_module, next_module,
                    lyr_out_permuted, cnns
                )

        if self.buml_model.sub_nns:
            for subnn in self.buml_model.sub_nns:
                iterate_and_permute(subnn.modules)

        iterate_and_permute(self.buml_model.modules)




    def permute(self, module, modules: list,
                prev_module, next_module,
                lyr_out_permuted: list, cnns: list):
        """
        It permutes input and output of `name` layer if needed
        
        Parameters:
            module: A buml module (either a layer, a subnn or a tensorop).
            modules (list): A list of modules.
            prev_module: The module defined before 'module' in the nn model.
            next_module: The module defined after 'module' in the nn model.
            lyr_out_permuted (list): A list containing the modules that their
                output has been already permuted.
            cnns (list): A list containing the names of layers to which 
                permutation applies.

        Returns:
            The module defined before 'module' in the nn model.
        
        """
        current_cnn, prev_cnn, next_cnn = False, False, False
        if isinstance(next_module, Layer):
            if next_module.__class__.__name__ in cnns:
                next_cnn = True

        if isinstance(module, Layer):
            if module.__class__.__name__ in cnns:
                current_cnn = True
                if module.name_module_input:
                    prev_module_name = module.name_module_input
                    prev_module = next((obj for obj in modules if
                                        obj.name == prev_module_name), None)

                if isinstance(next_module, Layer):
                    if next_module.name_module_input:
                        next_in = next_module.name_module_input
                        if next_in != module.name:
                            next_cnn = False

        if isinstance(prev_module, Layer):
            if prev_module.__class__.__name__ in cnns:
                prev_cnn = True
        else:
            prev_cnn = False
        if current_cnn:
            if not prev_cnn and prev_module not in lyr_out_permuted:
                module.permute_in = True
                lyr_out_permuted.append(prev_module)
            elif prev_cnn and prev_module not in lyr_out_permuted:
                if prev_module in lyr_out_permuted:
                    module.permute_in = True
            if not next_cnn:
                module.permute_out = True
                lyr_out_permuted.append(module.name)
        prev_module = module
        return prev_module



def transform_layer(lyr_type: str, lyr_params: dict,
                    padding_amount: int | None, layer_name=None):
    """
    It transforms layers and their params from TensorFlow to BUML.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.
        padding_amount (int | None): It  keeps track of padding in 
            ZeroPadding layer. In TF, padding is added to conv layers 
            using a separate layer, but in PyTorch and BUML it is defined
            as an attribute of the conv layer.
        lyr_name (str): The name of the layer.

    Returns:
        The type of the layer and its parameters in BUML. 
    """
    param_to_list(lyr_type, lyr_params, int2list_params,
                  lyrs_of_int2list_params)

    lyr_params, padding_amount = set_conv_padding(lyr_type, lyr_params,
                                                  padding_amount)

    lyr_params = process_params(lyr_type, lyr_params)
    lyr_params["name"] = layer_name
    if not lyr_type.startswith("ZeroPadding"):
        lyr_type = layers_mapping[lyr_type]

    return lyr_type, lyr_params, padding_amount



def process_params(lyr_type: str, lyr_params: dict):
    """
    It processes and transforms the layers' parameters.

    Parameters:
        lyr_type (str): The type of the layer (PyTorch).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.

    Returns:
        The parameters transformed into BUML. 
    """

    updated_lyr_params = {}
    process_positional_params(lyr_type, lyr_params, pos_params)

    lyrs_units = rnn_layers + ["Dense"]
    if lyr_type in lyrs_units and "units" in lyr_params:
        param = "out_features" if lyr_type == "Dense" else "hidden_size"
        updated_lyr_params[param] = lyr_params["units"]

    for param in lyr_params:
        if param == "activation":
            updated_lyr_params["actv_func"] = lyr_params[param]
        elif param == "return_sequences" and lyr_params[param] is True:
            updated_lyr_params["return_type"] = "full"
        elif param == "return_state" and lyr_params[param] is True:
            updated_lyr_params["return_type"] = "hidden"
        elif param in params_mapping:
            param_name = params_mapping[param]
            updated_lyr_params[param_name] = lyr_params[param]
        elif param == "units":
            pass
        else:
            print(f"parameter {param} of layer {lyr_type} is not found!")

    if "return_type" not in updated_lyr_params and lyr_type in rnn_layers:
        updated_lyr_params["return_type"] = "last"

    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params


def set_conv_padding(layer_type, lyr_params, padding_amount):
    """
    If padding is used before conv layers, its amount is stored in
    the 'padding_amount' attribute. This function adds the padding 
    to the conv layer.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.
        padding_amount (int | None): It  keeps track of padding in 
            ZeroPadding layer. In TF, padding is added to conv layers 
            using a separate layer, but in PyTorch and BUML it is defined
            as an attribute of the conv layer.

    Returns:
        The type of the layer and its parameters in BUML. 
    """


    if layer_type.startswith("ZeroPadding"):
        padding_amount = lyr_params["padding"]
    elif layer_type.startswith("Conv"):
        if padding_amount is not None:
            lyr_params["padding_amount"] = padding_amount
            padding_amount = None

    return lyr_params, padding_amount
