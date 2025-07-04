"""
Module to extract information from the AST of a neural network
written in PyTorch and transforms it to a BUML model. 
It also extracts data and model configration attributes. 
"""
import ast
from besser.generators.nn_migration.ast_parser_nn import ASTParser
from besser.generators.nn_migration.torch2tf.definitions import (
    actv_fun_mapping, cnn_layers, loss_func_mapping
)

from besser.BUML.metamodel.nn import NN, Layer
import besser.BUML.metamodel.nn as mm_classes

from besser.generators.nn_migration.torch2tf.definitions import (
    layers_mapping, params_mapping, static_params,
    pos_params, int2list_params, lyrs_of_int2list_params
)
from besser.generators.nn_migration.transform_code import (
    process_positional_params, set_static_params, param_to_list
)


class ASTParserTorch(ASTParser):
    """
    Class visiting and parsing PyTorch code AST.

    Attributes:
        input_nn_type (str): The type of the nn input architecture.
        only_nn (str): Whether to process only the model definition or also
            its configuration and dataset.
        activation_functions (dict): it keeps track of the activation
            function of layers. Keys are layer names and values are
            activation function names.        
    """

    def __init__(self, input_nn_type: str, only_nn: bool):
        super().__init__(input_nn_type, only_nn)

        self.activation_functions = {}

    def handle_init(self, node: ast.Assign):
        """
        It retrieves the sub_nn layers, adds their activation functions 
        as parameters and stores them in the 'sub_nn' dict. It also 
        retreives the layers and their parameters and stores them in
        the 'layers' dict.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            # Checks if it is a Sequential or any NN layer
            module_type = node.value.func.attr
            if module_type == "Sequential":
                self.handle_sequential_layers(node, module_name)
            else:
                lyr_type, lyr_params = self.extract_layer(node.value)
                if lyr_type not in actv_fun_mapping:
                    lyr_type, lyr_params = transform_layer(
                        lyr_type, lyr_params, module_name
                    )
                    buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
                    self.buml_model.add_layer(buml_layer)
                else:
                    self.activation_functions[module_name] = lyr_type

        # Used to get the proper order from forward the method
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
        permute = False
        # Extracts layers within Sequential
        for elt in node.value.args:
            if isinstance(elt, ast.Call):
                lyr_type, lyr_params = self.extract_layer(elt)

                if lyr_type in actv_fun_mapping:
                    subnn.layers[-1].actv_func = actv_fun_mapping[lyr_type]
                elif lyr_type == "Permute":
                    last_lyr = subnn.layers[-1] if subnn.layers else None

                    if last_lyr is not None:
                        last_lyr_type = subnn.layers[-1].__class__.__name__
                        if last_lyr_type in cnn_layers:
                            subnn.layers[-1].permute_out = True
                        else:
                            permute = True
                    else:
                        permute = True
                else:
                    lyr_type, lyr_params = transform_layer(
                        lyr_type, lyr_params
                    )

                    lyr_params["name"] = f"layer_{layer_id}"
                    if permute:
                        lyr_params["permute_in"] = True
                        permute = False

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
        - sets the activation function as attribute of its layer.
        - adds permute_in attributes to cnn layers if they are preceeded by
        permute tensorop (the permute op is sometimes used before a cnn layer
        to make pytorch and tensorflow models equivalent as cnn in both 
        frameworks receive data in a different order).
        - sets the order of modules in buml model and processes tensorops.  

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model. 
        """
        if node.value.func.value.id == "self":
            # Populates inputs_outputs and layer_of_output from forward method
            module_name = node.value.func.attr
            self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                node.targets[0].id]
            self.layer_of_output[node.targets[0].id] = module_name

            is_subnn_obj = next((obj for obj in self.buml_model.sub_nns if
                                 obj.name == module_name), None)

            if module_name in self.activation_functions:
                prev_lyr_name = self.previous_assign.value.func.attr
                prev_lyr_obj = next((obj for obj in self.buml_model.modules if
                                     obj.name == prev_lyr_name), None)
                actv = self.activation_functions[module_name]
                prev_lyr_obj.actv_func = actv_fun_mapping[actv]

            elif not is_subnn_obj:
                self.is_permute_before_cnn(module_name)

            if module_name not in self.activation_functions:
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
        self.previous_assign = node

    def handle_forward_tuple_assignment(self, node: ast.Assign):
        """
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model. 
        """
        module_name = node.value.func.attr
        if node.targets[0].elts[0].id == "_":
            if isinstance(node.targets[0].elts[1], ast.Tuple):
                rnn_out = node.targets[0].elts[1].elts[0].id

            else:
                rnn_out = node.targets[0].elts[1].id
        else:
            rnn_out = node.targets[0].elts[0].id
            lyr_obj = next((obj for obj in self.buml_model.layers if
                            obj.name == module_name), None)
            lyr_obj.return_type = "full"


        rnn_in = node.value.args[0].id
        self.inputs_outputs[module_name] = [rnn_in, rnn_out]
        self.layer_of_output[rnn_out] = module_name
        module_obj = next((obj for obj in self.buml_model.layers if
                           obj.name == module_name), None)
        if not module_obj:
            module_obj = next((obj for obj in self.buml_model.sub_nns if
                               obj.name == module_name), None)
        self.buml_model.modules.append(module_obj)
        self.previous_assign = node

    def handle_forward_slicing(self, node: ast.Assign):
        """
        It handles rnn slicing calls such as 'x = x[:, -1, :]

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model. 
        """

        prev_module_name = self.previous_assign.value.func.attr
        lyr_obj = next((obj for obj in self.buml_model.layers if
                        obj.name == prev_module_name), None)

        if isinstance(node.value.slice, ast.UnaryOp):
            lyr_obj.return_type = "hidden"

        elif len(node.value.slice.elts) == 3:
            lyr_obj.return_type = "last"
        else:
            print("ast.Subscript is not recognised!")
        self.previous_assign = node


    def get_path_data(self, node: ast.Assign):
        """
        It extracts the path for training and test data
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the data_config dict with the data path. 
        """
        keywords = node.value.keywords
        path = next(
            (k.value.value for k in keywords if k.arg == 'root'), None
        )
        if "train" in node.targets[0].id or "train" in path:
            self.data_config["train_data"]["path_data"] = path

        elif "test" in node.targets[0].id or "test" in path:
            self.data_config["test_data"]["path_data"] = path

        else:
            print("Path is not recognised!")


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
        transform_args = node.value.args[0].elts
        #resize_arg = next(
        #    (arg for arg in transform_args if arg.func.attr == "Resize"), None
        #)
        normalize_arg = next((
            arg for arg in transform_args if arg.func.attr == "Normalize"),
            None
        )
        #if resize_arg:
        #    if isinstance(resize_arg.args[0], ast.Name):
        #        for past_node in self.unprocessed_nodes:
        #            name_node = (
        #                past_node
        #                if isinstance(past_node.targets[0], ast.Name)
        #                else None
        #            )
        #        if name_node:
        #            if name_node.targets[0].id == resize_arg.args[0].id:
        #                sizes = [i.value for i in name_node.value.elts]
        #                self.data_config["train_data"]["images_size"] = sizes

        #    elif isinstance(resize_arg.args[0], ast.Tuple):
        #        sizes = [i.value for i in resize_arg.args[0].elts]
        #        self.data_config["train_data"]["images_size"] = sizes

        if normalize_arg:
            self.data_config["train_data"]["normalize_images"] = True


    def is_permute_before_cnn(self, lyr_name: str):
        """
        It adds the permute op as parameter to its following layer if
        it is a cnn layer.

        Parameters:
            lyr_name (str): The name of the layer.

        Returns:
            None, but populates the buml model.
        """
        lyr_obj = next((obj for obj in self.buml_model.layers if
                        obj.name == lyr_name), None)
        lyr_type = lyr_obj.__class__.__name__
        if (lyr_type in cnn_layers and len(self.buml_model.tensor_ops)!=0):

            if (isinstance(self.previous_assign.targets[0], ast.Name) and
                isinstance(self.previous_assign.value, ast.Call)):
                if self.previous_assign.value.func.value.id != "self":
                    ops_name = self.previous_assign.value.func.attr
                    if ops_name == "permute":
                        lyrs = self.buml_model.layers
                        lyr_obj = next((obj for obj in lyrs if
                                        obj.name == lyr_name), None)
                        lyr_obj.permute_in = True
                        self.buml_model.tensor_ops.pop()
                        self.buml_model.modules.pop()


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
        if op_type == "permute":
            tensorop_param = self.extract_tensorop_permute(op_args)
        elif op_type == "cat":
            tensorop_param = self.extract_tensorop_concatenate(node)
        elif op_type == "mul" or op_type == "matmul":
            layers_of_tensors = [self.layer_of_output[op_args[0].id],
                                 self.layer_of_output[op_args[1].id]]
            tensorop_param = {"tns_type": op_type+"tiply",
                              "layers_of_tensors": layers_of_tensors}
        elif op_type == "transpose":
            transpose_dim = [op_args[i].value for i in range(len(op_args))]
            tensorop_param = {"tns_type": op_type,
                              "transpose_dim": transpose_dim}
        elif op_type == "reshape":
            reshape_dim = [op_args[0].value, op_args[1].value]
            tensorop_param = {"tns_type": op_type,
                              "reshape_dim": reshape_dim}
        else:
            print(f"{op_type} is not recognized!")
            return

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1


    def extract_tensorop_concatenate(self, node):
        """
        It extracts the concatenate tensorop information.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            The tensorop parameters.
        """
        ops_args = node.value.args[0].elts
        tensorop_param = None
        if isinstance(ops_args[0], ast.Subscript):
            prev_lyr_name = self.previous_assign.value.func.attr
            lyr_obj = next((obj for obj in self.buml_model.layers if
                            obj.name == prev_lyr_name), None)
            lyr_obj.return_type =  "hidden"
        else:
            layers_of_tensors = [self.layer_of_output[ops_args[0].id],
                                 self.layer_of_output[ops_args[1].id]]
            cat_dim = self.param_value(node.value.keywords[0].value)

            tensorop_param = {"tns_type": "concatenate",
                              "layers_of_tensors": layers_of_tensors,
                              "concatenate_dim": cat_dim}

        return tensorop_param


    def extract_tensorop_permute(self, ops_args):
        """
        It extracts the permute tensorop information.

        Returns:
            The tensorop parameters.
        """
        tensorop_param = None
        modules = self.buml_model.modules
        prev_module = modules[-1] if modules else None
        if isinstance(prev_module, Layer):
            lyr_type = prev_module.__class__.__name__
            if lyr_type in cnn_layers:
                prev_module.permute_out = True
        else:
            permute_dim=[ops_args[0].value, ops_args[1].value,
                            ops_args[2].value]
            tensorop_param = {"tns_type": "permute",
                              "permute_dim": permute_dim}
        return tensorop_param


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
        if isinstance(node.value.func.value, ast.Name):
            #if node.value.func.value.id == "datasets":
            #    self.get_path_data(node)
            if node.value.func.value.id == "transforms":
                self.get_images_attr(node)
            elif "Loss" in node.value.func.attr:
                loss = node.value.func.attr
                cnf = self.data_config["config"]
                cnf["loss_function"] = loss_func_mapping[loss]

        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.attr == "optim":
                self.get_params_from_optimizer(node)

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
                metrics = [elt.value for elt in node.value.elts]
                self.data_config["config"]["metrics"] = metrics
        elif isinstance(node.value, ast.Constant):
            self.handle_outer_constant_assignment(node)
        elif isinstance(node.value, ast.Tuple):
            self.handle_outer_tuple_assignment(node)
        else:
            self.unprocessed_nodes.append(node)




def transform_layer(lyr_type: str, lyr_params: dict,
                    lyr_name: str | None = None):
    """
    It transforms layers and their params from PyTorch to BUML.

    Parameters:
        lyr_type (str): The type of the layer (PyTorch).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.
        lyr_name (str | None): The name of the layer.

    Returns:
        The type of the layer and its parameters in BUML. 
    """

    param_to_list(lyr_type, lyr_params, int2list_params,
                               lyrs_of_int2list_params)

    lyr_params = process_params(lyr_type, lyr_params)
    lyr_params["name"] = lyr_name

    lyr_type = layers_mapping[lyr_type]
    return lyr_type, lyr_params



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

    for param in lyr_params:
        if param in params_mapping:
            updated_lyr_params[params_mapping[param]] = lyr_params[param]
        elif param == "actv_func":
            updated_lyr_params[param] = lyr_params[param]
        else:
            print(f"parameter {param} of layer {lyr_type} is not found!")


    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params
