"""
Module providing a class that extracts information from the AST of a 
neural network written in PyTorch.
"""
import ast
from besser.generators.nn_reverse.code2buml.ast_parser import ASTParser
from besser.generators.nn_reverse.torch2buml.definitions import (
    lookup_actv_fun, rnn_cnn_layers
)
from besser.generators.nn_reverse.torch2buml.transform_functions import (
    transform_actv_func
)

class ASTParserTorch(ASTParser):
    """Class visiting and parsing PyTorch code AST"""
    def __init__(self):
        super().__init__()
        self.activation_functions = {}

    def handle_init(self, node):
        """
        It retrieves the sub_nn layers, adds their activation functions 
        as parameters and stores them in the 'sub_nn' dict. It also 
        retreives the layers and their parameters and stores them in
        the 'layers' dict
        """
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            # Check if it is a Sequential or any NN layer
            module_type = node.value.func.attr
            if module_type == "Sequential":
                self.modules["sub_nns"][module_name] = {}
                layer_id = 0
                # Extract layers within Sequential
                for elt in node.value.args:
                    if isinstance(elt, ast.Call):
                        layer, params = self.extract_layer(elt)
                        if layer in lookup_actv_fun:
                            transform_actv_func(layer,
                                                self.modules,
                                                self.activation_functions,
                                                in_forward=False)
                        else:
                            sub_nn = self.modules["sub_nns"][module_name]
                            sub_nn[layer_id] = [layer, params]
                            layer_id+=1
            else:
                layer, params = self.extract_layer(node.value)
                if layer not in lookup_actv_fun:
                    self.modules["layers"][module_name] = [layer, params]
                else:
                    self.activation_functions[module_name] = layer

    def handle_forward_simple_call(self, node):
        """
        This method: 1) retrieves the input and output variables of 
        modules and the activation functions and adds them as 
        parameters to their layers. It also adds the permute op as 
        parameter to its following layer if it is of type cnn or 
        rnn (the permute op is sometimes used before a cnn and rnn layer 
        to make pytorch and tensorflow models equivalent as cnn and rnn
        in both frameworks receive data in a different order). This 
        method also extracts tensorops details and stores the order 
        of calling the modules in the 'modules' dict.
        """
        if node.value.func.value.id == "self":
            #populate inputs_outputs and layer_of_output from forward method
            module_name = node.value.func.attr
            self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                node.targets[0].id]
            self.layer_of_output[node.targets[0].id] = module_name

            if module_name in self.activation_functions:
                transform_actv_func(module_name, self.modules,
                                    self.activation_functions)

            elif module_name not in self.modules["sub_nns"]:
                self.is_permute_before_rnn_cnn(module_name)

            if module_name not in self.activation_functions:
                self.populate_modules(module_name)
        else:
            #tensorops
            self.extract_tensorop(node)
        self.previous_assign = node

    def handle_forward_tuple_assignment(self, node):
        """
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        """
        module_name = node.value.func.attr
        if node.targets[0].elts[0].id == "_":
            if isinstance(node.targets[0].elts[1], ast.Tuple):
                rnn_out = node.targets[0].elts[1].elts[0].id

            else:
                rnn_out = node.targets[0].elts[1].id
        else:
            rnn_out = node.targets[0].elts[0].id
            #we set the return type full here. The others are set or
            #overwritten based on other conditions
            self.modules["layers"][module_name][1]["return_type"] = "full"

        rnn_in = node.value.args[0].id
        self.inputs_outputs[module_name] = [rnn_in, rnn_out]
        self.layer_of_output[rnn_out] = module_name
        self.is_permute_before_rnn_cnn(module_name)
        self.populate_modules(module_name)
        self.previous_assign = node

    def handle_forward_slicing(self, node):
        """It handles rnn slicing calls such as 'x = x[:, -1, :]'"""
        if isinstance(node.value.slice, ast.UnaryOp):
            prev_module_name = self.previous_assign.value.func.attr
            prev_module_params = self.modules["layers"][prev_module_name][1]
            prev_module_params["return_type"] = "hidden"
        elif len(node.value.slice.elts) == 3:
            prev_module_name = self.previous_assign.value.func.attr
            prev_module_params = self.modules["layers"][prev_module_name][1]
            prev_module_params["return_type"] = "last"
        else:
            print("ast.Subscript is not recognised!")
        self.previous_assign = node


    def get_path_data(self, node):
        """It extracts the path for training and test data"""
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

    def get_images_attr(self, node):
        """It extracts information related to images."""
        self.data_config["train_data"]["input_format"] = "images"
        transform_args = node.value.args[0].elts
        resize_arg = next(
            (arg for arg in transform_args if arg.func.attr == "Resize"), None
        )
        normalize_arg = next((
            arg for arg in transform_args if arg.func.attr == "Normalize"),
            None
        )
        if resize_arg:
            if isinstance(resize_arg.args[0], ast.Name):
                for past_node in self.unprocessed_nodes:
                    name_node = (
                        past_node
                        if isinstance(past_node.targets[0], ast.Name)
                        else None
                    )
                if name_node:
                    if name_node.targets[0].id == resize_arg.args[0].id:
                        sizes = [i.value for i in name_node.value.elts]
                        self.data_config["train_data"]["images_size"] = sizes

            elif isinstance(resize_arg.args[0], ast.Tuple):
                sizes = [i.value for i in resize_arg.args[0].elts]
                self.data_config["train_data"]["images_size"] = sizes
        elif normalize_arg:
            self.data_config["train_data"]["normalize_images"] = True


    def is_permute_before_rnn_cnn(self, layer_name):
        """
        It adds the permute op as parameter to its following layer if
        it is a cnn or rnn layer.
        """
        if (self.modules["layers"][layer_name][0] in rnn_cnn_layers and
            len(self.modules["tensorops"])!=0):
            #make sure the previous module is neither a layer nor a sub_nn
            if (isinstance(self.previous_assign.targets[0], ast.Name) and
                isinstance(self.previous_assign.value, ast.Call)):
                if self.previous_assign.value.func.value.id != "self":
                    ops_name = self.previous_assign.value.func.attr
                    if ops_name == "permute":
                        layer_param = self.modules["layers"][layer_name][1]
                        layer_param["permute_dim"] = True
                        self.modules["tensorops"].popitem()
                        self.modules["order"].popitem()


    def extract_tensorop(self, node):
        """It extracts the tensorop name and its parameters."""
        ops_name = node.value.func.attr
        ops_args = node.value.args
        tensorop_param = None
        if ops_name == "permute":
            permute_dim=[ops_args[0].value, ops_args[1].value,
                            ops_args[2].value]
            tensorop_param = {"type": "permute", "permute_dim": permute_dim}
        elif ops_name == "cat":
            tensorop_param = self.extract_tensorop_concatenate(node)
        elif ops_name == "mul" or ops_name == "matmul":
            layers_of_tensors = [self.layer_of_output[ops_args[0].id],
                                 self.layer_of_output[ops_args[1].id]]
            tensorop_param = {"type": ops_name+"tiply",
                              "layers_of_tensors": layers_of_tensors}
        elif ops_name == "transpose":
            transpose_dim = [ops_args[i].value for i in range(len(ops_args))]
            tensorop_param = {"type": ops_name,
                              "transpose_dim": transpose_dim}
        elif ops_name == "reshape":
            reshape_dim = [ops_args[0].value, ops_args[1].value]
            tensorop_param = {"type": ops_name,
                              "reshape_dim": reshape_dim}
        else:
            print(f"{ops_name} is not recognized!")
            return

        if tensorop_param:
            tensorops_id = "op_"+str(self.tensor_op_counter)
            self.modules["tensorops"][tensorops_id] = tensorop_param
            self.populate_modules(tensorops_id)
            self.tensor_op_counter+=1


    def extract_tensorop_concatenate(self, node):
        """
        It extracts the concatenate tensorop information
        """
        ops_args = node.value.args[0].elts
        tensorop_param = None
        if isinstance(ops_args[0], ast.Subscript):
            prev_module_name = self.previous_assign.value.func.attr
            module_param = self.modules["layers"][prev_module_name][1]
            module_param["return_type"] = "hidden"
        else:
            layers_of_tensors = [self.layer_of_output[ops_args[0].id],
                                 self.layer_of_output[ops_args[1].id]]
            cat_dim = node.value.keywords[0].value.value
            tensorop_param = {"type": "concatenate",
                              "layers_of_tensors": layers_of_tensors,
                              "concatenate_dim": cat_dim}
        return tensorop_param

    def handle_outer_attribute_assignment(self, node):
        """
        It visits and extracts information from assignment statements
        (node attributes) called outside the NN class.
        """
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "datasets":
                self.get_path_data(node)
            elif node.value.func.value.id == "transforms":
                self.get_images_attr(node)
            elif "Loss" in node.value.func.attr:
                loss = node.value.func.attr
                self.data_config["config"]["loss_function"] = loss

        elif isinstance(node.value.func.value, ast.Attribute):
            if (node.value.func.attr == "DataLoader" and
                "batch_size" not in self.data_config["config"]):
                keywords = node.value.keywords
                batch_size = next((
                    k.value.value for k in keywords
                    if k.arg == 'batch_size'
                ), None)
                self.data_config["config"]["batch_size"] = batch_size
            elif node.value.func.value.attr == "optim":
                self.get_params_from_optimizer(node)

    def handle_outer_assignment(self, node):
        """
        It visits and extracts information from assignment statements
        called outside the NN class.
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
        else:
            self.unprocessed_nodes.append(node)
