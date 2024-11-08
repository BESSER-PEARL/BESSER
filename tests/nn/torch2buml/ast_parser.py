"""
Module providing a class that extracts information from the AST of a 
neural network written in PyTorch.
"""
import ast

from tests.nn.torch_to_buml.definitions import lookup_actv_fun, rnn_cnn_layers

class ASTParser(ast.NodeVisitor):
    """Class visiting and parsing the AST"""
    def __init__(self):
        super().__init__()
        self.previous_assign = None
        self.modules = {"layers": {},
                        "tensorops": {},
                        "sub_nns": {},
                        "order": {}}
        self.data_config = {"config": {},
                            "train_data": {"name": "train_data"},
                            "test_data": {"name": "test_data"}}
        self.inputs_outputs = {}
        self.layer_of_output = {}
        self.tensor_op_counter = 1
        self.in_class = False
        self.unprocessed_nodes = []
        self.nn_name = ""

    def visit_ClassDef(self, node):
        """
        It visits a ClassDef node in the AST, representing a class 
        definition in the source code. This method is called for each 
        class definition encountered in the AST (in our case only one 
        class representing the NN). It extracts the name of the NN, and
        visits its init and forward methods.

        Parameters:
        ----------
        node : ast.ClassDef
            The AST node representing a class definition.

        Returns:
        -------
        None
            The collected information is stored in instance attributes 
            or other data structures as needed.
        """
        self.nn_name = node.name
        self.in_class = True
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self.visit(child)
        self.in_class = False


    def visit_FunctionDef(self, node):
        """
        It visits a Function node in the AST, representing a function
        in the source code. It collects information and stores it in 
        instance attributes or other data structures as needed.

        Parameters:
        ----------
        node : ast.Function
            The AST node representing a function.

        Returns:
        -------
        None
            The collected information is stored in instance attributes 
            or other data structures as needed.
        """
        if node.name == "load_data":
            self.data_config["train_data"]["input_format"] = "csv"
        self.generic_visit(node)


    def visit_Assign(self, node):
        """
        It visits an Assign node in the AST, representing an assignment 
        statement in the source code. This method processes assignments
        by visiting nodes where values are assigned to variables.


        Parameters:
        ----------
        node : ast.Assign
            The AST node representing an assignment statement.

        Returns:
        -------
        None
            The collected information is stored in instance attributes 
            or other data structures as needed.
        """
        if self.in_class:
            self.handle_nn_definition(node)
        else:
            self.handle_outer_assignments(node)


    def visit_For(self, node):
        """
        It visits a For node in the AST, representing a for loop in the 
        source code. It collects information and stores it in instance 
        attributes or other data structures as needed.

        Parameters:
        ----------
        node : ast.For
            The AST node representing a for loop.

        Returns:
        -------
        None
            The collected information is stored in instance attributes 
            or other data structures as needed.
        """
        if (isinstance(node.iter, ast.Call) and
            "epochs" not in self.data_config["config"]):
            if node.iter.func.id == "range":
                self.data_config["config"]["epochs"] = node.iter.args[0].value
        elif (isinstance(node.iter, ast.Name) and
              isinstance(node.target, ast.Name)):
            if isinstance(node.body[0], ast.Assign):
                if (node.iter.id == "metrics" and
                    isinstance(node.body[0].value, ast.List)):
                    self.data_config["train_data"]["task_type"] = "multi_class"
                elif (node.iter.id == "metrics" and
                      "classification" in self.data_config["config"]):
                    self.data_config["train_data"]["task_type"] = "binary"



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

    def get_transform_attribute(self, node):
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


    def get_params_from_optimizer(self, node):
        """It extracts information related to the optimizer."""
        self.data_config["config"]["optimizer"] = node.value.func.attr.lower()
        keywords = node.value.keywords
        learning_rate = next((
            k.value.value for k in keywords if k.arg == 'lr'), None
        )
        momentum = next(
            (k.value.value for k in keywords if k.arg == 'momentum'), None
        )
        weight_decay = next(
            (k.value.value for k in keywords if k.arg == 'weight_decay'), None
        )

        self.data_config["config"]["learning_rate"] = learning_rate
        if momentum:
            self.data_config["config"]["momentum"] = momentum
        if weight_decay:
            self.data_config["config"]["weight_decay"] = weight_decay

    def handle_simple_outer_calls(self, node):
        """
        It extracts information from simple assignment statements 
        called outside the NN class.
        """
        if node.value.func.id == self.nn_name:
            self.nn_name = node.targets[0].id
        elif node.value.func.id == "classification_report":
            self.data_config["config"]["classification"] = True
        elif node.value.func.id == "mean_absolute_error":
            self.data_config["train_data"]["task_type"] = "regression"
            self.data_config["config"]["metrics"] = ["mae"]
        elif node.value.func.id == "load_data":
            if node.targets[0].id == "train_dataset":
                path = node.value.args[0].value
                self.data_config["train_data"]["path_data"] = path
            elif node.targets[0].id == "test_dataset":
                path = node.value.args[0].value
                self.data_config["test_data"]["path_data"] = path


    def handle_outer_assignments(self, node):
        """
        It visits and extracts information from assignment statements
        called outside the NN class.
        """
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            if isinstance(node.value.func.value, ast.Name):
                if node.value.func.value.id == "datasets":
                    self.get_path_data(node)
                elif node.value.func.value.id == "transforms":
                    self.get_transform_attribute(node)
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
        elif (isinstance(node.value, ast.Call) and
              isinstance(node.value.func, ast.Name)):
            self.handle_simple_outer_calls(node)

        elif (isinstance(node.value, ast.List) and
              isinstance(node.targets[0], ast.Name)):
            if node.targets[0].id == "metrics":
                metrics = [elt.value for elt in node.value.elts]
                self.data_config["config"]["metrics"] = metrics
        else:
            self.unprocessed_nodes.append(node)

    def handle_forward_simple_call(self, node):
        """
        This function: 1) retrieves the input and output variables of 
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

            if module_name.split("_")[0] in lookup_actv_fun:
                self.transform_actv_func(module_name)

            elif module_name not in self.modules["sub_nns"]:
                self.is_permute_before_rnn_cnn(module_name)

            if module_name.split("_")[0] not in lookup_actv_fun:
                self.populate_modules(module_name)
        else:
            #tensorops
            self.extract_tensorops(node)
        self.previous_assign = node

    def handle_forward_tuple_assignment(self, node):
        """
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        """
        if node.targets[0].elts[0].id == "_":
            if isinstance(node.targets[0].elts[1], ast.Tuple):
                rnn_out = node.targets[0].elts[1].elts[0].id
            else:
                rnn_out = node.targets[0].elts[1].id
        else:
            rnn_out = node.targets[0].elts[0].id
        rnn_in = node.value.args[0].id
        module_name = node.value.func.attr
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
                        if layer in lookup_actv_fun.values():
                            self.transform_actv_func(layer, in_forward=False)
                        else:
                            sub_nn = self.modules["sub_nns"][module_name]
                            sub_nn[layer_id] = [layer, params]
                            layer_id+=1
            else:
                layer, params = self.extract_layer(node.value)
                if module_name.split("_")[0] not in lookup_actv_fun:
                    self.modules["layers"][module_name] = [layer, params]


    def handle_nn_definition(self, node):
        """
        It is used to visit assignment nodes inside the class 
        (NN class).
        """
        # Init method
        if isinstance(node.targets[0], ast.Attribute):
            self.handle_init(node)

        #Forward method, simple calls
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.Call)):
            self.handle_forward_simple_call(node)

        #Forward RNN
        elif (isinstance(node.targets[0], ast.Tuple) and
              isinstance(node.value, ast.Call)):
            self.handle_forward_tuple_assignment(node)

        #Forward RNN
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.Subscript)):
            self.handle_forward_slicing(node)


    def extract_params(self, call_node):
        """
        This method extracts the layers attributes (the keywords) and 
        returns them in a dictionary.
        """
        def get_param(param):
            if isinstance(param, ast.Constant):
                return param.value
            elif isinstance(param, ast.Tuple):
                values = [el.value for el in param.elts]
                return values
            elif isinstance(param, ast.List):  # List values
                return values
            elif (isinstance(param, ast.UnaryOp) and
                  isinstance(param.op, ast.USub)):  # Negative numbers
                # Handle UnaryOp with USub for negative numbers
                if isinstance(param.operand, ast.Constant):
                    return -param.operand.value
            else:
                print("unhandled type")
                return None

        # Extract and print parameters for a layer
        params = []
        keyword_params = {}
        for arg in call_node.args:
            value = get_param(arg)
            if value is not None:
                params.append(value)
        #keyword_params["params"] = params

        for keyword in call_node.keywords:
            value = get_param(keyword.value)
            if value is not None:
                keyword_params[keyword.arg] = value
        return keyword_params


    def extract_layer(self, call_node):
        """It extracts the layer type and its parameters."""
        layer_type = call_node.func.attr
        params = self.extract_params(call_node)
        return layer_type, params

    def transform_actv_func(self, activ_func, in_forward=True):
        """
        It adds the activation function as parameter to the previous 
        layer.
        """
        if in_forward:
            previous_layer = list(self.modules["order"].keys())[-1]
            previous_layer_param = self.modules["layers"][previous_layer][1]
            previous_layer_param["actv_func"] = activ_func.split("_")[0]
        else:
            last_nn_name = list(self.modules["sub_nns"].keys())[-1]
            last_nn = self.modules["sub_nns"][last_nn_name]
            last_layer_name = list(last_nn.keys())[-1]
            #In sub_nns, the activation function does not have a name.
            #We get the name of its layer and add it as param.
            activ_func_buml = [k for (k, v) in lookup_actv_fun.items()
                                 if v == activ_func][0]
            last_nn[last_layer_name][1]["actv_func"] = activ_func_buml

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

    def extract_tensorops(self, node):
        """It extracts the tensorop name and its parameters."""
        ops_name = node.value.func.attr
        ops_args = node.value.args
        tensorop_param = None
        if ops_name == "permute":
            permute_dim=[ops_args[0].value, ops_args[1].value,
                            ops_args[2].value]
            tensorop_param = {"type": "permute", "permute_dim": permute_dim}
        elif ops_name == "cat":
            ops_args = node.value.args[0].elts
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
        elif ops_name == "mul" or ops_name == "matmul":
            layers_of_tensors = [self.layer_of_output[ops_args[0].id],
                                 self.layer_of_output[ops_args[1].id]]
            tensorop_param = {"type": ops_name+"tiply",
                              "layers_of_tensors": layers_of_tensors}
        elif ops_name == "transpose":
            transpose_dim = [ops_args[0].value, ops_args[1].value,
                                ops_args[2].value]
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


    def populate_modules(self, module_name):
        """
        It stores the layers, tensorops and sub_nns in 
        the modules dict.
        """
        if module_name in self.modules["sub_nns"]:
            self.modules["order"][module_name] = "sub_nn"
        elif module_name in self.modules["tensorops"]:
            self.modules["order"][module_name] = "tensorop"
        else:
            self.modules["order"][module_name] = "layer"
