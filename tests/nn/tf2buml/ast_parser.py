"""
Module providing a class that extracts information from the AST of a 
neural network written in TensorFlow.
"""

import ast

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
            self.handle_forward(node)

    def handle_init(self, node):
        """
        It retrieves the sub_nn layers and stores them in the 'sub_nn' 
        dict. It also retreives the layers and their parameters and 
        stores them in the 'layers' dict.
        """
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name)):
            module_type = node.value.func.id
            if module_type == "Sequential":
                self.modules["sub_nns"][module_name] = {}
                layer_id = 0
                # Extract layers within Sequential
                for elt in node.value.args[0].elts:
                    if isinstance(elt, ast.Call):
                        layer, params = self.extract_layer(elt)
                        self.modules["sub_nns"][module_name][layer_id] = [layer, params]
                        layer_id+=1

        #simple calls to layers
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            layer, params = self.extract_layer(node.value)
            if len(node.value.args)>0: #rnn bidirectional
                if (isinstance(node.value.args[0], ast.Call) and
                    node.value.func.attr == "Bidirectional"):
                    layer, params = self.extract_layer(node.value.args[0])
                    params["bidirectional"] = True
            self.modules["layers"][module_name] = [layer, params]

    def handle_forward(self, node):
        """
        This method retrieves the input and output variables of modules
        and extracts tensorops details and stores the order of calling 
        the modules in the 'modules' dict.
        """
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "self":
                #populate inputs_outputs and layer_of_output
                module_name = node.value.func.attr
                self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                    node.targets[0].id]
                self.layer_of_output[node.targets[0].id] = module_name
                self.populate_modules(module_name)
            else:
                #tensorops
                self.extract_tensorops(node)
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.value.id == "tf":
                self.extract_tensorops(node)

        self.previous_assign = node

    def extract_params(self, call_node):
        """
        This method extracts the layers attributes (the keywords) and 
        returns them in a dictionary.
        """
        def get_param(param):
            if isinstance(param, ast.Constant):
                return param.value
            if isinstance(param, ast.Tuple):
                values = [el.value for el in param.elts]
                return values
            if isinstance(param, ast.List):  # List values
                return values
            if (isinstance(param, ast.UnaryOp) and
                  isinstance(param.op, ast.USub)):  # Negative numbers
                # Handle UnaryOp with USub for negative numbers
                if isinstance(param.operand, ast.Constant):
                    return -param.operand.value
            if isinstance(param, ast.Call):
                return None

            print("unhandled type", param)
            return param

        params = []
        keyword_params = {}
        for arg in call_node.args:
            value = get_param(arg)
            if value is not None:
                params.append(value)

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


    def extract_tensorops(self, node):
        """It extracts the tensorop name and its parameters."""
        op_name = node.value.func.attr
        op_args = node.value.args
        if op_name == "concat":
            op_args = node.value.args[0].elts
            layers_of_tensors = [self.layer_of_output[op_args[0].id],
                                 self.layer_of_output[op_args[1].id]]
            cat_dim = node.value.keywords[0].value.value
            tensorop_param = {"type": "concatenate",
                              "layers_of_tensors": layers_of_tensors,
                              "concatenate_dim": cat_dim}
        elif op_name == "matmul" or op_name == "multiply":
            op_type = "matmultiply" if op_name == "matmul" else "multiply"
            layers_of_tensors = [self.layer_of_output[op_args[0].id],
                                 self.layer_of_output[op_args[1].id]]
            tensorop_param = {"type": op_type,
                              "layers_of_tensors": layers_of_tensors}
        elif op_name == "transpose":
            op_args = node.value.keywords[0].value.elts
            transpose_dim = [op_args[0].value, op_args[1].value,
                             op_args[2].value]
            tensorop_param = {"type": op_name,
                              "transpose_dim": transpose_dim}
        elif op_name == "reshape":
            reshape_dim = [op_args[i].value for i in range(1, len(op_args))]
            tensorop_param = {"type": op_name,
                              "reshape_dim": reshape_dim}
        else:
            print(f"{op_name} is not recognized!")
            return

        self.modules["tensorops"]["op_"+str(self.tensor_op_counter)] = tensorop_param
        self.populate_modules("op_"+str(self.tensor_op_counter))
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


    def handle_outer_assignments(self, node):
        """
        It visits and extracts information from assignment statements
        called outside the NN class.
        """
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            if isinstance(node.value.func.value, ast.Name):
                if node.value.func.attr == "batch":
                    self.data_config["config"]["batch_size"] = node.value.args[0].value
            elif isinstance(node.value.func.value, ast.Attribute):
                if node.value.func.value.attr == "preprocessing":
                    keywords = node.value.keywords
                    for k in keywords:
                        if k.arg == "batch_size":
                            self.data_config["config"]["batch_size"] = k.value.value
                elif node.value.func.value.attr == "optimizers":
                    self.get_params_from_optimizer(node)
                elif node.value.func.value.attr == "losses":
                    self.data_config["config"]["loss_function"] = node.value.func.attr
        elif (isinstance(node.value, ast.Call) and
              isinstance(node.value.func, ast.Name)):
            self.handle_simple_outer_calls(node)
        elif (isinstance(node.value, ast.List) and
              isinstance(node.targets[0], ast.Name)):
            if node.targets[0].id == "metrics":
                elts = node.value.elts
                self.data_config["config"]["metrics"] = [elt.value for elt in elts]
        else:
            self.unprocessed_nodes.append(node)


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
        elif node.value.func.id == "load_dataset": #correct regression
            path = node.value.args[0].value
            if path.endswith("csv"):
                self.data_config["train_data"]["input_format"] = "csv"
            if "train" in node.targets[0].id or "train" in path:
                self.data_config["train_data"]["path_data"] = node.value.args[0].value
            elif "test" in node.targets[0].id or "test" in path:
                self.data_config["test_data"]["path_data"] = node.value.args[0].value
        elif node.value.func.id == "compute_mean_std":
            self.get_images_attr(node)

    def get_images_attr(self, node):
        """It extracts information related to images."""
        self.data_config["train_data"]["input_format"] = "images"
        if node.targets[0].elts[1].id != "_":
            self.data_config["train_data"]["normalize_images"] = True
        else:
            self.data_config["train_data"]["normalize_images"] = False
        node_args = node.value.keywords
        for node_arg in node_args:
            if node_arg.arg == "target_size":
                for past_node in self.unprocessed_nodes:
                    if isinstance(past_node.targets[0], ast.Name):
                        if past_node.targets[0].id == node_arg.value.id:
                            sizes = [i.value for i in past_node.value.elts]
                            self.data_config["train_data"]["images_size"] = sizes


    def get_params_from_optimizer(self, node):
        """It extracts information related to the optimizer."""
        self.data_config["config"]["optimizer"] = node.value.func.attr.lower()
        keywords = node.value.keywords
        learning_rate = next(
            (k.value.value for k in keywords if k.arg == 'learning_rate'), None
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
