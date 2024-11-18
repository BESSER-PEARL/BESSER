"""
Module providing a class that extracts information from the AST of a 
neural network written in TensorFlow or PyTorch.
"""

import ast
from abc import abstractmethod

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
            self.handle_outer_assignment(node)

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


    @abstractmethod
    def handle_init(self, node):
        """It handles the statements in the NN init method."""

    @abstractmethod
    def handle_forward_simple_call(self, node):
        """
        This method is relevant for PyTorch. 
        It 1) retrieves the input and output variables of 
        modules and the activation functions and adds them as 
        parameters to their layers. It also adds the permute op as 
        parameter to its following layer if it is of type cnn or 
        rnn (the permute op is sometimes used before a cnn and rnn layer 
        to make pytorch and tensorflow models equivalent as cnn and rnn
        in both frameworks receive data in a different order). This 
        method also extracts tensorops details and stores the order 
        of calling the modules in the 'modules' dict.
        """

    def handle_forward_tuple_assignment(self, node):
        """
        This method is relevant for PyTorch.
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        """

    def handle_forward_slicing(self, node):
        """
        This method is relevant for PyTorch.
        It handles rnn slicing calls such as 'x = x[:, -1, :]'
        """


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

    @abstractmethod
    def extract_tensorop(self, node):
        """It extracts the tensorop name and its parameters."""


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

    @abstractmethod
    def handle_outer_assignment(self, node):
        """
        It visits and extracts information from assignment statements
        called outside the NN class.
        """

    def handle_outer_simple_assignment(self, node):
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
                self.data_config["train_data"]["path_data"] = path
            elif "test" in node.targets[0].id or "test" in path:
                self.data_config["test_data"]["path_data"] = path
        elif node.value.func.id == "compute_mean_std":
            self.get_images_attr(node)

    @abstractmethod
    def get_images_attr(self, node):
        """It extracts information related to images."""


    def get_params_from_optimizer(self, node):
        """It extracts information related to the optimizer."""
        self.data_config["config"]["optimizer"] = node.value.func.attr.lower()
        keywords = node.value.keywords
        learning_rate = next(
            (k.value.value for k in keywords
             if k.arg in {'learning_rate', 'lr'}),
             None)
        momentum = next(
            (k.value.value for k in keywords if k.arg == 'momentum'),
            None)
        weight_decay = next(
            (k.value.value for k in keywords if k.arg == 'weight_decay'),
            None)
        self.data_config["config"]["learning_rate"] = learning_rate
        if momentum:
            self.data_config["config"]["momentum"] = momentum
        if weight_decay:
            self.data_config["config"]["weight_decay"] = weight_decay
