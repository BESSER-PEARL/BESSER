"""
Module providing a class that extracts information from the AST of a 
neural network written in TensorFlow or PyTorch and transforms it to
a BUML model.
It also extracts data and model configration attributes.
"""

import ast
from abc import abstractmethod
from besser.BUML.metamodel.nn import NN, Layer
from besser.generators.nn_reverse_copy.transform_code import (
    set_remaining_params
)

class ASTParser(ast.NodeVisitor):
    """
    Class visiting and parsing the AST.

    Attributes:
        input_nn_type (str): The type of the nn input architecture.
        only_nn (str): Whether to process only the model definition or also
            its configuration and dataset.
        buml_model (NN): The BUML NN model.
        previous_assign (ast.AST | None): It keeps track of the previous 
            module in the forward method.
        data_config (dict): A dict to keep track of NN config and 
            data attributes.
        inputs_outputs (dict): It keeps track of input and output variables 
            of layers.
        layer_of_output (dict): It keeps track of name of layers given their
            output var.
        tensor_op_counter (int): Counter used to assign names of tensorops.
        in_class (bool): It tracks the processing of NN architecture class.
        unprocessed_nodes (list): It keeps track of unprocessed nodes to 
            retrieve other variables later, like 'image_size'. 

    """
    def __init__(self, input_nn_type: str, only_nn: bool):
        super().__init__()

        self.input_nn_type: str = input_nn_type
        self.only_nn: bool = only_nn
        self.buml_model: NN = NN(name="my_nn") #the name will be updated later
        self.previous_assign: ast.AST | None = None

        self.data_config: dict = {"config": {}, "train_data": {},
                                  "test_data": {}}
        self.inputs_outputs: dict = {}
        self.layer_of_output: dict = {}
        self.tensor_op_counter: int = 1
        self.in_class: bool = False
        self.unprocessed_nodes: list = []


    def visit_ClassDef(self, node: ast.ClassDef):
        """
        It visits a ClassDef node in the AST, representing a class 
        definition in the source code. This method is called for each 
        class definition encountered in the AST (in case of 'subclassing'
        architecture, the model is defined inside a class). It extracts 
        the name of the NN, and visits its init and forward methods
        to create the BUML model.

        Parameters:
            node (ast.ClassDef): The AST node representing a class definition.

        Returns:
            None, but the the buml model is created. 
        """

        # Retrieve the name of the model
        if isinstance(node.bases[0], ast.Attribute):
            base_name = node.bases[0].attr
        else: #isinstance(node.bases[0], ast.Name):
            base_name = node.bases[0].id

        if base_name == "Model" or base_name == "Module":
            self.buml_model.name = node.name
            self.in_class = True

        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self.visit(child)

        # Set 'input_reused' and 'name_module_input' layer parameters
        self.set_remaining_lyr_params()
        # Add permute before and after conv blocks to make TF and Pytorch eqv
        self.add_permute_dim()
        # Set in_class var to False at the end of NN architecture processing
        self.in_class = False


    def set_remaining_lyr_params(self):
        """
        It iterates through the layers to set their 'input_reused'
        and 'name_module_input' parameters.
        """
        for module in self.buml_model.modules:
            if isinstance(module, Layer):
                set_remaining_params(
                    module, self.inputs_outputs, self.layer_of_output
                )


    def add_permute_dim(self):
        """
        It permutes input and output of PyTorch cnn layers if needed
        to make PyTorch and Tensorflow equivalent.
        It is only applied to TensorFlow code and implemented in the
        ASTParserTF child class.
        """


    def visit_For(self, node: ast.For):
        """
        It visits a For node in the AST, representing a for loop in the 
        source code. It collects information and stores it in instance 
        attributes or other data structures as needed.

        Parameters:
            node(ast.For): The AST node representing a for loop.

        Returns:
            None, but collects some config and data attributes and stores 
                them in self.data_config dictionary.
        """
        if self.only_nn is False:
            if (isinstance(node.iter, ast.Name) and
                  isinstance(node.target, ast.Name)):
                if isinstance(node.body[0], ast.Assign):
                    if (node.iter.id == "metrics" and
                        isinstance(node.body[0].value, ast.List)):
                        dt_conf = self.data_config["train_data"]
                        dt_conf["task_type"] = "multi_class"

                    elif (node.iter.id == "metrics" and
                          "classification" in self.data_config["config"]):
                        self.data_config["train_data"]["task_type"] = "binary"

        else:
            self.generic_visit(node)



    def visit_Assign(self, node: ast.Assign):
        """
        It visits an Assign node in the AST, representing an assignment 
        statement in the source code. This method processes assignments
        by visiting nodes where values are assigned to variables.


        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model and collects attributes 
                for config and data in data_config dict.
        """
        if self.input_nn_type == "subclassing":
            if self.in_class:
                self.handle_subclassing_nn(node)

        elif self.input_nn_type == "sequential":
            self.handle_sequential_nn(node)

        if not self.only_nn:
            self.handle_outer_assignments(node)


    def visit_AnnAssign(self, node: ast.AnnAssign):
        """
        It visits annotated assigments. Checks if the model is annotated.

        Parameters:
            node (ast.AnnAssign): The AST node representing an annotated 
                assignment statement.

        Returns:
            None, but it populates the buml model.
        """
        if self.input_nn_type == "sequential":
            self.handle_sequential_nn(node)



    def handle_sequential_nn(self, node: ast.Assign):
        """
        It handles the sequential NN model architecture.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.

        """
        seq = False
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id == "Sequential":
                    seq = True
            elif isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "Sequential":
                    seq = True
        if seq:
            target = (
                node.targets[0] if hasattr(node, 'targets') else node.target
            )
            if isinstance(target, ast.List):
                self.buml_model.name = target.attr

            elif isinstance(target, ast.Name):
                self.buml_model.name = target.id

            elif isinstance(target, ast.Attribute):
                self.buml_model.name = target.attr

            self.handle_sequential_layers(node, self.buml_model.name)

            self.add_permute_dim()



    @abstractmethod
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



    def handle_subclassing_nn(self, node: ast.Assign):
        """
        It is used to visit assignment nodes inside the NN class in
        subclassing architecture.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
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
    def handle_init(self, node: ast.Assign):
        """
        It handles the statements in the NN init method.

        Parameters:
            node (ast.Assign): The AST node representing an assignment 
                statement.

        Returns:
            None, but populates the BUML model. 
        """

    @abstractmethod
    def handle_forward_simple_call(self, node: ast.Assign):
        """
        This method: 
        - retrieves the input and output variables of modules 
        and populates 'inputs_outputs' and 'layer_of_output' dictionaries.
        - sets the activation function as attribute of its layer (for PyTorch).
        - adds permute_in attributes to cnn layers if they are preceeded by
        permute tensorop (the permute op is sometimes used before a cnn layer
        to make pytorch and tensorflow models equivalent as cnn in both 
        frameworks receive data in a different order). Relevant for PyTorch.
        - sets the order of modules in buml model and processes tensorops.  

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model. 
        """

    def handle_forward_tuple_assignment(self, node: ast.Assign):
        """
        This method is relevant for PyTorch.
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment 
                statement.

        Returns:
            None, but populates the BUML model. 
        """

    def handle_forward_slicing(self, node: ast.Assign):
        """
        This method is relevant for PyTorch.
        It handles rnn slicing calls such as 'x = x[:, -1, :]'
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment 
                statement.

        Returns:
            None, but populates the BUML model. 
        """


    def param_value(self, param: ast.Call):
        """
        Get the value of a parameter based on its type.
        
        Parameters:
            node (ast.Call): The AST node representing a call statement.

        Returns:
            The value of the parameter.
        """
        if isinstance(param, ast.Constant):
            return param.value
        elif isinstance(param, ast.Tuple):
            values = [el.value for el in param.elts]
            return values
        elif isinstance(param, ast.List):  # List values
            values = [el.value for el in param.elts]
            return values
        elif (isinstance(param, ast.UnaryOp) and
            isinstance(param.op, ast.USub)):  # Negative numbers
            # Handle UnaryOp with USub for negative numbers
            if isinstance(param.operand, ast.Constant):
                return -param.operand.value
        elif isinstance(param, ast.Name):
            return param.id
        elif isinstance(param, ast.Attribute):
            value = self.param_value(param.value)
            return f"{value}.{param.attr}"
        elif isinstance(param, ast.Call):
            func_name = self.param_value(param.func)
            args = ", ".join(self.param_value(arg) for arg in param.args)
            return f"{func_name}({args})"

        print("unhandled type", param)
        return param

    def extract_layer_params(self, call_node: ast.Call):
        """
        This method extracts the layers attributes (keywords 
        and positional params) and returns them as a dictionary.

        Parameters:
            node (ast.Call): The AST node representing a call statement.

        Returns:
            A dictionary containing the parameters.
        """
        params = []
        keyword_params = {}
        for arg in call_node.args:
            value = self.param_value(arg)
            if value is not None:
                params.append(value)

        for keyword in call_node.keywords:
            value = self.param_value(keyword.value)
            if value is not None:
                keyword_params[keyword.arg] = value
        keyword_params["positional_params"] = params
        return keyword_params


    def extract_layer(self, call_node: ast.Call):
        """
        It extracts the layer type and its parameters.
        
        Parameters:
            node (ast.Call): The AST node representing a call statement.

        Returns:
            The layer type and its parameters.
        """
        if isinstance(call_node.func, ast.Name):
            layer_type = call_node.func.id
        else: #isinstance(call_node.func, ast.Attribute):
            layer_type = call_node.func.attr

        params = self.extract_layer_params(call_node)
        return layer_type, params

    @abstractmethod
    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """



    @abstractmethod
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


    def handle_outer_simple_assignment(self, node: ast.Assign):
        """
        It extracts information from simple assignment statements 
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config and data in 
                data_config dict.
        """
        if node.value.func.id == self.buml_model.name:
            self.buml_model.name = node.targets[0].id
        elif node.value.func.id == "classification_report":
            self.data_config["config"]["classification"] = True
        elif node.value.func.id == "mean_absolute_error":
            self.data_config["train_data"]["task_type"] = "regression"
            self.data_config["config"]["metrics"] = ["mae"]
        elif node.value.func.id == "compute_mean_std":
            self.get_images_attr(node)


    def handle_outer_constant_assignment(self, node: ast.Assign):
        """
        It extracts information from constant assignment statements 
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config and data in 
                data_config dict.
        """
        #if node.value.func.id == self.buml_model.name:
        #    self.buml_model.name = node.targets[0].id
        #elif node.value.func.id == "classification_report":
        #    self.data_config["config"]["classification"] = True
        #elif node.value.func.id == "mean_absolute_error":
        #    self.data_config["train_data"]["task_type"] = "regression"
        #    self.data_config["config"]["metrics"] = ["mae"]
        if node.targets[0].id == "batch_size":
            self.data_config["config"]["batch_size"] = node.value.value
        elif node.targets[0].id == "train_path":
            path = node.value.value
            if path.endswith("csv"):
                self.data_config["train_data"]["input_format"] = "csv"
            self.data_config["train_data"]["path_data"] = path
        elif node.targets[0].id == "test_path":
            self.data_config["test_data"]["path_data"] = node.value.value
        elif node.targets[0].id == "epochs":
            self.data_config["config"]["epochs"] = node.value.value
        elif node.targets[0].id == "image_size":
            self.data_config["train_data"]["images_size"] = node.value.value


    def handle_outer_tuple_assignment(self, node: ast.Assign):
        """
        It extracts information from tuple assignment statements 
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects images_size attribute in 
                data_config dict.
        """
        if node.targets[0].id == "image_size":
            size = [i.value for i in node.value.elts]
            self.data_config["train_data"]["images_size"] = size


    @abstractmethod
    def get_images_attr(self, node: ast.Assign):
        """
        It extracts information related to images.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for data in data_config dict.
        """


    def get_params_from_optimizer(self, node: ast.Assign):
        """
        It extracts information related to the optimizer.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config in data_config dict.
        """
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
