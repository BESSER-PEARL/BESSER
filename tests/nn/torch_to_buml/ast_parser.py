import ast

from tests.nn.torch_to_buml.definitions import lookup_actv_fun, rnn_cnn_layers

class ASTParser(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.previous_assign = None
        self.layers = {}
        self.sub_nns = {}
        self.tensorops = {}
        self.modules = {}
        self.inputs_outputs = {}
        self.layer_of_output = {}
        self.tensor_op_counter = 1
        self.in_class = False
        self.train_data = {"name": "train_data"}
        self.test_data = {"name": "test_data"}
        self.configuration = {}
        self.unprocessed_nodes = []

    
    def visit_ClassDef(self, node):
        self.nn_name = node.name #"nn_model"
        self.in_class = True
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self.visit(child)
        self.in_class = False


    def visit_FunctionDef(self, node):
        if node.name == "load_data":
            self.train_data["input_format"] = "csv"
        self.generic_visit(node)


    def visit_Assign(self, node):
        if self.in_class:
            self.handle_nn_definition(node)
        else:
            self.handle_outer_assignments(node)


    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and "epochs" not in self.configuration:
            if node.iter.func.id == "range":
                self.configuration["epochs"] = node.iter.args[0].value
        elif isinstance(node.iter, ast.Name) and isinstance(node.target, ast.Name):
            if isinstance(node.body[0], ast.Assign):
                if node.iter.id == "metrics" and isinstance(node.body[0].value, ast.List):
                    self.train_data["task_type"] = "multi_class"
                elif node.iter.id == "metrics" and "classification" in self.configuration:
                    self.train_data["task_type"] = "binary"

        #self.generic_visit(node)

    def get_path_data(self, node):
        keywords = node.value.keywords
        path = next((k.value.value for k in keywords if k.arg == 'root'), None)
        if "train" in node.targets[0].id or "train" in path:
            self.train_data["path_data"] = path
        elif "test" in node.targets[0].id or "test" in path:
            self.test_data["path_data"] = path
        else:
            print("Path is not recognised!")

    def get_transform_attribute(self, node):
        self.train_data["input_format"] = "images"
        transform_args = node.value.args[0].elts
        for arg in transform_args:
            if arg.func.attr == "Resize":
                if isinstance(arg.args[0], ast.Name):
                    self.train_data["resize_name"] = arg.args[0].id
                    for past_node in self.unprocessed_nodes:
                        if isinstance(past_node.targets[0], ast.Name):
                            if past_node.targets[0].id == arg.args[0].id:
                                sizes = [i.value for i in past_node.value.elts]
                                self.train_data["images_size"] = sizes

                elif isinstance(arg.args[0], ast.Tuple):
                    sizes = [i.value for i in arg.args[0].elts]
                    self.train_data["images_size"] = sizes
            elif arg.func.attr == "Normalize":
                self.train_data["normalize_images"] = True


    def get_params_from_optimizer(self, node):
        self.configuration["optimizer"] = node.value.func.attr.lower()
        keywords = node.value.keywords
        learning_rate = next((k.value.value for k in keywords if k.arg == 'lr'), None)
        momentum = next((k.value.value for k in keywords if k.arg == 'momentum'), None)
        weight_decay = next((k.value.value for k in keywords if k.arg == 'weight_decay'), None)
        
        self.configuration["learning_rate"] = learning_rate
        if momentum:
            self.configuration["momentum"] = momentum
        if weight_decay:
            self.configuration["weight_decay"] = weight_decay

    def handle_simple_outer_calls(self, node):
        if node.value.func.id == self.nn_name:
            self.nn_name = node.targets[0].id    
        elif node.value.func.id == "classification_report":
            self.configuration["classification"] = True
        elif node.value.func.id == "mean_absolute_error":
            self.train_data["task_type"] = "regression"
            self.configuration["metrics"] = ["mae"]
        elif node.value.func.id == "load_data":
            if node.targets[0].id == "train_dataset":
                self.train_data["path_data"] = node.value.args[0].value
            elif node.targets[0].id == "test_dataset":
                self.test_data["path_data"] = node.value.args[0].value


    def handle_outer_assignments(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if isinstance(node.value.func.value, ast.Name):
                if node.value.func.value.id == "datasets":
                    self.get_path_data(node)
                elif node.value.func.value.id == "transforms":
                    self.get_transform_attribute(node)
                elif "Loss" in node.value.func.attr:
                    self.configuration["loss_function"] = node.value.func.attr

            elif isinstance(node.value.func.value, ast.Attribute): 
                if node.value.func.attr == "DataLoader" and "batch_size" not in self.configuration:
                    keywords = node.value.keywords
                    batch_size = next((k.value.value for k in keywords if k.arg == 'batch_size'), None)
                    self.configuration["batch_size"] = batch_size
                elif node.value.func.value.attr == "optim":
                    self.get_params_from_optimizer(node)
        elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):  
            self.handle_simple_outer_calls(node)

        elif isinstance(node.value, ast.List) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "metrics":
                elts = node.value.elts
                self.configuration["metrics"] = [elt.value for elt in elts]
    
        else:
            self.unprocessed_nodes.append(node)


    def handle_forward_simple_call(self, node):  
        """This function: 1) retrieves the input and output variables of modules and
           the activation functions and adds them as parameters to their layers.
           It also adds the permute op as parameter to its following layers if
           they are of type cnn or rnn (the permute op is sometimes used before cnn
           and rnn layer to make pytorch and tensorflow models equivalent as cnn and
           rnn in both frameworks receive data in a different order).
           This function also extracts tensorops details and stores the order of
           calling the modules in the 'modules' dict"""              
        if node.value.func.value.id == "self":
            #populate inputs_outputs and layer_of_output from forward method
            module_name = node.value.func.attr
            self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                node.targets[0].id] 
            self.layer_of_output[node.targets[0].id] = module_name
            
            if module_name.split("_")[0] in lookup_actv_fun:
                self.transform_actv_func(module_name)

            elif module_name not in self.sub_nns:
                self.is_permute_before_rnn_cnn(module_name)

            if module_name.split("_")[0] not in lookup_actv_fun:
                self.populate_modules(module_name)
        else: 
            #tensorops
            self.extract_tensorops(node)
        self.previous_assign = node

    def handle_forward_tuple_assignment(self, node):
        """It handles rnn tuple assignments such as 'x, _ = self.l4(x)'"""
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
            self.layers[prev_module_name][1]["return_type"] = "hidden"
        elif len(node.value.slice.elts) == 3:
            prev_module_name = self.previous_assign.value.func.attr
            self.layers[prev_module_name][1]["return_type"] = "last"
        else:
            print("ast.Subscript is not recognised!")
        self.previous_assign = node

    def handle_init(self, node):
        """It retrieves the sub_nn layers, adds their activation functions 
           as parameters and stores them in the 'sub_nn' dict.
           It also retreives the layers and their parameters and stores them in
           the 'layers' dict"""
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and 
            isinstance(node.value.func, ast.Attribute)):
            # Check if it is a Sequential or any NN layer
            module_type = node.value.func.attr
            if module_type == "Sequential":
                self.sub_nns[module_name] = {}
                id = 0
                # Extract layers within Sequential
                for elt in node.value.args:
                    if isinstance(elt, ast.Call):
                        layer, params = self.extract_layer(elt)
                        if layer in lookup_actv_fun.values():
                            self.transform_actv_func(layer, in_forward=False)
                        else:
                            self.sub_nns[module_name][id] = [layer, params]
                            id+=1
            else:
                layer, params = self.extract_layer(node.value)
                if module_name.split("_")[0] not in lookup_actv_fun:
                    self.layers[module_name] = [layer, params]


    def handle_nn_definition(self, node):
        # Init method
        if isinstance(node.targets[0], ast.Attribute):
            self.handle_init(node)
        
        #Forward method, simple calls
        elif isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            self.handle_forward_simple_call(node)

        #Forward RNN
        elif isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Call):
            self.handle_forward_tuple_assignment(node)
            
        #Forward RNN
        elif isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Subscript):
            self.handle_forward_slicing(node)

    
        
        
    def extract_params(self, call_node):
        
        def get_param(param):
            if isinstance(param, ast.Constant):
                return param.value
            elif isinstance(param, ast.Tuple):
                values = [el.value for el in param.elts]
                return values
            elif isinstance(param, ast.List):  # List values
                return values
            elif isinstance(param, ast.UnaryOp) and isinstance(param.op, ast.USub):  # Negative numbers
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
        layer_type = call_node.func.attr
        params = self.extract_params(call_node)
        return layer_type, params
    

    def transform_actv_func(self, activ_func, in_forward=True):
        """this function adds the activation function as parameter to the previous layer"""
        if in_forward:
            previous_layer = list(self.modules.keys())[-1]
            self.layers[previous_layer][1]["actv_func"] = activ_func.split("_")[0]
        else:
            last_nn_name = list(self.sub_nns.keys())[-1]
            last_layer_name = list(self.sub_nns[last_nn_name].keys())[-1]
            #in sub_nns, the activation function does not have a name, it is 
            # called directly. Here we get its name in buml
            activ_func_buml = [k for (k, v) in lookup_actv_fun.items() if v == activ_func][0]
            self.sub_nns[last_nn_name][last_layer_name][1]["actv_func"] = activ_func_buml

    def is_permute_before_rnn_cnn(self, layer_name):
        if self.layers[layer_name][0] in rnn_cnn_layers and len(self.tensorops)!=0:
            #make sure the previous module is neither a layer nor a sub_nn
            if (isinstance(self.previous_assign.targets[0], ast.Name) and 
                isinstance(self.previous_assign.value, ast.Call)):
                if self.previous_assign.value.func.value.id != "self":
                    ops_name = self.previous_assign.value.func.attr
                    if ops_name == "permute": 
                        self.layers[layer_name][1]["permute_dim"] = True
                        self.tensorops.popitem()
                        self.modules.popitem()

    def extract_tensorops(self, node):
        ops_name = node.value.func.attr
        ops_args = node.value.args
        if ops_name == "permute":
            permute_dim=[ops_args[0].value, ops_args[1].value,
                            ops_args[2].value]
            tensorop_param = {"type": "permute", "permute_dim": permute_dim}
        elif ops_name == "cat":
            ops_args = node.value.args[0].elts
            if isinstance(ops_args[0], ast.Subscript):
                prev_module_name = self.previous_assign.value.func.attr
                self.layers[prev_module_name][1]["return_type"] = "hidden"
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
        
        
        self.tensorops["op_"+str(self.tensor_op_counter)] = tensorop_param
        self.populate_modules("op_"+str(self.tensor_op_counter))
        self.tensor_op_counter+=1


    def populate_modules(self, module_name):
        if module_name in self.sub_nns:
            self.modules[module_name] = "sub_nn"
        elif module_name in self.tensorops:
            self.modules[module_name] = "tensorop"
        else:
            self.modules[module_name] = "layer"