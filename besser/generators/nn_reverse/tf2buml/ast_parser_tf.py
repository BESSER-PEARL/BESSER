"""
Module providing a class that extracts information from the AST of a 
neural network written in TensorFlow.
"""

import ast
from besser.generators.nn_reverse.code2buml.ast_parser_nn import ASTParser

class ASTParserTF(ASTParser):
    """Class visiting and parsing TensorFlow code AST"""

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
                self.handle_sequential_layers(node, module_name)

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


    def handle_sequential_layers(self, node, seq_name):
        """
        It retrieves layers of a sequential model.
        It can be used for the main sequential nn and sub-nns.
        """
        dict_layers = {}
        self.modules["sub_nns"][seq_name] = {}
        layer_id = 1
        # Extract layers within Sequential
        for elt in node.value.args[0].elts:
            if isinstance(elt, ast.Call):
                layer, params = self.extract_layer(elt)
                if len(elt.args)>0: #rnn bidirectional
                    if (isinstance(elt.args[0], ast.Call) and
                        isinstance(elt.func, ast.Attribute)):
                        if  elt.func.attr == "Bidirectional":
                            layer, params = self.extract_layer(elt.args[0])
                            params["bidirectional"] = True

                dict_layers[f"layer_{layer_id}"] = [layer, params]
                layer_id+=1
            elif isinstance(elt, ast.Name):
                dict_layers[elt.id] = "predefined"
        self.modules["sub_nns"][seq_name] = dict_layers


    def handle_forward_simple_call(self, node):
        """
        This method retrieves the input and output variables of modules
        and extracts tensorops details and stores the order of calling 
        the modules in the 'modules' dict.
        """
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "self":
                module_name = node.value.func.attr
                #populate inputs_outputs and layer_of_output
                self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                    node.targets[0].id]
                self.layer_of_output[node.targets[0].id] = module_name
                self.populate_modules(module_name)
            else:
                #tensorops
                self.extract_tensorop(node)
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.value.id == "tf":
                self.extract_tensorop(node)
        self.previous_assign = node


    def handle_forward_tuple_assignment(self, node):
        """
        This method is relevant for PyTorch.
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        """
        module_name = node.value.func.attr
        outputs = [elem.id for elem in node.targets[0].elts if elem.id != "_"]

        self.inputs_outputs[module_name] = [node.value.args[0].id,
                                            outputs]
        for output in outputs:
            self.layer_of_output[output] = module_name
        self.populate_modules(module_name)

    def extract_tensorop(self, node):
        """It extracts the tensorop name and its parameters."""
        op_name = node.value.func.attr
        op_args = node.value.args
        tensorop_param = None
        if op_name == "concat":
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
        elif op_name == "matmul" or op_name == "multiply":
            op_type = "matmultiply" if op_name == "matmul" else "multiply"
            layers_of_tensors = [self.layer_of_output[op_args[0].id],
                                 self.layer_of_output[op_args[1].id]]
            tensorop_param = {"tns_type": op_type,
                              "layers_of_tensors": layers_of_tensors}
        elif op_name == "transpose":
            op_args = node.value.keywords[0].value.elts
            transpose_dim = [op_args[0].value, op_args[1].value,
                             op_args[2].value]
            tensorop_param = {"tns_type": op_name,
                              "transpose_dim": transpose_dim}
        elif op_name == "reshape":
            reshape_dim = [op_args[i].value for i in range(1, len(op_args))]
            tensorop_param = {"tns_type": op_name,
                              "reshape_dim": reshape_dim}
        else:
            print(f"{op_name} is not recognized!")

        if tensorop_param:
            tensorops_id = "op_"+str(self.tensor_op_counter)
            self.modules["tensorops"][tensorops_id] = tensorop_param
            self.populate_modules(tensorops_id)
            self.tensor_op_counter+=1

    def handle_outer_attribute_assignment(self, node):
        """
        It visits and extracts information from assignment statements
        (node attributes) called outside the NN class.
        """
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.attr == "batch":
                config = self.data_config["config"]
                config["batch_size"] = node.value.args[0].value
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.attr == "preprocessing":
                keywords = node.value.keywords
                for k in keywords:
                    if k.arg == "batch_size":
                        config = self.data_config["config"]
                        config["batch_size"] = k.value.value
            elif node.value.func.value.attr == "optimizers":
                self.get_params_from_optimizer(node)
            elif node.value.func.value.attr == "losses":
                config = self.data_config["config"]
                config["loss_function"] = node.value.func.attr


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
                elts = node.value.elts
                config = self.data_config["config"]
                config["metrics"] = [elt.value for elt in elts]
        else:
            self.unprocessed_nodes.append(node)


    def get_images_attr(self, node):
        """It extracts information related to images."""
        self.data_config["train_data"]["input_format"] = "images"
        if node.targets[0].elts[1].id != "_":
            self.data_config["train_data"]["normalize_images"] = True
        else:
            self.data_config["train_data"]["normalize_images"] = False
        node_args = node.value.keywords
        size_arg = next(
            (elem for elem in node_args if elem.arg == "target_size"), None
        )
        if size_arg:
            for past_node in self.unprocessed_nodes:
                name_node = (
                    past_node
                    if isinstance(past_node.targets[0], ast.Name)
                    else None
                )
            if name_node:
                if name_node.targets[0].id == size_arg.value.id:
                    sizes = [i.value for i in name_node.value.elts]
                    self.data_config["train_data"]["images_size"] = sizes
