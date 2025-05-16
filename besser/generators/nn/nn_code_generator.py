"""
This module defines the `NNCodeGenerator` class that is inherited
by `TFGenerator` and `PytorchGenerator` to generates code for neural 
networks based on the B-UML model.
"""

import os
from typing import Callable
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.nn import NN
from besser.generators import GeneratorInterface
from besser.generators.nn.tf.utils_tf import (
    SetupLayerSyntax as SetupLayerTF
)
from besser.generators.nn.pytorch.utils_pytorch import (
    SetupLayerSyntax as SetupLayerTorch
)
from besser.generators.nn.pytorch.utils_pytorch import adjust_actv_func_name
from besser.generators.nn.utils_nn import handle_layer, handle_tensorop, \
    add_in_out_var_to_subnn


class NNCodeGenerator(GeneratorInterface):
    """
    NNCodeGenerator is a class that implements the GeneratorInterface and 
    is inherited by `TFGenerator` and `PytorchGenerator` to generates code 
    for neural networks training and evaluation based on the B-UML model. 

    Args:
        model (NN): An instance of the NN Model class representing 
            the B-UML model.
        setup_layer (Union[SetupLayerTF, SetupLayerTorch]): The class 
            that defines the syntax of layers.
        get_tensorop_syntax (Callable): The function that defines the
            syntax of tensorops.
        output_dir (str, optional): The output directory where the 
            generated code will be saved. Defaults to None.
        file_name (str): The name of the file where the generated
            code is stored.
        template_name (str): The name of the jinja template.
        template_dir (str): The name of the directory where the jinja 
            template `template_name` is stored. Either `tf` or `pytorch`.
        generation_type (str): 'subclassing' or 'sequential'
        channel_last (bool, optional): If true, PyTorch conv layers will 
            have their input and output permuted to match TF convention.
        modules_details (dict): A dict storing the NN modules syntax and 
            attributes.
        
    """
    def __init__(self, model: NN,
                 setup_layer: SetupLayerTF | SetupLayerTorch,
                 get_tensorop_syntax: Callable, generation_type: str,
                 template_dir: str, channel_last: bool | None = None,
                 file_name: str = "nn.py", output_dir: str = None):

        super().__init__(model, output_dir)
        self.setup_layer: SetupLayerTF | SetupLayerTorch = setup_layer
        self.get_tensorop_syntax: Callable = get_tensorop_syntax
        self.generation_type: str = generation_type
        self.channel_last: bool = channel_last
        self.template_dir: str = template_dir
        self.file_name: str = file_name

        if self.generation_type == "subclassing":
            self.template_name = f"template_{template_dir}_subclassing.py.j2"
        else:
            self.template_name = f"template_{template_dir}_sequential.py.j2"

        self.modules_details: dict = self.get_modules_details()


    def get_modules_details(self) -> str:
        """
        A module can be a layer, a sub_nn or a tensorop.
        The `modules_details` dict is created to keep track of 
        the syntax of modules, their tensor input variables, and 
        their tensor output variables in the forward method. 
        It has this structure:
                {"name_module": [syntax, out_var, in_var]}
        - syntax: The syntax of calling the module.
        - out_var: the output tensor variable of the module.
        - in_var: the input tensor variable of module.
        Example (TensoFlow):
                  {"l2": 
                  ["self.l2 = layers.Dense(units=40, activation='relu')",
                  "x_1", 
                  "x"]}

        For the case of layers, an additional element is added to 
        the list, representing the layer object.
        """
        counter_subnn: int = 0
        modules_details: dict = {}
        if "torch" in self.template_name:
            actv_func = True
        else:
            actv_func = False

        is_seq = False
        if self.generation_type == "sequential":
            is_seq = True
        for module in self.model.modules:
            module_type = module.__class__.__name__
            if module_type == "NN":
                subnn_details = {}
                for sub_nn_layer in module.layers:
                    handle_layer(
                        sub_nn_layer, self.setup_layer, subnn_details,
                        self.channel_last, actv_func, is_seq, is_subnn=True
                    )
                name_sub_nn = f"{module.name}_{counter_subnn}_nn"
                modules_details[name_sub_nn] = subnn_details
                counter_subnn += 1
                add_in_out_var_to_subnn(modules_details)
            elif module_type != "TensorOp":
                handle_layer(
                    module, self.setup_layer, modules_details,
                    self.channel_last, actv_func, is_seq, is_subnn=False
                )
            else:
                handle_tensorop(
                    module, modules_details, self.get_tensorop_syntax
                )
        if actv_func:
            adjust_actv_func_name(modules_details)

        return modules_details

    def generate(self, *args):
        """
        Generates NN code based on the provided B-UML model and saves 
        it to the specified output directory.
        If the output directory was not specified, the code generated 
        will be stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named nn_code.py
        """
        file_name = f"{self.file_name[:-3]}_{self.generation_type}.py"
        file_path = self.build_generation_path(file_name=file_name)
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), self.template_dir, "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template(self.template_name)

        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                model=self.model, modules_details=self.modules_details,
                generation_type=self.generation_type)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
