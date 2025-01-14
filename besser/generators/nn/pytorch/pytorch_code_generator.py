"""
This module defines the `PyTorchGenerator` class that generates
PyTorch code for neural networks based on the B-UML model.
"""

from typing import Callable
from besser.BUML.metamodel.nn import NN
from besser.generators.nn.pytorch.utils_pytorch import SetupLayerSyntax, \
    get_tensorop_syntax
from besser.generators.nn.nn_code_generator import NNCodeGenerator


class PytorchGenerator(NNCodeGenerator):
    """
    PytorchGenerator is a class that inherits from `NNCodeGenerator`.
    It generates Pytorch code for neural networks training and evaluation 
    based on the B-UML input model.

    Args:
        model (NN): An instance of the NN Model class representing 
            the B-UML model.
        setup_layer (SetupLayerSyntax): The class that defines
            the syntax of layers.
        setup_tensorop (Callable): The function that defines the
            syntax of tensorops.
        output_dir (str, optional): The output directory where the 
            generated code will be saved. Defaults to None.
        file_name (str): The name of the file where the generated
            code is stored.
        template (str): The name of the jinja template.
    """
    def __init__(self, model: NN, output_dir: str, generation_type: str,
                 channel_last: bool = True):

        setup_layer: SetupLayerSyntax = SetupLayerSyntax
        setup_tensorop: Callable = get_tensorop_syntax

        template_dir: str = "pytorch"
        file_name: str = "pytorch_nn.py"

        super().__init__(model, setup_layer, setup_tensorop, generation_type,
                         channel_last, template_dir, file_name, output_dir)
