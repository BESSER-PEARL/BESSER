"""
This module defines the `TFGenerator` class that generates
TF code for neural networks based on the B-UML model.
"""

from collections.abc import Callable

from besser.BUML.metamodel.nn import NN
from besser.generators.nn.nn_code_generator import NNCodeGenerator
from besser.generators.nn.tf.utils_tf import SetupLayerSyntax, get_tensorop_syntax


class TFGenerator(NNCodeGenerator):
    """
    TFGenerator is a class that inherits from `NNCodeGenerator`.
    It generates TF code for neural networks training and evaluation
    based on the B-UML input model.

    Args:
        model (NN): An instance of the NN Model class representing
            the B-UML model.
        setup_layer (SetupLayerSyntax): The class that defines the
            syntax of layers.
        setup_tensorop (Callable): The function that defines the
            syntax of tensorops.
        output_dir (str, optional): The output directory where the
            generated code will be saved. Defaults to None.
        file_name (str): The name of the file where the generated
            code is stored.
        template_dir (str): The name of the jinja template directory.
        generation_type (str): 'subclassing' or 'sequential'.
    """
    def __init__(self, model: NN, output_dir: str | None = None,
                 generation_type: str = "subclassing",
                 strip_layer_counter_suffix: bool = False):

        setup_layer: SetupLayerSyntax = SetupLayerSyntax
        setup_tensorop: Callable = get_tensorop_syntax

        template_dir: str = "tf"
        file_name: str = "tf_nn.py"

        super().__init__(model, setup_layer, setup_tensorop, generation_type,
                         template_dir, file_name=file_name,
                         output_dir=output_dir,
                         strip_layer_counter_suffix=strip_layer_counter_suffix)

    def _cleanup_lambda_syntax(self, module, syntax, prev_out_var):
        """TensorFlow-specific variable extraction, mapping, and cleanup."""
        import re

        # Extract variable from syntax
        if hasattr(module, 'tns_type') and module.tns_type == 'dropout':
            match = re.search(r'\)\(([^,)]+)', syntax)
        else:
            match = re.search(r'\(([^,)]+)', syntax)
        var_replace = match.group(1).strip() if match else str(prev_out_var)

        # Map old variable to x
        syntax = re.sub(r'\b' + re.escape(var_replace) + r'\b', 'x', syntax)

        # Remove training parameter from dropout
        if hasattr(module, 'tns_type') and module.tns_type == 'dropout':
            syntax = syntax.replace(', training=training', '')

        return syntax
