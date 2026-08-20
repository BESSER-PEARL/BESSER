"""
This module defines the `PyTorchGenerator` class that generates
PyTorch code for neural networks based on the B-UML model.
"""

from typing import Callable
import re
from besser.BUML.metamodel.nn import NN
from besser.BUML.metamodel.nn.neural_network import (
    LinearLayer, Conv1D, Conv2D, Conv3D, SimpleRNNLayer, LSTMLayer, GRULayer
)
from besser.generators.nn.pytorch.utils_pytorch import SetupLayerSyntax, \
    get_tensorop_syntax
from besser.generators.nn.nn_code_generator import NNCodeGenerator


class PytorchGenerator(NNCodeGenerator):
    """
    PytorchGenerator is a class that inherits from `NNCodeGenerator`.
    It generates Pytorch code for neural networks training and
    evaluation based on the B-UML input model.

    Attributes:
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
        template_dir (str): The name of the jinja template directory.
        generation_type (str): 'subclassing' or 'sequential'
        channel_last (bool, optional): If true, PyTorch conv layers
            will have their input and output permuted to match TF
            convention.
    """
    def __init__(self, model: NN, output_dir: str | None = None,
                 generation_type: str = "subclassing",
                 channel_last: bool = False,
                 strip_layer_counter_suffix: bool = False):

        self._validate_required_layer_attributes(model)

        setup_layer: SetupLayerSyntax = SetupLayerSyntax
        setup_tensorop: Callable = get_tensorop_syntax

        template_dir: str = "pytorch"
        file_name: str = "pytorch_nn.py"

        super().__init__(model, setup_layer, setup_tensorop, generation_type,
                         template_dir, channel_last, file_name, output_dir,
                         strip_layer_counter_suffix=strip_layer_counter_suffix)

    def _validate_dropout_for_sequential(self, module):
        """Handle PyTorch-specific dropout constraints."""
        if (hasattr(module, 'dropout_training_aware') and
            module.dropout_training_aware is True
        ):
            raise ValueError(
                "PyTorch sequential generation does not support dropout"
                "tensorops with dropout_training_aware=True. "
                "Use subclassing mode instead."
            )

    def _cleanup_lambda_syntax(self, module, syntax, prev_out_var):
        """PyTorch-specific variable extraction, mapping,
        and cleanup."""

        # If syntax is just a bare variable, return 'x'
        if re.match(r'^[a-zA-Z_]\w*$', syntax):
            return 'x'

        tns_type = module.tns_type if hasattr(module, 'tns_type') else None

        # Method calls on variable (var.method(...))
        if tns_type in {'permute', 'mean', 'max', 'squeeze', 'unsqueeze',
                       'transpose', 'repeat', 'reshape', 'shape_dim'}:
            syntax = re.sub(r'^[a-zA-Z_]\w*\.', 'x.', syntax)

        # Function calls with variable as first arg (func(var, ...))
        elif tns_type in {'interpolate', 'normalize', 'pad', 'multiply',
                         'zeros_like', 'split', 'dropout'}:
            syntax = re.sub(r'\([a-zA-Z_]\w*\b', '(x', syntax, count=1)

        # Binary operations (var op value)
        elif tns_type in {'binop_add', 'binop_subtract', 'binop_multiply',
                         'binop_divide', 'binop_floor_divide'}:
            syntax = re.sub(r'^[a-zA-Z_]\w*\b', 'x', syntax)

        # Subscript (var[...])
        elif tns_type == 'subscript':
            syntax = re.sub(r'^[a-zA-Z_]\w*\b', 'x', syntax)

        else:
            # Fallback: replace first identifier
            syntax = re.sub(r'^[a-zA-Z_]\w*\b', 'x', syntax)

        return syntax

    def _wrap_in_lambda(self, syntax):
        """PyTorch uses Lambda module wrapper instead of raw lambda
        in sequential."""
        return f"Lambda(lambda x: {syntax})"

    @staticmethod
    def _validate_required_layer_attributes(model):
        """Validate that critical layer attributes are set for PyTorch
        generation."""
        for module in model.modules:
            if isinstance(module, LinearLayer):
                if module.in_features is None:
                    raise ValueError(
                        f"PyTorch Linear layer '{module.name}' requires "
                        "'in_features' to be set. Cannot generate code "
                        "with in_features=None."
                    )

            elif isinstance(module, (Conv1D, Conv2D, Conv3D)):
                if module.in_channels is None:
                    raise ValueError(
                        f"PyTorch Conv layer '{module.name}' requires "
                        "'in_channels' to be set. Cannot generate code "
                        "with in_channels=None."
                    )

            elif isinstance(module, (SimpleRNNLayer, LSTMLayer, GRULayer)):
                if module.input_size is None:
                    raise ValueError(
                        f"PyTorch RNN layer '{module.name}' requires "
                        "'input_size' to be set. Cannot generate code "
                        "with input_size=None."
                    )
