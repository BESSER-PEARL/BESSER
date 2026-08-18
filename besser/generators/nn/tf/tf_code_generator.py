"""
This module defines the `TFGenerator` class that generates
TF code for neural networks based on the B-UML model.
"""

from typing import Callable
from besser.BUML.metamodel.nn import NN
from besser.generators.nn.tf.utils_tf import SetupLayerSyntax, \
    get_tensorop_syntax
from besser.generators.nn.nn_code_generator import NNCodeGenerator


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

        if generation_type == "sequential":
            self._apply_tf_sequential_constraints()

    def _apply_tf_sequential_constraints(self):
        """Apply TensorFlow-specific sequential mode constraints and transformations."""
        # Block dropout with explicit training_aware=False
        for module in self.model.modules:
            if (hasattr(module, 'tns_type') and module.tns_type == 'dropout' and
                hasattr(module, 'dropout_training_aware') and module.dropout_training_aware is False):
                raise ValueError(
                    "TensorFlow sequential generation does not support dropout tensorops with "
                    "dropout_training_aware=False. Use subclassing mode instead."
                )

        # Wrap certain tensorops in lambda for sequential compatibility
        LAMBDA_WRAP_TYPES = {
            'pad', 'identity', 'reshape', 'permute', 'normalize', 'interpolate',
            'transpose', 'squeeze', 'unsqueeze', 'mean', 'max', 'repeat',
            'zeros_like', 'shape_dim', 'subscript', 'split'
        }

        # Track previous module's output variable for sequential flow
        prev_out_var = None

        for module_name, module_details in self.modules_details.items():
            if module_name.endswith('_op'):
                # Find the corresponding module
                module = next((m for m in self.model.modules if f"{m.name}_op" == module_name), None)
                if not module:
                    continue

                needs_lambda = False

                # Block split with split_sizes != 1 (multi-output breaks sequential API)
                if hasattr(module, 'tns_type') and module.tns_type == 'split':
                    split_sizes = getattr(module, 'split_sizes', None)
                    if split_sizes != 1:
                        raise ValueError(
                            f"TensorFlow sequential generation does not support split tensorop '{module.name}' "
                            f"with split_sizes={split_sizes}. Only split_sizes=1 is allowed. "
                            f"Use subclassing mode instead."
                        )

                # Check if it's dropout with training_aware=True (needs lambda to wrap and remove training param)
                if hasattr(module, 'tns_type') and module.tns_type == 'dropout':
                    if hasattr(module, 'dropout_training_aware') and module.dropout_training_aware is True:
                        needs_lambda = True

                # Check if it's a type that needs lambda wrapping
                elif hasattr(module, 'tns_type') and module.tns_type in LAMBDA_WRAP_TYPES:
                    needs_lambda = True

                # Check if it's a binary op (binop_X) with tensor + scalar
                elif hasattr(module, 'tns_type') and module.tns_type.startswith('binop_'):
                    if hasattr(module, 'layers_of_tensors') and module.layers_of_tensors:
                        has_string = any(isinstance(x, str) for x in module.layers_of_tensors)
                        has_scalar = any(not isinstance(x, str) for x in module.layers_of_tensors)
                        if has_string and has_scalar:
                            needs_lambda = True

                # Check if it's multiply with tensor + scalar
                elif hasattr(module, 'tns_type') and module.tns_type == 'multiply':
                    if hasattr(module, 'layers_of_tensors') and module.layers_of_tensors:
                        has_string = any(isinstance(x, str) for x in module.layers_of_tensors)
                        has_scalar = any(not isinstance(x, str) for x in module.layers_of_tensors)
                        if has_string and has_scalar:
                            needs_lambda = True

                # Apply lambda wrapping to the syntax
                if needs_lambda and module_details and prev_out_var:
                    import re
                    original_syntax = module_details[0]

                    # Extract variable from syntax
                    if hasattr(module, 'tns_type') and module.tns_type == 'dropout':
                        # Dropout: variable is in second parens: Dropout(rate)(var)
                        match = re.search(r'\)\(([^,)]+)', original_syntax)
                        var_to_replace = match.group(1).strip() if match else str(prev_out_var)
                    else:
                        # Other ops: variable is first arg: func(var, ...)
                        match = re.search(r'\(([^,)]+)', original_syntax)
                        var_to_replace = match.group(1).strip() if match else str(prev_out_var)

                    updated_syntax = re.sub(r'\b' + re.escape(var_to_replace) + r'\b', 'x', original_syntax)

                    if hasattr(module, 'tns_type') and module.tns_type == 'dropout':
                        updated_syntax = updated_syntax.replace(', training=training', '')

                    module_details[0] = f"lambda x: {updated_syntax}"

            # Update prev_out_var with current module's output
            if module_details:
                # Check if this is a split operation with multiple outputs
                if module_name.endswith('_op'):
                    module_obj = module_details[2] if len(module_details) > 2 else None
                    if module_obj and hasattr(module_obj, 'tns_type') and module_obj.tns_type == 'split':
                        if hasattr(module_obj, 'output_vars') and module_obj.output_vars:
                            # Track split output variables
                            prev_out_var = module_obj.output_vars[0] if len(module_obj.output_vars) == 1 else module_obj.output_vars
                        else:
                            prev_out_var = module_details[1]
                    else:
                        prev_out_var = module_details[1]
                else:
                    prev_out_var = module_details[1]  # out_var is second element
