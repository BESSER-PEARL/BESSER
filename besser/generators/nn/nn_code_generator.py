"""
This module defines the `NNCodeGenerator` class that is inherited
by `TFGenerator` and `PytorchGenerator` to generates code for neural
networks based on the B-UML model.
"""

import os
import re
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
    add_in_out_var_to_subnn, renumber_tensorop_variables


class NNCodeGenerator(GeneratorInterface):
    """
    NNCodeGenerator is a class that implements the GeneratorInterface
    and is inherited by `TFGenerator` and `PytorchGenerator` to
    generates code for neural networks training and evaluation based
    on the B-UML model.

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
            template `template_name` is stored. Either `tf`
            or `pytorch`.
        generation_type (str): 'subclassing' or 'sequential'
        channel_last (bool, optional): If true, PyTorch conv layers
            will have their input and output permuted to match
            TF convention.
        modules_details (dict): A dict storing the NN modules syntax
            and attributes.

    """
    def __init__(self, model: NN,
                 setup_layer: SetupLayerTF | SetupLayerTorch,
                 get_tensorop_syntax: Callable, generation_type: str,
                 template_dir: str, channel_last: bool | None = None,
                 file_name: str = "nn.py", output_dir: str = None,
                 strip_layer_counter_suffix: bool = False):

        super().__init__(model, output_dir)
        self.setup_layer: SetupLayerTF | SetupLayerTorch = setup_layer
        self.get_tensorop_syntax: Callable = get_tensorop_syntax
        self.generation_type: str = generation_type
        self.channel_last: bool = channel_last
        self.template_dir: str = template_dir
        self.file_name: str = file_name
        # Flag to control stripping of counter suffix (_N) from
        # layer names. Set to True for migration (where parser adds
        # suffixes), False for direct BUML generation
        self.strip_layer_counter_suffix: bool = strip_layer_counter_suffix

        if self.generation_type == "subclassing":
            self.template_name = f"template_{template_dir}_subclassing.py.j2"
        else:
            self.template_name = f"template_{template_dir}_sequential.py.j2"
            self._validate_sequential_flow()

        self.modules_details: dict = self.get_modules_details()
        self.has_training_aware_dropout: bool = (
            self._check_training_aware_dropout()
        )

        if self.generation_type == "sequential":
            LAMBDA_WRAP_TYPES = {
                'pad', 'identity', 'reshape', 'permute', 'normalize',
                'interpolate', 'transpose', 'squeeze', 'unsqueeze', 'mean',
                'max', 'repeat', 'zeros_like', 'shape_dim', 'subscript',
                'split', 'dropout'
            }
            self._apply_sequential_constraints(LAMBDA_WRAP_TYPES)

    def _validate_sequential_flow(self):
        """Validate that model has linear flow suitable for sequential
        generation."""
        if not self.model.modules:
            return

        module_names = [mod.name for mod in self.model.modules]
        output_map = {}

        # Build output variable map for input_var resolution
        for i, module in enumerate(self.model.modules):
            if hasattr(module, 'output_var') and module.output_var:
                output_map[module.output_var] = module.name
            else:
                # Default output variable pattern
                default_var = f"x_{i}" if i > 0 else "x"
                output_map[default_var] = module.name

        errors = []

        # Validate each module consumes from exactly the previous one
        for i in range(1, len(self.model.modules)):
            module = self.model.modules[i]
            prev_name = module_names[i - 1]
            module_type = module.__class__.__name__

            if module_type == "TensorOp":
                # Check TensorOp input sources
                if getattr(module, 'layers_of_tensors', None):
                    # Filter to string refs only (exclude scalars)
                    string_refs = [
                        x for x in module.layers_of_tensors
                        if isinstance(x, str)
                    ]

                    # Validate all string refs are real module names
                    invalid = [x for x in string_refs if x not in module_names]
                    if invalid:
                        raise ValueError(
                            f"TensorOp '{module.name}' references "
                            f"non-existent modules: {invalid}"
                        )

                    # Check sequential constraint
                    if len(string_refs) == 0:
                        errors.append(
                            f"TensorOp '{module.name}' at position {i} "
                            "has no module inputs (only scalars or empty), "
                            "breaks sequential flow"
                        )
                    elif len(string_refs) > 1:
                        errors.append(
                            f"TensorOp '{module.name}' at position {i} "
                            "consumes from multiple modules "
                            f"{string_refs}, breaks sequential flow"
                        )
                    elif string_refs[0] != prev_name:
                        errors.append(
                            f"TensorOp '{module.name}' at position {i} "
                            "consumes from non-adjacent module "
                            f"'{string_refs[0]}', breaks sequential flow"
                        )
                elif hasattr(module, 'input_var') and module.input_var:
                    # Resolve input_var to module name
                    source = output_map.get(module.input_var)
                    if source != prev_name:
                        errors.append(
                            f"TensorOp '{module.name}' at position {i} "
                            f"consumes from non-adjacent module '{source}' "
                            "(via input_var), breaks sequential flow"
                        )
                # Else: auto-consumes previous (sequential)

            else:
                # Layer validation
                if getattr(module, 'name_module_input', None):
                    if module.name_module_input != prev_name:
                        errors.append(
                            f"Layer '{module.name}' at position {i} consumes "
                            "from non-adjacent module "
                            f"'{module.name_module_input}', breaks "
                            "sequential flow"
                        )
                elif hasattr(module, 'input_var') and module.input_var:
                    source = output_map.get(module.input_var)
                    if source != prev_name:
                        errors.append(
                            f"Layer '{module.name}' at position {i} consumes "
                            "from non-adjacent module '{source}' "
                            f"(via input_var), breaks sequential flow"
                        )
                # Else: auto-consumes previous (sequential)

        if errors:
            raise ValueError(
                f"Sequential generation requires linear flow. Violations:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    def _apply_sequential_constraints(self, lambda_wrap_types):
        """Apply sequential mode constraints and transformations."""
        # Track previous module's output variable for sequential flow
        prev_out_var = None
        any_lambda_used = False

        for module_name, module_details in self.modules_details.items():
            if module_name.endswith('_op'):
                module = next(
                    (m for m in self.model.modules
                     if f"{m.name}_op" == module_name),
                    None
                )
                if not module:
                    continue

                needs_lambda = False

                # Block split with split_sizes != 1
                # (multi-output breaks sequential API)
                if hasattr(module, 'tns_type') and module.tns_type == 'split':
                    split_sizes = getattr(module, 'split_sizes', None)
                    if split_sizes != 1:
                        raise ValueError(
                            f"Sequential generation does not support split "
                            f"tensorop '{module.name}' with "
                            f"split_sizes={split_sizes}. Only split_sizes=1 "
                            "is allowed. Use subclassing mode instead."
                        )

                # Check if it's a type that needs lambda wrapping
                t_type = getattr(module, 'tns_type', None)
                if (t_type and module.tns_type in lambda_wrap_types):
                    # Framework-specific dropout validation
                    if module.tns_type == 'dropout':
                        self._validate_dropout_for_sequential(module)
                    needs_lambda = True

                # Check if it's a binary op with tensor + scalar
                elif (t_type and module.tns_type.startswith('binop_')):
                    if getattr(module, 'layers_of_tensors', None):
                        lot = module.layers_of_tensors
                        has_string = any(isinstance(x, str) for x in lot)
                        has_scalar = any(not isinstance(x, str) for x in lot)
                        if has_string and has_scalar:
                            needs_lambda = True

                # Check if it's multiply with tensor + scalar
                elif t_type and module.tns_type == 'multiply':
                    if getattr(module, 'layers_of_tensors', None):
                        lot = module.layers_of_tensors
                        has_string = any(isinstance(x, str) for x in lot)
                        has_scalar = any(not isinstance(x, str) for x in lot)
                        if has_string and has_scalar:
                            needs_lambda = True

                # Apply lambda wrapping to the syntax
                if needs_lambda and module_details and prev_out_var:
                    original_syntax = module_details[0]
                    updated_syntax = self._cleanup_lambda_syntax(
                        module, original_syntax, prev_out_var
                    )
                    module_details[0] = self._wrap_in_lambda(updated_syntax)
                    any_lambda_used = True

            # Update prev_out_var with current module's output
            if module_details:
                if module_name.endswith('_op'):
                    module_obj = (
                        module_details[2]
                        if len(module_details) > 2 else None
                    )
                    if (
                        module_obj
                        and t_type
                        and module_obj.tns_type == 'split'
                    ):
                        if getattr(module_obj, 'output_vars', None):
                            ov = module_obj.output_vars
                            prev_out_var = ov[0] if len(ov) == 1 else ov
                        else:
                            prev_out_var = module_details[1]
                    else:
                        prev_out_var = module_details[1]
                else:
                    prev_out_var = module_details[1]

        # Store flag for template to conditionally
        # generate Lambda class
        self.modules_details['_lambda_needed'] = any_lambda_used


    def _validate_dropout_for_sequential(self, module):
        """Hook for framework-specific dropout validation
        in sequential mode."""
        pass

    def _cleanup_lambda_syntax(self, module, syntax, prev_out_var):
        """Hook for framework-specific variable extraction, mapping,
        and syntax cleanup."""
        return syntax

    def _wrap_in_lambda(self, syntax):
        """Hook for framework-specific lambda wrapping."""
        return f"lambda x: {syntax}"

    def _validate_multi_input_var(self):
        """Validate that modules don't use multi-variable input_var
        unless they support it."""
        multi_input_allowed = {
            'concatenate', 'binop_add', 'binop_subtract', 'binop_divide',
            'binop_multiply', 'binop_floor_divide', 'multiply', 'matmultiply'
        }

        errors = []
        for module in self.model.modules:
            module_type = module.__class__.__name__
            current_input_var = getattr(module, 'input_var', None)

            if current_input_var and ', ' in current_input_var:
                if module_type == "TensorOp":
                    tns_type = getattr(module, 'tns_type', None)
                    if tns_type not in multi_input_allowed:
                        errors.append(
                            f"TensorOp '{module.name}' (type: {tns_type}) "
                            f"has input_var '{current_input_var}' "
                            "with multiple variables, not supported "
                            "for this operation. Specify a single variable."
                        )
                else:
                    errors.append(
                        f"{module_type} '{module.name}' has input_var "
                        f"'{current_input_var}' with multiple variables, "
                        "which is not supported. Specify a single variable."
                    )

        if errors:
            raise ValueError(
                "Invalid multi-variable input_var:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    def _check_training_aware_dropout(self):
        """Check if model has any dropout tensorop with
        training_aware=True."""
        for module in self.model.modules:
            if module.__class__.__name__ == "TensorOp":
                if (
                    hasattr(module, 'tns_type')
                    and module.tns_type == 'dropout'
                    and getattr(module, 'dropout_training_aware', False)
                ):
                    return True
        return False

    def _detect_module_reuse(self):
        """First pass: detect tensor value reuse (branching)
        by counting module usage."""
        module_usage_count = {}
        referenced_tensorops = set()

        for module in self.model.modules:
            if module.__class__.__name__ == "TensorOp":
                if getattr(module, 'layers_of_tensors', None):
                    for item in module.layers_of_tensors:
                        if isinstance(item, str):
                            cnt = module_usage_count.get(item, 0) + 1
                            module_usage_count[item] = cnt
                            if item.startswith("op_"):
                                referenced_tensorops.add(item)

        return module_usage_count, referenced_tensorops

    def _mark_tensorops_with_reused_inputs(self, module_usage_count):
        """Second pass: mark tensorops using multi-use inputs with
        input_reused=True."""
        for module in self.model.modules:
            if module.__class__.__name__ == "TensorOp":
                if getattr(module, 'layers_of_tensors', None):
                    for input_module in module.layers_of_tensors:
                        if (
                            isinstance(input_module, str)
                            and module_usage_count.get(input_module, 0) > 1
                        ):
                            if (
                                not hasattr(module, 'input_reused')
                                or not module.input_reused
                            ):
                                module.input_reused = True
                            break

    def _process_module(self, module, modules_details, actv_func, is_seq,
                        counter_subnn, referenced_tensorops):
        """Process a single module (NN, Layer, or TensorOp)
        and update modules_details."""
        module_type = module.__class__.__name__

        if module_type == "NN":
            subnn_details = {}
            for sub_nn_layer in module.layers:
                handle_layer(
                    sub_nn_layer, self.setup_layer, subnn_details,
                    self.channel_last, actv_func, is_seq, is_subnn=True,
                    model=self.model,
                    strip_counter_suffix=self.strip_layer_counter_suffix
                )
            # Use original module name + _nn (no counter
            # needed with original names)
            name_sub_nn = f"{module.name}_nn"
            modules_details[name_sub_nn] = subnn_details
            add_in_out_var_to_subnn(modules_details, subnn_obj=module)
            return counter_subnn

        elif module_type != "TensorOp":
            handle_layer(
                module, self.setup_layer, modules_details,
                self.channel_last, actv_func, is_seq, is_subnn=False,
                model=self.model,
                strip_counter_suffix=self.strip_layer_counter_suffix
            )
            if getattr(module, 'inline_only', False):
                layer_key = f"{module.name}_layer"
                if layer_key in modules_details:
                    modules_details[layer_key].append('INLINE_ONLY')
            return counter_subnn

        else:
            handle_tensorop(
                module, modules_details, self.get_tensorop_syntax,
                referenced_tensorops=referenced_tensorops,
                channel_last=self.channel_last
            )
            return counter_subnn

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
        modules_details: dict = {}
        actv_func = "torch" in self.template_name
        is_seq = self.generation_type == "sequential"

        module_usage_count, referenced_tensorops = self._detect_module_reuse()
        self._mark_tensorops_with_reused_inputs(module_usage_count)
        self._validate_multi_input_var()

        counter_subnn = 0
        for module in self.model.modules:
            counter_subnn = self._process_module(
                module, modules_details, actv_func, is_seq, counter_subnn,
                referenced_tensorops
            )

        if actv_func:
            adjust_actv_func_name(modules_details)

        renumber_tensorop_variables(modules_details)

        return modules_details

    def _clean_generated_code(self, code: str) -> str:
        """Clean excessive blank lines from generated code.
        Remove all blank lines from forward/call methods and normalize
        spacing throughout the file."""

        # First pass: clean forward/call methods (subclassing only)
        pattern = (
            r'(    def (?:forward|call)\([^)]*\):.*?\n)'
            r'((?:.*?\n)*?)('
            r'        return .*?\n)'
            r'((?=    def |def |class |\nif __name__|$))'
        )
        def remove_blank_lines(match):
            header = match.group(1)
            body = match.group(2)
            return_line = match.group(3)
            next_section = match.group(4)

            lines = body.split('\n')
            cleaned_lines = [line for line in lines if line.strip() != '']
            cleaned_body = '\n'.join(cleaned_lines)
            if cleaned_body:
                cleaned_body += '\n'

            return header + cleaned_body + return_line + '\n\n' + next_section

        code = re.sub(
            pattern, remove_blank_lines, code, flags=re.DOTALL | re.MULTILINE
        )

        # Second pass: normalize excessive blank lines globally (both modes)
        code = re.sub(r'\n\n\n+', '\n\n', code)

        # Clean up leading/trailing blank lines
        code = code.strip() + '\n'

        return code

    def generate(self, *args):
        """
        Generates NN code based on the provided B-UML model and saves
        it to the specified output directory.
        If the output directory was not specified, the code generated
        will be stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file
                named nn_code.py
        """
        file_name = f"{self.file_name[:-3]}_{self.generation_type}.py"
        file_path = self.build_generation_path(file_name=file_name)
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), self.template_dir, "templates")
        env = Environment(loader=FileSystemLoader(templates_path))

        # Add custom filter to strip counter suffix
        def strip_counter_suffix_filter(name):
            """Strip trailing _cN where N is one or more digits 
            (counter added by parser)."""
            return re.sub(r'_c\d+$', '', name)
        env.filters['strip_counter_suffix'] = strip_counter_suffix_filter

        template = env.get_template(self.template_name)

        generated_code = template.render(
            model=self.model, modules_details=self.modules_details,
            generation_type=self.generation_type,
            strip_layer_counter_suffix=self.strip_layer_counter_suffix,
            has_training_aware_dropout=self.has_training_aware_dropout)

        # Post-process: clean excessive blank lines
        generated_code = self._clean_generated_code(generated_code)

        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
