"""
This module provides the `SetupLayerSyntax` class and the
`get_tensorop_syntax` function.
The `SetupLayerSyntax` class is used to define the syntax
of layers in TensorFlow, while `get_tensorop_syntax` defines the
tensorOps.
"""

from besser.BUML.metamodel.nn import TensorOp, Layer
from besser.generators.nn import utils_nn as utils


class SetupLayerSyntax:
    """
    This class is used to get TensorFlow layer syntax from BUML layer object.
    It processes the layers based on their type.

    Attributes:
        layer (Layer): the BUML layer object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        permute_out (bool | None): Whether to add a permute tensorop after
            the layer. It is only relevant for PyTorch and used here just to
            facilitate shared processing logic.
        permute_in (bool | None): Whether to add a permute tensorop before
            the layer. It is only relevant for PyTorch and used here just to
            facilitate shared processing logic.

    Returns:
        None, but stores the layers and their attributes in the
        modules_details dictionary.
    """
    def __init__(self, layer: Layer, modules_details: dict, is_subnn: bool = False):
        self.layer: Layer = layer
        self.modules_details: dict = modules_details
        self.is_subnn: bool = is_subnn
        self.permute_out: bool | None = None
        self.permute_in: bool | None = None
        # Track shared activation layers
        if not hasattr(self, '_shared_activations'):
            SetupLayerSyntax._shared_activations = {}

    def setup_general_layer(self):
        """It defines the syntax of general layers."""
        actv_func = self.setup_actv_func()
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = layers"
        if cls_name == "LinearLayer":
            use_bias = "True" if self.layer.bias else "False"
            lyr = (
                f"{lyr}.Dense(units={self.layer.out_features}, "
                f"use_bias={use_bias}, activation={actv_func})"
            )
        elif cls_name == "FlattenLayer":
            lyr = f"{lyr}.Flatten()"
        elif cls_name == "GeneralLayer":
            # GeneralLayer with only actv_func (standalone activation)
            lyr = f"self.{lyr_name} = layers.Activation('{self.layer.actv_func}')"
        else: #cls_name == "EmbeddingLayer"
            padding_idx = self.layer.padding_idx
            mask_zero = "True" if padding_idx is not None else "False"

            # If padding_idx != 0, add remapping layer
            if padding_idx is not None and padding_idx != 0:
                remap_lyr = (
                    f"self.{lyr_name}_remap = layers.Lambda("
                    f"lambda x: tf.where(x == {padding_idx}, 0, "
                    f"tf.where(x == 0, {padding_idx}, x)))#"
                )
                lyr = (
                    f"{remap_lyr}{lyr}.Embedding(input_dim={self.layer.num_embeddings}, "
                    f"output_dim={self.layer.embedding_dim}, mask_zero={mask_zero})"
                )
            else:
                lyr = (
                    f"{lyr}.Embedding(input_dim={self.layer.num_embeddings}, "
                    f"output_dim={self.layer.embedding_dim}, mask_zero={mask_zero})"
                )
        return lyr

    def setup_standalone_activation(self, out_var, in_var):
        """It defines the syntax for standalone activation layer."""
        actv_func = self.layer.actv_func
        lyr_name = self.layer.name

        # Use the actual layer name to preserve original PyTorch names
        syntax = f"self.{lyr_name} = layers.Activation('{actv_func}')"

        # Store in modules_details with _activ suffix so it's recognized as activation
        self.modules_details[lyr_name + "_activ"] = [syntax, out_var, in_var]

        # Return None to signal handle_layer not to add _layer entry
        return None

    def _get_permute_dims_tf(self, dim: str, permute_in: bool):
        """Calculate permute dimensions for TensorFlow transpose."""
        if dim is None or dim == "1":
            return [0, 2, 1]
        elif dim == "2":
            return [0, 3, 1, 2] if permute_in else [0, 2, 3, 1]
        else:
            return [0, 4, 1, 2, 3] if permute_in else [0, 2, 3, 4, 1]

    def add_permute(self, lyr_name: str, dim: str, in_var_layer: str,
                    permute_in: bool = True, sequential: bool = False,
                    is_subnn: bool = False):
        """
        It adds transpose operation for CNN layers that need permutation.

        Args:
            lyr_name (str): the name of the layer.
            dim (str): the dimensionality of the layer ('1', '2' or '3').
            in_var_layer (str): the input variable notation of the layer.
            permute_in (bool): Whether to permute the input of the layer.
            sequential (bool): Whether the layer is in a seq architecture.
            is_subnn (bool): Whether the layer is in a subnn model.

        Returns:
            None, but stores the transpose tensorop in modules_details.
        """
        perm_name = f"{lyr_name}_{'in' if permute_in else 'out'}_op"
        perm_dim = self._get_permute_dims_tf(dim, permute_in)
        perm_str = ', '.join(map(str, perm_dim))
        transpose_syntax = f"tf.transpose({in_var_layer}, perm=[{perm_str}])"
        self.modules_details[perm_name] = [transpose_syntax, in_var_layer]

    def _build_batchnorm_tf(self, lyr_name: str):
        """Build BatchNormalization syntax with PyTorch to TF param conversion."""
        epsilon = self.layer.eps
        momentum = 1 - self.layer.momentum  # INVERT: PyTorch 0.1 -> TF 0.9
        center = "True" if self.layer.affine else "False"
        scale = "True" if self.layer.affine else "False"

        # Warn if track_running_stats=False (TF always tracks)
        if not self.layer.track_running_stats:
            print(f"Warning: BatchNorm '{lyr_name}' has track_running_stats=False. "
                  f"TensorFlow will use running statistics during inference, "
                  f"which differs from PyTorch behavior (batch stats).")

        return (
            f"self.{lyr_name} = layers.BatchNormalization(epsilon={epsilon}, "
            f"momentum={momentum}, center={center}, scale={scale})"
        )

    def _build_layernorm_tf(self, lyr_name: str):
        """Build LayerNormalization syntax with PyTorch to TF axis conversion."""
        norm_shape = self.layer.normalized_shape
        epsilon = self.layer.eps
        center = "True" if self.layer.affine else "False"
        scale = "True" if self.layer.affine else "False"

        if isinstance(norm_shape, list):
            num_axes = len(norm_shape)
            axis_indices = list(range(-num_axes, 0))
            return (
                f"self.{lyr_name} = layers.LayerNormalization(axis={axis_indices}, "
                f"epsilon={epsilon}, center={center}, scale={scale})"
            )
        else:
            return (
                f"self.{lyr_name} = layers.LayerNormalization(axis=-1, epsilon={epsilon}, "
                f"center={center}, scale={scale})"
            )

    def setup_layer_modifier(self):
        """It defines the syntax of layers' modifiers."""
        cls_name = self.layer.__class__.__name__
        parent_cls = self.layer.__class__.mro()[1].__name__
        lyr_name = self.layer.name

        if parent_cls == "NormalizationLayer":
            if cls_name == "BatchNormLayer":
                return self._build_batchnorm_tf(lyr_name)
            else: # cls_name == "LayerNormLayer"
                return self._build_layernorm_tf(lyr_name)
        else: # cls_name == "DropoutLayer"
            if hasattr(self.layer, 'dimension') and self.layer.dimension:
                return f"self.{lyr_name} = layers.SpatialDropout{self.layer.dimension}D(rate={self.layer.rate})"
            else:
                return f"self.{lyr_name} = layers.Dropout(rate={self.layer.rate})"

    def add_separate_activation_if_needed(self, out_var, in_var):
        """Add separate activation for layers that don't support activation param."""
        cls_name = self.layer.__class__.__name__
        parent_cls = self.layer.__class__.mro()[1].__name__

        # BatchNorm, LayerNorm, Dropout, Embedding, Pooling don't support activation
        unsupported = parent_cls in ["NormalizationLayer", "LayerModifier"] or \
                     cls_name in ["EmbeddingLayer", "PoolingLayer", "FlattenLayer"]

        if unsupported and self.layer.actv_func:
            actv_func = self.layer.actv_func

            # Use shared activation layer
            if actv_func not in SetupLayerSyntax._shared_activations:
                shared_name = f"activation_{actv_func}"
                SetupLayerSyntax._shared_activations[actv_func] = shared_name
                # Add shared activation definition (DEF: to skip forward generation)
                syntax = f"self.{shared_name} = layers.Activation('{actv_func}')"
                self.modules_details[shared_name + "_activ"] = [f"DEF:{syntax}", None, None]

            # Add call reference for this layer
            shared_name = SetupLayerSyntax._shared_activations[actv_func]
            lyr_name = self.layer.name
            actv_call_key = f"{lyr_name}_activ"
            # Use CALL: prefix to indicate this is a call to shared activation
            self.modules_details[actv_call_key] = [f"CALL:{shared_name}", out_var, out_var]

    def setup_rnn(self):
        """It defines the syntax of rnn layers."""
        actv_func = self.setup_actv_func()
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        layer_type = cls_name[:-5]
        use_bias_str = "True" if self.layer.bias else "False"
        lyr = (
            f"layers.{layer_type}(units={self.layer.hidden_size}, "
            f"activation={actv_func}, use_bias={use_bias_str}, dropout={self.layer.dropout}"
        )

        if self.layer.return_type == "full":
            lyr = f"{lyr}, return_sequences=True)"
        elif self.layer.return_type == "both":
            lyr = f"{lyr}, return_sequences=True, return_state=True)"
        elif self.layer.return_type == "hidden":
            lyr = f"{lyr}, return_state=True)"
        else:
            lyr = f"{lyr})"

        if self.layer.bidirectional is True:
            lyr = f"self.{lyr_name} = layers.Bidirectional({lyr})"
        else:
            lyr = f"self.{lyr_name} = {lyr}"

        return lyr

    def setup_actv_func(self):
        """
        It formats the activation function as attribute of the layer.
        """
        if hasattr(self.layer, 'actv_func'):
            activ = self.layer.actv_func
            list_func = ["relu", "tanh", "sigmoid", "softmax", "leaky_relu", "gelu"]
            if activ is not None:
                if activ in list_func:
                    return f"'{self.layer.actv_func}'"
                else:
                    return f"{self.layer.actv_func}"
            else:
                # Set default activation for RNN layers
                # PyTorch RNN defaults to tanh, LSTM/GRU also use tanh for cell state
                if self.layer.__class__.__name__ in ['SimpleRNNLayer', 'LSTMLayer', 'GRULayer']:
                    return "'tanh'"
                return None

    def setup_conv(self, lyr_name: str, cls_name: str):
        """
        It defines the syntax of convolutional layers.

        Args:
            lyr_name (str): The name of the layer.
            cls_name (str): The name of its class.

        Returns:
            lyr (str): The syntax of the layer in TensorFlow.
        """
        actv_func = self.setup_actv_func()
        dim = cls_name[-2:-1]
        filters = self.layer.out_channels
        pad_type = self.layer.padding_type
        kernel = utils.format_value(self.layer.kernel_dim)
        stride = utils.format_value(self.layer.stride_dim)
        dilation = utils.format_value(self.layer.dilation)
        groups = self.layer.groups
        use_bias = "True" if self.layer.bias else "False"
        pad_amount = self.layer.padding_amount
        self.permute_in = self.layer.permute_in
        self.permute_out = self.layer.permute_out
        self.dim = dim
        lyr = ""
        if pad_amount != 0:
            lyr = (
                f"self.{lyr_name}_pad = layers.ZeroPadding{dim}D("
                f"padding={pad_amount})#"
            )
        lyr = (
            f"{lyr}self.{lyr_name} = layers.Conv{dim}D(filters={filters}, "
            f"kernel_size={kernel}, strides={stride}, "
            f"padding='{pad_type}', dilation_rate={dilation}, groups={groups}, "
            f"use_bias={use_bias}, activation={actv_func})"
        )
        return lyr

    def _build_standard_pooling_tf(self, lyr_name: str, pl_type: str, dim: str):
        """Build syntax for standard max or average pooling."""
        pl = "MaxPool" if pl_type == "max" else "AveragePooling"
        kernel = utils.format_value(self.layer.kernel_dim)
        stride = utils.format_value(self.layer.stride_dim)
        pad_type = self.layer.padding_type

        # Handle explicit padding with ZeroPadding layer
        pad_amount = self.layer.padding_amount if hasattr(self.layer, 'padding_amount') else 0
        lyr = ""
        if pad_amount != 0:
            lyr = f"self.{lyr_name}_pad = layers.ZeroPadding{dim}D(padding={pad_amount})#"
            pad_type = 'valid'

        lyr += (
            f"self.{lyr_name} = layers.{pl}{dim}D(pool_size={kernel}, "
            f"strides={stride}, padding='{pad_type}')"
        )
        return lyr

    def _build_global_pooling_tf(self, lyr_name: str, pl_type: str, dim: str):
        """Build syntax for global pooling."""
        typ = pl_type.split("_")[1]
        pl = f"Global{typ[0].upper()}{typ[1:]}Pooling"
        return f"self.{lyr_name} = layers.{pl}{dim}D()"

    def _build_adaptive_pooling_tf(self, lyr_name: str, pl_type: str, dim: str):
        """Build syntax for adaptive pooling."""
        pl = "AdaptiveAveragePooling" if pl_type == "adaptive_average" else "AdaptiveMaxPooling"
        size = utils.format_value(self.layer.output_dim)
        return f"self.{lyr_name} = tfa.layers.{pl}{dim}D(output_size={size})"

    def setup_pooling(self, lyr_name: str):
        """
        It defines the syntax of pooling layers.

        Args:
            lyr_name (str): The name of the layer.

        Returns:
            lyr (str): The syntax of the layer in TensorFlow.
        """
        pl_type = self.layer.pooling_type
        dim = self.layer.dimension[-2:-1]
        self.dim = dim
        self.permute_in = self.layer.permute_in
        self.permute_out = self.layer.permute_out

        if pl_type in ("max", "average"):
            return self._build_standard_pooling_tf(lyr_name, pl_type, dim)
        elif pl_type.startswith("global"):
            return self._build_global_pooling_tf(lyr_name, pl_type, dim)
        else:
            return self._build_adaptive_pooling_tf(lyr_name, pl_type, dim)

    def setup_cnn(self):
        """It defines the syntax of cnn layers (conv and pooling)."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name

        if cls_name == "PoolingLayer":
            lyr = self.setup_pooling(lyr_name)
        else:
            lyr = self.setup_conv(lyr_name, cls_name)

        return lyr


def _get_prev_out_var_for_simple_ops(in_var, modules_details):
    """Get previous output variable for operations that don't use layers_of_tensors."""
    if in_var is not None:
        return in_var
    elif len(list(modules_details.keys())) == 0:
        return "x"
    else:
        prev_module = list(modules_details.keys())[-1]
        return utils.get_previous_out_var(modules_details, prev_module)


def _handle_interpolate(tensorop, modules_details, in_var):
    """Handle interpolate tensorop syntax."""
    prev_out_var = _get_prev_out_var_for_simple_ops(in_var, modules_details)
    size = tensorop.interpolate_size
    scale = tensorop.interpolate_scale
    mode = tensorop.interpolate_mode if hasattr(tensorop, 'interpolate_mode') else 'nearest'

    if scale is not None:
        size_expr = (f"[tf.shape({prev_out_var})[1] * {scale}, "
                     f"tf.shape({prev_out_var})[2] * {scale}]")
        return f"tf.image.resize({prev_out_var}, size={size_expr}, method='{mode}')"
    elif size is not None:
        return f"tf.image.resize({prev_out_var}, size={list(size)}, method='{mode}')"
    else:
        return f"tf.image.resize({prev_out_var}, size=tf.shape({prev_out_var})[1:3], method='{mode}')"


def _handle_pad(tensorop, modules_details, in_var):
    """Handle pad tensorop syntax."""
    prev_out_var = _get_prev_out_var_for_simple_ops(in_var, modules_details)
    pad_amount = (tensorop.pad_amount if hasattr(tensorop, 'pad_amount')
                  and tensorop.pad_amount else (0, 0, 0, 0))
    mode = tensorop.pad_mode if hasattr(tensorop, 'pad_mode') else 'constant'

    if len(pad_amount) == 4:
        left, right, top, bottom = pad_amount
        paddings = f"[[0, 0], [{top}, {bottom}], [{left}, {right}], [0, 0]]"
    else:
        paddings = "[[0, 0], [0, 0], [0, 0], [0, 0]]"

    mode_map = {
        'constant': 'CONSTANT',
        'reflect': 'REFLECT',
        'replicate': 'SYMMETRIC'
    }
    tf_mode = mode_map.get(mode, 'CONSTANT')
    return f"tf.pad({prev_out_var}, {paddings}, mode='{tf_mode}')"


def _handle_dropout_syntax(tensorop, modules_details, in_var):
    """Handle dropout tensorop syntax."""
    prev_out_var = _get_prev_out_var_for_simple_ops(in_var, modules_details)
    rate = tensorop.dropout_rate if hasattr(tensorop, 'dropout_rate') else 0.5
    training_aware = getattr(tensorop, 'dropout_training_aware', False)

    if training_aware:
        return f"tf.keras.layers.Dropout({rate})({prev_out_var}, training=training)"
    else:
        return f"tf.nn.dropout({prev_out_var}, rate={rate})"


def _handle_reshape_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle reshape tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"tf.reshape({prev_out_var}, [{params}])"


def _check_all_2d_layers(tensorop, modules_details):
    """Check if all source layers produce 2D output."""
    for source_layer in tensorop.layers_of_tensors:
        if isinstance(source_layer, str) and source_layer + "_layer" in modules_details:
            layer_details = modules_details[source_layer + "_layer"]
            if len(layer_details) > 3 and layer_details[3]:
                layer_obj = layer_details[3]
                class_name = (layer_obj.__class__.__name__
                              if hasattr(layer_obj, '__class__') else '')
                if not any(name in class_name for name in ['Linear', 'Flatten']):
                    return False
        else:
            return False
    return True


def _check_2d_tensor_sources(tensorop, modules_details):
    """Check if any source produces 2D tensors."""
    for source_layer in tensorop.layers_of_tensors:
        if not isinstance(source_layer, str):
            continue

        # Check operations
        if source_layer + "_op" in modules_details:
            op_syntax = (modules_details[source_layer + "_op"][0]
                         if modules_details[source_layer + "_op"] else "")
            patterns = ['squeeze', 'flatten', 'reduce_max', 'reduce_mean', 'amax']
            if any(pattern in op_syntax.lower() for pattern in patterns):
                return True
            if '_subscript_' in source_layer:
                return True

        # Check layers
        elif source_layer + "_layer" in modules_details:
            layer_details = modules_details[source_layer + "_layer"]
            if len(layer_details) > 3 and layer_details[3]:
                layer_obj = layer_details[3]
                if not hasattr(layer_obj, '__class__'):
                    continue

                class_name = layer_obj.__class__.__name__
                # RNN hidden states
                if class_name in ('SimpleRNNLayer', 'LSTMLayer', 'GRULayer'):
                    if (hasattr(layer_obj, 'return_type') and
                            layer_obj.return_type in ('hidden', 'both')):
                        return True
                # FlattenLayer
                elif 'Flatten' in class_name:
                    return True
                # LinearLayer
                elif 'Linear' in class_name:
                    if _check_linear_input_2d(layer_obj, modules_details):
                        return True
    return False


def _check_linear_input_2d(layer_obj, modules_details):
    """Check if LinearLayer input is 2D."""
    if not hasattr(layer_obj, 'name_module_input') or not layer_obj.name_module_input:
        return False

    input_source = layer_obj.name_module_input

    # Check if input is from an op that produces 2D
    if input_source + "_op" in modules_details:
        input_op_syntax = modules_details[input_source + "_op"][0]
        if any(p in input_op_syntax.lower()
               for p in ['reduce', 'squeeze', 'flatten', 'amax']):
            return True

    # Check if input is from RNN hidden state
    if input_source.endswith('__hidden') or input_source.endswith('__cell'):
        return True

    # Check if input is from another 2D-producing layer
    if input_source + "_layer" in modules_details:
        input_layer_details = modules_details[input_source + "_layer"]
        if len(input_layer_details) > 3 and input_layer_details[3]:
            input_layer_obj = input_layer_details[3]
            input_class_name = (input_layer_obj.__class__.__name__
                                if hasattr(input_layer_obj, '__class__') else '')
            if any(name in input_class_name for name in ['Linear', 'Flatten']):
                return True

    return False


def _determine_concat_axis_for_conv(modules_details):
    """Determine concat axis based on Conv layer dimension."""
    for key, val in modules_details.items():
        if isinstance(val, list) and len(val) > 3 and val[3]:
            layer_obj = val[3]
            if hasattr(layer_obj, '__class__'):
                layer_class = layer_obj.__class__.__name__
                if layer_class.startswith('Conv'):
                    if '1D' in layer_class:
                        return 2
                    elif '2D' in layer_class:
                        return 3
                    elif '3D' in layer_class:
                        return 4
    return None


def _handle_concatenate_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle concatenate tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var

    axis = tensorop.concatenate_dim

    if axis == 1 and hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
        if _check_all_2d_layers(tensorop, modules_details):
            return f"tf.concat([{params}], axis=1)"

        is_2d_tensor = _check_2d_tensor_sources(tensorop, modules_details)
        if not is_2d_tensor:
            new_axis = _determine_concat_axis_for_conv(modules_details)
            if new_axis is not None:
                axis = new_axis

    return f"tf.concat([{params}], axis={axis})"


def _infer_tensor_dimensionality(tensorop, modules_details):
    """Infer tensor dimensionality from source layer."""
    num_dims = 3  # default fallback
    if not (hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors):
        return num_dims

    source_layer = tensorop.layers_of_tensors[0]
    if not isinstance(source_layer, str):
        return num_dims

    # Check if source is a layer
    if source_layer + '_layer' in modules_details:
        layer_obj = (modules_details[source_layer + '_layer'][3]
                     if len(modules_details[source_layer + '_layer']) > 3 else None)
        if layer_obj and hasattr(layer_obj, '__class__'):
            layer_class = layer_obj.__class__.__name__
            # Determine dimensionality based on layer type
            if layer_class in ['Dense', 'LinearLayer']:
                has_rnn = any(
                    'RNN' in str(v[3].__class__.__name__)
                    if len(v) > 3 and hasattr(v[3], '__class__') else False
                    for k, v in modules_details.items()
                    if k.endswith('_layer') and isinstance(v, list)
                )
                num_dims = 3 if has_rnn else 2
            elif 'RNN' in layer_class or 'LSTM' in layer_class or 'GRU' in layer_class:
                num_dims = 3 if (hasattr(layer_obj, 'return_sequences') and
                                 layer_obj.return_sequences) else 2
            elif 'Conv1D' in layer_class or 'Embedding' in layer_class:
                num_dims = 3
            elif 'Conv2D' in layer_class:
                num_dims = 4
            elif 'Conv3D' in layer_class:
                num_dims = 5

    # Check if source is an op
    elif source_layer + '_op' in modules_details:
        op_syntax = (modules_details[source_layer + '_op'][0]
                     if modules_details[source_layer + '_op'] else "")
        if 'squeeze' in op_syntax.lower():
            num_dims = 2
        elif 'expand_dims' in op_syntax.lower() or 'unsqueeze' in op_syntax.lower():
            num_dims = 3

    return num_dims


def _handle_transpose_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle transpose tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var

    transpose_dims = tensorop.transpose_dim
    if len(transpose_dims) == 2:
        dim0, dim1 = transpose_dims
        num_dims = _infer_tensor_dimensionality(tensorop, modules_details)
        num_dims = max(num_dims, max(dim0, dim1) + 1)
        perm = list(range(num_dims))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        perm_str = ", ".join(map(str, perm))
        return f"tf.transpose({prev_out_var}, perm=[{perm_str}])"
    else:
        return f"tf.transpose({prev_out_var}, perm=[{params}])"


def _handle_permute_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle permute tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"tf.transpose({prev_out_var}, perm=[{params}])"


def _handle_multiply_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle multiply tensorop syntax."""
    return f"tf.math.multiply({params})"


def _handle_mean_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle mean tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    axis = tensorop.reduce_dim
    return f"tf.reduce_mean({prev_out_var}, axis={axis})"


def _handle_max_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle max tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    axis = tensorop.reduce_dim
    return f"tf.reduce_max({prev_out_var}, axis={axis})"


def _is_rnn_hidden_squeeze(tensorop, modules_details, prev_out_var):
    """Check if this is squeeze(0) on an RNN hidden state."""
    if tensorop.reduce_dim != 0:
        return False
    if not (hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors):
        return False

    source_layer = tensorop.layers_of_tensors[0]
    if not isinstance(source_layer, str):
        return False

    base_layer = source_layer.replace('__hidden', '').replace('__cell', '')
    layer_key = f"{base_layer}_layer"
    if layer_key not in modules_details:
        return False

    layer_details = modules_details[layer_key]
    if len(layer_details) <= 4:
        return False

    layer_obj = layer_details[3] if len(layer_details) > 3 else None
    hidden_var = layer_details[4]

    if (layer_obj and hasattr(layer_obj, '__class__') and
            layer_obj.__class__.__name__ in ['SimpleRNNLayer', 'LSTMLayer', 'GRULayer'] and
            prev_out_var == hidden_var):
        return True

    return False


def _adjust_squeeze_axis_for_pooling(tensorop, modules_details, axis):
    """Adjust squeeze axis for 1D pooling layers."""
    if axis != -1:
        return axis
    if not (hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors):
        return axis

    source_layer = tensorop.layers_of_tensors[0]
    if not isinstance(source_layer, str):
        return axis

    # Check if source is a 1D pooling layer
    if source_layer + "_layer" in modules_details:
        layer_obj = (modules_details[source_layer + "_layer"][3]
                     if len(modules_details[source_layer + "_layer"]) > 3 else None)
        if (layer_obj and hasattr(layer_obj, '__class__') and
                'Pooling' in layer_obj.__class__.__name__):
            if hasattr(layer_obj, 'dimension') and layer_obj.dimension == '1D':
                return 1

    # Check if source is a concat op from 1D pooling
    elif source_layer + "_op" in modules_details:
        op_details = modules_details[source_layer + "_op"]
        if len(op_details) > 0 and isinstance(op_details[0], str):
            if 'tf.concat' in op_details[0]:
                return 1

    return axis


def _handle_squeeze_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle squeeze tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var

    if _is_rnn_hidden_squeeze(tensorop, modules_details, prev_out_var):
        return prev_out_var

    if tensorop.reduce_dim is not None:
        axis = tensorop.reduce_dim
        axis = _adjust_squeeze_axis_for_pooling(tensorop, modules_details, axis)
        return f"tf.squeeze({prev_out_var}, axis={axis})"
    else:
        return f"tf.squeeze({prev_out_var})"


def _handle_unsqueeze_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle unsqueeze tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var

    axis = tensorop.reduce_dim
    if axis == 0 and hasattr(tensorop, 'is_rnn_initial_state') and tensorop.is_rnn_initial_state:
        return f"SKIP:{prev_out_var}"
    else:
        return f"tf.expand_dims({prev_out_var}, axis={axis})"


def _handle_zeros_like_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle zeros_like tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"tf.zeros_like({prev_out_var})"


def _handle_split_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle split tensorop syntax.

    TensorFlow's tf.split returns a list of tensors.
    The outputs will be unpacked in the template based on the assignment.
    """
    if in_var is not None:
        prev_out_var = in_var

    split_dim = tensorop.split_dim if hasattr(tensorop, 'split_dim') else 0
    split_sizes = tensorop.split_sizes if hasattr(tensorop, 'split_sizes') else None

    if split_sizes is None:
        # If no split size specified, return the input unchanged (shouldn't happen)
        return prev_out_var

    # tf.split expects num_or_size_splits and axis
    # If split_sizes is an int, it's the size of each chunk
    # If it's a list, it's the sizes of each chunk
    if isinstance(split_sizes, list):
        # List of sizes: tf.split(x, num_or_size_splits=[32, 32], axis=1)
        sizes_str = str(split_sizes)
        return f"tf.split({prev_out_var}, num_or_size_splits={sizes_str}, axis={split_dim})"
    else:
        # Single size: need to determine number of splits from context
        # For now, assume equal splits with given size
        # This will create: tf.split(x, num_or_size_splits=2, axis=1) if splitting into 2 parts
        return f"tf.split({prev_out_var}, num_or_size_splits={split_sizes}, axis={split_dim})"


def _handle_normalize_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle normalize tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    axis = tensorop.reduce_dim
    return f"tf.nn.l2_normalize({prev_out_var}, axis={axis})"


def _handle_repeat_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle repeat tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"tf.tile({prev_out_var}, [{params}])"


def _handle_binop_add_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle binop_add tensorop syntax."""
    return (f"{params[0]} + {params[1]}" if isinstance(params, list)
            else f"tf.add({params})")


def _handle_binop_subtract_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle binop_subtract tensorop syntax."""
    return (f"{params[0]} - {params[1]}" if isinstance(params, list)
            else f"tf.subtract({params})")


def _handle_binop_multiply_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle binop_multiply tensorop syntax."""
    return (f"{params[0]} * {params[1]}" if isinstance(params, list)
            else f"tf.multiply({params})")


def _handle_binop_divide_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle binop_divide tensorop syntax."""
    return (f"{params[0]} / {params[1]}" if isinstance(params, list)
            else f"tf.divide({params})")


def _handle_binop_floor_divide_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle binop_floor_divide tensorop syntax."""
    return (f"{params[0]} // {params[1]}" if isinstance(params, list)
            else f"tf.math.floordiv({params})")


def _remap_subscript_for_conv(tensorop, modules_details, subscript_pattern):
    """Remap subscript indices for Conv layers (PyTorch to TF axis conversion)."""
    if not (hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors):
        return subscript_pattern

    source_layer = tensorop.layers_of_tensors[0]
    if not isinstance(source_layer, str):
        return subscript_pattern

    if source_layer + "_layer" not in modules_details:
        return subscript_pattern

    layer_obj = (modules_details[source_layer + "_layer"][3]
                 if len(modules_details[source_layer + "_layer"]) > 3 else None)
    if not (layer_obj and hasattr(layer_obj, '__class__')):
        return subscript_pattern

    layer_class = layer_obj.__class__.__name__

    # Conv1D: PyTorch [B,C,L] → TF [B,L,C], remap dims 1↔2
    if layer_class == 'Conv1D' and subscript_pattern.count(',') == 2:
        parts = subscript_pattern.strip('[]').split(',')
        if len(parts) == 3:
            subscript_pattern = f"[{parts[0]},{parts[2]},{parts[1]}]"
    # Conv2D: PyTorch [B,C,H,W] → TF [B,H,W,C], remap dims
    elif layer_class == 'Conv2D' and subscript_pattern.count(',') == 3:
        parts = subscript_pattern.strip('[]').split(',')
        if len(parts) == 4:
            subscript_pattern = f"[{parts[0]},{parts[2]},{parts[3]},{parts[1]}]"

    return subscript_pattern


def _handle_subscript_syntax(tensorop, modules_details, in_var, prev_out_var, params, inputs_outputs=None):
    """Handle subscript tensorop syntax."""
    # Check inputs_outputs to get the actual source variable from original code
    if inputs_outputs and tensorop.name in inputs_outputs:
        source_var = inputs_outputs[tensorop.name][0]
    else:
        # Fallback: use prev_out_var or in_var
        source_var = in_var if in_var is not None else prev_out_var

    subscript_pattern = tensorop.subscript_indices
    subscript_pattern = _remap_subscript_for_conv(tensorop, modules_details, subscript_pattern)
    return f"{source_var}{subscript_pattern}"


def _handle_shape_dim_syntax(tensorop, modules_details, in_var, prev_out_var, params, inputs_outputs=None):
    """Handle shape_dim tensorop syntax."""
    # Check inputs_outputs to get the actual source variable from original code
    if inputs_outputs and tensorop.name in inputs_outputs:
        source_var = inputs_outputs[tensorop.name][0]
    else:
        # Fallback: check layers_of_tensors
        tensors = tensorop.layers_of_tensors
        if isinstance(tensors[0], str):
            if f"{tensors[0]}_layer" in modules_details or f"{tensors[0]}_op" in modules_details:
                source_tensors = utils.get_layers_output_for_tensorops(tensors, modules_details)
                source_var = source_tensors[0]
            else:
                source_var = tensors[0]
        else:
            source_var = tensors[0]

    dim_index = tensorop.reduce_dim
    return f"tf.shape({source_var})[{dim_index}]"


def _handle_identity_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle identity operation to preserve variable assignments."""
    # Identity just assigns the input variable to the output variable
    # Return the input variable directly (e.g., "x_1") so the template generates: output_var = x_1
    if in_var is not None:
        return in_var
    return prev_out_var


def _handle_default_syntax(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle default case (matmul)."""
    return f"tf.matmul({params})"


# Dispatch table for tensorop handlers
_TENSOROP_SYNTAX_HANDLERS = {
    "identity": _handle_identity_syntax,
    "reshape": _handle_reshape_syntax,
    "concatenate": _handle_concatenate_syntax,
    "transpose": _handle_transpose_syntax,
    "permute": _handle_permute_syntax,
    "multiply": _handle_multiply_syntax,
    "mean": _handle_mean_syntax,
    "max": _handle_max_syntax,
    "squeeze": _handle_squeeze_syntax,
    "unsqueeze": _handle_unsqueeze_syntax,
    "zeros_like": _handle_zeros_like_syntax,
    "split": _handle_split_syntax,
    "normalize": _handle_normalize_syntax,
    "repeat": _handle_repeat_syntax,
    "binop_add": _handle_binop_add_syntax,
    "binop_subtract": _handle_binop_subtract_syntax,
    "binop_multiply": _handle_binop_multiply_syntax,
    "binop_divide": _handle_binop_divide_syntax,
    "binop_floor_divide": _handle_binop_floor_divide_syntax,
    "subscript": _handle_subscript_syntax,
    "shape_dim": _handle_shape_dim_syntax,
}


def get_tensorop_syntax(tensorop: TensorOp, modules_details: dict,
                        in_var: str | None = None, inputs_outputs: dict | None = None):
    """
    It defines the syntax of tensorops.

    Parameters:
        tensorop (TensorOp): The TensorOp BUML object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        in_var (str | None): the input variable notation of the tensorop
            (e.g., 'x', 'x_1', ...).
        inputs_outputs (dict | None): Optional dict mapping tensorop names to [input_var, output_var].

    Returns:
        ts_op_synt (str): the syntax of the tensorop in TensorFlow.

    """
    tns_type = tensorop.tns_type

    # Handle operations that don't use layers_of_tensors
    if tns_type == "interpolate":
        return _handle_interpolate(tensorop, modules_details, in_var)
    elif tns_type == "pad":
        return _handle_pad(tensorop, modules_details, in_var)
    elif tns_type == "dropout":
        return _handle_dropout_syntax(tensorop, modules_details, in_var)

    # For other operations, get params
    prev_out_var, params = utils.get_tensorop_params(tensorop,
                                                     modules_details,
                                                     get_rnn_hidden_var,
                                                     inputs_outputs)

    # Dispatch to appropriate handler
    handler = _TENSOROP_SYNTAX_HANDLERS.get(tns_type, _handle_default_syntax)
    # Pass inputs_outputs to shape_dim and subscript handlers so they can use the correct source variable
    if tns_type in ("shape_dim", "subscript"):
        return handler(tensorop, modules_details, in_var, prev_out_var, params, inputs_outputs)
    return handler(tensorop, modules_details, in_var, prev_out_var, params)


def get_rnn_hidden_var(layer_details, base_module):
    """
    Get the correct variable name for RNN hidden state in TensorFlow.

    For both return_type="both" and "hidden": uses index 4 if available, else index 1

    Arguments:
        layer_details: The layer details from modules_details
        base_module: The base module name (e.g., "rnn")

    Returns:
        The variable name to use for the hidden state
    """
    if len(layer_details) > 4:
        return layer_details[4]
    else:
        # Fallback to regular output if hidden var not available
        return layer_details[1]
