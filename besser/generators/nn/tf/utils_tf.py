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

        # Inside Sequential blocks, use direct layer call instead of shared activation
        if self.is_subnn:
            # Direct layer definition for use in Sequential
            return f"self.{lyr_name} = layers.Activation('{actv_func}')"

        # For standalone activations outside Sequential, use shared activation pattern
        if actv_func not in SetupLayerSyntax._shared_activations:
            shared_name = f"activation_{actv_func}"
            SetupLayerSyntax._shared_activations[actv_func] = shared_name
            # Add shared activation definition (DEF: to skip forward generation)
            syntax = f"self.{shared_name} = layers.Activation('{actv_func}')"
            self.modules_details[shared_name + "_activ"] = [f"DEF:{syntax}", None, None]

        # Add call entry for this specific activation
        shared_name = SetupLayerSyntax._shared_activations[actv_func]
        actv_call_key = f"{lyr_name}_activ"
        self.modules_details[actv_call_key] = [f"CALL:{shared_name}", out_var, in_var]

        # Return None to signal handle_layer not to add _layer entry
        return None

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
        if permute_in:
            perm_name = f"{lyr_name}_in_op"
        else:
            perm_name = f"{lyr_name}_out_op"

        if dim is None:
            perm_dim = [0, 2, 1]
        else:
            if dim == "1":
                perm_dim = [0, 2, 1]
            elif dim == "2":
                if permute_in:
                    perm_dim = [0, 3, 1, 2]
                else:
                    perm_dim = [0, 2, 3, 1]
            else:
                if permute_in:
                    perm_dim = [0, 4, 1, 2, 3]
                else:
                    perm_dim = [0, 2, 3, 4, 1]

        perm_str = ', '.join(map(str, perm_dim))
        transpose_syntax = f"tf.transpose({in_var_layer}, perm=[{perm_str}])"

        self.modules_details[perm_name] = [transpose_syntax, in_var_layer]

    def setup_layer_modifier(self):
        """It defines the syntax of layers' modifiers."""
        cls_name = self.layer.__class__.__name__
        parent_cls = self.layer.__class__.mro()[1].__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = layers"
        if parent_cls == "NormalizationLayer":
            if cls_name == "BatchNormLayer":
                # Convert PyTorch BatchNorm params to TensorFlow
                epsilon = self.layer.eps  # Direct mapping
                momentum = 1 - self.layer.momentum  # INVERT: PyTorch 0.1 -> TF 0.9
                center = "True" if self.layer.affine else "False"  # Learn beta
                scale = "True" if self.layer.affine else "False"   # Learn gamma

                # Warn if track_running_stats=False (TF always tracks)
                if not self.layer.track_running_stats:
                    print(f"Warning: BatchNorm '{lyr_name}' has track_running_stats=False. "
                          f"TensorFlow will use running statistics during inference, "
                          f"which differs from PyTorch behavior (batch stats).")

                lyr = (
                    f"{lyr}.BatchNormalization(epsilon={epsilon}, momentum={momentum}, "
                    f"center={center}, scale={scale})"
                )
            else: #cls_name == "LayerNormLayer"
                # PyTorch LayerNorm(shape) vs TensorFlow LayerNormalization(axis):
                # PyTorch takes dimension size, TF takes axis index
                # nn.LayerNorm([d1, d2, ..., dn]) normalizes over last n dimensions
                # LayerNormalization should use axis=[-n, ..., -1]
                norm_shape = self.layer.normalized_shape
                epsilon = self.layer.eps  # Direct mapping
                center = "True" if self.layer.affine else "False"  # Learn beta
                scale = "True" if self.layer.affine else "False"   # Learn gamma

                if isinstance(norm_shape, list):
                    num_axes = len(norm_shape)
                    axis_indices = list(range(-num_axes, 0))
                    lyr = (
                        f"{lyr}.LayerNormalization(axis={axis_indices}, epsilon={epsilon}, "
                        f"center={center}, scale={scale})"
                    )
                else:
                    lyr = (
                        f"{lyr}.LayerNormalization(axis=-1, epsilon={epsilon}, "
                        f"center={center}, scale={scale})"
                    )
        else: #cls_name == "DropoutLayer"
            # Use SpatialDropout for 1D/2D/3D variants, regular Dropout otherwise
            if hasattr(self.layer, 'dimension') and self.layer.dimension:
                lyr = f"{lyr}.SpatialDropout{self.layer.dimension}D(rate={self.layer.rate})"
            else:
                lyr = f"{lyr}.Dropout(rate={self.layer.rate})"
        return lyr

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

        if pl_type == "max" or pl_type == "average":
            pl = "MaxPool" if pl_type == "max" else "AveragePooling"
            kernel = utils.format_value(self.layer.kernel_dim)
            stride = utils.format_value(self.layer.stride_dim)
            pad_type = self.layer.padding_type

            # Handle explicit padding with ZeroPadding layer
            pad_amount = self.layer.padding_amount if hasattr(self.layer, 'padding_amount') else 0
            lyr = ""
            if pad_amount != 0:
                lyr = (
                    f"self.{lyr_name}_pad = layers.ZeroPadding{dim}D("
                    f"padding={pad_amount})#"
                )
                # When using explicit padding, pooling layer should use 'valid'
                pad_type = 'valid'

            lyr += (
                f"self.{lyr_name} = layers.{pl}{dim}D(pool_size={kernel}, "
                f"strides={stride}, padding='{pad_type}')"
            )
        elif pl_type.startswith("global"):
            typ = pl_type.split("_")[1]
            pl = f"Global{typ[0].upper()}{typ[1:]}Pooling"
            lyr = (
                f"self.{lyr_name} = layers.{pl}{dim}D()"
            )
        else:
            if pl_type == "adaptive_average":
                pl = "AdaptiveAveragePooling"
            else:
                pl = "AdaptiveMaxPooling"

            size = utils.format_value(self.layer.output_dim)
            lyr = (
                f"self.{lyr_name} = tfa.layers.{pl}{dim}D(output_size={size})"
            )
        return lyr

    def setup_cnn(self):
        """It defines the syntax of cnn layers (conv and pooling)."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name

        if cls_name == "PoolingLayer":
            lyr = self.setup_pooling(lyr_name)
        else:
            lyr = self.setup_conv(lyr_name, cls_name)

        return lyr


def get_tensorop_syntax(tensorop: TensorOp, modules_details: dict,
                        in_var: str | None = None):
    """
    It defines the syntax of tensorops.

    Parameters:
        tensorop (TensorOp): The TensorOp BUML object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        in_var (str | None): the input variable notation of the tensorop
            (e.g., 'x', 'x_1', ...).

    Returns:
        ts_op_synt (str): the syntax of the tensorop in PyTorch.

    """
    # Check if this is a no-op tensorop (e.g., squeeze on RNN hidden marked during parsing)
    if hasattr(tensorop, 'is_noop') and tensorop.is_noop:
        # Return just the variable name for assignment (e.g., h_squeezed = x_1_h)
        if in_var is not None:
            return in_var
        # Get the source variable from layers_of_tensors
        if hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
            source_layers = tensorop.layers_of_tensors
            prev_out_var_list = utils.get_layers_output_for_tensorops(source_layers, modules_details, None)
            if prev_out_var_list and len(prev_out_var_list) > 0:
                return prev_out_var_list[0]
        # Fallback to getting prev var from modules
        if len(list(modules_details.keys())) == 0:
            return "x"
        prev_module = list(modules_details.keys())[-1]
        return utils.get_previous_out_var(modules_details, prev_module)

    tns_type = tensorop.tns_type

    # Handle operations that don't use layers_of_tensors first
    if tns_type in ["interpolate", "pad", "dropout"]:
        if in_var is not None:
            prev_out_var = in_var
        elif len(list(modules_details.keys())) == 0:
            prev_out_var = "x"
        else:
            prev_module = list(modules_details.keys())[-1]
            prev_out_var = utils.get_previous_out_var(modules_details, prev_module)

        if tns_type == "interpolate":
            # F.interpolate upsampling/downsampling
            size = tensorop.interpolate_size
            scale = tensorop.interpolate_scale
            mode = tensorop.interpolate_mode if hasattr(tensorop, 'interpolate_mode') else 'nearest'

            if scale is not None:
                # Use scale_factor
                ts_op_synt = f"tf.image.resize({prev_out_var}, size=[tf.shape({prev_out_var})[1] * {scale}, tf.shape({prev_out_var})[2] * {scale}], method='{mode}')"
            elif size is not None:
                # Use explicit size
                ts_op_synt = f"tf.image.resize({prev_out_var}, size={list(size)}, method='{mode}')"
            else:
                ts_op_synt = f"tf.image.resize({prev_out_var}, size=tf.shape({prev_out_var})[1:3], method='{mode}')"
        elif tns_type == "pad":
            # F.pad padding operation
            pad_amount = tensorop.pad_amount if hasattr(tensorop, 'pad_amount') and tensorop.pad_amount else (0, 0, 0, 0)
            mode = tensorop.pad_mode if hasattr(tensorop, 'pad_mode') else 'constant'

            # Convert PyTorch pad format (left, right, top, bottom) to TF format [[top, bottom], [left, right]]
            if len(pad_amount) == 4:
                left, right, top, bottom = pad_amount
                paddings = f"[[0, 0], [{top}, {bottom}], [{left}, {right}], [0, 0]]"
            else:
                paddings = "[[0, 0], [0, 0], [0, 0], [0, 0]]"

            if mode == 'constant':
                ts_op_synt = f"tf.pad({prev_out_var}, {paddings}, mode='CONSTANT')"
            elif mode == 'reflect':
                ts_op_synt = f"tf.pad({prev_out_var}, {paddings}, mode='REFLECT')"
            elif mode == 'replicate':
                ts_op_synt = f"tf.pad({prev_out_var}, {paddings}, mode='SYMMETRIC')"
            else:
                ts_op_synt = f"tf.pad({prev_out_var}, {paddings}, mode='CONSTANT')"
        else:  # dropout
            # F.dropout operation
            rate = tensorop.dropout_rate if hasattr(tensorop, 'dropout_rate') else 0.5
            training_aware = getattr(tensorop, 'dropout_training_aware', False)

            if training_aware:
                # Use layers.Dropout for training-aware dropout
                # This will be handled as a layer, not inline operation
                # For now, generate inline with training parameter
                ts_op_synt = f"tf.keras.layers.Dropout({rate})({prev_out_var}, training=training)"
            else:
                # Always apply dropout (not training-aware)
                ts_op_synt = f"tf.nn.dropout({prev_out_var}, rate={rate})"

        return ts_op_synt

    # For other operations, get params as usual
    prev_out_var, params = utils.get_tensorop_params(tensorop,
                                                     modules_details)
    if in_var is not None:
        prev_out_var = in_var
    if tns_type == "reshape":
        ts_op_synt = f"tf.reshape({prev_out_var}, [{params}])"
    elif tns_type == "concatenate":
        axis = tensorop.concatenate_dim
        # Fix axis for concat with Conv layers: PyTorch channels-first vs TF channels-last
        # Only apply if sources produce multi-dimensional tensors (3D+), not 2D
        if axis == 1 and hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
            # First check: if all sources are layers that produce 2D output, keep axis=1
            all_2d_layers = True
            for source_layer in tensorop.layers_of_tensors:
                if isinstance(source_layer, str) and source_layer + "_layer" in modules_details:
                    layer_details = modules_details[source_layer + "_layer"]
                    if len(layer_details) > 3 and layer_details[3]:
                        layer_obj = layer_details[3]
                        class_name = layer_obj.__class__.__name__ if hasattr(layer_obj, '__class__') else ''
                        # Check if it's a layer that produces 2D output (LinearLayer, FlattenLayer)
                        if not any(name in class_name for name in ['Linear', 'Flatten']):
                            all_2d_layers = False
                            break
                else:
                    all_2d_layers = False
                    break

            if all_2d_layers:
                # All sources are layers that produce 2D output, keep axis=1
                ts_op_synt = f"tf.concat([{params}], axis=1)"
            else:
                # Check if any source is from operations or layers that produce 2D tensors
                # (squeeze, flatten, reduce_max, reduce_mean, subscript slicing, Dense with 2D input, etc.)
                is_2d_tensor = False
                for source_layer in tensorop.layers_of_tensors:
                    if isinstance(source_layer, str):
                        # Check operations
                        if source_layer + "_op" in modules_details:
                            op_syntax = modules_details[source_layer + "_op"][0] if modules_details[source_layer + "_op"] else ""
                            # Check for operations that reduce to 2D
                            if any(pattern in op_syntax.lower() for pattern in ['squeeze', 'flatten', 'reduce_max', 'reduce_mean', 'amax']):
                                is_2d_tensor = True
                                break
                            # Check for subscript operations that slice to 2D (e.g., [:, -1, :])
                            if '_subscript_' in source_layer:
                                is_2d_tensor = True
                                break
                        # Check layers (LinearLayer with 2D input produces 2D output, FlattenLayer always produces 2D, RNN hidden states are 2D)
                        elif source_layer + "_layer" in modules_details:
                            layer_details = modules_details[source_layer + "_layer"]
                            if len(layer_details) > 3 and layer_details[3]:
                                layer_obj = layer_details[3]
                                # RNN hidden states are always 2D in TensorFlow
                                if hasattr(layer_obj, '__class__') and layer_obj.__class__.__name__ in ('SimpleRNNLayer', 'LSTMLayer', 'GRULayer'):
                                    # If using hidden state (not sequence output), it's 2D
                                    if hasattr(layer_obj, 'return_type') and layer_obj.return_type in ('hidden', 'both'):
                                        is_2d_tensor = True
                                        break
                                # FlattenLayer always produces 2D output
                                elif hasattr(layer_obj, '__class__') and 'Flatten' in layer_obj.__class__.__name__:
                                    is_2d_tensor = True
                                    break
                                # LinearLayer outputs are same dimensionality as inputs
                                # If input is 2D (from reduce, squeeze, other Linear, RNN hidden, Flatten, etc.), output is 2D
                                elif hasattr(layer_obj, '__class__') and 'Linear' in layer_obj.__class__.__name__:
                                    # Check if LinearLayer's input is 2D by checking name_module_input
                                    if hasattr(layer_obj, 'name_module_input') and layer_obj.name_module_input:
                                        input_source = layer_obj.name_module_input
                                        # Check if input is from an op that produces 2D
                                        if input_source + "_op" in modules_details:
                                            input_op_syntax = modules_details[input_source + "_op"][0]
                                            if any(p in input_op_syntax.lower() for p in ['reduce', 'squeeze', 'flatten', 'amax']):
                                                is_2d_tensor = True
                                                break
                                        # Check if input is from RNN hidden state (ends with __hidden or __cell suffix)
                                        elif input_source.endswith('__hidden') or input_source.endswith('__cell'):
                                            is_2d_tensor = True
                                            break
                                        # Check if input is from another 2D-producing layer (LinearLayer, FlattenLayer)
                                        elif input_source + "_layer" in modules_details:
                                            input_layer_details = modules_details[input_source + "_layer"]
                                            if len(input_layer_details) > 3 and input_layer_details[3]:
                                                input_layer_obj = input_layer_details[3]
                                                input_class_name = input_layer_obj.__class__.__name__ if hasattr(input_layer_obj, '__class__') else ''
                                                # If Linear or Flatten, output is 2D
                                                if any(name in input_class_name for name in ['Linear', 'Flatten']):
                                                    is_2d_tensor = True
                                                    break

                # Only convert axis if sources are NOT 2D tensors
                if not is_2d_tensor:
                    # Find Conv layer dimension to determine target axis
                    for key, val in modules_details.items():
                        if isinstance(val, list) and len(val) > 3 and val[3]:
                            layer_obj = val[3]
                            if hasattr(layer_obj, '__class__'):
                                layer_class = layer_obj.__class__.__name__
                                if layer_class.startswith('Conv'):
                                    if '1D' in layer_class:
                                        axis = 2
                                    elif '2D' in layer_class:
                                        axis = 3
                                    elif '3D' in layer_class:
                                        axis = 4
                                    break
                ts_op_synt = f"tf.concat([{params}], axis={axis})"
        else:
            ts_op_synt = f"tf.concat([{params}], axis={axis})"
    elif tns_type == "transpose":
        # PyTorch transpose(dim0, dim1) swaps two dims - convert to full perm
        transpose_dims = tensorop.transpose_dim
        if len(transpose_dims) == 2:
            dim0, dim1 = transpose_dims
            # Infer tensor dimensionality from source layer
            num_dims = 3  # default fallback
            if hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
                source_layer = tensorop.layers_of_tensors[0] if isinstance(tensorop.layers_of_tensors[0], str) else None
                if source_layer:
                    # Check if source is a layer
                    if source_layer + '_layer' in modules_details:
                        layer_obj = modules_details[source_layer + '_layer'][3] if len(modules_details[source_layer + '_layer']) > 3 else None
                        if layer_obj and hasattr(layer_obj, '__class__'):
                            layer_class = layer_obj.__class__.__name__
                            # Determine dimensionality based on layer type
                            if layer_class in ['Dense', 'LinearLayer']:
                                # Dense keeps input dimensionality - check if it's 2D or 3D
                                # For now, check if preceded by flatten/pool → 2D, otherwise 3D
                                # Simple heuristic: if no RNN before, likely 2D
                                has_rnn = any('RNN' in str(v[3].__class__.__name__) if len(v) > 3 and hasattr(v[3], '__class__') else False
                                             for k, v in modules_details.items() if k.endswith('_layer') and isinstance(v, list))
                                num_dims = 3 if has_rnn else 2
                            elif 'RNN' in layer_class or 'LSTM' in layer_class or 'GRU' in layer_class:
                                # RNN with return_sequences → 3D, without → 2D
                                num_dims = 3 if (hasattr(layer_obj, 'return_sequences') and layer_obj.return_sequences) else 2
                            elif 'Conv1D' in layer_class or 'Embedding' in layer_class:
                                num_dims = 3
                            elif 'Conv2D' in layer_class:
                                num_dims = 4
                            elif 'Conv3D' in layer_class:
                                num_dims = 5
                    # Check if source is an op
                    elif source_layer + '_op' in modules_details:
                        op_syntax = modules_details[source_layer + '_op'][0] if modules_details[source_layer + '_op'] else ""
                        # Infer from operation
                        if 'squeeze' in op_syntax.lower():
                            num_dims = 2  # squeeze typically reduces to 2D
                        elif 'expand_dims' in op_syntax.lower() or 'unsqueeze' in op_syntax.lower():
                            num_dims = 3  # adds a dimension
            # Ensure num_dims is at least max(dim0, dim1) + 1
            num_dims = max(num_dims, max(dim0, dim1) + 1)
            # Create full permutation by swapping the two specified dims
            perm = list(range(num_dims))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            perm_str = ", ".join(map(str, perm))
            ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{perm_str}])"
        else:
            # Fallback for other cases
            ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{params}])"
    elif tns_type == "permute":
        # Note: permute_in flag on layers controls whether transpose is added before the layer.
        # For TensorFlow, permute_in is skipped in utils_nn.py since TF uses channels-last throughout.
        ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{params}])"
    elif tns_type == "multiply":
        ts_op_synt = f"tf.math.multiply({params})"
    elif tns_type == "mean":
        axis = tensorop.reduce_dim
        ts_op_synt = f"tf.reduce_mean({prev_out_var}, axis={axis})"
    elif tns_type == "max":
        axis = tensorop.reduce_dim
        ts_op_synt = f"tf.reduce_max({prev_out_var}, axis={axis})"
    elif tns_type == "squeeze":
        # Check if this is squeeze(0) on an RNN hidden state
        # In PyTorch, RNN hidden states have shape (num_layers, batch, hidden)
        # In TensorFlow, they have shape (batch, hidden) - no layers dimension
        # So squeeze(0) on RNN hidden state should be a no-op in TensorFlow
        is_rnn_hidden_squeeze = False
        if tensorop.reduce_dim == 0 and hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
            source_layer = tensorop.layers_of_tensors[0] if isinstance(tensorop.layers_of_tensors[0], str) else None
            if source_layer:
                # Check if source is an RNN layer and prev_out_var is its hidden state
                # Strip __hidden/__cell suffix (AST parser adds these for RNN state variables)
                base_layer = source_layer.replace('__hidden', '').replace('__cell', '')
                layer_key = f"{base_layer}_layer"
                if layer_key in modules_details:
                    layer_details = modules_details[layer_key]
                    # layer_details structure: [syntax, out_var, in_var, layer_obj, hidden_var, ...]
                    # For RNN layers with return_state, hidden var is at index 4
                    if len(layer_details) > 4:
                        layer_obj = layer_details[3] if len(layer_details) > 3 else None
                        hidden_var = layer_details[4]
                        # Check if this is an RNN/LSTM/GRU layer and prev_out_var is the hidden state
                        if (layer_obj and hasattr(layer_obj, '__class__') and
                            layer_obj.__class__.__name__ in ['SimpleRNNLayer', 'LSTMLayer', 'GRULayer'] and
                            prev_out_var == hidden_var):
                            is_rnn_hidden_squeeze = True

        if is_rnn_hidden_squeeze:
            # No-op: hidden state already has correct shape in TensorFlow
            ts_op_synt = prev_out_var
        elif tensorop.reduce_dim is not None:
            axis = tensorop.reduce_dim
            # Fix axis for squeeze after 1D pooling: PyTorch channels-first vs TF channels-last
            # After 1D AdaptiveAvgPool/MaxPool: PyTorch [B,C,1], TF [B,1,C]
            # PyTorch squeeze(-1) removes sequence dim, TF should use axis=1
            if axis == -1 and hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
                source_layer = tensorop.layers_of_tensors[0] if isinstance(tensorop.layers_of_tensors[0], str) else None
                if source_layer:
                    # Check if source is a 1D pooling layer
                    if source_layer + "_layer" in modules_details:
                        layer_obj = modules_details[source_layer + "_layer"][3] if len(modules_details[source_layer + "_layer"]) > 3 else None
                        if layer_obj and hasattr(layer_obj, '__class__') and 'Pooling' in layer_obj.__class__.__name__:
                            # Check if it's 1D pooling - convert squeeze(-1) axis
                            # After 1D pooling: PyTorch [B,C,1] squeeze(-1)→[B,C], TF [B,1,C] squeeze(1)→[B,C]
                            if hasattr(layer_obj, 'dimension') and layer_obj.dimension == '1D':
                                axis = 1  # Convert to TF channels-last format
                    # Check if source is a concat op that comes from 1D pooling
                    elif source_layer + "_op" in modules_details:
                        op_details = modules_details[source_layer + "_op"]
                        # Check if this is a concatenate op by looking at the syntax
                        if len(op_details) > 0 and isinstance(op_details[0], str) and 'tf.concat' in op_details[0]:
                            # The concat was already converted to use axis=2 for 1D pooling sources
                            # So squeeze(-1) should become squeeze(1) to remove the sequence dimension
                            axis = 1
            ts_op_synt = f"tf.squeeze({prev_out_var}, axis={axis})"
        else:
            ts_op_synt = f"tf.squeeze({prev_out_var})"
    elif tns_type == "unsqueeze":
        axis = tensorop.reduce_dim
        # Skip unsqueeze(0) when output is used as RNN initial_state
        # PyTorch: initial_state shape is [num_layers, B, H]
        # TensorFlow: initial_state shape is [B, H]
        # So unsqueeze(0) that adds num_layers dimension should be skipped
        if axis == 0 and hasattr(tensorop, 'is_rnn_initial_state') and tensorop.is_rnn_initial_state:
            # Skip this unsqueeze - TensorFlow doesn't need the num_layers dimension
            ts_op_synt = f"SKIP:{prev_out_var}"
        else:
            ts_op_synt = f"tf.expand_dims({prev_out_var}, axis={axis})"
    elif tns_type == "zeros_like":
        ts_op_synt = f"tf.zeros_like({prev_out_var})"
    elif tns_type == "normalize":
        # F.normalize(x, p=2, dim=1) -> tf.nn.l2_normalize(x, axis=1)
        # Currently only supports L2 normalization (p=2)
        axis = tensorop.reduce_dim
        ts_op_synt = f"tf.nn.l2_normalize({prev_out_var}, axis={axis})"
    elif tns_type == "repeat":
        # PyTorch .repeat(1, t, 1) -> TensorFlow tf.tile(x, [1, t, 1])
        # params contains resolved repeat counts (variables resolved to their actual names)
        ts_op_synt = f"tf.tile({prev_out_var}, [{params}])"
    elif tns_type == "binop_add":
        ts_op_synt = f"{params[0]} + {params[1]}" if isinstance(params, list) else f"tf.add({params})"
    elif tns_type == "binop_subtract":
        ts_op_synt = f"{params[0]} - {params[1]}" if isinstance(params, list) else f"tf.subtract({params})"
    elif tns_type == "binop_multiply":
        ts_op_synt = f"{params[0]} * {params[1]}" if isinstance(params, list) else f"tf.multiply({params})"
    elif tns_type == "binop_divide":
        ts_op_synt = f"{params[0]} / {params[1]}" if isinstance(params, list) else f"tf.divide({params})"
    elif tns_type == "binop_floor_divide":
        ts_op_synt = f"{params[0]} // {params[1]}" if isinstance(params, list) else f"tf.math.floordiv({params})"
    elif tns_type == "subscript":
        # General subscripting/slicing operation
        # subscript_indices contains the slice pattern as a string (e.g., "[-1]", "[:, -1, :]")
        # Need to convert axes for Conv layers (PyTorch channels-first → TF channels-last)
        subscript_pattern = tensorop.subscript_indices

        # Check if source is a Conv layer that needs axis remapping
        if hasattr(tensorop, 'layers_of_tensors') and tensorop.layers_of_tensors:
            source_layer = tensorop.layers_of_tensors[0] if isinstance(tensorop.layers_of_tensors[0], str) else None
            if source_layer and source_layer + "_layer" in modules_details:
                layer_obj = modules_details[source_layer + "_layer"][3] if len(modules_details[source_layer + "_layer"]) > 3 else None
                if layer_obj and hasattr(layer_obj, '__class__'):
                    layer_class = layer_obj.__class__.__name__
                    # Conv1D: PyTorch [B,C,L] → TF [B,L,C], remap dims 1↔2
                    # Conv2D: PyTorch [B,C,H,W] → TF [B,H,W,C], remap dims 1→3, 2→1, 3→2
                    if layer_class == 'Conv1D' and subscript_pattern.count(',') == 2:
                        # 3D subscript for Conv1D: swap dims 1 and 2
                        # Parse pattern like "[:, :, -1]" and swap positions
                        parts = subscript_pattern.strip('[]').split(',')
                        if len(parts) == 3:
                            # Swap parts[1] and parts[2]
                            subscript_pattern = f"[{parts[0]},{parts[2]},{parts[1]}]"
                    elif layer_class == 'Conv2D' and subscript_pattern.count(',') == 3:
                        # 4D subscript for Conv2D: [B, C, H, W] → [B, H, W, C]
                        # Remap: [:, c, h, w] → [:, h, w, c]
                        parts = subscript_pattern.strip('[]').split(',')
                        if len(parts) == 4:
                            subscript_pattern = f"[{parts[0]},{parts[2]},{parts[3]},{parts[1]}]"

        ts_op_synt = f"{prev_out_var}{subscript_pattern}"
    elif tns_type == "shape_dim":
        # Extract shape dimension (e.g., b = tf.shape(inp)[0])
        # Get input tensor from layers_of_tensors, not prev_out_var
        tensors = tensorop.layers_of_tensors
        if isinstance(tensors[0], str):
            if tensors[0] == 'INPUT':
                source_var = 'inp'
            else:
                # Check if the module exists in modules_details
                if f"{tensors[0]}_layer" in modules_details or f"{tensors[0]}_op" in modules_details:
                    source_tensors = utils.get_layers_output_for_tensorops(tensors, modules_details)
                    source_var = source_tensors[0]
                else:
                    # Module not found, use the tensor name directly (might be a variable)
                    source_var = tensors[0]
        else:
            source_var = tensors[0]

        dim_index = tensorop.reduce_dim
        ts_op_synt = f"tf.shape({source_var})[{dim_index}]"
    else:
        ts_op_synt = f"tf.matmul({params})"
    return ts_op_synt
