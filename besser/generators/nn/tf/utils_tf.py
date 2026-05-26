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
            lyr = (
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
    prev_out_var, params = utils.get_tensorop_params(tensorop,
                                                     modules_details)
    if in_var is not None:
        prev_out_var = in_var

    tns_type = tensorop.tns_type
    if tns_type == "reshape":
        ts_op_synt = f"tf.reshape({prev_out_var}, [{params}])"
    elif tns_type == "concatenate":
        axis = tensorop.concatenate_dim
        ts_op_synt = f"tf.concat([{params}], axis={axis})"
    elif tns_type == "transpose":
        # PyTorch transpose(dim0, dim1) swaps two dims - convert to full perm
        transpose_dims = tensorop.transpose_dim
        if len(transpose_dims) == 2:
            # Assume 3D tensor (batch, dim1, dim2) - most common case
            # Create full permutation by swapping the two specified dims
            dim0, dim1 = transpose_dims
            perm = list(range(3))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            perm_str = ", ".join(map(str, perm))
            ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{perm_str}])"
        else:
            # Fallback for other cases
            ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{params}])"
    elif tns_type == "permute":
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
        if tensorop.reduce_dim is not None:
            axis = tensorop.reduce_dim
            ts_op_synt = f"tf.squeeze({prev_out_var}, axis={axis})"
        else:
            ts_op_synt = f"tf.squeeze({prev_out_var})"
    elif tns_type == "unsqueeze":
        axis = tensorop.reduce_dim
        ts_op_synt = f"tf.expand_dims({prev_out_var}, axis={axis})"
    elif tns_type == "binop_add":
        ts_op_synt = f"{params[0]} + {params[1]}" if isinstance(params, list) else f"tf.add({params})"
    elif tns_type == "binop_subtract":
        ts_op_synt = f"{params[0]} - {params[1]}" if isinstance(params, list) else f"tf.subtract({params})"
    elif tns_type == "binop_multiply":
        ts_op_synt = f"{params[0]} * {params[1]}" if isinstance(params, list) else f"tf.multiply({params})"
    elif tns_type == "binop_divide":
        ts_op_synt = f"{params[0]} / {params[1]}" if isinstance(params, list) else f"tf.divide({params})"
    elif tns_type == "subscript":
        # General subscripting/slicing operation
        # subscript_indices contains the slice pattern as a string (e.g., "[-1]", "[:, -1, :]")
        ts_op_synt = f"{prev_out_var}{tensorop.subscript_indices}"
    elif tns_type == "shape_dim":
        # Extract shape dimension (e.g., b = tf.shape(x)[0])
        dim_index = tensorop.reduce_dim
        ts_op_synt = f"tf.shape({prev_out_var})[{dim_index}]"
    else:
        ts_op_synt = f"tf.matmul({params})"
    return ts_op_synt
