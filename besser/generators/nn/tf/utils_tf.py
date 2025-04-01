"""
This module provides the `SetupLayerSyntax` class and the 
`get_tensorop_syntax` function.
The `SetupLayerSyntax` class is used to define the syntax 
of layers in TensorFlow, while `get_tensorop_syntax` defines the 
tensorOps.
"""

from besser.generators.nn import utils_nn as utils


class SetupLayerSyntax:
    """
    This class is used to map the BUML synatx for layers to TensorFlow
    syntax.
    It processes the layers based on their type.
    """
    def __init__(self, layer, modules_details):
        self.layer = layer
        self.modules_details: dict = modules_details
        self.permute_out: bool = None
        self.permute_in: bool = None

    def setup_general_layer(self):
        """It defines the syntax of general layers."""
        actv_func = self.setup_actv_func()
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = layers"
        if cls_name == "LinearLayer":
            lyr = (
                f"{lyr}.Dense(units={self.layer.out_features}, "
                f"activation={actv_func})"
            )
        elif cls_name == "FlattenLayer":
            lyr = f"{lyr}.Flatten()"
        else: #cls_name == "EmbeddingLayer"
            lyr = (
                f"{lyr}.Embedding(input_dim={self.layer.num_embeddings}, "
                f"output_dim={self.layer.embedding_dim})"
            )
        return lyr

    def setup_layer_modifier(self):
        """It defines the syntax of layers' modifiers."""
        cls_name = self.layer.__class__.__name__
        parent_cls = self.layer.__class__.mro()[1].__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = layers"
        if parent_cls == "NormalizationLayer":
            if cls_name == "BatchNormLayer":
                lyr = f"{lyr}.BatchNormalization()"
            else: #cls_name == "LayerNormLayer"
                norm_shape = self.layer.normalized_shape
                if norm_shape[1] is not None:
                    if norm_shape[2] is not None:
                        shape = [norm_shape[0], norm_shape[1], norm_shape[2]]
                    else:
                        shape = [norm_shape[0], norm_shape[1]]
                else:
                    shape = [norm_shape[0]]

                lyr = f"{lyr}.LayerNormalization(axis={shape})"
        else: #cls_name == "DropoutLayer"
            lyr = f"{lyr}.Dropout(rate={self.layer.rate})"
        return lyr

    def setup_rnn(self):
        """It defines the syntax of rnn layers."""
        actv_func = self.setup_actv_func()
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        layer_type = cls_name[:-5]
        lyr = (
            f"layers.{layer_type}(units={self.layer.hidden_size}, "
            f"activation={actv_func}, dropout={self.layer.dropout}"
        )

        if self.layer.return_type == "full":
            lyr = f"{lyr}, return_sequences=True)"
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
            list_func = ["relu", "tanh", "sigmoid","softmax", "leaky_relu"]
            if activ is not None:
                if activ in list_func:
                    return f"'{self.layer.actv_func}'"
                else:
                    return f"{self.layer.actv_func}"
            else:
                return None

    def setup_conv(self, lyr_name, cls_name):
        """It defines the syntax of convolutional layers."""
        actv_func = self.setup_actv_func()
        dim = cls_name[-2:-1]
        filters = self.layer.out_channels
        pad_type = self.layer.padding_type
        kernel = utils.format_value(self.layer.kernel_dim)
        stride = utils.format_value(self.layer.stride_dim)
        pad_amount = self.layer.padding_amount
        lyr = ""
        if pad_amount != 0:
            lyr = (
                f"self.{lyr_name}_pad = layers.ZeroPadding{dim}D("
                f"padding={pad_amount})#"
            )
        lyr = (
            f"{lyr}self.{lyr_name} = layers.Conv{dim}D(filters={filters}, "
            f"kernel_size={kernel}, strides={stride}, "
            f"padding='{pad_type}', activation={actv_func})"
        )
        return lyr

    def setup_pooling(self, lyr_name):
        """It defines the syntax of pooling layers."""
        pl_type = self.layer.pooling_type
        dim = self.layer.dimension[-2:-1]
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


def get_tensorop_syntax(tensorop, modules_details, in_var=None):
    """It defines the syntax of tensorops."""
    prev_out_var, params = utils.get_tensorop_params(tensorop,
                                                     modules_details)
    if in_var is not None:
        prev_out_var = in_var

    tns_type = tensorop.tns_type
    if tns_type == "reshape":
        ts_op_synt = f"tf.reshape({prev_out_var}, {params})"
    elif tns_type == "concatenate":
        axis = tensorop.concatenate_dim
        ts_op_synt = f"tf.concat([{params}], axis={axis})"
    elif tns_type == "transpose":
        ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{params}])"
    elif tns_type == "permute":
        ts_op_synt = f"tf.transpose({prev_out_var}, perm=[{params}])"
    elif tns_type == "multiply":
        ts_op_synt = f"tf.math.multiply({params})"
    else:
        ts_op_synt = f"tf.matmul({params})"
    return ts_op_synt
