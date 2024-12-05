"""
This module provides the `SetupLayerSyntax` class and the 
`get_tensorop_syntax` function.
The `SetupLayerSyntax` class is used to define the syntax 
of layers in PyTorch, while `get_tensorop_syntax` defines the 
tensorOps.
"""

from besser.BUML.metamodel.nn import TensorOp
from besser.generators.nn import utils_nn as utils


class SetupLayerSyntax:
    """
    This class is used to map the BUML synatx for layers to PyTorch 
    syntax.
    It processes the layers based on their type.
    """
    def __init__(self, layer, modules_details):
        self.layer = layer
        self.modules_details = modules_details
        self.permute_out = None
        self.permute_in = None
        self.dim = None

    def setup_general_layer(self):
        """It defines the syntax of general layers."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = nn"

        if cls_name == "LinearLayer":
            in_f = self.layer.in_features
            out_f = self.layer.out_features
            lyr = f"{lyr}.Linear(in_features={in_f}, out_features={out_f})"
        elif cls_name == "FlattenLayer":
            st_dim = self.layer.start_dim
            en_dim = self.layer.end_dim
            lyr = f"{lyr}.Flatten(start_dim={st_dim}, end_dim={en_dim})"
        else: #cls_name == "EmbeddingLayer"
            nm = self.layer.num_embeddings
            dm = self.layer.embedding_dim
            lyr = f"{lyr}.Embedding(num_embeddings={nm}, embedding_dim={dm})"

        return lyr

    def setup_layer_modifier(self):
        """It defines the syntax of layers' modifiers."""
        cls_name = self.layer.__class__.__name__
        parent_cls = self.layer.__class__.mro()[1].__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = nn"
        if parent_cls == "NormalizationLayer":
            if cls_name == "BatchNormLayer":
                dim = self.layer.dimension[0]
                num_f = self.layer.num_features
                lyr = f"{lyr}.BatchNorm(){dim}d(num_features={num_f})"
            else: #cls_name == "LayerNormLayer"
                norm_shape = self.layer.normalized_shape
                if norm_shape[1] is not None:
                    if norm_shape[2] is not None:
                        shape = [norm_shape[0], norm_shape[1], norm_shape[2]]
                    else:
                        shape = [norm_shape[0], norm_shape[1]]
                else:
                    shape = [norm_shape[0]]

                lyr = f"{lyr}.LayerNorm(normalized_shape={shape})"
        else: #cls_name == "DropoutLayer"
            lyr = f"{lyr}.Dropout(p={self.layer.rate})"
        return lyr

    def add_permute(self, lyr_name, dim, in_var_layer):
        """It permutes the input of the layer"""
        if dim is None:
            perm_dim = [0, 2, 1]
        else:
            if dim == "1":
                perm_dim = [0, 2, 1]
            elif dim == "2":
                perm_dim = [0, 3, 1, 2]
            else:
                perm_dim = [0, 4, 1, 2, 3]

        tns = TensorOp(name=f"{lyr_name}_in_op", tns_type="permute",
                       permute_dim=perm_dim)
        tns_out = utils.handle_tensorop
        self.modules_details = tns_out(tns, self.modules_details,
                                       get_tensorop_syntax,
                                       in_var_layer)


    def setup_rnn(self):
        """It defines the syntax of rnn layers."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name

        if self.layer.permute_dim:
            permute = TensorOp(name=f"{lyr_name}_op", tns_type="permute",
                               permute_dim=[0, 2, 1])
            tns_out = utils.handle_tensorop
            self.modules_details = tns_out(permute, self.modules_details,
                                           get_tensorop_syntax)

        layer_type = cls_name[:-5]
        in_sz = self.layer.input_size
        h_sz = self.layer.hidden_size
        bd = self.layer.bidirectional
        drp = self.layer.dropout
        btch = self.layer.batch_first
        lyr = (
            f"self.{lyr_name} = nn.{layer_type}(input_size={in_sz}, "
            f"hidden_size={h_sz}, bidirectional={bd}, dropout={drp}, "
            f"batch_first={btch})"
        )
        return lyr, self.modules_details

    def setup_actv_func(self):
        """It defines the syntax of activation functions."""
        actv_func = self.layer.actv_func
        lyr = None
        if hasattr(self.layer, 'actv_func'):
            if actv_func == "relu":
                lyr = "self.relu_activ = nn.ReLU()"
            elif actv_func == "leaky_relu":
                lyr = "self.leaky_relu_activ = nn.LeakyReLU()"
            elif actv_func == "sigmoid":
                lyr = "self.sigmoid_activ = nn.Sigmoid()"
            elif actv_func == "softmax":
                lyr = "self.softmax_activ = nn.Softmax()"
            elif actv_func == "tanh":
                lyr = "self.tanh_activ = nn.Tanh()"
        return lyr

    def setup_cnn(self):
        """It defines the syntax of cnn layers (conv and pooling)."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name

        if cls_name == "PoolingLayer":
            lyr = self.setup_pooling(lyr_name)
        else:
            lyr = self.setup_conv(lyr_name, cls_name)

        #self.modules_details is returned to have the same structure
        #of setup_rnn from torch
        return lyr, self.modules_details

    def setup_conv(self, lyr_name, cls_name):
        """It defines the syntax of convolutional layers."""
        dim = cls_name[-2:-1]
        in_chan = self.layer.in_channels
        out_chan = self.layer.out_channels
        kernel = utils.format_value(self.layer.kernel_dim)
        stride = utils.format_value(self.layer.stride_dim)
        pad = self.layer.padding_amount
        self.permute_in = self.layer.permute_in
        self.permute_out = self.layer.permute_out
        self.dim = dim
        lyr = (
            f"self.{lyr_name} = nn.Conv{dim}d(in_channels={in_chan}, "
            f"out_channels={out_chan}, kernel_size={kernel}, "
            f"stride={stride}, padding={pad})"
        )
        return lyr


    def setup_pooling(self, lyr_name):
        """It defines the syntax of pooling layers."""
        pl_type = self.layer.pooling_type
        dim = self.layer.dimension[-2:-1]
        self.dim = dim

        self.permute_in = self.layer.permute_in
        self.permute_out = self.layer.permute_out

        if pl_type == "max" or pl_type == "average":
            pl = "Max" if pl_type == "max" else "Avg"
            kernel = utils.format_value(self.layer.kernel_dim)
            stride = utils.format_value(self.layer.stride_dim)
            pad = self.layer.padding_amount
            lyr = (
                f"self.{lyr_name} = nn.{pl}Pool{dim}d(kernel_size={kernel}, "
                f"stride={stride}, padding={pad})"
            )
        else:
            if pl_type == "adaptive_average":
                pl = "AdaptiveAvg"
            else:
                pl = "AdaptiveMax"

            size = utils.format_value(self.layer.output_dim)
            lyr = (
                f"self.{lyr_name} = nn.{pl}Pool{dim}d(output_size={size})"
            )
        return lyr

def get_tensorop_syntax(tensorop, modules_details, in_var=None):
    """It defines the syntax of tensorops."""

    prev_out_var, params = utils.get_tensorop_params(tensorop,
                                                     modules_details)
    if in_var is not None:
        prev_out_var = in_var

    tns_type = tensorop.tns_type
    if tns_type == "reshape":
        ts_op_synt = f"{prev_out_var}.reshape({params})"
    elif tns_type == "concatenate":
        dim = tensorop.concatenate_dim
        ts_op_synt = f"torch.cat(({params}), dim={dim})"
    elif tns_type == "transpose":
        ts_op_synt = f"{prev_out_var}.transpose({params})"
    elif tns_type == "permute":
        ts_op_synt = f"{prev_out_var}.permute({params})"
    elif tns_type == "multiply":
        ts_op_synt = f"torch.mul({params})"
    else:
        ts_op_synt = f"torch.matmul({params})"
    return ts_op_synt
