"""
This module provides the `SetupLayerSyntax` class and the
`get_tensorop_syntax` function along with two functions to
process the activation function.
The `SetupLayerSyntax` class is used to define the syntax
of layers in PyTorch, while `get_tensorop_syntax` defines the
tensorOps.
"""

from besser.BUML.metamodel.nn import TensorOp, Layer
from besser.generators.nn import utils_nn as utils


class SetupLayerSyntax:
    """
    This class is used to get PyTorch layer syntax from BUML layer object.
    It processes the layers based on their type.

    Attributes:
        layer (Layer): the BUML layer object.
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.
        permute_out (bool | None): Whether to add a permute tensorop after
            the layer.
        permute_in (bool | None): Whether to add a permute tensorop before
            the layer.
        dim (str | None): The dimentionality of the layer

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
        self.dim: str | None = None

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
            padding_idx = self.layer.padding_idx
            if padding_idx is not None:
                lyr = f"{lyr}.Embedding(num_embeddings={nm}, embedding_dim={dm}, padding_idx={padding_idx})"
            else:
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
                eps = self.layer.eps
                momentum = self.layer.momentum
                affine = "True" if self.layer.affine else "False"
                track_stats = "True" if self.layer.track_running_stats else "False"
                lyr = (
                    f"{lyr}.BatchNorm{dim}d(num_features={num_f}, eps={eps}, "
                    f"momentum={momentum}, affine={affine}, "
                    f"track_running_stats={track_stats})"
                )
            else: #cls_name == "LayerNormLayer"
                norm_shape = self.layer.normalized_shape
                eps = self.layer.eps
                elementwise_affine = "True" if self.layer.affine else "False"
                lyr = (
                    f"{lyr}.LayerNorm(normalized_shape={norm_shape}, eps={eps}, "
                    f"elementwise_affine={elementwise_affine})"
                )
        else: #cls_name == "DropoutLayer"
            # Use Dropout1d/2d/3d for spatial variants, regular Dropout otherwise
            if hasattr(self.layer, 'dimension') and self.layer.dimension:
                lyr = f"{lyr}.Dropout{self.layer.dimension}d(p={self.layer.rate})"
            else:
                lyr = f"{lyr}.Dropout(p={self.layer.rate})"
        return lyr

    def add_permute(self, lyr_name: str, dim: str, in_var_layer: str,
                    permute_in: bool = True, sequential: bool = False,
                    is_subnn: bool = False):
        """
        It permutes the input and output of conv layers

        Args:
            lyr_name (str): the name of the layer.
            dim (str): the dimentionality of the layer ('1', '2' or '3').
            in_var_layer (str): the input variable notation of the layer
                (e.g., 'x', 'x_1', ...).
            permute_in (bool): Whether to permute the input of the layer.
            sequential (bool): Whether the layer is in a seq architecture.
            is_subnn (bool): Whether the layer is in a subnn model.

        Returns:
            None, but stores the permute module in the
            modules_details dictionary.

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

        if sequential or is_subnn:
            self.modules_details[perm_name] = [f"Permute(dims={perm_dim})",
                                               in_var_layer]
        else:
            tns = TensorOp(
                name=perm_name, tns_type="permute", permute_dim=perm_dim
            )
            tns_out = utils.handle_tensorop
            tns_out(
                tns, self.modules_details, get_tensorop_syntax, in_var_layer
            )


    def setup_rnn(self):
        """It defines the syntax of rnn layers."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        layer_type = cls_name[:-5]

        # Map BUML layer names to PyTorch layer names
        layer_mapping = {
            "SimpleRNN": "RNN",
            "LSTM": "LSTM",
            "GRU": "GRU"
        }
        layer_type = layer_mapping.get(layer_type, layer_type)

        in_sz = self.layer.input_size
        h_sz = self.layer.hidden_size
        bd = self.layer.bidirectional
        drp = self.layer.dropout
        btch = self.layer.batch_first
        bs = self.layer.bias
        lyr = (
            f"self.{lyr_name} = nn.{layer_type}(input_size={in_sz}, "
            f"hidden_size={h_sz}, bidirectional={bd}, dropout={drp}, "
            f"batch_first={btch}, bias={bs})"
        )
        return lyr


    def setup_actv_func(self):
        """It defines the syntax of activation functions."""
        lyr = None
        activs = {"relu": "ReLU", "leaky_relu": "LeakyReLU",
                  "sigmoid": "Sigmoid", "softmax": "Softmax", "tanh": "Tanh", "gelu": "GELU"}
        if hasattr(self.layer, 'actv_func'):
            actv = self.layer.actv_func
            if actv in activs:
                lyr = f"self.actv_func_{actv} = nn.{activs[actv]}()"
            elif actv is not None:
                if actv.startswith("self"):
                    lyr = f"self.actv_func_{actv[5:]}"
                else:
                    lyr = f"self.actv_func_{actv}"
                lyr = f"{lyr} = get_activation_function({actv})"
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

    def setup_conv(self, lyr_name: str, cls_name: str):
        """
        It defines the syntax of convolutional layers.

        Args:
            lyr_name (str): The name of the layer.
            cls_name (str): The name of its class.

        Returns:
            lyr (str): The syntax of the layer in PyTorch.
        """
        dim = cls_name[-2:-1]
        in_chan = self.layer.in_channels
        out_chan = self.layer.out_channels
        kernel = utils.format_value(self.layer.kernel_dim)
        stride = utils.format_value(self.layer.stride_dim)
        pad = self.layer.padding_amount
        dilation = utils.format_value(self.layer.dilation)
        groups = self.layer.groups
        bias = self.layer.bias
        self.permute_in = self.layer.permute_in
        self.permute_out = self.layer.permute_out
        self.dim = dim
        lyr = (
            f"self.{lyr_name} = nn.Conv{dim}d(in_channels={in_chan}, "
            f"out_channels={out_chan}, kernel_size={kernel}, "
            f"stride={stride}, padding={pad}, dilation={dilation}, "
            f"groups={groups}, bias={bias})"
        )
        return lyr


    def setup_pooling(self, lyr_name: str):
        """
        It defines the syntax of pooling layers.

        Args:
            lyr_name (str): The name of the layer.

        Returns:
            lyr (str): The syntax of the layer in PyTorch.
        """
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
        elif pl_type.startswith("global"):
            out_dim = (1,) * int(dim)
            lyr = (
                f"self.{lyr_name} = nn.AdaptiveAvgPool{dim}d({out_dim})"
            )
            # or tensor.mean(dim=(2, 3, 4), keepdim=True)
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
                                                     modules_details,
                                                     get_rnn_hidden_var)
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
    elif tns_type == "mean":
        dim = tensorop.reduce_dim
        ts_op_synt = f"{prev_out_var}.mean(dim={dim})"
    elif tns_type == "max":
        if tensorop.reduce_dim is not None:
            dim = tensorop.reduce_dim
            ts_op_synt = f"{prev_out_var}.max(dim={dim})[0]"
        else:
            ts_op_synt = f"{prev_out_var}.max()"
    elif tns_type == "zeros_like":
        ts_op_synt = f"torch.zeros_like({prev_out_var})"
    elif tns_type == "squeeze":
        if tensorop.reduce_dim is not None:
            dim = tensorop.reduce_dim
            ts_op_synt = f"{prev_out_var}.squeeze({dim})"
        else:
            ts_op_synt = f"{prev_out_var}.squeeze()"
    elif tns_type == "unsqueeze":
        dim = tensorop.reduce_dim
        ts_op_synt = f"{prev_out_var}.unsqueeze({dim})"
    elif tns_type == "normalize":
        # tf.nn.l2_normalize(x, axis=1) -> F.normalize(x, p=2, dim=1)
        dim = tensorop.reduce_dim
        ts_op_synt = f"F.normalize({prev_out_var}, p=2, dim={dim})"
    elif tns_type == "repeat":
        # TensorFlow tf.tile(x, [1, t, 1]) -> PyTorch .repeat(1, t, 1)
        # params contains resolved repeat counts
        ts_op_synt = f"{prev_out_var}.repeat({params})"
    elif tns_type == "pad":
        # TensorFlow tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode='REFLECT')
        # -> PyTorch F.pad(x, (left,right,top,bottom,...), mode='reflect')
        # Note: TF format is [[dim0_before, dim0_after], [dim1_before, dim1_after], ...]
        # PyTorch format is (lastdim_left, lastdim_right, 2ndlast_left, 2ndlast_right, ...)
        pad_amount = tensorop.pad_amount
        pad_mode = tensorop.pad_mode.lower() if hasattr(tensorop, 'pad_mode') else 'constant'

        # Convert TF padding format to PyTorch format (reverse order, flatten)
        if pad_amount and isinstance(pad_amount, list):
            # TF format: [[N], [H], [W], [C]] (NHWC) or [[N], [C], [H], [W]] depending on format
            # PyTorch F.pad doesn't pad batch dimension, and expects (left, right, top, bottom, ...)
            # Skip first dimension (batch) and reverse the rest
            # [[0,0],[1,1],[2,2],[3,3]] -> skip [0,0], reverse [1,1],[2,2],[3,3] -> (3,3,2,2,1,1)
            pt_pad = []
            # Skip first element (batch dimension) and reverse the rest
            for pair in reversed(pad_amount[1:]):
                if isinstance(pair, list) and len(pair) == 2:
                    pt_pad.extend(pair)
            pad_tuple = tuple(pt_pad)
            ts_op_synt = f"F.pad({prev_out_var}, {pad_tuple}, mode='{pad_mode}')"
        else:
            ts_op_synt = f"F.pad({prev_out_var}, {pad_amount})"
    elif tns_type == "dropout":
        # TensorFlow tf.nn.dropout(x, rate=0.5) -> PyTorch F.dropout(x, p=0.5)
        dropout_rate = tensorop.dropout_rate if hasattr(tensorop, 'dropout_rate') else 0.5
        ts_op_synt = f"F.dropout({prev_out_var}, p={dropout_rate})"
    elif tns_type == "interpolate":
        # TensorFlow tf.image.resize(x, [H, W], method='bilinear')
        # -> PyTorch F.interpolate(x, size=(H, W), mode='bilinear')
        size = tensorop.interpolate_size if hasattr(tensorop, 'interpolate_size') else None
        mode = tensorop.interpolate_mode if hasattr(tensorop, 'interpolate_mode') else 'bilinear'

        if size:
            if isinstance(size, tuple):
                ts_op_synt = f"F.interpolate({prev_out_var}, size={size}, mode='{mode}')"
            else:
                ts_op_synt = f"F.interpolate({prev_out_var}, size={size}, mode='{mode}')"
        else:
            ts_op_synt = f"F.interpolate({prev_out_var}, mode='{mode}')"
    elif tns_type == "binop_add":
        ts_op_synt = f"{params.split(', ')[0]} + {params.split(', ')[1]}" if ', ' in params else f"torch.add({params})"
    elif tns_type == "binop_subtract":
        ts_op_synt = f"{params.split(', ')[0]} - {params.split(', ')[1]}" if ', ' in params else f"torch.subtract({params})"
    elif tns_type == "binop_multiply":
        ts_op_synt = f"{params.split(', ')[0]} * {params.split(', ')[1]}" if ', ' in params else f"torch.multiply({params})"
    elif tns_type == "binop_divide":
        ts_op_synt = f"{params.split(', ')[0]} / {params.split(', ')[1]}" if ', ' in params else f"torch.divide({params})"
    elif tns_type == "binop_floor_divide":
        ts_op_synt = f"{params.split(', ')[0]} // {params.split(', ')[1]}" if ', ' in params else f"torch.floor_divide({params})"
    elif tns_type == "subscript":
        # General subscripting/slicing operation
        # subscript_indices contains the slice pattern as a string (e.g., "[-1]", "[:, -1, :]")
        ts_op_synt = f"{prev_out_var}{tensorop.subscript_indices}"
    elif tns_type == "shape_dim":
        # Extract shape dimension (e.g., b = x.size(0))
        # Get input tensor from layers_of_tensors, not prev_out_var
        tensors = tensorop.layers_of_tensors
        if isinstance(tensors[0], str):
            if tensors[0] == 'INPUT':
                source_var = 'x'
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
        ts_op_synt = f"{source_var}.size({dim_index})"
    else:
        ts_op_synt = f"torch.matmul({params})"
    return ts_op_synt


def adjust_actv_func_name(modules_details: dict):
    """
    Renames activation functions as activ_func_1, activ_func_2, ...

    Parameters:
        modules_details (dict): A dict storing the NN modules syntax and
            attributes.

    Returns:
        None, but stores the activation function syntax in the dictionary.
    """
    actv_dict = {}
    counter = 1
    for mdl_name, mdl_details in modules_details.items():
        if mdl_name.split("_")[-1] == "activ":
            synt = mdl_details[0]
            if "get_activation_function" in synt:
                activ_type = synt.split("(")[1].split(")")[0]
                if activ_type not in actv_dict:
                    actv_dict[activ_type] = f"activ_func_{counter}"
                    counter+=1
                activ_def = synt.split("=")[1]
                mdl_details[0] = f"self.{actv_dict[activ_type]} = {activ_def}"


def get_activation_function(activ: str):
    """
    It returns the activation function dynamically if the user does not
    explicitely provide the activation function name in the BUML model.

    Arguments:
        activ (str): The name of the activation function.

    Returns:
        The activation function
    """
    activ_func = {"relu": "ReLU", "leaky_relu": "LeakyReLU",
                  "sigmoid": "Sigmoid", "softmax": "Softmax", "tanh": "Tanh"}
    activ = activ.lower()

    if activ in activ_func:
        return f"nn.{activ_func[activ]}()"
    raise ValueError(f"The activation function {activ} is invalid")


def get_rnn_hidden_var(layer_details, base_module):
    """
    Get the correct variable name for RNN hidden state in PyTorch.

    For return_type="both": hidden var is at index 4 (e.g., x_1_h)
    For return_type="hidden": hidden var is at index 1 after [-1] extraction (e.g., x_1)

    Arguments:
        layer_details: The layer details from modules_details
        base_module: The base module name (e.g., "rnn")

    Returns:
        The variable name to use for the hidden state
    """
    layer_obj = layer_details[3] if len(layer_details) > 3 else None

    if layer_obj and hasattr(layer_obj, 'return_type'):
        if layer_obj.return_type == "both" and len(layer_details) > 4:
            # Hidden variable is separate (x_1_h)
            return layer_details[4]
        else:
            # For return_type="hidden", hidden is in index 1 after [-1] extraction
            return layer_details[1]
    elif len(layer_details) > 4:
        return layer_details[4]
    else:
        # Fallback to regular output if hidden var not available
        return layer_details[1]
