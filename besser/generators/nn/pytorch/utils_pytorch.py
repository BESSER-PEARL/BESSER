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
    def __init__(self, layer: Layer, modules_details: dict,
                 is_subnn: bool = False):
        self.layer: Layer = layer
        self.modules_details: dict = modules_details
        self.is_subnn: bool = is_subnn
        self.permute_out: bool | None = None
        self.permute_in: bool | None = None
        self.dim: str | None = None
        # Track shared activation layers
        if not hasattr(SetupLayerSyntax, '_shared_activations'):
            SetupLayerSyntax._shared_activations = {}

    def setup_general_layer(self):
        """It defines the syntax of general layers."""
        cls_name = self.layer.__class__.__name__
        lyr_name = self.layer.name
        lyr = f"self.{lyr_name} = nn"

        if cls_name == "LinearLayer":
            in_f = self.layer.in_features
            out_f = self.layer.out_features
            bias = self.layer.bias
            lyr = (
                f"{lyr}.Linear(in_features={in_f}, out_features={out_f}, "
                f"bias={bias})"
            )
        elif cls_name == "FlattenLayer":
            st_dim = self.layer.start_dim
            en_dim = self.layer.end_dim
            lyr = f"{lyr}.Flatten(start_dim={st_dim}, end_dim={en_dim})"
        else: # cls_name == "EmbeddingLayer"
            nm = self.layer.num_embeddings
            dm = self.layer.embedding_dim
            padding_idx = self.layer.padding_idx

            # Set permute flags for channel order conversion
            self.permute_out = self.layer.permute_out
            self.dim = "1"

            if padding_idx is not None:
                lyr = (
                    f"{lyr}.Embedding(num_embeddings={nm}, "
                    f"embedding_dim={dm}, padding_idx={padding_idx})"
                )
            else:
                lyr = (
                    f"{lyr}.Embedding(num_embeddings={nm}, "
                    f"embedding_dim={dm})"
                )
        return lyr

    def setup_standalone_activation(self, out_var, in_var,
                                    original_layer_name=None):
        """It defines the syntax for standalone activation layer."""
        actv_func = self.layer.actv_func
        lyr_name = self.layer.name

        # Map activation function names to PyTorch layer classes
        activs = {"relu": "ReLU", "leaky_relu": "LeakyReLU", 
                  "sigmoid": "Sigmoid", "softmax": "Softmax",
                  "tanh": "Tanh", "gelu": "GELU"}
        pytorch_actv = activs.get(actv_func, actv_func.capitalize())

        if actv_func == 'softmax':
            return f"self.{lyr_name} = nn.Softmax(dim=-1)"
        else:
            return f"self.{lyr_name} = nn.{pytorch_actv}()"

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
                track_stats = (
                    "True" if self.layer.track_running_stats else "False"
                )
                # Set permute flags for TF->PyTorch channel order conversion
                self.permute_in = self.layer.permute_in
                self.permute_out = self.layer.permute_out
                self.dim = dim

                lyr = (
                    f"{lyr}.BatchNorm{dim}d(num_features={num_f}, eps={eps}, "
                    f"momentum={momentum}, affine={affine}, "
                    f"track_running_stats={track_stats})"
                )
            else: # cls_name == "LayerNormLayer"
                norm_shape = self.layer.normalized_shape
                eps = self.layer.eps
                elementwise_affine = "True" if self.layer.affine else "False"
                lyr = (
                    f"{lyr}.LayerNorm(normalized_shape={norm_shape}, "
                    f"eps={eps}, elementwise_affine={elementwise_affine})"
                )
        else: # cls_name == "DropoutLayer"
            # Set permute flags for channel order conversion
            self.permute_out = self.layer.permute_out
            if hasattr(self.layer, 'dimension') and self.layer.dimension:
                # Strip 'D' suffix if present (e.g., '1D' -> '1', '2D' -> '2')
                self.dim = self.layer.dimension.rstrip('D')
            else:
                self.dim = "1"  # Default dimension for regular Dropout

            # Use Dropout1d/2d/3d for spatial variants
            if hasattr(self.layer, 'dimension') and self.layer.dimension:
                # Strip 'D' suffix if present (e.g., '1D' -> '1', '2D' -> '2')
                dim_num = self.layer.dimension.rstrip('D')
                lyr = f"{lyr}.Dropout{dim_num}d(p={self.layer.rate})"
            else:
                lyr = f"{lyr}.Dropout(p={self.layer.rate})"
        return lyr

    def _get_permute_dims(self, dim: str, permute_in: bool):
        """
        Calculate permute dimensions based on layer dimensionality.

        Args:
            dim (str): the dimentionality of the layer ('1', '2' or '3').
            permute_in (bool): Whether to permute the input of the layer.

        Returns:
            list: The permutation dimensions.
        """
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
        perm_name = f"{lyr_name}_{'in' if permute_in else 'out'}_op"
        perm_dim = self._get_permute_dims(dim, permute_in)

        if sequential or is_subnn:
            self.modules_details[perm_name] = [
                f"Permute(dims={perm_dim})", in_var_layer
            ]
        else:
            tns = TensorOp(name=perm_name, tns_type="permute",
                           permute_dim=perm_dim)
            utils.handle_tensorop(tns, self.modules_details,
                                  get_tensorop_syntax, in_var_layer)


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

        # For SimpleRNN, add nonlinearity parameter (internal activation)
        # actv_func in BUML maps to nonlinearity in PyTorch nn.RNN
        if (layer_type == "RNN" and hasattr(self.layer, 'actv_func')
            and self.layer.actv_func):
            nonlin = self.layer.actv_func
            lyr = (
                f"self.{lyr_name} = nn.{layer_type}(input_size={in_sz}, "
                f"hidden_size={h_sz}, nonlinearity='{nonlin}', "
                f"bidirectional={bd}, dropout={drp}, batch_first={btch}, "
                f"bias={bs})"
            )
            # Clear actv_func so setup_actv_func() won't create a
            # separate activation layer (activation is already handled 
            # internally by the nonlinearity parameter)
            self.layer.actv_func = None
        else:
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
                  "sigmoid": "Sigmoid", "softmax": "Softmax",
                  "tanh": "Tanh", "gelu": "GELU"}
        if hasattr(self.layer, 'actv_func'):
            actv = self.layer.actv_func
            if actv in activs:
                if actv == 'softmax':
                    lyr = f"self.actv_func_{actv} = nn.{activs[actv]}(dim=-1)"
                else:
                    lyr = f"self.actv_func_{actv} = nn.{activs[actv]}()"
            elif actv is not None and actv is not False:
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


    def _build_standard_pooling(self, lyr_name: str, pl_type: str, dim: str):
        """Build syntax for standard max or average pooling."""
        pl = "Max" if pl_type == "max" else "Avg"
        kernel = utils.format_value(self.layer.kernel_dim)
        stride = utils.format_value(self.layer.stride_dim)
        pad = self.layer.padding_amount
        return (
            f"self.{lyr_name} = nn.{pl}Pool{dim}d(kernel_size={kernel}, "
            f"stride={stride}, padding={pad})"
        )

    def _build_adaptive_pooling(self, lyr_name: str, pl_type: str, dim: str):
        """Build syntax for adaptive pooling layers."""
        if pl_type.startswith("global"):
            out_dim = (1,) * int(dim)
            return f"self.{lyr_name} = nn.AdaptiveAvgPool{dim}d({out_dim})"
        else:
            pl = (
                "AdaptiveAvg" if pl_type == "adaptive_average"
                else "AdaptiveMax"
            )
            size = utils.format_value(self.layer.output_dim)
            return f"self.{lyr_name} = nn.{pl}Pool{dim}d(output_size={size})"

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

        if pl_type in ("max", "average"):
            return self._build_standard_pooling(lyr_name, pl_type, dim)
        else:
            return self._build_adaptive_pooling(lyr_name, pl_type, dim)


def _handle_reshape_pytorch(tensorop, modules_details, in_var, prev_out_var,
                            params):
    """Handle reshape tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"{prev_out_var}.reshape({params})"


def _handle_split_pytorch(tensorop, modules_details, in_var, prev_out_var,
                          params):
    """Handle split tensorop syntax.

    Uses torch.chunk when split_sizes is an integer (number of chunks,
    e.g., from TF migration).
    torch.chunk(tensor, chunks, dim) splits into N equal chunks
    (like TF's num_or_size_splits).
    torch.split(tensor, size, dim) splits into chunks of SIZE
    (different semantics).
    """
    split_sizes = tensorop.split_sizes
    split_dim = tensorop.split_dim
    if in_var is not None:
        prev_out_var = in_var

    # Use torch.chunk for integer split_sizes (number of chunks)
    # This matches TF's tf.split(num_or_size_splits=N) behavior
    if isinstance(split_sizes, int):
        return f"torch.chunk({prev_out_var}, {split_sizes}, dim={split_dim})"
    else:
        # If split_sizes is a list, use torch.split with list of sizes
        return (
            f"torch.split({prev_out_var}, "
            f"split_size_or_sections={split_sizes}, dim={split_dim})"
        )

def _handle_concatenate_pytorch(tensorop, modules_details, in_var,
                                prev_out_var, params):
    """Handle concatenate tensorop syntax."""
    dim = tensorop.concatenate_dim
    return f"torch.cat(({params}), dim={dim})"


def _handle_transpose_pytorch(tensorop, modules_details, in_var, prev_out_var,
                              params):
    """Handle transpose tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"{prev_out_var}.transpose({params})"


def _handle_permute_pytorch(tensorop, modules_details, in_var, prev_out_var,
                            params):
    """Handle permute tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"{prev_out_var}.permute({params})"


def _handle_multiply_pytorch(tensorop, modules_details, in_var, prev_out_var,
                             params):
    """Handle multiply tensorop syntax."""
    return f"torch.mul({params})"


def _handle_mean_pytorch(tensorop, modules_details, in_var, prev_out_var,
                         params):
    """Handle mean tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    dim = tensorop.reduce_dim
    return f"{prev_out_var}.mean(dim={dim})"


def _handle_max_pytorch(tensorop, modules_details, in_var, prev_out_var,
                        params):
    """Handle max tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    if tensorop.reduce_dim is not None:
        dim = tensorop.reduce_dim
        keepdims = getattr(tensorop, 'reduce_keepdims', False)
        if keepdims:
            return f"{prev_out_var}.max(dim={dim}, keepdim=True)[0]"
        else:
            return f"{prev_out_var}.max(dim={dim})[0]"
    else:
        return f"{prev_out_var}.max()"


def _handle_zeros_like_pytorch(tensorop, modules_details, in_var,
                               prev_out_var, params):
    """Handle zeros_like tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"torch.zeros_like({prev_out_var})"


def _handle_squeeze_pytorch(tensorop, modules_details, in_var, prev_out_var,
                            params):
    """Handle squeeze tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    if tensorop.reduce_dim is not None:
        dim = tensorop.reduce_dim
        return f"{prev_out_var}.squeeze({dim})"
    else:
        return f"{prev_out_var}.squeeze()"


def _handle_unsqueeze_pytorch(tensorop, modules_details, in_var, prev_out_var,
                              params):
    """Handle unsqueeze tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    dim = tensorop.reduce_dim
    return f"{prev_out_var}.unsqueeze({dim})"


def _handle_normalize_pytorch(tensorop, modules_details, in_var, prev_out_var,
                              params):
    """Handle normalize tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    dim = tensorop.reduce_dim
    return f"F.normalize({prev_out_var}, dim={dim})"


def _handle_repeat_pytorch(tensorop, modules_details, in_var, prev_out_var,
                           params):
    """Handle repeat tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    return f"{prev_out_var}.repeat({params})"


def _handle_pad_pytorch(tensorop, modules_details, in_var, prev_out_var,
                        params):
    """Handle pad tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var

    pad_amount = tensorop.pad_amount
    pad_mode = (
        tensorop.pad_mode.lower() if hasattr(tensorop, 'pad_mode')
        else 'constant'
    )
    pad_value = (
        tensorop.pad_value 
        if hasattr(tensorop, 'pad_value') and tensorop.pad_value is not None
        else 0
    )

    # pad_amount is nested list: [[left, right], [top, bottom], ...]
    # PyTorch F.pad expects flat tuple in reverse dimension order
    # For 2D: [[left, right], [top, bottom]] -> 
    # (left, right, top, bottom)
    pt_pad = []
    for dim_pad in pad_amount:
        if isinstance(dim_pad, list) and len(dim_pad) == 2:
            pt_pad.extend(dim_pad)

    pad_tuple = tuple(pt_pad)

    # Only include value parameter for constant mode
    if pad_mode == 'constant' and pad_value != 0:
        return (
            f"F.pad({prev_out_var}, {pad_tuple}, mode='{pad_mode}', "
            f"value={pad_value})"
        )
    else:
        return f"F.pad({prev_out_var}, {pad_tuple}, mode='{pad_mode}')"


def _handle_dropout_pytorch(tensorop, modules_details, in_var, prev_out_var,
                            params):
    """Handle dropout tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var
    dropout_rate = (
        tensorop.dropout_rate if hasattr(tensorop, 'dropout_rate') else 0.5
    )
    # Handle dropout_training_aware attribute
    if hasattr(tensorop, 'dropout_training_aware'):
        if tensorop.dropout_training_aware is True:
            training_arg = "training=self.training"
        elif tensorop.dropout_training_aware is False:
            training_arg = "training=True"
        else:  # None
            raise ValueError(
                f"PyTorch generation does not support dropout tensorop"
                f" '{tensorop.name}' with dropout_training_aware=None. "
                "Set it to True or False."
            )
        return f"F.dropout({prev_out_var}, p={dropout_rate}, {training_arg})"
    else:
        return f"F.dropout({prev_out_var}, p={dropout_rate})"


def _handle_interpolate_pytorch(tensorop, modules_details, in_var,
                                prev_out_var, params):
    """Handle interpolate tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var


    size  = getattr(tensorop, 'interpolate_size', None)
    scale = getattr(tensorop, 'interpolate_scale', None)
    mode  = getattr(tensorop, 'interpolate_mode', 'bilinear')

    # PyTorch supported modes
    pytorch_modes = {'nearest', 'linear', 'bilinear', 'bicubic',
                     'trilinear', 'area', 'nearest-exact'}

    # Warn if mode not supported in PyTorch
    if mode not in pytorch_modes:
        print(
            f"Warning: interpolate mode '{mode}' not supported in PyTorch, "
            f"using 'bilinear' instead"
        )
        mode = 'bilinear'

    if size is not None:
        return f"F.interpolate({prev_out_var}, size={size}, mode='{mode}')"
    elif scale is not None:
        return (
            f"F.interpolate({prev_out_var}, scale_factor={scale}, "
            f"mode='{mode}')"
        )
    else:
        raise ValueError(
            "interpolate tensorop requires either interpolate_size "
            "or interpolate_scale"
        )

def _handle_binop_add_pytorch(tensorop, modules_details, in_var, prev_out_var,
                              params):
    """Handle binop_add tensorop syntax."""
    return (
        f"{params.split(', ')[0]} + {params.split(', ')[1]}"
        if ', ' in params
        else f"torch.add({params})"
    )

def _handle_binop_subtract_pytorch(tensorop, modules_details, in_var,
                                   prev_out_var, params):
    """Handle binop_subtract tensorop syntax."""
    return (
        f"{params.split(', ')[0]} - {params.split(', ')[1]}"
        if ', ' in params
        else f"torch.subtract({params})"
    )

def _handle_binop_multiply_pytorch(tensorop, modules_details, in_var,
                                   prev_out_var, params):
    """Handle binop_multiply tensorop syntax."""
    return (
        f"{params.split(', ')[0]} * {params.split(', ')[1]}"
        if ', ' in params
        else f"torch.multiply({params})"
    )

def _handle_binop_divide_pytorch(tensorop, modules_details, in_var,
                                 prev_out_var, params):
    """Handle binop_divide tensorop syntax."""
    return (
        f"{params.split(', ')[0]} / {params.split(', ')[1]}"
        if ', ' in params
        else f"torch.divide({params})"
    )

def _handle_binop_floor_divide_pytorch(tensorop, modules_details, in_var, prev_out_var, params):
    """Handle binop_floor_divide tensorop syntax."""
    return (f"{params.split(', ')[0]} // {params.split(', ')[1]}" if ', ' in params
            else f"torch.floor_divide({params})")

def _handle_subscript_pytorch(tensorop, modules_details, in_var,
                              prev_out_var, params):
    """Handle subscript tensorop syntax."""
    if in_var is not None:
        prev_out_var = in_var

    # Build subscript string from structured list
    def build_subscript_string(indices):
        elements = []
        for elem in indices:
            if elem["type"] == "slice":
                start = (
                    str(elem["start"]) if elem["start"] is not None else ""
                )
                stop = (
                    str(elem["stop"]) if elem["stop"] is not None else ""
                )
                step = (str(elem["step"]) if elem["step"] is not None else ""
                )
                if step:
                    elements.append(f"{start}:{stop}:{step}")
                else:
                    elements.append(
                        f"{start}:{stop}" if start or stop else ":"
                    )
            elif elem["type"] == "index":
                elements.append(str(elem["value"]))
        return "[" + ", ".join(elements) + "]"

    subscript_str = build_subscript_string(tensorop.subscript_indices)
    return f"{prev_out_var}{subscript_str}"


def _handle_shape_dim_pytorch(tensorop, modules_details, in_var, prev_out_var,
                              params):
    """Handle shape_dim tensorop syntax."""
    tensors = tensorop.layers_of_tensors
    if isinstance(tensors[0], str):
        if tensors[0] == 'INPUT':
            source_var = 'x'
        else:
            if (f"{tensors[0]}_layer" in modules_details
                or f"{tensors[0]}_op" in modules_details):
                source_tensors = utils.get_layers_output_for_tensorops(
                    tensors, modules_details
                )
                source_var = source_tensors[0]
            else:
                source_var = tensors[0]
    else:
        # Scalar value - render as int if whole number
        scalar = tensors[0]
        source_var = str(int(scalar)) if scalar.is_integer() else str(scalar)

    dim_index = tensorop.reduce_dim
    return f"{source_var}.size({dim_index})"


def _handle_default_pytorch(tensorop, modules_details, in_var, prev_out_var,
                            params):
    """Handle default case (matmul)."""
    return f"torch.matmul({params})"


def _handle_identity_pytorch(tensorop, modules_details, in_var, prev_out_var,
                             params):
    """Handle identity operation (variable assignment like residual = x)."""
    # For identity, just return the input variable
    # The input variable is determined by get_input_var
    if in_var is not None:
        return in_var
    return prev_out_var


# Dispatch table for PyTorch tensorop handlers
_PYTORCH_TENSOROP_HANDLERS = {
    "reshape": _handle_reshape_pytorch,
    "split": _handle_split_pytorch,
    "concatenate": _handle_concatenate_pytorch,
    "transpose": _handle_transpose_pytorch,
    "permute": _handle_permute_pytorch,
    "multiply": _handle_multiply_pytorch,
    "mean": _handle_mean_pytorch,
    "max": _handle_max_pytorch,
    "zeros_like": _handle_zeros_like_pytorch,
    "squeeze": _handle_squeeze_pytorch,
    "unsqueeze": _handle_unsqueeze_pytorch,
    "normalize": _handle_normalize_pytorch,
    "repeat": _handle_repeat_pytorch,
    "pad": _handle_pad_pytorch,
    "dropout": _handle_dropout_pytorch,
    "interpolate": _handle_interpolate_pytorch,
    "binop_add": _handle_binop_add_pytorch,
    "binop_subtract": _handle_binop_subtract_pytorch,
    "binop_multiply": _handle_binop_multiply_pytorch,
    "binop_divide": _handle_binop_divide_pytorch,
    "binop_floor_divide": _handle_binop_floor_divide_pytorch,
    "subscript": _handle_subscript_pytorch,
    "shape_dim": _handle_shape_dim_pytorch,
    "identity": _handle_identity_pytorch,
}


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

    tns_type = tensorop.tns_type
    handler = _PYTORCH_TENSOROP_HANDLERS.get(tns_type,
                                             _handle_default_pytorch)
    return handler(tensorop, modules_details, in_var, prev_out_var, params)


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
                    counter += 1
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
    For return_type="hidden": hidden var is at index 1 after [-1] 
    extraction (e.g., x_1)

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
            # For return_type="hidden", hidden is in index 1 
            # after [-1] extraction
            return layer_details[1]
    elif len(layer_details) > 4:
        return layer_details[4]
    else:
        # Fallback to regular output if hidden var not available
        return layer_details[1]
