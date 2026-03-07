"""
Neural Network Model Builder: Generates Python code for BESSER NN models.
"""

from besser.BUML.metamodel.nn import (
    NN, Configuration, TensorOp,
    Conv1D, Conv2D, Conv3D, PoolingLayer,
    SimpleRNNLayer, LSTMLayer, GRULayer,
    LinearLayer, FlattenLayer, EmbeddingLayer,
    DropoutLayer, LayerNormLayer, BatchNormLayer,
)


def _is_attr_set(layer, attr_name: str) -> bool:
    """Check if a boolean attribute was explicitly set (ticked) in the editor."""
    return attr_name in getattr(layer, '_set_attrs', set())


def _collect_used_types(model: NN, used_types: set = None) -> set:
    """
    Recursively collect all types used in the model (including sub_nns).

    Args:
        model: The NN model to analyze.
        used_types: Set to accumulate used types.

    Returns:
        Set of class names used in the model.
    """
    if used_types is None:
        used_types = set()

    # Always need NN
    used_types.add('NN')

    # Check for Configuration
    if model.configuration:
        used_types.add('Configuration')

    # Check modules (layers and tensor_ops)
    for module in model.modules:
        module_type = module.__class__.__name__
        if module_type == 'NN':
            # Recursively collect from sub_nn
            _collect_used_types(module, used_types)
        else:
            used_types.add(module_type)

    # Also check sub_nns list directly
    for sub_nn in model.sub_nns:
        _collect_used_types(sub_nn, used_types)

    return used_types


def nn_model_to_code(model: NN, file_path: str):
    """
    Generates Python code for an NN model, including any sub_nns.

    The generated code follows the alexnet.py pattern:
    1. First, sub_nns are defined as standalone NN objects
    2. Then the main NN is created with add_sub_nn() calls

    Args:
        model (NN): The neural network model.
        file_path (str): The path to save the generated code.
    """
    # Collect only the types actually used in the model
    used_types = _collect_used_types(model)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Generate imports for only the used types
        f.write("from besser.BUML.metamodel.nn import (\n")
        f.write(f"    {', '.join(sorted(used_types))},\n")
        f.write(")\n\n")

        # Track variable names for sub_nns
        sub_nn_vars = {}

        # First, write all sub_nns (recursively)
        if model.sub_nns:
            _write_all_sub_nns(f, model.sub_nns, sub_nn_vars)

        # Main NN model
        main_var = _name_to_var(model.name)
        f.write(f"# Neural Network: {model.name}\n")
        f.write(f"{main_var} = NN(name='{model.name}')\n")

        # Add modules in order (sub_nns, layers, tensor_ops)
        # Using model.modules preserves the order from the diagram's NNNext relationships
        if model.modules:
            layer_counter = 0
            tensor_op_counter = 0
            f.write("# Modules (in order)\n")
            for module in model.modules:
                module_type = module.__class__.__name__
                if module_type == "NN":
                    # Sub-NN
                    sub_var = sub_nn_vars.get(module.name, _name_to_var(module.name))
                    f.write(f"{main_var}.add_sub_nn({sub_var})\n\n")
                elif module_type == "TensorOp":
                    # TensorOp
                    tensor_op_var = f"tensor_op_{tensor_op_counter}"
                    _write_tensor_op(f, module, tensor_op_var)
                    f.write(f"{main_var}.add_tensor_op({tensor_op_var})\n\n")
                    tensor_op_counter += 1
                else:
                    # Layer
                    layer_var = f"layer_{layer_counter}"
                    _write_layer(f, module, layer_var)
                    f.write(f"{main_var}.add_layer({layer_var})\n\n")
                    layer_counter += 1

        # Configuration
        if model.configuration:
            f.write("# Configuration\n")
            _write_configuration(f, model.configuration, "config")
            f.write(f"{main_var}.add_configuration(config)\n\n")


def _name_to_var(name: str) -> str:
    """Convert an NN name to a valid Python variable name."""
    # Replace spaces and hyphens with underscores, lowercase
    var = name.replace(' ', '_').replace('-', '_').lower()
    # Ensure it starts with a letter
    if var and var[0].isdigit():
        var = 'nn_' + var
    return var if var else 'nn_model'


def _write_all_sub_nns(f, sub_nns: list, sub_nn_vars: dict, written: set = None):
    """
    Recursively write all sub_nns, ensuring dependencies are written first.

    Args:
        f: File handle
        sub_nns: List of sub NN models
        sub_nn_vars: Dict to track variable names (name -> var_name)
        written: Set of already written NN names
    """
    if written is None:
        written = set()

    for sub_nn in sub_nns:
        if sub_nn.name in written:
            continue

        # First, recursively write any nested sub_nns
        if sub_nn.sub_nns:
            _write_all_sub_nns(f, sub_nn.sub_nns, sub_nn_vars, written)

        # Now write this sub_nn
        var_name = _name_to_var(sub_nn.name)
        sub_nn_vars[sub_nn.name] = var_name

        f.write(f"# Sub-Network: {sub_nn.name}\n")
        f.write(f"{var_name} = NN(name='{sub_nn.name}')\n")

        # Add modules in order (preserves NNNext relationship order)
        if sub_nn.modules:
            layer_counter = 0
            tensor_op_counter = 0
            for module in sub_nn.modules:
                module_type = module.__class__.__name__
                if module_type == "NN":
                    # Nested sub-NN
                    nested_var = sub_nn_vars.get(module.name, _name_to_var(module.name))
                    f.write(f"{var_name}.add_sub_nn({nested_var})\n")
                elif module_type == "TensorOp":
                    # TensorOp
                    tensor_op_var = f"{var_name}_tensor_op_{tensor_op_counter}"
                    _write_tensor_op(f, module, tensor_op_var)
                    f.write(f"{var_name}.add_tensor_op({tensor_op_var})\n")
                    tensor_op_counter += 1
                else:
                    # Layer
                    layer_var = f"{var_name}_layer_{layer_counter}"
                    _write_layer(f, module, layer_var)
                    f.write(f"{var_name}.add_layer({layer_var})\n")
                    layer_counter += 1

        f.write("\n")
        written.add(sub_nn.name)


def _write_layer(f, layer, var_name: str):
    """Write a layer definition."""
    if isinstance(layer, Conv1D):
        _write_conv1d(f, layer, var_name)
    elif isinstance(layer, Conv2D):
        _write_conv2d(f, layer, var_name)
    elif isinstance(layer, Conv3D):
        _write_conv3d(f, layer, var_name)
    elif isinstance(layer, PoolingLayer):
        _write_pooling(f, layer, var_name)
    elif isinstance(layer, SimpleRNNLayer):
        _write_rnn(f, layer, var_name)
    elif isinstance(layer, LSTMLayer):
        _write_lstm(f, layer, var_name)
    elif isinstance(layer, GRULayer):
        _write_gru(f, layer, var_name)
    elif isinstance(layer, LinearLayer):
        _write_linear(f, layer, var_name)
    elif isinstance(layer, FlattenLayer):
        _write_flatten(f, layer, var_name)
    elif isinstance(layer, EmbeddingLayer):
        _write_embedding(f, layer, var_name)
    elif isinstance(layer, DropoutLayer):
        _write_dropout(f, layer, var_name)
    elif isinstance(layer, LayerNormLayer):
        _write_layer_norm(f, layer, var_name)
    elif isinstance(layer, BatchNormLayer):
        _write_batch_norm(f, layer, var_name)
    else:
        f.write(f"# Unknown layer type: {type(layer).__name__}\n")


def _format_optional(name: str, value, default=None, quote_str=True):
    """Format an optional parameter for output."""
    if value is None or value == default:
        return ""
    if isinstance(value, str) and quote_str:
        return f", {name}='{value}'"
    return f", {name}={value}"


def _write_conv1d(f, layer: Conv1D, var_name: str):
    """Write Conv1D layer definition."""
    params = [
        f"name='{layer.name}'",
        f"kernel_dim={layer.kernel_dim}",
        f"out_channels={layer.out_channels}",
    ]
    if layer.in_channels:
        params.append(f"in_channels={layer.in_channels}")
    if layer.stride_dim:
        params.append(f"stride_dim={layer.stride_dim}")
    if layer.padding_amount:
        params.append(f"padding_amount={layer.padding_amount}")
    if layer.padding_type and layer.padding_type != "valid":
        params.append(f"padding_type='{layer.padding_type}'")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")
    if layer.permute_in:
        params.append(f"permute_in={layer.permute_in}")
    if layer.permute_out:
        params.append(f"permute_out={layer.permute_out}")

    f.write(f"{var_name} = Conv1D({', '.join(params)})\n")


def _write_conv2d(f, layer: Conv2D, var_name: str):
    """Write Conv2D layer definition."""
    params = [
        f"name='{layer.name}'",
        f"kernel_dim={layer.kernel_dim}",
        f"out_channels={layer.out_channels}",
    ]
    if layer.in_channels:
        params.append(f"in_channels={layer.in_channels}")
    if layer.stride_dim:
        params.append(f"stride_dim={layer.stride_dim}")
    if layer.padding_amount:
        params.append(f"padding_amount={layer.padding_amount}")
    if layer.padding_type and layer.padding_type != "valid":
        params.append(f"padding_type='{layer.padding_type}'")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")
    if layer.permute_in:
        params.append(f"permute_in={layer.permute_in}")
    if layer.permute_out:
        params.append(f"permute_out={layer.permute_out}")

    f.write(f"{var_name} = Conv2D({', '.join(params)})\n")


def _write_conv3d(f, layer: Conv3D, var_name: str):
    """Write Conv3D layer definition."""
    params = [
        f"name='{layer.name}'",
        f"kernel_dim={layer.kernel_dim}",
        f"out_channels={layer.out_channels}",
    ]
    if layer.in_channels:
        params.append(f"in_channels={layer.in_channels}")
    if layer.stride_dim:
        params.append(f"stride_dim={layer.stride_dim}")
    if layer.padding_amount:
        params.append(f"padding_amount={layer.padding_amount}")
    if layer.padding_type and layer.padding_type != "valid":
        params.append(f"padding_type='{layer.padding_type}'")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")
    if layer.permute_in:
        params.append(f"permute_in={layer.permute_in}")
    if layer.permute_out:
        params.append(f"permute_out={layer.permute_out}")

    f.write(f"{var_name} = Conv3D({', '.join(params)})\n")


def _write_pooling(f, layer: PoolingLayer, var_name: str):
    """Write PoolingLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"pooling_type='{layer.pooling_type}'",
        f"dimension='{layer.dimension}'",
    ]
    if layer.kernel_dim:
        params.append(f"kernel_dim={layer.kernel_dim}")
    if layer.stride_dim:
        params.append(f"stride_dim={layer.stride_dim}")
    if layer.padding_amount:
        params.append(f"padding_amount={layer.padding_amount}")
    if layer.padding_type and layer.padding_type != "valid":
        params.append(f"padding_type='{layer.padding_type}'")
    if layer.output_dim:
        params.append(f"output_dim={layer.output_dim}")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")
    if layer.permute_in:
        params.append(f"permute_in={layer.permute_in}")
    if layer.permute_out:
        params.append(f"permute_out={layer.permute_out}")

    f.write(f"{var_name} = PoolingLayer({', '.join(params)})\n")


def _write_rnn(f, layer: SimpleRNNLayer, var_name: str):
    """Write SimpleRNNLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"hidden_size={layer.hidden_size}",
    ]
    if layer.input_size:
        params.append(f"input_size={layer.input_size}")
    if layer.return_type:
        params.append(f"return_type='{layer.return_type}'")
    if _is_attr_set(layer, 'bidirectional'):
        params.append(f"bidirectional={layer.bidirectional}")
    if layer.dropout:
        params.append(f"dropout={layer.dropout}")
    if _is_attr_set(layer, 'batch_first'):
        params.append(f"batch_first={layer.batch_first}")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = SimpleRNNLayer({', '.join(params)})\n")


def _write_lstm(f, layer: LSTMLayer, var_name: str):
    """Write LSTMLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"hidden_size={layer.hidden_size}",
    ]
    if layer.input_size:
        params.append(f"input_size={layer.input_size}")
    if layer.return_type:
        params.append(f"return_type='{layer.return_type}'")
    if _is_attr_set(layer, 'bidirectional'):
        params.append(f"bidirectional={layer.bidirectional}")
    if layer.dropout:
        params.append(f"dropout={layer.dropout}")
    if _is_attr_set(layer, 'batch_first'):
        params.append(f"batch_first={layer.batch_first}")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = LSTMLayer({', '.join(params)})\n")


def _write_gru(f, layer: GRULayer, var_name: str):
    """Write GRULayer definition."""
    params = [
        f"name='{layer.name}'",
        f"hidden_size={layer.hidden_size}",
    ]
    if layer.input_size:
        params.append(f"input_size={layer.input_size}")
    if layer.return_type:
        params.append(f"return_type='{layer.return_type}'")
    if _is_attr_set(layer, 'bidirectional'):
        params.append(f"bidirectional={layer.bidirectional}")
    if layer.dropout:
        params.append(f"dropout={layer.dropout}")
    if _is_attr_set(layer, 'batch_first'):
        params.append(f"batch_first={layer.batch_first}")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = GRULayer({', '.join(params)})\n")


def _write_linear(f, layer: LinearLayer, var_name: str):
    """Write LinearLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"out_features={layer.out_features}",
    ]
    if layer.in_features:
        params.append(f"in_features={layer.in_features}")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = LinearLayer({', '.join(params)})\n")


def _write_flatten(f, layer: FlattenLayer, var_name: str):
    """Write FlattenLayer definition."""
    params = [f"name='{layer.name}'"]
    if layer.start_dim is not None and layer.start_dim != 1:
        params.append(f"start_dim={layer.start_dim}")
    if layer.end_dim is not None and layer.end_dim != -1:
        params.append(f"end_dim={layer.end_dim}")
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = FlattenLayer({', '.join(params)})\n")


def _write_embedding(f, layer: EmbeddingLayer, var_name: str):
    """Write EmbeddingLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"num_embeddings={layer.num_embeddings}",
        f"embedding_dim={layer.embedding_dim}",
    ]
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = EmbeddingLayer({', '.join(params)})\n")


def _write_dropout(f, layer: DropoutLayer, var_name: str):
    """Write DropoutLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"rate={layer.rate}",
    ]
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = DropoutLayer({', '.join(params)})\n")


def _write_layer_norm(f, layer: LayerNormLayer, var_name: str):
    """Write LayerNormLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"normalized_shape={layer.normalized_shape}",
    ]
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = LayerNormLayer({', '.join(params)})\n")


def _write_batch_norm(f, layer: BatchNormLayer, var_name: str):
    """Write BatchNormLayer definition."""
    params = [
        f"name='{layer.name}'",
        f"num_features={layer.num_features}",
        f"dimension='{layer.dimension}'",
    ]
    if layer.actv_func:
        params.append(f"actv_func='{layer.actv_func}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{layer.name_module_input}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = BatchNormLayer({', '.join(params)})\n")


def _write_tensor_op(f, tensor_op: TensorOp, var_name: str):
    """Write TensorOp definition.

    Only outputs attributes relevant to the specific tns_type:
    - 'reshape': reshape_dim
    - 'concatenate': concatenate_dim + layers_of_tensors
    - 'multiply': layers_of_tensors
    - 'matmultiply': layers_of_tensors
    - 'permute': permute_dim
    - 'transpose': transpose_dim
    - input_reused is optional for all types
    """
    params = [
        f"name='{tensor_op.name}'",
        f"tns_type='{tensor_op.tns_type}'",
    ]

    tns_type = tensor_op.tns_type

    # Only output attributes relevant to the specific tns_type
    if tns_type == 'concatenate':
        if tensor_op.concatenate_dim is not None:
            params.append(f"concatenate_dim={tensor_op.concatenate_dim}")
        if tensor_op.layers_of_tensors:
            params.append(f"layers_of_tensors={tensor_op.layers_of_tensors}")
    elif tns_type in ('multiply', 'matmultiply'):
        if tensor_op.layers_of_tensors:
            params.append(f"layers_of_tensors={tensor_op.layers_of_tensors}")
    elif tns_type == 'reshape':
        if tensor_op.reshape_dim:
            params.append(f"reshape_dim={tensor_op.reshape_dim}")
    elif tns_type == 'transpose':
        if tensor_op.transpose_dim:
            params.append(f"transpose_dim={tensor_op.transpose_dim}")
    elif tns_type == 'permute':
        if tensor_op.permute_dim:
            params.append(f"permute_dim={tensor_op.permute_dim}")

    # input_reused is optional for all types - only output if explicitly set to True
    if tensor_op.input_reused:
        params.append(f"input_reused={tensor_op.input_reused}")

    f.write(f"{var_name} = TensorOp({', '.join(params)})\n")


def _write_configuration(f, config: Configuration, var_name: str):
    """Write Configuration definition."""
    params = [
        f"batch_size={config.batch_size}",
        f"epochs={config.epochs}",
        f"learning_rate={config.learning_rate}",
        f"optimizer='{config.optimizer}'",
        f"loss_function='{config.loss_function}'",
        f"metrics={config.metrics}",
    ]
    if config.weight_decay:
        params.append(f"weight_decay={config.weight_decay}")
    if config.momentum:
        params.append(f"momentum={config.momentum}")

    f.write(f"{var_name} = Configuration({', '.join(params)})\n")
