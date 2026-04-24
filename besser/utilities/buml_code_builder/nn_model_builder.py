"""
Neural Network Model Builder: Generates Python code for BESSER NN models.
"""

from besser.BUML.metamodel.nn import (
    NN, Configuration, TensorOp,
    Conv1D, Conv2D, Conv3D, PoolingLayer,
    SimpleRNNLayer, LSTMLayer, GRULayer,
    LinearLayer, FlattenLayer, EmbeddingLayer,
    DropoutLayer, LayerNormLayer, BatchNormLayer,
    Dataset,
)
from besser.utilities.buml_code_builder.common import _escape_python_string, safe_var_name
from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit


def _is_attr_set(layer, attr_name: str) -> bool:
    """Check if an attribute was explicitly set (ticked/entered) in the editor."""
    return is_explicit(layer, attr_name)


def _esc(value) -> str:
    """Escape a user-controlled string for safe interpolation into a single-
    quoted Python literal.

    Guards against code injection when the generated file is later `exec()`'d
    (the documented contract for BUML builder output). Returns an empty string
    for ``None`` so callers always get a string.
    """
    if value is None:
        return ''
    return _escape_python_string(str(value))


def _fmt_metrics(metrics) -> str:
    """Format a list of metric names as a Python list literal, escaping each
    entry so user-controlled strings cannot break out of the quotes."""
    if not metrics:
        return '[]'
    escaped = [f"'{_esc(m)}'" for m in metrics]
    return '[' + ', '.join(escaped) + ']'


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

    # Check for datasets
    for ds in (getattr(model, 'train_data', None), getattr(model, 'test_data', None)):
        if ds is not None:
            used_types.add('Dataset')
            if getattr(ds, 'image', None) is not None:
                used_types.add('Image')

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


def nn_model_to_code(model: NN, file_path: str, model_var_name: str = None):
    """
    Generates Python code for an NN model, including any sub_nns.

    The generated code follows the alexnet.py pattern:
    1. First, sub_nns are defined as standalone NN objects
    2. Then the main NN is created with add_sub_nn() calls

    Args:
        model (NN): The neural network model.
        file_path (str): The path to save the generated code.
        model_var_name (str, optional): Override the main NN's Python variable
            name (used by ``project_to_code`` so multiple NNs in one project
            get unique suffixes — ``my_nn_1`` / ``my_nn_2`` — instead of
            colliding on the same ``my_nn`` binding).
    """
    # Collect only the types actually used in the model
    used_types = _collect_used_types(model)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Generate imports for only the used types
        f.write("from besser.BUML.metamodel.nn import (\n")
        f.write(f"    {', '.join(sorted(used_types))},\n")
        f.write(")\n\n")

        # Main NN's variable name — caller can override for project-level
        # disambiguation; otherwise derive from the model's name.
        main_var = model_var_name if model_var_name else _name_to_var(model.name)

        # Track which variable names are already bound in this file so that
        # sub-NNs never collide with the main NN, sibling sub-NNs, layers,
        # tensor_ops, config, or datasets. Seed with the reserved prefixes
        # used below and with the main NN's own var name.
        used_names = {main_var, 'image', 'config', 'train_data', 'test_data'}

        # Track variable names for sub_nns keyed by id(sub_nn) — two sibling
        # sub-NNs with the same display name are still distinct objects and
        # must get distinct variable names, or the second silently clobbers
        # the first when emitted as ``add_sub_nn(<var>)``.
        sub_nn_vars = {}

        # First, write all sub_nns (recursively)
        if model.sub_nns:
            _write_all_sub_nns(f, model.sub_nns, sub_nn_vars, used_names=used_names,
                               name_prefix=main_var)

        f.write(f"# Neural Network: {model.name}\n")
        f.write(f"{main_var} = NN(name='{_esc(model.name)}')\n")

        # Add modules in order (sub_nns, layers, tensor_ops)
        # Using model.modules preserves the order from the diagram's NNNext relationships
        if model.modules:
            layer_counter = 0
            tensor_op_counter = 0
            f.write("# Modules (in order)\n")
            for module in model.modules:
                module_type = module.__class__.__name__
                if module_type == "NN":
                    # Sub-NN — resolve by identity, not by name, so two
                    # siblings with the same display name still get their
                    # own bindings.
                    sub_var = sub_nn_vars.get(id(module))
                    if sub_var is None:
                        # Fall back to a fresh unique name (shouldn't happen
                        # if _write_all_sub_nns ran; defensive only).
                        sub_var = _unique_var(_name_to_var(module.name), used_names)
                    f.write(f"{main_var}.add_sub_nn({sub_var})\n\n")
                elif module_type == "TensorOp":
                    tensor_op_var = _unique_var(f"tensor_op_{tensor_op_counter}", used_names)
                    _write_tensor_op(f, module, tensor_op_var)
                    f.write(f"{main_var}.add_tensor_op({tensor_op_var})\n\n")
                    tensor_op_counter += 1
                else:
                    layer_var = _unique_var(f"layer_{layer_counter}", used_names)
                    _write_layer(f, module, layer_var)
                    f.write(f"{main_var}.add_layer({layer_var})\n\n")
                    layer_counter += 1

        # Configuration
        if model.configuration:
            f.write("# Configuration\n")
            _write_configuration(f, model.configuration, "config")
            f.write(f"{main_var}.add_configuration(config)\n\n")

        # Datasets
        if getattr(model, 'train_data', None) is not None:
            f.write("# Training dataset\n")
            _write_dataset(f, model.train_data, "train_data")
            f.write(f"{main_var}.add_train_data(train_data)\n\n")
        if getattr(model, 'test_data', None) is not None:
            f.write("# Test dataset\n")
            _write_dataset(f, model.test_data, "test_data")
            f.write(f"{main_var}.add_test_data(test_data)\n\n")


def _name_to_var(name: str) -> str:
    """Convert an NN name to a valid Python variable name.

    Delegates to :func:`safe_var_name` in ``common.py`` so Python keywords,
    non-identifier characters, leading digits, consecutive underscores, and
    empty strings are all handled consistently. Previously this helper only
    replaced spaces/hyphens, which meant a layer named ``class`` or
    ``my.net`` produced invalid Python at ``exec()`` time.
    """
    return safe_var_name(name) if name else 'nn_model'


def _unique_var(base: str, used_names: set) -> str:
    """Return ``base`` (or ``base_2``, ``base_3``, …) not already in ``used_names``.

    Mutates ``used_names`` to include the returned value so subsequent calls
    see it as taken. Needed because two siblings with colliding safe names
    (``my-net`` and ``my_net`` both map to ``my_net``) would otherwise clobber
    each other when emitted as top-level variable bindings.
    """
    if base not in used_names:
        used_names.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used_names:
        i += 1
    picked = f"{base}_{i}"
    used_names.add(picked)
    return picked


def _write_all_sub_nns(f, sub_nns: list, sub_nn_vars: dict, written: set = None,
                       used_names: set = None, name_prefix: str = None):
    """
    Recursively write all sub_nns, ensuring dependencies are written first.

    Args:
        f: File handle
        sub_nns: List of sub NN models
        sub_nn_vars: Dict to track variable names, keyed by ``id(sub_nn)`` so
            two siblings with the same display name still get distinct vars.
        written: Set of ``id(sub_nn)`` values already written (also keyed by
            identity — two distinct sub-NNs with the same name must both be
            emitted, not collapsed).
        used_names: Set of variable names already bound in this file. New
            sub-NN vars are chosen via :func:`_unique_var` so they never
            shadow the main NN's bindings, layer counters, or each other.
        name_prefix: Optional prefix (typically the main NN's var name) to
            scope sub-NN variables. Prevents cross-NN collisions when
            ``project_to_code`` concatenates multiple NNs that each contain
            a sub-NN with the same display name.
    """
    if written is None:
        written = set()
    if used_names is None:
        used_names = set()

    for sub_nn in sub_nns:
        if id(sub_nn) in written:
            continue

        # First, recursively write any nested sub_nns
        if sub_nn.sub_nns:
            _write_all_sub_nns(f, sub_nn.sub_nns, sub_nn_vars, written,
                               used_names=used_names, name_prefix=name_prefix)

        # Pick a unique variable name for this sub-NN
        base_var = _name_to_var(sub_nn.name)
        if name_prefix and name_prefix != base_var:
            base_var = f"{name_prefix}_{base_var}"
        var_name = _unique_var(base_var, used_names)
        sub_nn_vars[id(sub_nn)] = var_name

        f.write(f"# Sub-Network: {sub_nn.name}\n")
        f.write(f"{var_name} = NN(name='{_esc(sub_nn.name)}')\n")

        # Add modules in order (preserves NNNext relationship order)
        if sub_nn.modules:
            layer_counter = 0
            tensor_op_counter = 0
            for module in sub_nn.modules:
                module_type = module.__class__.__name__
                if module_type == "NN":
                    # Nested sub-NN — resolve by identity
                    nested_var = sub_nn_vars.get(id(module))
                    if nested_var is None:
                        nested_var = _unique_var(_name_to_var(module.name), used_names)
                    f.write(f"{var_name}.add_sub_nn({nested_var})\n")
                elif module_type == "TensorOp":
                    tensor_op_var = _unique_var(f"{var_name}_tensor_op_{tensor_op_counter}", used_names)
                    _write_tensor_op(f, module, tensor_op_var)
                    f.write(f"{var_name}.add_tensor_op({tensor_op_var})\n")
                    tensor_op_counter += 1
                else:
                    layer_var = _unique_var(f"{var_name}_layer_{layer_counter}", used_names)
                    _write_layer(f, module, layer_var)
                    f.write(f"{var_name}.add_layer({layer_var})\n")
                    layer_counter += 1

        f.write("\n")
        written.add(id(sub_nn))


def _write_layer(f, layer, var_name: str):
    """Write a layer definition."""
    if isinstance(layer, (Conv1D, Conv2D, Conv3D)):
        _write_conv(f, layer, var_name)
    elif isinstance(layer, PoolingLayer):
        _write_pooling(f, layer, var_name)
    elif isinstance(layer, (SimpleRNNLayer, LSTMLayer, GRULayer)):
        _write_rnn_like(f, layer, var_name)
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


def _write_conv(f, layer, var_name: str):
    """Write a convolutional layer (Conv1D, Conv2D, or Conv3D) definition."""
    class_name = type(layer).__name__
    params = [
        f"name='{_esc(layer.name)}'",
        f"kernel_dim={layer.kernel_dim}",
        f"out_channels={layer.out_channels}",
    ]
    if layer.in_channels is not None:
        params.append(f"in_channels={layer.in_channels}")
    if layer.stride_dim is not None:
        params.append(f"stride_dim={layer.stride_dim}")
    if _is_attr_set(layer, 'padding_amount') or layer.padding_amount:
        params.append(f"padding_amount={layer.padding_amount}")
    if _is_attr_set(layer, 'padding_type') or (
        layer.padding_type and layer.padding_type != "valid"
    ):
        params.append(f"padding_type='{_esc(layer.padding_type)}'")
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")
    if _is_attr_set(layer, 'permute_in'):
        params.append(f"permute_in={layer.permute_in}")
    if _is_attr_set(layer, 'permute_out'):
        params.append(f"permute_out={layer.permute_out}")

    f.write(f"{var_name} = {class_name}({', '.join(params)})\n")


def _write_pooling(f, layer: PoolingLayer, var_name: str):
    """Write PoolingLayer definition."""
    params = [
        f"name='{_esc(layer.name)}'",
        f"pooling_type='{_esc(layer.pooling_type)}'",
        f"dimension='{_esc(layer.dimension)}'",
    ]
    if layer.kernel_dim is not None:
        params.append(f"kernel_dim={layer.kernel_dim}")
    if layer.stride_dim is not None:
        params.append(f"stride_dim={layer.stride_dim}")
    if _is_attr_set(layer, 'padding_amount') or layer.padding_amount:
        params.append(f"padding_amount={layer.padding_amount}")
    if _is_attr_set(layer, 'padding_type') or (
        layer.padding_type and layer.padding_type != "valid"
    ):
        params.append(f"padding_type='{_esc(layer.padding_type)}'")
    if _is_attr_set(layer, 'output_dim') or layer.output_dim:
        params.append(f"output_dim={layer.output_dim}")
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")
    if _is_attr_set(layer, 'permute_in'):
        params.append(f"permute_in={layer.permute_in}")
    if _is_attr_set(layer, 'permute_out'):
        params.append(f"permute_out={layer.permute_out}")

    f.write(f"{var_name} = PoolingLayer({', '.join(params)})\n")


def _write_rnn_like(f, layer, var_name: str):
    """Write an RNN-like layer (SimpleRNNLayer, LSTMLayer, or GRULayer) definition."""
    class_name = type(layer).__name__
    params = [
        f"name='{_esc(layer.name)}'",
        f"hidden_size={layer.hidden_size}",
    ]
    if layer.input_size is not None:
        params.append(f"input_size={layer.input_size}")
    if layer.return_type:
        params.append(f"return_type='{_esc(layer.return_type)}'")
    if _is_attr_set(layer, 'bidirectional'):
        params.append(f"bidirectional={layer.bidirectional}")
    if _is_attr_set(layer, 'dropout'):
        params.append(f"dropout={layer.dropout}")
    if _is_attr_set(layer, 'batch_first'):
        params.append(f"batch_first={layer.batch_first}")
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = {class_name}({', '.join(params)})\n")


def _write_linear(f, layer: LinearLayer, var_name: str):
    """Write LinearLayer definition."""
    params = [
        f"name='{_esc(layer.name)}'",
        f"out_features={layer.out_features}",
    ]
    if layer.in_features is not None:
        params.append(f"in_features={layer.in_features}")
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = LinearLayer({', '.join(params)})\n")


def _write_flatten(f, layer: FlattenLayer, var_name: str):
    """Write FlattenLayer definition."""
    params = [f"name='{_esc(layer.name)}'"]
    if _is_attr_set(layer, 'start_dim') or (
        layer.start_dim is not None and layer.start_dim != 1
    ):
        params.append(f"start_dim={layer.start_dim}")
    if _is_attr_set(layer, 'end_dim') or (
        layer.end_dim is not None and layer.end_dim != -1
    ):
        params.append(f"end_dim={layer.end_dim}")
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = FlattenLayer({', '.join(params)})\n")


def _write_embedding(f, layer: EmbeddingLayer, var_name: str):
    """Write EmbeddingLayer definition."""
    params = [
        f"name='{_esc(layer.name)}'",
        f"num_embeddings={layer.num_embeddings}",
        f"embedding_dim={layer.embedding_dim}",
    ]
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = EmbeddingLayer({', '.join(params)})\n")


def _write_dropout(f, layer: DropoutLayer, var_name: str):
    """Write DropoutLayer definition."""
    params = [
        f"name='{_esc(layer.name)}'",
        f"rate={layer.rate}",
    ]
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = DropoutLayer({', '.join(params)})\n")


def _write_layer_norm(f, layer: LayerNormLayer, var_name: str):
    """Write LayerNormLayer definition."""
    params = [
        f"name='{_esc(layer.name)}'",
        f"normalized_shape={layer.normalized_shape}",
    ]
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
    if _is_attr_set(layer, 'input_reused'):
        params.append(f"input_reused={layer.input_reused}")

    f.write(f"{var_name} = LayerNormLayer({', '.join(params)})\n")


def _write_batch_norm(f, layer: BatchNormLayer, var_name: str):
    """Write BatchNormLayer definition."""
    params = [
        f"name='{_esc(layer.name)}'",
        f"num_features={layer.num_features}",
        f"dimension='{_esc(layer.dimension)}'",
    ]
    if layer.actv_func:
        params.append(f"actv_func='{_esc(layer.actv_func)}'")
    if layer.name_module_input:
        params.append(f"name_module_input='{_esc(layer.name_module_input)}'")
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
        f"name='{_esc(tensor_op.name)}'",
        f"tns_type='{_esc(tensor_op.tns_type)}'",
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

    # input_reused is optional for all types - only output when explicitly set
    if _is_attr_set(tensor_op, 'input_reused'):
        params.append(f"input_reused={tensor_op.input_reused}")

    f.write(f"{var_name} = TensorOp({', '.join(params)})\n")


def _write_dataset(f, dataset: Dataset, var_name: str):
    """Write a Dataset (and optional Image) definition."""
    if dataset.image is not None:
        image = dataset.image
        f.write(f"image = Image(shape={image.shape}, normalize={image.normalize})\n")
    params = [
        f"name='{_esc(dataset.name)}'",
        f"path_data='{_esc(dataset.path_data)}'",
    ]
    task_type = getattr(dataset, 'task_type', None)
    if task_type:
        params.append(f"task_type='{_esc(task_type)}'")
    input_format = getattr(dataset, 'input_format', None)
    if input_format:
        params.append(f"input_format='{_esc(input_format)}'")
    if dataset.image is not None:
        params.append("image=image")
    f.write(f"{var_name} = Dataset({', '.join(params)})\n")


def _write_configuration(f, config: Configuration, var_name: str):
    """Write Configuration definition."""
    params = [
        f"batch_size={config.batch_size}",
        f"epochs={config.epochs}",
        f"learning_rate={config.learning_rate}",
        f"optimizer='{_esc(config.optimizer)}'",
        f"loss_function='{_esc(config.loss_function)}'",
        f"metrics={_fmt_metrics(config.metrics)}",
    ]
    if _is_attr_set(config, 'weight_decay') or (
        config.weight_decay is not None and config.weight_decay != 0
    ):
        params.append(f"weight_decay={config.weight_decay}")
    if _is_attr_set(config, 'momentum') or (
        config.momentum is not None and config.momentum != 0
    ):
        params.append(f"momentum={config.momentum}")

    f.write(f"{var_name} = Configuration({', '.join(params)})\n")
