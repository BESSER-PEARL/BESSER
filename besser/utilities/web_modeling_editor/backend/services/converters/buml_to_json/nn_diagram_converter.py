"""
NN BUML -> v4 JSON converter.

Emits the v4 wire shape (``{nodes, edges}``) directly. Each layer's
attributes collapse onto ``data.attributes: dict`` (snake_case keys per
the spec at ``docs/source/migrations/uml-v4-shape.md``).
"""

import ast
import json
import threading
import uuid
from typing import Any

import besser.BUML.metamodel.nn as nn_module
from besser.BUML.metamodel.nn import NN, Configuration, Dataset
from besser.utilities.buml_code_builder.nn_model_builder import is_attr_explicitly_set
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_edge,
    make_node,
)


_NN_JSON_NAMESPACE = uuid.UUID('7a1a0c7e-0e9d-4e1c-a0e6-6e5b4e0c7e00')

_id_state = threading.local()


def _reset_id_state() -> None:
    _id_state.counter = 0


def _new_id(hint: str = '') -> str:
    counter = getattr(_id_state, 'counter', 0)
    _id_state.counter = counter + 1
    key = f"{counter}:{hint}"
    return str(uuid.uuid5(_NN_JSON_NAMESPACE, key))


_MODULE_TYPE_MAP = {
    'Conv1D':          'Conv1DLayer',
    'Conv2D':          'Conv2DLayer',
    'Conv3D':          'Conv3DLayer',
    'PoolingLayer':    'PoolingLayer',
    'SimpleRNNLayer':  'RNNLayer',
    'LSTMLayer':       'LSTMLayer',
    'GRULayer':        'GRULayer',
    'LinearLayer':     'LinearLayer',
    'FlattenLayer':    'FlattenLayer',
    'EmbeddingLayer':  'EmbeddingLayer',
    'DropoutLayer':    'DropoutLayer',
    'LayerNormLayer':  'LayerNormalizationLayer',
    'BatchNormLayer':  'BatchNormalizationLayer',
    'TensorOp':        'TensorOp',
}


def _is_attr_set(obj, attr_name: str) -> bool:
    """True when an attribute has to be emitted for the model to round-trip.

    Shares the code builder's rule so BUML->JSON and BUML->Python agree:
    the editor's explicit-set sidecar OR a value that diverges from the
    metamodel's declared default. The second half is what keeps hand-written
    BUML models (and models rebuilt by :func:`_parse_nn_buml_ast`, which never
    populate the sidecar) from silently losing their attributes.
    """
    return is_attr_explicitly_set(obj, attr_name)


def _fmt_value(value: Any) -> str:
    """Convert a Python value to the string stored in v4 ``data.attributes``.

    Lists are formatted as ``[a, b, c]`` with string items single-quoted (the
    processor strips the quotes again). Items containing ``,``, ``[``, ``]``
    or ``'`` would round-trip incorrectly, so they are rejected instead of
    silently truncated.

    Dicts (and lists containing them — ``TensorOp.subscript_indices`` is a
    ``list[dict]``) are emitted as real JSON via :func:`json.dumps`. ``str()``
    would produce a Python repr with single quotes, which the frontend's
    ``JSON.parse`` rejects outright.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict) or (
        isinstance(value, (list, tuple))
        and any(isinstance(item, dict) for item in value)
    ):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        parts = []
        for v in value:
            if isinstance(v, str):
                # Quote string items for proper semantics (parser strips quotes during parse)
                if (',' in v or ']' in v or '[' in v or "'" in v):
                    raise ValueError(
                        f"List item {v!r} contains unsupported characters (,[]') "
                        f"that would break round-trip through the editor format"
                    )
                parts.append(f"'{v}'")
            else:
                formatted = _fmt_value(v)
                parts.append(formatted)
        return '[' + ', '.join(parts) + ']'
    return str(value)


def _append_base_layer_fields(fields: list[tuple[str, Any, str, bool]], module) -> None:
    """Append base-layer fields (inherited from Layer) to the fields list."""
    if _is_attr_set(module, 'is_layer_call'):
        fields.append(('is_layer_call', module.is_layer_call, 'bool', False))
    if module.input_var:
        fields.append(('input_var', module.input_var, 'str', False))
    if module.output_var:
        fields.append(('output_var', module.output_var, 'str', False))


def _module_fields(module) -> list[tuple[str, Any, str, bool]]:
    """
    Return the list of (field_name, value, type_hint, mandatory) tuples to emit for a module.

    Mirrors the builder's emission rules: only emits fields that are explicitly set
    or diverge from defaults, so round-trip stays minimal.
    """
    cls = type(module).__name__
    fields: list[tuple[str, Any, str, bool]] = []

    if cls in ('Conv1D', 'Conv2D', 'Conv3D'):
        fields.append(('name', module.name, 'str', True))
        fields.append(('kernel_dim', module.kernel_dim, 'List', True))
        fields.append(('out_channels', module.out_channels, 'int', True))
        if module.in_channels is not None:
            fields.append(('in_channels', module.in_channels, 'int', False))
        if module.stride_dim is not None:
            fields.append(('stride_dim', module.stride_dim, 'List', False))
        if _is_attr_set(module, 'padding_amount') or module.padding_amount:
            fields.append(('padding_amount', module.padding_amount, 'int', False))
        if _is_attr_set(module, 'padding_type') or (module.padding_type and module.padding_type != 'valid'):
            fields.append(('padding_type', module.padding_type, 'str', False))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))
        # Conv-specific optional fields
        if _is_attr_set(module, 'dilation') or module.dilation:
            fields.append(('dilation', module.dilation, 'List', False))
        if _is_attr_set(module, 'groups') or (module.groups is not None and module.groups != 1):
            fields.append(('groups', module.groups, 'int', False))
        if _is_attr_set(module, 'bias'):
            fields.append(('bias', module.bias, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'PoolingLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('pooling_type', module.pooling_type, 'str', True))
        fields.append(('dimension', module.dimension, 'str', True))
        if module.kernel_dim is not None:
            fields.append(('kernel_dim', module.kernel_dim, 'List', False))
        if module.stride_dim is not None:
            fields.append(('stride_dim', module.stride_dim, 'List', False))
        if _is_attr_set(module, 'padding_amount') or module.padding_amount:
            fields.append(('padding_amount', module.padding_amount, 'int', False))
        if _is_attr_set(module, 'padding_type') or (module.padding_type and module.padding_type != 'valid'):
            fields.append(('padding_type', module.padding_type, 'str', False))
        if _is_attr_set(module, 'output_dim') or module.output_dim:
            fields.append(('output_dim', module.output_dim, 'List', False))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls in ('SimpleRNNLayer', 'LSTMLayer', 'GRULayer'):
        fields.append(('name', module.name, 'str', True))
        fields.append(('hidden_size', module.hidden_size, 'int', True))
        if module.input_size is not None:
            fields.append(('input_size', module.input_size, 'int', False))
        if module.return_type:
            fields.append(('return_type', module.return_type, 'str', False))
        if _is_attr_set(module, 'bidirectional'):
            fields.append(('bidirectional', module.bidirectional, 'bool', False))
        if _is_attr_set(module, 'dropout'):
            fields.append(('dropout', module.dropout, 'float', False))
        if _is_attr_set(module, 'batch_first'):
            fields.append(('batch_first', module.batch_first, 'bool', False))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        if _is_attr_set(module, 'bias'):
            fields.append(('bias', module.bias, 'bool', False))
        # RNN-specific optional fields
        if module.hx_source:
            fields.append(('hx_source', module.hx_source, 'str', False))
        if module.hidden_state_var:
            fields.append(('hidden_state_var', module.hidden_state_var, 'str', False))
        if _is_attr_set(module, 'hidden_unused'):
            fields.append(('hidden_unused', module.hidden_unused, 'bool', False))
        if module.hidden_subscript_source:
            fields.append(('hidden_subscript_source', module.hidden_subscript_source, 'str', False))
        if module.hidden_subscript_target:
            fields.append(('hidden_subscript_target', module.hidden_subscript_target, 'str', False))
        # LSTM-specific fields
        if cls == 'LSTMLayer':
            if module.cell_state_var:
                fields.append(('cell_state_var', module.cell_state_var, 'str', False))
            if _is_attr_set(module, 'cell_unused'):
                fields.append(('cell_unused', module.cell_unused, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'LinearLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('out_features', module.out_features, 'int', True))
        if module.in_features is not None:
            fields.append(('in_features', module.in_features, 'int', False))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        # Linear-specific optional fields
        if _is_attr_set(module, 'bias'):
            fields.append(('bias', module.bias, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'FlattenLayer':
        fields.append(('name', module.name, 'str', True))
        if _is_attr_set(module, 'start_dim') or (
            module.start_dim is not None and module.start_dim != 1
        ):
            fields.append(('start_dim', module.start_dim, 'int', False))
        if _is_attr_set(module, 'end_dim') or (
            module.end_dim is not None and module.end_dim != -1
        ):
            fields.append(('end_dim', module.end_dim, 'int', False))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'EmbeddingLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('num_embeddings', module.num_embeddings, 'int', True))
        fields.append(('embedding_dim', module.embedding_dim, 'int', True))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        # Embedding-specific optional fields
        if _is_attr_set(module, 'padding_idx') or module.padding_idx is not None:
            fields.append(('padding_idx', module.padding_idx, 'int', False))
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'DropoutLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('rate', module.rate, 'float', True))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        # Dropout-specific optional fields
        if module.dimension:
            fields.append(('dimension', module.dimension, 'str', False))
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'LayerNormLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('normalized_shape', module.normalized_shape, 'List', True))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        # LayerNorm-specific optional fields
        if _is_attr_set(module, 'eps') or (module.eps is not None and module.eps != 1e-5):
            fields.append(('eps', module.eps, 'float', False))
        if _is_attr_set(module, 'affine'):
            fields.append(('affine', module.affine, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'BatchNormLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('num_features', module.num_features, 'int', True))
        fields.append(('dimension', module.dimension, 'str', True))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        # BatchNorm-specific optional fields
        if _is_attr_set(module, 'eps') or (module.eps is not None and module.eps != 1e-5):
            fields.append(('eps', module.eps, 'float', False))
        if _is_attr_set(module, 'momentum') or (module.momentum is not None and module.momentum != 0.1):
            fields.append(('momentum', module.momentum, 'float', False))
        if _is_attr_set(module, 'affine'):
            fields.append(('affine', module.affine, 'bool', False))
        if _is_attr_set(module, 'track_running_stats'):
            fields.append(('track_running_stats', module.track_running_stats, 'bool', False))
        # BatchNorm inherits permute_in / permute_out as real constructor
        # parameters (unlike LayerNorm, which hard-codes them to False).
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))
        _append_base_layer_fields(fields, module)

    elif cls == 'TensorOp':
        fields.append(('name', module.name, 'str', True))
        fields.append(('tns_type', module.tns_type, 'str', True))
        tns_type = module.tns_type
        # Common TensorOp fields (all types)
        if module.input_var:
            fields.append(('input_var', module.input_var, 'str', False))
        if module.output_var:
            fields.append(('output_var', module.output_var, 'str', False))
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))
        # Type-specific parameters
        if tns_type == 'concatenate':
            if module.concatenate_dim is not None:
                fields.append(('concatenate_dim', module.concatenate_dim, 'int', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
            if module.actual_vars is not None:
                fields.append(('actual_vars', module.actual_vars, 'List', False))
        elif tns_type in ('multiply', 'matmultiply') and module.layers_of_tensors is not None:
            fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        elif tns_type == 'reshape':
            if module.reshape_dim is not None:
                fields.append(('reshape_dim', module.reshape_dim, 'List', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        elif tns_type == 'transpose' and module.transpose_dim is not None:
            fields.append(('transpose_dim', module.transpose_dim, 'List', False))
        elif tns_type == 'permute':
            if module.permute_dim is not None:
                fields.append(('permute_dim', module.permute_dim, 'List', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        elif tns_type == 'repeat':
            if module.repeat_dim is not None:
                fields.append(('repeat_dim', module.repeat_dim, 'List', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        # Binary operations
        elif tns_type in ('binop_add', 'binop_subtract', 'binop_multiply',
                           'binop_divide', 'binop_floor_divide'):
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
            if module.actual_vars is not None:
                fields.append(('actual_vars', module.actual_vars, 'List', False))
        # Reduce operations
        elif tns_type in ('shape_dim', 'mean', 'max', 'squeeze', 'unsqueeze', 'normalize'):
            if module.reduce_dim is not None:
                fields.append(('reduce_dim', module.reduce_dim, 'int', False))
            if module.shape_dim is not None:
                fields.append(('shape_dim', module.shape_dim, 'int', False))
            if tns_type == 'max' and module.reduce_keepdims is not None:
                fields.append(('reduce_keepdims', module.reduce_keepdims, 'bool', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        # Subscript
        elif tns_type == 'subscript':
            if module.subscript_indices is not None:
                fields.append(('subscript_indices', module.subscript_indices, 'List', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        # Interpolate
        elif tns_type == 'interpolate':
            if module.interpolate_size is not None:
                fields.append(('interpolate_size', module.interpolate_size, 'List', False))
            if module.interpolate_scale is not None:
                fields.append(('interpolate_scale', module.interpolate_scale, 'float', False))
            if module.interpolate_mode:
                fields.append(('interpolate_mode', module.interpolate_mode, 'str', False))
        # Pad
        elif tns_type == 'pad':
            if module.pad_amount is not None:
                fields.append(('pad_amount', module.pad_amount, 'List', False))
            if module.pad_mode:
                fields.append(('pad_mode', module.pad_mode, 'str', False))
            if module.pad_value is not None:
                fields.append(('pad_value', module.pad_value, 'float', False))
        # Dropout (TensorOp variant)
        elif tns_type == 'dropout':
            if module.dropout_rate is not None:
                fields.append(('dropout_rate', module.dropout_rate, 'float', False))
            if module.dropout_training_aware is not None:
                fields.append(('dropout_training_aware', module.dropout_training_aware, 'bool', False))
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        # Split
        elif tns_type == 'split':
            if module.split_dim is not None:
                fields.append(('split_dim', module.split_dim, 'int', False))
            if module.split_sizes is not None:
                fields.append(('split_sizes', module.split_sizes, 'int', False))
            if module.output_vars is not None:
                fields.append(('output_vars', module.output_vars, 'List', False))
        # Identity and zeros_like
        elif tns_type in ('identity', 'zeros_like'):
            if module.layers_of_tensors is not None:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))

    return fields


def _configuration_fields(config: Configuration) -> list[tuple[str, Any, str, bool]]:
    fields = [
        ('batch_size',     config.batch_size,     'int',  True),
        ('epochs',         config.epochs,         'int',  True),
        ('learning_rate',  config.learning_rate,  'float', True),
        ('optimizer',      config.optimizer,      'str',  True),
        ('loss_function',  config.loss_function,  'str',  True),
        ('metrics',        config.metrics,        'List', True),
    ]
    if _is_attr_set(config, 'weight_decay') or (
        config.weight_decay is not None and config.weight_decay != 0
    ):
        fields.append(('weight_decay', config.weight_decay, 'float', False))
    if _is_attr_set(config, 'momentum') or (
        config.momentum is not None and config.momentum != 0
    ):
        fields.append(('momentum', config.momentum, 'float', False))
    return fields


def _dataset_fields(dataset: Dataset) -> list[tuple[str, Any, str, bool]]:
    fields: list[tuple[str, Any, str, bool]] = [
        ('name',      dataset.name,      'str', True),
        ('path_data', dataset.path_data, 'str', True),
    ]
    task_type = getattr(dataset, 'task_type', None)
    if task_type:
        fields.append(('task_type', task_type, 'str', False))
    input_format = getattr(dataset, 'input_format', None)
    if input_format:
        fields.append(('input_format', input_format, 'str', False))
    if dataset.image is not None:
        fields.append(('shape', dataset.image.shape, 'List', False))
        fields.append(('normalize', dataset.image.normalize, 'bool', False))
    return fields


def _attrs_dict(fields: list[tuple[str, Any, str, bool]], layer_kind: str | None = None) -> dict:
    """Collapse the field list into the v4 ``data.attributes`` dict.

    Some attribute slugs collide across layer kinds (e.g. ``dimension`` lives
    on both PoolingLayer and BatchNormalizationLayer with different semantics).
    The frontend disambiguates via ``qualifySlug(layerKind, slug)``; mirror the
    same prefix when emitting so v4 round-trips on both sides.
    """
    out: dict = {}
    prefix = _LAYER_KIND_PREFIX.get(layer_kind or '')
    for field_name, value, _hint, _mandatory in fields:
        key = f'{prefix}.{field_name}' if (prefix is not None and field_name in _COLLIDING_SLUGS) else field_name
        out[key] = _fmt_value(value)
    return out


# Keep in sync with json_to_buml/nn_diagram_processor.py and the frontend
# qualifySlug() helper.
_COLLIDING_SLUGS = frozenset({'dimension'})
_LAYER_KIND_PREFIX = {
    'PoolingLayer': 'pooling',
    'BatchNormalizationLayer': 'batch_normalization',
}


def _emit_module_node(module, parent_id: str, x: int, y: int, nodes: list) -> str:
    cls = type(module).__name__
    parent_type = _MODULE_TYPE_MAP.get(cls)
    if parent_type is None:
        raise ValueError(
            f"Cannot emit module of type {cls!r}: no mapping to v4 node type."
        )
    node_id = _new_id()
    attrs = _attrs_dict(_module_fields(module), layer_kind=parent_type)
    nodes.append(make_node(
        node_id=node_id,
        type_=parent_type,
        data={"name": module.name, "attributes": attrs},
        position={"x": x, "y": y},
        parent_id=parent_id,
        width=200,
        height=120,
    ))
    return node_id


def _emit_container_node(name: str, x: int, y: int, width: int, height: int, nodes: list,
                         input_var: str | None = None,
                         return_vars: list[str] | None = None) -> str:
    container_id = _new_id()
    data: dict[str, Any] = {"name": name}
    # The NN's forward signature (NN.input_var / NN.return_vars) lives on the
    # container node: ``data.input_var`` (str), ``data.return_vars`` (list).
    if input_var:
        data["input_var"] = input_var
    if return_vars:
        data["return_vars"] = list(return_vars)
    nodes.append(make_node(
        node_id=container_id,
        type_="NNContainer",
        data=data,
        position={"x": x, "y": y},
        width=width,
        height=height,
    ))
    return container_id


def _emit_nn_reference_node(ref_name: str, parent_id: str, x: int, y: int, nodes: list) -> str:
    ref_id = _new_id()
    nodes.append(make_node(
        node_id=ref_id,
        type_="NNReference",
        data={"name": ref_name, "referenceTarget": ref_name},
        position={"x": x, "y": y},
        parent_id=parent_id,
        width=110,
        height=110,
    ))
    return ref_id


def _emit_nnnext(source_id: str, target_id: str, edges: list) -> None:
    edges.append(make_edge(
        edge_id=_new_id(),
        source=source_id,
        target=target_id,
        type_="NNNext",
        data={"name": "next", "points": []},
    ))


def _emit_container_link(rel_type: str, source_id: str, target_id: str, edges: list) -> None:
    edges.append(make_edge(
        edge_id=_new_id(),
        source=source_id,
        target=target_id,
        type_=rel_type,
        data={"points": []},
        source_handle="top",
        target_handle="bottom",
    ))


def _emit_configuration_node(config: Configuration, x: int, y: int, nodes: list) -> str:
    node_id = _new_id()
    attrs = _attrs_dict(_configuration_fields(config))
    nodes.append(make_node(
        node_id=node_id,
        type_="Configuration",
        data={"name": "Configuration", "attributes": attrs},
        position={"x": x, "y": y},
        width=160,
        height=200,
    ))
    return node_id


def _emit_dataset_node(dataset: Dataset, parent_type: str, x: int, y: int, nodes: list) -> str:
    node_id = _new_id()
    attrs = _attrs_dict(_dataset_fields(dataset))
    nodes.append(make_node(
        node_id=node_id,
        type_=parent_type,
        data={"name": parent_type, "attributes": attrs},
        position={"x": x, "y": y},
        width=200,
        height=120,
    ))
    return node_id


def _emit_nn_container(nn: NN, y_base: int, nodes: list, edges: list,
                       sub_nn_ids: dict[int, str] | None = None) -> str:
    step = 140
    module_count = max(len(nn.modules), 1)
    width = 30 + module_count * step
    container_x = -width // 2
    # NN.return_vars is a comma-separated string; the editor stores a list.
    return_vars_list = (
        [v.strip() for v in nn.return_vars.split(",") if v.strip()]
        if nn.return_vars else None
    )
    container_id = _emit_container_node(
        nn.name, container_x, y_base, width, 250, nodes,
        input_var=nn.input_var or None,
        return_vars=return_vars_list,
    )

    prev_id: str | None = None
    for i, module in enumerate(nn.modules):
        x = container_x + 20 + i * step
        y = y_base + 60
        module_type = type(module).__name__
        if module_type == 'NN' and sub_nn_ids and id(module) in sub_nn_ids:
            element_id = _emit_nn_reference_node(module.name, container_id, x, y, nodes)
        else:
            element_id = _emit_module_node(module, container_id, x, y, nodes)
        if prev_id is not None:
            _emit_nnnext(prev_id, element_id, edges)
        prev_id = element_id

    return container_id


def _collect_all_sub_nns(nn_model: NN) -> list:
    ordered: list = []
    seen: set = set()

    def _visit(node: NN) -> None:
        for child in getattr(node, 'sub_nns', ()) or ():
            if id(child) in seen:
                continue
            seen.add(id(child))
            _visit(child)
            ordered.append(child)

    _visit(nn_model)
    return ordered


def nn_model_to_json(nn_model: NN) -> dict[str, Any]:
    """Convert a BUML NN instance into the v4 NNDiagram model dict."""
    # Deterministic IDs: reset the per-conversion counter so the same input
    # produces the same JSON bytes on every call.
    _reset_id_state()
    nodes: list = []
    edges: list = []

    # Emit every transitively-reachable sub-NN, not just the top-level list;
    # keyed by id(sub_nn) so two sibling sub-NNs with the same display name
    # stay distinct.
    sub_nn_ids: dict[int, str] = {}
    y_cursor = -700
    for sub_nn in _collect_all_sub_nns(nn_model):
        cid = _emit_nn_container(sub_nn, y_cursor, nodes, edges, sub_nn_ids=sub_nn_ids)
        sub_nn_ids[id(sub_nn)] = cid
        y_cursor += 300

    main_container_id = _emit_nn_container(nn_model, y_cursor, nodes, edges, sub_nn_ids=sub_nn_ids)

    if nn_model.configuration is not None:
        config_id = _emit_configuration_node(nn_model.configuration, 700, y_cursor, nodes)
        _emit_container_link('NNComposition', config_id, main_container_id, edges)

    ds_x = -400
    ds_y = y_cursor + 350
    if getattr(nn_model, 'train_data', None) is not None:
        train_id = _emit_dataset_node(nn_model.train_data, 'TrainingDataset', ds_x, ds_y, nodes)
        _emit_container_link('NNAssociation', train_id, main_container_id, edges)
        ds_x += 300
    if getattr(nn_model, 'test_data', None) is not None:
        test_id = _emit_dataset_node(nn_model.test_data, 'TestDataset', ds_x, ds_y, nodes)
        _emit_container_link('NNAssociation', test_id, main_container_id, edges)

    return {
        "version": "4.0.0",
        "type": "NNDiagram",
        "title": "",
        "size": {"width": 1520, "height": 800},
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }


def _nn_add_method_whitelist():
    return {
        'add_layer', 'add_tensor_op', 'add_sub_nn',
        'add_configuration', 'add_train_data', 'add_test_data',
    }


def _parse_nn_buml_ast(content: str, nn_module):
    """Safely reconstruct an NN model from Python-looking BUML."""
    allowed_classes = {
        name: getattr(nn_module, name)
        for name in dir(nn_module) if not name.startswith('_')
    }
    allowed_add_methods = _nn_add_method_whitelist()

    def _eval(node, env):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand, env)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in allowed_classes:
                return allowed_classes[node.id]
            raise ValueError(f"Unknown name in NN BUML: {node.id!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    raise ValueError(
                        "Iterable unpacking (*expr) is not supported "
                        "in NN BUML literals."
                    )
            return [_eval(e, env) for e in node.elts]
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if k is None:
                    raise ValueError(
                        "Dict unpacking (**expr) is not supported "
                        "in NN BUML literals."
                    )
            return {_eval(k, env): _eval(v, env)
                    for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Set):
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    raise ValueError(
                        "Iterable unpacking (*expr) is not supported "
                        "in NN BUML literals."
                    )
            return {_eval(e, env) for e in node.elts}
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    "Only calls to whitelisted NN metamodel classes are "
                    "allowed in expressions; got "
                    f"{ast.dump(node.func, annotate_fields=False)!r}"
                )
            func = _eval(node.func, env)
            if func not in allowed_classes.values():
                raise ValueError(
                    f"Call to non-whitelisted class {node.func.id!r}"
                )
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise ValueError(
                        "Positional *args unpacking is not supported in NN BUML."
                    )
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError(
                        "**kwargs unpacking is not supported in NN BUML."
                    )
            args = [_eval(a, env) for a in node.args]
            kwargs = {kw.arg: _eval(kw.value, env) for kw in node.keywords}
            return func(*args, **kwargs)
        raise ValueError(
            f"Disallowed expression in NN BUML: {type(node).__name__}"
        )

    def _run_stmt(stmt, env):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    raise ValueError(
                        "Only simple-name assignment targets are allowed in NN BUML."
                    )
            value = _eval(stmt.value, env)
            for target in stmt.targets:
                env[target.id] = value
            return
        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant):
                return
            if isinstance(stmt.value, ast.Call) and \
                    isinstance(stmt.value.func, ast.Attribute) and \
                    isinstance(stmt.value.func.value, ast.Name):
                method = stmt.value.func.attr
                if method not in allowed_add_methods:
                    raise ValueError(
                        f"Disallowed method call {method!r} in NN BUML; "
                        f"allowed: {sorted(allowed_add_methods)}"
                    )
                for a in stmt.value.args:
                    if isinstance(a, ast.Starred):
                        raise ValueError(
                            "Positional *args unpacking is not supported in NN BUML add_* builder calls."
                        )
                for kw in stmt.value.keywords:
                    if kw.arg is None:
                        raise ValueError(
                            "**kwargs unpacking is not supported in NN BUML add_* builder calls."
                        )
                target = _eval(stmt.value.func.value, env)
                args = [_eval(a, env) for a in stmt.value.args]
                kwargs = {kw.arg: _eval(kw.value, env)
                          for kw in stmt.value.keywords}
                getattr(target, method)(*args, **kwargs)
                return
            raise ValueError(
                f"Disallowed top-level expression: "
                f"{ast.dump(stmt.value, annotate_fields=False)!r}"
            )
        raise ValueError(
            f"Disallowed top-level statement: {type(stmt).__name__}"
        )

    try:
        tree = ast.parse(content, mode='exec')
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse NN BUML content: {exc}") from exc

    env: dict[str, Any] = {}
    for stmt in tree.body:
        _run_stmt(stmt, env)
    return env


def nn_buml_to_json(content: str) -> dict[str, Any]:
    """Convert an NN model Python section to the editor NNDiagram model dict.

    Uses an ``ast.parse`` whitelist visitor rather than ``exec`` so that
    uploaded BUML content cannot escape the sandbox (``().__class__.__bases__
    .__subclasses__()``-style tricks) or run arbitrary system calls.
    """
    env = _parse_nn_buml_ast(content, nn_module)

    all_nns = [v for v in env.values() if isinstance(v, NN)]
    if not all_nns:
        raise ValueError("No NN instance found in the NN BUML content")

    referenced: set = set()
    _walk_stack = list(all_nns)
    while _walk_stack:
        nn = _walk_stack.pop()
        for sub in getattr(nn, 'sub_nns', ()) or ():
            if id(sub) in referenced:
                continue
            referenced.add(id(sub))
            _walk_stack.append(sub)
    main_nns = [nn for nn in all_nns if id(nn) not in referenced]
    if len(main_nns) > 1:
        names = ', '.join(sorted(n.name for n in main_nns))
        raise ValueError(
            f"NN BUML contains {len(main_nns)} top-level NN instances "
            f"({names}); exactly one is required. Link the others via "
            f"add_sub_nn(...) to mark them as sub-networks."
        )
    if not main_nns:
        raise ValueError(
            "NN BUML has no top-level NN: every NN instance is referenced "
            "as a sub-network by another. Remove one of the add_sub_nn "
            "links to designate a main network."
        )
    main_nn = main_nns[0]

    return nn_model_to_json(main_nn)
