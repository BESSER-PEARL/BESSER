"""
NN BUML → frontend JSON converter.

Converts a BUML NN model into the element/relationship structure consumed
by the web editor for NNDiagrams.
"""

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from besser.BUML.metamodel.nn import NN, Configuration, Dataset
from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit


# Stable namespace for uuid5-based element IDs. Using a fixed UUID here means
# the same BUML NN model, converted twice, produces byte-identical JSON — which
# makes round-trip snapshot tests reliable and helps downstream content-addressing.
_NN_JSON_NAMESPACE = uuid.UUID('7a1a0c7e-0e9d-4e1c-a0e6-6e5b4e0c7e00')

# Thread-local monotonically-increasing counter. Reset at the top of every
# public conversion call (see nn_model_to_json). Thread-local so concurrent
# requests in the FastAPI thread pool don't interleave counter states.
_id_state = threading.local()


def _reset_id_state() -> None:
    """Reset the deterministic-ID counter for a fresh conversion."""
    _id_state.counter = 0


def _new_id(hint: str = '') -> str:
    """Return a deterministic UUID for the next element in the current conversion.

    Sequence number is derived from a thread-local counter that ``nn_model_to_json``
    resets. The optional ``hint`` (e.g. 'container', 'attr:name') is folded into
    the uuid5 key so different call sites at the same counter position still
    produce distinct IDs — and identical inputs produce identical IDs across runs.
    """
    counter = getattr(_id_state, 'counter', 0)
    _id_state.counter = counter + 1
    key = f"{counter}:{hint}"
    return str(uuid.uuid5(_NN_JSON_NAMESPACE, key))


# Metamodel class → (frontend parent type, attribute type suffix)
_MODULE_TYPE_MAP = {
    'Conv1D':          ('Conv1DLayer',              'Conv1D'),
    'Conv2D':          ('Conv2DLayer',              'Conv2D'),
    'Conv3D':          ('Conv3DLayer',              'Conv3D'),
    'PoolingLayer':    ('PoolingLayer',             'Pooling'),
    'SimpleRNNLayer':  ('RNNLayer',                 'RNN'),
    'LSTMLayer':       ('LSTMLayer',                'LSTM'),
    'GRULayer':        ('GRULayer',                 'GRU'),
    'LinearLayer':     ('LinearLayer',              'Linear'),
    'FlattenLayer':    ('FlattenLayer',             'Flatten'),
    'EmbeddingLayer':  ('EmbeddingLayer',           'Embedding'),
    'DropoutLayer':    ('DropoutLayer',             'Dropout'),
    'LayerNormLayer':  ('LayerNormalizationLayer',  'LayerNorm'),
    'BatchNormLayer':  ('BatchNormalizationLayer',  'BatchNorm'),
    'TensorOp':        ('TensorOp',                 'TensorOp'),
}


def _is_attr_set(obj, attr_name: str) -> bool:
    """True when an attribute was explicitly toggled in the editor.

    Delegates to the sidecar bookkeeping so the metamodel objects don't carry
    editor-specific state.
    """
    return is_explicit(obj, attr_name)


def _attr_type_for(field: str, suffix: str) -> str:
    """Build the attribute `type` used by the frontend (e.g. 'NameAttributeConv2D')."""
    pascal = ''.join(part.capitalize() for part in field.split('_'))
    return f"{pascal}Attribute{suffix}"


def _fmt_value(value: Any) -> str:
    """Convert a Python value to the string representation used in the JSON 'value' field.

    Lists are formatted as ``[a, b, c]``; string items are kept bare (no quotes)
    because the processor's parsers tolerate both ``[a, b]`` and ``['a', 'b']``
    and downstream consumers expect the bare form. The one exception is items
    containing ``,`` or ``]``, which would round-trip incorrectly — those are
    rejected so we fail fast instead of silently truncating.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        # Tuples reach here when the metamodel setter accepted a tuple for
        # a list-typed field (e.g. a hand-authored kernel_dim=(3, 3)). Treat
        # them the same as lists so round-trip doesn't stringify the tuple
        # syntax into something the processor can't re-parse cleanly.
        parts = []
        for v in value:
            formatted = _fmt_value(v)
            if isinstance(v, str) and (',' in v or ']' in v or '[' in v):
                raise ValueError(
                    f"List item {v!r} contains unsupported characters (,[]) "
                    f"that would break round-trip through the editor format"
                )
            parts.append(formatted)
        return '[' + ', '.join(parts) + ']'
    return str(value)


def _make_attr(parent_id: str, field: str, value: Any, suffix: str,
               bounds: Dict[str, int], attr_type_hint: str = 'str',
               mandatory: bool = False) -> Dict[str, Any]:
    """Build a single attribute element child dict."""
    formatted = _fmt_value(value)
    return {
        'id': _new_id(),
        'name': f'{field} = {formatted}',
        'type': _attr_type_for(field, suffix),
        'owner': parent_id,
        'bounds': dict(bounds),
        'code': '',
        'visibility': 'public',
        'attributeType': attr_type_hint,
        'implementationType': 'none',
        'stateMachineId': '',
        'quantumCircuitId': '',
        'isOptional': not mandatory,
        'isDerived': False,
        'attributeName': field,
        'value': formatted,
        'isMandatory': mandatory,
    }


def _module_fields(module) -> List[Tuple[str, Any, str, bool]]:
    """
    Return the list of (field_name, value, type_hint, mandatory) tuples to emit for a module.

    Mirrors the builder's emission rules: only emits fields that are explicitly set
    or diverge from defaults, so round-trip stays minimal.
    """
    cls = type(module).__name__
    fields: List[Tuple[str, Any, str, bool]] = []

    if cls in ('Conv1D', 'Conv2D', 'Conv3D'):
        fields.append(('name', module.name, 'str', True))
        fields.append(('kernel_dim', module.kernel_dim, 'List', True))
        fields.append(('out_channels', module.out_channels, 'int', True))
        if module.in_channels is not None:
            fields.append(('in_channels', module.in_channels, 'int', False))
        if module.stride_dim is not None:
            fields.append(('stride_dim', module.stride_dim, 'List', False))
        if _is_attr_set(module, 'padding_amount') or module.padding_amount:
            # Emit when user explicitly set (even to 0) or when non-default.
            fields.append(('padding_amount', module.padding_amount, 'int', False))
        if _is_attr_set(module, 'padding_type') or (module.padding_type and module.padding_type != 'valid'):
            fields.append(('padding_type', module.padding_type, 'str', False))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))
        # permute_in/out are bools in the metamodel; gate by explicit-set so
        # a user-toggled False also round-trips back to the editor.
        if _is_attr_set(module, 'permute_in'):
            fields.append(('permute_in', module.permute_in, 'bool', False))
        if _is_attr_set(module, 'permute_out'):
            fields.append(('permute_out', module.permute_out, 'bool', False))

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

    elif cls == 'DropoutLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('rate', module.rate, 'float', True))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))

    elif cls == 'LayerNormLayer':
        fields.append(('name', module.name, 'str', True))
        fields.append(('normalized_shape', module.normalized_shape, 'List', True))
        if module.actv_func:
            fields.append(('actv_func', module.actv_func, 'str', False))
        if module.name_module_input:
            fields.append(('name_module_input', module.name_module_input, 'str', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))

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

    elif cls == 'TensorOp':
        fields.append(('name', module.name, 'str', True))
        fields.append(('tns_type', module.tns_type, 'str', True))
        tns_type = module.tns_type
        if tns_type == 'concatenate':
            if module.concatenate_dim is not None:
                fields.append(('concatenate_dim', module.concatenate_dim, 'int', False))
            if module.layers_of_tensors:
                fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        elif tns_type in ('multiply', 'matmultiply') and module.layers_of_tensors:
            fields.append(('layers_of_tensors', module.layers_of_tensors, 'List', False))
        elif tns_type == 'reshape' and module.reshape_dim:
            fields.append(('reshape_dim', module.reshape_dim, 'List', False))
        elif tns_type == 'transpose' and module.transpose_dim:
            fields.append(('transpose_dim', module.transpose_dim, 'List', False))
        elif tns_type == 'permute' and module.permute_dim:
            fields.append(('permute_dim', module.permute_dim, 'List', False))
        if _is_attr_set(module, 'input_reused'):
            fields.append(('input_reused', module.input_reused, 'bool', False))

    return fields


def _configuration_fields(config: Configuration) -> List[Tuple[str, Any, str, bool]]:
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


def _dataset_fields(dataset: Dataset) -> List[Tuple[str, Any, str, bool]]:
    fields: List[Tuple[str, Any, str, bool]] = [
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


def _emit_module(module, owner_id: str, x: int, y: int,
                 elements: Dict[str, Dict[str, Any]]) -> str:
    """Create a layer/tensor_op element (with its attribute children) inside a container."""
    cls = type(module).__name__
    try:
        parent_type, suffix = _MODULE_TYPE_MAP[cls]
    except KeyError as exc:
        raise ValueError(
            f"Cannot emit module of type {cls!r}: no mapping to frontend "
            f"element type. Add {cls!r} to _MODULE_TYPE_MAP."
        ) from exc
    element_id = _new_id()
    attr_ids: List[str] = []

    attr_bounds = {'x': x, 'y': y, 'width': 210, 'height': 20}
    for field, value, attr_type_hint, mandatory in _module_fields(module):
        attr = _make_attr(element_id, field, value, suffix, attr_bounds,
                          attr_type_hint=attr_type_hint, mandatory=mandatory)
        elements[attr['id']] = attr
        attr_ids.append(attr['id'])

    elements[element_id] = {
        'id': element_id,
        'name': parent_type,
        'type': parent_type,
        'owner': owner_id,
        'bounds': {'x': x, 'y': y, 'width': 110, 'height': 110},
        'attributes': attr_ids,
        'methods': [],
    }
    return element_id


def _emit_container(name: str, x: int, y: int, width: int, height: int,
                    elements: Dict[str, Dict[str, Any]]) -> str:
    container_id = _new_id()
    elements[container_id] = {
        'id': container_id,
        'name': name,
        'type': 'NNContainer',
        'owner': None,
        'bounds': {'x': x, 'y': y, 'width': width, 'height': height},
    }
    return container_id


def _emit_nn_reference(ref_name: str, owner_id: str, x: int, y: int,
                       elements: Dict[str, Dict[str, Any]]) -> str:
    ref_id = _new_id()
    elements[ref_id] = {
        'id': ref_id,
        'name': ref_name,
        'type': 'NNReference',
        'owner': owner_id,
        'bounds': {'x': x, 'y': y, 'width': 110, 'height': 110},
        'referencedNN': ref_name,
    }
    return ref_id


def _emit_nnnext(source_id: str, target_id: str,
                 relationships: Dict[str, Dict[str, Any]]) -> None:
    rel_id = _new_id()
    relationships[rel_id] = {
        'id': rel_id,
        'name': 'next',
        'type': 'NNNext',
        'owner': None,
        'bounds': {'x': 0, 'y': 0, 'width': 30, 'height': 31},
        'path': [{'x': 0, 'y': 0}, {'x': 30, 'y': 0}],
        'source': {'direction': 'Right', 'element': source_id, 'multiplicity': '', 'role': ''},
        'target': {'direction': 'Left', 'element': target_id, 'multiplicity': '', 'role': ''},
        'isManuallyLayouted': False,
    }


def _emit_configuration(config: Configuration, x: int, y: int,
                        elements: Dict[str, Dict[str, Any]]) -> str:
    element_id = _new_id()
    attr_ids: List[str] = []
    attr_bounds = {'x': x, 'y': y, 'width': 210, 'height': 20}
    for field, value, attr_type_hint, mandatory in _configuration_fields(config):
        attr = _make_attr(element_id, field, value, 'Configuration', attr_bounds,
                          attr_type_hint=attr_type_hint, mandatory=mandatory)
        elements[attr['id']] = attr
        attr_ids.append(attr['id'])
    elements[element_id] = {
        'id': element_id,
        'name': 'Configuration',
        'type': 'Configuration',
        'owner': None,
        'bounds': {'x': x, 'y': y, 'width': 160, 'height': 200},
        'attributes': attr_ids,
        'methods': [],
    }
    return element_id


def _emit_dataset(dataset: Dataset, parent_type: str, x: int, y: int,
                  elements: Dict[str, Dict[str, Any]]) -> str:
    element_id = _new_id()
    attr_ids: List[str] = []
    attr_bounds = {'x': x, 'y': y, 'width': 210, 'height': 20}
    for field, value, attr_type_hint, mandatory in _dataset_fields(dataset):
        attr = _make_attr(element_id, field, value, 'Dataset', attr_bounds,
                          attr_type_hint=attr_type_hint, mandatory=mandatory)
        elements[attr['id']] = attr
        attr_ids.append(attr['id'])
    elements[element_id] = {
        'id': element_id,
        'name': parent_type,
        'type': parent_type,
        'owner': None,
        'bounds': {'x': x, 'y': y, 'width': 110, 'height': 110},
        'attributes': attr_ids,
        'methods': [],
    }
    return element_id


def _emit_nn_container(nn: NN, y_base: int,
                       elements: Dict[str, Dict[str, Any]],
                       relationships: Dict[str, Dict[str, Any]],
                       sub_nn_ids: Optional[Dict[str, str]] = None) -> str:
    """
    Emit a container holding the modules of a single NN.
    Returns the container id. Lays out modules left-to-right with NNNext relationships.
    """
    step = 140
    module_count = max(len(nn.modules), 1)
    width = 30 + module_count * step
    container_x = -width // 2
    container_id = _emit_container(nn.name, container_x, y_base,
                                    width=width, height=250, elements=elements)

    prev_id: Optional[str] = None
    for i, module in enumerate(nn.modules):
        x = container_x + 20 + i * step
        y = y_base + 60
        module_type = type(module).__name__
        # Pass sub_nn_ids through even when we recurse into a sub-NN's own
        # container, so nested ``NN``-in-``modules`` entries route to
        # _emit_nn_reference instead of _emit_module (which would KeyError
        # on ``_MODULE_TYPE_MAP['NN']``).
        if module_type == 'NN' and sub_nn_ids and module.name in sub_nn_ids:
            element_id = _emit_nn_reference(module.name, container_id, x, y, elements)
        else:
            element_id = _emit_module(module, container_id, x, y, elements)
        if prev_id is not None:
            _emit_nnnext(prev_id, element_id, relationships)
        prev_id = element_id

    return container_id


def _collect_all_sub_nns(nn_model: NN) -> list:
    """Return every transitively-reachable sub-NN in DFS post-order.

    Ensures deeply nested NNReferences (A → B → C) resolve — each level gets
    its own container emitted, not just the top-level sub_nns list.
    """
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


def nn_model_to_json(nn_model: NN) -> Dict[str, Any]:
    """
    Convert a BUML NN instance into a web-editor NNDiagram model dict
    (the inner `model` payload).
    """
    # Deterministic IDs: reset the per-conversion counter so the same input
    # produces the same JSON bytes on every call (helpful for round-trip tests
    # and for diffing BUML-to-JSON output).
    _reset_id_state()
    elements: Dict[str, Dict[str, Any]] = {}
    relationships: Dict[str, Dict[str, Any]] = {}

    # Emit every transitively-reachable sub-NN, not just the top-level list.
    # The processor traverses NNReferences at any depth; we must emit a
    # container for each one or the resulting NNReferences in deeper
    # containers won't resolve on round-trip.
    sub_nn_ids: Dict[str, str] = {}
    y_cursor = -700
    for sub_nn in _collect_all_sub_nns(nn_model):
        # Build sub_nn_ids incrementally so a sub-NN's own container can
        # reference deeper sub-NNs already emitted.
        cid = _emit_nn_container(sub_nn, y_cursor, elements, relationships, sub_nn_ids=sub_nn_ids)
        sub_nn_ids[sub_nn.name] = cid
        y_cursor += 300

    # Main container
    _emit_nn_container(nn_model, y_cursor, elements, relationships, sub_nn_ids=sub_nn_ids)

    # Configuration (unowned, placed to the right)
    if nn_model.configuration is not None:
        _emit_configuration(nn_model.configuration, x=700, y=y_cursor, elements=elements)

    # Datasets (unowned, placed below)
    ds_x = -400
    ds_y = y_cursor + 350
    if getattr(nn_model, 'train_data', None) is not None:
        _emit_dataset(nn_model.train_data, 'TrainingDataset', ds_x, ds_y, elements)
        ds_x += 300
    if getattr(nn_model, 'test_data', None) is not None:
        _emit_dataset(nn_model.test_data, 'TestDataset', ds_x, ds_y, elements)

    return {
        'version': '3.0.0',
        'type': 'NNDiagram',
        'size': {'width': 1520, 'height': 800},
        'interactive': {'elements': {}, 'relationships': {}},
        'elements': elements,
        'relationships': relationships,
        'assessments': {},
    }


def nn_buml_to_json(content: str) -> Dict[str, Any]:
    """
    Convert an NN model Python section to the editor NNDiagram model dict.

    Executes the BUML code, finds the top-level NN instance (the one that is not
    referenced as a sub_nn of another), and converts it to JSON.
    """
    import besser.BUML.metamodel.nn as nn_module

    safe_globals: Dict[str, Any] = {
        '__builtins__': {
            'set': set, 'list': list, 'dict': dict, 'tuple': tuple,
            'str': str, 'int': int, 'float': float, 'bool': bool,
            'len': len, 'range': range,
            'True': True, 'False': False, 'None': None,
            'print': lambda *a, **kw: None,
        },
    }
    for name in dir(nn_module):
        if not name.startswith('_'):
            safe_globals[name] = getattr(nn_module, name)

    # Strip imports (all metamodel names are already in safe_globals)
    cleaned_lines: List[str] = []
    in_import_block = False
    for line in content.splitlines():
        stripped = line.lstrip()
        if in_import_block:
            if ')' in line:
                in_import_block = False
            continue
        if stripped.startswith(('import ', 'from ')):
            if '(' in line and ')' not in line:
                in_import_block = True
            continue
        cleaned_lines.append(line)
    cleaned_content = '\n'.join(cleaned_lines)

    local_vars: Dict[str, Any] = {}
    try:
        exec(cleaned_content, safe_globals, local_vars)
    except Exception as exc:
        raise ValueError(f"Failed to execute NN BUML content: {exc}") from exc

    # Find the main NN (not a sub_nn of any other NN in local scope)
    all_nns = [v for v in local_vars.values() if isinstance(v, NN)]
    if not all_nns:
        raise ValueError("No NN instance found in the NN BUML content")

    referenced = {id(s) for nn in all_nns for s in nn.sub_nns}
    main_nns = [nn for nn in all_nns if id(nn) not in referenced]
    main_nn = main_nns[-1] if main_nns else all_nns[-1]

    return nn_model_to_json(main_nn)
