"""
NN diagram processing for converting v4 JSON to BUML format.

Reads the v4 wire shape (``{nodes, edges}``) natively. Each layer node
stores its attributes inline as ``data.attributes: dict[str, str]`` per
the spec at ``docs/source/migrations/uml-v4-shape.md`` — there are no
separate ``*Attribute*Layer*`` child elements in v4.
"""

from besser.BUML.metamodel.nn import (
    NN,
    Configuration,
    TensorOp,
    Conv1D,
    Conv2D,
    Conv3D,
    PoolingLayer,
    SimpleRNNLayer,
    LSTMLayer,
    GRULayer,
    LinearLayer,
    FlattenLayer,
    EmbeddingLayer,
    DropoutLayer,
    LayerNormLayer,
    BatchNormLayer,
    Dataset,
    Image,
)
import bisect
import ast
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.utils import sanitize_name
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data,
)
from besser.utilities.buml_code_builder.nn_explicit_attrs import mark_explicit

# Keep these aligned with the whitelists the NN metamodel setters enforce.
_ALLOWED_POOLING_TYPES = (
    'average', 'adaptive_average', 'max', 'adaptive_max',
    'global_average', 'global_max',
)
_ALLOWED_RETURN_TYPES = ('hidden', 'last', 'full')
_ALLOWED_TASK_TYPES = ('binary', 'multi_class', 'regression')
_ALLOWED_INPUT_FORMATS = ('csv', 'images')
_ALLOWED_OPTIMIZERS = ('sgd', 'adam', 'adamW', 'adagrad')
_ALLOWED_LOSS_FUNCTIONS = ('crossentropy', 'binary_crossentropy', 'mse')
_ALLOWED_METRICS = ('accuracy', 'precision', 'recall', 'f1-score', 'mae')
_ALLOWED_TNS_TYPES = (
    'concatenate', 'multiply', 'matmultiply', 'reshape', 'transpose', 'permute',
)
_ALLOWED_PADDING_TYPES = ('same', 'valid')
_CONV_EXPECTED_DIMS = {'Conv1D': 1, 'Conv2D': 2, 'Conv3D': 3}


# v4: layer attributes live as a snake_case dict on ``data.attributes``.
# This mapping translates the legacy "<Slug>Attribute" prefix used at call
# sites into the v4 dict key. Kept verbatim for source-level compatibility
# with the existing ``get_element_attribute(node, 'NameAttribute', ...)``
# style call pattern.
_ATTR_KEY_TO_NAME = {
    'NameAttribute': 'name',
    'KernelDimAttribute': 'kernel_dim',
    'StrideDimAttribute': 'stride_dim',
    'OutChannelsAttribute': 'out_channels',
    'InChannelsAttribute': 'in_channels',
    'PaddingAmountAttribute': 'padding_amount',
    'PaddingTypeAttribute': 'padding_type',
    'ActvFuncAttribute': 'actv_func',
    'NameModuleInputAttribute': 'name_module_input',
    'InputReusedAttribute': 'input_reused',
    'PermuteInAttribute': 'permute_in',
    'PermuteOutAttribute': 'permute_out',
    'PoolingTypeAttribute': 'pooling_type',
    'DimensionAttribute': 'dimension',  # see _LAYER_KIND_PREFIX for qualified-slug disambiguation
    'OutputDimAttribute': 'output_dim',
    'HiddenSizeAttribute': 'hidden_size',
    'InputSizeAttribute': 'input_size',
    'ReturnTypeAttribute': 'return_type',
    'BidirectionalAttribute': 'bidirectional',
    'DropoutAttribute': 'dropout',
    'BatchFirstAttribute': 'batch_first',
    'OutFeaturesAttribute': 'out_features',
    'InFeaturesAttribute': 'in_features',
    'StartDimAttribute': 'start_dim',
    'EndDimAttribute': 'end_dim',
    'NumEmbeddingsAttribute': 'num_embeddings',
    'EmbeddingDimAttribute': 'embedding_dim',
    'RateAttribute': 'rate',
    'NormalizedShapeAttribute': 'normalized_shape',
    'NumFeaturesAttribute': 'num_features',
    'TnsTypeAttribute': 'tns_type',
    'ConcatenateDimAttribute': 'concatenate_dim',
    'LayersOfTensorsAttribute': 'layers_of_tensors',
    'ReshapeDimAttribute': 'reshape_dim',
    'TransposeDimAttribute': 'transpose_dim',
    'PermuteDimAttribute': 'permute_dim',
    'BatchSizeAttribute': 'batch_size',
    'EpochsAttribute': 'epochs',
    'LearningRateAttribute': 'learning_rate',
    'OptimizerAttribute': 'optimizer',
    'LossFunctionAttribute': 'loss_function',
    'MetricsAttribute': 'metrics',
    'WeightDecayAttribute': 'weight_decay',
    'MomentumAttribute': 'momentum',
    'PathDataAttribute': 'path_data',
    'TaskTypeAttribute': 'task_type',
    'InputFormatAttribute': 'input_format',
    'ShapeAttribute': 'shape',
    'NormalizeAttribute': 'normalize',
}


# Some attribute slugs collide across layer kinds (e.g. `dimension` exists on
# both PoolingLayer and BatchNormalizationLayer with different semantics).
# The frontend disambiguates via `qualifySlug(layerKind, slug)` →
# `<prefix>.<slug>`. Mirror the same mapping here so v4 inputs round-trip.
_COLLIDING_SLUGS = frozenset({'dimension'})
_LAYER_KIND_PREFIX = {
    'PoolingLayer': 'pooling',
    'BatchNormalizationLayer': 'batch_normalization',
}


def get_element_attribute(node: dict, attr_key: str, elements_or_default=None, default=None):
    """Read a layer attribute.

    Production code passes a v4 node and reads from
    ``node.data.attributes`` (a dict). The legacy three-arg form
    ``(element, attr_key, elements)`` is preserved for targeted unit
    tests at ``tests/utilities/web_modeling_editor/converters/nn/`` that
    pass v3 element + elements dicts directly.
    """
    name = _ATTR_KEY_TO_NAME.get(attr_key)
    # Distinguish the two call forms by whether the third arg is a dict.
    if isinstance(elements_or_default, dict):
        elements = elements_or_default
        if name is not None:
            for attr_id in (node or {}).get('attributes', []) or []:
                attr_el = elements.get(attr_id, {})
                if attr_el.get('attributeName') == name:
                    return attr_el.get('value', default)
        # Legacy fallback: prefix-match on type with boundary check.
        for attr_id in (node or {}).get('attributes', []) or []:
            attr_el = elements.get(attr_id, {})
            attr_type = attr_el.get('type', '')
            if attr_type.startswith(attr_key):
                rest = attr_type[len(attr_key):]
                if rest == '' or rest[0].isupper():
                    return attr_el.get('value', default)
        return default

    # v4 form.
    if name is None:
        return elements_or_default if elements_or_default is not None else default
    data = node_data(node)
    attrs = data.get("attributes") or {}
    if not isinstance(attrs, dict):
        return elements_or_default if elements_or_default is not None else default
    fallback = elements_or_default if elements_or_default is not None else default
    if name in _COLLIDING_SLUGS:
        prefix = _LAYER_KIND_PREFIX.get((node or {}).get('type', ''))
        if prefix is not None:
            qualified = f'{prefix}.{name}'
            if qualified in attrs:
                return attrs[qualified]
    return attrs.get(name, fallback)


def parse_tuple_or_int(value, default=None):
    """Parse a value that could be an int, tuple, or list string."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (tuple, list)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return parts
        if value.startswith('(') and value.endswith(')'):
            inner = value[1:-1].strip()
            if not inner:
                return ()
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                return default
        try:
            return int(value)
        except ValueError:
            pass
        if ',' in value:
            parts = [p.strip() for p in value.split(',') if p.strip()]
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                return default
    return default


def parse_bool(value, default=False):
    """Parse a value that could be a boolean or string representation."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


def parse_float(value, default=None):
    """Parse a value as float."""
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '':
            return default
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"'{value}' is not a valid number — expected a float "
                f"(use '.' as the decimal separator, e.g. '0.5')."
            ) from exc
    try:
        return float(value)
    except TypeError as exc:
        raise ValueError(
            f"{type(value).__name__} value is not a valid number "
            f"— expected a float"
        ) from exc
    except ValueError:
        return default


def parse_list_of_ints(value, default=None):
    """Parse a value and ensure it returns a List[int]."""
    if value is None:
        return default
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except (ValueError, TypeError):
            return default
    if isinstance(value, tuple):
        try:
            return [int(v) for v in value]
        except (ValueError, TypeError):
            return default
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return default
        if value.startswith('(') and value.endswith(')'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return default
        try:
            return [int(value)]
        except ValueError:
            pass
        if ',' in value:
            parts = [p.strip() for p in value.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return default
    return default


def parse_dimension(value, default=None):
    """Parse a dimension value and return the string format "1D", "2D", or "3D"."""
    if value is None:
        return default
    if isinstance(value, int):
        if value in (1, 2, 3):
            return f"{value}D"
        return default
    if isinstance(value, str):
        value = value.strip().upper()
        if value in ('1D', '2D', '3D'):
            return value
        try:
            dim = int(value)
            if dim in (1, 2, 3):
                return f"{dim}D"
        except ValueError:
            pass
    return default


def _get_dataset_attr(node_or_element: dict, key: str, default=None, elements: dict = None):
    """Read a dataset attribute.

    For a v4 node, reads from ``data.attributes`` (a dict). When the
    legacy two-argument form (``element, elements``) is used, walks
    ``element.attributes`` (a list of ids) and matches by ``attributeName``.
    The two-arg form is retained only to keep targeted unit tests for
    ``create_dataset(element, elements)`` working without rewriting the
    fixtures.
    """
    if elements is not None:
        for attr_id in (node_or_element or {}).get('attributes', []) or []:
            attr_el = elements.get(attr_id, {})
            if attr_el.get('attributeName') == key:
                return attr_el.get('value', default)
        return default
    data = node_data(node_or_element)
    attrs = data.get("attributes") or {}
    if not isinstance(attrs, dict):
        return default
    return attrs.get(key, default)


def create_dataset(node_or_element: dict, elements: dict = None) -> Dataset:
    """Create a Dataset (and optional Image).

    Production code passes a single v4 node. Legacy unit tests at
    ``tests/utilities/web_modeling_editor/converters/nn/`` build minimal
    v3 element + elements dicts and call ``create_dataset(element,
    elements)``; the two-arg form is preserved for that purpose only.
    """
    def _attr(key, default=None):
        return _get_dataset_attr(node_or_element, key, default=default, elements=elements)

    name = _attr('name')
    if not name:
        raise ValueError("Dataset missing mandatory 'name' attribute")
    path_data = _attr('path_data')
    if not path_data:
        raise ValueError("Dataset missing mandatory 'path_data' attribute")

    task_type = _attr('task_type') or None
    if task_type is not None and task_type not in _ALLOWED_TASK_TYPES:
        raise ValueError(
            f"Dataset '{name}' has invalid task_type '{task_type}'. "
            f"Allowed values: {', '.join(_ALLOWED_TASK_TYPES)}."
        )
    input_format = _attr('input_format') or None
    if input_format is not None and input_format not in _ALLOWED_INPUT_FORMATS:
        raise ValueError(
            f"Dataset '{name}' has invalid input_format '{input_format}'. "
            f"Allowed values: {', '.join(_ALLOWED_INPUT_FORMATS)}."
        )

    image = None
    shape_raw = _attr('shape')
    normalize_raw = _attr('normalize')
    shape_present = shape_raw is not None and shape_raw != ''
    normalize_present = normalize_raw is not None and str(normalize_raw) != ''
    if shape_present or normalize_present or input_format == 'images':
        if shape_raw is None or shape_raw == '':
            shape = [256, 256]
        elif isinstance(shape_raw, (list, tuple)):
            shape = list(shape_raw)
        else:
            try:
                shape = ast.literal_eval(str(shape_raw))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Dataset '{name}' has malformed 'shape' attribute "
                    f"{shape_raw!r}: expected a list like [32, 32, 3]"
                ) from exc
            if not isinstance(shape, (list, tuple)):
                raise ValueError(
                    f"Dataset '{name}' 'shape' attribute must be a list "
                    f"(got {type(shape).__name__})"
                )
            shape = list(shape)
        normalize = str(normalize_raw).lower() == 'true' if normalize_raw is not None else False
        image = Image(shape=shape, normalize=normalize)

    return Dataset(name=name, path_data=path_data, task_type=task_type,
                   input_format=input_format, image=image)


def process_nn_diagram(json_data):
    """Process a v4 NN Diagram JSON and return an NN model."""
    if 'project' in json_data:
        project_data = json_data.get('project', {})
        diagrams = project_data.get('diagrams', {})
        nn_diagram_data = diagrams.get('NNDiagram')
        if isinstance(nn_diagram_data, list):
            if not nn_diagram_data:
                raise ValueError("Project contains no NNDiagram entries")
            nn_diagram_data = nn_diagram_data[0]
        elif isinstance(nn_diagram_data, dict):
            pass
        else:
            raise ValueError(
                f"Unexpected NNDiagram payload type: {type(nn_diagram_data).__name__}"
            )

        title = nn_diagram_data.get('title', 'Neural_Network')
        model_data = nn_diagram_data.get('model', {})
    else:
        title = json_data.get('title', 'Neural_Network')
        model_data = json_data.get('model', {})
        if not model_data:
            model_data = json_data

    title = sanitize_name(title)

    nodes = (model_data or {}).get('nodes') or []
    edges = (model_data or {}).get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    # Step 1: identify all NNContainer nodes.
    containers = {}  # container_id -> name
    for node in nodes:
        if node.get('type') == 'NNContainer':
            containers[node.get('id')] = sanitize_name(node_data(node).get('name', 'Neural_Network'))

    # Step 2: identify NNReference nodes.
    nn_references = {}  # node_id -> referenced_container_name
    for node in nodes:
        if node.get('type') == 'NNReference':
            data = node_data(node)
            ref_name = sanitize_name(data.get('referenceTarget') or data.get('name', ''))
            nn_references[node.get('id')] = ref_name

    # Step 3: group layers / tensorops by parent container.
    layers_by_container = {cid: {} for cid in containers}
    tensor_ops_by_container = {cid: {} for cid in containers}
    refs_by_container = {cid: [] for cid in containers}
    configuration = None

    layer_factories = {
        'Conv1DLayer': lambda n: _create_conv_layer(n, Conv1D, [1]),
        'Conv2DLayer': lambda n: _create_conv_layer(n, Conv2D, [1, 1]),
        'Conv3DLayer': lambda n: _create_conv_layer(n, Conv3D, [1, 1, 1]),
        'PoolingLayer': create_pooling_layer,
        'RNNLayer': lambda n: _create_rnn_like_layer(n, SimpleRNNLayer),
        'LSTMLayer': lambda n: _create_rnn_like_layer(n, LSTMLayer),
        'GRULayer': lambda n: _create_rnn_like_layer(n, GRULayer),
        'LinearLayer': create_linear_layer,
        'FlattenLayer': create_flatten_layer,
        'EmbeddingLayer': create_embedding_layer,
        'DropoutLayer': create_dropout_layer,
        'LayerNormalizationLayer': create_layer_norm_layer,
        'BatchNormalizationLayer': create_batch_norm_layer,
    }

    for node in nodes:
        node_id = node.get('id')
        node_type = node.get('type', '') or ''
        parent_id = node.get('parentId')

        if node_type == 'NNContainer':
            continue
        if node_type == 'NNReference':
            if parent_id and parent_id in containers:
                refs_by_container[parent_id].append((node_id, nn_references.get(node_id)))
            continue
        if node_type == 'Configuration':
            configuration = create_configuration(node)
            continue

        layer = None
        tensor_op = None
        factory = layer_factories.get(node_type)
        if factory is not None:
            layer = factory(node)
        elif node_type == 'TensorOp':
            tensor_op = create_tensor_op(node)

        if parent_id and parent_id in containers:
            if layer:
                layers_by_container[parent_id][node_id] = layer
            if tensor_op:
                tensor_ops_by_container[parent_id][node_id] = tensor_op

    # Step 4: NNNext adjacency.
    outgoing_connections = {}
    incoming_connections = {}
    for edge in edges:
        if edge.get('type') != 'NNNext':
            continue
        source_id = edge.get('source')
        target_id = edge.get('target')
        if source_id and target_id:
            outgoing_connections.setdefault(source_id, []).append(target_id)
            incoming_connections.setdefault(target_id, []).append(source_id)

    # Step 5: create NNs per container in dependency order.
    referenced_names = set(nn_references.values())
    name_by_id = dict(containers.items())
    ref_graph = {cname: set() for cname in name_by_id.values()}
    for c_id, refs in refs_by_container.items():
        c_name = name_by_id.get(c_id)
        if not c_name:
            continue
        for _, ref_name in refs:
            if ref_name:
                ref_graph[c_name].add(ref_name)

    _white, _gray, _black = 0, 1, 2
    color = {cname: _white for cname in name_by_id.values()}
    dep_order: list = []

    def _visit_container(name: str, path: list) -> None:
        if color.get(name) == _gray:
            cycle_chain = ' -> '.join(path + [name])
            raise ValueError(
                f"NNReference cycle detected among NNContainers: {cycle_chain}. "
                f"Break the cycle by removing one of the NNReference links."
            )
        if color.get(name) != _white:
            return
        color[name] = _gray
        for dep in sorted(ref_graph.get(name, ())):
            if dep in color:
                _visit_container(dep, path + [name])
        color[name] = _black
        dep_order.append(name)

    for cname in name_by_id.values():
        if color[cname] == _white:
            _visit_container(cname, [])

    name_to_id = {cname: cid for cid, cname in name_by_id.items()}
    container_order = [(name_to_id[cname], cname) for cname in dep_order]

    nn_by_name = {}
    for container_id, container_name in container_order:
        nn = NN(name=container_name)
        nn_by_name[container_name] = nn

        container_layers = layers_by_container[container_id]
        container_tensor_ops = tensor_ops_by_container[container_id]
        container_refs = refs_by_container[container_id]

        all_module_ids = (set(container_layers.keys()) |
                          set(container_tensor_ops.keys()) |
                          set(ref_id for ref_id, _ in container_refs))

        module_name_by_id = {}
        for mid, layer in container_layers.items():
            module_name_by_id[mid] = getattr(layer, 'name', '') or ''
        for mid, top in container_tensor_ops.items():
            module_name_by_id[mid] = getattr(top, 'name', '') or ''
        for rid, rname in container_refs:
            module_name_by_id[rid] = rname or ''

        ordered_modules = topological_sort(
            all_module_ids, outgoing_connections, name_by_id=module_name_by_id,
        )

        roots = [mid for mid in all_module_ids
                 if not incoming_connections.get(mid)]
        if len(roots) > 1:
            root_names = sorted(
                module_name_by_id.get(mid) or mid for mid in roots
            )
            raise ValueError(
                f"Container '{container_name}': {len(roots)} modules have no "
                f"incoming NNNext edge ({', '.join(root_names)}); all modules "
                f"after the entry point must be connected via NNNext."
            )

        container_refs_dict = dict(container_refs)
        for module_id in ordered_modules:
            if module_id in container_layers:
                nn.add_layer(container_layers[module_id])
            elif module_id in container_tensor_ops:
                nn.add_tensor_op(container_tensor_ops[module_id])
            elif module_id in container_refs_dict:
                ref_name = container_refs_dict.get(module_id)
                if not ref_name:
                    raise ValueError(
                        f"NNReference in container '{container_name}' has no "
                        f"'referenceTarget' field"
                    )
                if ref_name not in nn_by_name:
                    raise ValueError(
                        f"NNReference '{ref_name}' in container "
                        f"'{container_name}' does not match any NNContainer"
                    )
                nn.add_sub_nn(nn_by_name[ref_name])

    # Step 6: pick the top-level NN.
    top_level = [cname for _, cname in containers.items() if cname not in referenced_names]
    if len(top_level) > 1:
        raise ValueError(
            f"NN diagram contains {len(top_level)} top-level NNContainers "
            f"({', '.join(top_level)}); exactly one is required. Connect the "
            f"others via NNReference elements to mark them as sub-networks."
        )
    if top_level:
        main_nn = nn_by_name.get(top_level[0])
    elif nn_by_name:
        raise ValueError(
            "NN diagram has no top-level NNContainer: every container is "
            "referenced by an NNReference. Remove one of the references."
        )
    else:
        main_nn = NN(name=title)

    if configuration:
        main_nn.add_configuration(configuration)

    train_nodes = [n for n in nodes if n.get('type') == 'TrainingDataset']
    test_nodes = [n for n in nodes if n.get('type') == 'TestDataset']

    if len(train_nodes) > 1:
        raise ValueError(
            f"NN diagram contains {len(train_nodes)} TrainingDataset elements; "
            f"only one is supported per NN model"
        )
    if len(test_nodes) > 1:
        raise ValueError(
            f"NN diagram contains {len(test_nodes)} TestDataset elements; "
            f"only one is supported per NN model"
        )
    if train_nodes:
        main_nn.add_train_data(create_dataset(train_nodes[0]))
    if test_nodes:
        main_nn.add_test_data(create_dataset(test_nodes[0]))

    main_nn.validate(raise_exception=True)
    return main_nn


def topological_sort(module_ids, outgoing_connections, name_by_id=None):
    """Topologically sort modules based on their NNNext connections."""
    in_degree = {mid: 0 for mid in module_ids}
    for source_id, targets in outgoing_connections.items():
        if source_id in module_ids:
            for target_id in targets:
                if target_id in in_degree:
                    in_degree[target_id] += 1

    def _sort_key(mid):
        if name_by_id is None:
            return mid
        return (name_by_id.get(mid, ''), mid)

    queue = sorted(
        (mid for mid, deg in in_degree.items() if deg == 0),
        key=_sort_key,
    )
    result = []
    while queue:
        current = queue.pop(0)
        result.append(current)
        for target_id in outgoing_connections.get(current, []):
            if target_id in in_degree:
                in_degree[target_id] -= 1
                if in_degree[target_id] == 0:
                    if name_by_id is None:
                        bisect.insort(queue, target_id)
                    else:
                        tkey = _sort_key(target_id)
                        lo, hi = 0, len(queue)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if _sort_key(queue[mid]) < tkey:
                                lo = mid + 1
                            else:
                                hi = mid
                        queue.insert(lo, target_id)

    remaining = [mid for mid in module_ids if mid not in result]
    if remaining:
        raise ValueError(
            "NNNext relationships form a cycle involving "
            f"{len(remaining)} module(s); cannot determine layer order"
        )
    return result


# ---------------------------------------------------------------------------
# Layer creation functions
# ---------------------------------------------------------------------------

def _create_conv_layer(node, conv_class, default_stride):
    """Create a convolutional layer (Conv1D, Conv2D, or Conv3D) from v4 data."""
    class_name = conv_class.__name__
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError(f"{class_name} layer missing mandatory 'name' attribute")

    kernel_dim_raw = get_element_attribute(node, 'KernelDimAttribute')
    kernel_dim = parse_list_of_ints(kernel_dim_raw)
    if kernel_dim is None:
        raise ValueError(f"{class_name} layer '{name}' missing mandatory 'kernel_dim' attribute")

    expected_dim = _CONV_EXPECTED_DIMS.get(class_name)
    if expected_dim is not None and len(kernel_dim) != expected_dim:
        raise ValueError(
            f"{class_name} layer '{name}' kernel_dim has {len(kernel_dim)} "
            f"element(s), expected {expected_dim}."
        )

    out_channels_raw = get_element_attribute(node, 'OutChannelsAttribute')
    out_channels = parse_tuple_or_int(out_channels_raw)
    if out_channels is None:
        raise ValueError(f"{class_name} layer '{name}' missing mandatory 'out_channels' attribute")

    layer = conv_class(
        name=sanitize_name(name),
        kernel_dim=kernel_dim,
        out_channels=out_channels,
    )

    stride = get_element_attribute(node, 'StrideDimAttribute')
    if stride is not None:
        stride_dim_parsed = parse_list_of_ints(stride, default_stride)
        if expected_dim is not None and stride_dim_parsed is not None \
                and len(stride_dim_parsed) != expected_dim:
            raise ValueError(
                f"{class_name} layer '{name}' stride_dim has "
                f"{len(stride_dim_parsed)} element(s), expected {expected_dim}."
            )
        layer.stride_dim = stride_dim_parsed

    in_channels = get_element_attribute(node, 'InChannelsAttribute')
    if in_channels is not None:
        layer.in_channels = parse_tuple_or_int(in_channels)

    padding = get_element_attribute(node, 'PaddingAmountAttribute')
    if padding is not None:
        layer.padding_amount = parse_tuple_or_int(padding, 0)
        mark_explicit(layer, 'padding_amount')

    padding_type = get_element_attribute(node, 'PaddingTypeAttribute')
    if padding_type:
        if padding_type not in _ALLOWED_PADDING_TYPES:
            raise ValueError(
                f"{class_name} layer '{name}' has invalid padding_type "
                f"'{padding_type}'. Allowed values: "
                f"{', '.join(_ALLOWED_PADDING_TYPES)}."
            )
        layer.padding_type = padding_type
        mark_explicit(layer, 'padding_type')

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    permute_in = get_element_attribute(node, 'PermuteInAttribute')
    if permute_in is not None and str(permute_in).strip() != '':
        layer.permute_in = parse_bool(permute_in)
        mark_explicit(layer, 'permute_in')

    permute_out = get_element_attribute(node, 'PermuteOutAttribute')
    if permute_out is not None and str(permute_out).strip() != '':
        layer.permute_out = parse_bool(permute_out)
        mark_explicit(layer, 'permute_out')

    return layer


def create_pooling_layer(node):
    """Create a PoolingLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("PoolingLayer missing mandatory 'name' attribute")

    pooling_type = get_element_attribute(node, 'PoolingTypeAttribute')
    if not pooling_type:
        raise ValueError(f"PoolingLayer '{name}' missing mandatory 'pooling_type' attribute")
    if pooling_type not in _ALLOWED_POOLING_TYPES:
        raise ValueError(
            f"PoolingLayer '{name}' has invalid pooling_type '{pooling_type}'. "
            f"Allowed values: {', '.join(_ALLOWED_POOLING_TYPES)}."
        )

    dimension_raw = get_element_attribute(node, 'DimensionAttribute')
    dimension = parse_dimension(dimension_raw)
    if dimension is None:
        raise ValueError(f"PoolingLayer '{name}' missing mandatory 'dimension' attribute")

    kernel_dim_raw = get_element_attribute(node, 'KernelDimAttribute')
    kernel_dim = parse_list_of_ints(kernel_dim_raw)
    if kernel_dim is None and not (pooling_type.startswith("adaptive") or pooling_type.startswith("global")):
        if dimension == "1D":
            kernel_dim = [2]
        elif dimension == "2D":
            kernel_dim = [2, 2]
        elif dimension == "3D":
            kernel_dim = [2, 2, 2]

    stride_raw = get_element_attribute(node, 'StrideDimAttribute')
    stride_dim = parse_list_of_ints(stride_raw)

    layer = PoolingLayer(
        name=sanitize_name(name),
        pooling_type=pooling_type,
        dimension=dimension,
        kernel_dim=kernel_dim,
        stride_dim=stride_dim,
    )

    padding = get_element_attribute(node, 'PaddingAmountAttribute')
    if padding is not None:
        layer.padding_amount = parse_tuple_or_int(padding, 0)
        mark_explicit(layer, 'padding_amount')

    padding_type = get_element_attribute(node, 'PaddingTypeAttribute')
    if padding_type:
        if padding_type not in _ALLOWED_PADDING_TYPES:
            raise ValueError(
                f"PoolingLayer '{name}' has invalid padding_type "
                f"'{padding_type}'. Allowed values: "
                f"{', '.join(_ALLOWED_PADDING_TYPES)}."
            )
        layer.padding_type = padding_type
        mark_explicit(layer, 'padding_type')

    output_dim = get_element_attribute(node, 'OutputDimAttribute')
    if output_dim is not None:
        layer.output_dim = parse_list_of_ints(output_dim)
        mark_explicit(layer, 'output_dim')

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    permute_in = get_element_attribute(node, 'PermuteInAttribute')
    if permute_in is not None and str(permute_in).strip() != '':
        layer.permute_in = parse_bool(permute_in)
        mark_explicit(layer, 'permute_in')

    permute_out = get_element_attribute(node, 'PermuteOutAttribute')
    if permute_out is not None and str(permute_out).strip() != '':
        layer.permute_out = parse_bool(permute_out)
        mark_explicit(layer, 'permute_out')

    return layer


def _create_rnn_like_layer(node, rnn_class):
    """Create an RNN-like layer (SimpleRNNLayer, LSTMLayer, or GRULayer)."""
    class_name = rnn_class.__name__
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError(f"{class_name} missing mandatory 'name' attribute")

    hidden_size_raw = get_element_attribute(node, 'HiddenSizeAttribute')
    hidden_size = parse_tuple_or_int(hidden_size_raw)
    if hidden_size is None:
        raise ValueError(f"{class_name} '{name}' missing mandatory 'hidden_size' attribute")

    layer = rnn_class(
        name=sanitize_name(name),
        hidden_size=hidden_size,
    )

    return_type = get_element_attribute(node, 'ReturnTypeAttribute')
    if return_type:
        if return_type not in _ALLOWED_RETURN_TYPES:
            raise ValueError(
                f"{class_name} '{name}' has invalid return_type "
                f"'{return_type}'. Allowed values: "
                f"{', '.join(_ALLOWED_RETURN_TYPES)}."
            )
        layer.return_type = return_type

    input_size = get_element_attribute(node, 'InputSizeAttribute')
    if input_size is not None:
        layer.input_size = parse_tuple_or_int(input_size)

    bidirectional = get_element_attribute(node, 'BidirectionalAttribute')
    if bidirectional is not None:
        layer.bidirectional = parse_bool(bidirectional)
        mark_explicit(layer, 'bidirectional')

    dropout = get_element_attribute(node, 'DropoutAttribute')
    if dropout is not None:
        layer.dropout = parse_float(dropout, 0.0)
        mark_explicit(layer, 'dropout')

    batch_first = get_element_attribute(node, 'BatchFirstAttribute')
    if batch_first is not None:
        layer.batch_first = parse_bool(batch_first)
        mark_explicit(layer, 'batch_first')

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_linear_layer(node):
    """Create a LinearLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("LinearLayer missing mandatory 'name' attribute")

    out_features_raw = get_element_attribute(node, 'OutFeaturesAttribute')
    out_features = parse_tuple_or_int(out_features_raw)
    if out_features is None:
        raise ValueError(f"LinearLayer '{name}' missing mandatory 'out_features' attribute")

    layer = LinearLayer(
        name=sanitize_name(name),
        out_features=out_features,
    )

    in_features = get_element_attribute(node, 'InFeaturesAttribute')
    if in_features is not None:
        layer.in_features = parse_tuple_or_int(in_features)

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_flatten_layer(node):
    """Create a FlattenLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("FlattenLayer missing mandatory 'name' attribute")

    layer = FlattenLayer(name=sanitize_name(name))

    start_dim = get_element_attribute(node, 'StartDimAttribute')
    if start_dim is not None:
        layer.start_dim = parse_tuple_or_int(start_dim)
        mark_explicit(layer, 'start_dim')

    end_dim = get_element_attribute(node, 'EndDimAttribute')
    if end_dim is not None:
        layer.end_dim = parse_tuple_or_int(end_dim)
        mark_explicit(layer, 'end_dim')

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_embedding_layer(node):
    """Create an EmbeddingLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("EmbeddingLayer missing mandatory 'name' attribute")

    num_embeddings_raw = get_element_attribute(node, 'NumEmbeddingsAttribute')
    num_embeddings = parse_tuple_or_int(num_embeddings_raw)
    if num_embeddings is None:
        raise ValueError(f"EmbeddingLayer '{name}' missing mandatory 'num_embeddings' attribute")

    embedding_dim_raw = get_element_attribute(node, 'EmbeddingDimAttribute')
    embedding_dim = parse_tuple_or_int(embedding_dim_raw)
    if embedding_dim is None:
        raise ValueError(f"EmbeddingLayer '{name}' missing mandatory 'embedding_dim' attribute")

    layer = EmbeddingLayer(
        name=sanitize_name(name),
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_dropout_layer(node):
    """Create a DropoutLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("DropoutLayer missing mandatory 'name' attribute")

    rate_raw = get_element_attribute(node, 'RateAttribute')
    rate = parse_float(rate_raw)
    if rate is None:
        raise ValueError(f"DropoutLayer '{name}' missing mandatory 'rate' attribute")

    layer = DropoutLayer(
        name=sanitize_name(name),
        rate=rate,
    )

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_layer_norm_layer(node):
    """Create a LayerNormLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("LayerNormLayer missing mandatory 'name' attribute")

    normalized_shape_raw = get_element_attribute(node, 'NormalizedShapeAttribute')
    normalized_shape = parse_tuple_or_int(normalized_shape_raw)
    if normalized_shape is None:
        raise ValueError(f"LayerNormLayer '{name}' missing mandatory 'normalized_shape' attribute")

    layer = LayerNormLayer(
        name=sanitize_name(name),
        normalized_shape=normalized_shape,
    )

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_batch_norm_layer(node):
    """Create a BatchNormLayer from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("BatchNormLayer missing mandatory 'name' attribute")

    num_features_raw = get_element_attribute(node, 'NumFeaturesAttribute')
    num_features = parse_tuple_or_int(num_features_raw)
    if num_features is None:
        raise ValueError(f"BatchNormLayer '{name}' missing mandatory 'num_features' attribute")

    dimension_raw = get_element_attribute(node, 'DimensionAttribute')
    dimension = parse_dimension(dimension_raw)
    if dimension is None:
        raise ValueError(f"BatchNormLayer '{name}' missing mandatory 'dimension' attribute")

    layer = BatchNormLayer(
        name=sanitize_name(name),
        num_features=num_features,
        dimension=dimension,
    )

    actv_func = get_element_attribute(node, 'ActvFuncAttribute')
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(node, 'NameModuleInputAttribute')
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_tensor_op(node):
    """Create a TensorOp from a v4 node."""
    name = get_element_attribute(node, 'NameAttribute')
    if not name:
        raise ValueError("TensorOp missing mandatory 'name' attribute")

    tns_type = get_element_attribute(node, 'TnsTypeAttribute')
    if not tns_type:
        raise ValueError(f"TensorOp '{name}' missing mandatory 'tns_type' attribute")
    if tns_type not in _ALLOWED_TNS_TYPES:
        raise ValueError(
            f"TensorOp '{name}' has invalid tns_type '{tns_type}'. "
            f"Allowed values: {', '.join(_ALLOWED_TNS_TYPES)}."
        )

    concatenate_dim = get_element_attribute(node, 'ConcatenateDimAttribute')
    if concatenate_dim is not None:
        concatenate_dim = parse_tuple_or_int(concatenate_dim)

    layers_of_tensors_raw = get_element_attribute(node, 'LayersOfTensorsAttribute')
    layers_of_tensors = None
    if layers_of_tensors_raw:
        def _coerce(item):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                return item
            s = str(item).strip().strip("'\"")
            if not s:
                return s
            try:
                return float(s)
            except ValueError:
                return s
        if isinstance(layers_of_tensors_raw, str):
            val = layers_of_tensors_raw.strip()
            if val.startswith('[') and val.endswith(']'):
                val = val[1:-1]
            layers_of_tensors = [_coerce(layer) for layer in val.split(',') if layer.strip()]
        elif isinstance(layers_of_tensors_raw, list):
            layers_of_tensors = [_coerce(layer) for layer in layers_of_tensors_raw]

    reshape_dim = get_element_attribute(node, 'ReshapeDimAttribute')
    if reshape_dim is not None:
        reshape_dim = parse_list_of_ints(reshape_dim)

    transpose_dim = get_element_attribute(node, 'TransposeDimAttribute')
    if transpose_dim is not None:
        transpose_dim = parse_list_of_ints(transpose_dim)

    permute_dim = get_element_attribute(node, 'PermuteDimAttribute')
    if permute_dim is not None:
        permute_dim = parse_list_of_ints(permute_dim)

    types_requiring_layers = ['multiply', 'matmultiply', 'concatenate']
    if tns_type in types_requiring_layers and not layers_of_tensors:
        raise ValueError(
            f"TensorOp '{name}' of type '{tns_type}' requires 'layers_of_tensors' attribute. "
            f"Please add the 'layers_of_tensors' attribute specifying which layers to use."
        )

    if tns_type == 'concatenate' and concatenate_dim is None:
        raise ValueError(
            f"TensorOp '{name}' of type 'concatenate' requires 'concatenate_dim' attribute. "
            f"Please specify the dimension along which to concatenate."
        )

    if tns_type == 'permute' and permute_dim is None:
        raise ValueError(
            f"TensorOp '{name}' of type 'permute' requires 'permute_dim' attribute. "
            f"Please specify the permutation dimensions."
        )

    if tns_type == 'transpose' and transpose_dim is None:
        raise ValueError(
            f"TensorOp '{name}' of type 'transpose' requires 'transpose_dim' attribute. "
            f"Please specify the transpose dimensions."
        )

    if tns_type == 'reshape' and reshape_dim is None:
        raise ValueError(
            f"TensorOp '{name}' of type 'reshape' requires 'reshape_dim' attribute. "
            f"Please specify the target shape."
        )

    tensor_op_params = {
        'name': sanitize_name(name),
        'tns_type': tns_type,
    }

    if tns_type == 'concatenate':
        tensor_op_params['concatenate_dim'] = concatenate_dim
        tensor_op_params['layers_of_tensors'] = layers_of_tensors
    elif tns_type in ('multiply', 'matmultiply'):
        tensor_op_params['layers_of_tensors'] = layers_of_tensors
    elif tns_type == 'reshape':
        tensor_op_params['reshape_dim'] = reshape_dim
    elif tns_type == 'transpose':
        tensor_op_params['transpose_dim'] = transpose_dim
    elif tns_type == 'permute':
        tensor_op_params['permute_dim'] = permute_dim

    tensor_op = TensorOp(**tensor_op_params)

    input_reused = get_element_attribute(node, 'InputReusedAttribute')
    if input_reused is not None:
        tensor_op.input_reused = parse_bool(input_reused)
        mark_explicit(tensor_op, 'input_reused')

    return tensor_op


def create_configuration(node):
    """Create a Configuration from a v4 node."""
    batch_size_raw = get_element_attribute(node, 'BatchSizeAttribute')
    batch_size = parse_tuple_or_int(batch_size_raw)
    if batch_size is None:
        raise ValueError("Configuration missing mandatory 'batch_size' attribute")

    epochs_raw = get_element_attribute(node, 'EpochsAttribute')
    epochs = parse_tuple_or_int(epochs_raw)
    if epochs is None:
        raise ValueError("Configuration missing mandatory 'epochs' attribute")

    learning_rate_raw = get_element_attribute(node, 'LearningRateAttribute')
    learning_rate = parse_float(learning_rate_raw)
    if learning_rate is None:
        raise ValueError("Configuration missing mandatory 'learning_rate' attribute")

    optimizer = get_element_attribute(node, 'OptimizerAttribute')
    if not optimizer:
        raise ValueError("Configuration missing mandatory 'optimizer' attribute")
    if optimizer not in _ALLOWED_OPTIMIZERS:
        raise ValueError(
            f"Configuration has invalid optimizer '{optimizer}'. "
            f"Allowed values: {', '.join(_ALLOWED_OPTIMIZERS)}."
        )

    loss_function = get_element_attribute(node, 'LossFunctionAttribute')
    if not loss_function:
        raise ValueError("Configuration missing mandatory 'loss_function' attribute")
    if loss_function not in _ALLOWED_LOSS_FUNCTIONS:
        raise ValueError(
            f"Configuration has invalid loss_function '{loss_function}'. "
            f"Allowed values: {', '.join(_ALLOWED_LOSS_FUNCTIONS)}."
        )

    metrics_str = get_element_attribute(node, 'MetricsAttribute')
    if not metrics_str:
        raise ValueError("Configuration missing mandatory 'metrics' attribute")

    if isinstance(metrics_str, str):
        if metrics_str.startswith('[') and metrics_str.endswith(']'):
            inner = metrics_str[1:-1].strip()
            metrics = [m.strip().strip("'\"") for m in inner.split(',') if m.strip()]
        else:
            metrics = [m.strip().strip("'\"") for m in metrics_str.split(',') if m.strip()]
    elif isinstance(metrics_str, list):
        metrics = [m.strip("'\"") if isinstance(m, str) else m for m in metrics_str]
    else:
        raise ValueError("Configuration 'metrics' attribute has invalid format")
    invalid_metrics = [m for m in metrics if m not in _ALLOWED_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Configuration has invalid metric(s) {invalid_metrics}. "
            f"Allowed values: {', '.join(_ALLOWED_METRICS)}."
        )

    config = Configuration(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=metrics,
    )

    weight_decay = get_element_attribute(node, 'WeightDecayAttribute')
    if weight_decay is not None:
        config.weight_decay = parse_float(weight_decay, 0.0)
        mark_explicit(config, 'weight_decay')

    momentum = get_element_attribute(node, 'MomentumAttribute')
    if momentum is not None:
        config.momentum = parse_float(momentum, 0.0)
        mark_explicit(config, 'momentum')

    return config
