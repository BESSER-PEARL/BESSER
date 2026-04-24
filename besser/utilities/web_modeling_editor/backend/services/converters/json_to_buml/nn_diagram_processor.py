"""
NN diagram processing for converting JSON to BUML format.

This module converts Neural Network diagram JSON from the web editor
to BESSER's B-UML NN metamodel (NN, Layer, TensorOp, Configuration).
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
from besser.utilities.buml_code_builder.nn_explicit_attrs import mark_explicit

# Keep these aligned with the whitelists the NN metamodel setters enforce
# (besser/BUML/metamodel/nn/neural_network.py). Re-asserting them here lets
# us raise a user-facing error that names the diagram element, before the
# raw metamodel ValueError (which knows nothing about which layer failed).
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
_CONV_EXPECTED_DIMS = {'Conv1D': 1, 'Conv2D': 2, 'Conv3D': 3}


def _get_attr_by_name(element: dict, attr_name: str, elements: dict, default=None):
    """Get an attribute value by its attributeName field (used for dataset elements)."""
    for attr_id in element.get('attributes', []):
        attr_el = elements.get(attr_id, {})
        if attr_el.get('attributeName') == attr_name:
            return attr_el.get('value', default)
    return default


def create_dataset(element: dict, elements: dict) -> Dataset:
    """Create a Dataset (and optional Image) from a TrainingDataset/TestDataset element."""
    name = _get_attr_by_name(element, 'name', elements)
    if not name:
        raise ValueError("Dataset missing mandatory 'name' attribute")
    path_data = _get_attr_by_name(element, 'path_data', elements)
    if not path_data:
        raise ValueError("Dataset missing mandatory 'path_data' attribute")

    task_type = _get_attr_by_name(element, 'task_type', elements) or None
    if task_type is not None and task_type not in _ALLOWED_TASK_TYPES:
        raise ValueError(
            f"Dataset '{name}' has invalid task_type '{task_type}'. "
            f"Allowed values: {', '.join(_ALLOWED_TASK_TYPES)}."
        )
    input_format = _get_attr_by_name(element, 'input_format', elements) or None
    if input_format is not None and input_format not in _ALLOWED_INPUT_FORMATS:
        raise ValueError(
            f"Dataset '{name}' has invalid input_format '{input_format}'. "
            f"Allowed values: {', '.join(_ALLOWED_INPUT_FORMATS)}."
        )

    image = None
    if input_format == 'images':
        shape_raw = _get_attr_by_name(element, 'shape', elements)
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
        normalize_raw = _get_attr_by_name(element, 'normalize', elements)
        normalize = str(normalize_raw).lower() == 'true' if normalize_raw is not None else False
        image = Image(shape=shape, normalize=normalize)

    return Dataset(name=name, path_data=path_data, task_type=task_type,
                   input_format=input_format, image=image)


# Map the `attr_key` prefix used at call sites (e.g. 'NameAttribute',
# 'KernelDimAttribute') to the `attributeName` field emitted by the
# frontend/converter (e.g. 'name', 'kernel_dim'). Using the `attributeName`
# field rather than a substring match on `type` avoids collisions between
# attributes whose type names share a prefix/suffix.
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
    'DimensionAttribute': 'dimension',
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
}


def get_element_attribute(element: dict, attr_key: str, elements: dict, default=None):
    """
    Get an attribute value from an element's attribute children.

    Lookup is done via the `attributeName` field emitted by the frontend (which
    is stable across layer types), with an exact-prefix fallback on `type`
    (e.g. 'NameAttributeConv2D' starts with 'NameAttribute'). Substring
    matching is deliberately avoided to prevent collisions between attributes
    whose type names share segments (e.g. 'NameAttribute' vs
    'NameModuleInputAttribute').

    Args:
        element: The parent element
        attr_key: The attribute suffix to look for (e.g., 'NameAttribute')
        elements: All elements dictionary
        default: Default value if not found

    Returns:
        The attribute value or default
    """
    attribute_name = _ATTR_KEY_TO_NAME.get(attr_key)
    attribute_ids = element.get('attributes', [])

    for attr_id in attribute_ids:
        attr_element = elements.get(attr_id, {})

        # Preferred: match by `attributeName` (set consistently by the converter).
        if attribute_name is not None and attr_element.get('attributeName') == attribute_name:
            return attr_element.get('value', default)

        # Fallback for legacy payloads that only carry `type`: require an
        # exact prefix match so 'NameAttribute' doesn't pick up
        # 'NameModuleInputAttributeConv2D'.
        attr_type = attr_element.get('type', '')
        if attr_type.startswith(attr_key):
            # Reject partial matches where the next char isn't the layer-suffix
            # boundary (e.g. 'NameAttributeExtra' should not match 'NameAttribute').
            rest = attr_type[len(attr_key):]
            if rest == '' or rest[0].isupper():
                return attr_element.get('value', default)

    return default


def parse_tuple_or_int(value, default=None):
    """
    Parse a value that could be an int, tuple, or list string.

    Args:
        value: The value to parse (could be "3", "(3, 3)", "[3]", "3,3", etc.)
        default: Default value if parsing fails

    Returns:
        int, tuple of ints, or list of ints (for list notation)
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, (tuple, list)):
        return value

    if isinstance(value, str):
        value = value.strip()

        # Handle empty string
        if not value:
            return default

        # Check if it's a list notation [...]
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                # Try as strings if not ints
                return parts

        # Check if it's a tuple notation (...)
        if value.startswith('(') and value.endswith(')'):
            inner = value[1:-1].strip()
            if not inner:
                return ()
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                return default

        # Try to parse as int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to parse as comma-separated values
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
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_list_of_ints(value, default=None):
    """
    Parse a value and ensure it returns a List[int].

    The metamodel expects kernel_dim, stride_dim etc. to be List[int].

    Args:
        value: The value to parse (could be "[3]", "(3, 3)", "3", 3, [3], etc.)
        default: Default value if parsing fails (should be a list like [3])

    Returns:
        List[int]: The parsed value as a list of integers
    """
    if value is None:
        return default

    # If already a list, ensure all elements are ints
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except (ValueError, TypeError):
            return default

    # If tuple, convert to list
    if isinstance(value, tuple):
        try:
            return [int(v) for v in value]
        except (ValueError, TypeError):
            return default

    # If single int, wrap in list
    if isinstance(value, (int, float)):
        return [int(value)]

    if isinstance(value, str):
        value = value.strip()

        # Handle empty string
        if not value:
            return default

        # Check if it's a list notation [...]
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return default

        # Check if it's a tuple notation (...)
        if value.startswith('(') and value.endswith(')'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return default

        # Try to parse as single int
        try:
            return [int(value)]
        except ValueError:
            pass

        # Try to parse as comma-separated values
        if ',' in value:
            parts = [p.strip() for p in value.split(',') if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return default

    return default


def parse_dimension(value, default=None):
    """
    Parse a dimension value and return the string format "1D", "2D", or "3D".

    The metamodel expects dimension as a string like "1D", "2D", "3D".

    Args:
        value: The value to parse (could be "1D", "2D", "3D", "1d", "2d", "3d", 1, 2, 3, etc.)
        default: Default value if parsing fails (None for mandatory attributes)

    Returns:
        str: The dimension as a string ("1D", "2D", or "3D"), or None if invalid
    """
    if value is None:
        return default

    # If already an integer, convert to string format
    if isinstance(value, int):
        if value in (1, 2, 3):
            return f"{value}D"
        return default

    if isinstance(value, str):
        value = value.strip().upper()
        # Already in correct format
        if value in ('1D', '2D', '3D'):
            return value
        # Try parsing as integer
        try:
            dim = int(value)
            if dim in (1, 2, 3):
                return f"{dim}D"
        except ValueError:
            pass

    return default


def process_nn_diagram(json_data):
    """
    Process NN Diagram JSON and return an NN model.

    This function:
    1. Identifies all NNContainers and creates separate NN objects for each
    2. Parses layers/TensorOps within each container
    3. Uses NNNext relationships to determine layer ordering within each container
    4. Handles NNReference elements by using add_sub_nn() to reference other NNs
    5. Returns the main NN model (with sub_nns for referenced containers)

    Args:
        json_data: The JSON data from the frontend. Can be in two formats:
            1. Diagram format: {"title": "...", "model": {"elements": {...}}}
            2. Project format: {"project": {"diagrams": {"NNDiagram": {"model": {...}}}}}

    Returns:
        NN: A B-UML NN model (main NN with sub_nns for referenced containers)
    """
    # Handle both project format and diagram format
    if 'project' in json_data:
        project_data = json_data.get('project', {})
        diagrams = project_data.get('diagrams', {})
        nn_diagram_data = diagrams.get('NNDiagram')
        if isinstance(nn_diagram_data, list):
            if not nn_diagram_data:
                raise ValueError("Project contains no NNDiagram entries")
            nn_diagram_data = nn_diagram_data[0]
        elif isinstance(nn_diagram_data, dict):
            # Legacy single-diagram shape: use as-is.
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

    elements = model_data.get('elements', {})
    relationships = model_data.get('relationships', {})

    # Step 1: Identify all NNContainers and their names
    containers = {}  # container_id -> container_name
    container_by_name = {}  # container_name -> container_id
    for elem_id, elem in elements.items():
        if elem.get('type') == 'NNContainer':
            name = sanitize_name(elem.get('name', 'Neural_Network'))
            containers[elem_id] = name
            container_by_name[name] = elem_id

    # Step 2: Identify NNReference elements and what they reference
    nn_references = {}  # element_id -> referenced_container_name
    for elem_id, elem in elements.items():
        if elem.get('type') == 'NNReference':
            ref_name = sanitize_name(elem.get('referencedNN', elem.get('name', '')))
            nn_references[elem_id] = ref_name

    # Step 3: Group layers/tensorops by their owner container
    # Also track elements without owner (global Configuration)
    layers_by_container = {cid: {} for cid in containers}  # container_id -> {elem_id -> layer}
    tensor_ops_by_container = {cid: {} for cid in containers}
    refs_by_container = {cid: [] for cid in containers}  # container_id -> [(elem_id, ref_name)]
    configuration = None

    for element_id, element in elements.items():
        element_type = element.get('type', '')
        owner_id = element.get('owner')

        # Skip attribute elements
        if 'Attribute' in element_type:
            continue

        # Skip containers themselves
        if element_type == 'NNContainer':
            continue

        # Handle NNReference
        if element_type == 'NNReference':
            if owner_id and owner_id in containers:
                ref_name = nn_references.get(element_id)
                refs_by_container[owner_id].append((element_id, ref_name))
            continue

        # Handle Configuration (global, no owner)
        if element_type == 'Configuration':
            configuration = create_configuration(element, elements)
            continue

        # Create layer or tensor_op
        layer = None
        tensor_op = None

        if element_type == 'Conv1DLayer':
            layer = create_conv1d_layer(element, elements)
        elif element_type == 'Conv2DLayer':
            layer = create_conv2d_layer(element, elements)
        elif element_type == 'Conv3DLayer':
            layer = create_conv3d_layer(element, elements)
        elif element_type == 'PoolingLayer':
            layer = create_pooling_layer(element, elements)
        elif element_type == 'RNNLayer':
            layer = create_rnn_layer(element, elements)
        elif element_type == 'LSTMLayer':
            layer = create_lstm_layer(element, elements)
        elif element_type == 'GRULayer':
            layer = create_gru_layer(element, elements)
        elif element_type == 'LinearLayer':
            layer = create_linear_layer(element, elements)
        elif element_type == 'FlattenLayer':
            layer = create_flatten_layer(element, elements)
        elif element_type == 'EmbeddingLayer':
            layer = create_embedding_layer(element, elements)
        elif element_type == 'DropoutLayer':
            layer = create_dropout_layer(element, elements)
        elif element_type == 'LayerNormalizationLayer':
            layer = create_layer_norm_layer(element, elements)
        elif element_type == 'BatchNormalizationLayer':
            layer = create_batch_norm_layer(element, elements)
        elif element_type == 'TensorOp':
            tensor_op = create_tensor_op(element, elements)

        # Add to appropriate container
        if owner_id and owner_id in containers:
            if layer:
                layers_by_container[owner_id][element_id] = layer
            if tensor_op:
                tensor_ops_by_container[owner_id][element_id] = tensor_op

    # Step 4: Build NNNext adjacency lists (per container)
    outgoing_connections = {}  # source_id -> [target_ids]
    incoming_connections = {}  # target_id -> [source_ids]

    for rel_id, rel in relationships.items():
        rel_type = rel.get('type', '')
        if rel_type == 'NNNext':
            source_id = rel.get('source', {}).get('element')
            target_id = rel.get('target', {}).get('element')

            if source_id and target_id:
                if source_id not in outgoing_connections:
                    outgoing_connections[source_id] = []
                outgoing_connections[source_id].append(target_id)

                if target_id not in incoming_connections:
                    incoming_connections[target_id] = []
                incoming_connections[target_id].append(source_id)

    # Step 5: Create NN objects for each container
    nn_by_name = {}  # container_name -> NN object

    # First, create NNs for containers that are referenced (no NNReference pointing to them)
    # These are "standalone" sub-networks
    referenced_names = set(nn_references.values())

    # Build a reference graph (container_name -> set of referenced container_names)
    # so we can order containers by dependency: referenced containers must be
    # created before the containers that reference them, at any depth.
    name_by_id = dict(containers.items())
    ref_graph = {cname: set() for cname in name_by_id.values()}
    for c_id, refs in refs_by_container.items():
        c_name = name_by_id.get(c_id)
        if not c_name:
            continue
        for _, ref_name in refs:
            if ref_name:
                ref_graph[c_name].add(ref_name)

    # Depth-first post-order traversal: emits each container after all its
    # transitive dependencies. GRAY coloring detects any cycle (direct or
    # through intermediaries) and surfaces a clear error listing the chain.
    _WHITE, _GRAY, _BLACK = 0, 1, 2
    color = {cname: _WHITE for cname in name_by_id.values()}
    dep_order: list = []

    def _visit_container(name: str, path: list) -> None:
        if color.get(name) == _GRAY:
            cycle_chain = ' -> '.join(path + [name])
            raise ValueError(
                f"NNReference cycle detected among NNContainers: {cycle_chain}. "
                f"Break the cycle by removing one of the NNReference links."
            )
        if color.get(name) != _WHITE:
            return
        color[name] = _GRAY
        for dep in sorted(ref_graph.get(name, ())):
            if dep in color:  # ignore references that don't resolve (handled later)
                _visit_container(dep, path + [name])
        color[name] = _BLACK
        dep_order.append(name)

    for cname in name_by_id.values():
        if color[cname] == _WHITE:
            _visit_container(cname, [])

    name_to_id = {cname: cid for cid, cname in name_by_id.items()}
    container_order = [(name_to_id[cname], cname) for cname in dep_order]

    for container_id, container_name in container_order:
        nn = NN(name=container_name)
        nn_by_name[container_name] = nn

        # Get all modules (layers + tensor_ops + refs) for this container
        container_layers = layers_by_container[container_id]
        container_tensor_ops = tensor_ops_by_container[container_id]
        container_refs = refs_by_container[container_id]

        # Build set of all module IDs in this container (including refs)
        all_module_ids = (set(container_layers.keys()) |
                         set(container_tensor_ops.keys()) |
                         set(ref_id for ref_id, _ in container_refs))

        # Topologically sort modules based on NNNext within this container
        ordered_modules = topological_sort(all_module_ids, outgoing_connections)

        # Add modules in order
        container_refs_dict = dict(container_refs)
        for module_id in ordered_modules:
            if module_id in container_layers:
                nn.add_layer(container_layers[module_id])
            elif module_id in container_tensor_ops:
                nn.add_tensor_op(container_tensor_ops[module_id])
            elif module_id in container_refs_dict:
                # This is an NNReference - add the referenced NN as sub_nn
                ref_name = container_refs_dict.get(module_id)
                if not ref_name:
                    raise ValueError(
                        f"NNReference in container '{container_name}' has no "
                        f"'referencedNN' field"
                    )
                if ref_name not in nn_by_name:
                    raise ValueError(
                        f"NNReference '{ref_name}' in container "
                        f"'{container_name}' does not match any NNContainer"
                    )
                nn.add_sub_nn(nn_by_name[ref_name])

    # Step 6: Determine the main NN to return.
    # The main NN is the one that is NOT referenced by any NNReference.
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
        # Every container is referenced (mutual reference) — still broken, but
        # surface the issue clearly instead of silently picking one.
        raise ValueError(
            "NN diagram has no top-level NNContainer: every container is "
            "referenced by an NNReference. Remove one of the references."
        )
    else:
        # No containers at all — produce an empty NN so empty diagrams still export.
        main_nn = NN(name=title)

    # Add configuration to the main NN
    if configuration:
        main_nn.add_configuration(configuration)

    # Step 7: Parse datasets (TrainingDataset / TestDataset) and attach to main NN.
    # The metamodel only holds one of each, so collect first and raise on duplicates
    # rather than letting the last one silently overwrite the previous.
    train_datasets = []
    test_datasets = []
    for element_id, element in elements.items():
        element_type = element.get('type', '')
        if element_type == 'TrainingDataset':
            train_datasets.append(element)
        elif element_type == 'TestDataset':
            test_datasets.append(element)

    if len(train_datasets) > 1:
        raise ValueError(
            f"NN diagram contains {len(train_datasets)} TrainingDataset elements; "
            f"only one is supported per NN model"
        )
    if len(test_datasets) > 1:
        raise ValueError(
            f"NN diagram contains {len(test_datasets)} TestDataset elements; "
            f"only one is supported per NN model"
        )
    if train_datasets:
        main_nn.add_train_data(create_dataset(train_datasets[0], elements))
    if test_datasets:
        main_nn.add_test_data(create_dataset(test_datasets[0], elements))

    return main_nn


def topological_sort(module_ids, outgoing_connections):
    """
    Topologically sort modules based on their NNNext connections.
    First layer = the one with no incoming connections from other modules.

    Args:
        module_ids: Set of module IDs
        outgoing_connections: Dict mapping source_id to list of target_ids

    Returns:
        List of module IDs in topological order
    """
    # Kahn's algorithm
    in_degree = {mid: 0 for mid in module_ids}

    # Calculate in-degrees (only from connections within module_ids)
    for source_id, targets in outgoing_connections.items():
        if source_id in module_ids:
            for target_id in targets:
                if target_id in in_degree:
                    in_degree[target_id] += 1

    # Find all nodes with in-degree 0 (first layers - no incoming from other modules)
    # Sort for deterministic ordering when multiple nodes share in-degree 0
    queue = sorted(mid for mid, deg in in_degree.items() if deg == 0)
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        # Decrease in-degree for all neighbors
        for target_id in outgoing_connections.get(current, []):
            if target_id in in_degree:
                in_degree[target_id] -= 1
                if in_degree[target_id] == 0:
                    bisect.insort(queue, target_id)

    # Any nodes still missing from the result have a non-zero in-degree,
    # which means they participate in a cycle (disconnected components
    # always have at least one zero-in-degree node). That is a malformed
    # network, so surface it instead of silently accepting it.
    remaining = [mid for mid in module_ids if mid not in result]
    if remaining:
        raise ValueError(
            "NNNext relationships form a cycle involving "
            f"{len(remaining)} module(s); cannot determine layer order"
        )

    return result


# Layer creation functions

def _create_conv_layer(element, elements, conv_class, default_stride):
    """Create a convolutional layer (Conv1D, Conv2D, or Conv3D) from element data."""
    class_name = conv_class.__name__
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError(f"{class_name} layer missing mandatory 'name' attribute")

    kernel_dim_raw = get_element_attribute(element, 'KernelDimAttribute', elements)
    kernel_dim = parse_list_of_ints(kernel_dim_raw)
    if kernel_dim is None:
        raise ValueError(f"{class_name} layer '{name}' missing mandatory 'kernel_dim' attribute")

    # Validate kernel_dim length before passing to the constructor — otherwise
    # the metamodel setter raises ("kernel_dim list must have exactly N elements")
    # without naming the layer, which surfaces as a cryptic 400 to the user.
    expected_dim = _CONV_EXPECTED_DIMS.get(class_name)
    if expected_dim is not None and len(kernel_dim) != expected_dim:
        raise ValueError(
            f"{class_name} layer '{name}' kernel_dim has {len(kernel_dim)} "
            f"element(s), expected {expected_dim}."
        )

    out_channels_raw = get_element_attribute(element, 'OutChannelsAttribute', elements)
    out_channels = parse_tuple_or_int(out_channels_raw)
    if out_channels is None:
        raise ValueError(f"{class_name} layer '{name}' missing mandatory 'out_channels' attribute")

    layer = conv_class(
        name=sanitize_name(name),
        kernel_dim=kernel_dim,
        out_channels=out_channels,
    )

    # Optional attributes
    stride = get_element_attribute(element, 'StrideDimAttribute', elements)
    if stride is not None:
        stride_dim_parsed = parse_list_of_ints(stride, default_stride)
        if expected_dim is not None and stride_dim_parsed is not None \
                and len(stride_dim_parsed) != expected_dim:
            raise ValueError(
                f"{class_name} layer '{name}' stride_dim has "
                f"{len(stride_dim_parsed)} element(s), expected {expected_dim}."
            )
        layer.stride_dim = stride_dim_parsed

    in_channels = get_element_attribute(element, 'InChannelsAttribute', elements)
    if in_channels is not None:
        layer.in_channels = parse_tuple_or_int(in_channels)

    padding = get_element_attribute(element, 'PaddingAmountAttribute', elements)
    if padding is not None:
        layer.padding_amount = parse_tuple_or_int(padding, 0)

    padding_type = get_element_attribute(element, 'PaddingTypeAttribute', elements)
    if padding_type:
        layer.padding_type = padding_type

    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    permute_in = get_element_attribute(element, 'PermuteInAttribute', elements)
    if permute_in is not None and str(permute_in).strip() != '':
        # permute_in/out are booleans in the metamodel, not integer lists —
        # they toggle dimension permutation in the PyTorch generator.
        layer.permute_in = parse_bool(permute_in)
        mark_explicit(layer, 'permute_in')

    permute_out = get_element_attribute(element, 'PermuteOutAttribute', elements)
    if permute_out is not None and str(permute_out).strip() != '':
        layer.permute_out = parse_bool(permute_out)
        mark_explicit(layer, 'permute_out')

    return layer


def create_conv1d_layer(element, elements):
    """Create a Conv1D layer from element data."""
    return _create_conv_layer(element, elements, Conv1D, [1])


def create_conv2d_layer(element, elements):
    """Create a Conv2D layer from element data."""
    return _create_conv_layer(element, elements, Conv2D, [1, 1])


def create_conv3d_layer(element, elements):
    """Create a Conv3D layer from element data."""
    return _create_conv_layer(element, elements, Conv3D, [1, 1, 1])


def create_pooling_layer(element, elements):
    """Create a PoolingLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("PoolingLayer missing mandatory 'name' attribute")

    pooling_type = get_element_attribute(element, 'PoolingTypeAttribute', elements)
    if not pooling_type:
        raise ValueError(f"PoolingLayer '{name}' missing mandatory 'pooling_type' attribute")
    if pooling_type not in _ALLOWED_POOLING_TYPES:
        raise ValueError(
            f"PoolingLayer '{name}' has invalid pooling_type '{pooling_type}'. "
            f"Allowed values: {', '.join(_ALLOWED_POOLING_TYPES)}."
        )

    dimension_raw = get_element_attribute(element, 'DimensionAttribute', elements)
    dimension = parse_dimension(dimension_raw)
    if dimension is None:
        raise ValueError(f"PoolingLayer '{name}' missing mandatory 'dimension' attribute")

    # kernel_dim is required for non-adaptive/global pooling - provide default based on dimension
    kernel_dim_raw = get_element_attribute(element, 'KernelDimAttribute', elements)
    kernel_dim = parse_list_of_ints(kernel_dim_raw)

    # If kernel_dim not provided and it's not adaptive/global pooling, use default
    if kernel_dim is None and not (pooling_type.startswith("adaptive") or pooling_type.startswith("global")):
        if dimension == "1D":
            kernel_dim = [2]
        elif dimension == "2D":
            kernel_dim = [2, 2]
        elif dimension == "3D":
            kernel_dim = [2, 2, 2]

    stride_raw = get_element_attribute(element, 'StrideDimAttribute', elements)
    stride_dim = parse_list_of_ints(stride_raw)

    layer = PoolingLayer(
        name=sanitize_name(name),
        pooling_type=pooling_type,
        dimension=dimension,
        kernel_dim=kernel_dim,
        stride_dim=stride_dim,
    )

    padding = get_element_attribute(element, 'PaddingAmountAttribute', elements)
    if padding is not None:
        layer.padding_amount = parse_tuple_or_int(padding, 0)

    padding_type = get_element_attribute(element, 'PaddingTypeAttribute', elements)
    if padding_type:
        layer.padding_type = padding_type

    output_dim = get_element_attribute(element, 'OutputDimAttribute', elements)
    if output_dim is not None:
        layer.output_dim = parse_list_of_ints(output_dim)

    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    permute_in = get_element_attribute(element, 'PermuteInAttribute', elements)
    if permute_in is not None and str(permute_in).strip() != '':
        # permute_in/out are booleans in the metamodel, not integer lists.
        layer.permute_in = parse_bool(permute_in)
        mark_explicit(layer, 'permute_in')

    permute_out = get_element_attribute(element, 'PermuteOutAttribute', elements)
    if permute_out is not None and str(permute_out).strip() != '':
        layer.permute_out = parse_bool(permute_out)
        mark_explicit(layer, 'permute_out')

    return layer


def _create_rnn_like_layer(element, elements, rnn_class):
    """Create an RNN-like layer (SimpleRNNLayer, LSTMLayer, or GRULayer) from element data."""
    class_name = rnn_class.__name__
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError(f"{class_name} missing mandatory 'name' attribute")

    hidden_size_raw = get_element_attribute(element, 'HiddenSizeAttribute', elements)
    hidden_size = parse_tuple_or_int(hidden_size_raw)
    if hidden_size is None:
        raise ValueError(f"{class_name} '{name}' missing mandatory 'hidden_size' attribute")

    layer = rnn_class(
        name=sanitize_name(name),
        hidden_size=hidden_size,
    )

    # Optional attributes
    return_type = get_element_attribute(element, 'ReturnTypeAttribute', elements)
    if return_type:
        if return_type not in _ALLOWED_RETURN_TYPES:
            raise ValueError(
                f"{class_name} '{name}' has invalid return_type "
                f"'{return_type}'. Allowed values: "
                f"{', '.join(_ALLOWED_RETURN_TYPES)}."
            )
        layer.return_type = return_type

    input_size = get_element_attribute(element, 'InputSizeAttribute', elements)
    if input_size is not None:
        layer.input_size = parse_tuple_or_int(input_size)

    bidirectional = get_element_attribute(element, 'BidirectionalAttribute', elements)
    if bidirectional is not None:
        layer.bidirectional = parse_bool(bidirectional)
        mark_explicit(layer, 'bidirectional')

    dropout = get_element_attribute(element, 'DropoutAttribute', elements)
    if dropout is not None:
        layer.dropout = parse_float(dropout, 0.0)
        mark_explicit(layer, 'dropout')

    batch_first = get_element_attribute(element, 'BatchFirstAttribute', elements)
    if batch_first is not None:
        layer.batch_first = parse_bool(batch_first)
        mark_explicit(layer, 'batch_first')

    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_rnn_layer(element, elements):
    """Create a SimpleRNNLayer from element data."""
    return _create_rnn_like_layer(element, elements, SimpleRNNLayer)


def create_lstm_layer(element, elements):
    """Create an LSTMLayer from element data."""
    return _create_rnn_like_layer(element, elements, LSTMLayer)


def create_gru_layer(element, elements):
    """Create a GRULayer from element data."""
    return _create_rnn_like_layer(element, elements, GRULayer)


def create_linear_layer(element, elements):
    """Create a LinearLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("LinearLayer missing mandatory 'name' attribute")

    out_features_raw = get_element_attribute(element, 'OutFeaturesAttribute', elements)
    out_features = parse_tuple_or_int(out_features_raw)
    if out_features is None:
        raise ValueError(f"LinearLayer '{name}' missing mandatory 'out_features' attribute")

    layer = LinearLayer(
        name=sanitize_name(name),
        out_features=out_features,
    )

    # Optional attributes
    in_features = get_element_attribute(element, 'InFeaturesAttribute', elements)
    if in_features is not None:
        layer.in_features = parse_tuple_or_int(in_features)

    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_flatten_layer(element, elements):
    """Create a FlattenLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("FlattenLayer missing mandatory 'name' attribute")

    layer = FlattenLayer(name=sanitize_name(name))

    # Optional attributes
    start_dim = get_element_attribute(element, 'StartDimAttribute', elements)
    if start_dim is not None:
        layer.start_dim = parse_tuple_or_int(start_dim)

    end_dim = get_element_attribute(element, 'EndDimAttribute', elements)
    if end_dim is not None:
        layer.end_dim = parse_tuple_or_int(end_dim)

    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_embedding_layer(element, elements):
    """Create an EmbeddingLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("EmbeddingLayer missing mandatory 'name' attribute")

    num_embeddings_raw = get_element_attribute(element, 'NumEmbeddingsAttribute', elements)
    num_embeddings = parse_tuple_or_int(num_embeddings_raw)
    if num_embeddings is None:
        raise ValueError(f"EmbeddingLayer '{name}' missing mandatory 'num_embeddings' attribute")

    embedding_dim_raw = get_element_attribute(element, 'EmbeddingDimAttribute', elements)
    embedding_dim = parse_tuple_or_int(embedding_dim_raw)
    if embedding_dim is None:
        raise ValueError(f"EmbeddingLayer '{name}' missing mandatory 'embedding_dim' attribute")

    layer = EmbeddingLayer(
        name=sanitize_name(name),
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )

    # Optional attributes
    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_dropout_layer(element, elements):
    """Create a DropoutLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("DropoutLayer missing mandatory 'name' attribute")

    rate_raw = get_element_attribute(element, 'RateAttribute', elements)
    rate = parse_float(rate_raw)
    if rate is None:
        raise ValueError(f"DropoutLayer '{name}' missing mandatory 'rate' attribute")

    layer = DropoutLayer(
        name=sanitize_name(name),
        rate=rate,
    )

    # Optional attributes
    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_layer_norm_layer(element, elements):
    """Create a LayerNormLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("LayerNormLayer missing mandatory 'name' attribute")

    normalized_shape_raw = get_element_attribute(element, 'NormalizedShapeAttribute', elements)
    normalized_shape = parse_tuple_or_int(normalized_shape_raw)
    if normalized_shape is None:
        raise ValueError(f"LayerNormLayer '{name}' missing mandatory 'normalized_shape' attribute")

    layer = LayerNormLayer(
        name=sanitize_name(name),
        normalized_shape=normalized_shape,
    )

    # Optional attributes
    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_batch_norm_layer(element, elements):
    """Create a BatchNormLayer from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("BatchNormLayer missing mandatory 'name' attribute")

    num_features_raw = get_element_attribute(element, 'NumFeaturesAttribute', elements)
    num_features = parse_tuple_or_int(num_features_raw)
    if num_features is None:
        raise ValueError(f"BatchNormLayer '{name}' missing mandatory 'num_features' attribute")

    dimension_raw = get_element_attribute(element, 'DimensionAttribute', elements)
    dimension = parse_dimension(dimension_raw)
    if dimension is None:
        raise ValueError(f"BatchNormLayer '{name}' missing mandatory 'dimension' attribute")

    layer = BatchNormLayer(
        name=sanitize_name(name),
        num_features=num_features,
        dimension=dimension,
    )

    # Optional attributes
    actv_func = get_element_attribute(element, 'ActvFuncAttribute', elements)
    if actv_func:
        layer.actv_func = actv_func

    name_module_input = get_element_attribute(element, 'NameModuleInputAttribute', elements)
    if name_module_input:
        layer.name_module_input = name_module_input

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        layer.input_reused = parse_bool(input_reused)
        mark_explicit(layer, 'input_reused')

    return layer


def create_tensor_op(element, elements):
    """Create a TensorOp from element data."""
    name = get_element_attribute(element, 'NameAttribute', elements)
    if not name:
        raise ValueError("TensorOp missing mandatory 'name' attribute")

    tns_type = get_element_attribute(element, 'TnsTypeAttribute', elements)
    if not tns_type:
        raise ValueError(f"TensorOp '{name}' missing mandatory 'tns_type' attribute")

    # Parse all attributes first before creating TensorOp
    # (because TensorOp validates required attributes in tns_type setter)
    concatenate_dim = get_element_attribute(element, 'ConcatenateDimAttribute', elements)
    if concatenate_dim is not None:
        concatenate_dim = parse_tuple_or_int(concatenate_dim)

    layers_of_tensors_raw = get_element_attribute(element, 'LayersOfTensorsAttribute', elements)
    layers_of_tensors = None
    if layers_of_tensors_raw:
        if isinstance(layers_of_tensors_raw, str):
            # Handle "['layer1', 'layer2']" or "[layer1, layer2]" or "layer1, layer2" format
            val = layers_of_tensors_raw.strip()
            if val.startswith('[') and val.endswith(']'):
                val = val[1:-1]
            # Split by comma and strip whitespace and quotes from each layer name
            layers_of_tensors = [layer.strip().strip("'\"") for layer in val.split(',') if layer.strip()]
        elif isinstance(layers_of_tensors_raw, list):
            # Also strip quotes from list items
            layers_of_tensors = [layer.strip("'\"") if isinstance(layer, str) else layer for layer in layers_of_tensors_raw]

    reshape_dim = get_element_attribute(element, 'ReshapeDimAttribute', elements)
    if reshape_dim is not None:
        reshape_dim = parse_list_of_ints(reshape_dim)

    transpose_dim = get_element_attribute(element, 'TransposeDimAttribute', elements)
    if transpose_dim is not None:
        transpose_dim = parse_list_of_ints(transpose_dim)

    permute_dim = get_element_attribute(element, 'PermuteDimAttribute', elements)
    if permute_dim is not None:
        permute_dim = parse_list_of_ints(permute_dim)

    # Validate required attributes based on tns_type
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

    # Create TensorOp with only relevant attributes based on tns_type
    # This prevents unnecessary attributes from being stored in the object
    tensor_op_params = {
        'name': sanitize_name(name),
        'tns_type': tns_type,
    }

    # Only include attributes relevant to the specific tns_type
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

    input_reused = get_element_attribute(element, 'InputReusedAttribute', elements)
    if input_reused is not None:
        tensor_op.input_reused = parse_bool(input_reused)
        mark_explicit(tensor_op, 'input_reused')

    return tensor_op


def create_configuration(element, elements):
    """Create a Configuration from element data."""
    batch_size_raw = get_element_attribute(element, 'BatchSizeAttribute', elements)
    batch_size = parse_tuple_or_int(batch_size_raw)
    if batch_size is None:
        raise ValueError("Configuration missing mandatory 'batch_size' attribute")

    epochs_raw = get_element_attribute(element, 'EpochsAttribute', elements)
    epochs = parse_tuple_or_int(epochs_raw)
    if epochs is None:
        raise ValueError("Configuration missing mandatory 'epochs' attribute")

    learning_rate_raw = get_element_attribute(element, 'LearningRateAttribute', elements)
    learning_rate = parse_float(learning_rate_raw)
    if learning_rate is None:
        raise ValueError("Configuration missing mandatory 'learning_rate' attribute")

    optimizer = get_element_attribute(element, 'OptimizerAttribute', elements)
    if not optimizer:
        raise ValueError("Configuration missing mandatory 'optimizer' attribute")
    if optimizer not in _ALLOWED_OPTIMIZERS:
        raise ValueError(
            f"Configuration has invalid optimizer '{optimizer}'. "
            f"Allowed values: {', '.join(_ALLOWED_OPTIMIZERS)}."
        )

    loss_function = get_element_attribute(element, 'LossFunctionAttribute', elements)
    if not loss_function:
        raise ValueError("Configuration missing mandatory 'loss_function' attribute")
    if loss_function not in _ALLOWED_LOSS_FUNCTIONS:
        raise ValueError(
            f"Configuration has invalid loss_function '{loss_function}'. "
            f"Allowed values: {', '.join(_ALLOWED_LOSS_FUNCTIONS)}."
        )

    metrics_str = get_element_attribute(element, 'MetricsAttribute', elements)
    if not metrics_str:
        raise ValueError("Configuration missing mandatory 'metrics' attribute")

    # Parse metrics (could be comma-separated or list notation)
    if isinstance(metrics_str, str):
        # Handle "[accuracy]" format
        if metrics_str.startswith('[') and metrics_str.endswith(']'):
            inner = metrics_str[1:-1].strip()
            metrics = [m.strip() for m in inner.split(',') if m.strip()]
        else:
            metrics = [m.strip() for m in metrics_str.split(',') if m.strip()]
    elif isinstance(metrics_str, list):
        metrics = metrics_str
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

    # Optional attributes
    weight_decay = get_element_attribute(element, 'WeightDecayAttribute', elements)
    if weight_decay is not None:
        config.weight_decay = parse_float(weight_decay, 0.0)
        mark_explicit(config, 'weight_decay')

    momentum = get_element_attribute(element, 'MomentumAttribute', elements)
    if momentum is not None:
        config.momentum = parse_float(momentum, 0.0)
        mark_explicit(config, 'momentum')

    return config
