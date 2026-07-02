"""BPMN processing for converting JSON to BUML format.

Reads the v4 wire shape (``{nodes, edges}``) natively — mirrors
``json_to_buml/class_diagram_processor.py``. Ported from the v3 (elements/
relationships) processor of the same name; the only changes are the JSON
access layer (flat ``nodes``/``edges`` lists instead of ``elements``/
``relationships`` dicts, ``node["parentId"]`` instead of ``element["owner"]``,
and one edge ``type`` per BPMN flow kind instead of a single ``"BPMNFlow"``
type with a ``flowType`` discriminator — see
``packages/library/lib/utils/versionConverter.ts``'s ``convertV3EdgeTypeToV4``).
The BUML BPMN metamodel algorithm (containment via the owner/parentId chain,
flow routing via ``process_of``) is unchanged.

``BPMNElement.layout`` / ``BPMNConnectingObject.layout`` are declared opaque
by the metamodel (type-checked as dict-or-None, content never interpreted),
so this port is free to redefine the internal stash shape for v4 — see
``_layout_dict`` / ``_flow_layout_dict``.
"""

import logging

from besser.BUML.metamodel.bpmn import (
    Artifact,
    Association,
    BPMNModel,
    CallActivity,
    Collaboration,
    DataAssociation,
    DataObject,
    DataStore,
    EndEvent,
    FlowNode,
    Gateway,
    GatewayType,
    Group,
    IntermediateEvent,
    Lane,
    LoopCharacteristics,
    MessageFlow,
    Participant,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TaskType,
    TextAnnotation,
    Transaction,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    BPMN_DIAGRAM_TYPES,
)
from besser.utilities.web_modeling_editor.backend.services.converters.bpmn_event_mapping import (
    parse_event_type,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data, node_bounds,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)


# v4 node `type` (lowerCamelCase, packages/library/lib/nodes/types.ts) -> metamodel class.
_EVENT_CLASSES = {
    "bpmnStartEvent": StartEvent,
    "bpmnIntermediateEvent": IntermediateEvent,
    "bpmnEndEvent": EndEvent,
}

_ACTIVITY_CLASSES = {
    "bpmnTask": Task,
    "bpmnSubprocess": SubProcess,
    "bpmnTransaction": Transaction,
    "bpmnCallActivity": CallActivity,
}

# v4 edge `type` -> BUML connecting-object class (versionConverter.ts convertV3EdgeTypeToV4).
_FLOW_CLASS_FOR_EDGE_TYPE = {
    "BPMNSequenceFlow": SequenceFlow,
    "BPMNMessageFlow": MessageFlow,
    "BPMNAssociationFlow": Association,
    "BPMNDataAssociationFlow": DataAssociation,
}


def _layout_dict(node: dict) -> dict:
    """Opaque layout passthrough for a node: id + parentId + bounds + style keys."""
    layout = {
        "id": node.get("id"),
        "parentId": node.get("parentId"),
        "bounds": node_bounds(node),
    }
    data = node_data(node)
    for style_key in ("fillColor", "strokeColor", "textColor"):
        if style_key in data:
            layout[style_key] = data[style_key]
    return layout


def _flow_layout_dict(edge: dict) -> dict:
    """Opaque layout passthrough for an edge (flow)."""
    edge_data = edge.get("data") or {}
    return {
        "id": edge.get("id"),
        "points": edge_data.get("points") or [],
        "source_direction": edge.get("sourceHandle"),
        "target_direction": edge.get("targetHandle"),
        "isManuallyLayouted": edge_data.get("isManuallyLayouted", False),
    }


def _build_node(node: dict):
    """Construct the metamodel object for one v4 ``nodes[]`` entry, or ``None`` for
    an unknown type. Raises ConversionError for a known type with an unrecognised
    enum string."""
    node_type = node.get("type")
    data = node_data(node)
    name = data.get("name", "") or ""

    if node_type in _ACTIVITY_CLASSES:
        cls = _ACTIVITY_CLASSES[node_type]
        try:
            loop = LoopCharacteristics(data.get("marker", "none") or "none")
        except ValueError as exc:
            raise ConversionError(
                f"Unknown BPMN loop marker '{data.get('marker')}' on element '{name}'."
            ) from exc
        if cls is Task:
            try:
                task_type = TaskType(data.get("taskType", "default") or "default")
            except ValueError as exc:
                raise ConversionError(
                    f"Unknown BPMN task type '{data.get('taskType')}' on Task '{name}'."
                ) from exc
            return Task(name=name, task_type=task_type, loop_characteristics=loop)
        return cls(name=name, loop_characteristics=loop)

    if node_type in _EVENT_CLASSES:
        cls = _EVENT_CLASSES[node_type]
        event_type = data.get("eventType", "default") or "default"
        try:
            direction, definition = parse_event_type(cls, event_type)
        except ValueError as exc:
            raise ConversionError(
                f"Unknown BPMN event type '{event_type}' on {cls.__name__} '{name}': {exc}"
            ) from exc
        return cls(name=name, direction=direction, event_definition=definition)

    if node_type == "bpmnGateway":
        try:
            gateway_type = GatewayType(data.get("gatewayType", "exclusive") or "exclusive")
        except ValueError as exc:
            raise ConversionError(
                f"Unknown BPMN gateway type '{data.get('gatewayType')}' on Gateway '{name}'."
            ) from exc
        return Gateway(name=name, gateway_type=gateway_type)

    if node_type == "bpmnDataObject":
        return DataObject(name=name)
    if node_type == "bpmnDataStore":
        return DataStore(name=name)
    if node_type == "bpmnAnnotation":
        return TextAnnotation(name="", text=name)
    if node_type == "bpmnGroup":
        return Group(name=name)
    if node_type == "bpmnSwimlane":
        # Forward-compat only: no bpmnSwimlane entry exists yet in
        # packages/library/lib/nodes/types.ts / lib/constants.ts palette /
        # bpmnConstraints.ts drop-rules. Unreachable from the running app
        # until the frontend adds it; kept so no second backend PR is needed then.
        return Lane(name=name)
    if node_type == "bpmnPool":
        return Participant(name=name, process=Process(name=name))

    return None


def _build_flow(edge: dict, source, target):
    """Construct the metamodel flow object for one v4 ``edges[]`` entry."""
    edge_data = edge.get("data") or {}
    # BPMNDiagramEdge.tsx renders data.label; name kept for parity with every
    # other v4 edge type's authoring convention (class associations, StateTransition...).
    name = edge_data.get("name") or edge_data.get("label") or ""

    flow_cls = _FLOW_CLASS_FOR_EDGE_TYPE.get(edge.get("type"))
    if flow_cls is None:
        raise ConversionError(f"Unknown BPMN edge type '{edge.get('type')}' on relationship '{name}'.")

    if flow_cls is SequenceFlow:
        flow = SequenceFlow(source=source, target=target, name=name)
        if edge_data.get("isDefault"):
            try:
                flow.is_default = True
            except ValueError as exc:
                logger.warning(
                    "SequenceFlow '%s' marked default but its source cannot carry one (%s); "
                    "downgrading to is_default=False.", name, exc,
                )
        return flow
    return flow_cls(source=source, target=target, name=name)


def _outer_process(container):
    while isinstance(container, SubProcess):
        container = container.container
    return container if isinstance(container, Process) else None


def process_bpmn_diagram(json_data: dict) -> BPMNModel:
    """Convert a v4 BPMN diagram (``{nodes, edges}``) into a ``BPMNModel``.
    v3-shape input is not supported — mirrors process_class_diagram's contract."""
    title = json_data.get('title') or 'Generated_BPMN_Model'
    model_payload = json_data.get('model') or {}
    model_type = model_payload.get('type')
    if model_type and model_type not in BPMN_DIAGRAM_TYPES:
        logger.warning(
            "BPMN diagram envelope type is '%s', expected one of %s.",
            model_type, BPMN_DIAGRAM_TYPES,
        )

    nodes = model_payload.get('nodes') or []
    edges = model_payload.get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    # --- Pass 1: build node objects (no containment yet) -------------------
    raw_by_id: dict = {n.get("id"): n for n in nodes if n.get("id")}
    obj_by_id: dict = {}
    pools: list = []  # (node_id, Participant) — preserves JSON order

    for node in nodes:
        node_id = node.get("id")
        try:
            obj = _build_node(node)
        except ConversionError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConversionError(f"Could not build BPMN element '{node_id}': {exc}") from exc
        if obj is None:
            logger.warning("BPMN node '%s' has unknown type '%s'; skipping.", node_id, node.get("type"))
            continue
        obj.layout = _layout_dict(node)
        obj_by_id[node_id] = obj
        if isinstance(obj, Participant):
            pools.append((node_id, obj))

    # --- Pass 2: containment via `parentId` ---------------------------------
    process_of: dict = {}
    synthetic_process = None

    def _ensure_synthetic_process():
        nonlocal synthetic_process
        if synthetic_process is None:
            synthetic_process = Process(name=title)
        return synthetic_process

    for node_id, obj in obj_by_id.items():
        if not isinstance(obj, Lane):
            continue
        parent_id = (raw_by_id.get(node_id) or {}).get("parentId")
        parent_obj = obj_by_id.get(parent_id) if parent_id else None
        if isinstance(parent_obj, Participant) and parent_obj.process is not None:
            parent_obj.process.add_lane(obj)
        else:
            logger.warning("BPMN Lane '%s' has no Pool parent; orphan lane skipped.", node_id)

    for node_id, obj in obj_by_id.items():
        if isinstance(obj, (Participant, Lane, DataStore)):
            continue

        parent_id = (raw_by_id.get(node_id) or {}).get("parentId")
        parent_obj = obj_by_id.get(parent_id) if parent_id else None

        target_container = None
        target_lane = None

        if parent_obj is None:
            target_container = _ensure_synthetic_process()
        elif isinstance(parent_obj, Lane):
            target_lane = parent_obj
            for _, part in pools:
                if part.process is not None and target_lane in part.process.lanes:
                    target_container = part.process
                    break
            if target_container is None:
                target_container = _ensure_synthetic_process()
        elif isinstance(parent_obj, Participant):
            target_container = parent_obj.process or _ensure_synthetic_process()
        elif isinstance(parent_obj, SubProcess):
            target_container = parent_obj
        else:
            logger.warning(
                "BPMN element '%s' has unexpected parent type '%s'; placing at top level.",
                node_id, type(parent_obj).__name__,
            )
            target_container = _ensure_synthetic_process()

        if isinstance(obj, FlowNode):
            target_container.add_flow_node(obj)
            if target_lane is not None:
                target_lane.add_flow_node(obj)
            process_of[obj] = target_container
        elif isinstance(obj, Artifact):
            outer = _outer_process(target_container) or _ensure_synthetic_process()
            outer.add_artifact(obj)
            process_of[obj] = outer
        elif isinstance(obj, DataObject):
            outer = _outer_process(target_container) or _ensure_synthetic_process()
            outer.add_data_object(obj)
            process_of[obj] = outer

    data_stores = {obj for obj in obj_by_id.values() if isinstance(obj, DataStore)}

    if pools:
        collaboration = Collaboration(name=title, participants={part for _, part in pools})
        processes = {part.process for _, part in pools if part.process is not None}
    else:
        collaboration = None
        processes = set()
    if synthetic_process is not None:
        processes.add(synthetic_process)

    # --- Pass 3: build flows from `edges` -----------------------------------
    for edge in edges:
        edge_id = edge.get("id")
        edge_type = edge.get("type")
        if edge_type not in _FLOW_CLASS_FOR_EDGE_TYPE:
            continue  # not a BPMN flow edge (e.g. a stray comment link) — skip quietly

        source = obj_by_id.get(edge.get("source"))
        target = obj_by_id.get(edge.get("target"))
        if source is None or target is None:
            logger.warning(
                "BPMN flow '%s' has a dangling endpoint (source=%s, target=%s); skipping.",
                edge_id, edge.get("source"), edge.get("target"),
            )
            continue

        try:
            flow = _build_flow(edge, source, target)
        except ConversionError:
            raise
        except (TypeError, ValueError) as exc:
            logger.warning("BPMN flow '%s' could not be built (%s); skipping.", edge_id, exc)
            continue

        flow.layout = _flow_layout_dict(edge)

        if isinstance(flow, MessageFlow):
            if collaboration is not None:
                collaboration.add_message_flow(flow)
            else:
                logger.warning(
                    "BPMN MessageFlow '%s' present but the diagram has no pools; skipping.", edge_id,
                )
            continue

        endpoint_container = process_of.get(source) or process_of.get(target)
        if endpoint_container is None:
            logger.warning("BPMN flow '%s' has no resolvable container; skipping.", edge_id)
            continue

        if isinstance(flow, SequenceFlow):
            if isinstance(endpoint_container, (Process, SubProcess)):
                endpoint_container.add_sequence_flow(flow)
        elif isinstance(flow, Association):
            outer = _outer_process(endpoint_container)
            if outer is not None:
                outer.add_association(flow)
        elif isinstance(flow, DataAssociation):
            outer = _outer_process(endpoint_container)
            if outer is not None:
                outer.add_data_association(flow)

    return BPMNModel(name=title, processes=processes, collaboration=collaboration, data_stores=data_stores)
