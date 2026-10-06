"""BPMN conversion from BUML to JSON format.

Emits the v4 wire shape (``{nodes, edges}``) directly — mirrors
``buml_to_json/class_diagram_converter.py`` (``make_node``/``make_edge`` from
``_node_builders``, no v3-shape intermediate). Ported from the v3 converter of
the same name; the metamodel walk (pools -> lanes -> process contents -> data
stores -> flows) is unchanged, only the emitted shape differs (flat node/edge
lists instead of ``elements``/``relationships`` dicts, ``parentId`` instead of
``owner``, one edge ``type`` per flow kind instead of ``"BPMNFlow"`` +
``flowType``).

Two entry points:

* ``bpmn_object_to_json(model: BPMNModel) -> dict`` — converts a metamodel object
  directly. Mirror of ``json_to_buml.bpmn_diagram_processor.process_bpmn_diagram``.
* ``bpmn_buml_to_json(content: str) -> dict`` — loads a BPMN BUML ``.py`` source string
  through the safe AST-allowlist loader, finds the resulting ``BPMNModel``, and delegates to
  ``bpmn_object_to_json``. The ``.py`` files emitted by
  ``besser.utilities.buml_code_builder.bpmn_model_builder.bpmn_model_to_code`` are
  exactly what this wrapper expects.

Design points (mirror of the processor):

* **Stable round-trip ids.** ``id_for(obj)`` reuses the original WME id stashed in
  ``obj.layout["id"]`` when present, falling back to a fresh uuid. The metamodel stays
  id-free; ``layout`` is the per-element side-channel.
* **Layout fallback.** When ``layout`` is missing / partial (a freshly-built model, or
  one loaded from BUML code that didn't emit layout), a deterministic grid layout fills
  the gaps.
* **Deterministic walk.** Set members are walked in ``timestamp`` order via
  ``besser.utilities.sort_by_timestamp`` so the export is reproducible (BPMN names are
  not unique — sorting by name is not an option).
"""

import logging
import uuid

from besser.BUML.metamodel.bpmn import (
    Activity,
    Association,
    BPMNConnectingObject,
    BPMNModel,
    CallActivity,
    Collaboration,
    DataAssociation,
    DataObject,
    DataStore,
    EndEvent,
    Event,
    EventDefinitionType,
    EventDirection,
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
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.constants.constants import BPMN_DIAGRAM_TYPE
from besser.utilities.web_modeling_editor.backend.services.converters.bpmn_event_mapping import (
    serialise_event_type,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
    make_node, make_edge,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)


_TYPE_FOR_CLASS = {
    Task: "bpmnTask",
    Transaction: "bpmnTransaction",
    SubProcess: "bpmnSubprocess",
    StartEvent: "bpmnStartEvent",
    IntermediateEvent: "bpmnIntermediateEvent",
    EndEvent: "bpmnEndEvent",
    Gateway: "bpmnGateway",
    DataObject: "bpmnDataObject",
    DataStore: "bpmnDataStore",
    TextAnnotation: "bpmnAnnotation",
    Group: "bpmnGroup",
    Lane: "bpmnSwimlane",
    Participant: "bpmnPool",
}

_DEFAULT_SIZE = {
    "bpmnTask": {"width": 110, "height": 60},
    "bpmnSubprocess": {"width": 250, "height": 150},
    "bpmnTransaction": {"width": 250, "height": 150},
    "bpmnCallActivity": {"width": 110, "height": 60},
    "bpmnStartEvent": {"width": 40, "height": 40},
    "bpmnIntermediateEvent": {"width": 40, "height": 40},
    "bpmnEndEvent": {"width": 40, "height": 40},
    "bpmnGateway": {"width": 50, "height": 50},
    "bpmnDataObject": {"width": 40, "height": 50},
    "bpmnDataStore": {"width": 60, "height": 50},
    "bpmnAnnotation": {"width": 200, "height": 80},
    "bpmnGroup": {"width": 240, "height": 200},
    "bpmnSwimlane": {"width": 800, "height": 120},
    "bpmnPool": {"width": 800, "height": 240},
}

_EDGE_TYPE_FOR_FLOW_CLASS = {
    SequenceFlow: "BPMNSequenceFlow",
    MessageFlow: "BPMNMessageFlow",
    Association: "BPMNAssociationFlow",
    DataAssociation: "BPMNDataAssociationFlow",
}


def _wme_type_for(obj) -> str:
    cls = type(obj)
    if cls.__name__ == "CallActivity":
        return "bpmnCallActivity"
    return _TYPE_FOR_CLASS.get(cls, "")


class _GridLayout:
    """Deterministic per-class grid placement when layout['bounds'] is absent.
    Ported unchanged from v3 — pure geometry, no wire-shape dependency."""

    _ROW_FOR_TYPE = {
        "bpmnPool": 0, "bpmnSwimlane": 1,
        "bpmnStartEvent": 2, "bpmnTask": 2, "bpmnGateway": 2,
        "bpmnCallActivity": 2, "bpmnIntermediateEvent": 2, "bpmnEndEvent": 2,
        "bpmnSubprocess": 3, "bpmnTransaction": 3,
        "bpmnDataObject": 4, "bpmnDataStore": 4,
        "bpmnAnnotation": 5, "bpmnGroup": 5,
    }

    def __init__(self):
        self._x_for_row: dict = {}

    def next_bounds(self, wme_type: str) -> dict:
        size = _DEFAULT_SIZE.get(wme_type, {"width": 100, "height": 60})
        row = self._ROW_FOR_TYPE.get(wme_type, 6)
        x = self._x_for_row.get(row, 40)
        y = 40 + row * 180
        self._x_for_row[row] = x + size["width"] + 40
        return {"x": x, "y": y, "width": size["width"], "height": size["height"]}


def bpmn_object_to_json(model: BPMNModel) -> dict:
    """Convert a ``BPMNModel`` into a v4 WME BPMN diagram JSON dict
    (``version``, ``type``, ``title``, ``size``, ``nodes``, ``edges``,
    ``interactive``, ``assessments``)."""
    nodes: list = []
    edges: list = []
    id_map: dict = {}
    grid = _GridLayout()

    def id_for(obj) -> str:
        if obj not in id_map:
            stashed = (obj.layout or {}).get("id") if hasattr(obj, "layout") else None
            id_map[obj] = stashed or str(uuid.uuid4())
        return id_map[obj]

    if model.collaboration is not None:
        for participant in sort_by_timestamp(model.collaboration.participants):
            _emit_pool(participant, nodes, id_for, grid)

    pooled_processes = (
        {p.process for p in model.collaboration.participants if p.process is not None}
        if model.collaboration is not None else set()
    )
    for process in sort_by_timestamp(model.processes):
        if process in pooled_processes:
            continue
        _emit_process_contents(process, parent_id=None, nodes=nodes, id_for=id_for, grid=grid)

    if model.collaboration is not None:
        for participant in sort_by_timestamp(model.collaboration.participants):
            if participant.process is None:
                continue
            _emit_process_contents(
                participant.process, parent_id=id_for(participant),
                nodes=nodes, id_for=id_for, grid=grid,
            )

    for data_store in sort_by_timestamp(model.data_stores):
        node = _emit_node(data_store, parent_id=None, id_for=id_for, grid=grid)
        if node:
            nodes.append(node)

    _make_positions_parent_relative(nodes, id_map)

    emitted_ids = {node["id"] for node in nodes}
    for process in sort_by_timestamp(model.processes):
        for flow in sort_by_timestamp(process.sequence_flows):
            _emit_flow(flow, edges, id_for, emitted_ids)
        for flow in sort_by_timestamp(process.associations):
            _emit_flow(flow, edges, id_for, emitted_ids)
        for flow in sort_by_timestamp(process.data_associations):
            _emit_flow(flow, edges, id_for, emitted_ids)

    # Sequence flows nested in SubProcesses
    for sub in _walk_subprocesses(model):
        for flow in sort_by_timestamp(sub.sequence_flows):
            _emit_flow(flow, edges, id_for, emitted_ids)

    if model.collaboration is not None:
        for flow in sort_by_timestamp(model.collaboration.message_flows):
            _emit_flow(flow, edges, id_for, emitted_ids)

    return {
        "version": "4.0.0",
        "type": BPMN_DIAGRAM_TYPE,
        "title": getattr(model, "name", "") or "",
        "size": _compute_envelope_size(nodes),
        "nodes": nodes,
        "edges": edges,
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }


def _make_positions_parent_relative(nodes: list, id_map: dict) -> None:
    """``layout["bounds"]`` are absolute; a v4 ``position`` is relative to its
    ``parentId`` node. Only applied when both bounds came from a layout (grid
    fallback positions are already relative)."""
    obj_for_id = {node_id: obj for obj, node_id in id_map.items()}

    def has_bounds(node_id) -> bool:
        obj = obj_for_id.get(node_id)
        return bool((getattr(obj, "layout", None) or {}).get("bounds"))

    absolute = {node["id"]: dict(node["position"]) for node in nodes}
    for node in nodes:
        parent_id = node.get("parentId")
        if parent_id in absolute and has_bounds(node["id"]) and has_bounds(parent_id):
            node["position"] = {
                "x": absolute[node["id"]]["x"] - absolute[parent_id]["x"],
                "y": absolute[node["id"]]["y"] - absolute[parent_id]["y"],
            }


def _emit_pool(participant: Participant, nodes: list, id_for, grid: "_GridLayout") -> None:
    pool_node = _emit_node(participant, parent_id=None, id_for=id_for, grid=grid)
    if pool_node:
        nodes.append(pool_node)
    if participant.process is None:
        return
    pool_id = id_for(participant)
    for lane in sort_by_timestamp(participant.process.lanes):
        lane_node = _emit_node(lane, parent_id=pool_id, id_for=id_for, grid=grid)
        if lane_node:
            nodes.append(lane_node)


def _emit_process_contents(process: Process, parent_id, nodes: list, id_for, grid: "_GridLayout") -> None:
    for node in sort_by_timestamp(process.flow_nodes):
        node_parent_id = id_for(node.lane) if node.lane is not None else parent_id
        emitted = _emit_node(node, parent_id=node_parent_id, id_for=id_for, grid=grid)
        if emitted:
            nodes.append(emitted)
        if isinstance(node, SubProcess):
            _emit_subprocess_children(node, nodes, id_for, grid)

    for artifact in sort_by_timestamp(process.artifacts):
        emitted = _emit_node(artifact, parent_id=parent_id, id_for=id_for, grid=grid)
        if emitted:
            nodes.append(emitted)
    for data_object in sort_by_timestamp(process.data_objects):
        emitted = _emit_node(data_object, parent_id=parent_id, id_for=id_for, grid=grid)
        if emitted:
            nodes.append(emitted)


def _emit_subprocess_children(sub: SubProcess, nodes: list, id_for, grid: "_GridLayout") -> None:
    sub_id = id_for(sub)
    for node in sort_by_timestamp(sub.flow_nodes):
        emitted = _emit_node(node, parent_id=sub_id, id_for=id_for, grid=grid)
        if emitted:
            nodes.append(emitted)
        if isinstance(node, SubProcess):
            _emit_subprocess_children(node, nodes, id_for, grid)


def _emit_node(obj, parent_id, id_for, grid: "_GridLayout"):
    wme_type = _wme_type_for(obj)
    if not wme_type:
        logger.warning("BPMN export: unknown metamodel class %s; skipping.", type(obj).__name__)
        return None

    layout = obj.layout or {}
    bounds = layout.get("bounds") or grid.next_bounds(wme_type)

    name = obj.text if isinstance(obj, TextAnnotation) else obj.name

    data: dict = {"name": name}
    for style_key in ("fillColor", "strokeColor", "textColor", "highlight"):
        if style_key in layout:
            data[style_key] = layout[style_key]

    if isinstance(obj, Task):
        data["taskType"] = obj.task_type.value
    if isinstance(obj, Activity):
        data["marker"] = obj.loop_characteristics.value
    if isinstance(obj, Event):
        data["eventType"] = serialise_event_type(obj)
    if isinstance(obj, Gateway):
        data["gatewayType"] = obj.gateway_type.value

    return make_node(
        node_id=id_for(obj), type_=wme_type, data=data,
        position={"x": bounds["x"], "y": bounds["y"]},
        parent_id=parent_id, width=bounds["width"], height=bounds["height"],
    )


def _points_from_path(layout: dict) -> list:
    """Absolute v4 points from a v3 flow layout (``path`` relative to ``bounds``),
    as stored by ``.py`` files exported before the v4 migration."""
    path = layout.get("path") or []
    bounds = layout.get("bounds") or {}
    return [
        {"x": p.get("x", 0) + bounds.get("x", 0), "y": p.get("y", 0) + bounds.get("y", 0)}
        for p in path
    ]


def _emit_flow(flow: "BPMNConnectingObject", edges: list, id_for, emitted_ids: set) -> None:
    edge_type = _EDGE_TYPE_FOR_FLOW_CLASS.get(type(flow))
    if edge_type is None:
        logger.warning("BPMN export: unknown connecting object %s; skipping.", type(flow).__name__)
        return

    source_id = id_for(flow.source)
    target_id = id_for(flow.target)
    if source_id not in emitted_ids or target_id not in emitted_ids:
        # Endpoint wasn't emitted (defensive — happens only on a malformed model).
        logger.warning(
            "BPMN flow '%s' references an endpoint that was not emitted; skipping.", flow.name,
        )
        return

    layout = flow.layout or {}
    edge_data: dict = {
        "name": flow.name,
        "label": flow.name,  # BPMNDiagramEdge.tsx reads data.label
        "points": layout.get("points") or _points_from_path(layout),
        "isManuallyLayouted": layout.get("isManuallyLayouted", False),
    }
    if isinstance(flow, SequenceFlow) and flow.is_default:
        edge_data["isDefault"] = True

    edges.append(make_edge(
        edge_id=id_for(flow), source=source_id, target=target_id,
        type_=edge_type, data=edge_data,
        source_handle=layout.get("source_direction") or "right",
        target_handle=layout.get("target_direction") or "left",
    ))


def _walk_subprocesses(model: BPMNModel):
    """Yield every ``SubProcess`` (including nested ones) in the model."""
    def _recurse(container):
        for node in container.flow_nodes:
            if isinstance(node, SubProcess):
                yield node
                yield from _recurse(node)
    for process in model.processes:
        yield from _recurse(process)


def _compute_envelope_size(nodes: list) -> dict:
    """Bounding box of all top-level node bounds, with a sensible minimum."""
    max_x = 0
    max_y = 0
    for node in nodes:
        if node.get("parentId"):
            continue  # child positions are relative to their parent
        position = node.get("position") or {}
        right = (position.get("x") or 0) + (node.get("width") or 0)
        bottom = (position.get("y") or 0) + (node.get("height") or 0)
        max_x = max(max_x, right)
        max_y = max(max_y, bottom)
    return {"width": max(int(max_x) + 40, 800), "height": max(int(max_y) + 40, 600)}


def bpmn_buml_to_json(content: str) -> dict:
    """Convert a BPMN BUML ``.py`` source string into a v4 WME BPMN diagram JSON dict.

    Loads ``content`` through ``safe_load_buml`` (AST allowlist — no raw ``exec``),
    locates the resulting ``BPMNModel``, and delegates to :func:`bpmn_object_to_json`.

    Raises:
        ConversionError: if the source fails to parse / execute, or if no
            ``BPMNModel`` instance is produced.
    """
    allowed_names = {
        "BPMNModel": BPMNModel, "Process": Process, "Collaboration": Collaboration,
        "Participant": Participant, "Task": Task, "TaskType": TaskType,
        "LoopCharacteristics": LoopCharacteristics, "SubProcess": SubProcess,
        "Transaction": Transaction, "CallActivity": CallActivity, "StartEvent": StartEvent,
        "IntermediateEvent": IntermediateEvent, "EndEvent": EndEvent,
        "EventDirection": EventDirection, "EventDefinitionType": EventDefinitionType,
        "Gateway": Gateway, "GatewayType": GatewayType, "SequenceFlow": SequenceFlow,
        "MessageFlow": MessageFlow, "Association": Association, "DataAssociation": DataAssociation,
        "DataObject": DataObject, "DataStore": DataStore, "Lane": Lane, "Group": Group,
        "TextAnnotation": TextAnnotation, "set": set,
        "Project": lambda *args, **kwargs: None,
    }

    cleaned_lines = []
    in_import_block = False
    for line in content.splitlines():
        stripped = line.lstrip()
        if in_import_block:
            if ")" in line:
                in_import_block = False
            continue
        if stripped.startswith(("import ", "from ")):
            if "(" in line and ")" not in line:
                in_import_block = True
            continue
        if any(gen in line for gen in ["Generator(", ".generate("]):
            continue
        cleaned_lines.append(line)
    cleaned_content = "\n".join(cleaned_lines)

    try:
        local_vars = safe_load_buml(cleaned_content, allowed_names)
    except (SyntaxError, NameError, TypeError, ValueError) as exc:
        raise ConversionError(f"BPMN BUML file failed to execute: {exc}") from exc

    model = _find_bpmn_model(local_vars)
    if model is None:
        raise ConversionError(
            "BPMN BUML file produced no BPMNModel — expected a top-level variable "
            "(`bpmn_model = BPMNModel(...)` is the convention emitted by `bpmn_model_to_code`)."
        )
    return bpmn_object_to_json(model)


def _find_bpmn_model(namespace: dict):
    """Return the ``BPMNModel`` from the loaded namespace, preferring ``bpmn_model``."""
    candidate = namespace.get("bpmn_model")
    if isinstance(candidate, BPMNModel):
        return candidate
    for value in namespace.values():
        if isinstance(value, BPMNModel):
            return value
    return None
