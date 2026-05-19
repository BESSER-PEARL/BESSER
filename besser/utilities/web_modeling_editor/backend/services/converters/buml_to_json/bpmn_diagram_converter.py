"""BPMN conversion from BUML to JSON format.

Two entry points:

* ``bpmn_object_to_json(model: BPMNModel) -> dict`` — converts a metamodel object
  directly. Mirror of ``json_to_buml.bpmn_diagram_processor.process_bpmn_diagram``.
* ``bpmn_to_json(content: str) -> dict`` — execs a BPMN BUML ``.py`` source string
  in a fresh namespace, finds the resulting ``BPMNModel``, and delegates to
  ``bpmn_object_to_json``. The ``.py`` files emitted by
  ``besser.utilities.buml_code_builder.bpmn_model_builder.bpmn_model_to_code`` are
  exactly what this wrapper expects.

Design points (mirror of the processor):

* **Stable round-trip ids.** ``id_for(obj)`` reuses the original WME id stashed in
  ``obj.layout["id"]`` when present, falling back to a fresh uuid. This is the §6 / D5-Q4
  mechanism — the metamodel stays id-free; ``layout`` is the per-element side-channel.
* **Layout fallback.** When ``layout`` is missing / partial (a freshly-built model, or
  one loaded from BUML code that didn't emit layout), a deterministic grid layout fills
  the gaps; flow paths reuse ``services.utils.layout_calculator``.
* **Deterministic walk.** Set members are walked in ``timestamp`` order via
  ``besser.utilities.sort_by_timestamp`` so the export is reproducible (BPMN names are
  not unique — sorting by name is not an option).
"""

import logging
import uuid

from besser.BUML.metamodel.bpmn import (
    Activity,
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    Association,
    BPMNConnectingObject,
    BPMNModel,
    DataAssociation,
    DataObject,
    DataStore,
    EndEvent,
    Event,
    Gateway,
    Group,
    IntermediateEvent,
    Lane,
    MessageFlow,
    Participant,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TextAnnotation,
    Transaction,
)
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    BPMN_DIAGRAM_TYPE,
    BPMN_RELATIONSHIP_TYPE,
)
from besser.utilities.web_modeling_editor.backend.services.converters.bpmn_event_mapping import (
    serialise_event_type,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError
from besser.utilities.web_modeling_editor.backend.services.utils import (
    calculate_connection_points,
    calculate_path_points,
    calculate_relationship_bounds,
    determine_connection_direction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WME element-type and default-bounds tables
# ---------------------------------------------------------------------------

# Reverse of the processor's dispatch — every concrete metamodel type that maps to a
# distinct WME ``elements[id]["type"]`` string.
_TYPE_FOR_CLASS = {
    Task: "BPMNTask",
    AgenticTask: "BPMNTask",          # SEAA'25 subclass: same WME element type
    Transaction: "BPMNTransaction",   # checked before SubProcess (subclass)
    SubProcess: "BPMNSubprocess",
    "CallActivity": "BPMNCallActivity",  # resolved below
    StartEvent: "BPMNStartEvent",
    IntermediateEvent: "BPMNIntermediateEvent",
    EndEvent: "BPMNEndEvent",
    Gateway: "BPMNGateway",
    AgenticGateway: "BPMNGateway",    # SEAA'25 subclass: same WME element type
    DataObject: "BPMNDataObject",
    DataStore: "BPMNDataStore",
    TextAnnotation: "BPMNAnnotation",
    Group: "BPMNGroup",
    Lane: "BPMNSwimlane",
    AgenticLane: "BPMNSwimlane",      # SEAA'25 subclass: same WME element type
    Participant: "BPMNPool",
}

# Default per-class bounds when ``layout["bounds"]`` is missing.
_DEFAULT_BOUNDS = {
    "BPMNTask": {"width": 110, "height": 60},
    "BPMNSubprocess": {"width": 250, "height": 150},
    "BPMNTransaction": {"width": 250, "height": 150},
    "BPMNCallActivity": {"width": 110, "height": 60},
    "BPMNStartEvent": {"width": 40, "height": 40},
    "BPMNIntermediateEvent": {"width": 40, "height": 40},
    "BPMNEndEvent": {"width": 40, "height": 40},
    "BPMNGateway": {"width": 50, "height": 50},
    "BPMNDataObject": {"width": 40, "height": 50},
    "BPMNDataStore": {"width": 60, "height": 50},
    "BPMNAnnotation": {"width": 200, "height": 80},
    "BPMNGroup": {"width": 240, "height": 200},
    "BPMNSwimlane": {"width": 800, "height": 120},
    "BPMNPool": {"width": 800, "height": 240},
}

_FLOW_TYPE_FOR_CLASS = {
    SequenceFlow: "sequence",
    MessageFlow: "message",
    Association: "association",
    DataAssociation: "data association",
}


# WME's BPMNTask / BPMNGateway / BPMNSwimlane always serialise these SEAA'25
# fields with hard defaults when the element is not agentic. Mirror exactly so
# BESSER-emitted JSON matches WME's own JSON byte-for-byte on non-agentic
# elements. The values are taken from WME's `dev/bpmn`
# packages/editor/.../bpmn-{task,gateway,swimlane}.ts ``default*`` statics.
_WME_TASK_DEFAULTS = {
    "isAgentic": False,
    "reflectionMode": "none",
    "trustScore": 0,
    # `collaborationMode` is a WME extension beyond paper §4.2 Fig 3b. BESSER
    # doesn't store it (01-... §6.5 Q-E); emitted as a placeholder so WME's
    # deserialiser keeps the field present.
    "collaborationMode": "voting",
}
_WME_GATEWAY_DEFAULTS = {
    "isAgentic": False,
    "gatewayRole": "diverging",
    "collaborationMode": "voting",
    # WME's hard default; emitted unconditionally so the JSON shape stays
    # WME-compatible even for non-agentic gateways (which have no merging
    # strategy concept). BESSER's diverging AgenticGateway stores None and
    # also emits this placeholder (03-... §3.3 decision).
    "mergingStrategy": "majority",
    "trustScore": 0,
}
_WME_LANE_DEFAULTS = {
    "isAgentic": False,
    "role": "worker",
    "trustScore": 0,
}
_WME_FLOW_AGENTIC_DEFAULTS = {
    # AgenticMessageFlow is out of scope (01-... Q-D / §6.5). WME's BPMNFlow
    # always carries these fields; emit defaults so the JSON shape stays
    # WME-compatible.
    "isAgentic": False,
    "collaborationMode": "voting",
    "mergingStrategy": "majority",
}


def _wme_type_for(obj) -> str:
    """Return the WME ``type`` string for a metamodel object."""
    cls = type(obj)
    if cls.__name__ == "CallActivity":
        return "BPMNCallActivity"
    return _TYPE_FOR_CLASS.get(cls, "")


# ---------------------------------------------------------------------------
# bpmn_object_to_json
# ---------------------------------------------------------------------------

def bpmn_object_to_json(model: BPMNModel) -> dict:
    """Convert a ``BPMNModel`` into a WME BPMN diagram JSON dict.

    Args:
        model: A ``BPMNModel`` metamodel instance.

    Returns:
        Dict in the standard Apollon BPMN envelope (``version``, ``type``, ``size``,
        ``elements``, ``relationships``, ``interactive``, ``assessments``).
    """
    elements: dict = {}
    relationships: dict = {}

    # Per-call id map: stable round-trip ids via the layout stash; fresh uuids otherwise.
    id_map: dict = {}
    grid = _GridLayout()

    def id_for(obj) -> str:
        if obj not in id_map:
            stashed = (obj.layout or {}).get("id") if hasattr(obj, "layout") else None
            id_map[obj] = stashed or str(uuid.uuid4())
        return id_map[obj]

    # --- 1. Pools (and their lanes) ----------------------------------------
    if model.collaboration is not None:
        for participant in sort_by_timestamp(model.collaboration.participants):
            _emit_pool(participant, elements, id_for, grid)

    # --- 2. Pool-less Processes (no Participant) ---------------------------
    pooled_processes = (
        {p.process for p in model.collaboration.participants if p.process is not None}
        if model.collaboration is not None
        else set()
    )
    for process in sort_by_timestamp(model.processes):
        if process in pooled_processes:
            continue
        _emit_process_contents(
            process, owner_id=None, elements=elements, id_for=id_for, grid=grid,
        )

    # --- 3. Pool-ful Processes' contents -----------------------------------
    if model.collaboration is not None:
        for participant in sort_by_timestamp(model.collaboration.participants):
            if participant.process is None:
                continue
            _emit_process_contents(
                participant.process,
                owner_id=id_for(participant),
                elements=elements,
                id_for=id_for,
                grid=grid,
            )

    # --- 4. Model-level data stores ----------------------------------------
    for data_store in sort_by_timestamp(model.data_stores):
        elements[id_for(data_store)] = _emit_node(
            data_store, owner_id=None, id_for=id_for, grid=grid,
        )

    # --- 5. Flows ----------------------------------------------------------
    for process in sort_by_timestamp(model.processes):
        for flow in sort_by_timestamp(process.sequence_flows):
            _emit_flow(flow, relationships, elements, id_for)
        for flow in sort_by_timestamp(process.associations):
            _emit_flow(flow, relationships, elements, id_for)
        for flow in sort_by_timestamp(process.data_associations):
            _emit_flow(flow, relationships, elements, id_for)

    # Sequence flows nested in SubProcesses
    for sub in _walk_subprocesses(model):
        for flow in sort_by_timestamp(sub.sequence_flows):
            _emit_flow(flow, relationships, elements, id_for)

    if model.collaboration is not None:
        for flow in sort_by_timestamp(model.collaboration.message_flows):
            _emit_flow(flow, relationships, elements, id_for)

    # --- 6. Envelope -------------------------------------------------------
    size = _compute_envelope_size(elements)

    return {
        "version": "3.0.0",
        "type": BPMN_DIAGRAM_TYPE,
        "size": size,
        "interactive": {"elements": {}, "relationships": {}},
        "elements": elements,
        "relationships": relationships,
        "assessments": {},
    }


# ---------------------------------------------------------------------------
# Element emission helpers
# ---------------------------------------------------------------------------

def _emit_pool(participant: Participant, elements: dict, id_for, grid: "_GridLayout") -> None:
    """Emit a pool (``BPMNPool``) and its lanes (``BPMNSwimlane``)."""
    pool_id = id_for(participant)
    elements[pool_id] = _emit_node(participant, owner_id=None, id_for=id_for, grid=grid)

    if participant.process is None:
        return
    for lane in sort_by_timestamp(participant.process.lanes):
        elements[id_for(lane)] = _emit_node(
            lane, owner_id=pool_id, id_for=id_for, grid=grid,
        )


def _emit_process_contents(process: Process, owner_id, elements: dict,
                           id_for, grid: "_GridLayout") -> None:
    """Emit every child of ``process``: flow nodes (recursing into sub-processes),
    artifacts, and data objects.

    ``owner_id`` is the WME owner pointer for top-level children: the pool id when the
    process is in a pool, ``None`` otherwise. Children inside a lane resolve their owner
    to the lane id; children inside a sub-process resolve to the sub-process id.
    """
    for node in sort_by_timestamp(process.flow_nodes):
        node_owner_id = id_for(node.lane) if node.lane is not None else owner_id
        elements[id_for(node)] = _emit_node(
            node, owner_id=node_owner_id, id_for=id_for, grid=grid,
        )
        if isinstance(node, SubProcess):
            _emit_subprocess_children(node, elements, id_for, grid)

    for artifact in sort_by_timestamp(process.artifacts):
        elements[id_for(artifact)] = _emit_node(
            artifact, owner_id=owner_id, id_for=id_for, grid=grid,
        )
    for data_object in sort_by_timestamp(process.data_objects):
        elements[id_for(data_object)] = _emit_node(
            data_object, owner_id=owner_id, id_for=id_for, grid=grid,
        )


def _emit_subprocess_children(sub: SubProcess, elements: dict,
                              id_for, grid: "_GridLayout") -> None:
    """Recursively emit a sub-process's flow nodes (their owner is the sub-process)."""
    sub_id = id_for(sub)
    for node in sort_by_timestamp(sub.flow_nodes):
        elements[id_for(node)] = _emit_node(
            node, owner_id=sub_id, id_for=id_for, grid=grid,
        )
        if isinstance(node, SubProcess):
            _emit_subprocess_children(node, elements, id_for, grid)


def _emit_node(obj, owner_id, id_for, grid: "_GridLayout") -> dict:
    """Build one ``elements[id]`` entry for a metamodel object."""
    wme_type = _wme_type_for(obj)
    if not wme_type:
        # Should not happen — guard anyway.
        logger.warning("BPMN export: unknown metamodel class %s; skipping.", type(obj).__name__)
        return {}

    layout = obj.layout or {}
    bounds = layout.get("bounds") or grid.next_bounds(wme_type)

    name = obj.name
    if isinstance(obj, TextAnnotation):
        # The metamodel keeps the body in `text` — WME stores it back in `name`.
        name = obj.text

    entry: dict = {
        "id": id_for(obj),
        "name": name,
        "type": wme_type,
        "owner": owner_id,
        "bounds": bounds,
    }
    for style_key in ("fillColor", "strokeColor", "textColor", "highlight"):
        if style_key in layout:
            entry[style_key] = layout[style_key]

    # Subtype attrs — only on the relevant types.
    if isinstance(obj, Task):
        entry["taskType"] = obj.task_type.value
    if isinstance(obj, Activity):
        entry["marker"] = obj.loop_characteristics.value
    if isinstance(obj, Event):
        entry["eventType"] = serialise_event_type(obj)
    if isinstance(obj, Gateway):
        entry["gatewayType"] = obj.gateway_type.value

    # SEAA'25 agentic fields — every Task / Gateway / Lane entry carries
    # them in WME's JSON shape (defaulted when the element is not agentic).
    if isinstance(obj, Task):
        entry.update(_WME_TASK_DEFAULTS)
        if isinstance(obj, AgenticTask):
            entry["isAgentic"] = True
            entry["reflectionMode"] = obj.reflection_mode.value
            entry["trustScore"] = obj.trust_score
            # `collaborationMode` stays at the WME placeholder ("voting") --
            # BESSER doesn't store it (01-... §6.5 Q-E).
    if isinstance(obj, Gateway):
        entry.update(_WME_GATEWAY_DEFAULTS)
        if isinstance(obj, AgenticGateway):
            entry["isAgentic"] = True
            entry["gatewayRole"] = obj.gateway_role.value
            entry["collaborationMode"] = obj.collaboration_mode.value
            # mergingStrategy: emit the agentic value when MERGING (non-None);
            # on DIVERGING gateways (None in the metamodel) leave the WME
            # placeholder "majority" from _WME_GATEWAY_DEFAULTS so the JSON
            # shape stays byte-identical to WME's own export (03-... §3.3).
            if obj.merging_strategy is not None:
                entry["mergingStrategy"] = obj.merging_strategy.value
            entry["trustScore"] = obj.trust_score
    if isinstance(obj, Lane):
        entry.update(_WME_LANE_DEFAULTS)
        if isinstance(obj, AgenticLane):
            entry["isAgentic"] = True
            entry["role"] = obj.role.value
            entry["trustScore"] = obj.trust_score

    return entry


# ---------------------------------------------------------------------------
# Flow emission helpers
# ---------------------------------------------------------------------------

def _emit_flow(flow: BPMNConnectingObject, relationships: dict,
               elements: dict, id_for) -> None:
    """Build one ``relationships[id]`` entry for a flow."""
    flow_type = _FLOW_TYPE_FOR_CLASS.get(type(flow))
    if flow_type is None:
        logger.warning(
            "BPMN export: unknown connecting object %s; skipping.", type(flow).__name__,
        )
        return

    source_id = id_for(flow.source)
    target_id = id_for(flow.target)
    source_entry = elements.get(source_id)
    target_entry = elements.get(target_id)
    if source_entry is None or target_entry is None:
        # Endpoint wasn't emitted (defensive — happens only on a malformed model).
        logger.warning(
            "BPMN flow '%s' references an endpoint that was not emitted; skipping.",
            flow.name,
        )
        return

    layout = flow.layout or {}
    source_dir, target_dir = _resolve_directions(layout, source_entry, target_entry)
    source_point = calculate_connection_points(source_entry["bounds"], source_dir)
    target_point = calculate_connection_points(target_entry["bounds"], target_dir)
    path = layout.get("path") or calculate_path_points(
        source_point, target_point, source_dir, target_dir,
    )
    bounds = layout.get("bounds") or calculate_relationship_bounds(path)

    entry: dict = {
        "id": id_for(flow),
        "name": flow.name,
        "type": BPMN_RELATIONSHIP_TYPE,
        "owner": layout.get("owner"),
        "bounds": bounds,
        "path": path,
        "source": {
            "direction": source_dir,
            "element": source_id,
            "bounds": {"x": source_point["x"], "y": source_point["y"], "width": 0, "height": 0},
        },
        "target": {
            "direction": target_dir,
            "element": target_id,
            "bounds": {"x": target_point["x"], "y": target_point["y"], "width": 0, "height": 0},
        },
        "isManuallyLayouted": layout.get("isManuallyLayouted", False),
        "flowType": flow_type,
    }
    if isinstance(flow, SequenceFlow):
        entry["isDefault"] = flow.is_default

    # AgenticMessageFlow is out of scope (01-... Q-D). WME's BPMNFlow always
    # carries these fields; emit WME defaults so the JSON shape stays
    # WME-compatible regardless of flow class.
    entry.update(_WME_FLOW_AGENTIC_DEFAULTS)

    relationships[id_for(flow)] = entry


def _resolve_directions(layout: dict, source_entry: dict, target_entry: dict):
    """Pick connection directions for an emitted flow.

    Prefers the directions stashed in ``layout`` so a round-tripped flow keeps the same
    connection sides; falls back to ``determine_connection_direction`` from the bounds.
    """
    source_dir = layout.get("source_direction")
    target_dir = layout.get("target_direction")
    if source_dir and target_dir:
        return source_dir, target_dir
    return determine_connection_direction(source_entry["bounds"], target_entry["bounds"])


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _walk_subprocesses(model: BPMNModel):
    """Yield every ``SubProcess`` (including nested ones) in the model."""
    def _recurse(container):
        for node in container.flow_nodes:
            if isinstance(node, SubProcess):
                yield node
                yield from _recurse(node)

    for process in model.processes:
        yield from _recurse(process)


def _compute_envelope_size(elements: dict) -> dict:
    """Bounding box of all element bounds, with a sensible minimum."""
    if not elements:
        return {"width": 800, "height": 600}
    max_x = 0
    max_y = 0
    for entry in elements.values():
        bounds = entry.get("bounds") or {}
        right = (bounds.get("x") or 0) + (bounds.get("width") or 0)
        bottom = (bounds.get("y") or 0) + (bounds.get("height") or 0)
        if right > max_x:
            max_x = right
        if bottom > max_y:
            max_y = bottom
    return {"width": max(int(max_x) + 40, 800), "height": max(int(max_y) + 40, 600)}


# ---------------------------------------------------------------------------
# Grid layout (deterministic fallback when layout is missing)
# ---------------------------------------------------------------------------

class _GridLayout:
    """Deterministic per-class grid placement used only when ``layout["bounds"]`` is
    absent. Each WME type advances its own row so nodes of different sizes don't overlap.
    """

    _ROW_FOR_TYPE = {
        "BPMNPool": 0,
        "BPMNSwimlane": 1,
        "BPMNStartEvent": 2,
        "BPMNTask": 2,
        "BPMNGateway": 2,
        "BPMNCallActivity": 2,
        "BPMNIntermediateEvent": 2,
        "BPMNEndEvent": 2,
        "BPMNSubprocess": 3,
        "BPMNTransaction": 3,
        "BPMNDataObject": 4,
        "BPMNDataStore": 4,
        "BPMNAnnotation": 5,
        "BPMNGroup": 5,
    }

    def __init__(self):
        self._x_for_row: dict = {}

    def next_bounds(self, wme_type: str) -> dict:
        size = _DEFAULT_BOUNDS.get(wme_type, {"width": 100, "height": 60})
        row = self._ROW_FOR_TYPE.get(wme_type, 6)
        x = self._x_for_row.get(row, 40)
        y = 40 + row * 180
        self._x_for_row[row] = x + size["width"] + 40
        return {"x": x, "y": y, "width": size["width"], "height": size["height"]}


# ---------------------------------------------------------------------------
# bpmn_to_json — BUML .py source string → WME JSON
# ---------------------------------------------------------------------------

def bpmn_to_json(content: str) -> dict:
    """Convert a BPMN BUML ``.py`` source string into a WME BPMN diagram JSON dict.

    Execs ``content`` in a fresh namespace, locates the resulting ``BPMNModel``, and
    delegates to :func:`bpmn_object_to_json`. The ``.py`` files emitted by
    ``besser.utilities.buml_code_builder.bpmn_model_builder.bpmn_model_to_code`` are
    exactly what this wrapper expects.

    Args:
        content: BPMN BUML Python source code as a string.

    Returns:
        A WME BPMN diagram JSON dict (the standard Apollon envelope).

    Raises:
        ConversionError: if the source fails to parse / execute, or if no
            ``BPMNModel`` instance is produced.
    """
    namespace: dict = {}
    try:
        exec(content, namespace)
    except (SyntaxError, NameError, TypeError, ValueError) as exc:
        raise ConversionError(f"BPMN BUML file failed to execute: {exc}") from exc

    model = _find_bpmn_model(namespace)
    if model is None:
        raise ConversionError(
            "BPMN BUML file produced no BPMNModel — expected a top-level variable "
            "(`bpmn_model = BPMNModel(...)` is the convention emitted by "
            "`bpmn_model_to_code`)."
        )
    return bpmn_object_to_json(model)


def _find_bpmn_model(namespace: dict):
    """Return the ``BPMNModel`` from the exec'd namespace, preferring ``bpmn_model``."""
    candidate = namespace.get("bpmn_model")
    if isinstance(candidate, BPMNModel):
        return candidate
    for value in namespace.values():
        if isinstance(value, BPMNModel):
            return value
    return None
