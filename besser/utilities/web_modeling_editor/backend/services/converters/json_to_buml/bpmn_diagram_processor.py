"""BPMN processing for converting JSON to BUML format.

Returns a ``BPMNModel`` metamodel instance from the WME BPMN diagram JSON envelope. The
algorithm is the standard nodes-first / edges-second pass with a containment pass in
between (see ``.claude/bpmn/03-bpmn-converters-guide.md`` §4).

Design points:

* **Identity is by object** (decision D5): WME ids ride in ``element.layout`` so the
  converter pair can round-trip them, but the metamodel itself stays id-free.
* **Layout is opaque** (decision D8): ``BPMNElement.layout`` carries the WME bounds /
  path / stash without the metamodel ever interpreting it.
* **Validation is a separate concern**: this processor never calls ``BPMNModel.validate``
  — it only produces a ``BPMNModel``. Callers run validation if they need it.
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
    BPMN_DIAGRAM_TYPE,
    BPMN_RELATIONSHIP_TYPE,
)
from besser.utilities.web_modeling_editor.backend.services.converters.bpmn_event_mapping import (
    parse_event_type,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WME element-type → metamodel class dispatch
# ---------------------------------------------------------------------------

_EVENT_CLASSES = {
    "BPMNStartEvent": StartEvent,
    "BPMNIntermediateEvent": IntermediateEvent,
    "BPMNEndEvent": EndEvent,
}

_ACTIVITY_CLASSES = {
    "BPMNTask": Task,
    "BPMNSubprocess": SubProcess,
    "BPMNTransaction": Transaction,
    "BPMNCallActivity": CallActivity,
}


def _layout_dict(elem_id: str, elem: dict) -> dict:
    """Build the opaque ``layout`` passthrough for a node element.

    Stores everything the converter needs to round-trip the WME shape that is *not* part
    of the abstract syntax: original id, owner pointer, bounds, and any styling fields.
    """
    layout = {
        "id": elem_id,
        "owner": elem.get("owner"),
        "bounds": elem.get("bounds"),
    }
    for style_key in ("fillColor", "strokeColor", "textColor", "highlight"):
        if style_key in elem:
            layout[style_key] = elem[style_key]
    return layout


def _flow_layout_dict(rel_id: str, rel: dict) -> dict:
    """Build the opaque ``layout`` passthrough for a relationship element."""
    return {
        "id": rel_id,
        "owner": rel.get("owner"),
        "bounds": rel.get("bounds"),
        "path": rel.get("path"),
        "source_direction": (rel.get("source") or {}).get("direction"),
        "target_direction": (rel.get("target") or {}).get("direction"),
        "isManuallyLayouted": rel.get("isManuallyLayouted", False),
    }


def _build_node(elem: dict):
    """Construct the metamodel object for one WME ``elements[id]`` entry.

    Returns ``None`` for unknown types (caller logs and skips). Pool elements are
    constructed *with* their ``Process`` so containment in pass 2 has somewhere to land.

    Raises:
        ConversionError: if a known WME type carries an unrecognised enum string for one
            of its subtype attributes.
    """
    elem_type = elem.get("type")
    name = elem.get("name", "") or ""

    if elem_type in _ACTIVITY_CLASSES:
        cls = _ACTIVITY_CLASSES[elem_type]
        try:
            loop = LoopCharacteristics(elem.get("marker", "none") or "none")
        except ValueError as exc:
            raise ConversionError(
                f"Unknown BPMN loop marker '{elem.get('marker')}' on element '{name}'."
            ) from exc
        if cls is Task:
            try:
                task_type = TaskType(elem.get("taskType", "default") or "default")
            except ValueError as exc:
                raise ConversionError(
                    f"Unknown BPMN task type '{elem.get('taskType')}' on Task '{name}'."
                ) from exc
            return Task(name=name, task_type=task_type, loop_characteristics=loop)
        # SubProcess / Transaction / CallActivity all take the same args (no task_type).
        return cls(name=name, loop_characteristics=loop)

    if elem_type in _EVENT_CLASSES:
        cls = _EVENT_CLASSES[elem_type]
        event_type = elem.get("eventType", "default") or "default"
        try:
            direction, definition = parse_event_type(cls, event_type)
        except ValueError as exc:
            raise ConversionError(
                f"Unknown BPMN event type '{event_type}' on {cls.__name__} '{name}': {exc}"
            ) from exc
        return cls(name=name, direction=direction, event_definition=definition)

    if elem_type == "BPMNGateway":
        try:
            gateway_type = GatewayType(elem.get("gatewayType", "exclusive") or "exclusive")
        except ValueError as exc:
            raise ConversionError(
                f"Unknown BPMN gateway type '{elem.get('gatewayType')}' on Gateway '{name}'."
            ) from exc
        return Gateway(name=name, gateway_type=gateway_type)

    if elem_type == "BPMNDataObject":
        return DataObject(name=name)
    if elem_type == "BPMNDataStore":
        return DataStore(name=name)
    if elem_type == "BPMNAnnotation":
        # WME stores the annotation body in `name`; the metamodel keeps it in `text`.
        return TextAnnotation(name="", text=name)
    if elem_type == "BPMNGroup":
        return Group(name=name)
    if elem_type == "BPMNSwimlane":
        return Lane(name=name)
    if elem_type == "BPMNPool":
        # Build the Pool's Process eagerly so pass 2 containment can attach to it.
        return Participant(name=name, process=Process(name=name))

    return None


def _build_flow(rel: dict, source, target):
    """Construct the metamodel flow object for one WME ``relationships[id]`` entry.

    Branches on ``flowType``. Wraps ``is_default``: an illegal value (WME source can't
    legally carry a default) downgrades to ``False`` with a warning rather than aborting.

    Raises:
        ConversionError: if ``flowType`` is missing or unknown.
        TypeError / ValueError: if endpoint types violate the connecting object's rules
            (caller logs and skips).
    """
    name = rel.get("name", "") or ""
    flow_type = rel.get("flowType")

    if flow_type == "sequence":
        flow = SequenceFlow(source=source, target=target, name=name)
        if rel.get("isDefault"):
            try:
                flow.is_default = True
            except ValueError as exc:
                logger.warning(
                    "SequenceFlow '%s' marked default but its source cannot carry one (%s); "
                    "downgrading to is_default=False.", name, exc,
                )
        return flow
    if flow_type == "message":
        return MessageFlow(source=source, target=target, name=name)
    if flow_type == "association":
        return Association(source=source, target=target, name=name)
    if flow_type == "data association":
        return DataAssociation(source=source, target=target, name=name)

    raise ConversionError(f"Unknown BPMN flowType '{flow_type}' on relationship '{name}'.")


def _outer_process(container):
    """Walk ``SubProcess`` containment up to the enclosing ``Process``, or ``None``."""
    while isinstance(container, SubProcess):
        container = container.container
    return container if isinstance(container, Process) else None


def process_bpmn_diagram(json_data: dict) -> BPMNModel:
    """Convert a WME BPMN diagram (JSON) into a ``BPMNModel`` instance.

    Args:
        json_data: Dictionary with the standard Apollon ``UMLModel`` envelope; the BPMN
            payload sits under ``json_data["model"]`` with ``elements`` (nodes) and
            ``relationships`` (flows) maps keyed by id.

    Returns:
        ``BPMNModel``. ``model.collaboration`` is ``None`` for a pool-less diagram and a
        ``Collaboration`` (with one ``Participant`` per pool) otherwise.

    Raises:
        ConversionError: structural failures (missing ``model`` key, unknown WME element
            type with no metamodel mapping, unknown enum strings, unknown ``flowType``).
            Non-fatal cases (dangling flow endpoint, illegal ``isDefault``) log and skip.
    """
    model_data = json_data.get("model")
    if not model_data:
        raise ConversionError("BPMN diagram JSON is missing the 'model' key.")

    if model_data.get("type") and model_data.get("type") != BPMN_DIAGRAM_TYPE:
        # Don't reject — WME may not have aligned yet (the WME-side agent task tracked in
        # `.claude/bpmn/bpmn-metamodel-work.md`). Log so the mismatch is visible.
        logger.warning(
            "BPMN diagram envelope type is '%s', expected '%s'.",
            model_data.get("type"), BPMN_DIAGRAM_TYPE,
        )

    elements = model_data.get("elements") or {}
    relationships = model_data.get("relationships") or {}
    name = json_data.get("title") or "Generated_BPMN_Model"

    # --- Pass 1: build node objects (no containment yet) -------------------
    node_by_id: dict = {}
    pools: list = []  # (elem_id, Participant) — preserves JSON order

    for elem_id, elem in elements.items():
        try:
            obj = _build_node(elem)
        except ConversionError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                f"Could not build BPMN element '{elem_id}': {exc}"
            ) from exc
        if obj is None:
            logger.warning(
                "BPMN element '%s' has unknown type '%s'; skipping.",
                elem_id, elem.get("type"),
            )
            continue
        obj.layout = _layout_dict(elem_id, elem)
        node_by_id[elem_id] = obj
        if isinstance(obj, Participant):
            pools.append((elem_id, obj))

    # --- Pass 2: containment via the `owner` chain --------------------------
    # `process_of` covers every placed element (FlowNode, Artifact, DataObject) so the
    # flow router (Pass 3) can resolve a flow's owning Process via either endpoint.
    process_of: dict = {}
    synthetic_process = None

    def _ensure_synthetic_process():
        nonlocal synthetic_process
        if synthetic_process is None:
            synthetic_process = Process(name=name)
        return synthetic_process

    # Sub-pass 2a: attach Lanes to their pool's Process first, so flow-node placement
    # in 2b can route into the correct lane.
    for elem_id, obj in node_by_id.items():
        if not isinstance(obj, Lane):
            continue
        owner_id = elements[elem_id].get("owner")
        owner_obj = node_by_id.get(owner_id) if owner_id else None
        if isinstance(owner_obj, Participant) and owner_obj.process is not None:
            owner_obj.process.add_lane(obj)
        else:
            logger.warning(
                "BPMN Lane '%s' has no Pool owner; orphan lane skipped.", elem_id,
            )

    # Sub-pass 2b: place every other element. Skip Pools (no owner-based containment) and
    # Lanes (already done) and DataStores (live on the model, not in a process).
    for elem_id, obj in node_by_id.items():
        if isinstance(obj, (Participant, Lane, DataStore)):
            continue

        owner_id = elements[elem_id].get("owner")
        owner_obj = node_by_id.get(owner_id) if owner_id else None

        target_container = None
        target_lane = None

        if owner_obj is None:
            target_container = _ensure_synthetic_process()
        elif isinstance(owner_obj, Lane):
            target_lane = owner_obj
            # The lane was attached to its pool's Process in 2a — find that Process.
            for _, part in pools:
                if part.process is not None and target_lane in part.process.lanes:
                    target_container = part.process
                    break
            if target_container is None:
                target_container = _ensure_synthetic_process()
        elif isinstance(owner_obj, Participant):
            target_container = owner_obj.process or _ensure_synthetic_process()
        elif isinstance(owner_obj, SubProcess):
            target_container = owner_obj
        else:
            logger.warning(
                "BPMN element '%s' has unexpected owner type '%s'; placing at top level.",
                elem_id, type(owner_obj).__name__,
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

    # Data stores live model-wide (D8 / spec §10.3 — root-level element).
    data_stores = {obj for obj in node_by_id.values() if isinstance(obj, DataStore)}

    # --- Build collaboration / processes set -------------------------------
    if pools:
        collaboration = Collaboration(
            name=name,
            participants={part for _, part in pools},
        )
        processes = {part.process for _, part in pools if part.process is not None}
    else:
        collaboration = None
        processes = set()
    if synthetic_process is not None:
        processes.add(synthetic_process)

    # --- Pass 3: build flows from `relationships` --------------------------
    for rel_id, rel in relationships.items():
        if rel.get("type") != BPMN_RELATIONSHIP_TYPE:
            logger.warning(
                "BPMN relationship '%s' has unexpected type '%s'; skipping.",
                rel_id, rel.get("type"),
            )
            continue

        source = node_by_id.get((rel.get("source") or {}).get("element"))
        target = node_by_id.get((rel.get("target") or {}).get("element"))
        if source is None or target is None:
            logger.warning(
                "BPMN flow '%s' has a dangling endpoint (source=%s, target=%s); skipping.",
                rel_id,
                (rel.get("source") or {}).get("element"),
                (rel.get("target") or {}).get("element"),
            )
            continue

        try:
            flow = _build_flow(rel, source, target)
        except ConversionError:
            raise
        except (TypeError, ValueError) as exc:
            logger.warning(
                "BPMN flow '%s' could not be built (%s); skipping.", rel_id, exc,
            )
            continue

        flow.layout = _flow_layout_dict(rel_id, rel)

        if isinstance(flow, MessageFlow):
            if collaboration is not None:
                collaboration.add_message_flow(flow)
            else:
                logger.warning(
                    "BPMN MessageFlow '%s' present but the diagram has no pools; skipping.",
                    rel_id,
                )
            continue

        # SequenceFlow / Association / DataAssociation: route via either endpoint.
        endpoint_container = process_of.get(source) or process_of.get(target)
        if endpoint_container is None:
            logger.warning(
                "BPMN flow '%s' has no resolvable container; skipping.", rel_id,
            )
            continue

        if isinstance(flow, SequenceFlow):
            # SequenceFlows go into their endpoints' immediate container (Process or
            # SubProcess). The metamodel's E2 rule guarantees both endpoints share one.
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

    return BPMNModel(
        name=name,
        processes=processes,
        collaboration=collaboration,
        data_stores=data_stores,
    )
