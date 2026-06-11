"""BPMN Model Builder.

Generates Python code from a ``BPMNModel`` metamodel instance. The emitted code can be
``exec()``'d to reconstruct an identical model — that is the contract
``services/converters/buml_to_json/bpmn_diagram_converter.bpmn_to_json`` relies on.

Build style — option (b) per ``.claude/bpmn/04-bpmn-code-builder-guide.md``: empty
containers first, then ``add_*`` calls. Mirrors ``state_machine_builder.py`` so the
emitted modules read the same way as the existing diagram-type builders.
"""

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    Association,
    BPMNModel,
    CallActivity,
    DataAssociation,
    EndEvent,
    Gateway,
    Group,
    IntermediateEvent,
    MessageFlow,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TextAnnotation,
    Transaction,
)
from besser.utilities.buml_code_builder.common import (
    _escape_python_string,
    safe_var_name,
)
from besser.utilities.utils import sort_by_timestamp


# ---------------------------------------------------------------------------
# Variable name dispenser
# ---------------------------------------------------------------------------

class _NameDispenser:
    """Mints a unique Python variable name per metamodel object.

    BPMN names can be empty, repeated, or whitespace-only (decision D5 — identity is by
    object). The dispenser combines the metamodel class name with a sanitised version
    of ``obj.name`` and a numeric suffix when needed, then caches the result so all
    later references (e.g. flow endpoints) resolve to the same identifier.
    """

    def __init__(self):
        self._used: set = set()
        self._for_obj: dict = {}

    def name_for(self, obj) -> str:
        if obj in self._for_obj:
            return self._for_obj[obj]
        prefix = type(obj).__name__.lower()
        base = safe_var_name(obj.name, lowercase=True) or prefix
        candidate = prefix if base == prefix else f"{prefix}_{base}"
        suffix = 1
        while candidate in self._used:
            candidate = f"{prefix}_{base}_{suffix}"
            suffix += 1
        self._used.add(candidate)
        self._for_obj[obj] = candidate
        return candidate


# ---------------------------------------------------------------------------
# Constructor emitters — one per concrete metamodel class
# ---------------------------------------------------------------------------

def _quoted(value: str) -> str:
    """Single-quoted, escape-safe Python string literal for an arbitrary value."""
    return f"'{_escape_python_string(value or '')}'"


def _emit_layout_if_present(obj, var: str, body: list) -> None:
    """Emit ``<var>.layout = {...}`` when the object carries a non-empty layout dict.

    ``repr`` produces a valid Python literal for str / int / float / bool / None / dict /
    list payloads — exactly what the converter stashes (see
    ``json_to_buml/bpmn_diagram_processor._layout_dict``). Skip when ``layout`` is None
    or empty so a freshly-built model emits clean code.
    """
    if getattr(obj, "layout", None):
        body.append(f"{var}.layout = {repr(obj.layout)}")


def _emit_flow_node(node, container_var: str, dispenser: _NameDispenser,
                    body: list, needed: set) -> str:
    """Emit the constructor + ``add_flow_node`` call for one flow node.

    Returns the variable name. Adds the right metamodel symbols to ``needed`` so the
    final import block can list only what's used.
    """
    var = dispenser.name_for(node)

    if isinstance(node, AgenticTask):
        # AgenticTask is a Task subclass — emit it before the Task branch.
        needed.update({
            "AgenticTask", "TaskType", "LoopCharacteristics", "ReflectionMode",
        })
        ref_kwarg = ""
        if node.agent_diagram_ref is not None:
            ref_kwarg = f", agent_diagram_ref={_quoted(node.agent_diagram_ref)}"
        body.append(
            f"{var} = AgenticTask(name={_quoted(node.name)}, "
            f"task_type=TaskType.{node.task_type.name}, "
            f"loop_characteristics=LoopCharacteristics.{node.loop_characteristics.name}, "
            f"reflection_mode=ReflectionMode.{node.reflection_mode.name}, "
            f"trust_score={node.trust_score}{ref_kwarg})"
        )
    elif isinstance(node, Task):
        needed.update({"Task", "TaskType", "LoopCharacteristics"})
        body.append(
            f"{var} = Task(name={_quoted(node.name)}, "
            f"task_type=TaskType.{node.task_type.name}, "
            f"loop_characteristics=LoopCharacteristics.{node.loop_characteristics.name})"
        )
    elif isinstance(node, Transaction):
        # Transaction is a SubProcess subclass — emit it before the SubProcess branch.
        needed.update({"Transaction", "LoopCharacteristics"})
        body.append(
            f"{var} = Transaction(name={_quoted(node.name)}, "
            f"loop_characteristics=LoopCharacteristics.{node.loop_characteristics.name})"
        )
    elif isinstance(node, SubProcess):
        needed.update({"SubProcess", "LoopCharacteristics"})
        body.append(
            f"{var} = SubProcess(name={_quoted(node.name)}, "
            f"loop_characteristics=LoopCharacteristics.{node.loop_characteristics.name})"
        )
    elif isinstance(node, CallActivity):
        needed.update({"CallActivity", "LoopCharacteristics"})
        body.append(
            f"{var} = CallActivity(name={_quoted(node.name)}, "
            f"loop_characteristics=LoopCharacteristics.{node.loop_characteristics.name})"
        )
    elif isinstance(node, StartEvent):
        needed.update({"StartEvent", "EventDirection", "EventDefinitionType"})
        body.append(
            f"{var} = StartEvent(name={_quoted(node.name)}, "
            f"direction=EventDirection.{node.direction.name}, "
            f"event_definition=EventDefinitionType.{node.event_definition.name})"
        )
    elif isinstance(node, IntermediateEvent):
        needed.update({"IntermediateEvent", "EventDirection", "EventDefinitionType"})
        body.append(
            f"{var} = IntermediateEvent(name={_quoted(node.name)}, "
            f"direction=EventDirection.{node.direction.name}, "
            f"event_definition=EventDefinitionType.{node.event_definition.name})"
        )
    elif isinstance(node, EndEvent):
        needed.update({"EndEvent", "EventDirection", "EventDefinitionType"})
        body.append(
            f"{var} = EndEvent(name={_quoted(node.name)}, "
            f"direction=EventDirection.{node.direction.name}, "
            f"event_definition=EventDefinitionType.{node.event_definition.name})"
        )
    elif isinstance(node, AgenticGateway):
        # AgenticGateway is a Gateway subclass — emit it before the Gateway branch.
        needed.update({
            "AgenticGateway", "GatewayType", "GatewayRole",
        })
        # _quoted escapes newlines, so a multi-line governance DSL stays a valid
        # single-line literal that exec's back to the original string.
        gov_kwarg = ""
        if node.governance_dsl is not None:
            gov_kwarg = f", governance_dsl={_quoted(node.governance_dsl)}"
        body.append(
            f"{var} = AgenticGateway(name={_quoted(node.name)}, "
            f"gateway_type=GatewayType.{node.gateway_type.name}, "
            f"gateway_role=GatewayRole.{node.gateway_role.name}, "
            f"trust_score={node.trust_score}{gov_kwarg})"
        )
    elif isinstance(node, Gateway):
        needed.update({"Gateway", "GatewayType"})
        body.append(
            f"{var} = Gateway(name={_quoted(node.name)}, "
            f"gateway_type=GatewayType.{node.gateway_type.name})"
        )
    else:
        # Should not happen — every concrete FlowNode is covered above.
        raise TypeError(
            f"bpmn_model_to_code: unexpected FlowNode type {type(node).__name__}"
        )

    _emit_layout_if_present(node, var, body)
    body.append(f"{container_var}.add_flow_node({var})")
    return var


def _emit_artifact(artifact, process_var: str, dispenser: _NameDispenser,
                   body: list, needed: set) -> str:
    var = dispenser.name_for(artifact)
    if isinstance(artifact, TextAnnotation):
        needed.add("TextAnnotation")
        # WME / metamodel split: `text` holds the body; `name` stays empty.
        body.append(
            f"{var} = TextAnnotation(name='', text={_quoted(artifact.text)})"
        )
    elif isinstance(artifact, Group):
        needed.add("Group")
        body.append(f"{var} = Group(name={_quoted(artifact.name)})")
    else:
        raise TypeError(
            f"bpmn_model_to_code: unexpected Artifact type {type(artifact).__name__}"
        )
    _emit_layout_if_present(artifact, var, body)
    body.append(f"{process_var}.add_artifact({var})")
    return var


def _emit_data_object(data_object, process_var: str, dispenser: _NameDispenser,
                      body: list, needed: set) -> str:
    var = dispenser.name_for(data_object)
    needed.add("DataObject")
    body.append(f"{var} = DataObject(name={_quoted(data_object.name)})")
    _emit_layout_if_present(data_object, var, body)
    body.append(f"{process_var}.add_data_object({var})")
    return var


def _emit_lane(lane, process_var: str, dispenser: _NameDispenser,
               body: list, needed: set) -> str:
    """Emit a Lane constructor + ``add_lane`` + an ``add_flow_node`` per member."""
    var = dispenser.name_for(lane)
    if isinstance(lane, AgenticLane):
        needed.update({"AgenticLane", "AgentRole"})
        ref_kwarg = ""
        if lane.agent_diagram_ref is not None:
            ref_kwarg = f", agent_diagram_ref={_quoted(lane.agent_diagram_ref)}"
        mult_kwarg = ""
        if lane.multiplicity > 1:
            mult_kwarg = f", multiplicity={lane.multiplicity}"
        body.append(
            f"{var} = AgenticLane(name={_quoted(lane.name)}, "
            f"role=AgentRole.{lane.role.name}, "
            f"trust_score={lane.trust_score}{mult_kwarg}{ref_kwarg})"
        )
    else:
        needed.add("Lane")
        body.append(f"{var} = Lane(name={_quoted(lane.name)})")
    _emit_layout_if_present(lane, var, body)
    body.append(f"{process_var}.add_lane({var})")
    for member in sort_by_timestamp(lane.flow_nodes):
        member_var = dispenser.name_for(member)
        body.append(f"{var}.add_flow_node({member_var})")
    return var


def _emit_sequence_flow(flow: SequenceFlow, container_var: str,
                        dispenser: _NameDispenser, body: list, needed: set) -> None:
    needed.add("SequenceFlow")
    src = dispenser.name_for(flow.source)
    tgt = dispenser.name_for(flow.target)
    parts = [f"source={src}", f"target={tgt}"]
    if flow.name:
        parts.append(f"name={_quoted(flow.name)}")
    if flow.is_default:
        parts.append("is_default=True")
    body.append(f"{container_var}.add_sequence_flow(SequenceFlow({', '.join(parts)}))")
    _emit_flow_layout_if_present(flow, body)


def _emit_association(flow: Association, process_var: str,
                      dispenser: _NameDispenser, body: list, needed: set) -> None:
    needed.add("Association")
    src = dispenser.name_for(flow.source)
    tgt = dispenser.name_for(flow.target)
    parts = [f"source={src}", f"target={tgt}"]
    if flow.name:
        parts.append(f"name={_quoted(flow.name)}")
    body.append(f"{process_var}.add_association(Association({', '.join(parts)}))")
    _emit_flow_layout_if_present(flow, body)


def _emit_data_association(flow: DataAssociation, process_var: str,
                           dispenser: _NameDispenser, body: list, needed: set) -> None:
    needed.add("DataAssociation")
    src = dispenser.name_for(flow.source)
    tgt = dispenser.name_for(flow.target)
    parts = [f"source={src}", f"target={tgt}"]
    if flow.name:
        parts.append(f"name={_quoted(flow.name)}")
    body.append(
        f"{process_var}.add_data_association(DataAssociation({', '.join(parts)}))"
    )
    _emit_flow_layout_if_present(flow, body)


def _emit_message_flow(flow: MessageFlow, dispenser: _NameDispenser,
                       body: list, needed: set) -> None:
    src = dispenser.name_for(flow.source)
    tgt = dispenser.name_for(flow.target)
    parts = [f"source={src}", f"target={tgt}"]
    if flow.name:
        parts.append(f"name={_quoted(flow.name)}")
    needed.add("MessageFlow")
    body.append(f"collaboration.add_message_flow(MessageFlow({', '.join(parts)}))")
    _emit_flow_layout_if_present(flow, body)


def _emit_flow_layout_if_present(flow, body: list) -> None:
    """Connecting objects can't carry a follow-up ``.layout =`` line directly because
    they're constructed inline inside the ``add_*`` call. The layout is intentionally
    not preserved on flows in the .py round-trip (the converter recomputes it from
    bounds). Hook present in case we ever need to emit it under a temporary variable.
    """
    return  # no-op by design — see docstring


# ---------------------------------------------------------------------------
# Container walk
# ---------------------------------------------------------------------------

def _emit_container(container, container_var: str, dispenser: _NameDispenser,
                    body: list, needed: set) -> None:
    """Emit the flow nodes (recursing into sub-processes) and sequence flows of a
    Process or SubProcess."""
    for node in sort_by_timestamp(container.flow_nodes):
        _emit_flow_node(node, container_var, dispenser, body, needed)
        if isinstance(node, SubProcess):
            sub_var = dispenser.name_for(node)
            body.append("")
            body.append(f"# Sub-process contents: {node.name or '(unnamed)'}")
            _emit_container(node, sub_var, dispenser, body, needed)

    if container.sequence_flows:
        body.append("")
        body.append("# Sequence flows")
    for flow in sort_by_timestamp(container.sequence_flows):
        _emit_sequence_flow(flow, container_var, dispenser, body, needed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BANNER = "####################\n#    BPMN MODEL    #\n####################"


def bpmn_model_to_code(model: BPMNModel, file_path: str = None,
                       model_var_name: str = "bpmn_model") -> str:
    """Generate Python code that reconstructs ``model`` when ``exec()``'d.

    Args:
        model: The ``BPMNModel`` instance to serialise.
        file_path: Optional path to write the generated code to.
        model_var_name: Variable name to use for the top-level model
            (default: ``"bpmn_model"``).

    Returns:
        The generated Python source as a string.
    """
    body: list = []
    needed: set = {"BPMNModel"}
    dispenser = _NameDispenser()

    # 2. Empty model + (3) empty processes
    body.append(f"{model_var_name} = BPMNModel(name={_quoted(model.name)})")
    body.append("")

    process_vars: dict = {}  # process -> var
    for process in sort_by_timestamp(model.processes):
        needed.add("Process")
        var = dispenser.name_for(process)
        process_vars[process] = var
        body.append(f"# --- Process: {process.name or '(unnamed)'} ---")
        body.append(f"{var} = Process(name={_quoted(process.name)})")
        _emit_layout_if_present(process, var, body)
        body.append(f"{model_var_name}.add_process({var})")
        body.append("")

    # 4. Per-process contents
    for process in sort_by_timestamp(model.processes):
        var = process_vars[process]

        # 4a — flow nodes (recursing into sub-processes) + sequence flows
        if process.flow_nodes:
            body.append(f"# Flow nodes in: {process.name or '(unnamed)'}")
            _emit_container(process, var, dispenser, body, needed)
            body.append("")

        # 4b — artifacts (annotations / groups)
        if process.artifacts:
            body.append(f"# Artifacts in: {process.name or '(unnamed)'}")
            for artifact in sort_by_timestamp(process.artifacts):
                _emit_artifact(artifact, var, dispenser, body, needed)
            body.append("")

        # 4c — data objects
        if process.data_objects:
            body.append(f"# Data objects in: {process.name or '(unnamed)'}")
            for data_object in sort_by_timestamp(process.data_objects):
                _emit_data_object(data_object, var, dispenser, body, needed)
            body.append("")

        # 4d — lanes (after flow nodes so members are already emitted)
        if process.lanes:
            body.append(f"# Lanes in: {process.name or '(unnamed)'}")
            for lane in sort_by_timestamp(process.lanes):
                _emit_lane(lane, var, dispenser, body, needed)
            body.append("")

        # 4e — associations + data associations
        if process.associations:
            body.append(f"# Associations in: {process.name or '(unnamed)'}")
            for assoc in sort_by_timestamp(process.associations):
                _emit_association(assoc, var, dispenser, body, needed)
            body.append("")
        if process.data_associations:
            body.append(f"# Data associations in: {process.name or '(unnamed)'}")
            for da in sort_by_timestamp(process.data_associations):
                _emit_data_association(da, var, dispenser, body, needed)
            body.append("")

    # 5. Collaboration (must come AFTER processes — Participant references a Process)
    if model.collaboration is not None:
        needed.update({"Collaboration", "Participant"})
        body.append("# --- Collaboration ---")
        body.append(
            f"collaboration = Collaboration(name={_quoted(model.collaboration.name)})"
        )
        _emit_layout_if_present(model.collaboration, "collaboration", body)
        for participant in sort_by_timestamp(model.collaboration.participants):
            part_var = dispenser.name_for(participant)
            process_ref = (process_vars.get(participant.process)
                           if participant.process is not None else None)
            ctor_args = [f"name={_quoted(participant.name)}"]
            if process_ref:
                ctor_args.append(f"process={process_ref}")
            body.append(f"{part_var} = Participant({', '.join(ctor_args)})")
            _emit_layout_if_present(participant, part_var, body)
            body.append(f"collaboration.add_participant({part_var})")
        for mflow in sort_by_timestamp(model.collaboration.message_flows):
            _emit_message_flow(mflow, dispenser, body, needed)
        body.append(f"{model_var_name}.collaboration = collaboration")
        body.append("")

    # 6. Data stores — model-level
    if model.data_stores:
        needed.add("DataStore")
        body.append("# --- Data stores ---")
        for ds in sort_by_timestamp(model.data_stores):
            ds_var = dispenser.name_for(ds)
            body.append(f"{ds_var} = DataStore(name={_quoted(ds.name)})")
            _emit_layout_if_present(ds, ds_var, body)
            body.append(f"{model_var_name}.add_data_store({ds_var})")
        body.append("")

    # ---- Assemble: header → imports → body
    import_lines = _format_imports(needed)
    pieces = [
        _BANNER,
        "",
        import_lines,
        "",
    ] + body
    # Strip a single trailing blank line so the output ends with "\n" not "\n\n".
    while pieces and pieces[-1] == "":
        pieces.pop()
    result = "\n".join(pieces) + "\n"

    if file_path:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(result)

    return result


def _format_imports(needed: set) -> str:
    """Format the ``from besser.BUML.metamodel.bpmn import (...)`` block.

    Sorted alphabetically; one symbol per line for readability and for stable diffs.
    """
    if not needed:
        # Defensive — should never happen because BPMNModel is always added.
        return ""
    sorted_names = sorted(needed)
    inner = ",\n    ".join(sorted_names)
    return f"from besser.BUML.metamodel.bpmn import (\n    {inner},\n)"
