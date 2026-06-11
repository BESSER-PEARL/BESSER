"""BPMN 2.0 XML generator.

Emits vendor-neutral BPMN 2.0 XML (no ``<camunda:*>`` / ``<zeebe:*>`` /
``<flowable:*>`` extensions) from a ``BPMNModel``. The output is readable by every
BPMN-aware tool (Camunda 7 / 8, Flowable, jBPM, bpmn-js, Camunda Modeler,
SpiffWorkflow, Operaton, …).

Round-trip safe when the input model was built via
``process_bpmn_diagram`` — original WME ids ride in ``BPMNElement.layout`` and
are reused via :meth:`BPMNGenerator._id_for_obj` when they're NCName-valid.

See ``.claude/bpmn/05-bpmn-generator-guide.md`` for the design and the §11 known
limitations (multi-process ``dataStoreReference``, sub-process artifact hoisting).
"""

import os
import re
import xml.etree.ElementTree as ET

from besser.BUML.metamodel.bpmn import (
    Activity,
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    BPMNModel,
    CallActivity,
    Collaboration,
    DataAssociation,
    EndEvent,
    Event,
    EventDefinitionType,
    EventDirection,
    FlowNode,
    Gateway,
    GatewayType,
    Group,
    IntermediateEvent,
    LoopCharacteristics,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TaskType,
    TextAnnotation,
    Transaction,
)
from besser.generators import GeneratorInterface
from besser.utilities.utils import sort_by_timestamp


# ---------------------------------------------------------------------------
# XML namespaces — keys become the xmlns prefixes in the emitted file.
# ---------------------------------------------------------------------------

_NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
    # SEAA'25 agentic extension namespace -- verbatim from WME's
    # common/types.ts (AGENTIC_NS_URI). See `01-...` D8.
    "agentic": "https://www.besser-pearl.org/bpmn/agentic",
}

# BESSER-PEARL is the publishing org for files emitted by this generator.
# `xmlns:bpmn=<OMG URI>` is what makes this a BPMN file; `targetNamespace` is
# just the source identifier — see 05- guide §2.
_TARGET_NAMESPACE = "http://besser-pearl.org/bpmn"


# ---------------------------------------------------------------------------
# Dispatch tables (Python class → XML element name)
# ---------------------------------------------------------------------------

_TASK_TAG_FOR_TYPE = {
    TaskType.DEFAULT: "task",
    TaskType.USER: "userTask",
    TaskType.SERVICE: "serviceTask",
    TaskType.SEND: "sendTask",
    TaskType.RECEIVE: "receiveTask",
    TaskType.MANUAL: "manualTask",
    TaskType.BUSINESS_RULE: "businessRuleTask",
    TaskType.SCRIPT: "scriptTask",
}

_GATEWAY_TAG_FOR_TYPE = {
    GatewayType.EXCLUSIVE: "exclusiveGateway",
    GatewayType.INCLUSIVE: "inclusiveGateway",
    GatewayType.PARALLEL: "parallelGateway",
    GatewayType.COMPLEX: "complexGateway",
    GatewayType.EVENT_BASED: "eventBasedGateway",
}

_EVENT_DEFINITION_TAG = {
    EventDefinitionType.MESSAGE: "messageEventDefinition",
    EventDefinitionType.TIMER: "timerEventDefinition",
    EventDefinitionType.SIGNAL: "signalEventDefinition",
    EventDefinitionType.ESCALATION: "escalationEventDefinition",
    EventDefinitionType.ERROR: "errorEventDefinition",
    EventDefinitionType.COMPENSATION: "compensateEventDefinition",
    EventDefinitionType.LINK: "linkEventDefinition",
    EventDefinitionType.CONDITIONAL: "conditionalEventDefinition",
    EventDefinitionType.TERMINATE: "terminateEventDefinition",
}


# NCName validity — BPMN ids must start with letter or underscore, no spaces,
# allowed chars [A-Za-z0-9_-.].
_NCNAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _qname(prefix: str, local: str) -> str:
    """Build an ElementTree expanded-name (``{ns-uri}local``) for a namespaced tag."""
    return f"{{{_NS[prefix]}}}{local}"


def _emit_agentic_extension(host_el, obj) -> None:
    """Emit ``<bpmn:extensionElements><agentic:agentic .../></bpmn:extensionElements>``
    as the next child of ``host_el`` when ``obj`` is a SEAA'25 agentic subclass.

    Attribute presence + ordering mirror WME's ``emitAgenticExtension()``:

    * ``role`` -- iff ``AgenticLane``.
    * ``reflectionMode`` -- iff ``AgenticTask``.
    * ``gatewayRole`` -- iff ``AgenticGateway``.
    * ``trustScore`` -- on every agentic subclass.
    * ``agentDiagramRef`` -- iff carried + set (the agentic task, WME guide 11;
      or the legacy lane carrier). Never on gateways.

    A merging ``AgenticGateway`` with a ``governance_dsl`` additionally emits a
    sibling ``<agentic:governance>`` CDATA-style child (escaped text -- stdlib
    ET has no native CDATA, but the text content round-trips through any XML
    parser identically).

    No-op for non-agentic objects so callers can invoke it unconditionally.
    """
    if not isinstance(obj, (AgenticTask, AgenticGateway, AgenticLane)):
        return

    attrs: dict = {}
    if isinstance(obj, AgenticLane):
        attrs["role"] = obj.role.value
        # WME 3c: swarm size. Emit only when > 1 (absence = default 1),
        # matching WME's 04D2 exporter.
        if obj.multiplicity > 1:
            attrs["multiplicity"] = str(obj.multiplicity)
    if isinstance(obj, AgenticTask):
        attrs["reflectionMode"] = obj.reflection_mode.value
    if isinstance(obj, AgenticGateway):
        attrs["gatewayRole"] = obj.gateway_role.value
    attrs["trustScore"] = str(obj.trust_score)
    # agentDiagramRef rides whatever construct carries it (the agentic task,
    # WME guide 11; or the legacy lane), emitted only when set.
    ref = getattr(obj, "agent_diagram_ref", None)
    if ref is not None:
        attrs["agentDiagramRef"] = ref

    ext_el = ET.SubElement(host_el, _qname("bpmn", "extensionElements"))
    ET.SubElement(ext_el, _qname("agentic", "agentic"), attrib=attrs)
    # Governance DSL: sibling child of <agentic:agentic>,
    # merging gateways only, emitted when set.
    gov = getattr(obj, "governance_dsl", None)
    if isinstance(obj, AgenticGateway) and gov is not None and gov.strip() != "":
        ET.SubElement(ext_el, _qname("agentic", "governance")).text = gov


# ---------------------------------------------------------------------------
# BPMNGenerator
# ---------------------------------------------------------------------------

class BPMNGenerator(GeneratorInterface):
    """Generate a BPMN 2.0 XML file from a ``BPMNModel`` instance.

    Args:
        model (BPMNModel): The BPMN model to serialise.
        output_dir (str, optional): Directory for the ``.bpmn`` output file.
        file_name (str, optional): Output filename (default ``"bpmn_diagram.bpmn"``).
    """

    def __init__(self, model: BPMNModel, output_dir: str = None,
                 file_name: str = "bpmn_diagram.bpmn"):
        super().__init__(model, output_dir)
        self._file_name = file_name
        # Per-call id map + collision tracker. Reset on every ``generate()`` so
        # repeated calls on the same instance produce identical output.
        self._id_for: dict = {}
        self._used_ids: set = set()
        # Helper id counters for non-metamodel ids (LaneSet, DataObjectRef,
        # DataStoreRef, BPMNDiagram, BPMNPlane, Shape, Edge). Per-call.
        self._fresh_counter: dict = {}

    # --- public API ---------------------------------------------------------

    def generate(self) -> str:
        """Build the BPMN XML tree, write it to ``<output_dir>/<file_name>``.

        Returns:
            The absolute path of the written ``.bpmn`` file.
        """
        # Reset per-call state — `generate()` must be idempotent.
        self._id_for = {}
        self._used_ids = set()
        self._fresh_counter = {}

        output_dir = self.build_generation_dir()
        output_path = os.path.join(output_dir, self._file_name)

        # Register namespaces so ElementTree emits `bpmn:` etc. prefixes.
        for prefix, uri in _NS.items():
            ET.register_namespace(prefix, uri)

        definitions = ET.Element(
            _qname("bpmn", "definitions"),
            attrib={
                "id": "Definitions_1",
                "targetNamespace": _TARGET_NAMESPACE,
            },
        )

        # 1. Root-level data stores (BPMN 2.0.2 §10.3).
        for ds in sort_by_timestamp(self.model.data_stores):
            ET.SubElement(
                definitions, _qname("bpmn", "dataStore"),
                attrib={"id": self._id_for_obj(ds, "DataStore"),
                        "name": ds.name or ""},
            )

        # 2. One <bpmn:process> per Process.
        processes_sorted = sort_by_timestamp(self.model.processes)
        first_process = processes_sorted[0] if processes_sorted else None
        for process in processes_sorted:
            self._emit_process(definitions, process, is_first_process=(process is first_process))

        # 3. <bpmn:collaboration> if pool-ful.
        if self.model.collaboration is not None:
            self._emit_collaboration(definitions, self.model.collaboration)

        # 4. <bpmndi:BPMNDiagram> if any element carries layout.
        if self._any_layout_present():
            self._emit_diagram_interchange(definitions)

        # 5. Pretty-print and write.
        ET.indent(definitions, space="  ")
        tree = ET.ElementTree(definitions)
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)
        return output_path

    # --- id strategy --------------------------------------------------------

    def _id_for_obj(self, obj, prefix: str) -> str:
        """Return a stable BPMN-id for ``obj``.

        Reuses ``obj.layout["id"]`` when it's NCName-valid (so WME round-trips
        keep their original ids). Otherwise mints a deterministic counter-based
        id via :meth:`_fresh_id`. Resolves collisions by appending ``_N``.

        Determinism: walk order is timestamp-sorted, so the same model produces
        identical ids across calls (test ``test_same_model_same_bytes``).
        """
        if obj in self._id_for:
            return self._id_for[obj]
        stashed = (getattr(obj, "layout", None) or {}).get("id")
        if isinstance(stashed, str) and _NCNAME_RE.match(stashed):
            candidate = stashed
            base = candidate
            suffix = 1
            while candidate in self._used_ids:
                candidate = f"{base}_{suffix}"
                suffix += 1
            self._used_ids.add(candidate)
        else:
            candidate = self._fresh_id(prefix)
        self._id_for[obj] = candidate
        return candidate

    def _fresh_id(self, prefix: str) -> str:
        """Mint a deterministic id for an XML-only construct (LaneSet, Shape, …)."""
        n = self._fresh_counter.get(prefix, 0) + 1
        self._fresh_counter[prefix] = n
        candidate = f"{prefix}_{n}"
        while candidate in self._used_ids:
            n += 1
            self._fresh_counter[prefix] = n
            candidate = f"{prefix}_{n}"
        self._used_ids.add(candidate)
        return candidate

    # --- process / sub-process ---------------------------------------------

    def _emit_process(self, parent, process: Process, is_first_process: bool) -> None:
        process_el = ET.SubElement(
            parent, _qname("bpmn", "process"),
            attrib={"id": self._id_for_obj(process, "Process"),
                    "name": process.name or "",
                    "isExecutable": "false"},
        )

        # 2a. laneSet (if any).
        if process.lanes:
            lane_set = ET.SubElement(
                process_el, _qname("bpmn", "laneSet"),
                attrib={"id": self._fresh_id("LaneSet")},
            )
            for lane in sort_by_timestamp(process.lanes):
                lane_el = ET.SubElement(
                    lane_set, _qname("bpmn", "lane"),
                    attrib={"id": self._id_for_obj(lane, "Lane"),
                            "name": lane.name or ""},
                )
                # SEAA'25 agentic extensionElements — must precede
                # <flowNodeRef> per BPMN 2.0 tLane schema. No-op for base Lane.
                _emit_agentic_extension(lane_el, lane)
                for member in sort_by_timestamp(lane.flow_nodes):
                    ref = ET.SubElement(lane_el, _qname("bpmn", "flowNodeRef"))
                    ref.text = self._id_for_obj(member, type(member).__name__)

        # 2b. Flow nodes (recursing into sub-processes via _emit_flow_node).
        for node in sort_by_timestamp(process.flow_nodes):
            self._emit_flow_node(process_el, node)

        # 2c. Sequence flows in this process.
        for flow in sort_by_timestamp(process.sequence_flows):
            self._emit_sequence_flow(process_el, flow)

        # 2d. Data store references — emit only in the first process (§11.1).
        if is_first_process:
            for ds in sort_by_timestamp(self.model.data_stores):
                ET.SubElement(
                    process_el, _qname("bpmn", "dataStoreReference"),
                    attrib={"id": self._fresh_id("DataStoreRef"),
                            "name": ds.name or "",
                            "dataStoreRef": self._id_for_obj(ds, "DataStore")},
                )

        # 2e. Data objects + their reference siblings (per spec).
        for do in sort_by_timestamp(process.data_objects):
            do_id = self._id_for_obj(do, "DataObject")
            ET.SubElement(
                process_el, _qname("bpmn", "dataObject"),
                attrib={"id": do_id},
            )
            ET.SubElement(
                process_el, _qname("bpmn", "dataObjectReference"),
                attrib={"id": self._fresh_id("DataObjectRef"),
                        "name": do.name or "",
                        "dataObjectRef": do_id},
            )

        # 2f. Artifacts (annotations, groups).
        for artifact in sort_by_timestamp(process.artifacts):
            self._emit_artifact(process_el, artifact)

        # 2g. Associations.
        for assoc in sort_by_timestamp(process.associations):
            ET.SubElement(
                process_el, _qname("bpmn", "association"),
                attrib={"id": self._id_for_obj(assoc, "Association"),
                        "sourceRef": self._id_for_obj(
                            assoc.source, type(assoc.source).__name__),
                        "targetRef": self._id_for_obj(
                            assoc.target, type(assoc.target).__name__)},
            )

        # 2h. Data associations — direction inferred from endpoints (§3.5).
        for da in sort_by_timestamp(process.data_associations):
            self._emit_data_association(process_el, da)

    def _emit_flow_node(self, parent, node: FlowNode) -> None:
        # 1. Tag selection.
        if isinstance(node, Task):
            tag = _TASK_TAG_FOR_TYPE[node.task_type]
        elif isinstance(node, Transaction):
            # Subclass of SubProcess — check first so SubProcess branch doesn't catch it.
            tag = "transaction"
        elif isinstance(node, SubProcess):
            tag = "subProcess"
        elif isinstance(node, CallActivity):
            tag = "callActivity"
        elif isinstance(node, StartEvent):
            tag = "startEvent"
        elif isinstance(node, IntermediateEvent):
            tag = ("intermediateCatchEvent"
                   if node.direction is EventDirection.CATCH
                   else "intermediateThrowEvent")
        elif isinstance(node, EndEvent):
            tag = "endEvent"
        elif isinstance(node, Gateway):
            tag = _GATEWAY_TAG_FOR_TYPE[node.gateway_type]
        else:
            return  # defensive — every concrete FlowNode handled above.

        attrs = {"id": self._id_for_obj(node, type(node).__name__),
                 "name": node.name or ""}

        # 2. default= attribute on Activity / Gateway sources when default_flow is set.
        if isinstance(node, (Activity, Gateway)) and node.default_flow is not None:
            attrs["default"] = self._id_for_obj(node.default_flow, "Flow")

        el = ET.SubElement(parent, _qname("bpmn", tag), attrib=attrs)

        # 2b. SEAA'25 agentic extensionElements — must be the first child of
        #     any tFlowNode per the BPMN 2.0 schema. No-op for non-agentic.
        _emit_agentic_extension(el, node)

        # 3. <bpmn:incoming> / <bpmn:outgoing> children — required by Camunda
        #    Modeler / bpmn-js (§3.1). Sorted for deterministic output.
        for incoming in sort_by_timestamp(node.incoming()):
            ref = ET.SubElement(el, _qname("bpmn", "incoming"))
            ref.text = self._id_for_obj(incoming, "Flow")
        for outgoing in sort_by_timestamp(node.outgoing()):
            ref = ET.SubElement(el, _qname("bpmn", "outgoing"))
            ref.text = self._id_for_obj(outgoing, "Flow")

        # 4. Activity loop / multi-instance marker (§3.4).
        if isinstance(node, Activity):
            if node.loop_characteristics is LoopCharacteristics.STANDARD_LOOP:
                ET.SubElement(el, _qname("bpmn", "standardLoopCharacteristics"))
            elif node.loop_characteristics is LoopCharacteristics.PARALLEL_MI:
                ET.SubElement(
                    el, _qname("bpmn", "multiInstanceLoopCharacteristics"),
                    attrib={"isSequential": "false"},
                )
            elif node.loop_characteristics is LoopCharacteristics.SEQUENTIAL_MI:
                ET.SubElement(
                    el, _qname("bpmn", "multiInstanceLoopCharacteristics"),
                    attrib={"isSequential": "true"},
                )

        # 5. Event definition — nested child for non-NONE events (§3.3).
        if isinstance(node, Event) and node.event_definition is not EventDefinitionType.NONE:
            ET.SubElement(
                el, _qname("bpmn", _EVENT_DEFINITION_TAG[node.event_definition]),
            )

        # 6. Sub-process recursion — nest child flow nodes + sequence flows inside.
        if isinstance(node, SubProcess):
            for child in sort_by_timestamp(node.flow_nodes):
                self._emit_flow_node(el, child)
            for flow in sort_by_timestamp(node.sequence_flows):
                self._emit_sequence_flow(el, flow)

    def _emit_sequence_flow(self, parent, flow: SequenceFlow) -> None:
        attrs = {"id": self._id_for_obj(flow, "Flow"),
                 "sourceRef": self._id_for_obj(
                     flow.source, type(flow.source).__name__),
                 "targetRef": self._id_for_obj(
                     flow.target, type(flow.target).__name__)}
        if flow.name:
            attrs["name"] = flow.name
        ET.SubElement(parent, _qname("bpmn", "sequenceFlow"), attrib=attrs)

    def _emit_artifact(self, parent, artifact) -> None:
        if isinstance(artifact, TextAnnotation):
            ann_el = ET.SubElement(
                parent, _qname("bpmn", "textAnnotation"),
                attrib={"id": self._id_for_obj(artifact, "Annotation")},
            )
            text_el = ET.SubElement(ann_el, _qname("bpmn", "text"))
            text_el.text = artifact.text or ""
        elif isinstance(artifact, Group):
            ET.SubElement(
                parent, _qname("bpmn", "group"),
                attrib={"id": self._id_for_obj(artifact, "Group"),
                        "categoryValueRef": ""},  # placeholder — no category metamodel
            )

    def _emit_data_association(self, process_el, da: DataAssociation) -> None:
        """Emit a <bpmn:dataInputAssociation> or <bpmn:dataOutputAssociation>.

        Per spec: input is nested inside the consumer (target FlowNode); output
        inside the producer (source FlowNode). The metamodel guarantees exactly
        one DataElement + one FlowNode endpoint (E4), so the direction is
        unambiguous.
        """
        if isinstance(da.target, FlowNode):
            host_node = da.target
            other = da.source
            tag = "dataInputAssociation"
            ref_tag = "sourceRef"
        else:
            # source is the FlowNode, target is the DataElement
            host_node = da.source
            other = da.target
            tag = "dataOutputAssociation"
            ref_tag = "targetRef"

        host_id = self._id_for_obj(host_node, type(host_node).__name__)
        # Find the host's emitted element inside process_el by id.
        host_el = process_el.find(f".//*[@id='{host_id}']")
        if host_el is None:
            # Defensive — host wasn't emitted (e.g. lives in a sub-process the
            # association references across a boundary). Skip.
            return
        da_el = ET.SubElement(
            host_el, _qname("bpmn", tag),
            attrib={"id": self._id_for_obj(da, "DataAssoc")},
        )
        ref_el = ET.SubElement(da_el, _qname("bpmn", ref_tag))
        # `sourceRef`/`targetRef` here are *child elements* (per BPMN spec, not
        # attributes inside dataInput/OutputAssociation — different from
        # sequenceFlow's attribute form).
        ref_el.text = self._id_for_obj(other, type(other).__name__)

    # --- collaboration ------------------------------------------------------

    def _emit_collaboration(self, parent, collaboration: Collaboration) -> None:
        coll_el = ET.SubElement(
            parent, _qname("bpmn", "collaboration"),
            attrib={"id": self._id_for_obj(collaboration, "Collaboration"),
                    "name": collaboration.name or ""},
        )
        for participant in sort_by_timestamp(collaboration.participants):
            attrs = {"id": self._id_for_obj(participant, "Participant"),
                     "name": participant.name or ""}
            if participant.process is not None:
                attrs["processRef"] = self._id_for_obj(participant.process, "Process")
            ET.SubElement(coll_el, _qname("bpmn", "participant"), attrib=attrs)
        for mflow in sort_by_timestamp(collaboration.message_flows):
            attrs = {"id": self._id_for_obj(mflow, "MessageFlow"),
                     "sourceRef": self._id_for_obj(
                         mflow.source, type(mflow.source).__name__),
                     "targetRef": self._id_for_obj(
                         mflow.target, type(mflow.target).__name__)}
            if mflow.name:
                attrs["name"] = mflow.name
            mf_el = ET.SubElement(coll_el, _qname("bpmn", "messageFlow"), attrib=attrs)
            # SEAA'25 agentic extensionElements -- first child of tMessageFlow per
            # the BPMN 2.0 schema. No-op for a non-agentic MessageFlow.
            _emit_agentic_extension(mf_el, mflow)

    # --- diagram interchange ----------------------------------------------

    def _any_layout_present(self) -> bool:
        for node in self.model.all_flow_nodes():
            if getattr(node, "layout", None):
                return True
        for flow in self.model.all_connecting_objects():
            if getattr(flow, "layout", None):
                return True
        for process in self.model.processes:
            for collection in (process.artifacts, process.data_objects, process.lanes):
                for obj in collection:
                    if getattr(obj, "layout", None):
                        return True
        if self.model.collaboration is not None:
            for participant in self.model.collaboration.participants:
                if getattr(participant, "layout", None):
                    return True
        return False

    def _emit_diagram_interchange(self, parent) -> None:
        diagram = ET.SubElement(
            parent, _qname("bpmndi", "BPMNDiagram"),
            attrib={"id": self._fresh_id("BPMNDiagram")},
        )
        if self.model.collaboration is not None:
            plane_element_id = self._id_for_obj(self.model.collaboration, "Collaboration")
        else:
            only_process = next(iter(sort_by_timestamp(self.model.processes)), None)
            plane_element_id = (self._id_for_obj(only_process, "Process")
                                if only_process is not None else "")
        plane = ET.SubElement(
            diagram, _qname("bpmndi", "BPMNPlane"),
            attrib={"id": self._fresh_id("BPMNPlane"),
                    "bpmnElement": plane_element_id},
        )

        # One BPMNShape per element with bounds.
        for obj in self._shape_iter():
            bounds = (getattr(obj, "layout", None) or {}).get("bounds")
            if not bounds:
                continue
            shape = ET.SubElement(
                plane, _qname("bpmndi", "BPMNShape"),
                attrib={"id": self._fresh_id("Shape"),
                        "bpmnElement": self._id_for_obj(obj, type(obj).__name__)},
            )
            ET.SubElement(
                shape, _qname("dc", "Bounds"),
                attrib={"x": str(bounds.get("x", 0)),
                        "y": str(bounds.get("y", 0)),
                        "width": str(bounds.get("width", 0)),
                        "height": str(bounds.get("height", 0))},
            )

        # One BPMNEdge per connecting object with a path.
        for flow in sort_by_timestamp(self.model.all_connecting_objects()):
            path = (getattr(flow, "layout", None) or {}).get("path")
            if not path:
                continue
            edge = ET.SubElement(
                plane, _qname("bpmndi", "BPMNEdge"),
                attrib={"id": self._fresh_id("Edge"),
                        "bpmnElement": self._id_for_obj(flow, type(flow).__name__)},
            )
            for point in path:
                ET.SubElement(
                    edge, _qname("di", "waypoint"),
                    attrib={"x": str(point.get("x", 0)),
                            "y": str(point.get("y", 0))},
                )

    def _shape_iter(self):
        """Yield every element that may carry on-canvas bounds, in deterministic order."""
        if self.model.collaboration is not None:
            for part in sort_by_timestamp(self.model.collaboration.participants):
                yield part
        for process in sort_by_timestamp(self.model.processes):
            for lane in sort_by_timestamp(process.lanes):
                yield lane
        for node in sort_by_timestamp(self.model.all_flow_nodes()):
            yield node
        for process in sort_by_timestamp(self.model.processes):
            for do in sort_by_timestamp(process.data_objects):
                yield do
            for artifact in sort_by_timestamp(process.artifacts):
                yield artifact
        for ds in sort_by_timestamp(self.model.data_stores):
            yield ds
