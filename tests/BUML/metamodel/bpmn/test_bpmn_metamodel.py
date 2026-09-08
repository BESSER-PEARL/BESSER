"""Tests for the BPMN metamodel (``besser.BUML.metamodel.bpmn``).

Five groups, following ``.claude/bpmn/02-bpmn-metamodel-implementation-guide.md`` §10:
    1. Construction validation -- setters raise on bad input; free-text names are accepted.
    2. Invariants -- the derived ``default_flow``.
    3. ``BPMNModel.validate()`` -- rules E1-E11 and warnings W1-W4.
    4. Identity -- BPMN elements use object identity (no ``__eq__`` / ``__hash__``).
    5. ``Project`` integration.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    Activity, Artifact, Association, BPMNConnectingObject, BPMNElement, BPMNModel,
    CallActivity, Collaboration, DataAssociation, DataElement, DataObject, DataStore,
    EndEvent, Event, EventDefinitionType, EventDirection, FlowNode, Gateway, GatewayType,
    Group, IntermediateEvent, Lane, MessageFlow, Participant, Process, SequenceFlow,
    StartEvent, SubProcess, Task, TextAnnotation, Transaction,
)
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import Model


# ---------------------------------------------------------------------------
# Group 1 -- construction validation
# ---------------------------------------------------------------------------

def test_free_text_names_are_accepted():
    """BPMN labels are free text: spaces, empty, and None (-> "") must not raise (D5)."""
    assert Task("Place Order").name == "Place Order"
    assert Task("Review & approve").name == "Review & approve"
    assert Task("").name == ""
    assert Task().name == ""
    assert Task(None).name == ""
    assert BPMNModel("My BPMN Diagram").name == "My BPMN Diagram"


def test_name_must_be_str_or_none():
    with pytest.raises(TypeError):
        Task(123)
    with pytest.raises(TypeError):
        BPMNModel(123)


def test_abstract_classes_cannot_be_instantiated():
    for abstract_cls in (FlowNode, Activity, Event, Artifact, DataElement):
        with pytest.raises(TypeError):
            abstract_cls("x")
    with pytest.raises(TypeError):
        # BPMNConnectingObject needs source/target; the abstract guard fires first.
        BPMNConnectingObject(Task(), Task())


def test_concrete_subtypes_instantiate():
    assert isinstance(Transaction(), SubProcess)
    assert isinstance(CallActivity(), Activity)
    assert isinstance(Group(), BPMNElement)
    assert isinstance(DataObject(), DataElement)
    assert isinstance(DataStore(), DataElement)


def test_enum_attributes_are_type_checked():
    with pytest.raises(TypeError):
        Task("t", task_type="user")
    with pytest.raises(TypeError):
        Gateway("g", gateway_type="exclusive")
    with pytest.raises(TypeError):
        Task("t", loop_characteristics="loop")


def test_layout_must_be_dict_or_none():
    assert Task("t", layout={"x": 1}).layout == {"x": 1}
    assert Task("t").layout is None
    with pytest.raises(TypeError):
        Task("t", layout=[1, 2])


def test_sequence_flow_endpoints_must_be_flow_nodes():
    task = Task("t")
    data = DataObject("d")
    with pytest.raises(TypeError):
        SequenceFlow(task, data)
    with pytest.raises(TypeError):
        SequenceFlow(data, task)


def test_message_flow_endpoints_must_be_message_eligible():
    task = Task("t")
    gateway = Gateway("g")
    data = DataObject("d")
    assert MessageFlow(task, StartEvent("s"))            # Activity / Event ok
    with pytest.raises(TypeError):
        MessageFlow(task, gateway)                       # gateways are not message-eligible
    with pytest.raises(TypeError):
        MessageFlow(task, data)


def test_association_needs_an_artifact_endpoint():
    annotation = TextAnnotation(text="note")
    task = Task("t")
    assert Association(annotation, task)
    assert Association(task, annotation)
    with pytest.raises(ValueError):
        Association(task, Task("t2"))


def test_data_association_needs_one_data_and_one_node():
    data = DataObject("d")
    task = Task("t")
    assert DataAssociation(data, task)
    assert DataAssociation(task, data)
    with pytest.raises(ValueError):
        DataAssociation(task, Task("t2"))                # two flow nodes
    with pytest.raises(ValueError):
        DataAssociation(data, DataObject("d2"))          # two data elements


def test_is_default_only_on_eligible_source():
    activity = Task("a")
    target = Task("b")
    # Activity source -- ok.
    assert SequenceFlow(activity, target, is_default=True).is_default is True
    # Exclusive / inclusive / complex gateways -- ok.
    for gw_type in (GatewayType.EXCLUSIVE, GatewayType.INCLUSIVE, GatewayType.COMPLEX):
        gateway = Gateway("g", gateway_type=gw_type)
        assert SequenceFlow(gateway, Task("x"), is_default=True).is_default is True
    # Parallel / event-based gateways and events -- rejected.
    for gw_type in (GatewayType.PARALLEL, GatewayType.EVENT_BASED):
        gateway = Gateway("g", gateway_type=gw_type)
        with pytest.raises(ValueError):
            SequenceFlow(gateway, Task("x"), is_default=True)
    with pytest.raises(ValueError):
        SequenceFlow(StartEvent("s"), Task("x"), is_default=True)


def test_is_default_must_be_bool():
    with pytest.raises(TypeError):
        SequenceFlow(Task("a"), Task("b"), is_default="yes")


def test_start_event_direction_is_catch_only():
    assert StartEvent("s").direction is EventDirection.CATCH
    with pytest.raises(ValueError):
        StartEvent("s", direction=EventDirection.THROW)


def test_end_event_direction_is_throw_only():
    assert EndEvent("e").direction is EventDirection.THROW
    with pytest.raises(ValueError):
        EndEvent("e", direction=EventDirection.CATCH)


def test_intermediate_event_direction_is_free():
    assert IntermediateEvent("i").direction is EventDirection.CATCH
    catch = IntermediateEvent("i", direction=EventDirection.CATCH,
                              event_definition=EventDefinitionType.MESSAGE)
    throw = IntermediateEvent("i", direction=EventDirection.THROW,
                              event_definition=EventDefinitionType.MESSAGE)
    assert catch.direction is EventDirection.CATCH
    assert throw.direction is EventDirection.THROW


def test_illegal_event_definition_is_rejected():
    # TERMINATE is end-only.
    with pytest.raises(ValueError):
        StartEvent("s", event_definition=EventDefinitionType.TERMINATE)
    # TIMER is not a legal end-event definition.
    with pytest.raises(ValueError):
        EndEvent("e", event_definition=EventDefinitionType.TIMER)
    # An intermediate throw event has no "none" form in the WME subset.
    with pytest.raises(ValueError):
        IntermediateEvent("i", direction=EventDirection.THROW)
    # CONDITIONAL is catch-only for intermediate events.
    with pytest.raises(ValueError):
        IntermediateEvent("i", direction=EventDirection.THROW,
                          event_definition=EventDefinitionType.CONDITIONAL)


def test_event_definition_must_be_enum():
    with pytest.raises(TypeError):
        StartEvent("s", event_definition="message")
    # IntermediateEvent's direction setter is the plain Event one, so a non-enum direction
    # fails the type check (StartEvent/EndEvent override it to a CATCH/THROW ValueError).
    with pytest.raises(TypeError):
        IntermediateEvent("i", direction="catch")


def test_container_and_lane_back_references():
    start, task = StartEvent("s"), Task("t")
    process = Process("p", flow_nodes={start, task})
    assert task.container is process
    assert start.container is process
    lane = Lane("lane", flow_nodes={task})
    assert task.lane is lane
    # Removing clears the back-reference.
    process.remove_flow_node(task)
    assert task.container is None
    lane.remove_flow_node(task)
    assert task.lane is None


def test_lane_rejects_node_already_in_another_lane():
    task = Task("t")
    Lane("lane_1", flow_nodes={task})
    with pytest.raises(ValueError):
        Lane("lane_2").add_flow_node(task)


def test_collection_setters_are_type_checked():
    with pytest.raises(TypeError):
        Process("p", flow_nodes={Task("t"), "not a node"})
    with pytest.raises(TypeError):
        Process("p", sequence_flows={"not a flow"})
    with pytest.raises(TypeError):
        BPMNModel("m", processes={"not a process"})
    with pytest.raises(TypeError):
        Collaboration("c", participants={"not a participant"})


# ---------------------------------------------------------------------------
# Group 2 -- the derived default_flow invariant
# ---------------------------------------------------------------------------

def test_default_flow_is_derived_from_is_default():
    gateway = Gateway("g", gateway_type=GatewayType.EXCLUSIVE)
    a, b = Task("a"), Task("b")
    default = SequenceFlow(gateway, a, is_default=True)
    other = SequenceFlow(gateway, b)
    Process("p", flow_nodes={gateway, a, b}, sequence_flows={default, other})
    assert gateway.default_flow is default


def test_default_flow_is_none_without_a_default():
    activity = Task("a")
    target = Task("b")
    Process("p", flow_nodes={activity, target}, sequence_flows={SequenceFlow(activity, target)})
    assert activity.default_flow is None


def test_default_flow_tracks_is_default_changes():
    activity = Task("a")
    target = Task("b")
    flow = SequenceFlow(activity, target)
    Process("p", flow_nodes={activity, target}, sequence_flows={flow})
    assert activity.default_flow is None
    flow.is_default = True
    assert activity.default_flow is flow
    flow.is_default = False
    assert activity.default_flow is None


def test_parallel_gateway_default_flow_is_always_none():
    gateway = Gateway("g", gateway_type=GatewayType.PARALLEL)
    target = Task("t")
    Process("p", flow_nodes={gateway, target}, sequence_flows={SequenceFlow(gateway, target)})
    assert gateway.default_flow is None


# ---------------------------------------------------------------------------
# Group 3 -- BPMNModel.validate()
# ---------------------------------------------------------------------------

def test_fixtures_validate_clean(minimal_process_model, collaboration_model, gateway_model):
    for model in (minimal_process_model, collaboration_model, gateway_model):
        result = model.validate(raise_exception=False)
        assert result["success"] is True, result["errors"]
        assert result["errors"] == []
        assert result["warnings"] == []


def test_validate_raises_when_requested():
    t1, t2 = Task("t1"), Task("t2")
    p1 = Process("p1", flow_nodes={t1}, sequence_flows={SequenceFlow(t1, t2)})
    p2 = Process("p2", flow_nodes={t2})
    model = BPMNModel("m", processes={p1, p2})
    with pytest.raises(ValueError):
        model.validate(raise_exception=True)
    # raise_exception=False returns the dict instead.
    assert model.validate(raise_exception=False)["success"] is False


def test_e1_dangling_endpoint_reference():
    orphan = Task("orphan")            # never added to any process
    task = Task("task")
    process = Process("p", flow_nodes={task}, sequence_flows={SequenceFlow(orphan, task)})
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("not present in the model" in e for e in errors)


def test_e2_sequence_flow_crosses_container_boundary():
    t1, t2 = Task("t1"), Task("t2")
    p1 = Process("p1", flow_nodes={t1}, sequence_flows={SequenceFlow(t1, t2)})
    p2 = Process("p2", flow_nodes={t2})
    model = BPMNModel("m", processes={p1, p2})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("crosses a process" in e for e in errors)


def test_e3_message_flow_within_a_single_pool():
    t1, t2 = Task("t1"), Task("t2")
    process = Process("p", flow_nodes={t1, t2})
    participant = Participant("Pool", process=process)
    message = MessageFlow(t1, t2)
    collaboration = Collaboration("c", participants={participant}, message_flows={message})
    model = BPMNModel("m", processes={process}, collaboration=collaboration)
    errors = model.validate(raise_exception=False)["errors"]
    assert any("same pool" in e for e in errors)


def test_e4_data_association_endpoint_composition():
    data, task = DataObject("d"), Task("t")
    data_assoc = DataAssociation(data, task)
    # Mutate via the (intentionally lenient) endpoint setter to break the pair invariant.
    data_assoc.source = Task("t2")     # now both ends are FlowNodes
    process = Process("p", flow_nodes={task}, data_objects={data},
                      data_associations={data_assoc})
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("exactly one DataElement" in e for e in errors)


def test_e5_association_needs_an_artifact_endpoint():
    annotation, task = TextAnnotation(text="note"), Task("t")
    association = Association(annotation, task)
    association.source = Task("t2")    # now neither end is an Artifact
    process = Process("p", flow_nodes={task}, artifacts={annotation},
                      associations={association})
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("at least one Artifact" in e for e in errors)


def test_e6_default_flow_on_ineligible_source():
    activity, target = Task("a"), Task("b")
    flow = SequenceFlow(activity, target, is_default=True)
    parallel = Gateway("par", gateway_type=GatewayType.PARALLEL)
    # The SequenceFlow endpoint setter accepts any FlowNode, so the source can be swapped
    # to an ineligible one while is_default stays True -- exactly what E6 guards.
    flow.source = parallel
    process = Process("p", flow_nodes={activity, target, parallel}, sequence_flows={flow})
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("cannot carry a default flow" in e for e in errors)


def test_e7_at_most_one_default_per_source():
    gateway = Gateway("g", gateway_type=GatewayType.EXCLUSIVE)
    a, b = Task("a"), Task("b")
    process = Process(
        "p",
        flow_nodes={gateway, a, b},
        sequence_flows={
            SequenceFlow(gateway, a, is_default=True),
            SequenceFlow(gateway, b, is_default=True),
        },
    )
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("outgoing default" in e for e in errors)


def test_e8_event_boundary_rules():
    start, task, end = StartEvent("s"), Task("t"), EndEvent("e")
    # A flow targeting a start event and a flow leaving an end event.
    process = Process(
        "p",
        flow_nodes={start, task, end},
        sequence_flows={SequenceFlow(task, start), SequenceFlow(end, task)},
    )
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("targets StartEvent" in e for e in errors)
    assert any("originates from EndEvent" in e for e in errors)


def test_e9_event_definition_safety_net():
    # Construction enforces the legality table; this simulates a direct mutation that
    # bypasses the setter, which validate() must still catch.
    event = IntermediateEvent("i", direction=EventDirection.THROW,
                              event_definition=EventDefinitionType.MESSAGE)
    event._Event__event_definition = EventDefinitionType.CONDITIONAL  # illegal for throw
    process = Process("p", flow_nodes={event})
    model = BPMNModel("m", processes={process})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("illegal event_definition" in e for e in errors)


def test_e10_lane_membership():
    task = Task("t")
    p1 = Process("p1", flow_nodes={task})          # task.container is p1
    lane = Lane("lane", flow_nodes={task})         # task.lane is lane
    p2 = Process("p2", lanes={lane})               # but the lane lives in p2
    model = BPMNModel("m", processes={p1, p2})
    errors = model.validate(raise_exception=False)["errors"]
    assert any("not contained by that process" in e for e in errors)


def test_e11_participant_process_must_be_in_model():
    process = Process("p", flow_nodes={Task("t")})
    participant = Participant("Pool", process=process)
    collaboration = Collaboration("c", participants={participant})
    # The participant's process is NOT in model.processes.
    model = BPMNModel("m", processes=set(), collaboration=collaboration)
    errors = model.validate(raise_exception=False)["errors"]
    assert any("not in the model" in e for e in errors)


def test_w1_process_with_no_start_event(minimal_process_model):
    process = Process("p", flow_nodes={Task("only a task")})
    model = BPMNModel("m", processes={process})
    warnings = model.validate(raise_exception=False)["warnings"]
    assert any("no start event" in w for w in warnings)


def test_w2_unreachable_flow_node():
    process = Process("p", flow_nodes={StartEvent("s"), Task("orphan")})
    model = BPMNModel("m", processes={process})
    warnings = model.validate(raise_exception=False)["warnings"]
    assert any("unreachable" in w for w in warnings)


def test_w3_default_flow_with_single_outgoing():
    activity, target = Task("a"), Task("b")
    process = Process("p", flow_nodes={activity, target},
                      sequence_flows={SequenceFlow(activity, target, is_default=True)})
    model = BPMNModel("m", processes={process})
    warnings = model.validate(raise_exception=False)["warnings"]
    assert any("default flow but only" in w for w in warnings)


def test_w4_empty_model_and_empty_process():
    empty_model = BPMNModel("empty")
    warnings = empty_model.validate(raise_exception=False)["warnings"]
    assert any("no processes" in w for w in warnings)

    model_with_empty_process = BPMNModel("m", processes={Process("p")})
    warnings = model_with_empty_process.validate(raise_exception=False)["warnings"]
    assert any("no flow nodes" in w for w in warnings)


def test_validate_handles_nested_subprocess():
    """A SubProcess is itself a container; its children are reached by the accessors."""
    inner_start = StartEvent("inner start")
    inner_task = Task("inner task")
    sub = SubProcess(
        "sub",
        flow_nodes={inner_start, inner_task},
        sequence_flows={SequenceFlow(inner_start, inner_task)},
    )
    outer_start, outer_end = StartEvent("outer start"), EndEvent("outer end")
    process = Process(
        "p",
        flow_nodes={outer_start, sub, outer_end},
        sequence_flows={SequenceFlow(outer_start, sub), SequenceFlow(sub, outer_end)},
    )
    model = BPMNModel("m", processes={process})
    assert inner_task in model.all_flow_nodes()
    assert model.validate(raise_exception=False)["success"] is True


# ---------------------------------------------------------------------------
# Group 4 -- identity
# ---------------------------------------------------------------------------

def test_bpmn_elements_use_object_identity():
    # Two tasks with the same (or empty) name are distinct set members.
    assert len({Task("same"), Task("same")}) == 2
    assert len({Task(), Task()}) == 2
    task = Task("t")
    assert task == task
    assert task != Task("t")


# ---------------------------------------------------------------------------
# Group 5 -- Project integration
# ---------------------------------------------------------------------------

def test_bpmn_model_is_a_model(minimal_process_model):
    assert isinstance(minimal_process_model, Model)


def test_bpmn_model_fits_in_a_project(minimal_process_model):
    project = Project("project", models=[minimal_process_model])
    assert minimal_process_model in project.models
