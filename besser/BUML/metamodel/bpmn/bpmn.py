"""BPMN metamodel for B-UML.

A first-class B-UML model for BPMN diagrams, alongside ``structural`` / ``state_machine`` /
``gui``. Implements the design in ``.claude/bpmn/01-bpmn-metamodel-design.md`` (reviewed and
locked) following the build plan in ``.claude/bpmn/02-bpmn-metamodel-implementation-guide.md``.

Hierarchy::

    Element -> NamedElement -> BPMNElement -> {FlowNode, BPMNConnectingObject, Artifact,
                                               DataElement, Lane, Participant,
                                               Collaboration, Process}
    NamedElement -> Model -> BPMNModel

Growth path (01-... §3.5) -- constructs designed but intentionally NOT implemented yet.
Each becomes a plain attribute when pulled into scope; no restructuring is needed:

* ``SubProcess.is_expanded`` / ``is_event_subprocess`` / ``is_ad_hoc`` / ``triggered_by_event``
* ``Transaction.method`` ; ``CallActivity.called_element``
* ``DataObject.is_collection`` + ItemAware typing ; ``DataStore.data_store_ref``
* ``Group.category_value`` ; ``SequenceFlow.condition_expression``
* multiple ``EventDefinition`` s / parallel-multiple events

Note: ``Process`` and ``SubProcess`` both expose ``flow_nodes`` / ``sequence_flows`` (rather
than the ``children_*`` names sketched in 01-... §2.2), so the two are duck-type compatible
as flow-element containers -- ``FlowNode.outgoing()`` / ``incoming()`` and ``BPMNModel``'s
accessors stay free of attribute-name branching. Containment semantics are unchanged.
"""

from enum import Enum

from besser.BUML.metamodel.structural import Model, NamedElement


# ---------------------------------------------------------------------------
# Enumerations (decision D7 -- plain enum.Enum; .value strings match the WME enums)
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """The kind of a BPMN Task (BPMN 2.0.2 §10.2.3)."""
    DEFAULT = "default"
    USER = "user"
    SERVICE = "service"
    SEND = "send"
    RECEIVE = "receive"
    MANUAL = "manual"
    BUSINESS_RULE = "business-rule"
    SCRIPT = "script"


class GatewayType(Enum):
    """The kind of a BPMN Gateway (BPMN 2.0.2 §10.5)."""
    EXCLUSIVE = "exclusive"
    INCLUSIVE = "inclusive"
    PARALLEL = "parallel"
    COMPLEX = "complex"
    EVENT_BASED = "event-based"


class LoopCharacteristics(Enum):
    """The loop / multi-instance marker of an Activity (BPMN 2.0.2 §10.2.8)."""
    NONE = "none"
    STANDARD_LOOP = "loop"
    PARALLEL_MI = "parallel multi instance"
    SEQUENTIAL_MI = "sequential multi instance"


class EventDirection(Enum):
    """Whether an Event catches or throws its trigger (BPMN 2.0.2 §10.4)."""
    CATCH = "catch"
    THROW = "throw"


class EventDefinitionType(Enum):
    """The trigger kind of an Event (BPMN 2.0.2 §10.4.5). ``NONE`` is a plain/none event."""
    NONE = "none"
    MESSAGE = "message"
    TIMER = "timer"
    SIGNAL = "signal"
    CONDITIONAL = "conditional"
    ESCALATION = "escalation"
    ERROR = "error"
    COMPENSATION = "compensation"
    LINK = "link"
    TERMINATE = "terminate"


# (EventClassName, EventDirection) -> frozenset of legal EventDefinitionType.
# Derived from the WME event enums (01-... §3.2) -- WME-subset (decision D1).
_LEGAL_EVENT_DEFINITIONS = {
    ("StartEvent", EventDirection.CATCH): frozenset({
        EventDefinitionType.NONE, EventDefinitionType.MESSAGE, EventDefinitionType.TIMER,
        EventDefinitionType.CONDITIONAL, EventDefinitionType.SIGNAL,
        EventDefinitionType.ESCALATION, EventDefinitionType.ERROR,
        EventDefinitionType.COMPENSATION, EventDefinitionType.LINK,
    }),
    ("EndEvent", EventDirection.THROW): frozenset({
        EventDefinitionType.NONE, EventDefinitionType.MESSAGE, EventDefinitionType.ESCALATION,
        EventDefinitionType.ERROR, EventDefinitionType.COMPENSATION,
        EventDefinitionType.SIGNAL, EventDefinitionType.TERMINATE,
    }),
    ("IntermediateEvent", EventDirection.CATCH): frozenset({
        EventDefinitionType.NONE, EventDefinitionType.MESSAGE, EventDefinitionType.TIMER,
        EventDefinitionType.CONDITIONAL, EventDefinitionType.LINK, EventDefinitionType.SIGNAL,
    }),
    ("IntermediateEvent", EventDirection.THROW): frozenset({
        EventDefinitionType.MESSAGE, EventDefinitionType.TIMER, EventDefinitionType.ESCALATION,
        EventDefinitionType.LINK, EventDefinitionType.COMPENSATION, EventDefinitionType.SIGNAL,
    }),
}


def _legal_event_definitions(event):
    """Resolve the legal ``EventDefinitionType`` set for ``event`` by its base event
    class (via ``isinstance``) and direction, or ``None`` when no rule applies.

    Keying on the class hierarchy rather than ``type(event).__name__`` means a subclass
    of ``StartEvent`` / ``EndEvent`` / ``IntermediateEvent`` is still validated against
    its parent's table instead of being silently skipped.
    """
    for base in (StartEvent, EndEvent, IntermediateEvent):
        if isinstance(event, base):
            return _LEGAL_EVENT_DEFINITIONS.get((base.__name__, event.direction))
    return None


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _checked_set(values, expected_type, label: str) -> set:
    """Coerce ``values`` to a set, raising TypeError if any element is not ``expected_type``."""
    result = set(values)
    for value in result:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{label} must contain {expected_type.__name__} instances, "
                f"got {type(value).__name__}"
            )
    return result


def _source_can_carry_default(source) -> bool:
    """BPMN 2.0.2 §8.3.13: a default sequence flow's source must be an Activity or an
    exclusive / inclusive / complex Gateway."""
    if isinstance(source, Activity):
        return True
    if isinstance(source, Gateway):
        return source.gateway_type in {GatewayType.EXCLUSIVE, GatewayType.INCLUSIVE,
                                       GatewayType.COMPLEX}
    return False


def _default_outgoing_flow(node):
    """The node's outgoing SequenceFlow marked ``is_default``, if any (BPMN 2.0.2 §8.3.13)."""
    return next((flow for flow in node.outgoing() if flow.is_default), None)


# ---------------------------------------------------------------------------
# Base element
# ---------------------------------------------------------------------------

class BPMNElement(NamedElement):
    """Base class for every BPMN abstract-syntax element.

    Relaxes ``NamedElement.name`` (decision D5): a BPMN label is free text and is very often
    empty (gateways, events, flows). Also carries ``layout`` -- an opaque diagram-interchange
    passthrough the metamodel never interprets (decision D8); only the converters read it.

    Args:
        name (str): The element label. Empty allowed; ``None`` is coerced to ``""``.
        layout (dict): Opaque DI data (bounds, waypoints, ...). Stored untouched.
        metadata (Metadata): Inherited from NamedElement.
        timestamp (datetime): Inherited from NamedElement.

    Attributes:
        name (str): The element label (free text; may be empty).
        layout (dict): Opaque DI passthrough, or None.
    """

    def __init__(self, name: str = "", layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.layout = layout

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the BPMN label. Permits empty / whitespace / spaces; ``None`` -> ``""``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"BPMN element name must be a str or None, got {type(name).__name__}"
            )
        # Bypass NamedElement's strict identifier validation by writing its private slot.
        self._NamedElement__name = name

    @property
    def layout(self) -> dict:
        """dict: Get the opaque DI passthrough, or None."""
        return self.__layout

    @layout.setter
    def layout(self, layout: dict):
        """dict: Set the opaque DI passthrough.

        Raises:
            TypeError: if layout is neither a dict nor None.
        """
        if layout is not None and not isinstance(layout, dict):
            raise TypeError(f"layout must be a dict or None, got {type(layout).__name__}")
        self.__layout = layout

    def __repr__(self):
        return f"{type(self).__name__}(name='{self.name}')"


# ---------------------------------------------------------------------------
# Flow nodes
# ---------------------------------------------------------------------------

class FlowNode(BPMNElement):
    """Abstract base for anything a SequenceFlow can connect (BPMN 2.0.2 FlowNode).

    Attributes:
        lane (Lane): The lane this node belongs to, or None.
        container (Process | SubProcess): The flow-element container that owns this node,
            or None. Maintained by the container; used by ``incoming()`` / ``outgoing()``
            and by ``BPMNModel.validate()``.
    """

    def __init__(self, name: str = "", layout: dict = None, metadata=None, timestamp=None):
        if type(self) is FlowNode:
            raise TypeError("FlowNode is abstract and cannot be instantiated directly")
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.__lane = None
        self.__container = None

    @property
    def lane(self):
        """Lane: Get the lane this node belongs to, or None."""
        return self.__lane

    @lane.setter
    def lane(self, lane):
        """Lane: Set the lane this node belongs to.

        Raises:
            TypeError: if lane is neither a Lane nor None.
        """
        if lane is not None and not isinstance(lane, Lane):
            raise TypeError(f"lane must be a Lane or None, got {type(lane).__name__}")
        self.__lane = lane

    @property
    def container(self):
        """Process | SubProcess: Get the container that owns this node, or None."""
        return self.__container

    @container.setter
    def container(self, container):
        """Process | SubProcess: Set the owning flow-element container.

        Raises:
            TypeError: if container is not a Process, SubProcess, or None.
        """
        if container is not None and not isinstance(container, (Process, SubProcess)):
            raise TypeError(
                f"container must be a Process, SubProcess, or None, "
                f"got {type(container).__name__}"
            )
        self.__container = container

    def outgoing(self) -> set:
        """set[SequenceFlow]: Sequence flows leaving this node (derived from its container)."""
        if self.__container is None:
            return set()
        return {flow for flow in self.__container.sequence_flows if flow.source is self}

    def incoming(self) -> set:
        """set[SequenceFlow]: Sequence flows entering this node (derived from its container)."""
        if self.__container is None:
            return set()
        return {flow for flow in self.__container.sequence_flows if flow.target is self}


class Activity(FlowNode):
    """Abstract base for BPMN Activities -- Task, SubProcess, Transaction, CallActivity
    (BPMN 2.0.2 §10.2).

    Attributes:
        loop_characteristics (LoopCharacteristics): The loop / multi-instance marker.
        default_flow (SequenceFlow): Derived -- the outgoing sequence flow marked
            ``is_default``, or None (BPMN 2.0.2 §8.3.13).
    """

    def __init__(self, name: str = "", loop_characteristics: "LoopCharacteristics" = None,
                 layout: dict = None, metadata=None, timestamp=None):
        if type(self) is Activity:
            raise TypeError("Activity is abstract and cannot be instantiated directly")
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.loop_characteristics = (loop_characteristics if loop_characteristics is not None
                                     else LoopCharacteristics.NONE)

    @property
    def loop_characteristics(self) -> "LoopCharacteristics":
        """LoopCharacteristics: Get the loop / multi-instance marker."""
        return self.__loop_characteristics

    @loop_characteristics.setter
    def loop_characteristics(self, loop_characteristics: "LoopCharacteristics"):
        """LoopCharacteristics: Set the loop / multi-instance marker.

        Raises:
            TypeError: if not a LoopCharacteristics.
        """
        if not isinstance(loop_characteristics, LoopCharacteristics):
            raise TypeError(
                f"loop_characteristics must be a LoopCharacteristics, "
                f"got {type(loop_characteristics).__name__}"
            )
        self.__loop_characteristics = loop_characteristics

    @property
    def default_flow(self):
        """SequenceFlow: The outgoing sequence flow marked ``is_default``, or None (derived)."""
        return _default_outgoing_flow(self)


class Task(Activity):
    """A BPMN Task -- an atomic Activity (BPMN 2.0.2 §10.2.3).

    Args:
        name (str): The task label (may be empty).
        task_type (TaskType): The task's kind (``TaskType.DEFAULT`` as default).
        loop_characteristics (LoopCharacteristics): Inherited from Activity.

    Attributes:
        task_type (TaskType): The task's kind.
    """

    def __init__(self, name: str = "", task_type: "TaskType" = None,
                 loop_characteristics: "LoopCharacteristics" = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, loop_characteristics=loop_characteristics,
                         layout=layout, metadata=metadata, timestamp=timestamp)
        self.task_type = task_type if task_type is not None else TaskType.DEFAULT

    @property
    def task_type(self) -> "TaskType":
        """TaskType: Get the task type."""
        return self.__task_type

    @task_type.setter
    def task_type(self, task_type: "TaskType"):
        """TaskType: Set the task type.

        Raises:
            TypeError: if not a TaskType.
        """
        if not isinstance(task_type, TaskType):
            raise TypeError(f"task_type must be a TaskType, got {type(task_type).__name__}")
        self.__task_type = task_type

    def __repr__(self):
        return f"Task(name='{self.name}', task_type={self.task_type})"


class SubProcess(Activity):
    """A BPMN Sub-Process -- an Activity that is *also* a flow-element container
    (BPMN 2.0.2 §10.2.5).

    Growth (01-... §3.5): ``is_expanded``, ``is_event_subprocess``, ``is_ad_hoc``,
    ``triggered_by_event``.

    Attributes:
        flow_nodes (set[FlowNode]): Child flow nodes. Maintains each child's ``container``.
        sequence_flows (set[SequenceFlow]): Child sequence flows.
    """

    def __init__(self, name: str = "", loop_characteristics: "LoopCharacteristics" = None,
                 flow_nodes: set = None, sequence_flows: set = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, loop_characteristics=loop_characteristics,
                         layout=layout, metadata=metadata, timestamp=timestamp)
        self.__flow_nodes = set()
        self.__sequence_flows = set()
        self.flow_nodes = flow_nodes if flow_nodes is not None else set()
        self.sequence_flows = sequence_flows if sequence_flows is not None else set()

    @property
    def flow_nodes(self) -> set:
        """set[FlowNode]: Get the child flow nodes."""
        return self.__flow_nodes

    @flow_nodes.setter
    def flow_nodes(self, flow_nodes: set):
        """set[FlowNode]: Set the child flow nodes; maintains each child's ``container``.

        Raises:
            TypeError: if any element is not a FlowNode.
        """
        flow_nodes = set(flow_nodes)
        for node in flow_nodes:
            if not isinstance(node, FlowNode):
                raise TypeError(
                    f"flow_nodes must contain FlowNode instances, got {type(node).__name__}"
                )
        for node in self.__flow_nodes:
            if node.container is self:
                node.container = None
        for node in flow_nodes:
            node.container = self
        self.__flow_nodes = flow_nodes

    def add_flow_node(self, node: "FlowNode"):
        """Add a child flow node, setting its ``container``.

        Raises:
            TypeError: if node is not a FlowNode.
        """
        if not isinstance(node, FlowNode):
            raise TypeError(f"node must be a FlowNode, got {type(node).__name__}")
        node.container = self
        self.__flow_nodes.add(node)

    def remove_flow_node(self, node: "FlowNode"):
        """Remove a child flow node, clearing its ``container``."""
        if node in self.__flow_nodes:
            if node.container is self:
                node.container = None
            self.__flow_nodes.discard(node)

    @property
    def sequence_flows(self) -> set:
        """set[SequenceFlow]: Get the child sequence flows."""
        return self.__sequence_flows

    @sequence_flows.setter
    def sequence_flows(self, sequence_flows: set):
        """set[SequenceFlow]: Set the child sequence flows.

        Raises:
            TypeError: if any element is not a SequenceFlow.
        """
        self.__sequence_flows = _checked_set(sequence_flows, SequenceFlow, "sequence_flows")

    def add_sequence_flow(self, flow: "SequenceFlow"):
        """Add a child sequence flow.

        Raises:
            TypeError: if flow is not a SequenceFlow.
        """
        if not isinstance(flow, SequenceFlow):
            raise TypeError(f"flow must be a SequenceFlow, got {type(flow).__name__}")
        self.__sequence_flows.add(flow)

    def remove_sequence_flow(self, flow: "SequenceFlow"):
        """Remove a child sequence flow."""
        self.__sequence_flows.discard(flow)

    def __repr__(self):
        return (f"{type(self).__name__}(name='{self.name}', "
                f"flow_nodes={len(self.flow_nodes)}, sequence_flows={len(self.sequence_flows)})")


class Transaction(SubProcess):
    """A BPMN Transaction -- a SubProcess with transactional behaviour (BPMN 2.0.2 §10.2.5).

    Decision Q7: modelled as a ``SubProcess`` subtype, following the spec, even though WME
    treats Transaction and SubProcess as siblings.

    Growth (01-... §3.5): ``method`` (Compensate / Store / Image).
    """


class CallActivity(Activity):
    """A BPMN Call Activity -- invokes a reusable Process or global Task (BPMN 2.0.2 §10.2.6).

    Growth (01-... §3.5): ``called_element`` (reference to the invoked Process / global task).
    """


class Event(FlowNode):
    """Abstract base for BPMN Events (BPMN 2.0.2 §10.4).

    Orthogonal model (decision D2): an event is described by a ``direction`` (catch / throw)
    and an ``event_definition`` (message / timer / ...), instead of WME's flat string enum.
    The legal ``(class, direction) -> event_definition`` combinations are enforced
    construction-time against ``_LEGAL_EVENT_DEFINITIONS``.

    Growth (01-... §3.5): ``parallel_multiple``, multiple event definitions.

    Attributes:
        direction (EventDirection): Whether the event catches or throws.
        event_definition (EventDefinitionType): The trigger kind (``NONE`` for a plain event).
    """

    def __init__(self, name: str = "", direction: "EventDirection" = None,
                 event_definition: "EventDefinitionType" = None,
                 layout: dict = None, metadata=None, timestamp=None):
        if type(self) is Event:
            raise TypeError("Event is abstract and cannot be instantiated directly")
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.__direction = None
        self.__event_definition = None
        self.direction = direction if direction is not None else self._default_direction()
        self.event_definition = (event_definition if event_definition is not None
                                 else EventDefinitionType.NONE)

    def _default_direction(self) -> "EventDirection":
        """EventDirection: Subclass hook for the default direction."""
        raise NotImplementedError

    @property
    def direction(self) -> "EventDirection":
        """EventDirection: Get the event direction (catch / throw)."""
        return self.__direction

    @direction.setter
    def direction(self, direction: "EventDirection"):
        """EventDirection: Set the event direction.

        Raises:
            TypeError: if not an EventDirection.
            ValueError: if the resulting (class, direction, event_definition) is illegal.
        """
        if not isinstance(direction, EventDirection):
            raise TypeError(
                f"direction must be an EventDirection, got {type(direction).__name__}"
            )
        self.__direction = direction
        self._check_event_definition()

    @property
    def event_definition(self) -> "EventDefinitionType":
        """EventDefinitionType: Get the event definition (trigger kind)."""
        return self.__event_definition

    @event_definition.setter
    def event_definition(self, event_definition: "EventDefinitionType"):
        """EventDefinitionType: Set the event definition.

        Raises:
            TypeError: if not an EventDefinitionType.
            ValueError: if the resulting (class, direction, event_definition) is illegal.
        """
        if not isinstance(event_definition, EventDefinitionType):
            raise TypeError(
                f"event_definition must be an EventDefinitionType, "
                f"got {type(event_definition).__name__}"
            )
        self.__event_definition = event_definition
        self._check_event_definition()

    def _check_event_definition(self):
        """Enforce the (class, direction) -> legal event_definition table. Guarded so it
        does not fire mid-``__init__`` before both axes are set."""
        if self.__direction is None or self.__event_definition is None:
            return
        legal = _legal_event_definitions(self)
        if legal is not None and self.__event_definition not in legal:
            raise ValueError(
                f"{type(self).__name__} with direction {self.__direction.name} cannot have "
                f"event_definition {self.__event_definition.name}"
            )

    def __repr__(self):
        return (f"{type(self).__name__}(name='{self.name}', direction={self.direction}, "
                f"event_definition={self.event_definition})")


class StartEvent(Event):
    """A BPMN Start Event -- always a catching event (BPMN 2.0.2 §10.4.2).

    Growth (01-... §3.5): ``is_interrupting``.
    """

    def _default_direction(self) -> "EventDirection":
        return EventDirection.CATCH

    @Event.direction.setter
    def direction(self, direction: "EventDirection"):
        """EventDirection: A StartEvent's direction is always ``CATCH``.

        Raises:
            ValueError: if direction is not ``EventDirection.CATCH``.
        """
        if direction is not EventDirection.CATCH:
            raise ValueError("StartEvent.direction must be EventDirection.CATCH")
        super(StartEvent, StartEvent).direction.fset(self, direction)


class IntermediateEvent(Event):
    """A BPMN Intermediate Event -- may catch or throw (BPMN 2.0.2 §10.4.4)."""

    def _default_direction(self) -> "EventDirection":
        return EventDirection.CATCH


class EndEvent(Event):
    """A BPMN End Event -- always a throwing event (BPMN 2.0.2 §10.4.3)."""

    def _default_direction(self) -> "EventDirection":
        return EventDirection.THROW

    @Event.direction.setter
    def direction(self, direction: "EventDirection"):
        """EventDirection: An EndEvent's direction is always ``THROW``.

        Raises:
            ValueError: if direction is not ``EventDirection.THROW``.
        """
        if direction is not EventDirection.THROW:
            raise ValueError("EndEvent.direction must be EventDirection.THROW")
        super(EndEvent, EndEvent).direction.fset(self, direction)


class Gateway(FlowNode):
    """A BPMN Gateway -- controls sequence-flow divergence / convergence (BPMN 2.0.2 §10.5).

    Concrete: gateways have no subclasses; ``gateway_type`` discriminates.

    Attributes:
        gateway_type (GatewayType): The kind of gateway.
        default_flow (SequenceFlow): Derived -- the outgoing sequence flow marked
            ``is_default``, or None. Always None for parallel / event-based gateways
            (BPMN 2.0.2 §8.3.13).
    """

    def __init__(self, name: str = "", gateway_type: "GatewayType" = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.gateway_type = gateway_type if gateway_type is not None else GatewayType.EXCLUSIVE

    @property
    def gateway_type(self) -> "GatewayType":
        """GatewayType: Get the gateway kind."""
        return self.__gateway_type

    @gateway_type.setter
    def gateway_type(self, gateway_type: "GatewayType"):
        """GatewayType: Set the gateway kind.

        Raises:
            TypeError: if not a GatewayType.
        """
        if not isinstance(gateway_type, GatewayType):
            raise TypeError(
                f"gateway_type must be a GatewayType, got {type(gateway_type).__name__}"
            )
        self.__gateway_type = gateway_type

    @property
    def default_flow(self):
        """SequenceFlow: The outgoing sequence flow marked ``is_default``, or None (derived)."""
        return _default_outgoing_flow(self)

    def __repr__(self):
        return f"Gateway(name='{self.name}', gateway_type={self.gateway_type})"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class Artifact(BPMNElement):
    """Abstract base for BPMN Artifacts -- visual-only, never a sequence / message endpoint
    (BPMN 2.0.2 §8.3.1)."""

    def __init__(self, name: str = "", layout: dict = None, metadata=None, timestamp=None):
        if type(self) is Artifact:
            raise TypeError("Artifact is abstract and cannot be instantiated directly")
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)


class TextAnnotation(Artifact):
    """A BPMN Text Annotation -- a free-text note (BPMN 2.0.2 §8.3.1).

    Attributes:
        text (str): The annotation body.
    """

    def __init__(self, name: str = "", text: str = "", layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.text = text

    @property
    def text(self) -> str:
        """str: Get the annotation body."""
        return self.__text

    @text.setter
    def text(self, text: str):
        """str: Set the annotation body. ``None`` is coerced to ``""``.

        Raises:
            TypeError: if text is neither a str nor None.
        """
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str or None, got {type(text).__name__}")
        self.__text = text

    def __repr__(self):
        return f"TextAnnotation(name='{self.name}', text='{self.text}')"


class Group(Artifact):
    """A BPMN Group -- a visual overlay with no semantic containment (BPMN 2.0.2 §8.3.1).

    Growth (01-... §3.5): ``category_value``.
    """


# ---------------------------------------------------------------------------
# Data elements
# ---------------------------------------------------------------------------

class DataElement(BPMNElement):
    """Abstract base for BPMN data elements -- the ItemAware seam (BPMN 2.0.2 §10.3)."""

    def __init__(self, name: str = "", layout: dict = None, metadata=None, timestamp=None):
        if type(self) is DataElement:
            raise TypeError("DataElement is abstract and cannot be instantiated directly")
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)


class DataObject(DataElement):
    """A BPMN Data Object (BPMN 2.0.2 §10.3).

    Growth (01-... §3.5): ``is_collection``, ItemAware typing.
    """


class DataStore(DataElement):
    """A BPMN Data Store -- definition-level; lives on ``BPMNModel`` (BPMN 2.0.2 §10.3).

    Growth (01-... §3.5): ``data_store_ref``.
    """


# ---------------------------------------------------------------------------
# Connecting objects (decision D6 -- four classes under one base)
# ---------------------------------------------------------------------------

class BPMNConnectingObject(BPMNElement):
    """Abstract base for the four BPMN connecting objects.

    ``source`` and ``target`` are required (non-None). Each concrete subclass overrides
    ``_check_endpoint`` with its own endpoint type rule; model-wide rules (same-process,
    cross-pool, ...) live in ``BPMNModel.validate()``.

    Attributes:
        source (BPMNElement): The connection's source element.
        target (BPMNElement): The connection's target element.
    """

    def __init__(self, source, target, name: str = "", layout: dict = None,
                 metadata=None, timestamp=None):
        if type(self) is BPMNConnectingObject:
            raise TypeError(
                "BPMNConnectingObject is abstract and cannot be instantiated directly"
            )
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.__source = None
        self.__target = None
        self.source = source
        self.target = target

    @property
    def source(self):
        """BPMNElement: Get the connection's source element."""
        return self.__source

    @source.setter
    def source(self, source):
        """BPMNElement: Set the connection's source element.

        Raises:
            TypeError: if source is not an acceptable endpoint for this connection kind.
        """
        self._check_endpoint(source, "source")
        self.__source = source

    @property
    def target(self):
        """BPMNElement: Get the connection's target element."""
        return self.__target

    @target.setter
    def target(self, target):
        """BPMNElement: Set the connection's target element.

        Raises:
            TypeError: if target is not an acceptable endpoint for this connection kind.
        """
        self._check_endpoint(target, "target")
        self.__target = target

    def _check_endpoint(self, endpoint, role: str):
        """Type-check a single endpoint. Base requires any BPMNElement; subclasses tighten.

        Raises:
            TypeError: if endpoint is not acceptable.
        """
        if not isinstance(endpoint, BPMNElement):
            raise TypeError(
                f"{type(self).__name__} {role} must be a BPMNElement, "
                f"got {type(endpoint).__name__}"
            )

    def __repr__(self):
        source_name = self.source.name if self.source is not None else None
        target_name = self.target.name if self.target is not None else None
        return (f"{type(self).__name__}(name='{self.name}', "
                f"source='{source_name}', target='{target_name}')")


class SequenceFlow(BPMNConnectingObject):
    """A BPMN Sequence Flow -- orders FlowNodes within a Process / SubProcess
    (BPMN 2.0.2 §8.3.13).

    Growth (01-... §3.5): ``condition_expression``.

    Attributes:
        is_default (bool): Whether this is the source's default flow. Settable ``True`` only
            when the source can legally carry a default (BPMN 2.0.2 §8.3.13).
    """

    def __init__(self, source, target, name: str = "", is_default: bool = False,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(source, target, name=name, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.__is_default = False
        self.is_default = is_default

    def _check_endpoint(self, endpoint, role: str):
        """A SequenceFlow connects FlowNode -> FlowNode.

        Raises:
            TypeError: if endpoint is not a FlowNode.
        """
        if not isinstance(endpoint, FlowNode):
            raise TypeError(
                f"SequenceFlow {role} must be a FlowNode, got {type(endpoint).__name__}"
            )

    @property
    def is_default(self) -> bool:
        """bool: Get whether this is the source's default flow."""
        return self.__is_default

    @is_default.setter
    def is_default(self, value: bool):
        """bool: Set whether this is the source's default flow.

        Raises:
            TypeError: if value is not a bool.
            ValueError: if set ``True`` and the source cannot legally carry a default
                (BPMN 2.0.2 §8.3.13).
        """
        if not isinstance(value, bool):
            raise TypeError(f"is_default must be a bool, got {type(value).__name__}")
        if value and not _source_can_carry_default(self.source):
            raise ValueError(
                "is_default may only be set on a sequence flow whose source is an Activity "
                "or an exclusive / inclusive / complex Gateway (BPMN 2.0.2 §8.3.13)"
            )
        self.__is_default = value

    def __repr__(self):
        source_name = self.source.name if self.source is not None else None
        target_name = self.target.name if self.target is not None else None
        return (f"SequenceFlow(name='{self.name}', source='{source_name}', "
                f"target='{target_name}', is_default={self.is_default})")


class MessageFlow(BPMNConnectingObject):
    """A BPMN Message Flow -- communication across pools (BPMN 2.0.2 §9.3). Lives on a
    Collaboration.

    Endpoints are message-eligible nodes (Activities / Events) *or* whole Participants
    (pools). BPMN 2.0.2 §9.3 permits a message flow to attach to a Pool directly; the WME
    editor draws agentic collaboration message flows pool-to-pool, so the metamodel accepts
    ``Participant`` endpoints too."""

    def _check_endpoint(self, endpoint, role: str):
        """A MessageFlow connects message-eligible nodes (Activities / Events) or
        Participants (pools).

        Raises:
            TypeError: if endpoint is not an Activity, Event, or Participant.
        """
        if not isinstance(endpoint, (Activity, Event, Participant)):
            raise TypeError(
                f"MessageFlow {role} must be an Activity, Event, or Participant, "
                f"got {type(endpoint).__name__}"
            )


class Association(BPMNConnectingObject):
    """A BPMN Association -- links an Artifact to another element (BPMN 2.0.2 §8.3.1).

    Growth (01-... §3.5): ``direction`` (None / One / Both).
    """

    def __init__(self, source, target, name: str = "", layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(source, target, name=name, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        if not (isinstance(self.source, Artifact) or isinstance(self.target, Artifact)):
            raise ValueError("Association must have at least one Artifact endpoint")


class DataAssociation(BPMNConnectingObject):
    """A BPMN Data Association -- links a DataElement and a FlowNode (BPMN 2.0.2 §10.3)."""

    def __init__(self, source, target, name: str = "", layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(source, target, name=name, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        ends = (self.source, self.target)
        data_count = sum(1 for end in ends if isinstance(end, DataElement))
        node_count = sum(1 for end in ends if isinstance(end, FlowNode))
        if not (data_count == 1 and node_count == 1):
            raise ValueError(
                "DataAssociation must connect exactly one DataElement and one FlowNode"
            )

    def _check_endpoint(self, endpoint, role: str):
        """A DataAssociation endpoint is a DataElement or a FlowNode.

        Raises:
            TypeError: if endpoint is neither.
        """
        if not isinstance(endpoint, (DataElement, FlowNode)):
            raise TypeError(
                f"DataAssociation {role} must be a DataElement or FlowNode, "
                f"got {type(endpoint).__name__}"
            )


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

class Lane(BPMNElement):
    """A BPMN Lane -- a sub-partition of a Process within a Pool (BPMN 2.0.2 §10.7).

    Attributes:
        flow_nodes (set[FlowNode]): The flow nodes that belong to this lane. Maintains each
            member's ``lane`` back-reference.
    """

    def __init__(self, name: str = "", flow_nodes: set = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.__flow_nodes = set()
        self.flow_nodes = flow_nodes if flow_nodes is not None else set()

    @property
    def flow_nodes(self) -> set:
        """set[FlowNode]: Get the lane's member flow nodes."""
        return self.__flow_nodes

    @flow_nodes.setter
    def flow_nodes(self, flow_nodes: set):
        """set[FlowNode]: Set the lane's members; maintains each member's ``lane`` back-ref.

        Raises:
            TypeError: if any element is not a FlowNode.
        """
        flow_nodes = set(flow_nodes)
        for node in flow_nodes:
            if not isinstance(node, FlowNode):
                raise TypeError(
                    f"flow_nodes must contain FlowNode instances, got {type(node).__name__}"
                )
        for node in self.__flow_nodes:
            if node.lane is self:
                node.lane = None
        for node in flow_nodes:
            node.lane = self
        self.__flow_nodes = flow_nodes

    def add_flow_node(self, node: "FlowNode"):
        """Add a member flow node, setting its ``lane`` back-reference.

        Raises:
            TypeError: if node is not a FlowNode.
            ValueError: if node already belongs to a different lane.
        """
        if not isinstance(node, FlowNode):
            raise TypeError(f"node must be a FlowNode, got {type(node).__name__}")
        if node.lane is not None and node.lane is not self:
            raise ValueError(f"FlowNode '{node.name}' already belongs to another lane")
        node.lane = self
        self.__flow_nodes.add(node)

    def remove_flow_node(self, node: "FlowNode"):
        """Remove a member flow node, clearing its ``lane`` back-reference."""
        if node in self.__flow_nodes:
            if node.lane is self:
                node.lane = None
            self.__flow_nodes.discard(node)

    def __repr__(self):
        return f"Lane(name='{self.name}', flow_nodes={len(self.flow_nodes)})"


class Process(BPMNElement):
    """A BPMN Process -- a top-level flow-element container (BPMN 2.0.2 §10).

    Uses the same ``flow_nodes`` / ``sequence_flows`` attribute names as ``SubProcess`` so
    the two are duck-type compatible as flow-element containers.

    Attributes:
        flow_nodes (set[FlowNode]): Flow nodes directly in this process. Maintains each
            node's ``container`` back-reference.
        sequence_flows (set[SequenceFlow]): Sequence flows in this process.
        lanes (set[Lane]): The process's lanes.
        artifacts (set[Artifact]): Annotations and groups.
        data_objects (set[DataObject]): Data objects in this process.
        associations (set[Association]): Associations in this process.
        data_associations (set[DataAssociation]): Data associations in this process.
    """

    def __init__(self, name: str = "", flow_nodes: set = None, sequence_flows: set = None,
                 lanes: set = None, artifacts: set = None, data_objects: set = None,
                 associations: set = None, data_associations: set = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.__flow_nodes = set()
        self.__sequence_flows = set()
        self.__lanes = set()
        self.__artifacts = set()
        self.__data_objects = set()
        self.__associations = set()
        self.__data_associations = set()
        self.flow_nodes = flow_nodes if flow_nodes is not None else set()
        self.sequence_flows = sequence_flows if sequence_flows is not None else set()
        self.lanes = lanes if lanes is not None else set()
        self.artifacts = artifacts if artifacts is not None else set()
        self.data_objects = data_objects if data_objects is not None else set()
        self.associations = associations if associations is not None else set()
        self.data_associations = data_associations if data_associations is not None else set()

    @property
    def flow_nodes(self) -> set:
        """set[FlowNode]: Get the process's flow nodes."""
        return self.__flow_nodes

    @flow_nodes.setter
    def flow_nodes(self, flow_nodes: set):
        """set[FlowNode]: Set the process's flow nodes; maintains each node's ``container``.

        Raises:
            TypeError: if any element is not a FlowNode.
        """
        flow_nodes = set(flow_nodes)
        for node in flow_nodes:
            if not isinstance(node, FlowNode):
                raise TypeError(
                    f"flow_nodes must contain FlowNode instances, got {type(node).__name__}"
                )
        for node in self.__flow_nodes:
            if node.container is self:
                node.container = None
        for node in flow_nodes:
            node.container = self
        self.__flow_nodes = flow_nodes

    def add_flow_node(self, node: "FlowNode"):
        """Add a flow node, setting its ``container``.

        Raises:
            TypeError: if node is not a FlowNode.
        """
        if not isinstance(node, FlowNode):
            raise TypeError(f"node must be a FlowNode, got {type(node).__name__}")
        node.container = self
        self.__flow_nodes.add(node)

    def remove_flow_node(self, node: "FlowNode"):
        """Remove a flow node, clearing its ``container``."""
        if node in self.__flow_nodes:
            if node.container is self:
                node.container = None
            self.__flow_nodes.discard(node)

    @property
    def sequence_flows(self) -> set:
        """set[SequenceFlow]: Get the process's sequence flows."""
        return self.__sequence_flows

    @sequence_flows.setter
    def sequence_flows(self, sequence_flows: set):
        """set[SequenceFlow]: Set the process's sequence flows.

        Raises:
            TypeError: if any element is not a SequenceFlow.
        """
        self.__sequence_flows = _checked_set(sequence_flows, SequenceFlow, "sequence_flows")

    def add_sequence_flow(self, flow: "SequenceFlow"):
        """Add a sequence flow.

        Raises:
            TypeError: if flow is not a SequenceFlow.
        """
        if not isinstance(flow, SequenceFlow):
            raise TypeError(f"flow must be a SequenceFlow, got {type(flow).__name__}")
        self.__sequence_flows.add(flow)

    def remove_sequence_flow(self, flow: "SequenceFlow"):
        """Remove a sequence flow."""
        self.__sequence_flows.discard(flow)

    @property
    def lanes(self) -> set:
        """set[Lane]: Get the process's lanes."""
        return self.__lanes

    @lanes.setter
    def lanes(self, lanes: set):
        """set[Lane]: Set the process's lanes.

        Raises:
            TypeError: if any element is not a Lane.
        """
        self.__lanes = _checked_set(lanes, Lane, "lanes")

    def add_lane(self, lane: "Lane"):
        """Add a lane.

        Raises:
            TypeError: if lane is not a Lane.
        """
        if not isinstance(lane, Lane):
            raise TypeError(f"lane must be a Lane, got {type(lane).__name__}")
        self.__lanes.add(lane)

    def remove_lane(self, lane: "Lane"):
        """Remove a lane."""
        self.__lanes.discard(lane)

    @property
    def artifacts(self) -> set:
        """set[Artifact]: Get the process's artifacts (annotations, groups)."""
        return self.__artifacts

    @artifacts.setter
    def artifacts(self, artifacts: set):
        """set[Artifact]: Set the process's artifacts.

        Raises:
            TypeError: if any element is not an Artifact.
        """
        self.__artifacts = _checked_set(artifacts, Artifact, "artifacts")

    def add_artifact(self, artifact: "Artifact"):
        """Add an artifact.

        Raises:
            TypeError: if artifact is not an Artifact.
        """
        if not isinstance(artifact, Artifact):
            raise TypeError(f"artifact must be an Artifact, got {type(artifact).__name__}")
        self.__artifacts.add(artifact)

    def remove_artifact(self, artifact: "Artifact"):
        """Remove an artifact."""
        self.__artifacts.discard(artifact)

    @property
    def data_objects(self) -> set:
        """set[DataObject]: Get the process's data objects."""
        return self.__data_objects

    @data_objects.setter
    def data_objects(self, data_objects: set):
        """set[DataObject]: Set the process's data objects.

        Raises:
            TypeError: if any element is not a DataObject.
        """
        self.__data_objects = _checked_set(data_objects, DataObject, "data_objects")

    def add_data_object(self, data_object: "DataObject"):
        """Add a data object.

        Raises:
            TypeError: if data_object is not a DataObject.
        """
        if not isinstance(data_object, DataObject):
            raise TypeError(
                f"data_object must be a DataObject, got {type(data_object).__name__}"
            )
        self.__data_objects.add(data_object)

    def remove_data_object(self, data_object: "DataObject"):
        """Remove a data object."""
        self.__data_objects.discard(data_object)

    @property
    def associations(self) -> set:
        """set[Association]: Get the process's associations."""
        return self.__associations

    @associations.setter
    def associations(self, associations: set):
        """set[Association]: Set the process's associations.

        Raises:
            TypeError: if any element is not an Association.
        """
        self.__associations = _checked_set(associations, Association, "associations")

    def add_association(self, association: "Association"):
        """Add an association.

        Raises:
            TypeError: if association is not an Association.
        """
        if not isinstance(association, Association):
            raise TypeError(
                f"association must be an Association, got {type(association).__name__}"
            )
        self.__associations.add(association)

    def remove_association(self, association: "Association"):
        """Remove an association."""
        self.__associations.discard(association)

    @property
    def data_associations(self) -> set:
        """set[DataAssociation]: Get the process's data associations."""
        return self.__data_associations

    @data_associations.setter
    def data_associations(self, data_associations: set):
        """set[DataAssociation]: Set the process's data associations.

        Raises:
            TypeError: if any element is not a DataAssociation.
        """
        self.__data_associations = _checked_set(
            data_associations, DataAssociation, "data_associations"
        )

    def add_data_association(self, data_association: "DataAssociation"):
        """Add a data association.

        Raises:
            TypeError: if data_association is not a DataAssociation.
        """
        if not isinstance(data_association, DataAssociation):
            raise TypeError(
                f"data_association must be a DataAssociation, "
                f"got {type(data_association).__name__}"
            )
        self.__data_associations.add(data_association)

    def remove_data_association(self, data_association: "DataAssociation"):
        """Remove a data association."""
        self.__data_associations.discard(data_association)

    def __repr__(self):
        return (f"Process(name='{self.name}', flow_nodes={len(self.flow_nodes)}, "
                f"sequence_flows={len(self.sequence_flows)}, lanes={len(self.lanes)})")


class Participant(BPMNElement):
    """A BPMN Participant -- a Pool; references the Process it contains (BPMN 2.0.2 §9.2).

    Attributes:
        process (Process): The process this participant (pool) contains, or None.
    """

    def __init__(self, name: str = "", process: "Process" = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.process = process

    @property
    def process(self):
        """Process: Get the process this participant contains, or None."""
        return self.__process

    @process.setter
    def process(self, process):
        """Process: Set the process this participant contains.

        Raises:
            TypeError: if process is neither a Process nor None.
        """
        if process is not None and not isinstance(process, Process):
            raise TypeError(f"process must be a Process or None, got {type(process).__name__}")
        self.__process = process

    def __repr__(self):
        process_name = self.process.name if self.process is not None else None
        return f"Participant(name='{self.name}', process='{process_name}')"


class Collaboration(BPMNElement):
    """A BPMN Collaboration -- participants (pools) and the message flows between them
    (BPMN 2.0.2 §9). Present only when the diagram has pools.

    Attributes:
        participants (set[Participant]): The pools in this collaboration.
        message_flows (set[MessageFlow]): The message flows between participants.
    """

    def __init__(self, name: str = "", participants: set = None, message_flows: set = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, layout=layout, metadata=metadata, timestamp=timestamp)
        self.__participants = set()
        self.__message_flows = set()
        self.participants = participants if participants is not None else set()
        self.message_flows = message_flows if message_flows is not None else set()

    @property
    def participants(self) -> set:
        """set[Participant]: Get the collaboration's participants."""
        return self.__participants

    @participants.setter
    def participants(self, participants: set):
        """set[Participant]: Set the collaboration's participants.

        Raises:
            TypeError: if any element is not a Participant.
        """
        self.__participants = _checked_set(participants, Participant, "participants")

    def add_participant(self, participant: "Participant"):
        """Add a participant.

        Raises:
            TypeError: if participant is not a Participant.
        """
        if not isinstance(participant, Participant):
            raise TypeError(
                f"participant must be a Participant, got {type(participant).__name__}"
            )
        self.__participants.add(participant)

    def remove_participant(self, participant: "Participant"):
        """Remove a participant."""
        self.__participants.discard(participant)

    @property
    def message_flows(self) -> set:
        """set[MessageFlow]: Get the collaboration's message flows."""
        return self.__message_flows

    @message_flows.setter
    def message_flows(self, message_flows: set):
        """set[MessageFlow]: Set the collaboration's message flows.

        Raises:
            TypeError: if any element is not a MessageFlow.
        """
        self.__message_flows = _checked_set(message_flows, MessageFlow, "message_flows")

    def add_message_flow(self, message_flow: "MessageFlow"):
        """Add a message flow.

        Raises:
            TypeError: if message_flow is not a MessageFlow.
        """
        if not isinstance(message_flow, MessageFlow):
            raise TypeError(
                f"message_flow must be a MessageFlow, got {type(message_flow).__name__}"
            )
        self.__message_flows.add(message_flow)

    def remove_message_flow(self, message_flow: "MessageFlow"):
        """Remove a message flow."""
        self.__message_flows.discard(message_flow)

    def __repr__(self):
        return (f"Collaboration(name='{self.name}', participants={len(self.participants)}, "
                f"message_flows={len(self.message_flows)})")


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class BPMNModel(Model):
    """The root of a BPMN model -- a first-class B-UML model (BPMN 2.0.2 §8).

    Pool-less diagram: one Process, ``collaboration`` is None.
    Pool-ful diagram: a Collaboration whose Participants each reference a Process here.

    Like ``StateMachine``, ``BPMNModel`` does not populate the inherited ``Model.elements``
    set; it exposes its own accessors (``all_flow_nodes()``, ``all_sequence_flows()``,
    ``all_connecting_objects()``) instead.

    Attributes:
        processes (set[Process]): The processes in this model (normally at least one).
        collaboration (Collaboration): The collaboration, or None for a pool-less diagram.
        data_stores (set[DataStore]): Definition-level data stores (BPMN 2.0.2 -- a
            ``dataStore`` is a root-level element).
    """

    def __init__(self, name: str, processes: set = None, collaboration: "Collaboration" = None,
                 data_stores: set = None, metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.__processes = set()
        self.__collaboration = None
        self.__data_stores = set()
        self.processes = processes if processes is not None else set()
        self.collaboration = collaboration
        self.data_stores = data_stores if data_stores is not None else set()

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the model name. Relaxed like ``BPMNElement`` -- a BPMN diagram name may
        be free text.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"BPMN model name must be a str or None, got {type(name).__name__}"
            )
        self._NamedElement__name = name

    @property
    def processes(self) -> set:
        """set[Process]: Get the processes in this model."""
        return self.__processes

    @processes.setter
    def processes(self, processes: set):
        """set[Process]: Set the processes in this model.

        Raises:
            TypeError: if any element is not a Process.
        """
        self.__processes = _checked_set(processes, Process, "processes")

    def add_process(self, process: "Process"):
        """Add a process.

        Raises:
            TypeError: if process is not a Process.
        """
        if not isinstance(process, Process):
            raise TypeError(f"process must be a Process, got {type(process).__name__}")
        self.__processes.add(process)

    def remove_process(self, process: "Process"):
        """Remove a process."""
        self.__processes.discard(process)

    @property
    def collaboration(self):
        """Collaboration: Get the collaboration, or None for a pool-less diagram."""
        return self.__collaboration

    @collaboration.setter
    def collaboration(self, collaboration):
        """Collaboration: Set the collaboration.

        Raises:
            TypeError: if collaboration is neither a Collaboration nor None.
        """
        if collaboration is not None and not isinstance(collaboration, Collaboration):
            raise TypeError(
                f"collaboration must be a Collaboration or None, "
                f"got {type(collaboration).__name__}"
            )
        self.__collaboration = collaboration

    @property
    def data_stores(self) -> set:
        """set[DataStore]: Get the definition-level data stores."""
        return self.__data_stores

    @data_stores.setter
    def data_stores(self, data_stores: set):
        """set[DataStore]: Set the definition-level data stores.

        Raises:
            TypeError: if any element is not a DataStore.
        """
        self.__data_stores = _checked_set(data_stores, DataStore, "data_stores")

    def add_data_store(self, data_store: "DataStore"):
        """Add a data store.

        Raises:
            TypeError: if data_store is not a DataStore.
        """
        if not isinstance(data_store, DataStore):
            raise TypeError(
                f"data_store must be a DataStore, got {type(data_store).__name__}"
            )
        self.__data_stores.add(data_store)

    def remove_data_store(self, data_store: "DataStore"):
        """Remove a data store."""
        self.__data_stores.discard(data_store)

    # --- derived accessors -------------------------------------------------

    def _all_containers(self) -> set:
        """set: All flow-element containers -- every Process plus every nested SubProcess."""
        result = set()

        def _collect(container):
            result.add(container)
            for node in container.flow_nodes:
                if isinstance(node, SubProcess):
                    _collect(node)

        for process in self.__processes:
            _collect(process)
        return result

    def all_flow_nodes(self) -> set:
        """set[FlowNode]: Every flow node, including those nested in SubProcesses."""
        return {node for container in self._all_containers()
                for node in container.flow_nodes}

    def all_sequence_flows(self) -> set:
        """set[SequenceFlow]: Every sequence flow, including those nested in SubProcesses."""
        return {flow for container in self._all_containers()
                for flow in container.sequence_flows}

    def all_connecting_objects(self) -> set:
        """set[BPMNConnectingObject]: Every connecting object in the model."""
        result = set(self.all_sequence_flows())
        for process in self.__processes:
            result |= process.associations
            result |= process.data_associations
        if self.__collaboration is not None:
            result |= self.__collaboration.message_flows
        return result

    # --- validation --------------------------------------------------------

    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the BPMN model according to its structural rules.

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: ``{"success": bool, "errors": list[str], "warnings": list[str]}``.
        """
        errors: list = []
        warnings: list = []

        self._validate_endpoint_references(errors)
        self._validate_flow_endpoint_types(errors)
        self._validate_default_flows(errors)
        self._validate_event_boundaries(errors)
        self._validate_event_definitions(errors)
        self._validate_lane_membership(errors)
        self._validate_collaboration(errors)
        self._warn_structural_smells(warnings)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _all_elements(self) -> set:
        """set: Every node-like element a connecting object may reference."""
        elements = set(self.all_flow_nodes())
        for process in self.__processes:
            elements |= process.artifacts
            elements |= process.data_objects
        elements |= self.__data_stores
        # Participants (pools) are valid MessageFlow endpoints (BPMN 2.0.2 §9.3),
        # so they belong to the reference universe for E1.
        if self.__collaboration is not None:
            elements |= self.__collaboration.participants
        return elements

    def _participant_of(self, node):
        """Participant: The participant whose process (transitively) contains ``node``,
        or None. A ``Participant`` endpoint (a pool used directly as a MessageFlow end)
        resolves to itself, so E3's same-pool check stays meaningful for pool-to-pool
        message flows."""
        if isinstance(node, Participant):
            return node
        container = node.container
        while isinstance(container, SubProcess):
            container = container.container
        if not isinstance(container, Process) or self.__collaboration is None:
            return None
        for participant in self.__collaboration.participants:
            if participant.process is container:
                return participant
        return None

    def _validate_endpoint_references(self, errors: list):
        """E1: every connecting object's source / target is reachable in this model."""
        universe = self._all_elements()
        for conn in self.all_connecting_objects():
            for role, endpoint in (("source", conn.source), ("target", conn.target)):
                if endpoint is None:
                    errors.append(f"{type(conn).__name__} '{conn.name}' has no {role}.")
                elif endpoint not in universe:
                    errors.append(
                        f"{type(conn).__name__} '{conn.name}' {role} '{endpoint.name}' "
                        f"({type(endpoint).__name__}) is not present in the model."
                    )

    def _validate_flow_endpoint_types(self, errors: list):
        """E2-E5: per-kind endpoint rules for the four connecting objects."""
        # E2 -- SequenceFlow: FlowNode -> FlowNode, same immediate container.
        for flow in self.all_sequence_flows():
            if not (isinstance(flow.source, FlowNode) and isinstance(flow.target, FlowNode)):
                errors.append(f"SequenceFlow '{flow.name}' must connect two FlowNodes.")
                continue
            if flow.source.container is None or flow.target.container is None:
                errors.append(
                    f"SequenceFlow '{flow.name}' has an endpoint that is not in any process."
                )
            elif flow.source.container is not flow.target.container:
                errors.append(
                    f"SequenceFlow '{flow.name}' crosses a process / sub-process boundary "
                    f"(source and target are in different containers)."
                )
        # E3 -- MessageFlow: message-eligible endpoints in different participants.
        if self.__collaboration is not None:
            for flow in self.__collaboration.message_flows:
                source_part = self._participant_of(flow.source)
                target_part = self._participant_of(flow.target)
                if (source_part is not None and target_part is not None
                        and source_part is target_part):
                    errors.append(
                        f"MessageFlow '{flow.name}' connects two nodes in the same pool "
                        f"'{source_part.name}'; message flows must cross pool boundaries."
                    )
        # E4 / E5 -- DataAssociation and Association endpoint composition.
        for process in self.__processes:
            for data_assoc in process.data_associations:
                ends = (data_assoc.source, data_assoc.target)
                data_count = sum(1 for end in ends if isinstance(end, DataElement))
                node_count = sum(1 for end in ends if isinstance(end, FlowNode))
                if not (data_count == 1 and node_count == 1):
                    errors.append(
                        f"DataAssociation '{data_assoc.name}' must connect exactly one "
                        f"DataElement and one FlowNode."
                    )
            for assoc in process.associations:
                if not (isinstance(assoc.source, Artifact)
                        or isinstance(assoc.target, Artifact)):
                    errors.append(
                        f"Association '{assoc.name}' must have at least one Artifact endpoint."
                    )

    def _validate_default_flows(self, errors: list):
        """E6: a default flow's source can legally carry one. E7: at most one default
        outgoing flow per source (so the derived ``default_flow`` is unambiguous)."""
        default_count: dict = {}
        for flow in self.all_sequence_flows():
            if flow.is_default:
                if not _source_can_carry_default(flow.source):
                    errors.append(
                        f"SequenceFlow '{flow.name}' is marked default but its source "
                        f"'{flow.source.name}' ({type(flow.source).__name__}) cannot carry "
                        f"a default flow (BPMN 2.0.2 §8.3.13)."
                    )
                default_count[flow.source] = default_count.get(flow.source, 0) + 1
        for source, count in default_count.items():
            if count > 1:
                errors.append(
                    f"{type(source).__name__} '{source.name}' has {count} outgoing default "
                    f"sequence flows; at most one is allowed."
                )

    def _validate_event_boundaries(self, errors: list):
        """E8: no SequenceFlow targets a StartEvent or originates from an EndEvent."""
        for flow in self.all_sequence_flows():
            if isinstance(flow.target, StartEvent):
                errors.append(
                    f"SequenceFlow '{flow.name}' targets StartEvent '{flow.target.name}'; "
                    f"start events cannot have incoming sequence flows."
                )
            if isinstance(flow.source, EndEvent):
                errors.append(
                    f"SequenceFlow '{flow.name}' originates from EndEvent "
                    f"'{flow.source.name}'; end events cannot have outgoing sequence flows."
                )

    def _validate_event_definitions(self, errors: list):
        """E9: re-check the (event class, direction) -> legal event_definition table."""
        for node in self.all_flow_nodes():
            if isinstance(node, Event):
                legal = _legal_event_definitions(node)
                if legal is not None and node.event_definition not in legal:
                    errors.append(
                        f"{type(node).__name__} '{node.name}' has illegal event_definition "
                        f"{node.event_definition.name} for direction {node.direction.name}."
                    )

    def _validate_lane_membership(self, errors: list):
        """E10: a lane's members belong to that lane's process; each node in <= 1 lane."""
        for process in self.__processes:
            for lane in process.lanes:
                for node in lane.flow_nodes:
                    if node.lane is not lane:
                        errors.append(
                            f"FlowNode '{node.name}' is listed in lane '{lane.name}' but "
                            f"its lane back-reference points elsewhere (node in two lanes?)."
                        )
                    owning = node.container
                    while isinstance(owning, SubProcess):
                        owning = owning.container
                    if owning is not process:
                        errors.append(
                            f"FlowNode '{node.name}' is in lane '{lane.name}' of process "
                            f"'{process.name}' but is not contained by that process."
                        )

    def _validate_collaboration(self, errors: list):
        """E11: every participant references a process that is in this model."""
        if self.__collaboration is None:
            return
        for participant in self.__collaboration.participants:
            if participant.process is None:
                errors.append(f"Participant '{participant.name}' has no process.")
            elif participant.process not in self.__processes:
                errors.append(
                    f"Participant '{participant.name}' references process "
                    f"'{participant.process.name}' which is not in the model."
                )

    def _warn_structural_smells(self, warnings: list):
        """W1-W4: non-blocking structural smells."""
        # W4 -- empty model / empty process.
        if not self.__processes:
            warnings.append(f"BPMNModel '{self.name}' has no processes.")
        for process in self.__processes:
            if not process.flow_nodes:
                warnings.append(f"Process '{process.name}' has no flow nodes.")
                continue
            # W1 -- process with no start event.
            if not any(isinstance(node, StartEvent) for node in process.flow_nodes):
                warnings.append(f"Process '{process.name}' has no start event.")
        # W2 -- non-start flow node with no incoming sequence flow (unreachable).
        for node in self.all_flow_nodes():
            if isinstance(node, StartEvent):
                continue
            if not node.incoming():
                warnings.append(
                    f"{type(node).__name__} '{node.name}' has no incoming sequence flow "
                    f"(unreachable)."
                )
        # W3 -- a source with a default flow but only one outgoing flow.
        for node in self.all_flow_nodes():
            if isinstance(node, (Activity, Gateway)) and node.default_flow is not None:
                outgoing_count = len(node.outgoing())
                if outgoing_count <= 1:
                    warnings.append(
                        f"{type(node).__name__} '{node.name}' has a default flow but only "
                        f"{outgoing_count} outgoing flow(s)."
                    )

    def __repr__(self):
        return (f"BPMNModel(name='{self.name}', processes={len(self.processes)}, "
                f"collaboration={self.collaboration is not None}, "
                f"data_stores={len(self.data_stores)})")
