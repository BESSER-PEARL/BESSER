"""UML Component metamodel for B-UML.

A first-class B-UML model for UML Component diagrams, alongside ``structural`` /
``state_machine`` / ``gui`` / ``bpmn``. Implements the design in
``.claude/component-deployment/01-component-deployment-design.md`` (reviewed and
locked 2026-05-18).

Hierarchy::

    Element -> NamedElement -> ComponentElement -> {Component, Interface, Permission,
                                                    ComponentRelationship}
                            -> Model -> ComponentModel

    Component <|-- Subsystem (a Component that is also a container)
    Component <|-- Skill
    Component <|-- Tool

    ComponentRelationship <|-- InterfaceProvided
                          <|-- InterfaceRequired
                          <|-- ComponentDependency <|-- AgenticEdge

Growth path (01-... §3.5) -- constructs designed but intentionally NOT implemented yet.
Each becomes a plain attribute when pulled into scope; no restructuring is needed:

* UML ``Realization`` as a distinct relationship class (currently absorbed into
  ``Component.realizes: List[str]`` of structural ``Class`` IDs).
* ``«autoscale»`` policy attributes (sibling to NR-6 in the requirements review).
* AgentGroup as a resource grouping (NR-6) -- reuse ``Subsystem`` as the structural
  cluster + an ``Artifact`` (deployment side) that manifests the Subsystem.

The agentic stereotype profile (decision D3, hybrid) carries the swarm intent via
typed discriminator attributes for the load-bearing distinctions (``AgentCategory``,
``Locality``, ``AgenticEdgeKind``, ``is_human``) plus a free-form ``stereotypes:
List[str]`` passthrough on every element for the long tail.
"""

from enum import Enum
from typing import List, Optional

from besser.BUML.metamodel.structural import Model, NamedElement


# ---------------------------------------------------------------------------
# Enumerations (decision D3 -- plain enum.Enum; .value strings match the WME
# stereotype strings so converters can map by value)
# ---------------------------------------------------------------------------

class AgentCategory(Enum):
    """The agent's role in a multi-agent governance pattern.

    ``NONE`` marks a non-agent Component (a plain UML Component, an external
    dependency, a Skill, a Tool). The other four values describe agent roles
    common to multi-agent governance patterns (Solution does domain work,
    Supervision evaluates / approves, Consensus mediates merge / decision,
    Collaboration orchestrates dispatch). Not tied to any specific agent
    framework or taxonomy -- worked examples (e.g. MOSAICO UNP) are
    use-case-level illustrations only.
    """
    NONE = "none"
    SOLUTION = "solution"
    SUPERVISION = "supervision"
    CONSENSUS = "consensus"
    COLLABORATION = "collaboration"


class Locality(Enum):
    """Where a Component / Artifact / Node is hosted, generalised from a
    binary ``«external»`` stereotype per NR-5 of the requirements review.

    * ``LOCAL`` -- owned and hosted by us.
    * ``EXTERNAL`` -- third-party API or service we don't own.
    * ``HYBRID`` -- on-prem with cloud fallback / mirror.
    """
    LOCAL = "local"
    EXTERNAL = "external"
    HYBRID = "hybrid"


class AgenticEdgeKind(Enum):
    """The kind of an ``AgenticEdge`` -- a typed dependency between agents
    and / or capabilities.

    Agent <-> agent kinds (carry permissions per A-4 of the requirements review)::
        DELEGATES, SUPERVISES, REVISES, COLLABORATES

    Agent -> capability kinds::
        HAS (-> Skill), USES (-> Tool), GRANTED (-> Permission)

    Capability <-> capability kinds::
        IMPLEMENTS (Tool -> Skill)
    """
    DELEGATES = "delegates"
    SUPERVISES = "supervises"
    REVISES = "revises"
    COLLABORATES = "collaborates"
    HAS = "has"
    USES = "uses"
    GRANTED = "granted"
    IMPLEMENTS = "implements"


# Edge kinds that conceptually flow between two agents.
_AGENT_TO_AGENT_KINDS = frozenset({
    AgenticEdgeKind.DELEGATES,
    AgenticEdgeKind.SUPERVISES,
    AgenticEdgeKind.REVISES,
    AgenticEdgeKind.COLLABORATES,
})


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


def _checked_str_list(values, label: str) -> List[str]:
    """Coerce ``values`` to a list of str, raising TypeError on a non-str entry."""
    result = list(values)
    for value in result:
        if not isinstance(value, str):
            raise TypeError(
                f"{label} must contain str entries, got {type(value).__name__}"
            )
    return result


def _is_agent_endpoint(component: "Component") -> bool:
    """True if a Component is a valid agent-edge endpoint (agent or human)."""
    return component.agent_category is not AgentCategory.NONE or component.is_human


# ---------------------------------------------------------------------------
# Base element (decision D9 -- relaxed name, opaque layout passthrough)
# ---------------------------------------------------------------------------

class ComponentElement(NamedElement):
    """Base class for every UML Component-diagram abstract-syntax element.

    Relaxes ``NamedElement.name`` (decision D9, mirrors BPMN D5): a Component-diagram
    label is free text and may contain spaces (``"Code Tester"``), colons
    (``"merge:approve"``), or be empty (a freshly-dropped element). Also carries
    ``layout`` -- an opaque diagram-interchange passthrough the metamodel never
    interprets (decision D10); only the converters read it. The opaque ``layout``
    dict is also where converters stash the stable WME element ID so cross-diagram
    string-ID references survive round-trips.

    Args:
        name (str): The element label. Empty allowed; ``None`` is coerced to ``""``.
        stereotypes (List[str]): Free-form stereotype strings (decision D3 long
            tail). Defaults to ``[]``.
        layout (dict): Opaque DI data. Stored untouched.
        metadata (Metadata): Inherited from NamedElement.
        timestamp (datetime): Inherited from NamedElement.

    Attributes:
        name (str): The element label (free text; may be empty).
        stereotypes (List[str]): Free-form stereotype passthrough.
        layout (dict): Opaque DI passthrough, or None.
    """

    def __init__(self, name: str = "", stereotypes: List[str] = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.stereotypes = stereotypes if stereotypes is not None else []
        self.layout = layout

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the Component-element label. Permits empty / whitespace /
        spaces / punctuation; ``None`` -> ``""``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"ComponentElement name must be a str or None, got {type(name).__name__}"
            )
        # Bypass NamedElement's strict identifier validation by writing its private slot.
        self._NamedElement__name = name

    @property
    def stereotypes(self) -> List[str]:
        """List[str]: Get the free-form stereotype strings."""
        return self.__stereotypes

    @stereotypes.setter
    def stereotypes(self, stereotypes: List[str]):
        """List[str]: Set the free-form stereotype strings.

        Raises:
            TypeError: if not a list of str.
        """
        self.__stereotypes = _checked_str_list(stereotypes, "stereotypes")

    @property
    def layout(self) -> Optional[dict]:
        """dict: Get the opaque DI passthrough, or None."""
        return self.__layout

    @layout.setter
    def layout(self, layout: Optional[dict]):
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
# Components (decision D11 -- everything-is-a-Component; Skill / Tool / Subsystem
# subclass it. Permission is a separate ComponentElement per A-4.)
# ---------------------------------------------------------------------------

class Component(ComponentElement):
    """A UML Component -- the canonical node in a Component diagram.

    The "everything is a Component" base. ``Skill`` / ``Tool`` / ``Subsystem``
    subclass this so they pick up the same container / relationship machinery.

    Args:
        name (str): The component label.
        agent_category (AgentCategory): The agent's role. ``NONE`` (default) for
            non-agent components (plain UML Components, external dependencies,
            Skills, Tools).
        locality (Locality): Where the component is hosted (NR-5). ``LOCAL`` by
            default.
        is_human (bool): True for a ``«actor / human»`` component (Human Developer
            in the swarm scenarios). Default False.
        realizes (List[str]): Cross-diagram IDs of structural ``Class`` es this
            component realizes (UML Realization). Default ``[]``.
        process_model_refs (List[str]): Cross-diagram IDs of BPMN ``Process`` es
            this component participates in (D3 drill-down). Default ``[]``.
        stereotypes, layout, metadata, timestamp: Inherited from ComponentElement.

    Attributes:
        agent_category (AgentCategory): The agent role.
        locality (Locality): Hosting locality.
        is_human (bool): Whether this is a human ``«actor / human»`` component.
        realizes (List[str]): Cross-diagram Class IDs.
        process_model_refs (List[str]): Cross-diagram BPMN Process IDs.
        parent (Subsystem | None): The containing Subsystem (set by the
            ``Subsystem`` container, or None if at model root).
    """

    def __init__(self, name: str = "", agent_category: AgentCategory = None,
                 locality: Locality = None, is_human: bool = False,
                 realizes: List[str] = None, process_model_refs: List[str] = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.agent_category = (agent_category
                               if agent_category is not None else AgentCategory.NONE)
        self.locality = locality if locality is not None else Locality.LOCAL
        self.is_human = is_human
        self.realizes = realizes if realizes is not None else []
        self.process_model_refs = (process_model_refs
                                   if process_model_refs is not None else [])
        self.__parent: Optional["Subsystem"] = None

    @property
    def agent_category(self) -> AgentCategory:
        """AgentCategory: Get the agent role."""
        return self.__agent_category

    @agent_category.setter
    def agent_category(self, agent_category: AgentCategory):
        """AgentCategory: Set the agent role.

        Raises:
            TypeError: if not an AgentCategory.
        """
        if not isinstance(agent_category, AgentCategory):
            raise TypeError(
                f"agent_category must be an AgentCategory, "
                f"got {type(agent_category).__name__}"
            )
        self.__agent_category = agent_category

    @property
    def locality(self) -> Locality:
        """Locality: Get the hosting locality."""
        return self.__locality

    @locality.setter
    def locality(self, locality: Locality):
        """Locality: Set the hosting locality.

        Raises:
            TypeError: if not a Locality.
        """
        if not isinstance(locality, Locality):
            raise TypeError(f"locality must be a Locality, got {type(locality).__name__}")
        self.__locality = locality

    @property
    def is_human(self) -> bool:
        """bool: Whether this is a human ``«actor / human»`` component."""
        return self.__is_human

    @is_human.setter
    def is_human(self, is_human: bool):
        """bool: Set whether this is a human ``«actor / human»`` component.

        Raises:
            TypeError: if not a bool.
        """
        if not isinstance(is_human, bool):
            raise TypeError(f"is_human must be a bool, got {type(is_human).__name__}")
        self.__is_human = is_human

    @property
    def realizes(self) -> List[str]:
        """List[str]: Cross-diagram IDs of structural Classes this component realizes."""
        return self.__realizes

    @realizes.setter
    def realizes(self, realizes: List[str]):
        """List[str]: Set the cross-diagram Class IDs.

        Raises:
            TypeError: if not a list of str.
        """
        self.__realizes = _checked_str_list(realizes, "realizes")

    @property
    def process_model_refs(self) -> List[str]:
        """List[str]: Cross-diagram IDs of BPMN Processes this component participates in."""
        return self.__process_model_refs

    @process_model_refs.setter
    def process_model_refs(self, process_model_refs: List[str]):
        """List[str]: Set the cross-diagram BPMN Process IDs.

        Raises:
            TypeError: if not a list of str.
        """
        self.__process_model_refs = _checked_str_list(
            process_model_refs, "process_model_refs"
        )

    @property
    def parent(self) -> Optional["Subsystem"]:
        """Subsystem: The containing Subsystem, or None if at model root."""
        return self.__parent

    @parent.setter
    def parent(self, parent: Optional["Subsystem"]):
        """Subsystem: Set the containing Subsystem.

        Raises:
            TypeError: if parent is neither a Subsystem nor None.
        """
        if parent is not None and not isinstance(parent, Subsystem):
            raise TypeError(
                f"parent must be a Subsystem or None, got {type(parent).__name__}"
            )
        self.__parent = parent


class Subsystem(Component):
    """A UML Subsystem -- a Component that is *also* a flow-element container.

    Apollon emits Subsystem as a separate element type (with stereotype
    ``«subsystem»``). UML 2.5 deprecated the dedicated Subsystem metaclass to a
    stereotype on Component, but we keep the subclass for round-trip clarity
    with WME's wire shape.

    Attributes:
        children (set[Component]): Child components nested in this subsystem.
            Each child's ``parent`` back-reference is maintained automatically.
    """

    def __init__(self, name: str = "", children: set = None,
                 agent_category: AgentCategory = None, locality: Locality = None,
                 is_human: bool = False, realizes: List[str] = None,
                 process_model_refs: List[str] = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, agent_category=agent_category, locality=locality,
                         is_human=is_human, realizes=realizes,
                         process_model_refs=process_model_refs,
                         stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.__children: set = set()
        self.children = children if children is not None else set()

    @property
    def children(self) -> set:
        """set[Component]: Get the child components."""
        return self.__children

    @children.setter
    def children(self, children: set):
        """set[Component]: Set the child components; maintains each child's
        ``parent`` back-reference.

        Raises:
            TypeError: if any element is not a Component.
        """
        children = set(children)
        for child in children:
            if not isinstance(child, Component):
                raise TypeError(
                    f"children must contain Component instances, "
                    f"got {type(child).__name__}"
                )
        for existing in self.__children:
            if existing.parent is self:
                existing.parent = None
        for child in children:
            child.parent = self
        self.__children = children

    def add_child(self, child: "Component"):
        """Add a child component; sets its ``parent``.

        Raises:
            TypeError: if child is not a Component.
        """
        if not isinstance(child, Component):
            raise TypeError(f"child must be a Component, got {type(child).__name__}")
        child.parent = self
        self.__children.add(child)

    def remove_child(self, child: "Component"):
        """Remove a child component; clears its ``parent``."""
        if child in self.__children:
            if child.parent is self:
                child.parent = None
            self.__children.discard(child)


class Skill(Component):
    """A capability an agent can exercise (per supervisor directive D1).

    Subclassed from ``Component`` for typed ``isinstance`` checks and future
    generator targeting (NR-4: skill -> workspace artifact at code-gen time).
    The agent-only fields inherited from ``Component`` (``agent_category``,
    ``is_human``, ``process_model_refs``) should stay at their defaults on a
    Skill -- ``ComponentModel.validate()`` warns (W6) if a user sets them.
    """


class Tool(Component):
    """A concrete external integration an agent can call (per directive D1).

    Like ``Skill``, the agent-only inherited fields should stay at their
    defaults on a Tool (W6 warns otherwise).
    """


# ---------------------------------------------------------------------------
# Interfaces and permissions
# ---------------------------------------------------------------------------

class Interface(ComponentElement):
    """A UML provided / required Interface -- a standalone named element
    referenced from ``InterfaceProvided`` / ``InterfaceRequired`` relationships
    (decision D11 -- separate class, not inlined on Component, matching UML 2.5
    and Apollon's wire shape).
    """


class Permission(ComponentElement):
    """A named permission token -- an authority a Component may carry on an
    ``AgenticEdge`` (decision D11, per A-4 of the requirements review).

    Unlike Skill / Tool, ``Permission`` is *not* a Component subclass: it never
    appears as a Component-typed endpoint of an `InterfaceProvided` /
    `InterfaceRequired` / plain `ComponentDependency`. It is only referenced from
    (a) the target of an ``AgenticEdge`` with ``kind=GRANTED``, or (b) the
    ``permissions`` list of an ``AgenticEdge`` between two agents.

    Args:
        name (str): The permission label.
        scope (str): The permission scope, e.g. ``"repo:merge:approve"``.
            Must be a non-empty string (validated construction-time).
        stereotypes, layout, metadata, timestamp: Inherited from ComponentElement.

    Attributes:
        scope (str): The permission scope.
    """

    def __init__(self, name: str = "", scope: str = "",
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.scope = scope

    @property
    def scope(self) -> str:
        """str: Get the permission scope."""
        return self.__scope

    @scope.setter
    def scope(self, scope: str):
        """str: Set the permission scope.

        Raises:
            TypeError: if scope is not a str.
        """
        if not isinstance(scope, str):
            raise TypeError(f"scope must be a str, got {type(scope).__name__}")
        self.__scope = scope

    def __repr__(self):
        return f"Permission(name='{self.name}', scope='{self.scope}')"


# ---------------------------------------------------------------------------
# Relationships (decision D11 -- abstract base with endpoint type-check seam)
# ---------------------------------------------------------------------------

class ComponentRelationship(ComponentElement):
    """Abstract base for the four Component-diagram relationship kinds.

    ``source`` and ``target`` are required (non-None). Each concrete subclass
    overrides ``_check_endpoint`` with its own endpoint type rule; the model-wide
    validator (``ComponentModel.validate()``) re-checks agentic-state rules at
    validate-time.

    Attributes:
        source (ComponentElement): The relationship's source element.
        target (ComponentElement): The relationship's target element.
    """

    def __init__(self, source, target, name: str = "",
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        if type(self) is ComponentRelationship:
            raise TypeError(
                "ComponentRelationship is abstract and cannot be instantiated directly"
            )
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.__source = None
        self.__target = None
        self.source = source
        self.target = target

    @property
    def source(self):
        """ComponentElement: Get the source element."""
        return self.__source

    @source.setter
    def source(self, source):
        """ComponentElement: Set the source element.

        Raises:
            TypeError: if source is not an acceptable endpoint for this relationship.
        """
        self._check_endpoint(source, "source")
        self.__source = source

    @property
    def target(self):
        """ComponentElement: Get the target element."""
        return self.__target

    @target.setter
    def target(self, target):
        """ComponentElement: Set the target element.

        Raises:
            TypeError: if target is not an acceptable endpoint for this relationship.
        """
        self._check_endpoint(target, "target")
        self.__target = target

    def _check_endpoint(self, endpoint, role: str):
        """Type-check a single endpoint. Base requires any ``ComponentElement``;
        subclasses tighten.

        Raises:
            TypeError: if endpoint is not acceptable.
        """
        if not isinstance(endpoint, ComponentElement):
            raise TypeError(
                f"{type(self).__name__} {role} must be a ComponentElement, "
                f"got {type(endpoint).__name__}"
            )

    def __repr__(self):
        source_name = self.source.name if self.source is not None else None
        target_name = self.target.name if self.target is not None else None
        return (f"{type(self).__name__}(name='{self.name}', "
                f"source='{source_name}', target='{target_name}')")


class InterfaceProvided(ComponentRelationship):
    """A Component -> Interface edge: the Component *provides* the Interface
    (UML 2.5 BehavioredClassifier -> realizedInterface, lollipop notation)."""

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, Component):
                raise TypeError(
                    f"InterfaceProvided source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Interface):
                raise TypeError(
                    f"InterfaceProvided target must be an Interface, "
                    f"got {type(endpoint).__name__}"
                )


class InterfaceRequired(ComponentRelationship):
    """A Component -> Interface edge: the Component *requires* the Interface
    (UML 2.5 Component -> requiredInterface, socket notation)."""

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, Component):
                raise TypeError(
                    f"InterfaceRequired source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Interface):
                raise TypeError(
                    f"InterfaceRequired target must be an Interface, "
                    f"got {type(endpoint).__name__}"
                )


class ComponentDependency(ComponentRelationship):
    """A plain UML dependency between two Components (e.g. ``«use»``,
    ``«import»``). Carries an arbitrary ``stereotypes`` list inherited from
    ``ComponentElement``.

    Endpoint rule: both source and target are ``Component`` s. Tighter than the
    abstract base's ``ComponentElement`` to exclude Interfaces and Permissions.
    """

    def _check_endpoint(self, endpoint, role: str):
        if not isinstance(endpoint, Component):
            raise TypeError(
                f"{type(self).__name__} {role} must be a Component, "
                f"got {type(endpoint).__name__}"
            )


class AgenticEdge(ComponentDependency):
    """A typed agentic dependency between agents and / or capabilities (decision
    D11 / Q11=c).

    The eight kinds (``AgenticEdgeKind``) cover the supervisor's directive D1
    edge set (``«delegates»``, ``«supervises»``, ``«revises»``, ``«collaborates»``),
    the agent -> capability wiring (``«has»``, ``«uses»``, ``«granted»``) and the
    capability composition (``«implements»``).

    Construction-time endpoint type rules (TypeError if violated)::

        DELEGATES / SUPERVISES / REVISES / COLLABORATES: Component -> Component
        HAS:        Component -> Skill
        USES:       Component -> Tool
        GRANTED:    Component -> Permission
        IMPLEMENTS: Tool -> Skill

    Runtime agentic-state rules (``ComponentModel.validate()`` errors E5..E10)::

        for agent-to-agent kinds, both endpoints must satisfy
        ``agent_category != NONE OR is_human``.

    Permissions attribute: only carried meaningfully on agent-to-agent kinds.
    W3 warns if non-agent-to-agent kinds carry non-empty ``permissions``.

    Args:
        source: The relationship's source element (see endpoint table above).
        target: The relationship's target element.
        kind (AgenticEdgeKind): The edge kind.
        permissions (List[Permission]): Permissions conferred on this edge
            (meaningful for agent <-> agent kinds; empty by default).
        name, stereotypes, layout, metadata, timestamp: Inherited.

    Attributes:
        kind (AgenticEdgeKind): The edge kind.
        permissions (List[Permission]): The permissions list.
    """

    def __init__(self, source, target, kind: AgenticEdgeKind,
                 permissions: List["Permission"] = None,
                 name: str = "", stereotypes: List[str] = None,
                 layout: dict = None, metadata=None, timestamp=None):
        if not isinstance(kind, AgenticEdgeKind):
            raise TypeError(
                f"kind must be an AgenticEdgeKind, got {type(kind).__name__}"
            )
        # Set kind BEFORE source/target so _check_endpoint can read it.
        self.__kind = kind
        self.__permissions: List["Permission"] = []
        super().__init__(source=source, target=target, name=name,
                         stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.permissions = permissions if permissions is not None else []

    @property
    def kind(self) -> AgenticEdgeKind:
        """AgenticEdgeKind: Get the edge kind."""
        return self.__kind

    @kind.setter
    def kind(self, kind: AgenticEdgeKind):
        """AgenticEdgeKind: Set the edge kind. Re-validates the current
        (source, target) against the new kind.

        Raises:
            TypeError: if not an AgenticEdgeKind, or if the current source/target
                are not consistent with the new kind.
        """
        if not isinstance(kind, AgenticEdgeKind):
            raise TypeError(
                f"kind must be an AgenticEdgeKind, got {type(kind).__name__}"
            )
        previous, self.__kind = self.__kind, kind
        try:
            # Re-check endpoints if both already set (mid-construction skip).
            if self.source is not None:
                self._check_endpoint(self.source, "source")
            if self.target is not None:
                self._check_endpoint(self.target, "target")
        except TypeError:
            self.__kind = previous
            raise

    @property
    def permissions(self) -> List["Permission"]:
        """List[Permission]: Get the permissions list."""
        return self.__permissions

    @permissions.setter
    def permissions(self, permissions: List["Permission"]):
        """List[Permission]: Set the permissions list.

        Raises:
            TypeError: if any element is not a Permission.
        """
        permissions = list(permissions)
        for permission in permissions:
            if not isinstance(permission, Permission):
                raise TypeError(
                    f"permissions must contain Permission instances, "
                    f"got {type(permission).__name__}"
                )
        self.__permissions = permissions

    def _check_endpoint(self, endpoint, role: str):
        """Per-kind construction-time endpoint type rule."""
        kind = self.__kind
        if kind == AgenticEdgeKind.HAS:
            if role == "source" and not isinstance(endpoint, Component):
                raise TypeError(
                    f"AgenticEdge[HAS] source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
            if role == "target" and not isinstance(endpoint, Skill):
                raise TypeError(
                    f"AgenticEdge[HAS] target must be a Skill, "
                    f"got {type(endpoint).__name__}"
                )
        elif kind == AgenticEdgeKind.USES:
            if role == "source" and not isinstance(endpoint, Component):
                raise TypeError(
                    f"AgenticEdge[USES] source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
            if role == "target" and not isinstance(endpoint, Tool):
                raise TypeError(
                    f"AgenticEdge[USES] target must be a Tool, "
                    f"got {type(endpoint).__name__}"
                )
        elif kind == AgenticEdgeKind.GRANTED:
            if role == "source" and not isinstance(endpoint, Component):
                raise TypeError(
                    f"AgenticEdge[GRANTED] source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
            if role == "target" and not isinstance(endpoint, Permission):
                raise TypeError(
                    f"AgenticEdge[GRANTED] target must be a Permission, "
                    f"got {type(endpoint).__name__}"
                )
        elif kind == AgenticEdgeKind.IMPLEMENTS:
            if role == "source" and not isinstance(endpoint, Tool):
                raise TypeError(
                    f"AgenticEdge[IMPLEMENTS] source must be a Tool, "
                    f"got {type(endpoint).__name__}"
                )
            if role == "target" and not isinstance(endpoint, Skill):
                raise TypeError(
                    f"AgenticEdge[IMPLEMENTS] target must be a Skill, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # agent <-> agent kinds -- type check only (state check at validate)
            if not isinstance(endpoint, Component):
                raise TypeError(
                    f"AgenticEdge[{kind.name}] {role} must be a Component, "
                    f"got {type(endpoint).__name__}"
                )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class ComponentModel(Model):
    """The root of a UML Component model -- a first-class B-UML model.

    Like ``BPMNModel``, ``ComponentModel`` does not populate the inherited
    ``Model.elements`` set; it exposes its own typed accessors.

    Args:
        name (str): The model name (free text; relaxed like ``ComponentElement``).
        components (set[Component]): All components (incl. Subsystems, Skills,
            Tools) in this model. Subsystem children also count here.
        interfaces (set[Interface]): Interfaces in this model.
        permissions (set[Permission]): Permissions in this model.
        relationships (set[ComponentRelationship]): All relationships in this
            model (incl. ``InterfaceProvided`` / ``InterfaceRequired`` / plain
            ``ComponentDependency`` / ``AgenticEdge``).

    Attributes:
        components, interfaces, permissions, relationships: as above.
    """

    def __init__(self, name: str, components: set = None,
                 interfaces: set = None, permissions: set = None,
                 relationships: set = None, metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.__components: set = set()
        self.__interfaces: set = set()
        self.__permissions: set = set()
        self.__relationships: set = set()
        self.components = components if components is not None else set()
        self.interfaces = interfaces if interfaces is not None else set()
        self.permissions = permissions if permissions is not None else set()
        self.relationships = relationships if relationships is not None else set()

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the model name. Relaxed like ``ComponentElement``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"ComponentModel name must be a str or None, got {type(name).__name__}"
            )
        self._NamedElement__name = name

    @property
    def components(self) -> set:
        """set[Component]: Get the components in this model."""
        return self.__components

    @components.setter
    def components(self, components: set):
        """set[Component]: Set the components in this model.

        Raises:
            TypeError: if any element is not a Component.
        """
        self.__components = _checked_set(components, Component, "components")

    def add_component(self, component: "Component"):
        """Add a component.

        Raises:
            TypeError: if component is not a Component.
        """
        if not isinstance(component, Component):
            raise TypeError(
                f"component must be a Component, got {type(component).__name__}"
            )
        self.__components.add(component)

    def remove_component(self, component: "Component"):
        """Remove a component."""
        self.__components.discard(component)

    @property
    def interfaces(self) -> set:
        """set[Interface]: Get the interfaces in this model."""
        return self.__interfaces

    @interfaces.setter
    def interfaces(self, interfaces: set):
        """set[Interface]: Set the interfaces in this model.

        Raises:
            TypeError: if any element is not an Interface.
        """
        self.__interfaces = _checked_set(interfaces, Interface, "interfaces")

    def add_interface(self, interface: "Interface"):
        """Add an interface.

        Raises:
            TypeError: if interface is not an Interface.
        """
        if not isinstance(interface, Interface):
            raise TypeError(
                f"interface must be an Interface, got {type(interface).__name__}"
            )
        self.__interfaces.add(interface)

    def remove_interface(self, interface: "Interface"):
        """Remove an interface."""
        self.__interfaces.discard(interface)

    @property
    def permissions(self) -> set:
        """set[Permission]: Get the permissions in this model."""
        return self.__permissions

    @permissions.setter
    def permissions(self, permissions: set):
        """set[Permission]: Set the permissions in this model.

        Raises:
            TypeError: if any element is not a Permission.
        """
        self.__permissions = _checked_set(permissions, Permission, "permissions")

    def add_permission(self, permission: "Permission"):
        """Add a permission.

        Raises:
            TypeError: if permission is not a Permission.
        """
        if not isinstance(permission, Permission):
            raise TypeError(
                f"permission must be a Permission, got {type(permission).__name__}"
            )
        self.__permissions.add(permission)

    def remove_permission(self, permission: "Permission"):
        """Remove a permission."""
        self.__permissions.discard(permission)

    @property
    def relationships(self) -> set:
        """set[ComponentRelationship]: Get the relationships in this model."""
        return self.__relationships

    @relationships.setter
    def relationships(self, relationships: set):
        """set[ComponentRelationship]: Set the relationships in this model.

        Raises:
            TypeError: if any element is not a ComponentRelationship.
        """
        self.__relationships = _checked_set(
            relationships, ComponentRelationship, "relationships"
        )

    def add_relationship(self, relationship: "ComponentRelationship"):
        """Add a relationship.

        Raises:
            TypeError: if relationship is not a ComponentRelationship.
        """
        if not isinstance(relationship, ComponentRelationship):
            raise TypeError(
                f"relationship must be a ComponentRelationship, "
                f"got {type(relationship).__name__}"
            )
        self.__relationships.add(relationship)

    def remove_relationship(self, relationship: "ComponentRelationship"):
        """Remove a relationship."""
        self.__relationships.discard(relationship)

    # --- derived accessors -------------------------------------------------

    def all_components(self) -> set:
        """set[Component]: Every component, expanded through Subsystem nesting."""
        result: set = set()

        def _collect(component: Component):
            result.add(component)
            if isinstance(component, Subsystem):
                for child in component.children:
                    _collect(child)

        for component in self.__components:
            _collect(component)
        return result

    def agentic_edges(self) -> set:
        """set[AgenticEdge]: Convenience -- every AgenticEdge in this model."""
        return {r for r in self.__relationships if isinstance(r, AgenticEdge)}

    # --- validation --------------------------------------------------------

    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the Component model according to its structural rules.

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: ``{"success": bool, "errors": list[str], "warnings": list[str]}``.
        """
        errors: list = []
        warnings: list = []

        self._validate_endpoint_references(errors)
        self._validate_relationship_endpoint_types(errors)
        self._validate_agentic_state(errors)
        self._validate_permission_membership(errors)
        self._validate_unique_component_names(errors)
        self._validate_permission_scopes(errors)
        self._validate_subsystem_membership(errors)
        self._warn_structural_smells(warnings)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _all_elements(self) -> set:
        """set[ComponentElement]: Every element a relationship may reference."""
        return (self.all_components()
                | self.__interfaces
                | self.__permissions)

    def _validate_endpoint_references(self, errors: list):
        """E1: every relationship's source / target is reachable in this model."""
        universe = self._all_elements()
        for rel in self.__relationships:
            for role, endpoint in (("source", rel.source), ("target", rel.target)):
                if endpoint is None:
                    errors.append(
                        f"{type(rel).__name__} '{rel.name}' has no {role}."
                    )
                elif endpoint not in universe:
                    errors.append(
                        f"{type(rel).__name__} '{rel.name}' {role} "
                        f"'{endpoint.name}' ({type(endpoint).__name__}) is "
                        f"not present in the model."
                    )

    def _validate_relationship_endpoint_types(self, errors: list):
        """E2-E4 + E6-E9: per-kind endpoint type rules.

        Construction-time setters already enforce these, but a user could mutate
        ``kind`` post-construction (re-validated by the setter) or assemble an
        invalid model by manually wiring ``source`` / ``target`` through the
        private slots. validate() is the second line of defence.
        """
        for rel in self.__relationships:
            if isinstance(rel, AgenticEdge):
                kind = rel.kind
                if kind == AgenticEdgeKind.HAS:
                    if not (isinstance(rel.source, Component)
                            and isinstance(rel.target, Skill)):
                        errors.append(
                            f"AgenticEdge[HAS] '{rel.name}' must be "
                            f"Component -> Skill."
                        )
                elif kind == AgenticEdgeKind.USES:
                    if not (isinstance(rel.source, Component)
                            and isinstance(rel.target, Tool)):
                        errors.append(
                            f"AgenticEdge[USES] '{rel.name}' must be "
                            f"Component -> Tool."
                        )
                elif kind == AgenticEdgeKind.GRANTED:
                    if not (isinstance(rel.source, Component)
                            and isinstance(rel.target, Permission)):
                        errors.append(
                            f"AgenticEdge[GRANTED] '{rel.name}' must be "
                            f"Component -> Permission."
                        )
                elif kind == AgenticEdgeKind.IMPLEMENTS:
                    if not (isinstance(rel.source, Tool)
                            and isinstance(rel.target, Skill)):
                        errors.append(
                            f"AgenticEdge[IMPLEMENTS] '{rel.name}' must be "
                            f"Tool -> Skill."
                        )
                else:  # agent <-> agent
                    if not (isinstance(rel.source, Component)
                            and isinstance(rel.target, Component)):
                        errors.append(
                            f"AgenticEdge[{kind.name}] '{rel.name}' must connect "
                            f"two Components."
                        )
            elif isinstance(rel, InterfaceProvided):
                if not (isinstance(rel.source, Component)
                        and isinstance(rel.target, Interface)):
                    errors.append(
                        f"InterfaceProvided '{rel.name}' must be "
                        f"Component -> Interface."
                    )
            elif isinstance(rel, InterfaceRequired):
                if not (isinstance(rel.source, Component)
                        and isinstance(rel.target, Interface)):
                    errors.append(
                        f"InterfaceRequired '{rel.name}' must be "
                        f"Component -> Interface."
                    )
            elif isinstance(rel, ComponentDependency):
                if not (isinstance(rel.source, Component)
                        and isinstance(rel.target, Component)):
                    errors.append(
                        f"ComponentDependency '{rel.name}' must connect "
                        f"two Components."
                    )

    def _validate_agentic_state(self, errors: list):
        """E5: agent-to-agent edges require both endpoints to be agents
        (``agent_category != NONE`` OR ``is_human``)."""
        for edge in self.agentic_edges():
            if edge.kind in _AGENT_TO_AGENT_KINDS:
                if not (isinstance(edge.source, Component)
                        and _is_agent_endpoint(edge.source)):
                    errors.append(
                        f"AgenticEdge[{edge.kind.name}] '{edge.name}' source "
                        f"'{edge.source.name}' is not an agent "
                        f"(agent_category=NONE and is_human=False)."
                    )
                if not (isinstance(edge.target, Component)
                        and _is_agent_endpoint(edge.target)):
                    errors.append(
                        f"AgenticEdge[{edge.kind.name}] '{edge.name}' target "
                        f"'{edge.target.name}' is not an agent "
                        f"(agent_category=NONE and is_human=False)."
                    )

    def _validate_permission_membership(self, errors: list):
        """E10: every ``Permission`` carried by an ``AgenticEdge`` is in this
        model's permissions set."""
        for edge in self.agentic_edges():
            for permission in edge.permissions:
                if permission not in self.__permissions:
                    errors.append(
                        f"AgenticEdge[{edge.kind.name}] '{edge.name}' carries "
                        f"Permission '{permission.name}' which is not in the model."
                    )

    def _validate_unique_component_names(self, errors: list):
        """E11: Component names within a model are unique."""
        seen: dict = {}
        for component in self.all_components():
            name = component.name
            if name == "":
                continue
            if name in seen:
                errors.append(
                    f"Duplicate component name '{name}': {type(seen[name]).__name__} "
                    f"and {type(component).__name__}."
                )
            else:
                seen[name] = component

    def _validate_permission_scopes(self, errors: list):
        """E12: ``Permission.scope`` is non-empty."""
        for permission in self.__permissions:
            if permission.scope == "":
                errors.append(
                    f"Permission '{permission.name}' has an empty scope."
                )

    def _validate_subsystem_membership(self, errors: list):
        """E13: ``Component.parent``, if set, points at a Subsystem that is in
        the model's components."""
        for component in self.all_components():
            parent = component.parent
            if parent is not None and parent not in self.all_components():
                errors.append(
                    f"Component '{component.name}' has parent Subsystem "
                    f"'{parent.name}' which is not in the model."
                )

    def _warn_structural_smells(self, warnings: list):
        """W1-W7: non-blocking structural smells.

        W1: agent component with zero outgoing HAS / USES edges.
        W2: agent component with zero process_model_refs.
        W3: non-agent-to-agent edge carrying non-empty permissions (NR-3).
        W4: dangling Component.realizes ID (cross-diagram, can't be checked here
            -- left as a project-level promotion in 04-).
        W5: dangling Component.process_model_refs (same -- project-level).
        W6: Skill / Tool with agent-only fields set non-default.
        W7: Interface with no incoming InterfaceProvided / InterfaceRequired.
        """
        # Build outgoing-by-kind index.
        outgoing_by_kind: dict = {}
        for edge in self.agentic_edges():
            outgoing_by_kind.setdefault(edge.source, set()).add(edge.kind)

        for component in self.all_components():
            if isinstance(component, (Skill, Tool)):
                # W6 -- agent fields should be default on Skill / Tool.
                if component.agent_category is not AgentCategory.NONE:
                    warnings.append(
                        f"{type(component).__name__} '{component.name}' has "
                        f"agent_category={component.agent_category.name}; "
                        f"agent-only fields should be default on Skill / Tool."
                    )
                if component.is_human:
                    warnings.append(
                        f"{type(component).__name__} '{component.name}' has "
                        f"is_human=True; agent-only fields should be default "
                        f"on Skill / Tool."
                    )
                continue
            if _is_agent_endpoint(component):
                # W1 -- agent with no HAS / USES outgoing.
                kinds = outgoing_by_kind.get(component, set())
                if not (kinds & {AgenticEdgeKind.HAS, AgenticEdgeKind.USES}):
                    warnings.append(
                        f"Component '{component.name}' is an agent but has no "
                        f"outgoing «has» / «uses» edges."
                    )
                # W2 -- agent with no process_model_refs.
                if not component.process_model_refs:
                    warnings.append(
                        f"Component '{component.name}' is an agent but appears in "
                        f"no BPMN process (process_model_refs is empty)."
                    )

        # W3 -- non-agent-to-agent kinds carrying permissions.
        for edge in self.agentic_edges():
            if edge.permissions and edge.kind not in _AGENT_TO_AGENT_KINDS:
                warnings.append(
                    f"AgenticEdge[{edge.kind.name}] '{edge.name}' carries "
                    f"{len(edge.permissions)} permission(s); permissions are "
                    f"meaningful only on agent <-> agent kinds."
                )

        # W7 -- orphan interface.
        used_interfaces: set = set()
        for rel in self.__relationships:
            if isinstance(rel, (InterfaceProvided, InterfaceRequired)):
                used_interfaces.add(rel.target)
        for interface in self.__interfaces:
            if interface not in used_interfaces:
                warnings.append(
                    f"Interface '{interface.name}' has no incoming "
                    f"InterfaceProvided / InterfaceRequired (orphan interface)."
                )

    def __repr__(self):
        return (f"ComponentModel(name='{self.name}', "
                f"components={len(self.components)}, "
                f"interfaces={len(self.interfaces)}, "
                f"permissions={len(self.permissions)}, "
                f"relationships={len(self.relationships)})")
