"""UML Component agentic extension -- the collaborative-agent swarm profile.

A stereotype-as-subclass extension of the base UML Component metamodel
(``uml_component.py``). It is a **pure addition**: the base module is not
modified and remains valid vanilla UML 2.5 modelling on its own. This mirrors
the BPMN track's ``bpmn/bpmn.py`` (base) + ``bpmn/agentic.py`` (extension)
split.

Implements the design in ``.claude/component-deployment/01-...`` (decision D3,
the hybrid stereotype profile) as restructured by ``04-...`` (2026-05-20): the
profile was originally baked into ``uml_component.py``; ``04-`` extracted it
here so the base is citable as pure UML.

Adds:

* ``AgenticComponent`` -- a ``Component`` that is an agent (carries an
  ``agent_category`` role and ``process_model_refs`` cross-diagram links to
  the BPMN processes it participates in).
* ``Skill`` / ``Tool`` -- agent capabilities (plain ``Component`` subclasses;
  agentic-notation vocabulary).
* ``Permission`` -- an authority carried on an ``AgenticEdge``.
* ``AgenticEdge`` -- a typed dependency between agents and / or capabilities
  (``AgenticEdgeKind``: «delegates» / «supervises» / «revises» / «collaborates»
  / «has» / «uses» / «granted» / «implements»).
* ``AgenticComponentModel`` -- a ``ComponentModel`` that also holds the
  ``permissions`` collection and runs the agentic validation rules.

The two enums' ``.value`` strings match the WME ``stereotype`` wire tokens so
the JSON converters stay a string-passthrough.
"""

from enum import Enum
from typing import List

from besser.BUML.metamodel.uml_component.uml_component import (
    Component,
    ComponentDependency,
    ComponentElement,
    ComponentModel,
)


# ---------------------------------------------------------------------------
# Module-private helpers (duplicated from uml_component.py so this extension
# module is self-contained -- mirrors the uml_deployment.py precedent)
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


# ---------------------------------------------------------------------------
# Enumerations (decision D3 -- plain enum.Enum; .value strings match the WME
# stereotype strings so converters can map by value)
# ---------------------------------------------------------------------------

class AgentCategory(Enum):
    """The agent's role in a multi-agent governance pattern.

    ``NONE`` marks an ``AgenticComponent`` whose role has not yet been
    specified (it is still an agent -- being an ``AgenticComponent`` is what
    makes it one). The other four values describe agent roles common to
    multi-agent governance patterns (Solution does domain work, Supervision
    evaluates / approves, Consensus mediates merge / decision, Collaboration
    orchestrates dispatch). Not tied to any specific agent framework or
    taxonomy -- worked examples (e.g. MOSAICO UNP) are use-case-level
    illustrations only.
    """
    NONE = "none"
    SOLUTION = "solution"
    SUPERVISION = "supervision"
    CONSENSUS = "consensus"
    COLLABORATION = "collaboration"


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
# Agentic components (decision D11, restructured by 04- -- the agent profile
# is an AgenticComponent subclass rather than fields on the base Component)
# ---------------------------------------------------------------------------

class AgenticComponent(Component):
    """An agent in a collaborative-agent swarm.

    A ``Component`` carrying the agentic profile (decision D3 typed-slot
    carriers). The base ``Component`` is pure UML 2.5; ``AgenticComponent``
    adds the swarm intent: an agent role and cross-diagram references to the
    BPMN processes the agent participates in.

    Args:
        name (str): The component label.
        agent_category (AgentCategory): The agent's role. ``NONE`` (default)
            means "agent, role not yet specified".
        process_model_refs (List[str]): Cross-diagram IDs of BPMN ``Process`` es
            this agent participates in (D3 drill-down). Default ``[]``.
        locality, realizes, stereotypes, layout, metadata, timestamp: Inherited
            from ``Component``.

    Attributes:
        agent_category (AgentCategory): The agent role.
        process_model_refs (List[str]): Cross-diagram BPMN Process IDs.
    """

    def __init__(self, name: str = "", agent_category: AgentCategory = None,
                 process_model_refs: List[str] = None,
                 locality=None, realizes: List[str] = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, locality=locality, realizes=realizes,
                         stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.agent_category = (agent_category
                               if agent_category is not None
                               else AgentCategory.NONE)
        self.process_model_refs = (process_model_refs
                                   if process_model_refs is not None else [])

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
    def process_model_refs(self) -> List[str]:
        """List[str]: Cross-diagram IDs of BPMN Processes this agent participates in."""
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


class Skill(Component):
    """A capability an agent can exercise (per supervisor directive D1).

    A plain ``Component`` (not an agent -- it carries no agentic profile);
    subclassed for typed ``isinstance`` checks and future generator targeting
    (NR-4: skill -> workspace artifact at code-gen time). It is part of the
    agentic-notation vocabulary, hence its home in ``agentic.py``.
    """


class Tool(Component):
    """A concrete external integration an agent can call (per directive D1).

    Like ``Skill``, a plain ``Component`` -- a capability, not an agent.
    """


# ---------------------------------------------------------------------------
# Permission (per A-4 of the requirements review -- a separate ComponentElement)
# ---------------------------------------------------------------------------

class Permission(ComponentElement):
    """A named permission token -- an authority a Component may carry on an
    ``AgenticEdge`` (per A-4 of the requirements review).

    Not a ``Component`` subclass: it never appears as a Component-typed
    endpoint of an ``InterfaceProvided`` / ``InterfaceRequired`` / plain
    ``ComponentDependency``. It is referenced only from (a) the target of an
    ``AgenticEdge`` with ``kind=GRANTED``, or (b) the ``permissions`` list of
    an ``AgenticEdge`` between two agents.

    Args:
        name (str): The permission label.
        scope (str): The permission scope, e.g. ``"repo:merge:approve"``.
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


def _is_agent_endpoint(component) -> bool:
    """True if ``component`` is a valid agent endpoint for an ``AgenticEdge``.

    Post-04- (the base/agentic split): an agent *is* an ``AgenticComponent``.
    """
    return isinstance(component, AgenticComponent)


# ---------------------------------------------------------------------------
# Agentic relationship
# ---------------------------------------------------------------------------

class AgenticEdge(ComponentDependency):
    """A typed agentic dependency between agents and / or capabilities (decision
    D11 / Q11=c).

    The eight kinds (``AgenticEdgeKind``) cover the supervisor's directive D1
    edge set (``«delegates»``, ``«supervises»``, ``«revises»``, ``«collaborates»``),
    the agent -> capability wiring (``«has»``, ``«uses»``, ``«granted»``) and the
    capability composition (``«implements»``).

    Construction-time endpoint type rules (TypeError if violated)::

        DELEGATES / SUPERVISES / REVISES / COLLABORATES:
            AgenticComponent -> AgenticComponent
        HAS:        AgenticComponent -> Skill
        USES:       AgenticComponent -> Tool
        GRANTED:    AgenticComponent -> Permission
        IMPLEMENTS: Tool -> Skill

    Because "is an agent" is now the ``AgenticComponent`` type, the old runtime
    agentic-state check (E5) is fully enforced at construction time -- the
    ``AgenticComponentModel`` validator re-checks it only as a second line of
    defence against a model wired through the private slots.

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
            if role == "source" and not isinstance(endpoint, AgenticComponent):
                raise TypeError(
                    f"AgenticEdge[HAS] source must be an AgenticComponent, "
                    f"got {type(endpoint).__name__}"
                )
            if role == "target" and not isinstance(endpoint, Skill):
                raise TypeError(
                    f"AgenticEdge[HAS] target must be a Skill, "
                    f"got {type(endpoint).__name__}"
                )
        elif kind == AgenticEdgeKind.USES:
            if role == "source" and not isinstance(endpoint, AgenticComponent):
                raise TypeError(
                    f"AgenticEdge[USES] source must be an AgenticComponent, "
                    f"got {type(endpoint).__name__}"
                )
            if role == "target" and not isinstance(endpoint, Tool):
                raise TypeError(
                    f"AgenticEdge[USES] target must be a Tool, "
                    f"got {type(endpoint).__name__}"
                )
        elif kind == AgenticEdgeKind.GRANTED:
            if role == "source" and not isinstance(endpoint, AgenticComponent):
                raise TypeError(
                    f"AgenticEdge[GRANTED] source must be an AgenticComponent, "
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
        else:  # agent <-> agent kinds
            if not isinstance(endpoint, AgenticComponent):
                raise TypeError(
                    f"AgenticEdge[{kind.name}] {role} must be an "
                    f"AgenticComponent, got {type(endpoint).__name__}"
                )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class AgenticComponentModel(ComponentModel):
    """The root of an *agentic* UML Component model.

    Extends the pure-UML ``ComponentModel`` with the agentic profile's
    model-level concerns: the ``permissions`` collection and the agentic
    validation rules (agent-endpoint typing, permission membership / scope,
    and the agentic structural-smell warnings). The converters build an
    ``AgenticComponentModel`` whenever a diagram contains any agentic element
    (``AgenticComponent`` / ``Skill`` / ``Tool`` / ``Permission`` /
    ``AgenticEdge``); a pure-UML diagram yields a plain ``ComponentModel``.

    Args:
        name (str): The model name.
        components (set[Component]): Components in this model.
        interfaces (set[Interface]): Interfaces in this model.
        permissions (set[Permission]): Permissions in this model.
        relationships (set[ComponentRelationship]): Relationships in this model.

    Attributes:
        permissions (set[Permission]): The model's permissions.
    """

    def __init__(self, name: str, components: set = None,
                 interfaces: set = None, permissions: set = None,
                 relationships: set = None, metadata=None, timestamp=None):
        super().__init__(name=name, components=components, interfaces=interfaces,
                         relationships=relationships,
                         metadata=metadata, timestamp=timestamp)
        self.__permissions: set = set()
        self.permissions = permissions if permissions is not None else set()

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

    def agentic_edges(self) -> set:
        """set[AgenticEdge]: Convenience -- every AgenticEdge in this model."""
        return {r for r in self.relationships if isinstance(r, AgenticEdge)}

    def _all_elements(self) -> set:
        """set[ComponentElement]: base elements plus this model's permissions."""
        return super()._all_elements() | self.__permissions

    # --- validation --------------------------------------------------------

    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the agentic Component model.

        Runs the base ``ComponentModel`` structural checks, then appends the
        agentic checks (agentic-edge endpoint typing, permission membership /
        scope, and the W1/W2/W3 agentic smells).

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: ``{"success": bool, "errors": list[str], "warnings": list[str]}``.
        """
        result = super().validate(raise_exception=False)
        errors = result["errors"]
        warnings = result["warnings"]

        self._validate_agentic_endpoint_types(errors)
        self._validate_permission_membership(errors)
        self._validate_permission_scopes(errors)
        self._warn_agentic_smells(warnings)

        result["success"] = len(errors) == 0
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _validate_agentic_endpoint_types(self, errors: list):
        """E5-E9: per-kind endpoint type rules for AgenticEdge.

        Construction-time setters already enforce these; this is the second
        line of defence against a model wired through the private slots. The
        agent-to-agent rule subsumes the old agentic-state check (E5): an agent
        *is* an ``AgenticComponent``.
        """
        for edge in self.agentic_edges():
            kind = edge.kind
            if kind == AgenticEdgeKind.HAS:
                if not (isinstance(edge.source, AgenticComponent)
                        and isinstance(edge.target, Skill)):
                    errors.append(
                        f"AgenticEdge[HAS] '{edge.name}' must be "
                        f"AgenticComponent -> Skill."
                    )
            elif kind == AgenticEdgeKind.USES:
                if not (isinstance(edge.source, AgenticComponent)
                        and isinstance(edge.target, Tool)):
                    errors.append(
                        f"AgenticEdge[USES] '{edge.name}' must be "
                        f"AgenticComponent -> Tool."
                    )
            elif kind == AgenticEdgeKind.GRANTED:
                if not (isinstance(edge.source, AgenticComponent)
                        and isinstance(edge.target, Permission)):
                    errors.append(
                        f"AgenticEdge[GRANTED] '{edge.name}' must be "
                        f"AgenticComponent -> Permission."
                    )
            elif kind == AgenticEdgeKind.IMPLEMENTS:
                if not (isinstance(edge.source, Tool)
                        and isinstance(edge.target, Skill)):
                    errors.append(
                        f"AgenticEdge[IMPLEMENTS] '{edge.name}' must be "
                        f"Tool -> Skill."
                    )
            else:  # agent <-> agent
                if not (isinstance(edge.source, AgenticComponent)
                        and isinstance(edge.target, AgenticComponent)):
                    errors.append(
                        f"AgenticEdge[{kind.name}] '{edge.name}' must connect "
                        f"two AgenticComponents."
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

    def _validate_permission_scopes(self, errors: list):
        """E12: ``Permission.scope`` is non-empty."""
        for permission in self.__permissions:
            if permission.scope == "":
                errors.append(
                    f"Permission '{permission.name}' has an empty scope."
                )

    def _warn_agentic_smells(self, warnings: list):
        """W1-W3: non-blocking agentic structural smells.

        W1: AgenticComponent with zero outgoing «has» / «uses» edges.
        W2: AgenticComponent with no process_model_refs.
        W3: non-agent-to-agent edge carrying non-empty permissions (NR-3).
        """
        # Build outgoing-by-kind index.
        outgoing_by_kind: dict = {}
        for edge in self.agentic_edges():
            outgoing_by_kind.setdefault(edge.source, set()).add(edge.kind)

        for component in self.all_components():
            if not isinstance(component, AgenticComponent):
                continue
            # W1 -- agent with no HAS / USES outgoing.
            kinds = outgoing_by_kind.get(component, set())
            if not (kinds & {AgenticEdgeKind.HAS, AgenticEdgeKind.USES}):
                warnings.append(
                    f"AgenticComponent '{component.name}' has no outgoing "
                    f"«has» / «uses» edges."
                )
            # W2 -- agent with no process_model_refs.
            if not component.process_model_refs:
                warnings.append(
                    f"AgenticComponent '{component.name}' appears in no BPMN "
                    f"process (process_model_refs is empty)."
                )

        # W3 -- non-agent-to-agent kinds carrying permissions.
        for edge in self.agentic_edges():
            if edge.permissions and edge.kind not in _AGENT_TO_AGENT_KINDS:
                warnings.append(
                    f"AgenticEdge[{edge.kind.name}] '{edge.name}' carries "
                    f"{len(edge.permissions)} permission(s); permissions are "
                    f"meaningful only on agent <-> agent kinds."
                )

    def __repr__(self):
        return (f"AgenticComponentModel(name='{self.name}', "
                f"components={len(self.components)}, "
                f"interfaces={len(self.interfaces)}, "
                f"permissions={len(self.permissions)}, "
                f"relationships={len(self.relationships)})")
