"""SEAA'25 BPMN agentic extension.

A stereotype-as-subclass extension of the BPMN base metamodel, adding three
classes from the SEAA'25 BPMN paper (Ait et al., SEAA'25 -- *Towards Modeling
Human-Agentic Collaborative Workflows: A BPMN Extension*):

* ``AgenticTask`` -- a ``Task`` with a reflection marker (paper §4.2, Fig 3b).
* ``AgenticGateway`` -- a ``Gateway`` with collaboration semantics; restricted
  to PARALLEL / INCLUSIVE gateway types per paper §4.3 (AgenticAND / AgenticOR).
* ``AgenticLane`` -- a ``Lane`` with an agent role + trust score (paper §4.1,
  Fig 3a).

The base BPMN module (``bpmn.py``) is **not modified** -- this file is a pure
addition. Implements the design in ``.claude/bpmn-agentic/01-…`` v3 (locked
2026-05-19) following the build plan in
``.claude/bpmn-agentic/02-seaa25-metamodel-extension-guide.md``.

The five enums and their ``.value`` strings mirror Adem's WME fork on branch
``dev/bpmn`` (``packages/editor/src/main/packages/bpmn/common/types.ts``) so
the future JSON converter (``03-…``) is a string-passthrough.

``AgenticGateway`` maintains two tri-attribute invariants via auto-managing
setters (see ``__doc__`` on the class), so the common ergonomic path
(change ``collaboration_mode`` and let ``merging_strategy`` reset to a legal
default) is single-statement.
"""

from enum import Enum

from besser.BUML.metamodel.bpmn.bpmn import (
    Gateway,
    GatewayType,
    Lane,
    MessageFlow,
    Task,
)


# ---------------------------------------------------------------------------
# Enumerations (values mirror WME's TypeScript enums verbatim)
# ---------------------------------------------------------------------------

class ReflectionMode(Enum):
    """SEAA'25 «AgenticTask» reflection mode (paper §4.2).

    The paper distinguishes SelfReflection / CrossReflection / HumanReflection
    subclasses; flattened here to a single attribute enum (matching WME's
    ``BPMNReflectionMode``). ``NONE`` = «AgenticTask» without a reflective loop.
    """
    NONE = "none"
    SELF = "self"
    CROSS = "cross"
    HUMAN = "human"


class CollaborationMode(Enum):
    """SEAA'25 «AgenticGateway» collaboration mode (paper §4.3, Fig 4).

    Cooperation hierarchy (Voting / Role / Debate) plus Competition.
    """
    VOTING = "voting"
    ROLE = "role"
    DEBATE = "debate"
    COMPETITION = "competition"


class MergingStrategy(Enum):
    """SEAA'25 «AgenticGateway» merging strategy (paper §4.3, Table 2).

    Legality per collaboration mode is enforced via
    ``_LEGAL_MERGING_STRATEGIES`` and the ``AgenticGateway`` setters.
    """
    MAJORITY = "majority"
    ABSOLUTE_MAJORITY = "absolute-majority"
    MINORITY = "minority"
    LEADER_DRIVEN = "leader-driven"
    COMPOSED = "composed"
    FASTEST = "fastest"
    MOST_COMPLETE = "most-complete"


class GatewayRole(Enum):
    """SEAA'25 «AgenticGateway» role (paper §4.3).

    Diverging-vs-merging axis of an agentic gateway pair. Maintains the
    invariant ``merging_strategy is None ⇔ gateway_role == DIVERGING`` via
    ``AgenticGateway``'s setters.
    """
    DIVERGING = "diverging"
    MERGING = "merging"


class AgentRole(Enum):
    """SEAA'25 «AgenticLane» profile role (paper §4.1).

    The paper notes the enum is extensible (e.g. ``"coder"``); kept minimal
    here for the foundation. Add new members as the paper / WME grow them.
    """
    WORKER = "worker"
    MANAGER = "manager"


# ---------------------------------------------------------------------------
# Lookup tables (module-private)
# ---------------------------------------------------------------------------

# Legal merging strategies per collaboration mode. Mirrors WME's
# ``mergingStrategiesFor`` (paper §4.3, Table 2). Debate borrows from
# voting+role per the paper's last paragraph in §4.3.
_LEGAL_MERGING_STRATEGIES = {
    CollaborationMode.VOTING: frozenset({
        MergingStrategy.MAJORITY,
        MergingStrategy.ABSOLUTE_MAJORITY,
        MergingStrategy.MINORITY,
    }),
    CollaborationMode.ROLE: frozenset({
        MergingStrategy.LEADER_DRIVEN,
        MergingStrategy.COMPOSED,
    }),
    CollaborationMode.COMPETITION: frozenset({
        MergingStrategy.FASTEST,
        MergingStrategy.MOST_COMPLETE,
    }),
    CollaborationMode.DEBATE: frozenset({
        MergingStrategy.MAJORITY,
        MergingStrategy.ABSOLUTE_MAJORITY,
        MergingStrategy.MINORITY,
        MergingStrategy.LEADER_DRIVEN,
        MergingStrategy.COMPOSED,
    }),
}


# The default each collaboration mode picks when ``merging_strategy`` needs
# to be auto-set (e.g. flipping ``gateway_role`` from DIVERGING to MERGING).
_DEFAULT_MERGING_STRATEGY = {
    CollaborationMode.VOTING: MergingStrategy.MAJORITY,
    CollaborationMode.ROLE: MergingStrategy.LEADER_DRIVEN,
    CollaborationMode.COMPETITION: MergingStrategy.FASTEST,
    CollaborationMode.DEBATE: MergingStrategy.MAJORITY,
}


# AgenticGateway is restricted to PARALLEL and INCLUSIVE gateway types per
# paper §4.3 (AgenticAND / AgenticOR). EXCLUSIVE / COMPLEX / EVENT_BASED are
# excluded -- the override on ``AgenticGateway.gateway_type`` enforces this.
_AGENTIC_ELIGIBLE_GATEWAY_TYPES = frozenset({
    GatewayType.PARALLEL,
    GatewayType.INCLUSIVE,
})


def _validate_trust_score(value) -> int:
    """Validate a trust score: ``int`` in ``[0, 100]``.

    Raises:
        TypeError: if not an int (excluding bool, which is technically int).
        ValueError: if out of range.

    Returns the value unchanged on success.
    """
    # bool is a subclass of int in Python; reject explicitly so users don't
    # accidentally set trust_score=True (which would coerce to 1 silently).
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"trust_score must be an int, got {type(value).__name__}"
        )
    if not 0 <= value <= 100:
        raise ValueError(f"trust_score must be in [0, 100], got {value}")
    return value


# ---------------------------------------------------------------------------
# AgenticTask
# ---------------------------------------------------------------------------

class AgenticTask(Task):
    """SEAA'25 «AgenticTask» -- a Task with a reflection marker (paper §4.2, Fig 3b).

    Args:
        name (str): The task label (may be empty; inherited from BPMNElement).
        reflection_mode (ReflectionMode): The reflection mode (default ``NONE``,
            meaning «AgenticTask» without a reflective loop).
        trust_score (int): The trust score, 0-100 (default 0).
        collaboration_mode (CollaborationMode): The task's collaboration mode
            (default ``VOTING``). A WME 04D1 D-D1 extension **beyond** paper
            §4.2 Fig 3b -- independent of any gateway; no merging-strategy
            invariant applies to a task's collaboration mode.
        agent_diagram_ref (str | None): Opaque id of the AgentDiagram this
            task's agent behavior is defined by (SEAA'25 cross-diagram link,
            WME guide 11). Default None. Pass-through -- no UUID validation,
            no resolution. Canonical carrier; supersedes the lane carrier of
            WME 08 (which is kept on ``AgenticLane`` for legacy tolerance).
        task_type (TaskType): Inherited from Task.
        loop_characteristics (LoopCharacteristics): Inherited from Activity.
        layout (dict): Inherited (opaque DI passthrough).
        metadata, timestamp: Inherited.

    Attributes:
        reflection_mode (ReflectionMode): The reflection mode.
        trust_score (int): The trust score.
        collaboration_mode (CollaborationMode): The collaboration mode.
        agent_diagram_ref (str | None): The AgentDiagram reference.
    """

    def __init__(self, name: str = "", reflection_mode: "ReflectionMode" = None,
                 trust_score: int = 0,
                 collaboration_mode: "CollaborationMode" = None,
                 agent_diagram_ref: str = None,
                 task_type=None, loop_characteristics=None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, task_type=task_type,
                         loop_characteristics=loop_characteristics,
                         layout=layout, metadata=metadata, timestamp=timestamp)
        self.reflection_mode = (reflection_mode if reflection_mode is not None
                                else ReflectionMode.NONE)
        self.trust_score = trust_score
        self.collaboration_mode = (collaboration_mode if collaboration_mode is not None
                                   else CollaborationMode.VOTING)
        self.agent_diagram_ref = agent_diagram_ref

    @property
    def reflection_mode(self) -> "ReflectionMode":
        """ReflectionMode: Get the reflection mode."""
        return self.__reflection_mode

    @reflection_mode.setter
    def reflection_mode(self, value: "ReflectionMode"):
        """ReflectionMode: Set the reflection mode.

        Raises:
            TypeError: if not a ReflectionMode.
        """
        if not isinstance(value, ReflectionMode):
            raise TypeError(
                f"reflection_mode must be a ReflectionMode, got {type(value).__name__}"
            )
        self.__reflection_mode = value

    @property
    def trust_score(self) -> int:
        """int: Get the trust score (0-100)."""
        return self.__trust_score

    @trust_score.setter
    def trust_score(self, value: int):
        """int: Set the trust score.

        Raises:
            TypeError: if not an int.
            ValueError: if out of [0, 100].
        """
        self.__trust_score = _validate_trust_score(value)

    @property
    def collaboration_mode(self) -> "CollaborationMode":
        """CollaborationMode: Get the task's collaboration mode.

        WME 04D1 D-D1 extension beyond paper §4.2 -- independent of any gateway;
        no merging-strategy invariant applies to a task's collaboration mode.
        """
        return self.__collaboration_mode

    @collaboration_mode.setter
    def collaboration_mode(self, value: "CollaborationMode"):
        """CollaborationMode: Set the task's collaboration mode.

        Raises:
            TypeError: if not a CollaborationMode.
        """
        if not isinstance(value, CollaborationMode):
            raise TypeError(
                f"collaboration_mode must be a CollaborationMode, "
                f"got {type(value).__name__}"
            )
        self.__collaboration_mode = value

    @property
    def agent_diagram_ref(self):
        """str | None: Get the opaque id of the AgentDiagram this task's agent
        behavior is defined by (SEAA'25 cross-diagram link, WME guide 11 --
        canonical carrier; supersedes the lane carrier of WME 08). ``None`` when unset."""
        return self.__agent_diagram_ref

    @agent_diagram_ref.setter
    def agent_diagram_ref(self, value):
        """str | None: Set the AgentDiagram reference.

        Opaque pass-through -- no UUID-format check, no typed resolution (the
        value is whatever id the project assigned the AgentDiagram).

        Raises:
            TypeError: if not a str or None.
        """
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"agent_diagram_ref must be a str or None, got {type(value).__name__}"
            )
        self.__agent_diagram_ref = value

    def __repr__(self):
        return (f"AgenticTask(name='{self.name}', "
                f"reflection_mode={self.reflection_mode}, "
                f"trust_score={self.trust_score}, "
                f"collaboration_mode={self.collaboration_mode}, "
                f"agent_diagram_ref={self.agent_diagram_ref!r})")


# ---------------------------------------------------------------------------
# AgenticGateway
# ---------------------------------------------------------------------------

class AgenticGateway(Gateway):
    """SEAA'25 «AgenticGateway» -- a Gateway with collaboration semantics (paper §4.3, Fig 4).

    Per paper §4.3, only PARALLEL (AgenticAND) and INCLUSIVE (AgenticOR) gateways
    may be agentic; the inherited ``gateway_type`` setter is overridden to
    enforce this.

    Tri-attribute invariants (auto-maintained by setters):

    * **INV-A:** ``merging_strategy is None`` ⇔ ``gateway_role == DIVERGING``.
    * **INV-B:** when ``merging_strategy`` is not None, it must be in
      ``_LEGAL_MERGING_STRATEGIES[collaboration_mode]``.

    The setters auto-manage these so the common path is single-statement:
    flipping ``gateway_role`` to MERGING auto-sets ``merging_strategy`` to the
    legal default for the current ``collaboration_mode``; changing
    ``collaboration_mode`` resets ``merging_strategy`` to the new mode's
    default if the previous value is no longer legal.

    Args:
        name (str): The gateway label (may be empty).
        gateway_type (GatewayType): PARALLEL or INCLUSIVE only (default PARALLEL).
        gateway_role (GatewayRole): DIVERGING or MERGING (default DIVERGING).
        collaboration_mode (CollaborationMode): default VOTING.
        merging_strategy (MergingStrategy): None for diverging gateways
            (auto-cleared); auto-set for merging gateways if omitted. Explicit
            value validated against ``_LEGAL_MERGING_STRATEGIES``.
        trust_score (int): 0-100 (default 0).
        layout, metadata, timestamp: Inherited.

    Attributes:
        gateway_role (GatewayRole): The gateway role.
        collaboration_mode (CollaborationMode): The collaboration mode.
        merging_strategy (MergingStrategy | None): The merging strategy.
        trust_score (int): The trust score.
    """

    def __init__(self, name: str = "", gateway_type: "GatewayType" = None,
                 gateway_role: "GatewayRole" = None,
                 collaboration_mode: "CollaborationMode" = None,
                 merging_strategy: "MergingStrategy" = None,
                 trust_score: int = 0,
                 layout: dict = None, metadata=None, timestamp=None):
        # Default gateway_type to PARALLEL (Gateway's default of EXCLUSIVE would
        # be rejected by our setter override). Caller can pass INCLUSIVE
        # explicitly; anything else -> ValueError via the override.
        if gateway_type is None:
            gateway_type = GatewayType.PARALLEL
        super().__init__(name=name, gateway_type=gateway_type, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        # Initialise private slots before the property setters run so they
        # can safely read each other.
        self.__gateway_role = None
        self.__collaboration_mode = None
        self.__merging_strategy = None
        # Assign through setters in the order that lets auto-management
        # converge: collaboration_mode first (so default lookups work), then
        # gateway_role (auto-sets/clears merging_strategy), then the explicit
        # merging_strategy override (validated).
        self.collaboration_mode = (collaboration_mode if collaboration_mode is not None
                                   else CollaborationMode.VOTING)
        self.gateway_role = (gateway_role if gateway_role is not None
                             else GatewayRole.DIVERGING)
        if merging_strategy is not None:
            self.merging_strategy = merging_strategy
        self.trust_score = trust_score

    # --- gateway_type setter override (eligibility check) -------------------

    @Gateway.gateway_type.setter
    def gateway_type(self, value: "GatewayType"):
        """GatewayType: Set the gateway kind. Restricted to PARALLEL or INCLUSIVE.

        Raises:
            TypeError: if not a GatewayType.
            ValueError: if not in {PARALLEL, INCLUSIVE} (paper §4.3 excludes
                EXCLUSIVE / COMPLEX / EVENT_BASED).
        """
        if not isinstance(value, GatewayType):
            raise TypeError(
                f"gateway_type must be a GatewayType, got {type(value).__name__}"
            )
        if value not in _AGENTIC_ELIGIBLE_GATEWAY_TYPES:
            raise ValueError(
                f"AgenticGateway.gateway_type must be PARALLEL or INCLUSIVE, "
                f"got {value} (paper §4.3 excludes EXCLUSIVE / COMPLEX / "
                f"EVENT_BASED)"
            )
        # Write to the base's mangled slot, matching the BPMNElement.name
        # override pattern in the base metamodel.
        self._Gateway__gateway_type = value

    # --- gateway_role -------------------------------------------------------

    @property
    def gateway_role(self) -> "GatewayRole":
        """GatewayRole: Get the gateway role (diverging / merging)."""
        return self.__gateway_role

    @gateway_role.setter
    def gateway_role(self, value: "GatewayRole"):
        """GatewayRole: Set the gateway role. Auto-manages ``merging_strategy``
        to maintain INV-A.

        Raises:
            TypeError: if not a GatewayRole.
        """
        if not isinstance(value, GatewayRole):
            raise TypeError(
                f"gateway_role must be a GatewayRole, got {type(value).__name__}"
            )
        self.__gateway_role = value
        if value == GatewayRole.DIVERGING:
            # INV-A: clear strategy on diverging gateways.
            self.__merging_strategy = None
        elif self.__merging_strategy is None:
            # MERGING with no strategy yet -- auto-set the default for the
            # current collaboration_mode.
            self.__merging_strategy = _DEFAULT_MERGING_STRATEGY[self.__collaboration_mode]

    # --- collaboration_mode -------------------------------------------------

    @property
    def collaboration_mode(self) -> "CollaborationMode":
        """CollaborationMode: Get the collaboration mode."""
        return self.__collaboration_mode

    @collaboration_mode.setter
    def collaboration_mode(self, value: "CollaborationMode"):
        """CollaborationMode: Set the collaboration mode. Auto-resets
        ``merging_strategy`` if the current value is no longer legal for the
        new mode.

        Raises:
            TypeError: if not a CollaborationMode.
        """
        if not isinstance(value, CollaborationMode):
            raise TypeError(
                f"collaboration_mode must be a CollaborationMode, "
                f"got {type(value).__name__}"
            )
        self.__collaboration_mode = value
        # INV-B: if we're merging and the current strategy is illegal for the
        # new mode, reset it to the new mode's default. Diverging gateways
        # keep merging_strategy=None unchanged.
        if (self.__merging_strategy is not None
                and self.__merging_strategy not in _LEGAL_MERGING_STRATEGIES[value]):
            self.__merging_strategy = _DEFAULT_MERGING_STRATEGY[value]

    # --- merging_strategy ---------------------------------------------------

    @property
    def merging_strategy(self):
        """MergingStrategy | None: Get the merging strategy. ``None`` iff
        ``gateway_role == DIVERGING`` (INV-A)."""
        return self.__merging_strategy

    @merging_strategy.setter
    def merging_strategy(self, value):
        """MergingStrategy | None: Set the merging strategy.

        Raises:
            TypeError: if not a MergingStrategy or None.
            ValueError: if INV-A or INV-B is violated.
        """
        if self.__gateway_role == GatewayRole.DIVERGING:
            if value is not None:
                raise ValueError(
                    "merging_strategy must be None when gateway_role is DIVERGING "
                    "(paper §4.3: strategy applies only at the merging gateway)"
                )
            self.__merging_strategy = None
            return
        # MERGING:
        if value is None:
            raise ValueError(
                "merging_strategy must not be None when gateway_role is MERGING"
            )
        if not isinstance(value, MergingStrategy):
            raise TypeError(
                f"merging_strategy must be a MergingStrategy or None, "
                f"got {type(value).__name__}"
            )
        if value not in _LEGAL_MERGING_STRATEGIES[self.__collaboration_mode]:
            legal = sorted(s.name for s in _LEGAL_MERGING_STRATEGIES[self.__collaboration_mode])
            raise ValueError(
                f"merging_strategy {value.name} is not legal for "
                f"collaboration_mode {self.__collaboration_mode.name}; "
                f"legal values: {legal}"
            )
        self.__merging_strategy = value

    # --- trust_score --------------------------------------------------------

    @property
    def trust_score(self) -> int:
        """int: Get the trust score (0-100)."""
        return self.__trust_score

    @trust_score.setter
    def trust_score(self, value: int):
        """int: Set the trust score.

        Raises:
            TypeError: if not an int.
            ValueError: if out of [0, 100].
        """
        self.__trust_score = _validate_trust_score(value)

    def __repr__(self):
        return (f"AgenticGateway(name='{self.name}', "
                f"gateway_type={self.gateway_type}, "
                f"gateway_role={self.gateway_role}, "
                f"collaboration_mode={self.collaboration_mode}, "
                f"merging_strategy={self.merging_strategy}, "
                f"trust_score={self.trust_score})")


# ---------------------------------------------------------------------------
# AgenticLane
# ---------------------------------------------------------------------------

class AgenticLane(Lane):
    """SEAA'25 «AgenticLane» -- a Lane with a profile role + trust score (paper §4.1, Fig 3a).

    Args:
        name (str): The lane label (inherited; may be empty).
        role (AgentRole): The profile role (default WORKER).
        trust_score (int): 0-100 (default 0).
        agent_diagram_ref (str | None): Opaque id of the AgentDiagram this
            lane's agent is defined by (SEAA'25 cross-diagram link, WME 08).
            Default None. Pass-through -- no UUID validation, no resolution.
        flow_nodes (set[FlowNode]): Inherited from Lane.
        layout (dict): Inherited.
        metadata, timestamp: Inherited.

    Attributes:
        role (AgentRole): The agent profile role.
        trust_score (int): The trust score.
        agent_diagram_ref (str | None): The AgentDiagram reference.
    """

    def __init__(self, name: str = "", role: "AgentRole" = None,
                 trust_score: int = 0, agent_diagram_ref: str = None,
                 flow_nodes: set = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, flow_nodes=flow_nodes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.role = role if role is not None else AgentRole.WORKER
        self.trust_score = trust_score
        self.agent_diagram_ref = agent_diagram_ref

    @property
    def role(self) -> "AgentRole":
        """AgentRole: Get the profile role (worker / manager)."""
        return self.__role

    @role.setter
    def role(self, value: "AgentRole"):
        """AgentRole: Set the profile role.

        Raises:
            TypeError: if not an AgentRole.
        """
        if not isinstance(value, AgentRole):
            raise TypeError(
                f"role must be an AgentRole, got {type(value).__name__}"
            )
        self.__role = value

    @property
    def trust_score(self) -> int:
        """int: Get the trust score (0-100)."""
        return self.__trust_score

    @trust_score.setter
    def trust_score(self, value: int):
        """int: Set the trust score.

        Raises:
            TypeError: if not an int.
            ValueError: if out of [0, 100].
        """
        self.__trust_score = _validate_trust_score(value)

    @property
    def agent_diagram_ref(self):
        """str | None: Get the opaque id of the AgentDiagram this lane's agent
        is defined by (SEAA'25 cross-diagram link, WME 08). ``None`` when unset.

        **Legacy carrier.** WME guide 11 moved the canonical task->agent link to
        ``AgenticTask.agent_diagram_ref``; this lane field is retained only for
        round-tripping legacy projects (matching WME's tolerant-but-inert lane
        carrier). New links should be authored on ``AgenticTask``."""
        return self.__agent_diagram_ref

    @agent_diagram_ref.setter
    def agent_diagram_ref(self, value):
        """str | None: Set the AgentDiagram reference.

        Opaque pass-through -- no UUID-format check, no typed resolution (the
        value is whatever id the project assigned the AgentDiagram). A future
        ``resolve_agent_diagram(project)`` helper can turn it into a typed
        AgentModel when a consumer wants one; not implemented here.

        Raises:
            TypeError: if not a str or None.
        """
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"agent_diagram_ref must be a str or None, got {type(value).__name__}"
            )
        self.__agent_diagram_ref = value

    def __repr__(self):
        return (f"AgenticLane(name='{self.name}', role={self.role}, "
                f"trust_score={self.trust_score}, "
                f"agent_diagram_ref={self.agent_diagram_ref!r})")


# ---------------------------------------------------------------------------
# AgenticMessageFlow
# ---------------------------------------------------------------------------

class AgenticMessageFlow(MessageFlow):
    """SEAA'25 «AgenticMessageFlow» -- a MessageFlow with collaboration semantics
    (paper §4.3 / WME 04D1).

    A cross-pool message flow carrying a collaboration mode + merging strategy,
    used when agents in different pools collaborate. Unlike ``AgenticGateway``
    there is **no** ``gateway_role`` (a message flow has no diverging/merging
    axis), so ``merging_strategy`` is always set (never None).

    Invariant (auto-maintained by the setters):

    * **INV-B:** ``merging_strategy`` is in
      ``_LEGAL_MERGING_STRATEGIES[collaboration_mode]``. Changing
      ``collaboration_mode`` resets ``merging_strategy`` to the new mode's
      default if the previous value is no longer legal.

    Args:
        source, target: Inherited from MessageFlow (Activity/Event endpoints).
        name (str): The flow label (may be empty).
        collaboration_mode (CollaborationMode): default VOTING.
        merging_strategy (MergingStrategy): default
            ``_DEFAULT_MERGING_STRATEGY[collaboration_mode]`` when omitted.
            Explicit value validated against ``_LEGAL_MERGING_STRATEGIES``.
        trust_score (int): 0-100 (default 0).
        layout, metadata, timestamp: Inherited.

    Attributes:
        collaboration_mode (CollaborationMode): The collaboration mode.
        merging_strategy (MergingStrategy): The merging strategy (never None).
        trust_score (int): The trust score.
    """

    def __init__(self, source, target, name: str = "",
                 collaboration_mode: "CollaborationMode" = None,
                 merging_strategy: "MergingStrategy" = None,
                 trust_score: int = 0,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(source=source, target=target, name=name, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        # Initialise private slots before the setters run so they can read
        # each other safely.
        self.__collaboration_mode = None
        self.__merging_strategy = None
        # collaboration_mode first (so the default lookup works), then
        # merging_strategy (auto-set to the mode default if omitted).
        self.collaboration_mode = (collaboration_mode if collaboration_mode is not None
                                   else CollaborationMode.VOTING)
        if merging_strategy is None:
            self.__merging_strategy = _DEFAULT_MERGING_STRATEGY[self.__collaboration_mode]
        else:
            self.merging_strategy = merging_strategy
        self.trust_score = trust_score

    @property
    def collaboration_mode(self) -> "CollaborationMode":
        """CollaborationMode: Get the collaboration mode."""
        return self.__collaboration_mode

    @collaboration_mode.setter
    def collaboration_mode(self, value: "CollaborationMode"):
        """CollaborationMode: Set the collaboration mode. Auto-resets
        ``merging_strategy`` if the current value is no longer legal.

        Raises:
            TypeError: if not a CollaborationMode.
        """
        if not isinstance(value, CollaborationMode):
            raise TypeError(
                f"collaboration_mode must be a CollaborationMode, "
                f"got {type(value).__name__}"
            )
        self.__collaboration_mode = value
        if (self.__merging_strategy is not None
                and self.__merging_strategy not in _LEGAL_MERGING_STRATEGIES[value]):
            self.__merging_strategy = _DEFAULT_MERGING_STRATEGY[value]

    @property
    def merging_strategy(self) -> "MergingStrategy":
        """MergingStrategy: Get the merging strategy (never None)."""
        return self.__merging_strategy

    @merging_strategy.setter
    def merging_strategy(self, value: "MergingStrategy"):
        """MergingStrategy: Set the merging strategy.

        Raises:
            TypeError: if not a MergingStrategy (None is not allowed -- a message
                flow has no diverging role, so the strategy is always set).
            ValueError: if not legal for the current collaboration_mode (INV-B).
        """
        if not isinstance(value, MergingStrategy):
            raise TypeError(
                f"merging_strategy must be a MergingStrategy, got {type(value).__name__}"
            )
        if value not in _LEGAL_MERGING_STRATEGIES[self.__collaboration_mode]:
            legal = sorted(s.name for s in _LEGAL_MERGING_STRATEGIES[self.__collaboration_mode])
            raise ValueError(
                f"merging_strategy {value.name} is not legal for "
                f"collaboration_mode {self.__collaboration_mode.name}; "
                f"legal values: {legal}"
            )
        self.__merging_strategy = value

    @property
    def trust_score(self) -> int:
        """int: Get the trust score (0-100)."""
        return self.__trust_score

    @trust_score.setter
    def trust_score(self, value: int):
        """int: Set the trust score.

        Raises:
            TypeError: if not an int.
            ValueError: if out of [0, 100].
        """
        self.__trust_score = _validate_trust_score(value)

    def __repr__(self):
        source_name = self.source.name if self.source is not None else None
        target_name = self.target.name if self.target is not None else None
        return (f"AgenticMessageFlow(name='{self.name}', "
                f"source='{source_name}', target='{target_name}', "
                f"collaboration_mode={self.collaboration_mode}, "
                f"merging_strategy={self.merging_strategy}, "
                f"trust_score={self.trust_score})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ReflectionMode",
    "CollaborationMode",
    "MergingStrategy",
    "GatewayRole",
    "AgentRole",
    # Classes
    "AgenticTask",
    "AgenticGateway",
    "AgenticLane",
    "AgenticMessageFlow",
]
