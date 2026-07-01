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
addition.
"""

from enum import Enum

from besser.BUML.metamodel.bpmn.bpmn import (
    Gateway,
    GatewayType,
    Lane,
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


class GatewayRole(Enum):
    """SEAA'25 «AgenticGateway» role (paper §4.3).

    Diverging-vs-merging axis of an agentic gateway pair.
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


def _validate_swarm_size(value) -> int:
    """Validate a swarm size: ``int`` ``>= 1``.

    The swarm size of an «AgenticLane» — how many identical copies of the
    lane's agent participate. No upper cap.

    Raises:
        TypeError: if not an int (bool excluded — it is technically int).
        ValueError: if < 1.

    Returns the value unchanged on success.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"swarm_size must be an int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"swarm_size must be >= 1, got {value}")
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
        agent_diagram_ref (str | None): Opaque id of the AgentDiagram this
            task's agent behavior is defined by (SEAA'25 cross-diagram link,
            WME guide 11). Default None. Pass-through -- no UUID validation,
            no resolution. Canonical carrier.
        task_type (TaskType): Inherited from Task.
        loop_characteristics (LoopCharacteristics): Inherited from Activity.
        layout (dict): Inherited (opaque DI passthrough).
        metadata, timestamp: Inherited.

    Attributes:
        reflection_mode (ReflectionMode): The reflection mode.
        trust_score (int): The trust score.
        agent_diagram_ref (str | None): The AgentDiagram reference.
    """

    def __init__(self, name: str = "", reflection_mode: "ReflectionMode" = None,
                 trust_score: int = 0,
                 agent_diagram_ref: str = None,
                 task_type=None, loop_characteristics=None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, task_type=task_type,
                         loop_characteristics=loop_characteristics,
                         layout=layout, metadata=metadata, timestamp=timestamp)
        self.reflection_mode = (reflection_mode if reflection_mode is not None
                                else ReflectionMode.NONE)
        self.trust_score = trust_score
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
    def agent_diagram_ref(self):
        """str | None: Get the opaque id of the AgentDiagram this task's agent
        behavior is defined by (SEAA'25 cross-diagram link, WME guide 11 --
        canonical carrier). ``None`` when unset."""
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
                f"agent_diagram_ref={self.agent_diagram_ref!r})")


# ---------------------------------------------------------------------------
# AgenticGateway
# ---------------------------------------------------------------------------

class AgenticGateway(Gateway):
    """SEAA'25 «AgenticGateway» -- a Gateway with collaboration semantics (paper §4.3, Fig 4).

    Per paper §4.3, only PARALLEL (AgenticAND) and INCLUSIVE (AgenticOR) gateways
    may be agentic; the inherited ``gateway_type`` setter is overridden to
    enforce this.

    Args:
        name (str): The gateway label (may be empty).
        gateway_type (GatewayType): PARALLEL or INCLUSIVE only (default PARALLEL).
        gateway_role (GatewayRole): DIVERGING or MERGING (default DIVERGING).
        trust_score (int): 0-100 (default 0).
        governance_dsl (str | None): Opaque Governance-DSL snippet authored by
            WME's ``generateGovernanceDsl`` for a merging gateway. Default None.
            BESSER carries and round-trips it, never derives it; stored
            tolerantly on any gateway.
        layout, metadata, timestamp: Inherited.

    Attributes:
        gateway_role (GatewayRole): The gateway role.
        trust_score (int): The trust score.
        governance_dsl (str | None): The Governance-DSL snippet.
    """

    def __init__(self, name: str = "", gateway_type: "GatewayType" = None,
                 gateway_role: "GatewayRole" = None,
                 trust_score: int = 0,
                 governance_dsl: str = None,
                 layout: dict = None, metadata=None, timestamp=None):
        # Default gateway_type to PARALLEL (Gateway's default of EXCLUSIVE would
        # be rejected by our setter override). Caller can pass INCLUSIVE
        # explicitly; anything else -> ValueError via the override.
        if gateway_type is None:
            gateway_type = GatewayType.PARALLEL
        super().__init__(name=name, gateway_type=gateway_type, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        # Initialise private slot before the property setter runs.
        self.__gateway_role = None
        self.gateway_role = (gateway_role if gateway_role is not None
                             else GatewayRole.DIVERGING)
        self.trust_score = trust_score
        self.governance_dsl = governance_dsl

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
        """GatewayRole: Set the gateway role.

        Raises:
            TypeError: if not a GatewayRole.
        """
        if not isinstance(value, GatewayRole):
            raise TypeError(
                f"gateway_role must be a GatewayRole, got {type(value).__name__}"
            )
        self.__gateway_role = value

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

    # --- governance_dsl -----------------------------------------------------

    @property
    def governance_dsl(self):
        """str | None: Get the Governance-DSL snippet for this (merging) gateway,
        or ``None``. Opaque free text authored by WME's ``generateGovernanceDsl``;
        BESSER carries and round-trips it, never derives it. Only meaningful on a
        MERGING gateway, but stored tolerantly on any (no invariant)."""
        return self.__governance_dsl

    @governance_dsl.setter
    def governance_dsl(self, value):
        """str | None: Set the Governance-DSL snippet.

        Raises:
            TypeError: if not a str or None.
        """
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"governance_dsl must be a str or None, got {type(value).__name__}"
            )
        self.__governance_dsl = value

    def __repr__(self):
        return (f"AgenticGateway(name='{self.name}', "
                f"gateway_type={self.gateway_type}, "
                f"gateway_role={self.gateway_role}, "
                f"trust_score={self.trust_score}, "
                f"governance_dsl={'<set>' if self.governance_dsl else None})")


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
        swarm_size (int): Swarm size — how many identical copies of this
            lane's agent participate (>= 1, default 1). Flows to the Deployment
            artifact ``[N]``. WME meeting 2026-06-08 point #3.
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
                 swarm_size: int = 1,
                 flow_nodes: set = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, flow_nodes=flow_nodes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.role = role if role is not None else AgentRole.WORKER
        self.trust_score = trust_score
        self.agent_diagram_ref = agent_diagram_ref
        self.swarm_size = swarm_size

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
    def swarm_size(self) -> int:
        """int: Get the swarm size (>= 1, default 1)."""
        return self.__swarm_size

    @swarm_size.setter
    def swarm_size(self, value: int):
        """int: Set the swarm size.

        Raises:
            TypeError: if not an int.
            ValueError: if < 1.
        """
        self.__swarm_size = _validate_swarm_size(value)

    @property
    def agent_diagram_ref(self):
        """str | None: Get the opaque id of the AgentDiagram this lane's agent
        is defined by (SEAA'25 cross-diagram link, WME 08). ``None`` when unset.

        **Legacy carrier.** WME guide 11 moved the canonical task->agent link to
        ``AgenticTask.agent_diagram_ref``; this lane field is retained only for
        round-tripping legacy projects. New links should be authored on
        ``AgenticTask``."""
        return self.__agent_diagram_ref

    @agent_diagram_ref.setter
    def agent_diagram_ref(self, value):
        """str | None: Set the AgentDiagram reference.

        Opaque pass-through -- no UUID-format check, no typed resolution.

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
                f"swarm_size={self.swarm_size}, "
                f"agent_diagram_ref={self.agent_diagram_ref!r})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ReflectionMode",
    "GatewayRole",
    "AgentRole",
    # Classes
    "AgenticTask",
    "AgenticGateway",
    "AgenticLane",
]
