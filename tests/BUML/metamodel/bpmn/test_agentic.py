"""Tests for the SEAA'25 BPMN agentic extension (``besser.BUML.metamodel.bpmn.agentic``).

Covers the test matrix in
``.claude/bpmn-agentic/02-seaa25-metamodel-extension-guide.md`` §9.1:
enum integrity, AgenticTask construction + setters, AgenticGateway
construction + eligibility + tri-attribute invariants + the full
strategy-per-mode matrix, AgenticLane construction + setters, backward
compatibility with the BPMN base, and ``__repr__``.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticMessageFlow,
    AgenticTask,
    AgentRole,
    CollaborationMode,
    Gateway,
    GatewayRole,
    GatewayType,
    Lane,
    LoopCharacteristics,
    MergingStrategy,
    MessageFlow,
    Participant,
    Process,
    ReflectionMode,
    Task,
    TaskType,
)


# ---------------------------------------------------------------------------
# Enum integrity (T-enum-1 .. T-enum-5)
# ---------------------------------------------------------------------------

def test_reflection_mode_values():
    """ReflectionMode .value strings match WME's BPMNReflectionMode."""
    assert ReflectionMode.NONE.value == "none"
    assert ReflectionMode.SELF.value == "self"
    assert ReflectionMode.CROSS.value == "cross"
    assert ReflectionMode.HUMAN.value == "human"
    assert len(ReflectionMode) == 4


def test_collaboration_mode_values():
    """CollaborationMode .value strings match WME's BPMNCollaborationMode."""
    assert CollaborationMode.VOTING.value == "voting"
    assert CollaborationMode.ROLE.value == "role"
    assert CollaborationMode.DEBATE.value == "debate"
    assert CollaborationMode.COMPETITION.value == "competition"
    assert len(CollaborationMode) == 4


def test_merging_strategy_values():
    """MergingStrategy .value strings match WME's BPMNMergingStrategy (paper Table 2)."""
    assert MergingStrategy.MAJORITY.value == "majority"
    assert MergingStrategy.ABSOLUTE_MAJORITY.value == "absolute-majority"
    assert MergingStrategy.MINORITY.value == "minority"
    assert MergingStrategy.LEADER_DRIVEN.value == "leader-driven"
    assert MergingStrategy.COMPOSED.value == "composed"
    assert MergingStrategy.FASTEST.value == "fastest"
    assert MergingStrategy.MOST_COMPLETE.value == "most-complete"
    assert len(MergingStrategy) == 7


def test_gateway_role_values():
    """GatewayRole .value strings match WME's BPMNGatewayRole."""
    assert GatewayRole.DIVERGING.value == "diverging"
    assert GatewayRole.MERGING.value == "merging"
    assert len(GatewayRole) == 2


def test_agent_role_values():
    """AgentRole .value strings match WME's BPMNAgentRole."""
    assert AgentRole.WORKER.value == "worker"
    assert AgentRole.MANAGER.value == "manager"
    assert len(AgentRole) == 2


# ---------------------------------------------------------------------------
# AgenticTask (T-task-1 .. T-task-8)
# ---------------------------------------------------------------------------

def test_agentic_task_defaults():
    """Constructed with no SEAA args, defaults to NONE reflection / 0 trust."""
    t = AgenticTask(name="review")
    assert t.reflection_mode == ReflectionMode.NONE
    assert t.trust_score == 0


def test_agentic_task_isinstance_task():
    """AgenticTask IS a Task — subclass identity holds."""
    t = AgenticTask(name="review")
    assert isinstance(t, Task)
    assert isinstance(t, AgenticTask)


@pytest.mark.parametrize("mode", list(ReflectionMode))
def test_agentic_task_reflection_setter_accepts_all_values(mode):
    """The setter accepts every ReflectionMode member."""
    t = AgenticTask(name="t", reflection_mode=mode)
    assert t.reflection_mode == mode


def test_agentic_task_reflection_type_error():
    """Non-ReflectionMode (e.g. string) raises TypeError."""
    with pytest.raises(TypeError, match="reflection_mode must be a ReflectionMode"):
        AgenticTask(name="t", reflection_mode="self")


@pytest.mark.parametrize("score", [0, 1, 50, 99, 100])
def test_agentic_task_trust_score_in_range(score):
    """Trust score accepts the full [0, 100] range."""
    t = AgenticTask(name="t", trust_score=score)
    assert t.trust_score == score


@pytest.mark.parametrize("score", [-1, -100, 101, 1000])
def test_agentic_task_trust_score_out_of_range(score):
    """Trust score raises ValueError for out-of-range integers (no silent clamp)."""
    with pytest.raises(ValueError, match=r"trust_score must be in \[0, 100\]"):
        AgenticTask(name="t", trust_score=score)


@pytest.mark.parametrize("value", ["85", 0.5, None, True])
def test_agentic_task_trust_score_type_error(value):
    """Trust score raises TypeError for non-int (including bool — bool is an int subclass but rejected)."""
    with pytest.raises(TypeError, match="trust_score must be an int"):
        AgenticTask(name="t", trust_score=value)


def test_agentic_task_inherits_task_kwargs():
    """Task's own kwargs (task_type, loop_characteristics) pass through to the base."""
    t = AgenticTask(name="t", task_type=TaskType.USER,
                    loop_characteristics=LoopCharacteristics.STANDARD_LOOP)
    assert t.task_type == TaskType.USER
    assert t.loop_characteristics == LoopCharacteristics.STANDARD_LOOP


# ---------------------------------------------------------------------------
# AgenticGateway — basic (T-gw-1 .. T-gw-7)
# ---------------------------------------------------------------------------

def test_agentic_gateway_defaults():
    """Defaults: PARALLEL / DIVERGING / VOTING / None / 0."""
    g = AgenticGateway(name="g")
    assert g.gateway_type == GatewayType.PARALLEL
    assert g.gateway_role == GatewayRole.DIVERGING
    assert g.collaboration_mode == CollaborationMode.VOTING
    assert g.merging_strategy is None  # INV-A: DIVERGING -> None
    assert g.trust_score == 0


def test_agentic_gateway_isinstance_gateway():
    """AgenticGateway IS a Gateway."""
    g = AgenticGateway(name="g")
    assert isinstance(g, Gateway)
    assert isinstance(g, AgenticGateway)


@pytest.mark.parametrize("gw_type", [GatewayType.PARALLEL, GatewayType.INCLUSIVE])
def test_agentic_gateway_type_eligible(gw_type):
    """PARALLEL and INCLUSIVE are accepted (paper §4.3)."""
    g = AgenticGateway(name="g", gateway_type=gw_type)
    assert g.gateway_type == gw_type


@pytest.mark.parametrize("gw_type", [
    GatewayType.EXCLUSIVE,
    GatewayType.COMPLEX,
    GatewayType.EVENT_BASED,
])
def test_agentic_gateway_type_rejects_ineligible(gw_type):
    """EXCLUSIVE / COMPLEX / EVENT_BASED raise ValueError (paper §4.3 exclusion)."""
    with pytest.raises(ValueError, match="must be PARALLEL or INCLUSIVE"):
        AgenticGateway(name="g", gateway_type=gw_type)


def test_agentic_gateway_type_setter_after_construction():
    """The eligibility check also fires on later assignments, not only construction."""
    g = AgenticGateway(name="g", gateway_type=GatewayType.PARALLEL)
    g.gateway_type = GatewayType.INCLUSIVE  # OK
    assert g.gateway_type == GatewayType.INCLUSIVE
    with pytest.raises(ValueError, match="must be PARALLEL or INCLUSIVE"):
        g.gateway_type = GatewayType.EXCLUSIVE


def test_agentic_gateway_type_type_error():
    """Non-GatewayType raises TypeError."""
    with pytest.raises(TypeError, match="gateway_type must be a GatewayType"):
        AgenticGateway(name="g", gateway_type="parallel")


@pytest.mark.parametrize("score", [-1, 101])
def test_agentic_gateway_trust_score_out_of_range(score):
    """Trust score validation shared with AgenticTask."""
    with pytest.raises(ValueError, match=r"trust_score must be in \[0, 100\]"):
        AgenticGateway(name="g", trust_score=score)


# ---------------------------------------------------------------------------
# AgenticGateway — tri-attribute invariants (T-gw-inv-1 .. T-gw-inv-8)
# ---------------------------------------------------------------------------

def test_inv_a_diverging_strategy_is_none():
    """A fresh DIVERGING gateway has merging_strategy = None (INV-A)."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.DIVERGING)
    assert g.merging_strategy is None


def test_inv_a_merging_strategy_auto_set():
    """Flipping DIVERGING -> MERGING auto-sets strategy to the default for current mode."""
    g = AgenticGateway(name="g", collaboration_mode=CollaborationMode.VOTING)
    assert g.merging_strategy is None
    g.gateway_role = GatewayRole.MERGING
    # _DEFAULT_MERGING_STRATEGY[VOTING] = MAJORITY
    assert g.merging_strategy == MergingStrategy.MAJORITY


def test_inv_a_merging_strategy_auto_set_for_role():
    """The auto-set default tracks the current collaboration_mode."""
    g = AgenticGateway(name="g", collaboration_mode=CollaborationMode.ROLE)
    g.gateway_role = GatewayRole.MERGING
    assert g.merging_strategy == MergingStrategy.LEADER_DRIVEN


def test_inv_a_merging_strategy_auto_set_for_competition():
    g = AgenticGateway(name="g", collaboration_mode=CollaborationMode.COMPETITION)
    g.gateway_role = GatewayRole.MERGING
    assert g.merging_strategy == MergingStrategy.FASTEST


def test_inv_a_merging_to_diverging_clears():
    """Flipping MERGING -> DIVERGING auto-clears the strategy to None."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.VOTING)
    assert g.merging_strategy is not None
    g.gateway_role = GatewayRole.DIVERGING
    assert g.merging_strategy is None


def test_inv_a_setter_rejects_strategy_on_diverging():
    """Explicitly setting merging_strategy on a DIVERGING gateway raises ValueError."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.DIVERGING)
    with pytest.raises(ValueError, match="must be None when gateway_role is DIVERGING"):
        g.merging_strategy = MergingStrategy.MAJORITY


def test_inv_a_setter_rejects_none_on_merging():
    """Explicitly setting None on a MERGING gateway raises ValueError."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.VOTING)
    with pytest.raises(ValueError, match="must not be None when gateway_role is MERGING"):
        g.merging_strategy = None


def test_inv_b_mode_change_resets_illegal_strategy():
    """Switching collaboration_mode auto-resets merging_strategy if previous value is illegal."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.VOTING,
                       merging_strategy=MergingStrategy.MAJORITY)
    g.collaboration_mode = CollaborationMode.COMPETITION
    # MAJORITY is not legal for COMPETITION -> reset to default (FASTEST)
    assert g.merging_strategy == MergingStrategy.FASTEST


def test_inv_b_mode_change_keeps_legal_strategy():
    """Switching mode does NOT change the strategy if the current value is still legal."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.VOTING,
                       merging_strategy=MergingStrategy.MAJORITY)
    g.collaboration_mode = CollaborationMode.DEBATE  # debate accepts MAJORITY
    assert g.merging_strategy == MergingStrategy.MAJORITY


def test_inv_b_explicit_illegal_strategy_rejected():
    """Explicit set of an illegal strategy raises ValueError listing legal values."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.VOTING)
    with pytest.raises(ValueError, match="not legal for collaboration_mode VOTING"):
        g.merging_strategy = MergingStrategy.FASTEST


def test_inv_b_diverging_mode_change_keeps_none():
    """A DIVERGING gateway changing collaboration_mode keeps merging_strategy = None."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.DIVERGING,
                       collaboration_mode=CollaborationMode.VOTING)
    g.collaboration_mode = CollaborationMode.COMPETITION
    assert g.merging_strategy is None


def test_construction_explicit_strategy_without_merging_role_fails():
    """Passing merging_strategy without gateway_role=MERGING raises ValueError.

    The default gateway_role is DIVERGING; explicit merging_strategy on a
    DIVERGING gateway violates INV-A.
    """
    with pytest.raises(ValueError, match="must be None when gateway_role is DIVERGING"):
        AgenticGateway(name="g", merging_strategy=MergingStrategy.MAJORITY)


# ---------------------------------------------------------------------------
# AgenticGateway — full strategy-per-mode matrix (T-gw-matrix-*)
# ---------------------------------------------------------------------------

# Legal strategies per mode, mirroring _LEGAL_MERGING_STRATEGIES.
_LEGAL = {
    CollaborationMode.VOTING: {
        MergingStrategy.MAJORITY,
        MergingStrategy.ABSOLUTE_MAJORITY,
        MergingStrategy.MINORITY,
    },
    CollaborationMode.ROLE: {
        MergingStrategy.LEADER_DRIVEN,
        MergingStrategy.COMPOSED,
    },
    CollaborationMode.COMPETITION: {
        MergingStrategy.FASTEST,
        MergingStrategy.MOST_COMPLETE,
    },
    CollaborationMode.DEBATE: {
        MergingStrategy.MAJORITY,
        MergingStrategy.ABSOLUTE_MAJORITY,
        MergingStrategy.MINORITY,
        MergingStrategy.LEADER_DRIVEN,
        MergingStrategy.COMPOSED,
    },
}


@pytest.mark.parametrize("mode", list(CollaborationMode))
def test_legal_strategies_accepted_per_mode(mode):
    """For each (mode, legal_strategy) pair, explicit assignment succeeds."""
    legal = _LEGAL[mode]
    for strategy in legal:
        g = AgenticGateway(
            name="g",
            gateway_role=GatewayRole.MERGING,
            collaboration_mode=mode,
            merging_strategy=strategy,
        )
        assert g.merging_strategy == strategy


@pytest.mark.parametrize("mode", list(CollaborationMode))
def test_illegal_strategies_rejected_per_mode(mode):
    """For each (mode, illegal_strategy) pair, explicit assignment raises."""
    legal = _LEGAL[mode]
    all_strategies = set(MergingStrategy)
    illegal = all_strategies - legal
    for strategy in illegal:
        g = AgenticGateway(
            name="g",
            gateway_role=GatewayRole.MERGING,
            collaboration_mode=mode,
        )
        with pytest.raises(ValueError, match="not legal for collaboration_mode"):
            g.merging_strategy = strategy


def test_debate_borrows_from_voting_and_role():
    """Paper §4.3 last paragraph: debate accepts voting AND role strategies; not competition."""
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.DEBATE)
    # Voting strategies:
    g.merging_strategy = MergingStrategy.MAJORITY
    g.merging_strategy = MergingStrategy.ABSOLUTE_MAJORITY
    g.merging_strategy = MergingStrategy.MINORITY
    # Role strategies:
    g.merging_strategy = MergingStrategy.LEADER_DRIVEN
    g.merging_strategy = MergingStrategy.COMPOSED
    # Competition strategies are rejected:
    with pytest.raises(ValueError):
        g.merging_strategy = MergingStrategy.FASTEST
    with pytest.raises(ValueError):
        g.merging_strategy = MergingStrategy.MOST_COMPLETE


# ---------------------------------------------------------------------------
# AgenticGateway — setter type checks
# ---------------------------------------------------------------------------

def test_gateway_role_type_error():
    g = AgenticGateway(name="g")
    with pytest.raises(TypeError, match="gateway_role must be a GatewayRole"):
        g.gateway_role = "diverging"


def test_collaboration_mode_type_error():
    g = AgenticGateway(name="g")
    with pytest.raises(TypeError, match="collaboration_mode must be a CollaborationMode"):
        g.collaboration_mode = "voting"


def test_merging_strategy_type_error_on_merging():
    g = AgenticGateway(name="g", gateway_role=GatewayRole.MERGING)
    with pytest.raises(TypeError, match="merging_strategy must be a MergingStrategy"):
        g.merging_strategy = "majority"


# ---------------------------------------------------------------------------
# AgenticLane (T-lane-1 .. T-lane-7)
# ---------------------------------------------------------------------------

def test_agentic_lane_defaults():
    """Defaults: role=WORKER, trust_score=0."""
    lane = AgenticLane(name="Reviewer")
    assert lane.role == AgentRole.WORKER
    assert lane.trust_score == 0


def test_agentic_lane_isinstance_lane():
    """AgenticLane IS a Lane."""
    lane = AgenticLane(name="Reviewer")
    assert isinstance(lane, Lane)
    assert isinstance(lane, AgenticLane)


@pytest.mark.parametrize("role", list(AgentRole))
def test_agentic_lane_role_setter(role):
    """Setter accepts every AgentRole member."""
    lane = AgenticLane(name="lane", role=role)
    assert lane.role == role


def test_agentic_lane_role_type_error():
    with pytest.raises(TypeError, match="role must be an AgentRole"):
        AgenticLane(name="lane", role="worker")


def test_agentic_lane_trust_score_in_range():
    lane = AgenticLane(name="lane", trust_score=85)
    assert lane.trust_score == 85


@pytest.mark.parametrize("score", [-1, 101])
def test_agentic_lane_trust_score_out_of_range(score):
    with pytest.raises(ValueError, match=r"trust_score must be in \[0, 100\]"):
        AgenticLane(name="lane", trust_score=score)


def test_agentic_lane_relaxed_name():
    """AgenticLane inherits BPMNElement's relaxed name (empty / spaces allowed)."""
    AgenticLane(name="")  # empty allowed
    AgenticLane(name="Code Reviewer")  # space allowed
    AgenticLane(name="merge:approve")  # punctuation allowed


# ---------------------------------------------------------------------------
# AgenticLane.agent_diagram_ref — S1 (WME 08 cross-diagram link)
# ---------------------------------------------------------------------------

def test_agentic_lane_agent_diagram_ref_default_none():
    """Constructed without the kwarg → agent_diagram_ref is None (S1-mm-1)."""
    lane = AgenticLane(name="Reviewer")
    assert lane.agent_diagram_ref is None


def test_agentic_lane_agent_diagram_ref_str():
    """Accepts an arbitrary string (e.g. a UUID) verbatim — opaque (S1-mm-2)."""
    ref = "3f0a1c2d-4e5b-4f6a-9012-3456789abcde"
    lane = AgenticLane(name="Reviewer", agent_diagram_ref=ref)
    assert lane.agent_diagram_ref == ref


def test_agentic_lane_agent_diagram_ref_none_allowed():
    """Explicit None is accepted (S1-mm-3)."""
    lane = AgenticLane(name="Reviewer", agent_diagram_ref=None)
    assert lane.agent_diagram_ref is None


@pytest.mark.parametrize("value", [123, 0.5, ["x"], {"a": 1}, True])
def test_agentic_lane_agent_diagram_ref_type_error(value):
    """Non-str / non-None raises TypeError (S1-mm-4)."""
    with pytest.raises(TypeError, match="agent_diagram_ref must be a str or None"):
        AgenticLane(name="Reviewer", agent_diagram_ref=value)


def test_agentic_lane_repr_includes_ref():
    """__repr__ includes agent_diagram_ref (S1-mm-5)."""
    lane = AgenticLane(name="Reviewer", role=AgentRole.MANAGER, trust_score=90,
                       agent_diagram_ref="ref-123")
    r = repr(lane)
    assert "agent_diagram_ref='ref-123'" in r


# ---------------------------------------------------------------------------
# AgenticTask.agent_diagram_ref — R-a (WME guide 11 canonical task->agent link)
# ---------------------------------------------------------------------------

def test_agentic_task_agent_diagram_ref_default_none():
    """Constructed without the kwarg → agent_diagram_ref is None (R-a-mm-1)."""
    task = AgenticTask(name="Review")
    assert task.agent_diagram_ref is None


def test_agentic_task_agent_diagram_ref_str():
    """Accepts an arbitrary string (e.g. a UUID) verbatim — opaque (R-a-mm-2)."""
    ref = "3f0a1c2d-4e5b-4f6a-9012-3456789abcde"
    task = AgenticTask(name="Review", agent_diagram_ref=ref)
    assert task.agent_diagram_ref == ref


def test_agentic_task_agent_diagram_ref_none_allowed():
    """Explicit None is accepted (R-a-mm-3)."""
    task = AgenticTask(name="Review", agent_diagram_ref=None)
    assert task.agent_diagram_ref is None


@pytest.mark.parametrize("value", [123, 0.5, ["x"], {"a": 1}, True])
def test_agentic_task_agent_diagram_ref_type_error(value):
    """Non-str / non-None raises TypeError (R-a-mm-4)."""
    with pytest.raises(TypeError, match="agent_diagram_ref must be a str or None"):
        AgenticTask(name="Review", agent_diagram_ref=value)


def test_agentic_task_repr_includes_ref():
    """__repr__ includes agent_diagram_ref (R-a-mm-5)."""
    task = AgenticTask(name="Review", trust_score=90, agent_diagram_ref="ref-123")
    assert "agent_diagram_ref='ref-123'" in repr(task)


# ---------------------------------------------------------------------------
# AgenticGateway.governance_dsl — R-b (governance-dsl guide 02; opaque CDATA)
# ---------------------------------------------------------------------------

def test_agentic_gateway_governance_dsl_default_none():
    """Constructed without the kwarg → governance_dsl is None (R-b-mm-1)."""
    gw = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL)
    assert gw.governance_dsl is None


def test_agentic_gateway_governance_dsl_str():
    """Accepts an arbitrary (multi-line) string verbatim — opaque (R-b-mm-2)."""
    dsl = "Scopes:\n    Tasks:\n        MergeDecision\nMajorityPolicy P {\n}"
    gw = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.MERGING, governance_dsl=dsl)
    assert gw.governance_dsl == dsl


def test_agentic_gateway_governance_dsl_role_independent():
    """No invariant ties governance_dsl to gateway_role — a diverging gateway
    may hold it too (R-b-mm-3)."""
    gw = AgenticGateway(name="Split", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.DIVERGING, governance_dsl="x")
    assert gw.governance_dsl == "x"
    assert gw.merging_strategy is None  # INV-A still holds independently


@pytest.mark.parametrize("value", [123, 0.5, ["x"], {"a": 1}, True])
def test_agentic_gateway_governance_dsl_type_error(value):
    """Non-str / non-None raises TypeError (R-b-mm-4)."""
    with pytest.raises(TypeError, match="governance_dsl must be a str or None"):
        AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                       governance_dsl=value)


def test_agentic_gateway_repr_marks_governance():
    """__repr__ marks governance_dsl as set without dumping the blob (R-b-mm-5)."""
    gw = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.MERGING, governance_dsl="big\nblob")
    assert "governance_dsl=<set>" in repr(gw)
    plain = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL)
    assert "governance_dsl=None" in repr(plain)


# ---------------------------------------------------------------------------
# AgenticTask.collaboration_mode — S2 (WME 04D1 D-D1; paper deviation)
# ---------------------------------------------------------------------------

def test_agentic_task_collaboration_mode_default_voting():
    """Default collaboration_mode is VOTING (S2-mm-1)."""
    t = AgenticTask(name="review")
    assert t.collaboration_mode == CollaborationMode.VOTING


@pytest.mark.parametrize("mode", list(CollaborationMode))
def test_agentic_task_collaboration_mode_setter(mode):
    """Setter accepts every CollaborationMode member (S2-mm-2)."""
    t = AgenticTask(name="t", collaboration_mode=mode)
    assert t.collaboration_mode == mode


def test_agentic_task_collaboration_mode_type_error():
    """Non-CollaborationMode raises TypeError (S2-mm-3)."""
    with pytest.raises(TypeError, match="collaboration_mode must be a CollaborationMode"):
        AgenticTask(name="t", collaboration_mode="voting")


def test_agentic_task_collaboration_mode_independent():
    """collaboration_mode is independent — no merging-strategy machinery on the task (S2-mm-4)."""
    t = AgenticTask(name="t", reflection_mode=ReflectionMode.SELF, trust_score=40)
    t.collaboration_mode = CollaborationMode.COMPETITION
    # Setting it does not disturb the other fields, and the task has no
    # gateway_role / merging_strategy attributes.
    assert t.reflection_mode == ReflectionMode.SELF
    assert t.trust_score == 40
    assert not hasattr(t, "merging_strategy")
    assert not hasattr(t, "gateway_role")


def test_agentic_task_repr_includes_collaboration_mode():
    """__repr__ includes collaboration_mode (S2-mm-5)."""
    t = AgenticTask(name="review", collaboration_mode=CollaborationMode.DEBATE)
    assert "CollaborationMode.DEBATE" in repr(t)


# ---------------------------------------------------------------------------
# AgenticMessageFlow — S3 (WME 04D1)
# ---------------------------------------------------------------------------

def test_agentic_message_flow_defaults():
    """Defaults: VOTING / MAJORITY / 0 (S3-mm-1)."""
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"))
    assert amf.collaboration_mode == CollaborationMode.VOTING
    assert amf.merging_strategy == MergingStrategy.MAJORITY  # _DEFAULT[VOTING]
    assert amf.trust_score == 0


def test_agentic_message_flow_isinstance_message_flow():
    """AgenticMessageFlow IS a MessageFlow (S3-mm-2)."""
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"))
    assert isinstance(amf, MessageFlow)
    assert isinstance(amf, AgenticMessageFlow)


def test_agentic_message_flow_endpoint_check():
    """Non-eligible endpoint (Gateway) raises TypeError (inherited from MessageFlow) (S3-mm-3)."""
    bad = Gateway(name="g", gateway_type=GatewayType.EXCLUSIVE)
    with pytest.raises(TypeError, match="must be an Activity, Event, or Participant"):
        AgenticMessageFlow(source=bad, target=Task(name="B"))


def test_agentic_message_flow_participant_endpoints():
    """Pool-to-pool endpoints are accepted (R-R1 / BPMN 2.0.2 §9.3)."""
    p1 = Participant(name="P1", process=Process(name="proc1"))
    p2 = Participant(name="P2", process=Process(name="proc2"))
    amf = AgenticMessageFlow(source=p1, target=p2,
                             collaboration_mode=CollaborationMode.ROLE,
                             merging_strategy=MergingStrategy.LEADER_DRIVEN)
    assert amf.source is p1 and amf.target is p2


def test_agentic_message_flow_merging_never_none():
    """merging_strategy=None auto-defaults; explicit None to the setter raises (S3-mm-4)."""
    # None at construction -> default for the mode (not None).
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                             collaboration_mode=CollaborationMode.ROLE,
                             merging_strategy=None)
    assert amf.merging_strategy == MergingStrategy.LEADER_DRIVEN
    # Explicit None to the setter is a type error (no diverging role here).
    with pytest.raises(TypeError, match="merging_strategy must be a MergingStrategy"):
        amf.merging_strategy = None


def test_agentic_message_flow_inv_b_explicit_illegal():
    """(VOTING, FASTEST) raises ValueError listing legal values (S3-mm-5)."""
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                             collaboration_mode=CollaborationMode.VOTING)
    with pytest.raises(ValueError, match="not legal for collaboration_mode VOTING"):
        amf.merging_strategy = MergingStrategy.FASTEST


def test_agentic_message_flow_mode_change_resets_illegal():
    """Switching to a mode that disallows the current strategy auto-resets it (S3-mm-6)."""
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                             collaboration_mode=CollaborationMode.VOTING,
                             merging_strategy=MergingStrategy.MAJORITY)
    amf.collaboration_mode = CollaborationMode.COMPETITION
    assert amf.merging_strategy == MergingStrategy.FASTEST  # _DEFAULT[COMPETITION]


def test_agentic_message_flow_mode_change_keeps_legal():
    """Switching to a mode that still allows the strategy keeps it (S3-mm-7)."""
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                             collaboration_mode=CollaborationMode.VOTING,
                             merging_strategy=MergingStrategy.MAJORITY)
    amf.collaboration_mode = CollaborationMode.DEBATE  # debate accepts MAJORITY
    assert amf.merging_strategy == MergingStrategy.MAJORITY


@pytest.mark.parametrize("score", [-1, 101])
def test_agentic_message_flow_trust_score_out_of_range(score):
    """Trust score validation shared with the other agentic classes (S3-mm-8)."""
    with pytest.raises(ValueError, match=r"trust_score must be in \[0, 100\]"):
        AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                           trust_score=score)


def test_agentic_message_flow_trust_score_type_error():
    """Non-int trust score raises TypeError (S3-mm-8)."""
    with pytest.raises(TypeError, match="trust_score must be an int"):
        AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                           trust_score="50")


def test_agentic_message_flow_repr():
    """__repr__ includes endpoints + the three agentic fields (S3-mm-9)."""
    amf = AgenticMessageFlow(source=Task(name="A"), target=Task(name="B"),
                             collaboration_mode=CollaborationMode.VOTING,
                             merging_strategy=MergingStrategy.MINORITY,
                             trust_score=50)
    r = repr(amf)
    assert "AgenticMessageFlow" in r
    assert "source='A'" in r and "target='B'" in r
    assert "CollaborationMode.VOTING" in r
    assert "MergingStrategy.MINORITY" in r
    assert "trust_score=50" in r


def test_agentic_message_flow_in_all():
    """AgenticMessageFlow is importable from the package root (S3-mm-10)."""
    from besser.BUML.metamodel.bpmn import AgenticMessageFlow as AMF  # noqa: F401
    assert AMF is AgenticMessageFlow


# ---------------------------------------------------------------------------
# Backward compatibility (T-bc-1)
# ---------------------------------------------------------------------------

def test_existing_bpmn_imports_unchanged():
    """The agentic re-export does not break the BPMN base's public API."""
    # All of these must still import via the existing path.
    from besser.BUML.metamodel.bpmn import (  # noqa: F401
        BPMNModel,
        Gateway,
        Lane,
        Process,
        SequenceFlow,
        StartEvent,
        Task,
    )
    # And constructing the base classes still works:
    t = Task(name="vanilla")
    assert not isinstance(t, AgenticTask)
    g = Gateway(name="vanilla", gateway_type=GatewayType.EXCLUSIVE)
    assert not isinstance(g, AgenticGateway)
    lane = Lane(name="vanilla")
    assert not isinstance(lane, AgenticLane)


# ---------------------------------------------------------------------------
# __repr__ (T-repr-*)
# ---------------------------------------------------------------------------

def test_agentic_task_repr():
    t = AgenticTask(name="review", reflection_mode=ReflectionMode.CROSS, trust_score=85)
    r = repr(t)
    assert "AgenticTask" in r
    assert "name='review'" in r
    assert "ReflectionMode.CROSS" in r
    assert "trust_score=85" in r


def test_agentic_gateway_repr():
    g = AgenticGateway(name="vote", gateway_role=GatewayRole.MERGING,
                       collaboration_mode=CollaborationMode.VOTING,
                       merging_strategy=MergingStrategy.MAJORITY,
                       trust_score=85)
    r = repr(g)
    assert "AgenticGateway" in r
    assert "name='vote'" in r
    assert "GatewayRole.MERGING" in r
    assert "CollaborationMode.VOTING" in r
    assert "MergingStrategy.MAJORITY" in r
    assert "trust_score=85" in r


def test_agentic_lane_repr():
    lane = AgenticLane(name="Reviewer", role=AgentRole.MANAGER, trust_score=85)
    r = repr(lane)
    assert "AgenticLane" in r
    assert "name='Reviewer'" in r
    assert "AgentRole.MANAGER" in r
    assert "trust_score=85" in r
