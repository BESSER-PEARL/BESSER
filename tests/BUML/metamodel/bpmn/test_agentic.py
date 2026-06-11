"""Tests for the SEAA'25 BPMN agentic extension (``besser.BUML.metamodel.bpmn.agentic``).

Covers enum integrity, AgenticTask construction + setters, AgenticGateway
construction + eligibility, AgenticLane construction + setters,
backward compatibility with the BPMN base, and ``__repr__``.

P3' rationalization: CollaborationMode, MergingStrategy, AgenticMessageFlow
have been removed. Tests that depended on those have been deleted.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    AgentRole,
    Gateway,
    GatewayRole,
    GatewayType,
    Lane,
    LoopCharacteristics,
    Participant,
    Process,
    ReflectionMode,
    Task,
    TaskType,
)


# ---------------------------------------------------------------------------
# Enum integrity
# ---------------------------------------------------------------------------

def test_reflection_mode_values():
    """ReflectionMode .value strings match WME's BPMNReflectionMode."""
    assert ReflectionMode.NONE.value == "none"
    assert ReflectionMode.SELF.value == "self"
    assert ReflectionMode.CROSS.value == "cross"
    assert ReflectionMode.HUMAN.value == "human"
    assert len(ReflectionMode) == 4


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
# AgenticTask
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


def test_agentic_task_no_collaboration_mode():
    """AgenticTask has no collaboration_mode attribute (P3' rationalization)."""
    t = AgenticTask(name="t")
    assert not hasattr(t, "collaboration_mode")


# ---------------------------------------------------------------------------
# AgenticGateway — basic
# ---------------------------------------------------------------------------

def test_agentic_gateway_defaults():
    """Defaults: PARALLEL / DIVERGING / 0 trust."""
    g = AgenticGateway(name="g")
    assert g.gateway_type == GatewayType.PARALLEL
    assert g.gateway_role == GatewayRole.DIVERGING
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


def test_gateway_role_type_error():
    g = AgenticGateway(name="g")
    with pytest.raises(TypeError, match="gateway_role must be a GatewayRole"):
        g.gateway_role = "diverging"


def test_agentic_gateway_no_collaboration_mode():
    """AgenticGateway has no collaboration_mode attribute (P3' rationalization)."""
    g = AgenticGateway(name="g")
    assert not hasattr(g, "collaboration_mode")


def test_agentic_gateway_no_merging_strategy():
    """AgenticGateway has no merging_strategy attribute (P3' rationalization)."""
    g = AgenticGateway(name="g")
    assert not hasattr(g, "merging_strategy")


# ---------------------------------------------------------------------------
# AgenticLane
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
# AgenticLane.multiplicity — 3c (WME 2026-06-08 point #3 swarm size)
# ---------------------------------------------------------------------------

def test_agentic_lane_multiplicity_default():
    """Constructed without the kwarg → multiplicity is 1 (single agent)."""
    lane = AgenticLane(name="Coder")
    assert lane.multiplicity == 1


def test_agentic_lane_multiplicity_set():
    lane = AgenticLane(name="Coder", multiplicity=3)
    assert lane.multiplicity == 3


@pytest.mark.parametrize("value", [0, -1])
def test_agentic_lane_multiplicity_out_of_range(value):
    with pytest.raises(ValueError, match=r"multiplicity must be >= 1"):
        AgenticLane(name="Coder", multiplicity=value)


@pytest.mark.parametrize("value", [True, 2.0])
def test_agentic_lane_multiplicity_type_error(value):
    with pytest.raises(TypeError, match="multiplicity must be an int"):
        AgenticLane(name="Coder", multiplicity=value)


def test_agentic_lane_repr_includes_multiplicity():
    lane = AgenticLane(name="Coder", multiplicity=3)
    assert "multiplicity=3" in repr(lane)


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
    """No invariant ties governance_dsl to gateway_role (R-b-mm-3)."""
    gw = AgenticGateway(name="Split", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.DIVERGING, governance_dsl="x")
    assert gw.governance_dsl == "x"


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
# Backward compatibility
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
# __repr__
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
                       trust_score=85)
    r = repr(g)
    assert "AgenticGateway" in r
    assert "name='vote'" in r
    assert "GatewayRole.MERGING" in r
    assert "trust_score=85" in r


def test_agentic_lane_repr():
    lane = AgenticLane(name="Reviewer", role=AgentRole.MANAGER, trust_score=85)
    r = repr(lane)
    assert "AgenticLane" in r
    assert "name='Reviewer'" in r
    assert "AgentRole.MANAGER" in r
    assert "trust_score=85" in r
