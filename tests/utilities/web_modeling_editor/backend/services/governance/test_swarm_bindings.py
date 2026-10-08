"""Binding an agentic BPMN process onto the swarm's agents.

Covers ``services.governance.swarm_bindings``: governed merges (owner agent,
candidate producers, per-state keying, producer edge tagging) and the
human-facing entry role, all read from the ``BPMNModel`` the JSON processor
builds from an editor-shaped BPMN diagram.
"""

import pytest

pytest.importorskip("governancedsl")

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.bpmn_diagram_processor import (  # noqa: E402
    process_bpmn_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (  # noqa: E402
    GovernanceDslValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.governance.swarm_bindings import (  # noqa: E402
    attach_entry_role_to_agents,
    attach_governance_to_agents,
    merge_state_for_gateway,
    producer_agent_names,
)


class _Agent:
    def __init__(self, name):
        self.name = name


GOV = """\
// WME-generated header
Scopes:
    Tasks :
        mergeTask
Participants:
    Individuals :
        (Agent) Coder {
            confidence : 0.8
        },
        (Agent) Reviewer {
            confidence : 0.6
        }
MajorityPolicy mergePolicy {
    Scope: mergeTask
    DecisionType as BooleanDecision
    Participant list : Coder, Reviewer
    Parameters:
        ratio : 0.5
}
"""
GW = "gw1"


def _lane(lane_id, ref=None):
    lane = {"id": lane_id, "name": lane_id, "type": "BPMNSwimlane", "owner": "pool"}
    if ref is not None:
        lane.update(isAgentic=True, role="solution", agentDiagramRef=ref)
    return lane


def _task(task_id, owner):
    return {"id": task_id, "name": task_id, "type": "BPMNTask", "owner": owner}


def _gateway(gateway_id, owner, governance=GOV):
    return {"id": gateway_id, "name": gateway_id, "type": "BPMNGateway", "owner": owner,
            "gatewayType": "parallel", "isAgentic": True, "gatewayRole": "merging",
            "governanceDsl": governance}


def _flow(flow_id, source, target, flow_type="sequence"):
    return {"id": flow_id, "name": "", "type": "BPMNFlow", "flowType": flow_type,
            "source": {"element": source}, "target": {"element": target}}


def _model(elements: list, flows: list):
    pool = {"id": "pool", "name": "Swarm", "type": "BPMNPool", "owner": None}
    payload = {"title": "Swarm", "model": {
        "type": "BPMNDiagram",
        "elements": {e["id"]: e for e in [pool] + elements},
        "relationships": {f["id"]: f for f in flows},
    }}
    return process_bpmn_diagram(payload)


def _merge_model(governance=GOV, extra_flows=()):
    """Coder-lane and reviewer-lane tasks both flow into a merge owned by the reviewer lane."""
    return _model(
        [_lane("laneC", "refC"), _lane("laneR", "refR"), _task("taskC", "laneC"),
         _task("taskR", "laneR"), _gateway(GW, "laneR", governance)],
        [_flow("f1", "taskC", GW), _flow("f2", "taskR", GW), *extra_flows],
    )


def _agents():
    return {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}


def _gateway_of(model, gateway_id=GW):
    return next(n for n in model.all_flow_nodes() if n.layout["id"] == gateway_id)


# --- producers -------------------------------------------------------------------

def test_producers_are_the_agents_of_incoming_sequence_flows():
    model = _merge_model()
    assert producer_agent_names(model, _gateway_of(model), _agents()) == [
        "AgentCoder", "AgentReviewer"]


def test_outgoing_and_message_flows_are_not_producers():
    model = _model(
        [_lane("laneC", "refC"), _lane("laneR", "refR"), _task("taskC", "laneC"),
         _task("taskR", "laneR"), _gateway(GW, "laneR")],
        [_flow("f1", "taskC", GW), _flow("f2", GW, "taskR")],
    )
    assert producer_agent_names(model, _gateway_of(model), _agents()) == ["AgentCoder"]


def test_two_branches_from_one_lane_collapse_to_one_producer():
    model = _model(
        [_lane("laneC", "refC"), _lane("laneR", "refR"), _task("taskC", "laneC"),
         _task("taskC2", "laneC"), _gateway(GW, "laneR")],
        [_flow("f1", "taskC", GW), _flow("f2", "taskC2", GW)],
    )
    assert producer_agent_names(model, _gateway_of(model), _agents()) == ["AgentCoder"]


# --- governance ------------------------------------------------------------------

def test_governance_lands_on_the_gateway_owner_with_producers():
    agents = _agents()
    attach_governance_to_agents(_merge_model(), agents)
    (summary,) = agents["refR"]._governance
    assert summary["policy_type"] == "MajorityPolicy"
    assert summary["producers"] == ["AgentCoder", "AgentReviewer"]
    assert not hasattr(agents["refC"], "_governance")
    assert not hasattr(agents["refR"], "_governance_by_state")


def test_malformed_gateway_dsl_raises_with_gateway_context():
    with pytest.raises(GovernanceDslValidationError,
                       match=r"Invalid Governance DSL on merging gateway 'gw1'"):
        attach_governance_to_agents(_merge_model("MajorityPolicy broken {{{"), _agents())


def test_ungoverned_gateway_is_a_noop():
    agents = _agents()
    attach_governance_to_agents(_merge_model(governance=""), agents)
    assert not hasattr(agents["refR"], "_governance")


def test_merge_state_for_gateway_resolves_via_flow_marker():
    agent = _Agent("AgentReviewer")
    agent._a2a = {"outbound": [], "inbound": [
        {"flow": GW, "target_state": "Address merge decision", "peer": "AgentCoder"}]}
    assert merge_state_for_gateway(agent, GW) == "Address merge decision"
    assert merge_state_for_gateway(agent, "other") is None
    assert merge_state_for_gateway(agent, None) is None
    assert merge_state_for_gateway(_Agent("X"), GW) is None


def test_bound_gateway_is_keyed_per_merge_state():
    agents = _agents()
    agents["refR"]._a2a = {"outbound": [], "inbound": [
        {"flow": GW, "target_state": "Address merge decision"}]}
    attach_governance_to_agents(_merge_model(), agents)
    owner = agents["refR"]
    summary = owner._governance_by_state["Address merge decision"]
    assert summary is owner._governance[0]
    assert summary["gateway_id"] == GW
    assert summary["merge_state"] == "Address merge decision"


def test_owner_of_two_gateways_keys_each_merge_state():
    model = _model(
        [_lane("laneC", "refC"), _lane("laneR", "refR"), _task("taskC", "laneC"),
         _task("taskR", "laneR"), _gateway(GW, "laneR"), _gateway("gw2", "laneR")],
        [_flow("f1", "taskC", GW), _flow("f2", "taskR", GW),
         _flow("f3", "taskC", "gw2"), _flow("f4", "taskR", "gw2")],
    )
    agents = _agents()
    agents["refR"]._a2a = {"outbound": [], "inbound": [
        {"flow": GW, "target_state": "Merge A"}, {"flow": "gw2", "target_state": "Merge B"}]}
    attach_governance_to_agents(model, agents)
    assert set(agents["refR"]._governance_by_state) == {"Merge A", "Merge B"}
    assert len(agents["refR"]._governance) == 2


def test_producer_edge_into_a_governed_gateway_is_tagged():
    agents = _agents()
    agents["refC"]._a2a = {"outbound": [{"peer": "AgentReviewer", "flow": "f1"},
                                        {"peer": "AgentReviewer", "flow": "f9"}],
                           "inbound": []}
    attach_governance_to_agents(
        _merge_model(extra_flows=[_flow("f9", "taskC", "taskR")]), agents)
    first, second = agents["refC"]._a2a["outbound"]
    assert first["target_gateway"] == GW
    assert "target_gateway" not in second


# --- entry role ------------------------------------------------------------------

def test_start_event_lane_is_human_facing_even_when_it_owns_a_merge():
    model = _model(
        [_lane("laneC", "refC"), _lane("laneR", "refR"), _task("taskC", "laneC"),
         {"id": "start", "name": "", "type": "BPMNStartEvent", "owner": "laneR"},
         _gateway(GW, "laneR")],
        [_flow("f1", "taskC", GW)],
    )
    agents = _agents()
    attach_entry_role_to_agents(model, agents)
    assert agents["refR"]._human_facing is True
    assert not hasattr(agents["refC"], "_human_facing")


def test_start_event_outside_lanes_marks_the_lane_it_starts():
    model = _model(
        [_lane("laneR", "refR"), _task("taskR", "laneR"),
         {"id": "start", "name": "", "type": "BPMNStartEvent", "owner": "pool"}],
        [_flow("f1", "start", "taskR")],
    )
    agents = _agents()
    attach_entry_role_to_agents(model, agents)
    assert agents["refR"]._human_facing is True


def test_flow_from_a_non_agentic_lane_is_human_facing():
    model = _model(
        [_lane("human"), _lane("laneR", "refR"), _task("taskH", "human"), _task("taskR", "laneR")],
        [_flow("f1", "taskH", "taskR")],
    )
    agents = _agents()
    attach_entry_role_to_agents(model, agents)
    assert agents["refR"]._human_facing is True


def test_agent_to_agent_flow_marks_no_lane():
    model = _model(
        [_lane("laneC", "refC"), _lane("laneR", "refR"), _task("taskC", "laneC"),
         _task("taskR", "laneR")],
        [_flow("f1", "taskC", "taskR")],
    )
    agents = _agents()
    attach_entry_role_to_agents(model, agents)
    assert not hasattr(agents["refC"], "_human_facing")
    assert not hasattr(agents["refR"], "_human_facing")
