"""The candidate PRODUCERS of a merging gateway are derived from the BPMN
sequence flows that flow INTO it (source task → owning lane → agent), decoupled from
the policy's voter list. Covers ``_producer_agent_names``."""
# Prime the backend `services` package before the router so its __init__ resolves the
# generation_router ↔ github_deploy_api cycle in the order the app boots uses (importing
# the router first would catch it partially initialized).
import pytest

from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GovernanceDslValidationError,
)
import besser.utilities.web_modeling_editor.backend.services

from besser.utilities.web_modeling_editor.backend.routers.generation_router import (
    _attach_entry_role_to_agents,
    _attach_governance_to_agents,
    _merge_state_for_gateway,
    _producer_agent_names,
)


class _Agent:
    def __init__(self, name):
        self.name = name

_VALID_GOVERNANCE_DSL = """\
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
# A tiny BPMN: coder-lane task and reviewer-lane task both flow into the merging
# gateway (owned by the reviewer lane). Producers must be {AgentCoder, AgentReviewer}.
GW = "gw1"
ITEMS = {
    "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
    "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
    "taskC": {"id": "taskC", "type": "BPMNTask", "owner": "laneC"},
    "taskR": {"id": "taskR", "type": "BPMNTask", "owner": "laneR"},
    GW: {"id": GW, "type": "BPMNGateway", "owner": "laneR"},
}
LANE_REF = {"laneC": "refC", "laneR": "refR"}
AGENTS = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}


def _flow(src, tgt, flow_type="sequence"):
    return {"type": "BPMNFlow", "flowType": flow_type,
            "source": {"element": src}, "target": {"element": tgt}}


def test_producers_are_incoming_flow_sources():
    rels = [_flow("taskC", GW), _flow("taskR", GW)]
    names = _producer_agent_names(GW, rels, ITEMS, LANE_REF, AGENTS)
    assert sorted(names) == ["AgentCoder", "AgentReviewer"]


def test_outgoing_flow_is_not_a_producer():
    # A flow LEAVING the gateway must not make its target a producer.
    rels = [_flow("taskC", GW), _flow(GW, "taskR")]
    names = _producer_agent_names(GW, rels, ITEMS, LANE_REF, AGENTS)
    assert names == ["AgentCoder"]


def test_non_sequence_flow_into_gateway_is_ignored():
    rels = [_flow("taskC", GW), _flow("taskR", GW, flow_type="message")]
    names = _producer_agent_names(GW, rels, ITEMS, LANE_REF, AGENTS)
    assert names == ["AgentCoder"]


def test_missing_flow_type_defaults_to_sequence():
    rel = {"type": "BPMNFlow", "source": {"element": "taskC"}, "target": {"element": GW}}
    names = _producer_agent_names(GW, [rel], ITEMS, LANE_REF, AGENTS)
    assert names == ["AgentCoder"]


def test_duplicate_branches_from_same_lane_collapse_to_one_agent():
    rels = [_flow("taskC", GW), _flow("taskC", GW)]
    names = _producer_agent_names(GW, rels, ITEMS, LANE_REF, AGENTS)
    assert names == ["AgentCoder"]


def test_dangling_source_is_skipped():
    rels = [_flow("ghost", GW), _flow("taskC", GW)]
    names = _producer_agent_names(GW, rels, ITEMS, LANE_REF, AGENTS)
    assert names == ["AgentCoder"]


# Invalid Governance DSL is rejected before attachment; ungoverned gateways are no-ops.

class _Input:
    """Minimal stand-in for ProjectInput: only `.diagrams` is read."""
    def __init__(self, diagrams):
        self.diagrams = diagrams


def _bpmn_input(gov_text):
    # coder-lane and reviewer-lane tasks both flow into a merging gateway owned by the
    # reviewer lane; the gateway carries `gov_text` as its governanceDsl.
    elements = {
        "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "taskC": {"id": "taskC", "type": "BPMNTask", "owner": "laneC"},
        "taskR": {"id": "taskR", "type": "BPMNTask", "owner": "laneR"},
        GW: {"id": GW, "type": "BPMNGateway", "owner": "laneR", "governanceDsl": gov_text},
    }
    relationships = {
        "f1": _flow("taskC", GW),
        "f2": _flow("taskR", GW),
    }
    return _Input({"BPMN": [{"model": {"elements": elements, "relationships": relationships}}]})


def _attach_and_get_owner_gov(gov_text):
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    _attach_governance_to_agents(_bpmn_input(gov_text), agents)
    owner = agents["refR"]                      # the gateway's owning lane → owner agent
    assert getattr(owner, "_governance", None)  # a policy was attached
    return owner._governance[0]


def test_malformed_gateway_dsl_raises_with_gateway_context():
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}

    with pytest.raises(
        GovernanceDslValidationError,
        match=r"Invalid Governance DSL on merging gateway 'gw1'",
    ):
        _attach_governance_to_agents(
            _bpmn_input("MajorityPolicy broken {{{"),
            agents,
        )


def test_ungoverned_gateway_is_a_noop():
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}

    _attach_governance_to_agents(_bpmn_input(""), agents)

    assert getattr(agents["refC"], "_governance", None) is None
    assert getattr(agents["refR"], "_governance", None) is None
# ---------------------------------------------------------------------------
# When WME emits the `flow=` binding, governance is ALSO keyed per
# merge STATE (agent._governance_by_state); the flat list stays as back-compat fallback.
# The live frontend emits none of this yet, so the per-state path is dormant: an agent
# without the binding falls back to the flat list exactly as before.
# ---------------------------------------------------------------------------

GOV = _VALID_GOVERNANCE_DSL

def _resolver_agent(inbound):
    a = _Agent("AgentReviewer")
    a._a2a = {"outbound": [], "inbound": inbound}
    return a


def test_merge_state_for_gateway_resolves_via_flow_marker():
    a = _resolver_agent([
        {"flow": GW, "target_state": "Address merge decision", "peer": "AgentCoder"}])
    assert _merge_state_for_gateway(a, GW) == "Address merge decision"


def test_merge_state_for_gateway_none_without_binding():
    assert _merge_state_for_gateway(_Agent("X"), GW) is None          # no _a2a at all
    a = _resolver_agent([{"flow": "other", "target_state": "S"}])
    assert _merge_state_for_gateway(a, GW) is None                    # flow mismatch
    assert _merge_state_for_gateway(a, None) is None                  # no gateway id


def test_w3_binding_keys_governance_per_state():
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    agents["refR"]._a2a = {"outbound": [], "inbound": [
        {"flow": GW, "target_state": "Address merge decision", "peer": "AgentCoder",
         "intent": "merge", "source_state": "review"}]}
    _attach_governance_to_agents(_bpmn_input(GOV), agents)
    owner = agents["refR"]
    by_state = getattr(owner, "_governance_by_state", None)
    assert by_state is not None and set(by_state) == {"Address merge decision"}
    # the SAME summary object lives in both the per-state map and the flat fallback
    assert by_state["Address merge decision"] is owner._governance[0]


def test_missing_w3_binding_falls_back_to_flat_list_only():
    owner = _attach_and_get_owner_gov(GOV)              # owner has no _a2a binding
    # _attach_and_get_owner_gov already asserts the flat list is populated; the per-state
    # map must be absent so the live (single-merge) render is byte-identical.
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    _attach_governance_to_agents(_bpmn_input(GOV), agents)
    assert getattr(agents["refR"], "_governance_by_state", None) is None
    assert len(agents["refR"]._governance) == 1


def test_multi_gateway_agent_keys_each_merge_state():
    gw2 = "gw2"
    elements = {
        "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "taskC": {"id": "taskC", "type": "BPMNTask", "owner": "laneC"},
        "taskR": {"id": "taskR", "type": "BPMNTask", "owner": "laneR"},
        GW: {"id": GW, "type": "BPMNGateway", "owner": "laneR", "governanceDsl": GOV},
        gw2: {"id": gw2, "type": "BPMNGateway", "owner": "laneR", "governanceDsl": GOV},
    }
    relationships = {
        "f1": _flow("taskC", GW), "f2": _flow("taskR", GW),
        "f3": _flow("taskC", gw2), "f4": _flow("taskR", gw2),
    }
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": relationships}}]})
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    agents["refR"]._a2a = {"outbound": [], "inbound": [
        {"flow": GW, "target_state": "Address merge decision — A"},
        {"flow": gw2, "target_state": "Address merge decision — B"}]}
    _attach_governance_to_agents(inp, agents)
    owner = agents["refR"]
    assert set(owner._governance_by_state) == {
        "Address merge decision — A", "Address merge decision — B"}
    assert len(owner._governance) == 2                  # flat fallback carries BOTH


# ---------------------------------------------------------------------------
# a producer's outbound A2A edge is stamped with the GOVERNED gateway
# its BPMN sequence flow feeds (`target_gateway`), so the producer tags its PUSH message and
# the owner dispatches it to the right merge. The per-state summary records its gateway id.
# ---------------------------------------------------------------------------

def test_producer_outbound_edge_is_stamped_with_target_gateway():
    # Producer (coder lane) drafts into the governed gateway GW; its agent-diagram a2a:out
    # carries the BPMN sequence-flow id "f1" (whose target IS the gateway).
    elements = {
        "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "taskC": {"id": "taskC", "type": "BPMNTask", "owner": "laneC"},
        GW: {"id": GW, "type": "BPMNGateway", "owner": "laneR", "governanceDsl": GOV},
    }
    # the sequence flow id is the rel key; its target is the gateway
    relationships = {"f1": _flow("taskC", GW)}
    relationships["f1"]["id"] = "f1"
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": relationships}}]})
    coder = _Agent("AgentCoder")
    coder._a2a = {"outbound": [{"peer": "AgentReviewer", "flow": "f1"}], "inbound": []}
    agents = {"refC": coder, "refR": _Agent("AgentReviewer")}
    _attach_governance_to_agents(inp, agents)
    assert coder._a2a["outbound"][0]["target_gateway"] == GW
    # (the owner-side per-state binding is covered by test_per_state_summary_records_gateway_id)


def test_non_governed_flow_target_is_not_stamped():
    elements = {
        "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "taskC": {"id": "taskC", "type": "BPMNTask", "owner": "laneC"},
        GW: {"id": GW, "type": "BPMNGateway", "owner": "laneR", "governanceDsl": GOV},
        "plain": {"id": "plain", "type": "BPMNTask", "owner": "laneR"},
    }
    relationships = {"f9": _flow("taskC", "plain")}      # flow to a non-governed task
    relationships["f9"]["id"] = "f9"
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": relationships}}]})
    coder = _Agent("AgentCoder")
    coder._a2a = {"outbound": [{"peer": "AgentReviewer", "flow": "f9"}], "inbound": []}
    _attach_governance_to_agents(inp, {"refC": coder, "refR": _Agent("AgentReviewer")})
    assert "target_gateway" not in coder._a2a["outbound"][0]   # target not governed → no tag


def test_per_state_summary_records_gateway_id():
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    agents["refR"]._a2a = {"outbound": [], "inbound": [
        {"flow": GW, "target_state": "Address merge decision"}]}
    _attach_governance_to_agents(_bpmn_input(GOV), agents)
    summary = agents["refR"]._governance_by_state["Address merge decision"]
    assert summary["gateway_id"] == GW
    assert summary["merge_state"] == "Address merge decision"


# ---------------------------------------------------------------------------
# Entry-role derivation — a lane is human-facing iff it owns a BPMN start event (or a start
# event's outgoing flow targets it, or it receives a flow from a non-agentic source). This
# is authoritative even when the lane ALSO has A2A inbound edges (the hybrid merge owner).
# Covers _attach_entry_role_to_agents.
# ---------------------------------------------------------------------------

def test_start_event_owner_lane_is_human_facing():
    # The start event is OWNED by the reviewer lane, which also owns a merging gateway the
    # coder pushes into → the reviewer must be marked human-facing despite the back-edge.
    elements = {
        "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "start": {"id": "start", "type": "BPMNStartEvent", "owner": "laneR"},
        GW: {"id": GW, "type": "BPMNGateway", "owner": "laneR"},
    }
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": {}}}]})
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    _attach_entry_role_to_agents(inp, agents)
    assert getattr(agents["refR"], "_human_facing", None) is True
    assert getattr(agents["refC"], "_human_facing", None) is None   # worker stays unmarked


def test_start_event_outgoing_flow_target_is_human_facing():
    # The start event sits outside the lanes (on the pool) and its flow targets a task in the
    # reviewer lane → that lane is the entry.
    elements = {
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "start": {"id": "start", "type": "BPMNStartEvent", "owner": "pool"},
        "taskR": {"id": "taskR", "type": "BPMNTask", "owner": "laneR"},
    }
    rels = {"f1": _flow("start", "taskR")}
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": rels}}]})
    agents = {"refR": _Agent("AgentReviewer")}
    _attach_entry_role_to_agents(inp, agents)
    assert getattr(agents["refR"], "_human_facing", None) is True


def test_flow_from_non_agentic_lane_is_human_facing():
    # A human/external lane (no agentDiagramRef) hands work to the agentic lane → entry.
    elements = {
        "human": {"id": "human", "type": "BPMNSwimlane"},               # no agentDiagramRef
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "taskH": {"id": "taskH", "type": "BPMNTask", "owner": "human"},
        "taskR": {"id": "taskR", "type": "BPMNTask", "owner": "laneR"},
    }
    rels = {"f1": _flow("taskH", "taskR")}
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": rels}}]})
    agents = {"refR": _Agent("AgentReviewer")}
    _attach_entry_role_to_agents(inp, agents)
    assert getattr(agents["refR"], "_human_facing", None) is True


def test_agent_to_agent_flow_does_not_mark_human_facing():
    # A pure A2A flow between two agentic lanes (no start event, no human source) marks
    # neither lane human-facing → the generator's no-inbound heuristic decides.
    elements = {
        "laneC": {"id": "laneC", "type": "BPMNSwimlane", "agentDiagramRef": "refC"},
        "laneR": {"id": "laneR", "type": "BPMNSwimlane", "agentDiagramRef": "refR"},
        "taskC": {"id": "taskC", "type": "BPMNTask", "owner": "laneC"},
        "taskR": {"id": "taskR", "type": "BPMNTask", "owner": "laneR"},
    }
    rels = {"f1": _flow("taskC", "taskR")}
    inp = _Input({"BPMN": [{"model": {"elements": elements, "relationships": rels}}]})
    agents = {"refC": _Agent("AgentCoder"), "refR": _Agent("AgentReviewer")}
    _attach_entry_role_to_agents(inp, agents)
    assert getattr(agents["refC"], "_human_facing", None) is None
    assert getattr(agents["refR"], "_human_facing", None) is None


def test_no_bpmn_is_noop():
    agents = {"refR": _Agent("AgentReviewer")}
    _attach_entry_role_to_agents(_Input({}), agents)
    assert getattr(agents["refR"], "_human_facing", None) is None
