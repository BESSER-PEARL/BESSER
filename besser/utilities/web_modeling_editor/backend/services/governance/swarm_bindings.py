"""Bind an agentic BPMN process onto the swarm's BAF agents.

Docker Compose generation bakes one BAF agent per deployed artifact. The BPMN
diagram decides two things those agents need, both read here from the
``BPMNModel`` built by ``process_bpmn_diagram``:

* **Governance** -- an ``AgenticGateway`` carrying a Governance DSL policy is a
  governed merge. Its owning ``AgenticLane`` names (via ``agent_diagram_ref``)
  the agent that runs the merge; the agents on the branches flowing into the
  gateway are the candidate producers. Stamped as ``agent._governance`` (one
  summary per governed gateway) and, when the agent's ``a2a:in;flow=<gateway>``
  tag binds the gateway to a state, ``agent._governance_by_state``. Producers'
  outbound A2A edges are stamped with the ``target_gateway`` they feed.
* **Entry role** -- a lane that starts the process or receives work from a
  non-agentic actor faces the human user. Stamped as ``agent._human_facing``.

The ``DockerComposeGenerator`` reads these attributes; an agent with none of
them renders exactly as without a BPMN diagram.
"""

from typing import Optional

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    BPMNModel,
    FlowNode,
    StartEvent,
)
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GovernanceDslValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.governance.govdsl_runtime import (
    summarize_governance,
)


def _element_id(element) -> Optional[str]:
    """The editor id of a BPMN element (kept in its layout by the JSON processor)."""
    return element.layout.get("id") if element.layout else None


def _agentic_lane(node) -> Optional[AgenticLane]:
    """The agent-linked lane ``node`` belongs to, or None."""
    if not isinstance(node, FlowNode):
        return None
    lane = node.lane
    if isinstance(lane, AgenticLane) and lane.agent_diagram_ref:
        return lane
    return None


def _lane_agent(node, agent_models_by_id: dict):
    """The agent implementing the agentic lane ``node`` belongs to, or None."""
    lane = _agentic_lane(node)
    return agent_models_by_id.get(lane.agent_diagram_ref) if lane is not None else None


def producer_agent_names(model: BPMNModel, gateway: AgenticGateway,
                         agent_models_by_id: dict) -> list:
    """Names of the agents whose branches flow into ``gateway``, in flow order.

    Only sequence flows produce candidates: each flow's source node resolves through
    its agentic lane to an agent. This keeps the producers independent of the
    voters the policy lists.
    """
    names: list = []
    for flow in sort_by_timestamp(model.all_sequence_flows()):
        if flow.target is not gateway:
            continue
        agent = _lane_agent(flow.source, agent_models_by_id)
        if agent is not None and agent.name not in names:
            names.append(agent.name)
    return names


def merge_state_for_gateway(agent, gateway_id: Optional[str]) -> Optional[str]:
    """The agent state a governed gateway binds to, or None.

    The binding is the ``a2a:in;flow=<gateway-id>`` tag of the agent's guarded
    transition (parsed onto ``agent._a2a['inbound']``): its ``target_state`` is
    the merge state. Without a binding the governance stays on the agent's flat
    ``_governance`` list only.
    """
    if not gateway_id:
        return None
    for edge in (getattr(agent, "_a2a", None) or {}).get("inbound", []):
        if edge.get("flow") == gateway_id:
            return edge.get("target_state") or edge.get("source_state") or None
    return None


def attach_governance_to_agents(model: BPMNModel, agent_models_by_id: dict) -> None:
    """Stamp each governed merge onto the agent of the lane that owns the gateway.

    Raises:
        GovernanceDslValidationError: if a gateway's Governance DSL is invalid
            (the message names the gateway).
    """
    gateways = [node for node in sort_by_timestamp(model.all_flow_nodes())
                if isinstance(node, AgenticGateway) and node.governance_dsl]
    governed_ids = {gateway: _element_id(gateway) for gateway in gateways}

    # A producer's outbound A2A edge carries the id of the BPMN flow it sends on;
    # when that flow enters a governed gateway, tag the edge with the gateway so
    # the owner dispatches the pushed candidate to the right merge state.
    flow_to_gateway = {
        _element_id(flow): governed_ids[flow.target]
        for flow in model.all_connecting_objects()
        if flow.target in governed_ids and _element_id(flow)
    }
    if flow_to_gateway:
        for agent in agent_models_by_id.values():
            for edge in (getattr(agent, "_a2a", None) or {}).get("outbound", []):
                gateway_id = flow_to_gateway.get(edge.get("flow"))
                if gateway_id:
                    edge["target_gateway"] = gateway_id

    for gateway in gateways:
        try:
            summary = summarize_governance(gateway.governance_dsl)
        except GovernanceDslValidationError as exc:
            label = gateway.name or governed_ids[gateway] or "<unnamed>"
            raise GovernanceDslValidationError(
                f"Invalid Governance DSL on merging gateway '{label}': {exc}"
            ) from exc
        if summary is None:
            continue
        agent = _lane_agent(gateway, agent_models_by_id)
        if agent is None:
            continue
        summary["producers"] = producer_agent_names(model, gateway, agent_models_by_id)
        gateway_id = governed_ids[gateway]
        state_name = merge_state_for_gateway(agent, gateway_id)
        if state_name:
            # The gateway id is the owner's per-merge dispatch key, matched against
            # the producers' stamped ``target_gateway``.
            summary["gateway_id"] = gateway_id
            summary["merge_state"] = state_name
            by_state = getattr(agent, "_governance_by_state", None)
            if by_state is None:
                by_state = {}
                agent._governance_by_state = by_state
            by_state[state_name] = summary
        agent._governance = (getattr(agent, "_governance", None) or []) + [summary]


def attach_entry_role_to_agents(model: BPMNModel, agent_models_by_id: dict) -> None:
    """Mark the agents of human-facing lanes with ``agent._human_facing = True``.

    A lane is human-facing (the swarm's user-facing trigger) when it:

    1. holds a start event, or
    2. holds the target of a start event's outgoing flow (a start event outside
       any lane that kicks off a task in it), or
    3. receives a flow from a non-agentic source -- a node in a lane with no agent,
       in no lane, or a pool (a human or external actor).

    This is authoritative over the generator's "no inbound A2A peer" heuristic, so
    a coordinator that starts the process and also owns governed merges (and thus
    receives A2A pushes) is still the entry. A model without such a lane stamps
    nothing and the generator keeps its heuristic.
    """
    def _stamp(node) -> None:
        agent = _lane_agent(node, agent_models_by_id)
        if agent is not None:
            agent._human_facing = True

    for node in model.all_flow_nodes():
        if isinstance(node, StartEvent):
            _stamp(node)

    for flow in model.all_connecting_objects():
        source, target = flow.source, flow.target
        if isinstance(source, StartEvent):
            _stamp(target)
            continue
        if _agentic_lane(target) is None:
            continue
        # An agent-to-agent flow is A2A and an intra-lane flow is internal.
        if _agentic_lane(source) is not None:
            continue
        _stamp(target)
