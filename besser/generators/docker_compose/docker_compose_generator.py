"""Docker Compose generator — turns a UML DeploymentModel into docker-compose.yml."""
import inspect
import logging
import os
import re

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Locality,
)
from besser.generators import GeneratorInterface
from besser.generators.agents.baf_generator import BAFGenerator
# Tested tally engine; its source is baked into governed agents.
from besser.generators.agents import governance_engine as _gov_engine
from besser.utilities import sort_by_timestamp

logger = logging.getLogger(__name__)


def _safe_service_name(name: str) -> str:
    """Convert a deployment element label to a Docker Compose-safe service/network name.

    Applies camelCase → snake_case conversion first, then lowercases everything
    and replaces runs of non-alphanumeric characters with a single underscore.

    Examples: "AgentRuntime" → "agent_runtime", "llm_gateway" → "llm_gateway",
              "LLMEndpoint" → "llm_endpoint", "Code Tester" → "code_tester".
    """
    s = (name or "").strip()
    # Insert underscore between a lowercase/digit and an uppercase (e.g. tR → t_R)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Insert underscore between a run of uppercase and the start of a new word
    # e.g. "LLMEndpoint" → "LLM_Endpoint"
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s or 'unnamed'


# v2 A2A — default system prompt when the Agent diagram carries no LLM-reply prompt.
def _default_prompt(name: str, role: str) -> str:
    if role == 'entry':
        return (f"You are {name}, the human-facing coordinator of an agent swarm. "
                f"Delegate the task to your team, then synthesize their results into "
                f"one concise answer for the user.")
    return (f"You are {name}, an agent in a collaborative swarm. Complete the task "
            f"you are given concisely; if you received collaborator inputs, use them.")


def _facing_flags(agent, has_inbound_peer: bool):
    """Decouple the two capabilities the legacy single ``role`` conflated:

    - ``human_facing`` — runs the websocket/Streamlit UI and publishes the host ports.
      AUTHORITATIVE when the backend stamped ``agent._human_facing`` (derived from the
      BPMN start event / a non-agentic inbound flow; see ``_attach_entry_role_to_agents``).
      When the flag is ABSENT (no BPMN, single-diagram, or a unit test) it falls back to
      the legacy heuristic ``not has_inbound_peer`` — i.e. an agent with no inbound A2A
      peer is the entry — so legacy renders stay byte-identical.
    - ``a2a_server`` — has ≥1 inbound A2A peer that resolves to a running service, so it
      must run the headless A2A server on 8000 to receive pushes. Computed from the
      topology exactly as before, independent of ``human_facing``.

    An agent can be BOTH (e.g. a reviewer/coordinator that owns the BPMN start event AND
    owns the governed merge gateways the producers push into): it runs both platforms and
    publishes both port sets. ``role`` is kept (derived) only for back-compat (the legacy
    descriptor key, the bake log line, and the default-prompt framing).
    """
    flag = getattr(agent, '_human_facing', None)
    human_facing = (not has_inbound_peer) if flag is None else bool(flag)
    a2a_server = bool(has_inbound_peer)
    role = 'entry' if human_facing else 'worker'
    return human_facing, a2a_server, role


def _governance_for(agent):
    """The first governance summary stashed on the agent by the backend handler, or None.
    The generator forwards it verbatim; no parsing here.

    v1 wires exactly ONE governed merge per agent (the agent owns one merge/tally path in
    its generated BAF state). An agent whose lane owns more than one governed merging
    gateway is the unsupported case: only the first policy is wired, so emit a visible
    warning rather than silently dropping the rest (per-gateway BAF states are future
    work). The summaries are ordered as the gateways were encountered."""
    blobs = getattr(agent, '_governance', None) or []
    # A per-state-bound agent governs each merge at its own state (`_governance_by_state`
    # → the faithful `_MERGES` dispatch), so nothing is dropped: suppress the legacy
    # "only the first is wired" warning. It still fires for an un-bound multi-gateway agent.
    if len(blobs) > 1 and not getattr(agent, '_governance_by_state', None):
        logger.warning(
            "[governance] agent %r owns %d governed merging gateways; v1 wires only the "
            "first (%s). The remaining %d are dropped — split them across lanes/agents to "
            "govern each separately.",
            getattr(agent, 'name', '?'), len(blobs),
            blobs[0].get('policy_type'), len(blobs) - 1)
    return blobs[0] if blobs else None


# bake the WHOLE engine module (not per-function getsource) so the baked copy
# keeps its `import re` + the `_BALLOT_RE` module global that `parse_ballot` depends on.
# The count then runs in-container with no besser/ANTLR import (single source of truth).
_GOV_ENGINE_SRC = inspect.getsource(_gov_engine)
_VOTING_POLICIES = frozenset(("VotingPolicy", "MajorityPolicy", "AbsoluteMajorityPolicy"))


def _governance_star_from(gov, service_names: set, self_id: str):
    """build a governed *voting* merge owner's candidate-vote STAR from an
    EXPLICIT governance summary.

    Returns (peers, to_peers, gov) — `peers`/`to_peers` REPLACE the topology peer set so
    the owner addresses every producer and voter directly; `gov` is the governance summary
    augmented with the producer service list, vote weights, unresolved participants, owner
    flags, the self service id, and the baked engine source. Returns None when `gov` is
    falsy, the policy is non-voting (Leader/Consensus/fallback keep the single-round topology path), OR no producer resolves to a running service (a candidate-selection vote needs
    at least one candidate — degrade to the single-round path).

    PRODUCERS and VOTERS are decoupled:
      * producers = the BPMN branches flowing into the gateway (``gov['producers']``,
        stamped by the backend); each yields one round-1 candidate output;
      * voters    = the policy's own participant list (``gov['participants']``); each casts
        one round-2 ballot.
    The owner produces a candidate only if it is itself a producer (``owner_produces``)
    and votes only if it is itself a participant (``owner_votes``). Either set may include
    the owner; both run in-process, never as a peer.

    This is the per-summary core: ``_governance_star`` calls it with the agent's first/only
    summary (legacy single-merge path); ``_governed_merge_states`` calls it once per bound
    merge state (per-state governance). The frozen tally engine is untouched.
    """
    if not gov:
        return None
    gov = dict(gov)  # copy — never mutate the shared summary on agent._governance
    gov['is_voting'] = gov.get('policy_type') in _VOTING_POLICIES
    if not gov['is_voting']:
        return None

    # Voters — from the policy participant list. weights[service] -> vote weight.
    weights, voter_services, unresolved, owner_votes = {}, [], [], False
    for p in (gov.get('participants') or []):
        svc = _safe_service_name(p.get('name', ''))
        conf = p.get('confidence')
        w = conf if isinstance(conf, (int, float)) and conf > 0 else 1.0
        if not svc:
            continue
        if svc == self_id:
            owner_votes = True
            weights[svc] = w            # owner votes in-process; keep its weight
        elif svc in service_names:
            if svc not in weights:
                weights[svc] = w
                voter_services.append(svc)
        else:
            unresolved.append(p.get('name', svc))   # voter with no service -> visible abstain

    # Producers — from the BPMN flows into the gateway. owner produces in-process.
    producer_services, owner_produces, seen_prod = [], False, set()
    for name in (gov.get('producers') or []):
        svc = _safe_service_name(name)
        if not svc:
            continue
        if svc == self_id:
            owner_produces = True
        elif svc in service_names:
            if svc not in seen_prod:
                seen_prod.add(svc)
                producer_services.append(svc)
        else:
            unresolved.append(name)                 # producing branch with no service

    if not producer_services and not owner_produces:
        # A voting policy that resolves NO candidate producer cannot run a vote; the merge
        # silently degraded to the single-round path before. Surface it: an empty
        # gov['producers'] means the BPMN flow→lane→agent trace found nothing; a non-empty
        # list that still resolves to no service means the producer agent names don't match
        # any deployment artifact/service name.
        logger.warning(
            "[governance] voting policy on %r resolves no candidate producer to a running "
            "service — the vote cannot run, so the merge falls back to a single round. "
            "traced producers=%s | swarm services=%s",
            self_id, gov.get('producers'), sorted(service_names))
        return None     # no candidates possible -> fall back to the item-35 path

    # The owner must reach producers (round 1) and the other voters (round 2): the peer
    # set is their union, minus the owner (it runs both rounds in-process).
    peers, seen_peer = [], set()
    for i, svc in enumerate(producer_services + voter_services):
        if svc in seen_peer:
            continue
        seen_peer.add(svc)
        peers.append({'service': svc, 'kind': None, 'order': i, 'state': ''})

    gov['weights'] = weights
    gov['producer_services'] = producer_services
    gov['owner_produces'] = owner_produces
    gov['owner_votes'] = owner_votes
    gov['unresolved'] = sorted(set(unresolved))
    gov['self_service'] = self_id
    gov['engine_src'] = _GOV_ENGINE_SRC
    return peers, [p['service'] for p in peers], gov


def _governance_star(agent, service_names: set, self_id: str):
    """the per-agent star: build from the agent's first/only governance summary
    (``_governance_for`` → ``blobs[0]``). Thin wrapper over ``_governance_star_from`` for
    the legacy single-merge path; behavior stays byte-identical for that path."""
    return _governance_star_from(_governance_for(agent), service_names, self_id)


def _governed_merge_states(agent, service_names: set, self_id: str) -> list:
    """State-aware governance grouping for per-merge bindings.

    For each merge state bound on ``agent._governance_by_state`` (keyed by the ``"Address merge decision"`` state name; populated by ``_attach_governance_to_agents``
    only when the gateway carries the ``flow=`` binding), compute that state's OWN
    candidate-vote star by REUSING ``_governance_star_from`` — one star per merge state
    instead of one per agent. The frozen tally engine is untouched; each state is just a
    new caller.

    Returns ``[]`` when no per-state binding exists, so legacy/non-bound agents carry no
    ``states`` richness and the live render (which never reads ``states``) stays
    byte-identical. Each entry: ``{name, peers, guards, governance, is_merge}`` where
    ``guards`` are the authored inbound transitions targeting the merge state (the flags),
    surfaced from ``agent._a2a`` for the future faithful render to route on.
    """
    by_state = getattr(agent, '_governance_by_state', None) or {}
    if not by_state:
        return []
    tags = getattr(agent, '_a2a', None) or {}
    inbound = tags.get('inbound', []) if isinstance(tags, dict) else []
    states = []
    for state_name, gov in by_state.items():
        star = _governance_star_from(gov, service_names, self_id)
        peers = star[0] if star is not None else []
        star_gov = star[2] if star is not None else None
        # The non-voting per-state summary still drives a single-round merge at the state,
        # so surface the raw gov (sans engine bake) when the star degrades/declines.
        gov_payload = star_gov if star_gov is not None else dict(gov)
        guards = [
            {'intent': e.get('intent', ''), 'peer': e.get('peer', ''),
             'source_state': e.get('source_state', '')}
            for e in inbound if e.get('target_state') == state_name
        ]
        is_voting = bool(star_gov and star_gov.get('is_voting'))
        # Flat, JSON-safe per-merge config the worker dispatches on at runtime. Emitted into
        # the agent as a Python literal (repr) so bools/None are Python, not JSON, and the
        # frozen tally engine is the only shared piece. Voting-only fields come from the star.
        merge_config = {
            'state': state_name,
            'policy_type': gov_payload.get('policy_type'),
            'ratio': gov_payload.get('ratio'),
            'is_voting': is_voting,
            'requires_human': bool(gov_payload.get('requires_human')),
            'instruction': gov_payload.get('instruction') or '',
            'summary': gov_payload.get('summary') or '',
            'weights': (star_gov.get('weights') if star_gov else {}) or {},
            'producers': (star_gov.get('producer_services') if star_gov else []) or [],
            'owner_produces': bool(star_gov and star_gov.get('owner_produces')),
            'owner_votes': bool(star_gov and star_gov.get('owner_votes')),
            'unresolved': (star_gov.get('unresolved') if star_gov else []) or [],
            'self': (star_gov.get('self_service') if star_gov else self_id),
        }
        states.append({
            'name': state_name,
            # The owner dispatches an incoming PUSH message to this block when the producer's
            # `target_gateway` equals merge_key (the binding gateway id stamped at attach).
            'merge_key': gov.get('gateway_id'),
            'guard_intent': guards[0]['intent'] if guards else '',
            'peers': peers,
            'guards': guards,
            'governance': gov_payload,
            'is_voting': is_voting,
            'is_merge': True,
            'merge_config': merge_config,
            'merge_config_py': repr(merge_config),
        })
    return states


def _merge_sends_for(agent, service_names: set, self_id: str) -> list:
    """The ordered, non-deduped list of this agent's outbound governed-merge edges
    that feed a governed gateway (``target_gateway`` stamped at attach). Each is one stage of
    the faithful pipeline: the producer threads the task through them in turn, tagging each
    PUSH with its gateway so the owner runs that merge and returns the merged result, which
    feeds the next stage. Deduping by service (as ``peers`` does) would collapse two merges
    to the same owner into one — exactly what this list avoids. Empty for legacy agents
    (no ``target_gateway`` on any edge) → the pipeline render stays dormant + byte-identical.
    """
    tags = getattr(agent, '_a2a', None) or {}
    sends = []
    for edge in (tags.get('outbound', []) if isinstance(tags, dict) else []):
        gw = edge.get('target_gateway')
        if not gw:
            continue
        svc = _resolve_peer_service(edge, service_names)
        if svc and svc != self_id:
            sends.append({'service': svc, 'target_gateway': gw, 'kind': edge.get('kind'),
                          'order': edge.get('order', 9999), 'state': edge.get('state', '')})
    sends.sort(key=lambda s: s['order'])
    return sends


def _union_merge_state_peers(descriptor: dict) -> None:
    """DNS reachability: the agent must resolve every per-state peer even
    though ``_fanout(only=…)`` slices the fan-out per merge state at runtime. Union the
    per-state peer sets (from ``descriptor['states']``) onto the descriptor peer list,
    deduped and order-preserving. No-op when there are no bound merge states (legacy/non-bound
    agents), so the live render stays byte-identical."""
    states = descriptor.get('states') or []
    if not states:
        return
    peers = descriptor.setdefault('peers', [])
    seen = {p['service'] for p in peers}
    order = len(peers)
    for st in states:
        for p in st.get('peers', []):
            if p['service'] in seen:
                continue
            seen.add(p['service'])
            merged = dict(p)
            merged['order'] = order
            order += 1
            peers.append(merged)
    descriptor['to_peers'] = [p['service'] for p in peers]


def _prefer_ui_single_merge_governance(descriptor: dict, explicit_human_facing: bool) -> None:
    """Prefer the UI/work_body governance path for the explicit human-facing voting owner."""
    states = descriptor.get('states') or []
    governance = descriptor.get('governance') or {}
    if not explicit_human_facing:
        return
    if descriptor.get('merge_sends'):
        return
    if len(states) != 1:
        return
    if not governance.get('is_voting'):
        return
    descriptor['states'] = []


def _a2a_descriptor(agent, service_names: set, self_service: str = None) -> dict:
    """Classify a baked agent for A2A wiring from its boundary states.

    - `to_<peer>` / `from_<peer>` states name a peer; `_safe_service_name(suffix)`
      is matched against the swarm's service names. Peers NOT in `service_names`
      (e.g. a non-agentic `from_<Human>` handoff) are ignored — that's how the
      entry stays the entry despite an inbound boundary (R3, DAG-exact).
    - human_facing / a2a_server flags (see ``_facing_flags``): ``a2a_server`` iff it has
      ≥1 inbound peer that IS a service; ``human_facing`` is authoritative from
      ``agent._human_facing`` (the BPMN start-event derivation) and falls back to the legacy
      "no inbound peer ⇒ entry" heuristic when the flag is absent. ``role`` is derived for
      back-compat. An agent can be BOTH (a human-facing merge owner).
    - prompt = the first LLMReply prompt found on a non-boundary state, else a default.
    """
    # This agent's own service id, so a self-referential `from_<self>`
    # boundary (the WME lane→Agent derivation mislabels the lane's own start/handoff
    # with the lane's own name) does NOT count as an inbound peer. `self_service` is
    # the artifact's safe service name (passed by the bake loop); fall back to the
    # agent name's safe form when called standalone (e.g. a unit test).
    self_id = self_service or _safe_service_name(getattr(agent, 'name', 'Agent'))
    to_peers, from_peers, prompt = [], [], None
    for st in getattr(agent, 'states', []) or []:
        nm = (getattr(st, 'name', '') or '')
        if nm.startswith('to_'):
            peer = _safe_service_name(nm[3:])
            if peer in service_names and peer != self_id:
                to_peers.append(peer)
        elif nm.startswith('from_'):
            peer = _safe_service_name(nm[5:])
            # Worker iff a `from_<peer>` resolves to a DIFFERENT swarm service.
            # `peer == self_id` is the self-referential `from_<self>` and
            # must be ignored — otherwise the entry is wrongly demoted to a worker.
            if peer in service_names and peer != self_id:
                from_peers.append(peer)
        else:
            body = getattr(st, 'body', None)
            actions = getattr(body, 'actions', None) if body else None
            if actions and actions[0].__class__.__name__ == 'LLMReply':
                prompt = getattr(actions[0], 'prompt', None) or prompt
    human_facing, a2a_server, role = _facing_flags(agent, bool(from_peers))
    name = getattr(agent, 'name', 'Agent')
    sorted_peers = sorted(set(to_peers))
    descriptor = {
        'role': role,
        'human_facing': human_facing,
        'a2a_server': a2a_server,
        'agent_id': _safe_service_name(name),
        'to_peers': sorted_peers,
        # `peers` shim so the kind-aware template is single-path. The legacy
        # convention has no `kind`, so every peer renders as a plain channel (kind=None),
        # i.e. exactly today's broadcast fan-out behavior.
        'peers': [{'service': p, 'kind': None, 'order': i, 'state': ''}
                  for i, p in enumerate(sorted_peers)],
        'source': 'convention',
        'prompt': prompt or _default_prompt(name, role),
        'governance': _governance_for(agent),
        'greeting': f"Hi! I'm {name}. Give me a task for the team.",
    }
    # a governed voting merge addresses the policy participant STAR, not the
    # BPMN to_peers; override the peer set + attach weights/engine.
    star = _governance_star(agent, service_names, self_id)
    if star is not None:
        descriptor['peers'], descriptor['to_peers'], descriptor['governance'] = star
    # State-aware governance. [] for legacy
    # agents → no-op union → byte-identical render.
    descriptor['states'] = _governed_merge_states(agent, service_names, self_id)
    _union_merge_state_peers(descriptor)
    # Ordered pipeline of governed-merge sends (un-deduped). Drives the
    # entry/initiator un-flatten: thread the task through each merge in turn. [] for legacy.
    descriptor['merge_sends'] = _merge_sends_for(agent, service_names, self_id)
    _prefer_ui_single_merge_governance(descriptor, getattr(agent, '_human_facing', None) is True)
    # Bake the frozen tally engine once for any agent that OWNS a governed merge
    # (each merge in _MERGES shares it) OR INITIATES a pipeline (finalizes a human-approved
    # stage at the entry, O1). Absent for legacy agents → no bake → byte-identical.
    if descriptor['states'] or descriptor['merge_sends']:
        descriptor['engine_src'] = _GOV_ENGINE_SRC
    # Does any outbound edge feed a governed gateway? A producer (even one
    # without merge states of its own, e.g. the entry) must then tag its PUSH messages with
    # the target gateway. False for legacy agents → _a2a_call renders byte-identically.
    descriptor['has_merge_targets'] = bool(descriptor['merge_sends']) or any(
        p.get('target_gateway') for p in descriptor.get('peers', []))
    return descriptor


def _resolve_peer_service(edge: dict, service_names: set):
    """Map an a2a edge's (ref|peer) to a swarm service name, else None.

    `ref` (the peer's AgentDiagram UUID) is the authoritative link, but the bake loop
    keys services by `_safe_service_name(artifact.name)`, not by UUID — so unless a
    {uuid → svc} index is threaded in, address by `peer` name, matching how
    `_a2a_descriptor` already resolves to_/from_ peers. `ref` is recorded on the edge
    for traceability and future UUID-keyed addressing.
    """
    peer = _safe_service_name(edge.get("peer", ""))
    return peer if peer in service_names else None


def _a2a_descriptor_from_tags(agent, service_names: set, self_service: str = None) -> dict:
    """Build an A2A descriptor from agent._a2a tags — the preferred path.

    Mirrors `_a2a_descriptor`'s contract (role/agent_id/to_peers/prompt/greeting) and
    adds `peers` (ordered, with kind) and `inbound` for the per-kind template. Peers not
    in `service_names` (dangling ref/name) are dropped — same tolerance as the legacy
    `from_<Human>` handoff.
    """
    tags = getattr(agent, '_a2a', None) or {}
    self_id = self_service or _safe_service_name(getattr(agent, 'name', 'Agent'))
    name = getattr(agent, 'name', 'Agent')

    peers, seen = [], set()                       # ordered, deduped, self-filtered
    for edge in tags.get('outbound', []):         # already order-sorted by the parser
        svc = _resolve_peer_service(edge, service_names)
        if svc and svc != self_id and svc not in seen:
            seen.add(svc)
            peers.append({'service': svc, 'kind': edge.get('kind'),
                          'order': edge.get('order', 9999),
                          'state': edge.get('state', ''),
                          # The governed gateway this outbound edge feeds (set
                          # by _attach_governance_to_agents from the BPMN flow→gateway map);
                          # the producer tags its PUSH message with it so the owner routes
                          # to the right merge state. None for non-merge edges.
                          'target_gateway': edge.get('target_gateway')})

    inbound_peers = {
        _resolve_peer_service(e, service_names)
        for e in tags.get('inbound', [])
    } - {None, self_id}
    human_facing, a2a_server, role = _facing_flags(agent, bool(inbound_peers))

    # Prompt: reuse the first LLMReply prompt on a non-boundary state, else a default
    # (identical heuristic to _a2a_descriptor for parity).
    prompt = None
    for st in getattr(agent, 'states', []) or []:
        body = getattr(st, 'body', None)
        actions = getattr(body, 'actions', None) if body else None
        if actions and actions[0].__class__.__name__ == 'LLMReply':
            prompt = getattr(actions[0], 'prompt', None) or prompt
    descriptor = {
        'role': role,
        'human_facing': human_facing,
        'a2a_server': a2a_server,
        'agent_id': _safe_service_name(name),
        'to_peers': [p['service'] for p in peers],     # back-compat (existing template/tests)
        'peers': peers,                                # NEW: per-kind, ordered
        'inbound': sorted(inbound_peers),              # NEW
        'source': 'tags',                              # provenance (vs 'convention')
        'prompt': prompt or _default_prompt(name, role),
        'governance': _governance_for(agent),
        'greeting': f"Hi! I'm {name}. Give me a task for the team.",
    }
    # a governed voting merge addresses the policy participant STAR, not the
    # BPMN to_peers; override the peer set + attach weights/engine.
    star = _governance_star(agent, service_names, self_id)
    if star is not None:
        descriptor['peers'], descriptor['to_peers'], descriptor['governance'] = star
    # State-aware governance. [] for legacy
    # agents → no-op union → byte-identical render.
    descriptor['states'] = _governed_merge_states(agent, service_names, self_id)
    _union_merge_state_peers(descriptor)
    # Ordered pipeline of governed-merge sends (un-deduped). Drives the
    # entry/initiator un-flatten: thread the task through each merge in turn. [] for legacy.
    descriptor['merge_sends'] = _merge_sends_for(agent, service_names, self_id)
    _prefer_ui_single_merge_governance(descriptor, getattr(agent, '_human_facing', None) is True)
    # Bake the frozen tally engine once for any agent that OWNS a governed merge
    # (each merge in _MERGES shares it) OR INITIATES a pipeline (finalizes a human-approved
    # stage at the entry, O1). Absent for legacy agents → no bake → byte-identical.
    if descriptor['states'] or descriptor['merge_sends']:
        descriptor['engine_src'] = _GOV_ENGINE_SRC
    # Does any outbound edge feed a governed gateway? A producer (even one
    # without merge states of its own, e.g. the entry) must then tag its PUSH messages with
    # the target gateway. False for legacy agents → _a2a_call renders byte-identically.
    descriptor['has_merge_targets'] = bool(descriptor['merge_sends']) or any(
        p.get('target_gateway') for p in descriptor.get('peers', []))
    return descriptor


class DockerComposeGenerator(GeneratorInterface):
    """Generate a docker-compose.yml from a UML DeploymentModel.

    Maps UML Deployment elements to Compose services and networks following
    the deployment-to-Compose mapping:

    - Artifact → service (locality decides ``build:`` vs ``image:``)
    - DeploymentRelation.multiplicity.max → ``deploy.replicas`` (when > 1)
    - Node → named network under ``networks:``
    - CommunicationPath → bridging ``<a>_<b>_link`` network both nodes join
    - DeploymentDependency (Artifact → Artifact) → ``depends_on:``
    - Interface / InterfaceProvided / InterfaceRequired → not mapped (v1)

    ``Artifact.manifests`` is emitted as a ``# manifests:`` comment only (v1
    — not resolved against the Component model).
    """

    def __init__(self, model: DeploymentModel, output_dir: str = None,
                 agent_models_by_id: dict = None):
        super().__init__(model, output_dir)
        # {AgentDiagram-uuid → BUML Agent model}, supplied by the
        # project-level router handler. Empty on the single-diagram path, in
        # which case no build contexts are baked (compose-only behavior).
        self.agent_models_by_id = agent_models_by_id or {}

    def generate(self):
        file_path = self.build_generation_path(file_name="docker-compose.yml")
        templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # entry/human-facing — the set whose service publishes host ports + runs the UI;
        # a2a_servers — the set whose service runs the A2A server (≥1 inbound peer). The two
        # are decoupled: a hybrid (a human-facing merge owner) is in BOTH.
        human_facing_services, a2a_server_services = self._compute_service_flags()
        services, networks = self._build_view(self.model, human_facing_services)
        template = env.get_template("docker-compose.yml.j2")
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(template.render(services=services, networks=networks))
        print("Code generated in the location: " + file_path)
        # Bake a BAF build context per resolvable agentic LOCAL artifact.
        self._bake_agent_contexts(env, human_facing_services, a2a_server_services)

    def _bake_agent_contexts(self, env: Environment, entry_services: set = None,
                             a2a_server_services: set = None) -> None:
        """For each LOCAL Artifact carrying a resolvable ``agent_model_ref``,
        bake a build context (``<output_dir>/<svc_name>/``) containing the BAF
        ``agent.py`` + ``config.yaml`` (via BAFGenerator) and a ``Dockerfile``.

        The directory name is ``_safe_service_name(art.name)`` — identical to the
        ``build: ./<svc_name>`` the compose emits for this artifact, so the two
        line up. No-ops when ``agent_models_by_id`` is empty (single-diagram
        path). LOCAL artifacts with no/unresolvable ref are skipped (logged).
        """
        if not self.agent_models_by_id:
            return
        base_dir = os.path.dirname(
            self.build_generation_path(file_name="docker-compose.yml")
        )
        dockerfile_tpl = env.get_template("Dockerfile.j2")

        # v2 A2A — the set of swarm service names, so peer boundary states can be
        # resolved to real services (and non-service handoffs like `from_<Human>`
        # ignored). Mirrors the LOCAL+resolvable filter used in the bake loop.
        service_names = {
            _safe_service_name(a.name)
            for a in self.model.all_artifacts()
            if a.locality == Locality.LOCAL
            and getattr(a, "agent_model_ref", None)
            and self.agent_models_by_id.get(a.agent_model_ref) is not None
        }
        # The A2A agent template lives next to the generic BAF template
        # (agents/templates), not in this generator's templates dir, so it needs
        # its own loader rather than the docker_compose `env` above.
        agent_tpl_dir = os.path.join(
            os.path.dirname(inspect.getfile(BAFGenerator)), "templates"
        )
        a2a_env = Environment(
            loader=FileSystemLoader(agent_tpl_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        a2a_tpl = a2a_env.get_template("baf_a2a_agent_template.py.j2")

        for art in self.model.all_artifacts():
            if art.locality != Locality.LOCAL:
                continue
            ref = getattr(art, "agent_model_ref", None)
            if not ref:
                continue
            agent = self.agent_models_by_id.get(ref)
            if agent is None:
                print(f"[docker_compose] artifact '{art.name}' references agent "
                      f"'{ref}' but no matching AgentDiagram was found — "
                      f"skipping its build context.")
                continue
            svc_name = _safe_service_name(art.name)
            ctx_dir = os.path.join(base_dir, svc_name)
            os.makedirs(ctx_dir, exist_ok=True)
            # BAF agent.py + config.yaml into the build context.
            BAFGenerator(agent, output_dir=ctx_dir).generate()

            # v2 A2A — if this agent has boundary states (it participates in the
            # swarm topology), OVERWRITE the generic agent.py with the A2A render.
            # pass THIS service's name so a `from_<self>` boundary can't
            # demote the entry to a worker (the agent_id and the service line up via
            # _safe_service_name, but pass it explicitly to be robust).
            # precedence: explicit WME a2a: tags (agent._a2a) ▸ the legacy
            # to_/from_ state-name convention ▸ self-contained. Absent tags ⇒ exactly the
            # current path (back-compat guarantee).
            if getattr(agent, '_a2a', None):
                descriptor = _a2a_descriptor_from_tags(agent, service_names, self_service=svc_name)
            else:
                descriptor = _a2a_descriptor(agent, service_names, self_service=svc_name)
            # A service that runs NO A2A server has nothing listening for /a2a — any peer
            # edge pointing at it always "Connection refused"s. Drop those services from
            # every agent's peers. A PURE entry (human-facing, no inbound peers) is such a
            # service; but a HYBRID (human-facing AND a2a_server — e.g. a merge owner that
            # also owns the BPMN start event) DOES listen, so it must NOT be dropped. Hence
            # the drop is keyed on "no A2A server", not on "is human-facing": in a legacy
            # render (no hybrids) the two sets coincide, so this stays byte-identical.
            no_server = service_names - (a2a_server_services or set())
            if no_server:
                descriptor['peers'] = [p for p in descriptor.get('peers', [])
                                       if p.get('service') not in no_server]
                descriptor['to_peers'] = [s for s in descriptor.get('to_peers', [])
                                          if s not in no_server]
            has_boundaries = bool(descriptor['to_peers']) or descriptor['a2a_server']
            if has_boundaries:
                with open(os.path.join(ctx_dir, f"{agent.name}.py"),
                          mode="w", encoding="utf-8") as f:
                    f.write(a2a_tpl.render(agent=agent, a2a=descriptor))
                print(f"[docker_compose] A2A-wired ({descriptor['role']}): {svc_name} "
                      f"-> to_peers={descriptor['to_peers']}")

            # Dockerfile referencing the agent script (name unchanged).
            agent_script = f"{agent.name}.py"
            with open(os.path.join(ctx_dir, "Dockerfile"),
                      mode="w", encoding="utf-8") as f:
                f.write(dockerfile_tpl.render(agent_script=agent_script))
            print(f"[docker_compose] baked build context: {ctx_dir}")

    def _compute_service_flags(self) -> tuple:
        """Return ``(human_facing_services, a2a_server_services)`` — the two decoupled
        capability sets, keyed by service name.

        - ``human_facing_services`` — agents that run the websocket/Streamlit UI and get
          the published host ports. Authoritative via ``agent._human_facing`` (the BPMN
          start-event derivation); falls back to the legacy no-inbound-peer heuristic.
        - ``a2a_server_services`` — agents that run the headless A2A server (≥1 inbound
          peer). Used to decide which peer edges are reachable (the bake-loop drop).

        Mirrors the LOCAL+resolvable filter in _bake_agent_contexts(). When
        agent_models_by_id is empty (single-diagram path) returns two empty sets.

        Emits a generation-time warning when there ARE resolvable agentic services but NONE
        is human-facing — a closed worker-only swarm has no user-facing trigger: nothing
        publishes a port and nothing initiates (check the BPMN has a start event in an
        agentic lane).
        """
        if not self.agent_models_by_id:
            return set(), set()
        service_names = {
            _safe_service_name(a.name)
            for a in self.model.all_artifacts()
            if a.locality == Locality.LOCAL
            and getattr(a, 'agent_model_ref', None)
            and self.agent_models_by_id.get(a.agent_model_ref) is not None
        }
        human_facing: set = set()
        a2a_servers: set = set()
        any_service = False
        for art in self.model.all_artifacts():
            if art.locality != Locality.LOCAL:
                continue
            ref = getattr(art, 'agent_model_ref', None)
            if not ref:
                continue
            agent = self.agent_models_by_id.get(ref)
            if agent is None:
                continue
            svc = _safe_service_name(art.name)
            any_service = True
            # same tag ▸ convention precedence as the bake loop, so the
            # split (and the published ports) agree with what's baked.
            if getattr(agent, '_a2a', None):
                descriptor = _a2a_descriptor_from_tags(agent, service_names, self_service=svc)
            else:
                descriptor = _a2a_descriptor(agent, service_names, self_service=svc)
            if descriptor['human_facing']:
                human_facing.add(svc)
            if descriptor['a2a_server']:
                a2a_servers.add(svc)
        if any_service and not human_facing:
            logger.warning(
                "[a2a] no entry/human-facing agent derived — the swarm has no user-facing "
                "trigger; check the BPMN has a start event in an agentic lane")
        return human_facing, a2a_servers

    def _compute_entry_services(self) -> set:
        """The set of human-facing service names (they publish the host ports). Thin
        back-compat wrapper over ``_compute_service_flags``."""
        return self._compute_service_flags()[0]

    def _build_view(self, model: DeploymentModel,
                    entry_services: set = None) -> tuple:
        """Resolve the metamodel into ordered dicts the template renders.

        Doing the graph walk here keeps the template declarative and lets tests
        assert on the view dicts directly.  Returns ``(services, networks)`` —
        plain-dict lists, deterministic order via ``sort_by_timestamp``.
        """
        all_artifacts = list(sort_by_timestamp(model.all_artifacts()))
        all_nodes = list(sort_by_timestamp(model.all_nodes()))
        all_rels = list(sort_by_timestamp(model.relationships))

        node_to_net = {id(n): _safe_service_name(n.name) for n in all_nodes}

        # ----- Pass 1: artifact → set of node python-ids it's deployed on ---
        # Sources: explicit DeploymentRelations AND containment (parent).
        art_nodes_map: dict = {id(a): set() for a in all_artifacts}

        for rel in all_rels:
            if isinstance(rel, DeploymentRelation):
                src_key = id(rel.source)
                if src_key in art_nodes_map:
                    art_nodes_map[src_key].add(id(rel.target))

        for art in all_artifacts:
            if art.parent is not None:
                art_nodes_map[id(art)].add(id(art.parent))

        # ----- Pass 2: networks per artifact (from node membership) ----------
        art_nets: dict = {id(a): [] for a in all_artifacts}

        def _add_net(art_key: int, net: str) -> None:
            if net not in art_nets[art_key]:
                art_nets[art_key].append(net)

        for art in all_artifacts:
            for node_id in art_nodes_map[id(art)]:
                net = node_to_net.get(node_id)
                if net:
                    _add_net(id(art), net)

        # ----- Pass 3: replicas from DeploymentRelation.multiplicity ---------
        art_replicas: dict = {}
        for rel in all_rels:
            if isinstance(rel, DeploymentRelation):
                src_key = id(rel.source)
                if src_key not in art_replicas:
                    mult = rel.multiplicity
                    if mult.max > 1:
                        art_replicas[src_key] = mult.max

        # ----- Pass 4a: DeploymentDependency → depends_on -------------------
        # source depends_on target (source must start after target is running).
        art_depends: dict = {id(a): [] for a in all_artifacts}
        for rel in all_rels:
            if isinstance(rel, DeploymentDependency):
                if isinstance(rel.source, Artifact) and isinstance(rel.target, Artifact):
                    # Synthetic artifacts (component projections, art.manifests != [])
                    # are not rendered as services, so a dependency on one would point at a
                    # non-existent service. They also reuse the physical artifact's service
                    # name, which is how an Artifact→Component dependency degenerates into a
                    # self-reference (e.g. agent_coder -> agent_coder, a compose cycle). Skip
                    # manifest targets, and guard self-deps as a belt-and-suspenders.
                    if rel.target.manifests:
                        continue
                    svc_target = _safe_service_name(rel.target.name)
                    if svc_target == _safe_service_name(rel.source.name):
                        continue
                    deps = art_depends[id(rel.source)]
                    if svc_target not in deps:
                        deps.append(svc_target)

        # ----- Pass 5: CommunicationPath → bridging networks ----------------
        cp_nets: list = []
        seen_cp: set = set()
        for rel in all_rels:
            if isinstance(rel, CommunicationPath):
                src_id = id(rel.source)
                tgt_id = id(rel.target)
                src_net = node_to_net.get(src_id, _safe_service_name(rel.source.name))
                tgt_net = node_to_net.get(tgt_id, _safe_service_name(rel.target.name))
                link = f"{src_net}_{tgt_net}_link"
                if link not in seen_cp:
                    seen_cp.add(link)
                    cp_nets.append(link)
                for art in all_artifacts:
                    if (src_id in art_nodes_map[id(art)]
                            or tgt_id in art_nodes_map[id(art)]):
                        _add_net(id(art), link)

        # ----- Build service dicts -------------------------------------------
        # Capability/resource stereotype tokens — these artifacts represent
        # externally hosted services (LLM API, vector DB, RAG store), not
        # deployable containers.
        _CAP_TOKENS = frozenset(['llm', 'db', 'rag', 'tool', 'skill'])
        _entry = entry_services or set()

        services = []
        for art in all_artifacts:
            # Synthetic artifacts (created from WME DeploymentComponent)
            # have art.manifests != [] — they are logical component projections
            # that duplicate the physical artifact's service name.
            if art.manifests:
                continue

            # Capability/resource artifacts are hosted externally, not built
            # as Docker images.
            if any(t in _CAP_TOKENS for t in art.stereotypes):
                continue

            svc_name = _safe_service_name(art.name)
            art_key = id(art)

            if art.locality == Locality.LOCAL:
                build = f"./{svc_name}"
                image = None
                is_hybrid = False
            elif art.locality == Locality.EXTERNAL:
                build = None
                image = f"{svc_name}:latest"
                is_hybrid = False
            else:  # HYBRID
                build = None
                image = f"{svc_name}:latest"
                is_hybrid = True

            services.append({
                'name': svc_name,
                'build': build,
                'image': image,
                'is_hybrid': is_hybrid,
                'networks': art_nets.get(art_key, []),
                'replicas': art_replicas.get(art_key),
                'depends_on': art_depends.get(art_key, []),
                'ports': ['5001:5000', '8765:8765'] if svc_name in _entry else [],
                'stereotypes': ', '.join(art.stereotypes) if art.stereotypes else None,
                'manifests': ', '.join(art.manifests) if art.manifests else None,
            })

        # ----- Build network dicts ------------------------------------------
        seen_nets: set = set()
        networks = []
        for node in all_nodes:
            net_name = node_to_net[id(node)]
            if net_name not in seen_nets:
                seen_nets.add(net_name)
                kind_comment = f"  # kind: {node.kind.value}" if node.kind else ""
                networks.append({'name': net_name, 'kind_comment': kind_comment})
        for link in cp_nets:
            if link not in seen_nets:
                seen_nets.add(link)
                networks.append({'name': link, 'kind_comment': ""})

        return services, networks
