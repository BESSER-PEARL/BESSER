from besser.generators.docker_compose.docker_compose_generator import (
    _a2a_descriptor,
    _a2a_descriptor_from_tags,
    _resolve_peer_service,
)


class _S:
    def __init__(self, name):
        self.name = name
        self.body = None


class _A:
    def __init__(self, name, state_names):
        self.name = name
        self.states = [_S(n) for n in state_names]


def test_self_referential_from_does_not_demote_entry():
    # Supervisor with a stray self-referential `from_supervisor` + a real
    # outbound `to_coder`. It must stay an ENTRY, and the self-from must be ignored.
    sup = _A('Supervisor', ['coordinate_work', 'from_supervisor', 'to_coder'])
    services = {'supervisor', 'coder', 'reviewer'}
    d = _a2a_descriptor(sup, services, self_service='supervisor')
    assert d['role'] == 'entry'
    assert d['to_peers'] == ['coder']


def test_real_inbound_peer_makes_worker():
    coder = _A('Coder', ['write_code', 'from_supervisor', 'to_reviewer'])
    services = {'supervisor', 'coder', 'reviewer'}
    d = _a2a_descriptor(coder, services, self_service='coder')
    assert d['role'] == 'worker'
    assert d['to_peers'] == ['reviewer']


# ---------------------------------------------------------------------------
# human_facing / a2a_server flags — decoupled from the legacy single `role`.
# ---------------------------------------------------------------------------

def test_pure_entry_flags():
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    d = _a2a_descriptor(sup, {'supervisor', 'coder'}, self_service='supervisor')
    assert d['human_facing'] is True and d['a2a_server'] is False
    assert d['role'] == 'entry'                           # legacy key still derived


def test_pure_worker_flags():
    coder = _A('Coder', ['write_code', 'from_supervisor'])
    d = _a2a_descriptor(coder, {'supervisor', 'coder'}, self_service='coder')
    assert d['human_facing'] is False and d['a2a_server'] is True
    assert d['role'] == 'worker'


def test_human_facing_flag_is_authoritative_over_inbound():
    # An agent WITH inbound peers (a2a_server) that the backend marked human-facing is a
    # HYBRID: both flags true. The legacy heuristic alone would have demoted it to a worker.
    rev = _A('Reviewer', ['review', 'from_worker'])
    rev._human_facing = True
    d = _a2a_descriptor(rev, {'reviewer', 'worker'}, self_service='reviewer')
    assert d['human_facing'] is True and d['a2a_server'] is True
    assert d['role'] == 'entry'                           # derived from human_facing


def test_human_facing_flags_via_tags_path():
    rev = _tagged('Reviewer',
                  outbound=[{"peer": "Worker", "ref": "w", "order": 1, "kind": "supervises", "state": "s"}],
                  inbound=[{"peer": "Worker", "ref": "w", "order": 9999, "kind": "revises"}])
    rev._human_facing = True
    d = _a2a_descriptor_from_tags(rev, {'reviewer', 'worker'}, self_service='reviewer')
    assert d['human_facing'] is True and d['a2a_server'] is True


def _w3_gov(gateway_id, participants, producers, policy_type="VotingPolicy"):
    return {"policy_type": policy_type, "ratio": 0.5, "requires_human": False,
            "participants": participants, "producers": producers,
            "gateway_id": gateway_id, "merge_state": "Address_merge_decision",
            "instruction": "...", "summary": "...", "raw": "..."}


def test_human_facing_single_voting_merge_owner_prefers_ui_governance():
    gov = _w3_gov(
        "gw1",
        [_p("Reviewer", 0.93), _p("Coder", 0.82)],
        ["Coder"],
        policy_type="MajorityPolicy",
    )
    rev = _tagged(
        "Reviewer",
        inbound=[{
            "peer": "Coder",
            "ref": "u",
            "flow": "gw1",
            "target_state": "Address_merge_decision",
            "order": 9999,
            "kind": "revises",
        }],
    )
    rev._human_facing = True
    rev._governance = [gov]
    rev._governance_by_state = {"Address_merge_decision": gov}

    d = _a2a_descriptor_from_tags(rev, {"reviewer", "coder"}, self_service="reviewer")

    assert d["human_facing"] is True and d["a2a_server"] is True
    assert d["states"] == []
    assert d["governance"]["is_voting"] is True
    assert d["governance"]["producer_services"] == ["coder"]
    assert d["governance"]["owner_votes"] is True


def test_human_facing_multi_merge_owner_keeps_state_dispatch():
    gov_a = _w3_gov("gw1", [_p("Reviewer", 0.93), _p("Coder", 0.82)], ["Coder"])
    gov_b = _w3_gov("gw2", [_p("Reviewer", 0.93), _p("Coder", 0.82)], ["Coder"])
    rev = _tagged("Reviewer", inbound=[
        {"peer": "Coder", "flow": "gw1", "target_state": "merge_a", "order": 9999, "kind": "revises"},
        {"peer": "Coder", "flow": "gw2", "target_state": "merge_b", "order": 9999, "kind": "revises"},
    ])
    rev._human_facing = True
    rev._governance = [gov_a, gov_b]
    rev._governance_by_state = {"merge_a": gov_a, "merge_b": gov_b}

    d = _a2a_descriptor_from_tags(rev, {"reviewer", "coder"}, self_service="reviewer")

    assert {s["merge_key"] for s in d["states"]} == {"gw1", "gw2"}


def test_non_service_from_human_is_ignored():
    # `from_Human` resolves to no service → entry stays entry (pre-existing behavior).
    sup = _A('Supervisor', ['coordinate_work', 'from_human', 'to_coder'])
    services = {'supervisor', 'coder'}
    d = _a2a_descriptor(sup, services, self_service='supervisor')
    assert d['role'] == 'entry'


# ---------------------------------------------------------------------------
# `peers` shim on the legacy descriptor (template single-path)
# ---------------------------------------------------------------------------

def test_legacy_descriptor_emits_plain_channel_peers_shim():
    sup = _A('Supervisor', ['coordinate_work', 'to_coder', 'to_reviewer'])
    services = {'supervisor', 'coder', 'reviewer'}
    d = _a2a_descriptor(sup, services, self_service='supervisor')
    assert d['source'] == 'convention'
    # one peer entry per to_peer, all plain-channel (kind=None) → renders as today
    assert [p['service'] for p in d['peers']] == d['to_peers']
    assert all(p['kind'] is None for p in d['peers'])


# ---------------------------------------------------------------------------
# tag-aware descriptor (_a2a_descriptor_from_tags)
# ---------------------------------------------------------------------------

def _tagged(name, outbound=None, inbound=None):
    a = _A(name, [])
    a._a2a = {"outbound": outbound or [], "inbound": inbound or []}
    return a


def test_tags_entry_with_delegate_peer():
    sup = _tagged('Supervisor', outbound=[
        {"peer": "AgentCoder", "ref": "u", "order": 1, "kind": "delegates", "state": "coordinate"},
    ])
    services = {'agent_supervisor', 'agent_coder'}
    d = _a2a_descriptor_from_tags(sup, services, self_service='agent_supervisor')
    assert d['source'] == 'tags'
    assert d['role'] == 'entry'
    assert d['to_peers'] == ['agent_coder']
    assert d['peers'][0]['kind'] == 'delegates'
    assert d['peers'][0]['service'] == 'agent_coder'


def test_tags_inbound_makes_worker():
    coder = _tagged(
        'Coder',
        outbound=[{"peer": "AgentReviewer", "ref": "u", "order": 1,
                   "kind": "supervises", "state": "write_code"}],
        inbound=[{"peer": "AgentSupervisor", "ref": "u", "order": 9999, "kind": "delegates"}],
    )
    services = {'agent_supervisor', 'agent_coder', 'agent_reviewer'}
    d = _a2a_descriptor_from_tags(coder, services, self_service='agent_coder')
    assert d['role'] == 'worker'
    assert d['to_peers'] == ['agent_reviewer']
    assert d['inbound'] == ['agent_supervisor']


def test_tags_dangling_peer_dropped():
    sup = _tagged('Supervisor', outbound=[
        {"peer": "GhostPeer", "ref": "", "order": 1, "kind": "delegates", "state": "s"},
    ])
    services = {'agent_supervisor'}
    d = _a2a_descriptor_from_tags(sup, services, self_service='agent_supervisor')
    assert d['to_peers'] == []          # GhostPeer not in services → dropped, no raise
    assert d['role'] == 'entry'


def test_tags_ordered_and_deduped():
    sup = _tagged('Supervisor', outbound=[
        {"peer": "AgentB", "ref": "u", "order": 2, "kind": "collaborates", "state": "s"},
        {"peer": "AgentA", "ref": "u", "order": 1, "kind": "delegates", "state": "s"},
        {"peer": "AgentA", "ref": "u", "order": 3, "kind": "delegates", "state": "s"},
    ])
    services = {'agent_supervisor', 'agent_a', 'agent_b'}
    d = _a2a_descriptor_from_tags(sup, services, self_service='agent_supervisor')
    # input order preserved (parser already order-sorted upstream); deduped on service
    assert d['to_peers'] == ['agent_b', 'agent_a']


def test_resolve_peer_service_by_name():
    assert _resolve_peer_service({"peer": "AgentCoder"}, {"agent_coder"}) == "agent_coder"
    assert _resolve_peer_service({"peer": "Ghost"}, {"agent_coder"}) is None


# ---------------------------------------------------------------------------
# governed voting owner fans out to the PRODUCER ∪ VOTER star, with
# producers (BPMN flows into the gateway) and voters (policy participants) decoupled.
# ---------------------------------------------------------------------------

def _voting_gov(participants, producers=None, policy_type="VotingPolicy"):
    return {"policy_type": policy_type, "ratio": 0.5, "requires_human": False,
            "participants": participants, "producers": producers or [],
            "instruction": "...", "summary": "...", "raw": "..."}


def _p(name, confidence=None, kind="agent"):
    return {"name": name, "kind": kind, "confidence": confidence, "roles": []}


def test_producers_and_voters_are_decoupled():
    # Coder produces but does NOT vote; Reviewer votes but does NOT produce; the owner
    # (Supervisor) votes (it is a participant) but does not produce. The star is the
    # union of producers and non-owner voters: {coder} ∪ {reviewer}.
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    sup._governance = [_voting_gov(
        participants=[_p("Supervisor", 0.9), _p("Reviewer", 0.6)],
        producers=["Coder"])]
    d = _a2a_descriptor(sup, {"supervisor", "coder", "reviewer"}, self_service="supervisor")
    gov = d["governance"]
    assert sorted(d["to_peers"]) == ["coder", "reviewer"]      # union, not just topology
    assert gov["producer_services"] == ["coder"]              # round-1 targets
    assert gov["weights"] == {"supervisor": 0.9, "reviewer": 0.6}  # voters only
    assert "coder" not in gov["weights"]                       # a producer is not a voter
    assert gov["owner_produces"] is False                      # Supervisor not a producer
    assert gov["owner_votes"] is True                          # Supervisor is a participant
    assert gov["is_voting"] is True
    assert gov["engine_src"]                                   # baked source present
    assert gov["self_service"] == "supervisor"


def test_owner_produces_when_it_is_a_producer():
    # Owner sits on an incoming branch → it is a producer; it self-produces in-process
    # (no peer) and still votes because it is also a participant.
    sup = _A('Supervisor', [])
    sup._governance = [_voting_gov(
        participants=[_p("Supervisor", 0.9), _p("Coder", 0.8)],
        producers=["Supervisor", "Coder"])]
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    gov = d["governance"]
    assert gov["owner_produces"] is True
    assert gov["owner_votes"] is True
    assert gov["producer_services"] == ["coder"]              # owner produces in-proc, not a peer
    assert sorted(d["to_peers"]) == ["coder"]


def test_owner_does_not_vote_when_absent_from_participants():
    # Owner is the judge but is NOT listed in the policy → it must not cast a ballot.
    sup = _A('Supervisor', [])
    sup._governance = [_voting_gov(
        participants=[_p("Coder", 0.8), _p("Reviewer", 0.6)],
        producers=["Coder", "Reviewer"])]
    d = _a2a_descriptor(sup, {"supervisor", "coder", "reviewer"}, self_service="supervisor")
    gov = d["governance"]
    assert gov["owner_votes"] is False
    assert "supervisor" not in gov["weights"]
    assert sorted(d["to_peers"]) == ["coder", "reviewer"]


def test_unresolved_producer_and_voter_are_recorded():
    # An unresolved producer AND an unresolved voter both surface as visible abstains;
    # Coder resolves so the star still runs.
    sup = _A('Supervisor', [])
    sup._governance = [_voting_gov(
        participants=[_p("Coder", 0.8), _p("GhostVoter", 0.5)],
        producers=["Coder", "GhostProducer"])]
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    gov = d["governance"]
    assert gov["unresolved"] == ["GhostProducer", "GhostVoter"]
    assert gov["producer_services"] == ["coder"]
    assert [p["service"] for p in d["peers"]] == ["coder"]


def test_no_candidates_degrades_to_topology():
    # Voting policy but no producer resolves and the owner is not a producer →
    # _governance_star returns None → the item-35 single-round topology is kept.
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    sup._governance = [_voting_gov(
        participants=[_p("Coder", 0.8)], producers=["Ghost"])]
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    assert d["to_peers"] == ["coder"]                          # topology, not a star
    assert "is_voting" not in d["governance"]                  # raw summary, unaugmented


def test_non_voting_policy_keeps_topology_peers():
    # LeaderDrivenPolicy → _governance_star returns None → topology unchanged.
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    sup._governance = [_voting_gov([_p("Coder", 0.8)], producers=["Coder"],
                                   policy_type="LeaderDrivenPolicy")]
    d = _a2a_descriptor(sup, {"supervisor", "coder", "reviewer"}, self_service="supervisor")
    assert d["to_peers"] == ["coder"]                          # topology, not the star
    assert "weights" not in d["governance"]                    # raw summary, unaugmented
    assert "is_voting" not in d["governance"]


def test_no_governance_descriptor_unchanged():
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    assert d["to_peers"] == ["coder"]
    assert d["governance"] is None


def test_voting_star_defaults_missing_confidence_to_unit_weights():
    sup = _A("Supervisor", [])
    sup._governance = [_voting_gov(
        participants=[_p("Coder"), _p("Supervisor")],
        producers=["Coder", "Supervisor"],
        policy_type="MajorityPolicy",
    )]

    descriptor = _a2a_descriptor(
        sup,
        {"supervisor", "coder"},
        self_service="supervisor",
    )
    governance = descriptor["governance"]

    assert governance["is_voting"] is True
    assert governance["weights"] == {"coder": 1.0, "supervisor": 1.0}
    assert governance["owner_votes"] is True
    assert governance["producer_services"] == ["coder"]


def test_governed_voting_star_via_tags_path():
    # the preferred (tags) builder must apply the same star override.
    sup = _tagged('Supervisor', outbound=[
        {"peer": "Coder", "ref": "u", "order": 1, "kind": "delegates", "state": "s"},
    ])
    sup._governance = [_voting_gov([_p("Coder", 0.8), _p("Reviewer", 0.6)],
                                   producers=["Coder"])]
    d = _a2a_descriptor_from_tags(sup, {"supervisor", "coder", "reviewer"},
                                  self_service="supervisor")
    assert sorted(d["to_peers"]) == ["coder", "reviewer"]
    assert d["governance"]["is_voting"] is True
    assert d["governance"]["weights"]["reviewer"] == 0.6


# ---------------------------------------------------------------------------
# State-aware descriptor. When per-state binding keys governance per merge
# STATE (agent._governance_by_state), the descriptor grows a `states[]` list, one star
# per merge state (reusing _governance_star). DORMANT: the live template never reads
# `states`, so a legacy agent (no per-state binding) carries `states == []` and renders
# byte-for-byte as today.
# ---------------------------------------------------------------------------

def test_states_empty_without_per_state_binding():
    # A governed single-merge agent with NO _governance_by_state → no `states` richness.
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    sup._governance = [_voting_gov([_p("Coder", 0.8)], producers=["Coder"])]
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    assert d["states"] == []
    # the per-agent star path is untouched (back-compat fallback still drives the render)
    assert d["governance"]["is_voting"] is True


def test_no_governance_has_empty_states():
    sup = _A('Supervisor', ['coordinate_work', 'to_coder'])
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    assert d["states"] == []


def test_single_merge_state_star_matches_per_agent_star():
    # A 1-merge W3-bound agent: its per-STATE star must equal today's per-AGENT star
    # (same summary, same engine) — the 1-merge parity guarantee from the test plan.
    gov = _voting_gov([_p("Supervisor", 0.9), _p("Reviewer", 0.6)], producers=["Coder"])
    sup = _A('Supervisor', [])
    sup._governance = [gov]
    sup._governance_by_state = {"Address merge decision": gov}
    services = {"supervisor", "coder", "reviewer"}
    d = _a2a_descriptor(sup, services, self_service="supervisor")
    assert [s["name"] for s in d["states"]] == ["Address merge decision"]
    st = d["states"][0]
    assert st["is_merge"] is True
    state_gov = st["governance"]
    # identical to the per-agent star the legacy path computes
    assert state_gov["producer_services"] == d["governance"]["producer_services"] == ["coder"]
    assert state_gov["weights"] == d["governance"]["weights"]
    assert state_gov["owner_votes"] == d["governance"]["owner_votes"] is True
    assert [p["service"] for p in st["peers"]] == ["coder", "reviewer"]


def test_multi_merge_states_union_peers_for_dns():
    # Two merge states reached under different guards: the per-agent star (blobs[0]) only
    # reaches merge A's producer, but the descriptor peer set must UNION both states so
    # every per-state peer is DNS-reachable (runtime _fanout(only=…) slices per state).
    gov_a = _voting_gov([_p("Supervisor", 0.9)], producers=["Coder"])
    gov_b = _voting_gov([_p("Supervisor", 0.9)], producers=["Tester"])
    sup = _A('Supervisor', [])
    sup._governance = [gov_a]                                   # legacy star → coder only
    sup._governance_by_state = {"merge A": gov_a, "merge B": gov_b}
    d = _a2a_descriptor(sup, {"supervisor", "coder", "tester"}, self_service="supervisor")
    assert {s["name"] for s in d["states"]} == {"merge A", "merge B"}
    assert set(d["to_peers"]) == {"coder", "tester"}           # union, not just blobs[0]


def test_states_via_tags_path_carries_guards():
    # The tags builder emits states too; guards come from the inbound a2a edges that
    # target the merge state (the flags the faithful render will route on).
    gov = _voting_gov([_p("Coder", 0.8)], producers=["Coder"])
    sup = _tagged('Supervisor', outbound=[
        {"peer": "Coder", "ref": "u", "order": 1, "kind": "delegates", "state": "s"}])
    sup._a2a["inbound"] = [
        {"peer": "Coder", "flow": "gw1", "intent": "ready",
         "target_state": "Address merge decision", "source_state": "review"}]
    sup._governance_by_state = {"Address merge decision": gov}
    d = _a2a_descriptor_from_tags(sup, {"supervisor", "coder"}, self_service="supervisor")
    st = next(s for s in d["states"] if s["name"] == "Address merge decision")
    assert st["guards"] == [{"intent": "ready", "peer": "Coder", "source_state": "review"}]


# ---------------------------------------------------------------------------
# routing keys: each merge state carries its dispatch key (merge_key =
# gateway id) + a Python-literal merge_config; the descriptor bakes the engine once and
# flags producer agents (has_merge_targets); the producer's peer carries target_gateway.
# ---------------------------------------------------------------------------

def _w3_gov(gateway_id, participants, producers, policy_type="MajorityPolicy"):
    return {"policy_type": policy_type, "ratio": 0.5, "requires_human": False,
            "participants": participants, "producers": producers,
            "gateway_id": gateway_id, "instruction": "i", "summary": "s", "raw": "r"}


def test_state_carries_merge_key_and_config_literal():
    gov = _w3_gov("gw1", [_p("Supervisor", 0.9), _p("Coder", 0.8)], ["Coder"])
    sup = _A("Supervisor", [])
    sup._governance = [gov]
    sup._governance_by_state = {"Address merge decision": gov}
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    st = d["states"][0]
    assert st["merge_key"] == "gw1"
    # merge_config_py is a Python literal (bools/None are Python, not JSON)
    cfg = eval(st["merge_config_py"], {})              # noqa: S307 — trusted generator output
    assert cfg["policy_type"] == "MajorityPolicy"
    assert cfg["is_voting"] is True
    assert cfg["producers"] == ["coder"] and cfg["self"] == "supervisor"
    assert cfg["owner_votes"] is True
    # the engine is baked once on the descriptor for the faithful worker
    assert d["engine_src"] and "def tally" in d["engine_src"]


def test_non_voting_merge_config_is_minimal():
    gov = _w3_gov("gw2", [_p("Supervisor", 0.9)], ["Coder"], policy_type="LeaderDrivenPolicy")
    sup = _A("Supervisor", [])
    sup._governance = [gov]
    sup._governance_by_state = {"merge": gov}
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    cfg = eval(d["states"][0]["merge_config_py"], {})  # noqa: S307
    assert cfg["is_voting"] is False
    assert cfg["weights"] == {} and cfg["producers"] == []


def test_producer_peer_carries_target_gateway_and_flag():
    sup = _tagged("Producer", outbound=[
        {"peer": "Owner", "ref": "u", "order": 1, "kind": "revises", "state": "draft",
         "target_gateway": "gw1"}])
    d = _a2a_descriptor_from_tags(sup, {"producer", "owner"}, self_service="producer")
    assert d["peers"][0]["target_gateway"] == "gw1"
    assert d["has_merge_targets"] is True


def test_legacy_agent_has_no_merge_targets_and_no_engine():
    sup = _A("Supervisor", ["coordinate", "to_coder"])
    d = _a2a_descriptor(sup, {"supervisor", "coder"}, self_service="supervisor")
    assert d["has_merge_targets"] is False
    assert "engine_src" not in d                        # no states → no bake
    assert d["states"] == []
    assert d["merge_sends"] == []


# ---------------------------------------------------------------------------
# merge_sends is the ORDERED, NON-deduped pipeline of governed
# merges a producer feeds. Two edges to the SAME owner (two gateways) must stay TWO sends.
# ---------------------------------------------------------------------------

def test_merge_sends_keeps_two_merges_to_same_owner():
    prod = _tagged("Producer", outbound=[
        {"peer": "Owner", "ref": "u", "order": 2, "kind": "revises", "state": "review",
         "target_gateway": "gw2"},
        {"peer": "Owner", "ref": "u", "order": 1, "kind": "revises", "state": "draft",
         "target_gateway": "gw1"}])
    d = _a2a_descriptor_from_tags(prod, {"producer", "owner"}, self_service="producer")
    # two stages, ORDERED by `order` (gw1 before gw2), NOT collapsed to one owner peer
    assert [(s["service"], s["target_gateway"]) for s in d["merge_sends"]] == [
        ("owner", "gw1"), ("owner", "gw2")]
    assert d["has_merge_targets"] is True
    # the broadcast peer set still dedups to one owner (only merge_sends keeps both)
    assert d["to_peers"] == ["owner"]


def test_non_merge_outbound_is_not_a_send():
    prod = _tagged("Producer", outbound=[
        {"peer": "Owner", "ref": "u", "order": 1, "kind": "delegates", "state": "s"}])
    d = _a2a_descriptor_from_tags(prod, {"producer", "owner"}, self_service="producer")
    assert d["merge_sends"] == []                        # no target_gateway → not a pipeline step
