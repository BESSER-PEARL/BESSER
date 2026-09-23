"""Deterministic governance vote tally for the BAF A2A merge agent.

Self-contained (stdlib only) so the docker_compose generator can bake this module's
SOURCE verbatim into the generated agent.py — the count then runs IN the container
with no besser/ANTLR import. The same functions are unit-tested in BESSER
(tests/generators/agents/test_governance_engine.py), so the baked copy is the tested
copy. The helpers implement candidate-selection voting semantics used by governed
merge agents.
"""
import re

_BALLOT_RE = re.compile(r"BALLOT:\s*(C\d+)", re.IGNORECASE)


def parse_ballot(text, candidate_ids):
    """Return the candidate id a reply voted for, or None (abstain / unparseable).

    Looks for the last `BALLOT: C<k>` line (last wins, so a model that restates the
    instruction then answers is read correctly). Only ids in `candidate_ids` count.
    """
    if not text:
        return None
    hits = [m.group(1).upper() for m in _BALLOT_RE.finditer(text)]
    valid = {c.upper(): c for c in candidate_ids}
    for h in reversed(hits):
        if h in valid:
            return valid[h]
    return None


def tally(policy_type, ratio, candidates, ballots):
    """Count candidate-selection votes deterministically. Returns an audit record.

    candidates: [{"id": "C1", "producer": "agent_x#1"}, ...]
    ballots:    [{"voter","source","vote": "C2"|None,"weight": float}, ...]
                vote == None is an abstain.
    Weighting: VotingPolicy uses each ballot's `weight` (participant confidence);
    Majority/AbsoluteMajority use weight 1 per ballot. Winner = the candidate with the
    greatest (weighted) support; `met_ratio` records whether its share cleared `ratio`.
    AbsoluteMajority ratios over ALL ballots (abstentions in the denominator); the
    others over the ballots that actually voted.
    """
    ids = [c["id"] for c in candidates]
    weighted = policy_type == "VotingPolicy"
    scores = {cid: 0.0 for cid in ids}
    cast, abstain = 0, 0
    for b in ballots:
        v = b.get("vote")
        if v in scores:
            scores[v] += float(b.get("weight", 1.0)) if weighted else 1.0
            cast += 1
        else:
            abstain += 1
    # Denominator for the winning candidate's share (same units as `scores`):
    #   VotingPolicy            → total weight actually cast (confidence-weighted)
    #   AbsoluteMajorityPolicy  → every ballot, abstentions included (the "all eligible" rule)
    #   MajorityPolicy / other  → the ballots that actually voted
    if weighted:
        denom = sum(scores.values())
    elif policy_type == "AbsoluteMajorityPolicy":
        denom = float(cast + abstain)
    else:
        denom = float(cast)
    outcome, top = None, -1.0
    for cid in ids:                       # deterministic: first id wins ties by order
        if scores[cid] > top:
            top, outcome = scores[cid], cid
    share = (top / denom) if denom > 0 else 0.0
    thr = ratio if isinstance(ratio, (int, float)) else 0.5
    return {
        "policy": policy_type,
        "ratio": thr,
        "candidates": [{"id": c["id"], "producer": c.get("producer")} for c in candidates],
        "ballots": [{"voter": b.get("voter"), "source": b.get("source"),
                     "vote": b.get("vote"), "weight": b.get("weight", 1.0)} for b in ballots],
        "scores": scores,
        "cast": cast,
        "abstain": abstain,
        "outcome": outcome,
        "share": round(share, 4),
        "met_ratio": share >= thr,
    }
