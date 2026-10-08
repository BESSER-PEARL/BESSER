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


# Policies whose rule is "MORE than the ratio" (a strict majority). VotingPolicy's
# rule is "reaches the ratio threshold".
_STRICT_POLICIES = ("MajorityPolicy", "AbsoluteMajorityPolicy")


def tally(policy_type, ratio, candidates, ballots):
    """Count candidate-selection votes deterministically. Returns an audit record.

    candidates: [{"id": "C1", "producer": "agent_x#1"}, ...]
    ballots:    [{"voter","source","vote": "C2"|None,"weight": float}, ...]
                vote == None is an abstain.

    Weighting: VotingPolicy uses each ballot's `weight` (participant confidence);
    Majority/AbsoluteMajority use weight 1 per ballot. The leading candidate's
    share is its support over the denominator: the weight actually cast
    (VotingPolicy), every ballot including abstentions (AbsoluteMajority), or the
    ballots that voted (Majority).

    Decision rule: a candidate wins only if it is the UNIQUE leader and its share
    meets the ratio (default 0.5) -- strictly above it for Majority and
    AbsoluteMajority ("more than half" at the default), at or above it for
    VotingPolicy. A tie for the lead or a share below the ratio decides nothing:
    `outcome` is None and the merge escalates instead of picking a candidate.
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
    if weighted:
        denom = sum(scores.values())
    elif policy_type == "AbsoluteMajorityPolicy":
        denom = float(cast + abstain)
    else:
        denom = float(cast)
    top = max(scores.values()) if scores else 0.0
    leaders = [cid for cid in ids if scores[cid] == top] if cast else []
    share = (top / denom) if denom > 0 else 0.0
    thr = ratio if isinstance(ratio, (int, float)) else 0.5
    met_ratio = share > thr if policy_type in _STRICT_POLICIES else share >= thr
    outcome = leaders[0] if len(leaders) == 1 and met_ratio else None
    return {
        "policy": policy_type,
        "ratio": thr,
        "candidates": [{"id": c["id"], "producer": c.get("producer")} for c in candidates],
        "ballots": [{"voter": b.get("voter"), "source": b.get("source"),
                     "vote": b.get("vote"), "weight": b.get("weight", 1.0)} for b in ballots],
        "scores": scores,
        "cast": cast,
        "abstain": abstain,
        "leaders": leaders,
        "outcome": outcome,
        "share": round(share, 4),
        "met_ratio": met_ratio,
    }


def decision_footer(decision):
    """The audit line appended to a governed merge's reply."""
    votes = ", ".join(f"{k}={v:g}" for k, v in decision["scores"].items())
    if decision["outcome"] is not None:
        result = f"winner {decision['outcome']} (share {decision['share']:g}, met ratio)"
    elif len(decision["leaders"]) > 1:
        result = f"no decision: tie between {', '.join(decision['leaders'])}"
    else:
        result = f"no decision: best share {decision['share']:g} below ratio"
    return ("\n\n— governance —\n"
            f"policy {decision['policy']} (ratio {decision['ratio']}); {result}; "
            f"votes [{votes}]; abstain {decision['abstain']}")


def decision_reply(decision, cand_text):
    """The merge result: the winning candidate verbatim, or -- when the policy
    decided nothing -- an escalation that lists every candidate and selects none."""
    if decision["outcome"] is not None:
        return cand_text[decision["outcome"]] + decision_footer(decision)
    slate = "\n\n".join(f"[{c['id']}] (from {c['producer']})\n{cand_text.get(c['id'], '')}"
                         for c in decision["candidates"])
    return ("The governance policy reached no decision, so no candidate was selected; "
            "a human must decide.\n\nCandidate outputs:\n" + slate + decision_footer(decision))
