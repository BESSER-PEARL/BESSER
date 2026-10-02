from besser.generators.agents.governance_engine import parse_ballot, tally

C = [{"id": "C1", "producer": "a"}, {"id": "C2", "producer": "b"}]


def _b(vote, weight=1.0, source="agent", voter="x"):
    return {"voter": voter, "source": source, "vote": vote, "weight": weight}


def test_parse_ballot_last_valid_wins():
    assert parse_ballot("...\nBALLOT: C1\nactually BALLOT: C2", ["C1", "C2"]) == "C2"
    assert parse_ballot("no ballot here", ["C1"]) is None
    assert parse_ballot("BALLOT: C9", ["C1", "C2"]) is None  # out-of-range → abstain


def test_majority_pass_at_ratio_boundary():
    # 2 of 3 votes for C2 → share 0.667 ≥ 0.5
    d = tally("MajorityPolicy", 0.5, C, [_b("C1"), _b("C2"), _b("C2")])
    assert d["outcome"] == "C2" and d["met_ratio"] is True


def test_majority_exact_boundary_half():
    # 1 vs 1 + abstain: tie → C1 by order; share 0.5 over cast (2) == ratio → met
    d = tally("MajorityPolicy", 0.5, C, [_b("C1"), _b("C2"), _b(None)])
    assert d["abstain"] == 1 and d["cast"] == 2
    assert d["outcome"] == "C1" and d["share"] == 0.5 and d["met_ratio"] is True


def test_majority_below_ratio_not_met():
    # V3 boundary (below side): 2 for C1 of 5 cast → share 0.4 < 0.5 → winner but NOT met.
    d = tally("MajorityPolicy", 0.5, C,
              [_b("C1"), _b("C1"), _b("C2"), _b("C2"), _b("C2")])
    # C2 actually wins here (3 vs 2); flip to test the loser-side share explicitly:
    assert d["outcome"] == "C2" and d["share"] == 0.6 and d["met_ratio"] is True
    # A genuine below-threshold case needs a higher bar: same votes, ratio 0.75.
    d2 = tally("MajorityPolicy", 0.75, C,
               [_b("C1"), _b("C1"), _b("C2"), _b("C2"), _b("C2")])
    assert d2["outcome"] == "C2" and d2["share"] == 0.6 and d2["met_ratio"] is False


def test_voting_below_ratio_not_met():
    # Weighted: C1 0.3 vs C2 0.2 → C1 wins with share 0.3/0.5 = 0.6; ratio 0.7 → not met.
    d = tally("VotingPolicy", 0.7, C, [_b("C1", 0.3), _b("C2", 0.2)])
    assert d["outcome"] == "C1" and d["share"] == 0.6 and d["met_ratio"] is False


def test_absolute_majority_below_ratio_when_abstentions_drag_it_down():
    # 2 for C1 of 5 ballots (3 abstain) → share 2/5 = 0.4 < 0.5 → NOT met, even though C1
    # is unopposed. This is the AbsoluteMajority rule: abstentions count against the winner.
    d = tally("AbsoluteMajorityPolicy", 0.5, C,
              [_b("C1"), _b("C1"), _b(None), _b(None), _b(None)])
    assert d["outcome"] == "C1" and d["share"] == 0.4 and d["met_ratio"] is False


def test_voting_is_confidence_weighted():
    # C1: 0.9 (owner) ; C2: 0.4 + 0.4 = 0.8 → weighted winner is C1
    d = tally("VotingPolicy", 0.5, C,
              [_b("C1", 0.9, source="owner", voter="sup"), _b("C2", 0.4), _b("C2", 0.4)])
    assert d["outcome"] == "C1"
    assert d["scores"]["C1"] == 0.9 and round(d["scores"]["C2"], 2) == 0.8


def test_absolute_majority_counts_abstain_in_denominator():
    # 2 for C1, 2 abstain → share 2/4 = 0.5 over ALL ballots
    d = tally("AbsoluteMajorityPolicy", 0.5, C, [_b("C1"), _b("C1"), _b(None), _b(None)])
    assert d["share"] == 0.5 and d["met_ratio"] is True


def test_owner_and_human_ballots_count():
    d = tally("MajorityPolicy", 0.5, C,
              [_b("C2", source="agent"), _b("C2", source="owner", voter="sup"),
               _b("C1", source="human", voter="human")])
    assert d["outcome"] == "C2" and d["cast"] == 3
    sources = {b["source"] for b in d["ballots"]}
    assert sources == {"agent", "owner", "human"}


def test_absent_participant_is_visible_not_dropped():
    # an unresolved participant enters as an abstain ballot
    d = tally("MajorityPolicy", 0.5, C, [_b("C1"), _b(None, voter="offline_agent")])
    assert d["abstain"] == 1
    assert any(b["voter"] == "offline_agent" and b["vote"] is None for b in d["ballots"])
