from besser.generators.agents.governance_engine import (
    decision_footer,
    decision_reply,
    parse_ballot,
    tally,
)

C = [{"id": "C1", "producer": "a"}, {"id": "C2", "producer": "b"}]


def _b(vote, weight=1.0, source="agent", voter="x"):
    return {"voter": voter, "source": source, "vote": vote, "weight": weight}


def test_parse_ballot_last_valid_wins():
    assert parse_ballot("...\nBALLOT: C1\nactually BALLOT: C2", ["C1", "C2"]) == "C2"
    assert parse_ballot("no ballot here", ["C1"]) is None
    assert parse_ballot("BALLOT: C9", ["C1", "C2"]) is None  # out-of-range → abstain


def test_majority_pass_above_ratio():
    # 2 of 3 votes for C2 → share 0.667 > 0.5
    d = tally("MajorityPolicy", 0.5, C, [_b("C1"), _b("C2"), _b("C2")])
    assert d["outcome"] == "C2" and d["met_ratio"] is True


def test_one_one_tie_is_not_a_majority():
    # 1 vs 1 + abstain: a tie for the lead decides nothing, and half is not "more than half".
    d = tally("MajorityPolicy", 0.5, C, [_b("C1"), _b("C2"), _b(None)])
    assert d["abstain"] == 1 and d["cast"] == 2
    assert d["leaders"] == ["C1", "C2"] and d["share"] == 0.5
    assert d["met_ratio"] is False and d["outcome"] is None


def test_weighted_tie_decides_nothing_even_at_the_ratio():
    d = tally("VotingPolicy", 0.5, C, [_b("C1", 0.4), _b("C2", 0.4)])
    assert d["met_ratio"] is True and d["leaders"] == ["C1", "C2"] and d["outcome"] is None


def test_majority_below_ratio_decides_nothing():
    d = tally("MajorityPolicy", 0.5, C,
              [_b("C1"), _b("C1"), _b("C2"), _b("C2"), _b("C2")])
    assert d["outcome"] == "C2" and d["share"] == 0.6 and d["met_ratio"] is True
    # Same votes, higher bar: C2 still leads but below the ratio → no decision.
    d2 = tally("MajorityPolicy", 0.75, C,
               [_b("C1"), _b("C1"), _b("C2"), _b("C2"), _b("C2")])
    assert d2["leaders"] == ["C2"] and d2["share"] == 0.6
    assert d2["met_ratio"] is False and d2["outcome"] is None


def test_voting_below_ratio_decides_nothing():
    # Weighted: C1 0.3 vs C2 0.2 → C1 leads with share 0.3/0.5 = 0.6; ratio 0.7 → no decision.
    d = tally("VotingPolicy", 0.7, C, [_b("C1", 0.3), _b("C2", 0.2)])
    assert d["leaders"] == ["C1"] and d["share"] == 0.6
    assert d["met_ratio"] is False and d["outcome"] is None


def test_voting_policy_decides_at_the_ratio():
    # VotingPolicy's rule is "reaches the ratio": share == ratio is enough.
    d = tally("VotingPolicy", 0.6, C, [_b("C1", 0.6), _b("C2", 0.4)])
    assert d["share"] == 0.6 and d["met_ratio"] is True and d["outcome"] == "C1"


def test_absolute_majority_below_ratio_when_abstentions_drag_it_down():
    # 2 for C1 of 5 ballots (3 abstain) → share 2/5 = 0.4: abstentions count against the
    # leader, so even an unopposed C1 is not selected.
    d = tally("AbsoluteMajorityPolicy", 0.5, C,
              [_b("C1"), _b("C1"), _b(None), _b(None), _b(None)])
    assert d["leaders"] == ["C1"] and d["share"] == 0.4
    assert d["met_ratio"] is False and d["outcome"] is None


def test_voting_is_confidence_weighted():
    # C1: 0.9 (owner) ; C2: 0.4 + 0.4 = 0.8 → weighted winner is C1
    d = tally("VotingPolicy", 0.5, C,
              [_b("C1", 0.9, source="owner", voter="sup"), _b("C2", 0.4), _b("C2", 0.4)])
    assert d["outcome"] == "C1"
    assert d["scores"]["C1"] == 0.9 and round(d["scores"]["C2"], 2) == 0.8


def test_absolute_majority_needs_more_than_half_of_all_ballots():
    # 2 for C1, 2 abstain → share 2/4 = 0.5 over ALL ballots: not MORE than half.
    d = tally("AbsoluteMajorityPolicy", 0.5, C, [_b("C1"), _b("C1"), _b(None), _b(None)])
    assert d["share"] == 0.5 and d["met_ratio"] is False and d["outcome"] is None
    d2 = tally("AbsoluteMajorityPolicy", 0.5, C, [_b("C1"), _b("C1"), _b("C1"), _b(None)])
    assert d2["share"] == 0.75 and d2["outcome"] == "C1"


def test_no_ballot_cast_decides_nothing():
    d = tally("MajorityPolicy", 0.5, C, [_b(None), _b(None)])
    assert d["leaders"] == [] and d["outcome"] is None


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


def test_decision_reply_presents_the_winner_verbatim():
    d = tally("MajorityPolicy", 0.5, C, [_b("C2"), _b("C2"), _b("C1")])
    reply = decision_reply(d, {"C1": "first", "C2": "second"})
    assert reply.startswith("second") and "winner C2" in reply


def test_decision_reply_escalates_without_a_decision():
    d = tally("MajorityPolicy", 0.5, C, [_b("C1"), _b("C2")])
    reply = decision_reply(d, {"C1": "first", "C2": "second"})
    assert "no candidate was selected" in reply
    assert "[C1] (from a)\nfirst" in reply and "[C2] (from b)\nsecond" in reply
    assert "no decision: tie between C1, C2" in decision_footer(d)
