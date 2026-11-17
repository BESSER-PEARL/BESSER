"""validate_app's description does not ask for a call the end_turn gate already makes.

r11-web1, turn 6: validate_app alone, with no code change since the turn-4 call,
then end_turn, whose gate runs the same checks again (a whole Sonnet turn for an
identical result). The description said "Call after a coherent change and before
claiming completion", against the Phase 2 note "never just before you finish".
"""
from besser.spec_driven_agent.agent.tools import get_all_tools_including_generators


def _description() -> str:
    return next(tool["description"] for tool in get_all_tools_including_generators()
                if tool["name"] == "validate_app")


def test_it_does_not_ask_for_a_call_before_finishing():
    assert "before claiming completion" not in _description()


def test_it_says_ending_the_turn_runs_it():
    assert "ending your turn runs it" in _description().lower()
