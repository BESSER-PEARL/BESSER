"""Phase 2 tells the model that ending its turn already runs validate_app.

The end-of-turn gate calls ``_validate_app`` itself, yet models spent a turn on one
more validate_app right before ending, with no change since the last one (live runs
r9-web1 turn 6, r9-p81 turn 17).
"""
from tests.spec_driven_agent.prompt_budget import FIXTURES, phase2_request


def test_phase2_prompt_says_ending_the_turn_validates(tmp_path):
    system, _ = phase2_request(FIXTURES["shelf"](), tmp_path)
    assert "Ending your turn runs it by itself: call it after code changes, never just before you finish." in system
