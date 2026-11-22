"""A resumed run keeps the list a restored test_api correction was sent as.

Review 4: ``submitted_requests`` was not in the checkpoint snapshot, so after a resume
resending the model's own list was a "change" that needed a reason, and then dropped
the restored check again.
"""

import json

from besser.spec_driven_agent.state.checkpoint import api_scenario_snapshot, restore_api_scenarios

OWN = [{"method": "POST", "path": "/books", "json": {"title": "Dune"}, "expected_status": 201}]
RESTORED = [*OWN, {"method": "GET", "path": "/books/7", "expected_status": 404}]


def _record(**extra):
    return {"scenario_id": "flow", "scenario": {"requests": RESTORED, "backend": None},
            "revision": "r0", "report": {"status": "passed"}, "correction_history": [], **extra}


def test_the_submitted_list_survives_a_resume():
    snapshot = json.loads(json.dumps(api_scenario_snapshot([_record(submitted_requests=OWN)])))
    restored = restore_api_scenarios(snapshot)["named:flow"]
    assert restored["submitted_requests"] == OWN and restored["scenario"]["requests"] == RESTORED


def test_a_checkpoint_without_it_still_loads():
    restored = restore_api_scenarios(json.loads(json.dumps(api_scenario_snapshot([_record()]))))["named:flow"]
    assert "submitted_requests" not in restored


def test_an_invalid_list_is_dropped_and_the_scenario_kept():
    """Review 5: an empty or malformed list raised on every checkpoint and on the recipe."""
    for bad in ([], "POST /books", [{"method": "FETCH", "path": "/books"}]):
        saved = api_scenario_snapshot([_record(submitted_requests=bad)])
        assert saved[0]["scenario"]["requests"] == RESTORED and "submitted_requests" not in saved[0]
        restored = restore_api_scenarios([{**saved[0], "submitted_requests": bad}])["named:flow"]
        assert restored["scenario"]["requests"] == RESTORED and "submitted_requests" not in restored
