"""A created-and-read-back record is evidence even if its scenario failed.

confirmed_create_paths gated on the scenario's overall verdict. _run_requests
marks a report "failed" when ANY request had an assertion failure, so one
unrelated failure discarded proof the function had already checked itself --
the loop independently requires the POST to be 2xx with no failures, a later
GET to be 2xx with no failures, and the same id plus one matching field.

The harness manufactures that case: tools.py instructs the model to "Test
happy paths AND invalid input/state transitions", and action_gap_tasks asks
for an action driven to its successful AND refused outcome, in one scenario.
Following our own instruction emptied this set, so _apply_scenario_evidence
could not retire the "create unverified:" finding; it was re-prefixed
"runtime unverified:" and promoted to a BLOCKER. A run that demonstrably
created and read back a Room was told it could not create a Room, and spent
fix attempts on a non-defect.

Boot remains a whole-report precondition: nothing a dead app said is evidence.
"""
import pytest

from besser.spec_driven_agent.validation.api_probe import confirmed_create_paths

_CREATE = {"index": 0, "method": "POST", "path": "/room/", "status": 201,
           "json": {"id": 1, "roomNumber": "101"}, "failures": []}
_READ = {"index": 1, "method": "GET", "path": "/room/", "status": 200,
         "json": [{"id": 1, "roomNumber": "101"}], "failures": []}


def _report(responses, status="failed", boot="ok"):
    return {"status": status, "boot": boot, "responses": responses}


def test_a_passing_scenario_is_still_confirmed():
    assert confirmed_create_paths(
        _report([_CREATE, _READ], status="passed")) == {"/room"}


@pytest.mark.parametrize("peer, why", [
    ({"index": 2, "method": "POST", "path": "/bill/", "status": 500,
      "json": {"detail": "boom"}, "failures": ["server error"]},
     "an unrelated entity's failure"),
    ({"index": 2, "method": "POST", "path": "/room/", "status": 400,
      "json": {"detail": "duplicate"}, "failures": ["expected HTTP 2xx, got 400"]},
     "a deliberate negative probe the model was told to write"),
])
def test_a_failed_peer_request_does_not_discard_the_evidence(peer, why):
    """The regression."""
    assert confirmed_create_paths(_report([_CREATE, _READ, peer])) == {"/room"}, why


def test_a_dead_app_proves_nothing():
    """Boot stays a whole-report gate."""
    assert confirmed_create_paths(
        _report([_CREATE, _READ], status="passed", boot="failed")) == set()


@pytest.mark.parametrize("responses, why", [
    ([{**_CREATE, "status": 400, "failures": ["expected HTTP 2xx, got 400"]}, _READ],
     "the create itself failed"),
    ([_CREATE, {**_READ, "status": 500, "failures": ["assertion"]}],
     "the read-back itself failed"),
    ([_CREATE],
     "nothing read the record back"),
    ([_CREATE, {**_READ, "json": [{"id": 9, "roomNumber": "999"}]}],
     "the read returned a different record"),
    ([{**_CREATE, "truncated": True}, _READ],
     "the create response was truncated, so the shape is unknown"),
    ([_CREATE, {**_READ, "method": "POST"}],
     "the follow-up was not a GET"),
])
def test_it_still_refuses_without_real_evidence(responses, why):
    """The reason this check exists: a 200 or a model's assertion is not proof.
    Loosening the scenario gate must not loosen any of these."""
    assert confirmed_create_paths(_report(responses)) == set(), why


def test_only_the_entity_with_evidence_is_confirmed():
    """One entity's proof must not vouch for another's."""
    bill_create = {"index": 2, "method": "POST", "path": "/bill/", "status": 201,
                   "json": {"id": 7, "billNumber": "B1"}, "failures": []}

    assert confirmed_create_paths(
        _report([_CREATE, _READ, bill_create])) == {"/room"}
