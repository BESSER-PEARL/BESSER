"""A test_api correction that leaves out checks that passed keeps them, instead of being refused.

Live evidence, r12-p61 turns 9-10: the model's corrected scenario left out
``GET /user/`` and ``POST /user/``, which had passed; the correction was refused and
the model spent a turn resending the whole scenario. The passing checks now go back
at the end of the corrected scenario, their references renumbered, and the result
says so. A correction still cannot weaken a request it keeps, and a left-out check
that reads a response it would no longer have is still refused.
"""
import copy
from unittest import mock

from tests.spec_driven_agent.test_test_api_correction_keeps_checks import AUTH, CORRECTION, FAILED, FIRST, _correct, _orch


def test_left_out_checks_return_at_the_end_with_their_references_renumbered(tmp_path):
    orch = _orch(tmp_path, FIRST, FAILED)
    result, probe = _correct(orch, CORRECTION)
    requests = probe.call_args.args[1]
    assert requests[:len(CORRECTION)] == CORRECTION
    appended = requests[len(CORRECTION):]
    assert [r["path"] for r in appended] == ["/treatmentplant/", "/auth/login", "/auth/login", "/auth/me",
                                             "/dashboard/plants/geojson/", "/dashboard/geojson/", "/auth/logout",
                                             "/auth/me"]
    login = len(CORRECTION) + 2  # old request 6, the successful login
    assert appended[3]["headers"] == {"Authorization": f"Bearer {{{{{login}.access_token}}}}"}
    assert AUTH not in [r.get("headers") for r in appended]
    assert "added back at the end" in result["restored_checks"]


def test_a_left_out_check_reading_a_kept_request_is_still_refused(tmp_path):
    old = [{"method": "POST", "path": "/plant/", "json": {"name": "a"}},
           {"method": "GET", "path": "/plant/{{0.id}}/"}]
    orch = _orch(tmp_path, old, {})
    result, probe = _correct(orch, [{**old[0], "expected_status": 201}])
    assert not probe.called and "GET /plant/{{0.id}}/ -> 2xx" in result["error"]


def test_more_than_twenty_requests_is_still_refused(tmp_path):
    old = [{"method": "GET", "path": f"/item{i}/"} for i in range(12)]
    orch = _orch(tmp_path, old, {})
    new = [{"method": "GET", "path": f"/other{i}/"} for i in range(12)]
    result, probe = _correct(orch, copy.deepcopy(new))
    assert not probe.called and "Correction refused" in result["error"]


# Review of the restore (r12-p61 follow-up): a check re-added at the end runs after the
# corrected requests changed the state, so it can fail on a correct app.
OLD = [{"method": "GET", "path": "/stats", "expected_fields": {"count": 0}},
       {"method": "POST", "path": "/books", "json": {"title": "Dune"}, "expected_status": 201},
       {"method": "PUT", "path": "/books/1", "json": {"title": 5}, "expected_status": 200}]
FIX = [{"method": "POST", "path": "/books", "json": {"title": "Dune"}, "expected_status": 201},
       {"method": "PUT", "path": "/books/1", "json": {"title": "Dune II"}, "expected_status": 200}]


def _send(orch, requests, report, reason="the PUT body must send a string title"):
    args = {"scenario_id": "flow", "requests": copy.deepcopy(requests)}
    if reason:
        args["correction_reason"] = reason
    with mock.patch("besser.spec_driven_agent.validation.api_probe.probe_api_scenario",
                    return_value=report) as probe:
        return orch._test_api(args), probe


def _passed(n):
    return {"status": "passed", "boot": "ok", "assertion_failures": [],
            "responses": [{"index": i, "status": 200, "failures": []} for i in range(n)]}


def test_a_restored_check_that_fails_at_the_end_keeps_the_saved_scenario(tmp_path):
    orch = _orch(tmp_path, OLD, {2: {"status": 422, "failures": ["expected HTTP 200, got 422"]}})
    failing = {"status": "failed", "boot": "ok",
               "responses": [{"index": i, "status": 200, "failures": []} for i in range(3)],
               "assertion_failures": [{"index": 2, "message": "count: expected 0, got 1"}]}
    result, _ = _send(orch, FIX, failing)
    assert "Correction refused" in result["error"] and "not saved" in result["error"]
    assert result["report"] is failing
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == OLD


def test_nothing_is_said_to_have_run_when_the_probe_did_not_start(tmp_path):
    orch = _orch(tmp_path, OLD, {2: {"status": 422, "failures": ["expected HTTP 200, got 422"]}})
    result, _ = _send(orch, FIX, {"status": "error", "boot": "not_started", "error": "request 0: bad method"})
    assert "restored_checks" not in result
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == OLD


def test_resending_its_own_list_after_a_restore_reruns_the_restored_scenario(tmp_path):
    orch = _orch(tmp_path, OLD, {2: {"status": 422, "failures": ["expected HTTP 200, got 422"]}})
    result, _ = _send(orch, FIX, _passed(3))
    assert "by scenario_id alone" in result["restored_checks"]
    combined = orch._api_scenarios["named:flow"]["scenario"]["requests"]
    assert combined == [*FIX, OLD[0]]

    again, probe = _send(orch, FIX, _passed(3), reason=None)
    assert "error" not in again and probe.call_args.args[1] == combined
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == combined


def test_a_correction_with_restored_checks_is_not_saved_when_the_app_did_not_boot(tmp_path):
    """Review 4: a boot error ran nothing, yet the correction was saved with the check
    appended and restored_checks said it passed."""
    orch = _orch(tmp_path, OLD, {2: {"status": 422, "failures": ["expected HTTP 200, got 422"]}})
    for boot in ("boot_error", "timeout", "missing_dependency"):
        result, _ = _send(orch, FIX, {"status": "error", "boot": boot, "error": "ImportError: Book"})
        assert "did not boot" in result["error"] and "restored_checks" not in result
        assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == OLD


def test_the_models_own_invalid_request_is_answered_with_its_own_error(tmp_path):
    """Review 5: a probe that never started (the model's request is invalid) read as
    "The app did not boot"; the probe's own error is the answer, nothing saved."""
    orch = _orch(tmp_path, OLD, {2: {"status": 422, "failures": ["expected HTTP 200, got 422"]}})
    invalid = {"status": "error", "boot": "not_started", "error": "ValueError: request 0: expected_status"}
    result, _ = _send(orch, FIX, invalid)
    assert result["error"] == invalid["error"] and "restored_checks" not in result
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == OLD


def test_an_empty_request_list_is_refused_before_anything_runs(tmp_path):
    """Review 5: ``requests: []`` with a reason was saved, and every later checkpoint and
    the recipe then raised on the empty list."""
    orch = _orch(tmp_path, OLD, {})
    result, probe = _send(orch, [], _passed(1))
    assert not probe.called and "at least one request" in result["error"]
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == OLD
