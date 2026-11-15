"""A test_api correction may fix the failing part of a scenario, not drop the checks that passed.

Live run r9-p61: the first ``auth_and_dashboard`` scenario had 17 requests (anonymous
401s, duplicate 409, wrong/right login, /auth/me, logout then 401) and one wrong
reference, ``{{8.id}}`` for a create that answers ``{"treatmentplant": {...}}``. The
"correction" resubmitted 8 requests without any auth check, and that weakened scenario
is what the run retained and replayed as passed. The two request lists below are the
run's (``.besser_trace.jsonl``, turns 6 and 7); the trace redacts the Authorization
values, rebuilt here as references to the login/register token. The trace cuts the
first report, so its failures are reconstructed: the wrong reference (10) and the two
reads that counted detections it never created (11, 13).
"""
import copy
from unittest import mock

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.shelf_app import shelf_model

AUTH = {"Authorization": "Bearer {{6.access_token}}"}
PLANT = {"name": "Beggen", "region": "Centre", "latitude": 49.64, "longitude": 6.14}
DETECTION = {"concentration": 12.5, "method": "RT-qPCR", "sampleDate": "2024-03-04", "week": 10}
FIRST = [
    {"method": "GET", "path": "/dashboard/metrics/", "expected_status": 401},
    {"method": "GET", "path": "/treatmentplant/", "expected_status": 401},
    {"method": "POST", "path": "/auth/register", "expected_status": 201,
     "json": {"username": "alice", "email": "alice@example.com", "password": "secret123"},
     "expected_fields": {"user.username": "alice"}},
    {"method": "POST", "path": "/auth/register", "expected_status": 409,
     "json": {"username": "alice", "email": "b@example.com", "password": "secret123"}},
    {"method": "POST", "path": "/auth/register", "expected_status": 422,
     "json": {"username": "bob", "email": "bob@example.com", "password": "short"}},
    {"method": "POST", "path": "/auth/login", "expected_status": 401,
     "json": {"username": "alice", "password": "wrongpass1"}},
    {"method": "POST", "path": "/auth/login", "expected_status": 200,
     "json": {"username": "alice", "password": "secret123"}},
    {"method": "GET", "path": "/auth/me", "headers": AUTH, "expected_fields": {"username": "alice"}},
    {"method": "POST", "path": "/treatmentplant/", "headers": AUTH, "json": PLANT, "expected_status": [200, 201]},
    {"method": "POST", "path": "/virus/", "headers": AUTH, "expected_status": [200, 201],
     "json": {"name": "SARS-CoV-2", "family": "Coronaviridae"}},
    {"method": "POST", "path": "/detection/", "headers": AUTH, "expected_status": [200, 201],
     "json": {**DETECTION, "plant": "{{8.id}}", "virus": "{{9.id}}"}},
    {"method": "GET", "path": "/dashboard/metrics/", "headers": AUTH,
     "expected_fields": {"plantCount": 1, "detectionCount": 1}},
    {"method": "GET", "path": "/dashboard/plants/geojson/", "headers": AUTH,
     "expected_fields": {"features.0.geometry.type": "Point"}},
    {"method": "GET", "path": "/dashboard/detections/?region=centre", "headers": AUTH,
     "expected_fields": {"detections.0.concentration": 12.5}},
    {"method": "GET", "path": "/dashboard/geojson/", "headers": AUTH, "expected_fields": {"type": "FeatureCollection"}},
    {"method": "POST", "path": "/auth/logout", "headers": AUTH, "expected_status": 200},
    {"method": "GET", "path": "/auth/me", "headers": AUTH, "expected_status": 401},
]
TOKEN = {"Authorization": "Bearer {{0.access_token}}"}
CORRECTION = [
    {"method": "POST", "path": "/auth/register", "expected_status": 201,
     "json": {"username": "alice", "email": "alice@example.com", "password": "secret123"}},
    {"method": "POST", "path": "/treatmentplant/", "headers": TOKEN, "json": PLANT, "expected_status": [200, 201]},
    {"method": "POST", "path": "/virus/", "headers": TOKEN, "expected_status": [200, 201],
     "json": {"name": "SARS-CoV-2", "family": "Coronaviridae"}},
    {"method": "POST", "path": "/detection/", "headers": TOKEN, "expected_status": [200, 201],
     "json": {**DETECTION, "plant": "{{1.treatmentplant.id}}", "virus": "{{2.virus.id}}"}},
    {"method": "GET", "path": "/dashboard/metrics/", "headers": TOKEN,
     "expected_fields": {"plantCount": 1, "detectionCount": 1, "averageConcentration": 12.5}},
    {"method": "GET", "path": "/dashboard/detections/?region=centre", "headers": TOKEN,
     "expected_fields": {"detections.0.concentration": 12.5, "weekly.0.week": 10}},
    {"method": "GET", "path": "/dashboard/detections/?plant_ids={{1.treatmentplant.id}}", "headers": TOKEN,
     "expected_fields": {"detections.0.method": "RT-qPCR"}},
    {"method": "POST", "path": "/dashboard/geojson/", "headers": TOKEN, "expected_status": 422},
]
REASON = "Create responses nest ids under treatmentplant.id / virus.id; my test used the wrong paths."
FAILED = {10: {"failures": ["ValueError: response 8 has no field 'id'"]},
          11: {"status": 200, "failures": ["detectionCount: expected 1, got 0"]},
          13: {"status": 200, "failures": ["detections.0.concentration: index 0 out of range"]}}


def _report(requests, failed):
    responses = []
    for index, request in enumerate(requests):
        expected = request.get("expected_status", 200)
        item = {"index": index, "method": request["method"], "path": request["path"], "failures": [],
                "status": expected[0] if isinstance(expected, list) else expected}
        item.update(failed.get(index, {}))
        if "status" not in failed.get(index, {"status": 0}):
            item.pop("status")
        responses.append(item)
    return {"status": "failed" if failed else "passed", "boot": "ok", "responses": responses,
            "assertion_failures": [{"index": i, "message": m} for i, f in failed.items() for m in f["failures"]]}


def _orch(tmp_path, requests, failed):
    orch = LLMOrchestrator(llm_client=mock.MagicMock(), domain_model=shelf_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_import_smoke_check=True)
    orch._api_scenarios["named:flow"] = {
        "scenario": {"requests": copy.deepcopy(requests), "backend": None}, "scenario_id": "flow",
        "revision": "r0", "report": _report(requests, failed), "correction_history": []}
    return orch


def _correct(orch, requests):
    with mock.patch("besser.spec_driven_agent.validation.api_probe.probe_api_scenario",
                    return_value={"status": "passed", "boot": "ok", "responses": []}) as probe:
        result = orch._test_api({"scenario_id": "flow", "requests": requests, "correction_reason": REASON})
    return result, probe


def test_the_r9_p61_correction_is_refused_naming_the_dropped_auth_checks(tmp_path):
    orch = _orch(tmp_path, FIRST, FAILED)
    result, probe = _correct(orch, CORRECTION)
    assert not probe.called
    error = result["error"]
    # Matched by method and route only: the routes the correction no longer calls.
    for check in ("GET /treatmentplant/ -> 401", "POST /auth/login -> 401", "POST /auth/login -> 200",
                  "GET /auth/me -> 2xx", "GET /dashboard/plants/geojson/ -> 2xx", "GET /dashboard/geojson/ -> 2xx",
                  "POST /auth/logout -> 200", "GET /auth/me -> 401"):
        assert check in error
    assert "more)" not in error and "own scenario_id" in error
    assert len(error) < 600
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == FIRST


def test_fixing_only_the_failing_reference_is_accepted(tmp_path):
    orch = _orch(tmp_path, FIRST, FAILED)
    fixed = copy.deepcopy(FIRST)
    fixed[10]["json"].update(plant="{{8.treatmentplant.id}}", virus="{{9.virus.id}}")
    result, probe = _correct(orch, fixed)
    assert probe.called and "error" not in result
    assert orch._api_scenarios["named:flow"]["scenario"]["requests"] == fixed


def test_failing_requests_and_those_reading_them_may_be_dropped(tmp_path):
    old = [{"method": "POST", "path": "/plant/", "json": {"name": "a"}},
           {"method": "GET", "path": "/plant/{{0.id}}/"},
           {"method": "GET", "path": "/virus/", "expected_status": 200}]
    orch = _orch(tmp_path, old, {0: {"status": 422, "failures": ["expected HTTP 2xx, got 422"]}})
    result, probe = _correct(orch, old[2:])
    assert probe.called and "error" not in result
    orch = _orch(tmp_path, old, {0: {"status": 422, "failures": ["expected HTTP 2xx, got 422"]}})
    result, probe = _correct(orch, old[:2])
    assert not probe.called and "GET /virus/ -> 200" in result["error"]


TODO_CREATE = {"method": "POST", "path": "/todoitem/", "json": {"title": "a"}}


@pytest.mark.parametrize("old, new", [
    pytest.param([TODO_CREATE, {"method": "GET", "path": "/todoitem/1/"}],
                 [TODO_CREATE, {"method": "GET", "path": "/todoitem/{{0.todoitem.id}}/"}],
                 id="hard-coded id replaced by a reference"),
    pytest.param([TODO_CREATE], [{**TODO_CREATE, "expected_status": 201}], id="2xx tightened to 201"),
    pytest.param([{**TODO_CREATE, "expected_status": [200, 201]}], [{**TODO_CREATE, "expected_status": 200}],
                 id="200|201 tightened to 200"),
    pytest.param([{"method": "DELETE", "path": "/todoitem/1/", "expected_status": 200}],
                 [{"method": "DELETE", "path": "/todoitem/1/", "expected_status": 403}],
                 id="status corrected to the specification's 403"),
    pytest.param([{"method": "GET", "path": "/todoitem"}, {"method": "GET", "path": "/x/?region=centre"}],
                 [{"method": "GET", "path": "/todoitem/"}, {"method": "GET", "path": "/x/?region=Centre"}],
                 id="trailing slash and query value fixed"),
])
def test_legitimate_corrections_are_accepted(tmp_path, old, new):
    orch = _orch(tmp_path, old, {})
    result, probe = _correct(orch, new)
    assert probe.called and "error" not in result, result
