"""The BESSER Lab phase 2 contract: every route of the table, its models, its gate, and the request limits.

Until the lab service is rebuilt (W4-A) and wired (W5-A), every route answers 501 behind the lab gate. The
editor's types are generated from this schema, so the names checked here are the editor's too.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabMessageRequest,
    LabNoteInput,
    LabPullRequestRequest,
    LabThreadItem,
    LabThreadMessage,
    LabThreadNotice,
    LabThreadPullRequests,
    LabThreadRun,
)
from besser.utilities.web_modeling_editor.backend.services.lab.service import (
    LabRefusal,
    LabService,
    create_lab_service,
)

SESSION = {"X-GitHub-Session": "sess-octocat"}
CHANGE_ID = "abcdef012345"
NOT_BUILT = "Not built yet: BESSER Lab phase 2 is being built."

# (method, OpenAPI path, query, JSON body, request model, answer: response model or status code)
ROUTES = [
    ("GET", "/besser_api/lab/home", None, None, None, "LabHomeResponse"),
    ("GET", "/besser_api/lab/changes", None, None, None, "LabChangesResponse"),
    ("POST", "/besser_api/lab/change", None, None, None, "LabChangeDetail"),
    ("GET", "/besser_api/lab/change", {"changeId": CHANGE_ID}, None, None, "LabChangeDetail"),
    ("POST", "/besser_api/lab/message", None, {"message": "Add a note element", "changeId": CHANGE_ID},
     "LabMessageRequest", None),
    ("POST", "/besser_api/lab/pull-request", None, {"changeId": CHANGE_ID}, "LabPullRequestRequest",
     "LabPullRequestsResponse"),
    ("POST", "/besser_api/lab/change/drop", None, {"changeId": CHANGE_ID}, "LabChangeRequest",
     "LabChangesResponse"),
    ("POST", "/besser_api/lab/change/close", None, {"changeId": CHANGE_ID}, "LabChangeRequest",
     "LabChangeDetail"),
    ("POST", "/besser_api/lab/change/update-base", None, {"changeId": CHANGE_ID, "repo": "besser"},
     "LabUpdateBranchRequest", "LabUpdateBranchResponse"),
    ("GET", "/besser_api/lab/diff", {"changeId": CHANGE_ID, "repo": "besser", "path": "README.md"}, None, None,
     "LabDiffResponse"),
    ("GET", "/besser_api/lab/review", {"changeId": CHANGE_ID}, None, None, "LabReviewResponse"),
    ("GET", "/besser_api/lab/preview", None, None, None, "LabPreviewView"),
    ("POST", "/besser_api/lab/preview/slot", None, {"changeId": CHANGE_ID}, "LabTakeSlotRequest",
     "LabPreviewSlot"),
    ("POST", "/besser_api/lab/preview/slot/touch", None, {"slot": 0}, "LabTouchSlotRequest", 204),
    ("POST", "/besser_api/lab/preview/restart", None, {"slot": 0, "service": "backend"},
     "LabPreviewRestartRequest", "LabPreviewView"),
    ("GET", "/besser_api/lab/preview/logs/{service}", {"slot": 0, "lines": 50}, None, None, "LabPreviewLog"),
    ("GET", "/besser_api/lab/settings", None, None, None, "LabSettingsResponse"),
    ("POST", "/besser_api/lab/settings/refresh-bases", None, None, None, "LabSettingsResponse"),
    ("GET", "/besser_api/lab/audit", {"month": "2026-11"}, None, None, "LabAuditResponse"),
    ("GET", "/besser_api/lab/memory", None, None, None, "LabMemoryResponse"),
    ("POST", "/besser_api/lab/memory", None, {"kind": "decision", "title": "T", "body": "B"}, "LabNoteInput",
     "LabNote"),
    ("PUT", "/besser_api/lab/memory/{note_id}", None, {"kind": "decision", "title": "T", "body": "B"},
     "LabNoteInput", "LabNote"),
    ("DELETE", "/besser_api/lab/memory/{note_id}", None, None, None, 204),
    ("POST", "/besser_api/lab/memory/proposals/dismiss", None, {"changeId": CHANGE_ID, "id": "p1"},
     "LabProposalRef", 204),
]
ROUTE_IDS = [f"{method} {path}" for method, path, *_ in ROUTES]

PATH_VALUES = {"{service}": "backend", "{note_id}": "m7"}

REMOVED = [
    ("GET", "/besser_api/lab/state"),
    ("POST", "/besser_api/lab/change/park"),
    ("POST", "/besser_api/lab/change/resume"),
    ("POST", "/besser_api/lab/preview/reset"),
]


def _url(path: str) -> str:
    for placeholder, value in PATH_VALUES.items():
        path = path.replace(placeholder, value)
    return path


def _call(client, method, path, query, body, headers=SESSION):
    return client.request(method, _url(path), params=query, json=body, headers=headers)


def _ref(schema: dict) -> str:
    return schema["$ref"].rsplit("/", 1)[-1]


@pytest.fixture(scope="module")
def openapi() -> dict:
    return app.openapi()


@pytest.fixture
def lab_on(route_settings, app_state):
    """The lab on, octocat allowed and signed in, mallory signed in but not allowed."""
    route_settings.lab_enabled = True
    route_settings.lab_allowed_logins = "octocat"
    app_state.github_sessions.user_tokens.set("sess-octocat", {"access_token": "gho_octo", "username": "octocat"})
    app_state.github_sessions.user_tokens.set("sess-mallory", {"access_token": "gho_mal", "username": "mallory"})
    return route_settings


# -- The route table in the OpenAPI schema ---------------------------------------------------------------

def test_the_lab_paths_are_exactly_the_table(openapi):
    lab_operations = {(method.upper(), path) for path, item in openapi["paths"].items()
                      if path.startswith("/besser_api/lab/") for method in item}
    assert lab_operations == {(method, path) for method, path, *_ in ROUTES}


@pytest.mark.parametrize("method, path, query, body, request_model, answer", ROUTES, ids=ROUTE_IDS)
def test_each_route_declares_its_request_and_answer(openapi, method, path, query, body, request_model, answer):
    operation = openapi["paths"][path][method.lower()]
    if request_model is None:
        assert "requestBody" not in operation
    else:
        assert _ref(operation["requestBody"]["content"]["application/json"]["schema"]) == request_model
    if isinstance(answer, int):
        assert str(answer) in operation["responses"]
    elif answer is not None:
        assert _ref(operation["responses"]["200"]["content"]["application/json"]["schema"]) == answer
    assert operation["description"] or operation["summary"]


def _refusal_statuses(operation: dict) -> set[str]:
    return {status for status, response in operation["responses"].items()
            if response.get("content", {}).get("application/json", {}).get("schema", {}).get("$ref", "")
            .endswith("/LabRefusalResponse")}


def test_refusals_are_documented_with_their_shape(openapi):
    """``{"detail": {"code", "message"}}`` (R13) is in the schema, so the editor has every refusal code."""
    schemas = openapi["components"]["schemas"]
    assert _ref(schemas["LabRefusalResponse"]["properties"]["detail"]) == "LabRefusalDetail"
    assert set(schemas["LabRefusalDetail"]["properties"]["code"]["enum"]) == {
        "change_busy", "runs_full", "user_cap", "instance_cap", "not_owner", "preview_full", "not_found", "state"}
    assert _refusal_statuses(openapi["paths"]["/besser_api/lab/message"]["post"]) == {"404", "409", "429"}
    assert _refusal_statuses(openapi["paths"]["/besser_api/lab/change/drop"]["post"]) == {"403", "404", "409"}
    assert _refusal_statuses(openapi["paths"]["/besser_api/lab/preview/slot"]["post"]) == {"404", "409"}


def test_query_parameters_carry_their_bounds(openapi):
    logs = {p["name"]: p for p in openapi["paths"]["/besser_api/lab/preview/logs/{service}"]["get"]["parameters"]}
    assert logs["lines"]["schema"]["minimum"] == 1 and logs["lines"]["schema"]["maximum"] == 2000
    assert logs["lines"]["schema"]["default"] == 200 and not logs["lines"]["required"]
    assert logs["slot"]["required"] and logs["slot"]["schema"]["minimum"] == 0
    assert set(logs["service"]["schema"]["enum"]) == {"backend", "frontend", "modeling_agent"}

    change = {p["name"]: p for p in openapi["paths"]["/besser_api/lab/change"]["get"]["parameters"]}
    assert change["changeId"]["required"] and change["changeId"]["schema"]["pattern"] == "^[0-9a-f]{12}$"

    diff = {p["name"]: p for p in openapi["paths"]["/besser_api/lab/diff"]["get"]["parameters"]}
    assert diff["changeId"]["required"] and not diff["repo"]["required"] and not diff["path"]["required"]

    audit = {p["name"]: p for p in openapi["paths"]["/besser_api/lab/audit"]["get"]["parameters"]}
    assert not audit["month"]["required"]


@pytest.mark.parametrize("method, path", REMOVED)
def test_phase_1_routes_are_gone(openapi, client, lab_on, method, path):
    assert path not in openapi["paths"]
    assert client.request(method, path, json={}, headers=SESSION).status_code in (404, 405)


# -- The gate and the 501 placeholder ------------------------------------------------------------------

@pytest.mark.parametrize("method, path, query, body, request_model, answer", ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("headers", [SESSION, {}], ids=["signed-in", "anonymous"])
def test_every_route_is_404_with_the_lab_off(client, route_settings, app_state, headers, method, path, query, body,
                                             request_model, answer):
    route_settings.lab_enabled = False
    app_state.github_sessions.user_tokens.set("sess-octocat", {"access_token": "gho_octo", "username": "octocat"})

    response = _call(client, method, path, query, body, headers=headers)

    assert response.status_code == 404 and response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("method, path, query, body, request_model, answer", ROUTES, ids=ROUTE_IDS)
def test_every_route_needs_an_allowed_session(client, lab_on, method, path, query, body, request_model, answer):
    assert _call(client, method, path, query, body, headers={}).status_code == 401
    assert _call(client, method, path, query, body, headers={"X-GitHub-Session": "sess-mallory"}).status_code == 403


@pytest.mark.parametrize("method, path, query, body, request_model, answer", ROUTES, ids=ROUTE_IDS)
def test_every_route_answers_501_until_the_lab_is_rebuilt(client, lab_on, method, path, query, body,
                                                          request_model, answer):
    response = _call(client, method, path, query, body)

    assert response.status_code == 501, response.text
    assert response.json() == {"detail": NOT_BUILT}


def test_a_bad_change_id_is_refused_before_the_placeholder(client, lab_on):
    response = client.get("/besser_api/lab/change", params={"changeId": "../../etc"}, headers=SESSION)
    assert response.status_code == 422


def test_the_service_placeholder_starts_no_lab(route_settings, caplog):
    route_settings.lab_enabled = True

    assert create_lab_service(route_settings) is None
    assert "BESSER Lab is being rebuilt; its routes answer 501" in caplog.text
    assert LabService.__doc__


def test_a_refusal_carries_code_message_and_status():
    refusal = LabRefusal("change_busy", "ana's build is running.", 409)

    assert (refusal.code, refusal.message, refusal.status) == ("change_busy", "ana's build is running.", 409)
    assert str(refusal) == "ana's build is running."


# -- Request limits -----------------------------------------------------------------------------------

@pytest.mark.parametrize("change_id", ["abc", "ABCDEF012345", "abcdef0123456", "../../etc/pa", None])
def test_a_message_needs_a_well_formed_change_id(change_id):
    body = {"message": "hi"} if change_id is None else {"message": "hi", "changeId": change_id}
    with pytest.raises(ValidationError):
        LabMessageRequest.model_validate(body)
    assert LabMessageRequest.model_validate({"message": "hi", "changeId": CHANGE_ID}).changeId == CHANGE_ID


def test_a_pull_request_needs_its_change_id():
    with pytest.raises(ValidationError):
        LabPullRequestRequest.model_validate({})
    request = LabPullRequestRequest.model_validate(
        {"changeId": CHANGE_ID, "closesIssues": [{"repo": "besser", "number": 744}]})
    assert request.closesIssues[0].number == 744


def test_a_note_holds_at_most_1200_characters_and_a_100_character_title():
    LabNoteInput.model_validate({"kind": "convention", "title": "t" * 100, "body": "b" * 1200})
    with pytest.raises(ValidationError):
        LabNoteInput.model_validate({"kind": "convention", "title": "t", "body": "b" * 1201})
    with pytest.raises(ValidationError):
        LabNoteInput.model_validate({"kind": "convention", "title": "t" * 101, "body": "b"})


# -- The change thread ------------------------------------------------------------------------------------

THREAD_ITEMS = [
    ({"type": "message", "id": "0123456789ab", "at": 1.0, "login": "ana", "text": "Add a note",
      "action": "auto", "model": None}, LabThreadMessage),
    ({"type": "run", "runId": "f" * 32, "at": 2.0, "login": "ana", "status": "running", "lab": None,
      "error": None, "costUsd": None, "planVersion": None}, LabThreadRun),
    ({"type": "pull_requests", "at": 3.0, "login": "jordi", "checksNote": "2 passed.",
      "pullRequests": [{"repo": "besser", "number": 812, "url": "https://github.com/o/r/pull/812",
                        "branch": "lab/jordi/note", "base": "development"}]}, LabThreadPullRequests),
    ({"type": "notice", "at": 4.0, "login": "ana", "text": "Updated from development."}, LabThreadNotice),
]


@pytest.mark.parametrize("item, model", THREAD_ITEMS, ids=[i["type"] for i, _ in THREAD_ITEMS])
def test_a_thread_item_is_parsed_by_its_type(item, model):
    assert type(TypeAdapter(LabThreadItem).validate_python(item)) is model


def test_a_thread_item_of_an_unknown_type_is_refused():
    with pytest.raises(ValidationError):
        TypeAdapter(LabThreadItem).validate_python({"type": "other", "at": 1.0, "login": "ana", "text": "x"})
