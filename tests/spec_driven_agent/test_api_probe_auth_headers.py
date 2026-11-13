"""test_api can authenticate a scenario, and never records the credential.

Production run 438889bc: every create on a token-protected API answered 401,
because a test_api request could carry no header. Constructibility reported
"create unverified", and only a passing POST->GET pair retires that - which
needed a token the tool had no way to send.
"""
import json
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.validation import api_probe

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")

MAIN_API = '''
import secrets
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()
USERS, TOKENS, NOTES = {}, {}, {}


def current_user(authorization: str = Header(default="")):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token not in TOKENS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return TOKENS[token]


@app.post("/auth/register")
def register(body: dict):
    USERS[body["username"]] = body["password"]
    return {"username": body["username"]}


@app.post("/auth/login")
def login(body: dict):
    if USERS.get(body["username"]) != body["password"]:
        raise HTTPException(status_code=401, detail="bad credentials")
    token = secrets.token_urlsafe(24)
    TOKENS[token] = body["username"]
    return {"access_token": token, "token_type": "bearer"}


@app.post("/note/")
def create_note(body: dict, user: str = Depends(current_user)):
    note = {"id": len(NOTES) + 1, "title": body["title"], "owner": user}
    NOTES[note["id"]] = note
    return note


@app.get("/note/{note_id}/")
def read_note(note_id: int, user: str = Depends(current_user)):
    return NOTES[note_id]


@app.get("/whoami")
def whoami(authorization: str = Header(default="")):
    return {"echo": authorization}
'''

AUTH = {"Authorization": "Bearer {{1.access_token}}"}
SCENARIO = [
    {"method": "POST", "path": "/auth/register", "json": {"username": "ann", "password": "pw-123456"}},
    {"method": "POST", "path": "/auth/login", "json": {"username": "ann", "password": "pw-123456"}},
    {"method": "POST", "path": "/note/", "json": {"title": "first"}, "headers": AUTH},
    {"method": "GET", "path": "/note/{{2.id}}/", "headers": AUTH, "expected_fields": {"title": "first"}},
    {"method": "GET", "path": "/whoami", "headers": AUTH},
]


def backend(tmp_path):
    (tmp_path / "main_api.py").write_text(MAIN_API, encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    return tmp_path


def test_register_login_then_authenticated_create_and_read_pass(tmp_path):
    report = api_probe.probe_api_scenario(str(backend(tmp_path)), SCENARIO)

    assert report["boot"] == "ok", report
    assert report["status"] == "passed", report
    assert [response["status"] for response in report["responses"]] == [200, 200, 200, 200, 200]
    # The POST->GET pair that clears "create unverified" is now reachable.
    assert api_probe.confirmed_create_paths(report) == {"/note"}
    assert report["responses"][2]["headers"] == ["Authorization"]


def test_without_the_header_the_same_create_is_refused(tmp_path):
    unauthenticated = [dict(request) for request in SCENARIO[:3]]
    del unauthenticated[2]["headers"]
    report = api_probe.probe_api_scenario(str(backend(tmp_path)), unauthenticated)
    assert report["responses"][2]["status"] == 401


def test_the_credential_never_appears_in_the_report(tmp_path):
    report = api_probe.probe_api_scenario(str(backend(tmp_path)), SCENARIO)
    assert report["status"] == "passed", report
    serialized = json.dumps(report)
    # The login answer and the app echoing the header both reach the report.
    assert report["responses"][1]["json"]["access_token"] == "[REDACTED]"
    assert report["responses"][4]["json"]["echo"] == "[REDACTED]"
    assert "Bearer " not in serialized.replace("Bearer {{", "")


@pytest.mark.parametrize("name", [
    "Host", "Cookie", "Connection", "Transfer-Encoding", "Content-Length",
    "X-Forwarded-For", "X-Real-IP", "X-HTTP-Method-Override", "Proxy-Authorization", "Bad Name",
])
def test_unsafe_headers_are_rejected_before_anything_runs(tmp_path, name):
    request = {"method": "GET", "path": "/whoami", "headers": {name: "value"}}
    report = api_probe.probe_api_scenario(str(backend(tmp_path)), [request])
    assert report["status"] == "error" and report["boot"] == "not_started"
    assert "not allowed" in report["error"]


def test_header_values_must_be_text_and_errors_never_quote_them(tmp_path):
    for value in (7, "Bearer abc\r\nX-Evil: 1", "Bearer é"):
        report = api_probe.probe_api_scenario(
            str(backend(tmp_path)), [{"method": "GET", "path": "/whoami", "headers": {"Authorization": value}}])
        assert report["boot"] == "not_started"
        assert "abc" not in report["error"] and "é" not in report["error"]


def test_x_api_key_headers_are_allowed_and_redacted_for_display():
    api_probe._validate_requests([{"method": "GET", "path": "/x", "headers": {
        "X-API-Key": "k", "X-Tenant-ID": "t", "Accept": "application/json"}}])
    assert api_probe.redact_headers({"X-API-Key": "k", "X-Tenant-ID": "t", "Authorization": "Bearer s"}) == {
        "X-API-Key": "[REDACTED]", "X-Tenant-ID": "t", "Authorization": "[REDACTED]"}


def test_orchestrator_result_and_trace_never_carry_the_header_value(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    backend(tmp_path)
    literal = "Bearer literal-secret-token-value"
    scenario = [dict(SCENARIO[4], headers={"Authorization": literal})]
    client = SimpleNamespace(model="mock-model", usage=UsageTracker("mock-model"))
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=DomainModel(name="Probe", types={Class(name="Note")}),
        output_dir=str(tmp_path), enable_checkpointing=False, enable_requirements_ledger=False,
        enable_toolchain_validation=False)
    block = SimpleNamespace(id="t1", name="test_api", input={"scenario_id": "whoami", "requests": scenario})

    result = orchestrator._execute_single_tool(block, turn=0)
    shown = orchestrator._test_api({"action": "get", "scenario_id": "whoami"})

    assert json.loads(result["content"])["status"] == "passed", result
    trace = (tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8")
    for text in (result["content"], json.dumps(shown), trace, json.dumps(orchestrator.tool_calls_log)):
        assert "literal-secret-token-value" not in text
    assert shown["requests"][0]["headers"] == {"Authorization": "[REDACTED]"}


# Tokens the scenario never sends back: an unknown key holding a JWT, and a
# nested refresh token. The value-only scrub could not learn either of them.
UNSENT_TOKENS_API = MAIN_API + '''

def _jwt():
    return "eyJhbGciOiJIUzI1NiJ9." + secrets.token_urlsafe(24) + "." + secrets.token_urlsafe(32)


ISSUED = []


@app.post("/auth/signup")
def signup(body: dict):
    USERS[body["username"]] = body["password"]
    ISSUED.append(_jwt())
    return {"id": 1, "username": body["username"], "issued": ISSUED[-1]}


@app.post("/auth/session")
def session(body: dict):
    token, refresh = _jwt(), secrets.token_urlsafe(24)
    TOKENS[token] = body["username"]
    ISSUED.append(token)
    return {"data": {"session": {"access_token": token, "refresh_token": refresh}}}


@app.get("/issued")
def issued():
    return ISSUED
'''

UNSENT_AUTH = {"Authorization": "Bearer {{1.data.session.access_token}}"}
UNSENT_SCENARIO = [
    {"method": "POST", "path": "/auth/signup", "json": {"username": "ann", "password": "pw-123456"}},
    {"method": "POST", "path": "/auth/session", "json": {"username": "ann", "password": "pw-123456"},
     "expected_fields": {"data.session.refresh_token": "wrong"}},
    {"method": "POST", "path": "/note/", "json": {"title": "first"}, "headers": UNSENT_AUTH},
    {"method": "GET", "path": "/note/{{2.id}}/", "headers": UNSENT_AUTH, "expected_fields": {"title": "first"}},
    {"method": "GET", "path": "/issued", "expected_fields": {"0": "wrong"}},
]


def unsent_backend(tmp_path):
    (tmp_path / "main_api.py").write_text(UNSENT_TOKENS_API, encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    return tmp_path


def test_tokens_never_sent_back_are_redacted_and_references_still_authenticate(tmp_path):
    """Sonnet p6 runs: a register response's JWT, never reused in a header, reached
    the model and .besser_recipe.json unredacted while the login token was hidden."""
    report = api_probe.probe_api_scenario(str(unsent_backend(tmp_path)), UNSENT_SCENARIO)

    assert report["boot"] == "ok", report
    assert [response["status"] for response in report["responses"]] == [200, 200, 200, 200, 200]
    assert report["responses"][3]["failures"] == []
    assert api_probe.confirmed_create_paths(report) == {"/note"}
    assert report["responses"][0]["json"]["issued"] == "[REDACTED]"
    assert report["responses"][1]["json"]["data"]["session"] == {
        "access_token": "[REDACTED]", "refresh_token": "[REDACTED]"}
    # Failure messages quote the actual value: a JWT and a credential field.
    assert report["responses"][4]["json"] == ["[REDACTED]"] * 2
    assert report["responses"][4]["failures"] == ["0: expected 'wrong', got '[REDACTED]'"]
    assert report["responses"][1]["failures"] == [
        "data.session.refresh_token: expected 'wrong', got '[REDACTED]'"]
    assert "eyJ" not in json.dumps(report)


def test_redaction_keeps_ordinary_fields():
    shown = api_probe._scrub({"author": "Ann", "max_tokens": 5, "token_type": "bearer", "keywords": "a",
                              "accessToken": "opaque-value", "user": {"password": "pw"}}, ())
    assert shown == {"author": "Ann", "max_tokens": 5, "token_type": "bearer", "keywords": "a",
                     "accessToken": "[REDACTED]", "user": {"password": "[REDACTED]"}}


def test_orchestrator_result_and_saved_snapshot_never_carry_unsent_tokens(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker
    from besser.spec_driven_agent.state.checkpoint import api_scenario_snapshot

    unsent_backend(tmp_path)
    client = SimpleNamespace(model="mock-model", usage=UsageTracker("mock-model"))
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=DomainModel(name="Probe", types={Class(name="Note")}),
        output_dir=str(tmp_path), enable_checkpointing=False, enable_requirements_ledger=False,
        enable_toolchain_validation=False)
    block = SimpleNamespace(id="t1", name="test_api", input={"scenario_id": "unsent", "requests": UNSENT_SCENARIO})

    result = orchestrator._execute_single_tool(block, turn=0)
    shown = orchestrator._test_api({"action": "get", "scenario_id": "unsent"})
    saved = api_scenario_snapshot(orchestrator._api_scenarios.values(), include_reports=True)

    report = json.loads(result["content"])
    assert [response["status"] for response in report["responses"]] == [200, 200, 200, 200, 200], report
    for text in (result["content"], json.dumps(shown), json.dumps(saved)):
        assert "eyJ" not in text
        assert '"refresh_token": "[REDACTED]"' in text
