"""The BESSER Lab routes over HTTP, on a real working copy of two git repositories under ``tmp_path``."""

import copy
import json
import os
import subprocess
import sys

import pytest
from pydantic import SecretStr

from besser.spec_driven_agent.lab import checks as lab_checks
from besser.spec_driven_agent.lab import session as session_module
from besser.spec_driven_agent.lab.contracts import (
    BuildOutcome,
    ChangeKind,
    CheckResult,
    LoopResult,
    Plan,
    RepoSpec,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.lab import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.lab import service as service_module
from besser.utilities.web_modeling_editor.backend.services.lab.model_proposal import proposal_from_plan
from besser.utilities.web_modeling_editor.backend.services.lab.preview import PreviewService, PreviewSupervisor
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    PullRequestError,
    PullRequestInfo,
)
from besser.utilities.web_modeling_editor.backend.services.lab.service import LabService, create_lab_service
from tests.spec_driven_agent.lab.conftest import ScriptedClient, StreamingScriptedClient, end_turn, tool_use
from tests.utilities.web_modeling_editor.backend.lab.test_lab_runner import FakeClient, FakeSession

SESSION = {"X-GitHub-Session": "sess-octocat"}

ROUTES = [
    ("POST", "/besser_api/lab/message", {"message": "hi"}),
    ("GET", "/besser_api/lab/state", None),
    ("POST", "/besser_api/lab/pull-request", {}),
    ("POST", "/besser_api/lab/change/park", {}),
    ("POST", "/besser_api/lab/change/resume", {"changeId": "abcdef012345"}),
    ("POST", "/besser_api/lab/change/drop", {}),
    ("POST", "/besser_api/lab/preview/reset", {}),
]


def git(cwd, *args) -> str:
    result = subprocess.run(["git", "-c", "user.name=setup", "-c", "user.email=setup@example.com",
                             "-c", "core.autocrlf=false", *args], cwd=cwd, check=True, capture_output=True)
    return result.stdout.decode("utf-8").strip()


@pytest.fixture
def lab_settings(route_settings):
    route_settings.lab_enabled = True
    route_settings.lab_allowed_logins = "octocat"
    route_settings.lab_llm_api_key = SecretStr("sk-ant-lab-key-0123456789")
    route_settings.lab_llm_model = "claude-test"
    return route_settings


@pytest.fixture
def workspace(tmp_path):
    """BESSER and the modeling agent, each a git checkout on its base branch with one commit."""
    root = tmp_path / "work"
    repos = []
    for key, name, base in (("besser", "besser", "development"), ("modeling_agent", "modeling-agent", "develop")):
        path = root / name
        path.mkdir(parents=True)
        git(path, "init", "-q", "-b", base)
        (path / "README.md").write_text(f"{name}\n", encoding="utf-8")
        git(path, "add", "-A")
        git(path, "commit", "-q", "-m", "initial")
        repos.append(RepoSpec(key, str(path), base, "BESSER-PEARL", name))
    return LabWorkspace(str(root), repos=repos)


@pytest.fixture
def lab(workspace, lab_settings, app_state):
    """The lab service on ``app.state.lab``, and the signed-in users octocat (allowed) and mallory."""
    service = LabService(workspace)
    app.state.lab = service
    app_state.github_sessions.user_tokens.set("sess-octocat", {"access_token": "gho_octo", "username": "octocat"})
    app_state.github_sessions.user_tokens.set("sess-mallory", {"access_token": "gho_mal", "username": "mallory"})
    yield service
    del app.state.lab


def start_change(lab, kind=ChangeKind.BUGFIX):
    change = lab.workspace.start_change(user="octocat", title="Fix the note", kind=kind)
    path = os.path.join(lab.workspace.repos["besser"].path, "besser", "note.py")
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("NOTE = 1\n")
    return change


def call(client, method, path, body, headers=SESSION):
    return client.request(method, path, json=body, headers=headers)


def branch(lab, repo) -> str:
    return git(lab.workspace.repos[repo].path, "branch", "--show-current")


# -- The gate ------------------------------------------------------------------------------------------

# Well-formed JSON that the request models refuse: with the lab off the gate answers before the body is validated.
REFUSED_BODIES = [
    ("POST", "/besser_api/lab/message", {}),
    ("POST", "/besser_api/lab/message", ["not", "an", "object"]),
    ("POST", "/besser_api/lab/change/resume", {"changeId": "../../etc"}),
]


@pytest.mark.parametrize("method, path, body", ROUTES + REFUSED_BODIES)
@pytest.mark.parametrize("headers", [SESSION, {}], ids=["signed-in", "anonymous"])
def test_every_lab_route_is_404_when_the_lab_is_off(client, app_state, lab_settings, method, path, body, headers):
    # As on a public editor: no lab service on ``app.state`` (``get_lab_service`` would answer 503).
    lab_settings.lab_enabled = False
    assert getattr(app.state, "lab", None) is None

    response = call(client, method, path, body, headers=headers)

    assert response.status_code == 404 and response.json() == {"detail": "Not Found"}


_OPENAPI_OF_A_FRESH_APP = """
import json
from fastapi.testclient import TestClient
from besser.utilities.web_modeling_editor.backend.backend import app
broken = TestClient(app).post("/besser_api/lab/message", content=b"{not json",
                              headers={"Content-Type": "application/json"})
print(json.dumps({"openapi": app.openapi(), "broken_json": broken.status_code}))
"""


def _fresh_app(enabled: str | None) -> dict:
    env = {name: value for name, value in os.environ.items() if name.upper() != "BESSER_LAB_ENABLED"}
    if enabled:
        env["BESSER_LAB_ENABLED"] = enabled
    result = subprocess.run([sys.executable, "-c", _OPENAPI_OF_A_FRESH_APP], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_openapi_schema_is_the_same_with_the_lab_on_or_off():
    """The editor commits one export and CI compares it with a lab-off app, so the lab routes are always in it."""
    off, on = _fresh_app(None), _fresh_app("true")

    assert json.dumps(off["openapi"], indent=2) == json.dumps(on["openapi"], indent=2)
    assert {path for path in off["openapi"]["paths"] if path.startswith("/besser_api/lab/")} == {
        path for _, path, _ in ROUTES}
    # A body that is not JSON at all fails before the lab gate runs: 422 rather than 404, accepted.
    assert off["broken_json"] == on["broken_json"] == 422


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_lab_route_needs_the_session_of_an_allowed_login(client, lab, method, path, body):
    assert call(client, method, path, body, headers={}).status_code == 401
    assert call(client, method, path, body, headers={"X-GitHub-Session": "sess-mallory"}).status_code == 403


def test_a_lab_without_a_working_copy_answers_503(client, lab_settings, app_state):
    app_state.github_sessions.user_tokens.set("sess-octocat", {"access_token": "gho_octo", "username": "octocat"})
    app.state.lab = None
    try:
        response = client.get("/besser_api/lab/state", headers=SESSION)
    finally:
        del app.state.lab

    assert response.status_code == 503
    assert "BESSER_LAB_WORKSPACE_ROOT" in response.json()["detail"]


def test_no_lab_service_without_a_workspace_root(route_settings):
    route_settings.lab_enabled = True
    route_settings.lab_workspace_root = None

    assert create_lab_service(route_settings) is None


def test_a_working_copy_that_is_not_ready_is_logged_and_kept(route_settings, tmp_path, caplog):
    route_settings.lab_enabled = True
    route_settings.lab_workspace_root = str(tmp_path / "empty")

    service = create_lab_service(route_settings)

    assert service.workspace.root == str(tmp_path / "empty")
    assert service.preview is None
    assert any("not ready" in record.message for record in caplog.records)


# -- State and change actions ----------------------------------------------------------------------------

def test_the_state_of_a_lab_without_changes(client, lab):
    response = client.get("/besser_api/lab/state", headers=SESSION)

    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": True, "user": "octocat", "change": None, "parked": [],
                               "preview": {"enabled": False, "services": []}, "run": None}


def test_the_state_names_the_run_in_progress(client, lab):
    """A client that lost its stream finds the run to follow (``/spec-driven/runs/{id}/events``) or stop."""
    assert lab.try_begin_run("a" * 32)
    try:
        run = client.get("/besser_api/lab/state", headers=SESSION).json()["run"]
    finally:
        lab.end_run()

    assert run["runId"] == "a" * 32 and isinstance(run["startedAt"], float)
    assert client.get("/besser_api/lab/state", headers=SESSION).json()["run"] is None


def test_the_state_shows_the_active_change_and_its_files_now(client, lab):
    change = start_change(lab)

    state = client.get("/besser_api/lab/state", headers=SESSION).json()

    assert state["change"] == {
        "step": "plan", "changeId": change.id, "title": "Fix the note", "kind": "bugfix",
        "repoChanges": [{"repo": "besser", "branch": "lab/octocat/fix-the-note", "base": "development",
                         "files": [{"path": "besser/note.py", "status": "added"}]}],
        "checks": [], "checklist": [], "summary": "", "stopReason": "completed"}


def test_the_state_lists_the_preview_services(client, lab, tmp_path):
    lab.preview = PreviewSupervisor([PreviewService(name="backend", argv=("python",), cwd=str(tmp_path), env={},
                                                    watch=(), port=9100)], log_dir=str(tmp_path / "logs"))

    preview = client.get("/besser_api/lab/state", headers=SESSION).json()["preview"]

    assert preview == {"enabled": True, "services": [{"name": "backend", "running": False, "detail": "not started"}]}


def test_park_resume_and_drop_move_the_working_copy(client, lab):
    change = start_change(lab)

    parked = call(client, "POST", "/besser_api/lab/change/park", {})
    assert parked.status_code == 200, parked.text
    assert parked.json()["change"] is None
    assert parked.json()["parked"] == [{"changeId": change.id, "title": "Fix the note", "kind": "bugfix",
                                        "createdAt": change.created_at}]
    assert branch(lab, "besser") == "development"

    resumed = call(client, "POST", "/besser_api/lab/change/resume", {"changeId": change.id})
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["change"]["repoChanges"][0]["files"] == [{"path": "besser/note.py", "status": "added"}]
    assert branch(lab, "besser") == "lab/octocat/fix-the-note"

    dropped = call(client, "POST", "/besser_api/lab/change/drop", {})
    assert dropped.status_code == 200, dropped.text
    assert (dropped.json()["change"], dropped.json()["parked"]) == (None, [])
    assert branch(lab, "besser") == "development"
    assert git(lab.workspace.repos["besser"].path, "branch", "--list", "lab/*") == ""


@pytest.mark.parametrize("path, message", [
    ("/besser_api/lab/change/park", "no active change to park"),
    ("/besser_api/lab/change/drop", "no active change to drop"),
    ("/besser_api/lab/pull-request", "no active change to open a pull request for"),
])
def test_an_action_on_no_active_change_is_409(client, lab, path, message):
    response = call(client, "POST", path, {})

    assert response.status_code == 409
    assert message in response.json()["detail"]


def test_resume_while_a_change_is_active_is_409_with_gits_reason(client, lab):
    first = start_change(lab)
    call(client, "POST", "/besser_api/lab/change/park", {})
    lab.workspace.start_change(user="octocat", title="Another change", kind=ChangeKind.OTHER)

    response = call(client, "POST", "/besser_api/lab/change/resume", {"changeId": first.id})

    assert response.status_code == 409
    assert "Another change" in response.json()["detail"]


def test_resume_refuses_a_malformed_change_id(client, lab):
    assert call(client, "POST", "/besser_api/lab/change/resume", {"changeId": "../../etc"}).status_code == 422


def test_the_preview_reset_checks_out_the_base_branches_only_without_an_active_change(client, lab):
    start_change(lab)
    refused = call(client, "POST", "/besser_api/lab/preview/reset", {})
    assert refused.status_code == 409
    assert "park or drop it" in refused.json()["detail"]

    call(client, "POST", "/besser_api/lab/change/park", {})
    git(lab.workspace.repos["modeling_agent"].path, "switch", "-q", "-c", "someone-elses-branch")

    reset = call(client, "POST", "/besser_api/lab/preview/reset", {})

    assert reset.status_code == 200, reset.text
    assert (branch(lab, "besser"), branch(lab, "modeling_agent")) == ("development", "develop")
    assert len(reset.json()["parked"]) == 1


@pytest.mark.parametrize("method, path, body", [
    ("POST", "/besser_api/lab/message", {"message": "hi"}),
    ("POST", "/besser_api/lab/change/park", {}),
    ("POST", "/besser_api/lab/pull-request", {}),
    ("POST", "/besser_api/lab/preview/reset", {}),
])
def test_one_run_or_action_at_a_time(client, lab, method, path, body):
    assert lab.try_begin_run("f" * 32)
    try:
        response = call(client, method, path, body)
    finally:
        lab.end_run()

    assert response.status_code == 409
    assert "in progress" in response.json()["detail"]


# -- Pull requests --------------------------------------------------------------------------------------

def test_the_pull_request_route_opens_them_with_the_users_token(client, lab, monkeypatch):
    change = start_change(lab)
    seen = {}

    async def fake_open(workspace, active, *, access_token, login, title, body):
        seen.update(change=active.id, access_token=access_token, login=login, title=title, body=body)
        return [PullRequestInfo("besser", 7, "https://github.com/BESSER-PEARL/BESSER/pull/7",
                                "lab/octocat/fix-the-note", "development")]

    monkeypatch.setattr(service_module, "open_pull_requests", fake_open)

    response = call(client, "POST", "/besser_api/lab/pull-request", {"title": "fix: the note", "body": None})

    assert response.status_code == 200, response.text
    assert response.json() == {"pullRequests": [{
        "repo": "besser", "number": 7, "url": "https://github.com/BESSER-PEARL/BESSER/pull/7",
        "branch": "lab/octocat/fix-the-note", "base": "development"}],
        "checksNote": "The lab ran no checks on this change. The repositories' full test suites did not run in "
                      "the lab."}
    assert seen == {"change": change.id, "access_token": "gho_octo", "login": "octocat", "title": "fix: the note",
                    "body": None}


def test_a_pull_request_from_the_card_of_a_change_that_is_no_longer_active_is_409(client, lab, monkeypatch):
    # The card of the first change stays in the chat after it is parked and another change starts.
    first = start_change(lab)
    lab.workspace.park(first)
    second = lab.workspace.start_change(user="octocat", title="Second change", kind=ChangeKind.BUGFIX)
    pushed = []

    async def fake_open(workspace, active, **kwargs):
        pushed.append(active.id)
        return []

    monkeypatch.setattr(service_module, "open_pull_requests", fake_open)

    response = call(client, "POST", "/besser_api/lab/pull-request", {"changeId": first.id, "title": None,
                                                                      "body": None})

    assert response.status_code == 409
    assert response.json()["detail"] == "This change is not the active one: resume it on the lab page first."
    assert pushed == []
    assert lab.try_begin_run("f" * 32)  # the lab is free again
    lab.end_run()
    ok = call(client, "POST", "/besser_api/lab/pull-request", {"changeId": second.id})
    assert ok.status_code == 200, ok.text
    assert pushed == [second.id]


def test_a_refused_pull_request_is_409_with_its_message(client, lab, monkeypatch):
    start_change(lab)

    async def refuse(*args, **kwargs):
        raise PullRequestError("This change touches .github/workflows/ci.yml, which the sign-in token cannot push.")

    monkeypatch.setattr(service_module, "open_pull_requests", refuse)

    response = call(client, "POST", "/besser_api/lab/pull-request", {})

    assert response.status_code == 409
    assert ".github/workflows/ci.yml" in response.json()["detail"]
    assert lab.try_begin_run("f" * 32)  # the lab is free again


# -- The run --------------------------------------------------------------------------------------------

def sse_events(response) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in response.text.split("\n") if line.startswith("data: ")]


def test_a_message_streams_the_run_and_the_state_keeps_its_checks(client, lab, monkeypatch):
    change = start_change(lab)
    handled = []

    def script(session, message):
        session.callbacks.on_phase("build", "Building the change")
        session.callbacks.on_phase("check", "Running the checks")
        handled.append((message, session.workspace))
        build = BuildOutcome(loop=LoopResult("completed", final_text="Fixed the note."),
                             changes=lab.workspace.repo_changes(change),
                             checks=[CheckResult("besser:lint", "besser", "Ruff", passed=False, summary="1 error")],
                             summary="besser: 1 file added")
        return SessionOutcome(step="build", change=change, build=build)

    monkeypatch.setattr(runner_module, "create_llm_client", lambda provider, **kw: FakeClient())
    monkeypatch.setattr(runner_module, "LabSession",
                        lambda workspace, **kw: FakeSession(workspace, **kw, script=script, record=[]))

    response = client.post("/besser_api/lab/message", json={"message": "Fix the note", "action": "build"},
                           headers=SESSION)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    got = sse_events(response)
    assert [e["event"] for e in got][0] == "start" and got[-1]["event"] == "done"
    assert response.headers["X-BESSER-Run-Id"] == got[0]["runId"]
    assert [e["phase"] for e in got if e["event"] == "phase"] == ["build", "check"]
    assert got[-1]["changedFiles"] == ["besser/besser/note.py"]
    # The model's own account of the build, for the card; the state does not keep it.
    assert got[-1]["lab"]["finalText"] == "Fixed the note."
    assert handled == [("Fix the note", lab.workspace)]

    # A restart: the checks are kept with the change, not in the service's memory.
    app.state.lab = LabService(lab.workspace)
    state = client.get("/besser_api/lab/state", headers=SESSION).json()
    assert state["change"]["step"] == "build"
    assert state["change"]["checks"] == [{"id": "besser:lint", "repo": "besser", "title": "Ruff", "passed": False,
                                          "skipped": False, "summary": "1 error"}]
    assert state["change"]["summary"] == "besser: 1 file added"
    assert "finalText" not in state["change"]
    assert state["run"] is None
    assert lab.try_begin_run("f" * 32)  # released at the end of the run

    # The pull requests say that only these quick checks ran.
    async def fake_open(*args, **kwargs):
        return []

    monkeypatch.setattr(service_module, "open_pull_requests", fake_open)
    pull = call(client, "POST", "/besser_api/lab/pull-request", {}).json()
    assert pull["checksNote"] == ("The lab's last build ran only the checks of the touched areas: 0 passed, "
                                  "1 failed, 0 skipped. The repositories' full test suites did not run in the lab.")


def test_the_real_lab_session_plans_a_change_through_the_route(client, lab, monkeypatch):
    """The route, the runner and P6's LabSession with a scripted model: a plan step creates the change."""
    plan = {"kind": "bugfix", "title": "Fix the README", "summary": "A typo in the README.",
            "steps": ["Fix the typo"], "repos": ["besser"], "questions": ["Which spelling?"]}
    model = StreamingScriptedClient([
        tool_use("read_file", {"path": "besser/README.md"}, text="Reading the README."),
        tool_use("submit_plan", plan),
    ], cost_per_call=0.01)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda provider, **kw: model)

    response = client.post("/besser_api/lab/message", json={"message": "Fix the typo in the README"},
                           headers=SESSION)

    got = sse_events(response)
    assert response.status_code == 200, response.text
    assert got[0]["event"] == "start" and got[-1]["event"] == "done", got
    assert {"event": "phase", "phase": "plan"}.items() <= next(e for e in got if e["event"] == "phase").items()
    assert any(e["event"] == "tool_call" and e["tool"] == "read_file" for e in got)
    assert "Reading the README." in "".join(e["delta"] for e in got if e["event"] == "text")
    lab_payload = got[-1]["lab"]
    change = lab.workspace.active_change()
    assert (lab_payload["step"], lab_payload["changeId"], lab_payload["kind"]) == ("plan", change.id, "bugfix")
    assert lab_payload["plan"]["questions"] == ["Which spelling?"] and lab_payload["modelProposal"] is None
    assert branch(lab, "besser") == "lab/octocat/fix-the-readme"
    assert client.get("/besser_api/lab/state", headers=SESSION).json()["change"]["step"] == "plan"


# -- A metamodel change from plan to build, with the user's review in the editor ----------------------------

NOTE_PLAN = {
    "kind": "metamodel", "title": "Add a Note metaclass", "summary": "A note metaclass.",
    "steps": ["Add besser/BUML/metamodel/note.py"], "repos": ["besser"],
    "model_spec": {"classes": [{"name": "Element", "is_abstract": True},
                               {"name": "Note", "attributes": [{"name": "text", "type": "str"}]}],
                   "generalizations": [{"child": "Note", "parent": "Element"}]},
}


@pytest.fixture
def scripted_models(monkeypatch):
    """Each run gets the next scripted model; the checks pass without running anything."""
    queue = []
    monkeypatch.setattr(runner_module, "create_llm_client", lambda provider, **kw: queue[0])
    def passing(specs, repos, **kw):
        return lab_checks.run_checks(specs, repos, **{**kw, "runner": lambda spec, cwd: (0, "ok")})

    monkeypatch.setattr(session_module, "run_checks", passing)
    return queue


def _diagram(diagram_id, title, model, **config):
    return {"id": diagram_id, "title": title, "model": model, **({"config": config} if config else {})}


def _project_of(diagrams, active):
    return {"id": "p1", "type": "Project", "name": "lab", "createdAt": "2026-11-19T00:00:00Z",
            "currentDiagramType": "ClassDiagram", "currentDiagramIndices": {"ClassDiagram": active},
            "diagrams": {"ClassDiagram": diagrams}}


def _corrected(model):
    """The user's review in the editor: Note renamed to Memo, attribute ``b: int`` added."""
    model = copy.deepcopy(model)
    note = next(e for e in model["elements"].values() if e.get("name") == "Note")
    note["name"] = "Memo"
    attribute = copy.deepcopy(model["elements"][note["attributes"][0]])
    attribute.update(id="attr-b", name="b", attributeType="int", owner=note["id"])
    model["elements"]["attr-b"] = attribute
    note["attributes"].append("attr-b")
    return model


def _users_own_model():
    plan = Plan(kind=ChangeKind.OTHER, title="Library", summary="", steps=[], repos=[],
                model_spec={"classes": [{"name": "Library", "attributes": [{"name": "name", "type": "str"}]}]})
    return proposal_from_plan(plan)["model"]


@pytest.mark.parametrize("marker", ["labProposalChangeId", "labChangeId"])
def test_the_build_follows_the_proposal_the_user_corrected_not_the_open_diagram(
        client, lab, scripted_models, marker):
    scripted_models[:] = [ScriptedClient([tool_use("submit_plan", NOTE_PLAN)])]
    planned = sse_events(client.post("/besser_api/lab/message", json={"message": "add notes"}, headers=SESSION))
    lab_payload = planned[-1]["lab"]
    assert lab_payload["step"] == "plan" and lab_payload["modelProposal"] is not None, planned

    # The user corrected the proposal (kept or still pending), then went back to their own model's tab.
    proposal = _diagram("d2", "Add a Note metaclass", _corrected(lab_payload["modelProposal"]["model"]),
                        **{marker: lab_payload["changeId"]})
    project = _project_of([_diagram("d1", "Library", _users_own_model()), proposal], active=0)
    build_model = ScriptedClient([end_turn("Added the Memo metaclass.")])
    scripted_models[:] = [build_model]
    body = {"message": "build it", "action": "build", "project": project}
    built = sse_events(client.post("/besser_api/lab/message", json=body, headers=SESSION))

    assert built[-1]["event"] == "done" and built[-1]["lab"]["step"] == "build", built
    first_message = build_model.calls[0][1][0]["content"]
    assert "Memo" in first_message and "b: int" in first_message
    assert "Library" not in first_message


def test_without_the_changes_diagram_the_build_reads_no_class_diagram(client, lab, scripted_models):
    """The proposal was discarded (or never shown): the user's own model is not the extension's model."""
    scripted_models[:] = [ScriptedClient([tool_use("submit_plan", NOTE_PLAN)])]
    sse_events(client.post("/besser_api/lab/message", json={"message": "add notes"}, headers=SESSION))
    build_model = ScriptedClient([end_turn("Added the Note metaclass.")])
    scripted_models[:] = [build_model]
    project = _project_of([_diagram("d1", "Library", _users_own_model(), labProposalChangeId="0123456789ab")],
                          active=0)

    body = {"message": "build it", "action": "build", "project": project}
    built = sse_events(client.post("/besser_api/lab/message", json=body, headers=SESSION))

    assert built[-1]["event"] == "done", built
    assert "Library" not in build_model.calls[0][1][0]["content"]


# -- The config flag ------------------------------------------------------------------------------------

def test_the_config_says_whether_this_instance_is_a_lab(client, route_settings):
    assert client.get("/besser_api/spec-driven/config").json()["lab"] == {"enabled": False}

    route_settings.lab_enabled = True

    assert client.get("/besser_api/spec-driven/config").json()["lab"] == {"enabled": True}


@pytest.mark.parametrize("path", ["/besser_api/spec-driven/generate", "/besser_api/spec-driven/resume/" + "a" * 32])
def test_a_lab_instance_starts_no_spec_driven_run(client, route_settings, monkeypatch, path):
    # nginx sends /spec-driven/ to the lab backend (the lab run's cancel route), and a run asks for no sign-in.
    from tests.utilities.web_modeling_editor.backend.spec_driven.test_spec_driven_router import _build_project_body

    monkeypatch.delenv("BESSER_DEMO_TOKEN", raising=False)
    body = _build_project_body(provider="sponsored", llm_model="gpt-5.6-terra")
    body.pop("api_key", None)

    # Off: as before, the first gate refuses an unauthorised sponsored run.
    assert client.post(path, json=body).status_code == 403

    route_settings.lab_enabled = True

    response = client.post(path, json=body)
    assert response.status_code == 404 and response.json() == {"detail": "Not Found"}


def test_the_lab_state_folder_comes_from_the_settings(route_settings, tmp_path):
    route_settings.lab_enabled = True
    route_settings.lab_workspace_root = str(tmp_path / "work")
    route_settings.lab_state_dir = str(tmp_path / "state")

    service = create_lab_service(route_settings)

    assert service.workspace.state_dir == str(tmp_path / "state")
    assert create_lab_service(route_settings.model_copy(update={"lab_state_dir": None})).workspace.state_dir == (
        os.path.join(str(tmp_path / "work"), ".lab"))


def test_the_last_run_is_kept_in_the_state_folder(tmp_path):
    # Outside the working copy, the lab agent's shell cannot rewrite the checks the state shows.
    root = tmp_path / "work2"
    path = root / "besser"
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "development")
    (path / "README.md").write_text("besser\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    workspace = LabWorkspace(str(root), repos=[RepoSpec("besser", str(path), "development", "o", "besser")],
                             state_dir=str(tmp_path / "state2"))
    service = LabService(workspace)
    change = workspace.start_change(user="octocat", title="Fix", kind=ChangeKind.BUGFIX)

    service.remember_run(runner_module.lab_done(SessionOutcome(
        step="build", change=change,
        build=BuildOutcome(loop=LoopResult("completed"), changes=[], checks=[], summary="nothing"))))

    assert (tmp_path / "state2" / "changes" / change.id / "last_run.json").is_file()
    assert not (root / ".lab").exists()
    assert service.state("octocat").change.summary == "nothing"
