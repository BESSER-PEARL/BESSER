"""The BESSER Lab routes over HTTP, on the lab service of a real lab layout under ``tmp_path``.

Runs use the fake session of ``test_lab_runner.py`` (each outcome played exactly), except the issue test, which
drives the real ``LabSession`` with a scripted model and a fake GitHub reader. The gate of every route (404 with
the lab off, 401, 403, 503) and the route table are in ``test_lab_contract.py``.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace

import httpx
from types import SimpleNamespace

import pytest

from besser.utilities.web_modeling_editor.backend.routers import lab_router
from besser.spec_driven_agent.lab.github_tools import GitHubReadError
from fastapi import FastAPI

import besser
from besser.spec_driven_agent.lab.contracts import ChangeKind
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.web_modeling_editor.backend import backend
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models.lab import LabPreviewLog, LabPreviewView
from besser.utilities.web_modeling_editor.backend.services.lab import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.lab.audit import AuditEntry
from besser.utilities.web_modeling_editor.backend.services.lab.preview import read_log_tail
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    PullRequestError,
    PullRequestInputError,
    PullRequestSignInError,
)
from tests.spec_driven_agent.lab.conftest import (  # noqa: F401
    ScriptedClient,
    end_turn,
    lab_layout,
    lab_layout_template,
)
from tests.spec_driven_agent.lab.test_session import FakeReader, issue_data
from tests.utilities.web_modeling_editor.backend.lab.test_lab_contract import ROUTES as CONTRACT_ROUTES
from tests.utilities.web_modeling_editor.backend.lab.test_lab_lifecycle import quiet_lifespan  # noqa: F401
from tests.utilities.web_modeling_editor.backend.lab.test_lab_runner import (  # noqa: F401
    lab_settings,
    plan_outcome,
    session_script,
)
from tests.utilities.web_modeling_editor.backend.lab.test_lab_service import ADMIN, lab_env, make_service  # noqa: F401
from tests.utilities.web_modeling_editor.backend.lab.test_lab_service import user as lab_user

LOGINS = ("jordi", "ana", "ben", ADMIN)
GITHUB_TOKEN = "gho_" + "Q7w8E9r0" * 4 + "Tt1Y"


def session(login: str) -> dict[str, str]:
    return {"X-GitHub-Session": f"sess-{login}"}


@pytest.fixture
def signed_in(app_state):
    """Every allowed login and mallory (not allowed) signed in."""
    for login in (*LOGINS, "mallory"):
        app_state.github_sessions.user_tokens.set(f"sess-{login}", {"access_token": f"gho_{login}", "username": login})
    return app_state


@pytest.fixture
def lab(lab_env, lab_settings, signed_in):  # noqa: F811
    """The lab service on ``app.state.lab``."""
    service = make_service(lab_env.workspace(), lab_settings)
    app.state.lab = service
    yield service
    del app.state.lab


def post_message(client, change_id: str, login: str, text: str = "Add a note element", **fields):
    return client.post("/besser_api/lab/message", json={"message": text, "changeId": change_id, **fields},
                       headers=session(login))


def sse_events(body: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


# -- The OpenAPI schema and a lab that cannot start ---------------------------------------------------------------

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
    # From the checkout under test: ``python -c`` imports from its working folder first, not an installed copy.
    checkout = os.path.dirname(os.path.dirname(os.path.abspath(besser.__file__)))
    result = subprocess.run([sys.executable, "-c", _OPENAPI_OF_A_FRESH_APP], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=300, cwd=checkout)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_openapi_schema_is_the_same_with_the_lab_on_or_off():
    """The editor commits one export and CI compares it with a lab-off app, so the lab routes are always in it."""
    off, on = _fresh_app(None), _fresh_app("true")

    assert json.dumps(off["openapi"], indent=2) == json.dumps(on["openapi"], indent=2)
    assert {path for path in off["openapi"]["paths"] if path.startswith("/besser_api/lab/")} == {
        path for _, path, *_ in CONTRACT_ROUTES}
    # A body that is not JSON at all fails before the lab gate runs: 422 rather than 404, accepted.
    assert off["broken_json"] == on["broken_json"] == 422


def test_a_lab_that_could_not_start_answers_503_with_the_reason(client, lab_settings, signed_in):  # noqa: F811
    app.state.lab = None
    app.state.lab_unavailable = "The base clones are not ready: run scripts/lab/setup_workspace.sh."
    try:
        response = client.get("/besser_api/lab/home", headers=session("ana"))
    finally:
        del app.state.lab, app.state.lab_unavailable

    assert response.status_code == 503
    assert response.json()["detail"] == ("BESSER Bootstrapped cannot start: The base clones are not ready: run "
                                         "scripts/lab/setup_workspace.sh.")


def test_a_lab_task_that_failed_does_not_stop_the_shutdown(lab_env, quiet_lifespan, monkeypatch, caplog):  # noqa: F811
    monkeypatch.setenv("BESSER_LAB_ENABLED", "true")
    stopped = []

    class FailingLab:
        preview = None

        async def refresh_loop(self):
            raise RuntimeError("the refresh died")

        def shutdown(self):
            stopped.append(True)

    monkeypatch.setattr(backend, "create_lab_service", lambda settings: FailingLab())

    async def main() -> None:
        async with backend.lifespan(FastAPI()):
            await asyncio.sleep(0.01)  # the task has died

    asyncio.run(main())

    assert stopped == [True]
    assert "lab-base-refresh had failed" in caplog.text and "the refresh died" in caplog.text


# -- Changes ---------------------------------------------------------------------------------------------------

def test_home_new_change_detail_and_list(client, lab):
    created = client.post("/besser_api/lab/change", headers=session("ana"))
    assert created.status_code == 200, created.text
    change_id = created.json()["changeId"]
    assert (created.json()["status"], created.json()["owner"]) == ("draft", "ana")

    home = client.get("/besser_api/lab/home", headers=session(ADMIN)).json()
    assert (home["user"], home["runs"]["cap"]) == ({"login": ADMIN, "isAdmin": True}, 2)
    assert client.get("/besser_api/lab/change", params={"changeId": change_id},
                      headers=session("ben")).json()["changeId"] == change_id
    assert [c["changeId"] for c in client.get("/besser_api/lab/changes", headers=session("ben")).json()["changes"]] \
        == [change_id]


def test_an_unknown_change_is_404_with_the_refusal_shape(client, lab):
    response = client.get("/besser_api/lab/change", params={"changeId": "0123456789ab"}, headers=session("ana"))

    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "not_found", "message": "There is no change 0123456789ab."}}


def test_drop_by_a_non_owner_is_403_and_by_the_owner_or_an_admin_allowed(client, lab):
    mine, other = (lab.workspace.create_draft(owner="ana") for _ in range(2))

    refused = client.post("/besser_api/lab/change/drop", json={"changeId": mine.id}, headers=session("ben"))
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "not_owner"
    assert "@ana" in refused.json()["detail"]["message"]

    by_admin = client.post("/besser_api/lab/change/drop", json={"changeId": other.id}, headers=session(ADMIN))
    assert by_admin.status_code == 200 and [c["changeId"] for c in by_admin.json()["changes"]] == [mine.id]
    by_owner = client.post("/besser_api/lab/change/drop", json={"changeId": mine.id}, headers=session("ana"))
    assert by_owner.status_code == 200 and by_owner.json() == {"changes": []}


def test_closing_a_change_another_viewer_closed_answers_its_detail(client, lab):
    change = lab.workspace.create_draft(owner="ana")
    lab.workspace.save_change(replace(change, title="Add a note", status="closed", outcome="merged",
                                      closed_at=time.time()))

    response = client.post("/besser_api/lab/change/close", json={"changeId": change.id}, headers=session("ben"))

    assert response.status_code == 200, response.text
    assert (response.json()["status"], response.json()["phase"]) == ("closed", "merged")


def test_a_draft_cannot_be_updated_from_its_base(client, lab):
    draft = lab.workspace.create_draft(owner="ana")

    response = client.post("/besser_api/lab/change/update-base", json={"changeId": draft.id, "repo": "besser"},
                           headers=session("ana"))

    assert response.status_code == 409 and response.json()["detail"]["code"] == "state"


def test_the_diff_of_a_closed_change_is_refused_and_a_path_needs_its_repository(client, lab):
    opened = lab.workspace.open_change(lab.workspace.create_draft(owner="ana"), title="Note",
                                       kind=ChangeKind.EDITOR_ELEMENT)

    assert client.get("/besser_api/lab/diff", params={"changeId": opened.id},
                      headers=session("ana")).json()["changeId"] == opened.id
    no_repo = client.get("/besser_api/lab/diff", params={"changeId": opened.id, "path": "README.md"},
                         headers=session("ana"))
    assert no_repo.status_code == 422

    lab.workspace.close(lab.workspace.get_change(opened.id), outcome="merged")
    closed = client.get("/besser_api/lab/diff", params={"changeId": opened.id}, headers=session("ana"))
    assert closed.status_code == 409 and closed.json()["detail"]["code"] == "state"
    assert "pull request" in closed.json()["detail"]["message"]


# -- Pull requests -----------------------------------------------------------------------------------------------

def test_the_issue_picker_lists_open_issues_of_one_repository_with_the_users_token(client, lab, monkeypatch):
    asked = []

    class FakeReader:
        def __init__(self, github, repos, loop, *, login):
            self.token, self.login = github.token, login

        def search(self, query, repo, kind, state, limit):
            asked.append((self.token, self.login, query, repo, kind, state, limit))
            if query == "boom":
                raise GitHubReadError("signed_out")
            if query == "down":
                raise GitHubReadError("failed", detail="GitHub answered HTTP 502")
            return [{"repo": repo, "number": 835, "kind": "issue", "title": "Improve the readme", "state": "open",
                     "url": "https://github.com/BESSER-PEARL/BESSER/issues/835", "author": "jcabot",
                     "updated_at": "2026-11-25"}]

    monkeypatch.setattr(lab_router, "LabGitHubReader", FakeReader)
    monkeypatch.setattr(lab_router, "GitHubService", lambda token: SimpleNamespace(token=token))

    listed = client.get("/besser_api/lab/issues", params={"repo": "besser"}, headers=session("ana"))
    assert listed.status_code == 200, listed.text
    assert listed.json() == {"issues": [{"repo": "besser", "number": 835, "title": "Improve the readme",
                                         "author": "jcabot", "updatedAt": "2026-11-25",
                                         "url": "https://github.com/BESSER-PEARL/BESSER/issues/835"}]}
    client.get("/besser_api/lab/issues", params={"repo": "frontend", "q": "  readme link "}, headers=session("ana"))
    # Open issues only, the most recently updated when nothing is searched for, read as the signed-in user.
    assert [call[2:] for call in asked] == [("sort:updated-desc", "besser", "issue", "open", 30),
                                            ("readme link", "frontend", "issue", "open", 30)]
    assert {call[1] for call in asked} == {"ana"} and all(call[0] for call in asked)

    assert client.get("/besser_api/lab/issues", params={"repo": "besser", "q": "boom"},
                      headers=session("ana")).status_code == 401
    assert client.get("/besser_api/lab/issues", params={"repo": "besser", "q": "down"},
                      headers=session("ana")).status_code == 502
    assert client.get("/besser_api/lab/issues", params={"repo": "not-a-repo"}, headers=session("ana")).status_code == 422
    assert client.get("/besser_api/lab/issues", params={"repo": "besser"}).status_code == 401
    assert client.get("/besser_api/lab/issues", params={"repo": "besser"}, headers=session("mallory")).status_code == 403


def test_closing_an_issue_the_change_did_not_start_from_is_400(client, lab, monkeypatch):
    change = lab.workspace.open_change(lab.workspace.create_draft(owner="ana"), title="Note",
                                       kind=ChangeKind.EDITOR_ELEMENT)
    lab.workspace.save_change(replace(change, issues=[{"repo": "besser", "number": 744, "title": "t",
                                                       "closes": False}]))
    pushed = []

    async def open_pull_requests(user, change_id, *, title, body, closes_issues):
        # The service's rule (pull_requests._closing_issues), raised before anything is pushed.
        if any((ref.repo, ref.number) != ("besser", 744) for ref in closes_issues):
            raise PullRequestInputError("frontend#12 is not an issue of this change: its pull requests close only "
                                        "the issues the change started from.")
        pushed.append([(ref.repo, ref.number) for ref in closes_issues])
        return [], "No check ran."

    monkeypatch.setattr(lab, "open_pull_requests", open_pull_requests)

    refused = client.post("/besser_api/lab/pull-request", headers=session("ana"), json={
        "changeId": change.id, "closesIssues": [{"repo": "besser", "number": 744}, {"repo": "frontend", "number": 12}]})
    assert refused.status_code == 400 and refused.json()["detail"]["code"] == "bad_request"
    assert "frontend#12 is not an issue of this change" in refused.json()["detail"]["message"]
    assert pushed == []

    allowed = client.post("/besser_api/lab/pull-request", headers=session("ana"), json={
        "changeId": change.id, "closesIssues": [{"repo": "besser", "number": 744}]})
    assert allowed.status_code == 200 and allowed.json() == {"pullRequests": [], "checksNote": "No check ran."}
    assert pushed == [[("besser", 744)]]


@pytest.mark.parametrize("error, status, detail", [
    (PullRequestError("This change has no modifications to open a pull request for."), 409,
     {"code": "state", "message": "This change has no modifications to open a pull request for."}),
    (PullRequestSignInError("GitHub refused your sign-in: sign in again."), 401,
     "GitHub refused your sign-in: sign in again."),
])
def test_pull_request_failures_answer_409_state_or_401(client, lab, monkeypatch, error, status, detail):
    change = lab.workspace.create_draft(owner="ana")

    async def open_pull_requests(user, change_id, *, title, body, closes_issues):
        raise error

    monkeypatch.setattr(lab, "open_pull_requests", open_pull_requests)

    response = client.post("/besser_api/lab/pull-request", json={"changeId": change.id}, headers=session("ana"))

    assert (response.status_code, response.json()["detail"]) == (status, detail)


# -- Runs (D4 tests 5 and 6) -------------------------------------------------------------------------------------

async def _until(event: threading.Event, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not event.is_set():
        assert time.monotonic() < deadline, "the run did not start"
        await asyncio.sleep(0.01)


def test_two_runs_on_two_changes_stream_at_once_and_the_lab_refuses_a_third_and_a_busy_change(
        lab, session_script):  # noqa: F811
    changes = [lab.workspace.create_draft(owner=owner) for owner in ("jordi", "ana", "ben")]
    started = {change.id: threading.Event() for change in changes}
    release = threading.Event()

    def script(fake, text):
        change_id = fake.handled[-1]["change_id"]
        fake.callbacks.on_phase("plan", "Planning the change")
        started[change_id].set()
        release.wait(30)
        return plan_outcome(next(change for change in changes if change.id == change_id))

    session_script(script)

    async def main():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60) as ac:
            def post(change, login):
                return ac.post("/besser_api/lab/message", json={"message": "go", "changeId": change.id},
                               headers=session(login))

            first = asyncio.create_task(post(changes[0], "ana"))
            await _until(started[changes[0].id])
            second = asyncio.create_task(post(changes[1], "ben"))
            await _until(started[changes[1].id])
            both_running = not first.done() and not second.done()
            third = await post(changes[2], "jordi")
            busy = await post(changes[0], "jordi")
            home = (await ac.get("/besser_api/lab/home", headers=session("jordi"))).json()
            release.set()
            return both_running, third, busy, home, await first, await second

    both_running, third, busy, home, first, second = asyncio.run(main())

    assert both_running
    assert third.status_code == 429 and third.json()["detail"]["code"] == "runs_full"
    assert "2 of 2" in third.json()["detail"]["message"] and "@ana" in third.json()["detail"]["message"]
    assert busy.status_code == 409 and busy.json()["detail"]["code"] == "change_busy"
    assert busy.json()["detail"]["message"].startswith("@ana's plan of this change is in progress")
    assert sorted((run["login"], run["step"]) for run in home["runs"]["inProgress"]) == [("ana", "plan"),
                                                                                        ("ben", "plan")]
    for response in (first, second):
        assert response.status_code == 200 and sse_events(response.text)[-1]["event"] == "done"
    assert lab.home(lab_user("jordi")).runs.inProgress == []


def test_a_spent_month_answers_409_user_cap(client, lab, session_script):  # noqa: F811
    lab.audit.append(AuditEntry(at=time.time(), login="ana", action="run", cost_usd=74.9))
    change = lab.workspace.create_draft(owner="ana")

    response = post_message(client, change.id, "ana")

    assert response.status_code == 409 and response.json()["detail"]["code"] == "user_cap"
    assert session_script.sessions == []
    assert client.post("/besser_api/lab/change", headers=session("ana")).json()["detail"]["code"] == "user_cap"


def test_a_message_to_a_closed_or_unknown_change_is_refused(client, lab, session_script):  # noqa: F811
    assert post_message(client, "0123456789ab", "ana").status_code == 404
    change = lab.workspace.create_draft(owner="ana")
    lab.workspace.save_change(replace(change, status="closed", outcome="closed", closed_at=time.time()))

    response = post_message(client, change.id, "ana")

    assert response.status_code == 409 and response.json()["detail"]["code"] == "state"


def test_a_run_that_could_not_start_is_given_back(client, lab, spec_driven_runtime, monkeypatch):
    change = lab.workspace.create_draft(owner="ana")

    async def store_failed(*args, **kwargs):
        raise RuntimeError("the run store is full")

    monkeypatch.setattr(spec_driven_runtime.durable_runs, "start", store_failed)

    with pytest.raises(RuntimeError, match="the run store is full"):
        post_message(client, change.id, "ana")

    [run] = lab.change_detail(change.id).thread
    assert (run.status, run.error.message) == ("failed", "The run could not be started.")
    assert lab.home(lab_user("ana")).runs.inProgress == [] and lab.budget.meters("ana").user.reservedUsd == 0


def test_the_thread_after_a_run_holds_the_message_running_and_done(client, lab, session_script):  # noqa: F811
    change = lab.workspace.create_draft(owner="jordi")
    session_script(lambda fake, text: plan_outcome(change))

    response = post_message(client, change.id, "ana")
    run_id = response.headers["X-BESSER-Run-Id"]
    detail = client.get("/besser_api/lab/change", params={"changeId": change.id}, headers=session("ben")).json()

    assert sse_events(response.text)[-1]["event"] == "done"
    sent, run = detail["thread"]
    assert (sent["type"], sent["login"], sent["text"]) == ("message", "ana", "Add a note element")
    assert (run["type"], run["runId"], run["status"], run["login"]) == ("run", run_id, "done", "ana")
    assert run["lab"]["step"] == "plan" and run["lab"]["plan"]["title"] == "Add a note element"
    assert [record["runId"] for record in detail["runs"]] == [run_id]


def test_a_message_with_an_issue_reaches_the_plans_first_message(client, lab, monkeypatch):
    """K7: the issue is read before the plan loop (fake reader) and recorded on the change."""
    reader = FakeReader({("besser", 744): issue_data(744, "Auto-fix loop ignores ruff", "The loop skips ruff.")})
    model = ScriptedClient([end_turn("Which part of the loop?"), end_turn("Which part of the loop?")])
    readers = []

    def make_reader(github, repos, loop, *, login=""):
        readers.append((github.access_token, sorted(repos), login))
        return reader

    monkeypatch.setattr(runner_module, "LabGitHubReader", make_reader)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda provider, **kw: model)
    change = lab.workspace.create_draft(owner="ana")

    response = post_message(client, change.id, "ana", "Start from this issue", issue={"repo": "besser", "number": 744})

    assert sse_events(response.text)[-1]["event"] == "done", response.text
    first = model.calls[0][1][0]["content"]
    assert "Auto-fix loop ignores ruff" in first and "The loop skips ruff." in first
    assert first.index("The loop skips ruff.") < first.index("Start from this issue")
    assert reader.calls == [("besser", 744)]
    assert readers == [("gho_ana", sorted(lab.workspace.base_repos), "ana")]
    assert [(i["repo"], i["number"]) for i in lab.workspace.get_change(change.id).issues] == [("besser", 744)]


# -- Preview restart and logs (D1 test 11) ---------------------------------------------------------------------

class FakePool:
    """A preview pool of one slot whose restarts are recorded; logs are read (and redacted) as the pool does."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.restarts: list[tuple[int, object]] = []

    def view(self) -> LabPreviewView:
        return LabPreviewView(enabled=True, detail="", slots=[])

    def restart(self, slot, service=None) -> LabPreviewView:
        if slot != 0:
            raise KeyError(slot)
        self.restarts.append((slot, service))
        return self.view()

    def log_tail(self, slot, service, lines) -> LabPreviewLog:
        if slot != 0:
            raise KeyError(slot)
        tail, truncated = read_log_tail(os.path.join(self.log_dir, f"{service}.log"), lines)
        return LabPreviewLog(slot=slot, service=service, lines=tail, truncated=truncated)

    def after_build(self, change_id):
        pass


@pytest.fixture
def pool(lab, tmp_path):
    lab.preview = FakePool(str(tmp_path / "preview-logs"))
    os.makedirs(lab.preview.log_dir)
    return lab.preview


def test_restart_restarts_the_named_service_of_the_slot(client, pool):
    response = client.post("/besser_api/lab/preview/restart", json={"slot": 0, "service": "modeling_agent"},
                           headers=session("ana"))
    every = client.post("/besser_api/lab/preview/restart", json={"slot": 0}, headers=session("ana"))
    unknown = client.post("/besser_api/lab/preview/restart", json={"slot": 3}, headers=session("ana"))

    assert response.status_code == every.status_code == 200
    assert response.json() == {"enabled": True, "detail": "", "slots": []}
    assert pool.restarts == [(0, "modeling_agent"), (0, None)]
    assert unknown.status_code == 404 and unknown.json()["detail"]["code"] == "not_found"


def test_restart_is_gated_and_refused_with_the_preview_off(client, lab, route_settings):
    body = {"slot": 0, "service": "backend"}
    assert client.post("/besser_api/lab/preview/restart", json=body).status_code == 401
    assert client.post("/besser_api/lab/preview/restart", json=body, headers=session("mallory")).status_code == 403

    off = client.post("/besser_api/lab/preview/restart", json=body, headers=session("ana"))
    assert off.status_code == 409 and off.json()["detail"]["code"] == "state"
    assert "BESSER_LAB_PREVIEW_ENABLED" in off.json()["detail"]["message"]

    route_settings.lab_enabled = False
    assert client.post("/besser_api/lab/preview/restart", json=body, headers=session("ana")).status_code == 404


def test_logs_are_redacted_bounded_and_of_known_services(client, pool):
    with open(os.path.join(pool.log_dir, "backend.log"), "w", encoding="utf-8") as handle:
        handle.write("".join(f"line {i}\n" for i in range(10)) + f"token {GITHUB_TOKEN}\n")

    def logs(service: str, **params):
        return client.get(f"/besser_api/lab/preview/logs/{service}", params={"slot": 0, **params},
                          headers=session("ana"))

    response = logs("backend", lines=3)
    assert response.status_code == 200
    body = response.json()
    assert (body["slot"], body["service"], body["truncated"]) == (0, "backend", True)
    assert body["lines"][:2] == ["line 8", "line 9"] and GITHUB_TOKEN not in body["lines"][2]
    assert len(logs("backend").json()["lines"]) == 11  # 200 by default
    assert logs("database").status_code == 422
    assert logs("backend", lines=0).status_code == 422 and logs("backend", lines=2001).status_code == 422
    assert logs("backend", lines=2000).status_code == 200


# -- Team memory (K7) ---------------------------------------------------------------------------------------------

def test_memory_routes_with_the_memory_off_answer_409_state(client, lab_env, lab_settings, signed_in):  # noqa: F811
    # A state folder inside the workspace root: code the agent runs could plant a note there.
    workspace = LabWorkspace(lab_env.root, lab_env.specs(), state_dir=os.path.join(lab_env.root, "state"),
                             git_root=lab_env.git_root)
    app.state.lab = make_service(workspace, lab_settings)
    note = {"kind": "decision", "title": "Keep notes short", "body": "One idea per note."}
    try:
        listed = client.get("/besser_api/lab/memory", headers=session("ana"))
        added = client.post("/besser_api/lab/memory", json=note, headers=session("ana"))
        mallory = client.post("/besser_api/lab/memory", json=note, headers=session("mallory"))
        lab_settings.lab_enabled = False
        off = client.get("/besser_api/lab/memory", headers=session("ana"))
    finally:
        del app.state.lab

    assert listed.status_code == 200 and listed.json()["enabled"] is False
    assert added.status_code == 409 and added.json()["detail"]["code"] == "state"
    assert "BESSER_LAB_STATE_DIR" in added.json()["detail"]["message"]
    assert mallory.status_code == 403 and off.status_code == 404


def test_memory_notes_round_trip(client, lab):
    note = {"kind": "convention", "title": "Docs first", "body": "Write the docs with the code.",
            "docsTarget": "CLAUDE.md"}
    added = client.post("/besser_api/lab/memory", json=note, headers=session("ana"))
    assert added.status_code == 200, added.text
    note_id = added.json()["id"]

    edited = client.put(f"/besser_api/lab/memory/{note_id}", json={**note, "title": "Docs with the code"},
                        headers=session("ben"))
    assert edited.status_code == 200 and edited.json()["title"] == "Docs with the code"
    assert client.delete(f"/besser_api/lab/memory/{note_id}", headers=session("ben")).status_code == 204
    assert client.get("/besser_api/lab/memory", headers=session("ben")).json()["notes"] == []
    gone = client.delete(f"/besser_api/lab/memory/{note_id}", headers=session("ben"))
    assert gone.status_code == 404 and gone.json()["detail"]["code"] == "not_found"
    dismissed = client.post("/besser_api/lab/memory/proposals/dismiss", json={"changeId": "0123456789ab", "id": "p"},
                            headers=session("ben"))
    assert dismissed.status_code == 404


def test_settings_and_audit(client, lab):
    settings = client.get("/besser_api/lab/settings", headers=session("ana")).json()
    assert settings["admins"] == [ADMIN] and settings["limits"]["perRunUsd"] == 2.5

    client.post("/besser_api/lab/change", headers=session("ana"))
    audit = client.get("/besser_api/lab/audit", headers=session("ana")).json()
    assert [entry["action"] for entry in audit["entries"]] == ["change_created"]
    assert client.get("/besser_api/lab/audit", params={"month": "2026-13"}, headers=session("ana")).status_code == 422
