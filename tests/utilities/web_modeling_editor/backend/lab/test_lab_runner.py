"""A BESSER Lab run as SSE: the ticket, the thread, the session's wiring, its callbacks, outcome and errors.

The session is a fake with ``LabSession``'s API, so each outcome and error can be played exactly; everything
else is real: the runner, the lab service on a lab layout under ``tmp_path`` (base clones, change worktrees),
its thread, ledgers and budget, and the runtime's run ids. ``test_lab_router.py`` drives runs through the route
and the real ``LabSession`` with a scripted model.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic import SecretStr

from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    BuildOutcome,
    BuildReport,
    ChangeKind,
    CheckResult,
    ChecklistItem,
    FileChange,
    IssueRef,
    LoopResult,
    NoteDraft,
    Plan,
    RepoChange,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabCheckRun,
    LabMessageRequest,
    LabPullRequestStatus,
    LabReviewResponse,
)
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.lab import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.lab import service as service_module
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.github_reader import LabGitHubReader
from besser.utilities.web_modeling_editor.backend.services.lab.model_proposal import proposal_from_plan
from besser.utilities.web_modeling_editor.backend.services.lab.runner import LabRunner
from tests.spec_driven_agent.lab.conftest import lab_layout, lab_layout_template, write  # noqa: F401
from tests.utilities.web_modeling_editor.backend.lab.test_lab_service import ALLOWED, ADMIN, lab_env, make_service  # noqa: F401

# No known key prefix: the stream's generic token patterns would redact an ``sk-ant-`` key anyway.
LAB_KEY = "labkey0123456789abcdefXYZ"
LOGIN = "ana"
USER = LabUser(login=LOGIN, session_id="sess-ana", access_token="gho_ana_token")
MODEL_SPEC = {
    "classes": [{"name": "Note", "attributes": [{"name": "text", "type": "str"}], "is_abstract": False}],
    "enumerations": [], "associations": [], "generalizations": [],
}
FAILING_TAIL = "FAILED tests/test_note.py::test_note - AssertionError: 1 != 2"


def plan_outcome(change) -> SessionOutcome:
    plan = Plan(kind=ChangeKind.METAMODEL, title="Add a note element", summary="Notes on class diagrams.",
                steps=["Add the Note metaclass", "Convert it both ways"], repos=["besser", "frontend"],
                questions=["Should notes have a colour?"], model_spec=MODEL_SPEC)
    return SessionOutcome(step="plan", change=change, plan=plan, cost_usd=0.05)


def build_outcome(change, stop_reason: str = "completed") -> SessionOutcome:
    build = BuildOutcome(
        loop=LoopResult(stop_reason=stop_reason, final_text="Added the Note metaclass."),
        changes=[
            RepoChange("frontend", "lab/ana/note", "develop",
                       (FileChange("packages/editor/src/main/packages/note.ts", "added"),)),
            RepoChange("besser", "lab/ana/note", "development",
                       (FileChange("besser/BUML/metamodel/structural/note.py", "added"),
                        FileChange("tests/BUML/metamodel/test_note.py", "added"))),
        ],
        checks=[CheckResult("besser:tests", "besser", "Tests of the touched areas", passed=True, summary="3 passed",
                            duration_seconds=12.5),
                CheckResult("besser:docs", "besser", "Docs", passed=False, skipped=True, summary="sphinx missing")],
        checklist=[ChecklistItem("metamodel.tests", "Tests under tests/BUML/metamodel/", True),
                   ChecklistItem("metamodel.docs", "Docs in docs/source/buml_language", None)],
        summary="besser: 2 files added",
        report=BuildReport(summary="Added Note.", deviations=("No colour",), not_done=("Docs",)),
    )
    drafts = [NoteDraft(kind="decision", title="Notes have no colour", body="Decided in review.",
                        repos=("besser",), change_kinds=("metamodel",), why="It came up twice.")]
    return SessionOutcome(step="build", change=change, build=build, cost_usd=0.4, stop_reason=stop_reason,
                          note_proposals=drafts)


class FakeClient:
    """Stands in for the provider client: only its cumulative cost is read."""

    def __init__(self):
        self.usage = SimpleNamespace(estimated_cost=0.0)


class FakeSession:
    """``LabSession``'s API; ``script(session, message)`` plays the run in the worker thread."""

    def __init__(self, workspace, *, script, record, **wiring):
        self.workspace = workspace
        self.wiring = wiring
        self.client, self.limits, self.callbacks = wiring["client"], wiring["limits"], wiring["callbacks"]
        self._script = script
        record.append(self)
        self.handled = []

    def handle(self, message, *, change_id, login, action="auto", model_spec=None, issues=()):
        self.handled.append({"message": message, "change_id": change_id, "login": login, "action": action,
                             "model_spec": model_spec, "issues": list(issues)})
        return self._script(self, message)


@pytest.fixture
def lab_settings(route_settings):
    route_settings.lab_enabled = True
    route_settings.lab_allowed_logins = ALLOWED
    route_settings.lab_admin_logins = ADMIN
    route_settings.lab_llm_provider = "anthropic"
    route_settings.lab_llm_model = "claude-test"
    route_settings.lab_plan_llm_model = None
    route_settings.lab_llm_api_key = SecretStr(LAB_KEY)
    route_settings.lab_max_cost_usd = 2.5
    route_settings.lab_max_runtime_seconds = 600
    route_settings.lab_max_concurrent_runs = 2
    route_settings.lab_user_monthly_cap_usd = 75.0
    route_settings.lab_monthly_cap_usd = 300.0
    route_settings.llm_cost_emitter_interval_seconds = 0.02
    return route_settings


@pytest.fixture
def lab(lab_env, lab_settings):  # noqa: F811 - lab_env is the imported fixture
    """The lab service on a lab layout, wired as ``create_lab_service`` wires it."""
    return make_service(lab_env.workspace(), lab_settings)


@pytest.fixture
def draft(lab):
    return lab.workspace.create_draft(owner="jordi")


@pytest.fixture
def opened(lab):
    """An open change of jordi's with its worktrees."""
    return lab.workspace.open_change(lab.workspace.create_draft(owner="jordi"), title="Add a note element",
                                     kind=ChangeKind.METAMODEL)


@pytest.fixture
def session_script(monkeypatch):
    """Install ``script`` as the session of the next runs; returns the sessions created and the clients."""
    sessions: list = []
    state = {"script": None}
    clients: list = []

    def fake_client(provider, *, api_key=None, model=None, base_url=None):
        clients.append({"provider": provider, "api_key": api_key, "model": model, "base_url": base_url})
        return FakeClient()

    monkeypatch.setattr(runner_module, "create_llm_client", fake_client)
    monkeypatch.setattr(runner_module, "LabSession", lambda workspace, **kw: FakeSession(
        workspace, **kw, script=state["script"], record=sessions))

    def install(script):
        state["script"] = script
        return sessions

    install.sessions = sessions
    install.clients = clients
    return install


def parse(frame: bytes) -> dict:
    text = frame.decode("utf-8")
    assert text.endswith("\n\n")
    data = next(line for line in text.split("\n") if line.startswith("data: "))
    return json.loads(data[len("data: "):])


async def start_run(lab, runtime, settings, request, user=USER):
    """What the route does: reserve the run id, let the service start the run, build the runner."""
    run_id = uuid.uuid4().hex
    cancel_event = await runtime.active_runs.reserve(run_id)
    ticket = lab.begin_run(request.changeId, run_id, user.login, own_key=request.llm is not None)
    return LabRunner(request, user=user, lab=lab, runtime=runtime, settings=settings, ticket=ticket,
                     cancel_event=cancel_event)


def run_lab(lab, runtime, settings, request) -> tuple[str, list[bytes]]:
    """Run one message to its end; the run id and the raw frames."""
    async def go():
        runner = await start_run(lab, runtime, settings, request)
        return runner.run_id, [frame async for frame in runner.stream()]

    return asyncio.run(go())


def events(frames) -> list[dict]:
    return [parse(frame) for frame in frames]


def message(change, text: str = "Add a note element to class diagrams", **fields) -> LabMessageRequest:
    return LabMessageRequest(message=text, changeId=change.id, **fields)


def assert_free(lab, change) -> None:
    """The change and the run slot were given back: another run may start on the change."""
    ticket = lab.begin_run(change.id, "f" * 32, "ben")
    lab.end_run(ticket, status="cancelled", record=service_module.LabRunRecord(
        runId="f" * 32, step="plan", login="ben", model="m", usd=0.0, turns=0, elapsedSeconds=0.0,
        stopReason="cancelled", startedAt=time.time()), done=None, error=None)


# -- A plan, the thread and the ledgers ---------------------------------------------------------------------

def test_a_plan_streams_start_progress_and_a_done_event_with_the_plan(
        lab, draft, spec_driven_runtime, lab_settings, session_script):
    def script(session, text):
        session.callbacks.on_phase("plan", "Planning the change")
        session.callbacks.on_progress(1, "read_file", "executing", "CLAUDE.md")
        session.callbacks.on_text("I read the metamodel.")
        session.callbacks.on_progress(2, "submit_plan", "executing", None)
        session.callbacks.on_text("Here is the plan.")
        return plan_outcome(draft)

    session_script(script)
    run_id, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft))
    every = events(frames)
    # The 20 ms cost emitter may add cost events anywhere on a loaded machine; the final one precedes done.
    assert every[-2]["event"] == "cost"
    got = [e for e in every[:-2] if e["event"] != "cost"] + every[-1:]

    assert [e["event"] for e in got] == ["start", "phase", "tool_call", "text", "tool_call", "text", "done"]
    # The run's cost cap is the one the service reserved for it.
    assert got[0] == {"event": "start", "runId": run_id, "provider": "anthropic", "llmModel": "claude-test",
                      "maxCost": 2.5, "maxRuntime": 600}
    assert got[1]["phase"] == "plan" and got[1]["message"] == "Planning the change"
    assert got[5]["delta"] == "\n\nHere is the plan."
    done = got[-1]
    assert (done["runId"], done["downloadUrl"], done["isZip"], done["mode"]) == (run_id, "", False, "modify")
    payload = done["lab"]
    assert (payload["step"], payload["changeId"], payload["title"]) == ("plan", draft.id, "Add a note element")
    assert payload["plan"]["steps"] == ["Add the Note metaclass", "Convert it both ways"]
    assert payload["modelProposal"]["diagramType"] == "ClassDiagram"
    assert payload["cost"]["runs"] == 1 and payload["noteProposals"] == [] and payload["report"] is None


def test_the_thread_after_a_run_holds_the_message_running_and_done(
        lab, draft, spec_driven_runtime, lab_settings, session_script):
    seen_while_running = []

    def script(session, text):
        seen_while_running.extend(lab.change_detail(draft.id).thread)
        return plan_outcome(draft)

    session_script(script)
    project = _project(draft.id)
    run_id, _ = run_lab(lab, spec_driven_runtime, lab_settings, message(draft, project=project))

    # While it runs: the message, then the run as ``running``.
    assert [(item.type, getattr(item, "status", None)) for item in seen_while_running] == [
        ("message", None), ("run", "running")]
    detail = lab.change_detail(draft.id)
    sent, run = detail.thread
    assert (sent.login, sent.text, sent.action) == (LOGIN, "Add a note element to class diagrams", "auto")
    assert sent.model["type"] == "ClassDiagram"  # the change's class diagram the message carried
    assert (run.runId, run.login, run.status, run.planVersion) == (run_id, LOGIN, "done", 1)
    assert run.lab.step == "plan" and run.lab.plan.title == "Add a note element" and run.error is None
    # The run record, the audit line and the settled budget.
    [record] = detail.runs
    assert (record.runId, record.step, record.login, record.model, record.stopReason) == (
        run_id, "plan", LOGIN, "claude-test", "completed")
    assert [(e.action, e.run_id) for e in lab.audit.entries(lab.audit.current_month()) if e.action == "run"] == [
        ("run", run_id)]
    assert lab.budget.meters(LOGIN).user.reservedUsd == 0
    assert_free(lab, draft)


def test_the_session_gets_the_labs_wiring(lab, draft, spec_driven_runtime, lab_settings, session_script):
    lab_settings.lab_plan_llm_model = "claude-plan"
    session_script(lambda session, text: plan_outcome(draft))

    run_id, _ = run_lab(lab, spec_driven_runtime, lab_settings,
                        message(draft, issue={"repo": "besser", "number": 744}))

    assert session_script.clients == [{"provider": "anthropic", "api_key": LAB_KEY, "model": "claude-test",
                                       "base_url": None}]
    [session] = session_script.sessions
    wiring = session.wiring
    assert session.workspace is lab.workspace
    assert (wiring["limits"].max_cost_usd, wiring["limits"].max_runtime_seconds) == (2.5, 600.0)
    assert wiring["environments"] is lab.environments and wiring["memory"] is lab.memory
    assert wiring["sandbox_policy"] is lab.sandbox_policy()
    # BESSER_LAB_PLAN_LLM_MODEL reaches the session, which uses it for the plan loop.
    assert wiring["plan_model"] == "claude-plan"
    reader = wiring["github_reader"]
    assert isinstance(reader, LabGitHubReader) and reader._login == LOGIN
    assert set(reader._repos) == set(lab.workspace.base_repos)
    assert session.handled == [{"message": "Add a note element to class diagrams", "change_id": draft.id,
                                "login": LOGIN, "action": "auto", "model_spec": None,
                                "issues": [IssueRef("besser", 744)]}]
    # A plan's record names the plan model.
    assert lab.change_detail(draft.id).runs[0].model == "claude-plan"


def test_the_runs_cost_cap_is_what_the_month_has_left(lab, draft, spec_driven_runtime, lab_settings, session_script):
    lab.audit.append(service_module.AuditEntry(at=time.time(), login=LOGIN, action="run", cost_usd=74.0))
    session_script(lambda session, text: plan_outcome(draft))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft))

    assert events(frames)[0]["maxCost"] == pytest.approx(1.0)
    assert session_script.sessions[0].limits.max_cost_usd == pytest.approx(1.0)


# -- A build ------------------------------------------------------------------------------------------------

def test_a_build_reports_files_checks_report_and_note_proposals_and_commits_as_the_user(
        lab, opened, spec_driven_runtime, lab_settings, session_script):
    tree = lab.workspace.change_repos(opened)[REPO_BESSER].path

    def script(session, text):
        write(os.path.join(tree, "besser", "note.py"), "NOTE = 1\n")
        return build_outcome(opened)

    session_script(script)
    run_id, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(opened, "go", action="build"))
    done = events(frames)[-1]

    assert done["event"] == "done" and done["changedFileCount"] == 3
    payload = done["lab"]
    assert payload["checks"][0]["durationSeconds"] == 12.5 and payload["checks"][1]["durationSeconds"] == 0.0
    assert payload["checklist"][1] == {"id": "metamodel.docs", "text": "Docs in docs/source/buml_language",
                                       "done": None}
    assert payload["report"] == {"summary": "Added Note.", "deviations": ["No colour"], "notDone": ["Docs"]}
    assert payload["noteProposals"] == [{
        "id": f"{run_id[:8]}-1", "kind": "decision", "title": "Notes have no colour", "body": "Decided in review.",
        "repos": ["besser"], "changeKinds": ["metamodel"], "docsTarget": "", "why": "It came up twice."}]
    assert payload["finalText"] == "Added the Note metaclass."
    # The build's work is committed, authored by the login that asked for it.
    assert lab.workspace.run_git(REPO_BESSER, "log", "-1", "--format=%an", change=opened).strip() == LOGIN
    assert lab.workspace.run_git(REPO_BESSER, "status", "--porcelain", change=opened).strip() == ""
    # The proposals wait for approval; the run's step is recorded.
    assert [p.id for p in lab._read_proposals(opened.id)] == [f"{run_id[:8]}-1"]
    assert lab.change_detail(opened.id).runs[0].step == "build"


def test_a_build_whose_work_cannot_be_committed_says_so(lab, opened, spec_driven_runtime, lab_settings,
                                                        session_script, monkeypatch):
    def refuse(change, message, *, login):
        raise LabWorkspaceError("besser: the merge from development still has conflict markers in README.md")

    monkeypatch.setattr(lab.workspace, "commit_all", refuse)
    session_script(lambda session, text: build_outcome(opened))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(opened, "go", action="build"))
    got = events(frames)

    assert got[-2]["event"] == "error" and got[-2]["code"] == "INCOMPLETE"
    assert "not committed" in got[-2]["message"] and "conflict markers in README.md" in got[-2]["message"]
    assert got[-1]["event"] == "done"


def test_fix_ci_message_carries_the_failing_log_tail_and_forces_build(
        lab, opened, spec_driven_runtime, lab_settings, session_script, monkeypatch):
    reads = []

    async def fetch_review(workspace, change, github, cache, *, fresh=False):
        reads.append((change.id, github.access_token, fresh))
        return _review(opened.id)

    monkeypatch.setattr(service_module.review, "fetch_review", fetch_review)
    session_script(lambda session, text: build_outcome(opened))

    async def go():
        runner = await start_run(lab, spec_driven_runtime, lab_settings,
                                 message(opened, "Please fix CI", action="plan",
                                         review={"task": "fix_ci", "repo": "besser"}))
        step = runner.ticket.step
        frames = [frame async for frame in runner.stream()]
        return step, frames

    step, frames = asyncio.run(go())

    assert step == "build"
    [handled] = session_script.sessions[0].handled
    assert handled["action"] == "build"
    assert handled["message"].startswith("Please fix CI")
    assert FAILING_TAIL in handled["message"] and 'untrusted="true"' in handled["message"]
    # Read fresh, with the user's own token.
    assert reads == [(opened.id, USER.access_token, True)]
    assert events(frames)[-1]["event"] == "done"
    assert lab.change_detail(opened.id).thread[0].action == "build"


def test_a_review_task_whose_pull_requests_cannot_be_read_fails_before_the_session(
        lab, opened, spec_driven_runtime, lab_settings, session_script, monkeypatch):
    async def fetch_review(workspace, change, github, cache, *, fresh=False):
        raise service_module.pull_requests.PullRequestSignInError("GitHub refused your sign-in: sign in again.")

    monkeypatch.setattr(service_module.review, "fetch_review", fetch_review)
    session_script(lambda session, text: build_outcome(opened))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings,
                        message(opened, "fix", review={"task": "fix_ci"}))

    assert events(frames)[-1]["code"] == "BAD_REQUEST" and "sign in again" in events(frames)[-1]["message"]
    assert session_script.sessions == []
    run = lab.change_detail(opened.id).thread[-1]
    assert (run.status, run.error.code) == ("failed", "BAD_REQUEST")
    assert_free(lab, opened)


# -- Stopping, early ends and errors ------------------------------------------------------------------------

def test_a_stream_closed_early_keeps_the_change_until_the_session_has_ended(
        lab, opened, spec_driven_runtime, lab_settings, session_script):
    """What durable_runs does when its store fails or the backend stops: it closes the stream."""
    started, finished = threading.Event(), threading.Event()

    def script(session, text):
        started.set()
        session.callbacks.on_progress(1, "write_file", "executing", "a.py")
        time.sleep(0.5)  # a tool call in progress does not see the cancel
        finished.set()
        return build_outcome(opened, "cancelled")

    session_script(script)

    async def go():
        runner = await start_run(lab, spec_driven_runtime, lab_settings, message(opened, "go"))
        stream = runner.stream()
        async for frame in stream:
            if parse(frame)["event"] == "tool_call":
                break
        await stream.aclose()
        return runner.run_id, finished.is_set()

    run_id, session_had_ended = asyncio.run(go())

    assert started.is_set() and session_had_ended
    assert asyncio.run(spec_driven_runtime.active_runs.request_cancellation(run_id)) is False
    run = lab.change_detail(opened.id).thread[-1]
    assert (run.runId, run.status) == (run_id, "cancelled")
    assert_free(lab, opened)


def test_cost_ticks_while_the_session_works_and_once_at_the_end(
        lab, opened, spec_driven_runtime, lab_settings, session_script):
    def script(session, text):
        session.client.usage.estimated_cost = 0.12
        time.sleep(0.2)  # ten emitter intervals
        session.client.usage.estimated_cost = 0.3
        return build_outcome(opened)

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(opened, "go"))
    got = events(frames)

    costs = [e for e in got if e["event"] == "cost"]
    assert len(costs) >= 3 and costs[0]["usd"] == 0.12
    assert got[-2]["event"] == "cost" and got[-2]["usd"] == 0.3 and got[-1]["event"] == "done"
    assert got[-1]["lab"]["cost"]["usd"] == pytest.approx(0.3)
    assert lab.change_detail(opened.id).runs[0].usd == pytest.approx(0.3)


@pytest.mark.parametrize("stop_reason, code", [
    ("cost_cap", "COST_CAP"), ("timeout", "TIMEOUT"), ("max_turns", "INCOMPLETE"), ("api_error", "INCOMPLETE"),
])
def test_a_run_that_stopped_early_warns_then_reports_what_it_did(
        lab, opened, spec_driven_runtime, lab_settings, session_script, stop_reason, code):
    session_script(lambda session, text: build_outcome(opened, stop_reason))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(opened, "go"))
    got = events(frames)

    assert got[-2]["event"] == "error" and got[-2]["code"] == code
    assert got[-1]["event"] == "done" and got[-1]["lab"]["stopReason"] == stop_reason
    assert lab.change_detail(opened.id).runs[0].stopReason == stop_reason


def test_the_cancel_route_stops_the_session_through_the_run_id(
        lab, opened, spec_driven_runtime, lab_settings, session_script):
    started = threading.Event()

    def script(session, text):
        started.set()
        deadline = time.monotonic() + 10
        while session.callbacks.should_continue() and time.monotonic() < deadline:
            time.sleep(0.01)
        return build_outcome(opened, "cancelled")

    session_script(script)

    async def go():
        runner = await start_run(lab, spec_driven_runtime, lab_settings, message(opened, "go"))
        consumer = asyncio.create_task(_collect(runner.stream()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        found = await spec_driven_runtime.active_runs.request_cancellation(runner.run_id)
        frames = await consumer
        again = await spec_driven_runtime.active_runs.request_cancellation(runner.run_id)
        return found, frames, again

    found, frames, again = asyncio.run(go())
    got = events(frames)

    assert found is True and again is False
    # E2E C-D3: the stream ends with a done event that says so and carries the checks that ran.
    assert got[-1]["event"] == "done" and got[-1]["cancelled"] is True
    assert [c["id"] for c in got[-1]["lab"]["checks"]] == ["besser:tests", "besser:docs"]
    assert all(e["event"] != "error" for e in got)
    assert lab.change_detail(opened.id).thread[-1].status == "cancelled"
    assert_free(lab, opened)


async def _collect(stream) -> list[bytes]:
    return [frame async for frame in stream]


def test_a_refused_key_is_invalid_key_and_gives_the_run_back(
        lab, draft, spec_driven_runtime, lab_settings, session_script):
    def script(session, text):
        raise InvalidApiKeyError("401 invalid x-api-key")

    session_script(script)
    run_id, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft))

    assert events(frames)[-1]["event"] == "error" and events(frames)[-1]["code"] == "INVALID_KEY"
    run = lab.change_detail(draft.id).thread[-1]
    assert (run.status, run.error.code) == ("failed", "INVALID_KEY")
    assert lab.change_detail(draft.id).runs[0].stopReason == "api_error"
    assert asyncio.run(spec_driven_runtime.active_runs.request_cancellation(run_id)) is False
    assert_free(lab, draft)


def test_a_client_that_cannot_be_built_is_invalid_key(lab, draft, spec_driven_runtime, lab_settings, monkeypatch):
    def no_client(provider, *, api_key=None, model=None, base_url=None):
        raise ValueError("No API key for anthropic")

    monkeypatch.setattr(runner_module, "create_llm_client", no_client)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft))

    assert [e["event"] for e in events(frames)] == ["start", "error"]
    assert events(frames)[-1]["code"] == "INVALID_KEY"
    assert_free(lab, draft)


def test_an_unexpected_session_error_is_internal_and_logged(
        lab, draft, spec_driven_runtime, lab_settings, session_script, caplog):
    def script(session, text):
        raise RuntimeError("boom in the session")

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft))
    got = events(frames)

    assert got[-1] == {"event": "error", "code": "INTERNAL", "message": "Internal server error", "reason": None,
                       "resumeAvailable": None}
    assert "boom in the session" not in b"".join(frames).decode()
    assert any("crashed" in r.message for r in caplog.records)
    assert_free(lab, draft)


def test_a_session_refusal_of_the_change_is_a_bad_request(
        lab, draft, spec_driven_runtime, lab_settings, session_script):
    def script(session, text):
        raise LabWorkspaceError(f"change {draft.id} is a draft: it has no plan yet; send a plan first")

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft, "go", action="build"))

    assert events(frames)[-1]["code"] == "BAD_REQUEST" and "is a draft" in events(frames)[-1]["message"]
    assert lab.change_detail(draft.id).runs[0].step == "build"


def test_the_lab_key_never_reaches_the_stream(lab, draft, spec_driven_runtime, lab_settings, session_script):
    def script(session, text):
        session.callbacks.on_text(f"the key is {LAB_KEY}")
        return plan_outcome(draft)

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, message(draft))

    assert LAB_KEY not in b"".join(frames).decode()


# -- The user's own key ------------------------------------------------------------------------------------

USER_KEY = "sk-user-own-key-123456"


def test_a_message_with_the_users_key_runs_on_it_and_keeps_it_out_of_everything_written(
        lab, draft, spec_driven_runtime, lab_settings, session_script):
    lab_settings.lab_plan_llm_model = "lab-plan-model"

    def script(session, text):
        session.callbacks.on_text(f"the key is {USER_KEY}")
        return plan_outcome(draft)

    sessions = session_script(script)
    request = message(draft, llm={"provider": "openai", "api_key": USER_KEY, "llm_model": "gpt-test"})
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, request)

    assert session_script.clients == [{"provider": "openai", "api_key": USER_KEY, "model": "gpt-test", "base_url": None}]
    # The lab's plan model belongs to the lab's key: a user's key plans on the user's model.
    assert sessions[0].wiring["plan_model"] is None
    start = events(frames)[0]
    assert (start["provider"], start["llmModel"]) == ("openai", "gpt-test")
    assert lab.change_detail(draft.id).runs[0].model == "gpt-test"
    # The key is in no frame, no file of the lab's state and not in the request's own text form.
    assert USER_KEY not in b"".join(frames).decode()
    for folder, _, files in os.walk(lab.workspace.state_dir):
        for name in files:
            with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as handle:
                assert USER_KEY not in handle.read(), name
    assert USER_KEY not in repr(request) and USER_KEY not in request.model_dump_json()
    # Paid by its owner: the month's spending did not move.
    assert lab.budget.meters(USER.login).user.spentUsd == 0.0


def test_a_custom_endpoint_is_refused_unless_the_deploy_allows_it(
        lab, draft, spec_driven_runtime, lab_settings, session_script):
    session_script(lambda session, text: plan_outcome(draft))
    lab_settings.llm_allow_custom_base_url = False
    request = message(draft, llm={"provider": "openai", "api_key": USER_KEY, "base_url": "http://10.0.0.5:8000/v1"})
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, request)

    assert events(frames)[-1]["code"] == "BAD_REQUEST" and session_script.clients == []
    assert_free(lab, draft)


@pytest.mark.parametrize("llm", [
    {"provider": "anthropic", "api_key": "   "},
    {"provider": "free", "api_key": USER_KEY},
    {"provider": "openai", "api_key": USER_KEY, "llm_model": "bad model"},
    {"provider": "openai", "api_key": USER_KEY, "base_url": "file:///etc/passwd"},
])
def test_a_malformed_user_key_is_refused_by_the_request(draft, llm):
    with pytest.raises(ValidationError):
        message(draft, llm=llm)


# -- The class diagram the steps follow ---------------------------------------------------------------

def _project(change_id: str, model: dict | None = None) -> ProjectInput:
    """The proposal of change ``change_id``, as the editor marks it."""
    model = model or proposal_from_plan(plan_outcome(None).plan)["model"]
    return ProjectInput.model_validate({
        "id": "p1", "type": "Project", "name": "Lab", "createdAt": "2026-11-19T00:00:00Z",
        "diagrams": {"ClassDiagram": [{"id": "d1", "title": "Note", "model": model,
                                       "config": {"labProposalChangeId": change_id}}]},
    })


def _review(change_id: str) -> LabReviewResponse:
    check = LabCheckRun(name="tests", app="github-actions", state="failure", url="https://github.com/x/1",
                        logTail=FAILING_TAIL, logNote="")
    pull = LabPullRequestStatus(repo="besser", number=812, url="https://github.com/BESSER-PEARL/BESSER/pull/812",
                                draft=True, headSha="a" * 40, state="open", pushedSha="a" * 40,
                                mergeable="mergeable", mergeState="clean", reviewDecision=None,
                                checksState="failure", checks=[check], reviews=[], comments=[])
    return LabReviewResponse(changeId=change_id, pullRequests=[pull], lifecycle="open", closable=False,
                             closableNote="A pull request is open.", nextPollSeconds=120, fetchedAt=time.time())


@pytest.mark.parametrize("kind, action, expected", [
    (ChangeKind.METAMODEL, "auto", MODEL_SPEC),
    (ChangeKind.DIAGRAM_TYPE, "build", MODEL_SPEC),
    # The user's own application model is not part of a generator change.
    (ChangeKind.GENERATOR, "auto", None),
    # E2E A-1: a plan revision starts from the diagram the user corrected.
    (ChangeKind.METAMODEL, "plan", MODEL_SPEC),
])
def test_a_step_of_a_model_change_reads_the_changes_class_diagram(
        lab, spec_driven_runtime, lab_settings, session_script, kind, action, expected):
    change = lab.workspace.open_change(lab.workspace.create_draft(owner="jordi"), title="Note", kind=kind)
    session_script(lambda session, text: plan_outcome(change))

    run_lab(lab, spec_driven_runtime, lab_settings, message(change, "build it", action=action,
                                                            project=_project(change.id)))

    assert session_script.sessions[0].handled[0]["model_spec"] == expected


def test_a_draft_does_not_read_the_project(lab, draft, spec_driven_runtime, lab_settings, session_script):
    session_script(lambda session, text: plan_outcome(draft))

    run_lab(lab, spec_driven_runtime, lab_settings, message(draft, project=_project(draft.id)))

    assert session_script.sessions[0].handled[0]["model_spec"] is None


def test_an_unreadable_class_diagram_is_a_bad_request(lab, spec_driven_runtime, lab_settings, session_script):
    change = lab.workspace.open_change(lab.workspace.create_draft(owner="jordi"), title="N",
                                       kind=ChangeKind.METAMODEL)
    model = proposal_from_plan(plan_outcome(None).plan)["model"]
    note = next(i for i, e in model["elements"].items() if e["type"] == "Class")
    model["relationships"] = {"r1": {"id": "r1", "type": "ClassBidirectional", "name": "bad",
                                     "source": {"element": note, "multiplicity": "5..2"},
                                     "target": {"element": note, "multiplicity": "1"}}}
    session_script(lambda session, text: plan_outcome(change))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings,
                        message(change, "go", project=_project(change.id, model)))
    got = events(frames)

    assert got[-1]["code"] == "BAD_REQUEST" and "class diagram could not be read" in got[-1]["message"]
    assert session_script.sessions == []
    assert_free(lab, change)
