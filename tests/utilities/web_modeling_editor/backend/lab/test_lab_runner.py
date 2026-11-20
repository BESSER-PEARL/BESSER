"""A BESSER Lab run as SSE: the session's callbacks, its outcome and its errors, mapped to Spec-Driven events.

The session is a fake with ``LabSession``'s API, so each outcome and error can be played exactly; everything
else is real: the runner, the lab service with a working copy under ``tmp_path``, the runtime's run ids.
``test_lab_router.py`` drives the real ``LabSession`` with a scripted model through the route.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.lab.contracts import (
    BuildOutcome,
    ChangeKind,
    CheckResult,
    ChecklistItem,
    FileChange,
    LabChange,
    LoopResult,
    Plan,
    RepoChange,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.web_modeling_editor.backend.models.lab import LabMessageRequest
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.lab import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.model_proposal import proposal_from_plan
from besser.utilities.web_modeling_editor.backend.services.lab.runner import LabRunner
from besser.utilities.web_modeling_editor.backend.services.lab.service import LabService
from pydantic import SecretStr

# No known key prefix: the stream's generic token patterns would redact an ``sk-ant-`` key anyway.
LAB_KEY = "labkey0123456789abcdefXYZ"
USER = LabUser(login="octocat", session_id="sess-1", access_token="gho_user")
CHANGE = LabChange(id="abcdef012345", slug="note-element", title="Add a note element", user="octocat",
                   kind=ChangeKind.METAMODEL, created_at=1_700_000_000.0,
                   branches={"besser": "lab/octocat/note-element"})
MODEL_SPEC = {
    "classes": [{"name": "Note", "attributes": [{"name": "text", "type": "str"}], "is_abstract": False}],
    "enumerations": [], "associations": [], "generalizations": [],
}


def plan_outcome() -> SessionOutcome:
    plan = Plan(kind=ChangeKind.METAMODEL, title="Add a note element", summary="Notes on class diagrams.",
                steps=["Add the Note metaclass", "Convert it both ways"], repos=["besser", "frontend"],
                questions=["Should notes have a colour?"], model_spec=MODEL_SPEC)
    return SessionOutcome(step="plan", change=CHANGE, plan=plan, cost_usd=0.05)


def build_outcome(stop_reason: str = "completed") -> SessionOutcome:
    build = BuildOutcome(
        loop=LoopResult(stop_reason=stop_reason, final_text="Added the Note metaclass."),
        changes=[
            RepoChange("frontend", "lab/octocat/note-element", "develop",
                       (FileChange("packages/editor/src/main/packages/note.ts", "added"),)),
            RepoChange("besser", "lab/octocat/note-element", "development",
                       (FileChange("besser/BUML/metamodel/structural/note.py", "added"),
                        FileChange("tests/BUML/metamodel/test_note.py", "added"))),
        ],
        checks=[CheckResult("besser:tests", "besser", "Tests of the touched areas", passed=True, summary="3 passed",
                            duration_seconds=12.5),
                CheckResult("besser:docs", "besser", "Docs", passed=False, skipped=True, summary="sphinx missing")],
        checklist=[ChecklistItem("metamodel.tests", "Tests under tests/BUML/metamodel/", True),
                   ChecklistItem("metamodel.docs", "Docs in docs/source/buml_language", None)],
        summary="besser: 2 files added",
    )
    return SessionOutcome(step="build", change=CHANGE, build=build, cost_usd=0.4, stop_reason=stop_reason)


class FakeClient:
    """Stands in for the provider client: only its cumulative cost is read."""

    def __init__(self):
        self.usage = SimpleNamespace(estimated_cost=0.0)


class FakeSession:
    """``LabSession``'s API; ``script(session, message, **kwargs)`` plays the run in the worker thread."""

    def __init__(self, workspace, *, client, limits, callbacks, script, record):
        self.workspace, self.client, self.limits, self.callbacks = workspace, client, limits, callbacks
        self._script = script
        record.append(self)
        self.handled = []

    def handle(self, message, *, user, action="auto", model_spec=None):
        self.handled.append({"message": message, "user": user, "action": action, "model_spec": model_spec})
        return self._script(self, message)


@pytest.fixture
def lab_settings(route_settings):
    route_settings.lab_enabled = True
    route_settings.lab_allowed_logins = "octocat"
    route_settings.lab_llm_provider = "anthropic"
    route_settings.lab_llm_model = "claude-test"
    route_settings.lab_llm_api_key = SecretStr(LAB_KEY)
    route_settings.lab_max_cost_usd = 2.5
    route_settings.lab_max_runtime_seconds = 600
    route_settings.llm_cost_emitter_interval_seconds = 0.02
    return route_settings


@pytest.fixture
def lab(tmp_path):
    """A lab service on a working copy with no repositories: the fake session never touches git."""
    return LabService(LabWorkspace(str(tmp_path / "work"), repos=[]))


@pytest.fixture
def session_script(monkeypatch):
    """Install ``script`` as the session of the next runs; returns the sessions created and the clients."""
    sessions: list = []
    state = {"script": lambda session, message: plan_outcome()}
    clients: list = []

    def fake_client(provider, *, api_key=None, model=None):
        clients.append({"provider": provider, "api_key": api_key, "model": model})
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


async def _start(lab, runtime, settings, request):
    run_id = uuid.uuid4().hex
    cancel_event = await runtime.active_runs.reserve(run_id)
    assert lab.try_begin_run(run_id)
    runner = LabRunner(request, user=USER, lab=lab, runtime=runtime, settings=settings, run_id=run_id,
                       cancel_event=cancel_event)
    return run_id, runner


def run_lab(lab, runtime, settings, request=None) -> tuple[str, list[bytes]]:
    """Run one message to its end; the run id and the raw frames."""
    request = request or LabMessageRequest(message="Add a note element to class diagrams")

    async def go():
        run_id, runner = await _start(lab, runtime, settings, request)
        return run_id, [frame async for frame in runner.stream()]

    return asyncio.run(go())


def events(frames) -> list[dict]:
    return [parse(frame) for frame in frames]


def test_a_plan_streams_start_progress_and_a_done_event_with_the_plan(
        lab, spec_driven_runtime, lab_settings, session_script):
    def script(session, message):
        session.callbacks.on_phase("plan", "Planning the change")
        session.callbacks.on_progress(1, "read_file", "executing", "CLAUDE.md")
        session.callbacks.on_text("I read the metamodel.")
        session.callbacks.on_progress(2, "submit_plan", "executing", None)
        session.callbacks.on_text("Here is the plan.")
        return plan_outcome()

    session_script(script)
    run_id, frames = run_lab(lab, spec_driven_runtime, lab_settings)
    every = events(frames)
    # The 20 ms cost emitter may add cost events anywhere on a loaded machine; the final one precedes done.
    assert every[-2]["event"] == "cost"
    got = [e for e in every[:-2] if e["event"] != "cost"] + every[-1:]

    assert [e["event"] for e in got] == ["start", "phase", "tool_call", "text", "tool_call", "text", "done"]
    start = got[0]
    assert start == {"event": "start", "runId": run_id, "provider": "anthropic", "llmModel": "claude-test",
                     "maxCost": 2.5, "maxRuntime": 600}
    assert got[1]["phase"] == "plan" and got[1]["message"] == "Planning the change"
    assert got[2] == {"event": "tool_call", "turn": 1, "tool": "read_file", "status": "executing",
                      "summary": None, "detail": "CLAUDE.md"}
    # A new turn's text starts a new paragraph.
    assert got[5]["delta"] == "\n\nHere is the plan."
    done = got[-1]
    assert (done["runId"], done["downloadUrl"], done["fileName"], done["isZip"], done["mode"]) == (
        run_id, "", "", False, "modify")
    assert done["changedFiles"] == [] and done["changedFileCount"] == 0
    lab_payload = done["lab"]
    assert lab_payload["step"] == "plan"
    assert (lab_payload["changeId"], lab_payload["title"], lab_payload["kind"]) == (
        "abcdef012345", "Add a note element", "metamodel")
    assert lab_payload["plan"] == {"kind": "metamodel", "title": "Add a note element",
                                   "summary": "Notes on class diagrams.",
                                   "steps": ["Add the Note metaclass", "Convert it both ways"],
                                   "repos": ["besser", "frontend"], "questions": ["Should notes have a colour?"]}
    proposal = lab_payload["modelProposal"]
    assert (proposal["title"], proposal["diagramType"], proposal["model"]["type"]) == (
        "Add a note element", "ClassDiagram", "ClassDiagram")
    assert [e["name"] for e in proposal["model"]["elements"].values() if e["type"] == "Class"] == ["Note"]
    assert lab_payload["summary"] == "Notes on class diagrams." and lab_payload["stopReason"] == "completed"
    assert lab_payload["finalText"] == ""


def test_a_plan_step_without_a_plan_reports_the_models_last_words(
        lab, spec_driven_runtime, lab_settings, session_script):
    # A question back to the user is otherwise only in the streamed text, which a re-attached card misses.
    session_script(lambda session, message: SessionOutcome(step="plan", final_text="Which diagram type?"))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)

    lab_payload = events(frames)[-1]["lab"]
    assert (lab_payload["step"], lab_payload["plan"], lab_payload["changeId"]) == ("plan", None, "")
    assert lab_payload["summary"] == "Which diagram type?" and lab_payload["finalText"] == ""


def test_a_plan_that_changes_nothing_is_an_answer_with_no_change_and_no_model(
        lab, spec_driven_runtime, lab_settings, session_script):
    # E2E C-D4: the request was already met. Even a model the agent sent along is not proposed: no change owns it.
    plan = Plan(kind=ChangeKind.METAMODEL, title="Note already exists", summary="The Note metaclass is already there.",
                steps=[], repos=[], model_spec=MODEL_SPEC)
    session_script(lambda session, message: SessionOutcome(step="plan", plan=plan))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)

    lab_payload = events(frames)[-1]["lab"]
    assert (lab_payload["changeId"], lab_payload["plan"]["repos"], lab_payload["modelProposal"]) == ("", [], None)
    assert lab_payload["summary"] == "The Note metaclass is already there."


def test_the_session_gets_the_lab_settings_and_the_server_key(lab, spec_driven_runtime, lab_settings, session_script):
    run_lab(lab, spec_driven_runtime, lab_settings)

    assert session_script.clients == [{"provider": "anthropic", "api_key": LAB_KEY, "model": "claude-test"}]
    session = session_script.sessions[0]
    assert (session.limits.max_cost_usd, session.limits.max_runtime_seconds) == (2.5, 600.0)
    assert session.handled == [{"message": "Add a note element to class diagrams", "user": "octocat",
                                "action": "auto", "model_spec": None}]


def test_a_build_reports_files_per_repository_checks_and_checklist(
        lab, spec_driven_runtime, lab_settings, session_script):
    session_script(lambda session, message: build_outcome())

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, LabMessageRequest(message="go", action="build"))
    done = events(frames)[-1]

    assert done["event"] == "done"
    assert done["changedFiles"] == ["frontend/packages/editor/src/main/packages/note.ts",
                                    "besser/besser/BUML/metamodel/structural/note.py",
                                    "besser/tests/BUML/metamodel/test_note.py"]
    assert done["changedFileCount"] == 3
    lab_payload = done["lab"]
    assert lab_payload["step"] == "build" and lab_payload["plan"] is None and lab_payload["modelProposal"] is None
    assert lab_payload["repoChanges"][1] == {
        "repo": "besser", "branch": "lab/octocat/note-element", "base": "development",
        "files": [{"path": "besser/BUML/metamodel/structural/note.py", "status": "added"},
                  {"path": "tests/BUML/metamodel/test_note.py", "status": "added"}]}
    assert lab_payload["checks"] == [
        {"id": "besser:tests", "repo": "besser", "title": "Tests of the touched areas", "passed": True,
         "skipped": False, "summary": "3 passed", "durationSeconds": 12.5},
        {"id": "besser:docs", "repo": "besser", "title": "Docs", "passed": False, "skipped": True,
         "summary": "sphinx missing", "durationSeconds": 0.0}]
    assert lab_payload["checklist"] == [
        {"id": "metamodel.tests", "text": "Tests under tests/BUML/metamodel/", "done": True},
        {"id": "metamodel.docs", "text": "Docs in docs/source/buml_language", "done": None}]
    assert lab_payload["summary"] == "besser: 2 files added"
    # The model's own closing account, which the card shows above the files.
    assert lab_payload["finalText"] == "Added the Note metaclass."


def test_the_last_run_of_a_change_is_kept_with_the_change(lab, spec_driven_runtime, lab_settings, session_script):
    session_script(lambda session, message: build_outcome())

    run_lab(lab, spec_driven_runtime, lab_settings)

    # Read back by a new service: a restart keeps the checks.
    remembered = LabService(lab.workspace)._last_run("abcdef012345")
    assert remembered.step == "build" and remembered.checks[0].summary == "3 passed"
    assert remembered.checklist[1].done is None


def test_a_last_run_that_cannot_be_read_is_logged_and_left_out(lab, caplog):
    path = lab._last_run_path("abcdef012345")
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")

    assert lab._last_run("abcdef012345") is None
    assert any("cannot be read" in record.message for record in caplog.records)


def test_a_stream_closed_early_keeps_the_lab_until_the_session_has_ended(
        lab, spec_driven_runtime, lab_settings, session_script):
    """What durable_runs does when its store fails or the backend stops: it closes the stream."""
    started, finished = threading.Event(), threading.Event()

    def script(session, message):
        started.set()
        session.callbacks.on_progress(1, "write_file", "executing", "a.py")
        time.sleep(0.5)  # a tool call in progress does not see the cancel
        finished.set()
        return build_outcome("cancelled")

    session_script(script)

    async def go():
        run_id, runner = await _start(lab, spec_driven_runtime, lab_settings, LabMessageRequest(message="go"))
        stream = runner.stream()
        async for frame in stream:
            if parse(frame)["event"] == "tool_call":
                break
        await stream.aclose()
        return run_id, finished.is_set()

    run_id, session_had_ended = asyncio.run(go())

    assert started.is_set() and session_had_ended
    assert lab.try_begin_run("f" * 32)
    assert asyncio.run(spec_driven_runtime.active_runs.request_cancellation(run_id)) is False


def test_cost_ticks_while_the_session_works_and_once_at_the_end(
        lab, spec_driven_runtime, lab_settings, session_script):
    def script(session, message):
        session.client.usage.estimated_cost = 0.12
        time.sleep(0.2)  # ten emitter intervals
        session.client.usage.estimated_cost = 0.3
        return build_outcome()

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)
    got = events(frames)

    costs = [e for e in got if e["event"] == "cost"]
    assert len(costs) >= 3
    assert costs[0]["usd"] == 0.12
    assert got[-2]["event"] == "cost" and got[-2]["usd"] == 0.3 and got[-1]["event"] == "done"


@pytest.mark.parametrize("stop_reason, code", [
    ("cost_cap", "COST_CAP"), ("timeout", "TIMEOUT"), ("max_turns", "INCOMPLETE"), ("api_error", "INCOMPLETE"),
])
def test_a_run_that_stopped_early_warns_then_reports_what_it_did(
        lab, spec_driven_runtime, lab_settings, session_script, stop_reason, code):
    session_script(lambda session, message: build_outcome(stop_reason))

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)
    got = events(frames)

    assert got[-2]["event"] == "error" and got[-2]["code"] == code
    assert "working copy" in got[-2]["message"]
    assert got[-1]["event"] == "done" and got[-1]["lab"]["stopReason"] == stop_reason


def test_the_cancel_route_stops_the_session_through_the_run_id(
        lab, spec_driven_runtime, lab_settings, session_script):
    started = threading.Event()

    def script(session, message):
        started.set()
        deadline = time.monotonic() + 10
        while session.callbacks.should_continue() and time.monotonic() < deadline:
            time.sleep(0.01)
        return build_outcome("cancelled")

    session_script(script)

    async def go():
        run_id, runner = await _start(lab, spec_driven_runtime, lab_settings, LabMessageRequest(message="go"))
        consumer = asyncio.create_task(_collect(runner.stream()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        # What POST /spec-driven/cancel/{run_id} calls.
        found = await spec_driven_runtime.active_runs.request_cancellation(run_id)
        frames = await consumer
        again = await spec_driven_runtime.active_runs.request_cancellation(run_id)
        return found, frames, again

    found, frames, again = asyncio.run(go())
    got = events(frames)

    assert found is True
    # E2E C-D3: the stream ends with a done event that says so and carries the checks that ran.
    assert got[-1]["event"] == "done" and got[-1]["cancelled"] is True
    assert got[-1]["lab"]["stopReason"] == "cancelled"
    assert [c["id"] for c in got[-1]["lab"]["checks"]] == ["besser:tests", "besser:docs"]
    assert all(e["event"] != "error" for e in got)
    # The run id and the lab are free again.
    assert again is False
    assert lab.try_begin_run("f" * 32)


async def _collect(stream) -> list[bytes]:
    return [frame async for frame in stream]


def test_a_refused_key_is_invalid_key_and_releases_the_lab(lab, spec_driven_runtime, lab_settings, session_script):
    def script(session, message):
        raise InvalidApiKeyError("401 invalid x-api-key")

    session_script(script)
    run_id, frames = run_lab(lab, spec_driven_runtime, lab_settings)
    got = events(frames)

    assert got[-1]["event"] == "error" and got[-1]["code"] == "INVALID_KEY"
    assert lab.try_begin_run("f" * 32)
    assert asyncio.run(spec_driven_runtime.active_runs.request_cancellation(run_id)) is False


def test_a_client_that_cannot_be_built_is_invalid_key(lab, spec_driven_runtime, lab_settings, monkeypatch):
    def no_client(provider, *, api_key=None, model=None):
        raise ValueError("No API key for anthropic")

    monkeypatch.setattr(runner_module, "create_llm_client", no_client)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)

    assert [e["event"] for e in events(frames)] == ["start", "error"]
    assert events(frames)[-1]["code"] == "INVALID_KEY"
    assert lab.try_begin_run("f" * 32)


def test_an_unexpected_session_error_is_internal_and_logged(
        lab, spec_driven_runtime, lab_settings, session_script, caplog):
    def script(session, message):
        raise RuntimeError("boom in the session")

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)
    got = events(frames)

    assert got[-1] == {"event": "error", "code": "INTERNAL", "message": "Internal server error", "reason": None,
                       "resumeAvailable": None}
    assert "boom in the session" not in b"".join(frames).decode()
    assert any("crashed" in r.message for r in caplog.records)
    assert lab.try_begin_run("f" * 32)


def test_the_lab_key_never_reaches_the_stream(lab, spec_driven_runtime, lab_settings, session_script):
    def script(session, message):
        session.callbacks.on_text(f"the key is {LAB_KEY}")
        return plan_outcome()

    session_script(script)
    _, frames = run_lab(lab, spec_driven_runtime, lab_settings)

    assert LAB_KEY not in b"".join(frames).decode()


# -- The class diagram the build follows ---------------------------------------------------------------

def _project() -> ProjectInput:
    """The proposal of change abcdef012345, as the editor marks it."""
    model = proposal_from_plan(plan_outcome().plan)["model"]
    return ProjectInput.model_validate({
        "id": "p1", "type": "Project", "name": "Lab", "createdAt": "2026-11-19T00:00:00Z",
        "diagrams": {"ClassDiagram": [{"id": "d1", "title": "Note", "model": model,
                                       "config": {"labProposalChangeId": "abcdef012345"}}]},
    })


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
    lab.workspace.save_change(LabChange(id="abcdef012345", slug="note", title="Note", user="octocat", kind=kind))

    run_lab(lab, spec_driven_runtime, lab_settings,
            LabMessageRequest(message="build it", action=action, project=_project()))

    assert session_script.sessions[0].handled[0]["model_spec"] == expected


def test_without_an_active_change_the_project_is_not_read(lab, spec_driven_runtime, lab_settings, session_script):
    run_lab(lab, spec_driven_runtime, lab_settings, LabMessageRequest(message="plan it", project=_project()))

    assert session_script.sessions[0].handled[0]["model_spec"] is None


def test_an_unreadable_class_diagram_is_a_bad_request(lab, spec_driven_runtime, lab_settings, session_script):
    lab.workspace.save_change(LabChange(id="abcdef012345", slug="n", title="N", user="octocat",
                                        kind=ChangeKind.METAMODEL))
    model = proposal_from_plan(plan_outcome().plan)["model"]
    note = next(i for i, e in model["elements"].items() if e["type"] == "Class")
    relationship = {"id": "r1", "type": "ClassBidirectional", "name": "bad",
                    "source": {"element": note, "multiplicity": "5..2"},
                    "target": {"element": note, "multiplicity": "1"}}
    model["relationships"] = {"r1": relationship}
    project = ProjectInput.model_validate({
        "id": "p1", "type": "Project", "name": "Lab", "createdAt": "2026-11-19T00:00:00Z",
        "diagrams": {"ClassDiagram": [{"id": "d1", "title": "Note", "model": model,
                                       "config": {"labProposalChangeId": "abcdef012345"}}]}})

    _, frames = run_lab(lab, spec_driven_runtime, lab_settings, LabMessageRequest(message="go", project=project))
    got = events(frames)

    assert got[-1]["code"] == "BAD_REQUEST" and "class diagram could not be read" in got[-1]["message"]
    assert session_script.sessions == []
    assert lab.try_begin_run("f" * 32)
