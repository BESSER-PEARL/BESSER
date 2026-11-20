"""LabSession: whole messages driven by a scripted model in real git repositories.

The working copy is the server's layout (BESSER with the editor submodule, and the
modeling agent). The checks' commands are replaced by a scripted runner; which
checks run is still decided by the real ``checks_for``.
"""

import json
import os
import shutil
import subprocess
import time

import pytest

from besser.spec_driven_agent.lab import checks
from besser.spec_driven_agent.lab import session as session_module
from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_MODELING_AGENT,
    ChangeKind,
    ExtraTool,
    LabCallbacks,
    LabLimits,
    Plan,
    RepoSpec,
)
from besser.spec_driven_agent.lab.kinds import guidance
from besser.spec_driven_agent.lab.session import CHECKS_FAILED, PLAN_REMINDER, LabSession
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from tests.spec_driven_agent.lab.conftest import end_turn, tool_use

PLAN = {
    "kind": "agent",
    "title": "Add a greeting",
    "summary": "The modeling agent greets the user.",
    "steps": ["Add modeling-agent/src/greeting.py and tests/test_greeting.py"],
    "repos": ["modeling_agent"],
}
GREETING = ("write_file", {"path": "modeling-agent/src/greeting.py", "content": "GREETING = 'hello'\n"})
GREETING_TEST = ("write_file", {"path": "modeling-agent/tests/test_greeting.py",
                                "content": "from src.greeting import GREETING\n\n\ndef test_greeting():\n"
                                           "    assert GREETING == 'hello'\n"})
FAILED_TESTS = (1, "FAILED tests/test_greeting.py::test_greeting - AssertionError\n1 failed, 3 passed")


def git(cwd, *args):
    result = subprocess.run(
        ["git", "-c", "user.name=setup", "-c", "user.email=setup@example.com", "-c", "core.autocrlf=false",
         "-c", "protocol.file.allow=always", *args],
        cwd=cwd, check=True, capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def init_repo(path, branch, files):
    os.makedirs(path)
    git(path, "init", "-q", "-b", branch)
    for name, text in files.items():
        target = os.path.join(path, *name.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A lab root: BESSER with the editor as a submodule, and the modeling agent."""
    base = tmp_path_factory.mktemp("lab_session_template")
    editor_origin = str(base / "editor-origin")
    init_repo(editor_origin, "develop", {"packages/webapp/src/app.ts": "export const app = 1;\n"})
    root = base / "root"
    besser = str(root / "besser")
    init_repo(besser, "development", {"README.md": "BESSER\n", "besser/core.py": "CORE = 1\n"})
    git(besser, "submodule", "add", "-q", editor_origin, FRONTEND_SUBMODULE_PATH)
    git(besser, "commit", "-q", "-m", "add the editor")
    init_repo(str(root / "modeling-agent"), "develop", {
        "src/agent.py": "AGENT = 1\n",
        "tests/test_agent.py": "def test_agent():\n    assert True\n",
    })
    return root


@pytest.fixture
def ws(template, tmp_path):
    target = tmp_path / "root"
    shutil.copytree(template, target)
    return LabWorkspace(str(target))


class ScriptedRunner:
    """Stands in for the checks' commands: (exit code, output) per check id, the last one repeating."""

    def __init__(self, results=None):
        self.results = {key: list(value) for key, value in (results or {}).items()}
        self.calls = []

    def __call__(self, spec, cwd, should_continue=None):
        self.calls.append(spec.id)
        script = self.results.get(spec.id)
        if not script:
            return 0, "4 passed"
        return script.pop(0) if len(script) > 1 else script[0]


@pytest.fixture
def runner(monkeypatch):
    runner = ScriptedRunner()
    monkeypatch.setattr(checks, "default_runner", runner)
    return runner


def start(ws, kind=ChangeKind.AGENT, title="Add a greeting", approved=True):
    change = ws.start_change(user="Jordi", title=title, kind=kind)
    change.plan_approved = approved
    ws.save_change(change)
    ws.append_note(change, f"## Plan\nTitle: {title}")
    return change


def branch_of(ws, repo):
    return git(ws.repos[repo].path, "rev-parse", "--abbrev-ref", "HEAD")


def check_ids(ws, outcome):
    """The checks the real ``checks_for`` picks for the build's files, in order."""
    return [spec.id for spec in checks.checks_for(outcome.build.changes, ws.repos)]


def last_user_text(messages):
    content = messages[-1]["content"]
    return content if isinstance(content, str) else "\n".join(
        block.get("text") or str(block.get("content", "")) for block in content)


# -- plan --------------------------------------------------------------------------------------------------


def test_auto_without_a_change_plans_read_only_and_creates_the_change_from_the_plan(ws, scripted_client):
    client = scripted_client([tool_use("read_file", {"path": "modeling-agent/src/agent.py"}),
                              tool_use("submit_plan", PLAN)])

    outcome = LabSession(ws, client=client).handle("Make the agent greet the user", user="Jordi")

    assert outcome.step == "plan" and outcome.stop_reason == "completed"
    assert outcome.plan.title == "Add a greeting" and outcome.plan.kind is ChangeKind.AGENT
    change = ws.active_change()
    assert change.id == outcome.change.id
    assert (change.title, change.kind, change.user, change.plan_approved) == (
        "Add a greeting", ChangeKind.AGENT, "Jordi", False)
    assert all(branch_of(ws, repo) == "lab/jordi/add-a-greeting" for repo in ws.repos)
    system, messages, tools = client.calls[0]
    assert "## This step: plan" in system
    assert "submit_plan" in tools and "read_file" in tools
    assert not {"write_file", "run_command", "run_checks", "scaffold_generator"} & set(tools)
    # The tools work at the root, so the prompt's repository paths resolve.
    assert "AGENT = 1" in last_user_text(client.calls[1][1])
    notes = ws.notes(change)
    assert "## Request\nMake the agent greet the user" in notes and "Title: Add a greeting" in notes


def test_a_plan_written_as_text_gets_one_reminder_to_call_submit_plan(ws, scripted_client):
    client = scripted_client([end_turn("My plan: add a greeting."), tool_use("submit_plan", PLAN)])

    outcome = LabSession(ws, client=client).handle("Greet the user", user="Jordi")

    assert outcome.plan.title == "Add a greeting" and ws.active_change() is not None
    assert last_user_text(client.calls[1][1]) == PLAN_REMINDER


def test_no_change_is_created_when_the_model_never_submits_a_plan(ws, scripted_client):
    client = scripted_client([end_turn("Which agent?"), end_turn("Still no plan.")])

    outcome = LabSession(ws, client=client).handle("Greet the user", user="Jordi")

    assert len(client.calls) == 2
    assert outcome.step == "plan" and outcome.plan is None and outcome.change is None
    # The model's last words (often a question) are the only thing the user can read.
    assert outcome.final_text == "Still no plan."
    assert ws.active_change() is None
    assert branch_of(ws, REPO_MODELING_AGENT) == "develop"


def test_a_plan_that_changes_nothing_creates_no_change_and_the_next_message_continues_it(ws, scripted_client,
                                                                                        monkeypatch):
    # Live run C (D4): "The request is already met, so I plan no change" still made a change and a branch in
    # all three repositories. A plan that touches no repository (nothing to do, or only questions) is the
    # model's answer, not a change.
    def no_change_plan_tool(on_plan):
        def handle(arguments):
            on_plan(Plan(kind=ChangeKind.OTHER, title="Check the docstring", repos=[], steps=[],
                         summary="get_build_sha already has a docstring, so nothing changes.",
                         questions=["Do you want different wording?"]))
            return "Plan submitted."
        return ExtraTool(schema={"name": "submit_plan", "description": "d",
                                 "input_schema": {"type": "object", "properties": {}}},
                         handler=handle, writes=False)

    monkeypatch.setattr(session_module, "submit_plan_tool", no_change_plan_tool)
    client = scripted_client([tool_use("submit_plan", {}), tool_use("submit_plan", {})])
    session = LabSession(ws, client=client)

    outcome = session.handle("Give get_build_sha a docstring", user="Jordi")

    assert outcome.step == "plan" and outcome.change is None
    assert outcome.plan.summary == "get_build_sha already has a docstring, so nothing changes."
    assert ws.active_change() is None
    assert {repo: branch_of(ws, repo) for repo in ws.repos} == {
        "frontend": "develop", "besser": "development", "modeling_agent": "develop"}
    # Kept like a question back: the user's answer continues the same request.
    session.handle("Yes: say it returns None when not stamped", user="Jordi")
    first = client.calls[1][1][0]["content"]
    assert "The user: Give get_build_sha a docstring" in first
    assert "nothing changes." in first and "Do you want different wording?" in first


def test_a_plan_on_an_active_change_revises_it_and_asks_for_approval_again(ws, scripted_client):
    change = start(ws, kind=ChangeKind.OTHER, title="Greet people")
    client = scripted_client([tool_use("submit_plan", PLAN)])

    outcome = LabSession(ws, client=client).handle("Only in the agent, please", user="Jordi", action="plan")

    system = client.calls[0][0]
    assert "## This change: other" in system and "Title: Greet people" in system
    stored = ws.active_change()
    assert stored.id == change.id == outcome.change.id
    assert (stored.title, stored.kind, stored.plan_approved) == ("Add a greeting", ChangeKind.AGENT, False)
    assert stored.branches == change.branches
    assert "## Request (plan revised)\nOnly in the agent, please" in ws.notes(stored)


def test_a_reply_to_a_plan_not_yet_approved_revises_it_and_builds_nothing(ws, scripted_client, runner):
    # The plan card asks "say what to change": a correction must not start a paid build.
    change = start(ws, kind=ChangeKind.OTHER, title="Greet people", approved=False)
    client = scripted_client([tool_use("submit_plan", PLAN)])

    outcome = LabSession(ws, client=client).handle("No, put it in the agent", user="Jordi")

    assert outcome.step == "plan" and outcome.change.id == change.id
    assert not {"write_file", "run_command", "run_checks"} & set(client.calls[0][2])
    stored = ws.active_change()
    assert (stored.title, stored.plan_approved) == ("Add a greeting", False)
    assert "## Request (plan revised)\nNo, put it in the agent" in ws.notes(stored)
    assert runner.calls == []


def test_a_question_back_on_a_plan_being_revised_is_kept_in_the_notes(ws, scripted_client):
    change = start(ws, approved=False)
    client = scripted_client([end_turn("Which file?"), end_turn("Which file do you mean?")])

    outcome = LabSession(ws, client=client).handle("Why step 3?", user="Jordi")

    assert outcome.plan is None and outcome.final_text == "Which file do you mean?"
    assert ("## Request without a plan\nThe user: Why step 3?\nThe lab agent: Which file do you mean?"
            in ws.notes(change))


def test_the_answer_to_a_question_from_the_plan_step_continues_the_request(ws, scripted_client):
    client = scripted_client([end_turn("Which agent?"), end_turn("Which agent do you mean?"),
                              tool_use("submit_plan", PLAN)])
    session = LabSession(ws, client=client)
    session.handle("Make the agent greet", user="Jordi")

    outcome = session.handle("The modeling agent", user="Jordi")

    first = client.calls[2][1][0]["content"]
    assert "The user: Make the agent greet\nThe lab agent: Which agent do you mean?" in first
    assert first.endswith("## The user's message\nThe modeling agent")
    notes = ws.notes(outcome.change)
    assert "Make the agent greet" in notes and "The user: The modeling agent" in notes
    # Planned once: the next request starts from nothing.
    assert session._pending("Jordi") == ""


def test_a_pending_request_is_only_continued_by_its_own_user(ws, scripted_client):
    client = scripted_client([end_turn("Which agent?"), end_turn("Which agent do you mean?"),
                              tool_use("submit_plan", PLAN)])
    session = LabSession(ws, client=client)
    session.handle("Make the agent greet", user="Jordi")

    session.handle("Add a greeting to the agent", user="Ana")

    assert "Make the agent greet" not in client.calls[2][1][0]["content"]


def test_the_editors_class_diagram_reaches_the_model(ws, scripted_client):
    client = scripted_client([tool_use("submit_plan", PLAN)])
    spec = {"classes": [{"name": "Note", "attributes": [{"name": "text", "type": "str"}]}]}

    LabSession(ws, client=client).handle("Add a note element", user="Jordi", model_spec=spec)

    first = client.calls[0][1][0]["content"]
    assert "## The class diagram of this change in the editor" in first and "- Note: text: str" in first


def test_a_working_copy_that_is_not_ready_fails_before_the_model_is_called(ws, scripted_client):
    broken = LabWorkspace(ws.root, [RepoSpec(REPO_BESSER, os.path.join(ws.root, "missing"), "development", "o", "n")])
    client = scripted_client([tool_use("submit_plan", PLAN)])

    with pytest.raises(LabWorkspaceError, match="does not exist"):
        LabSession(broken, client=client).handle("Greet the user", user="Jordi")
    assert client.calls == []


def test_uncommitted_work_in_the_working_copy_fails_before_the_model_is_called(ws, scripted_client):
    # Otherwise the plan is paid for and start_change refuses it afterwards.
    with open(os.path.join(ws.repos[REPO_MODELING_AGENT].path, "src", "agent.py"), "a", encoding="utf-8") as f:
        f.write("STRAY = 1\n")
    client = scripted_client([tool_use("submit_plan", PLAN)])

    with pytest.raises(LabWorkspaceError, match="uncommitted changes"):
        LabSession(ws, client=client).handle("Greet the user", user="Jordi")
    assert client.calls == []


# -- build -------------------------------------------------------------------------------------------------


def test_plan_then_build_builds_the_change_checks_it_and_reports_it(ws, scripted_client, runner):
    client = scripted_client([
        tool_use("submit_plan", PLAN),
        tool_use(*GREETING, GREETING_TEST),
        tool_use("run_checks", {}),
        end_turn("Added a greeting and its test."),
    ])
    session = LabSession(ws, client=client)
    session.handle("Make the agent greet the user", user="Jordi")

    outcome = session.handle("Yes, build it", user="Jordi", action="build")

    assert outcome.step == "build" and outcome.stop_reason == "completed"
    assert ws.active_change().plan_approved is True
    system, messages, tools = client.calls[1]
    assert "## This step: build" in system and "Title: Add a greeting" in system
    assert guidance(ChangeKind.AGENT).strip().splitlines()[0] in system
    assert "Yes, build it" in messages[0]["content"]
    assert {"write_file", "run_command", "run_checks"} <= set(tools)
    assert not {"submit_plan", "scaffold_generator"} & set(tools)
    # The run_checks tool and the check after the loop both ran the agent's checks.
    ids = check_ids(ws, outcome)
    assert "modeling_agent:tests" in ids and runner.calls == ids * 2
    assert f"Checks: 0 failed, 0 skipped, {len(ids)} passed." in last_user_text(client.calls[3][1])

    build = outcome.build
    [repo_change] = build.changes
    assert repo_change.repo == REPO_MODELING_AGENT and repo_change.branch == "lab/jordi/add-a-greeting"
    assert {(f.path, f.status) for f in repo_change.files} == {("src/greeting.py", "added"),
                                                               ("tests/test_greeting.py", "added")}
    assert all(c.passed for c in build.checks)
    assert {"common.tests", "common.no_workflows", "common.i18n"} <= {item.id for item in build.checklist}
    assert "added src/greeting.py" in build.summary and build.summary.endswith("Added a greeting and its test.")
    assert build.loop.turns == 3
    notes = ws.notes(ws.active_change())
    assert "## Build request\nYes, build it" in notes and "## Build result (completed)" in notes
    assert "added src/greeting.py" in notes


def test_failed_checks_go_back_to_the_model_in_the_same_conversation(ws, scripted_client, runner):
    start(ws)
    runner.results["modeling_agent:tests"] = [FAILED_TESTS, (0, "5 passed")]
    client = scripted_client([
        tool_use(*GREETING),
        end_turn("Added the greeting."),
        tool_use(*GREETING_TEST),
        end_turn("Added the greeting and a test that passes."),
    ])

    outcome = LabSession(ws, client=client).handle("Build it", user="Jordi")

    fix_messages = client.calls[2][1]
    assert fix_messages[:3] == client.calls[1][1][:3]
    assert fix_messages[-2]["role"] == "assistant"
    failure = last_user_text(fix_messages)
    assert failure.startswith(CHECKS_FAILED) and "FAILED modeling_agent:tests" in failure
    assert "AssertionError" in failure
    build = outcome.build
    assert outcome.stop_reason == "completed" and all(c.passed for c in build.checks)
    assert build.loop.turns == 4 and build.loop.final_text == "Added the greeting and a test that passes."
    assert len(build.loop.tool_calls) == 2
    assert {f.path for f in build.changes[0].files} == {"src/greeting.py", "tests/test_greeting.py"}


def test_fix_rounds_are_bounded_and_the_last_failure_is_reported(ws, scripted_client, runner):
    start(ws)
    runner.results["modeling_agent:tests"] = [FAILED_TESTS]
    client = scripted_client([tool_use(*GREETING), end_turn("Done."), end_turn("Could not fix it.")])

    outcome = LabSession(ws, client=client, max_check_rounds=1).handle("Build it", user="Jordi")

    assert len(client.calls) == 3
    assert outcome.stop_reason == "completed"
    # The fix round changed nothing, so its checks are not run again.
    assert runner.calls == check_ids(ws, outcome)
    failed = [c for c in outcome.build.checks if not c.passed and not c.skipped]
    assert [c.spec_id for c in failed] == ["modeling_agent:tests"]
    assert "1 failed" in outcome.build.summary and outcome.build.summary.endswith("Could not fix it.")


def test_a_skipped_check_does_not_start_a_fix_round(ws, scripted_client, runner):
    start(ws)
    runner.results["modeling_agent:tests"] = [(checks.MISSING_TOOL, "pytest is not installed")]
    client = scripted_client([tool_use(*GREETING), end_turn("Done.")])

    outcome = LabSession(ws, client=client).handle("Build it", user="Jordi")

    assert len(client.calls) == 2
    assert [c.spec_id for c in outcome.build.checks if c.skipped] == ["modeling_agent:tests"]


def test_all_loops_of_a_message_share_the_cost_cap(ws, scripted_client, runner):
    start(ws)
    runner.results["modeling_agent:tests"] = [FAILED_TESTS]
    client = scripted_client([tool_use(*GREETING), end_turn("Done."), end_turn("never sent")], cost_per_call=1.0)

    outcome = LabSession(ws, client=client, limits=LabLimits(max_cost_usd=1.5)).handle("Build it", user="Jordi")

    # The fix round starts above the cap, so it stops before calling the model, and the checks are not rerun.
    assert len(client.calls) == 2
    assert runner.calls == check_ids(ws, outcome)
    assert outcome.stop_reason == "cost_cap" and outcome.cost_usd == pytest.approx(2.0)
    assert [c.spec_id for c in outcome.build.checks if not c.passed] == ["modeling_agent:tests"]
    assert outcome.build.loop.final_text == "Done."
    assert "## Build result (cost_cap)" in ws.notes(ws.active_change())


def test_a_generator_change_offers_the_scaffold_and_other_kinds_do_not(ws, scripted_client, runner):
    start(ws, kind=ChangeKind.GENERATOR, title="Add a CSV generator")
    client = scripted_client([end_turn("Nothing yet.")])

    outcome = LabSession(ws, client=client).handle("Build it", user="Jordi")

    assert "scaffold_generator" in client.calls[0][2]
    assert outcome.build.changes == [] and "No files changed." in outcome.build.summary
    assert runner.calls == []


def test_without_shell_the_build_offers_no_run_command(ws, scripted_client, runner):
    start(ws)
    client = scripted_client([end_turn("Nothing to do.")])

    LabSession(ws, client=client, allow_shell=False).handle("Build it", user="Jordi")

    assert "run_command" not in client.calls[0][2] and "write_file" in client.calls[0][2]


def test_the_sessions_change_state_wins_over_what_code_run_in_the_build_wrote(ws, scripted_client, monkeypatch):
    # The file tools refuse .lab/, but run_command and the checks run code that can write there.
    change = start(ws)
    state = os.path.join(ws.root, ".lab", "changes", change.id, "state.json")
    with open(state, encoding="utf-8") as handle:
        tampered = json.load(handle)
    tampered.update(title="Hijacked", plan_approved=False, branches={REPO_BESSER: "development"})

    def tampering_check(spec, cwd, should_continue=None):
        with open(state, "w", encoding="utf-8") as handle:
            json.dump(tampered, handle)
        return 0, "4 passed"

    monkeypatch.setattr(checks, "default_runner", tampering_check)
    client = scripted_client([tool_use(*GREETING), end_turn("Done.")])

    LabSession(ws, client=client).handle("Build it", user="Jordi")

    stored = ws.active_change()
    assert (stored.title, stored.plan_approved, stored.branches) == (
        "Add a greeting", True, change.branches)


def test_the_checks_after_the_loop_stop_at_the_runs_time_limit(ws, scripted_client, runner, monkeypatch):
    start(ws)
    timeouts, clock = [], [0.0]
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + clock[0])

    def first_check_takes_the_rest_of_the_run(spec, cwd, should_continue=None):
        timeouts.append(spec.timeout_seconds)
        clock[0] = 120.0
        return 0, "ok"

    monkeypatch.setattr(checks, "default_runner", first_check_takes_the_rest_of_the_run)
    client = scripted_client([tool_use(*GREETING), end_turn("Done.")])

    outcome = LabSession(ws, client=client, limits=LabLimits(max_runtime_seconds=60)).handle("Build it", user="Jordi")

    # The first check may use only what the run has left, the others are not started at all.
    assert len(timeouts) == 1 and timeouts[0] <= 60
    first, *rest = outcome.build.checks
    assert first.passed and rest
    assert all(not c.passed and c.summary == "Not run: the run reached its 60 s limit" for c in rest)
    # The fix round that follows stops on the time limit before calling the model.
    assert outcome.stop_reason == "timeout" and len(client.calls) == 2


def test_a_stop_during_the_checks_after_the_loop_is_reported_as_cancelled(ws, scripted_client, monkeypatch):
    start(ws)
    ran = []

    def check_during_which_the_user_presses_stop(spec, cwd, should_continue=None):
        ran.append(spec.id)
        return 0, "4 passed"

    monkeypatch.setattr(checks, "default_runner", check_during_which_the_user_presses_stop)
    callbacks = LabCallbacks(should_continue=lambda: not ran)
    client = scripted_client([tool_use(*GREETING), end_turn("Done.")])

    outcome = LabSession(ws, client=client, callbacks=callbacks).handle("Build it", user="Jordi")

    assert ran == ["modeling_agent:ruff"]
    assert outcome.stop_reason == "cancelled"
    assert "## Build result (cancelled)" in ws.notes(ws.active_change())


def test_a_stop_during_the_models_own_run_checks_call_stops_its_checks_and_the_run(ws, scripted_client,
                                                                                   monkeypatch):
    start(ws)
    ran, phases = [], []

    def check_during_which_the_user_presses_stop(spec, cwd, should_continue=None):
        ran.append(spec.id)
        return 0, "4 passed"

    monkeypatch.setattr(checks, "default_runner", check_during_which_the_user_presses_stop)
    callbacks = LabCallbacks(should_continue=lambda: not ran, on_phase=lambda phase, text: phases.append(phase))
    client = scripted_client([tool_use(*GREETING), tool_use("run_checks", {}), end_turn("never sent")])

    outcome = LabSession(ws, client=client, callbacks=callbacks).handle("Build it", user="Jordi")

    # One check ran inside the tool call, its phase was streamed, and nothing ran after the stop.
    assert ran == ["modeling_agent:ruff"] and phases.count("check") == 1
    assert outcome.stop_reason == "cancelled" and len(client.calls) == 2


def test_a_stopped_build_reports_the_checks_that_ran_and_lets_the_runner_stop_a_check(ws, scripted_client,
                                                                                     monkeypatch):
    # Live run C (D3): ruff, import layers and tests ran in the model's run_checks call, then the user
    # stopped; the saved result said "Checks: none ran".
    start(ws)
    stopped, stops = [], []

    def checks_until_the_user_presses_stop(spec, cwd, should_continue=None):
        stops.append(should_continue)
        if spec.id == "modeling_agent:imports":
            stopped.append(True)  # the real runner kills the check here and returns STOPPED
            return checks.STOPPED, "Stopped: the run was stopped"
        time.sleep(0.02)
        return 0, "All checks passed!"

    monkeypatch.setattr(checks, "default_runner", checks_until_the_user_presses_stop)
    callbacks = LabCallbacks(should_continue=lambda: not stopped)
    client = scripted_client([tool_use(*GREETING), tool_use("run_checks", {}), end_turn("never sent")])

    outcome = LabSession(ws, client=client, callbacks=callbacks).handle("Build it", user="Jordi")

    assert outcome.stop_reason == "cancelled" and len(client.calls) == 2
    # The session's runner hands the run's stop to the check, so a running check can be ended.
    assert stops and all(stop is callbacks.should_continue for stop in stops)
    ruff, imports = outcome.build.checks
    assert ruff.spec_id == "modeling_agent:ruff" and ruff.passed and ruff.duration_seconds > 0
    assert imports.spec_id == "modeling_agent:imports" and imports.skipped
    assert imports.summary == "Stopped: the run was stopped before this check finished"
    assert "Checks: none ran" not in outcome.build.summary
    assert "Checks: 1 passed, 0 failed, 1 skipped." in ws.notes(ws.active_change())


def test_the_models_own_run_checks_call_stops_at_the_runs_time_limit(ws, scripted_client, monkeypatch):
    start(ws)
    timeouts, clock = [], [0.0]
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + clock[0])

    def first_check_takes_the_rest_of_the_run(spec, cwd, should_continue=None):
        timeouts.append(spec.timeout_seconds)
        clock[0] = 120.0
        return 0, "ok"

    monkeypatch.setattr(checks, "default_runner", first_check_takes_the_rest_of_the_run)
    client = scripted_client([tool_use(*GREETING), tool_use("run_checks", {}), end_turn("never sent")])

    outcome = LabSession(ws, client=client, limits=LabLimits(max_runtime_seconds=60)).handle("Build it", user="Jordi")

    # Inside the tool call: the first check is cut to the time left, the rest never start; so after the loop.
    assert len(timeouts) == 1 and timeouts[0] <= 60
    assert outcome.stop_reason == "timeout" and len(client.calls) == 2
    assert all(c.summary == "Not run: the run reached its 60 s limit" for c in outcome.build.checks)


def test_a_pending_plan_exchange_is_kept_in_the_workspaces_state_folder(template, tmp_path, scripted_client):
    root = tmp_path / "root"
    shutil.copytree(template, root)
    state_dir = tmp_path / "state"
    ws = LabWorkspace(str(root), state_dir=str(state_dir))
    client = scripted_client([end_turn("Which agent?"), end_turn("Which agent do you mean?"),
                              tool_use("submit_plan", PLAN)])
    session = LabSession(ws, client=client)

    session.handle("Make the agent greet", user="Jordi")

    assert (state_dir / "pending.json").is_file() and not (root / ".lab").exists()
    outcome = session.handle("The modeling agent", user="Jordi")
    assert "Make the agent greet" in client.calls[2][1][0]["content"]
    assert not (state_dir / "pending.json").exists() and not (root / ".lab").exists()
    assert (state_dir / "changes" / outcome.change.id / "notes.md").is_file()


def test_a_cancelled_build_runs_no_check(ws, scripted_client, runner):
    start(ws)
    client = scripted_client([end_turn("never sent")])
    callbacks = LabCallbacks(should_continue=lambda: False)

    outcome = LabSession(ws, client=client, callbacks=callbacks).handle("Build it", user="Jordi")

    assert outcome.stop_reason == "cancelled" and client.calls == [] and runner.calls == []
    assert outcome.build.checks == []


def test_build_without_an_active_change_and_unknown_actions_are_refused(ws, scripted_client):
    client = scripted_client([])
    session = LabSession(ws, client=client)

    with pytest.raises(LabWorkspaceError, match="no active change"):
        session.handle("Build it", user="Jordi", action="build")
    with pytest.raises(ValueError, match="Unknown lab action"):
        session.handle("Build it", user="Jordi", action="ship")
    assert client.calls == []


def test_phases_are_reported(ws, scripted_client, runner):
    phases = []
    client = scripted_client([tool_use("submit_plan", PLAN), tool_use(*GREETING), end_turn("Done.")])
    session = LabSession(ws, client=client, callbacks=LabCallbacks(on_phase=lambda p, m: phases.append(p)))

    session.handle("Greet the user", user="Jordi")
    outcome = session.handle("Go", user="Jordi", action="build")

    assert phases == ["plan", "build"] + ["check"] * len(outcome.build.checks)
