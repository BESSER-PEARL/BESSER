"""A non-web app is checked like a web app, and ships readable.

Live evidence (prod v8.6.1, run 0e794481, gpt-5.6-terra): "a local Windows 11
todo app" produced a Tkinter + SQLite desktop app. Lists, tags and people
could only be added (never edited or deleted), reminders were stored but
never fired, 87 statements were packed onto one line with semicolons, and
two buttons were cut off. Phase 3 reported none of it: the acceptance matrix
that measures a web app's entities does not read a native one.
"""

import os
import shutil

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.agent.prompt_builder import _other_target_rule
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger
from besser.spec_driven_agent.planning.agent_wiring import AGENT_DATA_KIND
from besser.spec_driven_agent.planning.target import Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker

DESKTOP = Target("other", "Windows 11 desktop todo app", True)
CLI = Target("other", "Python CLI", False)
WEB = Target("web_app", "todo web app", True)

# Shape of the shipped run: statements joined with semicolons.
_DENSE = "def toggle(item):\n    done=item['status']!='DONE'; item['status']='DONE' if done else 'TODO'; return item\n"


class _OfflineClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, *args, **kwargs):
        raise AssertionError("never called")


def _model():
    def entity(name):
        cls = Class(name=name)
        cls.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
        return cls

    base = entity("Base")
    base.is_abstract = True
    return DomainModel(name="Todos", types={entity("TodoItem"), entity("Tag"), base})


def _orch(tmp_path, target):
    orch = LLMOrchestrator(llm_client=_OfflineClient(), domain_model=_model(), output_dir=str(tmp_path))
    orch._instructions = "a local Windows 11 todo app"
    orch.target = target
    return orch


# -- 1. Each entity of a non-web app is judged, and reported, never blocked --

TAG = "Users can create, view, edit and delete Tag records from the Windows 11 desktop todo app's screens."
TODO = "Users can create, view, edit and delete TodoItem records from the Windows 11 desktop todo app's screens."


def test_every_entity_of_a_desktop_app_is_judged(tmp_path):
    judged = _orch(tmp_path, DESKTOP)._requirements_to_judge()
    assert [(r["kind"], r["text"]) for r in judged] == [("entity_crud", TAG), ("entity_crud", TODO)]


def test_web_and_screenless_targets_get_no_entity_checks(tmp_path):
    # A web app's entities are measured by the acceptance matrix; a CLI has no screens.
    for target in (WEB, CLI, None):
        assert _orch(tmp_path, target)._requirements_to_judge() == []


def test_entity_checks_are_not_planned_as_user_requirements(tmp_path):
    orch = _orch(tmp_path, DESKTOP)
    orch._requirements = []  # the user's request stated nothing testable
    assert orch._requirements_for_validation() == []
    assert "create, view, edit and delete" not in orch._planner_instructions(orch._instructions)


def test_a_missing_entity_path_is_a_warning_not_a_blocker():
    from besser.spec_driven_agent.validation.issues import _classify_issue
    issues = requirements_ledger.ledger_issues([
        {"id": 1, "kind": "entity_crud", "status": "partial", "text": TAG, "note": "tags can only be added"},
        {"id": 2, "kind": "entity_crud", "status": "implemented", "text": TODO},
    ])
    assert issues == [f"entity coverage: {TAG} (partial) (tags can only be added)"]
    assert _classify_issue(issues[0]).severity == "warning"


def test_the_judge_step_reports_entity_gaps_as_warnings(tmp_path, monkeypatch):
    from besser.spec_driven_agent.validation.issues import _classify_issue
    orch = _orch(tmp_path, DESKTOP)
    orch._requirements = []
    seen = []

    def judge(requirements, digest, client, **kwargs):
        seen.extend(r["text"] for r in requirements)
        return [{"id": r["id"], "text": r["text"], "kind": r["kind"], "evidence": "", "note": "",
                 "status": "partial" if "Tag" in r["text"] else "implemented"} for r in requirements]

    monkeypatch.setattr(requirements_ledger, "judge_coverage", judge)
    monkeypatch.setattr(requirements_ledger, "verify_evidence", lambda verdicts, output_dir: verdicts)
    issues = orch._collect_requirement_issues()
    assert seen == [TAG, TODO]
    assert [_classify_issue(i).severity for i in issues] == ["warning"]


# -- 2. LLM-written Python ships formatted -------------------------------------

needs_ruff = pytest.mark.skipif(not shutil.which("ruff"), reason="ruff is not installed")


def _write(tmp_path, rel, text):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@needs_ruff
def test_llm_written_python_is_formatted_and_still_compiles(tmp_path):
    orch = _orch(tmp_path, DESKTOP)
    app = _write(tmp_path, "todo_app.py", _DENSE)
    orch.tool_calls_log = [{"tool": "write_file", "input": {"path": "todo_app.py"}, "success": True}]
    orch._format_python_output()
    text = app.read_text(encoding="utf-8")
    assert ";" not in text
    compile(text, "todo_app.py", "exec")
    assert not orch._format_notes


@needs_ruff
def test_generator_files_edited_files_and_modify_runs_keep_their_layout(tmp_path):
    orch = _orch(tmp_path, DESKTOP)
    scaffold = _write(tmp_path, "backend/main_api.py", _DENSE)
    orch.executor._generator_files.add("backend/main_api.py")
    orch.tool_calls_log = [{"tool": "write_file", "input": {"path": "backend/main_api.py"}, "success": True}]
    orch._format_python_output()
    assert scaffold.read_text(encoding="utf-8") == _DENSE

    edited = _write(tmp_path, "edited.py", _DENSE)
    orch.tool_calls_log = [{"tool": "modify_file", "input": {"path": "edited.py"}, "success": True}]
    orch._format_python_output()
    assert edited.read_text(encoding="utf-8") == _DENSE

    user_file = _write(tmp_path, "user_code.py", _DENSE)
    orch.tool_calls_log = [{"tool": "write_file", "input": {"path": "user_code.py"}, "success": True}]
    orch._modify_mode = True
    orch._format_python_output()
    assert user_file.read_text(encoding="utf-8") == _DENSE


def test_missing_ruff_is_reported_not_silent(tmp_path, monkeypatch):
    orch = _orch(tmp_path, DESKTOP)
    _write(tmp_path, "todo_app.py", _DENSE)
    orch.tool_calls_log = [{"tool": "write_file", "input": {"path": "todo_app.py"}, "success": True}]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    orch._format_python_output()
    from besser.spec_driven_agent.validation.issues import _classify_issue
    assert [_classify_issue(n).severity for n in orch._format_notes] == ["warning"]
    assert "ruff format did not run" in orch._format_notes[0]


@needs_ruff
def test_a_path_outside_the_workspace_is_never_formatted(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text(_DENSE, encoding="utf-8")
    try:
        orch = _orch(tmp_path, DESKTOP)
        orch.tool_calls_log = [{"tool": "write_file", "input": {"path": f"../{outside.name}"}, "success": True},
                               {"tool": "write_file", "input": {"path": str(outside)}, "success": True}]
        orch._format_python_output()
        assert outside.read_text(encoding="utf-8") == _DENSE
    finally:
        os.remove(outside)


# -- 4. The non-web rule -------------------------------------------------------

def test_non_web_rule_asks_for_visible_controls_and_nothing_unrequested():
    rule = _other_target_rule(DESKTOP)
    assert "visible at the default window size" in rule
    assert "default window size" not in _other_target_rule(CLI)
    # Installers and data-folder policy are features: built only when asked for.
    assert "PyInstaller" not in rule and "LOCALAPPDATA" not in rule


def test_no_entity_checks_when_a_besser_generator_built_the_base(tmp_path):
    # Live: an agent-only run (generate_baf) got three "entity coverage" warnings.
    orch = _orch(tmp_path, DESKTOP)
    orch._generator_used = "generate_baf"
    assert orch._requirements_to_judge() == []


# -- A class model + an agent model: the agent must use the model's data -------
#
# Live (2026-10-30): "build my task assistant chatbot" on a class model + agent
# model shipped the BAF scaffold alone in 7 seconds; its add_task intent replied
# with canned text. The web and desktop runs wired their chat to the tasks.

def _agent():
    from besser.BUML.metamodel.state_machine.agent import Agent, Intent
    agent = Agent("TaskAssistant")
    for name in ("greet", "add_task", "list_tasks"):
        agent.intents.append(Intent(name=name, training_sentences=[name]))
    agent.new_state(name="idle", initial=True)
    return agent


def _agent_orch(tmp_path, files):
    for rel, text in files.items():
        _write(tmp_path, rel, text)
    return LLMOrchestrator(llm_client=_OfflineClient(), domain_model=_model(), agent_model=_agent(),
                           output_dir=str(tmp_path))


def _agent_task(orch):
    return [t for t in orch._deterministic_gap_tasks()
            if isinstance(t, str) and "Connect the TaskAssistant agent" in t]


def _agent_requirement(orch):
    return [r for r in orch._requirements_to_judge() if r["kind"] == AGENT_DATA_KIND]


def test_a_create_run_with_an_agent_and_a_class_model_gets_the_task(tmp_path):
    # Whatever the scaffold holds: whether the agent uses the data is the judge's
    # call. A name match passed shell bots whose classes were Message or User.
    orch = _agent_orch(tmp_path, {"baf/TaskAssistant.py": "from models import TodoItem\n"})
    assert _agent_task(orch)


def test_a_modify_run_is_judged_but_not_given_the_task_again(tmp_path):
    orch = _agent_orch(tmp_path, {})
    orch._modify_mode = True
    assert not _agent_task(orch)
    assert _agent_requirement(orch)


def test_the_judge_checks_the_agent_on_a_baf_scaffold(tmp_path):
    # The live failure was a generate_baf run, where entity checks are skipped.
    orch = _agent_orch(tmp_path, {})
    orch._generator_used = "generate_baf"
    [requirement] = _agent_requirement(orch)
    assert "TaskAssistant agent" in requirement["text"]
    assert "add_task ('add_task')" in requirement["text"]
    assert "Base, Tag, TodoItem records" in requirement["text"]


def test_no_agent_task_or_requirement_without_an_agent_model(tmp_path):
    orch = _orch(tmp_path, DESKTOP)
    assert not [t for t in orch._deterministic_gap_tasks() if "agent" in str(t).lower()]
    assert not [r for r in orch._requirements_to_judge() if r["kind"] == AGENT_DATA_KIND]


def test_an_agent_that_never_uses_the_data_is_a_blocker():
    from besser.spec_driven_agent.validation.issues import _classify_issue
    verdict = {"id": 4, "text": "The TaskAssistant agent handles ...", "kind": AGENT_DATA_KIND,
               "status": "missing", "note": "add_task only replies 'Task added!'"}
    [issue] = requirements_ledger.ledger_issues([verdict])
    assert _classify_issue(issue).severity == "blocker"


def test_an_import_is_not_evidence_that_the_agent_uses_the_data(tmp_path):
    _write(tmp_path, "bot.py", "from models import TodoItem\n\ndef add(session):\n    TodoItem.create(title=session.message)\n")
    base = {"id": 1, "text": "agent uses data", "kind": AGENT_DATA_KIND, "status": "implemented"}
    cited = lambda line: requirements_ledger.verify_evidence([dict(base, evidence=f"bot.py: {line}")], str(tmp_path))
    assert cited("from models import TodoItem")[0]["status"] == "unverified"
    assert cited("TodoItem.create(title=session.message)")[0]["status"] == "implemented"


def test_a_linked_parent_directory_is_never_formatted(tmp_path):
    # Review: the model writes pkg/x.py, then swaps pkg for a link to outside.
    from besser.spec_driven_agent.validation.toolchain import format_python_files
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_text(_DENSE, encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    try:
        os.symlink(outside, workspace / "pkg", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    assert format_python_files(str(workspace), ["pkg/x.py"]) == []
    assert (outside / "x.py").read_text(encoding="utf-8") == _DENSE


def test_native_ui_markup_reaches_the_judge(tmp_path):
    # Live (2026-10-31, WPF run): MainWindow.xaml held the Delete button, the
    # judge never saw it, and a correct requirement was reported unverified.
    for rel in ("MainWindow.xaml", "Views/Main.axaml", "qml/Main.qml",
                "app/src/main/res/layout/activity_main.xml", "app/src/main/res/layout-land/row.xml",
                "app/src/main/AndroidManifest.xml", "pom.xml"):
        _write(tmp_path, rel, "<Button Content=\"Delete\"/>\n")
    seen = set(requirements_ledger._source_files(str(tmp_path)))
    assert {"MainWindow.xaml", "Views/Main.axaml", "qml/Main.qml",
            "app/src/main/res/layout/activity_main.xml", "app/src/main/res/layout-land/row.xml"} <= seen
    assert "app/src/main/AndroidManifest.xml" not in seen and "pom.xml" not in seen
