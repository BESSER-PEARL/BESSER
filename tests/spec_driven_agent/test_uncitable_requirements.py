"""Requirements no cited line can show are reported as warnings, not blockers.

An absence ("the app has no data export") and a preservation ("editing keeps working as
it does now") cannot be proved by quoting a line of code, and the judge answers both
"unverified". An unverified one is a warning that it cannot be shown, never a "met":

- an absence says only that: no scan of the app's files can tell an excluded feature
  from a requested one, or see every way an app can provide it;
- a preservation names the files the judge named for it that are unchanged, or those
  that changed, since the previous version. Unchanged files do not show the
  behaviour still works.

The judge's own missing or partial verdict for either kind stays a blocker.
"""
import hashlib

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.user_wording import user_sentence
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

NO_EXPORT = {"id": 1, "text": "The app has no data export", "kind": "absence"}
KEEP_EDITING = {"id": 1, "text": "Editing and deleting a task keep working as they do now", "kind": "preservation"}
TABLE = "frontend/src/components/TaskTable.tsx"
API = "frontend/src/api.ts"
HOME = "frontend/src/pages/Home.tsx"
UNVERIFIED = {"status": "unverified", "evidence": "", "note": "the excerpts do not establish it",
              "inspection_paths": [TABLE]}


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - judge_coverage is replaced
        raise AssertionError("no LLM call expected")


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def app(tmp_path):
    _write(tmp_path, "backend/pydantic_classes.py", "class Task:\n    title: str\n")
    _write(tmp_path, "backend/requirements.txt", "fastapi==0.115.0\nsqlalchemy>=2.0\n")
    _write(tmp_path, "backend/routers/task.py", "from fastapi import APIRouter\nrouter = APIRouter()\n\n\n"
                                                "@router.get('/task/')\ndef tasks():\n    return []\n")
    _write(tmp_path, "frontend/package.json", '{"dependencies": {"react": "18.3.1", "axios": "1.7.0"}}')
    _write(tmp_path, API, "export const remove = (id: number) => fetch(`/task/${id}`, { method: 'DELETE' });\n")
    _write(tmp_path, TABLE, "import { remove } from '../api';\n\n"
                            "export const TaskTable = ({ rows }) => rows.map((r) => <button onClick={() => remove(r.id)} />);\n")
    _write(tmp_path, HOME, "import { TaskTable } from '../components/TaskTable';\n\n"
                           "export default function Home() {\n  return <TaskTable rows={[]} />;\n}\n")
    return tmp_path


def _orchestrator(app, requirement):
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Tasks", types={Class(name="Task")}),
                           output_dir=str(app), enable_tracing=False, enable_checkpointing=False)
    orch._instructions = requirement["text"]
    orch._generator_used = "generate_web_app"
    orch._requirements = [dict(requirement)]
    return orch


def _judge(monkeypatch, requirement, *answers):
    digests, script = [], list(answers)

    def judge(requirements, digest, client, **kwargs):
        digests.append(digest)
        return [dict(requirement, **script.pop(0))]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return digests


def _requirement_issues(issues):
    return [issue for issue in issues if issue.startswith("requirement")]


# ------------------------------------------------------------------ extraction


@pytest.mark.parametrize("kind", [ledger.ABSENCE_KIND, ledger.PRESERVATION_KIND])
def test_extraction_keeps_the_absence_and_preservation_kinds(kind):
    class _ToolClient:
        model = "tool-model"
        usage = UsageTracker("tool-model")
        _client = object()

        def chat(self, system, messages, tools, force_tool=None, model_override=None):
            block = type("B", (), {"type": "tool_use", "name": force_tool,
                                   "input": {"requirements": [{"text": "A requirement", "kind": kind}]}})()
            return {"stop_reason": "tool_use", "content": [block]}

    assert ledger.extract_requirements("Build a task list.", _ToolClient())[0]["kind"] == kind


# --------------------------------------------------------------------- absence


def test_an_unverified_absence_is_a_warning_that_it_cannot_be_shown(app, monkeypatch):
    digests = _judge(monkeypatch, NO_EXPORT, UNVERIFIED)
    orch = _orchestrator(app, NO_EXPORT)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert len(digests) == 1  # no focused re-judge for an unverified absence
    assert issue.startswith("requirement not provable: R1")
    assert _classify_issue(issue).severity == "warning"
    assert "not counted as a blocker" in user_sentence(issue)
    (verdict,) = orch._requirement_verdicts
    assert verdict["status"] == ledger.NOT_PROVABLE


PLAIN_ABSENCE_NOTE = "Something the app must not have cannot be shown by citing code. Not counted as a blocker."


@pytest.mark.parametrize("text", ["The app has no data export", "Customers place orders without logging in"])
def test_the_absence_note_does_not_depend_on_its_wording_or_the_apps_files(app, monkeypatch, text):
    """No scan of the files can tell an excluded feature from a requested one, so none is named."""
    _write(app, "backend/requirements.txt", "fastapi==0.115.0\npasslib[bcrypt]==1.7.4\n")
    _write(app, "backend/routers/session.py", "from fastapi import APIRouter\nrouter = APIRouter()\n\n\n"
                                              "@router.post('/login')\ndef login():\n    return {}\n")
    requirement = {"id": 1, "text": text, "kind": "absence"}
    _judge(monkeypatch, requirement, UNVERIFIED)
    orch = _orchestrator(app, requirement)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1")
    assert orch._requirement_verdicts[0]["note"] == PLAIN_ABSENCE_NOTE


def test_an_absence_the_judge_found_missing_stays_a_blocker(app, monkeypatch):
    _judge(monkeypatch, NO_EXPORT, {"status": "missing", "evidence": "", "note": "an export button downloads every order"})
    (issue,) = _requirement_issues(_orchestrator(app, NO_EXPORT)._collect_requirement_issues())
    assert issue.startswith("requirement: R1") and "export button" in issue


def test_a_missing_absence_naming_an_omitted_file_is_judged_again(app, monkeypatch):
    _write(app, "frontend/src/export.css", ".export { color: green; }\n")
    digests = _judge(monkeypatch, NO_EXPORT,
                     {"status": "missing", "evidence": "", "note": "frontend/src/export.css styles an export button"},
                     {"status": "missing", "evidence": "", "note": "the export button is styled"})
    _orchestrator(app, NO_EXPORT)._collect_requirement_issues()
    assert len(digests) == 2 and "### frontend/src/export.css" in digests[1]


# ---------------------------------------------------------------- preservation


def _seeded(app, requirement):
    orch = _orchestrator(app, requirement)
    orch._seed_source_hashes = ledger.source_hashes(str(app))
    return orch


def test_a_preservation_whose_files_are_unchanged_is_a_warning_not_met(app, monkeypatch):
    """Unchanged files do not show the behaviour still works: here the page no longer mounts the table."""
    orch = _seeded(app, KEEP_EDITING)
    _write(app, HOME, "export default function Home() {\n  return <h1>Tasks</h1>;\n}\n")  # this run's edit
    _judge(monkeypatch, KEEP_EDITING, UNVERIFIED, UNVERIFIED)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1")
    (verdict,) = orch._requirement_verdicts
    assert verdict["status"] == ledger.NOT_PROVABLE
    assert f"unchanged since the previous version: {TABLE}" in verdict["note"]


def test_a_preservation_whose_dependency_changed_is_a_warning(app, monkeypatch):
    orch = _seeded(app, KEEP_EDITING)
    _write(app, API, "export const remove = (id: number) => Promise.resolve(id);\n")
    _judge(monkeypatch, KEEP_EDITING, UNVERIFIED, UNVERIFIED)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1") and API in issue
    assert _classify_issue(issue).severity == "warning"


def test_a_preservation_naming_no_file_is_a_warning(app, monkeypatch):
    orch = _seeded(app, KEEP_EDITING)
    nothing = dict(UNVERIFIED, inspection_paths=[])
    _judge(monkeypatch, KEEP_EDITING, nothing, nothing)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1") and "no file implementing it was named" in issue


def test_without_a_previous_version_the_generated_files_are_the_baseline(app, monkeypatch):
    # The page mounting the table is one of the files it rests on.
    stored = {rel: (app / rel).read_bytes().decode("utf-8") for rel in (TABLE, API, HOME)}
    monkeypatch.setattr(ledger, "_generator_baselines", lambda output_dir, files: dict(stored))
    _judge(monkeypatch, KEEP_EDITING, UNVERIFIED, UNVERIFIED)
    orch = _orchestrator(app, KEEP_EDITING)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1") and "unchanged since the previous version" in issue
    assert hashlib.sha256(stored[TABLE].encode()).hexdigest() == ledger.source_hashes(str(app))[TABLE]


# ---------------------------------------------------------------------- report


def test_the_report_explains_a_warning_and_a_missing_absence():
    report = build_verification_report({"requirements": [
        dict(KEEP_EDITING, status=ledger.NOT_PROVABLE, note="Changed since the previous version: x.tsx."),
        dict(NO_EXPORT, id=2, status="missing", note="an export button downloads every order"),
    ]})
    assert not report.verified
    assert [item.why for item in report.notVerified if item.kind == "requirement"] == [
        "Changed since the previous version: x.tsx."]
    (found,) = report.shippedUnenforced
    assert found.why == "the delivered app has what it excludes; an export button downloads every order"

