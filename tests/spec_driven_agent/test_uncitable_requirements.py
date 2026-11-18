"""Requirements no cited line can show are settled without one, or reported as warnings.

"The app has no sign-in" and "keep the existing editing working as before" cannot be
proved by quoting a line of code, and the judge answers both "unverified" ("the
excerpts do not establish absence"). They were blockers on every pass, so a working
app was reported as incomplete. Now:

- an absence of sign-in or of agents is checked in the app's files: found, it is a
  blocker naming what was found; absent, the requirement is met; any other absence is
  a warning;
- a preservation is met when the files the judge named for it, and the files they rest
  on, are unchanged since the previous version; otherwise it is a warning.
"""
import hashlib

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.user_wording import user_sentence
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

NO_SIGN_IN = {"id": 1, "text": "The app has no sign-in and no agents", "kind": "absence"}
NO_PAYMENTS = {"id": 1, "text": "There is no login and no online payment", "kind": "absence"}
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


def test_an_absence_found_absent_is_met_without_another_judge_call(app, monkeypatch):
    digests = _judge(monkeypatch, NO_SIGN_IN, UNVERIFIED)
    orch = _orchestrator(app, NO_SIGN_IN)
    issues = orch._collect_requirement_issues()
    assert len(digests) == 1
    assert not _requirement_issues(issues), issues
    (verdict,) = orch._requirement_verdicts
    assert verdict["status"] == ledger.CHECKED
    assert "no authentication library or sign-in route" in verdict["note"]
    assert "no agents folder or agent library" in verdict["note"]


def test_agent_code_in_an_agents_folder_is_a_blocker(app, monkeypatch):
    _write(app, "backend/agents/helpdesk/helpdesk.py", "agent = None\n")
    _judge(monkeypatch, NO_SIGN_IN, UNVERIFIED)
    issues = _orchestrator(app, NO_SIGN_IN)._collect_requirement_issues()
    (issue,) = _requirement_issues(issues)
    assert issue.startswith("requirement: R1") and "agents folder backend/agents" in issue
    assert _classify_issue(issue).severity == "blocker"


@pytest.mark.parametrize("path,text,found", [
    ("backend/requirements.txt", "fastapi==0.115.0\npasslib[bcrypt]==1.7.4\n", "passlib in backend/requirements.txt"),
    ("frontend/package.json", '{"dependencies": {"react": "18.3.1", "next-auth": "4.24.0"}}',
     "next-auth in frontend/package.json"),
    ("backend/routers/session.py", "from fastapi import APIRouter\nrouter = APIRouter(prefix='/auth')\n\n\n"
                                   "@router.post('/login')\ndef login():\n    return {}\n",
     "route /auth/login in backend/routers/session.py"),
])
def test_an_authentication_library_or_route_is_a_blocker(app, monkeypatch, path, text, found):
    _write(app, path, text)
    _judge(monkeypatch, NO_SIGN_IN, UNVERIFIED)
    (issue,) = _requirement_issues(_orchestrator(app, NO_SIGN_IN)._collect_requirement_issues())
    assert issue.startswith("requirement: R1") and found in issue


def test_an_absence_naming_anything_else_is_a_warning(app, monkeypatch):
    _judge(monkeypatch, NO_PAYMENTS, UNVERIFIED)
    orch = _orchestrator(app, NO_PAYMENTS)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1")
    assert "no authentication library or sign-in route" in issue
    assert _classify_issue(issue).severity == "warning"
    assert "not counted as a blocker" in user_sentence(issue)


def test_an_author_is_not_authentication(app, monkeypatch):
    no_authors = {"id": 1, "text": "There are no author pages", "kind": "absence"}
    _judge(monkeypatch, no_authors, UNVERIFIED)
    (issue,) = _requirement_issues(_orchestrator(app, no_authors)._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1") and "authentication" not in issue


def test_an_absence_the_judge_found_missing_stays_a_blocker(app, monkeypatch):
    _judge(monkeypatch, NO_PAYMENTS, {"status": "missing", "evidence": "", "note": "a checkout page takes cards"})
    (issue,) = _requirement_issues(_orchestrator(app, NO_PAYMENTS)._collect_requirement_issues())
    assert issue.startswith("requirement: R1") and "checkout page" in issue


# ---------------------------------------------------------------- preservation


def _seeded(app, requirement):
    orch = _orchestrator(app, requirement)
    orch._seed_source_hashes = ledger.source_hashes(str(app))
    return orch


def test_a_preservation_whose_files_are_unchanged_is_met(app, monkeypatch):
    orch = _seeded(app, KEEP_EDITING)
    _write(app, HOME, "export default function Home() {\n  return <h1>Tasks</h1>;\n}\n")  # this run's edit
    _judge(monkeypatch, KEEP_EDITING, UNVERIFIED, UNVERIFIED)
    issues = orch._collect_requirement_issues()
    assert not _requirement_issues(issues), issues
    (verdict,) = orch._requirement_verdicts
    assert verdict["status"] == ledger.CHECKED and TABLE in verdict["note"]


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
    assert not _requirement_issues(orch._collect_requirement_issues())
    assert hashlib.sha256(stored[TABLE].encode()).hexdigest() == ledger.source_hashes(str(app))[TABLE]


# ---------------------------------------------------------------------- report


def test_the_report_lists_a_checked_requirement_as_verified_and_explains_a_warning():
    report = build_verification_report({"requirements": [
        dict(NO_SIGN_IN, status=ledger.CHECKED, note="Checked in the app's files: no agents folder."),
        dict(KEEP_EDITING, id=2, status=ledger.NOT_PROVABLE, note="Changed since the previous version: x.tsx."),
    ]})
    assert [item.how for item in report.verified] == ["Checked in the app's files: no agents folder."]
    assert [item.why for item in report.notVerified if item.kind == "requirement"] == [
        "Changed since the previous version: x.tsx."]
