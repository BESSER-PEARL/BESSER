"""Requirements no cited line can show are reported as warnings, not blockers.

"The app has no sign-in" and "keep the existing editing working as before" cannot be
proved by quoting a line of code, and the judge answers both "unverified" ("the
excerpts do not establish absence"). They were blockers on every pass, so a working
app was reported as incomplete. Now an unverified one is a warning that says what was
checked, never a "met":

- an absence of sign-in or of agents is checked in the app's files and the warning
  names what was found or what was checked. A find is never a blocker: it cannot
  tell the excluded feature from a requested one ("no sign-in for visitors" beside
  an admin login). Nothing found is no proof of absence;
- a preservation names the files the judge named for it that are unchanged, or those
  that changed, since the previous version. Unchanged files do not show the
  behaviour still works.
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


def test_an_absence_found_absent_is_a_warning_saying_what_was_checked(app, monkeypatch):
    digests = _judge(monkeypatch, NO_SIGN_IN, UNVERIFIED)
    orch = _orchestrator(app, NO_SIGN_IN)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert len(digests) == 1  # no focused re-judge for an unverified absence
    # Nothing found is not proof: no check sees every way an app can sign users in.
    assert issue.startswith("requirement not provable: R1")
    assert _classify_issue(issue).severity == "warning"
    (verdict,) = orch._requirement_verdicts
    assert verdict["status"] == ledger.NOT_PROVABLE
    assert "no authentication library or sign-in route found" in verdict["note"]
    assert "no agent code or agent library found" in verdict["note"]


def _settled_absence(app, monkeypatch, text):
    requirement = {"id": 1, "text": text, "kind": "absence"}
    _judge(monkeypatch, requirement, UNVERIFIED)
    (issue,) = _requirement_issues(_orchestrator(app, requirement)._collect_requirement_issues())
    return issue


@pytest.mark.parametrize("path,text,found", [
    ("backend/agents/helpdesk/helpdesk.py", "agent = None\n", "agents folder backend/agents"),
    ("backend/support.py", "from baf.core.agent import Agent\n", "agent code backend/support.py"),
    ("backend/requirements.txt", "fastapi==0.115.0\npasslib[bcrypt]==1.7.4\n", "passlib in backend/requirements.txt"),
    ("frontend/package.json", '{"dependencies": {"react": "18.3.1", "next-auth": "4.24.0"}}',
     "next-auth in frontend/package.json"),
    ("frontend/package.json", '{"dependencies": {"react": "18.3.1", "@clerk/nextjs": "5.0.0"}}',
     "@clerk/nextjs in frontend/package.json"),
    ("frontend/package.json", '{"dependencies": {"react": "18.3.1", "passport-local": "1.0.0"}}',
     "passport-local in frontend/package.json"),
    ("backend/routers/session.py", "from fastapi import APIRouter\nrouter = APIRouter(prefix='/auth')\n\n\n"
                                   "@router.post('/login')\ndef login():\n    return {}\n",
     "route /auth/login in backend/routers/session.py"),
    ("backend/routers/account.py", "from fastapi import APIRouter\nrouter = APIRouter()\n\n\n"
                                   "@router.post('/api/v1/sign-in')\ndef enter():\n    return {}\n",
     "route /api/v1/sign-in in backend/routers/account.py"),
])
def test_a_find_is_a_warning_naming_it_never_a_blocker(app, monkeypatch, path, text, found):
    """A find cannot tell the excluded feature from a requested one ("No sign-in for visitors"
    beside an admin login); as a blocker it would send the repair loop to delete it."""
    _write(app, path, text)
    issue = _settled_absence(app, monkeypatch, NO_SIGN_IN["text"])
    assert issue.startswith("requirement not provable: R1") and found in issue
    assert _classify_issue(issue).severity == "warning"


# Requests that exclude sign-in or agents only in part, beside a requested admin login.
ADMIN_LOGIN = {
    "backend/requirements.txt": "fastapi==0.115.0\npasslib[bcrypt]==1.7.4\n",
    "backend/routers/admin.py": "from fastapi import APIRouter\nrouter = APIRouter()\n\n\n"
                                "@router.post('/admin/login')\ndef login():\n    return {}\n",
}


@pytest.mark.parametrize("text", [
    "No sign-in for visitors", "No sign-in for visitors, admins log in",
    "Without signing in, visitors browse the catalogue", "No sign-in or sign-up for customers",
    "No sign-in, but admins log in with a password", "The app has no sign-up", "There is no sign-up",
    "No passwords", "The app has no authorization", "No authorisation, every user can do everything", "No login",
    "The app has no login and no agents", "No login, no agents.", "Without login", "No auth needed",
    "There are no logins for guests", "No sign-up; accounts are created by an admin",
    "The site has no sign up and no chatbot", "No sign-in. Admins use the admin login page",
    "No sign in for shoppers", "no login",
    "No login required to browse products", "The app must not store passwords in plain text",
])
def test_an_absence_beside_a_requested_login_is_a_warning(app, monkeypatch, text):
    for path, content in ADMIN_LOGIN.items():
        _write(app, path, content)
    issue = _settled_absence(app, monkeypatch, text)
    assert issue.startswith("requirement not provable: R1")
    assert "route /admin/login" in issue or "passlib" in issue or "agent" in text.lower() or "chatbot" in text


@pytest.mark.parametrize("path", [
    "frontend/src/components/agents/AgentCard.tsx", "backend/routers/agents/crud.py", "backend/agents/routes.py",
])
def test_an_agent_entitys_own_code_is_not_named_as_an_agent(app, monkeypatch, path):
    """Agents are the generator's ``agents/<name>/`` folder, the agent framework or code using it,
    not the screens and routes of a domain class called Agent."""
    _write(app, path, "x = 1\n")
    issue = _settled_absence(app, monkeypatch, "The app has no chatbot")
    assert issue.startswith("requirement not provable: R1")
    assert "found in the app" not in issue
    assert "no agent code or agent library found" in issue


def test_the_chatbot_folder_of_a_requested_chatbot_is_named_not_blocked(app, monkeypatch):
    _write(app, "agents/helper/helper.py", "x = 1\n")
    issue = _settled_absence(app, monkeypatch, "The chatbot must not give medical advice")
    assert issue.startswith("requirement not provable: R1") and "agents folder agents" in issue


def test_an_absence_naming_anything_else_is_a_warning(app, monkeypatch):
    _judge(monkeypatch, NO_PAYMENTS, UNVERIFIED)
    orch = _orchestrator(app, NO_PAYMENTS)
    (issue,) = _requirement_issues(orch._collect_requirement_issues())
    assert issue.startswith("requirement not provable: R1")
    assert "no authentication library or sign-in route found" in issue
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


def test_a_missing_absence_naming_an_omitted_file_is_judged_again(app, monkeypatch):
    _write(app, "frontend/src/checkout.css", ".pay { color: green; }\n")
    digests = _judge(monkeypatch, NO_PAYMENTS,
                     {"status": "missing", "evidence": "", "note": "frontend/src/checkout.css styles a payment form"},
                     {"status": "missing", "evidence": "", "note": "the payment form is styled"})
    _orchestrator(app, NO_PAYMENTS)._collect_requirement_issues()
    assert len(digests) == 2 and "### frontend/src/checkout.css" in digests[1]


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


def test_the_report_explains_a_warning_and_a_found_absence():
    report = build_verification_report({"requirements": [
        dict(KEEP_EDITING, status=ledger.NOT_PROVABLE, note="Changed since the previous version: x.tsx."),
        dict(NO_SIGN_IN, id=2, status="missing", note="a support chatbot answers on every page"),
    ]})
    assert not report.verified
    assert [item.why for item in report.notVerified if item.kind == "requirement"] == [
        "Changed since the previous version: x.tsx."]
    (found,) = report.shippedUnenforced
    assert found.why == "the delivered app has what it excludes; a support chatbot answers on every page"
