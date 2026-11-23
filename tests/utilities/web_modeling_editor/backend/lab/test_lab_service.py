"""The lab service: start-up (preview origins, hidden folders, the sandbox policy), run admission, who may stop
and drop, the phase of a change, the home and a change's detail, the preview and memory refusals.

``make_service`` and ``lab_env`` are shared with ``test_lab_lifecycle.py`` and ``test_lab_costs.py``.
"""

from __future__ import annotations

import functools
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import besser
from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_ORDER, ChangeKind
from besser.spec_driven_agent.lab.environments import Environments
from besser.spec_driven_agent.lab.memory import LabMemory
from besser.spec_driven_agent.lab.sandbox_policy import lab_sandbox_mounts
from besser.spec_driven_agent.lab.session import LabSession
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabChangeCost,
    LabCheck,
    LabDone,
    LabNoteInput,
    LabNoteProposal,
    LabPlan,
    LabProposalRef,
    LabRunRecord,
)
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.audit import AuditEntry, AuditLog
from besser.utilities.web_modeling_editor.backend.services.lab.budget import LabBudget
from besser.utilities.web_modeling_editor.backend.services.lab import service as service_module
from besser.utilities.web_modeling_editor.backend.services.lab.service import (
    REFUSAL_STATUS,
    LabRefusal,
    LabService,
    LabUnavailableError,
    create_lab_service,
    lab_hidden_paths,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings
from tests.spec_driven_agent.lab.conftest import lab_layout, lab_layout_template, write  # noqa: F401

ALLOWED = "jordi,ana,ben,root"
ADMIN = "root"
LAB_ORIGIN = "https://lab.example.org"


def user(login: str) -> LabUser:
    return LabUser(login=login, session_id=f"session-{login}", access_token=f"token-{login}")


@pytest.fixture
def lab_env(lab_layout, tmp_path, monkeypatch):  # noqa: F811 - lab_layout is the imported fixture
    """The environment of a lab instance on ``lab_layout``, its stores in folders of their own; returns the layout."""
    stores = tmp_path / "stores"
    env = {
        "BESSER_LAB_ENABLED": "true",
        "BESSER_LAB_ALLOWED_LOGINS": ALLOWED,
        "BESSER_LAB_ADMIN_LOGINS": ADMIN,
        "BESSER_LAB_WORKSPACE_ROOT": lab_layout.root,
        "BESSER_LAB_GIT_ROOT": lab_layout.git_root,
        "BESSER_LAB_STATE_DIR": lab_layout.state_dir,
        "BESSER_LAB_MAX_CONCURRENT_RUNS": "2",
        "BESSER_LAB_BASE_REFRESH_MINUTES": "0",
        "BESSER_LAB_LLM_API_KEY": "sk-lab-test-key",
        "DEPLOYMENT_URL": LAB_ORIGIN,
        "OAUTH_SESSIONS_PATH": str(stores / "oauth" / "states.enc"),
        "USER_TOKENS_PATH": str(stores / "tokens" / "sessions.enc"),
        "BESSER_LLM_RUN_STORE_PATH": str(stores / "runs" / "runs.db"),
        "BESSER_LLM_RUN_WORKSPACE_ROOT": str(stores / "run-workspaces"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for folder in ("oauth", "tokens", "runs", "run-workspaces"):
        (stores / folder).mkdir(parents=True)
    return lab_layout


def make_service(workspace, settings: BackendSettings | None = None, *, preview=None) -> LabService:
    """A service on ``workspace`` as ``create_lab_service`` wires it, without a preview by default."""
    settings = settings or BackendSettings()
    audit = AuditLog(workspace.state_dir)
    policy = functools.partial(lab_sandbox_mounts, workspace_root=workspace.root, git_root=workspace.git_root,
                               hidden=tuple(lab_hidden_paths(settings, workspace)))
    return LabService(workspace, settings=settings,
                      environments=Environments(os.path.join(workspace.root, ".cache"), mode="hardlink"),
                      sandbox_policy=policy, preview=preview, audit=audit,
                      budget=LabBudget(audit, per_run_usd=settings.lab_max_cost_usd,
                                       user_monthly_usd=settings.lab_user_monthly_cap_usd,
                                       instance_monthly_usd=settings.lab_monthly_cap_usd),
                      memory=LabMemory(workspace.state_dir, workspace.root))


def record(run_id: str, login: str, *, usd: float = 1.0, step: str = "build", stop: str = "completed",
           turns: int = 3) -> LabRunRecord:
    return LabRunRecord(runId=run_id, step=step, login=login, model="claude-sonnet-5-5", usd=usd, turns=turns,
                        elapsedSeconds=60.0, stopReason=stop, startedAt=time.time())


def done(change_id: str, *, step: str = "build", checks=(), plan: LabPlan | None = None, proposals=(),
         stop: str = "completed") -> LabDone:
    return LabDone(step=step, changeId=change_id, title="Add a note element", kind=ChangeKind.EDITOR_ELEMENT,
                   repoChanges=[], checks=list(checks), checklist=[], summary="What changed", stopReason=stop,
                   cost=LabChangeCost(usd=0.0, turns=0, elapsedSeconds=0.0, runs=0), plan=plan, modelProposal=None,
                   noteProposals=list(proposals))


def run(service: LabService, change_id: str, login: str, *, usd: float = 1.0, step: str = "build",
        stop: str = "completed", result: LabDone | None = None, status: str = "done") -> str:
    """One whole run through the service, as the runner drives it; returns its id."""
    run_id = uuid.uuid4().hex
    ticket = service.begin_run(change_id, run_id, login)
    ticket.step = step
    service.end_run(ticket, status=status, record=record(run_id, login, usd=usd, step=step, stop=stop),
                    done=result, error=None)
    return run_id


def refused(code: str, call, *args, **kwargs) -> LabRefusal:
    with pytest.raises(LabRefusal) as caught:
        call(*args, **kwargs)
    assert (caught.value.code, caught.value.status) == (code, REFUSAL_STATUS[code])
    return caught.value


# -- Start-up: D1 test 10 ----------------------------------------------------------------------------------

def test_a_preview_url_on_the_lab_origin_disables_that_slot_with_a_reason(lab_env, monkeypatch):
    monkeypatch.setenv("BESSER_LAB_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("BESSER_LAB_PREVIEW_URLS", f"{LAB_ORIGIN}/,https://preview-2.lab.example.org")

    service = create_lab_service(BackendSettings())
    try:
        view = service.preview_view()
        assert view.enabled
        first, second = view.slots
        assert not first.enabled and first.url is None
        assert "lab's own origin" in first.detail and "BESSER_LAB_PREVIEW_URLS" in first.detail
        assert second.enabled and second.url == "https://preview-2.lab.example.org"
        assert service.home(user("ana")).previewSlots == 1
    finally:
        service.shutdown()


def test_every_slot_on_the_lab_origin_turns_the_preview_off(lab_env, monkeypatch):
    monkeypatch.setenv("BESSER_LAB_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("BESSER_LAB_PREVIEW_URLS", LAB_ORIGIN)

    service = create_lab_service(BackendSettings())
    try:
        view = service.preview_view()
        assert not view.enabled and view.detail
        refused("state", service.take_preview_slot, user("ana"), service.workspace.create_draft(owner="ana").id)
    finally:
        service.shutdown()


def test_hidden_paths_hold_the_state_session_token_and_run_store_folders(lab_env):
    settings = BackendSettings()
    workspace = lab_env.workspace()
    stores = Path(settings.oauth_sessions_path).parent.parent

    hidden = lab_hidden_paths(settings, workspace)

    assert hidden[0] == os.path.abspath(lab_env.state_dir)
    for folder in ("oauth", "tokens", "runs", "run-workspaces"):
        assert str(stores / folder) in hidden
    assert os.path.dirname(os.path.dirname(os.path.abspath(besser.__file__))) in hidden
    assert len(hidden) == len(set(hidden))


def test_unset_stores_resolve_to_the_temp_folder_their_stores_use(lab_env, monkeypatch):
    import tempfile
    for key in ("OAUTH_SESSIONS_PATH", "USER_TOKENS_PATH", "BESSER_LLM_RUN_STORE_PATH",
                "BESSER_LLM_RUN_WORKSPACE_ROOT"):
        monkeypatch.delenv(key)

    hidden = lab_hidden_paths(BackendSettings(), lab_env.workspace())

    assert os.path.abspath(tempfile.gettempdir()) in hidden


def test_the_policy_reaches_the_session_and_hides_the_session_store(lab_env):
    service = make_service(lab_env.workspace())
    draft = service.workspace.create_draft(owner="ana")

    session = LabSession(service.workspace, client=None, environments=service.environments,
                         sandbox_policy=service.sandbox_policy())
    mounts = session.sandbox_policy(service.workspace.change_repos(draft), writable=False)

    assert session.sandbox_policy is service.sandbox_policy()
    settings = service.settings
    for folder in (os.path.dirname(settings.oauth_sessions_path), os.path.dirname(settings.user_tokens_path),
                   os.path.dirname(settings.llm_run_store_path), lab_env.state_dir):
        assert os.path.realpath(folder) in mounts.masked
    assert os.path.realpath(lab_env.git_root) in mounts.read_only


def test_a_store_inside_a_working_tree_refuses_to_start_the_lab(lab_env, monkeypatch, caplog):
    inside = os.path.join(lab_env.base_path(REPO_BESSER), "sessions")
    os.makedirs(inside)
    monkeypatch.setenv("OAUTH_SESSIONS_PATH", os.path.join(inside, "states.enc"))

    with pytest.raises(LabUnavailableError, match="overlap"):
        create_lab_service(BackendSettings())
    assert "cannot start" in caplog.text


def test_the_sandbox_switched_off_on_linux_refuses_to_start_the_lab(lab_env, monkeypatch, caplog):
    """Model-written code never runs unconfined where a sandbox exists: the shell sandbox's ``off`` override
    (meant for a single-tenant development host) does not apply to a lab."""
    monkeypatch.setattr(service_module, "sandbox_supported_platform", lambda: True)
    monkeypatch.setenv("BESSER_LLM_SHELL_SANDBOX", "off")

    with pytest.raises(LabUnavailableError, match="BESSER_LLM_SHELL_SANDBOX=off"):
        create_lab_service(BackendSettings())
    assert "cannot start" in caplog.text


def test_base_clones_that_are_not_ready_refuse_to_start_the_lab_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("BESSER_LAB_ENABLED", "true")
    monkeypatch.setenv("BESSER_LAB_WORKSPACE_ROOT", str(tmp_path / "work"))

    with pytest.raises(LabUnavailableError, match="setup_workspace.sh"):
        create_lab_service(BackendSettings())


def test_the_lab_off_builds_no_service(monkeypatch):
    monkeypatch.delenv("BESSER_LAB_ENABLED", raising=False)
    assert create_lab_service(BackendSettings()) is None


def test_a_refusal_takes_its_status_from_its_code():
    assert LabRefusal("runs_full", "full").status == 429
    assert LabRefusal("not_owner", "no", 403).status == 403
    with pytest.raises(ValueError):
        LabRefusal("change_busy", "busy", 429)


# -- Run admission -------------------------------------------------------------------------------------------

def test_a_busy_change_refuses_a_second_run_naming_who_runs_it(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    ticket = service.begin_run(change.id, "r1", "ana")
    ticket.step = "build"

    refusal = refused("change_busy", service.begin_run, change.id, "r2", "ben")

    assert "@ana's build" in refusal.message
    assert [e.detail for e in service.audit.entries(service.audit.current_month())
            if e.action == "run_refused"] == ["change_busy"]
    service.end_run(ticket, status="done", record=record("r1", "ana"), done=None, error=None)
    service.begin_run(change.id, "r2", "ben")  # free again


def test_runs_beyond_the_cap_are_refused_naming_the_runs_in_progress(lab_env):
    service = make_service(lab_env.workspace())
    changes = [service.workspace.create_draft(owner=login) for login in ("ana", "ben", "jordi")]
    service.begin_run(changes[0].id, "r1", "ana")
    service.begin_run(changes[1].id, "r2", "ben")

    refusal = refused("runs_full", service.begin_run, changes[2].id, "r3", "jordi")

    assert "2 of 2" in refusal.message and "@ana" in refusal.message and "@ben" in refusal.message


def test_a_spent_user_month_refuses_with_user_cap(lab_env):
    service = make_service(lab_env.workspace())
    service.audit.append(AuditEntry(at=time.time(), login="ana", action="run", cost_usd=74.9))

    refused("user_cap", service.begin_run, service.workspace.create_draft(owner="ana").id, "r1", "ana")
    service.begin_run(service.workspace.create_draft(owner="ben").id, "r2", "ben")


def test_a_run_on_the_users_own_key_reserves_nothing_and_is_not_counted_towards_the_caps(lab_env):
    service = make_service(lab_env.workspace())
    service.audit.append(AuditEntry(at=time.time(), login="ana", action="run", cost_usd=74.9))
    change = service.workspace.create_draft(owner="ana")
    refused("user_cap", service.begin_run, change.id, "r1", "ana")

    ticket = service.begin_run(change.id, "r2", "ana", own_key=True)
    assert (ticket.own_key, ticket.cost_cap_usd) == (True, service.settings.lab_max_cost_usd)
    assert service.budget.meters("ana").user.reservedUsd == 0.0
    service.end_run(ticket, status="done", record=record("r2", "ana", usd=3.0), done=None, error=None)

    # Its cost is in the audit log and on the change, and the month's spending did not move.
    line = service.audit.entries(service.audit.current_month())[-1]
    assert (line.action, line.cost_usd, line.own_key) == ("run", 3.0, True)
    assert service.budget.meters("ana").user.spentUsd == pytest.approx(74.9)
    assert service.change_detail(change.id).cost.usd == pytest.approx(3.0)
    # A reloaded log does not count it either.
    assert AuditLog(service.workspace.state_dir).spent("ana") == pytest.approx(74.9)


def test_a_lab_without_a_key_runs_only_on_the_users_own(lab_env, monkeypatch):
    monkeypatch.delenv("BESSER_LAB_LLM_API_KEY")
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    assert service.home(user("ana")).labKey is False

    refusal = refused("state", service.begin_run, change.id, "r1", "ana")
    assert "no API key of its own" in refusal.message
    assert service.begin_run(change.id, "r2", "ana", own_key=True).own_key is True


def test_a_spent_lab_month_refuses_with_instance_cap(lab_env):
    service = make_service(lab_env.workspace())
    for login in ("ben", "jordi", "root", "x"):
        service.audit.append(AuditEntry(at=time.time(), login=login, action="run", cost_usd=74.95))

    refused("instance_cap", service.begin_run, service.workspace.create_draft(owner="ana").id, "r1", "ana")
    refused("instance_cap", service.create_change, user("ana"))


def test_an_unknown_or_closed_change_refuses_a_run(lab_env):
    service = make_service(lab_env.workspace())
    refused("not_found", service.begin_run, "0123456789ab", "r1", "ana")
    change = service.workspace.create_draft(owner="ana")
    service.workspace.save_change(replace(change, status="closed", outcome="merged", closed_at=time.time()))

    refused("state", service.begin_run, change.id, "r2", "ana")


def test_the_starter_the_owner_and_an_admin_may_stop_a_run(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="jordi")
    service.begin_run(change.id, "r1", "ana")

    assert service.can_stop("r1", "ana") is True
    assert service.can_stop("r1", "Jordi") is True
    assert service.can_stop("r1", ADMIN) is True
    assert service.can_stop("r1", "ben") is False
    assert service.can_stop("another-run", "ana") is None


# -- Change actions -----------------------------------------------------------------------------------------

def test_only_the_owner_or_an_admin_drops_a_change(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")

    refusal = refused("not_owner", service.drop, user("ben"), change.id)
    assert "@ana" in refusal.message
    assert service.workspace.get_change(change.id)

    other = service.workspace.create_draft(owner="ana")
    listed = service.drop(user(ADMIN), other.id)
    assert other.id not in [summary.changeId for summary in listed.changes]
    service.drop(user("ana"), change.id)
    assert service.changes().changes == []


def test_a_running_change_cannot_be_dropped(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    service.begin_run(change.id, "r1", "ana")

    refused("change_busy", service.drop, user("ana"), change.id)


def test_a_new_change_is_a_draft_owned_by_its_creator_and_audited(lab_env):
    service = make_service(lab_env.workspace())

    detail = service.create_change(user("ana"))

    assert (detail.status, detail.owner, detail.phase, detail.view, detail.thread) == ("draft", "ana", "draft",
                                                                                     None, [])
    entries = service.audit.entries(service.audit.current_month())
    assert [(e.action, e.change_id) for e in entries] == [("change_created", detail.changeId)]
    assert service.budget.meters("ana").user.reservedUsd == 0


# -- Phase -------------------------------------------------------------------------------------------------

def _failing() -> LabCheck:
    return LabCheck(id="pytest", repo="besser", title="Tests", passed=False, skipped=False, summary="1 failed")


@pytest.mark.parametrize("history, phase", [
    ([], "draft"),
    ([dict(step="plan", result="plan")], "planned"),
    ([dict(step="build", result="build")], "built"),
    ([dict(step="build", result="failing")], "checks_failing"),
    ([dict(step="build", result="build", stop="cost_cap")], "stopped"),
    ([dict(step="build", result="failing"), dict(step="plan", result="plan")], "planned"),
])
def test_the_phase_follows_the_last_run(lab_env, history, phase):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    results = {"plan": lambda: done(change.id, step="plan", plan=LabPlan(
                   kind=ChangeKind.EDITOR_ELEMENT, title="Add a note element", summary="s", steps=[], repos=[],
                   questions=[])),
               "build": lambda: done(change.id), "failing": lambda: done(change.id, checks=[_failing()])}
    for step in history:
        run(service, change.id, "ana", step=step["step"], stop=step.get("stop", "completed"),
            result=results[step["result"]]())

    assert service.change_detail(change.id).phase == phase


def test_running_in_review_merged_and_closed_phases(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    service.begin_run(change.id, "r1", "ana")
    assert service.change_detail(change.id).phase == "running"

    reviewed = service.workspace.create_draft(owner="ana")
    service.workspace.save_change(replace(reviewed, pull_requests={REPO_BESSER: 7}))
    assert service.change_detail(reviewed.id).phase == "in_review"

    for outcome in ("merged", "closed"):
        closed = service.workspace.create_draft(owner="ana")
        service.workspace.save_change(replace(closed, status="closed", outcome=outcome, closed_at=time.time()))
        detail = service.change_detail(closed.id)
        assert (detail.phase, detail.outcome, detail.status) == (outcome, outcome, "closed")


# -- Home and change detail ------------------------------------------------------------------------------------

def test_the_home_shows_bases_runs_meters_and_memory(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    ticket = service.begin_run(change.id, "r1", "ana")
    ticket.step = "plan"

    home = service.home(user(ADMIN))

    assert (home.user.login, home.user.isAdmin) == (ADMIN, True)
    assert [base.repo for base in home.bases] == list(REPO_ORDER)
    assert all(not base.fork and base.upstream.startswith("BESSER-PEARL/") and len(base.sha) == 40
               for base in home.bases)
    assert [(r.runId, r.changeId, r.login, r.step) for r in home.runs.inProgress] == [("r1", change.id, "ana",
                                                                                         "plan")]
    assert home.runs.cap == 2
    assert home.meters.instance.reservedUsd == ticket.cost_cap_usd and home.meters.user.reservedUsd == 0
    assert (home.previewSlots, home.linkMode) == (0, "hardlink")
    assert (home.memory.enabled, home.memory.count, home.memory.max) == (True, 0, 60)
    assert service.home(user("ana")).user.isAdmin is False


def test_change_detail_of_an_open_change(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.open_change(service.workspace.create_draft(owner="ana"), title="Add a note element",
                                           kind=ChangeKind.EDITOR_ELEMENT)
    tree = service.workspace.change_repos(change)[REPO_BESSER].path
    write(os.path.join(tree, "besser", "note.py"), "NOTE = 1\n")
    service.thread(change.id).add_message(login="ben", text="Add a note element", action="auto", model=None)

    detail = service.change_detail(change.id)

    assert (detail.status, detail.phase, detail.owner, detail.repos) == ("open", "planned", "ana", [REPO_BESSER])
    assert detail.branches == {key: "lab/ana/add-a-note-element" for key in REPO_ORDER}
    assert [(b.repo, b.behind, b.latest == b.sha) for b in detail.bases] == [(key, 0, True) for key in REPO_ORDER]
    assert detail.behind == {key: 0 for key in REPO_ORDER}
    assert detail.view.step == "plan" and [r.repo for r in detail.view.repoChanges] == [REPO_BESSER]
    assert [item.type for item in detail.thread] == ["message"]
    assert (detail.requestedBy, detail.runs, detail.run, detail.pullRequests) == (["ben"], [], None, [])

    summary = service.changes().changes[0]
    assert (summary.changeId, summary.repos, summary.behind) == (change.id, [REPO_BESSER], detail.behind)


def test_end_run_keeps_the_last_run_and_the_plan(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    plan = LabPlan(kind=ChangeKind.EDITOR_ELEMENT, title="Add a note element", summary="s", steps=["a"],
                   repos=[REPO_BESSER], questions=[])

    run(service, change.id, "ana", step="plan", result=done(change.id, step="plan", plan=plan))

    folder = Path(lab_env.state_dir, "changes", change.id)
    assert LabPlan.model_validate_json((folder / "plan.json").read_text(encoding="utf-8")) == plan
    assert json.loads((folder / "last_run.json").read_text(encoding="utf-8"))["step"] == "plan"
    detail = service.change_detail(change.id)
    assert detail.view.summary == "What changed"
    assert [(item.type, item.status, item.planVersion) for item in detail.thread] == [("run", "done", 1)]


# -- Memory and preview refusals --------------------------------------------------------------------------------

def _proposal(note_id: str) -> LabNoteProposal:
    return LabNoteProposal(id=note_id, kind="decision", title="Keep notes short", body="One idea per note.",
                           repos=[], changeKinds=[], docsTarget="", why="It came up twice.")


def test_a_note_approved_from_a_proposal_leaves_the_pending_proposals(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    run(service, change.id, "ana", result=done(change.id, proposals=[_proposal("p1"), _proposal("p2")]))
    note = LabNoteInput(kind="decision", title="Keep notes short", body="One idea per note.",
                        fromProposal=LabProposalRef(changeId=change.id, id="p1"))

    stored = service.add_note(user("ben"), note)

    assert (stored.approvedBy, stored.sourceChange, stored.stale) == ("ben", change.id, False)
    assert [p.id for p in service._read_proposals(change.id)] == ["p2"]
    refused("not_found", service.add_note, user("ben"), note)  # already approved
    service.dismiss_proposal(user("ben"), LabProposalRef(changeId=change.id, id="p2"))
    assert service._read_proposals(change.id) == []
    assert [n.id for n in service.memory_view().notes] == [stored.id]


def test_the_thread_lists_only_the_proposals_still_pending(lab_env):
    # W5-V finding 3: the run card rebuilt from the thread offered notes already approved or dismissed.
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    run(service, change.id, "ana",
        result=done(change.id, proposals=[_proposal("p1"), _proposal("p2"), _proposal("p3")]))
    service.add_note(user("ben"), LabNoteInput(kind="decision", title="Keep notes short", body="One idea per note.",
                                               fromProposal=LabProposalRef(changeId=change.id, id="p1")))
    service.dismiss_proposal(user("ben"), LabProposalRef(changeId=change.id, id="p3"))

    (item,) = service.change_detail(change.id).thread

    assert [proposal.id for proposal in item.lab.noteProposals] == ["p2"]


def test_memory_off_refuses_writes_with_the_reason(lab_env):
    # A state folder inside the workspace root: code the agent runs could plant a note there.
    service = make_service(LabWorkspace(lab_env.root, lab_env.specs(), state_dir=os.path.join(lab_env.root, "state"),
                                        git_root=lab_env.git_root))
    note = LabNoteInput(kind="decision", title="t", body="b")

    refusal = refused("state", service.add_note, user("ana"), note)

    assert "BESSER_LAB_STATE_DIR" in refusal.message
    view = service.memory_view()
    assert (view.enabled, view.notes) == (False, []) and view.reason == refusal.message
    assert service.home(user("ana")).memory.enabled is False
    refused("not_found", service.dismiss_proposal, user("ana"), LabProposalRef(changeId="0123456789ab", id="p"))


def test_preview_actions_without_a_preview_are_refused(lab_env):
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")

    assert service.preview_view().enabled is False
    refused("state", service.take_preview_slot, user("ana"), change.id)
    refused("state", service.touch_preview_slot, user("ana"), 0)
    refused("not_found", service.take_preview_slot, user("ana"), "0123456789ab")
