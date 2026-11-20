"""BESSER Lab's service: the changes, their runs and ledgers, the preview pool and the team memory.

The lifespan builds it (:func:`create_lab_service`) and stores it on ``app.state.lab``; the lab routes and the
runner receive it through ``dependencies.get_lab_service``. Each change has a lock: a run holds it from
:meth:`LabService.begin_run` to :meth:`LabService.end_run`, a change action while it works. A change that is
held refuses (``change_busy``), it never waits; runs also take one of ``BESSER_LAB_MAX_CONCURRENT_RUNS`` slots
and reserve their cost cap against the monthly caps (``budget.py``).

Per change, beside its ``state.json``: ``runs.jsonl`` (one :class:`LabRunRecord` per run: cost and who asked),
``last_run.json`` (the last run's :class:`LabChangeView`), ``plan.json`` (the last plan), ``note_proposals.json``
(team notes the agent proposed, pending approval) and the thread (``thread.py``). The audit log (``audit.py``)
outlives dropped changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

from pydantic import BaseModel, ValidationError

import besser
from besser.spec_driven_agent.lab.contracts import (
    LAB_CACHE_DIRNAME,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    IssueRef,
    LabChange,
    NoteDraft,
    RepoChange,
)
from besser.spec_driven_agent.lab.environments import Environments, LabEnvironmentError, probe_link_mode
from besser.spec_driven_agent.lab.memory import (
    MAX_NOTES,
    LabMemory,
    LabMemoryFull,
    LabMemoryUnavailable,
    LabNoteNotFound,
)
from besser.spec_driven_agent.lab.memory import LabNote as MemoryNote
from besser.spec_driven_agent.lab.sandbox_policy import LabSandboxPolicyError, SandboxPolicy, lab_sandbox_mounts
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError, repo_specs_from_settings
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabAuditEntry,
    LabAuditResponse,
    LabBase,
    LabChangeCost,
    LabChangeDetail,
    LabChangesResponse,
    LabChangeSummary,
    LabChangeView,
    LabDiffResponse,
    LabDone,
    LabFileChange,
    LabHomeResponse,
    LabIssue,
    LabLimitsView,
    LabMemoryInfo,
    LabMemoryResponse,
    LabNote,
    LabNoteInput,
    LabNoteProposal,
    LabPlan,
    LabPreviewLog,
    LabPreviewSlot,
    LabPreviewView,
    LabProposalRef,
    LabPullRequest,
    LabRepoBaseStatus,
    LabRepoChange,
    LabReviewResponse,
    LabReviewTask,
    LabRun,
    LabRunRecord,
    LabRunsView,
    LabSettingsResponse,
    LabThreadItem,
    LabUpdateBranchResponse,
    LabUserView,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService
from besser.utilities.web_modeling_editor.backend.services.lab import diff_view, pull_requests, review
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser, allowed_logins, is_admin
from besser.utilities.web_modeling_editor.backend.services.lab.audit import (
    AuditEntry,
    AuditLog,
    append_json_line,
    read_json_lines,
)
from besser.utilities.web_modeling_editor.backend.services.lab.llm_proxy import create_llm_proxy_app, serve_llm_proxy
from besser.utilities.web_modeling_editor.backend.services.lab.preview import (
    PreviewFullError,
    PreviewPool,
    PreviewSandbox,
    PreviewUnavailableError,
)
from besser.utilities.web_modeling_editor.backend.services.lab.thread import ChangeThread
from besser.utilities.web_modeling_editor.backend.services.spec_driven.run_manager import default_store_path
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

if TYPE_CHECKING:  # budget.py imports LabRefusal from this module: LabBudget is imported where it is built
    from besser.utilities.web_modeling_editor.backend.services.lab.budget import LabBudget

logger = logging.getLogger(__name__)

#: The HTTP status of each refusal code (R13); the one place it is defined.
REFUSAL_STATUS = {
    "change_busy": 409, "runs_full": 429, "user_cap": 409, "instance_cap": 409, "not_owner": 403,
    "preview_full": 409, "not_found": 404, "state": 409,
}
#: The list of changes shows this many closed changes, newest first; older ones are in the audit log.
CLOSED_CHANGES_LISTED = 10
_PEARL_OWNER = "besser-pearl"
_RUNS_FILENAME = "runs.jsonl"
_LAST_RUN_FILENAME = "last_run.json"
_PLAN_FILENAME = "plan.json"
_PROPOSALS_FILENAME = "note_proposals.json"
_FINISHED = ("done", "failed", "cancelled")
_STEP_WORDS = {"plan": "plan", "build": "build", None: "run"}
_REPO_LABELS = {REPO_FRONTEND: "editor", REPO_BESSER: "BESSER", REPO_MODELING_AGENT: "modeling agent"}


class LabRefusal(Exception):
    """A lab action refused for a reason the user can act on.

    The route answers ``status`` with ``{"detail": {"code": code, "message": message}}``
    (``models.lab.LabRefusalResponse``).

    Args:
        code: One of ``models.lab.LabRefusalCode``.
        message: Written for the user; holds no secret.
        status: The HTTP status of the refusal; :data:`REFUSAL_STATUS` gives it, and a different one is a
            ``ValueError``.
    """

    def __init__(self, code: str, message: str, status: Optional[int] = None):
        expected = REFUSAL_STATUS[code]
        if status is not None and status != expected:
            raise ValueError(f"refusal {code!r} answers {expected}, not {status}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = expected


class LabUnavailableError(Exception):
    """The lab is enabled but cannot start; the message is what its routes answer with 503."""


@dataclass
class LabRunTicket:
    """A run that :meth:`LabService.begin_run` let start; :meth:`LabService.end_run` gives it back."""

    change_id: str
    run_id: str
    login: str
    #: the cost cap the run reserved: its ``max_cost_usd``
    cost_cap_usd: float
    started_at: float
    #: "plan" or "build" once the run has decided its step (the runner sets it); shown on the home's runs
    step: Optional[str] = None


@dataclass
class _Hold:
    """Who holds a change's lock: a run, or an action."""

    login: str
    #: "run", or the action ("drop", "close", ...)
    what: str
    title: str
    ticket: Optional[LabRunTicket] = None
    was_draft: bool = False


def repo_change_view(change: RepoChange) -> LabRepoChange:
    """A repository's changed files as the lab routes send them."""
    return LabRepoChange(repo=change.repo, branch=change.branch, base=change.base,
                         files=[LabFileChange(path=f.path, status=f.status) for f in change.files])


def lab_hidden_paths(settings: BackendSettings, workspace: LabWorkspace) -> list[str]:
    """The folders every lab sandbox masks besides the workspace root (D1 decision 6).

    The lab state folder, the folders of the OAuth and token stores, of the run store and the run workspaces
    (each where its store resolves an unset setting: the system temp folder), and the lab's own checkout (two
    levels above ``besser/__init__.py``).
    """
    def folder_of(store: Optional[str]) -> str:
        return os.path.dirname(os.path.abspath(store)) if store else tempfile.gettempdir()

    paths = [
        workspace.state_dir,
        folder_of(settings.oauth_sessions_path or settings.session_store_path),
        folder_of(settings.user_tokens_path or settings.session_store_path),
        folder_of(default_store_path(settings)),
        settings.llm_run_workspace_root or tempfile.gettempdir(),
        os.path.dirname(os.path.dirname(os.path.abspath(besser.__file__))),
    ]
    hidden: list[str] = []
    for path in paths:
        path = os.path.abspath(path)
        if path not in hidden:
            hidden.append(path)
    return hidden


async def run_preview_llm_proxy(settings: BackendSettings) -> None:
    """Serve the preview's LLM proxy until cancelled (a lifespan task); a port that cannot be bound is logged."""
    app = create_llm_proxy_app(api_key=settings.lab_preview_openai_api_key,
                               base_url=settings.lab_preview_openai_base_url,
                               max_requests_per_day=settings.lab_preview_llm_max_requests_per_day)
    try:
        await serve_llm_proxy(app, settings.lab_preview_llm_proxy_port)
    except RuntimeError as exc:
        logger.error("BESSER Lab: %s (BESSER_LAB_PREVIEW_LLM_PROXY_PORT); the preview assistant has no model "
                     "access until the backend restarts with a free port", exc)


class LabService:
    """The changes of a lab instance, their runs and ledgers, the preview pool and the team memory.

    Built by :func:`create_lab_service`; tests build it directly. Safe to call from several threads.
    """

    def __init__(self, workspace: LabWorkspace, *, settings: BackendSettings, environments: Environments,
                 sandbox_policy: SandboxPolicy, preview: Optional[PreviewPool], audit: AuditLog, budget: LabBudget,
                 memory: LabMemory, review_cache: Optional[review.ReviewCache] = None):
        self.workspace = workspace
        self.settings = settings
        self.environments = environments
        self.preview = preview
        self.audit = audit
        self.budget = budget
        self.memory = memory
        self.review_cache = review_cache or review.ReviewCache()
        self._policy = sandbox_policy
        self._lock = threading.Lock()
        #: change id -> who holds it
        self._holds: dict[str, _Hold] = {}
        #: run id -> the run's hold, while it runs
        self._runs: dict[str, _Hold] = {}
        # Orders the writers of each change's ledgers (runs.jsonl, note_proposals.json).
        self._files_lock = threading.Lock()
        #: when the bases were last fetched by the lab (0: not since start)
        self._fetched_at = 0.0
        self._closed = False

    # -- Policy and thread -------------------------------------------------------------------------------------

    def sandbox_policy(self) -> SandboxPolicy:
        """The mounts of a change's sandbox (shell, checks, installs, preview), hiding :func:`lab_hidden_paths`."""
        return self._policy

    def thread(self, change_id: str) -> ChangeThread:
        """The thread of change ``change_id``."""
        return ChangeThread(self.workspace.state_dir, change_id)

    # -- Runs ----------------------------------------------------------------------------------------------------

    def begin_run(self, change_id: str, run_id: str, login: str) -> LabRunTicket:
        """Let run ``run_id`` of ``login`` start on the change: its lock, a run slot, its cost cap reserved.

        Raises:
            LabRefusal: ``not_found``; ``state`` for a closed change; ``change_busy`` naming who holds it;
                ``runs_full`` at ``BESSER_LAB_MAX_CONCURRENT_RUNS``; ``user_cap`` or ``instance_cap``. Each
                refusal is written to the audit log (``run_refused``).
        """
        try:
            change = self._change(change_id)
            if change.status == "closed":
                raise LabRefusal("state", f'Change "{change.title}" is closed: start a new change to go on.')
            with self._lock:
                self._refuse_if_held(change_id)
                cap = self.settings.lab_max_concurrent_runs
                if len(self._runs) >= cap:
                    running = "; ".join(f'@{hold.login} on "{hold.title or "a new change"}"'
                                        for hold in self._runs.values())
                    raise LabRefusal("runs_full", f"{len(self._runs)} of {cap} lab runs are in progress "
                                                  f"({running}). Try again when one ends.")
                cost_cap = self.budget.reserve(login)
                ticket = LabRunTicket(change_id=change_id, run_id=run_id, login=login, cost_cap_usd=cost_cap,
                                      started_at=time.time())
                hold = _Hold(login=login, what="run", title=change.title, ticket=ticket,
                             was_draft=change.status == "draft")
                self._holds[change_id] = self._runs[run_id] = hold
            return ticket
        except LabRefusal as refusal:
            self._audit(login, "run_refused", change_id=change_id, run_id=run_id, detail=refusal.code)
            raise

    def end_run(self, ticket: LabRunTicket, *, status: str, record: LabRunRecord, done: Optional[LabDone],
                error: Optional[dict]) -> None:
        """Record how a run ended and give back what :meth:`begin_run` took.

        Writes the run's record (``runs.jsonl``), the thread's ``run`` line, the audit ``run`` line, then
        settles the budget (in that order, so the cost is never uncounted); on a run with a result also
        ``last_run.json``, ``plan.json`` for a plan and the proposed notes. Then releases the change and the
        run slot and lets the preview follow the build. A ledger that cannot be written is logged and the
        rest goes on; the release always happens.
        """
        if status not in _FINISHED:
            raise ValueError(f"not how a run ends: {status!r}")
        change_id = ticket.change_id
        try:
            self._logged("its run record", lambda: self._append_record(change_id, record))
            self._logged("its thread line", lambda: self.thread(change_id).run_finished(
                ticket.run_id, ticket.login, status=status, lab=done, error=error))
            self._logged("its audit line", lambda: self.audit.append(AuditEntry(
                at=time.time(), login=ticket.login, action="run", change_id=change_id, run_id=ticket.run_id,
                step=record.step, cost_usd=record.usd, stop_reason=record.stopReason, model=record.model,
                seconds=record.elapsedSeconds)))
            self.budget.settle(ticket.login, ticket.cost_cap_usd, record.usd)
            if done is not None:
                self._logged("its last run", lambda: self._remember(change_id, done))
            hold = self._runs.get(ticket.run_id)
            if hold is not None and hold.was_draft:
                self._audit_if_opened(ticket.login, change_id)
        finally:
            with self._lock:
                self._runs.pop(ticket.run_id, None)
                if self._holds.get(change_id) is not None and self._holds[change_id].ticket is ticket:
                    del self._holds[change_id]
        if self.preview is not None:
            self.preview.after_build(change_id)

    def can_stop(self, run_id: str, login: str) -> Optional[bool]:
        """Whether ``login`` may stop run ``run_id``: its starter, the change's owner or an admin.

        None when ``run_id`` is not a lab run in progress (the cancel route then behaves as without the lab).
        """
        with self._lock:
            hold = self._runs.get(run_id)
        if hold is None:
            return None
        who = login.strip().lower()
        if who == hold.login.lower() or is_admin(self.settings, login):
            return True
        try:
            owner = self.workspace.get_change(hold.ticket.change_id).owner
        except LabWorkspaceError:
            return False
        return who == owner.lower()

    # -- Reads ---------------------------------------------------------------------------------------------------

    def home(self, user: LabUser) -> LabHomeResponse:
        """The home: who is signed in, the bases, the runs in progress, the meters, the preview and the memory."""
        return LabHomeResponse(
            user=LabUserView(login=user.login, isAdmin=is_admin(self.settings, user.login)),
            bases=self._bases(), runs=LabRunsView(inProgress=self._runs_in_progress(),
                                                  cap=self.settings.lab_max_concurrent_runs),
            meters=self.budget.meters(user.login), previewSlots=self._preview_slots(),
            linkMode=self.environments.mode, memory=self._memory_info())

    def changes(self) -> LabChangesResponse:
        """Open changes and drafts, last updated first, then the newest :data:`CLOSED_CHANGES_LISTED` closed."""
        current = sorted(self.workspace.changes(statuses=("draft", "open")), key=lambda c: c.updated_at,
                         reverse=True)
        closed = sorted(self.workspace.changes(statuses=("closed",)), key=lambda c: c.closed_at,
                        reverse=True)[:CLOSED_CHANGES_LISTED]
        return LabChangesResponse(changes=[self._summary(change, self._records(change.id))
                                           for change in [*current, *closed]])

    def change_detail(self, change_id: str) -> LabChangeDetail:
        """One change: its summary, branches, bases, last run, thread and run records.

        Raises:
            LabRefusal: ``not_found``.
        """
        change = self._change(change_id)
        records = self._records(change_id)
        repo_changes = self._repo_changes(change)
        statuses = self._base_statuses(change) if change.status != "draft" else []
        summary = self._summary(change, records, repo_changes=repo_changes,
                                behind={status.repo: status.behind for status in statuses})
        return LabChangeDetail(
            **summary.model_dump(), branches=dict(change.branches), bases=statuses,
            view=self._view(change, records, repo_changes), thread=self._thread_items(change_id, records),
            runs=records)

    def diff(self, change_id: str, repo: Optional[str], path: Optional[str]) -> LabDiffResponse:
        """The change's working trees against their bases (blocking: git).

        Raises:
            LabRefusal: ``not_found``.
            LabWorkspaceError: The change is closed, or git failed.
            ValueError: ``path`` without ``repo``.
        """
        return diff_view.diff_response(self.workspace, self._change(change_id), repo=repo, path=path)

    async def review(self, user: LabUser, change_id: str) -> LabReviewResponse:
        """The change's pull requests as GitHub reports them, read with ``user``'s token (cached briefly).

        Raises:
            LabRefusal: ``not_found``.
            PullRequestError: GitHub refused or could not be reached (``PullRequestSignInError``: sign in again).
        """
        change = await asyncio.to_thread(self._change, change_id)
        return await review.fetch_review(self.workspace, change, GitHubService(user.access_token),
                                         self.review_cache)

    async def review_task_message(self, user: LabUser, change_id: str, task: LabReviewTask, user_text: str) -> str:
        """The build step's message for a review task, from a fresh read of the pull requests."""
        change = await asyncio.to_thread(self._change, change_id)
        status = await review.fetch_review(self.workspace, change, GitHubService(user.access_token),
                                           self.review_cache, fresh=True)
        return review.review_message(status, task, user_text)

    # -- Change actions ------------------------------------------------------------------------------------------

    def create_change(self, user: LabUser) -> LabChangeDetail:
        """A new draft owned by ``user``.

        Raises:
            LabRefusal: ``user_cap`` or ``instance_cap`` when no run could start this month.
        """
        # A trial reservation, given back at once: the caps' one rule and wording.
        self.budget.settle(user.login, self.budget.reserve(user.login), 0.0)
        change = self.workspace.create_draft(owner=user.login)
        self._audit(user.login, "change_created", change_id=change.id)
        return self.change_detail(change.id)

    def drop(self, user: LabUser, change_id: str) -> LabChangesResponse:
        """Delete the change: its trees, branches and state; its cost stays in the audit log.

        Raises:
            LabRefusal: ``not_found``; ``not_owner`` unless ``user`` owns it or is an admin; ``change_busy``.
        """
        change = self._change(change_id)
        if change.owner.lower() != user.login.lower() and not is_admin(self.settings, user.login):
            raise LabRefusal("not_owner", f'Only @{change.owner}, who started "{change.title or "this change"}", '
                                          "or a lab admin can drop it.")
        with self._holding(change_id, user.login, "drop", change.title):
            if self.preview is not None:
                self.preview.release(change_id)
            self.workspace.drop(self._change(change_id))
            self._audit(user.login, "change_dropped", change_id=change_id, detail=change.title)
        return self.changes()

    async def close(self, user: LabUser, change_id: str) -> LabChangeDetail:
        """Close the change once GitHub has its pull requests merged or closed: trees and branches go, state stays.

        The outcome is ``merged`` when every pull request is merged, else ``closed``; a notice in the thread
        says it to every viewer. A change already closed (by another viewer) answers its detail unchanged.

        Raises:
            LabRefusal: ``not_found``; ``change_busy``; ``state`` while it may not close (a pull request open,
                work not on GitHub; the message says which).
            PullRequestError: GitHub refused or could not be reached.
        """
        change = await asyncio.to_thread(self._change, change_id)
        with self._holding(change_id, user.login, "close", change.title):
            change = await asyncio.to_thread(self._change, change_id)  # another close may have ended meanwhile
            if change.status == "closed":
                return await asyncio.to_thread(self.change_detail, change_id)
            status = await review.fetch_review(self.workspace, change, GitHubService(user.access_token),
                                               self.review_cache, fresh=True)
            if not status.closable:
                raise LabRefusal("state", f'Change "{change.title}" cannot close yet: {status.closableNote}')
            outcome = "merged" if status.lifecycle == "merged" else "closed"
            await asyncio.to_thread(self._close, user, change_id, outcome)
        return await asyncio.to_thread(self.change_detail, change_id)

    async def update_from_base(self, user: LabUser, change_id: str, repo: str) -> LabUpdateBranchResponse:
        """Bring the change's ``repo`` up to its base: GitHub's update-branch with an open pull request, else a
        local merge (conflicts are left for a "Resolve conflicts" build step). The outcome goes in the thread.

        Raises:
            LabRefusal: ``not_found``; ``change_busy``; ``state`` for a change that is not open.
            PullRequestError: GitHub refused or could not be reached, or the merge was refused.
        """
        change = await asyncio.to_thread(self._change, change_id)
        self._require_open(change, "be updated from its base")
        with self._holding(change_id, user.login, "update", change.title):
            result = await review.update_branch(self.workspace, change, repo, access_token=user.access_token,
                                                github=GitHubService(user.access_token))
            await asyncio.to_thread(self._after_update, user, change_id, result)
        return result

    async def open_pull_requests(self, user: LabUser, change_id: str, *, title: Optional[str], body: Optional[str],
                                 closes_issues: Sequence[Any]) -> tuple[list[LabPullRequest], str]:
        """Push the change and open or update its draft pull requests with ``user``'s token and login.

        ``closes_issues`` (``repo``, ``number`` each) are the change's issues the pull requests close; empty
        keeps the stored ticks. Returns the pull requests and the note on the checks the change passed.

        Raises:
            LabRefusal: ``not_found``; ``change_busy``; ``state`` for a change that is not open.
            PullRequestError: Nothing to open, an issue that is not the change's, or a git or GitHub failure.
        """
        change = await asyncio.to_thread(self._change, change_id)
        self._require_open(change, "open pull requests")
        with self._holding(change_id, user.login, "pull requests", change.title):
            plan, last, records, requested_by = await asyncio.to_thread(self._pull_request_inputs, change)
            builds = [record for record in records if record.step == "build"]
            opened, note = await pull_requests.open_pull_requests(
                self.workspace, change, login=user.login, access_token=user.access_token, title=title, body=body,
                closes_issues=[IssueRef(issue.repo, issue.number) for issue in closes_issues] or None,
                plan=plan, last=last, model=builds[-1].model if builds else "", requested_by=requested_by)
            views = [LabPullRequest(repo=info.repo, number=info.number, url=info.url, branch=info.branch,
                                    base=info.base) for info in opened]
            await asyncio.to_thread(self._after_pull_requests, user, change_id, views, note)
        return views, note

    # -- Preview -------------------------------------------------------------------------------------------------

    def preview_view(self) -> LabPreviewView:
        """Every preview slot, or the preview off with the reason."""
        if self.preview is None:
            return LabPreviewView(enabled=False, slots=[],
                                  detail="The preview is off on this lab (BESSER_LAB_PREVIEW_ENABLED is not set).")
        return self.preview.view()

    def take_preview_slot(self, user: LabUser, change_id: str) -> LabPreviewSlot:
        """Show the change in a preview slot (prepared in the background).

        Raises:
            LabRefusal: ``not_found``; ``preview_full`` naming each slot's change and user; ``state`` when the
                preview is off or the change has no working trees.
        """
        change = self._change(change_id)
        try:
            slot = self._pool().take(change, user.login)
        except PreviewFullError as exc:
            raise LabRefusal("preview_full", str(exc)) from None
        except PreviewUnavailableError as exc:
            raise LabRefusal("state", str(exc)) from None
        self._audit(user.login, "preview_slot_taken", change_id=change_id, detail=f"slot {slot.slot + 1}")
        return slot

    def touch_preview_slot(self, user: LabUser, slot: int) -> None:
        """``user`` still uses ``slot``. Raises :class:`LabRefusal` ``not_found`` or ``state`` (preview off)."""
        with self._unknown_slot(slot):
            self._pool().touch(slot, user.login)

    def restart_preview(self, user: LabUser, slot: int, service: Optional[str]) -> LabPreviewView:
        """Restart one service of ``slot``, or all three. Raises :class:`LabRefusal` ``not_found`` or ``state``."""
        with self._unknown_slot(slot, service):
            view = self._pool().restart(slot, service)
        logger.info("BESSER Lab: @%s restarted %s of preview slot %d", user.login, service or "every service",
                    slot + 1)
        return view

    def preview_log(self, slot: int, service: str, lines: int) -> LabPreviewLog:
        """The end of one service's log, redacted. Raises :class:`LabRefusal` ``not_found`` or ``state``."""
        with self._unknown_slot(slot, service):
            return self._pool().log_tail(slot, service, lines)

    # -- Settings, bases and audit -------------------------------------------------------------------------------

    def settings_view(self, user: LabUser) -> LabSettingsResponse:
        """The repositories, limits, admins, this month's spending and the link mode."""
        settings = self.settings
        return LabSettingsResponse(
            bases=self._bases(),
            limits=LabLimitsView(perRunUsd=settings.lab_max_cost_usd, userMonthlyUsd=settings.lab_user_monthly_cap_usd,
                                 instanceMonthlyUsd=settings.lab_monthly_cap_usd,
                                 maxConcurrentRuns=settings.lab_max_concurrent_runs,
                                 previewSlots=self._preview_slots(),
                                 baseRefreshMinutes=settings.lab_base_refresh_minutes),
            admins=self.admins(), meters=self.budget.meters(user.login), linkMode=self.environments.mode)

    def admins(self) -> list[str]:
        """The admin logins that are also allowed, sorted (an admin who is not allowed is warned about once)."""
        return [login for login in sorted(allowed_logins(self.settings)) if is_admin(self.settings, login)]

    def refresh_bases(self, user: LabUser) -> LabSettingsResponse:
        """Fetch every base branch now (blocking). Open changes do not move.

        Raises:
            LabRefusal: ``state`` when a fetch fails; the message names the repository.
        """
        self._refresh_bases(user.login)
        return self.settings_view(user)

    async def refresh_loop(self) -> None:
        """Fetch the bases every ``BESSER_LAB_BASE_REFRESH_MINUTES`` (a lifespan task); 0 never does."""
        minutes = self.settings.lab_base_refresh_minutes
        if minutes == 0:
            logger.info("BESSER Lab: bases refresh by hand and when a change opens (BESSER_LAB_BASE_REFRESH_MINUTES=0)")
            return
        while not self._closed:
            await asyncio.sleep(minutes * 60)
            try:
                await asyncio.to_thread(self._refresh_bases, "")
            except LabRefusal as refusal:
                logger.warning("BESSER Lab: the scheduled refresh of the bases failed: %s", refusal.message)

    def audit_view(self, month: Optional[str]) -> LabAuditResponse:
        """One UTC month of the audit log (this month by default), with what each login's runs cost.

        Raises:
            ValueError: ``month`` is not ``YYYY-MM``.
        """
        month = month or self.audit.current_month()
        entries = self.audit.entries(month)
        totals: dict[str, float] = {}
        for entry in entries:
            if entry.spent_usd:
                totals[entry.login] = round(totals.get(entry.login, 0.0) + entry.spent_usd, 6)
        return LabAuditResponse(month=month, totalsByLogin=totals, entries=[
            LabAuditEntry(at=entry.at, login=entry.login, action=entry.action, changeId=entry.change_id,
                          runId=entry.run_id, step=entry.step, costUsd=entry.cost_usd, stopReason=entry.stop_reason,
                          detail=entry.detail) for entry in entries])

    # -- Team memory ---------------------------------------------------------------------------------------------

    def memory_view(self) -> LabMemoryResponse:
        """Every approved note; with the memory off (or its file unreadable), none and the reason."""
        try:
            notes = self.memory.notes()
        except LabMemoryUnavailable as exc:
            return LabMemoryResponse(enabled=False, reason=str(exc), notes=[])
        return LabMemoryResponse(enabled=True, reason="", notes=[_note_view(note) for note in notes])

    def add_note(self, user: LabUser, note: LabNoteInput) -> LabNote:
        """Store a note approved by ``user``; one approved from a proposal leaves the change's pending proposals.

        Raises:
            LabRefusal: ``state`` (memory off or full), ``not_found`` (the proposal).
            LabMemoryError: The note is not valid (the message lists why).
        """
        source = note.fromProposal
        if source is not None:
            self._proposal(source)  # refused before anything is stored
        with _memory_refusals():
            stored = self.memory.add(_draft(note), user.login, source_change=source.changeId if source else "")
        if source is not None:
            self._remove_proposal(source)
        return _note_view(stored)

    def update_note(self, user: LabUser, note_id: str, note: LabNoteInput) -> LabNote:
        """Replace a note's text and scope. Raises :class:`LabRefusal` ``state`` or ``not_found``."""
        with _memory_refusals():
            stored = self.memory.update(note_id, _draft(note))
        logger.info("BESSER Lab: @%s edited team note %s", user.login, note_id)
        return _note_view(stored)

    def delete_note(self, user: LabUser, note_id: str) -> None:
        """Delete a note. Raises :class:`LabRefusal` ``state`` or ``not_found``."""
        with _memory_refusals():
            self.memory.delete(note_id)
        logger.info("BESSER Lab: @%s deleted team note %s", user.login, note_id)

    def dismiss_proposal(self, user: LabUser, ref: LabProposalRef) -> None:
        """Drop a proposed note without storing it. Raises :class:`LabRefusal` ``not_found``."""
        self._proposal(ref)
        self._remove_proposal(ref)
        logger.info("BESSER Lab: @%s dismissed note proposal %s of change %s", user.login, ref.id, ref.changeId)

    def shutdown(self) -> None:
        """Stop the preview's services and the base refresh (the backend is stopping)."""
        self._closed = True
        if self.preview is not None:
            self.preview.stop()

    # -- Internals: changes and locks ----------------------------------------------------------------------------

    def _change(self, change_id: str) -> LabChange:
        try:
            return self.workspace.get_change(change_id)
        except LabWorkspaceError as exc:
            if str(exc).startswith(("no change", "invalid change id")):
                raise LabRefusal("not_found", f"There is no change {change_id}.") from None
            raise

    @staticmethod
    def _require_open(change: LabChange, action: str) -> None:
        if change.status != "open":
            reason = ("it is a draft: it gets its branches when its plan is submitted" if change.status == "draft"
                      else "it is closed")
            raise LabRefusal("state", f'Change "{change.title or change.id}" cannot {action}: {reason}.')

    def _refuse_if_held(self, change_id: str) -> None:
        """Raise ``change_busy`` naming the holder (lock held)."""
        hold = self._holds.get(change_id)
        if hold is None:
            return
        if hold.ticket is not None:
            step = _STEP_WORDS.get(hold.ticket.step, "run")
            message = (f"@{hold.login}'s {step} of this change is in progress. Wait for it to end, or stop it, "
                       "then try again.")
        else:
            message = f"@{hold.login} is working on this change ({hold.what}). Try again in a moment."
        raise LabRefusal("change_busy", message)

    @contextlib.contextmanager
    def _holding(self, change_id: str, login: str, what: str, title: str) -> Iterator[None]:
        """Hold the change for an action; refused (never waits) while a run or another action holds it."""
        with self._lock:
            self._refuse_if_held(change_id)
            hold = self._holds[change_id] = _Hold(login=login, what=what, title=title)
        try:
            yield
        finally:
            with self._lock:
                if self._holds.get(change_id) is hold:
                    del self._holds[change_id]

    def _running(self, change_id: str) -> Optional[LabRun]:
        with self._lock:
            hold = self._holds.get(change_id)
        if hold is None or hold.ticket is None:
            return None
        return _run_view(hold.ticket)

    def _runs_in_progress(self) -> list[LabRun]:
        with self._lock:
            tickets = [hold.ticket for hold in self._runs.values()]
        return sorted((_run_view(ticket) for ticket in tickets), key=lambda run: run.startedAt)

    def _close(self, user: LabUser, change_id: str, outcome: str) -> None:
        change = self._change(change_id)  # a fresh copy: the review may have saved this change
        try:
            self.workspace.close(change, outcome=outcome)
        except LabWorkspaceError as exc:
            raise LabRefusal("state", f'Change "{change.title}" cannot close: {exc}') from None
        if self.preview is not None:
            self.preview.release(change_id)
        how = "was merged" if outcome == "merged" else "was closed without being merged"
        self.thread(change_id).add_notice(user.login, f'Change "{change.title}" {how}; the lab closed it.')
        self._audit(user.login, "change_closed", change_id=change_id, detail=outcome)

    def _after_update(self, user: LabUser, change_id: str, result: LabUpdateBranchResponse) -> None:
        self.thread(change_id).add_notice(user.login, result.message)
        self._audit(user.login, "base_updated", change_id=change_id, detail=f"{result.repo}: {result.outcome}")

    def _after_pull_requests(self, user: LabUser, change_id: str, views: list[LabPullRequest], note: str) -> None:
        self.thread(change_id).add_pull_requests(user.login, views, note)
        self._audit(user.login, "pull_requests_opened", change_id=change_id,
                    detail=", ".join(f"{view.repo} #{view.number}" for view in views))

    def _pull_request_inputs(self, change: LabChange):
        """The change's plan, last run, run records and the logins that asked for it, for the lab block."""
        records = self._records(change.id)
        return (self._read_model(change.id, _PLAN_FILENAME, LabPlan),
                self._read_model(change.id, _LAST_RUN_FILENAME, LabChangeView), records,
                self._requested_by(change.id, records))

    def _refresh_bases(self, login: str) -> None:
        try:
            self.workspace.refresh_bases()
        except LabWorkspaceError as exc:
            raise LabRefusal("state", f"Fetching the base branches failed: {exc}") from None
        self._fetched_at = time.time()
        self._audit(login, "bases_refreshed", detail="by hand" if login else "scheduled")

    # -- Internals: views ---------------------------------------------------------------------------------------

    def _summary(self, change: LabChange, records: list[LabRunRecord], *,
                 repo_changes: Optional[list[LabRepoChange]] = None,
                 behind: Optional[dict[str, int]] = None) -> LabChangeSummary:
        """The change as the list shows it; ``behind`` is read for open changes only when not given."""
        if repo_changes is None:
            repo_changes = self._repo_changes(change)
        if behind is None:
            behind = self.workspace.base_status(change) if change.status == "open" else {}
        run = self._running(change.id)
        closed = change.status == "closed"
        return LabChangeSummary(
            changeId=change.id, title=change.title, kind=change.kind, status=change.status,
            outcome=change.outcome if closed and change.outcome else None, owner=change.owner,
            phase=self._phase(change, run, records), repos=[repo.repo for repo in repo_changes], behind=behind,
            run=run, pullRequests=self._pull_requests(change), cost=_cost(records),
            requestedBy=self._requested_by(change.id, records), createdAt=change.created_at,
            updatedAt=change.updated_at, closedAt=change.closed_at if closed else None)

    def _phase(self, change: LabChange, run: Optional[LabRun], records: list[LabRunRecord]) -> str:
        """Where the change stands: the first that holds of running, in review, stopped early, checks failing,
        built, planned, draft; ``merged`` or ``closed`` once closed."""
        if change.status == "closed":
            return "merged" if change.outcome == "merged" else "closed"
        if run is not None:
            return "running"
        if change.pull_requests:
            return "in_review"
        if records and records[-1].stopReason != "completed":
            return "stopped"
        last = self._read_model(change.id, _LAST_RUN_FILENAME, LabChangeView)
        if last is not None and last.step == "build":
            failing = any(not check.passed and not check.skipped for check in last.checks)
            return "checks_failing" if failing else "built"
        if change.status == "open" or os.path.isfile(self._path(change.id, _PLAN_FILENAME)):
            return "planned"
        return "draft"

    def _repo_changes(self, change: LabChange) -> list[LabRepoChange]:
        """What the change touched: its trees now when open, its last run's account otherwise."""
        if change.status == "open":
            return [repo_change_view(repo_change) for repo_change in self.workspace.repo_changes(change)]
        last = self._read_model(change.id, _LAST_RUN_FILENAME, LabChangeView)
        return list(last.repoChanges) if last is not None else []

    def _view(self, change: LabChange, records: list[LabRunRecord],
              repo_changes: list[LabRepoChange]) -> Optional[LabChangeView]:
        """The last run's view with the files, pull requests, cost and issues of now; None for a fresh draft."""
        now = {"repoChanges": repo_changes, "pullRequests": self._pull_requests(change), "cost": _cost(records),
               "issues": [_issue_view(issue) for issue in change.issues]}
        last = self._read_model(change.id, _LAST_RUN_FILENAME, LabChangeView)
        if last is not None:
            return last.model_copy(update=now)
        if change.status == "draft":
            return None
        return LabChangeView(step="build" if change.plan_approved else "plan", changeId=change.id, title=change.title,
                             kind=change.kind, checks=[], checklist=[], summary="", stopReason="completed", **now)

    def _pull_requests(self, change: LabChange) -> list[LabPullRequest]:
        views = []
        for repo in [key for key in REPO_ORDER if key in change.pull_requests]:
            owner, name, base = pull_requests.upstream_of(self.workspace, change, repo)
            number = change.pull_requests[repo]
            views.append(LabPullRequest(repo=repo, number=number,
                                        url=f"https://github.com/{owner}/{name}/pull/{number}",
                                        branch=change.branches.get(repo, ""), base=base))
        return views

    def _base_statuses(self, change: LabChange) -> list[LabRepoBaseStatus]:
        behind = self.workspace.base_status(change)
        statuses = []
        for repo in [key for key in REPO_ORDER if key in change.bases]:
            base = change.bases[repo]
            # Where LabWorkspace.refresh_bases keeps the upstream base, which base_status counts against.
            latest = self.workspace.run_git(repo, "rev-parse", f"refs/remotes/upstream/{base.branch}").strip()
            statuses.append(LabRepoBaseStatus(repo=repo, upstream=base.upstream, branch=base.branch, sha=base.sha,
                                              latest=latest, behind=behind.get(repo, 0)))
        return statuses

    def _bases(self) -> list[LabBase]:
        bases = []
        for repo in [key for key in REPO_ORDER if key in self.workspace.base_repos]:
            spec = self.workspace.base_repos[repo]
            bases.append(LabBase(repo=repo, upstream=f"{spec.github_owner}/{spec.github_name}",
                                 branch=spec.base_branch, sha=self.workspace.run_git(repo, "rev-parse", "HEAD").strip(),
                                 fetchedAt=self._fetched_at, fork=spec.github_owner.lower() != _PEARL_OWNER))
        return bases

    def _preview_slots(self) -> int:
        """The preview slots that can be used (a slot on the lab's own origin is off)."""
        if self.preview is None:
            return 0
        return sum(slot.enabled for slot in self.preview.view().slots)

    def _memory_info(self) -> LabMemoryInfo:
        try:
            count = len(self.memory.notes())
        except LabMemoryUnavailable as exc:
            return LabMemoryInfo(enabled=False, reason=str(exc), count=0, max=MAX_NOTES)
        return LabMemoryInfo(enabled=True, reason="", count=count, max=MAX_NOTES)

    def _thread_items(self, change_id: str, records: list[LabRunRecord]) -> list[LabThreadItem]:
        """The thread, each run's proposed notes narrowed to those still pending (not approved or dismissed)."""
        items = self.thread(change_id).items({record.runId: record for record in records})
        pending = {proposal.id for proposal in self._read_proposals(change_id)}
        for item in items:
            if item.type == "run" and item.lab is not None and item.lab.noteProposals:
                item.lab.noteProposals = [p for p in item.lab.noteProposals if p.id in pending]
        return items

    def _requested_by(self, change_id: str, records: list[LabRunRecord]) -> list[str]:
        """Logins that sent the change's messages, first first (its runs' logins when the thread has none)."""
        logins: list[str] = []
        for item in self.thread(change_id).items({record.runId: record for record in records}):
            if item.type == "message" and item.login not in logins:
                logins.append(item.login)
        for record in records:
            if record.login not in logins:
                logins.append(record.login)
        return logins

    # -- Internals: files ---------------------------------------------------------------------------------------

    def _path(self, change_id: str, filename: str) -> str:
        return os.path.join(self.workspace.state_dir, "changes", change_id, filename)

    def _records(self, change_id: str) -> list[LabRunRecord]:
        path = self._path(change_id, _RUNS_FILENAME)
        records = []
        for line in read_json_lines(path):
            try:
                records.append(LabRunRecord.model_validate(line))
            except ValidationError as exc:
                logger.warning("BESSER Lab: skipped a run record of %s: %d field errors", path, exc.error_count())
        return records

    def _append_record(self, change_id: str, record: LabRunRecord) -> None:
        with self._files_lock:
            append_json_line(self._path(change_id, _RUNS_FILENAME), record.model_dump(mode="json"))

    def _remember(self, change_id: str, done: LabDone) -> None:
        """Keep the run's view (``last_run.json``), its plan (``plan.json``) and its proposed notes."""
        view = LabChangeView.model_validate(done.model_dump(include=set(LabChangeView.model_fields)))
        _write_model(self._path(change_id, _LAST_RUN_FILENAME), view)
        if done.plan is not None:
            _write_model(self._path(change_id, _PLAN_FILENAME), done.plan)
        if done.noteProposals:
            with self._files_lock:
                pending = self._read_proposals(change_id)
                known = {proposal.id for proposal in pending}
                pending += [proposal for proposal in done.noteProposals if proposal.id not in known]
                self._write_proposals(change_id, pending)

    def _read_model(self, change_id: str, filename: str, model: type[BaseModel]) -> Any:
        path = self._path(change_id, filename)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return model.model_validate_json(handle.read())
        except (OSError, ValidationError) as exc:
            logger.warning("BESSER Lab: %s cannot be read and is left out: %s", path, exc)
            return None

    def _read_proposals(self, change_id: str) -> list[LabNoteProposal]:
        path = self._path(change_id, _PROPOSALS_FILENAME)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as handle:
                return [LabNoteProposal.model_validate(item) for item in json.load(handle)]
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("BESSER Lab: the proposed notes in %s cannot be read: %s", path, exc)
            return []

    def _write_proposals(self, change_id: str, proposals: list[LabNoteProposal]) -> None:
        path = self._path(change_id, _PROPOSALS_FILENAME)
        _write_text(path, json.dumps([proposal.model_dump(mode="json") for proposal in proposals], indent=2))

    def _proposal(self, ref: LabProposalRef) -> LabNoteProposal:
        found = next((p for p in self._read_proposals(ref.changeId) if p.id == ref.id), None)
        if found is None:
            raise LabRefusal("not_found", f"Change {ref.changeId} has no pending note proposal {ref.id}.")
        return found

    def _remove_proposal(self, ref: LabProposalRef) -> None:
        with self._files_lock:
            pending = self._read_proposals(ref.changeId)
            self._write_proposals(ref.changeId, [proposal for proposal in pending if proposal.id != ref.id])

    # -- Internals: audit, preview, logging ----------------------------------------------------------------------

    def _audit(self, login: str, action: str, *, change_id: Optional[str] = None, run_id: Optional[str] = None,
               detail: str = "") -> None:
        self._logged("an audit line", lambda: self.audit.append(AuditEntry(
            at=time.time(), login=login, action=action, change_id=change_id, run_id=run_id, detail=detail)))

    def _audit_if_opened(self, login: str, change_id: str) -> None:
        """A run of a draft whose plan gave it its branches: audit ``change_opened``."""
        try:
            change = self.workspace.get_change(change_id)
        except LabWorkspaceError:
            return  # dropped meanwhile
        if change.status == "open":
            self._audit(login, "change_opened", change_id=change_id, detail=change.title)

    @staticmethod
    def _logged(what: str, write) -> None:
        try:
            write()
        except OSError as exc:
            logger.warning("BESSER Lab: %s was not written: %s", what, exc)

    def _pool(self) -> PreviewPool:
        if self.preview is None:
            raise LabRefusal("state", "The preview is off on this lab (BESSER_LAB_PREVIEW_ENABLED is not set).")
        return self.preview

    @contextlib.contextmanager
    def _unknown_slot(self, slot: int, service: Optional[str] = None) -> Iterator[None]:
        try:
            yield
        except KeyError:
            what = f"slot {slot + 1}" if service is None else f"service {service!r} of slot {slot + 1}"
            raise LabRefusal("not_found", f"The preview has no {what}.") from None


# -- Module helpers ------------------------------------------------------------------------------------------------

def _run_view(ticket: LabRunTicket) -> LabRun:
    return LabRun(runId=ticket.run_id, changeId=ticket.change_id, login=ticket.login, step=ticket.step,
                  startedAt=ticket.started_at)


def _cost(records: list[LabRunRecord]) -> LabChangeCost:
    return LabChangeCost(usd=round(sum(record.usd for record in records), 6),
                         turns=sum(record.turns for record in records),
                         elapsedSeconds=round(sum(record.elapsedSeconds for record in records), 3), runs=len(records))


def _issue_view(issue: dict[str, Any]) -> LabIssue:
    return LabIssue(repo=str(issue.get("repo", "")), number=int(issue.get("number", 0)),
                    title=str(issue.get("title", "")), closes=bool(issue.get("closes", False)))


def _draft(note: LabNoteInput) -> NoteDraft:
    return NoteDraft(kind=note.kind, title=note.title, body=note.body, repos=tuple(note.repos),
                     change_kinds=tuple(kind.value for kind in note.changeKinds), docs_target=note.docsTarget)


def _note_view(note: MemoryNote) -> LabNote:
    return LabNote(id=note.id, kind=note.kind, title=note.title, body=note.body, repos=list(note.repos),
                   changeKinds=list(note.change_kinds), docsTarget=note.docs_target, sourceChange=note.source_change,
                   sourceUrl=note.source_url, approvedBy=note.approved_by, createdAt=note.created_at,
                   updatedAt=note.updated_at, lastReadAt=note.last_read_at, reads=note.reads, stale=note.is_stale())


@contextlib.contextmanager
def _memory_refusals() -> Iterator[None]:
    """The memory's refusals as lab refusals; an invalid note (``LabMemoryError``) goes on as it is."""
    try:
        yield
    except (LabMemoryUnavailable, LabMemoryFull) as exc:
        raise LabRefusal("state", str(exc)) from None
    except LabNoteNotFound as exc:
        raise LabRefusal("not_found", str(exc)) from None


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(path + ".tmp", path)


def _write_model(path: str, model: BaseModel) -> None:
    _write_text(path, model.model_dump_json(indent=2))


def _snapshot_node(environments: Environments, workspace: LabWorkspace) -> None:
    """Snapshot the base editor's node_modules (a background thread at start); without one, changes get npm ci."""
    try:
        environments.snapshot_node(workspace.base_repos[REPO_FRONTEND])
    except (LabEnvironmentError, OSError) as exc:
        logger.warning("BESSER Lab: no node_modules snapshot of the base editor (%s); each change's editor "
                       "environment is installed with npm ci instead", exc)


def create_lab_service(settings: BackendSettings) -> Optional[LabService]:
    """The lab service of a lab instance: the workspace, environments, sandbox policy, preview, ledgers, memory.

    Returns None when the lab is off. Blocking (git, a probe): the lifespan calls it in a thread.

    Raises:
        LabUnavailableError: The lab is on but cannot start (no workspace root, base clones not ready, folders
            that overlap the sandbox policy); its message is the routes' 503 detail and is logged at error.
    """
    # Here, not at the top: budget.py imports LabRefusal from this module, whichever of the two is imported first.
    from besser.utilities.web_modeling_editor.backend.services.lab.budget import LabBudget

    if not settings.lab_enabled:
        return None
    try:
        if not settings.lab_workspace_root:
            raise LabUnavailableError("BESSER_LAB_WORKSPACE_ROOT is not set.")
        upstreams = {REPO_BESSER: settings.lab_repo_besser, REPO_FRONTEND: settings.lab_repo_frontend,
                     REPO_MODELING_AGENT: settings.lab_repo_modeling_agent}
        workspace = LabWorkspace(settings.lab_workspace_root,
                                 repo_specs_from_settings(settings.lab_workspace_root, settings.lab_git_root,
                                                          upstreams),
                                 state_dir=settings.lab_state_dir, git_root=settings.lab_git_root)
        try:
            workspace.ensure_ready()
        except LabWorkspaceError as exc:
            raise LabUnavailableError(str(exc)) from None
        policy = functools.partial(lab_sandbox_mounts, workspace_root=workspace.root, git_root=workspace.git_root,
                                   hidden=tuple(lab_hidden_paths(settings, workspace)))
        try:  # an overlap refuses to start the lab rather than surface at the first run
            policy(workspace.base_repos, writable=True)
            policy(workspace.base_repos, writable=False)
        except LabSandboxPolicyError as exc:
            raise LabUnavailableError(f"The lab's folders overlap: {exc}") from None
    except LabUnavailableError as exc:
        logger.error("BESSER Lab is enabled but cannot start; its routes answer 503: %s", exc)
        raise

    cache = os.path.join(workspace.root, LAB_CACHE_DIRNAME)
    environments = Environments(cache, mode=probe_link_mode(cache))
    threading.Thread(target=_snapshot_node, args=(environments, workspace), name="lab-node-snapshot",
                     daemon=True).start()
    audit = AuditLog(workspace.state_dir)
    budget = LabBudget(audit, per_run_usd=settings.lab_max_cost_usd,
                       user_monthly_usd=settings.lab_user_monthly_cap_usd,
                       instance_monthly_usd=settings.lab_monthly_cap_usd)
    preview = None
    if settings.lab_preview_enabled:
        def sandbox_for(change: LabChange) -> PreviewSandbox:
            return PreviewSandbox(workspace.change_root(change), policy(workspace.change_repos(change)),
                                  bind=settings.lab_preview_bind)

        preview = PreviewPool(settings.lab_preview_urls, workspace, environments, sandbox_for=sandbox_for,
                              log_dir=os.path.join(workspace.state_dir, "preview-logs"),
                              bind=settings.lab_preview_bind,
                              proxy_port=(settings.lab_preview_llm_proxy_port
                                          if settings.lab_preview_openai_api_key.get_secret_value() else None),
                              deployment_url=settings.deployment_url)
    service = LabService(workspace, settings=settings, environments=environments, sandbox_policy=policy,
                         preview=preview, audit=audit, budget=budget,
                         memory=LabMemory(workspace.state_dir, workspace.root))
    service.admins()  # an admin not on the allowed list is warned about now, not at the first admin check
    logger.info("BESSER Lab started: %d run slots, preview %s, link mode %s", settings.lab_max_concurrent_runs,
                "on" if preview is not None else "off", environments.mode)
    return service
