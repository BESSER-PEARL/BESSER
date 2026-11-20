"""BESSER Lab on one backend instance: its working copy, its preview, and one run or change action at a time.

The lifespan builds the service (``create_lab_service``) and stores it on ``app.state.lab``; the lab routes
receive it through ``dependencies.get_lab_service``. Every action that moves branches, and every run, holds
the service's lock, so a park cannot switch branches under a build that is writing files.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from typing import Iterator, Optional

from pydantic import ValidationError

from besser.spec_driven_agent.lab.contracts import LabChange, RepoChange
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabChangeView,
    LabDone,
    LabFileChange,
    LabParkedChange,
    LabPreview,
    LabPreviewService,
    LabRepoChange,
    LabRun,
    LabStateResponse,
)
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.preview import PreviewSupervisor, default_services
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    PullRequestInfo,
    open_pull_requests,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

logger = logging.getLogger(__name__)


class LabBusyError(Exception):
    """A lab run or another change action is in progress."""


class LabStateError(Exception):
    """The working copy's state does not allow the action (for example, no active change)."""


def repo_change_view(change: RepoChange) -> LabRepoChange:
    """A repository's changed files as the lab routes send them."""
    return LabRepoChange(repo=change.repo, branch=change.branch, base=change.base,
                         files=[LabFileChange(path=f.path, status=f.status) for f in change.files])


def _checks_note(last: Optional[LabChangeView]) -> str:
    """What the user is told about the checks of a change opened as pull requests."""
    full = "The repositories' full test suites did not run in the lab."
    if last is None or not last.checks:
        return f"The lab ran no checks on this change. {full}"
    passed = sum(c.passed for c in last.checks)
    skipped = sum(c.skipped and not c.passed for c in last.checks)
    failed = len(last.checks) - passed - skipped
    return (f"The lab's last build ran only the checks of the touched areas: {passed} passed, {failed} failed, "
            f"{skipped} skipped. {full}")


class LabService:
    """The lab's working copy, its preview, and what the last run of each change reported."""

    def __init__(self, workspace: LabWorkspace, *, preview: Optional[PreviewSupervisor] = None):
        self.workspace = workspace
        self.preview = preview
        self._lock = threading.Lock()
        #: the run holding the lab, for a client that lost its stream
        self._run: Optional[LabRun] = None

    # -- the one run or action at a time --------------------------------------------------------

    def try_begin_run(self, run_id: str) -> bool:
        """Take the lab for run ``run_id`` without waiting; False when a run or an action holds it."""
        if not self._lock.acquire(blocking=False):
            return False
        self._run = LabRun(runId=run_id, startedAt=time.time())
        return True

    def end_run(self) -> None:
        """Give the lab back after a run."""
        self._run = None
        self._lock.release()

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise LabBusyError("A lab run or another change action is in progress; try again when it ends.")
        try:
            yield
        finally:
            self._lock.release()

    # -- state ----------------------------------------------------------------------------------

    def _last_run_path(self, change_id: str) -> str:
        # Beside the change's state.json, so parking keeps it and dropping the change removes it.
        return os.path.join(self.workspace.state_dir, "changes", change_id, "last_run.json")

    def remember_run(self, done: LabDone) -> None:
        """Keep the last run's checks and checklist with its change, for ``state``."""
        if not done.changeId:
            return
        view = LabChangeView(**done.model_dump(exclude={"plan", "modelProposal", "finalText"}))
        path = self._last_run_path(done.changeId)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as handle:
                handle.write(view.model_dump_json(indent=2))
            os.replace(path + ".tmp", path)
        except OSError as exc:
            logger.warning("BESSER Lab: the last run of change %s was not kept, the state will not show its "
                           "checks: %s", done.changeId, exc)

    def _last_run(self, change_id: str) -> Optional[LabChangeView]:
        path = self._last_run_path(change_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return LabChangeView.model_validate_json(handle.read())
        except (OSError, ValidationError) as exc:
            logger.warning("BESSER Lab: the last run of change %s cannot be read, its checks are not shown: %s",
                           change_id, exc)
            return None

    def state(self, login: str) -> LabStateResponse:
        """The signed-in user, the active change with its files now, the parked changes and the preview.

        Raises:
            LabWorkspaceError: git could not read the working copy.
        """
        active = self.workspace.active_change()
        return LabStateResponse(
            enabled=True,
            user=login,
            change=self._change_view(active) if active is not None else None,
            parked=[LabParkedChange(changeId=c.id, title=c.title, kind=c.kind, createdAt=c.created_at)
                    for c in self.workspace.parked_changes()],
            preview=LabPreview(
                enabled=self.preview is not None,
                services=[LabPreviewService(**s) for s in self.preview.status()] if self.preview else [],
            ),
            run=self._run,
        )

    def _change_view(self, change: LabChange) -> LabChangeView:
        repo_changes = [repo_change_view(c) for c in self.workspace.repo_changes(change)]
        last = self._last_run(change.id)
        if last is not None:
            return last.model_copy(update={"repoChanges": repo_changes})
        return LabChangeView(step="build" if change.plan_approved else "plan", changeId=change.id,
                             title=change.title, kind=change.kind, repoChanges=repo_changes, checks=[],
                             checklist=[], summary="", stopReason="completed")

    # -- change actions -------------------------------------------------------------------------

    def _active(self, action: str) -> LabChange:
        change = self.workspace.active_change()
        if change is None:
            raise LabStateError(f"There is no active change to {action}.")
        return change

    def park(self, login: str) -> LabStateResponse:
        """Commit the active change's work in progress and check out the base branches."""
        with self._exclusive():
            self.workspace.park(self._active("park"))
            self.refresh_preview()
            return self.state(login)

    def resume(self, login: str, change_id: str) -> LabStateResponse:
        """Check out a parked change again."""
        with self._exclusive():
            self.workspace.resume(change_id)
            self.refresh_preview()
            return self.state(login)

    def drop(self, login: str) -> LabStateResponse:
        """Discard the active change: its work, branches and notes."""
        with self._exclusive():
            change = self._active("drop")
            self.workspace.drop(change)
            self.refresh_preview()
            return self.state(login)

    def reset_preview(self, login: str) -> LabStateResponse:
        """Check out every repository's base branch; refused while a change is active."""
        with self._exclusive():
            active = self.workspace.active_change()
            if active is not None:
                raise LabStateError(f"Change '{active.title}' is active; park or drop it before resetting.")
            for key, spec in self.workspace.repos.items():
                self.workspace.run_git(key, "switch", "-q", spec.base_branch)
            self.refresh_preview()
            return self.state(login)

    async def open_pull_requests(self, user: LabUser, *, change_id: Optional[str] = None, title: Optional[str],
                                 body: Optional[str]) -> tuple[list[PullRequestInfo], str]:
        """Push the active change and open its draft pull requests with the user's GitHub token.

        ``change_id`` is the change the user asked for; another active change is refused before anything is
        pushed. Returns the pull requests and a note on the checks the change passed: the lab runs only the
        checks of the touched areas, never the repositories' full suites.
        """
        with self._exclusive():
            change = await asyncio.to_thread(self._active, "open a pull request for")
            if change_id is not None and change_id != change.id:
                raise LabStateError("This change is not the active one: resume it on the lab page first.")
            opened = await open_pull_requests(self.workspace, change, access_token=user.access_token,
                                              login=user.login, title=title, body=body)
            last = await asyncio.to_thread(self._last_run, change.id)
            return opened, _checks_note(last)

    # -- preview --------------------------------------------------------------------------------

    def refresh_preview(self) -> None:
        """Restart the preview services whose source changed; a failure is logged, not raised."""
        if self.preview is None:
            return
        try:
            restarted = self.preview.restart_changed()
        except OSError:
            logger.warning("BESSER Lab: the preview could not be restarted", exc_info=True)
            return
        if restarted:
            logger.info("BESSER Lab: restarted preview services %s", ", ".join(restarted))

    def close(self) -> None:
        """Stop the preview."""
        if self.preview is not None:
            self.preview.stop()


def create_lab_service(settings: BackendSettings) -> Optional[LabService]:
    """The lab service of a lab instance, with its preview started when it is enabled.

    Returns None when ``BESSER_LAB_WORKSPACE_ROOT`` is not set: every lab route then answers 503.
    """
    root = settings.lab_workspace_root
    if not root:
        logger.error("BESSER Lab is enabled but BESSER_LAB_WORKSPACE_ROOT is not set; the lab routes answer 503")
        return None
    workspace = LabWorkspace(root, state_dir=settings.lab_state_dir)
    try:
        workspace.ensure_ready()
    except LabWorkspaceError as exc:
        # Kept: the routes report the same error to the user until the working copy is fixed.
        logger.warning("BESSER Lab: the working copy at %s is not ready: %s", root, exc)
    preview = None
    if settings.lab_preview_enabled:
        # DEPLOYMENT_URL is the lab's public address, where nginx serves the preview.
        preview = PreviewSupervisor(default_services(workspace.repos, public_url=settings.deployment_url),
                                    log_dir=os.path.join(workspace.state_dir, "preview-logs"))
        preview.start()
    return LabService(workspace, preview=preview)
