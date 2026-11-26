"""BESSER Lab routes: changes, their runs and pull requests, the review loop, the preview, settings, team memory.

Every route needs the ``X-GitHub-Session`` of a login on ``BESSER_LAB_ALLOWED_LOGINS`` (``LabUserDep``: 404
with the lab off, 401 without a valid session, 403 for another login). Every allowed login may see and continue
any change; dropping one is for its owner or an admin. A refusal answers ``{"detail": {"code", "message"}}``
(``LabRefusalResponse``) with the status of its code.

The handlers parse the request and call ``LabService``; what runs git or reads files runs off the event loop.
"""

# No ``from __future__ import annotations``: FastAPI resolves the endpoints' annotations at import.
import asyncio
import functools
import contextlib
import uuid
from typing import Annotated, Any, Iterator, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from besser.spec_driven_agent.lab.github_tools import GitHubReadError
from besser.spec_driven_agent.lab.memory import LabMemoryError
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from besser.utilities.web_modeling_editor.backend.dependencies import (
    LabServiceDep,
    SettingsDep,
    SpecDrivenRuntimeDep,
)
from besser.utilities.web_modeling_editor.backend.models.lab import (
    CHANGE_ID_PATTERN,
    LabAuditResponse,
    LabChangeDetail,
    LabChangeRequest,
    LabChangesResponse,
    LabDiffResponse,
    LabHomeResponse,
    LabMemoryResponse,
    LabMessageRequest,
    LabNote,
    LabNoteInput,
    LabPreviewLog,
    LabPreviewRestartRequest,
    LabPreviewServiceName,
    LabPreviewSlot,
    LabPreviewView,
    LabProposalRef,
    LabPullRequestRequest,
    LabPullRequestsResponse,
    LabRefusalResponse,
    LabIssueOption,
    LabIssuesResponse,
    LabRepoKey,
    LabReviewResponse,
    LabSettingsResponse,
    LabTakeSlotRequest,
    LabTouchSlotRequest,
    LabUpdateBranchRequest,
    LabUpdateBranchResponse,
)
from besser.utilities.web_modeling_editor.backend.routers.auth import LabUserDep
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService
from besser.utilities.web_modeling_editor.backend.services.lab.github_reader import LabGitHubReader
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    PullRequestError,
    PullRequestInputError,
    PullRequestSignInError,
)
from besser.utilities.web_modeling_editor.backend.services.lab.runner import LabRunner
from besser.utilities.web_modeling_editor.backend.services.lab.service import (
    REFUSAL_STATUS,
    LabRefusal,
)

#: How many issues the home's picker lists at once.
ISSUE_PICKER_LIMIT = 30

router = APIRouter(prefix="/besser_api", tags=["lab"])

# Preview log lines one request may read: a screen of context; the log file itself is not capped.
DEFAULT_PREVIEW_LOG_LINES = 200
MAX_PREVIEW_LOG_LINES = 2000
# A repository-relative path in a diff request; far beyond any path in the repositories.
MAX_DIFF_PATH_CHARS = 1024

ChangeIdQuery = Annotated[str, Query(pattern=CHANGE_ID_PATTERN, description="The change's id.")]
NoteIdPath = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="The note's id.")]

_BAD_ISSUES = {400: {"description": "A ``closesIssues`` entry is not one of the change's issues: "
                                      "``{\"detail\": {\"code\": \"bad_request\", \"message\"}}``."}}


def _refusals(*codes: str, extra: Optional[dict[int, dict[str, Any]]] = None) -> dict[int | str, dict[str, Any]]:
    """The OpenAPI ``responses`` of a route that refuses with these codes (``extra``: other answers)."""
    by_status: dict[int, list[str]] = {}
    for code in codes:
        by_status.setdefault(REFUSAL_STATUS[code], []).append(code)
    responses: dict[int | str, dict[str, Any]] = {
        status: {"model": LabRefusalResponse, "description": "Refused: " + ", ".join(names)}
        for status, names in by_status.items()}
    responses.update(extra or {})
    return responses


def _refused(refusal: LabRefusal) -> HTTPException:
    return HTTPException(status_code=refusal.status, detail={"code": refusal.code, "message": refusal.message})


@contextlib.contextmanager
def _answers() -> Iterator[None]:
    """The service's refusals and errors as HTTP answers (R13); every message is written for the user."""
    try:
        yield
    except LabRefusal as refusal:
        raise _refused(refusal) from None
    except PullRequestSignInError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None
    except PullRequestInputError as exc:
        raise HTTPException(status_code=400, detail={"code": "bad_request", "message": str(exc)}) from None
    except (LabWorkspaceError, PullRequestError) as exc:
        raise HTTPException(status_code=409, detail={"code": "state", "message": str(exc)}) from None
    except ValidationError:
        raise  # a model the service built is a defect, not the user's request
    except (ValueError, LabMemoryError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


async def _call(function, *args: Any) -> Any:
    """A blocking service call (git, files) off the event loop, its refusals answered."""
    with _answers():
        return await asyncio.to_thread(function, *args)


# -- Home and changes ---------------------------------------------------------------------------------------

@router.get("/lab/home", response_model=LabHomeResponse)
async def lab_home(user: LabUserDep, lab: LabServiceDep) -> LabHomeResponse:
    """The signed-in user, the base of each repository, the runs in progress, this month's spending, the
    preview slots and the team memory's state."""
    return await _call(lab.home, user)


@router.get("/lab/changes", response_model=LabChangesResponse)
async def lab_changes(user: LabUserDep, lab: LabServiceDep) -> LabChangesResponse:
    """Every open change and draft, most recently updated first, then the 10 most recently closed changes."""
    return await _call(lab.changes)


@router.post("/lab/change", response_model=LabChangeDetail, responses=_refusals("user_cap", "instance_cap"))
async def lab_create_change(user: LabUserDep, lab: LabServiceDep) -> LabChangeDetail:
    """Start a change as a draft owned by the user: a thread and no branches until its first plan is submitted.

    Refused when the user's or the lab's monthly budget is spent.
    """
    return await _call(lab.create_change, user)


@router.get("/lab/change", response_model=LabChangeDetail, responses=_refusals("not_found"))
async def lab_get_change(user: LabUserDep, lab: LabServiceDep, changeId: ChangeIdQuery) -> LabChangeDetail:
    """One change: its files, last run, branches, bases, thread and run records."""
    return await _call(lab.change_detail, changeId)


@router.post("/lab/message", response_class=StreamingResponse,
             responses=_refusals("change_busy", "runs_full", "user_cap", "instance_cap", "not_found", "state"))
async def lab_message(body: LabMessageRequest, user: LabUserDep, lab: LabServiceDep, runtime: SpecDrivenRuntimeDep,
                      settings: SettingsDep) -> StreamingResponse:
    """Run the change's next step (plan or build) for the user's message, as Spec-Driven SSE events.

    The ``start`` event carries the run id that ``POST /spec-driven/cancel/{run_id}`` stops; the ``done``
    event carries the step's result in ``lab`` (``LabDone``). The run goes on when the stream drops, until no
    one has followed it for ``BESSER_LLM_DISCONNECTED_GRACE_SECONDS``; ``/spec-driven/runs/{run_id}/events``
    replays it. One run per change at a time, at most ``BESSER_LAB_MAX_CONCURRENT_RUNS`` across the lab; the
    run's cost cap is what is left of the per-run, the user's and the lab's monthly budgets.
    """
    run_id = uuid.uuid4().hex
    cancel_event = await runtime.active_runs.reserve(run_id)
    if cancel_event is None:  # a fresh UUID is never taken
        raise HTTPException(status_code=409, detail="Generated run ID is already active")
    try:
        ticket = await _call(functools.partial(lab.begin_run, body.changeId, run_id, user.login,
                                              own_key=body.llm is not None))
    except HTTPException:
        await runtime.active_runs.release(run_id, cancel_event)
        raise
    runner = LabRunner(body, user=user, lab=lab, runtime=runtime, settings=settings, ticket=ticket,
                       cancel_event=cancel_event)
    try:
        await runtime.durable_runs.start(
            run_id, runner.stream(), cancel=cancel_event.set,
            on_abandoned=cancel_event.set if settings.llm_cancel_abandoned_runs else None,
            disconnect_grace_seconds=settings.llm_disconnected_grace_seconds,
        )
    except Exception:
        # The stream never started, so it gives nothing back: the run is given back here, recorded as failed.
        await runner.abandon("The run could not be started.")
        raise
    return StreamingResponse(
        runtime.durable_runs.subscribe(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "X-BESSER-Run-Id": run_id},
    )


@router.post("/lab/pull-request", response_model=LabPullRequestsResponse,
             responses=_refusals("change_busy", "not_found", "state", extra=_BAD_ISSUES))
async def lab_pull_request(body: LabPullRequestRequest, user: LabUserDep, lab: LabServiceDep) -> LabPullRequestsResponse:
    """Push the change and open a draft pull request in each repository it touched, editor first, with the
    user's GitHub token; push to the pull requests already open.

    401 when GitHub refuses the user's sign-in, so the editor asks them to sign in again. The full test suites
    are not run first; ``checksNote`` tells the user which checks the change passed.
    """
    with _answers():
        opened, checks_note = await lab.open_pull_requests(user, body.changeId, title=body.title, body=body.body,
                                                           closes_issues=body.closesIssues)
    return LabPullRequestsResponse(pullRequests=opened, checksNote=checks_note)


@router.post("/lab/change/drop", response_model=LabChangesResponse,
             responses=_refusals("not_owner", "change_busy", "not_found"))
async def lab_drop_change(body: LabChangeRequest, user: LabUserDep, lab: LabServiceDep) -> LabChangesResponse:
    """Delete the change: its working trees, branches, thread and notes. Pull requests already opened stay on
    GitHub. Only its owner or an admin may drop it."""
    return await _call(lab.drop, user, body.changeId)


@router.post("/lab/change/close", response_model=LabChangeDetail,
             responses=_refusals("change_busy", "not_found", "state"))
async def lab_close_change(body: LabChangeRequest, user: LabUserDep, lab: LabServiceDep) -> LabChangeDetail:
    """Close the change, merged or not: its working trees and branches are removed, its state and thread kept.

    Refused while work is not pushed to its pull requests.
    """
    with _answers():
        return await lab.close(user, body.changeId)


@router.post("/lab/change/update-base", response_model=LabUpdateBranchResponse,
             responses=_refusals("change_busy", "not_found", "state"))
async def lab_update_base(body: LabUpdateBranchRequest, user: LabUserDep, lab: LabServiceDep) -> LabUpdateBranchResponse:
    """Bring one repository of the change up to its base branch.

    With an open pull request, GitHub updates the pull request's branch and the lab pulls it back; otherwise
    the lab merges the freshly fetched base into the change. Conflicts are left for a "Resolve conflicts"
    build step.
    """
    with _answers():
        return await lab.update_from_base(user, body.changeId, body.repo)


# -- Review -------------------------------------------------------------------------------------------------

@router.get("/lab/diff", response_model=LabDiffResponse, responses=_refusals("not_found", "state"))
async def lab_diff(
    user: LabUserDep,
    lab: LabServiceDep,
    changeId: ChangeIdQuery,
    repo: Annotated[Optional[LabRepoKey], Query(description="Only this repository.")] = None,
    path: Annotated[Optional[str], Query(max_length=MAX_DIFF_PATH_CHARS,
                                         description="Only this file of ``repo``, with a larger size cap.")] = None,
) -> LabDiffResponse:
    """The change's working trees against their bases, file by file, secrets redacted.

    Refused for a draft (no trees yet) and for a closed change (its branches are gone; the diff is on the pull
    request).
    """
    return await _call(lab.diff, changeId, repo, path)


@router.get("/lab/issues", response_model=LabIssuesResponse)
async def lab_issues(
    user: LabUserDep,
    lab: LabServiceDep,
    repo: Annotated[LabRepoKey, Query(description="The repository whose issues to list.")],
    q: Annotated[str, Query(max_length=200, description="Words to search the issues for; empty lists the "
                                                           "most recently updated.")] = "",
) -> LabIssuesResponse:
    """Open issues of one of the lab's repositories on GitHub, for the home's issue picker: those matching ``q``,
    or the most recently updated ones. Read with the signed-in user's token; 401 when GitHub no longer accepts
    it, 502 when GitHub cannot be read."""
    reader = LabGitHubReader(GitHubService(user.access_token), lab.workspace.base_repos,
                             asyncio.get_running_loop(), login=user.login)
    query = q.strip() or "sort:updated-desc"
    try:
        found = await asyncio.to_thread(reader.search, query, repo, "issue", "open", ISSUE_PICKER_LIMIT)
    except GitHubReadError as exc:
        raise HTTPException(status_code=401 if exc.reason == "signed_out" else 502, detail=str(exc)) from None
    return LabIssuesResponse(issues=[
        LabIssueOption(repo=item["repo"], number=item["number"], title=item["title"], author=item["author"],
                       updatedAt=item["updated_at"], url=item["url"])
        for item in found])


@router.get("/lab/review", response_model=LabReviewResponse,
            responses=_refusals("not_found"))
async def lab_review(user: LabUserDep, lab: LabServiceDep, changeId: ChangeIdQuery) -> LabReviewResponse:
    """The change's pull requests as GitHub reports them: state, CI checks with failing log tails, reviews and
    comments, and when to ask again."""
    with _answers():
        return await lab.review(user, changeId)


# -- Preview ------------------------------------------------------------------------------------------------

@router.get("/lab/preview", response_model=LabPreviewView)
async def lab_preview(user: LabUserDep, lab: LabServiceDep) -> LabPreviewView:
    """Every preview slot: the change it shows, who used it last, and the state of its three services."""
    return await _call(lab.preview_view)


@router.post("/lab/preview/slot", response_model=LabPreviewSlot,
             responses=_refusals("preview_full", "not_found", "state"))
async def lab_take_preview_slot(body: LabTakeSlotRequest, user: LabUserDep, lab: LabServiceDep) -> LabPreviewSlot:
    """Show the change in a preview slot: the slot already showing it, else a free one, else one nobody has
    used for 10 minutes. Answers at once; the slot's ``detail`` names the preparation step meanwhile."""
    return await _call(lab.take_preview_slot, user, body.changeId)


@router.post("/lab/preview/slot/touch", status_code=204, responses=_refusals("state"))
async def lab_touch_preview_slot(body: LabTouchSlotRequest, user: LabUserDep, lab: LabServiceDep) -> None:
    """Record that the user still uses the slot, so it is not taken over as idle."""
    await _call(lab.touch_preview_slot, user, body.slot)


@router.post("/lab/preview/restart", response_model=LabPreviewView, responses=_refusals("state"))
async def lab_restart_preview(body: LabPreviewRestartRequest, user: LabUserDep, lab: LabServiceDep) -> LabPreviewView:
    """Restart one service of a slot, or all three."""
    return await _call(lab.restart_preview, user, body.slot, body.service)


@router.get("/lab/preview/logs/{service}", response_model=LabPreviewLog, responses=_refusals("state"))
async def lab_preview_log(
    service: LabPreviewServiceName,
    user: LabUserDep,
    lab: LabServiceDep,
    slot: Annotated[int, Query(ge=0, description="The preview slot.")],
    lines: Annotated[int, Query(ge=1, le=MAX_PREVIEW_LOG_LINES,
                                description="How many of the last lines to read.")] = DEFAULT_PREVIEW_LOG_LINES,
) -> LabPreviewLog:
    """The end of one preview service's log, secrets redacted."""
    return await _call(lab.preview_log, slot, service, lines)


# -- Settings and usage -------------------------------------------------------------------------------------

@router.get("/lab/settings", response_model=LabSettingsResponse)
async def lab_settings(user: LabUserDep, lab: LabServiceDep) -> LabSettingsResponse:
    """The repositories and their bases, the lab's limits, its admins and this month's spending."""
    return await _call(lab.settings_view, user)


@router.post("/lab/settings/refresh-bases", response_model=LabSettingsResponse, responses=_refusals("state"))
async def lab_refresh_bases(user: LabUserDep, lab: LabServiceDep) -> LabSettingsResponse:
    """Fetch every repository's base branch now. Open changes keep the base they started from.

    Refused when a fetch fails; the message names the repository.
    """
    return await _call(lab.refresh_bases, user)


@router.get("/lab/audit", response_model=LabAuditResponse)
async def lab_audit(
    user: LabUserDep,
    lab: LabServiceDep,
    month: Annotated[Optional[str], Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
                                          description="``YYYY-MM``; the current UTC month when absent.")] = None,
) -> LabAuditResponse:
    """One UTC month of the lab's audit log: who did what to which change, and what it cost."""
    return await _call(lab.audit_view, month)


# -- Team memory --------------------------------------------------------------------------------------------

@router.get("/lab/memory", response_model=LabMemoryResponse)
async def lab_memory(user: LabUserDep, lab: LabServiceDep) -> LabMemoryResponse:
    """The team notes the lab agent reads, and whether the memory is on."""
    return await _call(lab.memory_view)


@router.post("/lab/memory", response_model=LabNote, responses=_refusals("not_found", "state"))
async def lab_add_note(body: LabNoteInput, user: LabUserDep, lab: LabServiceDep) -> LabNote:
    """Add a team note, approved by the user; with ``fromProposal`` it approves that proposal.

    Refused while the memory is off or full.
    """
    return await _call(lab.add_note, user, body)


@router.put("/lab/memory/{note_id}", response_model=LabNote, responses=_refusals("not_found", "state"))
async def lab_update_note(note_id: NoteIdPath, body: LabNoteInput, user: LabUserDep, lab: LabServiceDep) -> LabNote:
    """Replace a team note's text and scope."""
    return await _call(lab.update_note, user, note_id, body)


@router.delete("/lab/memory/{note_id}", status_code=204, responses=_refusals("not_found", "state"))
async def lab_delete_note(note_id: NoteIdPath, user: LabUserDep, lab: LabServiceDep) -> None:
    """Delete a team note."""
    await _call(lab.delete_note, user, note_id)


@router.post("/lab/memory/proposals/dismiss", status_code=204, responses=_refusals("not_found"))
async def lab_dismiss_note_proposal(body: LabProposalRef, user: LabUserDep, lab: LabServiceDep) -> None:
    """Dismiss a note the lab agent proposed: it is not kept."""
    await _call(lab.dismiss_proposal, user, body)
