"""BESSER Lab routes: changes, their runs and pull requests, the review loop, the preview, settings, team memory.

Every route needs the ``X-GitHub-Session`` of a login on ``BESSER_LAB_ALLOWED_LOGINS`` (``LabUserDep``: 404
with the lab off, 401 without a valid session, 403 for another login). Every allowed login may see and continue
any change; dropping one is for its owner or an admin. A refusal answers ``{"detail": {"code", "message"}}``
(``LabRefusalResponse``) with the status of its code.

The handlers answer 501 until the phase 2 lab service is built: the table, models and docstrings are final.
"""

# No ``from __future__ import annotations``: FastAPI resolves the endpoints' annotations at import.
from typing import Annotated, Any, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

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
    LabRepoKey,
    LabReviewResponse,
    LabSettingsResponse,
    LabTakeSlotRequest,
    LabTouchSlotRequest,
    LabUpdateBranchRequest,
    LabUpdateBranchResponse,
)
from besser.utilities.web_modeling_editor.backend.routers.auth import LabUserDep

router = APIRouter(prefix="/besser_api", tags=["lab"])

NOT_BUILT_DETAIL = "Not built yet: BESSER Lab phase 2 is being built."

# The HTTP status of each refusal code (R13).
REFUSAL_STATUS = {
    "change_busy": 409, "runs_full": 429, "user_cap": 409, "instance_cap": 409, "not_owner": 403,
    "preview_full": 409, "not_found": 404, "state": 409,
}

# Preview log lines one request may read: a screen of context; the log file itself is not capped.
DEFAULT_PREVIEW_LOG_LINES = 200
MAX_PREVIEW_LOG_LINES = 2000
# A repository-relative path in a diff request; far beyond any path in the repositories.
MAX_DIFF_PATH_CHARS = 1024

ChangeIdQuery = Annotated[str, Query(pattern=CHANGE_ID_PATTERN, description="The change's id.")]
NoteIdPath = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="The note's id.")]


def _refusals(*codes: str) -> dict[int | str, dict[str, Any]]:
    """The OpenAPI ``responses`` of a route that refuses with these codes."""
    by_status: dict[int, list[str]] = {}
    for code in codes:
        by_status.setdefault(REFUSAL_STATUS[code], []).append(code)
    return {status: {"model": LabRefusalResponse, "description": "Refused: " + ", ".join(names)}
            for status, names in by_status.items()}


def _not_built() -> HTTPException:
    return HTTPException(status_code=501, detail=NOT_BUILT_DETAIL)


# -- Home and changes ---------------------------------------------------------------------------------------

@router.get("/lab/home", response_model=LabHomeResponse)
async def lab_home(user: LabUserDep) -> LabHomeResponse:
    """The signed-in user, the base of each repository, the runs in progress, this month's spending, the
    preview slots and the team memory's state."""
    raise _not_built()


@router.get("/lab/changes", response_model=LabChangesResponse)
async def lab_changes(user: LabUserDep) -> LabChangesResponse:
    """Every open change and draft, most recently updated first, then the 10 most recently closed changes."""
    raise _not_built()


@router.post("/lab/change", response_model=LabChangeDetail, responses=_refusals("user_cap", "instance_cap"))
async def lab_create_change(user: LabUserDep) -> LabChangeDetail:
    """Start a change as a draft owned by the user: a thread and no branches until its first plan is submitted.

    Refused when the user's or the lab's monthly budget is spent.
    """
    raise _not_built()


@router.get("/lab/change", response_model=LabChangeDetail, responses=_refusals("not_found"))
async def lab_get_change(user: LabUserDep, changeId: ChangeIdQuery) -> LabChangeDetail:
    """One change: its files, last run, branches, bases, thread and run records."""
    raise _not_built()


@router.post("/lab/message", response_class=StreamingResponse,
             responses=_refusals("change_busy", "runs_full", "user_cap", "instance_cap", "not_found", "state"))
async def lab_message(body: LabMessageRequest, user: LabUserDep) -> StreamingResponse:
    """Run the change's next step (plan or build) for the user's message, as Spec-Driven SSE events.

    The ``start`` event carries the run id that ``POST /spec-driven/cancel/{run_id}`` stops; the ``done``
    event carries the step's result in ``lab`` (``LabDone``). The run goes on when the stream drops, until no
    one has followed it for ``BESSER_LLM_DISCONNECTED_GRACE_SECONDS``; ``/spec-driven/runs/{run_id}/events``
    replays it. One run per change at a time, at most ``BESSER_LAB_MAX_CONCURRENT_RUNS`` across the lab; the
    run's cost cap is what is left of the per-run, the user's and the lab's monthly budgets.
    """
    raise _not_built()


@router.post("/lab/pull-request", response_model=LabPullRequestsResponse,
             responses=_refusals("change_busy", "not_found", "state"))
async def lab_pull_request(body: LabPullRequestRequest, user: LabUserDep) -> LabPullRequestsResponse:
    """Push the change and open a draft pull request in each repository it touched, editor first, with the
    user's GitHub token; push to the pull requests already open.

    401 when GitHub refuses the user's sign-in, so the editor asks them to sign in again. The full test suites
    are not run first; ``checksNote`` tells the user which checks the change passed.
    """
    raise _not_built()


@router.post("/lab/change/drop", response_model=LabChangesResponse,
             responses=_refusals("not_owner", "change_busy", "not_found"))
async def lab_drop_change(body: LabChangeRequest, user: LabUserDep) -> LabChangesResponse:
    """Delete the change: its working trees, branches, thread and notes. Pull requests already opened stay on
    GitHub. Only its owner or an admin may drop it."""
    raise _not_built()


@router.post("/lab/change/close", response_model=LabChangeDetail,
             responses=_refusals("change_busy", "not_found", "state"))
async def lab_close_change(body: LabChangeRequest, user: LabUserDep) -> LabChangeDetail:
    """Close the change, merged or not: its working trees and branches are removed, its state and thread kept.

    Refused while work is not pushed to its pull requests.
    """
    raise _not_built()


@router.post("/lab/change/update-base", response_model=LabUpdateBranchResponse,
             responses=_refusals("change_busy", "not_found", "state"))
async def lab_update_base(body: LabUpdateBranchRequest, user: LabUserDep) -> LabUpdateBranchResponse:
    """Bring one repository of the change up to its base branch.

    With an open pull request, GitHub updates the pull request's branch and the lab pulls it back; otherwise
    the lab merges the freshly fetched base into the change. Conflicts are left for a "Resolve conflicts"
    build step.
    """
    raise _not_built()


# -- Review -------------------------------------------------------------------------------------------------

@router.get("/lab/diff", response_model=LabDiffResponse, responses=_refusals("not_found", "state"))
async def lab_diff(
    user: LabUserDep,
    changeId: ChangeIdQuery,
    repo: Annotated[Optional[LabRepoKey], Query(description="Only this repository.")] = None,
    path: Annotated[Optional[str], Query(max_length=MAX_DIFF_PATH_CHARS,
                                         description="Only this file of ``repo``, with a larger size cap.")] = None,
) -> LabDiffResponse:
    """The change's working trees against their bases, file by file, secrets redacted.

    Refused for a draft (no trees yet) and for a closed change (its branches are gone; the diff is on the pull
    request).
    """
    raise _not_built()


@router.get("/lab/review", response_model=LabReviewResponse, responses=_refusals("not_found"))
async def lab_review(user: LabUserDep, changeId: ChangeIdQuery) -> LabReviewResponse:
    """The change's pull requests as GitHub reports them: state, CI checks with failing log tails, reviews and
    comments, and when to ask again."""
    raise _not_built()


# -- Preview ------------------------------------------------------------------------------------------------

@router.get("/lab/preview", response_model=LabPreviewView)
async def lab_preview(user: LabUserDep) -> LabPreviewView:
    """Every preview slot: the change it shows, who used it last, and the state of its three services."""
    raise _not_built()


@router.post("/lab/preview/slot", response_model=LabPreviewSlot,
             responses=_refusals("preview_full", "not_found", "state"))
async def lab_take_preview_slot(body: LabTakeSlotRequest, user: LabUserDep) -> LabPreviewSlot:
    """Show the change in a preview slot: the slot already showing it, else a free one, else one nobody has
    used for 10 minutes. Answers at once; the slot's ``detail`` names the preparation step meanwhile."""
    raise _not_built()


@router.post("/lab/preview/slot/touch", status_code=204, responses=_refusals("state"))
async def lab_touch_preview_slot(body: LabTouchSlotRequest, user: LabUserDep) -> None:
    """Record that the user still uses the slot, so it is not taken over as idle."""
    raise _not_built()


@router.post("/lab/preview/restart", response_model=LabPreviewView, responses=_refusals("state"))
async def lab_restart_preview(body: LabPreviewRestartRequest, user: LabUserDep) -> LabPreviewView:
    """Restart one service of a slot, or all three."""
    raise _not_built()


@router.get("/lab/preview/logs/{service}", response_model=LabPreviewLog, responses=_refusals("state"))
async def lab_preview_log(
    service: LabPreviewServiceName,
    user: LabUserDep,
    slot: Annotated[int, Query(ge=0, description="The preview slot.")],
    lines: Annotated[int, Query(ge=1, le=MAX_PREVIEW_LOG_LINES,
                                description="How many of the last lines to read.")] = DEFAULT_PREVIEW_LOG_LINES,
) -> LabPreviewLog:
    """The end of one preview service's log, secrets redacted."""
    raise _not_built()


# -- Settings and usage -------------------------------------------------------------------------------------

@router.get("/lab/settings", response_model=LabSettingsResponse)
async def lab_settings(user: LabUserDep) -> LabSettingsResponse:
    """The repositories and their bases, the lab's limits, its admins and this month's spending."""
    raise _not_built()


@router.post("/lab/settings/refresh-bases", response_model=LabSettingsResponse, responses=_refusals("state"))
async def lab_refresh_bases(user: LabUserDep) -> LabSettingsResponse:
    """Fetch every repository's base branch now. Open changes keep the base they started from.

    Refused when a fetch fails; the message names the repository.
    """
    raise _not_built()


@router.get("/lab/audit", response_model=LabAuditResponse)
async def lab_audit(
    user: LabUserDep,
    month: Annotated[Optional[str], Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
                                          description="``YYYY-MM``; the current UTC month when absent.")] = None,
) -> LabAuditResponse:
    """One UTC month of the lab's audit log: who did what to which change, and what it cost."""
    raise _not_built()


# -- Team memory --------------------------------------------------------------------------------------------

@router.get("/lab/memory", response_model=LabMemoryResponse)
async def lab_memory(user: LabUserDep) -> LabMemoryResponse:
    """The team notes the lab agent reads, and whether the memory is on."""
    raise _not_built()


@router.post("/lab/memory", response_model=LabNote, responses=_refusals("not_found", "state"))
async def lab_add_note(body: LabNoteInput, user: LabUserDep) -> LabNote:
    """Add a team note, approved by the user; with ``fromProposal`` it approves that proposal.

    Refused while the memory is off or full.
    """
    raise _not_built()


@router.put("/lab/memory/{note_id}", response_model=LabNote, responses=_refusals("not_found", "state"))
async def lab_update_note(note_id: NoteIdPath, body: LabNoteInput, user: LabUserDep) -> LabNote:
    """Replace a team note's text and scope."""
    raise _not_built()


@router.delete("/lab/memory/{note_id}", status_code=204, responses=_refusals("not_found", "state"))
async def lab_delete_note(note_id: NoteIdPath, user: LabUserDep) -> None:
    """Delete a team note."""
    raise _not_built()


@router.post("/lab/memory/proposals/dismiss", status_code=204, responses=_refusals("not_found"))
async def lab_dismiss_note_proposal(body: LabProposalRef, user: LabUserDep) -> None:
    """Dismiss a note the lab agent proposed: it is not kept."""
    raise _not_built()
