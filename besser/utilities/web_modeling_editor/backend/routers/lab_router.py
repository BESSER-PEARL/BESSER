"""BESSER Lab routes: the lab chat's run stream and the actions on the user's change.

Every route needs the ``X-GitHub-Session`` of a login on ``BESSER_LAB_ALLOWED_LOGINS`` (``LabUserDep``: 404
with the lab off, 401 without a valid session, 403 for another login). One run or change action at a time:
another answers 409, as does an action the working copy's state refuses (the message says why).
"""

# No ``from __future__ import annotations``: FastAPI resolves the endpoints' annotations at import.
import asyncio
import uuid
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from besser.utilities.web_modeling_editor.backend.dependencies import (
    LabServiceDep,
    SettingsDep,
    SpecDrivenRuntimeDep,
)
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabMessageRequest,
    LabPullRequestRequest,
    LabPullRequestsResponse,
    LabResumeRequest,
    LabStateResponse,
)
from besser.utilities.web_modeling_editor.backend.routers.auth import LabUserDep
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    PullRequestError,
    PullRequestSignInError,
)
from besser.utilities.web_modeling_editor.backend.services.lab.runner import LabRunner
from besser.utilities.web_modeling_editor.backend.services.lab.service import LabBusyError, LabStateError

router = APIRouter(prefix="/besser_api", tags=["lab"])

# What the lab refuses in its current state; each message is written for the user and holds no secret
# (git and GitHub messages are redacted where they are raised).
_REFUSALS = (LabBusyError, LabStateError, LabWorkspaceError, PullRequestError)


async def _refusing(call, *args):
    """Run a blocking lab action off the event loop; a refusal becomes a 409 with its message."""
    try:
        return await asyncio.to_thread(call, *args)
    except _REFUSALS as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/lab/message", response_class=StreamingResponse)
async def lab_message(body: LabMessageRequest, user: LabUserDep, lab: LabServiceDep,
                      runtime: SpecDrivenRuntimeDep, settings: SettingsDep) -> StreamingResponse:
    """Run one step of the lab agent (plan or build) for the user's message, as Spec-Driven SSE events.

    The ``start`` event carries the run id that ``POST /spec-driven/cancel/{run_id}`` stops; the ``done``
    event carries the step's result in ``lab``. The run goes on when the stream drops, until no one has
    followed it for ``BESSER_LLM_DISCONNECTED_GRACE_SECONDS``; ``/spec-driven/runs/{run_id}/events`` replays it.
    """
    run_id = uuid.uuid4().hex
    if not lab.try_begin_run(run_id):
        # ``GET /lab/state`` names a run in progress, to follow or stop.
        raise HTTPException(status_code=409, detail="A lab run or another change action is in progress.")
    cancel_event = await runtime.active_runs.reserve(run_id)
    if cancel_event is None:  # a fresh UUID is never taken
        lab.end_run()
        raise HTTPException(status_code=409, detail="Generated run ID is already active")
    runner = LabRunner(body, user=user, lab=lab, runtime=runtime, settings=settings, run_id=run_id,
                       cancel_event=cancel_event)
    try:
        await runtime.durable_runs.start(
            run_id, runner.stream(), cancel=cancel_event.set,
            on_abandoned=cancel_event.set if settings.llm_cancel_abandoned_runs else None,
            disconnect_grace_seconds=settings.llm_disconnected_grace_seconds,
        )
    except Exception:
        # The stream never started, so it releases nothing: release here.
        await runtime.active_runs.release(run_id, cancel_event)
        lab.end_run()
        raise
    return StreamingResponse(
        runtime.durable_runs.subscribe(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "X-BESSER-Run-Id": run_id},
    )


@router.get("/lab/state", response_model=LabStateResponse)
async def lab_state(user: LabUserDep, lab: LabServiceDep) -> LabStateResponse:
    """The signed-in user, the active change with its changed files, the parked changes and the preview."""
    return await _refusing(lab.state, user.login)


@router.post("/lab/pull-request", response_model=LabPullRequestsResponse)
async def lab_pull_request(body: LabPullRequestRequest, user: LabUserDep, lab: LabServiceDep) -> dict:
    """Push the active change and open a draft pull request in each repository it touched, editor first.

    409 when ``changeId`` is not the active change; 401 when GitHub refuses the user's sign-in (the token), so
    the editor asks them to sign in again. The full test suites are not run first; ``checksNote`` tells the user
    which checks the change passed.
    """
    try:
        opened, checks_note = await lab.open_pull_requests(user, change_id=body.changeId, title=body.title,
                                                           body=body.body)
    except PullRequestSignInError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None
    except _REFUSALS as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"pullRequests": [asdict(info) for info in opened], "checksNote": checks_note}


@router.post("/lab/change/park", response_model=LabStateResponse)
async def lab_park(user: LabUserDep, lab: LabServiceDep) -> LabStateResponse:
    """Commit the active change's work in progress and check out the base branches."""
    return await _refusing(lab.park, user.login)


@router.post("/lab/change/resume", response_model=LabStateResponse)
async def lab_resume(body: LabResumeRequest, user: LabUserDep, lab: LabServiceDep) -> LabStateResponse:
    """Check out a parked change again; 409 while another change is active."""
    return await _refusing(lab.resume, user.login, body.changeId)


@router.post("/lab/change/drop", response_model=LabStateResponse)
async def lab_drop(user: LabUserDep, lab: LabServiceDep) -> LabStateResponse:
    """Discard the active change: its uncommitted work, its lab branches and its notes."""
    return await _refusing(lab.drop, user.login)


@router.post("/lab/preview/reset", response_model=LabStateResponse)
async def lab_preview_reset(user: LabUserDep, lab: LabServiceDep) -> LabStateResponse:
    """Check out the base branches in the working copy; 409 while a change is active."""
    return await _refusing(lab.reset_preview, user.login)
