"""Request and response models of the BESSER Lab routes (``routers/lab_router.py``).

The editor's side of this contract is ``shared/api/lab.ts`` in the web editor. The ``lab`` field of a lab
run's ``done`` event (``services/spec_driven/sse_events.py``) is :class:`LabDone`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from besser.spec_driven_agent.lab.contracts import ChangeKind
from besser.spec_driven_agent.planning.specification import MAX_SPECIFICATION_CHARS
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.responses import ResponseModel

LabStep = Literal["plan", "build"]
LabStopReason = Literal["completed", "cancelled", "timeout", "cost_cap", "max_turns", "api_error"]

# A pull request title is a commit subject; its body is free text GitHub caps at 65,536 characters.
_MAX_PR_TITLE_CHARS = 200
_MAX_PR_BODY_CHARS = 65_536


class LabMessageRequest(BaseModel):
    """Body of ``POST /besser_api/lab/message``."""

    message: str = Field(..., min_length=1, max_length=MAX_SPECIFICATION_CHARS)
    #: ``auto``: plan when there is no change, else build; ``plan`` and ``build`` force the step.
    action: Literal["auto", "plan", "build"] = "auto"
    #: The editor's project; the class diagram it marks as the change's is the model a plan revision starts
    #: from and the build step follows.
    project: Optional[ProjectInput] = None


class LabPullRequestRequest(BaseModel):
    """Body of ``POST /besser_api/lab/pull-request``; null fields take their defaults."""

    #: The change the editor's card shows; another active change is refused (409). Null: the active change.
    changeId: Optional[str] = Field(default=None, max_length=64)
    title: Optional[str] = Field(default=None, max_length=_MAX_PR_TITLE_CHARS)
    body: Optional[str] = Field(default=None, max_length=_MAX_PR_BODY_CHARS)


class LabResumeRequest(BaseModel):
    """Body of ``POST /besser_api/lab/change/resume``."""

    changeId: str = Field(..., pattern=r"^[0-9a-f]{12}$")


class LabFileChange(ResponseModel):
    path: str
    status: Literal["added", "modified", "deleted", "renamed"]


class LabRepoChange(ResponseModel):
    """What the change touched in one repository, against its base branch."""

    repo: str
    branch: str
    base: str
    files: list[LabFileChange]


class LabCheck(ResponseModel):
    """One of a repository's own checks, as the build step last ran it."""

    id: str
    repo: str
    title: str
    passed: bool
    skipped: bool
    summary: str
    #: How long the check ran, in seconds; 0 for one that did not run.
    durationSeconds: float = 0.0


class LabChecklistItem(ResponseModel):
    id: str
    text: str
    #: None when the rule cannot be judged from the files
    done: Optional[bool]


class LabPlan(ResponseModel):
    kind: ChangeKind
    title: str
    summary: str
    steps: list[str]
    repos: list[str]
    questions: list[str]


class LabModelProposal(ResponseModel):
    """The class diagram a metamodel or diagram-type plan proposes, as the editor's model JSON."""

    title: str
    diagramType: Literal["ClassDiagram"]
    model: dict[str, Any]


class LabChangeView(ResponseModel):
    """A change as ``GET /lab/state`` shows it: its files now, and the checks of its last build."""

    step: LabStep
    changeId: str
    title: str
    kind: ChangeKind
    repoChanges: list[LabRepoChange]
    checks: list[LabCheck]
    checklist: list[LabChecklistItem]
    summary: str
    stopReason: LabStopReason


class LabDone(LabChangeView):
    """The ``lab`` field of a lab run's ``done`` event."""

    plan: Optional[LabPlan]
    modelProposal: Optional[LabModelProposal]
    #: The model's own closing account of a build ("" for a plan); ``summary`` ends with it too.
    finalText: str = ""


class LabParkedChange(ResponseModel):
    changeId: str
    title: str
    kind: ChangeKind
    #: seconds since the epoch
    createdAt: float


class LabPreviewService(ResponseModel):
    name: str
    running: bool
    detail: str


class LabPreview(ResponseModel):
    enabled: bool
    services: list[LabPreviewService]


class LabRun(ResponseModel):
    """The lab run in progress: ``/spec-driven/runs/{runId}/events`` follows it, the cancel route stops it."""

    runId: str
    #: seconds since the epoch
    startedAt: float


class LabStateResponse(ResponseModel):
    """Response of ``GET /lab/state`` and of every change action."""

    enabled: bool
    user: str
    change: Optional[LabChangeView]
    parked: list[LabParkedChange]
    preview: LabPreview
    run: Optional[LabRun]


class LabPullRequest(ResponseModel):
    repo: str
    number: int
    url: str
    branch: str
    base: str


class LabPullRequestsResponse(ResponseModel):
    pullRequests: list[LabPullRequest]
    #: Which checks the change passed before it was pushed, for the user (the lab runs only the quick ones).
    checksNote: str
