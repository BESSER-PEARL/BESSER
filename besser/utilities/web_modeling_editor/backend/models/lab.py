"""Request and response models of the BESSER Lab routes (``routers/lab_router.py``).

The editor's side of this contract is ``shared/api/lab.ts`` in the web editor, typed from the OpenAPI schema
these models make. The ``lab`` field of a lab run's ``done`` event (``services/spec_driven/sse_events.py``) is
:class:`LabDone`. A lab refusal reaches the client as ``{"detail": {"code", "message"}}``
(:class:`LabRefusalResponse`).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from besser.spec_driven_agent.lab.contracts import ChangeKind
from besser.spec_driven_agent.lab.memory import MAX_BODY_CHARS, MAX_TITLE_CHARS
from besser.spec_driven_agent.planning.specification import MAX_SPECIFICATION_CHARS
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.responses import ResponseModel

LabStep = Literal["plan", "build"]
LabStopReason = Literal["completed", "cancelled", "timeout", "cost_cap", "max_turns", "api_error"]
#: ``auto``: plan while the change has no approved plan, else build; ``plan`` and ``build`` force the step.
LabAction = Literal["auto", "plan", "build"]
#: The repositories a change spans (``lab/contracts.py``: ``REPO_BESSER``, ``REPO_FRONTEND``, ``REPO_MODELING_AGENT``).
LabRepoKey = Literal["besser", "frontend", "modeling_agent"]
#: ``draft``: no branches yet (before the first plan is submitted); ``open``: branches and working trees;
#: ``closed``: merged or closed unmerged, trees and branches removed, state kept.
LabChangeStatus = Literal["draft", "open", "closed"]
#: Where a change stands, for the home's list and the change header.
LabPhase = Literal["draft", "planned", "running", "built", "checks_failing", "stopped", "in_review", "merged",
                   "closed"]
LabRefusalCode = Literal["change_busy", "runs_full", "user_cap", "instance_cap", "not_owner", "preview_full",
                         "not_found", "state"]
LabPreviewServiceName = Literal["backend", "frontend", "modeling_agent"]
LabPreviewState = Literal["stopped", "starting", "running", "failed"]
LabLinkMode = Literal["reflink", "hardlink"]
LabNoteKind = Literal["decision", "mistake", "convention"]
LabAuditAction = Literal["change_created", "change_opened", "run", "run_refused", "pull_requests_opened",
                         "change_closed", "change_dropped", "base_updated", "bases_refreshed", "preview_slot_taken"]

CHANGE_ID_PATTERN = r"^[0-9a-f]{12}$"

# A pull request title is a commit subject; its body is free text GitHub caps at 65,536 characters.
_MAX_PR_TITLE_CHARS = 200
_MAX_PR_BODY_CHARS = 65_536
# A change starts from a few issues (the plan fetches at most 3 references); more is not a real request.
_MAX_ISSUES = 20
# The ticked review comments one task acts on; a pull request's review threads are read 100 at a time.
_MAX_REVIEW_COMMENTS = 100
# A repository-relative path (a diff's file, a note's docs target); far beyond any path in the repositories.
_MAX_PATH_CHARS = 1_024
# Ids the client echoes back: GraphQL node ids, note and proposal ids.
_MAX_ID_CHARS = 128


# -- Requests --------------------------------------------------------------------------------------------

class LabIssueRef(BaseModel):
    """A GitHub issue of one of the lab's repositories, by number."""

    repo: LabRepoKey
    number: int = Field(..., ge=1)


class LabReviewTask(BaseModel):
    """A build step that acts on the change's pull requests (the ``review`` of a lab message)."""

    task: Literal["fix_ci", "address_comments", "resolve_conflicts"]
    #: The repository whose pull request to act on; null: every pull request of the change.
    repo: Optional[LabRepoKey] = None
    #: Review comments the user ticked (GraphQL node ids), whoever wrote them.
    commentIds: list[str] = Field(default_factory=list, max_length=_MAX_REVIEW_COMMENTS)


class LabMessageRequest(BaseModel):
    """Body of ``POST /besser_api/lab/message``: a message to one change, which runs its next step."""

    message: str = Field(..., min_length=1, max_length=MAX_SPECIFICATION_CHARS)
    action: LabAction = "auto"
    #: The editor's project; the class diagram it marks as the change's is the model a plan revision starts
    #: from and the build step follows.
    project: Optional[ProjectInput] = None
    #: The change the message goes to: the one in the editor's URL. A new change comes from ``POST /lab/change``.
    changeId: str = Field(..., pattern=CHANGE_ID_PATTERN)
    #: A task on the change's pull requests; it forces the build step.
    review: Optional[LabReviewTask] = None
    #: The issue the change starts from; its text is put in the plan's first message.
    issue: Optional[LabIssueRef] = None


class LabPullRequestRequest(BaseModel):
    """Body of ``POST /besser_api/lab/pull-request``; null fields take their defaults."""

    changeId: str = Field(..., pattern=CHANGE_ID_PATTERN)
    title: Optional[str] = Field(default=None, max_length=_MAX_PR_TITLE_CHARS)
    body: Optional[str] = Field(default=None, max_length=_MAX_PR_BODY_CHARS)
    #: Issues of the change the pull requests close ("Closes"); the others are listed as related. Each must be
    #: one of the change's issues.
    closesIssues: list[LabIssueRef] = Field(default_factory=list, max_length=_MAX_ISSUES)


class LabChangeRequest(BaseModel):
    """Body of the routes that act on one change (drop, close)."""

    changeId: str = Field(..., pattern=CHANGE_ID_PATTERN)


class LabUpdateBranchRequest(BaseModel):
    """Body of ``POST /besser_api/lab/change/update-base``: bring one repository of a change up to its base."""

    changeId: str = Field(..., pattern=CHANGE_ID_PATTERN)
    repo: LabRepoKey


class LabTakeSlotRequest(BaseModel):
    """Body of ``POST /besser_api/lab/preview/slot``: show this change in a preview slot."""

    changeId: str = Field(..., pattern=CHANGE_ID_PATTERN)


class LabTouchSlotRequest(BaseModel):
    """Body of ``POST /besser_api/lab/preview/slot/touch``: the slot is still in use."""

    slot: int = Field(..., ge=0)


class LabPreviewRestartRequest(BaseModel):
    """Body of ``POST /besser_api/lab/preview/restart``."""

    slot: int = Field(..., ge=0)
    #: The service to restart; null: all three.
    service: Optional[LabPreviewServiceName] = None


class LabProposalRef(BaseModel):
    """A note the lab agent proposed during a run of a change, pending approval."""

    changeId: str = Field(..., pattern=CHANGE_ID_PATTERN)
    #: The proposal's id within its change.
    id: str = Field(..., min_length=1, max_length=_MAX_ID_CHARS)


class LabNoteInput(BaseModel):
    """Body of ``POST`` and ``PUT /besser_api/lab/memory``: a team note as a lab user approves or edits it."""

    kind: LabNoteKind
    # The team memory's own limits (``lab/memory.py``): one screen, one idea.
    title: str = Field(..., min_length=1, max_length=MAX_TITLE_CHARS)
    body: str = Field(..., min_length=1, max_length=MAX_BODY_CHARS)
    #: Repositories the note applies to; empty: all.
    repos: list[LabRepoKey] = Field(default_factory=list)
    #: Kinds of change the note applies to; empty: every kind.
    changeKinds: list[ChangeKind] = Field(default_factory=list)
    #: The ``CLAUDE.md`` or skill the note belongs in when it states how the code works; required for a
    #: ``convention``.
    docsTarget: str = Field(default="", max_length=_MAX_PATH_CHARS)
    #: The proposal this note approves; it is removed from the change's pending proposals.
    fromProposal: Optional[LabProposalRef] = None


# -- A change and its last run -----------------------------------------------------------------------------

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


class LabBuildReport(ResponseModel):
    """The lab agent's own account of a build: what it did, where it departed from the plan, what is left."""

    summary: str
    deviations: list[str]
    notDone: list[str]


class LabChangeCost(ResponseModel):
    """What every run of a change cost together."""

    usd: float
    turns: int
    elapsedSeconds: float
    runs: int


class LabPullRequest(ResponseModel):
    """A pull request the lab opened for a change."""

    repo: str
    number: int
    url: str
    branch: str
    base: str


class LabIssue(ResponseModel):
    """An issue a change started from."""

    repo: str
    number: int
    title: str
    #: The user ticked "Closes" for it: the pull request body closes it at release.
    closes: bool


class LabChangeView(ResponseModel):
    """A change's files now, and what its last run reported."""

    step: LabStep
    changeId: str
    title: str
    kind: ChangeKind
    repoChanges: list[LabRepoChange]
    checks: list[LabCheck]
    checklist: list[LabChecklistItem]
    summary: str
    stopReason: LabStopReason
    #: The last build's report; None after a plan or a build that wrote none.
    report: Optional[LabBuildReport] = None
    pullRequests: list[LabPullRequest] = Field(default_factory=list)
    cost: LabChangeCost
    issues: list[LabIssue] = Field(default_factory=list)


class LabNoteProposal(ResponseModel):
    """A team note the lab agent proposed during a build; a lab user approves, edits or dismisses it."""

    id: str
    kind: LabNoteKind
    title: str
    body: str
    repos: list[str]
    changeKinds: list[ChangeKind]
    docsTarget: str
    #: One sentence for the approver: why the team should keep it.
    why: str


class LabDone(LabChangeView):
    """The ``lab`` field of a lab run's ``done`` event."""

    plan: Optional[LabPlan]
    modelProposal: Optional[LabModelProposal]
    #: The model's own closing account of a build ("" for a plan); ``summary`` ends with it too.
    finalText: str = ""
    noteProposals: list[LabNoteProposal] = Field(default_factory=list)


# -- Refusals, users, bases and budget --------------------------------------------------------------------

class LabRefusalDetail(ResponseModel):
    """Why the lab refused an action, written for the user."""

    code: LabRefusalCode
    message: str


class LabRefusalResponse(ResponseModel):
    """The body of every lab refusal: ``change_busy`` 409, ``runs_full`` 429, ``user_cap`` 409, ``instance_cap``
    409, ``not_owner`` 403, ``preview_full`` 409, ``not_found`` 404, ``state`` 409 (what the change's state
    refuses)."""

    detail: LabRefusalDetail


class LabUserView(ResponseModel):
    """The signed-in lab user."""

    login: str
    #: On ``BESSER_LAB_ADMIN_LOGINS``: may drop any change and stop any run.
    isAdmin: bool


class LabBase(ResponseModel):
    """A repository's base clone: the upstream branch new changes start from."""

    repo: str
    #: ``owner/name`` on GitHub, from ``BESSER_LAB_REPO_*``.
    upstream: str
    branch: str
    sha: str
    #: When the base was last fetched, in seconds since the epoch; 0 when never.
    fetchedAt: float
    #: The upstream is not a BESSER-PEARL repository (a fork, for testing).
    fork: bool


class LabMeter(ResponseModel):
    """Spending against one monthly cap, in USD."""

    spentUsd: float
    #: Held by runs in progress.
    reservedUsd: float
    capUsd: float


class LabMeters(ResponseModel):
    """This UTC month's spending of the signed-in user and of the whole lab."""

    #: ``YYYY-MM``
    month: str
    user: LabMeter
    instance: LabMeter


class LabRun(ResponseModel):
    """A lab run in progress: ``/spec-driven/runs/{runId}/events`` follows it, the cancel route stops it."""

    runId: str
    changeId: str
    #: Who started it.
    login: str
    #: None until the run has decided its step (an ``auto`` message).
    step: Optional[LabStep]
    #: seconds since the epoch
    startedAt: float


class LabRunsView(ResponseModel):
    """The runs in progress across all changes, and how many may run at once."""

    inProgress: list[LabRun]
    cap: int


class LabMemoryInfo(ResponseModel):
    """Whether the team memory is on, and how full it is."""

    enabled: bool
    #: Why it is off; "" when on.
    reason: str
    count: int
    max: int


class LabHomeResponse(ResponseModel):
    """Response of ``GET /lab/home``: who is signed in and what the lab can do now."""

    user: LabUserView
    bases: list[LabBase]
    runs: LabRunsView
    meters: LabMeters
    previewSlots: int
    #: How change environments share the stores: ``hardlink`` means in-place writes can reach other changes.
    linkMode: LabLinkMode
    memory: LabMemoryInfo


# -- Changes and their thread --------------------------------------------------------------------------

class LabRunRecord(ResponseModel):
    """One finished run of a change: its cost and who asked for it."""

    runId: str
    step: LabStep
    login: str
    model: str
    usd: float
    turns: int
    elapsedSeconds: float
    stopReason: LabStopReason
    #: seconds since the epoch
    startedAt: float


class LabChangeSummary(ResponseModel):
    """A change as the list of changes shows it."""

    changeId: str
    title: str
    kind: ChangeKind
    status: LabChangeStatus
    #: How a closed change ended; None while it is not closed.
    outcome: Optional[Literal["merged", "closed"]]
    owner: str
    phase: LabPhase
    #: Repositories with changes.
    repos: list[str]
    #: Commits each repository's base moved since the change started from it.
    behind: dict[str, int]
    #: The run in progress on this change.
    run: Optional[LabRun]
    pullRequests: list[LabPullRequest]
    cost: LabChangeCost
    #: Logins that sent the change's messages, first request first.
    requestedBy: list[str]
    #: seconds since the epoch
    createdAt: float
    updatedAt: float
    closedAt: Optional[float]


class LabChangesResponse(ResponseModel):
    """Response of ``GET /lab/changes``: open changes and drafts by last update, then at most 10 closed ones,
    newest first."""

    changes: list[LabChangeSummary]


class LabRepoBaseStatus(ResponseModel):
    """Where a change's repository started, and how far its base moved since."""

    repo: str
    upstream: str
    branch: str
    #: The base commit the change started from.
    sha: str
    #: The base branch's commit in the lab's base clone now.
    latest: str
    behind: int


class LabRunError(ResponseModel):
    """Why a run failed, written for the user."""

    code: str
    message: str


class LabThreadMessage(ResponseModel):
    """A message a lab user sent to the change."""

    type: Literal["message"]
    id: str
    #: seconds since the epoch
    at: float
    login: str
    text: str
    action: LabAction
    #: The class diagram the message carried (the editor's model JSON).
    model: Optional[dict[str, Any]]


class LabThreadRun(ResponseModel):
    """A run of the change: running, or how it ended."""

    type: Literal["run"]
    runId: str
    at: float
    login: str
    status: Literal["running", "done", "failed", "cancelled"]
    lab: Optional[LabDone]
    error: Optional[LabRunError]
    #: From the change's run records; None while running.
    costUsd: Optional[float]
    #: The plan's revision number for a plan run (1 for the first plan).
    planVersion: Optional[int]


class LabThreadPullRequests(ResponseModel):
    """Pull requests opened or updated for the change."""

    type: Literal["pull_requests"]
    at: float
    login: str
    pullRequests: list[LabPullRequest]
    checksNote: str


class LabThreadNotice(ResponseModel):
    """Something that happened to the change, for every viewer (an update from the base, a merge)."""

    type: Literal["notice"]
    at: float
    login: str
    text: str


LabThreadItem = Annotated[Union[LabThreadMessage, LabThreadRun, LabThreadPullRequests, LabThreadNotice],
                          Field(discriminator="type")]


class LabChangeDetail(LabChangeSummary):
    """One change with its branches, its bases, its last run, its thread and its run records."""

    branches: dict[str, str]
    bases: list[LabRepoBaseStatus]
    #: None for a draft without a run.
    view: Optional[LabChangeView]
    thread: list[LabThreadItem]
    runs: list[LabRunRecord]


class LabPullRequestsResponse(ResponseModel):
    pullRequests: list[LabPullRequest]
    #: Which checks the change passed before it was pushed, for the user (the lab runs only the quick ones).
    checksNote: str


# -- Review: diffs, pull request status, update from base -------------------------------------------------

class LabFileDiff(ResponseModel):
    """One file's diff against the change's base."""

    path: str
    #: The old path of a rename.
    oldPath: Optional[str]
    status: Literal["added", "modified", "deleted", "renamed"]
    #: None when binary.
    additions: Optional[int]
    deletions: Optional[int]
    binary: bool
    #: The unified diff, secrets redacted; None when binary or omitted.
    patch: Optional[str]
    #: Why ``patch`` is missing: binary, larger than one file may be, or the response already full.
    omitted: Optional[Literal["binary", "too_large", "response_full"]]
    #: Secrets replaced in ``patch``.
    redactions: int


class LabRepoDiff(ResponseModel):
    repo: LabRepoKey
    branch: str
    base: str
    files: list[LabFileDiff]


class LabDiffResponse(ResponseModel):
    """Response of ``GET /lab/diff``: the change's working trees against their bases."""

    changeId: str
    repos: list[LabRepoDiff]


class LabCheckRun(ResponseModel):
    """A CI check of a pull request's head commit."""

    name: str
    #: ``github-actions``, ``readthedocs``; "" for a commit status.
    app: str
    state: Literal["pending", "success", "failure", "neutral", "skipped"]
    url: Optional[str]
    #: The end of a failing Actions job's log, redacted.
    logTail: Optional[str]
    #: Why there is no log tail; "" otherwise.
    logNote: str


class LabReviewComment(ResponseModel):
    """A review or conversation comment on a pull request."""

    #: GraphQL node id.
    id: str
    author: str
    authorAssociation: str
    #: The author is an owner, member or collaborator: the agent acts on it by default.
    trusted: bool
    body: str
    url: str
    createdAt: str
    #: None: a general comment.
    path: Optional[str]
    line: Optional[int]
    startLine: Optional[int]
    resolved: bool
    outdated: bool


class LabReview(ResponseModel):
    """A submitted pull request review."""

    author: str
    body: str
    url: str
    submittedAt: Optional[str]
    state: Literal["APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING"]


class LabPullRequestStatus(ResponseModel):
    """A pull request of the change as GitHub reports it now."""

    repo: LabRepoKey
    number: int
    url: str
    draft: bool
    headSha: str
    state: Literal["open", "merged", "closed"]
    #: The lab's last push; it differs from ``headSha`` when someone else pushed.
    pushedSha: Optional[str]
    mergeable: Literal["mergeable", "conflicting", "unknown"]
    #: GitHub's ``mergeStateStatus`` in lower case (``behind``, ``clean``, ...).
    mergeState: str
    reviewDecision: Optional[Literal["APPROVED", "CHANGES_REQUESTED", "REVIEW_REQUIRED"]]
    checksState: Literal["pending", "success", "failure", "none"]
    checks: list[LabCheckRun]
    reviews: list[LabReview]
    #: Inline and general comments, oldest first.
    comments: list[LabReviewComment]


class LabReviewResponse(ResponseModel):
    """Response of ``GET /lab/review``: the change's pull requests on GitHub and what the change can do next."""

    changeId: str
    pullRequests: list[LabPullRequestStatus]
    lifecycle: Literal["open", "partly_merged", "merged", "closed"]
    #: No pull request is open, each is merged or closed, and nothing is unpushed: the change can close (the
    #: editor closes it by itself only when ``lifecycle`` is ``merged``; a closed one waits for a click).
    closable: bool
    #: Why it cannot close yet; "" otherwise.
    closableNote: str
    #: When to ask again; None: nothing is pending.
    nextPollSeconds: Optional[int]
    #: seconds since the epoch
    fetchedAt: float


class LabUpdateBranchResponse(ResponseModel):
    """Response of ``POST /lab/change/update-base``."""

    repo: str
    outcome: Literal["updated", "up_to_date", "conflict", "refused"]
    #: Files with conflicts, left for a "Resolve conflicts" build step.
    conflicts: list[str]
    message: str


# -- Preview ----------------------------------------------------------------------------------------------

class LabPreviewService(ResponseModel):
    """One of a preview slot's three services."""

    name: LabPreviewServiceName
    state: LabPreviewState
    detail: str
    #: seconds since the epoch; None while stopped.
    startedAt: Optional[float]
    #: The last redacted log lines of a failed service; empty otherwise.
    logTail: list[str]


class LabPreview(ResponseModel):
    """A preview: the change's backend, editor and modeling agent, each in its own sandbox."""

    enabled: bool
    #: What the Try it tab frames or opens; None when off.
    url: Optional[str]
    #: False only where the platform has no sandbox: the preview then runs unconfined.
    confined: bool
    #: Why it is off or unconfined, or what it is preparing; "" otherwise.
    detail: str
    services: list[LabPreviewService]


class LabPreviewSlot(LabPreview):
    """A preview slot and the change it shows."""

    slot: int
    #: None: the slot shows no change.
    changeId: Optional[str]
    title: Optional[str]
    owner: Optional[str]
    #: The last login that used the slot.
    usedBy: Optional[str]
    #: seconds since the epoch
    startedAt: Optional[float]
    lastUsedAt: Optional[float]


class LabPreviewView(ResponseModel):
    """Response of ``GET /lab/preview`` and of the preview actions: every slot."""

    enabled: bool
    #: Why the preview is off; "" otherwise.
    detail: str
    slots: list[LabPreviewSlot]


class LabPreviewLog(ResponseModel):
    """The end of one preview service's log, redacted."""

    slot: int
    service: LabPreviewServiceName
    lines: list[str]
    #: The log holds more lines than were sent.
    truncated: bool


# -- Settings and usage -------------------------------------------------------------------------------------

class LabLimitsView(ResponseModel):
    """The lab's limits, as the server is configured."""

    perRunUsd: float
    userMonthlyUsd: float
    instanceMonthlyUsd: float
    maxConcurrentRuns: int
    previewSlots: int
    #: 0: bases are refreshed by hand and when a change opens.
    baseRefreshMinutes: int


class LabSettingsResponse(ResponseModel):
    """Response of ``GET /lab/settings``: repositories, limits, admins and this month's spending."""

    bases: list[LabBase]
    limits: LabLimitsView
    admins: list[str]
    meters: LabMeters
    linkMode: LabLinkMode


class LabAuditEntry(ResponseModel):
    """One line of the lab's monthly audit log."""

    #: seconds since the epoch
    at: float
    login: str
    action: LabAuditAction
    changeId: Optional[str]
    runId: Optional[str]
    step: Optional[LabStep]
    costUsd: Optional[float]
    stopReason: Optional[LabStopReason]
    detail: str


class LabAuditResponse(ResponseModel):
    """Response of ``GET /lab/audit``: one UTC month of the audit log."""

    #: ``YYYY-MM``
    month: str
    entries: list[LabAuditEntry]
    #: USD spent per login this month.
    totalsByLogin: dict[str, float]


# -- Team memory --------------------------------------------------------------------------------------------

class LabNote(ResponseModel):
    """An approved team note; the lab agent reads it when it applies to the change."""

    id: str
    kind: LabNoteKind
    title: str
    body: str
    repos: list[str]
    changeKinds: list[ChangeKind]
    docsTarget: str
    #: The change it was proposed in; "" for a note written by hand.
    sourceChange: str
    sourceUrl: str
    approvedBy: str
    #: seconds since the epoch
    createdAt: float
    updatedAt: float
    #: 0 when never read.
    lastReadAt: float
    reads: int
    #: No change has read it for a long time: a person decides whether to keep it.
    stale: bool


class LabMemoryResponse(ResponseModel):
    """Response of ``GET /lab/memory``."""

    enabled: bool
    #: Why the memory is off; "" when on.
    reason: str
    notes: list[LabNote]
