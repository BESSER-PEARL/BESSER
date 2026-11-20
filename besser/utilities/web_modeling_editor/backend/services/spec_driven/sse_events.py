"""SSE event schema for the spec-driven generation endpoint.

Every event is a Pydantic model with an ``event`` discriminator. The
``format_sse`` helper serialises an event into the canonical SSE frame:

    event: <name>\\n
    data: <compact-json>\\n
    \\n

The ``event:`` header enables browser ``EventSource.addEventListener``
dispatch, and the JSON body repeats the ``event`` field so that simple
``onmessage`` / fetch-reader consumers can switch on it too. Both styles
work without special-casing.

The endpoint never includes user API keys in any event — there is no
``api_key`` field on any model so it cannot leak by accident.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, Field

from besser.utilities.secret_redaction import (
    redact_data,
)


class BaseSseEvent(BaseModel):
    """Common parent for all SSE events."""

    event: str


class StartEvent(BaseSseEvent):
    """Emitted once at the start of a run, before any BUML assembly."""

    event: Literal["start"] = "start"
    runId: str
    provider: Literal[
        "anthropic", "openai", "mistral", "nebius", "free", "sponsored", "pia", "gateway"
    ]
    llmModel: str
    maxCost: float
    maxRuntime: int


PhaseName = Literal["select", "generate", "gap", "customize", "validate"]


class PhaseEvent(BaseSseEvent):
    """Coarse-grained phase marker emitted as the pipeline advances.

    ``usd``, ``turns`` and ``elapsedSeconds`` are the run's totals when the
    phase started, measured like the ``cost`` event. A phase's own cost,
    turns and duration are the difference to the next phase, or to the final
    ``cost`` event for the last one.
    """

    event: Literal["phase"] = "phase"
    phase: PhaseName
    message: str
    usd: Optional[float] = None
    turns: Optional[int] = None
    elapsedSeconds: Optional[float] = None


class PhaseUpdateEvent(BaseSseEvent):
    """Adds details to an already-emitted phase row.

    Distinct from PhaseEvent so the frontend can update an existing
    timeline entry in place (rather than appending a duplicate row).
    Used by the gap analyser to surface the task list after the planning
    LLM call returns. ``details`` is rendered behind a chevron on the
    spec-driven card.
    """

    event: Literal["phase_update"] = "phase_update"
    phase: PhaseName
    details: str
    message: Optional[str] = None


class StepEvent(BaseSseEvent):
    """What the current phase is doing now, while one of its steps runs.

    Emitted during ``validate``: each check round, requirement judge call and
    fix attempt, where one step can take minutes. A step lasts until the next
    step, phase or terminal event. ``elapsedSeconds`` is the run's elapsed
    time when the step started, measured like the ``cost`` event, so a client
    can show how long the step has been running from the ``cost`` ticks.
    """

    event: Literal["step"] = "step"
    phase: PhaseName
    message: str
    elapsedSeconds: Optional[float] = None


class TextDeltaEvent(BaseSseEvent):
    """A streaming text delta from the underlying LLM response.

    ``delta`` may contain newlines; they're JSON-escaped by
    ``model_dump_json`` so the SSE frame stays single-line until the
    terminating blank line.
    """

    event: Literal["text"] = "text"
    delta: str


ToolCallStatus = Literal["executing", "done", "error"]


class ToolCallEvent(BaseSseEvent):
    """A tool invocation made by the LLM during Phase 2."""

    event: Literal["tool_call"] = "tool_call"
    turn: int
    tool: str
    status: ToolCallStatus = "executing"
    summary: Optional[str] = None
    # What the call was about (path / action / ids) plus how many calls the model
    # batched into this turn. Without it a finished run cannot be diagnosed from
    # the durable store (e.g. many task_list calls: bookkeeping or a
    # livelock?). Never contains file content.
    detail: Optional[str] = None


class ModelUpdateEvent(BaseSseEvent):
    """The LLM serving this run changed mid-run.

    Emitted when the provider's outage fallback switches the run to a
    different model (see ``OpenAIProvider._activate_fallback`` — the
    switch is sticky for the rest of the run). Without this event the
    run card keeps showing the model from the ``start`` event even
    though every subsequent turn is served by the fallback.

    ``reason`` is a short machine-readable cause the frontend maps to
    user-facing copy; today the only producer is the outage fallback
    (``primary_unavailable``).
    """

    event: Literal["model_update"] = "model_update"
    model: str
    previousModel: Optional[str] = None
    reason: str = "primary_unavailable"


class CostEvent(BaseSseEvent):
    """Periodic cost / runtime tick emitted by the cost emitter task.

    ``servedModel`` carries the model name the upstream provider
    actually served on the most recent call. Useful for audits where
    the requested model (e.g. ``gpt-5.5``) may get aliased server-side
    to a different variant. Optional because not every chunk knows it
    (Anthropic doesn't expose it the same way).
    """

    event: Literal["cost"] = "cost"
    usd: float
    turns: int
    elapsedSeconds: float
    servedModel: Optional[str] = None


class FindingsEvent(BaseSseEvent):
    """What one validation round found, during the ``validate`` phase.

    ``attempt`` is 0 for the first check and n for the re-check after fix
    attempt n; the tool calls between two findings events are that attempt's
    repairs. ``restored`` marks the last event of a run that went back to a
    better earlier tree: the findings then describe the delivered app.
    ``blockers`` is capped and each message clipped; the counts are true totals.
    """

    event: Literal["findings"] = "findings"
    attempt: int
    blockers: list[str] = Field(default_factory=list)
    blockerCount: int = 0
    issueCount: int = 0
    restored: bool = False


VerificationKind = Literal["requirement", "ocl_constraint", "api_workflow", "check"]


class VerificationItem(BaseModel):
    """One thing the run checked, could not check, or found missing.

    ``how`` is set on a verified item (what was actually run) and on a
    found-in-code item (the cited source); ``why`` on the other two (why
    nothing reached it, or what the delivered code is missing).
    Rendered in full by the client, so a value cut server-side ends in a
    visible truncation mark rather than mid-word.
    """

    kind: VerificationKind
    id: str = ""
    what: str
    how: Optional[str] = None
    why: Optional[str] = None
    # A requirement item's RequirementArtifact id in the run's trace (``DoneEvent.trace``):
    # the editor's id for a requirement it sent, ``<runId>:R<n>`` for one the run extracted.
    # ``id`` keeps the ``R<n>`` label the run's messages use.
    traceRef: Optional[str] = None


class VerificationCounts(BaseModel):
    """True totals, named in the schema.

    As a bare dict the three keys were unenforced, and a typo would have read
    as a missing key on the client rather than failing here.
    """

    verified: int = 0
    foundInCode: int = 0
    notVerified: int = 0
    shippedUnenforced: int = 0
    failed: int = 0


class VerificationReport(BaseModel):
    """The four states a blocker count collapses into one number.

    An app scored 11/11 with a passing booking workflow still double-sold
    rooms: two OCL constraints failed conversion and never reached the code.
    The run knew — it listed both and reported 21 blockers — but the number
    hid which. ``verified`` is "we ran it against the app and it works",
    ``foundInCode`` is "its source is cited, but nothing ran it",
    ``notVerified`` is "we could not check" (a finding not confirmed, a check
    that did not run), ``shippedUnenforced`` is "we checked and it is not met"
    (a rule missing in the code, a cited implementation that does not do it),
    ``failed`` is "we ran it and it failed" (a test workflow the agent wrote;
    the test or the app may be wrong, so it is not a rule shipped unenforced).
    A finding counted as a blocker is never in ``notVerified``. ``counts``
    holds the true totals; the lists are capped. ``failed`` is additive.
    """

    verified: list[VerificationItem] = Field(default_factory=list)
    foundInCode: list[VerificationItem] = Field(default_factory=list)
    notVerified: list[VerificationItem] = Field(default_factory=list)
    shippedUnenforced: list[VerificationItem] = Field(default_factory=list)
    failed: list[VerificationItem] = Field(default_factory=list)
    counts: VerificationCounts = Field(default_factory=VerificationCounts)


class DoneEvent(BaseSseEvent):
    """Terminal event on successful completion.

    ``downloadUrl`` points to the sibling ``GET /spec-driven/download/{runId}``
    endpoint which single-use-serves the generated ZIP or file. ``runId``
    is carried explicitly so clients don't have to parse it back out of
    the URL. The ``recipe`` field carries the contents of
    ``.besser_recipe.json`` that ``LLMOrchestrator`` writes at the end
    of every run.
    """

    event: Literal["done"] = "done"
    runId: str = ""
    downloadUrl: str
    fileName: str
    isZip: bool
    recipe: dict[str, Any] = Field(default_factory=dict)
    # Lightweight "what was generated" summary for the UI: how many user files
    # the run produced and the top-level entries (dir/file names at the project
    # root), so the client can show a concrete summary instead of a generic
    # "code is ready". Capped so a large tree can't bloat the event.
    fileCount: int = 0
    topLevel: list[str] = Field(default_factory=list)
    # Total LLM tokens the run consumed (0 for a purely deterministic run). Shown
    # on the card as "N tokens used" — the counterpart to the deterministic
    # card's "0 tokens".
    tokensUsed: int = 0
    # True when the run produced downloadable output but the customization
    # loop did NOT finish cleanly (e.g. a provider rate-limit, cost/runtime
    # cap, or turn cap cut it short). Clients should warn that the output
    # may be incomplete rather than report an unqualified success.
    incomplete: bool = False
    incompleteReason: Optional[str] = None
    # True when the run ended because a cancel request was honoured after it
    # had written files: the output is what it had made by then.
    cancelled: bool = False
    # 0 on a run that was cut short (the loop did not finish, or a cancel stopped
    # Phase 3), so clients use their cut-short copy. Otherwise, with
    # ``incomplete`` true: the unresolved blockers. With ``incomplete`` false:
    # the legacy "could not be verified" count released editors show, the
    # blockers (requirements found missing or unsupported, failed workflows,
    # which never set ``incomplete``) plus ``notConfirmedCount`` plus the notes
    # of checks the server has switched off (counted there before), and at
    # least one per class of checks that never ran when ``unverified``. New
    # clients read ``unresolvedBlockerCount`` and ``notConfirmedCount``.
    blockerCount: int = 0
    # Findings that are not confirmed: a check that reached no conclusion (a
    # citation not located, a guessed create refused, a check that did not run,
    # a dependency install left undone). Never blockers. A check the server has
    # switched off (``checksOff``) is not counted here. Additive.
    notConfirmedCount: int = 0
    # The real unresolved blockers, on every run (also one cut short): what
    # ``blockerCount`` cannot say on a finished run, where it is padded for
    # released editors. Additive.
    unresolvedBlockerCount: int = 0
    # True when the run is not incomplete but a whole class of checks left no
    # evidence on the delivered app: it was not checked against the request, it
    # was never started, no record could be created through its API, or its
    # frontend was never built. ``unverifiedReason`` is the sentence to show
    # ("The app was generated, but ..."; "The app was updated, but ..." on a
    # modify run). False whenever ``incomplete`` is true. Additive.
    unverified: bool = False
    unverifiedReason: Optional[str] = None
    # The classes of checks (``requirements``, ``runtime``, ``build``) this server has
    # switched off: the requirements check, the startup check, the build check or shell
    # tools. A property of the server, the same on every run there, so it never sets
    # ``unverified``; the editor may say it once. Additive.
    checksOff: list[str] = Field(default_factory=list)
    # What the run verified, could not verify, and shipped unenforced. This is
    # the field to render: blockerCount is one number over all three states.
    # Additive — existing clients that ignore it are unaffected.
    verification: VerificationReport = Field(default_factory=VerificationReport)
    # Three-way authorship split over the final output tree —
    # ``{generator_untouched, generator_llm_modified, llm_authored, total,
    # *_pct}`` — how much of the app the deterministic pipeline carried vs.
    # the LLM. Best-effort: ``None`` when the split could not be computed.
    fileSplit: Optional[dict[str, Any]] = None
    # Line-level authorship from the run's Agent Trace record
    # (``.agent-trace/<runId>.json``): ``{aiLines, aiLinesThisRun, totalLines,
    # aiPct, fileCount, tracePath, files: [{path, aiLines, totalLines, aiPct}]}``
    # with ``files`` sorted by AI lines and capped. ``aiLines`` counts lines
    # written by any run's LLM, ``aiLinesThisRun`` only this run's. Best-effort:
    # ``None`` when the record could not be built.
    aiShare: Optional[dict[str, Any]] = None
    # The run's trace: a trace-file fragment (``projectId`` null, the JSON of
    # ``trace_buml_to_json``, as the traced generation endpoints return) that the
    # editor merges into the project trace. One CodeGeneration ``spec-driven:<runId>``
    # with the generator and LLM calls, the files they wrote, the requirements and the
    # judge's requirement -> code links (see ``state/trace_model.py``). ``None`` when the
    # request did not set ``traceabilityEnabled`` or it could not be built; the same JSON
    # ships in the ZIP as ``.besser_trace_model.json``.
    trace: Optional[dict[str, Any]] = None
    # What this run changed, for the card's "What changed" summary.
    # ``mode`` is "create" for a new app and "modify" for an update of an
    # existing one; ``changedFiles`` are the paths the agent wrote, edited or
    # deleted through its file tools (capped; ``changedFileCount`` is the
    # total); ``modelChanges`` describe how the model differs from the one
    # the base app was built from (modify runs only).
    mode: Literal["create", "modify"] = "create"
    changedFiles: list[str] = Field(default_factory=list)
    changedFileCount: int = 0
    modelChanges: list[str] = Field(default_factory=list)


ErrorCode = Literal[
    "INVALID_KEY",
    "UPSTREAM_LLM",
    "COST_CAP",
    "TIMEOUT",
    "INCOMPLETE",
    "INTERNAL",
    "BAD_REQUEST",
    "CANCELLED",
]


class ErrorEvent(BaseSseEvent):
    """Error signal. May or may not be terminal depending on ``code``.

    * ``INVALID_KEY``, ``UPSTREAM_LLM``, ``INTERNAL``, ``BAD_REQUEST``,
      ``CANCELLED`` — terminal.
    * ``COST_CAP``, ``TIMEOUT``, ``INCOMPLETE`` — warnings emitted *before* the
      ``done`` event when the run still produced usable output but did not
      finish cleanly (budget exceeded, or the customization loop was cut short
      by a provider rate-limit / turn cap). The client should treat them as
      non-terminal warnings and wait for ``done`` to get the download URL.

    ``INTERNAL`` errors are always sent with the literal string
    ``"Internal server error"``; tracebacks go to ``logger.exception``
    only, never to the client. The one other ``INTERNAL`` message is the fixed
    one a server restart adds to a run it interrupted, with ``resumeAvailable``.
    """

    event: Literal["error"] = "error"
    code: ErrorCode
    message: str
    reason: Optional[Literal["user", "abandoned"]] = None
    resumeAvailable: Optional[bool] = None


def format_sse(event: BaseSseEvent, secrets: Iterable[str] = ()) -> bytes:
    """Serialize an event to an SSE frame.

    The ``event:`` header line enables ``EventSource.addEventListener``
    dispatch in the browser; the JSON body (which also contains the
    ``event`` field) supports plain ``onmessage`` / fetch-reader style
    consumers. Frame is always terminated with the mandatory blank line.
    ``secrets`` are literal values (the run's own key) redacted as well.
    """
    # This is the last serialization boundary before data reaches the browser.
    # Redact recursively so exception text, model prose, phase details, and the
    # embedded recipe cannot echo credential-shaped values.
    safe_body, _findings = redact_data(event.model_dump(mode="json"), secrets)
    body = json.dumps(safe_body, ensure_ascii=False, separators=(",", ":"))
    frame = f"event: {event.event}\ndata: {body}\n\n"
    return frame.encode("utf-8")
