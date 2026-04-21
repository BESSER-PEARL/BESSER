"""SSE event schema for the smart generation endpoint.

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

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class BaseSseEvent(BaseModel):
    """Common parent for all SSE events."""

    event: str


class StartEvent(BaseSseEvent):
    """Emitted once at the start of a run, before any BUML assembly."""

    event: Literal["start"] = "start"
    runId: str
    provider: Literal["anthropic", "openai"]
    llmModel: str
    maxCost: float
    maxRuntime: int


PhaseName = Literal["select", "generate", "gap", "customize", "validate"]


class PhaseEvent(BaseSseEvent):
    """Coarse-grained phase marker emitted as the pipeline advances."""

    event: Literal["phase"] = "phase"
    phase: PhaseName
    message: str


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


class CostEvent(BaseSseEvent):
    """Periodic cost / runtime tick emitted by the cost emitter task."""

    event: Literal["cost"] = "cost"
    usd: float
    turns: int
    elapsedSeconds: float


class DoneEvent(BaseSseEvent):
    """Terminal event on successful completion.

    ``downloadUrl`` points to the sibling ``GET /download-smart/{runId}``
    endpoint which single-use-serves the generated ZIP or file. The
    ``recipe`` field carries the contents of ``.besser_recipe.json``
    that ``LLMOrchestrator`` writes at the end of every run.
    """

    event: Literal["done"] = "done"
    downloadUrl: str
    fileName: str
    isZip: bool
    recipe: dict[str, Any] = Field(default_factory=dict)


ErrorCode = Literal[
    "INVALID_KEY",
    "UPSTREAM_LLM",
    "COST_CAP",
    "TIMEOUT",
    "INTERNAL",
    "BAD_REQUEST",
    "CANCELLED",
]


class ErrorEvent(BaseSseEvent):
    """Error signal. May or may not be terminal depending on ``code``.

    * ``INVALID_KEY``, ``UPSTREAM_LLM``, ``INTERNAL``, ``BAD_REQUEST``,
      ``CANCELLED`` — terminal.
    * ``COST_CAP``, ``TIMEOUT`` — warnings emitted *before* the ``done`` event
      when the run exceeded its budget but still produced usable output.
      The client should treat them as non-terminal warnings and wait for
      ``done`` to get the download URL.

    ``INTERNAL`` errors are always sent with the literal string
    ``"Internal server error"``; tracebacks go to ``logger.exception``
    only, never to the client.
    """

    event: Literal["error"] = "error"
    code: ErrorCode
    message: str


def format_sse(event: BaseSseEvent) -> bytes:
    """Serialize an event to an SSE frame.

    The ``event:`` header line enables ``EventSource.addEventListener``
    dispatch in the browser; the JSON body (which also contains the
    ``event`` field) supports plain ``onmessage`` / fetch-reader style
    consumers. Frame is always terminated with the mandatory blank line.
    """
    body = event.model_dump_json()
    frame = f"event: {event.event}\ndata: {body}\n\n"
    return frame.encode("utf-8")
