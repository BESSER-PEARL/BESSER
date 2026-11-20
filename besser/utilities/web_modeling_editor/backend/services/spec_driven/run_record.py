"""What a run did, for the ``run`` section of its recipe and its two log lines.

A shared output folder (a download, a pushed repository) used to say nothing
about the run behind it: whether it edited a previous result or started over
because that result had expired, which key tier and model ran it, which limit
stopped it, what the user was warned about. The runner fills a ``RunFacts`` as
the run goes; ``run_section`` turns it and the run's own orchestrator and client
into a small dict. The recipe goes into GitHub pushes, so the section holds
fixed categories and fixed sentences only, never an error's text, and the
public model label only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from besser.spec_driven_agent.state import agent_trace
from besser.utilities.provenance import get_besser_version, get_build_sha

# The two lines per run. A logger of its own: the backend installs no handler for
# the package's loggers, so an INFO record on them is written nowhere.
run_log = logging.getLogger("besser.spec_driven.runs")


def enable_run_log() -> None:
    """Write the run log to stderr, in the server's line format. Called at backend startup."""
    if not any(getattr(handler, "is_run_log", False) for handler in run_log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        handler.is_run_log = True
        run_log.addHandler(handler)
    run_log.setLevel(logging.INFO)

# ``mode_reason`` values.
FRESH = "fresh project"
EDIT = "edit of a previous result"
BASE_EXPIRED = "previous result expired, rebuilt from scratch"
BASE_NOT_COPIED = "previous result could not be copied, rebuilt from scratch"
RESUMED = "resumed"

# ``stopped_by`` values.
FINISHED, TIME_LIMIT, COST_LIMIT, TURN_LIMIT = "finished", "time limit", "cost limit", "turn limit"
PROVIDER_ERROR, INTERNAL_ERROR, CANCELLED = "provider error", "internal error", "cancelled"
STUCK, HANDED_OFF = "repeated edits did not land", "handed off to validation"
_STOP_REASONS = {"max_turns": TURN_LIMIT, "api_error": PROVIDER_ERROR, "cancelled": CANCELLED,
                 "cost_cap": COST_LIMIT, "timeout": TIME_LIMIT, "stuck_edit_loop": STUCK,
                 "validation_required": HANDED_OFF}

# ``stop_message``: the run card's sentence for each way a run ends, without its variable parts.
_STOP_MESSAGES = {
    TIME_LIMIT: "The run reached its time limit and was stopped before it finished.",
    COST_LIMIT: "The run reached its cost limit and was stopped before it finished.",
    TURN_LIMIT: "The customization loop reached its step limit before finishing every requested change.",
    PROVIDER_ERROR: "The model provider failed, so the output may be incomplete.",
    INTERNAL_ERROR: "The run hit an internal error after generating files, so the output may be incomplete.",
    CANCELLED: "The run was cancelled before it finished.",
    STUCK: "The customization loop stopped because repeated edits to the same file were not landing.",
    HANDED_OFF: "The customization loop handed off to validation before finishing every requested change.",
}
_UNRESOLVED = ("The app was generated, but unresolved implementation or verification issues remain; "
               "it is not verified complete.")

# ``warnings``: a warning the user was shown, by how its fixed text starts. A
# warning not listed here is left out: its text may quote an error.
_WARNINGS = (
    ("The previous generation has expired", "The previous result had expired, so the app was rebuilt from scratch."),
    ("Could not copy the previous generation",
     "The previous result could not be copied, so the app was rebuilt from scratch."),
    ("Cost cap reached", _STOP_MESSAGES[COST_LIMIT]),
    ("Free tier per-run usage limit reached", "The run reached the free tier's per-run usage limit."),
    ("Runtime cap reached", _STOP_MESSAGES[TIME_LIMIT]),
    ("The app was generated, but", _UNRESOLVED),
    ("I changed the app, but could not confirm the reported failure",
     "The reported failure could not be confirmed as fixed."),
    ("The model provider failed after generating files", _STOP_MESSAGES[PROVIDER_ERROR]),
    ("The run hit an internal error after generating files", _STOP_MESSAGES[INTERNAL_ERROR]),
    ("The customization loop was cut short by a provider error", _STOP_MESSAGES[PROVIDER_ERROR]),
    ("The customization loop reached its step limit", _STOP_MESSAGES[TURN_LIMIT]),
    ("The customization loop stopped because repeated edits", _STOP_MESSAGES[STUCK]),
    ("The customization loop handed off to validation", _STOP_MESSAGES[HANDED_OFF]),
    ("The run was cancelled before", _STOP_MESSAGES[CANCELLED]),
)
_ASSEMBLY_WARNING = "project diagram(s) could not be converted"

# The section stays far under 4 KB.
_MAX_WARNINGS = 8
_PHASES = (("select", "select"), ("generate", "generate"), ("gap", "plan"), ("customize", "customise"),
           ("validate", "validate"))
_WRITE_TOOLS = frozenset({"write_file"})
_EDIT_TOOLS = frozenset({"modify_file", "replace_file_lines"})


def key_tier(provider: str) -> str:
    """Who pays for the model: the server's keyless tiers, or the user's own key."""
    return provider if provider in ("free", "sponsored") else "own key"


def _fixed_warning(message: str) -> Optional[str]:
    if _ASSEMBLY_WARNING in message:
        return "A diagram of the project could not be converted, so the result is not verified complete."
    return next((sentence for start, sentence in _WARNINGS if message.startswith(start)), None)


@dataclass
class RunFacts:
    """What only the runner knows about a run, filled in as it goes."""

    mode: str = "generate"
    mode_reason: str = FRESH
    base_run_id: Optional[str] = None
    started: float = field(default_factory=time.monotonic)
    phase_started: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    #: ``PROVIDER_ERROR`` or ``INTERNAL_ERROR`` when the worker failed after writing files.
    late_error: Optional[str] = None
    stopped_by: Optional[str] = None
    stop_message: Optional[str] = None
    outcome: Optional[str] = None
    blocker_count: int = 0
    #: Findings that are not confirmed (a check that reached no conclusion); never blockers.
    not_confirmed: int = 0
    estimated_cost_usd: float = 0.0
    archive_written: bool = False
    #: Lines of the delivered files per author, from the Agent Trace record; None without one.
    lines: Optional[dict] = None

    def saw_phase(self, phase: str) -> None:
        self.phase_started.setdefault(phase, time.monotonic())

    def saw_warning(self, message: str) -> None:
        sentence = _fixed_warning(message)
        if sentence and sentence not in self.warnings and len(self.warnings) < _MAX_WARNINGS:
            self.warnings.append(sentence)

    def phase_seconds(self, now: float) -> dict:
        """Each phase's seconds, from its start to the next phase's."""
        starts = sorted((at, name) for name, at in self.phase_started.items())
        ends = [at for at, _ in starts[1:]] + [now]
        seconds = {name: round(end - at, 1) for (at, name), end in zip(starts, ends)}
        return {label: seconds[name] for name, label in _PHASES if name in seconds}

    def stop(self, *, stop_reason: str, cancelled: bool, cap_breach: bool, timed_out: bool,
             incomplete: bool, blocker_count: int, cost: float, not_confirmed: int = 0) -> None:
        """Record why a run that produced output ended, in the run card's fixed words."""
        if self.late_error:
            self.stopped_by = self.late_error
        elif cancelled or stop_reason == "cancelled":
            self.stopped_by = CANCELLED
        elif cap_breach:
            self.stopped_by = TIME_LIMIT if timed_out else COST_LIMIT
        else:
            self.stopped_by = _STOP_REASONS.get(stop_reason, FINISHED)
        self.stop_message = _STOP_MESSAGES.get(self.stopped_by) or (_UNRESOLVED if incomplete else None) or (
            f"The app was generated; {not_confirmed} finding(s) could not be confirmed and are listed "
            "for review." if not_confirmed else None)
        self.blocker_count = blocker_count
        self.not_confirmed = not_confirmed
        self.estimated_cost_usd = round(cost, 4)
        if self.stopped_by == CANCELLED:
            self.outcome = "cancelled"
        elif self.stopped_by in (TIME_LIMIT, COST_LIMIT, TURN_LIMIT):
            self.outcome = "budget_stop"
        elif self.stopped_by in (PROVIDER_ERROR, INTERNAL_ERROR):
            self.outcome = "failed"
        else:
            self.outcome = "completed_with_blockers" if incomplete or blocker_count else "completed"


def _files(tool_calls: list) -> tuple[int, int]:
    written, edited = set(), set()
    for call in tool_calls:
        path = (call.get("input") or {}).get("path") if isinstance(call, dict) else None
        if isinstance(path, str):
            if call.get("tool") in _WRITE_TOOLS:
                written.add(path)
            elif call.get("tool") in _EDIT_TOOLS:
                edited.add(path)
    return len(written), len(edited - written)


def run_section(facts: RunFacts, orchestrator: Any, *, provider: str, models: list, request_chars: int) -> dict:
    """The recipe's ``run`` section, from this run's own orchestrator: never from a recipe on disk,
    which is the base run's when this one failed before writing its own.

    ``models`` are the ids that ran; only their public labels are kept.
    """
    labels = list(dict.fromkeys(label for label in (agent_trace.normalize_model_id(provider, m) for m in models)
                                if label))
    target = getattr(orchestrator, "target", None)
    if getattr(orchestrator, "target_generator_bound", False):
        decided_by = "approved plan"
    elif getattr(orchestrator, "_routed", False):
        decided_by = "selector"
    else:
        decided_by = "not decided"
    asks_for_frontend = getattr(orchestrator, "_asks_for_frontend", None)
    tool_calls = getattr(orchestrator, "tool_calls_log", None) or []
    written, edited = _files(tool_calls)
    validate_started = getattr(orchestrator, "phase3_started_at", None)
    if validate_started is not None:
        # The validate phase event arrives once Phase 3 has run its checks; it started earlier.
        facts.phase_started["validate"] = min(validate_started, facts.phase_started.get("validate", validate_started))
    return {
        "mode": facts.mode,
        "mode_reason": facts.mode_reason,
        "base_run_id": facts.base_run_id,
        "target": getattr(target, "kind", None),
        "target_decided_by": decided_by,
        "frontend_requested": bool(asks_for_frontend()) if callable(asks_for_frontend) else None,
        "generator": getattr(orchestrator, "_generator_used", None),
        "key_tier": key_tier(provider),
        "models": labels,
        "limits": {
            "time_seconds": getattr(orchestrator, "max_runtime_seconds", None),
            "cost_usd": getattr(orchestrator, "max_cost_usd", None),
            "turns": getattr(orchestrator, "max_turns", None),
        },
        "stopped_by": facts.stopped_by,
        "stop_message": facts.stop_message,
        "outcome": facts.outcome,
        "warnings": list(facts.warnings),
        "duration_seconds": round(time.monotonic() - facts.started, 1),
        "phase_seconds": facts.phase_seconds(time.monotonic()),
        "estimated_cost_usd": facts.estimated_cost_usd,
        "counts": {
            "turns": getattr(orchestrator, "total_turns", None),
            "tool_calls": len(tool_calls),
            "files_written": written,
            "files_edited": edited,
            "blockers": facts.blocker_count,
            "not_confirmed": facts.not_confirmed,
        },
        "lines": facts.lines,
        "besser_version": get_besser_version(),
        "besser_build": get_build_sha(),
        "request_chars": request_chars,
    }


def start_line(run_id: str, facts: RunFacts, *, provider: str, model: Optional[str], generator: Optional[str]) -> str:
    """The log line of a run's start: ``key=value``, no request text, no key."""
    return (f"spec-driven run start run_id={run_id} mode={facts.mode} base_run_id={facts.base_run_id or '-'} "
            f"key_tier={key_tier(provider).replace(' ', '_')} "
            f"model={agent_trace.normalize_model_id(provider, model) or provider} generator={generator or 'auto'}")


# The recipe's ``usage`` names.
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens")
_LINE_FIELDS = ("total", "generator", "ai", "human", "unknown", "ai_this_run")


def end_line(run_id: str, facts: RunFacts, usage: Any = None) -> str:
    """The log line of a run's end, in the recipe's names; ``-`` for a value the run does not have."""
    def count(value: Any) -> Any:
        return value if isinstance(value, int) else "-"

    tokens = " ".join(f"{name}={count(getattr(usage, name, None))}" for name in _TOKEN_FIELDS)
    lines = " ".join(f"lines_{name}={(facts.lines or {}).get(name, '-')}" for name in _LINE_FIELDS)
    return (f"spec-driven run end run_id={run_id} outcome={facts.outcome or 'failed'} "
            f"duration_seconds={time.monotonic() - facts.started:.1f} blockers={facts.blocker_count} "
            f"not_confirmed={facts.not_confirmed} "
            f"estimated_cost_usd={facts.estimated_cost_usd:.4f} {tokens} {lines} "
            f"archive_written={str(facts.archive_written).lower()}")
