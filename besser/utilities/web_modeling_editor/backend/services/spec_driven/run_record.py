"""What a run did, for the ``run`` section of its recipe and its two log lines.

A shared output folder (a download, a pushed repository) used to say nothing
about the run behind it: whether it edited a previous result or started over
because that result had expired, which key tier and model ran it, which limit
stopped it, what the user was warned about. The runner fills a ``RunFacts`` as
the run goes; ``run_section`` turns it, the recipe and the orchestrator into a
small dict. The recipe goes into GitHub pushes, so the section holds the public
model label only, no key, no path and no host, and its texts are redacted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from besser.spec_driven_agent.state import agent_trace
from besser.utilities.provenance import get_besser_version, get_build_sha
from besser.utilities.secret_redaction import redact_text

# ``mode_reason`` values.
FRESH = "fresh project"
EDIT = "edit of a previous result"
BASE_EXPIRED = "previous result expired, rebuilt from scratch"
BASE_NOT_COPIED = "previous result could not be copied, rebuilt from scratch"
RESUMED = "resumed"

# ``stopped_by`` values, and the ``outcome`` of the end log line.
FINISHED, TIME_LIMIT, COST_LIMIT, TURN_LIMIT = "finished", "time limit", "cost limit", "turn limit"
PROVIDER_ERROR, CANCELLED = "provider error", "cancelled"
_STOP_REASONS = {"max_turns": TURN_LIMIT, "api_error": PROVIDER_ERROR, "cancelled": CANCELLED,
                 "cost_cap": COST_LIMIT, "timeout": TIME_LIMIT}

# The section stays far under 4 KB: a few warnings, each cut.
_MAX_WARNINGS = 8
_MAX_TEXT = 300
_PHASES = (("select", "select"), ("generate", "generate"), ("gap", "plan"), ("customize", "customise"),
           ("validate", "validate"))
_WRITE_TOOLS = frozenset({"write_file"})
_EDIT_TOOLS = frozenset({"modify_file", "replace_file_lines"})


def key_tier(provider: str) -> str:
    """Who pays for the model: the server's keyless tiers, or the user's own key."""
    return provider if provider in ("free", "sponsored") else "own key"


@dataclass
class RunFacts:
    """What only the runner knows about a run, filled in as it goes."""

    mode: str = "generate"
    mode_reason: str = FRESH
    base_run_id: Optional[str] = None
    started: float = field(default_factory=time.monotonic)
    phase_started: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    stopped_by: Optional[str] = None
    stop_message: Optional[str] = None
    outcome: Optional[str] = None
    blocker_count: int = 0
    estimated_cost_usd: float = 0.0
    archive_written: bool = False

    def saw_phase(self, phase: str) -> None:
        self.phase_started.setdefault(phase, time.monotonic())

    def saw_warning(self, message: str) -> None:
        if len(self.warnings) < _MAX_WARNINGS and message not in self.warnings:
            self.warnings.append(message)

    def phase_seconds(self, now: float) -> dict:
        """Each phase's seconds, from its first event to the next phase's."""
        starts = sorted((at, name) for name, at in self.phase_started.items())
        ends = [at for at, _ in starts[1:]] + [now]
        seconds = {name: round(end - at, 1) for (at, name), end in zip(starts, ends)}
        return {label: seconds[name] for name, label in _PHASES if name in seconds}

    def stop(self, *, stop_reason: str, cancelled: bool, cap_breach: Optional[str], timed_out: bool,
             message: Optional[str], incomplete: bool, blocker_count: int, cost: float) -> None:
        """Record why a run that produced output ended, in the words the run card used."""
        if cancelled or stop_reason == "cancelled":
            self.stopped_by = CANCELLED
        elif cap_breach:
            self.stopped_by = TIME_LIMIT if timed_out else COST_LIMIT
        else:
            self.stopped_by = _STOP_REASONS.get(stop_reason, FINISHED)
        self.stop_message = message
        self.blocker_count = blocker_count
        self.estimated_cost_usd = round(cost, 4)
        if self.stopped_by == CANCELLED:
            self.outcome = "cancelled"
        elif self.stopped_by in (TIME_LIMIT, COST_LIMIT, TURN_LIMIT):
            self.outcome = "budget_stop"
        elif self.stopped_by == PROVIDER_ERROR:
            self.outcome = "failed"
        else:
            self.outcome = "completed_with_blockers" if incomplete or blocker_count else "completed"


def _files(recipe: dict) -> tuple[int, int]:
    written, edited = set(), set()
    for call in recipe.get("tool_calls") or []:
        path = (call.get("input") or {}).get("path") if isinstance(call, dict) else None
        if isinstance(path, str):
            if call.get("tool") in _WRITE_TOOLS:
                written.add(path)
            elif call.get("tool") in _EDIT_TOOLS:
                edited.add(path)
    return len(written), len(edited - written)


def run_section(facts: RunFacts, recipe: dict, orchestrator: Any, *, provider: str, models: list,
                request_chars: int, secrets: tuple = ()) -> dict:
    """The recipe's ``run`` section. ``models`` are the ids that ran; only their public labels are kept."""
    labels = list(dict.fromkeys(label for label in (agent_trace.normalize_model_id(provider, m) for m in models)
                                if label))
    private = [m for m in models if m and m not in labels]

    def public(text: Optional[str]) -> Optional[str]:
        """A user-facing text as it may go into a repository: no key, no provider-side model id."""
        if not text:
            return None
        for model in private:
            text = text.replace(model, labels[0] if labels else provider)
        cleaned, _ = redact_text(" ".join(text.split())[:_MAX_TEXT], env_style=False, secrets=secrets)
        return cleaned

    target = recipe.get("target") if isinstance(recipe.get("target"), dict) else {}
    if getattr(orchestrator, "target_generator_bound", False):
        decided_by = "approved plan"
    elif getattr(orchestrator, "_routed", False):
        decided_by = "selector"
    else:
        decided_by = "not decided"
    asks_for_frontend = getattr(orchestrator, "_asks_for_frontend", None)
    written, edited = _files(recipe)
    section = {
        "mode": facts.mode,
        "mode_reason": facts.mode_reason,
        "base_run_id": facts.base_run_id,
        "target": target.get("kind"),
        "target_decided_by": decided_by,
        "frontend_requested": bool(asks_for_frontend()) if callable(asks_for_frontend) else None,
        "generator": recipe.get("generator_used"),
        "key_tier": key_tier(provider),
        "models": labels,
        "limits": {
            "time_seconds": getattr(orchestrator, "max_runtime_seconds", None),
            "cost_usd": getattr(orchestrator, "max_cost_usd", None),
            "turns": getattr(orchestrator, "max_turns", None),
        },
        "stopped_by": facts.stopped_by,
        "stop_message": public(facts.stop_message),
        "outcome": facts.outcome,
        "warnings": [public(message) for message in facts.warnings],
        "duration_seconds": round(time.monotonic() - facts.started, 1),
        "phase_seconds": facts.phase_seconds(time.monotonic()),
        "estimated_cost_usd": facts.estimated_cost_usd,
        "counts": {
            "turns": recipe.get("turns"),
            "tool_calls": recipe.get("tool_calls_count"),
            "files_written": written,
            "files_edited": edited,
            "blockers": facts.blocker_count,
        },
        "besser_version": get_besser_version(),
        "besser_build": get_build_sha(),
        "request_chars": request_chars,
    }
    return section


def start_line(run_id: str, facts: RunFacts, *, provider: str, model: Optional[str], generator: Optional[str]) -> str:
    """The log line of a run's start: ``key=value``, no request text, no key."""
    return (f"spec-driven run start run_id={run_id} mode={facts.mode} base_run_id={facts.base_run_id or '-'} "
            f"key_tier={key_tier(provider).replace(' ', '_')} "
            f"model={agent_trace.normalize_model_id(provider, model) or provider} generator={generator or 'auto'}")


def end_line(run_id: str, facts: RunFacts) -> str:
    """The log line of a run's end, in the ``run`` section's names."""
    return (f"spec-driven run end run_id={run_id} outcome={facts.outcome or 'failed'} "
            f"duration_seconds={time.monotonic() - facts.started:.1f} blockers={facts.blocker_count} "
            f"estimated_cost_usd={facts.estimated_cost_usd:.4f} archive_written={str(facts.archive_written).lower()}")
