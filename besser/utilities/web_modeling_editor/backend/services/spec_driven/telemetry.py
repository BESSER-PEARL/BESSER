"""Pilot-experiment telemetry: append-only JSONL event store + report.

Captures hard usage data during facilitated pilot sessions of the
Spec-Driven editor — prompts, agent actions, generation-run summaries,
delivery clicks, and friction signals — so the research report can be
written from numbers instead of memory.

Collection is OFF by default and double-gated:

* ``BESSER_TELEMETRY_ENABLED`` (server master switch) must be "1"/"true".
* Every event must carry a non-empty participant label (P1, P2, ...).
  Regular users never send one, so nothing is collected outside pilot
  sessions even when the master switch is on.

Storage mirrors the incident log (see ``incidents.py``): one append-only
JSONL file per session under a host-mounted directory, thread-safe,
best-effort, and guaranteed never to raise into a caller — telemetry
must not be able to affect a run or a chat. Line schema::

    {"at": iso8601-utc, "session": ..., "participant": ..., "kind": ..., "payload": {...}}

The report builder (``build_report``) aggregates the store into a
per-participant summary plus an overall table, in Markdown or CSV.
"""

from __future__ import annotations

import csv
import datetime
import fnmatch
import glob
import io
import json
import logging
import os
import re
import threading
from collections import Counter
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Identity patterns — shared with the request-model sanitizers and the
# collector endpoint so every producer validates identically.
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PARTICIPANT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,16}$")

# Event kinds accepted over HTTP (``POST /besser_api/telemetry/event``).
EVENT_KINDS = frozenset({"prompt", "agent_action", "delivery", "friction"})
# Recorded in-process only (by the spec-driven runner), never over HTTP.
RUN_SUMMARY_KIND = "run_summary"

# Serialized-payload cap enforced by the HTTP collector (422 beyond it).
MAX_PAYLOAD_BYTES = 8 * 1024
# Absolute defensive cap for in-process events: beyond this the payload
# is replaced rather than written, so one runaway producer cannot bloat
# the store or stall the append under the lock.
_MAX_INTERNAL_PAYLOAD_BYTES = 64 * 1024


def telemetry_enabled() -> bool:
    """True when the server master switch is on.

    Read from the environment on every call (not cached at import) so a
    facilitator can flip the flag without a code change and tests can
    monkeypatch it.
    """
    value = os.environ.get("BESSER_TELEMETRY_ENABLED", "").strip().lower()
    return value in ("1", "true")


def _telemetry_dir() -> str:
    """The store directory, created on demand. ``""`` when unusable."""
    directory = os.environ.get("BESSER_TELEMETRY_DIR", "/app/telemetry")
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        return ""
    return directory


def sanitize_session(value: Any) -> Optional[str]:
    """Return *value* when it is a valid session id, else ``None``.

    Fail-open by design: telemetry identity fields ride along on real
    requests (e.g. a generation run) and must never be able to fail them.
    """
    if isinstance(value, str) and SESSION_PATTERN.fullmatch(value):
        return value
    return None


def sanitize_participant(value: Any) -> Optional[str]:
    """Return *value* when it is a valid participant label, else ``None``."""
    if isinstance(value, str) and PARTICIPANT_PATTERN.fullmatch(value):
        return value
    return None


def record_event(session: str, participant: str, kind: str, payload: Any) -> None:
    """Append one telemetry event to the session's JSONL file. Never raises.

    Writes only when the master switch is on AND ``participant`` is a
    non-empty valid label; anything else is silently dropped. ``payload``
    must be JSON-serializable (``default=str`` covers stragglers).
    """
    try:
        if not telemetry_enabled():
            return
        session = sanitize_session(session)
        participant = sanitize_participant(participant)
        if not session or not participant:
            return
        if kind not in EVENT_KINDS and kind != RUN_SUMMARY_KIND:
            return
        directory = _telemetry_dir()
        if not directory:
            return

        try:
            serialized_payload = json.dumps(
                payload if isinstance(payload, dict) else {"value": payload},
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            serialized_payload = json.dumps(
                {"note": "payload was not serializable and was dropped"}
            )
        if len(serialized_payload.encode("utf-8")) > _MAX_INTERNAL_PAYLOAD_BYTES:
            serialized_payload = json.dumps(
                {"note": "payload exceeded the size limit and was dropped"}
            )

        line = (
            '{"at": ' + json.dumps(
                datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                )
            )
            + ', "session": ' + json.dumps(session)
            + ', "participant": ' + json.dumps(participant)
            + ', "kind": ' + json.dumps(kind)
            + ', "payload": ' + serialized_payload
            + "}"
        )
        path = os.path.join(directory, f"{session}.jsonl")
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # pragma: no cover — must never affect a caller
        logger.debug("Could not record telemetry event", exc_info=True)


# ---------------------------------------------------------------------
# Deterministic/LLM file split
# ---------------------------------------------------------------------


def llm_touched_paths(tool_calls_log: Optional[Iterable[Any]]) -> set[str]:
    """Workspace-relative paths the LLM successfully wrote or edited.

    Source of truth: the orchestrator's ``tool_calls_log`` — every
    ``write_file`` / ``modify_file`` call is appended there with its input
    and a success flag (guardrail rejections carry ``success=False`` and
    are excluded, since they changed nothing on disk). Paths are
    normalized to forward slashes to match the executor's
    ``_generator_files`` tags. Never raises; returns what it could parse.
    """
    touched: set[str] = set()
    try:
        for call in tool_calls_log or []:
            if not isinstance(call, dict):
                continue
            if call.get("tool") not in ("write_file", "modify_file"):
                continue
            if call.get("success") is False:
                continue
            path = (call.get("input") or {}).get("path")
            if isinstance(path, str) and path.strip():
                touched.add(path.replace("\\", "/"))
    except Exception:
        logger.debug("Could not extract LLM-touched paths", exc_info=True)
    return touched


def compute_file_split(
    output_dir: str,
    generator_files: Optional[Iterable[str]],
    llm_touched: Optional[Iterable[str]],
    excluded_patterns: Optional[Iterable[str]] = None,
) -> dict:
    """Three-way authorship split over a run's final output tree.

    Counts every user file (build/dependency directories pruned,
    ``.besser_*`` internals and runtime-artifact globs like ``*.zip`` /
    ``*.db`` skipped — the same exclusion set the packaging walk uses):

    * ``generator_untouched`` — written by the deterministic Phase-1
      generator (tagged in ``executor._generator_files``) and never
      successfully edited by the LLM this run.
    * ``generator_llm_modified`` — generator-authored, then edited by the
      LLM (a successful ``write_file`` / ``modify_file`` in
      ``tool_calls_log``).
    * ``llm_authored`` — everything else in the final tree.

    Returns counts, ``total``, and percentages (one decimal). Never
    raises; on failure returns whatever was counted so far.
    """
    split = {
        "generator_untouched": 0,
        "generator_llm_modified": 0,
        "llm_authored": 0,
        "total": 0,
    }
    try:
        generator_set = {str(p) for p in (generator_files or ())}
        touched_set = {str(p) for p in (llm_touched or ())}
        excluded = set(excluded_patterns or ())
        file_globs = [p for p in excluded if "*" in p or "?" in p]

        for root, dirs, files in os.walk(output_dir):
            dirs[:] = [d for d in dirs if d not in excluded]
            for name in files:
                if name.startswith(".besser_"):
                    continue
                if any(fnmatch.fnmatch(name, pattern) for pattern in file_globs):
                    continue
                rel = os.path.relpath(
                    os.path.join(root, name), output_dir
                ).replace("\\", "/")
                split["total"] += 1
                if rel in generator_set:
                    if rel in touched_set:
                        split["generator_llm_modified"] += 1
                    else:
                        split["generator_untouched"] += 1
                else:
                    split["llm_authored"] += 1
    except Exception:
        logger.debug("File split computation failed", exc_info=True)

    total = split["total"]
    for key in ("generator_untouched", "generator_llm_modified", "llm_authored"):
        split[f"{key}_pct"] = round(100.0 * split[key] / total, 1) if total else 0.0
    return split


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def read_events(participant: Optional[str] = None) -> list[dict]:
    """All stored events (optionally filtered by participant), oldest first
    within each session file. Malformed lines are skipped."""
    events: list[dict] = []
    directory = os.environ.get("BESSER_TELEMETRY_DIR", "/app/telemetry")
    if not os.path.isdir(directory):
        return events
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if participant and event.get("participant") != participant:
                        continue
                    events.append(event)
        except OSError:
            logger.debug("Could not read telemetry file %s", path, exc_info=True)
    return events


class _ParticipantStats:
    """Aggregation bucket for one participant."""

    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.prompts = 0
        self.agent_actions = 0
        self.friction = 0
        self.delivery: Counter = Counter()
        self.runs: list[dict] = []

    # -- run-derived metrics ------------------------------------------

    @property
    def outcome_counts(self) -> Counter:
        counts: Counter = Counter()
        for run in self.runs:
            outcome = str(run.get("outcome") or "unknown")
            if outcome.startswith("failed"):
                counts["failed"] += 1
            elif outcome in ("success", "incomplete"):
                counts[outcome] += 1
            else:
                counts["unknown"] += 1
        return counts

    @property
    def avg_duration(self) -> float:
        durations = [
            float(run["duration_seconds"])
            for run in self.runs
            if isinstance(run.get("duration_seconds"), (int, float))
        ]
        return sum(durations) / len(durations) if durations else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(
            int(run["tokens"]) for run in self.runs
            if isinstance(run.get("tokens"), (int, float))
        )

    @property
    def total_turns(self) -> int:
        return sum(
            int(run["turns"]) for run in self.runs
            if isinstance(run.get("turns"), (int, float))
        )

    @property
    def blockers_found(self) -> int:
        return sum(
            int(run["blockers_found"]) for run in self.runs
            if isinstance(run.get("blockers_found"), (int, float))
        )

    @property
    def blockers_remaining(self) -> int:
        return sum(
            int(run["blockers_remaining"]) for run in self.runs
            if isinstance(run.get("blockers_remaining"), (int, float))
        )

    @property
    def file_split(self) -> dict:
        totals = {
            "generator_untouched": 0,
            "generator_llm_modified": 0,
            "llm_authored": 0,
            "total": 0,
        }
        for run in self.runs:
            split = run.get("file_split")
            if not isinstance(split, dict):
                continue
            for key in totals:
                value = split.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
        total = totals["total"]
        for key in ("generator_untouched", "generator_llm_modified", "llm_authored"):
            totals[f"{key}_pct"] = (
                round(100.0 * totals[key] / total, 1) if total else 0.0
            )
        return totals

    def absorb(self, event: dict) -> None:
        session = event.get("session")
        if isinstance(session, str):
            self.sessions.add(session)
        kind = event.get("kind")
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        if kind == "prompt":
            self.prompts += 1
        elif kind == "agent_action":
            self.agent_actions += 1
        elif kind == "friction":
            self.friction += 1
        elif kind == "delivery":
            action = payload.get("action")
            self.delivery[str(action) if action else "unspecified"] += 1
        elif kind == RUN_SUMMARY_KIND:
            self.runs.append(payload)


def _aggregate(events: Iterable[dict]) -> dict[str, _ParticipantStats]:
    stats: dict[str, _ParticipantStats] = {}
    for event in events:
        participant = event.get("participant")
        if not isinstance(participant, str) or not participant:
            continue
        stats.setdefault(participant, _ParticipantStats()).absorb(event)
    return dict(sorted(stats.items()))


def _split_line(split: dict) -> str:
    return (
        f"{split['generator_untouched']} generator-untouched "
        f"({split['generator_untouched_pct']}%) / "
        f"{split['generator_llm_modified']} generator-LLM-modified "
        f"({split['generator_llm_modified_pct']}%) / "
        f"{split['llm_authored']} LLM-authored "
        f"({split['llm_authored_pct']}%)"
    )


def _render_markdown(stats: dict[str, _ParticipantStats]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = [
        "# BESSER Pilot Telemetry Report",
        "",
        f"Generated: {now}",
        f"Participants on record: {len(stats)}",
        "",
    ]
    if not stats:
        lines.append("No telemetry events have been recorded.")
        return "\n".join(lines) + "\n"

    for participant, s in stats.items():
        outcomes = s.outcome_counts
        split = s.file_split
        lines += [
            f"## Participant {participant}",
            "",
            f"- Sessions: {len(s.sessions)}",
            f"- Prompts: {s.prompts}",
            f"- Agent actions: {s.agent_actions}",
            f"- Generation runs: {len(s.runs)}",
            (
                f"- Run outcomes: {outcomes.get('success', 0)} success / "
                f"{outcomes.get('incomplete', 0)} incomplete / "
                f"{outcomes.get('failed', 0)} failed"
            ),
            f"- Average run duration: {round(s.avg_duration, 1)} s",
            f"- Total LLM turns: {s.total_turns}",
            f"- Total tokens: {s.total_tokens:,}",
            (
                f"- Validation blockers: {s.blockers_found} found / "
                f"{s.blockers_remaining} remaining"
            ),
            f"- File split ({split['total']} files): {_split_line(split)}",
        ]
        if s.delivery:
            delivery = ", ".join(
                f"{action} {count}" for action, count in sorted(s.delivery.items())
            )
            lines.append(f"- Delivery actions: {delivery}")
        lines.append(f"- Friction events: {s.friction}")
        lines.append("")

    # Overall aggregate table.
    lines += [
        "## Overall",
        "",
        "| Participant | Sessions | Prompts | Runs | Success | Incomplete | "
        "Failed | Avg duration (s) | Tokens | Gen untouched % | "
        "Gen LLM-modified % | LLM-authored % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for participant, s in stats.items():
        outcomes = s.outcome_counts
        split = s.file_split
        lines.append(
            f"| {participant} | {len(s.sessions)} | {s.prompts} | {len(s.runs)} "
            f"| {outcomes.get('success', 0)} | {outcomes.get('incomplete', 0)} "
            f"| {outcomes.get('failed', 0)} | {round(s.avg_duration, 1)} "
            f"| {s.total_tokens:,} | {split['generator_untouched_pct']} "
            f"| {split['generator_llm_modified_pct']} "
            f"| {split['llm_authored_pct']} |"
        )
    lines.append("")
    return "\n".join(lines)


_CSV_COLUMNS = [
    "participant", "sessions", "prompts", "agent_actions", "runs",
    "success", "incomplete", "failed", "avg_duration_seconds",
    "total_turns", "total_tokens", "blockers_found", "blockers_remaining",
    "files_total", "generator_untouched", "generator_llm_modified",
    "llm_authored", "generator_untouched_pct", "generator_llm_modified_pct",
    "llm_authored_pct", "delivery_events", "friction_events",
]


def _render_csv(stats: dict[str, _ParticipantStats]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for participant, s in stats.items():
        outcomes = s.outcome_counts
        split = s.file_split
        writer.writerow([
            participant, len(s.sessions), s.prompts, s.agent_actions,
            len(s.runs), outcomes.get("success", 0),
            outcomes.get("incomplete", 0), outcomes.get("failed", 0),
            round(s.avg_duration, 1), s.total_turns, s.total_tokens,
            s.blockers_found, s.blockers_remaining, split["total"],
            split["generator_untouched"], split["generator_llm_modified"],
            split["llm_authored"], split["generator_untouched_pct"],
            split["generator_llm_modified_pct"], split["llm_authored_pct"],
            sum(s.delivery.values()), s.friction,
        ])
    return buffer.getvalue()


def build_report(participant: Optional[str] = None, fmt: str = "md") -> str:
    """Aggregate the store into the pilot report (Markdown or CSV)."""
    stats = _aggregate(read_events(participant=participant))
    if fmt == "csv":
        return _render_csv(stats)
    return _render_markdown(stats)
