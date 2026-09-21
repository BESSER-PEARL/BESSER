"""Structured per-turn trace logging for smart generation.

A ``TraceWriter`` appends JSON-lines records to ``.besser_trace.jsonl``
inside the run's output directory. Each line is a self-contained record
with a timestamp, an event type, and a typed payload — so post-run
tooling (the ZIP inspector, a replay UI, a bug-report bundler) can
rehydrate exactly what happened without replaying the LLM.

Design constraints:

* **Append-only, crash-safe.** Each call does ``open("a", …)`` + flush so
  a process crash never loses events that were already emitted. We
  intentionally re-open the file on every write — cheap compared to the
  LLM round-trip, and avoids keeping a file handle open for the entire
  20-minute run.
* **Best-effort.** ``write()`` never raises: an I/O failure logs a
  warning and the run continues. Tracing is instrumentation, not a
  correctness concern.
* **Small surface.** One class, one method. We deliberately don't grow a
  DSL of event types here — callers pass an ``event`` string and a
  ``payload`` dict, and the writer stamps the rest (``ts``, ``run_id``,
  ``primary_kind``). Keeps the contract tight and easy to evolve.

The schema is documented alongside ``TRACE_FILENAME`` below and is the
public contract for anyone reading these files back.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

TRACE_FILENAME = ".besser_trace.jsonl"

# Canonical event names. Use the constants below rather than inlining
# strings so typos are caught at import time.
EVENT_RUN_START = "run_start"
EVENT_PHASE_ENTER = "phase_enter"
EVENT_PHASE_EXIT = "phase_exit"
EVENT_TURN_START = "turn_start"
EVENT_TOOL_CALL = "tool_call"
EVENT_COST_UPDATE = "cost_update"
EVENT_COMPACTION = "compaction"
EVENT_SNAPSHOT = "snapshot"
EVENT_ROLLBACK = "rollback"
EVENT_VALIDATION_ISSUE = "validation_issue"
EVENT_CHECKPOINT = "checkpoint"
EVENT_ERROR = "error"
EVENT_RUN_END = "run_end"


class TraceWriter:
    """Append structured JSONL records to a per-run trace file.

    Parameters
    ----------
    output_dir
        The run's output directory. The trace file lives at
        ``<output_dir>/.besser_trace.jsonl``; it's created on the first
        successful write, not on construction — so orchestrators that
        never get to Phase 2 leave no stray trace file behind.
    run_id
        Optional run identifier stamped onto every record. Defaults to
        the empty string for CLI / script users who don't have one.
    primary_kind
        Optional primary-model kind ("class", "state_machine", ...)
        stamped onto every record so downstream tools don't have to
        cross-reference another file to know what kind of run produced
        the trace.
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str = "",
        primary_kind: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.primary_kind = primary_kind
        self._path = os.path.join(output_dir, TRACE_FILENAME)

    @property
    def path(self) -> str:
        return self._path

    def write(self, event: str, **payload: Any) -> None:
        """Append a single trace record. Never raises.

        Extra keyword arguments become the ``payload`` object. The
        framing fields (``ts``, ``run_id``, ``primary_kind``, ``event``)
        are written unconditionally so every record is self-describing.
        """
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "primary_kind": self.primary_kind,
            "event": event,
            "payload": payload,
        }
        try:
            # Ensure the directory still exists — the orchestrator may
            # be midway through Phase 3 cleanup when an error is logged.
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str))
                fh.write("\n")
        except Exception as exc:
            # Instrumentation must not break the run.
            logger.debug("Failed to write trace record (%s): %s", event, exc)

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the last ``limit`` records from the trace file.

        Used by the ``DoneEvent`` recipe to surface recent activity to
        the client without streaming the whole trace. Returns an empty
        list if the trace file doesn't exist or can't be parsed.
        """
        if not os.path.isfile(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception:
            return []
        records: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Corrupt line (partial write on crash?) — skip it,
                # don't poison the whole tail.
                continue
        return records


class NullTraceWriter:
    """No-op trace writer for tests / environments where tracing is off.

    Exposes the same ``write`` / ``tail`` interface as :class:`TraceWriter`
    so callers don't need conditional branches.
    """

    path = ""

    def write(self, event: str, **payload: Any) -> None:  # noqa: D401
        return None

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        return []
