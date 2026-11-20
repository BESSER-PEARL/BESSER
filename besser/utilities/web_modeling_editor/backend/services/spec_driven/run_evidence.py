"""A run's full record, kept server-side apart from the recipe the user downloads.

The recipe ships in the app archive, so it is cut to a size budget, and the
run's trace lives in its workspace, which is deleted a few hours after the run.
:class:`RunEvidence` receives every trace record of a run (``TraceWriter``'s
mirror) and appends it, redacted, to ``<incident log dir>/runs/<run id>.jsonl``
as it happens, so a cancelled or crashed run keeps what it did. The run's
shell never sees that folder (``execution/sandbox.py`` masks the incident log
directory). Files older than ``BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS`` are
removed by the hourly cleanup.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Iterable

from besser.spec_driven_agent.state.tracing import EVENT_PHASE_ENTER, EVENT_PHASE_EXIT, EVENT_TOOL_CALL
from besser.utilities.secret_redaction import redact_text, redact_trace_text
from besser.utilities.web_modeling_editor.backend.services.spec_driven.incidents import incident_log_dir
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

logger = logging.getLogger(__name__)

EVIDENCE_DIRNAME = "runs"
# Any one string of a record, a written file's content included: first and last halves kept.
VALUE_CHARS = 64 * 1024
# A tool result, as the model read it: its first and last 8 KiB.
RESULT_CHARS = 16 * 1024

_LOCK = threading.Lock()


def evidence_dir() -> str:
    return os.path.join(incident_log_dir(), EVIDENCE_DIRNAME)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n<<cut {len(text) - 2 * half} of {len(text)} chars>>\n{text[-half:]}"


def _safe(value: Any, secrets: tuple[str, ...]) -> Any:
    """``value`` with every string redacted as the run's events and trace are, then capped."""
    if isinstance(value, str):
        return _cut(redact_trace_text(redact_text(value, secrets=secrets)[0]), VALUE_CHARS)
    if isinstance(value, (list, tuple)):
        return [_safe(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _safe(item, secrets) for key, item in value.items()}
    return value


class RunEvidence:
    """Appends a run's trace records to its server-side file; never raises."""

    def __init__(self, run_id: str, secrets: Iterable[str] = ()) -> None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))[:64] or "unknown"
        self.path = os.path.join(evidence_dir(), f"{safe_id}.jsonl")
        self._secrets = tuple(s for s in secrets if s)
        # The phases entered and not yet left, innermost last.
        self._phases: list[str] = []

    def __call__(self, record: dict[str, Any]) -> None:
        try:
            event = record.get("event")
            payload = dict(record.get("payload") or {})
            if event == EVENT_PHASE_ENTER:
                self._phases.append(str(payload.get("phase")))
            phase = self._phases[-1] if self._phases else None
            if event == EVENT_PHASE_EXIT and phase == payload.get("phase"):
                self._phases.pop()
            if event == EVENT_TOOL_CALL and isinstance(payload.get("result"), str):
                payload["result"] = _cut(redact_trace_text(
                    redact_text(payload["result"], secrets=self._secrets)[0]), RESULT_CHARS)
            line = json.dumps(_safe({**record, "phase": phase, "payload": payload}, self._secrets),
                              default=str, ensure_ascii=False)
            with _LOCK:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            logger.debug("Could not keep the run record in %s", self.path, exc_info=True)


def prune_run_evidence() -> None:
    """Remove run records older than ``BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS``."""
    directory = evidence_dir()
    cutoff = time.time() - BackendSettings().llm_run_evidence_retention_days * 86400
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        path = os.path.join(directory, name)
        try:
            if name.endswith(".jsonl") and os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError as exc:
            logger.debug("Skipping run record %s: %s", path, exc)
