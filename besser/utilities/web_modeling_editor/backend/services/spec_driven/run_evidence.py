"""A run's full record, kept server-side apart from the recipe the user downloads, when the operator asks.

The recipe ships in the app archive, so it is cut to a size budget, and the
run's trace lives in its workspace, which is deleted a few hours after the run.
When ``BESSER_INCIDENT_LOG_DIR`` is set (and ``BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS``
is not 0), :class:`RunEvidence` receives every trace record of a run
(``TraceWriter``'s mirror) and appends it, redacted, to
``<incident log dir>/runs/<run id>.jsonl`` as it happens, so a cancelled or
crashed run keeps what it did. The run's shell never sees that folder
(``execution/sandbox.py`` masks the incident log directory). The hourly cleanup
removes files older than the retention.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Iterable

from besser.spec_driven_agent.state.tracing import EVENT_PHASE_ENTER, EVENT_PHASE_EXIT, EVENT_TOOL_CALL
from besser.utilities.secret_redaction import redact_record_text
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

logger = logging.getLogger(__name__)

EVIDENCE_DIRNAME = "runs"
# Any one string of a record, a written file's content included: first and last halves kept.
VALUE_CHARS = 64 * 1024
# A tool result, as the model read it: its first and last 8 KiB.
RESULT_CHARS = 16 * 1024
# Text past each cut that is checked with the kept part, so a secret across the cut is
# matched whole: longer than any credential (an AWS session token is about 1 KiB).
_CUT_WINDOW = 8 * 1024

_LOCK = threading.Lock()


def evidence_dir() -> str | None:
    """``<BESSER_INCIDENT_LOG_DIR>/runs`` when the operator keeps run records, else None."""
    settings = BackendSettings()
    base = (settings.incident_log_dir or "").strip()
    if not base or settings.llm_run_evidence_retention_days <= 0:
        return None
    return os.path.join(base, EVIDENCE_DIRNAME)


def for_run(run_id: str, secrets: Iterable[str] = ()) -> RunEvidence | None:
    """The record of run ``run_id``, or None when the operator keeps none."""
    directory = evidence_dir()
    if directory is None:
        return None
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))[:64] or "unknown"
    return RunEvidence(os.path.join(directory, f"{safe_id}.jsonl"), secrets)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n<<cut {len(text) - 2 * half} of {len(text)} chars>>\n{text[-half:]}"


def _redacted_cut(text: str, limit: int, redact: Callable[[str], str]) -> str:
    """``text`` redacted and cut to ``limit``.

    When the kept parts and a window past each cut hold nothing to redact, the
    rest is not read; otherwise the whole text is redacted, then cut.
    """
    half = limit // 2
    if len(text) > limit + 2 * _CUT_WINDOW:
        head, tail = text[:half + _CUT_WINDOW], text[-(half + _CUT_WINDOW):]
        if redact(head) == head and redact(tail) == tail:
            return _cut(text, limit)
    return _cut(redact(text), limit)


class RunEvidence:
    """Appends a run's trace records to its server-side file; never raises."""

    def __init__(self, path: str, secrets: Iterable[str] = ()) -> None:
        self.path = path
        self._secrets = tuple(s for s in secrets if s)
        # The phases entered and not yet left, innermost last.
        self._phases: list[str] = []
        self._warned = False

    def _redact(self, text: str, path: str = "") -> str:
        return redact_record_text(text, path=path, secrets=self._secrets)

    def _safe(self, value: Any, limit: int, path: str = "") -> Any:
        if isinstance(value, str):
            return _redacted_cut(value, limit, lambda text: self._redact(text, path))
        if isinstance(value, (list, tuple)):
            return [self._safe(item, limit, path) for item in value]
        if isinstance(value, dict):
            return {key: self._safe(item, limit, path) for key, item in value.items()}
        return value

    def _result(self, result: str, path: str) -> str:
        """A tool result. As JSON, its strings are redacted unescaped, so line rules see real lines."""
        try:
            parsed = json.loads(result)
        except ValueError:
            return _redacted_cut(result, RESULT_CHARS, lambda text: self._redact(text, path))
        # Redacted already: cutting the text only drops characters.
        return _cut(json.dumps(self._safe(parsed, RESULT_CHARS, path), ensure_ascii=False), RESULT_CHARS)

    def __call__(self, record: dict[str, Any]) -> None:
        try:
            event = record.get("event")
            payload = dict(record.get("payload") or {})
            if event == EVENT_PHASE_ENTER:
                self._phases.append(str(payload.get("phase")))
            phase = self._phases[-1] if self._phases else None
            if event == EVENT_PHASE_EXIT and phase == payload.get("phase"):
                self._phases.pop()
            given = payload.get("input")
            path = given.get("path", "") if isinstance(given, dict) and isinstance(given.get("path"), str) else ""
            result = payload.pop("result", None) if event == EVENT_TOOL_CALL else None
            safe = self._safe({**record, "phase": phase, "payload": payload}, VALUE_CHARS, path)
            if isinstance(result, str):
                safe["payload"]["result"] = self._result(result, path)
            elif result is not None:
                safe["payload"]["result"] = self._safe(result, RESULT_CHARS, path)
            line = json.dumps(safe, default=str, ensure_ascii=False)
            with _LOCK:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            if not self._warned:
                self._warned = True
                logger.warning("Could not keep the run record in %s", self.path, exc_info=True)
            else:
                logger.debug("Could not keep the run record in %s", self.path, exc_info=True)


def prune_run_evidence() -> None:
    """Remove run records older than ``BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS``; nothing when records are off."""
    directory = evidence_dir()
    if directory is None:
        return
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
