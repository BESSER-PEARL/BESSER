"""Persistent incident log for user-facing generation failures.

"A user hit an issue → we have a record."  Every SSE ``error`` event a
run emits, and every exception that would otherwise close the stream
silently, is appended as one JSON line to a host-mounted file — so the
evidence survives container redeploys (the 94050aaa lesson: the only
traceback lived in container logs that a recreate destroyed).

Best-effort by design: the incident log must never break a run.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _incident_path() -> str:
    directory = os.environ.get("BESSER_INCIDENT_LOG_DIR", "/app/incidents")
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        return ""
    return os.path.join(directory, "incidents.jsonl")


def record_incident(
    kind: str,
    run_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    code: str | None = None,
    message: str | None = None,
    traceback_text: str | None = None,
    instructions: str | None = None,
) -> None:
    """Append one incident record. Never raises."""
    try:
        path = _incident_path()
        if not path:
            return
        entry = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "code": code,
            "message": (message or "")[:2000],
            "traceback": (traceback_text or "")[:8000] or None,
            "instructions": (instructions or "")[:300] or None,
        }
        line = json.dumps({k: v for k, v in entry.items() if v is not None},
                          ensure_ascii=False)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # pragma: no cover — must never break a run
        logger.debug("Could not record incident", exc_info=True)
