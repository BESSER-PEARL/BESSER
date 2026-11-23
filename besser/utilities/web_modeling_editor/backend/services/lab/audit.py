"""BESSER Lab's audit log: who did what to which change and run, and what it cost.

One JSON-lines file per UTC month, ``<state>/audit/<YYYY-MM>.jsonl``, append-only. It outlives dropped changes
(whose folders are deleted), and this month's ``run`` lines are the spending the monthly caps
(``budget.py``) count. The JSON-lines helpers here are also how a change's thread (``thread.py``) is stored.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

AUDIT_DIRNAME = "audit"
AUDIT_LINE_VERSION = 1
_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def month_of(at: float) -> str:
    """The UTC month (``YYYY-MM``) of a time in seconds since the epoch."""
    return datetime.fromtimestamp(at, tz=timezone.utc).strftime("%Y-%m")


def append_json_line(path: str, record: dict[str, Any]) -> None:
    """Append ``record`` to the JSON-lines file at ``path`` with one ``write()`` on an ``O_APPEND`` descriptor.

    When the file does not end in a newline (a write torn by a crash), the line starts with one, so only the
    torn line is lost. Not thread-safe on its own: the caller holds the lock that orders the file's writers.
    """
    data = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o644)
    try:
        size = os.fstat(fd).st_size
        if size:
            os.lseek(fd, size - 1, os.SEEK_SET)
            if os.read(fd, 1) != b"\n":
                data = b"\n" + data
        os.write(fd, data)
    finally:
        os.close(fd)


def read_json_lines(path: str) -> Iterator[dict[str, Any]]:
    """The JSON objects of a JSON-lines file, in order; none when it is missing.

    A line that is not a JSON object (a write torn by a crash) is skipped with a warning naming the file and line.
    """
    try:
        handle = open(path, "rb")
    except FileNotFoundError:
        return
    with handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                record = None
            if not isinstance(record, dict):
                logger.warning("Skipped line %d of %s: not a whole JSON object (a torn write?)", number, path)
                continue
            yield record


@dataclass(frozen=True)
class AuditEntry:
    """One line of the audit log.

    ``action`` is one of ``models.lab.LabAuditAction``; ``cost_usd`` counts towards the monthly caps on a
    ``run`` line only.
    """

    at: float
    login: str
    action: str
    change_id: Optional[str] = None
    run_id: Optional[str] = None
    step: Optional[str] = None
    cost_usd: Optional[float] = None
    stop_reason: Optional[str] = None
    model: str = ""
    seconds: Optional[float] = None
    detail: str = ""
    #: a run paid with the user's own key: recorded, and not counted towards the monthly caps
    own_key: bool = False

    def to_line(self) -> dict[str, Any]:
        """The JSON line: the fields in camelCase, with the line version ``v``."""
        return {"v": AUDIT_LINE_VERSION, "at": self.at, "login": self.login, "action": self.action,
                "changeId": self.change_id, "runId": self.run_id, "step": self.step, "costUsd": self.cost_usd,
                "stopReason": self.stop_reason, "model": self.model, "seconds": self.seconds,
                "detail": self.detail, "ownKey": self.own_key}

    @classmethod
    def from_line(cls, line: dict[str, Any]) -> "AuditEntry":
        """An entry from its JSON line; ``KeyError`` or ``TypeError`` when the line is not one."""
        return cls(at=float(line["at"]), login=str(line["login"]), action=str(line["action"]),
                   change_id=line.get("changeId"), run_id=line.get("runId"), step=line.get("step"),
                   cost_usd=line.get("costUsd"), stop_reason=line.get("stopReason"), model=line.get("model", ""),
                   seconds=line.get("seconds"), detail=line.get("detail", ""),
                   own_key=bool(line.get("ownKey", False)))

    @property
    def spent_usd(self) -> float:
        """What this entry counts towards the monthly caps."""
        return float(self.cost_usd or 0.0) if self.action == "run" and not self.own_key else 0.0


class AuditLog:
    """The lab's audit log under ``<state_dir>/audit/``, with this month's spending kept in memory.

    Args:
        state_dir: The lab's state folder (``BESSER_LAB_STATE_DIR``).
        clock: Seconds since the epoch; tests pass a fake one to cross a month boundary.
    """

    def __init__(self, state_dir: str, *, clock: Callable[[], float] = time.time):
        self._dir = os.path.join(state_dir, AUDIT_DIRNAME)
        self._clock = clock
        self._lock = threading.Lock()
        self._month = ""
        self._by_login: dict[str, float] = {}
        with self._lock:
            self._roll_month()

    def current_month(self) -> str:
        """This UTC month, ``YYYY-MM``."""
        return month_of(self._clock())

    def append(self, entry: AuditEntry) -> None:
        """Append ``entry`` to the file of its UTC month; a ``run`` of this month adds to the spending."""
        with self._lock:
            month = month_of(entry.at)
            append_json_line(self._path(month), entry.to_line())
            self._roll_month()
            if month == self._month and entry.spent_usd:
                self._by_login[entry.login] = self._by_login.get(entry.login, 0.0) + entry.spent_usd

    def entries(self, month: str) -> list[AuditEntry]:
        """Every entry of ``month`` (``YYYY-MM``), oldest first; ``ValueError`` for a malformed month."""
        if not _MONTH_PATTERN.match(month):
            raise ValueError(f"not a month (YYYY-MM): {month!r}")
        return self._read(month)

    def spent(self, login: Optional[str] = None) -> float:
        """What ``login``'s runs cost this UTC month; with ``login`` None, every login's runs together."""
        with self._lock:
            self._roll_month()
            if login is None:
                return sum(self._by_login.values())
            return self._by_login.get(login, 0.0)

    def _roll_month(self) -> None:
        """Load the month's totals when the clock entered a new month (and at start). Holds ``_lock``."""
        month = self.current_month()
        if month == self._month:
            return
        totals: dict[str, float] = {}
        for entry in self._read(month):
            if entry.spent_usd:
                totals[entry.login] = totals.get(entry.login, 0.0) + entry.spent_usd
        self._month, self._by_login = month, totals

    def _read(self, month: str) -> list[AuditEntry]:
        path = self._path(month)
        entries = []
        for number, line in enumerate(read_json_lines(path), start=1):
            try:
                entries.append(AuditEntry.from_line(line))
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipped entry %d of %s: not an audit entry", number, path)
        return entries

    def _path(self, month: str) -> str:
        return os.path.join(self._dir, f"{month}.jsonl")
