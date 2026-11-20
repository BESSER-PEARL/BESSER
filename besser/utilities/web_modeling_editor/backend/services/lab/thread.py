"""A change's thread: what was said and done to it, ``<state>/changes/<id>/thread.jsonl``.

Append-only JSON lines (``"v": 1``): messages (with their login, action and the class diagram they carried),
runs (``running``, then how they ended, with their ``LabDone``), pull-request events and notices. The editor
rebuilds the chat from it, so a reload, another browser or a teammate sees the same thread. Cost and authorship
stay in the change's run records (``runs.jsonl``), joined here by run id.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from pydantic import TypeAdapter, ValidationError

from besser.utilities.web_modeling_editor.backend.models.lab import (
    CHANGE_ID_PATTERN,
    LabDone,
    LabPullRequest,
    LabRunRecord,
    LabThreadItem,
)
from besser.utilities.web_modeling_editor.backend.services.lab.audit import append_json_line, read_json_lines

logger = logging.getLogger(__name__)

THREAD_FILENAME = "thread.jsonl"
THREAD_LINE_VERSION = 1
#: The class diagram a message carried is kept for the canvas to be rebuilt from; one larger than this would
#: make every read of the thread slow, so it is dropped (with a notice in the thread).
MAX_THREAD_MODEL_BYTES = 512 * 1024
#: The newest items the change view shows.
DEFAULT_THREAD_LIMIT = 500
_FINISHED_STATUSES = ("done", "failed", "cancelled")
_ITEM = TypeAdapter(LabThreadItem)
# Thread appends are rare and tiny: one lock orders the writers of every thread file in this process.
_APPEND_LOCK = threading.Lock()


class ChangeThread:
    """The thread of one change.

    Args:
        state_dir: The lab's state folder (``BESSER_LAB_STATE_DIR``).
        change_id: The change's id (12 hex digits); ``ValueError`` otherwise.
    """

    def __init__(self, state_dir: str, change_id: str):
        if not re.match(CHANGE_ID_PATTERN, change_id):
            raise ValueError(f"not a change id: {change_id!r}")
        self.change_id = change_id
        self.path = os.path.join(state_dir, "changes", change_id, THREAD_FILENAME)

    def add_message(self, *, login: str, text: str, action: str, model: Optional[dict[str, Any]]) -> str:
        """Record a message sent to the change and return its id.

        ``model`` is the class diagram the message carried (the editor's model JSON). One over
        ``MAX_THREAD_MODEL_BYTES`` is dropped, and a notice line after the message says so.
        """
        message_id = secrets.token_hex(6)
        too_big = model is not None and len(json.dumps(model, separators=(",", ":")).encode("utf-8")) > MAX_THREAD_MODEL_BYTES
        self._append({"type": "message", "id": message_id, "login": login, "text": text, "action": action,
                      "model": None if too_big else model})
        if too_big:
            logger.warning("Change %s: the class diagram of message %s is over %d bytes and was not kept",
                           self.change_id, message_id, MAX_THREAD_MODEL_BYTES)
            self.add_notice(login, f"The class diagram sent with this message was over "
                                   f"{MAX_THREAD_MODEL_BYTES // 1024} KB, so the lab did not keep it.")
        return message_id

    def run_started(self, run_id: str, login: str) -> None:
        """Record that ``login`` started run ``run_id`` on the change."""
        self._append({"type": "run", "runId": run_id, "login": login, "status": "running"})

    def run_finished(self, run_id: str, login: str, *, status: str, lab: Optional[LabDone],
                     error: Optional[dict[str, str]]) -> None:
        """Record how run ``run_id`` ended: ``done``, ``failed`` or ``cancelled``; ``error`` is ``{code, message}``."""
        if status not in _FINISHED_STATUSES:
            raise ValueError(f"not how a run ends: {status!r}")
        self._append({"type": "run", "runId": run_id, "login": login, "status": status,
                      "lab": None if lab is None else lab.model_dump(mode="json"), "error": error})

    def add_pull_requests(self, login: str, pull_requests: Sequence[LabPullRequest], checks_note: str) -> None:
        """Record pull requests ``login`` opened or updated for the change."""
        self._append({"type": "pull_requests", "login": login,
                      "pullRequests": [pr.model_dump(mode="json") for pr in pull_requests],
                      "checksNote": checks_note})

    def add_notice(self, login: str, text: str) -> None:
        """Record something that happened to the change, for every viewer."""
        self._append({"type": "notice", "login": login, "text": text})

    def items(self, runs: Mapping[str, LabRunRecord], limit: int = DEFAULT_THREAD_LIMIT) -> list[LabThreadItem]:
        """The thread's newest ``limit`` items, oldest first.

        A run is one item, where it started, holding its last line. ``costUsd`` comes from its record in ``runs``
        (the change's run records, by run id); ``planVersion`` numbers the finished plan runs, first plan 1.
        A line that is torn, of another version or of an unknown shape is skipped with a warning.
        """
        order: list[tuple[str, Any]] = []
        last_of_run: dict[str, dict[str, Any]] = {}
        for line in read_json_lines(self.path):
            if line.get("v") != THREAD_LINE_VERSION:
                logger.warning("Skipped a line of %s: version %r, not %d", self.path, line.get("v"),
                               THREAD_LINE_VERSION)
                continue
            run_id = line.get("runId")
            if line.get("type") == "run" and isinstance(run_id, str):
                if run_id not in last_of_run:
                    order.append(("run", run_id))
                last_of_run[run_id] = line
            else:
                order.append(("line", line))

        items: list[LabThreadItem] = []
        plans = 0
        for kind, value in order:
            data = {key: field for key, field in (last_of_run[value] if kind == "run" else value).items()
                    if key != "v"}
            if kind == "run":
                record = runs.get(value)
                finished_plan = record is not None and record.step == "plan" and data.get("status") == "done"
                plans += finished_plan
                data.setdefault("lab", None)
                data.setdefault("error", None)
                data["costUsd"] = record.usd if record is not None else None
                data["planVersion"] = plans if finished_plan else None
            try:
                items.append(_ITEM.validate_python(data))
            except ValidationError as error:
                logger.warning("Skipped a %r line of %s: %d field errors", data.get("type"), self.path,
                               error.error_count())
        return items[max(len(items) - limit, 0):]

    def _append(self, line: dict[str, Any]) -> None:
        with _APPEND_LOCK:
            append_json_line(self.path, {"v": THREAD_LINE_VERSION, "at": time.time(), **line})
