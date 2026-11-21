"""The API workflows a run kept, written beside the app so a later modify run replays them.

The recipe's copy is cut to fit its size bound (``recipe_size``), request bodies first;
this file keeps each workflow's definition whole within its own bound. The workspace is
writable by the model's tools, so the reader takes only a bounded file of valid definitions.
"""

from __future__ import annotations

import json
import logging
import os

from besser.spec_driven_agent.state.checkpoint import api_scenario_snapshot
from besser.utilities.workspace_fs import open_plain, open_plain_write

logger = logging.getLogger(__name__)

WORKFLOWS_FILENAME = ".besser_api_workflows.json"
MAX_WORKFLOWS_BYTES = 128 * 1024
# As many as a run retains.
MAX_WORKFLOWS = 10


def workflow_entries(records) -> list[dict]:
    """``{"scenario_id", "scenario", "status"}`` for retained ``test_api`` records."""
    return [{"scenario_id": saved["scenario_id"], "scenario": saved["scenario"],
             "status": saved["report"].get("status")}
            for saved in api_scenario_snapshot(records, include_reports=True)]


def save_workflows(output_dir: str, entries: list[dict]) -> None:
    """Write ``entries`` in order, whole, while they fit ``MAX_WORKFLOWS_BYTES``; say what did not."""
    kept, size = [], 0
    for entry in entries[:MAX_WORKFLOWS]:
        entry_size = len(json.dumps(entry).encode("utf-8")) + 2
        if size + entry_size > MAX_WORKFLOWS_BYTES - 32:
            logger.warning("API workflow %r not kept for later runs: %s is limited to %d bytes",
                           entry.get("scenario_id"), WORKFLOWS_FILENAME, MAX_WORKFLOWS_BYTES)
            continue
        kept.append(entry)
        size += entry_size
    path = os.path.join(output_dir, WORKFLOWS_FILENAME)
    try:
        if not kept:
            if os.path.isfile(path):
                os.remove(path)
            return
        with open_plain_write(path, "w", root=output_dir, encoding="utf-8") as handle:
            json.dump({"workflows": kept}, handle)
    except OSError:
        logger.warning("Could not write %s", WORKFLOWS_FILENAME, exc_info=True)


def load_workflows(output_dir: str) -> list[dict]:
    """The workflows ``save_workflows`` wrote, validated; ``[]`` when absent, oversized or invalid."""
    try:
        with open_plain(os.path.join(output_dir, WORKFLOWS_FILENAME), "rb", root=output_dir) as handle:
            data = handle.read(MAX_WORKFLOWS_BYTES + 1)
    except OSError:
        return []
    if len(data) > MAX_WORKFLOWS_BYTES:
        logger.warning("%s is over %d bytes; it is not read", WORKFLOWS_FILENAME, MAX_WORKFLOWS_BYTES)
        return []
    try:
        entries = json.loads(data).get("workflows")
        if not isinstance(entries, list) or len(entries) > MAX_WORKFLOWS:
            raise ValueError("not a list of at most ten workflows")
        statuses = [entry.get("status") if isinstance(entry, dict) else None for entry in entries]
        return [{"scenario_id": saved["scenario_id"], "scenario": saved["scenario"],
                 "status": status if isinstance(status, str) else None}
                for saved, status in zip(api_scenario_snapshot(entries), statuses)]
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError) as exc:
        logger.warning("%s ignored: %s", WORKFLOWS_FILENAME, exc)
        return []
