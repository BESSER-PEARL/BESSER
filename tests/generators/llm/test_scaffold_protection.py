"""The protect_scaffold configuration must not refuse into a dead end.

``protect_scaffold`` is set from ``_is_free_local_model(model)``, which keys on
the SHAPE of the model id (``qwen3:30b`` self-hosted vs
``qwen/qwen3-30b-a3b-instruct-2507`` gateway), so every measured run so far ran
with it OFF - and it switches ON in the on-prem install. Its refusals had never
been executed.

The named contradiction (a write_file refusal offering ``delete_file +
write_file`` while delete_file refuses generator files) was removed in
d55dee91. What remained was the mirror image: the delete refusal offered
``write_file`` unconditionally, while ``_write_file`` refuses a generator file
over 200 lines until two modify_file edits have landed. Every refusal must now
name the route the executor actually permits FOR THAT FILE.
"""
from __future__ import annotations

import json
import os

from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import FILE_TOOLS


def _protected(tmp_path, lines: int) -> tuple[ToolExecutor, str]:
    """A generator-owned file of ``lines`` lines under scaffold protection."""
    executor = ToolExecutor(workspace=str(tmp_path), protect_scaffold=True)
    rel = "backend/api.py"
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("".join(f"line_{n} = {n}\n" for n in range(lines)))
    executor._generator_files.add(rel)
    executor._read_file({"path": rel})
    return executor, rel


def _run(executor, tool, args) -> dict:
    return json.loads(executor.execute(tool, args))


def test_delete_refusal_on_a_large_scaffold_file_does_not_offer_a_refused_rewrite(tmp_path):
    executor, rel = _protected(tmp_path, 300)

    refusal = _run(executor, "delete_file", {"path": rel})["error"]
    assert "Refused to delete" in refusal
    assert "modify_file" in refusal
    # The advice must be true: write_file on this file is itself refused now.
    assert "unlocks only after two modify_file attempts" in refusal

    blocked = _run(executor, "write_file", {"path": rel, "content": "x = 1\n"})
    assert "error" in blocked, "the delete refusal named a rule the executor does not enforce"

    for n in (0, 1):
        landed = _run(executor, "modify_file", {
            "path": rel, "old_text": f"line_{n} = {n}", "new_text": f"line_{n} = {n + 10}",
        })
        assert landed.get("status") == "modified"
    unlocked = _run(executor, "write_file", {"path": rel, "content": "x = 1\n"})
    assert unlocked.get("status") == "written"


def test_delete_refusal_on_a_small_scaffold_file_offers_the_rewrite_that_works(tmp_path):
    executor, rel = _protected(tmp_path, 40)

    refusal = _run(executor, "delete_file", {"path": rel})["error"]
    assert "write_file it back" in refusal
    assert "unlocks only after two modify_file attempts" not in refusal

    # Following the advice must succeed on the first attempt.
    assert _run(executor, "write_file", {
        "path": rel, "content": "x = 1\n",
    }).get("status") == "written"


def test_protection_still_refuses_the_delete_and_permits_the_edit(tmp_path):
    executor, rel = _protected(tmp_path, 40)
    assert "error" in _run(executor, "delete_file", {"path": rel})
    assert os.path.isfile(os.path.join(str(tmp_path), rel))

    # A file the model created this run is still its own junk to remove.
    assert _run(executor, "write_file", {
        "path": "scratch.py", "content": "y = 2\n",
    }).get("status") == "written"
    assert _run(executor, "delete_file", {"path": "scratch.py"}).get("status") == "deleted"


def test_delete_file_description_does_not_advertise_deleting_the_scaffold():
    """The schema told the model deleting generator output was the tool's purpose.

    "remove dead files the deterministic generator left behind ... a leftover
    FastAPI main_api.py after you switched the project to Flask" is exactly
    what the system prompt forbids ("NEVER switch frameworks", "do NOT call
    delete_file on any file shown below") and what protect_scaffold refuses.
    """
    description = next(t["description"] for t in FILE_TOOLS if t["name"] == "delete_file").lower()
    assert "generator left behind" not in description
    assert "switched the project to flask" not in description
    assert "modify_file" in description
