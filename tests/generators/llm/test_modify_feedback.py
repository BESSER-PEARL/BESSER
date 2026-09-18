"""What the edit tools tell the model, and what they refuse.

The mechanisms mature coding agents converged on (OpenHands' str_replace
editor, Claude Code's Edit, SWE-agent), missing here until 2026-09-17:

* an ambiguous ``old_text`` says WHERE each occurrence is (enclosing def),
* a successful ``modify_file`` echoes the edited region back, numbered,
* a miss on a file the model never saw this run says so first,
* a quote copied from numbered output still matches,
* ``write_file`` over an existing file the model never saw is refused,
* ``task_list(action='drop', reason=...)`` is the honest exit for an item
  the user did not ask for (the live run marked "authentication" done with
  nothing built, because done was the only way past the end-turn gate).
"""

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.edit_apply import replace_most_similar_chunk
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tool_executor import ToolExecutor

STUB = (
    "import x\n"
    "\n"
    "def produce_bill():\n"
    "    try:\n"
    "        raise NotImplementedError\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "def cancel():\n"
    "    try:\n"
    "        raise NotImplementedError\n"
    "    except Exception:\n"
    "        pass\n"
)


def _executor(tmp_path, **files) -> ToolExecutor:
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


class TestAmbiguityNamesEachOccurrence:

    def test_error_lists_line_and_enclosing_def(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": STUB})
        ex.mark_known(["m.py"])
        res = ex._modify_file({
            "path": "m.py",
            "old_text": "    try:\n        raise NotImplementedError\n",
            "new_text": "    return 1\n",
        })
        assert "ambiguous" in res["error"]
        locs = res["occurrences"]
        assert len(locs) == 2
        assert "produce_bill" in locs[0] and "line 4" in locs[0]
        assert "cancel" in locs[1] and "line 10" in locs[1]


class TestPostEditEcho:

    def test_result_carries_the_edited_region_numbered(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": STUB})
        ex.mark_known(["m.py"])
        # The whole body: dropping only the try would orphan its except, and
        # since 2026-09-18 an edit that leaves a valid file unparseable is
        # refused rather than written with a diagnostic.
        res = ex._modify_file({
            "path": "m.py",
            "old_text": (
                "def cancel():\n    try:\n        raise NotImplementedError\n"
                "    except Exception:\n        pass\n"
            ),
            "new_text": "def cancel():\n    return 'cancelled'\n",
        })
        assert res["status"] == "modified", res
        snippet = res["snippet"]
        assert "|     return 'cancelled'" in snippet
        assert "  10| " in snippet           # numbered from the file, not from 1
        assert "produce_bill" not in snippet  # only the region plus 2 lines of context


class TestUnseenFile:

    def test_miss_on_a_never_read_file_says_so_first(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": STUB})
        res = ex._modify_file({"path": "m.py", "old_text": "nope\n", "new_text": "x\n"})
        assert res["error"].startswith("You have not read m.py this run")
        assert "read_file" in res["error"]

    @pytest.mark.parametrize("seen_by", ["read_file", "write_file", "mark_known"])
    def test_a_seen_file_gets_the_ordinary_miss_reply(self, tmp_path, seen_by):
        ex = _executor(tmp_path, **{"m.py": STUB})
        if seen_by == "read_file":
            ex._read_file({"path": "m.py"})
        elif seen_by == "write_file":
            ex.mark_known(["m.py"])                      # a rewrite needs to have seen it
            ex._write_file({"path": "m.py", "content": STUB})
        else:
            ex.mark_known(["m.py"])
        res = ex._modify_file({"path": "m.py", "old_text": "nope\n", "new_text": "x\n"})
        assert res["error"].startswith("old_text not found"), res["error"]

    def test_write_file_over_an_unseen_existing_file_is_refused(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": STUB})
        res = ex._write_file({"path": "m.py", "content": "rewritten\n"})
        assert "have not read it this run" in res["error"]
        assert (tmp_path / "m.py").read_text(encoding="utf-8") == STUB    # untouched

    def test_write_file_creates_new_files_and_rewrites_seen_ones(self, tmp_path):
        ex = _executor(tmp_path)
        assert ex._write_file({"path": "new.py", "content": "a = 1\n"})["status"] == "written"
        assert ex._write_file({"path": "new.py", "content": "a = 2\n"})["status"] == "written"


class TestNumberedQuotes:

    def test_quote_copied_from_numbered_output_still_matches(self):
        whole = "a = 1\nb = 2\nc = 3\n"
        part = "   2| b = 2\n   3| c = 3\n"
        assert replace_most_similar_chunk(whole, part, "b = 20\nc = 30\n") == "a = 1\nb = 20\nc = 30\n"

    def test_numbered_new_text_loses_its_prefixes_too(self):
        whole = "a = 1\nb = 2\n"
        res = replace_most_similar_chunk(whole, "   2| b = 2\n", "   2| b = 3\n")
        assert res == "a = 1\nb = 3\n"

    def test_a_partly_numbered_block_is_not_touched(self):
        whole = "a = 1\n1| b = 2\n"
        assert replace_most_similar_chunk(whole, "a = 1\n1| b = 2\n", "x\n") == "x\n"


class TestChecklistDrop:

    def test_drop_closes_the_item_with_its_reason(self, tmp_path):
        ex = ToolExecutor(workspace=str(tmp_path))
        ex.set_tasks(["Add authentication", "Implement Booking.cancel"])
        res = ex._task_list({"action": "drop", "id": 1, "reason": "the request never mentions auth"})
        assert res == {"status": "dropped", "id": 1, "open": 1}
        listing = ex._task_list({"action": "list"})["tasks"]
        assert listing[0]["status"] == "dropped"
        assert listing[0]["reason"] == "the request never mentions auth"
        assert listing[1]["status"] == "open"
        assert [t["text"] for t in ex.open_tasks()] == ["Implement Booking.cancel"]

    def test_drop_needs_an_id_and_a_reason(self, tmp_path):
        ex = ToolExecutor(workspace=str(tmp_path))
        ex.set_tasks(["Add authentication"])
        assert "reason" in ex._task_list({"action": "drop", "id": 1})["error"]
        assert "Unknown task id" in ex._task_list({"action": "drop", "id": 9, "reason": "x"})["error"]
        assert ex.open_tasks()

    def test_the_tool_schema_offers_drop(self):
        from besser.generators.llm.tools import VALIDATION_TOOLS
        tool = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")
        assert "drop" in tool["input_schema"]["properties"]["action"]["enum"]
        assert "reason" in tool["input_schema"]["properties"]


class TestScaffoldCopyInThePrompt:
    """The pasted scaffold copy is OFF by default (like other coding agents,
    file text reaches the model only through read_file, which is current).
    Opted in, files it inlines count as read; a big file it did not is not."""

    class _Block:
        def __init__(self, block_type, **kwargs):
            self.type = block_type
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _run(self, tmp_path):
        """Two modify_file misses (small file, then big file), then end_turn.
        Returns (system prompt seen on turn 3, tool_result texts in order)."""
        user = Class(name="User")
        user.attributes = {
            Property(name="id", type=PrimitiveDataType("int"), is_id=True),
            Property(name="name", type=PrimitiveDataType("str")),
        }
        model = DomainModel(name="M", types={user})
        Block = self._Block
        turns = {"n": 0}
        seen: dict = {}

        class Client:
            model = "mock"
            usage = UsageTracker("mock")

            def chat(self, system, messages, tools):
                turns["n"] += 1
                if turns["n"] == 3:
                    seen["system"] = system
                    seen["messages"] = [dict(m) for m in messages]
                    return {"stop_reason": "end_turn", "content": [Block("text", text="done")]}
                target = "backend/database.py" if turns["n"] == 1 else "backend/main_api.py"
                return {"stop_reason": "tool_use", "content": [Block(
                    "tool_use", name="modify_file", id=f"c{turns['n']}",
                    input={"path": target, "old_text": "nope nope\n", "new_text": "x\n"},
                )]}

        orch = LLMOrchestrator(
            llm_client=Client(), domain_model=model, output_dir=str(tmp_path), max_turns=10,
        )
        orch.run("Build a FastAPI backend")
        assert orch._generator_used == "generate_fastapi_backend", "precondition: scaffold ran"
        assert os.path.isfile(os.path.join(str(tmp_path), "backend", "main_api.py"))
        assert "messages" in seen, "mock never reached turn 3"

        results = []
        for m in seen["messages"]:
            if m.get("role") != "user" or not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results.append(json.dumps(b.get("content")))
        assert len(results) >= 2, results
        return seen["system"], results

    def test_off_by_default_so_every_file_must_be_read_first(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BESSER_LLM_INLINE_SCAFFOLD", raising=False)
        system, results = self._run(tmp_path)
        assert "Scaffold file contents" not in system
        assert "have not read" in results[0]          # database.py: not pasted, not read
        assert "have not read" in results[1]

    def test_opted_in_inlined_files_count_as_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BESSER_LLM_INLINE_SCAFFOLD", "1")
        system, results = self._run(tmp_path)
        assert "Scaffold file contents" in system
        assert "have not read" not in results[0]      # database.py (47 lines) was inlined
        assert "have not read" in results[1]          # main_api.py (200+ lines) was not
