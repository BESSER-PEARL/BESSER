"""Small safety/integration matrix for quotation-free edit recovery."""
import json
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent import tool_executor
from besser.spec_driven_agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.llm_client import UsageTracker


SOURCE = "# keep\nasync def action():\n    return False\n\ndef unrelated():\n    return False\n"


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


@pytest.mark.parametrize("failure", [None, "stale", "unread", "syntax", "other_file", "resume", "noop", "elision"])
def test_range_edits_preserve_guards_and_only_touch_the_selected_block(tmp_path, failure):
    target = tmp_path / "app.py"
    target.write_text(SOURCE, encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path))
    read = call(ex, "read_file", path="app.py", offset=1, limit=2)
    args = dict(path="app.py", read_id=read["read_id"], start_line=2, end_line=3,
                new_text="async def action():\n    return True\n")
    expected = SOURCE
    if failure == "stale":
        call(ex, "modify_file", path="app.py", old_text="# keep", new_text="# changed")
        expected = SOURCE.replace("# keep", "# changed")
    elif failure == "unread":
        args.update(end_line=6)
    elif failure == "syntax":
        args.update(new_text="    @route()\nasync def action():\n    return True\n")
    elif failure == "other_file":
        (tmp_path / "other.py").write_text(SOURCE, encoding="utf-8")
        args.update(path="other.py")
    elif failure == "resume":
        ex = ToolExecutor(workspace=str(tmp_path))
        fresh = call(ex, "read_file", path="app.py", offset=1, limit=2)
        assert fresh["read_id"] != read["read_id"]
    elif failure == "noop":
        args.update(new_text="async def action():\n    return False\n")
    elif failure == "elision":
        args.update(new_text="async def action():\n    ...\n")
    result = call(ex, "replace_file_lines", **args)
    if failure:
        assert "error" in result, result
        assert target.read_text(encoding="utf-8") == expected
        assert not result.get("status") == "already_applied"
    else:
        assert result["status"] == "modified", result
        expected = SOURCE.replace("return False", "return True", 1)
        assert target.read_text(encoding="utf-8") == expected
        # A replay must never consume the next block or duplicate an insertion.
        replay = call(ex, "replace_file_lines", **args)
        assert replay["rejection_kind"] == "stale_read"
        assert target.read_text(encoding="utf-8") == expected


def test_a_truncated_read_never_authorizes_unseen_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_executor, "MAX_FILE_READ", 45)
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path))
    read = call(ex, "read_file", path="app.py")
    assert read["truncated"]
    assert read["end_line"] == 2
    result = call(ex, "replace_file_lines", path="app.py", read_id=read["read_id"],
                  start_line=3, end_line=3, new_text="    return True\n")
    assert result["rejection_kind"] == "unread_range"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == SOURCE


@pytest.mark.parametrize("contents", ["x = 1\n", ""])
def test_the_displayed_terminal_empty_line_is_a_valid_eof_anchor(tmp_path, contents):
    (tmp_path / "app.py").write_text(contents, encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path))
    read = call(ex, "read_file", path="app.py")
    result = call(ex, "replace_file_lines", path="app.py", read_id=read["read_id"],
                  start_line=read["end_line"], end_line=read["end_line"], new_text="y = 2\n")
    assert result["status"] == "modified", result
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == contents + "y = 2\n"


@pytest.mark.parametrize("phase", ["generation", "repair"])
def test_repeated_qwen_shaped_quotation_failures_switch_strategy_and_recover(tmp_path, phase):
    """The w7zoeszt shape: first quoted line shifted, subsequent lines not.

    Exercise the real loop, executor, provider tool-choice, and trace together;
    a successful range edit must count as progress in BOTH phases.
    """
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")

    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def __init__(self):
            self.forced = []

        def chat(self, system, messages, tools, force_tool=None, **kwargs):
            self.forced.append(force_tool)
            n = len(self.forced)
            if n > 4:
                return {"stop_reason": "end_turn", "content": []}
            name = "modify_file"
            args = dict(path="app.py", old_text="    async def action():\n    return False\n",
                        new_text="async def action():\n    return True\n")
            if force_tool == "read_file":
                name, args = "read_file", dict(path="app.py", offset=1, limit=2)
            elif force_tool == "replace_file_lines":
                results = [json.loads(b["content"]) for m in messages if isinstance(m.get("content"), list)
                           for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
                # Either a read_file view or the span a missed modify_file located.
                read = next(r.get("located_range", r) for r in reversed(results)
                            if "read_id" in r or "located_range" in r)
                name, args = "replace_file_lines", dict(path="app.py", read_id=read["read_id"],
                    start_line=read.get("start_line", 2), end_line=read.get("end_line", 3),
                    new_text="async def action():\n    return True\n")
            return {"stop_reason": "tool_use", "content": [SimpleNamespace(
                type="tool_use", name=name, id=f"t{n}", input=args)]}

    client = Client()
    orch = LLMOrchestrator(llm_client=client, output_dir=str(tmp_path),
        state_machines=[SimpleNamespace(name="Test")], enable_checkpointing=False,
        max_turns=8, max_cost_usd=1, enable_tracing=True)
    if phase == "repair":
        assert orch._invoke_phase3_fix_loop([ValidationIssue("blocker", "Wrong behavior in app.py line 3")],
                                           is_first_attempt=True) == 1
    else:
        orch.run("Change action to return True; preserve unrelated.")
    # One miss, not two plus a re-read: the miss itself brackets the target
    # and issues the read_id, so the range edit is the very next call.
    assert client.forced[:2] == [None, "replace_file_lines"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == SOURCE.replace("return False", "return True", 1)
    assert not orch.executor._frozen("app.py")
    writes = [e for e in orch.tool_calls_log if e["tool"] == "replace_file_lines"]
    assert len(writes) == 1 and writes[0]["success"]
    trace = (tmp_path / ".besser_tool_inputs.jsonl").read_text(encoding="utf-8")
    assert '"tool": "replace_file_lines"' in trace
