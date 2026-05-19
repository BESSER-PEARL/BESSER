"""Tests for the per-file modify_file streak guard.

When the LLM makes N consecutive ``modify_file`` calls on the same path
the orchestrator must inject a high-salience reminder into the
conversation BEFORE the next LLM call. The reminder pushes the LLM
toward either rewriting the file in one shot (``write_file``) or
moving on — instead of dribbling out one-line edits.

This guard is separate from the legacy ``_is_stuck`` heuristic, which
fires on N uniform tool calls regardless of arguments and only injects
a low-salience warning into a tool_result.
"""

import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator


@pytest.fixture
def simple_model():
    StringType = PrimitiveDataType("str")
    IntegerType = PrimitiveDataType("int")
    user = Class(name="User")
    user.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
    }
    return DomainModel(name="TestModel", types={user})


class MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


def _seed_file(workspace: str, rel_path: str, content: str = "x = 1\n") -> None:
    """Create a file inside the workspace so modify_file has something
    to work with."""
    full = os.path.join(workspace, rel_path)
    os.makedirs(os.path.dirname(full) or workspace, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _extract_text_blocks(messages: list[dict]) -> list[str]:
    """Pull plain-text content out of any user message in ``messages``.

    The reminder is appended as a separate user message with a single
    ``{"type": "text", "text": ...}`` block — that's what we want to
    surface to the assertion.
    """
    texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "")
                    if isinstance(txt, str):
                        texts.append(txt)
        elif isinstance(content, str):
            texts.append(content)
    return texts


# ======================================================================
# Tests
# ======================================================================


class TestPerFileModifyGuard:

    def test_three_consecutive_modify_triggers_reminder(self, simple_model, tmp_path):
        """N=3 consecutive modify_file calls on the SAME path should
        inject a system-style reminder mentioning ``write_file`` and
        the file path into the conversation before turn 4."""
        # Seed the file so modify_file's old_text substitution is
        # plausible (the executor will still error on missing
        # old_text, but it captures the path for the guard).
        target = "app.py"
        _seed_file(str(tmp_path), target, content="line1\nline2\nline3\n")

        turn_counter = {"n": 0}
        captured_messages: dict[str, list[dict]] = {}

        class StreakClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                # Snapshot the messages that the orchestrator sent in
                # on turn 4 (i.e. AFTER three modify_file rounds have
                # completed). At that point the reminder should
                # already be appended.
                if turn_counter["n"] == 4:
                    captured_messages["turn4"] = [dict(m) for m in messages]
                    return {"stop_reason": "end_turn", "content": [
                        MockBlock("text", text="Stopping."),
                    ]}
                # Turns 1, 2, 3: emit a modify_file on the SAME path.
                # The actual edit will probably fail (old_text is
                # arbitrary) but the tool_executor records the call
                # and the orchestrator counts it. Errors-as-strings
                # are exactly the shape ``_is_stuck`` expects to
                # tolerate.
                return {"stop_reason": "tool_use", "content": [
                    MockBlock(
                        "tool_use",
                        name="modify_file",
                        input={
                            "path": target,
                            "old_text": f"line{turn_counter['n']}",
                            "new_text": f"updated{turn_counter['n']}",
                        },
                        id=f"c{turn_counter['n']}",
                    ),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=StreakClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_turns=10,
        )
        orchestrator.run("Build an app")

        assert "turn4" in captured_messages, (
            "Mock client never observed a 4th turn — the orchestrator "
            "ended Phase 2 early."
        )
        texts = _extract_text_blocks(captured_messages["turn4"])
        joined = "\n".join(texts)
        assert "write_file" in joined, (
            f"Reminder did not mention write_file. Texts: {texts!r}"
        )
        assert target in joined, (
            f"Reminder did not mention path {target!r}. Texts: {texts!r}"
        )
        # The reminder is tagged with a system-reminder marker so the
        # LLM treats it as a meta-instruction, not user content.
        assert "<system-reminder>" in joined

    def test_reminder_does_not_fire_when_interleaved_with_other_tool(
        self, simple_model, tmp_path,
    ):
        """If a different tool call breaks the streak (3 modify_file on
        the same path but with another tool between them), the
        reminder must NOT fire."""
        target = "app.py"
        _seed_file(str(tmp_path), target, content="line1\nline2\nline3\nline4\n")

        # Sequence: modify_file, modify_file, list_files, modify_file,
        # end_turn. That's 3 modify_file on the same path but the
        # list_files breaks the consecutive streak.
        turn_counter = {"n": 0}
        captured_messages: dict[str, list[dict]] = {}

        class InterleavedClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if turn_counter["n"] >= 5:
                    captured_messages[f"turn{turn_counter['n']}"] = [dict(m) for m in messages]
                    return {"stop_reason": "end_turn", "content": [
                        MockBlock("text", text="Done."),
                    ]}
                if turn_counter["n"] == 3:
                    # Interleave a different tool — this resets the
                    # streak in the modify-targets buffer.
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock(
                            "tool_use",
                            name="list_files",
                            input={},
                            id=f"c{turn_counter['n']}",
                        ),
                    ]}
                # Turns 1, 2, 4: modify_file on the same path.
                return {"stop_reason": "tool_use", "content": [
                    MockBlock(
                        "tool_use",
                        name="modify_file",
                        input={
                            "path": target,
                            "old_text": f"line{turn_counter['n']}",
                            "new_text": f"updated{turn_counter['n']}",
                        },
                        id=f"c{turn_counter['n']}",
                    ),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=InterleavedClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_turns=10,
        )
        orchestrator.run("Build an app")

        # Inspect every message the client saw across all turns —
        # the reminder must NOT have appeared anywhere.
        all_texts: list[str] = []
        for snapshot in captured_messages.values():
            all_texts.extend(_extract_text_blocks(snapshot))
        joined = "\n".join(all_texts)
        assert "<system-reminder>" not in joined, (
            "Per-file modify guard fired even though a non-modify tool "
            "broke the streak. Texts: %r" % all_texts
        )

    def test_two_consecutive_modify_does_not_trigger(self, simple_model, tmp_path):
        """At N-1 (=2) consecutive modify_file calls the reminder must
        NOT fire — that's the per-file threshold boundary."""
        target = "app.py"
        _seed_file(str(tmp_path), target, content="line1\nline2\n")

        turn_counter = {"n": 0}
        captured_messages: dict[str, list[dict]] = {}

        class BelowThresholdClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if turn_counter["n"] == 3:
                    captured_messages["turn3"] = [dict(m) for m in messages]
                    return {"stop_reason": "end_turn", "content": [
                        MockBlock("text", text="Done."),
                    ]}
                return {"stop_reason": "tool_use", "content": [
                    MockBlock(
                        "tool_use",
                        name="modify_file",
                        input={
                            "path": target,
                            "old_text": f"line{turn_counter['n']}",
                            "new_text": f"updated{turn_counter['n']}",
                        },
                        id=f"c{turn_counter['n']}",
                    ),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=BelowThresholdClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_turns=10,
        )
        orchestrator.run("Build an app")

        texts = _extract_text_blocks(captured_messages.get("turn3", []))
        joined = "\n".join(texts)
        assert "<system-reminder>" not in joined, (
            "Reminder fired at only 2 consecutive modify_file calls. "
            "Texts: %r" % texts
        )

    def test_reminder_targets_correct_path_when_two_files_modified(
        self, simple_model, tmp_path,
    ):
        """Modifying two distinct files in alternation should NOT
        trigger — the per-file streak only fires when N consecutive
        calls share the SAME path."""
        _seed_file(str(tmp_path), "a.py", content="aaa\n")
        _seed_file(str(tmp_path), "b.py", content="bbb\n")

        turn_counter = {"n": 0}
        captured_messages: dict[str, list[dict]] = {}

        class AlternatingClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if turn_counter["n"] >= 5:
                    captured_messages[f"turn{turn_counter['n']}"] = [dict(m) for m in messages]
                    return {"stop_reason": "end_turn", "content": [
                        MockBlock("text", text="Done."),
                    ]}
                # Alternate between a.py and b.py.
                target = "a.py" if turn_counter["n"] % 2 == 1 else "b.py"
                return {"stop_reason": "tool_use", "content": [
                    MockBlock(
                        "tool_use",
                        name="modify_file",
                        input={
                            "path": target,
                            "old_text": "aaa" if target == "a.py" else "bbb",
                            "new_text": f"updated{turn_counter['n']}",
                        },
                        id=f"c{turn_counter['n']}",
                    ),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=AlternatingClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_turns=10,
        )
        orchestrator.run("Build an app")

        all_texts: list[str] = []
        for snapshot in captured_messages.values():
            all_texts.extend(_extract_text_blocks(snapshot))
        joined = "\n".join(all_texts)
        assert "<system-reminder>" not in joined, (
            "Reminder fired when modify_file alternated between two "
            "different paths. Texts: %r" % all_texts
        )
