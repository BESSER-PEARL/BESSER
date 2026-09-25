"""Tests for the per-file modify_file streak guard.

When the LLM makes N consecutive ``modify_file`` calls on the same path
that ALL fail to match, the orchestrator must inject a high-salience
reminder into the conversation BEFORE the next LLM call, telling it to
read the file and copy ``old_text`` verbatim - never to rewrite the file
from memory. N successful edits to one file are healthy (five methods in
one router file) and must not trigger it: a live run
showed the old "call write_file" reminder firing on exactly that and
turning targeted edits into whole-file rewrites of scaffold code.

This guard is separate from the legacy ``_is_stuck`` heuristic, which
fires on N uniform tool calls regardless of arguments and only injects
a low-salience warning into a tool_result.
"""

import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


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


def _streak_turn4_text(simple_model, tmp_path, target: str, old_text) -> str:
    """Run three modify_file turns on ``target`` (``old_text(n)`` is the
    text quoted on turn n) and return the user-visible text the model is
    shown on turn 4 - where a streak reminder, if any, has been appended."""
    _seed_file(str(tmp_path), target, content="line1\nline2\nline3\n")
    turn_counter = {"n": 0}
    captured: dict[str, list[dict]] = {}

    class StreakClient:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, system, messages, tools):
            turn_counter["n"] += 1
            if turn_counter["n"] == 4:
                captured["turn4"] = [dict(m) for m in messages]
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Stopping."),
                ]}
            return {"stop_reason": "tool_use", "content": [
                MockBlock(
                    "tool_use",
                    name="modify_file",
                    input={
                        "path": target,
                        "old_text": old_text(turn_counter["n"]),
                        "new_text": f"updated{turn_counter['n']}",
                    },
                    id=f"c{turn_counter['n']}",
                ),
            ]}

    LLMOrchestrator(
        llm_client=StreakClient(),
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
    ).run("Build an app")
    assert "turn4" in captured, (
        "Mock client never observed a 4th turn — the orchestrator ended Phase 2 early."
    )
    return "\n".join(_extract_text_blocks(captured["turn4"]))


class TestPerFileModifyGuard:

    def test_syntax_refusals_are_not_reported_as_failed_text_matches(self, simple_model, tmp_path):
        source = "def action():\n    try:\n        return True\n    except ValueError:\n        return False\n"
        (tmp_path / "action.py").write_text(source, encoding="utf-8")
        orch = LLMOrchestrator(llm_client=type("Client", (), {"model": "mock-model", "usage": UsageTracker("mock-model")})(),
                               domain_model=simple_model, output_dir=str(tmp_path))
        for number in range(3):
            block = MockBlock("tool_use", name="modify_file", id=f"bad{number}", input={
                "path": "action.py", "old_text": "        return True",
                "new_text": f"    return {number}",
            })
            orch._execute_tool_blocks([block], number)
        assert orch._consecutive_modify_on_same_file() == "action.py"
        reminder = orch._build_modify_loop_reminder("action.py")
        assert "syntax guard" in reminder and "NOT applied" in reminder
        assert "CURRENT ON-DISK" in reminder and "try/except" in reminder
        assert "all failed to match" not in reminder and "write_file" not in reminder
        assert (tmp_path / "action.py").read_text(encoding="utf-8") == source

    def test_three_successful_edits_to_one_file_do_not_trigger(self, simple_model, tmp_path):
        """Three edits that each match and apply are ordinary work on one
        file, not a flail. Observed live: the reminder fired here and
        ordered a whole-file rewrite of a router the model was editing
        method by method."""
        joined = _streak_turn4_text(
            simple_model, tmp_path, "app.py", old_text=lambda n: f"line{n}",
        )
        assert "<system-reminder>" not in joined, joined

    def test_three_misses_on_one_file_trigger_a_read_and_copy_reminder(
        self, simple_model, tmp_path,
    ):
        """N=3 consecutive modify_file calls on the SAME path that all
        fail to match inject a system-style reminder before turn 4: name
        the path, tell the model to read the file and copy old_text
        verbatim, and never suggest rewriting it from memory."""
        target = "app.py"
        joined = _streak_turn4_text(
            simple_model, tmp_path, target, old_text=lambda n: f"nope{n}",
        )
        # The reminder is tagged with a system-reminder marker so the
        # LLM treats it as a meta-instruction, not user content.
        assert "<system-reminder>" in joined, joined
        assert target in joined
        assert "read_file" in joined and "replace_file_lines" in joined and "read_id" in joined, joined
        # Never points the model at a rewrite - that is what lost scaffold code.
        assert "write_file" not in joined, joined
        assert "complete new contents" not in joined, joined

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


# ======================================================================
# Interleaved re-reads: the shape the guard could not see
# ======================================================================
#
# A recorded run (Qwen3-30B). Phase 2 alternated
# modify_file and read_file on ONE file for 38 consecutive pairs:
#
#     t9  modify_file web_app/backend/sql_alchemy.py
#     t10 read_file   web_app/backend/sql_alchemy.py
#     t11 modify_file web_app/backend/sql_alchemy.py
#     ... 38 pairs ... 85 turns, 402s, $0.70, no progress
#
# `_is_stuck` DID fire ("Possible loop: modify_file" in the container log,
# repeatedly) but it only appends "Move on." inside a tool_result, which
# this model class ignores — test_elided_edit.py records the same note
# being ignored 61 times. The guard that uses the channel that works,
# `_consecutive_modify_on_same_file`, could not fire at all: it rejected
# any window containing a non-modify tool, and the interleaved read_file
# is exactly that. The flail's own rhythm disarmed the guard built for it.


def _interleaved_turn_text(simple_model, tmp_path, target: str, old_text,
                           stop_turn: int = 6) -> str:
    """Alternate modify_file / read_file on ``target`` and return the text
    the model is shown on ``stop_turn``."""
    _seed_file(str(tmp_path), target, content="line1\nline2\nline3\n")
    turn_counter = {"n": 0}
    captured: dict[str, list[dict]] = {}

    class InterleavedClient:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, system, messages, tools):
            turn_counter["n"] += 1
            n = turn_counter["n"]
            if n == stop_turn:
                captured["stop"] = [dict(m) for m in messages]
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Stopping."),
                ]}
            if n % 2 == 1:
                return {"stop_reason": "tool_use", "content": [
                    MockBlock("tool_use", name="modify_file", input={
                        "path": target,
                        "old_text": old_text(n),
                        "new_text": f"updated{n}",
                    }, id=f"m{n}"),
                ]}
            return {"stop_reason": "tool_use", "content": [
                MockBlock("tool_use", name="read_file",
                          input={"path": target}, id=f"r{n}"),
            ]}

    LLMOrchestrator(
        llm_client=InterleavedClient(),
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=12,
    ).run("Build an app")
    assert "stop" in captured, (
        "Mock client never reached the stop turn — Phase 2 ended early."
    )
    return "\n".join(_extract_text_blocks(captured["stop"]))


class TestInterleavedReadDoesNotDisarmTheGuard:

    def test_modify_read_modify_read_on_one_file_still_triggers(
        self, simple_model, tmp_path
    ):
        """The a5dce952 shape: every modify misses, each followed by a
        re-read of the same file. Re-reading the file you cannot edit is
        part of the flail, not progress that clears it."""
        joined = _interleaved_turn_text(
            simple_model, tmp_path, "app.py",
            old_text=lambda n: f"nomatch{n}",
        )
        assert "<system-reminder>" in joined, joined
        assert "app.py" in joined

    def test_interleaved_reads_with_successful_edits_do_not_trigger(
        self, simple_model, tmp_path
    ):
        """Edit, re-read to confirm, edit again is a healthy rhythm. The
        executor resets its miss count on a hit, so a matching edit must
        never be mistaken for the flail above."""
        joined = _interleaved_turn_text(
            simple_model, tmp_path, "app.py",
            old_text=lambda n: f"line{(n + 1) // 2}",
        )
        assert "<system-reminder>" not in joined, joined

    def test_a_read_of_a_different_file_still_breaks_the_streak(
        self, simple_model, tmp_path
    ):
        """Reading a DIFFERENT file is real exploration — the model went
        to look somewhere else — so it must still clear the streak."""
        _seed_file(str(tmp_path), "app.py", content="line1\nline2\nline3\n")
        _seed_file(str(tmp_path), "other.py", content="elsewhere\n")
        turn_counter = {"n": 0}
        captured: dict[str, list[dict]] = {}

        class MixedClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                n = turn_counter["n"]
                if n == 6:
                    captured["stop"] = [dict(m) for m in messages]
                    return {"stop_reason": "end_turn", "content": [
                        MockBlock("text", text="Stopping."),
                    ]}
                if n % 2 == 1:
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="modify_file", input={
                            "path": "app.py", "old_text": f"nomatch{n}",
                            "new_text": "x",
                        }, id=f"m{n}"),
                    ]}
                return {"stop_reason": "tool_use", "content": [
                    MockBlock("tool_use", name="read_file",
                              input={"path": "other.py"}, id=f"r{n}"),
                ]}

        LLMOrchestrator(
            llm_client=MixedClient(), domain_model=simple_model,
            output_dir=str(tmp_path), max_turns=12,
        ).run("Build an app")
        joined = "\n".join(_extract_text_blocks(captured["stop"]))
        assert "<system-reminder>" not in joined, joined
