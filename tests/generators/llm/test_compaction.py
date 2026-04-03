"""Tests for context compaction in the LLM orchestrator."""

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.orchestrator import (
    LLMOrchestrator, _estimate_tokens, COMPACT_TOKEN_THRESHOLD,
)
from besser.generators.llm.compaction import (
    _estimate_tokens as standalone_estimate_tokens,
    maybe_compact as standalone_maybe_compact,
    _summarize_messages as standalone_summarize_messages,
    COMPACT_TOKEN_THRESHOLD as STANDALONE_THRESHOLD,
    COMPACT_PRESERVE_RECENT as STANDALONE_PRESERVE_RECENT,
)


@pytest.fixture
def simple_model():
    StringType = PrimitiveDataType("str")
    cls = Class(name="Item")
    cls.attributes = {Property(name="name", type=StringType)}
    return DomainModel(name="Test", types={cls})


class TestTokenEstimation:

    def test_empty_messages(self):
        assert _estimate_tokens([]) == 0

    def test_string_content(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        est = _estimate_tokens(msgs)
        assert 90 <= est <= 110  # ~400/4 = 100

    def test_list_content(self):
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "content": "x" * 800}
        ]}]
        est = _estimate_tokens(msgs)
        assert 180 <= est <= 220  # ~800/4 = 200

    def test_scales_with_messages(self):
        small = [{"role": "user", "content": "hello"}]
        big = [{"role": "user", "content": "x" * 10000}] * 10
        assert _estimate_tokens(big) > _estimate_tokens(small) * 100


class TestCompaction:

    def _make_orchestrator(self, simple_model, tmp_path):
        from besser.generators.llm.llm_client import UsageTracker
        class MockClient:
            model = "mock"
            usage = UsageTracker("mock")
            def chat(self, **kw): return {"stop_reason": "end_turn", "content": []}
        return LLMOrchestrator(
            llm_client=MockClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

    def test_no_compaction_when_small(self, simple_model, tmp_path):
        orch = self._make_orchestrator(simple_model, tmp_path)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = orch._maybe_compact(messages)
        assert len(result) == len(messages)  # unchanged
        assert orch._compaction_count == 0

    def test_compaction_when_large(self, simple_model, tmp_path):
        orch = self._make_orchestrator(simple_model, tmp_path)
        # Create messages that exceed the token threshold
        big_content = "x" * (COMPACT_TOKEN_THRESHOLD * 5)  # Way over threshold
        messages = [
            {"role": "user", "content": "build something"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": big_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done step 1"}]},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok step 2"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": big_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done step 2"}]},
            {"role": "user", "content": "keep going"},
            {"role": "assistant", "content": [{"type": "text", "text": "step 3"}]},
        ]

        result = orch._maybe_compact(messages)
        assert len(result) < len(messages)  # compacted
        assert orch._compaction_count == 1
        # Recent messages preserved
        assert result[-1] == messages[-1]

        # Summary should mention earlier work
        summary_msg = result[0]
        assert "earlier" in summary_msg["content"].lower()

    def test_compaction_preserves_recent(self, simple_model, tmp_path):
        orch = self._make_orchestrator(simple_model, tmp_path)
        big = "x" * (COMPACT_TOKEN_THRESHOLD * 5)
        messages = []
        for i in range(20):
            messages.append({"role": "user", "content": f"turn {i}: {big[:1000]}"})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"ok {i}"}]})

        result = orch._maybe_compact(messages)
        # Last 6 messages should be preserved exactly
        assert result[-1] == messages[-1]
        assert result[-2] == messages[-2]

    def test_summarize_includes_tool_info(self, simple_model, tmp_path):
        orch = self._make_orchestrator(simple_model, tmp_path)
        orch.tool_calls_log = [
            {"tool": "generate_fastapi_backend", "turn": 1, "input": {}, "success": True},
            {"tool": "write_file", "turn": 2, "input": {"path": "auth.py"}, "success": True},
            {"tool": "modify_file", "turn": 3, "input": {"path": "main_api.py"}, "success": True},
            {"tool": "run_command", "turn": 4, "input": {"command": "python test.py"}, "success": True},
        ]
        messages = [{"role": "user", "content": "test"}] * 4
        summary = orch._summarize_messages(messages)
        assert "generate_fastapi_backend" in summary
        assert "write_file" in summary
        assert "modify_file" in summary


# ======================================================================
# Standalone module tests
# ======================================================================

class TestStandaloneCompaction:
    """Test the standalone compaction functions from compaction.py."""

    def test_standalone_estimate_tokens_matches_orchestrator(self):
        """Standalone _estimate_tokens should match the orchestrator import."""
        msgs = [{"role": "user", "content": "a" * 400}]
        assert standalone_estimate_tokens(msgs) == _estimate_tokens(msgs)

    def test_standalone_constants(self):
        """Constants are the same in both locations."""
        assert STANDALONE_THRESHOLD == COMPACT_TOKEN_THRESHOLD
        assert STANDALONE_PRESERVE_RECENT == 6

    def test_standalone_maybe_compact_no_compaction(self, tmp_path):
        """Standalone maybe_compact returns (messages, False) when small."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result, did_compact = standalone_maybe_compact(
            messages=messages,
            tool_calls_log=[],
            output_dir=str(tmp_path),
        )
        assert len(result) == len(messages)
        assert did_compact is False

    def test_standalone_maybe_compact_triggers(self, tmp_path):
        """Standalone maybe_compact returns (compacted, True) when large."""
        big_content = "x" * (STANDALONE_THRESHOLD * 5)
        messages = [
            {"role": "user", "content": "build something"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": big_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done step 1"}]},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok step 2"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": big_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done step 2"}]},
            {"role": "user", "content": "keep going"},
            {"role": "assistant", "content": [{"type": "text", "text": "step 3"}]},
        ]
        result, did_compact = standalone_maybe_compact(
            messages=messages,
            tool_calls_log=[],
            output_dir=str(tmp_path),
        )
        assert did_compact is True
        assert len(result) < len(messages)
        assert result[-1] == messages[-1]

    def test_standalone_summarize_messages(self, tmp_path):
        """Standalone _summarize_messages includes tool info."""
        tool_calls_log = [
            {"tool": "write_file", "turn": 1, "input": {}, "success": True},
            {"tool": "write_file", "turn": 2, "input": {}, "success": True},
        ]
        messages = [{"role": "user", "content": "test"}] * 2
        summary = standalone_summarize_messages(messages, tool_calls_log, str(tmp_path))
        assert "write_file(2x)" in summary

    def test_standalone_summarize_includes_files(self, tmp_path):
        """Standalone summarize lists files from output_dir."""
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "app.py"), "w") as f:
            f.write("x = 1")
        messages = [{"role": "user", "content": "test"}]
        summary = standalone_summarize_messages(messages, [], str(tmp_path))
        assert "app.py" in summary
