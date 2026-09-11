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
    _compact_model_recap,
    _estimate_tokens as standalone_estimate_tokens,
    maybe_compact as standalone_maybe_compact,
    _summarize_messages as standalone_summarize_messages,
    COMPACT_TOKEN_THRESHOLD as STANDALONE_THRESHOLD,
    COMPACT_PRESERVE_RECENT as STANDALONE_PRESERVE_RECENT,
)


def test_compaction_recap_keeps_bpmn_and_nn_identity():
    bpmn_model = type("BPMN", (), {"name": "Checkout", "processes": [object()]})()
    nn_model = type("NN", (), {"name": "Fraud", "modules": [object(), object()]})()

    recap = _compact_model_recap(bpmn_model=bpmn_model, nn_model=nn_model)

    assert "BPMN Checkout (1 processes)" in recap
    assert "Neural network Fraud (2 modules)" in recap


@pytest.fixture
def simple_model():
    StringType = PrimitiveDataType("str")
    cls = Class(name="Item")
    cls.attributes = {Property(name="name", type=StringType)}
    return DomainModel(name="Test", types={cls})


def _bulk(target_tokens: int) -> str:
    """Code-like text whose ESTIMATED size exceeds ``target_tokens``.

    Deliberately varied: a run of one repeated character (the old ``"x" * N``)
    is only "big" under the chars/4 heuristic - a real BPE tokenizer merges it
    into almost nothing, so such a payload silently stops exercising any
    threshold once an accurate counter is installed. Measuring with
    ``_estimate_tokens`` itself keeps these tests correct under either
    estimator.
    """
    lines: list[str] = []
    while True:
        i = len(lines)
        lines.append(
            f"    value_{i} = compute_total(items_{i}, offset={i})  "
            f"# step {i} of the pipeline, validated against schema_{i}"
        )
        if i % 64 == 0 and _estimate_tokens(
            [{"role": "user", "content": "\n".join(lines)}]
        ) > target_tokens:
            return "\n".join(lines)


class TestTokenEstimation:

    def test_empty_messages(self):
        assert _estimate_tokens([]) == 0

    def test_string_content(self):
        """Estimate tracks content size. The exact number is tokenizer-
        dependent (tiktoken when installed, chars/4 otherwise), so assert the
        PROPERTY - monotonic and in a sane band - not a heuristic constant."""
        text = "def compute(total, items):\n    return total + sum(items)\n" * 8
        est = _estimate_tokens([{"role": "user", "content": text}])
        assert 0 < est <= len(text)          # never more than one token/char
        assert est >= len(text) // 12        # never absurdly small for code
        longer = _estimate_tokens([{"role": "user", "content": text * 3}])
        assert longer > est * 2

    def test_chars_per_four_fallback_when_tiktoken_is_absent(self, monkeypatch):
        """The fallback must still work on a host without tiktoken - it is a
        declared dependency now, but compaction must never hard-fail on it."""
        import besser.generators.llm.compaction as c
        monkeypatch.setattr(c, "_TOKENIZER", None)
        monkeypatch.setattr(c, "_TOKENIZER_LOADED", True)
        est = _estimate_tokens([{"role": "user", "content": "a" * 400}])
        assert est > 0

    def test_list_content(self):
        """tool_result bodies are counted. Property-based for the same reason
        as test_string_content - the exact count is tokenizer-dependent."""
        body = "SELECT id, name FROM patients WHERE dept_id = :dept;\n" * 12
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "content": body}
        ]}]
        est = _estimate_tokens(msgs)
        assert 0 < est <= len(body)
        assert est >= len(body) // 12

    def test_tool_use_input_is_counted(self):
        """A dict-shaped tool_use block must not count as ZERO tokens -
        write_file arguments are the largest thing the model emits."""
        msgs = [{"role": "assistant", "content": [
            {"type": "tool_use", "name": "write_file",
             "input": {"path": "app.py", "content": "x = 1\n" * 200}},
        ]}]
        assert _estimate_tokens(msgs) > 100

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


class TestHarnessUpgrades:
    """Headroom threshold, safe cut boundaries, file-op memory."""

    def test_effective_threshold_clamps_genuinely_small_models(self):
        from besser.generators.llm.compaction import effective_threshold
        # Self-hosted on the LIST ollama box at a server-configured 32k.
        assert effective_threshold("devstral:24b") == 16_000
        assert effective_threshold("mistral-small-latest") == 16_000

    def test_effective_threshold_does_not_clamp_on_a_misattribution(self):
        """Regression for the read/compact/re-read spiral of 2026-09-10.

        A window stated too LOW is far worse than one left unknown: it made
        every few file reads trigger a lossy compaction, the model re-read what
        it lost, and a run burned 40 turns of read_file until the runtime cap.

        The rule is NOT "never clamp" - it is "clamp only on evidence from the
        deployment we actually call". Two corrections, both from 2026-09-11:

        - A bare "mistral" marker matched ``mistral-large-latest`` (256k),
          clamping a frontier cloud model to a 16k threshold. Still wrong; this
          test pins that it stays unclamped.
        - The qwen row WAS justified, just with a stale number. The model is
          served from our own Ollama box (ollama.besser-pearl.org, NOT Command
          Code as first concluded), and measurement there showed prefill is the
          binding cost and >64k prompts get truncated. It is clamped again, now
          at 60k - see test_self_hosted_qwen_is_clamped_to_fit_its_prefill_budget.
        """
        from besser.generators.llm.compaction import effective_threshold
        assert effective_threshold("mistral-large-latest") == STANDALONE_THRESHOLD
        # Evidence-based clamp, not a guess: strictly tighter than the default,
        # and never so tight that a couple of file reads trip compaction.
        qwen = effective_threshold("qwen3-coder:30b", reserve=32_768)
        assert 8_000 <= qwen < STANDALONE_THRESHOLD

    def test_self_hosted_qwen_is_clamped_to_fit_its_prefill_budget(self):
        """The local Ollama box advertises 131k but cannot usefully serve it.

        Measured 2026-09-11: prefill runs ~950 tok/s, so context size translates
        directly into per-turn latency, and a >64k prompt came back truncated.
        The clamp must keep history + max output inside 60k.
        """
        from besser.generators.llm.compaction import effective_threshold
        reserve = 32_768          # the scaffolded/from-scratch output budget
        for tag in ("qwen3-coder:30b", "qwen3.8:27b"):
            threshold = effective_threshold(tag, reserve=reserve)
            assert threshold + reserve <= 60_000, tag
            assert threshold >= 8_000, tag

    def test_cloud_models_keep_the_full_threshold(self):
        """The local clamp must not leak onto cloud-served models."""
        from besser.generators.llm.compaction import effective_threshold
        for tag in ("meituan/LongCat-2.0:free", "claude-sonnet-4-6",
                    "mistral-large-latest", "gpt-5.6-terra"):
            assert effective_threshold(tag, reserve=32_768) == STANDALONE_THRESHOLD, tag

    def test_clamped_threshold_never_goes_below_a_workable_size(self):
        """Whatever the reserve, we never hand back a threshold that cannot
        hold a couple of tool results - that is the spiral condition."""
        from besser.generators.llm.compaction import effective_threshold
        assert effective_threshold("devstral:24b", reserve=32_768) >= 8_000

    def test_effective_threshold_keeps_default_for_frontier_and_unknown(self):
        from besser.generators.llm.compaction import effective_threshold
        assert effective_threshold("gpt-5-mini") == STANDALONE_THRESHOLD
        assert effective_threshold(None) == STANDALONE_THRESHOLD
        assert effective_threshold("gpt-5.6-terra") == STANDALONE_THRESHOLD

    def test_small_model_compacts_earlier(self, tmp_path):
        """~20k tokens: under the 80k default, over qwen's clamped 16k."""
        # ~5k tokens each x 4 messages = ~20k total: comfortably under the
        # 80k default, comfortably over a genuinely-clamped 16k model.
        big = _bulk(5_000)
        messages = [{"role": "user", "content": "build"}]
        for _ in range(4):
            messages.append({"role": "assistant", "content": [{"type": "text", "text": big}]})
            messages.append({"role": "user", "content": "go on"})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})

        _, unclamped = standalone_maybe_compact(
            messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert unclamped is False
        _, clamped = standalone_maybe_compact(
            messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
            model="devstral:24b",   # still genuinely 32k-windowed
        )
        assert clamped is True

    def test_cut_never_orphans_tool_results(self, tmp_path):
        """When the naive cut would open the preserved tail on a
        tool_result message, the boundary walks back to include the
        paired assistant tool_use — and skips the synthetic assistant
        turn so roles still alternate."""
        big = _bulk(STANDALONE_THRESHOLD + 5_000)
        tool_use = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
        ]}
        tool_result = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "data"},
        ]}
        messages = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": [{"type": "text", "text": "planning"}]},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "step"}]},
            tool_use,       # pair head — naive cut (preserve_recent=6) starts BELOW this
            tool_result,    # pair tail
            {"role": "assistant", "content": [{"type": "text", "text": "more"}]},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "more"}]},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        # Every preserved tool_result must still be preceded by its tool_use.
        for i, msg in enumerate(result):
            if isinstance(msg.get("content"), list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in msg["content"]
            ):
                prev = result[i - 1]
                assert prev["role"] == "assistant"
                assert any(
                    isinstance(b, dict) and b.get("type") == "tool_use"
                    for b in prev["content"]
                )
        # Roles must alternate (no double-assistant from the synthetic turn).
        for i in range(1, len(result)):
            assert result[i]["role"] != result[i - 1]["role"]

    def test_summary_remembers_written_vs_read_files(self, tmp_path):
        tool_calls_log = [
            {"tool": "read_file", "turn": 1, "input": {"path": "a.py"}, "success": True},
            {"tool": "read_file", "turn": 2, "input": {"path": "b.py"}, "success": True},
            {"tool": "modify_file", "turn": 3, "input": {"path": "b.py"}, "success": True},
            {"tool": "write_file", "turn": 4, "input": {"path": "c.py"}, "success": True},
            {"tool": "write_file", "turn": 5, "input": {"path": "junk.py"}, "success": True},
            {"tool": "delete_file", "turn": 6, "input": {"path": "junk.py"}, "success": True},
        ]
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], tool_calls_log, str(tmp_path),
        )
        assert "WROTE or MODIFIED" in summary
        assert "b.py, c.py" in summary          # written, sorted; b.py not double-listed as read
        assert "Files you already read: a.py" in summary
        assert "junk.py" not in summary          # deleted paths drop out
