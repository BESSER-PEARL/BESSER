"""Tests for context compaction in the LLM orchestrator."""

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.pipeline.orchestrator import (
    LLMOrchestrator, _estimate_tokens, COMPACT_TOKEN_THRESHOLD,
)
from besser.spec_driven_agent.agent.compaction import (
    _compact_model_recap,
    _estimate_tokens as standalone_estimate_tokens,
    _tail_cut_index,
    maybe_compact as standalone_maybe_compact,
    _summarize_messages as standalone_summarize_messages,
    COMPACT_TOKEN_THRESHOLD as STANDALONE_THRESHOLD,
    COMPACT_MIN_PRESERVE_TOKENS,
    COMPACT_PRESERVE_TAIL_FRACTION,
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


def _default_tail_budget(threshold: int = STANDALONE_THRESHOLD) -> int:
    """The tail budget ``maybe_compact`` derives when none is passed."""
    return max(COMPACT_MIN_PRESERVE_TOKENS,
               int(threshold * COMPACT_PRESERVE_TAIL_FRACTION))


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
        import besser.spec_driven_agent.agent.compaction as c
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
        from besser.spec_driven_agent.providers.llm_client import UsageTracker
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
        # The tail is a share of the threshold, not a message count. It must
        # leave the majority of the budget for the summary to free, or
        # compaction frees nothing and fires again next turn.
        assert 0 < COMPACT_PRESERVE_TAIL_FRACTION <= 0.5
        assert COMPACT_MIN_PRESERVE_TOKENS >= 4_000

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
        from besser.spec_driven_agent.agent.compaction import effective_threshold
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
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        assert effective_threshold("mistral-large-latest") >= STANDALONE_THRESHOLD
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
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        reserve = 32_768          # the scaffolded/from-scratch output budget
        for tag in ("qwen3-coder:30b", "qwen3.8:27b"):
            threshold = effective_threshold(tag, reserve=reserve)
            assert threshold + reserve <= 60_000, tag
            assert threshold >= 8_000, tag

    def test_cloud_models_keep_the_full_threshold(self):
        """The local clamp must not leak onto cloud-served models."""
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        for tag in ("meituan/LongCat-2.0:free", "claude-sonnet-4-6",
                    "mistral-large-latest", "gpt-5.6-terra"):
            assert effective_threshold(tag, reserve=32_768) >= STANDALONE_THRESHOLD, tag

    def test_clamped_threshold_never_goes_below_a_workable_size(self):
        """Whatever the reserve, we never hand back a threshold that cannot
        hold a couple of tool results - that is the spiral condition."""
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        assert effective_threshold("devstral:24b", reserve=32_768) >= 8_000

    def test_effective_threshold_keeps_default_for_unknown_and_never_clamps_frontier(self):
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        assert effective_threshold("gpt-5-mini") == STANDALONE_THRESHOLD
        assert effective_threshold(None) == STANDALONE_THRESHOLD
        # Known 1M-class window: raised above the default, never clamped.
        assert effective_threshold("gpt-5.6-terra") > STANDALONE_THRESHOLD

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
        """A tool_result must never be the first surviving message.

        Under the TOKEN budget the boundary lands mid-pair far more often
        than it did under a fixed message count: the common shape is a
        ``write_file`` tool_use carrying a whole file body followed by a
        two-word ack, so the tail fits the ack and not the call. The cut
        then has to walk back onto the tool_use, or the provider rejects
        the request outright.
        """
        body = _bulk(_default_tail_budget() + 6_000)
        tool_use = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "write_file",
             "input": {"path": "api.py", "content": body}},
        ]}
        tool_result = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Wrote api.py"},
        ]}
        messages = [
            {"role": "user", "content": _bulk(STANDALONE_THRESHOLD)},
            {"role": "assistant", "content": [{"type": "text", "text": "planning"}]},
            {"role": "user", "content": "go"},
            tool_use,       # pair head — too big for the tail budget
            tool_result,    # pair tail — cheap, so the budget reaches it
            {"role": "assistant", "content": [{"type": "text", "text": "wrote it"}]},
        ]
        # Non-vacuous: the budget alone really does land on the tool_result.
        naive = _tail_cut_index(messages, _default_tail_budget())
        assert messages[naive] is tool_result

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        assert result[1] is tool_use          # walked back onto the pair head
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


class TestTokenBudgetTail:
    """The preserved tail is sized in tokens, not in messages.

    Six messages is the wrong unit: six ``write_file`` turns and six
    one-line turns differ by two orders of magnitude, so the old fixed
    count produced a tail that was either far bigger than the threshold it
    had to fit under, or a few hundred tokens of nothing.
    """

    def test_short_but_enormous_history_still_compacts(self, tmp_path):
        """A 6-message history way over threshold used to skip compaction.

        The old guard was ``len(messages) <= preserve_recent``, so six huge
        tool results went to the provider uncompacted every single turn.
        """
        big = _bulk(STANDALONE_THRESHOLD)
        messages = [
            {"role": "user", "content": "build the app"},
            {"role": "assistant", "content": [{"type": "text", "text": big}]},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": [{"type": "text", "text": big}]},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
        assert len(messages) == 6
        assert standalone_estimate_tokens(messages) > STANDALONE_THRESHOLD

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        assert standalone_estimate_tokens(result) < STANDALONE_THRESHOLD

    def test_fat_tail_is_cut_down_to_the_budget(self, tmp_path):
        """Six fat turns: the old tail was itself bigger than the threshold.

        Compaction that leaves the history above the threshold buys nothing
        - it fires again next turn, having paid a lossy summary for it.
        """
        fat = _bulk(20_000)
        messages = [{"role": "user", "content": "build"}]
        for i in range(8):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": fat}]}
            )
            messages.append({"role": "user", "content": f"next {i}"})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": fat}]})

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        # The last six messages alone are ~60k tokens; the budget is ~24k.
        assert standalone_estimate_tokens(messages[-6:]) > _default_tail_budget()
        assert standalone_estimate_tokens(result) < STANDALONE_THRESHOLD
        # The invariant: everything but the oldest kept message fits the
        # budget (that one is kept whatever it costs).
        tail = [m for m in result if m in messages]
        assert standalone_estimate_tokens(tail[1:]) <= _default_tail_budget()
        assert result[-1] == messages[-1]

    def test_thin_tail_keeps_far_more_than_six_messages(self, tmp_path):
        """The mirror case: cheap turns should not be thrown away.

        Twenty short turns cost a few hundred tokens together. Dropping all
        but six of them saves nothing and costs the model the thread of
        what it was doing.
        """
        messages = [{"role": "user", "content": _bulk(STANDALONE_THRESHOLD)}]
        for i in range(20):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": f"step {i}"}]}
            )
            messages.append({"role": "user", "content": f"ok {i}"})

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        preserved = [m for m in result if m in messages]
        assert len(preserved) > 6
        assert preserved == messages[1:]        # everything but the fat head

    def test_tail_keeps_at_least_the_last_message(self, tmp_path):
        """One message larger than the whole budget still survives."""
        messages = [
            {"role": "user", "content": _bulk(STANDALONE_THRESHOLD)},
            {"role": "assistant", "content": [{"type": "text", "text": "plan"}]},
            {"role": "user", "content": _bulk(_default_tail_budget() + 10_000)},
        ]
        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        assert result[-1] == messages[-1]

    def test_budget_scales_with_a_clamped_model_window(self, tmp_path):
        """A small-window model gets a proportionally smaller tail.

        A flat tail would be most of a clamped model's window, so
        compaction would free nothing on exactly the models that need it.
        """
        messages = [{"role": "user", "content": _bulk(40_000)}]
        for i in range(6):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": _bulk(3_000)}]}
            )
            messages.append({"role": "user", "content": f"go {i}"})

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
            model="devstral:24b",          # 32k window -> 16k threshold
        )
        assert did_compact is True
        tail = [m for m in result if m in messages]
        assert standalone_estimate_tokens(tail[1:]) <= _default_tail_budget(16_000)
        assert standalone_estimate_tokens(result) < 16_000

    def test_explicit_budget_overrides_the_derived_one(self, tmp_path):
        messages = [{"role": "user", "content": _bulk(STANDALONE_THRESHOLD)}]
        for i in range(10):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": _bulk(2_000)}]}
            )
            messages.append({"role": "user", "content": f"go {i}"})

        tight, _ = standalone_maybe_compact(
            messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
            preserve_tokens=2_500,
        )
        roomy, _ = standalone_maybe_compact(
            messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
            preserve_tokens=12_000,
        )
        assert standalone_estimate_tokens(tight) < standalone_estimate_tokens(roomy)

    def test_tail_cut_index_never_returns_an_empty_tail(self):
        messages = [{"role": "user", "content": "a" * 4_000}]
        assert _tail_cut_index(messages, 1) == 0
        assert _tail_cut_index([], 1_000) == 0


class TestWorkStateInSummary:
    """The summary must carry the work the end-of-run gate blocks on."""

    TASKS = [
        {"id": 1, "text": "Add DELETE /books/{isbn}", "done": False,
         "verify": lambda: True},
        {"id": 2, "text": "Wire the edit form to PUT", "done": False},
        {"id": 3, "text": "Seed the database", "done": False, "blocked": True,
         "blocked_reason": "its check still fails: no seed script on disk"},
        {"id": 4, "text": "Scaffold the backend", "done": True,
         "verification": "verified"},
        {"id": 5, "text": "Add nav links", "done": True,
         "verification": "unverified"},
        {"id": 6, "text": "Rewrite in Rust", "done": False,
         "dropped": "the user never asked for it"},
    ]
    WORK_STATE = {
        "tasks": TASKS,
        "blockers": ["missing module: app.services.booking",
                     "undefined name: Book in routers/books.py"],
        "contract_rules": ["Book.isbn is str in every layer, never parseInt it",
                           "id/created_at are server-owned, never in create forms"],
    }

    def test_open_items_survive_compaction(self, tmp_path):
        """This is the gap: the gate blocks on items the model cannot see."""
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state=self.WORK_STATE,
        )
        assert "OPEN 1" in summary
        assert "Add DELETE /books/{isbn}" in summary
        assert "OPEN 2" in summary
        assert "Wire the edit form to PUT" in summary
        assert "cannot finish while an item is open" in summary

    def test_verifier_carrying_items_are_flagged(self, tmp_path):
        """A verifier changes what 'done' costs - the model must know which
        items are machine-checked before it claims them."""
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state=self.WORK_STATE,
        )
        verifier_line = next(ln for ln in summary.splitlines() if "OPEN 1" in ln)
        plain_line = next(ln for ln in summary.splitlines() if "OPEN 2" in ln)
        assert "has verifier" in verifier_line
        assert "has verifier" not in plain_line

    def test_a_serialized_snapshot_can_flag_verifiers_too(self, tmp_path):
        """``task_snapshot()`` drops the live callable; ``has_verifier`` is
        the flag that shape carries instead."""
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state={"tasks": [
                {"id": 9, "text": "Checked item", "done": False, "has_verifier": True},
            ]},
        )
        assert "OPEN 9 [has verifier]: Checked item" in summary

    def test_blocked_done_and_dropped_are_distinguished(self, tmp_path):
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state=self.WORK_STATE,
        )
        assert "Checklist (2 open, 1 blocked, 2 done, 1 dropped)" in summary
        assert "BLOCKED 3: Seed the database" in summary
        assert "no seed script on disk" in summary
        assert "DONE (do not redo): 4, 5" in summary
        assert "1 of them unverified" in summary
        # A dropped item is counted, never re-listed as work.
        assert "Rewrite in Rust" not in summary

    def test_open_blockers_and_contract_rules_survive(self, tmp_path):
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state=self.WORK_STATE,
        )
        assert "Open blockers (2)" in summary
        assert "missing module: app.services.booking" in summary
        assert "Data contract" in summary
        assert "never parseInt it" in summary

    def test_blockers_accept_validation_issue_objects(self, tmp_path):
        """The orchestrator holds ``ValidationIssue`` objects, not strings."""
        issue = type("ValidationIssue", (), {"message": "syntax error: main.py:42"})()
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state={"blockers": [issue, {"message": "mapper config: Book"}]},
        )
        assert "syntax error: main.py:42" in summary
        assert "mapper config: Book" in summary

    def test_contract_rules_accept_one_block_of_text(self, tmp_path):
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state={"contract_rules": "Ids are the model types\n\nNo fake success"},
        )
        assert "Ids are the model types" in summary
        assert "No fake success" in summary

    def test_long_lists_are_capped_not_dumped(self, tmp_path):
        """The summary competes for context; it must not become the dump."""
        work_state = {
            "tasks": [{"id": i, "text": f"task number {i}", "done": False}
                      for i in range(1, 51)],
            "blockers": [f"blocker number {i}" for i in range(40)],
            "contract_rules": [f"rule number {i}" for i in range(40)],
        }
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state=work_state,
        )
        assert "Checklist (50 open" in summary
        assert summary.count("OPEN ") <= 12
        assert "more" in summary
        assert standalone_estimate_tokens(
            [{"role": "user", "content": summary}]
        ) < 2_000

    def test_a_single_item_is_not_truncated_to_uselessness(self, tmp_path):
        long_text = "Implement the booking confirmation email " * 20
        summary = standalone_summarize_messages(
            [{"role": "user", "content": "x"}], [], str(tmp_path),
            work_state={"tasks": [{"id": 1, "text": long_text, "done": False}]},
        )
        line = next(ln for ln in summary.splitlines() if "OPEN 1" in ln)
        assert "Implement the booking confirmation email" in line
        assert len(line) < 260

    def test_maybe_compact_threads_work_state_through(self, tmp_path):
        messages = [{"role": "user", "content": _bulk(STANDALONE_THRESHOLD)}]
        for i in range(6):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": f"step {i}"}]}
            )
            messages.append({"role": "user", "content": f"ok {i}"})

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
            work_state=self.WORK_STATE,
        )
        assert did_compact is True
        assert "OPEN 1" in result[0]["content"]
        assert "missing module: app.services.booking" in result[0]["content"]


class TestWorkStateIsOptional:
    """Nothing changes until the call site is wired."""

    # Byte-for-byte what the summary produced before the work-state section
    # existed. Pinned so a later edit cannot quietly reshape the part of the
    # summary an unwired call site still gets.
    GOLDEN = (
        "Earlier: 3 messages\n"
        "Tools: read_file(1x), write_file(1x)\n"
        "Files you already WROTE or MODIFIED (your edits are on disk — "
        "re-read before editing again, never rewrite blindly): app.py\n"
        "Files you already read: b.py\n"
        "Files: app.py"
    )

    def _fixture(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1")
        return (
            [{"role": "user", "content": "x"}] * 3,
            [{"tool": "write_file", "input": {"path": "app.py"}},
             {"tool": "read_file", "input": {"path": "b.py"}}],
            str(tmp_path),
        )

    def test_summary_without_work_state_is_unchanged(self, tmp_path):
        messages, log, out = self._fixture(tmp_path)
        assert standalone_summarize_messages(messages, log, out) == self.GOLDEN

    def test_explicit_none_matches_the_omitted_argument(self, tmp_path):
        messages, log, out = self._fixture(tmp_path)
        assert (standalone_summarize_messages(messages, log, out, work_state=None)
                == standalone_summarize_messages(messages, log, out))

    @pytest.mark.parametrize(
        "work_state",
        [None, {}, {"tasks": []}, {"tasks": None}, {"unknown_key": "x"},
         [], "tasks", 0],
    )
    def test_empty_or_malformed_state_adds_nothing(self, tmp_path, work_state):
        """A wrong-shaped argument must degrade to today's summary, never
        raise: this runs inside the generation loop."""
        messages, log, out = self._fixture(tmp_path)
        assert (standalone_summarize_messages(messages, log, out, work_state=work_state)
                == self.GOLDEN)

    def test_maybe_compact_without_work_state_adds_no_section(self, tmp_path):
        messages = [{"role": "user", "content": _bulk(STANDALONE_THRESHOLD)}]
        for i in range(6):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": f"step {i}"}]}
            )
            messages.append({"role": "user", "content": f"ok {i}"})

        result, did_compact = standalone_maybe_compact(
            messages=messages, tool_calls_log=[], output_dir=str(tmp_path),
        )
        assert did_compact is True
        assert "Work state" not in result[0]["content"]


def test_compaction_makes_no_llm_call(tmp_path, monkeypatch):
    """The summary is deterministic. OpenCode summarizes with a model; here
    the target is often a weak local Qwen3-30B-A3B, and asking it to
    summarize its own transcript at the moment its context is failing is a
    reliability liability. Deliberate - do not 'improve' it.
    """
    import besser.spec_driven_agent.agent.compaction as c

    def explode(*args, **kwargs):     # any outbound HTTP at all
        raise AssertionError("compaction must not call out to a model")

    monkeypatch.setattr(c.urllib.request, "urlopen", explode)
    messages = [{"role": "user", "content": _bulk(STANDALONE_THRESHOLD)}]
    for i in range(6):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"s{i}"}]})
        messages.append({"role": "user", "content": f"ok {i}"})

    first, _ = standalone_maybe_compact(
        messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
        work_state=TestWorkStateInSummary.WORK_STATE,
    )
    second, _ = standalone_maybe_compact(
        messages=list(messages), tool_calls_log=[], output_dir=str(tmp_path),
        work_state=TestWorkStateInSummary.WORK_STATE,
    )
    assert first[0]["content"] == second[0]["content"]
