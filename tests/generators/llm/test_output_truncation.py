"""Output-token ceiling and truncation recovery.

Regression tests for a live failure on 2026-09-10: a scaffolded run (the most
common path — deterministic generator, then LLM customisation) asked for a
React frontend, overran the 16_384 output ceiling on its FIRST customisation
turn, and Phase 2 exited with zero LLM writes.

Two defects, both fixed here:

1. ``_apply_adaptive_budget`` returned early for any scaffolded run, so the
   scaffolded-customise path kept the client default of 16_384 while pure
   from-scratch and modify runs both got FROM_SCRATCH_MAX_TOKENS.
2. A ``max_tokens``/``length`` stop_reason ended Phase 2 immediately. It is
   recoverable — the model can emit a smaller turn — so it is now fed back and
   retried, bounded by ``_MAX_TRUNCATION_RETRIES``.
"""

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
import besser.generators.llm.orchestrator as orchestrator_module
from besser.generators.llm.llm_client import FROM_SCRATCH_MAX_TOKENS
from besser.generators.llm.orchestrator import LLMOrchestrator


class _Block:
    def __init__(self, block_type, **kw):
        self.type = block_type
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 1, "cost_usd": 0.0}


class _ScriptedClient:
    """Returns scripted responses, then end_turn forever."""

    def __init__(self, responses):
        self.model = "test-model"
        self.usage = _Usage()
        self.max_tokens = 16_384
        self._responses = list(responses)
        self.chat_calls = 0
        self.seen_nudges: list[str] = []

    def chat(self, system=None, messages=None, tools=None, **kw):
        self.chat_calls += 1
        # Capture any standalone user-text nudge the loop injected.
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        self.seen_nudges.append(b["text"])
            break
        if self._responses:
            return self._responses.pop(0)
        return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _truncated():
    return {"stop_reason": "max_tokens", "content": [_Block("text", text="partial")]}


def _end_turn():
    return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _domain() -> DomainModel:
    s = PrimitiveDataType("str")
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=s, is_id=True)}
    return DomainModel(name="Library", types={book})


def _orch(tmp_path, client) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client,
        domain_model=_domain(),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )


@pytest.fixture(autouse=True)
def _no_gap_analysis(monkeypatch):
    """Keep the gate inert so these tests isolate truncation behaviour."""
    monkeypatch.setattr(
        orchestrator_module, "analyze_gaps_via_llm", lambda **kw: None
    )


# ----------------------------------------------------------------------
# 1. the output ceiling
# ----------------------------------------------------------------------


def test_scaffolded_run_gets_the_same_output_ceiling_as_from_scratch(tmp_path):
    """The live failure: a scaffolded customise turn writes whole new files
    (React pages, auth modules) exactly like a from-scratch run, so capping it
    at half the budget had no basis."""
    client = _ScriptedClient([])
    orch = _orch(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"   # scaffolded path

    orch._apply_adaptive_budget()

    assert client.max_tokens == FROM_SCRATCH_MAX_TOKENS
    assert orch._adaptive_budget_applied is True


def test_from_scratch_run_still_gets_the_ceiling(tmp_path):
    client = _ScriptedClient([])
    orch = _orch(tmp_path, client)
    orch._generator_used = None

    orch._apply_adaptive_budget()

    assert client.max_tokens == FROM_SCRATCH_MAX_TOKENS


def test_an_already_high_ceiling_is_not_lowered(tmp_path):
    client = _ScriptedClient([])
    client.max_tokens = FROM_SCRATCH_MAX_TOKENS * 2
    orch = _orch(tmp_path, client)
    orch._generator_used = "generate_django"

    orch._apply_adaptive_budget()

    assert client.max_tokens == FROM_SCRATCH_MAX_TOKENS * 2
    assert orch._adaptive_budget_applied is False


def test_cost_and_runtime_rails_are_never_touched(tmp_path):
    """Only the per-call output limit moves. Cost and runtime are
    user-authorised safety rails."""
    client = _ScriptedClient([])
    orch = _orch(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    cost, runtime = orch.max_cost_usd, orch.max_runtime_seconds

    orch._apply_adaptive_budget()

    assert (orch.max_cost_usd, orch.max_runtime_seconds) == (cost, runtime)


# ----------------------------------------------------------------------
# 2. truncation is recoverable, but bounded
# ----------------------------------------------------------------------


def test_one_truncation_does_not_end_the_run(tmp_path):
    """Previously this ended Phase 2 with zero writes on the FIRST truncation."""
    client = _ScriptedClient([_truncated(), _end_turn()])
    orch = _orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    assert client.chat_calls == 2, "loop stopped instead of retrying"
    assert orch._truncation_retries == 1
    assert orch._phase2_stop_reason == "completed"


def test_the_model_is_told_what_happened_and_what_to_do(tmp_path):
    client = _ScriptedClient([_truncated(), _end_turn()])
    orch = _orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    nudge = "\n".join(client.seen_nudges)
    assert "CUT OFF" in nudge
    assert "NOTHING was written" in nudge      # models assume partial writes
    assert "SMALLER turn" in nudge


def test_repeated_truncation_eventually_stops_the_run(tmp_path):
    """A model that keeps truncating is not adapting — stop rather than spend
    the cost cap rediscovering that."""
    client = _ScriptedClient([_truncated()] * 6)
    orch = _orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    assert orch._truncation_retries == orch._MAX_TRUNCATION_RETRIES
    assert orch._phase2_stop_reason == "api_error"
    assert "output token limit" in (orch._phase2_api_error or "")
    # One call per retry plus the final, give-up call.
    assert client.chat_calls == orch._MAX_TRUNCATION_RETRIES + 1


def test_retry_budget_is_a_per_run_total(tmp_path):
    """Deliberately NOT reset by a good turn: an unbounded allowance would let
    a pathological model truncate every other turn until the cost cap."""
    client = _ScriptedClient([
        _truncated(), _end_turn(), _truncated(), _truncated(), _truncated(),
    ])
    orch = _orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    assert orch._truncation_retries <= orch._MAX_TRUNCATION_RETRIES
