"""Phase 3 must act on a blocker, and must not spin when it cannot.

Run 15a8ac7d (2026-09-18) shipped an app that failed to import. Phase 3
detected the single blocker correctly and ran two fix attempts — each burning
its full 5 turns with ZERO tool calls, no cost or runtime cap hit. Neither
branch can produce that: ``end_turn`` returns after one turn and ``tool_use``
produces calls, so a third stop_reason fell through and the loop re-sent an
identical request five times.

Measured against the model that failed there (Qwen3-30B-A3B-Instruct, 12
trials): given only "file line N" it spends a turn on ``read_file`` (6/6);
given the offending lines it calls ``modify_file`` immediately (6/6).
"""
import os
import tempfile

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue

BLOCKER = (
    "Syntax error in web_app/backend/routers/booking_methods.py line 37: "
    "expected 'except' or 'finally' block"
)


@pytest.fixture
def orchestrator(tmp_path):
    target = tmp_path / "web_app" / "backend" / "routers"
    target.mkdir(parents=True)
    (target / "booking_methods.py").write_text(
        "\n".join(f"code line {i}" for i in range(1, 60)), encoding="utf-8")
    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.output_dir = str(tmp_path)
    return orch


def test_excerpt_is_produced_for_a_file_and_line(orchestrator):
    got = orchestrator._excerpts_for([ValidationIssue("blocker", BLOCKER)])
    assert len(got) == 1
    assert "booking_methods.py" in got[0]


def test_excerpt_is_centred_on_the_error_line(orchestrator):
    body = orchestrator._excerpts_for([ValidationIssue("blocker", BLOCKER)])[0]
    assert "   37| code line 37" in body


def test_excerpt_is_bounded_not_the_whole_file(orchestrator):
    """SWE-agent scores a bounded window above a full file (18.0 vs 12.7)."""
    body = orchestrator._excerpts_for([ValidationIssue("blocker", BLOCKER)])[0]
    numbered = [l for l in body.splitlines() if "| code line" in l]
    assert len(numbered) <= 12, "window must stay small"
    assert "code line 1\n" not in body


def test_lines_are_numbered_so_old_text_can_be_copied(orchestrator):
    body = orchestrator._excerpts_for([ValidationIssue("blocker", BLOCKER)])[0]
    assert all("|" in l for l in body.splitlines() if "code line" in l)


def test_a_blocker_without_a_file_yields_no_excerpt(orchestrator):
    assert orchestrator._excerpts_for(
        [ValidationIssue("blocker", "dependency conflict in requirements.txt")]) == []


def test_a_missing_file_is_skipped_silently(orchestrator):
    issue = ValidationIssue("blocker", "Syntax error in web_app/gone.py line 3: bad")
    assert orchestrator._excerpts_for([issue]) == []


def test_excerpt_count_is_capped(orchestrator, tmp_path):
    """A 40-blocker report must not paste 40 windows into the prompt."""
    issues = [ValidationIssue("blocker",
              f"Syntax error in web_app/backend/routers/booking_methods.py line {n}: bad")
              for n in range(10, 40)]
    assert len(orchestrator._excerpts_for(issues)) <= 3


def test_duplicate_file_and_line_is_not_repeated(orchestrator):
    issues = [ValidationIssue("blocker", BLOCKER), ValidationIssue("blocker", BLOCKER)]
    assert len(orchestrator._excerpts_for(issues)) == 1


def test_fix_loop_stops_on_an_unexpected_stop_reason(tmp_path):
    """The defect: any stop_reason other than end_turn / tool_use used to fall
    through, leaving `messages` untouched so the same request went out again."""
    class _MaxTokensClient:
        model = "mock-model"
        usage = type("Usage", (), {"estimated_cost": 0.0})()
        calls = 0

        def chat(self, system, messages, tools, **kwargs):
            self.calls += 1
            return {"stop_reason": "max_tokens", "content": []}

    client = _MaxTokensClient()
    orch = LLMOrchestrator(
        llm_client=client, state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path), max_cost_usd=10.0,
        enable_tracing=False, enable_checkpointing=False,
    )
    orch._invoke_phase3_fix_loop([ValidationIssue("blocker", BLOCKER)], is_first_attempt=True)
    assert client.calls == 1


def test_fix_prompt_tells_the_model_not_to_abbreviate():
    """Scans the class AND its mixins: the prompt text moved to a mixin once,
    and inspect.getsource(cls) sees only the class's own body, so this passed
    a file that no longer held the string it was checking for."""
    import inspect
    src = "".join(inspect.getsource(base) for base in LLMOrchestrator.__mro__
                  if base is not object)
    assert "never abbreviate" in src
