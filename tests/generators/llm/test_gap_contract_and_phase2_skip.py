"""Tests for the gap-analyser output contract and the Phase-2 skip.

The contract (see gap_analyzer module docstring):
  * ``None``  — analysis failed (LLM error, unparseable reply, mock client)
  * ``[]``    — the model judged the scaffold sufficient → Phase 2 may skip
  * ``[...]`` — focused task list

Also covers the supporting plumbing: always-valid-JSON model truncation,
whole-text task parsing, the relaxed write_file guardrail, and the
GENERATOR_TOOLS ↔ ToolExecutor._handlers parity (including qiskit).
"""

from __future__ import annotations

import json
import os

import pytest

from besser.generators.llm import gap_analyzer
from besser.generators.llm.gap_analyzer import (
    _parse_task_array,
    _safe_serialize_model,
    analyze_gaps_via_llm,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import GENERATOR_TOOLS


class _MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockClient:
    """Duck-typed client WITHOUT ``_client`` — gap analyser must skip it."""

    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")
        self.chat_calls = 0

    def chat(self, system, messages, tools):
        self.chat_calls += 1
        return {"stop_reason": "end_turn", "content": [_MockBlock("text", text="Done")]}


# ----------------------------------------------------------------------
# Output contract
# ----------------------------------------------------------------------


def test_mock_client_returns_none_not_empty():
    """A skipped analysis must be None (failure), never [] (no work)."""
    result = analyze_gaps_via_llm(
        instructions="add auth",
        generator_used="generate_fastapi_backend",
        domain_model=None,
        inventory="3 files",
        llm_client=_MockClient(),
    )
    assert result is None


def test_no_generator_fallback_mentions_failure_reason():
    result = analyze_gaps_via_llm(
        instructions="build it",
        generator_used=None,
        domain_model=None,
        inventory="",
        llm_client=_MockClient(),
        generator_failure="generate_sqlalchemy: reserved name 'Base'",
    )
    assert isinstance(result, list) and len(result) == 1
    assert "generate_sqlalchemy: reserved name 'Base'" in result[0]
    assert "avoid the cause" in result[0]


def test_no_generator_fallback_without_failure_keeps_legacy_text():
    result = analyze_gaps_via_llm(
        instructions="build it",
        generator_used=None,
        domain_model=None,
        inventory="",
        llm_client=_MockClient(),
    )
    assert isinstance(result, list) and len(result) == 1
    assert "No BESSER generator was used" in result[0]


# ----------------------------------------------------------------------
# Parsing / truncation plumbing
# ----------------------------------------------------------------------


def test_parse_task_array_survives_brackets_inside_tasks():
    text = '["update models[0] in api.py", "add CHECK (age > [0]) constraint"]'
    parsed = _parse_task_array(text)
    assert parsed == [
        "update models[0] in api.py",
        "add CHECK (age > [0]) constraint",
    ]


def test_parse_task_array_fenced_json():
    text = '```json\n["task one", "task two"]\n```'
    assert _parse_task_array(text) == ["task one", "task two"]


def test_safe_serialize_model_truncation_is_always_valid_json(monkeypatch):
    """A model far over budget must still serialize to parseable JSON."""
    huge = {
        "classes": [
            {
                "name": f"Class{i}",
                "attributes": [
                    {"name": f"attr{j}", "type": "str", "visibility": "public",
                     "doc": "x" * 50}
                    for j in range(30)
                ],
                "methods": [{"name": f"m{j}", "params": []} for j in range(20)],
                "inherited_attributes": [{"name": "base"} for _ in range(20)],
                "inherited_methods": [{"name": "bm"} for _ in range(20)],
            }
            for i in range(200)
        ],
    }
    monkeypatch.setattr(gap_analyzer, "serialize_domain_model", lambda _m: huge)
    payload = _safe_serialize_model(object())
    data = json.loads(payload)  # must never raise
    assert data.get("__truncated__") is True
    assert len(payload) <= gap_analyzer._MAX_MODEL_JSON_CHARS + 100


# ----------------------------------------------------------------------
# Phase-2 skip
# ----------------------------------------------------------------------


class _ExplodingClient(_MockClient):
    """chat() must never be reached when Phase 2 is skipped."""

    def chat(self, system, messages, tools):  # pragma: no cover - guard
        raise AssertionError("Phase 2 was not skipped — chat() was called")


def _make_orchestrator(tmp_path, client):
    class _SM:
        name = "DummySM"

    return LLMOrchestrator(
        llm_client=client,
        state_machines=[_SM()],
        output_dir=str(tmp_path),
    )


def test_phase2_skipped_when_gap_tasks_empty_and_generator_ran(tmp_path, monkeypatch):
    client = _ExplodingClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )
    progress: list[tuple] = []
    orch.on_progress = lambda *a: progress.append(a)

    orch._run_phase2("build a library api", extra_issues=[])

    assert orch.total_turns == 0
    assert orch._phase2_exited_cleanly is True
    assert any(a[1] == "__customize_skipped__" for a in progress)


def test_phase2_not_skipped_when_analysis_failed(tmp_path, monkeypatch):
    """None (failure) must keep today's behavior: loop without checklist."""
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: None,
    )

    orch._run_phase2("build a library api", extra_issues=[])

    assert client.chat_calls == 1  # entered the loop, got end_turn
    assert orch.total_turns == 1


def test_phase2_not_skipped_when_no_generator(tmp_path, monkeypatch):
    """[] without a deterministic scaffold is not trusted as 'done'."""
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    assert orch._generator_used is None
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )

    orch._run_phase2("build a library api", extra_issues=[])

    assert client.chat_calls == 1


def test_phase2_not_skipped_when_scoped_issues_present(tmp_path, monkeypatch):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )

    orch._run_phase2(
        "build a library api",
        extra_issues=["Fix syntax error in api.py line 3"],
    )

    assert client.chat_calls == 1


# ----------------------------------------------------------------------
# write_file guardrail relaxation
# ----------------------------------------------------------------------


def test_write_file_allowed_after_two_modifies(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    rel = "backend/api.py"
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("line_a = 1\nline_b = 2\n")
    executor._generator_files.add(rel)

    # Cold write on a small generated file → rejected, with both escapes named
    cold = json.loads(executor.execute("write_file", {"path": rel, "content": "x = 1\n"}))
    assert "error" in cold
    assert "modify_file" in cold["error"]
    assert "delete_file" in cold["error"]

    for old, new in (("line_a = 1", "line_a = 10"), ("line_b = 2", "line_b = 20")):
        result = json.loads(executor.execute(
            "modify_file", {"path": rel, "old_text": old, "new_text": new},
        ))
        assert result.get("status") == "modified"

    # After two demonstrated modify attempts, the rewrite is allowed
    rewrite = json.loads(executor.execute(
        "write_file", {"path": rel, "content": "x = 1\n"},
    ))
    assert rewrite.get("status") == "written"


# ----------------------------------------------------------------------
# Tool registry parity (incl. the qiskit registration)
# ----------------------------------------------------------------------


def test_every_generator_tool_has_an_executor_handler():
    handler_names = set(ToolExecutor._handlers.keys())
    for tool in GENERATOR_TOOLS:
        assert tool["name"] in handler_names, f"{tool['name']} has no handler"


def test_qiskit_without_circuit_surfaces_clean_error(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path), quantum_circuit=None)
    result = json.loads(executor.execute("generate_qiskit", {}))
    assert "error" in result
    assert "quantum circuit" in result["error"].lower()


def test_target_generator_override_is_binding(tmp_path):
    """A caller-specified generator must win without any LLM call."""
    client = _ExplodingClient()  # chat() raises if reached

    class _SM:
        name = "DummySM"

    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[_SM()],
        output_dir=str(tmp_path),
        target_generator="generate_fastapi_backend",
    )
    assert orch._select_generator("anything at all") == "generate_fastapi_backend"


def test_resume_seeds_prior_cost(tmp_path, monkeypatch):
    """A resumed run's cost cap must cover what the crashed run spent."""
    tracker = UsageTracker("mock-model")
    tracker.seed_cost(1.25)
    assert abs(tracker.estimated_cost - 1.25) < 1e-9
    # Token-based accounting still works on top of the seed
    tracker.output_tokens += 100_000  # sonnet fallback pricing: $1.50
    assert tracker.estimated_cost > 1.25
