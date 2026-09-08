"""Tests for the FIX/MODIFY success gate on ``LLMOrchestrator``.

A modify run seeded by a user-reported failure must:

  * feed the tool's own matching structural findings into Phase 2 as
    explicit fix instructions;
  * promote a matching finding from warning to blocker for the duration of
    the run, so the fix loop is driven at it and success gates on it;
  * report an honest "could not confirm the reported failure is fixed"
    outcome (not a clean success) when the matching finding persists.

From-scratch ``run()`` must be entirely unaffected — it never detects a
fix target, so classification and severity are identical to today's.
"""

from __future__ import annotations

import json

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.fix_target import parse_reported_target
from besser.generators.llm.orchestrator import LLMOrchestrator, ValidationIssue


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self):
        self.estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 0, "cost_usd": 0.0}


class _ScriptedClient:
    """No ``_client`` attr → model-sync and structured selectors skip it."""

    def __init__(self, responses=None):
        self.model = "test-model"
        self.usage = _Usage()
        self.max_tokens = 4096
        self._responses = list(responses or [])
        self.chat_calls = 0

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        self.chat_calls += 1
        if self._responses:
            return self._responses.pop(0)
        return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _domain(*names: str) -> DomainModel:
    string_type = PrimitiveDataType("str")
    types = set()
    for name in names:
        cls = Class(name=name)
        cls.attributes = {Property(name="title", type=string_type, is_id=True)}
        types.add(cls)
    return DomainModel(name="App", types=types)


def _orch(tmp_path, domain, client=None) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client or _ScriptedClient(),
        domain_model=domain,
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )


_FIX_INSTR = (
    "The stock app crashes: POST /createWatchlist returns a 400 error. "
    "Please fix it."
)


# ----------------------------------------------------------------------
# Promotion (pure) — matching finding warning -> blocker
# ----------------------------------------------------------------------


def test_promote_matching_finding_to_blocker(tmp_path):
    orch = _orch(tmp_path, _domain("Watchlist", "User"))
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target(_FIX_INSTR, orch.domain_model)

    issues = [
        ValidationIssue("warning", "acceptance: entity Watchlist — no frontend POST for it"),
        ValidationIssue("warning", "acceptance: entity User — no backend REST route"),
        ValidationIssue("style", "ruff: F401 unused import"),
    ]
    out = orch._promote_fix_target_findings(issues)

    watchlist = next(i for i in out if "watchlist" in i.message.lower())
    assert watchlist.severity == "blocker"
    assert watchlist.message.startswith(orch._FIX_TARGET_PREFIX)
    # Non-matching finding is untouched.
    user = next(i for i in out if "user" in i.message.lower())
    assert user.severity == "warning"


def test_promote_is_noop_when_not_a_fix_run(tmp_path):
    orch = _orch(tmp_path, _domain("Watchlist"))
    assert orch._is_fix_run is False
    issues = [ValidationIssue("warning", "acceptance: entity Watchlist — no frontend POST")]
    assert orch._promote_fix_target_findings(issues) == issues


# ----------------------------------------------------------------------
# Promotion via _collect_validation_issues (real acceptance matrix)
# ----------------------------------------------------------------------


def test_collect_validation_issues_promotes_matching_acceptance_finding(tmp_path):
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    orch = _orch(tmp_path, _domain("Watchlist"))
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target(_FIX_INSTR, orch.domain_model)

    issues = orch._collect_validation_issues()
    blockers = [i for i in issues if i.severity == "blocker"]
    assert any(
        "watchlist" in b.message.lower() and b.message.startswith(orch._FIX_TARGET_PREFIX)
        for b in blockers
    ), f"expected a promoted Watchlist blocker, got {[i.message for i in issues]}"


def test_collect_validation_issues_unchanged_for_first_generation(tmp_path):
    """Same workspace, but not a fix run: the acceptance finding for the
    entity stays a warning — first-generation classification is untouched."""
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    orch = _orch(tmp_path, _domain("Watchlist"))
    # _is_fix_run defaults False (run() never calls _detect_fix_target).
    issues = orch._collect_validation_issues()
    watchlist = [i for i in issues if "watchlist" in i.message.lower()]
    assert watchlist, "acceptance matrix should still report the entity"
    assert all(i.severity != "blocker" for i in watchlist)
    assert not any(i.message.startswith(orch._FIX_TARGET_PREFIX) for i in issues)


# ----------------------------------------------------------------------
# End-of-run gate verdict
# ----------------------------------------------------------------------


def test_gate_marks_unresolved_when_matching_blocker_persists(tmp_path):
    orch = _orch(tmp_path, _domain("Watchlist"))
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target(_FIX_INSTR, orch.domain_model)
    orch._validation_issues = [
        ValidationIssue(
            "blocker",
            orch._FIX_TARGET_PREFIX
            + "acceptance: entity Watchlist — no frontend POST for it",
        )
    ]
    orch._evaluate_fix_target_gate()
    assert orch._fix_target_resolved is False
    assert orch._fix_target_message is not None
    assert "could not confirm" in orch._fix_target_message.lower()
    # The honest message must not carry our internal prefix.
    assert orch._FIX_TARGET_PREFIX not in orch._fix_target_message


def test_gate_marks_resolved_when_no_matching_blocker(tmp_path):
    orch = _orch(tmp_path, _domain("Watchlist"))
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target(_FIX_INSTR, orch.domain_model)
    orch._validation_issues = [ValidationIssue("warning", "ruff: F401 unused import")]
    orch._evaluate_fix_target_gate()
    assert orch._fix_target_resolved is True
    assert orch._fix_target_message is None


def test_gate_soft_target_is_unverified(tmp_path):
    orch = _orch(tmp_path, _domain("Watchlist"))
    orch._is_fix_run = True
    orch._fix_target = parse_reported_target("it does not work, please fix it", orch.domain_model)
    assert orch._fix_target.is_soft
    orch._evaluate_fix_target_gate()
    # Neither confirmed nor denied — no false success, no false incomplete.
    assert orch._fix_target_resolved is None
    assert orch._fix_target_message is None


# ----------------------------------------------------------------------
# Full modify() wiring
# ----------------------------------------------------------------------


def _stub_modify_phases(orch, monkeypatch, phase3_issues):
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_remove_snapshot", lambda: None)

    def _phase3():
        orch._validation_issues = list(phase3_issues)

    monkeypatch.setattr(orch, "_run_phase3_validation", _phase3)


def test_modify_feeds_fix_scoped_issues_into_phase2(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    orch = _orch(tmp_path, _domain("Watchlist"))

    captured: dict = {}
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(
        orch, "_run_phase2",
        lambda instr, extra_issues=None: captured.update(issues=extra_issues),
    )
    monkeypatch.setattr(orch, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_remove_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_run_phase3_validation", lambda: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)

    orch.modify(_FIX_INSTR)

    assert orch._is_fix_run is True
    joined = "\n".join(captured["issues"])
    assert "must be resolved in this run" in joined
    assert "createwatchlist" in joined.lower() or "watchlist" in joined.lower()


def test_modify_gate_reports_unresolved_and_writes_recipe(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('x')\n", encoding="utf-8")
    orch = _orch(tmp_path, _domain("Watchlist"))

    persisting = ValidationIssue(
        "blocker",
        orch._FIX_TARGET_PREFIX
        + "acceptance: entity Watchlist — no frontend POST for it",
    )
    _stub_modify_phases(orch, monkeypatch, [persisting])

    orch.modify(_FIX_INSTR)

    assert orch._fix_target_resolved is False
    assert orch._fix_target_message
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["fix_run"]["detected"] is True
    assert recipe["fix_run"]["target_resolved"] is False
    assert recipe["fix_run"]["target_message"]


def test_run_never_detects_a_fix_target(tmp_path, monkeypatch):
    """First-generation run() must not activate the gate even when the
    instructions are full of fix vocabulary."""
    orch = _orch(tmp_path, _domain("Watchlist"))
    monkeypatch.setattr(orch, "_run_phase1", lambda instr: None)
    monkeypatch.setattr(orch, "_run_phase0_5_metadata", lambda instr: None)
    monkeypatch.setattr(orch, "_apply_adaptive_budget", lambda: None)
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_remove_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_run_phase3_validation", lambda: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)

    orch.run("fix the 400 error in POST /createWatchlist")

    assert orch._is_fix_run is False
    assert orch._fix_target is None
    assert orch._fix_target_message is None
