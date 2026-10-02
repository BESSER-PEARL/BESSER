"""The Phase 0 ``model contract:`` blocker must reach the recipe.

``_collect_model_contract_issues`` detects a mandatory creation cycle — the
live case where ``Booking`` required a ``ReservedRoom`` id and
``ReservedRoom`` required a ``Booking`` id, so the shipped app served 69
routes and could create neither entity. It is deliberately not part of
``_collect_validation_issues``: no edit to generated code can repair the
specification it was generated from.

Phase 3 then REPLACED ``_validation_issues`` wholesale with what
``_collect_validation_issues`` returned, so a clean Phase 3 erased the
finding: ``incomplete: false`` on a run with a fatal, already-detected
defect. It survived only when Phase 3 was skipped for budget — backwards.

``modify()`` never ran the collector at all, so a model edited between runs
was never checked.
"""

from __future__ import annotations

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Multiplicity,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue

StringType = PrimitiveDataType("str")


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"

    def __init__(self) -> None:
        self.usage = _Usage()

    def chat(self, system, messages, tools):  # pragma: no cover - never called
        return {"stop_reason": "end_turn", "content": []}


def _cyclic_model() -> DomainModel:
    """Booking 1..1 <-> ReservedRoom 1..1: neither can be created first."""
    booking = Class(name="Booking", attributes={Property(name="code", type=StringType)})
    room = Class(name="ReservedRoom", attributes={Property(name="label", type=StringType)})
    link = BinaryAssociation(name="booking_room", ends={
        Property(name="booking", type=booking, multiplicity=Multiplicity(1, 1)),
        Property(name="reservedRoom", type=room, multiplicity=Multiplicity(1, 1)),
    })
    return DomainModel(name="Hotel", types={booking, room}, associations={link})


def _acyclic_model() -> DomainModel:
    booking = Class(name="Booking", attributes={Property(name="code", type=StringType)})
    room = Class(name="ReservedRoom", attributes={Property(name="label", type=StringType)})
    link = BinaryAssociation(name="booking_room", ends={
        Property(name="booking", type=booking, multiplicity=Multiplicity(0, 1)),
        Property(name="reservedRoom", type=room, multiplicity=Multiplicity(1, 1)),
    })
    return DomainModel(name="Hotel", types={booking, room}, associations={link})


def _build(tmp_path, model: DomainModel) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_Client(),
        domain_model=model,
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )


def _neutralise_phase3(orch: LLMOrchestrator, monkeypatch, issues) -> None:
    """Run the real ``_run_phase3_validation`` with a scripted collector."""
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: list(issues))
    monkeypatch.setattr(orch, "_phase3_stop_requested",
                        lambda check_turn_budget=True: None)
    monkeypatch.setattr(orch, "_apply_runtime_gate", lambda: None)
    monkeypatch.setattr(orch, "_complete_repair_if_verified", lambda: None)


def _contract_blockers(orch: LLMOrchestrator) -> list[str]:
    return [i.message for i in orch._validation_issues
            if i.severity == "blocker" and i.message.startswith("model contract:")]


# ---------------------------------------------------------------- detection


def test_phase0_records_the_cycle_as_a_blocker(tmp_path):
    orch = _build(tmp_path, _cyclic_model())
    orch._collect_model_contract_issues()
    assert _contract_blockers(orch), "Phase 0 did not flag the mandatory cycle"
    assert orch._model_contract_issues


def test_phase0_is_silent_on_a_constructible_model(tmp_path):
    orch = _build(tmp_path, _acyclic_model())
    orch._collect_model_contract_issues()
    assert orch._validation_issues == []
    assert orch._model_contract_issues == []


# ------------------------------------------------- survival through Phase 3


def test_clean_phase3_does_not_erase_the_phase0_blocker(tmp_path, monkeypatch):
    """The regression: ``self._validation_issues = []`` on a clean Phase 3."""
    orch = _build(tmp_path, _cyclic_model())
    orch._collect_model_contract_issues()
    _neutralise_phase3(orch, monkeypatch, [])

    orch._run_phase3_validation()

    assert _contract_blockers(orch), (
        "a clean Phase 3 discarded the Phase 0 model-contract blocker"
    )
    assert sum(1 for i in orch._validation_issues if i.severity == "blocker") == 1


def test_phase3_with_its_own_findings_keeps_the_phase0_blocker(tmp_path, monkeypatch):
    """The other assignment site: ``self._validation_issues = list(issues)``."""
    orch = _build(tmp_path, _cyclic_model())
    orch._collect_model_contract_issues()
    _neutralise_phase3(orch, monkeypatch, [ValidationIssue("warning", "tsc: soft")])

    orch._run_phase3_validation()

    assert _contract_blockers(orch)
    assert any(i.message == "tsc: soft" for i in orch._validation_issues)


def test_phase3_reported_blockers_include_the_model_contract(tmp_path, monkeypatch):
    """What the runner reads to build ``_unfixed_blockers``."""
    orch = _build(tmp_path, _cyclic_model())
    orch._collect_model_contract_issues()
    _neutralise_phase3(orch, monkeypatch, [])
    orch._run_phase3_validation()

    reported = [i.message for i in orch._validation_issues if i.severity == "blocker"]
    assert len(reported) == 1
    assert "Mandatory creation cycle" in reported[0]


def test_no_phantom_blocker_when_the_model_is_fine(tmp_path, monkeypatch):
    """Carrying Phase 0 forward must not invent findings."""
    orch = _build(tmp_path, _acyclic_model())
    orch._collect_model_contract_issues()
    _neutralise_phase3(orch, monkeypatch, [])
    orch._run_phase3_validation()
    assert orch._validation_issues == []


def test_the_blocker_is_recorded_once(tmp_path, monkeypatch):
    """Phase 3 replaces, so the prepend must not stack duplicates."""
    orch = _build(tmp_path, _cyclic_model())
    orch._collect_model_contract_issues()
    _neutralise_phase3(orch, monkeypatch, [])
    orch._run_phase3_validation()
    orch._run_phase3_validation()
    assert len(_contract_blockers(orch)) == 1


# --------------------------------------------------------------- modify()


def test_modify_runs_the_model_contract_check(tmp_path, monkeypatch):
    """A model edited between runs was never checked."""
    orch = _build(tmp_path, _cyclic_model())

    calls: list[str] = []
    monkeypatch.setattr(orch, "_derive_and_apply_model_deltas", lambda instr: None)
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch, "_run_phase2",
                        lambda instructions, extra_issues=(): calls.append("phase2"))
    monkeypatch.setattr(orch, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_remove_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instructions, elapsed: None)
    monkeypatch.setattr(orch, "_finish_checkpoint", lambda: None)
    _neutralise_phase3(orch, monkeypatch, [])

    orch.modify("Rename the booking code field to reference")

    assert calls == ["phase2"], "the modify run did not reach Phase 2"
    assert _contract_blockers(orch), (
        "modify() shipped without ever checking the model contract"
    )
