"""Which ``DomainModel.validate`` findings Phase 0 promotes to ``model contract:`` blockers.

Every error is promoted. A warning is promoted when it makes any generated
CRUD app unusable, or, once Phase 1 has run, when it breaks what the
scaffold's generator emits (``_SCAFFOLD_CONTRACT_WARNING_PREFIXES``). A
warning about a degraded but working app (an isolated class) never is.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"

    def __init__(self) -> None:
        self.usage = _Usage()

    def chat(self, system, messages, tools):  # pragma: no cover - never called
        return {"stop_reason": "end_turn", "content": []}


def _build(tmp_path, model: DomainModel, generator: str | None = None) -> LLMOrchestrator:
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=model,
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )
    orch._generator_used = generator
    return orch


def _blockers(orch: LLMOrchestrator) -> list[str]:
    return [i.message for i in orch._validation_issues
            if i.severity == "blocker" and i.message.startswith("model contract:")]


def _model(*classes, associations=(), generalizations=(), enums=()) -> DomainModel:
    return DomainModel(name="M", types=set(classes) | set(enums), associations=set(associations),
                       generalizations=set(generalizations))


def _keyword_model() -> DomainModel:
    return _model(Class(name="Item", attributes={Property(name="from", type=StringType)}))


def _join_table_model() -> DomainModel:
    student = Class(name="Student", attributes={Property(name="n", type=StringType)})
    course = Class(name="Course", attributes={Property(name="t", type=StringType)})
    enrollment = Class(name="Enrollment", attributes={Property(name="g", type=StringType)})
    link = BinaryAssociation(name="Enrollment", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(0, "*")),
    })
    return _model(student, course, enrollment, associations=[link])


def _class_typed_model() -> DomainModel:
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    book = Class(name="Book", attributes={Property(name="writer", type=author)})
    return _model(author, book)


def _root_id_model() -> DomainModel:
    return _model(Class(name="Item", attributes={Property(name="id", type=StringType),
                                                 Property(name="name", type=StringType)}))


def test_an_error_is_promoted_before_any_generator_runs(tmp_path):
    orch = _build(tmp_path, _model(Class(name="Item", attributes={Property(name="2nd", type=StringType)})))
    orch._collect_model_contract_issues()
    assert len(_blockers(orch)) == 1
    assert "'2nd' of class 'Item' is not a valid identifier" in _blockers(orch)[0]


def test_a_stack_independent_warning_is_promoted_before_any_generator_runs(tmp_path):
    upper = Class(name="Order", attributes={Property(name="x", type=StringType)})
    lower = Class(name="order", attributes={Property(name="y", type=StringType)})
    orch = _build(tmp_path, _model(upper, lower))
    orch._collect_model_contract_issues()
    assert [b for b in _blockers(orch) if "Class names differ only in case" in b]


def test_a_scaffold_warning_waits_for_the_generator(tmp_path):
    orch = _build(tmp_path, _keyword_model())
    orch._collect_model_contract_issues()
    assert _blockers(orch) == []

    orch._generator_used = "generate_fastapi_backend"
    orch._collect_model_contract_issues()
    assert len(_blockers(orch)) == 1
    assert "Python keyword name" in _blockers(orch)[0]


def test_a_scaffold_warning_is_not_promoted_for_a_generator_it_does_not_break(tmp_path):
    orch = _build(tmp_path, _keyword_model(), generator="generate_java_classes")
    orch._collect_model_contract_issues()
    assert _blockers(orch) == []


@pytest.mark.parametrize("model_factory, family_generator, other_generator", [
    (_join_table_model, "generate_fastapi_backend", "generate_django"),
    (_class_typed_model, "generate_fastapi_backend", "generate_django"),
    (_root_id_model, "generate_django", "generate_fastapi_backend"),
])
def test_scaffold_warnings_follow_the_family_they_break(tmp_path, model_factory, family_generator,
                                                       other_generator):
    broken = _build(tmp_path / "a", model_factory(), generator=family_generator)
    broken._collect_model_contract_issues()
    assert len(_blockers(broken)) == 1

    unaffected = _build(tmp_path / "b", model_factory(), generator=other_generator)
    unaffected._collect_model_contract_issues()
    assert _blockers(unaffected) == []


def test_a_degrading_warning_is_never_promoted(tmp_path):
    orch = _build(tmp_path, _model(Class(name="Empty"), Class(name="Item", attributes={
        Property(name="name", type=StringType)})), generator="generate_web_app")
    orch._collect_model_contract_issues()
    assert _blockers(orch) == []


def test_collecting_again_after_phase1_records_nothing_twice(tmp_path):
    upper = Class(name="Order", attributes={Property(name="from", type=StringType)})
    lower = Class(name="order", attributes={Property(name="y", type=StringType)})
    orch = _build(tmp_path, _model(upper, lower))
    orch._collect_model_contract_issues()
    orch._generator_used = "generate_web_app"
    orch._collect_model_contract_issues()
    orch._collect_model_contract_issues()
    messages = _blockers(orch)
    assert len(messages) == len(set(messages)) == 2
    assert len(orch._model_contract_issues) == 2


def test_every_promoted_prefix_starts_a_real_warning():
    """A prefix that no longer matches a message would silently stop promoting it."""
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    book = Class(name="Book", attributes={
        Property(name="author_id", type=IntegerType), Property(name="writer", type=author),
        Property(name="Title", type=StringType), Property(name="title", type=StringType),
        Property(name="from", type=StringType),
    })
    writes = BinaryAssociation(name="writes", ends={
        Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
        Property(name="books", type=book, multiplicity=Multiplicity(1, 1)),
    })
    student = Class(name="Student", attributes={Property(name="id", type=StringType)})
    course = Class(name="course", attributes={Property(name="t", type=StringType)})
    course_upper = Class(name="Course", attributes={Property(name="u", type=StringType)})
    enrollment = BinaryAssociation(name="Student", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(0, "*")),
    })
    child = Class(name="Graduate", attributes={Property(name="id", type=IntegerType)})
    empty = Enumeration(name="Status", literals=set())
    used = Enumeration(name="Level", literals={EnumerationLiteral(name="HIGH")})
    model = _model(author, book, student, course, course_upper, child, enums=[empty, used],
                   associations=[writes, enrollment],
                   generalizations=[Generalization(general=student, specific=child)])
    warnings = [w.lower() for w in model.validate(raise_exception=False)["warnings"]]

    prefixes = set(LLMOrchestrator._MODEL_CONTRACT_WARNING_PREFIXES)
    for family in LLMOrchestrator._SCAFFOLD_CONTRACT_WARNING_PREFIXES.values():
        prefixes.update(family)
    unmatched = sorted(p for p in prefixes if not any(w.startswith(p) for w in warnings))
    assert unmatched == []
