"""Tests for the deterministic model-conformance checker and the Phase 2
done gate that uses it."""

import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Method,
    PrimitiveDataType, Property,
)
from besser.generators.llm.conformance import (
    ConformanceReport,
    _name_variants,
    check_conformance,
    format_missing_for_llm,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator


@pytest.fixture
def library_model():
    StringType = PrimitiveDataType("str")
    book = Class(name="BookLoan")
    book.attributes = {
        Property(name="title", type=StringType),
        Property(name="release_date", type=StringType),
    }
    book.methods = {Method(name="extend_loan")}
    status = Enumeration(
        name="LoanStatus",
        literals={EnumerationLiteral(name="ACTIVE"), EnumerationLiteral(name="RETURNED")},
    )
    return DomainModel(name="Library", types={book, status})


class TestNameVariants:
    def test_pascal_to_snake_and_camel(self):
        variants = _name_variants("BookLoan")
        assert "book_loan" in variants
        assert "bookLoan" in variants
        assert "BookLoan" in variants

    def test_snake_to_pascal(self):
        variants = _name_variants("release_date")
        assert "releaseDate" in variants
        assert "ReleaseDate" in variants


class TestCheckConformance:
    def test_empty_workspace_reports_everything_missing(self, library_model, tmp_path):
        report = check_conformance(library_model, str(tmp_path))
        assert report.expected > 0
        assert report.found == 0
        assert not report.ok

    def test_full_workspace_passes(self, library_model, tmp_path):
        (tmp_path / "models.py").write_text(
            "class BookLoan:\n"
            "    title: str\n"
            "    release_date: str\n"
            "    def extend_loan(self): ...\n"
            "\n"
            "class LoanStatus(Enum):\n"
            "    ACTIVE = 1\n"
            "    RETURNED = 2\n"
        )
        report = check_conformance(library_model, str(tmp_path))
        assert report.ok, f"unexpected missing: {[m.describe() for m in report.missing]}"
        assert report.ratio == 1.0

    def test_casing_variants_count_as_found(self, library_model, tmp_path):
        """A TS file using camelCase satisfies a snake_case model name."""
        (tmp_path / "models.ts").write_text(
            "interface BookLoan { title: string; releaseDate: string; }\n"
            "function extendLoan(loan: BookLoan) {}\n"
            "enum LoanStatus { ACTIVE, RETURNED }\n"
        )
        report = check_conformance(library_model, str(tmp_path))
        assert report.ok, f"unexpected missing: {[m.describe() for m in report.missing]}"

    def test_member_must_cooccur_with_owner(self, library_model, tmp_path):
        """An attribute name in an unrelated file does not satisfy the check."""
        (tmp_path / "other.py").write_text("title = 'hello'\nrelease_date = None\n")
        report = check_conformance(library_model, str(tmp_path))
        missing = {m.describe() for m in report.missing}
        assert "attribute 'BookLoan.title'" in missing

    def test_none_model_is_trivially_ok(self, tmp_path):
        report = check_conformance(None, str(tmp_path))
        assert report.ok
        assert report.ratio == 1.0

    def test_skips_node_modules(self, library_model, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("BookLoan title release_date extend_loan LoanStatus ACTIVE RETURNED")
        report = check_conformance(library_model, str(tmp_path))
        assert report.checked_files == 0
        assert not report.ok

    def test_format_missing_for_llm(self, library_model, tmp_path):
        report = check_conformance(library_model, str(tmp_path))
        text = format_missing_for_llm(report)
        assert "MODEL CONFORMANCE CHECK FAILED" in text
        assert "BookLoan" in text


class MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


def _end_turn_client():
    class MockClient:
        model = "mock-model"
        usage = UsageTracker("mock-model")
        calls = 0

        def chat(self, system, messages, tools):
            type(self).calls += 1
            return {"stop_reason": "end_turn", "content": [
                MockBlock("text", text="Done!"),
            ]}

    return MockClient()


class TestDoneGate:
    def test_gate_nudges_then_accepts(self, library_model, tmp_path):
        """With an empty workspace the gate fires its bounded nudges,
        then accepts the end_turn instead of looping forever."""
        client = _end_turn_client()
        orchestrator = LLMOrchestrator(
            llm_client=client,
            domain_model=library_model,
            output_dir=str(tmp_path),
            use_streaming=False,
        )
        # NestJS has no deterministic generator, so Phase 1 is skipped
        # and the workspace stays empty — the mock LLM writes nothing.
        orchestrator.run("Build a NestJS app")

        assert orchestrator._conformance_nudges == LLMOrchestrator._MAX_CONFORMANCE_NUDGES
        assert orchestrator._phase2_exited_cleanly is True

    def test_gate_quiet_when_workspace_conforms(self, library_model, tmp_path):
        (tmp_path / "models.py").write_text(
            "class BookLoan:\n"
            "    title: str\n"
            "    release_date: str\n"
            "    def extend_loan(self): ...\n"
            "class LoanStatus:\n"
            "    ACTIVE = 1\n"
            "    RETURNED = 2\n"
        )
        orchestrator = LLMOrchestrator(
            llm_client=_end_turn_client(),
            domain_model=library_model,
            output_dir=str(tmp_path),
            use_streaming=False,
        )
        orchestrator.run("Describe the model")

        assert orchestrator._conformance_nudges == 0
        assert orchestrator._phase2_exited_cleanly is True

    def test_gate_disabled_accepts_immediately(self, library_model, tmp_path):
        orchestrator = LLMOrchestrator(
            llm_client=_end_turn_client(),
            domain_model=library_model,
            output_dir=str(tmp_path),
            use_streaming=False,
            enable_conformance_gate=False,
        )
        orchestrator.run("Describe the model")

        assert orchestrator._conformance_nudges == 0
        assert orchestrator._conformance_report is None

    def test_gate_skipped_without_domain_model(self, tmp_path):
        orchestrator = LLMOrchestrator(
            llm_client=_end_turn_client(),
            gui_model=None,
            agent_model=None,
            state_machines=[object()],  # non-empty so primary_kind resolves
            output_dir=str(tmp_path),
            use_streaming=False,
        )
        orchestrator._run_phase2("Describe the model")
        assert orchestrator._conformance_nudges == 0

    def test_nudge_emits_phase_details(self, library_model, tmp_path):
        details_calls = []
        orchestrator = LLMOrchestrator(
            llm_client=_end_turn_client(),
            domain_model=library_model,
            output_dir=str(tmp_path),
            use_streaming=False,
            on_phase_details=lambda phase, details: details_calls.append((phase, details)),
        )
        orchestrator.run("Build a NestJS app")

        customize_details = [d for p, d in details_calls if p == "customize"]
        assert customize_details
        assert "Conformance check" in customize_details[0]

    def test_recipe_contains_conformance(self, library_model, tmp_path):
        import json

        orchestrator = LLMOrchestrator(
            llm_client=_end_turn_client(),
            domain_model=library_model,
            output_dir=str(tmp_path),
            use_streaming=False,
        )
        orchestrator.run("Describe the model")

        recipe_path = os.path.join(str(tmp_path), ".besser_recipe.json")
        with open(recipe_path) as f:
            recipe = json.load(f)
        assert recipe["conformance"] is not None
        assert recipe["conformance"]["expected"] > 0
        assert "missing" in recipe["conformance"]
