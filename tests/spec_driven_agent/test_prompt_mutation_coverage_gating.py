"""The relationship-mutation coverage block is left out only when nothing can need it.

``build_endpoint_manifest`` appended ~3.3k chars of mutation coverage to every Phase 2
prompt. It is now omitted when the requirements ledger is KNOWN to be empty (this
process extracted it) and the model holds no constraint, rejected constraint or lower
bound on a to-many end. Any requirement keeps it: the extractor's ``kind`` label flips
between runs of one request (the p8 rules were ``rule`` in 6 saved runs and ``action``
in 4, and live r10-p81 lost the block). The request shapes in it always stay: test_api
needs them. The request intro also claimed generated "pages" for a backend-only scaffold.
"""
from unittest import mock

import pytest

from besser.BUML.metamodel.structural import Constraint
from besser.spec_driven_agent.agent.prompt_builder import (
    build_endpoint_manifest, build_system_prompt, mutation_coverage_applies,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

COVERAGE = "Relationship mutation coverage"
SHAPES = "Request shapes (inherited fields included):"


def _orch(tmp_path, model, requirements, extracted=True):
    generate_backend(tmp_path, model)
    orch = LLMOrchestrator(llm_client=mock.MagicMock(), domain_model=model, output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._generator_used = GENERATOR
    orch._requirements = requirements
    orch._requirement_extraction_attempts = 1 if extracted else 0
    return orch


def test_known_empty_ledger_and_no_model_rule_drop_the_coverage_but_keep_the_request_shapes(tmp_path):
    orch = _orch(tmp_path, shelf_model(author=True), [])
    prompt = orch._build_system_prompt("Build the app")
    assert COVERAGE not in prompt
    assert SHAPES in prompt and "- BookCreate: " in prompt
    # The full block is still what the manifest renders by default (Phase 3, acceptance).
    assert COVERAGE in build_endpoint_manifest(str(tmp_path))


@pytest.mark.parametrize("kind", ["rule", "action", "ui", ""])
def test_any_requirement_keeps_the_coverage_whatever_its_kind(tmp_path, kind):
    orch = _orch(tmp_path, shelf_model(author=True),
                 [{"id": 1, "text": "A library holds at most 100 books.", "kind": kind}])
    assert COVERAGE in orch._build_system_prompt("Build the app")


def test_failed_extraction_keeps_the_coverage(tmp_path):
    orch = _orch(tmp_path, shelf_model(author=True), None)
    assert COVERAGE in orch._build_system_prompt("Build the app")


def test_an_empty_ledger_restored_on_resume_is_unknown(tmp_path):
    # A checkpoint restores the list but not whether this process extracted it.
    orch = _orch(tmp_path, shelf_model(author=True), [], extracted=False)
    assert COVERAGE in orch._build_system_prompt("Build the app")


def test_model_facts_that_keep_the_coverage():
    model = shelf_model(author=True)
    assert not mutation_coverage_applies(model, [])
    assert mutation_coverage_applies(model, [{"text": "any requirement", "kind": "action"}])
    assert mutation_coverage_applies(model, None)
    rejected = shelf_model(author=True)
    rejected.conversion_issues = [{"id": "c1", "expression": "self.books->size() < 100"}]
    assert mutation_coverage_applies(rejected, [])
    book = next(c for c in model.get_classes() if c.name == "Book")
    model.constraints = {Constraint(name="pages_positive", context=book, expression="self.pages > 0",
                                    language="OCL")}
    assert mutation_coverage_applies(model, [])
    # A lower bound on a to-many end: a library needs at least one book.
    fresh = shelf_model()
    books = next(end for association in fresh.associations for end in association.ends if end.name == "books")
    books.multiplicity.min = 1
    assert mutation_coverage_applies(fresh, [])


def _intro(inventory):
    prompt = build_system_prompt(shelf_model(), None, None, inventory, "Build the app", 10)
    return prompt.split("## User request", 1)[1].split("\n\n", 2)[1]


def test_request_intro_says_pages_only_when_the_scaffold_has_them():
    backend = "Generator `generate_fastapi_backend` produced 2 files:\n  - backend/main_api.py (10 bytes)"
    web = backend + "\n  - frontend/src/pages/BookList.tsx (10 bytes)"
    # The intro names no parts (the inventory above lists them), so it cannot claim pages.
    assert "pages" not in _intro(backend) and _intro(web) == _intro(backend)
