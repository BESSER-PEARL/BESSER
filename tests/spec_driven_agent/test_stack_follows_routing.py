"""The stack a run builds is the routing answer's, not a keyword in the request.

Phase 0.5 picked its manifest template from bare words of the request. A
domain word ("a cargo tracking company", "a model rocket club", "rust-proof
paint", "a Kotlin user-group meetup") planted a Rust or Kotlin manifest in a
Go, NestJS or FastAPI run, and the inventory told Phase 2 to build on it. The
system prompt's idiom block came from the same words: "rust-proof paint" put
the Rust (axum) block (about 170 tokens) into a FastAPI run's prompt, and "a
FastAPI backend for a model rocket club" got it instead of the Python block. The routing call already decided the stack
(``unsupported_stack``); the keywords are kept only for runs with no routing
answer.
"""

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.stack_metadata import idiom_guidance_section
from besser.spec_driven_agent.providers.llm_client import UsageTracker

API = {"target_kind": "backend", "target_description": "REST API", "has_user_interface": False}
WEB = {"target_kind": "web_app", "target_description": "web app", "has_user_interface": True}
RUST_IDIOMS = "Idiomatic conventions for Rust"


class _Selector:
    """A provider whose choose_generator call returns ``answer``."""

    model = "mock-model"
    usage = UsageTracker("mock-model")
    _client = object()
    planning_model = None

    def __init__(self, answer):
        self.answer = answer

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        return {"content": [{"type": "tool_use", "input": dict(self.answer)}]}


def _routed_phase0_5(tmp_path, model, instructions, answer):
    orch = LLMOrchestrator(llm_client=_Selector(answer), domain_model=model,
                           output_dir=str(tmp_path), enable_tracing=False,
                           enable_checkpointing=False)
    orch._instructions = instructions
    orch._select_generator(instructions)
    # Phase 1 wrote nothing: no generator fits, or the chosen one failed.
    orch._run_phase0_5_metadata(instructions)
    return orch


@pytest.mark.parametrize("instructions, answer", [
    ("Build a REST API in Go for a cargo tracking company.",
     {**API, "unsupported_stack": "Go", "generator": "NONE"}),
    ("Build a NestJS backend for a model rocket club: members, launches, rockets.",
     {**API, "unsupported_stack": "NestJS", "generator": "NONE"}),
    ("Build an Express API for a Kotlin user-group meetup calendar.",
     {**API, "unsupported_stack": "Express", "generator": "NONE"}),
    # A Python web app whose FastAPI scaffold failed.
    ("A web app for a paint shop selling rust-proof coatings.",
     {**WEB, "unsupported_stack": "", "generator": "generate_fastapi_backend"}),
])
def test_a_domain_word_plants_no_manifest(tmp_path, library_book_author_model, instructions, answer):
    orch = _routed_phase0_5(tmp_path, library_book_author_model, instructions, answer)
    assert orch._phase0_5_stack is None
    assert list(tmp_path.iterdir()) == []
    assert "Phase 0.5" not in orch._inventory


@pytest.mark.parametrize("instructions, stack, manifest", [
    ("Build the library backend in Rust with Axum.", "Rust + Axum", "Cargo.toml"),
    ("Build the library service on the JVM, Spring Boot please.", "Spring Boot", "build.gradle.kts"),
])
def test_the_routed_stack_still_gets_its_manifest(tmp_path, library_book_author_model,
                                                  instructions, stack, manifest):
    orch = _routed_phase0_5(tmp_path, library_book_author_model, instructions,
                            {**API, "unsupported_stack": stack, "generator": "NONE"})
    assert (tmp_path / manifest).is_file()
    assert manifest in orch._inventory


# -- the idiom block in the system prompt ------------------------------------

@pytest.mark.parametrize("instructions, routed_stack, expected", [
    # Domain words: the block the routing answer implies, not Rust's.
    ("A web app for a paint shop selling rust-proof coatings.", "", None),
    ("A FastAPI backend for a model rocket club.", "", "Flask / FastAPI (Python)"),
    ("Build a REST API in Go for a cargo tracking company.", "Go", None),
    ("Build a NestJS backend for a model rocket club.", "NestJS", "this stack"),
    ("Build a FastAPI app inspired by Rust services.", "", "Flask / FastAPI (Python)"),
    # A stack the user really asked for keeps its block.
    ("Build a REST API in Rust with axum.", "Rust + Axum", "Rust (axum)"),
    ("Build a Spring Boot API for orders.", "Spring Boot", "Spring Boot"),
    ("Build a Django project for a library.", "", "this stack"),
    ("A Next.js frontend on a FastAPI backend.", "Next.js", "Flask / FastAPI (Python)"),
    # The routing answer names a stack the request spells differently.
    ("Build the library service on the JVM.", "Spring Boot", "Spring Boot"),
])
def test_idiom_block_follows_the_routing_answer(instructions, routed_stack, expected):
    section = idiom_guidance_section(instructions, routed_stack)
    if expected is None:
        assert section == ""
    else:
        assert section.startswith(f"## Idiomatic conventions for {expected}\n")


def test_without_a_routing_answer_the_keywords_decide():
    assert idiom_guidance_section("Build a REST API in Rust with axum.").startswith(
        "## Idiomatic conventions for Rust (axum)")


def test_the_run_prompt_carries_no_rust_block_for_rust_proof_paint(tmp_path, library_book_author_model):
    instructions = "A web app for a paint shop selling rust-proof coatings."
    orch = _routed_phase0_5(tmp_path, library_book_author_model, instructions,
                            {**WEB, "unsupported_stack": "", "generator": "generate_fastapi_backend"})
    assert RUST_IDIOMS not in orch._build_system_prompt(instructions)
