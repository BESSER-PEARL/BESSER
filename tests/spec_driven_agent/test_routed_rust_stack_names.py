"""A routed stack named "Rocket" or "Cargo" is Rust.

Bare "rocket" and "cargo" left the Rust keywords because in a request they are
usually the domain ("a cargo tracking company"). The routing answer's
``unsupported_stack`` is a stack name the router chose, not user prose, but it
was matched against the same list: a routed "Rocket" got no Cargo.toml and no
Rust idiom block. Free-text matching stays as it is.

Only the bare name counts (review, round 6): in a longer routed name the words are
the domain again - "Python rocket simulator" is not Rust.
"""

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.stack_metadata import (
    detect_idiom_stack, detect_routed_stack, detect_stack,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker

API = {"target_kind": "backend", "target_description": "REST API", "has_user_interface": False}


class _Selector:
    model = "mock-model"
    usage = UsageTracker("mock-model")
    _client = object()
    planning_model = None

    def __init__(self, answer):
        self.answer = answer

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        return {"content": [{"type": "tool_use", "input": dict(self.answer)}]}


@pytest.mark.parametrize("stack", ["Rocket", "Cargo", " rocket  Framework "])
def test_a_routed_rust_framework_name_gets_the_cargo_manifest(tmp_path, library_book_author_model, stack):
    instructions = "Build the library backend with Rocket."
    orch = LLMOrchestrator(llm_client=_Selector({**API, "unsupported_stack": stack, "generator": "NONE"}),
                           domain_model=library_book_author_model, output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._instructions = instructions
    orch._select_generator(instructions)
    orch._run_phase0_5_metadata(instructions)

    assert orch._phase0_5_stack == "rust"
    assert (tmp_path / "Cargo.toml").is_file()


@pytest.mark.parametrize("stack", ["Rocket", "Cargo"])
def test_a_routed_rust_framework_name_gets_the_rust_idioms(stack):
    assert detect_idiom_stack(stack) == "rust_axum"


def test_routed_names_still_follow_the_stack_order():
    assert detect_routed_stack("Next.js") == "nextjs"
    assert detect_routed_stack("Go") is None
    assert detect_routed_stack("FastAPI with a cargo module") is None


@pytest.mark.parametrize("request_text", [
    "Build a REST API for a cargo tracking company.",
    "Build a NestJS backend for a model rocket club.",
])
def test_free_text_still_ignores_the_bare_words(request_text):
    assert detect_stack(request_text) is None
    assert detect_idiom_stack(request_text) == (None if "NestJS" not in request_text else "generic")


# Review (round 6): honouring the bare words anywhere in the routed name made these Rust.
@pytest.mark.parametrize("stack, idiom", [
    ("Cargo tracking (FastAPI)", "python_web"),
    ("Python rocket simulator", None),
    ("Go (cargo tracking API)", None),
    ("NestJS (model rocket club)", "generic"),
    ("Unity rocket game", None),
    ("Rocket.Chat app", None),
    ("Node.js cargo API", None),
])
def test_a_longer_routed_name_reads_the_words_as_the_domain(stack, idiom):
    assert detect_routed_stack(stack) is None
    assert detect_idiom_stack(stack) == idiom
