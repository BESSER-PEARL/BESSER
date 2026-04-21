"""Tests for build_system_prompt — verifies every available model type
ends up in the prompt, and that cross-model links are computed."""

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.prompt_builder import build_system_prompt


def _make_minimal_domain() -> DomainModel:
    StringType = PrimitiveDataType("str")
    order = Class(name="Order")
    order.attributes = {Property(name="id", type=StringType, is_id=True)}
    return DomainModel(name="Shop", types={order})


class _FakeClassifier:
    def __init__(self, name):
        self.name = name


class _FakeStateMachine:
    """Minimal stand-in that matches the shape ``serialize_state_machines``
    reads from."""

    def __init__(self, name, states=None, properties=None):
        self.name = name
        self.states = states or []
        self.properties = properties or []


class _FakeState:
    def __init__(self, name, initial=False, final=False, transitions=None):
        self.name = name
        self.initial = initial
        self.final = final
        self.transitions = transitions or []


def test_domain_model_always_present():
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build the app",
        max_turns=20,
    )
    assert "## Domain Model" in prompt
    assert '"Order"' in prompt


def test_optional_sections_are_omitted_when_models_absent():
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build the app",
        max_turns=20,
    )
    assert "## GUI Model" not in prompt
    assert "## Agent Model" not in prompt
    assert "## Object Model" not in prompt
    assert "## State Machines" not in prompt
    assert "## Quantum Circuit" not in prompt
    assert "## Cross-model links" not in prompt


def test_state_machine_and_cross_links_included():
    """When a state-machine name contains a class name, the prompt must
    emit a cross-model link section pointing at the governed class."""
    sm = _FakeStateMachine("OrderFSM", states=[
        _FakeState("Draft", initial=True),
        _FakeState("Closed", final=True),
    ])

    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build the app",
        max_turns=20,
        state_machines=[sm],
    )

    assert "## State Machines" in prompt
    assert "OrderFSM" in prompt
    # Cross-model link should bind OrderFSM → Order (class name in SM name).
    assert "## Cross-model links" in prompt
    assert '"governs_class": "Order"' in prompt
    assert '"state_machine": "OrderFSM"' in prompt


def test_user_instructions_rendered_verbatim():
    """The user's request is the source of truth for what to build —
    must appear verbatim in the prompt."""
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Add JWT auth and write Dockerfile",
        max_turns=20,
    )
    assert "Add JWT auth and write Dockerfile" in prompt
    assert "User request" in prompt


def test_scoped_issues_section_appears_when_present():
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build the app",
        scoped_issues=["Syntax error in main.py"],
        max_turns=20,
    )
    assert "Phase 1 validation" in prompt
    assert "Syntax error in main.py" in prompt


def test_inventory_included_when_non_empty():
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="Generated 12 files",
        instructions="Build the app",
        max_turns=20,
    )
    assert "## What was already generated" in prompt
    assert "Generated 12 files" in prompt
