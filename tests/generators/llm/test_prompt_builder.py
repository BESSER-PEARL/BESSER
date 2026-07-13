"""Tests for build_system_prompt — verifies every available model type
ends up in the prompt, and that cross-model links are computed."""

from types import SimpleNamespace

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
    assert "## BPMN Process Model" not in prompt
    assert "## Neural Network Model" not in prompt
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


def test_idiom_guidance_included_for_spring_boot():
    """#4b: naming a recognised stack must inject its idiom block."""
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build a Spring Boot backend for order management",
        max_turns=20,
    )
    assert "## Idiomatic conventions for Spring Boot" in prompt
    assert "ResponseStatusException" in prompt
    assert "Long" in prompt


def test_idiom_guidance_included_for_flask_and_rust():
    flask_prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build a Flask API for orders",
        max_turns=20,
    )
    assert "## Idiomatic conventions for Flask / FastAPI (Python)" in flask_prompt
    assert "Decimal" in flask_prompt

    rust_prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build a Rust axum backend for orders",
        max_turns=20,
    )
    assert "## Idiomatic conventions for Rust (axum)" in rust_prompt
    assert "Option<T>" in rust_prompt


def test_idiom_guidance_omitted_when_no_stack_named():
    """No stack mentioned -- the block must not appear (avoid prompt bloat)."""
    prompt = build_system_prompt(
        domain_model=_make_minimal_domain(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build a simple CRUD app for orders",
        max_turns=20,
    )
    assert "Idiomatic conventions" not in prompt


def test_bpmn_and_nn_primary_banners_are_explicit():
    for primary_kind, expected in (
        ("bpmn", "BPMN-driven run"),
        ("nn", "Neural-network-driven run"),
    ):
        prompt = build_system_prompt(
            domain_model=None,
            gui_model=None,
            agent_model=None,
            inventory="",
            instructions="Generate code",
            max_turns=20,
            primary_kind=primary_kind,
        )
        assert expected in prompt


def test_bpmn_and_nn_models_are_serialized_into_agent_context():
    start = SimpleNamespace(name="Start", lane=None)
    review = SimpleNamespace(name="Review request", lane=None)
    flow = SimpleNamespace(
        name="start-to-review",
        source=start,
        target=review,
        is_default=False,
    )
    process = SimpleNamespace(
        name="Approval",
        flow_nodes=[start, review],
        sequence_flows=[flow],
        lanes=[],
    )
    bpmn_model = SimpleNamespace(
        name="Approval workflow",
        processes=[process],
        collaboration=None,
    )
    dense = SimpleNamespace(
        name="classifier",
        actv_func="relu",
        name_module_input=None,
        input_reused=False,
        in_features=32,
        out_features=4,
    )
    nn_model = SimpleNamespace(
        name="Risk classifier",
        modules=[dense],
        configuration=None,
        train_data=None,
        test_data=None,
    )

    prompt = build_system_prompt(
        domain_model=None,
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Implement both specifications",
        max_turns=20,
        bpmn_model=bpmn_model,
        nn_model=nn_model,
        primary_kind="bpmn",
    )

    assert "## BPMN Process Model" in prompt
    assert "Approval workflow" in prompt
    assert "start-to-review" in prompt
    assert "## Neural Network Model" in prompt
    assert "Risk classifier" in prompt
    assert '"out_features": 4' in prompt
