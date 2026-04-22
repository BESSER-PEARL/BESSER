"""Tests that the LLM generator pipeline works without a domain model.

Covers the primary-model refactor: users can drive smart generation from
any BESSER diagram (state machine, agent, GUI, quantum, object), not
just a ClassDiagram. These tests assert that the wiring stays intact
when ``domain_model`` is None — the orchestrator builds, tool filtering
removes the domain-centric generators, and the tool executor surfaces
clean errors instead of crashing when the LLM calls a generator that
would need a domain model.
"""

from __future__ import annotations

import pytest

from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import (
    GENERATOR_TOOLS,
    get_all_tools_including_generators,
    get_tools_for,
)


class _MockClient:
    """Minimal LLM client stub — the orchestrator never actually calls it
    in these tests, but it needs a ``usage`` attribute for cost tracking.
    """

    def __init__(self) -> None:
        self.model = "test-model"
        self.usage = _MockUsage()

    def chat(self, *_, **__):  # pragma: no cover - not invoked
        raise AssertionError("tests should not call the LLM")


class _MockUsage:
    estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 0, "cost_usd": 0.0}


class _MockStateMachine:
    """Shape-compatible with orchestrator / compaction expectations."""

    def __init__(self, name: str = "OrderStateMachine") -> None:
        self.name = name


def test_orchestrator_accepts_state_machine_only(tmp_path):
    """An SM-only project must produce a working orchestrator instance."""
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        state_machines=[_MockStateMachine("OrderSM")],
        output_dir=str(tmp_path),
    )

    assert orch.domain_model is None
    assert orch.primary_kind == "state_machine"
    assert len(orch.state_machines) == 1


def test_orchestrator_rejects_empty_project():
    """No models at all → clear error. We never want a silent no-op run."""
    with pytest.raises(ValueError, match="at least one model"):
        LLMOrchestrator(llm_client=_MockClient())


def test_tool_filter_hides_domain_generators_when_no_model():
    """Every generator that needs a DomainModel should disappear from
    the LLM's tool list when no domain model is loaded. Keeps the LLM
    from wasting turns calling generators that would just error.
    """
    tools = get_tools_for(has_domain_model=False)
    tool_names = {t["name"] for t in tools}

    # Domain-only generators must be gone
    assert "generate_pydantic" not in tool_names
    assert "generate_fastapi_backend" not in tool_names
    assert "generate_django" not in tool_names
    assert "generate_sqlalchemy" not in tool_names
    # GUI-needing generators rely on domain too — also gone
    assert "generate_react" not in tool_names
    assert "generate_web_app" not in tool_names
    # Model-query tools are domain-only → gone
    assert "query_class" not in tool_names
    assert "validate_model" not in tool_names
    # Workspace / execution tools stay — they're model-agnostic
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "modify_file" in tool_names
    assert "run_command" in tool_names


def test_tool_filter_restores_domain_generators_when_model_present():
    """Sanity check: the non-filtered tool list is strictly a superset."""
    with_model = {t["name"] for t in get_tools_for(has_domain_model=True, has_gui_model=True)}
    without_model = {t["name"] for t in get_tools_for(has_domain_model=False)}
    all_tools = {t["name"] for t in get_all_tools_including_generators()}

    # With domain + gui we should have everything
    assert with_model == all_tools
    # Without domain, we lose the domain-requiring tools
    assert with_model > without_model


def test_tool_executor_surfaces_clear_error_without_domain_model(tmp_path):
    """The LLM might still request a domain-requiring tool (older prompt,
    race with tool filter). When it does, the executor must return a
    structured error — never crash with AttributeError.
    """
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=None)

    import json
    result = json.loads(executor.execute("generate_pydantic", {}))
    assert "error" in result
    assert "domain model" in result["error"].lower()


def test_tool_executor_model_query_without_domain_is_graceful(tmp_path):
    """query_class / list_classes_with / validate_model must also degrade
    gracefully rather than throwing through the handler."""
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=None)

    import json
    for tool_name in ("query_class", "list_classes_with", "get_constraints_for", "validate_model"):
        result = json.loads(executor.execute(
            tool_name,
            # bogus args — the guard should kick in before parsing them
            {"name": "X", "predicate": "is_abstract", "class_name": "X"},
        ))
        assert "error" in result, f"{tool_name} should return an error"


def test_generator_tools_cover_every_model_requirement():
    """Regression guard: if a new generator tool is added without an
    entry in ``_TOOL_MODEL_REQUIREMENTS``, ``get_tools_for`` would leak
    it into no-domain runs. This test forces the mapping to stay
    complete.
    """
    from besser.generators.llm.tools import _TOOL_MODEL_REQUIREMENTS

    for tool in GENERATOR_TOOLS:
        assert tool["name"] in _TOOL_MODEL_REQUIREMENTS, (
            f"Generator {tool['name']} has no entry in _TOOL_MODEL_REQUIREMENTS"
        )


def test_primary_kind_override_in_orchestrator(tmp_path):
    """When a project has both domain and state machines, the caller
    can force ``primary_kind='state_machine'`` to reframe the run."""
    class _MockDomain:
        def get_classes(self):
            return []

        def get_enumerations(self):
            return []

        associations = []

    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_MockDomain(),
        state_machines=[_MockStateMachine()],
        output_dir=str(tmp_path),
        primary_kind="state_machine",
    )
    assert orch.primary_kind == "state_machine"
    # Domain model still available, just not driving the framing
    assert orch.domain_model is not None
