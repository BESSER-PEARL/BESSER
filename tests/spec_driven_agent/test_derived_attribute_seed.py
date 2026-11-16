"""A derived attribute of the model becomes a Phase 2 task.

The scaffold stores a derived attribute as a plain column with a constant
default. Nothing planned its computation, so an app shipped the constant
unless the agent happened to notice the attribute while reading the model.
"""

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.gap_analyzer import derived_attribute_tasks
from besser.spec_driven_agent.providers.llm_client import UsageTracker


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        raise AssertionError("never called")


def _shop() -> DomainModel:
    level = Enumeration(name="Level", literals={EnumerationLiteral(name="LOW"), EnumerationLiteral(name="HIGH")})
    order = Class(name="Order")
    order.attributes = {
        Property(name="quantity", type=PrimitiveDataType("int")),
        Property(name="total", type=PrimitiveDataType("float"), is_derived=True),
        Property(name="isLate", type=PrimitiveDataType("bool"), is_derived=True),
        Property(name="priority", type=level, is_derived=True),
    }
    customer = Class(name="Customer")
    customer.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    return DomainModel(name="Shop", types={order, customer, level})


def test_each_class_with_derived_attributes_gets_one_task_naming_them():
    [task] = derived_attribute_tasks(_shop())
    assert task.startswith("Compute the derived attributes Order.isLate, Order.total:")
    assert "whenever the record is read" in task and "every write that changes" in task
    assert "never accept" in task


def test_a_derived_enumeration_is_left_to_its_initial_state_task():
    assert "priority" not in derived_attribute_tasks(_shop())[0]


def test_a_model_without_derived_attributes_gets_none():
    model = _shop()
    order = next(c for c in model.get_classes() if c.name == "Order")
    order.attributes = {a for a in order.attributes if not a.is_derived}
    assert derived_attribute_tasks(model) == []
    assert derived_attribute_tasks(None) == []


def _seeds(tmp_path, generator_used="generate_fastapi_backend", modify=False):
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=_shop(), output_dir=str(tmp_path))
    orch._generator_used = generator_used
    orch._instructions = "Build the shop API."
    orch._modify_mode = modify
    return [t for t in orch._deterministic_gap_tasks() if isinstance(t, str) and "derived" in t]


def test_the_task_is_seeded_on_a_new_fastapi_scaffold_only(tmp_path):
    assert len(_seeds(tmp_path)) == 1
    assert _seeds(tmp_path, generator_used=None) == []
    assert _seeds(tmp_path, modify=True) == [], "a modify run's app was built from this model already"
