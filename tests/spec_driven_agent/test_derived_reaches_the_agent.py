"""A derived attribute must be visible to the agent, not just to the generator.

Observed live: the spec said the booking statuses are "not set by
hand" and the total price is "not typed in"; the model marked all three
``is_derived``; and the generated create schema demanded all three. The agent
could not have fixed it — ``_attribute_entry`` never emitted ``is_derived``,
so the LLM saw an ordinary field.
"""
from besser.BUML.metamodel.structural import (
    Class, DomainModel, FloatType, Property, StringType,
)
from besser.spec_driven_agent.model_serializer import _attribute_entry


def test_derived_flag_is_serialized_for_the_llm():
    attr = Property(name="totalPrice", type=FloatType, is_derived=True)
    assert _attribute_entry(attr).get("is_derived") is True


def test_plain_attribute_carries_no_derived_key():
    """Absent rather than false — the payload stays compact."""
    attr = Property(name="bookingNumber", type=StringType)
    assert "is_derived" not in _attribute_entry(attr)


def test_derived_attribute_still_reports_name_and_type():
    attr = Property(name="totalPrice", type=FloatType, is_derived=True)
    entry = _attribute_entry(attr)
    assert entry["name"] == "totalPrice"
    assert entry["type"] == FloatType.name


def test_data_contract_tells_the_agent_what_derived_means():
    from besser.spec_driven_agent.agent import prompt_builder
    import inspect
    src = inspect.getsource(prompt_builder)
    assert "is_derived" in src, "the contract must name the flag the model emits"
    contract = src[src.index("Server-owned fields"):][:600]
    assert "COMPUTED" in contract or "computed" in contract


def test_whole_model_serialization_includes_the_flag():
    from besser.spec_driven_agent.model_serializer import serialize_domain_model
    booking = Class(name="Booking", attributes={
        Property(name="bookingNumber", type=StringType),
        Property(name="totalPrice", type=FloatType, is_derived=True),
    })
    dm = DomainModel(name="T", types={booking})
    import json
    blob = json.dumps(serialize_domain_model(dm))
    assert '"is_derived": true' in blob or '"is_derived":true' in blob
