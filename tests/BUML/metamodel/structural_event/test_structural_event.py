"""Metaclass checks for structural events and the ViolatingEvent link."""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)
from besser.BUML.metamodel.structural_event import (
    DeleteAssociation,
    InsertClass,
    UpdateAttribute,
    ViolatingEvent,
    event_from_dict,
    event_to_dict,
)


def _model():
    sale = Class("Sale", attributes={Property("amount", IntegerType)})
    customer = Class("Customer", attributes={Property("name", StringType)})
    purchases = BinaryAssociation("Purchases", ends={
        Property("customer", customer, multiplicity=Multiplicity(1, 1)),
        Property("sale", sale, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Shop", types={sale, customer}, associations={purchases}), sale, customer


def test_class_event_rejects_a_non_class():
    with pytest.raises(TypeError):
        InsertClass("Sale")


def test_update_attribute_rejects_a_non_property():
    _model_obj, sale, _customer = _model()
    with pytest.raises(TypeError):
        UpdateAttribute("amount", sale)


def test_violating_event_registers_and_detaches():
    model, sale, _customer = _model()
    constraint = Constraint("Positive", sale, "self.amount > 0", "OCL")
    model.add_constraint(constraint)
    link = ViolatingEvent(constraint, UpdateAttribute(next(iter(sale.attributes)), sale))
    assert link in constraint.violating_events
    assert link.constraint is constraint
    link.constraint = None
    assert constraint.violating_events == set()
    assert link.constraint is None


def test_violating_event_rejects_a_non_event():
    _model_obj, sale, _customer = _model()
    constraint = Constraint("Positive", sale, "self.amount > 0", "OCL")
    with pytest.raises(TypeError):
        ViolatingEvent(constraint, sale)


def test_event_round_trip_resolves_names():
    model, sale, _customer = _model()
    constraint = Constraint("Positive", sale, "self.amount > 0", "OCL")
    amount = next(attr for attr in sale.attributes if attr.name == "amount")
    link = ViolatingEvent(constraint, UpdateAttribute(amount, sale), approximated=True)
    payload = event_to_dict(link)
    assert payload == {
        "constraint": "Positive",
        "kind": "UpdateAttribute",
        "approximated": True,
        "class": "Sale",
        "attribute": "amount",
    }
    constraint.violating_events = set()
    restored = event_from_dict(payload, constraint, model)
    assert isinstance(restored.event, UpdateAttribute)
    assert restored.event.attribute is amount
    assert restored.approximated is True
    assert restored in constraint.violating_events


def test_association_event_round_trip():
    model, sale, _customer = _model()
    constraint = Constraint("Linked", sale, "self.customer->notEmpty()", "OCL")
    purchases = next(assoc for assoc in model.associations if assoc.name == "Purchases")
    link = ViolatingEvent(constraint, DeleteAssociation(purchases))
    payload = event_to_dict(link)
    assert payload["association"] == "Purchases"
    assert "class" not in payload
    constraint.violating_events = set()
    restored = event_from_dict(payload, constraint, model)
    assert isinstance(restored.event, DeleteAssociation)
    assert restored.event.target is purchases
