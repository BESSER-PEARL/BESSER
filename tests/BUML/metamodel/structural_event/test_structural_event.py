"""Metaclass checks for structural events and the ViolatingEvent association class."""

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
    DeleteLink,
    DeleteObject,
    InsertObject,
    UpdateAttribute,
    ViolatingEvent,
    event_from_dict,
    event_to_dict,
    set_violating_events,
    violating_events,
)


def _model():
    sale = Class("Sale", attributes={Property("amount", IntegerType)})
    customer = Class("Customer", attributes={Property("name", StringType)})
    purchases = BinaryAssociation("Purchases", ends={
        Property("customer", customer, multiplicity=Multiplicity(1, 1)),
        Property("sale", sale, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Shop", types={sale, customer}, associations={purchases}), sale, customer


def test_object_event_is_linked_to_the_class():
    _model_obj, sale, _customer = _model()
    event = DeleteObject(sale)
    assert event.class_ is sale
    assert repr(event) == "DeleteObject(Sale)"


def test_object_event_rejects_a_non_class():
    with pytest.raises(TypeError):
        InsertObject("Sale")


def test_update_attribute_is_linked_to_the_property():
    _model_obj, sale, _customer = _model()
    amount = next(attr for attr in sale.attributes if attr.name == "amount")
    event = UpdateAttribute(amount)
    assert event.attribute is amount
    assert repr(event) == "UpdateAttribute(amount, Sale)"


def test_update_attribute_rejects_a_non_property():
    with pytest.raises(TypeError):
        UpdateAttribute("amount")


def test_link_event_is_linked_to_the_association():
    model, _sale, _customer = _model()
    purchases = next(assoc for assoc in model.associations if assoc.name == "Purchases")
    event = DeleteLink(purchases)
    assert event.association is purchases
    assert repr(event) == "DeleteLink(Purchases)"


def test_violating_event_links_exactly_one_constraint_and_one_event():
    model, sale, _customer = _model()
    constraint = Constraint("Positive", sale, "self.amount > 0", "OCL")
    model.add_constraint(constraint)
    amount = next(iter(sale.attributes))
    event = UpdateAttribute(amount)
    link = ViolatingEvent(constraint, event)
    assert link.constraint is constraint
    assert link.structural_event is event
    assert link in violating_events(constraint)
    with pytest.raises(TypeError):
        link.constraint = None
    with pytest.raises(TypeError):
        ViolatingEvent(None, event)


def test_replacing_links_drops_them_from_the_constraint_without_clearing_their_ends():
    _model_obj, sale, _customer = _model()
    constraint = Constraint("Positive", sale, "self.amount > 0", "OCL")
    link = ViolatingEvent(constraint, UpdateAttribute(next(iter(sale.attributes))))
    set_violating_events(constraint, set())
    assert violating_events(constraint) == set()
    assert link.constraint is constraint
    assert isinstance(link.structural_event, UpdateAttribute)


def test_event_round_trip_resolves_names():
    model, sale, _customer = _model()
    constraint = Constraint("Positive", sale, "self.amount > 0", "OCL")
    amount = next(attr for attr in sale.attributes if attr.name == "amount")
    link = ViolatingEvent(constraint, UpdateAttribute(amount), approximated=True)
    payload = event_to_dict(link)
    assert payload == {
        "constraint": "Positive",
        "kind": "UpdateAttribute",
        "approximated": True,
        "class": "Sale",
        "attribute": "amount",
    }
    set_violating_events(constraint, set())
    restored = event_from_dict(payload, constraint, model)
    assert isinstance(restored.structural_event, UpdateAttribute)
    assert restored.structural_event.attribute is amount
    assert restored.approximated is True
    assert restored in violating_events(constraint)


def test_link_event_round_trip():
    model, sale, _customer = _model()
    constraint = Constraint("Linked", sale, "self.customer->notEmpty()", "OCL")
    purchases = next(assoc for assoc in model.associations if assoc.name == "Purchases")
    link = ViolatingEvent(constraint, DeleteLink(purchases))
    payload = event_to_dict(link)
    assert payload["association"] == "Purchases"
    assert payload["kind"] == "DeleteLink"
    assert "class" not in payload
    set_violating_events(constraint, set())
    restored = event_from_dict(payload, constraint, model)
    assert isinstance(restored.structural_event, DeleteLink)
    assert restored.structural_event.association is purchases
