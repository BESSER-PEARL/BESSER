"""Promoting a class to an association class must not orphan its own associations.

Live report (2026-09-22, RestaurantOrderingSystem): the editor showed OrderLine
on the canvas, and validation insisted

    Association 'lines' has end 'orderline' referencing type 'OrderLine'
    which is not in the domain model 'New_Project'.

The class was right there. The converter creates the plain Class, builds every
association against it, and only then -- on seeing the ClassLinkRel -- swaps a
new AssociationClass into ``domain_model.types`` and discards the original. Any
end still bound to the discarded object is now unreachable, and Class compares
by IDENTITY, so ``end.type not in self.__types`` fires on a name that is
plainly present.

It was unfixable from the UI: the editor's auto-fix removed and re-added the two
associations, which rebuilt exactly the same dangling references, so the user
was told "Applied 6 changes" while the errors stayed put.
"""
import pytest

from besser.BUML.metamodel.structural import AssociationClass, Class
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


def _cls(eid, name, owner=None):
    return {"id": eid, "name": name, "type": "Class", "attributes": [], "methods": [],
            "owner": owner, "bounds": {"x": 0, "y": 0, "width": 200, "height": 100}}


def _assoc(rid, name, src, tgt, rtype="ClassBidirectional", src_mult="1", tgt_mult="0..*"):
    return {"id": rid, "name": name, "type": rtype,
            "source": {"element": src, "multiplicity": src_mult, "role": ""},
            "target": {"element": tgt, "multiplicity": tgt_mult, "role": name}}


@pytest.fixture
def restaurant_shape():
    """Order--MenuItem with OrderLine as the association class, AND OrderLine
    separately associated to both ends -- exactly what the agent produced."""
    return {
        "title": "New_Project",
        "model": {
            "type": "ClassDiagram",
            "elements": {
                "c_order": _cls("c_order", "Order"),
                "c_item": _cls("c_item", "MenuItem"),
                "c_line": _cls("c_line", "OrderLine"),
            },
            "relationships": {
                "r_om": _assoc("r_om", "orderMenuItem", "c_order", "c_item"),
                # the promotion: OrderLine is the association class of r_om
                "r_link": {"id": "r_link", "name": "", "type": "ClassLinkRel",
                           "source": {"element": "c_line"},
                           "target": {"element": "r_om"}},
                "r_lines": _assoc("r_lines", "lines", "c_line", "c_order"),
                "r_ol": _assoc("r_ol", "orderLines_1", "c_item", "c_line"),
            },
        },
    }


def _domain(payload):
    result = process_class_diagram(payload)
    return result[0] if isinstance(result, tuple) else result


def test_every_association_end_resolves_after_promotion(restaurant_shape):
    """The regression: two ends pointed at a Class that had been discarded."""
    domain = _domain(restaurant_shape)

    orphans = [(a.name, e.name) for a in domain.associations
               for e in a.ends if e.type not in domain.types]

    assert orphans == [], f"ends left bound to a discarded object: {orphans}"


def test_the_model_validates(restaurant_shape):
    """What the user actually saw. validate() raises on the first error."""
    domain = _domain(restaurant_shape)

    domain.validate()


def test_the_promotion_still_happens(restaurant_shape):
    """Guard: rebinding must not be achieved by skipping the promotion."""
    domain = _domain(restaurant_shape)

    by_name = {t.name: t for t in domain.types if isinstance(t, Class)}
    assert isinstance(by_name["OrderLine"], AssociationClass)
    assert by_name["OrderLine"].association.name == "orderMenuItem"


def test_exactly_one_OrderLine_object_is_reachable(restaurant_shape):
    """The defect was two objects of the same name. Identity is what matters."""
    domain = _domain(restaurant_shape)

    reachable = {id(t) for t in domain.types if getattr(t, "name", None) == "OrderLine"}
    reachable |= {id(e.type) for a in domain.associations for e in a.ends
                  if getattr(e.type, "name", None) == "OrderLine"}

    assert len(reachable) == 1, "the plain Class and the AssociationClass both survive"


def test_a_plain_diagram_is_untouched():
    """No ClassLinkRel -> nothing to promote, nothing to rebind."""
    payload = {
        "title": "Plain",
        "model": {"type": "ClassDiagram",
                  "elements": {"a": _cls("a", "Book"), "b": _cls("b", "Author")},
                  "relationships": {"r": _assoc("r", "writes", "a", "b")}},
    }

    domain = _domain(payload)

    assert not any(isinstance(t, AssociationClass) for t in domain.types)
    domain.validate()
