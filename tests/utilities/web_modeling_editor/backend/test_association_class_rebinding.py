"""Promoting a class to an association class must not orphan its own associations.

Reported on a RestaurantOrderingSystem model: the editor showed OrderLine
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


def _cls(eid, name):
    """A v4 ``class`` node."""
    return {"id": eid, "type": "class",
            "position": {"x": 0, "y": 0}, "width": 200, "height": 100,
            "data": {"name": name, "stereotype": None, "attributes": [], "methods": []}}


def _assoc(rid, name, src, tgt, rtype="ClassBidirectional", src_mult="1", tgt_mult="0..*"):
    """A v4 association edge; the source role is left empty (class-name fallback)."""
    return {"id": rid, "type": rtype, "source": src, "target": tgt,
            "data": {"name": name,
                     "sourceRole": "", "sourceMultiplicity": src_mult,
                     "targetRole": name, "targetMultiplicity": tgt_mult,
                     "points": []}}


def _diagram(title, nodes, edges):
    return {"title": title,
            "model": {"version": "4.0.0", "type": "ClassDiagram", "nodes": nodes, "edges": edges}}


@pytest.fixture
def restaurant_shape():
    """Order--MenuItem with OrderLine as the association class, AND OrderLine
    separately associated to both ends -- exactly what the agent produced."""
    return _diagram(
        "New_Project",
        [
            _cls("c_order", "Order"),
            _cls("c_item", "MenuItem"),
            _cls("c_line", "OrderLine"),
        ],
        [
            _assoc("r_om", "orderMenuItem", "c_order", "c_item"),
            # the promotion: OrderLine is the association class of r_om
            # (v4: a ClassLinkRel edge from the class node to the association edge)
            {"id": "r_link", "type": "ClassLinkRel", "source": "c_line", "target": "r_om",
             "data": {"points": []}},
            _assoc("r_lines", "lines", "c_line", "c_order"),
            _assoc("r_ol", "orderLines_1", "c_item", "c_line"),
        ],
    )


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
    payload = _diagram("Plain", [_cls("a", "Book"), _cls("b", "Author")],
                       [_assoc("r", "writes", "a", "b")])

    domain = _domain(payload)

    assert not any(isinstance(t, AssociationClass) for t in domain.types)
    domain.validate()


def test_the_promoted_class_knows_its_own_associations(restaurant_shape):
    """Ends were re-pointed at the new AssociationClass, but the associations
    stayed registered on the discarded Class, so association_ends() on the
    promoted class came back empty. Generators read it to render the link's
    side of 'lines': the web-app scaffold of a hotel model
    (ExtraCharge -> BookingRoom) failed at mapper configuration."""
    domain = _domain(restaurant_shape)
    line = next(t for t in domain.types if getattr(t, "name", None) == "OrderLine")

    owners = {end.owner.name for end in line.association_ends()}

    assert {"lines", "orderLines_1"} <= owners


def test_the_generated_orm_configures(restaurant_shape, tmp_path):
    """End to end from editor JSON: the SQLAlchemy module must configure."""
    import importlib.util
    from sqlalchemy.orm import clear_mappers, configure_mappers
    from besser.generators.sql_alchemy import SQLAlchemyGenerator

    SQLAlchemyGenerator(_domain(restaurant_shape), output_dir=str(tmp_path)).generate()
    spec = importlib.util.spec_from_file_location("rebinding_orm", tmp_path / "sql_alchemy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        configure_mappers()
    finally:
        clear_mappers()
