"""Chapter 4 potentially-violating events, checked against thesis Table 4.7.

The ecommerce model and the five invariants are the running example of
Cabot and Teniente, *Incremental Integrity Checking in UML/OCL Conceptual
Schemas*. ``Time.now()`` does not parse in B-OCL; ``Date::now()`` is the
same literal and does not change the events.
"""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DateType,
    DomainModel,
    FloatType,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
)
from besser.BUML.metamodel.structural_event import (
    event_from_dict,
    event_to_dict,
    set_violating_events,
    violating_events,
)
from besser.BUML.notations.ocl import parse_ocl
from besser.BUML.notations.ocl.incremental import analyze_invariant, analyze_invariants


def _end(name, target, lower, upper):
    return Property(name, target, multiplicity=Multiplicity(lower, upper))


def ecommerce_model() -> DomainModel:
    """Figure 1.1 of the thesis, with the names the invariants navigate."""
    sale = Class("Sale", attributes={
        Property("paymentDate", DateType),
        Property("amount", FloatType),
    })
    shipment = Class("Shipment", attributes={
        Property("plannedShipDate", DateType),
    })
    product = Class("Product", attributes={
        Property("price", FloatType),
        Property("maxDiscount", IntegerType),
    })
    restricted = Class("RestrictedProduct", attributes={
        Property("maxUnits", IntegerType),
    })
    customer = Class("Customer")
    category = Class("Category", attributes={
        Property("maxPendingAmount", FloatType),
    })
    delivered_in = BinaryAssociation("DeliveredIn", ends={
        _end("sale", sale, 1, 1),
        _end("shipment", shipment, 0, "*"),
    })
    belongs_to = BinaryAssociation("BelongsTo", ends={
        _end("category", category, 0, 1),
        _end("customer", customer, 0, "*"),
    })
    purchases = BinaryAssociation("Purchases", ends={
        _end("buyer", customer, 1, 1),
        _end("sale", sale, 0, "*"),
    })
    model = DomainModel(
        "Ecommerce",
        types={sale, shipment, product, restricted, customer, category},
        associations={delivered_in, belongs_to, purchases},
    )
    model.generalizations = {Generalization(general=product, specific=restricted)}
    return model


# The navigation from Customer to Sale is ``sale`` in the thesis
# (``c.sale``). The association end on the Customer side of Purchases is
# named ``buyer`` here so it does not collide with BelongsTo's ``customer``
# end when both are navigable from Sale. The OCL still says ``c.sale``.
INVARIANTS = {
    "ValidShipDate": (
        "context Sale inv ValidShipDate: "
        "self.shipment->forAll(sh | sh.plannedShipDate <= self.paymentDate + 30)"
    ),
    "CorrectProduct": (
        "context Product inv CorrectProduct: self.price > 0 and self.maxDiscount <= 60"
    ),
    "NotTooPendingSales": (
        "context Category inv NotTooPendingSales: "
        "self.customer->forAll(c | c.sale->select(s | s.paymentDate > Date::now())"
        "->collect(sa | sa.amount)->sum() <= self.maxPendingAmount)"
    ),
    "AtLeastThreeCustomers": (
        "context Category inv AtLeastThreeCustomers: self.customer->size() >= 3"
    ),
    "NumberOfRestrictedProducts": (
        "context RestrictedProduct inv NumberOfRestrictedProducts: "
        "RestrictedProduct.allInstances()->size() <= 20"
    ),
}

TABLE_4_7 = {
    "ValidShipDate": {
        "UpdateAttribute(paymentDate, Sale)",
        "UpdateAttribute(plannedShipDate, Shipment)",
        "InsertLink(DeliveredIn)",
    },
    "CorrectProduct": {
        "InsertObject(Product)",
        "UpdateAttribute(price, Product)",
        "UpdateAttribute(maxDiscount, Product)",
    },
    "NotTooPendingSales": {
        "UpdateAttribute(maxPendingAmount, Category)",
        "InsertLink(BelongsTo)",
        "InsertLink(Purchases)",
        "UpdateAttribute(amount, Sale)",
        "UpdateAttribute(paymentDate, Sale)",
    },
    "AtLeastThreeCustomers": {
        "DeleteLink(BelongsTo)",
        "InsertObject(Category)",
    },
    "NumberOfRestrictedProducts": {
        "InsertObject(RestrictedProduct)",
        "SpecializeObject(RestrictedProduct)",
    },
}


def _events(constraint):
    return {repr(link.structural_event) for link in violating_events(constraint)}


def _load(model):
    for name, text in INVARIANTS.items():
        constraint = parse_ocl(text, model)
        constraint.name = name
        model.add_constraint(constraint)
    return analyze_invariants(model)


def test_table_4_7_exact_event_sets():
    model = ecommerce_model()
    results = _load(model)
    by_name = {constraint.name: constraint for constraint in model.constraints}
    for name, expected in TABLE_4_7.items():
        assert _events(by_name[name]) == expected, name
        assert results[name].warnings == []
        assert all(not link.approximated for link in violating_events(by_name[name]))


def test_valid_ship_date_excludes_class_population_events():
    """A new sale or a deleted shipment cannot violate the invariant alone."""
    model = ecommerce_model()
    _load(model)
    events = _events(next(c for c in model.constraints if c.name == "ValidShipDate"))
    assert "InsertObject(Sale)" not in events
    assert "DeleteObject(Shipment)" not in events
    assert "DeleteObject(Sale)" not in events


def test_at_least_three_customers_excludes_delete_customer():
    model = ecommerce_model()
    _load(model)
    events = _events(next(c for c in model.constraints if c.name == "AtLeastThreeCustomers"))
    assert "DeleteObject(Customer)" not in events
    assert "DeleteObject(Category)" not in events


def test_not_flips_only_the_child_nodes_own_events():
    """``not self.oclIsTypeOf(T)`` flips the type test, not a deeper event."""
    model = ecommerce_model()
    typed = parse_ocl(
        "context RestrictedProduct inv Typed: self.oclIsTypeOf(RestrictedProduct)",
        model,
    )
    negated = parse_ocl(
        "context RestrictedProduct inv NotTyped: not self.oclIsTypeOf(RestrictedProduct)",
        model,
    )
    analyze_invariant(typed)
    analyze_invariant(negated)
    assert _events(typed) == {"GeneralizeObject(Product)"}
    assert _events(negated) == {"SpecializeObject(RestrictedProduct)"}


def test_select_marked_plus_flips_events_drawn_in_its_body():
    """A select that may grow turns a decreasing body navigation into an insertion."""
    model = ecommerce_model()
    constraint = parse_ocl(
        "context Category inv Growing: "
        "self.customer->select(c | c.sale->size() >= 1)->size() <= 5",
        model,
    )
    analyze_invariant(constraint)
    events = _events(constraint)
    assert "InsertLink(Purchases)" in events
    assert "DeleteLink(Purchases)" not in events
    assert "InsertLink(BelongsTo)" in events


def test_specialize_class_only_when_a_supertype_feature_is_read():
    """A new subclass instance has no value yet for its own attributes."""
    model = ecommerce_model()
    own = parse_ocl(
        "context RestrictedProduct inv OwnUnits: self.maxUnits > 0",
        model,
    )
    inherited = parse_ocl(
        "context RestrictedProduct inv InheritedPrice: self.price > 0",
        model,
    )
    analyze_invariant(own)
    analyze_invariant(inherited)
    assert _events(own) == {
        "InsertObject(RestrictedProduct)",
        "UpdateAttribute(maxUnits, RestrictedProduct)",
    }
    assert _events(inherited) == {
        "InsertObject(RestrictedProduct)",
        "SpecializeObject(RestrictedProduct)",
        "UpdateAttribute(price, Product)",
    }


def test_unknown_operator_is_over_approximated():
    model = ecommerce_model()
    constraint = parse_ocl(
        "context Category inv HasCustomer: self.customer->includes(self.customer)",
        model,
    )
    result = analyze_invariant(constraint)
    events = {repr(link.structural_event): link.approximated for link in violating_events(constraint)}
    assert events["InsertLink(BelongsTo)"] is True
    assert events["DeleteLink(BelongsTo)"] is True
    assert result.warnings


def test_preconditions_are_left_untouched():
    model = ecommerce_model()
    sale = next(t for t in model.types if t.name == "Sale")
    pre = Constraint("PayPre", sale, "context Sale pre Paid: self.amount > 0", "OCL")
    model.add_constraint(pre)
    _load(model)
    assert "PayPre" not in analyze_invariants(model)
    assert violating_events(pre) == set()
    result = analyze_invariant(pre)
    assert result.warnings
    assert violating_events(pre) == set()


def test_editing_the_expression_leaves_the_links_in_place():
    model = ecommerce_model()
    constraint = parse_ocl(INVARIANTS["CorrectProduct"], model)
    analyze_invariant(constraint)
    before = violating_events(constraint)
    assert before
    constraint.expression = constraint.expression + " "
    assert violating_events(constraint) == before
    assert all(link.constraint is constraint for link in before)


def test_event_dict_round_trip():
    model = ecommerce_model()
    constraint = parse_ocl(INVARIANTS["NotTooPendingSales"], model)
    constraint.name = "NotTooPendingSales"
    analyze_invariant(constraint)
    payloads = [event_to_dict(link) for link in violating_events(constraint)]
    set_violating_events(constraint, set())
    restored = {event_from_dict(payload, constraint, model) for payload in payloads}
    assert None not in restored
    assert _events(constraint) == TABLE_4_7["NotTooPendingSales"]
    assert {link.approximated for link in restored} == {False}


def test_reassigning_the_same_links_keeps_them():
    model = ecommerce_model()
    constraint = parse_ocl(INVARIANTS["AtLeastThreeCustomers"], model)
    analyze_invariant(constraint)
    links = violating_events(constraint)
    set_violating_events(constraint, links)
    assert violating_events(constraint) == links
    assert all(link.constraint is constraint for link in links)


def test_json_round_trip_keeps_events_and_omits_them_when_absent():
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        class_buml_to_json,
        process_class_diagram,
    )

    bare = ecommerce_model()
    bare_json = class_buml_to_json(bare)
    assert all("violatingEvents" not in element for element in bare_json["elements"].values())

    model = ecommerce_model()
    constraint = parse_ocl(INVARIANTS["CorrectProduct"], model)
    constraint.name = "CorrectProduct"
    # The editor stores the full ``context ... inv`` text, not the pretty-printed body.
    constraint.expression = INVARIANTS["CorrectProduct"]
    model.add_constraint(constraint)
    analyze_invariant(constraint)
    exported = class_buml_to_json(model)
    boxes = [element for element in exported["elements"].values() if element.get("type") == "ClassOCLConstraint"]
    assert len(boxes) == 1
    assert {event["kind"] for event in boxes[0]["violatingEvents"]} == {
        "InsertObject", "UpdateAttribute",
    }
    assert boxes[0]["violatingEventsSource"] == constraint.expression

    loaded = process_class_diagram({"title": "Ecommerce", "model": exported})
    restored = next(c for c in loaded.constraints if c.name == "CorrectProduct")
    assert _events(restored) == TABLE_4_7["CorrectProduct"]
    assert restored.expression == constraint.expression
    assert loaded.ocl_element_ids["CorrectProduct"]


def test_python_export_recreates_the_events(tmp_path):
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code

    model = ecommerce_model()
    constraint = parse_ocl(INVARIANTS["AtLeastThreeCustomers"], model)
    constraint.name = "AtLeastThreeCustomers"
    model.add_constraint(constraint)
    analyze_invariant(constraint)
    path = tmp_path / "ecommerce.py"
    domain_model_to_code(model, str(path))
    code = path.read_text(encoding="utf-8")
    assert "from besser.BUML.metamodel.structural_event import (" in code
    assert "InsertObject" in code
    assert "DeleteLink" in code
    namespace: dict = {}
    exec(code, namespace)
    recreated = next(c for c in namespace["domain_model"].constraints if c.name == "AtLeastThreeCustomers")
    assert _events(recreated) == TABLE_4_7["AtLeastThreeCustomers"]


def test_preanalysis_endpoint_matches_the_running_example():
    from besser.utilities.web_modeling_editor.backend.services.converters import class_buml_to_json
    from besser.utilities.web_modeling_editor.backend.services.ocl.incremental_preanalysis import (
        incremental_preanalysis,
    )

    model = ecommerce_model()
    _load(model)
    # The endpoint recomputes from the box text. The editor stores the full
    # header, which is what the JSON converter emits and parses.
    for constraint in list(model.constraints):
        constraint.expression = INVARIANTS[constraint.name]
        set_violating_events(constraint, set())
    payload = {"title": "Ecommerce", "model": class_buml_to_json(model)}
    result = incremental_preanalysis(payload)
    by_name = {item["name"]: item for item in result["constraints"]}
    assert set(by_name) == set(TABLE_4_7)
    for name, expected in TABLE_4_7.items():
        item = by_name[name]
        assert item["elementId"]
        assert item["source"]
        assert item["warnings"] == []
        got = set()
        for event in item["events"]:
            if event["kind"] == "UpdateAttribute":
                got.add(f"UpdateAttribute({event['attribute']}, {event['class']})")
            elif "association" in event:
                got.add(f"{event['kind']}({event['association']})")
            else:
                got.add(f"{event['kind']}({event['class']})")
        assert got == expected
        assert all(event["approximated"] is False for event in item["events"])


def test_a_second_run_ignores_events_stored_by_the_first():
    """A rerun computes from the constraint text, not from the previous list.

    Stored events are what the editor sends back. Rebuilding one that no
    longer matches the model used to warn (``could not restore InsertClass``)
    even though the new analysis replaced the list.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters import class_buml_to_json
    from besser.utilities.web_modeling_editor.backend.services.ocl.incremental_preanalysis import (
        incremental_preanalysis,
    )

    model = ecommerce_model()
    _load(model)
    for constraint in list(model.constraints):
        constraint.expression = INVARIANTS[constraint.name]
        set_violating_events(constraint, set())
    payload = {"title": "Ecommerce", "model": class_buml_to_json(model)}
    first = incremental_preanalysis(payload)
    assert first["warnings"] == []

    for item in first["constraints"]:
        box = payload["model"]["elements"][item["elementId"]]
        box["violatingEvents"] = item["events"] + [{
            "kind": "InsertClass",
            "class": "Sale",
            "constraint": item["name"],
            "approximated": False,
        }]
        box["violatingEventsSource"] = item["source"]

    second = incremental_preanalysis(payload)
    assert second["warnings"] == []
    assert first["constraints"] == second["constraints"]
    # The caller's diagram still holds the previous list; only the computation drops it.
    assert any(
        event.get("kind") == "InsertClass"
        for element in payload["model"]["elements"].values()
        for event in element.get("violatingEvents") or []
    )


def test_preanalysis_rejects_a_diagram_that_is_not_a_class_diagram():
    from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError
    from besser.utilities.web_modeling_editor.backend.services.ocl.incremental_preanalysis import (
        incremental_preanalysis,
    )

    with pytest.raises(ConversionError):
        incremental_preanalysis({"title": "Flow", "model": {"type": "StateMachineDiagram", "elements": {}}})


def test_event_from_dict_returns_none_for_a_missing_element():
    model = ecommerce_model()
    constraint = parse_ocl(INVARIANTS["CorrectProduct"], model)
    missing = event_from_dict(
        {"kind": "InsertObject", "class": "Missing", "approximated": False},
        constraint,
        model,
    )
    assert missing is None
    bad_kind = event_from_dict({"kind": "RenameClass", "class": "Product"}, constraint, model)
    assert bad_kind is None
