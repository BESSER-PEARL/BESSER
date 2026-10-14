"""A method's return type must survive the editor's newer JSON format.

Attributes have long supported the format where the type lives in its own
``attributeType`` property and the name is bare. Methods never got that
branch: they parsed the return type out of the signature string only, so a
method written as ``renew()`` with ``attributeType: "bool"`` arrived with
``type=None``.

That matters beyond tidiness. The same class of omission on the PARAMETER
list caused 16 of 21 failures on one evaluation case: an absent key reads to
the agent as "not shown", not "there is none", so it invented a required
argument for a zero-argument method. The library spec says each action
"reports back whether it succeeded" - the return type is the contract for
that, and the agent could not see it.
"""

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


def _diagram(method: dict) -> dict:
    """A v4 class diagram with one ``Loan`` class holding the given method row."""
    return {
        "title": "T",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {"id": "c1", "type": "class",
                 "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                 "data": {"name": "Loan", "stereotype": None, "attributes": [],
                          "methods": [{"id": "m1", **method}]}},
            ],
            "edges": [],
        },
    }


def _only_method(diagram):
    model = process_class_diagram(diagram)
    cls = next(c for c in model.get_classes() if c.name == "Loan")
    return next(iter(cls.methods))


def test_the_separate_attribute_type_property_is_read():
    """The editor's newer shape: bare name, type in its own property."""
    method = _only_method(_diagram({"name": "renew()", "attributeType": "bool"}))

    assert method.type is not None, "the return type was dropped"
    assert getattr(method.type, "name", method.type) == "bool"


def test_the_legacy_return_type_property_is_read():
    method = _only_method(_diagram({"name": "renew()", "returnType": "int"}))

    assert getattr(method.type, "name", method.type) == "int"


def test_a_type_in_the_signature_still_wins():
    """The signature is the older format and must keep working unchanged."""
    method = _only_method(
        _diagram({"name": "renew(): date", "attributeType": "bool"})
    )

    assert getattr(method.type, "name", method.type) == "date"


def test_a_method_with_no_declared_type_stays_untyped():
    method = _only_method(_diagram({"name": "renew()"}))

    assert method.type is None


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_type_property_is_not_mistaken_for_one(empty):
    method = _only_method(_diagram({"name": "renew()", "attributeType": empty}))

    assert method.type is None


def test_a_stale_type_property_does_not_sink_the_whole_diagram():
    """Older editor saves left a fragment of the signature in ``attributeType``.

    The shipped ``library_full_stack`` template holds exactly this: name
    ``decrease_stock(qty: int)``, ``attributeType: "int): any"``. Reading it as
    a return type raised ConversionError, so spec-driven assembly dropped the
    domain model and, with it, the GUI. Before the property was read at all the
    method converted untyped, which is what it must still do.
    """
    method = _only_method(_diagram({
        "name": "+ decrease_stock(qty: int)", "attributeType": "int): any",
    }))

    assert method.name == "decrease_stock"
    assert method.type is None


@pytest.mark.parametrize("field", ["returnType", "attributeType"])
def test_the_any_placeholder_means_untyped(field):
    """``any`` is the v4 editor's "no explicit return type" placeholder.

    Every method the inspector creates carries ``returnType: "any"``, and
    ``buml_to_json`` writes ``"any"`` for an untyped method, so reading it as
    ``PrimitiveDataType('any')`` would type every default method (``-> any``,
    ``Object`` instead of ``void``) and break JSON -> B-UML -> JSON -> B-UML
    idempotence. See ``docs/source/migrations/uml-v4-shape.md``.
    """
    method = _only_method(_diagram({"name": "renew", field: "any"}))

    assert method.type is None


def test_an_untyped_method_stays_untyped_across_a_round_trip():
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        class_buml_to_json,
    )

    domain = process_class_diagram(_diagram({"name": "renew", "returnType": "any", "attributeType": "any"}))
    again = process_class_diagram({"title": "T", "model": class_buml_to_json(domain)})
    cls = next(c for c in again.get_classes() if c.name == "Loan")

    assert next(iter(cls.methods)).type is None
