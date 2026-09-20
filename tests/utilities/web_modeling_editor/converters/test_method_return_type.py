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
    return {
        "title": "T",
        "model": {
            "elements": {
                "c1": {"id": "c1", "type": "Class", "name": "Loan",
                       "attributes": [], "methods": ["m1"]},
                "m1": {"id": "m1", "type": "ClassMethod", **method},
            },
            "relationships": {},
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
