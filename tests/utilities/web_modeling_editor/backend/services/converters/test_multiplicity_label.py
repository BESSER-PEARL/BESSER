"""Tests for UML multiplicity-label formatting in the BUML -> JSON converter.

Covers issue #3: an association end with an exact multiplicity (min == max)
must render as a single value (``1..1`` -> ``1``) to match UML convention, while
ranges and unbounded upper bounds are left intact.
"""

import pytest

from besser.BUML.metamodel.structural import (
    Class, Property, Multiplicity, BinaryAssociation, DomainModel, StringType,
    UNLIMITED_MAX_MULTIPLICITY,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
    _format_multiplicity_label,
)


@pytest.mark.parametrize(
    "min_val, max_val, expected",
    [
        (1, 1, "1"),
        (5, 5, "5"),
        (0, 1, "0..1"),
        (2, 5, "2..5"),
        (1, UNLIMITED_MAX_MULTIPLICITY, "1..*"),
        (0, UNLIMITED_MAX_MULTIPLICITY, "0..*"),
    ],
)
def test_format_multiplicity_label(min_val, max_val, expected):
    assert _format_multiplicity_label(Multiplicity(min_val, max_val)) == expected


def test_exact_multiplicity_collapses_in_converter():
    """A 1..1 end serialises to '1'; an unbounded end keeps its range."""
    a = Class(name="A", attributes={Property(name="x", type=StringType)})
    b = Class(name="B", attributes={Property(name="y", type=StringType)})
    assoc = BinaryAssociation(
        name="AB",
        ends={
            Property(name="a", type=a, multiplicity=Multiplicity(1, 1)),
            Property(name="b", type=b, multiplicity=Multiplicity(1, "*")),
        },
    )
    model = DomainModel(name="M", types={a, b}, associations={assoc})

    result = class_buml_to_json(model)
    relationships = result.get("relationships") or (result.get("model") or {}).get("relationships", {})
    rels = list(relationships.values())
    assert len(rels) == 1

    mults = {rels[0]["source"]["multiplicity"], rels[0]["target"]["multiplicity"]}
    assert "1" in mults           # 1..1 collapsed (issue #3)
    assert "1..1" not in mults
    assert "1..*" in mults        # unbounded range untouched
