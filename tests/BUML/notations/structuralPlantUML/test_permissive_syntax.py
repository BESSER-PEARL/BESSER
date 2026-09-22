"""Regression tests for issue #236 — more flexibility in the PlantUML importer.

Covers:
- Single-dash ``-`` as an association edge (horizontal-layout shorthand).
- Alternative primitive type aliases: ``integer`` → int, ``string`` → str,
  ``boolean`` → bool.
- Trailing direction hint (``>`` / ``<``) after an association label.
"""

import textwrap

import pytest

from besser.BUML.metamodel.structural import DomainModel, IntegerType, StringType, BooleanType
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml


def _write_and_parse(tmp_path, source: str) -> DomainModel:
    path = tmp_path / "model.plantuml"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return plantuml_to_buml(plantUML_model_path=str(path))


def test_single_dash_association_with_label_arrow(tmp_path):
    """The example from issue #236 must parse without error."""
    model = _write_and_parse(tmp_path, """\
        @startuml
        class Source {
            id : integer
            name : string
        }
        class Target {
            id : integer
        }
        Source - "1..*" Target : connection >
        @enduml
        """)

    assert {c.name for c in model.get_classes()} == {"Source", "Target"}

    associations = list(model.associations)
    assert len(associations) == 1
    assoc = associations[0]
    assert assoc.name == "connection"

    ends_by_name = {end.type.name: end for end in assoc.ends}
    assert ends_by_name["Source"].multiplicity.min == 1
    assert ends_by_name["Source"].multiplicity.max == 1
    assert ends_by_name["Target"].multiplicity.min == 1
    # "*" is stored as UNLIMITED_MAX_MULTIPLICITY (9999) per project convention.
    assert ends_by_name["Target"].multiplicity.max == 9999


def test_primitive_type_aliases(tmp_path):
    """``integer``/``string``/``boolean`` must resolve to the same B-UML types
    as ``int``/``str``/``bool``."""
    model = _write_and_parse(tmp_path, """\
        @startuml
        class A {
            a_int : integer
            a_str : string
            a_bool : boolean
        }
        class B {
            b_int : int
            b_str : str
            b_bool : bool
        }
        @enduml
        """)

    attrs_a = {attr.name: attr.type for attr in model.get_class_by_name("A").attributes}
    attrs_b = {attr.name: attr.type for attr in model.get_class_by_name("B").attributes}

    assert attrs_a["a_int"] is IntegerType
    assert attrs_a["a_str"] is StringType
    assert attrs_a["a_bool"] is BooleanType
    # Aliases must resolve to the exact same singleton instance.
    assert attrs_a["a_int"] is attrs_b["b_int"]
    assert attrs_a["a_str"] is attrs_b["b_str"]
    assert attrs_a["a_bool"] is attrs_b["b_bool"]


def test_single_dash_keeps_double_dash_working(tmp_path):
    """Adding ``-`` as an alternative must not regress ``--`` associations."""
    model = _write_and_parse(tmp_path, """\
        @startuml
        class A {
            id : int
        }
        class B {
            id : int
        }
        A "1" -- "1..*" B : rel
        @enduml
        """)
    assert len(list(model.associations)) == 1


@pytest.mark.parametrize("arrow", ["<", ">"])
def test_label_direction_hint_is_ignored(tmp_path, arrow):
    """Trailing ``<`` / ``>`` after an association label is a rendering hint,
    not semantic — it must parse and produce the same model."""
    model = _write_and_parse(tmp_path, f"""\
        @startuml
        class A {{
            id : int
        }}
        class B {{
            id : int
        }}
        A - B : rel {arrow}
        @enduml
        """)
    assert len(list(model.associations)) == 1
    assert list(model.associations)[0].name == "rel"
