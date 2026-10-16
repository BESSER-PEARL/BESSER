"""Element names that are not Python identifiers must not reach the generated
domain code as raw variable names.

``NamedElement.name`` only rejects spaces and hyphens, so a class, attribute or
generalization name can hold ``;``, ``(`` or quotes. The builder wrote those
names as variables, so the generated file could be made to run arbitrary code.
"""
import ast
import builtins

import pytest

from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Multiplicity,
    Property,
    StringType,
)
from besser.utilities.buml_code_builder.common import (
    NameDispenser,
    attribute_var_name,
    identifier_part,
    method_var_name,
    safe_class_name,
)
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code

HOSTILE = 'A;print("PWN")#'


def _generate(model, tmp_path, objectmodel=None) -> str:
    path = tmp_path / "model.py"
    domain_model_to_code(model, str(path), objectmodel=objectmodel)
    return path.read_text(encoding="utf-8")


def _run(code: str) -> dict:
    """Execute ``code`` with ``print`` trapped, failing if the payload ran."""
    calls = []
    namespace = {"__builtins__": {**vars(builtins), "print": lambda *a, **k: calls.append(a)}}
    exec(code, namespace)
    assert calls == [], "a name escaped into executable code"
    return namespace


def test_identifier_part_keeps_valid_names_and_sanitises_the_rest():
    assert identifier_part("Book") == "Book"
    assert identifier_part("first_name") == "first_name"
    sanitised = identifier_part(HOSTILE)
    assert sanitised.isidentifier()
    assert sanitised.startswith("A_print__PWN___")
    # Stable, and different names never share a variable.
    assert identifier_part(HOSTILE) == sanitised
    assert identifier_part("A;b") != identifier_part("A:b")


def test_safe_class_name_sanitises_non_identifiers():
    assert safe_class_name("Person") == "Person"
    assert safe_class_name(HOSTILE).isidentifier()
    assert safe_class_name("Multiplicity") == "Multiplicity_"


def test_shared_variable_helpers_match_normal_names():
    assert attribute_var_name("Book", "title") == "Book_title"
    assert method_var_name("Book", "total(x: int)") == "Book_m_total"
    assert attribute_var_name(HOSTILE, "x").isidentifier()


def test_hostile_class_attribute_and_generalization_names_do_not_execute(tmp_path):
    hostile = Class(name=HOSTILE, attributes={Property(name='x;print("A")#', type=StringType)})
    parent = Class(name='P;print("G")#')
    model = DomainModel(
        name="Hostile",
        types={hostile, parent},
        generalizations={Generalization(general=parent, specific=hostile)},
    )
    code = _generate(model, tmp_path)
    ast.parse(code)
    rebuilt = _run(code)["domain_model"]
    assert {cls.name for cls in rebuilt.get_classes()} == {HOSTILE, 'P;print("G")#'}
    rebuilt_class = rebuilt.get_class_by_name(HOSTILE)
    assert {attr.name for attr in rebuilt_class.attributes} == {'x;print("A")#'}
    assert len(rebuilt.generalizations) == 1


def test_class_named_after_an_import_does_not_shadow_it(tmp_path):
    clash = Class(name="Multiplicity")
    other = Class(name="Other")
    assoc = BinaryAssociation(name="link", ends={
        Property(name="m", type=clash),
        Property(name="others", type=other, multiplicity=Multiplicity(0, "*")),
    })
    model = DomainModel(name="Clash", types={clash, other}, associations={assoc})
    rebuilt = _run(_generate(model, tmp_path))["domain_model"]
    assert {cls.name for cls in rebuilt.get_classes()} == {"Multiplicity", "Other"}


def test_normal_model_output_is_unchanged(tmp_path):
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    code = _generate(DomainModel(name="Library", types={book}), tmp_path)
    assert 'Book = Class(name="Book")\n' in code
    assert "Book_title: Property = Property(" in code
    assert "Book.attributes={Book_title}" in code


def test_object_with_a_non_identifier_slot_name_is_refused(tmp_path):
    cls = Class(name="Book", attributes={Property(name='t;print("O")#', type=StringType)})
    model = DomainModel(name="Library", types={cls})
    attr = next(iter(cls.attributes))
    book = Object(name="b1", classifier=cls, slots=[AttributeLink(attribute=attr, value=DataValue(
        classifier=StringType, value="x"))])
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        _generate(model, tmp_path, objectmodel=ObjectModel(name="Objects", objects={book}))


def test_name_dispenser_unprefixed_mode_keeps_case():
    taken = {"Book", "datetime"}
    dispenser = NameDispenser(taken, type_prefix=False, lowercase=False)
    first = Class(name="Book")
    second = Class(name="datetime")
    assert dispenser.name_for(first) == "Book_1"
    assert dispenser.name_for(second) == "datetime_1"
    assert dispenser.name_for(first) == "Book_1"
