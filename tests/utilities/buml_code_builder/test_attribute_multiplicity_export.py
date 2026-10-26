"""Attribute multiplicity and ``is_read_only`` survive a B-UML ``.py`` export (#458).

The builder wrote neither, so ``emails: str [0..*]`` came back as ``[1..1]``
and a read-only attribute came back writable, in the editor's ``.py`` import
as in a plain Python import. The Spring, JSON Schema and SQL generators read
attribute multiplicity, so the reloaded model generated different code.
"""
from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    Class,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
    UNLIMITED_MAX_MULTIPLICITY,
)
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    parse_buml_content,
)


def _reload(model, tmp_path):
    path = tmp_path / "model.py"
    domain_model_to_code(model, str(path))
    source = path.read_text(encoding="utf-8")
    return source, parse_buml_content(source)


def _attributes(model, class_name):
    return {a.name: a for a in model.get_class_by_name(class_name).attributes}


def test_attribute_multiplicity_and_read_only_survive_the_export(tmp_path):
    person = Class(name="Person", attributes={
        Property(name="emails", type=StringType, multiplicity=Multiplicity(0, "*")),
        Property(name="nicknames", type=StringType, multiplicity=Multiplicity(1, 3)),
        Property(name="code", type=IntegerType, is_id=True, is_read_only=True),
        Property(name="name", type=StringType),
    })
    source, reloaded = _reload(DomainModel(name="People", types={person}), tmp_path)
    attributes = _attributes(reloaded, "Person")

    assert (attributes["emails"].multiplicity.min, attributes["emails"].multiplicity.max) == (
        0, UNLIMITED_MAX_MULTIPLICITY)
    assert (attributes["nicknames"].multiplicity.min, attributes["nicknames"].multiplicity.max) == (1, 3)
    assert attributes["code"].is_read_only is True
    assert attributes["code"].is_id is True
    # The default (1..1, writable) adds nothing to the line.
    assert 'Property(name="name", type=StringType)' in source


def test_association_class_attributes_keep_them_too(tmp_path):
    student, course = Class(name="Student"), Class(name="Course")
    association = BinaryAssociation(name="enrolment", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(0, "*")),
    })
    enrolment = AssociationClass(name="Enrolment", association=association, attributes={
        Property(name="grades", type=IntegerType, multiplicity=Multiplicity(0, 5), is_read_only=True),
    })
    model = DomainModel(name="School", types={student, course, enrolment}, associations={association})
    _, reloaded = _reload(model, tmp_path)
    grades = _attributes(reloaded, "Enrolment")["grades"]

    assert (grades.multiplicity.min, grades.multiplicity.max) == (0, 5)
    assert grades.is_read_only is True
