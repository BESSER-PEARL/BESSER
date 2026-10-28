import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Association, BinaryAssociation, AssociationClass, Generalization, Multiplicity,
    IntegerType, FloatType, TimeType, DateTimeType,
)
from besser.generators.alloy import AlloyTranslationError
from tests.generators.alloy.utils_alloy_translation import set_constraint, translate


@pytest.fixture
def enrollment_model() -> DomainModel:
    """``Student`` 0..* -- 0..* ``Course``, with the association class ``Enrollment(grade: int)``."""
    student = Class("Student")
    course = Class("Course")
    enrolls = BinaryAssociation("enrolls", ends={
        Property("students", student, multiplicity=Multiplicity(0, "*")),
        Property("courses", course, multiplicity=Multiplicity(0, "*")),
    })
    enrollment = AssociationClass("Enrollment", attributes={Property("grade", IntegerType)}, association=enrolls)
    return DomainModel("Enrollment_model", types={student, course, enrollment}, associations={enrolls})


@pytest.fixture(params=[TimeType, DateTimeType, FloatType])
def meeting_model(request) -> DomainModel:
    """``Meeting(start: time)``, ``Meeting(start: datetime)`` and ``Meeting(start: float)``: time, datetime and
    float attributes have no Alloy translation."""
    meeting = Class("Meeting", attributes={Property("start", request.param)})
    return DomainModel("Meeting_model", types={meeting})


@pytest.fixture
def counter_model() -> DomainModel:
    """``Counter(value: int)``."""
    counter = Class("Counter", attributes={Property("value", IntegerType)})
    return DomainModel("Counter_model", types={counter})


def test_association_classes_are_unsupported(enrollment_model):
    with pytest.raises(AlloyTranslationError, match="Association classes are not supported"):
        translate(enrollment_model, scope=5)


def test_multiple_inheritance_is_unsupported():
    land, water, frog = Class("Land"), Class("Water"), Class("Frog")
    model = DomainModel("Amphibians", types={land, water, frog}, generalizations={
        Generalization(general=land, specific=frog), Generalization(general=water, specific=frog),
    })
    with pytest.raises(AlloyTranslationError, match=r"Class 'Frog' has several parents \(Land, Water\)"):
        translate(model, scope=5)


def test_associations_with_more_than_two_ends_are_unsupported():
    student, course, room = Class("Student"), Class("Course"), Class("Room")
    takes = Association("takes", ends={
        Property("student", student), Property("course", course), Property("room", room),
    })
    model = DomainModel("School", types={student, course, room}, associations={takes})
    with pytest.raises(AlloyTranslationError, match="Association 'takes' has more than two ends"):
        translate(model, scope=5)


def test_attribute_types_without_translation_are_unsupported(meeting_model):
    (start,) = meeting_model.get_class_by_name("Meeting").attributes
    with pytest.raises(AlloyTranslationError, match=f"Attributes of type '{start.type.name}' are not supported"):
        translate(meeting_model, scope=5)


def test_unsupported_ocl_is_a_translation_error(counter_model):
    # Real division has no Alloy translation
    set_constraint(counter_model, "Counter", "self.value / 2 > 1")
    with pytest.raises(AlloyTranslationError, match="Real division"):
        translate(counter_model, scope=5)
