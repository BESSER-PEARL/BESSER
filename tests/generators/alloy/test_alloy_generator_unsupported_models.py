import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, AssociationClass, Multiplicity, Constraint,
    IntegerType, TimeType, DateTimeType,
)
from besser.generators.alloy import AlloyGenerator, AlloyTranslationError


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


@pytest.fixture(params=[TimeType, DateTimeType])
def meeting_model(request) -> DomainModel:
    """``Meeting(start: time)`` and ``Meeting(start: datetime)``: time and datetime attributes have no Alloy
    translation."""
    meeting = Class("Meeting", attributes={Property("start", request.param)})
    return DomainModel("Meeting_model", types={meeting})


@pytest.fixture
def counter_model() -> DomainModel:
    """``Counter(value: int)`` with an OCL invariant using real division, which has no Alloy translation."""
    counter = Class("Counter", attributes={Property("value", IntegerType)})
    half = Constraint("half", counter, "context Counter inv half: self.value / 2 > 1", "OCL")
    return DomainModel("Counter_model", types={counter}, constraints={half})


def test_association_classes_are_unsupported(enrollment_model, tmp_path):
    generator = AlloyGenerator(enrollment_model, output_dir=str(tmp_path))
    with pytest.raises(AlloyTranslationError, match="Association classes are not supported"):
        generator.generate()


def test_attribute_types_without_translation_are_unsupported(meeting_model, tmp_path):
    (start,) = meeting_model.get_class_by_name("Meeting").attributes
    generator = AlloyGenerator(meeting_model, output_dir=str(tmp_path))
    with pytest.raises(AlloyTranslationError, match=f"Attributes of type '{start.type.name}' are not supported"):
        generator.generate()


def test_unsupported_ocl_is_a_translation_error(counter_model, tmp_path):
    generator = AlloyGenerator(counter_model, output_dir=str(tmp_path))
    with pytest.raises(AlloyTranslationError, match="Real division"):
        generator.generate()
