import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, IntegerType, Multiplicity, Property, StringType
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec, set_constraint, translate


@pytest.fixture
def spec() -> AlloyTestSpec:
    """``Book(isbn: str, id; note: str, optional; subtitle: str [0..1]; title: str)``."""
    book = Class("Book", attributes={
        Property("isbn", StringType, is_id=True),
        Property("note", StringType, is_optional=True),
        Property("subtitle", StringType, multiplicity=Multiplicity(0, 1)),
        Property("title", StringType),
    })
    model = DomainModel("Books", types={book})
    return translate(model, scope=5)


@pytest.fixture
def people_model() -> DomainModel:
    """``Person(age: int, optional)``."""
    person = Class("Person", attributes={Property("age", IntegerType, is_optional=True)})
    return DomainModel("People", types={person})


def test_optional_attributes_may_have_no_value(spec):
    assert "Book_note: lone Str" in spec.fields["Book"]
    assert "Book_subtitle: lone Str" in spec.fields["Book"]
    assert "Book_title: Str" in spec.fields["Book"]


def test_id_attributes_are_unique(spec):
    assert "all disj a, b: Book | a.Book_isbn != b.Book_isbn" in spec.facts


def test_comparison_on_optional_attribute_is_guarded(people_model):
    set_constraint(people_model, "Person", "self.age < 100")
    spec = translate(people_model, scope=5)
    assert "all self:this/Person|((some self.Person_age) and (self.Person_age < 100))" in spec.facts


def test_arithmetic_on_optional_attribute_is_guarded(people_model):
    set_constraint(people_model, "Person", "self.age + 1 > 0")
    spec = translate(people_model, scope=5)
    assert "all self:this/Person|((some self.Person_age) and (plus[self.Person_age, 1] > 0))" in spec.facts
