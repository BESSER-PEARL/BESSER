import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.alloy.alloy_translator import BumlToAlloy
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec


@pytest.fixture
def spec() -> AlloyTestSpec:
    """``Book(isbn: str, id; note: str, optional; title: str)``."""
    book = Class("Book", attributes={
        Property("isbn", StringType, is_id=True),
        Property("note", StringType, is_optional=True),
        Property("title", StringType),
    })
    model = DomainModel("Books", types={book})
    return AlloyTestSpec(BumlToAlloy(model, scope=5, seed=0).translate().render())


def test_optional_attributes_may_have_no_value(spec):
    assert "Book_note: lone Str" in spec.fields["Book"]
    assert "Book_title: Str" in spec.fields["Book"]


def test_id_attributes_are_unique(spec):
    assert "all disj a, b: Book | a.Book_isbn != b.Book_isbn" in spec.facts
