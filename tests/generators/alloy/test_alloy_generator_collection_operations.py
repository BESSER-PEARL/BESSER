import pytest

from besser.BUML.metamodel.structural import BinaryAssociation, Class, DomainModel, Multiplicity, Property
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


@pytest.fixture
def library_model() -> DomainModel:
    """``Library`` with ``books`` 0..*."""
    library, book = Class("Library"), Class("Book")
    located_in = BinaryAssociation("located_in", ends={
        Property("library", library, multiplicity=Multiplicity(1, 1)),
        Property("books", book, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Library", types={library, book}, associations={located_in})


def test_is_empty(library_model):
    spec = translate(library_model, "Library", "self.books->isEmpty()", scope=5)
    assert "all self:this/Library|(no self.Library_books)" in spec.facts


def test_not_empty(library_model):
    spec = translate(library_model, "Library", "self.books->notEmpty()", scope=5)
    assert "all self:this/Library|(some self.Library_books)" in spec.facts


def test_size_of_collect_as_set(library_with_ocl_v2):  # noqa: F811
    spec = translate(
        library_with_ocl_v2, "Library", "self.books->collect(b | b.authors)->asSet()->size() > 5", scope=5
    )
    # a body navigating from the iterator collects with a join
    assert "all self:this/Library|(#(((self.Library_books).Book_authors)) > 5)" in spec.facts


def test_collect_of_a_selection(library_with_ocl_v2):  # noqa: F811
    spec = translate(
        library_with_ocl_v2,
        "Library",
        "self.books->collect(b | b.authors->select(a | a.books->size() > 1))->asSet()->size() > 5",
        scope=5,
    )
    # any other body collects the set of its values
    assert "all self:this/Library|(#({collected: Author | some b : self.Library_books | " \
        "collected in {select0[b.Book_authors]}}) > 5)" in spec.facts
