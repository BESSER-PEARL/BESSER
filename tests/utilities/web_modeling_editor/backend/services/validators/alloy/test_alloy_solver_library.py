import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, StringType, IntegerType, DateType,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult
from tests.utilities.web_modeling_editor.backend.services.validators.alloy.utils_alloy_solver import (
    check_consistency, generate_object_models, needs_alloy,
)

# The tests run the Alloy Analyzer
pytestmark = needs_alloy


@pytest.fixture
def library_model() -> DomainModel:
    """``Library`` 1 -- * ``Book`` * -- 1..* ``Author``."""
    library = Class("Library", attributes={Property("name", StringType), Property("address", StringType)})
    book = Class("Book", attributes={
        Property("title", StringType), Property("pages", IntegerType), Property("release", DateType),
    })
    author = Class("Author", attributes={
        Property("name", StringType), Property("email", StringType), Property("birtdate", DateType),
    })
    lib_book = BinaryAssociation("lib_book_assoc", ends={
        Property("locatedIn", library, multiplicity=Multiplicity(1, 1)),
        Property("has", book, multiplicity=Multiplicity(0, "*")),
    })
    book_author = BinaryAssociation("book_author_assoc", ends={
        Property("writtenBy", author, multiplicity=Multiplicity(1, "*")),
        Property("publishes", book, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Library_model", types={library, book, author}, associations={lib_book, book_author})


def test_check_consistency(library_model):
    result, _ = check_consistency(library_model, scope=3)
    assert result == AlloyResult.SAT


def test_generate_four_object_models(library_model):
    result, object_models = generate_object_models(library_model, scope=3, num_models=4)
    assert result == AlloyResult.SAT
    assert len(object_models) == 4
