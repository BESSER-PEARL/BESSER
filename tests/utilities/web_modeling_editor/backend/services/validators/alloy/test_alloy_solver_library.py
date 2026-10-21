import os

import pytest

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, StringType, IntegerType, DateType,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult, AlloySolver

# The tests run the Alloy Analyzer, located through these environment variables
pytestmark = pytest.mark.skipif(
    not (os.getenv("JAVA_HOME") and os.getenv("BESSER_ALLOY_JAR")),
    reason="JAVA_HOME and BESSER_ALLOY_JAR must be set to run the Alloy Analyzer",
)


@pytest.fixture
def library_model() -> DomainModel:
    """The model of ``run_alloy_solver_object_generation_API.py``: ``Library`` 1 -- * ``Book`` * -- 1..* ``Author``."""
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


def test_check_consistency(library_model, tmp_path):
    solver = AlloySolver(library_model, output_dir=str(tmp_path))
    assert solver.check_consistency(scope=3) == AlloyResult.SAT


def test_generate_four_object_models(library_model, tmp_path):
    solver = AlloySolver(library_model, output_dir=str(tmp_path))
    result, object_models = solver.generate_object_models(scope=3, num_models=4)
    assert result == AlloyResult.SAT
    assert len(object_models) == 4
