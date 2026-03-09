import os
import pytest
from besser.generators.rest_api import RESTAPIGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation,
    Multiplicity, StringType, IntegerType
)


@pytest.fixture
def self_association_model():
    """Model with an Employee class that has a self-association (Manages)."""
    Employee = Class(name="Employee")

    employee_name: Property = Property(name="name", type=StringType)
    Employee.attributes = {employee_name}

    manages: BinaryAssociation = BinaryAssociation(
        name="Manages",
        ends={
            Property(name="manager", type=Employee, multiplicity=Multiplicity(0, 1)),
            Property(name="subordinates", type=Employee, multiplicity=Multiplicity(0, "*"))
        }
    )

    model = DomainModel(
        name="Self_Assoc_Model",
        types={Employee},
        associations={manages},
        generalizations={}
    )

    return model


@pytest.fixture
def library_book_model():
    """Standard Library-Book model without self-associations."""
    Library = Class(name="Library")
    Book = Class(name="Book")

    library_name: Property = Property(name="name", type=StringType)
    Library.attributes = {library_name}

    book_title: Property = Property(name="title", type=StringType)
    book_pages: Property = Property(name="pages", type=IntegerType)
    Book.attributes = {book_title, book_pages}

    has: BinaryAssociation = BinaryAssociation(
        name="Has",
        ends={
            Property(name="Book_end", type=Book, multiplicity=Multiplicity(0, "*")),
            Property(name="Library_end", type=Library, multiplicity=Multiplicity(1, 1))
        }
    )

    model = DomainModel(
        name="Library_Model",
        types={Library, Book},
        associations={has},
        generalizations={}
    )

    return model


def test_self_association_does_not_crash(self_association_model, tmp_path):
    """Generating a REST API from a model with a self-association should not crash
    and should produce at least one output file."""
    output_dir = str(tmp_path / "output_self")
    generator = RESTAPIGenerator(
        model=self_association_model,
        output_dir=output_dir
    )
    generator.generate()

    generated_files = os.listdir(output_dir)
    assert len(generated_files) > 0, "Expected at least one generated file in output directory"

    # The generator should produce rest_api.py (non-backend mode)
    rest_api_file = os.path.join(output_dir, "rest_api.py")
    assert os.path.isfile(rest_api_file), "rest_api.py should be generated"

    with open(rest_api_file, "r", encoding="utf-8") as f:
        content = f.read()

    # The generated code should reference the Employee class
    assert "employee" in content.lower(), "Generated REST API should contain Employee references"
    # Should contain FastAPI endpoint definitions
    assert "app" in content, "Generated code should define a FastAPI app"


def test_normal_association_does_not_crash(library_book_model, tmp_path):
    """Generating a REST API from a normal Library-Book model should not crash
    and should produce at least one output file."""
    output_dir = str(tmp_path / "output_normal")
    generator = RESTAPIGenerator(
        model=library_book_model,
        output_dir=output_dir
    )
    generator.generate()

    generated_files = os.listdir(output_dir)
    assert len(generated_files) > 0, "Expected at least one generated file in output directory"

    # The generator should produce rest_api.py (non-backend mode)
    rest_api_file = os.path.join(output_dir, "rest_api.py")
    assert os.path.isfile(rest_api_file), "rest_api.py should be generated"

    with open(rest_api_file, "r", encoding="utf-8") as f:
        content = f.read()

    # The generated code should reference both Library and Book
    assert "library" in content.lower(), "Generated REST API should contain Library references"
    assert "book" in content.lower(), "Generated REST API should contain Book references"
    # Should contain FastAPI endpoint definitions
    assert "app" in content, "Generated code should define a FastAPI app"
