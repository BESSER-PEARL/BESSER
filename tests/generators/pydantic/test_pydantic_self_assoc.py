import os
import pytest
from besser.generators.pydantic_classes import PydanticGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation,
    Multiplicity, StringType, IntegerType
)


@pytest.fixture
def employee_model():
    """Model with a self-association: Employee manages other Employees."""
    Employee = Class(name="Employee")

    emp_name: Property = Property(name="name", type=StringType)
    Employee.attributes = {emp_name}

    manages: BinaryAssociation = BinaryAssociation(
        name="Manages",
        ends={
            Property(name="manager", type=Employee, multiplicity=Multiplicity(0, 1)),
            Property(name="subordinates", type=Employee, multiplicity=Multiplicity(0, "*"))
        }
    )

    model = DomainModel(
        name="Employee_Model",
        types={Employee},
        associations={manages},
        generalizations={}
    )
    return model


@pytest.fixture
def library_model():
    """Simple Library-Book model with a normal (non-self) association."""
    Library = Class(name="Library")
    Book = Class(name="Book")

    lib_name: Property = Property(name="name", type=StringType)
    Library.attributes = {lib_name}

    book_title: Property = Property(name="title", type=StringType)
    Book.attributes = {book_title}

    has: BinaryAssociation = BinaryAssociation(
        name="Has",
        ends={
            Property(name="books", type=Book, multiplicity=Multiplicity(0, "*")),
            Property(name="library", type=Library, multiplicity=Multiplicity(1, 1))
        }
    )

    model = DomainModel(
        name="Library_Model",
        types={Library, Book},
        associations={has},
        generalizations={}
    )
    return model


def test_self_association_does_not_crash(employee_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = PydanticGenerator(model=employee_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "pydantic_classes.py")
    assert os.path.isfile(output_file)


def test_self_association_generates_both_fields(employee_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = PydanticGenerator(model=employee_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    assert "manager" in generated_code, "Expected 'manager' field in generated code"
    assert "subordinates" in generated_code, "Expected 'subordinates' field in generated code"


def test_normal_association(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = PydanticGenerator(model=library_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "pydantic_classes.py")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        generated_code = f.read()

    assert "books" in generated_code, "Expected 'books' field in generated code"
