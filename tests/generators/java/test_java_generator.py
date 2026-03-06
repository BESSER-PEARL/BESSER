import os
import pytest
from besser.generators.java_classes import JavaGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation, Multiplicity,
    StringType, IntegerType
)


@pytest.fixture
def library_model():
    """Standard model with normal associations."""
    Book = Class(name="Book")
    Library = Class(name="Library")

    Book_title: Property = Property(name="title", type=StringType)
    Book_pages: Property = Property(name="pages", type=IntegerType)
    Book.attributes = {Book_title, Book_pages}

    Library_name: Property = Property(name="name", type=StringType)
    Library.attributes = {Library_name}

    has: BinaryAssociation = BinaryAssociation(
        name="Has",
        ends={
            Property(name="books", type=Book, multiplicity=Multiplicity(0, "*")),
            Property(name="library", type=Library, multiplicity=Multiplicity(1, 1))
        }
    )

    model = DomainModel(
        name="Library_Model",
        types={Book, Library},
        associations={has},
        generalizations={}
    )
    return model


@pytest.fixture
def self_assoc_model():
    """Model with a self-association (Employee manages Employee)."""
    Employee = Class(name="Employee")

    Employee_name: Property = Property(name="name", type=StringType)
    Employee.attributes = {Employee_name}

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


def test_normal_association(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = JavaGenerator(model=library_model, output_dir=str(output_dir))
    generator.generate()

    library_file = os.path.join(str(output_dir), "Library.java")
    assert os.path.isfile(library_file)

    with open(library_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "private List<Book> books;" in code
    assert "getBooks()" in code
    assert "addBook(" in code


def test_self_association_does_not_crash(self_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = JavaGenerator(model=self_assoc_model, output_dir=str(output_dir))
    generator.generate()

    employee_file = os.path.join(str(output_dir), "Employee.java")
    assert os.path.isfile(employee_file)


def test_self_association_generates_both_fields(self_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = JavaGenerator(model=self_assoc_model, output_dir=str(output_dir))
    generator.generate()

    employee_file = os.path.join(str(output_dir), "Employee.java")
    with open(employee_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Both ends of the self-association should produce fields
    assert "private Employee manager;" in code or "private List<Employee> manager;" in code
    assert "private List<Employee> subordinates;" in code or "private Employee subordinates;" in code


def test_self_association_getters_setters(self_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = JavaGenerator(model=self_assoc_model, output_dir=str(output_dir))
    generator.generate()

    employee_file = os.path.join(str(output_dir), "Employee.java")
    with open(employee_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Should have getter for manager (single) and subordinates (list)
    assert "getManager()" in code
    assert "getSubordinates()" in code
    # Should have setter for single field and add method for list field
    assert "setManager(" in code
    assert "addTo" in code or "add" in code
