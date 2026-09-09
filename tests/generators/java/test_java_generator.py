import os
import shutil

import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation, Multiplicity,
    Method, Parameter,
    StringType, IntegerType, FloatType,
)
from besser.generators.java_classes import JavaGenerator


# Use shared fixtures from tests/conftest.py


@pytest.fixture
def library_model(simple_library_book_model):
    """Alias the shared simple Library-Book fixture."""
    return simple_library_book_model


@pytest.fixture
def self_assoc_model(employee_self_assoc_model):
    """Alias the shared Employee self-association fixture."""
    return employee_self_assoc_model


@pytest.fixture
def non_tmp_output_dir():
    """A directory in CWD whose path contains no 'tmp' — needed for package-name tests."""
    d = os.path.join(os.path.abspath("."), "java_test_pkg_output")
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def custom_role_name_model():
    """Library-Book where the Book end is named 'ownedBooks' (not 'books').

    Verifies that the generator uses end.name rather than classname.lower()+'s'.
    """
    library = Class(name="Library")
    book = Class(name="Book")
    library.attributes = {Property(name="name", type=StringType)}
    book.attributes = {Property(name="title", type=StringType)}
    assoc = BinaryAssociation(
        name="Owns",
        ends={
            Property(name="ownedBooks", type=book, multiplicity=Multiplicity(0, 9999), is_navigable=True),
            Property(name="owner", type=library, multiplicity=Multiplicity(1, 1), is_navigable=True),
        },
    )
    return DomainModel(name="Custom_Role_Model", types={library, book}, associations={assoc}, generalizations={})


@pytest.fixture
def model_with_methods():
    """Book class with a typed return method and a parameterised void method."""
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=StringType)}
    get_info = Method(name="getInfo", visibility="public", type=StringType)
    set_price = Method(name="setPrice", visibility="public", parameters=[Parameter(name="price", type=FloatType)])
    book.methods = {get_info, set_price}
    return DomainModel(name="Method_Model", types={book}, associations={}, generalizations={})


@pytest.fixture
def non_navigable_assoc_model():
    """Library-Book where the back-reference end (library) is not navigable."""
    library = Class(name="Library")
    book = Class(name="Book")
    library.attributes = {Property(name="name", type=StringType)}
    book.attributes = {Property(name="title", type=StringType)}
    assoc = BinaryAssociation(
        name="Has",
        ends={
            Property(name="books", type=book, multiplicity=Multiplicity(0, 9999), is_navigable=True),
            Property(name="library", type=library, multiplicity=Multiplicity(1, 1), is_navigable=False),
        },
    )
    return DomainModel(name="NonNav_Model", types={library, book}, associations={assoc}, generalizations={})


@pytest.fixture
def self_assoc_non_navigable_model():
    """Employee self-association where the 'reports' end is not navigable."""
    employee = Class(name="Employee")
    employee.attributes = {Property(name="name", type=StringType)}
    assoc = BinaryAssociation(
        name="Manages",
        ends={
            Property(name="manager", type=employee, multiplicity=Multiplicity(0, 1), is_navigable=True),
            Property(name="reports", type=employee, multiplicity=Multiplicity(0, 9999), is_navigable=False),
        },
    )
    return DomainModel(name="SelfAssoc_NonNav_Model", types={employee}, associations={assoc}, generalizations={})


@pytest.fixture
def no_attrib_list_assoc_model():
    """Container class with no attributes but a one-to-many List association.

    Exercises the overloaded constructor comma-fix: the first ArrayList param
    must not be preceded by a comma when the class has no regular attributes.
    """
    container = Class(name="Container")
    item = Class(name="Item")
    item.attributes = {Property(name="label", type=StringType)}
    assoc = BinaryAssociation(
        name="Contains",
        ends={
            Property(name="items", type=item, multiplicity=Multiplicity(0, 9999), is_navigable=True),
            Property(name="container", type=container, multiplicity=Multiplicity(1, 1), is_navigable=True),
        },
    )
    return DomainModel(name="NoAttrib_Model", types={container, item}, associations={assoc}, generalizations={})


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


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


def test_enum_file_generated(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    assert os.path.isfile(os.path.join(str(output_dir), "MemberType.java"))


def test_enum_declaration(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "MemberType.java"))
    assert "public enum MemberType" in code


def test_enum_literals_present(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "MemberType.java"))
    for literal in ("ADULT", "SENIOR", "STUDENT", "CHILD"):
        assert literal in code


def test_class_files_generated_alongside_enum(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    for name in ("Library.java", "Book.java", "Author.java"):
        assert os.path.isfile(os.path.join(str(output_dir), name))


def test_association_field_uses_role_name(custom_role_name_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=custom_role_name_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "private List<Book> ownedBooks;" in code
    assert "private List<Book> books;" not in code  # regression: old code used classname.lower()+'s'


def test_getter_uses_role_name(custom_role_name_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=custom_role_name_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "getOwnedbooks()" in code  # Jinja2 capitalize() lowercases all but first letter


def test_bidirectional_many_to_one_backref_field(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model, output_dir=str(output_dir)).generate()

    # Book (many side) should get a back-reference field to Library (one side)
    book_code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "private Library library;" in book_code


def test_bidirectional_backref_field_uses_role_name(custom_role_name_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=custom_role_name_model, output_dir=str(output_dir)).generate()

    book_code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "private Library owner;" in book_code
    assert "private Library library;" not in book_code  # regression: old code used classname.lower()


def test_method_stubs_generated(model_with_methods, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=model_with_methods, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "getInfo" in code
    assert "setPrice" in code


def test_method_stub_return_type_mapped(model_with_methods, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=model_with_methods, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "public String getInfo(" in code


def test_method_stub_parameter_type_mapped(model_with_methods, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=model_with_methods, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "float price" in code


def test_non_navigable_end_generates_no_field(non_navigable_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=non_navigable_assoc_model, output_dir=str(output_dir)).generate()

    # Non-navigable end must not produce a field in Book
    book_code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "private Library library;" not in book_code


def test_navigable_end_still_generates_field(non_navigable_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=non_navigable_assoc_model, output_dir=str(output_dir)).generate()

    library_code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "private List<Book> books;" in library_code


def test_self_assoc_non_navigable_end_excluded(self_assoc_non_navigable_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=self_assoc_non_navigable_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Employee.java"))
    assert "private Employee manager;" in code
    assert "private List<Employee> reports;" not in code  # non-navigable end excluded


def test_constructor_no_leading_comma_when_no_attributes(no_attrib_list_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=no_attrib_list_assoc_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Container.java"))
    # Overloaded constructor must not have a leading comma before the first ArrayList param
    assert "(, ArrayList" not in code
    assert "ArrayList<Item>" in code


def test_no_package_emitted_in_tmp_dir(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "package " not in code


def test_package_emitted_for_non_tmp_dir(library_model, non_tmp_output_dir):
    JavaGenerator(model=library_model, output_dir=non_tmp_output_dir).generate()

    code = _read(os.path.join(non_tmp_output_dir, "Library.java"))
    assert "package " in code


def test_inheritance_extends_keyword(library_model_with_inheritance, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_inheritance, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Horror.java"))
    assert "extends BookType" in code


def test_inheritance_super_call_in_constructor(library_model_with_inheritance, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_inheritance, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Horror.java"))
    assert "super(" in code


def test_all_subclasses_generated(library_model_with_inheritance, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_inheritance, output_dir=str(output_dir)).generate()

    for name in ("Horror.java", "History.java", "Science.java", "BookType.java"):
        assert os.path.isfile(os.path.join(str(output_dir), name))
