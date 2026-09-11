import os

import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, BinaryAssociation, Multiplicity,
    Generalization, Method, Parameter,
    StringType, IntegerType, FloatType,
)
from besser.generators.java_classes import JavaGenerator


@pytest.fixture
def library_model(simple_library_book_model):
    """Alias the shared simple Library-Book fixture."""
    return simple_library_book_model


@pytest.fixture
def self_assoc_model(employee_self_assoc_model):
    """Alias the shared Employee self-association fixture."""
    return employee_self_assoc_model


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
def model_with_void_method():
    """Book class with a no-arg void method."""
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=StringType)}
    book.methods = {Method(name="save", visibility="public")}
    return DomainModel(name="Void_Method_Model", types={book}, associations={}, generalizations={})


@pytest.fixture
def many_to_many_mixed_min_model():
    """Student-Course many-to-many: student end min=1, course end min=0."""
    student = Class(name="Student")
    course = Class(name="Course")
    student.attributes = {Property(name="studentId", type=StringType)}
    course.attributes = {Property(name="title", type=StringType)}
    assoc = BinaryAssociation(
        name="Enrollment",
        ends={
            Property(name="students", type=student, multiplicity=Multiplicity(1, 9999), is_navigable=True),
            Property(name="courses", type=course, multiplicity=Multiplicity(0, 9999), is_navigable=True),
        },
    )
    return DomainModel(name="M2M_Mixed_Min_Model", types={student, course}, associations={assoc}, generalizations={})


@pytest.fixture
def bare_subclass_model():
    """Subclass with no own attributes inheriting from a parent that has attributes."""
    vehicle = Class(name="Vehicle")
    vehicle.attributes = {Property(name="speed", type=IntegerType)}
    car = Class(name="Car")
    gen = Generalization(general=vehicle, specific=car)
    return DomainModel(name="Bare_Subclass_Model", types={vehicle, car}, associations={}, generalizations={gen})


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

    assert "private Employee manager;" in code or "private List<Employee> manager;" in code
    assert "private List<Employee> subordinates;" in code or "private Employee subordinates;" in code


def test_self_association_getters_setters(self_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = JavaGenerator(model=self_assoc_model, output_dir=str(output_dir))
    generator.generate()

    employee_file = os.path.join(str(output_dir), "Employee.java")
    with open(employee_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "getManager()" in code
    assert "getSubordinates()" in code
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


def test_enum_attribute_field_uses_enum_type(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Author.java"))
    assert "private MemberType member;" in code


def test_enum_attribute_field_not_none(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Author.java"))
    assert "private None" not in code


def test_association_field_uses_role_name(custom_role_name_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=custom_role_name_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "private List<Book> ownedBooks;" in code
    assert "private List<Book> books;" not in code


def test_getter_uses_role_name(custom_role_name_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=custom_role_name_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "getOwnedBooks()" in code


def test_bidirectional_many_to_one_backref_field(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model, output_dir=str(output_dir)).generate()

    book_code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "private Library library;" in book_code


def test_bidirectional_backref_field_uses_role_name(custom_role_name_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=custom_role_name_model, output_dir=str(output_dir)).generate()

    book_code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "private Library owner;" in book_code
    assert "private Library library;" not in book_code


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
    assert "private List<Employee> reports;" not in code


def test_constructor_no_leading_comma_when_no_attributes(no_attrib_list_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=no_attrib_list_assoc_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Container.java"))
    assert "(, " not in code
    assert "List<Item> items" in code


def test_explicit_package_name_emitted(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model, output_dir=str(output_dir), package_name="com.example").generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "package com.example;" in code


def test_no_package_when_package_name_is_none(library_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model, output_dir=str(output_dir), package_name=None).generate()

    code = _read(os.path.join(str(output_dir), "Library.java"))
    assert "package " not in code


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


def test_many_to_many_mixed_min_generates_field_on_owning_side(many_to_many_mixed_min_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=many_to_many_mixed_min_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Student.java"))
    assert "private List<Course> courses;" in code


def test_many_to_many_mixed_min_generates_field_on_other_side(many_to_many_mixed_min_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=many_to_many_mixed_min_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Course.java"))
    assert "private List<Student> students;" in code


def test_overloaded_constructor_uses_list_interface(no_attrib_list_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=no_attrib_list_assoc_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Container.java"))
    assert "List<Item> items" in code
    assert "ArrayList<Item> items" not in code


def test_enum_literals_no_blank_lines_between(library_model_with_enum, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "MemberType.java"))
    enum_body = code.split("{", 1)[1].rsplit("}", 1)[0]
    assert "\n\n" not in enum_body


def test_void_method_stub_signature(model_with_void_method, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=model_with_void_method, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "public void save()" in code


def test_non_navigable_book_retains_own_attributes(non_navigable_assoc_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=non_navigable_assoc_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Book.java"))
    assert "private String title;" in code


def test_primary_constructor_no_leading_comma_for_bare_subclass(bare_subclass_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    JavaGenerator(model=bare_subclass_model, output_dir=str(output_dir)).generate()

    code = _read(os.path.join(str(output_dir), "Car.java"))
    assert "(, " not in code
    assert "extends Vehicle" in code
