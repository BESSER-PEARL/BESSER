"""Design checks of ``DomainModel.validate()`` that catch models whose generated code breaks.

Each check has a model that trips it and the nearest correct model, which must
not. The errors are defects for every code generator; the warnings break a
specific family of generators (named in each message) and the Spec-Driven
Agent promotes them to blockers when it builds that family.
"""

from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    IntegerType,
    Method,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)


def _attr(name, type_=StringType, **kwargs):
    return Property(name=name, type=type_, **kwargs)


def _assoc(name, cls_a, role_a, mult_a, cls_b, role_b, mult_b, composite_a=False):
    """A binary association: ``role_a`` is the end typed by ``cls_a``."""
    return BinaryAssociation(name=name, ends={
        Property(name=role_a, type=cls_a, multiplicity=Multiplicity(*mult_a), is_composite=composite_a),
        Property(name=role_b, type=cls_b, multiplicity=Multiplicity(*mult_b)),
    })


def _errors(model):
    return model.validate(raise_exception=False)["errors"]


def _warnings(model):
    return model.validate(raise_exception=False)["warnings"]


def _with(prefix, messages):
    return [m for m in messages if m.startswith(prefix)]


# ------------------------------------------------------------ invalid identifiers (error)


def test_a_name_that_is_not_an_identifier_is_an_error():
    item = Class(name="Item", attributes={_attr("2nd_name"), _attr("price$")})
    errors = _errors(DomainModel(name="Shop", types={item}))
    assert "Attribute '2nd_name' of class 'Item' is not a valid identifier" in errors[0]
    assert any("'price$'" in e for e in errors)
    assert len(errors) == 2


def test_identifier_check_covers_every_generated_name():
    status = Enumeration(name="Status", literals={EnumerationLiteral(name="1st")})
    item = Class(name="Item", attributes={_attr("name")},
                 methods={Method(name="go", parameters=[Parameter(name="a.b", type=StringType)])})
    other = Class(name="Other", attributes={_attr("x")})
    link = _assoc("item#other", item, "items", (0, "*"), other, "others", (0, "*"))
    errors = _errors(DomainModel(name="Shop", types={item, other, status}, associations={link}))
    joined = "\n".join(errors)
    assert "Literal '1st' of enumeration 'Status'" in joined
    assert "Parameter 'a.b' of method 'Item.go'" in joined
    assert "Many-to-many association 'item#other'" in joined


def test_name_of_an_association_without_join_table_is_not_checked():
    # Only a many-to-many's name becomes code (its join table); any other
    # association name is only ever written inside a string.
    item = Class(name="Item", attributes={_attr("name")})
    other = Class(name="Other", attributes={_attr("x")})
    link = _assoc("item#other", item, "item", (0, 1), other, "others", (0, "*"))
    from_link = _assoc("from", item, "source", (0, 1), other, "targets", (0, "*"))
    model = DomainModel(name="Shop", types={item, other}, associations={link, from_link})
    assert _errors(model) == []
    assert _with("Python keyword name", _warnings(model)) == []


def test_valid_identifiers_including_unicode_pass():
    item = Class(name="Größe", attributes={_attr("_private"), _attr("name2"), _attr("prix_unité")})
    assert _errors(DomainModel(name="Shop", types={item})) == []


# ------------------------------------------------------------ attribute type not in the model (error)


def test_attribute_typed_by_an_enumeration_outside_the_model_is_an_error():
    color = Enumeration(name="Color", literals={EnumerationLiteral(name="RED")})
    car = Class(name="Car", attributes={_attr("color", color)})
    errors = _errors(DomainModel(name="Garage", types={car}))
    assert errors == [
        "Attribute 'color' of class 'Car' has type 'Color', which is not in the domain model 'Garage'. "
        "Generated code would refer to a type it never defines. Add 'Color' to the model or use a primitive type."
    ]


def test_attribute_with_an_unresolved_type_name_is_an_error_not_a_crash():
    """``Property(type="Author")`` keeps the string when no primitive has that name."""
    book = Class(name="Book", attributes={_attr("author", "Author")})
    errors = _errors(DomainModel(name="Lib", types={book}))
    assert len(errors) == 1 and "has type 'Author', which is not in the domain model" in errors[0]


def test_attribute_typed_by_an_enumeration_of_the_model_is_fine():
    color = Enumeration(name="Color", literals={EnumerationLiteral(name="RED")})
    car = Class(name="Car", attributes={_attr("color", color)})
    assert _errors(DomainModel(name="Garage", types={car, color})) == []


# ------------------------------------------------------------ attribute vs association end (error)


def test_attribute_and_end_with_one_name_is_an_error():
    author = Class(name="Author", attributes={_attr("name")})
    book = Class(name="Book", attributes={_attr("title"), _attr("author")})
    link = _assoc("writes", author, "author", (1, 1), book, "books", (0, "*"))
    errors = _errors(DomainModel(name="Lib", types={author, book}, associations={link}))
    assert len(errors) == 1
    assert errors[0].startswith("Class 'Book' has an attribute and an association end both named 'author'.")


def test_inherited_collision_is_reported_once_on_the_declaring_class():
    author = Class(name="Author", attributes={_attr("name")})
    work = Class(name="Work", attributes={_attr("author")})
    book = Class(name="Book", attributes={_attr("isbn")})
    link = _assoc("writes", author, "author", (0, 1), work, "works", (0, "*"))
    model = DomainModel(name="Lib", types={author, work, book}, associations={link},
                        generalizations={Generalization(general=work, specific=book)})
    collisions = [e for e in _errors(model) if "both named 'author'" in e]
    assert len(collisions) == 1 and collisions[0].startswith("Class 'Work'")


def test_attribute_and_end_with_distinct_names_are_fine():
    author = Class(name="Author", attributes={_attr("name")})
    book = Class(name="Book", attributes={_attr("title"), _attr("authorName")})
    link = _assoc("writes", author, "author", (1, 1), book, "books", (0, "*"))
    assert _errors(DomainModel(name="Lib", types={author, book}, associations={link})) == []


# ------------------------------------------------------------ Python keyword names (warning)


def test_python_keyword_names_are_warned():
    kind = Enumeration(name="Kind", literals={EnumerationLiteral(name="None"), EnumerationLiteral(name="A")})
    item = Class(name="Item", attributes={_attr("from"), _attr("kind", kind)})
    model = DomainModel(name="Shop", types={item, kind})
    found = _with("Python keyword name:", _warnings(model))
    assert any("Attribute 'from' of class 'Item' is the Python keyword 'from'" in w for w in found)
    assert any("Literal 'None' of enumeration 'Kind'" in w for w in found)
    assert model.validate(raise_exception=False)["success"] is True


def test_soft_keywords_and_builtins_are_not_warned():
    item = Class(name="List", attributes={_attr("type"), _attr("match"), _attr("from_")})
    assert _with("Python keyword name:", _warnings(DomainModel(name="Shop", types={item}))) == []


# ------------------------------------------------------------ class names differing in case (warning)


def test_class_names_differing_only_in_case_are_warned():
    upper = Class(name="Order", attributes={_attr("x")})
    lower = Class(name="order", attributes={_attr("y")})
    found = _with("Class names differ only in case:", _warnings(DomainModel(name="Shop", types={upper, lower})))
    assert found == [
        "Class names differ only in case: 'Order', 'order'. Generated database tables are named in lower case, "
        "so these classes map to the same table 'order' and the application fails at startup. "
        "Give the classes distinct names."
    ]


def test_distinct_class_names_are_fine():
    a = Class(name="Order", attributes={_attr("x")})
    b = Class(name="OrderLine", attributes={_attr("y")})
    assert _with("Class names differ", _warnings(DomainModel(name="Shop", types={a, b}))) == []


# ------------------------------------------------------------ attribute names differing in case (warning)


def test_attribute_names_differing_only_in_case_are_warned():
    person = Class(name="Person", attributes={_attr("Name"), _attr("name")})
    found = _with("Attribute names differ only in case:", _warnings(DomainModel(name="P", types={person})))
    assert len(found) == 1 and "'Name', 'name' in class 'Person'" in found[0]


def test_same_attribute_name_in_two_classes_is_fine():
    person = Class(name="Person", attributes={_attr("name")})
    pet = Class(name="Pet", attributes={_attr("Name")})
    assert _with("Attribute names differ", _warnings(DomainModel(name="P", types={person, pet}))) == []


# ------------------------------------------------------------ many-to-many join table names (warning)


def test_many_to_many_named_like_a_class_is_warned():
    student = Class(name="Student", attributes={_attr("n")})
    course = Class(name="Course", attributes={_attr("t")})
    enrollment = Class(name="Enrollment", attributes={_attr("grade")})
    link = _assoc("Enrollment", student, "students", (0, "*"), course, "courses", (0, "*"))
    model = DomainModel(name="School", types={student, course, enrollment}, associations={link})
    found = _with("Join table name clash:", _warnings(model))
    assert found == [
        "Join table name clash: class 'Enrollment' and many-to-many association 'Enrollment' map to the same "
        "table 'enrollment' (a many-to-many association gets a join table named after it, in lower case). "
        "The application fails at startup because the table is defined twice. Rename the association."
    ]


def test_two_many_to_many_associations_differing_in_case_are_warned_once():
    a = Class(name="A", attributes={_attr("x")})
    b = Class(name="B", attributes={_attr("y")})
    one = _assoc("Tags", a, "as1", (0, "*"), b, "bs1", (0, "*"))
    two = _assoc("tags", a, "as2", (0, "*"), b, "bs2", (0, "*"))
    found = _with("Join table name clash:", _warnings(DomainModel(name="M", types={a, b}, associations={one, two})))
    assert len(found) == 1


def test_one_to_many_or_association_class_named_like_a_class_is_fine():
    """Only a many-to-many without an association class gets a join table."""
    student = Class(name="Student", attributes={_attr("n")})
    course = Class(name="Course", attributes={_attr("t")})
    enrollment = Class(name="Enrollment", attributes={_attr("grade")})
    one_to_many = _assoc("Enrollment", student, "student", (0, 1), course, "courses", (0, "*"))
    takes = _assoc("Takes", student, "takers", (0, "*"), course, "taken", (0, "*"))
    takes_class = AssociationClass(name="Takes_", attributes={_attr("year")}, association=takes)
    model = DomainModel(name="School", types={student, course, enrollment, takes_class},
                        associations={one_to_many, takes})
    assert _with("Join table name clash:", _warnings(model)) == []


# ------------------------------------------------------------ foreign key column clash (warning)


def test_attribute_named_like_the_generated_foreign_key_is_warned():
    author = Class(name="Author", attributes={_attr("name")})
    book = Class(name="Book", attributes={_attr("title"), _attr("author_id", IntegerType)})
    link = _assoc("writes", author, "author", (1, 1), book, "books", (0, "*"))
    found = _with("Foreign key column clash:", _warnings(DomainModel(name="L", types={author, book},
                                                                    associations={link})))
    assert len(found) == 1
    assert "attribute 'author_id' of class 'Book'" in found[0] and "end 'author'" in found[0]


def test_to_many_end_does_not_get_a_foreign_key_column():
    author = Class(name="Author", attributes={_attr("name"), _attr("books_id", IntegerType)})
    book = Class(name="Book", attributes={_attr("title")})
    link = _assoc("writes", author, "author", (1, 1), book, "books", (0, "*"))
    assert _with("Foreign key column clash:", _warnings(DomainModel(name="L", types={author, book},
                                                                   associations={link}))) == []


# ------------------------------------------------------------ 'id' attributes (warning)


def test_subclass_declaring_id_is_warned():
    parent = Class(name="Vehicle", attributes={_attr("name")})
    child = Class(name="Car", attributes={_attr("id", IntegerType, is_id=True)})
    model = DomainModel(name="G", types={parent, child},
                        generalizations={Generalization(general=parent, specific=child)})
    found = _with("Subclass declares 'id':", _warnings(model))
    assert len(found) == 1 and "class 'Car'" in found[0]


def test_root_id_that_is_not_the_identifier_is_warned():
    item = Class(name="Item", attributes={_attr("id"), _attr("name")})
    found = _with("Attribute 'id' is not the identifier:", _warnings(DomainModel(name="S", types={item})))
    assert len(found) == 1 and "class 'Item'" in found[0]


def test_root_id_marked_as_identifier_is_fine():
    item = Class(name="Item", attributes={_attr("id", IntegerType, is_id=True), _attr("name")})
    parent = Class(name="Vehicle", attributes={_attr("code", is_id=True)})
    child = Class(name="Car", attributes={_attr("plate")})
    model = DomainModel(name="S", types={item, parent, child},
                        generalizations={Generalization(general=parent, specific=child)})
    warnings = _warnings(model)
    assert _with("Subclass declares 'id':", warnings) == []
    assert _with("Attribute 'id' is not the identifier:", warnings) == []


# ------------------------------------------------------------ empty enumerations (warning)


def test_enumeration_without_literals_is_warned():
    status = Enumeration(name="Status", literals=set())
    task = Class(name="Task", attributes={_attr("title")})
    found = _with("Empty enumeration:", _warnings(DomainModel(name="T", types={task, status})))
    assert len(found) == 1 and "'Status' has no literals" in found[0]


def test_enumeration_with_literals_is_fine():
    status = Enumeration(name="Status", literals={EnumerationLiteral(name="OPEN")})
    task = Class(name="Task", attributes={_attr("status", status)})
    assert _with("Empty enumeration:", _warnings(DomainModel(name="T", types={task, status}))) == []


# ------------------------------------------------------------ class-typed attributes (warning)


def test_attribute_typed_by_a_class_is_warned():
    author = Class(name="Author", attributes={_attr("name")})
    book = Class(name="Book", attributes={_attr("title"), _attr("writer", author)})
    found = _with("Class-typed attribute:", _warnings(DomainModel(name="L", types={author, book})))
    assert len(found) == 1
    assert "association from 'Book' to 'Author'" in found[0]


def test_association_instead_of_class_typed_attribute_is_fine():
    author = Class(name="Author", attributes={_attr("name")})
    book = Class(name="Book", attributes={_attr("title")})
    link = _assoc("writes", author, "writer", (1, 1), book, "books", (0, "*"))
    model = DomainModel(name="L", types={author, book}, associations={link})
    assert _with("Class-typed attribute:", _warnings(model)) == []


# ------------------------------------------------------------ isolated classes (warning)


def test_class_with_nothing_is_warned():
    empty = Class(name="Empty")
    item = Class(name="Item", attributes={_attr("name")})
    found = _with("Isolated class:", _warnings(DomainModel(name="S", types={empty, item})))
    assert len(found) == 1 and "'Empty'" in found[0]


def test_class_without_attributes_but_with_a_relationship_is_fine():
    tag = Class(name="Tag")                       # only linked
    marker = Class(name="Marker")                 # only a specialization
    service = Class(name="Service", methods={Method(name="run")})
    item = Class(name="Item", attributes={_attr("name")})
    link = _assoc("tags", tag, "tags", (0, "*"), item, "items", (0, "*"))
    model = DomainModel(name="S", types={tag, marker, service, item}, associations={link},
                        generalizations={Generalization(general=item, specific=marker)})
    assert _with("Isolated class:", _warnings(model)) == []


# ------------------------------------------------------------ abstract without concrete subclass (warning)


def test_abstract_class_without_concrete_subclass_is_warned():
    shape = Class(name="Shape", attributes={_attr("color")}, is_abstract=True)
    drawing = Class(name="Drawing", attributes={_attr("title")})
    link = _assoc("contains", shape, "shape", (1, 1), drawing, "drawings", (0, "*"))
    found = _with("Abstract class without concrete subclass:",
                  _warnings(DomainModel(name="D", types={shape, drawing}, associations={link})))
    assert len(found) == 1
    assert "'Shape' is abstract" in found[0]
    assert "Objects of 'Drawing' must each be linked to one" in found[0]


def test_abstract_class_with_a_concrete_grandchild_is_fine():
    shape = Class(name="Shape", attributes={_attr("color")}, is_abstract=True)
    polygon = Class(name="Polygon", attributes={_attr("sides", IntegerType)}, is_abstract=True)
    square = Class(name="Square", attributes={_attr("side", IntegerType)})
    model = DomainModel(name="D", types={shape, polygon, square}, generalizations={
        Generalization(general=shape, specific=polygon), Generalization(general=polygon, specific=square),
    })
    assert _with("Abstract class without concrete subclass:", _warnings(model)) == []


# ------------------------------------------------------------ part with two mandatory owners (warning)


def test_part_with_two_mandatory_composite_owners_is_warned():
    order = Class(name="Order", attributes={_attr("ref")})
    invoice = Class(name="Invoice", attributes={_attr("no")})
    line = Class(name="Line", attributes={_attr("qty", IntegerType)})
    a = _assoc("order_lines", order, "order", (1, 1), line, "lines", (0, "*"), composite_a=True)
    b = _assoc("invoice_lines", invoice, "invoice", (1, 1), line, "items", (0, "*"), composite_a=True)
    found = _with("Part with several mandatory owners:",
                  _warnings(DomainModel(name="S", types={order, invoice, line}, associations={a, b})))
    assert len(found) == 1
    assert "class 'Line' is a mandatory part of 'Invoice' (invoice_lines) and 'Order' (order_lines)" in found[0]


def test_part_with_one_mandatory_and_one_optional_owner_is_fine():
    order = Class(name="Order", attributes={_attr("ref")})
    invoice = Class(name="Invoice", attributes={_attr("no")})
    line = Class(name="Line", attributes={_attr("qty", IntegerType)})
    a = _assoc("order_lines", order, "order", (1, 1), line, "lines", (0, "*"), composite_a=True)
    b = _assoc("invoice_lines", invoice, "invoice", (0, 1), line, "items", (0, "*"), composite_a=True)
    model = DomainModel(name="S", types={order, invoice, line}, associations={a, b})
    assert _with("Part with several mandatory owners:", _warnings(model)) == []


# ------------------------------------------------------------ the canonical fixture stays clean


def test_library_fixture_has_no_design_findings(library_book_author_model):
    result = library_book_author_model.validate(raise_exception=False)
    assert result == {"success": True, "errors": [], "warnings": []}
