"""Invariants that navigate an association must be evaluated, not dropped.

Two defects hid them from ``/validate-diagram`` on an object diagram:

- the checker treated every ``self.<name>`` as an attribute, so an association
  end such as ``self.books`` (which has no slot) made it skip the invariant
  without reporting it at all;
- ``BOCLVisitorImpl`` (the parser ``bocl`` evaluates with) typed an untyped
  loop variable as the context class, so ``self.books->forAll(b | b.pages > 0)``
  failed with "Property 'pages' not found in context 'Library'".
"""

import pytest

from besser.BUML.metamodel.object import (
    AttributeLink, DataValue, Link, LinkEnd, Object, ObjectModel,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DomainModel, IntegerType, Multiplicity,
    Property, StringType,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)


@pytest.fixture()
def library():
    name, title, pages = Property("name", StringType), Property("title", StringType), Property("pages", IntegerType)
    library_cls = Class("Library", attributes={name})
    book_cls = Class("Book", attributes={title, pages})
    end_library = Property("library", library_cls, multiplicity=Multiplicity(1, 1))
    end_books = Property("books", book_cls, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation("Library_Book", ends={end_library, end_books})
    model = DomainModel("Libraries", types={library_cls, book_cls}, associations={assoc})

    def objects(book_pages):
        lib = Object("lib", library_cls, slots=[
            AttributeLink(attribute=name, value=DataValue(classifier=StringType, value="City")),
        ])
        books = []
        for i, n in enumerate(book_pages):
            book = Object(f"b{i}", book_cls, slots=[
                AttributeLink(attribute=title, value=DataValue(classifier=StringType, value=f"T{i}")),
                AttributeLink(attribute=pages, value=DataValue(classifier=IntegerType, value=n)),
            ])
            Link(f"l{i}", assoc, [LinkEnd(f"e{i}a", end_library, lib), LinkEnd(f"e{i}b", end_books, book)])
            books.append(book)
        return ObjectModel("objs", objects={lib, *books})

    return model, library_cls, book_cls, objects


def _check(model, context, expression, object_model):
    model.constraints = {Constraint(name="c", context=context, expression=expression, language="OCL")}
    return check_ocl_constraint(model, object_model)


@pytest.mark.parametrize(("expression", "satisfied_pages", "violated_pages"), [
    ("context Library inv: self.books->size() > 0", [100], []),
    ("context Library inv: self.books->forAll(b | b.pages > 0)", [100, 20], [100, -1]),
    ("context Library inv: self.books->exists(b | b.pages > 300)", [100, 400], [100, 200]),
    ("context Library inv: self.books->select(b | b.pages > 300)->size() <= 1", [400, 20], [400, 500]),
    ("context Library inv: Book.allInstances()->forAll(b | b.pages > 0)", [100], [-1]),
])
def test_association_navigation_is_evaluated(library, expression, satisfied_pages, violated_pages):
    model, library_cls, _book_cls, objects = library

    ok = _check(model, library_cls, expression, objects(satisfied_pages))
    assert ok["success"], ok
    assert len(ok["valid_constraints"]) == 1
    assert "Evaluates to: True" in ok["valid_constraints"][0]

    bad = _check(model, library_cls, expression, objects(violated_pages))
    assert not bad["success"], bad
    assert len(bad["invalid_constraints"]) == 1
    assert "Constraint violation" in bad["invalid_constraints"][0]


def test_unset_attribute_still_skips_the_invariant(library):
    """The user-model behaviour the attribute check exists for is unchanged."""
    model, library_cls, book_cls, objects = library
    om = objects([100])
    for obj in om.objects:
        if obj.classifier is book_cls:
            obj.slots = [s for s in obj.slots if s.attribute.name != "pages"]
    result = _check(model, book_cls, "context Book inv: self.pages > 0", om)
    assert result["valid_constraints"] == [] and result["invalid_constraints"] == []
