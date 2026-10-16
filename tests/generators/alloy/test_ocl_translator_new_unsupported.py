import pytest
from besser.BUML.metamodel.structural import Constraint
from besser.generators.alloy.alloy_translator_new import BumlToAlloy
from besser.generators.alloy.ocl_translator_new import UnsupportedOclError
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2 


@pytest.mark.parametrize("context, expression, error", [
    # collect and navigation from a collection return Bags: Alloy sets lose their duplicates
    ("Library", "self.books->collect(b | b.authors)->size() > 5", "Bag"),
    ("Library", "self.books.authors->size() > 5", "Bag"),
    ("Library", "self.books.pages = self.books.pages", "Bag"),
    # OCL '/' is real division, Alloy 'div' integer division
    ("Book", "self.pages / 2 > 1", "Real division"),
    # arithmetic and orderings on collections or non-integers
    ("Library", "self.books.pages + 1 > 0", "single integers"),
    ("Book", "self.title > 3", "single integers"),
    ("Book", "self.title < self.title", "String comparison"),
    # type operations on collections collect their results
    ("Library", "self.books.oclIsKindOf(Book)", "collections"),
])
def test_unsupported_construct(library_with_ocl_v2, context, expression, error):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match=error):
        add_invariant_to_spec(library_with_ocl_v2, context, expression)


def test_bag_as_set_size(library_with_ocl_v2):  # noqa: F811
    specification = add_invariant_to_spec(
        library_with_ocl_v2, "Library", "self.books->collect(b | b.authors)->asSet()->size() > 5"
    )
    assert "all self:this/Library|(#({collected: Author | some b : self.Library_books | " \
        "collected in b.Book_authors}) > 5)" in specification


def add_invariant_to_spec(model, context: str, expression: str) -> str:
    """Translates *model* with the single invariant ``context <context> inv: <expression>``."""
    model.constraints = {
        Constraint(
            name="test_inv",
            context=model.get_class_by_name(context),
            expression=f"context {context} inv test_inv: {expression}",
            language="OCL",
        )
    }
    return BumlToAlloy(model, scope=5).translate().render()