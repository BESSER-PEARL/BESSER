from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import set_constraint, translate


def test_implies_false(library_with_ocl_v2):  # noqa: F811
    set_constraint(library_with_ocl_v2, "Book", "self.pages > 10 implies false")
    spec = translate(library_with_ocl_v2, scope=5)
    assert "all self:this/Book|((self.Book_pages > 10) => (False = True))" in spec.facts


def test_xor(library_with_ocl_v2):  # noqa: F811
    set_constraint(library_with_ocl_v2, "Book", "self.pages > 10 xor self.stock > 0")
    spec = translate(library_with_ocl_v2, scope=5)
    assert "all self:this/Book|(not ((self.Book_pages > 10) <=> (self.Book_stock > 0)))" in spec.facts


def test_multiplication(library_with_ocl_v2):  # noqa: F811
    set_constraint(library_with_ocl_v2, "Book", "self.pages * 2 > 10")
    spec = translate(library_with_ocl_v2, scope=5)
    assert "all self:this/Book|(mul[self.Book_pages, 2] > 10)" in spec.facts


def test_implies_binds_looser_than_or(library_with_ocl_v2):  # noqa: F811
    set_constraint(library_with_ocl_v2, "Book", "self.pages > 10 implies self.stock > 0 or self.price > 0")
    spec = translate(library_with_ocl_v2, scope=5)
    assert "all self:this/Book|((self.Book_pages > 10) => ((self.Book_stock > 0) or (self.Book_price > 0)))" \
        in spec.facts
