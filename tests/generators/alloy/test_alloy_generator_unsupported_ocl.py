import pytest
from besser.generators.alloy.ocl_translator import UnsupportedOclError
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


# collect and navigation from a collection return Bags: Alloy sets lose their duplicates

def test_size_of_collect_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Bag"):
        translate(library_with_ocl_v2, "Library", "self.books->collect(b | b.authors)->size() > 5", scope=5)


def test_size_of_navigation_from_a_collection_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Bag"):
        translate(library_with_ocl_v2, "Library", "self.books.authors->size() > 5", scope=5)


def test_bag_equality_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Bag"):
        translate(library_with_ocl_v2, "Library", "self.books.pages = self.books.pages", scope=5)


# OCL '/' is real division, Alloy 'div' integer division

def test_real_division_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Real division"):
        translate(library_with_ocl_v2, "Book", "self.pages / 2 > 1", scope=5)


# arithmetic and orderings on collections or non-integers

def test_arithmetic_on_collections_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="single integers"):
        translate(library_with_ocl_v2, "Library", "self.books.pages + 1 > 0", scope=5)


def test_string_ordering_with_integer_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="single integers"):
        translate(library_with_ocl_v2, "Book", "self.title > 3", scope=5)


def test_string_ordering_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="String comparison"):
        translate(library_with_ocl_v2, "Book", "self.title < self.title", scope=5)


# uninterpreted strings have no length

def test_string_size_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="strings are uninterpreted"):
        translate(library_with_ocl_v2, "Book", "self.title.size() > 0", scope=5)


# string literals name their constants, so their characters must be valid in identifiers

def test_string_literal_with_space_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="the characters ' ' cannot be translated"):
        translate(library_with_ocl_v2, "Author", "self.name = 'van gogh'", scope=5)


# type operations on collections collect their results

def test_type_operation_on_collection_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="collections"):
        translate(library_with_ocl_v2, "Library", "self.books.oclIsKindOf(Book)", scope=5)


# type operations take classes: Real, Integer, ... have no signature

def test_is_kind_of_real_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Type operations with 'Real'"):
        translate(library_with_ocl_v2, "Book", "self.pages.oclIsKindOf(Real)", scope=5)


def test_as_type_real_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Type operations with 'Real'"):
        translate(library_with_ocl_v2, "Book", "self.pages.oclAsType(Real) > 1", scope=5)


def test_is_type_of_integer_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="Type operations with 'Integer'"):
        translate(library_with_ocl_v2, "Book", "self.pages.oclIsTypeOf(Integer)", scope=5)


# enumeration literals are not in the B-OCL grammar, and a string is never an enumeration value

def test_enumeration_equal_to_string_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="enumerations cannot be compared with strings"):
        translate(library_with_ocl_v2, "Book", "self.genre = 'Adventure'", scope=5)


def test_string_different_from_enumeration_is_unsupported(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(UnsupportedOclError, match="enumerations cannot be compared with strings"):
        translate(library_with_ocl_v2, "Book", "'Adventure' <> self.genre", scope=5)
