import pytest
from besser.generators.alloy.ocl_translator import OclTranslationError
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


# let is not in the B-OCL grammar

def test_let_is_a_syntax_error(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(OclTranslationError, match="Invalid OCL invariant"):
        translate(library_with_ocl_v2, "Book", "let n: Integer = self.pages in n > 10", scope=5)


def test_trailing_tokens_are_a_syntax_error(library_with_ocl_v2):  # noqa: F811
    with pytest.raises(OclTranslationError, match="Invalid OCL invariant"):
        translate(library_with_ocl_v2, "Book", "self.pages > 10 self.stock > 0", scope=5)
