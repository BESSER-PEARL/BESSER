import pytest
from besser.generators.alloy.ocl_translator import UnsupportedOclError
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec, translate


@pytest.mark.parametrize("value, characters", [
    ("Cortázar!", "'á', '!'"),
    ("a b", "' '"),
    ("x-y-z", "'-'"),
])
def test_strings_with_invalid_identifier_characters_are_unsupported(library_with_ocl_v2, value, characters):
    with pytest.raises(UnsupportedOclError, match=f"the translation of the characters {characters} is unsupported"):
        translate(library_with_ocl_v2, "Author", f"self.name = '{value}'", scope=5)


def test_string_constants_one_literal(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.name = 'borges'", scope=3)
    str_sigs = {sig for sig in spec.sigs if sig.startswith("one sig Str_")}
    assert len(str_sigs) == 3
    assert "one sig Str_borges extends Str" in spec.sigs
    assert "one sig Str_str0 extends Str" in str_sigs
    assert "one sig Str_str1 extends Str" in str_sigs
    assert "3 Str" in spec.run_scopes


def test_other_strings_skip_taken_names(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.name = 'str0'", scope=3)
    str_sigs = {sig for sig in spec.sigs if sig.startswith("one sig Str_")}
    assert len(str_sigs) == 3
    assert "one sig Str_str0 extends Str" in str_sigs
    assert "one sig Str_str1 extends Str" in str_sigs
    assert "one sig Str_str2 extends Str" in str_sigs
    assert "3 Str" in spec.run_scopes


def test_str_scope_holds_all_literals(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.name = 'a' or self.name = 'b' or self.name = 'c'", scope=2)
    str_sigs = {sig for sig in spec.sigs if sig.startswith("one sig Str_")}
    assert len(str_sigs) == 3
    assert "one sig Str_a extends Str" in spec.sigs
    assert "one sig Str_b extends Str" in spec.sigs
    assert "one sig Str_c extends Str" in spec.sigs
    assert "3 Str" in spec.run_scopes
    assert "2 Author" in spec.run_scopes



