import pytest
from besser.generators.alloy.ocl_translator import UnsupportedOclError
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec, translate


@pytest.mark.parametrize("value", ["01/02/2020", "Jan 5, 2020", "2020-02-30"])
def test_dates_that_are_not_iso_are_unsupported(library_with_ocl_v2, value):
    with pytest.raises(UnsupportedOclError, match=f"Date '{value}' is not supported: date literals must be ISO dates"):
        translate(library_with_ocl_v2, "Author", f"self.birth = '{value}'", scope=5)


@pytest.mark.parametrize("expression", ["Date::today()", "Date::today().addDays(3)"])
def test_date_expressions_are_unsupported(library_with_ocl_v2, expression):
    with pytest.raises(UnsupportedOclError, match="Date expression 'Date::today\\(\\).*' is not supported"):
        translate(library_with_ocl_v2, "Author", f"self.birth = {expression}", scope=5)


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
def test_date_orderings_are_unsupported(library_with_ocl_v2, operator):
    with pytest.raises(UnsupportedOclError, match=f"Date comparison '{operator}' is not supported"):
        translate(library_with_ocl_v2, "Author", f"self.birth {operator} '1950-01-01'", scope=5)


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
def test_date_attribute_orderings_are_unsupported(library_with_ocl_v2, operator):
    with pytest.raises(UnsupportedOclError, match=f"Date comparison '{operator}' is not supported"):
        translate(library_with_ocl_v2, "Author", f"self.books->forAll(b | b.release {operator} self.birth)", scope=5)


def test_date_singletons_one_literal(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.birth = '1950-01-01'", scope=3)
    assert len(spec.subsigs_of("Date")) == 3
    assert "one sig Date_1950_01_01 extends Date" in spec.sigs
    # The other dates are random (seed 0)
    assert "one sig Date_2017_04_12 extends Date" in spec.sigs
    assert "one sig Date_2037_11_17 extends Date" in spec.sigs
    assert "all self:this/Author|(dateEq[self.Author_birth,Date_1950_01_01])" in spec.facts
    assert "dateLt[Date_1950_01_01, Date_2017_04_12] and dateLt[Date_2017_04_12, Date_2037_11_17]" in spec.facts
    assert "3 Date" in spec.run_scopes


def test_repeated_date_literal_is_declared_once(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.birth = '1950-01-01' or self.birth <> '1950-01-01'", scope=3)
    assert len(spec.subsigs_of("Date")) == 3
    assert "one sig Date_1950_01_01 extends Date" in spec.sigs
    # The other dates are random (seed 0)
    assert "one sig Date_2017_04_12 extends Date" in spec.sigs
    assert "one sig Date_2037_11_17 extends Date" in spec.sigs
    assert "all self:this/Author|((dateEq[self.Author_birth,Date_1950_01_01]) " \
        "or (dateNe[self.Author_birth,Date_1950_01_01]))" in spec.facts
    assert "dateLt[Date_1950_01_01, Date_2017_04_12] and dateLt[Date_2017_04_12, Date_2037_11_17]" in spec.facts
    assert "3 Date" in spec.run_scopes


def test_date_literal_on_the_left(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "'1950-01-01' <> self.birth", scope=3)
    assert len(spec.subsigs_of("Date")) == 3
    assert "one sig Date_1950_01_01 extends Date" in spec.sigs
    # The other dates are random (seed 0)
    assert "one sig Date_2017_04_12 extends Date" in spec.sigs
    assert "one sig Date_2037_11_17 extends Date" in spec.sigs
    assert "all self:this/Author|(dateNe[Date_1950_01_01,self.Author_birth])" in spec.facts
    assert "dateLt[Date_1950_01_01, Date_2017_04_12] and dateLt[Date_2017_04_12, Date_2037_11_17]" in spec.facts


def test_dates_are_ordered_chronologically(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.birth = '2018-01-01'", scope=5)
    assert len(spec.subsigs_of("Date")) == 5
    assert "one sig Date_2018_01_01 extends Date" in spec.sigs
    # The other dates are random (seed 0)
    assert "one sig Date_2017_04_12 extends Date" in spec.sigs
    assert "one sig Date_2018_11_12 extends Date" in spec.sigs
    assert "one sig Date_2033_12_31 extends Date" in spec.sigs
    assert "one sig Date_2037_11_17 extends Date" in spec.sigs
    assert "all self:this/Author|(dateEq[self.Author_birth,Date_2018_01_01])" in spec.facts
    # The literal falls between the random dates
    assert "dateLt[Date_2017_04_12, Date_2018_01_01] and dateLt[Date_2018_01_01, Date_2018_11_12] " \
        "and dateLt[Date_2018_11_12, Date_2033_12_31] and dateLt[Date_2033_12_31, Date_2037_11_17]" in spec.facts


def test_date_literals_found_out_of_order_are_ordered(library_with_ocl_v2):
    spec = translate(library_with_ocl_v2, "Author", "self.birth = '1950-01-01' or self.birth = '1940-05-02'", scope=2)
    # Two literals fill the scope: no random dates
    assert len(spec.subsigs_of("Date")) == 2
    assert "one sig Date_1940_05_02 extends Date" in spec.sigs
    assert "one sig Date_1950_01_01 extends Date" in spec.sigs
    assert "all self:this/Author|((dateEq[self.Author_birth,Date_1950_01_01]) " \
        "or (dateEq[self.Author_birth,Date_1940_05_02]))" in spec.facts
    assert "dateLt[Date_1940_05_02, Date_1950_01_01]" in spec.facts


def test_date_scope_holds_all_literals(library_with_ocl_v2):
    spec = translate(
        library_with_ocl_v2, "Author",
        "self.birth = '1940-01-01' or self.birth = '1950-01-01' or self.birth = '1960-01-01'", scope=2,
    )
    assert len(spec.subsigs_of("Date")) == 3
    assert "one sig Date_1940_01_01 extends Date" in spec.sigs
    assert "one sig Date_1950_01_01 extends Date" in spec.sigs
    assert "one sig Date_1960_01_01 extends Date" in spec.sigs
    assert "all self:this/Author|(((dateEq[self.Author_birth,Date_1940_01_01]) " \
        "or (dateEq[self.Author_birth,Date_1950_01_01])) or (dateEq[self.Author_birth,Date_1960_01_01]))" in spec.facts
    assert "dateLt[Date_1940_01_01, Date_1950_01_01] and dateLt[Date_1950_01_01, Date_1960_01_01]" in spec.facts
    assert "3 Date" in spec.run_scopes
    assert "2 Author" in spec.run_scopes
