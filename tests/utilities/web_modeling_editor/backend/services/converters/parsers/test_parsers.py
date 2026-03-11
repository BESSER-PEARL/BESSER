"""
Comprehensive unit tests for the BESSER backend parsers:
  - attribute_parser.parse_attribute
  - method_parser.parse_method
  - multiplicity_parser.parse_multiplicity
  - text_parser.sanitize_text
"""

import pytest
from unittest.mock import MagicMock

from besser.utilities.web_modeling_editor.backend.services.converters.parsers.attribute_parser import parse_attribute
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.method_parser import parse_method
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.multiplicity_parser import parse_multiplicity
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.text_parser import sanitize_text

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY, Enumeration, Class, DomainModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_domain_model_with(*type_objs):
    """Build a minimal DomainModel mock whose .types contains the given objects."""
    dm = MagicMock(spec=DomainModel)
    dm.types = set(type_objs)
    return dm


# ===========================================================================
# attribute_parser.parse_attribute
# ===========================================================================

class TestParseAttribute:
    """Tests for parse_attribute(attribute_name, domain_model=None)."""

    # --- Happy path: visibility + name + primitive type -----------------------

    def test_public_symbol_with_type(self):
        vis, name, typ = parse_attribute("+ age: int")
        assert vis == "public"
        assert name == "age"
        assert typ == "int"

    def test_private_symbol_with_type(self):
        vis, name, typ = parse_attribute("- salary: float")
        assert vis == "private"
        assert name == "salary"
        assert typ == "float"

    def test_protected_symbol_with_type(self):
        vis, name, typ = parse_attribute("# name: str")
        assert vis == "protected"
        assert name == "name"
        assert typ == "str"

    def test_package_symbol_with_type(self):
        vis, name, typ = parse_attribute("~ data: bool")
        assert vis == "package"
        assert name == "data"
        assert typ == "bool"

    # --- Visibility symbol attached to name (no space) ------------------------

    def test_visibility_symbol_attached_to_name(self):
        vis, name, typ = parse_attribute("+age: int")
        assert vis == "public"
        assert name == "age"
        assert typ == "int"

    def test_private_symbol_attached_to_name(self):
        vis, name, typ = parse_attribute("-password: str")
        assert vis == "private"
        assert name == "password"
        assert typ == "str"

    # --- No visibility symbol (default public) --------------------------------

    def test_no_visibility_with_type(self):
        vis, name, typ = parse_attribute("title: string")
        assert vis == "public"
        assert name == "title"
        assert typ == "str"  # "string" normalizes to "str"

    # --- Type aliases / case insensitivity ------------------------------------

    def test_type_alias_integer(self):
        _, _, typ = parse_attribute("count: integer")
        assert typ == "int"

    def test_type_alias_double(self):
        _, _, typ = parse_attribute("value: double")
        assert typ == "float"

    def test_type_alias_boolean(self):
        _, _, typ = parse_attribute("flag: Boolean")
        assert typ == "bool"

    def test_type_date(self):
        _, _, typ = parse_attribute("created: date")
        assert typ == "date"

    def test_type_datetime(self):
        _, _, typ = parse_attribute("updated: datetime")
        assert typ == "datetime"

    def test_type_time(self):
        _, _, typ = parse_attribute("start: time")
        assert typ == "time"

    def test_type_timedelta(self):
        _, _, typ = parse_attribute("duration: timedelta")
        assert typ == "timedelta"

    def test_type_any(self):
        _, _, typ = parse_attribute("payload: any")
        assert typ == "any"

    # --- No type specified (defaults to str) ----------------------------------

    def test_name_only_defaults_to_str(self):
        vis, name, typ = parse_attribute("username")
        assert vis == "public"
        assert name == "username"
        assert typ == "str"

    def test_visibility_and_name_no_type(self):
        vis, name, typ = parse_attribute("- secret")
        assert vis == "private"
        assert name == "secret"
        assert typ == "str"

    def test_visibility_attached_no_type(self):
        vis, name, typ = parse_attribute("+visible")
        assert vis == "public"
        assert name == "visible"
        assert typ == "str"

    # --- Whitespace handling --------------------------------------------------

    def test_extra_whitespace(self):
        vis, name, typ = parse_attribute("  +   name  :  str  ")
        assert vis == "public"
        assert name == "name"
        assert typ == "str"

    # --- Domain model types (Enumeration / Class) -----------------------------

    def test_enum_type_from_domain_model(self):
        color_enum = Enumeration(name="Color", literals={})
        dm = _make_domain_model_with(color_enum)
        vis, name, typ = parse_attribute("+ status: Color", domain_model=dm)
        assert typ == "Color"

    def test_class_type_from_domain_model(self):
        address_cls = Class(name="Address", attributes={})
        dm = _make_domain_model_with(address_cls)
        vis, name, typ = parse_attribute("addr: Address", domain_model=dm)
        assert typ == "Address"

    # --- Invalid type raises ValueError ---------------------------------------

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid type"):
            parse_attribute("x: SomeUnknownType")

    def test_invalid_type_not_in_domain_model(self):
        dm = _make_domain_model_with()  # empty domain model
        with pytest.raises(ValueError, match="Invalid type"):
            parse_attribute("x: Foo", domain_model=dm)

    # --- Empty / degenerate name returns None tuple ---------------------------

    def test_empty_name_after_visibility_returns_none(self):
        vis, name, typ = parse_attribute("+: str")
        assert vis is None
        assert name is None
        assert typ is None

    def test_visibility_only_returns_none(self):
        vis, name, typ = parse_attribute("+")
        assert vis is None
        assert name is None
        assert typ is None

    # --- Space-separated visibility with type (branch in else) ----------------

    def test_space_separated_visibility_with_type(self):
        vis, name, typ = parse_attribute("+ name: str")
        assert vis == "public"
        assert name == "name"
        assert typ == "str"

    def test_space_separated_invalid_vis_defaults_to_public(self):
        # First token is not a valid visibility symbol, so it becomes the name
        vis, name, typ = parse_attribute("foo bar")
        # Without colon: parts = ["foo", "bar"]
        # parts[0]="foo" not in VISIBILITY_MAP, so visibility_symbol defaults to "+"
        assert vis == "public"
        assert name == "bar"
        assert typ == "str"


# ===========================================================================
# method_parser.parse_method
# ===========================================================================

class TestParseMethod:
    """Tests for parse_method(method_str, domain_model=None)."""

    # --- Docstring examples ---------------------------------------------------

    def test_docstring_example_notify(self):
        vis, name, params, ret = parse_method("+ notify(sms: str = 'message')")
        assert vis == "public"
        assert name == "notify"
        assert len(params) == 1
        assert params[0]["name"] == "sms"
        assert params[0]["type"] == "str"
        assert params[0]["default"] == "message"
        assert ret is None

    def test_docstring_example_findBook(self):
        # "- findBook(title: str): Book" needs a domain model with Book
        book_cls = Class(name="Book", attributes={})
        dm = _make_domain_model_with(book_cls)
        vis, name, params, ret = parse_method("- findBook(title: str): Book", domain_model=dm)
        assert vis == "private"
        assert name == "findBook"
        assert len(params) == 1
        assert params[0]["name"] == "title"
        assert params[0]["type"] == "str"
        assert ret == "Book"

    def test_docstring_example_validate(self):
        vis, name, params, ret = parse_method("validate()")
        assert vis == "public"
        assert name == "validate"
        assert params == []
        assert ret is None

    # --- Visibility parsing ---------------------------------------------------

    def test_public_visibility(self):
        vis, name, _, _ = parse_method("+ doStuff()")
        assert vis == "public"

    def test_private_visibility(self):
        vis, name, _, _ = parse_method("- doStuff()")
        assert vis == "private"

    def test_protected_visibility(self):
        vis, name, _, _ = parse_method("# doStuff()")
        assert vis == "protected"

    def test_package_visibility(self):
        vis, name, _, _ = parse_method("~ doStuff()")
        assert vis == "package"

    def test_no_visibility_defaults_to_public(self):
        vis, name, _, _ = parse_method("doStuff()")
        assert vis == "public"
        assert name == "doStuff"

    # --- No parentheses (not really a method) ---------------------------------

    def test_no_parens_returns_raw_string(self):
        vis, name, params, ret = parse_method("notAMethod")
        assert vis == "public"
        assert name == "notAMethod"
        assert params == []
        assert ret is None

    # --- Return types ---------------------------------------------------------

    def test_primitive_return_type(self):
        vis, name, params, ret = parse_method("+ getCount(): int")
        assert ret == "int"

    def test_return_type_alias(self):
        _, _, _, ret = parse_method("getFlag(): boolean")
        assert ret == "bool"

    def test_return_type_from_domain_model(self):
        person_cls = Class(name="Person", attributes={})
        dm = _make_domain_model_with(person_cls)
        _, _, _, ret = parse_method("+ getPerson(): Person", domain_model=dm)
        assert ret == "Person"

    def test_invalid_return_type_raises(self):
        with pytest.raises(ValueError, match="Invalid return type"):
            parse_method("+ calc(): UnknownType")

    def test_no_return_type(self):
        _, _, _, ret = parse_method("+ doWork()")
        assert ret is None

    def test_colon_but_empty_return_type(self):
        # "doWork():" — colon present but nothing after it
        _, _, _, ret = parse_method("doWork():")
        assert ret is None

    # --- Parameters -----------------------------------------------------------

    def test_single_typed_param(self):
        _, _, params, _ = parse_method("+ setName(name: str)")
        assert len(params) == 1
        assert params[0]["name"] == "name"
        assert params[0]["type"] == "str"

    def test_multiple_params(self):
        _, _, params, _ = parse_method("+ calc(a: int, b: float)")
        assert len(params) == 2
        assert params[0]["name"] == "a"
        assert params[0]["type"] == "int"
        assert params[1]["name"] == "b"
        assert params[1]["type"] == "float"

    def test_param_with_default_value(self):
        _, _, params, _ = parse_method("+ greet(name: str = 'World')")
        assert params[0]["name"] == "name"
        assert params[0]["type"] == "str"
        assert params[0]["default"] == "World"

    def test_param_with_double_quoted_default(self):
        _, _, params, _ = parse_method('+ greet(msg: str = "hello")')
        assert params[0]["default"] == "hello"

    def test_param_without_type_annotation(self):
        # "foo" alone has no colon or equals — type defaults to "any"
        _, _, params, _ = parse_method("doStuff(foo)")
        assert len(params) == 1
        assert params[0]["name"] == "foo"
        assert params[0]["type"] == "any"

    def test_param_with_default_but_no_type(self):
        _, _, params, _ = parse_method("doStuff(x = 42)")
        assert params[0]["name"] == "x"
        assert params[0]["default"] == "42"
        assert params[0]["type"] == "any"  # no colon, so keeps default 'any'

    def test_param_type_aliases(self):
        _, _, params, _ = parse_method("f(a: integer, b: boolean, c: double)")
        assert params[0]["type"] == "int"
        assert params[1]["type"] == "bool"
        assert params[2]["type"] == "float"

    def test_param_type_from_domain_model(self):
        color_enum = Enumeration(name="Color", literals={})
        dm = _make_domain_model_with(color_enum)
        _, _, params, _ = parse_method("+ paint(c: Color)", domain_model=dm)
        assert params[0]["type"] == "Color"

    def test_invalid_param_type_raises(self):
        with pytest.raises(ValueError, match="Invalid type"):
            parse_method("+ doStuff(x: Nonexistent)")

    def test_empty_params(self):
        _, _, params, _ = parse_method("+ doStuff()")
        assert params == []

    # --- Nested parentheses in defaults ---------------------------------------

    def test_nested_parens_in_default(self):
        _, name, params, _ = parse_method('+ run(cmd: str = "a(b)")')
        assert name == "run"
        assert len(params) == 1
        assert params[0]["name"] == "cmd"
        assert params[0]["default"] == "a(b)"

    # --- Multiple params with mixed features ----------------------------------

    def test_mixed_params(self):
        _, _, params, _ = parse_method("+ init(a: int, b, c: str = 'x')")
        assert len(params) == 3
        assert params[0]["name"] == "a"
        assert params[0]["type"] == "int"
        assert params[1]["name"] == "b"
        assert params[1]["type"] == "any"
        assert params[2]["name"] == "c"
        assert params[2]["type"] == "str"
        assert params[2]["default"] == "x"

    # --- Visibility attached to method name (no space) ------------------------

    def test_visibility_attached_no_space(self):
        vis, name, _, _ = parse_method("+run()")
        assert vis == "public"
        assert name == "run"

    def test_private_attached(self):
        vis, name, _, _ = parse_method("-_internal()")
        assert vis == "private"
        assert name == "_internal"

    # --- Whitespace handling --------------------------------------------------

    def test_leading_trailing_whitespace(self):
        vis, name, params, ret = parse_method("  + greet( name : str ) : str  ")
        assert vis == "public"
        assert name == "greet"
        assert ret == "str"

    # --- Return type with enum from domain model ------------------------------

    def test_enum_return_type(self):
        status_enum = Enumeration(name="Status", literals={})
        dm = _make_domain_model_with(status_enum)
        _, _, _, ret = parse_method("+ getStatus(): Status", domain_model=dm)
        assert ret == "Status"


# ===========================================================================
# multiplicity_parser.parse_multiplicity
# ===========================================================================

class TestParseMultiplicity:
    """Tests for parse_multiplicity(multiplicity_str)."""

    # --- None / empty / falsy input defaults to 1..1 --------------------------

    def test_none_defaults_to_1_1(self):
        m = parse_multiplicity(None)
        assert m.min == 1
        assert m.max == 1

    def test_empty_string_defaults_to_1_1(self):
        m = parse_multiplicity("")
        assert m.min == 1
        assert m.max == 1

    # --- Single star = 0..* ---------------------------------------------------

    def test_single_star(self):
        m = parse_multiplicity("*")
        assert m.min == 0
        assert m.max == UNLIMITED_MAX_MULTIPLICITY

    # --- Single integer = N..N ------------------------------------------------

    def test_single_zero_raises(self):
        """Single '0' means 0..0 which is invalid (max must be > 0)."""
        with pytest.raises(ValueError):
            parse_multiplicity("0")

    def test_single_one(self):
        m = parse_multiplicity("1")
        assert m.min == 1
        assert m.max == 1

    def test_single_five(self):
        m = parse_multiplicity("5")
        assert m.min == 5
        assert m.max == 5

    # --- Range notation -------------------------------------------------------

    def test_range_0_to_1(self):
        m = parse_multiplicity("0..1")
        assert m.min == 0
        assert m.max == 1

    def test_range_1_to_many(self):
        m = parse_multiplicity("1..*")
        assert m.min == 1
        assert m.max == UNLIMITED_MAX_MULTIPLICITY

    def test_range_0_to_many(self):
        m = parse_multiplicity("0..*")
        assert m.min == 0
        assert m.max == UNLIMITED_MAX_MULTIPLICITY

    def test_range_2_to_5(self):
        m = parse_multiplicity("2..5")
        assert m.min == 2
        assert m.max == 5

    def test_range_1_to_1(self):
        m = parse_multiplicity("1..1")
        assert m.min == 1
        assert m.max == 1

    def test_range_star_to_star(self):
        # "*..* " — parts[0]="*" → min=0; parts[1]="*" → max=UNLIMITED
        m = parse_multiplicity("*..*")
        assert m.min == 0
        assert m.max == UNLIMITED_MAX_MULTIPLICITY

    def test_range_with_empty_max(self):
        # "1.." — parts[1]="" → max=UNLIMITED
        m = parse_multiplicity("1..")
        assert m.min == 1
        assert m.max == UNLIMITED_MAX_MULTIPLICITY

    # --- Invalid / unparsable input defaults to 1..1 --------------------------

    def test_non_numeric_defaults_to_1_1(self):
        m = parse_multiplicity("abc")
        assert m.min == 1
        assert m.max == 1

    def test_range_non_numeric_defaults_to_1_1(self):
        m = parse_multiplicity("a..b")
        assert m.min == 1
        assert m.max == 1

    # --- Return type is Multiplicity ------------------------------------------

    def test_returns_multiplicity_instance(self):
        m = parse_multiplicity("1..*")
        assert isinstance(m, Multiplicity)


# ===========================================================================
# text_parser.sanitize_text
# ===========================================================================

class TestSanitizeText:
    """Tests for sanitize_text(text)."""

    # --- Pass-through for normal strings --------------------------------------

    def test_normal_ascii(self):
        assert sanitize_text("hello world") == "hello world"

    def test_empty_string(self):
        assert sanitize_text("") == ""

    # --- Non-string input returned as-is --------------------------------------

    def test_none_passthrough(self):
        assert sanitize_text(None) is None

    def test_int_passthrough(self):
        assert sanitize_text(42) == 42

    def test_list_passthrough(self):
        val = [1, 2]
        assert sanitize_text(val) is val

    # --- Control character removal --------------------------------------------

    def test_removes_null_byte(self):
        assert sanitize_text("ab\x00cd") == "abcd"

    def test_removes_bell(self):
        assert sanitize_text("ab\x07cd") == "abcd"

    def test_removes_backspace(self):
        assert sanitize_text("ab\x08cd") == "abcd"

    def test_removes_vertical_tab(self):
        assert sanitize_text("ab\x0bcd") == "abcd"

    def test_removes_form_feed(self):
        assert sanitize_text("ab\x0ccd") == "abcd"

    def test_removes_escape(self):
        assert sanitize_text("ab\x1bcd") == "abcd"

    def test_removes_delete(self):
        assert sanitize_text("ab\x7fcd") == "abcd"

    def test_preserves_newline(self):
        # \n = 0x0a, not in the stripped range
        assert sanitize_text("ab\ncd") == "ab\ncd"

    def test_preserves_carriage_return(self):
        # \r = 0x0d, not in the stripped range
        assert sanitize_text("ab\rcd") == "ab\rcd"

    def test_preserves_tab(self):
        # \t = 0x09, not in the stripped range
        assert sanitize_text("ab\tcd") == "ab\tcd"

    # --- Single quote escaping ------------------------------------------------

    def test_escapes_single_quote(self):
        assert sanitize_text("it's") == "it\\'s"

    def test_escapes_multiple_single_quotes(self):
        assert sanitize_text("it's a 'test'") == "it\\'s a \\'test\\'"

    # --- Unicode handling -----------------------------------------------------

    def test_unicode_letters_preserved(self):
        assert sanitize_text("cafe\u0301") is not None  # "cafe" + combining acute
        result = sanitize_text("cafe\u0301")
        # NFKD normalization decomposes; the combining character stays as-is
        assert "cafe" in result

    def test_unicode_emoji_preserved(self):
        # Emoji should survive (not control characters)
        result = sanitize_text("hello \U0001f600")
        assert "hello" in result

    def test_fullwidth_digits_normalized(self):
        # NFKD normalizes fullwidth digits to ASCII
        result = sanitize_text("\uff11\uff12\uff13")  # fullwidth 1, 2, 3
        assert result == "123"

    # --- Combined control chars and quotes ------------------------------------

    def test_mixed_control_and_quotes(self):
        result = sanitize_text("\x00it's\x07 fine\x7f")
        assert result == "it\\'s fine"

    # --- Whitespace strings ---------------------------------------------------

    def test_whitespace_only(self):
        assert sanitize_text("   ") == "   "

    def test_mixed_whitespace(self):
        assert sanitize_text(" \t \n ") == " \t \n "


# ===========================================================================
# Cross-cutting / integration-ish tests
# ===========================================================================

class TestParserIntegration:
    """Lightweight integration tests combining parsers as they would be used together."""

    def test_attribute_then_method_same_class(self):
        """Parse an attribute and a method that reference the same custom type."""
        book_cls = Class(name="Book", attributes={})
        dm = _make_domain_model_with(book_cls)

        vis_a, name_a, type_a = parse_attribute("+ library: Book", domain_model=dm)
        assert type_a == "Book"

        vis_m, name_m, params_m, ret_m = parse_method("+ findBook(title: str): Book", domain_model=dm)
        assert ret_m == "Book"

    def test_multiplicity_common_uml_patterns(self):
        """Verify the most common UML multiplicity notations."""
        cases = {
            "1": (1, 1),
            "1..1": (1, 1),
            "0..1": (0, 1),
            "1..*": (1, UNLIMITED_MAX_MULTIPLICITY),
            "0..*": (0, UNLIMITED_MAX_MULTIPLICITY),
            "*": (0, UNLIMITED_MAX_MULTIPLICITY),
        }
        for notation, (expected_min, expected_max) in cases.items():
            m = parse_multiplicity(notation)
            assert m.min == expected_min, f"Failed for '{notation}': min={m.min}"
            assert m.max == expected_max, f"Failed for '{notation}': max={m.max}"

    def test_sanitize_before_attribute_parse(self):
        """sanitize_text should clean an attribute string before parsing."""
        raw = "\x00+ name\x07: str"
        cleaned = sanitize_text(raw)
        vis, name, typ = parse_attribute(cleaned)
        assert name == "name"
        assert typ == "str"
