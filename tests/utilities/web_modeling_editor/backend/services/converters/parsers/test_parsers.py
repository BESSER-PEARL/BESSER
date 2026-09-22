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
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

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

    # --- Visibility + name + primitive type (parametrized) -------------------

    @pytest.mark.parametrize("attr_str, expected_vis, expected_name, expected_type", [
        ("+ age: int",       "public",    "age",      "int"),
        ("- salary: float",  "private",   "salary",   "float"),
        ("# name: str",      "protected", "name",     "str"),
        ("~ data: bool",     "package",   "data",     "bool"),
    ], ids=["public", "private", "protected", "package"])
    def test_visibility_with_type(self, attr_str, expected_vis, expected_name, expected_type):
        vis, name, typ = parse_attribute(attr_str)
        assert vis == expected_vis
        assert name == expected_name
        assert typ == expected_type

    # --- Visibility symbol attached to name (no space) -----------------------

    @pytest.mark.parametrize("attr_str, expected_vis, expected_name, expected_type", [
        ("+age: int",      "public",  "age",      "int"),
        ("-password: str", "private", "password", "str"),
    ], ids=["public-attached", "private-attached"])
    def test_visibility_attached_to_name(self, attr_str, expected_vis, expected_name, expected_type):
        vis, name, typ = parse_attribute(attr_str)
        assert vis == expected_vis
        assert name == expected_name
        assert typ == expected_type

    # --- No visibility symbol (default public) --------------------------------

    def test_no_visibility_with_type(self):
        vis, name, typ = parse_attribute("title: string")
        assert vis == "public"
        assert name == "title"
        assert typ == "str"  # "string" normalizes to "str"

    # --- Type aliases / case insensitivity (parametrized) --------------------

    @pytest.mark.parametrize("attr_str, expected_type", [
        ("count: integer",      "int"),
        ("value: double",       "float"),
        ("flag: Boolean",       "bool"),
        ("created: date",       "date"),
        ("updated: datetime",   "datetime"),
        ("start: time",         "time"),
        ("duration: timedelta", "timedelta"),
        ("payload: any",        "any"),
    ], ids=["integer->int", "double->float", "Boolean->bool", "date", "datetime", "time", "timedelta", "any"])
    def test_type_aliases(self, attr_str, expected_type):
        _, _, typ = parse_attribute(attr_str)
        assert typ == expected_type

    # --- No type specified (defaults to str) (parametrized) -------------------

    @pytest.mark.parametrize("attr_str, expected_vis, expected_name", [
        ("username",  "public",  "username"),
        ("- secret",  "private", "secret"),
        ("+visible",  "public",  "visible"),
    ], ids=["name-only", "vis-and-name", "vis-attached-no-type"])
    def test_defaults_to_str_when_no_type(self, attr_str, expected_vis, expected_name):
        vis, name, typ = parse_attribute(attr_str)
        assert vis == expected_vis
        assert name == expected_name
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

    @pytest.mark.parametrize("attr_str", ["+: str", "+"], ids=["empty-name-with-type", "visibility-only"])
    def test_degenerate_input_returns_none(self, attr_str):
        vis, name, typ = parse_attribute(attr_str)
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

    # --- Visibility parsing (parametrized) ------------------------------------

    @pytest.mark.parametrize("method_str, expected_vis", [
        ("+ doStuff()",  "public"),
        ("- doStuff()",  "private"),
        ("# doStuff()",  "protected"),
        ("~ doStuff()",  "package"),
        ("doStuff()",    "public"),
    ], ids=["public", "private", "protected", "package", "default-public"])
    def test_method_visibility(self, method_str, expected_vis):
        vis, _, _, _ = parse_method(method_str)
        assert vis == expected_vis

    def test_no_visibility_defaults_to_public_with_name(self):
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

    # --- Return types (parametrized) ------------------------------------------

    @pytest.mark.parametrize("method_str, expected_ret", [
        ("+ getCount(): int",   "int"),
        ("getFlag(): boolean",  "bool"),
        ("+ doWork()",          None),
        ("doWork():",           None),
    ], ids=["primitive-int", "alias-boolean->bool", "no-return", "colon-empty-return"])
    def test_return_type(self, method_str, expected_ret):
        _, _, _, ret = parse_method(method_str)
        assert ret == expected_ret

    def test_return_type_from_domain_model(self):
        person_cls = Class(name="Person", attributes={})
        dm = _make_domain_model_with(person_cls)
        _, _, _, ret = parse_method("+ getPerson(): Person", domain_model=dm)
        assert ret == "Person"

    def test_invalid_return_type_raises(self):
        with pytest.raises(ValueError, match="Invalid return type"):
            parse_method("+ calc(): UnknownType")

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
        # "foo" alone has no colon or equals -- type defaults to "any"
        _, _, params, _ = parse_method("doStuff(foo)")
        assert len(params) == 1
        assert params[0]["name"] == "foo"
        assert params[0]["type"] == "any"

    def test_param_with_default_but_no_type(self):
        _, _, params, _ = parse_method("doStuff(x = 42)")
        assert params[0]["name"] == "x"
        assert params[0]["default"] == "42"
        assert params[0]["type"] == "any"  # no colon, so keeps default 'any'

    # --- Parameter type aliases (parametrized) --------------------------------

    @pytest.mark.parametrize("param_str, expected_type", [
        ("integer", "int"),
        ("boolean", "bool"),
        ("double",  "float"),
    ], ids=["integer->int", "boolean->bool", "double->float"])
    def test_param_type_aliases(self, param_str, expected_type):
        _, _, params, _ = parse_method(f"f(x: {param_str})")
        assert params[0]["type"] == expected_type

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

    # --- Visibility attached to method name (no space) (parametrized) ---------

    @pytest.mark.parametrize("method_str, expected_vis, expected_name", [
        ("+run()",       "public",  "run"),
        ("-_internal()", "private", "_internal"),
    ], ids=["public-attached", "private-attached"])
    def test_visibility_attached_no_space(self, method_str, expected_vis, expected_name):
        vis, name, _, _ = parse_method(method_str)
        assert vis == expected_vis
        assert name == expected_name

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

    @pytest.mark.parametrize("mult_str", [None, ""], ids=["None", "empty-string"])
    def test_falsy_input_defaults_to_1_1(self, mult_str):
        m = parse_multiplicity(mult_str)
        assert m.min == 1
        assert m.max == 1

    # --- Single star = 0..* ---------------------------------------------------

    def test_single_star(self):
        m = parse_multiplicity("*")
        assert m.min == 0
        assert m.max == UNLIMITED_MAX_MULTIPLICITY

    # --- Single integer = N..N (parametrized) ---------------------------------

    def test_single_zero_raises(self):
        """Single '0' means 0..0 which is invalid (max must be > 0)."""
        with pytest.raises(ValueError):
            parse_multiplicity("0")

    @pytest.mark.parametrize("mult_str, expected", [
        ("1", (1, 1)),
        ("5", (5, 5)),
    ], ids=["one", "five"])
    def test_single_integer(self, mult_str, expected):
        m = parse_multiplicity(mult_str)
        assert m.min == expected[0]
        assert m.max == expected[1]

    # --- Range notation (parametrized) ----------------------------------------

    @pytest.mark.parametrize("mult_str, expected_min, expected_max", [
        ("0..1", 0, 1),
        ("1..*", 1, UNLIMITED_MAX_MULTIPLICITY),
        ("0..*", 0, UNLIMITED_MAX_MULTIPLICITY),
        ("2..5", 2, 5),
        ("1..1", 1, 1),
        ("*..*", 0, UNLIMITED_MAX_MULTIPLICITY),
        ("1..",  1, UNLIMITED_MAX_MULTIPLICITY),
    ], ids=["0..1", "1..*", "0..*", "2..5", "1..1", "*..*", "1..empty"])
    def test_range_notation(self, mult_str, expected_min, expected_max):
        m = parse_multiplicity(mult_str)
        assert m.min == expected_min
        assert m.max == expected_max

    # --- Invalid / unparsable input raises ConversionError --------------------

    @pytest.mark.parametrize("mult_str", [
        "abc",
        "a..b",
        "6dsfdsfsdf..1",
        "1..b",
        "1..2..3",
    ], ids=["non-numeric", "range-non-numeric", "garbage-min", "garbage-max", "too-many-dots"])
    def test_invalid_input_raises(self, mult_str):
        with pytest.raises(ConversionError):
            parse_multiplicity(mult_str)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ConversionError):
            parse_multiplicity("5..1")

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

    # --- Non-string input returned as-is (parametrized) -----------------------

    @pytest.mark.parametrize("value", [None, 42, [1, 2]], ids=["None", "int", "list"])
    def test_non_string_passthrough(self, value):
        result = sanitize_text(value)
        if value is None:
            assert result is None
        else:
            assert result == value

    # --- Control character removal (parametrized) -----------------------------

    @pytest.mark.parametrize("char, char_name", [
        ("\x00", "null byte"),
        ("\x07", "bell"),
        ("\x08", "backspace"),
        ("\x0b", "vertical tab"),
        ("\x0c", "form feed"),
        ("\x1b", "escape"),
        ("\x7f", "delete"),
    ], ids=["null", "bell", "backspace", "vtab", "formfeed", "escape", "delete"])
    def test_removes_control_character(self, char, char_name):
        assert sanitize_text(f"ab{char}cd") == "abcd"

    # --- Whitespace characters that should be preserved (parametrized) --------

    @pytest.mark.parametrize("char, char_name", [
        ("\n", "newline"),
        ("\r", "carriage return"),
        ("\t", "tab"),
    ], ids=["newline", "carriage-return", "tab"])
    def test_preserves_whitespace_character(self, char, char_name):
        assert sanitize_text(f"ab{char}cd") == f"ab{char}cd"

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

    @pytest.mark.parametrize("notation, expected_min, expected_max", [
        ("1",    1, 1),
        ("1..1", 1, 1),
        ("0..1", 0, 1),
        ("1..*", 1, UNLIMITED_MAX_MULTIPLICITY),
        ("0..*", 0, UNLIMITED_MAX_MULTIPLICITY),
        ("*",    0, UNLIMITED_MAX_MULTIPLICITY),
    ], ids=["1", "1..1", "0..1", "1..*", "0..*", "*"])
    def test_multiplicity_common_uml_patterns(self, notation, expected_min, expected_max):
        """Verify the most common UML multiplicity notations."""
        m = parse_multiplicity(notation)
        assert m.min == expected_min
        assert m.max == expected_max

    def test_sanitize_before_attribute_parse(self):
        """sanitize_text should clean an attribute string before parsing."""
        raw = "\x00+ name\x07: str"
        cleaned = sanitize_text(raw)
        vis, name, typ = parse_attribute(cleaned)
        assert name == "name"
        assert typ == "str"
