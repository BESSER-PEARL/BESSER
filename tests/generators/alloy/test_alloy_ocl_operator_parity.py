"""Golden-output parity tests for OCL operators translated to Alloy.

Expected strings were captured from the pre-migration translator that flattened
the ANTLR tree and parsed its tokens a second time. Each supported operator has
five expression variants so precedence, literals, and field navigation are
exercised alongside the operator itself.
"""

import pytest

from besser.generators.alloy.translate_ocl_alloy import ocl_to_alloy


INHERITS_FROM = {"Calc": ["_"]}
DATA = {"Calc": ["x:int", "y:int", "p:bool", "q:bool", "r:bool"]}

NUMERIC_OPERANDS = [
    ("self.x", "self.y", "self.Calc_x", "self.Calc_y"),
    ("self.x", "1", "self.Calc_x", "1"),
    ("2", "self.y", "2", "self.Calc_y"),
    ("(self.x + 1)", "self.y", "add[self.Calc_x, 1]", "self.Calc_y"),
    ("self.x", "(self.y * 2)", "self.Calc_x", "mul[self.Calc_y, 2]"),
]

BOOLEAN_OPERANDS = [
    ("self.p", "self.q", "isTrue[self.Calc_p]", "isTrue[self.Calc_q]"),
    ("self.p", "true", "isTrue[self.Calc_p]", "isTrue[True]"),
    ("false", "self.q", "not isTrue[True]", "isTrue[self.Calc_q]"),
    ("self.q", "self.r", "isTrue[self.Calc_q]", "isTrue[self.Calc_r]"),
    ("true", "false", "isTrue[True]", "not isTrue[True]"),
]

ARITHMETIC_OPERATORS = ["+", "-", "*", "/"]
COMPARISON_OPERATORS = ["=", "<>", "<", "<=", ">", ">="]
LOGICAL_OPERATORS = ["and", "or", "xor", "implies"]


def _fact(body: str) -> str:
    return f"fact{{ all self:this/Calc|{body}}}"


def _legacy_binary_body(operator: str, left: str, right: str) -> str:
    if operator == "+":
        return f"add[{left}, {right}]"
    if operator == "-":
        return f"minus[{left}, {right}]"
    if operator == "*":
        return f"mul[{left}, {right}]"
    if operator == "/":
        return f"div[{left}, {right}]"
    if operator == "xor":
        return f"((({left}) and not ({right})) or (not ({left}) and ({right})))"

    alloy_operator = {
        "<>": "!=",
        "implies": "=>",
    }.get(operator, operator)
    return f"({left} {alloy_operator} {right})"


def _translate(expression: str) -> str:
    return ocl_to_alloy(
        INHERITS_FROM,
        DATA,
        expression,
        context_name="Calc",
        enums={},
    )


@pytest.mark.parametrize("operator", ARITHMETIC_OPERATORS, ids=ARITHMETIC_OPERATORS)
@pytest.mark.parametrize(
    ("left", "right", "legacy_left", "legacy_right"),
    NUMERIC_OPERANDS,
    ids=["field-pair", "field-literal", "literal-field", "nested-left", "nested-right"],
)
def test_arithmetic_operator_matches_duplicate_parser_golden(
    operator, left, right, legacy_left, legacy_right
):
    expected_legacy = _fact(_legacy_binary_body(operator, legacy_left, legacy_right))

    assert _translate(f"{left} {operator} {right}") == expected_legacy


@pytest.mark.parametrize("operator", COMPARISON_OPERATORS, ids=COMPARISON_OPERATORS)
@pytest.mark.parametrize(
    ("left", "right", "legacy_left", "legacy_right"),
    NUMERIC_OPERANDS,
    ids=["field-pair", "field-literal", "literal-field", "nested-left", "nested-right"],
)
def test_comparison_operator_matches_duplicate_parser_golden(
    operator, left, right, legacy_left, legacy_right
):
    expected_legacy = _fact(_legacy_binary_body(operator, legacy_left, legacy_right))

    assert _translate(f"{left} {operator} {right}") == expected_legacy


@pytest.mark.parametrize("operator", LOGICAL_OPERATORS, ids=LOGICAL_OPERATORS)
@pytest.mark.parametrize(
    ("left", "right", "legacy_left", "legacy_right"),
    BOOLEAN_OPERANDS,
    ids=["field-pair", "field-true", "false-field", "different-fields", "true-false"],
)
def test_logical_operator_matches_duplicate_parser_golden(
    operator, left, right, legacy_left, legacy_right
):
    expected_legacy = _fact(_legacy_binary_body(operator, legacy_left, legacy_right))

    assert _translate(f"{left} {operator} {right}") == expected_legacy


NOT_CASES = [
    (
        "not self.p",
        _fact("not (self.Calc_p)"),
        _fact("not (isTrue[self.Calc_p])"),
    ),
    (
        "not self.q",
        _fact("not (self.Calc_q)"),
        _fact("not (isTrue[self.Calc_q])"),
    ),
    (
        "not true",
        _fact("not (isTrue[True])"),
        _fact("not (isTrue[True])"),
    ),
    (
        "not false",
        _fact("not (not isTrue[True])"),
        _fact("not (not isTrue[True])"),
    ),
    (
        "not (self.p and self.q)",
        _fact("not ((isTrue[self.Calc_p] and isTrue[self.Calc_q]))"),
        _fact("not ((isTrue[self.Calc_p] and isTrue[self.Calc_q]))"),
    ),
]


@pytest.mark.parametrize(
    ("expression", "legacy_golden", "current_golden"),
    NOT_CASES,
    ids=["boolean-field-p", "boolean-field-q", "true-literal", "false-literal", "nested-and"],
)
def test_not_operator_compares_current_output_with_duplicate_parser_golden(
    expression, legacy_golden, current_golden
):
    actual = _translate(expression)

    assert actual == current_golden
    assert (actual == legacy_golden) is (current_golden == legacy_golden)

