"""
OCL Constraint Utilities for Pydantic Generator

This module uses the B-OCL parser infrastructure to extract validation
information from OCL constraint expressions for use in Pydantic field validators.

It leverages the bocl ANTLR-based visitor parser and traverses the parsed
OCL expression tree.

Three result shapes are produced:

* Field validator (single property):
  ``{'property', 'python_operator', 'value_repr', 'message', 'message_repr', ...}``
  or ``{'property', 'python_expression', 'message', 'message_repr', ...}``
* Model validator (two or more properties of the same class):
  ``{'model_expression', 'message_repr', 'validator_name', ...}``
* Skipped (collections, relationships, anything not confidently translatable):
  ``{'skipped': True}``

Every generated expression is compiled with :func:`compile` and screened with
an AST white-list before being handed to the template, so the generator can
never emit Python code that does not parse.
"""

import ast
import re
from typing import Optional, Dict, Any, List
from antlr4 import InputStream, CommonTokenStream
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.visitor import BOCLVisitorImpl
from besser.BUML.notations.ocl.error_handling import BOCLErrorListener

from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, PropertyCallExpression,
    IntegerLiteralExpression, RealLiteralExpression,
    StringLiteralExpression, BooleanLiteralExpression,
    DateLiteralExpression, InfixOperator
)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: Matches a single- or double-quoted string literal (honouring backslash escapes).
_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

#: OCL constructs that cannot be expressed as a Pydantic validator on a Create model.
_COLLECTION_MARKERS = re.compile(
    r"->"
    r"|\b(?:forAll|exists|select|reject|collect|iterate|closure|allInstances"
    r"|oclIsKindOf|oclIsTypeOf|oclAsType|oclIsUndefined|oclIsNew"
    r"|isEmpty|notEmpty|implies|xor)\b"
    r"|\.\s*(?:size|sum|first|last|count|asSet|asBag|asSequence|asOrderedSet)\s*\("
)

#: OCL leftovers that must never survive into generated Python.
_RESIDUAL_OCL = re.compile(
    r"->"
    r"|\.\s*(?:matches|size|sum|toUpper|toLower|substring|indexOf|concat"
    r"|includes|excludes|oclIs\w+|oclAs\w+)\s*\("
)

#: ``<target>.matches('<regex>')`` where target is ``self.<prop>`` or the ``v`` placeholder.
_MATCHES_CALL = re.compile(
    r"(?P<target>self\s*\.\s*\w+|\bv\b)\s*\.\s*matches\s*\(\s*"
    r"(?:'(?P<sq>(?:[^'\\]|\\.)*)'|\"(?P<dq>(?:[^\"\\]|\\.)*)\")\s*\)"
)

#: A constraint body that consists of nothing but a single ``matches`` call.
_PURE_MATCHES = re.compile(
    r"^\s*self\s*\.\s*(?P<prop>\w+)\s*\.\s*matches\s*\(\s*"
    r"(?:'(?P<sq>(?:[^'\\]|\\.)*)'|\"(?P<dq>(?:[^\"\\]|\\.)*)\")\s*\)\s*$"
)

#: Names the generated validator bodies are allowed to reference.
_ALLOWED_NAMES = frozenset({'v', 'self', 're'})

#: AST nodes allowed inside a generated validator expression.
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Constant,
    ast.Load, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
)

#: OCL string escapes that are safe to resolve. ``\b`` is deliberately absent:
#: inside a regex it means "word boundary", not "backspace".
_OCL_ESCAPES = {'\\': '\\', "'": "'", '"': '"', 'n': '\n', 't': '\t', 'r': '\r'}

#: Emitted for constraints the generator refuses to translate.
SKIP_COMMENT_TEMPLATE = (
    "# NOTE: OCL constraint '{name}' involves collections/relationships "
    "and is not enforced by this Create model."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ocl_constraint(constraint, domain_model) -> Optional[Dict[str, Any]]:
    """
    Parse an OCL constraint using the BESSER OCL parser and extract validation info.

    This uses the ANTLR-based parser to properly parse OCL expressions,
    similar to the B-OCL-Interpreter's update_logical_exp function, and falls
    back to a regex-based translation when the tree cannot be exploited.

    Args:
        constraint: A Constraint object from the domain model
        domain_model: The DomainModel for context resolution

    Returns:
        A dict describing a field validator, a model validator, or a skipped
        constraint (``{'skipped': True}``). Never returns broken Python.
    """
    constraint_name = getattr(constraint, 'name', None)
    expression = constraint.expression
    body = _extract_expression_body(expression)

    if not body or _is_untranslatable(body):
        return {'skipped': True}

    result = _parse_with_antlr(constraint, domain_model)

    if result is None:
        result = _fallback_parse(expression)

    if result is None:
        result = _model_level_parse(body, constraint_name)

    if result is None:
        return {'skipped': True}

    return _finalize_result(result, constraint_name)


def get_constraints_for_class(
    constraints: set,
    class_name: str,
    domain_model,
    include_model_level: bool = False,
    include_skipped: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get all parsed constraints for a specific class.

    Args:
        constraints: Set of Constraint objects from the domain model
        class_name: Name of the class to get constraints for
        domain_model: The DomainModel for parsing context
        include_model_level: Include multi-property constraints, which need a
            model-level validator. Callers that only render per-field validators
            (e.g. the Django generator) leave this off.
        include_skipped: Include untranslatable constraints as ``{'skipped': True}``
            so the caller can leave a trace comment instead of dropping them.

    Returns:
        List of parsed constraint dicts, ordered by constraint name so the
        generated code is deterministic.
    """
    parsed = []
    used_validator_names = set()

    relevant = [
        c for c in constraints
        if c.context is not None and c.context.name == class_name and c.language == "OCL"
    ]

    for constraint in sorted(relevant, key=lambda c: (c.name or "")):
        result = parse_ocl_constraint(constraint, domain_model)
        if not result:
            continue
        if result.get('skipped') and not include_skipped:
            continue
        if 'model_expression' in result and not include_model_level:
            continue
        result['constraint_name'] = constraint.name
        if 'validator_name' in result:
            result['validator_name'] = _unique_name(
                result['validator_name'], used_validator_names
            )
        parsed.append(result)

    return parsed


def build_constraints_map(
    domain_model,
    include_model_level: bool = False,
    include_skipped: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping from class names to their parsed OCL constraints.

    Args:
        domain_model: The DomainModel object
        include_model_level: Include multi-property (model-level) constraints
        include_skipped: Include untranslatable constraints as ``{'skipped': True}``

    Returns:
        Dict mapping class name -> list of parsed constraints
    """
    constraints_map = {}

    if not domain_model.constraints:
        return constraints_map

    for cls in domain_model.get_classes():
        class_constraints = get_constraints_for_class(
            domain_model.constraints,
            cls.name,
            domain_model,
            include_model_level=include_model_level,
            include_skipped=include_skipped,
        )
        if class_constraints:
            constraints_map[cls.name] = class_constraints

    return constraints_map


# ---------------------------------------------------------------------------
# ANTLR-based extraction
# ---------------------------------------------------------------------------

def _parse_with_antlr(constraint, domain_model) -> Optional[Dict[str, Any]]:
    """
    Run the ANTLR visitor parser and try to extract validation info from the tree.

    Returns None when the parser errors out or the tree yields nothing usable.
    """
    try:
        input_stream = InputStream(constraint.expression)

        lexer = BOCLLexer(input_stream)
        lexer.removeErrorListeners()
        error_listener = BOCLErrorListener()
        lexer.addErrorListener(error_listener)

        stream = CommonTokenStream(lexer)
        parser = BOCLParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        tree = parser.oclFile()

        if error_listener.has_errors():
            return None

        visitor = BOCLVisitorImpl(domain_model, None, constraint.context)
        root = visitor.visit(tree)

        if root is None:
            return None

        return _extract_from_tree(root)
    except Exception:
        return None


def _extract_from_tree(tree) -> Optional[Dict[str, Any]]:
    """
    Extract validation info from a parsed OCL expression tree.

    Handles simple expressions of the form: self.property <operator> value
    Similar to the update_logical_exp function in B-OCL-Interpreter.

    Args:
        tree: The root of the parsed OCL expression tree

    Returns:
        Dict with validation info or None
    """
    if isinstance(tree, OperationCallExpression):
        result = _extract_from_operation(tree)
        if result:
            return result

    # Check if we need to traverse source to find the operation
    if hasattr(tree, 'source') and tree.source is not None:
        return _extract_from_tree(tree.source)

    return None


def _extract_from_operation(op_exp: OperationCallExpression) -> Optional[Dict[str, Any]]:
    """
    Extract validation info from an OperationCallExpression.

    Based on the B-OCL-Interpreter's handling in update_logical_exp.

    Args:
        op_exp: An OperationCallExpression node

    Returns:
        Dict with validation info or None
    """
    args = op_exp.arguments

    property_name = None
    operator = None
    value = None
    value_repr = None
    value_type = None

    for arg in args:
        if isinstance(arg, PropertyCallExpression):
            property_name = arg.property.name
        elif isinstance(arg, InfixOperator):
            operator = arg.get_infix_operator()
        elif isinstance(arg, IntegerLiteralExpression):
            value = arg.value
            value_repr = str(arg.value)
            value_type = 'int'
        elif isinstance(arg, RealLiteralExpression):
            value = arg.value
            value_repr = str(arg.value)
            value_type = 'float'
        elif isinstance(arg, StringLiteralExpression):
            value = arg.value
            value_repr = repr(arg.value)
            value_type = 'str'
        elif isinstance(arg, BooleanLiteralExpression):
            value = arg.value
            value_repr = str(arg.value)
            value_type = 'bool'
        elif isinstance(arg, DateLiteralExpression):
            value = arg.value
            value_repr = repr(str(arg.value))
            value_type = 'date'

    # Also check the referredOperation for the operator
    if operator is None and op_exp.referredOperation is not None:
        if hasattr(op_exp.referredOperation, 'get_infix_operator'):
            operator = op_exp.referredOperation.get_infix_operator()
        elif hasattr(op_exp.referredOperation, 'operator'):
            operator = op_exp.referredOperation.operator

    if property_name is None or operator is None or value is None:
        return None

    python_operator = _OPERATOR_MAP.get(operator, operator)

    return {
        'property': property_name,
        'operator': operator,
        'python_operator': python_operator,
        'value': value,
        'value_repr': value_repr,
        'value_type': value_type
    }


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

#: Map OCL operators to Python operators.
_OPERATOR_MAP = {
    '>': '>',
    '<': '<',
    '>=': '>=',
    '<=': '<=',
    '=': '==',
    '<>': '!='
}


def _extract_expression_body(expression: str) -> str:
    """
    Extract the OCL expression body from a full constraint string.
    """
    expression = expression.strip()
    match = re.search(
        r"context\s+\w+\s+inv(?:\s+\w+)?\s*:\s*(.+)",
        expression,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return expression


def _strip_string_literals(text: str) -> str:
    """
    Blank out quoted string literals so structural checks ignore their contents.
    """
    return _STRING_LITERAL.sub("''", text)


def _is_untranslatable(body: str) -> bool:
    """
    Report whether an OCL body uses constructs the Pydantic generator cannot express.
    """
    return bool(_COLLECTION_MARKERS.search(_strip_string_literals(body)))


def _apply_outside_quotes(text: str, transform) -> str:
    """
    Apply a transformation function only to text outside quoted string literals.
    """
    pieces = []
    position = 0
    for match in _STRING_LITERAL.finditer(text):
        pieces.append(transform(text[position:match.start()]))
        pieces.append(match.group(0))
        position = match.end()
    pieces.append(transform(text[position:]))
    return "".join(pieces)


def _replace_outside_quotes(text: str, pattern: str, repl: str) -> str:
    """
    Regex replace outside quoted string literals.
    """
    return _apply_outside_quotes(text, lambda part: re.sub(pattern, repl, part))


def _collapse_whitespace_outside_quotes(text: str) -> str:
    """
    Collapse consecutive whitespace outside quoted string literals.
    """
    return _apply_outside_quotes(text, lambda part: re.sub(r"\s+", " ", part)).strip()


def _unescape_ocl_string(raw: str) -> str:
    """
    Resolve the escape sequences of an OCL string literal.

    ``\\b`` is left untouched on purpose: within a regex it is a word boundary.
    """
    out = []
    index = 0
    length = len(raw)
    while index < length:
        char = raw[index]
        if char == '\\' and index + 1 < length and raw[index + 1] in _OCL_ESCAPES:
            out.append(_OCL_ESCAPES[raw[index + 1]])
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _regex_literal(regex: str) -> str:
    """
    Render a regex as a Python literal, preferring a raw string when it is safe.

    A raw string cannot end with an odd number of backslashes and cannot contain
    its own quote character or a newline, so those cases fall back to ``repr``.
    """
    trailing_backslashes = len(regex) - len(regex.rstrip('\\'))
    if trailing_backslashes % 2 == 0 and '\n' not in regex and '\r' not in regex:
        if "'" not in regex:
            return f"r'{regex}'"
        if '"' not in regex:
            return f'r"{regex}"'
    return repr(regex)


def _translate_matches(text: str) -> str:
    """
    Translate OCL ``<target>.matches('<regex>')`` calls into Python ``re.match`` calls.
    """
    def replace(match: "re.Match") -> str:
        raw = match.group('sq')
        if raw is None:
            raw = match.group('dq')
        regex = _unescape_ocl_string(raw)
        target = re.sub(r"\s+", "", match.group('target'))
        return f"re.match({_regex_literal(regex)}, {target}) is not None"

    return _MATCHES_CALL.sub(replace, text)


def _normalize_operators(expression: str) -> str:
    """
    Rewrite OCL comparison operators as Python operators, outside string literals.
    """
    normalized = _replace_outside_quotes(expression, r"<>", "!=")
    normalized = _replace_outside_quotes(normalized, r"(?<![<>=!])=(?!=)", "==")
    return normalized


def _normalize_ocl_expression(expression: str, property_name: str) -> str:
    """
    Normalize an OCL expression into a Python expression for a single property.

    The property reference becomes the ``v`` placeholder used by field validators.
    """
    normalized = _normalize_operators(expression)
    normalized = _replace_outside_quotes(
        normalized,
        rf"\bself\.{re.escape(property_name)}\b",
        "v"
    )
    normalized = _collapse_whitespace_outside_quotes(normalized)
    return _translate_matches(normalized)


def _normalize_model_expression(expression: str) -> str:
    """
    Normalize an OCL expression into a Python expression that keeps ``self.<prop>``
    references, as used by model-level validators.
    """
    normalized = _normalize_operators(expression)
    normalized = _collapse_whitespace_outside_quotes(normalized)
    normalized = _replace_outside_quotes(normalized, r"\bself\s*\.\s*(\w+)", r"self.\1")
    return _translate_matches(normalized)


def _safe_identifier(name: str) -> str:
    """
    Turn a constraint name into a valid Python identifier.
    """
    identifier = re.sub(r"\W", "_", name or "")
    if not identifier:
        identifier = "constraint"
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def _unique_name(base: str, used: set) -> str:
    """
    Return `base`, suffixed with a counter if it has already been handed out.
    """
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Regex-based parsing
# ---------------------------------------------------------------------------

def _fallback_parse(expression: str) -> Optional[Dict[str, Any]]:
    """
    Fallback regex-based parser for single-property OCL constraint expressions.
    Used when the ANTLR parser doesn't produce usable results.

    Args:
        expression: The OCL constraint expression string

    Returns:
        A dict with keys: property, operator, value, python_operator
        or for compound expressions: property, python_expression, message.
        Returns None if the expression references anything but one property.
    """
    # Extract the core OCL expression body (after "inv:")
    expression_body = _extract_expression_body(expression)
    if not expression_body:
        return None

    # Identify properties referenced in the expression
    properties = re.findall(r"\bself\.(\w+)\b", expression_body)
    if not properties:
        return None

    unique_properties = set(properties)
    if len(unique_properties) != 1:
        # Multi-property constraints are handled by _model_level_parse
        return None

    property_name = unique_properties.pop()
    multiple_references = len(properties) > 1

    # Pattern to match: self.<property> <operator> <value>
    simple_pattern = rf"self\.{re.escape(property_name)}\s*(>=|<=|<>|>|<|=)\s*(.+)"
    match = re.fullmatch(simple_pattern, expression_body.strip())
    if not match or multiple_references:
        # Handle compound expressions like "self.age > 10 and self.age < 20"
        python_expression = _normalize_ocl_expression(expression_body, property_name)
        return {
            'property': property_name,
            'python_expression': python_expression,
            'message': _build_compound_message(
                property_name, expression_body, python_expression
            )
        }

    operator = match.group(1)
    value_str = match.group(2).strip()

    python_operator = _OPERATOR_MAP.get(operator)
    if not python_operator:
        return None

    # Parse the value
    parsed_value = _parse_value(value_str)

    return {
        'property': property_name,
        'operator': operator,
        'python_operator': python_operator,
        'value': parsed_value['value'],
        'value_repr': parsed_value['repr'],
        'value_type': parsed_value['type']
    }


def _model_level_parse(body: str, constraint_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Build a model-level validator for constraints referencing 2+ properties
    of the same class (e.g. ``self.check_in <= self.check_out``).

    Args:
        body: The OCL expression body (already stripped of the context header)
        constraint_name: Name of the constraint, used for the validator name

    Returns:
        A dict with ``model_expression`` / ``message`` / ``validator_name`` or None
    """
    properties = re.findall(r"\bself\.(\w+)\b", body)
    if len(set(properties)) < 2:
        return None

    model_expression = _normalize_model_expression(body)
    readable = _collapse_whitespace_outside_quotes(body)

    return {
        'properties': sorted(set(properties)),
        'model_expression': model_expression,
        'validator_name': _safe_identifier(constraint_name),
        'message': _build_model_message(constraint_name, readable)
    }


def _build_compound_message(property_name: str, body: str, python_expression: str) -> str:
    """
    Build a human-readable error message for a single-property compound constraint.
    """
    pure_match = _PURE_MATCHES.match(body.strip())
    if pure_match:
        raw = pure_match.group('sq')
        if raw is None:
            raw = pure_match.group('dq')
        return f"{property_name} must match '{_unescape_ocl_string(raw)}'"

    if 're.match(' in python_expression:
        readable = _collapse_whitespace_outside_quotes(body)
        return f"{property_name} must satisfy: {readable}"

    message_expression = _replace_outside_quotes(python_expression, r"\bv\b", "")
    message_expression = _collapse_whitespace_outside_quotes(message_expression)
    return f"{property_name} must be {message_expression}"


def _build_model_message(constraint_name: Optional[str], readable: str) -> str:
    """
    Build a human-readable error message for a model-level constraint.
    """
    if constraint_name:
        return f"Constraint '{constraint_name}' violated: {readable}"
    return f"Constraint violated: {readable}"


def _parse_value(value_str: str) -> Dict[str, Any]:
    """
    Parse an OCL value and determine its type.
    """
    value_str = value_str.strip()

    # Check for integer
    try:
        int_val = int(value_str)
        return {'value': int_val, 'repr': str(int_val), 'type': 'int'}
    except ValueError:
        pass

    # Check for float
    try:
        float_val = float(value_str)
        return {'value': float_val, 'repr': str(float_val), 'type': 'float'}
    except ValueError:
        pass

    # Check for string (single quotes in OCL)
    if value_str.startswith("'") and value_str.endswith("'"):
        str_val = _unescape_ocl_string(value_str[1:-1])
        return {'value': str_val, 'repr': repr(str_val), 'type': 'str'}

    # Check for boolean
    if value_str.lower() in ('true', 'false'):
        bool_val = value_str.lower() == 'true'
        return {'value': bool_val, 'repr': str(bool_val), 'type': 'bool'}

    # Default: treat as raw value
    return {'value': value_str, 'repr': value_str, 'type': 'unknown'}


# ---------------------------------------------------------------------------
# Safety net
# ---------------------------------------------------------------------------

def _is_safe_expression(expression: str, allow_self: bool) -> bool:
    """
    Check that a generated expression parses and only uses white-listed constructs.

    ``compile(expression, '<ocl>', 'eval')`` guarantees the emitted code is at
    least syntactically valid; the AST walk additionally rejects leaked OCL
    identifiers, unknown free variables and arbitrary calls.
    """
    try:
        compile(expression, '<ocl>', 'eval')
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                return False
            if node.id == 'self' and not allow_self:
                return False
        elif isinstance(node, ast.Attribute):
            # Only single-level attribute access on a white-listed name.
            if not isinstance(node.value, ast.Name):
                return False
        elif isinstance(node, ast.Call):
            func = node.func
            is_re_match = (
                isinstance(func, ast.Attribute)
                and func.attr == 'match'
                and isinstance(func.value, ast.Name)
                and func.value.id == 're'
            )
            if not is_re_match:
                return False
        elif not isinstance(node, _ALLOWED_NODES):
            return False

    return True


def _finalize_result(
    result: Optional[Dict[str, Any]], constraint_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Validate a parsed constraint and enrich it with template-ready fields.

    Adds ``message_repr`` (a safely quoted Python literal, so the template never
    interpolates raw text into a quoted string) and ``uses_re``. Falls back to
    ``{'skipped': True}`` whenever the expression would not compile or still
    contains OCL syntax.
    """
    if result is None or result.get('skipped'):
        return result

    if 'model_expression' in result:
        expression = result['model_expression']
        allow_self = True
        result.setdefault('validator_name', _safe_identifier(constraint_name))
    elif 'python_expression' in result:
        expression = result['python_expression']
        allow_self = False
    else:
        expression = f"v {result['python_operator']} {result['value_repr']}"
        allow_self = False
        result.setdefault(
            'message',
            f"{result['property']} must be {result['python_operator']} {result['value_repr']}"
        )

    stripped = _strip_string_literals(expression)
    if _RESIDUAL_OCL.search(stripped) or not _is_safe_expression(expression, allow_self):
        return {'skipped': True}

    result['message_repr'] = repr(result['message'])
    result['uses_re'] = 're.match(' in stripped
    return result
