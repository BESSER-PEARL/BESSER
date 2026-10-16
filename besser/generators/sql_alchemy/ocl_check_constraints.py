"""Translate simple OCL invariants into SQL ``CHECK`` constraints.

Only a deliberately small, safe subset of OCL is translated: an invariant whose
context is a class with its own table and whose body is a comparison between
``self.<attribute>`` and a literal, or an ``and`` of such comparisons::

    context Book inv: self.pages > 0
    context Book inv: self.pages >= 1 and self.pages <= 1000
    context Book inv: self.title <> ''
    context Book inv: 0 < self.price

Anything else (``or``, ``not``, navigation, collection operations, enum literals,
attribute-to-attribute comparisons, inherited or derived attributes, ...) is
skipped silently, with a debug log entry: a constraint that cannot be enforced
by the database is still enforced by the application-level validators.

The result is rendered by the SQLAlchemy template as
``CheckConstraint(_sa_column(<name>, <type>) <op> <literal>, name=...)`` so the
column quoting and the literal rendering are left to each SQL dialect.
"""

import ast
import hashlib
import logging
import math
import re
from typing import Dict, List, Optional

from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression,
    IntegerLiteralExpression,
    OCLConstraint,
    OperationCallExpression,
    PropertyCallExpression,
    RealLiteralExpression,
    StringLiteralExpression,
    VariableExp,
)
from besser.BUML.metamodel.structural import Class, DomainModel, Property

logger = logging.getLogger(__name__)

#: OCL comparison operator -> Python operator used on the SQLAlchemy column.
_COMPARISON_OPERATORS = {
    "=": "==",
    "<>": "!=",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
}

#: Operator to use when the literal is on the left (``0 < self.x`` -> ``self.x > 0``).
_FLIPPED = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}

_NUMERIC_TYPES = {"int", "float"}

#: Attribute type -> literal classes it may be compared with, and the allowed operators.
_ALLOWED_BY_TYPE = {
    "int": ((IntegerLiteralExpression, RealLiteralExpression), set(_FLIPPED)),
    "float": ((IntegerLiteralExpression, RealLiteralExpression), set(_FLIPPED)),
    # String ordering depends on the database collation: equality only.
    "str": ((StringLiteralExpression,), {"==", "!="}),
    "bool": ((BooleanLiteralExpression,), {"==", "!="}),
}

_INVARIANT_HEADER = re.compile(r"^\s*context\s+(\w+)\s+inv\b")

#: Longest constraint name emitted: the lowest identifier limit of the supported
#: dialects (Oracle before 12.2).
_MAX_NAME_LENGTH = 30


class _NotTranslatable(Exception):
    """Raised internally when an expression falls outside the supported subset."""


def get_check_constraints(model: DomainModel, types: Dict[str, str],
                          table_classes: List[Class]) -> Dict[str, List[Dict[str, str]]]:
    """Collect the CHECK constraints derivable from the model's OCL invariants.

    Args:
        model: The domain model whose ``constraints`` are inspected.
        types: Attribute type name -> SQLAlchemy type expression, as used by the
            generated module (e.g. ``"int" -> "Integer_"``).
        table_classes: Classes that are mapped to their own table.

    Returns:
        Class name -> list of ``{"name": <constraint name>, "expression": <python>}``,
        ordered by constraint name, where ``expression`` is a SQLAlchemy column
        expression in the generated module's namespace. Classes without any
        translatable constraint are absent.
    """
    table_class_names = {cls.name for cls in table_classes}
    result: Dict[str, List[Dict[str, str]]] = {}
    used_names = set()
    for constraint in sorted(model.constraints, key=lambda c: (c.context.name, c.name)):
        context = constraint.context
        if context.name not in table_class_names:
            logger.debug("OCL constraint %r skipped: class %r has no table of its own",
                         constraint.name, context.name)
            continue
        expression = translate_constraint(constraint, model, types)
        if expression is None:
            continue
        name = _constraint_name(context.name, constraint.name, used_names)
        result.setdefault(context.name, []).append({"name": name, "expression": expression})
    return result


def translate_constraint(constraint, model: DomainModel, types: Dict[str, str]) -> Optional[str]:
    """Translate one OCL invariant into a SQLAlchemy column expression (as Python source).

    Returns ``None`` when the constraint is not in the supported subset.
    """
    ocl_ast = _parse(constraint, model)
    if ocl_ast is None:
        return None
    try:
        comparisons = _flatten_and(ocl_ast)
        parts = [_translate_comparison(comp, constraint.context, types) for comp in comparisons]
    except _NotTranslatable as exc:
        logger.debug("OCL constraint %r skipped: %s", constraint.name, exc)
        return None
    expression = parts[0] if len(parts) == 1 else f"_sa_and({', '.join(parts)})"
    # Defense in depth: the expression is written into a module that is executed.
    try:
        ast.parse(expression, mode="eval")
    except SyntaxError:  # pragma: no cover - every part is built from literals and repr()
        logger.debug("OCL constraint %r skipped: produced invalid Python", constraint.name)
        return None
    return expression


def _parse(constraint, model: DomainModel):
    """Return the OCL AST of an invariant, or ``None`` if it is not an OCL invariant."""
    if isinstance(constraint, OCLConstraint):
        return constraint.ast
    if (constraint.language or "").strip().upper() != "OCL":
        logger.debug("OCL constraint %r skipped: language %r", constraint.name, constraint.language)
        return None
    text = constraint.expression or ""
    header = _INVARIANT_HEADER.match(text)
    if header is None or header.group(1) != constraint.context.name or text.count("context") != 1:
        logger.debug("OCL constraint %r skipped: not a single invariant on its context class",
                     constraint.name)
        return None
    # Deferred import: the ANTLR parser is only needed when there are constraints.
    from besser.BUML.notations.ocl.api import parse_ocl  # pylint: disable=import-outside-toplevel
    try:
        return parse_ocl(text, model, context_class=constraint.context).ast
    except Exception as exc:  # pylint: disable=broad-except
        # Translation is best effort: a constraint the parser rejects is skipped.
        logger.debug("OCL constraint %r skipped: could not be parsed (%s)", constraint.name, exc)
        return None


def _flatten_and(expression) -> list:
    """Split nested ``a and b and c`` into its operands."""
    if isinstance(expression, OperationCallExpression) and _operator(expression) == "and":
        if len(expression.arguments) != 2:
            raise _NotTranslatable("unexpected 'and' shape")
        left, right = expression.arguments
        return _flatten_and(left) + _flatten_and(right)
    return [expression]


def _operator(expression: OperationCallExpression) -> str:
    return (expression.operation or "").strip().lower()


def _translate_comparison(expression, context: Class, types: Dict[str, str]) -> str:
    if not isinstance(expression, OperationCallExpression):
        raise _NotTranslatable("not a comparison")
    operator = _COMPARISON_OPERATORS.get(_operator(expression))
    if operator is None:
        raise _NotTranslatable(f"unsupported operation {expression.operation!r}")
    # Binary infix operations are stored as [left, InfixOperator, right].
    if len(expression.arguments) != 3:
        raise _NotTranslatable("unexpected comparison shape")
    left, _, right = expression.arguments
    if _is_self_attribute(left):
        prop_call, literal = left, right
    elif _is_self_attribute(right):
        prop_call, literal = right, left
        operator = _FLIPPED[operator]
    else:
        raise _NotTranslatable("no 'self.<attribute>' operand")

    attribute = _own_attribute(context, prop_call.property)
    type_name = attribute.type.name
    allowed = _ALLOWED_BY_TYPE.get(type_name)
    if allowed is None or type_name not in types:
        raise _NotTranslatable(f"attribute type {type_name!r} not supported")
    literal_classes, operators = allowed
    if not isinstance(literal, literal_classes):
        raise _NotTranslatable(f"operand is not a literal compatible with {type_name!r}")
    if operator not in operators:
        raise _NotTranslatable(f"operator not supported for {type_name!r}")

    value = _literal_value(literal, type_name)
    return f"_sa_column({attribute.name!r}, {types[type_name]}) {operator} {value!r}"


def _is_self_attribute(expression) -> bool:
    if not isinstance(expression, PropertyCallExpression):
        return False
    source = getattr(expression, "source", None)
    return isinstance(source, VariableExp) and source.name == "self"


def _own_attribute(context: Class, prop) -> Property:
    """The context class's own, stored, single-valued attribute named like ``prop``."""
    name = getattr(prop, "name", None)
    attribute = next((attr for attr in context.attributes if attr.name == name), None)
    if attribute is None:
        raise _NotTranslatable(f"{name!r} is not an attribute of {context.name!r} itself")
    if attribute.is_derived:
        raise _NotTranslatable(f"{name!r} is derived (computed by the application)")
    if attribute.multiplicity.max > 1:
        raise _NotTranslatable(f"{name!r} is multi-valued")
    return attribute


def _literal_value(literal, type_name: str):
    value = literal.value
    if type_name in _NUMERIC_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                value = float(value) if isinstance(literal, RealLiteralExpression) else int(value)
            except (TypeError, ValueError) as exc:
                raise _NotTranslatable("numeric literal could not be read") from exc
        if not math.isfinite(value):
            raise _NotTranslatable("non-finite numeric literal")
        return value
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise _NotTranslatable("boolean literal could not be read")
    if not isinstance(value, str):
        raise _NotTranslatable("string literal could not be read")
    if "\\" in value:
        # The parser keeps OCL escapes raw; their meaning is ambiguous in SQL.
        raise _NotTranslatable("string literal with escape sequences")
    return value


def _constraint_name(class_name: str, constraint_name: str, used: set) -> str:
    """Deterministic, dialect-safe constraint name, unique within the model."""
    base = re.sub(r"[^0-9a-zA-Z_]", "_", f"ck_{class_name}_{constraint_name}").lower()
    if len(base) > _MAX_NAME_LENGTH:
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
        base = f"{base[:_MAX_NAME_LENGTH - 7]}_{digest}"
    name, index = base, 1
    while name in used:
        index += 1
        suffix = f"_{index}"
        name = f"{base[:_MAX_NAME_LENGTH - len(suffix)]}{suffix}"
    used.add(name)
    return name
