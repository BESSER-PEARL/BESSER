"""Pretty-print B-OCL ASTs back to OCL source text.

Renders an OCL expression — or a full :class:`OCLConstraint` — to a
canonical string using the precedence hierarchy locked in by the post-grammar-
fix B-OCL grammar:

    Postfix > Unary > Mul > Add > Comp > AND > XOR > OR > IMPLIES

Parentheses are added only where strictly required by the precedence relation
between a node and its parent.
"""

from besser.BUML.metamodel.ocl.ocl import (
    OCLConstraint, OperationCallExpression, LoopExp, IfExp,
    PropertyCallExpression, VariableExp, TypeExp,
    IntegerLiteralExpression, RealLiteralExpression,
    BooleanLiteralExpression, StringLiteralExpression, DateLiteralExpression,
    InfixOperator,
)


# Precedence — higher binds tighter.
_PREC_TOP = 0
_PREC_IMPLIES = 10
_PREC_OR = 20
_PREC_XOR = 30
_PREC_AND = 40
_PREC_COMP = 50
_PREC_ADD = 60
_PREC_MUL = 70
_PREC_UNARY = 80
_PREC_POSTFIX = 90

_OP_PREC = {
    "implies": _PREC_IMPLIES,
    "or": _PREC_OR,
    "xor": _PREC_XOR,
    "and": _PREC_AND,
    "<": _PREC_COMP, ">": _PREC_COMP, "<=": _PREC_COMP, ">=": _PREC_COMP,
    "=": _PREC_COMP, "<>": _PREC_COMP,
    "+": _PREC_ADD, "-": _PREC_ADD,
    "*": _PREC_MUL, "/": _PREC_MUL,
}

_TYPE_TEST_NAMES = {
    "OCLISTYPEOF": "oclIsTypeOf",
    "OCLISKINDOF": "oclIsKindOf",
    "OCLASTYPE": "oclAsType",
}


def pretty_print(node) -> str:
    """Render an OCL AST or :class:`OCLConstraint` to OCL source text.

    For an :class:`OCLConstraint`, emits ``context <ClassName> inv: <expr>``.
    For a raw expression, emits the expression alone.
    """
    if isinstance(node, OCLConstraint):
        return f"context {node.context.name} inv: {_render(node.expression, _PREC_TOP)}"
    return _render(node, _PREC_TOP)


def _paren(text: str, child_prec: int, parent_prec: int) -> str:
    return f"({text})" if child_prec < parent_prec else text


def _render(node, parent_prec: int) -> str:
    if node is None:
        return ""
    if isinstance(node, OperationCallExpression):
        return _render_operation(node, parent_prec)
    if isinstance(node, LoopExp):
        return _render_loop(node)
    if isinstance(node, IfExp):
        return _render_if(node)
    if isinstance(node, PropertyCallExpression):
        return _render_property_call(node)
    if isinstance(node, VariableExp):
        return node.name
    if isinstance(node, TypeExp):
        return node.name
    if isinstance(node, BooleanLiteralExpression):
        return _render_boolean(node.value)
    if isinstance(node, IntegerLiteralExpression):
        return str(node.value)
    if isinstance(node, RealLiteralExpression):
        return str(node.value)
    if isinstance(node, StringLiteralExpression):
        return f"'{node.value}'"
    if isinstance(node, DateLiteralExpression):
        return str(node.value)
    # Fallback for bare structural-model objects (Property leaked from an
    # un-wrapped visitor) and anything else with a ``name``.
    if hasattr(node, "name"):
        return str(node.name)
    return repr(node)


def _render_boolean(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _render_property_call(node: PropertyCallExpression) -> str:
    if node.source is None:
        return node.property.name
    return f"{_render(node.source, _PREC_POSTFIX)}.{node.property.name}"


def _render_operation(node: OperationCallExpression, parent_prec: int) -> str:
    op = node.operation
    args = node.arguments

    if op == "not":
        operand = _render(args[0], _PREC_UNARY)
        return _paren(f"not {operand}", _PREC_UNARY, parent_prec)

    if op == "ALLInstances":
        # source is a TypeExp; ALLInstances is a type-method call.
        return f"{_render(node.source, _PREC_POSTFIX)}.allInstances()"

    if op == "Size":
        return f"{_render(node.source, _PREC_POSTFIX)}->size()"

    if op == "IsEmpty":
        return f"{_render(node.source, _PREC_POSTFIX)}->isEmpty()"

    if op in _TYPE_TEST_NAMES:
        method = _TYPE_TEST_NAMES[op]
        type_arg = args[0]
        return (f"{_render(node.source, _PREC_POSTFIX)}."
                f"{method}({_render(type_arg, _PREC_TOP)})")

    # Comparison or arithmetic (3-argument shape with InfixOperator marker).
    if len(args) == 3 and isinstance(args[1], InfixOperator):
        prec = _OP_PREC.get(op, _PREC_COMP)
        left = _render(args[0], prec)
        right = _render(args[2], prec + 1)
        return _paren(f"{left} {op} {right}", prec, parent_prec)

    # Boolean binary (2-argument shape).
    if op in ("and", "or", "xor", "implies") and len(args) == 2:
        prec = _OP_PREC[op]
        left = _render(args[0], prec)
        right = _render(args[1], prec + 1)
        return _paren(f"{left} {op} {right}", prec, parent_prec)

    # Generic method-call fallback (e.g. user-defined methods).
    rendered_args = ", ".join(
        _render(a, _PREC_TOP) for a in args if not isinstance(a, InfixOperator)
    )
    if node.source is not None:
        return f"{_render(node.source, _PREC_POSTFIX)}.{op}({rendered_args})"
    return f"{op}({rendered_args})"


def _render_loop(node: LoopExp) -> str:
    src = _render(node.source, _PREC_POSTFIX) if node.source is not None else ""
    iters = ", ".join(it.name for it in node.iterator)
    body = _render(node.body[0], _PREC_TOP) if node.body else ""
    if iters:
        return f"{src}->{node.name}({iters} | {body})"
    return f"{src}->{node.name}({body})"


def _render_if(node: IfExp) -> str:
    cond = _render(node.ifCondition, _PREC_TOP)
    then_e = _render(node.thenExpression, _PREC_TOP)
    else_e = _render(node.elseCondition, _PREC_TOP)
    return f"if {cond} then {then_e} else {else_e} endif"
