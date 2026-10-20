import re
from typing import Optional, Tuple

from antlr4 import InputStream, CommonTokenStream
from bocl.OCLWrapper import OCLWrapper
from besser.BUML.metamodel.ocl.ocl import LoopExp, PropertyCallExpression, VariableExp
from besser.BUML.metamodel.ocl.predicates import is_comparison, is_self
from besser.BUML.metamodel.ocl.walk import walk
from besser.BUML.metamodel.structural import Class, PrimitiveDataType, Property
from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.error_handling import BOCLErrorListener, BOCLSyntaxError
from besser.BUML.notations.ocl.pretty_printer import pretty_print

# Primitive types whose values the B-OCL evaluator (``bocl`` 1.0.1) cannot
# compare. It splices ``str(value)`` into a Python expression, and
# ``2026-01-01 09:00:00`` / ``09:00:00`` are not valid Python literals. Dates
# become the integer ``int(f"{year}{month}{day}")`` without zero padding, so
# 2026-01-15 (2026115) sorts after 2026-02-01 (202621) and never equals the
# zero-padded ``Date::today()`` (20260115).
_UNEVALUABLE_TEMPORAL_TYPES = {"date", "datetime", "time"}

_NOT_EVALUATED = "not evaluated against the object diagram"


def _parse_only(expression: str) -> None:
    """Run the OCL lexer + parser for syntax validation without evaluating.

    Raises BOCLSyntaxError if the expression is syntactically invalid.
    """
    input_stream = InputStream(expression)
    lexer = BOCLLexer(input_stream)
    lexer.removeErrorListeners()
    error_listener = BOCLErrorListener()
    lexer.addErrorListener(error_listener)

    stream = CommonTokenStream(lexer)
    parser = BOCLParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)

    parser.oclFile()

    if error_listener.has_errors():
        raise BOCLSyntaxError(error_listener.errors)


def _comparison_operand_limitation(operand) -> Optional[str]:
    """Why the evaluator cannot compare ``operand``, or ``None`` if it can."""
    if isinstance(operand, VariableExp):
        return "comparing objects"
    prop = operand.property if isinstance(operand, PropertyCallExpression) else operand
    if not isinstance(prop, Property):
        return None
    if isinstance(prop.type, Class):
        return "comparing objects"
    if isinstance(prop.type, PrimitiveDataType) and prop.type.name in _UNEVALUABLE_TEMPORAL_TYPES:
        return f"comparing {prop.type.name} values"
    return None


def _evaluation_limitation(ast) -> Optional[str]:
    """Return why the B-OCL evaluator cannot evaluate ``ast`` correctly.

    The checker evaluates invariants with ``bocl.OCLWrapper`` (the external
    ``bocl`` package). Its evaluator turns the AST into a Python string for
    ``eval()``; the constructs detected here either crash that ``eval()``
    with a Python ``SyntaxError`` or silently produce a wrong answer, so a
    perfectly valid constraint would be reported as violated (or as
    satisfied when it is not). They all parse fine, so the checker reports
    such a constraint as syntactically valid and not evaluated instead.

    Returns ``None`` when nothing unsupported is found, otherwise one
    ``"; "``-separated sentence naming each kind of unsupported construct
    once, with its first occurrence as the example.
    """
    reasons: dict[str, str] = {}  # kind -> message; first occurrence wins
    for node in walk(ast):
        if isinstance(node, LoopExp):
            if node.name == "closure":
                # bocl 1.0.1 has no closure: it yields True, ``()`` or a
                # Python syntax error whatever the object diagram.
                reasons.setdefault("closure", "the 'closure' iterator")
            names = [it.name for it in node.iterator]
            if len(names) > 1:
                # The evaluator binds every iterator of a loop to the same
                # element, so forAll(a, b | ...) only ever sees a == b.
                reasons.setdefault(
                    "iterators",
                    f"'{node.name}' with several iterator variables ({', '.join(names)})",
                )
            for body in node.body:
                for inner in walk(body):
                    if is_self(inner):
                        # Inside a loop body the evaluator resolves ``self``
                        # to the current element, not to the context object.
                        reasons.setdefault("self", f"'self' inside a '{node.name}' body")
                    elif isinstance(inner, LoopExp) and any(
                        isinstance(n, VariableExp) and n.name in names
                        for b in inner.body for n in walk(b)
                    ):
                        # Same cause as several iterators: the inner body
                        # cannot see the outer element.
                        reasons.setdefault(
                            "nested",
                            f"a nested '{inner.name}' that refers to the outer iterator "
                            f"variable '{names[0]}'",
                        )
        elif is_comparison(node):
            for operand in (node.arguments[0], node.arguments[2]):
                kind = _comparison_operand_limitation(operand)
                if kind:
                    reasons.setdefault(kind, f"{kind} ('{pretty_print(node)}')")
                    break
    if not reasons:
        return None
    return "; ".join(reasons.values()) + " - not supported by the evaluator yet"


def _evaluation_limitation_for(constraint, domain_model) -> Tuple[bool, Optional[str]]:
    """Parse ``constraint`` and run :func:`_evaluation_limitation` on its AST.

    Returns whether it parsed, and the limitation. A constraint that cannot be
    parsed here is left to the evaluator, which reports the parse error itself.
    """
    context_class = getattr(constraint, "context", None)
    try:
        parsed = parse_ocl(
            constraint.expression, domain_model,
            context_class=context_class if isinstance(context_class, Class) else None,
        )
    except Exception:  # noqa: BLE001 - any failure: let the evaluator report it
        return False, None
    return True, _evaluation_limitation(parsed.ast)


def extract_context_class_name(expression):
    """Extract the context class name from an OCL expression"""
    if not isinstance(expression, str):
        return ""
    # Look for pattern: context ClassName
    match = re.search(r'context\s+(\w+)', expression, re.IGNORECASE)
    return match.group(1) if match else ""

def _ensure_canonical_expression(constraint, kind: str, method=None) -> None:
    """In-place upgrade of legacy body-only ``constraint.expression``.

    Constraints loaded from BUML files generated by the *previous*
    ``domain_model_builder`` — before this PR taught it to emit full
    canonical text and ``Method.add_pre`` / ``add_post`` calls — store
    expressions as body-only (``"self.balance >= 0"``). The validator's
    BOCL-based syntax check and the BOCL evaluator both require the
    full ``context X (inv|pre|post) ...:`` form, so we lift body-only
    expressions to canonical here once, mutating ``constraint.expression``
    so every downstream consumer (display, parse, evaluate) sees the
    same full-text shape produced by the JSON-to-BUML pipeline.

    Pre/post need a ``method`` to reconstruct the ``::method(params)``
    segment; if none is provided we fall back to an invariant-shaped
    header rather than emitting a half-broken precondition.
    """
    expr = (constraint.expression or "").lstrip()
    if expr.lower().startswith("context"):
        return
    ctx = getattr(constraint, "context", None)
    ctx_name = getattr(ctx, "name", None) or "?"
    body = constraint.expression
    if kind == "invariant" or method is None:
        constraint.expression = f"context {ctx_name} inv {constraint.name}: {body}"
        return
    params = ", ".join(
        f"{p.name}: {(getattr(getattr(p, 'type', None), 'name', None) or 'any')}"
        for p in (getattr(method, "parameters", None) or [])
    )
    kind_kw = "pre" if kind == "precondition" else "post"
    constraint.expression = (
        f"context {ctx_name}::{method.name}({params}) {kind_kw}: {body}"
    )


def _collect_all_constraints(domain_model):
    """Yield ``(label, kind, constraint)`` for every OCL constraint in the
    model.

    Walks ``domain_model.constraints`` (invariants) + every class's methods'
    ``.pre`` / ``.post`` lists (method contracts). Labels include the
    class / method / kind / name so error messages point at a specific
    constraint box in the editor.

    The displayed expression is ``constraint.expression`` directly: every
    constraint built through the JSON-to-BUML pipeline carries the full
    canonical OCL text on its expression field (set in
    ``ocl_parser.parse_constraint_text``), so no reconstruction is needed
    here.

    ``kind`` is one of ``"invariant"`` / ``"precondition"`` /
    ``"postcondition"`` and is used by the caller to skip evaluation of
    pre/post against an object model — those are runtime contracts that
    bind to method parameters at *invocation* time, which the standalone
    validator has no way to do.
    """
    for c in domain_model.constraints:
        _ensure_canonical_expression(c, "invariant")
        ctx_name = c.context.name if c.context is not None else "?"
        yield (f"[{ctx_name} inv {c.name}]", "invariant", c)
    for cls in domain_model.types:
        if not isinstance(cls, Class):
            continue
        for method in cls.methods:
            for c in method.pre:
                _ensure_canonical_expression(c, "precondition", method)
                yield (f"[{cls.name}::{method.name} pre {c.name}]", "precondition", c)
            for c in method.post:
                _ensure_canonical_expression(c, "postcondition", method)
                yield (f"[{cls.name}::{method.name} post {c.name}]", "postcondition", c)


def check_ocl_constraint(domain_model, object_model = None):
    """
    Check all OCL constraints in the domain model — class invariants plus
    method preconditions and postconditions.

    Returns a dict with ``success``, ``message``, ``valid_constraints``,
    ``invalid_constraints`` and ``warnings``. ``warnings`` lists invariants
    that are syntactically valid but could not be evaluated against
    ``object_model`` (see :func:`_evaluation_limitation`); they also appear
    in ``valid_constraints``, since no violation was found.
    """
    try:
        constraints = list(_collect_all_constraints(domain_model))
        if not constraints:
            return {
                "success": True,
                "message": "",
                "valid_constraints": [],
                "invalid_constraints": [],
                "warnings": [],
            }

        valid_constraints = []
        invalid_constraints = []
        # Constraints that are syntactically valid but that the evaluator
        # cannot check against the object diagram (see
        # ``_evaluation_limitation``); surfaced as diagram warnings.
        warnings = []
        parser = OCLWrapper(domain_model, object_model)

        for label, kind, constraint in constraints:
            expression = constraint.expression
            description = getattr(constraint, "description", None)
            # Suffix appended to every message so non-technical users see the
            # plain-language reason alongside the raw OCL expression.
            explanation_suffix = f" — {description}" if description else ""
            # Pre/post are runtime contracts — evaluating them requires
            # binding method parameters to actual call arguments, which a
            # standalone validator has no way to do (the BOCL evaluator
            # would raise ``NameError: name 'qty' is not defined``). Limit
            # them to a syntax check regardless of whether an object model
            # is available.
            evaluable = (kind == "invariant") and object_model is not None
            try:
                if not evaluable:
                    # Syntax-only check — re-parse the stored expression.
                    # ``constraint.expression`` is the full canonical OCL
                    # text (set in ``parse_constraint_text``), so we can
                    # feed it directly to the BOCL parser.
                    try:
                        _parse_only(expression)
                        valid_constraints.append(
                            f"✅ {label} '{expression}'{explanation_suffix}"
                        )
                    except BOCLSyntaxError as syntax_err:
                        invalid_constraints.append(
                            f"❌ {label} '{expression}' - {syntax_err}{explanation_suffix}"
                        )
                else:
                    # Evaluate the invariant against the object model. Skip
                    # if there are no instances of the context class to
                    # evaluate against.
                    context_class_name = (
                        constraint.context.name
                        if getattr(constraint, "context", None) is not None
                        else extract_context_class_name(expression)
                    )
                    context_instances = [
                        obj for obj in object_model.objects
                        if hasattr(obj, 'classifier') and obj.classifier.name.lower() == context_class_name.lower()
                    ]

                    if not context_instances:
                        continue

                    # Skip constraints whose referenced attributes are unset on
                    # the instance(s). In user-profile ("user model") diagrams a
                    # blank criterion (e.g. an empty ``age``) is intentionally
                    # not materialised as a slot, so ``self.age`` has no value to
                    # evaluate. Evaluating anyway builds an expression like
                    # "  >= 0 and  <= 120" and blows up with a Python
                    # ``invalid syntax`` error. An unset attribute means "no
                    # constraint to check", so we skip instead.
                    # Only attributes count: an association end (``self.books``)
                    # has no slot, and an object without links is a valid value.
                    referenced_attrs = set(re.findall(r'self\.(\w+)', expression))
                    context_class = getattr(constraint, "context", None)
                    if isinstance(context_class, Class):
                        referenced_attrs &= {a.name for a in context_class.all_attributes()}
                    if referenced_attrs:
                        def _has_all_referenced(obj):
                            slot_names = {
                                s.attribute.name for s in getattr(obj, 'slots', [])
                            }
                            return referenced_attrs.issubset(slot_names)

                        if not all(_has_all_referenced(obj) for obj in context_instances):
                            continue

                    parsed, limitation = _evaluation_limitation_for(constraint, domain_model)
                    if limitation is None:
                        try:
                            result = parser.evaluate(constraint)
                        except BOCLSyntaxError:
                            raise
                        except Exception as eval_error:  # noqa: BLE001
                            if not parsed:
                                raise
                            # Valid B-OCL that the evaluator cannot handle yet:
                            # its crash is not the user's error.
                            result = None
                            limitation = f"the evaluator does not support it yet ({eval_error})"
                        if limitation is None and not isinstance(result, bool):
                            # An invariant is Boolean; anything else (e.g. the
                            # ``()`` the evaluator yields for a loop whose source
                            # it could not resolve) means it failed silently.
                            limitation = (
                                f"the evaluator returned a non-Boolean value ({result!r})"
                            )
                    if limitation is not None:
                        valid_constraints.append(
                            f"✅ {label} '{expression}' - Syntax valid; {_NOT_EVALUATED}: "
                            f"{limitation}{explanation_suffix}"
                        )
                        warnings.append(f"OCL {label} {_NOT_EVALUATED}: {limitation}")
                    elif result is True:
                        valid_constraints.append(
                            f"✅ {label} '{expression}' - Evaluates to: True{explanation_suffix}"
                        )
                    else:
                        # Prefer the natural-language description as the primary
                        # violation reason when one is provided.
                        violation_reason = description if description else "Evaluates to False"
                        invalid_constraints.append(
                            f"❌ {label} '{expression}' - Constraint violation: {violation_reason}"
                        )
            except Exception as e:
                invalid_constraints.append(
                    f"❌ {label} '{expression}' - Error: {str(e)}{explanation_suffix} \n"
                )

        return {
            "success": len(invalid_constraints) == 0,
            "valid_constraints": valid_constraints,
            "invalid_constraints": invalid_constraints,
            "warnings": warnings,
            "message": f"Found {len(valid_constraints)} valid and {len(invalid_constraints)} invalid constraints"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"{str(e)}",
            "valid_constraints": [],
            "invalid_constraints": [],
            "warnings": [],
        }
