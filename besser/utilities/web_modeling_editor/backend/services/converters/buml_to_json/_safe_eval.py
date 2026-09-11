"""Shared safe AST evaluator for the BUML -> JSON importers.

The per-diagram converters in this package reconstruct a BUML model object from
an *uploaded* ``.py`` source string, then walk that object into the editor's
JSON. Historically the reconstruction step used ``exec()`` inside a namespace
with a trimmed ``__builtins__`` dict. That sandbox is escapable — the classic
``().__class__.__bases__[0].__subclasses__()`` attribute-traversal trick needs
no builtins at all — so ``exec()`` on untrusted input is a remote-code-execution
risk regardless of how the globals are restricted.

``safe_exec`` replaces that ``exec()`` with a whitelist AST interpreter. It
``ast.parse``s the code and evaluates only a narrow, explicitly-enumerated set
of node types, seeded from the caller's ``allowed`` mapping (the same dict of
whitelisted BUML constructors each converter already builds). The resulting
namespace is returned so the existing JSON walkers and ``_find_*_model`` lookups
keep working unchanged.

Security boundary
-----------------
ALLOWED:
  * literals: str / int / float / bool / None and list / tuple / set / dict
    literals of allowed expressions;
  * ``Name`` lookups resolving to a prior assignment or to a whitelisted name;
  * calls to whitelisted callables by name (the BUML constructors + a handful
    of benign builtins the caller seeds, e.g. ``set``/``list``);
  * method calls on values already produced in the namespace — the fluent
    builder chains (``bpmn_model.add_process(...)``, ``qc.add_operation(...)``,
    ``gate.control_qubits.append(...)``);
  * attribute *reads* for enum-literal access (``TaskType.SEND``,
    ``ControlState.CONTROL``) and data attributes of namespace objects;
  * simple name / attribute assignment (``x = ...``, ``x.layout = {...}``),
    annotated assignment (``x: T = ...``);
  * ``if``/``try`` control flow needed to reproduce import semantics
    (``if __name__ == "__main__":`` guards, the builder's
    ``try: m.state_machine = sm  except NameError: pass`` optional wiring).

BLOCKED (raises ``UnsafeConstruct``, which no in-code ``try/except`` can swallow):
  * any attribute whose name starts with ``__`` — read, write, or call. This is
    the single control that defeats every known sandbox escape, all of which
    route through ``__class__`` / ``__globals__`` / ``__subclasses__`` / ``__bases__``
    / ``__mro__`` / ``__builtins__`` / ``__import__`` etc.;
  * calls whose target is neither a whitelisted name nor a method on a namespace
    value (so ``__import__(...)``, ``getattr(...)``, ``eval(...)`` all fail —
    those names are not in the whitelist);
  * method calls on ``str`` / ``bytes`` receivers (blocks ``"{0.__class__}".format``
    style attribute-leak gadgets that hide the dunder inside a string literal);
  * ``import`` (silently skipped — the names are already whitelisted), ``lambda``,
    comprehensions, generator expressions, f-strings, subscripting, ``*``/``**``
    unpacking, walrus, binary operators, and every statement type not listed
    above.
"""

import ast
from typing import Any, Dict, List


class UnsafeConstruct(ValueError):
    """Raised when the uploaded code contains a construct the evaluator refuses
    to run.

    Subclasses ``ValueError`` so the converters' existing ``except (ValueError,
    ...)`` wrappers map it to a 400-style failure. It is deliberately *never*
    caught by the evaluator's own ``try``/``except`` handling — a security
    refusal aborts the whole parse rather than being swallowed by an attacker's
    ``try: <payload> except Exception: pass``.
    """


# Builtin exceptions an uploaded file's ``try``/``except`` may legitimately
# catch. The domain-model builder emits, for methods wired to a state machine
# or quantum circuit that only exist in a *combined* project file:
#
#     try:
#         SomeClass_m_method.state_machine = sm
#     except NameError:
#         pass
#
# so that a class-diagram-only import (where ``sm`` is undefined) skips the
# wiring instead of crashing. We reproduce that control flow but keep the set
# of catchable exception types tiny and explicit.
_CATCHABLE_EXCEPTIONS: Dict[str, type] = {
    "NameError": NameError,
    "AttributeError": AttributeError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "Exception": Exception,
}

_COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _guard_attr(name: str) -> None:
    """Reject any dunder attribute access (read / write / method call)."""
    if name.startswith("__"):
        raise UnsafeConstruct(
            f"Access to dunder attribute {name!r} is not allowed"
        )


class _SafeEvaluator:
    """Whitelist AST interpreter. One instance per ``safe_exec`` call."""

    def __init__(self, resolvable: Dict[str, Any]):
        # ``resolvable`` is the flattened whitelist: seeded builtins + the
        # caller's constructor names. Name lookups fall back to it after the
        # per-run assignment namespace ``env``.
        self._resolvable = resolvable
        self.env: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Statement execution
    # ------------------------------------------------------------------ #
    def run(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"Failed to parse BUML content: {exc}") from exc
        self._run_body(tree.body)
        return self.env

    def _run_body(self, body: List[ast.stmt]) -> None:
        for stmt in body:
            self._run_stmt(stmt)

    def _run_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            # All metamodel names are already whitelisted; imports are noise.
            return
        if isinstance(stmt, ast.Assign):
            value = self._eval(stmt.value)
            for target in stmt.targets:
                self._assign(target, value)
            return
        if isinstance(stmt, ast.AnnAssign):
            # ``x: T = value`` — the annotation ``T`` is a type hint only, never
            # evaluated. Bare ``x: T`` (value is None) binds nothing.
            if stmt.value is None:
                return
            value = self._eval(stmt.value)
            self._assign(stmt.target, value)
            return
        if isinstance(stmt, ast.Expr):
            # A bare expression statement. Constants (docstrings / stray
            # literals) are ignored; everything else is evaluated for its side
            # effect (the fluent ``obj.add_*(...)`` builder calls).
            if isinstance(stmt.value, ast.Constant):
                return
            self._eval(stmt.value)
            return
        if isinstance(stmt, ast.If):
            if self._eval(stmt.test):
                self._run_body(stmt.body)
            else:
                self._run_body(stmt.orelse)
            return
        if isinstance(stmt, ast.Try):
            self._run_try(stmt)
            return
        if isinstance(stmt, ast.Pass):
            return
        raise UnsafeConstruct(
            f"Disallowed statement: {type(stmt).__name__}"
        )

    def _assign(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            return
        if isinstance(target, ast.Attribute):
            _guard_attr(target.attr)
            obj = self._eval(target.value)
            if isinstance(obj, (str, bytes, bytearray)):
                raise UnsafeConstruct(
                    "Attribute assignment on str/bytes values is not allowed"
                )
            setattr(obj, target.attr, value)
            return
        raise UnsafeConstruct(
            f"Only simple name / attribute assignment targets are allowed; "
            f"got {type(target).__name__}"
        )

    def _run_try(self, stmt: ast.Try) -> None:
        try:
            self._run_body(stmt.body)
        except UnsafeConstruct:
            # A security refusal is never catchable by uploaded code.
            raise
        except Exception as exc:  # noqa: BLE001 - reproducing Python semantics
            for handler in stmt.handlers:
                if self._handler_matches(handler, exc):
                    if handler.name:
                        self.env[handler.name] = exc
                    self._run_body(handler.body)
                    break
            else:
                raise
        else:
            self._run_body(stmt.orelse)
        finally:
            self._run_body(stmt.finalbody)

    def _handler_matches(self, handler: ast.ExceptHandler, exc: Exception) -> bool:
        if handler.type is None:
            return True  # bare ``except:`` (UnsafeConstruct already re-raised)
        types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        matched: List[type] = []
        for node in types:
            if not isinstance(node, ast.Name) or node.id not in _CATCHABLE_EXCEPTIONS:
                raise UnsafeConstruct(
                    "Only a small set of builtin exceptions "
                    f"({', '.join(sorted(_CATCHABLE_EXCEPTIONS))}) may be caught"
                )
            matched.append(_CATCHABLE_EXCEPTIONS[node.id])
        return isinstance(exc, tuple(matched))

    # ------------------------------------------------------------------ #
    # Expression evaluation
    # ------------------------------------------------------------------ #
    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id in self._resolvable:
                return self._resolvable[node.id]
            # Faithful to Python import semantics: an undefined name raises
            # NameError (catchable by an ``except NameError`` in the source),
            # NOT UnsafeConstruct.
            raise NameError(f"name {node.id!r} is not defined")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e) for e in self._elts(node.elts)]
        if isinstance(node, ast.Set):
            return {self._eval(e) for e in self._elts(node.elts)}
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if key is None:
                    raise UnsafeConstruct("Dict unpacking (**expr) is not allowed")
            return {self._eval(k): self._eval(v)
                    for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise UnsafeConstruct(
                f"Disallowed unary operator: {type(node.op).__name__}"
            )
        if isinstance(node, ast.Attribute):
            _guard_attr(node.attr)
            return getattr(self._eval(node.value), node.attr)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise UnsafeConstruct(
            f"Disallowed expression: {type(node).__name__}"
        )

    @staticmethod
    def _elts(elts: List[ast.expr]) -> List[ast.expr]:
        for e in elts:
            if isinstance(e, ast.Starred):
                raise UnsafeConstruct("Iterable unpacking (*expr) is not allowed")
        return elts

    def _eval_compare(self, node: ast.Compare) -> Any:
        left = self._eval(node.left)
        for op, right_node in zip(node.ops, node.comparators):
            impl = _COMPARE_OPS.get(type(op))
            if impl is None:
                raise UnsafeConstruct(
                    f"Disallowed comparison operator: {type(op).__name__}"
                )
            right = self._eval(right_node)
            if not impl(left, right):
                return False
            left = right
        return True

    def _eval_boolop(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = self._eval(value)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = self._eval(value)
                if result:
                    return result
            return result
        raise UnsafeConstruct(
            f"Disallowed boolean operator: {type(node.op).__name__}"
        )

    def _eval_call(self, node: ast.Call) -> Any:
        args = [self._eval(a) for a in self._elts(node.args)]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise UnsafeConstruct("Keyword unpacking (**kwargs) is not allowed")
            kwargs[kw.arg] = self._eval(kw.value)

        func = node.func
        if isinstance(func, ast.Name):
            # Only whitelisted callables may be invoked by bare name. A name
            # bound only in ``env`` (a reconstructed model object) is data, not
            # a constructor, and must not be called.
            if func.id not in self._resolvable:
                raise UnsafeConstruct(
                    f"Call to non-whitelisted name {func.id!r}"
                )
            target = self._resolvable[func.id]
            if not callable(target):
                raise UnsafeConstruct(
                    f"Whitelisted name {func.id!r} is not callable"
                )
            return target(*args, **kwargs)

        if isinstance(func, ast.Attribute):
            # Method call on a namespace value — the fluent builder chains.
            _guard_attr(func.attr)
            receiver = self._eval(func.value)
            if isinstance(receiver, (str, bytes, bytearray)):
                raise UnsafeConstruct(
                    "Method calls on str/bytes values are not allowed"
                )
            method = getattr(receiver, func.attr)
            return method(*args, **kwargs)

        raise UnsafeConstruct(
            "Only calls to whitelisted names or methods on namespace values "
            "are allowed"
        )


def safe_exec(code: str, allowed: Dict[str, Any]) -> Dict[str, Any]:
    """Safely evaluate ``code`` and return its resulting top-level namespace.

    Drop-in replacement for ``exec(code, allowed, local_vars)`` in the
    BUML -> JSON converters: pass the same ``safe_globals`` dict each converter
    already builds, and use the returned dict exactly like the old
    ``local_vars``.

    ``allowed`` may carry the ``exec``-style special keys ``__builtins__`` (a
    dict of benign builtins) and ``__name__`` (a sentinel string): the builtins
    are flattened into the resolvable namespace so bare ``set(...)`` / ``list(...)``
    calls keep working, and ``__name__`` stays resolvable so
    ``if __name__ == "__main__":`` guards evaluate to False (the guarded block is
    skipped, matching import semantics).

    Raises ``UnsafeConstruct`` (a ``ValueError``) on any disallowed construct,
    or ``NameError`` for an undefined name — never executes it.
    """
    resolvable: Dict[str, Any] = {}
    builtins = allowed.get("__builtins__")
    if isinstance(builtins, dict):
        resolvable.update(builtins)
    for key, value in allowed.items():
        if key == "__builtins__":
            continue
        resolvable[key] = value
    return _SafeEvaluator(resolvable).run(code)
