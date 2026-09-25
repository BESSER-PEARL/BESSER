"""
Structural and semantic lint for user-supplied Python code (CustomCodeAction
bodies and reasoning-state Tool code).

This module is NOT a security boundary. The semantic rules are an AST denylist
that is trivially bypassable (``getattr(__builtins__, "ev" + "al")``, attribute
chains on already-imported objects, ``importlib`` reached through another
module, ...). Its only job is to give users early, readable feedback on code
that is malformed or obviously out of place. Untrusted code must only ever run
inside the agent simulator sandbox, which is the real isolation boundary.

Rules enforced on CustomCodeAction source:
  1. Non-empty source.
  2. Valid Python syntax.
  3. Exactly one top-level statement.
  4. That statement must be a function definition (def ...).
  5. The function's first positional parameter must be named 'session'.
  6. (simulation only) No imports of high-risk modules.
  7. (simulation only) No calls to dangerous built-ins (exec, eval, __import__,
     compile, open, breakpoint).

Tool code (see :func:`lint_simulation_tool_code`) gets rules 1, 2, 6 and 7.
"""

import ast

from besser.utilities.web_modeling_editor.backend.services.exceptions import CodeValidationError

# Modules that give direct access to the OS, network, or interpreter internals.
_BLOCKED_MODULES: frozenset = frozenset({
    "os",
    "subprocess",
    "socket",
    "sys",
    "ctypes",
    "importlib",
    "shutil",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "marshal",
    "builtins",
    "pty",
    "signal",
})

# Built-in callables that allow arbitrary code execution or filesystem access.
_BLOCKED_BUILTINS: frozenset = frozenset({
    "exec",
    "eval",
    "__import__",
    "compile",
    "open",
    "breakpoint",
})


def _parse(source: str) -> ast.Module:
    """Rules 1-2: reject empty source and syntax errors."""
    if not source or not source.strip():
        raise CodeValidationError("Custom code body cannot be empty.")
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise CodeValidationError(f"Syntax error in custom code: {exc}") from exc


def _check_simulation_denylist(tree: ast.AST) -> None:
    """Rules 6-7: flag blocked-module imports and dangerous built-in calls."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise CodeValidationError(
                        f"Import of '{alias.name}' is not allowed during agent simulation."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise CodeValidationError(
                        f"Import from '{node.module}' is not allowed during agent simulation."
                    )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                raise CodeValidationError(
                    f"Call to '{node.func.id}()' is not allowed during agent simulation."
                )


def validate_custom_code_action(source: str, *, simulation: bool = False) -> None:
    """
    Lint a CustomCodeAction source string.

    Rules 1-5 (structural) are always enforced:
      - Non-empty, valid syntax, exactly one top-level def, first param named 'session'.

    Rules 6-7 (semantic) are only enforced when simulation=True:
      - No blocked-module imports, no dangerous built-in calls.
      - Standard code generation does not restrict which libraries are imported.

    This is a best-effort lint for early user feedback, NOT a security
    boundary: the semantic rules are easy to bypass. Isolation of untrusted
    code is the job of the agent simulator sandbox.

    Raises CodeValidationError with a human-readable message on any violation.
    Returns None when the source is valid.
    """
    tree = _parse(source)

    # Rule 3: Exactly one top-level statement.
    if len(tree.body) != 1:
        raise CodeValidationError(
            f"Custom code must contain exactly one top-level function definition "
            f"(found {len(tree.body)} top-level statement(s)). "
            "All code must be inside the function body — no imports, assignments, "
            "or other statements at the root level are allowed."
        )

    stmt = tree.body[0]

    # Rule 4: Must be a plain (non-async) function definition.
    if isinstance(stmt, ast.AsyncFunctionDef):
        raise CodeValidationError(
            "Async function definitions are not allowed in custom code. "
            "Use a regular 'def' instead."
        )
    if not isinstance(stmt, ast.FunctionDef):
        raise CodeValidationError(
            "Custom code must start with a function definition "
            "(e.g. 'def my_action(session: Session):')."
        )

    func: ast.FunctionDef = stmt

    # Rule 5: First positional parameter must be 'session'.
    positional_args = func.args.args
    if not positional_args or positional_args[0].arg != "session":
        raise CodeValidationError(
            "The function's first parameter must be named 'session' "
            "(e.g. 'def my_action(session: Session):')."
        )

    if not simulation:
        return

    # Rules 6 & 7 (simulation only): walk the entire function for blocked patterns.
    _check_simulation_denylist(func)


def lint_simulation_tool_code(source: str) -> None:
    """
    Lint a reasoning-state Tool's source before it is sent to the simulator.

    Tools are free-form callables (their first parameter is not ``session``),
    so only rules 1, 2, 6 and 7 apply. Like :func:`validate_custom_code_action`
    this is early user feedback, NOT a security boundary.

    Raises CodeValidationError with a human-readable message on any violation.
    """
    _check_simulation_denylist(_parse(source))
