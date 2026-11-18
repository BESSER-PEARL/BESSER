"""Naming the function of a custom transition condition.

The generated agent wires a custom transition to a function by name, so the condition function must carry
the generated name (``condition_<n>_<m>``), whatever its author called it (the editor's template is
``def condition(...)``). The code may define other functions: helpers keep their names.
"""

import ast
import re


def rename_condition_function(code: str, new_name: str) -> str:
    """Return ``code`` with its condition function renamed to ``new_name``.

    The condition function is the top-level function that no other function in the code calls, so a
    helper written above it keeps its name. If every function is called by another one (or there is
    more than one candidate), the last one defined wins. Code that does not parse falls back to the
    first ``def`` found, as the editor shows the error elsewhere.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return re.sub(r"(?m)^(\s*)def\s+\w+\s*\(", rf"\1def {new_name}(", code, count=1)

    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return code

    def called_by_another(function) -> bool:
        return any(
            isinstance(node, ast.Name) and node.id == function.name
            for other in functions if other is not function
            for node in ast.walk(other)
        )

    entry_points = [function for function in functions if not called_by_another(function)]
    condition = (entry_points or functions)[-1]

    old_name = condition.name
    lines = code.splitlines(keepends=True)
    start, end = condition.lineno - 1, condition.end_lineno
    # the ``def`` line, and any recursive call inside the function itself
    lines[start] = re.sub(rf"(\bdef\s+){re.escape(old_name)}\b", rf"\g<1>{new_name}", lines[start], count=1)
    for index in range(start + 1, end):
        lines[index] = re.sub(rf"\b{re.escape(old_name)}\b(?=\s*\()", new_name, lines[index])
    return "".join(lines)
