"""A method-parameter default must not escape into the generated router.

`BackendGenerator` renders `param.default_value` in four executable places and
two docstrings. The executable ones were hardened first; the docstrings were
missed, and they are the more dangerous half: `python_default` never raises for
a `str`, so generation SUCCEEDS and a value containing a triple-double-quote
closes the generated docstring and drops whatever follows at statement
indentation -- inside a router that `/besser_api/deploy-app` executes.

`default_value` reaches `Parameter` unvalidated from editor request JSON, so
this is reachable by anyone who can model.

This was the third site of the same sink found in one branch (after the
SQLAlchemy macro and its association-class copy), which is why this file tests
the generator end to end rather than the helper in isolation.
"""
import re

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, Parameter, PrimitiveDataType, Property,
)
from besser.generators.backend import BackendGenerator
from besser.generators.default_literals import InvalidDefaultValueError

STR = PrimitiveDataType("str")
INT = PrimitiveDataType("int")

# Closes the docstring, then emits a statement at body indentation.
DOCSTRING_ESCAPE = '"""\n    import os; os.system("id")\n    """'


def _model_with_param_default(default, param_type=STR):
    param = Parameter(name="note", type=param_type)
    param.default_value = default
    # Exercise executable default handling, not an intentionally inert 501 stub.
    method = Method(name="annotate", parameters={param},
                    code="def annotate(self, note):\n    return note\n")
    book = Class(name="Book")
    book.attributes = {Property(name="id", type=INT, is_id=True)}
    book.methods = {method}
    return DomainModel(name="M", types={book})


def _generate(model, out):
    BackendGenerator(model=model, output_dir=str(out)).generate()
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in out.rglob("*.py")
    )


def test_a_docstring_escape_cannot_break_out(tmp_path):
    """The regression this file exists for.

    The payload legitimately appears as TEXT -- inside a repr()'d string
    literal on the value line, and as flattened single-quoted text in the
    docstring. That is inert. What must never happen is the payload becoming
    EXECUTABLE, which only ast can answer: a substring check would either miss
    the real bug or fail on harmless data. (It failed on harmless data first.)
    """
    import ast

    code = _generate(_model_with_param_default(DOCSTRING_ESCAPE), tmp_path)
    tree = ast.parse(code)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "system"
    ]
    assert not calls, (
        "the modelled default escaped into an executable os.system(...) call "
        "at line(s) %s" % [c.lineno for c in calls]
    )

    # The payload must survive only as a string constant -- never as code.
    # (Do NOT assert on `import os`: the generated backend imports os itself.)
    payload_as_data = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "os.system" in node.value
        for node in ast.walk(tree)
    )
    assert payload_as_data, (
        "the payload is not present as a string constant -- either it was "
        "dropped entirely, or it landed somewhere ast does not see as data"
    )


def test_every_generated_file_still_parses(tmp_path):
    code = _generate(_model_with_param_default(DOCSTRING_ESCAPE), tmp_path)
    compile(code, "<generated>", "exec")


def test_a_numeric_default_is_not_stringified(tmp_path):
    """The value sites used to receive the RENDERED ANNOTATION rather than the
    modelled type, so `python_default` fell through to its quoted fallback and
    a legitimate `5` was emitted as `'5'`."""
    code = _generate(_model_with_param_default("5", INT), tmp_path)
    assert re.search(r"params\.get\('note',\s*5\)", code), (
        "an int default should render as 5, not '5' -- check the template "
        "passes param[4] (the modelled type), not param[1] (the annotation)"
    )


def test_a_legitimate_string_default_survives(tmp_path):
    code = _generate(_model_with_param_default("hello", STR), tmp_path)
    assert "'hello'" in code


@pytest.mark.parametrize("bad", ['__import__("os").getcwd()', "not-a-number"])
def test_a_non_literal_numeric_default_is_refused(bad, tmp_path):
    with pytest.raises(InvalidDefaultValueError):
        _generate(_model_with_param_default(bad, INT), tmp_path)
