"""Names holding a character ``str.splitlines`` breaks on must survive re-import.

Python's tokenizer ends a line only at ``\\n`` / ``\\r``, so a raw ``\\u2028`` inside
a string literal compiles. But every converter strips imports with
``content.splitlines()`` and re-joins with ``\\n``, and ``splitlines`` also breaks
on ``\\x0b \\x0c \\x1c \\x1d \\x1e \\x85 \\u2028 \\u2029``: the literal is cut in two
(a SyntaxError), and the tail of a ``#`` comment becomes a line of code.
"""
import ast

import pytest

from besser.BUML.metamodel.gui import DataSourceElement
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.utilities.buml_code_builder.common import (
    _comment_safe,
    _escape_python_string,
    bind_data_source,
)

SPLITLINES_BREAKS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]


def _as_converters_see_it(source: str) -> str:
    return "\n".join(source.splitlines())


@pytest.mark.parametrize("sep", SPLITLINES_BREAKS)
def test_an_escaped_literal_stays_one_line_and_round_trips(sep):
    value = f"a{sep}b'c\\d"
    source = f"x = '{_escape_python_string(value)}'\n"

    namespace = {}
    exec(compile(_as_converters_see_it(source), "<t>", "exec"), namespace)  # noqa: S102 - test literal

    assert namespace["x"] == value


@pytest.mark.parametrize("sep", SPLITLINES_BREAKS + ["\n", "\r"])
def test_a_comment_cannot_spill_into_code(sep):
    source = f"# Screen: {_comment_safe(f'x{sep}boom = 1')}\n"

    tree = ast.parse(_as_converters_see_it(source))

    assert tree.body == []


def test_bind_data_source_keeps_field_order():
    """``DataSourceElement.fields`` is a set whose setter rewrites ``field_names``
    in set order; the names were set first and so lost the declared order."""
    names = ["h", "c", "a", "f", "b", "g", "e", "d"]
    owner = Class(name="X", attributes={Property(name=n, type=StringType) for n in names})
    source = DataSourceElement(name="src", dataSourceClass=owner, fields=set())

    bind_data_source(source, DomainModel(name="D", types={owner}), "X", field_names=names)

    assert source.field_names == names
    assert {f.name for f in source.fields} == set(names)
