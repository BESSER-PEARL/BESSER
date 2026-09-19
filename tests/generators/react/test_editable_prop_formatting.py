"""A generated page must be editable by the agent that has to fix it.

``_format_prop`` serialised every structured prop with a compact
``json.dumps``, so a table's whole options dict landed on one line. In the
live hotel app ``Booking.tsx`` line 25 was **3,762 characters**, with six
lines over 500 in a 38-line file.

That is where the `frontend contract` blocker lived — an editable ``bill``
field the backend will not accept — and it survived all four live Qwen runs
(7aybctis, trilraak, lsrnaime, pcovsppe). Every attempt on that line came
back "old_text and new_text are identical": nothing retypes 3.7k characters
and changes one field. It was the single least editable line in the codebase
and it carried the defect the agent was asked to fix.

The frontend contract checker scans braces and uses ``json.raw_decode``, both
newline-tolerant, so it reads either form — this must stay true.
"""

import json

import pytest

from besser.generators.react.page_builder import PageBuilderMixin


SMALL = {"showHeader": True, "rowsPerPage": 5}
LARGE = {
    "showHeader": True, "stripedRows": False, "showPagination": True,
    "rowsPerPage": 5, "actionButtons": True,
    "columns": [
        {"label": f"Field{i}", "column_type": "field", "field": f"field{i}",
         "type": "str", "required": i % 2 == 0}
        for i in range(12)
    ],
}


@pytest.fixture
def builder():
    return PageBuilderMixin()


def test_a_small_prop_stays_on_one_line(builder):
    rendered = builder._format_prop("options", SMALL)

    assert "\n" not in rendered
    assert rendered == "options={" + json.dumps(SMALL, ensure_ascii=False) + "}"


def test_a_large_prop_is_written_across_lines(builder):
    builder._prop_indent = 8

    rendered = builder._format_prop("options", LARGE)

    lines = rendered.splitlines()
    assert len(lines) > 20, "the options dict must not stay on one line"
    assert max(len(line) for line in lines) < 200
    # Continuation lines align under the component, not at column 0.
    assert all(line.startswith(" " * 10) for line in lines[1:] if line.strip())


def test_the_value_survives_the_round_trip(builder):
    builder._prop_indent = 4

    rendered = builder._format_prop("options", LARGE)
    body = rendered[len("options={"):-1]

    assert json.loads(body) == LARGE


def test_the_contract_checker_still_reads_a_multi_line_table(builder, tmp_path):
    """The newline-tolerant scan is load-bearing; pin it."""
    from besser.generators.llm.validation.frontend_schema import _table_metadata

    builder._prop_indent = 8
    options = builder._format_prop("options", LARGE)
    binding = builder._format_prop("dataBinding", {"entity": "Booking", "endpoint": "/booking/"})
    source = f'export default () => <TableBlock {options} {binding} />;\n'

    found = list(_table_metadata(source))

    assert len(found) == 1
    _line, parsed_options, parsed_binding = found[0]
    assert parsed_options == LARGE
    assert parsed_binding["entity"] == "Booking"


def test_scalar_props_are_unchanged(builder):
    assert builder._format_prop("title", "Booking List") == 'title="Booking List"'
    assert builder._format_prop("showHeader", True) == "showHeader={true}"
    assert builder._format_prop("rowsPerPage", 5) == "rowsPerPage={5}"
