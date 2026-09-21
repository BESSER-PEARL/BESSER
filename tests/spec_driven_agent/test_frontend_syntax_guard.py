"""A frontend edit that breaks the file must be refused, like a Python one.

``_new_syntax_error`` returned ``None`` for every path that did not end in
``.py``. Over the 143 ``web_app`` runs in ``verification/spec-iterations``,
42 shipped a ``frontend/src/pages/Booking.tsx`` that esbuild cannot parse -
29% of the runs scored ``workflow_ok`` - and no other file in the archive
broke even once. All 42 regenerate from their own ``effective_project.json``
as valid TSX, so none is a template defect: each was corrupted by a recorded
``modify_file`` / ``replace_file_lines`` call the executor accepted and wrote.

The shapes below are the recorded ones. ``options={{...}}`` closed early over
a live tail is run 308z4wo2 t29; the duplicated ``{`` is run 06vtra8k t45 and
hotel-3/hotel-4; the missing comma is run 2_3evtiw t71. Every case here is
verified to APPLY against the pre-fix executor.
"""

import os
import tempfile

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.validation import frontend_source
from besser.spec_driven_agent.validation.toolchain import _demote_tsc_without_deps
from besser.spec_driven_agent.validation.write_diagnostics import diagnose_written_content


PAGE = """import React from 'react';
import TableBlock from '../components/TableBlock';

const Booking: React.FC = () => {
  return (
    <div id="page">
      <TableBlock id="table-booking-4" title="Booking List" options={{
          "showHeader": true,
          "rowsPerPage": 5,
          "columns": [
            {
              "label": "Id",
              "field": "id",
              "type": "int"
            },
            {
              "label": "Bill",
              "field": "billNumber",
              "type": "str"
            }
          ]
        }} />
    </div>
  );
};

export default Booking;
"""


@pytest.fixture
def executor():
    workspace = tempfile.mkdtemp()
    path = os.path.join(workspace, "Booking.tsx")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(PAGE)
    return ToolExecutor(workspace=workspace, per_write_diagnostics=False), path


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def test_modify_that_closes_the_literal_early_is_refused(executor):
    """Run 308z4wo2 t29: a prefix of ``options={{`` replaced by a self-closing
    block, leaving the original ``"columns": [...]`` orphaned below it."""
    tools, path = executor
    result = call(
        tools, "modify_file", path="Booking.tsx",
        old_text='          "showHeader": true,\n          "rowsPerPage": 5,\n',
        new_text='          "showHeader": true,\n          "rowsPerPage": 10\n        }} />\n',
    )
    assert result.get("rejection_kind") == "syntax_error", result
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == PAGE


def test_range_edit_that_duplicates_a_brace_is_refused(executor):
    """Run 06vtra8k t45: ``new_text`` re-sends the block's opening ``{`` while
    ``start_line`` points at the line after it."""
    tools, path = executor
    read = call(tools, "read_file", path="Booking.tsx")
    result = call(
        tools, "replace_file_lines", path="Booking.tsx", read_id=read["read_id"],
        start_line=17, end_line=19,
        new_text='            {\n              "label": "Bill",\n              "field": "billNumber",\n              "type": "str"\n',
    )
    assert result.get("rejection_kind") == "syntax_error", result
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == PAGE


def test_missing_comma_between_column_objects_is_refused(executor):
    """Run 2_3evtiw t71: delimiter-balanced, so only the JSON check sees it."""
    tools, path = executor
    result = call(
        tools, "modify_file", path="Booking.tsx",
        old_text='              "type": "int"\n            },\n',
        new_text='              "type": "int"\n            }\n',
    )
    assert result.get("rejection_kind") == "syntax_error", result
    assert "no longer parses" in result["error"]
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == PAGE


def test_write_file_rewrite_that_does_not_parse_is_refused(executor):
    tools, path = executor
    call(tools, "read_file", path="Booking.tsx")
    result = call(tools, "write_file", path="Booking.tsx",
                  content=PAGE.replace("export default Booking;", "}\nexport default Booking;"))
    assert result.get("rejection_kind") == "syntax_error", result
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == PAGE


def test_a_sound_frontend_edit_still_applies(executor):
    """The guard refuses regressions, not frontend edits."""
    tools, path = executor
    result = call(tools, "modify_file", path="Booking.tsx",
                  old_text='          "rowsPerPage": 5,', new_text='          "rowsPerPage": 25,')
    assert result.get("rejection_kind") is None, result
    with open(path, encoding="utf-8") as handle:
        assert '"rowsPerPage": 25' in handle.read()


def test_an_already_broken_file_can_still_be_repaired():
    """``before`` unparseable means the guard abstains - repair is not blocked."""
    workspace = tempfile.mkdtemp()
    path = os.path.join(workspace, "Booking.tsx")
    broken = PAGE.replace('            {\n              "label": "Bill",', '            {\n             {\n              "label": "Bill",')
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(broken)
    tools = ToolExecutor(workspace=workspace, per_write_diagnostics=False)
    result = call(tools, "modify_file", path="Booking.tsx",
                  old_text='            {\n             {\n', new_text='            {\n')
    assert result.get("rejection_kind") is None, result
    with open(path, encoding="utf-8") as handle:
        assert frontend_source.structure_error(handle.read()) is None


# ----------------------------------------------------------------------
# The scanner must not fire on valid code. These shapes each produced a
# false positive during calibration against 7,208 real-world TS/JS files.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    # An apostrophe in JSX text is not a string delimiter.
    "const A = () => <p>Don't {value} change this</p>;\n",
    # A regex after `=>`, holding a backtick and a quote (axios).
    "const ok = (s) => /^[-_a-zA-Z0-9^`|~,!#$%&'*+.]+$/.test(s);\n",
    # A regex inside a template substitution.
    "const q = (t) => `'${t.replace(/'/g, \"''\")}'`;\n",
    # `/>` and `</tag>` after `}` are JSX, never a regex.
    "const B = () => <Wrap>{(c) => <Inner {...c} />}</Wrap>;\n",
    # Shorthand properties in a JSX expression container are not JSON.
    "const C = () => <P.Provider value={{ a, b, c }}>{kids}</P.Provider>;\n",
    # A JS object literal with identifier keys is not JSON.
    "const D = () => <div style={{ marginTop: '20px' }} />;\n",
    # A trailing comma is legal TS and not a defect.
    'const E = () => <T options={{"a": [1, 2,],}} />;\n',
    # Braces inside a regex are not delimiters.
    "const F = /[{]\\d{2,3}[}]/;\n",
])
def test_valid_frontend_source_is_not_flagged(source):
    assert frontend_source.structure_error(source) is None
    assert frontend_source.json_container_faults(source) == []


def test_diagnostics_name_the_damage_in_a_file_that_arrived_broken():
    findings = diagnose_written_content("src/pages/Booking.tsx", PAGE.replace('            {\n              "label": "Bill",', '            {\n             {\n              "label": "Bill",'))
    assert findings and findings[0]["source"] == "frontend-source"


def test_a_tsc_syntax_error_is_not_demoted_to_advisory():
    """TS1xxx is the grammar; no ``npm install`` can change it."""
    issues: list[str] = []
    real = _demote_tsc_without_deps(
        ["src/pages/Booking.tsx(190,15): error TS1003: Identifier expected.",
         "src/main.tsx(1,17): error TS2307: Cannot find module 'react' or its type declarations."],
        "web_app/frontend", issues,
    )
    assert real == ["src/pages/Booking.tsx(190,15): error TS1003: Identifier expected."]
