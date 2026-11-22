"""One copied ``NNN| `` line must not reach the file.

Uniformly numbered text has been stripped since run fcdh0s9k (see
``test_numbered_new_text.py``). The gap was a MINORITY: run qwen1, turn 74,
``replace_file_lines`` on ``CompleteReservation.tsx`` lines 162-165 with a
new_text whose last line was copied from the read past the selected range:

     166|         <aside id="container_aside_2">

Four clean lines and one numbered one is not "uniform", so it was written
verbatim - and even stripped it would have duplicated line 166. The delivered
page did not compile. ``NEW_TEXT`` below is that call's new_text.

A lone numbered line that is NOT a copy of the file's own line N is still
written as-is (``test_numbered_new_text.py`` pins that); only a verbatim
copy of the display is refused.
"""

import os

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

NEW_TEXT = (
    '           <form id="form" className="record-card" onSubmit={(e) => { e.preventDefault(); }}>\n'
    "\n"
    '             <button type="submit">Submit</button>\n'
    "           </form>\n"
    "         </div>\n"
    ' 166|         <aside id="container_aside_2">\n'
)


def _page():
    lines = [f"        <p>line {n}</p>" for n in range(1, 171)]
    lines[161:165] = [
        '           <form id="form" className="record-card" onSubmit={(e) => { e.preventDefault(); }}>',
        "           </form>",
        "         </div>",
        "",
    ]
    lines[165] = '         <aside id="container_aside_2">'
    return "\n".join(lines) + "\n"


@pytest.fixture
def executor(tmp_path):
    (tmp_path / "Page.tsx").write_text(_page(), encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def _on_disk(executor):
    with open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8") as fh:
        return fh.read()


def test_replace_file_lines_refuses_the_live_shape(executor):
    read = call(executor, "read_file", path="Page.tsx", offset=150, limit=20)

    result = call(executor, "replace_file_lines", path="Page.tsx", read_id=read["read_id"],
                  start_line=162, end_line=165, new_text=NEW_TEXT)

    assert result.get("status") != "modified", result
    assert result.get("rejection_kind") == "line_number_prefix"
    assert "166|" in result["error"], "the refusal must name the offending line"
    assert _on_disk(executor) == _page(), "a refused edit must leave the file untouched"


def test_modify_file_refuses_a_copied_gutter_line(executor):
    result = call(executor, "modify_file", path="Page.tsx",
                  old_text="           </form>\n         </div>\n",
                  new_text="           </form>\n         </div>\n"
                           ' 166|         <aside id="container_aside_2">\n')

    assert result.get("rejection_kind") == "line_number_prefix", result
    assert "166|" not in _on_disk(executor)


def test_write_file_refuses_a_copied_gutter_line(executor):
    call(executor, "read_file", path="Page.tsx")
    content = _page().replace('         <aside id="container_aside_2">',
                              ' 166|         <aside id="container_aside_2">')

    result = call(executor, "write_file", path="Page.tsx", content=content)

    assert result.get("rejection_kind") == "line_number_prefix", result
    assert _on_disk(executor) == _page()


def test_uniform_numbering_is_still_stripped_and_written(executor):
    """The established behaviour: every line numbered is safe to strip."""
    read = call(executor, "read_file", path="Page.tsx", offset=160, limit=10)

    result = call(executor, "replace_file_lines", path="Page.tsx", read_id=read["read_id"],
                  start_line=163, end_line=163,
                  new_text=" 163|           </form>\n 164|         </div>")

    assert result["status"] == "modified", result
    assert "163|" not in _on_disk(executor)
