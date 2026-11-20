"""Reading one file in 10-line windows must be called out.

Pooled over three live Qwen runs, ``read_file`` was **56% of all tool calls**
— 136 reads against 64 edits. Run trilraak read
``web_app/backend/routers/booking.py`` 36 times across 29 distinct 10-line
spans (offset 170, 179, 189, 190, 199, 204, ...) on a 598-line file, roughly a
third of its 80-turn budget. ``MAX_FILE_READ`` is 60,000 characters, so the
whole file fits in a single read; nothing forced the crawl.

The old description actively invited it ("For large files (>200 lines), use
offset and limit ... Example: offset=50, limit=30"), and the unconditional
"Large file. Use offset/limit to read specific sections." hint repeated the
advice on every full read.
"""

import os
import tempfile

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


@pytest.fixture
def executor():
    workspace = tempfile.mkdtemp()
    body = "".join(f"line_{i} = {i}\n" for i in range(1, 599))
    with open(os.path.join(workspace, "router.py"), "w", encoding="utf-8") as handle:
        handle.write(body)
    return ToolExecutor(workspace=workspace)


def read(executor, **args):
    return executor.execute_typed("read_file", {"path": "router.py", **args}).payload


def test_the_third_narrow_read_of_one_file_is_called_out(executor):
    read(executor, offset=170, limit=10)
    read(executor, offset=179, limit=10)
    third = read(executor, offset=189, limit=10)

    assert "read 3 of router.py" in third["hint"]
    assert "10 of 599 lines" in third["hint"]  # includes the EOF anchor line
    assert "single read" in third["hint"]


def test_the_first_two_narrow_reads_are_not_nagged(executor):
    assert "hint" not in read(executor, offset=170, limit=10)
    assert "hint" not in read(executor, offset=179, limit=10)


def test_repeated_wide_reads_are_fine(executor):
    """A model reading real blocks is doing the right thing."""
    for offset in (0, 100, 200):
        result = read(executor, offset=offset, limit=120)

    assert "hint" not in result


def test_a_full_read_no_longer_advertises_pagination(executor):
    """The whole file fits; telling it to paginate is what caused the crawl."""
    result = read(executor)

    assert result["total_lines"] == 599
    assert "hint" not in result


def test_a_small_file_read_repeatedly_is_not_nagged(tmp_path):
    (tmp_path / "small.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))

    for _ in range(4):
        result = executor.execute_typed(
            "read_file", {"path": "small.py", "offset": 0, "limit": 2}).payload

    assert "hint" not in result


# One line longer than MAX_FILE_READ (a minified bundle). It used to come back
# as empty content naming the SAME offset again, on every retry.
BUNDLE = "var a=1;\n" + "x" * 70000 + "\nvar b=2;\n"


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "bundle.js").write_text(BUNDLE, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def _read(executor, **args):
    return executor.execute_typed("read_file", {"path": "bundle.js", **args}).payload


def test_a_line_over_the_read_cap_is_clipped_and_paging_moves_past_it(bundle):
    first = _read(bundle)
    assert first["end_line"] == 1 and "offset=1" in first["hint"]

    clipped = _read(bundle, offset=1)
    assert (clipped["start_line"], clipped["end_line"]) == (2, 2)
    assert clipped["content"].startswith("2| xxxx")
    assert "[line 2 clipped: 70000 chars]" in clipped["content"]
    assert "offset=2" in clipped["hint"]

    rest = _read(bundle, offset=2)
    assert rest["content"].startswith("3| var b=2;")
    assert "truncated" not in rest


def test_a_clipped_line_is_never_editable_by_range(bundle):
    clipped = _read(bundle, offset=1)
    whole = _read(bundle)
    for view in (clipped, whole):
        if "read_id" not in view:
            continue
        refused = bundle.execute_typed("replace_file_lines", {
            "path": "bundle.js", "read_id": view["read_id"], "start_line": 2,
            "end_line": 2, "new_text": "x\n"}).payload
        assert refused["rejection_kind"] == "unread_range", refused
    assert open(os.path.join(bundle.workspace, "bundle.js"), encoding="utf-8").read() == BUNDLE


def test_a_rewrite_that_would_drop_a_clipped_line_names_it(bundle):
    """write_file must refuse (a rewrite cannot reproduce 70,000 unseen chars),
    but not by sending the model back to a read that cannot show more."""
    for offset in (0, 1, 2):
        _read(bundle, offset=offset)

    refused = bundle.execute_typed("write_file", {
        "path": "bundle.js", "content": "var a=1;\nvar b=2;\n"}).payload

    assert "line 2" in refused["error"] and "modify_file" in refused["error"], refused
    assert "Read the rest" not in refused["error"]


def test_the_ladder_does_not_send_a_clipped_file_to_a_rewrite(bundle):
    for _ in range(2):
        refused = bundle.execute_typed("modify_file", {
            "path": "bundle.js", "old_text": "var c=3;", "new_text": "var c=4;"}).payload
        assert "error" in refused
    assert refused.get("edit_recovery", {}).get("next_tool") != "write_file", refused
