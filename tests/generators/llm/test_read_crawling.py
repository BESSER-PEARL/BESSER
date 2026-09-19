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

from besser.generators.llm.tool_executor import ToolExecutor


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
