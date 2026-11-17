"""A line whose numbered form is exactly ``MAX_FILE_READ`` long fits in one read.

``_unreadable_line`` treats only a longer line as unreadable, but ``_read_file`` searched
for a line break strictly before ``MAX_FILE_READ``: such a line was reported clipped, with
no read_id, while the write guard counted it as readable.
"""

import pytest

from besser.spec_driven_agent.agent.edit_apply import gutter_line
from besser.spec_driven_agent.agent.tool_executor import MAX_FILE_READ, ToolExecutor

_PREFIX = len(gutter_line(1, ""))


@pytest.mark.parametrize("extra", [-1, 0, 1])
def test_read_and_write_guard_agree_on_a_long_first_line(tmp_path, extra):
    line = "x" * (MAX_FILE_READ - _PREFIX + extra)
    (tmp_path / "big.py").write_text(line + "\nsecond = 2\n", encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path))

    result = ex._read_file({"path": "big.py"})
    unreadable = ex._unreadable_line("big.py")

    fits = len(gutter_line(1, line)) <= MAX_FILE_READ
    assert fits == (extra <= 0)
    assert unreadable == (None if fits else (1, len(line)))
    assert ("read_id" in result) is fits
    assert result["end_line"] == 1
    if fits:
        assert result["content"].startswith(gutter_line(1, line) + "\n... [truncated")
