"""A FIFO made through run_command must not hang the unsandboxed tools that
walk the workspace (search_in_files opened every walked file)."""
import os
import sys

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

pytestmark = pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "mkfifo"),
                                reason="needs POSIX FIFOs")


def test_search_skips_a_fifo(tmp_path):
    (tmp_path / "app.py").write_text("needle = 1\n", encoding="utf-8")
    os.mkfifo(tmp_path / "trap.py")
    executor = ToolExecutor(workspace=str(tmp_path))
    result = executor.execute("search_in_files", {"pattern": "needle"})
    assert "app.py" in str(result)
