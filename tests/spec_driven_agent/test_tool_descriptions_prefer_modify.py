"""The file-tool descriptions must agree with the prompt and the executor.

Until 2026-09-18 the schema told the model "1-2 edits per file; three or
more changes -> write_file" (tools.py, since 17886e0ff) while Rule 2 of the
system prompt said modify_file for every change and the executor refused
exactly those write_file calls. Aider's rule is the one the executor wants:
several small blocks, just the changing lines plus a few for uniqueness.
"""
from besser.spec_driven_agent.tools import FILE_TOOLS


def _desc(name):
    return next(t["description"] for t in FILE_TOOLS if t["name"] == name)


def test_neither_file_tool_sends_three_plus_changes_to_write_file():
    for name in ("modify_file", "write_file"):
        d = _desc(name).lower()
        assert "three or more" not in d, name
        assert "1-2 edits" not in d, name
        assert "fewer round-trips" not in d, name


def test_modify_file_asks_for_one_call_per_site_and_short_anchors():
    d = _desc("modify_file").lower()
    assert "per change" in d or "per edit" in d
    assert "same turn" in d
    assert "short" in d or "just the lines" in d


def test_write_file_is_for_new_files_and_deliberate_rewrites_only():
    d = _desc("write_file").lower()
    assert "new file" in d
    assert "modify_file" in d
