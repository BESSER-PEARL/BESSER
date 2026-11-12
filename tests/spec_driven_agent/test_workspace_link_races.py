"""A link or FIFO swapped in after ``_safe_path``'s check must not be followed.

With shell tools on, a background process the model started through
``run_command`` can swap ``x.txt`` (or its parent directory) between a real
entry and a link in a loop. ``_safe_path`` resolves and checks the path, then
the tool opens it; a link planted in between used to be followed, so
``read_file`` returned the unsandboxed worker's files and ``write_file``
overwrote them. These tests plant the swap deterministically, right after
every ``_safe_path`` check returns, and assert the outside file is neither
returned nor changed. The FIFO cases run in a thread with a timeout so a
regression fails instead of hanging the suite.
"""
import os
import threading

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

CONTENT = "alpha = 1\n"
SECRET = "OPENAI_API_KEY=sk-live-must-not-leak\n"
CALL_TIMEOUT = 5  # seconds; a tool blocked on a FIFO never returns


def call(executor, name, **args):
    """The tool's payload, or a failure when it does not return in time."""
    out = []
    worker = threading.Thread(target=lambda: out.append(executor.execute_typed(name, args).payload), daemon=True)
    worker.start()
    worker.join(CALL_TIMEOUT)
    assert out, f"{name} did not return within {CALL_TIMEOUT}s"
    return out[0]


@pytest.fixture
def setup(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "ws"
    (workspace / "sub").mkdir(parents=True)
    for rel in ("x.txt", "sub/x.txt"):
        (workspace / rel).write_text(CONTENT, encoding="utf-8")
    try:
        os.symlink(outside, tmp_path / "probe")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")
    executor = ToolExecutor(workspace=str(workspace), per_write_diagnostics=False)
    return executor, workspace, outside


def swap_after_each_check(monkeypatch, workspace, entry, plant):
    """Every ``_safe_path`` call sees the real ``entry``; it is replaced by
    ``plant(path)`` the moment the check returns (the racer's window)."""
    real, hidden = workspace / entry, workspace / (entry + ".real")
    original = ToolExecutor._safe_path

    def checked_then_swapped(self, rel_path):
        if os.path.lexists(hidden):
            os.remove(real)  # the planted link or FIFO
            os.rename(hidden, real)
        try:
            return original(self, rel_path)
        finally:
            os.rename(real, hidden)
            plant(str(real))

    monkeypatch.setattr(ToolExecutor, "_safe_path", checked_then_swapped)


def link_to(target):
    return lambda path: os.symlink(target, path)


# (rel path the tool names, entry that is swapped, link target)
LEAF = ("x.txt", "x.txt", "x.txt")
PARENT = ("sub/x.txt", "sub", "")
RACES = [pytest.param(LEAF, id="leaf"), pytest.param(PARENT, id="parent")]


def arm(monkeypatch, setup, race, outside_text):
    """Write the outside file, then start swapping. Returns (executor, rel, outside file)."""
    executor, workspace, outside = setup
    rel, entry, target = race
    victim = outside / "x.txt"
    victim.write_text(outside_text, encoding="utf-8")
    swap_after_each_check(monkeypatch, workspace, entry, link_to(outside / target if target else outside))
    return executor, rel, victim


@pytest.mark.parametrize("race", RACES)
def test_read_file_does_not_follow_a_swapped_link(monkeypatch, setup, race):
    executor, rel, _ = arm(monkeypatch, setup, race, SECRET)

    result = call(executor, "read_file", path=rel)

    assert "sk-live" not in str(result), result


@pytest.mark.parametrize("race", RACES)
def test_check_syntax_does_not_follow_a_swapped_link(monkeypatch, setup, race):
    executor, rel, _ = arm(monkeypatch, setup, race, "def broken(:\n")

    result = call(executor, "check_syntax", path=rel)

    assert "Syntax error" not in str(result), result


@pytest.mark.parametrize("race", RACES)
def test_write_file_does_not_write_through_a_swapped_link(monkeypatch, setup, race):
    executor = setup[0]
    call(executor, "read_file", path=race[0])  # write_file needs a full read first
    executor, rel, victim = arm(monkeypatch, setup, race, CONTENT)

    call(executor, "write_file", path=rel, content="PWNED\n")

    assert victim.read_text(encoding="utf-8") == CONTENT


@pytest.mark.parametrize("race", RACES)
def test_modify_file_does_not_write_through_a_swapped_link(monkeypatch, setup, race):
    executor, rel, victim = arm(monkeypatch, setup, race, CONTENT)

    call(executor, "modify_file", path=rel, old_text="alpha = 1", new_text="alpha = 'PWNED'")

    assert victim.read_text(encoding="utf-8") == CONTENT


@pytest.mark.parametrize("race", RACES)
def test_replace_file_lines_does_not_write_through_a_swapped_link(monkeypatch, setup, race):
    read = call(setup[0], "read_file", path=race[0])
    executor, rel, victim = arm(monkeypatch, setup, race, CONTENT)

    call(executor, "replace_file_lines", path=rel, read_id=read["read_id"],
         start_line=1, end_line=1, new_text="alpha = 'PWNED'\n")

    assert victim.read_text(encoding="utf-8") == CONTENT


def test_delete_file_does_not_delete_through_a_swapped_parent(monkeypatch, setup):
    executor, rel, victim = arm(monkeypatch, setup, PARENT, CONTENT)

    call(executor, "delete_file", path=rel)

    assert victim.exists()


def test_search_does_not_follow_a_swapped_link(monkeypatch, setup):
    executor, _, _ = arm(monkeypatch, setup, LEAF, SECRET)

    result = call(executor, "search_in_files", pattern="sk-live")

    assert result["matches"] == [], result


def test_task_evidence_does_not_read_through_a_swapped_link(monkeypatch, setup):
    executor, rel, _ = arm(monkeypatch, setup, LEAF, SECRET)

    _, error = executor._task_implementation_evidence(1, {"evidence": [
        {"id": 1, "path": rel, "quote": "sk-live-must-not-leak"}], "existing": True})

    assert error and "not a readable workspace file" in error, error
