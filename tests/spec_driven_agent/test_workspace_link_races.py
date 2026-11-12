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
from besser.utilities import workspace_fs

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


# -- FIFOs: never opened, never block the per-path lock ---------------------

needs_mkfifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no mkfifo on this platform")


@needs_mkfifo
@pytest.mark.parametrize("tool,args", [
    ("write_file", {"content": "x"}),
    ("read_file", {}),
    ("check_syntax", {}),
])
def test_a_fifo_is_refused_without_blocking(setup, tool, args):
    executor, workspace, _ = setup
    os.mkfifo(workspace / "pipe")

    result = call(executor, tool, path="pipe", **args)

    assert "error" in result, result


def fifo_after(monkeypatch, owner, attr, target, concerns):
    """Wrap ``owner.attr`` so that the first call ``concerns`` accepts is
    followed by ``target`` being replaced with a FIFO (the racer's window)."""
    real = getattr(owner, attr)
    armed = [True]

    def checked_then_fifo(*args, **kwargs):
        answer = real(*args, **kwargs)
        if armed[0] and concerns(*args, **kwargs):
            armed[0] = False
            os.remove(target)
            os.mkfifo(target)
        return answer

    monkeypatch.setattr(owner, attr, checked_then_fifo)


# Each check a FIFO can slip past: the tool's own isfile, open_plain's lstat,
# and open_plain_write's lstat of the name it is about to open.
SWAP_POINTS = {
    "isfile": lambda target: (os.path, "isfile", lambda path: os.fspath(path) == target),
    "read_check": lambda target: (workspace_fs, "_refuse_unless_plain_file",
                                  lambda path, root: os.fspath(path) == target),
    "write_check": lambda target: (os, "stat", lambda path, *a, follow_symlinks=True, **k:
                                   not follow_symlinks and os.path.basename(os.fspath(path)) == "x.txt"),
}


@needs_mkfifo
@pytest.mark.parametrize("tool,args,point", [
    ("read_file", {}, "isfile"),
    ("read_file", {}, "read_check"),
    ("check_syntax", {}, "read_check"),
    ("modify_file", {"old_text": "alpha = 1", "new_text": "alpha = 2"}, "read_check"),
    ("modify_file", {"old_text": "alpha = 1", "new_text": "alpha = 2"}, "write_check"),
    ("write_file", {"content": "beta = 2\n"}, "write_check"),
])
def test_a_fifo_swapped_in_after_a_check_does_not_block(monkeypatch, setup, tool, args, point):
    executor, workspace, _ = setup
    call(executor, "read_file", path="x.txt")  # write_file needs a full read first
    target = str(workspace / "x.txt")
    owner, attr, concerns = SWAP_POINTS[point](target)
    fifo_after(monkeypatch, owner, attr, target, concerns)

    result = call(executor, tool, path="x.txt", **args)

    assert "error" in result, result


def test_a_working_dir_is_not_created_through_a_swapped_parent(monkeypatch, setup):
    executor, _, _ = arm(monkeypatch, setup, PARENT, CONTENT)
    outside = setup[2]

    with pytest.raises(OSError):
        executor._safe_cwd("sub/build")

    assert not (outside / "build").exists()
