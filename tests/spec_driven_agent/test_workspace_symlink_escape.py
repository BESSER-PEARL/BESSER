"""A symlink planted in the workspace must not let the worker read outside it.

``run_command`` runs inside bwrap, but the file tools run in the unsandboxed
worker. ``ln -s /proc/self/environ leak.txt`` inside the sandbox leaves a link
that the worker resolves in ITS namespace, so ``search_in_files`` for ``KEY``
printed the worker's own environment - the LLM provider keys - back to the
model. Every tool that reads, lists or writes by path must refuse a link whose
target is outside the workspace.
"""
import os

import pytest

from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, install_probe
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

SECRET = "OPENAI_API_KEY=sk-live-must-not-leak"


@pytest.fixture
def planted(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "environ"
    secret.write_text(SECRET + "\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('hello')\n", encoding="utf-8")
    try:
        os.symlink(secret, workspace / "leak.txt")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")
    return ToolExecutor(workspace=str(workspace)), workspace, secret


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def test_search_in_files_does_not_follow_an_escaping_link(planted):
    executor, _, _ = planted

    result = call(executor, "search_in_files", pattern="API_KEY")

    assert result["matches"] == [], result


def test_list_files_does_not_expose_an_escaping_link(planted):
    executor, _, _ = planted

    result = call(executor, "list_files")

    assert [f["path"] for f in result["files"]] == ["app.py"], result


def test_list_files_survives_a_dangling_link(planted):
    executor, workspace, _ = planted
    os.symlink(workspace / "gone", workspace / "dangling")

    result = call(executor, "list_files")

    assert "error" not in result, result


def test_read_and_write_through_an_escaping_link_are_refused(planted):
    executor, _, secret = planted

    read = call(executor, "read_file", path="leak.txt")
    write = call(executor, "write_file", path="leak.txt", content="overwritten\n")

    assert SECRET not in str(read) and "error" in read, read
    assert "error" in write, write
    assert secret.read_text(encoding="utf-8") == SECRET + "\n"


def test_install_probe_does_not_write_through_a_planted_link(tmp_path):
    target = tmp_path / "worker_code.py"
    target.write_text("ORIGINAL = True\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    try:
        os.symlink(target, workspace / PROBE_FILENAME)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")

    assert install_probe(str(workspace))

    assert target.read_text(encoding="utf-8") == "ORIGINAL = True\n"
    assert not os.path.islink(workspace / PROBE_FILENAME)
