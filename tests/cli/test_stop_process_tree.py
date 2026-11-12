"""What 'besser apply' does to tests that run past test_timeout, on the Windows path too.

The Windows branch cannot run here: it is driven with ``os.name`` and ``taskkill`` replaced.
"""

import subprocess

import pytest

from besser.cli import commands


class _Process:
    pid = 4242

    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


@pytest.mark.parametrize("failure", [FileNotFoundError("taskkill"), subprocess.TimeoutExpired("taskkill", 5)])
def test_on_windows_the_test_process_is_killed_even_when_taskkill_fails(monkeypatch, failure):
    def taskkill(*args, **kwargs):
        raise failure

    monkeypatch.setattr(commands.os, "name", "nt")
    monkeypatch.setattr(commands.subprocess, "run", taskkill)
    process = _Process()

    commands._stop_process_tree(process)

    assert process.killed
