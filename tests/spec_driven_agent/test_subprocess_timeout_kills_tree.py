"""A subprocess timeout must hold when the command leaves a child running.

Phase 1's ``npm install`` had a 180 s timeout and still took ~11 minutes on
Windows; two runs hung past 35 minutes before the first model call.
``subprocess.run`` kills only the direct child (``npm.cmd``'s cmd.exe) and on
Windows then reads the pipes to EOF - which the surviving node process holds
open. On POSIX it returns, but leaves the child running.

The stand-in here is the same shape: a command whose child spawns a
long-lived grandchild that inherits stdout and writes a heartbeat file, so
"the tree is dead" is checkable without platform-specific process APIs.
"""
import os
import stat
import subprocess
import sys
import time

import pytest

from besser.spec_driven_agent.execution.sandbox import SandboxedCommand
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

TIMEOUT = 2
MARGIN = 8

_GRANDCHILD = (
    "import sys, time\n"
    "for _ in range(100):\n"
    "    with open(sys.argv[1], 'a') as fh:\n"
    "        fh.write('.')\n"
    "    time.sleep(0.2)\n"
)
_ROOT = (
    "import subprocess, sys, time\n"
    f"subprocess.Popen([sys.executable, '-c', {_GRANDCHILD!r}, sys.argv[1]])\n"
    "print('started', flush=True)\n"
    "time.sleep(20)\n"
)


def _assert_tree_is_dead(heartbeat):
    """The heartbeat file stops growing once the grandchild is gone."""
    deadline = time.monotonic() + 5
    while not os.path.exists(heartbeat) and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(0.6)
    before = os.path.getsize(heartbeat)
    time.sleep(1.2)
    assert os.path.getsize(heartbeat) == before, "the grandchild survived the timeout"


def _fake_npm(tmp_path, heartbeat):
    """An ``npm`` whose install never finishes and leaves a child behind."""
    script = tmp_path / "fake_npm.py"
    # npm's own arguments ("install ...") replace the heartbeat path argument.
    script.write_text(f"import sys\nsys.argv[1:] = [{str(heartbeat)!r}]\n" + _ROOT)
    if os.name == "nt":
        npm = tmp_path / "npm.cmd"
        npm.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        npm = tmp_path / "npm"
        npm.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        npm.chmod(npm.stat().st_mode | stat.S_IEXEC)
    return str(npm)


def test_run_bounded_kills_the_whole_tree_on_timeout(tmp_path):
    from besser.spec_driven_agent.execution.process import run_bounded

    heartbeat = tmp_path / "beat"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as info:
        run_bounded([sys.executable, "-c", _ROOT, str(heartbeat)], timeout=TIMEOUT)
    elapsed = time.monotonic() - started

    assert elapsed < TIMEOUT + MARGIN, f"timeout of {TIMEOUT}s took {elapsed:.1f}s"
    assert "started" in (info.value.output or ""), "output before the kill is kept"
    _assert_tree_is_dead(heartbeat)


def test_run_bounded_returns_output_like_subprocess_run(tmp_path):
    from besser.spec_driven_agent.execution.process import run_bounded

    result = run_bounded(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
        timeout=30, cwd=str(tmp_path),
    )

    assert result.returncode == 3
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_phase1_npm_install_honours_its_timeout(tmp_path, monkeypatch):
    """The recorded failure, end to end through the Phase 1 call site."""
    heartbeat = tmp_path / "beat"
    npm = _fake_npm(tmp_path, heartbeat)
    workspace = tmp_path / "out"
    frontend = workspace / "web_app" / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "package.json").write_text("{}")

    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.output_dir = str(workspace)
    orch.allow_shell_tools = True
    monkeypatch.setattr("shutil.which", lambda name: npm if name.startswith("npm") else None)
    monkeypatch.setattr(orchestrator_module, "_SCAFFOLD_INSTALL_TIMEOUT_SECONDS", TIMEOUT)

    started = time.monotonic()
    orch._install_scaffold_frontend_dependencies()
    elapsed = time.monotonic() - started

    assert elapsed < TIMEOUT + MARGIN, f"a {TIMEOUT}s install timeout took {elapsed:.1f}s"
    _assert_tree_is_dead(heartbeat)


def test_run_command_honours_its_timeout(tmp_path, monkeypatch):
    """The model's own ``npm install`` goes through run_command."""
    from besser.spec_driven_agent.agent import tool_executor as te

    heartbeat = tmp_path / "beat"
    npm = _fake_npm(tmp_path, heartbeat)
    monkeypatch.setattr(te, "COMMAND_TIMEOUT", TIMEOUT)
    # The sandbox is not what is under test; run the command as the shell would.
    monkeypatch.setattr(te, "sandboxed_command",
                        lambda command, **_: SandboxedCommand(command, True, "test"))

    executor = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    started = time.monotonic()
    result = executor._run_command({"command": f'"{npm}" install'})
    elapsed = time.monotonic() - started

    assert "timed out" in result.get("error", ""), result
    assert elapsed < TIMEOUT + MARGIN, f"a {TIMEOUT}s command timeout took {elapsed:.1f}s"
    _assert_tree_is_dead(heartbeat)
