"""The scaffold ships a package.json and no node_modules.

So the first thing the model tries against the generated frontend fails. On
run claude-sonnet-5-q0yzuo43 it reached for ``npx tsc`` at turn 27, spent
turns 28-40 working out why, and installed at turn 41 -- 14 turns, 171s and
32% of that run's spend on a prerequisite nothing was gating.

The fix belongs at scaffold time, not in validation: ``test_validation_honesty``
requires the validator to report the workspace rather than change it, and that
separation is deliberate. These tests pin the install to Phase 1 and pin the
permission it runs under.
"""
import os
import subprocess

import pytest

from besser.spec_driven_agent.orchestrator import LLMOrchestrator


class _Recorder:
    """Stands in for npm; records what would have been run."""

    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        os.makedirs(os.path.join(kwargs["cwd"], "node_modules"), exist_ok=True)
        return subprocess.CompletedProcess(command, 0, "", "")


def _orchestrator(tmp_path, *, allow_shell):
    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.output_dir = str(tmp_path)
    orch.allow_shell_tools = allow_shell
    return orch


def _scaffold(tmp_path):
    frontend = tmp_path / "web_app" / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "package.json").write_text('{"scripts":{"build":"vite build"}}')
    return frontend


def test_dependencies_are_installed_once_at_scaffold_time(tmp_path, monkeypatch):
    frontend = _scaffold(tmp_path)
    recorder = _Recorder()
    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(subprocess, "run", recorder)

    orch = _orchestrator(tmp_path, allow_shell=True)
    orch._install_scaffold_frontend_dependencies()

    assert len(recorder.calls) == 1, "one install, in the folder that declares deps"
    command, kwargs = recorder.calls[0]
    assert command[1] == "install"
    assert os.path.realpath(kwargs["cwd"]) == os.path.realpath(str(frontend))
    assert (frontend / "node_modules").is_dir()


def test_it_does_not_run_without_shell_permission(tmp_path, monkeypatch):
    """Same authorization the model's own install tool runs under."""
    _scaffold(tmp_path)
    recorder = _Recorder()
    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(subprocess, "run", recorder)

    _orchestrator(tmp_path, allow_shell=False)._install_scaffold_frontend_dependencies()

    assert recorder.calls == []


def test_an_already_installed_frontend_is_left_alone(tmp_path, monkeypatch):
    frontend = _scaffold(tmp_path)
    (frontend / "node_modules").mkdir()
    recorder = _Recorder()
    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(subprocess, "run", recorder)

    _orchestrator(tmp_path, allow_shell=True)._install_scaffold_frontend_dependencies()

    assert recorder.calls == [], "a resumed run must not reinstall"


def test_secrets_are_not_exposed_to_the_install(tmp_path, monkeypatch):
    """npm runs package lifecycle scripts, so the provider key must not be
    in its environment -- the same rule the build already follows."""
    _scaffold(tmp_path)
    recorder = _Recorder()
    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(subprocess, "run", recorder)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "never-pass-this-to-npm")
    monkeypatch.setenv("OPENAI_API_KEY", "never-pass-this-to-npm")

    _orchestrator(tmp_path, allow_shell=True)._install_scaffold_frontend_dependencies()

    env = recorder.calls[0][1]["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired("npm", 180),
    OSError("npm vanished"),
])
def test_a_failed_install_never_aborts_the_run(tmp_path, monkeypatch, failure):
    """Best effort: the model still has install_dependencies, and the
    validator still reports an uninstalled frontend exactly as before."""
    _scaffold(tmp_path)

    def boom(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(subprocess, "run", boom)

    _orchestrator(tmp_path, allow_shell=True)._install_scaffold_frontend_dependencies()


def test_no_npm_on_the_host_is_not_an_error(tmp_path, monkeypatch):
    _scaffold(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)

    _orchestrator(tmp_path, allow_shell=True)._install_scaffold_frontend_dependencies()
