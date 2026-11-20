"""The lab preview supervisor, driven with small Python scripts as fake services (no real server)."""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time

import pytest

from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT, RepoSpec
from besser.utilities.web_modeling_editor.backend.services.lab.preview import (
    PreviewService,
    PreviewSupervisor,
    default_services,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

WAIT_SECONDS = 30.0

# Writes {"pid", "env", "cwd"} to argv[1], appends a tick to argv[1] + ".ticks" while it lives.
# argv[2]: "serve" (default), "crash" (exits 3) or "spawn" (also starts a ticking grandchild).
FAKE_SERVICE = textwrap.dedent("""
    import json, os, subprocess, sys, time
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "serve"
    print("fake service up", flush=True)
    if mode == "spawn":
        subprocess.Popen([sys.executable, sys.argv[0], out + ".child"])
    with open(out + ".tmp", "w") as handle:
        json.dump({"pid": os.getpid(), "env": dict(os.environ), "cwd": os.getcwd()}, handle)
    os.replace(out + ".tmp", out)
    if mode == "crash":
        sys.exit(3)
    while True:
        with open(out + ".ticks", "a") as handle:
            handle.write(".")
        time.sleep(0.05)
""")


def _secret_setting_names() -> list[str]:
    """The environment names of the backend's ``SecretStr`` settings (the lab key, the OAuth secret, ...)."""
    names = [field.validation_alias for field in BackendSettings.model_fields.values()
             if "SecretStr" in str(field.annotation)]
    assert "BESSER_LAB_LLM_API_KEY" in names and "GITHUB_CLIENT_SECRET" in names
    return names


def _wait_for(condition, what: str):
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _ticking(path: str) -> bool:
    """True while some process still appends to ``path`` + ".ticks"."""
    ticks = path + ".ticks"
    before = os.path.getsize(ticks) if os.path.exists(ticks) else 0
    time.sleep(0.5)
    return (os.path.getsize(ticks) if os.path.exists(ticks) else 0) != before


def _status(supervisor: PreviewSupervisor, name: str) -> dict:
    return next(entry for entry in supervisor.status() if entry["name"] == name)


@pytest.fixture
def fake_script(tmp_path):
    path = tmp_path / "fake_service.py"
    path.write_text(FAKE_SERVICE, encoding="utf-8")
    return str(path)


@pytest.fixture
def supervisors():
    """Builds supervisors and always stops them, so no fake service outlives its test."""
    made: list[PreviewSupervisor] = []

    def make(services, log_dir):
        supervisor = PreviewSupervisor(services, log_dir=str(log_dir))
        made.append(supervisor)
        return supervisor

    yield make
    for supervisor in made:
        supervisor.stop()


def _service(name, script, out, *args, cwd, env=None, watch=(), port=1):
    return PreviewService(name=name, argv=(sys.executable, script, str(out), *args), cwd=str(cwd),
                          env=env or {}, watch=tuple(str(path) for path in watch), port=port)


def test_start_logs_reports_running_and_stop_ends_the_process(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path, port=9100)], tmp_path / "logs")

    assert supervisor.status() == [{"name": "backend", "running": False, "detail": "not started"}]
    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the fake service")

    assert supervisor.status() == [{"name": "backend", "running": True, "detail": "running on port 9100"}]
    assert _ticking(str(out))
    log = (tmp_path / "logs" / "backend.log").read_text(encoding="utf-8")
    assert "fake service up" in log

    supervisor.stop()
    assert supervisor.status() == [{"name": "backend", "running": False, "detail": "stopped"}]
    assert not _ticking(str(out))


def test_child_environment_is_the_safe_one_plus_the_service_entries(tmp_path, fake_script, supervisors,
                                                                       monkeypatch):
    monkeypatch.setenv("BESSER_LAB_LLM_API_KEY", "sk-lab-secret")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "oauth-secret")
    monkeypatch.setenv("BESSER_LAB_ALLOWED_LOGINS", "jordi")
    # Every secret the backend's settings read, and the providers' own keys.
    secrets = {name: f"secret-value-{index}" for index, name in enumerate(
        [*_secret_setting_names(), "ANTHROPIC_API_KEY", "OPENAI_API_KEY"])}
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "env.json"
    workdir = tmp_path / "work"
    workdir.mkdir()
    supervisor = supervisors(
        [_service("backend", fake_script, out, cwd=workdir, env={"CORS_ORIGINS": "http://x"})], tmp_path / "logs")

    supervisor.start()
    seen = _wait_for(lambda: _read_json(str(out)), "the fake service")

    env = seen["env"]
    assert "BESSER_LAB_LLM_API_KEY" not in env
    assert "GITHUB_CLIENT_SECRET" not in env
    assert "BESSER_LAB_ALLOWED_LOGINS" not in env
    assert "sk-lab-secret" not in json.dumps(env)
    assert not set(secrets) & set(env) and not any(value in json.dumps(env) for value in secrets.values())
    assert env["CORS_ORIGINS"] == "http://x"
    assert os.path.samefile(seen["cwd"], workdir)


def test_a_service_that_exits_reports_its_exit_code(tmp_path, fake_script, supervisors):
    out = tmp_path / "crash.json"
    supervisor = supervisors([_service("modeling_agent", fake_script, out, "crash", cwd=tmp_path)], tmp_path / "logs")

    supervisor.start()

    status = _wait_for(lambda: not _status(supervisor, "modeling_agent")["running"]
                       and _status(supervisor, "modeling_agent"), "the exit")
    assert status["detail"] == "exited with code 3, see modeling_agent.log"


def test_a_missing_executable_is_reported_and_the_other_services_start(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    missing = PreviewService(name="frontend", argv=("no-such-tool-for-the-lab", "vite"), cwd=str(tmp_path), env={},
                             watch=(), port=8180)
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path), missing], tmp_path / "logs")

    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the fake service")

    assert _status(supervisor, "frontend") == {
        "name": "frontend", "running": False, "detail": "no-such-tool-for-the-lab not found"}
    assert _status(supervisor, "backend")["running"] is True


def test_restart_starts_a_new_process_and_an_unknown_name_raises(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path)], tmp_path / "logs")
    supervisor.start()
    first = _wait_for(lambda: _read_json(str(out)), "the first process")["pid"]
    os.remove(out)

    supervisor.restart("backend")

    second = _wait_for(lambda: _read_json(str(out)), "the second process")["pid"]
    assert second != first
    assert _status(supervisor, "backend")["running"] is True
    with pytest.raises(KeyError):
        supervisor.restart("nope")


def _touch_later(path) -> None:
    """Give ``path`` a modification time clearly after its current one (file systems round mtimes)."""
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


def test_restart_changed_restarts_only_services_whose_watched_python_changed(tmp_path, fake_script, supervisors):
    source = tmp_path / "besser"
    (source / "node_modules" / "pkg").mkdir(parents=True)
    (source / "module.py").write_text("x = 1\n", encoding="utf-8")
    (source / "README.md").write_text("docs\n", encoding="utf-8")
    (source / "node_modules" / "pkg" / "build.py").write_text("y = 1\n", encoding="utf-8")
    backend_out, frontend_out = tmp_path / "backend.json", tmp_path / "frontend.json"
    supervisor = supervisors([
        _service("backend", fake_script, backend_out, cwd=tmp_path, watch=[source]),
        _service("frontend", fake_script, frontend_out, cwd=tmp_path),
    ], tmp_path / "logs")
    supervisor.start()
    first = _wait_for(lambda: _read_json(str(backend_out)), "the backend")["pid"]
    frontend_pid = _wait_for(lambda: _read_json(str(frontend_out)), "the frontend")["pid"]

    assert supervisor.restart_changed() == []
    _touch_later(source / "README.md")
    _touch_later(source / "node_modules" / "pkg" / "build.py")
    assert supervisor.restart_changed() == []

    os.remove(backend_out)
    _touch_later(source / "module.py")
    assert supervisor.restart_changed() == ["backend"]
    assert _wait_for(lambda: _read_json(str(backend_out)), "the restarted backend")["pid"] != first
    assert supervisor.restart_changed() == []

    (source / "new_module.py").write_text("z = 1\n", encoding="utf-8")
    assert supervisor.restart_changed() == ["backend"]
    assert _read_json(str(frontend_out))["pid"] == frontend_pid


def test_restart_changed_brings_back_a_service_whose_fix_landed(tmp_path, supervisors):
    source = tmp_path / "agent"
    source.mkdir()
    app = source / "app.py"
    app.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    service = PreviewService(name="modeling_agent", argv=(sys.executable, str(app)), cwd=str(source), env={},
                             watch=(str(source),), port=8865)
    supervisor = supervisors([service], tmp_path / "logs")
    supervisor.start()
    _wait_for(lambda: not _status(supervisor, "modeling_agent")["running"], "the broken service to exit")

    app.write_text("import time\nwhile True:\n    time.sleep(0.05)\n", encoding="utf-8")
    _touch_later(app)

    assert supervisor.restart_changed() == ["modeling_agent"]
    time.sleep(0.5)
    assert _status(supervisor, "modeling_agent")["running"] is True


def test_restart_changed_does_nothing_before_start_or_after_stop(tmp_path, fake_script, supervisors):
    source = tmp_path / "src"
    source.mkdir()
    (source / "module.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "backend.json"
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path, watch=[source])],
                             tmp_path / "logs")

    _touch_later(source / "module.py")
    assert supervisor.restart_changed() == []
    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the fake service")
    supervisor.stop()
    _touch_later(source / "module.py")

    assert supervisor.restart_changed() == []
    assert _status(supervisor, "backend")["running"] is False


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX; on Windows only the child is stopped")
def test_stop_also_ends_the_children_of_a_service(tmp_path, fake_script, supervisors):
    out = tmp_path / "parent.json"
    supervisor = supervisors([_service("frontend", fake_script, out, "spawn", cwd=tmp_path)], tmp_path / "logs")
    supervisor.start()
    _wait_for(lambda: _read_json(str(out) + ".child"), "the grandchild")
    assert _ticking(str(out) + ".child")

    supervisor.stop()

    assert not _ticking(str(out))
    assert not _ticking(str(out) + ".child")


def _repos(root) -> dict[str, RepoSpec]:
    besser = os.path.join(str(root), "besser")
    return {
        REPO_BESSER: RepoSpec(REPO_BESSER, besser, "development", "BESSER-PEARL", "BESSER"),
        REPO_FRONTEND: RepoSpec(REPO_FRONTEND, os.path.join(besser, "besser", "utilities", "web_modeling_editor",
                                                            "frontend"),
                                "develop", "BESSER-PEARL", "BESSER-Web-Modeling-Editor"),
        REPO_MODELING_AGENT: RepoSpec(REPO_MODELING_AGENT, os.path.join(str(root), "modeling-agent"), "develop",
                                      "BESSER-PEARL", "modeling-agent"),
    }


def test_default_services_run_the_working_copy_on_localhost_ports(tmp_path, caplog):
    repos = _repos(tmp_path)

    with caplog.at_level(logging.WARNING):
        backend, frontend, agent = default_services(repos)

    besser, agent_repo = repos[REPO_BESSER].path, repos[REPO_MODELING_AGENT].path
    assert backend.argv == (sys.executable, "-m", "besser.utilities.web_modeling_editor.backend.backend",
                            "--host", "127.0.0.1", "--port", "9100")
    assert backend.cwd == besser
    assert backend.env["PYTHONPATH"] == besser
    assert backend.env["CORS_ORIGINS"] == "http://localhost:8180,http://127.0.0.1:8180"
    # The lab's own flags and build stamp would otherwise pass through safe_subprocess_env().
    assert backend.env["BESSER_LAB_ENABLED"] == "false"
    assert backend.env["BESSER_LAB_PREVIEW_ENABLED"] == "false"
    assert backend.env["BESSER_BUILD_SHA"] == ""
    assert backend.watch == (os.path.join(besser, "besser"),)
    assert backend.port == 9100

    assert frontend.argv == ("npx", "vite", "--host", "127.0.0.1", "--port", "8180", "--strictPort")
    assert frontend.cwd == os.path.join(repos[REPO_FRONTEND].path, "packages", "webapp")
    # A dev build of the editor reads VITE_DEV_BACKEND_URL, not BACKEND_URL (shared/constants/constant.ts).
    assert frontend.env == {"VITE_DEV_BACKEND_URL": "http://localhost:9100/besser_api",
                            "UML_BOT_WS_URL": "ws://localhost:8865"}
    assert frontend.watch == ()

    assert agent.argv == (sys.executable, "modeling_agent.py")
    assert agent.cwd == agent_repo
    assert agent.env["PYTHONPATH"] == os.path.join(agent_repo, "src")
    assert agent.watch == (os.path.join(agent_repo, "src"), os.path.join(agent_repo, "modeling_agent.py"))
    assert agent.port == 8865
    assert "runs with" in caplog.text


def _venv_python(checkout: str) -> str:
    python = (os.path.join(checkout, "venv", "Scripts", "python.exe") if os.name == "nt"
              else os.path.join(checkout, "venv", "bin", "python"))
    os.makedirs(os.path.dirname(python))
    open(python, "w").close()
    return python


def test_default_services_behind_a_public_url_and_with_the_checkouts_own_venvs(tmp_path):
    repos = _repos(tmp_path)
    agent_python = _venv_python(repos[REPO_MODELING_AGENT].path)
    besser_python = _venv_python(repos[REPO_BESSER].path)

    backend, frontend, agent = default_services(repos, backend_port=9200, frontend_port=8280, agent_port=8965,
                                                public_url="https://lab.besser-pearl.org/")

    assert backend.env["CORS_ORIGINS"] == "https://lab.besser-pearl.org"
    assert backend.argv[0] == besser_python
    assert backend.argv[-1] == "9200"
    assert frontend.env == {
        "VITE_DEV_BACKEND_URL": "https://lab.besser-pearl.org/besser_api",
        "UML_BOT_WS_URL": "wss://lab.besser-pearl.org/agent",
        "__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS": "lab.besser-pearl.org",
    }
    assert frontend.argv[5] == "8280"
    assert agent.argv == (agent_python, "modeling_agent.py")
    assert agent.port == 8965
    for service in (backend, frontend, agent):
        assert not any(word in name for name in service.env for word in ("KEY", "SECRET", "TOKEN"))
