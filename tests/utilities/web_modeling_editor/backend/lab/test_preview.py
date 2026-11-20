"""The lab preview supervisor, driven with small Python scripts as fake services (no real server).

Most tests run the services through the platform's unconfined path (``sandboxed_command`` patched to say so), so
they run on any OS; the sandboxed launch is checked with its parts monkeypatched, and once for real on a Linux
host with bubblewrap and pasta.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import sys
import textwrap
import threading
import time

import pytest

from besser.spec_driven_agent.execution import sandbox
from besser.spec_driven_agent.execution.sandbox import SandboxedCommand, SandboxMounts, SandboxUnavailable
from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT, RepoSpec
from besser.utilities.web_modeling_editor.backend.services.lab import preview
from besser.utilities.web_modeling_editor.backend.services.lab.llm_proxy import PREVIEW_PROXY_PLACEHOLDER_KEY
from besser.utilities.web_modeling_editor.backend.services.lab.preview import (
    PreviewSandbox,
    PreviewService,
    PreviewSupervisor,
    default_services,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

WAIT_SECONDS = 30.0
# An Anthropic key's shape (sk-ant-api03-, 93 characters, AA); not a real key.
PLANTED_KEY = "sk-ant-api03-" + "Ab1Cd2Ef3Gh4" * 7 + "Xy1234567" + "AA"

# Writes {"pid", "env", "cwd"} to argv[1], appends a tick to argv[1] + ".ticks" while it lives.
# argv[2]: "serve" (default), "crash" (30 numbered lines and a key, then exits 3) or "spawn" (also starts a
# ticking grandchild).
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
        for number in range(30):
            print("line", number)
        print("Traceback: bad key " + sys.argv[3], flush=True)
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


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _status(supervisor: PreviewSupervisor, name: str):
    return next(entry for entry in supervisor.status() if entry.name == name)


@pytest.fixture
def fake_script(tmp_path):
    path = tmp_path / "fake_service.py"
    path.write_text(FAKE_SERVICE, encoding="utf-8")
    return str(path)


@pytest.fixture
def unconfined(monkeypatch):
    """Services run as on a platform without namespaces, whatever this host has."""
    monkeypatch.setattr(preview, "sandboxed_command",
                        lambda command, **kwargs: SandboxedCommand(list(command), False, "unconfined-platform"))


@pytest.fixture
def supervisors(tmp_path, unconfined):
    """Builds supervisors and always stops them, so no fake service outlives its test."""
    made: list[PreviewSupervisor] = []

    def make(services, log_dir):
        supervisor = PreviewSupervisor(services, log_dir=str(log_dir),
                                       sandbox=PreviewSandbox(root=str(tmp_path), mounts=SandboxMounts()))
        made.append(supervisor)
        return supervisor

    yield make
    for supervisor in made:
        supervisor.stop()


def _service(name, script, out, *args, cwd, env=None, watch=(), port=None):
    return PreviewService(name=name, argv=(sys.executable, script, str(out), *args), cwd=str(cwd),
                          env=env or {}, watch=tuple(str(path) for path in watch), port=port or _free_port())


def test_start_logs_reports_starting_and_stop_ends_the_process(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    port = _free_port()
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path, port=port)], tmp_path / "logs")

    before = _status(supervisor, "backend")
    assert (before.state, before.detail, before.startedAt, before.logTail) == ("stopped", "not started", None, [])
    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the fake service")

    # The process lives but nothing listens on its port.
    status = _status(supervisor, "backend")
    assert (status.state, status.detail) == ("starting", f"starting on port {port}")
    assert status.startedAt is not None
    assert _ticking(str(out))
    log = (tmp_path / "logs" / "backend.log").read_text(encoding="utf-8")
    assert "fake service up" in log
    assert supervisor.confined is False  # the unconfined platform path says so

    supervisor.stop()
    after = _status(supervisor, "backend")
    assert (after.state, after.detail, after.startedAt) == ("stopped", "stopped", None)
    assert not _ticking(str(out))


def test_running_once_its_port_accepts_a_connection(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    port = _free_port()
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path, port=port)], tmp_path / "logs")
    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the fake service")
    assert _status(supervisor, "backend").state == "starting"

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()
        status = _status(supervisor, "backend")

    assert (status.state, status.detail) == ("running", f"running on port {port}")
    # Once seen ready, a process stays running without probing again.
    assert _status(supervisor, "backend").state == "running"


def test_a_port_that_accepts_and_closes_at_once_is_not_ready():
    """pasta accepts on the host port and closes the connection while nothing listens inside the sandbox."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        def close_at_once():
            conn, _ = listener.accept()
            conn.close()

        closer = threading.Thread(target=close_at_once)
        closer.start()
        assert preview._accepts("127.0.0.1", port) is False
        closer.join(WAIT_SECONDS)
        # A server waits for the request: ready.
        assert preview._accepts("127.0.0.1", port) is True


def test_child_environment_is_the_safe_one_plus_the_service_entries(tmp_path, fake_script, supervisors,
                                                                       monkeypatch):
    monkeypatch.setenv("BESSER_LAB_LLM_API_KEY", "sk-lab-secret")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "oauth-secret")
    monkeypatch.setenv("GITHUB_CLIENT_ID", "oauth-client")
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
    assert "GITHUB_CLIENT_ID" not in env  # GitHub sign-in is off in the preview
    assert "BESSER_LAB_ALLOWED_LOGINS" not in env
    assert "sk-lab-secret" not in json.dumps(env)
    assert not set(secrets) & set(env) and not any(value in json.dumps(env) for value in secrets.values())
    assert env["CORS_ORIGINS"] == "http://x"
    assert os.path.samefile(seen["cwd"], workdir)


def test_a_service_that_exits_fails_with_the_redacted_end_of_its_log(tmp_path, fake_script, supervisors):
    out = tmp_path / "crash.json"
    supervisor = supervisors([_service("modeling_agent", fake_script, out, "crash", PLANTED_KEY, cwd=tmp_path)],
                             tmp_path / "logs")

    supervisor.start()

    status = _wait_for(lambda: _status(supervisor, "modeling_agent").state == "failed"
                       and _status(supervisor, "modeling_agent"), "the exit")
    assert status.detail == "exited with code 3"
    assert len(status.logTail) == preview.FAILED_LOG_TAIL_LINES
    assert status.logTail[-1] == "Traceback: bad key [REDACTED]"
    assert status.logTail[0] == "line 11"
    assert PLANTED_KEY not in json.dumps(status.model_dump())
    assert PLANTED_KEY in (tmp_path / "logs" / "modeling_agent.log").read_text(encoding="utf-8")


def test_log_tail_is_redacted_and_says_when_it_is_cut(tmp_path, fake_script, supervisors):
    out = tmp_path / "crash.json"
    supervisor = supervisors([_service("backend", fake_script, out, "crash", PLANTED_KEY, cwd=tmp_path)],
                             tmp_path / "logs")
    supervisor.start()
    _wait_for(lambda: _status(supervisor, "backend").state == "failed", "the exit")

    lines, truncated = supervisor.log_tail("backend", 2)
    assert lines == ["line 29", "Traceback: bad key [REDACTED]"] and truncated is True
    every, truncated = supervisor.log_tail("backend", 2000)
    assert every[0] == "fake service up" and len(every) == 32 and truncated is False
    with pytest.raises(KeyError):
        supervisor.log_tail("nope", 10)


def test_a_missing_executable_fails_and_the_other_services_start(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    missing = PreviewService(name="frontend", argv=("no-such-tool-for-the-lab", "vite"), cwd=str(tmp_path), env={},
                             watch=(), port=8180)
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path), missing], tmp_path / "logs")

    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the fake service")

    frontend = _status(supervisor, "frontend")
    assert (frontend.state, frontend.detail) == ("failed", "no-such-tool-for-the-lab not found")
    assert _status(supervisor, "backend").state == "starting"


def test_restart_starts_a_new_process_and_an_unknown_name_raises(tmp_path, fake_script, supervisors):
    out = tmp_path / "backend.json"
    supervisor = supervisors([_service("backend", fake_script, out, cwd=tmp_path)], tmp_path / "logs")
    supervisor.start()
    first = _wait_for(lambda: _read_json(str(out)), "the first process")["pid"]
    os.remove(out)

    supervisor.restart("backend")

    second = _wait_for(lambda: _read_json(str(out)), "the second process")["pid"]
    assert second != first
    assert _status(supervisor, "backend").state == "starting"
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


def test_restart_changed_watches_the_editors_lock_file(tmp_path, fake_script, supervisors):
    lock = tmp_path / "package-lock.json"
    lock.write_text("{}", encoding="utf-8")
    out = tmp_path / "frontend.json"
    supervisor = supervisors([_service("frontend", fake_script, out, cwd=tmp_path, watch=[lock])], tmp_path / "logs")
    supervisor.start()
    _wait_for(lambda: _read_json(str(out)), "the frontend")

    assert supervisor.restart_changed() == []
    _touch_later(lock)
    assert supervisor.restart_changed() == ["frontend"]


def test_restart_changed_brings_back_a_service_whose_fix_landed(tmp_path, supervisors):
    source = tmp_path / "agent"
    source.mkdir()
    app = source / "app.py"
    app.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    service = PreviewService(name="modeling_agent", argv=(sys.executable, str(app)), cwd=str(source), env={},
                             watch=(str(source),), port=_free_port())
    supervisor = supervisors([service], tmp_path / "logs")
    supervisor.start()
    _wait_for(lambda: _status(supervisor, "modeling_agent").state == "failed", "the broken service to exit")

    app.write_text("import time\nwhile True:\n    time.sleep(0.05)\n", encoding="utf-8")
    _touch_later(app)

    assert supervisor.restart_changed() == ["modeling_agent"]
    time.sleep(0.5)
    assert _status(supervisor, "modeling_agent").state == "starting"


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
    assert _status(supervisor, "backend").state == "stopped"


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


# -- The sandboxed launch, its parts monkeypatched --------------------------------------------------------------


class _FakeProcess:
    """A process that lives until it is stopped."""

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs, self.pid, self.code = list(argv), kwargs, 4242, None

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.code = -9
        return self.code

    def terminate(self):
        self.code = -15

    kill = terminate


@pytest.fixture
def sandboxed_launch(monkeypatch, tmp_path):
    """``sandboxed_command`` returns a bwrap plan with a private network; Popen, the pid, pasta and the spawner
    are recorded."""
    calls = {"plan": [], "popen": [], "pasta": [], "spawned": 0}

    def fake_sandboxed_command(command, **kwargs):
        calls["plan"].append(kwargs)
        return SandboxedCommand(["bwrap", "--unshare-net", "--chdir", kwargs["cwd"], "--", *command], False,
                                "bwrap", private_network=True)

    def fake_popen(argv, **kwargs):
        process = _FakeProcess(argv, **kwargs)
        calls["popen"].append(process)
        return process

    def fake_start_pasta(pid, stderr, *, tcp_in=(), tcp_out=()):
        calls["pasta"].append({"pid": pid, "tcp_in": tuple(tcp_in), "tcp_out": tuple(tcp_out)})
        return _FakeProcess(["pasta"])

    def fake_spawn(factory):
        calls["spawned"] += 1
        return factory()

    monkeypatch.setattr(preview, "sandboxed_command", fake_sandboxed_command)
    monkeypatch.setattr(preview.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(preview, "read_child_pid", lambda fd, timeout: 777)
    monkeypatch.setattr(preview, "start_pasta", fake_start_pasta)
    monkeypatch.setattr(preview, "spawn_long_lived", fake_spawn)
    monkeypatch.setattr(preview, "sandbox_supported_platform", lambda: True)
    # The fake pid must never reach os.killpg.
    monkeypatch.setattr(preview, "_signal_group", lambda process, *, kill: process.terminate())
    return calls


def _sandboxed_supervisor(tmp_path, port, *, tcp_out=(), bind="127.0.0.1"):
    mounts = SandboxMounts(writable=(str(tmp_path / "change" / "besser"),), read_only=(str(tmp_path / "git"),),
                           masked=(str(tmp_path / "work"),))
    service = PreviewService(name="modeling_agent", argv=(sys.executable, "modeling_agent.py"), cwd=str(tmp_path),
                             env={"PYTHONUNBUFFERED": "1"}, watch=(), port=port, tcp_out=tcp_out)
    return PreviewSupervisor([service], log_dir=str(tmp_path / "logs"),
                             sandbox=PreviewSandbox(root=str(tmp_path / "change"), mounts=mounts, bind=bind)), mounts


def test_launch_uses_a_private_network_and_forwards_only_the_services_ports(tmp_path, sandboxed_launch):
    port = _free_port()
    supervisor, mounts = _sandboxed_supervisor(tmp_path, port, tcp_out=(9300,))

    supervisor.start()

    (plan,) = sandboxed_launch["plan"]
    assert plan["private_network"] is True
    assert plan["workspace"] == str(tmp_path / "change") and plan["cwd"] == str(tmp_path)
    assert (plan["writable"], plan["read_only"], plan["masked"]) == (mounts.writable, mounts.read_only, mounts.masked)
    (process,) = sandboxed_launch["popen"]
    argv = process.argv
    assert argv[:3] == ["bwrap", "--info-fd", argv[2]] and int(argv[2]) in process.kwargs["pass_fds"]
    # The service waits for pasta's interface, then becomes the command.
    wait_at = argv.index("--") + 1
    assert argv[wait_at:wait_at + 4] == [sys.executable, "-I", "-S", "-c"]
    assert argv[-2:] == [shutil.which(sys.executable) or sys.executable, "modeling_agent.py"]
    assert sandboxed_launch["pasta"] == [{"pid": 777, "tcp_in": (("127.0.0.1", port),), "tcp_out": (9300,)}]
    # Both the sandbox and its pasta are forked from the long-lived spawner thread.
    assert sandboxed_launch["spawned"] == 2
    assert supervisor.confined is True
    assert _status(supervisor, "modeling_agent").state == "starting"

    supervisor.stop()
    assert process.code is not None and _status(supervisor, "modeling_agent").state == "stopped"


@pytest.mark.parametrize("private, host", [(True, "0.0.0.0"), (False, "127.0.0.1")])
def test_a_service_listens_on_every_address_only_inside_its_own_network_namespace(tmp_path, sandboxed_launch,
                                                                                  monkeypatch, private, host):
    """pasta delivers a forwarded port to the namespace's own address, not its loopback (a probe bound to 127.0.0.1
    inside was never reached through passt 2025_05), so inside it a service listens on every address; unconfined,
    in the host's network, only on loopback."""
    if not private:
        monkeypatch.setattr(preview, "sandboxed_command",
                            lambda command, **kwargs: SandboxedCommand(list(command), False, "unconfined-override"))
    service = PreviewService(name="backend", argv=(sys.executable, "-m", "backend", "--port", "9100"),
                             cwd=str(tmp_path), env={}, watch=(), port=9100, host_option="--host")
    supervisor = PreviewSupervisor([service], log_dir=str(tmp_path / "logs"), sandbox=PreviewSandbox(
        root=str(tmp_path / "change"), mounts=SandboxMounts(writable=())))

    supervisor.start()

    (process,) = sandboxed_launch["popen"]
    assert process.argv[-4:] == ["--port", "9100", "--host", host]
    assert len(sandboxed_launch["pasta"]) == (1 if private else 0)
    supervisor.stop()


def test_an_unavailable_sandbox_fails_the_service_and_never_runs_it_bare(tmp_path, sandboxed_launch, monkeypatch):
    def unavailable(command, **kwargs):
        raise SandboxUnavailable("a private network cannot be set up here (pasta (passt) is not installed)")

    monkeypatch.setattr(preview, "sandboxed_command", unavailable)
    supervisor, _ = _sandboxed_supervisor(tmp_path, _free_port())

    supervisor.start()

    status = _status(supervisor, "modeling_agent")
    assert status.state == "failed"
    assert status.detail == ("the preview sandbox is unavailable: a private network cannot be set up here "
                             "(pasta (passt) is not installed)")
    assert sandboxed_launch["popen"] == [] and sandboxed_launch["spawned"] == 0


def test_pasta_that_cannot_attach_fails_the_service_and_ends_the_sandbox(tmp_path, sandboxed_launch, monkeypatch):
    def no_pasta(pid, stderr, *, tcp_in=(), tcp_out=()):
        raise SandboxUnavailable("pasta (passt) is not installed")

    monkeypatch.setattr(preview, "start_pasta", no_pasta)
    supervisor, _ = _sandboxed_supervisor(tmp_path, _free_port())

    supervisor.start()

    status = _status(supervisor, "modeling_agent")
    assert (status.state, status.detail) == ("failed", "the preview sandbox is unavailable: pasta (passt) is not "
                                                       "installed")
    (process,) = sandboxed_launch["popen"]
    assert process.code is not None


def _private_network_works() -> bool:
    return sandbox.sandbox_selftest_error() is None and sandbox._network_selftest(shutil.which("bwrap"))[0]


# Listens on argv[1] inside its sandbox, on the address its ``--host`` option gives (argv[4:6]), then records whether
# it can reach the host listener on argv[2].
_NETWORK_PROBE = textwrap.dedent("""
    import socket, sys
    assert sys.argv[4] == "--host"
    server = socket.socket()
    server.bind((sys.argv[5], int(sys.argv[1])))
    server.listen()
    try:
        socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=3).close()
        reached = "reached"
    except OSError as exc:
        reached = "refused: " + type(exc).__name__
    with open(sys.argv[3], "w") as handle:
        handle.write(reached + " " + sys.argv[5])
    while True:
        conn, _ = server.accept()
        conn.sendall(b"hello")
        conn.close()
""")


@pytest.mark.skipif(sandbox.sandbox_selftest_error() is not None, reason="needs a working bubblewrap (Linux)")
def test_a_sandboxed_service_cannot_reach_host_loopback_but_its_port_is_forwarded(tmp_path):
    """D1 test 8, Linux part: pasta forwards the service's port in and nothing out."""
    if not _private_network_works():
        pytest.skip("needs pasta and /dev/net/tun")
    tree = tmp_path / "change" / "besser"
    tree.mkdir(parents=True)
    (tree / "probe.py").write_text(_NETWORK_PROBE, encoding="utf-8")
    port = _free_port()
    with socket.socket() as host_listener:
        host_listener.bind(("127.0.0.1", 0))
        host_listener.listen()
        service = PreviewService(name="backend",
                                 argv=(sys.executable, "probe.py", str(port), str(host_listener.getsockname()[1]),
                                       str(tree / "reach.txt")),
                                 cwd=str(tree), env={}, watch=(), port=port, host_option="--host")
        supervisor = PreviewSupervisor([service], log_dir=str(tmp_path / "logs"), sandbox=PreviewSandbox(
            root=str(tmp_path / "change"), mounts=SandboxMounts(writable=(str(tree),))))
        try:
            supervisor.start()
            _wait_for(lambda: (tree / "reach.txt").exists(), "the probe")
            reached = (tree / "reach.txt").read_text(encoding="utf-8")
            assert reached.startswith("refused") and reached.endswith(" 0.0.0.0")
            # pasta (passt 2025_05 here) delivers the forwarded port to the namespace's own address, so a service
            # listening on its loopback only would never be reached.
            _wait_for(lambda: _status(supervisor, "backend").state == "running", "the forwarded port")
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                assert conn.recv(5) == b"hello"
            assert supervisor.confined is True
        finally:
            supervisor.stop()


# -- The services of a change -----------------------------------------------------------------------------------


def _repos(root) -> dict[str, RepoSpec]:
    besser = os.path.join(str(root), "besser")
    git_root = str(root) + "-git"
    return {
        REPO_BESSER: RepoSpec(REPO_BESSER, besser, "development", "BESSER-PEARL", "BESSER",
                              git_dir=os.path.join(git_root, "besser")),
        REPO_FRONTEND: RepoSpec(REPO_FRONTEND, os.path.join(besser, "besser", "utilities", "web_modeling_editor",
                                                            "frontend"),
                                "develop", "BESSER-PEARL", "BESSER-Web-Modeling-Editor",
                                git_dir=os.path.join(git_root, "besser", "modules", "besser", "utilities",
                                                     "web_modeling_editor", "frontend")),
        REPO_MODELING_AGENT: RepoSpec(REPO_MODELING_AGENT, os.path.join(str(root), "modeling-agent"), "develop",
                                      "BESSER-PEARL", "modeling-agent",
                                      git_dir=os.path.join(git_root, "modeling-agent")),
    }


def test_default_services_run_the_working_copy_on_localhost_ports(tmp_path, caplog):
    repos = _repos(tmp_path)

    with caplog.at_level(logging.WARNING):
        backend, frontend, agent = default_services(repos)

    besser, agent_repo = repos[REPO_BESSER].path, repos[REPO_MODELING_AGENT].path
    assert backend.argv == (sys.executable, "-m", "besser.utilities.web_modeling_editor.backend.backend",
                            "--port", "9100")
    assert backend.host_option == "--host"  # the supervisor appends the listen address at launch
    assert backend.cwd == besser
    assert backend.env["PYTHONPATH"] == besser
    assert backend.env["CORS_ORIGINS"] == "http://localhost:8180,http://127.0.0.1:8180"
    assert backend.env["DEPLOYMENT_URL"] == "http://localhost:8180"
    # The lab's own flags and build stamp would otherwise pass through safe_subprocess_env().
    assert backend.env["BESSER_LAB_ENABLED"] == "false"
    assert backend.env["BESSER_LAB_PREVIEW_ENABLED"] == "false"
    assert backend.env["BESSER_BUILD_SHA"] == ""
    assert backend.watch == (os.path.join(besser, "besser"),)
    assert (backend.port, backend.tcp_out) == (9100, ())

    assert frontend.argv == ("npx", "vite", "--port", "8180", "--strictPort")
    assert frontend.host_option == "--host"
    assert frontend.cwd == os.path.join(repos[REPO_FRONTEND].path, "packages", "webapp")
    # A dev build of the editor reads VITE_DEV_BACKEND_URL, not BACKEND_URL (shared/constants/constant.ts).
    assert frontend.env == {"VITE_DEV_BACKEND_URL": "http://localhost:9100/besser_api",
                            "UML_BOT_WS_URL": "ws://localhost:8865"}
    assert frontend.watch == (os.path.join(repos[REPO_FRONTEND].path, "package-lock.json"),)
    assert (frontend.port, frontend.tcp_out) == (8180, ())

    assert agent.argv == (sys.executable, "modeling_agent.py")
    assert agent.cwd == agent_repo
    # The agent takes its port and allowed origins from the environment (the slot's, not a config.yaml).
    assert agent.env == {"PYTHONPATH": os.path.join(agent_repo, "src"), "PYTHONUNBUFFERED": "1",
                         "BESSER_BACKEND_URL": "", "BESSER_AGENT_WS_PORT": "8865",
                         "BESSER_AGENT_WS_ORIGIN": "http://localhost:8180",
                         "BESSER_AGENT_WS_ORIGIN_ALT": "http://127.0.0.1:8180"}
    assert agent.watch == (os.path.join(agent_repo, "src"), os.path.join(agent_repo, "modeling_agent.py"))
    assert (agent.port, agent.tcp_out) == (8865, ())
    assert "runs with" in caplog.text


def _venv_python(checkout: str) -> str:
    python = (os.path.join(checkout, "venv", "Scripts", "python.exe") if os.name == "nt"
              else os.path.join(checkout, "venv", "bin", "python"))
    os.makedirs(os.path.dirname(python))
    open(python, "w").close()
    return python


def test_default_services_behind_a_public_url_with_the_trees_venvs_and_the_llm_proxy(tmp_path):
    repos = _repos(tmp_path)
    agent_python = _venv_python(repos[REPO_MODELING_AGENT].path)
    besser_python = _venv_python(repos[REPO_BESSER].path)

    backend, frontend, agent = default_services(repos, backend_port=9110, frontend_port=8190, agent_port=8875,
                                                public_url="https://preview-2.lab.besser-pearl.org/",
                                                proxy_port=9300)

    # The preview talks only to its own origin.
    assert backend.env["CORS_ORIGINS"] == "https://preview-2.lab.besser-pearl.org"
    assert backend.env["DEPLOYMENT_URL"] == "https://preview-2.lab.besser-pearl.org"
    assert backend.argv[0] == besser_python
    assert backend.argv[-1] == "9110"
    assert frontend.env == {
        "VITE_DEV_BACKEND_URL": "https://preview-2.lab.besser-pearl.org/besser_api",
        "UML_BOT_WS_URL": "wss://preview-2.lab.besser-pearl.org/agent",
        "__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS": "preview-2.lab.besser-pearl.org",
    }
    assert frontend.argv[3] == "8190"
    assert agent.argv == (agent_python, "modeling_agent.py")
    assert agent.port == 8875
    assert agent.env["BESSER_AGENT_WS_PORT"] == "8875"
    # Blank would give the agent's default origin (a public editor), so both name the slot's origin.
    assert agent.env["BESSER_AGENT_WS_ORIGIN"] == agent.env["BESSER_AGENT_WS_ORIGIN_ALT"] == (
        "https://preview-2.lab.besser-pearl.org")
    # The agent reaches OpenAI only through the lab's proxy, with a placeholder key; pasta lets it reach that port.
    assert agent.env["OPENAI_BASE_URL"] == "http://127.0.0.1:9300/v1"
    assert agent.env["OPENAI_API_KEY"] == PREVIEW_PROXY_PLACEHOLDER_KEY
    assert agent.tcp_out == (9300,)
    for service in (backend, frontend, agent):
        assert "GITHUB_CLIENT_ID" not in service.env
        assert all(value in ("", PREVIEW_PROXY_PLACEHOLDER_KEY) for name, value in service.env.items()
                   if any(word in name for word in ("KEY", "SECRET", "TOKEN")))


# -- Whether the preview can run a tree's modeling agent ---------------------------------------------------------


def _agent_settings(agent_tree) -> str:
    return os.path.join(str(agent_tree), "src", "common", "settings.py")


def _write_agent_settings(agent_tree, text: str) -> str:
    path = _agent_settings(agent_tree)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# The agent's settings as of the lab's base: origins from the environment, the port only from config.yaml.
AGENT_SETTINGS_TODAY = 'ws_origin: str = Field("", validation_alias="BESSER_AGENT_WS_ORIGIN")\n'
AGENT_SETTINGS_WITH_PORT = AGENT_SETTINGS_TODAY + 'ws_port: int = Field(8765, validation_alias="BESSER_AGENT_WS_PORT")\n'


def test_an_agent_that_reads_its_port_only_from_config_yaml_is_not_run(tmp_path):
    tree = tmp_path / "modeling-agent"
    assert preview.agent_unavailable_reason(str(tree)).startswith("The assistant is not available in this preview")

    _write_agent_settings(tree, AGENT_SETTINGS_TODAY)
    reason = preview.agent_unavailable_reason(str(tree))
    assert reason.startswith("The assistant is not available in this preview: ")
    assert "config.yaml" in reason


def test_an_agent_whose_settings_declare_the_port_variable_is_run(tmp_path):
    tree = tmp_path / "modeling-agent"
    _write_agent_settings(tree, AGENT_SETTINGS_WITH_PORT)

    assert preview.agent_unavailable_reason(str(tree)) == ""


def test_the_agent_check_follows_no_link_and_reads_no_folder(tmp_path):
    """The tree is model-writable: a link out of it, or a folder in the file's place, counts as not declared."""
    outside = _write_agent_settings(tmp_path / "elsewhere", AGENT_SETTINGS_WITH_PORT)
    tree = tmp_path / "modeling-agent"
    os.makedirs(os.path.dirname(_agent_settings(tree)))
    try:
        os.symlink(outside, _agent_settings(tree))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available to this user")
    assert preview.agent_unavailable_reason(str(tree)) != ""

    os.remove(_agent_settings(tree))
    os.makedirs(_agent_settings(tree))
    assert preview.agent_unavailable_reason(str(tree)) != ""


def test_the_preview_uses_the_sandboxs_network_wait_helper():
    assert preview.network_wait_argv is sandbox.network_wait_argv
    assert not hasattr(preview, "_network_wait_argv")
