"""A shell session's private network: pasta's argv, mode resolution, fail-closed.

Verified live on experimental: sessions shared the worker's network namespace,
so ``curl http://127.0.0.1:9000/besser_api/`` from a command answered 200 (the
worker's own API) and two runs could not both bind :8000. Each session now gets
its own namespace with pasta attached. These tests fake bwrap and pasta; the
real ones run in ``test_shell_session_private_net.py``.
"""

import logging

import pytest

from besser.spec_driven_agent.agent.tools import get_tools_for
from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.sandbox import (
    SANDBOX_POLICY_ENV,
    SHELL_NETWORK_ENV,
    SandboxUnavailable,
    SandboxedCommand,
    pasta_argv,
    run_confined,
    sandboxed_command,
)
from besser.spec_driven_agent.execution.shell_session import ShellSession


# --------------------------------------------------------------------------- #
# pasta's argv: nothing forwarded, in either direction
# --------------------------------------------------------------------------- #
def _option_values(argv: list[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for i, arg in enumerate(argv):
        if arg in {"-t", "-u", "-T", "-U", "--runas", "--dns-forward", "--dns-host"}:
            values.setdefault(arg, []).append(argv[i + 1])
    return values


def test_pasta_forwards_nothing_and_maps_no_gateway():
    argv = pasta_argv("/usr/bin/pasta", 4242)
    values = _option_values(argv)

    # -T/-U none: a dropped one re-exposes the worker's loopback to the sandbox.
    for flag in ("-t", "-u", "-T", "-U"):
        assert values[flag] == ["none"], flag
    assert "--no-map-gw" in argv, "the gateway address would reach the worker"
    for flag in ("-f", "--config-net"):
        assert flag in argv
    assert values["--runas"] == ["0:0"]
    assert argv[0] == "/usr/bin/pasta" and argv[-1] == "4242"
    long_forwards = {"--tcp-ports", "--udp-ports", "--tcp-ns", "--udp-ns", "--map-gw"}
    assert not long_forwards & set(argv)


def test_pasta_forwards_dns_only_when_asked():
    assert "--dns-forward" not in pasta_argv("pasta", 1)
    argv = pasta_argv("pasta", 7, dns_host="127.0.0.11")
    values = _option_values(argv)
    assert values["--dns-forward"] == ["169.254.1.53"]
    assert values["--dns-host"] == ["127.0.0.11"]
    assert argv[-1] == "7"
    for flag in ("-t", "-u", "-T", "-U"):
        assert values[flag] == ["none"], flag


# --------------------------------------------------------------------------- #
# Mode resolution
# --------------------------------------------------------------------------- #
@pytest.fixture
def selftest(monkeypatch):
    """Set the network selftest's answer; returns the calls made."""
    class _Calls(list):
        result = (True, "")

    calls = _Calls()

    def _fake(bwrap):
        calls.append(bwrap)
        return calls.result

    monkeypatch.setattr(sandbox_mod, "_network_selftest", _fake)
    monkeypatch.setattr(sandbox_mod, "_shared_network_warned", False)
    return calls


def test_shared_never_runs_the_selftest(monkeypatch, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "shared")
    assert sandbox_mod._shell_network_private("/usr/bin/bwrap") is False
    assert selftest == []


def test_auto_is_private_when_the_selftest_passes(monkeypatch, selftest):
    monkeypatch.delenv(SHELL_NETWORK_ENV, raising=False)
    assert sandbox_mod._shell_network_private("/usr/bin/bwrap") is True


def test_auto_falls_back_to_shared_with_one_warning(monkeypatch, selftest, caplog):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "auto")
    selftest.result = (False, "pasta (passt) is not installed")
    with caplog.at_level(logging.WARNING, logger=sandbox_mod.logger.name):
        assert sandbox_mod._shell_network_private("/usr/bin/bwrap") is False
        assert sandbox_mod._shell_network_private("/usr/bin/bwrap") is False
    warnings = [r for r in caplog.records if "share the worker's network" in r.getMessage()]
    assert len(warnings) == 1


def test_private_fails_closed_when_unavailable(monkeypatch, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "private")
    selftest.result = (False, "/dev/net/tun is not available")
    with pytest.raises(SandboxUnavailable) as raised:
        sandbox_mod._shell_network_private("/usr/bin/bwrap")
    assert "/dev/net/tun" in raised.value.detail
    assert "/dev/net/tun" not in str(raised.value), "the model sees the generic refusal"


def test_an_unknown_mode_is_auto(monkeypatch, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "sometimes")
    selftest.result = (False, "no pasta")
    assert sandbox_mod._shell_network_private("/usr/bin/bwrap") is False
    assert selftest, "auto runs the selftest"


# --------------------------------------------------------------------------- #
# The bwrap plan
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_platform(monkeypatch, tmp_path):
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: True)
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sandbox_mod, "_selftest", lambda _b: (True, ""))
    monkeypatch.setattr(sandbox_mod, "_mount_args", lambda *_a, **_k: ["--ro-bind", "/", "/"])
    monkeypatch.setattr(sandbox_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sandbox_mod, "_loopback_nameserver", lambda: None)
    return tmp_path


def _plan(workspace, **kwargs):
    return sandboxed_command("true", workspace=str(workspace), cwd=str(workspace), **kwargs)


def test_a_private_session_unshares_the_network(monkeypatch, fake_platform, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "private")
    plan = _plan(fake_platform, shell_network=True)
    assert plan.private_network is True
    assert "--unshare-net" in plan.argv[:plan.argv.index("--")]
    assert "/etc/resolv.conf" not in plan.argv, "a reachable nameserver needs no override"


def test_a_loopback_nameserver_gets_pastas_forwarder(monkeypatch, fake_platform, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "private")
    monkeypatch.setattr(sandbox_mod, "_loopback_nameserver", lambda: "127.0.0.11")
    argv = _plan(fake_platform, shell_network=True).argv
    at = argv.index("/etc/resolv.conf")
    assert argv[at - 2] == "--ro-bind"
    with open(argv[at - 1]) as handle:
        assert handle.read().split() == ["nameserver", "169.254.1.53"]


def test_shared_mode_is_the_previous_argv(monkeypatch, fake_platform, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "shared")
    plan = _plan(fake_platform, shell_network=True)
    assert plan.private_network is False
    assert plan.argv == _plan(fake_platform).argv
    assert "--unshare-net" not in plan.argv


def test_validators_and_one_shot_commands_ignore_the_shell_network(
        monkeypatch, fake_platform, selftest):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "private")
    validator = _plan(fake_platform, network=False)
    assert "--unshare-net" in validator.argv and validator.private_network is False
    one_shot = _plan(fake_platform, network=True)
    assert "--unshare-net" not in one_shot.argv and one_shot.private_network is False
    assert selftest == []

    ran = []
    monkeypatch.setattr(sandbox_mod, "run_bounded", lambda argv, **_k: ran.append(argv)
                        or __import__("subprocess").CompletedProcess(argv, 0, "", ""))
    run_confined(["true"], workspace=str(fake_platform), cwd=str(fake_platform), timeout=5)
    assert "--unshare-net" in ran[0], "validators keep no network at all"


# --------------------------------------------------------------------------- #
# Session start: fail closed, never a shared-network session
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.killed = self.terminated = False
        self.waits = 0
        self.stdin = self.stdout = _Stream()
        self.pid = 1

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True

    def terminate(self):
        self.terminated = True


class _Stream:
    def close(self):
        pass


@pytest.fixture
def fake_spawn(monkeypatch):
    """bwrap's Popen and pasta replaced; records what was started."""
    seen = {"bwrap": [], "pasta": []}

    def _popen(argv, **kwargs):
        proc = _FakeProc()
        seen["bwrap"].append((argv, kwargs, proc))
        return proc

    def _pasta(pid, _stderr):
        proc = _FakeProc(seen.get("pasta_exit"))
        seen["pasta"].append((pid, proc))
        return proc

    monkeypatch.setattr(shell_session.subprocess, "Popen", _popen)
    monkeypatch.setattr(shell_session, "start_pasta", _pasta)
    monkeypatch.setattr(shell_session, "read_child_pid", lambda fd, timeout: 4242)
    return seen


def _hello(monkeypatch, reply):
    monkeypatch.setattr(shell_session._SessionProcess, "_exchange",
                        lambda self, message, timeout: reply)


def test_a_private_session_hands_bwrap_an_info_fd_and_attaches_pasta(monkeypatch, fake_spawn):
    _hello(monkeypatch, {"ready": True, "net": True})
    proc = shell_session._SessionProcess(["bwrap", "--unshare-net", "--", "sup"], ".",
                                         private_network=True)
    argv, kwargs, _ = fake_spawn["bwrap"][0]
    assert argv[1] == "--info-fd" and kwargs["pass_fds"] == (int(argv[2]),)
    assert argv[-2:] == ["sup", str(sandbox_mod.NETWORK_WAIT_SECONDS)]
    assert fake_spawn["pasta"][0][0] == 4242
    assert proc.network_alive()

    proc.kill()
    assert fake_spawn["bwrap"][0][2].killed and fake_spawn["pasta"][0][1].killed
    proc.close()
    assert fake_spawn["pasta"][0][1].waits >= 1, "pasta is reaped, never left a zombie"


def test_a_shared_session_has_no_pasta_and_no_info_fd(monkeypatch, fake_spawn):
    _hello(monkeypatch, {"ready": True, "net": None})
    proc = shell_session._SessionProcess(["bwrap", "--", "sup"], ".")
    argv, kwargs, _ = fake_spawn["bwrap"][0]
    assert argv == ["bwrap", "--", "sup"] and kwargs["pass_fds"] == ()
    assert fake_spawn["pasta"] == [] and proc.pasta is None
    proc.close()


def test_an_attach_failure_kills_bwrap_and_refuses(monkeypatch, fake_spawn):
    def _no_pid(fd, timeout):
        raise SandboxUnavailable("bwrap exited before reporting its sandbox pid")

    monkeypatch.setattr(shell_session, "read_child_pid", _no_pid)
    _hello(monkeypatch, {"ready": True, "net": True})
    with pytest.raises(SandboxUnavailable):
        shell_session._SessionProcess(["bwrap", "--"], ".", private_network=True)
    bwrap = fake_spawn["bwrap"][0][2]
    assert bwrap.waits >= 1 and fake_spawn["pasta"] == []


@pytest.mark.parametrize("pasta_exit,net", [(1, True), (None, False)],
                         ids=["pasta exited early", "no interface came up"])
def test_a_session_without_its_network_is_refused(monkeypatch, fake_spawn, pasta_exit, net):
    fake_spawn["pasta_exit"] = pasta_exit
    _hello(monkeypatch, {"ready": True, "net": net})
    with pytest.raises(SandboxUnavailable):
        shell_session._SessionProcess(["bwrap", "--"], ".", private_network=True)
    assert fake_spawn["bwrap"][0][2].waits >= 1
    assert fake_spawn["pasta"][0][1].waits >= 1


def test_a_private_refusal_reaches_the_model_as_the_generic_refusal(tmp_path, monkeypatch):
    from besser.spec_driven_agent.agent import tool_executor as te

    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda *_a, **_k: SandboxedCommand(["bwrap"], False, "bwrap",
                                                           private_network=True))

    def _refuse(*_a, **_k):
        raise SandboxUnavailable("the shell session's private network did not come up")

    monkeypatch.setattr(shell_session, "_SessionProcess", _refuse)
    ran = []
    monkeypatch.setattr(shell_session, "run_bounded", lambda *a, **k: ran.append(a))
    result = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)._run_command(
        {"command": "echo X"})
    assert result["error"] == te._SANDBOX_REFUSAL
    assert ran == [], "never retried on the shared network"


# --------------------------------------------------------------------------- #
# pasta dying mid-session restarts the session, and the model is told
# --------------------------------------------------------------------------- #
class _FakeSession:
    instances: list = []

    def __init__(self, argv, cwd, private_network=False):
        self.private_network = private_network
        self.net_ok = True
        self.closed = False
        _FakeSession.instances.append(self)

    def alive(self):
        return True

    def network_alive(self):
        return self.net_ok

    def run(self, command, cwd, env, timeout):
        return {"id": 1, "exit_code": 0, "stdout": "", "stderr": "", "cwd": cwd, "env": env}

    def close(self):
        self.closed = True

    def kill(self):
        pass


def test_a_dead_pasta_restarts_the_session_with_a_note(tmp_path, monkeypatch):
    _FakeSession.instances = []
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda *_a, **_k: SandboxedCommand(["bwrap"], False, "bwrap",
                                                           private_network=True))
    monkeypatch.setattr(shell_session, "_SessionProcess", _FakeSession)
    session = ShellSession(str(tmp_path))
    try:
        assert session.run("true", working_dir=None, timeout=5).notes == []
        first = _FakeSession.instances[0]
        assert first.private_network is True
        first.net_ok = False

        result = session.run("true", working_dir=None, timeout=5)

        assert len(_FakeSession.instances) == 2 and first.closed
        assert _FakeSession.instances[1].private_network is True
        assert any("lost its network" in note for note in result.notes), result.notes
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# What the model is told
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("private,fact", [(True, "localhost is this run's alone"),
                                          (False, "shares localhost")])
def test_run_commands_text_states_the_network_it_gets(monkeypatch, private, fact):
    monkeypatch.setattr(sandbox_mod, "shell_network_is_private", lambda: private)
    tools = {t["name"]: t for t in get_tools_for(has_domain_model=True, allow_shell=True)}
    assert fact in tools["run_command"]["description"]
    assert tools["run_command"]["description"].count("localhost") == 1


def test_the_shared_tool_definition_is_not_mutated(monkeypatch):
    from besser.spec_driven_agent.agent.tools import EXECUTION_TOOLS

    original = next(t for t in EXECUTION_TOOLS if t["name"] == "run_command")["description"]
    monkeypatch.setattr(sandbox_mod, "shell_network_is_private", lambda: True)
    get_tools_for(has_domain_model=True, allow_shell=True)
    get_tools_for(has_domain_model=True, allow_shell=True)
    assert next(t for t in EXECUTION_TOOLS if t["name"] == "run_command")["description"] \
        == original


def test_no_network_selftest_off_linux(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: False)
    monkeypatch.setattr(sandbox_mod, "_network_selftest",
                        lambda _b: pytest.fail("selftest ran without a sandbox"))
    assert sandbox_mod.shell_network_is_private() is False
