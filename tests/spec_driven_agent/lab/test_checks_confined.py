"""Checks and dependency installs run in the lab's sandbox (D1 tests 6 and 7, R3, R23)."""

import logging
import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from besser.spec_driven_agent.execution import sandbox
from besser.spec_driven_agent.execution.sandbox import SandboxedCommand, SandboxUnavailable
from besser.spec_driven_agent.lab import checks
from besser.spec_driven_agent.lab.checks import confined_command_runner, confined_runner, run_checks
from besser.spec_driven_agent.lab.contracts import CheckSpec, RepoSpec
from besser.spec_driven_agent.lab.sandbox_policy import lab_sandbox_mounts


def _repo(key: str, path: Path, git_dir: Path) -> RepoSpec:
    return RepoSpec(key=key, path=str(path), base_branch="develop", github_owner="o", github_name=key,
                    git_dir=str(git_dir))


def _spec(spec_id: str, code: str = "print('1 passed')", network: bool = False) -> CheckSpec:
    return CheckSpec(id=spec_id, repo="besser", title=spec_id, argv=(sys.executable, "-c", code),
                     timeout_seconds=60, network=network)


@pytest.fixture
def change(tmp_path):
    return make_change(tmp_path)


def make_change(tmp_path: Path) -> dict:
    """One change's trees, its git root and the lab's mounts for it."""
    root = tmp_path / "work"
    trees = root / "changes" / "abcdef012345"
    besser, agent = trees / "besser", trees / "modeling-agent"
    git_root = tmp_path / "work-git"
    for folder in (besser, agent, git_root / "besser"):
        folder.mkdir(parents=True)
    repos = {"besser": _repo("besser", besser, git_root / "besser"),
             "modeling_agent": _repo("modeling_agent", agent, git_root / "modeling-agent")}
    mounts = lab_sandbox_mounts(repos, workspace_root=str(root), git_root=str(git_root), hidden=[])
    return {"root": root, "trees": trees, "besser": besser, "git_root": git_root, "repos": repos,
            "mounts": mounts}


@pytest.fixture
def planned(monkeypatch):
    """Record every sandbox plan; the plan runs the command as is (an unconfined platform)."""
    calls = []

    def plan(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return SandboxedCommand(list(command), False, "unconfined-platform")

    monkeypatch.setattr(checks, "sandboxed_command", plan)
    return calls


def test_checks_run_confined_without_network(change, planned):
    """D1 test 6: every check goes through the sandbox with the lab's mounts and no network."""
    mounts = change["mounts"]
    specs = [_spec("besser:tests"), _spec("besser:ruff"), _spec("besser:needs-network", network=True)]

    results = run_checks(specs, change["repos"], runner=confined_runner(mounts))

    assert [r.passed for r in results] == [True, True, True]
    assert [call["network"] for call in planned] == [False, False, True]
    for call in planned:
        assert (call["writable"], call["read_only"], call["masked"]) == (mounts.writable, mounts.read_only,
                                                                        mounts.masked)
        assert call["cwd"] == os.path.normpath(str(change["besser"]))
        # The shell's workspace too: one $HOME per change, beside its trees.
        assert call["workspace"] == os.path.realpath(str(change["trees"]))
        assert call["command"][0] == shutil.which(sys.executable)


def test_a_check_says_when_it_ran_without_the_sandbox(change, planned):
    [result] = run_checks([_spec("besser:tests")], change["repos"], runner=confined_runner(change["mounts"]))

    assert result.passed and result.summary == "1 passed"
    assert result.output_tail.startswith("Note: this check ran without the sandbox (no namespace sandbox on ")


def test_a_check_whose_sandbox_cannot_start_is_skipped_with_the_reason(change, monkeypatch, caplog):
    def unavailable(command, **kwargs):
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")

    monkeypatch.setattr(checks, "sandboxed_command", unavailable)
    with caplog.at_level(logging.WARNING, logger=checks.__name__):
        [result] = run_checks([_spec("besser:tests")], change["repos"], runner=confined_runner(change["mounts"]))

    assert result.skipped and not result.passed
    assert result.summary == "the check sandbox is unavailable: bubblewrap (bwrap) is not installed"
    assert any("the check sandbox is unavailable" in record.getMessage() for record in caplog.records)


def test_a_sandbox_that_fails_to_set_up_skips_the_check(change, monkeypatch):
    # bwrap reports its own setup failure as a "bwrap: ..." line and never runs the command.
    failing = [sys.executable, "-c", "import sys; sys.stderr.write('bwrap: Can\\'t mount tmpfs\\n'); sys.exit(1)"]
    monkeypatch.setattr(checks, "sandboxed_command",
                        lambda command, **kwargs: SandboxedCommand(failing, False, "bwrap"))

    [result] = run_checks([_spec("besser:tests")], change["repos"], runner=confined_runner(change["mounts"]))

    assert result.skipped
    assert result.summary == "the check sandbox is unavailable: bwrap: Can't mount tmpfs"


def test_a_check_that_exits_like_the_sandbox_is_still_a_failure(change, planned):
    # Only the runner's own message makes exit code 125 a skip; model-written code cannot turn a failure into one.
    [result] = run_checks([_spec("besser:tests", "import sys; print('1 failed'); sys.exit(125)")], change["repos"],
                          runner=confined_runner(change["mounts"]))

    assert not result.passed and not result.skipped


def test_installs_get_the_lab_mounts_and_a_private_network(change, planned):
    """R3: a change's dependency install runs in its sandbox with a network of its own."""
    run = confined_command_runner(change["mounts"])

    code, output = run([sys.executable, "-c", "print('installed')"], str(change["besser"]))

    assert code == 0 and "installed" in output
    [call] = planned
    assert call["private_network"] is True
    assert (call["writable"], call["read_only"], call["masked"]) == (change["mounts"].writable,
                                                                    change["mounts"].read_only,
                                                                    change["mounts"].masked)


def test_a_private_network_plan_waits_for_its_network_then_runs_the_command():
    command = ["/venv/bin/uv", "pip", "install", "--", "x"]
    plan_argv = ["/usr/bin/bwrap", "--unshare-net", "--chdir", "/w", "--", *command]

    argv = sandbox.network_wait_argv(plan_argv, command, info_fd=7)

    assert argv[:3] == ["/usr/bin/bwrap", "--info-fd", "7"]
    assert argv[3:7] == ["--unshare-net", "--chdir", "/w", "--"]
    wrapper = argv[7:-len(command)]
    assert wrapper[:4] == [sys.executable, "-I", "-S", "-c"] and "os.execvp" in wrapper[4]
    assert wrapper[5] == str(sandbox.NETWORK_WAIT_SECONDS)
    assert argv[-len(command):] == command


def test_an_install_tool_that_is_not_installed_is_named(change, planned):
    code, output = confined_command_runner(change["mounts"])(["besser-lab-no-such-tool"], str(change["besser"]))

    assert code == checks.MISSING_TOOL and "besser-lab-no-such-tool is not installed" in output
    assert planned == []


# --- On a real sandbox (Linux with a working bubblewrap) --------------------------------------------------


_PROBE = """
import errno, os, socket, sys
# From the command line, or (a shell command) from probe_args.txt beside the script.
git_config, state, session, port = sys.argv[1:5] or open("probe_args.txt").read().splitlines()[:4]
def failure(exc):
    return errno.errorcode.get(exc.errno, str(exc))
try:
    with open(git_config, "a") as handle:
        handle.write("[filter \\"x\\"]\\n")
    print("config: written")
except OSError as exc:
    print("config:", failure(exc))
try:
    print("state:", os.listdir(state))
except OSError as exc:
    print("state:", failure(exc))
try:
    open(session).read()
    print("session: read")
except OSError as exc:
    print("session:", failure(exc))
probe = socket.socket()
probe.settimeout(3)
try:
    probe.connect(("127.0.0.1", int(port)))
    print("connect: ok")
except OSError as exc:
    print("connect:", failure(exc))
print("1 passed")
"""


@pytest.fixture
def confined_layout(change):
    with lab_private_folders(change) as layout:
        yield layout


@contextmanager
def lab_private_folders(change: dict):
    """The change plus the lab's private folders, outside /tmp (the sandbox's /tmp is its own tmpfs anyway), the
    probe of D1 test 7 in its BESSER tree and a listener on the host's loopback."""
    config = change["git_root"] / "besser" / "config"
    config.write_text("[core]\n", encoding="utf-8")
    private = Path(tempfile.mkdtemp(prefix="besser-lab-test-", dir=Path.home()))
    state, sessions = private / "state", private / "sessions"
    (state / "changes").mkdir(parents=True)
    sessions.mkdir()
    (sessions / "oauth_sessions.json").write_text("{}", encoding="utf-8")
    mounts = lab_sandbox_mounts(change["repos"], workspace_root=str(change["root"]),
                                git_root=str(change["git_root"]), hidden=[str(state), str(sessions)])
    (change["besser"] / "probe.py").write_text(_PROBE, encoding="utf-8")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    args = (str(config), str(state), str(sessions / "oauth_sessions.json"), str(listener.getsockname()[1]))
    (change["besser"] / "probe_args.txt").write_text("\n".join(args), encoding="utf-8")
    try:
        yield {**change, "mounts": mounts, "config": config, "args": args}
    finally:
        listener.close()
        shutil.rmtree(private, ignore_errors=True)


def assert_confined(output: str, config: Path, *, host_ports: bool = True) -> None:
    """The probe's output: no git config written, no lab state, no session file and, with ``host_ports``, no
    host port reached."""
    assert "config: EROFS" in output, output
    assert config.read_text(encoding="utf-8") == "[core]\n"
    assert "state: []" in output or "state: ENOENT" in output, output
    assert "session: ENOENT" in output, output
    if host_ports:
        assert "connect: ok" not in output and "connect: " in output, output


@pytest.mark.skipif(sandbox.sandbox_selftest_error() is not None, reason="needs a working bubblewrap (Linux)")
def test_a_check_cannot_reach_git_dirs_lab_state_sessions_or_host_ports(confined_layout):
    """D1 test 7, check side."""
    spec = CheckSpec(id="besser:tests", repo="besser", title="probe",
                     argv=(sys.executable, "probe.py", *confined_layout["args"]), timeout_seconds=60)

    [result] = run_checks([spec], confined_layout["repos"], runner=confined_runner(confined_layout["mounts"]))

    assert_confined(result.output_tail, confined_layout["config"])


@pytest.mark.skipif(sandbox.sandbox_selftest_error() is not None, reason="needs a working bubblewrap (Linux)")
def test_an_install_cannot_reach_git_dirs_lab_state_sessions_or_host_ports(confined_layout):
    """R3: an install has outbound NAT but no host loopback."""
    if not sandbox._network_selftest(shutil.which("bwrap"))[0]:
        pytest.skip("needs pasta and /dev/net/tun")
    run = confined_command_runner(confined_layout["mounts"])

    code, output = run([sys.executable, "probe.py", *confined_layout["args"]], str(confined_layout["besser"]))

    assert code == 0, output
    assert_confined(output, confined_layout["config"])
