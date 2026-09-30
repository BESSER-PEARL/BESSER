"""Sandbox command construction, fail-closed refusal and environment scrub."""
import logging
import subprocess

import pytest

from besser.utilities.web_modeling_editor.agent_simulator import sandbox

SESSIONS_ROOT = "/tmp/sessions"
WORK_DIR = "/tmp/sessions/0f8fad5b-d9cb-469f-a165-70867728950e"
COMMAND = ["/opt/venv/bin/python", "agent.py"]

# What `ls /` looks like in the simulator image.
FAKE_ROOT = [
    ("/app", None),
    ("/bin", "usr/bin"),
    ("/dev", None),
    ("/etc", None),
    ("/lib", "usr/lib"),
    ("/opt", None),
    ("/proc", None),
    ("/root", None),
    ("/run", None),
    ("/sys", None),
    ("/tmp", None),
    ("/usr", None),
    ("/var", None),
]


@pytest.fixture
def linux_with_bwrap(monkeypatch):
    """Pretend to be a Linux host where bwrap is installed and works."""
    monkeypatch.setattr(sandbox, "sandbox_supported_platform", lambda: True)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    monkeypatch.setattr(sandbox, "_selftest", lambda bwrap, uid: (True, ""))
    monkeypatch.setattr(sandbox, "_root_entries", lambda: list(FAKE_ROOT))
    monkeypatch.setattr(sandbox, "_resolve", lambda path: path)
    monkeypatch.delenv(sandbox.SANDBOX_POLICY_ENV, raising=False)


def _pairs(argv, flag):
    """Every (a, b) following ``flag`` in argv, e.g. --bind SRC DEST."""
    return [(argv[i + 1], argv[i + 2]) for i, arg in enumerate(argv) if arg == flag]


class TestSandboxedArgv:
    def test_wraps_command_in_bwrap_with_every_namespace(self, linux_with_bwrap):
        argv = sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)

        assert argv[0] == "/usr/bin/bwrap"
        for flag in ("--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                     "--unshare-cgroup-try", "--die-with-parent", "--new-session"):
            assert flag in argv
        # Network namespace is deliberately shared (LLM APIs + relay to 127.0.0.1).
        assert "--unshare-net" not in argv
        assert "--unshare-all" not in argv
        separator = argv.index("--")
        assert argv[separator + 1:] == COMMAND
        assert argv[argv.index("--chdir") + 1] == WORK_DIR

    def test_only_the_session_work_dir_is_writable(self, linux_with_bwrap):
        argv = sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)

        assert _pairs(argv, "--bind") == [(WORK_DIR, WORK_DIR)]
        read_only = {src for src, _ in _pairs(argv, "--ro-bind")}
        assert read_only == {"/app", "/etc", "/opt", "/root", "/sys", "/usr", "/var"}

    def test_sessions_root_top_level_and_virtual_dirs_are_never_bound(self, linux_with_bwrap):
        argv = sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)

        bound = {src for src, _ in _pairs(argv, "--ro-bind")}
        assert not bound & {"/tmp", "/proc", "/dev", "/run", SESSIONS_ROOT}
        # Private /proc and /dev, fresh tmpfs scratch space.
        assert argv[argv.index("--proc") + 1] == "/proc"
        assert argv[argv.index("--dev") + 1] == "/dev"
        tmpfs = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--tmpfs"]
        assert set(tmpfs) == {"/tmp", "/run", "/var/tmp", "/dev/shm"}
        # Merged-/usr symlinks are recreated, not bound.
        assert ("usr/bin", "/bin") in _pairs(argv, "--symlink")

    def test_sessions_root_elsewhere_hides_its_top_level(self, linux_with_bwrap):
        argv = sandbox.sandboxed_argv(
            COMMAND, work_dir="/var/sessions/abc", sessions_root="/var/sessions", uid=20000,
        )
        assert "/var" not in {src for src, _ in _pairs(argv, "--ro-bind")}
        assert _pairs(argv, "--bind") == [("/var/sessions/abc", "/var/sessions/abc")]

    def test_work_dir_outside_sessions_root_is_refused(self, linux_with_bwrap):
        with pytest.raises(sandbox.SandboxUnavailable, match="outside the sessions root"):
            sandbox.sandboxed_argv(COMMAND, work_dir="/app", sessions_root=SESSIONS_ROOT, uid=20000)


class TestFailClosed:
    def test_non_linux_platform_is_refused(self, monkeypatch):
        monkeypatch.setattr(sandbox, "sandbox_supported_platform", lambda: False)
        monkeypatch.setenv(sandbox.SANDBOX_POLICY_ENV, "off")  # not even the opt-out
        with pytest.raises(sandbox.SandboxUnavailable, match="only runs on Linux"):
            sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)

    def test_missing_bwrap_is_refused(self, linux_with_bwrap, monkeypatch):
        monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
        with pytest.raises(sandbox.SandboxUnavailable, match="bubblewrap"):
            sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)

    def test_failing_selftest_is_refused_with_the_docker_hint(self, linux_with_bwrap, monkeypatch):
        monkeypatch.setattr(
            sandbox, "_selftest", lambda bwrap, uid: (False, "bwrap: No permissions to create new namespace"),
        )
        with pytest.raises(sandbox.SandboxUnavailable, match="seccomp=unconfined"):
            sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)

    def test_explicit_opt_out_runs_unconfined_and_warns(self, linux_with_bwrap, monkeypatch, caplog):
        monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
        monkeypatch.setattr(sandbox, "_unconfined_warned", False)
        monkeypatch.setenv(sandbox.SANDBOX_POLICY_ENV, "off")
        with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
            argv = sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)
        assert argv == COMMAND
        assert "UNSANDBOXED" in caplog.text

    def test_unknown_policy_value_keeps_the_sandbox_mandatory(self, linux_with_bwrap, monkeypatch):
        monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
        monkeypatch.setenv(sandbox.SANDBOX_POLICY_ENV, "disabled")
        with pytest.raises(sandbox.SandboxUnavailable):
            sandbox.sandboxed_argv(COMMAND, work_dir=WORK_DIR, sessions_root=SESSIONS_ROOT, uid=20000)


class TestSelftest:
    def test_probe_runs_as_the_session_uid_and_is_cached(self, monkeypatch):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(sandbox, "_selftest_cache", {})
        monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

        assert sandbox._selftest("/usr/bin/bwrap", 20000) == (True, "")
        assert sandbox._selftest("/usr/bin/bwrap", 20000) == (True, "")
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0] == "/usr/bin/bwrap" and "--unshare-user" in argv and argv[-1] == "/bin/true"
        assert kwargs["user"] == 20000 and kwargs["group"] == 20000 and kwargs["extra_groups"] == []

    def test_probe_failure_reports_bwrap_last_line(self, monkeypatch):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, "", "noise\nbwrap: setting up uid map: Permission denied\n")

        monkeypatch.setattr(sandbox, "_selftest_cache", {})
        monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
        assert sandbox._selftest("/usr/bin/bwrap", 20000) == (False, "bwrap: setting up uid map: Permission denied")

    def test_probe_that_cannot_execute_is_a_failure(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise PermissionError("not permitted to switch uid")

        monkeypatch.setattr(sandbox, "_selftest_cache", {})
        monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
        ok, detail = sandbox._selftest("/usr/bin/bwrap", 20000)
        assert not ok and "could not be executed" in detail


class TestSafeSubprocessEnv:
    def test_only_allowlisted_non_secret_variables_survive(self, monkeypatch):
        for name in list(sandbox.os.environ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PATH", "/opt/venv/bin:/usr/bin")
        monkeypatch.setenv("LANG", "C.UTF-8")
        monkeypatch.setenv("AGENT_SIMULATOR_API_TOKEN", "server-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "operator-key")
        monkeypatch.setenv("SMTP_PASSWORD", "hunter2")
        monkeypatch.setenv("GITHUB_CLIENT_SECRET", "gh")
        monkeypatch.setenv("PYTHONPATH", "/app")

        env = sandbox.safe_subprocess_env({"HOME": "/tmp/sessions/x", "OPENAI_API_KEY": "user-key"})

        assert env == {
            "PATH": "/opt/venv/bin:/usr/bin",
            "LANG": "C.UTF-8",
            "HOME": "/tmp/sessions/x",
            # The per-session key the user supplied, never the operator's.
            "OPENAI_API_KEY": "user-key",
        }

    def test_path_defaults_when_absent(self, monkeypatch):
        for name in list(sandbox.os.environ):
            monkeypatch.delenv(name, raising=False)
        assert sandbox.safe_subprocess_env({})["PATH"] == sandbox.os.defpath
