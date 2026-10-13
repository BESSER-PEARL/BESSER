"""Session manager: slot pools, process launch contract, cleanup, file listing."""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid

import pytest
import yaml

from besser.utilities.web_modeling_editor.agent_simulator import session_manager as sm
from besser.utilities.web_modeling_editor.agent_simulator.sandbox import SandboxUnavailable, sandbox_selftest_error

IS_POSIX = os.name == "posix"


class FakeProcess:
    """Stand-in for the Popen of a running agent."""

    _next_pid = 40000

    def __init__(self, argv, **kwargs):
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.argv = argv
        self.kwargs = kwargs
        self.returncode = None
        self.stdout = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A SessionManager rooted in tmp_path whose launches are captured, not run."""
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(sm, "SESSIONS_ROOT", str(sessions_root))
    monkeypatch.setattr(sm, "MAX_SESSIONS", 2)
    monkeypatch.setattr(sm, "PORT_POOL", [7700, 7701])
    monkeypatch.setattr(sm, "UID_POOL", [20000, 20001])
    launched = []

    def fake_popen(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(sm, "sandboxed_argv", lambda command, **kwargs: ["bwrap", "--", *command])
    monkeypatch.setattr(sm.subprocess, "Popen", fake_popen)
    # Root-only calls, recorded instead of performed (and absent on Windows).
    monkeypatch.setattr(sm.os, "geteuid", lambda: 0, raising=False)
    chowned = []
    monkeypatch.setattr(sm.os, "chown", lambda path, uid, gid, **kw: chowned.append((path, uid)), raising=False)
    signals = []
    monkeypatch.setattr(sm.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    swept = []
    monkeypatch.setattr(sm, "_kill_uid_processes", lambda uid: swept.append(uid) or 0)

    manager = sm.SessionManager()
    manager.launched = launched
    manager.chowned = chowned
    manager.signals = signals
    manager.swept = swept
    return manager


def _create(manager, session_id=None, **overrides):
    kwargs = dict(
        agent_code="print('hi')",
        config_yaml="platforms:\n  websocket:\n    host: 0.0.0.0\n    port: 8765\n",
        env_vars={},
        event_list=[],
    )
    kwargs.update(overrides)
    return manager.create_session(session_id or str(uuid.uuid4()), **kwargs)


class TestCreateSession:
    def test_launch_contract(self, manager, monkeypatch):
        monkeypatch.setenv("AGENT_SIMULATOR_API_TOKEN", "server-secret")
        session = _create(manager, env_vars={"OPENAI_API_KEY": "user-key", "SMTP_PASSWORD": "nope"})

        process = manager.launched[0]
        assert process.argv == ["bwrap", "--", sys.executable, "agent.py"]
        kwargs = process.kwargs
        # Own session / process group so terminate can kill the whole group.
        assert kwargs["start_new_session"] is True
        # UID switch done by Popen itself, with setgroups([]) first.
        assert kwargs["user"] == session.uid == 20000
        assert kwargs["group"] == 20000
        assert kwargs["extra_groups"] == []
        assert kwargs["umask"] == 0o077
        # Limits applied in the child; a failure there makes Popen raise.
        assert kwargs["preexec_fn"] is sm._apply_rlimits
        assert kwargs["cwd"] == session.work_dir
        env = kwargs["env"]
        assert "AGENT_SIMULATOR_API_TOKEN" not in env
        assert "SMTP_PASSWORD" not in env
        assert env["OPENAI_API_KEY"] == "user-key"
        assert env["BESSER_WS_PORT"] == str(session.port)
        assert env["HOME"] == session.work_dir

    def test_config_pins_loopback_websocket_and_user_credentials(self, manager):
        session = _create(manager, env_vars={"OPENAI_API_KEY": "user-key"})
        with open(os.path.join(session.work_dir, "config.yaml"), encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["platforms"]["websocket"] == {"host": "127.0.0.1", "port": session.port}
        assert config["nlp"]["openai"]["api_key"] == "user-key"

    def test_work_dir_is_handed_to_the_session_uid(self, manager):
        session = _create(manager, support_files={"tools.py": "x = 1"}, workspace_paths=["data/in"])
        owned = {os.path.relpath(path, session.work_dir) for path, uid in manager.chowned if uid == session.uid}
        assert {".", "agent.py", "config.yaml", "tools.py", "tmp", "data", os.path.join("data", "in")} <= owned
        if IS_POSIX:
            assert os.stat(session.work_dir).st_mode & 0o777 == 0o700
            assert os.stat(os.path.join(session.work_dir, "agent.py")).st_mode & 0o777 == 0o600

    def test_sandbox_unavailable_refuses_before_writing_anything(self, manager, monkeypatch):
        def refuse(command, **kwargs):
            raise SandboxUnavailable("bubblewrap (bwrap) is not installed")

        monkeypatch.setattr(sm, "sandboxed_argv", refuse)
        with pytest.raises(SandboxUnavailable):
            _create(manager)
        assert manager.launched == []
        assert os.listdir(sm.SESSIONS_ROOT) == []
        assert manager.get_session_count() == 0
        # The slot went back to the pool.
        assert manager._used_ports == set() and manager._used_uids == set()

    def test_not_root_is_refused(self, manager, monkeypatch):
        monkeypatch.setattr(sm.os, "geteuid", lambda: 1000, raising=False)
        with pytest.raises(SandboxUnavailable, match="root"):
            _create(manager)
        assert manager.launched == []

    def test_path_traversal_in_support_files_is_rejected_and_cleaned_up(self, manager):
        with pytest.raises(ValueError):
            _create(manager, support_files={"../escape.py": "boom"})
        assert os.listdir(sm.SESSIONS_ROOT) == []
        assert manager._used_ports == set()

    def test_support_paths_are_normalised(self, manager):
        session = _create(manager, support_files={"C:\\tools\\a.py": "a", "/abs/b.py": "b"})
        assert os.path.isfile(os.path.join(session.work_dir, "tools", "a.py"))
        assert os.path.isfile(os.path.join(session.work_dir, "abs", "b.py"))

    def test_invalid_yaml_is_rejected(self, manager):
        with pytest.raises(ValueError, match="YAML"):
            _create(manager, config_yaml="platforms: [unclosed")

    def test_capacity_and_duplicates(self, manager):
        first = _create(manager)
        with pytest.raises(ValueError, match="already exists"):
            _create(manager, session_id=first.session_id)
        _create(manager)
        with pytest.raises(sm.SessionCapacityError):
            _create(manager)

    def test_popen_failure_releases_the_slot(self, manager, monkeypatch):
        def failing_popen(argv, **kwargs):
            raise subprocess.SubprocessError("Exception occurred in preexec_fn.")

        monkeypatch.setattr(sm.subprocess, "Popen", failing_popen)
        with pytest.raises(subprocess.SubprocessError):
            _create(manager)
        assert manager._used_uids == set() and os.listdir(sm.SESSIONS_ROOT) == []


class TestTerminate:
    def test_kills_the_group_sweeps_the_uid_and_frees_the_slot(self, manager, monkeypatch):
        session = _create(manager)
        process = manager.launched[0]

        # The fake exits once signalled.
        def exit_on_term(pgid, sig):
            manager.signals.append((pgid, sig))
            process.returncode = -sig

        monkeypatch.setattr(sm.os, "killpg", exit_on_term, raising=False)
        manager.terminate_session(session.session_id)

        assert manager.signals == [(process.pid, signal.SIGTERM)]
        assert manager.swept == [session.uid]
        assert manager._used_ports == set() and manager._used_uids == set()
        assert not os.path.exists(session.work_dir)

    def test_sigkill_after_grace_period(self, manager, monkeypatch):
        monkeypatch.setattr(sm, "_TERMINATE_GRACE_SECONDS", 0)
        session = _create(manager)
        process = manager.launched[0]

        def exit_on_kill(pgid, sig):
            manager.signals.append((pgid, sig))
            if sig == signal.SIGKILL:
                process.returncode = -sig

        monkeypatch.setattr(sm.os, "killpg", exit_on_kill, raising=False)
        manager.terminate_session(session.session_id)
        assert manager.signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]

    def test_already_exited_process_is_not_signalled(self, manager):
        session = _create(manager)
        manager.launched[0].returncode = 0
        manager.terminate_session(session.session_id)
        assert manager.signals == []  # its pid may already be recycled
        assert manager.swept == [session.uid]

    def test_survivors_quarantine_port_and_uid(self, manager, monkeypatch):
        monkeypatch.setattr(sm, "_kill_uid_processes", lambda uid: 1)
        session = _create(manager)
        manager.launched[0].returncode = 0
        manager.terminate_session(session.session_id)
        assert session.port in manager._used_ports and session.uid in manager._used_uids

    def test_unknown_session_only_removes_the_directory(self, manager):
        stale = os.path.join(sm.SESSIONS_ROOT, str(uuid.uuid4()))
        os.makedirs(stale)
        manager.terminate_session(os.path.basename(stale))
        assert not os.path.exists(stale)
        assert manager.swept == []

    def test_cleanup_expired(self, manager, monkeypatch):
        session = _create(manager)
        manager.launched[0].returncode = 0
        manager.cleanup_expired()
        assert manager.get_session(session.session_id) is None

    def test_sweep_refuses_non_session_uids(self):
        with pytest.raises(ValueError):
            sm._kill_uid_processes(0)


class TestSessionFiles:
    @pytest.fixture
    def work_dir(self, manager):
        session_id = str(uuid.uuid4())
        path = os.path.join(sm.SESSIONS_ROOT, session_id)
        os.makedirs(os.path.join(path, "tmp"))
        os.makedirs(os.path.join(path, "data", "empty"))
        with open(os.path.join(path, "agent.py"), "w", encoding="utf-8") as f:
            f.write("print('agent')")
        with open(os.path.join(path, "data", "notes.txt"), "w", encoding="utf-8") as f:
            f.write("notes")
        with open(os.path.join(path, "tmp", "scratch"), "w", encoding="utf-8") as f:
            f.write("hidden")
        return session_id, path

    def test_lists_regular_files_and_directories(self, manager, work_dir):
        session_id, _ = work_dir
        listing = manager.get_session_files(session_id)
        assert listing["files"] == [
            {"path": "agent.py", "content": "print('agent')"},
            {"path": "data/notes.txt", "content": "notes"},
        ]
        assert listing["directories"] == ["data", "data/empty"]

    def test_symlink_to_a_file_outside_the_work_dir_is_not_returned(self, manager, work_dir, tmp_path):
        session_id, path = work_dir
        secret = tmp_path / "outside_secret.txt"
        secret.write_text("TOP SECRET", encoding="utf-8")
        try:
            os.symlink(str(secret), os.path.join(path, "leak.txt"))
            os.symlink(str(tmp_path), os.path.join(path, "leakdir"), target_is_directory=True)
        except OSError as exc:  # Windows without Developer Mode / admin
            pytest.skip(f"cannot create symlinks on this machine: {exc}")

        listing = manager.get_session_files(session_id)

        paths = [f["path"] for f in listing["files"]]
        assert "leak.txt" not in paths
        assert not any(p.startswith("leakdir") for p in paths)
        assert "leakdir" not in listing["directories"]
        assert all("TOP SECRET" not in f["content"] for f in listing["files"])

    @pytest.mark.skipif(not IS_POSIX, reason="FIFOs only exist on POSIX")
    def test_fifo_is_skipped_without_blocking(self, manager, work_dir):
        session_id, path = work_dir
        os.mkfifo(os.path.join(path, "pipe"))
        paths = [f["path"] for f in manager.get_session_files(session_id)["files"]]
        assert "pipe" not in paths

    def test_large_file_is_summarised(self, manager, work_dir):
        session_id, path = work_dir
        with open(os.path.join(path, "big.bin"), "wb") as f:
            f.write(b"x" * (sm._MAX_FILE_SIZE_BYTES + 1))
        big = [f for f in manager.get_session_files(session_id)["files"] if f["path"] == "big.bin"]
        assert big and big[0]["content"].startswith("[File too large to display")

    def test_invalid_session_id(self, manager):
        assert manager.get_session_files("../etc") == {"files": [], "directories": []}


def _real_sandbox_available() -> bool:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        return False
    return sandbox_selftest_error(sm.UID_POOL[0]) is None


@pytest.mark.skipif(
    not _real_sandbox_available(),
    reason="needs Linux, root and a working bubblewrap (runs inside the simulator image)",
)
class TestRealSandbox:
    """End-to-end: a real sandboxed session, as in the simulator container."""

    @pytest.fixture(autouse=True)
    def sessions_root(self, monkeypatch):
        # Not under tmp_path: pytest's /tmp/pytest-of-root is 0700, and the
        # session uid must be able to traverse to its work dir.
        root = tempfile.mkdtemp(prefix="besser_sim_test_", dir="/tmp")
        os.chmod(root, 0o711)
        monkeypatch.setattr(sm, "SESSIONS_ROOT", root)
        yield root
        shutil.rmtree(root)

    def test_isolation_and_whole_session_kill(self, monkeypatch):
        monkeypatch.setenv("AGENT_SIMULATOR_API_TOKEN", "server-secret-token")
        manager = sm.SessionManager()
        sibling = os.path.join(sm.SESSIONS_ROOT, str(uuid.uuid4()))
        os.makedirs(sibling)
        agent_code = textwrap.dedent(f"""
            import os, subprocess, sys, time
            print("ENVIRON", open("/proc/1/environ", "rb").read().replace(b"\\0", b" ").decode(), flush=True)
            print("PIDS", sorted(p for p in os.listdir("/proc") if p.isdigit()), flush=True)
            print("SIBLING_VISIBLE", os.path.exists({sibling!r}), flush=True)
            print("ROOT_LIST", os.listdir({sm.SESSIONS_ROOT!r}), flush=True)
            # "/" itself is bwrap's private tmpfs; the container's dirs are read-only binds.
            writable = []
            for probe in ("/app/probe", "/etc/probe", "/opt/probe", "/usr/local/probe"):
                try:
                    open(probe, "w").close()
                    writable.append(probe)
                except OSError:
                    pass
            print("ROOTFS_WRITABLE", writable, flush=True)
            print("UID", os.getuid(), flush=True)
            # Double fork + setsid: must still die with the session.
            if os.fork() == 0:
                os.setsid()
                if os.fork() == 0:
                    time.sleep(600)
                os._exit(0)
            print("READY", flush=True)
            time.sleep(600)
        """)
        session = manager.create_session(str(uuid.uuid4()), agent_code, "", {}, [])
        lines = []
        deadline = time.time() + 30
        while time.time() < deadline and not any(line.startswith("READY") for line in lines):
            line = session.process.stdout.readline().decode()
            if not line:
                break
            lines.append(line.strip())
        output = "\n".join(lines)
        assert "READY" in output, output
        assert "server-secret-token" not in output
        assert "SIBLING_VISIBLE False" in output
        assert "ROOTFS_WRITABLE []" in output
        assert f"UID {session.uid}" in output
        assert f"ROOT_LIST ['{session.session_id}']" in output
        assert sm._live_pids_owned_by(session.uid)

        manager.terminate_session(session.session_id)

        assert sm._live_pids_owned_by(session.uid) == []
        assert session.uid not in manager._used_uids
        assert not os.path.exists(session.work_dir)

    def test_rlimits_bind_the_session(self):
        manager = sm.SessionManager()
        agent_code = "import resource; print('NOFILE', resource.getrlimit(resource.RLIMIT_NOFILE), flush=True)"
        session = manager.create_session(str(uuid.uuid4()), agent_code, "", {}, [])
        output = session.process.stdout.read().decode()
        manager.terminate_session(session.session_id)
        assert f"NOFILE ({sm._RLIMIT_NOFILE}, {sm._RLIMIT_NOFILE})" in output, output
