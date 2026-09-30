"""
Agent simulation session manager.

Each session gets its own work directory under ``SESSIONS_ROOT``, a dedicated
WebSocket port and a dedicated Unix UID from fixed pools, and one agent
process started inside the bubblewrap sandbox (see :mod:`.sandbox`).

Why a UID per session even though bwrap already gives each session its own
namespaces: ``RLIMIT_NPROC`` is accounted per real UID, so a shared UID would
let one session's threads exhaust every other session's process budget; the
per-UID ownership keeps sibling work directories unreadable even under the
``AGENT_SIMULATOR_SANDBOX=off`` development opt-out; and "every process owned
by the UID" is a complete kill list when a session ends. Switching UID is why
the simulator runs as root (with a reduced capability set, see
``docker-compose.yml``); agent code itself never runs as root.
"""
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import yaml

from besser.utilities.path_utils import normalize_relative_path
from besser.utilities.web_modeling_editor.agent_simulator.sandbox import (
    SandboxUnavailable,
    safe_subprocess_env,
    sandboxed_argv,
)

if sys.platform.startswith("linux"):
    import resource

logger = logging.getLogger(__name__)

MAX_SESSIONS = int(os.environ.get("AGENT_SIMULATOR_MAX_SESSIONS") or "5")
SESSION_LIFETIME_SECONDS = int(os.environ.get("AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS") or "900")
_PORT_POOL_START = int(os.environ.get("AGENT_SIMULATOR_PORT_POOL_START") or "7700")
PORT_POOL = list(range(_PORT_POOL_START, _PORT_POOL_START + MAX_SESSIONS))
SESSION_UID_BASE = int(os.environ.get("AGENT_SIMULATOR_SESSION_UID_BASE") or "20000")
UID_POOL = list(range(SESSION_UID_BASE, SESSION_UID_BASE + MAX_SESSIONS))
SESSIONS_ROOT = "/tmp/sessions"

# The agent's WebSocket server listens on loopback only: the simulator relays
# to it from inside the same network namespace, nothing else needs to reach it.
AGENT_WS_HOST = "127.0.0.1"

# Per-subprocess resource limits.
# RLIMIT_AS (virtual memory) must be generous: Python + ML libs (openai,
# numpy, pandas...) map several GB of address space via shared libraries even
# when resident memory is small. 4 GB is a safe floor.
_GIB = 1024 * 1024 * 1024
_MIB = 1024 * 1024
_RLIMIT_AS = int(os.environ.get("AGENT_SIMULATOR_RLIMIT_AS_GB") or "4") * _GIB
_RLIMIT_CPU = int(os.environ.get("AGENT_SIMULATOR_RLIMIT_CPU_SEC") or "120")
_RLIMIT_FSIZE = int(os.environ.get("AGENT_SIMULATOR_RLIMIT_FSIZE_MB") or "100") * _MIB
_RLIMIT_NPROC = int(os.environ.get("AGENT_SIMULATOR_RLIMIT_NPROC") or "64")
_RLIMIT_NOFILE = int(os.environ.get("AGENT_SIMULATOR_RLIMIT_NOFILE") or "1024")

_TERMINATE_GRACE_SECONDS = 5.0
_UID_SWEEP_TIMEOUT_SECONDS = 5.0

# Credentials the user may supply for one session, mapped to where BAF reads
# them in config.yaml. They reach the agent both as environment variables and
# in the config, overriding any placeholder the generated config contains.
SESSION_CREDENTIAL_CONFIG_PATHS: Dict[str, Tuple[str, ...]] = {
    "OPENAI_API_KEY": ("nlp", "openai", "api_key"),
    "HUGGINGFACEHUB_API_TOKEN": ("nlp", "huggingface", "token"),
    "REPLICATE_API_TOKEN": ("nlp", "replicate", "api_key"),
}

# Work-dir subdirectories never returned by the file listing.
_LISTING_SKIP_DIRS = frozenset({"tmp"})
_MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB per file


class SessionCapacityError(RuntimeError):
    """No session slot (session count, port or UID) is available right now."""


def _apply_rlimits() -> None:
    """Child-side (``preexec_fn``): cap the agent process's resources.

    Runs after the UID switch, so the limits bind the unprivileged session UID.
    Any failure propagates: :class:`subprocess.Popen` then raises in the parent
    and the session is refused rather than run without limits.
    """
    resource.setrlimit(resource.RLIMIT_AS, (_RLIMIT_AS, _RLIMIT_AS))
    resource.setrlimit(resource.RLIMIT_CPU, (_RLIMIT_CPU, _RLIMIT_CPU))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_RLIMIT_FSIZE, _RLIMIT_FSIZE))
    resource.setrlimit(resource.RLIMIT_NPROC, (_RLIMIT_NPROC, _RLIMIT_NPROC))
    resource.setrlimit(resource.RLIMIT_NOFILE, (_RLIMIT_NOFILE, _RLIMIT_NOFILE))


def _live_pids_owned_by(uid: int) -> List[int]:
    """PIDs of every non-zombie process whose real UID is ``uid`` (Linux /proc)."""
    pids: List[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status", encoding="utf-8") as status_file:
                status = status_file.read()
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited while we were scanning
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        real_uid = int(fields["Uid"].split()[0])
        state = fields["State"].strip()[:1]
        if real_uid == uid and state not in ("Z", "X"):
            pids.append(int(entry))
    return pids


def _kill_uid_processes(uid: int, timeout: float = _UID_SWEEP_TIMEOUT_SECONDS) -> int:
    """SIGKILL every process owned by ``uid`` until none is left.

    Returns:
        int: the number of processes still alive after ``timeout`` (0 when the
        UID is clean and may be handed to another session).
    """
    if uid not in UID_POOL:
        raise ValueError(f"Refusing to sweep uid {uid}: not a session uid")
    deadline = time.monotonic() + timeout
    remaining = _live_pids_owned_by(uid)
    while remaining and time.monotonic() < deadline:
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        time.sleep(0.05)
        remaining = _live_pids_owned_by(uid)
    return len(remaining)


@dataclass
class Session:
    """One running agent: its process, slot (port, UID) and work directory."""

    session_id: str
    port: int
    uid: int
    process: subprocess.Popen
    work_dir: str
    started_at: float
    event_list: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """True once the session outlived ``SESSION_LIFETIME_SECONDS``."""
        return (time.time() - self.started_at) > SESSION_LIFETIME_SECONDS

    def is_alive(self) -> bool:
        """True while the session's top-level process has not exited."""
        return self.process.poll() is None

    def terminate(self) -> None:
        """Stop the session's whole process group: SIGTERM, then SIGKILL.

        The process was started as a session leader, so its PID is also its
        process-group id. Killing bwrap tears down the sandbox's PID namespace
        (``--die-with-parent``), which reaches children that double-forked or
        called ``setsid`` inside it. Anything else still owned by the session
        UID is swept by :func:`_kill_uid_processes` afterwards.
        """
        # Only signal a leader that is not reaped yet: once reaped its PID (and
        # so the pgid) may be recycled by an unrelated process.
        if self.process.poll() is None:
            pgid = self.process.pid
            self._signal_group(pgid, signal.SIGTERM)
            try:
                self.process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                logger.info("Session %s ignored SIGTERM; sending SIGKILL", self.session_id)
                self._signal_group(pgid, signal.SIGKILL)
                self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()

    @staticmethod
    def _signal_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            logger.debug("Process group %d already gone (signal %d)", pgid, sig)


class SessionManager:
    """Registry of running agent sessions and of the port / UID pools.

    A port and a UID are only handed out again once every process that could
    still hold them is gone; a UID whose processes cannot be killed stays
    quarantined (logged as an error) instead of being reused.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._used_ports: set = set()
        self._used_uids: set = set()
        self._starting: set = set()  # ids between slot reservation and registration
        self._lock = RLock()

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        try:
            return str(uuid.UUID(str(session_id)))
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid session_id format") from exc

    @staticmethod
    def _session_work_dir(session_id: str) -> str:
        safe_session_id = SessionManager._validate_session_id(session_id)
        root = os.path.abspath(SESSIONS_ROOT)
        target = os.path.abspath(os.path.join(root, safe_session_id))
        if os.path.commonpath([root, target]) != root:
            raise ValueError("Invalid session_id path")
        return target

    def _acquire_slot(self, session_id: str) -> Tuple[int, int]:
        """Reserve a free (port, uid) pair for ``session_id``. Caller holds ``self._lock``."""
        if session_id in self._sessions or session_id in self._starting:
            raise ValueError(f"Session {session_id} already exists")
        if len(self._sessions) + len(self._starting) >= MAX_SESSIONS:
            raise SessionCapacityError(f"Maximum concurrent sessions ({MAX_SESSIONS}) reached")
        port = next((p for p in PORT_POOL if p not in self._used_ports), None)
        uid = next((u for u in UID_POOL if u not in self._used_uids), None)
        if port is None or uid is None:
            raise SessionCapacityError("No free port / uid for an agent session")
        self._used_ports.add(port)
        self._used_uids.add(uid)
        self._starting.add(session_id)
        return port, uid

    def _release_slot(self, port: int, uid: int) -> None:
        with self._lock:
            self._used_ports.discard(port)
            self._used_uids.discard(uid)

    @staticmethod
    def _secure_tree(root_path: str, uid: int) -> None:
        """Hand the work dir to ``uid``: dirs 0700, files 0600, nothing for others.

        chmod runs before chown, while root still owns the entry, so the
        simulator needs no CAP_FOWNER.
        """
        os.chmod(root_path, 0o700)
        os.chown(root_path, uid, uid, follow_symlinks=False)
        for current_root, dirnames, filenames in os.walk(root_path):
            for dirname in dirnames:
                target = os.path.join(current_root, dirname)
                os.chmod(target, 0o700)
                os.chown(target, uid, uid, follow_symlinks=False)
            for filename in filenames:
                target = os.path.join(current_root, filename)
                os.chmod(target, 0o600)
                os.chown(target, uid, uid, follow_symlinks=False)

    def get_session_count(self) -> int:
        """Number of sessions currently registered."""
        with self._lock:
            return len(self._sessions)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return the running session with this id, or ``None``."""
        with self._lock:
            return self._sessions.get(session_id)

    @staticmethod
    def _safe_join(work_dir: str, rel_path: str) -> str:
        target = os.path.abspath(os.path.join(work_dir, rel_path))
        root = os.path.abspath(work_dir)
        if os.path.commonpath([root, target]) != root:
            raise ValueError(f"Path escapes session directory: {rel_path}")
        return target

    def _write_support_files(self, work_dir: str, support_files: Dict[str, str]) -> None:
        for raw_path, content in support_files.items():
            target_path = self._safe_join(work_dir, normalize_relative_path(raw_path))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content or "")

    def _create_workspace_dirs(self, work_dir: str, workspace_paths: List[str]) -> None:
        for raw_path in workspace_paths:
            os.makedirs(self._safe_join(work_dir, normalize_relative_path(raw_path)), exist_ok=True)

    @staticmethod
    def _build_config(config_yaml: str, port: int, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """Parse the generated config.yaml and pin the session's WebSocket endpoint.

        Credentials the user supplied override whatever the generated config
        holds (usually a placeholder); a credential left empty keeps the
        generated value, which may be a real key stored in the agent config.
        """
        try:
            config = yaml.safe_load(config_yaml) if config_yaml else {}
        except yaml.YAMLError as exc:
            raise ValueError(f"config_yaml is not valid YAML: {exc}") from exc
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError("config_yaml must be a YAML mapping")
        websocket = config.setdefault("platforms", {}).setdefault("websocket", {})
        websocket["host"] = AGENT_WS_HOST
        websocket["port"] = port
        for env_key, config_path in SESSION_CREDENTIAL_CONFIG_PATHS.items():
            value = env_vars.get(env_key)
            if value:
                node = config
                for part in config_path[:-1]:
                    node = node.setdefault(part, {})
                node[config_path[-1]] = value
        return config

    @staticmethod
    def _session_env(env_vars: Dict[str, str], port: int, work_dir: str) -> Dict[str, str]:
        runtime_tmp = os.path.join(work_dir, "tmp")
        extra = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "BESSER_TEST_MODE": "1",
            "BESSER_WS_PORT": str(port),
            "NLTK_DATA": os.environ.get("NLTK_DATA", "/opt/nltk_data"),
            "HOME": work_dir,
            "TMPDIR": runtime_tmp,
        }
        for key in SESSION_CREDENTIAL_CONFIG_PATHS:
            if env_vars.get(key):
                extra[key] = env_vars[key]
        return safe_subprocess_env(extra)

    def create_session(
        self,
        session_id: str,
        agent_code: str,
        config_yaml: str,
        env_vars: Dict[str, str],
        event_list: List[str],
        support_files: Optional[Dict[str, str]] = None,
        workspace_paths: Optional[List[str]] = None,
    ) -> Session:
        """Materialise a work directory and start the agent inside the sandbox.

        Args:
            session_id (str): UUID chosen by the backend.
            agent_code (str): The generated ``agent.py``.
            config_yaml (str): The generated ``config.yaml``; its WebSocket host
                and port are overridden with the session's own.
            env_vars (Dict[str, str]): Per-session LLM credentials. Only the
                keys in ``SESSION_CREDENTIAL_CONFIG_PATHS`` are used.
            event_list (List[str]): Events the agent reacts to (for the UI).
            support_files (Optional[Dict[str, str]]): Extra files, by relative path.
            workspace_paths (Optional[List[str]]): Directories to create.

        Returns:
            Session: The registered, running session.

        Raises:
            ValueError: Invalid session id, path or config.
            SessionCapacityError: No free session slot.
            SandboxUnavailable: The sandbox cannot be started (fail closed).
            OSError | subprocess.SubprocessError: The process could not start,
                including a failure to apply the resource limits.
        """
        normalized_session_id = self._validate_session_id(session_id)
        with self._lock:
            port, uid = self._acquire_slot(normalized_session_id)
        work_dir = self._session_work_dir(normalized_session_id)
        try:
            process = self._start_process(
                work_dir, port, uid, agent_code, config_yaml, env_vars,
                support_files or {}, workspace_paths or [],
            )
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            with self._lock:
                self._starting.discard(normalized_session_id)
            self._release_slot(port, uid)
            raise

        session = Session(
            session_id=normalized_session_id,
            port=port,
            uid=uid,
            process=process,
            work_dir=work_dir,
            started_at=time.time(),
            event_list=event_list,
        )
        with self._lock:
            self._starting.discard(normalized_session_id)
            self._sessions[normalized_session_id] = session
        logger.info(
            "Created session %s on port %d (uid=%d, pid=%d)", normalized_session_id, port, uid, process.pid,
        )
        return session

    def _start_process(
        self,
        work_dir: str,
        port: int,
        uid: int,
        agent_code: str,
        config_yaml: str,
        env_vars: Dict[str, str],
        support_files: Dict[str, str],
        workspace_paths: List[str],
    ) -> subprocess.Popen:
        # Resolve the sandbox first: when it is unavailable nothing is written.
        argv = sandboxed_argv(
            [sys.executable, "agent.py"], work_dir=work_dir, sessions_root=SESSIONS_ROOT, uid=uid,
        )
        if os.geteuid() != 0:
            raise SandboxUnavailable("The agent simulator must run as root to give each session its own uid")
        config = self._build_config(config_yaml, port, env_vars)

        os.makedirs(work_dir, mode=0o700)
        with open(os.path.join(work_dir, "agent.py"), "w", encoding="utf-8") as f:
            f.write(agent_code)
        with open(os.path.join(work_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        # Sidecar files and workspace directories used by reasoning agents.
        self._write_support_files(work_dir, support_files)
        self._create_workspace_dirs(work_dir, workspace_paths)
        os.makedirs(os.path.join(work_dir, "tmp"), exist_ok=True)
        self._secure_tree(work_dir, uid)

        return subprocess.Popen(
            argv,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._session_env(env_vars, port, work_dir),
            user=uid,
            group=uid,
            extra_groups=[],  # setgroups([]): drop root's supplementary groups
            umask=0o077,
            start_new_session=True,  # own session + process group, see Session.terminate
            preexec_fn=_apply_rlimits,
        )

    def terminate_session(self, session_id: str) -> None:
        """Kill a session's processes, free its slot and remove its work dir.

        Safe to call for an unknown session (double delete, stale directory
        after a restart): the work directory is removed either way. The port
        and UID return to the pools only after every process of the UID is
        gone.
        """
        normalized_session_id = self._validate_session_id(session_id)
        with self._lock:
            session = self._sessions.pop(normalized_session_id, None)
        if session:
            session.terminate()
            survivors = _kill_uid_processes(session.uid)
            if survivors:
                logger.error(
                    "Session %s: %d process(es) of uid %d survived SIGKILL; "
                    "keeping port %d and uid %d out of the pool",
                    normalized_session_id, survivors, session.uid, session.port, session.uid,
                )
            else:
                self._release_slot(session.port, session.uid)
        shutil.rmtree(self._session_work_dir(normalized_session_id), ignore_errors=True)
        if session:
            logger.info("Terminated session %s", normalized_session_id)
        else:
            logger.debug("Removed stale work dir for unknown session %s", normalized_session_id)

    @staticmethod
    def _read_regular_file(abs_path: str, work_root: str) -> Optional[str]:
        """Read one file of the work dir without ever following a symlink.

        Returns ``None`` for anything that is not a regular file inside the
        work dir (symlink, FIFO, device, socket, swapped in mid-scan...).
        """
        before = os.lstat(abs_path)
        if not stat.S_ISREG(before.st_mode):
            return None
        real = os.path.realpath(abs_path)
        if os.path.commonpath([work_root, real]) != work_root:
            return None
        if before.st_size > _MAX_FILE_SIZE_BYTES:
            return f"[File too large to display: {before.st_size} bytes]"
        # O_NOFOLLOW: a symlink swapped in after lstat fails the open (ELOOP);
        # O_NONBLOCK: a FIFO swapped in cannot hang the reader.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(abs_path, flags)
        with os.fdopen(fd, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                return None
            data = handle.read(_MAX_FILE_SIZE_BYTES + 1)
        if len(data) > _MAX_FILE_SIZE_BYTES:
            return f"[File too large to display: {opened.st_size} bytes]"
        return data.decode("utf-8", errors="replace")

    def get_session_files(self, session_id: str) -> Dict[str, Any]:
        """Return the regular files and directories under the session work dir.

        Symlinks (to files or directories) are never followed nor returned:
        agent code controls the work dir, and the simulator reads it as root.
        The ``directories`` list lets the frontend render empty folders.
        """
        try:
            normalized = self._validate_session_id(session_id)
        except ValueError:
            return {"files": [], "directories": []}
        work_dir = self._session_work_dir(normalized)
        if not os.path.isdir(work_dir) or os.path.islink(work_dir):
            return {"files": [], "directories": []}
        work_root = os.path.realpath(work_dir)
        files = []
        directories = set()
        for root, dirnames, filenames in os.walk(work_dir, followlinks=False):
            # os.walk lists a symlink to a directory under dirnames; drop it.
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _LISTING_SKIP_DIRS and not os.path.islink(os.path.join(root, d))
            )
            rel_root = os.path.relpath(root, work_dir).replace("\\", "/")
            for dirname in dirnames:
                directories.add(f"{rel_root}/{dirname}" if rel_root != "." else dirname)
            for filename in sorted(filenames):
                abs_path = os.path.join(root, filename)
                rel_path = f"{rel_root}/{filename}" if rel_root != "." else filename
                try:
                    content = self._read_regular_file(abs_path, work_root)
                except OSError as exc:
                    logger.debug("Skipping %s in session %s: %s", rel_path, normalized, exc)
                    continue
                if content is not None:
                    files.append({"path": rel_path, "content": content})
        return {"files": files, "directories": sorted(directories)}

    def cleanup_expired(self) -> None:
        """Terminate every session that expired or whose process exited."""
        with self._lock:
            snapshot = list(self._sessions.items())
        expired = [sid for sid, s in snapshot if s.is_expired() or not s.is_alive()]
        for sid in expired:
            logger.info("Cleaning up expired/dead session %s", sid)
            self.terminate_session(sid)


session_manager = SessionManager()
