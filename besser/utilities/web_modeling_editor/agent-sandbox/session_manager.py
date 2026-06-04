"""
Agent sandbox session manager.

Manages subprocess-based agent sessions with resource limits and port pooling.
Each session gets an isolated /tmp/sessions/{id}/ directory and a dedicated
WebSocket port from the pool for the agent's WebSocket platform.
"""
import logging
import os
import resource
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

MAX_SESSIONS = 5
SESSION_LIFETIME_SECONDS = 900  # 15 minutes
PORT_POOL = list(range(7700, 7700 + MAX_SESSIONS))
SESSION_UID_BASE = int(os.environ.get("AGENT_SANDBOX_SESSION_UID_BASE", "20000"))
UID_POOL = list(range(SESSION_UID_BASE, SESSION_UID_BASE + MAX_SESSIONS))
SESSIONS_ROOT = "/tmp/sessions"

# Per-subprocess resource limits.
# RLIMIT_AS (virtual memory) must be generous: Python + ML libs (openai,
# numpy, pandas…) map several GB of address space via shared libraries even
# when resident memory is small. 4 GB is a safe floor.
_RLIMIT_AS = 4 * 1024 * 1024 * 1024  # 4 GB virtual memory
_RLIMIT_CPU = 120                      # 120 s CPU time
_RLIMIT_FSIZE = 100 * 1024 * 1024     # 100 MB max single file
_RLIMIT_NPROC = 64
_RLIMIT_NOFILE = 1024                  # Python import machinery needs ~200+


@dataclass
class Session:
    session_id: str
    port: int
    uid: int
    process: subprocess.Popen
    work_dir: str
    started_at: float
    event_list: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        return (time.time() - self.started_at) > SESSION_LIFETIME_SECONDS

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def terminate(self) -> None:
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._used_ports: set = set()
        self._used_uids: set = set()
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

    def _acquire_uid(self) -> Optional[int]:
        for uid in UID_POOL:
            if uid not in self._used_uids:
                self._used_uids.add(uid)
                return uid
        return None

    def _release_uid(self, uid: int) -> None:
        self._used_uids.discard(uid)

    @staticmethod
    def _set_owner_and_permissions(root_path: str, uid: int) -> None:
        os.chown(root_path, uid, uid)
        os.chmod(root_path, 0o700)
        for current_root, dirnames, filenames in os.walk(root_path):
            for dirname in dirnames:
                target = os.path.join(current_root, dirname)
                os.chown(target, uid, uid)
                os.chmod(target, 0o700)
            for filename in filenames:
                target = os.path.join(current_root, filename)
                os.chown(target, uid, uid)
                os.chmod(target, 0o600)

    def _acquire_port(self) -> Optional[int]:
        for port in PORT_POOL:
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        return None

    def _release_port(self, port: int) -> None:
        self._used_ports.discard(port)

    def get_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    @staticmethod
    def _normalize_session_relative_path(path: str) -> Optional[str]:
        """Normalize an incoming path so it stays inside the session folder."""
        if not isinstance(path, str):
            return None
        candidate = path.strip().replace("\\", "/")
        if not candidate:
            return None
        if ":" in candidate:
            candidate = candidate.split(":", 1)[1]
        candidate = candidate.lstrip("/")
        parts = [part for part in candidate.split("/") if part not in ("", ".", "..")]
        if not parts:
            return None
        return os.path.join(*parts)

    @staticmethod
    def _safe_join(work_dir: str, rel_path: str) -> str:
        target = os.path.abspath(os.path.join(work_dir, rel_path))
        root = os.path.abspath(work_dir)
        if os.path.commonpath([root, target]) != root:
            raise ValueError(f"Path escapes session directory: {rel_path}")
        return target

    def _write_support_files(self, work_dir: str, support_files: Dict[str, str]) -> None:
        for raw_path, content in (support_files or {}).items():
            normalized = self._normalize_session_relative_path(raw_path)
            if not normalized:
                continue
            target_path = self._safe_join(work_dir, normalized)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content or "")

    def _create_workspace_dirs(self, work_dir: str, workspace_paths: List[str]) -> None:
        for raw_path in workspace_paths or []:
            normalized = self._normalize_session_relative_path(raw_path)
            if not normalized:
                continue
            target_dir = self._safe_join(work_dir, normalized)
            os.makedirs(target_dir, exist_ok=True)

    def create_session(
        self,
        session_id: str,
        agent_code: str,
        config_yaml: str,
        env_vars: dict,
        event_list: list,
        support_files: Optional[Dict[str, str]] = None,
        workspace_paths: Optional[List[str]] = None,
    ) -> Session:
        normalized_session_id = self._validate_session_id(session_id)
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError(f"Maximum concurrent sessions ({MAX_SESSIONS}) reached")

            port = self._acquire_port()
            if port is None:
                raise RuntimeError("No available ports for agent WebSocket")

            uid = self._acquire_uid()
            if uid is None:
                self._release_port(port)
                raise RuntimeError("No available sandbox UID for agent session")

        work_dir = self._session_work_dir(normalized_session_id)
        os.makedirs(work_dir, exist_ok=True)
        os.chmod(work_dir, 0o700)

        # Write agent code
        agent_path = os.path.join(work_dir, "agent.py")
        with open(agent_path, "w", encoding="utf-8") as f:
            f.write(agent_code)

        # Write config.yaml, overriding the websocket port
        try:
            config = yaml.safe_load(config_yaml) or {}
        except Exception:
            config = {}

        config.setdefault("platforms", {}).setdefault("websocket", {})["port"] = port

        # Inject popup credentials into config so they take precedence over any
        # hardcoded placeholder values in the generated config.yaml.  Only override
        # a key when the caller actually supplied a non-empty value; otherwise leave
        # whatever the generated config already contains (which may be a real key the
        # user stored in their agent config).
        _CRED_MAP = [
            ("OPENAI_API_KEY",            ("nlp", "openai",      "api_key")),
            ("HUGGINGFACEHUB_API_TOKEN",  ("nlp", "huggingface", "token")),
            ("REPLICATE_API_TOKEN",       ("nlp", "replicate",   "api_key")),
        ]
        for env_key, config_path_parts in _CRED_MAP:
            value = env_vars.get(env_key)
            if value:
                node = config
                for part in config_path_parts[:-1]:
                    node = node.setdefault(part, {})
                node[config_path_parts[-1]] = value

        config_path = os.path.join(work_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Materialize generated sidecar files and workspace directories used by
        # reasoning agents in test mode.
        self._write_support_files(work_dir, support_files or {})
        self._create_workspace_dirs(work_dir, workspace_paths or [])

        # Ensure strict ownership and permissions before dropping privileges.
        self._set_owner_and_permissions(work_dir, uid)

        runtime_tmp = os.path.join(work_dir, "tmp")
        os.makedirs(runtime_tmp, exist_ok=True)
        os.chown(runtime_tmp, uid, uid)
        os.chmod(runtime_tmp, 0o700)

        # Subprocess environment: strict allowlist to avoid leaking container secrets.
        proc_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            "BESSER_TEST_MODE": "1",
            "BESSER_WS_PORT": str(port),
            "NLTK_DATA": os.environ.get("NLTK_DATA", "/opt/nltk_data"),
            "HOME": work_dir,
            "TMPDIR": runtime_tmp,
        }
        for key in ("OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "REPLICATE_API_TOKEN"):
            value = env_vars.get(key)
            if value:
                proc_env[key] = value

        def _set_limits():
            try:
                os.setgid(uid)
                os.setuid(uid)
                os.umask(0o077)
            except Exception as exc:
                # Fail fast: running untrusted code without UID isolation is unsafe.
                raise RuntimeError(f"Cannot switch sandbox subprocess to uid {uid}") from exc

            try:
                resource.setrlimit(resource.RLIMIT_AS, (_RLIMIT_AS, _RLIMIT_AS))
                resource.setrlimit(resource.RLIMIT_CPU, (_RLIMIT_CPU, _RLIMIT_CPU))
                resource.setrlimit(resource.RLIMIT_FSIZE, (_RLIMIT_FSIZE, _RLIMIT_FSIZE))
                resource.setrlimit(resource.RLIMIT_NPROC, (_RLIMIT_NPROC, _RLIMIT_NPROC))
                resource.setrlimit(resource.RLIMIT_NOFILE, (_RLIMIT_NOFILE, _RLIMIT_NOFILE))
            except Exception as exc:
                logger.warning("Could not set resource limits: %s", exc)

        process = subprocess.Popen(
            ["python", "agent.py"],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=proc_env,
            preexec_fn=_set_limits,
        )

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
            self._sessions[normalized_session_id] = session
        logger.info("Created session %s on port %d (pid=%d)", normalized_session_id, port, process.pid)
        return session

    def terminate_session(self, session_id: str) -> None:
        normalized_session_id = self._validate_session_id(session_id)
        with self._lock:
            session = self._sessions.pop(normalized_session_id, None)
            if session:
                self._release_port(session.port)
                self._release_uid(session.uid)
        if session:
            session.terminate()
        # Always remove the work directory even when the session is no longer in the
        # registry (e.g. double-delete calls or sandbox restart with stale dirs).
        work_dir = self._session_work_dir(normalized_session_id)
        shutil.rmtree(work_dir, ignore_errors=True)
        if session:
            logger.info("Terminated session %s", normalized_session_id)
        else:
            logger.debug("Removed stale work dir for unknown session %s", normalized_session_id)

    def cleanup_expired(self) -> None:
        with self._lock:
            snapshot = list(self._sessions.items())
        expired = [sid for sid, s in snapshot if s.is_expired() or not s.is_alive()]
        for sid in expired:
            logger.info("Cleaning up expired/dead session %s", sid)
            try:
                self.terminate_session(sid)
            except ValueError:
                logger.warning("Skipping invalid session id during cleanup: %s", sid)


session_manager = SessionManager()
