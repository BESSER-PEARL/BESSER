"""The preview of BESSER Lab: the working copy's backend, editor and modeling agent, run as child processes.

The lab changes BESSER's own source; the preview lets the user try that source. Each service is started
from the working copy with the command its repository documents, writes its output to a log file, and is
restarted when one of its watched Python files changes (the editor's dev server reloads by itself).
Children get ``safe_subprocess_env()`` plus their own entries, so they never see the lab's LLM key or the
GitHub OAuth secret.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import IO, Mapping, Sequence
from urllib.parse import urlsplit

from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT, RepoSpec
from besser.utilities.subprocess_env import safe_subprocess_env

logger = logging.getLogger(__name__)

#: How long a child has to exit after it is asked to before it is killed.
STOP_TIMEOUT_SECONDS = 10.0
#: Never scanned for watched files: dependencies and caches hold no source of the working copy.
_SKIPPED_DIRS = frozenset({"node_modules", "__pycache__"})
#: Interpreter of a checkout's own environment, as BESSER's and the modeling agent's CLAUDE.md set it up
#: (``python -m venv venv``).
_VENV_PYTHON = (
    os.path.join("venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join("venv", "bin", "python"))


def _checkout_python(checkout: str, name: str) -> str:
    """The checkout's own venv interpreter, else this process's (logged: its packages may differ)."""
    python = os.path.join(checkout, _VENV_PYTHON)
    if os.path.isfile(python):
        return python
    logger.warning("No %s in the %s checkout: the preview %s runs with %s", _VENV_PYTHON, name, name, sys.executable)
    return sys.executable


@dataclass(frozen=True)
class PreviewService:
    """One preview process: how to start it and which files restart it."""

    name: str
    argv: tuple[str, ...]
    #: absolute working directory
    cwd: str
    #: entries added to ``safe_subprocess_env()``
    env: Mapping[str, str]
    #: files or directories whose ``.py`` files restart the service when they change
    watch: tuple[str, ...]
    port: int


def default_services(repos: Mapping[str, RepoSpec], *, backend_port: int = 9100, frontend_port: int = 8180,
                     agent_port: int = 8865, public_url: str = "") -> tuple[PreviewService, ...]:
    """The backend, the editor's dev server and the modeling agent of the working copy.

    ``public_url`` is where the browser reaches the preview (the lab's nginx sends ``/besser_api/`` to the
    backend, ``/agent`` to the modeling agent and ``/`` to the editor); empty means each service on its own
    localhost port. The modeling agent reads its port from ``config.yaml`` in its checkout, which must say
    ``agent_port``.
    """
    besser_path = repos[REPO_BESSER].path
    frontend_path = repos[REPO_FRONTEND].path
    agent_path = repos[REPO_MODELING_AGENT].path

    public_url = public_url.rstrip("/")
    if public_url:
        parts = urlsplit(public_url)
        backend_url = f"{public_url}/besser_api"
        agent_url = ("wss" if parts.scheme == "https" else "ws") + public_url[len(parts.scheme):] + "/agent"
        editor_origins = f"{parts.scheme}://{parts.netloc}"
    else:
        backend_url = f"http://localhost:{backend_port}/besser_api"
        agent_url = f"ws://localhost:{agent_port}"
        editor_origins = f"http://localhost:{frontend_port},http://127.0.0.1:{frontend_port}"

    # A dev build of the editor reads its backend from VITE_DEV_BACKEND_URL (shared/constants/constant.ts).
    frontend_env = {"VITE_DEV_BACKEND_URL": backend_url, "UML_BOT_WS_URL": agent_url}
    if public_url:
        # Vite's dev server refuses requests whose Host header it does not know (nginx passes the public one).
        frontend_env["__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS"] = urlsplit(public_url).hostname or ""

    return (
        PreviewService(
            name="backend",
            # BESSER CLAUDE.md, "Running locally".
            argv=(_checkout_python(besser_path, "backend"), "-m",
                  "besser.utilities.web_modeling_editor.backend.backend",
                  "--host", "127.0.0.1", "--port", str(backend_port)),
            cwd=besser_path,
            # PYTHONPATH makes the working copy's ``besser`` win over any installed one. The lab's own flags and
            # build stamp pass safe_subprocess_env(): the preview is a plain editor backend, not a second lab.
            env={"PYTHONPATH": besser_path, "PYTHONUNBUFFERED": "1", "CORS_ORIGINS": editor_origins,
                 "BESSER_LAB_ENABLED": "false", "BESSER_LAB_PREVIEW_ENABLED": "false", "BESSER_BUILD_SHA": ""},
            watch=(os.path.join(besser_path, "besser"),),
            port=backend_port,
        ),
        PreviewService(
            name="frontend",
            # WME CLAUDE.md runs ``npm run dev`` (Vite); called directly, since that script pins port 8080.
            argv=("npx", "vite", "--host", "127.0.0.1", "--port", str(frontend_port), "--strictPort"),
            cwd=os.path.join(frontend_path, "packages", "webapp"),
            env=frontend_env,
            watch=(),
            port=frontend_port,
        ),
        PreviewService(
            name="modeling_agent",
            # modeling-agent CLAUDE.md; ``src/`` is its import root.
            argv=(_checkout_python(agent_path, "modeling agent"), "modeling_agent.py"),
            cwd=agent_path,
            env={"PYTHONPATH": os.path.join(agent_path, "src"), "PYTHONUNBUFFERED": "1"},
            watch=(os.path.join(agent_path, "src"), os.path.join(agent_path, "modeling_agent.py")),
            port=agent_port,
        ),
    )


def _python_mtimes(paths: Sequence[str]) -> dict[str, int]:
    """Modification time of every ``.py`` file under ``paths`` (a path may also be one file)."""
    mtimes: dict[str, int] = {}
    for root_path in paths:
        if os.path.isfile(root_path):
            mtimes[root_path] = os.stat(root_path).st_mtime_ns
            continue
        for directory, subdirs, files in os.walk(root_path):
            subdirs[:] = [d for d in subdirs if d not in _SKIPPED_DIRS and not d.startswith(".")]
            for file_name in files:
                if file_name.endswith(".py"):
                    path = os.path.join(directory, file_name)
                    try:
                        mtimes[path] = os.stat(path).st_mtime_ns
                    except FileNotFoundError:
                        continue  # deleted between the listing and the stat: the next scan sees it gone
    return mtimes


@dataclass
class _Running:
    """A started child, its log file and the watched files as they were when it started."""

    process: subprocess.Popen
    log: IO[bytes]
    mtimes: dict[str, int]


class PreviewSupervisor:
    """Starts, watches and stops the preview services. Safe to call from several threads."""

    def __init__(self, services: Sequence[PreviewService], *, log_dir: str):
        self._services = {service.name: service for service in services}
        self._log_dir = log_dir
        self._running: dict[str, _Running] = {}
        #: why a service is not running, by name ("stopped", "npx not found", ...)
        self._details: dict[str, str] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start every service that is not running."""
        with self._lock:
            for name in self._services:
                running = self._running.get(name)
                if running is None or running.process.poll() is not None:
                    self._launch(name)

    def stop(self) -> None:
        """Stop every service, with its whole process group on POSIX."""
        with self._lock:
            for name in list(self._running):
                self._terminate(name)

    def restart(self, name: str) -> None:
        """Stop ``name`` and start it again. Raises ``KeyError`` for an unknown service."""
        if name not in self._services:
            raise KeyError(name)
        with self._lock:
            self._terminate(name)
            self._launch(name)

    def status(self) -> list[dict]:
        """``[{"name", "running", "detail"}]`` in the order the services were given."""
        with self._lock:
            return [self._status_of(name) for name in self._services]

    def restart_changed(self) -> list[str]:
        """Restart the services whose watched Python files changed since they started; return their names.

        Nothing happens before ``start()`` or after ``stop()``. A service that exited is restarted too when
        its files changed, so a fix to the error that stopped it brings it back.
        """
        restarted = []
        with self._lock:
            for name, service in self._services.items():
                running = self._running.get(name)
                if not service.watch or running is None:
                    continue
                if _python_mtimes(service.watch) != running.mtimes:
                    logger.info("Lab preview: %s's source changed, restarting it", name)
                    self._terminate(name)
                    self._launch(name)
                    restarted.append(name)
        return restarted

    def _status_of(self, name: str) -> dict:
        service = self._services[name]
        running = self._running.get(name)
        if running is None:
            detail = self._details.get(name, "not started")
            return {"name": name, "running": False, "detail": detail}
        code = running.process.poll()
        if code is None:
            return {"name": name, "running": True, "detail": f"running on port {service.port}"}
        return {"name": name, "running": False, "detail": f"exited with code {code}, see {name}.log"}

    def _launch(self, name: str) -> None:
        """Start ``name``; a command that cannot be started is reported by ``status()``, not raised."""
        service = self._services[name]
        self._details.pop(name, None)
        executable = shutil.which(service.argv[0])
        if executable is None:
            self._details[name] = f"{service.argv[0]} not found"
            logger.warning("Lab preview: cannot start %s: %s not found", name, service.argv[0])
            return
        mtimes = _python_mtimes(service.watch)
        env = {**safe_subprocess_env(), **service.env}
        os.makedirs(self._log_dir, exist_ok=True)
        # One log per process: a restart starts it afresh.
        log = open(os.path.join(self._log_dir, f"{name}.log"), "wb")  # closed in _terminate
        # Its own process group, so that stopping it reaches its children (npx -> node) too.
        group = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        try:
            process = subprocess.Popen(
                [executable, *service.argv[1:]], cwd=service.cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, **group)
        except OSError as error:
            log.close()
            self._details[name] = f"could not start: {error.strerror or error}"
            logger.warning("Lab preview: cannot start %s: %s", name, error)
            return
        self._running[name] = _Running(process, log, mtimes)
        logger.info("Lab preview: started %s (pid %s)", name, process.pid)

    def _terminate(self, name: str) -> None:
        running = self._running.pop(name, None)
        if running is None:
            return
        self._details[name] = "stopped"
        process = running.process
        try:
            _signal_group(process, kill=False)
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("Lab preview: %s did not stop in %ss, killing it", name, STOP_TIMEOUT_SECONDS)
            # Whatever is left: the child after the timeout, or a grandchild that ignored the request.
            _signal_group(process, kill=True)
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        finally:
            running.log.close()


def _signal_group(process: subprocess.Popen, *, kill: bool) -> None:
    """Ask the child's process group to stop, or kill it. On Windows only the child itself is signalled."""
    if os.name == "nt":
        if process.poll() is None and kill:
            process.kill()
        elif process.poll() is None:
            process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
    except ProcessLookupError:
        return  # the whole group has already exited
