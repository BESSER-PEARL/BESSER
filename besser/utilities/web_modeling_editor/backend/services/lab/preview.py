"""The preview of BESSER Lab: a change's backend, editor and modeling agent, each in its own sandbox.

The lab changes BESSER's own source; the preview lets the user try a change. :class:`PreviewPool` holds a few
slots (one per origin in ``BESSER_LAB_PREVIEW_URLS``), each showing one change: slot *i* runs the change's three
services (:func:`default_services`) on ports ``9100+10i``, ``8180+10i`` and ``8865+10i`` under a
:class:`PreviewSupervisor`. Each service runs in the lab's sandbox (the change's trees read-write, the git root
read-only, the lab's own folders hidden) with a private network: pasta forwards exactly its own port in from the
host and, for the modeling agent, the LLM proxy's port out. On Linux a service whose sandbox cannot start fails;
it never runs unconfined. Only on a platform without namespaces (Windows, macOS) does it run unconfined, and the
preview says so. Children get ``safe_subprocess_env()`` plus their own entries, so they never see the lab's LLM key,
the preview's OpenAI key or the GitHub OAuth secret.
"""

from __future__ import annotations

import copy
import ipaddress
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import IO, Callable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from besser.spec_driven_agent.execution.sandbox import (
    NETWORK_WAIT_SECONDS,
    NETWORK_WAIT_SOURCE,
    SANDBOX_POLICY_ENV,
    SandboxedCommand,
    SandboxMounts,
    SandboxUnavailable,
    read_child_pid,
    sandbox_supported_platform,
    sandboxed_command,
    spawn_long_lived,
    start_pasta,
    stop_process,
)
from besser.spec_driven_agent.lab.checks import confined_command_runner
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    LabChange,
    RepoSpec,
)
from besser.spec_driven_agent.lab.environments import Environments, LabEnvironmentError
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.secret_redaction import redact_text
from besser.utilities.subprocess_env import safe_subprocess_env
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabPreviewLog,
    LabPreviewService,
    LabPreviewSlot,
    LabPreviewView,
)
from besser.utilities.web_modeling_editor.backend.services.lab.llm_proxy import PREVIEW_PROXY_PLACEHOLDER_KEY

logger = logging.getLogger(__name__)

#: The three services of a slot, in the order the preview lists them.
SERVICE_NAMES = ("backend", "frontend", "modeling_agent")
#: Ports of slot 0; slot *i* adds ``i * SLOT_PORT_STEP`` to each (nginx and the setup script use this scheme).
PREVIEW_BACKEND_PORT = 9100
PREVIEW_FRONTEND_PORT = 8180
PREVIEW_AGENT_PORT = 8865
SLOT_PORT_STEP = 10
#: A slot nobody has used for this long may be taken for another change (D4: a switch breaks a session in use).
SLOT_IDLE_TAKEOVER_SECONDS = 600

#: How long a child has to exit after it is asked to before it is killed.
STOP_TIMEOUT_SECONDS = 10.0
#: How long bwrap may take to report its sandbox's pid (the checks' and the shell's start timeout).
SANDBOX_START_SECONDS = 30.0
#: Log lines a failed service shows in its state.
FAILED_LOG_TAIL_LINES = 20
#: The end of a log read for a tail: far beyond 2,000 lines of server output, and bounded however big the log.
LOG_TAIL_MAX_BYTES = 1 << 20
#: A readiness probe's connect timeout, and how long it waits for pasta to reset a connection nobody accepts
#: inside the sandbox (a reset arrives at once; a listening server waits for the request instead).
PROBE_CONNECT_SECONDS = 0.5
PROBE_RESET_SECONDS = 0.2

#: Never scanned for watched files: dependencies and caches hold no source of the working copy.
_SKIPPED_DIRS = frozenset({"node_modules", "__pycache__"})
#: Interpreter of a checkout's own environment (``environments.py`` makes ``<tree>/venv``).
_VENV_PYTHON = (
    os.path.join("venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join("venv", "bin", "python"))
# Runs in the service's sandbox first: wait for pasta's interface, then become the service.
_NETWORK_WAIT_EXEC = ("import os, sys, time\n" + NETWORK_WAIT_SOURCE +
                      "if not wait_for_network(float(sys.argv[1])):\n"
                      "    sys.exit('the sandbox network did not come up')\n"
                      "os.execvp(sys.argv[2], sys.argv[2:])\n")
_REPO_LABELS = {REPO_FRONTEND: "editor", REPO_BESSER: "BESSER", REPO_MODELING_AGENT: "modeling agent"}
_UNCONFINED_DETAIL = "On this machine the preview runs without the sandbox."


class PreviewFullError(Exception):
    """Every usable slot shows another change that is in use; the message names each slot's change and user."""


class PreviewUnavailableError(Exception):
    """The preview cannot show this change: it is off, or the change has no working trees; the message says why."""


@dataclass(frozen=True)
class PreviewService:
    """One preview process: how to start it and which files restart it."""

    name: str
    argv: tuple[str, ...]
    #: absolute working directory
    cwd: str
    #: entries added to ``safe_subprocess_env()``
    env: Mapping[str, str]
    #: files, or directories whose ``.py`` files, restart the service when they change
    watch: tuple[str, ...]
    #: the port it listens on, forwarded in from the host
    port: int
    #: host loopback ports the sandbox may reach (the modeling agent: the LLM proxy)
    tcp_out: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreviewSandbox:
    """Where a slot's services run: the change's root (the sandbox's workspace, which gives the shell's
    ``$HOME``), the lab's mount policy for its trees, and the host address pasta binds the ports on."""

    root: str
    mounts: SandboxMounts
    bind: str = "127.0.0.1"


def _checkout_python(checkout: str, name: str) -> str:
    """The checkout's own venv interpreter, else this process's (logged: its packages may differ)."""
    python = os.path.join(checkout, _VENV_PYTHON)
    if os.path.isfile(python):
        return python
    logger.warning("No %s in the %s checkout: the preview %s runs with %s", _VENV_PYTHON, name, name, sys.executable)
    return sys.executable


def default_services(repos: Mapping[str, RepoSpec], *, public_url: str = "",
                     backend_port: int = PREVIEW_BACKEND_PORT, frontend_port: int = PREVIEW_FRONTEND_PORT,
                     agent_port: int = PREVIEW_AGENT_PORT,
                     proxy_port: Optional[int] = None) -> tuple[PreviewService, ...]:
    """The backend, the editor's dev server and the modeling agent of a change's trees.

    ``public_url`` is the slot's origin (its nginx block sends ``/besser_api/`` to the backend, ``/agent`` to the
    modeling agent and ``/`` to the editor); empty means each service on its own localhost port. The preview editor
    talks only to that origin, and the backend gets no GitHub client id, so sign-in is off in the preview.
    ``proxy_port``: the lab's LLM proxy, the modeling agent's only way to OpenAI (it gets a placeholder key); None
    leaves the agent without a server key. The agent reads its port from ``config.yaml`` in its tree, which must
    say ``agent_port``.
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
        deployment_url = public_url
    else:
        backend_url = f"http://localhost:{backend_port}/besser_api"
        agent_url = f"ws://localhost:{agent_port}"
        editor_origins = f"http://localhost:{frontend_port},http://127.0.0.1:{frontend_port}"
        deployment_url = f"http://localhost:{frontend_port}"

    # A dev build of the editor reads its backend from VITE_DEV_BACKEND_URL (shared/constants/constant.ts).
    frontend_env = {"VITE_DEV_BACKEND_URL": backend_url, "UML_BOT_WS_URL": agent_url}
    if public_url:
        # Vite's dev server refuses requests whose Host header it does not know (nginx passes the public one).
        frontend_env["__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS"] = urlsplit(public_url).hostname or ""

    # BESSER_BACKEND_URL empty: the agent's opt-in telemetry stays off (the preview has no route to the lab).
    agent_env = {"PYTHONPATH": os.path.join(agent_path, "src"), "PYTHONUNBUFFERED": "1", "BESSER_BACKEND_URL": ""}
    if proxy_port is not None:
        agent_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{proxy_port}/v1"
        agent_env["OPENAI_API_KEY"] = PREVIEW_PROXY_PLACEHOLDER_KEY

    return (
        PreviewService(
            name="backend",
            # BESSER CLAUDE.md, "Running locally".
            argv=(_checkout_python(besser_path, "backend"), "-m",
                  "besser.utilities.web_modeling_editor.backend.backend",
                  "--host", "127.0.0.1", "--port", str(backend_port)),
            cwd=besser_path,
            # PYTHONPATH makes the change's ``besser`` win over any installed one. The lab's own flags and build
            # stamp pass safe_subprocess_env(): the preview is a plain editor backend, not a second lab.
            env={"PYTHONPATH": besser_path, "PYTHONUNBUFFERED": "1", "CORS_ORIGINS": editor_origins,
                 "DEPLOYMENT_URL": deployment_url, "BESSER_LAB_ENABLED": "false",
                 "BESSER_LAB_PREVIEW_ENABLED": "false", "BESSER_BUILD_SHA": ""},
            watch=(os.path.join(besser_path, "besser"),),
            port=backend_port,
        ),
        PreviewService(
            name="frontend",
            # WME CLAUDE.md runs ``npm run dev`` (Vite); called directly, since that script pins port 8080.
            argv=("npx", "vite", "--host", "127.0.0.1", "--port", str(frontend_port), "--strictPort"),
            cwd=os.path.join(frontend_path, "packages", "webapp"),
            env=frontend_env,
            # Vite reloads the editor's source itself; new dependencies need a restart.
            watch=(os.path.join(frontend_path, "package-lock.json"),),
            port=frontend_port,
        ),
        PreviewService(
            name="modeling_agent",
            # modeling-agent CLAUDE.md; ``src/`` is its import root.
            argv=(_checkout_python(agent_path, "modeling agent"), "modeling_agent.py"),
            cwd=agent_path,
            env=agent_env,
            watch=(os.path.join(agent_path, "src"), os.path.join(agent_path, "modeling_agent.py")),
            port=agent_port,
            tcp_out=() if proxy_port is None else (proxy_port,),
        ),
    )


def _watched_mtimes(paths: Sequence[str]) -> dict[str, int]:
    """Modification time of each watched file and of every ``.py`` file under each watched directory."""
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


def read_log_tail(path: str, lines: int) -> tuple[list[str], bool]:
    """The last ``lines`` lines of the log at ``path``, redacted, and whether the log holds more.

    Reads at most :data:`LOG_TAIL_MAX_BYTES` from the end. A missing log is empty.
    """
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - LOG_TAIL_MAX_BYTES)
            handle.seek(start)
            data = handle.read()
    except FileNotFoundError:
        return [], False
    every = data.decode("utf-8", "replace").splitlines()
    if start > 0:
        every = every[1:]  # the first line read is cut
    tail = every[-lines:] if lines > 0 else []
    truncated = start > 0 or len(every) > len(tail)
    if not tail:
        return [], truncated
    return redact_text("\n".join(tail))[0].split("\n"), truncated


def _accepts(host: str, port: int) -> bool:
    """Whether a server listens on ``host:port``.

    pasta accepts on the host port as soon as it runs and resets the connection when nothing listens inside the
    sandbox, so a connect alone is not enough: a server keeps the connection open waiting for the request.
    """
    try:
        with socket.create_connection((host, port), timeout=PROBE_CONNECT_SECONDS) as conn:
            conn.settimeout(PROBE_RESET_SECONDS)
            try:
                return conn.recv(1) != b""
            except socket.timeout:
                return True
    except OSError:
        return False


def _probe_host(bind: str) -> str:
    """Where the host reaches a port pasta forwards from ``bind``."""
    address = ipaddress.ip_address(bind)
    if address.is_unspecified:
        return "::1" if address.version == 6 else "127.0.0.1"
    return str(address)


def _network_wait_argv(plan_argv: Sequence[str], command: Sequence[str], info_fd: int) -> list[str]:
    """The bwrap ``plan_argv`` (ending with ``command``) reporting its pid on ``info_fd``, the command started once
    pasta has configured the sandbox's interface."""
    head = list(plan_argv[:len(plan_argv) - len(command)])
    return [head[0], "--info-fd", str(info_fd), *head[1:],
            sys.executable, "-I", "-S", "-c", _NETWORK_WAIT_EXEC, str(NETWORK_WAIT_SECONDS), *command]


def _service_state(name: str, state: str, detail: str, started_at: Optional[float] = None,
                   log_tail: Sequence[str] = ()) -> LabPreviewService:
    return LabPreviewService(name=name, state=state, detail=detail, startedAt=started_at, logTail=list(log_tail))


@dataclass
class _Running:
    """A started service: its process, its pasta, its log and the watched files as they were when it started."""

    process: subprocess.Popen
    pasta: Optional[subprocess.Popen]
    log: IO[bytes]
    mtimes: dict[str, int]
    started_at: float
    #: its port has accepted a connection since it started
    ready: bool = False


class PreviewSupervisor:
    """Starts, watches and stops one slot's services, each in its own sandbox. Safe to call from several threads."""

    def __init__(self, services: Sequence[PreviewService], *, log_dir: str, sandbox: PreviewSandbox):
        self._services = {service.name: service for service in services}
        self._log_dir = log_dir
        self._sandbox = sandbox
        self._running: dict[str, _Running] = {}
        #: why a service could not be started, by name
        self._failures: dict[str, str] = {}
        #: why a service is stopped, by name ("stopped")
        self._details: dict[str, str] = {}
        self._confined = sandbox_supported_platform()
        self._lock = threading.Lock()

    @property
    def confined(self) -> bool:
        """False once a service ran without the sandbox (no namespaces here, or the operator turned it off)."""
        return self._confined

    def start(self) -> None:
        """Start every service that is not running."""
        with self._lock:
            for name in self._services:
                running = self._running.get(name)
                if running is None or running.process.poll() is not None:
                    self._terminate(name)
                    self._launch(name)

    def stop(self) -> None:
        """Stop every service, with its whole process group on POSIX, and its pasta."""
        with self._lock:
            for name in self._services:
                self._terminate(name)

    def restart(self, name: str) -> None:
        """Stop ``name`` and start it again. Raises ``KeyError`` for an unknown service."""
        if name not in self._services:
            raise KeyError(name)
        with self._lock:
            self._terminate(name)
            self._launch(name)

    def status(self) -> list[LabPreviewService]:
        """Each service's state, in the order the services were given.

        ``starting`` while its process lives and its port accepts no connection, ``running`` once it does,
        ``failed`` when it could not start or exited (with the end of its log, redacted), ``stopped`` otherwise.
        """
        with self._lock:
            return [self._status_of(name) for name in self._services]

    def restart_changed(self) -> list[str]:
        """Restart the services whose watched files changed since they started; return their names.

        Nothing happens before ``start()`` or after ``stop()``. A service that exited is restarted too when
        its files changed, so a fix to the error that stopped it brings it back.
        """
        restarted = []
        with self._lock:
            for name, service in self._services.items():
                running = self._running.get(name)
                if not service.watch or running is None:
                    continue
                if _watched_mtimes(service.watch) != running.mtimes:
                    logger.info("Lab preview: %s's source changed, restarting it", name)
                    self._terminate(name)
                    self._launch(name)
                    restarted.append(name)
        return restarted

    def log_tail(self, name: str, lines: int) -> tuple[list[str], bool]:
        """The last ``lines`` lines of ``name``'s log, redacted, and whether it holds more. ``KeyError`` when
        unknown."""
        if name not in self._services:
            raise KeyError(name)
        return read_log_tail(self._log_path(name), lines)

    def _log_path(self, name: str) -> str:
        return os.path.join(self._log_dir, f"{name}.log")

    def _failed(self, name: str, detail: str, started_at: Optional[float] = None) -> LabPreviewService:
        tail, _ = read_log_tail(self._log_path(name), FAILED_LOG_TAIL_LINES)
        return _service_state(name, "failed", detail, started_at, tail)

    def _status_of(self, name: str) -> LabPreviewService:
        service = self._services[name]
        running = self._running.get(name)
        if running is None:
            if name in self._failures:
                return self._failed(name, self._failures[name])
            return _service_state(name, "stopped", self._details.get(name, "not started"))
        code = running.process.poll()
        if code is not None:
            return self._failed(name, f"exited with code {code}", running.started_at)
        if not running.ready:
            host = _probe_host(self._sandbox.bind) if self._confined else "127.0.0.1"
            running.ready = _accepts(host, service.port)
        state = "running" if running.ready else "starting"
        return _service_state(name, state, f"{state} on port {service.port}", running.started_at)

    def _fail(self, name: str, detail: str) -> None:
        self._failures[name] = detail
        logger.warning("Lab preview: cannot start %s: %s", name, detail)

    def _launch(self, name: str) -> None:
        """Start ``name`` in its sandbox; a service that cannot start is reported by ``status()``, not raised."""
        service = self._services[name]
        self._failures.pop(name, None)
        self._details.pop(name, None)
        env = {**safe_subprocess_env(), **service.env}
        executable = shutil.which(service.argv[0], path=env.get("PATH"))
        if executable is None:
            self._fail(name, f"{service.argv[0]} not found")
            return
        command = [executable, *service.argv[1:]]
        mounts = self._sandbox.mounts
        try:
            plan = sandboxed_command(command, workspace=self._sandbox.root, cwd=service.cwd,
                                     writable=mounts.writable, read_only=mounts.read_only, masked=mounts.masked,
                                     private_network=True)
        except SandboxUnavailable as exc:
            self._fail(name, f"the preview sandbox is unavailable: {exc.detail}")
            return
        except ValueError as exc:  # a mount rule a later mount would undo
            self._fail(name, f"the preview sandbox is misconfigured: {exc}")
            return
        if not plan.sandboxed:
            self._confined = False
            reason = (f"{SANDBOX_POLICY_ENV}=off" if plan.mode == "unconfined-override"
                      else f"no namespace sandbox on {sys.platform}")
            logger.warning("Lab preview: %s runs without the sandbox (%s)", name, reason)
        mtimes = _watched_mtimes(service.watch)
        os.makedirs(self._log_dir, exist_ok=True)
        # One log per process: a restart starts it afresh.
        log = open(self._log_path(name), "wb")  # closed in _terminate
        try:
            process, pasta = self._start(plan, command, service, env, log)
        except SandboxUnavailable as exc:
            log.close()
            self._fail(name, f"the preview sandbox is unavailable: {exc.detail}")
            return
        except (OSError, ValueError) as exc:
            log.close()
            self._fail(name, f"could not start: {getattr(exc, 'strerror', None) or exc}")
            return
        self._running[name] = _Running(process, pasta, log, mtimes, time.time())
        logger.info("Lab preview: started %s (pid %s)", name, process.pid)

    def _start(self, plan: SandboxedCommand, command: list[str], service: PreviewService, env: dict[str, str],
               log: IO[bytes]) -> tuple[subprocess.Popen, Optional[subprocess.Popen]]:
        """Fork the service, and its pasta for a private network, from the long-lived spawner thread (bwrap's
        ``--die-with-parent`` fires when the forking thread ends)."""
        # Its own process group, so that stopping it reaches its children (npx -> node) too.
        group = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        options = dict(cwd=service.cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                       **group)
        if not plan.private_network:
            return spawn_long_lived(lambda: subprocess.Popen(plan.argv, **options)), None
        info_r, info_w = os.pipe()
        try:
            try:
                process = spawn_long_lived(lambda: subprocess.Popen(
                    _network_wait_argv(plan.argv, command, info_w), pass_fds=(info_w,), **options))
            finally:
                os.close(info_w)
            try:
                pid = read_child_pid(info_r, SANDBOX_START_SECONDS)
                pasta = spawn_long_lived(lambda: start_pasta(
                    pid, log, tcp_in=((self._sandbox.bind, service.port),), tcp_out=service.tcp_out))
            except (SandboxUnavailable, ValueError):
                _signal_group(process, kill=True)
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
                raise
        finally:
            os.close(info_r)
        return process, pasta

    def _terminate(self, name: str) -> None:
        self._failures.pop(name, None)
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
            stop_process(running.pasta)
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


# -- The pool ------------------------------------------------------------------------------------------------


@dataclass
class _Slot:
    """One slot and the change it shows; a copy taken under the pool's lock is rendered outside it."""

    index: int
    #: what the Try it tab frames or opens
    url: str
    #: the origin the services are told about; "" in local mode (each on its own localhost port)
    public_url: str
    ports: tuple[int, int, int]
    #: why the slot cannot be used; "" when it can
    disabled: str = ""
    change: Optional[LabChange] = None
    used_by: Optional[str] = None
    started_at: Optional[float] = None
    last_used_at: Optional[float] = None
    supervisor: Optional[PreviewSupervisor] = None
    #: the preparation step under way; "" when none
    step: str = ""
    #: why preparing or refreshing failed; "" otherwise
    error: str = ""
    #: bumped whenever the slot changes hands, so a preparation for an earlier change discards its work
    generation: int = 0


def _origin(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.scheme}://{parts.netloc}".lower()


def _bind_problem(bind: str) -> str:
    """Why ``bind`` cannot be pasta's host address, or ""."""
    try:
        ipaddress.ip_address(bind)
    except ValueError:
        return (f"BESSER_LAB_PREVIEW_BIND={bind!r} is not an IP address: pasta binds the preview's ports on an IP "
                "literal such as 127.0.0.1 (or 0.0.0.0 in Docker). The preview is off until it is fixed.")
    return ""


class PreviewPool:
    """The preview slots, each showing one change. Safe to call from several threads.

    ``urls``: one origin per slot (``BESSER_LAB_PREVIEW_URLS``); empty gives one slot on localhost ports. A slot
    whose origin is the lab's own (``deployment_url``) is off: the preview would run unreviewed code where the lab
    session lives. ``sandbox_for(change)`` gives the sandbox of a change's services (the service builds it from the
    lab's policy); ``proxy_port`` is the LLM proxy's, when it runs. Logs go to ``<log_dir>/slot-<i>/<service>.log``.
    """

    def __init__(self, urls: Sequence[str], workspace: LabWorkspace, environments: Environments, *,
                 sandbox_for: Callable[[LabChange], PreviewSandbox], log_dir: str, bind: str = "127.0.0.1",
                 proxy_port: Optional[int] = None, deployment_url: str = "",
                 clock: Callable[[], float] = time.time):
        self._workspace = workspace
        self._environments = environments
        self._sandbox_for = sandbox_for
        self._log_dir = log_dir
        self._proxy_port = proxy_port
        self._clock = clock
        self._lock = threading.Lock()
        self._detail = _bind_problem(bind)
        if self._detail:
            logger.warning("Lab preview: %s", self._detail)
        lab_origin = _origin(deployment_url) if deployment_url.strip() else ""
        self._slots: list[_Slot] = []
        for index, url in enumerate(urls or [""]):
            ports = tuple(port + index * SLOT_PORT_STEP
                          for port in (PREVIEW_BACKEND_PORT, PREVIEW_FRONTEND_PORT, PREVIEW_AGENT_PORT))
            public_url = url.strip().rstrip("/")
            slot = _Slot(index=index, url=public_url or f"http://localhost:{ports[1]}", public_url=public_url,
                         ports=ports)
            if public_url and lab_origin and _origin(public_url) == lab_origin:
                slot.disabled = (f"Slot {index + 1} is off: its URL {public_url} is the lab's own origin "
                                 "(DEPLOYMENT_URL), where the preview would run unreviewed code next to the lab "
                                 "session. Give each slot its own origin in BESSER_LAB_PREVIEW_URLS.")
                logger.warning("Lab preview: %s", slot.disabled)
            self._slots.append(slot)
        if not self._detail and all(slot.disabled for slot in self._slots):
            self._detail = "Every preview slot is off: see each slot's detail."

    # -- Reads -------------------------------------------------------------------------------------------------

    def view(self) -> LabPreviewView:
        """Every slot: the change it shows, who used it last, its services."""
        with self._lock:
            copies = [copy.copy(slot) for slot in self._slots]
        return LabPreviewView(enabled=not self._detail, detail=self._detail,
                              slots=[self._render(slot) for slot in copies])

    def log_tail(self, slot: int, service: str, lines: int) -> LabPreviewLog:
        """The end of one service's log in ``slot``, redacted. ``KeyError`` for an unknown slot or service."""
        self._slot(slot)
        if service not in SERVICE_NAMES:
            raise KeyError(service)
        tail, truncated = read_log_tail(os.path.join(self._slot_log_dir(slot), f"{service}.log"), lines)
        return LabPreviewLog(slot=slot, service=service, lines=tail, truncated=truncated)

    # -- Actions -----------------------------------------------------------------------------------------------

    def take(self, change: LabChange, login: str) -> LabPreviewSlot:
        """Show ``change`` in a slot: the one already showing it, else a free one, else the one idle longest once
        idle :data:`SLOT_IDLE_TAKEOVER_SECONDS`. Returns at once; the slot's environments are made and its
        services started in the background, ``detail`` naming the step meanwhile.

        Raises :class:`PreviewFullError` when every slot shows another change in use, and
        :class:`PreviewUnavailableError` when the preview is off or the change has no working trees.
        """
        if change.status != "open":
            raise PreviewUnavailableError(
                f"Change {change.id} has no working trees to preview yet (it is {change.status}).")
        with self._lock:
            if self._detail:
                raise PreviewUnavailableError(self._detail)
            now = self._clock()
            usable = [slot for slot in self._slots if not slot.disabled]
            slot = next((s for s in usable if s.change is not None and s.change.id == change.id), None)
            if slot is not None:
                slot.change, slot.used_by, slot.last_used_at = change, login, now
                if slot.error:  # taking it again retries what failed
                    self._prepare_locked(slot, change, slot.supervisor)
                    slot.supervisor = None
                return self._render(copy.copy(slot))
            slot = next((s for s in usable if s.change is None), None)
            if slot is None:
                idle = [s for s in usable if now - (s.last_used_at or 0.0) >= SLOT_IDLE_TAKEOVER_SECONDS]
                if not idle:
                    raise PreviewFullError(self._full_message(usable, now))
                slot = min(idle, key=lambda s: s.last_used_at or 0.0)
                logger.info("Lab preview: slot %d idle since %.0f s, handing it from change %s to %s", slot.index,
                            now - (slot.last_used_at or 0.0), slot.change.id if slot.change else None, change.id)
            previous = slot.supervisor
            slot.change, slot.used_by, slot.started_at, slot.last_used_at = change, login, now, now
            slot.supervisor = None
            self._prepare_locked(slot, change, previous)
            return self._render(copy.copy(slot))

    def touch(self, slot: int, login: str) -> None:
        """``login`` still uses ``slot``: it is not idle. ``KeyError`` for an unknown slot."""
        with self._lock:
            target = self._slot(slot)
            if target.change is not None:
                target.used_by, target.last_used_at = login, self._clock()

    def after_build(self, change_id: str) -> None:
        """A build of ``change_id`` ended: in each slot showing it, update the environments and restart the services
        whose files changed (in the background)."""
        with self._lock:
            targets = [(slot, slot.generation, slot.change, slot.supervisor) for slot in self._slots
                       if slot.change is not None and slot.change.id == change_id and slot.supervisor is not None]
        for slot, generation, change, supervisor in targets:
            threading.Thread(target=self._refresh, args=(slot, generation, change, supervisor),
                             name=f"lab-preview-slot-{slot.index}", daemon=True).start()

    def release(self, change_id: str) -> None:
        """``change_id`` was dropped or closed: free the slots showing it and stop their services."""
        stopping = []
        with self._lock:
            for slot in self._slots:
                if slot.change is not None and slot.change.id == change_id:
                    stopping.append(slot.supervisor)
                    self._clear_locked(slot)
        for supervisor in stopping:
            if supervisor is not None:
                supervisor.stop()

    def restart(self, slot: int, service: Optional[str] = None) -> LabPreviewView:
        """Restart one service of ``slot``, or all; a slot whose preparation failed prepares again.

        ``KeyError`` for an unknown slot or service.
        """
        if service is not None and service not in SERVICE_NAMES:
            raise KeyError(service)
        with self._lock:
            target = self._slot(slot)
            supervisor = target.supervisor
            if supervisor is None and target.change is not None and target.error:
                self._prepare_locked(target, target.change, None)
        if supervisor is not None:
            if service is None:
                supervisor.stop()
                supervisor.start()
            else:
                supervisor.restart(service)
        return self.view()

    def stop(self) -> None:
        """Stop every slot's services (the lab is shutting down)."""
        stopping = []
        with self._lock:
            for slot in self._slots:
                stopping.append(slot.supervisor)
                self._clear_locked(slot)
        for supervisor in stopping:
            if supervisor is not None:
                supervisor.stop()

    # -- Internals ---------------------------------------------------------------------------------------------

    def _slot(self, index: int) -> _Slot:
        if not 0 <= index < len(self._slots):
            raise KeyError(index)
        return self._slots[index]

    def _slot_log_dir(self, index: int) -> str:
        return os.path.join(self._log_dir, f"slot-{index}")

    @staticmethod
    def _clear_locked(slot: _Slot) -> None:
        slot.generation += 1
        slot.change = slot.used_by = slot.started_at = slot.last_used_at = slot.supervisor = None
        slot.step = slot.error = ""

    def _prepare_locked(self, slot: _Slot, change: LabChange, previous: Optional[PreviewSupervisor]) -> None:
        """Start preparing ``slot`` for ``change`` in the background (pool lock held)."""
        slot.generation += 1
        slot.step, slot.error = "Waiting to prepare the environments", ""
        threading.Thread(target=self._prepare, args=(slot, slot.generation, change, previous),
                         name=f"lab-preview-slot-{slot.index}", daemon=True).start()

    def _set_step(self, slot: _Slot, generation: int, step: str) -> bool:
        """Record ``step``; False when the slot changed hands meanwhile (the caller stops)."""
        with self._lock:
            if slot.generation != generation:
                return False
            slot.step = step
            return True

    def _ensure_environments(self, slot: _Slot, generation: int, change: LabChange,
                             sandbox: PreviewSandbox) -> Optional[dict[str, RepoSpec]]:
        """Make the change's environments with its sandboxed install runner (R3); None when the slot changed hands.
        Raises :class:`LabEnvironmentError`."""
        repos = self._workspace.change_repos(change)
        runner = confined_command_runner(sandbox.mounts)
        for key in REPO_ORDER:
            if key not in repos:
                continue
            if not self._set_step(slot, generation, f"Preparing the {_REPO_LABELS[key]} environment"):
                return None
            self._environments.ensure(repos[key], runner=runner)
        return repos

    def _prepare(self, slot: _Slot, generation: int, change: LabChange,
                 previous: Optional[PreviewSupervisor]) -> None:
        try:
            if previous is not None:
                self._set_step(slot, generation, "Stopping the change shown before")
                previous.stop()
            sandbox = self._sandbox_for(change)
            repos = self._ensure_environments(slot, generation, change, sandbox)
            if repos is None or not self._set_step(slot, generation, "Starting the services"):
                return
            backend, frontend, agent = slot.ports
            services = default_services(repos, public_url=slot.public_url, backend_port=backend,
                                        frontend_port=frontend, agent_port=agent, proxy_port=self._proxy_port)
            supervisor = PreviewSupervisor(services, log_dir=self._slot_log_dir(slot.index), sandbox=sandbox)
            supervisor.start()
        except LabEnvironmentError as exc:
            self._failed(slot, generation, f"Could not prepare the environments: {exc}")
            return
        except Exception as exc:  # a background thread: report it on the slot, never lose it
            logger.exception("Lab preview: slot %d could not show change %s", slot.index, change.id)
            self._failed(slot, generation, f"Could not start the preview: {exc}")
            return
        with self._lock:
            current = slot.generation == generation
            if current:
                slot.supervisor, slot.step = supervisor, ""
        if not current:
            supervisor.stop()

    def _refresh(self, slot: _Slot, generation: int, change: LabChange, supervisor: PreviewSupervisor) -> None:
        try:
            if self._ensure_environments(slot, generation, change, self._sandbox_for(change)) is None:
                return  # the slot changed hands meanwhile
            restarted = supervisor.restart_changed()
        except LabEnvironmentError as exc:
            self._failed(slot, generation, f"Could not update the environments after the build: {exc}")
            return
        except Exception as exc:  # a background thread: report it on the slot, never lose it
            logger.exception("Lab preview: slot %d could not follow the build of change %s", slot.index, change.id)
            self._failed(slot, generation, f"Could not restart the preview after the build: {exc}")
            return
        with self._lock:
            if slot.generation == generation:
                slot.step, slot.error = "", ""
        if restarted:
            logger.info("Lab preview: slot %d restarted %s after a build of change %s", slot.index,
                        ", ".join(restarted), change.id)

    def _failed(self, slot: _Slot, generation: int, error: str) -> None:
        logger.warning("Lab preview: slot %d: %s", slot.index, error)
        with self._lock:
            if slot.generation == generation:
                slot.step, slot.error = "", error

    def _full_message(self, usable: Sequence[_Slot], now: float) -> str:
        shown = []
        for slot in usable:
            change = slot.change
            minutes = int((now - (slot.last_used_at or now)) // 60)
            shown.append(f'slot {slot.index + 1} shows "{change.title}" (@{change.owner}), '
                         f"last used by @{slot.used_by} {minutes} min ago")
        return (f"Every preview slot is in use: {'; '.join(shown)}. A slot nobody has used for "
                f"{SLOT_IDLE_TAKEOVER_SECONDS // 60} minutes can be taken.")

    def _render(self, slot: _Slot) -> LabPreviewSlot:
        """``slot`` (a copy) as the routes answer it; asks its supervisor, so never called under the pool lock."""
        enabled = not self._detail and not slot.disabled
        supervisor = slot.supervisor
        confined = supervisor.confined if supervisor is not None else sandbox_supported_platform()
        if supervisor is not None:
            services = supervisor.status()
        elif slot.error:
            services = [_service_state(name, "failed", slot.error) for name in SERVICE_NAMES]
        elif slot.step:
            services = [_service_state(name, "starting", slot.step) for name in SERVICE_NAMES]
        else:
            services = [_service_state(name, "stopped", "no change shown") for name in SERVICE_NAMES]
        detail = (slot.disabled or self._detail or slot.error or slot.step
                  or ("" if confined else _UNCONFINED_DETAIL))
        change = slot.change
        return LabPreviewSlot(
            enabled=enabled, url=slot.url if enabled else None, confined=confined, detail=detail,
            services=services, slot=slot.index, changeId=change.id if change else None,
            title=change.title if change else None, owner=change.owner if change else None, usedBy=slot.used_by,
            startedAt=slot.started_at, lastUsedAt=slot.last_used_at)
