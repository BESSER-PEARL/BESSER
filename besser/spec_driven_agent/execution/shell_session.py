"""One persistent shell per run: ``run_command`` behaves like one terminal.

Where the namespace sandbox applies, a run gets one long-lived bubblewrap
sandbox, started at its first shell command, with exactly the mount plan,
masks, environment and network policy of :func:`sandbox.sandboxed_command`.
Its init runs :data:`SUPERVISOR_SOURCE`, a small stdlib-only Python loop that
reads length-prefixed JSON requests on its stdin and answers on its stdout.
Both are anonymous pipes held only by this worker process, so no other run and
nothing on disk can reach the channel.

Each request runs one command in a fresh ``bash`` whose starting directory
and environment the harness supplies; an ``EXIT`` trap writes the directory
and exported variables it ended with, and the harness keeps them for the next
command. The harness, not the sandbox, holds that state, so a session that is
restarted (idle timeout, crash) keeps the run's ``cd`` and ``export``.
Processes are what the session adds: a command's background children live on
until the session ends. A command runs in its own process group, and a timeout
kills that group only.

Why a supervisor over one long-lived interactive bash with sentinel markers:
separate stdout/stderr, exit codes, the 8 MB cap and a per-command group kill
all fall out of ``subprocess``, and nothing a command prints can be mistaken
for a protocol message.

The session ends when the run ends (:meth:`ShellSession.close`), after
:data:`IDLE_ENV` seconds without a command, when its run folder is removed,
or when the worker exits: the supervisor exits on EOF on its channel, which
the kernel delivers when this process dies however it dies, and bubblewrap's
init takes every remaining process in the PID namespace with it. At most
:data:`MAX_ENV` sessions live per worker; past that a command runs in a
one-off sandbox with the same confinement, as before this module.

With a private network (``BESSER_LLM_SHELL_NETWORK``, see :mod:`.sandbox`)
the sandbox has its own network namespace, and a ``pasta`` process, a child of
this worker, gives it outbound NAT; the session ends if pasta does.

Without a sandbox (Windows, macOS, ``BESSER_LLM_SHELL_SANDBOX=off``) each
command is a plain subprocess as before; only the directory and environment
carry over.
"""

import atexit
import base64
import json
import logging
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

from besser.spec_driven_agent.execution.process import (
    FLOOD_NOTE,
    MAX_CAPTURE_BYTES,
    _safe_subprocess_env,
    decode_output,
    run_bounded,
)
from besser.spec_driven_agent.execution.sandbox import (
    NETWORK_WAIT_SECONDS,
    NETWORK_WAIT_SOURCE,
    SandboxUnavailable,
    read_child_pid,
    sandboxed_command,
    start_pasta,
    stop_process,
)

logger = logging.getLogger(__name__)

# Seconds without a command after which a run's session is torn down; 0: never.
IDLE_ENV = "BESSER_LLM_SHELL_SESSION_IDLE_SECONDS"
_IDLE_DEFAULT = 900
# Live sessions per worker process; 0 gives every command its own sandbox.
MAX_ENV = "BESSER_LLM_SHELL_SESSION_MAX"
_MAX_DEFAULT = 10

_START_TIMEOUT = 30
# Past the command's own timeout, which the supervisor enforces.
_RESPONSE_GRACE = 30
_MAX_RESPONSE_BYTES = 64 << 20
# Environment carried between commands; beyond it, the change is not kept.
_MAX_ENV_BYTES = 256 << 10
_MAX_ENV_VARS = 1000
# Set by bash itself; a stale copy would be wrong in the next command.
_SHELL_OWNED_VARS = frozenset({"_", "SHLVL", "PWD"})

# Runs before the command, in the same bash: on exit, write the directory and
# the exported variables as NUL-separated ``cwd``, ``NAME=value``... to $1.
BASH_PRELUDE = (
    'readonly __besser_state="$1"; shift; '
    "__besser_save() { local __besser_v IFS=$' \\t\\n'; "
    "{ printf '%s\\0' \"$PWD\"; for __besser_v in $(compgen -e); do "
    "printf '%s=%s\\0' \"$__besser_v\" \"${!__besser_v}\"; done; } "
    '> "$__besser_state" 2>/dev/null; }; trap __besser_save EXIT'
)

SUPERVISOR_SOURCE = r'''
import base64, json, os, selectors, signal, struct, subprocess, sys, tempfile, time
''' + NETWORK_WAIT_SOURCE + r'''

MAX_FRAME = 4 << 20
STATE_LIMIT = 1 << 20


def harden():
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        # Not dumpable: /proc/<pid>/fd (the channel) and ptrace are closed
        # to the commands, which share this process's uid.
        libc.prctl(4, 0, 0, 0, 0)
        # So `pkill python` / `killall python3` do not match the supervisor.
        libc.prctl(15, b"besser-shell", 0, 0, 0)
    except Exception:
        pass
    # A handler, not SIG_IGN: children exec with the default disposition.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: None)


def read_exact(fd, n):
    buf = b""
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv(fd):
    head = read_exact(fd, 4)
    if head is None:
        return None
    (size,) = struct.unpack(">I", head)
    if size > MAX_FRAME:
        return None
    body = read_exact(fd, size)
    return None if body is None else json.loads(body)


def send(fd, message):
    data = json.dumps(message).encode()
    data = struct.pack(">I", len(data)) + data
    while data:
        data = data[os.write(fd, data):]


def read_state(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None, None
    try:
        raw = b""
        while len(raw) <= STATE_LIMIT:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)
        try:
            os.remove(path)
        except OSError:
            pass
    parts = raw.split(b"\0")
    if len(raw) > STATE_LIMIT or len(parts) < 2:
        return None, None
    env = {}
    for item in parts[1:-1]:
        name, sep, value = item.partition(b"=")
        if sep and name:
            env[os.fsdecode(name)] = os.fsdecode(value)
    return os.fsdecode(parts[0]), env


class Job:
    def __init__(self, request, proc, out, err, state, note):
        self.request, self.proc, self.state, self.note = request, proc, state, note
        self.bufs = {out: bytearray(), err: bytearray()}
        self.open = {out, err}
        self.out, self.err = out, err
        self.cap = int(request.get("max_bytes") or 8000000)
        self.deadline = time.monotonic() + float(request.get("timeout") or 120)
        self.started = time.monotonic()
        self.flooded = self.timed_out = False


def main():
    workspace, prelude, bash = sys.argv[1], sys.argv[2], sys.argv[3]
    # A private network: wait up to argv[4] seconds for pasta to configure it.
    net = wait_for_network(float(sys.argv[4])) if len(sys.argv) > 4 else None
    harden()
    chan_in, chan_out = os.dup(0), os.dup(1)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    state_dir = tempfile.mkdtemp(prefix="besser-shell-")
    selector = selectors.DefaultSelector()
    selector.register(chan_in, selectors.EVENT_READ, "channel")
    job = None
    serial = 0

    def pump(fd):
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            selector.unregister(fd)
            os.close(fd)
            if job is not None:
                job.open.discard(fd)
            return
        # A background process's output after its command returned is dropped.
        if job is not None and fd in job.bufs:
            buf = job.bufs[fd]
            if len(buf) <= job.cap + (1 << 20):
                buf += chunk
            if len(buf) > job.cap:
                job.flooded = True

    send(chan_out, {"ready": True, "net": net})
    while True:
        wait = None
        if job is not None:
            wait = max(0.0, min(0.05, job.deadline - time.monotonic()))
        for key, _ in selector.select(wait):
            if key.data != "channel":
                pump(key.fd)
                continue
            request = recv(chan_in)
            if request is None or request.get("op") == "close":
                return
            cwd, note = request.get("cwd") or workspace, None
            if not os.path.isdir(cwd):
                note = "%s no longer exists; the command ran in the workspace root." % cwd
                cwd = workspace
            env = request.get("env")
            serial += 1
            os.makedirs(state_dir, exist_ok=True)
            state = os.path.join(state_dir, str(serial))
            out_r, out_w = os.pipe()
            err_r, err_w = os.pipe()
            try:
                proc = subprocess.Popen(
                    [bash, "-c", prelude + "\n" + request["command"], "bash", state],
                    cwd=cwd, env=dict(os.environ) if env is None else env,
                    stdin=null, stdout=out_w, stderr=err_w, start_new_session=True)
            except OSError as exc:
                for fd in (out_r, out_w, err_r, err_w):
                    os.close(fd)
                send(chan_out, {"id": request.get("id"), "exit_code": 127, "stdout": "",
                                "stderr": base64.b64encode(str(exc).encode()).decode(),
                                "timed_out": False, "flooded": False, "cwd": None,
                                "env": None, "note": note})
                break
            os.close(out_w)
            os.close(err_w)
            for fd in (out_r, err_r):
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ, "output")
            selector.unregister(chan_in)
            job = Job(request, proc, out_r, err_r, state, note)
            break
        if job is None:
            continue
        code = job.proc.poll()
        if code is None:
            if not job.flooded and time.monotonic() < job.deadline:
                continue
            job.timed_out = not job.flooded
            # The command's own group only: the session and processes earlier
            # commands left running are untouched.
            try:
                os.killpg(job.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            code = job.proc.wait()
        # Everything bash wrote before it exited is already in the pipes: at
        # most a pipe buffer each. Bounded, since a background writer may be
        # refilling them.
        for fd in list(job.open):
            for _ in range(16):
                before = len(job.bufs[fd])
                pump(fd)
                if fd not in job.open or len(job.bufs[fd]) == before:
                    break
        cwd, env = read_state(job.state)
        send(chan_out, {
            "id": job.request.get("id"),
            "exit_code": None if job.timed_out else code,
            "stdout": base64.b64encode(bytes(job.bufs[job.out])).decode(),
            "stderr": base64.b64encode(bytes(job.bufs[job.err])).decode(),
            "timed_out": job.timed_out, "flooded": job.flooded,
            "cwd": cwd, "env": env, "note": job.note,
            "seconds": round(time.monotonic() - job.started, 3),
        })
        job = None
        selector.register(chan_in, selectors.EVENT_READ, "channel")


try:
    main()
except BrokenPipeError:
    pass  # the worker went away mid-reply; exiting ends the sandbox
'''


class _SessionDied(RuntimeError):
    """The supervisor went away (killed by a command, or torn down)."""


def _real(path: str) -> str:
    """realpath without the ``\\\\?\\`` prefix Windows sometimes adds, which
    would make an inside-the-workspace path compare as outside it."""
    real = os.path.realpath(path)
    return real[4:] if real.startswith("\\\\?\\") and not real.startswith("\\\\?\\UNC\\") else real


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, os.environ.get(name), default)
        return default


# --------------------------------------------------------------------------- #
# Spawning: bwrap's --die-with-parent is a PR_SET_PDEATHSIG, which fires when
# the THREAD that forked it exits. Tools run on a per-turn ThreadPoolExecutor,
# so a session forked there would be killed at the end of its first turn.
# Every session is forked from this one thread, which lives as long as the
# worker process.
# --------------------------------------------------------------------------- #
_spawn_queue: "queue.SimpleQueue" = queue.SimpleQueue()
_spawner: threading.Thread | None = None
_spawner_guard = threading.Lock()


def _spawner_loop() -> None:
    while True:
        factory, future = _spawn_queue.get()
        try:
            future.set_result(factory())
        except BaseException as exc:  # handed to the caller
            future.set_exception(exc)


def _spawn(factory):
    global _spawner
    with _spawner_guard:
        if _spawner is None or not _spawner.is_alive():
            _spawner = threading.Thread(target=_spawner_loop, name="besser-shell-spawner",
                                        daemon=True)
            _spawner.start()
    future: Future = Future()
    _spawn_queue.put((factory, future))
    return future.result()


class _SessionProcess:
    """One bubblewrap sandbox running the supervisor, its channel, and the
    pasta process giving it a private network (``private_network``)."""

    def __init__(self, argv: list[str], cwd: str, private_network: bool = False):
        self._stderr = tempfile.TemporaryFile()
        self.pasta: subprocess.Popen | None = None
        info_r = info_w = None
        if private_network:
            info_r, info_w = os.pipe()
            # The supervisor's argv ends the plan; its extra arg is the network wait.
            argv = [argv[0], "--info-fd", str(info_w), *argv[1:], str(NETWORK_WAIT_SECONDS)]
        try:
            self.proc = _spawn(lambda: subprocess.Popen(
                argv, cwd=cwd, env=_safe_subprocess_env(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._stderr, start_new_session=True,
                pass_fds=() if info_w is None else (info_w,),
            ))
        except OSError as exc:
            self._stderr.close()
            if info_r is not None:
                os.close(info_r)
            raise SandboxUnavailable(f"bwrap could not be executed: {exc}") from None
        finally:
            if info_w is not None:
                os.close(info_w)
        self._serial = 0
        if private_network:
            try:
                pid = read_child_pid(info_r, _START_TIMEOUT)
                # Also from the spawner thread; pasta is waited on by close().
                self.pasta = _spawn(lambda: start_pasta(pid, self._stderr))
            except SandboxUnavailable as exc:
                self.close()
                raise SandboxUnavailable(f"the shell session's network did not start: "
                                         f"{exc.detail}") from None
            finally:
                os.close(info_r)
        try:
            hello = self._exchange(None, _START_TIMEOUT)
        except _SessionDied:
            hello = None
        if not (isinstance(hello, dict) and hello.get("ready") is True):
            detail = self._startup_detail()
            self.close()
            raise SandboxUnavailable(f"the shell session did not start: {detail}")
        # Fail closed: never a session without the network it was planned with.
        if private_network and (hello.get("net") is not True or not self.network_alive()):
            detail = (f"pasta exited with status {self.pasta.returncode}"
                      if not self.network_alive()
                      else f"no interface after {NETWORK_WAIT_SECONDS} s")
            self.close()
            raise SandboxUnavailable(
                f"the shell session's private network did not come up ({detail})")

    def _startup_detail(self) -> str:
        try:
            self.proc.wait(timeout=5)
            self._stderr.seek(0)
            lines = self._stderr.read(8192).decode("utf-8", "replace").strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        bwrap = [line for line in lines if line.startswith("bwrap: ")]
        return (bwrap or lines[-1:] or [f"exit status {self.proc.returncode}"])[0]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def network_alive(self) -> bool:
        return self.pasta is None or self.pasta.poll() is None

    def _read(self, size: int) -> bytes:
        data = self.proc.stdout.read(size)
        if data is None or len(data) < size:
            raise _SessionDied("channel closed")
        return data

    def _exchange(self, message: dict | None, timeout: float):
        """Send ``message`` (None: just read) and return the reply.

        A supervisor that does not answer in time is killed, which closes the
        channel and turns the blocked read into :class:`_SessionDied`.
        """
        watchdog = threading.Timer(timeout, self.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            if message is not None:
                data = json.dumps(message).encode()
                self.proc.stdin.write(struct.pack(">I", len(data)) + data)
                self.proc.stdin.flush()
            (size,) = struct.unpack(">I", self._read(4))
            if size > _MAX_RESPONSE_BYTES:
                self.kill()
                raise _SessionDied(f"oversized reply ({size} bytes)")
            reply = json.loads(self._read(size))
        except (OSError, ValueError, struct.error) as exc:
            raise _SessionDied(str(exc)) from None
        finally:
            watchdog.cancel()
        if not isinstance(reply, dict):
            self.kill()
            raise _SessionDied("malformed reply")
        return reply

    def run(self, command: str, cwd: str, env: dict | None, timeout: float) -> dict:
        self._serial += 1
        reply = self._exchange({
            "id": self._serial, "command": command, "cwd": cwd, "env": env,
            "timeout": timeout, "max_bytes": MAX_CAPTURE_BYTES,
        }, timeout + _RESPONSE_GRACE)
        if reply.get("id") != self._serial:
            self.kill()
            raise _SessionDied("reply for another request")
        return reply

    def kill(self) -> None:
        for proc in (self.proc, self.pasta):
            try:
                if proc is not None:
                    proc.kill()
            except OSError:
                pass

    def close(self) -> None:
        """EOF on the channel ends the supervisor, and bwrap's init with it;
        pasta is then stopped and reaped."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            # Busy with a command; SIGKILL on bwrap, --die-with-parent does the rest.
            self.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error("shell session pid %s did not exit", self.proc.pid)
        stop_process(self.pasta)
        for stream in (self.proc.stdout, self._stderr):
            try:
                stream.close()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Registry: the per-worker cap, the idle reaper and teardown by run folder.
# --------------------------------------------------------------------------- #
_registry_lock = threading.Lock()
_live: "set[ShellSession]" = set()
_reaper: threading.Thread | None = None


def _acquire_slot(session: "ShellSession") -> bool:
    global _reaper
    with _registry_lock:
        if len(_live) >= _env_int(MAX_ENV, _MAX_DEFAULT):
            return False
        _live.add(session)
        if _reaper is None or not _reaper.is_alive():
            _reaper = threading.Thread(target=_reap_loop, name="besser-shell-reaper", daemon=True)
            _reaper.start()
        return True


def _release_slot(session: "ShellSession") -> None:
    with _registry_lock:
        _live.discard(session)


def _reap_loop() -> None:
    while True:
        time.sleep(1.0)
        idle = _env_int(IDLE_ENV, _IDLE_DEFAULT)
        with _registry_lock:
            sessions = list(_live)
        for session in sessions:
            try:
                session.reap_if_idle(idle)
            except Exception:
                logger.exception("shell session reaper failed")


def live_session_count() -> int:
    with _registry_lock:
        return len(_live)


def close_sessions_for(workspace: str) -> None:
    """End the session of the run whose folder is ``workspace``."""
    target = os.path.normcase(_real(workspace))
    with _registry_lock:
        sessions = [s for s in _live if os.path.normcase(s.workspace) == target]
    for session in sessions:
        session.close()


@atexit.register
def _close_all() -> None:
    with _registry_lock:
        sessions = list(_live)
    for session in sessions:
        session.close()


# --------------------------------------------------------------------------- #
# The per-run handle the tool executor holds.
# --------------------------------------------------------------------------- #
@dataclass
class ShellResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    # False when background processes did not survive this command.
    persistent: bool = False
    notes: list[str] = field(default_factory=list)


class ShellSession:
    """The run's shell: its directory, environment and (sandboxed) processes."""

    def __init__(self, workspace: str):
        self.workspace = _real(workspace)
        self.cwd = self.workspace
        self.env: dict[str, str] | None = None
        self._proc: _SessionProcess | None = None
        self._closed = False
        self._busy = False
        self._last_used = time.monotonic()
        self._pending_note: str | None = None
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    def _detach(self, note: str | None) -> "_SessionProcess | None":
        """Take the live process out of this session (caller holds _state_lock)."""
        proc, self._proc = self._proc, None
        if proc is not None:
            _release_slot(self)
            if note and not self._closed:
                self._pending_note = note
        return proc

    def close(self) -> None:
        """Tear the session down for good: every process in it is killed."""
        with self._state_lock:
            self._closed = True
            proc = self._detach(None)
        if proc is not None:
            proc.close()
            logger.info("shell session closed for %s", self.workspace)

    def stop_processes(self, reason: str) -> None:
        """End the live sandbox but keep the session (cwd, env) usable; the
        next command starts a fresh sandbox and is told ``reason``."""
        with self._state_lock:
            proc = self._detach(reason + "; processes it was running were stopped. "
                                "This command started a new session.")
        if proc is not None:
            proc.close()

    def reap_if_idle(self, idle_seconds: int) -> None:
        with self._state_lock:
            if self._proc is None or self._busy:
                return
            if self._proc.alive() and (
                    idle_seconds <= 0 or time.monotonic() - self._last_used < idle_seconds):
                return
            reason = ("the shell session was closed after %d s without a command" % idle_seconds
                      if self._proc.alive() else "the shell session stopped")
            proc = self._detach(reason + "; processes it was running were stopped. "
                                "This command started a new session.")
        proc.close()
        logger.info("shell session reaped for %s (%s)", self.workspace, reason)

    # -- commands ----------------------------------------------------------
    def run(self, command: str, *, working_dir: str | None, timeout: float,
            adopt_state: bool = True) -> ShellResult:
        """Run one command; ``working_dir`` runs it elsewhere without moving the shell.

        Raises :class:`SandboxUnavailable` when the command must not run.
        """
        with self._run_lock:
            notes: list[str] = []
            start = working_dir or self.cwd
            if not os.path.isdir(start):
                notes.append(f"{start} no longer exists; the command ran in the workspace root.")
                start = self.cwd = self.workspace
            plan = sandboxed_command(self._supervisor_argv(), workspace=self.workspace,
                                     cwd=self.workspace, shell_network=True)
            if plan.sandboxed:
                result, cwd, env = self._run_sandboxed(plan, command, start, timeout, notes)
            else:
                result, cwd, env = self._run_unconfined(command, start, timeout)
            if adopt_state:
                self._adopt(cwd, env, follow_cwd=working_dir is None, notes=result.notes)
            self._last_used = time.monotonic()
            result.notes[:0] = notes
            return result

    def _supervisor_argv(self) -> list[str]:
        bash = shutil.which("bash") or "/bin/bash"
        return [sys.executable, "-I", "-S", "-c", SUPERVISOR_SOURCE,
                self.workspace, BASH_PRELUDE, bash]

    def _run_sandboxed(self, plan, command, cwd, timeout, notes):
        dead = None
        with self._state_lock:
            proc = self._proc
            if proc is not None and not (proc.alive() and proc.network_alive()):
                what = "stopped" if not proc.alive() else "lost its network"
                dead = self._detach(f"the shell session {what}; processes it was running "
                                    "were stopped. This command started a new session.")
                proc = None
            if self._pending_note:
                notes.append(self._pending_note)
                self._pending_note = None
            persistent = proc is not None or (not self._closed and _acquire_slot(self))
            self._busy = True
        if dead is not None:
            dead.close()
        try:
            if proc is None:
                try:
                    proc = _SessionProcess(plan.argv, self.workspace,
                                           private_network=plan.private_network)
                except BaseException:
                    if persistent:
                        _release_slot(self)
                    raise
                if persistent:
                    with self._state_lock:
                        if self._closed:  # closed while starting
                            persistent = False
                            _release_slot(self)
                        else:
                            self._proc = proc
                else:
                    notes.append(
                        "This command ran in a one-off sandbox (the server's shell sessions "
                        "are all in use), so processes it left running were stopped when it "
                        "returned.")
            try:
                reply = proc.run(command, cwd, self.env, timeout)
            except _SessionDied as exc:
                logger.warning("shell session for %s ended mid-command: %s", self.workspace, exc)
                with self._state_lock:
                    if self._proc is proc:
                        self._detach(None)
                if persistent:
                    proc.close()
                return ShellResult(None, "", (
                    "The shell session ended while this command ran (its supervisor was "
                    "stopped); every process in it was stopped. The next command starts a "
                    "new session.")), None, None
            finally:
                if not persistent:
                    proc.close()
        finally:
            with self._state_lock:
                self._busy = False
        stdout = decode_output(base64.b64decode(reply.get("stdout") or ""))
        stderr = decode_output(base64.b64decode(reply.get("stderr") or ""))
        if reply.get("flooded"):
            stderr += FLOOD_NOTE
        result = ShellResult(reply.get("exit_code"), stdout, stderr,
                             timed_out=bool(reply.get("timed_out")), persistent=persistent)
        if reply.get("note"):
            result.notes.append(str(reply["note"]))
        return result, reply.get("cwd"), reply.get("env")

    def _run_unconfined(self, command, cwd, timeout):
        env = self.env if self.env is not None else _safe_subprocess_env()
        state_dir = tempfile.mkdtemp(prefix="besser-shell-state-")
        try:
            args, shell, reader = _unconfined_invocation(command, state_dir)
            try:
                completed = run_bounded(args, timeout=timeout, cwd=cwd, env=env, shell=shell)
            except subprocess.TimeoutExpired as exc:
                return ShellResult(None, exc.output or "", exc.stderr or "", timed_out=True), \
                    None, None
            new_cwd, new_env, code = reader()
            returncode = completed.returncode if code is None else code
            return ShellResult(returncode, completed.stdout or "", completed.stderr or ""), \
                new_cwd, new_env
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def _adopt(self, cwd, env, *, follow_cwd: bool, notes: list[str]) -> None:
        if isinstance(env, dict):
            cleaned = {str(k): str(v) for k, v in env.items()
                       if k and str(k) not in _SHELL_OWNED_VARS}
            size = sum(len(k) + len(v) for k, v in cleaned.items())
            if len(cleaned) > _MAX_ENV_VARS or size > _MAX_ENV_BYTES:
                notes.append("The environment grew past its limit, so this command's "
                             "changes to it were not kept.")
            else:
                self.env = cleaned
        if not follow_cwd or not isinstance(cwd, str) or not cwd:
            return
        real = _real(cwd)
        base = os.path.normcase(self.workspace)
        inside = os.path.normcase(real) == base or os.path.normcase(real).startswith(
            base.rstrip(os.sep) + os.sep)
        if inside and os.path.isdir(real):
            self.cwd = real
        else:
            self.cwd = self.workspace
            notes.append(f"The command ended in {cwd}, outside the run workspace; the next "
                         "command starts in the workspace root.")

    def relative_cwd(self) -> str:
        rel = os.path.relpath(self.cwd, self.workspace).replace("\\", "/")
        return "." if rel == "." else rel


def _parse_state(raw: bytes, sep: bytes):
    parts = raw.split(sep)
    if len(parts) < 2:
        return None, None
    env = {}
    for item in parts[1:]:
        name, found, value = item.partition(b"=")
        if found and name:
            env[os.fsdecode(name)] = os.fsdecode(value)
    return os.fsdecode(parts[0]), env


def _unconfined_invocation(command: str, state_dir: str):
    """``(args, shell, reader)`` for one unsandboxed command.

    ``reader()`` returns ``(cwd, env, exit code or None)`` from what the
    command left behind, or Nones when it recorded nothing (it ``exec``-ed,
    replaced the ``EXIT`` trap, or was killed).
    """
    if os.name == "nt":
        return _windows_invocation(command, state_dir)
    state = os.path.join(state_dir, "state")
    bash = shutil.which("bash")
    if not bash:
        return command, True, lambda: (None, None, None)

    def reader():
        try:
            with open(state, "rb") as handle:
                raw = handle.read(1 << 20)
        except OSError:
            return None, None, None
        cwd, env = _parse_state(raw.rstrip(b"\0"), b"\0")
        return cwd, env, None

    return [bash, "-c", BASH_PRELUDE + "\n" + command, "bash", state], False, reader


def _windows_invocation(command: str, state_dir: str):
    """cmd.exe keeps its own semantics; a trailer records the outcome.

    ``&`` binds loosest in cmd, so the trailer runs after the whole command
    line. ``%^ERRORLEVEL%`` survives the first expansion and ``call`` expands
    it, which is the command's exit status.
    """
    rc, cwd_file, env_file = (os.path.join(state_dir, name) for name in ("rc", "cwd", "env"))
    # A trailing ^ would escape the trailer's &.
    if any(ch in state_dir for ch in ' "&|<>^%!()') or command.endswith("^"):
        return command, True, lambda: (None, None, None)
    # No space before "&": cmd keeps it, so `set X=1 &` would set "1 ".
    trailer = (f"& (call echo %^ERRORLEVEL%)>{rc} & (cd)>{cwd_file} & (set)>{env_file}")

    def reader():
        try:
            with open(rc, "rb") as handle:
                code = int(handle.read().strip() or b"x")
            with open(cwd_file, "rb") as handle:
                cwd = handle.read().decode("oem", "replace").strip()
            with open(env_file, "rb") as handle:
                lines = handle.read().decode("oem", "replace").splitlines()
        except (OSError, ValueError):
            return None, None, None
        env = {}
        for line in lines:
            name, found, value = line.partition("=")
            if found and name:
                env[name] = value
        return cwd or None, env or None, code

    return command + trailer, True, reader
