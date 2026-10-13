"""Namespace sandbox for the model-authored shell (``run_command``).

Two confinement holes in an unsandboxed shared worker:

* the cwd lock constrains tool *arguments* (``path``, ``working_dir``), not
  the command string, so ``cd /workspace/runs/<other_run>`` could read and
  write a concurrent user's run;
* the worker is root in a single shared PID namespace, so ``/proc/1/environ``
  exposes the container's own tokens past the ``_safe_subprocess_env`` scrub.

Both close with the same measure: run the command in a mount namespace where
only this run's directory is bound, and a PID namespace where PID 1 is the
sandbox's own init. Per-run containers would be the stronger isolation.

bubblewrap over nsjail: ``bubblewrap`` is one package in Debian main, whereas
nsjail has to be built from source (protobuf, libnl, bison, flex), growing the
image. Both provide the namespaces.

Bubblewrap needs an unprivileged user namespace, and Docker's default seccomp
profile gates ``unshare`` / ``mount`` / ``pivot_root`` on CAP_SYS_ADMIN — so the
worker runs with ``seccomp=unconfined`` (plus ``apparmor=unconfined``, since the
docker-default AppArmor profile carries ``deny mount``). Neither grants a
capability. See ``docker-compose.prod.yml``.

Fail closed: when the sandbox is mandatory and cannot start, ``run_command``
refuses. It never falls back to an unconfined shell. Phase 3 validators that
execute generated code go through :func:`run_confined` under the same policy,
with the network unshared: a check that cannot be confined is skipped.

A shell session's network (:data:`SHELL_NETWORK_ENV`): with ``pasta`` (passt)
and ``/dev/net/tun`` it gets a network namespace of its own with outbound NAT
and no forwarding in either direction, so its ``localhost`` is private and the
worker's own listeners and other runs' servers are out of reach. Otherwise it
shares the worker's namespace.
"""

import ipaddress
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from besser.spec_driven_agent.execution.process import _safe_subprocess_env, run_bounded

logger = logging.getLogger(__name__)


# auto (default): sandbox mandatory wherever the platform can provide one.
# off: explicit operator opt-out, for a Linux host whose kernel forbids
# unprivileged user namespaces. Never set this on a multi-tenant host.
SANDBOX_POLICY_ENV = "BESSER_LLM_SHELL_SANDBOX"

# Mounted fresh rather than bound: a private /proc (so --unshare-pid actually
# hides the host PID 1) and a private /dev, plus scratch space nothing outside
# the sandbox should see.
# /dev/shm is listed because bwrap's --dev builds a minimal device tree that
# does not include it, and POSIX shared memory (multiprocessing, some npm and
# browser toolchains) fails without it.
_TMPFS_PATHS = ("/tmp", "/run", "/var/tmp", "/dev/shm")

# Top-level entries the mount plan never binds from the outside.
_VIRTUAL_TOPLEVEL = frozenset({"/proc", "/dev", "/tmp", "/run"})

# Every user's run telemetry. It lives under /app, which is otherwise bound
# read-only, so an empty tmpfs is mounted over it. Same default as telemetry.py.
_TELEMETRY_DIR_ENV = "BESSER_TELEMETRY_DIR"
_TELEMETRY_DIR_DEFAULT = "/app/telemetry"
# Same for the incident log: it names other runs, and a run id fetches output.
_INCIDENT_DIR_ENV = "BESSER_INCIDENT_LOG_DIR"
_INCIDENT_DIR_DEFAULT = "/app/incidents"

_SANDBOX_TIMEOUT = 30

# auto (default): private when the network selftest passes, else shared with
# one warning. private: unavailable -> commands refused. shared: the worker's
# own namespace.
SHELL_NETWORK_ENV = "BESSER_LLM_SHELL_NETWORK"
# pasta answers DNS sent here and forwards it to the container's nameserver,
# which the sandbox cannot reach when it is a loopback address (Docker's
# embedded 127.0.0.11).
_DNS_FORWARD_ADDR = "169.254.1.53"
_RESOLV_CONF = "/etc/resolv.conf"
_TUN_DEVICE = "/dev/net/tun"
# How long a sandbox waits for pasta to configure its interface.
NETWORK_WAIT_SECONDS = 2.0

# True once an interface other than lo has a route: pasta finished
# --config-net. /proc, not /sys: the sandbox's /sys is the container's.
# Needs `time` imported by the including source.
NETWORK_WAIT_SOURCE = r'''
def wait_for_network(seconds):
    deadline = time.monotonic() + seconds
    while True:
        for path, col in (("/proc/net/route", 0), ("/proc/net/ipv6_route", -1)):
            try:
                with open(path) as handle:
                    if any(line.split()[col] not in ("lo", "Iface")
                           for line in handle if line.strip()):
                        return True
            except OSError:
                pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
'''

_selftest_result: tuple[bool, str] | None = None
_network_selftest_result: tuple[bool, str] | None = None
_unconfined_warned = False
_shared_network_warned = False


class SandboxUnavailable(RuntimeError):
    """The sandbox could not be prepared. The command must NOT be run.

    ``str()`` is model-safe: it reaches findings and tool results the model
    reads. The cause, including the operator override, is ``detail`` and is
    logged here once; the model tried to set the override when it saw it.
    """

    def __init__(self, detail: str) -> None:
        super().__init__("the shell sandbox is unavailable on this server")
        self.detail = detail
        logger.error("Shell sandbox unavailable: %s", detail)


@dataclass(frozen=True)
class SandboxedCommand:
    """How to hand one model-authored command to :func:`subprocess.run`."""

    argv: "list[str] | str"
    use_shell: bool
    mode: str  # "bwrap" | "unconfined-platform" | "unconfined-override"
    # bwrap unshares the network; the caller attaches pasta (shell sessions).
    private_network: bool = False

    @property
    def sandboxed(self) -> bool:
        return self.mode == "bwrap"

    def startup_error(self, returncode: int, stderr: str) -> str | None:
        """The sandbox's own failure to start, as opposed to the command's.

        bwrap reports setup failures as a ``bwrap: ...`` line and exits
        non-zero without ever running the command. Reported as a sandbox
        error so the model is not sent chasing a compile error that never
        happened.
        """
        if not self.sandboxed or returncode == 0:
            return None
        first = (stderr or "").strip().splitlines()[:1]
        if first and first[0].startswith("bwrap: "):
            return first[0]
        return None


def _policy() -> str:
    raw = (os.environ.get(SANDBOX_POLICY_ENV) or "auto").strip().lower()
    if raw not in {"auto", "off"}:
        logger.warning(
            "%s=%r is not a known policy; using 'auto'", SANDBOX_POLICY_ENV, raw,
        )
        return "auto"
    return raw


def sandbox_supported_platform() -> bool:
    """True where a namespace sandbox is available in principle."""
    return sys.platform.startswith("linux")


def _warn_unconfined(reason: str) -> None:
    global _unconfined_warned
    if _unconfined_warned:
        return
    _unconfined_warned = True
    logger.warning(
        "Model-authored shell commands are running UNSANDBOXED: %s. Sibling "
        "run workspaces and /proc/1 are in view of every command.", reason,
    )


def _within(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _top_level(path: str) -> str:
    """``/workspace/runs/x`` -> ``/workspace``; the entry we must not bind."""
    parts = os.path.realpath(path).lstrip("/").split("/", 1)
    return "/" + parts[0]


def sandbox_home(workspace: str) -> str:
    """The run's own writable ``$HOME``, beside its workspace.

    /usr/local and /root stay read-only: shared between runs, a ``pip
    install`` in one run would land in every later run and in the worker
    itself. pip falls back to a user install under this ``$HOME``, and the
    npm / cargo caches follow it. Outside the workspace so it is never
    packaged, pushed or snapshotted; the ``besser_llm_`` prefix it inherits
    puts it under the run-root cleanup.
    """
    return workspace.rstrip("/\\") + ".sandbox-home"


def _mount_args(workspace: str, writable: "list[str] | None" = None) -> list[str]:
    """Bind the container filesystem minus this run's siblings.

    Everything the toolchain needs stays visible, read-only — a compile check,
    ``pip show`` and the app's own test suite must keep working — but the
    top-level directory the run workspaces live under is left out entirely.
    ``writable`` (default: the workspace) is bound back read-write, plus the
    run's :func:`sandbox_home`.
    """
    writable = [workspace] if writable is None else list(writable)
    hidden = {_top_level(path) for path in (workspace, *writable)}
    args: list[str] = []
    for entry in sorted(os.listdir("/")):
        path = "/" + entry
        if path in _VIRTUAL_TOPLEVEL or path in hidden:
            continue
        if os.path.islink(path):
            # Debian's merged /usr: /bin, /lib, /lib64 are symlinks into /usr.
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    args += ["--proc", "/proc", "--dev", "/dev"]
    for path in _TMPFS_PATHS:
        args += ["--tmpfs", path]
    for env_name, default in ((_TELEMETRY_DIR_ENV, _TELEMETRY_DIR_DEFAULT),
                              (_INCIDENT_DIR_ENV, _INCIDENT_DIR_DEFAULT)):
        masked = os.path.realpath(os.environ.get(env_name) or default)
        if os.path.isdir(masked) and not any(_within(masked, h) for h in hidden):
            args += ["--tmpfs", masked]
    home = sandbox_home(workspace)
    args += ["--bind", home, home, "--setenv", "HOME", home]
    # rustup finds its toolchains under $HOME/.rustup; keep the image's
    # (read-only) now that $HOME moved.
    rustup = os.environ.get("RUSTUP_HOME") or os.path.join(
        os.path.expanduser("~"), ".rustup")
    if os.path.isdir(rustup) and not any(_within(rustup, h) for h in hidden):
        args += ["--setenv", "RUSTUP_HOME", rustup]
    for path in writable:
        args += ["--bind", path, path]
    return args


def _isolation_args(network: bool = True) -> list[str]:
    return [
        # Validators get no network: generated code must not reach the
        # worker's neighbours or the internet while it is being checked.
        # A private shell session unshares it too, then pasta attaches.
        *([] if network else ["--unshare-net"]),
        # A user namespace is what buys the rest without any capability.
        "--unshare-user",
        # The point of the exercise: PID 1 becomes bwrap's own init, so
        # /proc/1/environ is the sandbox's scrubbed environment.
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        # The 120s timeout kills bwrap; this takes the whole PID namespace
        # with it instead of orphaning the command's children.
        "--die-with-parent",
        "--new-session",
    ]


def _selftest(bwrap: str) -> tuple[bool, str]:
    """Can this kernel / container actually give us the namespaces?

    Cached: the answer cannot change inside one worker process, and every
    run_command would otherwise pay for it.
    """
    global _selftest_result
    if _selftest_result is not None:
        return _selftest_result
    probe = [
        bwrap, *_isolation_args(),
        "--ro-bind", "/", "/", "--proc", "/proc", "--", "/bin/true",
    ]
    try:
        result = subprocess.run(
            probe, capture_output=True, text=True, timeout=_SANDBOX_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _selftest_result = (False, f"bwrap could not be executed: {exc}")
        return _selftest_result
    if result.returncode == 0:
        _selftest_result = (True, "")
    else:
        detail = (result.stderr or "").strip().splitlines()[-1:] or ["no output"]
        _selftest_result = (False, detail[0])
    return _selftest_result


def sandbox_selftest_error() -> str | None:
    """``None`` when a sandbox can be started here, else why it cannot.

    Operator check after a deploy::

        docker exec besser-wme-smartgen python -c \\
          "from besser.spec_driven_agent.execution.sandbox import sandbox_selftest_error as e; print(e() or 'sandbox OK')"
    """
    if not sandbox_supported_platform():
        return f"no namespace sandbox on {sys.platform}"
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "bubblewrap (bwrap) is not installed"
    ok, detail = _selftest(bwrap)
    return None if ok else detail


def _network_mode() -> str:
    raw = (os.environ.get(SHELL_NETWORK_ENV) or "auto").strip().lower()
    if raw not in {"auto", "private", "shared"}:
        logger.warning("%s=%r is not a known mode; using 'auto'", SHELL_NETWORK_ENV, raw)
        return "auto"
    return raw


def _container_nameserver() -> str | None:
    try:
        with open(_RESOLV_CONF, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    return parts[1]
    except OSError:
        pass
    return None


def _loopback_nameserver() -> str | None:
    """The container's nameserver when the sandbox cannot reach it directly."""
    server = _container_nameserver()
    try:
        loopback = server and ipaddress.ip_address(server.split("%")[0]).is_loopback
    except ValueError:
        return None
    return server if loopback else None


def _sandbox_resolv_conf() -> str:
    """A resolv.conf naming pasta's DNS forwarder, bound over the sandbox's."""
    path = os.path.join(tempfile.gettempdir(), "besser-sandbox-resolv.conf")
    if not os.path.isfile(path):
        fd, tmp = tempfile.mkstemp(prefix="besser-sandbox-resolv-", dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as handle:
            handle.write(f"nameserver {_DNS_FORWARD_ADDR}\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    return path


def pasta_argv(pasta: str, pid: int, dns_host: str | None = None) -> list[str]:
    """pasta for the network namespace of ``pid``: outbound NAT only.

    Every forwarding option is pinned to ``none``: -t/-u would publish sandbox
    ports on the worker, -T/-U would expose the worker's loopback to the
    sandbox, and without --no-map-gw the gateway address maps to the worker.
    """
    argv = [
        pasta, "-f", "-q", "--config-net",
        # As root, pasta would drop to nobody, which cannot join the userns.
        "--runas", "0:0",
        "-t", "none", "-u", "none", "-T", "none", "-U", "none",
        "--no-map-gw",
    ]
    if dns_host:
        argv += ["--dns-forward", _DNS_FORWARD_ADDR, "--dns-host", dns_host]
    return [*argv, str(pid)]


def start_pasta(pid: int, stderr) -> subprocess.Popen:
    """Attach pasta to ``pid``'s network namespace in the foreground; the
    caller waits on it (pasta's daemon mode leaves zombies under PID 1)."""
    pasta = shutil.which("pasta")
    if not pasta:
        raise SandboxUnavailable("pasta (passt) is not installed")
    try:
        return subprocess.Popen(
            pasta_argv(pasta, pid, _loopback_nameserver()), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=stderr, env=_safe_subprocess_env(),
        )
    except OSError as exc:
        raise SandboxUnavailable(f"pasta could not be executed: {exc}") from None


def read_child_pid(fd: int, timeout: float) -> int:
    """The sandbox's pid, from bwrap's ``--info-fd`` JSON."""
    deadline = time.monotonic() + timeout
    buf = b""
    while True:
        try:
            return int(json.loads(buf)["child-pid"])
        except (ValueError, KeyError, TypeError):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxUnavailable("bwrap did not report its sandbox pid in time")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            raise SandboxUnavailable("bwrap exited before reporting its sandbox pid")
        buf += chunk


def stop_process(proc: "subprocess.Popen | None") -> None:
    """SIGTERM, then SIGKILL; always reaped."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except OSError:
        pass


def _network_selftest(bwrap: str) -> tuple[bool, str]:
    """Can a sandbox get a private network here? Cached like :func:`_selftest`."""
    global _network_selftest_result
    if _network_selftest_result is not None:
        return _network_selftest_result
    if not shutil.which("pasta"):
        _network_selftest_result = (False, "pasta (passt) is not installed")
        return _network_selftest_result
    if not os.path.exists(_TUN_DEVICE):
        _network_selftest_result = (False, f"{_TUN_DEVICE} is not available")
        return _network_selftest_result
    probe = ("import sys, time\n" + NETWORK_WAIT_SOURCE +
             f"sys.exit(0 if wait_for_network({NETWORK_WAIT_SECONDS}) else 3)\n")
    info_r, info_w = os.pipe()
    proc = pasta = None
    try:
        with tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(
                    [bwrap, "--info-fd", str(info_w), *_isolation_args(network=False),
                     "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                     "--", sys.executable, "-I", "-S", "-c", probe],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                    pass_fds=(info_w,),
                )
            finally:
                os.close(info_w)
            pasta = start_pasta(read_child_pid(info_r, _SANDBOX_TIMEOUT), err)
            if proc.wait(timeout=_SANDBOX_TIMEOUT) == 0:
                _network_selftest_result = (True, "")
            else:
                err.seek(0)
                lines = err.read(8192).decode("utf-8", "replace").strip().splitlines()
                _network_selftest_result = (
                    False, lines[-1] if lines else "the sandbox's interface did not come up")
    except SandboxUnavailable as exc:
        _network_selftest_result = (False, exc.detail)
    except (OSError, subprocess.SubprocessError) as exc:
        _network_selftest_result = (False, f"network selftest failed: {exc}")
    finally:
        os.close(info_r)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        stop_process(pasta)
    return _network_selftest_result


def _shell_network_private(bwrap: str) -> bool:
    """Whether a shell session gets a private network, per :data:`SHELL_NETWORK_ENV`.

    Decided once per worker (the selftest is cached), never per session: a
    session never falls back to the shared namespace.
    """
    global _shared_network_warned
    mode = _network_mode()
    if mode == "shared":
        return False
    ok, detail = _network_selftest(bwrap)
    if ok:
        return True
    if mode == "private":
        raise SandboxUnavailable(
            f"{SHELL_NETWORK_ENV}=private but a private network cannot be set up "
            f"({detail}). Install 'passt' and give the container {_TUN_DEVICE}")
    if not _shared_network_warned:
        _shared_network_warned = True
        logger.warning(
            "Shell sessions share the worker's network namespace (%s): runs can "
            "reach each other's servers and the worker's own ports. Install 'passt' "
            "and give the container %s, or set %s.", detail, _TUN_DEVICE, SHELL_NETWORK_ENV)
    return False


def shell_network_is_private() -> bool:
    """True when ``run_command`` sessions get their own network namespace here."""
    if not sandbox_supported_platform() or _policy() == "off":
        return False
    bwrap = shutil.which("bwrap")
    if not bwrap or not _selftest(bwrap)[0]:
        return False
    try:
        return _shell_network_private(bwrap)
    except SandboxUnavailable:
        return False


def sandboxed_command(
    command: "str | list[str]", *, workspace: str, cwd: str,
    network: bool = True, writable: "list[str] | None" = None,
    shell_network: bool = False,
) -> SandboxedCommand:
    """Wrap one model-authored command for execution.

    ``command`` is a shell string, or an argv run without a shell.
    ``writable`` replaces the workspace as the read-write bind (a probe's
    scratch copy); the workspace's top level stays hidden either way.
    ``shell_network`` applies :data:`SHELL_NETWORK_ENV` (shell sessions): a
    private network is unshared here and the caller attaches pasta.

    Raises :class:`SandboxUnavailable` when a sandbox is mandatory here and
    cannot be started — the caller must refuse the command rather than run it
    unconfined.
    """
    use_shell = isinstance(command, str)
    if not sandbox_supported_platform():
        _warn_unconfined(f"no namespace sandbox exists on {sys.platform}")
        return SandboxedCommand(command, use_shell, "unconfined-platform")

    if _policy() == "off":
        _warn_unconfined(f"{SANDBOX_POLICY_ENV}=off was set explicitly")
        return SandboxedCommand(command, use_shell, "unconfined-override")

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable(
            "bubblewrap (bwrap) is not installed, so this command cannot be "
            "confined to its own run directory. Install the 'bubblewrap' "
            f"package, or set {SANDBOX_POLICY_ENV}=off on a single-tenant host"
        )
    ok, detail = _selftest(bwrap)
    if not ok:
        raise SandboxUnavailable(
            f"bubblewrap cannot create a sandbox here ({detail}). In Docker "
            "this needs security_opt seccomp=unconfined and "
            f"apparmor=unconfined; on a single-tenant host set "
            f"{SANDBOX_POLICY_ENV}=off"
        )

    private = network and shell_network and _shell_network_private(bwrap)
    dns = (["--ro-bind", _sandbox_resolv_conf(), _RESOLV_CONF]
           if private and _loopback_nameserver() else [])
    os.makedirs(sandbox_home(workspace), exist_ok=True)
    argv = [
        bwrap,
        *_isolation_args(network and not private),
        *_mount_args(workspace, writable),
        *dns,
        "--chdir", cwd,
        "--", *(["/bin/sh", "-c", command] if use_shell else command),
    ]
    return SandboxedCommand(argv, False, "bwrap", private_network=private)


def run_confined(
    argv: list[str], *, workspace: str, cwd: str, timeout: float,
    env: "dict[str, str] | None" = None, input: "str | None" = None,
    writable: "list[str] | None" = None, network: bool = False,
) -> subprocess.CompletedProcess:
    """Run a validator's command over generated code: sandboxed, no network
    unless the check cannot work without it (``network=True``).

    Same policy as ``run_command``: where the sandbox is mandatory (Linux,
    unless :data:`SANDBOX_POLICY_ENV` is ``off``) and cannot start, this raises
    :class:`SandboxUnavailable` and the caller skips the check - it never runs
    unconfined. On a platform with no namespaces it runs with a warning.
    Raises ``subprocess.TimeoutExpired`` like :func:`run_bounded`.
    """
    plan = sandboxed_command(argv, workspace=workspace, cwd=cwd,
                             network=network, writable=writable)
    # Never inherit the worker's environment (provider keys, tokens).
    if env is None:
        env = _safe_subprocess_env()
    result = run_bounded(plan.argv, timeout=timeout, cwd=cwd, env=env,
                         shell=plan.use_shell, input=input)
    startup_error = plan.startup_error(result.returncode, result.stderr)
    if startup_error:
        raise SandboxUnavailable(startup_error)
    return result
