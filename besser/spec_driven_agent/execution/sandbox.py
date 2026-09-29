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
"""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from besser.spec_driven_agent.execution.process import run_bounded

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

_SANDBOX_TIMEOUT = 30

_selftest_result: tuple[bool, str] | None = None
_unconfined_warned = False


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
    telemetry = os.path.realpath(
        os.environ.get(_TELEMETRY_DIR_ENV) or _TELEMETRY_DIR_DEFAULT)
    if os.path.isdir(telemetry) and not any(_within(telemetry, h) for h in hidden):
        args += ["--tmpfs", telemetry]
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


def sandboxed_command(
    command: "str | list[str]", *, workspace: str, cwd: str,
    network: bool = True, writable: "list[str] | None" = None,
) -> SandboxedCommand:
    """Wrap one model-authored command for execution.

    ``command`` is a shell string, or an argv run without a shell.
    ``writable`` replaces the workspace as the read-write bind (a probe's
    scratch copy); the workspace's top level stays hidden either way.

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

    os.makedirs(sandbox_home(workspace), exist_ok=True)
    argv = [
        bwrap,
        *_isolation_args(network),
        *_mount_args(workspace, writable),
        "--chdir", cwd,
        "--", *(["/bin/sh", "-c", command] if use_shell else command),
    ]
    return SandboxedCommand(argv, False, "bwrap")


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
    result = run_bounded(plan.argv, timeout=timeout, cwd=cwd, env=env,
                         shell=plan.use_shell, input=input)
    startup_error = plan.startup_error(result.returncode, result.stderr)
    if startup_error:
        raise SandboxUnavailable(startup_error)
    return result
