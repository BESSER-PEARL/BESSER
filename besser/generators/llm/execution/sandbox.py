"""Namespace sandbox for the model-authored shell (``run_command``).

Two confinement holes, both verified live inside the hosted worker:

* the cwd lock constrains tool *arguments* (``path``, ``working_dir``), not
  the command string. ``shell=True`` with no chroot meant
  ``cd /workspace/runs/<other_run> && cat .besser_trace.jsonl`` read a second
  user's spec out of ``payload.instructions`` — and that directory was
  writable, so one run could tamper with a concurrent one;
* the worker is root in a single shared PID namespace, so ``/proc/1/environ``
  handed back this container's own live tokens (``BESSER_FREE_LLM_TOKEN``
  among them), straight past the ``_safe_subprocess_env`` scrub.

Both are visibility problems and both close with the same measure: run the
command in a mount namespace where only this run's directory is bound, and a
PID namespace where PID 1 is the sandbox's own init. Per-run containers stay
the durable answer; this is the interim that costs one apt package.

bubblewrap over nsjail: ``bubblewrap`` is one package in Debian main, a single
line in the image's existing apt layer. nsjail has no Debian package and has to
be built from source (protobuf, libnl, bison, flex) onto an image already at
4.3 GB against a 15 GB disk. Both provide the namespaces; only one is free.

Bubblewrap needs an unprivileged user namespace, and Docker's default seccomp
profile gates ``unshare`` / ``mount`` / ``pivot_root`` on CAP_SYS_ADMIN — so the
worker runs with ``seccomp=unconfined`` (plus ``apparmor=unconfined``, since the
docker-default AppArmor profile carries ``deny mount``). Neither grants a
capability. See ``docker-compose.prod.yml``.

Fail closed: when the sandbox is mandatory and cannot start, ``run_command``
refuses. It never falls back to an unconfined shell.
"""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

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

# Bound read-write because the toolchain has to write there: pip installs into
# /usr/local, and $HOME holds the npm / pip caches and the rustup toolchain
# that PATH points at (/root/.cargo/bin). Everything else is read-only.
_WRITABLE_PATHS = ("/usr/local", "/root")

_SANDBOX_TIMEOUT = 30

_selftest_result: tuple[bool, str] | None = None
_unconfined_warned = False


class SandboxUnavailable(RuntimeError):
    """The sandbox could not be prepared. The command must NOT be run."""


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


def _mount_args(workspace: str) -> list[str]:
    """Bind the container filesystem minus this run's siblings.

    Everything the toolchain needs stays visible — a compile check, ``pip
    show`` and the app's own test suite must keep working — but the top-level
    directory the run workspaces live under is left out entirely and replaced
    by a bind of this one run's directory.
    """
    hidden = _top_level(workspace)
    args: list[str] = []
    for entry in sorted(os.listdir("/")):
        path = "/" + entry
        if path in _VIRTUAL_TOPLEVEL or path == hidden:
            continue
        if os.path.islink(path):
            # Debian's merged /usr: /bin, /lib, /lib64 are symlinks into /usr.
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    args += ["--proc", "/proc", "--dev", "/dev"]
    for path in _TMPFS_PATHS:
        args += ["--tmpfs", path]
    # $HOME as well as /root: the image runs as root, but a library caller on
    # Linux has its caches and often its venv somewhere else, and a read-only
    # home turns `pip install` into a permission error.
    home = os.environ.get("HOME") or ""
    for path in dict.fromkeys([*_WRITABLE_PATHS, home]):
        # A writable bind under the hidden top level would put the siblings
        # back in view — that is the hole this exists to close.
        if path and os.path.isdir(path) and not _within(path, hidden):
            args += ["--bind", path, path]
    args += ["--bind", workspace, workspace]
    return args


def _isolation_args() -> list[str]:
    return [
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
          "from besser.generators.llm.execution.sandbox import sandbox_selftest_error as e; print(e() or 'sandbox OK')"
    """
    if not sandbox_supported_platform():
        return f"no namespace sandbox on {sys.platform}"
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "bubblewrap (bwrap) is not installed"
    ok, detail = _selftest(bwrap)
    return None if ok else detail


def sandboxed_command(
    command: str, *, workspace: str, cwd: str,
) -> SandboxedCommand:
    """Wrap one model-authored command for execution.

    Raises :class:`SandboxUnavailable` when a sandbox is mandatory here and
    cannot be started — the caller must refuse the command rather than run it
    unconfined.
    """
    if not sandbox_supported_platform():
        _warn_unconfined(f"no namespace sandbox exists on {sys.platform}")
        return SandboxedCommand(command, True, "unconfined-platform")

    if _policy() == "off":
        _warn_unconfined(f"{SANDBOX_POLICY_ENV}=off was set explicitly")
        return SandboxedCommand(command, True, "unconfined-override")

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

    argv = [
        bwrap,
        *_isolation_args(),
        *_mount_args(workspace),
        "--chdir", cwd,
        "--", "/bin/sh", "-c", command,
    ]
    return SandboxedCommand(argv, False, "bwrap")
