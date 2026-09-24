"""Namespace sandbox and environment scrub for simulated agent processes.

Every agent session runs code the user drew (and, through custom code actions,
wrote). Without confinement that code would share the simulator's view of the
world: the sibling session directories under the sessions root, and
``/proc/1/environ`` (the simulator's own environment, which holds
``AGENT_SIMULATOR_API_TOKEN``). This module closes both the same way the
Spec-Driven Agent confines ``run_command``: each session process is started
inside bubblewrap with

* a user, PID, IPC, UTS and (where available) cgroup namespace, so PID 1 inside
  the sandbox is bwrap's own init and ``/proc`` only shows the session;
* a mount namespace in which the container filesystem is bound read-only, the
  top-level directory holding the sessions root is NOT bound at all, and only
  this session's work directory is bound back read-write.

The network namespace is deliberately NOT unshared: the agent must reach the
LLM provider APIs, and the simulator relays the user's messages to the agent's
WebSocket server on 127.0.0.1. The accepted residual risk (same trade-off as
the Spec-Driven Agent worker) is that agent code has outbound network access
and can connect to the other sessions' loopback ports and to the simulator API
port; the API is protected by ``X-Agent-Simulator-Token``, which agent code
cannot read (the PID namespace hides ``/proc/1/environ`` and
:func:`safe_subprocess_env` never passes it on).

Bubblewrap needs an unprivileged user namespace. Docker's default seccomp
profile gates ``unshare`` / ``mount`` / ``pivot_root`` behind CAP_SYS_ADMIN and
the docker-default AppArmor profile carries ``deny mount``, so the simulator
container runs with ``security_opt: [seccomp=unconfined, apparmor=unconfined]``
and no added capability (see ``docker-compose.yml``).

Fail closed: when the sandbox cannot start, sessions are refused. The only way
to run unconfined is the explicit operator opt-out
``AGENT_SIMULATOR_SANDBOX=off``, meant for single-tenant development hosts.
"""
import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# auto (default): the sandbox is mandatory. off: explicit operator opt-out for
# a single-tenant development host whose kernel forbids unprivileged user
# namespaces. Never set this on a multi-tenant deployment.
SANDBOX_POLICY_ENV = "AGENT_SIMULATOR_SANDBOX"

# Mounted fresh rather than bound: a private /proc (so --unshare-pid actually
# hides the simulator's PID 1) and a private /dev, plus scratch space nothing
# outside the sandbox should see. bwrap's --dev tree has no /dev/shm, and
# POSIX shared memory (multiprocessing) fails without it.
_TMPFS_PATHS = ("/tmp", "/run", "/var/tmp", "/dev/shm")

# Top-level entries the mount plan never binds from the outside.
_VIRTUAL_TOPLEVEL = frozenset({"/proc", "/dev", "/tmp", "/run"})

_SELFTEST_TIMEOUT_SECONDS = 30

# Environment variables safe to hand to agent code (locale and binary lookup).
# Everything else from the simulator's environment is dropped, including
# AGENT_SIMULATOR_API_TOKEN and any credential the operator configured.
_SAFE_ENV_ALLOWLIST = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ"})

# Name substrings that identify secret-like variables. They are stripped even
# when a name is in the allowlist.
_SECRET_SUBSTRINGS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "PRIVATE",
    "API_", "AUTH", "CERT", "SESSION",
)

_selftest_cache: Dict[Optional[int], Tuple[bool, str]] = {}
_unconfined_warned = False


class SandboxUnavailable(RuntimeError):
    """The sandbox cannot be started; the agent session must NOT be run."""


def sandbox_policy() -> str:
    """Return the configured sandbox policy, ``"auto"`` or ``"off"``."""
    raw = (os.environ.get(SANDBOX_POLICY_ENV) or "auto").strip().lower()
    if raw not in {"auto", "off"}:
        logger.warning("%s=%r is not a known policy; using 'auto'", SANDBOX_POLICY_ENV, raw)
        return "auto"
    return raw


def sandbox_supported_platform() -> bool:
    """True where a namespace sandbox (and the simulator itself) can run."""
    return sys.platform.startswith("linux")


def safe_subprocess_env(extra: Mapping[str, str]) -> Dict[str, str]:
    """Build the environment of an agent process.

    Only allowlisted, non-secret variables are copied from the simulator's
    environment; ``extra`` (the session's own settings and the LLM keys the
    user supplied for this session) is added on top, unfiltered.

    Args:
        extra (Mapping[str, str]): Variables set explicitly for this session.

    Returns:
        Dict[str, str]: The environment to pass to :class:`subprocess.Popen`.
    """
    env: Dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper not in _SAFE_ENV_ALLOWLIST:
            continue
        if any(marker in upper for marker in _SECRET_SUBSTRINGS):
            continue
        env[name] = value
    env.setdefault("PATH", os.defpath)
    env.update(extra)
    return env


def _warn_unconfined(reason: str) -> None:
    global _unconfined_warned
    if _unconfined_warned:
        return
    _unconfined_warned = True
    logger.warning(
        "Agent simulator sessions are running UNSANDBOXED: %s. Agent code can see "
        "the sessions root and the simulator's /proc/1/environ. Never do this on "
        "a multi-tenant host.", reason,
    )


def _resolve(path: str) -> str:
    return os.path.realpath(path)


def _within(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _top_level(path: str) -> str:
    """``/tmp/sessions/x`` -> ``/tmp``: the entry the mount plan must not bind."""
    parts = _resolve(path).lstrip("/").split("/", 1)
    return "/" + parts[0]


def _root_entries() -> List[Tuple[str, Optional[str]]]:
    """Top-level directories of ``/`` as ``(path, symlink target or None)``."""
    entries: List[Tuple[str, Optional[str]]] = []
    for entry in sorted(os.listdir("/")):
        path = "/" + entry
        if os.path.islink(path):
            entries.append((path, os.readlink(path)))
        elif os.path.isdir(path):
            entries.append((path, None))
    return entries


def isolation_args() -> List[str]:
    """Namespace flags shared by the session command and the self-test."""
    return [
        # A user namespace is what buys the rest without any capability.
        "--unshare-user",
        # PID 1 becomes bwrap's init, so /proc/1/environ is the sandbox's own
        # scrubbed environment and no other process of the container is visible.
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        # Killing bwrap (terminate / expiry) takes the whole PID namespace with
        # it instead of orphaning double-forked children.
        "--die-with-parent",
        "--new-session",
    ]


def mount_args(work_dir: str, sessions_root: str) -> List[str]:
    """Bind the container filesystem read-only, minus the sessions root.

    Args:
        work_dir (str): This session's work directory, the only writable path.
        sessions_root (str): Directory holding every session's work directory;
            its top-level entry is never bound, so sibling sessions stay hidden.

    Returns:
        List[str]: bwrap mount arguments.
    """
    if not _within(_resolve(work_dir), _resolve(sessions_root)):
        raise SandboxUnavailable(f"Session work dir {work_dir!r} is outside the sessions root")
    hidden = _top_level(sessions_root)
    args: List[str] = []
    for path, link_target in _root_entries():
        if path in _VIRTUAL_TOPLEVEL or path == hidden:
            continue
        if link_target is not None:
            # Debian's merged /usr: /bin, /lib, /lib64 are symlinks into /usr.
            args += ["--symlink", link_target, path]
        else:
            args += ["--ro-bind", path, path]
    args += ["--proc", "/proc", "--dev", "/dev"]
    for path in _TMPFS_PATHS:
        args += ["--tmpfs", path]
    args += ["--bind", work_dir, work_dir]
    return args


def _selftest(bwrap: str, uid: Optional[int]) -> Tuple[bool, str]:
    """Can this kernel / container give a process of ``uid`` the namespaces?

    Run as the same unprivileged UID the sessions use: a root probe can
    succeed where an unprivileged one is refused (``user.max_user_namespaces``,
    AppArmor userns restrictions). Cached per UID for the process lifetime.
    """
    if uid in _selftest_cache:
        return _selftest_cache[uid]
    probe = [
        bwrap, *isolation_args(),
        "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--", "/bin/true",
    ]
    identity = {} if uid is None else {"user": uid, "group": uid, "extra_groups": []}
    try:
        result = subprocess.run(
            probe, capture_output=True, text=True, timeout=_SELFTEST_TIMEOUT_SECONDS, check=False, **identity,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _selftest_cache[uid] = (False, f"bwrap could not be executed: {exc}")
        return _selftest_cache[uid]
    if result.returncode == 0:
        _selftest_cache[uid] = (True, "")
    else:
        detail = (result.stderr or "").strip().splitlines()[-1:] or ["no output"]
        _selftest_cache[uid] = (False, detail[0])
    return _selftest_cache[uid]


def sandbox_selftest_error(uid: Optional[int] = None) -> Optional[str]:
    """Return ``None`` when a sandbox can be started here, else the reason.

    Operator check after a deploy (probing as the first session UID)::

        docker exec besser-wme-agent-simulator \\
          python -m besser.utilities.web_modeling_editor.agent_simulator.sandbox 20000
    """
    if not sandbox_supported_platform():
        return f"no namespace sandbox on {sys.platform}"
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "bubblewrap (bwrap) is not installed"
    ok, detail = _selftest(bwrap, uid)
    return None if ok else detail


def sandboxed_argv(command: List[str], *, work_dir: str, sessions_root: str, uid: Optional[int]) -> List[str]:
    """Wrap an agent command for execution inside the sandbox.

    Args:
        command (List[str]): The command to run (e.g. ``[python, "agent.py"]``).
        work_dir (str): The session work directory (bound read-write, cwd).
        sessions_root (str): The directory holding every session's work dir.
        uid (Optional[int]): The UID the session will run as, used for the
            self-test.

    Returns:
        List[str]: The argv to hand to :class:`subprocess.Popen`.

    Raises:
        SandboxUnavailable: The platform cannot run the simulator, or the
            sandbox is mandatory and cannot be started. The caller must refuse
            the session instead of running it unconfined.
    """
    if not sandbox_supported_platform():
        raise SandboxUnavailable(f"The agent simulator only runs on Linux, not on {sys.platform}")
    if sandbox_policy() == "off":
        _warn_unconfined(f"{SANDBOX_POLICY_ENV}=off was set explicitly")
        return list(command)
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable(
            "bubblewrap (bwrap) is not installed, so agent sessions cannot be confined to "
            "their own work directory. Install the 'bubblewrap' package, or set "
            f"{SANDBOX_POLICY_ENV}=off on a single-tenant development host"
        )
    ok, detail = _selftest(bwrap, uid)
    if not ok:
        raise SandboxUnavailable(
            f"bubblewrap cannot create a sandbox here ({detail}). In Docker this needs "
            "security_opt seccomp=unconfined and apparmor=unconfined; on a single-tenant "
            f"development host set {SANDBOX_POLICY_ENV}=off"
        )
    return [
        bwrap,
        *isolation_args(),
        *mount_args(work_dir, sessions_root),
        "--chdir", work_dir,
        "--", *command,
    ]


if __name__ == "__main__":
    _probe_uid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _error = sandbox_selftest_error(_probe_uid)
    print(_error or "sandbox OK")
    sys.exit(1 if _error else 0)
