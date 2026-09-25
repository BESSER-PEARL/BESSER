"""Subprocess environment and artefact locations shared by tools and probes."""

import io
import locale
import os
import signal
import subprocess
import tempfile


# Workspace subdirectory holding the untruncated output of shell commands.
# Head+tail truncation drops the middle of a failing `tsc` / `npm run build`
# log, which is where its errors are; the full log is spilled here so
# search_in_files / read_file can still reach them. Run-internal: excluded
# from packaging, the push, the scaffold inventory and the recipe manifest.
COMMAND_OUTPUT_DIR = ".besser_command_output"


# Environment variables that are safe to expose to LLM-invoked subprocesses.
# These are needed for basic tooling to work (PATH for binaries, HOME for
# tool caches, LANG/LC_* for locale-aware tools, TMPDIR for scratch space,
# SystemRoot/USERPROFILE on Windows). Everything else — including provider
# API keys, deployment credentials, SMTP passwords, OAuth secrets — is
# stripped so the LLM cannot `printenv` them into generated code.
_SAFE_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "TMP", "TEMP",
    # Windows
    "SystemRoot", "SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "COMSPEC", "PATHEXT", "ProgramFiles", "ProgramData",
    # Python (harmless, often needed by tools)
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    # Node (harmless, often needed by tools)
    "NODE_PATH",
})

# Variable-name substrings that identify secret-like env vars. Even if
# a variable is accidentally in the allowlist, names matching these
# patterns are always stripped.
_SECRET_SUBSTRINGS: tuple[str, ...] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "PRIVATE",
    "API_", "AUTH", "CERT", "SESSION",
)


def _safe_subprocess_env() -> dict[str, str]:
    """Return a minimal subprocess environment with secrets stripped.

    Never pass the full ``os.environ`` to an LLM-invoked subprocess —
    that would leak provider API keys, OAuth secrets, SMTP credentials,
    and any other server-side configuration. This helper constructs a
    new environment by copying only allowlisted variables and
    deliberately drops anything whose name contains a secret-like
    substring, even if it's in the allowlist.
    """
    # Case-folded: Windows env var names are case-insensitive and os.environ
    # iterates them upper-cased (PROGRAMFILES, not "ProgramFiles"). Nothing
    # beyond the allowlist's own entries is admitted.
    allowed = {name.upper() for name in _SAFE_ENV_ALLOWLIST}
    safe: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper not in allowed:
            continue
        if any(substr in upper for substr in _SECRET_SUBSTRINGS):
            continue
        safe[name] = value
    # Ensure PATH exists even if the parent somehow didn't have it.
    safe.setdefault("PATH", os.defpath)
    # Suppress .pyc writes.
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and every process it started."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _decode(handle) -> str:
    handle.seek(0)
    # What text=True would produce, minus its strict decode errors.
    return io.TextIOWrapper(io.BytesIO(handle.read()),
                            encoding=locale.getpreferredencoding(False),
                            errors="replace").read()


def run_bounded(args, *, timeout: float, cwd: str | None = None,
                env: dict[str, str] | None = None,
                shell: bool = False) -> subprocess.CompletedProcess:
    """``subprocess.run(..., capture_output=True, text=True, timeout=...)``
    whose timeout holds when the command leaves a child running.

    ``subprocess.run`` kills only the direct child, and on Windows it then
    reads the pipes until EOF - which a surviving grandchild (``npm.cmd`` ->
    node) holds open, so a 180 s timeout became an 11-minute-plus hang. Here
    the command gets its own process group, the whole tree is killed on
    timeout, and output goes to temp files, which nothing can hold "open"
    against the reader. Raises ``subprocess.TimeoutExpired`` after the kill.
    """
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, cwd=cwd, env=env, shell=shell,
                                stdout=out, stderr=err, **group)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise subprocess.TimeoutExpired(args, timeout, output=_decode(out),
                                            stderr=_decode(err)) from None
        return subprocess.CompletedProcess(args, proc.returncode, _decode(out), _decode(err))
