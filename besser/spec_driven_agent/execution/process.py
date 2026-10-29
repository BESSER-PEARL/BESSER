"""Subprocess environment and artefact locations shared by tools and probes."""

import io
import locale
import os
import signal
import subprocess
import tempfile
import time

# The scrubbed child environment lives in ``besser.utilities.subprocess_env``
# (the KG consistency check uses it too); the agent's tools and probes import
# it from here under its historical private names.
from besser.utilities.subprocess_env import (  # noqa: F401
    CA_BUNDLE_ENV as _CA_BUNDLE_ENV,
    SAFE_ENV_ALLOWLIST as _SAFE_ENV_ALLOWLIST,
    SECRET_SUBSTRINGS as _SECRET_SUBSTRINGS,
    safe_subprocess_env as _safe_subprocess_env,
)


# Workspace subdirectory holding the untruncated output of shell commands.
# Head+tail truncation drops the middle of a failing `tsc` / `npm run build`
# log, which is where its errors are; the full log is spilled here so
# search_in_files / read_file can still reach them. Run-internal: excluded
# from packaging, the push, the scaffold inventory and the recipe manifest.
COMMAND_OUTPUT_DIR = ".besser_command_output"


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


# Per-stream capture cap. The temp files sit outside the sandbox and are read
# into worker memory; a `yes` or a log loop must not fill the disk or the heap.
MAX_CAPTURE_BYTES = 8_000_000
FLOOD_NOTE = f"\n[output exceeded {MAX_CAPTURE_BYTES // 1_000_000} MB; the command was killed]"


def _captured_size(handle) -> int:
    return os.fstat(handle.fileno()).st_size


def _dropped_marker(size: int) -> bytes:
    return f"\n\n... [{size - 2 * (MAX_CAPTURE_BYTES // 2)} bytes of output dropped] ...\n\n".encode()


def _text(data: bytes) -> str:
    # What text=True would produce, minus its strict decode errors.
    return io.TextIOWrapper(io.BytesIO(data),
                            encoding=locale.getpreferredencoding(False),
                            errors="replace").read()


def _decode(handle) -> str:
    size = _captured_size(handle)
    handle.seek(0)
    if size > MAX_CAPTURE_BYTES:
        half = MAX_CAPTURE_BYTES // 2
        head = handle.read(half)
        handle.seek(size - half)
        data = head + _dropped_marker(size) + handle.read(half)
    else:
        data = handle.read()
    return _text(data)


def decode_output(data: bytes) -> str:
    """:func:`_decode` for output captured in memory (the shell session)."""
    if len(data) > MAX_CAPTURE_BYTES:
        half = MAX_CAPTURE_BYTES // 2
        data = data[:half] + _dropped_marker(len(data)) + data[-half:]
    return _text(data)


def run_bounded(args, *, timeout: float, cwd: str | None = None,
                env: dict[str, str] | None = None,
                shell: bool = False,
                input: str | None = None) -> subprocess.CompletedProcess:
    """``subprocess.run(..., capture_output=True, text=True, timeout=...)``
    whose timeout holds when the command leaves a child running.

    ``subprocess.run`` kills only the direct child, and on Windows it then
    reads the pipes until EOF - which a surviving grandchild (``npm.cmd`` ->
    node) holds open, so a 180 s timeout became an 11-minute-plus hang. Here
    the command gets its own process group, the whole tree is killed on
    timeout, and output goes to temp files, which nothing can hold "open"
    against the reader. Raises ``subprocess.TimeoutExpired`` after the kill.
    A stream past :data:`MAX_CAPTURE_BYTES` also kills the tree; only its
    head and tail are read back. ``input`` is fed on stdin, from a temp file for the same reason.
    """
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, \
            tempfile.TemporaryFile() as stdin:
        if input is not None:
            stdin.write(input.encode(locale.getpreferredencoding(False)))
            stdin.seek(0)
        proc = subprocess.Popen(args, cwd=cwd, env=env, shell=shell,
                                stdin=stdin if input is not None else None,
                                stdout=out, stderr=err, **group)
        deadline = time.monotonic() + timeout
        flooded = False
        while True:
            try:
                proc.wait(timeout=min(0.05, max(deadline - time.monotonic(), 0)))
                break
            except subprocess.TimeoutExpired:
                pass
            flooded = max(_captured_size(out), _captured_size(err)) > MAX_CAPTURE_BYTES
            if flooded or time.monotonic() >= deadline:
                _kill_tree(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                if not flooded:
                    raise subprocess.TimeoutExpired(args, timeout, output=_decode(out),
                                                    stderr=_decode(err)) from None
                break
        stderr = _decode(err)
        if flooded:
            stderr += FLOOD_NOTE
        return subprocess.CompletedProcess(args, proc.returncode, _decode(out), stderr)
