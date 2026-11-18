"""Shell runbook: the procedure that turns shell access into verification.

Shell access on its own does not make a model verify anything: models given
a ``test_api`` tool rarely call it, and an agent with unrestricted shell can
make hundreds of calls without ever booting the application.

Two things are missing, and this module supplies both.

**1. A boot that cannot block.** A server started in the foreground of
``run_command`` burns the whole 120-second budget, is killed, and returns
nothing. Booting a web app from this tool therefore requires the model to know
a detach idiom that differs per platform (``… > log 2>&1 &`` on Linux,
``start /b … > log 2>&1`` on Windows) and to get it right first time - with a
120-second dead end as the penalty for getting it wrong. ``.besser_probe.py``
does the detaching in Python, so the model issues one literal command that
behaves identically on both platforms. The run's shell session keeps the
server running between commands (see ``execution/shell_session.py``).

**2. A procedure with a defined next step at every branch.** Telling a model
to "verify the app" is an exhortation; it measurably does not work. The
runbook below is a numbered sequence of literal commands in which every
outcome names the single next command to run.

Nothing here scores the application or decides what is correct - that stays
with the model and with the harness's own probe (``constructibility.py``).
This module only removes the mechanical obstacles between an edit and a
traceback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from typing import Iterable

from besser.spec_driven_agent.pipeline.constants import (
    _RECIPE_EXCLUDED_DIRS,
    RUN_DIR_PREFIX,
    WORKSPACE_SKIP_DIRS,
    is_excluded_output_file,
)
from besser.spec_driven_agent.settings import AgentSettings
from besser.utilities.workspace_fs import is_plain_relpath, open_plain, open_plain_write, remove_plain, walk_plain

logger = logging.getLogger(__name__)

PROBE_FILENAME = ".besser_probe.py"
# Written by the probe: the files that appeared under the backend while it ran the app.
RUNTIME_FILES_FILENAME = ".besser_probe.runtime_files.json"
# The record comes from the sandbox. The probe stops at MAX_RECORDED entries
# (about 300 KB); a larger record is not the probe's and is not parsed.
_MAX_RUNTIME_RECORD_BYTES = 1_000_000
_MAX_RUNTIME_RECORD_ENTRIES = 2000
_MAX_SCRATCH_SCAN_BYTES = 1_000_000
# Helper scripts by name; a .py qualifies only at the root of a workspace that is not the app.
_HELPER_SCRIPT_SUFFIXES = (".sh", ".ps1", ".bat", ".cmd")
# A workspace root holding one of these is the app itself, so its .py files are app code.
_APP_ROOT_MARKERS = ("__init__.py", "manage.py", "setup.py", "pyproject.toml")

# Environment kill switch (BESSER_LLM_SHELL_RUNBOOK). Default on whenever
# shell tools are enabled; ``0`` gives the shell without the procedure (an
# escape hatch if the section misfires on a stack the probe does not understand).
def runbook_enabled() -> bool:
    return AgentSettings().llm_shell_runbook


def _find_backend(output_dir: str) -> str | None:
    """Workspace-relative path of the generated FastAPI service, if present.

    Same signature as ``constructibility._fastapi_backends``: ``main_api.py``
    next to ``sql_alchemy.py``. The probe script only knows how to drive that
    scaffold family, so on any other stack the runbook renders nothing rather
    than handing the model commands that cannot work.
    """
    try:
        for root, dirs, files in walk_plain(output_dir):
            dirs[:] = sorted(d for d in dirs if d not in WORKSPACE_SKIP_DIRS and not d.startswith(".besser_"))
            if "main_api.py" in files and "sql_alchemy.py" in files:
                rel = os.path.relpath(root, output_dir).replace("\\", "/")
                return "." if rel == "." else rel
    except OSError:
        logger.debug("runbook: backend scan failed", exc_info=True)
    return None


def install_probe(output_dir: str) -> bool:
    """Write ``.besser_probe.py`` into the workspace root. Never raises.

    The ``.besser_`` prefix is the established marker for run-internal files:
    they are excluded from the delivered ZIP, the recipe manifest, the
    scaffold inventory, ``list_files`` and ``search_in_files``. The model
    reaches this one only through the literal path the runbook prints.
    """
    try:
        path = os.path.join(output_dir, PROBE_FILENAME)
        # The worker writes this unsandboxed: never through a planted link.
        with open_plain_write(path, "w", root=output_dir,
                              encoding="utf-8", newline="\n") as handle:
            handle.write(PROBE_SCRIPT)
        return True
    except OSError:
        logger.debug("runbook: could not install the probe script", exc_info=True)
        return False


@dataclass(frozen=True)
class RuntimeFileRemoval:
    """What the end-of-run removal of the app's own files did."""

    removed: tuple[str, ...] = ()
    # Still the app's and still in the output: the operating system refused the delete.
    not_removed: tuple[str, ...] = ()
    # Why the probe's record could not be used in full; "" when it could.
    note: str = ""


def _read_runtime_record(output_dir: str) -> tuple[dict, str]:
    """The probe's ``{path: [size, mtime_ns]}`` record and why it is incomplete, if it is."""
    record = os.path.join(output_dir, RUNTIME_FILES_FILENAME)
    try:
        if os.lstat(record).st_size > _MAX_RUNTIME_RECORD_BYTES:
            return {}, "the record of files the app wrote is too large to read"
        with open_plain(record, "r", root=output_dir, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}, ""
    except (OSError, ValueError, RecursionError):
        return {}, "the record of files the app wrote could not be read"
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        return {}, "the record of files the app wrote could not be read"
    overflow = data.get("overflow")
    note = (f"the app wrote more files than the record holds ({overflow} more)"
            if type(overflow) is int and overflow > 0 else "")
    return files, note


def _is_app_file_path(rel: object, backend: str) -> bool:
    """A path the probe can have recorded: a plain one under the backend folder,
    through no dot-folder, naming no run record, and free of the trailing dot or
    space under which Windows opens another file."""
    if not is_plain_relpath(rel) or not rel.startswith(backend + "/"):
        return False
    *folders, name = rel.split("/")
    return (not name.startswith(".besser_")
            and not any(folder.startswith(".") for folder in folders)
            and not any(part != part.rstrip(". ") for part in (*folders, name)))


def remove_app_runtime_files(output_dir: str, keep: Iterable[str]) -> RuntimeFileRemoval:
    """Delete what the app wrote while the probe ran it, and the probe's record of it.

    A generated signing key or an upload is state of that test run, not part
    of the app: shipped, every deployment would share it. The record comes
    from the sandbox and is not trusted. A file is deleted only when it is
    under the backend folder this function finds itself, is not in ``keep``
    (the workspace-relative paths a generator or a write tool wrote, or that
    were there when the run started) and still has the size and modification
    time the probe recorded. A file the output leaves out by name anyway is
    left where it is. Never raises.
    """
    root = os.path.realpath(output_dir)
    files, note = _read_runtime_record(root)
    backend = _find_backend(root)
    if files and backend is None:
        files, note = {}, "no backend folder was found, so the record of files the app wrote was not used"
    elif files and backend == ".":
        # At the workspace root "under the backend" confines nothing: a lock
        # file or seed data written during a probe call would go too.
        files, note = {}, ("the backend is the workspace root, so files the app wrote "
                           "cannot be told from the rest; none were removed")
    if len(files) > _MAX_RUNTIME_RECORD_ENTRIES:
        note = note or f"the record names {len(files)} files; only the first {_MAX_RUNTIME_RECORD_ENTRIES} were read"
    kept = {os.path.normcase(os.path.realpath(os.path.join(root, rel))) for rel in keep}
    removed: list[str] = []
    not_removed: list[str] = []
    rejected: list[str] = []
    for rel in sorted(files, key=str)[:_MAX_RUNTIME_RECORD_ENTRIES]:
        if not _is_app_file_path(rel, backend):
            rejected.append(repr(rel)[:120])
            continue
        *folders, name = rel.split("/")
        if is_excluded_output_file(name) or _RECIPE_EXCLUDED_DIRS.intersection(folders):
            continue
        path = os.path.join(root, *folders, name)
        try:
            info = os.lstat(path)
        except OSError:
            continue
        # A link, or a name the file system resolves to another file.
        resolved = os.path.normcase(os.path.realpath(path))
        if not stat.S_ISREG(info.st_mode) or resolved != os.path.normcase(path):
            rejected.append(repr(rel)[:120])
            continue
        if resolved in kept or [info.st_size, info.st_mtime_ns] != files[rel]:
            # Written on purpose, or edited since the app wrote it: it ships.
            continue
        try:
            remove_plain(path, root=root)
        except OSError as exc:
            logger.warning("runbook: could not remove %s, which the app wrote in its test run: %s",
                           rel, exc.strerror or type(exc).__name__)
            not_removed.append(rel)
        else:
            removed.append(rel)
    if rejected:
        logger.warning("runbook: ignored %d entr(ies) of %s that do not name a file the app wrote "
                       "under %s: %s", len(rejected), RUNTIME_FILES_FILENAME, backend,
                       ", ".join(rejected[:10]))
    if note:
        logger.warning("runbook: %s; such files may remain in the output", note)
    try:
        remove_plain(os.path.join(root, RUNTIME_FILES_FILENAME), root=root)
    except OSError:
        logger.debug("runbook: could not remove the runtime-file record", exc_info=True)
    return RuntimeFileRemoval(tuple(removed), tuple(not_removed), note)


def _read_small_text(root: str, rel: str) -> str | None:
    """A regular workspace file's text, or None when it is missing, special or too large."""
    path = os.path.join(root, *rel.split("/"))
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_SCRATCH_SCAN_BYTES:
            return None
        with open_plain(path, "r", root=root, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return None


def _is_helper_script(root: str, rel: str) -> bool:
    """A shell script, or a non-test ``.py`` at a workspace root that is not itself the app."""
    name = rel.rsplit("/", 1)[-1].lower()
    if name.endswith(_HELPER_SCRIPT_SUFFIXES):
        return True
    if not name.endswith(".py") or "/" in rel or name.startswith("test_") or name.endswith("_test.py"):
        return False
    return _find_backend(root) != "." and not any(
        os.path.exists(os.path.join(root, marker)) for marker in _APP_ROOT_MARKERS)


def _is_referenced(root: str, rel: str) -> bool:
    """Whether another workspace file names ``rel`` (or imports it, for a ``.py``)."""
    name = rel.rsplit("/", 1)[-1]
    stem = name[:-3] if name.endswith(".py") else None
    imports = re.compile(r"\b(?:import|from)\s+%s\b" % re.escape(stem)) if stem else None
    for folder, dirs, files in walk_plain(root):
        dirs[:] = [d for d in dirs if d not in _RECIPE_EXCLUDED_DIRS and not d.startswith(".")]
        for other in files:
            other_rel = os.path.relpath(os.path.join(folder, other), root).replace("\\", "/")
            if other.startswith(".besser_") or other_rel == rel:
                continue
            text = _read_small_text(root, other_rel)
            if text and (name in text or (imports and imports.search(text))):
                return True
    return False


def remove_run_folder_bound_files(output_dir: str, written: Iterable[str],
                                  removable: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sort the files in ``written`` that name the run's temporary workspace path.

    The folder is removed after the run, so such a file works only in this
    run's sandbox. One of ``removable`` (the agent's own writes) that is a
    standalone helper script no other file names is deleted. Every other such
    file is part of the app or used by it, and is kept. Returns ``(removed,
    kept)``. Never raises.
    """
    root = os.path.realpath(output_dir)
    if not os.path.basename(root).startswith(RUN_DIR_PREFIX):
        return (), ()
    needles = {root, os.path.abspath(output_dir)}
    needles |= {needle.replace("\\", "/") for needle in needles}
    removable = set(removable)
    removed: list[str] = []
    kept: list[str] = []
    for rel in sorted(set(written)):
        if not is_plain_relpath(rel):
            continue
        *folders, name = rel.split("/")
        if (name.startswith(".besser_") or is_excluded_output_file(name)
                or _RECIPE_EXCLUDED_DIRS.intersection(folders)):
            continue
        text = _read_small_text(root, rel)
        if not text or not any(needle in text for needle in needles):
            continue
        if rel in removable and _is_helper_script(root, rel) and not _is_referenced(root, rel):
            try:
                remove_plain(os.path.join(root, *folders, name), root=root)
                removed.append(rel)
                continue
            except OSError as exc:
                logger.warning("runbook: could not remove %s: %s", rel, exc.strerror or type(exc).__name__)
        kept.append(rel)
    return tuple(removed), tuple(kept)


def runbook_section(output_dir: str) -> str:
    """The runbook block for the system prompt, or "" when it does not apply.

    Returns "" when the kill switch is off, when the workspace holds no
    FastAPI backend, or when the script could not be written - never a
    procedure the model cannot actually execute.
    """
    if not runbook_enabled():
        return ""
    backend = _find_backend(output_dir)
    if backend is None:
        return ""
    if not install_probe(output_dir):
        return ""
    where = "" if backend == "." else f" (it finds the backend at `{backend}/`)"
    return f"""

## Runtime verification — run the app, do not just read it

Shell tools are enabled for this run. Reading code does not tell you whether
it works; the defects that ship are startup crashes, 422s on the payload the
form actually sends, and actions that answer 200 without changing anything.
All three are invisible in a diff and obvious in one request.

`{PROBE_FILENAME}` is already in the workspace root{where}. It starts the
server detached (a plain `python main_api.py` through `run_command` blocks for
the full 120s timeout and tells you nothing), captures its log, and prints
tracebacks. Run these from the workspace root, exactly as written:

```
python {PROBE_FILENAME} up                       # boot; prints BOOT_OK / BOOT_FAILED
python {PROBE_FILENAME} routes                   # the paths and schemas the app REALLY has
python {PROBE_FILENAME} req GET /health
python {PROBE_FILENAME} req POST /<entity> '{{"field": "value"}}'   # JSON body in single quotes
python {PROBE_FILENAME} req POST /<entity> @body.json            # body from a file; - reads stdin
python {PROBE_FILENAME} req POST /<entity>/upload -F "file=@sample.pdf"   # multipart upload
python {PROBE_FILENAME} req POST /<entity>/<id>/<action> {{}}
python {PROBE_FILENAME} req GET /<entity> -H "Authorization: Bearer <token>"
python {PROBE_FILENAME} log 60                   # last 60 lines of the server log
python {PROBE_FILENAME} down                     # stop it
```

The shell is one session for the whole run, so the server `up` starts keeps
running between commands: `routes` and `req` reach that same server, and `down`
stops it. If the session was restarted in between (a result's notes say so),
`req` and `routes` boot a fresh server themselves; the SQLite file keeps
records either way. The probe path is relative: if you `cd`-ed elsewhere, pass
`working_dir="."`. Files the app writes while it runs here (its database, a key
it generates, uploads) are not delivered with the app: its own code has to
create whatever it needs at startup.

### The procedure — every outcome has exactly one next step

1. `up`. If it prints **BOOT_FAILED**, the traceback printed under it is the
   real error: fix that file and line, then run `up` again. Nothing else you
   do matters until it prints BOOT_OK. If it prints **MISSING_DEPENDENCY**,
   call `install_dependencies` for that directory, then `up` again.
2. `routes`. Use the paths and field names it prints for every request below.
   Do not use the paths you remember writing - `routes` is the live truth.
3. `req GET /health`. Not 200 → step 1's log tail names the cause.
4. **Create one of every entity**, parents before children, reusing the `id`
   each create returns. A **500** is your bug: the log tail printed under the
   response holds the traceback - fix the named file and line, `down`, `up`,
   and repeat the SAME request. A **422** means your payload disagrees with
   the schema the app declares: re-read that schema in `routes` and correct
   the *payload*. Never relax a schema, drop a constraint or delete a
   validator to make a request pass.
5. **Call every action the specification describes**, with the literal `{{}}`
   body the generated button sends, on an entity you just created. Then
   `req GET /<entity>/<id>` and check the field the action was supposed to
   change actually changed. An action that answers 200 and changes nothing is
   a failure, not a pass - it is the single most common defect in these apps.
6. After any edit: `down`, then `up`. The server does not reload.
   A **401/403** means the endpoint needs auth: register and log in through
   the app's own endpoints (`routes auth`), then repeat the request with
   `-H "Authorization: Bearer <access_token from the login response>"`.
7. Only when every entity creates and every action both succeeds and has a
   visible effect may you report the app as working. If an action cannot be
   made to work, leave its checklist item blocked with what you observed -
   do not close it.

State what you actually observed - a status code and the value you read back -
not that you "verified" something.
"""


# ----------------------------------------------------------------------
# The workspace-resident script. Stdlib only (the generated backend's venv
# is not guaranteed to hold httpx or requests), and small enough that its
# output cannot eat the 15k-char tool-output budget.
# ----------------------------------------------------------------------

PROBE_SCRIPT = r'''#!/usr/bin/env python3
"""Drive the generated backend from the shell: boot it, call it, read its log.

Run-internal helper written by the BESSER generator; not part of the app.

    python .besser_probe.py up
    python .besser_probe.py routes
    python .besser_probe.py req POST /book '{"title": "x"}'
    python .besser_probe.py req GET /book -H "Authorization: Bearer <token>"
    python .besser_probe.py req POST /book/upload -F "file=@cover.png" -F "title=x"
    python .besser_probe.py log 60
    python .besser_probe.py down

Exists because run_command has a 120-second cap: a foreground server
consumes the whole budget and returns nothing. This starts the server as a
detached child with its output on disk, so the command returns in a second
or two on every platform.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(ROOT, ".besser_probe.state.json")
LOG = os.path.join(ROOT, ".besser_probe.server.log")
RUNTIME_FILES = os.path.join(ROOT, ".besser_probe.runtime_files.json")
SKIP = {"node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build"}
MAX_RECORDED = 2000          # files noted as written by the app; the rest are only counted
MAX_LIFETIME = 1800          # a forgotten server reaps itself after 30 minutes
BOOT_WAIT = 45
BODY_CHARS = 1800
# In the server's own command line, so `down` can tell it from an unrelated
# process holding the same pid (e.g. a pid recorded in another pid namespace).
SERVE_MARKER = "BESSER_PROBE_SERVE"


def find_backend():
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".besser_"))
        if "main_api.py" in files and "sql_alchemy.py" in files:
            return root
    return None


def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def reusable_port(port):
    """``port`` when nothing holds it, else None."""
    import socket
    if type(port) is not int or not 0 < port < 65536:
        return None
    with socket.socket() as sock:
        if os.name != "nt":
            # As uvicorn binds: the stopped server's TIME_WAIT does not block it.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return None
    return port


def read_state():
    try:
        with open(STATE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def log_tail(lines=40):
    try:
        with open(LOG, encoding="utf-8", errors="replace") as handle:
            content = handle.readlines()
    except OSError:
        return "(no server log yet)"
    return "".join(content[-lines:]).rstrip() or "(server log is empty)"


def app_files(backend):
    found = set()
    for root, dirs, files in os.walk(backend):
        # Dot-folders hold tool caches and run records, never the app's data.
        dirs[:] = [d for d in dirs if d not in SKIP and d != "htmlcov" and not d.startswith(".")]
        found.update(os.path.relpath(os.path.join(root, name), ROOT).replace("\\", "/")
                     for name in files if not name.startswith(".besser_"))
    return found


def file_stat(rel):
    try:
        info = os.stat(os.path.join(ROOT, rel))
    except OSError:
        return None
    return [info.st_size, info.st_mtime_ns]


def recording(action):
    """Run `action` (a boot or a request) and note the files that appeared
    under the backend meanwhile, each with its size and modification time: the
    running app wrote them (a generated key, an upload), so they are left out
    of the delivered app. A noted file that changed between two probe calls
    was edited by something else and is no longer noted."""
    backend = find_backend()
    if backend is None or os.path.samefile(backend, ROOT):
        # With the app at the workspace root its files cannot be told from the rest.
        return action()
    try:
        with open(RUNTIME_FILES, encoding="utf-8") as handle:
            noted = dict(json.load(handle)["files"])
    except (OSError, ValueError, TypeError, KeyError):
        noted = {}
    noted = {path: seen for path, seen in noted.items() if file_stat(path) == seen}
    before = app_files(backend)
    try:
        return action()
    finally:
        noted.update(dict.fromkeys(app_files(backend) - before))
        noted = {path: file_stat(path) for path in sorted(noted)}
        noted = {path: seen for path, seen in noted.items() if seen}
        if noted or os.path.exists(RUNTIME_FILES):
            kept = dict(list(noted.items())[:MAX_RECORDED])
            with open(RUNTIME_FILES, "w", encoding="utf-8") as handle:
                json.dump({"files": kept, "overflow": len(noted) - len(kept)}, handle)


def request(method, path, body, port, timeout=25, _hops=0, extra_headers=None, form=None):
    """``form``, when given, is an encoded multipart body and its Content-Type."""
    url = "http://127.0.0.1:%d%s" % (port, path if path.startswith("/") else "/" + path)
    data = None
    headers = {"Accept": "application/json"}
    headers.update(extra_headers or {})
    if form is not None:
        data, headers["Content-Type"] = form
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # FastAPI answers POST /person with 307 -> /person/, and urllib will
        # not replay a POST body across a redirect. Follow it ourselves with
        # the SAME method and body, which is what 307/308 mean; otherwise
        # every create reads as a bare "307" with an empty body.
        location = exc.headers.get("Location") if exc.headers else None
        if exc.code in (301, 302, 303, 307, 308) and location and _hops < 3:
            if location.startswith("http"):
                location = "/" + location.split("/", 3)[-1] if location.count("/") > 2 else "/"
            follow = "GET" if exc.code == 303 else method
            return request(follow, location, body, port, timeout, _hops + 1, extra_headers, form)
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:                    # connection refused, reset, timeout
        return 0, "%s: %s" % (type(exc).__name__, exc)


def is_up(port):
    status, _ = request("GET", "/openapi.json", None, port, timeout=3)
    return status == 200


def command_line(pid):
    """The command line of ``pid``, or "" when it is gone or unreadable."""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        pass
    try:
        if os.name == "nt":
            argv = ["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % pid]
        else:
            argv = ["ps", "-ww", "-o", "args=", "-p", str(pid)]
        return subprocess.run(argv, capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""


def is_probe_server(pid, port):
    """Whether ``pid`` is the server this probe started on ``port``."""
    if type(pid) is not int or pid <= 0 or pid in (os.getpid(), os.getppid()):
        return False
    return ("%s port=%s" % (SERVE_MARKER, port)) in command_line(pid)


def cmd_down(quiet=False):
    state = read_state()
    pid = state.get("pid")
    # The recorded pid may name another process now: after the shell session
    # restarted, pids are reused in the new pid namespace and it can even be a
    # sibling of this one. Only a process that is recognisably our own server
    # is stopped.
    if pid and not is_probe_server(pid, state.get("port")):
        pid = None
    if not pid:
        if not quiet:
            print("NOT_RUNNING")
        return 0
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        else:
            os.kill(pid, 15)
            time.sleep(0.6)
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    except ProcessLookupError:
        pass                            # already gone, e.g. with its sandbox
    except Exception as exc:
        print("STOP_FAILED %s: %s" % (type(exc).__name__, exc))
    # Only the port stays: the next boot takes it again while it is free, so a
    # URL written with it keeps working across restarts.
    try:
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump({"port": state.get("port")}, fh)
    except OSError:
        pass
    if not quiet:
        print("STOPPED pid=%s" % pid)
    return 0


def cmd_up(quiet=False):
    return recording(lambda: boot(quiet))[0]


def boot(quiet=False):
    """Start a server; returns (exit code, the port THIS call started).

    The port is returned, not re-read from the state file, which a concurrent
    invocation may have overwritten with its own.
    """
    backend = find_backend()
    if backend is None:
        print("NO_BACKEND: no directory holds both main_api.py and sql_alchemy.py.")
        return 2, None
    cmd_down(quiet=True)
    port = reusable_port(read_state().get("port")) or free_port()
    serve = (
        "# %s port=%d\n"
        "import os,sys,threading,time\n"
        "sys.path.insert(0, os.getcwd())\n"
        "threading.Timer(%d, lambda: os._exit(0)).start()\n"
        "import main_api\n"
        "app = getattr(main_api, 'app', None)\n"
        "if app is None:\n"
        "    app = next((v for v in vars(main_api).values()\n"
        "                if hasattr(v, 'openapi') and hasattr(v, 'routes')), None)\n"
        "if app is None:\n"
        "    raise SystemExit('main_api defines no FastAPI app')\n"
        "import uvicorn\n"
        "uvicorn.run(app, host='127.0.0.1', port=%d, log_level='info')\n"
    ) % (SERVE_MARKER, port, MAX_LIFETIME, port)
    handle = open(LOG, "w", encoding="utf-8")
    try:
        creation = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
            else {"start_new_session": True}
        child = subprocess.Popen([sys.executable, "-u", "-c", serve], cwd=backend,
                                 stdout=handle, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, **creation)
    finally:
        handle.close()
    rel = os.path.relpath(backend, ROOT).replace("\\", "/")
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump({"pid": child.pid, "port": port, "backend": rel}, fh)

    deadline = time.time() + BOOT_WAIT
    while time.time() < deadline:
        if child.poll() is not None:
            tail = log_tail(50)
            kind = "MISSING_DEPENDENCY" if "ModuleNotFoundError" in tail else "BOOT_FAILED"
            print("%s  backend=%s" % (kind, rel))
            print("--- the server died on startup; this is the real error ---")
            print(tail)
            if kind == "MISSING_DEPENDENCY":
                print("NEXT: install_dependencies(working_dir='%s'), then rerun `up`." % rel)
            else:
                print("NEXT: fix the file and line named in the traceback above, "
                      "then rerun `up`. Nothing else can be checked until it boots.")
            return 1, None
        if is_up(port):
            if not quiet:
                print("BOOT_OK  port=%d  backend=%s  pid=%d" % (port, rel, child.pid))
                print("NEXT: `routes` to see the paths and schemas the app really has.")
            return 0, port
        time.sleep(0.4)
    print("BOOT_TIMEOUT after %ds  backend=%s" % (BOOT_WAIT, rel))
    print(log_tail(40))
    print("NEXT: read the log above; the process started but never served "
          "/openapi.json. A blocking call at import time is the usual cause.")
    return 1, None


def ensure_up():
    """Port of a live server, booting one if none answers; None on failure.

    The run's shell session normally keeps the server `up` started; when the
    session was restarted (idle timeout, a one-off sandbox) it is gone, and
    this boots a new one; the SQLite file keeps records between them. A
    recorded port is reused only while our own server still holds it;
    otherwise whatever listens there now belongs to someone else.
    """
    state = read_state()
    port = state.get("port")
    if port and is_probe_server(state.get("pid"), port) and is_up(port):
        return port
    code, port = recording(lambda: boot(quiet=True))
    return port if code == 0 else None


def cmd_routes(argv):
    port = ensure_up()
    if not port:
        return 1
    status, body = request("GET", "/openapi.json", None, port)
    if status != 200:
        print("OPENAPI_FAILED %s\n%s" % (status, body[:400]))
        return 1
    spec = json.loads(body)
    wanted = argv[0].lower() if argv else None
    print("PATHS (method path <- required create fields)")
    for path, operations in sorted(spec.get("paths", {}).items()):
        if wanted and wanted not in path.lower():
            continue
        for method, operation in sorted(operations.items()):
            if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            fields = _body_fields(operation, spec)
            print("  %-6s %s%s" % (method.upper(), path, ("  <- " + fields) if fields else ""))
    print("NEXT: create one of every entity with `req POST <path> {...}`, "
          "parents first, reusing each returned id.")
    return 0


def _body_fields(operation, spec):
    """The request body's own field names and types, resolved one $ref deep."""
    try:
        schema = (operation["requestBody"]["content"]["application/json"]["schema"])
    except (KeyError, TypeError):
        return ""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        schema = spec.get("components", {}).get("schemas", {}).get(name, {})
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for name, definition in properties.items():
        kind = definition.get("type") or definition.get("format") or "?"
        if "anyOf" in definition:
            kind = "|".join(sorted({o.get("type", "?") for o in definition["anyOf"]}))
        parts.append("%s:%s%s" % (name, kind, "*" if name in required else ""))
    return ", ".join(parts[:14]) + (", ..." if len(parts) > 14 else "")


def split_headers(argv):
    """Remove ``-H 'Name: value'`` / ``--header 'Name: value'`` pairs from argv."""
    rest, headers = [], {}
    items = iter(argv)
    for item in items:
        if item in ("-H", "--header"):
            raw = next(items, "")
        elif item.startswith("--header="):
            raw = item[len("--header="):]
        else:
            rest.append(item)
            continue
        # cmd.exe does not strip single quotes, so 'Name: a b' arrives in parts.
        while raw[:1] in ("'", '"') and not (len(raw) > 1 and raw.endswith(raw[0])):
            part = next(items, None)
            if part is None:
                break
            raw += " " + part
        name, sep, value = raw.strip().strip("'\"").partition(":")
        if not sep or not name.strip():
            raise ValueError("a header must look like 'Name: value'")
        headers[name.strip()] = value.strip()
    return rest, headers


def split_form(argv):
    """Remove curl-style ``-F name=value`` / ``-F name=@path`` pairs from argv."""
    rest, fields = [], []
    items = iter(argv)
    for item in items:
        if item in ("-F", "--form"):
            raw = next(items, "")
        elif item.startswith("--form="):
            raw = item[len("--form="):]
        else:
            rest.append(item)
            continue
        name, sep, value = raw.strip().strip("'\"").partition("=")
        if not sep or not name:
            raise ValueError("a form field must look like name=value or name=@path")
        fields.append((name, value))
    return rest, fields


def encode_form(fields):
    """A multipart/form-data body and its Content-Type; ``@path[;type=mime]`` sends a file."""
    import mimetypes
    import uuid
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields:
        head = 'Content-Disposition: form-data; name="%s"' % name
        if value.startswith("@"):
            path, _, kind = value[1:].partition(";type=")
            with open(path, "rb") as handle:
                data = handle.read()
            kind = kind or mimetypes.guess_type(path)[0] or "application/octet-stream"
            head += '; filename="%s"\r\nContent-Type: %s' % (os.path.basename(path), kind)
        else:
            data = value.encode("utf-8")
        parts.append(b"--%s\r\n%s\r\n\r\n%s\r\n" % (boundary.encode(), head.encode("utf-8"), data))
    return b"".join(parts) + b"--%s--\r\n" % boundary.encode(), "multipart/form-data; boundary=" + boundary


def cmd_req(argv):
    try:
        argv, headers = split_headers(argv)
    except ValueError as exc:
        print('BAD_HEADER %s\nNEXT: pass it as -H "Authorization: Bearer <token>".' % exc)
        return 2
    try:
        argv, fields = split_form(argv)
        form = encode_form(fields) if fields else None
    except (ValueError, OSError) as exc:
        print('BAD_FORM %s\nNEXT: pass a field as -F "name=value" and a file as -F "file=@path".' % exc)
        return 2
    if len(argv) < 2:
        print("USAGE: req <METHOD> <PATH> ['JSON' | @file.json | - | -F name=value -F file=@path] [-H 'Name: value']")
        return 2
    method, path = argv[0], argv[1]
    unknown = [arg for arg in argv[2:] if arg.startswith("--")]
    if unknown:
        print("UNKNOWN_OPTION %s\nNEXT: req takes METHOD PATH, then one JSON body (or -F "
              "fields) and -H headers; read the status it prints yourself." % " ".join(unknown))
        return 2
    raw = " ".join(argv[2:]).strip()
    # A quote style that one shell strips and the other passes through must not
    # decide whether a request happens: cmd.exe keeps the single quotes bash
    # would have eaten, and both keep a doubled double-quote.
    while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    body = None
    if raw and form is not None:
        print("BAD_FORM a request carries a JSON body or -F fields, not both.")
        return 2
    if raw == "-":
        raw = sys.stdin.read().strip()
    elif raw.startswith("@"):
        try:
            with open(raw[1:], encoding="utf-8") as handle:
                raw = handle.read().strip()
        except OSError as exc:
            print("BAD_BODY %s\nNEXT: name a JSON file that exists, e.g. @body.json." % exc)
            return 2
    if raw:
        try:
            body = json.loads(raw)
        except ValueError as exc:
            if raw[:1] in "{[" and '"' not in raw:
                advice = "your shell removed the double quotes. Wrap the body in single quotes"
            else:
                advice = "pass the body as one JSON value in single quotes"
            print("BAD_JSON %s\nNEXT: %s: '{\"title\": \"x\"}'. For a body with a single quote "
                  "in it, pass - and send the JSON on stdin (a <<'EOF' here-document), "
                  "or @file.json." % (exc, advice))
            return 2
    elif method.upper() in ("POST", "PUT", "PATCH") and form is None:
        body = {}                       # the literal {} a generated button sends
    port = ensure_up()
    if not port:
        return 1
    status, text = recording(lambda: request(method, path, body, port, extra_headers=headers, form=form))
    # Header names only: a value can be a credential.
    sent = ("  (headers: %s)" % ", ".join(sorted(headers))) if headers else ""
    print("%s %s %s%s" % (status or "CONNECTION_FAILED", method.upper(), path, sent))
    print(text[:BODY_CHARS] + (" ...[truncated]" if len(text) > BODY_CHARS else ""))
    if status == 0:
        print("--- server log (last 40) ---")
        print(log_tail(40))
        print("NEXT: the server is not answering. Rerun `up` and fix what it reports.")
    elif status >= 500:
        print("--- server log (last 40) ---")
        print(log_tail(40))
        print("NEXT: a 5xx is a bug in the app. Fix the file and line in the "
              "traceback above, then `down`, `up`, and repeat THIS request.")
    elif status == 422:
        print("NEXT: the payload disagrees with the schema the app declares. "
              "Run `routes %s` and correct the PAYLOAD - do not weaken the schema."
              % path.strip("/").split("/")[0])
    elif status == 404:
        print("NEXT: that path does not exist. Run `routes` and use a real one.")
    elif status in (401, 403):
        if headers:
            print("NEXT: the credential was refused. Log in again through the app's own "
                  "endpoint and repeat with the token its response returns.")
        else:
            print("NEXT: the endpoint requires auth. Register and log in through the "
                  "app's own endpoints (`routes auth`), then repeat this request with "
                  '-H "Authorization: Bearer <access_token from the login response>".')
    elif 300 <= status < 400:
        print("NEXT: unfollowed redirect. Retry against the Location path "
              "(most often the same path with a trailing slash).")
    elif status >= 400:
        print("NEXT: read the body above; it names the rejected field or rule.")
    elif 200 <= status < 300 and method.upper() in ("POST", "PUT", "PATCH"):
        print("NEXT: a 2xx is not proof of effect. GET the entity back and check "
              "the field this call was supposed to change actually changed.")
    return 0 if 200 <= status < 300 else 1


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    action, rest = argv[0], argv[1:]
    if action == "up":
        return cmd_up()
    if action == "down":
        return cmd_down()
    if action == "routes":
        return cmd_routes(rest)
    if action == "req":
        return cmd_req(rest)
    if action == "log":
        try:
            count = int(rest[0]) if rest else 40
        except ValueError:
            count = 40
        print(log_tail(count))
        return 0
    print("UNKNOWN: %s. Use up | routes | req | log | down." % action)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''
