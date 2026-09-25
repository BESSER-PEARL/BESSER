"""Shell runbook: the procedure that turns shell access into verification.

Shell access on its own does not make a model verify anything: models given
a ``test_api`` tool rarely call it, and an agent with unrestricted shell can
make hundreds of calls without ever booting the application.

Two things are missing, and this module supplies both.

**1. A boot that cannot block.** ``run_command`` is
``subprocess.run(..., capture_output=True, timeout=120)``. That call returns
only when the child exits *and* the inherited pipes reach EOF, so a server
started in the foreground burns the whole 120-second budget and returns
nothing; a server started with ``&`` but without redirecting its output keeps
the pipe open and does exactly the same. Booting a web app from this tool
therefore requires the model to know a detach idiom that differs per platform
(``nohup … > log 2>&1 &`` on Linux, ``start /b … > log 2>&1`` on Windows) and
to get it right first time - with a 120-second dead end as the penalty for
getting it wrong. ``.besser_probe.py`` does the detaching in Python, so the
model issues one literal command that behaves identically on both platforms.

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

import logging
import os

logger = logging.getLogger(__name__)

PROBE_FILENAME = ".besser_probe.py"

# Environment kill switch. Default on whenever shell tools are enabled;
# ``0`` gives the shell without the procedure (an escape hatch if the section
# misfires on a stack the probe does not understand).
_RUNBOOK_ENV = "BESSER_LLM_SHELL_RUNBOOK"


def runbook_enabled() -> bool:
    return os.environ.get(_RUNBOOK_ENV, "1").strip().lower() not in ("0", "false", "no")


def _find_backend(output_dir: str) -> str | None:
    """Workspace-relative path of the generated FastAPI service, if present.

    Same signature as ``constructibility._fastapi_backends``: ``main_api.py``
    next to ``sql_alchemy.py``. The probe script only knows how to drive that
    scaffold family, so on any other stack the runbook renders nothing rather
    than handing the model commands that cannot work.
    """
    skip = {"node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build"}
    try:
        for root, dirs, files in os.walk(output_dir):
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith(".besser_"))
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
        with open(os.path.join(output_dir, PROBE_FILENAME), "w",
                  encoding="utf-8", newline="\n") as handle:
            handle.write(PROBE_SCRIPT)
        return True
    except OSError:
        logger.debug("runbook: could not install the probe script", exc_info=True)
        return False


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
python {PROBE_FILENAME} req POST /<entity> {{"field": "value"}}
python {PROBE_FILENAME} req POST /<entity>/<id>/<action> {{}}
python {PROBE_FILENAME} log 60                   # last 60 lines of the server log
python {PROBE_FILENAME} down                     # stop it
```

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
    python .besser_probe.py req POST /book {"title": "x"}
    python .besser_probe.py log 60
    python .besser_probe.py down

Exists because run_command is a blocking subprocess with a 120-second cap:
a foreground server consumes the whole budget and returns nothing, and a
backgrounded one that still holds the inherited stdout pipe does the same.
This starts the server as a detached child with its output on disk, so the
command returns in a second or two on every platform.
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
SKIP = {"node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build"}
MAX_LIFETIME = 1800          # a forgotten server reaps itself after 30 minutes
BOOT_WAIT = 45
BODY_CHARS = 1800


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


def request(method, path, body, port, timeout=25, _hops=0):
    url = "http://127.0.0.1:%d%s" % (port, path if path.startswith("/") else "/" + path)
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
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
            return request(follow, location, body, port, timeout, _hops + 1)
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:                    # connection refused, reset, timeout
        return 0, "%s: %s" % (type(exc).__name__, exc)


def is_up(port):
    status, _ = request("GET", "/openapi.json", None, port, timeout=3)
    return status == 200


def cmd_down(quiet=False):
    state = read_state()
    pid = state.get("pid")
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
    except Exception as exc:
        print("STOP_FAILED %s: %s" % (type(exc).__name__, exc))
    try:
        os.remove(STATE)
    except OSError:
        pass
    if not quiet:
        print("STOPPED pid=%s" % pid)
    return 0


def cmd_up():
    backend = find_backend()
    if backend is None:
        print("NO_BACKEND: no directory holds both main_api.py and sql_alchemy.py.")
        return 2
    cmd_down(quiet=True)
    port = free_port()
    serve = (
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
    ) % (MAX_LIFETIME, port)
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
            return 1
        if is_up(port):
            print("BOOT_OK  port=%d  backend=%s  pid=%d" % (port, rel, child.pid))
            print("NEXT: `routes` to see the paths and schemas the app really has.")
            return 0
        time.sleep(0.4)
    print("BOOT_TIMEOUT after %ds  backend=%s" % (BOOT_WAIT, rel))
    print(log_tail(40))
    print("NEXT: read the log above; the process started but never served "
          "/openapi.json. A blocking call at import time is the usual cause.")
    return 1


def cmd_routes(argv):
    state = read_state()
    port = state.get("port")
    if not port or not is_up(port):
        print("NOT_RUNNING\nNEXT: `up` first.")
        return 2
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


def cmd_req(argv):
    if len(argv) < 2:
        print("USAGE: req <METHOD> <PATH> [JSON]")
        return 2
    method, path = argv[0], argv[1]
    raw = " ".join(argv[2:]).strip()
    # A quote style that one shell strips and the other passes through must not
    # decide whether a request happens: cmd.exe keeps the single quotes bash
    # would have eaten, and both keep a doubled double-quote.
    while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    body = None
    if raw:
        try:
            body = json.loads(raw)
        except ValueError as exc:
            print("BAD_JSON %s\nNEXT: pass the body as one JSON object, e.g. "
                  '{"title": "x"}. Quote it for your shell.' % exc)
            return 2
    elif method.upper() in ("POST", "PUT", "PATCH"):
        body = {}                       # the literal {} a generated button sends
    state = read_state()
    port = state.get("port")
    if not port:
        print("NOT_RUNNING\nNEXT: `up` first.")
        return 2
    status, text = request(method, path, body, port)
    print("%s %s %s" % (status or "CONNECTION_FAILED", method.upper(), path))
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
        print("NEXT: the endpoint requires auth. Register and log in through the "
              "app's own endpoints (`routes auth`), then repeat with the token.")
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
