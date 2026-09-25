"""Bounded declarative API workflows on a disposable FastAPI/SQLite copy.

The caller must enforce the existing opt-in runtime/import-smoke setting. This
executes generated application code, just like the constructibility probe; it
is not an OS security sandbox or proof of complete specification coverage.
No commands, external URLs, credentials, or Python expressions are accepted.
The child imports no BESSER package and runs this same file as a script.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, unquote, urlsplit


_MARKER = "BESSER_API_SCENARIO_REPORT:"
_PROBE_TIMEOUT_SECONDS = 60
_MAX_REQUESTS = 20
_MAX_INPUT_BYTES = 64_000
_BODY_CHARS = 2000
_MAX_REPORTED_FAILURES = 5
_NO_JSON = object()
_REFERENCE = re.compile(r"\{\{(\d+)(?:\.([^{}]+))?\}\}")
_SCOPE = "Only the submitted scenario was checked; this is not complete specification verification."


def _error(message, boot="not_started", **extra):
    return {"status": "error", "boot": boot, "error": str(message)[:2000],
            "responses": [], "assertion_failures": [], "scope": _SCOPE, **extra}


def _local_path(path):
    if not isinstance(path, str) or not path.startswith("/") or len(path) > 2048:
        raise ValueError("path must be a local absolute API path of at most 2048 characters")
    parsed = urlsplit(path)
    decoded = unquote(parsed.path)
    if (parsed.scheme or parsed.netloc or parsed.fragment or decoded.startswith("//")
            or "\\" in unquote(path) or ".." in decoded.split("/")
            or any(ord(character) < 32 for character in unquote(path))):
        raise ValueError("external URLs, fragments, traversal and control characters are not allowed in path")
    return path


def _validate_requests(requests):
    if not isinstance(requests, list) or not 1 <= len(requests) <= _MAX_REQUESTS:
        raise ValueError(f"requests must contain 1 to {_MAX_REQUESTS} request objects")
    serialized = json.dumps(requests, allow_nan=False)
    if len(serialized.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise ValueError(f"scenario exceeds {_MAX_INPUT_BYTES} bytes")
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or set(request) - {"method", "path", "json", "expected_status", "expected_fields"}:
            raise ValueError(f"request {index} has an invalid shape or unsupported keys")
        if request.get("method") not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"request {index}: method must be GET, POST, PUT, PATCH or DELETE")
        _local_path(request.get("path"))
        if "expected_status" in request:
            statuses = request["expected_status"]
            statuses = statuses if isinstance(statuses, list) else [statuses]
            if not statuses or len(statuses) > 100 or any(type(s) is not int or not 100 <= s <= 599 for s in statuses):
                raise ValueError(f"request {index}: expected_status must be an HTTP status or nonempty list")
        fields = request.get("expected_fields", {})
        if not isinstance(fields, dict) or len(fields) > 50 or any(
            not isinstance(path, str) or not path or len(path) > 300 or any(not part for part in path.split("."))
            for path in fields
        ):
            raise ValueError(f"request {index}: expected_fields must map dotted JSON paths to literals")
    return requests


def _json_path(body, path):
    current = body
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdecimal() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError(f"JSON path {path!r} was not present")
    return current


def _resolve_references(value, previous, *, in_path=False):
    if isinstance(value, dict):
        return {key: _resolve_references(item, previous) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, previous) for item in value]
    if not isinstance(value, str):
        return value

    def resolve(match):
        index = int(match.group(1))
        if index >= len(previous) or previous[index] is _NO_JSON:
            raise ValueError(f"response {index} has no preceding JSON value to reference")
        return _json_path(previous[index], match.group(2) or "")

    match = _REFERENCE.fullmatch(value)
    if match and not in_path:
        return copy.deepcopy(resolve(match))

    def substitute(match):
        replacement = resolve(match)
        if isinstance(replacement, (dict, list)):
            raise ValueError("embedded references must resolve to scalar values")
        text = json.dumps(replacement) if replacement is None or isinstance(replacement, bool) else str(replacement)
        return quote(text, safe="") if in_path else text

    return _REFERENCE.sub(substitute, value)


def _same_literal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _same_literal(actual[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_literal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _bounded(value):
    encoded = json.dumps(value, ensure_ascii=True, default=str)
    return value if len(encoded) <= _BODY_CHARS else encoded[:_BODY_CHARS] + " [truncated]"


def confirmed_create_paths(report: dict) -> set[str]:
    """Routes with a created record subsequently observed through a real GET.

    A 200/health response or a model's assertion alone is insufficient. This
    narrow check recognizes the generated API's id-bearing record envelopes;
    unfamiliar/truncated shapes stay unverified instead of guessing.
    """
    # Boot is a whole-report precondition: nothing a dead app said is evidence.
    # The scenario's overall verdict is NOT: a report is "failed" if ANY
    # request failed, and scenarios deliberately mix happy paths with negative
    # probes. The loop below judges each POST/GET pair on its own, so one
    # unrelated failure cannot discard a record that was created and read back.
    if report.get("boot") != "ok":
        return set()

    def records(value):
        if isinstance(value, dict):
            if type(value.get("id")) in (int, str):
                return [value]
            return [record for child in value.values() for record in records(child)]
        if isinstance(value, list):
            return [record for child in value for record in records(child)]
        return []

    responses = report.get("responses", [])
    confirmed = set()
    for index, created in enumerate(responses):
        if (created.get("method") != "POST" or not isinstance(created.get("status"), int)
                or not 200 <= created["status"] < 300 or created.get("truncated") or created.get("failures")):
            continue
        path = created.get("path", "").split("?", 1)[0].rstrip("/")
        for read in responses[index + 1:]:
            read_path = read.get("path", "").split("?", 1)[0].rstrip("/")
            if (read.get("method") != "GET" or not isinstance(read.get("status"), int)
                    or not 200 <= read["status"] < 300 or read.get("truncated") or read.get("failures")
                    or not (read_path == path or read_path.startswith(path + "/"))):
                continue
            for inserted in records(created.get("json")):
                for observed in records(read.get("json")):
                    common = set(inserted) & set(observed) - {"id"}
                    if (_same_literal(inserted["id"], observed["id"])
                            and any(isinstance(inserted[key], (str, int, float, bool))
                                    and _same_literal(inserted[key], observed[key]) for key in common)):
                        confirmed.add(path)
    return confirmed


def probe_api_scenario(output_dir: str, requests: list[dict], *, backend: str | None = None) -> dict:
    """Run a scenario in one fresh app/database; the caller owns runtime opt-in.

    ``backend`` selects a relative service directory when a workspace contains
    multiple deterministic FastAPI backends. References such as ``{{0.room.id}}``
    use preceding response JSON only, preserving types for whole JSON values.
    """
    from besser.spec_driven_agent.validation.constructibility import _fastapi_backends, _SKIP_DIRS
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env

    try:
        _validate_requests(requests)
        root = Path(output_dir).resolve()
        candidates = [Path(folder).resolve() for folder in _fastapi_backends(str(root))]
        candidates = [folder for folder in candidates if folder.is_relative_to(root)]
        if backend is not None:
            if not isinstance(backend, str) or Path(backend).is_absolute() or ".." in backend.replace("\\", "/").split("/"):
                raise ValueError("backend must be an in-workspace relative directory")
            chosen = (root / backend).resolve()
            if not chosen.is_relative_to(root) or chosen not in candidates:
                raise ValueError("backend does not identify an in-workspace generated FastAPI service")
            candidates = [chosen]
        if len(candidates) != 1:
            return _error("Select exactly one generated FastAPI backend", backends=[
                folder.relative_to(root).as_posix() for folder in candidates[:20]])
        folder = candidates[0]
        relative = folder.relative_to(root).as_posix()
        with tempfile.TemporaryDirectory(prefix="besser_api_probe_") as work:
            scratch = Path(work)
            app_dir = scratch / "app"

            def ignore(directory, names):
                excluded = shutil.ignore_patterns(*_SKIP_DIRS, ".*", "venv", "*.db", "*.db-*", "*.sqlite*", "*.pem", "*.key")(directory, names)
                for name in names:
                    if name not in excluded and Path(directory, name).is_symlink():
                        raise ValueError("API probe refuses symlinks in application source")
                return excluded

            shutil.copytree(folder, app_dir, ignore=ignore)
            env = _safe_subprocess_env()
            env.update(DATABASE_URL="sqlite:///" + (scratch / "probe.db").as_posix(),
                       BESSER_API_PROBE_SCRATCH=str(scratch), TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch))
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=app_dir,
                    env=env, input=json.dumps(requests), capture_output=True, text=True,
                    timeout=min(_PROBE_TIMEOUT_SECONDS, 60),
                )
            except subprocess.TimeoutExpired:
                return _error(f"API scenario timed out after {min(_PROBE_TIMEOUT_SECONDS, 60)}s", "timeout", backend=relative)
            for line in reversed((result.stdout or "").splitlines()):
                if line.startswith(_MARKER):
                    report = json.loads(line[len(_MARKER):])
                    report["backend"] = relative
                    return report
            return _error((result.stderr or "child returned no scenario report")[-2000:], "probe_error", backend=relative)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        return _error(f"{type(exc).__name__}: {exc}")


def _install_guards(scratch):
    """Guard ordinary Python side effects; this is not an OS/container sandbox."""
    import sqlalchemy
    from sqlalchemy.engine import make_url

    def private_path(filename):
        if isinstance(filename, int):  # stdout/stderr and already-open handles
            return
        path = Path(os.fsdecode(filename)).resolve()
        if not path.is_relative_to(scratch):
            raise RuntimeError("API probe refused a write/database outside its scratch directory")

    original_create_engine = sqlalchemy.create_engine

    def create_engine(url, *args, **kwargs):
        parsed = make_url(url)
        if parsed.get_backend_name() != "sqlite":
            raise RuntimeError("API probe permits only an isolated SQLite database")
        return original_create_engine(url, *args, **kwargs)

    sqlalchemy.create_engine = create_engine

    def audit(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind",
                     "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            raise RuntimeError("API probe refuses network connections and external commands")
        if event == "sqlite3.connect":
            database = os.fsdecode(args[0])
            if database not in {"", ":memory:"}:
                if database.startswith("file:"):
                    database = unquote(database[5:].split("?", 1)[0])
                private_path(database)
        elif event == "open":
            filename, mode, flags = args
            if (mode and any(character in mode for character in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                private_path(filename)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate"}:
            private_path(args[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            private_path(args[0])
            private_path(args[1])

    sys.addaudithook(audit)


async def _run_requests(app, requests):
    import httpx

    previous, responses, failures = [], [], []
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://besser-probe.invalid", follow_redirects=False) as client:
            for index, request in enumerate(requests):
                item = {"index": index, "method": request["method"], "path": request["path"], "failures": []}
                body = _NO_JSON
                try:
                    path = _local_path(_resolve_references(request["path"], previous, in_path=True))
                    item["path"] = path
                    kwargs = {"json": _resolve_references(request["json"], previous)} if "json" in request else {}
                    try:
                        response = await client.request(request["method"], path, **kwargs)
                    except Exception as exc:
                        import traceback
                        locations = []
                        for frame in traceback.extract_tb(exc.__traceback__):
                            try:
                                filename = Path(frame.filename).resolve().relative_to(Path.cwd().resolve()).as_posix()
                            except ValueError:
                                continue
                            locations.append({"path": filename, "line": frame.lineno, "function": frame.name})
                        item["status"] = 500
                        item["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000], "locations": locations[-5:]}
                        raise
                    item["status"] = response.status_code
                    try:
                        body = response.json()
                        item["json"] = _bounded(body)
                        item["truncated"] = len(json.dumps(body, ensure_ascii=True)) > _BODY_CHARS
                    except ValueError:
                        item["text"] = response.text[:_BODY_CHARS]
                        item["truncated"] = len(response.text) > _BODY_CHARS
                    expected = request.get("expected_status", list(range(200, 300)))
                    expected = expected if isinstance(expected, list) else [expected]
                    if response.status_code not in expected:
                        item["failures"].append(f"expected HTTP {request.get('expected_status', '2xx')}, got {response.status_code}")
                    for path, expected_value in request.get("expected_fields", {}).items():
                        try:
                            actual = _json_path(body, path)
                            if not _same_literal(actual, expected_value):
                                item["failures"].append(f"{path}: expected {_bounded(expected_value)!r}, got {_bounded(actual)!r}")
                        except ValueError as exc:
                            item["failures"].append(str(exc))
                except Exception as exc:
                    item["failures"].append(f"{type(exc).__name__}: {exc}"[:1000])
                previous.append(body)
                count = len(item["failures"])
                item["failure_count"] = count
                item["failures"] = [message[:400] for message in item["failures"][:_MAX_REPORTED_FAILURES]]
                if count > _MAX_REPORTED_FAILURES:
                    item["omitted_failures"] = count - _MAX_REPORTED_FAILURES
                failures.extend({"index": index, "message": message} for message in item["failures"])
                responses.append(item)
    return {"status": "failed" if failures else "passed", "boot": "ok", "responses": responses,
            "assertion_failures": failures, "scope": _SCOPE}


def _worker():
    scratch_value = os.environ.get("BESSER_API_PROBE_SCRATCH")
    if not scratch_value:
        return _error("worker requires its private scratch directory")
    scratch = Path(scratch_value).resolve()
    if not scratch.name.startswith("besser_api_probe_") or Path.cwd().resolve() != scratch / "app":
        return _error("worker is not in its assigned application copy")
    loop = None
    try:
        requests = _validate_requests(json.loads(sys.stdin.read(_MAX_INPUT_BYTES + 1)))
        # The event loop creates its own local wakeup sockets. Create it before
        # denying network operations initiated by the generated application.
        loop = asyncio.new_event_loop()
        _install_guards(scratch)
        sys.path.insert(0, str(Path.cwd()))
        import main_api
        app = getattr(main_api, "app", None)
        if app is None:
            apps = [value for value in vars(main_api).values() if hasattr(value, "openapi") and hasattr(value, "router")]
            app = apps[0] if len(apps) == 1 else None
        if app is None or not hasattr(app, "router"):
            return _error("main_api defines no unambiguous FastAPI application", "no_app")
        return loop.run_until_complete(_run_requests(app, requests))
    except ModuleNotFoundError as exc:
        return _error(f"{type(exc).__name__}: {exc}", "missing_dependency")
    except BaseException as exc:
        return _error(f"{type(exc).__name__}: {exc}", "boot_error")
    finally:
        if loop is not None:
            loop.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use probe_api_scenario() from the opted-in runtime harness")
    print(_MARKER + json.dumps(_worker(), ensure_ascii=True, default=str))
