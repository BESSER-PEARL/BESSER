"""Bounded declarative API workflows on a disposable FastAPI/SQLite copy.

The caller must enforce the existing opt-in runtime/import-smoke setting. This
executes generated application code, just like the constructibility probe, in
the run sandbox with no network; it is not proof of complete specification
coverage.
No commands, external URLs or Python expressions are accepted. Request headers
are limited to an allow-list, and the values of sensitive ones, credential
response fields and JWT-shaped strings never appear in the report.
The child imports no BESSER package and runs this same file as a script, so it
masks the scenario's own secrets with ``_scrub`` rather than
``besser.utilities.secret_redaction``.
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
_REQUEST_KEYS = frozenset({"method", "path", "json", "headers", "expected_status", "expected_fields"})
_MAX_INPUT_BYTES = 64_000
_BODY_CHARS = 2000
_MAX_REPORTED_FAILURES = 5
_NO_JSON = object()
_MISSING = object()
_REFERENCE = re.compile(r"\{\{(\d+)(?:\.([^{}]+))?\}\}")
_MAX_HEADERS = 10
_HEADER_VALUE_CHARS = 4096
_HEADER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}")
# Everything else a client legitimately sets on an API call is an X-* header
# (X-API-Key, X-Tenant-ID, ...). Host, Cookie (the client keeps its own jar
# within a scenario), framing and hop-by-hop headers are refused, as are the
# X-* headers proxies use to assert a client identity or rewrite the request.
_PLAIN_HEADERS = frozenset({"accept", "accept-language", "content-type", "if-match", "if-none-match"})
_ALLOWED_HEADERS = _PLAIN_HEADERS | {"authorization", "idempotency-key"}
_DENIED_X_HEADERS = frozenset({"x-real-ip", "x-http-method-override", "x-method-override",
                               "x-original-url", "x-rewrite-url", "x-original-method"})
_SENSITIVE_PARTS = ("auth", "key", "token", "secret", "session", "signature", "password", "credential")
_REDACTED = "[REDACTED]"
# A response field holding a credential, judged on its snake/kebab/camel words: any secret
# word (``passwordNew``, ``password_reset_code``), ``token`` that is not a page cursor
# (``next_page_token``), or api/access/private/secret key, auth/reset code or session id, unless
# the next word only describes or identifies it (``password_changed_at``, ``token_count``, ``api_key_name``).
# A run of non-ASCII letters is a word of its own, so no letter is dropped and the ASCII
# camelCase words around it stay apart (``überAdmin``: ü, ber, admin).
_KEY_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[^\W\d_A-Za-z]+")
_SECRET_WORDS = frozenset({"password", "passwd", "passphrase", "secret", "jwt", "apikey", "credential",
                           "credentials", "otp"})
_DESCRIBING_WORDS = frozenset({"changed", "updated", "created", "at", "hint", "expires", "expiry", "expiration",
                               "length", "policy", "required", "strength", "count", "type", "santa",
                               "question", "sent", "id", "name", "prefix", "suffix", "last", "label",
                               "attempts", "min", "max"})
_CURSOR_WORDS = frozenset({"page", "next", "prev", "previous", "continuation", "cursor", "sync"})
_SECRET_PAIRS = {"key": frozenset({"api", "access", "private", "secret"}),
                 "code": frozenset({"auth", "authorization", "reset", "verification"}),
                 "id": frozenset({"session"})}
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}")
_SCOPE = "Only the submitted scenario was checked; this is not complete specification verification."
# ``boot`` values meaning the scenario never ran: the sandbox was unavailable,
# the worker timed out, or the harness lacks a dependency (httpx). Unknowns,
# not application failures - see ``runtime_check_unverified``.
DID_NOT_RUN_BOOTS = frozenset({"probe_error", "timeout", "missing_dependency"})


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


def _header_allowed(name):
    lowered = name.lower()
    if lowered in _ALLOWED_HEADERS:
        return True
    return lowered.startswith("x-") and lowered not in _DENIED_X_HEADERS and not lowered.startswith("x-forwarded-")


def _validate_headers(headers, index):
    if not isinstance(headers, dict) or len(headers) > _MAX_HEADERS:
        raise ValueError(f"request {index}: headers must be an object of at most {_MAX_HEADERS} entries")
    for name, value in headers.items():
        if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name) or not _header_allowed(name):
            raise ValueError(
                f"request {index}: header {str(name)[:64]!r} is not allowed; use Authorization, Accept, "
                "Content-Type, Accept-Language, If-Match, If-None-Match, Idempotency-Key or an X-* API header")
        _header_value(value, index, name)


def _header_value(value, index, name):
    # The message names the header, never the value: it may be a credential.
    if (not isinstance(value, str) or len(value) > _HEADER_VALUE_CHARS
            or any(ord(character) < 32 or ord(character) > 126 for character in value)):
        raise ValueError(f"request {index}: header {name} must be printable ASCII text of at most "
                         f"{_HEADER_VALUE_CHARS} characters")
    return value


def _sensitive_header(name):
    return not isinstance(name, str) or any(part in name.lower() for part in _SENSITIVE_PARTS)


def redact_headers(headers):
    """``headers`` with credential-bearing values (Authorization, X-API-Key, ...) hidden."""
    if not isinstance(headers, dict):
        return headers
    return {name: _REDACTED if _sensitive_header(name) else value for name, value in headers.items()}


def redact_scenario_headers(requests):
    """A copy of ``requests`` safe to show or log: header values hidden."""
    if not isinstance(requests, list):
        return requests
    return [dict(request, headers=redact_headers(request["headers"]))
            if isinstance(request, dict) and "headers" in request else request
            for request in requests]


def name_words(name: str) -> list[str]:
    """The lowercase words of a snake_case, camelCase or kebab-case name."""
    return [word.lower() for word in _KEY_WORDS.findall(name)]


def _credential_key(key):
    words = name_words(key) if isinstance(key, str) else []
    for index, word in enumerate(words):
        secret = (word in _SECRET_WORDS
                  or (word == "token" and not _CURSOR_WORDS.intersection(words[:index]))
                  or (index > 0 and words[index - 1] in _SECRET_PAIRS.get(word, ())))
        if secret and (index + 1 == len(words) or words[index + 1] not in _DESCRIBING_WORDS):
            return True
    return False


def _credential_field(key, value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)) or value == "":
        return False
    if not isinstance(value, str) and isinstance(key, str) and name_words(key)[-1:] == ["id"]:
        return False  # a numeric id references a record (a conference Session), it is no minted secret
    return _credential_key(key)


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


def _child_names(parent, key) -> list:
    """The names a value at ``parent.key`` goes by: ``key`` and, unless ``key`` repeats
    the parent's last word, the parent's word before it (``tokens.access`` is ``token_access``)."""
    if not isinstance(parent, str) or not isinstance(key, str):
        return [key]
    head = _singular(parent)
    last = name_words(head)[-1:]
    return [key] if not last or last[0] in name_words(key) else [key, f"{head}_{key}"]


def _path_names(path: str) -> list:
    """The names of the value at a dotted JSON path; a list item goes by its list's name."""
    names: list = []
    for part in path.split(".") if path else []:
        names = [_singular(name) for name in names] if part.isdecimal() else _child_names(
            names[0] if names else None, part)
    return names


def _credential_reference(match):
    return any(_credential_key(name) for name in _path_names(match.group(2) or ""))


def _sent_credentials(request, previous):
    """Credential-named response values ``request`` names by reference, e.g. /reset/{{1.reset_token}}."""
    sent = set()
    for match in _REFERENCE.finditer(json.dumps(
            [request.get(key) for key in ("path", "headers", "json", "expected_fields")])):
        if _credential_reference(match) and int(match.group(1)) < len(previous):
            body = previous[int(match.group(1))]
            value = _walk(body, _reference_path(body, match.group(2) or ""))
            if isinstance(value, str) and len(value) >= 8:
                sent.add(value)
    return sent


def _scrub(value, secrets, names=()):
    """``value`` with credentials hidden; ``names`` are what it is called (see ``_child_names``)."""
    if any(_credential_field(name, value) for name in names):
        return _REDACTED
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, _REDACTED)
        return _JWT.sub(_REDACTED, value)
    if isinstance(value, list):
        return [_scrub(item, secrets, [_singular(name) for name in names if isinstance(name, str)]) for item in value]
    if isinstance(value, dict):
        # A validation error echoes the submitted value: {"loc": [..., "password"], "input": ...}.
        loc = value.get("loc")
        field = next((part for part in reversed(loc) if isinstance(part, str)), None) if isinstance(loc, list) else None
        echoed = {"input", "ctx"} if _credential_key(field) else set()
        parent = names[0] if names else None
        return {key: _REDACTED if key in echoed else _scrub(item, secrets, _child_names(parent, key))
                for key, item in value.items()}
    return value


def _validate_requests(requests):
    if not isinstance(requests, list):
        raise ValueError("requests must be a list of request objects")
    if not 1 <= len(requests) <= _MAX_REQUESTS:
        raise ValueError(f"requests must contain 1 to {_MAX_REQUESTS} request objects, got {len(requests)}; "
                         "split a longer workflow into scenarios with their own scenario_id")
    serialized = json.dumps(requests, allow_nan=False)
    if len(serialized.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise ValueError(f"scenario exceeds {_MAX_INPUT_BYTES} bytes")
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be an object")
        unsupported = sorted(set(request) - _REQUEST_KEYS, key=str)
        if unsupported:
            raise ValueError(f"request {index}: unsupported key(s) {', '.join(map(repr, unsupported))}; "
                             f"a request takes {', '.join(sorted(_REQUEST_KEYS))}")
        if request.get("method") not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"request {index}: method must be GET, POST, PUT, PATCH or DELETE")
        _local_path(request.get("path"))
        if "headers" in request:
            _validate_headers(request["headers"], index)
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


def _walk(body, path):
    current = body
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdecimal() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _holds(value, path):
    if isinstance(value, list):
        return any(isinstance(item, dict) and _walk(item, path) is not _MISSING for item in value)
    return _walk(value, path) is not _MISSING


def _sole_wrapper(body, path):
    """The one dict key holding ``path``, as in {"dataset": {"id": 1}}, or None.

    With {"meta": {"id": ...}, "data": [...]} a guess could name the wrong id.
    """
    if not isinstance(body, dict):
        return None
    holders = [key for key, child in body.items()
               if isinstance(child, (dict, list)) and _holds(child, path)]
    if len(holders) == 1 and isinstance(body[holders[0]], dict) and "." not in holders[0]:
        return holders[0]
    return None


def _reference_path(body, path):
    """The path a ``{{N.path}}`` reference reads: ``path``, or it under the one wrapper holding it."""
    wrapper = _sole_wrapper(body, path) if _walk(body, path) is _MISSING else None
    return path if wrapper is None else f"{wrapper}.{path}"


def _json_path(body, path):
    current = _walk(body, path)
    if current is _MISSING:
        hint = ""
        if isinstance(body, dict) and body:
            wrapper = _sole_wrapper(body, path)
            if wrapper is not None:
                hint = f"; did you mean {wrapper + '.' + path!r}?"
            else:
                keys = ", ".join(map(repr, list(body)[:10])) + (", ..." if len(body) > 10 else "")
                hint = f"; the top-level keys are {keys}"
        elif isinstance(body, list):
            hint = f"; the body is a list of {len(body)} item(s)"
            if body and not path.split(".")[0].isdecimal():
                hint += f", so index one, e.g. {'0.' + path!r}"
        raise ValueError(f"JSON path {path!r} was not present{hint}")
    return current


def _resolve_references(value, previous, *, in_path=False, in_header=False, redact=False, notes=None):
    """Substitute ``{{N.path}}`` references from earlier responses.

    A path missing from a response that wraps the record under one key
    (``{"todolist": {"id": 1}}``) resolves through that key, and ``notes``,
    when given, records the path actually read.
    """
    if isinstance(value, dict):
        return {key: _resolve_references(item, previous, notes=notes) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, previous, notes=notes) for item in value]
    if not isinstance(value, str):
        return value

    def resolve(match):
        index = int(match.group(1))
        if index >= len(previous) or previous[index] is _NO_JSON:
            raise ValueError(f"response {index} has no preceding JSON value to reference")
        body, path = previous[index], match.group(2) or ""
        read = _reference_path(body, path)
        if read != path and notes is not None:
            notes.append(f"`{{{{{index}.{path}}}}}` resolved as `{{{{{index}.{read}}}}}`")
        return _json_path(body, read)

    match = _REFERENCE.fullmatch(value)
    if match and not (in_path or in_header):
        return copy.deepcopy(resolve(match))

    def substitute(match):
        if redact and _credential_reference(match):
            return _REDACTED
        replacement = resolve(match)
        if isinstance(replacement, (dict, list)):
            raise ValueError("embedded references must resolve to scalar values")
        text = json.dumps(replacement) if replacement is None or isinstance(replacement, bool) else str(replacement)
        return quote(text, safe="") if in_path else text

    return _REFERENCE.sub(substitute, value)


def _expected_shown(value, previous):
    """An expected value that resolved, as a failure message may quote it: each
    referenced value has its credentials hidden under the names the reference gives it."""
    if isinstance(value, dict):
        return {key: _expected_shown(item, previous) for key, item in value.items()}
    if isinstance(value, list):
        return [_expected_shown(item, previous) for item in value]
    match = _REFERENCE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return _resolve_references(value, previous, redact=True)
    body = previous[int(match.group(1))]
    read = _reference_path(body, match.group(2) or "")
    return _scrub(copy.deepcopy(_walk(body, read)), (), _path_names(read))


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
            # The worker compared the unredacted bodies; a report without that comparison
            # (an older saved one) is judged on its redacted copies.
            reads_back = read.get("reads_back")
            if (created.get("index") in reads_back if isinstance(reads_back, list)
                    else _reads_back(created.get("json"), read.get("json"))):
                confirmed.add(path)
    return confirmed


# Keys that make a body a record: whatever else it holds (a status, an error field) is its data.
_RECORD_KEYS = ("id", "pk", "uuid", "_id")
_FAILURE_FLAGS = ("success", "ok", "succeeded")
_FAILURE_KEYS = ("error", "errors", "detail")
_FAILURE_STATUSES = frozenset({"error", "failed", "failure"})


def declares_failure(body) -> bool:
    """Whether a response body is a failure envelope: no record, and an explicit flag that
    the call did not act (``success``/``ok``/``result`` false, an ``error`` or ``detail``,
    ``status: "error"``), at the top or in a wrapper. A list fails when all its items do."""
    if isinstance(body, list):
        return bool(body) and all(declares_failure(item) for item in body)
    if not isinstance(body, dict) or any(body.get(key) not in (None, "") for key in _RECORD_KEYS):
        return False
    # ``result: false`` is the scaffold's "declined to act"; a result of 0 is a value.
    if any(body.get(key, True) == 0 or str(body.get(key)).strip().lower() == "false" for key in _FAILURE_FLAGS)             or str(body.get("result")).strip().lower() == "false":
        return True
    status = body.get("status")
    if any(body.get(key) for key in _FAILURE_KEYS) or (
            isinstance(status, str) and status.strip().lower() in _FAILURE_STATUSES):
        return True
    return any(declares_failure(value) for value in body.values() if isinstance(value, (dict, list)))


def _records(value):
    if isinstance(value, dict):
        if type(value.get("id")) in (int, str):
            return [value]
        return [record for child in value.values() for record in _records(child)]
    if isinstance(value, list):
        return [record for child in value for record in _records(child)]
    return []


def _reads_back(created, observed) -> bool:
    """Whether ``observed`` holds a record of ``created``: the same id and one equal plain field."""
    for inserted in _records(created):
        for record in _records(observed):
            common = set(inserted) & set(record) - {"id"}
            # Two redacted values are equal whatever they hid, so they prove nothing.
            if (_same_literal(inserted["id"], record["id"])
                    and any(isinstance(inserted[key], (str, int, float, bool)) and inserted[key] != _REDACTED
                            and _same_literal(inserted[key], record[key]) for key in common)):
                return True
    return False


def probe_api_scenario(output_dir: str, requests: list[dict], *, backend: str | None = None) -> dict:
    """Run a scenario in one fresh app/database; the caller owns runtime opt-in.

    ``backend`` selects a relative service directory when a workspace contains
    multiple deterministic FastAPI backends. References such as ``{{0.room.id}}``
    use preceding response JSON only, preserving types for whole JSON values.
    """
    from besser.spec_driven_agent.validation.constructibility import _fastapi_backends, _SKIP_DIRS
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env, timeout_cause
    from besser.spec_driven_agent.execution.sandbox import SandboxUnavailable, run_confined

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
                result = run_confined(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"],
                    workspace=str(root), writable=[str(scratch)], cwd=str(app_dir),
                    env=env, input=json.dumps(requests),
                    timeout=min(_PROBE_TIMEOUT_SECONDS, 60),
                )
            except SandboxUnavailable as exc:
                return _error(str(exc), "probe_error", backend=relative)
            except subprocess.TimeoutExpired as exc:
                own = f"timed out after {min(_PROBE_TIMEOUT_SECONDS, 60)}s"
                cause = timeout_cause(exc, own)
                return _error(f"API scenario {own}" if cause == own else f"API scenario stopped: {cause}",
                              "timeout", backend=relative)
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

    previous, responses, failures, secrets = [], [], [], set()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://besser-probe.invalid", follow_redirects=False) as client:
            for index, request in enumerate(requests):
                item = {"index": index, "method": request["method"], "path": request["path"], "failures": []}
                body = _NO_JSON
                notes: list[str] = []
                try:
                    path = _local_path(_resolve_references(request["path"], previous, in_path=True, notes=notes))
                    item["path"] = _resolve_references(request["path"], previous, in_path=True, redact=True)
                    kwargs = ({"json": _resolve_references(request["json"], previous, notes=notes)}
                              if "json" in request else {})
                    if request.get("headers"):
                        headers = {}
                        for name, raw in request["headers"].items():
                            value = _header_value(
                                _resolve_references(raw, previous, in_header=True, notes=notes), index, name)
                            headers[name] = value
                            if _sensitive_header(name):
                                # The whole value and its credential part, e.g.
                                # the token after "Bearer ", wherever echoed.
                                secrets.update(part for part in (value, value.split(" ", 1)[-1].strip())
                                               if len(part) >= 8)
                        kwargs["headers"] = headers
                        item["headers"] = sorted(headers)
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
                        # References read the raw ``body``; the report keeps a redacted copy, cut after redaction.
                        shown = _scrub(body, ())
                        item["json"] = _bounded(shown)
                        item["truncated"] = len(json.dumps(shown, ensure_ascii=True)) > _BODY_CHARS
                    except ValueError:
                        item["text"] = response.text[:_BODY_CHARS]
                        item["truncated"] = len(response.text) > _BODY_CHARS
                    expected = request.get("expected_status", list(range(200, 300)))
                    expected = expected if isinstance(expected, list) else [expected]
                    if response.status_code not in expected:
                        item["failures"].append(f"expected HTTP {request.get('expected_status', '2xx')}, got {response.status_code}")
                    for path, written in request.get("expected_fields", {}).items():
                        try:
                            actual = _json_path(body, path)
                            # Braces that name no earlier value are part of the text: the
                            # value as written is accepted too.
                            if _same_literal(actual, written):
                                continue
                            resolved_notes: list[str] = []
                            try:
                                resolved = _resolve_references(written, previous, notes=resolved_notes)
                                unresolved = False
                            except ValueError:
                                # Braces that resolve to nothing: the expectation is the text as written.
                                resolved, unresolved = written, True
                            notes.extend(resolved_notes)
                            if not _same_literal(actual, resolved):
                                # A message is a string: redact the response values before quoting
                                # them. An expected value written out in full is the test's own text.
                                names = _path_names(path)
                                expected_value = written if unresolved else _expected_shown(written, previous)
                                if expected_value != written:
                                    expected_value = _scrub(expected_value, (), names)
                                actual = _scrub(actual, (), names)
                                item["failures"].append(f"{path}: expected {_bounded(expected_value)!r}, got {_bounded(actual)!r}")
                        except ValueError as exc:
                            item["failures"].append(str(exc))
                except Exception as exc:
                    item["failures"].append(f"{type(exc).__name__}: {exc}"[:1000])
                if notes:
                    item["notes"] = list(dict.fromkeys(notes))
                # An error or failure of this request may quote a credential it sent back.
                sent = sorted(_sent_credentials(request, previous), key=len, reverse=True)
                item["failures"] = _scrub(item["failures"], sent)
                secrets.update(sent)  # a later body may echo it back; scrubbed with the headers' below
                if "error" in item:
                    item["error"]["message"] = _scrub(item["error"]["message"], sent)
                if request["method"] == "GET" and body is not _NO_JSON:
                    # Compared here, before redaction: a record whose only field is a secret is read back too.
                    item["reads_back"] = [earlier for earlier, raw in enumerate(previous)
                                          if requests[earlier]["method"] == "POST" and raw is not _NO_JSON
                                          and _reads_back(raw, body)]
                previous.append(body)
                count = len(item["failures"])
                item["failure_count"] = count
                item["failures"] = [message[:400] for message in item["failures"][:_MAX_REPORTED_FAILURES]]
                if count > _MAX_REPORTED_FAILURES:
                    item["omitted_failures"] = count - _MAX_REPORTED_FAILURES
                failures.extend({"index": index, "message": message} for message in item["failures"])
                responses.append(item)
    ordered = sorted(secrets, key=len, reverse=True)
    return {"status": "failed" if failures else "passed", "boot": "ok", "responses": _scrub(responses, ordered),
            "assertion_failures": _scrub(failures, ordered), "scope": _SCOPE}


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
