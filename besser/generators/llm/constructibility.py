"""Phase 3: can every entity actually be created through the generated API?

Live run 9a6063ed (2026-09-18): the gap analyser asked for "validation in
create_booking that the guests do not exceed the room capacities across all
BookedRooms", and Phase 2 wrote exactly that - at insert time. But a
BookedRoom needs a Booking id, so at insert time a Booking never has any:
capacity was always 0, every POST /booking/ was a 400, and with it Bill and
BookedRoom were unreachable. The app passed every static gate (imports,
mappers, star-import names, 61 routes, four entities creating fine) and
shipped as "0 blockers".

No static check sees this: the rule is ordinary code, the schema is valid,
and reachability of a ``raise`` depends on data flow. So this check runs the
app. In a subprocess, on a temp copy with a temp SQLite database, it boots
``main_api`` in-process (no server, no network), derives one create payload
per entity from the app's own OpenAPI schema plus the ORM's relationships,
creates entities in dependency order, and reports an entity only when EVERY
schema-valid request for it is refused. "Every" is the point - a 400 can
mean "this request was wrong"; the defect is that no request can be right.
So each entity is tried with its enum literals, its booleans flipped and its
dates reversed before it is reported, and a 422 (our payload, not the app's
rule) never counts. Calibrated on nine delivered apps: the broken one is the
single finding; eight healthy ones yield nothing, one of them only because
the date-reversed variant is tried.

Deliberately silent where another check owns the defect: a mapper that
fails to configure is ``mapper config:``; a model-level creation cycle is
``model contract:`` (the probe simply finds nothing it can create first).

A finding names its fix site - the function serving the create route, the
file and the line inside it that constructs the entity - because run
7f918e11 (2026-09-18) showed a 30B model given only the symptom spending
two fix attempts without touching ``routers/booking.py``. The line feeds the
fix prompt's excerpt. When the refusal is a NOT NULL violation, the column
is named too, and the create-time-rule advice (right for 9a6063ed, wrong
here) is left out.

Runs in two halves. The parent (``collect_constructibility_issues``) is
ordinary orchestrator code. The child is this same file executed by PATH as
a script inside the app copy, so it imports only the standard library and
what the app itself needs - never ``besser``, which is not installed where
the harness interpreter runs the generated code.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

PREFIX = "create contract:"

_PROBE_TIMEOUT_SECONDS = 90
_MARKER = "BESSER_CONSTRUCTIBILITY_REPORT:"
_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".besser_snapshot", "dist", "build", "data",
})
_BODY_CHARS = 160
_PAYLOAD_CHARS = 300
_NOT_NULL_PATTERNS = (
    re.compile(r"NOT NULL constraint failed: \w+\.(\w+)"),   # SQLite
    re.compile(r'null value in column "(\w+)"'),              # PostgreSQL
)


# ---------------------------------------------------------------------------
# Parent: locate backends, run the child, turn its report into findings
# ---------------------------------------------------------------------------

def collect_constructibility_issues(output_dir: str) -> list[str]:
    """``create contract:`` blockers, one per entity no request can create."""
    from besser.generators.llm.tool_executor import _safe_subprocess_env

    issues: list[str] = []
    for folder in _fastapi_backends(output_dir):
        rel = os.path.relpath(folder, output_dir).replace("\\", "/")
        report = _run_probe(folder, _safe_subprocess_env())
        issues.extend(_issues_from_report(report, rel))
    return issues


def _fastapi_backends(output_dir: str) -> list[str]:
    """Folders holding the generated FastAPI service (``main_api.py`` next to
    ``sql_alchemy.py``) - the scaffold family the probe knows how to drive."""
    found: list[str] = []
    try:
        for root, dirs, files in os.walk(output_dir):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".besser_"))
            if "main_api.py" in files and "sql_alchemy.py" in files:
                found.append(root)
    except OSError:
        pass
    return found


def _run_probe(folder: str, env: dict) -> dict:
    """Execute the child on a scratch copy of ``folder``; never raises."""
    work = tempfile.mkdtemp(prefix="besser_probe_")
    try:
        app_dir = os.path.join(work, "app")
        shutil.copytree(
            folder, app_dir,
            ignore=shutil.ignore_patterns(*_SKIP_DIRS, ".besser_*", "*.db"),
        )
        env = dict(env)
        env["DATABASE_URL"] = "sqlite:///" + os.path.join(work, "probe.db").replace("\\", "/")
        try:
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__)],
                capture_output=True, text=True, cwd=app_dir, env=env,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {"boot": "probe_error",
                    "error": f"timed out after {_PROBE_TIMEOUT_SECONDS}s"}
        except OSError as exc:
            return {"boot": "probe_error", "error": f"could not be launched: {exc}"}
        for line in reversed((result.stdout or "").splitlines()):
            if line.startswith(_MARKER):
                try:
                    return json.loads(line[len(_MARKER):])
                except ValueError:
                    break
        tail = [ln for ln in (result.stderr or "").splitlines() if ln.strip()]
        return {"boot": "probe_error",
                "error": (tail[-1] if tail else "no report")[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _not_checked(reason: str) -> str:
    from besser.generators.llm.orchestrator import _check_did_not_run
    return _check_did_not_run("the constructibility probe", reason)


def _issues_from_report(report: dict, rel: str) -> list[str]:
    boot = report.get("boot")
    if boot == "mapper_error":
        return []  # the import smoke check reports this one
    if boot != "ok":
        return [_not_checked(f"{rel}: {report.get('error', boot)}")]

    entities: dict = report.get("entities") or {}
    verdicts = {name: e.get("verdict") for name, e in entities.items()}
    created = [n for n, v in verdicts.items() if v == "created"]
    probed = [v for v in verdicts.values() if v != "unresolved"]
    if probed and not created and all(v == "unauthorized" for v in probed):
        return [_not_checked(f"{rel}: every create endpoint requires authentication")]
    failed = sorted(n for n, v in verdicts.items() if v in ("rejected", "crashed"))
    if not failed:
        return []
    if not created and all(verdicts[n] == "crashed" for n in failed):
        # Nothing at all works: the app does not serve in this environment
        # (unreachable database, startup hook) - not evidence about entities.
        first = entities[failed[0]]["attempts"][0]
        return [_not_checked(
            f"{rel}: no create endpoint answered - "
            f"{failed[0]} got {first.get('status')} {first.get('body', '')[:_BODY_CHARS]}"
        )]

    issues: list[str] = []
    for name in failed:
        entry = entities[name]
        attempt = next(
            (a for a in entry["attempts"] if a.get("outcome") == entry["verdict"]),
            entry["attempts"][0],
        )
        dependents = sorted(
            other for other, e in entities.items()
            if e.get("verdict") == "unresolved"
            and any(target == name for _, target in e.get("unresolved", []))
        )
        body = (attempt.get("body") or "")[:_BODY_CHARS]
        payload = json.dumps(attempt.get("payload", {}), sort_keys=True)[:_PAYLOAD_CHARS]
        if entry["verdict"] == "rejected":
            how = f"every schema-valid request is rejected ({attempt.get('status')}: {body})"
        else:
            how = f"every schema-valid request fails with a server error ({attempt.get('status')}: {body})"
        text = (
            f"{PREFIX} {rel}: POST {entry['path']} cannot create a {name} - {how}; "
            f"a valid payload that was refused: {payload}"
        )
        site = entry.get("site")
        column = attempt.get("missing_column")
        if site:
            # " in <file> line <n>" is what the fix prompt's excerpt keys on.
            text += f". Fix site: {site['function']} in {rel}/{site['file']} line {site['line']}"
            if site.get("also"):
                text += " (also line " + ", ".join(str(n) for n in site["also"]) + ")"
        if column:
            text += (
                f" - the row is inserted without `{column}`, which the table declares "
                f"NOT NULL: set `{column}` there before the insert, or make the column "
                f"nullable / give it a default"
            )
        if dependents:
            names = " or ".join(dependents)
            text += f". {', '.join(dependents)} require a {name} id, so they cannot be created either"
            if not column:
                text += (
                    f". No {names} row can exist before the {name} it "
                    f"requires, so a create-time rule that needs them can never pass - "
                    f"enforce it where those rows are created, updated or deleted (or "
                    f"when the {name} is updated), or create them inline in the same request"
                )
        issues.append(text)
    return issues


def _missing_not_null_column(body: str) -> str | None:
    for pattern in _NOT_NULL_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Child: runs inside the app copy (cwd), prints one report line
# ---------------------------------------------------------------------------

def _resolve(schema: dict, schemas: dict) -> dict:
    if "$ref" in schema:
        return schemas.get(schema["$ref"].rsplit("/", 1)[-1], {})
    for alt in schema.get("anyOf", []):
        if alt.get("type") != "null":
            return _resolve(alt, schemas)
    return schema


def _sample(field: str, schema: dict, day_offset: int):
    low = field.lower()
    if "enum" in schema:
        return schema["enum"][0]
    kind, fmt = schema.get("type"), schema.get("format")
    if kind == "string":
        if fmt == "date":
            return (_dt.date.today() + _dt.timedelta(days=day_offset)).isoformat()
        if fmt == "date-time":
            return (_dt.datetime.now() + _dt.timedelta(days=day_offset)).isoformat()
        if fmt == "time":
            return "12:00:00"
        if "email" in low:
            return "probe@example.com"
        if "phone" in low:
            return "+12345678901"
        if "url" in low or "link" in low:
            return "https://example.com"
        return f"probe-{field}"
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return None


def _relationship_for(field: str, relationships: dict):
    """The ORM relationship a create-schema field stands for, if any."""
    if field in relationships:
        return relationships[field]
    for suffix in ("_id", "Id", "_ids", "Ids"):
        if field.endswith(suffix) and field[: -len(suffix)] in relationships:
            return relationships[field[: -len(suffix)]]
    return None


def _build_payload(schema: dict, schemas: dict, relationships: dict, ids: dict,
                   subclasses: dict, variant: tuple) -> tuple[dict, list]:
    """Minimal payload: required fields only, relationships from created ids."""
    props = schema.get("properties", {})
    required = [f for f in props if f in set(schema.get("required", []))]
    date_fields = [f for f in required if _resolve(props[f], schemas).get("format") == "date"]
    payload: dict = {}
    unresolved: list = []
    for field in required:
        rel = _relationship_for(field, relationships)
        if rel is not None:
            target, uselist = rel
            chosen = ids.get(target)
            if chosen is None:
                # A subclass row is a row of its parent (joined inheritance).
                chosen = next((ids[s] for s in subclasses.get(target, ()) if s in ids), None)
            if chosen is None:
                unresolved.append([field, target])
                continue
            payload[field] = [chosen] if uselist else chosen
            continue
        resolved = _resolve(props[field], schemas)
        if variant[0] in ("enum", "bool") and variant[1] == field:
            payload[field] = variant[2]
            continue
        offset = 1
        if field in date_fields:
            index = date_fields.index(field)
            offset = 1 + (len(date_fields) - 1 - index if variant[0] == "dates" else index)
        payload[field] = _sample(field, resolved, offset)
    return payload, unresolved


def _variants(schema: dict, schemas: dict):
    yield ("base",)
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        resolved = _resolve(props.get(field, {}), schemas)
        for literal in resolved.get("enum", [])[1:]:
            yield ("enum", field, literal)
        if resolved.get("type") == "boolean":
            yield ("bool", field, False)
    yield ("dates",)


def _extract_id(body):
    if isinstance(body, dict):
        if isinstance(body.get("id"), (int, str)):
            return body["id"]
        for value in body.values():
            if isinstance(value, dict) and isinstance(value.get("id"), (int, str)):
                return value["id"]
    return None


def _outcome(status) -> str:
    if not isinstance(status, int):
        return "crashed"
    if 200 <= status < 300:
        return "created"
    if status in (401, 403):
        return "unauthorized"
    if status == 422:
        return "invalid"      # our payload, not the app's rule
    if status >= 500:
        return "crashed"
    return "rejected"


_VERDICT_ORDER = ("created", "rejected", "crashed", "unauthorized", "invalid")


def _handler_site(app, path: str, entity: str) -> dict | None:
    """The function serving ``POST path``, and inside it the line that
    constructs ``entity`` - the place the fix goes. ``also`` lists the other
    lines of that file constructing the entity (a bulk route, typically),
    which need the same change."""
    import inspect
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path or "POST" not in (getattr(route, "methods", None) or ()):
            continue
        fn = inspect.unwrap(route.endpoint)
        try:
            file = inspect.getsourcefile(fn)
            lines, start = inspect.getsourcelines(fn)
        except (OSError, TypeError):
            return None
        if not file:
            return None
        construct = re.compile(r"\b" + re.escape(entity) + r"\(")
        offset = next((i for i, line in enumerate(lines) if construct.search(line)), 0)
        also: list[int] = []
        try:
            with open(file, encoding="utf-8") as fh:
                for number, line in enumerate(fh, 1):
                    if construct.search(line) and not start <= number < start + len(lines):
                        also.append(number)
        except OSError:
            pass
        return {
            "file": os.path.relpath(file, os.getcwd()).replace("\\", "/"),
            "function": fn.__name__,
            "line": start + offset,
            "also": also,
        }
    return None


def _probe_cwd() -> dict:
    sys.path.insert(0, os.getcwd())
    try:
        import main_api
    except ModuleNotFoundError as exc:
        return {"boot": "missing_dependency", "error": f"{type(exc).__name__}: {exc}"}
    except BaseException as exc:
        return {"boot": "import_error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        import sql_alchemy
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy.orm import configure_mappers
        configure_mappers()
    except ModuleNotFoundError as exc:
        return {"boot": "missing_dependency", "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"boot": "mapper_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    try:
        import httpx
    except ModuleNotFoundError as exc:
        return {"boot": "missing_dependency", "error": f"{type(exc).__name__}: {exc}"}

    app = getattr(main_api, "app", None)
    if not hasattr(app, "openapi"):
        app = next((v for v in vars(main_api).values()
                    if hasattr(v, "openapi") and hasattr(v, "routes")), None)
    if app is None:
        return {"boot": "no_app", "error": "main_api defines no FastAPI app"}
    try:
        spec = app.openapi()
    except Exception as exc:
        return {"boot": "probe_error", "error": f"openapi(): {type(exc).__name__}: {exc}"[:300]}
    schemas = spec.get("components", {}).get("schemas", {})

    orm: dict = {}
    for value in vars(sql_alchemy).values():
        if isinstance(value, type) and hasattr(value, "__mapper__"):
            try:
                orm[value.__name__] = {
                    r.key: (r.mapper.class_.__name__, bool(r.uselist))
                    for r in sa_inspect(value).relationships
                }
            except Exception:
                orm[value.__name__] = {}
    subclasses: dict = {}
    for name in orm:
        for base in getattr(sql_alchemy, name).__mro__[1:]:
            if base.__name__ in orm:
                subclasses.setdefault(base.__name__, []).append(name)

    creates: dict = {}
    for path, ops in spec.get("paths", {}).items():
        body = (ops.get("post") or {}).get("requestBody", {})
        ref = body.get("content", {}).get("application/json", {}).get("schema", {}).get("$ref", "")
        if ref.endswith("Create") and ref.rsplit("/", 1)[-1] in schemas:
            schema_name = ref.rsplit("/", 1)[-1]
            creates[schema_name[: -len("Create")]] = (path, schema_name)

    import asyncio

    async def run() -> dict:
        ids: dict = {}
        entities: dict = {}
        pending = dict(creates)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
            progress = True
            while pending and progress:
                progress = False
                for entity, (path, schema_name) in list(pending.items()):
                    if entity not in orm:
                        continue  # cannot tell attributes from foreign keys: unprovable
                    schema = schemas[schema_name]
                    _, unresolved = _build_payload(
                        schema, schemas, orm[entity], ids, subclasses, ("base",))
                    if unresolved:
                        continue
                    attempts: list = []
                    for variant in _variants(schema, schemas):
                        payload, _ = _build_payload(
                            schema, schemas, orm[entity], ids, subclasses, variant)
                        try:
                            response = await client.post(path, json=payload)
                            status, text = response.status_code, response.text or ""
                        except Exception as exc:
                            response, status = None, "EXC"
                            text = f"{type(exc).__name__}: {exc}"
                        outcome = _outcome(status)
                        attempts.append({
                            "variant": " ".join(str(v) for v in variant),
                            "status": status, "outcome": outcome,
                            "body": text[:_BODY_CHARS], "payload": payload,
                        })
                        column = _missing_not_null_column(text)
                        if column:
                            attempts[-1]["missing_column"] = column
                        if outcome == "created":
                            try:
                                ids[entity] = _extract_id(response.json())
                            except Exception:
                                ids[entity] = None
                            break
                    verdict = next(
                        (v for v in _VERDICT_ORDER if any(a["outcome"] == v for a in attempts)),
                        "invalid",
                    )
                    entities[entity] = {"path": path, "verdict": verdict, "attempts": attempts}
                    if verdict in ("rejected", "crashed"):
                        entities[entity]["site"] = _handler_site(app, path, entity)
                    del pending[entity]
                    progress = True
            for entity, (path, schema_name) in pending.items():
                _, unresolved = _build_payload(
                    schemas[schema_name], schemas, orm.get(entity, {}), ids, subclasses, ("base",))
                entities[entity] = {"path": path, "verdict": "unresolved", "unresolved": unresolved}
        return entities

    try:
        entities = asyncio.run(run())
    except Exception as exc:
        return {"boot": "probe_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"boot": "ok", "entities": entities}


if __name__ == "__main__":
    print(_MARKER + json.dumps(_probe_cwd(), default=str))
