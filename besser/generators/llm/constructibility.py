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
creates entities in dependency order, and reports what those requests actually
establish. Samples are NOT exhaustive: a 400/409/422 can be a correct refusal
of our guessed input. Such routes remain explicitly unverified, with a path
to verification through a valid API scenario; they are not declared broken.
Observed server failures remain concrete findings. Samples use distinct
identity/contact values across entities and retries, including subclasses
sharing one parent table, so the probe does not manufacture uniqueness errors.

Deliberately silent where another check owns the defect: a mapper that
fails to configure is ``mapper config:``. Unresolved creation dependencies
are reported as unverified; they are not proof that no valid workflow exists.

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
import itertools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

PREFIX = "create contract:"
UNVERIFIED_PREFIX = "create unverified:"

_PROBE_TIMEOUT_SECONDS = 90
_MARKER = "BESSER_CONSTRUCTIBILITY_REPORT:"
_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".besser_snapshot", "dist", "build", "data",
})
_BODY_CHARS = 160
_PAYLOAD_CHARS = 300
_MAX_VARIANTS = 20
_NOT_NULL_PATTERNS = (
    re.compile(r"NOT NULL constraint failed: \w+\.(\w+)"),   # SQLite
    re.compile(r'null value in column "(\w+)"'),              # PostgreSQL
)


# ---------------------------------------------------------------------------
# Parent: locate backends, run the child, turn its report into findings
# ---------------------------------------------------------------------------

def collect_constructibility_issues(output_dir: str) -> list[str]:
    """Observed server defects and explicit gaps in create-route verification."""
    from besser.generators.llm.execution.process import _safe_subprocess_env

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
    from besser.generators.llm.validation.issues import _check_did_not_run
    return _check_did_not_run("the constructibility probe", reason)


def _issues_from_report(report: dict, rel: str) -> list[str]:
    boot = report.get("boot")
    if boot in {"import_error", "no_app"}:
        site = report.get("site") or {}
        location = f"{rel}/{site['file']} line {site['line']}" if site.get("file") else rel
        return [f"application startup: {location}: {report.get('error', boot)}"]
    if boot == "mapper_error":
        return []  # the import smoke check reports this one
    if boot != "ok":
        return [_not_checked(f"{rel}: {report.get('error', boot)}")]

    entities: dict = report.get("entities") or {}
    issues: list[str] = []
    for name, entry in sorted(entities.items()):
        attempts = entry.get("attempts") or []
        # A later success does not erase an observed unhandled exception. Nor
        # may a 400 hide a 500 merely because it came first in a verdict list.
        failure = next((a for a in attempts if a.get("outcome") == "crashed"), None)
        if failure is None and entry.get("verdict") != "created":
            failure = next((a for a in attempts if a.get("missing_column")), None)
        if entry.get("verdict") == "created" and failure is None:
            continue
        attempt = failure or next((a for a in attempts if a.get("outcome") == "rejected"),
                                  attempts[0] if attempts else {})
        dependents = sorted(
            other for other, e in entities.items()
            if e.get("verdict") == "unresolved"
            and any(target == name for _, target in e.get("unresolved", []))
        )
        body = (attempt.get("body") or "")[:_BODY_CHARS]
        payload = json.dumps(attempt.get("payload", {}), sort_keys=True)[:_PAYLOAD_CHARS]
        if failure:
            text = (
                f"{PREFIX} {rel}: POST {entry['path']} - observed a server/persistence "
                f"failure creating {name} ({attempt.get('status')}: {body}); "
                f"attempted input: {payload}. This is one observed failure, not proof "
                "that every valid request fails. Preserve business constraints; "
                "handle invalid input without an unhandled server failure"
            )
        else:
            if attempt:
                reason = (f"autogenerated input was refused ({attempt.get('status')}: {body}); "
                          f"attempted input: {payload}")
                if attempt.get("outcome") == "unauthorized":
                    reason += "; this endpoint requires authentication"
            else:
                unresolved = entry.get("unresolved") or []
                reason = ("no request could be constructed; unresolved related IDs: "
                          + json.dumps(unresolved, sort_keys=True))
            text = (
                f"{UNVERIFIED_PREFIX} {rel}: POST {entry['path']} - {reason}. "
                "These guessed inputs do not prove this endpoint is broken. Use test_api "
                "with valid specification-based fixtures and read the persisted result; "
                "do not remove business validation to satisfy guessed samples"
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
                f"NOT NULL: populate/derive `{column}` before inserting, or reject "
                "invalid input explicitly; do not remove a required constraint to "
                "accept a probe fixture"
            )
        if dependents:
            text += (f". Dependent create probes for {', '.join(dependents)} could not "
                     f"run without a {name} id")
        if not failure and entry.get("deferred_relationships") and attempt.get("status") == 400:
            text += (
                ". Creation-order check: " + ", ".join(entry["deferred_relationships"])
                + f" accept existing child IDs, but those children require a {name} first. "
                "If the refusing rule needs those children, provide an atomic valid "
                "aggregate-creation workflow (such as nested creation), retaining the rule"
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

def _resolve(schema: dict, schemas: dict, _depth: int = 0) -> dict:
    if _depth >= 8:
        return {}
    if "$ref" in schema:
        return _resolve(schemas.get(schema["$ref"].rsplit("/", 1)[-1], {}), schemas, _depth + 1)
    for keyword in ("anyOf", "oneOf"):
        for alt in schema.get(keyword, []):
            if alt.get("type") != "null":
                return _resolve(alt, schemas, _depth + 1)
    if "allOf" in schema:
        merged = {k: v for k, v in schema.items() if k != "allOf"}
        for part in schema["allOf"]:
            part = _resolve(part, schemas, _depth + 1)
            props = {**merged.get("properties", {}), **part.get("properties", {})}
            required = list(dict.fromkeys([*merged.get("required", []), *part.get("required", [])]))
            merged.update(part)
            merged.update(properties=props, required=required)
        return merged
    return schema


def _association_link_schema(schema: dict, schemas: dict, _depth: int = 0) -> dict | None:
    """Find the scaffold's native association object, even in int|object unions."""
    if _depth >= 8:
        return None
    if "$ref" in schema:
        return _association_link_schema(
            schemas.get(schema["$ref"].rsplit("/", 1)[-1], {}), schemas, _depth + 1)
    for keyword in ("anyOf", "oneOf"):
        for alt in schema.get(keyword, []):
            found = _association_link_schema(alt, schemas, _depth + 1)
            if found is not None:
                return found
    resolved = _resolve(schema, schemas)
    if "target" in resolved.get("properties", {}):
        return resolved
    return None


def _sample(field: str, schema: dict, day_offset: int, *,
            entity: str = "entity", ordinal: int = 1, unique: bool = False):
    """Best-effort fixture, never a claim of business-valid input.

    The ordinal is global to this probe, not local to an entity: subclasses
    often share their parent's unique columns. Retries need fresh values too,
    since an unsuccessful endpoint may already have persisted a partial row.
    """
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
        if "email" in low or fmt == "email":
            return f"probe-{ordinal}-{entity.lower()}@example.com"
        if "phone" in low:
            return f"+1{ordinal:010d}"
        if fmt == "uuid":
            import uuid
            return str(uuid.UUID(int=ordinal))
        if "url" in low or "link" in low:
            return f"https://example.com/probe/{ordinal}"
        value = f"p{ordinal}-{entity.lower()}-{field}"
        if isinstance(schema.get("minLength"), int):
            value = value.ljust(schema["minLength"], "x")
        if isinstance(schema.get("maxLength"), int):
            value = value[:schema["maxLength"]]
        return value
    if kind in {"integer", "number"}:
        step = schema.get("multipleOf", 1)
        if not isinstance(step, (int, float)) or step <= 0:
            step = 1
        lower = schema.get("minimum", 1)
        if isinstance(schema.get("exclusiveMinimum"), (int, float)):
            lower = max(lower, schema["exclusiveMinimum"] + step)
        value = math.ceil(lower / step) * step
        if unique or low.endswith(("number", "code")) or low in {"id", "identifier"}:
            value += (ordinal - 1) * step
        upper = schema.get("maximum")
        if isinstance(schema.get("exclusiveMaximum"), (int, float)):
            exclusive = schema["exclusiveMaximum"] - step
            upper = min(upper, exclusive) if upper is not None else exclusive
        if upper is not None and value > upper:
            value = math.floor(upper / step) * step
        return int(value) if kind == "integer" else float(value)
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


def _deferred_relationships(entity: str, schema: dict, schemas: dict, orm: dict) -> list[str]:
    """Describe reciprocal ID dependencies, without calling them impossible.

    A rejected aggregate may need its child rows before those rows can obtain
    the aggregate's ID. This is a useful investigation hint, not a reason to
    remove a capacity/multiplicity/business rule.
    """
    found = []
    for field in schema.get("properties", {}):
        rel = _relationship_for(field, orm.get(entity, {}))
        if rel is None:
            continue
        target, _ = rel
        child = schemas.get(target + "Create", {})
        for required in child.get("required", []):
            back = _relationship_for(required, orm.get(target, {}))
            if back is not None and back[0] == entity:
                found.append(f"{field} -> {target}.{required}")
                break
    return found[:10]


def _leaf_reference(entity: str, schema: dict, schemas: dict, orm: dict,
                    creates: dict, ids: dict) -> str | None:
    """One already-constructible leaf whose fresh ID can avoid a reused link.

    No recursive fixture search: nested/aggregate prerequisites stay unverified
    when we do not know a valid replacement workflow.
    """
    for field in schema.get("required", []):
        rel = _relationship_for(field, orm.get(entity, {}))
        if rel is None or rel[1] or rel[0] not in creates or ids.get(rel[0]) is None:
            continue
        target = rel[0]
        leaf_schema = schemas[creates[target][1]]
        if not any(_relationship_for(f, orm.get(target, {})) is not None
                   for f in leaf_schema.get("required", [])):
            return target
    return None


def _build_payload(schema: dict, schemas: dict, relationships: dict, ids: dict,
                   subclasses: dict, variant: tuple, *, entity: str = "entity",
                   ordinal: int = 1, unique_fields: frozenset = frozenset()) -> tuple[dict, list]:
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
            # Native association classes carry attributes on each link. A raw
            # ID may be schema-accepted for legacy compatibility but omit the
            # required price/quantity/etc. Prefer the advertised object form.
            relation_schema = _resolve(props[field], schemas)
            item_schema = relation_schema.get("items", {}) if uselist else props[field]
            link_schema = _association_link_schema(item_schema, schemas)
            linked = chosen
            if link_schema is not None:
                linked = {"target": chosen}
                for attribute in link_schema.get("required", []):
                    if attribute != "target":
                        linked[attribute] = _sample(
                            attribute, _resolve(link_schema.get("properties", {}).get(attribute, {}), schemas),
                            1, entity=entity, ordinal=ordinal)
            payload[field] = [linked] if uselist else linked
            continue
        resolved = _resolve(props[field], schemas)
        if variant[0] in ("enum", "bool") and variant[1] == field:
            payload[field] = variant[2]
            continue
        offset = 1
        if field in date_fields:
            index = date_fields.index(field)
            offset = 1 + (len(date_fields) - 1 - index if variant[0] == "dates" else index)
        payload[field] = _sample(field, resolved, offset, entity=entity,
                                 ordinal=ordinal, unique=field in unique_fields)
    return payload, unresolved


def _variants(schema: dict, schemas: dict):
    yield ("base",)
    props = schema.get("properties", {})
    if sum(_resolve(props.get(f, {}), schemas).get("format") == "date"
           for f in schema.get("required", [])) > 1:
        yield ("dates",)
    for field in schema.get("required", []):
        resolved = _resolve(props.get(field, {}), schemas)
        for literal in resolved.get("enum", [])[1:]:
            yield ("enum", field, literal)
        if resolved.get("type") == "boolean":
            yield ("bool", field, False)


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


_VERDICT_ORDER = ("created", "crashed", "rejected", "unauthorized", "invalid")


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
        import traceback
        frames = []
        for frame in traceback.extract_tb(exc.__traceback__):
            try:
                if os.path.commonpath([os.path.abspath(frame.filename), os.getcwd()]) == os.getcwd():
                    frames.append(frame)
            except ValueError:
                continue  # dependency/interpreter frames may live on another drive
        site = ({"file": os.path.relpath(frames[-1].filename, os.getcwd()).replace("\\", "/"),
                 "line": frames[-1].lineno} if frames else {})
        return {"boot": "import_error", "error": f"{type(exc).__name__}: {exc}"[:1200], "site": site}
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
    unique_fields: dict = {}
    composite_links: set = set()
    for value in vars(sql_alchemy).values():
        if isinstance(value, type) and hasattr(value, "__mapper__"):
            try:
                mapper = sa_inspect(value)
                orm[value.__name__] = {
                    r.key: (r.mapper.class_.__name__, bool(r.uselist))
                    for r in mapper.relationships
                }
                unique_columns = {
                    col for table in mapper.tables
                    for constraint in (*table.constraints, *table.indexes)
                    if constraint.__class__.__name__ in {"UniqueConstraint", "PrimaryKeyConstraint"}
                    or getattr(constraint, "unique", False)
                    for col in constraint.columns
                }
                unique_fields[value.__name__] = frozenset(
                    prop.key for prop in mapper.column_attrs
                    if any(col.unique or col.primary_key or col in unique_columns
                           for col in prop.columns)
                )
                if any(
                    sum(bool(getattr(col, "foreign_keys", ())) for col in constraint.columns) >= 2
                    for table in mapper.tables for constraint in (*table.constraints, *table.indexes)
                    if constraint.__class__.__name__ in {"UniqueConstraint", "PrimaryKeyConstraint"}
                    or getattr(constraint, "unique", False)
                ):
                    composite_links.add(value.__name__)
            except Exception:
                orm.setdefault(value.__name__, {})
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
        ordinal = 0
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
            async def request(path, payload, variant):
                try:
                    response = await client.post(path, json=payload)
                    status, text = response.status_code, response.text or ""
                except Exception as exc:
                    response, status = None, "EXC"
                    text = f"{type(exc).__name__}: {exc}"
                attempt = {
                    "variant": " ".join(str(v) for v in variant),
                    "status": status, "outcome": _outcome(status),
                    "body": text[:_BODY_CHARS], "payload": payload,
                }
                column = _missing_not_null_column(text)
                if column:
                    attempt["missing_column"] = column
                if status == 409 and ("UNIQUE constraint failed" in text
                                      or "duplicate key value violates unique constraint" in text):
                    attempt["unique_collision"] = True
                return response, attempt

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
                    leaf_retry_used = False
                    for variant in itertools.islice(_variants(schema, schemas), _MAX_VARIANTS):
                        ordinal += 1
                        payload, _ = _build_payload(
                            schema, schemas, orm[entity], ids, subclasses, variant,
                            entity=entity, ordinal=ordinal,
                            unique_fields=unique_fields.get(entity, frozenset()))
                        response, attempt = await request(path, payload, variant)
                        attempts.append(attempt)
                        if (attempt.get("unique_collision") and entity in composite_links
                                and not leaf_retry_used):
                            leaf_retry_used = True
                            target = _leaf_reference(entity, schema, schemas, orm, creates, ids)
                            if target is not None:
                                leaf_path, leaf_schema_name = creates[target]
                                ordinal += 1
                                leaf_payload, _ = _build_payload(
                                    schemas[leaf_schema_name], schemas, orm[target], ids, subclasses,
                                    ("base",), entity=target, ordinal=ordinal,
                                    unique_fields=unique_fields.get(target, frozenset()))
                                leaf_response, leaf_attempt = await request(
                                    leaf_path, leaf_payload, ("fresh reference for", entity))
                                # Keep all observed server failures visible, including
                                # failures in the counterexample fixture itself.
                                entities[target]["attempts"].append(leaf_attempt)
                                try:
                                    fresh_id = (_extract_id(leaf_response.json())
                                                if leaf_attempt["outcome"] == "created" else None)
                                except Exception:
                                    fresh_id = None
                                if fresh_id is not None:
                                    ordinal += 1
                                    payload, _ = _build_payload(
                                        schema, schemas, orm[entity], {**ids, target: fresh_id}, subclasses,
                                        variant, entity=entity, ordinal=ordinal,
                                        unique_fields=unique_fields.get(entity, frozenset()))
                                    response, attempt = await request(path, payload, ("fresh reference", target))
                                    attempts.append(attempt)
                        if attempt["outcome"] == "created":
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
                    entities[entity]["deferred_relationships"] = _deferred_relationships(
                        entity, schema, schemas, orm)
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
