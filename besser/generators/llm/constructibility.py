"""Phase 3: does the delivered app actually do the job, when driven?

The harness verifies the application rather than asking the model to.
Measured across ~150 Qwen runs, the agent called ``test_api`` 0.06 times per
run against gpt-5.6's 13.1, left 14-21 of ~20 checklist items open, and
reported ``completed`` once in 135 - on a median 66 turns of 120, so not a
budget limit. Rewriting the checklist to name ``test_api`` literally produced
0.00 calls across 24 runs. Three attempts at persuasion failed, so this
module drives the workflow: create each aggregate, call each modelled action
on it with the literal ``{}`` the generated button posts, and read the record
back.

A status code is not a result. A live gpt-5.6-terra run answered its Renew
button ``200 {"success": false, "message": "a dueDate is required"}`` and the
acceptance oracle scored it 10/10, because nothing read past the status line.
The probe snapshots the entity and the size of every collection either side
of the call; an action that declares a refusal and changes nothing did not
run. Refusing is often correct, though - cancelling a checked-out booking
SHOULD fail - so the hard finding is entity-scoped and needs the aggregate to
have been created one request earlier, no modelled action to have moved it in
any lifecycle state its own create schema can select, and the model to state
that the action takes no parameters. Anything short of that is
``action unverified:``.

The first question it asked is still the first one it asks:
can every entity actually be created through the generated API?

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

Runs in two halves. The parent (``collect_constructibility_report``) is
ordinary orchestrator code; it also hands the child the modelled action list
in a JSON file, since the model is what states an action's parameters. The
child is this same file executed by PATH as a script inside the app copy, so
it imports only the standard library and what the app itself needs - never
``besser``, which is not installed where the harness interpreter runs the
generated code. The child's structured record is returned alongside the
rendered findings, so a caller that wants runtime facts reads them rather
than matching prefixes back out of prose.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
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
ACTION_PREFIX = "action call:"
ACTION_UNVERIFIED_PREFIX = "action unverified:"

_PROBE_TIMEOUT_SECONDS = 90
_MARKER = "BESSER_CONSTRUCTIBILITY_REPORT:"
_ACTIONS_ENV = "BESSER_PROBE_MODEL_ACTIONS"
# Entities the MODEL marks abstract. A generated router is right to refuse to
# instantiate one ("Person is abstract; create a Patron or Librarian
# instead"), so POSTing to it manufactures a blocker no repair can clear:
# gpt-5.6-terra-hzllh0l6 spent 17 rounds, 108 turns and $0.96 on exactly that.
_ABSTRACT_ENV = "BESSER_PROBE_ABSTRACT_ENTITIES"
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

def collect_constructibility_report(output_dir: str, domain_model=None) -> dict:
    """``{"issues": [...], "backends": [{"backend": rel, ...}]}``.

    The probe already computed per-entity and per-action runtime facts; the
    caller used to see only the rendered strings and had to re-derive the
    facts by matching prefixes back out of them. ``backends`` carries the
    child's own report so ``_phase3_tree_score`` can rank a tree on what the
    running app actually did.
    """
    from besser.generators.llm.execution.process import _safe_subprocess_env

    model_actions = _model_actions(domain_model)
    abstract = _abstract_entities(domain_model)
    issues: list[str] = []
    backends: list[dict] = []
    for folder in _fastapi_backends(output_dir):
        rel = os.path.relpath(folder, output_dir).replace("\\", "/")
        report = _run_probe(folder, _safe_subprocess_env(), model_actions, abstract)
        issues.extend(_issues_from_report(report, rel))
        backends.append({"backend": rel, **report})
    return {"issues": issues, "backends": backends}


def collect_constructibility_issues(output_dir: str, domain_model=None) -> list[str]:
    """Observed server defects and explicit gaps in create-route verification."""
    return collect_constructibility_report(output_dir, domain_model)["issues"]


def _abstract_entities(domain_model) -> list[str]:
    """Class names the model marks ``is_abstract``.

    Read defensively: an older serializer, or a model with no such notion,
    yields an empty list and the probe behaves exactly as before.
    """
    if domain_model is None:
        return []
    try:
        return sorted(
            cls.name for cls in domain_model.get_classes()
            if getattr(cls, "is_abstract", False)
        )
    except Exception:
        logger.debug("constructibility probe: abstract classes unreadable", exc_info=True)
        return []


def _model_actions(domain_model) -> dict:
    """``entity -> [{"name", "parameters", "return_type"}]`` from the MODEL.

    The parameter list and the return type only became readable today:
    ``_method_entry`` omitted ``parameters`` for a zero-argument method
    (a886947f) and the editor's newer format dropped the return type
    (ab03f9d7). Both keys are read defensively - an absent ``parameters``
    means "this serializer does not say", which disables every finding that
    leans on the zero-argument contract rather than asserting it.
    """
    if domain_model is None:
        return {}
    try:
        from besser.generators.llm.model_serializer import (
            _collect_inherited_methods, _method_entry,
        )
        classes = list(domain_model.get_classes())
    except Exception:
        logger.debug("constructibility probe: no modelled actions", exc_info=True)
        return {}
    out: dict = {}
    for cls in classes:
        pairs = [(m, None) for m in getattr(cls, "methods", None) or ()]
        try:
            pairs += _collect_inherited_methods(cls)
        except Exception:
            pass
        entries = []
        for method, _owner in pairs:
            try:
                entry = _method_entry(method)
            except Exception:
                continue
            name = str(entry.get("name") or "").split("(")[0].strip()
            if not name:
                continue
            entries.append({
                "name": name,
                # None (key absent) is "unstated", which is not "takes none".
                "parameters": entry.get("parameters"),
                "return_type": entry.get("return_type"),
            })
        if entries:
            out[cls.name] = sorted(entries, key=lambda e: e["name"])
    return out


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


def _run_probe(folder: str, env: dict, model_actions: dict | None = None,
               abstract_entities: list[str] | None = None) -> dict:
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
        if model_actions:
            actions_path = os.path.join(work, "model_actions.json")
            try:
                with open(actions_path, "w", encoding="utf-8") as fh:
                    json.dump(model_actions, fh)
                env[_ACTIONS_ENV] = actions_path
            except (OSError, TypeError, ValueError):
                logger.debug("constructibility probe: modelled actions not passed", exc_info=True)
        if abstract_entities:
            env[_ABSTRACT_ENV] = ",".join(abstract_entities)
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
    issues.extend(_action_issue_text(entry, rel) for entry in report.get("actions") or [])
    return issues


def _action_issue_text(entry: dict, rel: str) -> str:
    """Render one action-endpoint finding (see ``_probe_actions``)."""
    site = entry.get("site")
    if entry["verdict"] in ("inert", "inert unverified"):
        text = _inert_action_text(entry, rel)
    elif entry["verdict"] == "crashed":
        attempt = entry["attempt"]
        body = (attempt.get("body") or "")[:_BODY_CHARS]
        text = (
            f"{ACTION_PREFIX} {rel}: POST {entry['route']} - observed a server/persistence "
            f"failure calling {entry['action']} on a {entry['entity']} in state "
            f"{entry['state']} ({attempt.get('status')}: {body}). This is one observed "
            "failure, not proof that every call fails. Handle invalid/unexpected state "
            "without an unhandled server failure"
        )
    else:
        states = ", ".join(entry.get("states", []))
        statuses = ", ".join(str(s) for s in entry.get("statuses", []))
        text = (
            f"{ACTION_UNVERIFIED_PREFIX} {rel}: POST {entry['route']} - {entry['action']} on "
            f"{entry['entity']} refused every state this probe could construct via the "
            f"entity's own create fields ({states}; statuses {statuses}). A correctly "
            "implemented state-guarded action may legitimately refuse some states; this "
            "probe found none, among the states it can select, where the call succeeded. "
            "This can mean a defect in the guard/state comparison (e.g. comparing an enum "
            "member to its raw string value), or a precondition this probe cannot reach "
            "through this entity's own create fields. Not proof the action is broken; "
            "inspect the handler directly or exercise it with test_api once the reachable "
            "state is known"
        )
    if site:
        text += f". Fix site: {site['function']} in {rel}/{site['file']} line {site['line']}"
    return text


def _inert_action_text(entry: dict, rel: str) -> str:
    """Render a 2xx that declared failure and left the world unchanged.

    The status code is not the result. A live gpt-5.6-terra run answered the
    Renew button ``200 {"success": false, "message": "a dueDate is required"}``
    and the acceptance oracle scored it 10/10, because nothing read past the
    status line. So the probe reads the record back: a call that reports a
    refusal and changes no row did not happen, whatever it returned.

    Refusing is often right, which is why this is entity-scoped: it fires only
    when NO modelled action moved a freshly created instance, i.e. the
    aggregate has no first transition at all. "Cancel a checked-out booking"
    legitimately fails, but a booking created one request ago is not
    checked out.
    """
    call = entry["call"]
    body = (call.get("body") or "")[:_BODY_CHARS]
    prefix = ACTION_PREFIX if entry["verdict"] == "inert" else ACTION_UNVERIFIED_PREFIX
    text = (
        f"{prefix} {rel}: POST {entry['route']} - {entry['action']} answered HTTP "
        f"{call.get('status')} but declared failure ({call.get('marker')}) and changed "
        f"nothing: the {entry['entity']} created one request earlier, still in its initial "
        f"state, reads back identical after the call and no other record appeared "
        f"({body}). None of the modelled actions on {entry['entity']} "
        f"({', '.join(entry.get('siblings') or [entry['action']])}) moved a freshly created "
        "instance, so this aggregate has no first transition and the workflow cannot start"
    )
    if entry.get("zero_param"):
        text += (
            f". The model declares {entry['action']}() with no parameters and the generated "
            "button posts an empty body, so the call cannot be refused for want of an input: "
            "default whatever the specification leaves open"
        )
    text += (
        ". A 2xx is not a result - return the refusal as 4xx, or make the action perform the "
        "transition it models. Do not satisfy this by deleting a business rule"
    )
    if entry["verdict"] != "inert":
        text += (
            "; this probe reaches only the state the create route accepts, so the refusal "
            "may have a precondition it cannot construct"
        )
    return text


def _missing_not_null_column(body: str) -> str | None:
    for pattern in _NOT_NULL_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Child: runs inside the app copy (cwd), prints one report line
# ---------------------------------------------------------------------------

class _NetworkBlocked(OSError):
    """Raised in place of a real connect() the probe subprocess attempted."""


def _is_local_host(host) -> bool:
    if host in (None, ""):
        return True
    if isinstance(host, bytes):
        try:
            host = host.decode("idna")
        except Exception:
            return False
    if isinstance(host, str) and host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _install_network_guard() -> None:
    """Block outbound sockets for the rest of this (one-shot) subprocess.

    The generated app is LLM-authored and this probe executes it - nothing
    should stop it from making a real outbound call otherwise. Loopback stays
    allowed: on Windows, ``asyncio.run()`` itself opens a loopback socket pair
    for its wakeup self-pipe (no native ``socketpair()`` here), so blocking
    loopback would break the probe's own event loop, not just the app.

    Only ``connect``/``connect_ex`` are patched: ``socket.create_connection``
    and every higher-level client (httpx, requests, urllib, a raw DB driver)
    end up calling one of those two on a ``socket.socket`` instance, and the
    probe's own traffic never goes through a real socket at all - it talks to
    the app over ``httpx.ASGITransport`` (in-memory) and to SQLite (a file).
    Fails open: a probe that cannot install its guard should still run.
    """
    try:
        import socket

        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex

        def _check(address) -> None:
            host = address[0] if isinstance(address, tuple) and address else None
            if host is not None and not _is_local_host(host):
                raise _NetworkBlocked(
                    f"constructibility probe: outbound network access blocked ({host!r})")

        def guarded_connect(self, address):
            _check(address)
            return real_connect(self, address)

        def guarded_connect_ex(self, address):
            _check(address)
            return real_connect_ex(self, address)

        socket.socket.connect = guarded_connect
        socket.socket.connect_ex = guarded_connect_ex
    except Exception:
        logger.warning("constructibility probe: could not install the network guard", exc_info=True)


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
        # "Freshly created, in its initial state" has to mean it: a lifecycle
        # flag starts false. Sampling True made the probe approve an
        # already-approved timesheet and then read the handler's correct
        # refusal as a defect (gpt-5.6-terra-1s14uohe, 5/5 checks passed).
        if variant[0] == "initial" and resolved.get("type") == "boolean":
            payload[field] = False
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


# ---------------------------------------------------------------------------
# Action/method endpoints: ``POST /<entity>/{id}/methods/<action>/``
#
# A create probe proves nothing about these - they carry the app's actual
# behaviour and a static stub-scan (action_inventory.py) only catches an
# empty/placeholder body, not a handler with real code that is wrong. Live
# case: run iw82zzoc's registerArrival/registerDeparture/cancel each compare
# a loaded enum column to `SomeEnum.LITERAL.value` (a raw string) - that
# comparison is never true, in ANY state, so the handler refuses every call.
#
# A 4xx from one call is NOT evidence of a defect: refusing in the wrong
# state is the entire point of a guarded action. The only way to tell
# "refuses in every state" from "correctly refuses in THIS state" without a
# spec is to construct more than one state and see whether the same action
# ever succeeds anywhere. That is only possible for entities whose own
# create schema exposes the state-carrying field, and only names fields that
# read as the entity's own lifecycle ("status"/"state"/"stage"/"phase") -
# see Bill.settled in iw82zzoc, a required boolean that is NOT what
# registerPayment's guard reads (it reads the *linked Booking's*
# commercialStatus): varying it would have produced misleading evidence.
# Each action is probed against its OWN fresh batch of instances, never
# shared with another action, so one action's side effects can never taint
# another's evidence.
# ---------------------------------------------------------------------------

_ACTION_ROUTE_RE = re.compile(r"^/(?P<entity_seg>[^/{}]+)/\{[^/{}]+\}/methods/(?P<action>[^/{}]+)/?$")
_STATUS_FIELD_TOKENS = ("status", "state", "stage", "phase")
_MAX_STATE_LITERALS = 6      # literals sampled per status-like field
_MAX_ACTION_PROBE_REQUESTS = 300  # extra creates + action calls, combined
_MAX_COUNTED_ENTITIES = 16   # collections re-counted around one action call
_MAX_INERT_PER_ENTITY = 3
_MAX_STATE_VARIANTS = 5      # initial state + lifecycle literals, per aggregate
# A body that says the call did not do its job. ``success``/``ok``/``error``
# are the shapes LLM-authored handlers use; ``result: false`` is the
# deterministic scaffold's own, and is exactly what MethodButton.tsx reads as
# "the method declined to act" (``wasDeclined``). ``result: false`` alone is
# the weak one - a bool-returning query says the same thing honestly - so it
# never carries a blocker on its own.
_FAILURE_STATUS_WORDS = frozenset({
    "failed", "failure", "error", "errored", "rejected", "declined", "refused",
})


def _declared_failure(text: str) -> tuple[str, bool] | None:
    """``(marker, decisive)`` when the response admits it did not act."""
    try:
        body = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    for key in ("success", "ok", "succeeded"):
        value = body.get(key)
        if value is False or (isinstance(value, str) and value.strip().lower() == "false"):
            return f"{key}=false", True
    status = body.get("status")
    if isinstance(status, str) and status.strip().lower() in _FAILURE_STATUS_WORDS:
        return f"status={status.strip()}", True
    if body.get("error"):
        return "error", True
    result = body.get("result")
    if result is False or (isinstance(result, str) and result.strip().lower() == "false"):
        return "result=false", False
    return None


def _read_routes(spec: dict, creates: dict) -> dict:
    """entity -> the ``GET /<entity>/{id}/`` route that reads one back."""
    found: dict = {}
    for entity, (path, _schema) in creates.items():
        pattern = re.compile(r"^" + re.escape(path.rstrip("/")) + r"/\{[^/{}]+\}/?$")
        for candidate, ops in spec.get("paths", {}).items():
            if "get" in ops and pattern.match(candidate):
                found[entity] = candidate
                break
    return found


def _count_routes(spec: dict, creates: dict) -> dict:
    """entity -> its ``/count/`` route. An action that creates a row elsewhere
    (``produceBill``) leaves its own entity untouched; without this the probe
    would read that correct action as having done nothing."""
    found: dict = {}
    for entity, (path, _schema) in sorted(creates.items()):
        candidate = path.rstrip("/") + "/count/"
        if "get" in (spec.get("paths", {}).get(candidate) or {}):
            found[entity] = candidate
    return dict(itertools.islice(found.items(), _MAX_COUNTED_ENTITIES))


def _effect(before, after) -> bool | None:
    """Did the call change anything the probe can see? None = cannot tell."""
    if not before or not after:
        return None
    known = False
    if before.get("entity") is not None and after.get("entity") is not None:
        known = True
        if before["entity"] != after["entity"]:
            return True
    if before.get("counts") and after.get("counts"):
        known = True
        if before["counts"] != after["counts"]:
            return True
    return False if known else None


def _status_like_fields(schema: dict, schemas: dict) -> list[tuple[str, list]]:
    """Required enum fields named like the entity's own lifecycle state."""
    found = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if not any(token in field.lower() for token in _STATUS_FIELD_TOKENS):
            continue
        literals = _resolve(props.get(field, {}), schemas).get("enum") or []
        if len(literals) >= 2:
            found.append((field, literals[:_MAX_STATE_LITERALS]))
    return found


def _action_routes(spec: dict, creates: dict) -> dict:
    """entity -> [(action, route)], matching each route's leading path
    segment to the ``creates`` entity that owns it. Only plain
    ``/<entity>/{id}/methods/<action>/`` POST routes are handled; anything
    else is left to the static action inventory."""
    segment_to_entity = {}
    for entity, (path, _schema) in creates.items():
        segment_to_entity.setdefault(path.strip("/").split("/", 1)[0].lower(), entity)
    found: dict = {}
    for path, ops in spec.get("paths", {}).items():
        if "post" not in ops:
            continue
        match = _ACTION_ROUTE_RE.match(path)
        if not match:
            continue
        entity = segment_to_entity.get(match.group("entity_seg").lower())
        if entity is not None:
            found.setdefault(entity, []).append((match.group("action"), path))
    return found


def _action_handler_site(app, path: str) -> dict | None:
    """File/function/line of the POST handler serving ``path``."""
    import inspect
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path or "POST" not in (getattr(route, "methods", None) or ()):
            continue
        fn = inspect.unwrap(route.endpoint)
        try:
            file = inspect.getsourcefile(fn)
            _, start = inspect.getsourcelines(fn)
        except (OSError, TypeError):
            return None
        if not file:
            return None
        return {"file": os.path.relpath(file, os.getcwd()).replace("\\", "/"),
                "function": fn.__name__, "line": start}
    return None


async def _create_instance(request, schema, schemas, orm, subclasses, unique_fields,
                           entity, path, ids, variant, ordinal, label):
    """One fresh instance of ``entity``; ``(id or None, ordinal)``."""
    ordinal += 1
    payload, unresolved = _build_payload(
        schema, schemas, orm.get(entity, {}), ids, subclasses, variant,
        entity=entity, ordinal=ordinal,
        unique_fields=unique_fields.get(entity, frozenset()))
    if unresolved:
        return None, ordinal
    response, attempt = await request(path, payload, label)
    if attempt["outcome"] != "created":
        return None, ordinal
    try:
        return _extract_id(response.json()), ordinal
    except Exception:
        return None, ordinal


async def _probe_actions(app, spec, request, read_state, schemas, orm, subclasses,
                         unique_fields, creates, entities, ids, ordinal,
                         model_actions) -> tuple[list, list, int]:
    """Drive the workflow, not just the creates.

    Two passes, both on instances created for this action alone so one
    action's side effects can never taint another's evidence:

    A. the aggregate's own first move - create it, read it, call the action
       with the literal ``{}`` the generated button posts, read it again.
       An action that reports a refusal and changes nothing did not run.
    B. the state sweep that was here before - vary a status-like create field
       and see whether the action ever succeeds in ANY state it can build.
    """
    reports: list = []
    calls: list = []
    budget = [_MAX_ACTION_PROBE_REQUESTS]

    def spend() -> bool:
        budget[0] -= 1
        return budget[0] >= 0

    read_routes = _read_routes(spec, creates)
    count_routes = _count_routes(spec, creates)

    for entity, actions in _action_routes(spec, creates).items():
        if entities.get(entity, {}).get("verdict") != "created":
            continue  # the create probe already reports this entity, if broken
        path, schema_name = creates[entity]
        schema = schemas[schema_name]
        declared = {a.get("name"): a for a in model_actions.get(entity) or []}
        read_route = read_routes.get(entity)

        # ---------------------------------------------------- pass A: first move
        #
        # "Initial state" is a guess, and a wrong guess invents a defect. The
        # enum literal the create schema lists first is not the lifecycle's
        # start: SessionStatus sorts CANCELLED before SCHEDULED, so the first
        # build of this pass created a cancelled session and read cancel()'s
        # correct refusal as a dead workflow (gpt-5.6-terra-5d9otfvo /
        # jl_vbrf2, both 2/4). So where the entity's own create schema can
        # select a lifecycle state, every literal is tried before concluding
        # that nothing moves - and only for an aggregate that would otherwise
        # be reported, so a healthy app pays for one pass.
        state_variants: list[tuple] = [("initial",)]
        for field, literals in _status_like_fields(schema, schemas):
            state_variants += [("enum", field, literal) for literal in literals]
        state_variants = state_variants[:_MAX_STATE_VARIANTS]

        fresh: list[dict] = []
        crashed_actions: set = set()
        moved = False
        for index, state in enumerate(state_variants):
            observed: list[dict] = []
            for action, route in actions:
                if not spend():
                    break
                new_id, ordinal = await _create_instance(
                    request, schema, schemas, orm, subclasses, unique_fields,
                    entity, path, ids, state, ordinal, ("action-fresh", entity, action))
                if new_id is None:
                    continue
                before = await read_state(read_route, new_id, count_routes)
                if not spend():
                    break
                call_path = re.sub(r"\{[^/{}]+\}", str(new_id), route, count=1)
                _, attempt = await request(call_path, {}, ("action-fresh-call", action))
                if attempt.get("status") == 501:
                    if index == 0:
                        calls.append({"entity": entity, "action": action, "route": route,
                                      "verdict": "stub", "status": attempt.get("status")})
                    continue
                after = await read_state(read_route, new_id, count_routes)
                changed = _effect(before, after)
                marker = _declared_failure(attempt.get("body") or "")
                parameters = (declared.get(action) or {}).get("parameters")
                call = {
                    "entity": entity, "action": action, "route": route,
                    "status": attempt.get("status"), "outcome": attempt["outcome"],
                    "body": attempt.get("body"), "changed": changed,
                    "marker": marker[0] if marker else None,
                    "decisive": bool(marker and marker[1]),
                    # None where the serializer did not state the list at all.
                    "zero_param": None if parameters is None else not parameters,
                    "verdict": "crashed" if attempt["outcome"] == "crashed"
                               else "effective" if changed is True and attempt["outcome"] == "created"
                               else "refused" if attempt["outcome"] not in ("created",)
                               else "inert" if changed is False else "unknown",
                }
                observed.append(call)
                if index == 0:
                    fresh.append(call)
                    calls.append(call)
                if attempt["outcome"] == "crashed" and action not in crashed_actions:
                    crashed_actions.add(action)
                    reports.append({
                        "route": route, "action": action, "entity": entity, "verdict": "crashed",
                        "state": "freshly created", "attempt": attempt,
                        "site": _action_handler_site(app, route),
                    })
            # One action that moves the aggregate is its first transition; the
            # rest may then be refusing correctly.
            if any(c["changed"] is True and c["outcome"] == "created" for c in observed):
                moved = True
                break
            if index == 0 and not any(c["outcome"] == "created" and c["changed"] is False
                                      and c["marker"] for c in fresh):
                break  # nothing to report anyway; do not pay for the sweep

        if not moved:
            inert = [c for c in fresh if c["outcome"] == "created"
                     and c["changed"] is False and c["marker"]]
            decisive = [c for c in inert if c["decisive"] and c["zero_param"]]
            decisive_ids = {id(c) for c in decisive}
            siblings = sorted({c["action"] for c in fresh})
            for call in (decisive or inert)[:_MAX_INERT_PER_ENTITY]:
                reports.append({
                    "route": call["route"], "action": call["action"], "entity": entity,
                    "verdict": "inert" if id(call) in decisive_ids else "inert unverified",
                    "call": call, "siblings": siblings, "zero_param": call["zero_param"],
                    "site": _action_handler_site(app, call["route"]),
                })

        # ---------------------------------------------------- pass B: state sweep
        status_fields = _status_like_fields(schema, schemas)
        if not status_fields:
            continue  # no own-field lever to vary state; stay silent
        for action, route in actions:
            if not spend():
                break
            if action in crashed_actions:
                continue  # pass A already watched this handler raise
            samples: list[tuple[str, object]] = []
            for field, literals in status_fields:
                for literal in literals:
                    if not spend():
                        break
                    new_id, ordinal = await _create_instance(
                        request, schema, schemas, orm, subclasses, unique_fields,
                        entity, path, ids, ("enum", field, literal), ordinal,
                        ("action-state", field, literal))
                    if new_id is not None:
                        samples.append((f"{field}={literal}", new_id))
            if not samples:
                continue
            call_path_template = route
            attempts: list = []
            crashed = None
            stub = False
            for label, sample_id in samples:
                if not spend():
                    break
                call_path = re.sub(r"\{[^/{}]+\}", str(sample_id), call_path_template, count=1)
                _, attempt = await request(call_path, {"params": {}}, ("action-call", action, label))
                if attempt.get("status") == 501:
                    # The deterministic scaffold's own "no body in the model" marker
                    # (router_methods.py.j2) - action_inventory.py's static scan
                    # already reports this precisely. Reporting it again here as a
                    # "server failure" would be redundant and mislabel an honest stub.
                    stub = True
                    break
                attempts.append((label, attempt))
                if crashed is None and attempt["outcome"] == "crashed":
                    crashed = (label, attempt)
            if stub:
                continue
            if crashed is not None:
                reports.append({
                    "route": route, "action": action, "entity": entity, "verdict": "crashed",
                    "state": crashed[0], "attempt": crashed[1],
                    "site": _action_handler_site(app, route),
                })
            elif len(attempts) >= 2 and not any(a["outcome"] == "created" for _, a in attempts):
                reports.append({
                    "route": route, "action": action, "entity": entity, "verdict": "unverified",
                    "states": [label for label, _ in attempts],
                    "statuses": [a.get("status") for _, a in attempts],
                    "site": _action_handler_site(app, route),
                })
    return reports, calls, ordinal


def _load_model_actions() -> dict:
    """The modelled action list the parent wrote out, if it had a model."""
    path = os.environ.get(_ACTIONS_ENV)
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _probe_cwd() -> dict:
    _install_network_guard()
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
    abstract = {name.strip() for name in
                (os.environ.get(_ABSTRACT_ENV) or "").split(",") if name.strip()}
    skipped_abstract: list = []
    for path, ops in spec.get("paths", {}).items():
        body = (ops.get("post") or {}).get("requestBody", {})
        ref = body.get("content", {}).get("application/json", {}).get("schema", {}).get("$ref", "")
        if ref.endswith("Create") and ref.rsplit("/", 1)[-1] in schemas:
            schema_name = ref.rsplit("/", 1)[-1]
            entity = schema_name[: -len("Create")]
            if entity in abstract:
                # Refusing to instantiate it is the correct behaviour; a
                # concrete subclass carries the same rows.
                skipped_abstract.append(entity)
                continue
            creates[entity] = (path, schema_name)

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

            async def read_state(read_path, entity_id, count_routes):
                """What the app says exists right now: the row itself and the
                size of every collection. A key the app will not serve comes
                back None, which makes the effect unknown, never a finding."""
                snapshot: dict = {"entity": None, "counts": {}}
                if read_path:
                    url = re.sub(r"\{[^/{}]+\}", str(entity_id), read_path, count=1)
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            snapshot["entity"] = json.loads(
                                json.dumps(response.json(), sort_keys=True, default=str))
                    except Exception:
                        pass
                for name, route in count_routes.items():
                    try:
                        response = await client.get(route)
                        if response.status_code == 200:
                            snapshot["counts"][name] = json.dumps(
                                response.json(), sort_keys=True, default=str)
                    except Exception:
                        continue
                return snapshot

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

            actions, action_calls, _ordinal = await _probe_actions(
                app, spec, request, read_state, schemas, orm, subclasses, unique_fields,
                creates, entities, ids, ordinal, _load_model_actions())
        return {"entities": entities, "actions": actions, "action_calls": action_calls}

    try:
        result = asyncio.run(run())
    except Exception as exc:
        return {"boot": "probe_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"boot": "ok", "entities": result["entities"], "actions": result["actions"],
            "action_calls": result["action_calls"],
            # Named so a reader can tell "not probed because the model says
            # it is abstract" from "probed and passed".
            "skipped_abstract": sorted(set(skipped_abstract))}


if __name__ == "__main__":
    print(_MARKER + json.dumps(_probe_cwd(), default=str))
