"""Phase 3: once an app has accounts, what can one account do to another's records?

Every finding here is a warning for the run's report: it is never a blocker,
never sent to repair and never ranks a tree. It states what was observed; a
designed feature (a shared workspace, a role, a link the request asks for) can
look the same from outside, and the user decides.

The constructibility probe logs in and creates records; it never asks what a
second account may do to them. This check does, for a backend with a
registration route. It talks to the app the way any visitor could: through its
public HTTP API, as accounts it registers through the app's own routes, and as
a caller who is not logged in. It overrides no dependency, forges no token and
never opens the database.

Account A creates one record of each entity. Account B then sends every update
method of each record; where B is refused, A sends the same update, and only
when A is allowed is the refusal a rule of the app (a route that refuses
everyone is not one). B also sends the routes under each record that take the
record's own body, every other entity's create and update with A's record in
each relationship field whose foreign key is stored on A's record, and the
deletes. A reads its record back after each request, and that read-back is the
result: a request counts only when a field now holds the value B sent under
that name, a link now points at B's record, the record is gone, or A can no
longer read it. A timestamp, counter or derived value that moved on its own
counts for nothing. A change is reported where the app refused B on that record
elsewhere, and always on A's own account record.

Runs in two halves, like the constructibility probe: the parent
(``authorization_issues``) is orchestrator code, the child is this file
executed by path inside a scratch copy of the app, importing its sibling probes
and never ``besser``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

if __name__ == "__main__":  # the child, run by path beside its sibling probes and without ``besser``
    _STARTED = time.monotonic()
    import api_probe as _scenarios
    # constructibility.py reaches this sibling through the package unless it is the script itself.
    sys.modules.setdefault("besser.spec_driven_agent.validation.api_probe", _scenarios)
    import constructibility as _probe
else:
    from besser.spec_driven_agent.validation import api_probe as _scenarios
    from besser.spec_driven_agent.validation import constructibility as _probe

logger = logging.getLogger(__name__)

PREFIX = "authorization:"
COVERAGE_PREFIX = "authorization coverage:"

_MARKER = "BESSER_AUTHORIZATION_REPORT:"
_CHILD_TIMEOUT_SECONDS = 90
# The child stops asking this long after it started, boot included.
_WALL_SECONDS = 60
_MAX_REQUESTS = 1500
_REFUSED = (401, 403)
# How an app answers another account's request on a record: refused, or reported as absent.
_HIDDEN = (401, 403, 404)
# How a login route answers credentials it no longer knows; a 429 or a 500 says nothing.
_LOGIN_REFUSED = (400, 401, 403, 404)
# Samples for the accounts start here, so no record's sampled phone or name equals an account's.
_ACCOUNT_ORDINAL = 900_000
_STALE_TOKEN_ROUTES = 3
_ROUTES_SHOWN = 6
# Routes of the login flow besides register and login: open to a caller who is not logged in by nature.
_SESSION_PATH = re.compile(
    r"/(logout|signout|sign-out|refresh|forgot[-_]?password|reset[-_]?password|verify|confirm)[^/]*/?$", re.I)
_PATH_LITERAL = re.compile(r"""["'](/[^"'\n{}]*)["']""")
# What a stored or issued credential looks like, as opposed to a field merely named like one.
_HASH_SHAPE = re.compile(r"\$2[aby]\$|\$argon2|\$scrypt|pbkdf2|[0-9a-f]{32,}$|eyJ[\w-]{8,}\.", re.I)
_PASSWORD_WORDS = frozenset({"password", "passwd", "passphrase", "pwd"})
# Fields of an account record that say who the account is or how it logs in.
_IDENTITY_FIELD = re.compile(r"e-?mail|user_?name|login|passw|phone", re.I)
_CONTACT = r"(?:profiles?|contact (?:details?|info(?:rmation)?)|e-?mail(?: address(?:es)?)?|phone(?: numbers?)?)"
_PUBLIC = r"(?:public|publicly|visible to (?:everyone|anyone|all|visitors)|shown to (?:everyone|anyone|all|visitors))"
# A request that makes account data public: its contact data and its share tokens are then no finding.
_PUBLIC_CONTACT_RE = re.compile(
    rf"\b{_PUBLIC}\b[^.\n]{{0,60}}\b{_CONTACT}\b|\b{_CONTACT}\b[^.\n]{{0,60}}\b{_PUBLIC}\b", re.I)


# ---------------------------------------------------------------------------
# Parent: run the child for each backend with accounts, word what it saw
# ---------------------------------------------------------------------------

def authorization_issues(output_dir: str, backends: list[dict], request_text: str = "",
                         domain_model=None) -> list[str]:
    """This check's warnings for every backend in ``backends`` that has accounts; never raises.

    ``backends`` is the constructibility probe's report. A route that probe
    already reported as an ``open account writes:`` warning is not reported
    again.
    """
    from besser.spec_driven_agent.planning.requirements_ledger import original_request

    issues: list[str] = []
    for backend in backends:
        rel = backend["backend"]
        try:
            folder = os.path.join(output_dir, rel)
            walled = bool((backend.get("auth") or {}).get("source"))
            if backend.get("boot") != "ok" or not (walled or _declares_registration(folder)):
                continue
            request = original_request(request_text)
            abstract = _probe.abstract_entities(domain_model)
            report = _run_child(folder, workspace=output_dir, abstract=abstract)
            issues.extend(_findings(report, rel, request, upstream=backend.get("open_account_writes") or (),
                                    walled=walled))
        except Exception as exc:
            logger.warning("Authorization check crashed; cross-account access not checked", exc_info=True)
            issues.extend(_findings({"boot": "error", "error": f"it crashed: {type(exc).__name__}: {exc}"}, rel))
    return issues


def _declares_registration(folder: str) -> bool:
    """Whether the backend's source names a registration route: it has accounts, login wall met or not."""
    from besser.utilities.workspace_fs import walk_plain

    for root, dirs, files in walk_plain(folder):
        dirs[:] = sorted(d for d in dirs if d not in _probe._SKIP_DIRS and d != "venv" and not d.startswith("."))
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8-sig") as handle:
                    source = handle.read()
            except (OSError, UnicodeError):
                continue
            if any(_probe._REGISTER_PATH.search(literal) for literal in _PATH_LITERAL.findall(source)):
                return True
    return False


def _run_child(folder: str, *, workspace: str, abstract: list[str] | None = None) -> dict:
    """Execute the child, sandboxed with no network, on a scratch copy of ``folder``."""
    # Mirrors constructibility._run_probe, which can only launch its own file as the child.
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env, timeout_cause
    from besser.spec_driven_agent.execution.sandbox import SandboxUnavailable, run_confined
    from besser.utilities.workspace_fs import copytree_plain

    work = tempfile.mkdtemp(prefix="besser_probe_")
    try:
        app_dir = os.path.join(work, "app")
        copytree_plain(folder, app_dir, (*_probe._SKIP_DIRS, ".besser_*", "*.db"))
        env = _safe_subprocess_env()
        env["DATABASE_URL"] = "sqlite:///" + os.path.join(work, "probe.db").replace("\\", "/")
        if abstract:
            env[_probe._ABSTRACT_ENV] = ",".join(abstract)
        try:
            result = run_confined([sys.executable, os.path.abspath(__file__)], workspace=workspace,
                                  writable=[work], cwd=app_dir, env=env, timeout=_CHILD_TIMEOUT_SECONDS)
        except SandboxUnavailable as exc:
            return {"boot": "error", "error": str(exc)}
        except subprocess.TimeoutExpired as exc:
            return {"boot": "error", "error": timeout_cause(exc, f"timed out after {_CHILD_TIMEOUT_SECONDS}s")}
        except OSError as exc:
            return {"boot": "error", "error": f"could not be launched: {exc}"}
        for line in reversed((result.stdout or "").splitlines()):
            if line.startswith(_MARKER):
                try:
                    return json.loads(line[len(_MARKER):])
                except ValueError:
                    break
        tail = [ln for ln in (result.stderr or "").splitlines() if ln.strip()]
        return {"boot": "error", "error": (tail[-1] if tail else "no report")[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _listed(routes: list[str]) -> str:
    routes = list(dict.fromkeys(routes))
    more = len(routes) - _ROUTES_SHOWN
    return ", ".join(routes[:_ROUTES_SHOWN]) + (f" (+{more} more)" if more > 0 else "")


def _effect_text(effect: dict, rule: dict | None, account: bool) -> str:
    """One observed change of account A's record by account B, stated as observed."""
    entity, route, kind = effect["entity"], effect["route"], effect["kind"]
    if kind == "delete":
        text = f"another account could delete a {entity}{' account' if account else ''} through {route}"
    elif kind == "lockout":
        text = (f"after another account's request through {route} was accepted, "
                + ("the account could no longer read its own record" if account
                   else f"the account that created a {entity} could no longer read it"))
    else:
        keys = [effect["relation"]] if kind == "relation" else effect["keys"]
        text = f"another account could change {', '.join(f'{entity}.{key}' for key in keys)} through {route}"
        identity = [f"{entity}.{key}" for key in keys if account and _IDENTITY_FIELD.search(key)]
        if identity:
            text += f"; {', '.join(identity)} {'identifies' if len(identity) == 1 else 'identify'} the account"
    return text + (f"; {rule['route']} refused that account" if rule else "")


def _findings(report: dict, rel: str, request: str = "", *, upstream=(), walled: bool = True) -> list[str]:
    """What the child observed, worded for the user: what other callers could do, then what was not covered."""
    if report.get("boot") != "ok" or report.get("skipped"):
        # An app that offers no registration is told so only when the probe met its login wall.
        if report.get("no_accounts") and not walled:
            return []
        reason = report.get("skipped") or f"the check did not run ({report.get('error') or 'no report'})"
        return [f"{COVERAGE_PREFIX} {rel}: {reason}; what one account can do to another's records was not checked"]
    issues: list[str] = []
    public = bool(_PUBLIC_CONTACT_RE.search(request or ""))
    entities: dict = report.get("entities") or {}
    account = report.get("account_entity")
    # What the constructibility probe already reported about account records.
    anonymous_reported = {route for entry in upstream for route in entry.get("routes") or ()}
    taken_reported = {taken["route"] for entry in upstream for taken in entry.get("fields") or ()}

    for effect in report.get("effects") or []:
        entity, kind, route = effect["entity"], effect["kind"], effect["route"]
        facts = entities.get(entity) or {}
        rule = (facts.get("delete_rule") or facts.get("rule")) if kind == "delete" else facts.get("rule")
        if rule and rule["route"] == route:
            rule = None
        own_account = entity == account and kind in ("update", "delete", "lockout")
        if (rule or own_account) and not (own_account and route in taken_reported):
            issues.append(f"{PREFIX} {rel}: {_effect_text(effect, rule, own_account)}")

    open_routes = [entry["route"] for entry in report.get("anonymous") or [] if entry["route"] not in anonymous_reported]
    if open_routes:
        issues.append(f"{PREFIX} {rel}: a caller who is not logged in could successfully send {_listed(open_routes)}")

    exposed = report.get("exposed") or []
    # A share token of a record the request makes public is meant to be seen; a password never is.
    secret = {name for entry in exposed for name in entry["credentials"]
              if not public or set(_scenarios.name_words(name)) & _PASSWORD_WORDS}
    if secret:
        routes = [entry["route"] for entry in exposed if secret & set(entry["credentials"])]
        issues.append(f"{PREFIX} {rel}: {_listed(routes)} returned an account's "
                      + ", ".join(f"`{name}`" for name in sorted(secret)) + " to other callers")
    contact = sorted({name for entry in exposed for name in entry["fields"]})
    if contact and not public:
        issues.append(f"{PREFIX} {rel}: a caller who is not logged in could read an account's "
                      f"{' and '.join(contact)} through {_listed([e['route'] for e in exposed if e['fields']])}")

    stale = report.get("stale_token") or []
    if stale:
        issues.append(f"{PREFIX} {rel}: {_listed([entry['route'] for entry in stale])} still accepted the token of "
                      "an account that had deleted itself and could no longer log in")

    uncovered: dict = report.get("uncovered") or {}
    if uncovered:
        issues.append(
            f"{COVERAGE_PREFIX} {rel}: records of "
            + "; ".join(f"{entity} ({why})" for entity, why in sorted(uncovered.items()))
            + " could not be set up for the check, so what other accounts can do to them was not checked")
    if report.get("stopped"):
        issues.append(f"{COVERAGE_PREFIX} {rel}: the check stopped early ({report['stopped']}) after "
                      f"{report.get('requests', 0)} requests; the routes it had not reached were not checked")
    return issues


# ---------------------------------------------------------------------------
# Child: the decision on one observed request
# ---------------------------------------------------------------------------

_UNREADABLE = object()
_LOCKED_OUT = object()


def _record(body) -> dict | None:
    """The record in a read response: the object carrying ``id``, bare or one level down."""
    if isinstance(body, dict):
        if isinstance(body.get("id"), (int, str)):
            return body
        for value in body.values():
            if isinstance(value, dict) and isinstance(value.get("id"), (int, str)):
                return value
    return None


def _norm(name: str) -> str:
    return "".join(_scenarios.name_words(name))


def _landed(before: dict, after: dict, *, sent: dict | None = None,
            linked: tuple = (), link_keys: tuple = ()) -> list[str]:
    """Fields of a record now holding what another account supplied.

    ``sent`` is that account's request body: a field counts when it reads back
    the value sent under its own name. ``linked`` are ids of that account's
    records: a link field counts when it now points at one of them
    (``link_keys`` when the record shows them, else any ``*id`` field).
    Booleans and nulls are too easily equal by chance to count.
    """
    by_name = {_norm(key): value for key, value in (sent or {}).items()
               if value is not None and not isinstance(value, (bool, dict, list))}
    ids = {str(value) for value in linked if value is not None}
    shown = [key for key in link_keys if key in after]
    changed = []
    for key in sorted(set(before) & set(after)):
        old, new = before[key], after[key]
        if old == new or new is None or isinstance(new, (bool, dict, list)):
            continue
        is_link = key in shown if shown else _scenarios.name_words(key)[-1:] == ["id"]
        if (is_link and str(new) in ids) or (_norm(key) in by_name and by_name[_norm(key)] == new):
            changed.append(key)
    return changed


async def _effect(api, read, actor: str, method: str, path: str, body=None) -> dict | None:
    """One request by ``actor`` with the owner's record read either side of it.

    None when the owner cannot read the record beforehand: nothing to compare.
    """
    before = await read()
    if not isinstance(before, dict):
        return None
    status, answer = await api.call(actor, method, path, **({} if body is None else {"json": body}))
    return {"status": status, "answer": answer, "before": before, "after": await read()}


def _judge(seen: dict | None, *, sent: dict | None = None, linked: tuple = (), link_keys: tuple = ()) -> dict | None:
    """What the request did to the owner's record: ``{"gone"}``, ``{"locked"}``, ``{"keys"}`` or None.

    A 2xx that changed nothing is None, and so is a read-back that failed for
    a reason of its own. The owner refused its own record right after the
    other account's accepted request is a lockout.
    """
    if seen is None or seen["after"] is _UNREADABLE:
        return None
    if seen["after"] is None:
        return {"gone": True}
    if seen["after"] is _LOCKED_OUT:
        return {"locked": True} if 200 <= seen["status"] < 300 else None
    keys = _landed(seen["before"], seen["after"], sent=sent, linked=linked, link_keys=link_keys)
    return {"keys": keys} if keys else None


def _credential_shaped(value, password: str) -> bool:
    """Whether ``value`` is a credential, not a field that is only named like one: the account's
    own password, a hash or token format, or a long string of mixed characters."""
    if not isinstance(value, str) or " " in value:
        return False
    if value == password or _HASH_SHAPE.match(value):
        return True
    return (len(value) >= 20 and len(set(value)) >= 10
            and any(c.isdigit() for c in value) and any(c.isalpha() for c in value))


def _credentials_beside(body, identity: str, password: str) -> set:
    """Credential fields of the objects in ``body`` that carry ``identity``."""
    found: set = set()
    if isinstance(body, dict):
        if identity in body.values():
            found |= {key for key, value in body.items()
                      if _scenarios._credential_field(key, value) and _credential_shaped(value, password)}
        for value in body.values():
            found |= _credentials_beside(value, identity, password)
    elif isinstance(body, list):
        for item in body:
            found |= _credentials_beside(item, identity, password)
    return found


def _param_entity(param: str, entity: str | None, orm: dict) -> str | None:
    """The entity whose id a path parameter carries: the route's own, a related one, or one it names."""
    name = re.sub(r"(_id|Id|_pk)$", "", param)
    if entity and name.lower() in (entity.lower(), "id", "item", "pk"):
        return entity
    if entity and name in orm.get(entity, {}):
        return orm[entity][name][0]
    return next((other for other in sorted(orm) if other.lower() == name.lower()), None)


# ---------------------------------------------------------------------------
# Child: runs inside the app copy (cwd), prints one report line
# ---------------------------------------------------------------------------

class _Stop(Exception):
    """The request or time budget is spent."""


class _Api:
    """The app's public HTTP API as several callers, each with its own headers and cookie jar."""

    def __init__(self, httpx, app, started: float):
        self._httpx = httpx
        self._transport = httpx.ASGITransport(app=app)
        self._clients: dict = {}
        self._deadline = started + _WALL_SECONDS
        self.requests = 0

    def client(self, actor: str):
        if actor not in self._clients:
            # https: a login kept in a ``Secure`` cookie is sent back only over https.
            self._clients[actor] = self._httpx.AsyncClient(transport=self._transport, base_url="https://probe")
        return self._clients[actor]

    async def call(self, actor: str, method: str, path: str, **body) -> tuple[int, object]:
        """``(status, JSON or text)``; an exception the app did not handle is a 500."""
        if self.requests >= _MAX_REQUESTS:
            raise _Stop(f"the cap of {_MAX_REQUESTS} requests")
        if time.monotonic() > self._deadline:
            raise _Stop(f"the {_WALL_SECONDS}s time limit")
        self.requests += 1
        try:
            response = await self.client(actor).request(method, path, **body)
        except Exception as exc:
            return 500, f"{type(exc).__name__}: {exc}"
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()


def _column_keys(mapper, columns) -> list[str]:
    keys = []
    for column in columns:
        try:
            keys.append(mapper.get_property_by_column(column).key)
        except Exception:  # a column the class maps under no attribute
            keys.append(column.key)
    return sorted(keys)


def _describe(app, spec: dict, sql_alchemy, sa_inspect) -> dict:
    """The app as the check needs it: ORM relationships and the routes of each entity."""
    # Mirrors the ORM walk and create map constructibility._probe_cwd builds inline, which this child cannot
    # import. One difference, pinned by a test: an untemplated create route wins here, the probe keeps the last.
    orm: dict = {}
    unique: dict = {}
    child_keys: dict = {}  # (entity, relationship) -> foreign-key fields on the related record
    for value in vars(sql_alchemy).values():
        if not (isinstance(value, type) and hasattr(value, "__mapper__")):
            continue
        name = value.__name__
        try:
            mapper = sa_inspect(value)
            orm[name] = {r.key: (r.mapper.class_.__name__, bool(r.uselist)) for r in mapper.relationships}
            unique_columns = {
                col for table in mapper.tables
                for constraint in (*table.constraints, *table.indexes)
                if constraint.__class__.__name__ in {"UniqueConstraint", "PrimaryKeyConstraint"}
                or getattr(constraint, "unique", False)
                for col in constraint.columns
            }
            unique[name] = frozenset(
                prop.key for prop in mapper.column_attrs
                if any(col.unique or col.primary_key or col in unique_columns for col in prop.columns))
            for r in mapper.relationships:
                if r.direction.name == "ONETOMANY":
                    child_keys[(name, r.key)] = _column_keys(r.mapper, r.remote_side)
        except Exception:
            orm.setdefault(name, {})
    subclasses: dict = {}
    for name in orm:
        for base in getattr(sql_alchemy, name).__mro__[1:]:
            if base.__name__ in orm:
                subclasses.setdefault(base.__name__, []).append(name)

    paths: dict = spec.get("paths", {})
    schemas: dict = spec.get("components", {}).get("schemas", {})
    abstract = {name.strip() for name in (os.environ.get(_probe._ABSTRACT_ENV) or "").split(",") if name.strip()}
    creates: dict = {}
    other_tables: dict = {}
    # A route such as POST /project/{project_id}/tasks/ may take the same body; the plain one creates.
    for path, ops in sorted(paths.items(), key=lambda item: ("{" in item[0], item[0])):
        body = (ops.get("post") or {}).get("requestBody", {})
        name = body.get("content", {}).get("application/json", {}).get("schema", {}).get("$ref", "").rsplit("/", 1)[-1]
        entity = name[:-len("Create")]
        if not name.endswith("Create") or name not in schemas or entity in abstract:
            continue
        (creates if entity in orm else other_tables).setdefault(entity, (path, name))
    for entity, (path, name) in other_tables.items():
        try:
            tables = [table for table in _probe._tables_behind(app, path, orm) if table not in creates]
        except _probe._Unknowable:
            tables = []
        if len(tables) == 1:
            creates[tables[0]] = (path, name)
    collections = {entity: (f"/{entity.lower()}/", "") for entity in orm} | creates
    items = _probe._read_routes(spec, collections)
    routes: dict = {}
    for entity, (collection, _schema) in collections.items():
        bulk = collection.rstrip("/") + "/bulk/"
        routes[entity] = {"collection": collection, "item": items.get(entity), "bulk": bulk if bulk in paths else None}
    return {"app": app, "paths": paths, "schemas": schemas, "orm": orm, "unique": unique,
            "child_keys": child_keys, "subclasses": subclasses, "creates": creates, "routes": routes,
            "auth_routes": _probe._auth_routes(spec)}


class _Check:
    """One run against a booted app; ``report`` is what it observed."""

    def __init__(self, api, facts: dict):
        self.api = api
        self.app = facts["app"]
        self.paths, self.schemas, self.orm = facts["paths"], facts["schemas"], facts["orm"]
        self.unique, self.subclasses, self.child_keys = facts["unique"], facts["subclasses"], facts["child_keys"]
        self.creates, self.routes, self.auth_routes = facts["creates"], facts["routes"], facts["auth_routes"]
        self.segments = {route["collection"].strip("/").split("/", 1)[0].lower(): entity
                         for entity, route in sorted(self.routes.items())}
        self.account_entity: str | None = None
        self.subjects: dict = {}  # entity -> id of the record account A created
        self.mine: dict = {}      # entity -> id of a record that is account B's own
        self.created: list = []   # entities in the order account A created them
        self.guarded: set = set() # write routes that refused a caller who is not logged in
        self.ordinal = 0
        self.accounts = 0
        self.report: dict = {"boot": "ok", "account_entity": None,
                             "creates": {entity: path for entity, (path, _) in sorted(self.creates.items())},
                             "created": self.created, "uncovered": {}, "entities": {}, "effects": [],
                             "anonymous": [], "exposed": [], "stale_token": []}

    # ------------------------------------------------------------ requests

    def payload(self, entity: str, schema: dict, ids: dict, extra: str | None = None) -> dict | None:
        """A body for ``schema`` from the probe's samples; ``extra`` is filled as if required.

        None when a related id it needs is missing.
        """
        self.ordinal += 1
        if extra:
            schema = {**schema, "required": [*schema.get("required", []), extra]}
        body, unresolved = _probe._build_payload(
            schema, self.schemas, self.orm.get(entity, {}), ids, self.subclasses, ("base",),
            entity=entity, ordinal=self.ordinal, unique_fields=self.unique.get(entity, frozenset()))
        return None if unresolved else body

    def body_schema(self, op: dict) -> tuple[dict | None, str]:
        """``(JSON body schema as declared, name of the component it or its items refer to)``."""
        schema = (((op.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema")
        items = (schema or {}).get("items", {}) if (schema or {}).get("type") == "array" else schema or {}
        return schema, items.get("$ref", "").rsplit("/", 1)[-1]

    def body(self, op: dict, ids: dict) -> tuple[bool, object]:
        """``(buildable, JSON body)`` for ``op``; an operation without a body is ``(True, None)``."""
        schema, name = self.body_schema(op)
        if schema is None:
            return True, None
        many = schema.get("type") == "array"
        if name not in self.schemas:
            return True, [] if many else {}
        record = self.payload(re.sub(r"(Create|Update|Patch)$", "", name), self.schemas[name], ids)
        return record is not None, [record] if many else record

    def update_body(self, entity: str, op: dict, ids: dict, extra: str | None = None) -> dict | None:
        """A body for an update of ``entity`` with some text in it to read back."""
        schema = self.schemas.get(self.body_schema(op)[1])
        if not schema:
            return None
        return self.payload(entity, schema, ids, extra=extra or self.free_text(entity, schema)) or None

    def free_text(self, entity: str, schema: dict) -> str | None:
        """An optional plain-text field to fill when the required ones hold no text to read back."""
        props = schema.get("properties", {})

        def plain(field: str) -> bool:
            resolved = _probe._resolve(props[field], self.schemas)
            return (resolved.get("type") == "string" and "enum" not in resolved and "format" not in resolved
                    and _probe._relationship_for(field, self.orm.get(entity, {})) is None)

        if any(plain(field) for field in schema.get("required", []) if field in props):
            return None
        return next((field for field in sorted(props) if plain(field)), None)

    def entity_of(self, path: str) -> str | None:
        return self.segments.get(path.strip("/").split("/", 1)[0].lower())

    def url(self, template: str, entity: str | None, ids: dict | None = None) -> str | None:
        """``template`` with each parameter replaced by the id of the record it names, account A's by default."""
        ids = self.subjects if ids is None else ids
        missing = False

        def fill(match):
            nonlocal missing
            value = ids.get(_param_entity(match.group(1), entity, self.orm))
            missing |= value is None
            return str(value)

        url = re.sub(r"\{([^/{}]+)\}", fill, template)
        return None if missing else url

    def reader(self, entity: str):
        """Account A reading its record of ``entity``: its plain fields, None when gone."""
        url = self.url(self.routes[entity]["item"], entity)

        async def read():
            status, answer = await self.api.call("a", "GET", url)
            if status == 404:
                return None
            if status in _REFUSED:
                return _LOCKED_OUT
            record = _record(answer) if status == 200 else None
            if record is None:
                return _UNREADABLE
            return {key: value for key, value in record.items() if not isinstance(value, (dict, list))}

        return read

    def order(self) -> list[str]:
        """Entities with a readable record of account A, the account itself left out."""
        return [name for name in sorted(self.subjects) if name != self.account_entity
                and self.routes[name]["item"] and "get" in self.paths[self.routes[name]["item"]]]

    def update_methods(self, entity: str) -> list[str]:
        item = self.routes[entity]["item"]
        return [method for method in ("put", "patch") if item and method in self.paths[item]]

    def note(self, seen: dict | None, found: dict | None, *, entity: str, method: str, path: str,
             kind: str, **extra) -> bool:
        """Record what one request by account B did to account A's ``entity``; True when it did something."""
        if not found:
            return False
        kind = "delete" if found.get("gone") else "lockout" if found.get("locked") else kind
        self.report["effects"].append({
            "route": f"{method} {path}", "entity": entity, "kind": kind, "keys": found.get("keys", []),
            "status": seen["status"], **extra})
        return True

    # ------------------------------------------------------------ accounts

    def credentials(self, op: dict, actor: str, ordinal: int) -> tuple[str | None, object]:
        """The probe's register/login body for this account, with its own contact and free-text values."""
        kind, body = _probe._credential_body(op, self.schemas, (f"authz_{actor}", f"authz.{actor}@example.com"))
        if kind is None:
            return None, body
        for field, raw in _probe._body_schema(op, self.schemas)[1].get("properties", {}).items():
            resolved, value = _probe._resolve(raw, self.schemas), body.get(field)
            plain_text = (resolved.get("type") == "string" and "enum" not in resolved
                          and resolved.get("format") not in ("date", "date-time", "time"))
            sampled = isinstance(value, str) and not ("password" in field.lower() or "email" in field.lower()
                                                      or value == f"authz_{actor}")
            if plain_text and (sampled or (field not in body and "phone" in field.lower())):
                body[field] = _probe._sample(field, resolved, 1, entity="authz", ordinal=ordinal)
        return kind, body

    async def open_account(self, actor: str) -> dict:
        """Register ``actor`` and log it in through the app's own routes.

        ``{"id", "email", "phone", "identity", "ordinal"}``, or ``{"error"}``. ``identity`` is the value
        it logs in by (its e-mail, else its username), which marks its own record in a response.
        """
        self.accounts += 1
        account: dict = {"ordinal": _ACCOUNT_ORDINAL + self.accounts}
        token, done, errors = None, set(), []
        for step in ("register", "login"):
            if step not in self.auth_routes:
                continue
            path, op = self.auth_routes[step]
            kind, body = self.credentials(op, actor, account["ordinal"])
            if kind is None:
                errors.append(f"POST {path}: {body}")
                continue
            status, answer = await self.api.call(actor, "POST", path, **{kind: body})
            if not 200 <= status < 300:
                errors.append(f"POST {path} answered {status}")
                continue
            done.add(step)
            token = _probe._find_token(answer) or token
            if account.get("id") is None:
                account["id"] = _probe._extract_id(answer)
            if step == "register":
                account["email"] = next((v for k, v in body.items() if "email" in k.lower()), None)
                account["phone"] = next((v for k, v in body.items() if "phone" in k.lower()), None)
            account["identity"] = account.get("email") or next(
                (v for k, v in body.items() if "password" not in k.lower() and isinstance(v, str)), None)
        if "register" not in done or ("login" in self.auth_routes and "login" not in done):
            return {"error": "; ".join(errors) or "no registration route"}
        if token:
            self.api.client(actor).headers["Authorization"] = f"Bearer {token}"
        return account

    async def account_id(self, actor: str, account: dict):
        """The account's id: from its register/login answer, else from the app's own account list."""
        if account.get("id") is not None or not account.get("identity") or not self.account_entity:
            return account.get("id")
        status, answer = await self.api.call(actor, "GET", self.routes[self.account_entity]["collection"])
        rows = answer if status == 200 and isinstance(answer, list) else []
        return next((row["id"] for row in rows if isinstance(row, dict)
                     and account["identity"] in row.values() and isinstance(row.get("id"), (int, str))), None)

    # -------------------------------------------------------------- phases

    async def create_records(self) -> None:
        """Account A creates one record of every entity it can, in dependency order."""
        import itertools

        # A's record of the account entity is its account, never a second row made through the generic create.
        pending = {entity: spec for entity, spec in sorted(self.creates.items())
                   if entity not in self.subjects and entity != self.account_entity}
        uncovered = self.report["uncovered"]
        progress = True
        while pending and progress:
            progress = False
            for entity, (path, schema_name) in list(pending.items()):
                schema = self.schemas[schema_name]
                if self.payload(entity, schema, self.subjects) is None:
                    continue
                del pending[entity]
                progress = True
                for variant in itertools.islice(_probe._variants(schema, self.schemas), 3):
                    self.ordinal += 1
                    body, _ = _probe._build_payload(
                        schema, self.schemas, self.orm[entity], self.subjects, self.subclasses, variant,
                        entity=entity, ordinal=self.ordinal, unique_fields=self.unique.get(entity, frozenset()))
                    status, answer = await self.api.call("a", "POST", path, json=body)
                    new_id = _probe._extract_id(answer) if 200 <= status < 300 else None
                    if new_id is not None:
                        self.subjects[entity] = new_id
                        self.created.append(entity)
                        uncovered.pop(entity, None)
                        break
                    uncovered[entity] = f"POST {path} answered {status}"
        for entity, (path, _schema) in pending.items():
            uncovered[entity] = f"POST {path} needs a record that could not be created"
        for entity in self.created:
            if entity not in self.order():
                uncovered[entity] = "its record cannot be read back by id"

    def as_b(self) -> dict:
        """Ids for a request by account B: its own records where it has them, account A's elsewhere."""
        return {**self.subjects, **{entity: value for entity, value in self.mine.items() if value is not None}}

    async def their_own(self, entity: str):
        """The id of a record of ``entity`` that account B created itself, plain; None when it cannot."""
        if entity not in self.mine and entity in self.creates:
            path, schema_name = self.creates[entity]
            body = self.payload(entity, self.schemas[schema_name], self.as_b())
            status, answer = (0, None) if body is None else await self.api.call("b", "POST", path, json=body)
            self.mine[entity] = _probe._extract_id(answer) if 200 <= status < 300 else None
        return self.mine.get(entity)

    async def exposures(self, account: dict) -> None:
        """A caller who is not logged in reads every GET route; note where account A's contact data, or a
        credential beside the value it logs in by, shows."""
        marks = {name: account[name] for name in ("email", "phone") if account.get(name)}
        if not account.get("identity"):
            return
        for path in sorted(self.paths):
            op = self.paths[path].get("get")
            url = self.url(path, self.entity_of(path))
            if not op or url is None or _SESSION_PATH.search(path):
                continue
            # Flags such as ``detailed`` widen the answer to nested objects.
            flags = {p["name"]: "true" for p in op.get("parameters", []) if p.get("in") == "query"
                     and _probe._resolve(p.get("schema") or {}, self.schemas).get("type") == "boolean"}
            status, answer = await self.api.call("anonymous", "GET", url, params=flags)
            if status != 200:
                continue
            text = answer if isinstance(answer, str) else json.dumps(answer, default=str)
            fields = [name for name, value in sorted(marks.items()) if value in text]
            credentials = sorted(_credentials_beside(answer, account["identity"], _probe._PROBE_PASSWORD))
            if fields or credentials:
                self.report["exposed"].append({"route": f"GET {path}", "fields": fields, "credentials": credentials})

    async def direct_updates(self, entity: str) -> str | None:
        """Account B sends each update method of account A's record; the first route that refused it.

        A reference to an account in B's body names B itself: an app may refuse a request made in another
        account's name, and that refusal is no rule about the record.
        """
        path, refused = self.routes[entity]["item"], None
        for method in self.update_methods(entity):
            body = self.update_body(entity, self.paths[path][method], self.as_b())
            if not body:
                continue
            seen = await _effect(self.api, self.reader(entity), "b", method.upper(), self.url(path, entity), body)
            changed = self.note(seen, _judge(seen, sent=body), entity=entity, method=method.upper(), path=path,
                                kind="update")
            if seen and not changed and seen["status"] in _HIDDEN and refused is None:
                refused = (method, seen["status"])
        return refused

    async def owner_rule(self, entity: str, refused) -> None:
        """Account A sends the update account B was refused: allowed, the refusal is a rule of the app."""
        if refused is None:
            return
        method, status = refused
        path = self.routes[entity]["item"]
        current = await self.reader(entity)()
        body = self.update_body(entity, self.paths[path][method], self.subjects)
        if not body or not isinstance(current, dict):
            return
        # A re-sends the values its record shows, so the control changes nothing its session depends on.
        body.update({key: current[key] for key in body if current.get(key) is not None})
        allowed, _ = await self.api.call("a", method.upper(), self.url(path, entity), json=body)
        if 200 <= allowed < 300:
            self.report["entities"].setdefault(entity, {})["rule"] = {
                "route": f"{method.upper()} {path}", "refused": status}

    async def sub_routes(self, entity: str) -> None:
        """Account B sends the routes under account A's record.

        A route whose tail names a relationship links another of A's records,
        and A reads the relationship back. Any other route counts only when it
        takes the record's own create or update body and A reads B's values
        back: an action with a body of its own (a rating, a vote) is what other
        accounts are meant to send.
        """
        item = self.routes[entity]["item"]
        for path in sorted(self.paths):
            if path == item or not path.startswith(item.rstrip("/") + "/"):
                continue
            tail = path[len(item.rstrip("/")) + 1:].strip("/").split("/")
            relation = self.orm.get(entity, {}).get(tail[0])
            for method in ("post", "put", "patch"):
                op = self.paths[path].get(method)
                if not op:
                    continue
                if relation and len(tail) == 2 and tail[1].startswith("{"):
                    await self.link_route(entity, path, method, tail[0], relation[0], op)
                    continue
                url = self.url(path, entity)
                if url is None or re.sub(r"(Create|Update|Patch)$", "", self.body_schema(op)[1]) != entity:
                    continue
                buildable, body = self.body(op, self.as_b())
                if not buildable:
                    continue
                seen = await _effect(self.api, self.reader(entity), "b", method.upper(), url, body)
                self.note(seen, _judge(seen, sent=body if isinstance(body, dict) else None),
                          entity=entity, method=method.upper(), path=path, kind="update")

    async def link_route(self, entity: str, path: str, method: str, relation: str, target: str, op: dict) -> None:
        """Account B links account A's ``target`` record to A's record of ``entity`` through ``path``.

        A's record, not one of B's own: attaching oneself or one's own record to another account's
        (a like, a vote, a membership) is what such routes are for.
        """
        members = path[:path.rindex("{")]
        other = self.subjects.get(target)
        if other is None or "get" not in (self.paths.get(members) or {}):
            return
        url = self.url(members, entity) + str(other) + ("/" if path.endswith("/") else "")
        buildable, body = self.body(op, self.subjects)

        async def linked() -> set | None:
            status, answer = await self.api.call("a", "GET", self.url(members, entity))
            return {str(record["id"]) for record in _scenarios._records(answer)} if status == 200 else None

        before = await linked()
        if before is None or not buildable or str(other) in before:
            return
        status, _ = await self.api.call("b", method.upper(), url, **({} if body is None else {"json": body}))
        after = await linked()
        if after is not None and str(other) in after:
            self.note({"status": status}, {"keys": []}, entity=entity, method=method.upper(), path=path,
                      kind="relation", relation=relation, target=target)

    async def side_doors(self) -> None:
        """Account B creates and updates other entities naming account A's record in a relationship
        field whose foreign key is stored on A's record."""
        for (holder, field), keys in sorted(self.child_keys.items()):
            target = self.orm[holder][field][0]
            if holder not in self.creates:
                continue
            path, schema_name = self.creates[holder]
            schema = self.schemas[schema_name]
            field_name = next((name for name in schema.get("properties", {})
                               if _probe._relationship_for(name, {field: True})), None)
            if field_name is None:
                continue
            for victim in sorted(self.subjects):
                item = self.routes[victim]["item"]
                if (victim != target and victim not in self.subclasses.get(target, ())) or not item:
                    continue
                read = self.reader(victim)
                ids = {**self.as_b(), target: self.subjects[victim]}
                found_on = {"entity": victim, "kind": "link", "field": field_name, "holder": holder}
                body = self.payload(holder, schema, ids, extra=field_name)
                if body is not None:
                    seen = await _effect(self.api, read, "b", "POST", path, body)
                    made = _probe._extract_id(seen["answer"]) if seen else None
                    self.note(seen, _judge(seen, linked=(made,), link_keys=tuple(keys)),
                              method="POST", path=path, **found_on)
                # The update is tried on a record account B owns: its own account, or one it creates plain.
                mine = await self.their_own(holder)
                for method in self.update_methods(holder):
                    update = self.schemas.get(self.body_schema(self.paths[self.routes[holder]["item"]][method])[1])
                    if mine is None or not update or field_name not in update.get("properties", {}):
                        continue
                    body = self.payload(holder, update, ids, extra=field_name)
                    if body is None:
                        continue
                    url = re.sub(r"\{[^/{}]+\}", str(mine), self.routes[holder]["item"], count=1)
                    seen = await _effect(self.api, read, "b", method.upper(), url, body)
                    self.note(seen, _judge(seen, linked=(mine,), link_keys=tuple(keys)),
                              method=method.upper(), path=self.routes[holder]["item"], **found_on)

    async def anonymous_writes(self, methods: tuple) -> None:
        """A caller who is not logged in sends every write route of ``methods`` a valid request.

        Only the status is seen here, so an accepted request is a note for the reader, never a defect.
        """
        public = {path for path, _ in self.auth_routes.values()}
        account_create = (self.routes.get(self.account_entity) or {}).get("collection")
        for path in sorted(self.paths):
            # As in the probe, a templated path such as /event/{event_id}/register is no account route.
            if path in public or _SESSION_PATH.search(path) or ("{" not in path and (
                    _probe._REGISTER_PATH.search(path) or _probe._LOGIN_PATH.search(path))):
                continue
            entity = self.entity_of(path)
            for method in methods:
                op = self.paths[path].get(method)
                # Creating an account is what registration does.
                if not op or (method == "post" and path == account_create):
                    continue
                url = self.url(path, entity)
                buildable, body = self.body(op, self.subjects)
                if url is None or not buildable:
                    continue
                status, _ = await self.api.call(
                    "anonymous", method.upper(), url, **({} if body is None else {"json": body}))
                if 200 <= status < 300:
                    self.report["anonymous"].append({"route": f"{method.upper()} {path}", "status": status})
                elif status in _REFUSED:
                    self.guarded.add(f"{method.upper()} {path}")

    async def stale_token(self) -> None:
        """An account deletes itself; its old token is then sent to routes that require login."""
        route = self.routes.get(self.account_entity) or {}
        if not route.get("item") or "delete" not in self.paths[route["item"]] or "login" not in self.auth_routes:
            return
        account = await self.open_account("c")
        account_id = None if "error" in account else await self.account_id("c", account)
        if account_id is None:
            return
        status, _ = await self.api.call("c", "DELETE", re.sub(r"\{[^/{}]+\}", str(account_id), route["item"]))
        if not 200 <= status < 300:
            return
        path, op = self.auth_routes["login"]
        kind, body = self.credentials(op, "c", account["ordinal"])
        status, _ = await self.api.call("c again", "POST", path, **{kind: body})
        if status not in _LOGIN_REFUSED:
            return  # the account still logs in, or the login route did not answer the question
        tried = 0
        for entity in self.created:
            path, schema_name = self.creates[entity]
            schema = self.schemas[schema_name]
            # A record that must name its owner would name a deleted account; that refusal proves nothing.
            names_account = any((_probe._relationship_for(field, self.orm[entity]) or ("",))[0] == self.account_entity
                                for field in schema.get("required", []))
            if tried >= _STALE_TOKEN_ROUTES or names_account or f"POST {path}" not in self.guarded:
                continue
            body = self.payload(entity, schema, self.subjects)
            if body is None:
                continue
            tried += 1
            status, _ = await self.api.call("c", "POST", path, json=body)
            if 200 <= status < 300:
                self.report["stale_token"].append({"route": f"POST {path}", "status": status})

    async def deletes(self, entity: str) -> None:
        """Account B deletes account A's record, singly and then in bulk; a refused single delete is a rule."""
        route, read = self.routes[entity], self.reader(entity)
        if "delete" in self.paths[route["item"]]:
            seen = await _effect(self.api, read, "b", "DELETE", self.url(route["item"], entity))
            deleted = self.note(seen, _judge(seen), entity=entity, method="DELETE", path=route["item"], kind="delete")
            if seen and not deleted and seen["status"] in _HIDDEN:
                self.report["entities"].setdefault(entity, {})["delete_rule"] = {
                    "route": f"DELETE {route['item']}", "refused": seen["status"]}
        if route["bulk"] and "delete" in self.paths[route["bulk"]]:
            seen = await _effect(self.api, read, "b", "DELETE", route["bulk"], [self.subjects[entity]])
            self.note(seen, _judge(seen), entity=entity, method="DELETE", path=route["bulk"], kind="delete")

    async def account_record(self) -> None:
        """Account B on account A's own account record, after everything else: a change there can end A's session."""
        entity = self.account_entity
        route = self.routes.get(entity) or {}
        if entity not in self.subjects or not route.get("item") or "get" not in self.paths[route["item"]]:
            return
        await self.owner_rule(entity, await self.direct_updates(entity))
        await self.sub_routes(entity)
        await self.deletes(entity)

    async def run(self) -> None:
        if "register" not in self.auth_routes:
            self.report.update(skipped="the app has no open registration route, so two accounts could not "
                                       "be opened", no_accounts=True)
            return
        try:
            tables = _probe._tables_behind(self.app, self.auth_routes["register"][0], self.orm)
        except _probe._Unknowable:
            tables = []
        self.account_entity = self.report["account_entity"] = tables[0] if len(tables) == 1 else None
        # An app may make its first account an administrator; neither A nor B is that one.
        await self.open_account("first")
        a, b = await self.open_account("a"), await self.open_account("b")
        failed = next((account["error"] for account in (a, b) if "error" in account), None)
        if failed:
            self.report["skipped"] = (f"two accounts could not be registered and logged in through the app's own "
                                      f"routes ({failed})")[:400]
            return
        a_id, b_id = await self.account_id("a", a), await self.account_id("b", b)
        if self.account_entity and a_id is not None and b_id is not None:
            self.subjects[self.account_entity] = a_id
            self.mine[self.account_entity] = b_id
        await self.create_records()
        await self.exposures(a)
        for entity in self.order():
            await self.owner_rule(entity, await self.direct_updates(entity))
        # Records with a rule first: a link made from the other end would already be there.
        for entity in sorted(self.order(), key=lambda name: "rule" not in self.report["entities"].get(name, {})):
            await self.sub_routes(entity)
        await self.side_doors()
        await self.anonymous_writes(("post", "put", "patch"))
        await self.stale_token()
        # Deleting comes after everything that reads the records, children first.
        for entity in sorted(self.order(), key=lambda name: -self.created.index(name)):
            await self.deletes(entity)
        await self.account_record()
        await self.anonymous_writes(("delete",))


def _check_cwd(started: float) -> dict:
    _probe._install_network_guard()
    sys.path.insert(0, os.getcwd())
    try:
        import httpx
        import main_api
        import sql_alchemy
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy.orm import configure_mappers

        configure_mappers()
        app = getattr(main_api, "app", None)
        if not hasattr(app, "openapi"):
            app = next(v for v in vars(main_api).values() if hasattr(v, "openapi") and hasattr(v, "routes"))
        facts = _describe(app, app.openapi(), sql_alchemy, sa_inspect)
    except BaseException as exc:  # the constructibility probe reports a boot failure; here the check did not run
        return {"boot": "error", "error": f"the app did not start: {type(exc).__name__}: {exc}"[:300]}

    import asyncio

    async def run() -> dict:
        api = _Api(httpx, app, started)
        check = _Check(api, facts)
        try:
            await check.run()
        except _Stop as stop:
            check.report["stopped"] = str(stop)
        finally:
            await api.close()
        return {**check.report, "requests": api.requests}

    try:
        return asyncio.run(run())
    except Exception as exc:
        return {"boot": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}


if __name__ == "__main__":
    print(_MARKER + json.dumps(_check_cwd(_STARTED), default=str))
