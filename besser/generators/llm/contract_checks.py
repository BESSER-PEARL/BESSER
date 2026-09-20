"""Deterministic data-contract checks derived from the domain model.

The domain model declares identifier attributes (``is_id``) with concrete
types. Generated code that disagrees with those types — an ``int`` path
param for a string id, ``parseInt()`` on a string id, a ForeignKey column
typed ``Integer`` pointing at a string PK — compiles fine and then breaks
at the first click. LLMs (especially the weak/free-tier models this
pipeline must serve) make exactly these mistakes, so we catch them
mechanically instead of hoping.

Two consumers share this module:

* ``tool_executor`` runs :func:`lint_file` on every ``write_file`` /
  ``modify_file`` and appends the findings to the tool result — the model
  sees the violation while the file is still hot in context (the
  cheapest possible feedback loop).
* The Phase 3 validator sweeps the workspace with the same checks and
  reports blocker-level findings to the auto-fix loop.

Design rule: **precision over recall**. A false blocker triggers billable
fix turns and can mark a good run incomplete, so every blocker pattern
here is one we observed in real generated output. Fuzzier signals are
demoted to advisory findings (``blocker=False``).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# Model type names that serialize to a string on the wire.
_STRING_TYPES = frozenset({"str", "string", "uuid"})
_INT_TYPES = frozenset({"int", "integer"})

# Surrogate fields the backend owns. A declared domain PK with another
# name (e.g. ``isbn``) is client-supplied and deliberately NOT listed.
SERVER_OWNED_FIELDS = ("id", "created_at", "updated_at", "createdAt", "updatedAt")

_FRONTEND_EXTS = (".js", ".jsx", ".ts", ".tsx")


@dataclass(frozen=True)
class Finding:
    """One contract violation in one file."""

    path: str
    line: int
    message: str
    blocker: bool


@dataclass(frozen=True)
class DataContract:
    """The id-type facts extracted once from the domain model.

    pk_types maps class name -> declared id attribute ``(name, type_name)``;
    classes with no declared id attribute are absent (they get a
    server-generated integer surrogate and every layer agrees by default).
    """

    pk_types: dict  # class name -> (attr name, type name)
    # class name -> {method name: declared parameter count}. A modelled
    # action's parameter list is its callable contract: a zero-parameter
    # action has to succeed on an empty body, because that is all the
    # frontend's method button ever sends.
    action_arity: dict = field(default_factory=dict)

    @property
    def string_id_classes(self) -> list:
        return sorted(
            cls for cls, (_, t) in self.pk_types.items()
            if t.lower() in _STRING_TYPES
        )

    @property
    def has_int_ids(self) -> bool:
        return any(t.lower() in _INT_TYPES for _, t in self.pk_types.values())


def build_data_contract(domain_model) -> DataContract | None:
    """Extract the PK contract from a BUML domain model.

    Returns None when there is no model / no classes — every check
    downstream then short-circuits to "nothing to enforce".
    """
    if domain_model is None:
        return None
    try:
        classes = list(domain_model.get_classes())
    except Exception:
        return None
    if not classes:
        return None

    pk_types: dict = {}
    action_arity: dict = {}
    for cls in classes:
        for attr in getattr(cls, "attributes", []) or []:
            if getattr(attr, "is_id", False):
                attr_type = getattr(attr, "type", None)
                type_name = getattr(attr_type, "name", "") or ""
                pk_types[cls.name] = (attr.name, type_name)
                break
        # Method names may carry the signature ("renew()"); the route uses
        # the bare name, same as the backend template's clean_method_name.
        arities = {
            str(method.name).split("(")[0].strip():
                len(getattr(method, "parameters", None) or [])
            for method in getattr(cls, "methods", None) or []
        }
        if arities:
            action_arity[cls.name] = arities
    return DataContract(pk_types=pk_types, action_arity=action_arity)


# ---------------------------------------------------------------------------
# File-level lint
# ---------------------------------------------------------------------------

# parseInt(x) / Number(x) where x is id-ish: exactly `id`, ends in Id/ID,
# or a member access ending in .id — e.g. parseInt(reservationId),
# Number(params.id). Anything else named `...id` by coincidence is rare
# enough in generated CRUD code that this stays high-precision.
_PARSE_ID_RE = re.compile(
    r"\b(?:parseInt|Number)\s*\(\s*"
    r"([A-Za-z_$][\w$]*(?:\.[\w$]+)*)"
    r"\s*[,)]"
)


def _is_idish(expr: str) -> bool:
    tail = expr.rsplit(".", 1)[-1]
    return tail == "id" or tail.endswith(("Id", "ID", "_id"))


# `something: 'POST'` / `.post(` sharing a line with `id:` — a create
# payload probably sends the id. Too fuzzy across lines: advisory only.
_POST_WITH_ID_RE = re.compile(r"(?:\.post\s*\(|method\s*:\s*['\"]POST['\"]).*\bid\s*:")

# Backend faking success for a method it never implemented.
_FAKE_EXECUTED_RE = re.compile(r"['\"]status['\"]\s*:\s*['\"]executed['\"]")

# `class FooCreate(...):` block capture for the server-owned-field scan.
_CREATE_SCHEMA_RE = re.compile(
    r"^class\s+\w*Create\w*\s*\(.*?\):\s*\n((?:[ \t]+.*\n?)*)",
    re.MULTILINE,
)
_SERVER_OWNED_FIELD_RE = re.compile(
    r"^[ \t]+(%s)\s*:" % "|".join(SERVER_OWNED_FIELDS),
    re.MULTILINE,
)


def lint_file(rel_path: str, content: str, contract: DataContract | None) -> list:
    """Run every applicable contract check on one file's content."""
    if contract is None:
        return []
    rel = rel_path.replace("\\", "/")
    low = rel.lower()
    if low.endswith(".py"):
        return _lint_python(rel, content, contract)
    if low.endswith(_FRONTEND_EXTS):
        return _lint_frontend(rel, content, contract)
    return []


def _line_of(content: str, pos: int) -> int:
    return content.count("\n", 0, pos) + 1


def _lint_frontend(rel: str, content: str, contract: DataContract) -> list:
    findings: list = []
    string_ids = contract.string_id_classes
    if string_ids:
        # When the model also declares int ids, a parseInt may be legit
        # for that entity — demote to advisory instead of blocking.
        as_blocker = not contract.has_int_ids
        for m in _PARSE_ID_RE.finditer(content):
            if not _is_idish(m.group(1)):
                continue
            findings.append(Finding(
                path=rel,
                line=_line_of(content, m.start()),
                message=(
                    f"{m.group(0).strip()} — the model declares string ids "
                    f"({', '.join(string_ids)}); ids must stay strings, "
                    "never parseInt()/Number() them"
                ),
                blocker=as_blocker,
            ))
    for m in _POST_WITH_ID_RE.finditer(content):
        findings.append(Finding(
            path=rel,
            line=_line_of(content, m.start()),
            message=(
                "create request appears to send an `id` — `id` is "
                "server-owned and must not be in create payloads"
            ),
            blocker=False,
        ))
    return findings


# relationship(..., secondary="<name>"): SQLAlchemy resolves the string on the
# first query, so a name matching no table passes every static check and 500s
# each request touching the class (live run 52befadf, 2026-09-18).
_RELATIONSHIP_SECONDARY_RE = re.compile(
    r"relationship\([^)]*?\bsecondary\s*=\s*['\"]([^'\"]+)['\"]", re.S
)
_TABLE_NAME_RE = re.compile(r"\bTable_?\(\s*['\"]([^'\"]+)['\"]")
_TABLE_VAR_RE = re.compile(r"^(\w+)\s*=\s*Table_?\(", re.M)
_TABLENAME_RE = re.compile(r"__tablename__\s*=\s*['\"]([^'\"]+)['\"]")


# ---------------------------------------------------------------------------
# Zero-parameter modelled actions must stay callable with no arguments
# ---------------------------------------------------------------------------

# The names generated handlers give the request body. Deliberately short:
# `request`/`req` are usually the Starlette Request, not a payload.
_BODY_NAMES = frozenset({"params", "body", "payload", "data", "request_body"})
# FastAPI injects these without the caller supplying anything.
_INJECTED_ANNOTATIONS = frozenset({
    "Request", "Response", "BackgroundTasks", "WebSocket", "SecurityScopes",
    "Session", "AsyncSession", "HTTPAuthorizationCredentials",
})
_HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete"})
_PATH_PARAM_RE = re.compile(r"\{([^}:]+)")


def _routes_of(function) -> list:
    routes = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        if decorator.func.attr not in _HTTP_VERBS or not decorator.args:
            continue
        route = decorator.args[0]
        if isinstance(route, ast.Constant) and isinstance(route.value, str):
            routes.append(route.value)
    return routes


def _modelled_action(contract: DataContract, route: str):
    """Resolve ``/loan/{loan_id}/methods/renew/`` to ``("Loan", "renew", 0)``.

    Returns None whenever the match is not unambiguous - a wrong attribution
    would be a false blocker on somebody else's operation.
    """
    if "/methods/" not in route:
        return None
    head, _, tail = route.partition("/methods/")
    action = tail.strip("/").split("/")[0]
    if not action:
        return None
    segments = [s for s in head.strip("/").split("/") if s and not s.startswith("{")]
    hint = segments[0].lower() if segments else ""
    by_name = [
        (cls, name, count)
        for cls, methods in contract.action_arity.items()
        for name, count in methods.items()
        if name.lower() == action.lower()
    ]
    scoped = [m for m in by_name if m[0].lower() == hint] if hint else by_name
    candidates = scoped or by_name
    return candidates[0] if len(candidates) == 1 else None


def _is_4xx_raise(node) -> bool:
    if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
        return False
    called = getattr(node.exc.func, "id", None) or getattr(node.exc.func, "attr", "")
    if called != "HTTPException":
        return False
    status = next((kw.value for kw in node.exc.keywords if kw.arg == "status_code"), None)
    if status is None and node.exc.args:
        status = node.exc.args[0]
    return isinstance(status, ast.Constant) and status.value in (400, 422)


def _reads_body(node) -> bool:
    """`params`, and the `(params or {})` / `params or {}` wrappers."""
    return any(isinstance(n, ast.Name) and n.id in _BODY_NAMES for n in ast.walk(node))


def _body_get_key(node) -> tuple | None:
    """``<body>.get('k'[, default])`` -> ``(key, has_default)``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and _reads_body(node.func.value) and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return None
    has_default = len(node.args) > 1 and not (
        isinstance(node.args[1], ast.Constant) and node.args[1].value is None
    )
    return node.args[0].value, has_default


def _required_body_keys(expression) -> list:
    """Keys an expression makes mandatory.

    ``body.get('a') or body.get('b')`` demands one of the two; the same chain
    ending in a real value (``body.get('days') or 14``) demands nothing, and
    so does ``body.get('days') if body else 14``. ``... else None`` is not a
    value - it is the miss that the next line turns into a 4xx.
    """
    read = _body_get_key(expression)
    if read:
        return [] if read[1] else [read[0]]
    if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.Or):
        branches = [expression.values[-1]]
        alternatives = expression.values
    elif isinstance(expression, ast.IfExp):
        branches = [expression.body, expression.orelse]
        alternatives = branches
    else:
        return []
    if any(isinstance(b, ast.Constant) and b.value is not None for b in branches):
        return []
    if not any(_required_body_keys(b) for b in branches):
        return []
    return [key for value in alternatives for key in _required_body_keys(value)]


def _absence_keys(test, bound: dict) -> list:
    """Keys whose ABSENCE makes this ``if`` test true.

    ``not value`` and ``value is None`` are the miss; ``if value:`` is the hit,
    and its ``else`` is somebody else's branch. Getting that backwards reads a
    handler that defaults the value on the miss as one that refuses it.
    """
    def referenced(node) -> list:
        if isinstance(node, ast.Name):
            return list(bound.get(node.id, []))
        return _required_body_keys(node)

    if isinstance(test, ast.BoolOp):
        return [key for value in test.values for key in _absence_keys(value, bound)]
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return referenced(test.operand)
    if isinstance(test, ast.Compare) and len(test.ops) == 1 \
            and isinstance(test.ops[0], (ast.Is, ast.Eq)) \
            and isinstance(test.comparators[0], ast.Constant) \
            and test.comparators[0].value is None:
        return referenced(test.left)
    return []


def _demanded_values(function, route: str) -> list:
    """Values this handler makes the caller supply, in source order."""
    path_params = set(_PATH_PARAM_RE.findall(route))
    demanded: list = []

    # A signature parameter with no default: FastAPI 422s an empty body.
    arguments = list(function.args.posonlyargs) + list(function.args.args)
    defaults = list(function.args.defaults)
    required = arguments[:len(arguments) - len(defaults)]
    required += [a for a, d in zip(function.args.kwonlyargs, function.args.kw_defaults)
                 if d is None]
    for argument in required:
        annotation = getattr(argument.annotation, "id", None) or getattr(
            argument.annotation, "attr", "")
        if argument.arg in path_params or argument.arg in ("self", "database", "db"):
            continue
        if annotation in _INJECTED_ANNOTATIONS:
            continue
        demanded.append(argument.arg)

    defaulted, bound = set(), {}
    for node in ast.walk(function):
        read = _body_get_key(node)
        if read and read[1]:
            defaulted.add(read[0])
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            from_body = _required_body_keys(node.value)
            if from_body:
                bound[node.targets[0].id] = from_body

    keys: list = []
    for node in ast.walk(function):
        # body['k'] - a missing key is a KeyError, i.e. a 500.
        if isinstance(node, ast.Subscript) and _reads_body(node.value) \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.append(node.slice.value)
        # `if not <key>: raise 4xx` - the key is required in all but name.
        # Only the branch taken when the key is missing counts.
        if isinstance(node, ast.If) and any(
                _is_4xx_raise(s) for stmt in node.body for s in ast.walk(stmt)):
            keys.extend(_absence_keys(node.test, bound))

    for key in keys:
        if key not in defaulted and key not in demanded:
            demanded.append(key)
    return demanded


def _zero_arg_action_issues(rel: str, content: str, contract: DataContract) -> list:
    if not contract.action_arity or "/methods/" not in content:
        return []
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return []  # a half-written file is python_source's problem, not ours
    findings: list = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for route in _routes_of(function):
            action = _modelled_action(contract, route)
            if action is None or action[2] != 0:
                continue
            demanded = _demanded_values(function, route)
            if not demanded:
                continue
            findings.append(Finding(
                path=rel,
                line=function.lineno,
                message=(
                    f"{action[0]}.{action[1]} requires "
                    f"`{'`, `'.join(demanded)}` from the caller, but the model "
                    f"declares {action[0]}.{action[1]}() with no parameters — the "
                    "method button sends an empty body, so this action is "
                    "uncallable. Choose a sensible value in the handler (an "
                    "optional override is fine) instead of demanding one"
                ),
                blocker=True,
            ))
            break
    return findings


def _lint_python(rel: str, content: str, contract: DataContract) -> list:
    findings: list = _zero_arg_action_issues(rel, content, contract)

    # The generated method endpoints legitimately answer "executed" after
    # actually running the modeled body (a ``_impl`` function call in the
    # same file; body-less methods raise 501 since the template fix). Only
    # an "executed" with no execution machinery anywhere in the file is
    # the fake-success facade.
    if "_impl(" not in content:
        for m in _FAKE_EXECUTED_RE.finditer(content):
            findings.append(Finding(
                path=rel,
                line=_line_of(content, m.start()),
                message=(
                    'fake success response {"status": "executed"} — an '
                    "unimplemented modeled method must return HTTP 501 "
                    "(Not Implemented), never pretend it ran"
                ),
                blocker=True,
            ))

    for schema in _CREATE_SCHEMA_RE.finditer(content):
        block = schema.group(1)
        for field in _SERVER_OWNED_FIELD_RE.finditer(block):
            findings.append(Finding(
                path=rel,
                line=_line_of(content, schema.start(1) + field.start()),
                message=(
                    f"create schema declares server-owned field "
                    f"`{field.group(1)}` — the backend assigns it; remove "
                    "it from the create schema and the create form"
                ),
                blocker=True,
            ))

    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    for cls in contract.string_id_classes:
        c_low = cls.lower()
        # foo_id: int  — an int-typed param/field for a string-id class.
        for m in re.finditer(rf"\b{re.escape(c_low)}_id\s*:\s*int\b", content):
            findings.append(Finding(
                path=rel,
                line=_line_of(content, m.start()),
                message=(
                    f"`{m.group(0)}` — {cls}.id is a string in the model; "
                    "declare the param/field as str"
                ),
                blocker=True,
            ))
        # Integer FK column pointing at a string PK.
        for m in re.finditer(rf"ForeignKey\(\s*['\"]{re.escape(c_low)}", content):
            line_start = content.rfind("\n", 0, m.start()) + 1
            line_end = content.find("\n", m.start())
            line_text = content[line_start:line_end if line_end != -1 else None]
            if "Integer" in line_text:
                findings.append(Finding(
                    path=rel,
                    line=_line_of(content, m.start()),
                    message=(
                        f"Integer ForeignKey to `{c_low}` — {cls}.id is a "
                        "string in the model; the FK column must be String"
                    ),
                    blocker=True,
                ))
        # routers/reservation.py declaring `id: int` for a string-id entity.
        if stem == c_low:
            for m in re.finditer(r"\bid\s*:\s*int\b", content):
                findings.append(Finding(
                    path=rel,
                    line=_line_of(content, m.start()),
                    message=(
                        f"`id: int` — {cls}.id is a string in the model; "
                        "path params and fields for it must be str"
                    ),
                    blocker=True,
                ))

    # Advisory, not blocker: the table may be defined in another module, and
    # Phase 3's import smoke check is the gate that proves the mapper fails.
    tables = set(_TABLE_NAME_RE.findall(content))
    tables.update(_TABLE_VAR_RE.findall(content))
    tables.update(_TABLENAME_RE.findall(content))
    for m in _RELATIONSHIP_SECONDARY_RE.finditer(content):
        if m.group(1) in tables:
            continue
        findings.append(Finding(
            path=rel,
            line=_line_of(content, m.start(1)),
            message=(
                f'relationship(secondary="{m.group(1)}") names no table '
                f"defined in this file (tables here: "
                f"{', '.join(sorted(tables)) or 'none'}); SQLAlchemy resolves "
                "it on the first query and every request touching the class "
                "will fail"
            ),
            blocker=False,
        ))

    return findings


def format_findings(findings: list, limit: int = 5) -> str:
    """One-line-per-finding block for tool results / issue lists."""
    lines = [
        f"{f.path} line {f.line}: {f.message}"
        for f in findings[:limit]
    ]
    if len(findings) > limit:
        lines.append(f"... and {len(findings) - limit} more")
    return "\n".join(lines)
