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
import os
import re
from dataclasses import dataclass, field
from besser.spec_driven_agent.parsed_source import parse_source

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
class EndOwnership:
    """Which class each association end is a property OF.

    UML puts an end typed ``X`` on the class at the OPPOSITE end, so a role
    named on the ``Booking`` end is a property of ``Employee`` — the same
    fact ``model_serializer`` already spells out as ``"owner"`` in the
    prompt. Telling the model does not settle it: a model can read
    ``"owner": "Employee"`` and still write ``booking.bookingsHandled``,
    which raises at runtime.

    Only binary associations are recorded (an n-ary end has no single
    opposite) and only roles that resolve to exactly one owner model-wide.
    """

    owner: dict        # role -> the class the role is a property of
    partner: dict      # role -> the role the class at the other end carries
    far: dict          # role -> the class the role is typed with
    to_one: frozenset  # roles with an upper bound of 1, so `a.role` IS an
                       # instance and `a.role.x` can be typed in turn
    members: dict      # class -> every member name the model gives it
    kin: dict          # class -> its ancestors and descendants


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
    # Association-end ownership; None when the model declares no usable
    # binary association, in which case the inverted-end check is a no-op.
    ends: EndOwnership | None = None

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
        # Count only parameters the CALLER must supply. A parameter with a
        # default is callable on an empty body, which is the only thing the
        # frontend's method button ever sends - so a method whose parameters
        # all default belongs under the zero-argument rule below.
        arities = {
            str(method.name).split("(")[0].strip():
                sum(1 for p in (getattr(method, "parameters", None) or [])
                    if getattr(p, "default_value", None) is None)
            for method in getattr(cls, "methods", None) or []
        }
        if arities:
            action_arity[cls.name] = arities
    return DataContract(
        pk_types=pk_types,
        action_arity=action_arity,
        ends=_build_end_ownership(domain_model, classes),
    )


def _model_member_names(cls) -> set:
    """Every name the model itself puts on ``cls`` (own + inherited)."""
    names: set = set()
    for attribute in getattr(cls, "attributes", None) or []:
        name = getattr(attribute, "name", None)
        if name:
            names.add(str(name))
    try:
        for attribute in cls.inherited_attributes() or []:
            names.add(str(attribute.name))
    except Exception:
        pass
    for method in getattr(cls, "methods", None) or []:
        name = getattr(method, "name", None)
        if name:
            names.add(str(name).split("(")[0].strip())
    return names


def _generalization_kin(domain_model, class_names: set) -> tuple:
    """(ancestors, kin) per class, transitively.

    An end owned by an ancestor is inherited, so accessing it on the child
    is correct; an end owned by a descendant is at worst a missing downcast,
    which is not the inversion this check is about. Both directions are
    therefore excused, which is what ``kin`` is for.
    """
    parents: dict = {name: set() for name in class_names}
    for gen in getattr(domain_model, "generalizations", None) or []:
        child = getattr(getattr(gen, "specific", None), "name", None)
        parent = getattr(getattr(gen, "general", None), "name", None)
        if child in parents and parent in parents:
            parents[child].add(parent)

    def ancestors(name: str) -> set:
        seen, stack = set(), list(parents.get(name, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(parents.get(current, ()))
        return seen

    up = {name: ancestors(name) for name in class_names}
    kin = {name: set(up[name]) for name in class_names}
    for name, elders in up.items():
        for elder in elders:
            kin[elder].add(name)
    return up, kin


def _build_end_ownership(domain_model, classes) -> EndOwnership | None:
    """Resolve every binary association end to the class that carries it."""
    # An association class is skipped as a RECEIVER: the scaffold gives it
    # link-navigation attributes named after the participant classes
    # (``ReservedRoom.bookings``, ``ReservedRoom.rooms``), which collide
    # with real role names and would read as inversions.
    plain = [c for c in classes if getattr(c, "association", None) is None]
    class_names = {c.name for c in plain}
    if not class_names:
        return None

    owners: dict = {}
    partners: dict = {}
    fars: dict = {}
    multiples: set = set()
    for assoc in getattr(domain_model, "associations", None) or []:
        ends = list(getattr(assoc, "ends", None) or [])
        if len(ends) != 2:
            continue  # n-ary: this end has no single opposite
        for end, other in ((ends[0], ends[1]), (ends[1], ends[0])):
            role = str(getattr(end, "name", "") or "")
            owner = getattr(getattr(other, "type", None), "name", None)
            far = getattr(getattr(end, "type", None), "name", None)
            if not role or not owner:
                continue
            owners.setdefault(role, set()).add(str(owner))
            partners.setdefault(role, set()).add(
                str(getattr(other, "name", "") or ""))
            if far:
                fars.setdefault(role, set()).add(str(far))
            if getattr(getattr(end, "multiplicity", None), "max", 2) != 1:
                multiples.add(role)
    if not owners:
        return None

    # A role name reused by two associations has no single owner; it is
    # dropped from the check but still counts as legitimate on each class
    # that does own it.
    owner = {r: next(iter(o)) for r, o in owners.items() if len(o) == 1}
    partner = {r: next(iter(p)) for r, p in partners.items()
               if r in owner and len(p) == 1 and next(iter(p))}
    far = {r: next(iter(f)) for r, f in fars.items()
           if r in owner and len(f) == 1}

    owned_roles: dict = {}
    for role, holders in owners.items():
        for holder in holders:
            owned_roles.setdefault(holder, set()).add(role)

    ancestors, kin = _generalization_kin(domain_model, class_names)
    members = {}
    for cls in plain:
        names = _model_member_names(cls) | owned_roles.get(cls.name, set())
        for ancestor in ancestors.get(cls.name, ()):
            names |= owned_roles.get(ancestor, set())
        members[cls.name] = frozenset(names)
    return EndOwnership(owner=owner, partner=partner, far=far,
                        to_one=frozenset(set(far) - multiples),
                        members=members, kin=kin)


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
# each request touching the class.
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


_FAILURE_KEYS = frozenset({"success", "succeeded", "ok", "result", "renewed"})


def _is_failure_return(node) -> bool:
    """``return {"success": False, ...}`` - a refusal dressed as HTTP 200.

    E.g. a handler answering the empty body with 200 and
    ``{"success": false, "message": "An extended dueDate is required"}``,
    leaving dueDate untouched. The button is just as dead as on a 422, and
    a status-code-only reading scores it as a pass.
    """
    if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
        return False
    for key, value in zip(node.value.keys, node.value.values):
        if isinstance(key, ast.Constant) and key.value in _FAILURE_KEYS \
                and isinstance(value, ast.Constant) and value.value is False:
            return True
    return False


def _refuses(statements) -> bool:
    """A branch that ends the call without doing the work."""
    return any(_is_4xx_raise(node) or _is_failure_return(node)
               for statement in statements for node in ast.walk(statement))


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
        # `if not <key>: raise 4xx` (or return success=False) - the key is
        # required in all but name. Only the missing-key branch counts.
        if isinstance(node, ast.If) and _refuses(node.body):
            keys.extend(_absence_keys(node.test, bound))

    for key in keys:
        if key not in defaulted and key not in demanded:
            demanded.append(key)
    return demanded


def _zero_arg_action_issues(rel: str, content: str, contract: DataContract) -> list:
    if not contract.action_arity or "/methods/" not in content:
        return []
    try:
        tree = parse_source(content)
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


# --- Nondeterministic business outcome -------------------------------------
#
# A handler that answers an action (e.g. `Order.confirmPayment`) with
# `random.choice([True, False])` / `random.random() < 0.8` is an
# unimplemented action: the coin flip is the whole handler, with no state
# behind it, and it passes a single probe 50-90% of the time.
#
# The check keys on the random draw being a BOOLEAN that decides a route
# handler's response, not on `random` being imported: random identifiers
# (`random.choices(string.ascii_uppercase, k=6)`) and demo-data seeders
# that randomise flags are legitimate.
_RANDOM_MODULES = frozenset({"random", "secrets"})
# Draws that are a coin flip once compared against anything.
_COMPARED_DRAWS = frozenset({"random", "uniform", "randint", "randrange",
                             "getrandbits", "randbelow"})
_SEQUENCE_DRAWS = frozenset({"choice", "choices", "sample"})


def _random_aliases(tree) -> frozenset:
    """Names pulled in by ``from random import choice`` and friends."""
    return frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in _RANDOM_MODULES
        for alias in node.names
    )


def _random_draw(node, aliases: frozenset):
    """The function name if ``node`` is a call into random/secrets."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
            and func.value.id in _RANDOM_MODULES:
        return func.attr
    if isinstance(func, ast.Name) and func.id in aliases:
        return func.id
    return None


def _is_coin_flip(node, aliases: frozenset) -> bool:
    """``node``'s VALUE is a random boolean - not merely random."""
    draw = _random_draw(node, aliases)
    # random.choice([True, False]) - every element a bool literal.
    if draw in _SEQUENCE_DRAWS and node.args:
        sequence = node.args[0]
        if isinstance(sequence, (ast.List, ast.Tuple)) and sequence.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, bool)
                for e in sequence.elts):
            return True
    # random.getrandbits(1) / secrets.randbelow(2) - binary by construction.
    if draw in ("getrandbits", "randbelow") and len(node.args) == 1 \
            and isinstance(node.args[0], ast.Constant) \
            and node.args[0].value in (1, 2):
        return True
    # random.random() < 0.8, randint(0, 1) == 1, ...
    if isinstance(node, ast.Compare):
        return any(_random_draw(part, aliases) in _COMPARED_DRAWS
                   for part in (node.left, *node.comparators))
    return False


def _coin_flip_outcome_issues(rel: str, content: str) -> list:
    """A route handler whose answer is decided by a random boolean."""
    if not any(word in content for word in ("random", "secrets")):
        return []
    try:
        tree = parse_source(content)
    except (SyntaxError, ValueError):
        return []  # a half-written file is python_source's problem, not ours
    aliases = _random_aliases(tree)
    findings: list = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _routes_of(function):
            continue  # seeders, factories and helpers are not the app's answer
        # Locals bound to a coin flip, and the draw that produced each.
        flips: dict = {}
        for node in ast.walk(function):
            targets, value = [], None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value:
                targets, value = [node.target], node.value
            if value is not None and _is_coin_flip(value, aliases):
                for target in targets:
                    if isinstance(target, ast.Name):
                        flips[target.id] = value
        decisive = _decisive_flip(function, flips, aliases)
        if decisive is None:
            continue
        try:
            snippet = ast.unparse(decisive)
        except Exception:  # pragma: no cover - unparse is total on parsed trees
            snippet = "a random draw"
        findings.append(Finding(
            path=rel,
            line=getattr(decisive, "lineno", function.lineno),
            message=(
                f"`{snippet}` decides what this endpoint answers — the "
                "outcome is a coin flip, so nothing can tell a working "
                "operation from a broken one, and the same request gives "
                "different answers. Decide it from state the app actually "
                "holds (the request, the modeled attributes, the database); "
                "if the model gives you nothing to decide it with, return "
                "HTTP 501 (Not Implemented). Randomness is fine for ids and "
                "seed data, never for whether an operation succeeded"
            ),
            blocker=True,
        ))
    return findings


def _decisive_flip(function, flips: dict, aliases: frozenset):
    """The coin flip that reaches the response, or None."""
    for node in ast.walk(function):
        # `return {"success": <flip>}` / `return <flip>`
        if isinstance(node, ast.Return) and node.value is not None:
            for sub in ast.walk(node.value):
                if _is_coin_flip(sub, aliases):
                    return sub
                if isinstance(sub, ast.Name) and sub.id in flips:
                    return flips[sub.id]
        # `if <flip>: return ... else: return ...`
        if isinstance(node, ast.If):
            branch = [*node.body, *node.orelse]
            if not any(isinstance(s, ast.Return)
                       for stmt in branch for s in ast.walk(stmt)):
                continue
            for sub in ast.walk(node.test):
                if _is_coin_flip(sub, aliases):
                    return sub
                if isinstance(sub, ast.Name) and sub.id in flips:
                    return flips[sub.id]
    return None


# --- Association end read off the wrong class ------------------------------
#
# E.g. `AttributeError: 'Booking' object has no attribute 'bookingsHandled'`
# where the scaffold declares `Booking.handledBy` / `Employee.bookingsHandled`
# and an LLM-authored method body inverts them. The prompt states the
# ownership per end (`"owner": "Employee"`), so this is enforcement, not
# instruction.
#
# Three guards, each against a known false positive:
#
# 1. The model's own member list excuses the name. An end owned by the
#    receiver or by an ancestor, an attribute, a method, and any role name
#    two associations share are all silent. This is what keeps
#    self-associations and reused roles out; association classes are
#    dropped as receivers outright, because the scaffold gives them
#    link-navigation attributes named after the participant classes
#    (``ReservedRoom.bookings``) that collide with real role names.
#
# 2. The APP's own declarations excuse the name. An app may rename the
#    scaffold's `Booking.guest` to `Booking.guests` and use that name
#    consistently; a rename the whole app agrees on is not a defect, so the
#    check reads every class body and every `Class.attr = ...` in the tree
#    first (plus `backref=` / `back_populates=`, which declare a member on
#    the class at the far end) and only fires on a name that resolves
#    NOWHERE. That is why this is a workspace sweep and not a per-file
#    lint: the access and the declaration are in different files.
#
# 3. The receiver's class must be named outright in the source — a
#    `query(X)` chain, a constructor, `session.get(X, ...)`, an annotation
#    or the enclosing class body. No name-shape guessing.

# Query chains that yield ONE instance. `.all()` yields a list, whose
# ELEMENTS are instances - handled separately for `for x in ....all()`.
_ONE_TERMINALS = frozenset({"first", "one", "one_or_none", "scalar"})


def _query_class(node, known: frozenset, terminals: frozenset) -> str | None:
    """The model class ``db.query(X).filter(...).<terminal>()`` yields."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in terminals:
        return None
    current = node.func.value
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.func.attr == "query" and len(current.args) == 1 \
                and isinstance(current.args[0], ast.Name) \
                and current.args[0].id in known:
            return current.args[0].id
        current = current.func.value
    return None


def _instance_class(node, known: frozenset) -> str | None:
    """The model class a single-instance expression evaluates to."""
    if not isinstance(node, ast.Call):
        return None
    # Booking(...)
    if isinstance(node.func, ast.Name) and node.func.id in known:
        return node.func.id
    # session.get(Booking, pk)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
            and node.args and isinstance(node.args[0], ast.Name) \
            and node.args[0].id in known:
        return node.args[0].id
    return _query_class(node, known, _ONE_TERMINALS)


def _local_instance_types(function, known: frozenset, enclosing: str | None) -> dict:
    """Locals in ``function`` whose class this file names outright."""
    types: dict = {}
    conflicted: set = set()

    def bind(name: str, cls: str | None) -> None:
        if not cls:
            return
        if types.get(name, cls) != cls:
            conflicted.add(name)
        types[name] = cls

    if enclosing in known:
        bind("self", enclosing)
    for argument in (list(function.args.posonlyargs) + list(function.args.args)
                     + list(function.args.kwonlyargs)):
        annotation = argument.annotation
        if isinstance(annotation, ast.Name) and annotation.id in known:
            bind(argument.arg, annotation.id)

    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            bind(node.targets[0].id, _instance_class(node.value, known))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.annotation, ast.Name) and node.annotation.id in known:
                bind(node.target.id, node.annotation.id)
            elif node.value is not None:
                bind(node.target.id, _instance_class(node.value, known))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            bind(node.target.id,
                 _query_class(node.iter, known, frozenset({"all"})))
    for name in conflicted:
        types.pop(name, None)
    return types


def _inverted_end_message(rel: str, line: int, what: str, receiver: str,
                          role: str, ends: EndOwnership, blocker: bool) -> str:
    owner = ends.owner[role]
    carries = ""
    if ends.far.get(role) == receiver and ends.partner.get(role):
        carries = f"; `{receiver}` carries `{ends.partner[role]}`"
    prefix = "data contract:" if blocker else "data contract (advisory):"
    return (
        f"{prefix} {rel} line {line}: {what}, and nothing in the app "
        f"declares `{receiver}.{role}` — `{role}` is a property of `{owner}`, "
        f"not `{receiver}`{carries}. An association end named on one class is "
        f"a property of the class at the OPPOSITE end; the model states this "
        f"as \"owner\": \"{owner}\". Navigate it from `{owner}`, or query the "
        f"other side by its foreign key"
    )


def _class_scopes(tree) -> dict:
    """Function node -> the name of the class body it is defined in."""
    scopes: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes[child] = node.name
    return scopes


def _declared_members(tree, known: frozenset, declared: dict) -> None:
    """Accumulate every name this module attaches to a model class.

    Generous on purpose: anything that plausibly puts the name on the class
    counts, because a name that resolves is not the inversion we are after.
    """
    def note(cls, name) -> None:
        if cls in known and name:
            declared.setdefault(cls, set()).add(str(name))

    def relationship_target(call):
        """The class a ``relationship("Guest", ...)`` call points at."""
        if not call.args:
            return None
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value.rsplit(".", 1)[-1]
        if isinstance(first, ast.Name):
            return first.id
        return None

    def note_reverse(cls, value) -> None:
        # ``backref=``/``back_populates=`` name a member of the class at the
        # OTHER end - the scaffold's own way of declaring the opposite side.
        if not isinstance(value, ast.Call):
            return
        called = getattr(value.func, "id", None) or getattr(value.func, "attr", "")
        if called != "relationship":
            return
        far = relationship_target(value)
        for keyword in value.keywords:
            if keyword.arg in ("backref", "back_populates") \
                    and isinstance(keyword.value, ast.Constant) \
                    and isinstance(keyword.value.value, str):
                note(far, keyword.value.value)

    for node in ast.walk(tree):
        # `Booking.handledBy = relationship(...)` / `Booking.handledBy: T = ...`
        target, value = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            note(target.value.id, target.attr)
            note_reverse(target.value.id, value)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                note(node.name, child.name)
                # `self.x = ...` in __init__ and friends.
                for sub in ast.walk(child):
                    if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                        for target in (sub.targets if isinstance(sub, ast.Assign)
                                       else [sub.target]):
                            if isinstance(target, ast.Attribute) \
                                    and isinstance(target.value, ast.Name) \
                                    and target.value.id == "self":
                                note(node.name, target.attr)
            elif isinstance(child, ast.Assign):
                for sub in child.targets:
                    if isinstance(sub, ast.Name):
                        note(node.name, sub.id)
                note_reverse(node.name, child.value)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                note(node.name, child.target.id)
                note_reverse(node.name, child.value)
        if node.name in known:
            # A subclass gets whatever its in-app bases declare.
            entry = declared.setdefault(node.name, set())
            for base in node.bases:
                if isinstance(base, ast.Name):
                    entry.add(f"<base>{base.id}")


def _resolve_bases(declared: dict) -> dict:
    """Fold in-app base-class members into each subclass, transitively."""
    resolved: dict = {}

    def members(name: str, seen: frozenset) -> set:
        if name in resolved:
            return resolved[name]
        own = declared.get(name, set())
        out = {n for n in own if not n.startswith("<base>")}
        for entry in own:
            if entry.startswith("<base>"):
                base = entry[6:]
                if base in declared and base not in seen:
                    out |= members(base, seen | {name})
        if not seen:
            resolved[name] = out
        return out

    return {name: members(name, frozenset()) for name in declared}


def _walk_python(app_dir: str, known: frozenset) -> tuple:
    """Every parsed .py in the delivered tree, plus what it declares.

    ``(rel, tree, source)`` per file and ``class -> member names`` across
    all of them: the read and the declaration that would excuse it live in
    different files, so both sweeps need the whole workspace at once.
    """
    parsed: list = []
    declared: dict = {}
    for root, dirs, names in os.walk(app_dir):
        dirs[:] = [d for d in dirs
                   if d not in ("node_modules", "dist", "build", "__pycache__")]
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, app_dir).replace("\\", "/")
            if rel.startswith(".besser_"):
                continue
            try:
                if os.path.getsize(path) > 1_000_000:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            try:
                tree = parse_source(content)
            except (SyntaxError, ValueError):
                continue  # a half-written file is python_source's problem
            _declared_members(tree, known, declared)
            parsed.append((rel, tree, content))
    return parsed, _resolve_bases(declared)


def collect_inverted_end_issues(app_dir: str, contract: DataContract | None) -> list:
    """Association ends the delivered app reads off the wrong class.

    A workspace sweep, not a per-file lint: the access and the declaration
    that would excuse it live in different files. Returns ``data contract:``
    strings, ready for the Phase 3 issue list.
    """
    if contract is None or contract.ends is None:
        return []
    ends = contract.ends
    known = frozenset(ends.members)
    roles = tuple(ends.owner)

    files, app_members = _walk_python(app_dir, known)
    parsed = [(rel, tree) for rel, tree, content in files
              if any(role in content for role in roles)]
    if not parsed:
        return []

    def misplaced(receiver: str, role: str) -> bool:
        owner = ends.owner.get(role)
        return bool(
            owner and owner != receiver
            and receiver in ends.members
            and role not in ends.members[receiver]
            and role not in app_members.get(receiver, ())
            and owner not in ends.kin.get(receiver, ())
        )

    # (path, line, message): ast.walk order is not source order, and the
    # issue list is compared between runs.
    issues: list = []
    for rel, tree in parsed:
        scopes = _class_scopes(tree)
        seen: set = set()
        # `Booking(guests=[...])` - the constructor rejects the keyword
        # before any attribute is ever read.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            receiver = node.func.id
            if receiver not in known:
                continue
            for keyword in node.keywords:
                if keyword.arg and misplaced(receiver, keyword.arg):
                    issues.append(((rel, node.lineno, keyword.arg),
                                   _inverted_end_message(
                                       rel, node.lineno,
                                       f"`{receiver}(...)` is constructed with "
                                       f"`{keyword.arg}=`",
                                       receiver, keyword.arg, ends, True)))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            types = _local_instance_types(function, known, scopes.get(function))
            if not types:
                continue

            def holder(value) -> tuple:
                """(class, hops) for ``value``, following to-one ends.

                ``db_order.handledBy.warehouse`` needs the hop: the receiver
                of the bad read is an expression, not a variable. ``hops``
                is what decides severity - see below.
                """
                if isinstance(value, ast.Name):
                    return types.get(value.id), 0  # noqa: B023 - same iteration
                if not isinstance(value, ast.Attribute):
                    return None, 0
                base, hops = holder(value.value)
                if base and ends.owner.get(value.attr) == base \
                        and value.attr in ends.to_one:
                    return ends.far.get(value.attr), hops + 1
                return None, 0

            for node in ast.walk(function):
                if not isinstance(node, ast.Attribute):
                    continue
                receiver, hops = holder(node.value)
                if not receiver or not misplaced(receiver, node.attr):
                    continue
                key = (node.lineno, node.col_offset, receiver, node.attr)
                if key in seen:
                    continue  # a nested function is walked by its parent too
                seen.add(key)
                try:
                    shown = ast.unparse(node.value)
                except Exception:  # pragma: no cover - total on parsed trees
                    shown = receiver
                # A direct receiver is a blocker. Through a hop the read is
                # just as wrong but often sits on a path nothing exercises,
                # so it is reported as advisory and does not spend fix turns.
                issues.append(((rel, node.lineno, node.attr),
                               _inverted_end_message(
                                   rel, node.lineno,
                                   f"`{shown}` is a `{receiver}` here, so "
                                   f"`{shown}.{node.attr}` raises AttributeError",
                                   receiver, node.attr, ends, hops == 0)))
    return [message for _, message in sorted(issues)]


# An instance answers these without the app declaring them: SQLAlchemy's
# declarative base and Pydantic's BaseModel both contribute a public API,
# and one domain class name is frequently reused for both halves.
_INHERITED_MEMBERS = frozenset({
    "metadata", "registry", "awaitable_attrs",
    "dict", "json", "copy", "schema", "construct", "parse_obj", "parse_raw",
    "model_dump", "model_dump_json", "model_copy", "model_validate",
    "model_fields", "model_fields_set", "model_config", "model_extra",
    "model_rebuild", "model_json_schema",
})

_INJECTING_CALLS = frozenset({"Depends", "Security"})


def _injected_parameters(function) -> set:
    """Parameters FastAPI fills in.

    Their annotation names a dependency, not a domain instance. When the
    model happens to contain a class called ``Session``, ``database:
    Session = Depends(get_db)`` otherwise types every ``database.query``
    in the file as a read on that class.
    """
    args = function.args
    positional = list(args.posonlyargs) + list(args.args)
    pairs = list(zip(positional[len(positional) - len(args.defaults):],
                     args.defaults))
    pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d]
    return {argument.arg for argument, default in pairs
            if isinstance(default, ast.Call)
            and (getattr(default.func, "id", None)
                 or getattr(default.func, "attr", "")) in _INJECTING_CALLS}


def _shadowing_imports(tree, app_modules: frozenset) -> set:
    """Class names this module binds to something from OUTSIDE the app.

    ``from sql_alchemy import Clerk`` is the domain class itself; ``from
    sqlalchemy.orm import Session`` is not, however the model spells its
    own ``Session``. Only the second kind shadows.
    """
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] in app_modules:
                continue
        elif not isinstance(node, ast.Import):
            continue
        out.update(alias.asname or alias.name.split(".")[0]
                   for alias in node.names)
    return out


def _probed_attributes(function) -> set:
    """Attribute names the author checks with hasattr/getattr first."""
    return {node.args[1].value
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("hasattr", "getattr") and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)}


def _conditional_nodes(function) -> set:
    """Nodes that run only in some states, not on every call.

    Per child, not per statement: an ``if`` TEST evaluates whenever control
    reaches it and only the branches do not, ``for`` evaluates its iterable
    but may never enter the body, ``try`` always runs its body, and ``a and
    b`` always evaluates ``a``. Marking a whole statement conditional would
    demote reads that always run, such as the test of
    ``if db_clerk.warehouse_id is None``.
    """
    conditional: set = set()

    def always(node) -> list:
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            return [node.test]
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return [node.iter]
        if isinstance(node, ast.Try):
            return list(node.body)
        if isinstance(node, ast.BoolOp):
            return node.values[:1]
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp,
                             ast.DictComp)):
            return [node.generators[0].iter] if node.generators else []
        if isinstance(node, ast.Match):
            return [node.subject]
        return list(ast.iter_child_nodes(node))

    def walk(node, under: bool) -> None:
        if under:
            conditional.add(node)
        unconditional = {id(child) for child in always(node)}
        for child in ast.iter_child_nodes(node):
            walk(child, under or id(child) not in unconditional)

    for child in ast.iter_child_nodes(function):
        walk(child, False)
    return conditional


def collect_undeclared_attribute_issues(app_dir: str,
                                        contract: DataContract | None) -> list:
    """Attributes read off a model instance that nothing declares.

    The sibling of :func:`collect_inverted_end_issues`: same receiver
    typing, same "does anything declare it" test, but for any member name
    rather than only association roles. Roles stay that function's, so the
    two never report one read twice.

    This is a leading runtime-crash class in generated apps: a
    column named on the wrong class (``db_clerk.warehouse_id`` when
    ``warehouse_id`` is Product's), an audit field the scaffold never
    wrote (``db_loan.created_at``), a value the model invented
    (``db_warehouse.total_stock``). Each is an ``AttributeError`` - or a
    ``TypeError: invalid keyword argument`` when the same wrong name
    reaches the constructor - on the first request that runs the line.

    Only reads that execute on EVERY call of their function are reported.
    A read inside a branch is just as wrong, but the runtime probe cannot
    always reach it, and a blocker on a path nothing exercises spends fix
    turns on an app that works (e.g. a real ``Loan.returnDate`` crash
    behind ``if status == RETURNED``).
    """
    if contract is None or contract.ends is None:
        return []
    ends = contract.ends
    known = frozenset(ends.members)
    files, app_members = _walk_python(app_dir, known)
    if not files:
        return []

    app_modules = frozenset(
        rel.rsplit("/", 1)[-1][:-3] for rel, _tree, _content in files)

    issues: list = []
    for rel, tree, _content in files:
        scopes = _class_scopes(tree)
        shadowed = _shadowing_imports(tree, app_modules) & known
        seen: set = set()
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            injected = _injected_parameters(function)
            types = {name: cls for name, cls
                     in _local_instance_types(function, known,
                                              scopes.get(function)).items()
                     if cls not in shadowed and name not in injected}
            if not types:
                continue
            probed = _probed_attributes(function)
            conditional = _conditional_nodes(function)
            for node in ast.walk(function):
                # `Loan(processingFee=...)` is the same defect through the
                # constructor: SQLAlchemy answers it `TypeError: invalid
                # keyword argument` before any attribute is read.
                if isinstance(node, ast.Call)                         and isinstance(node.func, ast.Name)                         and node.func.id in ends.members                         and node.func.id not in shadowed                         and node not in conditional:
                    built = node.func.id
                    for keyword in node.keywords:
                        name = keyword.arg
                        if not name or name.startswith("_")                                 or name in ends.owner                                 or name in ends.members[built]                                 or name in app_members.get(built, ()):
                            continue
                        issues.append(((rel, node.lineno, name), (
                            f"data contract: {rel} line {node.lineno}: "
                            f"`{built}(...)` is constructed with `{name}=`, and "
                            f"nothing in the model or the app declares "
                            f"`{built}.{name}` - this line runs on every call "
                            f"and raises `TypeError: '{name}' is an invalid "
                            f"keyword argument for {built}`. Pass a member "
                            f"`{built}` actually has, or declare `{name}` "
                            f"on it")))
                if not isinstance(node, ast.Attribute):
                    continue
                if not isinstance(node.ctx, ast.Load):
                    continue  # a write binds the name, it never raises
                if not isinstance(node.value, ast.Name):
                    continue  # a hop is the inverted-end check's territory
                receiver = types.get(node.value.id)
                attribute = node.attr
                if not receiver or receiver not in ends.members:
                    continue
                if attribute.startswith("_") or attribute in _INHERITED_MEMBERS:
                    continue
                if attribute in probed or attribute in ends.owner:
                    continue
                if attribute in ends.members[receiver] \
                        or attribute in app_members.get(receiver, ()):
                    continue
                if node in conditional:
                    continue
                key = (node.lineno, node.col_offset, receiver, attribute)
                if key in seen:
                    continue  # a nested function is walked by its parent too
                seen.add(key)
                issues.append(((rel, node.lineno, attribute), (
                    f"data contract: {rel} line {node.lineno}: "
                    f"`{node.value.id}` is a `{receiver}` here, and nothing in "
                    f"the model or the app declares `{receiver}.{attribute}` - "
                    f"this line runs on every call and raises `AttributeError: "
                    f"'{receiver}' object has no attribute '{attribute}'`. "
                    f"Read a member `{receiver}` actually has, or declare "
                    f"`{attribute}` on it")))
    return [message for _, message in sorted(issues)]


def _lint_python(rel: str, content: str, contract: DataContract) -> list:
    findings: list = _zero_arg_action_issues(rel, content, contract)
    findings += _coin_flip_outcome_issues(rel, content)

    # The generated method endpoints legitimately answer "executed" after
    # actually running the modeled body (a ``_impl`` function call in the
    # same file; body-less methods raise 501). Only
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
