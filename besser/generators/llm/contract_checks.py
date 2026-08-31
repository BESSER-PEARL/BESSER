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

import re
from dataclasses import dataclass

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
    for cls in classes:
        for attr in getattr(cls, "attributes", []) or []:
            if getattr(attr, "is_id", False):
                attr_type = getattr(attr, "type", None)
                type_name = getattr(attr_type, "name", "") or ""
                pk_types[cls.name] = (attr.name, type_name)
                break
    return DataContract(pk_types=pk_types)


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


def _lint_python(rel: str, content: str, contract: DataContract) -> list:
    findings: list = []

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
