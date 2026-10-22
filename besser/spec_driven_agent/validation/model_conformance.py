"""Does the FastAPI backend follow the model change of a modify run?

Deterministic, no LLM (#659). After Phase 2 of a modify run whose class model
changed, the generated backend's ``sql_alchemy.py`` (ORM classes) and
``pydantic_classes.py`` (``<Class>Create`` schemas) are parsed with ``ast`` and
checked against the changed attributes only:

* a removed (or renamed-away) attribute still defined on its class;
* an added (or renamed-to) attribute missing;
* a changed primitive type still declared as the OLD type;
* a changed optionality the code still contradicts (ORM: now optional but
  NOT NULL; schema: now optional but required, or now required but Optional).

Only unambiguous findings: classes missing from a file, non-primitive types,
relationships, inherited definitions and other stacks are skipped silently.
Findings start with ``model conformance:`` and are Phase 3 blockers.
"""

from __future__ import annotations

import ast
import os
import re

from besser.spec_driven_agent.execution.workspace_fs import read_plain_text, walk_plain
from besser.spec_driven_agent.planning.model_change import pair_elements
from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR

_ORM_FILE = "sql_alchemy.py"
_SCHEMA_FILE = "pydantic_classes.py"
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", _SNAPSHOT_DIR, ".venv", "venv"}

# Python annotation / SQLAlchemy column type -> B-UML primitive type.
_ANNOTATION_TYPES = {
    "str": "str", "int": "int", "float": "float", "bool": "bool",
    "date": "date", "dt_date": "date", "datetime": "datetime", "dt_datetime": "datetime",
    "time": "time", "dt_time": "time", "timedelta": "timedelta", "dt_timedelta": "timedelta",
}
_COLUMN_TYPES = {
    "String": "str", "Text": "str", "Unicode": "str", "UnicodeText": "str", "VARCHAR": "str",
    "Integer": "int", "BigInteger": "int", "SmallInteger": "int",
    "Float": "float", "Double": "float", "Boolean": "bool",
    "Date": "date", "DateTime": "datetime", "Time": "time", "Interval": "timedelta",
}
_COLUMN_CALLS = {"mapped_column", "Column", "column_property"}
_SERVER_OWNED = {"id", "createdat", "updatedat"}


def _name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return None


def _bare(name: str | None) -> str | None:
    """``Mapped_`` / ``Optional_`` / ``Integer_`` are the generator's aliases."""
    return name[:-1] if name and name.endswith("_") and len(name) > 1 else name


def _unwrap_optional(node) -> tuple[object, bool]:
    """``(inner annotation, is optional)`` for ``Optional[X]``, ``Union[X, None]``, ``X | None``."""
    if isinstance(node, ast.Subscript) and _bare(_name(node.value)) == "Optional":
        return node.slice, True
    if isinstance(node, ast.Subscript) and _bare(_name(node.value)) == "Union":
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        rest = [i for i in items if not (isinstance(i, ast.Constant) and i.value is None)]
        if len(rest) == 1 and len(rest) < len(items):
            return rest[0], True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for keep, drop in ((node.left, node.right), (node.right, node.left)):
            if isinstance(drop, ast.Constant) and drop.value is None:
                return keep, True
    return node, False


class _Field:
    def __init__(self, annotation, value):
        self.annotation = annotation
        self.value = value

    @property
    def is_relationship(self) -> bool:
        return isinstance(self.value, ast.Call) and _name(self.value.func) == "relationship"

    def mapped(self) -> tuple[object, bool]:
        """The ORM annotation inside ``Mapped[...]`` and whether it is Optional."""
        node = self.annotation
        if isinstance(node, ast.Subscript) and _bare(_name(node.value)) == "Mapped":
            node = node.slice
        return _unwrap_optional(node) if node is not None else (None, False)

    def column_call(self):
        call = self.value
        if isinstance(call, ast.Call) and _bare(_name(call.func)) in _COLUMN_CALLS:
            return call
        return None


class _Module:
    """Classes of one parsed file: their fields, methods and base class names."""

    def __init__(self, tree: ast.Module):
        self.fields: dict[str, dict[str, _Field]] = {}
        self.methods: dict[str, set[str]] = {}
        self.bases: dict[str, list[str]] = {}
        self.outside: dict[str, set[str]] = {}  # ``Class.attr = ...`` at module level
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                fields, methods = {}, set()
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields[item.target.id] = _Field(item.annotation, item.value)
                    elif (isinstance(item, ast.Assign) and len(item.targets) == 1
                          and isinstance(item.targets[0], ast.Name)):
                        fields[item.targets[0].id] = _Field(None, item.value)
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
                self.fields[node.name] = fields
                self.methods[node.name] = methods
                self.bases[node.name] = [b for b in (_name(base) for base in node.bases) if b]
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    self.outside.setdefault(target.value.id, set()).add(target.attr)

    def lineage(self, cls: str) -> list[str]:
        """``cls`` and its ancestors defined in this file."""
        seen, todo = [], [cls]
        while todo:
            name = todo.pop(0)
            if name in self.fields and name not in seen:
                seen.append(name)
                todo.extend(self.bases.get(name, []))
        return seen

    def defines(self, cls: str, attr: str) -> bool:
        """``attr`` reachable on ``cls`` in any form: field, method, inherited, set outside."""
        return any(attr in self.fields[c] or attr in self.methods[c]
                   or attr in self.outside.get(c, ()) for c in self.lineage(cls))


def _parse(path: str, workspace: str) -> _Module | None:
    text = read_plain_text(path, root=workspace)
    if text is None:
        return None
    try:
        return _Module(ast.parse(text))
    except SyntaxError:
        return None  # the syntax check reports it


def _files(workspace: str, name: str) -> list[str]:
    found = []
    for root, dirs, files in walk_plain(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if name in files:
            found.append(os.path.join(root, name))
    return sorted(found)


def _end_names(summary: dict) -> dict[str, set[str]]:
    """Association end names per owning class (``Owner.end[..]``)."""
    out: dict[str, set[str]] = {}
    for entry in (summary.get("associations") or {}).values():
        for owner, end in re.findall(r"(\w+)\.(\w+)\[", str(entry.get("ends", ""))):
            out.setdefault(owner, set()).add(end)
    return out


def _attribute_checks(base: dict, new: dict):
    """Yield ``(class, rule, attribute, old entry, new entry, detail)`` per changed attribute.

    Rules: ``removed`` (old entry, detail = new name when renamed), ``added``,
    ``type`` and ``optional``. Only classes present in both models.
    """
    old_c, new_c = base.get("classes") or {}, new.get("classes") or {}
    ends = _end_names(new)
    for old_cls, cls in pair_elements(old_c, new_c):
        before = old_c[old_cls].get("attributes") or {}
        after = new_c[cls].get("attributes") or {}
        pairs = pair_elements(before, after)
        renamed = {o: n for o, n in pairs if o != n}
        kept = {n for o, n in pairs if o == n}
        in_use = set(after) | ends.get(cls, set())
        for name in sorted(set(before) - {o for o, _ in pairs} | set(renamed)):
            if name not in in_use:
                yield cls, "removed", name, before[name], None, renamed.get(name)
        for name in sorted(set(after) - kept):
            yield cls, "added", name, None, after[name], None
        for old, cur in pairs:
            b, a = before[old], after[cur]
            if b.get("type") != a.get("type") and b.get("type") and a.get("type"):
                yield cls, "type", cur, b, a, None
            if "optional" in b and "optional" in a and b["optional"] != a["optional"]:
                yield cls, "optional", cur, b, a, None


def _orm_findings(module: _Module, rel: str, checks) -> list[str]:
    issues = []
    for cls, rule, attr, old, new, renamed in checks:
        if cls not in module.fields:
            continue
        field = module.fields[cls].get(attr)
        if rule == "removed":
            if field is not None and not field.is_relationship:
                how = f"renamed to {cls}.{renamed}" if renamed else "removed"
                issues.append(
                    f"model conformance: {cls}.{attr} was {how} in the class diagram, but "
                    f"{rel} still defines {attr} on {cls}. Update the database model, the API "
                    "schemas and the UI to the current model.")
        elif rule == "added":
            if not module.defines(cls, attr):
                issues.append(
                    f"model conformance: {cls}.{attr} was added to the class diagram, but {rel} "
                    f"does not define it on {cls}. Add the column, then the API schemas and the UI.")
        elif field is not None and not field.is_relationship:
            inner, optional = field.mapped()
            call = field.column_call()
            if rule == "type":
                was, now = old["type"], new["type"]
                if was not in _COLUMN_TYPES.values() or now not in _COLUMN_TYPES.values():
                    continue
                declared = _ANNOTATION_TYPES.get(_bare(_name(inner)) or "")
                column = (_COLUMN_TYPES.get(_bare(_name(call.args[0])) or "")
                          if call is not None and call.args else None)
                if was in (declared, column):
                    issues.append(
                        f"model conformance: {cls}.{attr} changed type {was} -> {now} in the "
                        f"class diagram, but {rel} still declares it as {was} on {cls}.")
            elif rule == "optional" and new["optional"] and call is not None:
                nullable = any(k.arg == "nullable" and isinstance(k.value, ast.Constant)
                               and k.value.value is True for k in call.keywords)
                primary = any(k.arg == "primary_key" for k in call.keywords)
                if not (optional or nullable or primary):
                    issues.append(
                        f"model conformance: {cls}.{attr} is now optional in the class diagram, "
                        f"but {rel} still declares it NOT NULL on {cls} (use Optional and "
                        "nullable=True).")
    return issues


def _schema_findings(module: _Module, rel: str, checks) -> list[str]:
    issues = []
    for cls, rule, attr, old, new, renamed in checks:
        schema = f"{cls}Create"
        if schema not in module.fields:
            continue
        field = module.fields[schema].get(attr)
        if rule == "removed":
            if field is not None:
                how = f"renamed to {cls}.{renamed}" if renamed else "removed"
                issues.append(
                    f"model conformance: {cls}.{attr} was {how} in the class diagram, but "
                    f"{rel} still has the field {attr} in {schema}.")
        elif rule == "added":
            server_owned = new.get("derived") or attr.lower().replace("_", "") in _SERVER_OWNED
            if not server_owned and not module.defines(schema, attr):
                issues.append(
                    f"model conformance: {cls}.{attr} was added to the class diagram, but "
                    f"{rel} has no field {attr} in {schema}, so the API cannot set it.")
        elif field is not None and field.annotation is not None and not new.get("derived"):
            inner, optional = _unwrap_optional(field.annotation)
            if rule == "type":
                was, now = old["type"], new["type"]
                if (was in _COLUMN_TYPES.values() and now in _COLUMN_TYPES.values()
                        and _ANNOTATION_TYPES.get(_name(inner) or "") == was):
                    issues.append(
                        f"model conformance: {cls}.{attr} changed type {was} -> {now} in the "
                        f"class diagram, but {rel} still declares it as {was} in {schema}.")
            elif rule == "optional":
                if new["optional"] and not optional and field.value is None:
                    issues.append(
                        f"model conformance: {cls}.{attr} is now optional in the class diagram, "
                        f"but {rel} still requires it in {schema}.")
                elif not new["optional"] and optional:
                    issues.append(
                        f"model conformance: {cls}.{attr} is now required in the class diagram, "
                        f"but {rel} still declares it Optional in {schema}.")
    return issues


def model_conformance_issues(workspace: str, base: dict, new: dict) -> list[str]:
    """Findings for the attribute changes between two ``model_summary`` dicts.

    Empty when the workspace has neither file (another stack) or nothing changed.
    """
    checks = list(_attribute_checks(base, new))
    if not checks:
        return []
    issues = []
    for name, findings in ((_ORM_FILE, _orm_findings), (_SCHEMA_FILE, _schema_findings)):
        for path in _files(workspace, name):
            module = _parse(path, workspace)
            if module is not None:
                rel = os.path.relpath(path, workspace).replace("\\", "/")
                issues.extend(findings(module, rel, checks))
    return issues
