"""Read an existing code repository: its stack, and a domain model from its data classes.

Used when someone opens a GitHub repo BESSER did not create. The model is a
proposal the user accepts or edits on the canvas, so this reads only what code
states plainly and never runs it: Python sources are parsed with ``ast``.

Recognised data classes, in order of preference (the first kind found wins,
because API schemas usually restate the ORM models):

- SQLAlchemy models (``Column(...)`` / ``mapped_column(...)`` / ``Mapped[...]``),
- Django models (``models.Model`` subclasses),
- Pydantic models and dataclasses.

Foreign keys and relationships become associations. Stacks other than Python
are reported but contribute no model.
"""

import ast
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    BooleanType,
    Class,
    DateTimeType,
    DateType,
    DomainModel,
    FloatType,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
    TimeType,
    UNLIMITED_MAX_MULTIPLICITY,
)

logger = logging.getLogger(__name__)

_SKIPPED_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "env", "dist", "build", "site-packages",
    "migrations", "tests", "test", "buml", ".agent-trace",
}
# Bounds so a large repository cannot stall the import.
_MAX_PY_FILES = 2000
_MAX_FILE_BYTES = 512 * 1024

_TYPE_BY_NAME = {
    "int": IntegerType, "integer": IntegerType, "biginteger": IntegerType, "smallinteger": IntegerType,
    "integerfield": IntegerType, "bigintegerfield": IntegerType, "smallintegerfield": IntegerType,
    "positiveintegerfield": IntegerType, "autofield": IntegerType, "bigautofield": IntegerType,
    "float": FloatType, "numeric": FloatType, "decimal": FloatType, "floatfield": FloatType,
    "decimalfield": FloatType, "real": FloatType,
    "bool": BooleanType, "boolean": BooleanType, "booleanfield": BooleanType,
    "date": DateType, "datefield": DateType,
    "datetime": DateTimeType, "datetimefield": DateTimeType, "timestamp": DateTimeType,
    "time": TimeType, "timefield": TimeType,
}

_ORM_FIELD_CALLS = {"Column", "mapped_column"}
_DJANGO_RELATIONS = {"ForeignKey": (0, 1), "OneToOneField": (0, 1), "ManyToManyField": (0, UNLIMITED_MAX_MULTIPLICITY)}


@dataclass
class CodeAnalysis:
    """What the repository holds, as shown to the user before they accept the model."""
    stack: list[str] = field(default_factory=list)
    model_source: Optional[str] = None  # "SQLAlchemy models", "Django models", ...
    source_files: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stack": self.stack, "model_source": self.model_source,
            "source_files": self.source_files, "entities": self.entities,
        }


@dataclass
class _Field:
    name: str
    type_name: str
    is_id: bool = False
    optional: bool = False
    target: Optional[str] = None  # class or table this field refers to
    many: bool = False


@dataclass
class _Entity:
    name: str
    table: Optional[str]
    fields: list[_Field]
    source: str


def analyze_repository(root: str) -> tuple[CodeAnalysis, Optional[DomainModel]]:
    """The stack of the repo at ``root`` and a domain model of its data classes (None when there are none)."""
    analysis = CodeAnalysis(stack=_detect_stack(root))
    candidates: dict[str, list[_Entity]] = {"sqlalchemy": [], "django": [], "schema": []}
    for path in _python_files(root):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except (SyntaxError, ValueError):
            continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                kind, entity = _read_class(node, rel)
                if entity is not None:
                    candidates[kind].append(entity)

    for kind, label in (("sqlalchemy", "SQLAlchemy models"), ("django", "Django models"),
                        ("schema", "Pydantic models and dataclasses")):
        entities = _dedupe(candidates[kind])
        if entities:
            analysis.model_source = label
            analysis.source_files = sorted({e.source for e in entities})
            model = _build_model(entities)
            analysis.entities = sorted(c.name for c in model.get_classes())
            return analysis, model
    return analysis, None


def _python_files(root: str):
    count = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIPPED_DIRS and not d.startswith(".")]
        for name in files:
            path = os.path.join(current, name)
            if name.endswith(".py") and os.path.getsize(path) <= _MAX_FILE_BYTES:
                count += 1
                if count > _MAX_PY_FILES:
                    logger.warning("Read only the first %d Python files of %s", _MAX_PY_FILES, root)
                    return
                yield path


def _detect_stack(root: str) -> list[str]:
    found: list[str] = []

    def add(name: str) -> None:
        if name not in found:
            found.append(name)

    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIPPED_DIRS and not d.startswith(".")]
        depth = os.path.relpath(current, root).count(os.sep)
        if depth > 3:
            dirs[:] = []
        if "manage.py" in files:
            add("Django")
        if "pubspec.yaml" in files:
            add("Flutter")
        if "pom.xml" in files or "build.gradle" in files or "build.gradle.kts" in files:
            add("Java / Kotlin")
        if "go.mod" in files:
            add("Go")
        if "Cargo.toml" in files:
            add("Rust")
        if "package.json" in files:
            for framework in _js_frameworks(os.path.join(current, "package.json")):
                add(framework)
        for name in files:
            if name.endswith(".py") and depth <= 2:
                text = _head(os.path.join(current, name))
                if "FastAPI(" in text:
                    add("FastAPI")
                if "Flask(" in text:
                    add("Flask")
    return found


def _js_frameworks(package_json: str) -> list[str]:
    try:
        with open(package_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    deps = {**(data.get("dependencies") or {}), **(data.get("devDependencies") or {})}
    labels = (("next", "Next.js"), ("react", "React"), ("vue", "Vue"), ("@angular/core", "Angular"),
              ("svelte", "Svelte"), ("express", "Express"))
    return [label for dep, label in labels if dep in deps]


def _head(path: str, size: int = 65536) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(size)
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Reading one class
# --------------------------------------------------------------------------


def _base_names(node: ast.ClassDef) -> set[str]:
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
            if isinstance(base.value, ast.Name):
                names.add(f"{base.value.id}.{base.attr}")
    return names


def _call_name(node) -> Optional[str]:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _kw(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _const_str(node) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _read_class(node: ast.ClassDef, source: str) -> tuple[str, Optional[_Entity]]:
    bases = _base_names(node)
    decorators = {d.id if isinstance(d, ast.Name) else _call_name(d) for d in node.decorator_list}
    table = None
    orm_fields: list[_Field] = []
    annotated: list[_Field] = []
    django_fields: list[_Field] = []

    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name, value = statement.targets[0].id, statement.value
            if name == "__tablename__":
                table = _const_str(value)
            elif _call_name(value) in _ORM_FIELD_CALLS:
                orm_fields.append(_orm_field(name, value, None))
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and bases & {"models.Model", "Model"}:
                django = _django_field(name, value)
                if django is not None:
                    django_fields.append(django)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            if name.startswith("_") or name == "model_config":
                continue
            if statement.value is not None and _call_name(statement.value) in _ORM_FIELD_CALLS:
                orm_fields.append(_orm_field(name, statement.value, statement.annotation))
            elif _is_relationship(statement.value):
                continue
            else:
                annotated.append(_annotated_field(name, statement.annotation))

    if orm_fields:
        return "sqlalchemy", _Entity(node.name, table, orm_fields, source)
    if django_fields and bases & {"models.Model", "Model"}:
        return "django", _Entity(node.name, None, django_fields, source)
    if annotated and (bases & {"BaseModel", "SQLModel"} or "dataclass" in decorators):
        return "schema", _Entity(node.name, None, annotated, source)
    return "schema", None


def _is_relationship(value) -> bool:
    return _call_name(value) in {"relationship", "Relationship"}


def _orm_field(name: str, call: ast.Call, annotation) -> _Field:
    type_name, target = "str", None
    for arg in call.args:
        if _call_name(arg) == "ForeignKey" and arg.args:
            ref = _const_str(arg.args[0]) or ""
            target = ref.split(".")[0] or None
        elif isinstance(arg, ast.Name):
            type_name = arg.id
        elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
            type_name = arg.func.id
    is_id = _is_true(_kw(call, "primary_key"))
    nullable = _kw(call, "nullable")
    if annotation is not None:
        # Typed mapping: the annotation decides (Mapped[Optional[int]] is nullable).
        inner, optional = _unwrap_annotation(annotation)
        if type_name == "str" and inner:
            type_name = inner
    else:
        # A bare Column is nullable unless it is the primary key.
        optional = not is_id
    if isinstance(nullable, ast.Constant) and isinstance(nullable.value, bool):
        optional = nullable.value
    return _Field(name=name, type_name=type_name, is_id=is_id, optional=optional and not is_id, target=target)


def _django_field(name: str, call: ast.Call) -> Optional[_Field]:
    kind = call.func.attr if isinstance(call.func, ast.Attribute) else None
    if not kind or not kind.endswith("Field") and kind not in _DJANGO_RELATIONS:
        return None
    optional = _is_true(_kw(call, "null")) or _is_true(_kw(call, "blank"))
    if kind in _DJANGO_RELATIONS:
        ref = call.args[0] if call.args else _kw(call, "to")
        target = ref.id if isinstance(ref, ast.Name) else (_const_str(ref) or "").split(".")[-1] or None
        return _Field(name=name, type_name=kind, optional=optional, target=target,
                      many=kind == "ManyToManyField")
    return _Field(name=name, type_name=kind, is_id=_is_true(_kw(call, "primary_key")), optional=optional)


def _unwrap_annotation(annotation) -> tuple[Optional[str], bool]:
    """``Mapped[Optional[int]]`` -> ("int", True); ``List[Room]`` -> ("Room", False)."""
    optional = False
    node = annotation
    while isinstance(node, ast.Subscript):
        outer = node.value.id if isinstance(node.value, ast.Name) else getattr(node.value, "attr", "")
        if outer == "Optional":
            optional = True
        node = node.slice
        if isinstance(node, ast.Tuple) and node.elts:
            node = node.elts[0]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):  # int | None
        optional = True
        node = node.left
    if isinstance(node, ast.Name):
        return node.id, optional
    if isinstance(node, ast.Attribute):
        return node.attr, optional
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, optional
    return None, optional


def _annotated_field(name: str, annotation) -> _Field:
    many = isinstance(annotation, ast.Subscript) and getattr(annotation.value, "id", "") in {"List", "list", "Set", "set"}
    type_name, optional = _unwrap_annotation(annotation)
    return _Field(name=name, type_name=type_name or "str", optional=optional, many=many)


def _dedupe(entities: list[_Entity]) -> list[_Entity]:
    seen: dict[str, _Entity] = {}
    for entity in entities:
        if entity.name not in seen or len(entity.fields) > len(seen[entity.name].fields):
            seen[entity.name] = entity
    return list(seen.values())


# --------------------------------------------------------------------------
# Building the B-UML model
# --------------------------------------------------------------------------


def _build_model(entities: list[_Entity]) -> DomainModel:
    classes = {e.name: Class(name=e.name) for e in entities}
    by_table = {e.table: e.name for e in entities if e.table}
    by_lower = {name.lower(): name for name in classes}
    associations = []

    def resolve(target: Optional[str]) -> Optional[str]:
        if not target:
            return None
        return by_table.get(target) or by_lower.get(target.lower()) or by_lower.get(target.lower().rstrip("s"))

    for entity in entities:
        owner = classes[entity.name]
        for f in entity.fields:
            target = resolve(f.target) or (resolve(f.type_name) if f.type_name in classes else None)
            if target and target in classes:
                associations.append(_association(owner, classes[target], f))
                continue
            primitive = _TYPE_BY_NAME.get(f.type_name.lower(), StringType)
            owner.add_attribute(Property(
                name=f.name, type=primitive, owner=owner, is_id=f.is_id,
                multiplicity=Multiplicity(0 if f.optional else 1, 1),
            ))

    model = DomainModel(name="ImportedModel", types=set(classes.values()))
    names = set()
    for association in associations:
        if association.name not in names:
            names.add(association.name)
            model.add_association(association)
    return model


def _association(source: Class, target: Class, f: _Field) -> BinaryAssociation:
    """``source.f`` refers to ``target``: many sources per target unless the field holds many targets."""
    to_target = Property(
        name=f.name.removesuffix("_id") or target.name.lower(),
        type=target,
        multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY) if f.many else Multiplicity(0 if f.optional else 1, 1),
    )
    back = Property(
        name=f"{source.name.lower()}_set" if f.many else f"{source.name.lower()}s",
        type=source,
        multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY),
    )
    return BinaryAssociation(name=f"{source.name}_{f.name}", ends={to_target, back})
