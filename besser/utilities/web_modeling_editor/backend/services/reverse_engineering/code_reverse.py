"""Read an existing code repository: its stack, and a domain model from its data classes.

Used when someone opens a GitHub repo BESSER did not create. The model is a
proposal the user accepts or edits on the canvas, so this reads only what code
states plainly and never runs it: Python sources are parsed with ``ast``.

Recognised data classes, in order of preference (the first kind found wins,
because API schemas usually restate the ORM models):

- SQLAlchemy models (``Column(...)`` / ``mapped_column(...)`` / ``Mapped[...]``),
- Django models (``models.Model`` subclasses),
- Pydantic models and dataclasses.

Foreign keys, ``relationship(...)`` and secondary tables become associations,
named after the code's own relationships; ``enum.Enum`` classes become
enumerations. Import aliases (``Integer as Integer_``) are resolved first, since
BESSER's own generators write them. A class counts as a model when its bases
lead to a model base anywhere in the repo, and the fields of abstract bases and
mixins move into the classes that extend them. Stacks other than Python are
reported but contribute no model.
"""

import ast
import json
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Optional

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    BooleanType,
    Class,
    DateTimeType,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    Generalization,
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
_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "enum.Enum", "enum.IntEnum", "enum.StrEnum",
               "TextChoices", "IntegerChoices", "models.TextChoices", "models.IntegerChoices"}
_DJANGO_RELATIONS = {"ForeignKey", "OneToOneField", "ManyToManyField"}
_DJANGO_BASES = {"models.Model", "Model", "AbstractBaseUser", "AbstractUser"}
_SCHEMA_BASES = {"BaseModel", "SQLModel"}
# A schema field holding a list of these is a list of keys (``authors: List[int]``), named after the class.
_KEY_TYPES = {"int", "UUID"}
# Fields Django's auth base classes declare for their subclasses.
_DJANGO_AUTH_FIELDS = {
    "AbstractBaseUser": (("password", "CharField"), ("last_login", "DateTimeField")),
    "AbstractUser": (("username", "CharField"), ("first_name", "CharField"), ("last_name", "CharField"),
                     ("email", "EmailField"), ("password", "CharField"), ("last_login", "DateTimeField"),
                     ("date_joined", "DateTimeField")),
}

SCHEMA_MODEL_SOURCE = "Pydantic models and dataclasses"


@dataclass
class CodeAnalysis:
    """What the repository holds, as shown to the user before they accept the model."""
    stack: list[str] = field(default_factory=list)
    model_source: Optional[str] = None  # "SQLAlchemy models", "Django models", ...
    source_files: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    ai: bool = False  # True when an LLM proposed the model (ai_reverse), not this exact reader
    # Relationships read from the code that the model could not take, as "Source -> Target".
    skipped_relationships: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stack": self.stack, "model_source": self.model_source,
            "source_files": self.source_files, "entities": self.entities, "ai": self.ai,
            "skipped_relationships": self.skipped_relationships,
        }


@dataclass
class _Field:
    name: str
    type_name: str
    is_id: bool = False
    optional: bool = False
    target: Optional[str] = None  # class or table this field refers to
    many: bool = False
    one: bool = False  # one-to-one: at most one object on the other end
    back: Optional[str] = None  # the code's name for the other end (Django ``related_name``)


@dataclass
class _Entity:
    name: str
    table: Optional[str]
    fields: list[_Field]
    source: str
    bases: tuple = ()
    abstract: bool = False  # Django ``Meta.abstract``, a SQLAlchemy mixin or ``__abstract__``


@dataclass
class _Relation:
    """``owner.name = relationship(target, ...)``: names one end of an association."""
    owner: str
    name: str
    target: str
    foreign_keys: frozenset = frozenset()  # "Owner.column"
    secondary: Optional[str] = None
    backref: Optional[str] = None  # names the other end


@dataclass
class _Module:
    """What the repo's Python files declare besides their data classes."""
    enums: dict = field(default_factory=dict)  # name -> literal names
    relations: list = field(default_factory=list)
    secondaries: dict = field(default_factory=dict)  # variable -> referenced tables
    bases: dict = field(default_factory=dict)  # class name -> base names
    fk_helpers: set = field(default_factory=set)  # functions returning a foreign-key Column


def analyze_repository(root: str) -> tuple[CodeAnalysis, Optional[DomainModel]]:
    """The stack of the repo at ``root`` and a domain model of its data classes (None when there are none)."""
    analysis = CodeAnalysis(stack=_detect_stack(root))
    candidates: dict[str, list[_Entity]] = {"sqlalchemy": [], "django": [], "schema": []}
    module = _Module()
    # Two passes: base classes and column helpers are often declared in another file.
    trees = []
    for path in _python_files(root):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except (SyntaxError, ValueError):
            continue
        tree = _Unalias(_import_aliases(tree)).visit(tree)
        _read_module(tree, module)
        trees.append((os.path.relpath(path, root).replace(os.sep, "/"), tree))
    for rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                kind, entity = _read_class(node, rel, module)
                if entity is not None:
                    candidates[kind].append(entity)

    for kind, label in (("sqlalchemy", "SQLAlchemy models"), ("django", "Django models"),
                        ("schema", SCHEMA_MODEL_SOURCE)):
        entities = _with_inherited_fields(_dedupe(candidates[kind]))
        if entities:
            analysis.model_source = label
            analysis.source_files = sorted({e.source for e in entities})
            model = _build_model(entities, module)
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


def _simple_base_names(node: ast.ClassDef) -> tuple:
    """``(TimestampedModel, Model)`` for ``class X(TimestampedModel, models.Model)``, in order."""
    return tuple(b.id if isinstance(b, ast.Name) else b.attr for b in node.bases
                 if isinstance(b, (ast.Name, ast.Attribute)))


def _derives_from(bases: set[str], roots: set[str], module: _Module) -> bool:
    """True when one of ``bases``, or a class they extend anywhere in the repo, is in ``roots``."""
    seen, todo = set(), list(bases)
    while todo:
        base = todo.pop()
        if base in roots:
            return True
        if base not in seen:
            seen.add(base)
            todo.extend(module.bases.get(base, ()))
    return False


def _type_name(node) -> Optional[str]:
    """``Integer`` / ``db.Integer`` -> "Integer"."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


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


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """``from sqlalchemy import Integer as Integer_`` -> {"Integer_": "Integer"}."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname and alias.asname != alias.name:
                    aliases[alias.asname] = alias.name.split(".")[-1]
    return aliases


class _Unalias(ast.NodeTransformer):
    """Rewrites aliased names to the names they were imported as."""

    def __init__(self, aliases: dict[str, str]):
        self.aliases = aliases

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.aliases:
            node.id = self.aliases[node.id]
        return node


def _read_module(tree: ast.Module, module: _Module) -> None:
    """Class bases, enums (also Django choices nested in a model), relationships declared outside
    their class, secondary (many-to-many) tables, and helpers such as ``reference_col('users')``
    that return a foreign-key column."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            module.bases[node.name] = _simple_base_names(node)
            if _base_names(node) & _ENUM_BASES:
                literals = [t.id for s in node.body if isinstance(s, ast.Assign)
                            for t in s.targets if isinstance(t, ast.Name) and not t.id.startswith("_")]
                if literals:
                    module.enums[node.name] = literals
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(s, ast.Return) and _call_name(s.value) in _ORM_FIELD_CALLS
                and any(_call_name(c) == "ForeignKey" for c in ast.walk(s.value))
                for s in ast.walk(node)):
            module.fk_helpers.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute) \
                and isinstance(node.target.value, ast.Name):
            relation = _relation(node.target.value.id, node.target.attr, node.value)
            if relation is not None:
                module.relations.append(relation)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and _call_name(node.value) == "Table":
            tables = [(_const_str(c.args[0]) or "").split(".")[0] for col in node.value.args
                      if _call_name(col) == "Column" for c in col.args if _call_name(c) == "ForeignKey" and c.args]
            if len(tables) == 2 and all(tables):
                module.secondaries[node.targets[0].id] = tables


def _relation(owner: str, name: str, value) -> Optional[_Relation]:
    if not _is_relationship(value) or not value.args:
        return None
    target = _const_str(value.args[0]) or (value.args[0].id if isinstance(value.args[0], ast.Name) else None)
    if not target:
        return None
    keys = _kw(value, "foreign_keys")
    columns = frozenset(
        f"{k.value.id}.{k.attr}" for k in (keys.elts if isinstance(keys, (ast.List, ast.Tuple)) else [keys])
        if isinstance(k, ast.Attribute) and isinstance(k.value, ast.Name)
    )
    secondary = _kw(value, "secondary")
    backref = _kw(value, "backref")
    if _call_name(backref) == "backref" and backref.args:
        backref = backref.args[0]  # backref('profile', uselist=False)
    return _Relation(owner, name, target.split(".")[-1], columns,
                     secondary.id if isinstance(secondary, ast.Name) else _const_str(secondary),
                     _const_str(backref))


def _is_abstract(node: ast.ClassDef) -> bool:
    """Django ``class Meta: abstract = True`` or SQLAlchemy ``__abstract__ = True``."""
    for statement in node.body:
        if isinstance(statement, ast.ClassDef) and statement.name == "Meta":
            if any(isinstance(s, ast.Assign) and any(getattr(t, "id", None) == "abstract" for t in s.targets)
                   and _is_true(s.value) for s in statement.body):
                return True
        elif isinstance(statement, ast.Assign) and any(getattr(t, "id", None) == "__abstract__"
                                                       for t in statement.targets):
            return _is_true(statement.value)
    return False


def _read_class(node: ast.ClassDef, source: str, module: _Module) -> tuple[str, Optional[_Entity]]:
    bases = _base_names(node)
    is_django = _derives_from(bases, _DJANGO_BASES, module)
    decorators = {d.id if isinstance(d, ast.Name) else _call_name(d) for d in node.decorator_list}
    table = None
    orm_fields: list[_Field] = []
    annotated: list[_Field] = []
    django_fields: list[_Field] = []
    discriminator = None  # the column SQLAlchemy stores the subclass in, not domain data

    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name, value = statement.targets[0].id, statement.value
            if name == "__tablename__":
                table = _const_str(value)
            elif name == "__mapper_args__" and isinstance(value, ast.Dict):
                on = next((v for k, v in zip(value.keys, value.values) if _const_str(k) == "polymorphic_on"), None)
                discriminator = _const_str(on) or (on.id if isinstance(on, ast.Name) else None)
            elif _call_name(value) in _ORM_FIELD_CALLS:
                orm_fields.append(_orm_field(name, value, None))
            elif _call_name(value) in module.fk_helpers and value.args and _const_str(value.args[0]):
                nullable = _kw(value, "nullable")
                orm_fields.append(_Field(name=name, type_name="int", target=_const_str(value.args[0]).split(".")[0],
                                         optional=isinstance(nullable, ast.Constant) and nullable.value is True))
            elif _is_relationship(value):
                relation = _relation(node.name, name, value)
                if relation is not None:
                    module.relations.append(relation)
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and is_django:
                django = _django_field(name, value, node.name)
                if django is not None:
                    django_fields.append(django)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            if name.startswith("_") or name == "model_config":
                continue
            if statement.value is not None and _call_name(statement.value) in _ORM_FIELD_CALLS:
                orm_fields.append(_orm_field(name, statement.value, statement.annotation))
            elif _is_relationship(statement.value):
                relation = _relation(node.name, name, statement.value)
                if relation is not None:
                    module.relations.append(relation)
            elif _is_mapped(statement.annotation):
                # SQLAlchemy 2.0: a bare ``year: Mapped[int]`` is a column too.
                orm_fields.append(_annotated_field(name, statement.annotation))
            else:
                annotated.append(_annotated_field(name, statement.annotation))

    own_bases = _simple_base_names(node)
    orm_fields = [f for f in orm_fields if f.name != discriminator]
    if orm_fields:
        # A class with columns but no table that others extend is a mixin (``SurrogatePK``).
        return "sqlalchemy", _Entity(node.name, table, orm_fields, source, own_bases,
                                     _is_abstract(node) or table is None)
    if is_django:
        for base in own_bases:
            django_fields += [_Field(name=n, type_name=t) for n, t in _DJANGO_AUTH_FIELDS.get(base, ())]
    if django_fields and is_django:
        return "django", _Entity(node.name, None, django_fields, source, own_bases, _is_abstract(node))
    if annotated and (_derives_from(bases, _SCHEMA_BASES, module) or "dataclass" in decorators):
        return "schema", _Entity(node.name, None, annotated, source, own_bases)
    return "schema", None


def _is_mapped(annotation) -> bool:
    """``Mapped[...]`` or ``orm.Mapped[...]``."""
    if not isinstance(annotation, ast.Subscript):
        return False
    outer = annotation.value
    return (outer.id if isinstance(outer, ast.Name) else getattr(outer, "attr", "")) == "Mapped"


def _is_relationship(value) -> bool:
    return _call_name(value) in {"relationship", "Relationship"}


def _orm_field(name: str, call: ast.Call, annotation) -> _Field:
    type_name, target = "str", None
    for arg in call.args:
        if _call_name(arg) == "ForeignKey" and arg.args:
            ref = _const_str(arg.args[0]) or ""
            target = ref.split(".")[0] or None
        elif _type_name(arg):
            type_name = _type_name(arg)  # Integer, db.Integer
        elif _call_name(arg) == "Enum" and arg.args and isinstance(arg.args[0], ast.Name):
            type_name = arg.args[0].id  # Enum(Priority)
        elif isinstance(arg, ast.Call) and _type_name(arg.func):
            type_name = _type_name(arg.func)  # String(80), db.String(80)
    is_id = _is_true(_kw(call, "primary_key"))
    nullable = _kw(call, "nullable")
    if annotation is not None:
        # Typed mapping: the annotation decides (Mapped[Optional[int]] is nullable)
        # unless the column names a precise type (Integer, Date, Enum(...)).
        inner, optional = _unwrap_annotation(annotation)
        if inner and type_name.lower() not in _TYPE_BY_NAME and not target:
            type_name = inner
    else:
        # A bare Column is nullable unless it is the primary key.
        optional = not is_id
    if isinstance(nullable, ast.Constant) and isinstance(nullable.value, bool):
        optional = nullable.value
    return _Field(name=name, type_name=type_name, is_id=is_id, optional=optional and not is_id, target=target)


def _django_field(name: str, call: ast.Call, owner: str) -> Optional[_Field]:
    kind = call.func.attr if isinstance(call.func, ast.Attribute) else None
    if not kind or not kind.endswith("Field") and kind not in _DJANGO_RELATIONS:
        return None
    optional = _is_true(_kw(call, "null")) or _is_true(_kw(call, "blank"))
    if kind in _DJANGO_RELATIONS:
        ref = call.args[0] if call.args else _kw(call, "to")
        target = ref.id if isinstance(ref, ast.Name) else (_const_str(ref) or "").split(".")[-1] or None
        return _Field(name=name, type_name=kind, optional=optional, target=owner if target == "self" else target,
                      many=kind == "ManyToManyField", one=kind == "OneToOneField",
                      back=_const_str(_kw(call, "related_name")))
    choices = _kw(call, "choices")
    if isinstance(choices, ast.Attribute) and choices.attr == "choices" and isinstance(choices.value, ast.Name):
        kind = choices.value.id  # choices=Genre.choices: the field holds a Genre
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
    node, many = annotation, False
    while isinstance(node, ast.Subscript) and not many:  # List[X] inside Optional[...]
        many = getattr(node.value, "id", "") in {"List", "list", "Set", "set"}
        node = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) and node.slice.elts else node.slice
    type_name, optional = _unwrap_annotation(annotation)
    return _Field(name=name, type_name=type_name or "str", optional=optional, many=many)


def _dedupe(entities: list[_Entity]) -> list[_Entity]:
    seen: dict[str, _Entity] = {}
    for entity in entities:
        if entity.name not in seen or len(entity.fields) > len(seen[entity.name].fields):
            seen[entity.name] = entity
    return list(seen.values())


def _with_inherited_fields(entities: list[_Entity]) -> list[_Entity]:
    """Abstract bases and mixins that other entities extend are dropped; their fields move into the subclasses."""
    by_name = {e.name: e for e in entities}
    parents = {b for e in entities for b in e.bases if b in by_name and by_name[b].abstract}

    def fields(entity: _Entity, seen: frozenset) -> list[_Field]:
        own = list(entity.fields)
        names = {f.name for f in own}
        for base in entity.bases:
            if base in parents and base not in seen:
                for inherited in fields(by_name[base], seen | {base}):
                    if inherited.name not in names:
                        own.append(inherited)
                        names.add(inherited.name)
        return own

    return [replace(e, fields=fields(e, frozenset({e.name}))) for e in entities if e.name not in parents]


# --------------------------------------------------------------------------
# Building the B-UML model
# --------------------------------------------------------------------------


def _build_model(entities: list[_Entity], module: _Module) -> DomainModel:
    classes = {e.name: Class(name=e.name) for e in entities}
    enums = {name: Enumeration(name=name, literals={EnumerationLiteral(name=lit) for lit in literals})
             for name, literals in module.enums.items() if name not in classes}
    by_table = {e.table: e.name for e in entities if e.table}
    by_lower = {name.lower(): name for name in classes}
    parents = {e.name: [b for b in e.bases if b in classes and b != e.name] for e in entities}
    generalizations = {Generalization(general=classes[p], specific=classes[child])
                       for child, ps in parents.items() for p in ps}

    def ancestors(name: str) -> set[str]:
        found, todo = set(), list(parents[name])
        while todo:
            parent = todo.pop()
            if parent not in found:
                found.add(parent)
                todo.extend(parents[parent])
        return found

    # One namespace per inheritance family: an end may not repeat a name anywhere in the hierarchy.
    family = {name: name for name in classes}
    for child, ps in parents.items():
        for parent in ps:
            family[_root(family, child)] = _root(family, parent)
    names_by_family: dict[str, set[str]] = {}
    # Names already taken on each class: an end may not repeat an attribute or another end.
    used = {name: names_by_family.setdefault(_root(family, name), set()) for name in classes}
    declared = {e.name: {f.name for f in e.fields} for e in entities}
    links: list[tuple[_Entity, _Field, str]] = []

    def resolve(target: Optional[str]) -> Optional[str]:
        if not target:
            return None
        return by_table.get(target) or by_lower.get(target.lower()) or by_lower.get(target.lower().rstrip("s"))

    def relation(owner: str, target: str, column: Optional[str] = None,
                 secondary: Optional[str] = None) -> Optional[_Relation]:
        """The ``relationship(...)`` on ``owner`` that reaches ``target`` over this link, when it is unambiguous."""
        options = [r for r in module.relations if r.owner == owner and resolve(r.target) == target]
        if secondary is not None:
            options = [r for r in options if r.secondary == secondary]
        elif column is not None:
            exact = [r for r in options if column in r.foreign_keys]
            options = exact or [r for r in options if not r.foreign_keys and not r.secondary]
        return options[0] if len(options) == 1 else None

    def names(there: Optional[_Relation], back_here: Optional[_Relation]) -> tuple[Optional[str], Optional[str]]:
        """Both ends' names from the relationships declared on either side (``backref`` names the far one)."""
        return ((there.name if there else None) or (back_here.backref if back_here else None),
                (back_here.name if back_here else None) or (there.backref if there else None))

    def unique(owner: str, name: str) -> str:
        candidate, n = name, 2
        while candidate in used[owner]:
            candidate, n = f"{name}{n}", n + 1
        used[owner].add(candidate)
        return candidate

    for entity in entities:
        owner = classes[entity.name]
        inherited = set().union(*(declared[a] for a in ancestors(entity.name)))
        for f in entity.fields:
            target = resolve(f.target) or (resolve(f.type_name) if f.type_name in classes else None)
            if not target and f.many and f.type_name in _KEY_TYPES:
                target = resolve(f.name.removesuffix("_ids"))
            if f.name in inherited or (f.is_id and target in parents[entity.name]):
                continue  # declared by a parent, or the key joining a subclass table to its parent's
            if target and target in classes:
                links.append((entity, f, target))
                continue
            kind = enums.get(f.type_name) or _TYPE_BY_NAME.get(f.type_name.lower(), StringType)
            used[entity.name].add(f.name)
            owner.add_attribute(Property(
                name=f.name, type=kind, owner=owner, is_id=f.is_id,
                multiplicity=Multiplicity(0 if f.optional else 1, 1),
            ))

    def multiplicity(f: _Field) -> Multiplicity:
        return Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY) if f.many else Multiplicity(0 if f.optional else 1, 1)

    # Schema classes state both sides of a link (``Library.books`` and ``Book.library``): pair the two
    # fields when each is the only one pointing back at the other's class.
    typed = [i for i, (_, f, _) in enumerate(links) if not f.target]
    facing = {i: [j for j in typed if j != i and links[j][0].name == links[i][2] and links[j][2] == links[i][0].name]
              for i in typed}
    partner = {i: js[0] for i, js in facing.items() if len(js) == 1 and facing[js[0]] == [i]}

    associations = []
    for i, (entity, f, target) in enumerate(links):
        source, column = entity.name, f"{entity.name}.{f.name}"
        if i in partner:
            if partner[i] > i:
                other = links[partner[i]][1]
                associations.append(BinaryAssociation(name=f"{source}_{f.name}", ends={
                    Property(name=unique(source, f.name), type=classes[target], multiplicity=multiplicity(f)),
                    Property(name=unique(target, other.name), type=classes[source], multiplicity=multiplicity(other)),
                }))
            continue
        if target == source:
            # A self link declares both ends on one class over the same key: ``mentor`` and ``mentees``.
            ends = [r for r in module.relations
                    if r.owner == source and resolve(r.target) == source and column in r.foreign_keys]
            there = next((r for r in ends if r.name == f.name.removesuffix("_id")), None)
            others = [r for r in ends if r is not there]
            forward, back = names(there, others[0] if there and len(others) == 1 else None)
        else:
            forward, back = names(relation(source, target, column), relation(target, source, column))
        forward = forward or f.name.removesuffix("_id") or target.lower()
        back = back or f.back or (source.lower() if f.one else f"{source.lower()}_set" if f.many
                                  else f"{source.lower()}s")
        associations.append(BinaryAssociation(name=f"{source}_{f.name}", ends={
            Property(name=unique(source, forward), type=classes[target], multiplicity=multiplicity(f)),
            Property(name=unique(target, back), type=classes[source],
                     multiplicity=Multiplicity(0, 1 if f.one else UNLIMITED_MAX_MULTIPLICITY)),
        }))
    for table_var, (left, right) in module.secondaries.items():
        a, b = resolve(left), resolve(right)
        if not (a and b):
            continue
        # A table joining a class to itself is a self many-to-many (followers).
        to_b, to_a = names(relation(a, b, secondary=table_var),
                           relation(b, a, secondary=table_var) if a != b else None)
        to_b, to_a = to_b or f"{b.lower()}s", to_a or f"{a.lower()}s"
        associations.append(BinaryAssociation(name=f"{a}_{b}", ends={
            Property(name=unique(a, to_b), type=classes[b], multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
            Property(name=unique(b, to_a), type=classes[a], multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
        }))

    model = DomainModel(name="ImportedModel", types=set(classes.values()) | set(enums.values()),
                        generalizations=generalizations)
    # Two links between the same classes (two association tables, two foreign
    # keys) share a default name; the second one gets a numbered name.
    taken: set[str] = set()
    for association in associations:
        name, n = association.name, 2
        while name in taken:
            name, n = f"{association.name}_{n}", n + 1
        if name != association.name:
            logger.info("Code reading: association %s renamed to %s, its name was taken", association.name, name)
            association.name = name
        taken.add(name)
        model.add_association(association)
    return model


def _root(family: dict[str, str], name: str) -> str:
    while family[name] != name:
        name = family[name]
    return name
