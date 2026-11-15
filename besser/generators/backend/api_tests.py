"""The tests the backend generator writes for its own API (option ``tests``): ``tests/conftest.py`` and
``tests/test_generated_api.py`` in the backend folder.

For each concrete class, ``test_generated_api.py`` creates an object with a payload that is valid for the generated
API, reads it back, finds it in the list, updates it and deletes it. The payload is built here, from the
B-UML model, by the rules the generated code follows:

- the ``<Class>Create`` fields (``templates/router.py.j2`` and the pydantic template): every client-supplied
  attribute (``is_server_owned_attribute``), and the association ends the schema requires: a to-one end on
  the side that holds the foreign key with a lower bound of 1 (a 1:1 end whose foreign key is deferred is
  optional), a many-to-many end with a lower bound (its minimum number of ids), an end through an
  association class with a lower bound (link objects ``{"target": id, ...}``);
- the objects those ends name are created first, in dependency order, each with its own required ends; an
  abstract class is stood for by a concrete subclass (joined inheritance: a subclass row is a parent row);
- the OCL constraints the Create schema enforces (``ocl_utils.build_constraints_map``, as the pydantic
  generator calls it) are evaluated here on the candidate values, so the payload satisfies them: numeric
  comparisons, ``matches('regex')`` and expressions over several attributes;
- deleting an object that a related object needs (a lower bound on this class's end) is refused with 409:
  the test expects that refusal.

A class for which no valid payload can be derived gets a skipped test that says why, never a failing one.
The Spec-Driven Agent's ``validation/constructibility.py`` builds payloads too, but at run time from the
running app's OpenAPI schemas; that is a different input, so nothing is shared with it.
"""

from __future__ import annotations

import ast
import enum
import itertools
import logging
import os
import re
import types
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import AssociationClass, Class, DomainModel, Enumeration, Property
from besser.generators.backend.api_generator import get_association_classes, get_pk_names
from besser.generators.pydantic_classes.ocl_utils import build_constraints_map
from besser.generators.structural_utils import (
    get_deferred_fk_associations,
    get_foreign_keys,
    is_server_owned_attribute,
)

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
TESTS_FOLDER = "tests"

# Objects one test may create, prerequisites included: a deeper chain of required ends is skipped.
_MAX_OBJECTS = 30
# Value combinations tried for one OCL constraint over several attributes.
_MAX_COMBINATIONS = 5000
_BASE_DATE = date(2024, 1, 1)
# Attribute types whose value the read-back compares; an enumeration reads back as its value, the others
# (timedelta, any) in a database-dependent form.
_COMPARED_TYPES = ("str", "string", "int", "integer", "float", "bool", "boolean", "date", "datetime", "time")
_UPDATE_ORDER = ("str", "string", "int", "integer", "float", "date", "datetime", "time", "bool", "boolean")


class _Unbuildable(Exception):
    """No valid payload can be derived; the message is the skip reason."""


@dataclass
class _Object:
    """One object a test creates: the python variable that holds its id and the payload it is posted with."""

    var: str
    cls: Class
    fields: list[tuple[str, str]]  # (field, python source of its value), in payload order
    values: dict[str, Any]  # attribute -> python value, as the API validates it
    links: list[tuple[Any, Property]] = field(default_factory=list)  # (association, far end) linked at creation


@dataclass
class _ClassTest:
    """The test of one class: rendered lines, or the reason it is skipped."""

    name: str
    cls: Class
    lines: list[str] = field(default_factory=list)
    skip: str | None = None

    @property
    def skip_source(self) -> str:
        return repr(self.skip)


# ── values ─────────────────────────────────────────────────────────────────


def _type_name(attribute: Property) -> str:
    return "enum" if isinstance(attribute.type, Enumeration) else getattr(attribute.type, "name", "")


def _json_source(value: Any) -> str:
    """The python source of ``value`` as the JSON payload carries it."""
    if isinstance(value, enum.Enum):
        return repr(value.name)
    if isinstance(value, (date, datetime, time)):
        return repr(value.isoformat())
    if isinstance(value, timedelta):
        return repr(int(value.total_seconds()))
    return repr(value)


def _regex_sample(pattern: str, variant: int) -> str | None:
    """A string matching ``pattern``, varied by ``variant``; None when the pattern is beyond this sampler."""
    import re._parser as sre_parse  # noqa: PLC0415  (stdlib: the parser ``re`` compiles with)

    try:
        parsed = sre_parse.parse(pattern)
    except re.error:
        return None
    categories = {
        sre_parse.CATEGORY_DIGIT: "1234567890", sre_parse.CATEGORY_WORD: "abcdefghXYZ_01",
        sre_parse.CATEGORY_SPACE: " ", sre_parse.CATEGORY_NOT_DIGIT: "abcxyz",
        sre_parse.CATEGORY_NOT_WORD: "-.!", sre_parse.CATEGORY_NOT_SPACE: "abc123",
    }
    position = [variant]

    def pick(options: str) -> str:
        position[0] += 1
        return options[position[0] % len(options)]

    def char_set(items) -> str | None:
        if items and items[0][0] == sre_parse.NEGATE:
            excluded = set()
            for op, av in items[1:]:
                if op == sre_parse.LITERAL:
                    excluded.add(chr(av))
                elif op == sre_parse.RANGE:
                    excluded.update(chr(c) for c in range(av[0], av[1] + 1))
                else:
                    return None
            options = "".join(c for c in "abcxyzABC123" if c not in excluded)
            return options or None
        options = ""
        for op, av in items:
            if op == sre_parse.LITERAL:
                options += chr(av)
            elif op == sre_parse.RANGE:
                options += "".join(chr(c) for c in range(av[0], min(av[1], av[0] + 9) + 1))
            elif op == sre_parse.CATEGORY and av in categories:
                options += categories[av]
            else:
                return None
        return options or None

    def render(sequence) -> str | None:
        out = []
        for op, av in sequence:
            if op == sre_parse.LITERAL:
                out.append(chr(av))
            elif op == sre_parse.ANY:
                out.append(pick("abcxyz"))
            elif op == sre_parse.IN:
                options = char_set(av)
                if options is None:
                    return None
                out.append(pick(options))
            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                low, high, inner = av
                count = max(low, 1) if high != 0 else 0
                if high is not sre_parse.MAXREPEAT:
                    count = min(count, high)
                for _ in range(count):
                    part = render(inner)
                    if part is None:
                        return None
                    out.append(part)
            elif op == sre_parse.SUBPATTERN:
                part = render(av[-1])
                if part is None:
                    return None
                out.append(part)
            elif op == sre_parse.BRANCH:
                part = render(av[1][0])
                if part is None:
                    return None
                out.append(part)
            elif op == sre_parse.AT:
                continue
            else:
                return None
        return "".join(out)

    return render(parsed)


def _constants(expressions: list[str]) -> tuple[list[float], list[str], list[str]]:
    """The numeric and string constants of ``expressions`` and the patterns they ``re.fullmatch``."""
    numbers: list[float] = []
    strings: list[str] = []
    patterns: list[str] = []
    for source in expressions:
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                patterns.append(node.args[0].value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                    and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)) \
                    and not isinstance(node.operand.value, bool):
                numbers.append(-node.operand.value)
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, bool):
                    continue
                if isinstance(node.value, (int, float)):
                    numbers.append(node.value)
                elif isinstance(node.value, str):
                    strings.append(node.value)
    return numbers, [s for s in strings if s not in patterns], patterns


def _unique(values: list) -> list:
    seen: list = []
    for value in values:
        if not any(type(value) is type(other) and value == other for other in seen):
            seen.append(value)
    return seen


def _candidates(cls: Class, attribute: Property, ordinal: int, expressions: list[str]) -> list | None:
    """The values tried for ``attribute`` of an object numbered ``ordinal``, the likeliest first; None for a
    type no value is generated for."""
    kind = _type_name(attribute)
    numbers, strings, patterns = _constants(expressions)
    if kind == "enum":
        members = enum.Enum(attribute.type.name, {literal.name: literal.name for literal in attribute.type.literals})
        # The default first: a literal, or its name, qualified or not ("Status.TODO").
        default = str(getattr(attribute.default_value, "name", attribute.default_value)).rsplit(".", 1)[-1]
        return sorted(members, key=lambda member: (member.name != default, member.name))
    if kind in ("str", "string"):
        first = (f"{cls.name.lower()}-{attribute.name.lower()}-{ordinal}" if attribute.is_id
                 else f"{cls.name} {attribute.name} {ordinal}")
        samples = [_regex_sample(pattern, ordinal + shift) for pattern in patterns for shift in range(4)]
        return _unique([*(s for s in samples if s is not None), first, *strings, f"{first}x", "x" * 10, ""])
    if kind in ("int", "integer"):
        ints = [int(n) for n in numbers]
        derived = [v for n in ints for v in (n + ordinal, n, n + 1, n - 1, n * 2, n + 10, n - 10, n + 100)]
        return _unique([ordinal, *derived, 0, 1, 10, 100, 1000, -1, ordinal + 1000])
    if kind == "float":
        derived = [float(v) for n in numbers for v in (n + ordinal, n, n + 0.5, n - 0.5, n + 1, n - 1, n * 2, n + 10)]
        return _unique([ordinal + 0.5, *derived, 0.5, 1.0, 10.0, 100.0, 0.0, -1.0, ordinal + 1000.5])
    if kind in ("bool", "boolean"):
        return [True, False]
    if kind == "date":
        return [_BASE_DATE + timedelta(days=ordinal + shift) for shift in (0, 365, -365, 30, -30, 3650)]
    if kind == "datetime":
        start = datetime(_BASE_DATE.year, _BASE_DATE.month, _BASE_DATE.day, 9, 30)
        return [start + timedelta(days=ordinal + shift) for shift in (0, 365, -365, 30, -30, 3650)]
    if kind == "time":
        return [time(9, 30), time(12, 0), time(18, 45), time(0, 15), time(23, 30)]
    if kind == "timedelta":
        return [timedelta(hours=1), timedelta(days=1), timedelta(0)]
    return None


@dataclass(frozen=True)
class _Constraint:
    name: str
    attributes: tuple[str, ...]
    source: str  # a python expression over ``v`` (one attribute) or ``self``
    on_value: bool


def _class_constraints(cls: Class, constraints_map: dict) -> list[_Constraint]:
    """The OCL constraints the ``<cls>Create`` schema enforces: its own and its parents'."""
    found = []
    for owner in [cls, *sorted(cls.all_parents(), key=lambda parent: parent.name)]:
        for parsed in constraints_map.get(owner.name, []):
            if parsed.get("skipped") or parsed.get("rejected"):
                continue
            name = parsed.get("constraint_name") or "unnamed"
            if "model_expression" in parsed:
                source = parsed["model_expression"]
                names = tuple(sorted(set(re.findall(r"\bself\.(\w+)", source))))
                found.append(_Constraint(name, names, source, False))
            elif "property" in parsed:
                source = parsed.get("python_expression") or f"v {parsed['python_operator']} {parsed['value_repr']}"
                found.append(_Constraint(name, (parsed["property"],), source, True))
    return found


def _holds(constraint: _Constraint, values: dict[str, Any]) -> bool:
    """Whether the validator the pydantic generator writes for ``constraint`` accepts ``values``. The source
    passed its AST white-list (names ``v``, ``self``, ``re``; no calls but ``re.fullmatch``)."""
    if any(name not in values for name in constraint.attributes):
        return False
    scope: dict[str, Any] = {"re": re}
    if constraint.on_value:
        scope["v"] = values[constraint.attributes[0]]
    else:
        scope["self"] = types.SimpleNamespace(**values)
    try:
        return bool(eval(constraint.source, {"__builtins__": {}}, scope))  # noqa: S307 (white-listed source)
    except Exception:  # a validator that raises rejects the payload too
        return False


def _satisfy(constraints: list[_Constraint], candidates: dict[str, list], values: dict[str, Any],
             owner: str) -> dict[str, Any]:
    """``values`` changed where needed so that every constraint holds."""
    values = dict(values)
    for _ in range(2 * len(constraints) + 1):
        failing = [constraint for constraint in constraints if not _holds(constraint, values)]
        if not failing:
            return values
        constraint = failing[0]
        names = [name for name in constraint.attributes if name in candidates]
        if len(names) != len(constraint.attributes):
            raise _Unbuildable(f"the OCL constraint '{constraint.name}' of {owner} reads a value the payload "
                               "does not carry")
        touching = [other for other in constraints if set(other.attributes) & set(names)]
        for combination in itertools.islice(itertools.product(*(candidates[name] for name in names)),
                                            _MAX_COMBINATIONS):
            trial = {**values, **dict(zip(names, combination))}
            if all(_holds(other, trial) for other in touching):
                values = trial
                break
        else:
            raise _Unbuildable(f"no value found that satisfies the OCL constraint '{constraint.name}' of {owner}")
    raise _Unbuildable(f"no values found that satisfy the OCL constraints of {owner} together")


# ── the model as the generated API reads it ────────────────────────────────


class _ApiShape:
    """What the generated API asks of each class, read from the model once."""

    def __init__(self, model: DomainModel, http_methods: list[str]):
        self.model = model
        self.http_methods = set(http_methods)
        self.fkeys = get_foreign_keys(model)
        self.deferred = get_deferred_fk_associations(model)
        self.assoc_classes = get_association_classes(model)
        self.link_class = {info["association"]: name for name, info in self.assoc_classes.items()}
        self.pk_names = get_pk_names(model)
        self.constraints_map = build_constraints_map(model, include_model_level=True, include_skipped=True)
        self.classes = {cls.name: cls for cls in model.get_classes()}
        self.unmapped, self.uncreatable = self._generator_limits()

    def _generator_limits(self) -> tuple[str | None, dict[str, str]]:
        """Shapes the backend generator writes code for that does not run, though the model is valid: why no
        route of the app works (its SQLAlchemy mappers cannot be configured), or None; and, by class, why
        ``POST`` of that class fails. Their tests are skipped with that reason, not left to fail."""
        unmapped = None
        uncreatable: dict[str, str] = {}
        for cls in sorted(self.model.get_classes(), key=lambda cls: cls.name):
            # An abstract root without association ends is a concrete-table base (SQLAlchemy generator's
            # get_concrete_table_inheritance); any other parent is a table the class joins.
            concrete = [p for p in cls.parents() if p.is_abstract and not p.parents() and not p.association_ends()]
            if len(concrete) > 1 or len(cls.parents()) - len(concrete) > 1:
                unmapped = unmapped or (
                    f"{cls.name} specializes {', '.join(sorted(p.name for p in cls.parents()))}: the backend "
                    "generator maps a class onto one parent table or one abstract base only, so no route of the "
                    "generated app works. A limitation of the generator, not of the model")
        for association in sorted(self.model.associations, key=lambda assoc: assoc.name):
            ends = list(association.ends)
            if len(ends) != 2 or association.name in self.link_class:
                continue
            first, second = ends[0].type, ends[1].type
            to_many = [end.multiplicity.max > 1 for end in ends]
            if first is second and all(to_many):
                unmapped = unmapped or (
                    f"the many-to-many self-association '{association.name}' of {first.name}: the backend generator "
                    "writes it without the join columns SQLAlchemy needs (ambiguous join), so no route of the "
                    "generated app works. A limitation of the generator, not of the model")
            elif first is second and not any(to_many):
                for cls in [first, *first.all_specializations()]:
                    uncreatable.setdefault(cls.name, (
                        f"the one-to-one self-association '{association.name}' of {first.name}: the backend "
                        f"generator's POST /{cls.name.lower()}/ passes its foreign key under a name the table "
                        "does not have. A limitation of the generator, not of the model"))
            elif first is not second and not all(to_many):
                holder_name, _ = self.fkeys.get(association.name, ("", ""))
                holder = first if first.name == holder_name else second
                target = second if holder is first else first
                if target in holder.all_parents() or holder in target.parents():
                    unmapped = unmapped or (
                        f"the association '{association.name}' between {holder.name} and {target.name}, which "
                        f"{'specializes' if target in holder.all_parents() else 'is specialized by'} it: the "
                        "backend generator writes its foreign key beside the inheritance one without telling "
                        "SQLAlchemy which joins them (ambiguous inherit condition), so no route of the generated "
                        "app works. A limitation of the generator, not of the model")
        return unmapped, uncreatable

    def attributes(self, cls: Class) -> list[Property]:
        """The client-supplied attributes of ``cls``, inherited ones included, by name."""
        found = {}
        for owner in [*sorted(cls.all_parents(), key=lambda parent: parent.name), cls]:
            for attribute in owner.attributes:
                if not is_server_owned_attribute(attribute):
                    found[attribute.name] = attribute
        return [found[name] for name in sorted(found)]

    def is_creatable(self, cls: Class) -> bool:
        return not cls.is_abstract and not isinstance(cls, AssociationClass)

    def stand_ins(self, cls: Class) -> list[Class]:
        """The classes whose objects can stand for a ``cls``: itself when concrete, then its concrete
        subclasses, by name."""
        found = [cls] if self.is_creatable(cls) else []
        found += [sub for sub in sorted(cls.all_specializations(), key=lambda sub: sub.name) if self.is_creatable(sub)]
        return found

    def required_ends(self, cls: Class) -> list[tuple[Any, Property, int, str]]:
        """The association ends ``<cls>Create`` requires: (association, far end, how many, shape) with shape
        ``id``, ``ids`` or ``links``, by field name."""
        required = {}
        for owner in [cls, *sorted(cls.all_parents(), key=lambda parent: parent.name)]:
            for association in sorted(owner.associations, key=lambda assoc: assoc.name):
                ends = sorted(association.ends, key=lambda end: end.name)
                if len(ends) != 2:
                    continue
                if ends[0].type is ends[1].type:
                    pairs = [(ends[1], ends[0]), (ends[0], ends[1])]  # as the templates read a self association
                else:
                    near = ends[0] if ends[0].type is owner else ends[1]
                    pairs = [(near, ends[1] if near is ends[0] else ends[0])]
                for near, far in pairs:
                    shape = self._required_shape(association, near, far, self_association=len(pairs) == 2)
                    if shape:
                        required[far.name] = (association, far, max(far.multiplicity.min, 1), shape)
        return [required[name] for name in sorted(required)]

    def _required_shape(self, association, near: Property, far: Property, self_association: bool) -> str | None:
        if far.multiplicity.min == 0:
            return None
        if association.name in self.link_class:
            return "links" if far.multiplicity.max > 1 else None
        if not self_association and not far.is_navigable:
            return None
        if far.multiplicity.max > 1:
            return "ids" if near.multiplicity.max > 1 else None
        holds_fk = self_association or self.fkeys.get(association.name, [""])[0] == near.type.name
        if not holds_fk:
            return None
        if near.multiplicity.max == 1 and association.name in self.deferred and not self_association:
            return None
        return "id"

    def guards(self, association, far: Property) -> bool:
        """Whether the objects at ``far`` of ``association`` need at least one object at its other end: the
        generated delete of that object refuses to leave them without (``lower_bound_guards`` in
        ``router.py.j2``; never for a self association, whose ends it does not tell apart)."""
        ends = list(association.ends)
        if len(ends) != 2 or ends[0].type is ends[1].type:
            return False
        near = ends[0] if ends[1] is far else ends[1]
        whole = near.is_composite and near.multiplicity.max <= 1 and association.name not in self.link_class
        return far.is_navigable and near.multiplicity.min > 0 and not whole


# ── one test ───────────────────────────────────────────────────────────────


class _Planner:
    """The objects one test creates, prerequisites first."""

    def __init__(self, shape: _ApiShape):
        self.shape = shape
        self.objects: list[_Object] = []
        self.used: set[tuple[str, str]] = set()  # unique attribute values already taken

    def _state(self):
        return len(self.objects), set(self.used)

    def _restore(self, state) -> None:
        del self.objects[state[0]:]
        self.used = state[1]

    def values(self, cls: Class, ordinal: int, check_constraints: bool = True) -> dict[str, Any]:
        """Valid attribute values for an object of ``cls`` numbered ``ordinal``."""
        constraints = _class_constraints(cls, self.shape.constraints_map) if check_constraints else []
        candidates: dict[str, list] = {}
        for attribute in self.shape.attributes(cls):
            expressions = [c.source for c in constraints if attribute.name in c.attributes]
            options = _candidates(cls, attribute, ordinal, expressions)
            if options is None:
                raise _Unbuildable(f"{cls.name}.{attribute.name} has type {_type_name(attribute)}, "
                                   "for which no test value is generated")
            if attribute.is_id or attribute.is_external_id:
                options = [v for v in options if (attribute.name, _json_source(v)) not in self.used]
            if not options:
                raise _Unbuildable(f"no unused value for the unique {cls.name}.{attribute.name}")
            candidates[attribute.name] = options
        values = _satisfy(constraints, candidates, {name: options[0] for name, options in candidates.items()},
                          cls.name)
        for attribute in self.shape.attributes(cls):
            if attribute.is_id or attribute.is_external_id:
                self.used.add((attribute.name, _json_source(values[attribute.name])))
        return values

    def build(self, cls: Class, chain: tuple[str, ...] = ()) -> _Object:
        """Plan an object of ``cls`` after the objects its required ends name."""
        if cls.name in self.shape.uncreatable:
            raise _Unbuildable(self.shape.uncreatable[cls.name])
        chain = (*chain, cls.name)
        fields: list[tuple[str, str]] = []
        links = []
        for association, far, count, kind in self.shape.required_ends(cls):
            targets = [self._prerequisite(far, chain) for _ in range(count)]
            links.append((association, far))
            if kind == "id":
                fields.append((far.name, targets[0]))
            elif kind == "ids":
                fields.append((far.name, f"[{', '.join(targets)}]"))
            else:
                link_class = self.shape.classes[self.shape.link_class[association.name]]
                items = []
                for target in targets:
                    ordinal = len(self.objects) + 1
                    attrs = self.values(link_class, ordinal, check_constraints=False)
                    link_attrs = "".join(f", {name!r}: {_json_source(value)}" for name, value in attrs.items())
                    items.append(f"{{'target': {target}{link_attrs}}}")
                fields.append((far.name, f"[{', '.join(items)}]"))
        if len(self.objects) >= _MAX_OBJECTS:
            raise _Unbuildable(f"more than {_MAX_OBJECTS} objects would be created first")
        ordinal = len(self.objects) + 1
        values = self.values(cls, ordinal)
        payload = [(name, _json_source(value)) for name, value in values.items()]
        obj = _Object(var=f"{cls.name.lower()}_{ordinal}", cls=cls, fields=sorted(payload + fields),
                      values=values, links=links)
        self.objects.append(obj)
        return obj

    def _prerequisite(self, far: Property, chain: tuple[str, ...]) -> str:
        """The variable holding the id of a new object that can stand at ``far``."""
        target = far.type
        stand_ins = self.shape.stand_ins(target)
        if not stand_ins:
            raise _Unbuildable(f"{chain[-1]} requires a {target.name} ({far.name}), and {target.name} has no "
                               "class the API can create")
        reason = ""
        for stand_in in stand_ins:
            if stand_in.name in chain:
                reason = (f"{' -> '.join(chain)} -> {stand_in.name}: a cycle of required association ends, so "
                          "none of them can be created first")
                continue
            state = self._state()
            try:
                return self.build(stand_in, chain).var
            except _Unbuildable as exc:
                self._restore(state)
                reason = str(exc)
        raise _Unbuildable(reason)


def _update(shape: _ApiShape, planner: _Planner, obj: _Object) -> tuple[str, Any] | None:
    """An attribute of ``obj`` to change, and its new valid value; None when none can change."""
    constraints = _class_constraints(obj.cls, shape.constraints_map)
    pk = shape.pk_names.get(obj.cls.name, "id")
    # Never a key, own or inherited: a subclass's row is addressed by its parent's is_id attribute, which
    # is not the class's own primary key name.
    attributes = [a for a in shape.attributes(obj.cls)
                  if a.name != pk and not a.is_id and not a.is_external_id and _type_name(a) in _UPDATE_ORDER]
    attributes.sort(key=lambda a: (_UPDATE_ORDER.index(_type_name(a)), a.name))
    for attribute in attributes:
        current = obj.values[attribute.name]
        expressions = [c.source for c in constraints if attribute.name in c.attributes]
        options = list(_candidates(obj.cls, attribute, len(planner.objects) + 1, expressions) or [])
        if isinstance(current, str):
            options.insert(0, f"{current} updated")
        for option in options:
            if type(option) is type(current) and option == current:
                continue
            trial = {**obj.values, attribute.name: option}
            if all(_holds(c, trial) for c in constraints):
                return attribute.name, option
    return None


def _payload_source(obj: _Object) -> str:
    return "{" + ", ".join(f"{name!r}: {source}" for name, source in obj.fields) + "}"


def _compared(shape: _ApiShape, cls: Class) -> set[str]:
    """The attributes of ``cls`` whose value the read-back compares."""
    return {a.name for a in shape.attributes(cls) if _type_name(a) in _COMPARED_TYPES}


def _expected(shape: _ApiShape, obj: _Object, names=None) -> str:
    compared = _compared(shape, obj.cls)
    items = [(name, value) for name, value in obj.values.items()
             if name in compared and (names is None or name in names)]
    return "{" + ", ".join(f"{name!r}: {_json_source(value)}" for name, value in items) + "}"


def _class_test(shape: _ApiShape, cls: Class) -> _ClassTest:
    name = f"test_{cls.name.lower()}"
    if cls.is_abstract:
        return _ClassTest(name, cls, skip=f"{cls.name} is abstract: the tests of its subclasses create it")
    if isinstance(cls, AssociationClass):
        return _ClassTest(name, cls, skip=f"{cls.name} is an association class: its links are created with "
                                           "the objects it links, so test them in your own tests")
    if "POST" not in shape.http_methods:
        return _ClassTest(name, cls, skip="the API has no POST route (http_methods), so nothing can be created")
    if shape.unmapped:
        return _ClassTest(name, cls, skip=shape.unmapped)
    planner = _Planner(shape)
    try:
        obj = planner.build(cls)
    except _Unbuildable as exc:
        return _ClassTest(name, cls, skip=f"no valid {cls.name} payload: {exc}")
    return _ClassTest(name, cls, lines=_test_lines(shape, planner, obj))


def _test_lines(shape: _ApiShape, planner: _Planner, obj: _Object) -> list[str]:
    cls = obj.cls
    path, key, pk = f"/{cls.name.lower()}/", cls.name.lower(), shape.pk_names.get(cls.name, "id")
    lines = []
    for prerequisite in planner.objects[:-1]:
        p_cls = prerequisite.cls
        lines.append(f"{prerequisite.var} = _create(client, {f'/{p_cls.name.lower()}/'!r}, "
                     f"{p_cls.name.lower()!r}, {_payload_source(prerequisite)})"
                     f"[{shape.pk_names.get(p_cls.name, 'id')!r}]")
    lines.append(f"payload = {_payload_source(obj)}")
    lines.append(f"item_id = _create(client, {path!r}, {key!r}, payload)[{pk!r}]")
    if "GET" in shape.http_methods:
        lines.append(f"_assert_fields(_read(client, {path!r}, {key!r}, item_id), {_expected(shape, obj)})")
        lines += [f"listed = client.get({path!r})",
                  "assert listed.status_code == 200, listed.text",
                  f"assert item_id in [item.get({pk!r}) for item in listed.json()]"]
    if "PUT" in shape.http_methods and not cls.is_read_only:
        change = _update(shape, planner, obj)
        if change is not None:
            obj.values[change[0]] = change[1]
            lines.append(f"payload[{change[0]!r}] = {_json_source(change[1])}")
        lines += [f"response = client.put(_item({path!r}, item_id), json=payload)",
                  "assert response.status_code == 200, response.text"]
        if change is not None and "GET" in shape.http_methods and change[0] in _compared(shape, cls):
            lines.append(f"_assert_fields(_read(client, {path!r}, {key!r}, item_id), "
                         f"{_expected(shape, obj, {change[0]})})")
    if "DELETE" in shape.http_methods and not cls.is_read_only:
        needed_by = sorted({far.type.name for association, far in obj.links if shape.guards(association, far)})
        lines.append(f"response = client.delete(_item({path!r}, item_id))")
        if needed_by:
            lines.append(f"# The {', '.join(needed_by)} it was created with needs at least one {cls.name}: "
                         "the API refuses to delete it.")
            lines.append("assert response.status_code == 409, response.text")
            if "GET" in shape.http_methods:
                lines.append(f"assert client.get(_item({path!r}, item_id)).status_code == 200")
        else:
            lines.append("assert response.status_code == 200, response.text")
            if "GET" in shape.http_methods:
                lines.append(f"assert client.get(_item({path!r}, item_id)).status_code == 404")
    return lines


# ── files ──────────────────────────────────────────────────────────────────


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(_TEMPLATES_DIR), trim_blocks=True, lstrip_blocks=True,
                       keep_trailing_newline=True)


def class_tests(model: DomainModel, http_methods: list[str]) -> list[_ClassTest]:
    """The test of each class of ``model``, by class name."""
    shape = _ApiShape(model, http_methods)
    tests = [_class_test(shape, cls) for cls in sorted(model.get_classes(), key=lambda cls: cls.name)]
    for test in tests:
        if test.skip:
            logger.warning("API test of %s skipped: %s", test.cls.name, test.skip)
    return tests


def generate_api_tests(model: DomainModel, http_methods: list[str], output_dir: str) -> None:
    """Write ``tests/conftest.py`` and ``tests/test_generated_api.py`` into the backend folder ``output_dir``."""
    folder = os.path.join(output_dir, TESTS_FOLDER)
    os.makedirs(folder, exist_ok=True)
    env = _env()
    with open(os.path.join(folder, "conftest.py"), mode="w", encoding="utf-8", newline="\n") as f:
        f.write(env.get_template("tests_conftest.py.j2").render())
    with open(os.path.join(folder, "test_generated_api.py"), mode="w", encoding="utf-8", newline="\n") as f:
        f.write(env.get_template("tests_api.py.j2").render(model=model, tests=class_tests(model, http_methods)))
