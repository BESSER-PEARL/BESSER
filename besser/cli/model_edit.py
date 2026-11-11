"""``besser model``: change a domain model in its B-UML ``.py`` file, one command per change.

Without these commands a coding agent reads the model file and writes a script that patches it.
Here each change finds where the element is written and splices the change into the text:

- The file is executed (as every command that reads it does), so the model and the top-level
  variable holding each element are known; ``ast`` gives where each one is written.
- Only the characters of the change are replaced or inserted. Everything else in the file, comments,
  blank lines and layout included, is kept byte for byte.
- A new element follows its neighbours: a variable per element when they have one (as ``besser
  convert`` writes), written inline in its set otherwise (as ``besser new`` writes); a new item in a
  set written one per line goes on a line of its own.
- An element built by the file's own code (a helper function) cannot be edited in place: the change
  is one statement after the line that defines its variable (``Deal.add_attribute(...)``), or, when
  it has none, it is refused with a message that says so. Nothing is ever written from a partial edit.

``besser.cli.commands.model`` runs the operations, checks that the new text loads, holds the change
and has no validation error the old one did not have, and only then writes the file.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import keyword
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    Association,
    Class,
    Constraint,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    Method,
    Property,
)
from besser.cli.loading import LoadedModels, ModelLoadError, load_python_text

STRUCTURAL_MODULE = "besser.BUML.metamodel.structural"
# The type names ``besser info`` prints -> the constants of the structural metamodel.
PRIMITIVE_TYPES = {
    "str": "StringType", "int": "IntegerType", "float": "FloatType", "bool": "BooleanType", "date": "DateType",
    "datetime": "DateTimeType", "time": "TimeType", "timedelta": "TimeDeltaType", "any": "AnyType",
}
_TYPE_ALIASES = {"string": "str", "integer": "int", "boolean": "bool", "double": "float"}
# A set written on one line is wrapped before it grows past this many characters.
_MAX_LINE = 110
# The method that adds an element to each collection, for an owner the file builds with its own code.
_ADDERS = {
    "attributes": "add_attribute", "methods": "add_method", "literals": "add_literal", "types": "add_type",
    "associations": "add_association", "generalizations": "add_generalization", "constraints": "add_constraint",
}

_ATTRIBUTE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>[A-Za-z_]\w*)(?P<optional>\?)?(?:\s*=\s*(?P<default>.*))?$")
_END = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>[A-Za-z_]\w*)\s*\[(?P<low>\d+|\*)(?:\.\.(?P<high>\d+|\*))?\]$")
_SIGNATURE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)\s*(?:->\s*(?P<type>[A-Za-z_]\w*))?$")


class EditError(Exception):
    """The change cannot be made: the message says why and, when it can, what to do instead."""


@dataclass(frozen=True)
class Operation:
    """One change: an operation name (``add-attribute``) and its arguments as parsed from the command line."""

    name: str
    args: dict[str, Any]

    @property
    def line(self) -> str:
        """The change as ``besser model`` takes it, for messages."""
        words = [self.name]
        for value in self.args.values():
            if isinstance(value, (list, tuple)):
                words.extend(map(str, value))
            elif value not in (None, False):
                words.append(str(value))
        return " ".join(words)


@dataclass
class Edited:
    """The text after one operation, what it did in a few words, and how to check the reloaded model holds it."""

    text: str
    summary: str
    check: Callable[[DomainModel], str | None]
    warnings: list[str] = field(default_factory=list)


def _q(text: str) -> str:
    """``text`` as a Python string literal, in double quotes."""
    return json.dumps(text, ensure_ascii=False)


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _callee(node: ast.AST) -> str | None:
    """The name a call calls: ``Property`` for ``Property(...)`` and ``structural.Property(...)``."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _name_argument(call: ast.Call) -> ast.Constant | None:
    """The string constant a constructor call gives as ``name``."""
    for keyword_ in call.keywords:
        if keyword_.arg == "name":
            value = keyword_.value
            return value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0]
    return None


def _builds(node: ast.AST, element: Any) -> bool:
    """``node`` is the constructor call of ``element``: its class (or a base) called with the element's name."""
    if _callee(node) not in {cls.__name__ for cls in type(element).__mro__}:
        return False
    name = _name_argument(node)
    return name is not None and name.value == element.name


class _Text:
    """The model file's text, its syntax tree, and the edits planned on it (applied together by ``result``)."""

    def __init__(self, text: str):
        self.text = text
        self.tree = ast.parse(text)
        self._lines = text.splitlines(keepends=True)
        self._line_starts = [0]
        for line in self._lines:
            self._line_starts.append(self._line_starts[-1] + len(line))
        self.edits: list[tuple[int, int, str, int]] = []

    # positions

    def offset(self, line: int, byte_col: int) -> int:
        """The character offset of an ``ast`` position (its column counts UTF-8 bytes)."""
        if line - 1 >= len(self._lines):
            return len(self.text)
        prefix = self._lines[line - 1].encode("utf-8")[:byte_col]
        return self._line_starts[line - 1] + len(prefix.decode("utf-8", errors="ignore"))

    def start(self, node: ast.AST) -> int:
        return self.offset(node.lineno, node.col_offset)

    def end(self, node: ast.AST) -> int:
        return self.offset(node.end_lineno, node.end_col_offset)

    def line_start(self, offset: int) -> int:
        return self.text.rfind("\n", 0, offset) + 1

    def line_end(self, offset: int) -> int:
        """The offset just after the newline of the line holding ``offset``."""
        index = self.text.find("\n", offset)
        return len(self.text) if index < 0 else index + 1

    def line_offset(self, line: int) -> int:
        """The character offset where ``line`` (1-based) starts."""
        return self._line_starts[min(line - 1, len(self._line_starts) - 1)]

    def line_of(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1

    def indent(self, offset: int) -> str:
        start = self.line_start(offset)
        return re.match(r"[ \t]*", self.text[start:]).group()

    def segment(self, node: ast.AST) -> str:
        return self.text[self.start(node):self.end(node)]

    # statements

    def top_statement(self, node: ast.AST) -> ast.stmt:
        """The top-level statement ``node`` is part of."""
        start, end = self.start(node), self.end(node)
        for statement in self.tree.body:
            if self.start(statement) <= start and end <= self.end(statement):
                return statement
        raise EditError(f"line {node.lineno}: not part of a top-level statement")

    def assignment(self, name: str) -> ast.stmt | None:
        """The last top-level statement that binds ``name`` (``name = ...`` or ``name: T = ...``)."""
        found = None
        for statement in self.tree.body:
            if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == name for target in statement.targets):
                found = statement
            elif (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                  and statement.target.id == name and statement.value is not None):
                found = statement
        return found

    def attribute_statements(self, name: str, attribute: str | None = None) -> list[ast.stmt]:
        """The top-level statements that set (``name.x = ...``) or call (``name.f(...)``) on ``name``."""
        found = []
        for statement in self.tree.body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                targets = [statement.value.func]
            else:
                continue
            if any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                   and target.value.id == name and attribute in (None, target.attr) for target in targets):
                found.append(statement)
        return found

    # edits

    def replace(self, start: int, end: int, text: str) -> None:
        self.edits.append((start, end, text, len(self.edits)))

    def insert_before(self, statement: ast.stmt, code: str, separate: bool) -> None:
        """Insert the lines ``code`` before ``statement``; ``separate``: before the comment lines just above
        it too, followed by a blank line, as a block of its own."""
        position = self.line_start(self.start(statement))
        if separate:
            while position > 0:
                previous = self.line_start(position - 1)
                if not self.text[previous:position].lstrip().startswith("#"):
                    break
                position = previous
            code += "\n"
        self.replace(position, position, code)

    def insert_after(self, statement: ast.stmt, code: str) -> None:
        position = self.line_end(self.end(statement))
        if position == len(self.text) and not self.text.endswith("\n"):
            code = "\n" + code
        self.replace(position, position, code)

    def delete_statement(self, statement: ast.stmt) -> None:
        start, end = self.start(statement), self.end(statement)
        line_start, line_end = self.line_start(start), self.line_end(end)
        after = self.text[end:line_end].strip()
        if self.text[line_start:start].strip() or (after and not after.startswith("#")):
            raise EditError(f"line {statement.lineno} holds other code too; edit it by hand")
        # A block between blank lines takes one of them along, so no gap is left behind.
        blank_before = line_start == 0 or not self.text[self.line_start(line_start - 1):line_start].strip()
        next_end = self.line_end(line_end)
        if blank_before and line_end < len(self.text) and not self.text[line_end:next_end].strip():
            line_end = next_end
        self.replace(line_start, line_end, "")

    def deleted(self) -> list[tuple[int, int]]:
        return [(start, end) for start, end, text, _ in self.edits if not text and end > start]

    def replaced(self) -> list[tuple[int, int]]:
        """The ranges of the text that an edit deletes or replaces."""
        return [(start, end) for start, end, _, _ in self.edits if end > start]

    def is_deleted(self, node: ast.AST) -> bool:
        start, end = self.start(node), self.end(node)
        return any(low <= start and end <= high for low, high in self.deleted())

    def result(self) -> str:
        """The text with every planned edit. An edit inside a deleted range is dropped (its text goes
        anyway); an insertion goes before a replacement that starts where it does."""
        unique = {edit[:3]: edit for edit in reversed(self.edits)}.values()
        edits = sorted(unique, key=lambda edit: (edit[0], edit[1] > edit[0], edit[3]))
        deletions = [(start, end) for start, end, text, _ in edits if not text and end > start]
        kept: list[tuple[int, int, str, int]] = []
        for start, end, text, order in edits:
            if any(low <= start and end <= high and (start, end) != (low, high) and not (
                    start == end and start in (low, high)) for low, high in deletions):
                continue
            if kept and start < max(edit[1] for edit in kept):
                raise EditError("two parts of the change overlap; make it in two commands")
            kept.append((start, end, text, order))
        pieces, position = [], 0
        for start, end, text, _ in kept:
            pieces.append(self.text[position:start])
            pieces.append(text)
            position = end
        pieces.append(self.text[position:])
        return "".join(pieces)

    # items of a set, list or argument list

    def _item_indent(self, items: Sequence[ast.AST]) -> str:
        """The indentation of a new item on a line of its own: that of the first item on the last item's line."""
        last_line = items[-1].lineno
        first_on_line = next(item for item in items if item.lineno == last_line)
        start = self.start(first_on_line)
        line_start = self.line_start(start)
        if not self.text[line_start:start].strip():
            return self.text[line_start:start]
        return " " * (start - line_start)

    def add_item(self, items: Sequence[ast.AST], close: int, item: str) -> None:
        """Insert ``item`` after the last of ``items``, which end before the closing bracket at ``close``."""
        last_end = self.end(items[-1])
        between = self.text[last_end:close]
        comma = re.match(r"[ \t]*,", between)
        own_lines = "\n" in between
        indent = self._item_indent(items)
        item = item.replace("\n", "\n" + indent)  # an item of several lines keeps its shape
        if comma and own_lines:
            self.replace(last_end + comma.end(), last_end + comma.end(), f"\n{indent}{item},")
        elif comma:
            self.replace(last_end + comma.end(), last_end + comma.end(), f" {item},")
        elif own_lines:
            self.replace(last_end, last_end, f",\n{indent}{item}")
        else:
            line_length = self.line_end(last_end) - self.line_start(last_end)
            if line_length + len(item) + 2 > _MAX_LINE:
                self.replace(last_end, last_end, f",\n{indent}{item}")
            else:
                self.replace(last_end, last_end, f", {item}")

    def rewrite_items(self, container: ast.AST, items: Sequence[ast.AST], texts: Sequence[str]) -> None:
        """Replace the items of the set literal ``container`` with ``texts``, laid out as the items were."""
        open_end, close = self.start(container) + 1, self.end(container) - 1
        if not items or isinstance(container, ast.Call):
            self.replace(self.start(container), self.end(container), "{" + ", ".join(texts) + "}")
            return
        if "\n" not in self.text[open_end:self.start(items[0])]:
            self.replace(open_end, close, ", ".join(texts))
            return
        indent = self._item_indent(items)
        trailing = bool(re.match(r"\s*,", self.text[self.end(items[-1]):close]))
        closing = self.text[self.line_start(close):close] if not self.text[self.line_start(close):close].strip() else ""
        body = ",\n".join(f"{indent}{text}" for text in texts) + ("," if trailing else "")
        self.replace(open_end, close, f"\n{body}\n{closing}")

    def remove_items(self, container: ast.AST, items: Sequence[ast.AST], drop: set[int]) -> None:
        """Remove the items at the indices ``drop`` from the set or list literal ``container``."""
        if len(drop) == len(items):
            empty = "set()" if isinstance(container, ast.Set) else ("()" if isinstance(container, ast.Tuple) else "[]")
            self.replace(self.start(container), self.end(container), empty)
            return
        for index in sorted(drop, reverse=True):
            node = items[index]
            start, end = self.start(node), self.end(node)
            line_start, line_end = self.line_start(start), self.line_end(end)
            rest = self.text[end:line_end]
            if not self.text[line_start:start].strip() and re.fullmatch(r"[ \t]*,?[ \t]*(#.*)?\n?", rest):
                self.replace(line_start, line_end, "")
            elif index + 1 < len(items) and index + 1 not in drop:
                self.replace(start, self.start(items[index + 1]), "")
            else:
                previous = max(i for i in range(len(items)) if i not in drop and i < index) \
                    if any(i not in drop and i < index for i in range(len(items))) else None
                if previous is None:
                    self.replace(start, self.start(items[index + 1]), "")
                else:
                    self.replace(self.end(items[previous]), end, "")


def _items(container: ast.AST) -> list[ast.expr]:
    return list(container.elts) if isinstance(container, (ast.Set, ast.List, ast.Tuple)) else []


def parse_attribute(spec: str) -> tuple[str, str, bool, str | None]:
    """``name:type``, ``name:type?`` (optional) or ``name:type = default`` -> its parts."""
    match = _ATTRIBUTE.match(spec.strip())
    if not match:
        raise EditError(f"'{spec}' is not an attribute: write name:type, name:type? (optional) or name:type=default "
                        f"(types: {', '.join(PRIMITIVE_TYPES)}, or a class or enumeration of the model)")
    return match["name"], match["type"], bool(match["optional"]), match["default"]


def parse_end(spec: str) -> tuple[str, str, int, int]:
    """``name:Type[1]``, ``name:Type[0..1]``, ``name:Type[*]``, ``name:Type[1..*]`` -> name, type, bounds."""
    match = _END.match(spec.strip())
    if not match:
        raise EditError(f"'{spec}' is not an association end: write role:Class[multiplicity], "
                        "e.g. books:Book[*] or owner:User[0..1]")
    low_text, high_text = match["low"], match["high"]
    if high_text is None:
        low, high = (0, UNLIMITED_MAX_MULTIPLICITY) if low_text == "*" else (int(low_text), int(low_text))
    else:
        if low_text == "*":
            raise EditError(f"'{spec}': the lower bound of a multiplicity is a number")
        low = int(low_text)
        high = UNLIMITED_MAX_MULTIPLICITY if high_text == "*" else int(high_text)
    if high < low or high == 0:
        raise EditError(f"'{spec}': not a multiplicity")
    return match["name"], match["type"], low, high


class ModelEditor:
    """One operation on the domain model of a ``.py`` model file's text."""

    def __init__(self, path: Path, text: str, use: Sequence[str] = ()):
        self.path = path
        self.src = _Text(text)
        try:
            loaded, namespace = load_python_text(path, text)
        except ModelLoadError as exc:
            raise EditError(str(exc)) from exc
        self.loaded = loaded
        self.namespace = namespace
        loaded_model = pick_domain_model(loaded, use)
        self.model: DomainModel = loaded_model.model
        self.model_var = loaded_model.name
        self.vars: dict[int, str] = {}
        for name, value in namespace.items():
            if not name.startswith("__") and id(value) not in self.vars and self.src.assignment(name) is not None:
                self.vars[id(value)] = name
        self.reserved: set[str] = set()
        self.needed: set[str] = set()
        self.warnings: list[str] = []

    # names

    def var_of(self, element: Any) -> str | None:
        return self.vars.get(id(element))

    def free_name(self, base: str) -> str:
        base = re.sub(r"\W", "_", base) or "element"
        if base[0].isdigit() or keyword.iskeyword(base):
            base = f"_{base}"
        taken = set(self.namespace) | self.reserved | {
            token.string for token in _tokens(self.src.text) if token.type == tokenize.NAME}
        name, number = base, 2
        while name in taken:
            name, number = f"{base}_{number}", number + 1
        self.reserved.add(name)
        return name

    def _structural(self, name: str) -> str:
        """``name`` of the structural metamodel, imported if the file does not have it."""
        if name not in self.namespace:
            self.needed.add(name)
        return name

    def _type_var_style(self, name: str) -> str:
        """The variable for a new class or enumeration, spelled as the file spells those it has."""
        types = list(self.model.types)
        same = sum(self.var_of(element) == element.name for element in types)
        lower = sum(self.var_of(element) == _snake(element.name) for element in types)
        return self.free_name(_snake(name) if lower > same else name)

    # elements

    def find_type(self, name: str, kind: type | tuple[type, ...] = (Class, Enumeration)) -> Any:
        return find_type(self.model, name, kind)

    def resolve(self, path: str) -> Any:
        return resolve(self.model, path)

    def association_of(self, end: Property) -> Association | None:
        return next((association for association in self.model.associations if end in association.ends), None)

    def container_of(self, element: Any) -> tuple[Any, str]:
        """The element's owner and the collection of it that holds the element."""
        if isinstance(element, (Class, Enumeration)):
            return self.model, "types"
        if isinstance(element, Association):
            return self.model, "associations"
        if isinstance(element, Generalization):
            return self.model, "generalizations"
        if isinstance(element, Constraint):
            return self.model, "constraints"
        if isinstance(element, Method):
            return element.owner, "methods"
        if isinstance(element, EnumerationLiteral):
            return element.owner, "literals"
        association = self.association_of(element)
        if association is not None:
            return association, "ends"
        return element.owner, "attributes"

    def call_of(self, element: Any) -> ast.Call | None:
        """The constructor call that builds ``element``: its variable's value, or an item of its owner's set."""
        var = self.var_of(element)
        if var is not None:
            statement = self.src.assignment(var)
            value = getattr(statement, "value", None)
            return value if _builds(value, element) else None
        if isinstance(element, DomainModel):
            return None
        owner, attribute = self.container_of(element)
        literal = self.collection(owner, attribute)
        for item in _items(literal) if literal is not None else ():
            if _builds(item, element):
                return item
        return None

    def collection(self, owner: Any, attribute: str) -> ast.AST | None:
        """The set or list literal (or empty ``set()``) the file gives ``owner`` as its ``attribute``."""
        var = self.var_of(owner)
        if var is not None:
            statements = [statement for statement in self.src.attribute_statements(var, attribute)
                          if isinstance(statement, (ast.Assign, ast.AnnAssign))]
            if statements:
                return self._literal(statements[-1].value)
        call = self.call_of(owner)
        if call is not None:
            for keyword_ in call.keywords:
                if keyword_.arg == attribute:
                    return self._literal(keyword_.value)
        return None

    def _literal(self, node: ast.AST, follow: bool = True) -> ast.AST | None:
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            return node
        if _callee(node) in ("set", "list") and not node.args and not node.keywords:
            return node
        if follow and isinstance(node, ast.Name):
            statement = self.src.assignment(node.id)
            if statement is not None:
                return self._literal(statement.value, follow=False)
        return None

    def describe(self, element: Any) -> str:
        if isinstance(element, (Property, Method)) and isinstance(getattr(element, "owner", None), Class):
            return f"{element.owner.name}.{element.name}"
        if isinstance(element, EnumerationLiteral) and element.owner is not None:
            return f"{element.owner.name}.{element.name}"
        if isinstance(element, DomainModel):
            return f"the model {self.model_var}"
        return f"'{getattr(element, 'name', element)}'"

    # adding

    def _define(self, var: str, code: str, anchor: ast.stmt, separate: bool) -> str:
        self.src.insert_before(anchor, f"{var} = {code}\n", separate)
        return var

    def add(self, owner: Any, attribute: str, code: str, var: Callable[[], str], need_var: bool,
            separate: bool, before: ast.stmt | None = None) -> str:
        """Add the element ``code`` builds to the ``attribute`` set of ``owner``, where the file writes it.

        ``need_var``: the element must get a variable (other code refers to it). ``var`` names it, and
        its definition goes before the statement that holds the set, or before ``before``.
        Returns how the file now refers to it: its variable or its code.
        """
        literal = self.collection(owner, attribute)
        if literal is not None:
            items = _items(literal)
            use_var = need_var or (bool(items) and all(isinstance(item, ast.Name) for item in items))
            anchor = before or self.src.top_statement(literal)
            item = self._define(var(), code, anchor, separate) if use_var else code
            if not items:  # set() or [] or ()
                brackets = "[]" if isinstance(literal, ast.List) or _callee(literal) == "list" else "{}"
                self.src.replace(self.src.start(literal), self.src.end(literal), f"{brackets[0]}{item}{brackets[1]}")
            else:
                self.src.add_item(items, self.src.end(literal) - 1, item)
            return item
        call = self.call_of(owner)
        if call is not None:
            anchor = before or self.src.top_statement(call)
            item = self._define(var(), code, anchor, separate) if need_var else code
            arguments = sorted([*call.args, *call.keywords], key=self.src.start)
            self.src.add_item(arguments, self.src.end(call) - 1, f"{attribute}={{{item}}}")
            return item
        owner_var = self.var_of(owner)
        statement = self.src.assignment(owner_var) if owner_var else None
        if statement is None:
            raise EditError(f"cannot find where {self.describe(owner)} is written in {self.path.name}; edit it by hand")
        if need_var:
            item = var()
            self.src.insert_after(statement, f"{item} = {code}\n{owner_var}.{_ADDERS[attribute]}({item})\n")
            return item
        self.src.insert_after(statement, f"{owner_var}.{_ADDERS[attribute]}({code})\n")
        return code

    def model_parts(self) -> ast.stmt | None:
        """The first statement that writes the model or its associations, generalizations or constraints:
        a new class goes before it, so that they can refer to it."""
        statements = [self.src.top_statement(literal) for literal in (
            self.collection(self.model, attribute) for attribute in ("associations", "generalizations", "constraints"))
            if literal is not None]
        model = self.src.assignment(self.model_var)
        if model is not None:
            statements.append(model)
        return min(statements, key=lambda statement: statement.lineno, default=None)

    def type_code(self, name: str) -> str:
        """How the file refers to the type ``name``: a primitive's constant or a class's variable."""
        name = _TYPE_ALIASES.get(name, name)
        if name in PRIMITIVE_TYPES:
            return self._structural(PRIMITIVE_TYPES[name])
        element = self.find_type(name)
        var = self.var_of(element)
        if var is None:
            raise EditError(f"'{name}' has no variable in {self.path.name} to refer to it by; give it one")
        return var

    def property_code(self, spec: str, existing: Iterable[str]) -> tuple[str, str, str]:
        """The ``Property(...)`` of an attribute spec, its name, and the spec as ``besser info`` prints it."""
        name, type_name, optional, default = parse_attribute(spec)
        if name in existing:
            raise EditError(f"'{name}' exists already; rename or remove it first")
        type_name = _TYPE_ALIASES.get(type_name, type_name)
        code = f"{self._structural('Property')}(name={_q(name)}, type={self.type_code(type_name)}"
        if optional:
            code += ", is_optional=True"
        if default is not None:
            code += f", default_value={self.default_code(type_name, default.strip())}"
        shown = f"{name}:{type_name}{'?' if optional else ''}" + (f"={default.strip()}" if default is not None else "")
        return code + ")", name, shown

    def default_code(self, type_name: str, text: str) -> str:
        """The Python value of a default written on the command line, checked against the attribute's type."""
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = {"true": True, "false": False}.get(text.lower(), text)
        if type_name not in PRIMITIVE_TYPES:
            enumeration = self.find_type(type_name)
            if not isinstance(enumeration, Enumeration) or str(value) not in {lit.name for lit in enumeration.literals}:
                raise EditError(f"default '{text}' is not a literal of {type_name}")
            return _q(str(value))
        if type_name in ("int", "float") and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise EditError(f"default '{text}' is not a number")
        if type_name == "bool" and not isinstance(value, bool):
            raise EditError(f"default '{text}' is not true or false")
        return repr(value) if type_name in ("int", "float", "bool") else _q(str(value))

    def class_code(self, name: str, attributes: Sequence[str], abstract: bool) -> str:
        properties = []
        names: list[str] = []
        for spec in attributes:
            code, attribute, _ = self.property_code(spec, names)
            names.append(attribute)
            properties.append(code)
        code = f"{self._structural('Class')}(name={_q(name)}"
        if abstract:
            code += ", is_abstract=True"
        if properties:
            code += ", attributes={\n" + "".join(f"    {item},\n" for item in properties) + "}"
        return code + ")"

    def finish(self) -> str:
        """The new text: the planned edits, and an import of the metamodel names it uses."""
        if self.needed:
            self._import(sorted(self.needed))
        return self.src.result()

    def _import(self, names: list[str]) -> None:
        for statement in self.src.tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.module == STRUCTURAL_MODULE \
                    and not any(alias.name == "*" for alias in statement.names):
                aliases = list(statement.names)
                text = self.src.segment(statement)
                close = self.src.end(statement) - 1 if text.rstrip().endswith(")") else None
                if close is None:
                    self.src.replace(self.src.end(aliases[-1]), self.src.end(aliases[-1]), "".join(
                        f", {name}" for name in names))
                else:
                    for name in names:
                        self.src.add_item(aliases, close, name)
                return
        imports = [statement for statement in self.src.tree.body if isinstance(statement, (ast.Import, ast.ImportFrom))]
        line = f"from {STRUCTURAL_MODULE} import {', '.join(names)}\n"
        if imports:
            self.src.insert_after(imports[-1], line)
        elif self.src.tree.body:
            first = self.src.tree.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                self.src.insert_after(first, "\n" + line)
            else:
                self.src.insert_before(first, line, separate=False)
        else:
            self.src.replace(0, 0, line)


def find_type(model: DomainModel, name: str, kind: type | tuple[type, ...] = (Class, Enumeration)) -> Any:
    """The class or enumeration of ``model`` named ``name``."""
    element = model.get_type_by_name(name)
    if element is None or not isinstance(element, kind):
        what = "class" if kind is Class else "enumeration" if kind is Enumeration else "class or enumeration"
        known = sorted(t.name for t in model.types if isinstance(t, kind))
        raise EditError(f"{model.name} has no {what} named '{name}' (it has: {', '.join(known) or 'none'})")
    return element


def resolve(model: DomainModel, path: str) -> Any:
    """The element ``path`` names: ``Class``, ``Enumeration``, ``association``, ``constraint``,
    ``Class.attribute``, ``Class.method``, ``Class.end`` (an association end) or ``Enumeration.LITERAL``."""
    parts = path.split(".")
    if len(parts) == 1:
        for collection in (model.types, model.associations, model.constraints):
            found = next((element for element in collection if element.name == path), None)
            if found is not None:
                return found
        raise EditError(f"{model.name} has no class, enumeration, association or constraint named '{path}'")
    if len(parts) != 2:
        raise EditError(f"'{path}': name an element as Name or Owner.member")
    owner = find_type(model, parts[0])
    if isinstance(owner, Enumeration):
        members = list(owner.literals)
    else:
        members = [*owner.attributes, *owner.methods, *owner.association_ends()]
    found = next((member for member in members if member.name == parts[1]), None)
    if found is None:
        raise EditError(f"{parts[0]} has no member named '{parts[1]}' "
                        f"(it has: {', '.join(sorted(m.name for m in members)) or 'none'})")
    return found


def _tokens(text: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return []


def pick_domain_model(loaded: LoadedModels, use: Sequence[str]) -> Any:
    """The domain model to edit: the one ``use`` names, or the file's only one."""
    domains = loaded.of_kind("domain")
    if use:
        domains = [model for model in domains if model.names & set(use)]
    if len(domains) == 1:
        return domains[0]
    if not domains:
        raise EditError(f"{loaded.path} has no domain model{' named ' + ', '.join(use) if use else ''}")
    raise EditError(f"{loaded.path} holds {len(domains)} domain models; name one with --use "
                    f"({', '.join(model.name for model in domains)})")


# ── the operations ───────────────────────────────────────────────────────────


def _attribute_var_base(editor: ModelEditor, cls: Class, name: str) -> str:
    """The variable of a new attribute: the prefix the class's attribute variables share, then its name."""
    names = [var for var in (editor.var_of(attribute) for attribute in cls.attributes) if var]
    prefix = ""
    if names:
        common = names[0]
        for other in names[1:]:
            while not other.startswith(common):
                common = common[:-1]
        prefix = common[:common.rfind("_") + 1] if "_" in common else ""
    if not prefix:
        prefix = f"{editor.var_of(cls) or _snake(cls.name)}_"
    return f"{prefix}{name}"


def add_attribute(editor: ModelEditor, class_name: str, attributes: Sequence[str]) -> Edited:
    cls = editor.find_type(class_name, Class)
    existing = [attribute.name for attribute in cls.all_attributes()] + [end.name for end in cls.association_ends()]
    shown = []
    for spec in attributes:
        code, name, text = editor.property_code(spec, existing)
        existing.append(name)
        shown.append(text)
        editor.add(cls, "attributes", code, lambda name=name: editor.free_name(_attribute_var_base(editor, cls, name)),
                   need_var=False, separate=False)

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(class_name)
        missing = [spec for spec in attributes if found is None or
                   parse_attribute(spec)[0] not in {attribute.name for attribute in found.attributes}]
        return f"{class_name} has no {', '.join(missing)}" if missing else None

    return Edited(editor.finish(), f"added {class_name}.{', '.join(shown)}", check)


def add_class(editor: ModelEditor, name: str, attributes: Sequence[str], parent: str | None,
              abstract: bool) -> Edited:
    if editor.model.get_type_by_name(name) is not None:
        raise EditError(f"{editor.model.name} has a type named '{name}' already")
    parent_class = editor.find_type(parent, Class) if parent else None
    var = editor._type_var_style(name)
    code = editor.class_code(name, attributes, abstract)
    editor.add(editor.model, "types", code, lambda: var, need_var=True, separate=True, before=editor.model_parts())
    if parent_class is not None:
        parent_var = editor.type_code(parent)
        generalization = f"{editor._structural('Generalization')}(general={parent_var}, specific={var})"
        editor.add(editor.model, "generalizations", generalization,
                   lambda: editor.free_name(f"{var}_is_{parent_var}"), need_var=False, separate=True)

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(name)
        if found is None:
            return f"no class {name}"
        if parent and parent not in {p.name for p in found.parents()}:
            return f"{name} is not a {parent}"
        return None

    extra = f"({parent})" if parent else ""
    with_attributes = f" with {len(attributes)} attribute{'s' if len(attributes) > 1 else ''}" if attributes else ""
    return Edited(editor.finish(), f"added class {name}{extra}{with_attributes}", check)


def set_enum(editor: ModelEditor, name: str, literals: Sequence[str]) -> Edited:
    if len(set(literals)) != len(literals) or not literals:
        raise EditError("give each literal once, at least one")
    for literal in literals:
        if not re.fullmatch(r"[A-Za-z_]\w*", literal):
            raise EditError(f"'{literal}' is not a literal name (letters, digits, _)")
    enumeration = editor.model.get_type_by_name(name)
    literal_class = editor._structural("EnumerationLiteral")
    literal_code = {literal: f"{literal_class}(name={_q(literal)})" for literal in literals}
    if enumeration is None:
        var = editor._type_var_style(name)
        code = (f"{editor._structural('Enumeration')}(name={_q(name)}, literals={{"
                + ", ".join(literal_code.values()) + "})")
        # Before the classes, whose attributes may take it as their type.
        definitions = [editor.src.assignment(v) for v in map(editor.var_of, editor.model.get_classes()) if v]
        first = min((statement for statement in definitions if statement is not None),
                    key=lambda statement: statement.lineno, default=None)
        editor.add(editor.model, "types", code, lambda: var, need_var=True, separate=True, before=first)
        summary = f"added enumeration {name}: {', '.join(literals)}"
    else:
        if not isinstance(enumeration, Enumeration):
            raise EditError(f"'{name}' is a class, not an enumeration")
        current = {literal.name for literal in enumeration.literals}
        stranded = sorted(f"{cls.name}.{attribute.name}" for cls in editor.model.get_classes()
                          for attribute in cls.attributes if attribute.type is enumeration
                          and attribute.default_value is not None
                          and getattr(attribute.default_value, "name", attribute.default_value) not in literals)
        if stranded:
            raise EditError(f"the default of {', '.join(stranded)} is a literal this removes: change it first")
        if current == set(literals):
            raise EditError(f"{name} has these literals already")
        literal_set = editor.collection(enumeration, "literals")
        items = _items(literal_set) if literal_set is not None else []
        located = {_item_name(editor, item): item for item in items}
        if literal_set is not None and current <= set(located):
            by_var = bool(items) and all(isinstance(item, ast.Name) for item in items)
            texts = []
            for literal in literals:
                if literal in located:
                    texts.append(editor.src.segment(located[literal]))
                elif by_var:
                    texts.append(editor._define(editor.free_name(f"{editor.var_of(enumeration)}_{literal}"),
                                                literal_code[literal], editor.src.top_statement(literal_set), False))
                else:
                    texts.append(literal_code[literal])
            for literal, item in located.items():
                if literal not in literals and isinstance(item, ast.Name):
                    _forget_var(editor, item.id)
            editor.src.rewrite_items(literal_set, items, texts)
        else:
            var = editor.var_of(enumeration)
            statement = editor.src.assignment(var) if var else None
            if statement is None:
                raise EditError(f"cannot find where enumeration {name} is written; edit it by hand")
            editor.src.insert_after(statement, f"{var}.literals = {{{', '.join(literal_code.values())}}}\n")
        summary = f"set {name}: {', '.join(literals)}"

    def check(model: DomainModel) -> str | None:
        found = model.get_type_by_name(name)
        names = {literal.name for literal in getattr(found, "literals", ())}
        return None if names == set(literals) else f"{name} has {', '.join(sorted(names))}"

    return Edited(editor.finish(), summary, check)


def _item_name(editor: ModelEditor, item: ast.AST) -> str | None:
    """The name of the element an item of a set builds or refers to."""
    if isinstance(item, ast.Name):
        statement = editor.src.assignment(item.id)
        value = getattr(statement, "value", None)
        item = value if isinstance(value, ast.Call) else item
        if isinstance(item, ast.Name):
            element = editor.namespace.get(item.id)
            return getattr(element, "name", None)
    if isinstance(item, ast.Call):
        name = _name_argument(item)
        return name.value if name is not None else None
    return None


def add_association(editor: ModelEditor, name: str, first: str, second: str, composite: bool) -> Edited:
    if any(association.name == name for association in editor.model.associations):
        raise EditError(f"{editor.model.name} has an association named '{name}' already")
    ends = []
    for index, spec in enumerate((first, second)):
        end_name, type_name, low, high = parse_end(spec)
        target = editor.find_type(type_name, Class)
        other_type = parse_end((second, first)[index])[1]
        owner = editor.model.get_type_by_name(other_type)
        if owner is not None and end_name in {e.name for e in owner.association_ends()} | \
                {a.name for a in owner.attributes}:
            raise EditError(f"{other_type} has a member named '{end_name}' already")
        high_code = '"*"' if high == UNLIMITED_MAX_MULTIPLICITY else str(high)
        code = (f"{editor._structural('Property')}(name={_q(end_name)}, type={editor.type_code(target.name)}, "
                f"multiplicity={editor._structural('Multiplicity')}({low}, {high_code})")
        if composite and index == 0:  # the first end is the whole
            code += ", is_composite=True"
        ends.append(code + ")")
    code = f"{editor._structural('BinaryAssociation')}(name={_q(name)}, ends={{\n    {ends[0]},\n    {ends[1]},\n}})"
    editor.add(editor.model, "associations", code, lambda: editor.free_name(name), need_var=False, separate=True)

    def check(model: DomainModel) -> str | None:
        return None if any(a.name == name for a in model.associations) else f"no association {name}"

    return Edited(editor.finish(), f"added association {name}: {first} {second}", check)


def add_method(editor: ModelEditor, class_name: str, signature: str, code: str | None) -> Edited:
    cls = editor.find_type(class_name, Class)
    match = _SIGNATURE.match(signature.strip())
    if not match:
        raise EditError(f"'{signature}' is not a signature: write name(param:type, ...) -> type")
    name = match["name"]
    if name in {method.name for method in cls.methods}:
        raise EditError(f"{class_name} has a method '{name}' already")
    parameters = []
    for spec in filter(None, (part.strip() for part in match["params"].split(","))):
        parameter, type_name, _, default = parse_attribute(spec)
        parameter_code = (f"{editor._structural('Parameter')}(name={_q(parameter)}, "
                          f"type={editor.type_code(type_name)}")
        if default is not None:
            parameter_code += f", default_value={editor.default_code(_TYPE_ALIASES.get(type_name, type_name), default)}"
        parameters.append(parameter_code + ")")
    method = f"{editor._structural('Method')}(name={_q(name)}"
    if parameters:
        method += f", parameters=[{', '.join(parameters)}]"
    if match["type"]:
        method += f", type={editor.type_code(match['type'])}"
    if code:
        method += f", code={_q(code)}"
    method += ")"
    editor.add(cls, "methods", method, lambda: editor.free_name(f"{editor.var_of(cls) or _snake(class_name)}_{name}"),
               need_var=False, separate=False)

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(class_name)
        return None if found and name in {m.name for m in found.methods} else f"{class_name} has no method {name}"

    return Edited(editor.finish(), f"added {class_name}.{name}()", check)


def add_constraint(editor: ModelEditor, class_name: str, expression: str, name: str | None) -> Edited:
    cls = editor.find_type(class_name, Class)
    existing = {constraint.name for constraint in editor.model.constraints}
    if name is None:
        number = 1
        while f"{_snake(class_name)}_rule_{number}" in existing:
            number += 1
        name = f"{_snake(class_name)}_rule_{number}"
    elif name in existing:
        raise EditError(f"{editor.model.name} has a constraint named '{name}' already")
    expression = expression.strip()
    if not expression.startswith("context "):
        expression = f"context {class_name} inv {name}: {expression}"
    code = (f"{editor._structural('Constraint')}(name={_q(name)}, context={editor.type_code(cls.name)}, "
            f"expression={_q(expression)}, language=\"OCL\")")
    editor.add(editor.model, "constraints", code, lambda: editor.free_name(name), need_var=False, separate=True)

    def check(model: DomainModel) -> str | None:
        return None if any(c.name == name for c in model.constraints) else f"no constraint {name}"

    return Edited(editor.finish(), f"added constraint {name} on {class_name}", check)


def rename(editor: ModelEditor, path: str, new_name: str) -> Edited:
    element = editor.resolve(path)
    old_name = element.name
    if not re.fullmatch(r"[A-Za-z_]\w*", new_name):
        raise EditError(f"'{new_name}' is not a name (letters, digits, _)")
    owner, attribute = editor.container_of(element)
    if new_name in _siblings(editor, element, owner, attribute):
        raise EditError(f"'{new_name}' is taken")
    var = editor.var_of(element)
    new_var = _renamed_variable(editor, var, old_name, new_name) if var is not None else None
    call = editor.call_of(element)
    name_node = _name_argument(call) if call is not None else None
    if name_node is not None:
        editor.src.replace(editor.src.start(name_node), editor.src.end(name_node), _q(new_name))
    elif var is not None and editor.src.assignment(var) is not None:
        editor.src.insert_after(editor.src.assignment(var), f"{new_var or var}.name = {_q(new_name)}\n")
    else:
        raise EditError(f"cannot find where {editor.describe(element)} is written; edit it by hand")
    if new_var is not None:
        _rename_variable(editor, var, new_var)
    _rename_in_constraints(editor, element, old_name, new_name)
    new_path = f"{path.rsplit('.', 1)[0]}.{new_name}" if "." in path else new_name
    summary = f"renamed {path} to {new_path}" + (f" (variable {var} -> {new_var})" if new_var else "")

    def check(model: DomainModel) -> str | None:
        try:
            resolve(model, new_path)
        except EditError as exc:
            return str(exc)
        return None

    return Edited(editor.finish(), summary, check, editor.warnings)


def _renamed_variable(editor: ModelEditor, var: str, old_name: str, new_name: str) -> str | None:
    """The new variable of a renamed element whose variable spells its name (``Customer``, ``customer``)."""
    spellings = {old_name: new_name, _snake(old_name): _snake(new_name), old_name.lower(): new_name.lower()}
    spelled = spellings.get(var)
    if spelled is None or spelled == var or spelled in editor.namespace or keyword.iskeyword(spelled) \
            or hasattr(builtins, spelled):
        return None
    return spelled


def _siblings(editor: ModelEditor, element: Any, owner: Any, attribute: str) -> set[str]:
    """The names a renamed element must not take: those of the elements beside it."""
    if isinstance(element, Property):
        holder = owner
        if attribute == "ends":  # an end is a member of the class at the other end
            holder = next((end.type for end in owner.ends if end is not element), element.type)
        return {a.name for a in holder.all_attributes()} | {e.name for e in holder.association_ends()}
    if isinstance(element, (Class, Enumeration)):
        return {t.name for t in editor.model.types}
    return {getattr(item, "name", None) for item in getattr(owner, attribute, ())}


def _rename_variable(editor: ModelEditor, old: str, new: str) -> None:
    """Rename the Python variable ``old`` wherever the file uses it (not after a dot, not in strings)."""
    tokens = [token for token in _tokens(editor.src.text)
              if token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT)]
    for index, token in enumerate(tokens):
        if token.type == tokenize.NAME and token.string == old and (index == 0 or tokens[index - 1].string != "."):
            start = editor.src.line_offset(token.start[0]) + token.start[1]
            editor.src.replace(start, start + len(old), new)


def _rename_in_constraints(editor: ModelEditor, element: Any, old: str, new: str) -> None:
    """Update the OCL of the constraints that name the renamed class, member or literal."""
    if isinstance(element, (Class, Enumeration)):
        pattern = re.compile(rf"\b{re.escape(old)}\b")
    elif isinstance(element, Property):
        pattern = re.compile(rf"(?<=\.){re.escape(old)}\b")
    elif isinstance(element, EnumerationLiteral):
        pattern = re.compile(rf"(?<=::){re.escape(old)}\b")
    else:
        return
    for constraint in sorted(editor.model.constraints, key=lambda c: c.name):
        if not pattern.search(constraint.expression):
            continue
        call = editor.call_of(constraint)
        node = next((k.value for k in call.keywords if k.arg == "expression"), None) if call is not None else None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            segment = editor.src.segment(node)
            editor.src.replace(editor.src.start(node), editor.src.end(node), pattern.sub(new, segment))
        else:
            editor.warnings.append(f"constraint {constraint.name} mentions '{old}': update its expression")


def _forget_var(editor: ModelEditor, var: str) -> None:
    """Delete the statements that define ``var`` or set something on it."""
    statement = editor.src.assignment(var)
    if statement is not None:
        editor.src.delete_statement(statement)
    for other in editor.src.attribute_statements(var):
        editor.src.delete_statement(other)


def _removed_elements(editor: ModelEditor, element: Any) -> tuple[list[Any], list[Any]]:
    """What removing ``element`` removes: the elements themselves (a class takes its associations,
    generalizations and constraints along), and their members."""
    model = editor.model
    outer = [element]
    if isinstance(element, Class):
        outer += sorted((a for a in model.associations if any(end.type is element for end in a.ends)),
                        key=lambda a: a.name)
        outer += [g for g in model.generalizations if element in (g.general, g.specific)]
        outer += sorted((c for c in model.constraints if c.context is element), key=lambda c: c.name)
    if isinstance(element, Enumeration):
        users = [f"{cls.name}.{attribute.name}" for cls in model.get_classes() for attribute in cls.attributes
                 if attribute.type is element]
        if users:
            raise EditError(f"{element.name} is the type of {', '.join(sorted(users))}: remove or change those first")
    members: list[Any] = []
    for item in outer:
        if isinstance(item, Class):
            members += [*item.attributes, *item.methods]
        elif isinstance(item, Association):
            members += list(item.ends)
        elif isinstance(item, Enumeration):
            members += list(item.literals)
    return outer, members


def remove(editor: ModelEditor, path: str) -> Edited:
    element = editor.resolve(path)
    if isinstance(element, Property) and editor.association_of(element) is not None:
        element = editor.association_of(element)  # an end goes with its association
    outer, members = _removed_elements(editor, element)
    removed_vars: list[str] = []
    drops: dict[int, tuple[ast.AST, set[int]]] = {}
    for item in [*outer, *members]:
        var = editor.var_of(item)
        if var is not None:
            removed_vars.append(var)
            _forget_var(editor, var)
            continue
        call = editor.call_of(item)
        if call is None:
            if item in members:  # written inside its owner, or by the code that builds it
                continue
            raise EditError(f"cannot find where {editor.describe(item)} is written (is it built by a function?); "
                            "edit it by hand")
        if editor.src.is_deleted(call):
            continue
        literal = editor.collection(*editor.container_of(item))
        index = next(i for i, node in enumerate(_items(literal)) if node is call)
        drops.setdefault(id(literal), (literal, set()))[1].add(index)
    # A removed variable leaves the set of its owner; any other use of it is the user's to change.
    for item in [*outer, *members]:
        var = editor.var_of(item)
        literal = editor.collection(*editor.container_of(item)) if var is not None else None
        for index, node in enumerate(_items(literal) if literal is not None else ()):
            if isinstance(node, ast.Name) and node.id == var:
                drops.setdefault(id(literal), (literal, set()))[1].add(index)
    for literal, indices in drops.values():
        if not editor.src.is_deleted(literal):
            editor.src.remove_items(literal, _items(literal), indices)
    deleted = editor.src.replaced()
    still_used = sorted({
        token.start[0] for token in _tokens(editor.src.text)
        if token.type == tokenize.NAME and token.string in removed_vars and not any(
            low <= editor.src.line_offset(token.start[0]) + token.start[1] < high for low, high in deleted)
    })
    if still_used:
        lines = ", ".join(map(str, still_used))
        raise EditError(f"{editor.describe(element)} is still used on line{'s' if len(still_used) > 1 else ''} "
                        f"{lines} of {editor.path.name}: change those lines first")
    named = element.name if isinstance(element, Association) and "." in path else path
    also = [f"{type(item).__name__.lower().replace('binary', '')} {item.name}" for item in outer[1:]
            if isinstance(item, (Association, Constraint))]
    summary = f"removed {named}" + (f" and {', '.join(also)}" if also else "")

    def check(model: DomainModel) -> str | None:
        try:
            resolve(model, named)
        except EditError:
            return None
        return f"{named} is still there"

    return Edited(editor.finish(), summary, check)


# Operation name -> what it does, the arguments it takes (for help and batch lines) and its function.
OPERATIONS: dict[str, tuple[str, Callable[..., Edited]]] = {
    "add-class": ("add a class: NAME [ATTRIBUTE...] [--parent CLASS] [--abstract]", add_class),
    "add-attribute": ("add attributes to a class: CLASS ATTRIBUTE...", add_attribute),
    "add-association": ("add an association: NAME END END [--composite]", add_association),
    "add-method": ("add a method: CLASS 'name(param:type, ...) -> type' [--code CODE]", add_method),
    "add-constraint": ("add an OCL invariant: CLASS EXPRESSION [--name NAME]", add_constraint),
    "set-enum": ("create an enumeration or set its literals: NAME LITERAL...", set_enum),
    "rename": ("rename a class, enumeration, member, literal, association or constraint: PATH NEW_NAME", rename),
    "remove": ("remove an element (a class with its associations): PATH", remove),
}


def run_operation(path: Path, text: str, operation: Operation, use: Sequence[str] = ()) -> Edited:
    """Apply ``operation`` to the text of the model file at ``path``.

    Raises:
        EditError: The operation does not apply to this model, or the file cannot be edited where it is needed.
    """
    editor = ModelEditor(path, text, use)
    return OPERATIONS[operation.name][1](editor, **operation.args)
