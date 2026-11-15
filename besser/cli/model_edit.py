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

When the file holds a GUI model too, its tables bound to a class (one ``FieldColumn`` per attribute)
follow the class: see "the GUI model" below; ``add-screen`` adds a screen showing one.

``besser.cli.commands.model`` runs the operations, checks that the new text loads, holds the change
and has no validation error the old one did not have, and only then writes the file.
"""

from __future__ import annotations

import ast
import builtins
import datetime
import io
import itertools
import json
import keyword
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from antlr4 import InputStream

from besser.BUML.metamodel import structural as structural_metamodel
from besser.BUML.metamodel.gui import (
    Button,
    Chart,
    Link,
    LookupColumn,
    Menu,
    MetricCard,
    Screen,
    Table,
    Text,
    ViewContainer,
)
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
    Type,
)
from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.cli.loading import LoadedModels, ModelLoadError, load_python_text
from besser.cli.ocl_check import invariant_text
from besser.utilities import sort_by_timestamp

STRUCTURAL_MODULE = "besser.BUML.metamodel.structural"
GUI_MODULE = "besser.BUML.metamodel.gui"
# The type names ``besser info`` prints -> the constants of the structural metamodel.
PRIMITIVE_TYPES = {
    "str": "StringType", "int": "IntegerType", "float": "FloatType", "bool": "BooleanType", "date": "DateType",
    "datetime": "DateTimeType", "time": "TimeType", "timedelta": "TimeDeltaType", "any": "AnyType",
}
_TYPE_ALIASES = {"string": "str", "integer": "int", "boolean": "bool", "double": "float"}
# A set written on one line is wrapped before it grows past this many characters.
_MAX_LINE = 110
# A renamed member's name in a constraint: tell which mentions are of the member by trying every choice
# of them, up to this many mentions in one constraint (2**n parses); with more, the user updates it.
_MAX_OCL_OCCURRENCES = 6
# The name a renamed member takes while the constraints are parsed: nothing else in a model has it.
_OCL_PROBE = "besser_renamed_member_"
# The method that adds an element to each collection, for an owner the file builds with its own code.
_ADDERS = {
    "attributes": "add_attribute", "methods": "add_method", "literals": "add_literal", "types": "add_type",
    "associations": "add_association", "generalizations": "add_generalization", "constraints": "add_constraint",
    "columns": "columns.append", "view_elements": "view_elements.add",
}

_ATTRIBUTE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>[A-Za-z_]\w*)(?P<optional>\?)?(?:\s*=\s*(?P<default>.*))?$")
# What set-attribute changes: the attribute notation without its name, ``:type``, ``:type?``, ``=default``
# (``=`` alone drops it), in any combination.
_ATTRIBUTE_CHANGE = re.compile(
    r"^(?:(?P<name>[A-Za-z_]\w*)\s*)?(?::\s*(?P<type>[A-Za-z_]\w*)(?P<optional>\?)?)?\s*(?:=(?P<default>.*))?$")
_END = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>[A-Za-z_]\w*)\s*\[(?P<low>\d+|\*)(?:\.\.(?P<high>\d+|\*))?\]$")
_MULTIPLICITY = re.compile(r"^\[?\s*(?P<low>\d+|\*)\s*(?:\.\.\s*(?P<high>\d+|\*)\s*)?\]?$")
# The types the generated backend cannot give a default: it writes the default into its code as text, and
# the column of the type takes only a date or time object (SQLAlchemy refuses text for it on insert).
_NO_DEFAULT_TYPES = ("date", "datetime", "time", "timedelta")
# A line as ``ast`` counts lines: "\r\n", "\r" or "\n" ends it; a form feed or U+2028 does not (they do for
# str.splitlines).
_LINE = re.compile(r"[^\r\n]*(?:\r\n?|\n)|[^\r\n]+\Z")
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
    # How to check that the reloaded GUI models (all of the file's) hold the change, when it touches them.
    gui_check: Callable[[list[Any]], str | None] | None = None


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
        self._lines = _LINE.findall(text)
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
        commented = re.match(r"[ \t]*,[ \t]*#[^\r\n]*(\r\n|\r|\n)", between)
        if commented:  # the last item's comment stays on its line
            self.replace(last_end + commented.end(), last_end + commented.end(),
                         f"{indent}{item},{commented.group(1)}")
        elif comma and own_lines:
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
                        f"(types: {', '.join(PRIMITIVE_TYPES)}, or an enumeration of the model)")
    return match["name"], match["type"], bool(match["optional"]), match["default"]


def parse_end(spec: str) -> tuple[str, str, int, int]:
    """``name:Type[1]``, ``name:Type[0..1]``, ``name:Type[*]``, ``name:Type[1..*]`` -> name, type, bounds."""
    match = _END.match(spec.strip())
    if not match:
        raise EditError(f"'{spec}' is not an association end: write role:Class[multiplicity], "
                        "e.g. books:Book[*] or owner:User[0..1]")
    return (match["name"], match["type"], *_bounds(spec, match["low"], match["high"]))


def parse_multiplicity(spec: str) -> tuple[int, int]:
    """``[1]``, ``[0..1]``, ``[*]``, ``[1..*]`` (brackets optional) -> the bounds."""
    match = _MULTIPLICITY.match(spec.strip())
    if not match:
        raise EditError(f"'{spec}' is not a multiplicity: write [1], [0..1], [*], [1..*] or [n..m]")
    return _bounds(spec, match["low"], match["high"])


def _bounds(spec: str, low_text: str, high_text: str | None) -> tuple[int, int]:
    if high_text is None:
        low, high = (0, UNLIMITED_MAX_MULTIPLICITY) if low_text == "*" else (int(low_text), int(low_text))
    else:
        if low_text == "*":
            raise EditError(f"'{spec}': the lower bound of a multiplicity is a number")
        low = int(low_text)
        high = UNLIMITED_MAX_MULTIPLICITY if high_text == "*" else int(high_text)
    if high < low or high == 0:
        raise EditError(f"'{spec}': not a multiplicity")
    return low, high


def multiplicity_text(low: int, high: int) -> str:
    """Bounds as ``besser info`` prints them: ``[1]``, ``[0..1]``, ``[*]``, ``[1..*]``."""
    high_text = "*" if high >= UNLIMITED_MAX_MULTIPLICITY else str(high)
    if low == 0 and high_text == "*":
        return "[*]"
    return f"[{low}]" if str(low) == high_text else f"[{low}..{high_text}]"


class ModelEditor:
    """One operation on the domain model of a ``.py`` model file's text."""

    def __init__(self, path: Path, text: str, use: Sequence[str] = ()):
        self.path = path
        self.use = use
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
        # Metamodel module -> the names of it the new text uses and the file does not import.
        self.needed: dict[str, set[str]] = {}
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
        return self._metamodel(STRUCTURAL_MODULE, name)

    def _gui(self, name: str) -> str:
        """``name`` of the GUI metamodel, imported if the file does not have it."""
        return self._metamodel(GUI_MODULE, name)

    def _metamodel(self, module: str, name: str) -> str:
        if name not in self.namespace:
            self.needed.setdefault(module, set()).add(name)
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
        if not isinstance(element, (Type, Association, Generalization, Constraint, Method, EnumerationLiteral,
                                    Property)):
            # A GUI element without a variable is written inside another one: its call is the one building it.
            calls = [node for node in ast.walk(self.src.tree) if _builds(node, element)]
            return calls[0] if len(calls) == 1 else None
        owner, attribute = self.container_of(element)
        literal = self.collection(owner, attribute)
        for item in _items(literal) if literal is not None else ():
            # A generalization has no name: it is the one between its two classes.
            if _builds(item, element) if not isinstance(element, Generalization) \
                    else _generalizes(self, item, element.specific, element.general):
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
        association = self.association_of(element) if isinstance(element, Property) else None
        if association is not None:  # an end is a member of the class at the other end
            holder = next((end.type for end in association.ends if end is not element), element.type)
            return f"{holder.name}.{element.name}"
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
            self.add_to_literal(literal, item)
            return item
        call = self.call_of(owner)
        # A call that passes the collection as an expression (``columns=a + b``) cannot take it again.
        if call is not None and not any(keyword_.arg == attribute for keyword_ in call.keywords):
            anchor = before or self.src.top_statement(call)
            item = self._define(var(), code, anchor, separate) if need_var else code
            arguments = sorted([*call.args, *call.keywords], key=self.src.start)
            # A table's columns are a list: their order is the page's.
            brackets = "[]" if isinstance(getattr(owner, attribute, None), list) else "{}"
            self.src.add_item(arguments, self.src.end(call) - 1, f"{attribute}={brackets[0]}{item}{brackets[1]}")
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

    def add_to_literal(self, literal: ast.AST, item: str) -> None:
        """Add the item ``item`` (code or a variable) to the set or list literal (or empty ``set()``) ``literal``."""
        items = _items(literal)
        if items:
            self.src.add_item(items, self.src.end(literal) - 1, item)
            return
        brackets = "[]" if isinstance(literal, ast.List) or _callee(literal) == "list" else "{}"
        self.src.replace(self.src.start(literal), self.src.end(literal), f"{brackets[0]}{item}{brackets[1]}")

    def model_parts(self) -> ast.stmt | None:
        """The first statement that writes the model or its associations, generalizations or constraints:
        a new class goes before it, so that they can refer to it."""
        statements = [self.src.top_statement(literal) for literal in (
            self.collection(self.model, attribute) for attribute in ("associations", "generalizations", "constraints"))
            if literal is not None]
        model = self.src.assignment(self.model_var)
        if model is not None:
            statements.append(model)
        # And before the screens of a GUI model, so that a screen added for it can refer to it.
        for gui in self.gui_models():
            for module in gui.modules:
                literal = self.collection(module, "screens")
                if literal is not None:
                    statements.append(self.src.top_statement(literal))
        return min(statements, key=lambda statement: statement.lineno, default=None)

    # the GUI models

    def gui_models(self) -> list[Any]:
        return [loaded.model for loaded in self.loaded.of_kind("gui")]

    def tables(self) -> list[Table]:
        """The tables of the file: on the screens of its GUI models, or held by a variable."""
        found: dict[int, Table] = {}
        candidates = [element for gui in self.gui_models() for screen in _screens(gui) for element in _contents(screen)]
        for element in [*candidates, *self.namespace.values()]:
            if isinstance(element, Table):
                found.setdefault(id(element), element)
        return sorted(found.values(), key=lambda table: table.name)

    def tables_of(self, cls: Class) -> list[Table]:
        """The tables of the file bound to ``cls``."""
        return [table for table in self.tables() if _bound_class(table) is cls]

    def type_code(self, name: str, kinds: tuple[type, ...] = (Class, Enumeration)) -> str:
        """How the file refers to the type ``name``: a primitive's constant or the variable of a class or
        enumeration (of ``kinds``: an attribute's type is never a class)."""
        name = _TYPE_ALIASES.get(name, name)
        if name in PRIMITIVE_TYPES:
            return self._structural(PRIMITIVE_TYPES[name])
        element = self.model.get_type_by_name(name)
        if not isinstance(element, kinds):
            primitive = _TYPE_ALIASES.get(name.lower(), name.lower())
            hint = f"did you mean {primitive}? " if primitive in PRIMITIVE_TYPES else ""
            known = ", ".join(sorted(t.name for t in self.model.types if isinstance(t, kinds))) or "none"
            described = "a class or enumeration" if Class in kinds else "an enumeration"
            raise EditError(f"'{name}' is not a type: {hint}write {', '.join(PRIMITIVE_TYPES)}, or {described} "
                            f"of {self.model.name} ({known})")
        var = self.var_of(element)
        if var is None:
            raise EditError(f"'{name}' has no variable in {self.path.name} to refer to it by; give it one")
        return var

    def property_code(self, owner: str, spec: str, existing: Iterable[str],
                      derived: bool = False) -> tuple[str, str, str]:
        """The ``Property(...)`` of an attribute spec of the class ``owner``, its name, and the spec as
        ``besser info`` prints it. ``derived``: the server computes it, so clients never send it."""
        name, type_name, optional, default = parse_attribute(spec)
        if name in existing:
            raise EditError(f"'{name}' exists already; rename or remove it first")
        type_name = _TYPE_ALIASES.get(type_name, type_name)
        self.refuse_class_type(owner, name, type_name, optional)
        code =f"{self._structural('Property')}(name={_q(name)}, type={self.type_code(type_name, (Enumeration,))}"
        if optional:
            code += ", is_optional=True"
        if default is not None:
            code += f", default_value={self.default_code(type_name, default.strip())}"
        if derived:
            code += ", is_derived=True"
        shown = f"{name}:{type_name}{'?' if optional else ''}" + (f"={default.strip()}" if default is not None else "")
        return code + ")", name, shown + (" (derived)" if derived else "")

    def refuse_class_type(self, owner: str, name: str, type_name: str, optional: bool, first: str = "") -> None:
        """Refuse a class as the type of the attribute ``name`` of ``owner``, with the association to add
        instead (after ``first``, a command to run before it)."""
        target = self.model.get_type_by_name(type_name)
        if not isinstance(target, Class):
            return
        # The backends cannot store one (validate warns), and the class may be defined after the owner.
        role = _snake(owner)
        if role in {a.name for a in target.all_attributes()} | {e.name for e in target.association_ends()}:
            role = f"{role}_{name}"
        raise EditError(f"'{type_name}' is a class: an attribute cannot hold a class, an association links "
                        f"them: {first}besser model add-association {_snake(owner)}_{name} {role}:{owner}[*] "
                        f"{name}:{type_name}[{'0..1' if optional else '1'}]")

    def default_code(self, type_name: str, text: str) -> str:
        """The Python value of a default written on the command line, checked against the attribute's type."""
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = {"true": True, "false": False}.get(text.lower(), text)
        if type_name not in PRIMITIVE_TYPES:
            enumeration = self.find_type(type_name)
            names = sorted(lit.name for lit in getattr(enumeration, "literals", ()))
            if not isinstance(enumeration, Enumeration) or str(value) not in names:
                raise EditError(f"default '{text}' is not a literal of {type_name} ({', '.join(names)})")
            return _q(str(value))
        if type_name in _NO_DEFAULT_TYPES:
            stamped = " or name it created_at or updated_at (no default): the backend stamps those itself" \
                if type_name in ("date", "datetime") else ""
            raise EditError(f"default '{text}': a {type_name} attribute takes no default. The generated backend "
                            f"writes a default as text, which its {type_name} column refuses on insert, and has "
                            f"no 'now'. Leave it without one, make it optional ({type_name}?){stamped}")
        if type_name == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise EditError(f"default '{text}' is not an int (a whole number, e.g. 0)")
        if type_name == "float" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise EditError(f"default '{text}' is not a number (e.g. 0.5)")
        if type_name == "bool" and not isinstance(value, bool):
            raise EditError(f"default '{text}' is not true or false")
        return repr(value) if type_name in ("int", "float", "bool") else _q(str(value))

    def class_code(self, name: str, attributes: Sequence[str], abstract: bool) -> str:
        properties = []
        names: list[str] = []
        for spec in attributes:
            code, attribute, _ = self.property_code(name, spec, names)
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
        for module in sorted(self.needed):
            self._import(module, sorted(self.needed[module]))
        return _without_unused_imports(self.src.text, self.src.result())

    def _import(self, module: str, names: list[str]) -> None:
        for statement in self.src.tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.module == module \
                    and not any(alias.name == "*" for alias in statement.names):
                self._add_imported(statement, names)
                return
        imports = [statement for statement in self.src.tree.body if isinstance(statement, (ast.Import, ast.ImportFrom))]
        line = f"from {module} import {', '.join(names)}\n"
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

    def _add_imported(self, statement: ast.ImportFrom, names: list[str]) -> None:
        """Add ``names`` to the import ``statement``: in order when its names are sorted, after them
        otherwise; an import on one line that would grow past ``_MAX_IMPORT_LINE`` becomes one name a line."""
        aliases = list(statement.names)
        existing = [alias.name for alias in aliases]
        in_order = existing == sorted(existing)
        text = self.src.segment(statement)
        if not text.rstrip().endswith(")"):  # from module import A, B
            items = [self.src.segment(alias) for alias in aliases]
            items = sorted([*items, *names], key=lambda item: item.split()[0]) if in_order else [*items, *names]
            line = f"from {statement.module} import {', '.join(items)}"
            indent = self.src.indent(self.src.start(statement))
            if len(indent) + len(line) > _MAX_IMPORT_LINE:
                line = f"from {statement.module} import (\n" + "".join(f"{indent}    {item},\n" for item in items) \
                    + f"{indent})"
            self.src.replace(self.src.start(statement), self.src.end(statement), line)
            return
        close = self.src.end(statement) - 1
        for name in names:
            following = next((alias for alias in aliases if alias.name > name), None) if in_order else None
            if following is None:
                self.src.add_item(aliases, close, name)
                continue
            start = self.src.start(following)
            line_start = self.src.line_start(start)
            if self.src.text[line_start:start].strip():  # on a line with others: before it, on that line
                self.src.replace(start, start, f"{name}, ")
            else:
                self.src.replace(line_start, line_start, f"{self.src.text[line_start:start]}{name},\n")


# An import on one line is wrapped, one name a line, before it grows past this many characters (the code style's
# line target).
_MAX_IMPORT_LINE = 120


def _used_names(text: str, tree: ast.Module) -> dict[str, int]:
    """How many times each name is used in ``text`` outside its top-level imports (as a name, or a word of a
    string, which a name can be looked up by)."""
    imported_lines = {line for statement in tree.body if isinstance(statement, (ast.Import, ast.ImportFrom))
                      for line in range(statement.lineno, statement.end_lineno + 1)}
    counts: dict[str, int] = {}
    for token in _tokens(text):
        if token.start[0] in imported_lines:
            continue
        words = [token.string] if token.type == tokenize.NAME else \
            re.findall(r"\w+", token.string) if token.type == tokenize.STRING else []
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    return counts


def _without_unused_imports(before: str, after: str) -> str:
    """``after`` without the metamodel names it imports and no longer uses, which ``before`` used: the
    change left them unused. A name used anywhere (a ``__main__`` block, another model, a string) stays."""
    if "\r" in after or "\r" in before:  # tokenize and ast count lines alike only on "\n"
        return after
    try:
        tree = ast.parse(after)
        used_before, used_after = _used_names(before, ast.parse(before)), _used_names(after, tree)
    except SyntaxError:
        return after
    text = _Text(after)
    for statement in tree.body:
        if not (isinstance(statement, ast.ImportFrom) and (statement.module or "").startswith("besser.BUML.metamodel")):
            continue
        aliases = list(statement.names)
        drop = {index for index, alias in enumerate(aliases) if alias.name != "*"
                and used_before.get(alias.asname or alias.name) and not used_after.get(alias.asname or alias.name)}
        try:
            if len(drop) == len(aliases):
                text.delete_statement(statement)
            elif drop:
                text.remove_items(statement, aliases, drop)
        except EditError:  # the import shares its line with other code: it stays, unused but harmless
            continue
    return text.result()


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
    ``Class.attribute``, ``Class.method``, ``Class.end`` or ``association.end`` (an association end) or
    ``Enumeration.LITERAL``."""
    parts = path.split(".")
    if len(parts) == 1:
        for collection in (model.types, model.associations, model.constraints):
            found = next((element for element in collection if element.name == path), None)
            if found is not None:
                return found
        raise EditError(f"{model.name} has no class, enumeration, association or constraint named '{path}'")
    if len(parts) != 2:
        raise EditError(f"'{path}': name an element as Name or Owner.member")
    association = next((a for a in model.associations if a.name == parts[0]), None)
    if association is not None and model.get_type_by_name(parts[0]) is None:
        members = list(association.ends)
    else:
        try:
            owner = find_type(model, parts[0])
        except EditError as exc:
            associations = ", ".join(sorted(a.name for a in model.associations)) or "none"
            raise EditError(f"{exc}; nor an association (it has: {associations})") from exc
        members = list(owner.literals) if isinstance(owner, Enumeration) else [
            *owner.attributes, *owner.methods, *owner.association_ends()]
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
    """The domain model to edit: the one ``use`` names, or the file's only one (``use`` may name only a
    model of another kind, such as the GUI model ``add-screen`` adds to)."""
    domains = loaded.of_kind("domain")
    named = [model for model in domains if model.names & set(use)]
    names_other = any(model.names & set(use) for model in loaded.models if model.kind != "domain")
    if use and (named or not names_other):
        domains = named
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


def add_attribute(editor: ModelEditor, class_name: str, attributes: Sequence[str], derived: bool = False) -> Edited:
    cls = editor.find_type(class_name, Class)
    existing = [attribute.name for attribute in cls.all_attributes()] + [end.name for end in cls.association_ends()]
    shown = []
    # A table bound to the class (or to a subclass, see _tables_following) gets a column per new attribute,
    # which refers to its variable: the variable is defined before the class and before those tables.
    tables = _tables_following(editor, cls)
    literal = editor.collection(cls, "attributes")
    anchors = [editor.src.top_statement(node) for node in (
        literal, *(editor.collection(table, "columns") for table in tables)) if node is not None]
    before = min(anchors, key=lambda statement: statement.lineno) if tables and anchors else None
    added = []
    for spec in attributes:
        code, name, text = editor.property_code(class_name, spec, existing, derived)
        existing.append(name)
        shown.append(text)
        added.append((name, editor.add(
            cls, "attributes", code, lambda name=name: editor.free_name(_attribute_var_base(editor, cls, name)),
            need_var=bool(tables), separate=False, before=before)))
    columns = _add_columns(editor, tables, added) if tables else []

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(class_name)
        missing = [spec for spec in attributes if found is None or
                   parse_attribute(spec)[0] not in {attribute.name for attribute in found.attributes}]
        return f"{class_name} has no {', '.join(missing)}" if missing else None

    summary = f"added {class_name}.{', '.join(shown)}"
    if columns:
        summary += f"; column{'s' if len(added) > 1 else ''} added to table {', '.join(columns)}"
    return Edited(editor.finish(), summary, check, editor.warnings)


def add_class(editor: ModelEditor, name: str, attributes: Sequence[str], parent: str | None,
              abstract: bool, screen: bool, title: str | None = None, route: str | None = None,
              rows: int | None = None) -> Edited:
    # An option of the screen asks for the screen: refusing it cost a session its whole batch.
    screen = screen or title is not None or route is not None or rows is not None
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
    edited = Edited(editor.finish(), f"added class {name}{extra}{with_attributes}", check)
    if not screen:
        return edited
    # The screen is added to the text with the class, as 'add-screen' would add it next.
    with_screen = add_screen(ModelEditor(editor.path, edited.text, editor.use), name, title, route, rows)
    return Edited(with_screen.text, f"{edited.summary}; {with_screen.summary}", check, with_screen.warnings,
                  gui_check=with_screen.gui_check)


def add_parent(editor: ModelEditor, class_name: str, parent: str) -> Edited:
    cls = editor.find_type(class_name, Class)
    parent_class = editor.find_type(parent, Class)
    if parent_class is cls or cls in parent_class.all_parents():
        raise EditError(f"{parent} is {class_name} or one of its subclasses: a class cannot specialize itself")
    if parent_class in cls.all_parents():
        raise EditError(f"{class_name} is a {parent} already")
    var, parent_var = editor.type_code(class_name), editor.type_code(parent)
    generalization = f"{editor._structural('Generalization')}(general={parent_var}, specific={var})"
    editor.add(editor.model, "generalizations", generalization,
               lambda: editor.free_name(f"{var}_is_{parent_var}"), need_var=False, separate=True)
    columns = _inherited_columns(editor, cls, parent_class)

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(class_name)
        return None if found and parent in {p.name for p in found.parents()} else f"{class_name} is not a {parent}"

    return Edited(editor.finish(), f"made {class_name} a {parent}{columns}", check, editor.warnings)


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
        # Before the classes and their attributes defined on their own (as the GUI starter writes them),
        # which may take it as their type: set-attribute retypes such an attribute in place.
        elements = [element for cls in editor.model.get_classes() for element in (cls, *cls.attributes)]
        definitions = [editor.src.assignment(v) for v in map(editor.var_of, elements) if v]
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
        _refuse_literal_in_constraints(editor, [lit for lit in enumeration.literals if lit.name not in literals])
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

    return Edited(editor.finish(), summary, check, editor.warnings)


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


def _rename_model(editor: ModelEditor, model: Any, new_name: str) -> Edited:
    """Rename the domain model or the GUI model itself: the name its constructor is given."""
    old_name, kind = model.name, type(model)
    call = editor.call_of(model)
    name_node = _name_argument(call) if call is not None else None
    if name_node is None:
        raise EditError(f"cannot find where the model {old_name} is named in {editor.path.name}; edit it by hand")
    editor.src.replace(editor.src.start(name_node), editor.src.end(name_node), _q(new_name))
    follows = " (the API's title and the database file follow it)" if model is editor.model else ""

    def gui_check(guis: list[Any]) -> str | None:
        return None if any(gui.name == new_name for gui in guis) else f"no GUI model named {new_name}"

    return Edited(editor.finish(), f"renamed the model {old_name} to {new_name}{follows}",
                  (lambda checked: None if checked.name == new_name else f"the model is named {checked.name}")
                  if kind is DomainModel else (lambda checked: None),
                  editor.warnings, gui_check=None if kind is DomainModel else gui_check)


def rename(editor: ModelEditor, path: str, new_name: str, keep_pages: bool = False) -> Edited:
    if not re.fullmatch(r"[A-Za-z_]\w*", new_name):
        raise EditError(f"'{new_name}' is not a name (letters, digits, _)")
    # A model by its own name, when no element of it has that name: the starter's 'Library' becomes the
    # application's name, which the API's title and the database file take.
    models = [editor.model, *editor.gui_models()]
    named = [model for model in models if model.name == path]
    if named and "." not in path:
        try:
            editor.resolve(path)
        except EditError:
            return _rename_model(editor, named[0], new_name)
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
    quoted = ""
    if isinstance(element, EnumerationLiteral):
        _rename_defaults(editor, element, new_name)
        quoted = _rename_literal_texts(editor, element, new_name)
    new_path = f"{path.rsplit('.', 1)[0]}.{new_name}" if "." in path else new_name
    summary = f"renamed {path} to {new_path}" + (f" (variable {var} -> {new_var})" if new_var else "") + quoted
    if not keep_pages:
        summary += _rename_in_gui(editor, element, old_name, new_name)
    if isinstance(element, Class):
        summary += _relink(editor, old_name, new_name)
    if isinstance(element, Class):
        summary += _ends_named_after(editor, element, old_name, new_name)

    def check(model: DomainModel) -> str | None:
        try:
            resolve(model, new_path)
        except EditError as exc:
            return str(exc)
        return None

    return Edited(editor.finish(), summary, check, editor.warnings)


def _rename_in_gui(editor: ModelEditor, element: Any, old: str, new: str) -> str:
    """Rename what the GUI derives from a renamed class or attribute; returns what changed, for the summary.

    An attribute's column label derived from its name follows it. A class's screens and tables keep
    their titles unless derived from its name (``Authors``, ``Author``): those follow it, as do the
    heading of the screen and the label of a button that opens it.
    """
    if isinstance(element, Property) and isinstance(element.owner, Class):
        return _relabel_columns(editor, element, old, new)
    if not isinstance(element, Class):
        return ""
    # The plural last, so that it wins for a name that is its own plural (News -> Articles).
    renamed = dict(zip(reversed(_derived_titles(old)), reversed(_derived_titles(new))))
    renamed[f"Go to the {_plural(old).lower()}"] = f"Go to the {_plural(new).lower()}"
    tables = editor.tables_of(element)
    changes = []
    routes: list[tuple[str, str]] = []
    # A screen's derived titles change together or not at all: its name (the page's route), heading,
    # tables and the buttons that open it, so that one page never shows two names.
    kept: list[Any] = []
    for gui in editor.gui_models():
        screens = _screens(gui)
        for screen in screens:
            contents = _contents(screen)
            if not any(table in contents for table in tables):
                continue
            new_name = renamed.get(screen.name, screen.name)
            # A route derived from the name follows it: the default (/Name), or as add-screen writes it (/name).
            written = {f"/{screen.name.lower()}": f"/{new_name.lower()}"}
            new_route = f"/{new_name}" if screen.route_path == f"/{screen.name}" else \
                written.get(screen.route_path, screen.route_path)
            clash = _clashing_screen(screens, new_name, renamed.get(screen.description, screen.description), screen,
                                     route=new_route)
            if clash is not None:
                kept += [table for table in tables if table in contents]
                editor.warnings.append(f"screen {screen.name} keeps its titles, as do its heading, tables and the "
                                       f"buttons that open it: {new_name} would be the route or page file of screen "
                                       f"{clash.name} (they ignore case)")
                continue
            changes += _replace_strings(editor, editor.call_of(screen), ("name", "component_id", "description"),
                                        renamed)
            _replace_strings(editor, editor.call_of(screen), ("route_path",), written)
            if new_route != screen.route_path:
                routes.append((screen.route_path, new_route))
            for _, _, items in _menus(editor):
                for _, item in _menu_items_to(items, screen.route_path):
                    changes += _replace_strings(editor, item, ("label",), renamed)
                    _replace_strings(editor, item, ("url",), {screen.route_path: new_route})
            for heading in contents:
                if isinstance(heading, Text):
                    changes += _replace_strings(editor, editor.call_of(heading), ("content",), renamed)
            buttons = [item for other in screens for item in _contents(other)
                       if isinstance(item, Button) and item.targetScreen is screen]
            for button in buttons:
                changes += _replace_strings(editor, editor.call_of(button), ("label", "description"), renamed)
    for table in tables:
        if table not in kept:
            changes += _replace_strings(editor, editor.call_of(table), ("title",), renamed)
    titles = sorted({f"{old_title} -> {new_title}" for old_title, new_title in changes
                     if old_title in _derived_titles(old)})
    moved = f", route {', '.join(f'{old_route} -> {new_route}' for old_route, new_route in routes)}" if routes else ""
    return f"; GUI titles {', '.join(titles)}{moved} (--keep-pages keeps them)" if titles or routes else ""


def _relink(editor: ModelEditor, old: str, new: str) -> str:
    """Point the links to the backend routes of a renamed class (``/old/...``) at its new ones; returns
    what changed, for the summary."""
    old_part, new_part = f"/{old.lower()}/", f"/{new.lower()}/"
    moved = []
    for link in {id(e): e for gui in editor.gui_models() for screen in _screens(gui) for e in _contents(screen)
                 if isinstance(e, Link) and old_part in (e.url or "")}.values():
        if _replace_strings(editor, editor.call_of(link), ("url",), {link.url: link.url.replace(old_part, new_part)}):
            moved.append(link.label or link.name)
    return f"; link {', '.join(sorted(moved))} now to {new_part}" if moved else ""


def _relabel_columns(editor: ModelEditor, attribute: Property, old: str, new: str) -> str:
    """Relabel the columns of the renamed ``attribute`` labelled after its old name, wherever the file
    writes them (in the table's list, ``table.columns.append(...)``, ``[...] + [...]``, a variable);
    returns what changed, for the summary. A column the file does not write with the attribute's
    variable keeps its label, and a warning names its table."""
    old_label, new_label = _label(old), _label(new)
    relabelled, left = [], []
    for table in _tables_inheriting(editor, attribute.owner):
        shown = sum(_columns_of(column, attribute) and column.label == old_label for column in table.columns)
        if not shown:
            continue
        found = 0
        for call in _column_calls(editor, table):
            label, *shows = (_argument(call, name, index) for index, name in enumerate(_COLUMN_ARGUMENTS[_callee(call)]))
            if any(isinstance(node, ast.Name) and editor.namespace.get(node.id) is attribute for node in shows) \
                    and isinstance(label, ast.Constant) and label.value == old_label:
                editor.src.replace(editor.src.start(label), editor.src.end(label),
                                   _string_code(editor.src.segment(label), old_label, new_label))
                found += 1
        (relabelled if found else left).append(table.name)
        if found and found < shown:
            left.append(table.name)
    if left:
        editor.warnings.append(f"the column {old_label} of table {', '.join(left)} keeps its label: it is not "
                               f"written as FieldColumn(label=..., field=<variable>); relabel it by hand")
    return f"; column {old_label} -> {new_label} in table {', '.join(relabelled)}" if relabelled else ""


def _column_calls(editor: ModelEditor, table: Table) -> list[ast.Call]:
    """The column constructor calls the file writes for ``table``: in the statements that build it or
    set and call on its variable, and the values of the variables those name."""
    var = editor.var_of(table)
    if var is not None:
        statements = [editor.src.assignment(var)]
        for statement in editor.src.tree.body:  # table.columns = [...], table.columns.append(...)
            target = statement.targets[0] if isinstance(statement, ast.Assign) else \
                getattr(statement.value, "func", None) if isinstance(statement, ast.Expr) else None
            while isinstance(target, ast.Attribute):
                target = target.value
            if isinstance(target, ast.Name) and target.id == var:
                statements.append(statement)
    else:
        statements = [editor.call_of(table)]
    nodes = [node for statement in statements if statement is not None for node in ast.walk(statement)]
    for name in [node for node in nodes if isinstance(node, ast.Name)]:
        value = getattr(editor.src.assignment(name.id), "value", None)
        if _callee(value) in _COLUMN_ARGUMENTS:
            nodes.append(value)
    return list({id(node): node for node in nodes if _callee(node) in _COLUMN_ARGUMENTS}.values())


# The arguments of a column's constructor, in order: its label, then the attributes it shows.
_COLUMN_ARGUMENTS = {"FieldColumn": ("label", "field"), "LookupColumn": ("label", "path", "field")}


def _argument(call: ast.Call, name: str, index: int) -> ast.expr | None:
    """The argument ``name`` of ``call``, given by keyword or at position ``index``."""
    found = next((keyword_.value for keyword_ in call.keywords if keyword_.arg == name), None)
    return found if found is not None or index >= len(call.args) else call.args[index]


# Plurals no rule spells (``_plural``), and words that are their own plural.
_PLURALS = {
    "person": "people", "man": "men", "woman": "women", "child": "children", "mouse": "mice", "goose": "geese",
    "foot": "feet", "tooth": "teeth", "ox": "oxen", "leaf": "leaves", "loaf": "loaves", "thief": "thieves",
    "criterion": "criteria", "phenomenon": "phenomena", "datum": "data", "medium": "media",
}
_UNCOUNTABLE = {
    "news", "series", "species", "sheep", "fish", "deer", "equipment", "information", "software", "hardware",
    "feedback", "staff", "metadata", "furniture", "luggage", "baggage", "advice", "music",
}


def _plural(word: str) -> str:
    """The English plural of a class or role name (``books``, ``boxes``, ``categories``, ``shelves``,
    ``People``, ``SalesPeople``); a word already plural or uncountable (``News``, ``Books``) is kept.

    Only the last word of a compound name (``SalesPerson``, ``order_line``) changes.
    """
    match = re.search(r"([A-Z]?[a-z]+|[A-Z]+)$", word)
    if match is None:  # ends in a digit or "_"
        return word + "s"
    head, last = word[:match.start()], match.group()
    lower = last.lower()
    if lower in _PLURALS:
        plural = _PLURALS[lower]
        return head + (plural.upper() if last.isupper() and len(last) > 1 else
                       plural.capitalize() if last[0].isupper() else plural)
    if lower in _PLURALS.values() or lower in _UNCOUNTABLE:
        return word
    if lower.endswith("is"):
        return word[:-2] + "es"  # analysis -> analyses
    if lower.endswith(("ss", "us", "as", "os", "sh", "ch", "x", "z")):
        return word + "es"
    if lower.endswith("s"):  # Books, Categories, Series
        return word
    if re.search(r"[^aeiou]y$", lower):
        return word[:-1] + "ies"
    if lower.endswith("lf"):  # shelf, wolf, half
        return word[:-1] + "ves"
    if lower.endswith("ife"):  # knife, wife, life
        return word[:-2] + "ves"
    return word + "s"


def _ends_named_after(editor: ModelEditor, cls: Class, old: str, new: str) -> str:
    """A hint naming the ends of the renamed class ``cls`` still named after its old name ``old``
    (``authors`` after Author -> Writer), with the command that renames each; empty when there are none."""
    spellings = {_snake(old): _snake(new), old.lower(): new.lower(), old: new}
    commands = []
    for association in sorted(editor.model.associations, key=lambda a: a.name):
        for end in sorted(association.ends, key=lambda e: e.name):
            if end.type is not cls:
                continue
            holder = next((other.type for other in association.ends if other is not end), cls)
            renamed = next((named for old_form, new_form in spellings.items()
                            for named in ((old_form, new_form), (_plural(old_form), _plural(new_form)))
                            if end.name == named[0]), None)
            if renamed is not None:
                commands.append((f"{new if holder is cls else holder.name}.{end.name}", renamed[1]))
    if not commands:
        return ""
    return (f"; ends still named after {old}: {', '.join(described for described, _ in commands)} ("
            + "; ".join(f"rename {described} {name}" for described, name in commands) + ")")


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
    """Rename the Python variable ``old`` wherever the file uses it (see ``_references``)."""
    for node in _references(editor, {old}):
        start = editor.src.start(node)
        editor.src.replace(start, start + len(old), new)


def _references(editor: ModelEditor, names: Iterable[str]) -> list[ast.Name]:
    """Where the file refers to the variables ``names``: not a keyword argument's name (``title=``),
    an attribute after a dot (``x.title``) or text in a string, which only spell the same word."""
    names = set(names)
    return [node for node in ast.walk(editor.src.tree) if isinstance(node, ast.Name) and node.id in names]


def _ocl_tokens(expression: str) -> list[Any]:
    """The tokens of an OCL text (strings are one token each, so a name inside quotes is no name)."""
    lexer = BOCLLexer(InputStream(expression))
    lexer.removeErrorListeners()
    return [token for token in lexer.getAllTokens() if token.channel == 0]


def _parses(text: str, model: DomainModel, context: Class) -> bool:
    try:
        parse_ocl(text, model, context_class=context)
    except Exception:  # noqa: BLE001 - any failure: the text does not hold against the model
        return False
    return True


def _member_occurrences(editor: ModelEditor, member: Property, constraint: Constraint,
                        candidates: list[Any]) -> list[Any] | None:
    """Which of ``candidates`` (``.name`` tokens) navigate to ``member``; ``None`` when that cannot be told.

    Two classes may have a member of the same name, so the text alone does not say. The parse does:
    with ``member`` renamed to a name nothing else has, exactly one choice of the tokens to rename
    still parses.
    """
    text = invariant_text(constraint)
    shift = len(text) - len(constraint.expression)
    if len(candidates) > _MAX_OCL_OCCURRENCES or not _parses(text, editor.model, constraint.context):
        return None
    old = member.name
    member.name = _OCL_PROBE
    try:
        fits = []
        for size in range(len(candidates) + 1):
            for chosen in itertools.combinations(candidates, size):
                probe = _replace_tokens(text, [(token.start + shift, token.stop + shift) for token in chosen],
                                        _OCL_PROBE)
                if _parses(probe, editor.model, constraint.context):
                    fits.append(list(chosen))
    finally:
        member.name = old
    return fits[0] if len(fits) == 1 else None


def _constraints_using(editor: ModelEditor, member: Property, removed: Sequence[Any]) -> list[str]:
    """The names of the constraints (other than ``removed``) that navigate to ``member``.

    One does when it parses, and no longer does with the member under a name nothing else has.
    """
    users = []
    for constraint in sorted(editor.model.constraints, key=lambda c: c.name):
        if constraint in removed or not isinstance(constraint.context, Class):
            continue
        tokens = _ocl_tokens(constraint.expression or "")
        if not any(token.text == member.name and index and tokens[index - 1].type == BOCLLexer.DOT
                   for index, token in enumerate(tokens)):
            continue
        text = invariant_text(constraint)
        if not _parses(text, editor.model, constraint.context):
            continue
        old = member.name
        member.name = _OCL_PROBE
        try:
            if not _parses(text, editor.model, constraint.context):
                users.append(constraint.name)
        finally:
            member.name = old
    return users


def _replace_tokens(text: str, spans: Iterable[tuple[int, int]], new: str) -> str:
    """``text`` with each (start, inclusive stop) span replaced by ``new``."""
    for start, stop in sorted(spans, reverse=True):
        text = text[:start] + new + text[stop + 1:]
    return text


def _renamed_tokens(editor: ModelEditor, element: Any, constraint: Constraint, old: str) -> list[Any] | None:
    """The tokens of ``constraint`` that name ``element``, renamed from ``old``; ``None`` when unknown."""
    tokens = _ocl_tokens(constraint.expression)
    previous = [None, None, *(token.type for token in tokens)]
    named = [(index, token) for index, token in enumerate(tokens) if token.text == old]
    if isinstance(element, (Class, Enumeration)):  # a type name, not a member after "." or "->"
        return [token for index, token in named if previous[index + 1] not in (BOCLLexer.DOT, BOCLLexer.ARROW)]
    if isinstance(element, EnumerationLiteral):  # Enum::LITERAL of this enumeration only
        return [token for index, token in named if previous[index + 1] == BOCLLexer.DOUBLECOLON
                and index >= 2 and tokens[index - 2].text == element.owner.name]
    if isinstance(element, Constraint):  # its own "inv name"
        return [token for index, token in named if previous[index + 1] == BOCLLexer.INV] \
            if constraint is element else []
    if isinstance(element, Property):
        candidates = [token for index, token in named if previous[index + 1] == BOCLLexer.DOT]
        return _member_occurrences(editor, element, constraint, candidates) if candidates else []
    return []


def _rename_in_constraints(editor: ModelEditor, element: Any, old: str, new: str) -> None:
    """Update the OCL of the constraints that name the renamed class, member, literal or constraint.

    Only names are rewritten, never text in quotes; a member's name only where it navigates to that
    member (see ``_member_occurrences``). A constraint left as it was and broken by the rename is
    caught by the validation the edit has to pass.
    """
    for constraint in sorted(editor.model.constraints, key=lambda c: c.name):
        if old not in (constraint.expression or ""):
            continue
        tokens = _renamed_tokens(editor, element, constraint, old)
        if tokens == []:
            continue
        call = editor.call_of(constraint)
        node = next((k.value for k in call.keywords if k.arg == "expression"), None) if call is not None else None
        if tokens is None or not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            editor.warnings.append(f"constraint {constraint.name} mentions '{old}': update its expression")
            continue
        expression = _replace_tokens(node.value, [(token.start, token.stop) for token in tokens], new)
        editor.src.replace(editor.src.start(node), editor.src.end(node), _string_code(editor.src.segment(node),
                                                                                       node.value, expression))


def _string_code(segment: str, value: str, new_value: str) -> str:
    """``new_value`` as a Python string literal, quoted as ``segment`` (the literal of ``value``) is when it can."""
    quote = segment[:1]
    if quote in ("'", '"') and segment[1:-1] == value and quote not in new_value and "\\" not in new_value \
            and "\n" not in new_value:
        return f"{quote}{new_value}{quote}"
    return _q(new_value)


def _rename_defaults(editor: ModelEditor, literal: EnumerationLiteral, new: str) -> None:
    """Rename the literal in the defaults of the attributes its enumeration types."""
    enumeration = literal.owner
    for cls in sorted(editor.model.get_classes(), key=lambda c: c.name):
        for attribute in sorted(cls.attributes, key=lambda a: a.name):
            default = attribute.default_value
            if attribute.type is not enumeration or not isinstance(default, str):
                continue
            if default.strip().removeprefix(f"{enumeration.name}.") != literal.name:
                continue
            call = editor.call_of(attribute)
            node = next((k.value for k in call.keywords if k.arg == "default_value"), None) \
                if call is not None else None
            if not (isinstance(node, ast.Constant) and node.value == default):
                editor.warnings.append(f"the default of {cls.name}.{attribute.name} is {literal.name}: "
                                       f"change it to {new}")
                continue
            editor.src.replace(editor.src.start(node), editor.src.end(node),
                               _string_code(editor.src.segment(node), default,
                                            re.sub(rf"\b{re.escape(literal.name)}(?=\s*$)", new, default)))


# ── an enumeration literal in OCL text ───────────────────────────────────────
#
# OCL compares an enumeration attribute to the literal's name in quotes (self.status <> 'Done'): the text
# names the literal, so a rename or a removal of the literal changes what the constraint means.

_COMPARISONS = (BOCLLexer.EQUAL, BOCLLexer.NOTEQUAL)


def _compared_navigation(tokens: Sequence[Any], index: int) -> list[str] | None:
    """The navigation the quoted text at ``index`` is compared to (``x.status = 'Done'``,
    ``'Done' <> self.book.status``): its start (``self`` or a variable) and the names after it, or ``None``
    when it is compared to something else."""
    if index >= 3 and tokens[index - 1].type in _COMPARISONS:
        position, names = index - 2, []
        while position >= 1 and tokens[position].type == BOCLLexer.ID and tokens[position - 1].type == BOCLLexer.DOT:
            names.insert(0, tokens[position].text)
            position -= 2
        if names and tokens[position].type in (BOCLLexer.SELF, BOCLLexer.ID):
            return [tokens[position].text, *names]
    position = index + 2
    if index + 1 < len(tokens) and tokens[index + 1].type in _COMPARISONS and position < len(tokens) \
            and tokens[position].type in (BOCLLexer.SELF, BOCLLexer.ID):
        navigation = [tokens[position].text]
        while position + 2 < len(tokens) and tokens[position + 1].type == BOCLLexer.DOT \
                and tokens[position + 2].type == BOCLLexer.ID:
            navigation.append(tokens[position + 2].text)
            position += 2
        following = tokens[position + 1].type if position + 1 < len(tokens) else None
        if len(navigation) > 1 and following not in (BOCLLexer.LPAREN, BOCLLexer.ARROW):
            return navigation
    return None


def _literal_texts(editor: ModelEditor, literal: EnumerationLiteral) -> tuple[dict[str, list[Any]], list[str]]:
    """The quoted names of ``literal`` in the constraints, compared to an attribute its enumeration types
    (constraint name -> tokens), and the constraints that hold the quoted name where that cannot be told."""
    quoted = f"'{literal.name}'"
    classes = editor.model.get_classes()
    found: dict[str, list[Any]] = {}
    unclear = []
    for constraint in sorted(editor.model.constraints, key=lambda c: c.name):
        if quoted not in (constraint.expression or ""):
            continue
        tokens = _ocl_tokens(constraint.expression)
        mine, doubt = [], False
        for index, token in enumerate(tokens):
            if token.type != BOCLLexer.STRING_LITERAL or token.text != quoted:
                continue
            navigation = _compared_navigation(tokens, index)
            if navigation is not None and navigation[0] == "self" and len(navigation) == 2:
                # An attribute of the context class: its type is known.
                owners = [constraint.context] if isinstance(constraint.context, Class) else classes
                types = {id(a.type): a.type for c in owners for a in c.all_attributes() if a.name == navigation[1]}
            else:
                types = {id(a.type): a.type for c in classes for a in c.attributes
                         if navigation is not None and a.name == navigation[-1]}
            if navigation is None or (literal.owner in types.values() and len(types) > 1):
                doubt = True
            elif literal.owner in types.values():
                mine.append(token)
        if doubt:
            unclear.append(constraint.name)
        elif mine:
            found[constraint.name] = mine
    return found, unclear


def _rename_literal_texts(editor: ModelEditor, literal: EnumerationLiteral, new: str) -> str:
    """Rename the quoted name of a renamed literal in the constraints that compare it to an attribute of its
    enumeration; returns what changed, for the summary. Where that cannot be told, a warning names the constraint."""
    found, unclear = _literal_texts(editor, literal)
    by_name = {constraint.name: constraint for constraint in editor.model.constraints}
    changed = []
    for name, tokens in found.items():
        constraint = by_name[name]
        call = editor.call_of(constraint)
        node = next((k.value for k in call.keywords if k.arg == "expression"), None) if call is not None else None
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            unclear.append(name)
            continue
        start, end = editor.src.start(node), editor.src.end(node)
        # Literal::NAME in the same text, renamed by _rename_in_constraints: one edit makes both changes.
        spans = [(token.start, token.stop, f"'{new}'") for token in tokens]
        spans += [(token.start, token.stop, new) for token in _renamed_tokens(editor, literal, constraint,
                                                                              literal.name) or ()]
        expression = node.value
        for low, high, text in sorted(spans, reverse=True):
            expression = expression[:low] + text + expression[high + 1:]
        editor.src.edits = [edit for edit in editor.src.edits if edit[:2] != (start, end)]
        editor.src.replace(start, end, _string_code(editor.src.segment(node), node.value, expression))
        changed.append(name)
    for name in sorted(set(unclear)):
        editor.warnings.append(f"constraint {name} holds '{literal.name}' where it may not mean "
                               f"{literal.owner.name}.{literal.name}: check it says '{new}' where it should")
    return f"; constraint {', '.join(changed)}: '{literal.name}' -> '{new}'" if changed else ""


def _refuse_literal_in_constraints(editor: ModelEditor, literals: Sequence[EnumerationLiteral]) -> None:
    """Refuse removing ``literals`` a constraint names; warn of a constraint that may name one."""
    for literal in literals:
        found, unclear = _literal_texts(editor, literal)
        users = set(found) | {constraint.name for constraint in editor.model.constraints
                              if literal.name in (constraint.expression or "")
                              and _renamed_tokens(editor, literal, constraint, literal.name)}
        if users:
            raise EditError(f"constraint {', '.join(sorted(users))} compares to {literal.owner.name}.{literal.name}: "
                            f"change it first, or remove it (besser model remove {sorted(users)[0]})")
        editor.warnings.extend(f"constraint {name} holds '{literal.name}' where it may mean "
                               f"{literal.owner.name}.{literal.name}: check it" for name in unclear)


# ── changing an element in place ─────────────────────────────────────────────
#
# The element stays the object it was, written where it was: what refers to it (constraints, table columns,
# objects, method code) keeps referring to it. What the change would break is refused, named.

# The type names of besser info -> the types of the metamodel.
_PRIMITIVES = {name: getattr(structural_metamodel, constant) for name, constant in PRIMITIVE_TYPES.items()}
# What an OCL expression does with a value of a type; a constraint written for one kind does not hold for
# another. An enumeration's value is compared to quoted text, as a string is.
_VALUE_KINDS = {"int": "number", "float": "number", "str": "text", "bool": "boolean", "date": "time",
                "datetime": "time", "time": "time", "timedelta": "duration"}
# The Python values an object's attribute holds, per type.
_VALUE_CLASSES: dict[str, tuple[type, ...]] = {
    "int": (int,), "float": (float,), "str": (str,), "bool": (bool,), "date": (datetime.date,),
    "datetime": (datetime.datetime,), "time": (datetime.time,), "timedelta": (datetime.timedelta,),
}
# Property(name, type, owner, multiplicity, visibility, is_composite, is_navigable, is_id, is_read_only,
# is_optional, is_external_id, default_value): where an argument is when it is given by position.
_PROPERTY_POSITIONS = {"type": 1, "multiplicity": 3, "is_optional": 9, "default_value": 11}


def _type_name(type_: Any) -> str:
    return getattr(type_, "name", str(type_))


def _value_kind(type_: Any) -> str:
    return "text" if isinstance(type_, Enumeration) else _VALUE_KINDS.get(_type_name(type_), _type_name(type_))


def _default_text(value: Any) -> str:
    """A default as the command line writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(getattr(value, "name", value))


def _fits(value: Any, type_: Any) -> bool:
    """Whether an object's value is a value of ``type_``."""
    if value is None or _type_name(type_) == "any":
        return True
    if isinstance(type_, Enumeration):
        return str(getattr(value, "name", value)) in {literal.name for literal in type_.literals}
    classes = _VALUE_CLASSES.get(_type_name(type_), (object,))
    if isinstance(value, bool) and bool not in classes:
        return False
    if isinstance(value, datetime.datetime) and datetime.datetime not in classes:
        return False  # a datetime is a date to Python, not to a date column
    return isinstance(value, classes)


def _objects(editor: ModelEditor) -> list[tuple[str, Any]]:
    """The objects of the file's object models, with the name of their model."""
    return [(loaded.name, obj) for loaded in editor.loaded.of_kind("object") for obj in loaded.model.objects]


def _object_name(obj: Any) -> str:
    return str(getattr(obj, "name_", None) or obj.name)


def _unparsed(editor: ModelEditor) -> set[str]:
    """The constraints of the model that do not parse against it."""
    return {constraint.name for constraint in editor.model.constraints
            if isinstance(constraint.context, Class) and str(constraint.language).upper() == "OCL"
            and not _parses(invariant_text(constraint), editor.model, constraint.context)}


def _object_errors(editor: ModelEditor) -> set[str]:
    return {error for loaded in editor.loaded.of_kind("object")
            for error in loaded.model.validate(raise_exception=False)["errors"]}


def _broken_by(editor: ModelEditor, apply: Callable[[], None], undo: Callable[[], None]) -> tuple[list[str], list[str]]:
    """What a change, made on the loaded models by ``apply`` (and undone by ``undo``), breaks: the
    constraints that no longer parse, and the new errors of the object models."""
    constraints, errors = _unparsed(editor), _object_errors(editor)
    apply()
    try:
        return sorted(_unparsed(editor) - constraints), sorted(_object_errors(editor) - errors)
    finally:
        undo()


def _refuse_breaking(editor: ModelEditor, what: str, constraints: Sequence[str], errors: Sequence[str]) -> None:
    if constraints:
        raise EditError(f"constraint {', '.join(constraints)} would no longer hold against the model once {what}: "
                        f"change it first, or remove it (besser model remove {constraints[0]})")
    if errors:
        raise EditError(f"the object model would be invalid once {what}: {errors[0]} Change the objects first")


def _constructor_argument(call: ast.Call, name: str) -> ast.AST | None:
    """The argument ``name`` of a constructor call: its keyword, or its positional argument."""
    for keyword_ in call.keywords:
        if keyword_.arg == name:
            return keyword_
    position = _PROPERTY_POSITIONS.get(name) if _callee(call) == "Property" else None
    return call.args[position] if position is not None and position < len(call.args) else None


def _set_field(editor: ModelEditor, element: Any, field_name: str, code: str | None, default: str) -> None:
    """Make the field ``field_name`` of ``element`` the value ``code`` where the file writes it; ``None`` puts
    it back to its default (whose code is ``default``): the argument goes."""
    var = editor.var_of(element)
    later = [statement for statement in editor.src.attribute_statements(var, field_name)
             if isinstance(statement, (ast.Assign, ast.AnnAssign))] if var else []
    if later:  # element.field = ... after the constructor: that statement is what holds
        value = later[-1].value
        editor.src.replace(editor.src.start(value), editor.src.end(value), code if code is not None else default)
        return
    call = editor.call_of(element)
    if call is not None:
        node = _constructor_argument(call, field_name)
        arguments = sorted([*call.args, *call.keywords], key=editor.src.start)
        if isinstance(node, ast.keyword) and code is None:
            editor.src.remove_items(call, arguments, {arguments.index(node)})
        elif node is not None:
            value = node.value if isinstance(node, ast.keyword) else node
            editor.src.replace(editor.src.start(value), editor.src.end(value), code if code is not None else default)
        elif code is not None:
            editor.src.add_item(arguments, editor.src.end(call) - 1, f"{field_name}={code}")
        return
    statement = editor.src.assignment(var) if var else None
    if statement is None:
        raise EditError(f"cannot find where {editor.describe(element)} is written; edit it by hand")
    editor.src.insert_after(statement, f"{var}.{field_name} = {code if code is not None else default}\n")


def _attribute_text(type_name: str, optional: bool, default: str | None) -> str:
    """An attribute's type, ``?`` and default as add-attribute writes them, without the name."""
    return f"{type_name}{'?' if optional else ''}" + (f"={default}" if default is not None else "")


def set_attribute(editor: ModelEditor, path: str, spec: str) -> Edited:
    attribute = editor.resolve(path)
    if isinstance(attribute, Property) and editor.association_of(attribute) is not None:
        raise EditError(f"{path} is an association end: besser model set-multiplicity {path} "
                        f"{multiplicity_text(attribute.multiplicity.min, attribute.multiplicity.max)}")
    if not isinstance(attribute, Property) or not isinstance(attribute.owner, Class):
        raise EditError(f"'{path}' is not an attribute: name one as Class.attribute")
    cls, name = attribute.owner, attribute.name
    match = _ATTRIBUTE_CHANGE.match(spec.strip())
    example = f"e.g. besser model set-attribute {path} :int?=0"
    if not match or not (match["type"] or match["default"] is not None):
        raise EditError(f"'{spec}' changes nothing: write the parts to change as add-attribute writes them, "
                        f"without the name: :type, :type? (optional), =default, = (no default); {example}")
    if match["name"] not in (None, name):
        raise EditError(f"'{spec}' names another attribute: write :type, :type? or =default, the name stays "
                        f"(besser model rename {path} {match['name']} renames it)")
    old_type, old_optional, old_default = attribute.type, attribute.is_optional, attribute.default_value
    old_text = _attribute_text(_type_name(old_type), old_optional,
                               None if old_default is None else _default_text(old_default))
    type_name = _TYPE_ALIASES.get(match["type"], match["type"]) if match["type"] else _type_name(old_type)
    optional = bool(match["optional"]) if match["type"] else old_optional
    if match["type"]:
        editor.refuse_class_type(cls.name, name, type_name, optional, first=f"besser model remove {path}, then ")
        type_code = editor.type_code(type_name, (Enumeration,))
    new_type = _PRIMITIVES.get(type_name) or editor.model.get_type_by_name(type_name)
    retyped = new_type is not old_type
    default_given = match["default"].strip() if match["default"] is not None else None
    if default_given == "" and old_default is None:
        raise EditError(f"{path} has no default to drop")
    default = None if default_given == "" else default_given
    if default_given is None and old_default is not None:
        default = _default_text(old_default)
    default_code = None
    if default is not None and (default_given is not None or retyped):
        try:
            default_code = editor.default_code(type_name, default)
        except EditError as exc:
            if default_given is not None:
                raise
            q = "?" if optional else ""
            raise EditError(f"the default {default} of {path} does not fit {type_name} ({exc}): give one "
                            f"(besser model set-attribute {path} :{type_name}{q}=VALUE) or drop it "
                            f"(besser model set-attribute {path} :{type_name}{q}=)") from exc
    new_text = _attribute_text(type_name, optional, default)
    if new_text == old_text and not retyped:
        raise EditError(f"{path} is {old_text} already")
    if retyped:
        kinds = {_value_kind(old_type), _value_kind(new_type)}
        users = _constraints_using(editor, attribute, []) if len(kinds) == 2 and "any" not in kinds else []
        if users:
            raise EditError(f"constraint {', '.join(users)} uses {path} as {_type_name(old_type)} and would not "
                            f"hold for {type_name}: change it first, or remove it (besser model remove {users[0]})")
        wrong = [f"{_object_name(obj)} ({name}={_default_text(slot.value.value)})" for _, obj in _objects(editor)
                 for slot in obj.slots if slot.attribute is attribute and not _fits(slot.value.value, new_type)]
        if wrong:
            raise EditError(f"object {', '.join(sorted(wrong))} holds no {type_name}: change the value first")

        def retype() -> None:
            attribute.type = new_type

        def undo() -> None:
            attribute.type = old_type

        _refuse_breaking(editor, f"{path} is {type_name}", *_broken_by(editor, retype, undo))
        _set_field(editor, attribute, "type", type_code, "")
    if optional != old_optional:
        _set_field(editor, attribute, "is_optional", "True" if optional else None, "False")
    if default_given is not None or (retyped and default is not None):
        _set_field(editor, attribute, "default_value", default_code, "None")
    wanted_default = None if default is None else (
        ast.literal_eval(default_code) if default_code is not None else old_default)

    def check(model: DomainModel) -> str | None:
        try:
            found = resolve(model, f"{cls.name}.{name}")
        except EditError as exc:
            return str(exc)
        held = _attribute_text(_type_name(found.type), found.is_optional,
                               None if found.default_value is None else _default_text(found.default_value))
        wanted = _attribute_text(type_name, optional, None if wanted_default is None else _default_text(wanted_default))
        return None if held == wanted else f"{cls.name}.{name} is {held}"

    return Edited(editor.finish(), f"changed {cls.name}.{name} from {old_text} to {new_text}", check)


def set_multiplicity(editor: ModelEditor, path: str, multiplicity: str) -> Edited:
    end = editor.resolve(path)
    association = editor.association_of(end) if isinstance(end, Property) else None
    if association is None:
        raise EditError(f"'{path}' is not an association end: name one as Class.end or association.end")
    low, high = parse_multiplicity(multiplicity)
    old_low, old_high = end.multiplicity.min, end.multiplicity.max
    old_text, new_text = multiplicity_text(old_low, old_high), multiplicity_text(low, high)
    if old_text == new_text:
        raise EditError(f"{path} is {old_text} already")
    old_multiplicity = end.multiplicity

    def apply() -> None:
        end.multiplicity = structural_metamodel.Multiplicity(low, high)

    def undo() -> None:
        end.multiplicity = old_multiplicity

    _refuse_breaking(editor, f"{path} is {new_text}", *_broken_by(editor, apply, undo))
    high_code = '"*"' if high >= UNLIMITED_MAX_MULTIPLICITY else str(high)
    call = editor.call_of(end)
    node = _constructor_argument(call, "multiplicity") if call is not None else None
    value = node.value if isinstance(node, ast.keyword) else node
    if _callee(value) == "Multiplicity" and not value.keywords and len(value.args) == 2:
        # Only the bounds change: Multiplicity(0, "*") -> Multiplicity(1, "*"), quoted as the file quotes.
        bound_low, bound_high = value.args
        star = editor.src.segment(bound_high) if isinstance(bound_high, ast.Constant) and bound_high.value == "*" \
            else high_code
        editor.src.replace(editor.src.start(bound_low), editor.src.end(bound_high),
                           f"{low}, {star if high >= UNLIMITED_MAX_MULTIPLICITY else high}")
    else:
        _set_field(editor, end, "multiplicity", f"{editor._structural('Multiplicity')}({low}, {high_code})", "")
    holder = next((other.type for other in association.ends if other is not end), end.type)

    def check(model: DomainModel) -> str | None:
        found = next((e for a in model.associations if a.name == association.name for e in a.ends
                      if e.name == end.name), None)
        held = multiplicity_text(found.multiplicity.min, found.multiplicity.max) if found is not None else "missing"
        return None if held == new_text else f"{path} is {held}"

    return Edited(editor.finish(), f"changed {holder.name}.{end.name} ({association.name}) from {old_text} to "
                                   f"{new_text}", check)


def set_abstract(editor: ModelEditor, class_name: str, abstract: str) -> Edited:
    cls = editor.find_type(class_name, Class)
    wanted = abstract == "true"
    word = "abstract" if wanted else "concrete"
    if cls.is_abstract == wanted:
        raise EditError(f"{class_name} is {word} already")
    if wanted:
        objects = sorted(f"{_object_name(obj)} ({model_name})" for model_name, obj in _objects(editor)
                         if obj.classifier is cls)
        if objects:
            raise EditError(f"object {', '.join(objects)} is a {class_name}: an abstract class has no objects; "
                            "make each an object of a subclass first")
    _set_field(editor, cls, "is_abstract", "True" if wanted else None, "False")

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(class_name)
        return None if found is not None and found.is_abstract == wanted else f"{class_name} is not {word}"

    warnings = [f"no concrete class specializes {class_name}, so no {class_name} object can exist: add one "
                f"(besser model add-class NAME --parent {class_name})"] \
        if wanted and not any(not c.is_abstract for c in cls.all_specializations()) else []
    return Edited(editor.finish(), f"made {class_name} {word}", check, warnings)


def remove_parent(editor: ModelEditor, class_name: str, parent: str) -> Edited:
    cls = editor.find_type(class_name, Class)
    parent_class = editor.find_type(parent, Class)
    generalization = next((g for g in editor.model.generalizations
                           if g.specific is cls and g.general is parent_class), None)
    if generalization is None:
        parents = ", ".join(sorted(p.name for p in cls.parents())) or "none"
        raise EditError(f"{class_name} is no direct {parent} (its parents: {parents})")
    below = [cls, *sorted(cls.all_specializations(), key=lambda c: c.name)]
    before = {id(member): member for c in below for member in (*c.all_attributes(), *c.all_association_ends())}

    def detach() -> None:
        cls._delete_generalization(generalization)
        parent_class._delete_generalization(generalization)

    def attach() -> None:
        cls._add_generalization(generalization)
        parent_class._add_generalization(generalization)

    detach()
    try:
        kept = {id(member) for c in below for member in (*c.all_attributes(), *c.all_association_ends())}
    finally:
        attach()
    lost = [member for key, member in before.items() if key not in kept]
    held = sorted(f"{_object_name(obj)} ({slot.attribute.name})" for _, obj in _objects(editor)
                  if obj.classifier in below for slot in obj.slots if slot.attribute in lost)
    if held:
        raise EditError(f"object {', '.join(held)} holds a value of an attribute {class_name} has from {parent}: "
                        "change the objects first")
    _refuse_breaking(editor, f"{class_name} is no {parent}", *_broken_by(editor, detach, attach))
    # The generalization goes from where it is written.
    var = editor.var_of(generalization)
    literal = editor.collection(editor.model, "generalizations")
    items = _items(literal) if literal is not None else []
    index = next((i for i, item in enumerate(items) if (isinstance(item, ast.Name) and item.id == var)
                  or (var is None and _generalizes(editor, item, cls, parent_class))), None)
    if var is not None:
        _forget_var(editor, var)
    elif index is None:
        raise EditError(f"cannot find where {class_name} is made a {parent} in {editor.path.name}; edit it by hand")
    if index is not None:
        editor.src.remove_items(literal, items, {index})
    # A table of the class (or of a subclass) shows no column of what it no longer has.
    tables = []
    for table in _tables_inheriting(editor, cls):
        located = _column_items(editor, table)
        drop = {i for i, (column, _, _) in enumerate(located or ()) if any(_columns_of(column, m) for m in lost)}
        if located is None and any(_columns_of(column, m) for column in table.columns for m in lost):
            raise EditError(f"table {table.name} has a column of what {class_name} has from {parent}: "
                            "remove it by hand first")
        if drop:
            editor.src.remove_items(located[0][2], _items(located[0][2]), drop)
            tables.append(table.name)
    summary = f"{class_name} is no longer a {parent}"
    if tables:
        summary += f"; its columns of {parent} from table {', '.join(tables)}"

    def check(model: DomainModel) -> str | None:
        found = model.get_class_by_name(class_name)
        return None if found is not None and parent not in {p.name for p in found.parents()} \
            else f"{class_name} is still a {parent}"

    return Edited(editor.finish(), summary, check)


def _generalizes(editor: ModelEditor, item: ast.AST, specific: Class, general: Class) -> bool:
    """Whether the item ``Generalization(general=..., specific=...)`` of a set makes ``specific`` a ``general``."""
    if _callee(item) != "Generalization":
        return False
    names = {k.arg: k.value for k in item.keywords}
    values = [names.get("general", item.args[0] if item.args else None),
              names.get("specific", item.args[1] if len(item.args) > 1 else None)]
    return [editor.namespace.get(v.id) if isinstance(v, ast.Name) else None for v in values] == [general, specific]


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
    if isinstance(element, EnumerationLiteral):
        users = [f"{cls.name}.{attribute.name}" for cls in model.get_classes() for attribute in cls.attributes
                 if attribute.type is element.owner and attribute.default_value is not None
                 and str(getattr(attribute.default_value, "name", attribute.default_value)).strip()
                 .removeprefix(f"{element.owner.name}.") == element.name]
        if users:
            raise EditError(f"the default of {', '.join(sorted(users))} is {element.name}: change it first")
        _refuse_literal_in_constraints(editor, [element])
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
    for item in (item for item in [*outer, *members] if isinstance(item, Property)):
        users = _constraints_using(editor, item, outer)
        if users:
            raise EditError(f"constraint {', '.join(users)} uses {editor.describe(item)}: remove or change it first")
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
    # The GUI follows: a removed class's tables and screens go, and a removed attribute's columns.
    gui_removed = _gui_removals(editor, element) if isinstance(element, Class) else []
    for gui_element in gui_removed:
        _remove_gui_element(editor, gui_element, drops, removed_vars)
    for screen in (item for item in gui_removed if isinstance(item, Screen)):
        for _, _, items in _menus(editor):
            for index, _ in _menu_items_to(items, screen.route_path):
                drops.setdefault(id(items), (items, set()))[1].add(index)
    main_pages = _new_main_pages(editor, gui_removed, removed_vars)
    removed_properties = [item for item in [*outer, *members] if isinstance(item, Property)]
    column_tables = []
    # Any table may show them: a column of another class's table follows an end to the attribute.
    for table in editor.tables():
        if table in gui_removed:
            continue
        located = _column_items(editor, table)
        users = [column for column in table.columns if any(_columns_of(column, item) for item in removed_properties)]
        if users and located is None:
            raise EditError(f"table {table.name} has a column of {editor.describe(element)}: remove it by hand first")
        for index, (column, _, literal) in enumerate(located or ()):
            if column in users:
                drops.setdefault(id(literal), (literal, set()))[1].add(index)
        if users:
            column_tables.append(table.name)
    for literal, indices in drops.values():
        if not editor.src.is_deleted(literal):
            editor.src.remove_items(literal, _items(literal), indices)
    deleted = editor.src.replaced()
    still_used = sorted({node.lineno for node in _references(editor, removed_vars)
                         if not any(low <= editor.src.start(node) < high for low, high in deleted)})
    if still_used:
        lines = ", ".join(map(str, still_used))
        raise EditError(f"{editor.describe(element)} is still used on line{'s' if len(still_used) > 1 else ''} "
                        f"{lines} of {editor.path.name}: change those lines first")
    named = element.name if isinstance(element, Association) and "." in path else path
    also = [f"{type(item).__name__.lower().replace('binary', '')} {item.name}" for item in outer[1:]
            if isinstance(item, (Association, Constraint))]
    summary = f"removed {named}" + (f" and {', '.join(also)}" if also else "")
    named_gui = [f"{type(item).__name__.lower()} {item.name}" for item in gui_removed
                 if isinstance(item, (Screen, Table, Button))]
    if named_gui:
        summary += f"; and {', '.join(named_gui)}"
    if column_tables:
        summary += f"; its column from table {', '.join(column_tables)}"
    summary += main_pages

    def check(model: DomainModel) -> str | None:
        try:
            resolve(model, named)
        except EditError:
            return None
        return f"{named} is still there"

    return Edited(editor.finish(), summary, check, editor.warnings)


# ── the GUI model: screens and the tables that show a class ─────────────────
#
# A table bound to a class (``DataBinding``) shows one ``FieldColumn`` per attribute; the operations on
# the domain model keep those columns, and the titles derived from the class's name, in step.


def _contents(container: Any) -> list[Any]:
    """The view elements of a screen or container, at any depth, by name: ``view_elements`` is a set,
    and what a command says and writes must not change from one run to the next."""
    found = []
    for element in sorted(getattr(container, "view_elements", None) or (), key=lambda e: str(getattr(e, "name", ""))):
        found.append(element)
        found.extend(_contents(element))
    return found


def _screens(gui: Any) -> list[Screen]:
    """The screens of a GUI model, in the generator's order (modules, then screens, by name)."""
    return [screen for module in sorted(gui.modules, key=lambda m: m.name)
            for screen in sorted(module.screens, key=lambda s: s.name)]


def _lineage(cls: Class) -> list[Class]:
    """``cls`` and the classes it inherits from, each after its own parents (by name among equals):
    the order of a table's columns, parents first."""
    ordered: list[Class] = []

    def visit(current: Class) -> None:
        for parent in sorted(current.parents(), key=lambda c: c.name):
            visit(parent)
        if current not in ordered:
            ordered.append(current)

    visit(cls)
    return ordered


def _tables_inheriting(editor: ModelEditor, cls: Class) -> list[Table]:
    """The tables of ``cls`` and of its subclasses: those that may show an attribute of ``cls``."""
    found = {id(table): table for owner in (cls, *cls.all_specializations()) for table in editor.tables_of(owner)}
    return sorted(found.values(), key=lambda table: table.name)


def _tables_following(editor: ModelEditor, cls: Class) -> list[Table]:
    """The tables that get a column for a new attribute of ``cls``: its own, and those of its subclasses
    that show inherited columns (as ``add-screen`` makes them), or whose class inherited no attribute yet."""
    found = list(editor.tables_of(cls))
    for sub in cls.all_specializations():
        inherited = sub.inherited_attributes()
        found += [table for table in editor.tables_of(sub) if not inherited or any(
            getattr(column, "field", None) in inherited for column in table.columns)]
    return sorted({id(table): table for table in found}.values(), key=lambda table: table.name)


def _bound_class(table: Table) -> Any:
    return getattr(getattr(table, "data_binding", None), "domain_concept", None)


def _label(name: str) -> str:
    """A column's label derived from its attribute's name: ``due_date`` -> ``Due date``."""
    return (name[:1].upper() + name[1:]).replace("_", " ")


def _derived_titles(class_name: str) -> list[str]:
    """The titles ``add-screen`` and the starter derive from a class's name: ``Authors``, then ``Author``."""
    return [_plural(class_name), class_name]


def _column_items(editor: ModelEditor, table: Table) -> list[tuple[Any, ast.AST, ast.AST]] | None:
    """Each column of ``table`` with the item of the list literal that writes it, and that literal;
    ``None`` when the file does not write the columns as one list."""
    literal = editor.collection(table, "columns")
    items = _items(literal) if literal is not None else []
    if len(items) != len(table.columns):
        return None
    return [(column, item, literal) for column, item in zip(table.columns, items)]


def _columns_of(column: Any, member: Property) -> bool:
    return getattr(column, "field", None) is member or (isinstance(column, LookupColumn) and column.path is member)


def _replace_strings(editor: ModelEditor, call: ast.Call | None, keywords: Sequence[str],
                     renamed: dict[str, str]) -> list[tuple[str, str]]:
    """Replace the string arguments ``keywords`` of ``call`` that are a key of ``renamed``; returns the changes."""
    changes = []
    for keyword_ in call.keywords if call is not None else ():
        value = keyword_.value
        if keyword_.arg in keywords and isinstance(value, ast.Constant) and value.value in renamed:
            new = renamed[value.value]
            editor.src.replace(editor.src.start(value), editor.src.end(value),
                               _string_code(editor.src.segment(value), value.value, new))
            changes.append((value.value, new))
    return changes


def _attribute_variables(editor: ModelEditor, cls: Class) -> dict[int, str]:
    """The variable of each attribute of ``cls``; an attribute written inside its class's set gets one,
    defined before the class (as the starter writes them), so that a column can refer to it."""
    found = {id(attribute): editor.var_of(attribute) for attribute in cls.attributes if editor.var_of(attribute)}
    literal = editor.collection(cls, "attributes")
    items = _items(literal) if literal is not None else []
    by_name = {attribute.name: attribute for attribute in cls.attributes}
    texts, hoisted = [], False
    for item in items:
        name = _name_argument(item) if _callee(item) == "Property" else None
        attribute = by_name.get(name.value) if name is not None else None
        if attribute is None or id(attribute) in found:
            texts.append(editor.src.segment(item))
            continue
        var = editor.free_name(_attribute_var_base(editor, cls, attribute.name))
        editor.src.insert_before(editor.src.top_statement(literal), f"{var} = {editor.src.segment(item)}\n", False)
        found[id(attribute)] = var
        texts.append(var)
        hoisted = True
    if hoisted:
        # On one line when it fits, as the starter writes a class's attributes.
        start, end = editor.src.start(literal), editor.src.end(literal)
        line = (editor.src.text[editor.src.line_start(start):start] + "{" + ", ".join(texts) + "}"
                + editor.src.text[end:editor.src.line_end(end)].rstrip("\n"))
        if len(line) <= _MAX_LINE and isinstance(literal, ast.Set):
            editor.src.replace(start, end, "{" + ", ".join(texts) + "}")
        else:
            editor.src.rewrite_items(literal, items, texts)
    missing = sorted(attribute.name for attribute in cls.attributes if id(attribute) not in found)
    if missing:
        raise EditError(f"cannot find where {cls.name}.{', '.join(missing)} is written to refer to it from a "
                        f"table; give it a variable")
    return found


def _gui_model(editor: ModelEditor) -> Any:
    """The GUI model a new screen goes in: the file's only one, or the one ``--use`` names."""
    guis = editor.loaded.of_kind("gui")
    named = [gui for gui in guis if gui.names & set(editor.use)]
    guis = named or guis
    if not guis:
        raise EditError(f"{editor.path.name} has no GUI model to add a screen to (a project made with "
                        "'besser init --kind gui' has one)")
    if len(guis) > 1:
        raise EditError(f"{editor.path.name} holds {len(guis)} GUI models; name one with --use "
                        f"({', '.join(gui.name for gui in guis)})")
    return guis[0].model


def _clashing_screen(screens: Sequence[Screen], name: str, description: str | None,
                     besides: Screen | None = None, route: str | None = None) -> Screen | None:
    """The screen of ``screens`` (other than ``besides``) that a screen named ``name`` with ``description``
    (at ``route``; default: ``/name``) would collide with in the generated app: the same route but for
    case (routes ignore case, so one would open the other), or the same page file but for case (one
    overwrites the other on a case-insensitive file system, as macOS's and Windows' are)."""
    def keys(screen_route: str, screen_name: str, screen_description: str | None) -> set[tuple[str, str]]:
        return {("route", screen_route.lower()), ("file", _page_file(screen_name, screen_description).lower())}

    new = keys(route or f"/{name}", name, description)
    return next((other for other in screens if other is not besides
                 and keys(other.route_path, other.name, other.description) & new), None)


def _page_file(name: str, description: str | None) -> str:
    """The name of the page file the React generator writes for a screen, without ``.tsx`` (and without
    the number it appends to a name taken exactly): its description, or its name, in PascalCase with
    inner words lowercased (``OrderLines`` -> ``Orderlines``, ``Order lines`` -> ``OrderLines``)."""
    from besser.generators.react.page_builder import PageBuilderMixin  # noqa: PLC0415 - only for a GUI model

    return PageBuilderMixin._to_pascal_case(description or name.replace("_", " ").strip())


_ROUTE = re.compile(r"^(/[A-Za-z0-9][\w-]*)+$")


def _screen_name(title: str) -> str:
    """A screen's name from its title: the title's words joined (Book reviews -> BookReviews)."""
    name = "".join(word[:1].upper() + word[1:] for word in re.split(r"\W+", title))
    if not name:
        raise EditError(f"'{title}' is no title: give it letters or digits")
    return name


def _new_screen(editor: ModelEditor, title: str, route: str | None) -> tuple[Any, list[Screen], str, str, ast.AST]:
    """What a new screen titled ``title`` needs: the GUI model, its screens, the new screen's name and
    route (the one asked for, or its name in lower case), and the literal that lists the screens."""
    gui = _gui_model(editor)
    screens = _screens(gui)
    name = _screen_name(title)
    route = route or f"/{name.lower()}"
    if not _ROUTE.match(route):
        raise EditError(f"'{route}' is not a route: write /name or /group/name (letters, digits, - and _)")
    clash = _clashing_screen(screens, name, title, route=route)
    if clash is not None and clash.name.lower() != name.lower() and clash.route_path.lower() == route.lower():
        raise EditError(f"screen {clash.name} of {gui.name} is at {clash.route_path} already: give another --route")
    if clash is not None:
        raise EditError(f"{gui.name} has a screen named '{clash.name}' already: give another --title")
    module = min(gui.modules, key=lambda m: m.name, default=None)
    literal = editor.collection(module, "screens") if module is not None else None
    if literal is None:
        raise EditError(f"cannot find where the screens of {gui.name} are written in {editor.path.name}; "
                        "edit it by hand")
    return gui, screens, name, route, literal


def _arguments(first: str, parts: Sequence[str]) -> str:
    """``parts`` as arguments, a comma after each, on as many lines as they need: the first starts with
    ``first``, the others are aligned under what follows it."""
    lines: list[str] = []
    current: list[str] = []
    for part in parts:
        if current and len(first) + len(", ".join([*current, part])) + 1 > _MAX_LINE:
            lines.append(", ".join(current) + ",")
            current = []
        current.append(part)
    lines.append(", ".join(current) + ",")
    return first + f"\n{' ' * len(first)}".join(lines)


# How the pages the screen commands write look: utility classes of Tailwind, which the generated React
# application ships, set on the elements. A page then has a navigation bar, a centred column, a heading,
# cards side by side and a panel round a chart, with no hand-written CSS and nothing new in a generator.
_MENU_CLASSES = ("bg-slate-900", "px-6", "py-3", "text-sm", "font-medium", "[&_a]:text-slate-200",
                 "[&_a:hover]:text-white")
_PAGE_CLASSES = ("max-w-5xl", "mx-auto", "px-6", "py-6")
_HEADING_CLASSES = ("text-2xl", "font-bold", "text-slate-900", "mb-4")
_CARDS_CLASSES = ("grid", "grid-cols-1", "md:grid-cols-3", "gap-4", "mb-4")
_PANEL_CLASSES = ("bg-white", "rounded-xl", "shadow", "p-4", "mb-4")
_LINK_CLASSES = ("inline-block", "mb-4", "text-blue-700", "underline")


def _classes(names: Sequence[str]) -> str:
    return f"css_classes=[{', '.join(_q(name) for name in names)}]"


def _heading_code(editor: ModelEditor, var: str, title: str) -> str:
    """A page's heading, as an item of its content (written 8 columns in). Ordered first, so that what is
    added to the page later follows it."""
    text = _arguments(f"{' ' * 8}{editor._gui('Text')}(", [
        f"name={_q(var)}", f"component_id={_q(var)}", f"content={_q(title)}", 'tag_name="h1"',
        _classes(_HEADING_CLASSES), "display_order=1"])
    return text.strip()[:-1] + ")"


def _content_code(editor: ModelEditor, var: str, title: str, elements: Sequence[str]) -> str:
    """The container that holds a screen's content under its menu: a centred column."""
    items = "".join(f"        {element},\n" for element in elements)
    return (f"{var} = {editor._gui('ViewContainer')}(\n"
            f"    name={_q(var)}, component_id={_q(var)}, description={_q(title)}, display_order=1,\n"
            f"    {_classes(_PAGE_CLASSES)},\n"
            f"    view_elements={{\n{items}    }},\n)\n")


def _content_of(screen: Screen) -> Any:
    """Where what is added to a screen goes: the container the screen commands write under the menu (the
    screen's only plain container), or the screen itself when it has none (a page written otherwise)."""
    boxes = [element for element in screen.view_elements if type(element) is ViewContainer]
    return boxes[0] if len(boxes) == 1 else screen


def _next_order(container: Any) -> int:
    """The order of an element added after those ``container`` shows in an order (those without one come
    last on the page)."""
    return int(max((element.display_order for element in container.view_elements
                    if getattr(element, "display_order", None) is not None), default=0)) + 1


def _screen_code(editor: ModelEditor, var: str, name: str, title: str, route: str, main: bool,
                 elements: Sequence[str]) -> str:
    items = "".join(f"        {element},\n" for element in elements)
    arguments = [f"name={_q(name)}", f"component_id={_q(name)}", f"description={_q(title)}",
                 f"route_path={_q(route)}", *(["is_main_page=True"] if main else [])]
    return (f"{var} = {editor._gui('Screen')}(\n{_arguments('    ', arguments)}\n"
            f"    view_elements={{\n{items}    }},\n)\n")


def _menus(editor: ModelEditor) -> list[tuple[Menu, str, ast.AST]]:
    """The menus a command can keep in step with the screens, each with its variable and the literal of
    its items: those held by a variable, whose items the file writes as one set or list."""
    menus = {id(value): value for value in editor.namespace.values() if isinstance(value, Menu)}
    found = []
    for menu in sorted(menus.values(), key=lambda menu: menu.name):
        var, literal = editor.var_of(menu), editor.collection(menu, "menuItems")
        if var is not None and literal is not None:
            found.append((menu, var, literal))
    return found


def _navigation_menus(editor: ModelEditor, screens: Sequence[Screen]) -> list[tuple[Menu, str, ast.AST]]:
    """The menus (see ``_menus``) shown on every one of ``screens``: a new screen joins them."""
    return [found for found in _menus(editor) if all(found[0] in _contents(screen) for screen in screens)]


def _menu_items_to(literal: ast.AST, route: str) -> list[tuple[int, ast.Call]]:
    """The items of a menu's literal that link to ``route``, with their index."""
    found = []
    for index, item in enumerate(_items(literal)):
        url = _argument(item, "url", 1) if _callee(item) == "MenuItem" else None
        if isinstance(url, ast.Constant) and url.value == route:
            found.append((index, item))
    return found


def _link_screen(editor: ModelEditor, screens: Sequence[Screen], anchor: ast.stmt, screen_var: str,
                 screen_name: str, title: str, route: str, main: bool) -> tuple[list[str], str, str]:
    """Make a new screen (the variable ``screen_var``, defined before ``anchor``) reachable: an item in
    each navigation menu, which it shows too; without a menu, a button on the main page (on the new
    screen to the others, when it becomes the main page). Returns the view elements the screen gets,
    the code that follows its definition, and how it is reached, for the summary."""
    current = next((screen for screen in screens if screen.is_main_page), screens[0] if screens else None)
    if main and current is not None and current.is_main_page:
        _unmake_main_page(editor, current)
    said = "the main page" if main or current is None else ""
    menus = [(menu, var, items) for menu, var, items in _navigation_menus(editor, screens)
             if editor.src.start(editor.src.assignment(var)) < editor.src.start(anchor)]
    if menus:
        for _, _, items in menus:
            item = f"{editor._gui('MenuItem')}(label={_q(title)}, url={_q(route)})"
            if _items(items):
                editor.add_to_literal(items, item)
            else:  # one item a line from the first on
                editor.src.replace(editor.src.start(items), editor.src.end(items), f"{{\n        {item},\n    }}")
        said += f"{', ' if said else ''}in menu {', '.join(menu.name for menu, _, _ in menus)}"
        return [var for _, var, _ in menus], "", said
    if current is None:
        return [], "", said
    if not main:
        button = editor.free_name(f"to_{_snake(screen_name)}")
        return [], _navigation_button(editor, current, anchor, button, screen_var, screen_name, title), \
            f"button on {current.name}"
    code, opened = "", []
    for target in screens:
        target_var = editor.var_of(target)
        if target_var is None or editor.src.start(editor.src.assignment(target_var)) > editor.src.start(anchor):
            editor.warnings.append(f"no button on {screen_name} opens screen {target.name}: it has no variable "
                                   "above the screens' list to refer to it by; add the button by hand")
            continue
        button = editor.free_name(f"{_snake(screen_name)}_to_{_snake(target.name)}")
        code += _button_code(editor, button, target_var, target.name, target.description or target.name)
        code += f"{screen_var}.view_elements.add({button})\n"
        opened.append(target.name)
    return [], code, said + (f", with a button to {', '.join(opened)}" if opened else "")


def _unmake_main_page(editor: ModelEditor, screen: Screen) -> None:
    """Plan ``is_main_page=False`` on ``screen``, where the file makes it the main page."""
    call = editor.call_of(screen)
    flag = next((keyword_ for keyword_ in call.keywords if keyword_.arg == "is_main_page"), None) \
        if call is not None else None
    var = editor.var_of(screen)
    statements = [statement for statement in (editor.src.attribute_statements(var, "is_main_page") if var else [])
                  if isinstance(statement, ast.Assign)]
    value = statements[-1].value if statements else flag.value if flag is not None else None
    if value is None:
        raise EditError(f"cannot find where screen {screen.name} is made the main page in {editor.path.name}; "
                        "edit it by hand")
    editor.src.replace(editor.src.start(value), editor.src.end(value), "False")


def _has_screen(screen_name: str) -> Callable[[list[Any]], str | None]:
    def gui_check(guis: list[Any]) -> str | None:
        found = any(screen.name == screen_name for gui in guis for screen in _screens(gui))
        return None if found else f"no screen {screen_name}"
    return gui_check


def add_screen(editor: ModelEditor, class_name: str, title: str | None, route: str | None = None,
               rows: int | None = None) -> Edited:
    if rows is not None and rows < 1:
        raise EditError(f"--rows {rows}: a table shows at least 1 row a page")
    cls = editor.find_type(class_name, Class)
    for screen in _screens(_gui_model(editor)):
        if any(isinstance(element, Table) and _bound_class(element) is cls for element in _contents(screen)):
            raise EditError(f"screen {screen.name} shows {class_name} already")
    lineage = _lineage(cls)
    attributes = [attribute for owner in lineage for attribute in sort_by_timestamp(owner.attributes)]
    if not attributes:
        raise EditError(f"{class_name} has no attribute, own or inherited, for its table to show: add one first "
                        f"(besser model add-attribute {class_name} name:str)")
    title = title or _plural(class_name)
    # The name is the title's words joined (Book reviews -> BookReviews); the route, unless given, that in lower case.
    _, screens, screen_name, route, literal = _new_screen(editor, title, route)
    anchor = editor.src.top_statement(literal)
    fields = {key: var for owner in lineage for key, var in _attribute_variables(editor, owner).items()}
    base = _snake(class_name)
    table, screen_var = editor.free_name(f"{base}_table"), editor.free_name(f"{base}_screen")
    heading, content = editor.free_name(f"{base}_heading"), editor.free_name(f"{base}_page")
    columns = "".join(f"        {editor._gui('FieldColumn')}(label={_q(_label(attribute.name))}, "
                      f"field={fields[id(attribute)]}),\n" for attribute in attributes)
    shown, navigation, reached = _link_screen(editor, screens, anchor, screen_var, screen_name, title, route, False)
    code = (
        f"{table} = {editor._gui('Table')}(\n"
        f"    name={_q(table)}, component_id={_q(table)}, title={_q(title)},\n"
        f"    action_buttons=True, show_pagination=True,{f' rows_per_page={rows},' if rows else ''} "
        f"columns=[\n{columns}    ],\n)\n"
        f"{table}.data_binding = {editor._gui('DataBinding')}(domain_concept={editor.type_code(class_name)})\n"
        + _content_code(editor, content, title, [_heading_code(editor, heading, title), table])
        + _screen_code(editor, screen_var, screen_name, title, route, not screens, [*shown, content])
        + navigation
    )
    editor.src.insert_before(anchor, code, separate=True)
    editor.add_to_literal(literal, screen_var)
    # The title when it is not the name: 'Deal lines' is the screen DealLines, and the answer must show it took.
    titled = "" if title == screen_name else f" titled '{title}'"
    summary = (f"added screen {screen_name}{titled} at {route} (table of {class_name}: "
               f"{', '.join(a.name for a in attributes)}{f'; {rows} rows a page' if rows else ''}; {reached})")

    def gui_check(guis: list[Any]) -> str | None:
        shown = [screen for gui in guis for screen in _screens(gui) if screen.name == screen_name and any(
            isinstance(element, Table) and getattr(_bound_class(element), "name", None) == class_name
            for element in _contents(screen))]
        return None if shown else f"no screen {screen_name} showing {class_name}"

    return Edited(editor.finish(), summary, lambda model: None, editor.warnings, gui_check=gui_check)


def add_page(editor: ModelEditor, title: str, route: str | None, main: bool) -> Edited:
    _, screens, screen_name, route, literal = _new_screen(editor, title, route)
    anchor = editor.src.top_statement(literal)
    base = _snake(screen_name)
    screen_var, heading = editor.free_name(f"{base}_screen"), editor.free_name(f"{base}_heading")
    content = editor.free_name(f"{base}_page")
    shown, navigation, reached = _link_screen(editor, screens, anchor, screen_var, screen_name, title, route, main)
    code = (_content_code(editor, content, title, [_heading_code(editor, heading, title)])
            + _screen_code(editor, screen_var, screen_name, title, route, main or not screens, [*shown, content]))
    editor.src.insert_before(anchor, code + navigation, separate=True)
    editor.add_to_literal(literal, screen_var)
    return Edited(editor.finish(), f"added page {screen_name} at {route} ({reached or 'no link to it yet'})",
                  lambda model: None, editor.warnings, gui_check=_has_screen(screen_name))


def add_menu(editor: ModelEditor) -> Edited:
    gui = _gui_model(editor)
    screens = _screens(gui)
    existing = _navigation_menus(editor, screens)
    if existing:  # nothing to do: an error would undo the batch this line is part of
        return Edited(editor.finish(), f"menu {existing[0][0].name} links the screens of {gui.name} already "
                                       "(no change)", lambda model: None)
    module = min(gui.modules, key=lambda m: m.name, default=None)
    literal = editor.collection(module, "screens") if module is not None else None
    if literal is None:
        raise EditError(f"cannot find where the screens of {gui.name} are written in {editor.path.name}; "
                        "edit it by hand")
    # The menu is defined before the first statement that writes a screen, and each screen shows it.
    first = editor.src.top_statement(literal)
    var = editor.free_name("menu")
    after, last = "", None
    for screen in screens:
        elements, screen_var = editor.collection(screen, "view_elements"), editor.var_of(screen)
        statement = editor.src.top_statement(elements) if elements is not None else \
            editor.src.assignment(screen_var) if screen_var else None
        if statement is None:
            raise EditError(f"cannot find where screen {screen.name} is written in {editor.path.name} to put the "
                            "menu on it: give it a variable")
        if editor.src.start(statement) < editor.src.start(first):
            first = statement
        if elements is not None:
            editor.add_to_literal(elements, var)
        else:
            after += f"{screen_var}.view_elements.add({var})\n"
            last = statement if last is None or editor.src.start(statement) > editor.src.start(last) else last
    items = "".join(f"        {editor._gui('MenuItem')}(label={_q(screen.description or screen.name)}, "
                    f"url={_q(screen.route_path)}),\n" for screen in screens)
    # Ordered first: a page's elements without an order follow those that have one.
    listed = "{\n" + items + "    }" if items else "set()"
    editor.src.insert_before(first, (
        f"{var} = {editor._gui('Menu')}(\n"
        f"    name={_q(var)}, component_id={_q(var)}, description=\"Navigation\", display_order=0,\n"
        f"    {_classes(_MENU_CLASSES)},\n"
        f"    menuItems={listed},\n)\n"), separate=True)
    if last is not None:
        editor.src.insert_after(last, after)
    replaced = _drop_navigation_buttons(editor, screens)

    def gui_check(guis: list[Any]) -> str | None:
        bare = [screen.name for gui in guis for screen in _screens(gui)
                if not any(isinstance(element, Menu) and element.name == var for element in _contents(screen))]
        return f"screen {', '.join(bare)} does not show the menu" if bare else None

    summary = f"added menu {var}" + (f" to {', '.join(screen.name for screen in screens)}" if screens else
                                     " (no screen yet: each new one joins it)")
    if replaced:
        summary += f"; it replaces the button{'s' if len(replaced) > 1 else ''} to {', '.join(replaced)}"
    return Edited(editor.finish(), summary, lambda model: None, editor.warnings, gui_check=gui_check)


def _drop_navigation_buttons(editor: ModelEditor, screens: Sequence[Screen]) -> list[str]:
    """Remove the buttons that only open a screen, as ``add-screen`` writes them while there is no menu:
    the menu links every screen, and beside it they are a second, unstyled navigation. Returns the
    screens they opened. A button the file writes inside a set stays (removing it there and adding the
    menu to the same set would be two edits of one place)."""
    in_sets = {item.id for node in ast.walk(editor.src.tree) if isinstance(node, (ast.Set, ast.List))
               for item in node.elts if isinstance(item, ast.Name)}
    buttons = {id(element): element for screen in screens for element in _contents(screen)
               if isinstance(element, Button) and getattr(element, "targetScreen", None) in screens}
    opened: list[str] = []
    for button in sorted(buttons.values(), key=lambda button: button.name):
        var = editor.var_of(button)
        if var is None or var in in_sets:
            continue
        _remove_gui_element(editor, button, {}, [])
        opened.append(button.targetScreen.name)
    return list(dict.fromkeys(opened))  # each screen once, whatever the number of buttons to it


def _end_variable(editor: ModelEditor, end: Property) -> tuple[str, int]:
    """The variable of an association end and where it is defined; an end written inside its association
    gets one, defined before it, so that a column can follow it."""
    var = editor.var_of(end)
    if var is not None:
        return var, editor.src.start(editor.src.assignment(var))
    association = editor.association_of(end)
    literal = editor.collection(association, "ends")
    item = next((item for item in _items(literal) if _builds(item, end)), None) if literal is not None else None
    if item is None:
        raise EditError(f"cannot find where {editor.describe(end)} is written to follow it from a table; "
                        "give it a variable")
    statement = editor.src.top_statement(literal)
    var = editor.free_name(f"{editor.var_of(association) or _snake(association.name)}_{end.name}")
    editor.src.insert_before(statement, f"{var} = {editor.src.segment(item)}\n", False)
    editor.src.replace(editor.src.start(item), editor.src.end(item), var)
    return var, editor.src.start(statement)


def _attribute_named(cls: Class, name: str) -> Property:
    found = next((attribute for attribute in cls.all_attributes() if attribute.name == name), None)
    if found is None:
        known = ", ".join(sorted(attribute.name for attribute in cls.all_attributes())) or "none"
        raise EditError(f"{cls.name} has no attribute '{name}' (its attributes: {known})")
    return found


def _field_variables(editor: ModelEditor, attributes: Sequence[Property]) -> dict[int, str]:
    """The variable of each of ``attributes``, for a GUI element to refer to them by."""
    owners = {id(attribute.owner): attribute.owner for attribute in attributes}
    return {key: var for owner in owners.values() for key, var in _attribute_variables(editor, owner).items()}


def add_column(editor: ModelEditor, class_name: str, path: str, label: str | None) -> Edited:
    cls = editor.find_type(class_name, Class)
    end_name, _, attribute_name = path.partition(".")
    own = next((attribute for attribute in cls.all_attributes() if attribute.name == path), None)
    if own is not None:  # an attribute of the class itself: add-attribute adds this column already
        tables = [table for table in editor.tables_of(cls)
                  if not any(getattr(column, "field", None) is own and not isinstance(column, LookupColumn)
                             for column in table.columns)]
        if not tables:
            return Edited(editor.finish(), f"{class_name}.{path} is a column of its table already (no change)",
                          lambda model: None)
        names = _add_columns(editor, tables, [(own.name, _field_variables(editor, [own])[id(own)])])
        return Edited(editor.finish(), f"added column {_label(own.name)} to table {', '.join(names)}",
                      lambda model: None, editor.warnings)
    ends = {end.name: end for end in cls.all_association_ends()}
    if end_name not in ends or not attribute_name:
        to_one = sorted(name for name, end in ends.items() if end.multiplicity.max == 1)
        raise EditError(f"'{path}' is not end.attribute of {class_name}: write an association end that leads to "
                        f"one object ({', '.join(to_one) or 'it has none'}), then an attribute of its class")
    end = ends[end_name]
    if end.multiplicity.max != 1:
        raise EditError(f"{class_name}.{end_name} is {multiplicity_text(end.multiplicity.min, end.multiplicity.max)}: "
                        "a column shows one object, so the end must be [1] or [0..1]")
    attribute = _attribute_named(end.type, attribute_name)
    tables = [table for table in editor.tables_of(cls) if not any(
        isinstance(column, LookupColumn) and column.path is end and column.field is attribute
        for column in table.columns)]
    if not tables:
        raise EditError(f"no table of {class_name} to add the column to: " + (
            f"{path} is a column already" if editor.tables_of(cls) else
            f"add its screen first (besser model add-screen {class_name})"))
    end_var, defined = _end_variable(editor, end)
    field_var = _field_variables(editor, [attribute])[id(attribute)]
    label = label or _label(end.name)
    for table in tables:
        columns = editor.collection(table, "columns")
        if columns is not None and editor.src.start(editor.src.top_statement(columns)) < defined:
            raise EditError(f"{editor.describe(end)} is written after table {table.name} in {editor.path.name}: "
                            "move its association above the table")
        editor.add(table, "columns", f"{editor._gui('LookupColumn')}(label={_q(label)}, path={end_var}, "
                                     f"field={field_var})",
                   lambda table=table: editor.free_name(f"{editor.var_of(table) or table.name}_{end.name}"),
                   need_var=False, separate=False)

    def gui_check(guis: list[Any]) -> str | None:
        shown = [column for gui in guis for screen in _screens(gui) for table in _contents(screen)
                 if isinstance(table, Table) for column in table.columns
                 if isinstance(column, LookupColumn) and column.label == label]
        return None if shown else f"no column {label}"

    return Edited(editor.finish(), f"added column {label} ({path}) to table {', '.join(t.name for t in tables)}",
                  lambda model: None, editor.warnings, gui_check=gui_check)


def _column_key(column: Any) -> str:
    """A column as the commands name it: its attribute, or ``end.attribute`` of a related object."""
    field_ = getattr(column, "field", None)
    if field_ is None:
        return str(getattr(column, "label", ""))
    path = column.path if isinstance(column, LookupColumn) else None
    return f"{path.name}.{field_.name}" if path is not None else field_.name


def order_columns(editor: ModelEditor, class_name: str, columns: Sequence[str]) -> Edited:
    """Put the named columns first, in the order given, in every table of the class; the others follow in
    the order they had. Only the places of the list's items change: each keeps the code that writes it."""
    cls = editor.find_type(class_name, Class)
    tables = editor.tables_of(cls)
    if not tables:
        raise EditError(f"no table of {class_name}: add its screen first (besser model add-screen {class_name})")
    if len(set(columns)) != len(columns):
        raise EditError(f"a column is named twice: {' '.join(columns)}")
    ordered: dict[str, list[str]] = {}
    for table in tables:
        items = _column_items(editor, table)
        if items is None:
            raise EditError(f"table {table.name} does not write its columns as one list in {editor.path.name}: "
                            "reorder them there by hand")
        keys = [_column_key(column) for column, _, _ in items]
        unknown = [name for name in columns if name not in keys]
        if unknown:
            raise EditError(f"table {table.name} has no column {', '.join(unknown)} (it has: {', '.join(keys)}; a "
                            f"related object's column is end.attribute)")
        order = [*columns, *(key for key in keys if key not in columns)]
        ordered[table.name] = order
        texts = {key: editor.src.segment(item) for key, (_, item, _) in zip(keys, items)}
        for key, (_, item, _), new in zip(keys, items, order):
            if key != new:
                editor.src.replace(editor.src.start(item), editor.src.end(item), texts[new])

    def gui_check(guis: list[Any]) -> str | None:
        for gui in guis:
            for screen in _screens(gui):
                for table in _contents(screen):
                    if isinstance(table, Table) and table.name in ordered:
                        got = [_column_key(column) for column in table.columns]
                        if got != ordered[table.name]:
                            return f"table {table.name} has its columns as {', '.join(got)}"
        return None

    summary = "; ".join(f"columns of table {name}: {', '.join(order)}" for name, order in ordered.items())
    return Edited(editor.finish(), summary, lambda model: None, editor.warnings, gui_check=gui_check)


# ── cards and charts: one number, or one per value of an attribute ──────────

_VALUE = re.compile(r"^(?P<how>[A-Za-z]+):(?P<cls>[A-Za-z_]\w*)(?:\.(?P<attribute>[A-Za-z_]\w*))?$")
# What a card or chart computes -> the member of DataAggregation.
_AGGREGATIONS = {"sum": "SUM", "avg": "AVG", "count": "COUNT", "min": "MIN", "max": "MAX", "median": "MEDIAN"}
_CONDITION = re.compile(r"^(?P<field>[A-Za-z_]\w*)\s*(?P<op>==|!=|>=|<=|=|>|<)\s*(?P<value>.+)$")
_CHARTS = {"bar": "BarChart", "line": "LineChart", "pie": "PieChart"}


def _parse_value(editor: ModelEditor, spec: str) -> tuple[str, Class, Property | None]:
    """What ``sum:Deal.amount`` or ``count:Deal`` computes: how, over which class, of which attribute."""
    match = _VALUE.match(spec)
    how = match["how"].lower() if match else ""
    if how not in _AGGREGATIONS:
        raise EditError(f"'{spec}' is not a value: write sum:Class.attribute (also avg, min, max, median) or "
                        "count:Class")
    cls = editor.find_type(match["cls"], Class)
    if match["attribute"] is None:
        if how != "count":
            raise EditError(f"'{spec}': {how} is of an attribute, as {how}:{cls.name}.attribute")
        return how, cls, None
    attribute = _attribute_named(cls, match["attribute"])
    if how != "count" and _value_kind(attribute.type) != "number":
        raise EditError(f"{cls.name}.{attribute.name} is {_type_name(attribute.type)}: {how} is of an int or float "
                        "attribute")
    return how, cls, attribute


def _filter_expression(cls: Class, where: str | None) -> str | None:
    """``where`` as the generated pages read a filter: ``attribute op value`` conditions joined by
    ``and``. They drop without a word a condition they cannot read (and split a value at a comma), so
    anything else is refused here."""
    if where is None:
        return None
    if re.search(r"\s(or|OR)\s|\|\|", where):
        raise EditError(f"'{where}': the pages cannot read 'or'; they read conditions joined by 'and'")
    conditions = []
    for part in re.split(r"\s+and\s+|\s*&&\s*", where.strip()):
        match = _CONDITION.match(part.strip())
        attribute = next((a for a in cls.all_attributes() if match and a.name == match["field"]), None)
        if attribute is None:
            known = ", ".join(sorted(a.name for a in cls.all_attributes()))
            raise EditError(f"'{part.strip()}' is not a condition on {cls.name}: write attribute == value (also != > "
                            f">= < <=), several joined by 'and'; its attributes: {known}")
        value = match["value"].strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        operator = "==" if match["op"] == "=" else match["op"]
        kind = _value_kind(attribute.type)
        if isinstance(attribute.type, Enumeration):
            value = re.sub(rf"^{attribute.type.name}(\.|::)", "", value)
            literals = [literal.name for literal in sort_by_timestamp(attribute.type.literals)]
            if value not in literals or operator not in ("==", "!="):
                raise EditError(f"'{part.strip()}': compare {attribute.name} with == or != to a literal of "
                                f"{attribute.type.name} ({', '.join(literals)})")
        elif kind == "number":
            try:
                float(value)
            except ValueError:
                raise EditError(f"'{part.strip()}': {attribute.name} is a number, '{value}' is not") from None
        elif kind == "boolean":
            value = value.lower()
            if value not in ("true", "false") or operator not in ("==", "!="):
                raise EditError(f"'{part.strip()}': compare {attribute.name} with == or != to true or false")
        elif re.search(r"[,;&'\"]|\band\b", value):
            raise EditError(f"'{part.strip()}': the pages cannot read a value that holds , ; & a quote or 'and'")
        quoted = kind not in ("number", "boolean")
        conditions.append(f"{attribute.name} {operator} " + (f"'{value}'" if quoted else value))
    return " and ".join(conditions)


def _find_screen(editor: ModelEditor, name: str) -> Screen:
    """The screen ``name`` names: by its name, title or route, whatever the case."""
    gui = _gui_model(editor)
    screens = _screens(gui)
    found = [screen for screen in screens
             if name.lower() in (screen.name.lower(), (screen.description or "").lower(), screen.route_path.lower())]
    if len(found) != 1:
        known = ", ".join(f"{screen.name} ({screen.route_path})" for screen in screens) or "none"
        raise EditError(f"{gui.name} has no screen '{name}' (its screens: {known}); 'besser model add-page' adds one")
    return found[0]


def _add_bound(editor: ModelEditor, screen_name: str, title: str, kind: type, constructor: str, title_argument: str,
               options: Sequence[str], how: str, cls: Class, attribute: Property | None, label: Property | None,
               where: str | None) -> tuple[str, str, Callable[[list[Any]], str | None]]:
    """Add to a screen an element that shows data of a class (a card or a chart, built by ``constructor``):
    after what the screen shows already. Returns the screen's name, the element's and the check."""
    screen = _find_screen(editor, screen_name)
    if any(isinstance(element, kind) and title in (getattr(element, "title", None),
                                                   getattr(element, "metric_title", None))
           for element in _contents(screen)):
        raise EditError(f"screen {screen.name} has a {kind.__name__.lower()} titled '{title}' already")
    fields = _field_variables(editor, [a for a in (attribute, label) if a is not None])
    binding = [f"domain_concept={editor.type_code(cls.name, (Class,))}"]
    if label is not None:
        binding.append(f"label_field={fields[id(label)]}")
    if attribute is not None:
        binding.append(f"data_field={fields[id(attribute)]}")
    binding.append(f"aggregation={editor._gui('DataAggregation')}.{_AGGREGATIONS[how]}")
    filter_expression = _filter_expression(cls, where)
    if filter_expression is not None:
        binding.append(f"filter_expression={_q(filter_expression)}")
    var = editor.free_name(_snake(_screen_name(title)))
    data = _arguments(f"    data_binding={editor._gui('DataBinding')}(", binding)

    def code(order: int) -> str:
        arguments = [f"name={_q(var)}", f"component_id={_q(var)}", f"{title_argument}={_q(title)}", *options,
                     f"display_order={order}"]
        return f"{editor._gui(constructor)}(\n{_arguments('    ', arguments)}\n{data[:-1]}),\n)"

    target = _content_of(screen)
    row = next((element for element in target.view_elements
                if type(element) is ViewContainer and element.name.endswith("_cards")), None)
    literal = editor.collection(target, "view_elements")
    if kind is MetricCard and row is not None:  # beside the cards the page shows
        editor.add(row, "view_elements", code(_next_order(row)), lambda: var, need_var=True, separate=False)
    elif kind is MetricCard and target is not screen and literal is not None:
        # The page's first card starts the row of its cards, under the heading and above the rest.
        anchor = editor.src.top_statement(literal)
        cards = editor.free_name(f"{target.name.removesuffix('_page')}_cards")
        editor._define(var, code(1), anchor, False)
        editor._define(cards, (
            f"{editor._gui('ViewContainer')}(\n"
            f"    name={_q(cards)}, component_id={_q(cards)}, description=\"Cards\", display_order=1.5,\n"
            f"    {_classes(_CARDS_CLASSES)},\n"
            f"    view_elements={{\n        {var},\n    }},\n)"), anchor, False)
        editor.add_to_literal(literal, cards)
    else:  # after the elements shown in an order; those without one come last on the page
        editor.add(target, "view_elements", code(_next_order(target)), lambda: var, need_var=True, separate=False)

    def gui_check(guis: list[Any]) -> str | None:
        shown = any(element.name == var for gui in guis for found in _screens(gui) if found.name == screen.name
                    for element in _contents(found))
        return None if shown else f"screen {screen.name} does not show {var}"

    return screen.name, var, gui_check


def _computed(how: str, cls: Class, attribute: Property | None, where: str | None) -> str:
    return (f"{how} of {cls.name}.{attribute.name}" if attribute is not None else f"{how} of {cls.name}") + (
        f" where {_filter_expression(cls, where)}" if where else "")


def add_card(editor: ModelEditor, screen: str, title: str, value: str, where: str | None) -> Edited:
    how, cls, attribute = _parse_value(editor, value)
    # No trend: the pages have no earlier value to compare with, and would show a made-up one.
    screen_name, _, gui_check = _add_bound(editor, screen, title, MetricCard, "MetricCard", "metric_title",
                                           ["show_trend=False"], how, cls, attribute, None, where)
    return Edited(editor.finish(), f"added card '{title}' to screen {screen_name} ({_computed(how, cls, attribute, where)})",
                  lambda model: None, editor.warnings, gui_check=gui_check)


def add_link(editor: ModelEditor, screen: str, label: str, url: str) -> Edited:
    target = _find_screen(editor, screen)
    if not re.fullmatch(r"(https?://|/)\S+", url):
        raise EditError(f"'{url}' is not an address: write /route of the application, or http(s)://... (a route of "
                        "the backend needs its whole address)")
    if any(getattr(element, "label", None) == label and type(element).__name__ == "Link"
           for element in _contents(target)):
        raise EditError(f"screen {target.name} has a link '{label}' already")
    content = _content_of(target)
    var = editor.free_name(_snake(_screen_name(label)))
    code = _arguments(f"{editor._gui('Link')}(", [f"name={_q(var)}", f"component_id={_q(var)}",
                                                 f"description={_q(label)}", f"label={_q(label)}", f"url={_q(url)}",
                                                 _classes(_LINK_CLASSES),
                                                 f"display_order={_next_order(content)}"])[:-1] + ")"
    editor.add(content, "view_elements", code, lambda: var, need_var=True, separate=False)

    def gui_check(guis: list[Any]) -> str | None:
        shown = any(element.name == var for gui in guis for found in _screens(gui) if found.name == target.name
                    for element in _contents(found))
        return None if shown else f"screen {target.name} does not show {var}"

    return Edited(editor.finish(), f"added link '{label}' to screen {target.name} ({url})", lambda model: None,
                  editor.warnings, gui_check=gui_check)


def add_chart(editor: ModelEditor, screen: str, title: str, value: str, by: str, kind: str,
              where: str | None) -> Edited:
    how, cls, attribute = _parse_value(editor, value)
    label = _attribute_named(cls, by)
    screen_name, _, gui_check = _add_bound(editor, screen, title, Chart, _CHARTS[kind], "title",
                                           [_classes(_PANEL_CLASSES)], how, cls, attribute, label, where)
    return Edited(editor.finish(), f"added {kind} chart '{title}' to screen {screen_name} "
                                   f"({_computed(how, cls, attribute, where)} by {by})",
                  lambda model: None, editor.warnings, gui_check=gui_check)


def _button_code(editor: ModelEditor, button: str, screen_var: str, screen_name: str, title: str) -> str:
    """The code that defines the button ``button`` opening the screen ``screen_var``."""
    return (
        f"{button} = {editor._gui('Button')}(\n"
        f"    name={_q(button)}, component_id={_q(button)}, description={_q('Go to the ' + title.lower())}, "
        f"label={_q(title)},\n"
        f"    buttonType={editor._gui('ButtonType')}.CustomizableButton, "
        f"actionType={editor._gui('ButtonActionType')}.Navigate, targetScreen={screen_var},\n)\n"
        f"{button}.events = {{{editor._gui('Event')}(name={_q(button + '_click')}, actions={{\n"
        f"    {editor._gui('Transition')}(name={_q('go_to_' + _snake(screen_name))}, "
        f"target_screen={screen_var}),\n}})}}\n"
    )


def _navigation_button(editor: ModelEditor, main: Screen, anchor: ast.stmt, button: str, screen_var: str,
                       screen_name: str, title: str) -> str:
    """The code that defines the button ``button`` opening the screen ``screen_var`` (named ``screen_name``,
    titled ``title``) and puts it on the screen ``main``; the code goes before the statement ``anchor``."""
    code = _button_code(editor, button, screen_var, screen_name, title)
    main_var = editor.var_of(main)
    if main_var is not None:
        return code + f"{main_var}.view_elements.add({button})\n"
    elements = editor.collection(main, "view_elements")
    if elements is None or editor.src.start(elements) < editor.src.start(anchor):
        raise EditError(f"cannot add the button that opens {screen_name} to the main screen {main.name}: "
                        "give that screen a variable")
    editor.add_to_literal(elements, button)
    return code


def _new_main_pages(editor: ModelEditor, removed: Sequence[Any], removed_vars: Sequence[str]) -> str:
    """When ``removed`` takes the main screen of a GUI model, make the first screen left (in the
    generator's order: module, then screen name) the main page, with a button to each other screen left
    (the generated app navigates with buttons only); returns what changed, for the summary."""
    said = ""
    for gui in editor.gui_models():
        screens = _screens(gui)
        left = [screen for screen in screens if screen not in removed]
        if not left or not any(screen.is_main_page and screen in removed for screen in screens) \
                or any(screen.is_main_page for screen in left):
            continue
        main = left[0]
        _make_main_page(editor, main)
        if _navigation_menus(editor, left):  # the menu reaches every screen: no button needed
            said += f"; {main.name} is the main page now"
            continue
        module = next(module for module in gui.modules if main in module.screens)
        literal = editor.collection(module, "screens")
        if literal is None:
            raise EditError(f"cannot find where the screens of {gui.name} are written in {editor.path.name}; "
                            "edit it by hand")
        anchor = editor.src.top_statement(literal)
        opened = {item.targetScreen for item in _contents(main) if isinstance(item, Button)}
        code, buttons = "", []
        for target in left[1:]:
            target_var = editor.var_of(target)
            if target in opened:
                continue
            if target_var is None:
                editor.warnings.append(f"no button on {main.name} opens screen {target.name}: it has no variable "
                                       "to refer to it by; give it one and add the button")
                continue
            name = f"to_{_snake(target.name)}"
            button = name if name in removed_vars else editor.free_name(name)
            code += _navigation_button(editor, main, anchor, button, target_var, target.name,
                                       target.description or target.name)
            buttons.append(target.name)
        if code:
            editor.src.insert_before(anchor, code, separate=True)
        said += f"; {main.name} is the main page now" + (f", with a button to {', '.join(buttons)}" if buttons else "")
    return said


def _make_main_page(editor: ModelEditor, screen: Screen) -> None:
    """Plan ``is_main_page=True`` on ``screen``: in its constructor call, or after its variable."""
    call = editor.call_of(screen)
    if call is not None:
        flag = next((keyword_ for keyword_ in call.keywords if keyword_.arg == "is_main_page"), None)
        if flag is not None:
            editor.src.replace(editor.src.start(flag.value), editor.src.end(flag.value), "True")
        else:
            editor.src.add_item(sorted([*call.args, *call.keywords], key=editor.src.start), editor.src.end(call) - 1,
                                "is_main_page=True")
        return
    var = editor.var_of(screen)
    if var is None or editor.src.assignment(var) is None:
        raise EditError(f"cannot find where screen {screen.name} is written to make it the main page; "
                        "edit it by hand")
    editor.src.insert_after(editor.src.assignment(var), f"{var}.is_main_page = True\n")


def _add_columns(editor: ModelEditor, tables: Sequence[Table], added: Sequence[tuple[str, str]]) -> list[str]:
    """Add a column per new attribute (name, variable) to each of ``tables``; returns the names of those
    that got them. A table whose columns the file does not write where a command can add to them (a
    function builds it) keeps its columns: a warning gives the code to add by hand."""
    done = []
    for table in tables:
        columns = [f"{editor._gui('FieldColumn')}(label={_q(_label(name))}, field={var})" for name, var in added]
        try:
            for (name, _), column in zip(added, columns):
                editor.add(table, "columns", column,
                           lambda name=name: editor.free_name(f"{editor.var_of(table) or table.name}_{name}"),
                           need_var=False, separate=False)
        except EditError:
            editor.warnings.append(
                f"table {table.name} got no column for {', '.join(name for name, _ in added)}: {editor.path.name} "
                f"does not write its columns where a command can add to them; add {', '.join(columns)} by hand")
            continue
        done.append(table.name)
    return done


def _inherited_columns(editor: ModelEditor, cls: Class, parent: Class) -> str:
    """Give the tables that follow ``cls`` (see ``_tables_following``) a column per attribute it now
    inherits from ``parent``, after their columns; returns what changed, for the summary."""
    owners = [owner for owner in _lineage(parent) if owner not in cls.all_parents()]
    attributes = [attribute for owner in owners for attribute in sort_by_timestamp(owner.attributes)]
    tables = _tables_following(editor, cls)
    if not attributes or not tables:
        return ""
    fields = {key: var for owner in owners for key, var in _attribute_variables(editor, owner).items()}
    names = _add_columns(editor, tables, [(attribute.name, fields[id(attribute)]) for attribute in attributes])
    return (f"; column{'s' if len(attributes) > 1 else ''} {', '.join(a.name for a in attributes)} "
            f"added to table {', '.join(names)}")


def _gui_removals(editor: ModelEditor, cls: Class) -> list[Any]:
    """The GUI elements that go with the class ``cls``: its tables, the screens that show nothing else,
    and the buttons that open those screens."""
    tables = editor.tables_of(cls)
    removed: list[Any] = list(tables)
    screens = [screen for gui in editor.gui_models() for screen in _screens(gui)]
    for screen in screens:
        shown = [element for element in _contents(screen) if isinstance(element, Table)]
        if shown and all(table in tables for table in shown):
            removed.append(screen)
    gone = [element for element in removed if isinstance(element, Screen)]
    # What the removed screens hold goes with them, as do the buttons elsewhere that open them. What a
    # screen that stays shows too (a menu) stays.
    stays = [element for screen in screens if screen not in gone for element in _contents(screen)]
    held = [element for screen in gone for element in _contents(screen)
            if not isinstance(element, Menu) and not any(element is other for other in stays)]
    candidates = [element for screen in screens for element in _contents(screen)] + list(editor.namespace.values())
    for element in [*held, *candidates]:
        if element not in removed and (element in held or isinstance(element, Button) and element.targetScreen in gone):
            removed.append(element)
    return removed


def _remove_gui_element(editor: ModelEditor, element: Any, drops: dict[int, tuple[ast.AST, set[int]]],
                        removed_vars: list[str]) -> None:
    """Plan the removal of a GUI element: its variable and every statement on it, and its place in any set."""
    var = editor.var_of(element)
    if var is None:
        call = editor.call_of(element)
        if call is None:
            raise EditError(f"cannot find where {type(element).__name__.lower()} {element.name} is written; "
                            "edit it by hand")
        if editor.src.is_deleted(call):
            return
        for literal in (node for node in ast.walk(editor.src.tree) if isinstance(node, (ast.Set, ast.List))):
            if any(item is call for item in literal.elts):
                drops.setdefault(id(literal), (literal, set()))[1].add(
                    next(index for index, item in enumerate(literal.elts) if item is call))
                return
        raise EditError(f"cannot find where {type(element).__name__.lower()} {element.name} is written; "
                        "edit it by hand")
    removed_vars.append(var)
    _forget_var(editor, var)
    for statement in editor.src.tree.body:  # screen.view_elements.add(button): a statement on what it holds
        target = statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
            else getattr(statement.value, "func", None) if isinstance(statement, ast.Expr) else None
        while isinstance(target, ast.Attribute):
            target = target.value
        if isinstance(target, ast.Name) and target.id == var and not editor.src.is_deleted(statement):
            editor.src.delete_statement(statement)
    for literal in (node for node in ast.walk(editor.src.tree) if isinstance(node, (ast.Set, ast.List))):
        for index, item in enumerate(literal.elts):
            if isinstance(item, ast.Name) and item.id == var:
                drops.setdefault(id(literal), (literal, set()))[1].add(index)
    # screen.view_elements.add(button), module.screens.add(screen)
    for statement in editor.src.tree.body:
        value = getattr(statement, "value", None)
        if isinstance(statement, ast.Expr) and isinstance(value, ast.Call) and len(value.args) == 1 \
                and isinstance(value.args[0], ast.Name) and value.args[0].id == var:
            editor.src.delete_statement(statement)


# Operation name -> what it does, the arguments it takes (for help and batch lines) and its function.
OPERATIONS: dict[str, tuple[str, Callable[..., Edited]]] = {
    "add-class": ("add a class: NAME [ATTRIBUTE...] [--parent CLASS] [--abstract] [--screen [--title TITLE] "
                  "[--route /PATH] [--rows N]]", add_class),
    "add-attribute": ("add attributes to a class: CLASS ATTRIBUTE... [--derived]", add_attribute),
    "add-parent": ("make a class specialize another (a generalization): CLASS PARENT", add_parent),
    "add-association": ("add an association: NAME END END [--composite]", add_association),
    "add-method": ("add a method: CLASS 'name(param:type, ...) -> type' [--code CODE]", add_method),
    "add-constraint": ("add an OCL invariant: CLASS EXPRESSION [--name NAME]", add_constraint),
    "add-screen": ("add a screen with a table of a class to the GUI model: CLASS [--title TITLE] [--route /PATH] "
                   "[--rows N]",
                   add_screen),
    "add-page": ("add a screen with a heading, for cards and charts: TITLE [--route /PATH] [--main]", add_page),
    "add-menu": ("add a menu with a link to every screen, on every screen", add_menu),
    "add-column": ("add a table column showing an attribute of a related object: CLASS END.ATTRIBUTE [--label LABEL]",
                   add_column),
    "order-columns": ("put columns first in the tables of a class, in the order given: CLASS COLUMN...",
                      order_columns),
    "add-card": ("add a metric card to a screen: SCREEN TITLE sum:CLASS.ATTRIBUTE|count:CLASS [--where CONDITION]",
                 add_card),
    "add-link": ("add a link to a screen: SCREEN LABEL URL", add_link),
    "add-chart": ("add a chart to a screen: SCREEN TITLE VALUE --by ATTRIBUTE [--kind bar|line|pie] "
                  "[--where CONDITION]", add_chart),
    "set-enum": ("create an enumeration or set its literals: NAME LITERAL...", set_enum),
    "rename": ("rename a class, enumeration, member, literal, association or constraint: PATH NEW_NAME "
               "[--keep-pages]", rename),
    "remove": ("remove an element (a class with its associations): PATH", remove),
    "set-attribute": ("change an attribute's type, ?, default in place: CLASS.ATTRIBUTE :TYPE[?][=DEFAULT]",
                      set_attribute),
    "set-multiplicity": ("change an association end's multiplicity in place: ASSOCIATION.END|CLASS.END MULTIPLICITY",
                         set_multiplicity),
    "set-abstract": ("make a class abstract (true) or concrete (false): CLASS true|false", set_abstract),
    "remove-parent": ("make a class no longer specialize a parent (the class stays): CLASS PARENT", remove_parent),
}


def run_operation(path: Path, text: str, operation: Operation, use: Sequence[str] = ()) -> Edited:
    """Apply ``operation`` to the text of the model file at ``path``.

    Raises:
        EditError: The operation does not apply to this model, or the file cannot be edited where it is needed.
    """
    editor = ModelEditor(path, text, use)
    return OPERATIONS[operation.name][1](editor, **operation.args)
