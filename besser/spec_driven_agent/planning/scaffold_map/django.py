"""Scaffold map extractor for Django projects.

A project root is a directory holding a ``manage.py`` that mentions Django
(``django/myproject/`` for the agent's ``generate_django``), found by layout,
never by the generator that was asked for. Per root it reads:

* the settings module ``manage.py`` names (``INSTALLED_APPS``, ``MIDDLEWARE``,
  ``DATABASES``, the end of the file), following ``from .base import *`` and
  ``A + B`` list operands, and the URLconf ``ROOT_URLCONF`` names (its
  ``urlpatterns``, ``include()`` calls and ``django.urls`` import);
* every local app: an ``INSTALLED_APPS`` entry that is a directory under the
  root, or a directory with ``apps.py``/``models.py`` (reported when it is not
  installed). Per app: model classes with their fields, ``Meta``, ``clean()``
  and method stubs; admin registrations, ``ModelAdmin`` classes and admin
  actions; views, ModelForms, DRF serializers, the app ``urlpatterns`` and the
  templates, when they exist;
* ``requirements.txt`` and the migrations on disk (the generator ships none).

Generated Django files do not always parse (``models.py`` of a model with
OCL constraints, the GUI variant's views/forms/urls lose newlines). Such a
file is read with a light line scanner instead, so its entities still appear
with approximate spans, while the core lists the file under "fix these
first". Everything is keyed to domain entities by class names (models,
``Meta.model``) and name tokens (``create_book``, ``book_list.html``), never by
the generator's templates.
"""

from __future__ import annotations

import ast
import bisect
import dataclasses
import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import Optional

from besser.spec_driven_agent.planning.action_inventory import _placeholder_reason
from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_CLASS,
    KIND_FILE,
    KIND_FUNCTION,
    KIND_INSERT,
    KIND_ROUTE,
    KIND_STUB,
    Anchor,
    EntityMap,
    Recipe,
    ScaffoldMap,
)
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex, node_start

logger = logging.getLogger(__name__)

FAMILY = "django"

_TEST_DIRS = frozenset({"tests", "test"})
_MODEL_BASES = frozenset({"Model", "AbstractUser", "AbstractBaseUser", "PermissionsMixin", "PolymorphicModel"})
_CHOICES_BASES = frozenset({"TextChoices", "IntegerChoices", "Choices"})
_RELATION_CALLS = {"ForeignKey": "FK", "ManyToManyField": "M2M", "OneToOneField": "1:1"}
_VIEW_FILES = frozenset({"views.py", "api.py", "viewsets.py"})
_VIEW_DIRS = frozenset({"views", "api"})
_VIEW_SUFFIX_RE = re.compile(r"(?:List|Detail|Create|Update|Delete|Edit|Form)?(?:APIView|ViewSet|View)$")
# Third-party apps the scaffold or a typical change installs: INSTALLED_APPS label -> requirement name.
_PACKAGE_OF = {
    "jazzmin": "django-jazzmin",
    "rest_framework": "djangorestframework",
    "corsheaders": "django-cors-headers",
    "django_filters": "django-filter",
    "drf_spectacular": "drf-spectacular",
    "rest_framework_simplejwt": "djangorestframework-simplejwt",
}
_IMPORTED_APPS = ("rest_framework", "corsheaders", "django_filters", "drf_spectacular")
_STUB_STATUS = {"an empty action body": "empty body stub"}
_MAX_WARNINGS = 8
_MAX_LABEL_NAMES = 6
# Entity anchors are listed in this order (locate_code truncates long answers):
# the model and where to edit it first, the admin next, the GUI/API pieces last.
_ANCHOR_ORDER = {
    "model": 0, "add_field": 1, "clean": 2, "stub": 3, "method": 4, "admin": 5, "admin_class": 6,
    "admin_action": 7, "form": 8, "serializer": 9, "view": 10, "url": 11, "template": 12,
}

_SETTINGS_ENV_RE = re.compile(r"""DJANGO_SETTINGS_MODULE['"]\s*,\s*['"]([\w.]+)['"]""")
_ROOT_URLCONF_RE = re.compile(r"""^ROOT_URLCONF\s*=\s*['"]([\w.]+)['"]""", re.M)
_ENGINE_RE = re.compile(r"""['"]ENGINE['"]\s*:\s*['"](?:django\.db\.backends\.)?([\w.]+)['"]""")
# Line scanner (files that do not parse). A header may be glued to the end of the
# previous statement by the generator's newline stripping: "...(errors)class Author(Person):".
# No two quantifiers here can match the same run of characters (``\s*`` twice around an
# optional group, ``\s*[^:]+``): on a long line of spaces that backtracks quadratically.
_CLASS_LINE_RE = re.compile(
    r"^(?P<pre>.*?)\bclass\s+(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<bases>[^()]*)\)\s*)?:\s*(?:#.*)?$")
_DEF_LINE_RE = re.compile(r"^(?P<pre>.*?)\b(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<rest>.*)$")
# The return annotation stops at the next ')' as well as ':', so each ')' the greedy
# params group backs off to costs only the text up to the following one.
_DEF_REST_RE = re.compile(r"^(?P<params>.*)\)\s*(?:->[^:)]*)?:\s*(?:#.*)?$")
_FIELD_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(?P<call>[\w.]+)\(")
_ATTR_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(?P<value>\S.*)$")
_DECORATOR_LINE_RE = re.compile(r"^(?P<indent>\s*)@(?P<name>[\w.]+)\s*(?P<args>\(.*)?$")
_TOKEN_RE = re.compile(r"""(['"])(?P<s>[^'"\n]*)\1|(?P<n>\b[A-Za-z_]\w*\b)""")
_ATTR_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\((\s*\))?")
_NAME_CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\(")
_REGISTER_LINE_RE = re.compile(r"^(?:[\w]+\.)*site\.(?P<un>un)?register\(")
_URLPATTERNS_RE = re.compile(r"^urlpatterns\s*(?::[^=\n]*)?=\s*([\[(])", re.M)
_PATH_CALL_RE = re.compile(r"(?<![\w.])(?P<func>re_path|path|url)\(")
_ROUTER_REGISTER_RE = re.compile(r"^(?P<recv>\w+)\.register\(", re.M)
_INCLUDE_ARG_RE = re.compile(r"""^\s*include\(\s*(?:\(\s*)?(?:(['"])(?P<inc>[\w.]+)\1|(?P<obj>[\w.]+))""")
_NAME_KW_RE = re.compile(r"""\bname\s*=\s*(['"])(?P<name>[^'"]+)\1""")
_STRING_ARG_RE = re.compile(r"""^\s*r?(['"])(?P<s>.*?)\1\s*(?:,|\)|$)""", re.S)
_VIEW_ARG_RE = re.compile(r"^\s*(?P<view>[A-Za-z_][\w.]*)(?P<as_view>\.as_view\()?")
_EXTENDS_RE = re.compile(r"""\{%-?\s*extends\s+['"]([^'"]+)['"]""")
_BLOCK_RE = re.compile(r"\{%-?\s*block\s+(\w+)")
# TEMPLATES 'DIRS' with at least one entry: the project-level templates/ directory is searched.
_TEMPLATE_DIRS_RE = re.compile(r"""['"]DIRS['"]\s*:\s*[\[(]\s*[^\])\s]""")
# Parameters a view takes from its URL (``<int:pk>``, ``<slug:book_slug>``), by name.
_URL_KWARGS = ("pk", "id", "slug", "uuid")
_URL_KWARG_SUFFIXES = tuple("_" + k for k in _URL_KWARGS)                 # book_id, author_pk, post_slug
_ROUTE_KWARG_RE = re.compile(r"(?:<(?:\w+:)?|\(\?P<)(\w+)>")
_MIGRATION_RE = re.compile(r"^\d{4}_\w+\.py$")
_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
# ``[ \t]``, not ``\s``: with re.M, ``^\s*`` at every line start would run over all the blank lines below it.
_IMPORT_LINE_RE = re.compile(r"^[ \t]*(?:from|import)[ \t]+([A-Za-z_]\w*)", re.M)
_DJANGO_URLS_IMPORT_RE = re.compile(r"^from\s+django\.(?:urls|conf\.urls)\s+import\s+(?P<names>.+)$")
# Settings split over modules: ``from .base import *`` (cookiecutter ``config/settings/local.py``).
# The lookahead keeps the two whitespace runs apart when there are neither dots nor a module name.
_STAR_IMPORT_RE = re.compile(
    r"^from[ \t]+(?=[.\w])(?P<dots>\.*)(?P<module>(?:\w[\w.]*)?)[ \t]+import[ \t]+\*", re.M)
_PLAIN_INSTALLED_RE = re.compile(r"^INSTALLED_APPS\s*(?::[^=\n]*)?=(?!=)", re.M)
_PRIMARY_KEY_RE = re.compile(r"\bprimary_key\s*=\s*True\b")
# Primary keys a ``<int:...>`` converter and an ``id=`` lookup work with.
_INT_PK_FIELDS = frozenset({
    "AutoField", "BigAutoField", "SmallAutoField", "IntegerField", "BigIntegerField", "SmallIntegerField",
    "PositiveIntegerField", "PositiveBigIntegerField", "PositiveSmallIntegerField",
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dotted(node: Optional[ast.AST]) -> str:
    """``a.b.c`` for a Name/Attribute chain, else ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _last(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def _string(node: Optional[ast.AST]) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keyword(call: ast.Call, name: str) -> Optional[ast.AST]:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _targets(statement: ast.stmt) -> list[str]:
    if isinstance(statement, ast.Assign):
        nodes = list(statement.targets)
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        nodes = [statement.target]
    else:
        nodes = []
    return [n.id for n in nodes if isinstance(n, ast.Name)]


def _value_tokens(node: Optional[ast.AST]) -> list[str]:
    """Names, strings and booleans appearing in an expression (shallow, for Meta.model, list_display...)."""
    out: list[str] = []
    for sub in ast.walk(node) if node is not None else []:
        if isinstance(sub, ast.Name):
            out.append(sub.id)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, (str, bool)):
            out.append(str(sub.value))
    return out


def _text_tokens(text: str) -> list[str]:
    return [m.group("s") if m.group("s") is not None else m.group("n") for m in _TOKEN_RE.finditer(text)]


def _short_list(names: list[str], limit: int = _MAX_LABEL_NAMES) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" +{len(names) - limit}" if len(names) > limit else "")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _is_under(path: str, root: str) -> bool:
    return not root or path == root or path.startswith(root.rstrip("/") + "/")


def _join(root: str, *parts: str) -> str:
    return posixpath.join(root, *parts) if root else posixpath.join(*parts)


def _required_params(args: ast.arguments) -> list[str]:
    """Parameters a call must supply: positional ones without a default, then keyword-only ones."""
    positional = [a.arg for a in args.posonlyargs + args.args]
    required = positional[:len(positional) - len(args.defaults)] if args.defaults else positional
    required += [a.arg for a, default in zip(args.kwonlyargs, args.kw_defaults) if default is None]
    return required


def _match_bracket(text: str, open_pos: int) -> Optional[int]:
    """Offset of the bracket closing the one at ``open_pos`` (strings and comments skipped), or None."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    i, n = open_pos, len(text)
    while i < n:
        ch = text[i]
        if ch == "#":
            nl = text.find("\n", i)
            if nl < 0:
                return None
            i = nl
            continue
        if ch in "'\"":
            quote = text[i:i + 3] if text[i:i + 3] in ("'''", '"""') else ch
            j = i + len(quote)
            while j < n and not text.startswith(quote, j):
                if text[j] == "\\":
                    j += 1
                elif len(quote) == 1 and text[j] == "\n":
                    break
                j += 1
            i = j + len(quote)
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in ")]}":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return i
        i += 1
    return None


def _split_args(text: str) -> list[str]:
    """Top-level comma-separated arguments of a call's inside text (brackets and strings respected)."""
    parts: list[str] = []
    depth, start, i, quote = 0, 0, 0, ""
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    if text[start:].strip():
        parts.append(text[start:])
    return [p.strip() for p in parts]


class _Text:
    """A file's lines joined with '\\n', so offsets map exactly onto ``SourceIndex.lines`` numbers."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.joined = "\n".join(lines)
        self._starts = [0]
        for line in lines[:-1]:
            self._starts.append(self._starts[-1] + len(line) + 1)

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self._starts, offset)

    def offset_of(self, line_no: int, col: int = 0) -> int:
        return self._starts[line_no - 1] + col if 1 <= line_no <= len(self._starts) else len(self.joined)

    def last_code_line(self) -> int:
        return next((n for n in range(len(self.lines), 0, -1) if self.lines[n - 1].strip()), 0)


# ---------------------------------------------------------------------------
# A light, parser-independent view of Python definitions
# ---------------------------------------------------------------------------

@dataclass
class _Field:
    """A model (or form) field assignment in a class body."""

    name: str
    call: str               # "CharField", "ForeignKey", ...
    start: int
    end: int
    target: str = ""        # the related model of a relation field
    target_is_str: bool = False
    has_on_delete: bool = True
    primary_key: bool = False


@dataclass
class _Def:
    """A class or function, read from the AST or, for a file that does not parse, from its lines."""

    kind: str               # "class" | "def"
    name: str
    start: int              # first decorator
    line: int               # the class/def line
    end: int
    bases: list[str] = field(default_factory=list)
    decorators: list[tuple[str, list[str]]] = field(default_factory=list)   # (dotted name, tokens of the args)
    required: Optional[list[str]] = None    # required parameters incl. self; None = unknown
    body: list["_Def"] = field(default_factory=list)                        # methods, nested classes
    fields: list[_Field] = field(default_factory=list)
    attrs: dict[str, list[str]] = field(default_factory=dict)              # class-level NAME = value tokens
    stub: Optional[str] = None
    stub_line: int = 0
    calls: list[tuple[str, str, int]] = field(default_factory=list)        # obj.m(...) calls: (obj, m, n args)
    names_called: set[str] = field(default_factory=set)

    def child(self, name: str) -> Optional["_Def"]:
        return next((c for c in self.body if c.name == name), None)


def _wrapped_view(node: Optional[ast.AST]) -> str:
    """The view a ``path()`` target names: ``views.x``, ``X.as_view()``, or what a wrapper passes on.

    ``login_required(views.x)``, ``csrf_exempt(X.as_view())`` and
    ``cache_page(60)(views.x)`` all name ``views.x``/``X``; '' when nothing does.
    """
    if isinstance(node, ast.Call):
        func = _dotted(node.func)
        if _last(func) == "as_view":
            return func[:-len(".as_view")]
        for arg in node.args:
            view = _wrapped_view(arg)
            if view:
                return view
        return ""
    return _dotted(node)


def _names_in(node: Optional[ast.AST]) -> list[str]:
    """Every name and attribute in an expression, sorted (``login_required(views.x)`` -> login_required, views, x)."""
    names = set()
    for sub in ast.walk(node) if node is not None else []:
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
    return sorted(names)


def _field_from_ast(name: str, value: Optional[ast.AST], statement: ast.stmt) -> Optional[_Field]:
    if not isinstance(value, ast.Call):
        return None
    call = _last(_dotted(value.func))
    if not (call.endswith("Field") or call in _RELATION_CALLS):
        return None
    out = _Field(name=name, call=call, start=statement.lineno, end=statement.end_lineno or statement.lineno)
    primary_key = _keyword(value, "primary_key")
    out.primary_key = isinstance(primary_key, ast.Constant) and primary_key.value is True
    if call in _RELATION_CALLS:
        target = value.args[0] if value.args else _keyword(value, "to")
        out.target = _string(target) or _dotted(target)
        out.target_is_str = _string(target) is not None
        if call != "ManyToManyField":
            out.has_on_delete = _keyword(value, "on_delete") is not None or len(value.args) >= 2
    return out


def _def_from_ast(node: ast.AST) -> Optional[_Def]:
    if isinstance(node, ast.ClassDef):
        out = _Def(kind="class", name=node.name, start=node_start(node), line=node.lineno,
                   end=node.end_lineno or node.lineno,
                   bases=[d for d in (_dotted(b) for b in node.bases) if d],
                   decorators=[(_dotted(d.func if isinstance(d, ast.Call) else d), _value_tokens(d))
                               for d in node.decorator_list])
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child = _def_from_ast(item)
                if child is not None:
                    out.body.append(child)
            elif isinstance(item, (ast.Assign, ast.AnnAssign)) and _targets(item):
                name = _targets(item)[0]
                found = _field_from_ast(name, item.value, item)
                if found is not None:
                    out.fields.append(found)
                elif item.value is not None:
                    out.attrs[name] = _value_tokens(item.value)
        return out
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        out = _Def(kind="def", name=node.name, start=node_start(node), line=node.lineno,
                   end=node.end_lineno or node.lineno, required=_required_params(node.args),
                   decorators=[(_dotted(d.func if isinstance(d, ast.Call) else d), _value_tokens(d))
                               for d in node.decorator_list])
        out.stub = _placeholder_reason(node)
        if out.stub and node.body:
            out.stub_line = node.body[-1].lineno
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
                    out.calls.append((sub.func.value.id, sub.func.attr, len(sub.args) + len(sub.keywords)))
                elif isinstance(sub.func, ast.Name):
                    out.names_called.add(sub.func.id)
        return out
    return None


def _glued_indent(pre: str, header_complete: bool) -> Optional[int]:
    """Indentation of a class/def header found on a line, or None when it is not a header.

    A header normally starts after whitespace only. When the generator glued it to
    the end of the previous statement, the whitespace between the two is the
    header's own indentation (``raise ValidationError(errors)    def is_x(self):``).
    """
    if not pre.strip():
        return None if "\t" in pre else len(pre)
    stripped = pre.rstrip()
    if not header_complete or not (stripped[-1] in ")]}" or stripped[-1].isalnum()):
        return None
    return len(pre) - len(stripped)


class _LineScanner:
    """Definitions of a file that does not parse, from its lines (approximate spans)."""

    def __init__(self, text: _Text) -> None:
        self.t = text
        self.lines = text.lines
        self.headers: list[tuple[int, int, str, re.Match]] = []      # (line, indent, kind, match)
        for n, line in enumerate(self.lines, 1):
            if "class" in line:
                m = _CLASS_LINE_RE.match(line)
                if m:
                    indent = _glued_indent(m.group("pre"), True)
                    if indent is not None:
                        self.headers.append((n, indent, "class", m))
                        continue
            if "def" in line:
                m = _DEF_LINE_RE.match(line)
                if m:
                    indent = _glued_indent(m.group("pre"), bool(_DEF_REST_RE.match(m.group("rest"))))
                    if indent is not None:
                        self.headers.append((n, indent, "def", m))

    @staticmethod
    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    def _is_statement(self, n: int, max_indent: int) -> bool:
        line = self.lines[n - 1]
        text = line.strip()
        return bool(text) and not text.startswith(("#", ")", "]", "}")) and self._indent(line) <= max_indent

    def _decorated_start(self, line_no: int, indent: int) -> int:
        start = line_no
        while start > 1:
            m = _DECORATOR_LINE_RE.match(self.lines[start - 2])
            if not m or len(m.group("indent")) != indent:
                break
            start -= 1
        return start

    def _trim(self, start: int, end: int) -> int:
        while end > start and (not self.lines[end - 1].strip() or self.lines[end - 1].lstrip().startswith("#")):
            end -= 1
        return end

    def defs(self, start: int = 1, end: Optional[int] = None, indent: int = 0) -> list[_Def]:
        """The definitions at ``indent`` within lines ``start..end``."""
        end = len(self.lines) if end is None else end
        mine = [h for h in self.headers if start <= h[0] <= end and h[1] == indent]
        out: list[_Def] = []
        for i, (line_no, _, kind, m) in enumerate(mine):
            first = self._decorated_start(line_no, indent)
            stop = end
            if i + 1 < len(mine):
                stop = self._decorated_start(mine[i + 1][0], indent) - 1
            # A statement at this indentation (or less) also ends the block.
            for n in range(line_no + 1, stop + 1):
                if self._is_statement(n, indent) and not _DECORATOR_LINE_RE.match(self.lines[n - 1]):
                    stop = n - 1
                    break
            out.append(self._build(kind, m, first, line_no, self._trim(line_no, stop), indent))
        return out

    def _build(self, kind: str, m: re.Match, start: int, line_no: int, end: int, indent: int) -> _Def:
        decorators = []
        for n in range(start, line_no):
            dm = _DECORATOR_LINE_RE.match(self.lines[n - 1])
            if dm:
                decorators.append((dm.group("name"), _text_tokens(dm.group("args") or "")))
        d = _Def(kind=kind, name=m.group("name"), start=start, line=line_no, end=end, decorators=decorators)
        span_text = "\n".join(self.lines[line_no - 1:end])
        if kind == "class":
            d.bases = [b.strip() for b in (m.group("bases") or "").split(",") if re.fullmatch(r"\s*[\w.]+\s*", b)]
            d.body = self.defs(line_no + 1, end, indent + 4)
            self._class_members(d, line_no + 1, end, indent + 4)
            return d
        rest = _DEF_REST_RE.match(m.group("rest"))
        if rest is not None:
            try:
                fn = ast.parse(f"def _f({rest.group('params')}): pass").body[0]
                d.required = _required_params(fn.args)
            except Exception:  # whatever the parameter text is, an unknown signature is fine
                d.required = None
        body = span_text.split("\n", 1)[1] if "\n" in span_text else ""
        for n in range(line_no, end + 1):
            if "raise NotImplementedError" in self.lines[n - 1]:
                d.stub, d.stub_line = "NotImplementedError", n
                break
        for cm in _ATTR_CALL_RE.finditer(body):
            d.calls.append((cm.group(1), cm.group(2), 0 if cm.group(3) else 1))
        d.names_called = set(_NAME_CALL_RE.findall(body))
        return d

    def _class_members(self, d: _Def, start: int, end: int, indent: int) -> None:
        """Fields and simple attributes of a class body, outside its methods and nested classes."""
        taken = {n for child in d.body for n in range(child.start, child.end + 1)}
        n = start
        while n <= end:
            line = self.lines[n - 1]
            if n in taken or self._indent(line) != indent:
                n += 1
                continue
            stop = n
            fm = _FIELD_LINE_RE.match(line)
            am = _ATTR_LINE_RE.match(line)
            if fm and len(fm.group("indent")) == indent:
                open_pos = self.t.offset_of(n, fm.end() - 1)
                close = _match_bracket(self.t.joined, open_pos)
                stop = self.t.line_of(close) if close is not None else n
                stop = min(max(stop, n), end)
                call = _last(fm.group("call"))
                inside = self.t.joined[open_pos + 1:close] if close is not None else line[fm.end():]
                if call.endswith("Field") or call in _RELATION_CALLS:
                    f = _Field(name=fm.group("name"), call=call, start=n, end=stop,
                               primary_key=bool(_PRIMARY_KEY_RE.search(inside)))
                    if call in _RELATION_CALLS:
                        args = _split_args(inside)
                        first = args[0] if args else ""
                        sm = re.fullmatch(r"""(['"])([\w.]+)\1""", first)
                        f.target = sm.group(2) if sm else (first if re.fullmatch(r"[\w.]+", first) else "")
                        f.target_is_str = sm is not None
                        if call != "ManyToManyField":
                            f.has_on_delete = "on_delete" in inside or len(args) >= 2
                    d.fields.append(f)
                else:
                    d.attrs[fm.group("name")] = _text_tokens(inside)
            elif am and len(am.group("indent")) == indent:
                value = am.group("value")
                if value[:1] in "[({":
                    open_pos = self.t.offset_of(n, am.start("value"))
                    close = _match_bracket(self.t.joined, open_pos)
                    if close is not None:
                        stop = min(max(self.t.line_of(close), n), end)
                        value = self.t.joined[open_pos:close + 1]
                d.attrs[am.group("name")] = _text_tokens(value)
            n = stop + 1


# ---------------------------------------------------------------------------
# Module-level facts: settings lists, urlpatterns, admin registrations
# ---------------------------------------------------------------------------

@dataclass
class _Literal:
    """One list/tuple literal that contributes entries to a list setting."""

    file: str
    open_line: int
    close_line: Optional[int]               # None = never closed (a file that does not parse)
    close_col: int                          # column of the closing bracket on close_line
    items: list[tuple[str, int, int, int]]  # (string value or '', start line, end line, start column)
    via: str = ""                           # the list name it was reached through (``LOCAL_APPS``), if any


@dataclass
class _ListSetting:
    """``INSTALLED_APPS``/``MIDDLEWARE``: where it is assigned and the literals it is made of.

    ``A + B`` operands naming earlier module-level lists and ``+=`` statements
    are followed; an operand that cannot be read statically (a call, an
    environment lookup, an unknown name) makes the setting unresolved, so
    "X is not in INSTALLED_APPS" is never claimed from a partial list.
    """

    name: str
    start: int
    end: int
    literals: list[_Literal] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)      # append()/extend(), an `if` block, a module extending it
    resolved: bool = True

    @property
    def items(self) -> list[tuple[str, int, int, int]]:
        return [item for literal in self.literals for item in literal.items]

    def values(self) -> list[str]:
        return [item[0] for item in self.items if item[0]] + self.extra


@dataclass
class _Settings:
    rel: str                                # the module that assigns INSTALLED_APPS (anchors point here)
    named: str = ""                         # the module manage.py names, when it only extends ``rel``
    lists: dict[str, _ListSetting] = field(default_factory=dict)
    blocks: dict[str, tuple[int, int]] = field(default_factory=dict)    # DATABASES, TEMPLATES -> span
    last_statement: Optional[tuple[int, int]] = None
    root_urlconf: str = ""
    engine: str = ""

    def installed(self) -> list[str]:
        setting = self.lists.get("INSTALLED_APPS")
        return setting.values() if setting else []

    @property
    def apps_known(self) -> bool:
        """INSTALLED_APPS was found and read completely."""
        setting = self.lists.get("INSTALLED_APPS")
        return setting is not None and setting.resolved


@dataclass
class _UrlEntry:
    func: str               # path / re_path / url / register (a DRF router)
    route: str
    view: str               # dotted view reference: 'views.create_book', 'HomeView', 'admin.site.urls'
    name: str
    include: str            # the included module ('myapp.urls'), '?' when not a literal, else ''
    start: int
    end: int
    refs: list[str] = field(default_factory=list)   # every name in the target: login_required(views.x) -> x too


@dataclass
class _UrlConf:
    rel: str
    app: Optional["_App"]                   # None = the project URLconf
    start: int = 0
    line: int = 0
    end: int = 0
    close_line: Optional[int] = None
    entries: list[_UrlEntry] = field(default_factory=list)
    url_imports: list[str] = field(default_factory=list)
    import_line: int = 0
    imports_include: bool = False
    uses_include: bool = False
    approximate: bool = False

    @property
    def found(self) -> bool:
        return self.line > 0


@dataclass
class _Registration:
    file: str
    models: list[str]
    admin_class: str
    start: int
    end: int
    via: str                # "call" | "decorator"


@dataclass
class _App:
    label: str              # dotted path relative to the project root, e.g. "myapp"
    dir: str                # workspace-relative directory
    installed: bool
    files: list[str] = field(default_factory=list)

    def named(self, *basenames: str) -> list[str]:
        return [f for f in self.files if posixpath.basename(f) in basenames
                and posixpath.dirname(f) == self.dir]

    def in_dir(self, sub: str) -> list[str]:
        prefix = posixpath.join(self.dir, sub) + "/"
        return [f for f in self.files if f.startswith(prefix)]


@dataclass
class _Model:
    app: _App
    file: str
    d: _Def
    entity: str = ""

    @property
    def abstract(self) -> bool:
        meta = self.d.child("Meta")
        return meta is not None and "True" in meta.attrs.get("abstract", [])

    @property
    def proxy(self) -> bool:
        meta = self.d.child("Meta")
        return meta is not None and "True" in meta.attrs.get("proxy", [])


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

def _find_roots(index: SourceIndex) -> list[str]:
    """Directories holding a Django ``manage.py``, sorted."""
    roots = []
    for rel in index.files:
        if posixpath.basename(rel) != "manage.py":
            continue
        parts = rel.split("/")
        if any(p in _TEST_DIRS for p in parts[:-1]):
            continue
        if "django" in (index.text(rel) or "").lower():
            roots.append(posixpath.dirname(rel))
    return sorted(set(roots))


def extract(index: SourceIndex, class_names: list[str]) -> list[ScaffoldMap]:
    """One :class:`ScaffoldMap` per Django project root in ``index``.

    Read-only and deterministic; a file that does not parse is read with a
    line scanner (it is already reported through ``index.syntax_errors``).
    """
    roots = _find_roots(index)
    used_ids: dict[str, str] = {}      # anchor id -> file, shared so ids stay unique across roots
    maps: list[ScaffoldMap] = []
    for root in roots:
        nested = [other for other in roots if other != root and _is_under(other, root)]
        files = [f for f in index.under(root) if not any(_is_under(f, n) for n in nested)]
        family_id = FAMILY if len(roots) == 1 else f"{FAMILY}@{re.sub(r'[^0-9A-Za-z]+', '_', root) or 'root'}"
        try:
            maps.append(_ProjectMapper(index, root, files, class_names, family_id, used_ids).build())
        except Exception:  # one unexpected layout must not hide the other roots
            logger.debug("Scaffold map: Django root %r not mapped", root, exc_info=True)
    if len(maps) > 1:
        # One domain model, several projects: a class modelled in another root is not missing here.
        modelled = {name for m in maps for name, e in m.entities.items() if "model class" not in e.missing}
        for m in maps:
            for name in [n for n, e in m.entities.items()
                         if not e.anchors and "model class" in e.missing and n in modelled]:
                del m.entities[name]
    return maps


# ---------------------------------------------------------------------------
# One project root
# ---------------------------------------------------------------------------

class _ProjectMapper:
    """Collects one Django project's facts and renders its ScaffoldMap."""

    def __init__(self, index: SourceIndex, root: str, files: list[str], class_names: list[str],
                 family_id: str, used_ids: dict[str, str]) -> None:
        self.index = index
        self.root = root
        self.files = files
        self.file_set = set(files)
        self.class_names = [n for n in class_names if isinstance(n, str) and n.strip()]
        self.fid = family_id
        self.used_ids = used_ids
        self.canon: dict[str, str] = {n.lower(): n for n in self.class_names}
        self._defs_cache: dict[str, tuple[list[_Def], bool]] = {}
        self.settings: Optional[_Settings] = None
        self.project_pkg = ""
        self.project_urls: Optional[_UrlConf] = None
        self.apps: list[_App] = []
        self.models: list[_Model] = []
        self.enums: list[tuple[_App, str, _Def]] = []
        self.registrations: list[_Registration] = []
        self.unregistrations: list[_Registration] = []                     # admin.site.unregister(X)
        self.admin_classes: list[tuple[str, _Def, Optional[str]]] = []     # (file, class, entity)
        self.admin_actions: list[tuple[str, _Def, Optional[str]]] = []
        self.views: list[tuple[_App, str, _Def, Optional[str]]] = []
        self.forms: list[tuple[str, _Def, Optional[str]]] = []
        self.serializers: list[tuple[str, _Def, Optional[str]]] = []
        self.urlconfs: list[_UrlConf] = []
        self.templates: list[tuple[str, Optional[str]]] = []
        self.migrations: dict[str, list[str]] = {}
        self.requirements = _join(root, "requirements.txt") if _join(root, "requirements.txt") in self.file_set else ""
        self.warnings: list[str] = []
        self.extra_warnings = 0
        self.entities: dict[str, EntityMap] = {}
        self.app_anchors: list[Anchor] = []

    # -- ids, quotes, entities -----------------------------------------------

    def _id(self, *parts: str, file: str = "") -> str:
        """A workspace-unique id; a clash with another file's anchor is told apart by this file's stem."""
        base = ":".join([self.fid, *[p for p in parts if p]])
        if base in self.used_ids and file and self.used_ids[base] != file:
            base = f"{base}:{posixpath.splitext(posixpath.basename(file))[0]}"
        anchor_id, n = base, 2
        while anchor_id in self.used_ids:
            anchor_id = f"{base}:{n}"
            n += 1
        self.used_ids[anchor_id] = file
        return anchor_id

    def _quote(self, rel: str, *preferred: int, span: Optional[tuple[int, int]] = None) -> str:
        for line_no in preferred:
            quote = self.index.unique_quote(rel, line_no)
            if quote:
                return quote
        return self.index.first_unique_quote(rel, *span) if span else ""

    def _lookup(self, name: str) -> Optional[str]:
        return self.canon.get(name.lower()) if name else None

    def _from_token(self, token: str) -> Optional[str]:
        found = self._lookup(token)
        if found is None and token.lower().endswith("ies"):
            found = self._lookup(token[:-3] + "y")
        if found is None and token.lower().endswith("es"):
            found = self._lookup(token[:-2])
        if found is None and token.lower().endswith("s"):
            found = self._lookup(token[:-1])
        return found

    def _entity_of_name(self, name: str) -> Optional[str]:
        """The entity a name is about: ``create_book``, ``book_list`` -> Book.

        Runs of adjacent words are tried longest first, so ``order_item_list``
        and ``create_orderItem`` are OrderItem's, not Order's.
        """
        tokens = [t for t in re.split(r"[_\-.]+|(?<=[a-z0-9])(?=[A-Z])", name) if t][:8]
        for size in range(len(tokens), 0, -1):
            for i in range(len(tokens) - size + 1):
                found = self._from_token("".join(tokens[i:i + size]))
                if found:
                    return found
        return self._lookup(name)

    def _entity_of_class(self, d: _Def, suffix_re: Optional[re.Pattern] = None) -> Optional[str]:
        """A form/serializer/view class's entity: ``Meta.model`` / ``model`` / ``queryset``, else its name."""
        meta = d.child("Meta")
        for tokens in ([meta.attrs.get("model", [])] if meta else []) + [
                d.attrs.get("model", []), d.attrs.get("queryset", [])]:
            for token in tokens:
                found = self._lookup(token)
                if found:
                    return found
        stem = suffix_re.sub("", d.name) if suffix_re else d.name
        return self._lookup(stem) if stem else None

    def _defs(self, rel: str) -> tuple[list[_Def], bool]:
        """(top-level definitions, approximate?) of a Python file; [] when unreadable."""
        if rel not in self._defs_cache:
            tree = self._tree(rel)
            if isinstance(tree, ast.Module):
                defs = [d for d in (_def_from_ast(st) for st in tree.body) if d is not None]
                self._defs_cache[rel] = (defs, False)
            elif self.index.text(rel) is not None:
                self._defs_cache[rel] = (_LineScanner(_Text(self.index.lines(rel))).defs(), True)
            else:
                self._defs_cache[rel] = ([], False)
        return self._defs_cache[rel]

    def _broken(self, rel: str) -> bool:
        return rel in self.index.syntax_errors

    def _tree(self, rel: str) -> Optional[ast.Module]:
        """The parsed module, or None (a syntax error, or input too deep for the parser)."""
        try:
            tree = self.index.python_ast(rel)
        except Exception:  # RecursionError/MemoryError on pathological input: read it as text
            return None
        return tree if isinstance(tree, ast.Module) else None

    # -- project files -------------------------------------------------------

    def _module_file(self, dotted: str) -> Optional[str]:
        """``a.b`` -> ``<root>/a/b.py`` or ``<root>/a/b/__init__.py`` when it exists."""
        base = _join(self.root, *dotted.split("."))
        for candidate in (base + ".py", base + "/__init__.py"):
            if candidate in self.file_set:
                return candidate
        return None

    def _exists(self, rel: str) -> bool:
        """Whether ``rel`` is on disk, mapped or not: the index leaves files out past its caps.

        Checked before the map says a file does not exist.
        """
        if rel in self.file_set:
            return True
        exists = getattr(self.index, "exists", None)
        if callable(exists):
            return bool(exists(rel))
        return os.path.isfile(self.index.abspath(rel))

    def _module_exists(self, dotted: str) -> bool:
        base = _join(self.root, *dotted.split("."))
        return self._module_file(dotted) is not None or any(
            self._exists(candidate) for candidate in (base + ".py", base + "/__init__.py"))

    def _locate_settings(self) -> Optional[str]:
        """The settings module ``manage.py`` names (else a conventional one) whose modules set INSTALLED_APPS."""
        manage = self.index.text(_join(self.root, "manage.py")) or ""
        m = _SETTINGS_ENV_RE.search(manage)
        candidates: list[str] = []
        if m:
            found = self._module_file(m.group(1))
            if found:
                candidates.append(found)
                if found.endswith("/__init__.py"):
                    package = posixpath.dirname(found)
                    siblings = sorted(f for f in self.files if posixpath.dirname(f) == package and f.endswith(".py"))
                    candidates += sorted(siblings, key=lambda f: (posixpath.basename(f) != "base.py", f))
        candidates += sorted(f for f in self.files if f.endswith("/settings.py")
                             and posixpath.dirname(posixpath.dirname(f)) == self.root)
        for rel in candidates:
            if any("INSTALLED_APPS" in (self.index.text(f) or "") for f in self._settings_chain(rel)):
                return rel
        return candidates[0] if candidates else None

    def _star_import(self, rel: str, dots: str, module: str) -> Optional[str]:
        """The project file ``from <dots><module> import *`` in ``rel`` loads, or None."""
        if not dots:
            return self._module_file(module) if module else None
        base = posixpath.dirname(rel)
        for _ in range(len(dots) - 1):
            base = posixpath.dirname(base)
        target = _join(base, *module.split(".")) if module else base
        for candidate in (target + ".py", target + "/__init__.py"):
            if candidate in self.file_set:
                return candidate
        return None

    def _settings_chain(self, rel: str, seen: Optional[set[str]] = None) -> list[str]:
        """``rel`` preceded by the settings modules it star-imports (``from .base import *``), in load order."""
        seen = set() if seen is None else seen
        if rel in seen or len(seen) > 8:
            return []
        seen.add(rel)
        chain: list[str] = []
        for m in _STAR_IMPORT_RE.finditer(self.index.text(rel) or ""):
            target = self._star_import(rel, m.group("dots"), m.group("module"))
            if target:
                chain += self._settings_chain(target, seen)
        return chain + [rel]

    def _list_operand(self, rel: str, value: Optional[ast.AST], known: dict[str, _ListSetting]) -> _ListSetting:
        """The literals of a list setting's value: a list/tuple, a known list name, or a ``+`` of those."""
        out = _ListSetting(name="", start=0, end=0)
        if isinstance(value, (ast.List, ast.Tuple)):
            close = value.end_lineno or value.lineno
            out.literals.append(_Literal(
                file=rel, open_line=value.lineno, close_line=close, close_col=max(0, (value.end_col_offset or 1) - 1),
                items=[(_string(e) or "", e.lineno, e.end_lineno or e.lineno, e.col_offset) for e in value.elts]))
        elif isinstance(value, ast.Name) and value.id in known:
            found = known[value.id]
            out.literals = [lit if lit.via else dataclasses.replace(lit, via=value.id) for lit in found.literals]
            out.extra, out.resolved = list(found.extra), found.resolved
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            left = self._list_operand(rel, value.left, known)
            right = self._list_operand(rel, value.right, known)
            out.literals = left.literals + right.literals
            out.extra = left.extra + right.extra
            out.resolved = left.resolved and right.resolved
        else:
            out.resolved = False
        return out

    @staticmethod
    def _list_mutation(statement: ast.stmt, name: str) -> Optional[tuple[str, Optional[ast.AST]]]:
        """``NAME += value`` -> ('add', value); ``NAME.append/extend/insert(...)`` -> ('call', the call)."""
        if isinstance(statement, ast.AugAssign) and isinstance(statement.op, ast.Add) and _targets(statement) == [name]:
            return "add", statement.value
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and _dotted(call.func.value) == name and call.func.attr in ("append", "extend", "insert")):
            return "call", call
        return None

    @staticmethod
    def _strings(node: Optional[ast.AST]) -> list[str]:
        return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)] \
            if node is not None else []

    def _apply_change(self, rel: str, setting: _ListSetting, change: tuple[str, ast.AST],
                      known: dict[str, _ListSetting], keep_literals: bool) -> None:
        kind, node = change
        if kind == "add":
            added = self._list_operand(rel, node, known)
            if keep_literals:
                setting.literals += added.literals
            else:
                setting.extra += [item[0] for item in added.items if item[0]]
            setting.extra += added.extra
            setting.resolved = setting.resolved and added.resolved
        elif isinstance(node, ast.Call):
            args = node.args[1:] if isinstance(node.func, ast.Attribute) and node.func.attr == "insert" else node.args
            setting.extra += [value for arg in args for value in self._strings(arg)]

    def _read_list_settings(self, rel: str, tree: ast.Module, settings: _Settings,
                            known: dict[str, _ListSetting], mode: str) -> None:
        """INSTALLED_APPS/MIDDLEWARE (and the module-level lists they name) of one settings module.

        ``mode`` is "anchor" for the module the anchors point at, "names" for a
        module it star-imports (only its lists count, for ``A + B``) and
        "extend" for a module that star-imports it (only what it adds counts).
        """
        for statement in tree.body:
            targets = _targets(statement)
            for name in ("INSTALLED_APPS", "MIDDLEWARE") if mode != "names" else ():
                setting = settings.lists.get(name)
                change = self._list_mutation(statement, name)
                if change is not None:
                    if setting is None and mode == "anchor":   # `+=` on a list this map could not read
                        setting = settings.lists[name] = _ListSetting(
                            name=name, start=statement.lineno, end=statement.end_lineno or statement.lineno,
                            resolved=False)
                    if setting is not None:
                        self._apply_change(rel, setting, change, known, keep_literals=mode == "anchor")
                elif mode == "anchor" and name in targets and isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    value = self._list_operand(rel, statement.value, known)
                    if setting is None:     # a later `NAME = NAME + [...]` keeps the first statement's span
                        setting = settings.lists[name] = _ListSetting(
                            name=name, start=statement.lineno, end=statement.end_lineno or statement.lineno)
                    setting.literals, setting.extra, setting.resolved = value.literals, value.extra, value.resolved
                elif setting is not None and isinstance(statement, (ast.If, ast.Try, ast.With, ast.For)):
                    # Changed inside an if/try block (`if DEBUG: INSTALLED_APPS += [...]`): count what it adds.
                    for sub in ast.walk(statement):
                        nested = self._list_mutation(sub, name) if isinstance(sub, ast.stmt) else None
                        if nested is not None:
                            setting.extra += self._strings(nested[1])
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                for target in targets:
                    if mode != "names" and target in settings.lists:
                        known[target] = settings.lists[target]
                        continue
                    value = self._list_operand(rel, statement.value, known)
                    if value.literals or value.extra:
                        known[target] = value
            elif mode == "names" and targets and targets[0] in known and self._list_mutation(statement, targets[0]):
                previous, added = known[targets[0]], self._list_operand(rel, statement.value, known)
                known[targets[0]] = _ListSetting(name="", start=0, end=0, literals=previous.literals + added.literals,
                                                 extra=previous.extra + added.extra,
                                                 resolved=previous.resolved and added.resolved)
            if mode == "anchor":
                for name in ("DATABASES", "TEMPLATES"):
                    if name in targets and name not in settings.blocks:
                        settings.blocks[name] = (statement.lineno, statement.end_lineno or statement.lineno)

    def _read_settings(self, named: str) -> _Settings:
        """The settings ``named`` loads: anchors go to the module that assigns INSTALLED_APPS last.

        With ``from .base import *`` + ``INSTALLED_APPS += [...]`` (a settings
        package), that is ``base.py``; what ``named`` adds still counts as installed.
        """
        chain = self._settings_chain(named)
        primary = next((f for f in reversed(chain) if _PLAIN_INSTALLED_RE.search(self.index.text(f) or "")), named)
        settings = _Settings(rel=primary, named=named if named != primary else "")
        for rel in reversed(chain):         # the most derived module wins
            text = self.index.text(rel) or ""
            m = _ROOT_URLCONF_RE.search(text)
            if m and not settings.root_urlconf:
                settings.root_urlconf = m.group(1)
            engine = _ENGINE_RE.search(text)
            if engine and not settings.engine:
                settings.engine = engine.group(1)
        tree = self._tree(primary)
        if not isinstance(tree, ast.Module):
            self._read_settings_text(primary, settings)
            return settings
        known: dict[str, _ListSetting] = {}
        position = chain.index(primary) if primary in chain else len(chain)
        for i, rel in enumerate(chain):
            module = tree if rel == primary else self._tree(rel)
            mode = "names" if i < position else "anchor" if i == position else "extend"
            if isinstance(module, ast.Module):
                self._read_list_settings(rel, module, settings, known, mode)
            elif mode == "extend" and "INSTALLED_APPS" in settings.lists:
                settings.lists["INSTALLED_APPS"].resolved = False      # what it adds cannot be read
        if tree.body:
            last = tree.body[-1]
            settings.last_statement = (node_start(last), last.end_lineno or last.lineno)
        return settings

    def _read_settings_text(self, rel: str, settings: _Settings) -> None:
        """Unparseable settings: the same facts from the text."""
        t = _Text(self.index.lines(rel))
        for name in ("INSTALLED_APPS", "MIDDLEWARE", "DATABASES", "TEMPLATES"):
            m = re.search(rf"^{name}\s*=\s*([\[({{])", t.joined, re.M)
            if not m:
                continue
            close = _match_bracket(t.joined, m.start(1))
            start = t.line_of(m.start())
            end = t.line_of(close) if close is not None else t.last_code_line()
            if name in ("DATABASES", "TEMPLATES"):
                settings.blocks[name] = (start, end)
                continue
            items = []
            for n in range(start, end + 1):
                line = t.lines[n - 1].split("#", 1)[0]
                offset = line.find(m.group(1)) + 1 if n == start else 0
                for sm in re.finditer(r"""(['"])([^'"]+)\1""", line[offset:]):
                    items.append((sm.group(2), n, n, offset + sm.start()))
            after = t.lines[end - 1][close - t.offset_of(end) + 1:] if close is not None else ""
            settings.lists[name] = _ListSetting(name=name, start=start, end=end, literals=[_Literal(
                file=rel, open_line=start, close_line=end if close is not None else None,
                close_col=close - t.offset_of(end) if close is not None else 0, items=items)],
                resolved="+" not in after.split("#", 1)[0] and not re.search(
                    rf"^{name}[ \t]*\+=|^[ \t]*{name}\.(?:append|extend|insert)\(", t.joined, re.M))
        statements = [n for n in range(1, len(t.lines) + 1) if re.match(r"[A-Za-z_]", t.lines[n - 1])]
        if statements:
            settings.last_statement = (statements[-1], t.last_code_line())

    # -- apps ----------------------------------------------------------------

    def _discover_apps(self) -> None:
        installed = self.settings.installed() if self.settings else []
        # INSTALLED_APPS not read completely (a call, an env lookup): never claim an app is missing from it.
        unknown = not (self.settings and self.settings.apps_known)
        dirs: dict[str, bool] = {}
        for entry in installed:
            if entry.startswith("django."):
                continue
            parts = entry.split(".")
            for k in range(len(parts), 0, -1):
                candidate = _join(self.root, *parts[:k])
                if any(_join(candidate, marker) in self.file_set for marker in ("__init__.py", "apps.py", "models.py")):
                    dirs[candidate] = True
                    break
        for rel in self.files:
            if posixpath.basename(rel) not in ("apps.py", "models.py") and not rel.endswith("/models/__init__.py"):
                continue
            if "django" not in (self.index.text(rel) or ""):
                continue
            directory = posixpath.dirname(rel)
            if rel.endswith("/models/__init__.py"):
                directory = posixpath.dirname(directory)
            depth = len(posixpath.relpath(directory, self.root or ".").split("/"))
            parts = directory.split("/")
            if depth > 2 or directory == self.root or any(p in _TEST_DIRS or p == "migrations" for p in parts):
                continue
            dirs.setdefault(directory, False)
        if self.project_pkg and _join(self.project_pkg, "views.py") in self.file_set:
            dirs.setdefault(self.project_pkg, False)
        for directory in sorted(dirs):
            label = posixpath.relpath(directory, self.root or ".").replace("/", ".")
            installed_here = unknown or dirs[directory] or (directory == self.project_pkg
                                                            and _join(directory, "models.py") not in self.file_set)
            app = _App(label=label, dir=directory, installed=installed_here)
            nested = [d for d in dirs if d != directory and _is_under(d, directory)]
            app.files = [f for f in self.files if f.startswith(directory + "/")
                         and not any(_is_under(f, d) for d in nested)]
            self.apps.append(app)

    def _app_py(self, app: _App) -> list[str]:
        """The app's own Python sources (no migrations, no tests)."""
        out = []
        for rel in app.files:
            parts = posixpath.relpath(rel, app.dir).split("/")
            if not rel.endswith(".py") or "migrations" in parts[:-1] or any(p in _TEST_DIRS for p in parts[:-1]):
                continue
            if parts[-1] == "tests.py" or parts[-1].startswith("test_"):
                continue
            out.append(rel)
        return out

    def _read_apps(self) -> None:
        model_defs: list[tuple[_App, str, _Def]] = []
        for app in self.apps:
            models_files = app.named("models.py") + [f for f in app.in_dir("models") if f.endswith(".py")]
            for rel in models_files:
                for d in self._defs(rel)[0]:
                    if d.kind == "class":
                        model_defs.append((app, rel, d))
            self.migrations[app.label] = sorted(f for f in app.in_dir("migrations")
                                                if _MIGRATION_RE.match(posixpath.basename(f)))
            initial = _join(app.dir, "migrations", "0001_initial.py")
            if not self.migrations[app.label] and self._exists(initial):    # left out of the index
                self.migrations[app.label] = [initial]
        # Models: a Django model base, or (transitively) another model of this project.
        names: set[str] = set()
        for _ in range(5):
            before = len(names)
            for _, _, d in model_defs:
                if any(_last(b) in _MODEL_BASES or b in names for b in d.bases):
                    names.add(d.name)
            if len(names) == before:
                break
        for app, rel, d in model_defs:
            if d.name in names:
                entity = self._lookup(d.name) or d.name
                self.canon.setdefault(d.name.lower(), entity)
                self.models.append(_Model(app=app, file=rel, d=d, entity=entity))
            elif any(_last(b) in _CHOICES_BASES for b in d.bases):
                self.enums.append((app, rel, d))
        for app in self.apps:
            self._read_app_code(app)
        if self.project_urls is not None:
            self.urlconfs.append(self.project_urls)
        for app in self.apps:
            for rel in app.named("urls.py"):
                if self.project_urls is None or rel != self.project_urls.rel:
                    self.urlconfs.append(self._read_urlconf(rel, app))
        template_dirs = [_join(a.dir, "templates") for a in self.apps] + [_join(self.root, "templates")]
        for rel in self.files:
            if rel.endswith(".html") and any(rel.startswith(d + "/") for d in template_dirs):
                stem = posixpath.splitext(posixpath.basename(rel))[0]
                self.templates.append((rel, self._entity_of_name(stem)))

    def _read_app_code(self, app: _App) -> None:
        model_names = {m.d.name for m in self.models}
        for rel in self._app_py(app):
            base = posixpath.basename(rel)
            parent = posixpath.basename(posixpath.dirname(rel))
            is_admin = base == "admin.py" or parent == "admin"
            text = self.index.text(rel) or ""
            is_views = (base in _VIEW_FILES or parent in _VIEW_DIRS) and ("django" in text or "rest_framework" in text)
            defs, _ = self._defs(rel)
            if is_admin:
                self._read_admin(rel, defs, model_names)
            for d in defs:
                last_bases = [_last(b) for b in d.bases]
                if d.kind == "class" and "ModelForm" in last_bases:
                    self.forms.append((rel, d, self._entity_of_class(d, re.compile(r"Form$"))))
                elif d.kind == "class" and any(b.endswith("ModelSerializer") for b in last_bases):
                    self.serializers.append((rel, d, self._entity_of_class(d, re.compile(r"Serializer$"))))
                elif is_views and d.kind == "def":
                    self.views.append((app, rel, d, self._entity_of_name(d.name)))
                elif is_views and d.kind == "class" and (
                        any(b.endswith(("View", "ViewSet", "APIView", "Mixin")) for b in last_bases)
                        or _VIEW_SUFFIX_RE.search(d.name)):
                    self.views.append((app, rel, d, self._entity_of_class(d, _VIEW_SUFFIX_RE)))

    def _read_admin(self, rel: str, defs: list[_Def], model_names: set[str]) -> None:
        known = model_names | set(self.canon.values())
        tree = self._tree(rel)
        if isinstance(tree, ast.Module):
            for statement in tree.body:
                call = statement.value if isinstance(statement, ast.Expr) else None
                func = _dotted(call.func) if isinstance(call, ast.Call) else ""
                if not func.endswith(("site.register", "site.unregister")):
                    continue
                first = call.args[0] if call.args else _keyword(call, "model_or_iterable")
                admin_node = call.args[1] if len(call.args) > 1 else _keyword(call, "admin_class")
                items = first.elts if isinstance(first, (ast.List, ast.Tuple)) else [first]
                models = [name for name in (_last(_dotted(e)) for e in items) if name]
                if func.endswith("site.unregister"):
                    self.unregistrations.append(_Registration(
                        file=rel, models=models, admin_class="", start=statement.lineno,
                        end=statement.end_lineno or statement.lineno, via="unregister"))
                    continue
                self.registrations.append(_Registration(
                    file=rel, models=models, admin_class=_dotted(admin_node), start=statement.lineno,
                    end=statement.end_lineno or statement.lineno, via="call"))
        else:
            t = _Text(self.index.lines(rel))
            for n, line in enumerate(t.lines, 1):
                m = _REGISTER_LINE_RE.match(line)
                if not m:
                    continue
                open_pos = t.offset_of(n, m.end() - 1)
                close = _match_bracket(t.joined, open_pos)
                inside = t.joined[open_pos + 1:close] if close is not None else line[m.end():]
                args = _split_args(inside)
                positional = [a for a in args if not re.match(r"\w+\s*=", a)]
                admin_kw = next((a.split("=", 1)[1].strip() for a in args if a.startswith("admin_class")), "")
                if m.group("un"):
                    self.unregistrations.append(_Registration(
                        file=rel, models=_text_tokens(positional[0]) if positional else [], admin_class="",
                        start=n, end=t.line_of(close) if close is not None else n, via="unregister"))
                    continue
                self.registrations.append(_Registration(
                    file=rel, models=_text_tokens(positional[0]) if positional else [],
                    admin_class=positional[1] if len(positional) > 1 else admin_kw,
                    start=n, end=t.line_of(close) if close is not None else n, via="call"))
        admin_of = {r.admin_class: m for r in self.registrations if r.file == rel for m in r.models
                    if m in known and r.admin_class}
        action_owner: dict[str, Optional[str]] = {}
        for d in defs:
            if d.kind != "class" or not any(_last(b).endswith("ModelAdmin") for b in d.bases):
                continue
            decorated = [t for name, tokens in d.decorators if _last(name) == "register"
                         for t in tokens if t in known]
            if decorated:
                self.registrations.append(_Registration(file=rel, models=decorated, admin_class=d.name,
                                                        start=d.start, end=d.line, via="decorator"))
            model = decorated[0] if decorated else admin_of.get(d.name)
            entity = (self._lookup(model) or model) if model else self._lookup(re.sub(r"Admin$", "", d.name))
            self.admin_classes.append((rel, d, entity))
            for action in d.attrs.get("actions", []):
                action_owner.setdefault(action, entity)
        for d in defs:
            if d.kind != "def":
                continue
            is_action = any(_last(name) == "action" for name, _ in d.decorators) or d.name in action_owner
            if is_action:
                self.admin_actions.append((rel, d, action_owner.get(d.name) or self._entity_of_name(d.name)))

    def _read_urlconf(self, rel: str, app: Optional[_App]) -> _UrlConf:
        conf = _UrlConf(rel=rel, app=app)
        tree = self._tree(rel)
        if isinstance(tree, ast.Module):
            self._urlconf_from_ast(conf, tree)
        elif self.index.text(rel) is not None:
            conf.approximate = True
            self._urlconf_from_text(conf, _Text(self.index.lines(rel)))
        return conf

    def _urlconf_from_ast(self, conf: _UrlConf, tree: ast.Module) -> None:
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                names = [a.asname or a.name for a in statement.names]
                if "include" in names:
                    conf.imports_include = True
                if statement.module in ("django.urls", "django.conf.urls") and not conf.import_line:
                    conf.import_line, conf.url_imports = statement.lineno, names
            elif isinstance(statement, ast.Import) and any((a.asname or a.name) == "include"
                                                           for a in statement.names):
                conf.imports_include = True
            elif "urlpatterns" in _targets(statement):
                value = statement.value
                lists = []          # urlpatterns = [...], [...] + router.urls, [...] + static(...)
                while isinstance(value, ast.BinOp):
                    if isinstance(value.right, (ast.List, ast.Tuple)):
                        lists.insert(0, value.right)
                    value = value.left
                if isinstance(value, (ast.List, ast.Tuple)):
                    lists.insert(0, value)
                if not lists:
                    continue
                if not conf.found and not isinstance(statement, ast.AugAssign):
                    conf.start = conf.line = statement.lineno
                    conf.end = statement.end_lineno or statement.lineno
                    conf.close_line = lists[-1].end_lineno if lists[-1].end_lineno == conf.end else None
                for items in lists:
                    conf.entries += [e for e in (self._url_entry_ast(elt) for elt in items.elts) if e]
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                func = _dotted(call.func)
                if func.endswith(".register") and not func.endswith("site.register") and len(call.args) >= 2:
                    conf.entries.append(_UrlEntry(func="register", route=_string(call.args[0]) or "",
                                                  view=_dotted(call.args[1]), name="", include="",
                                                  start=statement.lineno, end=statement.end_lineno or statement.lineno))
        conf.uses_include = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "include"
                                for n in ast.walk(tree))

    @staticmethod
    def _url_entry_ast(node: ast.AST) -> Optional[_UrlEntry]:
        if not isinstance(node, ast.Call) or _last(_dotted(node.func)) not in ("path", "re_path", "url"):
            return None
        route = _string(node.args[0]) if node.args else _string(_keyword(node, "route"))
        target = node.args[1] if len(node.args) > 1 else _keyword(node, "view")
        view, include = "", ""
        if isinstance(target, ast.Call) and _last(_dotted(target.func)) == "include":
            arg = target.args[0] if target.args else None
            if isinstance(arg, ast.Tuple) and arg.elts:
                arg = arg.elts[0]
            include = _string(arg) or "?"
        else:
            view = _wrapped_view(target)
        return _UrlEntry(func=_last(_dotted(node.func)), route=route or "", view=view,
                         name=_string(_keyword(node, "name")) or "", include=include,
                         start=node.lineno, end=node.end_lineno or node.lineno,
                         refs=[] if include else _names_in(target))

    def _urlconf_from_text(self, conf: _UrlConf, t: _Text) -> None:
        for n, line in enumerate(t.lines, 1):
            m = _DJANGO_URLS_IMPORT_RE.match(line)
            if m and not conf.import_line:
                conf.import_line = n
                conf.url_imports = [x.strip() for x in m.group("names").strip("() ").split(",") if x.strip()]
            if re.match(r"^\s*(?:from\s+\S+\s+)?import\b.*\binclude\b", line):
                conf.imports_include = True
        m = _URLPATTERNS_RE.search(t.joined)
        if m:
            close = _match_bracket(t.joined, m.start(1))
            conf.start = conf.line = t.line_of(m.start())
            conf.end = t.line_of(close) if close is not None else t.last_code_line()
            conf.close_line = conf.end if close is not None else None
            limit = close if close is not None else len(t.joined)
            for pm in _PATH_CALL_RE.finditer(t.joined, m.end(), limit):
                open_pos = pm.end() - 1
                call_close = _match_bracket(t.joined, open_pos)
                inside = t.joined[open_pos + 1:call_close if call_close is not None else limit]
                conf.entries.append(self._url_entry_text(
                    pm.group("func"), inside, t.line_of(pm.start()),
                    t.line_of(call_close) if call_close is not None else conf.end))
            conf.uses_include = "include(" in t.joined[m.end():limit]
        for rm in _ROUTER_REGISTER_RE.finditer(t.joined):
            if rm.group("recv") in ("admin", "site"):
                continue
            close = _match_bracket(t.joined, rm.end() - 1)
            args = _split_args(t.joined[rm.end():close] if close is not None else "")
            if len(args) >= 2:
                sm = _STRING_ARG_RE.match(args[0])
                conf.entries.append(_UrlEntry(func="register", route=sm.group("s") if sm else "", view=args[1],
                                              name="", include="", start=t.line_of(rm.start()),
                                              end=t.line_of(close) if close is not None else t.line_of(rm.start())))

    @staticmethod
    def _url_entry_text(func: str, inside: str, start: int, end: int) -> _UrlEntry:
        args = _split_args(inside)
        route_match = _STRING_ARG_RE.match(args[0]) if args else None
        target = args[1] if len(args) > 1 else ""
        view, include = "", ""
        im = _INCLUDE_ARG_RE.match(target)
        refs: list[str] = []
        if im:
            include = im.group("inc") or "?"
        else:
            try:        # the argument usually parses even when the file does not
                node: Optional[ast.AST] = ast.parse(target.strip(), mode="eval").body
            except Exception:  # whatever the argument text is, the regex below still reads it
                node = None
            if node is not None:
                view, refs = _wrapped_view(node), _names_in(node)
            else:
                vm = _VIEW_ARG_RE.match(target)
                view = re.sub(r"\.as_view$", "", vm.group("view")) if vm else ""
                refs = sorted(set(re.findall(r"[A-Za-z_]\w*", target)))
        nm = _NAME_KW_RE.search(inside)
        return _UrlEntry(func=func, route=route_match.group("s") if route_match else "", view=view,
                         name=nm.group("name") if nm else "", include=include, start=start, end=max(start, end),
                         refs=refs)

    # -- build ---------------------------------------------------------------

    def build(self) -> ScaffoldMap:
        settings_rel = self._locate_settings()
        if settings_rel:
            self.settings = self._read_settings(settings_rel)
            self.project_pkg = posixpath.dirname(settings_rel)
            if settings_rel.endswith("/__init__.py") or posixpath.basename(self.project_pkg) == "settings":
                self.project_pkg = posixpath.dirname(self.project_pkg)
        urls_rel = None
        if self.settings and self.settings.root_urlconf:
            urls_rel = self._module_file(self.settings.root_urlconf)
        if urls_rel is None and self.project_pkg and _join(self.project_pkg, "urls.py") in self.file_set:
            urls_rel = _join(self.project_pkg, "urls.py")
        self._discover_apps()
        if urls_rel:
            self.project_urls = self._read_urlconf(urls_rel, None)
        self._read_apps()

        smap = ScaffoldMap(family=FAMILY, root=self.root, title=self._title())
        smap.anchors = self._project_anchors()
        self._entity_anchors()
        smap.anchors += self.app_anchors
        self._missing()
        self._detect_pitfalls()
        if self.extra_warnings:
            self.warnings.append(f"… and {self.extra_warnings} more Django problem(s) not listed here.")
        for name in sorted(self.entities, key=lambda n: (n.lower(), n)):
            entity = self.entities[name]
            entity.anchors = [a for _, _, a in sorted(
                (_ANCHOR_ORDER.get(a.id[len(self.fid) + 1:].split(":", 1)[0], 99), n, a)
                for n, a in enumerate(entity.anchors))]
            entity.summary = self._entity_summary(name)
            smap.entities[name] = entity
        smap.warnings = list(self.warnings)
        smap.summary = self._map_summary()
        smap.recipes = self._recipes(smap)
        return smap

    def _entity(self, name: str) -> EntityMap:
        if name not in self.entities:
            self.entities[name] = EntityMap(name=name)
        return self.entities[name]

    def _add(self, entity: Optional[str], anchor: Anchor) -> None:
        if entity:
            self._entity(entity).anchors.append(anchor)
        else:
            self.app_anchors.append(anchor)

    def _approx_hint(self, rel: str) -> str:
        scanned = self._defs_cache.get(rel, ([], False))[1]
        return (f"{posixpath.basename(rel)} does not parse: this span is approximate, fix the syntax error first"
                if self._broken(rel) or scanned else "")

    # -- insertion points ----------------------------------------------------

    def _module_end(self, rel: str) -> Optional[tuple[int, str]]:
        """Where top-level code is appended: after the last line, else before the last statement."""
        lines = self.index.lines(rel)
        last = next((n for n in range(len(lines), 0, -1) if lines[n - 1].strip()), 0)
        if not last:
            return None
        if self.index.unique_quote(rel, last):
            return last, "AFTER"
        defs, _ = self._defs(rel)
        starts = [d.start for d in defs]
        tree = self._tree(rel)
        if isinstance(tree, ast.Module) and tree.body:
            starts.append(node_start(tree.body[-1]))
        for line_no in sorted(set(starts), reverse=True):
            if self.index.unique_quote(rel, line_no):
                return line_no, "BEFORE"
        return None

    def _insert_anchor(self, entity: Optional[str], what: str, name: str, rel: str,
                       where: Optional[tuple[int, str]], label: str, hint: str, then: str = "") -> None:
        if where is None:
            return
        line_no, relation = where
        line = self.index.lines(rel)[line_no - 1]
        column = " (at column 0)" if relation == "AFTER" and line[:1].isspace() else ""
        self._add(entity, Anchor(
            id=self._id(what, (entity or "").lower(), name, file=rel), file=rel, kind=KIND_INSERT,
            label=f"{label} go {relation.lower()} this line", start_line=line_no, end_line=line_no,
            quote=self.index.unique_quote(rel, line_no),
            hint=f"insert {hint} {relation} this line{column}" + (f"; {then}" if then else ""), entity=entity))

    def _list_end(self, rel: str, setting: _ListSetting) -> Optional[tuple[int, str, str]]:
        """Where a new entry goes in a list setting: (line, AFTER|BEFORE, comma hint), or None.

        Inside the last literal of ``rel`` that can take one: after its last item
        when the closing bracket is not on that item's line, else before the
        closing bracket when the bracket starts its line. Never after a line
        that closes the list (``'myapp']``), which would put the entry outside it.
        """
        lines = self.index.lines(rel)
        comma = " (Python joins adjacent strings silently)"
        for literal in reversed([lit for lit in setting.literals if lit.file == rel]):
            _, first, last, col = literal.items[-1] if literal.items else ("", 0, 0, 0)
            if last > len(lines):
                continue
            needs_comma = bool(last) and not lines[last - 1].split("#", 1)[0].rstrip().endswith(",")
            via = f" (this is {literal.via}, which {setting.name} adds)" if literal.via else ""
            if last and last != literal.close_line and self.index.unique_quote(rel, last):
                return last, "AFTER", via + ("; this line has no trailing comma: add one first" + comma
                                             if needs_comma else "")
            close = literal.close_line
            if (close and close != literal.open_line and close <= len(lines)
                    and not lines[close - 1][:literal.close_col].strip() and self.index.unique_quote(rel, close)):
                return close, "BEFORE", via + ("; the entry above has no trailing comma: add one first" + comma
                                               if last and needs_comma else "")
            # `    'myapp']`: the last entry shares the closing bracket's line; a new one can still precede it.
            if (first and first != literal.open_line and not lines[first - 1][:col].strip()
                    and self.index.unique_quote(rel, first)):
                return first, "BEFORE", via + ("; the last entry shares its line with the closing bracket, so "
                                               "end the new entry with a comma")
        return None

    def _field_insert(self, model: _Model) -> Optional[tuple[int, str, str]]:
        """Where a new field goes in a model body: (line, AFTER|BEFORE, extra hint)."""
        rel, d = model.file, model.d
        members = sorted(c.start for c in d.body)
        fields = [f for f in d.fields if not members or f.end < members[0]] or d.fields
        for f in sorted(fields, key=lambda f: f.end, reverse=True):
            if self.index.unique_quote(rel, f.end):
                return f.end, "AFTER", ""
        first_member = next((c for c in sorted(d.body, key=lambda c: c.start)), None)
        if first_member is not None and self.index.unique_quote(rel, first_member.start):
            return first_member.start, "BEFORE", ""
        if self.index.unique_quote(rel, d.line):
            return d.line, "AFTER", " (below the class docstring, if any)"
        return None

    # -- anchors -------------------------------------------------------------

    def _project_anchors(self) -> list[Anchor]:
        anchors: list[Anchor] = []
        manage = _join(self.root, "manage.py")
        lines = self.index.lines(manage)
        env_line = next((n for n, line in enumerate(lines, 1) if "DJANGO_SETTINGS_MODULE" in line), 1)
        anchors.append(Anchor(
            id=self._id("manage", file=manage), file=manage, kind=KIND_FILE,
            label=f"manage.py: run makemigrations/migrate/runserver/check from {self.root or '.'}/",
            start_line=1, end_line=max(1, len(lines)),
            quote=self._quote(manage, env_line, span=(1, len(lines)))))
        s = self.settings
        if s is not None:
            rel = s.rel
            n_lines = len(self.index.lines(rel))
            anchors.append(Anchor(id=self._id("settings", file=rel), file=rel, kind=KIND_FILE,
                                  label="settings module" + (f" (loaded through {s.named}, which star-imports it)"
                                                             if s.named else ""),
                                  start_line=1, end_line=max(1, n_lines)))
            apps = s.lists.get("INSTALLED_APPS")
            if apps:
                local = [v for v in apps.values() if not v.startswith("django.")]
                builtin = len([v for v in apps.values() if v.startswith("django.")])
                anchors.append(Anchor(
                    id=self._id("installed_apps", file=rel), file=rel, kind=KIND_BLOCK,
                    label=f"INSTALLED_APPS ({_short_list(local)}" + (f" + {builtin} django.contrib" if builtin else "")
                    + ")", start_line=apps.start, end_line=apps.end,
                    quote=self._quote(rel, apps.start, span=(apps.start, apps.end)), hint=self._approx_hint(rel)))
                where = self._list_end(rel, apps)
                if where:
                    anchors.append(Anchor(
                        id=self._id("installed_apps_end", file=rel), file=rel, kind=KIND_INSERT,
                        label=f"new INSTALLED_APPS entries go {where[1].lower()} this line",
                        start_line=where[0], end_line=where[0], quote=self.index.unique_quote(rel, where[0]),
                        hint=f"insert new apps ('rest_framework', 'corsheaders', a new local app) {where[1]} "
                             f"this line{where[2]}"))
            middleware = s.lists.get("MIDDLEWARE")
            if middleware:
                anchors.append(Anchor(
                    id=self._id("middleware", file=rel), file=rel, kind=KIND_BLOCK,
                    label=f"MIDDLEWARE ({len(middleware.items) + len(middleware.extra)} entries)",
                    start_line=middleware.start,
                    end_line=middleware.end, quote=self._quote(rel, middleware.start,
                                                               span=(middleware.start, middleware.end))))
                where = self._list_end(rel, middleware)
                if where:
                    anchors.append(Anchor(
                        id=self._id("middleware_end", file=rel), file=rel, kind=KIND_INSERT,
                        label=f"new MIDDLEWARE entries go {where[1].lower()} this line",
                        start_line=where[0], end_line=where[0], quote=self.index.unique_quote(rel, where[0]),
                        hint=f"insert new middleware {where[1]} this line{where[2]}; order matters: "
                             "CorsMiddleware goes above CommonMiddleware"))
            for name, anchor_name in (("DATABASES", "databases"), ("TEMPLATES", "templates_setting")):
                if name in s.blocks:
                    start, end = s.blocks[name]
                    anchors.append(Anchor(
                        id=self._id(anchor_name, file=rel), file=rel, kind=KIND_BLOCK,
                        label=name + (f" ({s.engine})" if name == "DATABASES" and s.engine else ""),
                        start_line=start, end_line=end, quote=self._quote(rel, start, span=(start, end))))
            if s.last_statement:
                where = None
                lines = self.index.lines(rel)
                last = next((n for n in range(len(lines), 0, -1) if lines[n - 1].strip()), 0)
                if last and self.index.unique_quote(rel, last) and not lines[last - 1][:1].isspace():
                    where = (last, "AFTER")
                elif self.index.unique_quote(rel, s.last_statement[0]):
                    where = (s.last_statement[0], "BEFORE")
                if where:
                    anchors.append(Anchor(
                        id=self._id("settings_end", file=rel), file=rel, kind=KIND_INSERT,
                        label=f"new settings go {where[1].lower()} this line", start_line=where[0],
                        end_line=where[0], quote=self.index.unique_quote(rel, where[0]),
                        hint=f"insert new module-level settings (REST_FRAMEWORK, CORS_ALLOWED_ORIGINS...) "
                             f"{where[1]} this line"))
        if self.project_urls is not None:
            anchors += self._urlconf_anchors(self.project_urls, project=True)
        if self.requirements:
            rel = self.requirements
            text = self.index.text(rel) or ""
            lines = self.index.lines(rel)
            last = next((n for n in range(len(lines), 0, -1) if lines[n - 1].strip()), 0)
            if last and self.index.unique_quote(rel, last):
                newline = text.endswith("\n")
                anchors.append(Anchor(
                    id=self._id("requirements_end", file=rel), file=rel, kind=KIND_INSERT,
                    label="new requirements go after this line", start_line=last, end_line=last,
                    quote=self.index.unique_quote(rel, last),
                    hint="append new packages AFTER this line" + ("" if newline else
                         "; the file has NO trailing newline, so start the appended text with a newline")))
        return anchors

    def _urlconf_anchors(self, conf: _UrlConf, project: bool) -> list[Anchor]:
        """The urlpatterns block, its insertion point and the django.urls import of one URLconf."""
        anchors: list[Anchor] = []
        rel = conf.rel
        app = "" if project else conf.app.label if conf.app else ""
        block_id = "urlpatterns" if project else "app_urlpatterns"
        if conf.import_line and project:
            include_hint = ("" if conf.imports_include else
                            "include is NOT imported: change this line to `from django.urls import include, path` "
                            "before using include()")
            anchors.append(Anchor(
                id=self._id("urls_import", app, file=rel), file=rel, kind=KIND_BLOCK,
                label=f"from django.urls import {', '.join(conf.url_imports)}", start_line=conf.import_line,
                end_line=conf.import_line, quote=self.index.unique_quote(rel, conf.import_line), hint=include_hint))
        if not conf.found:
            return anchors
        routes = [self._entry_label(e) for e in conf.entries if e.func != "register"]
        anchors.append(Anchor(
            id=self._id(block_id, app, file=rel), file=rel, kind=KIND_BLOCK,
            label=("project urlpatterns: " if project else f"{app} urlpatterns: ")
            + (_short_list(routes, 4) if routes else "empty"),
            start_line=conf.start, end_line=conf.end, quote=self._quote(rel, conf.line, span=(conf.start, conf.end)),
            hint="; ".join(filter(None, [self._approx_hint(rel), "the list is never closed: add its `]` first"
                                         if conf.approximate and conf.close_line is None else ""]))))
        where = None
        if conf.close_line and conf.close_line != conf.line and self.index.unique_quote(rel, conf.close_line):
            where = (conf.close_line, "BEFORE")
        else:
            in_list = [e for e in conf.entries if e.func != "register" and conf.start <= e.start <= conf.end]
            if in_list and self.index.unique_quote(rel, in_list[-1].end) and in_list[-1].end != conf.line:
                where = (in_list[-1].end, "AFTER")
        if where:
            anchors.append(Anchor(
                id=self._id(block_id + "_end", app, file=rel), file=rel, kind=KIND_INSERT,
                label=f"new path(...) entries go {where[1].lower()} this line", start_line=where[0],
                end_line=where[0], quote=self.index.unique_quote(rel, where[0]),
                hint=f"insert new path(...) entries {where[1]} this line"
                + ("" if where[1] == "BEFORE" else "; end the previous entry with a comma")))
        return anchors

    @staticmethod
    def _entry_label(entry: _UrlEntry) -> str:
        route = f"'{entry.route}'"
        if entry.func == "register":
            return f"router {route} -> {entry.view}"
        if entry.include:
            return f"{route} -> include({entry.include})"
        return f"{route} -> {entry.view or '?'}"

    def _entity_anchors(self) -> None:
        for model in self.models:
            self._model_anchors(model)
        for _, rel, d in self.enums:
            literals = list(d.attrs)
            self.app_anchors.append(Anchor(
                id=self._id("enum", d.name, file=rel), file=rel, kind=KIND_CLASS,
                label=f"choices {d.name} ({_short_list(literals) or 'no members'})", start_line=d.start,
                end_line=d.end, quote=self._quote(rel, d.line, span=(d.start, d.end)), hint=self._approx_hint(rel)))
        self._admin_anchors()
        for app in self.apps:
            for what, names, label, hint, then in (
                    ("models_end", ("models.py",), "new model classes", "new model classes",
                     f"migration: {self._migration_step(app)}"),
                    ("admin_end", ("admin.py",), "new admin registrations",
                     "new admin.site.register(...) calls / ModelAdmin classes", ""),
                    ("views_end", ("views.py",), "new views", "new view functions or classes", "")):
                for rel in app.named(*names):
                    self._insert_anchor(None, what, app.label, rel, self._module_end(rel), label, hint, then)
        for form_rel, d, entity in self.forms:
            self._add(entity, Anchor(
                id=self._id("form", (entity or "").lower(), d.name, file=form_rel), file=form_rel, kind=KIND_CLASS,
                label=f"ModelForm {d.name}", start_line=d.start, end_line=d.end,
                quote=self._quote(form_rel, d.line, span=(d.start, d.end)), hint=self._approx_hint(form_rel),
                entity=entity))
        for ser_rel, d, entity in self.serializers:
            self._add(entity, Anchor(
                id=self._id("serializer", (entity or "").lower(), d.name, file=ser_rel), file=ser_rel,
                kind=KIND_CLASS, label=f"serializer {d.name}", start_line=d.start, end_line=d.end,
                quote=self._quote(ser_rel, d.line, span=(d.start, d.end)), entity=entity))
        for _, view_rel, d, entity in self.views:
            kind = "view" if d.kind == "def" else ("viewset" if d.name.endswith("ViewSet") else "class view")
            self._add(entity, Anchor(
                id=self._id("view", (entity or "").lower(), d.name, file=view_rel), file=view_rel,
                kind=KIND_FUNCTION if d.kind == "def" else KIND_CLASS, label=f"{kind} {d.name}",
                start_line=d.start, end_line=d.end, quote=self._quote(view_rel, d.line, span=(d.start, d.end)),
                hint=self._approx_hint(view_rel), entity=entity))
        view_entity = {d.name: entity for _, _, d, entity in self.views}
        for conf in self.urlconfs:
            if conf.app is not None:
                self.app_anchors.extend(self._urlconf_anchors(conf, project=False))
            for entry in conf.entries:
                self._url_anchor(conf, entry, view_entity)
        for rel, entity in self.templates:
            lines = self.index.lines(rel)
            head = "\n".join(lines[:5])
            extends = _EXTENDS_RE.search(head)
            self._add(entity, Anchor(
                id=self._id("template", (entity or "").lower(), posixpath.splitext(posixpath.basename(rel))[0],
                            file=rel), file=rel, kind=KIND_FILE,
                label=f"template {posixpath.basename(rel)}" + (f" (extends {extends.group(1)})" if extends else ""),
                start_line=1, end_line=max(1, len(lines)), quote=self.index.first_unique_quote(rel, 1, len(lines)),
                entity=entity))

    def _entry_entity(self, entry: _UrlEntry, view_entity: dict[str, Optional[str]]) -> Optional[str]:
        """The entity a url pattern serves: its view's, else its view name's, else its first route segment's."""
        view = _last(entry.view) if entry.view else ""
        entity = view_entity.get(view)
        if entity is None and view:
            entity = self._entity_of_name(view) or self._lookup(_VIEW_SUFFIX_RE.sub("", view))
        if entity is None:
            first = next((s for s in entry.route.strip("^$/").split("/") if s and "<" not in s and "(" not in s), "")
            entity = self._from_token(first) if first else None
        return entity

    def _url_anchor(self, conf: _UrlConf, entry: _UrlEntry, view_entity: dict[str, Optional[str]]) -> None:
        if conf.app is None and not entry.include and entry.view.startswith("admin."):
            return      # the project's admin/ route is part of the urlpatterns block
        if entry.include:
            return      # the urlpatterns block lists includes
        view = _last(entry.view) if entry.view else ""
        entity = self._entry_entity(entry, view_entity)
        name = entry.name or view or re.sub(r"\W+", "_", entry.route).strip("_") or "root"
        lines = self.index.lines(conf.rel)
        quoted = (f"'{entry.route}'", f'"{entry.route}"')
        route_line = next((n for n in range(entry.start, min(entry.end, len(lines)) + 1)
                           if any(q in lines[n - 1] for q in quoted)), entry.start)
        self._add(entity, Anchor(
            id=self._id("url", (entity or "").lower(), name, file=conf.rel), file=conf.rel, kind=KIND_ROUTE,
            label=self._entry_label(entry) + (f" (name {entry.name})" if entry.name else ""),
            start_line=entry.start, end_line=entry.end,
            quote=self._quote(conf.rel, route_line, span=(entry.start, entry.end)), entity=entity))

    def _parent(self, model: _Model) -> Optional[_Model]:
        """The project model ``model`` subclasses, if any."""
        by_name = {m.d.name: m for m in self.models}
        return next((by_name[b] for b in model.d.bases if b in by_name and by_name[b] is not model), None)

    def _model_anchors(self, model: _Model) -> None:
        rel, d, entity = model.file, model.d, model.entity
        key = entity.lower()
        hints = []
        parent = self._parent(model)
        if model.proxy:
            hints.append(f"proxy of {parent.d.name if parent else 'its base'}: no fields allowed (models.E017), "
                         "only methods, managers and Meta")
        elif parent is not None and parent.abstract:
            hints.append(f"extends abstract {parent.d.name} (its fields are inherited, no parent table)")
        elif parent is not None:
            hints.append(f"inherits {parent.d.name} (multi-table inheritance: its fields are not repeated here)")
        if model.abstract:
            hints.append("abstract (Meta.abstract): no table, never registered in the admin")
        if self._approx_hint(rel):
            hints.append(self._approx_hint(rel))
        self._add(entity, Anchor(
            id=self._id("model", key, file=rel), file=rel, kind=KIND_CLASS,
            label=f"model {d.name} ({self._field_digest(d) or 'no fields'})", start_line=d.start, end_line=d.end,
            quote=self._quote(rel, d.line, span=(d.start, d.end)), hint="; ".join(hints), entity=entity))
        where = None if model.proxy else self._field_insert(model)
        if where:
            line_no, relation, extra = where
            sync = []
            if any(e == entity for _, _, e in self.admin_classes):
                sync.append("add it to list_display")
            form = next((f.name for _, f, e in self.forms if e == entity), None)
            if form:
                sync.append(f"add it to {form} Meta.fields")
            self._add(entity, Anchor(
                id=self._id("add_field", key, file=rel), file=rel, kind=KIND_INSERT,
                label=f"new {d.name} fields go {relation.lower()} this line", start_line=line_no, end_line=line_no,
                quote=self.index.unique_quote(rel, line_no),
                hint=f"insert new fields {relation} this line{extra}, before class Meta; "
                     + (f"then {', '.join(sync)}; " if sync else "") + f"migration: {self._migration_step(model.app)}",
                entity=entity))
        actions: dict[str, str] = {}     # model method -> the admin action that calls it
        for _, action, owner in self.admin_actions:
            for _, method, _ in action.calls:
                if owner == entity:
                    actions.setdefault(method, action.name)
        for child in d.body:
            if child.kind != "def" or (child.name.startswith("__") and child.name.endswith("__")):
                continue
            span = (child.start, child.end)
            if child.name == "clean":
                self._add(entity, Anchor(
                    id=self._id("clean", key, file=rel), file=rel, kind=KIND_FUNCTION,
                    label=f"{d.name}.clean() (model validation, run by full_clean/ModelForm/admin)",
                    start_line=child.start, end_line=child.end, quote=self._quote(rel, child.line, span=span),
                    hint=self._approx_hint(rel), entity=entity))
            elif child.stub:
                hint = "replace the placeholder with the real body, keep the signature"
                if child.name in actions:
                    hint += f"; admin action {actions[child.name]} calls it with no arguments"
                self._add(entity, Anchor(
                    id=self._id("stub", key, child.name, file=rel), file=rel, kind=KIND_STUB,
                    label=f"{d.name}.{child.name}() not implemented", start_line=child.start, end_line=child.end,
                    quote=self._quote(rel, child.stub_line, child.line, span=span), hint=hint,
                    entity=entity, status=_STUB_STATUS.get(child.stub, f"{child.stub} stub")))
            else:
                self._add(entity, Anchor(
                    id=self._id("method", key, child.name, file=rel), file=rel, kind=KIND_FUNCTION,
                    label=f"method {d.name}.{child.name}()", start_line=child.start, end_line=child.end,
                    quote=self._quote(rel, child.line, span=span), entity=entity))

    def _admin_anchors(self) -> None:
        for reg in self.registrations:
            if reg.via != "call":
                continue
            for model in reg.models:
                entity = self._lookup(model)
                if entity is None:
                    continue
                self._add(entity, Anchor(
                    id=self._id("admin", entity.lower(), file=reg.file), file=reg.file, kind=KIND_BLOCK,
                    label=f"admin.site.register({model}" + (f", {reg.admin_class})" if reg.admin_class else ")"),
                    start_line=reg.start, end_line=reg.end,
                    quote=self._quote(reg.file, reg.start, span=(reg.start, reg.end)),
                    hint=("" if reg.admin_class else
                          f"to customise, define {model}Admin(admin.ModelAdmin) and change this line to "
                          f"register({model}, {model}Admin); never register {model} twice"),
                    entity=entity))
        for rel, d, entity in self.admin_classes:
            options = [name for name in d.attrs if not name.startswith("_")]
            decorated = any(_last(name) == "register" for name, _ in d.decorators)
            self._add(entity, Anchor(
                id=self._id("admin_class", (entity or "").lower(), "" if entity else d.name, file=rel), file=rel,
                kind=KIND_CLASS, label=f"{d.name} ({_short_list(options) or 'no options'})"
                + (" @admin.register" if decorated else ""), start_line=d.start, end_line=d.end,
                quote=self._quote(rel, d.line, span=(d.start, d.end)), hint=self._approx_hint(rel), entity=entity))
        for rel, d, entity in self.admin_actions:
            self._add(entity, Anchor(
                id=self._id("admin_action", (entity or "").lower(), d.name, file=rel), file=rel,
                kind=KIND_FUNCTION, label=f"admin action {d.name}", start_line=d.start, end_line=d.end,
                quote=self._quote(rel, d.line, span=(d.start, d.end)), entity=entity))

    # -- missing pieces and pitfalls -----------------------------------------

    def _mentioned_in_broken(self, name: str, files: list[str]) -> bool:
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.I)
        return any(self._broken(rel) and pattern.search(self.index.text(rel) or "") for rel in files)

    def _registration_replay(self) -> tuple[set[str], list[tuple[_Registration, str, str]]]:
        """(models registered in the end, (registration, model, earlier via) for each double registration).

        Registrations and ``admin.site.unregister(X)`` calls are replayed in
        file order, so an unregister between two registrations is not a clash.
        """
        live: dict[str, str] = {}
        twice: list[tuple[_Registration, str, str]] = []
        for reg in sorted(self.registrations + self.unregistrations, key=lambda r: (r.file, r.start, r.via)):
            for model in reg.models:
                if reg.via == "unregister":
                    live.pop(model, None)
                elif model in live:
                    twice.append((reg, model, live[model]))
                else:
                    live[model] = reg.via
        return set(live), twice

    def _missing(self) -> None:
        if not self.class_names:
            return
        models_files = sorted({rel for app in self.apps for rel in app.named("models.py")}
                              | {rel for app in self.apps for rel in app.in_dir("models")})
        admin_files = sorted({rel for app in self.apps for rel in app.named("admin.py")})
        modelled = {m.entity for m in self.models}
        registered = {self._lookup(m) or m for m in self._registration_replay()[0]}
        abstract = {m.entity for m in self.models if m.abstract}
        for name in self.class_names:
            entity = self._lookup(name) or name
            missing = []
            if entity not in modelled and not self._mentioned_in_broken(entity, models_files):
                missing.append("model class")
            if (admin_files and entity not in registered and entity not in abstract
                    and not self._mentioned_in_broken(entity, admin_files)):
                missing.append("admin registration")
            self._entity(entity).missing = missing

    def _warn(self, text: str) -> None:
        if len(self.warnings) < _MAX_WARNINGS:
            self.warnings.append(text)
        else:
            self.extra_warnings += 1

    def _detect_pitfalls(self) -> None:
        self._admin_pitfalls()
        self._model_pitfalls()
        self._url_pitfalls()
        self._primary_key_pitfalls()
        self._settings_pitfalls()

    def _admin_pitfalls(self) -> None:
        how = {"call": "admin.site.register", "decorator": "@admin.register"}
        for reg, model, earlier in self._registration_replay()[1]:
            if self._lookup(model) is None:
                continue
            both = sorted({earlier, reg.via})
            twice = " and ".join(how[v] for v in both) if len(both) == 2 else f"two {how[reg.via]} calls"
            self._warn(f"{reg.file}: {model} is registered twice ({twice}): AlreadyRegistered at startup.")
        # Methods per entity, inherited ones included (Author(Person) gets Person's methods).
        methods: dict[tuple[str, str], _Def] = {}
        by_name = {m.d.name: m for m in self.models}
        for m in self.models:
            chain, visited = [m], set()
            while chain:
                current = chain.pop()
                if current.d.name in visited:
                    continue
                visited.add(current.d.name)
                for child in current.d.body:
                    if child.kind == "def":
                        methods.setdefault((m.entity, child.name), child)
                chain += [by_name[b] for b in current.d.bases if b in by_name]
        # One line per admin file: a model with several parameterised methods must not flood the overview.
        broken: dict[str, list[tuple[str, str, list[str]]]] = {}
        for rel, action, entity in self.admin_actions:
            for _, method, n_args in action.calls:
                target = methods.get((entity or "", method))
                if target is None or target.required is None or n_args:
                    continue
                needed = target.required[1:]
                if needed:
                    broken.setdefault(rel, []).append((action.name, method, needed))
        for rel, calls in broken.items():
            calls.sort()
            if len(calls) == 1:
                name, method, needed = calls[0]
                self._warn(f"{rel} {name} calls obj.{method}() without its required {', '.join(needed)}: the "
                           "action fails on every selected row.")
            else:
                shown = _short_list([f"{name} -> {method}({', '.join(needed)})" for name, method, needed in calls], 3)
                self._warn(f"{rel}: {len(calls)} admin actions call a model method without its required arguments "
                           f"and fail on every selected row: {shown}.")

    def _model_pitfalls(self) -> None:
        by_app: dict[str, set[str]] = {}
        for m in self.models:
            by_app.setdefault(m.app.label, set()).add(m.d.name)
        local_labels = {app.label.rsplit(".", 1)[-1]: app.label for app in self.apps}
        for m in self.models:
            app_broken = any(self._broken(rel) for rel in m.app.named("models.py") + m.app.in_dir("models"))
            for f in m.d.fields:
                if f.call in ("ForeignKey", "OneToOneField") and not f.has_on_delete:
                    self._warn(f"{m.file}: {m.d.name}.{f.name} is a {f.call} without on_delete: TypeError when "
                               "the models are imported.")
                if not f.target_is_str or not f.target or f.target == "self" or "settings" in f.target or app_broken:
                    continue
                if "." in f.target:
                    label, target = f.target.rsplit(".", 1)
                    app_label = local_labels.get(label)
                    if app_label is None or target in by_app.get(app_label, set()):
                        continue
                elif f.target in by_app.get(m.app.label, set()):
                    continue
                self._warn(f"{m.file}: {m.d.name}.{f.name} references '{f.target}', which this app does not "
                           "define: makemigrations fails.")

    def _url_pitfalls(self) -> None:
        included = [e.include for conf in self.urlconfs for e in conf.entries if e.include]
        unknown_include = "?" in included
        local_tops = {app.label.split(".")[0] for app in self.apps}
        for conf in self.urlconfs:
            if conf.uses_include and not conf.imports_include and not conf.approximate:
                self._warn(f"{conf.rel} calls include() without importing it: NameError at startup.")
            for entry in conf.entries:
                if (entry.include and entry.include != "?" and entry.include.split(".")[0] in local_tops
                        and not self._module_exists(entry.include)):
                    self._warn(f"{conf.rel} includes '{entry.include}', which does not exist: ImportError at "
                               "startup.")
            if conf.app is None or unknown_include:
                continue
            module = conf.app.label + ".urls"
            if self.project_urls is not None and not any(i == module or i.endswith("." + module) for i in included):
                self._warn(f"{conf.rel} is never include()d, so its routes are unreachable: include it at "
                           f"`{self.fid}:urlpatterns_end`.")
        referenced: set[str] = set()
        route_kwargs: set[str] = set()
        for conf in self.urlconfs:
            for entry in conf.entries:
                referenced.update(entry.view.split(".") if entry.view else [])
                referenced.update(entry.refs)
                route_kwargs.update(_ROUTE_KWARG_RE.findall(entry.route))
        for rel in sorted({rel for _, rel, _, _ in self.views}):
            defs = [d for _, r, d, _ in self.views if r == rel]
            called = set().union(*(d.names_called for d in defs)) if defs else set()
            bases = {_last(b) for d in defs for b in d.bases}
            unrouted = [d.name for d in defs if d.name not in referenced and d.name not in called
                        and d.name not in bases and not d.name.startswith("_")
                        and (d.kind == "class" or self._view_signature(d, route_kwargs))]
            if unrouted:
                self._warn(f"{rel}: {_short_list(unrouted, 4)} " + ("has" if len(unrouted) == 1 else "have")
                           + " no route in any urls.py.")

    @staticmethod
    def _view_signature(d: _Def, route_kwargs: set[str]) -> bool:
        """``def x(request, pk)``: ``request``, then only what a URL passes (``pk``, ``book_id``, ``slug``...).

        ``handle_create(request, form_class, success_message)`` takes ``request``
        too, but it is a helper the views call: a route to it raises TypeError on
        every request. A signature the line scanner could not read is not claimed.
        """
        if d.required is None or d.required[:1] != ["request"]:
            return False
        return all(p in route_kwargs or p in _URL_KWARGS or p.endswith(_URL_KWARG_SUFFIXES) for p in d.required[1:])

    def _primary_key_pitfalls(self) -> None:
        """Views/urls that assume an integer ``id`` for a model whose primary key is a CharField (``is_id``).

        The generated GUI views look rows up with ``get_object_or_404(Seat, id=seat_id)``
        behind ``<int:seat_id>``: with ``code = CharField(primary_key=True)`` there is
        no ``id`` column, so every edit/delete raises FieldError.
        """
        keys: dict[str, tuple[str, str]] = {}       # model class -> (entity, pk field)
        for m in self.models:
            pk = next((f for f in m.d.fields if f.primary_key), None)
            if pk is not None and pk.call not in _INT_PK_FIELDS and pk.call not in _RELATION_CALLS:
                keys[m.d.name] = (m.entity, pk.name)
        if not keys:
            return
        lookups = {name: re.compile(rf"\b{re.escape(name)}\s*,\s*id\s*=|\b{re.escape(name)}\.objects\.\w+\(\s*id\s*=")
                   for name in keys}
        by_file: dict[str, tuple[list[str], set[str]]] = {}     # views file -> (views, models)
        for _, rel, d, _ in self.views:
            body = "\n".join(self.index.lines(rel)[d.start - 1:d.end])
            for name, pattern in lookups.items():
                if pattern.search(body):
                    views, models = by_file.setdefault(rel, ([], set()))
                    if d.name not in views:
                        views.append(d.name)
                    models.add(name)
        entities = {entity: name for name, (entity, _) in keys.items()}
        view_entity = {d.name: entity for _, _, d, entity in self.views}
        int_routes = {entities[e] for conf in self.urlconfs for e in [self._entry_entity(x, view_entity)
                                                                      for x in conf.entries if "<int:" in x.route]
                      if e in entities}
        for rel, (views, models) in sorted(by_file.items()):
            pks = [f"{name}.{keys[name][1]}" for name in sorted(models)]
            converters = " and <str:...> instead of <int:...> in their routes" if models & int_routes else ""
            self._warn(f"{rel}: {_short_list(views, 3)} look rows up by id= but the primary key of "
                       f"{_short_list(pks, 3)} is not id: FieldError on every call; use pk={converters}.")
        reported = {name for _, models in by_file.values() for name in models}
        for name in sorted(int_routes - reported):
            rel = next((conf.rel for conf in self.urlconfs for x in conf.entries
                        if "<int:" in x.route and self._entry_entity(x, view_entity) == keys[name][0]), "urls.py")
            self._warn(f"{rel}: the {name} routes use <int:...> but {name}.{keys[name][1]} is a "
                       "string primary key: use <str:...>.")

    def _settings_pitfalls(self) -> None:
        if self.settings is None:
            return
        installed_tops = {i.split(".")[0] for i in self.settings.installed()}
        for app in self.apps:
            if not app.installed and any(m.app is app for m in self.models):
                self._warn(f"{app.dir} has models but '{app.label}' is not in INSTALLED_APPS: Django refuses to "
                           "load them.")
        used: dict[str, str] = {}
        for app in self.apps:
            for rel in self._app_py(app):
                for top in _IMPORT_LINE_RE.findall(self.index.text(rel) or ""):
                    if top in _IMPORTED_APPS:
                        used.setdefault(top, rel)
        middleware = self.settings.lists.get("MIDDLEWARE")
        if middleware and any(v.startswith("corsheaders.") for v in middleware.values()):
            used.setdefault("corsheaders", self.settings.rel)
        for top, rel in sorted(used.items()):
            if top not in installed_tops and self.settings.apps_known:
                self._warn(f"'{top}' is used in {rel} but is not in INSTALLED_APPS.")
        if self.requirements:
            listed = set()
            for line in self.index.lines(self.requirements):
                m = _REQUIREMENT_NAME_RE.match(line.split("#", 1)[0])
                if m:
                    listed.add(m.group(1).lower().replace("_", "-"))
            for top in sorted(installed_tops | set(used)):
                package = _PACKAGE_OF.get(top)
                if package and package not in listed:
                    self._warn(f"{self.requirements} lacks {package} ('{top}' is used).")

    # -- summaries and recipes -----------------------------------------------

    def _field_digest(self, d: _Def) -> str:
        plain = [f.name for f in d.fields if f.call not in _RELATION_CALLS]
        relations = [f"{_RELATION_CALLS[f.call]} {f.name}->{_last(f.target) or '?'}"
                     for f in d.fields if f.call in _RELATION_CALLS]
        parts = [_short_list(plain, 5)] if plain else []
        parts += relations[:3] + ([f"+{len(relations) - 3} relations"] if len(relations) > 3 else [])
        return "; ".join(parts)

    def _primary_app(self) -> Optional[_App]:
        with_models = [a for a in self.apps if any(m.app is a for m in self.models)]
        installed = [a for a in with_models if a.installed] or with_models or self.apps
        return installed[0] if installed else None

    def _has_gui(self) -> bool:
        return any(conf.app is not None and conf.entries for conf in self.urlconfs) or any(
            d.name for _, _, d, _ in self.views)

    def _title(self) -> str:
        project = posixpath.basename(self.project_pkg) or posixpath.basename(self.root) or "project"
        if self.serializers or any(d.name.endswith("ViewSet") for _, _, d, _ in self.views):
            variant = "admin + DRF API"
        elif self._has_gui():
            variant = "admin + server-rendered views"
        else:
            variant = "admin-only scaffold, no app URLs"
        return f"Django project {project} ({variant})"

    def _entity_summary(self, entity: str) -> list[str]:
        parts: list[str] = []
        multi_app = len(self.apps) > 1
        for m in (m for m in self.models if m.entity == entity):
            text = "model" if m.d.name == entity else f"model {m.d.name}"
            text += f" in {m.app.label}" if multi_app else ""
            digest = self._field_digest(m.d)
            text += f" ({digest})" if digest else ""
            parent = self._parent(m)
            if m.proxy:
                text += f", proxy of {parent.d.name}" if parent else ", proxy"
            elif parent is not None:
                text += f", extends {'abstract ' if parent.abstract else ''}{parent.d.name}"
            if m.abstract:
                text += ", abstract"
            parts.append(text)
            if m.d.child("clean"):
                parts.append("clean()")
            stubs = [c.name for c in m.d.body if c.kind == "def" and c.stub]
            if stubs:
                parts.append("stubs " + _short_list(stubs, 4))
        admin_classes = [d.name for _, d, e in self.admin_classes if e == entity]
        actions = [d.name for _, d, e in self.admin_actions if e == entity]
        if admin_classes:
            # The conventional name is implied: "ModelAdmin" keeps a 15-entity overview within budget.
            name = "ModelAdmin" if admin_classes[0] == f"{entity}Admin" else f"admin {admin_classes[0]}"
            parts.append(name + (f" ({_plural(len(actions), 'action')})" if actions else ""))
        elif any((self._lookup(model) or model) == entity for r in self.registrations for model in r.models):
            parts.append("admin: plain register")
        # The GUI/API pieces as counts: locate_code entity=<name> lists them one by one.
        gui = []
        views = [d.name for _, _, d, e in self.views if e == entity]
        if views:
            gui.append(_plural(len(views), "view"))
        urls = [a for a in self._entity(entity).anchors if a.kind == KIND_ROUTE]
        if urls:
            gui.append(_plural(len(urls), "url"))
        forms = [d.name for _, d, e in self.forms if e == entity]
        if forms:
            gui.append("form " + _short_list(forms, 2))
        serializers = [d.name for _, d, e in self.serializers if e == entity]
        if serializers:
            gui.append("serializer " + _short_list(serializers, 2))
        templates = [rel for rel, e in self.templates if e == entity]
        if templates:
            gui.append(_plural(len(templates), "template"))
        if gui:
            parts.append(", ".join(gui))
        return parts

    def _map_summary(self) -> list[str]:
        lines: list[str] = []
        s = self.settings
        if s is not None:
            installed = s.installed()
            extras = []
            if s.engine:
                extras.append(f"DB {s.engine}")
            if "jazzmin" in installed:
                extras.append("admin theme django-jazzmin")
            if s.named:
                extras.append(f"extended by {s.named}")
            lines.append(f"settings {s.rel}" + (f" ({', '.join(extras)})" if extras else ""))
        conf = self.project_urls
        if conf is not None:
            routes = [self._entry_label(e) for e in conf.entries]
            only_admin = routes and all(e.view.startswith("admin.") for e in conf.entries)
            desc = ("admin/ only, no include() of an app" if only_admin
                    else _short_list(routes, 4) if routes else "no urlpatterns")
            lines.append(f"project urls {conf.rel}: {desc}")
        for app in self.apps:
            pieces = []
            has_models = any(m.app is app for m in self.models)
            # The package holding settings and the ROOT_URLCONF is not an app (unless it defines models).
            is_project = app.dir == self.project_pkg and not has_models
            if has_models:
                pieces.append("models")
            if app.named("admin.py"):
                pieces.append("admin")
            views = [d for a, _, d, _ in self.views if a is app]
            if app.named("views.py"):
                pieces.append(f"views ({len(views)})" if views else "views.py stub")
            if any(posixpath.dirname(rel) == app.dir for rel, _, _ in self.forms):
                pieces.append("forms")
            if any(posixpath.dirname(rel) == app.dir for rel, _, _ in self.serializers):
                pieces.append("serializers")
            confs = [c for c in self.urlconfs if c.app is app]
            if self.project_urls is not None and posixpath.dirname(self.project_urls.rel) == app.dir:
                pieces.append("the project urls.py")
            elif confs:
                pieces.append(f"urls.py ({sum(len(c.entries) for c in confs)} routes)")
            else:
                pieces.append("urls.py (not mapped)" if self._exists(_join(app.dir, "urls.py")) else "NO urls.py")
            templates = [rel for rel, _ in self.templates if rel.startswith(app.dir + "/")]
            if templates:
                pieces.append(f"{len(templates)} templates")
            if not is_project:
                migrations = self.migrations.get(app.label, [])
                pieces.append(f"{len(migrations)} migration(s)" if migrations else "no migrations yet")
            lines.append(f"{'project package' if is_project else 'app'} {app.label} ({app.dir}): {', '.join(pieces)}"
                         + ("" if app.installed or is_project else " — NOT in INSTALLED_APPS"))
        return lines

    def _migration_step(self, app: Optional[_App]) -> str:
        """What a model change needs next.

        Phase 2 may have no shell (no run_command) and nothing else runs
        makemigrations; a hand-written AddField with no initial migration under
        it breaks ``migrate``, so a migration is asked for only when some ship.
        """
        label = app.label if app else "<app>"
        if app is not None and self.migrations.get(app.label):
            return (f"a matching one, by `makemigrations {label}` with run_command, "
                    f"else by hand in {app.dir}/migrations/")
        return (f"`makemigrations {label}` only with run_command, never hand-written "
                "(none ship: makemigrations creates them all at setup)")

    def _base_templates(self) -> list[tuple[str, str, Optional[str]]]:
        """(name a new page extends, the block it fills, entity) of each base template, the likeliest first.

        A base defines a ``{% block %}``, extends nothing itself and is found by
        that name: under an app's ``templates/``, or the project's ``templates/``
        when TEMPLATES ``DIRS`` is set. Extending anything else raises
        TemplateDoesNotExist, or renders nothing.
        """
        dirs = [_join(a.dir, "templates") for a in self.apps]
        s = self.settings
        if s is not None and "TEMPLATES" in s.blocks:
            start, end = s.blocks["TEMPLATES"]
            if _TEMPLATE_DIRS_RE.search("\n".join(self.index.lines(s.rel)[start - 1:end])):
                dirs.append(_join(self.root, "templates"))
        dirs.sort(key=len, reverse=True)
        children: dict[str, int] = {}
        found: list[tuple[str, list[str], Optional[str]]] = []
        for rel, entity in self.templates:
            text = self.index.text(rel) or ""
            parent = _EXTENDS_RE.search(text)
            if parent:
                children[parent.group(1)] = children.get(parent.group(1), 0) + 1
                continue
            base_dir = next((d for d in dirs if rel.startswith(d + "/")), None)
            blocks = _BLOCK_RE.findall(text)
            if base_dir and blocks:
                found.append((posixpath.relpath(rel, base_dir), blocks, entity))
        found.sort(key=lambda b: ("content" not in b[1], posixpath.basename(b[0]) not in ("base.html", "layout.html"),
                                  -children.get(b[0], 0), b[0]))
        return [(name, "content" if "content" in blocks else blocks[0], entity) for name, blocks, entity in found]

    def _template_step(self, app_dir: str) -> str:
        """Where a new page's template goes, and the base template it can extend (named, never guessed)."""
        bases = self._base_templates()
        if not bases:
            return f"A standalone template in {app_dir}/templates/: no base template to extend."
        name, block, _ = bases[0]
        # The GUI scaffold has one base per entity (book.html, author.html), each with its own scripts.
        which = (f"its entity's base, e.g. '{name}'" if len(bases) > 1 and all(b[2] for b in bases)
                 else f"'{name}'")
        return f"Template in {app_dir}/templates/ extending {which} ({{% block {block} %}})."

    def _recipes(self, smap: ScaffoldMap) -> list[Recipe]:
        ids = {a.id for a in smap.all_anchors()}
        fid = self.fid

        def has(anchor_id: str) -> bool:
            return f"{fid}:{anchor_id}" in ids or any(i.startswith(f"{fid}:{anchor_id}:") for i in ids)

        def ref(anchor_id: str, fallback: str) -> str:
            return f"`{fid}:{anchor_id}`" if has(anchor_id) else fallback

        app = self._primary_app()
        label = app.label if app else "<app>"
        app_dir = app.dir if app else _join(self.root, "<app>")
        root = self.root or "."
        models_rel = _join(app_dir, "models.py")
        admin_rel = _join(app_dir, "admin.py")
        views_rel = _join(app_dir, "views.py")
        project_urls = self.project_urls.rel if self.project_urls else "the project urls.py"
        recipes: list[Recipe] = []
        sync = []
        if self.admin_classes:
            sync.append("list_display")
        if self.forms:
            sync.append("<Entity>Form Meta.fields")
        recipes.append(Recipe("Add a field", [
            f"`{fid}:add_field:<entity>` (before class Meta)" + (f", also {' and '.join(sync)}" if sync else "")
            + "; migration: see Run migrations."]))
        called = "; admin actions call it with no arguments." if self.admin_actions else "."
        if has("stub"):
            recipes.append(Recipe("Implement a model method", [f"Replace `{fid}:stub:<entity>:<method>`{called}"]))
        elif has("method"):
            recipes.append(Recipe("Implement a model method", [
                f"Edit `{fid}:method:<entity>:<method>` (keep the signature){called}"]))
        app_conf = next((c for c in self.urlconfs if c.app is app and c.found), None) if app else None
        include_import = (self.project_urls is not None and self.project_urls.import_line
                          and not self.project_urls.imports_include)
        template = self._template_step(app_dir)
        app_urls = _join(app_dir, "urls.py")
        if app_conf is not None:
            recipes.append(Recipe("Add a page or view", [
                f"View at {ref('views_end:' + label, views_rel)}, path(...) at "
                f"{ref('app_urlpatterns_end:' + label, app_conf.rel)}.", template]))
        elif app is not None and self._exists(app_urls):
            recipes.append(Recipe("Add a page or view", [
                f"View at {ref('views_end:' + label, views_rel)}, path(...) in {app_urls} (read it first).", template]))
        else:
            recipes.append(Recipe("Add a page or view", [
                f"View at {ref('views_end:' + label, views_rel)}; no app urls.py yet: CREATE {app_urls}, "
                f"path('', include('{label}.urls')) at {ref('urlpatterns_end', project_urls)}"
                + (f" (import include at `{fid}:urls_import`)" if include_import and has("urls_import") else "")
                + ".", template]))
        drf = self.serializers or "rest_framework" in (self.settings.installed() if self.settings else [])
        if drf:
            recipes.append(Recipe("Add a REST endpoint", [
                f"Extend the serializers and ViewSets in {app_dir}/ and register them on the app urls.py router."]))
        else:
            recipes.append(Recipe("Add a REST API", [
                f"DRF in {app_dir}/: serializers.py, ModelViewSets, a DefaultRouter in urls.py; 'rest_framework' "
                f"at {ref('installed_apps_end', 'INSTALLED_APPS')}, djangorestframework at "
                f"{ref('requirements_end', 'requirements.txt')}; not FastAPI/Flask."]))
        if self.registrations or self.admin_classes:
            recipes.append(Recipe("Customise the admin", [
                f"Edit `{fid}:admin_class:<entity>` or turn `{fid}:admin:<entity>` into "
                "register(<Entity>, <Entity>Admin); never register twice."]))
        recipes.append(Recipe("Add a new model", [
            f"Class at {ref('models_end:' + label, models_rel)}, register at "
            f"{ref('admin_end:' + label, admin_rel)}; migration: see Run migrations."]))
        # Phase 2 often has no shell: run_command may be absent, and nothing else runs makemigrations.
        command = f"`python manage.py makemigrations {label}`"
        if app is not None and self.migrations.get(app.label):
            recipes.append(Recipe("Run migrations", [
                f"Migrations ship in {app_dir}/migrations/, so a model change needs a matching one.",
                f"With run_command: {command} then `migrate`, from {root}/; without it, write the next "
                "migration file by hand."]))
        else:
            recipes.append(Recipe("Run migrations", [
                "Never hand-write a migration: none ship, makemigrations creates them all at setup.",
                f"With run_command: {command} then `migrate`, from {root}/."]))
        if self.settings and "django.contrib.auth" in self.settings.installed():
            recipes.append(Recipe("Add auth", [
                f"include('django.contrib.auth.urls') at {ref('urlpatterns_end', project_urls)}, "
                f"@login_required, {app_dir}/templates/registration/login.html."]))
        if self.settings:
            recipes.append(Recipe("Change settings", [
                f"{ref('installed_apps_end', 'INSTALLED_APPS')}, {ref('middleware_end', 'MIDDLEWARE')}, "
                f"{ref('databases', 'DATABASES')}, {ref('settings_end', 'the end of the settings')}; packages "
                f"{ref('requirements_end', 'requirements.txt')}."]))
        return recipes
