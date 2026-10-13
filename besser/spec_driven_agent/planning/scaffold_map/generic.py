"""Scaffold map extractor for every file the framework extractors leave unclaimed.

The FastAPI, Django and React extractors recognise application roots; this one
maps the rest of the workspace: the one-folder outputs of the specialist
generators (``baf/``, ``qiskit/``, ``bpmn/``, ``pytorch/``, ``tensorflow/``,
``json_object/``, ``python/``, ``java/``, ``sql/``, ``sqlalchemy/``,
``pydantic/``, ``supabase/``, ``rdf/``, ``jsonschema/`` ...), code written from
scratch, and the loose files next to a recognised root. Files are grouped by
the shallowest directory that holds no claimed root (normally the top-level
folder), and each group becomes one :class:`ScaffoldMap` of family ``generic``.
A scaffold claimed at the workspace root (``''``) owns its whole tree: only
folders holding a specialist scaffold are mapped next to it, never its own
``routers/`` or ``src/``.

Per file, from the AST (Python) or line patterns (TypeScript/JavaScript, Java):

* every top-level class and component, every method of a model class and
  every def of an agent, circuit or network, decorator-inclusive; placeholder
  bodies (``pass``/``...``, ``NotImplementedError``, HTTP 501, a ``# TODO``
  comment, ``return 'TODO'``, Java ``UnsupportedOperationException``) become
  ``stub`` anchors with a status. Other (plain) functions and methods are
  named in the file's label and the summary, not anchored: the overview lists
  every app-level anchor, and a folder of 30 test functions would fill it;
* trivial accessors (Python ``@property`` pairs, Java ``getX``/``setX``) folded
  into one block per class, so a 15-class scaffold stays readable;
* every other text file (.sql, .ttl, .json, .yaml, .md ...) as one ``file``
  anchor carrying its role, plus one block per SQL table and RDF class.

Cheap enrichments recognise the specialist scaffolds by content, never by the
generator that was asked for: BESSER agents (states, bodies, intents, entities,
LLMs, insertion points; body-less states and ``set_body`` of an undefined
function are reported), Qiskit circuits (the ``# Operations`` block,
placeholder gates, execution), PyTorch/TensorFlow networks (layers,
forward/call, training code) and BPMN processes (element ids, gateways whose
outgoing flows have no condition).

Anchor ids are ``generic:<what>:<file>[:<symbol>]``. The workspace-relative
path keeps them unique across groups without a root slug, and the symbol is a
name, never a line: ``generic:method:python/classes.py:Book.is_long``,
``generic:state:baf/support_bot.py:greeting``,
``generic:bpmn:bpmn/bpmn_diagram.bpmn:Gateway_1``. A repeated symbol gets
``~2``, ``~3`` (never ``:2``, which reads as a line number).
"""

from __future__ import annotations

import ast
import json
import logging
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
    KIND_STUB,
    Anchor,
    EntityMap,
    Recipe,
    ScaffoldMap,
)
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex, node_start

logger = logging.getLogger(__name__)

FAMILY = "generic"

# Caps keep the map compact; what they cut is counted in the map summary.
MAX_ANCHORS_PER_FILE = 40          # app-level anchors of one file
MAX_ANCHORS_PER_ENTITY_FILE = 12   # anchors of one domain class in one file
MAX_ANCHORS_PER_MAP = 400
MAX_FILE_ANCHORS_PER_MAP = 80
MAX_SUMMARY_LINES = 8
MAX_WARNINGS = 6
MAX_RECIPES = 4
_MAX_SUMMARY_CHARS = 200
_MAX_QUOTE_CHARS = 240             # a longer unique line (minified, generated data) is no usable edit anchor
_MAX_NAMES = 6

# Lower survives a cap longer.
_P_STUB, _P_KEY, _P_BLOCK, _P_FUNCTION, _P_ACCESSORS, _P_METHOD, _P_MINOR = range(7)

_PY_SUFFIXES = (".py",)
_SCRIPT_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_JAVA_SUFFIXES = (".java",)
_CODE_SUFFIXES = _PY_SUFFIXES + _SCRIPT_SUFFIXES + _JAVA_SUFFIXES

# Text files mapped as one ``file`` anchor; the value is the role when nothing more specific is found.
_TEXT_ROLES = {
    ".sql": "SQL", ".ttl": "RDF (Turtle)", ".rdf": "RDF/XML", ".owl": "OWL ontology", ".nt": "RDF triples",
    ".jsonld": "JSON-LD", ".bpmn": "BPMN process", ".xml": "XML", ".json": "JSON data", ".yaml": "YAML",
    ".yml": "YAML", ".toml": "TOML config", ".ini": "INI config", ".cfg": "config", ".md": "Markdown",
    ".txt": "text", ".rst": "reStructuredText", ".html": "HTML page", ".htm": "HTML page", ".css": "stylesheet",
    ".scss": "stylesheet", ".sh": "shell script", ".bat": "batch script", ".ps1": "PowerShell script",
    ".csv": "CSV data", ".graphql": "GraphQL schema", ".proto": "protobuf schema", ".dart": "Dart source",
    ".kt": "Kotlin source", ".kts": "Kotlin script", ".gradle": "Gradle build", ".rs": "Rust source",
    ".go": "Go source", ".rb": "Ruby source", ".php": "PHP source", ".cs": "C# source", ".c": "C source",
    ".h": "C header", ".cpp": "C++ source", ".hpp": "C++ header", ".swift": "Swift source",
    ".vue": "Vue component", ".svelte": "Svelte component",
}
_NAMED_ROLES = {
    "Dockerfile": "Dockerfile (container build)", "Makefile": "Makefile", "Procfile": "Procfile",
    "Gemfile": "Ruby dependencies",
}
_LOCK_FILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock", "Cargo.lock",
    "composer.lock", "Gemfile.lock", "uv.lock", "bun.lockb", "npm-shrinkwrap.json",
})
_SKIP_SUFFIXES = (".min.js", ".min.css", ".map")
# Folder names of the specialist generators, for the title when the content says nothing more specific.
_FOLDER_TITLES = {
    "baf": "BESSER agent (BAF)", "qiskit": "Qiskit quantum circuit", "bpmn": "BPMN process",
    "pytorch": "PyTorch neural network", "tensorflow": "TensorFlow/Keras neural network",
    "json_object": "Object model (JSON)", "python": "Python classes", "java": "Java classes",
    "sql": "SQL schema", "sqlalchemy": "SQLAlchemy models", "pydantic": "Pydantic models",
    "supabase": "Supabase SQL", "rdf": "RDF vocabulary", "jsonschema": "JSON Schema",
    "rest_api": "REST API (single file)",
}
# Content flavours, most specific first; the first two found name the map.
_FLAVOUR_TITLES = (
    ("baf", "BESSER agent (BAF)"), ("qiskit", "Qiskit quantum circuit"),
    ("pytorch", "PyTorch neural network"), ("tensorflow", "TensorFlow/Keras neural network"),
    ("bpmn", "BPMN process"), ("object_json", "Object model (JSON)"), ("sqlalchemy", "SQLAlchemy ORM models"),
    ("pydantic", "Pydantic models"), ("python_classes", "Python domain classes"), ("java", "Java classes"),
    ("supabase", "Supabase SQL migration"), ("sql", "SQL schema"), ("rdf", "RDF vocabulary"),
    ("json_schema", "JSON Schema"), ("django", "Django code"), ("fastapi", "FastAPI code"),
    ("flask", "Flask app"),
)
# Per-class scaffolds: the flavour, and what an absent class is missing. SQL is not listed: an
# abstract class has no table in the SQL generator's output, and Supabase names an association
# class's table after its association, so an absent table name proves nothing.
_PER_CLASS_PIECES = (
    ("python_classes", "Python class"), ("pydantic", "Pydantic model"), ("sqlalchemy", "ORM class"),
    ("java", "Java class"), ("rdf", "RDF class"),
)
# A class's pieces are collected only from a file that carries its generator's banner: a
# hand-written module may hold any subset of the model.
_PER_CLASS_MARKERS = {
    "python_classes": "# Definition of Classes", "pydantic": "# Classes are defined here",
    "sqlalchemy": "# Tables definition", "rdf": "# Class definition",
}
# Folders holding one of these next to a scaffold claimed at the workspace root are still mapped:
# no framework extractor maps them. Anything else there belongs to the root scaffold.
_SPECIALIST_FLAVOURS = frozenset({
    "baf", "qiskit", "pytorch", "tensorflow", "bpmn", "object_json", "python_classes", "java", "supabase",
    "sql", "rdf", "json_schema",
})
# Generated ORM / schema folders, specialist only under their generator's folder name.
_SPECIALIST_FOLDERS = {"sqlalchemy": "sqlalchemy", "pydantic": "pydantic"}
# Plain functions are anchored only in these, where every def is part of the scaffold's behaviour.
_BEHAVIOUR_FLAVOURS = frozenset({"baf", "qiskit", "pytorch", "tensorflow"})
# Schema classes named after their entity (``BookCreate`` -> ``Book``), for Pydantic files only.
_SCHEMA_SUFFIXES = ("Create", "Update", "Base", "Read", "Response", "Out")
_NN_BASES = frozenset({
    "nn.Module", "torch.nn.Module", "Module", "tf.keras.Model", "keras.Model", "Model",
    "tf.keras.layers.Layer", "keras.layers.Layer", "layers.Layer",
})
_NN_TRAINING_FUNCTIONS = ("prepare_datasets", "train", "evaluate", "load_dataset", "test")
_SKIP_STUB_DECORATORS = frozenset({"abstractmethod", "overload", "abstractproperty"})

_TODO_COMMENT_RE = re.compile(r"^\s*(?:#|//)\s*TODO\b[:\s-]*(.*)$")
_PY_DEF_RE = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")
_QISKIT_GATE_RE = re.compile(r"^(_function_gate_\w+?)_\d{6,}$")
_BANNER_RE = re.compile(r"^# ([A-Z][A-Z0-9 &/_-]*[A-Z0-9])\s*$")
_STATE_MARKER_RE = re.compile(r"^# (\w+)\s*$")
_PLACEHOLDER_GATE_RE = re.compile(r"create_placeholder\(\s*['\"]([^'\"]*)['\"]")
_EXECUTION_RE = re.compile(r"^# Execution\b\s*(?:\((.*)\))?")
_SHOTS_RE = re.compile(r"\bshots\s*=\s*(\d+)")
_REGISTER_RE = re.compile(r"^(\w+)\s*=\s*(Quantum|Classical)Register\((\d+)")
_XML_TAG_RE = re.compile(r"<(?:(\w+):)?(\w+)\b([^<>]*?)(/?)>")
# A name starts after a non-name character: from every position inside a long name the scan was quadratic.
_XML_ATTR_RE = re.compile(r"(?<![\w:.-])([\w:.-]+)\s*=\s*\"([^\"]*)\"")
_YOUR_PLACEHOLDER_RE = re.compile(r"\bYOUR[-_][A-Z0-9_-]+")
_SQL_TABLE_RE = re.compile(
    r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"?(\w+)\"?\.)?\"?(\w+)\"?\s*\(", re.I)
_SQL_REFERENCES_RE = re.compile(r"REFERENCES\s+(?:\"?(\w+)\"?\.)?\"?(\w+)\"?\s*\(\s*\"?id\"?\s*\)", re.I)
_SQL_ID_COLUMN_RE = re.compile(r"^\s*\"?id\"?\s+\w", re.I)
_RDF_CLASS_RE = re.compile(r"^(\w*:(\w+))\s+(?:rdf:type|a)\s+(?:rdfs|owl):Class\b")
_RDF_PROPERTY_RE = re.compile(r"^(\w*:\w+)\s+(?:rdf:type|a)\s+(?:rdf:Property|owl:\w*Property)\b")
_REQUIREMENT_RE = re.compile(r"^\s*[A-Za-z0-9][\w.\-\[\]]*")

# No two optional whitespace runs are adjacent in these patterns (``\s*\*?\s*``, ``\s*;?\s*`` and
# ``\s*(?::\s*[^=]+?)?\s*=`` were polynomial on a long blank run): ``\s*(?:\*\s*)?`` instead.
_JS_DECL_RE = re.compile(
    r"^(?P<export>export\s+(?P<default>default\s+)?)?(?:declare\s+)?"
    r"(?:(?P<async>async\s+)?function\s*(?:\*\s*)?(?:(?P<fn>[A-Za-z_$][\w$]*)\s*)?[(<]"
    r"|(?:abstract\s+)?class\s+(?P<cls>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*(?::(?P<type>[^=]+))?=\s*(?P<rhs>.*)"
    r"|(?P<kw>interface|enum)\s+(?P<iface>[A-Za-z_$][\w$]*))"
)
_JS_DEFAULT_EXPORT_RE = re.compile(r"^export\s+default\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?:;\s*)?$")
_JS_FUNCTION_RHS_RE = re.compile(
    r"^(?:async\s*)?(?:\(|function\b|[A-Za-z_$][\w$]*\s*=>|(?:React\.)?(?:memo|forwardRef|lazy)\s*\(|styled[.(])"
)
_JS_STUB_RE = re.compile(r"throw\s+new\s+Error\(\s*['\"`][^'\"`]*not\s+implemented", re.I)

_JAVA_TYPE_RE = re.compile(
    r"^(?P<ind>[ \t]*)(?:@\w+(?:\([^)]*\))?\s+)*"
    r"(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed|strictfp)\s+)*"
    r"(?P<kind>class|interface|enum|record)\s+(?P<name>\w+)"
)
_JAVA_METHOD_RE = re.compile(
    r"^(?P<ind>[ \t]+)(?:(?:public|protected|private|static|final|abstract|synchronized|native|default|strictfp)\s+)*"
    r"(?:<[^>]*>\s+)?(?:(?P<ret>[\w.$]+(?:\s*<[^(){};=]*>)?(?:\[\])*)\s+)?(?P<name>\w+)\s*\((?P<rest>[^;]*)$"
)
# The type is space-separated tokens (``Map<String, Integer>``) and never starts with a modifier: a
# class holding the space right before ``\s+``, and modifiers readable as either, made it quadratic.
_JAVA_FIELD_RE = re.compile(
    r"^[ \t]+(?:(?:private|protected|public|static|final|transient|volatile)\s+)+"
    r"(?!(?:private|protected|public|static|final|transient|volatile)\s)"
    r"[\w.$<>\[\],?]+(?: +[\w.$<>\[\],?]+)*\s+(\w+)\s*(?:=[^;]*)?;\s*$"
)
_JAVA_NOT_METHODS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "new", "else", "do", "try", "synchronized", "throw",
    "super", "this", "assert", "case",
})
_JAVA_ACCESSOR_RE = re.compile(r"^(?:get|set|is|has|add|remove)[A-Z_]\w*$")
_JAVA_STUB_RE = re.compile(r"throw\s+new\s+UnsupportedOperationException\b")
# The Java classes generator writes every constructor as `    public <Class> (` (space before the
# parenthesis) and no package line: together, the provenance its per-class files carry.
_JAVA_GENERATED_CTOR_RE = re.compile(r"^    public (\w+) \(")
_JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+[\w.]+\s*;")
_JS_NOT_CLASS_NAMES = frozenset({"extends", "implements"})

_BPMN_TAGS = frozenset({
    "process", "startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent", "boundaryEvent",
    "task", "userTask", "serviceTask", "scriptTask", "sendTask", "receiveTask", "manualTask",
    "businessRuleTask", "callActivity", "subProcess", "transaction", "exclusiveGateway", "inclusiveGateway",
    "parallelGateway", "complexGateway", "eventBasedGateway", "sequenceFlow", "lane", "participant",
    "dataObject", "dataStoreReference", "messageFlow",
})
_BPMN_CONDITIONAL_GATEWAYS = frozenset({"exclusiveGateway", "inclusiveGateway", "complexGateway"})


def extract(index: SourceIndex, class_names: list[str], claimed_roots: list[str]) -> list[ScaffoldMap]:
    """One ``generic`` :class:`ScaffoldMap` per group of files no other extractor claimed.

    A file under a non-empty claimed root is skipped. When some extractor
    claimed the workspace root (``''``), that scaffold owns the whole tree:
    top-level files are skipped and a folder is mapped only when it holds a
    specialist scaffold (agent, circuit, network, BPMN, per-class output ...),
    so a root backend's ``routers/`` or a root frontend's ``src/`` is never
    mapped twice. Read-only and deterministic; a group that cannot be mapped
    is skipped.
    """
    names: list[str] = []
    for name in class_names or []:
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name)
    claimed = [r for r in claimed_roots or [] if isinstance(r, str)]
    whole_workspace = any(not r.strip("/") for r in claimed)
    groups = _group_files(index, claimed)
    used_ids: set[str] = set()
    maps: list[ScaffoldMap] = []
    for root in sorted(groups):
        try:
            mapper = _GroupMapper(index, root, groups[root], names, used_ids)
            built = mapper.build()
            if built is not None and whole_workspace and not mapper.specialist():
                continue
        except Exception:  # one unexpected file layout must not hide the other groups
            logger.debug("Scaffold map: generic group %r not mapped", root, exc_info=True)
            continue
        if built is not None:
            maps.append(built)
    return maps


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _mappable(rel: str) -> bool:
    """Code and known text files; never lock files, bundles, dotfiles or binaries."""
    parts = rel.split("/")
    if any(part.startswith(".") for part in parts):
        return False
    name = parts[-1]
    if name in _LOCK_FILES or name.endswith(_SKIP_SUFFIXES):
        return False
    if name in _NAMED_ROLES or name.endswith(_CODE_SUFFIXES):
        return True
    return posixpath.splitext(name)[1].lower() in _TEXT_ROLES


def _group_files(index: SourceIndex, claimed_roots: list[str]) -> dict[str, list[str]]:
    """Unclaimed mappable files keyed by their group root.

    The group root is the shallowest directory of the file that neither is nor
    contains a claimed root (``backend/`` claimed: ``baf/agent.py`` -> ``baf``;
    ``web_app/backend`` claimed: ``web_app/agents/x/a.py`` -> ``web_app/agents``,
    ``web_app/README.md`` -> ``web_app``).
    """
    raw = [r for r in claimed_roots if isinstance(r, str)]
    whole_workspace = any(not r.strip("/") for r in raw)
    claimed = sorted({r.strip("/") for r in raw if r.strip("/")})
    groups: dict[str, list[str]] = {}
    for rel in index.files:
        if not _mappable(rel) or any(rel.startswith(root + "/") for root in claimed):
            continue
        dirs = rel.split("/")[:-1]
        if not dirs:
            if whole_workspace:
                continue
            key = ""
        else:
            key = "/".join(dirs)
            for depth in range(1, len(dirs) + 1):
                prefix = "/".join(dirs[:depth])
                if not any(root == prefix or root.startswith(prefix + "/") for root in claimed):
                    key = prefix
                    break
        groups.setdefault(key, []).append(rel)
    return groups


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _names(items: list[str], limit: int = _MAX_NAMES) -> str:
    unique = list(dict.fromkeys(i for i in items if i))
    shown = ", ".join(unique[:limit])
    return shown + (f" (+{len(unique) - limit})" if len(unique) > limit else "")


def _plural(count: int, word: str) -> str:
    if count == 1:
        return f"{count} {word}"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return f"{count} {word[:-1]}ies"
    return f"{count} {word}" + ("es" if word.endswith(("s", "x", "ch", "sh")) else "s")


def _symbol(anchor: Anchor) -> str:
    """The symbol of a ``generic:<what>:<file>:<symbol>`` id, without a ``~N`` repeat suffix ('' for files)."""
    _, _, rest = anchor.id.partition(f":{anchor.file}:")
    return re.sub(r"~\d+$", "", rest)


def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _decorator_name(node: ast.AST) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    return target.id if isinstance(target, ast.Name) else ""


def _is_property(fn: ast.AST) -> bool:
    for deco in getattr(fn, "decorator_list", []):
        if isinstance(deco, ast.Name) and deco.id in ("property", "cached_property"):
            return True
        if isinstance(deco, ast.Attribute) and deco.attr in ("setter", "getter", "deleter"):
            return True
    return False


def _is_main_guard(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and "__main__" in _unparse(node.test) and "__name__" in _unparse(node.test)


def _call_name(call: ast.AST) -> tuple[Optional[str], str]:
    """(owner variable, called name) of ``owner.name(...)`` / ``name(...)``."""
    if not isinstance(call, ast.Call):
        return None, ""
    func = call.func
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        return owner, func.attr
    return None, func.id if isinstance(func, ast.Name) else ""


def _call_chain(call: ast.AST) -> tuple[Optional[str], list[ast.Call]]:
    """``x.a(..).b(..).c(..)`` -> ('x', [call a, call b, call c])."""
    chain: list[ast.Call] = []
    node = call
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        chain.append(node)
        node = node.func.value
    chain.reverse()
    return (node.id if isinstance(node, ast.Name) else None), chain


def _import_modules(tree: ast.Module) -> list[str]:
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.append(node.module)
    return modules


def _stable_symbol(name: str) -> str:
    """A symbol for an id; Qiskit function-gate defs carry a per-run ``id()`` suffix that is dropped."""
    owner, _, short = name.rpartition(".")
    match = _QISKIT_GATE_RE.match(short)
    if match:
        short = match.group(1)
    return f"{owner}.{short}" if owner else short


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _block_end(lines: list[str], start: int, python: bool = False) -> int:
    """Last line (1-based) of the indentation block opened on line ``start``.

    The block ends before the next non-blank line indented no deeper than the
    opening line; a closing bracket there belongs to the block (``}``, ``};``,
    ``);``) unless it reopens one (``}: Props) {``). For Python, closing
    brackets are continuation (``):`` of a multi-line signature).
    """
    if not 1 <= start <= len(lines):
        return start
    base = _indent(lines[start - 1])
    last = start
    for number in range(start + 1, len(lines) + 1):
        line = lines[number - 1]
        stripped = line.strip()
        if not stripped:
            continue
        if _indent(line) <= base:
            if stripped[0] in "})]":
                # `}: Props) {` closes a multi-line signature and opens the body: continuation.
                if python or stripped.endswith(("{", "(", "[", "=>")):
                    last = number
                    continue
                return number
            if stripped.startswith("{") and not python:
                last = number      # Allman-style opening brace
                continue
            return last
        last = number
    return last


def _stub_status(fn: ast.AST, lines: list[str]) -> Optional[str]:
    """Why ``fn`` is a generated placeholder still to implement, or None."""
    if {_decorator_name(d) for d in fn.decorator_list} & _SKIP_STUB_DECORATORS:
        return None
    last = fn.body[-1] if fn.body else None
    reason = None
    # _placeholder_reason walks the whole body; only a body ending in a raise, a returned call or a
    # try (or one that is nothing but pass/...) can be a placeholder, so skip the walk otherwise.
    if len(fn.body) <= 2 or isinstance(last, (ast.Raise, ast.Try)) or (
            isinstance(last, ast.Return) and isinstance(last.value, ast.Call)):
        try:
            reason = _placeholder_reason(fn)
        except Exception:
            reason = None
    if reason == "an empty action body":
        if fn.name.startswith("__") and fn.name.endswith("__"):
            return None
        return "stub: empty body"
    if reason:
        return f"stub: {reason}"
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    if (len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str) and "TODO" in body[0].value.value):
        return "stub: returns a TODO placeholder"
    for number in range(fn.lineno + 1, min(fn.end_lineno or fn.lineno, len(lines)) + 1):
        match = _TODO_COMMENT_RE.match(lines[number - 1])
        if match:
            return "TODO: " + _clip(match.group(1).strip() or "placeholder", 60)
    return None


@dataclass
class _Item:
    """An anchor plus what the caps and the summaries need to know about it."""

    anchor: Anchor
    priority: int
    order: int
    symbol: str = ""     # the name in the id, without a repeat suffix


@dataclass
class _PyFile:
    rel: str
    tree: ast.Module
    lines: list[str]
    flavour: str
    classes: list[ast.ClassDef] = field(default_factory=list)


@dataclass
class _Agent:
    """What the recipes need to know about one BESSER agent file."""

    rel: str
    intents: bool
    states: bool
    llm: bool


@dataclass
class _BafState:
    name: str
    decl: ast.stmt
    reasoning: bool = False
    initial: bool = False
    body: Optional[tuple[str, ast.stmt]] = None
    fallback: Optional[tuple[str, ast.stmt]] = None
    destinations: list[str] = field(default_factory=list)
    marker: int = 0
    block_end: int = 0


# ---------------------------------------------------------------------------
# One group of files
# ---------------------------------------------------------------------------

class _GroupMapper:
    """Maps one group of unclaimed files into a ScaffoldMap."""

    def __init__(self, index: SourceIndex, root: str, files: list[str], class_names: list[str],
                 used_ids: set[str]) -> None:
        self.index = index
        self.root = root
        self.files = sorted(files)
        self.class_names = set(class_names)
        self.class_lookup = {name.lower(): name for name in sorted(class_names)}
        self.used_ids = used_ids
        self.items: dict[str, list[_Item]] = {}       # file -> anchors (without its file anchor)
        self.file_anchors: dict[str, Anchor] = {}
        self.order = 0
        self.flavours: list[str] = []
        self.labels: dict[tuple[str, str], str] = {}  # (file, qualified def name) -> label
        self.summaries: list[tuple[str, str]] = []    # (file, specialised one-liner), first in the summary
        self.code_summaries: dict[str, str] = {}
        self.other_files: dict[str, str] = {}        # file -> role
        self.model_class_files: list[str] = []
        self.warnings: list[str] = []
        self.recipes: list[Recipe] = []
        self.notes: list[str] = []
        self.found_classes: dict[str, set[str]] = {}  # flavour -> model classes a bannered file defines
        self.generated: set[str] = set()              # flavours whose files carry generator provenance
        self.py_files: dict[str, _PyFile] = {}
        self.tool_files: set[str] = set()
        self.first_file: dict[str, str] = {}          # flavour -> first file showing it
        self.plain: dict[str, list[str]] = {}         # file -> plain function/method names (not anchored)
        self.agents: list[_Agent] = []

    # -- ids, quotes, anchors ---------------------------------------------

    def _id(self, what: str, rel: str, name: str = "") -> str:
        base = f"{FAMILY}:{what}:{rel}" + (f":{name}" if name else "")
        anchor_id, n = base, 2
        while anchor_id in self.used_ids:
            anchor_id = f"{base}~{n}"
            n += 1
        self.used_ids.add(anchor_id)
        return anchor_id

    def _line_quote(self, rel: str, number: int) -> str:
        """Line ``number`` when it is unique in the file and short enough to anchor an edit on, else ''."""
        quote = self.index.unique_quote(rel, number)
        return quote if len(quote) <= _MAX_QUOTE_CHARS else ""

    def _first_quote(self, rel: str, start: int, end: int) -> str:
        for number in range(max(1, start), min(end, len(self.index.lines(rel))) + 1):
            quote = self._line_quote(rel, number)
            if quote:
                return quote
        return ""

    def _quote(self, rel: str, start: int, end: int, *preferred: int) -> str:
        for number in preferred:
            if start <= number <= end:
                quote = self._line_quote(rel, number)
                if quote:
                    return quote
        return self._first_quote(rel, start, end)

    def _add(self, what: str, rel: str, name: str, kind: str, label: str, start: int, end: int, *,
             prefer: tuple[int, ...] = (), quote: Optional[str] = None, hint: str = "",
             entity: Optional[str] = None, status: Optional[str] = None, priority: int = _P_FUNCTION) -> Anchor:
        end = max(start, end)
        anchor = Anchor(
            id=self._id(what, rel, name), file=rel, kind=kind, label=_clip(label, 110),
            start_line=start, end_line=end,
            quote=self._quote(rel, start, end, *prefer) if quote is None else quote,
            hint=hint, entity=entity, status=status,
        )
        self.items.setdefault(rel, []).append(_Item(anchor, priority, self.order, name))
        self.order += 1
        return anchor

    def _after_statement(self, rel: str, start: int, end: int) -> tuple[str, str]:
        """(quote, wording) for an insertion right after the statement spanning ``start..end``.

        The statement's last line when it is unique ("AFTER this line"); otherwise
        its first line, worded as the whole statement, because inserting after the
        opening line of a multi-line call would land inside its arguments.
        """
        quote = self._line_quote(rel, end)
        if quote:
            return quote, "AFTER this line"
        if end == start:
            return "", "AFTER end_line"
        return (self._line_quote(rel, start),
                "AFTER this statement, which ends at end_line (not after its first line)")

    def _flavour(self, flavour: str, rel: str) -> None:
        if flavour not in self.flavours:
            self.flavours.append(flavour)
        self.first_file.setdefault(flavour, rel)

    def _warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    def _entity(self, name: str, flavour: str = "") -> Optional[str]:
        """The model class ``name`` stands for, matched case-insensitively (``BookCreate`` too in Pydantic)."""
        found = self.class_lookup.get(name.lower())
        if found is None and flavour == "pydantic":
            for suffix in _SCHEMA_SUFFIXES:
                if name.endswith(suffix) and len(name) > len(suffix):
                    found = self.class_lookup.get(name[: -len(suffix)].lower())
                    if found:
                        break
        return found

    def specialist(self) -> bool:
        """Whether this group holds a scaffold no framework extractor maps (kept next to a root scaffold)."""
        if set(self.flavours) & _SPECIALIST_FLAVOURS:
            return True
        folder = _SPECIALIST_FOLDERS.get(posixpath.basename(self.root))
        return folder is not None and folder in self.flavours

    # -- driver -----------------------------------------------------------

    def build(self) -> Optional[ScaffoldMap]:
        for rel in self.files:
            if rel.endswith(_PY_SUFFIXES):
                try:
                    self._prepare_python(rel)
                except Exception:  # e.g. RecursionError on a pathological expression: mapped as unparsed
                    logger.debug("Scaffold map: %s not parsed", rel, exc_info=True)
                    self.py_files.pop(rel, None)
        for rel in self.files:
            try:
                if rel.endswith(_PY_SUFFIXES):
                    self._python(rel)
                elif rel.endswith(_SCRIPT_SUFFIXES):
                    self._script(rel)
                elif rel.endswith(_JAVA_SUFFIXES):
                    self._java(rel)
                elif rel.endswith(".bpmn"):
                    self._bpmn(rel)
                else:
                    self._text_file(rel)
            except Exception:  # an unexpected shape loses that file only
                logger.debug("Scaffold map: %s not mapped", rel, exc_info=True)
        if not self.file_anchors:
            return None
        return self._assemble()

    # -- Python -----------------------------------------------------------

    def _prepare_python(self, rel: str) -> None:
        tree = self.index.python_ast(rel)
        if not isinstance(tree, ast.Module):
            return
        lines = self.index.lines(rel)
        flavour = _python_flavour(tree, self.index.text(rel) or "")
        pf = _PyFile(rel, tree, lines, flavour, [n for n in tree.body if isinstance(n, ast.ClassDef)])
        self.py_files[rel] = pf
        if flavour == "baf":
            for node in ast.walk(tree):
                if _call_name(node)[1] == "load_tools":
                    for const in ast.walk(node):
                        if isinstance(const, ast.Constant) and isinstance(const.value, str) \
                                and const.value.endswith(".py"):
                            self.tool_files.add(posixpath.normpath(
                                posixpath.join(posixpath.dirname(rel), const.value)))

    def _python(self, rel: str) -> None:
        pf = self.py_files.get(rel)
        if pf is None:
            if self.index.text(rel) is not None:
                self._python_fallback(rel)
            return
        if pf.flavour not in ("python", "other"):
            self._flavour(pf.flavour, rel)
        marker = _PER_CLASS_MARKERS.get(pf.flavour)
        if marker is not None and marker in (self.index.text(rel) or ""):
            self.generated.add(pf.flavour)
            self.found_classes.setdefault(pf.flavour, set()).update(
                found for c in pf.classes if (found := self._entity(c.name, pf.flavour)))
        enrich = {"baf": self._baf, "qiskit": self._qiskit, "pytorch": self._nn, "tensorflow": self._nn,
                  "sqlalchemy": self._sqlalchemy}.get(pf.flavour)
        if rel in self.tool_files:
            for node in pf.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.labels[(rel, node.name)] = f"agent tool {node.name}()"
        if enrich is not None:
            try:
                enrich(pf)
            except Exception:  # the plain definitions below still map the file
                logger.debug("Scaffold map: %s enrichment failed for %s", pf.flavour, rel, exc_info=True)
        counts = {"classes": 0, "functions": 0}
        for node in pf.tree.body:
            if isinstance(node, ast.ClassDef):
                self._py_class(pf, node)
                counts["classes"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._py_function(pf, node, owner=None, entity=None)
                counts["functions"] += 1
            elif _is_main_guard(node):
                self._add("main", rel, "", KIND_BLOCK,
                          self.labels.get((rel, "__main__"), "`if __name__ == '__main__'` block"),
                          node.lineno, node.end_lineno or node.lineno, prefer=(node.lineno,), priority=_P_BLOCK)
        role = _PY_ROLES.get(pf.flavour, "Python module")
        if rel in self.tool_files:
            role = "agent tools (agent.load_tools)"
        detail = ", ".join(_plural(counts[k], k[:-2] if k == "classes" else k[:-1])
                           for k in ("classes", "functions") if counts[k])
        plain = [name for name in self.plain.get(rel, []) if "." not in name]
        if plain:
            detail += f"; plain: {_names(plain, 4)}"
        self._file_anchor(rel, f"{role}" + (f": {detail}" if detail else ""))
        self._code_summary(rel)

    def _python_fallback(self, rel: str) -> None:
        """A file that does not parse: top-level defs by line pattern, so it can still be located.

        The parse error itself is listed once, in the workspace's broken files;
        repeating it here (with its line number) would put a line into the overview.
        """
        lines = self.index.lines(rel)
        for number, line in enumerate(lines, 1):
            match = _PY_DEF_RE.match(line)
            if not match:
                continue
            start = number
            while start > 1 and lines[start - 2].startswith("@"):
                start -= 1
            is_class = line.startswith("class")
            name = _stable_symbol(match.group(1))
            gate = _QISKIT_GATE_RE.match(match.group(1))
            label = (f"function gate {gate.group(1)[len('_function_gate_'):]}" if gate
                     else f"{'class' if is_class else 'function'} {name}")
            self._add("class" if is_class else "function", rel, name,
                      KIND_CLASS if is_class else KIND_FUNCTION, f"{label} (file does not parse)",
                      start, _block_end(lines, number, python=True), prefer=(number,),
                      entity=self._entity(name) if is_class else None,
                      priority=_P_KEY if is_class else _P_FUNCTION)
        if rel in self.index.syntax_errors:
            self._file_anchor(rel, "Python module — does not parse (see the broken files)")
            self.code_summaries[rel] = f"{rel}: does not parse; fix it first"
        else:   # e.g. nested too deeply for the parser: not a syntax error, so not in the broken files
            self._file_anchor(rel, "Python module — too complex to parse, not analysed")
            self.code_summaries[rel] = f"{rel}: too complex to parse, not analysed; read it directly"

    def _py_class(self, pf: _PyFile, node: ast.ClassDef) -> None:
        rel = pf.rel
        entity = self._entity(node.name, pf.flavour)
        bases = ", ".join(_unparse(b) for b in node.bases)
        label = self.labels.get((rel, node.name)) or f"class {node.name}" + (f"({_clip(bases, 40)})" if bases else "")
        start, end = node_start(node), node.end_lineno or node.lineno
        self._add("class", rel, node.name, KIND_CLASS, label, start, end, prefer=(node.lineno,),
                  entity=entity, priority=_P_KEY)
        protocol = any("Protocol" in _unparse(b) for b in node.bases)
        run: list[ast.AST] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_property(item):
                run.append(item)
                continue
            self._py_properties(pf, node, run, entity)
            run = []
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._py_function(pf, item, owner=node.name, entity=entity, protocol=protocol)
        self._py_properties(pf, node, run, entity)

    def _py_properties(self, pf: _PyFile, cls: ast.ClassDef, run: list[ast.AST], entity: Optional[str]) -> None:
        """A contiguous run of ``@property`` / ``@x.setter`` defs, folded into one block."""
        if not run:
            return
        names = [fn.name for fn in run]
        start, end = node_start(run[0]), run[-1].end_lineno or run[-1].lineno
        self._add("properties", pf.rel, cls.name, KIND_BLOCK,
                  f"properties of {cls.name}: {_names(names)}", start, end,
                  prefer=tuple(fn.lineno for fn in run), entity=entity, priority=_P_ACCESSORS,
                  hint="a new attribute needs its @property / @<name>.setter pair here and an __init__ parameter")

    def _py_function(self, pf: _PyFile, fn: ast.AST, *, owner: Optional[str], entity: Optional[str],
                     protocol: bool = False) -> None:
        rel = pf.rel
        qualified = f"{owner}.{fn.name}" if owner else fn.name
        status = None if protocol else _stub_status(fn, pf.lines)
        prefix = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
        special = self.labels.get((rel, qualified))
        if not (status or special or entity or pf.flavour in _BEHAVIOUR_FLAVOURS):
            # A plain def: named in the file label and the summary, not anchored (see the module docstring).
            self.plain.setdefault(rel, []).append(_stable_symbol(qualified))
            return
        label = special or (
            f"{prefix}method {qualified}()" if owner else f"{prefix}function {fn.name}()")
        if status:
            priority = _P_STUB
        else:
            priority = _P_METHOD if owner else _P_FUNCTION
        self._add("method" if owner else "function", rel, _stable_symbol(qualified),
                  KIND_STUB if status else KIND_FUNCTION, label,
                  node_start(fn), fn.end_lineno or fn.lineno, prefer=(fn.lineno,), entity=entity,
                  status=status, priority=priority,
                  hint="stub: implement the body in place and keep the signature" if status else "")

    # -- BESSER agent (BAF) -------------------------------------------------

    def _baf(self, pf: _PyFile) -> None:
        rel, tree, lines = pf.rel, pf.tree, pf.lines
        agent_var, agent_name = _agent_var(tree)
        states: dict[str, _BafState] = {}
        intents: dict[str, list] = {}       # name -> [start, end, phrases, params]
        entities: dict[str, list] = {}      # name -> [start, end, entries]
        llms: list[tuple[str, str, ast.stmt]] = []
        default_llm: Optional[ast.stmt] = None
        defined = _defined_names(tree)
        conditions: dict[str, list[str]] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target = stmt.targets[0].id
                owner, called = _call_name(stmt.value)
                end = stmt.end_lineno or stmt.lineno
                if (owner == agent_var and called == "new_state") or called == "new_reasoning_state":
                    states[target] = _BafState(
                        target, stmt, reasoning=called == "new_reasoning_state",
                        initial=any(k.arg == "initial" and isinstance(k.value, ast.Constant) and k.value.value is True
                                    for k in stmt.value.keywords))
                elif owner == agent_var and called == "new_intent":
                    phrases = stmt.value.args[1] if len(stmt.value.args) > 1 else None
                    intents[target] = [stmt.lineno, end,
                                       len(phrases.elts) if isinstance(phrases, ast.List) else 0, []]
                elif owner == agent_var and called == "new_entity":
                    entries = next((k.value for k in stmt.value.keywords if k.arg == "entries"), None)
                    entities[target] = [stmt.lineno, end, len(entries.keys) if isinstance(entries, ast.Dict) else 0]
                elif called.startswith("LLM") and not called.endswith("Configuration"):
                    llms.append((target, called, stmt))
                elif target == "default_llm":
                    default_llm = stmt
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._baf_call(rel, stmt, agent_var, states, intents, conditions)
        self._flavour("baf", rel)
        # Banner sections, per-state markers and the run marker.
        banners = [(m.group(1), n) for n, line in enumerate(lines, 1)
                   if (m := _BANNER_RE.match(line)) and n > 1 and lines[n - 2].startswith("####")]
        run_marker = next((n for n, line in enumerate(lines, 1) if line.strip() == "# RUN APPLICATION"), 0)
        states_banner = next((n for name, n in banners if name == "STATES"), 0)
        boundaries = sorted({n - 1 for _, n in banners} | ({run_marker} if run_marker else set()))
        markers: list[tuple[int, str]] = []
        for number, line in enumerate(lines, 1):
            match = _STATE_MARKER_RE.match(line)
            if match and match.group(1) in states and number > states_banner:
                markers.append((number, match.group(1)))
        stops = sorted(set(boundaries) | {n for n, _ in markers})

        def _section_end(start: int, limits: list[int]) -> int:
            end = next((b for b in limits if b > start), len(lines) + 1) - 1
            while end > start and not lines[end - 1].strip():
                end -= 1
            return end

        for name, number in banners:
            end = _section_end(number, boundaries)
            counts = {"INTENTS": _plural(len(intents), "intent"), "ENTITIES": _plural(len(entities), "entity"),
                      "STATES": _plural(len(states), "state"),
                      "CUSTOM CONDITIONS": _plural(len(conditions), "condition")}.get(name, "")
            self._add("section", rel, name.lower().replace(" ", "_"), KIND_BLOCK,
                      f"{name} section" + (f" ({counts})" if counts else ""), number - 1, end, prefer=(number,),
                      priority=_P_BLOCK)
        if llms:
            first = llms[0][2]
            last = default_llm if default_llm is not None and default_llm.lineno > first.lineno else llms[-1][2]
            label = "LLMs: " + _names([f"{var} ({cls})" for var, cls, _ in llms], 3)
            if default_llm is not None:
                label += f"; default_llm = {_unparse(default_llm.value)}"
            self._add("llms", rel, "", KIND_BLOCK, label, first.lineno, last.end_lineno or last.lineno,
                      prefer=(first.lineno,), priority=_P_BLOCK,
                      hint="change provider/model here; API keys belong in config.yaml")
        for name, (start, end, entries) in entities.items():
            self._add("agent_entity", rel, name, KIND_BLOCK, f"agent entity {name} ({_plural(entries, 'entry')})",
                      start, end, prefer=(start,), priority=_P_MINOR)
        for name, (start, end, phrases, params) in intents.items():
            label = f"intent {name} ({_plural(phrases, 'training phrase')}"
            label += f"; params {_names(params, 3)})" if params else ")"
            self._add("intent", rel, name, KIND_BLOCK, label, start, end, prefer=(start,), priority=_P_MINOR)
        if intents:
            start, end = list(intents.values())[-1][:2]
            quote, where = self._after_statement(rel, start, end)
            self._add("intents_end", rel, "", KIND_INSERT, "after the last intent", start, end,
                      quote=quote, priority=_P_KEY,
                      hint=f"add a new `<name> = agent.new_intent('<name>', [...], description=...)` {where}; "
                           "parameters need an entity declared above")
        # States: per-state blocks, bodies, insertion points.
        for number, name in markers:
            state = states[name]
            if not state.marker:
                state.marker, state.block_end = number, _section_end(number, stops)
        for name, state in states.items():
            self._baf_state(rel, state, defined)
        if states:
            last = list(states.values())[-1].decl
            quote, where = self._after_statement(rel, last.lineno, last.end_lineno or last.lineno)
            self._add("state_decls_end", rel, "", KIND_INSERT, "after the last state declaration",
                      last.lineno, last.end_lineno or last.lineno, quote=quote, priority=_P_KEY,
                      hint=f"declare a new state {where}: `<name> = agent.new_state('<name>')`; "
                           "only one state may be initial=True")
        if run_marker:
            quote = self._line_quote(rel, run_marker)
            if quote:
                self._add("state_blocks_end", rel, "", KIND_INSERT, "end of the per-state blocks",
                          run_marker, run_marker, quote=quote, priority=_P_KEY,
                          hint="add a new state's block (`# <state>` comment, body def, set_body, transitions) "
                               "BEFORE this line")
        for cond, users in conditions.items():
            self.labels[(rel, cond)] = f"custom condition {cond} (guards {_names(users, 3)})"
        # Warnings found in this code.
        silent = [s.name for s in states.values() if not s.reasoning and s.body is None]
        if silent:
            self._warn(f"{rel}: state{'s' if len(silent) > 1 else ''} {_names(silent, 4)} "
                       f"{'have' if len(silent) > 1 else 'has'} no body (no set_body), so "
                       f"{'they reply' if len(silent) > 1 else 'it replies'} nothing when entered.")
        for call in ast.walk(tree):
            owner, called = _call_name(call)
            if owner == agent_var and called == "load_properties" and call.args \
                    and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str) \
                    and not posixpath.isabs(call.args[0].value):
                if not self._dockerised(rel):
                    self._warn(f"{rel} loads '{call.args[0].value}' relative to the working directory: start the "
                               f"agent from {posixpath.dirname(rel) or 'the workspace root'}/ or load "
                               f"os.path.join(_HERE, '{call.args[0].value}').")
                break
        no_llm = default_llm is not None and isinstance(default_llm.value, ast.Constant) \
            and default_llm.value.value is None
        summary = f"{rel}: agent {agent_name or agent_var}"
        parts = [f"states {_names(list(states))}" if states else "no states",
                 f"intents {_names(list(intents))}" if intents else "",
                 f"entities {_names(list(entities), 3)}" if entities else "",
                 f"LLM {_names([v for v, _, _ in llms], 2)}" if llms else
                 ("no LLM (default_llm = None)" if no_llm else "")]
        self.summaries.append((rel, summary + " — " + "; ".join(p for p in parts if p)))
        self.agents.append(_Agent(rel, bool(intents), bool(states), bool(llms) and not no_llm))

    def _dockerised(self, rel: str) -> bool:
        """Whether a Dockerfile next to the agent copies its folder and runs it from there (a web-app agent)."""
        dockerfile = posixpath.join(posixpath.dirname(rel), "Dockerfile")
        # A Dockerfile the index left out (its caps) still counts: the warning it silences would be false.
        text = self.index.text(dockerfile) if self.index.exists(dockerfile) else None
        if not text:
            return False
        name = re.escape(posixpath.basename(rel))
        # [ \t]*, not \s*: under re.M a leading \s* crosses newlines, quadratic on a run of blank lines.
        runs = re.search(rf"^[ \t]*(?:CMD|ENTRYPOINT)\b.*[\"'\s/]{name}(?:[\"'\s\]]|$)", text, re.M)
        return bool(runs) and re.search(r"^[ \t]*COPY\s+\.\s+\S+", text, re.M) is not None

    def _baf_call(self, rel: str, stmt: ast.Expr, agent_var: str, states: dict[str, _BafState],
                  intents: dict[str, list], conditions: dict[str, list[str]]) -> None:
        root, chain = _call_chain(stmt.value)
        if not chain:
            return
        head = chain[0]
        end = stmt.end_lineno or stmt.lineno
        if root in intents and head.func.attr == "parameter":
            intents[root][1] = max(intents[root][1], end)
            if head.args and isinstance(head.args[0], ast.Constant):
                intents[root][3].append(str(head.args[0].value))
            return
        if root not in states:
            return
        state = states[root]
        if head.func.attr in ("set_body", "set_fallback_body") and head.args and isinstance(head.args[0], ast.Name):
            fn = head.args[0].id
            if head.func.attr == "set_body":
                state.body = (fn, stmt)
                self.labels[(rel, fn)] = f"body of state {root}"
            else:
                state.fallback = (fn, stmt)
                self.labels[(rel, fn)] = f"fallback body of state {root}"
            return
        if chain[-1].func.attr != "go_to":
            return
        dest = chain[-1].args[0] if chain[-1].args else None
        if isinstance(dest, ast.Name):
            state.destinations.append(dest.id)
            if dest.id not in states:
                self._warn(f"{rel}: {root}.go_to({dest.id}) targets {dest.id}, which is not a declared state.")
        for call in chain:
            if call.func.attr in ("with_condition", "when_condition"):
                for arg in call.args:
                    if isinstance(arg, ast.Name):
                        conditions.setdefault(arg.id, []).append(root)
        if head.func.attr == "when_variable_matches_operation" and len(head.args) >= 3:
            op, target = head.args[1], head.args[2]
            if isinstance(op, ast.Attribute) and op.attr in ("gt", "lt", "ge", "le") \
                    and isinstance(target, ast.Constant) and isinstance(target.value, str) \
                    and re.fullmatch(r"-?\d+(?:\.\d+)?", target.value.strip()):
                variable = _unparse(head.args[0])
                self._warn(f"{rel}: {root}.when_variable_matches_operation compares {variable} with the STRING "
                           f"'{target.value}' (operator.{op.attr}); use the number {target.value.strip()}.")

    def _baf_state(self, rel: str, state: _BafState, defined: set[str]) -> None:
        label = f"state {state.name}" + (" (initial)" if state.initial else "")
        if state.reasoning:
            label += " [reasoning]"
        elif state.body:
            label += f": body {state.body[0]}"
        else:
            label += ": no body"
        if state.fallback:
            label += f"; fallback {state.fallback[0]}"
        if state.destinations:
            label += f"; → {_names(state.destinations, 4)}"
        decl_end = state.decl.end_lineno or state.decl.lineno
        if state.marker:
            start, end, prefer = state.marker, max(state.marker, state.block_end), (state.marker,)
        else:
            start, end, prefer = state.decl.lineno, decl_end, (state.decl.lineno,)
        self._add("state", rel, state.name, KIND_BLOCK, label, start, end, prefer=prefer, priority=_P_KEY,
                  hint="the state's body def, set_body/set_fallback_body and transitions; "
                       "add transitions at the end of this block")
        for slot in (state.body, state.fallback):
            if slot and slot[0] not in defined:
                fn, stmt = slot
                self._warn(f"{rel}: {_unparse(stmt.value)} names {fn}, which is not defined in the file "
                           "(NameError when the agent starts).")
        if not state.reasoning and state.body is None and state.marker:
            quote = self._line_quote(rel, state.marker)
            if quote:
                self._add("state_body", rel, state.name, KIND_INSERT, f"body slot of state {state.name}",
                          state.marker, state.marker, quote=quote, priority=_P_KEY,
                          hint=f"state {state.name} has no body: insert `def {state.name}_body(session: Session):` "
                               f"and `{state.name}.set_body({state.name}_body)` AFTER this line, before its "
                               "transitions")

    def _baf_recipes(self) -> list[Recipe]:
        """The agent recipes of every agent in the group (several in a web app's ``agents/``)."""
        if not self.agents:
            return []
        if len(self.agents) == 1:
            agent = self.agents[0].rel
            files = ""
        else:   # one id pattern for all, with the agent files named once
            agent = "<agent file>"
            files = f" Agent files: {_names([a.rel for a in self.agents], 4)}."
        llm = [a.rel for a in self.agents if a.llm]
        if len(llm) == len(self.agents):
            llm_step = "use default_llm.predict(...) for LLM answers."
        elif llm:
            llm_step = (f"default_llm.predict(...) works only in {_names(llm, 3)}; the other agents have "
                        "default_llm = None.")
        else:
            llm_step = ("no LLM is configured (default_llm = None): declare one in the LLMs section before "
                        "calling default_llm.predict(...).")
        recipes = [Recipe("Implement or change a state's behaviour", [
            f"Edit the def named in `<state>.set_body(...)` inside the state's block "
            f"(`{FAMILY}:state:{agent}:<state>`); a state without a body has a "
            f"`{FAMILY}:state_body:{agent}:<state>` insertion point.{files}",
            f"Bodies take `session: Session`; reply with session.reply(...); {llm_step}",
        ])]
        steps = ["Transitions: append `<src>.when_intent_matched(<intent>).go_to(<dest>)` at the end of the source "
                 "state's block; <dest> must be declared with the other states."]
        if any(a.states for a in self.agents):
            steps.append(f"New state: declare it after `{FAMILY}:state_decls_end:{agent}` and add its block before "
                         f"`{FAMILY}:state_blocks_end:{agent}`.")
        if any(a.intents for a in self.agents):
            steps.append(f"New intent: after `{FAMILY}:intents_end:{agent}`"
                         + (" (agents that have intents)." if not all(a.intents for a in self.agents) else "."))
        recipes.append(Recipe("Add a transition, state or intent", steps))
        return recipes

    # -- Qiskit -------------------------------------------------------------

    def _qiskit(self, pf: _PyFile) -> None:
        rel, lines = pf.rel, pf.lines
        self._flavour("qiskit", rel)
        stripped = [line.strip() for line in lines]
        ops = next((n for n, s in enumerate(stripped, 1) if s == "# Operations"), 0)
        draw = next((n for n, s in enumerate(stripped, 1) if s == "# Draw" and n > ops), 0)
        execution = next((n for n, s in enumerate(stripped, 1) if _EXECUTION_RE.match(s) and n > ops), 0)
        function_gates = 0
        for node in pf.tree.body:
            if isinstance(node, ast.FunctionDef):
                match = _QISKIT_GATE_RE.match(node.name)
                if match:
                    function_gates += 1
                    self.labels[(rel, node.name)] = f"function gate {match.group(1)[len('_function_gate_'):]}"
                elif node.name == "create_placeholder":
                    self.labels[(rel, node.name)] = "helper: placeholder instruction for unsupported gates"
        placeholders: list[str] = []
        statements = 0
        if ops:
            stop = draw or execution or len(lines) + 1
            op_lines = [n for n in range(ops + 1, stop) if stripped[n - 1]]
            # The generator separates operations by blank lines; one operation may span several
            # lines (an input value sets one qubit per line), so count the groups, not the lines.
            statements = sum(1 for i, n in enumerate(op_lines) if i == 0 or op_lines[i - 1] != n - 1)
            comment_ops = 0
            for number in op_lines:
                match = _PLACEHOLDER_GATE_RE.search(lines[number - 1])
                if match:
                    placeholders.append(match.group(1) or "gate")
                    self._add("placeholder", rel, match.group(1) or "gate", KIND_STUB,
                              f"placeholder gate {match.group(1)!r}", number, number, prefer=(number,),
                              status="stub: create_placeholder (no real gates)", priority=_P_STUB,
                              hint="stub: replace this line with the real gates (or a function gate defined in "
                                   "the '# Function Gate Definitions' section)")
                elif stripped[number - 1].startswith("#"):
                    comment_ops += 1
                    placeholders.append(stripped[number - 1][1:].strip())
                    self._add("unsupported_op", rel, f"op{comment_ops}", KIND_STUB,
                              f"comment-only operation: {stripped[number - 1][1:].strip()}", number, number,
                              prefer=(number,), status="stub: operation not generated", priority=_P_STUB,
                              hint="stub: this operation has no code yet; write it in place of the comment")
            last = op_lines[-1] if op_lines else ops
            self._add("operations", rel, "", KIND_BLOCK,
                      f"circuit operations ({_plural(statements, 'operation')}, "
                      f"{_plural(len(placeholders), 'placeholder')})", ops, last, prefer=(ops,), priority=_P_KEY,
                      hint="operations in model order, one blank-line-separated group each (a group may hold "
                           "several lines); registers are hard-coded as q[..]/c[..]")
            if draw and self._line_quote(rel, draw):
                self._add("operations_end", rel, "", KIND_INSERT, "after the last operation", draw, draw,
                          quote=self._line_quote(rel, draw), priority=_P_KEY,
                          hint="insert new gate operations BEFORE this line; keep the qc.measure lines last")
            elif op_lines:
                self._add("operations_end", rel, "", KIND_INSERT, "after the last operation", last, last,
                          prefer=(last,), priority=_P_KEY, hint="insert new gate operations AFTER this line")
        exec_label = ""
        if execution:
            end = len(lines)
            while end > execution and not stripped[end - 1]:
                end -= 1
            mode = _EXECUTION_RE.match(stripped[execution - 1]).group(1) or "run"
            shots = _SHOTS_RE.search("\n".join(lines[execution - 1:end]))
            exec_label = mode + (f", shots={shots.group(1)}" if shots else "")
            self._add("execution", rel, "", KIND_BLOCK, f"execution ({exec_label})", execution, end,
                      prefer=(execution,), priority=_P_BLOCK)
        registers = [f"{m.group(1)}[{m.group(3)}]" for line in lines if (m := _REGISTER_RE.match(line))]
        parts = [_plural(statements, "operation") + (f" ({_plural(len(placeholders), 'placeholder')})"
                                                      if placeholders else ""),
                 f"registers {_names(registers, 4)}" if registers else "", exec_label]
        self.summaries.append((rel, f"{rel}: Qiskit circuit — " + "; ".join(p for p in parts if p)))
        steps = []
        if placeholders:
            steps.append(f"Placeholder gates (`{FAMILY}:placeholder:{rel}:<name>`): replace the "
                         "`qc.append(create_placeholder(...))` line with real gates, or define a function gate in "
                         "the '# Function Gate Definitions' section and append it.")
        if ops:
            steps.append(f"New gates go before `{FAMILY}:operations_end:{rel}` (measurements stay last)."
                         + (" Function-gate def names end in a per-run number: never reference them from other "
                            "files." if function_gates else ""))
        if not any(_is_main_guard(n) for n in pf.tree.body):
            steps.append("There is no `if __name__ == '__main__':` guard, so importing the file runs the simulation: "
                         "wrap the circuit in a function before reusing or serving it.")
        if steps and not any(r.intent == "Change the circuit" for r in self.recipes):
            self.recipes.append(Recipe("Change the circuit", steps))

    # -- Neural networks (PyTorch / TensorFlow) -----------------------------

    def _nn(self, pf: _PyFile) -> None:
        rel = pf.rel
        framework = "PyTorch" if pf.flavour == "pytorch" else "TensorFlow"
        self._flavour(pf.flavour, rel)
        models: list[str] = []
        layer_total = 0
        first_layers_end = ""
        forward_id = ""
        for cls in pf.classes:
            if not any(_unparse(b) in _NN_BASES for b in cls.bases):
                continue
            models.append(cls.name)
            bases = _names([_unparse(b) for b in cls.bases], 2)
            self.labels[(rel, cls.name)] = f"{framework} model {cls.name}({bases})"
            for method in cls.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if method.name == "__init__":
                    layers = [st for st in method.body if isinstance(st, ast.Assign) and len(st.targets) == 1
                              and isinstance(st.targets[0], ast.Attribute)
                              and isinstance(st.targets[0].value, ast.Name) and st.targets[0].value.id == "self"
                              and isinstance(st.value, ast.Call)]
                    if not layers:
                        continue
                    layer_total += len(layers)
                    kinds = [f"{st.targets[0].attr} {_call_name(st.value)[1] or 'layer'}" for st in layers]
                    first, last = layers[0], layers[-1]
                    self._add("layers", rel, cls.name, KIND_BLOCK, f"layers of {cls.name}: {_names(kinds, 4)}",
                              first.lineno, last.end_lineno or last.lineno, prefer=(first.lineno,), priority=_P_KEY)
                    last_end = last.end_lineno or last.lineno
                    quote, where = self._after_statement(rel, last.lineno, last_end)
                    anchor = self._add(
                        "layers_end", rel, cls.name, KIND_INSERT, f"after the last layer of {cls.name}",
                        last.lineno, last_end, quote=quote, priority=_P_KEY,
                        hint=f"add a new layer {where} AND call it at the matching position in "
                             f"{'forward' if framework == 'PyTorch' else 'call'}(); in/out sizes must chain")
                    first_layers_end = first_layers_end or anchor.id
                elif method.name in ("forward", "call"):
                    steps = sum(1 for st in method.body if isinstance(st, ast.Assign))
                    self.labels[(rel, f"{cls.name}.{method.name}")] = (
                        f"{method.name} pass of {cls.name} ({_plural(steps, 'step')})")
                    forward_id = forward_id or f"{FAMILY}:method:{rel}:{cls.name}.{method.name}"
        sequential: list[str] = []
        for node in pf.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Call) and _unparse(node.value.func).endswith("Sequential"):
                name = node.targets[0].id
                sequential.append(name)
                args = node.value.args
                count = len(args[0].elts) if len(args) == 1 and isinstance(args[0], ast.List) else len(args)
                entries = _plural(count, "positional entry")
                self._add("sequential", rel, name, KIND_BLOCK,
                          f"{name} = {_unparse(node.value.func)}(...) with {entries} "
                          "(layer names are lost: address layers by position)",
                          node.lineno, node.end_lineno or node.lineno, prefer=(node.lineno,), priority=_P_KEY)
        training = [n.name for n in pf.tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in _NN_TRAINING_FUNCTIONS]
        for name in training:
            self.labels[(rel, name)] = f"training code: {name}()"
        if any(_is_main_guard(n) for n in pf.tree.body):
            self.labels[(rel, "__main__")] = "entry point: data, loss/optimizer, train, evaluate, save"
        parts = [f"model {_names(models, 2)} ({_plural(layer_total, 'layer')})" if models else "",
                 f"sequential {_names(sequential, 2)}" if sequential else "",
                 f"training {_names(training, 4)}" if training else "no training code"]
        self.summaries.append((rel, f"{rel}: {framework} network — " + "; ".join(p for p in parts if p)))
        steps = []
        if first_layers_end:
            steps.append(f"Add the layer at `{first_layers_end}` AND call it in `{forward_id or 'forward()'}` at the "
                         "matching position; shapes must chain.")
        if sequential:
            steps.append(f"Sequential models: insert the layer positionally in `{FAMILY}:sequential:{rel}:"
                         f"{sequential[0]}`.")
        if training:
            main = f" and `{FAMILY}:main:{rel}`" if any(_is_main_guard(n) for n in pf.tree.body) else ""
            steps.append(f"Training: edit {_names(training, 4)}{main}; dataset paths are hard-coded.")
        else:
            steps.append("There is no training code: add prepare_datasets/train/evaluate after the model and a "
                         "`__main__` block.")
        if not any(r.intent == "Change the network or its training" for r in self.recipes):
            self.recipes.append(Recipe("Change the network or its training", steps))

    # -- SQLAlchemy (standalone) --------------------------------------------

    def _sqlalchemy(self, pf: _PyFile) -> None:
        """Module-level ``<Class>.<end> = relationship(...)`` blocks, one per contiguous run of a class.

        The generator interleaves classes (an association class's ends sit
        between its partners'), so one block per class would span other
        classes' lines. A class's first run keeps the plain ``<Class>`` id (new
        relationships go there); later runs are named by their first end.
        """
        rel = pf.rel
        runs: list[tuple[str, list[ast.stmt]]] = []
        for node in pf.tree.body:
            target = None
            if isinstance(node, (ast.AnnAssign, ast.Assign)):
                target = node.target if isinstance(node, ast.AnnAssign) else (
                    node.targets[0] if len(node.targets) == 1 else None)
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and _call_name(node.value)[1] == "relationship"):
                runs.append(("", []))       # any other statement ends the current run
                continue
            if runs and runs[-1][0] == target.value.id:
                runs[-1][1].append(node)
            else:
                runs.append((target.value.id, [node]))
        seen: set[str] = set()
        for owner, statements in runs:
            if not statements:
                continue
            ends = [(s.target if isinstance(s, ast.AnnAssign) else s.targets[0]).attr for s in statements]
            start, end = statements[0].lineno, statements[-1].end_lineno or statements[-1].lineno
            if start > 1 and pf.lines[start - 2].lstrip().startswith("#--"):
                start -= 1
            name = owner if owner not in seen else f"{owner}.{ends[0]}"
            seen.add(owner)
            self._add("relationships", rel, name, KIND_BLOCK, f"relationships of {owner}: {_names(ends, 4)}",
                      start, end, prefer=(statements[0].lineno,), entity=self._entity(owner), priority=_P_BLOCK,
                      hint="relationships are assigned here, after all classes, never inside the class body")

    # -- TypeScript / JavaScript --------------------------------------------

    def _script(self, rel: str) -> None:
        if self.index.text(rel) is None:
            return
        lines = self.index.lines(rel)
        default_export = ""
        for line in lines:
            match = _JS_DEFAULT_EXPORT_RE.match(line)
            if match:
                default_export = match.group("name")
        components = functions = classes = 0
        jsx = rel.endswith((".tsx", ".jsx"))
        for number, line in enumerate(lines, 1):
            match = _JS_DECL_RE.match(line)
            if not match:
                continue
            name = match.group("fn") or match.group("cls") or match.group("var") or match.group("iface")
            if match.group("var"):
                rhs, type_ = (match.group("rhs") or "").strip(), match.group("type") or ""
                if not (_JS_FUNCTION_RHS_RE.match(rhs) or "FC" in type_ or "=>" in type_ or "Component" in type_):
                    continue
            if not name or name in _JS_NOT_CLASS_NAMES:     # `export default class extends Base {`
                name = "default"
            end = _block_end(lines, number)
            default = bool(match.group("default")) or name == default_export
            if match.group("cls") or match.group("iface"):
                word = match.group("kw") or "class"
                label, what, kind, priority = f"{word} {name}", "class", KIND_CLASS, _P_KEY
                classes += 1
            elif name[:1].isupper() and (jsx or "FC" in (match.group("type") or "")):
                label, what, kind, priority = f"component {name}", "function", KIND_FUNCTION, _P_KEY
                components += 1
            else:
                word = "hook" if re.match(r"use[A-Z]", name) else "function"
                label, what, kind, priority = f"{word} {name}()", "function", KIND_FUNCTION, _P_FUNCTION
                functions += 1
            if default:
                label += " (default export)"
            status = None
            if kind == KIND_FUNCTION:
                for inner in range(number, end + 1):
                    todo = _TODO_COMMENT_RE.match(lines[inner - 1])
                    if todo:
                        status = "TODO: " + _clip(todo.group(1).strip() or "placeholder", 60)
                        break
                    if _JS_STUB_RE.search(lines[inner - 1]):
                        status = "stub: throws 'not implemented'"
                        break
            if priority == _P_FUNCTION and not status:
                self.plain.setdefault(rel, []).append(name)     # a plain function: named, not anchored
                continue
            self._add(what, rel, name, KIND_STUB if status else kind, label, number, end, prefer=(number,),
                      status=status, priority=_P_STUB if status else priority,
                      hint="stub: implement the body in place and keep the signature" if status else "")
        detail = ", ".join(x for x in (_plural(components, "component") if components else "",
                                       _plural(functions, "function") if functions else "",
                                       _plural(classes, "type") if classes else "") if x)
        if self.plain.get(rel):
            detail += f"; plain: {_names(self.plain[rel], 4)}"
        language = "TypeScript" if rel.endswith((".ts", ".tsx")) else "JavaScript"
        self._file_anchor(rel, f"{language} module" + (f": {detail}" if detail else ""))
        self._code_summary(rel)

    # -- Java ---------------------------------------------------------------

    def _java(self, rel: str) -> None:
        if self.index.text(rel) is None:
            return
        lines = self.index.lines(rel)
        # (name, indent, start incl. annotation lines, end, keyword, declaration line)
        types: list[tuple[str, int, int, int, str, int]] = []
        for number, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith(("//", "*", "/*")):
                continue
            match = _JAVA_TYPE_RE.match(line)
            if match:
                start = number
                while start > 1 and lines[start - 2].strip().startswith("@"):
                    start -= 1
                types.append((match.group("name"), _indent(line), start, _block_end(lines, number),
                              match.group("kind"), number))
        if types:
            self._flavour("java", rel)
            self.found_classes.setdefault("java", set()).update(
                found for t in types if t[1] == 0 and (found := self._entity(t[0])))
            stem = posixpath.splitext(posixpath.basename(rel))[0]
            if not any(_JAVA_PACKAGE_RE.match(line) for line in lines) and any(
                    (m := _JAVA_GENERATED_CTOR_RE.match(line)) and m.group(1) == stem for line in lines):
                self.generated.add("java")
        methods: dict[str, list[tuple[str, int, int, Optional[str], int]]] = {}
        fields: dict[str, int] = {}
        for number, line in enumerate(lines, 1):
            owner = _innermost(types, number)
            if owner is None or number <= owner[5]:
                continue
            match = _JAVA_METHOD_RE.match(line)
            if match and match.group("name") not in _JAVA_NOT_METHODS \
                    and (match.group("ret") or "") not in _JAVA_NOT_METHODS \
                    and (match.group("ret") or match.group("name") == owner[0]) \
                    and _indent(line) > owner[1]:
                if any(m[1] <= number <= m[2] for m in methods.get(owner[0], [])):
                    continue
                start = number
                while start > 1 and lines[start - 2].strip().startswith("@"):
                    start -= 1
                end = _block_end(lines, number)
                body = "\n".join(lines[number - 1:end])
                status = None
                todo = next((m for m in (_TODO_COMMENT_RE.match(x) for x in lines[number:end]) if m), None)
                if _JAVA_STUB_RE.search(body):
                    status = "stub: throws UnsupportedOperationException"
                elif todo:
                    status = "TODO: " + _clip(todo.group(1).strip() or "placeholder", 60)
                methods.setdefault(owner[0], []).append((match.group("name"), start, end, status, number))
            elif _JAVA_FIELD_RE.match(line) and not any(
                    m[1] <= number <= m[2] for m in methods.get(owner[0], [])):
                fields[owner[0]] = number
        for name, indent, start, end, keyword, decl in types:
            entity = self._entity(name) if indent == 0 else None
            members = methods.get(name, [])
            constructors = sum(1 for m in members if m[0] == name)
            counts = ([_plural(constructors, "constructor")] if constructors else []) + (
                [_plural(len(members) - constructors, "method")] if len(members) > constructors else [])
            label = f"{keyword} {name}" + (f" ({', '.join(counts)})" if counts else "")
            self._add("class", rel, name, KIND_CLASS, label, start, end, prefer=(decl,), entity=entity,
                      priority=_P_KEY)
            if name in fields:
                line_no = fields[name]
                self._add("fields_end", rel, name, KIND_INSERT, f"after the last field of {name}", line_no, line_no,
                          prefer=(line_no,), entity=entity, priority=_P_BLOCK,
                          hint="add `private <Type> <name>;` AFTER this line, then a constructor parameter and "
                               "get/set accessors")
            run: list[tuple[str, int, int, Optional[str], int]] = []
            for member in members + [("", 0, 0, None, 0)]:
                accessor = bool(member[0]) and _JAVA_ACCESSOR_RE.match(member[0]) and member[3] is None \
                    and member[2] - member[1] <= 4
                if accessor:
                    run.append(member)
                    continue
                if len(run) > 1:
                    self._add("accessors", rel, name, KIND_BLOCK,
                              f"accessors of {name}: {_names([m[0] for m in run], 4)}", run[0][1], run[-1][2],
                              prefer=tuple(m[4] for m in run), entity=entity, priority=_P_ACCESSORS)
                else:
                    for single in run:
                        self._java_method(rel, name, single, entity)
                run = []
                if member[0]:
                    self._java_method(rel, name, member, entity)
        label = "Java source: " + _names([f"{t[0]}" for t in types], 3) if types else "Java source"
        plain = self.plain.get(rel, [])
        if plain:
            label += f"; plain: {_names(plain, 4)}"
        self._file_anchor(rel, label)
        self._code_summary(rel)

    def _java_method(self, rel: str, owner: str, member: tuple[str, int, int, Optional[str], int],
                     entity: Optional[str]) -> None:
        name, start, end, status, signature = member
        if not status and not entity:
            self.plain.setdefault(rel, []).append(f"{owner}.{name}")    # named, not anchored
            return
        label = f"constructor {owner}(...)" if name == owner else f"method {owner}.{name}()"
        # The signature line first: an annotation line (`@Override`) is unique only by chance.
        prefer = (signature, *range(signature + 1, min(end, signature + 3) + 1), *range(start, signature))
        self._add("method", rel, f"{owner}.{name}", KIND_STUB if status else KIND_FUNCTION, label, start, end,
                  prefer=prefer, entity=entity, status=status,
                  priority=_P_STUB if status else _P_METHOD,
                  hint="stub: implement the body in place and keep the signature" if status else "")

    # -- BPMN ---------------------------------------------------------------

    def _bpmn(self, rel: str) -> None:
        if self.index.text(rel) is None:
            return
        lines = self.index.lines(rel)
        elements: list[dict] = []
        for number, line in enumerate(lines, 1):
            for match in _XML_TAG_RE.finditer(line):
                prefix, tag, attrs, closed = match.group(1), match.group(2), match.group(3), match.group(4)
                if tag not in _BPMN_TAGS:
                    continue
                values = dict(_XML_ATTR_RE.findall(attrs))
                if "id" not in values:
                    continue
                end = number
                if not closed:
                    closing = f"</{prefix}:{tag}>" if prefix else f"</{tag}>"
                    end = next((n for n in range(number, len(lines) + 1) if closing in lines[n - 1]), number)
                elements.append({"tag": tag, "line": number, "end": end, "id": values["id"], "attrs": values})
        self._flavour("bpmn", rel)
        flows = {e["id"]: e for e in elements if e["tag"] == "sequenceFlow"}
        outgoing: dict[str, list[dict]] = {}
        for flow in flows.values():
            outgoing.setdefault(flow["attrs"].get("sourceRef", ""), []).append(flow)
        counts: dict[str, int] = {}
        not_executable = []
        unconditioned: list[str] = []
        for element in elements:
            tag, ident, attrs = element["tag"], element["id"], element["attrs"]
            name = attrs.get("name", "")
            kind_word = "gateway" if tag.endswith("Gateway") else "task" if tag.endswith("Task") or tag == "task" \
                else "event" if tag.endswith("Event") else "flow" if tag == "sequenceFlow" else tag
            counts[kind_word] = counts.get(kind_word, 0) + 1
            label = f"{tag} {name!r} ({ident})" if name else f"{tag} {ident}"
            status = None
            priority = _P_MINOR if tag == "sequenceFlow" else _P_BLOCK
            if tag == "process":
                label = f"process {name or ident} (isExecutable={attrs.get('isExecutable', '?')})"
                priority = _P_KEY
                if attrs.get("isExecutable") == "false":
                    not_executable.append(name or ident)
            elif tag == "sequenceFlow":
                label = f"flow {ident}: {attrs.get('sourceRef', '?')} → {attrs.get('targetRef', '?')}" + (
                    f" {name!r}" if name else "")
            elif tag in _BPMN_CONDITIONAL_GATEWAYS:
                bare = [f for f in outgoing.get(ident, []) if f["id"] != attrs.get("default")
                        and not any("conditionExpression" in lines[n - 1] for n in range(f["line"], f["end"] + 1))]
                if len(outgoing.get(ident, [])) > 1 and bare:
                    status = f"stub: no condition on {_names([f['id'] for f in bare], 3)}"
                    priority = _P_STUB
                    unconditioned.append(name or ident)
            self._add("bpmn", rel, ident, KIND_STUB if status else KIND_BLOCK, label, element["line"],
                      element["end"], prefer=(element["line"],), status=status, priority=priority,
                      hint="stub: add a <bpmn:conditionExpression> to each non-default outgoing flow, or decide in "
                           "the runner code" if status else "")
        detail = ", ".join(_plural(v, k) for k, v in counts.items() if k != "process")
        self._file_anchor(rel, "BPMN process" + (f": {detail}" if detail else ""))
        processes = [e["attrs"].get("name") or e["id"] for e in elements if e["tag"] == "process"]
        title = (f"process {processes[0]}" if len(processes) == 1
                 else f"processes {_names(processes, 4)}" if processes else "process ?")
        self.summaries.append((rel, f"{rel}: BPMN {title} — {detail or 'no elements'}"
                               + ("; not executable (isExecutable=false)" if not_executable else "")
                               + (f"; gateways without conditions: {_names(unconditioned, 3)}"
                                  if unconditioned else "")))
        if not any(r.intent == "Implement the process" for r in self.recipes):
            self.recipes.append(Recipe("Implement the process", [
                f"The XML has no code slot: implement each task in new code keyed by its XML id "
                f"(`{FAMILY}:bpmn:{rel}:<id>`); names can be empty or repeated.",
                "Gateway routing: add <bpmn:conditionExpression> to the non-default outgoing sequenceFlows, or "
                "decide in the runner; the gateway's `default` flow needs none.",
            ]))

    # -- other text files ---------------------------------------------------

    def _text_file(self, rel: str) -> None:
        text = self.index.text(rel)
        if text is None:
            return
        name = posixpath.basename(rel)
        suffix = posixpath.splitext(name)[1].lower()
        role = _NAMED_ROLES.get(name) or _TEXT_ROLES.get(suffix, "file")
        lines = self.index.lines(rel)
        if suffix == ".json":
            role = self._json_role(rel, name, text)
        elif suffix in (".yaml", ".yml"):
            placeholders = len(_YOUR_PLACEHOLDER_RE.findall(text))
            if name.startswith(("docker-compose", "compose")):
                services = _compose_services(lines)
                role = "Docker Compose" + (f": services {_names(services, 4)}" if services else "")
            elif any(pf.flavour == "baf" for pf in self.py_files.values()) or ("platforms:" in text and "nlp:" in text):
                role = "agent config"
            else:
                role = "YAML config"
            if placeholders:
                role += f", {_plural(placeholders, 'YOUR-* placeholder')} to fill"
        elif suffix == ".sql":
            role = self._sql_role(rel, text, lines)
        elif suffix in (".ttl", ".nt", ".owl", ".rdf"):
            role = self._rdf_role(rel, lines) or role
        elif suffix == ".txt" and name.startswith("requirements"):
            count = sum(1 for line in lines if _REQUIREMENT_RE.match(line) and not line.lstrip().startswith("#"))
            role = f"Python requirements ({_plural(count, 'package')})"
        elif suffix in (".md", ".txt", ".rst") and name.lower().startswith("readme"):
            role = "readme"
        elif suffix == ".md" and "/skills/" in f"/{rel}":
            role = "agent skill (Markdown playbook)"
        self._file_anchor(rel, role)
        self.other_files[rel] = role

    def _json_role(self, rel: str, name: str, text: str) -> str:
        if name.startswith(("tsconfig", "jsconfig")):
            return "TypeScript config"
        try:
            data = json.loads(text)
        except ValueError as exc:
            # The parser's message without its position: what reaches the prompt stays line-free.
            what = exc.msg if isinstance(exc, json.JSONDecodeError) else "not parseable"
            self._warn(f"{rel} is not valid JSON ({_clip(what, 60)}); fix it before editing.")
            return f"INVALID JSON ({_clip(what, 60)})"
        if name == "package.json" and isinstance(data, dict):
            deps = len(data.get("dependencies") or {}) + len(data.get("devDependencies") or {})
            return f"npm manifest ({_plural(deps, 'dependency')})"
        if isinstance(data, dict) and isinstance(data.get("objects"), list) \
                and all(isinstance(o, dict) and "class" in o for o in data["objects"]):
            self._flavour("object_json", rel)
            classes = [str(o.get("class")) for o in data["objects"]]
            self.summaries.append((rel, f"{rel}: object model {data.get('name', '')} — "
                                        f"{_plural(len(classes), 'object')} of {_names(sorted(set(classes)), 4)}"))
            if not any(r.intent == "Use the object data" for r in self.recipes):
                self.recipes.append(Recipe("Use the object data", [
                    f"{rel} lists objects by `id` with `class`, `attributes` and `relationships`; write a seeder or "
                    "fixtures loader that reads it and edit the JSON only to change data.",
                    "A relationship is keyed by the object's OWN role name (the reverse of UML navigation); object "
                    "ids are instance names, not database keys.",
                ]))
            return f"object model ({_plural(len(classes), 'object')})"
        if isinstance(data, dict) and ("$schema" in data or "definitions" in data or "$defs" in data):
            self._flavour("json_schema", rel)
            definitions = data.get("definitions") or data.get("$defs") or {}
            return f"JSON Schema ({_plural(len(definitions), 'definition')})"
        if isinstance(data, dict) and "openapi" in data:
            return f"OpenAPI spec ({_plural(len(data.get('paths') or {}), 'path')})"
        return "JSON data"

    def _sql_role(self, rel: str, text: str, lines: list[str]) -> str:
        """The file's role, plus one block per ``CREATE TABLE`` (linked to its model class by name)."""
        tables: list[tuple[str, str, int, int]] = []    # (schema, table, first line, last line)
        for number, line in enumerate(lines, 1):
            match = _SQL_TABLE_RE.match(line)
            if match:
                tables.append(((match.group(1) or "public").lower(), match.group(2), number,
                               _paren_end(lines, number)))
        policies = len(re.findall(r"\bCREATE\s+POLICY\b", text, re.I))
        supabase = "supabase" in rel.lower() or "auth.users" in text or "ROW LEVEL SECURITY" in text.upper()
        self._flavour("supabase" if supabase else "sql", rel)
        for schema, table, start, end in tables:
            self._add("table", rel, table, KIND_BLOCK, f"table {table}" + (f" ({schema})" if supabase else ""),
                      start, end, prefer=(start,), entity=self._entity(table), priority=_P_BLOCK,
                      hint="CREATE TABLE block; Supabase adds association columns later with ALTER TABLE"
                      if supabase else "CREATE TABLE block")
        # Tables without an id column that foreign keys reference by (id), in the same schema
        # (`REFERENCES auth.users(id)` is Supabase's own table, never the model's `users`).
        referenced = {((m.group(1) or "public").lower(), m.group(2).lower())
                      for m in _SQL_REFERENCES_RE.finditer(text)}
        missing_id = []
        for schema, table, start, end in tables:
            if (schema, table.lower()) not in referenced:
                continue
            if not any(_SQL_ID_COLUMN_RE.match(lines[n - 1]) for n in range(start + 1, end + 1)):
                missing_id.append(table)
        if missing_id:
            self._warn(f"{rel}: foreign keys reference (id) of {_names(missing_id, 3)}, but "
                       f"{'those tables have' if len(missing_id) > 1 else 'that table has'} no id column; add one "
                       "(e.g. \"id\" UUID PRIMARY KEY DEFAULT gen_random_uuid()) before applying the schema.")
        role = ("Supabase migration" if supabase else "SQL DDL") + f": {_plural(len(tables), 'table')}"
        if tables:
            role += f" ({_names([t[1] for t in tables], 4)})"
        if policies:
            role += f", {_plural(policies, 'RLS policy')}"
        return role

    def _rdf_role(self, rel: str, lines: list[str]) -> str:
        """The file's role, plus one block per RDF class (linked to its model class by name)."""
        classes = [(m.group(1), m.group(2), n) for n, line in enumerate(lines, 1)
                   if (m := _RDF_CLASS_RE.match(line))]
        properties = [m.group(1) for line in lines if (m := _RDF_PROPERTY_RE.match(line))]
        if not classes and not properties:
            return ""
        self._flavour("rdf", rel)
        bannered = _PER_CLASS_MARKERS["rdf"] in lines
        if bannered:
            self.generated.add("rdf")
        for qualified, local, number in classes:
            end = number
            while end < len(lines) and not lines[end - 1].rstrip().endswith(".") and lines[end].strip():
                end += 1        # a Turtle statement ends with ' .'
            entity = self._entity(local)
            self._add("rdf_class", rel, local, KIND_BLOCK, f"RDF class {qualified}", number, end,
                      prefer=(number,), entity=entity, priority=_P_BLOCK)
            if entity and bannered:
                self.found_classes.setdefault("rdf", set()).add(entity)
        duplicates = sorted({p for p in properties if properties.count(p) > 1})
        if duplicates:
            self._warn(f"{rel} declares {_names(duplicates, 3)} more than once; each extra rdfs:domain is read as an "
                       "intersection, so give same-named properties distinct names.")
        return f"RDF vocabulary: {_plural(len(classes), 'class')}, {_plural(len(properties), 'property')}"

    # -- file anchors and summaries ------------------------------------------

    def _file_anchor(self, rel: str, role: str) -> None:
        lines = self.index.lines(rel)
        end = max(1, len(lines))
        self.file_anchors[rel] = Anchor(
            id=self._id("file", rel), file=rel, kind=KIND_FILE, label=_clip(role, 110), start_line=1, end_line=end,
            quote=self._first_quote(rel, 1, min(end, 40)) if lines else "",
        )

    def _code_summary(self, rel: str) -> None:
        """One line for a code file; model classes are left to the per-entity lines."""
        items = [i for i in self.items.get(rel, []) if not i.anchor.entity]
        if any(i.anchor.entity for i in self.items.get(rel, [])):
            self.model_class_files.append(rel)
        classes = [i.symbol for i in items if i.anchor.kind == KIND_CLASS]
        functions = [i.symbol for i in items
                     if i.anchor.kind == KIND_FUNCTION and i.anchor.id.startswith(f"{FAMILY}:function:")]
        functions += [name for name in self.plain.get(rel, []) if "." not in name]
        stubs = [i.symbol for i in items if i.anchor.kind == KIND_STUB]
        parts = [f"classes {_names(classes, 5)}" if classes else "",
                 f"functions {_names(functions, 5)}" if functions else "",
                 f"stubs {_names(stubs, 4)}" if stubs else ""]
        detail = "; ".join(p for p in parts if p)
        if detail or rel not in self.model_class_files:
            self.code_summaries[rel] = f"{rel}: {detail}" if detail else rel

    # -- assembly -------------------------------------------------------------

    def _capped(self, rel: str) -> list[Anchor]:
        """The file's anchors after the per-file and per-entity caps, in line order."""
        items = self.items.get(rel, [])
        keep: list[_Item] = []
        dropped = 0
        groups: dict[Optional[str], list[_Item]] = {}
        for item in items:
            groups.setdefault(item.anchor.entity, []).append(item)
        for entity, group in groups.items():
            cap = MAX_ANCHORS_PER_FILE if entity is None else MAX_ANCHORS_PER_ENTITY_FILE
            ranked = sorted(group, key=lambda i: (i.priority, i.order))
            keep.extend(ranked[:cap])
            dropped += max(0, len(ranked) - cap)
        if dropped:
            self.notes.append(f"{rel}: {dropped} more definitions not mapped (cap); read the file")
        return [i.anchor for i in sorted(keep, key=lambda i: (i.anchor.start_line, i.order))]

    def _assemble(self) -> ScaffoldMap:
        smap = ScaffoldMap(family=FAMILY, root=self.root, title=self._title())
        total = 0
        files_with_anchor = 0
        skipped_files = 0
        for rel in self.files:
            if rel not in self.file_anchors:
                continue
            code = rel.endswith((*_CODE_SUFFIXES, ".bpmn"))
            if code or files_with_anchor < MAX_FILE_ANCHORS_PER_MAP:
                smap.anchors.append(self.file_anchors[rel])
                files_with_anchor += int(not code)
            anchors = self._capped(rel)
            if total + len(anchors) > MAX_ANCHORS_PER_MAP:
                skipped_files += 1      # the file stays locatable by path; its definitions are not listed
                continue
            for anchor in anchors:
                if anchor.entity:
                    entity = smap.entities.setdefault(anchor.entity, EntityMap(name=anchor.entity))
                    entity.anchors.append(anchor)
                else:
                    smap.anchors.append(anchor)
            total += len(anchors)
        if skipped_files:
            self.notes.append(f"definitions of {_plural(skipped_files, 'file')} not listed: the map's "
                              f"{MAX_ANCHORS_PER_MAP}-anchor cap was reached")
        for entity in smap.entities.values():
            entity.summary = self._entity_summary(entity)
        self._missing(smap)
        smap.summary = self._summary()
        smap.warnings = self.warnings[:MAX_WARNINGS]
        if len(self.warnings) > MAX_WARNINGS:
            smap.warnings.append(f"{len(self.warnings) - MAX_WARNINGS} more pitfalls in {self.root or 'this folder'} "
                                 "(call locate_code without arguments)")
        smap.recipes = (self._baf_recipes() + self.recipes + self._class_recipes())[:MAX_RECIPES]
        return smap

    def _entity_summary(self, entity: EntityMap) -> list[str]:
        by_file: dict[str, list[Anchor]] = {}
        for anchor in entity.anchors:
            by_file.setdefault(anchor.file, []).append(anchor)
        parts = []
        for rel, anchors in by_file.items():
            members = [(a, _symbol(a).rsplit(".", 1)[-1]) for a in anchors
                       if a.kind in (KIND_FUNCTION, KIND_STUB) and ":method:" in a.id]
            # Python's __init__ is a constructor like Java's; other dunders are no business methods.
            constructors = sum(1 for a, short in members if a.label.startswith("constructor") or short == "__init__")
            methods = [a for a, short in members
                       if not (a.label.startswith("constructor") or (short.startswith("__") and short.endswith("__")))]
            stubs = [_symbol(a).rsplit(".", 1)[-1] for a in anchors if a.kind == KIND_STUB]
            pieces = []
            if any(f"{FAMILY}:class:" in a.id for a in anchors):
                pieces.append("class")
            if any(f"{FAMILY}:table:" in a.id for a in anchors):
                pieces.append("table")
            if any(f"{FAMILY}:rdf_class:" in a.id for a in anchors):
                pieces.append("RDF class")
            if any(":properties:" in a.id or ":accessors:" in a.id for a in anchors):
                pieces.append("accessors")
            if any(":relationships:" in a.id for a in anchors):
                pieces.append("relationships")
            if constructors:
                pieces.append(_plural(constructors, "constructor"))
            if methods:
                pieces.append(_plural(len(methods), "method"))
            text = f"{rel}: {', '.join(pieces) or 'code'}"
            if stubs:
                text += f"; stub{'s' if len(stubs) > 1 else ''} {_names(stubs, 3)}"
            parts.append(text)
        return parts

    def _missing(self, smap: ScaffoldMap) -> None:
        """Classes of the model that a per-class scaffold has no definition for (modify runs)."""
        if not self.class_names:
            return
        for flavour, piece in _PER_CLASS_PIECES:
            found = self.found_classes.get(flavour)
            if not found:
                continue
            matched = found & self.class_names
            if flavour in self.generated:
                # Its generator's banner (Java: its constructor shape) proves the file was generated from
                # this model: one shared class is enough, however many classes the model gained since.
                if not matched:
                    continue
            elif len(matched) < min(2, len(self.class_names)) or 2 * len(matched) < len(self.class_names):
                continue    # hand-written: may hold any subset of the model
            elif self._partial():
                continue    # hand-written, and some of its code was never read: absence proves nothing
            for name in sorted(self.class_names - found):
                if flavour == "java" and self._java_file_on_disk(name):
                    continue    # left out by the index's caps, or too large to read: not missing
                entity = smap.entities.setdefault(name, EntityMap(name=name))
                if piece not in entity.missing:
                    entity.missing.append(piece)

    def _partial(self) -> bool:
        """Whether some code of this group is unseen: files the index left out (its caps) or could not read."""
        return bool(self.index.capped) or any(self.index.text(rel) is None
                                               for rel in self.files if rel.endswith(_CODE_SUFFIXES))

    def _java_file_on_disk(self, name: str) -> bool:
        """Whether ``<Name>.java`` exists next to this group's Java files, mapped or not.

        One file per class: a class the map did not read may still be on disk, so
        it is never reported missing before this check.
        """
        folders = sorted({posixpath.dirname(rel) for rel in self.files if rel.endswith(_JAVA_SUFFIXES)})
        return any(self.index.exists(posixpath.join(folder, f"{name}.java")) for folder in folders)

    def _summary(self) -> list[str]:
        lines = []
        for rel, text in self.summaries:
            # Stub bodies inside a specialised file (a BAF body's TODO, an empty training step) would
            # otherwise never reach the overview: the specialised line replaces the file's code line.
            stubs = [i.symbol for i in self.items.get(rel, []) if i.anchor.kind == KIND_STUB
                     and i.anchor.id.startswith((f"{FAMILY}:function:", f"{FAMILY}:method:"))]
            lines.append(text + (f"; stubs {_names(stubs, 4)}" if stubs else ""))
        specialised = {rel for rel, _ in self.summaries}
        if self.model_class_files:
            lines.append("model classes (see per entity) in " + _names(self.model_class_files, 4))
        code = [text for rel, text in self.code_summaries.items() if rel not in specialised]
        others = [f"{rel} [{role}]" if role not in ("file", "text", "Markdown") else rel
                  for rel, role in self.other_files.items() if rel not in specialised]
        room = max(1, MAX_SUMMARY_LINES - len(lines) - (1 if others else 0) - (1 if self.notes else 0))
        if len(code) > room:
            rest = len(code) - (room - 1)
            code = code[: room - 1] + [f"+{_plural(rest, 'more code file')}: "
                                       + _names([c.split(":", 1)[0] for c in code[room - 1:]], 4)]
        lines.extend(code)
        if others:
            lines.append("files: " + _names(others, 4))
        if self.notes:
            lines.append("not fully mapped: " + "; ".join(self.notes[:2])
                         + (f" (+{len(self.notes) - 2})" if len(self.notes) > 2 else ""))
        return [_clip(line, _MAX_SUMMARY_CHARS) for line in lines[:MAX_SUMMARY_LINES]]

    def _class_recipes(self) -> list[Recipe]:
        """Recipes of the per-class scaffolds, each step naming only what this code actually has."""
        recipes = []
        if "python_classes" in self.flavours:
            rel = self.first_file["python_classes"]
            steps = [(f"Methods: replace the placeholder body of each stub method in {rel}, keeping its signature."
                      if self._has_stub(rel) else f"Methods: add them inside the class in {rel}.")]
            if self._has_anchor(rel, "properties"):
                steps.append("Attributes: add the `__init__` parameter and a @property / @<name>.setter pair in the "
                             f"class's `{FAMILY}:properties:{rel}:<Class>` block; association setters keep both "
                             "ends in sync, so assign through them.")
            recipes.append(Recipe("Implement a method or add an attribute", steps))
        if "java" in self.flavours:
            java = [rel for rel in self.files if rel.endswith(_JAVA_SUFFIXES)]
            steps = [("Methods: replace `throw new UnsupportedOperationException(...)` in each stub method."
                      if any(self._has_stub(rel) for rel in java) else "Methods: add them inside the class body.")]
            if any(self._has_anchor(rel, "fields_end") for rel in java):
                steps.append(f"Fields: `private <Type> <name>;` at `{FAMILY}:fields_end:<file>:<Class>`, then a "
                             "constructor parameter and get/set accessors"
                             + ("; there is no package line and no build file." if "java" in self.generated else "."))
            recipes.append(Recipe("Implement a method or add a field (Java)", steps))
        if "sqlalchemy" in self.flavours:
            rel = self.first_file["sqlalchemy"]
            where = (f"relationships in the module-level `{FAMILY}:relationships:{rel}:<Class>` block, never inside "
                     "the class." if self._has_anchor(rel, "relationships") else "relationships too.")
            recipes.append(Recipe("Add a column or relationship (SQLAlchemy)", [
                f"Columns go in the class body (`{FAMILY}:class:{rel}:<Class>`); {where}",
                "There are no migrations: create_all does not alter an existing table.",
            ]))
        if "pydantic" in self.flavours:
            names = [c.name for pf in self.py_files.values() if pf.flavour == "pydantic" for c in pf.classes
                     if any("BaseModel" in _unparse(b) for b in c.bases)]
            if names and all(name.endswith("Create") for name in names):
                kind = ("Only request schemas (`<Entity>Create`, link payloads) are generated: add read/response "
                        "models next to them when an endpoint returns data.")
            elif any(name.endswith("Create") for name in names):
                kind = "`<Entity>Create` request schemas sit next to the entity models."
            else:
                kind = "There are no *Create schemas: the models are used as-is."
            recipes.append(Recipe("Add validation (Pydantic)", [
                "Validators go inside the model class as `@field_validator('<field>')` + `@classmethod`; attributes "
                f"are sorted by name, relationships follow, and no methods are generated. {kind}",
            ]))
        return recipes

    def _has_anchor(self, rel: str, what: str) -> bool:
        return any(item.anchor.id.startswith(f"{FAMILY}:{what}:") for item in self.items.get(rel, []))

    def _has_stub(self, rel: str) -> bool:
        return any(item.anchor.kind == KIND_STUB for item in self.items.get(rel, []))

    def _title(self) -> str:
        titles = [title for flavour, title in _FLAVOUR_TITLES if flavour in self.flavours]
        if titles:
            return " + ".join(titles[:2])
        folder = posixpath.basename(self.root) if self.root else ""
        if folder in _FOLDER_TITLES:
            return _FOLDER_TITLES[folder]
        languages = []
        for suffixes, name in ((_PY_SUFFIXES, "Python"), (_SCRIPT_SUFFIXES, "TypeScript/JavaScript"),
                               (_JAVA_SUFFIXES, "Java")):
            if any(f.endswith(suffixes) for f in self.files):
                languages.append(name)
        if languages:
            return " + ".join(languages) + " code"
        return "Workspace files" if not self.root else "Project files"


# ---------------------------------------------------------------------------
# Module-level analysis helpers
# ---------------------------------------------------------------------------

_PY_ROLES = {
    "baf": "BESSER agent (BAF)", "qiskit": "Qiskit circuit script", "pytorch": "PyTorch network",
    "tensorflow": "TensorFlow/Keras network", "sqlalchemy": "SQLAlchemy models", "pydantic": "Pydantic models",
    "python_classes": "Python domain classes", "django": "Django module", "fastapi": "FastAPI module",
    "flask": "Flask module",
}


def _python_flavour(tree: ast.Module, text: str) -> str:
    """What a parsed module looks like, from its imports and definitions."""
    modules = _import_modules(tree)
    roots = {m.split(".")[0] for m in modules}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if any(m == "baf" or m.startswith(("baf.", "besser.agent")) for m in modules) and _agent_var(tree)[0]:
        return "baf"
    if "qiskit" in roots:
        return "qiskit"
    nn_model = any(_unparse(b) in _NN_BASES for c in classes for b in c.bases) or any(
        isinstance(n, ast.Assign) and _unparse(getattr(n.value, "func", None)).endswith("Sequential")
        for n in tree.body)
    if "torch" in roots and nn_model:
        return "pytorch"
    if roots & {"tensorflow", "keras"} and nn_model:
        return "tensorflow"
    if "sqlalchemy" in roots and any(
            any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__tablename__"
                                                  for t in s.targets) for s in c.body) for c in classes):
        return "sqlalchemy"
    if "pydantic" in roots and any("BaseModel" in _unparse(b) for c in classes for b in c.bases):
        return "pydantic"
    for framework in ("django", "fastapi", "flask"):
        if framework in roots:
            return framework
    # Only the generator's banner makes plain classes "domain classes": a hand-written module of
    # dataclasses has no properties or __init__ to extend and may hold any subset of the model.
    if classes and _PER_CLASS_MARKERS["python_classes"] in text:
        return "python_classes"
    return "python"


def _agent_var(tree: ast.Module) -> tuple[str, str]:
    """(variable, agent name) of the module-level ``<var> = Agent('<name>')``, or ('', '')."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and _call_name(node.value)[1] == "Agent":
            args = node.value.args
            name = args[0].value if args and isinstance(args[0], ast.Constant) else ""
            return node.targets[0].id, str(name)
    return "", ""


def _defined_names(tree: ast.Module) -> set[str]:
    """Names bound at module level (defs, classes, assignments, imports)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _compose_services(lines: list[str]) -> list[str]:
    """Service names under a top-level ``services:`` key of a compose file."""
    services: list[str] = []
    inside = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            inside = line.rstrip() == "services:"
            continue
        match = re.match(r"^( +)([\w.-]+):\s*$", line)
        if inside and match and len(match.group(1)) == 2:
            services.append(match.group(2))
    return services


def _paren_end(lines: list[str], start: int, limit: int = 400) -> int:
    """Last line of the parenthesised statement opened on line ``start`` (SQL ``CREATE TABLE x (...)``)."""
    depth, opened = 0, False
    for number in range(start, min(len(lines), start + limit) + 1):
        for char in re.sub(r"'[^']*'|--.*$", "", lines[number - 1]):
            if char == "(":
                depth, opened = depth + 1, True
            elif char == ")":
                depth -= 1
        if opened and depth <= 0:
            return number
    return start


def _innermost(types: list[tuple], number: int) -> Optional[tuple]:
    """The innermost Java type whose range holds line ``number``."""
    best = None
    for item in types:
        if item[2] <= number <= item[3] and (best is None or item[2] >= best[2]):
            best = item
    return best
