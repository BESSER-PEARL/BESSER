"""Scaffold of a new deterministic generator in a BESSER checkout.

Writes every part BESSER's ``CLAUDE.md`` ("Adding a deterministic generator") and
``docs/source/contributing/create_generator.rst`` ask of a new generator: the package
(a ``GeneratorInterface`` subclass and one Jinja template), its ``GeneratorSpec`` in
``besser/generators/registry.py``, a test, the docs page with its toctree and table
entries, the pinned lists and snapshot of ``test_generator_registry.py``, and its
regeneration builder. The result is a working generator the lab agent then changes.

The checkout's files are read and edited as text (never imported): the registry
there is the one being changed, not the one this process runs.
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from pathlib import Path
from string import Template

from besser.spec_driven_agent.lab.contracts import ExtraTool

TEMPLATES_DIR = Path(__file__).with_name("templates")

REGISTRY = "besser/generators/registry.py"
REGISTRY_TEST = "tests/generators/test_generator_registry.py"
EDITOR_SNAPSHOT = "tests/generators/snapshots/web_editor_generators.json"
REGENERABLE_BUILDERS = "tests/generators/regenerable_builders.py"
GENERATORS_DOC = "docs/source/generators.rst"

#: The registry category and docs section of a scaffold; the agent moves it when another fits better.
CATEGORY = "object_oriented"
DOCS_TABLE_SECTION = "Choosing a Generator"
DOCS_TOCTREE_SECTION = "Frameworks & Languages"
#: Line length of the repository's Python (``pyproject.toml``).
MAX_LINE = 120

_NAME = re.compile(r"[a-z][a-z0-9_]*")
_EXTENSION = re.compile(r"\.[A-Za-z0-9]+")


class ScaffoldError(ValueError):
    """The generator cannot be scaffolded; nothing was written."""


def scaffold_generator(repo_path: str, name: str, *, class_name: str | None = None,
                       file_extension: str = ".txt") -> list[str]:
    """Write a new, working generator ``name`` into the BESSER checkout at ``repo_path``.

    Every file is planned before any is written, so a refusal leaves the checkout unchanged.
    Edited files keep their line endings.

    Args:
        repo_path: Root of the BESSER checkout.
        name: Generator id and package name (lower-case Python identifier).
        class_name: Generator class; defaults to the CamelCase name plus ``Generator``.
        file_extension: Extension of the one file the generator writes, with its dot.

    Returns:
        The paths written or changed, relative to ``repo_path`` with ``/`` separators.

    Raises:
        ScaffoldError: A name, class name or extension that is not valid, a generator or
            file that already exists, or a repository file whose layout is not the expected one.
    """
    if not _NAME.fullmatch(name) or keyword.iskeyword(name):
        raise ScaffoldError(f"{name!r} is not a valid generator name: use a lower-case Python identifier.")
    class_name = class_name or "".join(part.capitalize() for part in name.split("_")) + "Generator"
    if not class_name.isidentifier() or keyword.iskeyword(class_name):
        raise ScaffoldError(f"{class_name!r} is not a valid Python class name.")
    if not _EXTENSION.fullmatch(file_extension):
        raise ScaffoldError(f"{file_extension!r} is not a file extension such as '.txt'.")

    repo = Path(repo_path)
    title = name.replace("_", " ").title()
    heading = f"{title} Generator"
    values = {
        "name": name, "class_name": class_name, "title": title, "category": CATEGORY,
        "file_extension": file_extension, "output_file": f"{name}{file_extension}",
        "template_file": f"{name}{file_extension}.j2", "heading": heading, "underline": "=" * len(heading),
    }

    package = f"besser/generators/{name}"
    new_files = {
        f"{package}/__init__.py": _render("generator_init.py.tmpl", values),
        f"{package}/{name}_generator.py": _render("generator.py.tmpl", values),
        f"{package}/templates/{values['template_file']}": _render("output_template.j2.tmpl", values),
        f"tests/generators/{name}/test_{name}_generator.py": _render("test_generator.py.tmpl", values),
        f"docs/source/generators/{name}.rst": _render("docs_page.rst.tmpl", values),
    }
    taken = [path for path in (package, f"{package}.py", *new_files) if (repo / path).exists()]
    if taken:
        raise ScaffoldError(f"Generator {name!r} cannot be created: {', '.join(taken)} already exists.")

    registry = _read(repo, REGISTRY)
    if name in _registered_ids(registry):
        raise ScaffoldError(f"Generator {name!r} is already registered in {REGISTRY}.")
    edits = {
        REGISTRY: _add_registry_entry(registry, _render("registry_entry.py.tmpl", values)),
        REGISTRY_TEST: _add_to_pinned_lists(_read(repo, REGISTRY_TEST), name),
        EDITOR_SNAPSHOT: _add_to_snapshot(_read(repo, EDITOR_SNAPSHOT), name, class_name, values["output_file"],
                                          file_extension),
        REGENERABLE_BUILDERS: _add_builder(_read(repo, REGENERABLE_BUILDERS), name, class_name),
        GENERATORS_DOC: _add_to_docs(_read(repo, GENERATORS_DOC), name, _render("docs_table_row.rst.tmpl", values)),
    }

    for path, text in new_files.items():
        _write(repo, path, text, "\n")
    for path, text in edits.items():
        _write(repo, path, text, _newline(repo, path))
    return sorted([*new_files, *edits])


def scaffold_generator_tool(repo_path: str) -> ExtraTool:
    """The ``scaffold_generator`` tool of the build step, writing into the BESSER checkout ``repo_path``."""

    def handle(arguments: dict) -> str:
        # A refusal raises ScaffoldError (a ValueError): LabToolExecutor returns it as an error result.
        written = scaffold_generator(
            repo_path, str(arguments.get("name", "")), class_name=arguments.get("class_name") or None,
            file_extension=arguments.get("file_extension") or ".txt")
        return "Scaffolded, in the besser repository:\n" + "\n".join(f"- {p}" for p in written) + "\n\n" + _NEXT_STEPS

    schema = {
        "name": "scaffold_generator",
        "description": (
            "Create a new deterministic BESSER generator in the besser repository: the package with a "
            "GeneratorInterface subclass and one Jinja template, its GeneratorSpec in registry.py, a test, the "
            "docs page with its toctree and table entries, the pinned lists and snapshot of "
            "test_generator_registry.py and its regeneration builder. The generator works as written (it lists "
            "each class with its attributes); change it into the real one afterwards. Refuses an existing name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Generator id and package name, e.g. 'graphql_schema'"},
                "class_name": {"type": "string", "description": "Generator class (default: CamelCase + Generator)"},
                "file_extension": {"type": "string", "description": "Extension of the file it writes (default .txt)"},
            },
            "required": ["name"],
        },
    }
    return ExtraTool(schema=schema, handler=handle, writes=True)


_NEXT_STEPS = (
    "Next: replace the template and generate() with the real transformation, then update its test and docs page. "
    f"Its category in registry.py is {CATEGORY!r} and its docs page is in the {DOCS_TOCTREE_SECTION!r} toctree of "
    f"{GENERATORS_DOC}: move either when another fits better, and keep {EDITOR_SNAPSHOT} in step with every "
    "registry field you change. If its output stops being identical for one model, set regenerable = False and "
    f"move it from BUILDERS in {REGENERABLE_BUILDERS} to NOT_REGENERABLE in "
    "tests/generators/test_regenerable_generators.py, with a reason. To make it a Spec-Driven Agent tool, follow "
    "step 5 of 'Adding a deterministic generator' in CLAUDE.md. The editor's Generate menu shows it only once the "
    "frontend adds it (packages/webapp/src/main/app/shell/menus/generator-menu-config.ts and the generator type in "
    "shared/types/workspace-types.ts)."
)


def _render(template: str, values: dict[str, str]) -> str:
    return Template((TEMPLATES_DIR / template).read_text(encoding="utf-8")).substitute(values)


def _read(repo: Path, path: str) -> str:
    """The file's text with ``\\n`` line endings."""
    try:
        return (repo / path).read_bytes().decode("utf-8").replace("\r\n", "\n")
    except FileNotFoundError as exc:
        raise ScaffoldError(f"{path} is missing: is {repo} a BESSER checkout?") from exc


def _newline(repo: Path, path: str) -> str:
    return "\r\n" if b"\r\n" in (repo / path).read_bytes() else "\n"


def _write(repo: Path, path: str, text: str, newline: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.replace("\n", newline).encode("utf-8"))


def _assigned(tree: ast.Module, name: str, path: str) -> ast.expr:
    """The value assigned to the module-level ``name``."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            return node.value
    raise ScaffoldError(f"{path} has no {name} to add the generator to.")


def _registered_ids(registry: str) -> set[str]:
    return {
        kw.value.value for node in ast.walk(ast.parse(registry))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "GeneratorSpec"
        for kw in node.keywords if kw.arg == "id" and isinstance(kw.value, ast.Constant)
    }


def _last_element(text: str, name: str, path: str, kinds: tuple[type, ...]) -> tuple[list[str], ast.expr, ast.expr]:
    """The file's lines, and the first and last elements of the literal assigned to ``name``."""
    value = _assigned(ast.parse(text), name, path)
    elements = value.values if isinstance(value, ast.Dict) else getattr(value, "elts", None)
    if not isinstance(value, kinds) or not elements:
        raise ScaffoldError(f"{name} in {path} is not the non-empty literal the scaffold expects.")
    return text.split("\n"), elements[0], elements[-1]


def _column(line: str, byte_offset: int) -> int:
    """``ast`` columns count UTF-8 bytes."""
    return len(line.encode("utf-8")[:byte_offset].decode("utf-8"))


def _insert_after_element_line(text: str, name: str, path: str, kinds: tuple[type, ...], block: str) -> str:
    """``block`` inserted after the line of the last element of ``name``, which must end with its comma."""
    lines, _, last = _last_element(text, name, path, kinds)
    line = lines[last.end_lineno - 1]
    if line[_column(line, last.end_col_offset):].strip() != ",":
        raise ScaffoldError(f"The last entry of {name} in {path} does not end its line with a comma.")
    lines[last.end_lineno:last.end_lineno] = block.rstrip("\n").split("\n")
    return "\n".join(lines)


def _add_registry_entry(registry: str, entry: str) -> str:
    return _insert_after_element_line(registry, "GENERATORS", REGISTRY, (ast.Tuple,), entry)


def _append_item(text: str, name: str, item: str) -> str:
    """``item`` added after the last element of the list or set literal ``name``."""
    lines, first, last = _last_element(text, name, REGISTRY_TEST, (ast.List, ast.Set))
    index = last.end_lineno - 1
    line = lines[index]
    cut = _column(line, last.end_col_offset)
    joined = f'{line[:cut]}, "{item}"{line[cut:]}'
    if len(joined) > MAX_LINE:
        indent = " " * _column(lines[first.lineno - 1], first.col_offset)
        joined = f'{line[:cut]},\n{indent}"{item}"{line[cut:]}'
    lines[index] = joined
    return "\n".join(lines)


def _add_to_pinned_lists(test: str, name: str) -> str:
    """The editor's pinned ids, and the editor-only set: a scaffold is not an agent tool."""
    return _append_item(_append_item(test, "WEB_EDITOR_IDS", name), "WEB_EDITOR_ONLY", name)


def _add_to_snapshot(snapshot: str, name: str, class_name: str, output_file: str, file_extension: str) -> str:
    """``SUPPORTED_GENERATORS`` as ``test_web_editor_payload_matches_the_snapshot`` derives it."""
    data = json.loads(snapshot)
    data["supported_generators"].append(name)
    data["generators"][name] = {
        "generator_class": f"besser.generators.{name}.{name}_generator.{class_name}",
        "output_type": "file",
        "file_extension": file_extension,
        "category": CATEGORY,
        "requires_class_diagram": True,
        "required_diagram_type": None,
        "download_filename": output_file,
        "keep_generated_filename": False,
        "needs_beyond_one_diagram": None,
    }
    return json.dumps(data, indent=2) + "\n"


def _add_builder(builders: str, name: str, class_name: str) -> str:
    entry = f'    "{name}": _on_library("besser.generators.{name}", "{class_name}"),'
    return _insert_after_element_line(builders, "BUILDERS", REGENERABLE_BUILDERS, (ast.Dict,), entry)


def _directive_block_end(lines: list[str], heading: str, directive: str) -> int:
    """Index after the last indented line of the first ``directive`` under the section ``heading``."""
    try:
        start = lines.index(heading)
        directive_at = next(i for i in range(start, len(lines)) if lines[i].startswith(directive))
    except (ValueError, StopIteration) as exc:
        raise ScaffoldError(f"{GENERATORS_DOC} has no {directive} under {heading!r}.") from exc
    end = directive_at + 1
    for i in range(directive_at + 1, len(lines)):
        if lines[i] and not lines[i].startswith(" "):
            break
        if lines[i].strip():
            end = i + 1
    return end


def _add_to_docs(doc: str, name: str, table_row: str) -> str:
    """The page in a toctree and its row at the end of the "Choosing a Generator" table."""
    lines = doc.split("\n")
    toctree_end = _directive_block_end(lines, DOCS_TOCTREE_SECTION, ".. toctree::")
    lines.insert(toctree_end, f"   generators/{name}")
    table_end = _directive_block_end(lines, DOCS_TABLE_SECTION, ".. list-table::")
    lines[table_end:table_end] = table_row.rstrip("\n").split("\n")
    return "\n".join(lines)

