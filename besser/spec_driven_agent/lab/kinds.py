"""Change kinds of the lab agent: the steps each kind takes and the checklist that checks them.

The steps and paths come from the repositories' own guides: BESSER ``CLAUDE.md`` and
``docs/source/contributing/``, the web editor's ``CLAUDE.md`` and the modeling agent's ``CLAUDE.md``.
Each checklist rule is a function of the changed files and the working tree. A rule answers ``None``
when it cannot judge: its repository is not in the workspace, or a file it reads does not parse.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    ChangeKind,
    ChecklistItem,
    RepoChange,
    RepoSpec,
)

_REPO_TITLES = {REPO_BESSER: "BESSER", REPO_FRONTEND: "Web editor", REPO_MODELING_AGENT: "Modeling agent"}
_ALL = ""  # a step that applies to every repository

# Under BESSER's root.
_GENERATORS_DIR = "besser/generators/"
_REGISTRY = "besser/generators/registry.py"
_REGISTRY_TEST = "tests/generators/test_generator_registry.py"
_WEB_EDITOR_SNAPSHOT = "tests/generators/snapshots/web_editor_generators.json"
_AGENT_TOOLS_SNAPSHOT = "tests/generators/snapshots/agent_generator_tools.json"
_REGENERABLE_BUILDERS = "tests/generators/regenerable_builders.py"
_REGENERABLE_TEST = "tests/generators/test_regenerable_generators.py"
_TOOL_EXECUTOR = "besser/spec_driven_agent/agent/tool_executor.py"
_TOOLS_DOC = "docs/source/spec_driven_agent/tools.rst"
_GENERATORS_DOC = "docs/source/generators.rst"
_CONVERTERS = "besser/utilities/web_modeling_editor/backend/services/converters/"
_JSON_TO_BUML = _CONVERTERS + "json_to_buml/"
_BUML_TO_JSON = _CONVERTERS + "buml_to_json/"
# Under the editor's root.
_EDITOR_PACKAGES = "packages/editor/src/main/packages/"
_EDITOR_REGISTRIES = ("components.ts", "uml-elements.ts", "popups.ts")
# A palette entry lives in its diagram package's preview (``compose-preview.ts`` only declares the types).
_EDITOR_PREVIEW = re.compile(r"^packages/editor/src/main/packages/[^/]+/[^/]+-preview\.ts$")
_CREATE_PANE = "packages/editor/src/main/components/create-pane/create-pane.tsx"
_WEBAPP_SRC = "packages/webapp/src/"
_EDITOR_SRC = "packages/editor/src/"
_WEBAPP_STRINGS = "packages/i18n/en/webapp.json"
_EDITOR_STRINGS = "packages/i18n/en/editor.json"
# Under the modeling agent's root.
_AGENT_HANDLERS = "src/diagram_handlers/types/"
_AGENT_FACTORY = "src/diagram_handlers/registry/factory.py"
_AGENT_SCHEMAS = "src/schemas/"
_AGENT_PROTOCOL_TYPES = "src/protocol/types.py"
_AGENT_CLASSIFIER = "src/routing/unified_classifier.py"

_SOURCE_SUFFIXES = (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_TEST_DIRS = frozenset({"tests", "__tests__", "test"})
# Keys of webapp ``t('a.b')`` / ``i18n.t('a.b')`` and editor ``translate('a.b')`` calls with a literal key.
_WEBAPP_KEY = re.compile(r"(?<![\w$])(?:i18n\.)?t\(\s*(['\"])([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+)\1")
_EDITOR_KEY = re.compile(r"(?<![\w$])translate\(\s*(['\"])([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+)\1")
# Evidence lists longer than this are cut, so one rule cannot flood the model's context.
_MAX_LISTED = 8

# --------------------------------------------------------------------------------------------
# Guidance
# --------------------------------------------------------------------------------------------

_SOURCES = {
    ChangeKind.GENERATOR: 'BESSER CLAUDE.md "Adding a deterministic generator", '
                          "docs/source/contributing/create_generator.rst",
    ChangeKind.METAMODEL: "BESSER CLAUDE.md, docs/source/contributing/create_dsl.rst",
    ChangeKind.DIAGRAM_TYPE: "BESSER docs/source/contributing/diagram_dsl_workflow.rst, the editor's CLAUDE.md "
                             '"Adding a Diagram Type", the modeling agent\'s CLAUDE.md "Adding (or Auditing) a '
                             'Diagram Type"',
    ChangeKind.EDITOR_ELEMENT: 'the editor\'s CLAUDE.md "Adding a Diagram Element"',
    ChangeKind.BACKEND: 'BESSER CLAUDE.md "Where Things Live" and "CI/CD"',
    ChangeKind.AGENT: "the modeling agent's CLAUDE.md, BESSER CLAUDE.md \"Spec-Driven Agent\"",
    ChangeKind.BUGFIX: 'BESSER CLAUDE.md "Engineering standard"',
    ChangeKind.OTHER: "each repository's CLAUDE.md",
}

_STEPS: dict[ChangeKind, tuple[tuple[str, str], ...]] = {
    ChangeKind.GENERATOR: (
        (_ALL, "If the `scaffold_generator` tool is offered, start with it: it writes the skeleton of the "
               "BESSER steps below. Then write the real transformation and check every step."),
        (REPO_BESSER, "Create the package `besser/generators/<name>/` with an `__init__.py` that exports the "
                      "class. The class implements `GeneratorInterface` (`__init__(model, output_dir=None)`, "
                      "`generate()`; `output_dir=None` means `<cwd>/output`) and renders Jinja2 templates from "
                      "`besser/generators/<name>/templates/`. Output is deterministic (no timestamps in file "
                      "names); helpers several generators share go in `besser/utilities/`."),
        (REPO_BESSER, "Register it with one `GeneratorSpec` in `GENERATORS` of `besser/generators/registry.py` "
                      "(`class_path` as \"module:Class\", `output_type` \"file\" or \"zip\", `file_extension`, "
                      "`download_filename`, `requires_class_diagram` / `required_diagram_type`). The editor's "
                      "`SUPPORTED_GENERATORS` is derived from it; add the id to `WEB_EDITOR_IDS` in "
                      "`tests/generators/test_generator_registry.py` and to "
                      "`tests/generators/snapshots/web_editor_generators.json`."),
        (REPO_BESSER, "Cover regeneration: set `regenerable = True` and add a builder to "
                      "`tests/generators/regenerable_builders.py`, or list the id with a reason in "
                      "`NOT_REGENERABLE` of `tests/generators/test_regenerable_generators.py`."),
        (REPO_BESSER, "Add tests in `tests/generators/<name>/` (`test_*.py`) using the fixtures of "
                      "`tests/conftest.py`; assert the structure and the content of the output."),
        (REPO_BESSER, "Document it in `docs/source/generators/<name>.rst`, listed in a toctree and in the "
                      "\"Choosing a Generator\" table of `docs/source/generators.rst`. Walkthrough: "
                      "`docs/source/generators/build_generator.rst`."),
        (REPO_BESSER, "Only if the Spec-Driven Agent should call it: an `agent_tool` (with its `label`) on the "
                      "registry entry, its id in `AGENT_TOOL_ORDER`, a handler in `ToolExecutor._handlers` "
                      "(`besser/spec_driven_agent/agent/tool_executor.py`), a row in "
                      "`docs/source/spec_driven_agent/tools.rst`, and the tool in `AGENT_TOOLS` of "
                      "`tests/generators/test_generator_registry.py` and in "
                      "`tests/generators/snapshots/agent_generator_tools.json`."),
        (REPO_FRONTEND, "To offer it in the editor's Generate menu: an entry in "
                        "`packages/webapp/src/main/app/shell/menus/generator-menu-config.ts` and the id in the "
                        "`GeneratorType` union of `packages/webapp/src/main/shared/types/workspace-types.ts`."),
    ),
    ChangeKind.METAMODEL: (
        (REPO_BESSER, "Add the concepts under `besser/BUML/metamodel/` in the package of their model family, "
                      "reusing the base classes. Metamodel classes keep private attributes behind validating "
                      "setters that fail fast; `.validate()` collects errors instead of raising."),
        (REPO_BESSER, "Add tests in `tests/BUML/metamodel/`, mirroring the package, for valid and invalid models."),
        (REPO_BESSER, "Document the concepts under `docs/source/buml_language/` (and "
                      "`docs/source/buml_language.rst` for a new model family)."),
        (REPO_BESSER, "Keep the editor converters symmetric: a feature `json_to_buml` supports "
                      "(`besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/`) is "
                      "supported by `buml_to_json` "
                      "(`besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/`) too, "
                      "with a round-trip test (JSON to B-UML to JSON is the identity)."),
        (REPO_BESSER, "If the code a model is written back to changes, update `besser/utilities/buml_code_builder/`."),
    ),
    ChangeKind.DIAGRAM_TYPE: (
        (REPO_BESSER, "If the diagram carries new semantics, extend the metamodel first: "
                      "`besser/BUML/metamodel/`, tests in `tests/BUML/metamodel/`, docs in "
                      "`docs/source/buml_language/`."),
        (REPO_BESSER, "JSON to B-UML: a processor in "
                      "`besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/`, "
                      "registered in its `__init__.py`, re-exported from "
                      "`besser/utilities/web_modeling_editor/backend/services/converters/__init__.py`, and the "
                      "type added to `json_to_buml/project_converter.py`."),
        (REPO_BESSER, "B-UML to JSON: a converter in "
                      "`besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/` and the "
                      "type added to `buml_to_json/project_converter.py`."),
        (REPO_BESSER, "Validation under `besser/utilities/web_modeling_editor/backend/services/validators/`; a "
                      "generator that reads the diagram gets its branch in "
                      "`besser/utilities/web_modeling_editor/backend/services/generation/dispatch.py`; a new "
                      "endpoint uses `@handle_endpoint_errors` from "
                      "`besser/utilities/web_modeling_editor/backend/routers/error_handler.py`."),
        (REPO_FRONTEND, "Editor package: the type in `UMLDiagramType` "
                        "(`packages/editor/src/main/packages/diagram-type.ts`), its element types in "
                        "`packages/editor/src/main/packages/uml-element-type.ts` (relationships in "
                        "`packages/editor/src/main/packages/uml-relationship-type.ts` and "
                        "`packages/editor/src/main/packages/uml-relationships.ts`), a package folder under "
                        "`packages/editor/src/main/packages/` with its palette preview (`<package>/*-preview.ts`, "
                        "as `uml-class-diagram/class-preview.ts`), its elements in the registries "
                        "`components.ts`, `uml-elements.ts` and `popups.ts` there, and its case in the palette "
                        "switch of `packages/editor/src/main/components/create-pane/create-pane.tsx`."),
        (REPO_FRONTEND, "Webapp: `SupportedDiagramType`, `ALL_DIAGRAM_TYPES` and `BesserProject.diagrams` in "
                        "`packages/webapp/src/main/shared/types/project.ts`, with a `PROJECT_SCHEMA_VERSION` bump "
                        "and a migration; the sidebar in "
                        "`packages/webapp/src/main/app/shell/workspace-navigation.tsx`; generate-menu entries in "
                        "`packages/webapp/src/main/app/shell/menus/generator-menu-config.ts`. "
                        "Checklist: `packages/webapp/src/main/features/project/ADDING_NEW_DIAGRAM_TYPE.md`."),
        (REPO_FRONTEND, "Assistant: a converter and a modifier under "
                        "`packages/webapp/src/main/features/assistant/services/` (a converter without a modifier "
                        "silently falls back to the class diagram one)."),
        (REPO_MODELING_AGENT, "Schemas in `src/schemas/<type>.py`, re-exported from `src/schemas/__init__.py`; the "
                              "handler `src/diagram_handlers/types/<type>_diagram_handler.py` with its prompt in "
                              "`src/prompts/<type>.py`; the handler in `HANDLER_CLASSES` of "
                              "`src/diagram_handlers/registry/factory.py`."),
        (REPO_MODELING_AGENT, "Routing: the token in `_TARGET_DIAGRAM_TYPES` of `src/routing/unified_classifier.py` "
                              "and its vocabulary in `SYSTEM_PROMPT` of `src/prompts/classifier.py`; "
                              "`KEYWORD_TARGETS`, an `_IMPLICIT_PATTERNS` regex and `FALLBACK_PRIORITY` in "
                              "`src/orchestrator/workspace_orchestrator.py`."),
        (REPO_MODELING_AGENT, "`SUPPORTED_DIAGRAM_TYPES` in `src/protocol/types.py` (an unknown type is silently "
                              "treated as a class diagram), `DIAGRAM_TYPE_METADATA` in "
                              "`src/diagram_handlers/registry/metadata.py`, suggestions in "
                              "`src/conversation/suggestions.py`, quick responses in `src/app/state_bodies.py` and "
                              "`src/prompts/fallback.py`."),
        (REPO_MODELING_AGENT, "Tests: `tests/test_diagram_handlers.py` plus a `tests/test_<type>.py`, cases in "
                              "`tests/test_handler_envelopes.py`; a changed prompt updates its hash in "
                              "`tests/test_prompt_snapshots.py`. Walkthrough: "
                              "`docs/source/contributing/howto_guides.rst`."),
    ),
    ChangeKind.EDITOR_ELEMENT: (
        (REPO_FRONTEND, "Implement the element in its diagram package under `packages/editor/src/main/packages/` "
                        "(e.g. `uml-class-diagram/<element>/`): a model class extending `UMLElement` or "
                        "`UMLRelationship`, a React component and an update function; its type in "
                        "`packages/editor/src/main/packages/uml-element-type.ts`."),
        (REPO_FRONTEND, "Register it in `packages/editor/src/main/packages/`: `components.ts` (render component), "
                        "`uml-elements.ts` (model class) and `popups.ts` (property popup); its palette entry goes "
                        "in the diagram package's preview, e.g. `uml-class-diagram/class-preview.ts` "
                        "(`compose-preview.ts` only declares the preview types)."),
        (REPO_FRONTEND, "Draw with the themed wrappers of "
                        "`packages/editor/src/main/components/theme/themedComponents.ts`; a raw `<rect>`, "
                        "`<polygon>` or `<ellipse>` needs colour fallbacks."),
        (REPO_FRONTEND, "Strings go in `packages/i18n/en/editor.json`. The editor package has no test script: "
                        "cover the element through webapp tests in `__tests__/` folders under `packages/webapp/src/`."),
        (REPO_BESSER, "If the backend must read the element: the diagram's processor in "
                      "`besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/` and its "
                      "converter in `besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/`, "
                      "both ways, with a round-trip test."),
    ),
    ChangeKind.BACKEND: (
        (REPO_BESSER, "Routers in `besser/utilities/web_modeling_editor/backend/routers/` stay thin: parse the "
                      "request, call a service, map errors with `@handle_endpoint_errors`. The work goes in "
                      "`besser/utilities/web_modeling_editor/backend/services/`, request and response models in "
                      "`besser/utilities/web_modeling_editor/backend/models/`; a service never imports a "
                      "router. Every route carries the `/besser_api` prefix; a new router is registered in "
                      "`besser/utilities/web_modeling_editor/backend/backend.py`."),
        (REPO_BESSER, "A setting is a field of `BackendSettings` in "
                      "`besser/utilities/web_modeling_editor/backend/settings.py` with a `validation_alias`, and "
                      "a row in `ROWS` of `tests/utilities/web_modeling_editor/backend/test_settings.py`."),
        (REPO_BESSER, "Tests under `tests/utilities/web_modeling_editor/backend/`; endpoint and environment "
                      "variable tables in `docs/source/web_editor_backend.rst`."),
        (REPO_FRONTEND, "A change to a route, a model or a docstring changes the API schema: regenerate "
                        "`packages/webapp/openapi/besser-openapi.json` and "
                        "`packages/webapp/src/main/shared/api/generated/besser-openapi.d.ts` with "
                        "`npm run api:generate --workspace=webapp` (BESSER's `.github/scripts/check_openapi.py` "
                        "fails otherwise), and mirror the call in `packages/webapp/src/main/shared/api/`."),
    ),
    ChangeKind.AGENT: (
        (REPO_MODELING_AGENT, "Routing: the rulebook is `SYSTEM_PROMPT` in `src/prompts/classifier.py`; a "
                              "deterministic rule is a pure function in `src/routing/guards.py` with a row in "
                              "`tests/test_routing_guards.py`."),
        (REPO_MODELING_AGENT, "Prompt text lives in `src/prompts/`, never at a call site; a changed prompt updates "
                              "its hash in `tests/test_prompt_snapshots.py`."),
        (REPO_MODELING_AGENT, "Every LLM call goes through `LLMProvider` (`src/llm/provider.py`); action names are "
                              "`Action` members (`src/protocol/actions.py`); keep the import layers "
                              "(`lint-imports`)."),
        (REPO_MODELING_AGENT, "Tests in `tests/` (`tests/conftest.py` has `FakeSession` and `FakeLLM`); docs in "
                              "`docs/source/`."),
        (REPO_BESSER, "A new Spec-Driven Agent tool needs three edits: `besser/spec_driven_agent/agent/tools.py`, "
                      "`_TOOL_MODEL_REQUIREMENTS` in the same file, and a handler in `ToolExecutor._handlers` "
                      "(`besser/spec_driven_agent/agent/tool_executor.py`); tools, caps and configuration are "
                      "documented in `docs/source/spec_driven_agent/`."),
    ),
    ChangeKind.BUGFIX: (
        (_ALL, "Reproduce the defect first: a test that fails on the current code for the reported reason."),
        (_ALL, "Fix the root cause in the layer that owns it, keep the change to what the fix needs, and keep the "
               "test."),
    ),
    ChangeKind.OTHER: (
        (_ALL, "Find the layer that owns the change (BESSER `CLAUDE.md` \"Where Things Live\", the editor's and the "
               "agent's `CLAUDE.md`) and follow its conventions; reuse an existing helper before writing one."),
    ),
}

_EVERY_CHANGE: tuple[str, ...] = (
    "Read the `CLAUDE.md` of each repository before changing it.",
    "Source changes come with tests: BESSER's `tests/` mirrors the source tree, the editor keeps tests in "
    "`__tests__/` folders next to the code, the modeling agent in `tests/`.",
    "Keep `docs/source/` in step with public changes (BESSER `CLAUDE.md` \"Documentation Sync\").",
    "User-visible editor strings go through `t('...')` with the English text in `packages/i18n/en/webapp.json` "
    "(the editor package's `translate('...')` in `packages/i18n/en/editor.json`).",
    "Never change `.github/workflows/`: the sign-in token cannot push it.",
)


def guidance(kind: ChangeKind) -> str:
    """The steps of ``kind`` as the repositories' own guides state them, grouped by repository."""
    lines = [f"Steps for {kind.value.replace('_', ' ')} changes (from {_SOURCES[kind]}):"]
    steps = _STEPS[kind]
    for repo in (_ALL, REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT):
        texts = [text for step_repo, text in steps if step_repo == repo]
        if not texts:
            continue
        if repo != _ALL:
            lines.append(f"{_REPO_TITLES[repo]} (`{repo}`):")
        lines.extend(f"- {text}" for text in texts)
    lines.append("Every change:")
    lines.extend(f"- {text}" for text in _EVERY_CHANGE)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Checklist
# --------------------------------------------------------------------------------------------


class _Tree:
    """The changed files and the working tree of each repository, as the rules read them."""

    def __init__(self, changes: Sequence[RepoChange], repos: Mapping[str, RepoSpec]):
        self.repos = repos
        self._files: dict[str, dict[str, str]] = {}
        for change in changes:
            self._files.setdefault(change.repo, {}).update({f.path: f.status for f in change.files})

    def available(self, repo: str) -> bool:
        spec = self.repos.get(repo)
        return spec is not None and os.path.isdir(spec.path)

    def changed(self, repo: str, prefix: str = "") -> list[str]:
        """Changed paths of ``repo`` under ``prefix``, deletions included."""
        return sorted(p for p in self._files.get(repo, {}) if p.startswith(prefix))

    def status(self, repo: str, path: str) -> Optional[str]:
        return self._files.get(repo, {}).get(path)

    def all_changed(self) -> list[tuple[str, str]]:
        return [(repo, path) for repo in sorted(self._files) for path in sorted(self._files[repo])]

    def full_path(self, repo: str, rel: str) -> str:
        return os.path.join(self.repos[repo].path, *rel.split("/"))

    def exists(self, repo: str, rel: str) -> bool:
        return self.available(repo) and os.path.exists(self.full_path(repo, rel))

    def read(self, repo: str, rel: str) -> Optional[str]:
        """Text of a file, or None when the repository or the file is missing."""
        if not self.exists(repo, rel):
            return None
        with open(self.full_path(repo, rel), encoding="utf-8", errors="replace") as handle:
            return handle.read()

    def parse(self, repo: str, rel: str) -> Optional[ast.Module]:
        text = self.read(repo, rel)
        if text is None:
            return None
        try:
            return ast.parse(text)
        except SyntaxError:
            return None

    def load_json(self, repo: str, rel: str) -> Any:
        text = self.read(repo, rel)
        if text is None:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None


_Verdict = tuple[Optional[bool], str]


def _not_in_workspace(repo: str) -> _Verdict:
    return None, f"the {repo} repository is not in the workspace"


def _listing(paths: Iterable[str]) -> str:
    paths = list(paths)
    shown = ", ".join(paths[:_MAX_LISTED])
    return shown + (f" and {len(paths) - _MAX_LISTED} more" if len(paths) > _MAX_LISTED else "")


def _changed_rule(tree: _Tree, repo: str, prefixes: Sequence[str]) -> _Verdict:
    """True when every prefix has a changed path under it."""
    if not tree.available(repo):
        return _not_in_workspace(repo)
    found, missing = [], []
    for prefix in prefixes:
        paths = tree.changed(repo, prefix)
        (found if paths else missing).append(paths[0] if paths else prefix)
    if missing:
        return False, f"unchanged: {_listing(missing)}"
    return True, f"changed: {_listing(found)}"


# -- Generator ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _RegistryEntry:
    id: str
    module: str
    tool: Optional[str]


def _callee(node: ast.Call) -> str:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _literal(node: Optional[ast.AST]) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def _registry_entries(module: ast.Module) -> list[_RegistryEntry]:
    entries = []
    for node in ast.walk(module):
        if not (isinstance(node, ast.Call) and _callee(node) == "GeneratorSpec"):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        generator_id, class_path = _literal(keywords.get("id")), _literal(keywords.get("class_path"))
        if not isinstance(generator_id, str) or not isinstance(class_path, str):
            continue
        tool = keywords.get("agent_tool")
        tool_name = None
        if isinstance(tool, ast.Call) and _callee(tool) == "AgentTool":
            tool_name = _literal(next((kw.value for kw in tool.keywords if kw.arg == "name"), None))
        entries.append(_RegistryEntry(generator_id, class_path.partition(":")[0], tool_name))
    return entries


def _module_constants(module: Optional[ast.Module]) -> dict[str, Any]:
    """Top-level ``NAME = <literal>`` assignments of a module."""
    constants: dict[str, Any] = {}
    for node in (module.body if module else ()):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            literal = _literal(value)
            if literal is not None:
                constants[target.id] = literal
    return constants


def _generator_package(tree: _Tree, folders: Sequence[str], holds_module: bool) -> str:
    """The generator package (relative to ``besser/generators/``) of a file in ``folders``.

    It is the innermost folder with an ``__init__.py``, so ``nn/pytorch`` is not ``nn``; ``templates/`` is
    not a package. A module in a folder without one makes that folder the package (a new ``nn/jax/``
    missing its ``__init__.py`` is still ``nn/jax``).
    """
    folders = list(folders)
    if "templates" in folders:
        folders, holds_module = folders[:folders.index("templates")], False
    has_init = [n for n in range(len(folders), 0, -1)
                if tree.exists(REPO_BESSER, _GENERATORS_DIR + "/".join(folders[:n]) + "/__init__.py")]
    depth = len(folders) if holds_module or not has_init else has_init[0]
    return "/".join(folders[:depth])


def _generator_packages(tree: _Tree) -> list[str]:
    """Generator packages this change created, else those it touched."""
    created, touched = set(), set()
    for path in tree.changed(REPO_BESSER, _GENERATORS_DIR):
        folders = path.split("/")[2:-1]
        if not folders or folders[0].startswith(("_", ".")):
            continue
        package = _generator_package(tree, folders, path.endswith(".py"))
        if not package:
            continue
        touched.add(package)
        if path == f"{_GENERATORS_DIR}{package}/__init__.py" and tree.status(REPO_BESSER, path) in ("added",
                                                                                                    "renamed"):
            created.add(package)
    return sorted(created or touched)


def _package_entry(tree: _Tree, registry: Optional[ast.Module], package: str) -> Optional[_RegistryEntry]:
    """The registry entry whose class lives in generator package ``package``, not in one nested in it."""
    for entry in _registry_entries(registry) if registry else ():
        parts = entry.module.split(".")
        if parts[:2] != ["besser", "generators"] or len(parts) < 3:
            continue
        is_folder = os.path.isdir(tree.full_path(REPO_BESSER, _GENERATORS_DIR + "/".join(parts[2:])))
        folders = parts[2:] if is_folder else parts[2:-1]
        if _generator_package(tree, folders, not is_folder) == package:
            return entry
    return None


class _GeneratorRules:
    """The rules of one generator package, or of one registry entry; each returns a verdict."""

    def __init__(self, tree: _Tree, name: str, registry: Optional[ast.Module], entry: Optional[_RegistryEntry]):
        self.tree = tree
        self.name = name
        self.registry_parsed = registry is not None
        self.entry = entry
        # Tests and docs page are named after the package path, its last folder or the registry id.
        self.names = list(dict.fromkeys([name, name.rsplit("/", 1)[-1]] + ([entry.id] if entry else [])))
        self.module = "besser.generators." + name.replace("/", ".")

    def package(self) -> _Verdict:
        package_dir = _GENERATORS_DIR + self.name
        if not self.tree.exists(REPO_BESSER, f"{package_dir}/__init__.py"):
            return False, f"no {package_dir}/__init__.py"
        modules = [os.path.join(folder, file_name)
                   for folder, _dirs, files in os.walk(self.tree.full_path(REPO_BESSER, package_dir))
                   for file_name in files if file_name.endswith(".py")]
        # A nested generator may implement it through its enclosing package's base (nn/pytorch through nn/).
        parents = self.name.split("/")[:-1]
        for depth in range(len(parents), 0, -1):
            folder = self.tree.full_path(REPO_BESSER, _GENERATORS_DIR + "/".join(parents[:depth]))
            modules += [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.endswith(".py")]
        for module in modules:
            with open(module, encoding="utf-8", errors="replace") as handle:
                if "GeneratorInterface" in handle.read():
                    return True, f"{package_dir}/ implements GeneratorInterface"
        return False, f"no module in {package_dir}/ mentions GeneratorInterface"

    def registered(self) -> _Verdict:
        if not self.registry_parsed:
            return None, f"{_REGISTRY} is missing or does not parse"
        if self.entry is None:
            return False, f"no GeneratorSpec with a class_path in {self.module}"
        return True, f"GeneratorSpec id {self.entry.id!r}"

    def _needs_entry(self) -> Optional[_Verdict]:
        if self.entry is None:
            return None, "no registry entry to look up"
        return None

    def pinned(self) -> _Verdict:
        if (blocked := self._needs_entry()) is not None:
            return blocked
        pinned_ids = _module_constants(self.tree.parse(REPO_BESSER, _REGISTRY_TEST)).get("WEB_EDITOR_IDS")
        snapshot = self.tree.load_json(REPO_BESSER, _WEB_EDITOR_SNAPSHOT)
        if not isinstance(pinned_ids, list) or not isinstance(snapshot, dict):
            return None, f"cannot read WEB_EDITOR_IDS of {_REGISTRY_TEST} or {_WEB_EDITOR_SNAPSHOT}"
        missing = [where for where, ids in (("WEB_EDITOR_IDS", pinned_ids),
                                            (_WEB_EDITOR_SNAPSHOT, snapshot.get("supported_generators", ())))
                   if self.entry.id not in ids]
        return (False, f"{self.entry.id!r} missing from {_listing(missing)}") if missing else (
            True, f"{self.entry.id!r} pinned in WEB_EDITOR_IDS and the snapshot")

    def regeneration(self) -> _Verdict:
        if (blocked := self._needs_entry()) is not None:
            return blocked
        key = re.compile(rf"""["']{re.escape(self.entry.id)}["']\s*[:\]]""")
        for rel in (_REGENERABLE_BUILDERS, _REGENERABLE_TEST):
            if key.search(self.tree.read(REPO_BESSER, rel) or ""):
                return True, f"{self.entry.id!r} listed in {rel}"
        return False, f"{self.entry.id!r} in neither BUILDERS nor NOT_REGENERABLE"

    def tests(self) -> _Verdict:
        for candidate in self.names:
            folder = f"tests/generators/{candidate}"
            if self.tree.exists(REPO_BESSER, folder):
                found = [f for f in os.listdir(self.tree.full_path(REPO_BESSER, folder))
                         if f.startswith("test_") and f.endswith(".py")]
                if found:
                    return True, f"{folder}/{found[0]}"
        if "/" in self.name:
            # A nested generator may share its parent's tests (tests/generators/nn/ tests nn/pytorch).
            parent = "tests/generators/" + self.name.rsplit("/", 1)[0]
            imports = re.compile(rf"\b{re.escape(self.module)}\b")
            if self.tree.exists(REPO_BESSER, parent):
                for file_name in sorted(os.listdir(self.tree.full_path(REPO_BESSER, parent))):
                    if (file_name.startswith("test_") and file_name.endswith(".py")
                            and imports.search(self.tree.read(REPO_BESSER, f"{parent}/{file_name}") or "")):
                        return True, f"{parent}/{file_name} imports {self.module}"
        return False, "no test_*.py in " + " or ".join(f"tests/generators/{n}/" for n in self.names)

    def _doc_page(self) -> Optional[str]:
        return next((n for n in self.names if self.tree.exists(REPO_BESSER, f"docs/source/generators/{n}.rst")),
                    None)

    def docs_page(self) -> _Verdict:
        page = self._doc_page()
        if page is None:
            return False, "no " + " or ".join(f"docs/source/generators/{n}.rst" for n in self.names)
        return True, f"docs/source/generators/{page}.rst"

    def docs_toctree(self) -> _Verdict:
        text = self.tree.read(REPO_BESSER, _GENERATORS_DOC)
        if text is None:
            return None, f"no {_GENERATORS_DOC}"
        entries = _toctree_entries(text)
        listed = [n for n in self.names if f"generators/{n}" in entries]
        if listed:
            return True, f"generators/{listed[0]} is in a toctree"
        return False, f"no toctree of {_GENERATORS_DOC} lists " + " or ".join(f"generators/{n}" for n in self.names)

    def docs_table(self) -> _Verdict:
        if not self.tree.available(REPO_BESSER):
            return _not_in_workspace(REPO_BESSER)
        if self.tree.status(REPO_BESSER, _GENERATORS_DOC) is None:
            return False, f"{_GENERATORS_DOC} is unchanged, so its table has no new row"
        # The table names generators by display name, which no file maps to an id.
        return None, f"{_GENERATORS_DOC} changed; check the table row by reading it"

    def agent_tool(self) -> _Verdict:
        entry = self.entry
        order = _module_constants(self.tree.parse(REPO_BESSER, _REGISTRY)).get("AGENT_TOOL_ORDER", ())
        pinned = _module_constants(self.tree.parse(REPO_BESSER, _REGISTRY_TEST)).get("AGENT_TOOLS", {})
        snapshot = self.tree.load_json(REPO_BESSER, _AGENT_TOOLS_SNAPSHOT) or {}
        snapshot_names = [t.get("name") for t in snapshot.get("generator_tools", ()) if isinstance(t, dict)]
        handler = re.compile(rf"""["']{re.escape(entry.tool)}["']\s*:""")
        checks = (
            ("AGENT_TOOL_ORDER", entry.id in order),
            ("ToolExecutor._handlers", bool(handler.search(self.tree.read(REPO_BESSER, _TOOL_EXECUTOR) or ""))),
            (_TOOLS_DOC, f"``{entry.tool}``" in (self.tree.read(REPO_BESSER, _TOOLS_DOC) or "")),
            (f"AGENT_TOOLS of {_REGISTRY_TEST}", pinned.get(entry.id) == entry.tool),
            (_AGENT_TOOLS_SNAPSHOT, entry.tool in snapshot_names),
        )
        missing = [where for where, ok in checks if not ok]
        if missing:
            return False, f"{entry.tool} missing from {_listing(missing)}"
        return True, f"{entry.tool} is ordered, handled, documented and pinned"


def _toctree_entries(text: str) -> set[str]:
    """Document names listed in the ``.. toctree::`` directives of an RST text."""
    entries: set[str] = set()
    inside = False
    for line in text.splitlines():
        if line.strip().startswith(".. toctree::"):
            inside = True
            continue
        if not inside:
            continue
        if line.strip() and not line[0].isspace():
            inside = False
        elif line.strip() and not line.strip().startswith(":"):
            entries.add(line.strip())
    return entries


_GENERATOR_RULES: tuple[tuple[str, str, Callable[[_GeneratorRules], _Verdict]], ...] = (
    ("generator.package", "Package with __init__.py under besser/generators/<name>/ implementing GeneratorInterface",
     _GeneratorRules.package),
    ("generator.registry", "A GeneratorSpec for it in besser/generators/registry.py", _GeneratorRules.registered),
    ("generator.registry_pins", "Its id in WEB_EDITOR_IDS and the web editor snapshot of the registry test",
     _GeneratorRules.pinned),
    ("generator.regeneration", "Covered by the regeneration test (a builder or NOT_REGENERABLE)",
     _GeneratorRules.regeneration),
    ("generator.tests", "Tests in tests/generators/<name>/test_*.py", _GeneratorRules.tests),
    ("generator.docs_page", "Docs page docs/source/generators/<name>.rst", _GeneratorRules.docs_page),
    ("generator.docs_toctree", "The page listed in a toctree of docs/source/generators.rst",
     _GeneratorRules.docs_toctree),
    ("generator.docs_table", 'A row in the "Choosing a Generator" table of docs/source/generators.rst',
     _GeneratorRules.docs_table),
)
_AGENT_TOOL_RULE = ("generator.agent_tool",
                    "Agent tool ordered, handled, documented in tools.rst and pinned in the registry test",
                    _GeneratorRules.agent_tool)
# The rules a registry entry answers on its own, without a package of the change.
_ENTRY_RULES = (_GeneratorRules.pinned, _GeneratorRules.regeneration)


def _merge(verdicts: Sequence[tuple[str, _Verdict]]) -> _Verdict:
    """One verdict over several packages: any False wins, then any None."""
    dones = [done for _name, (done, _evidence) in verdicts]
    done = False if False in dones else None if None in dones else True
    if len(verdicts) == 1:
        return done, verdicts[0][1][1]
    return done, "; ".join(f"{name}: {evidence}" for name, (_done, evidence) in verdicts)


def _generator_items(tree: _Tree) -> list[ChecklistItem]:
    if not tree.available(REPO_BESSER):
        return [ChecklistItem(rule_id, text, *_not_in_workspace(REPO_BESSER))
                for rule_id, text, _rule in _GENERATOR_RULES]
    names = _generator_packages(tree)
    registry = tree.parse(REPO_BESSER, _REGISTRY)
    if not names:
        return _registry_items(tree, registry)
    packages = [_GeneratorRules(tree, name, registry, _package_entry(tree, registry, name)) for name in names]
    rules = list(_GENERATOR_RULES)
    if any(p.entry is not None and p.entry.tool for p in packages):
        rules.append(_AGENT_TOOL_RULE)
    items = []
    for rule_id, text, rule in rules:
        applicable = [p for p in packages if rule is not _GeneratorRules.agent_tool or (p.entry and p.entry.tool)]
        items.append(ChecklistItem(rule_id, text, *_merge([(p.name, rule(p)) for p in applicable])))
    return items


def _registry_items(tree: _Tree, registry: Optional[ast.Module]) -> list[ChecklistItem]:
    """A generator change without a package (e.g. offering a generator to the agent): the rules that read
    the registry, judged on every entry; only the entries that fail are named."""
    first_id, first_text = _GENERATOR_RULES[0][:2]
    package = ChecklistItem(first_id, first_text, None, "no generator package changed")
    if registry is None:
        return [package, ChecklistItem("generator.registry", _GENERATOR_RULES[1][1], None,
                                       f"{_REGISTRY} is missing or does not parse")]
    entries = [_GeneratorRules(tree, entry.id, registry, entry) for entry in _registry_entries(registry)]
    items = [package]
    for rule_id, text, rule in [r for r in _GENERATOR_RULES if r[2] in _ENTRY_RULES] + [_AGENT_TOOL_RULE]:
        judged = [(e.entry.id, rule(e)) for e in entries if rule is not _GeneratorRules.agent_tool or e.entry.tool]
        failing = [(entry_id, verdict) for entry_id, verdict in judged if verdict[0] is not True]
        verdict = _merge(failing) if failing else (True, f"all {len(judged)} registry entries")
        items.append(ChecklistItem(rule_id, text, *verdict))
    return items


# -- Metamodel, diagram type, editor element ----------------------------------------------------


def _metamodel_items(tree: _Tree) -> list[ChecklistItem]:
    return [
        ChecklistItem("metamodel.module", "The metamodel changed under besser/BUML/metamodel/",
                      *_changed_rule(tree, REPO_BESSER, ["besser/BUML/metamodel/"])),
        ChecklistItem("metamodel.tests", "Tests under tests/BUML/metamodel/",
                      *_changed_rule(tree, REPO_BESSER, ["tests/BUML/metamodel/"])),
        ChecklistItem("metamodel.docs", "Docs under docs/source/buml_language",
                      *_changed_rule(tree, REPO_BESSER, ["docs/source/buml_language"])),
        ChecklistItem("metamodel.converters", "Editor converters changed in both directions or in neither",
                      *_symmetric_converters(tree)),
    ]


def _symmetric_converters(tree: _Tree) -> _Verdict:
    if not tree.available(REPO_BESSER):
        return _not_in_workspace(REPO_BESSER)
    to_buml, to_json = tree.changed(REPO_BESSER, _JSON_TO_BUML), tree.changed(REPO_BESSER, _BUML_TO_JSON)
    if bool(to_buml) != bool(to_json):
        changed, unchanged = ("json_to_buml", "buml_to_json") if to_buml else ("buml_to_json", "json_to_buml")
        return False, f"{changed} changed but {unchanged} did not"
    return True, "both directions changed" if to_buml else "no converter changed"


def _editor_registries(tree: _Tree, extra: Sequence[str] = ()) -> _Verdict:
    """The registries, the ``extra`` files and at least one diagram package's ``*-preview.ts`` changed."""
    if not tree.available(REPO_FRONTEND):
        return _not_in_workspace(REPO_FRONTEND)
    required = [_EDITOR_PACKAGES + name for name in _EDITOR_REGISTRIES] + list(extra)
    missing = [path for path in required if tree.status(REPO_FRONTEND, path) is None]
    previews = [path for path in tree.changed(REPO_FRONTEND, _EDITOR_PACKAGES) if _EDITOR_PREVIEW.match(path)]
    if not previews:
        missing.append(_EDITOR_PACKAGES + "<package>/*-preview.ts")
    if missing:
        return False, f"unchanged: {_listing(missing)}"
    return True, f"changed: {_listing(required + previews)}"


def _diagram_type_items(tree: _Tree) -> list[ChecklistItem]:
    return [
        ChecklistItem("diagram_type.converters", "Backend converters in both directions (json_to_buml, buml_to_json)",
                      *_changed_rule(tree, REPO_BESSER, [_JSON_TO_BUML, _BUML_TO_JSON])),
        ChecklistItem("diagram_type.project_converters", "Both project converters include the type",
                      *_changed_rule(tree, REPO_BESSER, [_JSON_TO_BUML + "project_converter.py",
                                                         _BUML_TO_JSON + "project_converter.py"])),
        ChecklistItem("diagram_type.editor_type", "The type in the editor's diagram-type.ts",
                      *_changed_rule(tree, REPO_FRONTEND, [_EDITOR_PACKAGES + "diagram-type.ts"])),
        ChecklistItem("diagram_type.editor_registries",
                      "Its elements in components.ts, uml-elements.ts and popups.ts, its palette preview and its "
                      "create-pane.tsx case",
                      *_editor_registries(tree, [_CREATE_PANE])),
        ChecklistItem("diagram_type.project_types", "The type in the webapp's shared/types/project.ts",
                      *_changed_rule(tree, REPO_FRONTEND, ["packages/webapp/src/main/shared/types/project.ts"])),
        ChecklistItem("diagram_type.agent_handler", "A modeling agent handler, registered in factory.py",
                      *_changed_rule(tree, REPO_MODELING_AGENT, [_AGENT_HANDLERS, _AGENT_FACTORY])),
        ChecklistItem("diagram_type.agent_schema", "A modeling agent schema in src/schemas/",
                      *_changed_rule(tree, REPO_MODELING_AGENT, [_AGENT_SCHEMAS])),
        ChecklistItem("diagram_type.agent_supported", "The type in SUPPORTED_DIAGRAM_TYPES (src/protocol/types.py)",
                      *_changed_rule(tree, REPO_MODELING_AGENT, [_AGENT_PROTOCOL_TYPES])),
        ChecklistItem("diagram_type.agent_routing",
                      "The type in _TARGET_DIAGRAM_TYPES (src/routing/unified_classifier.py)",
                      *_changed_rule(tree, REPO_MODELING_AGENT, [_AGENT_CLASSIFIER])),
    ]


def _editor_element_items(tree: _Tree) -> list[ChecklistItem]:
    return [ChecklistItem("editor_element.registries",
                          "Registered in components.ts, uml-elements.ts and popups.ts, with its palette entry in "
                          "its diagram's *-preview.ts",
                          *_editor_registries(tree))]


# -- Every kind ---------------------------------------------------------------------------------


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (bool(_TEST_DIRS.intersection(parts[:-1])) or name.startswith("test_") or name.endswith("_test.py")
            or ".test." in name or ".spec." in name)


def _is_source_path(path: str) -> bool:
    return (path.endswith(_SOURCE_SUFFIXES) and not path.endswith(".d.ts") and not _is_test_path(path)
            and not path.startswith(("docs/", ".github/")))


def _tests_with_source(tree: _Tree) -> _Verdict:
    untested, tested = [], []
    for repo in sorted({repo for repo, path in tree.all_changed() if _is_source_path(path)}):
        paths = tree.changed(repo)
        (tested if any(_is_test_path(p) for p in paths) else untested).append(repo)
    if untested:
        return False, f"source changed without a test change in {_listing(untested)}"
    return True, f"tests changed in {_listing(tested)}" if tested else "no source file changed"


def _no_workflows(tree: _Tree) -> _Verdict:
    touched = [f"{repo}:{path}" for repo, path in tree.all_changed() if path.startswith(".github/workflows/")]
    if touched:
        return False, f"the sign-in token cannot push {_listing(touched)}"
    return True, "no workflow file changed"


def _has_key(strings: Any, key: str) -> bool:
    node = strings
    *parents, last = key.split(".")
    for part in parents:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    # i18next resolves a plural key ``x`` to ``x_one`` / ``x_other``.
    return isinstance(node, dict) and any(k in node for k in (last, f"{last}_one", f"{last}_other"))


def _strings_present(tree: _Tree) -> _Verdict:
    if not tree.available(REPO_FRONTEND):
        if tree.changed(REPO_FRONTEND):
            return _not_in_workspace(REPO_FRONTEND)
        return True, "no editor file changed"
    missing, used = [], 0
    for prefix, pattern, strings_file in ((_WEBAPP_SRC, _WEBAPP_KEY, _WEBAPP_STRINGS),
                                          (_EDITOR_SRC, _EDITOR_KEY, _EDITOR_STRINGS)):
        keys = set()
        for path in tree.changed(REPO_FRONTEND, prefix):
            if path.endswith((".ts", ".tsx")) and not _is_test_path(path):
                keys.update(match.group(2) for match in pattern.finditer(tree.read(REPO_FRONTEND, path) or ""))
        if not keys:
            continue
        strings = tree.load_json(REPO_FRONTEND, strings_file)
        if not isinstance(strings, dict):
            return None, f"cannot read {strings_file}"
        used += len(keys)
        missing.extend(f"{key} ({strings_file})" for key in sorted(keys) if not _has_key(strings, key))
    if missing:
        return False, f"missing from the English strings: {_listing(missing)}"
    return True, f"{used} keys used in changed files, all present" if used else "no translated string in changed files"


def _common_items(tree: _Tree) -> list[ChecklistItem]:
    return [
        ChecklistItem("common.tests", "Tests changed in every repository whose source changed",
                      *_tests_with_source(tree)),
        ChecklistItem("common.no_workflows", "No .github/workflows/ file changed", *_no_workflows(tree)),
        ChecklistItem("common.i18n", "Editor strings used in changed files are in packages/i18n/en/",
                      *_strings_present(tree)),
    ]


_KIND_ITEMS: dict[ChangeKind, Callable[[_Tree], list[ChecklistItem]]] = {
    ChangeKind.GENERATOR: _generator_items,
    ChangeKind.METAMODEL: _metamodel_items,
    ChangeKind.DIAGRAM_TYPE: _diagram_type_items,
    ChangeKind.EDITOR_ELEMENT: _editor_element_items,
}


def evaluate_checklist(kind: ChangeKind, changes: Sequence[RepoChange],
                       repos: Mapping[str, RepoSpec]) -> list[ChecklistItem]:
    """Judge the checklist of ``kind`` against the changed files and the working trees in ``repos``.

    The kind's own rules come first, then the rules of every change. Paths in ``changes`` are relative
    to their repository's root, with ``/`` separators, as git reports them.
    """
    tree = _Tree(changes, repos)
    kind_items = _KIND_ITEMS.get(kind)
    return (kind_items(tree) if kind_items else []) + _common_items(tree)


def format_checklist(items: Sequence[ChecklistItem]) -> str:
    """The checklist as plain text: a count line, then ``[x]`` done, ``[ ]`` missing, ``[?]`` not judged."""
    if not items:
        return "Checklist: nothing to check."
    counts = [sum(1 for item in items if item.done is value) for value in (True, False, None)]
    lines = [f"Checklist: {counts[0]} done, {counts[1]} missing, {counts[2]} not judged"]
    for item in items:
        mark = "[x]" if item.done is True else "[ ]" if item.done is False else "[?]"
        lines.append(f"{mark} {item.text}" + (f" ({item.evidence})" if item.evidence else ""))
    return "\n".join(lines)
