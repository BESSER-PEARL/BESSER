"""Change kinds: the guidance names real paths, and each checklist rule flips when its piece is missing."""

import json
import os
from pathlib import Path

import pytest

from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    ChangeKind,
    ChecklistItem,
    FileChange,
    RepoChange,
    RepoSpec,
)
from besser.spec_driven_agent.lab.kinds import evaluate_checklist, format_checklist, guidance

BESSER_ROOT = Path(__file__).resolve().parents[3]


def _repos(**paths) -> dict[str, RepoSpec]:
    return {key: RepoSpec(key, str(path), "development", "BESSER-PEARL", key) for key, path in paths.items()}


def _change(repo: str, files: dict[str, str]) -> RepoChange:
    return RepoChange(repo, f"lab/jordi/{repo}", "development",
                      tuple(FileChange(path, status) for path, status in files.items()))


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _by_id(items: list[ChecklistItem]) -> dict[str, ChecklistItem]:
    return {item.id: item for item in items}


# -- Guidance ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(ChangeKind), ids=lambda kind: kind.value)
def test_every_kind_has_steps_and_the_rules_of_every_change(kind):
    text = guidance(kind)
    assert text.startswith(f"Steps for {kind.value.replace('_', ' ')} changes")
    assert "Never change `.github/workflows/`" in text
    assert "Read the `CLAUDE.md` of each repository" in text


def test_diagram_type_guidance_covers_all_three_repositories():
    text = guidance(ChangeKind.DIAGRAM_TYPE)
    for heading in ("BESSER (`besser`):", "Web editor (`frontend`):", "Modeling agent (`modeling_agent`):"):
        assert heading in text
    assert "SUPPORTED_DIAGRAM_TYPES" in text and "project_converter.py" in text


def _besser_paths(text: str) -> list[str]:
    """Backticked paths of the BESSER section of a guidance text, cut before any ``<placeholder>``."""
    paths, inside = [], False
    for line in text.splitlines():
        if not line.startswith("- "):
            inside = line == "BESSER (`besser`):"
            continue
        if inside:
            for token in line.split("`")[1::2]:
                # Skipped: a route prefix, and the converter paths written relative to the folder named before.
                if "/" in token and " " not in token and not token.startswith(("/", "json_to_buml/", "buml_to_json/")):
                    paths.append(token.split("<")[0])
    return paths


@pytest.mark.parametrize("kind", list(ChangeKind), ids=lambda kind: kind.value)
def test_besser_paths_in_the_guidance_exist(kind):
    missing = [p for p in _besser_paths(guidance(kind)) if not (BESSER_ROOT / p).exists()]
    assert missing == []


# -- Generator: calibrated on generators that work --------------------------------------------------


def _existing_generator_change(package: str, test_dir: str, doc: str) -> RepoChange:
    """The files of a generator already in this checkout, reported as if a change had added them."""
    files = {}
    for folder in (f"besser/generators/{package}", f"tests/generators/{test_dir}"):
        for path in (BESSER_ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(BESSER_ROOT).as_posix()] = "added"
    files[f"docs/source/generators/{doc}.rst"] = "added"
    files["docs/source/generators.rst"] = "modified"
    files["besser/generators/registry.py"] = "modified"
    return _change(REPO_BESSER, files)


def test_a_registered_generator_with_an_agent_tool_passes_every_rule():
    items = _by_id(evaluate_checklist(ChangeKind.GENERATOR, [_existing_generator_change("rdf", "rdf", "rdf")],
                                      _repos(besser=BESSER_ROOT)))
    assert {item_id: item.done for item_id, item in items.items()} == {
        "generator.package": True,
        "generator.registry": True,
        "generator.registry_pins": True,
        "generator.regeneration": True,
        "generator.tests": True,
        "generator.docs_page": True,
        "generator.docs_toctree": True,
        "generator.docs_table": None,
        "generator.agent_tool": True,
        "common.tests": True,
        "common.no_workflows": True,
        "common.i18n": True,
    }


@pytest.mark.parametrize("package, test_dir, doc", [
    ("nn/pytorch", "nn", "pytorch"),       # nested, tested in its parent's folder
    ("nn/forecaster", "forecaster", "forecaster"),
])
def test_a_nested_generator_is_judged_as_itself_not_its_parent(package, test_dir, doc):
    items = evaluate_checklist(ChangeKind.GENERATOR, [_existing_generator_change(package, test_dir, doc)],
                               _repos(besser=BESSER_ROOT))
    assert [(i.id, i.evidence) for i in items if i.id.startswith("generator.") and i.done is not True
            and i.id != "generator.docs_table"] == []
    assert _by_id(items)["generator.registry"].evidence == f"GeneratorSpec id {doc!r}"


def test_a_generator_without_an_agent_tool_gets_no_agent_tool_rule():
    items = _by_id(evaluate_checklist(ChangeKind.GENERATOR, [_existing_generator_change("alloy", "alloy", "alloy")],
                                      _repos(besser=BESSER_ROOT)))
    assert "generator.agent_tool" not in items
    assert items["generator.registry"].evidence == "GeneratorSpec id 'alloy'"


# -- Generator: a small checkout where one piece at a time is taken away ----------------------------

_REGISTRY = '''
GENERATORS = (
    GeneratorSpec(id="python", class_path="besser.generators.python_classes:PythonGenerator"),
    GeneratorSpec(
        id="notes",
        class_path="besser.generators.notes:NotesGenerator",
        agent_tool=AgentTool(name="generate_notes", label="notes"),
    ),
)
AGENT_TOOL_ORDER: tuple[str, ...] = ("notes",)
'''
_GENERATORS_RST = """Code Generators
===============

.. toctree::
   :maxdepth: 1

   generators/python
   generators/notes

Writing one
-----------
"""


def _generator_checkout(root: Path) -> RepoChange:
    _write(root, "besser/generators/notes/__init__.py", "from .notes_generator import NotesGenerator\n")
    _write(root, "besser/generators/notes/notes_generator.py", "class NotesGenerator(GeneratorInterface):\n    pass\n")
    _write(root, "besser/generators/registry.py", _REGISTRY)
    _write(root, "tests/generators/notes/test_notes.py")
    _write(root, "tests/generators/test_generator_registry.py",
           'WEB_EDITOR_IDS = ["python", "notes"]\nAGENT_TOOLS = {"notes": "generate_notes"}\n')
    _write(root, "tests/generators/snapshots/web_editor_generators.json",
           json.dumps({"supported_generators": ["python", "notes"]}))
    _write(root, "tests/generators/snapshots/agent_generator_tools.json",
           json.dumps({"generator_tools": [{"name": "generate_notes"}]}))
    _write(root, "tests/generators/regenerable_builders.py", 'BUILDERS = {\n    "notes": _on_library(),\n}\n')
    _write(root, "tests/generators/test_regenerable_generators.py", 'NOT_REGENERABLE = {"terraform": "sets"}\n')
    _write(root, "besser/spec_driven_agent/agent/tool_executor.py", '_handlers = {\n    "generate_notes": _gen,\n}\n')
    _write(root, "docs/source/spec_driven_agent/tools.rst", "   * - ``generate_notes``\n")
    _write(root, "docs/source/generators.rst", _GENERATORS_RST)
    _write(root, "docs/source/generators/notes.rst", "Notes Generator\n")
    return _change(REPO_BESSER, {
        "besser/generators/notes/__init__.py": "added",
        "besser/generators/notes/notes_generator.py": "added",
        "besser/generators/registry.py": "modified",
        "tests/generators/notes/test_notes.py": "added",
        "docs/source/generators.rst": "modified",
        "docs/source/generators/notes.rst": "added",
    })


def _generator_items(root: Path, change: RepoChange) -> dict[str, ChecklistItem]:
    return _by_id(evaluate_checklist(ChangeKind.GENERATOR, [change], _repos(besser=root)))


def test_a_complete_generator_checkout_passes(tmp_path):
    items = _generator_items(tmp_path, _generator_checkout(tmp_path))
    assert [i for i in items.values() if i.done is not True and i.id != "generator.docs_table"] == []
    assert items["generator.docs_table"].done is None


def _replace(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


@pytest.mark.parametrize("breaks, rule_id, evidence", [
    (lambda r: (r / "besser/generators/notes/__init__.py").unlink(), "generator.package", "__init__.py"),
    (lambda r: _replace(r, "besser/generators/notes/notes_generator.py", "(GeneratorInterface)", ""),
     "generator.package", "GeneratorInterface"),
    (lambda r: _replace(r, "besser/generators/registry.py", "besser.generators.notes:", "besser.generators.other:"),
     "generator.registry", "besser.generators.notes"),
    (lambda r: _replace(r, "tests/generators/test_generator_registry.py", '"python", "notes"', '"python"'),
     "generator.registry_pins", "WEB_EDITOR_IDS"),
    (lambda r: _replace(r, "tests/generators/snapshots/web_editor_generators.json", ', "notes"', ""),
     "generator.registry_pins", "web_editor_generators.json"),
    (lambda r: _replace(r, "tests/generators/regenerable_builders.py", '"notes"', '"other"'),
     "generator.regeneration", "NOT_REGENERABLE"),
    (lambda r: (r / "tests/generators/notes/test_notes.py").unlink(), "generator.tests", "tests/generators/notes/"),
    (lambda r: (r / "docs/source/generators/notes.rst").unlink(), "generator.docs_page", "notes.rst"),
    (lambda r: _replace(r, "docs/source/generators.rst", "   generators/notes\n", ""),
     "generator.docs_toctree", "generators/notes"),
    (lambda r: _replace(r, "besser/generators/registry.py", '("notes",)', "()"),
     "generator.agent_tool", "AGENT_TOOL_ORDER"),
    (lambda r: _replace(r, "besser/spec_driven_agent/agent/tool_executor.py", "generate_notes", "other"),
     "generator.agent_tool", "ToolExecutor._handlers"),
    (lambda r: _replace(r, "docs/source/spec_driven_agent/tools.rst", "generate_notes", "other"),
     "generator.agent_tool", "tools.rst"),
    (lambda r: _replace(r, "tests/generators/test_generator_registry.py", '{"notes": "generate_notes"}', "{}"),
     "generator.agent_tool", "AGENT_TOOLS"),
    (lambda r: _replace(r, "tests/generators/snapshots/agent_generator_tools.json", "generate_notes", "x"),
     "generator.agent_tool", "agent_generator_tools.json"),
])
def test_each_missing_piece_fails_its_own_rule_and_only_it(tmp_path, breaks, rule_id, evidence):
    change = _generator_checkout(tmp_path)
    breaks(tmp_path)
    items = _generator_items(tmp_path, change)
    assert items[rule_id].done is False
    assert evidence in items[rule_id].evidence
    assert {i.id for i in items.values() if i.done is False} == {rule_id}
    if rule_id == "generator.registry":
        # Without an entry the rules that look the entry up cannot judge, and the agent tool rule goes.
        assert items["generator.registry_pins"].done is None
        assert "generator.agent_tool" not in items


def test_an_unchanged_generators_page_cannot_have_gained_a_table_row(tmp_path):
    change = _generator_checkout(tmp_path)
    files = tuple(f for f in change.files if f.path != "docs/source/generators.rst")
    items = _generator_items(tmp_path, RepoChange(change.repo, change.branch, change.base, files))
    assert items["generator.docs_table"].done is False


def test_a_registry_that_does_not_parse_is_not_judged(tmp_path):
    change = _generator_checkout(tmp_path)
    _write(tmp_path, "besser/generators/registry.py", "GENERATORS = (\n")
    items = _generator_items(tmp_path, change)
    assert items["generator.registry"].done is None
    assert "does not parse" in items["generator.registry"].evidence


def test_a_new_generator_nested_in_nn_is_not_credited_with_a_sibling(tmp_path):
    # nn/ holds the shared base; nn/pytorch is a registered, tested, documented sibling.
    _write(tmp_path, "besser/generators/nn/__init__.py")
    _write(tmp_path, "besser/generators/nn/nn_code_generator.py", "class NNCodeGenerator(GeneratorInterface):\n")
    _write(tmp_path, "besser/generators/nn/pytorch/__init__.py")
    _write(tmp_path, "besser/generators/registry.py", 'GENERATORS = (GeneratorSpec(id="pytorch", class_path='
           '"besser.generators.nn.pytorch.pytorch_code_generator:PytorchGenerator"),)\n')
    _write(tmp_path, "tests/generators/nn/test_torch.py", "from besser.generators.nn.pytorch import x\n")
    _write(tmp_path, "docs/source/generators/pytorch.rst")
    _write(tmp_path, "docs/source/generators.rst", ".. toctree::\n\n   generators/pytorch\n")
    _write(tmp_path, "besser/generators/nn/jax/__init__.py")
    _write(tmp_path, "besser/generators/nn/jax/jax_generator.py", "class JaxGenerator(NNCodeGenerator):\n")
    change = _change(REPO_BESSER, {"besser/generators/nn/jax/__init__.py": "added",
                                   "besser/generators/nn/jax/jax_generator.py": "added"})

    items = _generator_items(tmp_path, change)

    assert items["generator.package"].done is True  # through the nn/ base
    assert items["generator.registry"].done is False
    assert items["generator.registry"].evidence == "no GeneratorSpec with a class_path in besser.generators.nn.jax"
    assert items["generator.tests"].done is False
    assert items["generator.docs_page"].done is False
    assert "generator.agent_tool" not in items


def test_a_nested_package_missing_its_init_is_reported_as_itself(tmp_path):
    _write(tmp_path, "besser/generators/nn/__init__.py")
    _write(tmp_path, "besser/generators/nn/jax/jax_generator.py")
    items = _generator_items(tmp_path, _change(REPO_BESSER, {"besser/generators/nn/jax/jax_generator.py": "added"}))
    assert items["generator.package"].evidence == "no besser/generators/nn/jax/__init__.py"


_AGENT_TOOL_CHANGE = {  # offering an existing generator to the agent adds no package (CLAUDE.md step 5)
    "besser/generators/registry.py": "modified",
    "besser/spec_driven_agent/agent/tool_executor.py": "modified",
    "docs/source/spec_driven_agent/tools.rst": "modified",
    "tests/generators/test_generator_registry.py": "modified",
    "tests/generators/snapshots/agent_generator_tools.json": "modified",
}


def test_a_generator_change_without_a_package_judges_the_registry_entries():
    items = _by_id(evaluate_checklist(ChangeKind.GENERATOR, [_change(REPO_BESSER, _AGENT_TOOL_CHANGE)],
                                      _repos(besser=BESSER_ROOT)))
    assert items["generator.package"].done is None
    assert [i.id for i in items.values() if i.id.startswith("generator.")] == [
        "generator.package", "generator.registry_pins", "generator.regeneration", "generator.agent_tool"]
    assert all(items[i].done is True for i in ("generator.registry_pins", "generator.regeneration",
                                                "generator.agent_tool"))


def test_an_incomplete_agent_tool_without_a_package_is_caught(tmp_path):
    _generator_checkout(tmp_path)
    _replace(tmp_path, "docs/source/spec_driven_agent/tools.rst", "generate_notes", "other")
    items = _generator_items(tmp_path, _change(REPO_BESSER, _AGENT_TOOL_CHANGE))
    assert items["generator.agent_tool"].done is False
    assert items["generator.agent_tool"].evidence == (
        "generate_notes missing from docs/source/spec_driven_agent/tools.rst")


def test_generator_rules_without_the_besser_repository_are_not_judged(tmp_path):
    items = evaluate_checklist(ChangeKind.GENERATOR, [], _repos(frontend=tmp_path))
    generator_items = [i for i in items if i.id.startswith("generator.")]
    assert generator_items and all(i.done is None for i in generator_items)


# -- Metamodel, diagram type, editor element --------------------------------------------------------

_JSON_TO_BUML = "besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/"
_BUML_TO_JSON = "besser/utilities/web_modeling_editor/backend/services/converters/buml_to_json/"


@pytest.mark.parametrize("converters, symmetric", [
    ({}, True),
    ({_JSON_TO_BUML + "class_diagram_processor.py": "modified"}, False),
    ({_BUML_TO_JSON + "class_diagram_converter.py": "modified"}, False),
    ({_JSON_TO_BUML + "class_diagram_processor.py": "modified",
      _BUML_TO_JSON + "class_diagram_converter.py": "modified"}, True),
])
def test_metamodel_converters_change_in_both_directions_or_neither(tmp_path, converters, symmetric):
    files = {"besser/BUML/metamodel/structural/structural.py": "modified",
             "tests/BUML/metamodel/structural/test_note.py": "added",
             "docs/source/buml_language/model_types/structural.rst": "modified", **converters}
    items = _by_id(evaluate_checklist(ChangeKind.METAMODEL, [_change(REPO_BESSER, files)],
                                      _repos(besser=tmp_path)))
    assert items["metamodel.converters"].done is symmetric
    assert all(items[i].done for i in ("metamodel.module", "metamodel.tests", "metamodel.docs"))


def test_a_metamodel_change_without_tests_or_docs_fails_those_rules(tmp_path):
    change = _change(REPO_BESSER, {"besser/BUML/metamodel/structural/structural.py": "modified"})
    items = _by_id(evaluate_checklist(ChangeKind.METAMODEL, [change], _repos(besser=tmp_path)))
    assert items["metamodel.tests"].done is False and items["metamodel.docs"].done is False
    assert items["common.tests"].done is False


_DIAGRAM_TYPE_CHANGES = {
    REPO_BESSER: {_JSON_TO_BUML + "note_diagram_processor.py": "added",
                  _JSON_TO_BUML + "project_converter.py": "modified",
                  _BUML_TO_JSON + "note_diagram_converter.py": "added",
                  _BUML_TO_JSON + "project_converter.py": "modified",
                  "tests/utilities/web_modeling_editor/backend/test_note.py": "added"},
    REPO_FRONTEND: {"packages/editor/src/main/packages/diagram-type.ts": "modified",
                    **{f"packages/editor/src/main/packages/{n}": "modified"
                       for n in ("components.ts", "uml-elements.ts", "popups.ts", "note-diagram/note-preview.ts")},
                    "packages/editor/src/main/components/create-pane/create-pane.tsx": "modified",
                    "packages/webapp/src/main/shared/types/project.ts": "modified",
                    "packages/webapp/src/main/shared/types/__tests__/project.test.ts": "modified"},
    REPO_MODELING_AGENT: {"src/diagram_handlers/types/note_diagram_handler.py": "added",
                          "src/diagram_handlers/registry/factory.py": "modified",
                          "src/schemas/note.py": "added",
                          "src/protocol/types.py": "modified",
                          "src/routing/unified_classifier.py": "modified",
                          "tests/test_note.py": "added"},
}


def test_a_complete_diagram_type_across_three_repositories_passes(tmp_path):
    repos = _repos(**{key: tmp_path / key for key in _DIAGRAM_TYPE_CHANGES})
    for spec in repos.values():
        os.makedirs(spec.path)
    changes = [_change(repo, files) for repo, files in _DIAGRAM_TYPE_CHANGES.items()]
    items = evaluate_checklist(ChangeKind.DIAGRAM_TYPE, changes, repos)
    assert [(i.id, i.evidence) for i in items if i.done is not True] == []


def test_diagram_type_rules_of_a_repository_outside_the_workspace_are_not_judged(tmp_path):
    repos = _repos(besser=tmp_path, frontend=tmp_path)
    changes = [_change(REPO_BESSER, _DIAGRAM_TYPE_CHANGES[REPO_BESSER]),
               _change(REPO_FRONTEND, _DIAGRAM_TYPE_CHANGES[REPO_FRONTEND])]
    items = evaluate_checklist(ChangeKind.DIAGRAM_TYPE, changes, repos)
    agent_items = [i for i in items if i.id.startswith("diagram_type.agent_")]
    assert len(agent_items) == 4
    assert all(i.done is None and "modeling_agent repository is not in the workspace" in i.evidence
               for i in agent_items)


# The registries the Comments element (editor commit 6d4abb4f) changed; its palette entry is in class-preview.ts.
_COMMENTS_ELEMENT = {f"packages/editor/src/main/packages/{n}": "modified"
                     for n in ("components.ts", "uml-elements.ts", "popups.ts", "uml-class-diagram/class-preview.ts")}


def test_an_editor_element_registered_as_the_editor_does_passes(tmp_path):
    items = _by_id(evaluate_checklist(ChangeKind.EDITOR_ELEMENT, [_change(REPO_FRONTEND, _COMMENTS_ELEMENT)],
                                      _repos(frontend=tmp_path)))
    assert items["editor_element.registries"].done is True


@pytest.mark.parametrize("drop, evidence", [
    ("packages/editor/src/main/packages/popups.ts", "unchanged: packages/editor/src/main/packages/popups.ts"),
    # compose-preview.ts only declares the preview types: it is no palette entry.
    ("packages/editor/src/main/packages/uml-class-diagram/class-preview.ts",
     "unchanged: packages/editor/src/main/packages/<package>/*-preview.ts"),
])
def test_an_editor_element_missing_a_registry_or_its_palette_entry_fails(tmp_path, drop, evidence):
    files = {**{p: s for p, s in _COMMENTS_ELEMENT.items() if p != drop},
             "packages/editor/src/main/packages/compose-preview.ts": "modified"}
    items = _by_id(evaluate_checklist(ChangeKind.EDITOR_ELEMENT, [_change(REPO_FRONTEND, files)],
                                      _repos(frontend=tmp_path)))
    assert items["editor_element.registries"].done is False
    assert items["editor_element.registries"].evidence == evidence


def test_a_diagram_type_needs_its_create_pane_case(tmp_path):
    files = {p: s for p, s in _DIAGRAM_TYPE_CHANGES[REPO_FRONTEND].items() if not p.endswith("create-pane.tsx")}
    items = _by_id(evaluate_checklist(ChangeKind.DIAGRAM_TYPE, [_change(REPO_FRONTEND, files)],
                                      _repos(frontend=tmp_path)))
    assert items["diagram_type.editor_registries"].done is False
    assert "create-pane.tsx" in items["diagram_type.editor_registries"].evidence


# -- Rules of every change --------------------------------------------------------------------------


def test_a_bugfix_gets_only_the_rules_of_every_change(tmp_path):
    items = evaluate_checklist(ChangeKind.BUGFIX, [], _repos(besser=tmp_path))
    assert [i.id for i in items] == ["common.tests", "common.no_workflows", "common.i18n"]
    assert all(i.done for i in items)


def test_a_workflow_change_fails_because_the_token_cannot_push_it(tmp_path):
    change = _change(REPO_FRONTEND, {".github/workflows/ci.yml": "modified"})
    items = _by_id(evaluate_checklist(ChangeKind.OTHER, [change], _repos(frontend=tmp_path)))
    assert items["common.no_workflows"].done is False
    assert "frontend:.github/workflows/ci.yml" in items["common.no_workflows"].evidence


def test_tests_must_change_in_each_repository_whose_source_changed(tmp_path):
    changes = [_change(REPO_BESSER, {"besser/generators/sql/sql_generator.py": "modified",
                                     "tests/generators/sql/test_sql.py": "modified"}),
               _change(REPO_FRONTEND, {"packages/webapp/src/main/features/lab/LabPage.tsx": "added",
                                       "packages/i18n/en/webapp.json": "modified"})]
    items = _by_id(evaluate_checklist(ChangeKind.OTHER, changes, _repos(besser=tmp_path)))
    assert items["common.tests"].done is False
    assert items["common.tests"].evidence == "source changed without a test change in frontend"


def test_generated_api_types_and_docs_are_not_source_that_needs_tests(tmp_path):
    change = _change(REPO_FRONTEND, {"packages/webapp/src/main/shared/api/generated/besser-openapi.d.ts": "modified",
                                     "docs/source/conf.py": "modified"})
    items = _by_id(evaluate_checklist(ChangeKind.OTHER, [change], _repos(besser=tmp_path)))
    assert items["common.tests"].done is True


def _frontend_with_strings(root: Path) -> None:
    strings = {"lab": {"title": "BESSER Lab", "files_one": "{{count}} file", "files_other": "{{count}} files"}}
    _write(root, "packages/i18n/en/webapp.json", json.dumps(strings))
    _write(root, "packages/i18n/en/editor.json", json.dumps({"popup": {"name": "Name"}}))
    _write(root, "packages/editor/src/main/packages/note/note-update.tsx", "this.props.translate('popup.name')\n")
    _write(root, "packages/webapp/src/main/features/lab/__tests__/LabPage.test.tsx", "i18n.t('only.inATest')\n")


@pytest.mark.parametrize("source, missing", [
    ("t('lab.title'); t(\"lab.files\", { count });", None),
    ("t('lab.title'); i18n.t('lab.state.parked', 'Parked'); t(`lab.${key}`)", "lab.state.parked"),
])
def test_editor_strings_used_in_changed_files_must_be_in_the_english_bundle(tmp_path, source, missing):
    _frontend_with_strings(tmp_path)
    _write(tmp_path, "packages/webapp/src/main/features/lab/LabPage.tsx", source)
    change = _change(REPO_FRONTEND, {"packages/webapp/src/main/features/lab/LabPage.tsx": "added",
                                     "packages/webapp/src/main/features/lab/__tests__/LabPage.test.tsx": "added",
                                     "packages/editor/src/main/packages/note/note-update.tsx": "added"})
    item = _by_id(evaluate_checklist(ChangeKind.OTHER, [change], _repos(frontend=tmp_path)))["common.i18n"]
    if missing is None:
        assert item.done is True and item.evidence == "3 keys used in changed files, all present"
    else:
        assert item.done is False
        assert item.evidence == f"missing from the English strings: {missing} (packages/i18n/en/webapp.json)"


def test_editor_strings_of_an_editor_outside_the_workspace_are_not_judged(tmp_path):
    change = _change(REPO_FRONTEND, {"packages/webapp/src/main/features/lab/LabPage.tsx": "added"})
    item = _by_id(evaluate_checklist(ChangeKind.OTHER, [change], _repos(besser=tmp_path)))["common.i18n"]
    assert item.done is None


# -- Formatting -------------------------------------------------------------------------------------


def test_format_checklist_marks_done_missing_and_not_judged():
    items = [ChecklistItem("a", "Registered", True, "GeneratorSpec id 'notes'"),
             ChecklistItem("b", "Tests", False, "no test_*.py"),
             ChecklistItem("c", "Table row", None)]
    assert format_checklist(items) == (
        "Checklist: 1 done, 1 missing, 1 not judged\n"
        "[x] Registered (GeneratorSpec id 'notes')\n"
        "[ ] Tests (no test_*.py)\n"
        "[?] Table row"
    )
    assert format_checklist([]) == "Checklist: nothing to check."
