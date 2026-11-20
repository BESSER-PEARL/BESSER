---
name: besser-dev
description: >
  Contributor guide for developing BESSER itself
  (https://github.com/BESSER-PEARL/BESSER). Use this skill whenever the user
  is working *inside* the BESSER source tree: adding a new generator (the
  most common contribution), adding a new metamodel or sub-DSL under
  `besser/BUML/metamodel/`, writing pytest tests for generators or
  metamodels, writing JSON<->BUML converters for the web editor, writing
  Sphinx documentation under `docs/source/`, registering a generator in
  the generator registry (`besser/generators/registry.py`), or preparing a
  pull request to BESSER. Trigger on phrases like "add a new generator",
  "register in the web editor", "GeneratorInterface", "json_to_buml",
  "buml_to_json", "write tests for my generator", "build the docs", "open a
  PR to BESSER", or any work that touches `besser/generators/`,
  `besser/BUML/metamodel/`, `besser/utilities/web_modeling_editor/`, or
  `tests/`. Prefer this skill over besser-user when the user is
  contributing *to* BESSER rather than *using* BESSER to build something else.
metadata:
  when_to_read: "Adding or changing a generator, a metamodel package, editor converters, tests or Sphinx docs in BESSER."
---

# Contributing to BESSER

Worked examples for the common contributions. The facts every change needs live in the repository's
`CLAUDE.md`, and this skill does not repeat them:

| Need | Where |
|---|---|
| Setup (including the backend requirements the tests need) | `CLAUDE.md` "Essential Commands" > "Setup" |
| Running tests (always scoped to `tests/`) and what CI runs | `CLAUDE.md` "Testing" |
| Lint (`ruff check besser/` and `lint-imports`) | `CLAUDE.md` "Linting" |
| Docs gate (`python docs/check_docs.py`, fails on any Sphinx warning) | `CLAUDE.md` "Documentation" |
| CI jobs (tests, lint, import layers, OpenAPI check, wheel, docs) | `CLAUDE.md` "CI/CD" |
| Pull-request target (`development`) and commit style | `CLAUDE.md` "Commits and PRs" |
| Editor changes and the submodule pointer | `CLAUDE.md` "Frontend Submodule" |
| Generator checklist, traceability, pitfalls | `CLAUDE.md` "Adding a deterministic generator", "Making a generator traceable", "Pitfalls" |

Long-form versions: `docs/source/contributor_guide.rst` and `docs/source/contributing/`.

---

## Adding a generator

In the BESSER Lab, the build step of a generator change offers the `scaffold_generator` tool
(`besser/spec_driven_agent/lab/scaffolds/generator.py`), which writes every part below in one call: the package,
its `GeneratorSpec`, a test, the docs page with its toctree entry and table row, the pinned lists and
snapshot of `test_generator_registry.py`, and its regeneration builder. The result is a working generator to
change into the real one. Outside the lab there is no command for it: follow the steps by hand, with the
scaffold's templates (`besser/spec_driven_agent/lab/scaffolds/templates/`) as the reference.

### 1. Package and class

```
besser/generators/my_generator/
  __init__.py               # from .my_generator import MyGenerator
  my_generator.py           # the GeneratorInterface subclass
  templates/
    my_output.txt.j2
```

The current pattern builds the Jinja environment with `traced_environment` and writes every render with
`self._write_traced`, so the editor's traceability works with no further code:

```python
# besser/generators/my_generator/my_generator.py
"""My generator: one ``my_output.txt`` file from a B-UML structural model."""

import logging
import os

from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.utilities.generation_trace import traced_environment

logger = logging.getLogger(__name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
OUTPUT_FILE = "my_output.txt"


class MyGenerator(GeneratorInterface):
    """Writes ``my_output.txt``: each class of the model with its attributes."""

    regenerable = True  # same model -> same bytes; see "Regeneration" below

    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        """Render the template into ``OUTPUT_FILE`` in the output directory."""
        env = traced_environment(TEMPLATES_DIR, trim_blocks=True, lstrip_blocks=True)
        rendered = env.get_template("my_output.txt.j2").render(model=self.model)
        self._write_traced(OUTPUT_FILE, rendered)
        logger.info("Code generated in the location: %s", self.build_generation_path(OUTPUT_FILE))
```

```jinja
{# besser/generators/my_generator/templates/my_output.txt.j2 #}
{% for cls in model.get_classes() | sort(attribute="name") %}
{% trace cls, "class" %}
{{ cls.name }}
{% for attribute in cls.attributes | sort(attribute="name") %}
  {{ attribute.name }}: {{ attribute.type.name }}
{% endfor %}
{% endtrace %}
{% endfor %}
```

- `super().__init__(model, output_dir)` sets `self.model` and `self.output_dir` (`None` means `<cwd>/output`);
  `self.build_generation_path(name)` and `self.build_generation_dir()` give the output paths.
- `_write_traced(relative_path, rendered, *sources)` strips the trace markers, writes UTF-8 with `\n` endings
  and records `self.traced_files[relative_path]`. A file generated as a whole from one element passes
  `trace_source(element, rule)` as a source; text that is not a render goes through `self._write_file`.
  Never assign `traced_files`. A composite generator calls `self._merge_traced_files(child, prefix)`.
  Full guide: `docs/source/generators/traceable_generator.rst`.
- Sort what you iterate over (sets have no order) so the output is deterministic.

**JSX and other templates that use `{{ }}` themselves:** give the environment other variable delimiters, as
`besser/generators/react/react.py` does:

```python
env = traced_environment(
    templates_path,
    trim_blocks=True,
    lstrip_blocks=True,
    variable_start_string="[[",
    variable_end_string="]]",
)
```

Templates then print `[[ cls.name ]]` and leave JSX's `{value}` and `{{ style }}` alone.

### 2. Registry entry

One `GeneratorSpec` in `GENERATORS` (`besser/generators/registry.py`); the editor's `SUPPORTED_GENERATORS` and
the Spec-Driven Agent's generator tools are derived from it, so never edit those lists:

```python
    GeneratorSpec(
        id="my_generator",
        class_path="besser.generators.my_generator:MyGenerator",
        category="object_oriented",
        output_type="file",            # "zip" for several files
        file_extension=".txt",
        download_filename="my_output.txt",
    ),
```

The other fields (`requires_class_diagram`, `required_diagram_type`, `keep_generated_filename`,
`needs_beyond_one_diagram`, `agent_tool`) are described in `CLAUDE.md` step 2.

### 3. The lists the test suite pins

- `tests/generators/test_generator_registry.py`: add the id to `WEB_EDITOR_IDS`, and to `WEB_EDITOR_ONLY`
  unless it is also an agent tool; add its entry to `tests/generators/snapshots/web_editor_generators.json`.
- **Regeneration** (`tests/generators/test_regenerable_generators.py` fails for a generator in neither list):
  either `regenerable = True` on the class and a builder in `BUILDERS` of
  `tests/generators/regenerable_builders.py`:

  ```python
      "my_generator": _on_library("besser.generators.my_generator", "MyGenerator"),
  ```

  or `regenerable = False` and an entry in `NOT_REGENERABLE` with the reason (for example
  `"supabase": "a timestamp in the migration's file name"`).
- To offer it to the Spec-Driven Agent: `CLAUDE.md` step 5 (`agent_tool`, `AGENT_TOOL_ORDER`, the handler,
  the `tools.rst` row, the pinned lists).

### 4. Tests

`tests/generators/my_generator/test_my_generator.py`, with a shared fixture from `tests/conftest.py` and
pytest's `tmp_path`:

```python
"""Tests of the My generator."""

from besser.generators.my_generator import MyGenerator


def test_writes_every_class_with_its_attributes(library_book_author_model, tmp_path):
    generator = MyGenerator(library_book_author_model, output_dir=str(tmp_path))
    generator.generate()

    text = (tmp_path / "my_output.txt").read_text(encoding="utf-8")
    assert "Book" in text.splitlines()
    assert "  title: str" in text.splitlines()
    assert "my_output.txt" in generator.traced_files
```

For a traced generator, also check that a model whose elements carry `id`s generates the same bytes as the
same model without them (`tests/generators/backend/test_backend_trace.py`). Run the new tests and the
registry and regeneration tests (`tests/generators/test_generator_registry.py`,
`tests/generators/test_regenerable_generators.py`).

### 5. Docs

- `docs/source/generators/my_generator.rst`: what it produces from which model, a usage snippet, the output.
- In `docs/source/generators.rst`, both a toctree entry (`generators/my_generator`) **and** a row in the
  "Choosing a Generator" table.

### 6. The editor's Generate menu

The menu is the frontend's, not the registry's: `packages/webapp/src/main/app/shell/menus/generator-menu-config.ts`
and the generator type in `packages/webapp/src/main/shared/types/workspace-types.ts`, in the editor repository
(the submodule checkout). Commit there and move the pointer (`CLAUDE.md` "Frontend Submodule").

---

## Adding a metamodel package or sub-DSL

### Package

```
besser/BUML/metamodel/my_dsl/
  __init__.py     # re-exports the public classes
  my_dsl.py
```

Extend `NamedElement` or `Element`; keep attributes private behind validating setters that fail fast; name
things like the sibling packages (`uml_component/` is a recent, complete example).

### `validate()`

Rules that span several elements go in the model's `validate()`, which collects every problem instead of
stopping at the first (`ComponentModel.validate` in `besser/BUML/metamodel/uml_component/uml_component.py`):

```python
    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the model against its structural rules.

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: ``{"success": bool, "errors": list[str], "warnings": list[str]}``.
        """
        errors: list = []
        warnings: list = []

        self._validate_endpoint_references(errors)
        self._validate_unique_names(errors)
        self._warn_structural_smells(warnings)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result
```

A subclass model calls `super().validate(raise_exception=False)` first and appends to its lists.

### Converters (when the editor draws it)

Under `besser/utilities/web_modeling_editor/backend/services/converters/`:

- JSON -> B-UML: `json_to_buml/my_dsl_diagram_processor.py` with `process_my_dsl_diagram(json_data)`, imported
  in `json_to_buml/__init__.py` (before `project_converter`) and re-exported from `converters/__init__.py`.
- B-UML -> JSON: `buml_to_json/my_dsl_diagram_converter.py`.
- A diagram that is part of a project: both `json_to_buml/project_converter.py` and
  `buml_to_json/project_converter.py`.
- Elements the editor draws get their ids from `converters/element_ids.py` (`EditorIds`).

Keep both directions symmetric and test the round trip on a diagram shaped like the editor's real export
(`tests/utilities/web_modeling_editor/backend/services/converters/test_component_converters.py`, which defines
the `real_component_diagram` fixture):

```python
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.component_diagram_converter import (
    component_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.component_diagram_processor import (
    process_component_diagram,
)


def test_ids_and_types_survive_the_round_trip(real_component_diagram):
    model = process_component_diagram(real_component_diagram)
    out = component_object_to_json(model)

    original = real_component_diagram["model"]
    assert sorted(out["elements"]) == sorted(original["elements"])
    assert sorted(out["relationships"]) == sorted(original["relationships"])
    for element_id, element in original["elements"].items():
        assert out["elements"][element_id]["type"] == element["type"]
```

Layout (`bounds`, `path`) is recomputed on the way back, so compare ids, types, names and ownership, not
positions.

### Backend routes

One thin router per concern in `backend/routers/` (parse the request, call a service, map the result or
exception to HTTP, `@handle_endpoint_errors`); the work goes in `backend/services/`, request and response models
in `backend/models/`; `backend.py` only includes the routers. Validation of the new model joins
`services/validators/ocl_checker.py` behind `/validate-diagram`. A route or model change also changes the
editor's committed OpenAPI schema (`CLAUDE.md` "CI/CD").

### Tests and docs

- Tests in `tests/BUML/metamodel/my_dsl/`: valid construction, each setter's refusal
  (`pytest.raises(ValueError)` / `TypeError`), `validate()` errors and warnings; converter tests next to the
  others in `tests/utilities/web_modeling_editor/backend/services/converters/`.
- Docs: `docs/source/buml_language/model_types/my_dsl.rst`, added to the toctree in
  `docs/source/buml_language/model_types.rst`.
- A new diagram type in the editor as well: `docs/source/contributing/diagram_dsl_workflow.rst`.

---

## Test patterns

- Tests mirror the source tree under `tests/`, named `test_*.py`; reuse the fixtures of `tests/conftest.py`
  (`library_book_author_model`, `employee_self_assoc_model`, `simple_library_book_model`,
  `player_team_domain_model`, ...) and `tests/generators/conftest.py` before building a model inline.
- Output goes to `tmp_path`; never write into the checkout.
- Assert content (names, mappings, the line a rule produced), not only that a file exists.
- Error paths: `pytest.raises(ValueError, match="...")` on the message the user will see.
- Generated Python that must run: import it from `tmp_path` with `importlib` and exercise it.
- A fix comes with a test that fails before it (`CLAUDE.md` "Engineering standard").

---

## Sphinx docs

The site uses the `sphinx_immaterial` theme (`docs/source/conf.py`); sources are reStructuredText in
`docs/source/`. The gate is the one in `CLAUDE.md` "Documentation": any warning fails it, so a broken reference
or a page missing from every toctree fails CI.

| Path | Holds |
|---|---|
| `docs/source/buml_language/model_types/` | One page per metamodel |
| `docs/source/generators/` | One page per generator, plus `build_generator.rst`, `traceable_generator.rst`, `regenerating.rst` |
| `docs/source/spec_driven_agent/` | The Spec-Driven Agent and the lab |
| `docs/source/web_editor.rst`, `web_editor_backend.rst` | Editor workflows; backend endpoints and environment variables |
| `docs/source/utilities/` | Utility pages |
| `docs/source/contributing/` | Contributor guides |
| `docs/source/releases/` | Release notes |
| `docs/source/_static/`, `docs/source/img/` | Images and assets |

Cross-references:

- ``:doc:`generators/django` `` links a page (relative to the current file unless it starts with `/`).
- ``:ref:`label-name` `` links a section labelled `.. _label-name:`.
- ``:mod:`besser.generators.django` ``, ``:class:`besser.BUML.metamodel.structural.Class` ``,
  ``:meth:`~besser.generators.generator_interface.GeneratorInterface.generate` `` link the API reference
  (`~` shows only the last name).
