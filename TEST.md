# Test Suite Guide

This document is a contributor-facing map of BESSER's Python test suite: how the
tests are laid out, how to run them, what each area covers, and what CI gates on.

> **Scope.** This covers BESSER's **own Python tests** under `tests/`. The web
> editor **frontend** (`besser/utilities/web_modeling_editor/frontend`) is a
> separate git submodule with its own test suite and is **not** covered here.

At the time of writing the suite collects **1962 tests** across **134 test
files**. (The collected number is higher than the raw count of `def test_*`
functions because `@pytest.mark.parametrize` expands many of them.)

---

## Layout: `tests/` mirrors the source tree

Tests live under `tests/` and mirror the `besser/` package structure, so the
tests for a module sit at the analogous path:

| Source (`besser/…`)                              | Tests (`tests/…`)                                |
| ------------------------------------------------ | ------------------------------------------------ |
| `BUML/metamodel/…`                               | `BUML/metamodel/…`                               |
| `BUML/notations/…`                               | `BUML/notations/…`                               |
| `generators/<name>/`                             | `generators/<name>/`                             |
| `utilities/web_modeling_editor/backend/…`        | `utilities/web_modeling_editor/backend/…`        |
| `utilities/buml_code_builder/…`                  | `utilities/buml_code_builder/…`                  |

Two additional top-level test areas have no direct source mirror:

- `tests/workflows/` — offline full-pipeline (model → real generator → assert
  produced code) tests for the demo scenarios.
- `tests/live/` — the same scenarios driven over a **deployed** HTTP/SSE
  backend; skipped unless explicitly enabled (see below).

`pyproject.toml` sets `--import-mode=importlib` so that `tests/utilities/…` and
`besser/utilities/…` (same relative path) don't collide in the import namespace.

### Non-test files under `tests/`

Some `.py` files under `tests/` are **runnable example/demo scripts**, not pytest
modules (they define models or produce output, and pytest collects nothing from
them). Examples: `tests/BUML/metamodel/structural/library/library.py`,
`tests/BUML/notations/image_to_buml/image2buml.py`,
`tests/BUML/notations/kg_to_buml/kg2buml.py`, and the neural-network model
scripts under `tests/BUML/metamodel/nn/` (`alexnet.py`, `vgg16.py`, …). The
`output/` and `output_backend/` subfolders hold generated reference output used
as fixtures. There are also two stray docs — `tests/SMART_GEN_TESTING.md` and
its `.html` twin — documentation, not tests.

---

## Running the tests

Run everything from the repo root:

```bash
python -m pytest
```

### Optional-dependency ignores (local machines)

Two areas import heavy optional packages that aren't in `requirements.txt`. When
they're absent, pytest fails at **collection** (a `ModuleNotFoundError`) *before*
running unrelated tests — this is a collection error, **not** a test failure.
CI installs the deps; locally, the pragmatic sweep skips them:

```bash
python -m pytest tests/ \
  --ignore=tests/generators/nn \
  --ignore=tests/utilities/web_modeling_editor/backend/test_spreadsheet_import.py
```

- `tests/generators/nn/` needs **torch** and **tensorflow**.
- `tests/utilities/web_modeling_editor/backend/test_spreadsheet_import.py` needs **openpyxl**.

The OCL parser tests and converter tests depend on `bocl` (and the backend on
`yaml`/`docker`/`github`/`openai`). Where those are backend-only extras, the
converter conftests (`.../services/converters/conftest.py`,
`.../converters/nn/conftest.py`) fall back to a `MagicMock` **only if** the real
package isn't installed, so the converter unit tests still collect in a minimal
environment.

### Running one area

```bash
# One generator by keyword
python -m pytest tests/generators -k sqlalchemy

# One directory
python -m pytest tests/utilities/web_modeling_editor/backend/services/converters

# One structural slice
python -m pytest tests/BUML/metamodel/structural -k library

# A standalone example (sanity-check the install)
python tests/BUML/metamodel/structural/library/library.py
```

---

## Continuous integration

`.github/workflows/ci.yml` runs on pull requests to `master` and `development`
with three jobs:

| Job     | What it runs                                                                                      | What it gates |
| ------- | ------------------------------------------------------------------------------------------------- | ------------- |
| `tests` | `python -m pytest tests/ -q --tb=short --ignore=tests/generators/nn -x` on Python **3.11 & 3.12** | The full suite must pass (fail-fast `-x`); the `nn/` torch/tf tests are skipped in CI. |
| `lint`  | `ruff check besser/ --select F841,F401,F541,F811,E711,E721,E731,E741 --ignore E501`                | A focused lint set (unused vars/imports, f-string/redefinition, comparison & lambda style); line length is **not** enforced. |
| `docs`  | `bash docs/check-docs-warnings.sh`                                                                 | Sphinx builds and fails on any **new / non-allowlisted** warning. Two pre-existing systemic categories are allowlisted: *"duplicate object description"* and *"more than one target found for cross-reference"*. |

> Note: CI's `tests` job does **not** pass the `openpyxl` ignore, because CI
> installs the backend requirements which include it.

---

## Shared fixtures (conftest)

Prefer reusing these over duplicating models in new tests.

- **`tests/conftest.py`** — base domain-model fixtures available to the whole
  suite:
  - `library_book_author_model` — the canonical Library/Book/Author model
    (1..*/0..* + N:M associations), used across SQL, JSON Schema, RDF,
    SQLAlchemy, workflow tests.
  - `employee_self_assoc_model` — Employee with a reflexive manager/subordinates
    association (shared by the django/pydantic/rest_api self-association tests).
  - `simple_library_book_model` — minimal Library/Book (no Author), the
    "normal association" counterpart in self-assoc tests.
  - `player_class` / `team_class` / `player_team_domain_model` — Player/Team
    model for OCL constraint tests.

- **`tests/generators/conftest.py`** — richer generator fixtures, plus an
  **import-cycle breaker**: it stubs the converters module while importing
  `BAFGenerator` (whose import chain cycles through the backend services) and
  then restores the real modules. Fixtures:
  - `library_model_with_enum` — Library/Book/Author + a `MemberType` enum.
  - `library_model_with_inheritance` — the above extended with a `BookType`
    superclass and Horror/History/Science subclasses (used by SQLAlchemy tests).

- **Area-local conftests**:
  `tests/BUML/metamodel/bpmn/conftest.py` (three clean BPMN models),
  `tests/BUML/notations/ocl/conftest.py` (a Department/Employee `WorksIn`
  model), and the two converter conftests that mock optional backend deps.

---

## Per-area inventory

Counts below are approximate `def test_*` counts (pre-parametrization) to give a
sense of weight, not exact collected totals.

### `tests/BUML/` — metamodel & notations (~360)

**Metamodel** (`tests/BUML/metamodel/`, ~285):

- `structural/` — the core object model. `test_structural.py` (~56) covers
  classes, properties, associations, generalizations, multiplicities, name
  validation, inheritance shadowing; `test_type_checking.py` (~13) covers type
  rules.
- `ocl/` — Object Constraint Language (~76 across 9 files): expression walking
  (`test_walk`), cloning (`test_clone`), substitution (`test_substitute`),
  chained navigation (`test_chain`), predicates, `size()`, error messages, and
  `test_comprehensive_constraints`. `test_ocl_parser.py` holds older
  `OCLWrapper`-evaluation smoke tests.
- `bpmn/` — BPMN metamodel (~45): processes, flows, gateways, collaborations,
  `validate()`.
- `nn/` — neural-network metamodel (~47): layers, tensor ops, model structure.
  (Pure metamodel — no torch/tf needed here, unlike `generators/nn/`.)
- `object/` — instance/object models (~21): `test_object_mm` and a fluent-API
  builder.
- `gui/` (~10), `deployment/` (~7), `feature_model/` (~8),
  `state_machine/` (~5).

**Notations** (`tests/BUML/notations/`, ~75) — concrete-syntax parsers:

- `structuralPlantUML/` (~13, incl. permissive-syntax) and
  `structuralDrawIO/` (~9) — class-diagram parsers.
- `ocl/` (~36) — parse/normalize/pretty-print/wrapping-visitor round-trips.
- `deployment_grammar/` (~7), `nn_grammar/` (~3),
  `action_language/` parsing (~2).
- `objectPlantUML/` (~5) — **all currently `@pytest.mark.skip`** pending a
  DataValue parser fix.

### `tests/generators/` — code generators (~600)

One subdirectory per generator; each validates both structure (class names,
fields, endpoints) and content (business logic) of generated artifacts.

- **`llm/`** — by far the largest area (~358 across 24 files). Covers the
  Spec-Driven / "Vibe" agent pipeline: the OpenAI provider (`test_openai_provider`,
  ~61), the tool executor (`test_tool_executor`, ~48), the orchestrator
  safeguards (~26), model serialization (~25), phase metadata / phase-3
  autofix & toolchain, prompt building, compaction, checkpoints/resume,
  modify-loop guards, frontend/endpoint contracts, the shell-tool gate, and
  tracing.
- **`backend/`** (~62) — FastAPI backend + Pydantic generation: `test_backend`,
  full-UML and modular-layout variants, OCL→Pydantic integration, `ocl_utils`,
  and Pydantic association/inheritance coverage.
- **`sqlalchemy/`** (~27) — schema generation, `is_id`/`is_external_id` →
  primary-key/unique, defect-fix regressions (introspected via
  `sqlalchemy.inspect`).
- **`django/`** (~30) — models generation, OCL constraints, self-associations,
  `is_id`/`is_external_id` → `primary_key`/`unique`/`__str__`.
- **`bpmn/`** (~17), **`sql/`** (~15), **`agents/`** (~25 — BAF generator +
  personalization + reasoning), **`web_app/`** (~14 incl. multi-agent).
- Smaller per-generator suites: `python`, `java`, `flutter`, `react`, `rdf`,
  `qiskit`, `supabase`, `terraform`, `json_schema`, `json_object`,
  `rest_api`, `testgen`, `action_language`.
- **`nn/`** (~4) — torch & tensorflow codegen; **needs torch/tensorflow**,
  ignored in CI and in the local sweep.

### `tests/utilities/web_modeling_editor/backend/` — web editor backend (~570)

- **`services/converters/`** (~225) — the bidirectional JSON↔BUML converters.
  `test_converter_roundtrip.py` (~76) and the parser tests (~74) are the core;
  plus BPMN converters (~30), OCL pre/post, multiplicity labels, role-name
  warnings, NN and class-diagram `buml_to_json` safety, agent round-trips, and
  the single-diagram project fallback. **By design** these keep symmetric
  `json_to_buml` **and** `buml_to_json` coverage — that symmetry is intentional,
  not redundancy.
- **`smart_generation/`** (~122) — the smart-generation router/runner: request
  validators, concurrency & config, phase emission, SSE events, done-verdict,
  free-tier, GitHub import/push, model assembly/sync, modify-seed, preview,
  logging.
- **`services/`** (~30) — `safe_buml_loader` (safe `exec` of BUML code) and SVG
  post-processing.
- **`services/utils/`** (~60) — agent-config manual-mapping & recommendation
  utils, GUI personalization, user-profile utils.
- **`test_api_integration.py`** (~69) — endpoint-level integration over the
  FastAPI app.
- **`test_spreadsheet_import.py`** (~4) — **needs openpyxl**; ignored in the
  local sweep.
- **`converters/nn/`** (~56) — the NN diagram processor and templates.

### `tests/utilities/buml_code_builder/` — code builders (~137)

`test_builders.py` (~122) plus `test_bpmn_model_builder.py` (~15): verify that
BUML metamodel instances round-trip through generated, `exec()`-able Python code
(domain / agent / GUI / project / quantum builders).

### `tests/workflows/` — offline pipeline (~3)

`test_generation_workflows.py`: runs the **real** SQL / SQLAlchemy / Backend
generators on `library_book_author_model` end-to-end and asserts the produced
code's structure. Deterministic, no network — the CI regression net for the two
demo scenarios ("only a database", "database + backend").

### `tests/live/` — deployed-stack E2E (~8, skipped by default)

The same demo scenarios driven over the deployed backend, **off by default**:

- `test_generation_workflows_live.py` — `POST /besser_api/generate-output`;
  enable with `RUN_LIVE_BACKEND_TESTS=1` (needs `requests` + a running backend;
  `BACKEND_URL` overrides the host).
- `test_vibe_free_e2e.py` — the keyless **free-tier** Vibe agent over SSE;
  enable with `RUN_LIVE_FREE_E2E=1` (slow, non-deterministic, real LLM).

These are **not** duplicates of `tests/workflows/` — they are the intentional
*live* companion layer to the *offline* workflow tests.

---

## Known caveats when running locally

- **Optional-dep collection errors** (not failures): `tests/generators/nn/`
  (torch/tensorflow) and `test_spreadsheet_import.py` (openpyxl). Use the
  `--ignore` flags shown above.
- **Known pre-existing red** (from upstream `master`, unrelated to local work):
  `tests/utilities/web_modeling_editor/backend/services/converters/test_project_single_diagram_fallback.py::test_project_to_json_handles_main_guard`.
- **Permanently skipped:** all 5 tests in
  `tests/BUML/notations/objectPlantUML/test_object_plantUML.py`
  (`@pytest.mark.skip`, blocked on a DataValue parser fix), plus a few `xfail`s
  in `test_django_self_assoc.py` and `test_converter_roundtrip.py`.
