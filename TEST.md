# Test Suite Guide

A contributor-facing map of BESSER's Python test suite: layout, how to run it,
what each area covers, and what CI gates on. The web editor **frontend**
(`besser/utilities/web_modeling_editor/frontend`) is a separate git submodule
with its own test suite and is not covered here.

The suite collects **4707 tests** (`python -m pytest tests --collect-only -q`);
`@pytest.mark.parametrize` makes that higher than the number of `def test_*`
functions.

---

## Layout

`tests/` mirrors the `besser/` package, so the tests for a module sit at the
analogous path:

| Source (`besser/…`)                         | Tests (`tests/…`)                           | Tests |
| ------------------------------------------- | ------------------------------------------- | ----: |
| `BUML/metamodel/…`, `BUML/notations/…`      | `BUML/metamodel/…`, `BUML/notations/…`      |   531 |
| `generators/<name>/`                        | `generators/<name>/`                        |   582 |
| `spec_driven_agent/`                        | `spec_driven_agent/`                        |  2117 |
| `utilities/web_modeling_editor/backend/…`   | `utilities/web_modeling_editor/backend/…`   |  1094 |
| other `utilities/…`                         | `utilities/…`                               |   372 |

Two areas have no source mirror:

- `tests/workflows/` — offline full-pipeline tests (model → real generator →
  assertions on the produced code).
- `tests/live/` — scenarios driven over a running backend; skipped unless
  explicitly enabled (see [Live tests](#live-tests)).

`pyproject.toml` sets `--import-mode=importlib` so that `tests/utilities/…` and
`besser/utilities/…` (same relative path) don't collide.

Some `.py` files under `tests/` are runnable example scripts rather than pytest
modules, e.g. `tests/BUML/metamodel/structural/library/library.py`,
`tests/BUML/notations/image_to_buml/image2buml.py` and the neural-network
model scripts under `tests/BUML/metamodel/nn/` (`alexnet.py`, `vgg16.py`, …).
Recorded inputs used as fixtures live next to the tests that read them (e.g.
`tests/spec_driven_agent/fixtures/`, `tests/generators/pydantic/fixtures/`).

---

## Running the tests

Install the package and the backend requirements, then run from the repo root:

```bash
pip install -e .
pip install -r besser/utilities/web_modeling_editor/backend/requirements.txt
python -m pytest tests
```

Running one area:

```bash
python -m pytest tests/generators -k sqlalchemy
python -m pytest tests/spec_driven_agent
python -m pytest tests/utilities/web_modeling_editor/backend/spec_driven
```

The Spec-Driven Agent's shell tests use a bubblewrap sandbox. Where it cannot
start (no `bwrap`, or no unprivileged user namespaces, e.g. on Windows or
macOS), `tests/spec_driven_agent/conftest.py` runs them unconfined with a
warning; `test_shell_sandbox.py` sets its own policy either way.

---

## Continuous integration

`.github/workflows/ci.yml` runs on pull requests to `master` and `development`:

| Job     | What it runs | Gate |
| ------- | ------------ | ---- |
| `tests` | installs `bubblewrap`, then `python -m pytest tests/ -q --tb=short --ignore=tests/generators/nn -x` on Python 3.11 and 3.12 | the suite must pass (fail-fast) |
| `lint`  | `ruff check besser/ --select F841,F401,F541,F811,E711,E721,E731,E741 --ignore E501` | focused lint set; line length not enforced |
| `docs`  | `bash docs/check-docs-warnings.sh` | Sphinx build fails on any new, non-allowlisted warning |

---

## Shared fixtures

- **`tests/conftest.py`** — base domain models for the whole suite:
  `library_book_author_model`, `employee_self_assoc_model`,
  `simple_library_book_model`, and the Player/Team models for OCL tests.
- **`tests/generators/conftest.py`** — `library_model_with_enum`,
  `library_model_with_inheritance`, and an import-cycle breaker for
  `BAFGenerator`.
- **`tests/spec_driven_agent/conftest.py`** — sandbox fallback (above) and an
  autouse fixture that keeps tests off the network model catalog.
- Area-local conftests under `tests/BUML/metamodel/bpmn/`,
  `tests/BUML/notations/ocl/`, and the converter conftests, which fall back to
  a `MagicMock` for optional backend dependencies only when the real package
  is missing.

---

## Per-area overview

- **`BUML/`** — metamodel (structural, OCL, BPMN, NN, object, GUI,
  deployment, feature model, state machine, action language) and the
  concrete-syntax parsers (PlantUML, DrawIO, OCL, deployment and NN grammars,
  knowledge graph → BUML).
- **`generators/`** — one directory per generator, asserting both the
  structure and the content of generated artifacts, plus cross-generator
  regressions at the top level (default values, primary-key types, mutually
  required foreign keys).
- **`spec_driven_agent/`** — the Spec-Driven Agent library: LLM providers
  (Anthropic, OpenAI, Mistral, Nebius, free tier and its fallback chain),
  tool executor and edit application, orchestrator safeguards, compaction,
  checkpoints and resume, gap analysis and requirements ledger, Phase 3
  validation and fix loop, runtime/contract probes, and tracing. Recorded
  generated apps under `fixtures/` feed the validators.
- **`utilities/web_modeling_editor/backend/`** — the FastAPI backend:
  JSON↔BUML converters (`services/converters/`, deliberately symmetric
  coverage), validators, the agent simulator router, `test_api_integration.py`
  for endpoint-level checks, and `spec_driven/` for the Spec-Driven Agent
  router and runner (request validation, concurrency and config, SSE events,
  durable runs, free and sponsored tiers, GitHub import/push, model assembly,
  secret redaction, telemetry).
- **`utilities/buml_code_builder/`** — BUML models round-trip through
  generated, `exec()`-able Python code.
- **`workflows/`** — the real SQL / SQLAlchemy / backend generators run
  end-to-end on `library_book_author_model`. Deterministic, no network.

---

## Live tests

`tests/live/` drives a running backend and is skipped unless both the switch
and `BACKEND_URL` are set:

| File | Enable with | Covers |
| ---- | ----------- | ------ |
| `test_generation_workflows_live.py` | `RUN_LIVE_BACKEND_TESTS=1` | `POST /besser_api/generate-output` for the workflow scenarios, plus graceful 4xx on empty or malformed models |
| `test_spec_driven_free_e2e.py` | `RUN_LIVE_FREE_E2E=1` | a keyless free-tier run over `POST /besser_api/spec-driven/generate` producing a backend app and Rust classes |

```bash
RUN_LIVE_BACKEND_TESTS=1 BACKEND_URL=http://localhost:9000/besser_api \
    python -m pytest tests/live/test_generation_workflows_live.py
RUN_LIVE_FREE_E2E=1 BACKEND_URL=https://<host>/besser_api \
    python -m pytest tests/live/test_spec_driven_free_e2e.py -s
```

In PowerShell set the variables with `$env:NAME = "value"` first. The
free-tier test needs `BESSER_FREE_LLM_BASE_URL`, `BESSER_FREE_LLM_TOKEN` and
`BESSER_FREE_LLM_MODEL` configured on the server; it uses a real model, is
slow and non-deterministic, and asserts that generation produced the expected
kind of output, not that the generated app boots. It can also be run directly
(`python tests/live/test_spec_driven_free_e2e.py`) for a streamed, per-phase
log.

---

## Known state

A full local run (Windows, Python 3.11) gives 4671 passed, 31 skipped,
3 xfailed and 2 failed.

- **Failing:**
  `tests/spec_driven_agent/test_action_probe.py::test_an_unhandled_exception_is_reported_unconditionally`
  and
  `tests/spec_driven_agent/test_constructibility_probe.py::test_a_create_handler_that_dials_out_is_blocked_inside_the_probe`.
  In both, the probe reports the generated app's failure wrapped in an
  `ExceptionGroup` instead of the underlying error the test expects.
- **Skipped by design:** the `tests/live/` tests (see above); the bubblewrap
  sandbox tests and the agent-simulator session tests that need Linux
  (`test_shell_sandbox.py`, `agent_simulator/test_session_manager.py`);
  `test_backend_nn_methods.py` without `torch`; and the NN template tests in
  `converters/nn/test_nn_templates.py` when the frontend submodule is not
  checked out.
- **Expected failures (`xfail`):** three self-association rendering tests in
  `tests/generators/django/test_django_self_assoc.py` (the Django template
  emits duplicate fields for self-associations).
