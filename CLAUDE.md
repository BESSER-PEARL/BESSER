# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BESSER is a low-code platform for model-driven engineering:
- **B-UML**: a Python metamodel for domain, object, state-machine, GUI, agent, BPMN, neural-network,
  quantum, deployment, feature and project models, plus OCL constraints
- **Code generators**: B-UML → Django, FastAPI, SQLAlchemy, React, Flutter, Qiskit, Terraform, etc.
- **Spec-Driven Agent**: a *hybrid* generator — deterministic scaffold, then an LLM customization loop,
  then validation with a bounded auto-fix loop. A first-class peer of the deterministic generators.
- **Web Modeling Editor backend**: FastAPI services behind https://editor.besser-pearl.org
- **Frontend**: git submodule at `besser/utilities/web_modeling_editor/frontend` (has its own `CLAUDE.md`)

Long-form contributor docs live in `docs/source/` (start at `contributor_guide.rst`; published at
https://besser.readthedocs.io/). Examples: https://github.com/BESSER-PEARL/BESSER-examples

## Essential Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e .                    # editable install — this is what makes `besser` importable
# Required to run or test the editor backend. FastAPI, openpyxl, openai etc. are NOT in the root requirements.txt.
pip install -r besser/utilities/web_modeling_editor/backend/requirements.txt
pip install -r docs/requirements.txt   # docs toolchain (optional)
```

### Testing
```bash
python -m pytest tests/                                             # ALWAYS scope to tests/
python -m pytest tests/ -q --tb=short --ignore=tests/generators/nn -x   # what CI runs (Python 3.11 + 3.12)
python -m pytest tests/generators -k sqlalchemy                     # targeted
python tests/BUML/metamodel/structural/library/library.py           # standalone example, verifies the install
```
- Without `openpyxl`, pytest fails at *collection* (`test_spreadsheet_import.py` imports it at module level) —
  an error, not a test failure. It ships in the backend requirements file.
- `torch` / `tensorflow` are not needed: `tests/generators/nn/` passes without them (CI still ignores it).
- CI installs `bubblewrap`; without it the shell-sandbox tests skip and shell tests run unconfined.
- `governancedsl` (backend requirements, pinned `==0.1.1`) parses the Governance DSL of governed BPMN
  gateways; without it the governance tests skip. It depends on `besser`, so install it after `pip install -e .`
  or pip pulls `besser` from PyPI.

### Linting
```bash
# Exactly what CI runs. A bare `ruff check .` uses a different rule set and will disagree.
ruff check besser/ --select F841,F401,F541,F811,E711,E721,E731,E741 \
  --ignore E501 --exclude "*/BESSERActionLanguageParser.py"
```

### Documentation
```bash
cd docs && make html                # output in docs/build/html/
python docs/check_docs.py           # what CI runs (check-docs-warnings.sh is a wrapper)
```
The docs gate fails on **any** Sphinx warning or error, except an unreachable external intersphinx inventory
(an outage at docs.python.org is not a docs defect). Pass `--offline` to skip fetching them.

### Running locally
```bash
docker compose up --build           # backend on :9000, frontend on :8080
python -m besser.utilities.web_modeling_editor.backend.backend   # backend alone, on :9000
```
The Spec-Driven Agent's PIA / Local (Ollama) providers need `BESSER_LLM_ALLOW_CUSTOM_BASE_URL=true`
(the local `docker-compose.yml` sets it; running the backend alone does not).

### Docker images
The root `Dockerfile` has two targets. `backend` is Python + requirements + `besser`, no compilers.
`smartgen-worker` adds Node 20/`tsc`, JDK 21, Rust, `kotlinc`, build-essential and bubblewrap, for the
Spec-Driven Agent's Phase 3 and `run_command`. Production runs `backend:latest` for the backend service
and `smartgen_worker:latest` for `besser-wme-smartgen`; the local `docker-compose.yml` builds the worker
target, since one container serves both. Behind a TLS-inspecting proxy, put its certs in `ca-certs-extra/`
(gitignored) and build with `--build-arg TRUST_EXTRA_CAS=1`; both targets strip them before shipping.
Host requirements: `docs/source/spec_driven_agent/production_deployment.rst`.

## Where Things Live

Core flow: frontend JSON ⇄ (`json_to_buml` / `buml_to_json`) ⇄ B-UML objects → generators → code,
optionally followed by the Spec-Driven Agent's LLM customization + validation.

| Path | What |
|------|------|
| `besser/BUML/metamodel/` | Abstract syntax: `structural/` (DomainModel, Class, Property, associations, generalization), `state_machine/`, `gui/`, `action_language/` (BAL), `bpmn/`, `deployment/`, `feature_model/`, `nn/`, `object/`, `ocl/`, `project/`, `quantum/` |
| `besser/BUML/notations/` | ANTLR-based parsers (PlantUML class/object, OCL, NN, deployment, BAL), draw.io import, LLM-assisted mockup → model |
| `besser/utilities/image_to_buml.py`, `kg_to_buml.py` | Image → class diagram; TTL/RDF/JSON knowledge graph → class diagram |
| `besser/utilities/buml_code_builder/` | B-UML instance → Python code that `exec()`s back into the model; `common.py` has `safe_var_name()` and `_escape_python_string()` (use it for any user-controlled string) |
| `besser/generators/` | Deterministic generators, all `GeneratorInterface`; Jinja2 templates in `generators/<name>/templates/` |
| `besser/spec_driven_agent/` | Spec-Driven Agent engine (see below) |
| `…/web_modeling_editor/backend/` | FastAPI app (`backend.py`: middleware, routers, lifespan tasks) |
| `…/backend/routers/` | One router per concern (generation, conversion, validation, deployment, spec_driven, telemetry, agent_simulator); `error_handler.py` has `@handle_endpoint_errors` |
| `…/backend/config/generators.py` | `SUPPORTED_GENERATORS` registry (`GeneratorInfo`) + `get_filename_for_generator` |
| `…/backend/constants/constants.py` | API version, temp prefixes, CORS origins, most `BESSER_LLM_*` caps and flags |
| `…/backend/models/` | Pydantic request/response models (`diagram.py`, `project.py`, `responses.py`, `spec_driven.py`) |
| `…/backend/services/converters/` | `json_to_buml/` and `buml_to_json/`, one processor per diagram type + the project converter |
| `…/backend/services/validators/ocl_checker.py` | Metamodel + OCL validation behind `/validate-diagram` |
| `…/backend/services/deployment/` | Docker Compose deploy, GitHub OAuth / deploy (these routers are registered from here, not `routers/`) |
| `…/backend/services/spec_driven/` | Spec-Driven service layer (see below) |
| `…/backend/services/exceptions.py` | `BesserError` → `ConversionError`, `ValidationError` (incl. `CodeValidationError` → 400), `GenerationError`, `ConfigurationError` |
| `…/web_modeling_editor/agent_simulator/` | Separate container that runs generated BAF agents in a bubblewrap sandbox; only `agent_simulator_router.py` talks to it |

(`…` = `besser/utilities/web_modeling_editor`.) Router layout, middleware, endpoint list and environment
variables are documented in `docs/source/web_editor_backend.rst`. Backend facts that are easy to get wrong:
- Every router mounts under `/besser_api`, so a path is `/besser_api` + the decorator path. Exceptions
  declared on the app itself: `GET /health` (no prefix) and `GET /besser_api/`. OpenAPI UI is at `/docs`,
  not `/besser_api/docs`.
- Unhandled exceptions are deliberately flattened to a generic HTTP 500; the traceback is in the server log.
- No per-client rate limiting; the only throughput control is `BESSER_LLM_MAX_CONCURRENT_RUNS` on
  spec-driven runs (429 when full, 409 when that run id is already active).

### Multi-diagram projects
Some generators need several diagrams (e.g. `WebAppGenerator` = `ClassDiagram` + `GUINoCodeDiagram` + optional
`AgentDiagram`) and go through `POST /besser_api/generate-output-from-project`. `ProjectInput.diagrams` is
`Dict[str, List[DiagramInput]]`; `currentDiagramIndices` picks the active diagram per type, and per-diagram
`references` resolve cross-diagram dependencies by ID (stable across deletion/reordering). Old single-diagram
payloads are auto-converted by a Pydantic validator. Diagram types: `ClassDiagram`, `ObjectDiagram`,
`StateMachineDiagram`, `AgentDiagram`, `GUINoCodeDiagram`, `QuantumCircuitDiagram`, `UserDiagram`, `NNDiagram`, `BPMN`.

## Spec-Driven Agent

Pipeline, tools, severities, caps and configuration are documented in `docs/source/spec_driven_agent/`
(`how_it_works.rst`, `tools.rst`, `validation.rst`, `configuration.rst`, `runs.rst`). Map:

- `pipeline/orchestrator.py` (`LLMOrchestrator`) — Phase 1 deterministic scaffold (+ Phase 0.5 stack metadata
  when no generator fits, Phase 1.5 scaffold validation), Phase 2 LLM loop and its guards.
  `pipeline/phase3_repair.py` — Phase 3 validation + bounded auto-fix with best-tree snapshot/restore.
  `pipeline/constants.py` — loop caps.
- `agent/tools.py` — the LLM's tool surface; `agent/tool_executor.py` — tool implementations and the
  `task_list` checklist; `agent/edit_apply.py` — the lenient match ladder behind `modify_file`;
  `agent/compaction.py` / `agent/history_eviction.py` — context management.
- `providers/llm_client.py` — provider clients, keyless `free` and `sponsored` tiers, pricing, planning-model
  routing, free-tier fallback chain.
- `planning/gap_analyzer.py` — produces the Phase 2 checklist. Its return value is load-bearing: `None` =
  analysis failed, `[]` = scaffold already sufficient (Phase 2 may be skipped), a list = the tasks.
- `validation/` — code checks; `validation/issues.py::_classify_issue` assigns severity.
- `state/checkpoint.py`, `state/tracing.py` — per-run checkpoint and trace files.
- Service layer `backend/services/spec_driven/`: `runner.py` (drives a run, emits SSE), `run_manager.py`
  (durable runs: SQLite event store, sequence numbers assigned *before* any subscriber sees a frame, replay
  via `?after=` / `Last-Event-ID`, `interrupted` marking on restart), `sse_events.py` (typed event schema),
  `secret_redaction.py` (every SSE frame + the workspace before packaging).

Invariants and rules:
- **A new agent tool needs three edits**: `agent/tools.py`, `_TOOL_MODEL_REQUIREMENTS` in the same file, and a
  handler in `ToolExecutor._handlers` (`agent/tool_executor.py`); without the requirements entry it is
  offered on projects that cannot satisfy it. `tests/spec_driven_agent/test_added_generator_tools.py`
  asserts every generator tool has an entry.
- **New validator findings need a stable message prefix** classified in `_classify_issue` — the classifier
  keys on prefixes, not on which validator produced them. Severities: `blocker` (drives the auto-fix loop),
  `warning` (recorded), `style` (cosmetic ruff rules, recorded). Full table: `validation.rst`.
- **Runtime gate**: every Phase 3 exit, including budget exhaustion, runs a gate — a run cannot report
  complete while the delivered app cannot boot or create a record.
- **Client-visible contract**: request = `backend/models/spec_driven.py` (API key is a `SecretStr`, caps clamped
  by validators); stream = `services/spec_driven/sse_events.py`. Both are consumed by the frontend's
  spec-driven trigger — adding a field to a client-visible event means editing `sse_events.py`.
- Configuration is via `BESSER_LLM_*` / `BESSER_FREE_LLM_*` env vars: backend caps and flags in
  `backend/constants/constants.py`, engine knobs (compaction, history eviction, planning model, …) read
  directly in `besser/spec_driven_agent/`. All documented in `configuration.rst`.

### Security defaults — do not flip
- `BESSER_LLM_ENABLE_SHELL_TOOLS` — code default **off** (arbitrary shell on a shared BYOK host is RCE).
  `docker-compose.prod.yml` enables it only on the isolated `besser-wme-smartgen` worker, which has no
  `env_file`, receives only LLM credentials, and confines each run's shell with bubblewrap. Enable per
  service, never through the shared `.env`; changing the code default is a maintainer decision.
- `BESSER_LLM_ALLOW_CUSTOM_BASE_URL` — **off** by default (SSRF: the server would open a user-supplied URL).
  Requests carrying `base_url` (PIA, Local/Ollama) are rejected unless set. Keep it off on shared hosts and
  in `docker-compose.prod.yml`; only the local `docker-compose.yml` turns it on.

### Debugging a run
Each run workspace holds `.besser_trace.jsonl` (phases, turns, tool calls, costs, compaction, rollbacks,
findings), `.besser_checkpoint.json` (present only if Phase 2 did not exit cleanly — that is what makes a run
resumable) and `.besser_recipe.json` (final summary, validation issues, authorship split). The durable event
store is SQLite at `BESSER_LLM_RUN_STORE_PATH` or the system temp directory.

## Conventions

### Code style
- PEP 8, 4-space indentation, 120-char line target (pylint `max-line-length` in `pyproject.toml`; CI's ruff
  ignores `E501`, so long lines won't fail the build — keep them short anyway)
- Type hints on public APIs, docstrings; `snake_case` / `PascalCase` / `UPPER_CASE`; imports stdlib → third-party → local
- Metamodel classes use private attributes with validating setters (`NamedElement.name` rejects None/blank
  and warns on Python keywords); fail fast in setters
- `UNLIMITED_MAX_MULTIPLICITY = 9999` (`besser/BUML/metamodel/structural/structural.py`)

### Validation
Three modeling-side layers: construction (setters), metamodel (`.validate()`), OCL constraints. Collect
errors rather than raising, for unified reporting — `/validate-diagram` returns every metamodel and OCL error
in one response. (Spec-Driven code validation is a separate, fourth layer; see above.)

### Adding a deterministic generator
1. Package in `besser/generators/<name>/` implementing `GeneratorInterface` (`__init__(model, output_dir=None)`,
   `generate()`; `output_dir=None` means `<cwd>/output`); templates in `generators/<name>/templates/`
2. Register in `backend/config/generators.py`: `SUPPORTED_GENERATORS` (`GeneratorInfo`: `output_type`
   `"file"`/`"zip"`, `requires_class_diagram`, `required_diagram_type`) **and** `get_filename_for_generator`
3. Tests in `tests/generators/<name>/`
4. `docs/source/generators/<name>.rst`, added to a toctree **and** the "Choosing a Generator" table in
   `docs/source/generators.rst`
5. If the LLM agent should call it: a tool in `besser/spec_driven_agent/agent/tools.py` + `_TOOL_MODEL_REQUIREMENTS`

Walkthrough: `docs/source/generators/build_generator.rst` and `docs/source/contributing/create_generator.rst`.
`PytorchGenerator` / `TFGenerator` are registered only when `torch` / `tensorflow` import. The Spec-Driven
Agent is not in `SUPPORTED_GENERATORS`; it has its own router rather than `/generate-output`.

### Pitfalls
1. **Keep converters symmetric**: if `json_to_buml` supports a feature, `buml_to_json` must too
   (e.g. `class_diagram_processor.py` ↔ `class_diagram_converter.py`); test round-trips (JSON→BUML→JSON is identity)
2. **Determinism**: identical input → identical output (no timestamps in file names)
3. **Shared helpers** belong in `besser/utilities`, not in individual generators
4. **Temp directories** use the `besser_*` prefixes from `constants.py` (`besser_`, `besser_agent_`,
   `besser_csv_`, `besser_llm_`) so the hourly cleanup task (`services/cleanup.py`, >24 h) finds them;
   clean up with try/finally; stream large outputs as ZIPs
5. **Backend contract changes** (endpoints, request/response shapes) must be coordinated with the frontend's
   `shared/api/` layer
6. **Docs sync**: public-surface changes usually need `docs/source/` updates (see below)

## Testing Conventions
- Tests in `tests/` mirroring the source tree, named `test_*.py`; add tests for behavioral changes,
  especially metamodel and generator logic; assert structure (names, endpoints) and content
- Reuse fixtures from `tests/conftest.py` (`library_book_author_model`, `employee_self_assoc_model`,
  `simple_library_book_model`, `player_team_domain_model`, …) and `tests/generators/conftest.py`
- `pyproject.toml` sets `--import-mode=importlib` to avoid test/source namespace collisions

## Frontend Submodule

`besser/utilities/web_modeling_editor/frontend` → `BESSER-PEARL/BESSER-Web-Modeling-Editor`, branch `main`.
```bash
git submodule update --init --recursive                                       # first time
git submodule update --remote besser/utilities/web_modeling_editor/frontend   # fast-forward to tracked branch
git add besser/utilities/web_modeling_editor/frontend                         # record the new pointer
```
Cross-repo changes: implement each side in its own repo, update the submodule pointer here, link both PRs
and note the merge order.

## CI/CD
- `.github/workflows/ci.yml` — on PRs to `master`/`development`: tests (3.11, 3.12), the ruff command above,
  and the docs gate. It does **not** build the frontend.
- `security.yml` — CodeQL. `python-publish.yml` — PyPI release.
- `deploy-wme.yml` — manual (`workflow_dispatch`) build + push of backend, `smartgen_worker`, frontend and
  agent-simulator images, then EC2 deploy. A backend deploy recreates `besser-wme-backend` and
  `besser-wme-smartgen` and verifies the build stamp in both; it aborts before pushing if the host's compose
  file is missing a service or the worker is not on the `smartgen_worker` image.

## Documentation Sync
Keep `docs/source/` in step with code:
- `buml_language.rst` — metamodel additions
- `generators.rst` + `generators/<name>.rst` — generators (toctree **and** choosing table)
- `spec_driven_agent/` — pipeline, tools, severities, caps, config
- `web_editor.rst` — editor workflows; `spec_driven_agent/api.rst` — the spec-driven API contract
- `web_editor_backend.rst` — endpoint and environment-variable tables
- `utilities.rst`, `utilities/` — utilities (incl. `buml_code_builder.rst`, `agent_simulator.rst`)
- `contributor_guide.rst`, `ai_assistant_guide.rst` — workflow changes

## Commits and PRs
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`); short imperative subjects;
  topic branches (`feature/add-generator`)
- **Open pull requests against `development`, not `master`.**
- `.github/copilot-instructions.md` and `.cursorrules` only point here — edit `CLAUDE.md`, not them.
  See also `CONTRIBUTING.md`, `DEVELOPMENT_SETUP.md`, `GOVERNANCE.md`.
