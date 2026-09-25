# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BESSER is a low-code platform for building software through model-driven engineering. It consists of:
- **B-UML**: A Python-based metamodel for describing domain models, object (instance) models, state machines, GUI designs, agents, BPMN processes, neural networks, quantum circuits, deployments, feature models, and OCL constraints
- **Code Generators**: Transform B-UML models into executable code (Django, FastAPI, SQLAlchemy, Flutter, React, etc.)
- **Spec-Driven Agent**: A *hybrid* generator — deterministic scaffold, then an LLM customization loop, then validation with a bounded auto-fix loop. Lives in `besser/spec_driven_agent/` (the engine) and `besser/utilities/web_modeling_editor/backend/services/spec_driven/` (the service layer). It is a first-class part of the system, not an add-on.
- **Web Modeling Editor Backend**: FastAPI services powering the online visual editor at https://editor.besser-pearl.org
- **Frontend Submodule**: TypeScript/React UI at `besser/utilities/web_modeling_editor/frontend` (maintained separately)

## Essential Commands

### Setup and Installation
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Editable install — this is what makes `besser` importable
pip install -e .

# Required to run (or test) the web modeling editor backend.
# FastAPI, openpyxl, pyflakes, openai etc. are NOT in the root requirements.txt.
pip install -r besser/utilities/web_modeling_editor/backend/requirements.txt

# Docs toolchain (optional)
pip install -r docs/requirements.txt
```

### Testing
```bash
# Full suite — ALWAYS scope to tests/
python -m pytest tests/

# What CI runs
python -m pytest tests/ -q --tb=short --ignore=tests/generators/nn -x

# Targeted
python -m pytest tests/generators -k sqlalchemy
python -m pytest tests/BUML/metamodel/structural -k library

# Standalone example, to verify the install
python tests/BUML/metamodel/structural/library/library.py
```

### Linting
```bash
# Exactly what CI runs. A bare `ruff check .` uses a different rule set and will disagree.
ruff check besser/ --select F841,F401,F541,F811,E711,E721,E731,E741 \
  --ignore E501 --exclude "*/BESSERActionLanguageParser.py"
```

### Documentation
```bash
cd docs && make html          # build; output in docs/build/html/
bash docs/check-docs-warnings.sh   # what CI gates on
```
The docs gate is an **allowlist**, not a warning count: the build may emit warnings,
but only two systemic categories are tolerated (`duplicate object description` and
`more than one target found for cross-reference`). Anything else fails CI.

### Local Stack Deployment
```bash
docker compose up --build     # backend on :9000, frontend on :8080
```

### Running the backend alone
```bash
python -m besser.utilities.web_modeling_editor.backend.backend   # serves on :9000
```

## Architecture Overview

### Core Architecture Pattern: Metamodel → Conversion → Generation

```
Frontend JSON (visual editor)
    ↕ (json_to_buml / buml_to_json converters)
BUML Metamodel (Python objects)
    ↕ (notations parsers / code builders)
Generated Code (Django, React, SQL, etc.)
                    │
                    └─ optionally: LLM customization + validation (Spec-Driven Agent)
```

### Major Components

#### 1. B-UML Metamodel (`besser/BUML/metamodel/`)

The metamodel defines abstract syntax for all domain concepts:

- **`structural/`**: Core object-oriented modeling
  - `DomainModel` - container for all types
  - `Class`, `Property`, `Method` - OOP constructs
  - `Association` types: `BinaryAssociation`, `AssociationClass`
  - `Generalization` - inheritance relationships
  - `PrimitiveDataType`, `Enumeration` - data types
  - Base classes: `Element` → `NamedElement` → domain concepts

- **`state_machine/`**: Behavioral state modeling
  - `StateMachine`, `State`, `Transition`
  - `Condition` accepts optional `source` parameter (for serialization round-trips)
  - `StateMachine.validate()` returns `{success, errors, warnings}` dict

- **`gui/`**: UI modeling
  - `GUIModel` - top-level container
  - `Screen`, `Module` - logical organization
  - `ViewComponent`, `ViewContainer` - UI building blocks
  - `Style`, `Binding`, `EventsActions` - styling/interactivity

- **`action_language/`**: Business Action Language (BAL) for behavior specification

- **Other metamodels**: `bpmn/`, `deployment/`, `feature_model/`, `nn/` (neural networks),
  `object/` (instances), `ocl/`, `project/`, `quantum/`

**Key Pattern**: Private properties with getter/setter validation. Base classes define interfaces, subclasses add domain-specific behavior. `NamedElement.name` setter validates against None, empty/whitespace, and warns on Python keywords. Structural model validates attribute shadowing in inheritance hierarchies.

#### 2. Notations (`besser/BUML/notations/`)

Parse concrete syntaxes into metamodel instances:

- **`structuralPlantUML/`**: ANTLR-based PlantUML parser for class diagrams
- **`objectPlantUML/`**: Object diagram notation
- **`ocl/`**: Object Constraint Language support (`BOCLParser`, `BOCLLexer`)
- **`structuralDrawIO/`**: draw.io class-diagram import
- **`mockup_to_buml/`**, **`mockup_to_structural/`**: LLM-assisted mockup → GUI / structural model
- **`nn/`**, **`deployment/`**, **`action_language/`**: Specialized notation parsers

Two more model-entry paths live directly in `besser/utilities/`: `image_to_buml.py`
(photo/screenshot → class diagram) and `kg_to_buml.py` (TTL/RDF/JSON knowledge graph → class diagram).

**Pattern**: Grammar-based parsing (ANTLR) + listener pattern for AST traversal.

#### 3. Generators (`besser/generators/`)

Transform B-UML models into executable artifacts. All implement `GeneratorInterface`:

```python
class GeneratorInterface(ABC):
    def __init__(self, model: Model, output_dir: str = None): ...
    def generate(self): ...
    # helpers: build_generation_dir() / build_generation_path(file_name)
    # output_dir=None means "<cwd>/output"
```

**Generator Categories** (see `utilities/web_modeling_editor/backend/config/generators.py`):

- **Object-Oriented**: `PythonGenerator`, `JavaGenerator`, `PydanticGenerator`, `TestCaseGenerator`
- **Web Frameworks**: `DjangoGenerator`, `BackendGenerator` (FastAPI), `WebAppGenerator` (full-stack), `RESTAPIGenerator`
- **Databases**: `SQLGenerator`, `SQLAlchemyGenerator`, `SupabaseGenerator`
- **Data Formats**: `JSONSchemaGenerator`, `JSONObjectGenerator`, `RDFGenerator`
- **Frontend**: `ReactGenerator`, `FlutterGenerator`
- **AI/Agents**: `BAFGenerator` (BESSER Agent Framework)
- **Neural Networks**: `PytorchGenerator`, `TFGenerator` (registered only when `torch` / `tensorflow` import)
- **Quantum**: `QiskitGenerator`
- **Deployment**: `TerraformGenerator`
- **Business Process**: `BPMNGenerator`
- **Hybrid / LLM**: `besser/spec_driven_agent/` — see below. Not in `SUPPORTED_GENERATORS`;
  it is driven by its own router rather than `/generate-output`.

**Key Pattern**: Template-based generation with Jinja2. Templates live in `generators/[type]/templates/`.

#### 4. Spec-Driven Agent (`besser/spec_driven_agent/`)

The hybrid generator. Treat it as a peer of the deterministic generators, not a side project.

- **`pipeline/orchestrator.py`** (`LLMOrchestrator`) — owns the three phases and all the loop guards:
  - Phase 1 deterministic generation (plus Phase 0.5 stack-metadata when no generator fits, and Phase 1.5 validation of the scaffold)
  - Phase 2 LLM customization loop
  - Phase 3 validation + bounded auto-fix (`_MAX_TOOLCHAIN_FIX_ITERATIONS = 5` is a FLOOR - the loop runs `max(5, max_turns - total_turns)` rounds, best-tree snapshot/restore). A **runtime gate** runs on every Phase 3 exit including budget exhaustion: a run cannot report complete while the delivered app cannot boot or create a record
  - Severity classification lives in `_classify_issue` (`validation/issues.py`, re-exported here): `blocker` / `warning` / `style`
  - Guards worth knowing: turn cap (`MAX_TURNS = 120`), cost/runtime caps checked at turn boundaries, truncation recovery (`_MAX_TRUNCATION_RETRIES = 4`, per run, never reset), per-file modify-loop detection (`_PER_FILE_MODIFY_THRESHOLD = 3`), parallel tool execution grouped by write path (`_MAX_PARALLEL_WORKERS = 4`), checklist end_turn gate (`_MAX_TASK_NUDGES = 2`, 4 when an open item carries a verifier)
- **`agent/tools.py`** — declares the LLM's tool surface (files, model queries, validation/bookkeeping, generators, shell). **If you add a tool, add it to `_TOOL_MODEL_REQUIREMENTS` in the same file** so it is only offered when the models it needs are present; `tests/spec_driven_agent/test_added_generator_tools.py` asserts every generator tool has an entry.
- **`agent/tool_executor.py`** — implements the tools, plus the `task_list` checklist (batch `ids=[...]`, bounded verification retries: `_MAX_TASK_VERIFY_ATTEMPTS = 3`, after which an item is recorded *blocked* and stops holding the gate open).
- **`providers/llm_client.py`** — provider clients (`anthropic`, `openai`, `mistral`, `nebius`), the keyless `free` tier and the `sponsored` tier, pricing tables, the cheap planning-model routing, and the free-tier fallback chain.
- **`planning/gap_analyzer.py`** — the cheap planning call that produces the Phase 2 checklist. Its return value is load-bearing: `None` = analysis failed, `[]` = scaffold already sufficient (Phase 2 *may* be skipped), a list = the task list.
- **Validators** (`validation/`): `contract_checks.py` (model-derived data contract), `endpoint_coherence.py` (frontend fetch/axios URLs vs generated routes — report-only), `acceptance.py` (per-entity route/page/create matrix — report-only), `write_diagnostics.py` (same-turn parse + undefined-name check on every written file).
- **Context / recovery**: `agent/compaction.py` (lossy summarization above `BESSER_LLM_COMPACT_THRESHOLD`), `agent/history_eviction.py` (lossless file-body stubbing, opt-in), `state/checkpoint.py` (`.besser_checkpoint.json`, written per turn, deleted only on a clean Phase 2 exit), `state/tracing.py` (`.besser_trace.jsonl`).
- **`agent/edit_apply.py`** — the lenient `old_text` → `new_text` match ladder behind `modify_file`.

Service layer (`backend/services/spec_driven/`):
- **`runner.py`** — drives one run and emits SSE.
- **`run_manager.py`** — durable runs: a SQLite event store, monotonic sequence numbers assigned *before* any subscriber sees a frame, replay via `?after=` / `Last-Event-ID`, an abandonment grace period, and `interrupted` marking on restart.
- **`sse_events.py`** — the typed event schema. Adding a field to a client-visible event means editing this file.
- **`secret_redaction.py`** — applied on every SSE frame and over the workspace before packaging.
- **`preview.py`**, **`model_assembly.py`**, **`telemetry.py`**, **`incidents.py`**.

Behaviour is configured entirely through `BESSER_LLM_*` / `BESSER_FREE_LLM_*` environment
variables, defined in one place: `backend/constants/constants.py`. Two security-relevant
flags:

- `BESSER_LLM_ENABLE_SHELL_TOOLS` — code default **off**, because arbitrary shell on a
  shared BYOK host is RCE. `docker-compose.prod.yml` enables it only on the isolated
  `besser-wme-smartgen` worker, which has no `env_file` and receives only the LLM
  credentials, and each run's shell is confined by bubblewrap. Enable it per service,
  never through the shared `.env`; changing the code default is a maintainer decision.
- `BESSER_LLM_ALLOW_CUSTOM_BASE_URL` — **off** by default (SSRF: the server would open a
  user-supplied URL). A request carrying `base_url` (the editor's PIA and Local /
  self-hosted providers, e.g. Ollama) is rejected unless it is set. Keep it off on shared
  hosts; single-tenant or local installs that use those providers turn it on.

See `docs/source/spec_driven_agent/` for the user-facing documentation.

#### 5. Web Modeling Editor Backend (`besser/utilities/web_modeling_editor/backend/`)

FastAPI service with a **modular router architecture**. The application module is
`backend.py`, which defines the middleware stack, includes the routers, registers
exception handlers, and runs a lifespan that starts the temp-file cleanup task, the
download-registry sweeper, and the durable-run sweeper.

**Routers** (`backend/routers/`) — every one mounts at `prefix="/besser_api"`, so a
documented path is always `/besser_api` + the decorator path:

- **`generation_router.py`** - `/generate-output`, `/generate-output-from-project`, and the agent-personalization + GUI-personalization endpoints
- **`conversion_router.py`** - BUML import/export, CSV/XLSX reverse engineering, image-to-model, knowledge-graph-to-model, SVG rendering
- **`validation_router.py`** - `/validate-diagram` (metamodel + OCL), deprecated `/check-ocl`
- **`deployment_router.py`** - `/deploy-app` (Docker Compose) and `/feedback`
- **`spec_driven_router.py`** - the Spec-Driven Agent: generate / preview / resume / cancel / download / runs / runs-events / config / push-to-github / import-github-run
- **`telemetry_router.py`** - opt-in run telemetry collection and reporting
- **`agent_simulator_router.py`** - Live agent simulation (`/simulation`): generates the BAF agent and relays it to the agent simulator service
- **`error_handler.py`** - Centralized `@handle_endpoint_errors` decorator; anything that is not a known BESSER exception becomes a generic HTTP 500

GitHub OAuth and GitHub deployment routers are registered from `services/deployment/`
with `prefix="/besser_api"` rather than from `routers/`.

**Middleware** — applied in `backend.py`, outermost first:
- Request logging (`middleware/request_logging.py`) — UUID request IDs, timing, slow-request warnings (>1s)
- CORS — origins from `CORS_ORIGINS` or `DEFAULT_CORS_ORIGINS`; exposes `Content-Disposition` and `X-BESSER-Run-Id`
- `RequestSizeLimitMiddleware` — 50 MB global body cap → 413
- `SecurityHeadersMiddleware` — CSP, HSTS, `X-Frame-Options: DENY`, nosniff, Referrer-Policy, Permissions-Policy

There is no per-client rate limiting; the only throughput control is
`BESSER_LLM_MAX_CONCURRENT_RUNS` on spec-driven runs (429 when full, 409 when a run id is already active).

**Constants** (`backend/constants/constants.py`):
- API version, temp directory prefixes, generator defaults, CORS origins, relationship type mappings, and every `BESSER_LLM_*` cap and feature flag

**Core Services** (`backend/services/`):

- **Conversion Services** (`services/converters/json_to_buml/`, `services/converters/buml_to_json/`):
  - Bidirectional transformations between frontend JSON and B-UML metamodel
  - One processor per diagram type (class, state machine, agent, object, GUI, quantum, BPMN, NN) plus the project converter
  - Detailed parsers for attributes, methods, multiplicity, OCL constraints

- **Validation Services** (`services/validators/ocl_checker.py`):
  - 3-level validation: construction (setters), metamodel (`.validate()`), OCL constraints

- **Spec-Driven Services** (`services/spec_driven/`): see component 4

- **Deployment Services** (`services/deployment/`):
  - Docker Compose orchestration (`docker_deployment.py`)
  - GitHub integration (`github_service.py`, `github_oauth.py`, `github_deploy_api.py`)
  - Session store (`session_store.py`) for OAuth state management

- **Reverse Engineering** (`services/reverse_engineering/csv_reverse.py`): CSV/XLSX → domain model.
  (Image → class diagram lives in `besser/utilities/image_to_buml.py`, not here.)

- **Utils** (`services/utils/`): agent-config recommendation (LLM and rule-based), agent generation, GUI personalization, layout calculation, resource management, user profiles

- **Cleanup Service** (`services/cleanup.py`):
  - Background task removing temp directories older than 24 hours, run hourly
  - Prefixes: `besser_`, `besser_agent_`, `besser_csv_`, `besser_llm_`, `user_profile_`

- **Exception Hierarchy** (`services/exceptions.py`):
  - `BesserError` (base) → `ConversionError`, `ValidationError`, `GenerationError`, `ConfigurationError`; `CodeValidationError` (invalid custom agent code) is a `ValidationError` → HTTP 400

- **Feedback Service** (`services/feedback_service.py`): user feedback submissions over SMTP

- **Agent Simulator** (`web_modeling_editor/agent_simulator/`, a separate service in its own container, not under `services/`):
  - Runs generated BAF agents in a per-session bubblewrap sandbox; only `agent_simulator_router.py` talks to it
  - Documented in `docs/source/utilities/agent_simulator.rst`

**Key API Endpoints** (all prefixed `/besser_api`):
- `POST /besser_api/generate-output` - Single diagram → code
- `POST /besser_api/generate-output-from-project` - Multi-diagram project → code (e.g., WebApp needs ClassDiagram + GUINoCodeDiagram)
- `POST /besser_api/spec-driven/generate` - Hybrid LLM run, streamed as SSE
- `GET  /besser_api/spec-driven/runs/{run_id}/events?after=N` - Replay + follow a durable run
- `POST /besser_api/export-buml` - Diagram JSON → BUML Python code
- `POST /besser_api/get-json-model` - BUML file → JSON (auto-detects diagram type)
- `POST /besser_api/get-json-model-from-image` - Image → ClassDiagram JSON (via OpenAI)
- `POST /besser_api/validate-diagram` - Unified validation
- `POST /besser_api/deploy-app` - Docker Compose deployment
- `POST /besser_api/simulation/sessions` - Start a live agent simulation (see `agent_simulator_router.py`)

Two paths are declared on the app rather than a router: `GET /health` (no `/besser_api`
prefix) and `GET /besser_api/`. The OpenAPI UI is at `/docs` — *not* `/besser_api/docs`.

**Configuration Layer** (`backend/config/`):
- `generators.py` - Centralized generator registry with metadata (`GeneratorInfo` NamedTuple with `requires_class_diagram` and `required_diagram_type`)

**Models Layer** (`backend/models/`):
- `diagram.py`: `DiagramInput`, `FeedbackSubmission`
- `project.py`: `ProjectInput`
- `responses.py`: `DiagramExportResponse`, `ProjectExportResponse`, `ValidationResponse`, `ApiInfoResponse`, `FeedbackResponse`
- `spec_driven.py`: `SmartGenerateRequest` (API key is a `SecretStr`; caps clamped by field validators), `SmartPreviewRequest`, the GitHub push/import models

#### 6. BUML Code Builders (`besser/utilities/buml_code_builder/`)

Generate executable Python code from B-UML metamodel instances:
- `domain_model_builder.py` - DomainModel → Python code
- `agent_model_builder.py` - AgentModel → Python code
- `gui_model_builder.py` - GUIModel → Python code
- `project_builder.py` - Project → Python code
- `quantum_model_builder.py` - QuantumCircuit → Python code
- `state_machine_builder.py` - StateMachine → Python code
- `bpmn_model_builder.py` - BPMN model → Python code
- `nn_model_builder.py` - NN model → Python code
- `common.py` - Shared utilities: `safe_var_name()` (converts names to safe Python identifiers), `_escape_python_string()` (prevents code injection from user-controlled inputs)

**Pattern**: Generated code can be `exec()`'d to recreate the metamodel instance.

### Multi-Diagram Projects

Some generators require multiple diagram types:
- **WebAppGenerator**: Needs `ClassDiagram` (backend) + `GUINoCodeDiagram` (frontend) + optional `AgentDiagram`
- Projects use `ProjectInput` with `diagrams: Dict[str, List[DiagramInput]]` (multiple diagrams per type)
- `currentDiagramIndices: Dict[str, int]` tracks the active diagram per type
- Per-diagram `references: Dict[str, str]` resolve cross-diagram dependencies by ID (stable across deletions/reordering)
- Backward compatible: old single-diagram format auto-converts to arrays via Pydantic model validator
- Diagram types: `ClassDiagram`, `ObjectDiagram`, `StateMachineDiagram`, `AgentDiagram`,
  `GUINoCodeDiagram`, `QuantumCircuitDiagram`, `UserDiagram`, `NNDiagram`, `BPMN`

**Flow Example**:
```
1. Frontend sends ProjectInput with ClassDiagram + GUINoCodeDiagram
2. /besser_api/generate-output-from-project endpoint
3. Active ClassDiagram resolved via currentDiagramIndices or per-diagram references
4. ClassDiagram JSON → process_class_diagram → DomainModel
5. GUINoCodeDiagram JSON → process_gui_diagram → GUIModel
6. WebAppGenerator(domain_model, gui_model, agent_model)
7. Templates rendered → React/TypeScript + FastAPI backend
8. ZIP streamed to frontend
```

## Important Conventions

### Code Style
- PEP 8 with 4-space indentation and a 120-character line target (`pyproject.toml` sets pylint's `max-line-length`; CI's Ruff invocation ignores `E501`, so long lines will not fail the build — keep them short anyway)
- Type hints for public APIs, descriptive docstrings
- Naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- Import order: standard library, third-party, local modules

### Validation Strategy
Three layers on the modeling side:
1. **Construction validation**: Setter constraints in metamodel (e.g., multiplicity bounds)
2. **Metamodel validation**: `.validate()` method checks structural rules
3. **Constraint validation**: OCL constraints evaluated on models

All validation errors should be collected (not thrown) for unified reporting.

A **fourth, separate** layer applies to Spec-Driven Agent output — static checks over
*generated code* rather than over a model. Findings carry one of three severities:

| Severity  | What lands there | What the auto-fix loop does |
|-----------|------------------|-----------------------------|
| `blocker` | Python syntax errors; dependency conflicts; a Dockerfile referencing a missing file; unresolvable local imports (`missing module:`); undefined names behind a star import (`undefined name:`); an ORM module that fails to import or to configure its mappers in the subprocess smoke check (`mapper config:`); a requirement the user stated that the code does not implement, or one whose verification is incomplete (`requirement:`, `requirement partial:`, `requirement unverified:` — unknown evidence is not proof of absent behaviour, but it does block *verified* completion); a method button bound to a table of another entity; frontend-contract violations (blank-on-load router, dead submit handler, web-app request with no frontend at all, rival framework imported into the scaffold); data-contract violations; an entity the running app refuses to create for every schema-valid request (`create contract:`); an action handler observed returning 500 (`action call:`); an app that cannot boot or create a record when Phase 3 exits (`runtime gate:`); ruff `F811`/`F821`/`F822`/`F823`; `tsc`/`cargo`/`kotlinc` errors | Drives the loop. Rounds of (fix turns → re-validate); stops at zero. Ends immediately on a **replay** — a round that called no tool at all, so the tree is byte-identical and the next round would repeat it. A round whose edits were *rejected* gets a second attempt instead, since those often write source on the next round. Otherwise ends after **2** no-progress rounds or **3** on a blocker plateau. Progress counts a better score, changed source, **or a discharged obligation** (e.g. blockers closed through `test_api` / `task_list` without a write). The snapshot is re-taken on every strictly better tree and the **best** is restored, not the last, ranked by (boot broken, entities not created, actions not callable, hard blockers) — runtime first, because blocker count alone tracks poorly with whether the app works. Ledger verdicts are excluded from that count because judge passes are not stable |
| `warning` | Everything unclassified, including endpoint-coherence findings and the acceptance matrix | Recorded in the recipe only |
| `style`   | Cosmetic ruff rules: `F401`, `F841`, `E501`, whitespace, blank lines, import order | Recorded only |

When adding a validator, emit a message with a stable prefix and classify it in
`_classify_issue` — the classifier keys on prefixes, not on the validator that produced them.

### Generator Development
To add a new deterministic generator:
1. Create package in `besser/generators/[name]/`
2. Implement `GeneratorInterface` with `__init__(model, output_dir=None)` and `generate()`
3. Add templates in `generators/[name]/templates/`
4. Register in `utilities/web_modeling_editor/backend/config/generators.py` (`SUPPORTED_GENERATORS`, plus `get_filename_for_generator`)
5. Add tests in `tests/generators/[name]/`
6. Write `docs/source/generators/[name].rst` and add it to a toctree **and** the "Choosing a Generator" table in `docs/source/generators.rst`
7. If the LLM agent should be able to call it, add a tool to `besser/spec_driven_agent/agent/tools.py` **and** an entry to `_TOOL_MODEL_REQUIREMENTS`

### Resource Management
- Temp directories use the `besser_*` prefixes listed above so the cleanup task can find them
- Always use try-finally blocks for cleanup
- ZIP streaming for large outputs to avoid memory issues

## Frontend Integration

The frontend lives at `besser/utilities/web_modeling_editor/frontend` (git submodule pointing to `BESSER-PEARL/BESSER-Web-Modeling-Editor`, branch `main`). It has its own `CLAUDE.md` with detailed instructions for working in the frontend codebase.

```bash
# Initialize the submodule (first time)
git submodule update --init --recursive

# Fast-forward the submodule to the tip of the branch .gitmodules tracks
git submodule update --remote besser/utilities/web_modeling_editor/frontend
git add besser/utilities/web_modeling_editor/frontend
```

**Cross-repo changes**: If modifying both frontend and backend, implement each side in its respective repo, update the submodule pointer, and link both PRs with notes on merge order.

**Backend API contract**: If you change backend endpoints or request/response shapes, coordinate with the frontend's `shared/api/` layer. For the Spec-Driven Agent specifically, the contract is `models/spec_driven.py` (request) and `services/spec_driven/sse_events.py` (stream) — both are consumed by the frontend's spec-driven trigger.

## Testing Approach

- Place tests in `tests/` mirroring source structure
- Name test files `test_*.py`
- Add tests for behavioral changes, especially metamodel and generator logic
- **Centralized fixtures** in `tests/conftest.py` provide shared models (e.g., `library_book_author_model`, `employee_self_assoc_model`, `simple_library_book_model`, `player_team_domain_model`). Prefer reusing these over duplicating test models.
- Additional domain-specific fixtures in `tests/generators/conftest.py`
- Validate both structure (class names, endpoints) and content (business logic)
- `pyproject.toml` configures `--import-mode=importlib` to prevent namespace collisions between test and source packages

`tests/utilities/web_modeling_editor/backend/test_spreadsheet_import.py` imports `openpyxl`
at module level, so without it pytest stops at *collection* — an error, not a test failure.
`openpyxl` ships in the backend requirements file. `torch` / `tensorflow` are **not** needed:
`tests/generators/nn/` passes without them (CI still ignores that directory).

## CI/CD Pipelines

- **`.github/workflows/ci.yml`**: Three jobs on PRs to `master`/`development` — tests on Python **3.11 and 3.12**, Ruff lint (the exact invocation above), and a docs build gated by `docs/check-docs-warnings.sh`. It does **not** build the frontend.
- **`.github/workflows/security.yml`**: CodeQL security scanning.
- **`.github/workflows/deploy-wme.yml`**: Manual (`workflow_dispatch`) build + push of the backend / frontend / agent-simulator images and deploy to EC2; a backend deploy also recreates the `besser-wme-smartgen` worker and verifies the running build stamp.
- **`.github/workflows/python-publish.yml`**: PyPI release.

There is no `.github/dependabot.yml` in this repository.

## Documentation Sync

Keep docs in `docs/source/` synchronized with code changes:
- `buml_language.rst` - Metamodel additions (and the notation-support matrix)
- `generators.rst` - New generator documentation (toctree **and** the choosing table)
- `spec_driven_agent/` - Spec-Driven Agent: pipeline, tools, severities, caps, config
- `web_editor.rst` - Editor workflows and the spec-driven API contract
- `web_editor_backend.rst` - Endpoint and environment-variable tables
- `utilities.rst` - New utility documentation
- `contributor_guide.rst`, `ai_assistant_guide.rst` - Workflow changes

Build locally before committing: `cd docs && make html` (and `bash docs/check-docs-warnings.sh` for the CI verdict).

## Commit Conventions

Recent history uses Conventional Commits style:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code restructuring
- `docs:` - Documentation updates
- `test:` - Test additions/modifications

Keep subjects short and imperative. Use topic branches (`feature/add-generator`).
**Open pull requests against `development`, not `master`.**

## Related Files

- **`.github/copilot-instructions.md`**, **`.cursorrules`**: Pointers to this file for Copilot and Cursor; edit `CLAUDE.md`, not them
- **`CONTRIBUTING.md`**: Contribution workflow and expectations
- **`DEVELOPMENT_SETUP.md`**: Local environment setup
- **`README.md`**: Project overview and quick start
- **`GOVERNANCE.md`**: Project governance and decision-making
- **`docs/source/contributor_guide.rst`**: The long-form version of this file, for humans

## Key Technical Patterns

### Multiplicity Constant
```python
UNLIMITED_MAX_MULTIPLICITY = 9999  # besser/BUML/metamodel/structural/structural.py
```

### Metamodel Base Hierarchy
```
Element (base) → NamedElement → {Class, Property, Method, Association, ...}
```

### Generator Registry Pattern
Generators registered in `config/generators.py` with metadata:
```python
GeneratorInfo(
    generator_class=DjangoGenerator,
    output_type="zip",            # "file" or "zip"
    file_extension=".zip",
    category="web_framework",
    requires_class_diagram=True,  # whether it needs a class diagram as input
    required_diagram_type=None,   # e.g. "NNDiagram" / "QuantumCircuitDiagram" / BPMN
)
```
Neural network generators (PyTorch, TensorFlow) are conditionally registered only when their dependencies are installed.

### Bidirectional Converters
Always maintain symmetry:
- `json_to_buml/class_diagram_processor.py` ↔ `buml_to_json/class_diagram_converter.py`
- Same features supported in both directions

### Template Rendering
```python
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates/'))
template = env.get_template('model.py.j2')
output = template.render(model=domain_model, config=config)
```

## Common Pitfalls to Avoid

1. **Don't duplicate logic**: Shared helpers belong in `besser/utilities`, not in individual generators
2. **Maintain determinism**: Generators should produce identical output for identical input (avoid timestamps in file names)
3. **Validate early**: Use construction validation in setters to fail fast
4. **Clean up resources**: Always use try-finally for temp directories and file handles
5. **Keep converters symmetric**: If JSON→BUML supports a feature, BUML→JSON must too
6. **Test round-trips**: Especially for converters (JSON→BUML→JSON should be identity)
7. **Update docs**: Backend changes often require `docs/source/` updates
8. **Keep the security defaults**: `BESSER_LLM_ALLOW_CUSTOM_BASE_URL` and `BESSER_LLM_ENABLE_SHELL_TOOLS` both default off in code; deployments opt in per service (see the Spec-Driven Agent section). Don't flip the code defaults
9. **A new agent tool needs two edits**: `agent/tools.py` *and* `_TOOL_MODEL_REQUIREMENTS`, or it will be offered on projects that cannot satisfy it

## Debugging Tips

### Running Individual Examples
```bash
cd tests/BUML/metamodel/structural/library
python library.py
```

### Inspecting Generated Output
Deterministic generators default to `<cwd>/output` when `output_dir` is not passed.

### Debugging a Spec-Driven run
Each run workspace holds `.besser_trace.jsonl` (every phase, turn, tool call, cost update,
compaction, snapshot, rollback and validation finding), `.besser_checkpoint.json` (present
only if Phase 2 did not exit cleanly — that is what makes the run resumable), and
`.besser_recipe.json` (the final summary, including validation issues and the authorship
split). The durable event store is SQLite, at `BESSER_LLM_RUN_STORE_PATH` or the system
temp directory.

### Validation Debugging
`/validate-diagram` returns every metamodel and OCL error in one response (errors are collected, not raised); start there before reading validator code.

### Frontend-Backend Integration
Use browser DevTools Network tab to inspect API payloads. Backend returns detailed error messages — except for unhandled exceptions, which are deliberately flattened to "Internal server error"; the traceback is in the server log.

## Support Resources

- Documentation: https://besser.readthedocs.io/
- Online Editor: https://editor.besser-pearl.org/
- Examples: https://github.com/BESSER-PEARL/BESSER-examples
- Contributor Guide: `docs/source/contributor_guide.rst`
- AI Assistant Guide: `docs/source/ai_assistant_guide.rst`
