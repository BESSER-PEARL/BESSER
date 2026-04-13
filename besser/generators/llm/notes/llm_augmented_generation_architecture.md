# LLM-Augmented Code Generation Architecture
 claude --resume 8d98fa38-0643-40b0-86d4-ac20bdc0f215    
## The Core Idea

BESSER has 18+ deterministic code generators that produce reliable, tested code from domain models. But they're rigid — they can't add JWT auth, custom business logic, deployment configs, or domain-specific features.

LLMs (Claude, GPT) can write any code, but they hallucinate imports, invent nonexistent APIs, and produce non-reproducible output.

**This system combines both**: the LLM uses BESSER's generators as tools, producing a reliable deterministic base, then makes targeted modifications to add custom features. The domain model anchors everything — the LLM can't hallucinate entities that don't exist in the model.

```
┌─────────────────────────────────────────────────────┐
│           User: "Build me a web app with            │
│           JWT auth, PostgreSQL, dark mode"           │
│                                                     │
│  ┌──────────┐                                       │
│  │  B-UML   │  The model is the source of truth.    │
│  │  Model   │  User, Post, Comment with validated   │
│  │          │  relationships and OCL constraints.    │
│  └────┬─────┘                                       │
│       │                                             │
│       v                                             │
│  ┌──────────────────────────────────────────────┐   │
│  │            LLM Agent (Claude)                │   │
│  │                                              │   │
│  │  Phase 1: ORCHESTRATE generators             │   │
│  │  ├─ generate_fastapi_backend()               │   │
│  │  ├─ generate_react()                         │   │
│  │  └─ generate_sqlalchemy(dbms="postgresql")   │   │
│  │                                              │   │
│  │  Phase 2: ENHANCE generated code             │   │
│  │  ├─ read generated files                     │   │
│  │  ├─ add JWT middleware to FastAPI             │   │
│  │  ├─ add auth context to React                │   │
│  │  ├─ add dark mode theme provider             │   │
│  │  └─ write Dockerfiles + docker-compose       │   │
│  └──────────────────────────────────────────────┘   │
│       │                                             │
│       v                                             │
│  ┌──────────┐                                       │
│  │  Output  │  80% from generators (reliable)       │
│  │  .zip    │  20% from LLM (customized)            │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

---

## Why This Beats the Alternatives

### vs. Pure LLM (Lovable, v0, Bolt)

These tools generate everything from scratch with an LLM. They're impressive for demos but fragile in practice.

| Problem with pure LLM | How BESSER hybrid solves it |
|---|---|
| **Hallucinates entities** — invents a `Payment` model that doesn't exist | Model is the source of truth. LLM receives the validated model and cannot create entities outside it. If the user needs Payment, they model it first. |
| **Wrong relationships** — guesses `User has many Posts` when the model says otherwise | Associations come from the model with exact multiplicities. LLM reads them, doesn't invent them. |
| **Broken imports** — `from fastapi import something_that_doesnt_exist` | Generators produce correct imports every time. They're tested in CI. |
| **Non-reproducible** — run twice, get different code | Base code from generators is deterministic. Only LLM edits vary. Two runs with same model + generators = same foundation. |
| **Expensive** — LLM writes 100% of the code | Generators produce 80% for free (CPU only). LLM handles the 20% that requires creativity. |
| **No validation** — structural errors slip through | OCL constraints checked before generation. Model validates multiplicities, types, inheritance. |
| **Can't handle large codebases** — context window fills up | Generators handle the bulk. LLM only reads files it needs to modify — targeted, not exhaustive. |

### vs. Pure Deterministic (Current BESSER)

Current BESSER generators produce correct but generic code. A `DjangoGenerator` always produces the same Django project — no auth, no custom middleware, no deployment config.

| Limitation of pure generators | How LLM augmentation solves it |
|---|---|
| **No business logic** — generators can't add "if order total > $100, apply 10% discount" | LLM reads the generated code and adds custom logic in the right places |
| **No auth/permissions** — every generated API is open | LLM adds JWT/OAuth middleware, role-based access, login/register endpoints |
| **No deployment** — generators produce code, not infrastructure | LLM generates Dockerfiles, docker-compose, CI/CD configs, env files |
| **Rigid UI** — React generator produces template-driven pages | LLM adds theming, animations, custom components, responsive design |
| **One-size-fits-all** — same model always produces same output | Different instructions → different customizations. "Make it healthcare-compliant" vs. "make it a marketplace" |
| **No cross-cutting concerns** — logging, monitoring, error handling | LLM adds structured logging, Sentry integration, error boundaries |

### The Hybrid Advantage: Auditable, Explainable, Cost-Effective

```
Generator output:  main_api.py (200 lines, deterministic, tested)
LLM modifications:  +15 lines JWT middleware, +8 lines pagination
                    = 23 lines of LLM code to audit (not 200)
```

You can `diff` the generator baseline against the LLM-enhanced version. The LLM delta is small, focused, and auditable. If something breaks, you know it's in the 23 lines the LLM touched, not the 200 the generator produced.

---

## Architecture: Two Layers

### Layer 1: Python Library — `besser/generators/llm/`

A new generator that lives alongside existing generators. Usable by anyone with `pip install besser`:

```python
from besser.generators.llm import LLMGenerator

generator = LLMGenerator(
    model=domain_model,
    gui_model=gui_model,
    instructions="Build a web app with JWT auth, PostgreSQL, dark mode",
    api_key="sk-ant-...",
)
generator.generate()
# Output in ./output/ — ready to run
```

This follows BESSER's `GeneratorInterface` pattern exactly. It's a generator like any other, just smarter.

### Layer 2: Web Editor — New backend endpoint + frontend dialog

The web editor exposes it via `POST /llm-generate/from-project` — same pattern as all other generators. The frontend adds an "AI Generation" option.

---

## Component Design

### Component 1: Model Serializer — How the LLM Sees the Model

**File: `besser/generators/llm/model_serializer.py`**

The LLM never sees Python code. It receives a structured JSON representation of the domain model that's compact and unambiguous.

**Why JSON, not Python code?**
- LLMs reason better about structured data than code
- JSON is token-efficient (~2-3KB for a typical 10-class model)
- No parsing ambiguity
- Can't be confused with code to execute

**What it produces** (example for a blog app model):

```json
{
  "name": "BlogApp",
  "classes": [
    {
      "name": "User",
      "is_abstract": false,
      "attributes": [
        {"name": "id", "type": "int", "is_id": true, "multiplicity": "1..1"},
        {"name": "email", "type": "str", "is_id": false, "multiplicity": "1..1"},
        {"name": "username", "type": "str", "is_id": false, "multiplicity": "1..1"},
        {"name": "bio", "type": "str", "is_id": false, "multiplicity": "0..1"}
      ],
      "methods": [],
      "parents": []
    },
    {
      "name": "Post",
      "attributes": [
        {"name": "id", "type": "int", "is_id": true, "multiplicity": "1..1"},
        {"name": "title", "type": "str", "multiplicity": "1..1"},
        {"name": "content", "type": "str", "multiplicity": "1..1"},
        {"name": "published", "type": "bool", "multiplicity": "1..1"}
      ],
      "methods": [],
      "parents": []
    }
  ],
  "enumerations": [
    {"name": "PostStatus", "literals": ["DRAFT", "PUBLISHED", "ARCHIVED"]}
  ],
  "associations": [
    {
      "name": "User_Post",
      "ends": [
        {"role": "author", "class": "User", "multiplicity": "1..1", "navigable": true},
        {"role": "posts", "class": "Post", "multiplicity": "0..*", "navigable": true, "composite": false}
      ]
    }
  ],
  "constraints": [
    {"context": "User", "expression": "self.email->matches('[a-zA-Z0-9.]+@[a-zA-Z0-9.]+')"}
  ]
}
```

Similarly for `GUIModel` — serializes screens, components, data bindings, navigation transitions.

### Component 2: Tool Definitions — What the LLM Can Do

**File: `besser/generators/llm/tools.py`**

Three categories of tools:

#### Generator Tools (call existing BESSER generators)

Each existing generator becomes a callable tool with typed parameters. Tool definitions are built from the generator registry so new generators are automatically available.

| Tool | What it does | Key parameters |
|---|---|---|
| `generate_pydantic` | Pydantic BaseModel classes with validators | `backend`, `nested_creations` |
| `generate_sqlalchemy` | SQLAlchemy ORM with relationships | `dbms` (sqlite/postgresql/mysql/...) |
| `generate_fastapi_backend` | Complete FastAPI + ORM + Pydantic | `http_methods` |
| `generate_react` | React TypeScript app (needs GUI model) | — |
| `generate_django` | Full Django project | `project_name`, `app_name` |
| `generate_python` | Plain Python classes | — |
| `generate_java` | Java classes | — |
| `generate_sql` | Raw SQL CREATE TABLE statements | — |
| `generate_jsonschema` | JSON Schema from model | `mode` (regular/smart_data) |
| `generate_rest_api` | FastAPI REST endpoints | `http_methods` |
| `generate_flutter` | Flutter app (needs GUI model) | — |
| `generate_web_app` | Full-stack (React + FastAPI + Agent + Docker) | — |
| `generate_agent` | BESSER Agent Framework code | `generation_mode` |
| `generate_terraform` | Terraform deployment | — |

#### File Tools (read/modify generated code)

| Tool | What it does |
|---|---|
| `read_file` | Read contents of a file from the workspace |
| `write_file` | Create or overwrite a file |
| `modify_file` | Targeted search-and-replace edit (for surgical modifications) |
| `list_files` | List all files in the workspace with sizes |

#### Validation Tools

| Tool | What it does |
|---|---|
| `validate_model` | Run structural validation on the domain model |
| `check_python_syntax` | Verify a Python file parses correctly |

### Component 3: Tool Executor — Runs Tools Safely

**File: `besser/generators/llm/tool_executor.py`**

Executes tool calls in an isolated temp directory. Each generator creates its own subdirectory. File operations are path-traversal protected.

Key design decisions:
- **Path sandboxing**: All file operations validated against workspace root — no path traversal
- **Generator isolation**: Each generator writes to its own subdirectory (`pydantic/`, `backend/`, `frontend/`)
- **Error containment**: Tool failures return error strings to the LLM (so it can try a different approach) rather than crashing the entire pipeline

### Component 4: LLM Client — Anthropic Claude Adapter

**File: `besser/generators/llm/llm_client.py`**

Thin adapter around the Anthropic SDK. Uses Claude's native tool-use API.

```python
class ClaudeLLMClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, system: str, messages: list, tools: list) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16384,
            system=system,
            messages=messages,
            tools=tools,
        )
        return {
            "stop_reason": response.stop_reason,
            "content": response.content,
        }
```

**Why Claude?**
- Best tool-use capabilities (structured tool calling, not string-based)
- Strong code generation and modification ability
- Long context window for reading generated files
- Reliable at following system prompt instructions

**API key resolution** (following existing BESSER pattern from `agent_personalization.py`):
1. Direct parameter
2. Config dict keys: `api_key`, `anthropic_api_key`, `ANTHROPIC_API_KEY`
3. Environment variable `ANTHROPIC_API_KEY`

### Component 5: Orchestrator — The Agent Loop

**File: `besser/generators/llm/orchestrator.py`**

The core intelligence. Manages the conversation between Claude and the tools.

**The system prompt is critical** — it shapes how the LLM behaves:

```
You are a senior full-stack developer using the BESSER low-code platform.

You have a validated domain model and code generators as tools.
Your job: build what the user asks by combining generators with custom code.

## STRATEGY (follow this order):
1. CALL GENERATORS FIRST — never manually write what a generator can produce
2. READ the generated files to understand the baseline
3. MODIFY with targeted edits to add custom requirements
4. WRITE new files only for things no generator covers

## Domain Model (verified, validated — do NOT invent entities)
{model_json}

## GUI Model (if present)
{gui_json}

## Rules
- Generator output is tested and reliable — use it as the foundation
- Use modify_file for surgical edits, not full rewrites
- If you need an entity that isn't in the model, tell the user
- When done, summarize what you built and how to run it
```

**Agent loop**:

```
User instruction
    │
    v
┌─────────────────────────────────┐
│  Claude processes instruction   │
│  + model context + tools        │
│                                 │
│  Decides: "I need a FastAPI     │
│  backend first"                 │
│                                 │
│  → tool_use: generate_backend() │
└──────────────┬──────────────────┘
               │
               v
┌──────────────────────────────────┐
│  ToolExecutor runs BackendGen    │
│  → main_api.py, sql_alchemy.py  │
│  → returns file listing          │
└──────────────┬───────────────────┘
               │
               v
┌─────────────────────────────────┐
│  Claude reads the files         │
│  → tool_use: read_file(...)     │
│                                 │
│  Understands the baseline       │
│  Decides what to customize      │
│                                 │
│  → tool_use: modify_file(...)   │
│    (adds JWT, pagination, etc.) │
└──────────────┬──────────────────┘
               │
               v
          (repeat until done)
               │
               v
┌─────────────────────────────────┐
│  Claude: "Done. Here's what     │
│  I built and how to run it."    │
│  → stop_reason: end_turn        │
└─────────────────────────────────┘
```

**Safety limits**:
- `MAX_TURNS = 30` — hard cap on iterations
- Anthropic API timeout per call
- File operations sandboxed to workspace directory

### Component 6: The Generator Class — Library Interface

**File: `besser/generators/llm/llm_generator.py`**

Implements `GeneratorInterface` — same as every other BESSER generator:

```python
class LLMGenerator(GeneratorInterface):
    """
    LLM-augmented code generator.

    Uses Anthropic Claude to orchestrate BESSER's deterministic generators
    and customize their output based on natural language instructions.

    The LLM has access to all BESSER generators as callable tools.
    It calls them for reliable base code, then modifies the output
    to match user requirements.
    """

    def __init__(self, model: DomainModel, instructions: str,
                 api_key: str, llm_model: str = "claude-sonnet-4-20250514",
                 gui_model=None, agent_model=None, agent_config=None,
                 output_dir: str = None):
        super().__init__(model, output_dir)
        self.instructions = instructions
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.llm_client = ClaudeLLMClient(api_key=api_key, model=llm_model)

    def generate(self):
        output = self.build_generation_dir()
        orchestrator = LLMOrchestrator(
            llm_client=self.llm_client,
            domain_model=self.model,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            agent_config=self.agent_config,
            output_dir=output,
        )
        orchestrator.run(self.instructions)
```

**Usage as a library** (outside the web editor):

```python
from besser.BUML.metamodel.structural import *
from besser.generators.llm import LLMGenerator

# Define model (or load from file)
User = Class(name="User", attributes={
    Property(name="email", type=StringType, is_id=True),
    Property(name="name", type=StringType),
})
Post = Class(name="Post", attributes={
    Property(name="title", type=StringType),
    Property(name="content", type=StringType),
})
model = DomainModel(name="Blog", types={User, Post}, associations={...})

# Generate with LLM
generator = LLMGenerator(
    model=model,
    instructions="""
    Build a production-ready FastAPI backend with:
    - JWT authentication (email + password)
    - PostgreSQL with Alembic migrations
    - Rate limiting on all endpoints
    - Structured logging with request tracing
    - Docker + docker-compose for deployment
    """,
    api_key=os.environ["ANTHROPIC_API_KEY"],
)
generator.generate()  # Output in ./output/
```

### Component 7: Web Editor Endpoint

**File: `backend/routers/llm_generation_router.py`**

Follows the same pattern as `generation_router.py`:

```python
router = APIRouter(prefix="/llm-generate", tags=["llm-generation"])

@router.post("/from-project")
@handle_endpoint_errors
async def llm_generate_from_project(input_data: ProjectInput):
    """LLM-augmented generation from a multi-diagram project."""
    config = input_data.settings or {}
    instructions = config.get("instructions", "")
    api_key = config.get("api_key", "")
    llm_model = config.get("llm_model", "claude-sonnet-4-20250514")

    # Resolve models from project diagrams (same as generation_router)
    domain_model = process_class_diagram(...)
    gui_model = process_gui_diagram(...) if has_gui else None
    agent_model = process_agent_diagram(...) if has_agent else None

    # Create and run LLM generator
    with tempfile.TemporaryDirectory(prefix="besser_llm_") as tmpdir:
        generator = LLMGenerator(
            model=domain_model,
            gui_model=gui_model,
            agent_model=agent_model,
            instructions=instructions,
            api_key=api_key,
            llm_model=llm_model,
            output_dir=tmpdir,
        )
        await asyncio.to_thread(generator.generate)

        # ZIP and stream (same as generation_router)
        zip_path = shutil.make_archive(...)
        return StreamingResponse(...)
```

Register in `backend.py`:
```python
from .routers.llm_generation_router import router as llm_generation_router
app.include_router(llm_generation_router, prefix="/besser_api")
```

### Component 8: Frontend Integration

In the web editor frontend, add an "AI Generation" option:

**New in `features/generation/`:**
- `LLMGenerationDialog.tsx` — dialog with:
  - Instruction text area (multiline, rich text)
  - Claude model selector (sonnet/opus)
  - API key input (stored in localStorage, never sent to BESSER backend — goes directly to Anthropic)
  - "Generate" button
- Calls `POST /besser_api/llm-generate/from-project` with the current project + instructions
- Shows progress (initial: spinner; future: streaming tool-call updates via SSE)
- Returns a downloadable ZIP

---

## The "Improve" Flow

The user specifically mentioned: "if he need he can improve the code generated deterministically."

This means a two-step workflow:

### Step 1: Deterministic Generation (existing flow)
User clicks "Generate" → picks "FastAPI Backend" → gets `main_api.py`, `sql_alchemy.py`, `pydantic_classes.py`

### Step 2: AI Enhancement (new flow)
User clicks "Enhance with AI" → enters instructions → LLM reads the already-generated files and makes targeted improvements.

**This is a special mode of the orchestrator** where the workspace is pre-populated with generator output:

```python
class LLMOrchestrator:
    def run_enhancement(self, existing_files_dir: str, instructions: str) -> str:
        """Enhance already-generated code with LLM modifications."""
        # Copy existing files into workspace
        shutil.copytree(existing_files_dir, self.output_dir, dirs_exist_ok=True)

        # System prompt emphasizes: DON'T regenerate, IMPROVE what exists
        system = f"""You are enhancing existing BESSER-generated code.
        The workspace already contains generated files. Your job is to
        improve them based on the user's instructions.

        DO NOT regenerate files. Read them, understand them, then make
        targeted modifications.

        {model_context}
        """

        messages = [{"role": "user", "content": instructions}]
        # ... standard agent loop with file tools only (no generator tools) ...
```

**Frontend flow**: After deterministic generation, a "Enhance with AI" button appears on the download dialog. User enters instructions like "Add pagination to all list endpoints" or "Add dark mode support" and the LLM modifies the already-generated code.

---

## Advanced: Generation Recipe Logging

Every tool call the LLM makes can be logged as a "recipe":

```json
{
  "model_hash": "abc123",
  "instructions": "Build web app with JWT auth, PostgreSQL",
  "steps": [
    {"tool": "generate_fastapi_backend", "args": {"http_methods": ["GET","POST","PUT","DELETE"]}},
    {"tool": "read_file", "args": {"path": "backend/main_api.py"}},
    {"tool": "modify_file", "args": {"path": "backend/main_api.py", "old_text": "...", "new_text": "..."}},
    ...
  ]
}
```

This recipe can be:
- **Replayed** without an LLM (deterministic reproduction of a specific generation)
- **Shared** as a template ("JWT auth recipe", "PostgreSQL recipe")
- **Audited** to see exactly what the LLM changed
- **Version-controlled** alongside the model

This is a future enhancement but the architecture supports it from day one by logging all tool calls.

---

## Cost Analysis

### Per-generation cost estimate

For a typical web app generation (10-class model, JWT auth, PostgreSQL, React frontend):

| Component | API calls | Tokens (est.) | Cost (Claude Sonnet) |
|---|---|---|---|
| System prompt + model context | 1 | ~3,000 input | ~$0.009 |
| Generator tool calls (5-6 calls) | 5-6 turns | ~15,000 total | ~$0.045 |
| File reads (8-10 files) | 8-10 turns | ~30,000 total | ~$0.09 |
| Modifications (10-15 edits) | 10-15 turns | ~20,000 total | ~$0.06 |
| **Total** | **~30 turns** | **~68,000 tokens** | **~$0.20** |

Compare: Lovable/v0 generating the same app purely with LLM = ~200,000+ tokens = ~$0.60+

**The hybrid approach is 3x cheaper** because generators produce most of the code for free.

---

## New File Structure

```
besser/generators/llm/
├── __init__.py               — Exports LLMGenerator
├── llm_generator.py          — LLMGenerator(GeneratorInterface)
├── orchestrator.py           — Agent loop (Claude <-> tools)
├── model_serializer.py       — DomainModel/GUIModel -> JSON for LLM
├── tools.py                  — Tool definitions (generators + files + validation)
├── tool_executor.py          — Sandboxed tool execution
└── llm_client.py             — Anthropic Claude SDK adapter

besser/utilities/web_modeling_editor/backend/routers/
└── llm_generation_router.py  — Web editor endpoint

tests/generators/llm/
├── test_model_serializer.py  — Model -> JSON conversion tests
├── test_tool_executor.py     — Tool execution + path safety tests
└── test_orchestrator.py      — Integration test with mock LLM
```

## Dependencies

Add to `requirements.txt` (optional, like torch/tensorflow):
```
anthropic>=0.40.0  # Only needed for LLM generation
```

In `setup.cfg`:
```ini
[options.extras_require]
llm =
    anthropic>=0.40.0
```

Usage: `pip install besser[llm]`

## Implementation Phases

### Phase 1: Core Engine (library-only, no web editor)
1. `model_serializer.py` — serialize DomainModel + GUIModel to JSON
2. `tools.py` — define all tool schemas
3. `tool_executor.py` — implement tool handlers + path safety
4. `llm_client.py` — Claude adapter
5. `orchestrator.py` — agent loop
6. `llm_generator.py` — GeneratorInterface wrapper
7. Tests for each component

### Phase 2: Web Editor Integration
1. `llm_generation_router.py` — FastAPI endpoint
2. Register router in `backend.py`
3. Frontend `LLMGenerationDialog.tsx`
4. End-to-end testing

### Phase 3: Enhancement Mode
1. "Enhance with AI" flow (pre-populated workspace)
2. Frontend integration (post-generation enhancement button)
3. Recipe logging

### Phase 4: Advanced Features
1. SSE streaming for progress updates
2. Recipe replay (deterministic reproduction)
3. Recipe sharing/templates
4. Multi-provider support (OpenAI adapter)
