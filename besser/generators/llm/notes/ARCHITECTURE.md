# LLM Generator Architecture

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                   │
│  Domain Model (BUML) + Instructions + [GUI Model] + [Agent]     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 0: GENERATOR SELECTION                   │
│                                                                  │
│  LLM call (cheap, 1 turn):                                      │
│    "User wants X. Here are ALL BESSER generators:                │
│     generate_web_app, generate_fastapi_backend, generate_django, │
│     generate_react, generate_flutter, generate_pydantic, ...     │
│     Which one fits best?"                                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ LLM answers  │  │ LLM fails?   │  │ Keywords     │          │
│  │ "generate_   │  │ (API error)  │──▶│ fallback     │          │
│  │  fastapi_    │  └──────────────┘  │ "fastapi" →  │          │
│  │  backend"    │                     │  fastapi_gen │          │
│  └──────┬───────┘                     └──────┬───────┘          │
│         └──────────────┬─────────────────────┘                  │
└────────────────────────┼────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │ Generator found?        │
            ├─── YES ──┐   ┌── NO ───┤
            │          ▼   ▼         │
            │                        │
┌───────────┴─────────┐  ┌──────────┴──────────────────────────┐
│ PHASE 1: GENERATE   │  │ SKIP Phase 1                        │
│                      │  │ LLM writes everything from scratch  │
│ Run the generator    │  │ in Phase 2                          │
│ (deterministic,      │  └─────────────────────────────────────┘
│  no LLM)             │
│                      │
│ generate_fastapi_    │
│ backend() →          │
│  backend/            │
│   main_api.py        │
│   sql_alchemy.py     │
│   pydantic_classes.py│
│   requirements.txt   │
│                      │
│ Validate output:     │
│  ast.parse() .py     │
│  Dockerfile refs     │
│                      │
│ Snapshot workspace   │
│ Build inventory:     │
│  "4 files, CRUD for  │
│   User, Post, ..."   │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1.5: GAP ANALYSIS                             │
│                                                                  │
│  LLM call (cheap, 1 turn):                                      │
│    "Generator produced CRUD + ORM + schemas.                     │
│     User asked for: JWT auth, PostgreSQL, pagination, search.    │
│     What specific tasks are needed?"                             │
│                                                                  │
│  → Returns task list:                                            │
│    1. Write backend/auth.py (JWT)                                │
│    2. Modify sql_alchemy.py (PostgreSQL)                         │
│    3. Add pagination to GET endpoints                            │
│    4. Write Dockerfile + docker-compose                          │
│    5. Write README.md                                            │
│                                                                  │
│  Falls back to keyword matching if LLM fails                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: LLM CUSTOMIZATION                          │
│                                                                  │
│  System prompt:                                                  │
│    "Here's what exists: [inventory]                              │
│     Your tasks: [gap analysis results]                           │
│     Rules: modify don't rewrite, model is truth"                 │
│                                                                  │
│  Tools available (ALL):                                          │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│    │ File tools   │ │ Exec tools   │ │ Generators   │          │
│    │ read_file    │ │ run_command  │ │ gen_pydantic │          │
│    │ write_file   │ │ install_deps │ │ gen_fastapi  │          │
│    │ modify_file  │ │ check_syntax │ │ gen_django   │          │
│    │ search_files │ │              │ │ gen_react    │          │
│    │ list_files   │ │              │ │ ...          │          │
│    └──────────────┘ └──────────────┘ └──────────────┘          │
│                                                                  │
│  Agent loop:                                                     │
│    ┌─────────┐     ┌──────────┐     ┌──────────┐               │
│    │ LLM     │────▶│ Tool     │────▶│ Result   │──┐            │
│    │ thinks  │     │ executor │     │ back to  │  │            │
│    └─────────┘     └──────────┘     │ LLM      │  │            │
│         ▲                           └──────────┘  │            │
│         └─────────────────────────────────────────┘            │
│                                                                  │
│  Safety checks each turn:                                        │
│    ├── Cost > $5.00? → stop                                      │
│    ├── Time > 20 min? → stop                                     │
│    ├── Same tool 4x? → warn LLM                                 │
│    └── Context too big? → compact old messages                   │
│                                                                  │
│  Parallel execution:                                             │
│    If LLM calls 5 tools at once → ThreadPoolExecutor             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: VALIDATION & FIX                           │
│                                                                  │
│  Auto-checks (no LLM):                                          │
│    ├── ast.parse() all .py files                                 │
│    ├── Dockerfile refs exist?                                    │
│    ├── npm ci → npm install auto-fix                             │
│    ├── pip install --dry-run (catch ALL dep conflicts)           │
│    └── Count issues                                              │
│                                                                  │
│  If issues found:                                                │
│    → Log each issue clearly for user visibility                  │
│    → Give LLM 5 turns to fix                                     │
│    → If fixes make things WORSE → rollback to pre-Phase-3 state  │
│    → Remaining issues shown to user + saved in recipe            │
│                                                                  │
│  If no issues:                                                   │
│    → "Validation passed"                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│                                                                  │
│  ./output/my_app/                                                │
│    backend/                                                      │
│      main_api.py ......... (from generator + LLM modifications)  │
│      sql_alchemy.py ...... (from generator + LLM modifications)  │
│      pydantic_classes.py . (from generator)                      │
│      auth.py ............. (written by LLM)                      │
│      requirements.txt .... (from generator + LLM additions)      │
│      Dockerfile .......... (written by LLM)                      │
│    docker-compose.yml .... (written by LLM)                      │
│    README.md ............. (written by LLM)                      │
│    .besser_recipe.json ... (audit log)                           │
│                                                                  │
│  Recipe includes:                                                │
│    - model summary (classes, enums, associations)                │
│    - generator used                                              │
│    - all tool calls with inputs                                  │
│    - files: which from generator, which from LLM                 │
│    - token usage + estimated cost                                │
│    - elapsed time, turns, compactions                            │
│                                                                  │
│  Available via:                                                  │
│    ├── Python: LLMGenerator(model, instructions, provider=...)   │
│    ├── Providers: Anthropic (Claude) or OpenAI (GPT)             │
│    └── Interactive: orchestrator.fix_error("paste error here")   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
besser/generators/llm/
├── __init__.py          — Public API: LLMGenerator, LLMProvider, OpenAIProvider
├── llm_generator.py     — LLMGenerator(GeneratorInterface) — entry point
├── orchestrator.py      — Phase 0→1→2→3 coordinator
├── llm_client.py        — LLMProvider ABC, ClaudeLLMClient, OpenAIProvider
├── tool_executor.py     — Sandboxed tool execution (file, shell, generators)
├── tools.py             — Tool definitions (generators, file, execution, validation)
├── model_serializer.py  — BUML model → compact JSON for LLM context
├── gap_analyzer.py      — LLM-based + keyword-based gap analysis
├── prompt_builder.py    — System prompt + inventory construction
└── compaction.py        — Context compaction (token estimation, summarization)
```

## Features

| Feature | How |
|---|---|
| Multi-provider | Anthropic Claude + OpenAI GPT via `provider=` parameter |
| LLM generator selection | LLM picks the best generator, keyword fallback |
| LLM gap analysis | LLM analyzes what's missing, keyword fallback |
| Prompt caching | System prompt + tools cached across turns (~8x cost savings) |
| Cost tracking | Input/output/cache tokens tracked, USD cost estimated |
| Cost cap | Abort if `max_cost_usd` exceeded (default $5) |
| Runtime timeout | Abort if `max_runtime_seconds` exceeded (default 20 min) |
| Streaming | Real-time text output via `on_text` callback |
| Context compaction | Old messages summarized when context grows too large |
| Parallel execution | Multiple tool calls run concurrently (ThreadPoolExecutor) |
| Snapshot/rollback | Workspace snapshot before Phase 3, rollback if fixes make things worse |
| Phase 3 validation | Auto-checks: syntax, imports, Dockerfile refs, npm ci fix |
| Interactive fix | `orchestrator.fix_error("paste error")` resumes with error context |
| Retry with backoff | 3 attempts, exponential backoff on 429/5xx/timeouts |
| Loop detection | Warns LLM after 4 consecutive identical tool calls |
| Recipe logging | Full audit trail in `.besser_recipe.json` |
| Interactive fix | `orchestrator.fix_error("paste error")` or interactive prompt after generation |
