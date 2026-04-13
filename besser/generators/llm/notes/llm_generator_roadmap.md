# LLM Generator Roadmap

## Current State: 7.9/10 — Production MVP

The agent works: two-phase architecture (deterministic generator + LLM customization), prompt caching, streaming, cost tracking, retry logic, context compaction, Phase 3 validation. Tested on real projects (blog backend, NexaCRM).

Compared to Claude Code: ~50% sophistication. The gap is in planning, quality gates, failure recovery, and parallel execution.

---

## Quick Wins 

### 1. Cost Cap (`max_cost_usd`)
- Add parameter to LLMGenerator and orchestrator
- Check after each API call: if `usage.estimated_cost > max_cost_usd`, abort
- Default: $5.00 (prevents accidents)
- Print warning when approaching limit (80%) ok with defualjt 

### 2. Runtime Timeout (`max_runtime_seconds`)
- Add parameter with default 600 (10 minutes) can but 20 min default
- Check `time.monotonic() - start_time` at the start of each turn
- If exceeded, save what we have and exit gracefully

### 3. Validate Phase 1 Output
- After generator runs, before Phase 2 starts:
  - `py_compile` all .py files
  - Check Dockerfiles reference existing files
  - Verify package.json exists if frontend generated
- If issues found, log them and include in Phase 2 context ok 

### 4. LLM-Based Gap Analysis (replace keyword matching)
- Instead of hardcoded `_AUTH_KEYWORDS`, `_DB_KEYWORDS`, etc.
- Send one cheap Haiku call: "Given this model and these user instructions, what specific customization tasks are needed?"
- Haiku is fast (~1s) and cheap (~$0.001)
- Returns structured task list that's much smarter than regex yes perfect

---

## Medium Features 

### 5. Refactor Orchestrator
- Split the 730-line orchestrator.py into:
  - `phase_manager.py` — runs Phase 1/2/3 in sequence
  - `gap_analyzer.py` — analyzes what's missing (keyword or LLM-based)
  - `prompt_builder.py` — builds system prompt from inventory + tasks
  - `compaction.py` — context compaction logic
  - `orchestrator.py` — thin coordinator that wires everything together
- Each module testable independently yes clean code

### 6. Parallel Tool Execution
- When LLM calls multiple tools in one turn (e.g., 5 write_file calls)
- Execute them concurrently with `asyncio.gather()` or `ThreadPoolExecutor`
- Return all results at once
- Significant speedup for frontend generation (many small files) yes can do this 

### 7. Snapshot/Rollback
- After Phase 1: snapshot the workspace (copy to .besser_snapshot/)
- If Phase 3 validation makes things worse: restore from snapshot
- Uses: `shutil.copytree()` for snapshot, `shutil.rmtree() + rename` for restore yes if not to hard and light for a server in futur 

### 8. Interactive Error Feedback
- After generation, user runs the app and hits an error yes perfect 
- User pastes the error back (via API or CLI)
- Orchestrator resumes Phase 2 with the error as context
- LLM reads the error, fixes the code, verifies
- Requires: session persistence (save messages to disk) + resume capability

---

## Bigger Features (1-2 days each)

### 9. MCP Server (expose to Claude Code)
- Update https://github.com/BESSER-PEARL/BESSER-MCP-Server
- Expose generators as MCP tools: `mcp__besser__generate_fastapi`, `mcp__besser__generate_react`, etc.
- Expose model loading: `mcp__besser__load_buml_model` (from .py file)
- Expose validation: `mcp__besser__validate_model`
- Users add to `.claude/settings.json` and get BESSER tools in Claude Code
- Use claw-code's naming convention: `mcp__besser__<tool_name>` but if we are inside the source code we dont need the mcp right ?

### 10. Web Editor Endpoint
- New FastAPI route: `POST /besser_api/llm-generate/from-project`
- Receives ProjectInput + instructions + API key
- Runs LLMGenerator, returns ZIP
- Stream progress via SSE for real-time UI feedback
- Frontend: "AI Generate" button in the editor this will be in the futur not now we will even connect it top the modeling agent

### 11. Multi-Provider Support
- Provider adapter pattern (like claw-code's ProviderClient)
- `ClaudeProvider` (current), `OpenAIProvider` (new), `OllamaProvider` (future)
- Common interface: `chat(system, messages, tools) -> response`
- Tool calling format translation (Anthropic ↔ OpenAI)
- Support GPT-5.4 via the gateway this we sjould do i think at least first openai 

### 12. Template/Recipe Library
- Save successful recipes as reusable templates
- "FastAPI + JWT auth" recipe: deterministic steps that always work 
- Share recipes between users (export/import JSON)
- Apply recipe to new model: same steps, different entities
- Reduces LLM calls (some steps are deterministic) why not i want to hear more about that

---

## Vision Features (research needed)

### 13. Smart Gap Analysis with LLM
- Use Haiku for task decomposition (cheap, fast)
- "Given model X and instructions Y, list specific tasks"
- Returns structured JSON: [{task, files_to_create, files_to_modify, complexity}]
- Orchestrator feeds tasks one at a time to Sonnet/Opus yes why mpte

### 14. Quality Scoring
- After generation, auto-score the output:
  - Syntax check: all files parse? (0-10)
  - Import check: all imports resolve? (0-10)
  - Completeness: all requested features present? (0-10)
  - Docker: builds successfully? (0-10)
- Report: "Quality: 8/10 — 2 issues found"
- If score < 7, trigger Phase 3 fixes automatically ok i see idk

### 15. Learning from Recipes
- Analyze past successful recipes for patterns
- "For FastAPI + auth, the LLM always: writes auth.py, modifies main_api.py imports, adds Depends"
- Pre-compute these steps deterministically
- Only use LLM for novel/custom parts
- Reduces cost and increases reliability over time why note intersting

### 16. Enhance Mode
- User already has code (from generator or hand-written)
- "Enhance with AI" analyzes existing codebase
- LLM reads existing code, adds requested features
- No generator needed — works on any codebase
- Like Claude Code but with model awareness not now

---

## Priority Matrix

| Priority | Features | Effort | Impact |
|----------|----------|--------|--------|
| P0 (now) | #1 Cost cap, #2 Timeout | 2 hours | Safety |
| P1 (this week) | #4 LLM gap analysis, #3 Phase 1 validation | 4 hours | Quality |
| P2 (next week) | #9 MCP server, #10 Web editor endpoint | 2 days | Reach |
| P3 (month) | #5 Refactor, #6 Parallel exec, #11 Multi-provider | 1 week | Scale |
| P4 (quarter) | #12 Recipes, #13 Smart gap, #14 Quality scoring, #16 Enhance | 2 weeks | Vision |

---

## Architecture After All Improvements

```
User instruction + BUML Model
    │
    ▼
┌──────────────────────────────────────────────────┐
│ LLM Generator                                     │
│                                                    │
│  Phase 0: Smart Gap Analysis (Haiku — 1 cheap call)│
│    → Structured task list                          │
│                                                    │
│  Phase 1: Deterministic Generation                 │
│    → Select best generator (web_app, fastapi, etc.)│
│    → Run generator                                 │
│    → Validate output (py_compile, file checks)     │
│    → Inventory for context                         │
│                                                    │
│  Phase 2: LLM Customization (Sonnet — scoped tasks)│
│    → Execute tasks from gap analysis               │
│    → Parallel tool execution                       │
│    → Context compaction if needed                  │
│    → Cost cap enforcement                          │
│                                                    │
│  Phase 3: Validation & Fix                         │
│    → Syntax check, import check, Docker check      │
│    → LLM fixes issues (5 turns max)                │
│    → Quality scoring                               │
│    → Rollback if fixes make things worse            │
│                                                    │
│  Output: ZIP with recipe + quality report          │
└──────────────────────────────────────────────────┘
    │
    ▼
Available via:
  ├── Python library: LLMGenerator(model, instructions)
  ├── Web editor: POST /llm-generate/from-project
  ├── MCP server: mcp__besser__generate_*
  └── CLI: besser generate --llm "instructions"
```
