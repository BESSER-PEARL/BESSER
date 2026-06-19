# `besser.generators.llm` — Vibe-Driven (LLM-Augmented) Generator

The smart / vibe-driven generator. It produces a deterministic, model-faithful
scaffold and then drives an LLM to customise it from a natural-language
request. The model stays the source of truth: the LLM edits a correct baseline
rather than writing from a blank page.

> **User & API docs** are published:
> - Generator overview: `docs/source/generators/vibe_driven.rst`
> - HTTP / SSE contract: `docs/source/web_editor.rst` (AI Assistant & Vibe-Driven Generation)
>
> This README is the **developer** entry point to the package.

## Pipeline

```
select → generate → gap → customize → validate
```

1. **select** — pick the deterministic Phase-1 generator (or honour an approved target).
2. **generate** — run it to produce the baseline scaffold.
3. **gap** — a cheap planning LLM call (`gpt-4o-mini`) computes the task list, or decides the scaffold suffices (Phase 2 skipped).
4. **customize** — the main LLM loop edits files through a constrained tool surface.
5. **validate** — optional toolchain/compile pass; **disabled in production** (cost/latency), enabled by the bench.

## Module map

| File | Responsibility |
|---|---|
| `orchestrator.py` | Drives the phases; owns the customize loop, checkpoints, recipe. |
| `llm_generator.py` | `GeneratorInterface` entry point. |
| `llm_client.py` | Provider clients (Anthropic / OpenAI), pricing, `DEFAULT_MODELS`, usage tracking. |
| `prompt_builder.py` | Builds the system prompt; serialises the models; frames the GUI model as a hint vs. spec. |
| `gap_analyzer.py` | The planning pass. Contract: `None` = failure, `[]` = scaffold sufficient, `[...]` = task list. |
| `tools.py` / `tool_executor.py` | The LLM's tool specs and their sandboxed execution. |
| `model_serializer.py` | Serialises BUML models (domain / GUI / agent / object / state machine / quantum) into the prompt. |
| `checkpoint.py` / `compaction.py` | Run checkpointing (for resume) and context compaction. |
| `errors.py` | Typed exceptions mapped to SSE error codes (`INVALID_KEY`, `UPSTREAM_LLM`, …). |

The web runner, SSE event schema, and request model live under
`besser/utilities/web_modeling_editor/backend/services/smart_generation/` and
`.../routers/smart_generation_router.py`.

## Models

- Provider defaults: `claude-sonnet-4-6` (anthropic), `gpt-4o` (openai); override per run via `llm_model`.
- Planning (gap phase) always uses `gpt-4o-mini`.
- BYOK — the user's key is held as a `SecretStr`, read once, never logged or persisted.

> Note: the **modeling-agent** repo has its own separate model routing table
> (`modeling-agent/src/model_config.py`) for diagram generation in the editor.
> That is distinct from this package's model selection.

## Design notes & roadmap

- `notes/ARCHITECTURE.md`
- `notes/llm_augmented_generation_architecture.md`
- `notes/llm_generator_roadmap.md`
- `notes/gui_export_simplification_plan.md`

## Known gaps

- **Round-trip preservation** — re-generation does not yet carry forward
  hand-written edits. This is the biggest planned improvement; see the
  `ROUND_TRIP_PRESERVATION.md` design doc in the workspace root.
- **No build gate in prod** — the validate phase is off by default.

## Benchmark

`vibe-bench` (sibling repo, `BESSER-PEARL/vibe-bench`) compares this generator
against a naive LLM on the same task spec, and `probe_evolution.py` measures
drift / file loss across modification chains.
