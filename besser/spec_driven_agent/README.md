# `besser.spec_driven_agent` — the Spec-Driven Agent

The hybrid generator produces a deterministic structural baseline from BESSER
models, then uses an LLM to complete the user's natural-language specification.
The verbatim request remains authoritative: a model, summary, or scaffold may
not capture every requested behavior.

User and API docs live under `docs/source/spec_driven_agent/`:
`how_it_works.rst`, `api.rst`, and `validation.rst`. This README is the developer
entry point.

## Pipeline

```text
select → generate → plan → customize → validate/repair
```

1. Select a deterministic generator using the available models and approved
   target. Selection may use the configured provider.
2. Generate the structural baseline and inventory files and action endpoints.
3. Plan remaining work with the provider's planning model when available,
   alongside deterministic obligations and the original requirement ledger.
4. Customize through bounded file/model tools. Track implementation evidence
   separately from verified acceptance.
5. Collect findings and, when `auto_fix_issues` is enabled, repair and recheck
   within the remaining budgets. Unresolved defects or missing required
   verification remain incomplete.

Static/source checks and execution checks are distinct. The orchestrator defaults
to `enable_import_smoke_check=True` for generated Python startup/constructibility
checks; this also gates the disposable API-scenario tool. Compiler checks (`tsc`,
`cargo`, `kotlinc`) require `enable_toolchain_validation=True`, which defaults to
false. Missing tools or exhausted budgets can limit validation; skipped checks
are not passes. Shell tools have a separate opt-in. Executing generated code is
not an OS sandbox.

Frontend production builds require both toolchain validation and shell permission.
Only a harness-observed successful build against unchanged sources discharges that
check; generic shell-success logs are not build evidence. The mutation inventory
exposes reverse relationship inputs, native association roles, and public write
paths so a test's descriptive name is not mistaken for exercising those paths.

## Shell tools: a deployment decision

`run_command` / `install_dependencies` are off by default on every path
(`LLMOrchestrator`, `LLMGenerator`, the web runner). The web runner takes its
value from `BESSER_LLM_ENABLE_SHELL_TOOLS`, read once at import into
`backend/constants/constants.py`; no request field, header or query parameter
reaches `allow_shell_tools`, and the Pydantic request model drops an unknown
key in the body. That is the property that makes the hosted default a gate rather
than a suggestion, so keep the decision at process scope. The gate is enforced
in `ToolExecutor.execute_typed`, not only by filtering the advertised tool list.

A local or on-prem install sets the variable and gets the generate-test-fix
loop. `GET /besser_api/spec-driven/config` reports the live value as
`features.shell_tools_enabled` so a deploy can be checked from outside the
process. The local path already has a 120s per-command timeout, a
workspace-confined working directory (`_safe_cwd`), the stripped subprocess
environment from `execution/process.py`, an output cap and a denylist for the
obvious catastrophes. None of that is an OS sandbox: the command runs as the
backend user and may `cd` out of the workspace. `docs/source/spec_driven_agent/tools.rst`
states the guarantees and the non-guarantees.

When a stream overruns the cap, the untruncated output is spilled to
`COMMAND_OUTPUT_DIR` (`.besser_command_output/`) in the workspace and the tool
result carries `full_output_path`. Head+tail truncation discards the middle,
which for a failing `tsc` / `npm run build` is the diagnostics themselves.
`search_in_files` reaches that directory by exception; packaging, the push, the
scaffold inventory and the recipe manifest all exclude it. Each spilled stream
is capped at `MAX_SPILL_SIZE` so a runaway command cannot fill the disk.

## Responsibility map

| Area | Modules |
| --- | --- |
| Public entry and coordination | `__init__.py`, `llm_generator.py`, `orchestrator.py` |
| Planning, prompts, discovery | `gap_analyzer.py`, `action_inventory.py`, `mutation_inventory.py`, `prompt_builder.py`, `model_serializer.py`, `stack_metadata.py` |
| Authoritative request | `specification.py`, `user_request.py` |
| Tool contracts and execution | `tools.py`, `tool_executor.py`, `edit_apply.py` |
| Shared subprocess environment | `execution/process.py` |
| Shared findings and source contracts | `validation/issues.py`, `validation/python_source.py`, `validation/frontend_schema.py`, `validation/frontend_build.py`, `validation/frontend_contract.py`, `validation/python_imports.py`, `validation/toolchain.py` |
| Immediate source/model contracts | `write_diagnostics.py`, `contract_checks.py`, `frontend_bindings.py`, `endpoint_coherence.py` |
| Deterministic repair (no model, no LLM) | `import_repair.py`, `scaffold_repair.py` |
| Runtime probes | `constructibility.py`, `api_probe.py` |
| Requirements and scoped acceptance | `requirements_ledger.py`, `acceptance.py`, `fix_target.py` |
| Persistence, tracing, context | `checkpoint.py`, `tracing.py`, `compaction.py`, `history_eviction.py`, `errors.py` |
| Providers, routing, usage | `llm_client.py` |

The shared `validation/` and `execution/` modules do not import the orchestrator
or executor. Existing imports from `orchestrator.py` and `tool_executor.py` remain
available as compatibility aliases. New shared checks belong below their callers,
not in a coordinator imported by its own helpers.

Runtime probes execute themselves by file path in a child interpreter where
BESSER need not be installed. Keep BESSER imports inside parent-only functions;
the worker paths must remain standalone.

The web runner/event handling live in
`besser/utilities/web_modeling_editor/backend/services/spec_driven/`, the HTTP
router in `.../routers/spec_driven_router.py`, and request models in
`.../models/spec_driven.py`.

## Providers and compatibility

Provider defaults and accounting are defined in `llm_client.py`; callers may
override the run model. Planning uses `llm_client.planning_model` when supported,
with the configured fallback behavior. The modeling-agent repository has separate
model routing for editor/diagram generation.

Preserve public entry points, tool names/result shapes, checkpoint/recipe/trace
formats, edit receipts, per-path locking, and the single checklist state when
extracting modules. Existing implementation evidence is not a successful new
write. File organization must not change provider, retry, or validation policy.

Edit recovery is shared across phases: two text-edit refusals request a fresh
read, then a revision-bound `replace_file_lines` call. This avoids repeatedly
quoting an inaccurate `old_text`; syntax/replay safeguards remain in place.
Read handles authorize only displayed lines and are not persisted in checkpoints.
Rejected drafts are omitted from subsequent provider requests while their errors
and current-source excerpts remain; full original edit inputs stay in the trace
sidecar. This prevents a refused proposal becoming the agent's presumed source.
The OpenAI-compatible adapter recognizes complete native tool calls returned with
a normal `stop` label (observed with Nebius Qwen forced calls), in both streaming
and non-streaming paths. Explicit truncation/filter stops are not promoted to
tool execution. Never infer a tool call from prose.

## Verification scope

Use the relevant existing tests under `tests/spec_driven_agent/`, then the full
offline suite for refactors. No paid generation is needed for import organization.
Source checks, startup checks, and submitted API scenarios cover different failure
modes; passing them is not complete specification or benchmark acceptance.
