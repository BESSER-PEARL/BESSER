# A1 — Round-trip Corpus Sweep (final-analysis wave)

Owner: A1 (final-analysis wave)
Branch: `claude/refine-local-plan-sS9Zv`
Script: `tests/corpus/round_trip_sweep.mjs`

## Goal

Drive every v3-shape and v4-shape diagram fixture in the repo through the
`@besser/wme` v3↔v4 converters and assert lossless / idempotent
round-tripping. The library's per-diagram round-trip tests already
exercise one curated fixture per type; this sweep widens the net to the
full corpus (library tests + editor test resources + webapp templates +
backend Python-side fixtures) so we can flag any fixture the converters
silently mangle.

## Method

For each fixture we classify it as v3 or v4 (peeking through the
`{ title, model: {...} }` wrapper used by `tests/fixtures/v4/` and the
nested `project.diagrams.{type}[].model` shape used by webapp project
templates), then:

* **v3 input:** `migrate*V3ToV4(model)` → `convertV4ToV3*(v4)` →
  `migrate*V3ToV4(v3')` and assert `canon(v4) === canon(v4')` (the
  idempotence form already used by `packages/library/tests/round-trip/*.test.ts`).
* **v4 input:** `convertV4ToV3*(model)` → `migrate*V3ToV4(v3)` and
  assert `canon(v4) === canon(v4')` (lossless reverse round-trip).

The `canon()` helper sorts object keys deterministically and drops the
top-level `converted-diagram-<timestamp>` id that `convertV3ToV4`
re-stamps on every call (see `versionConverter.ts:1813`) — without that
filter every fixture would fail trivially on the timestamp.

`ObjectDiagram` has no dedicated reverse converter; mirroring
`packages/library/tests/round-trip/objectDiagram.test.ts`, we route
ObjectDiagram through `convertV4ToV3Class`.

## Corpus

| Source                                                                 | Count |
|------------------------------------------------------------------------|-------|
| `packages/library/tests/fixtures/v3/`                                  | 11    |
| `packages/editor/src/tests/unit/test-resources/` (v3 only)             | 5     |
| `packages/editor/src/main/packages/user-modeling/` (v3)                | 3     |
| `packages/webapp/src/main/templates/` (v4 + project-wrapped v4)        | 21    |
| `tests/fixtures/v3/`                                                   | 0     |
| `tests/fixtures/v4/` (incl. project-wrapped diagrams)                  | 7     |
| **Total**                                                              | **47**|

Notes on coverage:

* `packages/webapp/src/main/templates/pattern/gui/Complete.json` is a
  legacy GrapesJS GUI model (`"version": "0.21.13"`) and is **not** a UML
  diagram — the classifier correctly skips it.
* `personalized_gym_agent.json` and `library_full_stack.json` are
  ProjectInput wrappers; the script descends into
  `project.diagrams.{type}[].model` and surfaces each as its own entry.
  `personalized_gym_agent.json` also embeds two stale v3 AgentDiagram
  blobs at `…AgentDiagram[0].model.config.personalizationVariants[*].model`;
  these are not surfaced (they're config snapshots, not diagrams) but
  flagged here as a v3 leftover for the webapp-template cleanup pass.

## Results

```
Total fixtures: 47   PASS: 22   FAIL: 25
```

| Diagram type / direction      | PASS | FAIL |
|-------------------------------|------|------|
| ClassDiagram (v3)             | 8    | 0    |
| ClassDiagram (v4)             | 8    | 3    |
| ObjectDiagram (v3)            | 1    | 0    |
| ObjectDiagram (v4)            | 0    | 1    |
| StateMachineDiagram (v3)      | 0    | 1    |
| StateMachineDiagram (v4)      | 0    | 2    |
| AgentDiagram (v3)             | 2    | 4    |
| AgentDiagram (v4)             | 0    | 8    |
| UserDiagram (v3)              | 1    | 0    |
| UserDiagram (v4)              | 2    | 0    |
| NNDiagram (v3)                | 0    | 1    |
| NNDiagram (v4)                | 0    | 4    |
| CommunicationDiagram (v3)     | 0    | 1    |

## Failure clusters

Five distinct root causes account for all 25 failures.

### Cluster 1 — `tests/fixtures/v4/*.json` carry vestigial v3 metadata (7 failures)

```
v4->v3->v4 not lossless: $: keys differ (only-left=[id,interactive,size] only-right=[])
```

Affected: `agent_diagram_basic.json`, `class_diagram_basic.json`,
`deploy_project.json::ClassDiagram[0]`, `nn_diagram_basic.json`,
`object_diagram_basic.json`, `state_machine_basic.json`,
`webapp_smoke_project.json::ClassDiagram[0]` (the ObjectDiagram entry
also has `referenceDiagramData`).

**Theory:** these backend-side v4 fixtures still ship the v3-shape
top-level `size` and `interactive` fields and a stable `id` (e.g.
`cd-1`). The v4 → v3 converter doesn't pick up `size` / `interactive`
from the v4 root, and `convertV3ToV4` re-stamps `id` to a fresh
`converted-diagram-<Date.now()>` which our filter drops. The data is
not preserved in the round-trip — these fixtures should be regenerated
without the v3 metadata, or the converter should propagate `size` /
`interactive`.

### Cluster 2 — Agent edge `legacyShape` not preserved on v4 (4 failures)

```
v4->v3->v4 not lossless: $.edges[N].data: keys differ (only-left=[] only-right=[legacyShape])
```

Affected: `pattern/agent/{faqragagent,greetingagent,gymagent,libraryagent}.json`.

**Theory:** `convertV3ToV4` stamps a `legacyShape: 1|2|3|4|5` on agent
edge `data` (`versionConverter.ts:1491–1502`), but the source v4
fixtures were authored without it. After v4→v3→v4 the field appears.
Either the fixtures need `legacyShape` baked in (re-export from a
running editor) or `convertV3ToV4` should only stamp `legacyShape` when
the v3 edge actually carries one of the 5 legacy shapes.

### Cluster 3 — Agent / SM / NN node positions shift on round-trip (10 failures)

```
$.nodes[N].position.x: <a> vs <b>
```

Affected: 1 v3 NNDiagram, 1 v3 StateMachineDiagram, 4 v4 AgentDiagram
(dbagent, library\_full\_stack agent, personalized\_gym\_agent agent,
plus `agentDiagram.json` v3 fixture by the same edge-data class), 3 v4
NNDiagram (alexnet, lstm, tutorial), 1 v4 StateMachineDiagram
(traficlight). Deltas range from 30–740 px and are not random — they
match container offsets.

**Theory:** v3 stores absolute `bounds.x/y`; v4 stores `position` plus
`parentId` (relative to the parent container). The converter applies
parent-relative re-basing on v3→v4 but not the inverse on v4→v3 (or
vice versa). Nodes whose `parentId` is set (e.g. `AgentStateBody`
inside an `AgentState`, `NN` layer inside a container) lose the
absolute offset on the way out and pick it up again on the way in,
producing a stable but non-zero delta. SA-3 and SA-4 round-trip tests
sidestep this because the curated fixture has `parentId === undefined`.

### Cluster 4 — AgentDiagram edge `data.legacy` schema drift (4 failures)

```
$.edges[N].data.legacy: keys differ
  only-left=[operator,predefinedType,targetValue,variable] only-right=[custom,predefined,transitionType]
  only-left=[condition,customConditions,customEvent]       only-right=[custom,predefined,transitionType]
  only-left=[conditionValue]                                only-right=[custom,predefined]
```

Affected: `agentDiagram.json`, `agentTransitionShape3.json`,
`agentTransitionShape4.json`, `agentTransitionShape5.json` (v3 → v4
sweep). Shapes 1 and 2 round-trip cleanly.

**Theory:** `liftAgentTransitionDataToV4` (the SA-4 helper that
collapses the 5 legacy `AgentStateTransition*` shapes) flattens the
shape-specific keys (`operator`, `predefinedType`, `condition`, …)
into the `legacy` bag on the first v3→v4 hop, but
`convertV4ToV3Agent` re-emits the canonical
`{custom, predefined, transitionType}` set instead of the original
keys. The second migrate then keeps the canonical-only set, which
diverges from the first migrate's "legacy preserved verbatim" promise
in the `migrateAgentDiagramV3ToV4` docstring. Either the lift should
canonicalise on the first hop (so both v4s match) or the reverse should
re-hydrate the original keys.

### Cluster 5 — No converter for CommunicationDiagram (1 failure)

```
no migrator/reverse for type CommunicationDiagram
```

Affected: `packages/editor/src/tests/unit/test-resources/communication-diagram.json`.

**Theory:** The library doesn't expose `migrateCommunicationDiagramV3ToV4`
/ `convertV4ToV3Communication`. CommunicationDiagram is in the
`UMLDiagramType` enum (per `packages/editor/lib/typings.ts`) but is not
in the `MIGRATORS` table at `versionConverter.ts:1967–2069`. Either
add the converter (CommunicationDiagram is in scope per
`docs/source/migrations/uml-v4-shape.md`) or de-list this fixture and
its diagram type from the supported set.

## Top-3 critical failures

1. **`tests/fixtures/v4/object_diagram_basic.json`** —
   `v4->v3->v4 not lossless: $: keys differ (only-left=[id,interactive,referenceDiagramData,size]…)`.
   The v4 ObjectDiagram backend fixture loses `referenceDiagramData`
   (the linked-class-diagram pointer) on the round-trip — that's the
   one piece that distinguishes ObjectDiagram from a plain class
   diagram, and it's silently dropped.
2. **`packages/library/tests/fixtures/v3/agentDiagram.json`** —
   `$.edges[3].data.legacy keys differ`. The flagship v3 agent fixture
   does not survive the documented `legacy` round-trip (cluster 4); the
   SA-4 brief explicitly lists "the v3 → v4 → v3 cycle is
   information-equivalent" as a non-goal-loss invariant.
3. **`packages/webapp/src/main/templates/pattern/agent/dbagent.json`**
   (and 3 sibling agent templates) —
   `$.nodes[3].position.x: 0.5 vs -29.5`. Live webapp templates loaded
   in the editor would shift on save → reload because nested
   `AgentStateBody` / `AgentIntentBody` children re-base coordinates on
   the v4→v3 emit (cluster 3).

## Recommended follow-ups (not in scope for A1)

* P1: fix cluster-3 parent-relative coordinate handling in
  `convertV4ToV3Agent` / `convertV4ToV3StateMachine` — covers 10/25
  failures and is user-visible.
* P1: regenerate `tests/fixtures/v4/*.json` via the v4 path (drop
  `id`/`interactive`/`size`/`referenceDiagramData` if not preserved, or
  teach the converter to preserve them) — covers 7/25 failures.
* P2: tighten cluster-4 agent `legacy` round-trip; either canonicalise
  on the first migrate or re-hydrate original keys on reverse.
* P2: add `legacyShape` to the v4 agent fixtures (or gate it on v3
  presence) — covers cluster 2.
* P3: register a CommunicationDiagram migrator + reverse converter, or
  remove the orphan v3 fixture.
* Cleanup: strip the two embedded v3 `personalizationVariants[*].model`
  blobs in `personalized_gym_agent.json` once the
  agent-personalization config is migrated to v4.
