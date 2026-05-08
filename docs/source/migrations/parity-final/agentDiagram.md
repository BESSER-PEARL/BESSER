# AgentDiagram — Final Parity Check

Verdict: **PASS WITH GAPS** (2 visual regressions on the edge layer; no
data-model loss; functional parity on inspectors and migrators).

The new lib reaches data and behavioural parity with the old fork's
~960-LoC `agent-state-update.tsx` family and successfully promotes
`AgentIntentDescription` / `AgentIntentObjectComponent` to first-class
nodes (per SA-4) while keeping the v3 wire shape losslessly recoverable.
The two gaps below are pure SVG-marker / dash-array oversights in
`utils/edgeUtils.ts::getEdgeMarkerStyles`, easily fixed.

## 1. Element types — ✅

v3 (`packages/editor/.../agent-state-diagram/index.ts:AgentElementType`)
registers exactly 7 agent types: `AgentState`, `AgentStateBody`,
`AgentStateFallbackBody`, `AgentIntent`, `AgentIntentBody`,
`AgentRagElement`. Note v3 does NOT register
`AgentIntentDescription` or `AgentIntentObjectComponent` — these are
rendered inline inside `AgentIntent` (the description as a text row in
`agent-intent-description-component.tsx`; the object component as
nested children of the intent box).

The new lib's `packages/library/lib/nodes/agentDiagram/` ships 8
first-class node files: `AgentState`, `AgentStateBody`,
`AgentStateFallbackBody`, `AgentIntent`, `AgentIntentBody`,
`AgentIntentDescription`, `AgentIntentObjectComponent`, `AgentRagElement`.
The two extras (`AgentIntentDescription`,
`AgentIntentObjectComponent`) are EXTRA-in-v4 child nodes per SA-4 —
verified.

## 2. Edge types — ✅

`AgentStateTransition` and `AgentStateTransitionInit` both present in
`packages/library/lib/edges/edgeTypes/` and registered in
`edges/types.tsx:201-209`. Migrator inverts v3 names verbatim
(`versionConverter.ts:436-437`).

## 3. Per-element data fields — ✅

- `AgentState`: `name`, `replyType`, `stereotype`, `italic`, `underline`,
  fill/stroke/textColor — `versionConverter.ts:988-1005`,
  `AgentStateNodeProps`. Match.
- `AgentStateBody` / `AgentStateFallbackBody`: `replyType`,
  `ragDatabaseName`, `dbSelectionType`, `dbCustomName`, `dbQueryMode`,
  `dbOperation`, `dbSqlQuery`, `code`, `kind` — all preserved
  (`versionConverter.ts:1007-1039`).
- `AgentIntent`: `name`, `intent_description`, stereotype, italic,
  underline (`:1041-1058`).
- `AgentIntentBody`: training phrase on `name` (one-utterance-per-line
  textarea — see inspector below).
- `AgentIntentDescription`: text on `name` (rolled up to parent's
  `intent_description` on v3 export).
- `AgentIntentObjectComponent`: `entity`, `slot`, `value`, `name`
  (`:1070-1082`, `AgentIntentObjectComponentEditPanel.tsx:48-80`).
- `AgentRagElement`: `name`, `ragDatabaseName`, `dbSelectionType`,
  `dbCustomName`, `dbQueryMode`, `dbOperation`, `dbSqlQuery`, `ragType`
  — open question #5 resolution preserves BOTH `ragDatabaseName` and
  `dbCustomName` verbatim (`:1084-1107`,
  `AgentRagElementEditPanel.tsx:1-165`).
- `AgentStateTransition` (edge): `transitionType`, `predefined.{predefinedType,
  intentName, fileType, conditionValue}`, `custom.{event, condition[]}`,
  `params{}`, plus `legacy{}` bag and `legacyShape` discriminator for
  round-trip preservation
  (`AgentDiagramEdgeEditPanel.tsx:44-67`).

## 4. AgentStateTransition 5 legacy shapes — ✅

`liftAgentTransitionDataToV4` in `versionConverter.ts:1409-1633`
explicitly handles all five shapes via a priority cascade:

1. Canonical predefined (`{transitionType: 'predefined', predefined:
   {...}}`) — direct passthrough.
2. Canonical custom (`{transitionType: 'custom', custom: {...}}`) —
   direct passthrough.
3. Legacy flat predefined (`{predefinedType, variable/operator/targetValue}` or
   `{predefinedType, fileType}`) — collected into `predefined.conditionValue`.
4. Legacy `condition: 'custom_transition'` + `customEvent` /
   `customConditions` — folded into `custom.{event, condition}`.
5. Legacy nested `conditionValue.{events, conditions}` — folded into
   `custom.{event, condition[]}`.

Round-trip parametric tests at
`packages/library/tests/round-trip/agentDiagram.test.ts:259-338`
parameterize over five fixtures (`agentTransitionShape1.json` …
`agentTransitionShape5.json`) and assert (a) first-pass canonicalisation
and (b) v4 → v3 → v4 idempotence. ✅

## 5. Inspector form parity — ✅ (functional parity, MUI re-skin)

- **`AgentStateEditPanel`** (`AgentStateEditPanel.tsx`, 607 LoC):
  faithfully ports the v3 `~960-LoC agent-state-update.tsx`. 5-radio
  reply-mode picker for **both** body and fallback body (`text` / `llm`
  / `rag` / `db_reply` / `code`); per-mode editors including
  CodeMirror-Python for `code`, RAG dropdown sourced from sibling
  `AgentRagElement` nodes (`useMemo` over `nodes.filter(type ===
  'AgentRagElement')`), full DB-action editor via shared `RagDbFields`
  for `db_reply` mode; sibling-of-wrong-type deletion + auto-create
  semantics on mode-switch (matching v3's radio onChange logic at
  `agent-state-update.tsx:303-411`); `name`, `stereotype` (none /
  initial / final / intent), `italic`, `underline` fields. ✅
- **`AgentStateBodyEditPanel`** (188 LoC): standalone body editor
  exposing `name`, `kind` (entry / do / exit / on-transition),
  `replyType`, `code` (CodeMirror Python), and the shared
  `RagDbFields` block. Use case: editing a body popped directly from
  the canvas. ✅
- **`AgentIntentEditPanel`**: name only — description and training
  phrases now live on dedicated child nodes. ✅
- **`AgentIntentBodyEditPanel`**: multiline textarea for training
  phrases ("one per line" placeholder mirrors v3 behaviour). ✅
- **`AgentIntentDescriptionEditPanel`**: 3-row multiline textarea.
  Migrator rolls value back into parent `intent_description` on v3
  export. ✅
- **`AgentIntentObjectComponentEditPanel`**: `name`, `entity`, `slot`,
  `value`. ✅
- **`AgentRagElementEditPanel`**: `name`, `dbSelectionType`,
  `ragDatabaseName`, `dbCustomName`, `dbQueryMode`, `dbOperation`,
  `dbSqlQuery` (when sql), `ragType`. **Strict superset of v3** (v3
  showed only "Name of RAG DB" — see
  `agent-rag-element-update.tsx:21-27`); the SA-2.2 #22 expansion
  surfaces the DB-action fields previously buried inside the parent's
  body editor.
- **`AgentDiagramEdgeEditPanel`** (539 LoC,
  `AgentDiagramEdgeEditPanel.tsx`): mode toggle (predefined / custom),
  predefined dropdown (5 options), conditional sub-pickers
  `intentName` (Select sourced from sibling `AgentIntent` nodes when
  `predefinedType === 'when_intent_matched'`, with TextField fallback
  if no intents exist — SA-2.2 #23), `fileType` PDF/TXT/JSON Select for
  `when_file_received` (SA-2.2 #24), variable/operator/targetValue
  triple for `when_variable_operation_matched`. Custom branch: 7-event
  Select, condition list with **CodeMirror Python** (SA-2.2 #25,
  matching v3's `react-codemirror2 mode='python'` at
  `agent-state-transition-update.tsx:329-348`), per-condition delete.
  Numeric-keyed `params{}` editor. Edge style + flip via
  `EdgeStyleEditor` + `SwapHorizIcon` (SA-2.2 #26). ✅
- **`AgentDiagramInitEdgeEditPanel`** (17 LoC): "Initial-state
  marker. No editable fields." — matches v3 init edge having no
  inspector. ✅

## 6. Constraints — ✅

- **Single initial state**: enforced indirectly by the canvas — the
  init edge is created from a synthetic source so only one can be live.
  No explicit cross-edge validator was found in either v3 or v4; the
  constraint is structural (one init edge per diagram). Acceptable
  parity (no regression).
- **v4 → v3 export safety for EXTRA-in-v4 child types**:
  `convertV4ToV3Agent` in `versionConverter.ts:2640-2870` correctly
  (a) rolls `AgentIntentDescription.name` up to the parent intent's
  `intent_description` field if the parent doesn't already have one
  (`:2711-2744`), and (b) **drops** `AgentIntentDescription` and
  `AgentIntentObjectComponent` from the v3 elements map
  (`:2750-2764`). Tested at
  `agentDiagram.test.ts:340-454` with two regression cases. ✅

## 7. Visual shape — ⚠️ Two gaps

- **`AgentState` parent rectangle with body region**: ✅ — matches v3
  visual via rounded rect + header divider line
  (`AgentState.tsx:59-115`).
- **`AgentRagElement` shape**: ✅ but **note prompt expectation was
  wrong**. Prompt says "document-shape glyph"; v3 actually renders a
  cylinder (top/bottom ellipses + side rect — see
  `agent-rag-element-component.tsx:24-47` in the old fork). The new
  lib's `AgentRagElement.tsx:67-117` reproduces the cylinder
  faithfully. Visual parity preserved.
- **`AgentStateTransitionInit` line style**: ⚠️ — prompt's expected
  "dashed line with no source endpoint" does not match v3 either:
  v3's `agent-state-transition-init-component.tsx:62-87` renders a
  **solid** line with an arrow marker. The new lib's
  `AgentDiagramInitEdge.tsx:108-119` correctly omits a source-endpoint
  fixture (handled via `useStepPathEdge` reconnection logic) BUT…
- **Init-edge arrowhead missing**: see Critical Gap #1 below.
- **Regular `AgentStateTransition` arrowhead missing**: see Critical
  Gap #2 below.

## 8. Round-trip — ✅

`tests/round-trip/agentDiagram.test.ts` — 455 LoC, three describe
blocks:

1. v3-fixture migration assertions (one `it` block, ~150 LoC):
   verifies all 14 nodes, RAG dual-name preservation, init-edge
   passthrough, and shapes 1-4 inline.
2. Parametric `AgentStateTransition shape parameterized round-trip`
   over the 5 legacy fixtures.
3. `AgentIntentDescription / ObjectComponent v4 → v3 export safety`
   regression suite (2 it blocks).

Total: 8 it blocks. v3 → v4 → v3 → v4 idempotence established via the
canonical-form normalization helper at `:169-222`. ✅

## Critical gaps

**#1 — `AgentStateTransitionInit` has no arrowhead**:
`utils/edgeUtils.ts::getEdgeMarkerStyles` (lines 173-359) has no case
for `AgentStateTransitionInit`. It falls through to the default
branch (`{markerPadding, strokeDashArray: "0", offset: 0}` — no
`markerEnd`). v3 rendered the init edge with a plain arrow head
(`agent-state-transition-init-component.tsx:62-81`,
`<ThemedPath d="M0,29 L30,15 L0,1" />`). **Visual regression** — the
init edge now renders as a solid line with no arrow.

**#2 — `AgentStateTransition` has no arrowhead**: Same root cause.
`getEdgeMarkerStyles` is missing the `AgentStateTransition` case, so
regular transitions also fall through to default. v3 rendered them
with the same open-arrow marker
(`agent-state-transition-component.tsx:118-136`). **Visual
regression** — regular transitions also render arrow-less.

Recommended fix: add to `getEdgeMarkerStyles`:

```ts
case "AgentStateTransition":
case "AgentStateTransitionInit":
  return {
    markerPadding: EDGES.MARKER_PADDING,
    markerEnd: "url(#black-arrow)",   // or the open-arrow variant
    strokeDashArray: "0",
    offset: 0,
  }
```

Both edge components already plumb `markerEnd` through `EdgeInlineMarkers`
(`AgentDiagramEdge.tsx:194-199`,
`AgentDiagramInitEdge.tsx:122-129`); only the style table needs to feed
them.

## Other observations (non-blocking)

- `AgentStateEditPanel` uses **Checkbox** rows instead of v3's
  **radio** group for the reply-mode picker (`:561-573`). Functional
  semantics match (only one mode can be "active" because mode-switch
  deletes siblings of other types) but UX deviates. Minor.
- `dbSelectionType` enum: v3 used `{'default', 'custom'}` only;
  `AgentRagElementEditPanel.tsx:29` adds `'predefined'`. The migrator
  preserves whatever v3 supplied verbatim, so legacy data round-trips
  cleanly; `'predefined'` is a new option for new-diagram authoring.
  Likely intentional.
- Init-edge inspector renders only a "no editable fields" caption —
  v3's UMLRelationship base form gave it a name field + flip + delete;
  the new lib delegates flip / delete to the canvas toolbar instead.
  Functional parity via different surface.

## Files of interest

- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/agentDiagram/` (8 nodes)
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/edges/edgeTypes/AgentDiagramEdge.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/edges/edgeTypes/AgentDiagramInitEdge.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/agentDiagram/` (10 inspector files)
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/versionConverter.ts:309-319, 434-437, 988-1107, 1409-1633, 2024-2049, 2605-2870` (agent-specific migrator logic)
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/edgeUtils.ts:173-359` (**fix site for the two gaps**)
- `besser/utilities/web_modeling_editor/frontend/packages/library/tests/round-trip/agentDiagram.test.ts` (455 LoC)
