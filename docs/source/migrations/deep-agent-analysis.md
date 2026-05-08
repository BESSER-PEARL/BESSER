# AgentDiagram v3 vs v4 — Deep Analysis

Verdict: **PASS** — full functional parity, including round-trip
fidelity, with three minor cosmetic deviations. The two visual edge
gaps flagged in `parity-final/agentDiagram.md` (missing arrowheads on
`AgentStateTransition` / `AgentStateTransitionInit`) are now closed.
The `a31c4ec` height-auto-grow fix restores v3 behaviour.

Submodule HEAD checked: `a31c4ec` (frontend `chore(submodule) bump …
6 UI fixes from user testing`).

## Source files surveyed

- v3: `besser/utilities/web_modeling_editor/frontend/packages/editor/
  src/main/packages/agent-state-diagram/` — `index.ts`,
  `agent-state-preview.ts`, `agent-state/`, `agent-state-body/`,
  `agent-state-fallback-body/`, `agent-intent-object-component/`
  (intent + intent-body + intent-description + object-component
  combined), `agent-rag-element/`, `agent-state-transition/`,
  `agent-state-transition-init/`. Total update files surveyed:
  `agent-state-update.tsx` (968 LoC), `agent-state-body-update.tsx`
  (55 LoC), `agent-intent-update.tsx` (195 LoC),
  `agent-state-transition-update.tsx` (386 LoC). v3 has **no
  init-edge inspector** (no `agent-state-transition-init-update.tsx`).
- v4 nodes: `besser/utilities/web_modeling_editor/frontend/packages/
  library/lib/nodes/agentDiagram/` — `AgentState.tsx`,
  `AgentIntent.tsx`, `AgentIntentBody.tsx`,
  `AgentIntentDescription.tsx`, `AgentIntentObjectComponent.tsx`,
  `AgentRagElement.tsx`, `index.ts` (NO `AgentStateBody` /
  `AgentStateFallbackBody` files — bodies live inline on
  `AgentState.data.bodies`).
- v4 SVGs: `…/library/lib/components/svgs/nodes/agentDiagram/` —
  `AgentDiagramSVGs.tsx`, `index.ts`.
- v4 inspectors: `…/library/lib/components/inspectors/agentDiagram/`
  — 9 files. `AgentStateBodyEditPanel.tsx` was **deleted** by
  SA-FIX-Agent (`d8eda99`); the body editor is now embedded in
  `AgentStateEditPanel`.
- v4 edges: `…/library/lib/edges/edgeTypes/AgentDiagramEdge.tsx` and
  `AgentDiagramInitEdge.tsx`.
- Migrator: `…/library/lib/utils/versionConverter.ts:309-319` (type
  table), `:434-437` (relationship table), `:1056-1170` (v3→v4 body
  fold), `:1409-1633` (5-shape transition lifter), `:1976-2049`
  (v3→v4 element filter), `:2605-3010` (v4→v3 re-emit), `:3514-3628`
  (`normalizeAgentBodies`).
- Palette: `…/library/lib/constants.ts:919-977`
  (`UMLDiagramType.AgentDiagram` block).

## A — Palette entries (exhaustive)

`constants.ts:925-977` registers exactly **6 draggables** for
`AgentDiagram`:

| # | `type`                        | `defaultData` shape                                                 | SVG                            |
|---|-------------------------------|---------------------------------------------------------------------|--------------------------------|
| 1 | `AgentState`                  | `{ name: 'AgentState', replyType: 'text' }`                         | `AgentStateSVG`                |
| 2 | `AgentIntent`                 | `{ name: 'Intent', intent_description: '' }`                        | `AgentIntentSVG`               |
| 3 | `AgentIntentObjectComponent`  | `{ name: 'slot:entity' }`                                           | `AgentIntentObjectComponentSVG`|
| 4 | `AgentRagElement`             | `{ name:'RAG', ragDatabaseName:'', dbCustomName:'', dbSelectionType:'default', dbQueryMode:'llm_query' }` | `AgentRagElementSVG`           |
| 5 | `StateInitialNode`            | `{ name: '' }`                                                      | `StateInitialNodeSVG`          |
| 6 | `StateFinalNode`              | `{ name: '' }`                                                      | `StateFinalNodeSVG`            |

**Critical drop confirmed (G)**: NEITHER `AgentStateBody` /
`AgentStateFallbackBody` NOR `AgentIntentBody` /
`AgentIntentDescription` are exposed as draggables. Verified via
text grep on `constants.ts` — zero matches. The user-reported
"floating AgentStateBody" symptom can only be re-introduced via
malformed import data; for that case `normalizeAgentBodies`
(versionConverter:3514) folds floating bodies onto the nearest
`AgentState` and the `diagramStore` `addNode` / `setNodes` /
`setNodesAndEdges` guards (`diagramStore.ts:35,294,319,380`) refuse
the type — defense in depth.

**v3 comparison**: `agent-state-preview.ts:24-124` builds an inline
preview with 6 instances (empty-intent, RAG, empty-AgentState,
AgentState-with-body, AgentState-with-body-and-fallback,
StateInitialNode). v3 also did not expose `AgentStateBody` /
`AgentStateFallbackBody` / `AgentIntentBody` as standalone palette
entries. Net deltas:

- v4 surfaces `AgentIntentObjectComponent` as a standalone palette
  entry (additive UX); v3 only nested it inside `AgentIntent`.
- v4 surfaces `StateFinalNode`; v3 commented this out
  (`agent-state-preview.ts:115-119`).

Both deltas are additive.

## B — Canvas rendering

### `AgentState` — inline body rows (post-`d8eda99`)

`nodes/agentDiagram/AgentState.tsx:84-246` — bodies are read from
`data.bodies` (`AgentStateBodyRow[]`) and rendered as table-style
rows inside the parent `<svg>`. The component:

- Splits `bodies` by `kind === 'fallback'` into `mainBodies` and
  `fallbackBodies`.
- Draws a header divider when any body exists (`:209-218`).
- Draws an additional dashed fallback divider between main and
  fallback rows when both exist (`:222-233`, `strokeDasharray="3 2"`).
- Renders each row via `renderRow()` (`:33-82`): plain `<text>` for
  text/llm/rag/db_reply rows; `<foreignObject>` with monospace `<div>`
  for `replyType === 'code'`.
- **Auto-grow (`a31c4ec`)** at `:111-139`: a `useEffect` watches
  `requiredHeight = headerHeight + (mainBodies + fallbackBodies) ×
  ROW_HEIGHT(30) + (hasFallbackDivider ? 12 : 0) + 16`, and when the
  current `height` is smaller it patches the node via `setNodes` to
  bump `height`, `measured.height`, and `style.height`.

**v3 comparison**: `agent-state/agent-state.ts:74-115` — `render()`
walked `children`, computed per-row `y`, and assigned
`this.bounds.height = y` after the last row. **v3 had auto-grow.**
v4's `useEffect` restores the same effective behaviour, just shifted
from the model layer to a render-time side effect.

The pre-`a31c4ec` fixed-height bug (rows beyond the third spilled
outside the rectangle) is therefore a v4-only regression that has
been closed.

### `AgentIntent` — folded-corner rectangle with two dividers

`nodes/agentDiagram/AgentIntent.tsx:50-156` — folded-corner SVG path
(`d="M 0 0 H ${width} V ${height} H 30 L 0 ${height + 30} L 10
${height} H 10 0 Z"`), `"Intent: ${name}"` header, header divider at
`headerHeight`, second divider at `headerHeight +
AGENT_INTENT_DESCRIPTION_HEIGHT(30)` when both `data.intent_description`
and an inferred body row exist (PC-8 #3 fix). Light-green fill
`#E3F9E5` matching v3 default.

### `AgentIntentBody`, `AgentIntentDescription`, `AgentIntentObjectComponent`

Three child renderers (`AgentIntentBody.tsx`,
`AgentIntentDescription.tsx`, `AgentIntentObjectComponent.tsx`) each
draw a single text row inside the parent intent's bounds. They are
spawned by the `AgentIntent` inspector's `+ add` buttons (see C
below) and migrated in from v3's `intent_description` text + nested
`AgentIntentObjectComponent` elements by
`migrateAgentDiagramV3ToV4` in `versionConverter.ts:1056-1170`.

### `AgentRagElement` — cylinder

`nodes/agentDiagram/AgentRagElement.tsx:27-126` — top + bottom
ellipses + side rect (matches v3
`agent-rag-element-component.tsx:24-47` faithfully). Display label
resolves `dbCustomName ?? ragDatabaseName ?? name`. Default fill
`#E8F0FF`.

## C — `AgentStateEditPanel` (550 LoC)

Full v3 port at `inspectors/agentDiagram/AgentStateEditPanel.tsx`.
v3 source: `agent-state-update.tsx` (968 LoC).

| v3 field / control                                            | v4 location                          | Status |
|---------------------------------------------------------------|--------------------------------------|--------|
| `name`                                                        | `:440-447`                           | ✅ |
| `stereotype` Select (none / initial / final / intent)         | `:449-468` (4 options)               | ✅ |
| `italic` checkbox                                             | `:471-480`                           | ✅ |
| `underline` checkbox                                          | `:481-490`                           | ✅ |
| 5-mode reply picker (text / llm / rag / db_reply / code)      | `:495-518` (Agent Action), `:523-545` (Fallback) | ⚠ Checkbox-instead-of-radio |
| Per-mode body editor (text)                                   | `:264-305` — multiple text rows + "+ add" | ✅ |
| Per-mode body editor (LLM)                                    | `:307-326` — single multiline TextField | ✅ |
| Per-mode body editor (RAG)                                    | `:362-397` — `RagDbFields` + label override | ✅ |
| Per-mode body editor (DB action)                              | `:399-427` — `RagDbFields` (showDb) | ✅ |
| Per-mode body editor (Python code) — CodeMirror               | `:328-360` — `@uiw/react-codemirror` + `@codemirror/lang-python` | ✅ |
| RAG dropdown sourced from sibling `AgentRagElement` nodes     | `:145-154` (memo); `RagDbFields.tsx:105-132` | ✅ |
| Helper "Create an AgentRagElement from the palette first"     | `RagDbFields.tsx:120,132`            | ✅ |
| Mode-switch deletes siblings of other types + auto-creates    | `:207-258` (`setMode`)                | ✅ |
| Fallback section (separate radio block + editor)              | `:521-547`                           | ✅ |
| `colorOpen` / fillColor / lineColor / textColor               | `:434-437` (`NodeStyleEditor`)       | ✅ |

**Cosmetic deviation (only)**: v3 used `<input type="radio"
name="actionType">` (`agent-state-update.tsx:296-413`); v4 uses
`<Checkbox>` rows. Functional semantics match because mode-switch
deletes rows of the wrong type, so only one mode is ever active.
This was already noted in `parity-final/agentDiagram.md` and is not
a regression introduced after that audit.

## D — Other inspector panels

- `AgentIntentEditPanel.tsx` (312 LoC, post-SA-UX-FIX-2 rebuild):
  consolidated parent-intent inspector. Walks `parentId` chain to
  enumerate `AgentIntentBody`, `AgentIntentDescription`,
  `AgentIntentObjectComponent` children and edits them in place.
  Adds via `+ add` button stack a new child node under the intent.
  `setDescription` always mirrors onto the parent's
  `data.intent_description` so the v3 wire shape is preserved.
  **Verified: training phrases + description editing work.**
- `AgentIntentBodyEditPanel.tsx` (63 LoC): single textarea for the
  training phrase. Used when the user clicks a body child directly.
- `AgentIntentDescriptionEditPanel.tsx` (62 LoC): 3-row multiline
  textarea.
- `AgentIntentObjectComponentEditPanel.tsx` (83 LoC): `name`,
  `entity`, `slot`, `value`.
- `AgentRagElementEditPanel.tsx` (110 LoC): `name`, `dbSelectionType`,
  `ragDatabaseName`, `dbCustomName`, `dbQueryMode`, `dbOperation`,
  `dbSqlQuery` (CodeMirror Python via `RagDbFields`), `ragType`.
  **Strict superset of v3** (v3 showed only the "Name of RAG DB"
  field — see `agent-rag-element-update.tsx:21-27`).
- `AgentDiagramEdgeEditPanel.tsx` (539 LoC): predefined / custom mode
  toggle, 5 predefined types (`when_intent_matched`,
  `when_no_intent_matched`, `when_variable_operation_matched`,
  `when_file_received`, `when_no_file_received`), conditional
  sub-pickers (`intentName` Select sourced from sibling
  `AgentIntent`s with TextField fallback when none exist;
  `fileType` PDF/TXT/JSON; variable/operator/targetValue triple),
  custom branch with 7-event Select + condition list backed by
  CodeMirror Python (`:474-486`), numeric `params{}` editor, edge
  style + flip via `EdgeStyleEditor`.
- `AgentDiagramInitEdgeEditPanel.tsx` (17 LoC): "Initial-state
  marker. No editable fields." — matches v3 (no init-edge update
  file in v3 source).

## E — Edges

### Markers

`utils/edgeUtils.ts:343-356` — both `AgentStateTransition` (open
arrow, solid) and `AgentStateTransitionInit` (open arrow, dashed
`strokeDashArray: "10"`) cases are now present. **The two visual
gaps from `parity-final/agentDiagram.md` are CLOSED** (commit
`d8eda99` SA-FIX-Agent, point B). v3 markers: open arrow on both
edge types (`agent-state-transition-component.tsx:118-136`,
`agent-state-transition-init-component.tsx:62-87`); init edge was
**solid** in v3 — v4's dashed init line is a slight visual
deviation but it makes the two edge types easier to tell apart.

### 5 legacy transition shapes

`liftAgentTransitionDataToV4` (`versionConverter.ts:1409-1633`)
priority cascade: canonical predefined → canonical custom → legacy
flat predefined → legacy `condition: 'custom_transition'` + custom
fields → legacy nested `conditionValue`. Round-trip parametric
tests at `tests/round-trip/agentDiagram.test.ts:296-384` cover all
five fixtures and assert idempotence.

### Intent-name picker

`AgentDiagramEdgeEditPanel.tsx:125-358` — `intentNames` memo at
`:128-138` filters sibling `AgentIntent` nodes for the predefined
`when_intent_matched` Select; when no intents exist the field
falls back to a free-text `MuiTextField` with helper text "No
AgentIntent nodes — create one to enable the dropdown."

## F — v4 → v3 round-trip

`convertV4ToV3Agent` in `versionConverter.ts:2860-3010`:

- `AgentState` re-emits each row of `data.bodies` as a top-level
  `AgentStateBody` / `AgentStateFallbackBody` element with the
  **original id preserved** (`:2898`); preserves `replyType`,
  `ragDatabaseName`, `dbSelectionType`, `dbCustomName`,
  `dbQueryMode`, `dbOperation`, `dbSqlQuery`, `code`, `kind` (when
  ≠ `'do'`), `fillColor`, `textColor`. Re-emits the parent's
  `bodies: string[]` and `fallbackBodies: string[]` ordering arrays
  (`:2942-2943`).
- `AgentIntent` rolls a child `AgentIntentDescription`'s `name` up
  into the parent's `intent_description` if the parent doesn't
  already have one (`:2957-2970`).
- `AgentIntentDescription` and `AgentIntentObjectComponent` are
  **dropped** from the v3 elements map (`:2986-3000`) — v3's
  `AgentElementType` registry has no entry for them, so emitting
  them would silently fail. Round-trip parity is restored on
  re-import via `migrateAgentDiagramV3ToV4` re-creating the
  description child from the parent's `intent_description` text.

Round-trip tests at `tests/round-trip/agentDiagram.test.ts`:
**14 it blocks** (vs the 8 reported in `parity-final`):

1. v3-fixture migration assertions
2. v4 → v3 → v4 structural-equality
3. transition-rename round-trip
4. **inline-bodies re-emitted with original ids preserved**
5-9. parametric `AgentStateTransition shape (1-5)` round-trip
10-11. `AgentIntentDescription` / `ObjectComponent` v4 → v3 export
   safety
12. `normalizeAgentBodies` — fold floating `AgentStateBody`
13. `normalizeAgentBodies` — fold floating `AgentStateFallbackBody`
14. orphan-bodies dropped when no `AgentState` exists
15. no-op on a clean v4 model

## G — Floating-body prevention (CRITICAL)

User reported floating `AgentStateBody` nodes on the canvas. Three
defences are in place:

1. **Palette**: `constants.ts:919-977` does NOT register
   `AgentStateBody` / `AgentStateFallbackBody` /
   `AgentIntentBody` / `AgentIntentDescription` as draggable types.
   Cannot be dropped from the sidebar.
2. **Import normalisation**: `normalizeAgentBodies`
   (`versionConverter.ts:3514-3628`) runs from `importDiagram`
   (`:3640`) on every v4 fixture. Folds any top-level
   `AgentStateBody` / `AgentStateFallbackBody` onto the nearest
   `AgentState` (`parentId` first, else euclidean-nearest), with
   verbatim preservation of `replyType`, `ragDatabaseName`,
   `dbSelectionType`, `dbCustomName`, `dbQueryMode`, `dbOperation`,
   `dbSqlQuery`, `code`, `kind`, `fillColor`, `textColor`.
   Orphans are dropped with a warning.
3. **Runtime guard**: `diagramStore.ts:35,294,319,380` — `addNode`,
   `setNodes`, `setNodesAndEdges` all refuse incoming nodes whose
   `type` is `AgentStateBody` / `AgentStateFallbackBody`. Defence
   in depth against future regressions.

Test coverage: `tests/round-trip/agentDiagram.test.ts:490-614`
verifies all four floating-body scenarios.

## Critical gaps

None blocking. Three minor cosmetic deviations:

1. `AgentStateEditPanel` uses MUI `Checkbox` instead of v3's HTML
   `<input type="radio">` for the reply-mode picker. Functional
   semantics match — same documented in `parity-final`. Not a
   regression.
2. `AgentStateTransitionInit` v4 renders **dashed**;
   v3 rendered **solid**. Slight visual deviation but improves
   distinguishability from a regular transition. Not a regression
   relative to `parity-final` — that doc didn't catch this; it's
   the SA-FIX-Agent author's call.
3. `dbSelectionType` enum: v3 used `{'default', 'custom'}` only;
   `RagDbFields.tsx`/`AgentRagElementEditPanel.tsx` expose
   `'predefined'` as a third option for new-diagram authoring. The
   migrator preserves whatever v3 supplied verbatim, so legacy
   round-trip is clean. Likely intentional.

## Files of interest

- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/agentDiagram/AgentState.tsx:111-139` — auto-grow useEffect (`a31c4ec`)
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/agentDiagram/AgentState.tsx:33-82` — inline body row renderer
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/agentDiagram/AgentIntent.tsx:130-147` — second divider (PC-8 #3)
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/agentDiagram/AgentStateEditPanel.tsx:495-547` — 5-mode picker, two sections
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/agentDiagram/AgentIntentEditPanel.tsx:54-152` — consolidated rebuild walking child nodes
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/agentDiagram/AgentDiagramEdgeEditPanel.tsx:125-358,464-486` — intent picker + CodeMirror Python
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/edgeUtils.ts:343-356` — open-arrow markers (gap-fix)
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/versionConverter.ts:1409-1633` — 5-shape lifter
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/versionConverter.ts:2860-3010` — v4→v3 re-emit
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/versionConverter.ts:3514-3628` — `normalizeAgentBodies`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/store/diagramStore.ts:35,294,319,380` — runtime floating-body guard
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/constants.ts:919-977` — palette block (6 entries)
- `besser/utilities/web_modeling_editor/frontend/packages/library/tests/round-trip/agentDiagram.test.ts` (615 LoC, 14 it blocks)
