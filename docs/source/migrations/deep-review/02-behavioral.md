# Deep Review 02 — Behavioral Matrix

Read-only behavioral audit of the v3 → v4 (`packages/library`) migration.
Each user-facing action is traced from the trigger (palette drop /
connection-end / inspector `onChange`) down to the model-mutation site,
and compared against the v3 reference in `packages/editor/src/main/`.

Submodule HEAD audited: `05668dce84b9ff5e619134bb64a4851b68e033f6`
(other fix agents are concurrently writing — verdicts below reflect the
last fully-committed snapshot).

## Summary

39 actions audited. **24 PASS / 6 PARTIAL / 5 FAIL / 4 N/A**.

The single largest behavioral regression is that
`packages/library/lib/utils/edgeUtils.ts::getDefaultEdgeType()` has no
case for any of the four BESSER-only diagram types
(`StateMachineDiagram`, `AgentDiagram`, `UserDiagram`, `NNDiagram`).
All connections in those four diagrams fall through to the
`default: "ClassUnidirectional"` branch — i.e. dragging from one State
to another produces a `ClassUnidirectional` edge instead of a
`StateTransition`. The same defect blocks B4, B5, and the implicit
default-edge-type for any User-diagram link.

## Per-action table

| ID  | Action | Verdict | Code path | Gap |
|-----|--------|---------|-----------|-----|
| A1  | Drop node from palette places at drop position with v3 default `data` | PASS | `components/DraggableGhost.tsx:62` (`onDrop`) → `setNodes` with `defaultData` from `constants.ts::defaultDropElementConfigs` | — |
| A2 (class attr) | Drop attribute child into class | N/A | Not a palette drop in either v3 or v4 — inline editing in inspector (`ClassEditPanel.tsx`) | Behavior matches v3 |
| A2 (NN layer)   | Drop NN layer into NN container so `parentId` gets set | **FAIL** | `DraggableGhost.tsx:120-131` filters by `isParentNodeType`. `utils/nodeUtils.ts:225` does **not** include `NNContainer`, so the parent-search returns 0 candidates | NNContainer never receives children; the NN diagram cannot represent layer ⊂ container |
| A2 (BPMN/Package/UseCase/Component/Deployment) | Drop child into parent | PASS | `nodeUtils.ts:225` covers package/activity/useCaseSystem/componentSubsystem/deploymentNode/deploymentComponent + bpmnPool/group/subprocess/transaction/callActivity | — |
| A3  | Default-name uniqueness on collision (Class2, Class3 …) | N/A | v3 also drops every new `Class` with name "Class" (palette default `class-preview.ts`); the lib does the same. The expectation in the matrix is incorrect | No regression |
| B1  | Drag source-handle → target-handle creates an edge | PASS | `hooks/useConnect.ts::onConnect` (line 104) and `onConnectEnd` (line 127). Both compute `defaultEdgeType` via `getDefaultEdgeType(diagramType)` | — for the 13 stock diagram types |
| B2  | Class ↔ Class does NOT auto-create an OCL link | PASS | `useConnect.ts:23::resolveClassEdgeType` only returns `ClassOCLLink` when one endpoint is `ClassOCLConstraint`; otherwise returns `defaultEdgeType` | — |
| B3  | Class ↔ ClassOCLConstraint auto-creates `ClassOCLLink` | PASS | `useConnect.ts:23-31` (logic shared by `onConnect` + `onConnectEnd`) | — |
| B4  | AgentState ↔ AgentState produces `AgentStateTransition` | **FAIL** | `useConnect.ts::onConnect` calls `getDefaultEdgeType(diagramType)` which has **no `case "AgentDiagram"`** → falls through to `"ClassUnidirectional"`. Edge component `AgentDiagramEdge` is registered (`registerEdgeTypes({AgentStateTransition: …})`) but is never selected | Connections in AgentDiagram render as ClassUnidirectional, breaking visual + serialization parity |
| B5  | NN layer ↔ NN layer produces `NNNext` | **FAIL** | Same root cause: no `case "NNDiagram"` in `getDefaultEdgeType`. The three NN edge types (`NNNext`/`NNComposition`/`NNAssociation`) are registered but never resolved as default | All NN connections come out as ClassUnidirectional |
| (additional)    | StateMachine ↔ StateMachine produces `StateTransition` | **FAIL** | No `case "StateMachineDiagram"` in `getDefaultEdgeType`. Falls back to ClassUnidirectional | Adjacent regression — not in matrix but identical defect |
| (additional)    | UserDiagram default edge `UserModelLink` | **FAIL** | No `case "UserDiagram"` in `getDefaultEdgeType` | UserDiagram links wrong-typed |
| C1  | Click node → properties panel populates with editor for that node | PASS | `App.tsx:157` → `<PropertiesPanel/>` → `inspectors/registry.ts::getInspector(type, "edit")` resolves type-specific body | — |
| C2  | Edit a field → `data.X` updates → canvas re-renders | PASS | Each inspector's `update(...)` calls flow through `diagramStore.setNodes`/`onNodesChange`, which updates Y.Map under origin "store"; subscribers on store re-render | — |
| C3  | ER ↔ UML notation toggle re-renders without `editorRevision++` | PASS | `store/settingsStore.ts` mirrors `settingsService` and exposes `useClassNotation()` selector; class-row renderers subscribe directly. No editor remount | Replaces v3 `editorRevision++` hack as documented |
| C4  | Settings flips (`showAssociationNames`, `showInstancedObjects`, `showIconView`) live re-render | PASS | Same `settingsStore` selectors; subscribed renderers update on next paint | — |
| D1  | Type in attribute → undo reverts edit | PASS | All inspector `update` calls go through store setters, which `ydoc.transact(..., "store")`. `Y.UndoManager` is constructed with `trackedOrigins: new Set(["store", "remote"])` (`diagramStore.ts:114`) | — |
| D2  | Drag node → undo reverts position | PASS | `useNodeDragStop.ts:80` uses `setNodes` (origin "store"). UndoManager captures | — |
| D3  | Delete node → undo restores node + its edges | PASS | `diagramStore.onNodesChange` (lines 414-444) handles `remove` by also deleting connected edges in the same Yjs transaction → undo restores both | — |
| E1  | Export model JSON in v4 wire shape | PASS | `besser-editor.tsx:456 get model` returns `{ id, version: "4.0.0", nodes, edges, … }` with `mapFromReactFlowNodeToBesserNode` (`diagramTypeUtils.ts:22`) producing v4-shaped nodes (no v3 `elements` map) | — |
| E2  | `editor.exportAsSVG()` returns faithful SVG | PASS (code-only) | `besser-editor.tsx:367 exportAsSVG` → `exportModelAsSvg` mounts a hidden ReactFlow off-screen, waits for fonts + 2× rAF, calls `getSVG` | Visual fidelity untestable from static code review — see test gaps |
| E3  | Export PNG | N/A | Neither v3 nor v4 implements PNG in the editor itself — the webapp converts SVG → PNG via canvas | — |
| F1  | Box-select multiple nodes → selection in `diagramStore` | PASS | `diagramStore.onNodesChange` (line 358-399) handles batched `select` changes; `selectedElementIds` array updates accordingly | — |
| F2  | Cmd+C / Cmd+V duplicates with new IDs + offset | PASS | `useKeyboardShortcuts.ts:88-114` → `useSelectionForCopyPaste.ts::copySelectedElements`/`pasteElements`. New IDs via `generateUUID` + `pasteCount * CANVAS.PASTE_OFFSET_PX` offset | — |
| F3  | Backspace on multi-select removes nodes + their edges | PASS | `useKeyboardShortcuts.ts:49` handles `Delete`; React-Flow's default `deleteKeyCode='Backspace'` triggers `onNodesChange` with `remove`, which also strips connected edges in `diagramStore.ts:430-437` | — |
| F3 (edges to outside survive) | PARTIAL | The remove handler deletes edges of removed nodes, but does **not** distinguish "between selected" vs "to outside" — every connected edge is removed. v3 had identical behavior, so this is parity — but the matrix's expectation that "edges to outside survive" was incorrect for v3 too | Not a regression |
| G1  | ClassDiagram Enumeration hides Methods section | PASS | `ClassEditPanel.tsx:1036-1038` `{nodeData.stereotype !== "Enumeration" && (...)}` | — |
| G2  | Visibility uses symbol `+ - # ~`, not full word | PASS | `utils/typeNormalization.ts:71 VISIBILITY_SYMBOLS` + `utils/classifierMemberDisplay.ts:119` (`visSymbol = VISIBILITY_SYMBOLS[member.visibility ?? "public"]`). Inspector `<Select>` shows symbol labels (`ClassEditPanel.tsx:86-89`) but stores the word — round-trip-clean | — |
| G3  | ObjectDiagram classId picker auto-populates attributes | PASS | `ObjectEditPanel.tsx:423::handleClassChange` copies attrs from `availableClasses.find(...)` into `data.attributes` | — |
| G4  | Only one initial state per diagram | N/A | Not enforced in v3 either (no constraint in `uml-state-diagram/`). Matrix expectation incorrect | — |
| G5  | `AgentStateTransitionInit.source` is null | PARTIAL | The edge schema in `BesserEdge` (`typings.ts:42`) declares `source: string` (non-nullable). xyflow requires both source and target. The init-edge keeps a synthetic source instead of `null` (`AgentDiagramInitEdge.tsx:30-45`) | Wire-shape divergence vs. v3 (which serialized `source: null`). Backend converter at `services/converters/json_to_buml/agent_diagram_processor.py` may need to accept the synthetic source ID |
| G6  | NN container parent + layer children via `parentId` | **FAIL** | (a) `nodeUtils.ts::isParentNodeType` does not include `NNContainer`. (b) `utils/bpmnConstraints.ts::canDropIntoParent` has no rule for `parentType === "NNContainer"` (default `true` is unreachable because of (a)). Layer drops onto containers always come out as top-level nodes | Cannot build a hierarchical NN model — the entire NN-diagram parent-child contract is broken on the drop path |
| (cross-cutting) | StateBody / AgentState body parent-child | PARTIAL | `StateBody`, `AgentStateBody`, `AgentStateFallbackBody` etc. exist as node types but, like `NNContainer`, are not in `isParentNodeType`. v3 had `inside` semantics; lib has only the bpmnPool-style list | Verify whether v3's State diagram actually nests bodies via `parentId`; if it does, this is a parallel parent-child regression |
| (palette)       | Palette entries seeded for all 6 BESSER diagrams | PARTIAL | `constants.ts::defaultDropElementConfigs` has explicit entries for ClassDiagram + StateMachineDiagram (line ~775) + others; need to confirm AgentDiagram/UserDiagram/NNDiagram entries are registered via `registerPaletteEntry` from their respective `nodes/<diagram>/index.ts`. Surface looked complete, deeper audit pending | — |
| (edge inspectors) | Edge double-click → inspector opens for edge | PASS | `App.tsx:141` `onEdgeDoubleClick={onEdgeDoubleClick}` from `useElementInteractions` | — |
| (typings)       | `BesserEdge.points` always present | PASS | `mapFromReactFlowEdgeToBesserEdge` ensures `points: edge.data?.points ?? []` (`diagramTypeUtils.ts:51`) | — |
| (assessment)    | Assessment mode read-only edge cases | PASS | `App.tsx:82-89` gates editing on `isDiagramModifiable` | — |
| (Yjs sync)      | Remote edits propagate without infinite loop | PASS | `diagramStore` updates from yjs use `updateNodesFromYjs`/`updateEdgesFromYjs` which preserve local `selected` flags | — |

## Critical behavioral regressions

Ranked by severity (1 = most severe).

1. **`getDefaultEdgeType` is missing all four BESSER-only diagram types.**
   Files: `packages/library/lib/utils/edgeUtils.ts:769-803`. The `switch`
   has no case for `StateMachineDiagram`, `AgentDiagram`, `UserDiagram`,
   or `NNDiagram`, so every connection in those diagrams falls through
   to `"ClassUnidirectional"`. Effects:
   - B4 fail (Agent transitions → ClassUnidirectional)
   - B5 fail (NN connections → ClassUnidirectional)
   - StateMachine / User edges identical defect
   The unit test file `tests/unit/edgeUtils.test.ts:1031-1058` does not
   even assert these four cases — the regression is invisible to CI.
   **One-line fix per case + 4 new test rows.**

2. **NN container parent-child is broken at the drop path.**
   Files: `packages/library/lib/utils/nodeUtils.ts:225` and
   `packages/library/lib/utils/bpmnConstraints.ts:5`. `NNContainer` is
   missing from `isParentNodeType` (the gate that `DraggableGhost.tsx`
   and `useNodeDragStop.ts` both consult), and has no allow-list entry
   in `canDropIntoParent`. Layer drops onto a container therefore drop
   the layer at canvas top-level, so the NN model can never represent
   `Layer ⊂ NNContainer`. Backend's `nn_diagram_processor.py` has been
   updated for the parent-child shape (per the SA-5 brief), so the
   shapes are mismatched on round-trip.

3. **State machine / Agent body-parent-child likely identical defect.**
   `StateBody`, `AgentStateBody`, `AgentStateFallbackBody`, etc. follow
   the same registration pattern as `NNContainer` and are likewise
   absent from `isParentNodeType`. If v3 nests them via `parentId`, the
   same drop-path filter breaks the State and Agent diagrams. Needs a
   v3 cross-check (the matrix only listed NN explicitly).

4. **`AgentStateTransitionInit.source` is not null on the wire.**
   Files: `packages/library/lib/typings.ts:42` (`source: string`,
   non-optional) and `AgentDiagramInitEdge.tsx:30-45`. v3 emitted
   `source: null` for the init edge marker; v4 keeps a synthetic source
   id. The backend converter
   (`agent_diagram_processor.py::process_init_transitions`) must
   either learn to recognize this synthetic id or the lib must emit
   `null` (which xyflow does not accept directly — would need a
   serialization-time transform in `mapFromReactFlowEdgeToBesserEdge`).

5. **Test coverage cliff at `getDefaultEdgeType`.**
   Files: `tests/unit/edgeUtils.test.ts:1031-1058`. Adding the four
   missing diagram-type rows as `it.each` cases would catch issue #1
   and prevent regression once it's fixed. Also no unit test asserts
   that NNContainer is recognized as a parent type — adding one would
   catch issue #2.

## Test gaps (untestable from code reading)

Items below cannot be verified by static trace; proposed Playwright
specs follow each.

- **E2 (SVG visual fidelity)** — code shows the off-screen render
  pipeline executes, but pixel/layout fidelity vs. on-screen is
  unverifiable here.
  *Spec:* `tests/e2e/export-svg.spec.ts` — for each of the 6 BESSER
  diagrams, place a representative model, call `editor.exportAsSVG()`,
  and assert (a) SVG parses, (b) contains the expected element ids,
  (c) bounding box matches the on-screen `getBoundsOfNodes` within
  ±2 px.

- **B4 / B5 visual edge rendering** — even after the
  `getDefaultEdgeType` fix, the edge marker styles must match v3.
  *Spec:* `tests/e2e/connection-types.spec.ts` — palette-drag two
  AgentStates, drag a connection, assert the resulting edge has class
  `react-flow__edge-AgentStateTransition` and the SVG path picks up
  the configured marker. Repeat for NN, State, User.

- **C3 / C4 live re-render proof** — code paths exist; need a runtime
  check that flipping a setting paints the canvas without remount
  (no DOM node-id churn).
  *Spec:* `tests/e2e/settings-live.spec.ts` — capture the `id`
  attribute of the root canvas element, toggle `showIconView`, assert
  the same id post-toggle (no remount) and that a flagged node
  changed its rendered glyph.

- **D2 (drag-undo position fidelity)** — undo restores via Y.UndoManager,
  but the snap-to-grid behavior + parent re-resize on undo is
  position-sensitive.
  *Spec:* `tests/e2e/undo-drag.spec.ts` — drag a node from (200,200)
  to (300,300), undo, assert `node.position` is exactly (200,200) and
  no parent-resize artifacts remain.

- **F2 paste at viewport / paste counter** — the `pasteCountRef`
  increments per consecutive Cmd-V; behavior at the edge (paste after
  click-elsewhere clears the counter? — yes, because `c`/`x` reset it
  but pane click does not) is hard to reason about statically.
  *Spec:* `tests/e2e/paste-offset.spec.ts` — copy 1 node, paste 3
  times, assert positions are originalX + (1,2,3) × `PASTE_OFFSET_PX`.

- **F3 multi-select-then-Backspace, edges to outside** — code shows
  every connected edge is removed; matrix expectation that "edges to
  outside survive" needs validation against v3 actual behavior.
  *Spec:* `tests/e2e/multiselect-delete.spec.ts` — build a
  ClassDiagram with classes A, B, C and edges A→B, B→C, A→C; select
  A and B; press Backspace; assert what survives. Compare to v3
  output of the same scenario.

- **G5 init-edge serialization** — round-trip through the backend
  `agent_diagram_processor.py` to confirm the synthetic source id is
  treated equivalently to v3's `source: null`.
  *Spec:* `tests/round-trip/agentDiagramInitTransition.test.ts` — a
  pure-Vitest round-trip test using the existing converters,
  asserting the BUML `initial_state` is recovered identically.
