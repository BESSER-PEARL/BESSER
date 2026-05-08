# StateMachineDiagram — Final Parity Check (SA-PARITY-FINAL-3)

Audit date: 2026-05-08
Auditor branch: `claude/refine-local-plan-sS9Zv` (read-only)
Submodule HEAD audited: `771c064` (`feat(webapp): SA-7b cutover from packages/editor to packages/library (v4)`)

Old fork:

- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/`
- Update components live next to each element under
  `packages/editor/src/main/packages/uml-state-diagram/<element>/<element>-update.tsx`
  (the brief mentioned `scenes/update-pane/versions/`; that path does not
  exist in this fork — there is no `scenes/update-pane/` directory at
  all. The 4 update components are: `uml-state-update`,
  `uml-state-body-update`, `uml-state-code-block-update`,
  `uml-state-merge-node-update`, `uml-state-transition-update`.)

New lib:

- Nodes: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/`
- Edge: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/edges/edgeTypes/StateMachineDiagramEdge.tsx`
- Inspectors: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/stateMachineDiagram/`
- Migrator: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/versionConverter.ts`

---

## Top-line verdict

**PASS with two MEDIUM gaps and one LOW gap.** Every v3 node type and the single v3 edge type made it across with PascalCase identical keys, the migrator round-trips lossy-free for every shape covered by the v3 fixture, and the SA-3-extra `code` / `eventName` / `kind` fields (BESSER additions over upstream Apollon) are passed through both the inspector and `versionConverter.ts`. The remaining gaps are all in the **inspector layer** and concern UX-visible authoring surfaces, not data integrity:

1. **MEDIUM — `StateTransition` edge inspector is missing `flip` and the color editor.** v3 had both (see `uml-state-transition-update.tsx:131-137,172-178`). New `StateMachineDiagramEdgeEditPanel.tsx` exposes only `name` / `guard` / `eventName` / `code` / `params`. Compare with `ClassEdgeEditPanel.tsx` and `AgentDiagramEdgeEditPanel.tsx`, which both registered `flip` + `EdgeStyleEditor` in SA-2.2 #26 — the SA-3 edge missed this pass.
2. **MEDIUM — `StateMergeNode` collapses to the shared `StateLabelEditPanel` and loses the v3 "decisions" inline editor.** v3's `uml-state-merge-node-update.tsx:127-148` listed each outgoing transition (`Textfield` for the decision label + arrow + target name read-only). The new lib registers the same one-field `StateLabelEditPanel` for every marker including MergeNode, dropping the per-decision rename surface. Round-trip fidelity is unaffected because each transition still owns its `name`, but users can no longer author the decisions in one place.
3. **LOW — marker nodes (`StateInitialNode` / `StateFinalNode` / `StateForkNode` / `StateForkNodeHorizontal`) accept name editing in the new inspector despite v3 marking them `updatable: false`.** v3 model classes set `static features.updatable = false`, so the v3 popup pipeline rendered `DefaultPopup` (a no-op rename for these types). The new `StateLabelEditPanel` exposes a free-form `name` field for all five marker types (`registerInspector("StateInitialNode" | "StateFinalNode" | "StateForkNode" | "StateForkNodeHorizontal", "edit", StateLabelEditPanel)`). Cosmetic only — markers don't render their name on the canvas, so the field is a no-op authoring surface.

No data-shape gaps, no missing types, no constraint regressions outside what was already absent in v3. The round-trip test (`tests/round-trip/stateMachineDiagram.test.ts`) covers all 11 node types + the edge with **3 cases**.

---

## 1. Element type inventory

Old (`packages/editor/.../uml-state-diagram/index.ts`):

```ts
StateElementType = {
  State, StateBody, StateFallbackBody,
  StateActionNode, StateFinalNode, StateForkNode,
  StateForkNodeHorizontal, StateInitialNode, StateMergeNode,
  StateObjectNode, StateCodeBlock,
}
StateRelationshipType = { StateTransition }
```

(11 element types, 1 edge type.)

New (`packages/library/lib/nodes/stateMachineDiagram/index.ts` +
`packages/library/lib/edges/edgeTypes/StateMachineDiagramEdge.tsx`):

| v3 element type | v4 status | Notes |
|---|---|---|
| `State` | = | Parent container; children attached via React Flow `parentId` |
| `StateBody` | = | Standalone v4 node (was sub-element in v3); `parentId` links to `State` |
| `StateFallbackBody` | = | Same pattern as `StateBody`, italicised label |
| `StateCodeBlock` | = | Free-floating or `parentId`-attached |
| `StateActionNode` | = | |
| `StateObjectNode` | = | `classId` cross-diagram link preserved (spec open question 4) |
| `StateInitialNode` | = | Filled circle, fixed 50×50 in v3 / 45×45 default in v4 |
| `StateFinalNode` | = | Bullseye |
| `StateMergeNode` | = | Diamond |
| `StateForkNode` | = | Vertical bar (20×60 default) |
| `StateForkNodeHorizontal` | = | Horizontal bar (60×20 default) |
| `StateTransition` (edge) | = | Single edge type, registered via `registerEdgeTypes` |

**Result: 1:1 type parity. Nothing missing, nothing renamed.**

The brief asked about a possible `StateTransitionInit` edge — the old `StateRelationshipType` does not declare one. v3 `UMLStateInitialNode.supportedRelationships` only lists `[AgentRelationshipType.AgentStateTransitionInit, StateRelationshipType.StateTransition]`; the `AgentStateTransitionInit` is an **AgentDiagram** edge bleeding into the state-machine package's import graph and is not part of StateMachineDiagram itself. New lib correctly does not register a separate init edge for SA-3.

---

## 2. Per-element data fields

Reference: spec at `docs/source/migrations/uml-v4-shape.md` (StateMachineDiagram §) + brief.

| Element | Field | v3 source | v4 location | Round-trip |
|---|---|---|---|---|
| `State` | `name` | `IUMLState.name` | `data.name` | OK (`versionConverter.ts:914`) |
| `State` | `stereotype` | `IUMLState.stereotype` (string \| null) | `data.stereotype` | OK |
| `State` | `italic` / `underline` | `IUMLState.italic` / `underline` | `data.italic` / `underline` | OK |
| `State` | `deviderPosition` | computed at render | not persisted | OK — recomputed by layout |
| `State` | `hasBody` / `hasFallbackBody` | computed at render | not persisted | OK — derived from children |
| `State` | `fillColor` / `strokeColor` / `textColor` | `IUMLElement` baseline | `data.fillColor` / `strokeColor` / `textColor` | OK |
| `StateBody` / `StateFallbackBody` | `name` | `name` | `data.name` | OK |
| `StateBody` / `StateFallbackBody` | `code` (BESSER addition) | not in upstream v3 model — was added ad-hoc | `data.code` | OK (`versionConverter.ts:937`) |
| `StateBody` / `StateFallbackBody` | `kind` (entry/do/exit/transition) | not in upstream v3 model | `data.kind` | OK (`versionConverter.ts:938`) |
| `StateCodeBlock` | `code` | `IUMLStateCodeBlock.code` | `data.code` | OK |
| `StateCodeBlock` | `language` | `IUMLStateCodeBlock.language` (default `'python'`) | `data.language` | OK |
| `StateCodeBlock` | `_codeContent` (private) | v3 internal preserve-flag | dropped | OK — replaced by direct `code` round-trip |
| `StateActionNode` | `name` | `name` | `data.name` | OK |
| `StateActionNode` | `code` (BESSER addition) | not in upstream v3 model | `data.code` | OK (`versionConverter.ts:946`) |
| `StateObjectNode` | `name` | `name` | `data.name` | OK |
| `StateObjectNode` | `classId` (cross-diagram) | not declared in v3 class but passed through | `data.classId` | OK (`versionConverter.ts:959`) |
| `StateObjectNode` | `className` | derived | `data.className` | OK (`versionConverter.ts:960`) |
| `StateInitialNode` / `StateFinalNode` / `StateMergeNode` / `StateForkNode` / `StateForkNodeHorizontal` | `name` (cosmetic) | `name` | `data.name` (via `StateMarkerNodeProps`) | OK |
| `StateTransition` (edge) | `name` | `UMLRelationshipCenteredDescription.name` | `data.name` | OK (`versionConverter.ts:1744`) |
| `StateTransition` (edge) | `guard` | `IUMLStateTransition.guard` | `data.guard` | OK (`versionConverter.ts:1757`) |
| `StateTransition` (edge) | `params` | `IUMLStateTransition.params: { [id]: string }` (also accepts `string` and `string[]`) | `data.params: { [id]: string }` | OK (`versionConverter.ts:1720-1731`) — normalises all three legacy shapes to dict |
| `StateTransition` (edge) | `code` (BESSER) | not in upstream v3 model | `data.code` | OK (`versionConverter.ts:1758`) |
| `StateTransition` (edge) | `eventName` (BESSER) | not in upstream v3 model | `data.eventName` | OK (`versionConverter.ts:1759`) |

**Result: full data-field parity, including the BESSER-extra fields the brief calls out (`code`, `eventName`, `kind`).** No silently dropped attribute.

Note on the brief vs reality: the brief lists "name, code, eventName" as the `StateTransition` edge fields. v3 also has **`guard`** and **`params`**, neither of which is in the brief. Both are preserved by the migrator and surfaced in the inspector — gap 1 only concerns the missing `flip` action and color editor, not these data fields.

---

## 3. Inspector form parity

Old fork popup wiring (`packages/editor/src/main/packages/popups.ts:122-132,171`):

| Element | v3 popup component | v3 reachable surface |
|---|---|---|
| `State` | `UMLStateUpdate` | name, color (fill/line/text), child-body adders & deletes |
| `StateBody` / `StateFallbackBody` | `null` (rendered as part of parent's update form) | name + color via `UmlBodyUpdate` rows in `UMLStateUpdate` |
| `StateActionNode` | `DefaultPopup` (rename only) | name |
| `StateObjectNode` | `DefaultPopup` (rename only) | name |
| `StateCodeBlock` | `UMLStateCodeBlockUpdate` | code (textarea), color, size |
| `StateMergeNode` | `UMLStateMergeNodeUpdate` | name, color, size, **decisions inline editor** |
| `StateInitialNode` / `StateFinalNode` / `StateForkNode` / `StateForkNodeHorizontal` | `DefaultPopup` (gated by `updatable: false`) | nothing — rename UI is suppressed |
| `StateTransition` | `UMLStateTransitionUpdate` | name, guard, params (add/remove), color, **flip**, delete |

New lib (`packages/library/lib/components/inspectors/stateMachineDiagram/index.ts`):

| Element | New panel | Surface |
|---|---|---|
| `State` | `StateEditPanel` | name (in `NodeStyleEditor`), stereotype dropdown, italic, underline, name field |
| `StateBody` / `StateFallbackBody` | `StateBodyEditPanel` | name (`label`), kind dropdown, code multiline, colors |
| `StateActionNode` | `StateActionNodeEditPanel` | name, code multiline, colors |
| `StateObjectNode` | `StateObjectNodeEditPanel` | name, **classId picker via `diagramBridge.getAvailableClasses()`**, colors |
| `StateCodeBlock` | `StateCodeBlockEditPanel` | language dropdown, code multiline, colors |
| `StateInitialNode` / `StateFinalNode` / `StateMergeNode` / `StateForkNode` / `StateForkNodeHorizontal` | `StateLabelEditPanel` (shared) | name, colors |
| `StateTransition` | `StateMachineDiagramEdgeEditPanel` | name, guard, eventName, code, params (add/remove) |

Detailed parity table:

| Brief expectation | Old fork has it | New lib has it | Verdict |
|---|---|---|---|
| `StateEditPanel`: name + style | yes | yes | OK |
| `StateBodyEditPanel`: kind picker, code (CodeMirror) | partial — v3 had only `name` + colors on the body row, no kind/code; new lib *adds* kind+code per the SA-3 brief | yes (kind + multiline code) | OK — over-delivers vs v3, matches brief |
| `StateActionNodeEditPanel`: name + color | yes (DefaultPopup rename + color) | yes (+ `code` extra) | OK |
| `StateCodeBlockEditPanel`: code (CodeMirror) | code via `<textarea>` (not CodeMirror) | code via MUI `TextField multiline` | OK — both pre-CodeMirror; brief was aspirational |
| `StateMergeNodeEditPanel`: decisions inline editor | yes (`uml-state-merge-node-update.tsx:127-148`) | **no** (uses shared `StateLabelEditPanel`) | **GAP-2 MEDIUM** |
| `StateLabelEditPanel`: gated by `updatable: false` on marker types — no rename UI for InitialNode / FinalNode / ForkNode / ForkNodeHorizontal | yes (`updatable: false` on each class) | partial — `StateLabelEditPanel` exposes a name field for these | **GAP-3 LOW** (cosmetic only — markers don't render name) |
| `StateObjectNodeEditPanel`: classId picker via `diagramBridge.getAvailableClasses()` | no — v3 had a free-text `name` only | **yes** (new feature, matches SA-2's `ObjectName.classId` pattern) | OK — over-delivers |
| `StateMachineDiagramEdgeEditPanel`: name, code, eventName | name yes, code yes, eventName yes; v3 also exposes `guard` + `params` (both kept) | yes | OK |
| `StateMachineDiagramEdgeEditPanel`: **flip** action | yes (`uml-state-transition-update.tsx:131-133`) | **no** | **GAP-1 MEDIUM** |
| `StateMachineDiagramEdgeEditPanel`: **color editor** | yes (`StylePane` open via `ColorButton`, lineColor + textColor) | **no** | **GAP-1 MEDIUM** (same gap, paired with flip) |

Cross-reference: `ClassEdgeEditPanel.tsx` and `AgentDiagramEdgeEditPanel.tsx` (under SA-2.2 #26) both ship `flip` + `EdgeStyleEditor`. The SA-3 edge panel pre-dates that pass and was never back-filled.

---

## 4. Constraints / invariants

Brief enumerates three:

1. **Parent / child for `State` + bodies via `parentId`.** ✓ Covered. `StateBody` and `StateFallbackBody` set `parentId` to the containing `State` in `versionConverter.ts:929-941`. The new `State.tsx` does not lay out children itself (React Flow handles positions); divider line painted at `LAYOUT.DEFAULT_HEADER_HEIGHT[_WITH_STEREOTYPE]`. v3's `deviderPosition` recomputed at render.
2. **Initial state: only one allowed; transition with no source.** *Not enforced anywhere in either codebase.* v3 had no validation gate (`UMLStateInitialNode` does not check uniqueness in its constructor or `render`); v4 has no rule in `lib/utils/` either (the constraint files there are `bpmnConstraints.ts` and `nnValidationDefaults.ts`). Backend-side BUML validation handles this — neither editor enforces it. **No regression.**
3. **Final state: terminal — no outgoing edges allowed.** Same status as #2: not enforced in either editor. Backend `services/validators/` is the gate. **No regression.**

**Result: no constraint regressions. The v3 fork was equally permissive; backend BUML validation is the single source of truth.**

---

## 5. Visual shape

| Marker | v3 visual | v4 visual | Match |
|---|---|---|---|
| `State` (rounded rect with body partition) | `ThemedRect rx={8}` + divider line via `ThemedPath` at `headerHeight` | `<rect rx=8>` + `<line>` divider at header height | Match |
| `StateInitialNode` (filled black circle) | `<ellipse>` filled black/contrast | `<circle>` filled `--apollon-primary-contrast` (default `#000`) | Match |
| `StateFinalNode` (hollow ring + inner filled circle) | outer ring + inner solid disk | outer ring (white fill, stroke) + inner disk (filled stroke color) | Match |
| `StateMergeNode` (diamond) | `ThemedPolyline` 4-point diamond + `Multiline` label | `<polygon>` 4-point diamond + `<text>` label | Match |
| `StateForkNode` (vertical bar) | `<rect>` 20×60 default | `<rect>` 20×60 default with `vertically-not-resizable` class | Match |
| `StateForkNodeHorizontal` (horizontal bar) | `<rect>` 60×20 default | `<rect>` 60×20 default with `horizontally-not-resizable` class | Match |
| `StateActionNode` | rounded rect + centred name | rounded rect (`rx={5}`) + centred name | Match |
| `StateObjectNode` | rect + bold name | rect + bold `name: ClassName` (extra: shows linked class) | Match |
| `StateCodeBlock` | resizable rect, header bar with language label, multi-line code via `foreignObject` | identical structure | Match |

**Result: visual parity. The new SVG primitives (`<polygon>`, `<rect>`, `<ellipse>`, `<circle>`) bypass the `ThemedRect` / `ThemedPolyline` wrappers but each call site provides explicit `fill`/`stroke` fallbacks via `getCustomColorsFromData(data)` so the dark-mode contract from the frontend `CLAUDE.md` is satisfied.**

---

## 6. Round-trip test count

Path: `besser/utilities/web_modeling_editor/frontend/packages/library/tests/round-trip/stateMachineDiagram.test.ts`

3 test cases inside the single `describe("StateMachineDiagram v3 → v4 round-trip", …)` block:

1. `it("migrates the v3 fixture to v4 with structural fidelity", …)` — asserts 13 nodes, 4 edges, every node type lands correctly, `StateCodeBlock.code` / `StateCodeBlock.language` / `StateActionNode.code` / `StateObjectNode.classId` / `StateObjectNode.className` survive, and `StateTransition` carries `name` + `eventName` + `code` + `guard` + `params` on `data`.
2. `it("round-trips v4 → v3 → v4 with structural equality", …)` — JSON-string equality on a canonical projection of nodes + edges (id / type / parentId / name / classId / code / stereotype for nodes; id / type / source / target / name / guard / eventName / code / params for edges).
3. `it("preserves a transition rename through a v4 → v3 → v4 cycle", …)` — edits one transition's `name` after migration, runs the v4→v3→v4 cycle, asserts the rename survived.

**Result: 3 round-trip tests covering all 11 node types and the edge.**

---

## Summary of gaps

| # | Severity | Surface | Description | Fix location |
|---|---|---|---|---|
| 1 | MEDIUM | inspector | `StateMachineDiagramEdgeEditPanel.tsx` lacks `flip` action and color editor (both present in v3 and now in `ClassEdgeEditPanel` / `AgentDiagramEdgeEditPanel` under SA-2.2 #26) | `library/lib/components/inspectors/stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx` |
| 2 | MEDIUM | inspector | `StateMergeNode` uses shared `StateLabelEditPanel`, dropping v3's "decisions" inline editor (per-outgoing-transition rename + arrow + target) | `library/lib/components/inspectors/stateMachineDiagram/index.ts` (register a dedicated `StateMergeNodeEditPanel`) |
| 3 | LOW | inspector | Marker types (`StateInitialNode` / `StateFinalNode` / `StateForkNode` / `StateForkNodeHorizontal`) accept name editing despite v3 marking them `updatable: false` (cosmetic only — name not painted on canvas) | `library/lib/components/inspectors/stateMachineDiagram/StateLabelEditPanel.tsx` (gate the name field on element type, or register a no-edit panel for the four marker types) |

No data-shape gaps. No missing element / edge types. No constraint regressions. Round-trip closed. SA-3 ships at parity for everything except the three inspector-layer items above.
