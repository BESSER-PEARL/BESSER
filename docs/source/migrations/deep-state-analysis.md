# Deep audit — StateMachineDiagram (v3 → v4)

Read-only, evidence-based audit of every StateMachineDiagram element. Cross-references the v3 fork at `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/` against the v4 library at `besser/utilities/web_modeling_editor/frontend/packages/library/lib/`. All paths below are absolute from the repo root unless noted.

## Verdict

Structural parity is **largely complete**: all 11 v3 element kinds (`State`, `StateBody`, `StateFallbackBody`, `StateActionNode`, `StateObjectNode`, `StateInitialNode`, `StateFinalNode`, `StateMergeNode`, `StateForkNode`, `StateForkNodeHorizontal`, `StateCodeBlock`) plus the `StateTransition` edge are registered in v4 with PascalCase identifiers identical to v3, the migrator passes them through verbatim, and the bidirectional v3↔v4 round-trip preserves both the `parentId` body relationship and the BESSER-only `code`/`eventName` edge fields. Inspector parity is complete after SA-FIX-State PC-5/PC-6 added (a) the merge-node decisions editor, (b) the `updatable: false` short-circuit for Initial/Final/Fork/ForkHorizontal, (c) CodeMirror Python on the body / code-block / transition `code` fields, and (d) flip + style controls on the transition panel.

Three concrete defects remain: **(1)** the `StateTransition` edge has **no marker entry in `getEdgeMarkers`** (`packages/library/lib/utils/edgeUtils.ts:174-380`), so the edge renders as a plain line — v3 painted an open arrowhead via the inline `M0,29 L30,15 L0,1` path. **(2)** v3 carried a `guard` field on the transition that round-tripped via `IUMLStateTransition.guard`; v4 carries it on `data.guard` and the inspector exposes it, but the transition edge label in v4 only composes `[name] [guard]` and **omits the `params` dict from the rendered label** (v3 rendered `name [params...] [guard]`, see `uml-state-transition-component.tsx:46-61`). **(3)** Palette `StateActionNode` default `width: DEFAULT_ELEMENT_WIDTH (160)` mismatches v3's auto-sized bounds (`calculateNameBounds` in `uml-state-action-node.ts:13`); minor but visible on first drop.

## Top three gaps

1. **No arrowhead on `StateTransition`** — `packages/library/lib/utils/edgeUtils.ts` switch falls through to the default branch (no `markerEnd`). v3 used an inline open-V marker. Fix: add `case "StateTransition": return { markerPadding, markerEnd: "url(#black-arrow)", strokeDashArray: "0", offset: 0 }`. The `black-arrow` marker config (`constants.ts:277-283`) is already `type: "arrow", filled: false` — visual match for the v3 chevron.
2. **Transition label drops `params`** — `StateMachineDiagramEdge.tsx:124-128` only composes `name + [guard]`. v3 (`uml-state-transition-component.tsx:46-61`) builds `name [paramValues.join(", ")] [guard]`. Inspector lets users author params but they're never rendered on canvas.
3. **No declarative `updatable: false` parity for the markers** — v3 enforced `updatable: false` on `StateInitialNode`, `StateFinalNode`, `StateForkNode`, `StateForkNodeHorizontal` at the metamodel-level (`packages/editor/src/main/packages/uml-state-diagram/uml-state-initial-node/uml-state-initial-node.ts:13`, etc). v4 honours it only by way of `StateLabelEditPanel.tsx:23-28,41` short-circuiting to `null`; the `node.draggable` / `selectable` features still default `true` so toolbar actions (color picker, delete) remain reachable. Move the gate up to `node.selectable=false` or skip the toolbar for these types if strict parity is desired.

---

## A. Palette

`packages/library/lib/constants.ts:846-918` declares the StateMachineDiagram palette. Body / fallback-body are intentionally omitted — they're not draggable; they appear automatically on drop into a `State` per `bpmnConstraints.ts:96-103`.

| v4 palette type | width × height | `defaultData` | SVG preview |
|---|---|---|---|
| `State` | `DEFAULT_ELEMENT_WIDTH (160)` × 100 | `{ name: "State" }` | `StateSVG` |
| `StateInitialNode` | 45 × 45 | `{ name: "" }` | `StateInitialNodeSVG` |
| `StateFinalNode` | 45 × 45 | `{ name: "" }` | `StateFinalNodeSVG` |
| `StateActionNode` | `DEFAULT_ELEMENT_WIDTH (160)` × 50 | `{ name: "Action" }` | `StateActionNodeSVG` |
| `StateObjectNode` | `DEFAULT_ELEMENT_WIDTH (160)` × 50 | `{ name: "Object" }` | `StateObjectNodeSVG` |
| `StateMergeNode` | 80 × 80 | `{ name: "" }` | `StateMergeNodeSVG` |
| `StateForkNode` | 20 × 60 | `{ name: "" }` | `StateForkNodeSVG` |
| `StateForkNodeHorizontal` | 60 × 20 | `{ name: "" }` | `StateForkNodeHorizontalSVG` |
| `StateCodeBlock` | 200 × 150 | `{ name: "code", code: '# Sample code\\nprint("Hello World")', language: "python" }` | `StateCodeBlockSVG` |

v3 fork sizes (from `uml-state-diagram/state-preview.ts:25-99`): Initial / Final = 45×45 (matches), Fork = 20×60, ForkHorizontal = 60×20 (match), CodeBlock 150×150 (v4 = 200×150, slight upgrade), State `bounds.width/height` from constructor defaults. **`StateActionNode` width**: v3 used `calculateNameBounds`; v4 hard-codes `DEFAULT_ELEMENT_WIDTH (160)` — note the action / object SVGs draw a rounded rectangle exactly like v3 but the on-drop default ignores name length.

`StateBody` / `StateFallbackBody` are not in the palette — they're created via the inspector's body-add row in v3 (`uml-state-update.tsx:133-164`) and via dragging the matching node-type into a `State` parent in v4 (`bpmnConstraints.ts:96-103`).

## B. Canvas rendering — 11 element types

`packages/library/lib/nodes/stateMachineDiagram/State.tsx` — header with optional `«stereotype»` + `name` (`fontWeight=600`, `italic`/`underline` honoured), divider line at `LAYOUT.DEFAULT_HEADER_HEIGHT (40)` or `…WITH_STEREOTYPE (50)`, body region empty so children show through. `cornerRadius=8`. Min size 120×60. Matches v3 `uml-state-component.tsx:13-75`.

`StateBody.tsx` / `StateFallbackBody.tsx` — flat row, `<rect stroke="none">`, single `<text x={10}>` at `height/2 + 5`. Fallback body uses `fontStyle="italic"` for visual differentiation (v3 used the same `UMLStateMember` component for both). `hiddenHandles=[]` → no transition handles on body rows.

`StateActionNode.tsx` — rounded rectangle (`rx=5 ry=5`), centred `name`, `data.code` not painted (consistent with v3 — v3's component only renders `Text y={20}` + a `children` slot reserved for code-block). Resizable.

`StateObjectNode.tsx` — plain rectangle, bold centred text, label = `name` or `${name}: ${className}` if `className` set. **Cross-diagram**: inspector populates `data.classId` from the diagramBridge. v3 had no `classId` field on `UMLStateObjectNode` (`uml-state-object-node.ts:11`) — v4 adds it for parity with `ObjectName.classId`.

`StateInitialNode.tsx` — `<circle r={min(w,h)/2} stroke="none">`. Reads `data.fillColor` directly (NOT through `getCustomColorsFromData`) and falls back to `var(--besser-primary-contrast, #000000)` so the marker is always **solid black** by default, mirroring v3's `ThemedCircleContrast` behaviour. The fix-comment at `StateInitialNode.tsx:13-21` documents the SA-UX-FIX-2 incident where `getCustomColorsFromData` defaulted the fill to `--besser-background` (white) and the bullet vanished. Hidden corner handles → cardinal-only attachment.

`StateFinalNode.tsx` — outer hollow circle `r=0.45·min(w,h)`, inner solid `r=0.35·min(w,h)`, both centred. Stroke fallback `var(--besser-primary-contrast, #000000)`. Hidden corner handles.

`StateMergeNode.tsx` — diamond `<polygon>` (`width/2,0 → width,height/2 → width/2,height → 0,height/2`), bold centred label, `LINE_WIDTH=2`. Min 60×60. v3 used `ThemedPolyline` with `Multiline` text — same shape and label.

`StateForkNode.tsx` — vertical bar 20×≥60, fixed width via `NodeResizer minWidth=20 maxWidth=20`, height resizable. Reads `data.fillColor` directly with hard `#000000` fallback (PC-6 #1 fix mirroring `StateInitialNode`). Corner handles hidden, `className="vertically-not-resizable"`.

`StateForkNodeHorizontal.tsx` — mirror of above: ≥60×20, fixed height, width resizable. Same fillColor fix.

`StateCodeBlock.tsx` — header strip (`headerHeight=20`) filled with `strokeColor`, `Python`/`language` label in white, body in `<foreignObject>` with `<div>`s preserving tabs (`preserveTabs` replaces `\t` with 4 spaces). Resizable, min 150×100. Matches v3 visual at `uml-state-code-block-component.tsx:59-99`.

## C. Inspector parity

Registry in `packages/library/lib/components/inspectors/stateMachineDiagram/index.ts:24-39`. Slot mapping:

| node.type | panel | v3 popup |
|---|---|---|
| `State` | `StateEditPanel` | `UMLStateUpdate` |
| `StateBody` / `StateFallbackBody` | `StateBodyEditPanel` (CodeMirror Python) | (`null` — edited inline in `UMLStateUpdate`) |
| `StateActionNode` | `StateActionNodeEditPanel` (name + multiline `code`) | `DefaultPopup` |
| `StateObjectNode` | `StateObjectNodeEditPanel` (instance-name + class picker) | `DefaultPopup` |
| `StateCodeBlock` | `StateCodeBlockEditPanel` (CodeMirror + language `Select`) | `UMLStateCodeBlockUpdate` |
| `StateInitialNode` | `StateLabelEditPanel` → **null** (NON_UPDATABLE_TYPES) | `DefaultPopup` |
| `StateFinalNode` | `StateLabelEditPanel` → **null** | `DefaultPopup` |
| `StateMergeNode` | `StateMergeNodeEditPanel` (name + decisions list) | `UMLStateMergeNodeUpdate` |
| `StateForkNode` | `StateLabelEditPanel` → **null** | `DefaultPopup` |
| `StateForkNodeHorizontal` | `StateLabelEditPanel` → **null** | `DefaultPopup` |
| `StateTransition` | `StateMachineDiagramEdgeEditPanel` | `UMLStateTransitionUpdate` |

`updatable: false` invariant: ✅ honoured for Initial / Final / Fork / ForkHorizontal via the `NON_UPDATABLE_TYPES` set in `StateLabelEditPanel.tsx:23-28,41`. The popover still routes here, but the body returns `null` so the inspector collapses. Note v3 actually wired these to `DefaultPopup` not `null`, contradicting the metamodel `updatable: false` flag — v4 is *more correct* than v3 here.

`StateMergeNodeEditPanel` — implements the v3 decisions editor parity (PC-6 #3): walks `useDiagramStore(s => s.edges).filter(e => e.source === node.id)` and renders one row per outgoing edge with editable `name`, `→` arrow, target `Select` (sourced from `nodes`), and a delete button. v3 source: `uml-state-merge-node-update.tsx:128-148` — same structure. v3 also had `width`/`height` size inputs (`uml-state-merge-node-update.tsx:103-124`) which v4 omits in favour of `NodeResizer` handles.

`StateEditPanel` — name field, stereotype `Select` (v4 set: `none`, `initial`, `final`, `decision`, `fork`, `merge`), `italic`/`underline` checkboxes, color editor via `NodeStyleEditor`. v3 (`uml-state-update.tsx`) additionally rendered the bodies / fallback-bodies sections — those are now per-row child panels in v4.

`StateBodyEditPanel` — name + `kind` dropdown (`entry` / `do` / `exit` / `transition`) + CodeMirror Python `code` field. v3 had no per-body inspector (popup was `null`); v4 adds these BESSER fields and the v3↔v4 round-trip preserves them via `e.code` / `e.kind` (`versionConverter.ts:1000-1006`, `2681-2685`).

`StateActionNodeEditPanel` — name + multiline `code` field. v3 was `DefaultPopup` (name + style only); v4 promotes `code` to a first-class field.

`StateObjectNodeEditPanel` — instance name + `classId` picker. The class list comes from `diagramBridge.getAvailableClasses()` (`StateObjectNodeEditPanel.tsx:33-39`). On select, both `classId` and the cached `className` are written so the canvas can render `name: ClassName` without re-querying the bridge. v3 had no class picker.

`StateCodeBlockEditPanel` — language `Select` (`python`, `bal`) + CodeMirror Python `code` field. v3 used a styled `<textarea>` with manual tab-key handling (`uml-state-code-block-update.tsx:78-104`); v4 delegates tab handling to CodeMirror.

## D. Transition edge

`packages/library/lib/edges/edgeTypes/StateMachineDiagramEdge.tsx`. Type registered via side-effect `registerEdgeTypes({ StateTransition: StateMachineDiagramEdge })` at line 227. `edges/types.tsx:196-199` declares `{ allowMidpointDragging: true, showRelationshipLabels: true }`.

**Marker — DEFECT.** `getEdgeMarkers` (`packages/library/lib/utils/edgeUtils.ts:174-380`) has no `case "StateTransition"`; falls through to the default at lines 381-385 returning `{ markerPadding, strokeDashArray: "0", offset: 0 }` — no `markerEnd`. The component's `<EdgeInlineMarkers markerEnd={markerEnd} markerStart={markerStart} />` (line 152-158) therefore renders nothing. v3 painted an inline `M0,29 L30,15 L0,1` SVG path (`uml-state-transition-component.tsx:65-76`) — an open chevron. The closest match in v4 is `url(#black-arrow)` (`constants.ts:277-283`, `type: "arrow", filled: false` → outlined V-shape with round linecap). The fix is one switch-case in `edgeUtils.ts`.

**Fields parity:**

| field | v3 source | v4 storage |
|---|---|---|
| `name` | `UMLStateTransition.name` (inherited) | `edge.data.name` |
| `code` | `code` field on relationship (custom) | `edge.data.code` (CodeMirror Python in panel) |
| `eventName` | `eventName` field (custom) | `edge.data.eventName` |
| `kind` | not on edge in v3 (was on body element) | absent on v4 edge |
| `guard` | `IUMLStateTransition.guard` | `edge.data.guard` |
| `params` | `IUMLStateTransition.params: Record<string, string>` | `edge.data.params` |

The brief asks for `kind`. v3's transition didn't carry `kind`; the `kind` field is on the body element (`StateBodyEditPanel`'s `entry` / `do` / `exit` dropdown). If "transition kind" is the intended new field, it's missing — but it isn't a parity gap with v3.

**Flip action**: `handleSwap` at `StateMachineDiagramEdgeEditPanel.tsx:76-89` swaps `source/target` + `sourceHandle/targetHandle` on the edge — equivalent to v3's `UMLRelationshipRepository.flip` (`uml-state-transition-update.tsx:131-133`). Surfaced as `SwapHorizIcon` next to the color editor.

**Color editor**: `EdgeStyleEditor` is rendered at the top of the panel (`StateMachineDiagramEdgeEditPanel.tsx:111-123`) with stroke + text color, mirroring v3's `StylePane` with `lineColor textColor` (`uml-state-transition-update.tsx:172-178`).

**CodeMirror Python on `code`**: yes (`StateMachineDiagramEdgeEditPanel.tsx:162-173`), `extensions={[python()]}`.

**Label composition** (`StateMachineDiagramEdge.tsx:124-128`): `[name, guard ? \`[guard]\` : ""].filter(Boolean).join(" ")`. v3 (`uml-state-transition-component.tsx:46-61`) renders `name + [paramValues.join(", ")] + [guard]`. **`params` is missing from the v4 label.**

## E. Parent-child wiring

`isParentNodeType` (`packages/library/lib/utils/nodeUtils.ts:230-259`): includes `"State"` in the explicit list of BESSER-registered parent types. ✅

`canDropIntoParent` (`packages/library/lib/utils/bpmnConstraints.ts:96-103`):
```
if (parentType === "State") {
  return childType === "StateBody" || childType === "StateFallbackBody" || childType === "StateCodeBlock"
}
```
✅ Allows the three v3 child kinds. `StateActionNode`, `StateObjectNode`, marker nodes are explicitly NOT droppable into a `State` (matches v3 — those weren't `UMLStateMember` subclasses).

The child nodes themselves (`StateBody.tsx`, `StateFallbackBody.tsx`, `StateCodeBlock.tsx`) read `parentId` from React Flow node-props. `useHandleOnResize(parentId)` propagates resize to a parent if any. The `State.tsx` parent does not paint dividers between body and fallback-body regions (v3 painted divider at `deviderPosition` via `ThemedPath` in `uml-state-component.tsx:71-73`) — only one divider after the header. **Minor visual divergence**: with both body and fallback-body children, v3 drew an extra horizontal line at the boundary; v4 relies on the child rows' own bounding rectangles for separation.

## F. Migrator

`migrateStateMachineDiagramV3ToV4` (`packages/library/lib/utils/versionConverter.ts:2146-2156`) is a thin wrapper around `convertV3ToV4` with a type guard.

Forward direction (v3 → v4):
- Element types pass through verbatim via `convertV3NodeTypeToV4` (`versionConverter.ts:308-318`). PascalCase → PascalCase.
- `State` body collapse: per the SA-3 brief, `StateBody` / `StateFallbackBody` are kept as separate React-Flow nodes with `parentId`, NOT collapsed onto `State.data.bodies` (the brief overrides `uml-v4-shape.md`'s collapse recommendation — see comment at lines 2136-2144). v3's `owner` field maps directly to v4's `parentId`.
- `code` / `kind` BESSER editor extensions on body elements pass through (`versionConverter.ts:1000-1006`).
- `StateObjectNode.classId` / `className` preserved (`versionConverter.ts:1019-1029`).
- `StateCodeBlock.code` and `language` (default `python`) preserved.

Reverse direction (v4 → v3) at `convertV4ToV3StateMachine` (`versionConverter.ts:2610-2788`):
- Pre-indexes children by `parentId` into `{ bodies: [], fallbackBodies: [] }` slots so the `State` row can re-emit `bodies: string[]` and `fallbackBodies: string[]` arrays alongside `hasBody` / `hasFallbackBody` booleans. Order is preserved from React Flow's child traversal order.
- Edge data round-trips: `name`, `guard`, `params` (always emitted as object dict — v3 deserializer accepts string / array / object), `code`, `eventName`. ✅ All BESSER additions preserved.
- `path` rebuilt as `points.map(p => ({ x: p.x - minX, y: p.y - minY }))` with the bounding-box `(minX, minY)` written to `bounds`.

Round-trip property: forward then reverse should be information-equivalent (v3 → v4 → v3' where v3' is canonicalised). Tests for this should live in `packages/library/lib/utils/__tests__/versionConverter.state.spec.ts` (not verified here — read-only audit).

## G. Cross-diagram (StateObjectNode ↔ ClassDiagram)

`StateObjectNodeEditPanel.tsx:33-39` calls `diagramBridge.getAvailableClasses()` inside a `useMemo` keyed on `nodes`. The return type is `IClassInfo[]` (`packages/library/lib/services/diagramBridge.ts:82,211`). Selection writes both `classId` and the cached `className` so the canvas label `${name}: ${className}` doesn't require live bridge resolution on every render (`StateObjectNode.tsx:36-37`).

The bridge is a singleton (`diagramBridge.ts:568`), populated by webapp-side glue via `setClassDiagramData(...)`. v3 did not have this picker — the round-trip through `versionConverter` simply preserves whatever `classId`/`className` is on the v3 element (lines 1019-1029, 2700-2706).

---

## Cross-references

- v3 source: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/`
- v4 nodes: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/`
- v4 SVGs: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/svgs/nodes/stateMachineDiagram/StateMachineSVGs.tsx`
- v4 inspectors: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/stateMachineDiagram/`
- v4 edge: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/edges/edgeTypes/StateMachineDiagramEdge.tsx`
- v4 palette: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/constants.ts:846-918`
- v4 migrator: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/versionConverter.ts:2146-2156` (forward), `:2594-2788` (reverse)
- Edge marker resolution: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/edgeUtils.ts:174-385`
- Parent/child rules: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/bpmnConstraints.ts:82-103`, `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/nodeUtils.ts:230-259`
- Existing per-component audits: `docs/source/migrations/per-component/PC-5-state-bodies-edge.md`, `PC-6-state-markers.md`
