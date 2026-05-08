# Deep audit — parent/child node interactions (v3 → v4)

Read-only, evidence-based audit of the three BESSER-only "container" parent
shapes the v4 canvas must honour: `NNContainer ⊃ NN layers`, `State ⊃
StateBody|StateFallbackBody|StateCodeBlock`, and `AgentIntent ⊃
AgentIntentBody|AgentIntentDescription|AgentIntentObjectComponent`. Each is
compared against its v3 metamodel source under
`besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/`.

All paths are absolute from the repo root.

## Verdict

The drop-time wiring is **correct in shape** — `isParentNodeType`
recognises all four custom parents, `canDropIntoParent` enforces the
right whitelist for each, and both the palette ghost
(`DraggableGhost.tsx`) and the canvas drag-stop (`useNodeDragStop.ts`)
funnel through these two predicates so a layer dropped on
`NNContainer` (or a body dropped on `State` / `AgentIntent`) is
correctly assigned `parentId`. React-Flow's runtime handles the rest:
parent drag carries children, parent delete cascades to children
(default RF behaviour), and `resizeAllParents` grows ancestors when a
child overflows.

**Critical gap**: the runtime predicates are not covered by unit tests —
`packages/library/tests/unit/bpmnConstraints.test.ts:1-249` only exercises
BPMN, package, activity, useCaseSystem, componentSubsystem, and
deployment parents; **no test asserts that NN datasets are rejected
inside `NNContainer`, that `State` accepts only its three body kinds,
or that `AgentIntent` accepts only its three child kinds**. Likewise
`packages/library/tests/unit/nodeUtils.test.ts:347-360` lists
`isParentNodeType` cases for the v3-style parents only — `NNContainer`,
`State`, `AgentState`, `AgentIntent` are missing. A regression that
flips one of those branches (e.g. removing the literal-string compare
during a future enum widening) would silently downgrade the drop UX
without breaking CI.

Two other concrete defects (detailed below):

1. `AgentIntent` (v3 `AgentElementType.AgentIntent`) declared
   `features.droppable: false` — drops were programmatically blocked.
   v4 treats it as droppable via `isParentNodeType` and
   `canDropIntoParent`; that is the **intentional** divergence (v3
   spawned bodies via the inspector's "+ Add body" row, v4 wants
   palette-drop parity), but **no migration note exists** so the
   round-tripper does not emit `features.droppable: false` either way.
2. `useNodeDragStop` (and `DraggableGhost`) call `resizeAllParents`
   only when a child is added or moved **into** a parent — a child
   that is already inside a parent and is dragged toward the parent's
   edge does grow the parent, but the parent's *origin* never shrinks
   when the child is dragged inward. v3's `UMLContainer.render`
   recomputed bounds from the child set every layout pass, so the v3
   container collapsed back to its minimum after a child shrank; v4
   leaves stale dead-space until manual resize.

## Top five issues

1. **No unit-test coverage for the three v4 BESSER parents**. Add
   `describe("canDropIntoParent – NNContainer / State / AgentIntent /
   AgentState")` tests in
   `packages/library/tests/unit/bpmnConstraints.test.ts` mirroring the
   BPMN style, plus `isParentNodeType("NNContainer" | "State" |
   "AgentState" | "AgentIntent")` cases in `nodeUtils.test.ts`.
2. **NN dataset constraint is single-source-of-truth at risk**.
   `NN_LAYER_KINDS_IN_CONTAINER` in `bpmnConstraints.ts:60-76`
   hard-codes the 14 allowed layer kinds; the palette in
   `lib/constants.ts:1033-1188` is the parallel source. A new layer
   kind added to the palette and forgotten here would drop straight
   through to the canvas root with no `parentId`. Same for State /
   AgentIntent: their child-type whitelists are literal-string
   conjunctions in `bpmnConstraints.ts:97-114`. Fix: derive both lists
   from a shared registry exported from `lib/constants.ts`.
3. **Parent does not shrink when children move inward**. `resizeAllParents`
   only grows. v3 `UMLContainer.render` recomputed bounds from the
   child bounding box (`computeBoundingBoxForElements`,
   `nn-container.ts:67-82`) every layout pass. v4 leaves dead-space
   until manual resize. A symmetric `recomputeParentMinBounds(node,
   nodes)` helper called on `onNodeDragStop` would close the gap.
4. **Cascade-delete of parent → children is implicit, not enforced**.
   `App.tsx:148` wires `onBeforeDelete` from `useElementInteractions`,
   but that hook (`useElementInteractions.ts:30-32`) only gates on
   `isDiagramModifiable` — nothing inspects the parent/child graph.
   Cascade therefore relies entirely on React-Flow's default
   `applyNodeChanges` semantics: when a parent is removed, its
   children with `parentId === parent.id` become **orphaned at the
   canvas root**, not deleted. v3 `UMLContainer.delete` recursively
   removed `ownedElements`. Add an `onBeforeDelete` step that, for
   `NNContainer | State | AgentIntent`, expands the delete set to
   include `nodes.filter(n => n.parentId === doomed.id)`.
5. **`AgentState` is treated as a parent type but rejects every drop**.
   `bpmnConstraints.ts:117-121` returns `false` unconditionally when
   `parentType === "AgentState"`, **and** `isParentNodeType("AgentState")
   === true` (`nodeUtils.ts:256`). The combination is intentional per
   the comment ("bodies are inlined, not nested") but the resulting
   UX is a parent-shaped highlight on hover that silently rejects the
   drop. Either drop `AgentState` from `isParentNodeType` (cleaner —
   the inline-body design needs no React-Flow parent recognition) or
   surface a "use the inspector +Add body affordance" toast on the
   rejected drop.

---

## Per-parent matrix

For each of the four parents, the eight audit dimensions:

| Dimension | `NNContainer` | `State` | `AgentIntent` | `AgentState` |
|---|---|---|---|---|
| `isParentNodeType(type)` recognises | yes (`nodeUtils.ts:254`) | yes (`nodeUtils.ts:255`) | yes (`nodeUtils.ts:257`) | yes (`nodeUtils.ts:256`) |
| `canDropIntoParent(parent, child)` rule | whitelist of 14 layer kinds + `NNReference` (`bpmnConstraints.ts:60-92`) | whitelist `StateBody / StateFallbackBody / StateCodeBlock` (`bpmnConstraints.ts:97-103`) | whitelist `AgentIntentBody / AgentIntentDescription / AgentIntentObjectComponent` (`bpmnConstraints.ts:108-114`) | always returns `false` (`bpmnConstraints.ts:117-121`) |
| `parentId` propagation on palette drop | yes — `DraggableGhost.tsx:122-138, 155-159, 169` writes `parentId` and translates position to parent-local | yes (same path) | yes (same path) | n/a — `canDropIntoParent` returns `false`, so `parentId` is never set |
| `parentId` propagation on canvas drag | yes — `useNodeDragStop.ts:49-108` — `getIntersectingNodes` ∩ `isParentNodeType` ∩ `canDropIntoParent`, then re-projects to parent-local coords and calls `resizeAllParents` | yes (same path) | yes (same path) | n/a — drop rejected, child remains at canvas root |
| Resize: parent grows on overflow | yes — `resizeAllParents` (`nodeUtils.ts:27-60`) extends width/height when child crosses right/bottom edge, and shifts origin + width/height when child crosses left/top edge | yes (same helper) | yes (same helper) | n/a |
| Resize: parent shrinks when child moves inward | **no** — `resizeAllParents` only grows. v3 container `render()` recomputed from child bounding box every layout pass | **no** | **no** | n/a |
| Drag parent → moves children | yes — React-Flow's native behaviour: the child's `position` is parent-local so children render with parent's translation automatically. `getPositionOnCanvas(node, allNodes)` (`nodeUtils.ts:5-25`) walks the parent chain to compute absolute coordinates when needed | yes (same RF default) | yes (same RF default) | n/a |
| Delete parent → cascades to children | **partial** — React-Flow default leaves children orphaned at canvas root. `App.tsx:148` `onBeforeDelete` only gates on diagram-modifiable; no expansion of delete set. Children survive with stale `parentId` (which then points at a missing node — `getPositionOnCanvas` handles that gracefully via the `nodeMap.get` warning at `nodeUtils.ts:82-85`) | partial (same) | partial (same) | n/a — bodies live on `data.bodies`, not as nodes, so the issue is moot for AgentState |
| v3 reference (metamodel `features.droppable`) | `nn-container.ts:13-17` — `droppable: true, resizable: true` | `uml-state.ts:26-30` — `droppable: false, resizable: 'WIDTH'` | `agent-intent.ts:24-28` — `droppable: false, resizable: 'WIDTH'` | (sibling — `agent-state-diagram/agent-state/`) v3 carried bodies via `serialize`/`render`; v4 inlines onto `data.bodies` |
| Constraint correctness vs v3 | dataset/configuration excluded from container — matches v3 `NNAssociation` design | matches v3: only the three child kinds are visually placed inside the state | matches v3 visual: bodies / description / object-component nest under intent. **`droppable:false` v3 metamodel flag not surfaced** in v4 (intentional: v4 wants drop UX) | intentional inversion: v4 forbids drops because bodies are inline |

---

## A. `NNContainer` — `bpmnConstraints.ts:90-92`, `nn-container.ts`

**Allowed children** (15 entries):
`Conv1DLayer`, `Conv2DLayer`, `Conv3DLayer`, `PoolingLayer`, `RNNLayer`,
`LSTMLayer`, `GRULayer`, `LinearLayer`, `FlattenLayer`, `EmbeddingLayer`,
`DropoutLayer`, `LayerNormalizationLayer`, `BatchNormalizationLayer`,
`TensorOp`, `NNReference`.

**Excluded** (correct per the audit brief): `TrainingDataset`,
`TestDataset`, `Configuration`. Verified by reading both
`bpmnConstraints.ts:60-76` (the predicate) and the palette at
`lib/constants.ts:1167-1188` — the three excluded entries appear in the
palette but never in the whitelist, so dragging a dataset over an
`NNContainer` skips the parent assignment in
`useNodeDragStop.ts:49-62`. They bind to the container via
`NNAssociation` edges instead (see
`packages/library/lib/edges/edgeTypes/NNAssociation.tsx`).

**v3 reference**:
`packages/editor/src/main/packages/nn-diagram/nn-container/nn-container.ts:13-17`
declares `static features = { ...UMLContainer.features, resizable:
true, droppable: true }`. The v3 `render()` (`nn-container.ts:58-93`)
recomputed `bounds` from `computeBoundingBoxForElements([{ bounds:
calculatedNamedBounds }, ...absoluteElements])` so the container could
both grow and shrink to its child set every layout pass. **v4 has no
shrink path** — see issue #3.

**Resize bounds**: v4 enforces `minWidth=200, minHeight=140` on the
visual `NodeResizer` (`NNContainer.tsx:48-51`), matching v3's
`minWidth=200, minHeight=150` (off by 10px on min height — minor).

---

## B. `State` — `bpmnConstraints.ts:97-103`, `uml-state.ts`

**Allowed children**: `StateBody`, `StateFallbackBody`, `StateCodeBlock`.

**v3 reference**: `uml-state.ts:25-30` declares `droppable: false,
resizable: 'WIDTH'`. v3 spawned bodies through the inspector ("+ Add
body" rows in `uml-state-update.tsx`) — they were never palette-drop
targets. v4 deliberately accepts drops to reach palette-drop parity
(`StateCodeBlock` is in the palette at `lib/constants.ts:846-918`
according to `deep-state-analysis.md`). The runtime predicate fields
the bodies correctly.

**Render contract**: `State.tsx:106-113` draws a header divider at
`headerHeight`. The body region below the divider is empty so children
(rendered as separate React-Flow nodes via `parentId`) show through.
This mirrors v3's render at
`uml-state-component.tsx` which laid bodies out at `body.bounds.y = y;
y += body.bounds.height` (`uml-state.ts:91-103`).

**Header height feature parity**: v3 had `stereotypeHeaderHeight = 50`
and `nonStereotypeHeaderHeight = 40` (`uml-state.ts:31-32`). v4
reproduces via `LAYOUT.DEFAULT_HEADER_HEIGHT_WITH_STEREOTYPE` /
`LAYOUT.DEFAULT_HEADER_HEIGHT` (`State.tsx:36-38`). Match.

**Resize bounds**: v4 `minWidth=120, minHeight=60` (`State.tsx:46-48`).
v3 had no explicit min — it relied on `Text.size` measuring the name
to grow `bounds.width`. Acceptable divergence.

---

## C. `AgentIntent` — `bpmnConstraints.ts:108-114`, `agent-intent.ts`

**Allowed children**: `AgentIntentBody`, `AgentIntentDescription`,
`AgentIntentObjectComponent`.

**v3 reference**: `agent-intent.ts:23-30` (the file is named
`agent-intent.ts` inside `agent-intent-object-component/`, slightly
confusing). Same `droppable: false, resizable: 'WIDTH'` story as
`UMLState`. v4 treats it as droppable to keep drop UX uniform across
parents.

**Render contract**: `AgentIntent.tsx:67-148` draws the v3
folded-corner intent rectangle (`d="M 0 0 H ${width} V ${height} H 30
L 0 ${height + 30} L 10 ${height} H 10 0 Z"`) plus two divider lines —
the header divider at `headerHeight` and a second divider at
`headerHeight + AGENT_INTENT_DESCRIPTION_HEIGHT (30)` when both
description and at least one body row exist. v3 source for the second
divider: `agent-intent-object-component.tsx:115-119`.

**Visual subtlety**: v3 mutated `element.name = "Intent: " + this.name`
inside `render()` — a known v3 bug (the prefix accumulated on
re-renders if not reset). v4 emits the prefix at render time only
(`AgentIntent.tsx:95, 109`), keeping `data.name` clean. Confirmed
no-op for the round-tripper because the v3-side mutation happened
post-deserialize, not pre-serialize.

---

## D. `AgentState` — `bpmnConstraints.ts:117-121`, `nodeUtils.ts:256`

**Allowed children**: none. `canDropIntoParent("...", "AgentState") ===
false` for every child type.

**Why it's still in `isParentNodeType`**: the comment at
`bpmnConstraints.ts:116-121` explains the design — without the rule,
the default `return true` (`bpmnConstraints.ts:215`) would silently
accept drops, breaking the inline-body invariant
(`diagramStore.ts:34-57` — `dropFloatingAgentBodies` defends the
*storage* layer; this rule defends the *drop* layer). Adding
`AgentState` to `isParentNodeType` lets the drop-handler match the
parent and then explicitly reject — without the recognition, the
fallback path would have created a `parentId = AgentState.id`
relationship, leading the floating-body guard to silently delete the
body next time `setNodes` ran.

This is a **defensible** but **surprising** UX. The hover highlight
suggests "I will accept this drop" then nothing happens. See issue #5
for the suggested follow-up.

---

## E. Shared infrastructure

**Drop-time wiring**:

- `DraggableGhost.tsx:122-138` — palette drop. `getIntersectingNodes`
  → filter by `isParentNodeType` ∩ `canDropIntoParent`, take the
  last (deepest) intersection as the parent, project position to
  parent-local coords (`DraggableGhost.tsx:155-159`), then write
  `parentId` on the new node (`DraggableGhost.tsx:169`).
- `useNodeDragStop.ts:49-108` — canvas drag. Same predicate pair, plus
  a "is this a *new* parent" branch (`useNodeDragStop.ts:84-108`) that
  re-projects coordinates and calls `resizeAllParents`.

**Resize wiring**: `resizeAllParents` (`nodeUtils.ts:27-60`) walks
ancestors via `node.parentId`, growing each parent's
width/height and shifting its origin + sibling positions when the
child crosses the negative edge. Tested at
`packages/library/tests/unit/nodeUtils.test.ts:150-279`. Symmetric
shrink is missing.

**Drag-children wiring**: React-Flow's default — child positions are
stored parent-local, parent translation is applied at render. No
custom code needed. `getPositionOnCanvas` (`nodeUtils.ts:5-25`)
walks the chain to recover the canvas-absolute coordinate when an
event handler needs it (drop-on-itself avoidance, intersection tests).

**Delete wiring**: `App.tsx:117-156` mounts `<ReactFlow>` with no
`onNodesDelete` handler, only `onBeforeDelete` (which is a permission
gate, `useElementInteractions.ts:30-32`). `diagramStore`'s
`onNodesChange.remove` branch (`diagramStore.ts:469-489`) deletes
**only the explicitly removed node** plus its connected edges (via
`getConnectedEdges`). **Children are not expanded.** React-Flow's
own `applyNodeChanges` orphans them.

**Topological sort**: `sortNodesTopologically` (`nodeUtils.ts:62-101`)
ensures parents appear before children in the array — required by
React-Flow for correct rendering order. Used after every drop
(`useNodeDragStop.ts:100, 113`).

---

## F. Recommended follow-up

1. Add unit tests covering all four BESSER parents to
   `bpmnConstraints.test.ts` and `nodeUtils.test.ts`.
2. Extract `NN_LAYER_KINDS_IN_CONTAINER`, `STATE_CHILD_KINDS`,
   `AGENT_INTENT_CHILD_KINDS` to a single registry exported from
   `lib/nodes/types.ts` so palette and predicate share their source.
3. Add `recomputeParentMinBounds` to `useNodeDragStop` so parents
   shrink symmetrically.
4. Extend `onBeforeDelete` to expand the doomed-set with
   `nodes.filter(n => n.parentId === doomed.id)` for the four parent
   types — closes the orphan-children leak. Alternatively, hook
   `onNodesDelete` and run the same expansion there.
5. Consider removing `AgentState` from `isParentNodeType` and instead
   letting the storage-layer guard (`dropFloatingAgentBodies`) catch
   the misuse. The hover-but-reject UX is the worst of both worlds.

---

*Audited: 2026-05-08. Branch: `claude/refine-local-plan-sS9Zv`.*
