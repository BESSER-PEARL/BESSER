# A5 — Cross-diagram Bridge Correctness

Final-analysis wave audit of `packages/library/lib/services/diagramBridge.ts`
in the BESSER WME v4 (React-Flow) port.

Branch: `claude/refine-local-plan-sS9Zv`. Submodule SHA pinned at
`7a3b82a403583ec7a3a9927e32886a3fde0baf02`
(`feature/migration-react-flow`). All file:line references below are
relative to that SHA — read with
`git -C besser/utilities/web_modeling_editor/frontend show 7a3b82a:<path>`.

---

## 1. Bridge methods — v4-shape walks

| Method | v4 walk | Evidence |
| --- | --- | --- |
| `getAvailableClasses()` | OK | `diagramBridge.ts:208` filters `data.nodes`; `isClassNode` keys on `node.type === "class"` (v4) with v3 stereotype tolerance (`diagramBridge.ts:91-99`). Maps `node.data.{name,icon}`, `node.data.attributes` (array shape `{id,name,attributeType,visibility,defaultValue}` — `diagramBridge.ts:255-263`). |
| `getRelatedClasses(classId)` | OK | `diagramBridge.ts:158-198`: walks `data.edges` using `rel.source` / `rel.target` (React-Flow scalar IDs, not v3 `{element}` objects); filters by `rel.type !== STEREOTYPE_INHERITANCE`. Recurses via inheritance edges (`source = child`, `target = parent`, matching v3 semantics). |
| `getAvailableAssociations(srcId, tgtId)` | OK | `diagramBridge.ts:307-360`: builds source / target hierarchy sets, then walks `data.edges`, skipping inheritance / realization. Reads association metadata from `edge.data.{name,sourceRole,sourceMultiplicity,targetRole,targetMultiplicity}` (v4 shape — v3 stored these on `relationship.source` / `relationship.target` sub-objects). |
| `getClassHierarchy(classId)` | OK | `diagramBridge.ts:486-520`: walks `data.nodes` to find the seed class, then `data.edges` filtered on `STEREOTYPE_INHERITANCE` with `rel.source === currentClassId`. Pushes `currentClass.data.name` into the chain. |
| `getStateMachineDiagrams()` | OK (cache-only) | `diagramBridge.ts:524-526` returns the `IDiagramReference[]` array fed in via `setStateMachineDiagrams`. No node walk — population is the embedder's responsibility (`ApollonEditorComponent.tsx:84`). |
| `getQuantumCircuitDiagrams()` | OK (cache-only) | `diagramBridge.ts:539-541` — symmetric with state-machine handling, fed via `setQuantumCircuitDiagrams` (`ApollonEditorComponent.tsx:85`). |

Auxiliary observations:

- `localStorage` fallback in `getClassDiagramData()` tolerates legacy v3
  caches (`elements` / `relationships` dicts → `nodes` / `edges`
  arrays) — `diagramBridge.ts:142-149`. Round-tripped data stays v4
  shape after the first `setClassDiagramData()`.
- `getAllClassesInHierarchy()` (private helper for
  `getAvailableAssociations`) walks both directions of inheritance —
  `diagramBridge.ts:368-410`, which matches v3 behaviour.

No method falls back to `model.elements` / `model.relationships`
anywhere. **All six methods walk v4 ✅**.

---

## 2. Per-consumer correctness

| Consumer | Bridge call | Evidence | Status |
| --- | --- | --- | --- |
| `ObjectName.classId` (Object → Class) | `diagramBridge.getAvailableClasses()` for the picker; `diagramBridge.getClassDiagramData()` directly when collecting sibling enumerations. | `ObjectEditPanel.tsx:382` (picker), `:337` (enumerations), `:435-471` (commit handler). | ✅ |
| `StateObjectNode.classId` (StateMachine → Class) | `diagramBridge.getAvailableClasses()` for the picker. | `StateObjectNodeEditPanel.tsx:35`. Display label composed from `name` + cached `className` in `StateObjectNode.tsx:36-37`. | ✅ |
| `UserModelName.classId` (User → Class) | Walks `diagramBridge.getClassDiagramData().nodes[*].data.attributes` to resolve linked class attribute / enumeration literals. | `UserModelNameEditPanel.tsx:79-92` (linked attribute), `:100-122` (enumeration literals). The v3 `elements[id]` keyed lookup is correctly replaced with a `nodes` array walk. | ✅ |
| Class methods `implementationType: 'state_machine'` → `stateMachineId` | `diagramBridge.getStateMachineDiagrams()` populates the dropdown. | `ClassEditPanel.tsx:738`, dropdown at `:566-583`. Bridge population path: `ApollonEditorComponent.tsx:84`. | ✅ |
| Class methods `implementationType: 'quantum_circuit'` → `quantumCircuitId` | `diagramBridge.getQuantumCircuitDiagrams()` populates the dropdown. | `ClassEditPanel.tsx:739`, dropdown at `:584-601`. Bridge population: `ApollonEditorComponent.tsx:85`. | ✅ |
| `ObjectLink.associationId` (Object → Class association) | `diagramBridge.getAvailableAssociations(sourceClassId, targetClassId)` after walking each ObjectName end to resolve `classId`. | `ObjectLinkEditPanel.tsx:53-71`, name display via `getRelationshipDisplayName` at `:122` and `:169`. | ✅ |
| AgentDiagram RAG element → dataset names | **Not bridge-mediated** — `nodes.filter(n => n.type === 'AgentRagElement')` walks the same diagram's nodes. | `AgentStateBodyEditPanel.tsx:72`, `AgentStateEditPanel.tsx:144`. | ✅ (intentional — sibling-node reference, no cross-diagram bridge needed) |

**All seven consumers correctly call the bridge ✅**.

---

## 3. Resolution-flow trace — `ObjectName.classId`

### Fixture (hand-authored v4 shape)

```ts
// ── ClassDiagram model ──
const classDiagram: UMLModel = {
  id: 'cd-1',
  type: 'ClassDiagram',
  nodes: [
    {
      id: 'cls-customer',
      type: 'class',
      data: {
        name: 'Customer',
        stereotype: null,
        attributes: [
          { id: 'attr-name',  name: 'name',  attributeType: 'str', visibility: 'public' },
          { id: 'attr-email', name: 'email', attributeType: 'str', visibility: 'public' },
        ],
        methods: [],
      },
      position: { x: 0, y: 0 },
    },
    {
      id: 'cls-person',
      type: 'class',
      data: {
        name: 'Person',
        stereotype: null,
        attributes: [
          { id: 'attr-age', name: 'age', attributeType: 'int', visibility: 'public' },
        ],
        methods: [],
      },
      position: { x: 0, y: 200 },
    },
  ],
  edges: [
    {
      id: 'inh-1',
      type: 'ClassInheritance',
      source: 'cls-customer',  // child
      target: 'cls-person',    // parent
      data: {},
    },
  ],
}

// ── ObjectDiagram model ──
const objectDiagram: UMLModel = {
  id: 'od-1',
  type: 'ObjectDiagram',
  references: { ClassDiagram: 'cd-1' },
  nodes: [
    {
      id: 'obj-1',
      type: 'objectName',
      data: {
        name: 'aliceInstance',
        classId: 'cls-customer',     // ← cross-diagram link under audit
        className: 'Customer',
        attributes: [],
      },
      position: { x: 0, y: 0 },
    },
  ],
  edges: [],
}
```

### Step-by-step resolution

1. **Project load.** `loadProjectThunk` resolves the active diagram
   (`objectDiagram`). Because its type is `'ObjectDiagram'`,
   `setupBridgeForActiveDiagram` (`workspaceSlice.ts:124-150`)
   resolves `references.ClassDiagram === 'cd-1'`, finds
   `classDiagram`, and calls
   `diagramBridge.setClassDiagramData(classDiagram)` —
   `workspaceSlice.ts:135`. Cache also persisted to
   `localStorage["besser-class-diagram-bridge-data"]`.
2. **Tab mounts.** `DiagramTabs` mirrors the same call when the
   ClassDiagram reference changes
   (`DiagramTabs.tsx:147-160`).
3. **User opens `obj-1` inspector.** `ObjectEditPanel` runs
   `diagramBridge.getAvailableClasses()` (`ObjectEditPanel.tsx:382`):
   - `getClassDiagramData()` returns `classDiagram` from the
     in-memory cache.
   - `getAvailableClasses()` filters `data.nodes` via `isClassNode`
     (matches `type === 'class'` for both) →
     `[{ id: 'cls-customer', ... }, { id: 'cls-person', ... }]`.
   - For `cls-customer`, `getAllAttributesWithInheritance` walks the
     inheritance chain: starts at `cls-customer`, finds
     `inh-1` with `source === cls-customer` → recurses into
     `cls-person`. Result attributes (parent-first):
     `[age, name, email]` (deduped on `id`,
     `diagramBridge.ts:288-300`).
4. **Inspector renders dropdown.** `availableClasses.find(c => c.id === nodeData.classId)`
   matches `cls-customer`; the picker shows `Customer` selected and
   the linked-class attribute list `[age, name, email]` for the
   per-row `attributeId` selector
   (`ObjectEditPanel.tsx:402-407`).
5. **User commits a class change.** `handleClassChange`
   (`ObjectEditPanel.tsx:423-471`) writes `classId`, mirrors
   `className`, and rebuilds `attributes` from the bridge result.
   The display label `aliceInstance: Customer` then renders via
   the `name` field on the node (the SVG itself only renders
   `data.name` at `ObjectNameSVG.tsx:80-86`; the inspector pins the
   composite name on commit).

The flow is end-to-end consistent with v3 once you substitute
`elements`/`relationships` for `nodes`/`edges` and the v3
nested-object roles for the v4 flat `edge.data.*` shape.

---

## 4. workspaceSlice → bridge sync

`workspaceSlice.ts` only pushes to the bridge in two places:

- `loadProjectThunk` (line `165`) — initial project load.
- `switchDiagramTypeThunk` (line `213`) — when the user switches the
  active diagram type (e.g. ClassDiagram → ObjectDiagram).

Both go through the helper `setupBridgeForActiveDiagram`
(`workspaceSlice.ts:124-150`), which only fires
`diagramBridge.setClassDiagramData(...)` when the **target** type is
`ObjectDiagram` or `UserDiagram`. `DiagramTabs.tsx:147-160` adds a third
trigger: when the user changes the ObjectDiagram's
`references.ClassDiagram` selector.

### Misses

**Issue 1 — `updateDiagramModelThunk.fulfilled` does not refresh the
bridge.** `workspaceSlice.ts:686-694` updates Redux state and persists
the new model, but never calls `diagramBridge.setClassDiagramData(...)`.
Concretely:

- User on ObjectDiagram (bridge populated with sibling ClassDiagram).
- User switches to the referenced ClassDiagram, adds an `Enumeration`
  class or a new attribute, saves.
- `ApollonEditorComponent.subscribeToModelChange` fires
  `updateDiagramModelThunk` (`ApollonEditorComponent.tsx:129-134`),
  which saves to storage but **does not** notify the bridge.
- User switches back to the ObjectDiagram. The bridge cache still
  holds the pre-edit ClassDiagram until `switchDiagramTypeThunk`
  runs `setupBridgeForActiveDiagram` — that path is OK and rehydrates
  the bridge.

The risk window is narrow (only between save on ClassDiagram and
diagram-type switch), but consumers reading the bridge while the user
is *still on the ClassDiagram tab* hit stale data — most visible in
`ClassEditPanel.tsx:36-52` (`collectEnumerationNames`) and the
top-level `availableClassNames` list at `:725-733`. The dropdown for
"existing types" / "association class names" inside the ClassDiagram's
own inspector therefore lags by exactly one save. This is a v3 → v4
regression: the v3 path was `model-state.ts:195` which read the model
directly off Redux state; v4 routes the same query through the bridge
cache without keeping the cache in sync.

**Issue 2 — Bridge cache is never refreshed for the *active*
ClassDiagram.** `setupBridgeForActiveDiagram` only sets the bridge
when the new active diagram is `ObjectDiagram` or `UserDiagram`
(`workspaceSlice.ts:127-148`). When the active diagram **is** a
ClassDiagram, the bridge keeps whatever it had from the previous
session (or `null` on a clean session). `ClassEditPanel` queries the
bridge regardless (`ClassEditPanel.tsx:42`,
`:725-733`, `:738-739`) — so opening the ClassDiagram first thing
after `clearDiagramData()` (or with no prior ObjectDiagram session)
yields an empty enumeration list and an empty `availableClassNames`
collection until something else populates the bridge.

Both issues share a single root cause: the bridge is positioned as a
*cross-diagram* dependency injector but is also being consulted by
*same-diagram* readers (ClassEditPanel) — the population strategy
only handles the cross-diagram half. Cleanest fix: in
`updateDiagramModelThunk.fulfilled`, when
`activeDiagramType === 'ClassDiagram'`, call
`diagramBridge.setClassDiagramData(action.payload.model)`; and add the
ClassDiagram branch to `setupBridgeForActiveDiagram`.

---

## Summary

- All six bridge methods correctly walk v4 `{nodes, edges}`.
- All seven consumers call the bridge with v4-shape inputs.
- Resolution flow for `ObjectName.classId` traces cleanly end-to-end.
- Two staleness risks in the population pipeline
  (`workspaceSlice.ts`) — see Issues 1 & 2 above.
