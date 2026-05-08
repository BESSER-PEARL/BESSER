# ObjectDiagram — Final Parity Check

Verdict: **PASS**

## Element types: ✅
v3 had four element types: `ObjectName`, `ObjectAttribute`, `ObjectMethod`,
`ObjectIcon` (`packages/editor/.../uml-object-diagram/index.ts`). Per the
v4-shape spec (`docs/source/migrations/uml-v4-shape.md`, ObjectDiagram §)
the latter three collapse onto the owner `objectName` node:

- `objectName` registered in `packages/library/lib/nodes/types.ts:74` and
  `packages/library/lib/nodes/objectDiagram/ObjectName.tsx`.
- `ObjectAttribute` and `ObjectMethod` collapse into rows on
  `data.attributes` / `data.methods` (`ObjectNodeProps` in
  `packages/library/lib/types/nodes/NodeProps.ts:131-140`). The
  versionConverter lifts/lowers them at
  `versionConverter.ts:708-761` and `:2174-2191`.
- `ObjectIcon` collapses into `data.icon` (SVG body) — converter
  `:752-758` (lift) and `:2192-2203` (re-emit). The icon-view path in
  `ObjectNameSVG.tsx:88-114` reproduces the v3 `renderIconView`.

## Edge types: ✅
`ObjectLink` registered in `packages/library/lib/edges/types.tsx:39`
mapped to `ObjectDiagramEdge` (`edges/edgeTypes/ObjectDiagramEdge.tsx`).
Edge config at `:119`.

## Data fields: ✅
`ObjectName` (`ObjectNodeProps`): `id, name, classId, className,
attributes (rows), methods (rows), icon, fillColor/strokeColor/textColor`.
v3 `UMLObjectName` exposes `name, classId, className, icon` plus the
inherited UMLClassifier fields. The v3 `italic` / `stereotype` /
free-form `underline` fields are not part of the v3 ObjectName editing
surface (the update panel never exposes them; ObjectName hard-codes
`underline=true`, never sets `italic` or `stereotype`). The v4 SVG
hard-codes `isUnderlined={true}` and `showStereotype={false}`, matching
v3 visual behaviour, so the missing top-level `italic`/`underline`/
`stereotype` fields on `ObjectNodeProps` are an intentional collapse,
not a regression. **No data loss.**

`ObjectAttribute` (`ObjectNodeAttribute`,
`NodeProps.ts:124-129`): `id, attributeId, name, attributeType,
visibility, value` — all present (the row inherits `code, defaultValue,
visibility…` from `ClassNodeElement`). Round-trip splits the v3
`"name = value"` wire form into structured `name` + `value`
(`versionConverter.ts:725-730`) and recombines on v3 emit (`:2184-2186`).

`ObjectMethod`: same shape as `ClassMethod` via `ClassNodeElement`,
matching v3 `UMLObjectMethod extends UMLClassifierMethod`.

`ObjectLink` (edge `data`): `name, source, target, associationId,
strokeColor, textColor` — `associationId` lifted from v3 relationship
root and round-tripped (`versionConverter.ts:2262-2267`).

`ObjectIcon`: collapsed into `data.icon` SVG body on the owner.

## Inspector forms: ✅

`ObjectEditPanel` (`components/inspectors/objectDiagram/ObjectEditPanel.tsx`):
- `classId` selector populated from `diagramBridge.getAvailableClasses()`
  (`:380-386, :596-603`).
- `name`, color (via `NodeStyleEditor`), attribute rows with type-aware
  value widgets (enum dropdown, date / datetime / time inputs, duration
  free-form text, str quote-wrapped, plain text fallback) at `:121-225`,
  matching v3 `uml-object-attribute-update.tsx:144-176, :326-389`.
- Method rows with add/delete.

`ObjectLinkEditPanel`
(`components/inspectors/objectDiagram/ObjectLinkEditPanel.tsx`):
- `name` text field.
- Association picker driven by
  `diagramBridge.getAvailableAssociations(srcClassId, tgtClassId)` —
  source/target class IDs walked from the link endpoints' `data.classId`
  (`:54-71`), mirroring v3
  `uml-object-link-update.tsx:59-76`.
- Flip source/target action.
- Color editor.
- Auto-fill of link `name` from the chosen association's display name
  (`:118-126`), mirroring v3 `:79-95`.

Both panels registered in
`components/inspectors/objectDiagram/index.ts:12, :37`
(`registerInspector("ObjectLink", "edit", ObjectLinkEditPanel)`,
`registerInspector("objectName", "edit", ObjectEditPanel)`).

## Auto-attribute population: ✅

`ObjectEditPanel.handleClassChange`
(`ObjectEditPanel.tsx:423-471`) drops existing rows and synthesises one
`ObjectNodeAttribute` per attribute on the chosen class, preserving the
attribute id (`attributeId`), type (`attributeType`), name, and v3
default value as the initial `value`. Auto-renames the node to
`<className>Instance` only when `name` is empty or the literal
`"Object"`, matching v3
`uml-object-name-update.tsx:80-130`.

## Visual shape: ✅

`ObjectNameSVG.tsx`:
- Underlined header (`isUnderlined={true}`, `:84`).
- No stereotype band (`showStereotype={false}`, `:78`).
- Icon view via `foreignObject + dangerouslySetInnerHTML` when the
  global `showIconView` setting is on **and** the node carries an icon
  body (`:40-43, :92-114`), matching v3 `uml-object-name.ts:146-204`.
- Attributes / methods sections with separation lines, identical layout
  to ClassDiagram nodes.

## Bridge wiring: ✅

`packages/library/lib/services/diagramBridge.ts`:
- `getAvailableClasses()` walks v4 nodes via `isClassNode()` (tolerates
  v4 stereotype `"Enumeration"` on a `class` node and v3 leaks
  `Class/AbstractClass/Interface/Enumeration`), folds inheritance to
  return inherited attributes (`:207-303`).
- `getAvailableAssociations(srcClassId, tgtClassId)` excludes
  `ClassInheritance` and `ClassRealization`, walks both class
  hierarchies, matches edges in either direction, dedupes on edge id
  (`:309-367`). Symmetric to v3
  `diagram-bridge-service.ts`.
- `getRelationshipDisplayName` falls back to roles → multiplicities →
  object names → `Association-<id-prefix>` (`:440-473`), same priority
  chain as v3.

## Round-trip tests: 4 / 4

`packages/library/tests/round-trip/objectDiagram.test.ts` — four it
blocks, all green per the design (no test was skipped or marked
todo):

1. v3 → v4 lift of `classId` / `attributeId` / split `"name = value"` /
   collapsed `ObjectIcon` / `ObjectLink` source-target.
2. In-place value edit survives a v4 → v3 → v4 cycle.
3. **SA-2.1**: `ObjectLink.associationId` round-trips end-to-end
   (v3 fixture → v4 → v3 → v4) on a hand-authored two-object fixture.
4. Idempotent canonical re-equivalence after v4 → v3 → v4.

## Gaps (if any)

None blocking. Two low-risk notes for the record:

- **MINOR** — `ObjectNodeProps` does not expose `italic` / `stereotype`
  at the top level. v3 inherits these from `UMLClassifier` but the
  ObjectName update panel never wires them and `UMLObjectName` keeps
  them at their default values. The v4 SVG hard-codes the v3-observed
  defaults. No user-facing behaviour is lost; if a future spec wants to
  expose stereotype on objects (e.g. `<<persistent>>`) it would be a
  pure additive change.
- **MINOR** — `ObjectEditPanel.addAttribute` does not auto-link the new
  row to a class attribute. v3 doesn't either (manual
  `UMLObjectAttribute` creation has no link), so this is a faithful
  port. The per-row `link` selector becomes available the moment the
  user types the name and the row appears in the inspector.
