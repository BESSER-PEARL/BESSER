# PC-3 — Class edges + OCL

**Verdict:** Substantial parity. All nine v3 ClassDiagram edge types are
registered in the new lib and bind to the unified `ClassDiagramEdge`
renderer + `ClassEdgeEditPanel` inspector. Auto-detection of OCL/LinkRel
is correctly absent (the new lib never coerces a connection between two
regular classes into `ClassOCLLink` / `ClassLinkRel`). The OCL constraint
node is implemented as a sticky-note shape with folded corner — visually
distinct from a class rectangle. **Three concrete gaps remain** (see
"Top gaps" below); none block migration but each is a behaviour-parity
miss versus the v3 source-of-truth `uml-class-association-update.tsx`.

This is a read-only audit. No code changes.

---

## Source-of-truth files audited

### v3 (old)

- `packages/editor/src/main/packages/uml-class-diagram/uml-class-association/uml-class-association-update.tsx`
  (sole edit-form for every `UMLAssociation`-derived edge type)
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-inheritance/uml-class-inheritance.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-realization/uml-class-realization.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-composition/uml-class-composition.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-aggregation/uml-class-aggregation.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-bidirectional/uml-class-bidirectional.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-unidirectional/uml-class-unidirectional.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-dependency/uml-class-dependency.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-ocl-link/uml-class-ocl-link.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-link-rel/uml-class-link-rel.ts`
- `packages/editor/src/main/packages/uml-class-diagram/uml-class-ocl/uml-class-ocl-constraint-component.tsx`
  (the v3 OCL-constraint sticky-note node visual)

### v4 (new lib)

- `packages/library/lib/edges/edgeTypes/ClassDiagramEdge.tsx` (unified renderer)
- `packages/library/lib/edges/types.tsx` (registry + `edgeConfig`)
- `packages/library/lib/utils/edgeUtils.ts` (`getEdgeMarkerStyles`,
  `getDefaultEdgeType`)
- `packages/library/lib/edges/labelTypes/EdgeEndLabels.tsx` (renders
  `sourceMultiplicity` / `sourceRole` / `targetMultiplicity` /
  `targetRole`)
- `packages/library/lib/components/inspectors/classDiagram/ClassEdgeEditPanel.tsx`
  (single inspector for all 9 edge types)
- `packages/library/lib/components/inspectors/classDiagram/index.ts`
  (registers `ClassEdgeEditPanel` against all 9 types)
- `packages/library/lib/hooks/useConnect.ts` (`onConnect` / `onConnectEnd`)
- `packages/library/lib/utils/edgeUtils.ts` `getDefaultEdgeType` →
  `ClassUnidirectional` for `ClassDiagram`
- `packages/library/lib/nodes/classDiagram/ClassOCLConstraint.tsx`
  (sticky-note OCL node)
- `packages/library/lib/components/inspectors/classDiagram/ClassOCLConstraintEditPanel.tsx`

---

## Edge-type registration & marker rendering

All nine edge types are registered to `ClassDiagramEdge` in
`packages/library/lib/edges/types.tsx` (lines 22–35) and configured in
`edgeConfig` (lines 102–113). Markers come from
`getEdgeMarkerStyles(edgeType)` in
`packages/library/lib/utils/edgeUtils.ts`:

| Edge type | v3 visual | v4 marker / dasharray | Match |
| --- | --- | --- | --- |
| ClassBidirectional | plain solid line, no markers | no marker, dash `0` | yes |
| ClassUnidirectional | solid line + filled arrow at target | `url(#black-arrow)`, dash `0` | yes |
| ClassAggregation | solid line + open diamond at source | `url(#white-rhombus)` at **end**, dash `0` | side mismatch (see Gap-3) |
| ClassComposition | solid line + filled diamond at source | `url(#black-rhombus)` at **end**, dash `0` | side mismatch (see Gap-3) |
| ClassInheritance | solid line + open triangle at target | `url(#white-triangle)`, dash `0` | yes |
| ClassRealization | dashed line + open triangle at target | `url(#white-triangle)`, dash `10` | yes |
| ClassDependency | dashed line + arrow at target | `url(#black-arrow)`, dash `10` | yes |
| ClassOCLLink | dashed dependency-style line + open arrow (per v3 visual) | **default branch:** no marker, dash `0` (plain solid) | **NO — Gap-1** |
| ClassLinkRel | plain solid line, no markers | **default branch:** no marker, dash `0` | yes |

`registerEdgeTypes` (types.tsx:88) preserves the same registry reference
so runtime additions remain visible — nothing in the audit path mutates
the class entries.

## Inspector parity (`ClassEdgeEditPanel`)

`packages/library/lib/components/inspectors/classDiagram/ClassEdgeEditPanel.tsx`
is registered for all nine edge types via the side-effect import in
`classDiagram/index.ts:CLASS_EDGE_TYPES` (lines 16–28). It mirrors the
v3 form layout:

| v3 control | v4 control | Match |
| --- | --- | --- |
| `<Header>popup.association</Header>` + `<ColorButton>` | `EdgeStyleEditor` with `label="Association"` + colour swatches | yes |
| flip action `<Button>...<ExchangeIcon /></Button>` invoking `UMLRelationshipRepository.flip` | `IconButton` with `SwapHorizIcon` swapping `source ↔ target` and `sourceHandle ↔ targetHandle` | yes |
| name `<Textfield>`, hidden when `type === ClassInheritance` | `MuiTextField` `Association name`, hidden when `NON_DIRECTIONAL_TYPES.has(type)` | partial — see Gap-2 |
| 4-entry association-type Dropdown (Uni / Bi / Composition / Inheritance — Aggregation / Dependency / Realization commented out) | 9-entry Select (the four v3 entries plus Aggregation, Realization, Dependency, OCL Link, Link (BESSER)) | divergence by design (`SA-2.1` comment): authoring parity restored, not just the v3 4-entry subset |
| per-end multiplicity `<Textfield>` placeholder `'1..1'`, `'(1,1) or 1..1'` when ER | `MuiTextField` placeholder `'1..1'` / `'(1,1) or 1..1'`, driven by `useSettingsStore.classNotation` | yes (placeholder); writes are raw — see Gap-2.5 |
| per-end role `<Textfield>` | `MuiTextField` (`sourceRole` / `targetRole`) | yes |
| `StylePane lineColor textColor` colour pickers | `EdgeStyleEditor` with `strokeColor` / `textColor` keys | yes |
| `autoFocus` on source-multiplicity field | `autoFocus` on source-multiplicity field | yes |

Form is shown only for the eight non-inheritance/realization types.

## Auto-detection on connect

Required: drawing an edge between two regular classes must NEVER
auto-pick `ClassOCLLink` or `ClassLinkRel`. Audited in
`packages/library/lib/hooks/useConnect.ts`:

- `onConnect` (line 86) builds the new edge with
  `type: defaultEdgeType` from
  `getDefaultEdgeType(diagramType)`.
- `onConnectEnd` (line 100, fallback drop-on-node path) does the same:
  `type: defaultEdgeType` (line 187).
- `getDefaultEdgeType("ClassDiagram")` →
  `"ClassUnidirectional"` (`packages/library/lib/utils/edgeUtils.ts:773`).
- No branch in `useConnect.ts` inspects either endpoint's node type.
  Therefore neither `ClassOCLLink` nor `ClassLinkRel` is ever produced
  by the connect handlers regardless of source/target node type.

Verdict: the auto-detection requirement is satisfied — both BESSER-
specific edge types are reachable only through explicit user selection
in the inspector's type dropdown (lines 60–61 of
`ClassEdgeEditPanel.tsx`).

## OCL constraint node visual

`packages/library/lib/nodes/classDiagram/ClassOCLConstraint.tsx`
(registered as `"ClassOCLConstraint"` in
`packages/library/lib/nodes/types.ts:75,163`) renders:

- An SVG path with the v3 sticky-note silhouette
  `M 0 0 L w-fold 0 L w fold L w h L 0 h Z` plus a folded-corner overlay
  `M w-fold 0 L w-fold fold L w fold` (lines 121–135).
- Default colours: `fillColor "#fff8c4"` (sticky-note yellow),
  `strokeColor "#bda21f"`, `textColor "#3a2e00"` (lines 106–108) —
  always overridable via `data.fillColor / strokeColor / textColor`.
- Constraint name as a 12 px / 600-weight header, optional
  `«inv» / «pre» / «post»` badge derived from
  `_OCL_HEADER_RE.exec(data.expression)` (lines 25–48), and a wrapped
  monospace expression body (`wrapText`, line 50).

Compared to the v3 `uml-class-ocl-constraint-component.tsx`:

- v3 uses `ThemedPath` (theme fill/stroke from styled-components),
  defaulting to whatever the theme provides — not an explicit yellow.
  The new lib hard-codes the sticky-note yellow, which matches the
  user-visible v3 paper colour but is no longer theme-driven.
- Same folded-corner geometry (v3 used `width-15` for the fold; v4 uses
  `fold = 14`).
- Same badge-derivation regex (`_OCL_HEADER_RE`) and same `«inv»/«pre»/
  «post»` short labels.

Verdict: distinct yellow / sticky-note shape — NOT a class rectangle.

## Top 3 gaps

### Gap-1 — `ClassOCLLink` falls to the default marker branch

`packages/library/lib/edges/types.tsx:29–35` and `:109–113` document
that `ClassOCLLink` should render as a dashed dependency-style stroke
with an open arrow, mirroring v3. But
`packages/library/lib/utils/edgeUtils.ts:174–266` has no `case
"ClassOCLLink"` — it falls through to the `default` branch
(`strokeDashArray: "0"`, no `markerEnd`), so it ends up indistinguishable
from `ClassLinkRel` (a plain solid line). Visual divergence from v3,
which drew OCL link as dashed with an arrow.

### Gap-2 — `ClassRealization` hides name / multiplicity / role; v3 only hides them for `ClassInheritance`

In v3 `uml-class-association-update.tsx:75` the toggle is
`const isInheritance = element.type === ClassRelationshipType.ClassInheritance`.
The new inspector
(`ClassEdgeEditPanel.tsx:64–67,89`) defines
`NON_DIRECTIONAL_TYPES = new Set(["ClassInheritance", "ClassRealization"])`
and uses that to gate the name field, multiplicity rows and role rows.
A v3 user could still set association name / multiplicity / role on a
`ClassRealization` edge; the new lib hides those controls, dropping that
authoring affordance.

### Gap-3 — Aggregation / Composition diamond rendered on the wrong end

v3 (and standard UML) places the open / filled diamond on the source
(whole) end. The new lib's
`getEdgeMarkerStyles` returns the diamond as `markerEnd` (target) for
both `ClassAggregation` and `ClassComposition`
(`edgeUtils.ts:195–208`). Unless the upstream renderer flips it (the
`ClassDiagramEdge` consumer just forwards `markerEnd` / `markerStart`
straight into `BaseEdge` and `EdgeInlineMarkers`), users will see the
diamond on the part end, not the whole end. This is the most visible
notation regression.

## Bonus parity nit (Gap-2.5, low-severity)

The v3 form normalises ER cardinality to UML on every keystroke via
`erCardinalityToUML(value)` before storing
(`uml-class-association-update.tsx:194–203`). The new inspector's
`updateData({ sourceMultiplicity: e.target.value })`
(`ClassEdgeEditPanel.tsx:193–194,225–227`) writes the raw input, so a
user typing `(1,N)` in ER mode will store `(1,N)` rather than `0..*`.
The placeholder hint is correct, but the normalising side-effect is
gone. The function `erCardinalityToUML` is still exported from
`packages/library/lib/utils/multiplicity.ts:45` and re-exported from
`lib/index.tsx:77`, so the wiring fix is one-line.
