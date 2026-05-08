# SA-DEEP-EDGE-MARKERS-EXHAUSTIVE — Edge Marker Audit

Comprehensive walk of every BESSER-relevant edge type registered in
`packages/library/lib/edges/edgeTypes/` and switched in
`packages/library/lib/utils/edgeUtils.ts::getEdgeMarkerStyles`,
verified against the v3 metamodel SVG renderers under
`packages/editor/src/main/packages/<diagram>/<edge>/<edge>-component.tsx`.

This deepens the partial sweep done by SA-DEEP-3 with stroke styles,
arrow position (markerStart vs markerEnd), filled vs hollow shapes,
and dead/duplicate cases.

## Audit table — per edge

Legend: `MS` = markerStart, `ME` = markerEnd. `solid` = no dasharray.
`hollow` = stroke-only outline; `filled` = solid fill.

### ClassDiagram

| Edge | v3 visual (source) | v4 case | Match? |
|------|--------------------|---------|--------|
| ClassInheritance | ME = hollow Triangle, solid (`uml-association-component.tsx::Marker.Triangle`) | ME `white-triangle`, solid | OK |
| ClassRealization | ME = hollow Triangle, dashed (stroke=`7`) | ME `white-triangle`, dash `10` | OK (dash length differs slightly, visually equivalent) |
| ClassDependency | ME = open Arrow (V-shape), dashed (stroke=`7`) | ME `black-arrow`, dash `10` | OK (same shape/stroke; dash length differs) |
| ClassUnidirectional | ME = open Arrow, solid | ME `black-arrow`, solid | OK |
| ClassBidirectional | no marker, solid | no marker, solid | OK |
| ClassAggregation | ME = hollow Rhombus, solid (path drawn whole→part with marker at end) | MS `white-rhombus`, solid (path drawn part→whole with marker at start) | OK in result, **convention inverted** — v4 places diamond via `markerStart` while v3 uses `markerEnd`; correct pixels rely on consistent source/target convention in JSON converter |
| ClassComposition | ME = filled Rhombus (`RhombusFilled`), solid | MS `black-rhombus`, solid | Same caveat as Aggregation — different end of the edge, same outcome iff converter swaps source/target consistently |
| ClassOCLLink | **no marker**, dashed (stroke=`5,5`) — v3 explicitly comments out `Marker.Arrow` | ME `class-ocl-link-marker` (custom open arrow), dash `4 2` | **Mismatch (intentional addition)** — v4 *adds* an arrow that v3 deliberately omits. PC-3 fix comment claims "open arrow" but v3 source has it commented out. |
| ClassLinkRel | no marker, dashed (stroke=`5,5`) | **falls through to `default`** — no marker, **solid** | **Broken** — v4 renders solid where v3 was dashed. |

### StateMachineDiagram / AgentDiagram

| Edge | v3 visual | v4 case | Match? |
|------|-----------|---------|--------|
| StateTransition | ME = open Arrow (V-shape), solid (`uml-state-transition-component.tsx`) | ME `black-arrow`, solid | OK |
| AgentStateTransition | ME = open Arrow, solid (`agent-state-transition-component.tsx`) | ME `black-arrow`, solid (matched at line 196 case-block; **the dedicated block at line 351 is dead code**) | OK at runtime; clean-up needed |
| AgentStateTransitionInit | ME = open Arrow, **solid** (`agent-state-transition-init-component.tsx` — no `strokeDasharray`) | ME `black-arrow`, **dash `10`** | **Broken** — v4 dashes a line that v3 renders solid. The block comment claiming "init edge additionally uses a dashed stroke" contradicts the v3 source. |

### ObjectDiagram / UserDiagram

| Edge | v3 visual | v4 case | Match? |
|------|-----------|---------|--------|
| ObjectLink (also UserModelLink alias) | no marker, solid, **strokeWidth=2** (`uml-object-link-component.tsx`) | no marker, solid (default `strokeWidth=1` from `BaseEdge`) | OK markers; minor — v4 line is thinner than v3. Cosmetic only. |

### NNDiagram

| Edge | v3 visual | v4 case | Match? |
|------|-----------|---------|--------|
| NNNext | ME = open Arrow (`NNAssociationComponent` registered for `NNRelationshipType.NNNext` in `components.ts:412`), solid | ME `black-arrow`, solid | OK |
| NNComposition | ME = filled Rhombus (`RhombusFilled`), solid; v3 reverses path when source is NNContainer so the diamond ends up on container side | MS `black-rhombus`, solid | OK in result (same diamond/colour at container side via different mechanism) |
| NNAssociation | no marker, solid (`NNAssociationLineComponent`) | no marker, solid | OK |

### ActivityDiagram

| Edge | v3 visual | v4 case | Match? |
|------|-----------|---------|--------|
| ActivityControlFlow | ME = open Arrow, solid (`uml-activity-control-flow-component.tsx`) | ME `black-arrow`, solid | OK |

## Missing / wrong cases

1. **`ClassLinkRel` — missing case.** Falls through to `default` and renders solid with no marker. v3 = dashed `5,5`, no marker.
2. **`AgentStateTransitionInit` — wrong stroke.** v4 sets `strokeDashArray: "10"`, v3 source has no `strokeDasharray` (solid). The PC-8 comment that justifies this is wrong about v3 behavior.
3. **`AgentStateTransition` — duplicate / dead case.** Listed both in the combined `StateTransition`/`AgentStateTransition` block (line 195-202) and again standalone (line 351-357). The standalone block is unreachable.
4. **`ClassOCLLink` — added marker not in v3.** v3 explicitly comments out the arrow (`// return Marker.Arrow;`) so v3 has no marker. v4 adds `class-ocl-link-marker`. May be an intentional design improvement (the comment in v4 says "PC-3 fix"), but it diverges from v3 baseline.
5. **`ObjectLink` strokeWidth.** v3 = `strokeWidth={2}`, v4 default = 1. Cosmetic.
6. **`ClassAggregation` / `ClassComposition` end convention.** Output looks correct *if* the JSON-to-BUML converter swaps source/target between v3 (whole→part) and v4 (part→whole) consistently for these two edges. Tests should round-trip a composition where the diamond stays on the whole.

## Recommendation per gap (one-liner each)

1. **ClassLinkRel** — add a case mirroring v3: no marker, `strokeDashArray: "5 5"`.
2. **AgentStateTransitionInit** — change `strokeDashArray: "10"` to `"0"` (solid) to match v3, and update the SA-FIX-Agent / PC-8 comment.
3. **AgentStateTransition duplicate** — delete the standalone case block (lines 351–357) so only the combined `StateTransition`/`AgentStateTransition` block remains.
4. **ClassOCLLink** — either revert to no marker (match v3) or document the PC-3 marker addition as an intentional v4 visual upgrade in `migrations/uml-v4-shape.md`.
5. **ObjectLink width** — set `strokeWidth: 2` for `ObjectLink` (and `UserModelLink` alias) in the BaseEdge style or via a small CSS rule.
6. **Aggregation / Composition source-end test** — add a Playwright snapshot that verifies the diamond stays on the whole-side after a JSON → BUML → JSON round-trip.

## Total

- **Edges audited:** 13 (BESSER-relevant; non-BESSER edges like BPMN/UseCase/Component/Deployment/Flowchart/SyntaxTree/PetriNet/ReachabilityGraph/Communication scoped out per task brief).
- **Wrong / divergent markers:** **5** (`ClassLinkRel`, `AgentStateTransitionInit`, `AgentStateTransition` duplicate, `ClassOCLLink` extra marker, `ObjectLink` width).
- **End-convention concerns (correct iff converter consistent):** 2 (`ClassAggregation`, `ClassComposition`).
