# ClassDiagram — Final Parity Check

Verdict: **PASS (with documented MINOR gaps)**

Submodule HEAD checked: `771c064` (SA-7b cutover). All sources read via the
`besser/utilities/web_modeling_editor/frontend` submodule.

## Element types: ✅

v3 ships eight ClassDiagram element types (see
`packages/editor/src/main/packages/uml-class-diagram/index.ts`):
`Package`, `Class`, `AbstractClass`, `Interface`, `Enumeration`,
`ClassAttribute`, `ClassMethod`, `ClassOCLConstraint`. v4 maps these
according to `docs/source/migrations/uml-v4-shape.md` "Mapping rules
(ClassDiagram)" §:

| v3 type                | v4 representation                                                | Location |
|------------------------|------------------------------------------------------------------|----------|
| `Package`              | node `type: "package"`                                           | `packages/library/lib/nodes/types.ts:72`, `nodes/classDiagram/Package.tsx` |
| `Class`                | node `type: "class"`, `data.stereotype` undefined                | `nodes/types.ts:73`, `nodes/classDiagram/Class.tsx` |
| `AbstractClass`        | node `type: "class"`, `data.stereotype === "Abstract"`           | `types/nodes/enums/ClassType.ts:2`, converter `utils/versionConverter.ts:690-691` |
| `Interface`            | node `type: "class"`, `data.stereotype === "Interface"`          | `enums/ClassType.ts:3`, converter `:692-693` |
| `Enumeration`          | node `type: "class"`, `data.stereotype === "Enumeration"`        | `enums/ClassType.ts:4`, converter `:694-695` |
| `ClassAttribute`       | row on owner's `data.attributes: ClassNodeElement[]`             | `types/nodes/NodeProps.ts:55-83`, converter `versionConverter.ts:678-679` |
| `ClassMethod`          | row on owner's `data.methods: ClassNodeElement[]`                | `NodeProps.ts:55-83`, converter `:680-681` |
| `ClassOCLConstraint`   | row on owner's `data.oclConstraints: ClassOCLConstraint[]`       | `NodeProps.ts:104-112`, converter `:682-683` |

The v3 `ClassOCLConstraint` collapse onto its owner is the recommended
mapping in `uml-v4-shape.md` §"Mapping rules (ClassDiagram)" bullet 4.

## Edge types: ✅

All nine v3 relationship subtypes are registered as v4 edge types in
`packages/library/lib/edges/types.tsx:22-35` (mapping → `ClassDiagramEdge`):

`ClassAggregation`, `ClassInheritance`, `ClassRealization`,
`ClassComposition`, `ClassBidirectional`, `ClassUnidirectional`,
`ClassDependency`, `ClassOCLLink`, `ClassLinkRel`. Edge config in the
same file `:102-113`. Inspector slots registered in
`components/inspectors/classDiagram/index.ts:15-31`. PopoverManager edit
slots in `components/popovers/PopoverManager.tsx:259-267`.

## Data fields: ✅

`ClassNodeProps` (`types/nodes/NodeProps.ts:85-95`):
`name, fillColor, strokeColor, textColor, stereotype, attributes, methods,
oclConstraints`.

`ClassNodeElement` row (`:55-83`) for attributes / methods carries the
full BESSER member shape: `id, name, attributeType, visibility, code,
implementationType, stateMachineId, quantumCircuitId, isOptional,
isDerived, isId, isExternalId, defaultValue, parameters, returnType`.
Mirrors v3 `IUMLClassifierMember`
(`packages/common/uml-classifier/uml-classifier-member.ts:64-76`) and
`MethodImplementationType` (`:57-62`).

`ClassOCLConstraint` (`:104-112`): `id, name, expression, description,
kind`. v3's `IUMLClassOCLConstraint`
(`uml-class-ocl-constraint.ts:8-12`) had only `constraint` (renamed to
`expression`) and `description`. The v4 shape is a strict superset —
`name` and `kind` are new opt-in fields.

The converter lifts every v3 row field into the v4 row
(`versionConverter.ts:583-636` `extractClassifierMember`) and re-emits
them on the reverse path (`:childRowToV3` from line ~1900). OCL is
lifted at `:643-657` `extractOCLConstraint` and re-emitted around
`:2128-2138`.

## Inspector forms: ✅

The same body is bound to both surfaces. `App.tsx:43` side-effect-imports
`./components/inspectors`, which calls
`registerInspector("class", "edit", ClassEditPanel)`
(`components/inspectors/index.ts:36`). The registry override runs *after*
`PopoverManager`'s `registerInspectors("edit", editPopovers)`
(`PopoverManager.tsx:615`), so both `PropertiesPanel` and
`PopoverManager` resolve `ClassEditPanel` for `class` nodes. The legacy
`ClassEditPopover` and `EditableAttributeList` files survive in
`components/popovers/classDiagram/` but are unreachable at runtime.

`ClassEditPanel`
(`components/inspectors/classDiagram/ClassEditPanel.tsx`):

| v3 field                              | v4 widget                                                                | Source line |
|---------------------------------------|--------------------------------------------------------------------------|-------------|
| Class name                            | `MuiTextField` with `safeIdentifier` sanitiser on commit                | `:746-752`, sanitiser `:60-61` |
| Color (fill / line / text)            | `NodeStyleEditor` (Paint roller → `ColorButtons`)                        | `:928-931` |
| Stereotype toggle (Abstract / Interface / Enumeration) | `StereotypeButtonGroup`                                                  | `:933-936` |
| Attribute visibility (`+ - # ~`)      | `Select` with `VISIBILITIES` const                                       | `:82-87, :176-189` |
| Attribute name (sanitised)            | `MuiTextField`, regex `[^a-zA-Z0-9_]`                                    | `:190-201` |
| Attribute type (primitives + classes + enumerations + custom) | `Select` with `PRIMITIVE_TYPES`, sibling-class list, sibling-enum list, `__custom__` sentinel | `:70-80, :202-234, :41-53` |
| `isId` / `isExternalId` / `isOptional` / `isDerived` checkboxes | `FormControlLabel` + `Checkbox`. Mutual-exclusion on `isId` ⇄ `isOptional` and `isExternalId` ⇄ `isOptional` matches v3. | `:254-306` |
| Default value (free-form)             | `MuiTextField`                                                           | `:308-319` |
| Method visibility / name / delete     | Same controls as attribute row                                           | `:379-407` |
| Method return type (primitives + classes + custom) | `Select`                                                                 | `:409-453` |
| Method parameters (name + type rows, Enter-to-add) | parameter-row `MuiTextField`s + `+ add parameter` input                  | `:455-529` |
| Method `implementationType` (`none` / `code` / `bal` / `state_machine` / `quantum_circuit`) | `Select` with `IMPLEMENTATION_TYPES`                                     | `:89-98, :531-565` |
| `stateMachineId` (cross-diagram)      | `Select` populated from `diagramBridge.getStateMachineDiagrams()`        | `:566-583`, bridge call `:738` |
| `quantumCircuitId` (cross-diagram)    | `Select` populated from `diagramBridge.getQuantumCircuitDiagrams()`      | `:584-601`, bridge call `:739` |
| Method body (`code` / `bal`)          | CodeMirror editor with `python()` extension and tab indent               | `:604-641` |
| OCL constraint name + expression      | `MuiTextField` (single-line) + `MuiTextField multiline`                  | `:670-694` |
| Methods section hidden when stereotype = `Enumeration` | `nodeData.stereotype !== "Enumeration"` guard                            | `:970-1002` |

`ClassEdgeEditPanel`
(`components/inspectors/classDiagram/ClassEdgeEditPanel.tsx`):

| v3 field (`uml-class-association-update.tsx`) | v4 widget                                                       | Source line |
|----------------------------------------------|------------------------------------------------------------------|-------------|
| Color (line / text)                          | `EdgeStyleEditor`                                                | `:137-148` |
| Flip (`ExchangeIcon`)                        | `IconButton` with `SwapHorizIcon`, swaps `source`/`target` + handles | `:112-125, :141-147` |
| Association name (hidden for Inheritance / Realization) | `MuiTextField`                                                   | `:64-67, :153-162` |
| Type-picker dropdown (Bi / Uni / Aggregation / Composition / Inheritance / Realization / Dependency) | `Select` with `EDGE_TYPE_OPTIONS`                                | `:47-62, :164-174` |
| Source / Target multiplicity                 | `MuiTextField` with placeholder `"1..1"` (or `"(1,1) or 1..1"` when `classNotation === "ER"`) | `:93-94, :186-196, :219-228` |
| Source / Target role                         | `MuiTextField`                                                   | `:198-209, :230-241` |

## Constraints: ✅

- **Multiplicity bounds**: parsed and converted between UML and ER forms
  via `utils/multiplicity.ts:8-21, 29-35, 45-59`. ER↔UML round-trip in
  `parseMultiplicity` / `toERCardinality` / `erCardinalityToUML`.
- **Enumeration cannot have methods**: `ClassEditPanel` hides the entire
  Methods section when `stereotype === "Enumeration"` (`:970-1002`),
  matching v3 `uml-classifier-update.tsx:344` and v3's
  `create()` guard at `:442-444` that refuses to create a method on an
  Enumeration. The v4 layer has no equivalent data-layer refusal — a
  programmatic `setNodes` could still push a method onto an Enumeration.
  v3's same path is also a UI-only constraint, so this is not a
  regression.
- **Inheritance cycles**: neither v3 nor v4 prevent cycles. v3
  `UMLClassInheritance` has no cycle check; the user is responsible.
  v4 matches.
- **OCL expression validation**: surfaced via the backend `/validate-diagram`
  endpoint. Both v3 and v4 send the raw expression text to the backend
  for parsing — no client-side OCL validation in either codebase.
- **Class name sanitisation**: `safeIdentifier` strips
  `[^a-zA-Z0-9_]` on commit (`ClassEditPanel.tsx:60-61, :750`), mirroring
  v3 `uml-classifier-update.tsx:475`. Same regex applied on attribute
  names at `ClassEditPanel.tsx:198, :791`.

## Visual shape: ✅

Class-rectangle layout (`components/svgs/nodes/classDiagram/ClassSVG.tsx`):
- `<StyledRect>` border + fill (`:72-78`) honour `data.fillColor` /
  `data.strokeColor`.
- `HeaderSection` renders stereotype band (height
  `LAYOUT.DEFAULT_HEADER_HEIGHT_WITH_STEREOTYPE`) when `stereotype` set,
  otherwise plain header height. Matches v3
  `UMLClassifier.headerHeight` (`uml-classifier.ts:57-59`).
- `SeparationLine` between header / attributes / methods (`:95-99,
  :115-119`).
- ER-mode hide-methods on Class/AbstractClass: `Class.tsx:67-79` reads
  `useClassNotation()` and re-formats rows on toggle without an editor
  remount. v3 hid methods in render via
  `uml-classifier.ts:86-91, :140-143`. v4 still renders the SVG section
  (the row count is the same; widths are recomputed and the canvas does
  not pad), but the methods compartment is not separately collapsed —
  this matches the v3 behaviour for the *non-class* stereotypes.

Edge markers (`utils/edgeUtils.ts:175-238`):

| Edge type             | Marker (v3 visual → v4)                            | Stroke    |
|----------------------|----------------------------------------------------|-----------|
| `ClassInheritance`   | hollow triangle (`white-triangle`)                 | solid     |
| `ClassRealization`   | hollow triangle (`white-triangle`)                 | dashed (`10`) |
| `ClassComposition`   | filled diamond (`black-rhombus`)                   | solid     |
| `ClassAggregation`   | hollow diamond (`white-rhombus`)                   | solid     |
| `ClassUnidirectional`| filled arrow (`black-arrow`)                       | solid     |
| `ClassBidirectional` | no marker                                          | solid     |
| `ClassDependency`    | filled arrow (`black-arrow`)                       | dashed (`10`) |
| `ClassOCLLink`       | falls through `default:` — no marker, solid stroke | solid (matches v3) |
| `ClassLinkRel`       | falls through `default:` — no marker, solid stroke | solid (matches v3) |

v3's `ClassOCLLink` and `ClassLinkRel` both use the plain
`UMLAssociationComponent` (`packages/components.ts:170-171`) which is a
solid line without markers. v4's default fallback gives the same visual,
even though the comment in `lib/edges/types.tsx:31-32` describes a
dashed-with-arrow OCL visual that neither v3 nor v4 implements. **No
regression**, but the comment is misleading — flagged as MINOR below.

## Round-trip tests: 4 / 4

`packages/library/tests/round-trip/classDiagram.test.ts`:

1. v3 → v4 lift of a representative fixture (Package + classes + attributes
   + methods + OCL).
2. In-place attribute edit survives a v4 → v3 → v4 cycle.
3. **SA-2.1**: `ClassOCLLink` and `ClassLinkRel` round-trip end-to-end.
4. Idempotent canonical re-equivalence (`toV4(toV3(toV4(m))) === toV4(m)`).

All four it-blocks are normal `it(...)` calls — none skipped or
`it.todo`.

## Gaps (if any)

- **MEDIUM** — `ClassNodeProps` and the inspector do not expose
  `description`, `uri`, or `icon`. The v4-shape spec
  (`docs/source/migrations/uml-v4-shape.md:166`) lists these on
  `ClassNodeData`. v3 surfaces them via the `StylePane`
  (`uml-classifier-update.tsx:217-225` — `showDescription showUri showIcon`).
  The v4 `NodeStyleEditor` only edits `name` + colors. **Round-trip
  impact**: any v3 fixture carrying these fields loses them on
  conversion. v3 → v4 lift omits them
  (`versionConverter.ts:660-666, 698-704`); v4 → v3 emit also omits them
  (`:2103-2119`). File path: `lib/types/nodes/NodeProps.ts:85-95`,
  `lib/components/ui/StyleEditor/NodeStyleEditor.tsx`,
  `lib/utils/versionConverter.ts`.
- **MEDIUM** — `ClassEdgeEditPanel.updateData` for multiplicity does not
  invoke `erCardinalityToUML` before storing. v3
  `uml-class-association-update.tsx:201-202` normalises ER input
  (`(0,N)` → `0..*`) on every commit so storage is always UML form.
  Without it, an ER-mode user typing `(1,N)` keeps that literal in
  storage, which the round-trip tests don't currently catch (they write
  UML form). The helper exists at
  `lib/utils/multiplicity.ts:45-59` and is exported from `index.tsx:45`
  — it's just not called. File path:
  `lib/components/inspectors/classDiagram/ClassEdgeEditPanel.tsx:96-102,
  :193-228`.
- **MINOR** — `italic` / `underline` flags on classifiers are dropped
  through the converter. v3 `UMLClassifier.italic` /
  `.underline` (`uml-classifier.ts:49-50`) are set automatically when
  the user picks `AbstractClass` (italic), so nothing user-set is lost
  in normal flows, but bespoke v3 fixtures with explicit
  `italic`/`underline` overrides do not round-trip. File path:
  `lib/types/nodes/NodeProps.ts:85-95`, converter
  `lib/utils/versionConverter.ts:660-704`.
- **MINOR** — v3 inspector exposes a "Quick Code" button that creates a
  new method and seeds a Python-template body
  (`uml-classifier-update.tsx:424-431, :461-472`). v4
  `ClassEditPanel.addMethod` (`:840-866`) creates a bare method with
  `implementationType: "none"`. Power users used to the v3 shortcut will
  feel the regression, but every action it performed is reachable in two
  steps in the v4 panel. File path:
  `lib/components/inspectors/classDiagram/ClassEditPanel.tsx:840-866`.
- **MINOR** — v3 inspector exposes per-row "move up / move down" arrows
  for both attributes and methods
  (`uml-classifier-update.tsx:255-273, :355-373`). v4 has no reorder
  control on the inspector; rows are reorderable only via the legacy
  `EditableAttributeList` drag-and-drop in the popover, which is
  *unreachable at runtime* because the registry override binds the
  panel-only `ClassEditPanel` to both surfaces. Net effect: no reorder
  UX in v4. File path:
  `lib/components/inspectors/classDiagram/ClassEditPanel.tsx`.
- **MINOR** — Misleading comment on `lib/edges/types.tsx:29-33` claims
  `ClassOCLLink` "draws as a dashed dependency-style arrow (open arrow +
  dotted stroke)". The actual implementation falls through
  `getEdgeMarkerStyles` `default:`, yielding a plain solid line —
  matching v3, but contradicting the comment. Cosmetic doc update only.
- **MINOR** — `ClassSVG.tsx:113-129` always renders the methods
  separation line even when `methods.length === 0` (uses `>= 0`). The
  visual artifact is masked because the `RowBlockSection` is empty, but
  the SVG path is emitted. v3's `uml-classifier.ts:135-144` skipped the
  loop when there were no methods. Cosmetic only — no test failure.
