# ClassDiagram Deep Analysis (v3 vs v4)

Audited: 2026-05-08.
v3 ref: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-class-diagram/` and `packages/editor/src/main/packages/common/uml-classifier/`.
v4 ref: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/` at submodule SHA `a31c4eca3eb8058412e77ad5840365d0397ddc0c` (branch `feature/migration-react-flow`).
Parent branch: `claude/refine-local-plan-sS9Zv`.

---

## Top-line verdict

**PARTIAL.** v4 covers Class / AbstractClass / Enumeration / OCL constraint / Package / ColorDescription render and the seven canonical edge types with reasonable parity, including a real port of `formatDisplayName` and the ER-notation toggles for multiplicity / classNotation. However there is a long tail of inspector-level regressions: the `description` / `uri` / `icon` style fields are dropped entirely from the inspector (still present in `data` and the migrator, but no UI to set them), the v3 enumeration→type-picker driver and the role/multiplicity inspector wiring on `data.source.*`/`data.target.*` are mismatched (v4 reads/writes flat `sourceRole`/`sourceMultiplicity`/`targetRole`/`targetMultiplicity`, while the v3 wire form and `convertV4ToV3Class` round-trip via `relationship.source.role` / `relationship.target.role`, which means **data only exists on the inverse migrator output, not on the v3 input that v4 consumes**), the `Interface` palette + `Interface` stereotype radio are stubbed out (acknowledged TODO in code), the ER underline-on-isId for class attributes is missing in the v4 SVG, and the `showIconView` setting hook is not consumed by ClassDiagram in v4 (only ObjectDiagram). Several smaller divergences listed in §C/§D below.

---

## Section A — Palette

### v3 palette (`class-preview.ts:15-201`)

The v3 palette emits these on drag-out:
1. `Class` with one starter attribute (`+ classAttribute: str`).
2. `Class` with one starter attribute + one starter method.
3. `Enumeration` with three starter literals (`enumAttribute_1..3`) and `bounds.height = 140`.
4. `ClassOCLConstraint` with body `"OCL Constraint"`.

`AbstractClass`, `Interface`, and `Package` are commented out (lines 21-27, 87-148).

### v4 palette (`packages/library/lib/constants.ts:385-505`)

| Entry | v3 default name | v3 attributes | v4 default name | v4 attributes | Status |
|-------|----------------|---------------|-----------------|---------------|--------|
| Class (no methods) | `Class` | 1× `+ classAttribute: str` | `Class` | 1× `attribute / public / str` | OK shape; v3 used `translate('sidebar.classAttribute')` so the seed name is i18n-driven; v4 uses literal `"attribute"`. |
| Class (with method) | `Class` | 1 attr + 1 method | `Class` | 1 attr + 1 method | OK |
| AbstractClass | not in v3 palette | n/a | `Abstract` w/ `stereotype: ClassType.Abstract` + 1 attr + 1 method | NEW — restored in v4 (per `class-preview.ts:88-115` which had it commented out). |
| Interface | commented out (`class-preview.ts:118-148`) | n/a | not in v4 palette | OK — both hide; but `StereotypeButtonGroup.tsx:12-18` also explicitly drops Interface. |
| Enumeration | `Enumeration`, h=140, 3 attrs `enumAttribute_1`/`_2`/`_3` (i18n) | `Enumeration` w/ `stereotype: ClassType.Enumeration` h=140, 3 attrs `Enum_1`/`_2`/`_3` | OK shape; literal names changed from `enumAttribute_*` to `Enum_*` (per code comment to ensure Python-identifier validity). |
| ClassOCLConstraint | name=`""`, `constraint: "OCL Constraint"` | `name: "constraint"`, `expression: "context Class inv: true"` | OK shape; defaults differ. |
| Package | commented out (`class-preview.ts:21-27`) | n/a | not in v4 palette (`Package.tsx` exists; `dropElementConfigs.ClassDiagram` omits it) | OK |
| ColorDescription / ColorLegend | v3 has full element + popup but is registered globally, not per-diagram (`packages/components.ts:[UMLElementType.ColorLegend]`); old palette includes it | not in v4 ClassDiagram palette (intentionally hidden — `constants.ts:1340-1354`) | NEW DELTA |

**Mismatches:**
- Enumeration starter literal names (`enumAttribute_*` → `Enum_*`) — intentional fix.
- v3 i18n keys `sidebar.classAttribute` / `sidebar.classMethod` are not used in v4 (literal English strings only).

---

## Section B — Canvas rendering

| Element | v3 component | v4 component | Diff |
|---------|--------------|--------------|------|
| `Class` (no stereotype) | `uml-classifier-component.tsx:17-83` (header band 40, divider line, attributes/methods compartments, ER toggles) | `nodes/classDiagram/Class.tsx:66-235` + `components/svgs/nodes/classDiagram/ClassSVG.tsx:23-146` | Header heights match (40 / 50 with stereotype, both layouts). v3 clips text to bounding rect via `<clipPath>`. v4 doesn't clip — relies on calculated `minWidth` via `measureTextWidth`. v3 hide-methods-compartment for `ER` mode is honoured by `uml-classifier.ts:86-91`; v4 ClassSVG (`isEnumeration` branch only) does NOT honour ER for plain `Class` — methods compartment stays visible in v4 ER mode. **DIFFERENCE: ER hide-methods compartment for plain Class/AbstractClass is lost in v4.** |
| `AbstractClass` | `uml-abstract-class.ts:9-30` — `italic: true`, `stereotype: 'abstract'` (lowercase) | Stereotype = `ClassType.Abstract` ("Abstract", capitalized). `Class.tsx:isItalic = italic ?? stereotype === 'Abstract'` | OK — v4 preserves italic via auto-derivation; freeform stereotype string also preserved. **Case mismatch:** v3 stores `'abstract'` (lowercase, used for header band rendering `«abstract»`); v4 stores `'Abstract'` (capital). The v4 SVG renders `«${stereotype}»` directly (`ClassSVG.tsx:88` via `HeaderSection`), so v4 produces `«Abstract»` instead of v3's `«abstract»`. |
| `Interface` | `uml-interface.ts:8-17` — `stereotype: 'interface'` | Capitalized as `'Interface'` if migrated. Not in palette/`StereotypeButtonGroup`. | Same case-mismatch issue as Abstract. |
| `Enumeration` | `uml-enumeration.ts:9-25` — `stereotype: 'enumeration'`, `connectable: false` | Stereotype = `ClassType.Enumeration` ("Enumeration"). Connection rejected by `useConnect.isConnectionAllowed` → `bpmnConstraints.isEnumerationClassNode` (which checks `data.stereotype === "Enumeration"`, capitalized) | OK. **Sub-issue:** `scaffoldObjectsFromClasses.ts:104,264` checks `stereotype === 'enumeration'` (lowercase) for skipping abstract/interface/enumeration in object scaffolding. Defaults are capitalized in v4, so **scaffolding will create object nodes for Enumerations in v4** — a regression. See gap #2 below. |
| `ClassAttribute` row | `uml-classifier-member-component.tsx:23-104` — uses `displayName`/`displayNameER`, has icon-view branch, has ER underline-on-isId branch | row rendered inside `ClassSVG` via `RowBlockSection` + `formatDisplayName`. No icon-view branch. **No ER underline-on-isId.** | **REGRESSION:** ER notation displays the `id` flag with an underline on the attribute name in v3 (`uml-classifier-member-component.tsx:57-60,86-92`); v4 has no equivalent — `formatDisplayName` strips `{id}` for ER but never sets a text decoration. |
| `ClassMethod` row | same component as attributes (`uml-classifier-member-component.tsx`) | same row component (`RowBlockSection`) | OK shape. v3 hides methods entirely in ER mode (`uml-classifier-member-component.tsx:42-44`); v4 doesn't (already noted). |
| `ClassOCLConstraint` (free-standing sticky note) | `uml-class-ocl-constraint-component.tsx:27-138` — folded-corner shape, badge derived from `context X inv\|pre\|post`, wrapped text | `nodes/classDiagram/ClassOCLConstraint.tsx:84-204` | OK. Behaviour parity is good. v4 uses fixed yellow `#fff8c4` fill default; v3 used theme default. |
| `Package` | `uml-class-package-component.tsx:5-32` — folded-tab on top-left | `nodes/classDiagram/Package.tsx` + `PackageSVG` | OK. |
| `ColorDescription` / `ColorLegend` | `color-legend-component.tsx` (v3) | `ColorDescription.tsx` + `ColorDescriptionSVG.tsx` | Element exists in v4 but is dropped from palette. |

---

## Section C — Update / inspector panels

### `Class` / `AbstractClass` / `Enumeration`

v3: `uml-classifier-update.tsx:138-507` (single component for all three). v4: `ClassEditPanel.tsx:806-1166`.

| Field | v3 | v4 | Status |
|-------|----|-----|--------|
| Name field | `Textfield value={element.name}` (`:208`) sanitised via `rename` (`:475` → `replace(/[^a-zA-Z0-9_]/g,'')`) | `NodeStyleEditor` + `handleDataFieldUpdate('name', …)` with `safeIdentifier` (`:62`) | OK |
| Color editor | `<StylePane fillColor lineColor textColor />` (`:222-225`) | `NodeStyleEditor` (`:1052-1055`) | OK colors. |
| `description` text | `<StylePane showDescription>` (`:219`) | **NOT EXPOSED.** `NodeStyleEditor` only handles colors. | **REGRESSION** |
| `uri` text | `<StylePane showUri>` (`:220`) | **NOT EXPOSED** | **REGRESSION** |
| `icon` upload | `<StylePane showIcon>` (`:221`) | **NOT EXPOSED** | **REGRESSION** |
| Type/stereotype switch | `<Switch>` with `AbstractClass`, `Enumeration` items (`:229-240`) — no `Class` choice (toggle off returns to plain Class) | `<StereotypeButtonGroup>` with `Abstract`, `Enumeration` (Interface omitted, `:15-18`) | OK behavior. Different widget. |
| Attributes section | `attributes.map` w/ `UmlAttributeUpdate` rows + textfield "+ attribute: str" | `nodeData.attributes.map` w/ `<AttributeRow>` rows + add field + explicit "+" IconButton | OK shape. |
| **Attribute reorder up/down** | `<ReorderControls>` per row (`:255-274`) | NOT IMPLEMENTED in v4 | **REGRESSION** |
| Methods section | hidden when `isEnumeration` (`:344`) | `nodeData.stereotype !== 'Enumeration'` guard (`:1108`) | OK |
| **Method reorder up/down** | per-row reorder (`:355-374`) | NOT IMPLEMENTED | **REGRESSION** |
| **OCL constraints UI** | not in v3 update pane (constraints are separate elements) | OCL row code present in v4 but **explicitly NOT rendered** in `ClassEditPanel` (`:1155-1163`) | Spec'd-out. Constraints only edit via the free-standing node inspector. |
| Quick "Code" method button | `<QuickCodeButton onClick={createMethodWithCode}>` `:424-430` | NOT IMPLEMENTED — v4 uses gear-icon to expose code editor on demand | DELTA (v4 has it elsewhere) |

### Attribute row

v3: `uml-classifier-attribute-update.tsx:189-455`. v4: `ClassEditPanel.tsx:147-382`.

| Field | v3 | v4 | Status |
|-------|----|-----|--------|
| Visibility dropdown | 4 options shown as symbols (`+ - # ~`) `:407-412` | 4 options shown as symbols `:201-214` | OK |
| Name | identifier-sanitised text field | identifier-sanitised text field | OK |
| Type dropdown | primitives + enumeration options merged `:421-427` | primitives + classes + enumerations + "custom…" `:231-263` | **DELTA:** v4 includes class names as types (used for object-typed attributes), v3 only included primitives + enum names. v4 better for the BESSER metamodel. |
| isOptional toggle | `<StylePane isOptional onOptionalChange>` always shown when `colorOpen` (`:439-440`) | hidden by `showSettings` gear icon (`:304-358`) | DELTA — v3 shows in style pane, v4 collapses behind gear; conceptually the same but different UX. |
| isDerived toggle | same as above | gear-collapsed | OK |
| isId toggle | same | gear-collapsed | OK |
| isExternalId toggle | same | gear-collapsed | OK |
| defaultValue | `<StylePane defaultValue ... attributeType enumerationLiterals>` — type-aware widget that, when `attributeType` is an Enumeration, drops a selector for that enum's literals | plain text input (`:365-377`) — comment explicitly drops the enumeration-literal picker | **REGRESSION:** v3 offered an enumeration-literal dropdown for `defaultValue` when the type was an enumeration. v4 drops this. |
| Color row | `<StylePane fillColor textColor>` per row | not exposed per-row | **REGRESSION** (per-row colors gone). |
| Delete button | `<TrashIcon>` | `<DeleteIcon>` | OK |
| Reorder | `<ReorderControls>` wrap | not present | **REGRESSION** (already noted) |

### Method row

v3: `uml-classifier-method-update.tsx:175-499`. v4: `ClassEditPanel.tsx:397-740`.

| Field | v3 | v4 | Status |
|-------|----|-----|--------|
| Visibility dropdown | 4 options as symbols `:357-361` | 4 options as symbols `:458-462` | OK |
| Method name | text field; locked in `code/bal` modes (`:363-372`) | text field, identifier-sanitised; not locked in code modes | DELTA — v3 locked the name when a code body existed (auto-extracted from `def name(...)`); v4 lets the user edit independently. |
| **Method auto-rename from code body** | `handleCodeChange` parses `def name(...)` and rewrites `name` (`:256-296`) | **NOT IMPLEMENTED** | **REGRESSION** |
| Return type | implied via name parse (`signature += : returnType`); no separate dropdown | explicit "returns" Select with primitives + classes + custom `:498-542` | DELTA — v4's return-type field is independent of the name (v3 stuffed it in the name). |
| Parameters | implied via name parse | explicit `parameters: ClassifierMethodParameter[]` rows `:548-622` | DELTA — v4 is structured; v3 is name-string. |
| ImplementationType | dropdown (`'none'`/`'code'`/`'bal'`/`'state_machine'`/`'quantum_circuit'`) `:387-392` | same dropdown, gear-collapsed `:626-696` | OK |
| StateMachine select | `<DiagramDropdown>` w/ placeholder + diagrams `:402-410` | `<Select>` w/ placeholder + diagrams `:660-676` | OK; v3 shows "No state machines available" badge when empty (`:411-414`); v4 just shows an empty select. **MINOR REGRESSION** |
| QuantumCircuit select | same with empty-state badge `:419-441` | `<Select>` empty-state-less | same minor regression |
| Code editor | `<CodeMirror>` Python mode `:474-490` | `<CodeMirror>` w/ `python()` extension `:719-733` | OK |
| Color row | per-row `StylePane fillColor textColor` | not exposed per-row | **REGRESSION** (already noted) |
| Reorder | `<ReorderControls>` | not present | **REGRESSION** |

### `ClassAssociation` (edge) update

v3: `uml-class-association-update.tsx:59-206`. v4: `ClassEdgeEditPanel.tsx:69-261`.

| Field | v3 | v4 | Status |
|-------|----|-----|--------|
| Color | `<StylePane lineColor textColor>` | `<EdgeStyleEditor strokeColor textColor>` | OK |
| Flip action | `<ExchangeIcon onClick={flip}>` swaps source/target via `UMLRelationshipRepository.flip` | `<SwapHorizIcon>` swaps `e.source/target/sourceHandle/targetHandle` | OK |
| Delete | yes `:94-96` | not in v4 panel — handled by NodeToolbar | DELTA |
| Name (hidden for inheritance) | `<Textfield>` (`:107-119`) | `<MuiTextField>` (`:154-162`) | OK |
| Edge-type Select | 7 options (Bi, Uni, Aggregation, Comp, Inheritance, Realization, Dependency) but four commented out (`:122-143`) so v3 actually exposes only 3 (Uni, Bi, Comp, Inh) | 7 options, all enabled (`:54-62`) | DELTA — v4 surfaces all 7. v4 hides type select entirely for `ClassOCLLink` / `ClassLinkRel` (auto-detected). |
| Source multiplicity | `Textfield value={element.source.multiplicity}` `:152-159` — placeholder switches `1..1`/`(1,1) or 1..1` for ER | `data.sourceMultiplicity` (flat field, **not nested**) `:191-207` | **CRITICAL** — see gap #1 |
| Source role | `element.source.role` | `data.sourceRole` (flat) | same |
| Target multiplicity | `element.target.multiplicity` | `data.targetMultiplicity` | same |
| Target role | `element.target.role` | `data.targetRole` | same |
| ER cardinality normalize | `erCardinalityToUML(value)` on commit `:198-203` | same on `onBlur` `:201-205,238-242` | OK |
| Style pane: lineColor textColor | yes (`:98-104`) | yes via `EdgeStyleEditor` | OK |

### `ClassOCLConstraint` (sticky-note) update

v3: `uml-class-ocl-constraint-update.tsx:68-156`. v4: `ClassOCLConstraintEditPanel.tsx:20-77`.

| Field | v3 | v4 | Status |
|-------|----|-----|--------|
| `constraint` body textarea | `<StyledTextarea value={element.constraint}>` `:84-89` | `expression` MuiTextField multiline `:53-62` | OK (note field rename: `constraint` → `expression`; the migrator handles the rename `versionConverter.ts:752-754`). |
| `description` textarea | `<DescriptionTextarea>` `:93-99` | `description` MuiTextField multiline `:63-74` | OK |
| Color | `<StylePane lineColor textColor fillColor>` `:108-115` | `NodeStyleEditor` `:48-51` | OK |
| **`name` field** | not in v3 update pane (constraint name lives on `name` but the pane only shows constraint+description) | also not in v4 panel — comment notes it's "not surfaced" `:11-15` | OK consistent. |
| `kind` field | not in v3 | not in v4 panel | OK |

### `ColorLegend` / `ColorDescription` update

v3: `color-legend-update.tsx`. v4: panel hidden (no inspector registration). DELTA but acknowledged in code.

---

## Section D — Edges

### Marker mapping

| Edge type | v3 marker (`uml-association-component.tsx:125-142`) + stroke (`:168-177`) | v4 marker (`edgeUtils.ts:173-247`) | Status |
|-----------|---------------------------------------------------------------|-------------------------------------|--------|
| `ClassBidirectional` | no marker, solid | no marker, solid | OK |
| `ClassUnidirectional` | `Marker.Arrow` end, solid | `markerEnd: url(#black-arrow)`, solid | OK |
| `ClassAggregation` | `Marker.Rhombus` (white) at end (refX=30 on end) | `markerStart: url(#white-rhombus)` | **DELTA — v3 puts diamond on `end` of polyline, v4 puts diamond on `start`.** v3's "Marker" pattern attaches to `markerEnd` because `getMarkerForTypeForUMLAssociation` is wired through `markerEnd` via `<ThemedPolyline markerEnd={…}>`. v4 swaps to `markerStart`. **The visual semantics are inverted** — UML aggregation places the diamond on the *whole* (composite) end. v4's comment claims "diamond on the source (whole) end" — which means the user must draw source = whole. v3's flow is the opposite: the diamond goes on the user-marked end. Concrete impact: round-tripping through `convertV4ToV3Class` will silently swap which class is the whole. |
| `ClassComposition` | `Marker.RhombusFilled` at end | `markerStart: url(#black-rhombus)` | same inversion |
| `ClassInheritance` | `Marker.Triangle` (white) at end | `markerEnd: url(#white-triangle)` | OK |
| `ClassRealization` | `Marker.Triangle` at end + dasharray "7" | `markerEnd: url(#white-triangle)` + dasharray "10" | OK pattern, dasharray differs (`7` vs `10`). |
| `ClassDependency` | `Marker.Arrow` at end + dasharray "7" | `markerEnd: url(#black-arrow)` + dasharray "10" | OK pattern, dasharray differs. |
| `ClassOCLLink` | no marker, dasharray "5,5" | `markerEnd: url(#class-ocl-link-marker)`, dasharray "4 2" | DELTA |
| `ClassLinkRel` | no marker, dasharray "5,5" | not enumerated in `getEdgeMarkerStyles` | **GAP** — `ClassLinkRel` falls through to default styling. |

### Default edge type when drawing

- v3: `componentTypes` registry for ClassDiagram drag-out — most recent default depends on the toolbar selection.
- v4: `getDefaultEdgeType("ClassDiagram") → "ClassBidirectional"` (`edgeUtils.ts:806-807`).

OK; `ClassDiagram` defaulting to `ClassBidirectional` is intentional per the comment.

### Connection validation

- v3: `UMLClass.supportedRelationships` (`uml-class.ts:12-22`) lists 9 types. `UMLEnumeration.features.connectable = false` (`uml-enumeration.ts:13-16`).
- v4: rejects edges whose endpoint has `data.stereotype === "Enumeration"` (`bpmnConstraints.ts:21-26,44-45`). Auto-flips type to `ClassOCLLink` when one endpoint is `ClassOCLConstraint` (`useConnect.ts:26-34`).

Status: OK for the subset v4 implements (Enumeration block + OCL auto-detect). v3 also blocks per-source `supportedRelationships` (e.g. you can't drag a `ClassDependency` between two AbstractClasses). v4 does not enforce per-edge-type allowlists — any non-enum class can connect to any non-enum class with any of the 7 edge types. **GAP** but acceptable for a freer authoring experience.

---

## Section E — Default `data` shape on drop

| Palette entry | v3 (`UMLClass(values)` → element + child elements via `model.create()`) | v4 (`defaultData` in `dropElementConfigs.ClassDiagram`) | Status |
|---------------|-------------------------------------------------------------|----------------------------------------------------|--------|
| Class (no methods) | `UMLClass({name})` + child `UMLClassAttribute({name, owner, visibility:'public', attributeType:'str'})` | `{name:"Class", attributes:[{id, name:"attribute", visibility:"public", attributeType:"str"}], methods:[]}` | OK |
| Class (with method) | as above + `UMLClassMethod({name, owner})` (default visibility `'public'`, attributeType `'str'`) | same with `methods:[{id, name:"method", visibility:"public", attributeType:"any", returnType:"any", parameters:[], implementationType:"none"}]` | DELTA: v3 default method has `attributeType:"str"`; v4 has `attributeType:"any" returnType:"any"`. v4 is more correct (methods don't have a type, only a return type). |
| Abstract | not in palette | `{name:"Abstract", stereotype:"Abstract", attrs/methods like Class}` | NEW |
| Enumeration | `UMLEnumeration({name, bounds:{height:140}})` + 3 child `UMLClassAttribute` with `name='enumAttribute_N'` and **no visibility/attributeType set** | `{name:"Enumeration", stereotype:"Enumeration", attributes: [{id, name:"Enum_1"}, {id, name:"Enum_2"}, {id, name:"Enum_3"}], methods:[]}` | OK semantics. v4 omits `visibility/attributeType` on enum literals; v3 also omits these (just sets `name`). |
| ClassOCLConstraint | `ClassOCLConstraint({constraint:"OCL Constraint"})` — note: NO `name`, NO `description` | `{name:"constraint", expression:"context Class inv: true"}` | DELTA: v3 default body `"OCL Constraint"` (placeholder); v4 default body is a valid OCL fragment `"context Class inv: true"`. |

**Findings:**
- v4 Class entries call `generateUUID()` once per *module load* not per drop, so every drop reuses the same row IDs. That is a real bug — see gap #5.
- v3 omitted Enumeration `attributeType` defaults; v4 mirrors that.

---

## Section F — Migrator round-trips

`convertV3ToV4` (`versionConverter.ts:679-738`) handles `Class | AbstractClass | Interface | Enumeration` by collapsing to v4 `node.type === "class"` with `data.stereotype` set; child `ClassAttribute` / `ClassMethod` / `ClassOCLConstraint` elements are folded onto `data.attributes` / `data.methods` / `data.oclConstraints`.

`convertV4ToV3Class` (`versionConverter.ts:2223-2483`) does the inverse: emits `Class` / `AbstractClass` / `Interface` / `Enumeration` based on `data.stereotype` (`:2228-2233`), re-emits child elements (`:2278-2303`), and emits a free-standing `ClassOCLConstraint` for v4 `node.type === "ClassOCLConstraint"` (`:2367-2397`).

| Element | Round-trip check | Notes |
|---------|------------------|-------|
| `Class` | v3 → v4 → v3 | OK (stereotype freeform fall-through `:709-712`). |
| `AbstractClass` | v3 (`stereotype: 'abstract'`) → v4 (`stereotype: 'Abstract'`) → v3 (`type: 'AbstractClass'`, no `stereotype`) | **CASE BUG**: v3 stores `'abstract'` lowercase; v4 stores `'Abstract'` capitalized; v4→v3 emits `type:'AbstractClass'` (correct). The `«Abstract»` band on canvas will display capitalized in v4 but lowercase in v3. Cosmetic but observable on round-trip. |
| `Interface` | same case-shift | same |
| `Enumeration` | same case-shift | same; **scaffold bug downstream**: `scaffoldObjectsFromClasses.ts:104,264` checks lowercase, breaks. |
| `ClassAttribute` | new format passes through (`extractClassifierMember:543-575`); legacy parses via `parseLegacyNameFormat` (`:580-596`) and rewrites canonical fields | OK |
| `ClassMethod` | same as attribute | OK; v3 `code/implementationType/stateMachineId/quantumCircuitId` preserved; v4-only `parameters[]` and `returnType` are NOT round-tripped to v3 (`childRowToV3:2492-2531` does not emit them — v3 element type doesn't have them). **DATA LOSS if user edits `parameters[]` then exports v3.** |
| `ClassOCLConstraint` (free-standing) | v3 `constraint` field → v4 `data.expression`; v4 → v3 `constraint`; `description` and `kind` round-trip | OK |
| `ClassOCLConstraint` (owned by class) | v3 child element with `owner === classId` → collapsed onto v4 `data.oclConstraints`; v4 → re-emit as v3 child element with `owner: classNodeId` (`:2284-2303`) | OK |
| `Comments` (sticky notes) | v3 `Comments` → v4 `comment` (`versionConverter.ts:2540`). | Out of scope but listed for completeness. |
| `ColorLegend` / `ColorDescription` | v3 `ColorLegend` → v4 `colorDescription` (`nodes/types.ts:79`); v4→v3 inverse is via `invertNodeType:2533-2591` which has `comment: "Comments"` but does NOT list `colorDescription`. So v4 `colorDescription` round-trips as v3 element with `type: "colorDescription"` (wrong — should be `ColorLegend`) | **GAP** — see gap #6 below. |

---

## Section G — Settings hooks

| Setting | v3 consumer (file:line) | v4 consumer | Wired? |
|---------|-------------------------|-------------|--------|
| `classNotation: 'UML' \| 'ER'` | `uml-classifier.ts:87` (drops methods compartment in ER); `uml-classifier-component.tsx:20` (no methods divider in ER); `uml-classifier-member-component.tsx:32-92` (ER underline-on-isId; hide methods); `uml-association-component.tsx:149-164` (ER midpoint diamond + cardinality); `uml-class-association-update.tsx:81` (placeholder hint) | `useClassNotation()` consumed in `nodes/classDiagram/Class.tsx:77` (formats display via `formatDisplayName`) and `inspectors/classDiagram/ClassEdgeEditPanel.tsx:76` (placeholder hint). `formatDisplayName` honours UML/ER. | PARTIAL — see gaps. ER underline-on-isId not implemented (`utils/classifierMemberDisplay.ts` has no underline output). ER hide-methods-compartment for plain Class/AbstractClass not implemented in `ClassSVG.tsx`. ER midpoint diamond on `ClassDiagramEdge` not implemented. |
| `showAssociationNames` | `uml-association-component.tsx:148,198-209` | `edges/edgeTypes/ClassDiagramEdge.tsx:111-118,207-224` | OK |
| `showInstancedObjects` | `ObjectName` only (not Class) | `ObjectNameSVG.tsx:56-59` (object-only) | OK (n/a to ClassDiagram) |
| `showIconView` | `uml-classifier-member-component.tsx:31,36-38` (Class **and** Object members hidden in icon view) | `ObjectNameSVG.tsx:44-46` only — no class-row consumer | **REGRESSION**: ClassAttribute/Method icon-view hide-rows is missing. |

---

## Section H — Webapp scaffold

`scaffoldObjectsFromClasses.ts` walks `classModel.nodes[]` directly:
- `isConcreteClassNode` rejects `stereotype.toLowerCase() ∈ {abstract, interface, enumeration}` (`:264`).
- `firstEnumLiteral` matches `data.stereotype === 'enumeration'` (lowercase, `:104`).

v4's `ClassType.Enumeration` is `'Enumeration'` (capitalized). **Result: enumerations are not skipped during scaffolding** — the helper will emit `objectName` instances of every Enumeration, and `firstEnumLiteral` will fail to find them when seeding values for enum-typed attributes. **CRITICAL GAP** — the toLowerCase compare in `isConcreteClassNode` actually saves it for the rejection branch, but `firstEnumLiteral` does NOT lowercase the comparison, so enum-typed attribute defaults regress. See gap #2.

Otherwise the scaffold reads `data.attributes[].name`, `data.attributes[].attributeType`, `data.attributes[].defaultValue`, `data.attributes[].id` — these all match the v4 `ClassNodeElement` shape produced by the migrator and palette.

---

## Critical gaps (severity-ranked)

1. **CRITICAL — Edge inspector reads/writes flat `data.sourceMultiplicity` / `data.sourceRole` / `data.targetMultiplicity` / `data.targetRole` but the v3 wire form (and `convertV4ToV3Class`) round-trip these via `relationship.source.multiplicity` / `relationship.source.role` / `relationship.target.multiplicity` / `relationship.target.role`.**
   - Files: `packages/library/lib/components/inspectors/classDiagram/ClassEdgeEditPanel.tsx:191-256`; `packages/library/lib/utils/versionConverter.ts:2430-2451` (the inverse migrator already nests them under `source` / `target`).
   - v3 has: `<Textfield value={element.source.multiplicity} onChange={onUpdate('multiplicity', 'source')}>` (`uml-class-association-update.tsx:152-159`) and `update<UMLAssociation>(element.id, { [end]: { ...element[end], [type]: storedValue } })` (`:202`).
   - v4 has: `value={data.sourceMultiplicity ?? ""}` (`ClassEdgeEditPanel.tsx:197`) — flat field on edge `data`.
   - Inspector reads/writes `data.sourceMultiplicity`, but `convertV3ToV4` for Class edges does not produce that field — so after a v3→v4 import, **the inspector is empty even though the multiplicity exists on `relationship.source.multiplicity`**. The migrator (`convertV3ToV4`) for class edges does not lift `source.multiplicity → data.sourceMultiplicity`. Confirm this by inspecting `convertV3ToV4`'s edge-emit branch (the `Class*` cases just pass `edge.type` through without lifting endpoint fields).
   - Action: in `versionConverter.ts` add to the v3→v4 edge migration: `data.sourceMultiplicity = rel.source?.multiplicity; data.targetMultiplicity = rel.target?.multiplicity; data.sourceRole = rel.source?.role; data.targetRole = rel.target?.role;`.

2. **CRITICAL — Scaffold helper uses lowercase `'enumeration'` while v4 stereotype is `'Enumeration'`.**
   - File: `packages/webapp/src/main/features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts:104` (`isEnumNode = n.type === 'class' && data.stereotype === 'enumeration'`) and similar `:264`.
   - v4 emits `ClassType.Enumeration` = `'Enumeration'` (`packages/library/lib/types/nodes/enums/ClassType.ts:4`).
   - Action: change the comparisons to `(data.stereotype || '').toString().toLowerCase() === 'enumeration'` (already done in `:264` for `isConcreteClassNode`) — apply the same to `:104` `firstEnumLiteral` and `:108` `isLegacyEnum` line.

3. **CRITICAL — `parameters[]` and `returnType` data loss on round-trip.**
   - File: `packages/library/lib/utils/versionConverter.ts::childRowToV3:2492-2531`.
   - v4 method rows carry `parameters: ClassifierMethodParameter[]` and `returnType: string` (added per the inspector at `ClassEditPanel.tsx:498-622`). The inverse migrator does not emit them onto the v3 `ClassMethod` element, so saving back to v3 wire form drops the parameter list and return type. On re-import, v4 falls back to `parseLegacyNameFormat` against `name`, which only recovers a single inline return type and no params.
   - Action: serialize `parameters` and `returnType` onto the v3 `ClassMethod` element (use a BESSER-only field, e.g. `parameters` array and `returnType` string). Adjust `extractClassifierMember` to read them back on import.

4. **HIGH — `description` / `uri` / `icon` style fields are not editable in v4.**
   - Files: v3 `<StylePane showDescription showUri showIcon>` (`uml-classifier-update.tsx:219-221`); v4 `NodeStyleEditor` only handles colors (`packages/library/lib/components/ui/StyleEditor/NodeStyleEditor.tsx`).
   - Migrator preserves `description`/`uri`/`icon` on `data` (`versionConverter.ts:732-734`), but the user has no way to set or view them in v4.
   - Action: add a "Documentation" expandable panel to `ClassEditPanel.tsx` (between `StereotypeButtonGroup` and the Attributes header) with three text fields bound to `data.description`, `data.uri`, `data.icon`.

5. **HIGH — Palette `defaultData` row IDs are module-scope constants, so every drop reuses the same UUID.**
   - File: `packages/library/lib/constants.ts:402,422,433,453,464,484-486,489,503` — `id: generateUUID()` is called inside the literal, evaluated at module load. The resulting `defaultData` object is then frozen and shared across every drop.
   - Drop a class twice → both new nodes have an attribute row with the same UUID → React-Flow row tracking is broken.
   - Action: change `defaultData` to a factory: e.g. `defaultData: () => ({ ... id: generateUUID() ... })` and have the drop site invoke it on each drop. Alternatively, materialise IDs at the drop call site.

6. **MEDIUM — `colorDescription` reverse mapping missing in `invertNodeType`.**
   - File: `packages/library/lib/utils/versionConverter.ts::invertNodeType:2533-2591` does not list `colorDescription → ColorLegend`.
   - v4→v3 export will emit `type: "colorDescription"` instead of `type: "ColorLegend"`, breaking any v3-only tooling that reads the colour legend.
   - Action: add `colorDescription: "ColorLegend"` to the `invertNodeType` map (and confirm the forward map at `:201-373`).

7. **MEDIUM — ER hide-methods compartment for plain `Class` / `AbstractClass` is missing.**
   - File: v3 `uml-classifier.ts:86-91` zeros `hasMethods` in ER mode; `uml-classifier-component.tsx:78-79` skips the methods divider; `uml-classifier-member-component.tsx:42-44` returns `null` for class methods in ER.
   - v4 `ClassSVG.tsx:121` only hides the methods compartment for `isEnumeration`, not for ER mode. `formatDisplayName` does not have a hide branch.
   - Action: in `Class.tsx` and/or `ClassSVG.tsx`, gate the methods compartment on `useClassNotation() === 'UML' || stereotype === 'Interface'` to mirror v3.

8. **MEDIUM — ER underline-on-isId for class attributes is missing.**
   - File: v3 `uml-classifier-member-component.tsx:60,86-92` underlines the row when `isId && ER`. v4 `formatDisplayName` strips `{id}` for ER but never sets a text decoration.
   - Action: extend `RowBlockSection` (or `formatDisplayName`'s caller) to forward an `underline: boolean` flag derived from `isId && classNotation === 'ER'`, and apply `textDecoration="underline"` to the row text in ClassSVG.

9. **MEDIUM — ER midpoint diamond on edges is missing in v4.**
   - File: v3 `uml-association-component.tsx:158-234` draws an ER-flavored named diamond at the midpoint and replaces the UML arrow markers when `classNotation === 'ER'` and edge ∈ {Bi, Uni, Aggregation, Composition}.
   - v4 `ClassDiagramEdge.tsx` reads `showAssociationNames` but does not branch on classNotation for ER rendering.
   - Action: in `ClassDiagramEdge.tsx`, when `classNotation === 'ER'` and edge type is one of the four ER-capable ones, suppress `markerEnd`/`markerStart` and render an SVG `<polygon points="-30,0 0,-15 30,0 0,15">` at `pathMiddlePosition`.

10. **MEDIUM — ClassAggregation / ClassComposition diamond placement is on the start (source) end in v4 vs the end (target) in v3.**
    - File: `packages/library/lib/utils/edgeUtils.ts:195-210` — `markerStart` for both.
    - v3 `uml-association-component.tsx:131-134,189-198` — diamond is on `markerEnd` of the polyline.
    - The visual semantics matter (UML aggregation: diamond on the "whole" end). v4's comment claims "diamond on the source (whole) end"; v3's convention is target = whole.
    - Action: decide on convention (recommend v3 = end), and either flip `markerStart` → `markerEnd` here, or flip the source/target on import in `convertV3ToV4` (less invasive). If the latter, update `convertV4ToV3Class` to swap back.

11. **MEDIUM — `showIconView` setting has no ClassDiagram consumer in v4.**
    - File: v3 `uml-classifier-member-component.tsx:31,36-38` hides Class members under icon view; v4 only gates ObjectName.
    - Action: extend `Class.tsx` / `ClassSVG.tsx` to honour `useSettingsStore((s) => s.showIconView)` — drop attribute and method rows when icon view is active.

12. **LOW — Per-row reorder up/down buttons are missing.**
    - Files: v3 `uml-classifier-update.tsx:255-274,355-374` — `<ReorderControls>` per attribute and method.
    - Action: add up/down IconButtons to `AttributeRow` / `MethodRow` in `ClassEditPanel.tsx`; reorder by mutating `data.attributes` / `data.methods` arrays.

13. **LOW — Per-row color editor missing.**
    - v3 attribute and method rows expose `<StylePane fillColor textColor>` per-row (`uml-classifier-attribute-update.tsx:434-443`).
    - Action: add color affordance to v4 `AttributeRow` / `MethodRow` (or accept the regression — class members rarely need per-row colors).

14. **LOW — `defaultValue` enumeration-literal picker dropped in v4.**
    - v3 `uml-classifier-attribute-update.tsx:447-450` — when `attributeType` is an Enumeration name, `<StylePane defaultValue ... enumerationLiterals>` shows a dropdown of that enum's literals.
    - v4 just shows a free-text input (`ClassEditPanel.tsx:365-377`).
    - Action: detect when `attributeType ∈ enumerationNames` (already collected in `ClassEditPanel.tsx:42-54`) and render a `Select` of literals instead of a `MuiTextField`.

15. **LOW — Method auto-rename from code-body is dropped.**
    - v3 `uml-classifier-method-update.tsx:256-296` parses `def name(args): return_type` from the code editor and rewrites the row name.
    - v4: no equivalent.
    - Action: optional — port the regex into `MethodRow` `onChange` for the code editor.

16. **LOW — Empty-state badges for state-machine / quantum-circuit selectors removed.**
    - v3 shows "No state machines available" `<DiagramRefLabel>` `:411-414`.
    - v4: shows an empty `<Select>` with placeholder.
    - Action: trivial UX fix.

17. **LOW — Stereotype case mismatch (`'Abstract'` vs `'abstract'`).**
    - v3 banner reads `«abstract»` (lowercase); v4 reads `«Abstract»` because `ClassType.Abstract === 'Abstract'`.
    - Cosmetic but visible on every Abstract class.
    - Action: in `ClassSVG.tsx` (or `HeaderSection`), lowercase the stereotype string when rendering, OR change `ClassType` enum to lowercase — pick one and document.

18. **LOW — `Interface` palette + stereotype option hidden.**
    - v3: commented out in `class-preview.ts` and `uml-classifier-update.tsx:233-236`.
    - v4: explicitly excluded in `StereotypeButtonGroup.tsx:12-18` and `dropElementConfigs.ClassDiagram`.
    - Action: leave hidden until the round-trip is wired (acknowledged TODO in code).

19. **LOW — `ClassLinkRel` not enumerated in `getEdgeMarkerStyles` switch.**
    - File: `packages/library/lib/utils/edgeUtils.ts:173-300` — no `case "ClassLinkRel":`.
    - Falls through to the `default` branch with no markers / strokes.
    - Action: either add a case (matching v3's "no marker, dasharray '5,5'") or remove the type from the inspector.

---

## Sign-off

PARTIAL — ClassDiagram parity is functionally usable for canvas + drag-out + basic round-trip, but inspector regressions (gap #1 multiplicity wiring, gap #4 description/uri/icon) and migrator data loss (gap #3 parameters/returnType) are real and need fixes before declaring v4 ready to retire v3.
