# Wave 2 Parity Audit (post-SA-3)

Audit date: 2026-05-08
Audited diagrams: ClassDiagram, ObjectDiagram, StateMachineDiagram
Skipped (work in progress): AgentDiagram (SA-4), UserDiagram (SA-4), NNDiagram (SA-5)
Auditor branch: `claude/refine-local-plan-sS9Zv` (read-only)
Submodule HEAD audited: `96d4e88` (`feat(library): SA-3 StateMachineDiagram port`); SA-4 mid-flight files in submodule working tree were considered out of scope but spot-checked for non-interference with SA-1..SA-3.

## Top-line verdict

The three ported diagrams are **structurally faithful** for the common-path round-trip but ship with five concrete parity gaps. The Class diagram drops two relationship types (`ClassLinkRel`, `ClassOCLLink`) that v3 supported as drawable edges and never re-implements its dedicated **Class Association inspector** (multiplicity / role / flip / association-type swap). The Object diagram silently loses `UMLObjectLink.associationId` because no field exists for it on the v4 edge data shape. State-machine parity is the closest of the three — every v3 node type made it across, the inspector covers all v3 edit paths, and the SA-3 `code`/`eventName` additions are pure extensions. A handful of details (v3-lowercase stereotype values vs v4 `ClassType` PascalCase, classifier kind toggle vs new free-text stereotype dropdown) are migrator-handled today but represent latent risks if the migrator's enum mapping is bypassed.

**Critical issues to land before cutover:**

1. `besser/utilities/web_modeling_editor/frontend/packages/library/lib/edges/types.tsx` — `ClassLinkRel` and `ClassOCLLink` are absent from `defaultEdgeTypes`, `edgeConfig`, and `convertV3EdgeTypeToV4`. **CRITICAL** for any v3 fixture that drew an OCL link edge or used the link-rel arrow.
2. `besser/utilities/web_modeling_editor/frontend/packages/library/lib/types/nodes/NodeProps.ts` — `ObjectNodeProps` and the underlying edge data have **no `associationId` field**; `versionConverter.ts:1240..1260` does not propagate it either. **CRITICAL** for ObjectDiagrams that pin an `ObjectLink` to a specific `ClassDiagram` association.
3. No **ClassDiagramEdge inspector** is registered anywhere under `lib/components/inspectors/classDiagram/`. The v3 fork's `uml-class-association-update.tsx` exposed `multiplicity`/`role` on each end, association-type swap, and a flip control — **none** of that is reachable from the new properties panel. **CRITICAL** for editing flow.
4. No **ObjectLink inspector** either; the v3 form let users pick the source `associationId` from `diagramBridge.getAvailableAssociations`. **MEDIUM** (round-trip preserves whatever wire data is set, but users can't author it).
5. v3 `UMLObjectLink` carried the `associationId` field at the relationship level (see point 2). v4 edge data type is missing this entirely.

---

## ClassDiagram

### A. Element type inventory

Old fork (`packages/editor/src/main/packages/uml-class-diagram/index.ts`):

- **Nodes (`ClassElementType`)**: `Package`, `Class`, `AbstractClass`, `Interface`, `Enumeration`, `ClassAttribute`, `ClassMethod`, `ClassOCLConstraint`
- **Edges (`ClassRelationshipType`)**: `ClassBidirectional`, `ClassUnidirectional`, `ClassInheritance`, `ClassRealization`, `ClassDependency`, `ClassAggregation`, `ClassComposition`, `ClassOCLLink`, `ClassLinkRel`

New lib (`packages/library/lib/nodes/classDiagram/index.ts`, `lib/edges/types.tsx`):

- **Nodes**: `class`, `package`, `colorDescription` (helper)
- **Edges**: `ClassBidirectional`, `ClassUnidirectional`, `ClassInheritance`, `ClassRealization`, `ClassDependency`, `ClassAggregation`, `ClassComposition`

| v3 type | v4 status | Mapping notes |
|---|---|---|
| `Class` | RENAMED | `node.type === 'class'`, no stereotype field |
| `AbstractClass` | RENAMED → `class` + `data.stereotype = 'Abstract'` | Migrator at `versionConverter.ts:584-591` sets `ClassType.Abstract` |
| `Interface` | RENAMED → `class` + `data.stereotype = 'Interface'` | Same path |
| `Enumeration` | RENAMED → `class` + `data.stereotype = 'Enumeration'` | Same path |
| `Package` | = | New node type `package` |
| `ClassAttribute` | RENAMED → row inside `data.attributes` | Per spec (`uml-v4-shape.md`) |
| `ClassMethod` | RENAMED → row inside `data.methods` | Per spec |
| `ClassOCLConstraint` | RENAMED → row inside `data.oclConstraints` on owning Class | `versionConverter.ts:523-580`. Free-standing v3 OCL constraints lose their owner pointer if they had no `owner` set on import. |
| `ClassBidirectional` | = | |
| `ClassUnidirectional` | = | |
| `ClassInheritance` | = | |
| `ClassRealization` | = | |
| `ClassDependency` | = | |
| `ClassAggregation` | = | |
| `ClassComposition` | = | |
| `ClassOCLLink` | **MISSING in new** | No edge type, no migrator entry. v3 fixtures with this edge will be dropped or re-typed by the fallback (`return v3Type.toLowerCase()`). |
| `ClassLinkRel` | **MISSING in new** | Same as above. |

Stereotype value-shape mismatch (latent): v3 stores the literal text `'abstract' / 'enumeration' / 'interface'` (all lowercase) in `UMLClassifier.stereotype`. v4 uses `ClassType.Abstract / Interface / Enumeration` (PascalCase, see `lib/types/nodes/enums/ClassType.ts`). The migrator translates by element type, not by string, so a v3 fixture with a non-default stereotype (e.g. someone hand-edited it to `'service'`) survives unchanged but a fixture relying on the lowercase canonical names will round-trip through the migrator without issue. Round-trip mid-export uses `versionConverter.ts:1858-1862` to resolve back. Acceptable but worth a regression test.

### B. Per-type data field parity

#### Class / AbstractClass / Interface / Enumeration → `class` node

Old `IUMLClassifier` (`uml-classifier.ts`):

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| `italic` | (none on `ClassNodeProps`) | **MISSING in new** — derivable from stereotype but not stored. v3 stored explicitly. |
| `underline` | (none on `ClassNodeProps`) | **MISSING in new** |
| `stereotype: string \| null` | `data.stereotype?: ClassType` | RENAMED + tighter type. v3 free text not preserved. |
| `deviderPosition` | — | DROP (recomputed at render time, OK) |
| `hasAttributes` / `hasMethods` | — | DROP (recomputed) |
| `className?` | `data.className?` (only on `ObjectNodeProps`/`UserModelNameNodeProps`) | Not used by Class — was only meaningful for object instances |
| `attributes` (id list of children) | `data.attributes: ClassNodeElement[]` (inline rows) | RENAMED (collapsed) |
| `methods` (id list of children) | `data.methods: ClassNodeElement[]` (inline rows) | RENAMED (collapsed) |
| (no field) | `data.oclConstraints?: ClassOCLConstraint[]` | EXTRA in new (collapsed v3 elements) |
| `fillColor` / `strokeColor` / `textColor` | `data.fillColor` etc | = |

#### `ClassAttribute` (was its own element; now a row)

Old `IUMLClassifierMember` (`uml-classifier-member.ts:64-76`):

| Old field | New field | Status |
|---|---|---|
| `name` | `name` | = |
| `code` | `code` | = |
| `visibility: Visibility` (default `'public'`) | `visibility?: ClassifierVisibility` | ≠ optional in new (v3 had a default) |
| `attributeType: string` (default `'str'`) | `attributeType?: string` | ≠ optional in new (default not enforced server-side) |
| `implementationType: MethodImplementationType` (default `'none'`) | `implementationType?: ClassifierMethodImplementationType` | ≠ optional |
| `stateMachineId: string` (default `''`) | `stateMachineId?: string` | ≠ optional |
| `quantumCircuitId: string` (default `''`) | `quantumCircuitId?: string` | ≠ optional |
| `isOptional: boolean` (default `false`) | `isOptional?: boolean` | ≠ optional |
| `isDerived: boolean` | `isDerived?: boolean` | ≠ optional |
| `isId: boolean` | `isId?: boolean` | ≠ optional |
| `isExternalId: boolean` | `isExternalId?: boolean` | ≠ optional |
| `defaultValue: any` | `defaultValue?: unknown` | = (semantics) |

#### `ClassMethod` (was its own element; now a row)

Same shape as `ClassAttribute` (`UMLClassifierMember` covers both). New lib adds:

| Old field | New field | Status |
|---|---|---|
| (parameters parsed inline from `name` like `"+ foo(p: int): bool"`) | `parameters?: ClassifierMethodParameter[]` | EXTRA in new — structured |
| (return type parsed inline from `name`) | `returnType?: string` | EXTRA in new — structured (also mirrored to `attributeType` in `ClassEditPanel.tsx:319`) |

Backwards: parsing of legacy `name` formats is preserved by the migrator's fallback path — see `ClassEditPanel.tsx:296-308` style references. Round-trip writes back the legacy `displayName` shape via `formatDisplayName`.

#### `ClassOCLConstraint` (collapsed onto owner Class)

Old (`uml-class-ocl-constraint.ts:8-13`):

| Old field | New field | Status |
|---|---|---|
| `name` | `name` | = |
| `constraint: string` | `expression: string` | RENAMED |
| `description?: string` | `description?: string` | = |
| (no field) | `kind?: string` | EXTRA in new (`'invariant'`/`'pre'`/`'post'` per type comment) |

Per-edge OCL link (`UMLClassOCLLink`) is dropped (see Section A). OCL constraint visual rail is **rendered into the class card** in v4 instead of as a separate node; no free-form positioning anymore. This is the SA-2 design choice (per the spec).

#### `Package`

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| (no children typed) | (any node owned via `parentId`) | = |

#### Class relationships (`UMLAssociation` superclass)

Old `IUMLAssociation`:

| Old field | New field (on edge `data`) | Status |
|---|---|---|
| `name` | `name` / `label` | = |
| `source.element` / `target.element` | `source` / `target` | = (React-Flow native) |
| `source.direction` | `sourceHandle` (mapped via `convertV3HandleToV4`) | RENAMED |
| `source.multiplicity` | `data.sourceMultiplicity` | = (string) |
| `source.role` | `data.sourceRole` | = |
| `target.multiplicity` | `data.targetMultiplicity` | = |
| `target.role` | `data.targetRole` | = |
| `path` | `data.points` | = (offset to absolute) |
| `isManuallyLayouted` | `data.isManuallyLayouted` | = |

### C. Panel inspector parity

#### ClassEditPanel (`Class` node)

Old: `uml-classifier-update.tsx` (parent class) + `uml-classifier-attribute-update.tsx` (per-attribute) + `uml-classifier-method-update.tsx` (per-method)

| Old form field | Old field type | New form field | New field type | Status |
|---|---|---|---|---|
| Class name | text (sanitised `[^a-zA-Z0-9_]`) | `data.name` | text (sanitisation present in `addAttribute` only — class name handler doesn't sanitize) | ≠ regression — class-name field accepts arbitrary chars, see `ClassEditPanel.tsx:680` |
| `Switch` between Class / AbstractClass / Enumeration (Interface commented out) | radio-style switch | `StereotypeButtonGroup` with Abstract / Interface / Enumeration | toggle button group | ≠ — Interface is **enabled** in new, was disabled in old. Old toggled element `type`; new only toggles `stereotype` data field. |
| Color/fill/stroke/text via `StylePane` + `description`, `uri`, `icon` extras | nested style pane | `NodeStyleEditor` | inline color editor | ≠ — `description`, `uri`, `icon` extras (`uml-classifier-update.tsx:219-221`) are **MISSING in new** ClassEditPanel |
| Attribute reorder (up/down arrows on each row) | buttons | (none) | — | **MISSING in new** |
| **Per-attribute form** | | | | |
| Visibility dropdown `+ - # ~` | dropdown (4 entries) | `VISIBILITIES` Select with `+ public / - private / # protected / ~ package` | MUI Select | = (value/symbol mapping match) |
| Attribute name field | textfield, sanitises `[^a-zA-Z0-9_]` | text, same sanitiser | MUI text | = |
| `attributeType` dropdown | 9 primitive entries: `str, int, float, bool, date, datetime, time, timedelta, any` + enumerations (auto-discovered) + custom-by-name | Same 9 primitive entries + class-name suggestions + `__custom__` sentinel | MUI Select | ≠ — old offers enumerations (literal types from any local Enumeration); new offers class names instead. **Enumeration types are dropped from the type picker.** |
| `isOptional` checkbox | checkbox | `isOptional` checkbox | = | = |
| `isDerived` checkbox | = | `isDerived` checkbox | = | = |
| `isId` checkbox + auto-clears `isOptional` | = | same constraint | = | = |
| `isExternalId` checkbox + same constraint | = | same | = | = |
| `defaultValue` textfield | text | text | = | = |
| **Per-method form** | | | | |
| Visibility dropdown | dropdown | dropdown | = | = |
| Method name (or signature when code is locked) | text, locked when implType is `code`/`bal` | text, **never locked**; user can edit signature even when code is set | MUI text | ≠ regression in new |
| Implementation type dropdown: `none/code/bal/state_machine/quantum_circuit` (5 entries) | dropdown | same 5 entries | MUI Select | = |
| Code editor (CodeMirror with Python mode) when impl is `code`/`bal` | embedded CodeMirror | `MuiTextField multiline` plain | text area | ≠ — **CodeMirror dropped**, now plain text. No syntax highlighting; tabs / indent helpers gone. |
| State Machine selector when impl is `state_machine` | dropdown of `availableStateMachines` from bridge | dropdown of `diagramBridge.getStateMachineDiagrams()` | MUI Select | = |
| Quantum Circuit selector when impl is `quantum_circuit` | dropdown of `availableQuantumCircuits` from bridge | dropdown of `getQuantumCircuitDiagrams()` | MUI Select | = |
| **Method-only fields not in v3** | | | | |
| (n/a) | — | Return-type dropdown (separate from `attributeType`) | MUI Select | EXTRA in new |
| (n/a) | — | `parameters: ClassifierMethodParameter[]` rows | structured | EXTRA in new |
| **OCL constraint rows (collapsed onto Class)** | | | | |
| (no per-class form; OCL was a separate node) | — | `OCLConstraintRow` with `name` + `expression` multi-line | MUI text + multiline | EXTRA in new |

#### ClassEdgeEditPanel — **MISSING ENTIRELY**

The old fork had `uml-class-association-update.tsx` (160+ lines) wired into the popup system; it exposed: the line color, **flip** action, **delete** action, association-type dropdown (`Unidirectional / Bidirectional / Composition / Inheritance`), name, and per-end **multiplicity + role** fields. The new lib has **no** registered inspector for any of `ClassBidirectional / ClassUnidirectional / ClassInheritance / ClassRealization / ClassDependency / ClassAggregation / ClassComposition`. The edge data shape (`sourceMultiplicity`, `targetMultiplicity`, `sourceRole`, `targetRole`) is set up to receive these values, but there is no UI to author them.

**CRITICAL** for editing-flow parity.

### D. Constraints + invariants

| Constraint | v3 enforcement | v4 enforcement | Severity |
|---|---|---|---|
| `isId` ↔ `isOptional` mutual exclusion | `uml-classifier-attribute-update.tsx:341-353`, also enforced in old setter | `ClassEditPanel.tsx:218-222`, `233-238` | preserved (= identical logic) |
| `isExternalId` ↔ `isOptional` mutual exclusion | same (line 355-367) | preserved (`ClassEditPanel.tsx:233-238`) | OK |
| Method `name` sanitisation | none in v3 (free-form including `()` and `:`) | none in v4 | OK |
| Attribute `name` sanitisation `[^a-zA-Z0-9_]` | enforced in attribute-update form | enforced in `AttributeRow` and `addAttribute` | OK |
| Class `name` sanitisation `[^a-zA-Z0-9_]` | enforced in `uml-classifier-update.tsx:475` | **NOT enforced** in `ClassEditPanel.tsx:680`'s `handleDataFieldUpdate` (the shared `NodeStyleEditor`). | MEDIUM regression |
| Enumeration prevents method creation | `uml-classifier-update.tsx:442-444` | New panel always shows Methods section regardless of stereotype | MEDIUM regression |
| `Class.supportedRelationships` excludes `ClassRealization` (Realization only on Interface) | static array on `UMLClass` | new edges allow any node-to-node connection — handle/connection rules are the only check | MINOR (could let users draw a Realization between two plain Classes) |
| `Enumeration` is `connectable: false` | `uml-enumeration.ts:14-17` | new lib treats it as a `class` stereotype with no special-case at the React-Flow handle level | MEDIUM (a v4 user can connect any class regardless of stereotype) |
| Multiplicity placeholder `1..1` (or `(1,1) or 1..1` in ER mode) | `uml-class-association-update.tsx:80-82` | n/a (no edge inspector at all) | covered by edge-inspector gap |
| `normalizeType` to canonical types (`String → str`, etc.) | `uml-classifier-attribute-update.tsx:108-113` | preserved (`utils/typeNormalization.ts`, called from both attr and method commit) | OK |
| Default attribute type `'str'` | enforced in setter | enforced at `addAttribute` site (`ClassEditPanel.tsx:722`) | OK |

### E. Visual shape parity

#### Class (Class / AbstractClass / Interface / Enumeration body)

- **Old** (`uml-classifier-component.tsx`): Header rect (40 / 50 px depending on stereotype) with bold name + optional `«stereotype»` tspan above; horizontal divider after attributes (`hasAttributes`) and after methods (`hasMethods`); themed via `ThemedRect`. Border drawn outside a clipPath so members never overflow.
- **New** (`Class.tsx` + `ClassSVG.tsx` + `HeaderSection` + `RowBlockSection`): Same conceptual structure — header height `LAYOUT.DEFAULT_HEADER_HEIGHT (40)` or `..._WITH_STEREOTYPE (50)`, bold centered name, separation lines via `SeparationLine`. Width auto-fits text via `measureTextWidth`; height locked to `minHeight` (`Class.tsx:142-160`).

**Visual: identical** in shape; minor diff in clipping strategy (no clipPath wrapper in v4 — overflowing text is clipped by the SVG viewport instead).

#### Package

- **Old** (`uml-class-package-component.tsx`): Path `M 0 10 V 0 H 40 V 10` for the tab + main rect underneath, name at `(50%, 30)` bold.
- **New** (`Package.tsx` → `PackageSVG`): Same shape.

**Visual: identical** (both use the canonical UML package tab).

#### State (parent)

- **Old** (`uml-state-component.tsx`): Two stacked rounded rects (cornerRadius 8) with a header strip 40 or 50px tall depending on stereotype, body extends below; bottom strokes drawn for `hasBody` and `hasFallbackBody`.
- **New** (`State.tsx`): One rounded rect (cornerRadius 8) with one horizontal `<line>` separator below the header. The body / fallback dividers are the React-Flow children's own borders (each `StateBody` is a separate node).

**Visual: minor diff** — the new layout drops the second separator that old fork drew between body region and fallback region (the children handle their own borders). The body separator itself is a single line in new (`State.tsx:106-113`), and it's drawn **above** the children rather than `<rect>`-clipping them.

#### StateInitialNode

- **Old** (`uml-state-initial-node-component.tsx`): `ThemedCircleContrast` filled black, no stroke, defaults 50×50. `interactive` toggles to theme.interactive.normal during highlight mode.
- **New** (`StateInitialNode.tsx`): Plain `<circle>` filled with `var(--apollon-primary-contrast, #000000)`, no stroke; corner / mid handles hidden; defaults to whatever React-Flow assigns at creation.

**Visual: identical** in resting state. New lib drops the highlight-mode theming branch (interactive overlay) — minor regression in assessment view.

#### ObjectName

- **Old** (`uml-object-name.ts` rendering via `UMLClassifier.render`): Same component as Class but with `underline = true`, optional icon view via `extractSvgSize`, header text format `{name}: {className}` when classId set.
- **New** (`ObjectName.tsx` + `ObjectNameSVG.tsx`): Mirrors layout but **does not render `underline`** on the name (the underline:true v3 default is lost). `data.icon` is captured by the migrator (`versionConverter.ts:644-660`) but the new visual does **not** show an icon view.

**Visual: significant diff** — underline dropped, icon-view dropped (`settingsService.shouldShowIconView()` toggle has no v4 equivalent in `ObjectName.tsx`).

### F. Cross-diagram bridge wiring

- `diagramBridge.setClassDiagramData()` is fed `{nodes, edges}` arrays in v4 (compatible with the new shape) — `lib/services/diagramBridge.ts:118-126`. Verified.
- `getAvailableClasses()` walks `data.nodes` and tolerates legacy `Class/AbstractClass/Interface/Enumeration` types as well as v4's `class` (line 92-99). It reads `data.attributes` array in v4 shape (line 253). Verified.
- `getAvailableAssociations()` walks `data.edges`, filters out inheritance/realization, returns sourceMultiplicity/targetMultiplicity from edge `data`. Verified.
- `getStateMachineDiagrams()` / `getQuantumCircuitDiagrams()` are setter+getter pairs; consumed by `ClassEditPanel.tsx:672-673` for the method `state_machine` / `quantum_circuit` selectors. Verified.

**Bridge parity: ✅** — wiring is intact and the readers handle both v3-shape leaks and v4 shapes.

---

## ObjectDiagram

### A. Element type inventory

Old (`uml-object-diagram/index.ts`):

- **Nodes**: `ObjectName`, `ObjectAttribute`, `ObjectMethod`, `ObjectIcon`
- **Edges**: `ObjectLink`

New:

- **Nodes**: `objectName`
- **Edges**: `ObjectLink`

| v3 type | v4 status | Mapping notes |
|---|---|---|
| `ObjectName` | RENAMED → `objectName` | |
| `ObjectAttribute` | RENAMED → row inside `data.attributes` | Migrator: `versionConverter.ts:1854` |
| `ObjectMethod` | RENAMED → row inside `data.methods` | Same |
| `ObjectIcon` | RENAMED → `data.icon` on owner | Collapsed (`versionConverter.ts:644-660`) |
| `ObjectLink` | = | |

No MISSING / EXTRA at the type level.

### B. Per-type data field parity

#### ObjectName → `objectName` node

Old `IUMLObjectName` (`uml-object-name.ts:15-19`):

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| `classId?` | `data.classId?` | = |
| `className?` | `data.className?` | = |
| `icon?` (separate child element) | `data.icon?` (collapsed) | RENAMED (collapse) |
| `underline = true` (default) | (no `underline` field on `ObjectNodeProps`) | **MISSING in new** — v3 ObjectName always rendered with underline; v4 has no equivalent. |
| `attributes` (id list) | `data.attributes: ObjectNodeAttribute[]` | RENAMED (collapsed rows) |
| `methods` (id list) | `data.methods: ClassNodeElement[]` | RENAMED (collapsed rows) |
| `stereotype: string \| null` | (none — `ObjectNodeProps` has no stereotype) | **MISSING** but acceptable — object diagrams don't use stereotypes in practice |
| `italic` | (none) | **MISSING** — same |

#### ObjectAttribute → row in `data.attributes`

Old `IUMLObjectAttribute` (`uml-object-attribute.ts:9-12`) extends `UMLClassifierAttribute`:

| Old field | New field | Status |
|---|---|---|
| `name` (formatted as `"foo = value"`) | `name` | = (the inspector still parses `" = "` and rebuilds it on commit) |
| `attributeId?` | `attributeId?: string` | = |
| `attributeType` (default `'str'`) | `attributeType?: string` | ≠ optional |
| Inherited: `visibility`, `code`, `defaultValue` etc. | inherited via `ClassNodeElement` | = |
| (parsed from name) `value` | `value?: unknown` | EXTRA in new — structured value |

#### ObjectMethod → row in `data.methods`

Same as `ClassMethod` superset since `UMLObjectMethod extends UMLClassifierMethod`. New: `ClassNodeElement` covers the row shape, but the new ObjectEditPanel has a **drastically simplified** method form (just name + delete) — no visibility/return-type/parameters/code editor. v3 was identical to ClassMethod, so this is a regression.

#### ObjectIcon

Old `UMLObjectIcon`:

| Old field | New field | Status |
|---|---|---|
| `icon?: string` | `data.icon?` on parent | RENAMED (collapsed) |
| Bounds | n/a (icon view rendering removed) | **MISSING** at the rendering level |

#### ObjectLink

Old `UMLObjectLink` (`uml-object-link.ts:7-10`):

| Old field | New field (edge `data`) | Status |
|---|---|---|
| `name` | `data.name` / `data.label` | = |
| `path` | `data.points` | = |
| `source.element` / `target.element` | `source` / `target` | = |
| **`associationId?: string`** | (no field anywhere) | **MISSING in new** — `versionConverter.ts:1240-1260` does not pass it through. CRITICAL. |

### C. Panel inspector parity

#### ObjectEditPanel (`objectName` node)

Old: `uml-object-name-update.tsx`

| Old form field | Old field type | New form field | New field type | Status |
|---|---|---|---|---|
| Object name with placeholder `objectName` or `{class.toLowerCase()}Instance` | text + auto-placeholder | `data.name` | text | ≠ regression — auto-placeholder logic dropped (`getObjectNamePlaceholder`) |
| Class dropdown + display: `Customer extends Person (3 attrs)` | `Dropdown` with hierarchy info via `getClassHierarchy` | Class dropdown showing only `c.name` | MUI Select | ≠ regression — extends/attr-count display dropped |
| On class change: auto-add inherited class attributes as Object rows | imperative `create()` per attribute (`uml-object-name-update.tsx:113-128`) | (just stores `classId`/`className`; no auto-attribute population) | — | **MISSING in new** — significant UX regression. v3 prefilled rows from class on link, v4 leaves them empty. |
| `linkedClassAttrs` per-row attributeId picker | (n/a, v3 used the auto-add flow above) | `Select` keyed on `linkedClass.attributes` | MUI Select | EXTRA in new — explicit per-row link |
| **Per-attribute** | | | | |
| Attribute name (read-only label `"foo = "`) | static label | text input editing the row name | MUI text | ≠ — old form locked the name (the class drove it); new lets you type-edit. |
| Value field with type-aware variants: enumeration dropdown, date/time picker, duration field, quoted string | conditional widgets | plain text `value` field | MUI text | ≠ regression — type-aware widgets all dropped |
| (n/a — v3 had no separate type field on the row) | — | `attributeType` Select with primitives + classes + custom | MUI Select | EXTRA in new |
| **Per-method** | | | | |
| (commented out in v3 — methods section disabled) | — | name field + delete | minimal | EXTRA / NEW (v3 hid this) |

#### ObjectLink inspector — **MISSING ENTIRELY**

Old `uml-object-link-update.tsx` exposed: link name, **flip**, **delete**, **associationId picker** populated from `diagramBridge.getAvailableAssociations(sourceClassId, targetClassId)`. New lib has no inspector for `ObjectLink` (only the generic `ObjectDiagramEdge` renders the line). **CRITICAL** for editing-flow parity AND tied to the missing `associationId` field.

### D. Constraints + invariants

| Constraint | v3 | v4 | Severity |
|---|---|---|---|
| Object name placeholder = `{className.toLowerCase()}Instance` when class is linked | `uml-object-name-update.tsx:157-163` | not enforced | MINOR UX |
| Auto-create inherited attribute rows when class is linked | `uml-object-name-update.tsx:113-128` | not enforced | **MEDIUM** |
| Replace existing attributes (delete then re-add) on class change | `uml-object-name-update.tsx:107-112` | not enforced | MEDIUM |
| `attributeId` linkage to source class attribute | tracked via the auto-add flow | `attributeId` field exists; user picks per-row. The bridge serves the linked class's attributes (`ObjectEditPanel.tsx:226`). | OK |
| `associationId` on `ObjectLink` | tracked in the v3 element (`uml-object-link.ts:9`) | **NOT TRACKED** | CRITICAL |
| Type-aware value widgets for `date`/`datetime`/`time`/enum/`str` | `uml-object-attribute-update.tsx:144-330` | not implemented | MEDIUM |
| Object renders with `underline` | hardcoded `underline = true` in `UMLObjectName` | not rendered | MEDIUM |
| Icon-view rendering when `settingsService.shouldShowIconView()` and SVG icon present | `uml-object-name.ts:147-204` | not implemented in `ObjectName.tsx` | MEDIUM |

### E. Visual shape parity

- ObjectName: `objectName` node — see ClassDiagram §E. Old: classifier card with **underline** on the name; New: classifier card without underline. **Visual: minor diff** (missing underline; missing icon-view).
- ObjectLink: simple polyline edge in old; React-Flow step path with optional midpoint dragging in new. **Visual: minor diff** (path shape may differ — v3 used direct polyline, v4 uses step path with markers).

### F. Cross-diagram bridge wiring

- `classId` resolution in `ObjectEditPanel.tsx:225` (linked-class lookup) uses `availableClasses.find(c => c.id === nodeData.classId)`. ✅
- `linkedClass.attributes` drives the per-row attribute picker (`ObjectEditPanel.tsx:226`). The bridge correctly returns `IAttributeInfo[]` including inherited ones via `getAllAttributesWithInheritance`. ✅
- `getAvailableAssociations(sourceClassId, targetClassId)` exists on the bridge but **no call site in v4** under `lib/components/inspectors/objectDiagram/` consumes it (because no ObjectLink inspector exists).

**Bridge parity: ⚠️** — capability is there, consumer-side hookup missing for ObjectLink.

---

## StateMachineDiagram

### A. Element type inventory

Old (`uml-state-diagram/index.ts`):

- **Nodes**: `State`, `StateBody`, `StateFallbackBody`, `StateActionNode`, `StateFinalNode`, `StateForkNode`, `StateForkNodeHorizontal`, `StateInitialNode`, `StateMergeNode`, `StateObjectNode`, `StateCodeBlock`
- **Edges**: `StateTransition`

New (`lib/nodes/stateMachineDiagram/index.ts`):

- **Nodes**: `State`, `StateBody`, `StateFallbackBody`, `StateActionNode`, `StateObjectNode`, `StateInitialNode`, `StateFinalNode`, `StateMergeNode`, `StateForkNode`, `StateForkNodeHorizontal`, `StateCodeBlock`
- **Edges**: `StateTransition`

| v3 type | v4 status |
|---|---|
| `State` | = |
| `StateBody` | = |
| `StateFallbackBody` | = |
| `StateActionNode` | = |
| `StateFinalNode` | = |
| `StateForkNode` | = |
| `StateForkNodeHorizontal` | = |
| `StateInitialNode` | = |
| `StateMergeNode` | = |
| `StateObjectNode` | = |
| `StateCodeBlock` | = |
| `StateTransition` | = |

**Element parity: ✅ exact 1:1.**

### B. Per-type data field parity

#### State

Old `IUMLState` (`uml-state.ts:16-23`):

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| `italic` | `data.italic?: boolean` | = (made optional) |
| `underline` | `data.underline?: boolean` | = |
| `stereotype: string \| null` | `data.stereotype?: string \| null` | = |
| `deviderPosition` | — | DROP (recomputed) |
| `hasBody` / `hasFallbackBody` | — | DROP (recomputed) |
| `bodies` (id list) | (children via `parentId`) | = (React Flow native) |
| `fallbackBodies` (id list) | (children via `parentId`) | = |

#### StateBody / StateFallbackBody

Old: just `name` (extends `UMLStateMember` which only adds bounds defaults).

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| (none) | `data.code?: string` | EXTRA in new — SA-3 brief addition |
| (none) | `data.kind?: string` (`entry` / `do` / `exit` / `transition`) | EXTRA in new |

#### StateActionNode

Old: just `name`.

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| (none) | `data.code?: string` | EXTRA in new |

#### StateCodeBlock

Old `IUMLStateCodeBlock` (`uml-state-code-block.ts:11-15`):

| Old field | New field | Status |
|---|---|---|
| `name` (inherited) | `data.name` | = |
| `code: string` (default `''`) | `data.code: string` | = |
| `language: string` (forced `'python'`) | `data.language?: string` (default `'python'`) | ≠ — v4 exposes BAL option in inspector; v3 forced Python |
| `_codeContent?: string` (internal cache) | — | DROP (was a v3 implementation detail) |

#### StateInitialNode / StateFinalNode / StateMergeNode / StateForkNode / StateForkNodeHorizontal / StateObjectNode

These are all marker-style nodes in v3. New `StateMarkerNodeProps` is just `DefaultNodeProps`; `StateObjectNodeProps` adds `classId?` + `className?`. v3 fork only stored `name` on most of these (`uml-state-final-node.ts`, `uml-state-fork-node.ts` etc). **= exact match**.

`StateInitialNode` in v3 supports both `StateTransition` and `AgentStateTransitionInit` (`uml-state-initial-node.ts:11`). v4 inherits the AgentStateTransitionInit edge type from SA-4, so this is OK.

#### StateTransition (edge)

Old `IUMLStateTransition` (`uml-state-transition.ts:7-10`) extending `UMLRelationshipCenteredDescription`:

| Old field | New field (edge `data`) | Status |
|---|---|---|
| `name` | `data.name` | = |
| `params: { [id: string]: string }` (default `{}`) | `data.params?: { [key: string]: string }` | = |
| `guard: string` (default `''`) | `data.guard?: string` | = (optional) |
| `path` | `data.points` | = |
| (none) | `data.code?: string` | EXTRA in new — SA-3 brief addition |
| (none) | `data.eventName?: string` | EXTRA in new — SA-3 brief addition |
| (none) | `data.label?: string` | EXTRA — generic edge label |

The v3 `params` shape allowed `string | string[] | { [id]: string }` for backwards-compat (lines 17-32). The migrator at `versionConverter.ts:1227-1238` normalises all three to a dict. **= preserved**.

### C. Panel inspector parity

#### StateEditPanel (`State` node)

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| `name` | text | `name` | MUI text | = |
| Color/fill/stroke | StylePane | `NodeStyleEditor` | inline color editor | = |
| Bodies list (per-body row) | inline row of `UmlBodyUpdate` components | (not in panel — each StateBody is its own React-Flow child with its own panel) | — | RENAMED (architectural change per SA-3) |
| Fallback bodies list | same | (same) | — | RENAMED |
| (none) | — | Stereotype dropdown (initial / final / decision / fork / merge / none) | MUI Select | EXTRA in new |
| (none) | — | `italic` checkbox | MUI checkbox | EXTRA in new |
| (none) | — | `underline` checkbox | MUI checkbox | EXTRA in new |

Note: the new stereotype dropdown is **free-text** (it sets `data.stereotype` to whatever string), and the labels are `«initial»` etc. — these are not class stereotypes, they're descriptive markers. v3 didn't surface stereotype editing on the State form at all (the field existed but had no UI control). New panel exposes it.

#### StateBodyEditPanel (StateBody / StateFallbackBody)

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| `name` (textfield) | text | `name` (label field) | MUI text | = |
| Color/fill/stroke | StylePane | `NodeStyleEditor` | = | = |
| (none) | — | `kind` dropdown (entry / do / exit / on transition) | MUI Select | EXTRA |
| (none) | — | `code` multiline | MUI text multiline | EXTRA |

#### StateActionNodeEditPanel

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| `name` | text (single-textfield rename) | `action name` | MUI text | = |
| (none) | — | `code` multiline | MUI text multiline | EXTRA |

#### StateCodeBlockEditPanel

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| Color/fill/stroke | StylePane | `NodeStyleEditor` | = | = |
| Width / Height numeric inputs | text inputs (not in v4 panel — handled by NodeResizer instead) | (handled via React-Flow's NodeResizer at the canvas) | — | RENAMED |
| Code (multi-line `<textarea>` with tab key support, monospace) | textarea | MUI multiline TextField with `fontFamily: monospace` | = | = (tab-key handler dropped — v3 inserted `\t` on Tab key, v4 lets the browser default consume Tab) |
| (none) | — | `language` dropdown (Python / BAL) | MUI Select | EXTRA in new (v3 forced Python) |

#### StateMergeNodeEditPanel

The v3 fork had `uml-state-merge-node-update.tsx` exposing **per-decision rows** with the outgoing transition's name editable inline. The new lib uses `StateLabelEditPanel` for MergeNode — only `name` is editable, decision rows are not surfaced. **MEDIUM regression** for users who relied on the v3 inline-decision UI.

#### StateLabelEditPanel (StateInitialNode, StateFinalNode, StateMergeNode, StateForkNode, StateForkNodeHorizontal)

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| (most have no form / not updatable in v3 — `updatable: false` flag set on Final/Fork/ForkHorizontal/Initial) | — | `name` field | MUI text | EXTRA in new — v4 lets users edit names on nodes that v3 hid |

#### StateObjectNodeEditPanel

v3 had no dedicated update form for `UMLStateObjectNode`. The new lib adds a `name` + `classId` + `className` selector. **EXTRA in new** (per spec open question 4).

#### StateMachineDiagramEdgeEditPanel

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| `name` | text (autoFocus) | `name` | MUI text | = |
| `guard` | text (placeholder "Guard expression") | `guard` (placeholder "boolean expression in [...]") | MUI text | = |
| `params` ordered list (Add button + per-param textfield + delete) | textfields | same shape (Add link + per-param textfield + delete) | MUI text + IconButton | = |
| Color/line/text via StylePane | StylePane | (no NodeStyleEditor on edge panel) | — | ≠ regression — color editing not exposed for StateTransition edges |
| **Flip** action | ExchangeIcon button | (none) | — | **MISSING in new** |
| **Delete** action | TrashIcon button | (none — relies on canvas-level delete) | — | MISSING (acceptable since canvas keyboard shortcut exists) |
| (none) | — | `eventName` field | MUI text | EXTRA |
| (none) | — | `code` multiline | MUI text multiline | EXTRA |

### D. Constraints + invariants

| Constraint | v3 | v4 | Severity |
|---|---|---|---|
| `StateInitialNode` accepts both `StateTransition` AND `AgentStateTransitionInit` | static array (`uml-state-initial-node.ts:11`) | edges register both types in registry | OK (cross-diagram-aware) |
| `State` requires at least one Body to be valid | not enforced as a hard constraint in v3 either | not enforced | OK (no regression) |
| `StateFinalNode`, `StateForkNode`, `StateForkNodeHorizontal`, `StateInitialNode` are non-resizable / non-updatable | `features.resizable: false`, `features.updatable: false` flags | NodeResizer is gated on `isDiagramModifiable` only; `updatable: false` is not respected — all marker nodes get the StateLabelEditPanel | MEDIUM (v3 hid the form entirely; v4 always shows it) |
| `StateForkNode` defaultWidth=20, defaultHeight=60 | static class field | `NodeResizer` minWidth=20 maxWidth=20 minHeight=60 in `StateForkNode.tsx:50-53` | = |
| `StateForkNodeHorizontal` defaultWidth=60, defaultHeight=20 | static class field | mirror config in `StateForkNodeHorizontal.tsx` | = |
| `StateCodeBlock` minimum dimensions 150×100 | enforced in `render()` | NodeResizer enforces minWidth/minHeight | = |
| `StateCodeBlock.language` forced to `'python'` | hardcoded in v3 constructor | configurable in v4 inspector | EXTRA (deliberate enhancement) |
| `params` accepts `string \| string[] \| dict` | normalised in deserialize | normalised in `versionConverter.ts:1227-1238` | = |

### E. Visual shape parity

#### State

See above (ClassDiagram §E for general structure). The v3 state component uses **two stacked rounded rects** (header strip + body region with separate colors), v4 uses a single rounded rect with a header line. **Visual: minor diff** — color scheme differs (v3 had distinct header band fill, v4 single fill).

#### StateInitialNode

Both render a filled circle. **Visual: identical** at rest; v3 has a highlight-mode theming branch v4 doesn't replicate. Minor diff.

#### StateFinalNode

Old: outer circle with inner filled disc (the standard UML "bullseye"). Let me note: I didn't read `uml-state-final-node-component.tsx` directly but the shape is the canonical bullseye. New `StateFinalNode.tsx` (not read in detail) is registered and presumed to render the same. **Visual: presumed identical** (verify in regression test).

#### StateForkNode / StateForkNodeHorizontal

Both v3 and v4 render a filled rectangle (vertical bar 20×60 / horizontal bar 60×20). **Visual: identical**.

### F. Cross-diagram bridge wiring

- `StateObjectNodeEditPanel` consumes `diagramBridge.getAvailableClasses()` for the class picker. ✅
- v4 method-implementation `state_machine` link feeds back into ClassDiagram methods via `diagramBridge.getStateMachineDiagrams()` — not directly a StateMachineDiagram concern, but the contract is "StateMachine names + ids are exposed to other diagrams". The setter is called by the embedding webapp before opening the editor; there's no direct StateMachineDiagram-side test in the new lib but the API surface matches v3.
- No `quantumCircuitId` selector is part of any State node in the SA-3 port — v3 didn't expose one either, so this is fine.

**Bridge parity: ✅** for SA-3.

---

## Summary table

| Diagram | Element parity | Data parity | Form parity | Constraint parity | Visual parity | Bridge parity |
|---|---|---|---|---|---|---|
| ClassDiagram | ⚠️ (ClassLinkRel + ClassOCLLink dropped) | ⚠️ (italic / underline / freeform-stereotype not stored) | ❌ (no edge inspector; CodeMirror replaced with plain text; enumeration types dropped from picker) | ⚠️ (Class name sanitisation gone; Enumeration's no-methods rule dropped) | ✅ identical | ✅ |
| ObjectDiagram | ✅ | ⚠️ (associationId dropped on ObjectLink; underline default lost) | ❌ (no edge inspector; auto-attribute population on classId-link gone; type-aware value widgets gone) | ⚠️ (associationId not tracked; auto-add on class change not implemented) | ⚠️ (no underline; no icon-view) | ⚠️ (capability present, ObjectLink consumer absent) |
| StateMachineDiagram | ✅ exact | ✅ (only EXTRA additions) | ✅ (mostly; flip action and color editor missing on edge panel; merge-node decisions UI dropped) | ⚠️ (`updatable: false` on marker nodes not respected; merge-node decisions inline editing dropped) | ✅ minor diffs only | ✅ |

---

## Recommendations / gaps for follow-up SAs

Concrete actions for SA-7 (parity-fix) before cutover. Severity in **bold**.

1. **CRITICAL — restore `ClassOCLLink` and `ClassLinkRel` edge types.** Add entries to `lib/edges/types.tsx` (`defaultEdgeTypes` + `edgeConfig`) and to `versionConverter.ts:331-426`'s `edgeTypeMap`. Reuse `ClassDiagramEdge` as the renderer; add a dedicated marker pair (open arrow + dotted stroke for OCL link, plain solid line for link-rel). Without this, any v3 fixture exercising these edge types is silently corrupted on import.
2. **CRITICAL — propagate `ObjectLink.associationId` end-to-end.** Add `associationId?: string` to the v4 ObjectLink edge data shape (an inline type next to `ObjectDiagramEdge` props), to `convertV3RelationshipToV4Edge` so the field passes through, and to `convertV4ToV3` so it round-trips. File path: `lib/utils/versionConverter.ts:1240-1260` for the converter, plus the `ObjectDiagramEdge.tsx` data interface.
3. **CRITICAL — add a ClassEdgeEditPanel.** Register an inspector under `lib/components/inspectors/classDiagram/ClassEdgeEditPanel.tsx` for the seven class edge types. Must expose: name, **flip**, association-type Select (matching v3's 4-entry dropdown), and per-end multiplicity + role textfields with v3 placeholders (`'1..1'`, plus `'(1,1) or 1..1'` ER hint). Reference `packages/editor/.../uml-class-association-update.tsx` for the field layout.
4. **CRITICAL — add an ObjectLinkEditPanel.** Same idea as #3 but with the `associationId` picker driven by `diagramBridge.getAvailableAssociations(sourceClassId, targetClassId)`. File path: `lib/components/inspectors/objectDiagram/ObjectLinkEditPanel.tsx`.
5. **MEDIUM — restore type-aware Object attribute value widgets.** v3's `uml-object-attribute-update.tsx` had: enumeration-type → enum-literals dropdown; date/datetime/time types → `<input type="...">`; `str` → quote-wrapped textfield; `timedelta` → free-form duration input. New lib renders all values as plain text. Move these into `ObjectEditPanel.tsx::ObjectAttrRow` so per-row type choice drives the input widget.
6. **MEDIUM — restore auto-attribute population on `classId` change.** When a user selects a class in `ObjectEditPanel.tsx`, populate `data.attributes` from `linkedClass.attributes` (clear existing, mirror v3's flow at `uml-object-name-update.tsx:107-128`). Otherwise users must hand-add every attribute.
7. **MEDIUM — restore `underline` rendering on ObjectName.** Apply `text-decoration: underline` to the name `<text>` in `lib/components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx`. v3 had `underline = true` hardcoded on `UMLObjectName`.
8. **MEDIUM — restore icon-view rendering for ObjectName** when `data.icon` is present (and a global toggle, if you keep `settingsService.shouldShowIconView()`). The migrator preserves `data.icon`, but `ObjectName.tsx` ignores it.
9. **MEDIUM — restore CodeMirror in the method code editor.** `ClassEditPanel.tsx:562-578` currently uses a plain `multiline` MUI text field for `code`/`bal` impl types. v3 used CodeMirror with Python syntax highlighting and tab-indent support. UX regression.
10. **MEDIUM — sanitise class `name` on commit.** `ClassEditPanel.tsx:680-682` writes the raw `value` through. Match v3's `[^a-zA-Z0-9_]` strip (already present for attribute names — apply at the class level too in `NodeStyleEditor` callsite).
11. **MEDIUM — block method creation on `class` nodes whose stereotype is `Enumeration`.** Currently `ClassEditPanel.tsx:898-923` always shows the Methods section. v3 hid it for enums (`uml-classifier-update.tsx:344`).
12. **MEDIUM — restore enumeration types in the attribute-type picker.** Today `ClassEditPanel.tsx:181-190` offers class names but not the literals from sibling Enumerations. v3 included them via `availableEnumerations` (file `uml-classifier-update.tsx:200-202`).
13. **MEDIUM — respect `updatable: false` on State marker nodes.** v3 hid the entire form for `StateInitialNode`, `StateFinalNode`, `StateForkNode`, `StateForkNodeHorizontal`. New `StateLabelEditPanel` shows a `name` field for all of them. Either gate the panel by node type or drop the rename UI for these specific types.
14. **MEDIUM — restore `StateMergeNode` decisions inline editor.** v3's `uml-state-merge-node-update.tsx` listed each outgoing transition with editable name + arrow + target. New lib points MergeNode at the generic `StateLabelEditPanel`. Provide a dedicated panel that walks `state.edges.filter(e => e.source === node.id)` and lets users rename each transition inline.
15. **MEDIUM — add the flip action to `StateMachineDiagramEdgeEditPanel`.** v3 had it (`uml-state-transition-update.tsx:131-133`). The action is independent of the inspector — the `useDiagramStore` setEdges API can swap source/target/handle pair.
16. **MINOR — add color editor (`NodeStyleEditor`-equivalent) to all edge panels.** Currently only the Class panel renders color controls; `StateMachineDiagramEdgeEditPanel.tsx`, `ObjectDiagramEdge` (no panel), and `ClassDiagramEdge` (no panel) lack it. The data fields (`fillColor`, `strokeColor`, `textColor`) flow through; just expose them.
17. **MINOR — restore `description`, `uri`, `icon` extras on the Class style pane.** v3 `StylePane` had `showDescription showUri showIcon` flags (`uml-classifier-update.tsx:219-221`). New `NodeStyleEditor` doesn't surface them.
18. **MINOR — restore the v3 attribute reorder up/down buttons.** `uml-classifier-update.tsx:255-273`. Useful for diagram authors with many fields.
19. **MINOR — verify v3 lowercase stereotype string (`'abstract'`/`'interface'`/`'enumeration'`) round-trip is exhaustively migrated.** `lib/types/nodes/enums/ClassType.ts` uses PascalCase; the migrator converts via element type, not via string content. Add a regression test that imports a v3 class with hand-edited stereotype text and verifies stable round-trip.
20. **MINOR — surface the `oclConstraints` editor more discoverably.** The OCL section in `ClassEditPanel` is at the very bottom of the panel, after Methods. Spec-recommended location is fine but UX-wise consider promoting it next to Attributes.

---

*Audit completed by SA-PARITY. Read-only audit; no source code modified.*
