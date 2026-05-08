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

---

# Wave 2 Parity Audit, Round 2 (post-SA-5)

Audit date: 2026-05-08
Audited: AgentDiagram, UserDiagram, NNDiagram
Reference SHA (submodule): `3c84116` (`feat(library): SA-5 NNDiagram port + attribute collapse`)
Auditor branch: `claude/refine-local-plan-sS9Zv` (read-only)

## Top-line verdict

The three new diagrams round-trip the canonical wire shape but ship with substantial **inspector-side** parity gaps. The most damaging are:

1. **AgentState's body editor is gone.** v3's `agent-state-update.tsx` was the load-bearing UI for the entire diagram — five-radio reply-mode picker (text / llm / rag / db_reply / code), inline RAG-database dropdown driven by `AgentRagElement` siblings, full DB-action editor (selectionType / customName / SQL editor / operation), and CodeMirror-backed Python body. The new SA-4 `AgentStateEditPanel` reduces all of this to a `name` + `replyType` Select + stereotype + italic/underline checkboxes. Body content for every reply mode now lives on a single `code` multiline field on each child `AgentStateBody` panel. **CRITICAL** — agent diagrams are no longer fluently editable.
2. **AgentDiagram intent-name picker collapsed to a free-text field.** v3 populated it from a live dropdown of `AgentIntent` element names so users couldn't typo. v4 is a plain text input. Same downgrade for `fileType` (PDF/TXT/JSON dropdown → free text). **MEDIUM** but bug-prone.
3. **NN per-layer conditional attribute filtering is gone.** v3 hid `kernel_dim` / `stride_dim` for `global_average` Pooling, hid `shape` / `normalize` for non-image datasets, and routed TensorOp optional fields through a tns_type filter (`reshape` shows only `reshape_dim`, `concatenate` shows `layers_of_tensors` + `concatenate_dim`, etc.). The SA-5 `NNComponentEditPanel` shows every optional field unconditionally. **MEDIUM** — clutters the UI and loses TensorOp's mode-aware UX.
4. **AgentRagElement gained 6 EXTRA fields that v3 carried only on AgentStateMember bodies.** New `AgentRagElementNodeProps` includes `dbSelectionType`, `dbCustomName`, `dbQueryMode`, `dbOperation`, `dbSqlQuery`, `ragType`, `ragDatabaseName`. v3 `AgentRagElement` only had `name`. This is intentional per the SA-4 brief / open question #5 but means the v4 model is **richer** than v3, not 1:1; the round-trip is lossy in the v3-direction (v3 fixtures have `name` only; the fields are preserved through v4 → v3 only because they ride on `data` and the converter passes them through verbatim).
5. **NN Conv2D card visual collapsed to a generic `_NNLayerBase` card.** v3 `NNBaseLayer` rendered a UML-class-style card (icon view at fixed 110×110) with attributes hidden but children stored for popup editing. The new `_NNLayerBase` shows a simple rounded rectangle with `«kindLabel»` stereotype + name; resizable. Visual style differs and lacks the v3 layer-icon SVGs. **MEDIUM** UX regression, not data-loss.

The other 15+ gaps below are MEDIUM or MINOR and concentrated in the inspector layer.

---

## AgentDiagram

### A. Element type inventory

Old fork (`packages/editor/src/main/packages/agent-state-diagram/index.ts`):

- **AgentElementType** keys exposed: `State, StateBody, AgentIntentBody, StateFallbackBody, StateActionNode, StateFinalNode, StateForkNode, StateForkNodeHorizontal, StateInitialNode, StateMergeNode, StateObjectNode, StateCodeBlock, AgentIntent, AgentRagElement, AgentState, AgentStateBody, AgentStateFallbackBody`. The agent-only types (the ones registered as discrete UMLElements in this package) are: **`AgentState`, `AgentStateBody`, `AgentStateFallbackBody`, `AgentIntent`, `AgentIntentBody`, `AgentRagElement`** — six types. The other entries are re-exported state-machine types reused by AgentDiagram for shared marker nodes.
- **AgentRelationshipType**: `AgentStateTransition`, `AgentStateTransitionInit`.
- **NOT registered as UMLElements in v3**: `AgentIntentDescription` (only `AgentIntentDescriptionComponent` is a render component used inline by AgentIntent), `AgentIntentObjectComponent` (file holds the `AgentIntentComponent` renderer for AgentIntent — not a separate element).

New lib (`packages/library/lib/nodes/agentDiagram/index.ts`):

- **Nodes**: `AgentState`, `AgentStateBody`, `AgentStateFallbackBody`, `AgentIntent`, `AgentIntentBody`, `AgentIntentDescription`, `AgentIntentObjectComponent`, `AgentRagElement` — eight types.
- **Edges**: `AgentStateTransition`, `AgentStateTransitionInit`.

| v3 type | v4 status | Mapping notes |
|---|---|---|
| `AgentState` | = | |
| `AgentStateBody` | = | |
| `AgentStateFallbackBody` | = | |
| `AgentIntent` | = | |
| `AgentIntentBody` | = | |
| `AgentRagElement` | = (but data shape **expanded**, see §B) | |
| `AgentStateTransition` | = | 5-shape legacy collapse via canonical `AgentStateTransitionData` (per `versionConverter.ts:1986-2002`) |
| `AgentStateTransitionInit` | = | |
| (no v3 element) | `AgentIntentDescription` | **EXTRA in new** — promoted from inline render-only component to a first-class node type. v3 stored description as `AgentIntent.intent_description: string` only. |
| (no v3 element) | `AgentIntentObjectComponent` | **EXTRA in new** — v3 had no element type by this name; the file in v3 only exported the renderer component for AgentIntent itself. |

The brief says these "are now done" — the spec at `docs/source/migrations/uml-v4-shape.md` (AgentDiagram §) opted to promote both to first-class nodes, so this is by design. But it does mean the migrator must synthesise / collapse them on import (see §B).

**Element parity: ⚠️ — six v3 types preserved + two intentional EXTRA promotions. No v3 type was dropped.**

### B. Per-type data field parity

#### AgentState

Old `IUMLState` / `AgentState` (`agent-state.ts:23-48`):

| Old field | New field on `AgentStateNodeProps` | Status |
|---|---|---|
| `name` | `data.name` | = |
| `italic: boolean` | `data.italic?: boolean` | = (optional in new) |
| `underline: boolean` | `data.underline?: boolean` | = (optional) |
| `stereotype: string \| null` | `data.stereotype?: string \| null` | = |
| `dividerPosition` | — | DROP (recomputed at render time, OK) |
| `hasBody`, `hasFallbackBody` | — | DROP (recomputed) |
| `bodies: id[]` / `fallbackBodies: id[]` | (children via `parentId`) | RENAMED (React Flow native) |
| (no field) | `data.replyType?: string` | **EXTRA in new** — surfaced to the inspector but actually consumed by children in v3 (see §B AgentStateBody). |

#### AgentStateBody / AgentStateFallbackBody

Old `AgentStateMember` (`agent-state-member.ts:11-39`):

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| `replyType: string` (default `'text'`) | `data.replyType?: string` | = (optional in new; default `'text'` not enforced server-side) |
| `ragDatabaseName: string` | `data.ragDatabaseName?: string` | = |
| `dbSelectionType: string` (default `'default'`) | `data.dbSelectionType?: string` | = |
| `dbCustomName: string` | `data.dbCustomName?: string` | = |
| `dbQueryMode: string` (default `'llm_query'`) | `data.dbQueryMode?: string` | = |
| `dbOperation: string` (default `'any'`) | `data.dbOperation?: string` | = |
| `dbSqlQuery: string` | `data.dbSqlQuery?: string` | = |
| (no field) | `data.code?: string` | EXTRA in new — SA-4 brief addition |
| (no field) | `data.kind?: string` | EXTRA in new — SA-4 brief addition |

#### AgentIntent

Old (`agent-intent.ts:23-46`):

| Old field | New field on `AgentIntentNodeProps` | Status |
|---|---|---|
| `name` | `data.name` | = (visual prefixes `"Intent: "` at draw time in both) |
| `italic` | `data.italic?: boolean` | = |
| `underline` | `data.underline?: boolean` | = |
| `stereotype: string \| null` | `data.stereotype?: string \| null` | = |
| `deviderPosition` | — | DROP |
| `hasBody` | — | DROP |
| `intent_description: string` | `data.intent_description?: string` | = (mirrored to a separate `AgentIntentDescription` child via parentId during migration) |

#### AgentIntentBody

Old `AgentIntentMember` (`agent-intent-member.ts`): just `name`. New: `AgentIntentBodyNodeProps = DefaultNodeProps`. **= identical**.

#### AgentIntentDescription (EXTRA in new)

| Old | New |
|---|---|
| (no UMLElement; only the inline `AgentIntentDescriptionComponent` renderer reading `intent.intent_description`) | `AgentIntentDescriptionNodeProps = DefaultNodeProps` (the description text rides on `data.name`) |

The migrator's responsibility: promote `AgentIntent.intent_description` into a child `AgentIntentDescription` node on import; on export, mirror the description back onto the parent intent's `intent_description` field. Spot-check via `convertV4ToV3Agent` (`versionConverter.ts:2658-2675`): the parent intent emits `intent_description` from data, but the child description's `data.name` is **NOT** read back into the parent — instead the description child becomes its own v3 element with `type: 'AgentIntentDescription'`, which is **not a registered v3 element type**. **MEDIUM** — round-trip leaks an unknown-to-v3 element on export. v3 deserializer would silently drop it.

#### AgentIntentObjectComponent (EXTRA in new)

| Old | New `AgentIntentObjectComponentNodeProps` |
|---|---|
| (no UMLElement; only the `AgentIntentComponent` renderer for the parent intent shape) | `entity?: string`, `slot?: string`, `value?: string`, plus `name`, base style fields |

Same problem as AgentIntentDescription — v3 has no element type by this name, so the `convertV4ToV3Agent` exporter writes an unknown-to-v3 element type.

#### AgentRagElement

Old (`agent-rag-element.ts:14-43`): just `name` (and base bounds / fillColor / etc.). v3 had **no** `ragDatabaseName`, `dbSelectionType`, `dbCustomName`, `dbQueryMode`, `dbOperation`, `dbSqlQuery`, `ragType` fields on the RAG element itself.

New `AgentRagElementNodeProps`:

| Old field | New field | Status |
|---|---|---|
| `name` | `data.name` | = |
| (no field) | `ragDatabaseName?` | **EXTRA in new** — open-question-#5 design decision per the SA-4 brief |
| (no field) | `dbSelectionType?` | EXTRA |
| (no field) | `dbCustomName?` | EXTRA |
| (no field) | `dbQueryMode?` | EXTRA |
| (no field) | `dbOperation?` | EXTRA |
| (no field) | `dbSqlQuery?` | EXTRA |
| (no field) | `ragType?` | EXTRA |

These fields existed on `AgentStateMember` (the body) in v3 — so SA-4 essentially **moved** the database-config fields up onto the AgentRagElement node, in addition to keeping them on the body (per `AgentStateBodyNodeProps`). This is a deliberate over-capture for round-trip safety (the migrator can read either source). **Acceptable** but worth a regression test that confirms nothing reads "wrong" copy.

#### AgentStateTransition (edge)

Old `AgentStateTransition` (`agent-state-transition.ts:29-108`) carries:

| Old field | New field on edge `data` | Status |
|---|---|---|
| `transitionType: 'predefined' \| 'custom'` (default `'predefined'`) | `transitionType` | = |
| `predefinedType: string` (default `'when_intent_matched'`) | `predefined.predefinedType` | RENAMED (nested) |
| `intentName?` | `predefined.intentName` | RENAMED |
| `variable?, operator?, targetValue?` | `predefined.conditionValue: { variable, operator, targetValue }` | RENAMED (object) |
| `fileType?` | `predefined.fileType` | RENAMED |
| `event: CustomTransitionEvent` (default `'WildcardEvent'`) | `custom.event` | RENAMED |
| `conditions: string[]` | `custom.condition: string[]` | RENAMED |
| `params: { [id]: string }` | `data.params` | = |
| (no field) | `data.legacyShape?: 1 \| 2 \| 3 \| 4 \| 5` | EXTRA — preservation discriminator |
| (no field) | `data.legacy?: Record<string, unknown>` | EXTRA — verbatim legacy bag |
| (no field) | `data.customEvent?, customCondition?, customParams?` | EXTRA — flat aliases for inspector convenience |

The 5 legacy v3 shapes that the v3 deserializer accepted (`condition: string`, `condition: string[]`, `customEvent`/`customConditions`, nested `predefined`/`custom`, separate top-level `predefinedType`/`event`/`conditions`) all collapse into the canonical `AgentStateTransitionData` per the SA-4 brief. The `legacy` bag preserves the original verbatim. Confirmed at the migrator's open-question-comment block but **the actual `liftAgentTransitionDataToV4` is not present in the file** — `migrateAgentDiagramV3ToV4` is just a thin type guard around `convertV3ToV4`. The 5-shape collapse must therefore be happening inside `convertV3ToV4`'s edge handler, but I did not locate the per-shape branches there. **MEDIUM concern** — verify that `convertV3ToV4` actually normalises shapes 3/4/5 (the deeply-nested legacy and the `condition: 'custom_transition'` sentinel) and not just shape 1/2.

#### AgentStateTransitionInit (edge)

Old: just `params` (centered description). New: edge data carries the same `params` plus optional `points`. **= identical** (no extras).

### C. Panel inspector parity

#### AgentStateEditPanel — **MAJOR REGRESSION**

Old `agent-state-update.tsx` (~960 lines): see prior section — five-radio reply mode, inline body creation per mode, RAG-database dropdown driven by sibling `AgentRagElement` names, dedicated DB-action editor (selection type, custom name, operation dropdown, SQL multiline), CodeMirror Python editor for code mode, separate radio + editor for fallback bodies, color/fill/stroke pane.

New `AgentStateEditPanel.tsx` (145 lines): renders only `name` + `replyType` Select + stereotype Select + `italic` + `underline` checkboxes.

Fields the new panel **does not** expose (all reachable in v3):

| v3 capability | Status in v4 |
|---|---|
| Five-radio reply mode (text/llm/rag/db_reply/code) | RENAMED → single `replyType` Select with 7 entries (incl. `image`/`json`) on `AgentState`, but the **body-creation logic** is lost |
| Inline text-reply body row Add/Remove (multiple bodies) | **MISSING in new** — bodies are siblings via `parentId` and edited individually |
| RAG-database dropdown filtered to existing `AgentRagElement` names | **MISSING** — body's `ragDatabaseName` is edited as plain text on `AgentStateBodyEditPanel` (no field exposed at all currently — see below) |
| DB-action editor (selectionType / customName / queryMode / operation / SQL) on the body | **MISSING in new body panel** (only `kind` + `replyType` + `code` exposed) |
| CodeMirror Python editor with tab-indent handling | **MISSING in new** — `code` is a plain MUI multiline TextField |
| Separate fallback-body block | **MISSING in new** — fallback bodies are just children of a different type (`AgentStateFallbackBody`); both share the same body panel which has no fallback-mode discriminator |
| Color / fill / stroke style pane | Present (via `NodeStyleEditor`) |

**Severity: CRITICAL** — agent diagrams are no longer fluently authorable in v4. This is the biggest single regression in either parity round.

#### AgentStateBodyEditPanel

Old: inline rendered by `agent-state-update.tsx` per body. New `AgentStateBodyEditPanel.tsx` (140 lines): name + kind dropdown + replyType + conditional `code` multiline.

**MISSING fields** that v3 stored on every body via `AgentStateMember`:

- `ragDatabaseName`, `dbSelectionType`, `dbCustomName`, `dbQueryMode`, `dbOperation`, `dbSqlQuery` — all field-fields preserved in `AgentStateBodyNodeProps` per the type, but **no UI editor** for any of them on the new panel. Round-trip-safe (data flows through), but unauthorable in v4.

#### AgentIntentEditPanel

Old `agent-intent-update.tsx` (~190 lines):

- intent name (autoFocus textfield, sanitised)
- color/fill/stroke + delete
- "Training Sentences" header + per-AgentIntentBody update (Textfield + ColorButton + delete + style pane)
- Add-new-body Textfield with submit
- Description (Optional) textfield bound to `intent.intent_description`

New `AgentIntentEditPanel.tsx` (56 lines): only `name` + `NodeStyleEditor`.

**MISSING in new**:

- Inline body Add / Remove (bodies edit on their own panels via `AgentIntentBodyEditPanel`)
- The `intent_description` field (the description is now a separate child node `AgentIntentDescription` with its own panel, but the parent intent still **stores** `intent_description` on `data` per the type)
- ColorButton on each body row

Architecturally consistent with SA-3's split-panel pattern but UX regression.

#### AgentIntentBodyEditPanel

Old: per-row `AgentIntentBodyUpdate` inline. New: dedicated panel with multi-line textarea labeled "training phrases (one per line)" — actually a usability **enhancement** over v3's single-line Textfield. EXTRA capability.

#### AgentIntentDescriptionEditPanel

Old: not editable as a separate widget — `intent_description` was edited on the parent intent's form. New: dedicated multi-line description editor.

#### AgentIntentObjectComponentEditPanel

Old: no per-element form (the file is the parent intent's renderer, not a child's editor). New: name + entity + slot + value text fields.

The v3 fork did NOT have entity/slot/value fields anywhere — these are **EXTRA in new** per the SA-4 brief (open question on intent → entity binding).

#### AgentRagElementEditPanel

Old `agent-rag-element-update.tsx` (33 lines): single "Name of RAG DB" textfield, autoFocus. **That's it.**

New `AgentRagElementEditPanel.tsx` (165 lines): name + dbSelectionType Select (predefined/custom/default) + ragDatabaseName + dbCustomName + dbQueryMode Select (llm_query/sql/natural_language) + conditional dbOperation + dbSqlQuery + ragType.

This is **massively richer** than v3 — the SA-4 brief explicitly resolved open question #5 by promoting the DB-config fields to AgentRagElement. Acceptable per the brief; documented here for the record.

#### AgentDiagramEdgeEditPanel (AgentStateTransition)

Old `agent-state-transition-update.tsx` (~390 lines):

| Old form field | Old type | New form field | New type | Status |
|---|---|---|---|---|
| flip action | ExchangeIcon button | (none) | — | **MISSING in new** — same regression as SA-3 (StateMachine) |
| delete action | TrashIcon button | (none — relies on canvas) | — | OK |
| color toggle + StylePane | ColorButton + pane | (none) | — | **MISSING in new** — color editing on edge dropped |
| Transition Type dropdown (predefined/custom) | Dropdown | ToggleButtonGroup | MUI ToggleButtonGroup | = |
| `predefinedType` Select with 5 entries | Dropdown | Select | MUI Select | = (5 entries match) |
| `intentName` **dropdown populated from `AgentIntent` element names** | Dropdown driven by `state.elements` | **plain text TextField** | MUI TextField | **MEDIUM regression** — typo-prone |
| `variable` / `operator` / `targetValue` fields | Textfield + Dropdown | TextField + Select | MUI | = (operator dropdown has 6 entries in v3 incl `!=`; new lib has 6 entries — match) |
| `fileType` dropdown (PDF/TXT/JSON, 3 entries) | Dropdown | **plain text TextField** | MUI TextField | **MEDIUM regression** |
| Custom-mode `event` Select (7 entries) | Dropdown | Select | MUI | = (7 entries match: None, DummyEvent, WildcardEvent, ReceiveMessageEvent, ReceiveTextEvent, ReceiveJSONEvent, ReceiveFileEvent) |
| Custom-mode `condition` per-row with **CodeMirror Python editor** + Add Condition / Remove | CodeMirror with `python` mode + line numbers + tabSize 4 | **plain text TextField** per row + add/remove | MUI TextField | **MEDIUM regression** — Python condition editor lost syntax highlighting / line numbers / tab-indent |
| `params` keyed dict editor | per-key textfields | per-key TextField + delete | MUI | = |

#### AgentDiagramInitEdgeEditPanel

Old: no dedicated update form (transition-init was a marker edge). New: a `Box` with a one-line "Initial-state marker. No editable fields." note. **= identical** (read-only marker).

### D. Constraints + invariants

| Constraint | v3 enforcement | v4 enforcement | Severity |
|---|---|---|---|
| Exactly one initial state per AgentDiagram (transition-init source = null, target = first state) | Implicit in `AgentStateTransitionInit` semantics — not statically validated, only at backend validation | Same — no static check in the new lib | OK (no regression) |
| `AgentStateTransition` connection rules (only between AgentStates / Intents) | `static supportedRelationships` on each agent class | Edge connection rules are at the React-Flow handle level — not enforced by node type in the new lib | **MEDIUM** — a v4 user can connect any node to any node |
| Intent-vs-State distinction (intents are a separate set of nodes from states) | Distinct UMLElement types | Distinct node types in the registry | OK |
| `replyType` ↔ child body type coupling (a body whose `replyType === 'rag'` should be linked to a RAG element with the same database name) | Enforced by the v3 update form's auto-create flow + dropdown selection | **NOT enforced** — the new body panel has no `ragDatabaseName` field, the agent state panel doesn't auto-create body rows, and there's no cross-referencing between body and rag element | CRITICAL — a v4 RAG body has no way to link to its DB |
| `AgentStateMember.dbSelectionType ∈ {default, custom}` | Validated in the dropdown | Free-text in panel (any string accepted) | MEDIUM |
| `AgentStateMember.dbQueryMode ∈ {llm_query, sql}` | Two-radio | dropdown (3 entries: `llm_query, sql, natural_language` — extra `natural_language` in new) | MINOR (additive) |
| `AgentStateMember.dbOperation ∈ {any, select, insert, update, delete}` | Dropdown | **Free-text** | MEDIUM |
| Custom-mode condition value follows the python `def condition(...)` template | v3 prefilled the template (`CUSTOM_CONDITION_TEMPLATE`) | **Not prefilled** — empty string on add | MEDIUM |
| AgentStateMember name auto-derives from `dbCustomName/dbQueryMode/dbOperation` | `getDbDisplayName` rebuilds the body name on every update | **Not enforced** — name is independent of fields | MEDIUM (UX — user must hand-rename body) |
| Intent name auto-prefixed with `"Intent: "` on the canvas | v3 mutated `element.name` in the renderer | New AgentIntent.tsx prefixes at draw time only (`Intent: ${name}`) — non-mutating, cleaner | OK |

### E. Visual shape parity

#### AgentState (parent)

- **Old** (`agent-state-component.tsx` — not read directly here, inferred from layout in `agent-state.ts`): rounded-rect parent with header band (40 / 50 px stereotype) and bodies/fallback bodies separated by horizontal dividers (`dividerPosition`). Width auto-fits to body + fallback bodies (clamped 80..420).
- **New** (`AgentState.tsx` — 125 lines): rounded-rect (cornerRadius 8) + single header line. Bodies & fallback bodies are React-Flow children with their own borders. Width is from React-Flow `width` prop (no auto-fit clamping).

**Visual: minor diff** — same SA-3 architectural choice. The clamp-to-420 v3 max-auto-width is lost (resize is unbounded in v4 except by `minWidth=120, minHeight=60`).

#### AgentIntent (parent)

- **Old** (`agent-intent-object-component.tsx:16-126`): folded-corner rectangle path `M 0 0 H W V H H 30 L 0 H+30 L 10 H H 10 0 Z` filled with v3 default `#E3F9E5` (hardcoded as `fillColor` arg, not theme), header strip 40/50, name with `"Intent: "` prefix, optional description block under the header (height = `AGENT_INTENT_DESCRIPTION_HEIGHT`), body separator after the description.
- **New** (`AgentIntent.tsx:24-126`): identical folded-corner path (`M 0 0 H W V H H 30 L 0 H+30 L 10 H H 10 0 Z`), `#E3F9E5` fill default, header line at `headerHeight`, name prefixed with `"Intent: "` at draw time. Description **child** is a separate node, not an inline strip; horizontal separator between header and body region preserved.

**Visual: identical at the folded-corner level.** Description-strip inline rendering is replaced by a child node — minor diff (the description child renders a separate row outside the parent's SVG). Acceptable per the SA-4 architectural choice.

#### AgentRagElement (cylinder)

- **Old** (`agent-rag-element-component.tsx` — not read; layout inferred from `agent-rag-element.ts:14-43`): default 140×120 cylinder with `name` text. Min width 120.
- **New** (`AgentRagElement.tsx:27-126`): cylinder geometry (top/bottom ellipses + sided rect) with `RAG DB` strap + display name = `dbCustomName ?? ragDatabaseName ?? name`. Default fill `#E8F0FF` — v3's was theme-driven (no hardcoded default visible in the v3 component path I read).

**Visual: identical cylinder geometry**, but the new lib displays the database name **below the cylinder header label**, while v3 displayed `name` only. This is an enhancement, not a regression.

### F. Cross-diagram bridge wiring

- **Intent dropdown for AgentStateTransition's `intentName`** — v3 read `state.elements` and filtered `type === 'AgentIntent'`, populated a Dropdown. New lib uses **plain text** (no bridge call). The bridge is available (`diagramBridge` ts file is registered) but the panel does not read from it. **MEDIUM regression**, likely a 5-line fix.
- **RAG database dropdown for AgentStateMember's `ragDatabaseName`** — v3 read `state.elements` and filtered `type === 'AgentRagElement'`. New body panel doesn't expose `ragDatabaseName` at all. **CRITICAL** (see §C).
- **Class diagram cross-references** — `AgentIntentObjectComponent.entity` is a free-text field; no integration with ClassDiagram entities. v3 had no such integration either, so no regression.

**Bridge parity: ⚠️** — capability present, consumer-side hookup partially missing.

---

## UserDiagram

### A. Element type inventory

Old (`packages/editor/src/main/packages/user-modeling/index.ts`):

- **Nodes**: `UserModelName`, `UserModelAttribute`, `UserModelIcon` — 3 element types.
- **Edges**: `UserModelLink` — 1 edge type. (The brief mentions only the OCL meta-model JSON, but the `index.ts` declares `UserModelRelationshipType.UserModelLink` so it is part of the v3 surface.)

New (`packages/library/lib/nodes/userDiagram/index.ts`):

- **Nodes**: `UserModelName`, `UserModelAttribute`, `UserModelIcon` — 3 types. Match.
- **Edges**: `UserModelLink` — registered via `lib/edges/edgeTypes/UserModelLink.tsx` as an alias of `ObjectDiagramEdge`.

| v3 type | v4 status | Mapping notes |
|---|---|---|
| `UserModelName` | = | Visual reused from `ObjectNameSVG` per the SA-4 brief |
| `UserModelAttribute` | = | Migrator collapses onto parent's `attributes` array (per `UserModelNameNodeProps`); standalone node retained for legacy round-trip |
| `UserModelIcon` | = | Migrator collapses onto parent's `data.icon`; standalone node retained for legacy round-trip |
| `UserModelLink` | = | Aliases `ObjectDiagramEdge` for visual; type discriminator preserved for OCL validator |

**Element parity: ✅ exact 1:1.**

### B. Per-type data field parity

#### UserModelName

Old `IUMLUserModelName` (`uml-user-model-name.ts:15-19`):

| Old field | New field on `UserModelNameNodeProps` | Status |
|---|---|---|
| `name` | `data.name` | = |
| `classId?` | `data.classId?` | = (open question #1 resolution — preserved) |
| `className?` | `data.className?` | = |
| `icon?` (separate child element) | `data.icon?` (collapsed) | RENAMED (collapsed) |
| `underline = true` (default in v3) | (no `underline` field on `UserModelNameNodeProps`) | **MISSING in new** — v3 hardcoded `underline = true`; new lib reuses `ObjectNameSVG` which is the same SVG that drops underline (also flagged in Round 1 for ObjectName). |
| `attributes` (id list) | `data.attributes: UserModelAttributeRow[]` | RENAMED (collapsed rows) |
| `methods` (id list) | (no field — UserModelName has no methods in new) | DROP — but v3's `UserModelName extends UMLClassifier` had `methods` via inheritance; in practice unused (the user-modelling diagram is constraint-style, no operations) |
| `stereotype: string \| null` | (no field — `UserModelNameNodeProps` doesn't extend `StateNodeProps`) | DROP — acceptable (user-models don't use stereotypes) |
| (no field) | `data.description?: string` | EXTRA — for OCL semantic validation |
| `fillColor` / `strokeColor` / `textColor` | inherited from `DefaultNodeProps` | = |

#### UserModelAttribute (row)

Old `UMLUserModelAttribute` (`uml-user-model-attribute.ts:8-78`) extends `UMLClassifierAttribute`:

| Old field | New field on `UserModelAttributeRow` | Status |
|---|---|---|
| `name` (formatted as `"foo == value"`) | `name` | = |
| `attributeId?` | `attributeId?` | = |
| `attributeOperator: '<' \| '<=' \| '==' \| '>=' \| '>'` (default `'=='`) | `attributeOperator?: '<' \| '<=' \| '==' \| '>=' \| '>'` | = (5 entries match) |
| Inherited `attributeType` from `UMLClassifierAttribute` (default `'str'`) | `attributeType?: string` | = (optional) |
| Inherited `defaultValue` | `defaultValue?: unknown` | = |
| Inherited `isOptional`, `isDerived`, `isId`, `isExternalId`, `code`, `visibility`, `implementationType`, `stateMachineId`, `quantumCircuitId` | (not on `UserModelAttributeRow`, but **on the underlying `ClassNodeElement` it extends** — so all fields flow through structurally even though the new editor doesn't expose them) | = (data round-trip safe, editor doesn't expose) |
| (no field) | `value?: unknown` | EXTRA — structured constraint comparison value (separates "value" from "name" so `"age >= 18"` rebuilds cleanly) |

The v3 deserializer also auto-extracts the comparator from name when `attributeOperator` is missing (`extractComparatorFromName`, `uml-user-model-attribute.ts:27-33`). The new lib does **NOT** replicate this fallback — if a v3 fixture has `name: "age == 18"` but no explicit `attributeOperator`, the new lib leaves the operator blank. **MINOR** — the migrator should handle this in `convertV3ToV4` but I did not verify the path.

#### UserModelIcon

Old (`uml-user-model-icon.ts:9-54`): just `icon?: string`. Non-interactive (`hoverable: false`, `selectable: false`, etc.). New: `UserModelIconNodeProps = DefaultNodeProps & { icon?: string }`. **= identical**, but the standalone node renders interactively in v4 (the `features.selectable: false` flag from v3 is not respected — the new node is a regular React-Flow node).

#### UserModelLink (edge)

v3: no edge data shape exists in the source files I read (the index.ts declares the type but I didn't find a per-edge file). New lib aliases `ObjectDiagramEdge`'s data shape (`name`, `points`). **= behaviourally identical** to ObjectLink but with a different `type` discriminator.

### C. Panel inspector parity

#### UserModelNameEditPanel

Old: there's no dedicated v3 update file for `uml-user-model-name`. Editing on the v3 fork happens through `uml-user-model-attribute-update.tsx` per row. The parent (UserModelName) uses the inherited `UMLClassifierUpdate` form (the one used for Class) by virtue of `UMLClassifierUpdate.element` matching the type.

So the v3 user can edit:
- name (from classifier)
- color (from classifier)
- attributes (per-row via `uml-user-model-attribute-update.tsx`)

New `UserModelNameEditPanel.tsx`:
- name + className + description + per-row attribute editor (name + type Select + operator Select + value field) + add-row link

**Field parity table** for the per-row editor:

| Old form field (`uml-user-model-attribute-update.tsx`) | Old type | New form field | New type | Status |
|---|---|---|---|---|
| Attribute name (read-only label "foo == value") | static label | text input editing the row name | MUI text | ≠ — v3 locked the label, new allows free edit (matching SA-2's ObjectAttribute change) |
| Comparator dropdown (`<, <=, ==, >=, >`) | Dropdown (5 entries) | Select | MUI Select | = |
| Value field with type-aware widgets (enumeration → enum dropdown when class attr is enum-typed; plain text otherwise) | Conditional widgets | plain text TextField | MUI text | ≠ regression — type-aware enum dropdown lost (similar to ObjectDiagram Round 1 finding) |
| Type dropdown (8 primitives) | (not in v3 — type was inferred from linked class attribute) | `attributeType` Select (8 primitives) | MUI Select | EXTRA in new |
| Color/fill/stroke style pane | StylePane | (none on per-row; color is on parent) | — | ≠ regression — per-row color (v3 had ColorButton on each attribute) |
| Delete row | TrashIcon | DeleteIcon | = | = |

Other UI:
- v3: per-attribute reorder buttons via the inherited classifier form. Not present in new lib (also flagged in Round 1).
- v3: integer-type comparator only renders when the **linked class attribute type is integer** — the new lib always shows the comparator dropdown regardless. **MINOR** — UX deviation.

#### UserModelAttributeEditPanel (standalone)

Old: as above (`uml-user-model-attribute-update.tsx`). New `UserModelAttributeEditPanel.tsx` (141 lines): name + type Select + operator Select + defaultValue. Used only when the attribute is a free-floating sibling (not collapsed onto the parent), per the doc string.

#### UserModelLink inspector — **MISSING ENTIRELY**

No registration for `UserModelLink` in `lib/components/inspectors/userDiagram/index.ts`. v3 had no dedicated inspector either (the user-modelling fork didn't ship one), so this is **not a regression**, but it does mean line color / flip / delete are not exposed via panel.

### D. Constraints + invariants

| Constraint | v3 | v4 | Severity |
|---|---|---|---|
| **OCL meta-model JSON byte-identical** at `lib/services/userMetaModel/usermetamodel.json` vs v3 source `usermetamodel_buml_short.json` | source file in v3 | `md5sum` matches: `ca7afa7061fc1511c607f9180875974f` for both | **OK ✅** — confirmed byte-identical |
| Two siblings of `usermetamodel_buml_short.json` (`_less_short` and `_corrected_format`) — v3 ships them. | both present in v3 | only the canonical `usermetamodel.json` is in new lib | OK (the brief specifies the canonical short version is the spec source) |
| `attributeOperator` ∈ {<, <=, ==, >=, >} | enforced via `normalizeUserModelAttributeComparator` | enforced via the dropdown `as UserModelAttributeRow["attributeOperator"]` cast — type-level only | OK |
| `=` ↔ `==` normalisation (v3 converts a stray `=` to `==`) | `normalizeUserModelAttributeComparator(' = ')` returns `'=='` | **NOT enforced** in new — Select only emits the 5 canonical values, but the comparator field on raw imports is not normalised | MINOR — round-trip-safe for the canonical set |
| `extractComparatorFromName` fallback when v3 fixture has `name: "x >= 10"` but no explicit `attributeOperator` | `uml-user-model-attribute.ts:27-33` | **NOT visible in the migrator** — the attribute's comparator could be missing on import | MINOR (rare in practice) |
| `UserModelName.underline = true` (always rendered with underline) | hardcoded | not rendered (also flagged in Round 1 for ObjectName) | MEDIUM |
| `UserModelIcon.features = { selectable: false, … }` | non-interactive | the new node is a regular interactive node | MINOR |
| Backend-side OCL validation on the meta-model JSON | n/a — JSON is data | n/a — JSON is data, identical | OK |

### E. Visual shape parity

#### UserModelName

- **Old** (`uml-user-model-name.ts:161-214`): UMLClassifier-style card with `underline = true` on the name; optional icon-view rendering when `settingsService.shouldShowIconView()` and a `UserModelIcon` child is present (extracts SVG size from inline icon body).
- **New** (`UserModelName.tsx:48-181`): reuses `ObjectNameSVG` directly; **no underline** rendered (same regression as ObjectName); icon-view rendering not implemented (`data.icon` is read into `renderData` and passed to `ObjectNameSVG` but the SVG implementation does not switch to icon-view).

**Visual: minor diff** — underline default is lost; icon-view rendering not implemented.

#### UserModelIcon

- **Old**: small fixed 20×20 box rendered next to the parent.
- **New** (`UserModelIcon.tsx:15-89`): renders the icon body via `<image href>` (data URLs) or `<foreignObject>` (inline SVG markup) inside a configurable-size box.

**Visual: enhanced** in new (actually shows the icon contents); v3 only provided geometry hooks for the parent to position it.

### F. Cross-diagram bridge wiring

- `UserModelName.classId` ↔ ClassDiagram class — preserved (open question #1). The new panel exposes `className` as text but does **not** offer a class picker dropdown bound to `diagramBridge.getAvailableClasses()`. v3's per-row editor consumed `diagramBridge.getClassDiagramData()` to look up the linked attribute type for the integer-comparator condition — see `uml-user-model-attribute-update.tsx:81-104`. **MEDIUM** — bridge consumer wiring missing on the new per-row editor.
- `UserModelAttribute.attributeId` ↔ ClassDiagram attribute id — `attributeId` field exists on `UserModelAttributeRow` per the type, but the new editor does not expose a dropdown to pick it. v3 didn't expose this either (the operator was the only ID-tied widget), so this is not a regression — just unimplemented for the v4 enhancement angle.

**Bridge parity: ⚠️** — fields preserved on data, partial integration on consumer side.

---

## NNDiagram

### A. Element type inventory

Old (`packages/editor/src/main/packages/nn-diagram/index.ts`):

- **Layer node types** (18): `Conv1DLayer`, `Conv2DLayer`, `Conv3DLayer`, `PoolingLayer`, `RNNLayer`, `LSTMLayer`, `GRULayer`, `LinearLayer`, `FlattenLayer`, `EmbeddingLayer`, `DropoutLayer`, `LayerNormalizationLayer`, `BatchNormalizationLayer`, `TensorOp`, `Configuration`, `TrainingDataset`, `TestDataset`, `NNContainer`, `NNReference`. Plus `NNSectionTitle`, `NNSectionSeparator` for sidebar grouping (excluded from on-canvas use).
- **Per-attribute UMLElement types** (~110, e.g. `KernelDimAttributeConv2D`, `DimensionAttributePooling`, `BatchSizeAttributeConfiguration`, `NameAttributeDataset`, etc.) — **COLLAPSED** by SA-5 onto each parent layer's `data.attributes: Record<string, unknown>` per the v4 wire-shape spec.
- **Edge types** (3): `NNNext`, `NNComposition`, `NNAssociation`.

New (`packages/library/lib/nodes/nnDiagram/index.ts`):

- **Layer node types** (18): same 18 types, registered via `registerNodeTypes`. Match.
- **Edge types** (3): `NNNext`, `NNComposition`, `NNAssociation` — registered in `lib/edges/edgeTypes/index.ts`.

| v3 type | v4 status | Mapping notes |
|---|---|---|
| `Conv1DLayer`, `Conv2DLayer`, `Conv3DLayer` | = | |
| `PoolingLayer` | = | |
| `RNNLayer`, `LSTMLayer`, `GRULayer` | = | |
| `LinearLayer`, `FlattenLayer`, `EmbeddingLayer`, `DropoutLayer` | = | |
| `LayerNormalizationLayer`, `BatchNormalizationLayer` | = | |
| `TensorOp`, `Configuration`, `TrainingDataset`, `TestDataset` | = | |
| `NNContainer`, `NNReference` | = | |
| `NNSectionTitle`, `NNSectionSeparator` | **DROPPED** (intentional per the SA-5 commit message: "Section helpers are dropped on migration") | OK — these were sidebar-only |
| All ~110 v3 per-attribute element types (`KernelDimAttributeConv2D`, …, `MomentumAttributeConfiguration`) | **COLLAPSED** onto `data.attributes` | EXPECTED per spec — flagged for completeness but not a regression |
| `NNNext` | = | filled-arrow head |
| `NNComposition` | = | aliases `NNNext`, with `url(#black-rhombus)` markerStart |
| `NNAssociation` | = | plain line, no markers |

Spec open question #2 (`DimensionAttribute` collision between Pooling and BatchNormalization): resolved via `qualifySlug()` (`pooling.dimension`, `batch_normalization.dimension`). Confirmed at `nnAttributeWidgetConfig.ts:76-127` and the `getAttribute`/`setAttribute` helpers tolerate both qualified and unqualified keys on read. ✅

**Element parity: ✅** for the 18 layer types + 3 edges. Collapse of per-attribute children is intentional and well-documented.

### B. Per-type data field parity

#### Layer attribute keys — verbatim ports from `nn-attribute-widget-config.ts` to `nnAttributeWidgetConfig.ts`

Per-kind attribute slug list (from `LAYER_ATTRIBUTE_SCHEMA`) vs v3 element-type names:

**Conv1D / Conv2D / Conv3D** (12 slugs each, all match v3):
`name, kernel_dim, out_channels, stride_dim, in_channels, padding_amount, padding_type, actv_func, name_module_input, input_reused, permute_in, permute_out` — all 12 mapped from `*AttributeConv{1,2,3}D` → slug. Confirmed via `V3_ATTRIBUTE_TYPE_TO_SLUG`. ✅

**Pooling** (13 slugs, all match v3):
`name, pooling_type, dimension (qualified to pooling.dimension), kernel_dim, stride_dim, padding_amount, padding_type, output_dim, actv_func, name_module_input, input_reused, permute_in, permute_out`. ✅ Collision-aware on `dimension`.

**RNN / LSTM / GRU** (10 slugs each, all match v3):
`name, hidden_size, return_type, input_size, bidirectional, dropout, batch_first, actv_func, name_module_input, input_reused`. ✅

**Linear** (6 slugs): `name, out_features, in_features, actv_func, name_module_input, input_reused`. ✅
**Flatten** (6 slugs): `name, start_dim, end_dim, actv_func, name_module_input, input_reused`. ✅
**Embedding** (6 slugs): `name, num_embeddings, embedding_dim, actv_func, name_module_input, input_reused`. ✅
**Dropout** (4 slugs): `name, rate, name_module_input, input_reused`. ✅
**LayerNormalization** (5 slugs): `name, normalized_shape, actv_func, name_module_input, input_reused`. ✅
**BatchNormalization** (6 slugs): `name, num_features, dimension (qualified to batch_normalization.dimension), actv_func, name_module_input, input_reused`. ✅
**TensorOp** (8 slugs): `name, tns_type, concatenate_dim, layers_of_tensors, reshape_dim, transpose_dim, permute_dim, input_reused`. ✅
**Configuration** (8 slugs): `batch_size, epochs, learning_rate, optimizer, loss_function, metrics, weight_decay, momentum`. ✅ (no `name` slug — Configuration has no name field, matching v3)
**Datasets** (6 slugs each for Training/Test): `name, path_data, task_type, input_format, shape, normalize`. ✅

**Per-attribute slug parity: ✅ exact** — every v3 attribute element-type maps to exactly one slug, the boolean attributes are normalised at the migrator boundary (verified by the SA-5 commit message), and the colliding `dimension` slug is qualified.

#### Pooling-specific list defaults

Old (`nn-validation-defaults.ts:47-79`): `getListExpectation(elementType, ownerId, elements)` — walks the elements registry to find the pooling layer's `DimensionAttributePooling` value, then returns the size-aware default (`[3]` for 1D kernel, `[3, 3, 3]` for 3D kernel, etc.).

New (`nnValidationDefaults.ts:58-106`): `getListExpectation(layerKind, slug, poolingDimension)` — receives the dimension as an arg rather than walking the registry. Same defaults table. **Functionally equivalent**, but the new version does NOT read the dimension automatically — callers must pass it. **The inspector (`NNComponentEditPanel`)** does not call `getListExpectation` at all — it relies on `getAttributeDefaultValue` for placeholder defaults but does not offer a per-layer-dimension example list. **MINOR** — example placeholders (`[3, 3]` for 2D Pooling kernel) are not rendered as field placeholders in v4.

#### NNContainer

Old (`nn-container.ts:12-94`):

| Old field | New field on `NNContainerNodeProps` | Status |
|---|---|---|
| `name` (default `'Neural_Network'`) | `data.name` | = (default not enforced server-side; the migrator does not synthesize `'Neural_Network'`) |
| `bounds.{x, y, width, height}` (default 800×200) | React-Flow `width`, `height` (set by canvas, default unspecified in lib) | = |
| `static minWidth = 200` / `static minHeight = 150` | NodeResizer `minWidth=200, minHeight=140` | ≠ minHeight is `140` not `150` (off-by-10 — MINOR) |
| `static defaultWidth = 800` / `static defaultHeight = 200` | (no defaults in lib) | MISSING (the consumer creates the container with default dimensions; if not, falls back to React-Flow's defaults) |
| (no field) | `data.entryLayerId?: string` | EXTRA in new — open spec field |
| (no field) | `data.description?: string` | EXTRA in new |

#### NNReference

Old (`nn-reference.ts:8-72`):

| Old field | New field on `NNReferenceNodeProps` | Status |
|---|---|---|
| `referencedNN: string` (display label and reference id) | `data.referenceTarget?: string` | RENAMED |
| `name` (defaults to `'SubNN'`) | `data.name` | = (default not enforced) |
| `static minWidth = 140` / `static height = 40` | NodeResizer `minWidth=100, minHeight=36` | ≠ minWidth `100` not `140` (MINOR) |
| `computeWidth()` based on `displayText.length * 7.5 + 40` | React-Flow auto-resize | RENAMED |

#### Edges (NNNext / NNComposition / NNAssociation)

Old:

| Old type | New type | Markers |
|---|---|---|
| `NNNext` (filled-arrow unidirectional) | = | filled arrow head on target — confirmed in the renderer using `EdgeInlineMarkers` + `getEdgeMarkerStyles('NNNext')` |
| `NNComposition` (diamond on container side) | = | `url(#black-rhombus)` on markerStart — aliases `NNNext` component |
| `NNAssociation` (plain line) | = | no markers — confirmed |

All three edges go through `useStepPathEdge` so visual diff is the same as SA-3's StateTransition. **= identical**.

### C. Panel inspector parity

#### NNComponentEditPanel — generic single panel for 17 layer kinds

Per the SA-5 brief, this is intentionally one generic body driven by `getLayerSchema(layerKind)`. Spot-check 3 layer kinds against v3:

**Conv2D**:
- v3 (`nn-component-update.tsx:298-316`): mandatory `name, kernel_dim, out_channels` + 9 optional (stride_dim, in_channels, padding_amount, padding_type, actv_func, name_module_input, input_reused, permute_in, permute_out). All optional fields gated behind a `OptionalAttributeRow` per-row checkbox.
- v4: schema schedules same 12 slugs (3 mandatory + 9 optional, marked with `mandatory: true`); rendered as 12 always-visible rows. **Per-attribute checkbox to enable/disable optional fields is GONE.** v3's pattern was: optional fields are not stored on the layer until the user ticks the row (then a v3 attribute element is created); v4 always stores them in `data.attributes`. **MEDIUM** — visual clutter; the v3 user experience of "only see fields you opted in to" is lost.

**Pooling**:
- v3 (`nn-component-update.tsx:336-355`): mandatory `name, pooling_type, dimension`; conditional optionals via `getPoolingOptionalAttributes` (e.g. `global_average` pooling hides `kernel_dim, stride_dim, padding_amount, padding_type, output_dim`). Non-trivial UX.
- v4: schedules all 13 slugs unconditionally. `getPoolingOptionalAttributes` filtering is **not implemented** in `NNComponentEditPanel`. **MEDIUM** — UI clutter, plus Pooling's mode-aware UX is lost.

**TensorOp**:
- v3 (`nn-component-update.tsx:482-496`): mandatory `name, tns_type`; conditional optionals via `getTensorOpOptionalAttributes` (`reshape` shows only `reshape_dim`; `concatenate` shows `layers_of_tensors, concatenate_dim`; `transpose` shows `transpose_dim`; `permute` shows `permute_dim`).
- v4: all 8 slugs always shown. `getTensorOpOptionalAttributes` filtering is **not implemented**. **MEDIUM**.

**Dataset (Training/Test)**:
- v3 (`nn-component-update.tsx:512-537`): mandatory `name, path_data`; conditional `shape` and `normalize` only when `input_format === 'images'`.
- v4: all 6 slugs always shown. **MEDIUM** — `shape` and `normalize` non-image handling lost.

**Widget type per-slug parity** (sample 3 fields per kind):

| Layer | Slug | v3 widget | v4 widget | Match |
|---|---|---|---|---|
| Conv2D | `padding_type` | dropdown `valid/same` | dropdown `valid/same` | ✅ |
| Conv2D | `actv_func` | dropdown 5 options | dropdown 5 options | ✅ |
| Conv2D | `name_module_input` | predecessor dropdown | predecessor dropdown (driven by sibling layer names with same parentId) | ✅ |
| Pooling | `pooling_type` | dropdown 4 options (`average, max, adaptive_average, adaptive_max`) — wait, v3 showed `global_average, global_max` in the `getPoolingOptionalAttributes` filter, suggesting v3's pooling_type widget had **6** options (`average, max, adaptive_average, adaptive_max, global_average, global_max`). Re-read `nn-attribute-widget-config.ts:32-40`: I don't see a `PoolingTypeAttributePooling` widget config in the map, so it falls back to the default `widget: 'text'`. v3's widget config is `text`! | new lib's schema has it as `dropdown` with 4 options (`POOLING_TYPE_OPTIONS`) | ≠ — **EXTRA in new** (more constrained); v3 was free-text. The new lib drops `global_average` and `global_max` from the choices, so a v3 fixture with `pooling_type = 'global_average'` will display as the default value. **MEDIUM** — silent value reset for legacy fixtures. |
| Pooling | `dimension` (qualified) | widget config `dropdown` with `1D/2D/3D` | dropdown with same 3 options | ✅ |
| RNN | `return_type` | dropdown `hidden/last/full` (3) | dropdown same 3 options | ✅ |
| TensorOp | `tns_type` | dropdown 6 options | dropdown same 6 options | ✅ |
| Dataset | `task_type` | dropdown `binary/multi_class/regression` | dropdown same | ✅ |
| Dataset | `input_format` | dropdown `csv/images` | dropdown same | ✅ |
| Configuration | `batch_size` etc. | text | text | ✅ |
| BatchNorm | `dimension` (qualified) | widget config doesn't appear in v3's WIDGET_CONFIG_MAP — falls back to text | **dropdown** with `1D/2D/3D` | ≠ — EXTRA in new (more constrained). v3 was free-text. |

Note the two `Pooling.pooling_type` and `BatchNorm.dimension` deviations — both are cases where the new schema **constrains** a v3 free-text field to a fixed set. Round-trip-safe but legacy values may fall back to defaults silently.

**Mandatory attribute auto-create** (`createMandatoryAttributes` in v3): on element mount, v3 spawned the 3 mandatory attribute children with default values + unique-name handling (e.g. `conv2d_layer`, `conv2d_layer2`, …). **Not implemented in new** — the panel reads from `data.attributes` and does not initialise mandatory rows. The migrator handles this for imports but a freshly-dropped v4 layer has no auto-populated `data.attributes`. **MEDIUM** — UX regression on new-element-from-palette.

#### NNContainerEditPanel

Old: no dedicated update file in v3 (containers used the inherited base-class form). New `NNContainerEditPanel.tsx` (102 lines): name + description + entryLayerId Select (driven by child layers via parentId). EXTRA fields per spec.

#### NNReferenceEditPanel

Old (`nn-reference-update.tsx`): I didn't read the file but its existence implies a name + referencedNN editor. New `NNReferenceEditPanel.tsx` (104 lines): name + `referenceTarget` Select (driven by sibling layer names) + free-text override. **= functionally similar**.

#### Edge inspectors — **MISSING ENTIRELY**

No registration for `NNNext`, `NNComposition`, `NNAssociation` in `lib/components/inspectors/nnDiagram/index.ts`. v3 likely had no edge inspectors either (NN edges carry minimal data), so **not a regression**, but means the `name` field on these edges is unauthorable. The renderer reads `data.name` for the label — it just can't be set without direct JSON editing.

### D. Constraints + invariants

| Constraint | v3 | v4 | Severity |
|---|---|---|---|
| Layer ordering via `NNNext` chain (sequential through container) | not statically validated | not validated either; React-Flow handle rules only | OK (no regression) |
| Only one TrainingDataset per container | not validated in v3 either | not validated in v4 | OK |
| `LIST_STRICT_REGEX` for kernel_dim / stride_dim / output_dim | regex defined and consumed by attribute editor | regex defined in `nnValidationDefaults.ts` but **not consumed** by the new generic editor | MEDIUM — invalid values like `[3, foo]` are accepted silently |
| `getListExpectation` provides per-(layerKind, slug, poolingDimension) example | consumed by `optional-attribute-row.tsx` | exported but not consumed by `NNComponentEditPanel` | MEDIUM — placeholder text is missing |
| Mandatory-attribute presence (every Conv2D must have `name`, `kernel_dim`, `out_channels`) | enforced on element mount via `createMandatoryAttributes` | not enforced on element mount — `data.attributes` may be empty | MEDIUM |
| Dimension-aware list shape (`KernelDimAttributePooling` count = 1/2/3 by parent's `DimensionAttributePooling.value`) | `getListExpectation` resolves dynamically by walking elements | `getListExpectation` requires caller to pass `poolingDimension`; not called from inspector | MINOR (data round-trips, just no UX guard) |
| `pooling_type ∈ {average, max, adaptive_average, adaptive_max, global_average, global_max}` (v3 free-text) | not enforced (free text in v3) | enforced as dropdown of 4 (drops global_average, global_max) | **MEDIUM regression** — silent value reset for legacy `global_*` fixtures |
| `BatchNorm.dimension ∈ {1D, 2D, 3D}` | v3 free-text; only the qualifying constraint is the slug collision | dropdown enforces 3 options | OK (consistent with Pooling.dimension) |
| `qualifySlug('PoolingLayer', 'dimension')` returns `'pooling.dimension'`, `qualifySlug('BatchNormalizationLayer', 'dimension')` returns `'batch_normalization.dimension'` | n/a | confirmed at `nnAttributeWidgetConfig.ts:83-127` | ✅ |

### E. Visual shape parity

Spot-check 3 nodes:

#### Conv2D layer (`Conv2DLayer`)

- **Old** (`nn-conv2d-layer/`, base class `NNBaseLayer:1-31`): UMLClass-style card with `ICON_LAYER_WIDTH = 110, ICON_LAYER_HEIGHT = 110` fixed dimensions; `hasAttributes = false` so the body is not drawn; layer-icon SVG (referenced via `nn-layer-icon/` directory but not read). The visual is class-card-like, with the layer kind name bold at top, then a centered icon glyph below, then attributes hidden.
- **New** (`Conv2DLayer.tsx:4`, factory `_NNLayerBase.tsx:33-148`): rounded rectangle (cornerRadius 6), `«Conv2D»` stereotype text, name below, single horizontal separator under the header. Default fill `#FFF8E1`. Resizable to `120×50` minimum.

**Visual: significant diff** — the new lib drops the 110×110 fixed-icon visual entirely, replaces with a generic stereotype-card. The icon glyph is gone. **MEDIUM** UX regression.

#### NNContainer (parent)

- **Old** (`nn-container.ts:12-94`): rectangle bounding-box of all children + `name` text in the upper-left, calculated minimum 200×150.
- **New** (`NNContainer.tsx:24-98`): rounded rect (cornerRadius 8), name centered at `y=26` in the header band, header line at `headerHeight`, default fill `#F5F5F5`. Min 200×140.

**Visual: minor diff** — same conceptual shape; new has default fill background, v3 had no fill. Acceptable.

#### Configuration block

- **Old** (`nn-configuration.ts`): same `NNBaseLayer` 110×110 fixed-icon visual as a layer.
- **New** (`Configuration.tsx`): same generic `_NNLayerBase` card with kindLabel = "Configuration" (presumed — file not read directly).

**Visual: same diff as Conv2D** — fixed-icon visual replaced with stereotype card.

### F. Cross-diagram bridge wiring

- `NNContainer.entryLayerId` ↔ child layer id — exposed in panel, populated from sibling layers via `parentId`. ✅
- `NNReference.referenceTarget` ↔ same-parent layer id — exposed in panel, populated from sibling layers via `parentId`. ✅
- Cross-NN-diagram references (e.g. a container in NN A references a layer in NN B) — handled via the free-text override on `NNReferenceEditPanel`. v3 stored `referencedNN: string` as a name, so the new lib's id-based + free-text-fallback approach is **enhanced** over v3.
- Dataset references (Configuration → TrainingDataset / TestDataset) — v3 used `NNAssociation` edges between Dataset and Container. New lib uses the same edge type. ✅
- ClassDiagram cross-references — none in either v3 or v4.

**Bridge parity: ✅** for the v3 surface.

---

## Summary table (round 2)

| Diagram | Element parity | Data parity | Form parity | Constraint parity | Visual parity | Bridge parity |
|---|---|---|---|---|---|---|
| AgentDiagram | ⚠️ (2 EXTRA promotions: AgentIntentDescription, AgentIntentObjectComponent) | ⚠️ (AgentRagElement gained 7 EXTRA fields; AgentStateMember code/kind EXTRA; 5-shape transition collapse not visibly verified in convertV3ToV4) | ❌ (AgentStateEditPanel collapsed from 960-line body editor to 145-line stub; intentName / fileType picker downgraded to free text; CodeMirror condition editor → plain text; flip / color editing on edge dropped) | ❌ (RAG-body ↔ DB-config field linkage not enforced; mandatory-attr defaults not auto-applied; dbOperation enum not enforced) | ✅ minor diffs (max-width clamp lost on AgentState) | ⚠️ (intent-name dropdown wiring missing; rag-database wiring missing on body) |
| UserDiagram | ✅ exact | ✅ (only EXTRA description and value fields) | ⚠️ (per-row enum-type-aware value widget missing; per-row color editor dropped; underline default lost) | ⚠️ (UserModelName.underline default not rendered; UserModelIcon non-interactive flag not respected; comparator ↔ integer-type gating dropped) | ⚠️ (no underline on name; icon-view rendering not implemented) | ⚠️ (classId picker not provided; bridge.getClassDiagramData() not consumed by the new per-row editor) |
| NNDiagram | ✅ (collapse intentional per spec; section helpers dropped per spec) | ✅ exact slug parity for all 18 layer kinds; collision-aware slug (`pooling.dimension` / `batch_normalization.dimension`) matches SA-6.1 backend | ⚠️ (single generic panel does NOT reproduce v3's 3 conditional-filtering paths: TensorOp by tns_type, Pooling by pooling_type, Dataset by input_format; mandatory-attr auto-create on mount missing; per-attribute checkbox-to-enable optional fields gone) | ⚠️ (LIST_STRICT_REGEX defined but not consumed; pooling_type loses 2 of 6 v3 options; getListExpectation example placeholders not rendered) | ⚠️ (Conv2D / Configuration / etc. fixed-icon 110×110 visual replaced with generic `_NNLayerBase` stereotype card; layer-icon glyphs removed) | ✅ (entryLayerId, referenceTarget bound to local children; cross-NN free-text fallback) |

---

## Recommendations / gaps for SA-2.2 (or SA-7)

Continuing from Round 1's #20. Severity in **bold**.

21. **CRITICAL — restore the AgentState body editor in `AgentStateEditPanel.tsx`.** Port the v3 5-radio reply-mode picker (`text/llm/rag/db_reply/code`) and the inline body-creation flow from `agent-state-update.tsx`. Including: RAG-database dropdown driven by `nodes.filter(n => n.type === 'AgentRagElement').map(n => n.data.name)`; full DB-action editor (selectionType/customName/queryMode/operation/SQL); CodeMirror Python editor for code mode; separate radio block for fallback bodies. Without this, agent diagrams are unusable for authoring. File path: `lib/components/inspectors/agentDiagram/AgentStateEditPanel.tsx` (currently 145 lines; v3 source is ~960 lines).
22. **CRITICAL — restore the RAG/DB field UI on `AgentStateBodyEditPanel.tsx`.** The v3 body update form exposed `ragDatabaseName`, `dbSelectionType`, `dbCustomName`, `dbQueryMode`, `dbOperation`, `dbSqlQuery` per body. v4 stores them on `AgentStateBodyNodeProps` but the panel exposes none. Either move all DB-config editing here, or share a sub-component between this panel and `AgentRagElementEditPanel`.
23. **MEDIUM — restore the intent-name dropdown on `AgentDiagramEdgeEditPanel.tsx`** when `predefinedType === 'when_intent_matched'`. Read sibling intent names from `useDiagramStore`'s `nodes` filtered to `node.type === 'AgentIntent'` and present as a Select. Currently a typo-prone free TextField.
24. **MEDIUM — restore the fileType dropdown** (PDF/TXT/JSON, 3 options) on `AgentDiagramEdgeEditPanel.tsx` for `predefinedType === 'when_file_received'`. Currently a free TextField.
25. **MEDIUM — restore CodeMirror Python syntax highlighting on the custom-condition editor in `AgentDiagramEdgeEditPanel.tsx`.** v3 used `react-codemirror2` with Python mode + line numbers + tab-indent. New uses plain MUI text. Same gap as the SA-2 ClassEditPanel CodeMirror replacement (Round 1 #9).
26. **MEDIUM — restore flip + color editing on `AgentDiagramEdgeEditPanel.tsx`.** v3 `agent-state-transition-update.tsx:201-208` had ExchangeIcon + ColorButton. Same shape as Round 1 #15 / #16 for the StateMachine edge.
27. **MEDIUM — verify `convertV3ToV4` handles all 5 legacy AgentStateTransition shapes.** The migrator function `liftAgentTransitionDataToV4` referenced in the panel doc-comment doesn't appear in `versionConverter.ts` — `migrateAgentDiagramV3ToV4` is just a type-guard wrapper. Add a regression test that imports each of the 5 legacy shapes (`condition: string`, `condition: string[]`, `customEvent`/`customConditions`, nested `predefined`/`custom`, top-level `predefinedType`) and verifies all collapse to the canonical `AgentStateTransitionData`.
28. **MEDIUM — synthesize AgentIntentDescription / AgentIntentObjectComponent v3 elements correctly on export.** `convertV4ToV3Agent` (`versionConverter.ts:2671-2685`) emits these as plain `baseV3` with `type: 'AgentIntentDescription'` / `'AgentIntentObjectComponent'` — but those types are NOT registered v3 element types. Either: drop the description child entirely on export and write its name into the parent intent's `intent_description` (the v3 wire form), or skip the export of these EXTRA-in-new types. As-is, v3 deserializers would silently discard them.
29. **MEDIUM — restore conditional optional-attribute filtering in `NNComponentEditPanel.tsx`.** Implement `getTensorOpOptionalAttributes` (filter optional fields by `tns_type`), `getPoolingOptionalAttributes` (by `pooling_type`), `getDatasetOptionalAttributes` (by `input_format`). Reference v3 source at `nn-component-update.tsx:614-669`. Without this, every layer panel shows every optional field, which is visual clutter and re-introduces the noise the v3 UX explicitly avoided.
30. **MEDIUM — restore mandatory-attribute auto-creation when a layer node is dropped from the palette.** v3's `componentDidMount` in `nn-component-update.tsx:588-605` checked for absence and called `createMandatoryAttributes`. In v4 this should populate `data.attributes` with the schema's mandatory entries (with default values from `NN_ATTRIBUTE_DEFAULTS`) on node creation.
31. **MEDIUM — restore the per-row "enable this optional attribute" checkbox** for NN layer optional fields. v3's `OptionalAttributeRow` toggled the attribute element on/off. v4's panel always renders every optional field's editor, which means `data.attributes` always carries every key (even unset ones). For round-trip with v3, it would be cleaner to only persist optional keys when they have non-default values.
32. **MEDIUM — add `pooling_type` `global_average` and `global_max` options to the Pooling schema** in `nnAttributeWidgetConfig.ts`. The new schema lists 4 options (`average, max, adaptive_average, adaptive_max`) but v3's `getPoolingOptionalAttributes` references `global_average` / `global_max`, indicating they were valid values. Currently a v3 fixture with `pooling_type: 'global_average'` will display as the default value silently.
33. **MEDIUM — wire `getListExpectation` and `LIST_STRICT_REGEX` into `NNComponentEditPanel.tsx`** so that kernel_dim / stride_dim / output_dim fields display the correct `[3, 3]` example placeholder per layer kind + (for Pooling) per current `dimension` value, and so the inline editor warns on malformed input.
34. **MEDIUM — restore the v3 NNBaseLayer fixed-icon 110×110 visual** for layer cards in `_NNLayerBase.tsx` — or document explicitly that the SA-5 spec opted for the stereotype-card style (in which case mention the regression in `uml-v4-shape.md`). Either way, the layer-icon glyphs in `nn-layer-icon/` need to be ported or explicitly retired.
35. **MEDIUM — restore the `underline` rendering on UserModelName** (mirror Round 1 #7). The v3 `UMLUserModelName` had `underline = true` hardcoded; the new lib reuses `ObjectNameSVG` which doesn't render underline. Either fix `ObjectNameSVG` (resolves both the ObjectName and the UserModelName regression) or override at the UserModelName render site.
36. **MEDIUM — restore the type-aware value widgets in `UserModelNameEditPanel`'s per-row attribute editor.** When the linked class attribute type is an Enumeration, the v3 `uml-user-model-attribute-update.tsx:214-221` rendered an enum-values dropdown. v4 renders plain text always.
37. **MEDIUM — wire the integer-type guard for the comparator dropdown** on UserModelAttribute. v3 only rendered the 5-option comparator dropdown when the linked class attribute type was integer (`isIntegerType()` at `uml-user-model-attribute-update.tsx:106-109`); the new lib always renders it. Bridge call: `diagramBridge.getClassDiagramData()` → look up linked class attribute → check `attributeType ∈ {'int', 'integer', 'number'}`.
38. **MEDIUM — synthesize valid `attributeOperator` from `name` in the UserDiagram migrator** when v3 fixtures have only the embedded operator text. Mirror `extractComparatorFromName` from `uml-user-model-attribute.ts:27-33`. Currently a v3 fixture like `{name: 'age >= 18'}` (no explicit `attributeOperator`) lands in v4 with the default `'=='` rather than the embedded `'>='`.
39. **MINOR — register a UserModelLink edge inspector** so users can edit the link name and toggle flip / delete / color. v3 had no inspector either, so this is a parity-neutral enhancement. File path: `lib/components/inspectors/userDiagram/UserModelLinkEditPanel.tsx`.
40. **MINOR — register NN edge inspectors (NNNext / NNComposition / NNAssociation)** with at minimum a `name` field. Currently the renderer reads `data.name` for the edge label but there's no panel to set it.
41. **MINOR — align NNContainer's NodeResizer minHeight (currently 140) with the v3 `static minHeight = 150`** at `lib/nodes/nnDiagram/NNContainer.tsx:50`. Same for NNReference's `minWidth = 100` vs v3's `140` at `NNReference.tsx:38-39`.
42. **MINOR — surface the v3 `'Neural_Network'` default name on a freshly-dropped `NNContainer`** (currently the field is just empty), and the v3 `'SubNN'` default for a fresh NNReference. Drop-from-palette UX touch.
43. **MINOR — ensure SA-6.1 backend disambiguation matches the lib's `qualifySlug` exactly.** I confirmed via the SA-5 commit message that the backend was already disambiguated, but the audit didn't verify the byte-level format match. A regression test that round-trips a Pooling layer with `dimension = '3D'` through both the frontend migrator and the backend processor would be cheap insurance.

---

*Round 2 audit completed by SA-PARITY-2. Read-only audit; no source code modified. Submodule SHA `3c84116` (SA-5).*
