# PC-2 — Class Edit Panel parity audit

Read-only audit of the Class inspector migration.

- **v3 source-of-truth (old)**:
  - `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-update.tsx`
  - `…/uml-classifier-attribute-update.tsx`
  - `…/uml-classifier-method-update.tsx`
- **v4 target (new)**: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/classDiagram/ClassEditPanel.tsx`

Status legend: **PASS** = feature is present and matches the brief. **GAP** = missing, regressed, or behaves differently from the brief. **N/A in v3** = the brief asks v4 to expose something v3 did not (so the new panel is judged on its own).

---

## 1. Class-level fields

| Field / widget | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Class name | `Textfield` + `rename` sanitises with `/[^a-zA-Z0-9_]/g` (line 475) | `NodeStyleEditor` name input + `safeIdentifier()` in `handleDataFieldUpdate` (line 60, 750) | **PASS** | Sanitisation regex matches v3 verbatim. |
| Stereotype dropdown | `Switch` between `Class` / `AbstractClass` / `Enumeration` (Interface commented out, line 233) | `StereotypeButtonGroup` exposes `Abstract` / `Interface` / `Enumeration` | **PASS (improved)** | v4 re-enables Interface that v3 had disabled. |
| Italic checkbox | Implicit — driven by the type switch (`AbstractClass.italic = true`) | Implicit — driven by `StereotypeButtonGroup` (no explicit checkbox) | **GAP** | Brief asks for explicit italic checkbox. Neither v3 nor v4 expose one as a discrete control; the inspector relies on stereotype side-effects. v4 is no worse than v3, but the brief flags this as required. |
| Underline checkbox | Implicit — `underline` is a model property mutated by element ctor only | Not exposed | **GAP** | Same caveat as italic — implicit only. Brief calls for explicit toggle. |
| Freeform stereotype text | Not supported | Not supported | **GAP** | Neither version offers a free-text stereotype field. Brief flags this as required. |
| Fill colour | `StylePane fillColor` toggle | `NodeStyleEditor` colour panel → `fillColor` | **PASS** | |
| Stroke / line colour | `StylePane lineColor` | `NodeStyleEditor` colour panel → `strokeColor` | **PASS** | |
| Text colour | `StylePane textColor` | `NodeStyleEditor` colour panel → `textColor` | **PASS** | |
| Description | `StylePane showDescription` (line 350 of style-pane) | Not surfaced — `NodeStyleEditor` has no description field | **GAP** | v4 does not propagate `description` to the inspector body. |
| URI | `StylePane showUri` (line 364) | Not surfaced | **GAP** | Same — `uri` field absent from v4 inspector. |
| Icon | `StylePane showIcon` (line 378) | Not surfaced | **GAP** | Same — `icon` field absent. |
| OCL constraints | Not editable in v3 inspector (separate `UMLClassOCL` element on canvas) | `OCLConstraintRow` list with name + multi-line OCL expression, add/delete buttons | **PASS (new in v4)** | Matches the brief; v4 inlines what v3 left as a sibling node. |
| Delete class button | `<TrashIcon>` next to header (line 210) | None — relies on canvas-level delete | **GAP** | Minor regression; brief did not call this out explicitly. |

---

## 2. Per-attribute row

| Field / widget | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Attribute name | `NameField` + `replace(/[^a-zA-Z0-9_]/g, '')` (attribute-update line 217 / 288) | `MuiTextField` + same regex (panel line 198) | **PASS** | |
| Type dropdown — primitives | `PRIMITIVE_TYPES` array of 9 entries | Identical 9-entry `PRIMITIVE_TYPES` constant (panel line 70) | **PASS** | List ported verbatim. |
| Type dropdown — class names | **Not present in v3** (only primitives + enumerations) | Inserted as `── classes ──` group, populated from `diagramBridge.getAvailableClasses()` | **PASS (improved)** | Matches the brief; v4 surface is strictly larger. |
| Type dropdown — enumeration literals | `availableEnumerations` group (line 271) | `enumerationNames` group (panel line 223), pulled via `collectEnumerationNames()` | **PASS** | |
| Type dropdown — custom / free-text | Not supported | `custom…` sentinel + free-text `MuiTextField` row (panel line 242) normalised through `normalizeType()` | **PASS (improved)** | |
| Visibility selector | Dropdown showing **only** `+ - # ~` symbols (label = symbol, attribute-update line 116) | Dropdown showing `"+ public" / "- private" / …` — **labels include the full word** (panel line 82) | **GAP** | Brief explicitly says the v4 control "must show only `+ - # ~` symbols, not full words". v4 currently violates this. |
| Default value | `StylePane onDefaultValueChange` plain string field (attribute-update line 369) | Inline `MuiTextField` plain string (panel line 308) | **PASS** | |
| `isId` flag | `StylePane onIdChange` checkbox (attribute-update line 444) — **collapsed inside the colour pane** | `Checkbox` rendered **inline below the row** (panel line 257) | **GAP** | Brief: "should be hidden behind a per-row settings expand, NOT inline." v4 places these flags on a permanent inline row. |
| `isExternalId` flag | `StylePane onExternalIdChange` (attribute-update line 446) — collapsed | Inline checkbox (panel line 274) | **GAP** | Same as above. |
| `isOptional` flag | `StylePane onOptionalChange` (attribute-update line 440) — collapsed | Inline checkbox (panel line 287) | **GAP** | Same as above. |
| `isDerived` flag | `StylePane onDerivedChange` (attribute-update line 442) — collapsed | Inline checkbox (panel line 297) | **GAP** | Same as above. |
| Mutual exclusion (`isId` ↔ `isOptional`) | Enforced (attribute-update lines 347 & 360) | Enforced (panel line 263) | **PASS** | Logic preserved. |
| Per-attribute colour pane | `StylePane fillColor textColor` per row | Not present per-row | **GAP** | Minor; brief did not require per-row colour. |
| Reorder up/down buttons (per attribute row) | `ReorderControls` with `ArrowUpIcon` / `ArrowDownIcon` (uml-classifier-update lines 254-274) | **Not implemented** in v4 | **GAP** | Brief flags this explicitly. |
| Delete row button | Trash icon | `IconButton` + `DeleteIcon` (panel line 235) | **PASS** | |
| Add attribute control | `Textfield` with placeholder `"+ attribute: str"` and `onSubmit` (line 305) | `MuiTextField` with placeholder `"+ Add attribute (Enter)"` (panel line 950) | **PASS** | |

---

## 3. Per-method row

| Field / widget | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Method name | `NameField` (method-update line 363) | `MuiTextField` (panel line 394) | **PASS** | v4 keeps the name purely as the identifier (no embedded params), v3 used the field for the whole signature `+ name(params): ret`. Different model, both valid. |
| Return type | Not a discrete field — encoded into the name signature string | `Select` reusing `PRIMITIVE_TYPES` + class names + custom (panel line 410) | **PASS (improved)** | Brief asks for a discrete `returnType` selector — v4 delivers it. |
| Visibility | Dropdown showing only `+ - # ~` (method-update line 124) | Dropdown showing `"+ public" / …` — full words (panel line 388) | **GAP** | Same regression as attribute visibility. |
| Parameters list | None — parameters are typed into the name string and parsed by `parseMethod` (method-update line 224) | First-class `parameters: ClassifierMethodParameter[]` editor with name+type per row, add/delete (panel line 460-528) | **PASS (improved)** | Brief explicitly asks for a parameters list. |
| Code (CodeMirror Python) | `react-codemirror2` `mode: 'python'` (method-update line 11-14, 478) | `@uiw/react-codemirror` + `@codemirror/lang-python` (panel line 14-15, 625) | **PASS** | Library swapped (v3 = CodeMirror 5 wrapper, v4 = CodeMirror 6) but Python syntax + tab-indent semantics preserved. |
| Implementation type dropdown | 5 options: `none` / `code` / `bal` / `state_machine` / `quantum_circuit` (line 127) | Identical 5 options (panel line 89) | **PASS** | Labels match verbatim. |
| `stateMachineId` selector | `DiagramDropdown` with placeholder + items from `availableStateMachines` (line 397-415) | `Select` with `"— Select State Machine —"` placeholder (panel line 567) | **PASS** | Both gated by `implementationType === 'state_machine'`. |
| `quantumCircuitId` selector | Symmetric (line 422-440) | Symmetric (panel line 584) | **PASS** | |
| BAL ↔ Python template auto-fill | `getCodeTemplate(implType, methodName)` (line 137) — switches template on type change | Not implemented; CodeMirror just opens with the existing `code` value | **GAP** | Minor regression in DX — the v3 panel pre-seeded a template body when switching to `bal` or `code`. |
| Cross-field clearing on impl-type change | Clears `code` / `stateMachineId` / `quantumCircuitId` based on type (method-update line 307-329) | Same clearing logic (panel line 539-555) | **PASS** | |
| Reorder up/down buttons (per method row) | Same `ReorderControls` as attributes (uml-classifier-update lines 354-374) | **Not implemented** | **GAP** | Brief flags this explicitly. |
| Delete row button | Trash icon | `IconButton` + `DeleteIcon` | **PASS** | |
| Add method control | `Textfield` with placeholder `"+ method(param: str): str or →"` (line 401) | `MuiTextField` with placeholder `"+ Add method (Enter)"` (panel line 984) | **PASS** | |

---

## 4. Section-level behaviour

| Behaviour | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Methods section hidden for Enumeration stereotype | `{!isEnumeration && (…)}` wraps the whole methods block (line 344) | `{nodeData.stereotype !== "Enumeration" && (…)}` (panel line 970) | **PASS** | Comment in panel explicitly references the v3 line. |
| Attributes label switches to "Literals" for Enumeration | `isEnumeration ? translate('popup.literals') : translate('popup.attributes')` (line 245) | Always renders `Attributes` (panel line 939) | **GAP** | Minor copy regression for Enumeration nodes. |
| Quick-Code button | `QuickCodeButton` next to method add field; calls `createMethodWithCode` to seed a stub (line 424-430, 461-472) | **Not implemented** | **GAP** | Brief flags this explicitly. |
| Auto-focus / keyboard navigation between rows | `fieldToFocus` state machine + `onSubmitKeyUp` chains (lines 287-339) | Not preserved | **GAP** | DX regression; out of brief but worth tracking. |

---

## 5. Summary scorecard

- **PASS**: 23 fields/behaviours
- **PASS (improved over v3)**: 7 (class-name type dropdown, custom-type, OCL editor, returnType selector, parameters list, Interface stereotype, ImplementationType clearing logic)
- **GAP**: 13

### Top gaps ranked by spec severity

1. **Attribute settings (`isId` / `isExternalId` / `isOptional` / `isDerived`) are inline** — the brief explicitly requires them behind a per-row settings expand. Currently they always render as a flat checkbox row beneath every attribute.
2. **Visibility dropdown labels include full words** ("+ public", "- private", …) for both attributes and methods. The brief mandates symbol-only labels (`+ - # ~`).
3. **Reorder up/down buttons missing on every row** — both attribute and method rows lost the `ReorderControls` v3 had.
4. **Quick-Code button missing** from the method add row (no equivalent of v3's `📝 Code`).
5. **Class-level `description`, `uri`, `icon` fields not surfaced** — `NodeStyleEditor` does not bind these node-data fields.
6. **Freeform stereotype text and explicit italic / underline checkboxes absent** — both versions hide these behind the stereotype switch, but the brief calls for explicit controls.
7. **Enumeration heading still says "Attributes"** instead of switching to "Literals".
8. **Method-impl-type code template auto-seed lost** — switching to `code` / `bal` no longer pre-populates a stub body.

---

## 6. Files inspected

- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-update.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-attribute-update.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-method-update.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/components/style-pane/style-pane.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/classDiagram/ClassEditPanel.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/ui/StereotypeButtonGroup.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/ui/StyleEditor/NodeStyleEditor.tsx`

**Verdict**: PARTIAL — v4 is structurally on par for most rows and adds OCL / parameters / returnType improvements, but six v3 affordances (visibility-symbol labels, settings-expand for flags, reorder buttons, Quick-Code, description/URI/icon fields, "Literals" label) need follow-up work to reach parity per the brief.
