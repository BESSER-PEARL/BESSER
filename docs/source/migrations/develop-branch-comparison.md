# Develop branch comparison (cf2049d -> 173d620 + fix-wave-3)

Audit performed against `BESSER-WEB-MODELING-EDITOR@cf2049d` (origin/develop)
versus current submodule HEAD `feature/migration-react-flow@d8eda99`. The
user is hitting parity gaps in browser; this is the consolidated catalogue
to feed back into the SA-FIX wave.

Method: every claim below is anchored to a path under
`/tmp/wme-develop/packages/editor/src/main/packages/...` (develop) and
`packages/library/lib/...` (current). Read sites confirmed, not inferred.

---

## Per-area findings

### A. Palette element types

Develop palettes are produced by `compose-preview.ts` per diagram (e.g.
`uml-class-diagram/class-preview.ts`); current palettes are declared once
in `packages/library/lib/constants.ts::defaultDropElementConfigs`.

| Diagram | Develop palette items | Library palette items | Delta |
| --- | --- | --- | --- |
| ClassDiagram | Class (with attribute), Class (with attribute + method), Enumeration (3 literals), `ClassOCLConstraint` | class, abstract-class, enumeration | MISSING `ClassOCLConstraint` draggable. Abstract-class is library-only (develop comments it out in `class-preview.ts:88`). Interface and Package are deliberately suppressed in develop too -- parity OK. |
| ObjectDiagram | `ObjectName` only (palette), `ObjectAttribute`/`ObjectMethod` are inline children | `objectName` only | OK. |
| StateMachineDiagram | State, Initial, Final, Action, Object, Merge, Fork (V/H), CodeBlock | Same set | OK (SA-3 parity). |
| AgentDiagram | `AgentState`, `AgentIntent`, `AgentRagElement`, `StateInitialNode`, `StateFinalNode`, `StateCodeBlock`, *and* `UMLStateActionNode`/`StateObjectNode` per `agent-state-preview.ts` | `AgentState`, `AgentIntent`, `AgentIntentObjectComponent`, `AgentRagElement`, `StateInitialNode`, `StateFinalNode` | EXTRA in library: `AgentIntentObjectComponent` is exposed as a draggable; develop builds it inline inside Intent. MISSING in library: `StateActionNode`, `StateObjectNode`, `StateCodeBlock` -- these are dropped via the develop preview but absent in `constants.ts:843-893`. |
| UserDiagram | `UMLUserModelName` (one per meta-model class) + a generic `UMLUserModelName` + `UMLUserModelIcon` driven through `composeUserModelPreview` (reads `usermetamodel_buml_short.json`) | `UserModelName` per meta-model class + generic `UserModelName` + `UserModelIcon` | SHAPE OK, *but*: `UMLUserModelAttribute` carries `attributeOperator` (`< / <= / == / >= / >`) and `attributeId` -- library type defines them but the inspector does not expose all comparators consistently and the rendered row uses `=` when `value` is empty. See E + N below. |
| NNDiagram | Section title + separator preview elements wrap drag-sources into `NN Structure / NN Layers / NN TensorOps / NN Configuration / NN Datasets` (see `nn-preview.ts:280-340`) | Flat list, every entry is a leaf in `constants.ts:943-1088` | MISSING palette section headers and dividers. The 18 layer entries dump in one column. |
| Activity / UseCase / Component / Deployment / SyntaxTree / Petri / Reachability / Flowchart / BPMN / Sfc | Develop and library entries match item-for-item. | -- | OK. |

### B. Default edge type when drawing connections

Develop: `packages/editor/src/main/packages/uml-relationship-type.ts:62-79`
defines `DefaultUMLRelationshipType[ClassDiagram] = ClassBidirectional`,
`AgentDiagram = AgentStateTransition`, `NNDiagram = NNNext`,
`UserDiagram = UserModelLink`, `StateMachineDiagram = StateTransition`.

Library: `packages/library/lib/utils/edgeUtils.ts:797-831`
returns `ClassUnidirectional` for ClassDiagram, and falls through to the
same default for AgentDiagram, NNDiagram, UserDiagram, StateMachineDiagram
(none of those are in the switch).

| Diagram | Develop default | Library default | Delta |
| --- | --- | --- | --- |
| ClassDiagram | `ClassBidirectional` | `ClassUnidirectional` | WRONG. |
| StateMachineDiagram | `StateTransition` | falls through -> `ClassUnidirectional` | WRONG. |
| AgentDiagram | `AgentStateTransition` | falls through -> `ClassUnidirectional` | WRONG. |
| NNDiagram | `NNNext` | falls through -> `ClassUnidirectional` | WRONG. |
| UserDiagram | `UserModelLink` | falls through -> `ClassUnidirectional` | WRONG. |
| ObjectDiagram, ActivityDiagram, UseCase, Component, Deployment, Flowchart, SyntaxTree, ReachabilityGraph, BPMN, Sfc, Communication, PetriNet | match | match | OK. |

### C. Default node `data` shape on drop

Develop: `class-preview.ts:30-86` constructs `UMLClass` with
`UMLClassAttribute({ visibility: 'public', attributeType: 'str' })`.
The dropped node carries structured fields, not raw `+ attribute: str`.

Library: `constants.ts:386-422` writes
`attributes: [{ id, name: "+ attribute: Type" }]` -- a *raw* legacy-format
string, no `attributeType`, no `visibility`, no `isOptional`/etc. Class
node's `formatRow` (Class.tsx:27) only re-formats rows that have at least
one structured field, so newly dropped attributes never render markers.

| Element | Develop default | Library default | Delta |
| --- | --- | --- | --- |
| Class attribute | `{name:'attribute', visibility:'public', attributeType:'str'}` -> renders `+ attribute: str` | `{name:'+ attribute: Type'}` (raw) | Type wrong (`Type` vs `str`); structure missing. |
| Class method | `{name:'method', visibility:'public', attributeType:'any'}` | `{name:'+ method()'}` raw | Same shape gap. |
| Enumeration literal | `enumAttribute_1` ... `_3` (i18n `"enumAttribute":"Case"`, suffixed `_1/_2/_3`) -> renders `Case_1`, `Case_2`, `Case_3` | `Case 1`, `Case 2`, `Case 3` (space, no underscore) | Backend often expects identifier-safe names; user reported `case 1` (sic) instead of `Enum_1`. |
| AgentState | `name: 'AgentState'`, body added inline as `AgentStateBody` row | `name: 'AgentState', replyType: 'text'` | OK structurally; verify `bodies[]` initialised to `[]` (currently shipping). |
| AgentIntent | `name: 'Intent Name', intent_description: ''` | `name:'Intent', intent_description:''` | Minor label drift. |
| RAG | `name:'RAG DB Name'` | `name:'RAG'` + `dbSelectionType:'default'` etc | Library is richer (intentional, see PC-7). |

### D. Default attribute type ("Add attribute" button)

Develop and library agree: `attributeType: 'str'`. Confirmed in
`uml-classifier-attribute-update.tsx:160` (legacy parser default) and
`uml-classifier-member.ts:93` (class field default), and in
`packages/library/lib/components/inspectors/classDiagram/ClassEditPanel.tsx:861`
(`addAttribute` -> `attributeType: 'str'`).

User-visible mismatch comes from the *palette default* -- the dropped
attribute carries the literal string `+ attribute: Type`, so the
inspector parses `Type` as the type when the user clicks the row.
Fix is in C (palette `defaultData`), not in the inspector.

### E. Attribute name sanitisation on commit

Develop: `uml-classifier-attribute-update.tsx:217` and `:288` --
`String(newName).replace(/[^a-zA-Z0-9_]/g, '')`. Spaces, dots, dashes
all stripped; underscore preserved.

Library: same regex in `ClassEditPanel.tsx:860`
(`addAttribute`). However, the *inline rename* path (typing into a row's
name field via the inspector after creation) is hidden in the same
`onPatch({ name })` call -- we need to confirm `onPatch` for `name`
also strips. Quick read of `ClassEditPanel.tsx` shows free-text
`MuiTextField onChange={(e) => onPatch({ name: e.target.value })}`
for method rows (line 449) -- *no sanitisation*. Mark as PARTIAL.

### F. Attribute markers on canvas (isId / isExternalId / isDerived / isOptional)

Develop: `uml-classifier-member.ts:111-132` `displayName` returns
`+ /name?: type = default {id, external id}`. Renderer
(`uml-classifier-member-component.tsx:60-64`) uses that string directly,
plus underline when `isId` is true (ER mode only).

Library: `utils/classifierMemberDisplay.ts:115-149` is a faithful port
(UML and ER modes). Class.tsx:27-54 (`formatRow`) re-formats every row
that has at least one structured field.

The user-visible bug: rows dropped from the palette (see C) start with
no structured fields, so `hasStructuredFields` is false and the raw
name passes through. Once the user opens the inspector and toggles a
flag, the row gets `isId=true` etc. and -- now `hasStructuredFields`
becomes true -- the `+ name: str {id}` form takes effect. So markers
*are* wired, they just look broken because the dropped default never
populates the structure. Fix is in C.

### G. Method rendering -- format and inspector layout

Develop method `displayName`: `uml-classifier-member.ts:128` produces
`+ method(param: str): returnType`. Method-update inspector (separate
from attribute-update) uses MUI-free `Textfield`/`Dropdown`/`Button`
controls (styled-components).

Library: `formatDisplayName` already covers methods (signature carries
`(...)` so the parser splits on the colon after the last `)`). MethodRow
inspector (`ClassEditPanel.tsx:376-540`) is built with **MUI**
(`Box`, `Stack`, `Select`, `MenuItem`, `IconButton`, `MuiTextField`,
`Tooltip`). Webapp uses Radix + Tailwind. Framework mismatch -- see M.

### H. ClassDiagram `ClassOCLConstraint` draggable

Develop: `class-preview.ts:195-198` adds an `OCL Constraint` node to
the palette unconditionally. Type: `ClassOCLConstraint` (registered as
a class element type at `uml-class-diagram/index.ts:9`).

Library: `constants.ts:[UMLDiagramType.ClassDiagram]` only contains
`class`, `class` (abstract), `class` (enumeration). `ClassOCLConstraint`
exists as a node implementation
(`packages/library/lib/nodes/classDiagram/ClassOCLConstraint.tsx`)
but is NOT registered in the palette config -- there is no entry whose
`type === 'ClassOCLConstraint'`. So the user cannot drop one.

### I. ColorDescription / Comments

Develop has `Comments` in `packages/common/comments/` but it is **not**
in any per-diagram palette. It is added through
`components/create-pane/create-pane.tsx:121` -- a separate "Create"
pane mechanism, not the sidebar. There is **no** `ColorDescription`
element type anywhere in develop.

Library: `constants.ts:1183-1189` declares `ColorDescriptionConfig` and
`Sidebar.tsx:161-187` renders it for *every* diagram (after a divider)
unconditionally. User wants it hidden -- it has no develop counterpart
to mirror.

| Item | Develop | Library | Delta |
| --- | --- | --- | --- |
| `ColorDescription` (sidebar) | does not exist | always-visible draggable on every diagram | should be removed or guarded behind a setting. |
| `Comments` (sidebar) | not in sidebar (in create-pane) | not present | parity gap if the create-pane is ever ported. |

### J. UserDiagram parity walk

Develop tree (`packages/editor/src/main/packages/user-modeling/`):
- `index.ts` -- `UserModelElementType` (`UserModelName`, `UserModelAttribute`, `UserModelIcon`); `UserModelRelationshipType` (`UserModelLink`).
- `user-model-preview.ts` -- iterates `getAvailableClasses()` and produces one `UMLUserModelName` per class with `classId` / `className` / `icon` populated; pre-fills `UMLUserModelAttribute` rows from the meta-model.
- `usermetamodel_buml_short.json` (and short_corrected, less_short variants) -- shipped meta-model used by `composeUserModelPreview`.
- `uml-user-model-name/uml-user-model-name.ts` -- extends `UMLClassifier`, supports icon view + normal view, has `extractSvgSize` for inline SVG icons.
- `uml-user-model-attribute/uml-user-model-attribute.ts` -- defines `attributeOperator` (`<, <=, ==, >=, >`), normaliser, default `==`, parses operator from legacy name.
- `uml-user-model-attribute/uml-user-model-attribute-update.tsx` -- inspector for operator + value.
- `uml-user-model-icon/uml-user-model-icon.ts` -- icon node.
- `semantic-validation-ocl.md` -- documentation.

Library tree (`packages/library/lib/nodes/userDiagram/`):
- `UserModelName.tsx` -- equivalent to develop's `uml-user-model-name`,
  reuses `ObjectNameSVG`, supports `attributeOperator` via
  `formatUserModelAttribute`, but does **not** branch on `shouldShowIconView`
  vs normal view -- it just renders the object form. Develop's
  `renderIconView` (icon-only chip) is missing.
- `UserModelAttribute.tsx`, `UserModelIcon.tsx` -- present.
- `services/userMetaModel/index.ts` -- wraps `usermetamodel.json` and
  exposes `getUserModelNamePaletteEntries()` (consumed by `constants.ts:903`).
  Develop also has `_short` and `_less_short` JSONs; library has the
  full one only. PARITY OK for the palette.

Concrete gaps:
1. Library has no equivalent of develop's `composeUserModelPreview` icon-view branch. When the global setting `shouldShowIconView` is on, develop renders a row of icon-only chips; library always renders the full attribute box.
2. Library's `UserModelAttribute` row inspector (`UserModelAttributeEditPanel.tsx`) needs verification that all 5 operators are exposed.
3. The bidirectional converter (json_to_buml/buml_to_json) parity is not in scope of this audit but should be sanity-checked: develop emits `attributeId` / `attributeOperator` keys.

The user's "nothing in common" perception is most likely driven by item 1 (icon-view rendering) plus the operator default (`==` in develop, library's `formatUserModelAttribute:42` defaults to `==` correctly) feeling different from a class-style `name: type`.

### K. NN palette sections

Develop builds the sidebar with `NNSectionTitle` and `NNSectionSeparator`
elements interleaved between the layer drag-sources -- see
`nn-preview.ts:280-340` (`structureTitle`, `layersTitle`, `tensorOpsTitle`,
`configurationTitle`, `datasetsTitle`, plus 4 separators and a spacer).

Library: `constants.ts:943-1088` is a flat array. `Sidebar.tsx:125-159`
maps it without any section logic. There is no `NNSectionTitle` /
`NNSectionSeparator` registered in `dropElementConfigs` and no special
rendering branch. RESULT: 18 layer thumbnails stack vertically with no
grouping, which is what the user sees.

### L. AgentState inline body rendering

Develop: `agent-state-preview.ts:57-110` builds a non-empty AgentState
*and* an empty one in the palette by attaching `AgentStateBody` /
`AgentStateFallbackBody` children. The runtime
`agent-state/agent-state.tsx` (not shown above; cross-checked with our
`AgentState.tsx`) renders the rows inline.

Library: `nodes/agentDiagram/AgentState.tsx:13-90` --
`SA-FIX-Agent: AgentState renders body sections inline`. Verified:
`renderRow` formats each `AgentStateBodyRow`, splits into `mainBodies`
and `fallbackBodies`, draws a divider line between them at
`fallbackDividerY`. Confirmed inline rendering is in place per
`d8eda99`. NO GAP.

The palette entry (`constants.ts:843`) does not seed an initial body
into `defaultData` -- develop palette does. New AgentStates land
empty until the user adds a body row in the inspector. Parity-minor.

### M. UI panel framework

Develop update panes: styled-components-based, see e.g.
`packages/editor/src/main/components/update-pane/...` and the
attribute-update inspector at
`uml-classifier-attribute-update.tsx:13-50`. No MUI, no Tailwind -- just
the editor's own `Textfield` / `Dropdown` / `Button` controls.

Webapp: Tailwind + Radix UI primitives.
- `packages/webapp/package.json` lists `@radix-ui/react-collapsible`,
  `react-dialog`, `react-dropdown-menu`, `react-label`, `react-radio-group`,
  `react-select`, `react-separator`, `react-slot`, `react-tooltip`.
- Tailwind dark mode via class.

Library inspectors:
- `packages/library/package.json:73` -- `"@mui/material": "6.4.2"`.
- 27/35 inspector files import `@mui/material`. Examples:
  `ClassEditPanel.tsx`, `MethodRow`, `NNContainerEditPanel`,
  `RagDbFields`, `UserModelAttributeEditPanel`,
  `AgentDiagramInitEdgeEditPanel`, `ClassOCLConstraintEditPanel`,
  `UserModelNameEditPanel`, `NNReferenceEditPanel`,
  `ClassEdgeEditPanel`, `AgentIntentBodyEditPanel`.

Library has UI primitives at
`packages/library/lib/components/ui/`: `DividerLine`, `Typography`,
`HeaderSwitchElement`, `PrimaryButton`, `StereotypeButtonGroup`,
`StyleEditor`, `TextField`. None of them are MUI-based but they don't
yet cover Select, MenuItem, Stack, IconButton, Tooltip, RadioGroup --
which is why the inspectors fell back to MUI during the migration.

For the user's request to migrate to webapp's framework (Radix +
Tailwind), the work is: replace MUI imports across 27 inspector files
with Radix wrappers in `library/lib/components/ui/` (extending the
existing primitives) plus Tailwind classes. Largest delta: `Select` ->
`@radix-ui/react-select`; `Box`/`Stack` -> `<div>` with utility classes;
`MuiTextField` -> the existing `TextField.tsx`; `IconButton` -> a new
button primitive.

---

## Critical gaps to fix (severity-ranked)

1. **(blocking, all class-like attributes)** Palette `defaultData` for ClassAttribute / ClassMethod / Enumeration literal must use structured fields (`visibility`, `attributeType`, no `+ ` prefix in `name`). See section C. Without this fix, sections D, F (markers), and the user's "type is `Type`" report all stay broken regardless of inspector improvements.
2. **(blocking, every diagram with custom edges)** `getDefaultEdgeType` is wrong for ClassDiagram (`ClassUnidirectional` -> should be `ClassBidirectional`) and missing entirely for StateMachineDiagram, AgentDiagram, NNDiagram, UserDiagram (all fall through to `ClassUnidirectional`). Section B.
3. **(high, ClassDiagram)** Add `ClassOCLConstraint` to the ClassDiagram palette. Node implementation already exists; only `constants.ts:[UMLDiagramType.ClassDiagram]` needs an entry. Section H.
4. **(high, NNDiagram UX)** Implement `NNSectionTitle` / `NNSectionSeparator` palette entries plus the rendering branch in `Sidebar.tsx` so the 18 NN layers group into 5 sections. Section K.
5. **(high, ClassDiagram)** Hide / remove the global `ColorDescription` draggable from `Sidebar.tsx:161-187`, or gate it behind a setting. Develop has no equivalent. Section I.
6. **(medium, UserDiagram)** Implement icon-view rendering branch in `UserModelName.tsx` mirroring develop's `composeUserModelPreview` `composeIconView` path. Section J item 1.
7. **(medium, MethodRow)** Sanitise method `name` on commit: `replace(/[^a-zA-Z0-9_()<>:, ]/g, '')` or similar -- currently free-text. Section E. Also confirm attribute-row inline rename uses the same sanitiser.
8. **(medium, AgentDiagram palette)** Add palette entries for `StateActionNode`, `StateObjectNode`, `StateCodeBlock` (consumed by develop's `agent-state-preview.ts`). Remove or de-prioritise `AgentIntentObjectComponent` palette entry -- develop builds it inline. Section A.
9. **(low, Enumeration default literals)** Rename palette defaults from `Case 1 / Case 2 / Case 3` to `Enum_1 / Enum_2 / Enum_3` (or whatever name pattern the user wants). Section C.
10. **(refactor, large)** Migrate inspector panels from MUI to Radix + Tailwind. 27 files, 11 distinct components. Extend `library/lib/components/ui/` with `Select`, `IconButton`, `Tooltip`, `RadioGroup`, `Collapsible`, `Stack` (utility wrapper). Section M. This is the biggest single piece of work and unblocks visual parity with the webapp shell.

---

## Recommended fix-agent assignments

| Gap | Owner | Notes |
| --- | --- | --- |
| 1, 9 (palette `defaultData` + enum names) | **SA-FIX-CLASS-FUND** | Touches `constants.ts:386-422`. Light-weight; do this first because it unblocks 4 user-reported bugs at once. |
| 2 (default edge per diagram) | **SA-FIX-CLASS-FUND** (ClassDiagram) + **SA-FIX-Agent** (Agent) + **SA-FIX-NN-DROPS** (NN) + **SA-FIX-User** (User) + **SA-FIX-State** (StateMachine) | Single switch-statement edit each in `edgeUtils.ts:797`. |
| 3 (`ClassOCLConstraint` palette) | **SA-FIX-CLASS-FUND** | Add `{type:'ClassOCLConstraint', defaultData:{constraint:'OCL Constraint'}, svg: ClassOCLConstraintSVG}`. Node + popover already exist. |
| 4 (NN palette sections) | **SA-FIX-NN-DROPS** | Either (a) introduce `kind:'sectionTitle'/'separator'` markers in `DropElementConfig` and branch in `Sidebar.tsx`, or (b) port `NNSectionTitle`/`Separator` element types verbatim. (a) is lighter. |
| 5 (kill global `ColorDescription`) | **SA-FIX-CLASS-FUND** | Delete `ColorDescriptionConfig` block in `Sidebar.tsx:161-187`. If the analytics opt-in feature wants it back, make it per-diagram in `dropElementConfigs`. |
| 6 (UserDiagram icon view) | **SA-FIX-User** | Port `composeIconView` from `user-model-preview.ts:20-81` into `UserModelName.tsx`, gated on `useShowIconView()` (settings store). |
| 7 (Method name sanitisation) | **SA-FIX-CLASS-FUND** | Tweak `MethodRow.onPatch` and the attribute-row name onChange in `ClassEditPanel.tsx`. |
| 8 (Agent palette items) | **SA-FIX-Agent** | Add 3 palette entries; remove `AgentIntentObjectComponent`. |
| 10 (MUI -> Radix migration) | **SA-FIX-UI-FRAMEWORK** (new) | Multi-PR. First PR: extend `library/lib/components/ui/` with the 6 missing primitives. Subsequent PRs: 3-5 inspector files at a time. Cannot land before 1-9 because the inspectors are still in flux. |

---

## File pointers (current branch)

- Palette config: `packages/library/lib/constants.ts:376-1189`
- Default edge type: `packages/library/lib/utils/edgeUtils.ts:797-831`
- Class node renderer: `packages/library/lib/nodes/classDiagram/Class.tsx`
- Display formatter: `packages/library/lib/utils/classifierMemberDisplay.ts:115-149`
- Class inspector (MUI): `packages/library/lib/components/inspectors/classDiagram/ClassEditPanel.tsx`
- Sidebar: `packages/library/lib/components/Sidebar.tsx`
- Agent state node (inline bodies confirmed): `packages/library/lib/nodes/agentDiagram/AgentState.tsx`
- User meta-model service: `packages/library/lib/services/userMetaModel/index.ts`

## Develop file pointers (for cross-reference)

- Class preview: `/tmp/wme-develop/packages/editor/src/main/packages/uml-class-diagram/class-preview.ts`
- NN preview (sections): `/tmp/wme-develop/packages/editor/src/main/packages/nn-diagram/nn-preview.ts:280-340`
- User preview: `/tmp/wme-develop/packages/editor/src/main/packages/user-modeling/user-model-preview.ts`
- Default edge type table: `/tmp/wme-develop/packages/editor/src/main/packages/uml-relationship-type.ts:62-79`
- Member display logic: `/tmp/wme-develop/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-member.ts:111-152`
