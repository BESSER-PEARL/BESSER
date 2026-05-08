# SA-PARITY-FINAL-5 — UserDiagram

Final 1:1 parity check on the SA-4 UserDiagram cutover from
`packages/editor/src/main/packages/user-modeling/` (v3) to
`packages/library/lib/{nodes,components/inspectors,edges}/userDiagram/` +
`packages/library/lib/services/userMetaModel/` (v4).

Submodule HEAD audited: `771c064`.
Parent branch: `claude/refine-local-plan-sS9Zv`.
Read-only audit; no source changes.

---

## Verdict

`PASS WITH MINOR GAPS`. Element types, edge type, OCL meta-model JSON,
v3 → v4 migrator (including the SA-2.2 #38 comparator-from-name
synthesis), v4 → v3 inverse migrator, and the round-trip test suite
match the spec. The remaining gaps live in the inspector layer and one
visual-shape default; none of them block round-trip data fidelity.

---

## 1. Element types

| v3 | v4 | Status |
|---|---|---|
| `UserModelName` (`uml-user-model-name/uml-user-model-name.ts`) | `UserModelName` (`packages/library/lib/nodes/userDiagram/UserModelName.tsx`) | `=` |
| `UserModelAttribute` (`uml-user-model-attribute/uml-user-model-attribute.ts`) | `UserModelAttribute` (`packages/library/lib/nodes/userDiagram/UserModelAttribute.tsx`, registered for legacy round-trip) | `=` |
| `UserModelIcon` (`uml-user-model-icon/uml-user-model-icon.ts`) | `UserModelIcon` (`packages/library/lib/nodes/userDiagram/UserModelIcon.tsx`) | `=` |

All three node types are present and registered via
`packages/library/lib/nodes/userDiagram/index.ts:16-20`.

**Element parity: OK.**

## 2. Edge types

`UserModelLink` is exported and registered:

- `packages/library/lib/edges/edgeTypes/UserModelLink.tsx:15` aliases
  `ObjectDiagramEdge` per the SA-4 brief.
- `packages/library/lib/edges/edgeTypes/UserModelLink.tsx:18`
  side-effect-registers `UserModelLink` into `_edgeTypeRegistry`.
- `packages/library/lib/edges/types.tsx:212` lists it in
  `EDGE_TYPE_FEATURES` with `allowMidpointDragging: true`, matching
  ObjectDiagramEdge.

**Edge parity: OK.**

## 3. Per-element data fields

### UserModelName (`UserModelNameNodeProps`,
`packages/library/lib/types/nodes/NodeProps.ts:471-480`)

| v3 (`uml-user-model-name.ts:15-19`) | v4 | Status |
|---|---|---|
| `name` | `data.name` | `=` |
| `classId?` | `data.classId?` | `=` (open-question #1 resolution) |
| `className?` | `data.className?` | `=` |
| `icon?` (separate child element) | `data.icon?` (collapsed) | `=` (collapsed) |
| `attributes` (id list) | `data.attributes: UserModelAttributeRow[]` | `=` (collapsed inline rows) |
| `underline = true` (hardcoded in v3 class) | rendered via `ObjectNameSVG` with `isUnderlined={true}` (`ObjectNameSVG.tsx:83`) | `=` (mirrors SA-2.1 ObjectName fix) |
| `methods` (inherited from `UMLClassifier`, unused) | not present | DROP (intentional — UserModel is constraint-style) |
| (none) | `data.description?` | EXTRA (used by OCL semantic validation) |

### UserModelAttribute (`UserModelAttributeRow`,
`packages/library/lib/types/nodes/NodeProps.ts:457-463`)

| v3 (`uml-user-model-attribute.ts`) | v4 | Status |
|---|---|---|
| `name` (`"foo == 18"` formatted) | `name` | `=` |
| `attributeId?` | `attributeId?` | `=` |
| `attributeOperator: '<' \| '<=' \| '==' \| '>=' \| '>'` (default `'=='`) | `attributeOperator?` (5 values) | `=` |
| Inherited `attributeType` (default `'str'`) | `attributeType?` | `=` |
| Inherited `defaultValue` | `defaultValue?` | `=` |
| (none) | `value?: unknown` | EXTRA — separates value from name so `"age >= 18"` rebuilds cleanly |
| `fillColor` / `textColor` (inherited) | inherited from `ClassNodeElement` | `=` |

`UserModelAttributeNodeProps` (standalone node,
`packages/library/lib/types/nodes/NodeProps.ts:488-492`) is the
container variant when an attribute is unowned; it carries
`attributeType?`, `defaultValue?`, `attributeOperator?`. It does not
carry `attributeId` or `value`, but those slots only matter for owned
rows so this is fine.

### UserModelIcon (`UserModelIconNodeProps`,
`packages/library/lib/types/nodes/NodeProps.ts:499-501`)

`{ icon?: string }` — matches v3 `IUMLUserModelIcon`. v3 marked the
element as non-interactive (`features = { selectable: false, … }`,
`uml-user-model-icon.ts:17-25`); the v4 standalone node is regular
interactive. Minor UX deviation; not relevant to data fidelity.

**Field parity: OK with documented EXTRAs.**

## 4. Inspector form parity

### `UserModelNameEditPanel`
(`packages/library/lib/components/inspectors/userDiagram/UserModelNameEditPanel.tsx`)

What is implemented:

- `name`, `className`, `description` text fields.
- `classId` lookup helpers (`lookupLinkedAttribute`,
  `lookupEnumerationLiterals`) for child-row widget dispatch.
- Per-row attribute editor with type-aware widgets:
  - Enumeration → MUI `Select` of literals (SA-2.2 #36).
  - Bool → MUI `Select` of `true / false`.
  - Date / Datetime / Time → typed `TextField`.
  - String → quoted-style `TextField` placeholder.
  - Other primitives → plain `TextField`.
- **Integer-gated comparator dropdown (SA-2.2 #37)**: the
  `effectiveType` check at `UserModelNameEditPanel.tsx:130-200`
  (`INTEGER_TYPES = {int, integer, number}`) gates rendering of the
  comparator `Select` to integer-typed rows only. Mirrors v3's
  `isIntegerType()` at `uml-user-model-attribute-update.tsx:106-109`.
  **VERIFIED.**
- Add / delete row affordances.
- `NodeStyleEditor` for fillColor/strokeColor/textColor (parent only).

What is **missing**:

- **`classId` picker dropdown bound to `diagramBridge.getAvailableClasses()`**.
  `data.classId` is preserved on the model but the panel only renders a
  free-text `className` `TextField`
  (`UserModelNameEditPanel.tsx:354-361`); there is no `Select` of
  available class nodes. v3 didn't ship a dedicated UserModelName form
  either (the user-modelling fork inherited the classifier form from
  `UMLClassifierUpdate`), so this is **not a v3 regression** — but it
  is the brief's expectation for SA-PARITY-FINAL-5. **MEDIUM**.
- Per-row `ColorButton` / `StylePane`. v3
  (`uml-user-model-attribute-update.tsx:233-240`) attached a `ColorButton`
  to each attribute row; v4 only edits color on the parent. **MINOR**.

### `UserModelAttributeEditPanel` (standalone,
`packages/library/lib/components/inspectors/userDiagram/UserModelAttributeEditPanel.tsx`)

What is implemented: name `TextField`, type `Select`, operator
`Select`, defaultValue `TextField`, plus `NodeStyleEditor` for color.

What is **off-spec**:

- **Comparator dropdown is unconditionally rendered** at
  `UserModelAttributeEditPanel.tsx:102-119` (no `isIntegerType` guard).
  The brief calls for "comparator (only when integer-typed), color".
  v3 (`uml-user-model-attribute-update.tsx:187-213`) only rendered the
  comparator when the linked class attribute was integer-typed and a
  `baseAttributeName` was resolvable. The standalone node has no
  diagramBridge link to query, but the brief still asks for the gate
  by `attributeType`. **MINOR** — the standalone path is rare (legacy
  fixtures only).

### `UserModelIconEditPanel`

Not registered.
`packages/library/lib/components/inspectors/userDiagram/index.ts:11-12`
only registers `UserModelName` and `UserModelAttribute`. Per the doc
string at `index.ts:6` ("the icon node has no editor — it surfaces via
the parent's panel"), this is intentional and matches v3 (which had
no icon update form either). The brief's "UserModelIconEditPanel:
glyph / image picker" is therefore an enhancement target rather than
an existing v3 surface — **call-out, not a regression**.

**Inspector parity: PASS with two MINOR gaps + one MEDIUM (classId
picker).**

## 5. Migrator (v3 → v4)

`migrateUserDiagramV3ToV4` is defined at
`packages/library/lib/utils/versionConverter.ts:2061-2071`. It
delegates to `convertV3ToV4` with a `UserDiagram` type guard.

The migrator collapses `UserModelAttribute` and `UserModelIcon` child
elements onto their owner `UserModelName` per the v4 wire-shape spec
(`versionConverter.ts:1845-1853`). The `UserModelName` case
(`versionConverter.ts:1118-1196`) in `extractNodeData`:

- Walks `allElements` for `owner === element.id && type === 'UserModelAttribute'` and pushes a row with all fields.
- **Synthesizes `attributeOperator` from the row name** when no
  explicit field is present (`versionConverter.ts:1145-1166` calls
  `extractAttributeOperatorFromName` defined at
  `versionConverter.ts:499-508`). Mirrors v3's
  `extractComparatorFromName` (`uml-user-model-attribute.ts:27-33`).
  Explicit `c.attributeOperator` wins over name-extraction (matching
  v3's `deserialize` precedence at `uml-user-model-attribute.ts:73-77`).
- Collapses `classId` / `className` / `description` / `icon`.

The standalone `UserModelAttribute` case at
`versionConverter.ts:1198-1222` also synthesizes
`attributeOperator` from name when missing.

**SA-2.2 #38 regression test** (`tests/round-trip/userDiagram.test.ts:135-214`):

- Embedded `>=` extracted (line 205).
- Embedded `<` extracted (line 207).
- No comparator in name → defaults to `==` (line 209).
- Explicit `attributeOperator` field beats name-extraction (line 211).
- No comparator anywhere → undefined (line 213).

**VERIFIED.**

## 6. OCL meta-model

```
md5: ca7afa7061fc1511c607f9180875974f
- packages/editor/src/main/packages/user-modeling/usermetamodel_buml_short.json
- packages/library/lib/services/userMetaModel/usermetamodel.json
```

**Byte-identical. OK.** (The two siblings of the v3 source —
`usermetamodel_buml_less_short.json` and
`usermetamodel_buml_short_corrected_format.json` — are not vendored
into the new lib; the canonical short version is the only one the
brief specifies.)

## 7. Visual shape

### UserModelName

- v3 `uml-user-model-name.ts:23` hardcoded `underline = true` on the
  classifier name.
- v4 `UserModelName.tsx:165-172` reuses `ObjectNameSVG`, which renders
  the header through `<HeaderSection isUnderlined={true} … />`
  (`ObjectNameSVG.tsx:76-86`).
- The underline is **rendered** in v4 — mirroring SA-2.1's ObjectName
  fix. **OK.**

v3's icon-view branch (`uml-user-model-name.ts:182-210`,
`settingsService.shouldShowIconView()`) is not yet replicated — the
inline icon body is passed to `ObjectNameSVG.renderData` but the SVG
does not switch to icon-view layout. Same regression as ObjectName,
documented in Wave-2 audit. **NOT in scope of SA-PARITY-FINAL-5**.

### UserModelIcon

v4 `UserModelIcon.tsx:36-79` renders the inline SVG body (via
`foreignObject`) or a data URL (via `<image href>`) inside the node
bounds. v3 only provided geometry hooks for the parent; the v4
behaviour is an **enhancement**, not a regression.

**Visual parity: OK** for the SA-2.1-mirroring underline; SA-4
icon-view parity remains a Wave-2 follow-up.

## 8. Round-trip tests

`tests/round-trip/userDiagram.test.ts` ships with **4 `it` cases**:

1. `migrates the v3 fixture to v4 with structural fidelity` (line 24).
2. `round-trips v4 → v3 → v4 with structural equality` (line 62).
3. `preserves an attribute rename through a v4 → v3 → v4 cycle`
   (line 108).
4. `synthesizes attributeOperator from embedded comparator in name`
   (line 135) — **SA-2.2 #38 regression**.

Test (4) covers all five comparator-extraction cases listed in §5.

**Round-trip parity: OK.**

---

## Critical gaps (would block sign-off)

- None. Element types, edge type, OCL meta-model JSON, migrator, inverse
  migrator, and the round-trip suite all match the brief.

## Minor gaps (work follow-ups)

1. **classId picker dropdown** missing from `UserModelNameEditPanel` —
   `data.classId` is preserved but the panel only edits free-text
   `className`. The bridge already exposes
   `getAvailableClasses()`; rendering a `Select` of those is the
   straight path. **MEDIUM.**
2. **Per-row color picker** missing from
   `UserModelNameEditPanel.AttrRow` — v3 had a `ColorButton` per row.
   **MINOR.**
3. **Standalone `UserModelAttributeEditPanel` always shows comparator**
   — should be gated by `INTEGER_TYPES.has(attributeType)` to mirror
   v3 and the SA-2.2 #37 fix used in the parent's panel. **MINOR.**
4. **`UserModelIconEditPanel`** — not registered (intentional per
   inspector index doc string; v3 had no icon form either). The brief
   asks for "glyph / image picker"; this is an enhancement target,
   not a regression. **CALL-OUT.**

## Out of scope (Wave-2 follow-ups, not SA-PARITY-FINAL-5)

- Icon-view rendering on `UserModelName` when
  `settingsService.shouldShowIconView()` is on (same regression as
  ObjectName).
- `UserModelIcon.features.selectable: false` non-interactive flag —
  v4 node is regular interactive.
