# PC-9: UserDiagram

Read-only audit of the UserDiagram cutover from v3
(`packages/editor/src/main/packages/user-modeling/uml-user-model-name/`,
`uml-user-model-attribute/`, `uml-user-model-icon/` + their `*-update.tsx` companions
+ `usermetamodel_buml_short.json`) to v4
(`packages/library/lib/nodes/userDiagram/`,
`packages/library/lib/components/inspectors/userDiagram/`,
`packages/library/lib/services/userMetaModel/usermetamodel.json`,
`packages/library/lib/edges/edgeTypes/UserModelLink.tsx`,
`packages/library/lib/utils/versionConverter.ts` UserDiagram §,
`packages/webapp/src/main/shared/types/project.ts::buildUserDiagramSeedNodes`,
`packages/webapp/src/main/app/store/workspaceSlice.ts::setupBridgeForActiveDiagram`).

Submodule HEAD audited: `34b390c`.
Parent branch: `claude/refine-local-plan-sS9Zv`.
Read-only audit; no source changes.

## Sources

### Old (`packages/editor/src/main/packages/user-modeling/`)

- `uml-user-model-name/uml-user-model-name.ts` (model — extends `UMLClassifier`,
  hardcoded `underline = true`, fields `classId?` / `className?` / `icon?`,
  `serialize`/`deserialize` round-trip; `reorderChildren` keeps
  `UserModelAttribute` rows + (unused) methods; `render` branches on
  `settingsService.shouldShowIconView()`; private `extractSvgSize` parses
  inline SVG `width`/`height`/`viewBox` for icon sizing; static
  `supportedRelationships = [UserModelLink, Link]`)
- `uml-user-model-attribute/uml-user-model-attribute.ts` (model — extends
  `UMLClassifierAttribute`. `USER_MODEL_ATTRIBUTE_COMPARATORS = ['<', '<=',
  '==', '>=', '>']`, `DEFAULT_COMPARATOR = '=='`,
  `normalizeUserModelAttributeComparator(raw)` (single `=` → `==`; unknown →
  `==`), `extractComparatorFromName(name)` (regex
  `^(?:.*?)(<=|>=|==|=|<|>)`), fields `attributeId?` / `attributeOperator`.
  `deserialize` precedence: explicit `attributeOperator` field wins,
  otherwise extracted from `name`)
- `uml-user-model-attribute/uml-user-model-attribute-update.tsx` (per-row popup
  — name/value parsing via `parseAttributeValue` regex; `Textfield` for value;
  `Dropdown` of 5 comparators **only when** `baseAttributeName` is resolvable
  AND `isIntegerType()` is true (`int` / `integer` / `number`,
  `:106-109`); enumeration `Dropdown` of literals when the linked class
  attribute's type is an Enumeration class; per-row `ColorButton` +
  `StylePane fillColor textColor`)
- `uml-user-model-icon/uml-user-model-icon.ts` (model — extends `UMLElement`,
  `features = { hoverable: false, selectable: false, movable: false,
  connectable: false, droppable: false, updatable: false }`, single `icon?`
  field, `render` clamps `bounds.width` to multiples of 10)
- `user-model-preview.ts` (`composeUserModelPreview` — sidebar drag-source
  generator; for `shouldShowInstancedObjects() && hasClassDiagramData()`
  spawns one `UMLUserModelName` instance per
  `diagramBridge.getAvailableClasses()` class with attributes pre-populated
  from the linked class. **This is the v3 mechanism that surfaced the 4
  user-meta-model classes (Personal_Information / Skill / Education /
  Disability)** — `setupBridgeForActiveDiagram` set the bridge data to
  `usermetamodel_buml_short.json` whenever the UserDiagram tab activated)
- `index.ts` (`UserModelElementType.{UserModelName, UserModelAttribute,
  UserModelIcon}`, `UserModelRelationshipType.UserModelLink`)
- `popups.ts:73-75` (`UserModelName → UMLObjectNameUpdate`,
  `UserModelAttribute → null`, `UserModelIcon → null`)
- `uml-object-diagram/uml-object-name/uml-object-name-update.tsx:209-221`
  (the `isUserModelElement` branch — **classId dropdown is HIDDEN for
  UserModelName**: `showClassSelection = !isUserModelElement && availableClasses.length > 0`)
- `usermetamodel_buml_short.json` (canonical reference user-meta-model loaded
  into `diagramBridge` for UserDiagram; defines Personal_Information / Skill /
  Education / Disability + supporting enums and links)

### New (`packages/library/lib/`)

- `nodes/userDiagram/UserModelName.tsx` (React-Flow node — uses
  `ObjectNameSVG` for visual, computes `minWidth`/`minHeight` from text +
  attribute count, formats each attribute via `formatUserModelAttribute`
  (`name op value` when value present and op not already embedded), reuses
  `PopoverManager`, `NodeResizer`, `DefaultNodeWrapper`)
- `nodes/userDiagram/UserModelAttribute.tsx` (legacy stand-alone row — single
  rect + text label, only used when the migrator finds an unowned
  UserModelAttribute; primary editing surface is the parent's panel)
- `nodes/userDiagram/UserModelIcon.tsx` (renders inline SVG body via
  `foreignObject` or `<image>` for `data:` / http URLs; placeholder text
  "icon" when no `icon`)
- `nodes/userDiagram/index.ts` (side-effect registration of all 3 node types)
- `components/inspectors/userDiagram/UserModelNameEditPanel.tsx` (parent
  inspector — `name` / `className` / `description` text fields; per-row
  `AttrRow` with name TextField, type Select (8 primitives), comparator
  Select **gated by `INTEGER_TYPES` set on the resolved type**, value widget
  picked by type (Enumeration → MUI Select of literals via
  `lookupEnumerationLiterals`, bool → Select, date/datetime/time → typed
  TextField, str → quoted-style TextField, else plain TextField); add/delete
  affordances; `NodeStyleEditor` for parent colors only)
- `components/inspectors/userDiagram/UserModelAttributeEditPanel.tsx`
  (stand-alone attribute inspector — name / type / **unconditional**
  comparator Select / defaultValue / `NodeStyleEditor`)
- `components/inspectors/userDiagram/index.ts` (registers
  `UserModelNameEditPanel` and `UserModelAttributeEditPanel`; **no
  `UserModelIconEditPanel`** — explicitly noted as intentional in the doc
  string, matching v3 which had no icon-update form)
- `edges/edgeTypes/UserModelLink.tsx`
  (`export const UserModelLink = ObjectDiagramEdge`, side-effect-registered
  via `registerEdgeTypes`; `EDGE_TYPE_FEATURES` carries
  `allowMidpointDragging: true`)
- `services/userMetaModel/usermetamodel.json` (vendored OCL meta-model JSON)
- `index.tsx:97-101`
  (`import userMetaModelJson from "./services/userMetaModel/usermetamodel.json"`;
  `export const userMetaModel = userMetaModelJson` — the webapp imports this
  via `@besser/wme`)
- `utils/versionConverter.ts:1144-1256` (UserDiagram §) +
  `:1874-1881` (collapse filter) + `:2087-2097` (`migrateUserDiagramV3ToV4`
  type-guard wrapper) + `:499-508` (`extractAttributeOperatorFromName`) +
  `:2931-3060` (`convertV4ToV3User` inverse migrator)
- `types/nodes/NodeProps.ts:473-508` (`UserModelAttributeRow`,
  `UserModelNameNodeProps`, `UserModelAttributeNodeProps`,
  `UserModelIconNodeProps`)
- `components/svgs/nodes/userDiagram/UserDiagramSVGs.tsx`
  (sidebar palette previews — **single static "Alice: User" preview**, NO
  per-class palette generation; this is the closest analogue to v3's
  `composeUserModelPreview` but does NOT replicate its bridge-driven loop)
- `webapp/src/main/shared/types/project.ts:259-327`
  (`buildUserDiagramSeedNodes` — SA-UX-FIX-2 (B3) replacement seed for v3's
  `composeUserModelPreview` palette: hard-codes 4 cards
  `Personal_Information / Skill / Education / Disability`)
- `webapp/src/main/shared/types/project.ts:391-394`
  (wires the seed only when `type === UMLDiagramType.UserDiagram` inside
  `createEmptyDiagram`)
- `webapp/src/main/shared/types/project.ts:518-526`
  (project-version migration: a project that lacks a `UserDiagram` array
  gets one added via `createEmptyDiagram` — which seeds the 4 cards)
- `webapp/src/main/app/store/workspaceSlice.ts:140-147` (when entering a
  UserDiagram tab, the bridge is populated with `userMetaModel` from
  `@besser/wme`, so inspectors that walk the bridge see the user-meta-model
  classes)

## Verdict

**PASS WITH USER-VISIBLE CONTENT-LOSS BUG** for any pre-seed UserDiagram
already persisted before SA-UX-FIX-2 (B3) landed. The migrator, OCL JSON,
edge alias, and round-trip are byte-/structurally clean (md5 of OCL JSON
matches v3 verbatim; `migrateUserDiagramV3ToV4` preserves all
`UserModelName` parents and synthesizes `attributeOperator` from embedded
comparators per SA-2.2 #38). The new lib's UserDiagram is functionally
present. **The user's "only ONE class" complaint is the seed-vs-existing-diagram
mismatch documented in §"Root cause"** below, not a migrator data-loss
bug.

## Coverage matrix

| Feature | Old | New | Notes |
| --- | --- | --- | --- |
| Element types | `UserModelName` / `UserModelAttribute` / `UserModelIcon` (`packages/user-modeling/index.ts:1-5`) | identical (`nodes/userDiagram/index.ts:16-20`) | Parity. |
| Edge type | `UserModelLink` (`index.ts:7-9`) | `UserModelLink = ObjectDiagramEdge` aliased + registered (`edges/edgeTypes/UserModelLink.tsx:15-18`) | Parity. |
| `UserModelName.name` underline | hardcoded `underline = true` (`uml-user-model-name.ts:23`) | `ObjectNameSVG isUnderlined={true}` (`ObjectNameSVG.tsx:83`) | Parity (mirrors SA-2.1). |
| `UserModelName.classId` / `className` | preserved (`uml-user-model-name.ts:15-19,33-43,46-58,60-71`) | preserved (`UserModelName.tsx:143-145`, `NodeProps.ts:491-493`) | Parity. |
| `UserModelName.icon` (child or inline) | child `UserModelIcon` element (`uml-user-model-name.ts:51-56`) | collapsed `data.icon?` on owner (`versionConverter.ts:1212-1219`); also surfaces as standalone if unowned | Parity (collapsed shape per spec). |
| `UserModelName.description` | absent | `data.description?` (used by OCL semantic validation) | EXTRA — additive. |
| `UserModelName.methods` (inherited but unused) | inherited from `UMLClassifier` | dropped from v4 props | Intentional drop — user model is constraint-style. |
| `UserModelAttribute` row id / attributeId / name / type / operator / value / colors | all 7+ via class fields and inherited `UMLClassifierAttribute` | `UserModelAttributeRow` extends `ClassNodeElement` (`NodeProps.ts:473-479`) — `id`/`name`/`attributeType`/`attributeOperator`/`attributeId`/`value`/`defaultValue`/`fillColor`/`textColor` | Parity. |
| 5-option comparator (`<` / `<=` / `==` / `>=` / `>`) | `USER_MODEL_ATTRIBUTE_COMPARATORS` (`uml-user-model-attribute.ts:8`) | identical literal union (`NodeProps.ts:475`) and `COMPARATORS = ["<","<=","==",">=",">"]` in inspectors | Parity. |
| Comparator default `==` | `DEFAULT_COMPARATOR = '=='` (`uml-user-model-attribute.ts:11`) | `?? "=="` defaults at render and in inspector value props | Parity. |
| Comparator gated by integer type (SA-2.2 #37) | `isIntegerType() = ['int','integer','number'].includes(t)` (`uml-user-model-attribute-update.tsx:106-109`) | `INTEGER_TYPES = new Set(["int","integer","number"])` and `isInteger && (<Select …>)` (`UserModelNameEditPanel.tsx:57,137,200-218`) | Parity in parent panel. |
| Comparator gated by integer type — standalone attribute inspector | n/a in v3 (no separate panel) | **Comparator Select unconditionally rendered** (`UserModelAttributeEditPanel.tsx:102-119`) | **MINOR off-spec** — should reuse `INTEGER_TYPES` gate; rare path. |
| `attributeOperator` from name extraction | `extractComparatorFromName` regex (`uml-user-model-attribute.ts:27-33`); `deserialize` falls back when no explicit field (`:73-77`) | `extractAttributeOperatorFromName` (`versionConverter.ts:499-508`) called from migrator (`:1178-1180, 1232-1235`) | Parity (SA-2.2 #38 verified). |
| `attributeOperator` precedence: explicit field wins | `deserialize :73-77` | migrator `c.attributeOperator ?? extractAttributeOperatorFromName(c.name)` (`:1179-1180`) | Parity. |
| Single `=` normalised to `==` | `normalizeUserModelAttributeComparator` (`:13-25`) | inside `extractAttributeOperatorFromName` — single `=` matched then mapped to `==` | Parity. |
| classId picker dropdown in parent inspector | **hidden** for UserModelName (`uml-object-name-update.tsx:209-221`) | absent — only free-text `className` field (`UserModelNameEditPanel.tsx:354-361`) | **Parity with v3** (NOT a regression). The brief calls for adding a dropdown — that's an enhancement target, not a v3 surface. |
| Per-row `ColorButton` + `StylePane` | yes (`uml-user-model-attribute-update.tsx:233-240`) | absent; only parent-level `NodeStyleEditor` | **MINOR gap** — v3 had per-row colors. |
| Enumeration value dropdown | `getEnumerationValues()` (`uml-user-model-attribute-update.tsx:111-134`) | `lookupEnumerationLiterals` walks `diagramBridge.getClassDiagramData().nodes` (`UserModelNameEditPanel.tsx:98-125`) | Parity (different bridge shape — see §"Bridge shape mismatch" below). |
| Bool / date / datetime / time / string widgets | partial — string treated as plain `Textfield` | typed widgets per type (`UserModelNameEditPanel.tsx:240-278`) | EXTRA — improvement. |
| Inline icon rendering inside `UserModelName` | only `renderObject` legacy path; standalone child element shows up | `UserModelIcon` standalone renders SVG body via `foreignObject` (`UserModelIcon.tsx:50-79`); no in-card icon body when child is collapsed onto parent | **MINOR visual gap** — when icon is collapsed onto parent's `data.icon`, the parent renderer (`ObjectNameSVG`) does not display the icon body. Same regression as ObjectName per Wave-2. |
| `UserModelIcon.features.selectable: false` | non-interactive (`uml-user-model-icon.ts:17-25`) | regular interactive node | **MINOR UX deviation** — not relevant to data fidelity. |
| Sidebar palette: per-class drag sources | `composeUserModelPreview` spawns one `UMLUserModelName` per `getAvailableClasses()` class with attribute rows (`user-model-preview.ts:99-141`) | `UserDiagramSVGs.tsx` static "Alice: User" preview only | **MAJOR feature gap** — see §"Root cause" below. |
| OCL meta-model JSON | `usermetamodel_buml_short.json` (md5 `ca7afa70…`) | `services/userMetaModel/usermetamodel.json` (md5 `ca7afa70…`) | **Byte-identical** — confirmed. |
| `userMetaModel` exported from lib | n/a (loaded by webapp from editor pkg) | `import userMetaModelJson … export const userMetaModel = userMetaModelJson` (`index.tsx:97-101`) | Parity. |
| Migrator: `UserModelAttribute` collapse onto owner | n/a (v3 stored as separate elements) | `:1874-1881` filter + `:1144-1199` collapse loop | Parity (per spec). |
| Migrator: `UserModelIcon` collapse onto owner | n/a | `:1874-1881` filter + `:1212-1219` resolve | Parity. |
| Migrator: standalone (unowned) attribute survives | n/a | `:1874-1881` (`element.owner` guard) + case `:1224-1248` | Parity. |
| Inverse migrator (`convertV4ToV3User`) | n/a | `:2964-3046` re-expands rows + icon child | Parity (round-trip test passes). |
| Round-trip test coverage | n/a | `tests/round-trip/userDiagram.test.ts` (4 `it` cases, including SA-2.2 #38 5-case regression) | OK. |

## Root cause: "the new UserDiagram has only ONE class with nothing in common with v3's user model"

There are **three** orthogonal mechanisms that together produced the user's
experience. Unpacking them:

### 1. v3 had no UserDiagram default seed either

v3's `createEmptyDiagram(UserDiagram)` produced an empty canvas. The "4
user-meta-model classes" the user remembers from v3 were never seeded into
the diagram itself — they came from the **sidebar palette**, generated
on-the-fly by `composeUserModelPreview` (`user-model-preview.ts:83-141`).
That palette generator iterated
`diagramBridge.getAvailableClasses()` (which the webapp's
`setupBridgeForActiveDiagram` populated with `usermetamodel_buml_short.json`
each time the UserDiagram tab opened). So when the user dragged a
`UserModelName` from the sidebar, they got pre-populated drag sources for
Personal_Information / Skill / Education / Disability / etc. — backed by
the user-meta-model.

**The v3 baseline UX was: open UserDiagram → empty canvas + per-class
sidebar previews → drag classes onto canvas → populate.**

### 2. v4 lib does NOT replicate `composeUserModelPreview`

`packages/library/lib/components/svgs/nodes/userDiagram/UserDiagramSVGs.tsx`
ships only a **single hard-coded "Alice: User" sidebar preview**
(`UserDiagramSVGs.tsx:10-48`). There is no v4 equivalent that consumes
`diagramBridge.getAvailableClasses()` to spawn per-class palette items.
This is the structural feature-gap that PC-9 reveals: the v4 sidebar
cannot offer "drag a Personal_Information" / "drag a Skill" / etc.; the
user only sees one generic placeholder. The `userMetaModel` JSON IS
loaded into the bridge by `setupBridgeForActiveDiagram` but the lib's
sidebar consumer never reads it.

**v4 actual UX for a fresh UserDiagram, lib-only: empty canvas + ONE
sidebar drag source ("Alice: User"). No per-class previews.**

### 3. The webapp's seed (`buildUserDiagramSeedNodes`) only fires inside `createEmptyDiagram`

To compensate for §2, the webapp added `buildUserDiagramSeedNodes`
(`webapp/src/main/shared/types/project.ts:259-327`). It hard-codes 4 cards
(Personal_Information / Skill / Education / Disability) with
attribute rows (no `classId` / `className` linkage to the user-meta-model
JSON — pure copies of class names / attribute names).

**The seed only runs when `createEmptyDiagram(UserDiagram)` is called.**
That happens in two places:

1. `createDefaultProject` (`:471`) — fresh project gets all 4 cards.
2. The "missing UserDiagram" auto-add migration (`:519-520`) — when an
   existing project's `obj.diagrams.UserDiagram` is undefined, a fresh
   seeded card-set is added.

The seed does **NOT** retrofit a project that already has a
`UserDiagram` entry but whose `model.nodes` is empty (e.g., a project
saved before SA-UX-FIX-2 (B3) landed, or a project where the user
deleted the seeded cards). For those projects, the user opens the tab
and sees a blank canvas + one sidebar preview ⇒ "only ONE class wtf".

The seed is also not gated by class linkage — `data.classId` /
`data.className` are NEVER set on seeded cards, so the inspector's
`lookupLinkedAttribute` and `lookupEnumerationLiterals` paths return
empty. The integer-comparator gate still fires for `int` rows like
`Personal_Information.age`, but enum-typed rows like
`Personal_Information.gender (GenderEnum)` are NOT recognised as
enumerations (the bridge has `Class` nodes named `Gender` etc., but
the seed row's `attributeType` is the literal string `"GenderEnum"`
which doesn't appear in the bridge — see §"Seed-vs-bridge mismatch"
below).

### Summary root cause

> **The user is opening a UserDiagram that was created (or last saved)
> before `buildUserDiagramSeedNodes` landed, OR they cleared the seeded
> cards. There is no per-class sidebar palette in the v4 lib, so they
> see ONE placeholder ("Alice: User") instead of v3's 4 user-meta-model
> drag sources. The v4 lib's content-loss is structural, not a
> migration-time data-loss bug.**

The migrator preserves all `UserModelName` nodes and their attribute /
icon children faithfully (verified in `tests/round-trip/userDiagram.test.ts`
+ the personalized_gym_agent template's 6 UserModelName cards
round-tripping through v3 fixtures unchanged). No v3 user data is lost
in transit.

## Top-3 gaps (ordered by user impact)

1. **MAJOR — Sidebar palette: no per-class generation.** v3's
   `composeUserModelPreview` produced one drag-source per
   `getAvailableClasses()` class; v4 ships a single static "Alice: User"
   preview (`UserDiagramSVGs.tsx:10-48`). With the bridge already
   populated with `userMetaModel`, this should be a straight port —
   read `diagramBridge.getAvailableClasses()` and emit one preview each
   (with `classId` / `className` set so the inspector's value-widget
   selection works). This is the structural cause of the user's
   "only one class" complaint.

2. **MEDIUM — Seed uses primitive-type strings for enum-typed columns.**
   `buildUserDiagramSeedNodes` (`project.ts:269-305`) emits
   `attributeType: 'GenderEnum'` / `'CharacteristicsEnum'` / `'DegreeEnum'`
   / `'AspectsEnum'` and (typo at `:303`) `attributeType: 'description'`
   for `Disability.name`. None of these are in
   `UserModelNameEditPanel.PRIMITIVE_TYPES`, so the `Select` shows a
   blank value and offers no enum literals (no `classId` link, so
   `lookupEnumerationLiterals` returns `[]`). The seed should either
   set `classId` / `className` so the bridge resolves the enum class,
   or fall back to `'str'`. The `'description'` value at `:303` is a
   straightforward typo (should be `'str'`).

3. **MEDIUM — Project migration does not retrofit existing empty UserDiagrams.**
   `:518-526` only seeds when `obj.diagrams.UserDiagram` is undefined.
   Projects saved between v4 cutover and SA-UX-FIX-2 (B3) shipping
   have an empty `UserDiagram` array — they do not benefit from the
   seed. Either widen the migration to detect "exists-but-empty"
   UserDiagrams (e.g. `model.nodes.length === 0`) and reseed, or ship
   a one-off project-schema bump.

### Other gaps (lower priority, kept for completeness)

4. **MINOR** — `UserModelAttributeEditPanel` (standalone) renders the
   comparator Select unconditionally (`:102-119`). v3 gated by
   `isIntegerType()`; the parent panel matches v3. Standalone path is
   rare (legacy round-trip only).
5. **MINOR** — Per-row `ColorButton` / per-row `StylePane` missing in
   `UserModelNameEditPanel.AttrRow`. v3 (`uml-user-model-attribute-update.tsx:233-240`)
   attached one per row.
6. **MINOR (visual)** — Collapsed `data.icon` body is not rendered
   inside the parent `UserModelName` SVG (only the standalone
   `UserModelIcon` node renders the body). Same regression as
   ObjectName per Wave-2.
7. **CALL-OUT (v3 parity, NOT a regression)** — `classId` picker dropdown
   missing from `UserModelNameEditPanel`. v3 explicitly hid the dropdown
   for UserModelName (`uml-object-name-update.tsx:221`); both v3 and v4
   only edit `className` as free text. The SA-PARITY-FINAL-5 brief
   asked for adding a dropdown — that is an enhancement target.
8. **CALL-OUT** — `UserModelIconEditPanel` not registered. Per the
   `inspectors/userDiagram/index.ts:6` doc string, this is intentional
   and matches v3 (which had no icon-update form either).

## Bridge shape mismatch (informational)

v3's `lookupEnumerationLiterals` (in
`uml-user-model-attribute-update.tsx:111-134`) walks
`diagramBridge.getClassDiagramData().elements` (the v3 flat map shape).
v4's equivalent (`UserModelNameEditPanel.tsx:98-125`) walks
`data.nodes ?? []` (the v4 React-Flow array shape). When the webapp
populates the bridge for a UserDiagram, it passes the **raw imported
JSON** (`userMetaModel as unknown as UMLModel` —
`workspaceSlice.ts:143`), which is the v3 flat-elements shape, NOT a v4
nodes/edges shape. The v4 inspector's `for (const node of data.nodes ?? [])`
loop therefore iterates an empty array on every UserDiagram tab activation
— enum-literal lookup silently returns `[]`. The fact that v3
`elements`-shape data is not exposed as v4 `nodes`-shape is silent — no
runtime error — but the enum picker never works on UserDiagram, even when
a row's `classId` / `className` IS set. This is on the same code-path that
the seed-with-no-classId issue exposes; a single fix for either gap should
also normalise the bridge shape (e.g., run the imported user-meta-model
through `migrateClassDiagramV3ToV4` before `setClassDiagramData`).

## OCL meta-model JSON byte-equality

```
md5: ca7afa7061fc1511c607f9180875974f
- packages/editor/src/main/packages/user-modeling/usermetamodel_buml_short.json
- packages/library/lib/services/userMetaModel/usermetamodel.json
```

**Byte-identical.** Confirmed via `md5sum`. The two siblings of the v3
source (`usermetamodel_buml_less_short.json` md5 `76236017…`, and
`usermetamodel_buml_short_corrected_format.json` md5 `8f40ed8d…`) are not
vendored into the new lib; the canonical `_short` version is the only one
the brief specifies.

## Round-trip / migration verification

`tests/round-trip/userDiagram.test.ts` ships 4 `it` cases:

1. v3 fixture → `migrateUserDiagramV3ToV4` produces v4 with
   `UserModelAttribute` children collapsed onto `data.attributes`
   (line 24).
2. v4 → v3 → v4 structural equality (line 62).
3. attribute rename survives v4 → v3 → v4 (line 108).
4. SA-2.2 #38: `attributeOperator` synthesised from name with full
   5-case regression (line 135) — embedded `>=` extracted, embedded `<`
   extracted, single `=` normalised to `==`, explicit field wins, no
   comparator → undefined.

All UserModelName nodes from a v3 fixture are preserved through the
migrator. **No content-loss bug at the migration layer.**

## Out of scope (Wave-2 / brief follow-ups)

- Icon-view rendering of the inline icon body inside the parent's SVG
  when `settingsService.shouldShowIconView()` is on — Wave-2 (same
  regression as ObjectName).
- `UserModelIcon.features.selectable: false` non-interactive flag — UX
  deviation, not data-fidelity.
- `UserModelIconEditPanel` glyph / image picker — enhancement (v3 had
  no icon-update form either).
