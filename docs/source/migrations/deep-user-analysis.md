# SA-DEEP-USER — UserDiagram v3 vs v4 audit

Submodule HEAD: `771c064` (parent branch `claude/refine-local-plan-sS9Zv`).
READ-ONLY. The previous parity-final report
(`docs/source/migrations/parity-final/userDiagram.md`) declared
"PASS WITH MINOR GAPS"; this audit re-checks every requested point with
file:line citations and is brutal where the previous report was generous.

The user complaint — *"we should have the view icone like what we had into
the older one"* — turns out to be the most important finding: v4 has the
icon plumbing wired but the **settings-toggle gate is missing**, so icon
view is *not* the same behaviour as v3.

---

## Verdict

**FAIL on icon-view parity (user-reported gap C+G).** The remainder of the
brief (palette generator, header underline, attribute row format, classId
picker, OCL JSON identity, comparator-from-name migrator, edge alias)
substantially matches v3. But the user-reported icon issue is real and
unfixed: v4 `UserModelNameSVG` never consults
`settingsService.shouldShowIconView()` / `useSettingsStore.showIconView`,
and `UserModelName.tsx` does **not** delegate to `ObjectNameSVG` despite
what the parity-final report claimed. The icon body renders whenever
`data.icon` is non-empty, which is the *inverse* of v3's behaviour
(v3 required the global toggle ON in addition to a non-empty icon).

---

## A. Palette — dynamic per-class drag sources

### v3
`packages/editor/src/main/packages/user-modeling/user-model-preview.ts:12-143`
exports `composeUserModelPreview` which:

- consults `settingsService.shouldShowIconView()` (line 16) and branches
  to `composeIconView` or `composeNormalView`;
- in **both** branches: calls
  `diagramBridge.getAvailableClasses()` (lines 25, 96) and emits **one
  `UMLUserModelName` per class** (`line 28-78`, `99-141`), naming it
  `${className.toLowerCase()}_1` (lines 29, 100) and stamping each
  `classInfo.attribute` as a `UMLUserModelAttribute` row in
  `${attr.name} = ` form (lines 58-65, 119-132);
- in icon-view, also stamps a `UMLUserModelIcon` child (lines 48-55).

The bridge is pre-loaded with the user-meta-model JSON elsewhere, so
`getAvailableClasses()` returns Personal_Information, Skill, Education,
Disability (User is filtered as the placeholder root).

### v4
`packages/library/lib/components/svgs/nodes/userDiagram/UserDiagramSVGs.tsx:308-319`
`getUserModelNamePaletteEntries()`:

- reads `getUserMetaModelClasses()` from
  `lib/services/userMetaModel/index.ts:66-94` (which walks the v3-shape
  flat-element JSON and **excludes `User`** at line 72);
- returns one entry per class with `attributes` and a per-class SVG
  (`makeUserModelPaletteSVG`, lines 262-290) that pre-stamps
  `${className.toLowerCase()}_1` and `${attrName} =` rows — matches v3
  format exactly.

`lib/constants.ts:986-1021` consumes that list:

```
[UMLDiagramType.UserDiagram]: [
  ...getUserModelNamePaletteEntries().map((entry) => ({
    type: "UserModelName" as never,
    width: DROPS.DEFAULT_ELEMENT_WIDTH,
    height: 40 + entry.attributes.length * 30 + 10,
    defaultData: { name: `${entry.className.toLowerCase()}: ${entry.className}`,
                   className, attributes: <stamped> },
    svg: entry.svg,
  })),
  { /* Alice : User static fallback */ },
  { /* UserModelIcon (icon glyph) */ },
],
```

### Status — A: PASS
Personal_Information, Skill, Education, Disability all expand at module-
load. JSON inspection confirms those four `Class` entries
(`grep '"name":' usermetamodel.json` — User + the 4 expected names).

### Caveats
- **Drop-name format diverges from v3:** v3 stamped instance name
  `personal_Information_1` (preserving the post-underscore casing per
  `composeUserModelPreview:29`); v4 default-data name is
  `${className.toLowerCase()}: ${className}` →
  `"personal_information: Personal_Information"` — a header label rather
  than an instance-name. The palette-PREVIEW SVG at `UserDiagramSVGs.tsx:279`
  uses the v3 form, but the dropped node's `data.name` is the colon form,
  so the two are visually inconsistent. **MINOR.**
- The v3 `composeUserModelPreview` icon-view branch
  (`user-model-preview.ts:20-81`) is not present in v4 — v4 has only one
  palette form. With no settings-aware palette, the toggle has no effect
  on the sidebar previews. **MINOR.**

---

## B. Canvas rendering

### v3 spec
`packages/editor/src/main/packages/user-modeling/uml-user-model-name/uml-user-model-name.ts:21-233`:

- `underline: boolean = true` (line 23) — header always underlined.
- `render()` (line 161) checks `settingsService.shouldShowIconView()` (line 162)
  AND `children.some(... type === UserModelIcon && x.icon && x.icon.trim() !== '')`
  (lines 165-171); only when **both** are true does it call
  `renderIconView` (line 174). Otherwise → `renderNormalView` →
  `super.render` which draws the `name = value` rows.
- No visibility symbols — `UMLUserModelAttribute extends UMLClassifierAttribute`
  but the row is rendered by the user-model SVG which never draws `+/-`.

### v4 implementation
`packages/library/lib/nodes/userDiagram/UserModelName.tsx:1-189` calls
its own `UserModelNameSVG` (line 4 import, 165 render) — **NOT
ObjectNameSVG**. The parity-final report at line 223-227 claimed v4
"reuses `ObjectNameSVG`"; that is incorrect.

`UserDiagramSVGs.tsx:64-192` `UserModelNameSVG`:

- header `name : className` rendered with
  `textDecoration="underline"` (lines 124-140) → underlined header **OK**.
- attribute rows rendered through `RowBlockSection` with
  `itemElementType="attribute"` (line 181) — no visibility symbols **OK**.
- formatter `formatUserModelAttributeForDisplay` (`UserModelName.tsx:39-54`)
  emits `name op value` exactly as the brief specifies.
- **icon-view branch is gated on `data.icon` only** (line 92,
  `hasIcon = typeof icon === "string" && icon.trim() !== ""`). When an icon
  string is present, lines 145-165 render it via `<foreignObject>` and
  the attributes block is suppressed (line 167 `{!hasIcon && …}`).
  **The global toggle is never consulted.** `grep` for
  `useSettingsStore|showIconView|settingsService` in
  `lib/components/svgs/nodes/userDiagram/` and `lib/nodes/userDiagram/`
  returns **zero hits**.

### Status — B: PARTIAL
- header underline + `name op value` rows + no visibility: **OK**.
- icon-view gating: **FAIL**. v3 required `shouldShowIconView()` AND a
  non-empty icon; v4 only requires the icon. With the toggle off, v4 still
  renders the icon body and hides attributes — the opposite of v3.

For comparison, `ObjectNameSVG.tsx:44-46` correctly gates on
`useSettingsStore((s) => s.showIconView)`. The user-diagram SVG should
mirror that.

---

## C. Inspector

### v3 spec
- The v3 fork delegated to `UMLObjectNameUpdate` for the user diagram and
  pre-loaded the user-meta-model JSON into `diagramBridge` so the same
  `getAvailableClasses()` call returned the user-meta-model's classes.
- `uml-user-model-attribute-update.tsx` (lines 81-134) reads
  `diagramBridge.getClassDiagramData()`, looks up enumerations as
  `candidate.type === 'Enumeration'`, gates the comparator dropdown on
  `isIntegerType()` (line 187), and renders an enum-literal `<Dropdown>`
  when `enumValues.length > 0` (line 214).

### v4 implementation
`packages/library/lib/components/inspectors/userDiagram/UserModelNameEditPanel.tsx`:

- `buildMetaContext` (lines 100-118) reads `getUserMetaModelV4()` directly
  and walks `node.data.stereotype === "Enumeration"` (line 108) — i.e.
  the inspector reads the JSON, **not** the bridge. Matches the brief's
  requirement that the picker be driven by the meta-model JSON.
- Class-picker `<Select>` (lines 451-467) iterates `metaCtx.classes`.
- `onClassChange` (lines 367-398) resets attributes, stamps fresh ones
  from the chosen meta-class, and updates `classId` / `className` —
  matches v3's auto-stamp semantics.
- `AttrRow` (lines 127-313):
  - resolves the linked meta-attribute via `metaCtx.classes` (lines 137-143);
  - effective type lower-cased (line 145) → INTEGER / BOOLEAN / DATE /
    DATETIME / TIME / STRING checks (lines 149-154);
  - integer-gated comparator dropdown (lines 232-250);
  - enum literal dropdown when `enumLiterals.length > 0` (lines 258-271);
  - `<MuiTextField type="date|time|datetime-local">` for date kinds
    (lines 283-291);
  - string placeholder hints `value (will be quoted: "…")` (line 297) —
    note this is only a placeholder; **no actual quote-wrap is applied**
    when the value is persisted. `currentValue` is stored verbatim.

`UserModelAttributeEditPanel.tsx` mirrors the same logic for standalone
nodes (lines 66-263) — the previous parity-final §279 minor-gap noting
"always shows comparator" is **stale**: line 167 now correctly gates on
`isInteger`. Confirmed.

### Status — C: PASS with three sub-gaps
1. **String quote-wrap is cosmetic only.** Brief says "str quote-wrap";
   the v4 implementation only places the hint in the placeholder and does
   not actually wrap the persisted `value` in quotes. v3 had the same
   placeholder-only behaviour, so this is parity-neutral but the brief's
   wording suggests it should actually wrap. **MINOR.**
2. **No per-row color picker.** v3
   `uml-user-model-attribute-update.tsx:233-238` had `<ColorButton>` and
   a `StylePane`. v4 row has only delete (line 252). **MINOR.**
3. **Boolean handling is not in the v3 spec but is added here.** Lines
   272-282 add a true/false `<Select>` for `bool` types. **EXTRA — not a
   regression.**

---

## D. UserModelLink edge

`packages/library/lib/edges/edgeTypes/UserModelLink.tsx:1-19` —
`export const UserModelLink = ObjectDiagramEdge`. Side-effect-registers
itself (line 18). v3 likewise had no inspector for `UserModelLink`
(`UserModelRelationshipType` is a single-string registry, no
update-pane file under
`packages/editor/src/main/packages/user-modeling/`).

`packages/library/lib/edges/types.tsx` lists `UserModelLink` in
`EDGE_TYPE_FEATURES` with `allowMidpointDragging: true`.

### Status — D: PASS (parity-neutral)

---

## E. OCL meta-model JSON byte-identity

```
md5sum
ca7afa7061fc1511c607f9180875974f  packages/editor/src/main/packages/user-modeling/usermetamodel_buml_short.json
ca7afa7061fc1511c607f9180875974f  packages/library/lib/services/userMetaModel/usermetamodel.json
```

### Status — E: PASS — byte-identical.

The two siblings `usermetamodel_buml_less_short.json`
(md5 `76236017b24aced1921863e8a973ce90`) and
`usermetamodel_buml_short_corrected_format.json`
(md5 `8f40ed8db09be3a7e06a8b81c505ba19`) are not vendored into v4; the
canonical short form is the only one the brief specifies and it round-trips
unchanged.

---

## F. Migrator — synthesize attributeOperator from name

`packages/library/lib/utils/versionConverter.ts`:

- `extractAttributeOperatorFromName` (lines 510-519) — regex
  `/^(?:.*?)(<=|>=|==|=|<|>)/`, normalizes `=` to `==`. Mirrors v3's
  `extractComparatorFromName` (`uml-user-model-attribute.ts:27-33`)
  except it returns `undefined` when no comparator is found, letting the
  caller decide whether to fall back.
- `case "UserModelName"` (lines 1238-1316): walks `allElements` for
  `owner === element.id && type === "UserModelAttribute"` (lines 1253-1257)
  and builds the inline `attributes` array. The synthesizing block
  (lines 1272-1274):

  ```ts
  const synthesizedOperator =
    c.attributeOperator ??
    extractAttributeOperatorFromName(c.name)
  ```

  → explicit field wins, name-extracted is the fallback. Matches v3's
  `deserialize` precedence at `uml-user-model-attribute.ts:73-77`.
- `case "UserModelAttribute"` (lines 1318-1342): same precedence applied
  to standalone rows.

Round-trip test `tests/round-trip/userDiagram.test.ts` case (4) ("synthesizes
attributeOperator from embedded comparator in name") covers all five
comparator variants per the parity-final report §256.

### Status — F: PASS.

---

## G. User-reported gap — "view icone like what we had into the older one"

The user said **two** things in the earlier conversation: (1) "nothing in
common" between v3 and v4 (which is hyperbolic — A/D/E/F/most of B and C
are in common), and (2) explicitly asked for the v3 icon view.

### What v3 actually did

`uml-user-model-name.ts:161-180`:

```ts
render(layer, children = []) {
  const shouldShowIconView = settingsService.shouldShowIconView();
  if (shouldShowIconView) {
    const hasValidIcon = children.some(
      (x) => x.type === UserModelIcon && x.icon && x.icon.trim() !== ''
    );
    if (hasValidIcon) {
      return this.renderIconView(layer, children);  // <-- swap layout
    }
    return this.renderNormalView(layer, children);
  }
  return this.renderNormalView(layer, children);
}
```

`renderIconView` (lines 182-210) sets node bounds to fit the SVG, drops
the attributes table entirely, and returns `[this, icon]` — the icon
glyph is the body.

### What v4 actually does

`UserDiagramSVGs.tsx:64-192`:

```ts
const hasIcon = typeof icon === "string" && icon.trim() !== ""
…
{hasIcon && (
  <foreignObject … dangerouslySetInnerHTML={{ __html: icon as string }} />
)}
{!hasIcon && attributes.length > 0 && ( /* attribute rows */ )}
```

`grep useSettingsStore|showIconView|settingsService` in
`lib/components/svgs/nodes/userDiagram/` + `lib/nodes/userDiagram/` →
**0 matches**. Confirmed: the global toggle is never read.

### Bug shape

| Toggle | data.icon | v3 result | v4 result |
|---|---|---|---|
| OFF | empty | normal (rows) | normal (rows) — match |
| OFF | non-empty | **normal (rows)** | **icon body, rows hidden** — MISMATCH |
| ON  | empty | normal (rows) | normal (rows) — match |
| ON  | non-empty | **icon body, rows hidden** | icon body, rows hidden — match |

So when an authoring-time icon is present and the user toggles "show
icon view" OFF, v3 falls back to the rows but v4 keeps showing the icon.
That is the bug the user is reporting in colourful terms. The v4 code
also has no path to surface the **palette icon glyph** in icon-view —
v3's `composeUserModelPreview` icon-view branch (lines 20-81) returned
the icon-element preview; v4's palette is icon-agnostic.

The parity-final report §229-233 acknowledges this (calling it "Wave-2
follow-up") but the SA-DEEP-USER brief explicitly elevates it to a
**critical gap** because it is the user-visible regression.

### Status — G: FAIL (the reported gap is real and unfixed).

Suggested fix: in `UserDiagramSVGs.tsx`, mirror `ObjectNameSVG.tsx:44-46`:

```ts
const showIconView = useSettingsStore((s) => s.showIconView)
const iconViewActive = showIconView && hasIcon
{iconViewActive && (<foreignObject …/>)}
{!iconViewActive && attributes.length > 0 && (<rows/>)}
```

(`ObjectNameSVG` already does exactly this, so the pattern is in-repo.)

---

## Top 3 gaps

1. **G: icon-view toggle is not gated** — `UserModelNameSVG` always shows
   `data.icon` when present, ignoring `useSettingsStore.showIconView`.
   This is the user-reported regression and the most visible bug. Fix is
   ~6 lines, mirroring `ObjectNameSVG.tsx:44-46`.
2. **A (minor): drop-time `data.name` diverges from preview SVG** —
   palette preview shows `personal_Information_1` (v3 form) but the
   `defaultData.name` shipped on drop is
   `personal_information: Personal_Information`. Pick one and apply it
   in both places (`constants.ts:992` + `UserDiagramSVGs.tsx:279`).
3. **C (minor): missing per-row color picker** — v3
   `uml-user-model-attribute-update.tsx:233-238` had a `<ColorButton>` +
   `<StylePane>`; v4 row has only delete. Fix lives in
   `UserModelNameEditPanel.tsx:AttrRow` lines 252-254.

## Things the previous parity-final report got wrong

- "v4 reuses `ObjectNameSVG`" (parity-final §223-225) — **false**;
  `UserModelName.tsx:4` imports `UserModelNameSVG` from
  `@/components/svgs/nodes/userDiagram`, a dedicated module owning its
  own underline + icon rendering.
- "Standalone `UserModelAttributeEditPanel` always shows comparator"
  (parity-final §277-279) — **stale**;
  `UserModelAttributeEditPanel.tsx:167` correctly gates on `isInteger`.

## Things the user got wrong

- "Nothing in common" — also false. A, D, E, F and the bulk of B+C are
  byte/behaviour identical. The user is responding to the icon-view bug
  (G) which is real but localised.
