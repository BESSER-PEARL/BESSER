# SA-DEEP-OBJECT — ObjectDiagram v3 vs v4 Audit

**Scope.** Exhaustive evidence-based parity audit of the ObjectDiagram between
the legacy `packages/editor/` (v3) and the migrated `packages/library/` (v4).
Read-only — no source modifications. Same depth as SA-DEEP-CLASS.

**Source files inspected.**

- v3
  - `packages/editor/src/main/packages/uml-object-diagram/index.ts`
  - `packages/editor/src/main/packages/uml-object-diagram/object-preview.ts`
  - `packages/editor/src/main/packages/uml-object-diagram/uml-object-name/{uml-object-name.ts, uml-object-name-component.tsx, uml-object-name-update.tsx}`
  - `packages/editor/src/main/packages/uml-object-diagram/uml-object-attribute/{uml-object-attribute.ts, uml-object-attribute-update.tsx}`
  - `packages/editor/src/main/packages/uml-object-diagram/uml-object-method/uml-object-method.ts`
  - `packages/editor/src/main/packages/uml-object-diagram/uml-object-icon/uml-object-icon.ts`
  - `packages/editor/src/main/packages/uml-object-diagram/uml-object-link/{uml-object-link.ts, uml-object-link-component.tsx, uml-object-link-update.tsx}`
  - `packages/editor/src/main/packages/{components,uml-elements,popups}.ts` (registry rows)
  - `packages/editor/src/main/services/settings/settings-service.ts`

- v4
  - `packages/library/lib/nodes/objectDiagram/{ObjectName.tsx, index.ts}`
  - `packages/library/lib/components/svgs/nodes/objectDiagram/{ObjectNameSVG.tsx, index.ts}`
  - `packages/library/lib/components/inspectors/objectDiagram/{ObjectEditPanel.tsx, ObjectLinkEditPanel.tsx, index.ts}`
  - `packages/library/lib/components/popovers/{PopoverManager.tsx, objectDiagram/ObjectEditPopover.tsx, edgePopovers/ObjectDiagramEdgeEditPopover.tsx}`
  - `packages/library/lib/edges/{edgeTypes/ObjectDiagramEdge.tsx, types.tsx, EdgeProps.ts}`
  - `packages/library/lib/utils/{versionConverter.ts, classifierMemberDisplay.ts, edgeUtils.ts}`
  - `packages/library/lib/types/nodes/NodeProps.ts`
  - `packages/library/lib/services/settingsService.ts`, `lib/store/settingsStore.ts`
  - `packages/library/lib/constants.ts` (`dropElementConfigs.ObjectDiagram`)

---

## A. Palette element types

### v3
`ObjectElementType` (`uml-object-diagram/index.ts:1-6`) declares **four**
element types:

```ts
ObjectName, ObjectAttribute, ObjectMethod, ObjectIcon
```

`ObjectRelationshipType.ObjectLink` (`index.ts:8-10`) is the single edge type.

The palette preview (`object-preview.ts`) renders **only the `ObjectName`**
element as a draggable shape — there is no separate "blank" / "with attrs"
variant. When the global setting `showInstancedObjects` is enabled and the
class-diagram bridge has data, the preview lane is also augmented with
*instance shapes* — one preview per available class, pre-populated with that
class' attributes and (optionally) icon. The `ObjectAttribute` /
`ObjectMethod` / `ObjectIcon` types are owned children, never standalone
palette items. `popups.ts:70-72` confirms this — they each map to `null`
(no inspector), only `ObjectName` and `ObjectLink` get one.

The default seed when dragging a fresh ObjectName from the palette is one
attribute row (`object-preview.ts:131-141` for the generic case): name
`Object`, with one `UMLObjectAttribute` whose name is the localized
`sidebar.objectAttribute` string.

### v4
`packages/library/lib/constants.ts:507-519` registers exactly one
`dropElementConfigs.ObjectDiagram` entry:

```ts
{ type: "objectName", width: DROPS.DEFAULT_ELEMENT_WIDTH, height: 70,
  defaultData: { name: "Object",
                 attributes: [{ id: generateUUID(), name: "attribute = value" }],
                 methods: [] },
  svg: ObjectNameSVG }
```

No `ObjectAttribute` / `ObjectMethod` / `ObjectIcon` palette entries —
attribute rows live inline on `ObjectNodeProps.attributes`, the icon is
collapsed onto `data.icon`, and methods are not modelled at all. The
`methods: []` field on the default data is **vestigial** (kept only because
the type system's earlier object node shape held it; the v4 SVG and inspector
both ignore it).

### Verdict
`A.1` — palette parity: v3 had one palette element (`ObjectName`); v4 has the
same single palette element (`objectName`). MATCH.
`A.2` — v4 default data carries a stray `methods: []`; harmless but worth
cleaning up since the v4 type `ObjectNodeProps` defined at
`types/nodes/NodeProps.ts:167-182` does **not** declare a `methods` field.
LOW-SEV.

---

## B. Canvas rendering

### v3
- `UMLObjectName.render()` (`uml-object-name.ts:146-167`) branches on
  `settingsService.shouldShowIconView()` and, if `true` AND the object owns
  an `ObjectIcon` child with non-empty SVG body, calls `renderIconView()`
  — which positions the icon under the header. Otherwise it falls back to
  `renderNormalView()` which delegates to `super.render()` (UMLClassifier).
- The header uses `underline: boolean = true` (`uml-object-name.ts:24`), so
  the React component (`uml-object-name-component.tsx:115, 127`) emits
  `textDecoration="underline"` on the name. Stereotype band is supported
  (`uml-object-name-component.tsx:104-120`) — when set, two `tspan`s are
  emitted: a smaller `«stereotype»` line and the underlined name below.
- `hasAttributes` divider line is drawn at `headerHeight` when there are
  attribute children (`uml-object-name-component.tsx:137-139`).
- `hasMethods` divider is also drawn when methods exist
  (`uml-object-name-component.tsx:140-142`) — this is the v3 method row.
- `displayName` for `UMLObjectAttribute` (`uml-object-attribute.ts:23-25`)
  returns the bare `name` — no visibility, no `: type`, no `{id}` markers.
  Object rows never carry visibility / id semantics.
- `UMLObjectMethod` (`uml-object-method.ts`) is a thin subclass of
  `UMLClassifierMethod`. Renders via the standard classifier code path; the
  popup `popups.ts:71` does NOT register an inspector for it (`null`).
  Method rows therefore exist on the canvas but are uneditable from the
  popup; only the `UMLObjectNameUpdate` form had a (commented-out) "Add
  method" textfield (`uml-object-name-update.tsx:345-389`).
- `UMLObjectIcon` is a child element with `selectable = false`, `movable
  = false`, `updatable = false` (`uml-object-icon.ts:18-26`). Bounds are
  set by the parent during render.

### v4
- `ObjectName.tsx` (`nodes/objectDiagram/ObjectName.tsx:37-191`) renders a
  resizable `DefaultNodeWrapper` whose body is `ObjectNameSVG`. The SVG file
  itself (`components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx`) is the
  source of truth for rendering decisions:
  - `HeaderSection` always passes `isUnderlined={true}` (line 102) — header
    parity preserved.
  - Stereotype band: `hasStereotype = !!stereotype` (line 31) gates the
    `«…»` band; `headerHeight` swaps to `LAYOUT.DEFAULT_HEADER_HEIGHT_WITH_STEREOTYPE`.
  - Icon view branch (lines 41-46, 111-133): `useSettingsStore(s =>
    s.showIconView)` AND `data.icon` non-empty → renders the icon body via
    `<foreignObject>` with `dangerouslySetInnerHTML`. This is the v4
    equivalent of v3 `renderIconView`.
  - Attributes section: rendered only when `iconViewActive === false` AND
    `showInstancedObjects && attributes.length > 0` (line 138). When
    visible, a `SeparationLine` at `headerHeight` mirrors v3's divider.
  - **No methods section** (line 135-138 comment). SA-FIX-OBJECT-DEEP fix
    is in place.
- The row formatter `formatObjectAttribute` (`ObjectName.tsx:27-35`) routes
  through `formatObjectMember` (`utils/classifierMemberDisplay.ts:97-103`),
  which emits `name = value` (or `name` when `value` is empty). No
  visibility, no type, no `{id}` markers — matches v3
  `UMLObjectAttribute.displayName`.

### Verdict
- `B.1` — header underline: MATCH.
- `B.2` — stereotype band: MATCH (PC-4 Gap 1 closed in both `ObjectName.tsx`
  width budget and `ObjectNameSVG.tsx` band rendering).
- `B.3` — icon view: MATCH (icon SVG body rendered inline; both gated by
  `showIconView` setting AND presence of icon content).
- `B.4` — `ObjectMethod` row removed in v4: CONFIRMED. No methods section
  is ever rendered, no method element is registered in the palette or in
  any drop config, and `ObjectNodeProps` carries no `methods` field.
- `B.5` — divider line above attributes: MATCH (`SeparationLine` at
  `headerHeight` when attributes are visible).

Caveat C.5 (below) — "show attributes only when `showInstancedObjects` is
true" is a v4-only behavioural change that does NOT match v3 for *manually
authored* objects.

---

## C. Inspector panel

### v3 (`uml-object-name-update.tsx`)
- `Textfield` for object `name` (line 226).
- `Class:` dropdown shown only when `availableClasses.length > 0`
  (`showClassSelection`, line 221). Driven by
  `diagramBridge.getAvailableClasses()` which folds inheritance — see
  `getClassDisplayName` (`uml-object-name-update.tsx:63-79`) which appends
  `extends Parent, Other` and ` (N attrs)`.
- `onClassChange` (`uml-object-name-update.tsx:80-130`):
  - sets `classId` on the element,
  - if name is `"Object"` or empty, replaces with `<className>Instance`,
  - **deletes all existing `UMLObjectAttribute` children**,
  - creates one new `UMLObjectAttribute` per `selectedClass.attributes`
    entry, name set to `"<attrName> = <defaultValue>"`, `attributeId`
    pinned to the class attribute id, `attributeType` cached.
- Per-attribute row inspector is `UMLObjectAttributeUpdate`
  (`uml-object-attribute-update.tsx`) — the rich, type-aware widget:
  - parses `name = value` and renders the `name = ` label as a fixed
    span and the value as either:
    - a `Dropdown` for enumeration types (lookup via `diagramBridge`),
    - a typed `<input type="date|datetime-local|time">` for date/time
      types,
    - a custom duration input for `timedelta`/`duration`/`period`/`timespan`,
    - a quoted `Textfield` (with `"…"` wrappers) when `attributeType === 'str'`,
    - a plain `Textfield` for everything else (`getAttributeType`
      fallback).
  - placeholder text uses the attribute type for hinting.
- The "Add attribute" textfield is **commented out** in v3
  (`uml-object-name-update.tsx:307-343`). Only auto-population via
  `onClassChange` produces attribute rows.
- The methods section (lines 345-389) is also commented out.
- Color editor via `StylePane` (fillColor + lineColor + textColor at the
  object level; per-attribute fillColor + textColor).

### v4 (`ObjectEditPanel.tsx`)
- `NodeStyleEditor` for name + colors (line 324-328); placeholder
  `<className>Instance` parity with v3.
- `Select` driven by `diagramBridge.getAvailableClasses()` (line 339-367).
  The display label appends the inheritance chain (`extends Parent, …`)
  and `(N attrs)` — full PC-4 Gap 2 parity with v3
  `getClassDisplayName`.
- `handleClassChange` (line 197-245):
  - drops all existing rows when the user picks "— Unlinked —";
  - rebuilds rows from `selectedClass.attributes` — id new, name set to
    bare `a.name` (no `= value` baked into the name; the value is stored
    structurally in `row.value` from the default), `attributeId` pinned,
    `attributeType` cached;
  - renames the object to `<className>Instance` only if `name` is empty
    or literal `"Object"` (matches v3 condition).
- Per-attribute row (`ObjectAttrRow`, lines 42-110):
  - **Plain `name = value` two-textfield widget**, no type-aware
    dropdowns, no date/time pickers, no quoted-string widget. This is
    the SA-FIX-OBJECT-DEEP simplification.
  - `displayType` (resolved via `attributeId` ➝ `name` ➝
    stored `attributeType`) is rendered as a small italic caption
    (`: <type>`) beside the value field — read-only. So the type
    *does* still surface in the UI.
- "Add attribute" textfield (line 382-396) is **always present**. The user
  can always type a name + Enter (or blur) to append a new row.

### Verdict
- `C.1` — class dropdown: MATCH.
- `C.2` — auto-populate attributes on class change: MATCH (with one minor
  divergence — see C.6).
- `C.3` — type still surfaces in the UI: PARTIAL. v4 surfaces the type as a
  read-only caption (good — the spec asks for this). v3 surfaced it as
  active widget behaviour (date picker, enum dropdown, etc.). The
  *user-direction* simplification per the prompt is intentional.
- `C.4` — per-row widget simplified to plain text: CONFIRMED. v4 uses two
  flat MUI textfields plus `=` glyph.
- `C.5` — color editor: MATCH at the node level (`NodeStyleEditor`). NO
  per-row color editor in v4 (v3 had one). LOW-SEV gap.
- `C.6` — v3 stored attribute names as `<name> = <defaultValue>`; v4 stores
  bare `name` + structured `value` field. The version converter
  (versionConverter.ts:776-781 import / 2348-2350 export) round-trips this
  correctly. MATCH.

---

## D. ObjectLink edge

### v3 (`uml-object-link.ts`, `uml-object-link-component.tsx`,
`uml-object-link-update.tsx`)
- Shape: plain `<polyline>`, `strokeColor`, `strokeWidth=2`, no
  fill, **no markers** (`uml-object-link-component.tsx:5-14`).
- `IUMLObjectLink` carries the optional root-level `associationId` field;
  serializer/deserializer round-trip it (`uml-object-link.ts:7-30`).
- Inspector:
  - `name` text field (line 152-157).
  - flip (swap source/target) action (line 125-127).
  - association picker (lines 159-185) driven by
    `diagramBridge.getAvailableAssociations(sourceClassId, targetClassId)`
    — class IDs walked from `source` / `target` ObjectName nodes' `classId`.
  - color editor (`StylePane`, lines 132-138) — `lineColor` + `textColor`.
  - "No associations available" hint when both endpoints are unlinked or
    the bridge returns nothing (lines 187-191).

### v4
- Edge component `ObjectDiagramEdge.tsx`:
  - Uses `useStepPathEdge` → step path (orthogonal segments). v3 used
    `<polyline>` from a Manhattan layout — visually equivalent.
  - `markerEnd`/`markerStart` resolved through `getEdgeMarkerStyles`
    (`utils/edgeUtils.ts:173-187`): `ObjectLink` is grouped with
    `ClassBidirectional` / `DeploymentAssociation` / `SyntaxTreeLink` /
    `CommunicationLink` and produces `markerPadding`, no actual marker
    head — i.e. **no arrow, plain line**. MATCH with v3.
  - `strokeDashArray: "0"` ⇒ solid line. MATCH.
  - `useEdgeConfig` (`edges/types.tsx:119`) → `{ allowMidpointDragging:
    true }`. Midpoint dragging exists in v4 but not v3 — additive, no
    regression.
- Inspector wiring is **non-uniform**:
  - `PopoverManager.tsx:275` registers `ObjectLink` →
    `ObjectDiagramEdgeEditPopover` (the **stub** — only the style
    editor, no name field, no association picker, no flip).
  - `inspectors/objectDiagram/index.ts:12` registers `ObjectLink` /
    `edit` → `ObjectLinkEditPanel` (the **rich** body — name, flip,
    association picker, style editor).
  - The active inspector is whichever surface is rendered. With the
    properties-panel surface (`usePropertiesPanel = true` default) the
    `ObjectLinkEditPanel` is reached via
    `inspectors/registry.ts::getInspector("ObjectLink", "edit")`. With the
    floating popover surface the `ObjectDiagramEdgeEditPopover` stub is
    reached. **This is a real divergence** — see `D.1` below.
- Round-trip of `associationId`:
  - V3 → V4 (`versionConverter.ts:1828-1838, 1880-1884`): pulls
    `relationship.associationId` (root-level on V3) into the v4 edge's
    `data.associationId`.
  - V4 → V3 (`versionConverter.ts:2456-2461`): writes
    `data.associationId` back to `relationship.associationId` at root
    level on the V3 element.
  - MATCH — `associationId` survives a full round trip on the
    `ObjectLink` edge.

### Verdict
- `D.1` — **inspector divergence**: with the floating popover surface, the
  ObjectLink shows only the style editor (no name, no association
  picker, no flip). The `ObjectDiagramEdgeEditPopover` is a stub. The
  rich body lives only in `ObjectLinkEditPanel`. HIGH-SEV gap.
- `D.2` — visual marker style (no arrowhead): MATCH.
- `D.3` — `associationId` round-trip: MATCH.
- `D.4` — flip action: rich body has it; stub does not.

---

## E. Default data on drop

### v3
The palette preview (`object-preview.ts:124-141`) creates a fresh `ObjectName`
seeded with **one attribute row** (`UMLObjectAttribute`, name set to a
localized "attribute" placeholder). `ownedElements = [umlObjectMember.id]`.

### v4
`constants.ts:512-516` seeds `defaultData` with **one attribute row**:
```ts
attributes: [{ id: generateUUID(), name: "attribute = value" }]
```

Also includes the vestigial `methods: []` (see A.2).

### Verdict
`E.1` — fresh-drop seeded with one attribute row: MATCH.
`E.2` — placeholder text differs ("Object Attribute" / "attribute" in v3 vs
literal `"attribute = value"` in v4). LOW-SEV cosmetic; reasonable since v4
parses the `= value` portion as the structured `value` field on edit.

---

## F. Migrator

### v3 → V4 (`versionConverter.ts:760-827`)
- Walks `allElements` for children where `owner === element.id`:
  - `ObjectAttribute` → `ObjectNodeAttribute`. The `name = value` form on
    the v3 row gets split into bare `name` and structured `value` (lines
    776-781).
  - `ObjectMethod` → **DROP** (lines 783-794). `console.warn` logged
    with the offending id and parent id; no node row, no rendered
    output. SA-FIX-OBJECT-DEEP guarantee held. CONFIRMED.
- `classId` / `className` lifted from the V3 element verbatim (lines
  801-806).
- `stereotype` lifted onto the v4 node (lines 807-813).
- `ObjectIcon` collapsed: walk children for the icon element and copy
  `child.icon` (an SVG body string) into `data.icon` (lines 818-824).
- The migrator's outer node-skip filter (lines 1939-1949) skips
  `ObjectAttribute`, `ObjectMethod`, `ObjectIcon` — they're not turned
  into stand-alone nodes. CORRECT.

### V4 → V3 (`versionConverter.ts:2304-2366`)
- `objectName` → V3 `ObjectName` element with:
  - `attributes`: list of child ids,
  - `methods: []` (line 2325) — empty array on the V3 element. The
    SA-FIX-OBJECT-DEEP comment (lines 2322-2325, 2353-2354) is explicit:
    no `ObjectMethod` row is ever emitted. CONFIRMED.
  - `classId` / `className` re-emitted only when set (line 2329-2330).
  - `stereotype` re-emitted only when non-null (line 2333-2334).
  - For each attribute: `childRowToV3(attr, … "ObjectAttribute")`. When
    `attr.value` is set, the v3 child's `name` is recombined to
    `"<name> = <value>"` (lines 2348-2350). `attributeId` pinned.
- `ObjectIcon`: when `data.icon` is set, a synthetic V3 child element is
  emitted with id `"<nodeId>-icon"`, type `ObjectIcon`, body in `icon`
  (lines 2356-2366).

### Verdict
- `F.1` — V3→V4: drops legacy `ObjectMethod` rows with a `console.warn`.
  CONFIRMED in code at `versionConverter.ts:783-794`.
- `F.2` — V4→V3 round-trip never re-introduces methods. CONFIRMED at
  `versionConverter.ts:2322-2325, 2353-2354`.
- `F.3` — `associationId` round-trip on `ObjectLink`: CONFIRMED, see D.3.
- `F.4` — Icon collapse + re-emit: parity preserved.

---

## G. `showInstancedObjects` setting

### v3
`settings-service.ts:13` declares `showInstancedObjects: boolean` (default
`true`). `shouldShowInstancedObjects()` (line 180-182) is consumed by
`object-preview.ts:34, 145` to decide whether to add per-class instance
preview shapes to the palette/preview lane.

### v4
- Field declared in `services/settingsService.ts:13` (default `true`,
  line 31).
- Mirrored into the Zustand `useSettingsStore` and exposed via
  `useSettingsStore(s => s.showInstancedObjects)`.
- Consumed in **one** v4 site:
  `components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx:56-59`:
  ```ts
  const showInstancedObjects = useSettingsStore((s) => s.showInstancedObjects)
  const showAttributes = showInstancedObjects && attributes.length > 0
  ```
  When the toggle is OFF, the entire **attributes section is suppressed**
  on every ObjectName node — only the underlined header is rendered.

### Verdict
- `G.1` — setting is wired into the store and read at render time: YES.
- `G.2` — does it work? **The toggle is overloaded.** In v3 the setting
  controls the *palette preview lane only* (whether instance shapes are
  appended to the sidebar preview). In v4 the same setting controls
  *whether attribute rows render on every ObjectName on the canvas*. With
  the setting OFF, **all manually authored attribute rows disappear from
  every object instance**. This is a behavioural divergence and likely
  not what an author wants. HIGH-SEV.

---

## H. "Hide Add attribute when classId is set"

User direction (per the prompt): when `classId` is set, the user wants the
"Add attribute" button HIDDEN — attributes should auto-populate from the
linked class only.

### v3
The "Add attribute" textfield is commented out unconditionally
(`uml-object-name-update.tsx:307-343`). Effectively v3 hides it always,
not specifically when `classId` is set. Auto-populate runs in
`onClassChange`. Hide rule for "with classId" is *coincidentally* satisfied
because the field is hidden in all cases.

### v4
`ObjectEditPanel.tsx:382-396` renders the "Add attribute" textfield
**unconditionally**. The render does not branch on `nodeData.classId`.
Therefore when the user picks a class from the dropdown, the auto-populated
rows appear AND the "+ Add attribute (Enter)" textfield is still shown
below them.

### Verdict
- `H.1` — v4 does NOT hide the "Add attribute" textfield when `classId` is
  set. The user direction is **NOT IMPLEMENTED**. HIGH-SEV gap.

---

## Critical gaps (ranked)

| ID | Sev | Description | Fix locus |
|----|-----|-------------|-----------|
| `D.1` | HIGH | Floating-popover ObjectLink inspector is a style-only stub — no name, no association picker, no flip. Properties-panel surface gets the rich `ObjectLinkEditPanel`; popover surface does not. | `components/popovers/edgePopovers/ObjectDiagramEdgeEditPopover.tsx` — replace stub body with a wrapper that delegates to / mirrors `ObjectLinkEditPanel`. |
| `G.2` | HIGH | `showInstancedObjects` re-purposed: in v4 it hides attribute rows on every canvas ObjectName, not just palette preview shapes. Toggling it off destroys author-visible content. | `components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx:56-59` — either gate by "is preview render" instead of every render, or rename the v4 setting / drop the gate. |
| `H.1` | HIGH | "Add attribute" textfield is shown even when `classId` is set, against user direction. Auto-population only runs at the moment of class change; the trailing textfield then lets the user add rows that won't be linked to a class attribute and have no `attributeType`. | `components/inspectors/objectDiagram/ObjectEditPanel.tsx:382-396` — wrap in `{!nodeData.classId && (…)}`. |
| `C.5` | LOW | Per-attribute row in v4 inspector has no color picker (v3 had `fillColor` + `textColor` on each row). | `ObjectEditPanel.tsx::ObjectAttrRow` — add a `<ColorPicker>` button if needed (only if v4 row colors are used elsewhere). |
| `A.2` | LOW | Vestigial `methods: []` on `dropElementConfigs.ObjectDiagram[0].defaultData`. Type `ObjectNodeProps` does not declare `methods`. | `constants.ts:507-519` — remove `methods: []` from the default data. |
| `E.2` | LOW | Default attribute placeholder text differs ("attribute = value" literal in v4). Cosmetic. | `constants.ts:514` — match v3 i18n string if desired. |

No `BLOCKING` gaps were found for the migrator / round-trip path. F.1, F.2,
F.3 all confirm clean v3↔v4 parity for `ObjectMethod` dropping and
`associationId` preservation.

---

## Sign-off

- **Palette parity (A):** clean. One palette element on each side.
- **Canvas rendering (B):** parity preserved (header underline, stereotype
  band, icon view, attribute divider). `ObjectMethod` row truly gone in
  v4 — no palette entry, no SVG path, no node-prop field.
- **Inspector (C):** node inspector parity good (class picker + dropdown +
  auto-populate + display-type caption); per-row widget intentionally
  simplified to plain text per the user direction. Per-row color picker
  was dropped (LOW).
- **ObjectLink (D):** node panel surface is correct; popover surface is a
  stub (HIGH gap D.1). Edge visual + `associationId` round-trip clean.
- **Default drop data (E):** one attribute row seeded — MATCH.
- **Migrator (F):** clean. v3→v4 drops `ObjectMethod` with a warning,
  v4→v3 emits `methods: []`, `associationId` survives both directions.
- **Settings (G):** wired but overloaded — `showInstancedObjects` in v4
  collapses canvas attributes (HIGH gap G.2).
- **Add-attribute hide rule (H):** NOT IMPLEMENTED (HIGH gap H.1).

Overall verdict: **mostly aligned**, three HIGH-severity behavioural gaps
(`D.1`, `G.2`, `H.1`) require lib-side fixes before this can be considered
parity-complete.
