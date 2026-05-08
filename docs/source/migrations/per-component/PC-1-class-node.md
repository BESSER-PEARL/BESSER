# PC-1 — Class node (canvas) parity audit

Read-only audit of the Class node migration. The Class node is the on-canvas
classifier shape — header band + attribute compartment + method compartment —
shared by `Class`, `AbstractClass`, `Interface`, and `Enumeration` stereotype
variants.

- **v3 source-of-truth (old)**:
  - `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-class-diagram/uml-class/uml-class.ts`
  - `…/uml-class-diagram/uml-abstract-class/uml-abstract-class.ts`
  - `…/uml-class-diagram/uml-interface/uml-interface.ts`
  - `…/uml-class-diagram/uml-enumeration/uml-enumeration.ts`
  - `…/common/uml-classifier/uml-classifier.ts` (base layout/render)
  - `…/common/uml-classifier/uml-classifier-component.tsx` (SVG render of the box)
  - `…/common/uml-classifier/uml-classifier-member-component.tsx` (row render)
  - `…/uml-class-diagram/class-preview.ts` (sidebar preview)
- **v4 target (new)**:
  - `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/classDiagram/Class.tsx`
  - `…/components/svgs/nodes/classDiagram/ClassSVG.tsx`
  - `…/components/svgs/nodes/HeaderSection.tsx`
  - `…/components/svgs/nodes/RowBlockSection.tsx`
  - `…/types/nodes/NodeProps.ts` (`ClassNodeProps`)
  - `…/types/nodes/enums/ClassType.ts`
  - `…/initialElements.ts` (palette previews)

Submodule pin: `34b390cfdc0da6080d1e9b14bb19682cd68a1520` ("SA-7b.2 finish v4-native rewrite").

Status legend: **PASS** = feature is present and matches v3. **GAP** =
missing, regressed, or behaves differently. **N/A** = scope item the brief
flags but neither version supports (called out for completeness).

---

## Overall verdict — **GAPS** (5 high-severity, 4 medium, 2 low)

The new Class node renders the right *shape* and matches v3 on header/row
typography, fill/stroke/text colors, and four-side connect handles. It also
does an arguably cleaner job of width auto-fitting via Zustand re-render on
the ER↔UML toggle. However it loses three semantically-meaningful v3
behaviours (freeform stereotype, explicit italic/underline flags, and
ER-mode method-compartment hiding) and introduces two SVG layout bugs
(method-row Y offset uses the wrong constant; Enumeration still draws an
empty methods compartment + divider). It also drops in-place rename
(double-click now opens the popover instead of editing the name on the
canvas — though the popover does carry a name field, so the function exists
in another shape).

---

## 1. Visual — dimensions, header band, stereotype label

| Axis | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Header height (no stereotype) | 40 px (`UMLClassifier.nonStereotypeHeaderHeight`) | 40 px (`LAYOUT.DEFAULT_HEADER_HEIGHT`) | **PASS** | Constants match. |
| Header height (with stereotype) | 50 px (`UMLClassifier.stereotypeHeaderHeight`) | 50 px (`LAYOUT.DEFAULT_HEADER_HEIGHT_WITH_STEREOTYPE`) | **PASS** | |
| Attribute / method row height | 30 px (`UMLClassifierMember.bounds.height = computeDimension(1.0, 30)`) | 30 px (`LAYOUT.DEFAULT_ATTRIBUTE_HEIGHT` / `DEFAULT_METHOD_HEIGHT`) | **PASS** | |
| Header band fill | `<ThemedRect fillColor={element.fillColor}>` over full width × headerHeight | `<rect fill={fillColor}>` rendered by `HeaderSection` (full width × headerHeight) | **PASS** | |
| Stereotype label `«…»` | `<tspan ... dy={-8} fontSize="85%">` above name `<tspan dy={18}>` | `<tspan dy="-8" fontSize="85%">` above name `<tspan dy="18">` | **PASS** | dy and font size match. |
| Stereotype source | `element.stereotype: string \| null` — fully freeform (any string) | `data.stereotype?: ClassType` — enum-only (`Abstract` / `Interface` / `Enumeration`) | **GAP — high** | v3 lets `UMLAbstractClass` set `stereotype = 'abstract'`, `UMLInterface` set `'interface'`, `UMLEnumeration` set `'enumeration'`, **and** lets a user write any custom string. v4 hard-narrows to a 3-value enum, so any v3 model with a custom stereotype string round-trips to `undefined`. |
| Italic name | Driven by `element.italic` (boolean) — `<text fontStyle={element.italic ? 'italic' : undefined}>` (line 53–66 of `uml-classifier-component.tsx`) | Driven by `stereotype === ClassType.Abstract` only (line 66 of `HeaderSection.tsx`) | **GAP — high** | v3 stores italic as an independent flag on `UMLClassifier` and lets it be set via deserialization. v4 collapses the concept onto stereotype identity, so `italic = true` cannot be expressed without choosing the Abstract stereotype, and `italic = false` cannot be expressed *with* Abstract. |
| Underline name | `textDecoration={element.underline ? 'underline' : undefined}` on the name tspan AND on the no-stereotype `<Text>` wrapper | `HeaderSection` accepts an `isUnderlined` prop but `ClassSVG` never passes it (defaults to `false`); `ClassNodeProps` has no `underline` field at all | **GAP — high** | v3 supports underline on the class header (used by e.g. static class convention). v4 has the plumbing inside `HeaderSection` but `ClassNodeProps` does not declare the field, so it is unreachable from the data side. |
| Class name display | `name + (className ? ': ' + className : '')` for non-`UserModelName` types | `data.name` only — `className` field not present on `ClassNodeProps` | **GAP — medium** | v3 supports an instance/class binding label (used by Object-of-Class scenarios). v4 omits this on Class — it does keep it on `ObjectName`, but the symmetry on Class is lost. |
| Fill / stroke / text color | Per-element fields (`fillColor`, `strokeColor`, `textColor`) consumed by `ThemedRect` / `Text` | Same three fields, consumed by `getCustomColorsFromData` and forwarded to inner `<rect>` / `<CustomText>` | **PASS** | Fallbacks differ (`var(--besser-primary-contrast)` vs themed-component fallback) but behave equivalently. |
| Outer border | `<ThemedRect width="100%" height="100%" strokeColor={element.strokeColor}>` drawn **outside** clip path → always visible | `<StyledRect x=0 y=0 width=width height=height stroke={strokeColor}>` (line 72 of `ClassSVG.tsx`) | **PASS** | |
| Header / body clip path | `<clipPath id={clip-…}>` clips text overflow inside the box | None — overflow relies on `<svg overflow="visible">` and React Flow node bounds | **GAP — low** | v3 explicitly clips text to the bounds so a long attribute name does not bleed out of a small box. v4 has no clipPath, so text can leak past the right edge until the auto-grow effect kicks in (next render). Visible during in-progress resize. |
| Icon view | n/a for Class — old `UMLClassifierMemberComponent` only hides for `ObjectAttribute` / `ObjectMethod` / `UserModelAttribute` | n/a for Class — `shouldShowIconView` only consulted by `ObjectNameSVG` | **PASS** | Both versions ignore icon-view on Class. |

---

## 2. Layout — compartments, dividers, height growth

| Axis | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Attribute Y placement | `y = headerHeight + Σ attribute heights` (loop in `UMLClassifier.render`) | `RowBlockSection` placed at `offsetFromTop={headerHeight}`, internal `y = index * attributeHeight` | **PASS** | |
| Method Y placement | `y = deviderPosition + Σ method heights` (continues from attribute loop) | `RowBlockSection` placed at `offsetFromTop={headerHeight + attributes.length * methodHeight}` (line 125 of `ClassSVG.tsx`) | **GAP — high** | v4 multiplies `attributes.length` by `methodHeight` instead of `attributeHeight`. They are both 30 px today (so the bug is invisible), but the constants are independent — any future change to one drifts the other. The first separator line (line 116) uses the correct `attributeHeight`, so a regression on one constant produces a visible misalignment. |
| Header→attributes divider | `<ThemedPath>` only when `element.hasAttributes` | `<SeparationLine>` always rendered (`attributes.length >= 0` is always true, line 92) | **GAP — low** | When a class has zero attributes, v3 drew no divider after the header (matches UML spec — empty compartment = no compartment). v4 always draws it, leaving a thin line directly under the header even on bare classes. |
| Attributes→methods divider | `<ThemedPath>` only when `element.hasMethods && stereotype !== 'enumeration' && !isERClassifier` | `<SeparationLine>` always rendered (line 113–119) | **GAP — high** | v3 suppresses the methods divider for Enumeration (UML spec — enums have a literal compartment, not a methods compartment) and for ER-flavoured Class/AbstractClass (ER has no operations). v4 always draws it. Combined with the gap below, the v4 Enumeration variant displays a hollow methods compartment. |
| Methods compartment for Enumeration | `UMLEnumeration.render` skips the methods loop because the stereotype is `'enumeration'` and the type-check in `UMLClassifierMemberComponent` returns early for non-attribute children | `ClassSVG` has no `isEnumeration` branch — `processedMethods` is rendered unconditionally | **GAP — high** | This is a UML spec violation in v4. The popover at `ClassEditPanel.tsx:970` does hide the methods *editor* for Enumeration, so the user cannot add new methods, but any pre-existing method rows on an Enumeration node still draw on the canvas, and an empty methods compartment + divider is always painted. |
| ER notation — drops methods | `UMLClassifier.render` checks `settingsService.getClassNotation() === 'ER'` and skips the methods loop for `Class` / `AbstractClass`; member component returns `null` for class methods in ER | The new `formatRow` calls `formatDisplayName(…, classNotation)` for per-row text formatting only — there is no compartment-skip logic | **GAP — medium** | ER-mode Class/AbstractClass nodes in v4 still show the methods compartment (with the same divider bug as Enumeration). The brief lists ER notation as a v3 feature to preserve. |
| Box height grows with content | `this.bounds.height = y` after laying out all rows; layouter resizes the element | `useEffect` enforces `minHeight` (`calculateMinHeight(headerHeight, attrCount, methodCount, attrH, methodH)`); `NodeResizer` has `maxHeight={minHeight}` so the box is locked to exactly `minHeight` | **PASS** | Equivalent net effect — height ≡ content height. |
| Member text overflow | Clipped by `<clipPath>` if user resizes below text width; no horizontal scroll | No clip; `useEffect` enforces `width >= minWidth`, so overflow cannot occur but user cannot resize below text either | **PASS (different mechanism)** | v3 allowed the user to resize *smaller* than text (clipped). v4 forbids it (clamps `width` upward). Net visual outcome differs — see "behavior — resize" below. |

---

## 3. Behavior — rename, resize, drag-to-connect

| Axis | v3 (old) | v4 (new) | Status | Notes |
|---|---|---|---|---|
| Rename via double-click | Double-click opens an in-place editable text overlay (handled by `Editable` / `EditableText` in `EditableTextField`) on the header | Double-click triggers `setPopOverElementId(node.id)` (line 38 of `useElementInteractions.ts`) → opens `ClassEditPopover`, which carries the name field via `NodeStyleEditor` | **GAP — medium** | The function moves to a popover. v3 supports both (in-place + side panel); v4 only supports popover/inspector. Discoverability regresses — users who expect to type directly on the header don't get the in-place editor. |
| Resize handles | `static features.resizable: 'WIDTH'` — only horizontal resize, height is content-driven | `<NodeResizer minWidth={minWidth} minHeight={minHeight} maxHeight={minHeight}>` — width-only resize, height locked to content | **PASS** | Both effectively width-only. |
| Resize-below-text | Allowed in v3 — text is visually clipped but the user keeps the smaller box | Forbidden in v4 — `useEffect` snaps width back up to `minWidth` whenever it drops below | **GAP — low** | Subjective regression: v3 lets a power-user save real estate by accepting clipped labels; v4 always wins back the width. Likely intentional (cleaner default), but call it out. |
| Connect handles per side | `UMLClassifier` exposes connection points across the full perimeter (4 sides, multiple anchor points each) via the layouter | `DefaultNodeWrapper` renders 5 `<Handle>` elements per side (corner, mid-corner, center, mid-corner, corner) — 12 visible primary handles total around the perimeter | **PASS** | Drag-to-connect works from all four sides; visual handle layout differs but functionality matches. |
| `connectable` on Enumeration | `UMLEnumeration.features.connectable = false` — **enumerations cannot be relationship sources** in v3 | No equivalent disable — Enumeration variant still renders the same handle set as Class | **GAP — medium** | v3 deliberately blocks connecting Enumeration to anything (UML spec — enums are leaves). v4 lets you drag an association out of an Enumeration to any node. |

---

## 4. Stereotype variants — visual distinctness

| Stereotype | v3 visual signal | v4 visual signal | Status |
|---|---|---|---|
| `Class` (no stereotype) | No `«…»` line; 40 px header; name in regular weight | No `«…»` line; 40 px header; name in regular weight | **PASS** |
| `AbstractClass` | `«abstract»` tspan + name in italic (driven by `italic = true`) | `«Abstract»` tspan + name in italic (driven by `stereotype === ClassType.Abstract` switch in `HeaderSection.tsx:66`) | **PASS — case differs** | The displayed text reads `«Abstract»` (capital A) in v4 vs `«abstract»` in v3 — the enum value `ClassType.Abstract = "Abstract"` is the literal label rendered. Minor cosmetic regression. |
| `Interface` | `«interface»` tspan; name in regular weight; methods compartment kept | `«Interface»` tspan; name in regular weight; methods compartment kept | **PASS — case differs** | Same case-mismatch as Abstract. |
| `Enumeration` | `«enumeration»` tspan; **no methods compartment, no divider after attributes**; `connectable = false` | `«Enumeration»` tspan; **methods compartment + divider always present**; `connectable` not disabled | **GAP — high** | Visual distinctness lost — a v4 Enumeration with stale methods data looks identical to a Class. See §2 row 4 for spec rationale. |

---

## 5. Hidden flags — italic / underline / freeform stereotype

| Flag | v3 (old) | v4 (new) | Status |
|---|---|---|---|
| `italic` (independent of stereotype) | `IUMLClassifier.italic: boolean` — settable per-element, persisted, deserialized | Not present on `ClassNodeProps` — italic only via `ClassType.Abstract` enum value | **GAP — high** |
| `underline` | `IUMLClassifier.underline: boolean` — same shape as italic | Not present on `ClassNodeProps`. `HeaderSection` *accepts* an `isUnderlined` prop but `ClassSVG` never threads anything to it | **GAP — high** |
| Freeform stereotype | `stereotype: string \| null` — any string accepted | `stereotype?: ClassType` — enum-only | **GAP — high** |
| `description` / `uri` / `icon` | All present in v3 element fields and rendered when `StylePane.show*` flags set | Not present on `ClassNodeProps` | **GAP — medium** (also flagged in PC-2) |

---

## Severity-tagged gap list

### High (block parity)
1. **Freeform stereotype lost.** v3 `stereotype: string \| null` → v4 `stereotype?: ClassType` enum. v3 models with custom stereotype text round-trip to `undefined`. Either widen the type back to `string` and look up `ClassType` only for the italic/spec-suppression paths, or document the regression. (`Class.tsx:68`, `NodeProps.ts:88`)
2. **`italic` flag dropped.** Header italic is now derived solely from `stereotype === ClassType.Abstract`. v3 lets italic be true without `«abstract»` (and false with). Re-add `italic?: boolean` on `ClassNodeProps` and forward it to `HeaderSection` independently of stereotype identity. (`HeaderSection.tsx:66`)
3. **`underline` flag unreachable.** `HeaderSection` accepts `isUnderlined` but the data type does not declare it and `ClassSVG` never passes it. Add `underline?: boolean` to `ClassNodeProps` and thread it through. (`HeaderSection.tsx:23`, `ClassSVG.tsx:80–89`)
4. **Enumeration draws a methods compartment and divider.** UML spec violation. Branch on `stereotype === ClassType.Enumeration` (and on ER-mode Class/AbstractClass) to skip both the second `SeparationLine` and the methods `RowBlockSection`. (`ClassSVG.tsx:112–130`)
5. **Method-row Y offset uses `methodHeight` where it should use `attributeHeight`.** Hidden today because both constants equal 30 px, but the divider above (line 116) uses the correct constant — drifting either constant produces a visible misalignment. (`ClassSVG.tsx:125`)

### Medium
6. **ER-mode method drop missing.** `UMLClassifier.render` skipped the methods compartment for Class/AbstractClass when `settingsService.getClassNotation() === 'ER'`. v4's `formatRow` only re-formats per-row strings; the compartment still renders. (`Class.tsx:67–79`, no compartment-level branch)
7. **Enumeration is connectable.** v3 set `UMLEnumeration.features.connectable = false`. v4 has no equivalent — drag-to-connect works from Enumeration nodes. Either gate `<Handle isConnectableStart>` on `stereotype !== ClassType.Enumeration` or drop the handle markup entirely on Enumerations. (`DefaultNodeWrapper.tsx`, no Class-specific gate)
8. **In-place rename via double-click removed.** v4 routes double-click to the popover instead of the in-place editor. Function preserved (popover has the name field) but discoverability regresses. (`useElementInteractions.ts:34–39`)
9. **`description` / `uri` / `icon` / `className` fields not exposed on Class.** Mirrors the PC-2 finding — applies to both the inspector and the canvas-side rendering. (`NodeProps.ts:85–95`)

### Low
10. **Header→attributes divider drawn for empty attribute lists.** v3 hid the divider when `hasAttributes === false`. v4 always draws it. (`ClassSVG.tsx:92–99`)
11. **Stereotype label case differs.** `«Abstract»` / `«Interface»` / `«Enumeration»` (v4 enum-value capitalisation) vs `«abstract»` / `«interface»` / `«enumeration»` (v3 lowercase strings). Cosmetic but visible. Consider lowercasing in the `<tspan>` render. (`HeaderSection.tsx:48`)

---

## Files reviewed (absolute paths)

- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/classDiagram/Class.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/svgs/nodes/classDiagram/ClassSVG.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/svgs/nodes/HeaderSection.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/svgs/nodes/RowBlockSection.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/types/nodes/NodeProps.ts`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/types/nodes/enums/ClassType.ts`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/initialElements.ts`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/wrappers/DefaultNodeWrapper.tsx`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/useElementInteractions.ts`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/layoutUtils.ts`
- `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/library/lib/constants.ts`
- v3 (read via `git show 34b390c:…`):
  - `…/packages/editor/src/main/packages/uml-class-diagram/uml-class/uml-class.ts`
  - `…/packages/editor/src/main/packages/uml-class-diagram/uml-abstract-class/uml-abstract-class.ts`
  - `…/packages/editor/src/main/packages/uml-class-diagram/uml-interface/uml-interface.ts`
  - `…/packages/editor/src/main/packages/uml-class-diagram/uml-enumeration/uml-enumeration.ts`
  - `…/packages/editor/src/main/packages/common/uml-classifier/uml-classifier.ts`
  - `…/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-component.tsx`
  - `…/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-member-component.tsx`
  - `…/packages/editor/src/main/packages/common/uml-classifier/uml-classifier-member.ts`
  - `…/packages/editor/src/main/packages/uml-class-diagram/class-preview.ts`
