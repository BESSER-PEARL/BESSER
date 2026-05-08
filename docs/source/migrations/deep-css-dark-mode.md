# Deep Dive: CSS Dark Mode in `packages/library`

Audit of dark-mode coverage in the new `@besser/library` package, complementing
SA-DEEP-3's note that `packages/library/lib/styles/` ships **no `.dark` cascade
rules** — the entire theme switch relies on JS `setProperty` from
`webapp/.../theme-switcher.ts:14` mutating CSS variables on `document.documentElement`.

> Source paths in this doc are rooted at
> `besser/utilities/web_modeling_editor/frontend/packages/library/lib/`.

---

## 1. Theme Switching Mechanism (recap)

**Single entry point:** `packages/webapp/src/main/shared/utils/theme-switcher.ts`

```ts
for (const [themingVar, value] of Object.entries(selectedTheme)) {
  root.style.setProperty(themingVar, value);            // line 14
}
root.setAttribute('data-theme', theming);               // line 18
root.classList.toggle('dark', theming === 'dark');      // line 19
```

Source of truth: `packages/webapp/src/main/themings.json` — 21 vars defined for
both `light` and `dark`.

The library ships **0 `.dark` cascades** and **0 `prefers-color-scheme` queries**
(verified — `grep -rEn '\.dark|prefers-color-scheme'` in `packages/library`
returns nothing). Dark mode therefore depends entirely on:
1. Each `--besser-*` variable being defined in `themings.json`
2. Each component reading it through `var(--besser-*)`
3. No hardcoded colors bypassing the variable system

---

## 2. Variable Coverage Table

Library uses **231** `var(--besser-*)` references across 16 distinct variables.
Of those 16, only **12** are defined in `themings.json`. The remaining 4 fall
back to their hardcoded literals in **all themes**.

| Variable | Light value | Dark value | Defined in `themings.json`? | Used in library |
|----------|-------------|-----------|--------------|-----------|
| `--besser-primary` | `#3e8acc` | `#3e8acc` *(unchanged)* | yes | hover/selected outlines, edge-overlay, scroll-overlay-hint, ButtonGroup borders |
| `--besser-primary-contrast` | `#212529` | `white` | yes | strokeColor/textColor default; **most SVG `stroke=` and text `fill=`** |
| `--besser-secondary` | `#6c757d` | `#888` | yes | CustomControls disabled icon |
| `--besser-background` | `white` | `#0f172a` | yes | fillColor default; HeaderSection band, Sidebar bg, MiniMap bg, Panel bg |
| `--besser-background-variant` | `#f8f9fa` | `#303530` | yes | Panel shadow, control-button hover |
| `--besser-background-inverse` | `#000000` | `#ffffff` | yes | unused in library |
| `--besser-gray` | `#e9ecef` | `#242724` | yes | CustomBackground dot color, inspector divider |
| `--besser-gray-variant` | `#495057` | `#f8f9fa` | yes | control-button border, inspector label |
| `--besser-grid` | `rgba(36,39,36,0.1)` | `#303530` | yes | scroll-overlay backdrop, Background grid |
| `--besser-interactive-selection` | *(not in themings)* | *(not in themings)* | **no** (fallback `#f39c12`) | exam selection halo |
| `--besser-guide-vertical` | *(not in themings)* | *(not in themings)* | **no** (fallback `#d63031`) | alignmentGuides.css |
| `--besser-guide-horizontal` | *(not in themings)* | *(not in themings)* | **no** (fallback `#0984e3`) | alignmentGuides.css |
| `--besser-warning-yellow` | *(not in themings)* | *(not in themings)* | **no** (fallback `#ffc800`) | theme/styles.tsx, styles/theme.ts |
| `--besser-text` | *(not in themings)* | *(not in themings)* | **no** (fallback `#000`) | UserModelNameEditPanel.tsx:194, :267 |
| `--besser-text-muted` | *(not in themings)* | *(not in themings)* | **no** (fallback `#6c757d`) | ObjectEditPanel.tsx:95 |
| `--besser-gray-700` | *(not in themings)* | *(not in themings)* | **no** (no fallback → resolves to *initial*) | ObjectLinkEditPanel.tsx:175 |

**Coverage: 12 / 16 = 75%** of library-referenced vars switch on theme toggle.
The remaining 4 (`--besser-text`, `--besser-text-muted`, `--besser-warning-yellow`,
`--besser-gray-700`) are **dead variables** — they were referenced during the v3→v4
port but never wired into `themings.json`, so they always render the fallback or
nothing.

Note: `themings.json` defines 9 additional vars not used by library at all
(`--besser-alert-*`, `--besser-switch-box-*`, `--besser-list-group-color`,
`--besser-btn-outline-secondary-color`, `--besser-modal-bottom-border`) —
these are webapp-only.

---

## 3. Hardcoded Colors That Bypass the Variable System

These literals do **not** flip in dark mode regardless of `setProperty` calls.
Listed file:line followed by the hardcoded value.

### Critical (visible on canvas, every light/dark switch)

| File:Line | Hardcoded color | Context |
|-----------|----------------|---------|
| `nodes/agentDiagram/AgentRagElement.tsx:54` | `#E8F0FF` | sticky-blue cylinder fill (when `fillColor==='white'`) |
| `nodes/agentDiagram/AgentRagElement.tsx:55` | `#668` | cylinder default stroke (when no `strokeColor`) |
| `nodes/agentDiagram/AgentIntent.tsx:48` | `#E3F9E5` | sticky-green intent shape fill |
| `nodes/nnDiagram/NNContainer.tsx:39` | `#F5F5F5` | NN container body fill |
| `nodes/nnDiagram/NNReference.tsx:31` | `#FFFDE7` | reference card fill |
| `nodes/classDiagram/ClassOCLConstraint.tsx:106-108` | `#fff8c4`, `#bda21f`, `#3a2e00` | OCL sticky-note fill / stroke / text |
| `nodes/common/Comment.tsx:88-90` | `#fff8c4`, `#bda21f`, `#3a2e00` | free-form Comment sticky-note |
| `nodes/stateMachineDiagram/StateCodeBlock.tsx:87` | `#fff` | header band language label text |
| `nodes/stateMachineDiagram/StateCodeBlock.tsx:111` | `#000` | code body text fallback when `textColor` empty |
| `components/svgs/nodes/stateMachineDiagram/StateMachineSVGs.tsx:104` | `white` | sub-state inner shape fill |
| `components/svgs/nodes/stateMachineDiagram/StateMachineSVGs.tsx:313` | `#fff` | StateCodeBlockSVG language text |
| `components/svgs/nodes/agentDiagram/AgentDiagramSVGs.tsx:68` | `#E3F9E5` | Intent palette preview |
| `components/svgs/nodes/agentDiagram/AgentDiagramSVGs.tsx:110,118,126` | `#E8F0FF` (×3) | RagElement palette preview rect + ellipses |
| `components/svgs/nodes/agentDiagram/AgentDiagramSVGs.tsx:111,119,127` | `#668` (×3) | RagElement palette stroke |
| `components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx:237` | `#F5F5F5` | NNContainer palette body |
| `components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx:265,274` | `white` | NNContainer palette layer rects |
| `components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx:266,275` | `#999` | NNContainer palette layer stroke |
| `components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx:303` | `#FFFDE7` | NNReference palette body |
| `components/svgs/AssessmentIcon.tsx:43-44` | `#f0f0f0`, `#ccc` | "no-score yet" assessment chip |

### Edges — every diagram type

The label-bg padding rect on every edge type is `fill="lightgray"`:
- `edges/edgeTypes/{ClassDiagramEdge, ObjectDiagramEdge, StateMachineDiagramEdge, ActivityDiagramEdge, AgentDiagramEdge, AgentDiagramInitEdge, BPMNDiagramEdge, ComponentDiagramEdge, CommunicationDiagramEdge, DeploymentDiagramEdge, FlowChartEdge, NNNext, ReachabilityGraphArc, SfcDiagramEdge}.tsx`
- 14 files, single `fill="lightgray"` per edge label box (~line 156–236).

### Inspectors / popovers (less critical but visible)

| File:Line | Hardcoded color |
|-----------|-----------------|
| `components/popovers/SeeFeedbackAssessmentBox.tsx:44` | `#ccc` |
| `components/popovers/GiveFeedbackAssessmentBox.tsx:123` | `#ccc` |
| `components/popovers/deploymentDiagram/DeploymentNodeEditPopover.tsx:84` | `#fff` (input bg) |
| `components/AssessmentSelectionDebug.tsx:33,99` | `rgba(255,255,255,0.9)`, `#666` |
| `components/debug/AssessmentSelectionDebug.tsx:36,43` | `white`, `0 2px 10px rgba(0,0,0,0.1)` |
| `components/AssessmentSelectableElement.tsx:113-115` | `rgba(25,118,210,0.2)` etc. (assessment overlay) |
| `components/wrapper/AssessmentSelectableWrapper.tsx:112-130` | same pattern |
| `components/propertiesPanel/PropertiesPanel.tsx:139` | shadow `rgba(0,0,0,0.06)` |
| `styles/inspector-theme.ts:55-67` | MUI palette: `#2a8fbd`, `#0f172a`, `#495057`, `#ffffff`, `#e9ecef` (literals required by MUI's `decomposeColor`, **only** the styleOverride blocks below use `var()`) |

### MiniMap

| File:Line | Hardcoded color |
|-----------|-----------------|
| `components/CustomMiniMap.tsx:636` | `fill="gray"` for unknown-shape minimap node |

**Hardcoded color count (raw `fill=`/`stroke=`/`color:`/`backgroundColor:` literals): ~75** across the library, ~50 of which are visually significant on the dark canvas (the rest are debug overlays or assessment chrome rarely visible).

---

## 4. The "white-sentinel" Anti-Pattern

`utils/layoutUtils.ts:47-52` — `getCustomColorsFromData`:

```ts
const fillColor  = data.fillColor  || "var(--besser-background)"
const strokeColor = data.strokeColor || "var(--besser-primary-contrast)"
const textColor   = data.textColor   || "var(--besser-primary-contrast)"
```

That returns the `var(...)` **string**, not the resolved color. Five components
then test the returned value against the literal `"white"`:

- `nodes/agentDiagram/AgentRagElement.tsx:54`
- `nodes/agentDiagram/AgentIntent.tsx:48`
- `nodes/nnDiagram/NNContainer.tsx:39`
- `nodes/nnDiagram/NNReference.tsx:31`
- `nodes/nnDiagram/_NNLayerBase.tsx:160`

**The check `fillColor === "white"` is never true** — `getCustomColorsFromData`
returns `"var(--besser-background)"`. So the override branch (which would apply
`#E8F0FF` / `#F5F5F5` / `#FFFDE7` / `#E3F9E5`) only fires when the user has
explicitly stored `fillColor: "white"` on the node data — which is the exact
v3 sentinel value. New nodes created in v4 won't carry it, so:

1. **In light mode**: new RagElement/Intent/NNContainer/NNReference render
   white instead of their sticky-color palette, breaking the visual identity.
2. **In dark mode**: they render `#0f172a` (very dark slate) and fail badly
   because the cylinder/intent stroke (`#668` hardcoded) collapses into the bg.

The palette-preview SVGs in `components/svgs/nodes/{agent,nn}Diagram/` *don't*
use the sentinel — they pin the literals directly (`fill="#E3F9E5"` etc.).
That's why **the sidebar palette stays correctly tinted in light mode but
clashes in dark mode**, while the canvas itself silently renders the new
nodes white-on-white in light and dark-on-dark in dark.

---

## 5. Component-by-Component Dark-Mode Status

### a. AgentRagElement (cylinder)
- Body fill: `#E8F0FF` literal — does **not** flip. On `--besser-background = #0f172a` (dark), it stays light blue → high contrast that's actually visible. **Acceptable but unintentional.**
- Body stroke: `#668` literal — does **not** flip. **Hard to see on light blue body either way.**

### b. AgentIntent (pentagon)
- Body fill: `#E3F9E5` literal → still light green in dark mode. **Visible but inconsistent.**
- Stroke: `var(--besser-primary-contrast, #000)` → flips correctly to white in dark, but white-on-light-green clashes badly. **Regression.**

### c. NNContainer
- Body fill: `#F5F5F5` literal — barely lighter than light bg in light mode, *light grey island* in dark mode. **Visible but accidental.**
- Stroke: themed (correct).
- Title text: themed (correct).

### d. NNReference
- Body fill: `#FFFDE7` literal — pale yellow. **Always pale yellow regardless of theme** → bright in dark mode, hard to read text on it.
- Stroke: themed.

### e. ClassOCLConstraint sticky-note
- Hard literals at lines 106–108 (`#fff8c4`, `#bda21f`, `#3a2e00`). **Sticky stays yellow with brown text** — ironically the most readable element in dark mode, but also the most visually jarring (looks like a Post-it pasted on a black canvas).

### f. Free-form Comment node
- Same hardcoded sticky palette as OCLConstraint. Same behavior.

### g. ObjectName underline
- Header `<rect>` filled with `var(--besser-background)` → flips correctly.
- Underline rendered via `textDecoration="underline"` on themed `<CustomText>` (whose fill defaults to `var(--besser-primary-contrast)`). **Themed correctly.**

### h. HeaderSection / stereotype band
- Rect fill: `var(--besser-background, white)` → themed.
- Text fill: themed via prop.
- **No regression.**

### i. CodeMirror Python panes
- The library does **not** use CodeMirror — `find … codemirror` returned no library matches. The `StateCodeBlock` is a **plain `<foreignObject>` of `<div>`s**, not a CodeMirror instance.
- Container: `fillColor` from `getCustomColorsFromData` (themes).
- Header band: `fill={strokeColor}` (themes — usually black/white).
- Header label text: `fill="#fff"` **literal** → in light mode reads white text on near-black header (correct). In dark mode the strokeColor flips to white, so label is white-on-white → **invisible**. **Major regression.**
- Code body text: `color: textColor || "#000"` — themed when present, hardcoded fallback otherwise.

### j. Edge labels
- Background pad rect `fill="lightgray"` × 14 edge types. Light grey reads OK on white but creates a glaring band on the `#0f172a` dark canvas. **Consistent regression across every diagram type.**

---

## 6. Manual Toggle Test (theoretical, since no runtime here)

Tracing `setProperty(--besser-background, '#0f172a')` through the renderer:

| Element | Light value | After dark toggle | Verdict |
|---------|-------------|------------------|---------|
| Editor canvas bg | `white` | `#0f172a` | OK |
| Class node body | `var(--besser-background)` → `white` | `#0f172a` | OK (text flips too) |
| Class node text | `var(--besser-primary-contrast)` → `#212529` | `white` | OK |
| Class node stroke | `var(--besser-primary-contrast)` | `white` | OK |
| Edge stroke (xy-edge-stroke chain) | `--xy-edge-stroke` ← `--besser-primary-contrast` | white | OK |
| Edge label bg pad | `lightgray` literal | `lightgray` | **FAIL** |
| StateCodeBlock header text | `#fff` literal | `#fff` (and header is now `white`) | **FAIL — invisible** |
| AgentRagElement body | `#E8F0FF` literal | `#E8F0FF` | technically visible but jarring |
| AgentIntent body | `#E3F9E5` literal | `#E3F9E5` | jarring |
| NNContainer body | `#F5F5F5` literal | `#F5F5F5` | jarring |
| OCL sticky-note | `#fff8c4` literal | `#fff8c4` | jarring (intentional? sticky-note should "always look yellow") |
| Comment sticky | same | same | jarring (same rationale) |
| MiniMap unknown-shape node | `gray` | `gray` | visible both modes |
| Inspector panel labels | `var(--besser-primary-contrast)` | white | OK |
| Inspector "muted" text | `var(--besser-text-muted, #6c757d)` | `#6c757d` (var never set) | **FAIL — never flips** |
| Inspector "linked" text | `var(--besser-text)` | unset (no fallback) | **FAIL — undefined behavior** |
| Inspector caption | `var(--besser-gray-700)` | unset | **FAIL — undefined behavior** |
| Alignment guides (red/blue) | `#d63031` / `#0984e3` literals via undefined var | same | OK (intentional accent) |

---

## 7. Top 5 Dark-Mode Regressions

Ranked by visibility on a typical multi-diagram canvas:

1. **StateCodeBlock language label invisible in dark mode** — `nodes/stateMachineDiagram/StateCodeBlock.tsx:87` writes `fill="#fff"` over a header whose color is themed via `strokeColor` (white in dark) → white-on-white. Same bug in `components/svgs/nodes/stateMachineDiagram/StateMachineSVGs.tsx:313` for the palette preview.

2. **Edge label backgrounds glare on dark canvas** — `fill="lightgray"` on 14 distinct edge types (one each in `edges/edgeTypes/*.tsx`). Every diagram type ships this regression.

3. **The "white sentinel" never matches** in `getCustomColorsFromData` consumers — newly-created RagElement / Intent / NNContainer / NNReference / NN layer nodes lose their v3 sticky palette in *light* mode (white-on-white) and in *dark* mode resolve to `var(--besser-background) = #0f172a` for the body with hardcoded `#668` stroke that practically vanishes. Affects `nodes/agentDiagram/AgentRagElement.tsx:54`, `AgentIntent.tsx:48`, `nodes/nnDiagram/NNContainer.tsx:39`, `NNReference.tsx:31`, `_NNLayerBase.tsx:160`.

4. **Inspector "muted text" / "linked text" / caption columns never flip** — `--besser-text`, `--besser-text-muted`, `--besser-gray-700` referenced at `components/inspectors/userDiagram/UserModelNameEditPanel.tsx:194,267`, `components/inspectors/objectDiagram/ObjectEditPanel.tsx:95`, `ObjectLinkEditPanel.tsx:175` are **never declared** in `themings.json`. In dark mode the user/object edit panels show light-grey-on-dark labels at low contrast, and the linked-attribute display falls through to the browser default (likely `currentColor` ≈ white) producing inconsistent rendering between rows.

5. **Sticky-note family stays bright yellow on dark canvas** — `ClassOCLConstraint.tsx:106-108` and `Comment.tsx:88-90` ship the literal Post-it palette (`#fff8c4` / `#bda21f` / `#3a2e00`). This is *probably* intentional ("a sticky note should always look like a sticky note"), but it has not been documented as a design decision and the contrast against a `#0f172a` canvas is jarring without a darker yellow alternative.

---

## 8. Recommendations (out of scope, for tracking)

1. Add `--besser-text`, `--besser-text-muted`, `--besser-warning-yellow`, `--besser-gray-700`, `--besser-interactive-selection`, `--besser-guide-vertical`, `--besser-guide-horizontal` to `themings.json` (both modes).
2. Replace the `fillColor === "white"` sentinel with `fillColor || DEFAULT_FILL` and stop returning `var()` strings from `getCustomColorsFromData`.
3. Replace `fill="lightgray"` on edge labels with `var(--besser-background, white)` (matches HeaderSection pattern).
4. Replace `fill="#fff"` on StateCodeBlock header text with `var(--besser-background, #fff)` (header bg is `strokeColor`, so background-tinted text gives correct contrast in both modes).
5. Add a `--besser-sticky-{bg,border,text}` token family for OCL/Comment notes with a darker variant for dark mode (e.g. `#3d2f00` background + `#ffe98a` text), or document the deliberate "always yellow" rule.
