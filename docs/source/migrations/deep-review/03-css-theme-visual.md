# Deep Review 03 — CSS / Theme / Visual

Read-only audit of the SA-DEBRAND `--apollon-*` → `--besser-*` rename and
the React-Flow rewrite. Branch: `claude/refine-local-plan-sS9Zv`.
Submodule HEAD inspected: `feature/migration-react-flow @ 173d620`.

## A. CSS variable consistency

Inventory was taken across `packages/library/lib/` and
`packages/webapp/src/` (the editor package is legacy and not part of v4).

**Variable declarations** are concentrated in two static maps:

- `packages/webapp/src/main/themings.json` — runtime theme map injected
  via `shared/utils/theme-switcher.ts:14` (`root.style.setProperty`),
  not a CSS rule. Two themes: `light` and `dark`. 19 keys each.
- `packages/library/lib/constants.ts:119-141` (`CSS_VARIABLE_FALLBACKS`)
  — fallback constants, **not** declarations on a selector. Used only
  to seed JS string literals.

The only real CSS-rule declaration is in `packages/library/lib/styles/app.css:187`:

```css
.besser-interactive-selection {
  --besser-interactive-selection-color: var(--besser-interactive-selection, #f39c12);
}
```

So in CSS-rule terms there is exactly **1 declaration**. In the wider
"variable surface" sense (themings.json + fallbacks), the count is 19.

- **Total `--besser-*` declarations (themings.json keys):** 19
- **Total CSS-rule declarations:** 1
- **Total consumers (`var(--besser-*` references):** 239 references / 14 unique names
- **Light/dark coverage:** 100% (themings.json defines all 19 keys for both `light` and `dark`)

**Unique consumed names:**

```
--besser-background
--besser-background-variant
--besser-gray
--besser-gray-700
--besser-gray-variant
--besser-grid
--besser-guide-horizontal
--besser-guide-vertical
--besser-interactive-selection
--besser-interactive-selection-color
--besser-primary
--besser-primary-contrast
--besser-secondary
--besser-warning-yellow
```

### Missing declarations (consumed but not declared in either themings.json or CSS_VARIABLE_FALLBACKS)

| Variable                            | Sites                                                                                                | Has fallback? | Severity |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------- | -------- |
| `--besser-gray-700`                 | `packages/library/lib/components/inspectors/objectDiagram/ObjectLinkEditPanel.tsx:175`               | ❌ no fallback | High — text colour resolves to browser default |
| `--besser-guide-horizontal`         | `packages/library/lib/styles/alignmentGuides.css:26`                                                 | ✅ #0984e3    | Low — alignment guide tinted via fallback |
| `--besser-guide-vertical`           | `packages/library/lib/styles/alignmentGuides.css:22`                                                 | ✅ #d63031    | Low — alignment guide tinted via fallback |
| `--besser-warning-yellow`           | `packages/library/lib/styles/theme.ts:13`, `packages/library/lib/components/theme/styles.tsx:5`      | ✅ #ffc800    | Medium — theme exposes `warningYellow` token but the runtime variable is named `--besser-alert-warning-yellow` (mismatch) |

The `--besser-warning-yellow` ↔ `--besser-alert-warning-yellow` mismatch
is the most consequential: theming.json declares the latter, the theme
object reads the former. Dark-mode warning surfaces will silently fall
back to the v3 default `#ffc800` instead of switching to the light-mode
`#ffc107` / dark-mode `#ffffff` declared in themings.json.

### Dead variables (declared in themings.json but never consumed)

11 of 19 declared theme keys have **zero consumers**:

```
--besser-alert-warning-yellow         (note: not --besser-warning-yellow)
--besser-alert-warning-background
--besser-alert-warning-border
--besser-background-inverse
--besser-alert-danger-color
--besser-alert-danger-background
--besser-alert-danger-border
--besser-switch-box-border-color
--besser-list-group-color
--besser-btn-outline-secondary-color
--besser-modal-bottom-border
```

Note `--besser-alert-warning-yellow` is shipped in themings.json but
the consumers ask for `--besser-warning-yellow` (no `alert-` prefix) —
a debrand rename inconsistency.

The 8 actually-consumed declarations: `primary`, `primary-contrast`,
`secondary`, `background`, `background-variant`, `gray`, `gray-variant`,
`grid`, `interactive-selection` (9 of 19 — count includes the implicit
selection variant).

## B. Theme parity vs v3

v3 theme: `packages/editor/src/main/components/theme/styles.ts:15-38`
(`apollonTheme`, styled-components).

v4 theme: `packages/library/lib/styles/theme.ts:9-30` (`besserTheme`,
plus `packages/library/lib/components/theme/styles.tsx`).

| v3 token (apollon)        | v3 default value             | v4 variable (besser)              | v4 default value | Same? |
| ------------------------- | ---------------------------- | --------------------------------- | ---------------- | ----- |
| `--apollon-primary`       | `#2a8fbd`                    | `--besser-primary`                | `#2a8fbd` (theme.ts) / `#3e8acc` (constants.ts/themings.json) | ⚠️ Drift between fallback maps |
| `--apollon-primary-contrast` | `#212529`                 | `--besser-primary-contrast`       | `#000000` (constants.ts) / `#212529` (themings.json light) | ⚠️ Drift |
| `--apollon-secondary`     | `#6c757d`                    | `--besser-secondary`              | `#6c757d`        | ✅    |
| `--apollon-warning-yellow`| `#ffc800`                    | `--besser-warning-yellow`         | `#ffc800` (theme.ts only) | ⚠️ No runtime declaration |
| `--apollon-background`    | `#f8f9fb`                    | `--besser-background`             | `#ffffff` (constants.ts) / `white` (themings.json) | ❌ Lighter (was off-white, now pure white) |
| `--apollon-background-variant` | `#e5e5e5`               | `--besser-background-variant`     | `#f8f9fa`        | ❌ Different shade |
| `--apollon-grid`          | `rgba(0, 0, 0, 0.08)`        | `--besser-grid`                   | `rgba(36, 39, 36, 0.1)` | ❌ Different |
| `--apollon-grid-minor`    | `rgba(0, 0, 0, 0.2)`         | (none)                            | —                | ❌ Dropped |
| `--apollon-gray`          | `#e9ecef`                    | `--besser-gray`                   | `#e9ecef`        | ✅    |
| `--apollon-graylight`     | `rgb(170, 170, 170)`         | (none)                            | —                | ❌ Dropped |
| `--apollon-gray-variant`  | `#343a40`                    | `--besser-gray-variant`           | `#495057`        | ❌ Different shade |

Three v3 tokens have no v4 equivalent: `grid-minor`, `graylight`. v3's
`graylight` was used in subtle dividers; consumers in v4 fall back to
`gray-variant` / `gray`.

The `--besser-primary` value drifts between three sources:

- `packages/library/lib/styles/theme.ts:11`            → `#2a8fbd`
- `packages/library/lib/components/theme/styles.tsx:3` → `#2a8fbd`
- `packages/library/lib/constants.ts:121`              → `#3e8acc`
- `packages/webapp/src/main/themings.json:3,24`        → `#3e8acc`

theme.ts is the dead path (TODO: "These values are not used yet"). The
runtime value is `#3e8acc` from themings.json. The fallback baked into
SVG attributes via `STROKE_COLOR` / `FILL_COLOR` is `#000000` / `#ffffff`
respectively — so when the user opens an SVG export in a tool that
strips CSS variables, the result loses the brand blue and renders pure
black-on-white. That matches v3 behaviour but isn't perfect.

## C. Class-based dark mode

`packages/webapp/.../theme-switcher.ts:18-19` toggles both
`<html data-theme="dark">` AND `<html class="dark">` (the latter for
Tailwind dark variants on the webapp shell).

The library does **not** read either flag. There are zero `.dark`
selectors in `packages/library/lib/styles/`. Dark-mode switching works
only because `theme-switcher.ts` rewrites every variable via
`root.style.setProperty(name, value)` at runtime. This is the v3 fork's
mechanism preserved verbatim.

Spot-check of the 5 critical variables (light → dark per
`themings.json`):

| Variable                  | Light        | Dark        | Override mechanism present? |
| ------------------------- | ------------ | ----------- | --------------------------- |
| `--besser-background`     | `white`      | `#0f172a`   | ✅ JS reassignment via theme-switcher |
| `--besser-primary-contrast` (foreground) | `#212529` | `white` | ✅ JS reassignment |
| `--besser-primary`        | `#3e8acc`    | `#3e8acc`   | ✅ JS reassignment (same value, intentional) |
| `--besser-gray-variant` (accent) | `#495057` | `#f8f9fa` | ✅ JS reassignment |
| `--besser-modal-bottom-border` (border-ish) | `#e9ecef` | `#979797` | ⚠️ Declared but not consumed |

**Implication**: If a future consumer relies on a CSS-only `.dark` cascade
(eg. ports a Tailwind component that uses `.dark:bg-…`), the library
will not respond. All theme-aware library styles must keep going through
`var(--besser-*)`.

## D. Per-diagram spot checks (BESSER-relevant nodes)

Defaults sourced from
`packages/library/lib/constants.ts:376-1100` (palette `width`/`height`
seed values for `defaultDropElementConfigs`). v3 sources noted inline
when relevant.

| Diagram | Node | Width | Height | Border | Fill | Header / divider | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ClassDiagram | `Class` (default) | 160 | 100 | solid stroke (`StyledRect`, `LAYOUT.LINE_WIDTH=2`) | `var(--besser-background, #fff)` | `HeaderSection` + `SeparationLine` after each section | ⚠️ width regressed from v3's 200 → 160 |
| ClassDiagram | `Abstract` (stereotype) | 160 | 110 | solid | as above | header gets stereotype tspan + name (italic via `ClassType.Abstract`) | ⚠️ width regression same as above |
| ClassDiagram | `Enumeration` | 160 | 140 | solid | as above | NO methods compartment (PC-1 fix applied at `ClassSVG.tsx:39`) | ✅ correct (literal compartment only) |
| ObjectDiagram | `ObjectName` (default) | 160 | 70 | solid | as above | underlined name in header, no stereotype | ⚠️ width regressed from v3's 200 → 160; height matches |
| ObjectDiagram | `ObjectName` w/ icon view | 160 | 70+ | solid | as above | header + foreignObject icon | ✅ correct (SA-2.1 — feature parity) |
| ObjectDiagram | `ObjectLink` edge | n/a | n/a | n/a | n/a | n/a | ⚠️ inspector references `--besser-gray-700` (undeclared) |
| StateMachineDiagram | `State` | 160 | 100 | solid, **rx=8** | `var(--besser-background)` (note: v3 had no fill on the outer rect, drew sub-fills) | header divider drawn unconditionally at `headerHeight` | ⚠️ v3 drew divider only when `hasBody` — v4 always draws it |
| StateMachineDiagram | `StateInitialNode` | 45 | 45 | none | `var(--besser-primary-contrast)` (overrides theme to ensure visibility) | n/a | ✅ correct (SA-UX-FIX-2 B2 — solid disc, ignores theme) |
| StateMachineDiagram | `StateFinalNode` | 45 | 45 | outer ring stroke 1.5px | white outer / primary-contrast inner | radii 0.9 / 0.7 of half-min | ⚠️ stroke 1.5px — v3 used theme default 1px |
| AgentDiagram | `AgentState` | 160 | 100 | solid, rx=8 | `getCustomColorsFromData` → background var | header divider when bodies exist + dashed fallback divider | ✅ correct (SA-FIX-Agent inline rows match v3) |
| AgentDiagram | `AgentIntent` | 160 | 100 | solid 1.2px | `#E3F9E5` (light-green tint, hardcoded) | folded-corner path + header | ⚠️ stroke 1.2px deviates from theme `LINE_WIDTH=2`; fill colour is hardcoded — won't change in dark mode |
| AgentDiagram | `AgentRagElement` | 160 | 120 | (file not inspected here) | — | — | not spot-checked |
| UserDiagram | `UserModelName` | 160 | 100 | reuses `ObjectNameSVG` | reuses ObjectNameSVG | header + attribute rows | ✅ delegates to ObjectNameSVG; same width regression as ObjectName |
| UserDiagram | `UserModelIcon` | 50 | 50 | none | from data | foreignObject for inline SVG / `<image>` for data URLs | ✅ correct |
| UserDiagram | per-class palette entries | 160 | `40 + n*30 + 10` | as above | as above | as above | ✅ dynamic-height palette mirrors v3 user-model-preview |
| NNDiagram | `NNContainer` | 320 | 200 | solid, rx=8 | `#F5F5F5` (hardcoded) | header bar | ⚠️ fill colour hardcoded — won't dark-mode |
| NNDiagram | `Conv2DLayer` (rep. layer) | 160 | 140 | solid, rx=6 | white default | stereotype tspan + name + 80×80 PNG icon below divider | ✅ correct (SA-UX-FIX-2 B4 — icons restored) |
| NNDiagram | `TrainingDataset` | 160 | 140 | solid, rx=6 | white | as above | ✅ correct |

**Width regression** is the most pervasive issue: v3 used `width: 200`
for Class and Object node palette entries (and BUML Class import / paste
operations seed identical bounds); v4 lowered the palette default to
`DROPS.DEFAULT_ELEMENT_WIDTH = 160`. Existing diagrams loaded from a
saved project keep their stored width, but new drops are 20% narrower.
This is consistent across all node types using `DROPS.DEFAULT_ELEMENT_WIDTH`.

## E. Edge marker styles

Source: `packages/library/lib/utils/edgeUtils.ts::getEdgeMarkerStyles`.
Marker IDs resolved against `MARKER_CONFIGS` in `constants.ts:247-339`.

| Edge type | v3 expected | v4 actual | Verdict |
| --- | --- | --- | --- |
| `ClassInheritance` | hollow triangle target | `markerEnd: white-triangle` | ✅ correct |
| `ClassRealization` | hollow triangle target + dashed | `white-triangle` + `strokeDashArray: 10` | ✅ correct |
| `ClassDependency` | open arrow target + dashed | `black-arrow` (filled=false in MARKER_CONFIGS) + dashed | ✅ correct |
| `ClassUnidirectional` | open arrow target | `black-arrow` (open V) | ✅ correct |
| `ClassBidirectional` | no markers | no markers, plain dashed | ✅ correct |
| `ClassComposition` | filled diamond at SOURCE (PC-3 #3) | `markerStart: black-rhombus` | ✅ correct (PC-3 fix applied, line 205-213) |
| `ClassAggregation` | hollow diamond at SOURCE | `markerStart: white-rhombus` | ✅ correct (PC-3 fix applied, line 195-204) |
| `ClassOCLLink` | open arrow target + dotted | `markerEnd: url(#class-ocl-link-marker)` + dashed `4 2` | ❌ **`class-ocl-link-marker` is NOT in `MARKER_CONFIGS`** — `InlineMarker` returns `null` for unknown IDs (`InlineMarker.tsx:92`), so the marker never renders. PC-3 #1 still broken. Should reference `black-arrow` instead. |
| `ClassLinkRel` | solid line | falls through to `default` (no markers, solid) | ✅ correct (matches v3 plain-line intent) |
| `StateTransition` | open arrow target | falls through to `default` (no markers) | ❌ **NO MARKER RENDERED**. v3 drew an open arrow at target via inline `<marker>` (`uml-state-transition-component.tsx:65-76`). Add a `case "StateTransition"` returning `markerEnd: "url(#black-arrow)"`. |
| `AgentStateTransition` | open arrow target | falls through to `default` (no markers) | ❌ **NO MARKER RENDERED** (PC-8 #1 still broken). v3 source is identical to StateTransition. Same fix. |
| `AgentStateTransitionInit` | open arrow at target + maybe dashed | falls through to `default` (no markers) | ❌ **NO MARKER RENDERED**. v3 had its own marker definition. Add a case mirroring StateTransition (with optional `strokeDashArray: "10"`). |
| `ObjectLink` | solid line, no markers | line 175-184: `strokeDashArray: "0"`, no markers | ✅ correct |
| `NNNext` | open arrow target | `markerEnd: black-arrow` | ✅ correct |
| `NNComposition` | diamond at NNContainer side (source) | `markerStart: black-rhombus` | ✅ correct |
| `NNAssociation` | open arrow (per task brief) / plain line (per v4 comment) | line 364-369: no marker, solid | ⚠️ ambiguous — code comment claims "plain line (Dataset ↔ NNContainer)" but the task brief asks for `open arrow`. Confirming with v3 source recommended. |

## Critical visual regressions

1. **`ClassOCLLink` marker missing** — `getEdgeMarkerStyles` returns
   `markerEnd: "url(#class-ocl-link-marker)"` (`edgeUtils.ts:252`) but
   `class-ocl-link-marker` is **not registered in `MARKER_CONFIGS`** at
   `constants.ts:247-339`. `InlineMarker` returns `null` for unknown IDs
   (`InlineMarker.tsx:92`), so OCL-link edges render as a dashed line
   with no arrowhead. The fix is to use `markerEnd: "url(#black-arrow)"`
   — `black-arrow` already has `filled: false`, which is the open-V
   shape required.

2. **`StateTransition` / `AgentStateTransition` / `AgentStateTransitionInit`
   render with no markers** — none of these three are listed in the
   switch statement of `getEdgeMarkerStyles`, so they fall through to
   `default` (no `markerEnd`). v3 components define inline `<marker>`
   elements with an open V. PC-8 #1 was meant to fix `AgentStateTransition`
   but the fix has not been applied. All three transitions should have
   `markerEnd: "url(#black-arrow)"`; the init variant additionally needs
   `strokeDashArray: "10"` per v3 conventions.

3. **`--besser-warning-yellow` debrand rename mismatch** —
   `themings.json` declares `--besser-alert-warning-yellow` (with the
   `alert-` prefix), but `theme.ts:13` and
   `components/theme/styles.tsx:5` consume `--besser-warning-yellow`
   (no prefix). Warning surfaces stay on the hardcoded `#ffc800`
   fallback in both light and dark modes — they do not theme-switch.
   Either rename the consumers or rename the theme key.

4. **`--besser-gray-700` consumed without a declaration or fallback** —
   `ObjectLinkEditPanel.tsx:175` writes
   `sx={{ color: "var(--besser-gray-700)" }}` (no fallback). With no
   declaration anywhere, the empty-state caption resolves to MUI's
   default text colour (which is `--besser-primary-contrast` — black or
   white depending on theme), breaking the intended muted-grey look.
   Add a `--besser-gray-700` declaration to `themings.json` or change
   the consumer to `var(--besser-gray-variant)` (already declared and
   theme-aware).

5. **Default-width regression for Class / Object / UserModelName nodes
   (160 vs v3 200)** — Palette drops are 20% narrower than v3. Saved
   diagrams keep stored bounds, but new elements sit smaller and labels
   wrap sooner. v3's default at
   `packages/editor/.../class-preview.ts:45` was `200`. Restore by
   bumping `DROPS.DEFAULT_ELEMENT_WIDTH` from 160 → 200, or by raising
   only the affected palette entries (Class / ObjectName /
   UserModelName) so other diagram types keep their current sizing.

Other minor regressions (not in top 5): `StateFinalNode` outer-ring
stroke is 1.5px instead of the theme's 1px line width; `State` node
divider draws even when there is no body (v3 only drew it when
`hasBody`); `AgentIntent` and `NNContainer` have hardcoded fill colours
(`#E3F9E5`, `#F5F5F5`) that won't theme-switch in dark mode; v3 tokens
`grid-minor` and `graylight` were dropped without replacement. The
`besser-interactive-selection-color` is the only true CSS-rule
declaration in the library — every other "declaration" is JS-injected
at runtime via `theme-switcher.ts`. A future consumer that loads the
library outside the webapp shell (e.g. Storybook) gets only the
hardcoded fallbacks unless it also imports themings.json.
