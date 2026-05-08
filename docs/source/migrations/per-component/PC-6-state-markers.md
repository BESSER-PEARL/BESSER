# PC-6: State machine marker nodes

Read-only audit of the StateMachineDiagram marker / leaf nodes (`StateInitialNode`, `StateFinalNode`, `StateMergeNode`, `StateForkNode`, `StateForkNodeHorizontal`, `StateActionNode`, `StateObjectNode`) — their canvas SVGs, palette previews, and inspector panels.

## Sources

### Old (`packages/editor/src/main/packages/uml-state-diagram/`)
- `uml-state-initial-node/uml-state-initial-node.ts` — model. `static features = { ...UMLElement.features, resizable: false, updatable: false }`. Default bounds 50×50.
- `uml-state-initial-node/uml-state-initial-node-component.tsx` — `<ThemedCircleContrast>` (`fill: theme.color.primaryContrast`, i.e. **black**), `r = min(w,h)/2`, `strokeColor="none"`.
- `uml-state-final-node/uml-state-final-node.ts` — `updatable: false`, `resizable: false`. Default 50×50.
- `uml-state-final-node/uml-state-final-node-component.tsx` — outer `ThemedCircle` (hollow, `r=45%`) + inner `ThemedCircle` filled with `element.strokeColor` at `r=35%`.
- `uml-state-merge-node/uml-state-merge-node.ts` — *no* `updatable: false` (mergeable / nameable). Bounds via `calculateNameBounds`.
- `uml-state-merge-node/uml-state-merge-node-component.tsx` — `ThemedPolyline` diamond + `Multiline` bold name centred.
- `uml-state-merge-node/uml-state-merge-node-update.tsx` — popup: name `Textfield`, `ColorButton`/`StylePane`, **size inputs (50–1000)**, and the **decisions section** that lists every outgoing transition with an editable `Textfield` (`relationship.name`) → `ArrowRightIcon` → target `Body` label.
- `uml-state-fork-node/uml-state-fork-node.ts` — `updatable: false`. `defaultWidth=20, defaultHeight=60`. Constructor pins width to default.
- `uml-state-fork-node/uml-state-fork-node-component.tsx` — `ThemedRect width=100% height=100%` with **`fillColor={element.strokeColor}`** (so the bar is black by default).
- `uml-state-fork-node-horizontal/uml-state-fork-node-horizontal.ts` — `updatable: false`. `defaultWidth=60, defaultHeight=20`. Constructor pins height.
- `uml-state-fork-node-horizontal/uml-state-fork-node-horizontal-component.tsx` — same `ThemedRect` pattern, fill = strokeColor.
- `uml-state-action-node/uml-state-action-node-component.tsx` — `ThemedRect rx=5 ry=5` (rounded), `Text y={20}` for the name, `children` slot for code-block.
- `uml-state-object-node/uml-state-object-node-component.tsx` — plain `ThemedRect` + `Multiline` bold name; no separate inspector for `classId` in v3 (the node simply holds an instance name).

`ThemedCircleContrast` (`packages/editor/src/main/components/theme/themedComponents.ts:78–88`) defaults `fill = theme.color.primaryContrast` — **`#000000` in the light theme** — and `stroke = theme.color.background` (white). That is the load-bearing fill for the v3 initial node.

### New (`packages/library/lib/`)
- `nodes/stateMachineDiagram/StateInitialNode.tsx` — `<circle r=min(w,h)/2 fill={fillColor || "var(--besser-primary-contrast, #000000)"} stroke="none">`.
- `nodes/stateMachineDiagram/StateFinalNode.tsx` — outer circle `r=0.9·min(w,h)/2` (`fill={fillColor || "white"}`, `stroke=strokeColor || "var(--besser-primary-contrast, #000000)"`, strokeWidth 1.5) + inner `r=0.7·min(w,h)/2` filled with `stroke`.
- `nodes/stateMachineDiagram/StateMergeNode.tsx` — `<polygon>` diamond, fill/stroke from `getCustomColorsFromData`, bold centred `<text>`.
- `nodes/stateMachineDiagram/StateForkNode.tsx` — `<rect>` filled with `fillColor || strokeColor`. Width pinned 20 via `NodeResizer minWidth=20 maxWidth=20`.
- `nodes/stateMachineDiagram/StateForkNodeHorizontal.tsx` — `<rect>` mirror, height pinned 20 via `minHeight=20 maxHeight=20`.
- `nodes/stateMachineDiagram/StateActionNode.tsx` — `<rect rx=5 ry=5>`, centred name, `data.code` not painted on the canvas.
- `nodes/stateMachineDiagram/StateObjectNode.tsx` — `<rect>` + bold centred `<text>` showing `name` or `${name}: ${className}` if `className` is set.
- `components/svgs/nodes/stateMachineDiagram/StateMachineSVGs.tsx` — palette previews. Initial uses `fill="var(--besser-primary-contrast, #000)"` directly; Fork / ForkHorizontal use the same. Action / Object / Merge use `fill="var(--besser-background, white)" stroke="var(--besser-primary-contrast, #000)"`.
- `components/inspectors/stateMachineDiagram/StateLabelEditPanel.tsx` — shared body for *all five* marker nodes; renders `NodeStyleEditor` + a `MuiTextField label="label"` for `data.name`.
- `components/inspectors/stateMachineDiagram/StateActionNodeEditPanel.tsx` — `NodeStyleEditor`, name field, multiline `code` field.
- `components/inspectors/stateMachineDiagram/StateObjectNodeEditPanel.tsx` — `NodeStyleEditor`, instance-name field, `classId` `Select` populated from `diagramBridge.getAvailableClasses()`.
- `components/inspectors/stateMachineDiagram/index.ts` — registers all five marker nodes (Initial / Final / Merge / Fork / ForkHorizontal) against `StateLabelEditPanel`.
- `utils/layoutUtils.ts:47–52` — `getCustomColorsFromData` returns `fillColor = data.fillColor || "var(--besser-background)"`, `strokeColor = data.strokeColor || "var(--besser-primary-contrast)"`.
- `constants.ts:122–145` — `--besser-primary-contrast: #000000`, `--besser-background: #ffffff`.
- `constants.ts:780–848` — palette `defaultData` for every state-marker omits `fillColor` (just `{ name: "" }`).

## Verdict

**Visual parity is broken on the StateInitialNode and both fork bars** because the helper `getCustomColorsFromData` substitutes the *background* CSS variable for a missing `data.fillColor`, and the palette `defaultData` never sets one. Inspector parity also has two clear regressions (no marker `updatable:false` gating; lost merge-node decisions editor). Action and object nodes carry through correctly, with the object node gaining a `classId` picker that v3 didn't have.

## Critical-check trace — why StateInitialNode renders white

1. `constants.ts:780–848` declares `defaultData: { name: "" }` for `StateInitialNode`, `StateFinalNode`, `StateMergeNode`, `StateForkNode`, `StateForkNodeHorizontal`. There is **no** `fillColor` baked into the default node payload.
2. On render, `StateInitialNode.tsx:27` calls `const { fillColor } = getCustomColorsFromData(data)`.
3. `utils/layoutUtils.ts:48–51` resolves a missing `data.fillColor` to `"var(--besser-background)"` — a non-empty string. **It does not return `undefined`.**
4. `StateInitialNode.tsx:59` then evaluates `fill={fillColor || "var(--besser-primary-contrast, #000000)"}`. Because `fillColor` is the truthy string `"var(--besser-background)"`, the `||` short-circuit *never reaches* the `#000000` fallback.
5. `--besser-background` resolves to `#ffffff` (`constants.ts:129`). The circle paints **white on a white canvas**, hence the user-visible "the bullet disappeared" symptom.

Note: this is the symptom of the SA-DEBRAND `--apollon-*` → `--besser-*` rename only insofar as the helper was rewritten alongside the rename — there is no surviving old-named variable. The actual root cause is the *fallback choice* in `getCustomColorsFromData`: it defaults the fill to *background*, not to `transparent` / `undefined`. v3 hard-coded `theme.color.primaryContrast` (black) for this node specifically via `ThemedCircleContrast`, which is what made the marker appear black even when no element fillColor was set. The new lib's `<circle>` is plain SVG without that themed wrapper, so the same defaulting strategy that works for "shape with white interior + black border" silently flips this node's body to background.

The same problem hits `StateForkNode.tsx:31` and `StateForkNodeHorizontal.tsx:29`:
```tsx
const { strokeColor, fillColor } = getCustomColorsFromData(data)
const fill = fillColor || strokeColor
```
`fillColor` is the truthy string `"var(--besser-background)"` again, so `fill` is white — even though v3 explicitly forced `fillColor={element.strokeColor}` to keep the bar black. With no stroke and a white fill the bars are invisible too.

`StateFinalNode` is a partial victim: the *outer* ring has `fill={fillColor || "white"}`, which is also white whether the user customised it or not, but that matches the v3 hollow-ring intent so the bullseye still reads. The inner disk uses `fill={stroke}` with `stroke = strokeColor || "var(--besser-primary-contrast, #000000)"`, and `getCustomColorsFromData` returns `var(--besser-primary-contrast)` for missing strokes — which *does* resolve to black, so the inner dot is fine.

`StateMergeNode`, `StateActionNode`, `StateObjectNode` all use `fill={fillColor}` directly with a visible stroke (`stroke={strokeColor}`), which is the same v3 contract — these render correctly.

### Actual fill values found in source

| Node                     | New lib fill expression                                                                 | Resolved at runtime (no custom fill)        | v3 fill                                                       |
| ------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------- |
| `StateInitialNode`       | `fillColor \|\| "var(--besser-primary-contrast, #000000)"` (`StateInitialNode.tsx:59`)   | `var(--besser-background)` → **#ffffff**     | `theme.color.primaryContrast` → **#000000**                    |
| `StateFinalNode` outer   | `fillColor \|\| "white"` (`StateFinalNode.tsx:60`)                                       | `var(--besser-background)` → **#ffffff**     | `theme.color.background` → **#ffffff** (parity)                |
| `StateFinalNode` inner   | `fill={stroke}` (`StateFinalNode.tsx:68`); `stroke = strokeColor \|\| "...primary-contrast..."` | `var(--besser-primary-contrast)` → **#000000** | `theme.color.primaryContrast` → **#000000** (parity)         |
| `StateForkNode`          | `fill = fillColor \|\| strokeColor` (`StateForkNode.tsx:31`)                             | `var(--besser-background)` → **#ffffff**     | `element.strokeColor` defaulted to **#000000**                 |
| `StateForkNodeHorizontal`| `fill = fillColor \|\| strokeColor` (`StateForkNodeHorizontal.tsx:29`)                   | `var(--besser-background)` → **#ffffff**     | `element.strokeColor` defaulted to **#000000**                 |
| `StateMergeNode`         | `fill={fillColor}` (`StateMergeNode.tsx:52`)                                             | `var(--besser-background)` → **#ffffff** with visible black stroke | `element.fillColor` (white) with stroke               |
| `StateActionNode`        | `fill={fillColor}` (`StateActionNode.tsx:58`)                                            | `var(--besser-background)` → **#ffffff** with visible stroke | `element.fillColor` with stroke                           |
| `StateObjectNode`        | `fill={fillColor}` (`StateObjectNode.tsx:61`)                                            | `var(--besser-background)` → **#ffffff** with visible stroke | `element.fillColor` with stroke                           |

The palette previews in `StateMachineSVGs.tsx` are unaffected — they hard-code their fills to `var(--besser-primary-contrast, #000)` for Initial / Fork / ForkHorizontal so the sidebar drag sources still look black, which is presumably why the regression escaped review.

## Coverage matrix

| Feature                                                                 | Old                                                              | New                                                                 | Notes                                                                                                                          |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `StateInitialNode` solid black bullet                                  | `ThemedCircleContrast` defaults to `theme.color.primaryContrast` | `<circle fill={fillColor \|\| "...#000000"}>` but `fillColor` resolves to background | **Gap 1 (critical)** — renders white. See trace above.                                                                          |
| `StateInitialNode` default bounds                                       | 50×50                                                            | 45×45 (`constants.ts:790–791`)                                       | Doc-comment in source claims "matches v3 50×50" — minor mismatch.                                                              |
| Initial-node corner/mid handles hidden                                  | n/a (single-port v3 model)                                       | `hiddenHandles=[TopLeft, TopRight, RightTop, RightBottom, BottomRight, BottomLeft, LeftBottom, LeftTop]` (`StateInitialNode.tsx:36–44`) | Parity (only 4 cardinal sides).                                                                                                  |
| `StateFinalNode` hollow ring + inner disk                               | two `ThemedCircle`s, outer hollow, inner = strokeColor           | `<circle>` outer (white) + inner filled with `stroke`                | Visual parity — outer fill white, inner black.                                                                                 |
| `StateFinalNode` ring radii                                             | outer 45 %, inner 35 %                                           | outer `0.9·min/2 = 45%`, inner `0.7·min/2 = 35%`                     | Parity.                                                                                                                        |
| `StateMergeNode` diamond shape                                          | `ThemedPolyline` 5-point closing diamond                          | `<polygon>` 4-point diamond                                          | Parity (polygon auto-closes).                                                                                                  |
| `StateMergeNode` centred bold name                                      | `Multiline fontWeight="bold"` centred                            | `<text>` `fontWeight="bold"`, `textAnchor="middle"`                  | Parity for single line; multi-line wrap **not** reproduced (`<text>` doesn't wrap). Minor gap.                                  |
| `StateForkNode` vertical bar                                            | `ThemedRect width=100% height=100% fill=strokeColor`             | `<rect>` `fill = fillColor \|\| strokeColor`                          | **Gap 2** — see fill table; bar is white.                                                                                      |
| `StateForkNode` width pinned                                            | constructor pins `bounds.width=20`                               | `NodeResizer minWidth=20 maxWidth=20`                                | Parity.                                                                                                                        |
| `StateForkNodeHorizontal` horizontal bar                                | identical pattern, height pinned 20                              | `<rect>` + `NodeResizer minHeight=20 maxHeight=20`                   | Same fill bug.                                                                                                                 |
| `StateActionNode` rounded rectangle                                     | `ThemedRect rx=5 ry=5`                                           | `<rect rx=5 ry=5>`                                                   | Parity.                                                                                                                        |
| `StateActionNode` name text positioning                                 | `Text y={20}` (top-anchored)                                     | `<text x={width/2} y={height/2 + 5}>` (centred)                      | **Gap** — text positioning differs; v3 renders the action label near the top edge, new lib centres it.                          |
| `StateObjectNode` rectangle + name                                      | `ThemedRect` + bold `Multiline`                                  | `<rect>` + bold centred `<text>`                                     | Parity.                                                                                                                        |
| `StateObjectNode` `classId` reference                                   | not surfaced in v3                                               | `data.classId` + `Select` populated by `diagramBridge.getAvailableClasses()` (`StateObjectNodeEditPanel.tsx:78–102`); canvas label switches to `${name}: ${className}` (`StateObjectNode.tsx:37`) | New lib improvement.                                                                                                            |
| `updatable: false` invariant for marker nodes                           | `UMLStateInitialNode`, `UMLStateFinalNode`, `UMLStateForkNode`, `UMLStateForkNodeHorizontal` set `updatable: false` (no popup name field) | All five Initial / Final / Merge / Fork / ForkHorizontal share `StateLabelEditPanel`, which **always** renders `<MuiTextField label="label">` for `data.name` | **Gap 3** — see below.                                                                                                          |
| `StateMergeNode` inline decisions editor                                | `UMLStateMergeNodeUpdate` lists every outgoing transition with editable `name` `Textfield` and target label (lines 127–148) | `StateLabelEditPanel` (no decisions block)                           | **Gap 4** — see below.                                                                                                          |
| Size inputs on merge popup                                              | width / height number inputs (50–1000) on the popup              | `NodeResizer` handles only                                          | Acceptable (canvas resize replaces popup form).                                                                                |
| Style pane (fill / line / text colour)                                  | `StylePane` per element                                          | `NodeStyleEditor` per node (`StateLabelEditPanel.tsx:41–44`)        | Parity.                                                                                                                        |
| Palette previews carry through black fill                               | n/a                                                              | `StateMachineSVGs.tsx` hard-codes `var(--besser-primary-contrast, #000)` for Initial / Fork / ForkHorizontal | Parity at sidebar level — masks the canvas regression.                                                                          |
| Auto-create child bodies                                                | container State auto-creates body / fallback                     | matches via `nodes/index.ts` registration                            | Out of scope for marker audit.                                                                                                  |

## Top 3 gaps

1. **(critical) `StateInitialNode` and both fork bars render white instead of solid black on a fresh node.** Root cause: `getCustomColorsFromData` (`utils/layoutUtils.ts:47–52`) returns `data.fillColor || "var(--besser-background)"` — a non-empty string when `data.fillColor` is missing — so the `||` fallbacks in `StateInitialNode.tsx:59`, `StateForkNode.tsx:31`, and `StateForkNodeHorizontal.tsx:29` are dead code paths. Combined with `constants.ts` palette `defaultData: { name: "" }` (no `fillColor`), every newly-dropped initial / fork / fork-horizontal node renders with its body filled to `--besser-background` (`#ffffff`). v3 forced the fill via `ThemedCircleContrast` (black) for Initial and `fillColor={element.strokeColor}` (black) for the forks, both of which were stripped during the React-Flow port. Fix options: (a) bake `fillColor: "var(--besser-primary-contrast)"` into the marker's `defaultData` in `constants.ts`; (b) skip the `getCustomColorsFromData` helper for these three nodes and read `data.fillColor` directly so the `|| "...#000000"` fallback fires when undefined; (c) change `getCustomColorsFromData` to return `data.fillColor` as-is (possibly `undefined`) and let each consumer pick its own fallback. Option (a) preserves the existing helper contract used by every other shape and is the smallest change.

2. **Marker-node inspectors ignore the `updatable: false` invariant.** v3 stamped `static features = { ...UMLElement.features, updatable: false }` on `UMLStateInitialNode`, `UMLStateFinalNode`, `UMLStateForkNode`, `UMLStateForkNodeHorizontal` so that double-clicking a marker did **not** open a name editor — markers are nameless symbols. The new lib registers all five (Initial / Final / Merge / Fork / ForkHorizontal) against the shared `StateLabelEditPanel`, which always renders an MUI `TextField label="label"` bound to `data.name` (`StateLabelEditPanel.tsx:46–53`). Result: users can now type a name onto an initial bullet or fork bar that has no place to display it — silently storing data that the canvas ignores and round-trip serialisers may emit. Either route the four `updatable:false` nodes to a colour-only panel (style editor, no text field), or add an `editableName: false` config flag to `StateLabelEditPanel` and gate the text field on it. `StateMergeNode` should remain editable since v3 *did* allow naming it.

3. **`StateMergeNode` lost its outgoing-transitions decisions editor.** `UMLStateMergeNodeUpdate` (v3, lines 127–148) iterated `state.elements` for relationships whose `source.element === merge.id`, and rendered each as `Textfield(decision.name) → ArrowRightIcon → Body(target.name)` — letting the user label the conditional branches inline from the merge node's popup. The new `StateLabelEditPanel` has no decisions block; outgoing-transition labels can only be edited by selecting each edge individually (via `StateMachineDiagramEdgeEditPanel`). This is a real workflow regression for decision-style state machines where the merge is the natural focal point. Either reinstate a decisions section in a dedicated `StateMergeNodeEditPanel` (using `useDiagramStore` to filter `edges` by `source === elementId`, mirroring the v3 query) or document the cut and add a "see edges" affordance.

## Out-of-scope observations

- `StateActionNode` text positioning shifts from top-anchored (`y=20`) in v3 to vertically-centred in the new lib. Visually different but probably acceptable since v3 left a code-block child slot below; new lib edits `code` in the inspector and keeps a single label.
- `StateMergeNode` switches from `Multiline` (auto-wrap) to `<text>` (single line). Long merge labels will overflow the diamond; v3 wrapped them.
- `StateInitialNode` default bounds are 45×45 in `constants.ts` (with a comment claiming "45×45 — matches v3"), but v3's `UMLStateInitialNode.bounds` ships at 50×50. Cosmetic.

## Files

- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateInitialNode.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateFinalNode.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateMergeNode.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateForkNode.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateForkNodeHorizontal.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateActionNode.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/stateMachineDiagram/StateObjectNode.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/svgs/nodes/stateMachineDiagram/StateMachineSVGs.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/stateMachineDiagram/StateLabelEditPanel.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/stateMachineDiagram/StateActionNodeEditPanel.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/stateMachineDiagram/StateObjectNodeEditPanel.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/stateMachineDiagram/index.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/layoutUtils.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/constants.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-initial-node/uml-state-initial-node.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-initial-node/uml-state-initial-node-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-final-node/uml-state-final-node-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-merge-node/uml-state-merge-node-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-merge-node/uml-state-merge-node-update.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-fork-node/uml-state-fork-node.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-fork-node/uml-state-fork-node-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-fork-node-horizontal/uml-state-fork-node-horizontal.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-fork-node-horizontal/uml-state-fork-node-horizontal-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-action-node/uml-state-action-node-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/uml-state-object-node/uml-state-object-node-component.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/components/theme/themedComponents.ts`
