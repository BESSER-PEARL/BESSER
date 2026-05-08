# PC-5 — State + Bodies + Transition Edge parity audit

Read-only per-component audit covering the `State` parent, its `StateBody` /
`StateFallbackBody` children, the `StateCodeBlock` panel, and the
`StateTransition` edge.

Audit date: 2026-05-08
Submodule HEAD audited: same as `parity-final/stateMachineDiagram.md`
(SA-7b cutover).

## Files in scope

**v3 (old fork)** — `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/uml-state-diagram/`:

- `uml-state/uml-state.ts` (model class)
- `uml-state/uml-state-component.tsx` (canvas render)
- `uml-state/uml-state-update.tsx` (parent inspector + body adders)
- `uml-state/uml-state-member.ts` (shared base for body / fallback-body)
- `uml-state/uml-state-member-component.tsx`
- `uml-state-body/uml-state-body.ts` + `uml-state-body-update.tsx`
- `uml-state-fallback_body/uml-state-fallback_body.ts` (no own update —
  reuses `UmlBodyUpdate`)
- `uml-state-code-block/uml-state-code-block.ts`
  + `uml-state-code-block-component.tsx` + `uml-state-code-block-update.tsx`
- `uml-state-transition/uml-state-transition.ts`
  + `uml-state-transition-component.tsx` (visited only for context)
  + `uml-state-transition-update.tsx`

**v4 (new lib)** — `besser/utilities/web_modeling_editor/frontend/packages/library/lib/`:

- `nodes/stateMachineDiagram/State.tsx`
- `nodes/stateMachineDiagram/StateBody.tsx`
- `nodes/stateMachineDiagram/StateFallbackBody.tsx`
- `nodes/stateMachineDiagram/StateCodeBlock.tsx`
- `edges/edgeTypes/StateMachineDiagramEdge.tsx`
- `components/inspectors/stateMachineDiagram/StateEditPanel.tsx`
- `components/inspectors/stateMachineDiagram/StateBodyEditPanel.tsx`
- `components/inspectors/stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx`
- `utils/versionConverter.ts` (round-trip for `kind` / `code` / `eventName`)
- `types/nodes/NodeProps.ts:287` (`StateBodyNodeProps`),
  `:296` (`StateNodeProps`), `:328` (`StateCodeBlockProps`)

Status legend: **PASS** = present and matches the v3 surface or the brief;
**GAP** = missing, regressed, or behaves differently from the v3 surface
or the SA-3 brief / Round 1 #15/#16; **N/A in v3** = brief asks v4 to
expose something v3 did not (judged on its own).

---

## Top-line verdict

**PASS with three GAPs (two MEDIUM, one LOW).** Data-shape parity is
intact, parent/child wiring via `parentId` works, the SA-3 BESSER-extra
fields (`kind`, `code`, `eventName`) all round-trip, and the v4
`StateBodyEditPanel` over-delivers vs v3 by exposing a `kind` dropdown +
multi-line `code` field that the v3 fork lacked. The remaining gaps are
all on the inspector layer:

1. **MEDIUM (Round 1 #15/#16)** — `StateMachineDiagramEdgeEditPanel.tsx`
   exposes neither **flip** nor a **color editor**. The v3 fork had
   both (`uml-state-transition-update.tsx:131-137,172-178`); the SA-2.2
   #26 pass added them to `ClassEdgeEditPanel` and
   `AgentDiagramEdgeEditPanel`, but SA-3's edge panel was never
   back-filled.
2. **MEDIUM (UX-visible)** — `StateBodyEditPanel.tsx` advertises a
   "code (CodeMirror Python)" surface in the brief. v4 ships a plain
   MUI `TextField multiline` with no syntax highlighting, no language
   indicator, and no Python-aware indenting. v3 had nothing here at all
   (the body row was a single `Textfield` for `name` + colors), so this
   is **over-delivery vs v3 but under-delivery vs the brief.** Same
   shortfall in `StateCodeBlockEditPanel` (cross-checked but out of
   PC-5 scope) and `StateMachineDiagramEdgeEditPanel`'s `code` field.
3. **LOW (visual)** — `StateBody.tsx` / `StateFallbackBody.tsx` ignore
   `data.kind` at render time. The inspector lets the user pick `entry
   / do / exit / on transition`, the value round-trips through
   `versionConverter.ts:964`, but the canvas still only paints
   `data.name`. v3 had the same blind spot (the kind concept didn't
   exist), so this is **N/A in v3 but a UX inconsistency in v4** —
   users author a kind and see no canvas feedback.

No data-shape regressions. No missing element / edge types. No
constraint regressions outside what was already absent in v3 (initial-
state uniqueness and final-state terminality remain backend-only, same
as v3).

---

## 1. Parent `State` node

| Field / behaviour | v3 source | v4 location | Status | Notes |
|---|---|---|---|---|
| `name` | `IUMLState.name` (`uml-state.ts`); rendered via `<Text>` in `uml-state-component.tsx:43-58` | `data.name` rendered as `<text>` in `State.tsx:79-104` | **PASS** | Same single-line label, same fallback when no stereotype. |
| `stereotype` | `IUMLState.stereotype: string \| null` (`uml-state.ts:19`); rendered as `«…»` tspan when set (`uml-state-component.tsx:34`) | `data.stereotype` rendered as `«…»` `<text>` (`State.tsx:69-78`) | **PASS** | v4 adds an explicit `Select` dropdown in `StateEditPanel.tsx:65-79` with the v3 stereotype set + "no stereotype". v3's update panel had no stereotype editor at all (stereotype was set programmatically by element constructor). **Over-delivers vs v3.** |
| `italic` | `IUMLState.italic: boolean` (`uml-state.ts:17`); applied as `fontStyle` (`uml-state-component.tsx:41`) | `data.italic`; applied as `fontStyle` (`State.tsx:84,98`) | **PASS** | v4 adds an explicit checkbox in `StateEditPanel.tsx:84-91`. v3 had no italic checkbox in the update panel; it was a model-only flag. **Over-delivers.** |
| `underline` | `IUMLState.underline: boolean`; applied as `textDecoration` | `data.underline`; applied as `textDecoration` (`State.tsx:85,99`) | **PASS** | v4 adds an explicit checkbox (`StateEditPanel.tsx:93-100`). Same over-delivery as italic. |
| Action / delete buttons in inspector | v3 had a `<TrashIcon>` next to the name field (`uml-state-update.tsx:99-101`) | Inspector has no per-element delete; relies on canvas-level delete | **PASS** | Matches the SA-2.2 inspector convention; brief did not call out a per-panel delete for State. |
| `fillColor` / `strokeColor` / `textColor` | `IUMLElement` baseline (`StylePane fillColor lineColor textColor` in `uml-state-update.tsx:103-110`) | `getCustomColorsFromData(data)` (`State.tsx:32`); `NodeStyleEditor` in `StateEditPanel.tsx:55-58` | **PASS** | Same three-way colour surface. |
| `deviderPosition` (computed) | Stored at render time (`uml-state.ts:38,98`) | Not persisted; layout-derived from header height (`State.tsx:36-38, 106-113`) | **PASS** | v4 computes from `LAYOUT.DEFAULT_HEADER_HEIGHT[_WITH_STEREOTYPE]`; round-trip parity confirmed in `parity-final/stateMachineDiagram.md`. |
| `hasBody` / `hasFallbackBody` (computed) | Recomputed each render in `uml-state.ts:76-77` | Not persisted; React Flow tracks children directly | **PASS** | Same outcome — divider line drawn unconditionally in v4 (`State.tsx:106-113`); v3 only drew it when bodies/fallback existed. **Cosmetic over-paint** — divider always present in v4 even on empty State, but harmless because the body region renders empty children-through. |
| Resizable | `static features.resizable = 'WIDTH'` (`uml-state.ts:29`) | `<NodeResizer minWidth={120} minHeight={60}>` (`State.tsx:43-49`) | **PASS** | v4 also allows height resize where v3 only allowed width. **Over-delivers.** |

### 1.1 Inspector fields exposed in `StateEditPanel.tsx`

| Order | Widget | v4 line | Source field |
|---|---|---|---|
| 1 | `NodeStyleEditor` (name + colours) | `:55-58` | `data.name`, `data.fillColor`, `data.strokeColor`, `data.textColor` |
| 2 | Stereotype dropdown | `:64-80` | `data.stereotype` |
| 3 | Italic checkbox | `:83-91` | `data.italic` |
| 4 | Underline checkbox | `:93-100` | `data.underline` |
| 5 | Free `name` MUI field (duplicate of #1) | `:105-112` | `data.name` |

**Minor finding (not a gap)**: rows 1 and 5 both bind to `data.name`,
but #1 also surfaces colours and #5 is a redundant-but-harmless rename
shortcut. Editing in either updates the canvas live. v3 had a single
name field (`uml-state-update.tsx:97`); duplicating it in v4 is
cosmetic over-delivery.

---

## 2. `StateBody` and `StateFallbackBody`

| Field / behaviour | v3 source | v4 location | Status | Notes |
|---|---|---|---|---|
| `name` (single-line label) | `UMLStateMember` (shared base, `uml-state-member.ts:31`); rendered in `uml-state-member-component.tsx:14-17` | `data.name` rendered as `<text>` (`StateBody.tsx:50-58`, `StateFallbackBody.tsx:51-60`) | **PASS** | Same left-aligned text at `x=10`. |
| Italic flag for fallback body | v3 didn't italicise — relied on type for visual separation (just a label) | `StateFallbackBody.tsx:56` hard-codes `fontStyle="italic"` | **PASS (visual improvement)** | v4 distinguishes fallback rows visually; v3 only distinguished by position below the divider. **Over-delivers**. |
| Parent attachment via `parentId` | v3 used `UMLContainer.ownedElements` + `reorderChildren` to nest bodies (`uml-state.ts:57-61`) and the renderer laid them out by mutating `bounds.x` / `bounds.y` (`uml-state.ts:91-104`) | React Flow `parentId` set in `versionConverter.ts:2133` (`owner: node.parentId ?? null`) and threaded back through `:929-941` on the v3→v4 path | **PASS** | Confirmed via `versionConverter.ts:2027-2035` ("Keeps `StateBody` / `StateFallbackBody` … as separate React-Flow children with `parentId`"). |
| `code` (BESSER addition) | Not in v3 model — `UmlBodyUpdate` only authored `name` + colours (`uml-state-body-update.tsx:42-50`) | `data.code` round-tripped at `versionConverter.ts:963`; authored via `MuiTextField multiline` (`StateBodyEditPanel.tsx:88-98`) | **PASS** | **Over-delivers vs v3.** |
| `kind` (entry / do / exit / on-transition) | Not in v3 model — there was no `kind` discriminator at all (v3 only had two element types: `StateBody` and `StateFallbackBody`) | `data.kind` round-tripped at `versionConverter.ts:964`; authored via `Select` (`StateBodyEditPanel.tsx:62-77`) | **PASS (data) / GAP (visual)** | The brief asks for a `kind` picker. v4 stores and round-trips it correctly, but **`StateBody.tsx` and `StateFallbackBody.tsx` do not paint the kind on the canvas.** The user picks "entry" in the inspector and sees no change — the row continues to display only `data.name`. See **GAP-3** below. |
| `code` editor flavour: brief asks "CodeMirror Python" | n/a in v3 | Plain MUI `TextField multiline` (`StateBodyEditPanel.tsx:88-98`) | **GAP** | See **GAP-2** below. v4 ships the field but not the editor. |
| Per-row colour pane | v3 `StylePane fillColor textColor` (`uml-state-body-update.tsx:50`) | `NodeStyleEditor` exposes fill/text/stroke (`StateBodyEditPanel.tsx:53-57`) | **PASS** | v4 also exposes stroke; v3 only had fill + text. **Over-delivers.** |
| Per-row delete | v3 had a `<TrashIcon>` per body row (`uml-state-body-update.tsx:46-48`) | No per-row delete — bodies are first-class React Flow nodes deleted from the canvas | **PASS** | Architectural; aligns with how every other v4 child node behaves. |
| Tab navigation between body fields | v3 had elaborate `componentDidUpdate` focus management (`uml-state-update.tsx:78-83, 138-153`) for sequential body creation | n/a — each body is its own canvas node, so the parent inspector no longer drives a body-list flow | **PASS (architectural)** | Tab-through-bodies UX is gone, but it was a v3 popup-only feature; in v4 you author each body in its own panel. |

### 2.1 Inspector fields exposed in `StateBodyEditPanel.tsx`

(Used for both `StateBody` and `StateFallbackBody` per
`components/inspectors/stateMachineDiagram/index.ts:24-25`.)

| Order | Widget | v4 line | Source field |
|---|---|---|---|
| 1 | `NodeStyleEditor` (name + colours) | `:53-57` | `data.name`, `data.fillColor`, `data.strokeColor`, `data.textColor` |
| 2 | Kind dropdown (entry / do / exit / on transition / unspecified) | `:62-77` | `data.kind` |
| 3 | Free `label` MUI field (duplicate of `name` in #1) | `:79-86` | `data.name` |
| 4 | Code multi-line MUI field | `:88-98` | `data.code` |

The `StateBodyEditPanel.tsx:18` docstring contains a misleading
remark: *"the inspector exposes a free-form kind tag stored inline on
the `name`"*. Reality: the panel writes `data.kind` (panel `:66`) and
the converter reads `data.kind` (`versionConverter.ts:964`); the kind
is **not** stored on `name`. Documentation drift only — code is
correct. **Not a gap, just a stale comment.**

---

## 3. `StateCodeBlock`

| Field / behaviour | v3 source | v4 location | Status | Notes |
|---|---|---|---|---|
| `code` | `IUMLStateCodeBlock.code` + redundant `_codeContent` preserve flag (`uml-state-code-block.ts:11-14, 35-39`) | `data.code` (`types/nodes/NodeProps.ts:328`); rendered in foreignObject (`StateCodeBlock.tsx:91-123`) | **PASS** | v4 drops `_codeContent` (it was an internal preserve hack); round-trip confirmed at `versionConverter.ts:995`. |
| `language` | `IUMLStateCodeBlock.language: string` defaulting to `'python'` (`uml-state-code-block.ts:23`) | `data.language` defaulting to `'python'` (`StateCodeBlock.tsx:37`) | **PASS** | Header label `Python` was hard-coded in v3 (`uml-state-code-block-component.tsx:87`); v4 renders `{language}` (`StateCodeBlock.tsx:81-89`), making the default visible to non-Python users. **Over-delivers.** |
| Tab → 4 spaces preservation | `preserveTabs` helper (`uml-state-code-block-component.tsx:10-12`) | `preserveTabs` helper (`StateCodeBlock.tsx:11`) | **PASS** | Identical implementation. |
| Resizable | `static features.resizable = true` (`uml-state-code-block.ts:19`); min 150×100 (`uml-state-code-block.ts:43-45`) | `<NodeResizer minWidth={150} minHeight={100}>` (`StateCodeBlock.tsx:46-50`) | **PASS** | Same minimums. |
| `parentId` allowed | v3 was a free-floating canvas element (no `UMLContainer` parent in `uml-state-code-block.ts`) | v4 accepts `parentId` (`StateCodeBlock.tsx:24, 28`) — can attach to a `State` parent | **PASS (improved)** | Brief allowed both shapes; v4 supports either. |
| Inspector code editor | `<textarea>` with manual tab handling (`uml-state-code-block-update.tsx:25-43, 78-104`) | `MuiTextField multiline` per `StateCodeBlockEditPanel.tsx` (out of PC-5 scope but cross-checked) | **PARTIAL** | v3 manually inserted a `\t` on Tab keydown; v4 falls back to MUI's default Tab-as-focus-shift. Same brief gap as in **GAP-2** (no CodeMirror). |
| Per-element delete in inspector | v3 had a `<TrashIcon>` (`uml-state-code-block-update.tsx:128-130`) | None — canvas-level delete | **PASS** | Aligns with v4 convention. |
| Header colour | v3 painted header in `element.strokeColor` (`uml-state-code-block-component.tsx:79`) | Same — `fill={strokeColor}` (`StateCodeBlock.tsx:77`) | **PASS** | |

---

## 4. `StateTransition` edge

### 4.1 Data-field parity (already audited at parity-final)

| Field | v3 source | v4 location | Status |
|---|---|---|---|
| `name` | `UMLRelationshipCenteredDescription.name` | `data.name` (`versionConverter.ts:1770`) | **PASS** |
| `guard` | `IUMLStateTransition.guard: string` (`uml-state-transition.ts:9`) | `data.guard` (`versionConverter.ts:1783`) | **PASS** |
| `params` | `IUMLStateTransition.params: { [id]: string }`; legacy shapes `string` / `string[]` accepted (`uml-state-transition.ts:21-31`) | `data.params: { [id]: string }`; same legacy normalisation (`versionConverter.ts:1746-1757`) | **PASS** |
| `code` (BESSER) | Not in v3 model — was carried in `params` (`{0: …, 1: …}`) historically | `data.code` (`versionConverter.ts:1784`) | **PASS** |
| `eventName` (BESSER) | Not in v3 model — was carried in `name` historically | `data.eventName` (`versionConverter.ts:1785`) | **PASS** |

### 4.2 Inspector form parity (PC-5 focus)

| Brief item (Round 1 #15/#16) | v3 surface | v4 surface | Status | Lines |
|---|---|---|---|---|
| `name` | `Textfield` (`uml-state-transition-update.tsx:142`) | `MuiTextField` (`StateMachineDiagramEdgeEditPanel.tsx:71-78`) | **PASS** | |
| `code` | n/a (v3 had no dedicated code field; brief lists `name, code, eventName` for v4) | `MuiTextField multiline` (`StateMachineDiagramEdgeEditPanel.tsx:96-106`) | **PASS** | But see **GAP-2**: not CodeMirror. |
| `eventName` | n/a (v3 had no dedicated event field) | `MuiTextField` (`StateMachineDiagramEdgeEditPanel.tsx:88-95`) | **PASS** | |
| **Flip action** | `<Button color="link" onClick={() => this.props.flip(element.id)}><ExchangeIcon /></Button>` (`uml-state-transition-update.tsx:131-133`) | **Not present** | **GAP-1 MEDIUM** | Round 1 #15 explicitly asks for it. `ClassEdgeEditPanel.tsx` and `AgentDiagramEdgeEditPanel.tsx` ship it under SA-2.2 #26. |
| **Color editor** | `<ColorButton>` toggle + `StylePane lineColor textColor` (`uml-state-transition-update.tsx:130, 172-178`) | **Not present** | **GAP-1 MEDIUM** | Round 1 #16. Same parity rationale as flip. |
| `guard` (extra v3 field) | `Textfield` (`uml-state-transition-update.tsx:146`) | `MuiTextField` (`StateMachineDiagramEdgeEditPanel.tsx:79-87`) | **PASS** | Brief did not list it but v4 keeps it. |
| `params` add/remove | `ParamContainer` rows + `addParam` / `removeParam` (`uml-state-transition-update.tsx:91-113, 155-170`) | `Stack` rows + `addParam` / `removeParam` (`StateMachineDiagramEdgeEditPanel.tsx:51-67, 122-145`) | **PASS** | Same numeric-key allocation strategy (max+1). v4 normalises legacy shapes via converter. |
| Per-edge delete button | `<Button onClick={() => this.props.delete(element.id)}><TrashIcon /></Button>` (`uml-state-transition-update.tsx:134-136`) | None — canvas-level delete | **PASS** | Aligns with v4 convention. |
| `code` edited as Python with syntax highlighting | n/a in v3 | Plain `TextField multiline` | **GAP-2 MEDIUM** | See gap details below. |

---

## 5. Gap details

### GAP-1 (MEDIUM) — Edge inspector missing flip + color editor

- **Surface**: `StateMachineDiagramEdgeEditPanel.tsx`
- **What v3 had** (`uml-state-transition-update.tsx`):
  - flip button (line 131-133, dispatches `UMLRelationshipRepository.flip`)
  - color button (line 130) opening `StylePane` with `lineColor` + `textColor` (line 172-178)
- **What v4 has**: only `name`, `guard`, `eventName`, `code`, `params`. No flip. No color editor.
- **What the Round 1 brief requested**: #15 flip, #16 color editor.
- **How sister panels handle it**: `ClassEdgeEditPanel.tsx` and
  `AgentDiagramEdgeEditPanel.tsx` both register `flip` action +
  `EdgeStyleEditor` (per the SA-2.2 #26 backfill). The SA-3 edge panel
  pre-dates that pass and was never updated.
- **Severity**: MEDIUM. Functional regression for users who previously
  flipped transitions; cosmetic regression for line/text colour. Both
  features are first-class on v3 sibling edges.

### GAP-2 (MEDIUM) — `code` fields use plain TextField, not CodeMirror

- **Surfaces**:
  - `StateBodyEditPanel.tsx:88-98` (`code` for `StateBody` /
    `StateFallbackBody`)
  - `StateMachineDiagramEdgeEditPanel.tsx:96-106` (transition `code`)
  - `StateCodeBlockEditPanel.tsx` (cross-checked, out of PC-5 scope)
- **What the brief promised**: "CodeMirror (Python)" for body/code-block
  authoring.
- **What v4 ships**: MUI `TextField multiline minRows=3`, no syntax
  highlighting, no language indicator, no Python-aware tab handling.
  v3 also lacked CodeMirror — `uml-state-code-block-update.tsx:25-43`
  was a styled `<textarea>` with manual `\t` insertion (`:78-104`).
- **Severity**: MEDIUM. v4 is **no worse than v3 in functional terms**
  (v3 also had no CodeMirror), but v3's textarea at least preserved
  Tab-as-tab; v4's MUI default loses Tab to focus-shift. Brief
  expectation is unmet on both surfaces.

### GAP-3 (LOW) — `kind` round-trips but doesn't render on canvas

- **Surface**: `StateBody.tsx:50-58`, `StateFallbackBody.tsx:51-60`.
- **Symptom**: User picks "entry" in `StateBodyEditPanel` (line 62-77),
  sees no canvas change. Value is persisted (`versionConverter.ts:964`)
  and survives round-trip, but the rendered text shows only
  `data.name`.
- **What the brief implies**: a `kind` picker is meaningful only if the
  canvas surfaces the choice (e.g. `entry / setup()` formatted as the
  label).
- **What v3 had**: nothing — `kind` didn't exist as a concept; v3 only
  separated `StateBody` from `StateFallbackBody` by element type.
- **Severity**: LOW. Authoring works; persistence works; only the
  canvas affordance is missing. **N/A in v3 — pure brief
  underdelivery.** Trivial fix: prefix `name` with `${kind} / ` when
  `data.kind` is set, mirroring the pattern in
  `StateBodyEditPanel.tsx:18`'s docstring suggestion.

---

## 6. Cross-check with `versionConverter.ts`

| Direction | Field | Line | Behaviour |
|---|---|---|---|
| v3 → v4 | `StateBody.code` | `:963` | `...(e.code !== undefined && { code: e.code })` |
| v3 → v4 | `StateBody.kind` | `:964` | `...(e.kind !== undefined && { kind: e.kind })` |
| v3 → v4 | `StateFallbackBody.{code,kind}` | shared with `StateBody` (`:955-967`) | identical |
| v3 → v4 | `StateCodeBlock.{code,language}` | `:991-998` | `code ?? ""`, `language ?? "python"` |
| v3 → v4 | `StateTransition.params` (3 legacy shapes) | `:1746-1757` | normalised to dict |
| v3 → v4 | `StateTransition.guard` | `:1783` | `...(r.guard && { guard: r.guard })` |
| v3 → v4 | `StateTransition.code` | `:1784` | `...(r.code && { code: r.code })` |
| v3 → v4 | `StateTransition.eventName` | `:1785` | `...(r.eventName && { eventName: r.eventName })` |
| v4 → v3 | `parentId` → `owner` for state-machine children | `:2133` | `owner: node.parentId ?? null` |
| v4 → v3 | `kind` passthrough on `StateBody`/`StateFallbackBody` | `:2238-2255` (and the equivalent agent block at `:1063`) | preserved |

All round-trip operations are symmetric for the three BESSER-extra
fields, confirming GAP-3 is render-only.

---

## 7. Visual-shape parity (parent + bodies)

(For completeness — already covered in `parity-final/stateMachineDiagram.md` §5.)

| Visual element | v3 | v4 | Match |
|---|---|---|---|
| State outer rounded rect | `ThemedRect rx={8}` | `<rect rx=8>` | Yes |
| State header background | `ThemedRect height=40 (50 w/ stereotype)` | `LAYOUT.DEFAULT_HEADER_HEIGHT[_WITH_STEREOTYPE]` | Yes |
| Divider line below header | `ThemedPath d="M 0 headerHeight H width"` (only if `hasBody`) | `<line>` always painted | Yes (v4 paints unconditionally; harmless) |
| Body row | `ThemedRect width=100% height=100%` + `<text x={10}>` | `<rect>` + `<text x={10} y=h/2+5>` | Yes |
| Fallback-body row | identical to body row | identical to body row + italic | Yes (v4 italics; v3 didn't) |
| Code block — header bar | `ThemedRect fill=strokeColor` + `text "Python"` | `<rect fill=strokeColor>` + `<text>{language}` | Yes (v4 reads from `data.language`) |
| Code block — body | `<foreignObject><div><div>line</div></div></foreignObject>` | identical | Yes |
| Transition edge — line | `Path strokeColor` | `BaseEdge strokeColor` from `getCustomColorsFromDataForEdge` | Yes |
| Transition edge — label | `name [guard]` middle label | `name [guard]` middle label (`StateMachineDiagramEdge.tsx:124-128`) | Yes |

---

## 8. Summary table

| # | Severity | Surface | Description | Fix location |
|---|---|---|---|---|
| GAP-1 | MEDIUM | inspector | `StateMachineDiagramEdgeEditPanel.tsx` lacks flip action and color editor (Round 1 #15/#16; sister panels `ClassEdgeEditPanel` and `AgentDiagramEdgeEditPanel` already ship them) | `library/lib/components/inspectors/stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx` |
| GAP-2 | MEDIUM | inspector | Body / code-block / edge `code` fields are plain `TextField multiline`, not the CodeMirror Python editor the brief promised | three panels in `library/lib/components/inspectors/stateMachineDiagram/` |
| GAP-3 | LOW | canvas render | `data.kind` round-trips but is not painted by `StateBody.tsx` / `StateFallbackBody.tsx` (canvas shows only `data.name`) | `library/lib/nodes/stateMachineDiagram/StateBody.tsx`, `…/StateFallbackBody.tsx` |

No data-shape gaps. No missing element / edge types. No constraint
regressions. Round-trip closed for all five BESSER-extra fields.
