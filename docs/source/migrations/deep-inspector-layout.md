# SA-DEEP-INSPECTOR-LAYOUT — Inspector Panel Audit

Read-only audit of every inspector body under
`packages/library/lib/components/inspectors/` (submodule HEAD
`fc979af`), comparing UX density, section organisation, field types,
column widths, settings-disclosure UX, button placement, and
cross-inspector consistency against the v3 source-of-truth in
`packages/editor/src/main/components/update-pane/` (host) and
`packages/editor/src/main/packages/**/*-update.tsx` (per-element bodies).

> **Source-of-truth path note.** The brief referenced
> `packages/editor/src/main/scenes/update-pane/versions/`, which does not
> exist on this submodule HEAD. v3 update-pane lives at
> `packages/editor/src/main/components/update-pane/` (`update-pane.tsx`,
> `versions/update-pane-fixed.tsx`,
> `versions/update-pane-movable.tsx`). The per-element form bodies
> rendered inside that pane are the `*-update.tsx` files under
> `packages/editor/src/main/packages/**/`. This audit treats those
> per-element files as the field-arrangement source-of-truth, and the
> three host files as the section-padding / divider / header
> source-of-truth.

> **Style harmonisation rules used below**
> - v3 used styled-components with theme tokens (`theme.color.gray`,
>   `theme.color.primary`). v4 uses MUI components plus CSS variables
>   (`var(--besser-gray, #e9ecef)`, `var(--besser-primary)`). Per-row
>   borders, gear-button colour, "+ add" link colour all reference these
>   variables — Tailwind classes do not appear inside inspector bodies
>   (consistent with the editor-package convention `styled-components`
>   inside the editor; in v4-library it is MUI `sx`). Mixing MUI `sx`,
>   CSS variables and inline `style` is the harmonisation pattern.
> - v3 sections were wrapped in `<Section>` (8px vertical padding) with a
>   small uppercase `<SectionHeader>` (11px, opacity 0.6, letter-spacing
>   0.5px) above each block, separated by `<Divider />`. v4 inspectors
>   use `<Typography variant="h6">` or `variant="subtitle2"` or
>   `variant="caption"` for section headers — inconsistent across
>   panels. Most panels use `gap: 1` on the parent `<Box>` instead of
>   per-section padding.
> - v3 add-row was a `Textfield` with `outline` and a `+ placeholder`.
>   v4 has split this into either an explicit `+` `IconButton` next to
>   the section header, an `Add` `<Button>` with text label, or an
>   inline placeholder textfield — three different patterns coexist.

---

## 1. classDiagram

### 1.1 ClassEditPanel.tsx (1200 LoC)

Source-of-truth (v3): `uml-classifier-update.tsx` (507 LoC) + child rows
`uml-classifier-attribute-update.tsx` (456 LoC),
`uml-classifier-method-update.tsx` (499 LoC).

Status: **largely matches v3 with explicit annotations**, but some
density and layout deviations.

**Findings (line numbers reference the v4 inspector unless prefixed
`v3:`)**:

- **Section headers** — uses `Typography variant="h6"` (lines 1068,
  1100, 1149) where v3 used a small uppercase `SectionHeader`
  (`v3 uml-classifier-update.tsx:56-62`, 11px / 0.5px-tracked /
  opacity 0.6). `h6` is bigger, denser at the top and pushes the
  attribute table down.
- **AttributeRow** (lines 187-382) — v3 had `+/-` reorder buttons in
  a left gutter (`v3:64-91, 254-274`); v4 has dropped them entirely.
  The user can no longer reorder attributes from this panel. Major
  user-visible regression.
- **Visibility column** width: `minWidth: 70` (line 207); v3 used
  `min-width: 44px` (`v3 uml-classifier-attribute-update.tsx:38`).
  v4 is too wide for the four single-character symbols.
- **Type column** width: `minWidth: 110` (line 235); v3 was
  `min-width: 80px` (`v3 :43`). v4 again wider than necessary.
- **Per-row gear button** (lines 266-277) — adds a third action
  icon on the row (gear + delete) plus a fourth (custom-type
  textfield) in a column. v3 hung the flag/default editor inside
  `StylePane` opened from the `ColorButton` (`v3 :432-451`), so the
  inline row was visibly less busy: visibility + name + type +
  color + delete (5 widgets), while v4 row has visibility + name +
  type + gear + delete (5 widgets), but also exposes a
  custom-type textfield + flag-checkbox row + default-value
  textfield once expanded — total 8 widgets at full disclosure
  with no nested popover, dragging the row to ~80px tall.
- **No ColorButton on rows** — v3 attribute rows offered per-row
  fill/text colour overrides (`v3 :240`, `:428`, `:433-451`). v4 has
  no per-row colour controls (only the parent class picks colour via
  `NodeStyleEditor`).
- **Settings disclosure** — on attribute rows the gear icon (line
  267) opens flags + default value below; on method rows (line 481)
  the same gear opens parameters + impl + code editor. Same icon,
  different content; no header label on the disclosure block.
  Compared to v3 which used the `ColorButton` (paint-bucket) for
  the disclosure, v4's choice of the pencil/edit icon is fine but
  not annotated for screen readers (no `aria-label`).
- **Add-row UX** — both `+ Add attribute` (lines 1122-1136) and the
  per-section `+` icon-button (line 1104) coexist. v3 only had the
  textfield (`v3 :305-342`). The duplicate "+" button is redundant.
- **Method row return-type** (lines 498-542) lives on its own row
  below the name; v3 method rows kept return-type inline as part of
  the method-signature textfield (`v3 uml-classifier-method-update
  .tsx:363-372`). v4's design is arguably clearer, but no longer
  matches the placeholder hint of v3 (`method(param: type):
  returnType`). Acceptable deviation.
- **CodeMirror block height** (line 707) `minHeight: 80` for code
  rows; v3 used a resizable wrapper `min-height: 150px` /
  `max-height: 400px` (`v3 :92-104`). v4 is too short — fits ~5
  lines of Python; user has to scroll inside CM for any non-trivial
  body.
- **Metadata fields** (lines 1068-1097) — v3 surfaced
  description/uri/icon inside the `StylePane` (a popover); v4 puts
  them inline at the top of the body. This burns ~120px of
  vertical real estate above attributes for fields that are usually
  empty. Recommend collapsing these behind a disclosure.
- **No flip/delete shortcut** at the top of the panel (v3
  `uml-classifier-update.tsx:209-213` had a top-right trash). v4
  relies on the canvas/right-rail delete; OK.

### 1.2 ClassEdgeEditPanel.tsx (261 LoC)

Source-of-truth (v3): `uml-class-association-update.tsx` (205 LoC).

Status: **close parity, small layout drift**.

- **Top header** present in v3 (`v3 :87-89` `<Header>popup.association
  </Header>`); v4 only has the colour bar via `EdgeStyleEditor`
  (lines 137-148) with `label="Association"`. Acceptable mapping.
- **Type-picker width** unconstrained (line 168 `<Select size="small"`
  without `sx`) — full-width in v4; v3 wrapped in a `Dropdown`
  styled to a stable column. Minor.
- **Source/Target labels** (lines 184-186, 222-224) use
  `Typography variant="subtitle2" fontWeight="bold"`; v3 used the
  source/target node's actual name as the section header
  (`v3 :149`, `:168`). v4 lost that affordance — a user editing two
  associations can't tell which class each end refers to. Minor
  regression.
- **Multiplicity / role rows**: caption widths `minWidth: 80`
  (lines 188, 209, 226, 246); v3 used a `<Body>` inline label with
  `marginRight: '0.5em'` and let the textfield flex. v4 burns 80px
  on the label column, narrowing the field unnecessarily.

### 1.3 ClassOCLConstraintEditPanel.tsx (77 LoC)

Source-of-truth (v3): `uml-class-ocl-constraint-update.tsx` (referenced).

Status: **minimal, but fields trimmed below v3**. v4 hides `name` /
`kind` (per the inline comment lines 11-15) — intentional. The two
remaining textfields use plain `placeholder` instead of the consistent
`label` pattern used by sibling panels (e.g. `ClassEdgeEditPanel`
line 158). Inconsistent.

---

## 2. objectDiagram

### 2.1 ObjectEditPanel.tsx (409 LoC)

Source-of-truth (v3): `uml-object-name-update.tsx` (449 LoC).

Status: **deliberately simplified; some v3 features dropped**.

- **No methods section** — intentional and documented (lines
  404-406). Correct (objects are instances).
- **No visibility column** on attribute rows (line 42 onwards) — v3
  had no visibility on object rows either (object rows used
  `UMLObjectAttribute` not `UMLClassAttribute`). Parity OK.
- **No type-aware value widget** — comment lines 36-41 say "v3's
  type-aware widgets (date pickers, enum dropdowns, custom-type
  editors) are intentionally dropped". This is the **biggest
  user-visible regression vs v3** in the object panel: a date-typed
  attribute is now a free-text field, an enum-typed attribute won't
  give a literal dropdown. v3's `uml-object-attribute-update.tsx`
  rendered type-aware widgets here. Fix priority: high.
- **Class picker** caption `minWidth: 70` (line 336) — same drift
  as elsewhere; the `class` label is 5 characters but the column
  reserves 70px.
- **Header for "Attributes"** uses `Typography variant="h6"` (line
  372), inconsistent with v3 small-uppercase header style (same
  drift as ClassEditPanel).
- **Add-row affordance** (lines 388-402) — only an inline
  textfield, no `+ Add` icon button next to the section header
  (compare to ClassEditPanel which has both).

### 2.2 ObjectLinkEditPanel.tsx (182 LoC)

Source-of-truth (v3): `uml-object-link-update.tsx` (referenced).

Status: **good parity**.

- **Field stack consistent** with `ClassEdgeEditPanel`
  (color/flip/divider/name/picker pattern).
- **Caption widths** for `association` and helper text (line 156)
  use `minWidth: 80` and `minWidth: 70` — drift between sibling
  panels (this one says 80 like edge, ClassEditPanel says 70 like
  state). Should pick one project-wide.

---

## 3. stateMachineDiagram

### 3.1 StateEditPanel.tsx (115 LoC)

v3: `uml-state-update.tsx` (227 LoC).

Status: **minimal port**. Render order is unconventional:
`NodeStyleEditor` → divider → stereotype → italic/underline →
`name` (lines 105-112). v3 kept name first. **Fix**: hoist the name
field above the stereotype/style controls so the most-edited field is
top-of-form. Italic/underline checkboxes shown as a single Stack with
no parent label — v3 grouped them under a `Style` header.

### 3.2 StateBodyEditPanel.tsx (117 LoC)

v3: `uml-state-body-update.tsx`.

Status: **OK**, but `kind` dropdown (lines 64-82) covers entry/do/exit
plus a free-form transition; v3's body-kind was a typed enum on the
element subclass. Storing `kind` on `data.name` inline (per comment
lines 16-19) is a workaround that future round-trips may not preserve
cleanly. Code editor block 80px-tall (line 99) is too small.

### 3.3 StateActionNodeEditPanel.tsx (68 LoC)

Status: **uses plain `MuiTextField multiline minRows=4`** for code (line
58-65), not CodeMirror. Inconsistent with sibling state-machine
inspectors that use CodeMirror (`StateBodyEditPanel`,
`StateCodeBlockEditPanel`, `StateMachineDiagramEdgeEditPanel`). Same
field, two widgets across the family. Fix: switch to CodeMirror.

### 3.4 StateCodeBlockEditPanel.tsx (100 LoC)

Status: **OK**. CodeMirror block 160px (line 82) — bigger than the
80px used in StateBody/StateMachineDiagramEdge, which is correct for a
dedicated code-block node. The language dropdown exposes `bal` even
though only Python highlighting is wired (line 87 `extensions=[python()]`).
v3 limited language to Python.

### 3.5 StateLabelEditPanel.tsx (74 LoC)

Status: **correctly short-circuits** on `NON_UPDATABLE_TYPES`
(lines 23-28, 41) per v3 invariant. Good.

### 3.6 StateMachineDiagramEdgeEditPanel.tsx (216 LoC)

v3: `uml-state-transition-update.tsx` (200 LoC).

Status: **good parity, slightly busier than v3**.

- v3 had Name → Guard → Parameters; v4 adds `event name` (line 145)
  and `code` (lines 152-174) as primary fields. These are SA-3
  brief additions, not v3 parity. They are documented inline.
- Parameters use a `+ add` text-link (line 187) instead of v3's
  `<Button>Add</Button>` (`v3 :151-153`). Inconsistent with
  `AgentIntentEditPanel` which uses `Button variant="text"`.
- Code editor 80px tall (line 159) — fine for a transition action.

### 3.7 StateMergeNodeEditPanel.tsx (174 LoC)

v3: `uml-state-merge-node-update.tsx` (189 LoC).

Status: **decisions table is good**. Each row has `name` text → `→`
arrow → target Select → delete (lines 122-171). Matches v3.
Caveats:
- No "+ add decision" button (only existing edges are listed); v3
  let you create a new decision-edge inline.
- Header pattern uses `Typography variant="caption"` (line 116) which is
  a smaller, lighter header than ClassEditPanel's `h6`.

### 3.8 StateObjectNodeEditPanel.tsx (106 LoC)

Status: **minimal**, only edits name + classId. Caption width 70px again.

---

## 4. agentDiagram

### 4.1 AgentStateEditPanel.tsx (550 LoC)

v3: `agent-state-update.tsx` (968 LoC).

Status: **best-in-class port**, with the most v3 parity. Two-section
layout (Agent Action / Agent Fallback Action, lines 495-547) mirrors
v3.

Concerns:
- **Reply-mode picker rendered as a column of `Checkbox`es** (lines
  502-518) rather than radio buttons. v3 used radios for mutual
  exclusion. Checkboxes visually communicate "multi-select" even
  though clicking always sets a single mode (lines 207-258). Should
  use `RadioGroup` for correct semantics.
- Same checkbox pattern repeated for fallback (lines 530-545) —
  same fix.
- CodeMirror block 150px (line 344) is reasonable but no resize
  affordance; v3 had `resize: both` (`v3 uml-classifier-method-update
  .tsx:92-95`).
- Italic/underline live mid-form (lines 470-491) — same complaint as
  StateEditPanel: should be grouped with style controls.

### 4.2 AgentIntentEditPanel.tsx (312 LoC)

Status: **comprehensive consolidated form** (per the SA-UX-FIX-2 B1
note, lines 16-28).

- **Add buttons** use `<Button size="small" variant="text">+ add</Button>`
  (lines 191-197, 232-238). Inconsistent with the `+ add` text-link
  used in `StateMachineDiagramEdgeEditPanel` (line 187) and the
  `IconButton` `+` used in `ClassEditPanel` (line 1104).
- Entity-slot rows (lines 252-308) render four separate
  `MuiTextField`s in a 2×2 grid plus a delete button — visually
  dense; could collapse `entity` / `slot` / `value` behind a "+"
  disclosure when only `name` is set.

### 4.3 AgentRagElementEditPanel.tsx (110 LoC)

Status: **OK**, uses `RagDbFields` (lines 92-97) for shared
DB-action editing. `ragType` exposed as a free-form textfield (lines
99-107) with helper `optional discriminator` — vague placeholder; not
documented elsewhere in the audit doc set.

### 4.4 AgentDiagramEdgeEditPanel.tsx (539 LoC)

v3: `agent-state-transition-update.tsx`.

Status: **most-feature-complete edge inspector**. Toggle-button-group
(lines 290-298) for predefined/custom switch is appropriate.

- Caption widths drift: `minWidth: 90` (line 303), `minWidth: 70`
  (line 425) within the same panel.
- Parameters block uses the `+ add` text-link pattern again.
- Custom condition CodeMirror is 120px tall (line 471) — taller
  than the 80px used in sibling panels, defensible since custom
  conditions are usually multi-statement Python.

### 4.5 AgentDiagramInitEdgeEditPanel.tsx (17 LoC)

Status: **fine**. One-line "no editable fields" note. Could include
the source/target node names for orientation.

### 4.6 AgentIntentBodyEditPanel.tsx (63 LoC)

Status: **OK**. 4-row multiline textarea (line 56) — appropriate.

### 4.7 AgentIntentDescriptionEditPanel.tsx (62 LoC)

Status: **OK**. Identical shape to `AgentIntentBodyEditPanel` minus
the placeholder. The two could share a base component.

### 4.8 AgentIntentObjectComponentEditPanel.tsx (83 LoC)

Status: **redundant with the consolidated form in AgentIntentEditPanel**.
Four stacked `MuiTextField`s with `label`s (lines 48-80). Functional
but cramped — would benefit from a 2-column grid (entity/slot side by
side) like the parent form.

---

## 5. userDiagram

### 5.1 UserModelNameEditPanel.tsx (552 LoC)

v3 routes through the same `UMLObjectNameUpdate` (`v3
uml-object-name-update.tsx`).

Status: **good v3 parity**, with an explicit per-row text-color
picker (lines 253-296). Concerns:

- **Native color input** (line 278-294) — uses an HTML `<input
  type="color">` styled into a circle. Inconsistent with all other
  panels which delegate colour to `NodeStyleEditor` /
  `EdgeStyleEditor`. Pulls the OS native picker; visual treatment
  diverges from MUI. Right-click-to-reset is non-standard UX (line
  273-275).
- **Comparator dropdown** rendered inline only for integer types
  (line 233-251) — correct v3 parity, but no caption so users may
  not realise the `==` button is contextual.
- **Type column** uses `minWidth: 110` (line 222) — same as
  ClassEditPanel. Consistent within library, wider than v3.
- **Description field** (lines 515-524) is a fully-expanded
  multiline textfield in the middle of the form — same gripe as
  ClassEditPanel: should collapse behind a disclosure when empty.

### 5.2 UserModelAttributeEditPanel.tsx (262 LoC)

Status: **mostly redundant** with the per-row form in
UserModelNameEditPanel (the migrator collapses attributes onto the
parent — comment lines 17-21). The standalone form has the same
type-aware value widget set (lines 194-259). Should keep its layout
identical to the per-row form for consistency — which it does, except
the standalone form puts each field on its own labelled row whereas
the in-row form groups name + type + color + delete on one line.

---

## 6. nnDiagram

### 6.1 NNComponentEditPanel.tsx (564 LoC)

v3: `nn-component-update.tsx`.

Status: **schema-driven**, the only inspector that reads its field
list from a config (`getLayerSchema(layerKind)`, line 151).

- **Mandatory vs optional separation** (lines 276-277, 362-396)
  with a "optional attributes" caption header — matches v3 intent.
- **Per-row enable checkbox** (lines 444-450) is appropriate but
  the checkbox sits *inside* the `<Stack>` to the left of the
  caption, with no consistent column width — narrow captions
  shorter than wide ones; the checkboxes don't line up vertically
  across rows.
- **Caption column** (e.g. line 464 `minWidth: 100`) is wider than
  every other inspector, due to NN attribute names being longer
  (`learning_rate`, `kernel_dim`, …). Defensible.
- **Helper text on malformed list values** (lines 552-557) is
  good. Error styling via `error={malformed}` MUI prop — also
  good.
- **No grouping by category** — TensorOp/Pooling/Dataset
  conditional filtering (lines 282-318) reduces the field count,
  but other layers can still surface 8-12 optional fields in one
  flat scroll. v3 grouped optionals under a disclosure
  (`OptionalAttributeRow` per the comment line 274).

### 6.2 NNContainerEditPanel.tsx (102 LoC)

Status: **OK**. Caption width 100px on `entryLayerId` (line 82)
matches the rest of the nnDiagram set. Description field same
inline-multiline pattern as elsewhere.

### 6.3 NNReferenceEditPanel.tsx (104 LoC)

Status: **redundant fields**. Renders a Select for
`referenceTarget` (lines 78-92) **and** a free-text override
field for the same value (lines 93-101). Two controls editing one
field is confusing. The free-text field should appear only when
the user toggles a "manual override" mode.

---

## 7. common

### 7.1 CommentEditPanel.tsx (61 LoC)

v3: `comments-update.tsx`.

Status: **OK**. One multiline textarea bound to `name`; consistent
with v3.

---

## Top 10 actionable cleanups (ranked by user-visible impact)

| # | Inspector(s) | Issue | Intended fix |
|---|---|---|---|
| 1 | `ClassEditPanel` (lines 187-382, 397-740) | Attribute & Method rows lost the v3 reorder gutter (`v3 uml-classifier-update.tsx:64-91, 254-274`) — users cannot reorder rows from the inspector. | Restore a left-side `↑/↓` gutter on each row, mirroring v3's `ReorderRow` / `ReorderControls`. |
| 2 | `ObjectEditPanel` (lines 42-111) | Object attribute values lost type-aware widgets (date / enum / bool / number); all values are now plain text (commented as intentional, lines 36-41). | Re-introduce the v3 widget switch from `UserModelNameEditPanel`'s `AttrRow` (lines 302-355). Reuse it as a shared row. |
| 3 | `AgentStateEditPanel` (lines 502-518, 530-545) | Reply-mode picker uses `Checkbox` for mutually-exclusive options — confusing semantics. | Replace with MUI `RadioGroup` of `Radio` items, single-select. |
| 4 | `ClassEditPanel` (lines 1068-1097) and `UserModelNameEditPanel` (515-524), `NNContainerEditPanel` (71-80) | `description` / `uri` / `icon` fields rendered inline at the top burn ~120px even when empty. | Wrap them in an MUI `Accordion` or behind a "Metadata" disclosure (matches v3 which kept them inside `StylePane`). |
| 5 | `ClassEdgeEditPanel` (lines 184-186, 222-224) | Source / target sections lose the actual node names (v3 `uml-class-association-update.tsx:149, 168`). | Resolve `edge.source` / `edge.target` against `nodes` and render the class name as the section header. |
| 6 | `StateActionNodeEditPanel` (lines 55-65) | Code field is a plain `MuiTextField multiline` while sibling state panels use CodeMirror. | Swap to CodeMirror with `python()` extension matching `StateBodyEditPanel` lines 102-113. |
| 7 | All inspectors | Section-header tag drifts: `Typography variant="h6"` (Class/Object), `subtitle2 fontWeight=bold` (ClassEdge), `caption` (StateMerge/AgentIntent). | Introduce a shared `<InspectorSectionHeader>` (uppercase, 11px, opacity 0.6, letter-spacing 0.5px) mirroring v3's `SectionHeader` (`v3 uml-classifier-update.tsx:56-62`) and apply across the library. |
| 8 | All inspectors | Add-row affordance is inconsistent: `IconButton +` (Class line 1104), `+ add` text-link (StateMachineDiagramEdge line 187), `<Button variant="text">+ add</Button>` (AgentIntent line 192). | Pick one: text-link `+ add` (matches v3 `Button color="link"`). Promote into a shared `<AddRowButton>` component. |
| 9 | `ClassEditPanel` (lines 187-382) | AttributeRow column widths drift from v3: visibility 70px (v3 44px), type 110px (v3 80px). The single-character visibility column is needlessly wide. | Reduce to v3 widths: visibility `minWidth: 44`, type `minWidth: 80`. Cascade to UserModel / Object panels. |
| 10 | `NNReferenceEditPanel` (lines 78-101) | Two controls (Select + free-form override) edit the same `referenceTarget` field. | Show the override only behind an "Advanced" disclosure or replace with a single `Autocomplete` that accepts both list selection and free typing. |

---

## Cross-inspector consistency cheatsheet

| Concern | Class | Object | State | Agent | User | NN |
|---|---|---|---|---|---|---|
| Section header style | `h6` | `h6` | `caption` | `caption`/`subtitle2` | `caption` | `caption` |
| Caption col width | 70 | 70-80 | 70 | 70-110 | 70-110 | 100 |
| Code field widget | CodeMirror | n/a | mixed (CM + plain) | CodeMirror | n/a | n/a |
| "+ add" widget | IconButton | none (textfield) | text-link | mixed | text-link | per-row checkbox |
| Per-row reorder | **missing** | n/a | n/a | n/a | n/a | n/a |
| Per-row colour | none | none | none | none | native input | none |
| Settings disclosure | gear icon | none | none | none | none | per-row checkbox |
| Visibility col width | 70 | n/a | n/a | n/a | n/a | n/a |
| Type col width | 110 | n/a | n/a | n/a | 110 | n/a |
