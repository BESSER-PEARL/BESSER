# PC-7 — AgentState / AgentStateBody / AgentStateFallbackBody

Read-only audit of the `AgentState` parent + its two body kinds across the
v3 fork (`packages/editor/...`) and the new lib (`packages/library/...`).

Verdict: **PARTIAL PARITY — three behavioural regressions on the body
inspector that need follow-up tickets before cutover.**

The five-mode reply picker, the per-mode editors, and the shared RAG/DB
field surface are all present in `AgentStateEditPanel.tsx` +
`AgentStateBodyEditPanel.tsx` + `RagDbFields.tsx`, and the wire fields
round-trip via the migrator. The gaps below are all on the inspector
layer — node rendering parity is fine.

## Sources audited

v3 (the "old fork"):

- `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/agent-state-diagram/agent-state/agent-state.ts` (model class, 117 LoC)
- `.../agent-state/agent-state-component.tsx` (SVG render, 99 LoC)
- `.../agent-state/agent-state-member.ts` (shared body model, 144 LoC)
- `.../agent-state/agent-state-update.tsx` (the load-bearing v3 update form, **969 LoC** — close enough to the brief's "~960")
- `.../agent-state-body/agent-state-body.ts` (3 LoC stub on `AgentStateMember`)
- `.../agent-state-body/agent-state-body-update.tsx` (text-row sub-editor, 56 LoC)
- `.../agent-state-fallback-body/agent-state-fallback-body.ts` (3 LoC stub)

The brief points at `packages/editor/src/main/scenes/update-pane/versions/agent-state-update.tsx` — that path does not exist in this checkout. The actual v3 form lives at the path above (`.../packages/agent-state-diagram/agent-state/agent-state-update.tsx`) and `scenes/update-pane/versions/` contains only `update-pane-fixed.tsx` / `update-pane-movable.tsx`. Treating the in-package form as the source of truth (line counts and prop signatures match the brief).

v4 (the new lib):

- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/agentDiagram/AgentState.tsx` (125 LoC)
- `.../nodes/agentDiagram/AgentStateBody.tsx` (91 LoC)
- `.../nodes/agentDiagram/AgentStateFallbackBody.tsx` (80 LoC)
- `.../components/inspectors/agentDiagram/AgentStateEditPanel.tsx` (607 LoC, `AgentState` parent inspector)
- `.../components/inspectors/agentDiagram/AgentStateBodyEditPanel.tsx` (188 LoC, body-only popover)
- `.../components/inspectors/agentDiagram/RagDbFields.tsx` (234 LoC, shared RAG/DB sub-form)

## 1. Five-radio reply-mode picker — ✅ present (control swapped to Checkbox)

v3 `agent-state-update.tsx` lines 297-411 defines five `<input type="radio">` rows under `name="actionType"`:

| Value          | Label                |
| -------------- | -------------------- |
| `textReply`    | Text Reply           |
| `LLM`          | LLM automatic reply  |
| `rag`          | RAG reply            |
| `dbReply`      | DB action            |
| `pythonCode`   | Python Code          |

The five-row block is repeated (lines 543-657) for the **fallback** action under `name="fallbackActionType"`.

v4 `AgentStateEditPanel.tsx:65-73` declares the same five modes as
`REPLY_MODES` (`text` / `llm` / `rag` / `db_reply` / `code`) and renders
them in two near-identical blocks at lines 555-575 (action) and 583-603
(fallback). All five are present, the labels match v3 verbatim, and
`setMode(...)` (lines 222-286) replicates v3's "delete siblings of the
wrong reply type, create one of the new type if absent" behaviour.

⚠️ **Control mismatch (P2):** v4 uses `<Checkbox>` instead of `<Radio>`
(`AgentStateEditPanel.tsx:564-569` and `:592-597`). The five options are
mutually exclusive in both UIs (only one mode is ever active because
`setMode` deletes the sibling bodies), but a checkbox group conveys
"multi-select" semantics, which a screen reader will read out as such.
v3 used a true radio group. Swap to MUI `<RadioGroup>` + `<Radio>` to
match v3 semantics — pure UI fix, no data shape change.

## 2. Per reply-mode editors

### 2.1 `text` — ✅ functional parity, minor UX delta

v3 (`agent-state-update.tsx:417-469`) renders one `Textfield` per text
body via `BotBodyUpdate`, plus an empty "+ submit-to-create" Textfield
for adding a new one. Each row carries its own `ColorButton` + trash
icon (see `agent-state-body-update.tsx`).

v4 (`AgentStateEditPanel.tsx:294-342`) renders one `MuiTextField` per
existing text body with a delete `IconButton`, and a `+ add text body`
caption-link that creates a new empty text body on click.

Functional parity ✅. UX deltas: v4 has no per-row colour picker
(`StylePane`/`ColorButton`) and no Tab-key focus-shift between rows
(v3's `bodyRefs` array). Both deltas are likely acceptable simplifications
for v4's Material UI surface, but flag them as **P3** if pixel parity
matters.

### 2.2 `llm` — ✅ TextField present (multiline upgrade)

v3 (`agent-state-update.tsx:530-535`) just shows the helper text
*"An automated response will be generated."* with no editable field —
the LLM body's `name` is hard-coded to `"AI response 🪄"`
(`agent-state-update.tsx:334`).

v4 (`AgentStateEditPanel.tsx:345-366`) shows a multiline `MuiTextField`
labelled "System prompt" wired to the LLM body's `name`. This is a
**superset** of v3 (the brief's "TextField for text/llm" calls for
exactly this), so this counts as ✅. No regression — the v3 hard-coded
default is now editable.

### 2.3 `rag` — ⚠️ dropdown sourced correctly but always free-text on empty pool

v3 (`agent-state-update.tsx:230-237` + `:499-526`) builds
`ragDatabaseNames` from
`Object.values(elements).filter(el => el.type === AgentElementType.AgentRagElement && typeof el.name === 'string').map(el => el.name.trim())`.
When the list is non-empty it renders an apollon `Dropdown`; when the
list is empty it falls back to the helper line *"No RAG databases
available. Create one from the palette first."* (no free-text field).

v4 `AgentStateEditPanel.tsx:140-152` builds the equivalent
`ragDatabaseOptions` from `nodes.filter(n => n.type === 'AgentRagElement')`.
The mode renders `RagDbFields` (with `showRag` only) plus an extra
optional "RAG prompt" textbox.

`RagDbFields.tsx:79-121`: when `ragDatabaseOptions.length > 0` it shows
an MUI `Select` (matches v3). When the list is empty, it falls back to a
**free-text** `MuiTextField` for `ragDatabaseName`. v3 deliberately did
NOT allow free-text in the empty case — it forced the user to create a
sibling `AgentRagElement` first. This is a behavioural change with mild
data-quality risk (typed names that don't match any sibling RAG element
will silently be accepted).

⚠️ **Gap (P2):** when `ragDatabaseOptions` is empty, render the v3
helper line + a "Create RAG element" affordance instead of a free-text
input — see `RagDbFields.tsx:103-119`.

### 2.4 `db_reply` — ✅ full DB-action editor present

v3 `renderDbReplyEditor` (`agent-state-update.tsx:809-943`) exposes:

- `dbSelectionType` (`Default` / `Custom`) — Dropdown
- `dbCustomName` — Textfield, shown only when `selectionType === 'custom'`
- `dbOperation` (`any` / `select` / `insert` / `update` / `delete`) — Dropdown
- `dbQueryMode` (`llm_query` / `sql`) — radio pair
- `dbSqlQuery` — multiline Textfield, shown only when `queryMode === 'sql'`
- LLM helper line shown when `queryMode === 'llm_query'`

v4 `RagDbFields.tsx:123-231` renders the exact same six fields, in the
same conditional order, with the same vocabulary. Mapping is 1:1 — verified.

⚠️ **Minor delta (P3):** v3 uses a *radio pair* for `dbQueryMode`; v4
uses a `Select` (`RagDbFields.tsx:179-201`). v3's SQL editor is a plain
multiline `Textfield`; v4's is a CodeMirror Python instance
(`:203-223`). Both are functional supersets but the SQL editor uses
Python lexing, which highlights `SELECT` as an unknown identifier.
Switch CodeMirror's extension to `sql()` from `@codemirror/lang-sql` for
correct lexing.

### 2.5 `code` — ✅ CodeMirror Python present

v3 (`agent-state-update.tsx:471-498`) wraps `react-codemirror2`
(`Controlled as CodeMirror`) with `mode: 'python'`, `theme: 'material'`,
`lineNumbers: true`, `tabSize: 4`, `indentWithTabs: true`, sized via
`ResizableCodeMirrorWrapper`. Default body text:
`"def action_name(session: AgentSession):\n\n\n\n\n"`.

v4 `AgentStateEditPanel.tsx:368-407` uses `@uiw/react-codemirror` +
`@codemirror/lang-python` with `lineNumbers: true`, `tabSize: 4`,
`indentOnInput: true`. Default text matches v3 verbatim
(`:77-78 CODE_BODY_DEFAULT`). The editor writes to BOTH `code` and `name`
on every change (`:395-397`) — that's the v3→v4 split (v3 stored code on
`name` only) with backwards-compat reads (`:380-383`). Round-trip safe.

⚠️ **Minor delta (P3):** no `material` theme — v4 uses CodeMirror's
default light theme. No `indentWithTabs: true`; v4 indents with spaces.
Both are visual / formatting deltas, not data-loss issues.

## 3. Fallback bodies — ✅ separate radio block present

v3 dedicates lines 540-761 to the fallback section: a duplicate
five-radio block under `name="fallbackActionType"` plus duplicated
per-mode editors that target the `AgentStateFallbackBody` children
instead of `AgentStateBody`. The block is visually separated by a
`<Divider />` (`:537-539`).

v4 `AgentStateEditPanel.tsx:578-604` mirrors this exactly: a
`<DividerLine />` followed by a `Typography subtitle2` heading "Agent
Fallback Action", a duplicate `REPLY_MODES.map(...)` (now keyed
`fallback-${m.value}`), and the same `renderBodyEditor("AgentStateFallbackBody")`
helper. The two sections share `setMode` / `getActiveMode` /
`createChild` / `removeChildrenWhere` parameterised on the `kind`
discriminator — no logic duplication. ✅.

## 4. Style fields (italic / underline / stereotype) — ✅

v3 surfaces `italic` / `underline` / `stereotype` only via the model
class defaults (`agent-state.ts:43-45`); the v3 update form does NOT
expose editors for these — they're set at construction time. The state
component reads them at render (`agent-state-component.tsx:31-58`) so
they're rendered correctly when set.

v4 `AgentStateEditPanel.tsx:506-548` upgrades this: a stereotype `Select`
(with `«initial»` / `«final»` / `«intent»` options + "— none —"), and
two `FormControlLabel` checkboxes for italic / underline. This is a
**superset** of v3 — counts as ✅, no regression. The render side
(`AgentState.tsx:35-107`) reads `stereotype` / `italic` / `underline`
from `data` and renders them identically to v3.

## 5. AgentStateBody panel: same body editor + 6 RAG/DB fields — ⚠️ partial

The brief asks for `AgentStateBodyEditPanel.tsx` to expose the same body
editor as the parent inspector plus the six RAG/DB fields
(`ragDatabaseName`, `dbSelectionType`, `dbCustomName`, `dbQueryMode`,
`dbOperation`, `dbSqlQuery`).

`AgentStateBodyEditPanel.tsx` provides:

- `name` (line 100-107): single-line `MuiTextField` ✅
- `kind` (line 109-128): MUI Select with values `entry` / `do` / `exit` /
  `on` / "— none —" — **EXTRA in v4, not in v3 model**. v3
  `AgentStateMember` (`agent-state-member.ts:33-39`) does NOT have a
  `kind` field. This appears to be a UML-state-machine carryover that
  leaks here. Storing `kind` on an agent body has no effect on
  generation or round-trip.
- `replyType` (line 130-146): MUI Select with the seven values
  `text` / `image` / `json` / `llm` / `code` / `rag` / `db_reply`. v3
  `AgentStateMember.replyType` only ever held `text` / `llm` / `rag` /
  `db_reply` / `code` (set by `agent-state-update.tsx:296-411`). The
  extra `image` and `json` values are not used anywhere else in v3 and
  produce dangling data.
- `code` editor (`:148-168`): CodeMirror Python, shown when
  `replyType === 'code'` ✅
- RAG/DB fields (`:170-185`): renders `RagDbFields` for `rag` /
  `db_reply`. Provides all six fields ✅

⚠️ **Gap A (P1, regression):** `kind` field is exposed but is not part
of the agent-state metamodel. Either drop the field from this panel or
confirm with the metamodel owner that it was intentionally added (and
update `AgentStateBodyNodeProps` + the migrator).
`AgentStateBodyEditPanel.tsx:35-41` and `:109-128`.

⚠️ **Gap B (P2):** `replyType` Select includes `image` and `json`,
which v3 never produces and `AgentStateEditPanel`'s `REPLY_MODES`
doesn't list. Selecting either value puts the body into an unreachable
state for the parent inspector — picking `image` here will then read
back as `text` in the parent (because `getActiveMode` only checks the
five canonical values). Trim to the five canonical reply types.
`AgentStateBodyEditPanel.tsx:43-51`.

⚠️ **Gap C (P2):** body inspector does not expose the parent's
five-radio mode picker — switching from `db_reply` to `text` in the
body inspector is a one-step Select change with no sibling cleanup,
unlike `setMode` in the parent which deletes siblings of other types.
The two inspectors will desync if both are open. Consider: either
delegate `replyType` changes to a shared `setMode` helper or restrict
the body inspector to read-only `replyType` (force the user to flip
modes from the parent).

## 6. Render-side parity — ✅

`AgentState.tsx` renders the rounded rect, the optional «stereotype»
header line, the name with italic/underline, and the divider line at
`headerHeight` — matches v3's `agent-state-component.tsx` 1:1 modulo
the SVG-image bot icon at `:59-65` (a hard-coded base64 PNG dropped at
the top-right). v4 omits the bot icon entirely. **Tiny visual delta
(P3)** — re-add the icon if pixel-parity matters.

`AgentStateBody.tsx`: renders the body name on a single line, with a
`foreignObject`-wrapped `<div>` for `replyType === 'code'` (monospace,
`whiteSpace: 'pre'`) — matches v3's
`agent-state-member-component.tsx` code path semantically.

`AgentStateFallbackBody.tsx`: renders the same shape as the body but
with `opacity={0.85}` and a dashed top divider line + italic name. This
visually distinguishes fallback from primary, where v3 relied on the
parent's `dividerPosition` to draw the divider once
(`agent-state-component.tsx:94-96`). Both produce a divider line; the
v4 approach is cleaner now that bodies are first-class react-flow nodes.

## Top 3 gaps (priority-ordered)

1. **(P1) Body inspector exposes a `kind` field that doesn't exist in
   the v3 metamodel** — `AgentStateBodyEditPanel.tsx:35-41, 109-128`. Pure
   bleed-through from a state-machine inspector template; will silently
   write a no-op field on every body. Drop it or wire it through the
   migrator.
2. **(P2) Body inspector's `replyType` Select offers `image` and
   `json`** — `AgentStateBodyEditPanel.tsx:43-51`. Two of the seven
   options are unreachable from the parent inspector and produce bodies
   that the parent's `getActiveMode` mis-classifies as `text`. Trim to
   the canonical five.
3. **(P2) RAG dropdown silently falls back to free-text when no sibling
   `AgentRagElement` exists** — `RagDbFields.tsx:103-119`. v3 instead
   showed a "Create one from the palette first" helper. The free-text
   path lets users type names that match no real RAG element, which the
   backend will then fail on at generation time. Restore the v3 helper
   line.

Lower-priority items captured inline above (P3): missing `material`
CodeMirror theme, missing per-row colour picker on text bodies,
checkbox-vs-radio control mismatch, missing top-right bot icon,
SQL editor lexed as Python.
