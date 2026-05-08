# PC-11 — Cross-cutting UX surfaces parity audit

Read-only audit of the migration's three shared UX surfaces:

1. **Right-side `<PropertiesPanel>` ↔ floating `<PopoverManager>`** (mutual
   exclusion design specified in SA-1).
2. **Sidebar / palette drag-drop preview** seeded by the open registry
   (SA-1) + the inlined diagram-type maps in `constants.ts` (SA-3).
3. **Settings service** (`settingsService` mirrored into a Zustand store):
   `usePropertiesPanel`, `classNotation` (ER ↔ UML), `showInstancedObjects`,
   `showIconView`, `showAssociationNames`. Per CLAUDE.md these must
   live-render via Zustand subscription, never via an `editorRevision++`
   bump that would clear undo history.
4. **Shared `NodeStyleEditor` color editor** (SA-2.1 mandate: surface fill
   / stroke / text color on every editable element kind).
5. **Style-pane extras** flagged in `parity-final/classDiagram.md:193-206`:
   `description`, `uri`, `icon`.

Sources of truth:

- v3 (old): `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/`
- v4 (new): `besser/utilities/web_modeling_editor/frontend/packages/library/lib/`

Status legend: **PASS** = behaviour ports cleanly. **GAP** = missing,
regressed, or contradicts the migration brief. **FIXED-WITH-CAVEAT** =
the gating exists but a sibling code path can still produce the bug
under specific user actions.

---

## 1. PropertiesPanel ↔ PopoverManager mutual exclusion

| Concern | Location | Status | Notes |
|---|---|---|---|
| `PropertiesPanel` mount gated on `usePropertiesPanel` | `lib/App.tsx:79, :157` (`useUsePropertiesPanel()` selector → conditional render) | **PASS** | Selector is `useSettingsStore(s => s.usePropertiesPanel)`; toggle re-renders `App` reactively (no editor remount). |
| `PopoverManager` body suppressed when panel mode is on | `lib/components/popovers/PopoverManager.tsx:658-662` (`if (!anchorEl \|\| !popupEnabled \|\| usePropertiesPanel) return null`) | **PASS** | Guard is the third disjunct of the early-return — short-circuits before the `<GenericPopover>` is mounted. |
| Panel `selectedId` source | `PropertiesPanel.tsx:40-52` (reads `useDiagramStore.selectedElementIds[0]`) | **PASS** | Driven by React-Flow selection — single-click is sufficient. |
| Popover open trigger | `lib/hooks/useElementInteractions.ts:34-46` (only `onNodeDoubleClick` / `onEdgeDoubleClick`) | **PASS** | Selection alone does NOT call `setPopOverElementId`. |
| **Edit-pencil button on `NodeToolbar`** | `lib/components/toolbars/NodeToolbar.tsx:38-43` (calls `setPopOverElementId(elementId)` unconditionally) | **GAP** | Toolbar is rendered for every selected node (`isVisible={isDiagramModifiable && !!selected}`). Clicking the pencil sets `popoverElementId`; the popover then renders **only because** of the suppression check above. If the suppression check were removed (or the user toggles `usePropertiesPanel` while a popover is already open), both surfaces appear together. |
| **Edit-pencil button on `GenericEdge`** | `lib/edges/GenericEdge.tsx:200` (`onEditClick={() => setPopOverElementId(id)}`) | **GAP** | Same issue as `NodeToolbar`. |

### Root cause of the user-reported "both panels render at once"

The `usePropertiesPanel` *suppression* in `PopoverManager.tsx:660` is in
place. **However**, every `<NodeToolbar>` and every `<GenericEdge>` still
unconditionally exposes an Edit pencil that calls
`setPopOverElementId(elementId)`. There are **77 callsites** of
`<PopoverManager>` (`grep -c` over `lib/nodes` + `lib/edges`), each of
which subscribes to the same `popoverElementId`. When the user is in
properties-panel mode and clicks the Edit pencil, the popover **does
not** open today (the `usePropertiesPanel` early-return wins). But:

1. The **pencil itself is still visible**, which the user reads as "the
   editor offered me a popover *and* a panel side-by-side" — SA-1's
   design wants only the panel surface in panel mode.
2. There is **no guard at the toolbar level**: `NodeToolbar.tsx` does not
   read `useUsePropertiesPanel()`. If a future refactor removes the
   suppression in `PopoverManager` (e.g. for the assessment-feedback
   flow, which has no panel equivalent in v4 yet), the bug becomes a
   real double-render.
3. In the **assessment** modes (`feedbackGive` / `feedbackSee`), the
   panel does not currently hide the popover — `inspectorKind` in
   `PropertiesPanel.tsx:65-70` honours those kinds, but the popover
   suppression is `usePropertiesPanel`-only and doesn't branch on mode.
   So in `BesserMode.Assessment`, **both** surfaces will mount today.

**Recommended fixes** (out-of-scope for this audit):

- Add `useUsePropertiesPanel()` to `NodeToolbar.tsx` and hide the pencil
  when the panel is the active surface (also in `GenericEdge.tsx:200`).
- Make the `PopoverManager` early-return mode-aware so the popover-only
  feedback flows in `Assessment` mode coexist with the panel cleanly,
  or extend the panel's `inspectorKind` to all three modes.

---

## 2. Palette / sidebar drag-drop preview

`Sidebar.tsx:125-159` reads from `dropElementConfigs[diagramType]` (the
mutable proxy registry exported from `constants.ts:1162-1169`). The
registry is seeded from `defaultDropElementConfigs` and extended via
`registerPaletteEntry()`.

### Per-diagram entry comparison vs v3 `class-preview.ts`

| Diagram | v3 entries | v4 entries | Δ |
|---|---|---|---|
| **ClassDiagram** | `class` (no body), `class` + attribute + method, `enumeration` + 3 cases, `ClassOCLConstraint` (lines 30-198 of `class-preview.ts`; **Package, AbstractClass, Interface are commented out** in v3 lines 21-27, 87-115, 117-148) | `package`, `class`, `class[stereotype=Abstract]`, `class[stereotype=Enumeration]`, `class[stereotype=Interface]` (`constants.ts:377-436`) | **GAP** — v4 added back the three entries v3 had explicitly deleted. The user wants Interface and Package hidden until functional. **Abstract** is a borderline case (the brief did not call it out). |
| `ObjectDiagram` | one `objectName` preview | identical (`:437-449`) | **PASS** |
| `ActivityDiagram` | initial / final / action / object / merge / fork / forkH + Activity container | identical (`:450-507`) | **PASS** |
| `UseCaseDiagram` | useCase, actor, system | identical (`:508-530`) | **PASS** |
| `CommunicationDiagram` | objectName | identical (`:531-543`) | **PASS** |
| `ComponentDiagram` | component, subsystem, interface | identical (`:544-567`) | **PASS** |
| `DeploymentDiagram` | node, component, artifact, interface | identical (`:568-602`) | **PASS** |
| `SyntaxTree` | nonterminal, terminal | identical (`:603-618`) | **PASS** |
| `PetriNet` | transition, place | identical (`:619-636`) | **PASS** |
| `ReachabilityGraph` | marking | identical (`:637-645`) | **PASS** |
| `Flowchart` | terminal, process, decision, I/O, function call | identical (`:646-682`) | **PASS** |
| `BPMN` | task, subprocess, transaction, callActivity, group, annotation, start/intermediate/end events, gateway, dataObject, dataStore, pool | identical (`:683-775`) | **PASS** |
| `StateMachineDiagram` (SA-3) | n/a in stock v4-upstream — added by BESSER | State, StateInitialNode, StateFinalNode, StateActionNode, StateObjectNode, StateMergeNode, StateForkNode, StateForkNodeHorizontal, StateCodeBlock (`:780-848`) | **PASS** — matches v3 BESSER fork. State body / fallback-body intentionally omitted (created inside parent State). |
| `AgentDiagram` (SA-4) | n/a in stock — BESSER addition | AgentState, AgentIntent, AgentIntentObjectComponent, AgentRagElement, StateInitialNode, StateFinalNode (`:855-907`) | **PASS** — body / fallback-body / intent body / description child nodes are non-droppable as in v3. |
| `UserDiagram` (SA-4) | n/a in stock — BESSER addition | UserModelName, UserModelIcon (`:909-927`) | **PASS** |
| `NNDiagram` (SA-5) | n/a in stock — BESSER addition | NNContainer, 13 layer kinds, TensorOp, Configuration, TrainingDataset, TestDataset, NNReference (`:934-1076`) | **PASS** |
| `Sfc` | start, step, jump, transitionBranch, actionTable | identical (`:1077-1124`) | **PASS** |
| `ColorDescription` (auxiliary, all diagrams) | always shown after a divider | mounted after the diagram entries (`Sidebar.tsx:161-187`) | **PASS** |

**Top palette gaps**:

- **HIGH** — `ClassDiagram` exposes `Interface` and `Package` palette
  items the user wants hidden. v3 has these commented out
  (`packages/editor/src/main/packages/uml-class-diagram/class-preview.ts:21-27, 117-148`).
  Remove `:377-385` (Package) and `:424-435` (Interface) from
  `lib/constants.ts`. `Abstract` (`:396-407`) is borderline — flagged
  for triage but not in the user's explicit hide list.
- **LOW** — Sidebar has no notion of "diagram-type-aware ordering" beyond
  source order in the constants object; v3 used the same approach so
  parity holds.

The palette registry itself (mutable proxy + `registerPaletteEntry`) is
correctly seeded from defaults and extended by SA-3 / SA-4 / SA-5
without TDZ cycles (`constants.ts:1135-1169`). The DropElementConfig
contract (`type`, `width`, `height`, `defaultData`, `svg`, `marginTop?`)
matches v3's preview shape closely enough that `Sidebar.tsx` and
`DraggableGhost.tsx:39-50` consume it directly.

---

## 3. Settings service: live-render audit

`packages/library/lib/services/settingsService.ts:30-36` defines
`DEFAULT_SETTINGS` with five keys. `lib/store/settingsStore.ts:32-58`
mirrors them into a Zustand store seeded by `settingsService.getSettings()`
and kept in sync via `settingsService.onSettingsChange`. Per CLAUDE.md
("**Don't bump `editorRevision` for view-only toggles — that clears undo
history**"), each key must have at least one component subscribing to
the store via a fine-grained selector.

Live consumers (verified by `grep` across `lib/`):

| Setting | Selector | Consumer | Status |
|---|---|---|---|
| `usePropertiesPanel` | `useUsePropertiesPanel()` (`store/settingsStore.ts:79-80`); also `usePanelMode()` (`store/propertiesPanelStore.ts:53-54`) | `App.tsx:79, :157` (panel mount) + `PopoverManager.tsx:658-662` (popover suppression) | **PASS** |
| `classNotation` (ER ↔ UML) | `useClassNotation()` (`store/settingsStore.ts:75-76`) | `nodes/classDiagram/Class.tsx:67` (re-formats every row through `formatRow`); `inspectors/classDiagram/ClassEdgeEditPanel.tsx:76` (multiplicity widget) | **PASS** — `Class.tsx:69-79` uses `useMemo([attributes, classNotation])` so the toggle re-fits the node width without an editor remount, exactly as the brief asks. |
| `showIconView` | `useSettingsStore(s => s.showIconView)` | `components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx:40` (single subscription on the object-name SVG) | **PASS** for `ObjectName`. **GAP** — no other element kind reads it. v3 surfaced an icon-mode for class nodes too (`packages/editor/.../uml-classifier-update.tsx:217-225` `showIcon`); v4 does not propagate it to `Class.tsx` or to `User`/`Agent` containers that v3 supported. |
| `showAssociationNames` | none | none | **GAP** — the toggle exists in `settingsService` and the Zustand store exposes it via the `useSettings()` selector (`store/settingsStore.ts:69`), but **no component subscribes**. The relationship-edge components (`edges/edgeTypes/ClassDiagramEdge.tsx`, `…/ObjectDiagramEdge.tsx`, `…/AgentDiagramEdge.tsx`, `…/CommunicationDiagramEdge.tsx`) render the association-name label unconditionally. v3 honoured this flag in the renderer; the v4 port lost it. |
| `showInstancedObjects` | none | none | **GAP** — the toggle exists, the store exposes it via `useSettings()`, but **no component subscribes**. The flag is meant for the object-diagram instance-preview overlay (v3: `packages/editor/.../uml-classifier-update.tsx`); v4 has no consumer at all. |

`SettingsService.notifyListeners()` correctly calls every listener on
mutation (`settingsService.ts:114-122`); `useSettingsStore` registers
**one** listener at module init (`store/settingsStore.ts:42-44`), so
mutations propagate via a single fan-out. No `editorRevision++` is ever
called for these settings — confirmed clean.

---

## 4. Color editor coverage (shared `NodeStyleEditor` / `EdgeStyleEditor`)

SA-2.1 requires every editable element kind to expose fill / stroke /
text color via the shared `NodeStyleEditor` (or `EdgeStyleEditor` for
edges). Audit by inspecting every file under
`lib/components/inspectors/**/*EditPanel.tsx`:

| Inspector | Style editor present | Status |
|---|---|---|
| `classDiagram/ClassEditPanel.tsx:966` | `NodeStyleEditor` | **PASS** |
| `classDiagram/ClassEdgeEditPanel.tsx:137` | `EdgeStyleEditor` | **PASS** |
| `classDiagram/ClassOCLConstraintEditPanel.tsx:58` | `NodeStyleEditor` | **PASS** |
| `objectDiagram/ObjectEditPanel.tsx` | `NodeStyleEditor` (verified by `grep -L`) | **PASS** |
| `objectDiagram/ObjectLinkEditPanel.tsx` | `EdgeStyleEditor` | **PASS** |
| `stateMachineDiagram/State*EditPanel.tsx` | `NodeStyleEditor` | **PASS** |
| `stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx` | **MISSING** (`grep -L` reports it as one of two files without any `*StyleEditor` import) | **GAP** |
| `agentDiagram/AgentState*EditPanel.tsx`, `AgentIntent*EditPanel.tsx`, `AgentIntentObjectComponentEditPanel.tsx`, `AgentIntentDescriptionEditPanel.tsx`, `AgentRagElementEditPanel.tsx` | `NodeStyleEditor` | **PASS** |
| `agentDiagram/AgentDiagramEdgeEditPanel.tsx:267` | `EdgeStyleEditor` | **PASS** |
| `agentDiagram/AgentDiagramInitEdgeEditPanel.tsx` | **MISSING** (the second file without any style editor) | **GAP** |
| `nnDiagram/NNComponentEditPanel.tsx`, `NNContainerEditPanel.tsx`, `NNReferenceEditPanel.tsx` | `NodeStyleEditor` | **PASS** |
| `userDiagram/UserModelNameEditPanel.tsx`, `UserModelAttributeEditPanel.tsx` | `NodeStyleEditor` | **PASS** |

**Per-attribute / per-method colour panes** (v3 supported via the
`StylePane` collapsing under each row): not implemented in any v4 panel.
Tracked in `PC-2-class-edit-panel.md:51` ("Per-attribute colour pane —
`StylePane fillColor textColor` per row — Not present per-row — **GAP**
(minor)"). The audit accepts this as a deliberate v4 simplification (the
classifier-level `NodeStyleEditor` covers the common case).

---

## 5. Style-pane extras: `description`, `uri`, `icon`

`parity-final/classDiagram.md:193-206` flagged this in the final
ClassDiagram audit. Re-confirmed here for the cross-cutting brief.

| Field | v3 source | v4 location | Status |
|---|---|---|---|
| `description` (free-text) | `StylePane showDescription` (`uml-classifier-update.tsx:217-225, 350`) | `lib/types/nodes/NodeProps.ts:108-109` declares the type; **no inspector input** in `ClassEditPanel.tsx`. `Class` node renderer also doesn't surface it. | **GAP** — declared in types but unwired in the inspector and the renderer. v3→v4 lift drops the field; v4→v3 emit drops it. |
| `uri` | `StylePane showUri` (`uml-classifier-update.tsx:217-225, 364`) | Not declared on `ClassNodeProps` (`NodeProps.ts:85-130`); not in any inspector | **GAP** |
| `icon` (data URL or canonical id) | `StylePane showIcon` (`:217-225, 378`) | Same — type declared on `ObjectNodeProps` (`NodeProps.ts:124-125`) for the icon-view mode (consumed by `ObjectNameSVG.tsx:40`), but no inspector lets the user **set** it. | **GAP** |

**Round-trip impact** (per `parity-final/classDiagram.md:201-206`): any
v3 fixture carrying these fields loses them on conversion;
`utils/versionConverter.ts:660-666, 698-704, 2103-2119` already drop
them in both directions, so the regression is silent.

---

## Top 3 cross-cutting gaps (priority-ordered)

1. **Palette regression — `Interface` and `Package` re-introduced for
   `ClassDiagram`** (`lib/constants.ts:377-385, :424-435`). v3
   explicitly commented these out
   (`uml-class-diagram/class-preview.ts:21-27, 117-148`). User wants
   them hidden until functional. Quick fix: delete those two array
   entries; consider gating `Abstract` (`:396-407`) too pending product
   review.
2. **`showAssociationNames` and `showInstancedObjects` have no live
   consumers** (`store/settingsStore.ts:67, :69`). The settings
   round-trip through localStorage and even appear in the Zustand store,
   but no element renderer subscribes — the toggles are dead controls.
   `showIconView` is wired only on `ObjectNameSVG.tsx:40`; class /
   user / agent containers ignore it. Fix scope: subscribe the
   relationship-edge components (`ClassDiagramEdge.tsx` and the four
   sister edge files) to `useSettings(s => s.showAssociationNames)` and
   gate the label render; subscribe the object-instance preview overlay
   to `showInstancedObjects`.
3. **Style-pane extras (`description`, `uri`, `icon`) are missing from
   `ClassEditPanel`** (gap inherited from
   `parity-final/classDiagram.md:193-206`). v3's `StylePane` exposed all
   three; v4 declares `description` on the type but the inspector body
   never renders it; `uri` is undeclared; `icon` is consumed only by the
   object renderer and never settable. Round-trip drop is silent.

Honourable mentions (one rung below the top three):

- The Edit-pencil button on `NodeToolbar` and `GenericEdge` does not
  honour `usePropertiesPanel`. The popover suppression in
  `PopoverManager.tsx:660` masks the symptom today, but the toolbar
  affordance is still rendered, which the user reads as duplication.
- `StateMachineDiagramEdgeEditPanel.tsx` and
  `AgentDiagramInitEdgeEditPanel.tsx` lack the shared `EdgeStyleEditor`
  surface (the only two `*EditPanel.tsx` files without any style editor
  import).
- In `BesserMode.Assessment`, the popover and panel **can** coexist —
  the suppression is `usePropertiesPanel`-only and doesn't branch on
  mode, while `PropertiesPanel` itself does honour `feedbackGive`/
  `feedbackSee` (`PropertiesPanel.tsx:65-70`).

---

## Verdict

**Conditional PASS with three blocking gaps.** The shared inspector
registry (`registry.ts`) and the Zustand-mirrored settings service are
architecturally sound: `usePropertiesPanel` and `classNotation` both
live-render via fine-grained selectors with no `editorRevision++`
bumps, exactly as SA-1 / CLAUDE.md mandate. The user-reported "both
surfaces render together" symptom is caused by the **Edit-pencil
toolbar affordance**, not by missing gating in `PopoverManager` itself.
The top three blocking gaps before declaring full parity are: (1)
remove the regressed `Interface` / `Package` palette entries; (2) wire
`showAssociationNames` and `showInstancedObjects` to live consumers; (3)
add `description` / `uri` / `icon` to `ClassEditPanel` so the v3 →
v4 → v3 round-trip stops dropping them silently.
