# Deep Review — Webapp Store ↔ Lib Editor Sync

Read-only audit of how the webapp's Redux store stays in sync with the
`@besser/wme` editor library. Covers initial mount, edit propagation,
`editorRevision` bumps, tab switching, undo/redo, schema migration,
the cross-diagram bridge, and storage-sync loops.

Submodule HEAD audited: same as branch `claude/refine-local-plan-sS9Zv`.
All findings are derived from the committed snapshot — no runtime tracing.

## Summary

| Dimension                            | Verdict     |
|--------------------------------------|-------------|
| 1. Initial mount                     | PASS w/ caveat |
| 2. Edit propagation                  | PASS        |
| 3. `editorRevision` bumps            | PASS        |
| 4. Switching diagrams (tabs)         | PARTIAL     |
| 5. Undo / redo                       | PASS        |
| 6. First-load migration `migrateProjectToV5` | PASS |
| 7. `diagramBridge` `setClassDiagramData` | **FAIL** (PC-A5 confirmed) |
| 8. Storage-sync loops / `withoutNotify` | PASS    |

The single largest regression is dimension 7 — the
`diagramBridge.setClassDiagramData` call in `DiagramTabs.tsx` reads the
*Redux* class-diagram model, which lags the *editor's* live model by the
300 ms debounce window. ObjectDiagram users editing a referenced
ClassDiagram and immediately switching to its child ObjectDiagram see
the previous-save's class palette, not the current one.

---

## 1. Initial mount

`BesserEditorComponent.tsx` mounts inside `EditorView` after the
workspace shell has dispatched `loadProjectThunk`. The setup effect
(lines 99-140) only fires when `editorRevision !== 0`, and
`loadProjectThunk.fulfilled` increments it from 0 → 1.

**Sequence (ASCII):**

```
WorkspaceShell mount
        │
        ├── dispatch(loadProjectThunk(id))
        │           │
        │           ├── ProjectStorageRepository.loadProject  ──► returns BesserProject
        │           │   (ensureProjectMigrated runs idempotently)
        │           │
        │           ├── setupBridgeForActiveDiagram(project, activeDiagram, type)
        │           │     └── if Object/User → diagramBridge.setClassDiagramData(...)
        │           │
        │           └── fulfilled reducer:
        │                   activeDiagram, activeDiagramType, editorOptions = …
        │                   editorRevision += 1   (0 → 1)
        │
        ├── React re-renders BesserEditorComponent (selectors fire)
        │
        └── setup effect (deps [editorRevision]):
                ├── lastHandledRevisionRef.current = 1
                ├── await cleanupEditor()                  (no prior editor → noop)
                ├── new BesserEditor(container, options)
                ├── await nextEditor.nextRender
                ├── nextEditor.model = activeDiagram.model (via reduxDiagramRef.current)
                ├── modelSubscriptionRef = subscribeToModelChange(callback)
                └── setEditor(nextEditor)
```

**Race notes:**

- `reduxDiagramRef.current = reduxDiagram` (line 35) is updated on every
  render *during* the render phase — by the time the setup effect runs
  (post-commit), the ref already points at the resolved diagram. ✓
- `setupRunRef` (line 22) and `runId` (line 106) guard against double
  setups when `editorRevision` bumps twice in rapid succession (e.g.
  `loadProjectThunk` → `switchDiagramTypeThunk`).
- Setting `nextEditor.model = …` calls the lib's
  `recreateEditor(state)` (apollon-editor.ts:50-57), which is
  synchronous after `await nextRender`. Subsequent `subscribeToModelChange`
  fires only on user input, so the initial assignment does NOT
  re-emit through `updateDiagramModelThunk`. ✓

**Caveat (caught but benign):** if the project has *no* diagrams of any
visible type, `activeDiagram` is `null` and the setter is skipped — the
editor mounts with the default empty model from `defaultEditorOptions`.
This is the intended path; no UI fallback is missing.

---

## 2. Edit propagation

```
User keystroke / drag in canvas
        │
        ├── lib internal Redux dispatches (via undoable reducer)
        │
        ├── ModelState changes
        │
        └── modelSubscribers[id] fired with new UMLModel
                │
                └── BesserEditorComponent callback (line 129):
                        ├── clearTimeout(debouncedSaveRef.current)
                        ├── setTimeout(300 ms):
                        │     dispatch(updateDiagramModelThunk({ model }))
                        │             │
                        │             ├── reads project, activeDiagramType, activeDiagramIndex
                        │             ├── updated = { ...current, ...updates, lastUpdate: now }
                        │             ├── ProjectStorageRepository.withoutNotify(() =>
                        │             │     ProjectStorageRepository.updateDiagram(...))
                        │             │       → localStorage write (notifyChange suppressed)
                        │             │
                        │             └── fulfilled reducer:
                        │                   state.activeDiagram = updated
                        │                   project.diagrams[type][idx] = updated
                        │                   (no editorRevision bump — ✓)
                        │
                        └── debouncedSaveRef = null (after fire)
```

**Pass criteria:**
- 300 ms coalescing prevents per-keystroke localStorage thrash. ✓
- `withoutNotify` prevents `useStorageSync` from re-dispatching
  `syncProjectFromStorage`, which would otherwise overwrite Redux with
  a value identical to what the thunk just wrote (no real loop, but
  wasteful). ✓
- Reducer mutates `state.activeDiagram` in place (Immer); selector
  identity changes so DiagramTabs etc. re-derive. ✓

**Edge case:** on unmount with a pending debounce, `cleanupEditor`
(line 55) clears the timeout. The pending edit is **lost** — see Issue
2 in the regressions list.

---

## 3. `editorRevision` bumps — call-site audit

Every `state.editorRevision += 1` is in `workspaceSlice.ts`:

| Line | Action | Justified? |
|------|--------|------------|
| 568  | `bumpEditorRevision` reducer (manual)        | Caller-dependent (see below) |
| 633  | `loadProjectThunk.fulfilled`                 | Yes — full project swap |
| 648  | `createProjectThunk.fulfilled`               | Yes — fresh project |
| 667  | `switchDiagramTypeThunk.fulfilled`           | Yes — diagram type changed |
| 680  | `switchDiagramIndexThunk.fulfilled`          | Yes — different diagram |
| 735  | `addDiagramThunk.fulfilled`                  | Yes — new diagram becomes active |
| 748  | `removeDiagramThunk.fulfilled`               | Yes — active may shift |

**Manual `bumpEditorRevision` callers (3 sites):**

| File | Line | Purpose | Verdict |
|------|------|---------|---------|
| `WorkspaceShell.tsx` | 567 | Switch agent base/personalised model variant (model body changes) | **OK** — model content swap |
| `WorkspaceShell.tsx` | 594 | Same, variant select path | **OK** |
| `DiagramTabs.tsx`    | 156 | ObjectDiagram class-reference dropdown changed | **OK** — palette must rebind to a different ClassDiagram (structural) |
| `useModelInjection.ts` | 231 | Assistant created a new tab + injected systemSpec | **OK** — addDiagram already bumped, this is redundant but harmless |
| `useModelInjection.ts` | 398 | Assistant fallback when `modelingService` not ready | **OK** — model body replaced |

**No view-only bumps found.** The frontend's CLAUDE.md warning ("Don't
bump editorRevision for view-only toggles — that clears undo history")
is honored. The class-notation toggle (UML/ER) and the
`showAssociationNames` / `showInstancedObjects` toggles route through
`settingsService.onSettingsChange` (lib internal), not Redux — see
`packages/editor/src/main/scenes/application.tsx`.

**Minor finding:** `useModelInjection.ts:231` bumps *after*
`addDiagramThunk` + `switchDiagramIndexThunk` have *already* bumped
twice. Net: the editor reinitialises three times for one assistant
"create new tab" command. Not user-visible (the loading spinner masks
it) but wasteful. See Issue 4.

---

## 4. Switching diagrams (tabs) — destroy + recreate

```
User clicks tab N
        │
        └── DiagramTabs.handleSwitchTab(index)
                │
                ├── if onRequestTabSwitch (validation gate) returns false → abort
                │
                └── dispatch(switchDiagramIndexThunk({ diagramType, index }))
                        │
                        ├── ProjectStorageRepository.withoutNotify(() =>
                        │     ProjectStorageRepository.switchDiagramIndex(...))
                        │       → updates currentDiagramIndices[type] = N in localStorage
                        │
                        └── fulfilled reducer:
                                activeDiagram = diagram (the new one)
                                activeDiagramIndex = N
                                editorRevision += 1
                                project.currentDiagramIndices[type] = N

React re-renders BesserEditorComponent
        │
        └── setup effect (editorRevision changed):
                ├── runId = ++setupRunRef.current
                ├── await cleanupEditor()
                │     ├── clearTimeout(debouncedSaveRef.current)   ← pending 300 ms save flushed?
                │     ├── editor.unsubscribeFromModelChange(...)
                │     └── editor.destroy()    (deferred via setTimeout(0))
                │
                ├── new BesserEditor(...)
                ├── await nextRender
                ├── nextEditor.model = currentDiagram.model    (the NEW diagram)
                └── re-subscribe + setEditor
```

**Bodies preserved? Conditional:** if the user typed within 300 ms of
clicking a tab, `cleanupEditor` *clears* the debounced save without
flushing. The `updateDiagramModelThunk` never runs → the keystroke is
**lost**. The GUI editor at `GraphicalUIEditor.tsx:165-204` has the
correct pattern (synchronous flush in cleanup); the UML
`BesserEditorComponent` does not.

**Verdict: PARTIAL** — destroy+recreate is correct, but the
debounced-save lifecycle is not. See Issue 2 in the regressions list.

---

## 5. Undo / redo

The lib's `keyboard-eventlistener.tsx` (lines 134-160) intercepts
`Ctrl/Cmd+Z` and `Ctrl/Cmd+Shift+Z` directly when the editor canvas
has focus, dispatching `UndoRepository.undo()` /
`UndoRepository.redo()` against the lib's *internal* Redux store
(separate from the webapp's). The undoable reducer at
`undo/undo-reducer.ts:10` snapshots actions where `action.undoable === true`.

The webapp does NOT install a Cmd+Z handler for UML diagrams (only
quantum: `useCircuitKeyboard.ts:34`). There is no fight: the lib owns
UML undo, the webapp owns quantum undo, and they target different
canvases.

```
Cmd+Z in UML canvas
        │
        └── lib KeyboardEventListener.keyDown
                ├── event.preventDefault()
                ├── UndoRepository.undo() → lib store dispatches @@undo/UNDO
                │
                ├── lib reducer rolls back (does NOT bump webapp's editorRevision)
                │
                └── modelSubscribers[id] fires with reverted UMLModel
                        │
                        └── BesserEditorComponent debounced callback (300 ms)
                                └── updateDiagramModelThunk → Redux + localStorage
```

The callback path means undo is also persisted — Cmd+Z in the editor
ends up in `besser_projects` 300 ms later. ✓

**Caveat:** the lib mentioned in the spec — "via Yjs UndoManager" — is
**not** what's wired. The `@besser/wme` package in this submodule uses
a hand-rolled `undoable` higher-order reducer (`undo-reducer.ts`),
not Y.js. Y.UndoManager is referenced in the `packages/library` (v4)
work elsewhere (`docs/.../02-behavioral.md:46`), but that lib is not
wired into the webapp at this branch. The audited path is the v3-style
Redux undoable reducer — which is fine and tested.

---

## 6. First-load migration

`migrateProjectToV5` runs inside `ensureProjectMigrated` (project.ts:578-581),
which is called by `ProjectStorageRepository.loadProject(...)` for every
project read. `getCurrentProject` and `loadProject` are idempotent — the
gate is `project.schemaVersion >= 5` (line 636), and `migrateProjectToV5`
sets `project.schemaVersion = 5` (line 653) on success.

**Idempotence guarantees:**
- Already-v5 projects short-circuit at line 636. ✓
- Per-diagram migration loops only mutate when
  `isV3UMLModel(d.model)` returns true (line 642) — v4 models are
  no-ops. ✓
- On `migrateUMLModelV3ToV4` throw, `d.model` is left untouched
  (line 645-649) and `schemaVersion` is *still* set to 5 — meaning a
  partially-failed migration is **not retried** on next load (despite
  the comment promising "next launch will retry"). See Issue 5.
- The retrofit at `retrofitEmptyUserDiagrams` (line 594) runs on *every*
  load regardless of schemaVersion — it's idempotent because it only
  touches diagrams whose `nodes.length === 0`.

The `loadProject` thunk explicitly does not write the migrated project
back to storage, so a successful v4→v5 migration that's never followed
by a save (read-only session) leaves the on-disk schema at v4. The next
load re-migrates from v4 — cheap, but redundant.

---

## 7. `diagramBridge` re-call cadence (PC-A5)

Three call sites push into `diagramBridge.setClassDiagramData`:

| Site | Trigger | Source of model |
|------|---------|-----------------|
| `workspaceSlice.ts:135` (`setupBridgeForActiveDiagram`) | `loadProjectThunk` / `switchDiagramTypeThunk` | Redux project snapshot |
| `DiagramTabs.tsx:154` (`useEffect`)                    | Every change to `classDiagrams` array reference (i.e. every save of any ClassDiagram) | Redux |
| `useImportDiagram.ts:149`                              | After import finalisation | Redux |
| `BesserEditorComponent.tsx:84-85`                     | `setStateMachineDiagrams` / `setQuantumCircuitDiagrams` (NOT the class data) | Redux |

**The PC-A5 lag is real:**

```
t=0 ms       User types 'X' on a Class node in ClassDiagram
t=0 ms       lib modelSubscribers fire → debounce starts
t=200 ms     User clicks ObjectDiagram tab
t=200 ms     dispatch(switchDiagramIndexThunk(...))
             ├── localStorage update (currentDiagramIndices)
             └── editorRevision += 1
t=200 ms     React re-render
             └── DiagramTabs useEffect:
                   classDiagrams array did NOT change yet (the keystroke is
                   still in the 300 ms debounce in BesserEditorComponent)
                 → diagramBridge.setClassDiagramData(STALE classDiagrams[i].model)
t=200 ms     BesserEditorComponent destroys+recreates with ObjectDiagram
             but the Object palette is built from the stale ClassDiagram
t=200 ms     Stale debounced save was attached to the old (now-destroyed)
             editor — BUT cleanupEditor cleared the timeout (Issue 2).
             The X keystroke is lost AND the bridge never gets the new value.
```

In a milder scenario (typing → wait > 300 ms → switch tab), the save
fires, `classDiagrams` updates, but the bridge sync is **one save
behind** until the next render that observes a `classDiagrams`
identity change. The `useEffect` at line 147 *does* eventually re-call
`setClassDiagramData(refModel)` — but only after the next render
following the save. PC-A5 documents this as "lags by one save"; the
audit confirms.

**Verdict: FAIL** for the strict "live mirror" interpretation,
**PASS** for "eventually consistent". The fix is to read the current
class diagram model directly from the lib editor (via
`besserEditor.model` when ObjectDiagram references the class diagram
held by another editor instance — which doesn't currently happen,
since only one editor lives at a time). A simpler fix: re-call
`setClassDiagramData` from within `subscribeToModelChange`'s callback
in `BesserEditorComponent` whenever the active diagram is a
ClassDiagram.

---

## 8. Storage-sync loops & `withoutNotify`

All thunks in `workspaceSlice.ts` that call `ProjectStorageRepository`
write methods do so inside `withoutNotify`:

| Thunk | Write call | `withoutNotify` |
|-------|------------|-----------------|
| `createProjectThunk` (190)         | createNewProject     | ✓ |
| `switchDiagramTypeThunk` (213)     | switchDiagramType    | ✓ |
| `switchDiagramIndexThunk` (235)    | switchDiagramIndex   | ✓ |
| `updateDiagramModelThunk` (263)    | updateDiagram        | ✓ |
| `updateQuantumDiagramThunk` (295)  | updateDiagram        | ✓ |
| `setPerspectiveEnabledThunk` (370) | saveProject          | ✓ |
| `applyPerspectivePresetThunk` (420)| saveProject          | ✓ |
| `updateDiagramReferencesThunk` (455)| updateDiagramReferences | ✓ |
| `addDiagramThunk` (488)            | addDiagram           | ✓ |
| `removeDiagramThunk` (510)         | removeDiagram        | ✓ |
| `renameDiagramThunk` (554)         | updateDiagram        | ✓ |
| `updateProjectInfo` reducer (580)  | saveProject          | ✓ |

**100 % coverage.** No thunk-driven write would cause a
`useStorageSync` re-dispatch.

**Editors that bypass thunks (intentional):**
- `GraphicalUIEditor.tsx:116, 192, 487, 541` — direct `updateDiagram` calls.
  Relies on `notifyChange` to push back into Redux via `useStorageSync`. ✓
- `useCircuitPersistence.ts:58` — same pattern for QuantumCircuit. ✓
- `AgentConfigurationPanel.tsx:421` — direct `updateDiagram`,
  notifyChange → Redux. ✓

**Direct `saveProject` outside `withoutNotify` (legitimate):**
- `useImportDiagram.ts:142`, `useImportDiagramKG.ts:73`,
  `useImportDiagramPicture.ts:97`, `GitHubSidebar.tsx:166,199`,
  `useAutoCommit.ts` — all of these *want* the listener to fire so
  Redux picks up the new project. ✓

---

## Top 5 regressions

1. **Bridge lag for ObjectDiagram (PC-A5).** `diagramBridge.setClassDiagramData`
   is fed from the *Redux* `classDiagrams` snapshot, which trails the live
   ClassDiagram editor by one debounce window (300 ms). Edits to a
   ClassDiagram followed by an immediate ObjectDiagram tab switch use
   stale palette data. Fix: invoke `setClassDiagramData` from inside
   the model-change subscription in `BesserEditorComponent` whenever
   the active diagram type is `ClassDiagram`, so the bridge mirrors the
   live editor, not Redux.

2. **Pending UML save dropped on tab switch.** `BesserEditorComponent.cleanupEditor`
   (line 56-60) calls `clearTimeout(debouncedSaveRef.current)` without
   firing the pending save. The GUI editor flushes synchronously
   (`GraphicalUIEditor.tsx:173-203`) — the UML editor must do the
   same. Reproducer: type a character in a class name and click another
   tab within 300 ms — the keystroke is silently lost.

3. **`migrateProjectToV5` writes `schemaVersion = 5` even on per-diagram
   failure.** Line 653 unconditionally sets the version to 5 after
   the loop, but the per-diagram catch (line 645-649) leaves v3 models
   in place and only logs to console. The promise that "next launch
   will retry" is wrong — the `if (!schemaVersion || < 5)` gate
   short-circuits on the next load. Fix: track per-diagram success and
   only bump `schemaVersion` when *all* diagrams migrated cleanly,
   or move the bump inside the success path.

4. **Triple editor reinit on assistant "new tab".** `useModelInjection.ts:208-232`
   chains `addDiagramThunk` (bump) → `switchDiagramIndexThunk` (bump) →
   `updateDiagramModelThunk` (no bump) → manual `bumpEditorRevision`
   (third bump). Net: the lib editor is destroyed and re-created three
   times in close succession; each `recreateEditor` rebuilds the SVG
   tree. Fix: drop the manual bump (the prior thunks already bumped),
   or replace the chain with a single project-level write + one bump.

5. **`refreshProjectStateThunk` doesn't bump revision but
   `loadProjectThunk` does — inconsistent contract.** The variant
   switch in `WorkspaceShell.tsx:566-567` uses `refreshProjectStateThunk`
   *and then* a manual `bumpEditorRevision`. If a future caller
   forgets the second dispatch, the editor will silently keep the old
   model in its lib store while Redux holds the new one. Document the
   pairing requirement (or fold the bump into a dedicated thunk).
