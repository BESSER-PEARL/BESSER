# PC-12 — `BesserEditor` Public API Audit

Verdict: **PASS (with documented MINOR gaps)**

Submodule scope: `besser/utilities/web_modeling_editor/frontend`. All
sources read from the library at
`packages/library/lib/besser-editor.tsx` and the webapp consumer at
`packages/webapp/src/main/features/editors/uml/BesserEditorComponent.tsx`.

The webapp works end-to-end through the surface today. The only methods
listed in the SA scope that are **missing** are the ones that were already
flagged in `docs/source/migrations/api-surface-diff.md` as "MISSING — no
webapp use today, plan-only" (`subscribeToSelectionChange`,
`subscribeToApollonErrors` / its rename, `select(ids)`, `undo`, `redo`).
None of them block the webapp; they remain on the SA-7 follow-up list.

---

## 1. Method-by-method inventory

Source of truth: `packages/library/lib/besser-editor.tsx` (the `.d.ts`
under `dist/` matches and was regenerated from this file).

### 1.1 Present, signature OK

| Method / accessor | Line | Behaviour |
|---|---|---|
| `constructor(element, options?)` | 59-169 | Validates `element instanceof HTMLElement`, builds Zustand stores around a fresh `Y.Doc`, mounts `<AppWithProvider>` via `ReactDOM.createRoot`. Initialises the `ready` promise **before** render so callers can `await editor.ready` immediately after `new BesserEditor(...)`. |
| `get model(): UMLModel` | 456-470 | Pulls `nodes`, `edges`, `diagramId` from `diagramStore` and `diagramTitle`/`diagramType` from `metadataStore`, runs each node/edge through `mapFromReactFlowNodeTo…`, hard-codes `version: "4.0.0"`, attaches `assessments` and (optionally) `interactive`. |
| `set model(model)` | 472-481 | Replaces nodes/edges/assessments/interactive in place via `diagramStore` setters and updates metadata via `metadataStore.updateMetaData`. **Does NOT recreate the editor** — the React tree, the `Y.Doc`, the React Flow instance and the subscriber list all survive. (See §3 for the lifecycle implication.) |
| `subscribeToModelChange(cb)` | 378-387 | Wraps `diagramStore.subscribe(() => cb(this.model))` and registers the unsubscriber under the next free numeric id. Returns the id. Fires synchronously after every Zustand state change (see §4). |
| `subscribeToDiagramNameChange(cb)` | 389-398 | Mirrors the same pattern over `metadataStore`'s `diagramTitle` field. |
| `subscribeToAssessmentSelection(cb)` | 400-409 | Subscribes to `assessmentSelectionStore.selectedElementIds`. |
| `unsubscribe(id)` | 411-417 | Looks up `subscribers[id]`, calls it, deletes the slot. No-op if id is unknown. |
| `unsubscribeFromModelChange(id)` | 198-200 | `@deprecated` thin alias for `unsubscribe(id)`. **Used today** by `BesserEditorComponent.tsx:66`. |
| `exportAsSVG(options?)` | 367-369 | Delegates to the static `exportModelAsSvg(this.model, options)`. |
| `static exportModelAsSvg(model, options?, theme?)` | 244-361 | Mounts the editor into a hidden 4000×4000 div, awaits `document.fonts.ready`, double-rAF + 150 ms `setTimeout` to let layout settle, then computes a 60 px-margin clip via `getRenderedDiagramBounds` and serialises with `getSVG`. Cleans up `svgRoot.unmount()`, removes the container, destroys its `Y.Doc`. The `theme` parameter is currently a `void theme` no-op (parity gap with v3 — flagged in §6). |
| `destroy()` | 219-236 | Iterates `subscribers`, calls each unsubscriber, clears the map, calls `syncManager.stopSync()`, `root.unmount()`, `ydoc.destroy()`, nulls the React Flow instance. The whole body is wrapped in `try { … } catch { /* ignore */ }`, so a partial failure won't throw out of cleanup. |
| `get view()` / `set view(v)` | 491-497 | Round-trip on `metadataStore.view`. |
| `set diagramType(type)` | 213-217 | Calls `metadataStore.updateDiagramType` and **clears nodes/edges/assessments**. Note: there is no matching `get diagramType` — read it via `getDiagramMetadata().diagramType`. |
| `addOrUpdateAssessment(a)` | 499-501 | Forwards to `diagramStore.addOrUpdateAssessment`. |
| `getSelectedElements()` | 483-489 | Returns `assessmentSelectionStore.selectedElementIds` only when `mode === Assessment && readonly`, otherwise the modelling-mode `diagramStore.selectedElementIds`. |
| `toggleInteractiveElementsMode(forceEnabled?)` | 431-443 | Sets `view` to `Highlight` or `Modelling`. `forceEnabled` overrides the toggle (true → Highlight, false → Modelling). |
| `getInteractiveForSerialization()` | 445-449 | Returns `diagramStore.getInteractiveForSerialization()` (or `undefined` when no interactive map exists). |
| `getDiagramMetadata()` | 451-454 | `{ diagramTitle, diagramType }`. |
| `updateDiagramTitle(name)` | 427-429 | Forwards to `metadataStore.updateDiagramTitle`. |
| `get ready: Promise<void>` | 184-186 | Resolves once `setReactFlowInstance` is called from `<AppWithProvider onReactFlowInit={…}>`. The promise is constructed **inside the constructor before render**, so `await editor.ready` is safe immediately after `new BesserEditor(…)`. |
| `get nextRender: Promise<void>` | 188-193 | `@deprecated` getter that returns `this.ready`. **Used today** by `BesserEditorComponent.tsx:117`. |

### 1.2 Missing from the new lib

These are still on the SA-7 follow-up list per
`docs/source/migrations/api-surface-diff.md` §B:

| Method | Webapp call sites today | Notes |
|---|---|---|
| `select(ids: string[])` | none | The only webapp `editor.select(...)` call is in `GraphicalUIEditor.tsx`, which is the GrapesJS editor — a different `editor` object. Adding the method is a parity item, not a webapp blocker. |
| `subscribeToSelectionChange(cb)` | none | Selection state lives in `diagramStore.selectedElementIds`, identical infrastructure to the model/name subscriptions; trivial to add — five lines mirroring `subscribeToDiagramNameChange`. |
| `undo()` / `redo()` | none | The Yjs `UndoManager` is fully wired in `diagramStore` (`undoManager.undo()` / `.redo()` exist on the store), but no editor-class method delegates to them. The in-canvas keyboard shortcut (`Ctrl+Z`) works because the editor's own listeners drive the store directly. External callers cannot trigger undo/redo today. |
| `subscribeToApollonErrors(cb)` / its rename `subscribeToBesserErrors` | none | No error-bus exists in the new lib at all — there is no `errorStore`, no error listener registry. The string `BesserError` / `ApollonError` does not appear anywhere under `packages/library/lib/`. Adding it is greenfield work, not a rename. |
| `unsubscribeFromApollonErrors(id)` / rename | none | Pair of the above. |

---

## 2. Yjs UndoManager — confirm no network sync

**Verdict: confirmed local-only.**

- `packages/library/package.json` declares **only** `yjs: 13.6.20` as a
  Yjs-related dependency. There is **no** `y-websocket`, **no**
  `y-webrtc`, **no** `y-protocols` (transport-layer protocols), **no**
  `@hocuspocus/provider`. (`grep -rE "y-websocket|y-webrtc|WebsocketProvider|WebrtcProvider"`
  on `packages/library/` returns zero hits.)
- `lib/sync/yjsSyncClass.ts` only observes the local `Y.Doc` and routes
  serialised updates through `sendBroadcastMessage`/`handleReceivedData`
  callbacks **that the host application must wire up**. The library
  itself never opens a socket. The webapp does **not** call
  `editor.sendBroadcastMessage(...)` (verified in
  `BesserEditorComponent.tsx`), so the broadcast path is dormant.
- The `UndoManager` is constructed in `diagramStore.initializeUndoManager`
  (line 108-137) over four Y.Maps: nodes, edges, assessments, metadata.
  It runs entirely against the in-memory `Y.Doc`. No network IO.
- The constructor only calls `initializeUndoManager` when
  `mode === Modelling && !options.collaborationEnabled`
  (`besser-editor.tsx:143-148`). The webapp never sets
  `collaborationEnabled`, so undo/redo is always available locally.

In short: the library is `Y.Doc + UndoManager + observer-broadcast hooks`.
There is no transport. The webapp does not enable any.

---

## 3. Editor lifecycle

`constructor → render → ready resolves → model setter → destroy`:

1. **Constructor (`besser-editor.tsx:59-169`)** — Validates the
   container, allocates the readiness promise (`readyPromise =
   new Promise(r ⇒ resolveReady = r)`) **before** calling
   `root.render(...)`. Builds the `Y.Doc`, the five Zustand stores, the
   `YjsSyncClass`, applies all `options.*` defaults (mode, view,
   readonly, debug, scrollLock, enablePopups, model). Conditionally
   bootstraps the `UndoManager`. Mounts `<AppWithProvider
   onReactFlowInit={this.setReactFlowInstance.bind(this)} />`.
2. **Render** — `<AppWithProvider />` initialises `<ReactFlow />`; React
   Flow fires `onInit(instance)` → `setReactFlowInstance(instance)`
   stores the instance and **calls `resolveReady()`** (line 176).
3. **`ready` resolves** — Any code holding a reference to
   `editor.ready` unblocks. Webapp does so at
   `BesserEditorComponent.tsx:117` (`await nextEditor.nextRender`).
4. **`model` setter** — Pushes new nodes/edges/assessments/interactive
   into the existing `diagramStore` and updates metadata. The React
   tree, the `Y.Doc`, the subscriber map, and the `UndoManager` are
   **preserved**. **Implication**: setting `model` does *not* clear
   undo history; nor does it re-fire `ready`. If the host app needs a
   true reset, it must either call `destroy()` + `new BesserEditor(...)`
   (which is what the webapp does via the `editorRevision` Redux
   counter — see §5) or manage the `UndoManager` separately.
5. **`destroy()`** — Steps in §1.1. The whole body is wrapped in
   `try/catch { /* ignore */ }` (lines 220-235), so it cannot throw.
   `subscribers` is cleared synchronously **before** `root.unmount()`,
   which means subscriber callbacks fired during the unmount tick are
   suppressed (Zustand's unsubscribe is sync). Zustand stores are not
   explicitly disposed — they are GC'd once the `BesserEditor`
   reference drops, which is the documented Zustand pattern.

The lifecycle behaves as advertised. No leaks identified in the cleanup
path.

---

## 4. Subscription semantics

- **`subscribeToModelChange`** — Backed by Zustand's
  `store.subscribe(listener)`, which fires the listener **synchronously
  immediately after** `set(...)` is called inside the store, on the
  same tick as the mutation. Every drag, every text edit that lands in
  the store triggers the callback **synchronously after the mutation
  commits**. The webapp guards against the resulting torrent with a
  300 ms `setTimeout` debounce
  (`BesserEditorComponent.tsx:129-134`).
- **`subscribeToDiagramNameChange`** — Same store-level semantics,
  applied to `metadataStore`. Fires synchronously when
  `updateDiagramTitle` is called, including programmatic updates.
- **`subscribeToAssessmentSelection`** — Same pattern over the
  assessment-selection store; fires whenever the listener mutates
  `selectedElementIds`. Used by assessment-mode UIs (no webapp call
  site today).
- **Selection on click** — Selection state is mutated by ReactFlow's
  `onNodesChange` / `onEdgesChange` handlers in `diagramStore`
  (`diagramStore.ts:360-396`, `:467-502`), which run synchronously
  during the click tick. There is **no** dedicated
  `subscribeToSelectionChange`, so today the webapp would have to
  watch model-change callbacks (which fire on every selection-related
  store update because nodes are part of the same store) to observe
  selection. This is wasteful but correct; it explains why the
  selection-change subscription is on the SA-7 follow-up list.

---

## 5. Webapp integration — `BesserEditorComponent.tsx`

Path: `packages/webapp/src/main/features/editors/uml/BesserEditorComponent.tsx`.
The component drives the editor through three coordinating refs and one
revision counter:

- **`destroyEditorDeferred`** (lines 39-52) — Wraps `editor.destroy()` in
  a `setTimeout(0, …)` to push it to the next macrotask, dodging React's
  "unmount during render" warnings when a destroy lands inside a render
  transition. Resolves a promise so `cleanupEditor` can `await` it.
- **`cleanupEditor`** (54-70) — Clears the pending debounced save,
  detaches the model subscription via the **deprecated**
  `unsubscribeFromModelChange(id)` alias (line 66), then awaits
  `destroyEditorDeferred(editor)`.
- **`debouncedSave`** (line 21, used at 130-134) — A single
  `setTimeout` ref. Every model-change callback resets the timer (300
  ms) before dispatching `updateDiagramModelThunk({ model })`. Without
  this, every Yjs-driven keystroke would write to localStorage.
- **`editorRevision`-driven setup** (99-140) — The component watches
  `editorRevision` from `workspaceSlice`. When it changes (and is
  non-zero), it:
  1. Bumps `setupRunRef.current` to invalidate any in-flight setup.
  2. Records `lastHandledRevisionRef.current = editorRevision` to
     dedupe re-runs.
  3. `await cleanupEditor()` to tear down the previous instance.
  4. Constructs a fresh `new BesserEditor(container, options)`.
  5. `await nextEditor.nextRender` — this is the **deprecated alias**;
     should migrate to `await nextEditor.ready` once SA-7 sweeps the
     callers.
  6. Re-checks `runId === setupRunRef.current` to abort if a newer
     revision arrived during the await.
  7. Loads `currentDiagram.model` via the `model` setter (line 125).
     Because the setter does not recreate the editor, this is fine —
     the editor was already fresh from step 4.
  8. Re-subscribes via `subscribeToModelChange(...)` and stashes the
     subscriber id in `modelSubscriptionRef`.
  9. `setEditor(nextEditor)` exposes the instance to other features via
     `BesserEditorContext`.

The `editorRevision` counter is the webapp's mechanism for "true reset"
— bumping it forces a full destroy/recreate. Per the frontend
`CLAUDE.md`, that counter **must not** be bumped for view-only toggles
because it clears undo history.

**Verdict for the integration**: it consumes the surface correctly. Two
debt items (already on the SA-7 list):

1. Webapp still calls `nextRender` and `unsubscribeFromModelChange` —
   both are tagged `@deprecated` in the library and aliased to the new
   names. Functional today; cosmetic SA-7 sweep.
2. The component re-saves the entire model on every store change. With
   `subscribeToSelectionChange` available, the debounced-save handler
   could ignore pure-selection mutations.

---

## 6. Top gaps in the public API

1. **No `select(ids)` / `subscribeToSelectionChange(cb)` / no error
   subscription bus.** `select(ids)` and `subscribeToSelectionChange`
   are five-line additions over the existing `diagramStore` (the state
   already exists; only the editor-class wrappers are missing).
   `subscribeToApollonErrors` (and its rename) is greenfield: there is
   no error store, no error listener registry, no `BesserError` /
   `ApollonError` symbol anywhere in the library. None of these block
   the webapp today, but they are part of the v3 public surface that
   the migration plan committed to preserving.
2. **No external `undo()` / `redo()` methods.** The `UndoManager` is
   fully wired (Y.Maps tracked, capture timeout configured, stack
   listeners installed) and reachable via `diagramStore.getState().undo()
   / .redo()`, but the editor class itself exposes neither. External
   callers (toolbar buttons, hotkey overrides) cannot drive undo/redo
   without reaching through the private store. Two-line fix.
3. **The `model` setter does not reset undo history, but `destroy + new`
   does.** This is the right design, but it is not surfaced in the API
   docs. Combined with the webapp's `editorRevision` mechanism, the
   contract is: "bump editorRevision when you need a clean undo
   history, otherwise just assign `model`". The
   `static exportModelAsSvg`'s `theme` parameter is also a documented
   no-op (`void theme` at line 249) — minor surface-area drift versus
   v3's typed theme override.

---

## Files referenced

- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/besser-editor.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/index.tsx`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/typings.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/sync/yjsSyncClass.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/library/lib/store/diagramStore.ts`
- `besser/utilities/web_modeling_editor/frontend/packages/library/package.json`
- `besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/features/editors/uml/BesserEditorComponent.tsx`
- `docs/source/migrations/api-surface-diff.md`
