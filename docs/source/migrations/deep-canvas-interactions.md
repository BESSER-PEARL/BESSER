# Deep Canvas Interactions Audit (SA-DEEP-COPY-PASTE-MULTI)

Audit of canvas interactions in v4 (`packages/library/`) versus v3 (`packages/editor/src/main/`):
selection, multi-select, copy/paste/duplicate, delete, keyboard shortcuts, context
menus, toolbar buttons, palette drops, and re-parenting.

> **Scope.** v4 = the React-Flow / Zustand / Yjs library at
> `besser/utilities/web_modeling_editor/frontend/packages/library/lib/`.
> v3 = the legacy Apollon canvas at
> `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/`.

## Per-interaction matrix

| # | Interaction | v4 implemented? | v4 file(s) | v3 behaviour / file(s) | Notes |
|---|---|---|---|---|---|
| 1 | **Click selection** populates `diagramStore.selectedElementIds` | YES | `library/lib/store/diagramStore.ts` (`onNodesChange`/`onEdgesChange` `select` branch L409-L450, L516-L555); `usePaneClicked.ts` clears it | `services/uml-element/uml-element-repository.ts` `select`/`deselect` thunks dispatched from `canvas/keyboard-eventlistener.tsx` and selectable mixin | Functional. Also kept mirrored in `node.selected`/`edge.selected`. |
| 2 | **Box (drag-rectangle) multi-select** | NO — relies on stock React-Flow defaults only | `library/lib/App.tsx` (no `selectionOnDrag`, `selectionMode`, `selectionKeyCode`, `panOnDrag` props passed) | `components/canvas/mouse-eventlistener.tsx` (custom `selectionRectangle` state, L39-L220) | React-Flow default needs Shift held; with `panOnDrag=true` (default) plain drag pans. v3 supported plain-drag marquee select. |
| 3 | **Cmd/Ctrl+A — Select all** | YES | `library/lib/hooks/useKeyboardShortcuts.ts` L81-L86; `useSelectionForCopyPaste.ts` `selectAll` L39-L48 | `keyboard-eventlistener.tsx` L136-L138 → `UMLElementRepository.select()` | Parity. |
| 4 | **Cmd/Ctrl+C — Copy** | YES | `useKeyboardShortcuts.ts` L88-L96; `useSelectionForCopyPaste.ts` `copySelectedElements` L56-L75; `utils/copyPasteUtils.ts` (`createClipboardData`) | `keyboard-eventlistener.tsx` L140-L151; `services/copypaste/copy-repository.ts` | v4 uses `navigator.clipboard.writeText` (JSON) — only works in secure context. v3 used Redux-only `CopyState` (no OS clipboard). v4 lacks v3’s native-text-selection escape hatch (v3 lets browser keep copy if a DOM text selection exists). |
| 5 | **Cmd/Ctrl+V — Paste with offset and new IDs** | YES | `useSelectionForCopyPaste.ts` `pasteElements` L77-L214 | `services/copypaste/copy-repository.ts` `paste` | v4 generates new UUIDs for nodes/edges, offsets by `CANVAS.PASTE_OFFSET_PX × pasteCount`, regenerates child attribute/method/actionRow IDs, preserves parent/child via `nodeIdMap`, drops edges whose endpoints are not in the clipboard. |
| 6 | **Cmd/Ctrl+X — Cut** | YES | `useKeyboardShortcuts.ts` L98-L106; `useSelectionForCopyPaste.ts` `cutSelectedElements` L216-L260 | Not present in v3 | v4-only feature. |
| 7 | **Backspace / Delete — delete selected** | PARTIAL | `useKeyboardShortcuts.ts` L49-L56 only handles **Delete**. `Backspace` falls through to React-Flow’s default `deleteKeyCode="Backspace"` + `onBeforeDelete` (`useElementInteractions.ts` L30-L33). Edge cleanup on node remove is in `diagramStore.onNodesChange` L479-L489 via `getConnectedEdges`. Selection-aware bulk delete uses `getEdgesToRemove`/`getAllNodesToInclude` (`utils/copyPasteUtils.ts`). | `keyboard-eventlistener.tsx` L124-L128 calls `UMLElementRepository.delete()` for both Backspace & Delete | v4 has two divergent code paths for delete: (a) custom hook → `deleteSelectedElements` (descendants + connected edges removed), (b) React-Flow default Backspace → only the directly-selected nodes (descendants are not cascaded; only connected edges are). This is inconsistent. |
| 8 | **Cmd/Ctrl+Z / Y — Undo / Redo (Yjs UndoManager)** | YES | `useKeyboardShortcuts.ts` L64-L80; `diagramStore.ts` `initializeUndoManager`/`undo`/`redo` L144-L203; `CustomControls.tsx` undo/redo buttons | `services/undo/undo-repository.ts` (Redux time-travel) | v4 wraps `Y.UndoManager` over `nodesMap/edgesMap/assessmentsMap` with 500 ms capture timeout, tracked origin `"store" | "remote"`. |
| 9 | **Arrow keys — nudge selected nodes (1 / 10 px with Shift)** | NO | _no implementation_ — `useKeyboardShortcuts.ts` does not handle `ArrowUp/Down/Left/Right` | `keyboard-eventlistener.tsx` L96-L123 (10 px nudge per key — no Shift modifier) and L164-L173 `keyUp` to commit move | Lost in v4. v3 also did not have a 1 px Shift variant; v4 just has nothing. |
| 10 | **Right-click context menu (canvas / node / edge)** | NO | _no implementation_. App.tsx does not pass `onPaneContextMenu`, `onNodeContextMenu`, or `onEdgeContextMenu`. The only `onContextMenu` in the lib is in `inspectors/userDiagram/UserModelNameEditPanel.tsx:273` (inspector text field, not canvas). | _no implementation in v3 either_ | Both versions lack a custom context menu. Browser default menu appears on right-click. |
| 11 | **Toolbar buttons — flip / color / edit pencil** | YES — but gated and split between two surfaces | Floating: `components/toolbars/NodeToolbar.tsx` (Delete + Edit), `components/toolbars/edgeToolBar/CustomEdgeToolBar.tsx` (Delete + Edit). Edit pencil is **hidden** when `usePropertiesPanel=true` (PC-11.1). Flip + color live in inspector panels (`components/inspectors/classDiagram/ClassEdgeEditPanel.tsx` L142, `agentDiagram/AgentDiagramEdgeEditPanel.tsx` L266-L272, `stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx` L111-L117, `objectDiagram/ObjectLinkEditPanel.tsx` L106, L136) | v3 had a single floating toolbar. | No flip/color affordance on the **floating** toolbar — must open the side panel. Node toolbar has no color picker / flip at all (flip is edges-only). |
| 12 | **Palette drag → set `parentId` when dropped on a parent (NN-layer into NNContainer etc.)** | YES | `components/DraggableGhost.tsx` L122-L138, L155-L159 (`getIntersectingNodes` filtered by `isParentNodeType` and `canDropIntoParent`); `resizeAllParents` called L179-L181 | v3 sidebar drag → `services/uml-container` parent assignment | `bpmnConstraints.canDropIntoParent` whitelists which children fit which container types (BPMN pools/lanes, NN containers, etc.). |
| 13 | **Drag a child outside its parent — re-parent or auto-grow?** | RE-PARENT (and grow remaining parent) | `hooks/useNodeDragStop.ts` L29-L125: if drop point intersects a new compatible parent → re-parent + adjust position; if no parent intersection → strip `parentId`; if same parent → `resizeAllParents` to grow | v3 grew the parent on internal drag, ejected on outside drop | Behaviour is consistent — but there is no auto-grow when dragging a child near (but inside) the parent edge; only `resizeAllParents` when explicitly inside the parent body. |

## Missing / broken interactions

| Severity | Item | Impact |
|---|---|---|
| **High** | **Arrow-key nudging (#9)** is completely gone. v3 supported 10 px per key. The brief asked for 1 / 10 px (Shift) — neither variant is implemented in v4. | Keyboard accessibility regression; can’t fine-tune layout without dragging. |
| **High** | **Box / marquee selection (#2)** depends entirely on React-Flow defaults. With `panOnDrag` defaulting to `true` and no `selectionOnDrag`/`selectionMode` configured, plain-drag on the canvas pans rather than rubber-bands. Users must hold Shift to draw a selection rectangle. v3’s `mouse-eventlistener.tsx` had a full custom marquee. | Multi-select discoverability is poor; not at parity with v3. |
| **High** | **Backspace vs Delete inconsistency (#7)**. Custom hook handles only `Delete` (and cascades via `getAllNodesToInclude`). `Backspace` falls back to React-Flow’s built-in delete, which only removes directly-selected nodes (plus the edges automatically removed by `diagramStore.onNodesChange`’s `getConnectedEdges` branch). Two key paths produce different delete semantics for descendants. | Functional bug — Backspace on a parent leaves orphan children. |
| **Medium** | **No right-click context menu (#10)** for canvas, node, or edge. ReactFlow exposes `onPaneContextMenu`, `onNodeContextMenu`, `onEdgeContextMenu`, but `App.tsx` wires none of them. v3 had none either, but the brief explicitly asks. | No discoverable Cut/Copy/Paste/Duplicate/Delete actions outside the keyboard. |
| **Medium** | **No "Duplicate" command** (Cmd/Ctrl+D). The shortcut is wired in `useKeyboardShortcuts.ts` L116-L121 but it calls `clearSelection()` instead of duplicating. The label "case 'd'" with `clearSelection()` looks like a leftover bug (likely meant to be "deselect all" but unconventional binding for that — Escape already deselects on L43-L47). | Confusing keybinding; no duplicate path. |
| **Medium** | **Floating NodeToolbar lacks flip/color** (#11). All edge/node colour + edge-flip controls live only in the inspector panel. With `usePropertiesPanel=true` the inspector replaces the pencil; with it `false` the user has to double-click to open the popover before changing colour. | Two-click overhead vs. v3 single-click toolbar. |
| **Low** | **Clipboard-only copy** (#4). `copySelectedElements` returns `false` outside `window.isSecureContext` and there is no in-memory fallback (v3 used a Redux `CopyState`). On `http://` deployments and some test harnesses copy/paste silently no-ops. | Edge-case break. |
| **Low** | **Copy hijacks DOM text selection.** v4 always `event.preventDefault()`s on Cmd/Ctrl+C if any element is selected on the canvas. v3 explicitly let the browser keep the native copy when `window.getSelection()` had text (`keyboard-eventlistener.tsx` L144-L147). | Can’t copy error-toast text while a node is also selected. |
| **Low** | **No auto-grow during drag**, only on `dragStop`. While dragging a child past the parent edge, the parent stays at its old size and the child appears to spill until release. | Cosmetic; matches v3. |

## Files of record (v4)

- Canvas wiring: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/App.tsx`
- Diagram store + Yjs: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/store/diagramStore.ts`
- Keyboard: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/useKeyboardShortcuts.ts`
- Selection / clipboard: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/useSelectionForCopyPaste.ts`
- Clipboard helpers: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/copyPasteUtils.ts`
- Pane click clears selection: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/usePaneClicked.ts`
- Single-element delete: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/useHandleDelete.ts`
- Drag-stop / re-parenting: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/useNodeDragStop.ts`
- Drag-start / alignment guides: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/hooks/useNodeDrag.ts`
- Palette drop / parent assignment: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/DraggableGhost.tsx`
- Drop-allowed predicate: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/utils/bpmnConstraints.ts`
- Floating toolbars: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/toolbars/NodeToolbar.tsx`, `.../toolbars/edgeToolBar/CustomEdgeToolBar.tsx`
- Inspector flip/color: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/classDiagram/ClassEdgeEditPanel.tsx`, `.../objectDiagram/ObjectLinkEditPanel.tsx`, `.../stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx`, `.../agentDiagram/AgentDiagramEdgeEditPanel.tsx`

## Files of record (v3)

- Keyboard: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/components/canvas/keyboard-eventlistener.tsx`
- Marquee selection: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/components/canvas/mouse-eventlistener.tsx`
- Copy/paste: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/services/copypaste/`
- Undo/redo: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/services/undo/`
- Element CRUD: `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/services/uml-element/`

## Summary counts

- **Implemented**: 9 / 13 (#1, #3, #4, #5, #6, #8, #11 (partial), #12, #13)
- **Partial**: 1 / 13 (#7 — Delete works fully, Backspace falls back to RF default)
- **Missing**: 3 / 13 (#2 plain-drag marquee, #9 arrow-key nudge, #10 context menu)
