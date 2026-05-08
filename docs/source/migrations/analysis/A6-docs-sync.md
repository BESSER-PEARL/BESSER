# A6 — Documentation Sync Audit (post-migration)

**Wave:** Final analysis
**Scope:** `docs/source/**/*.rst` (Sphinx). Migration artifacts under
`docs/source/migrations/` are intentionally excluded — they document the
migration itself.
**Goal:** Find any RST page that still references v3-shape (`model.elements` /
`model.relationships`), the old `ApollonEditor` class, the v2.5 npm package
(`@besser/wme@2.5.0`), the old editor module path
(`packages/editor/src/main/index.ts`), the editor's webpack build, or dropped
APIs (`importPatch`, `subscribeToModelChangePatches`, `nextRender`, …).

---

## Summary

- **Files scanned:** 150 RST files under `docs/source/` (excluding
  `docs/source/releases/**` historical changelogs and
  `docs/source/migrations/**` migration artifacts).
- **Clean (✅):** 149
- **Needs update (❌):** 1
  - `docs/source/contributing/diagram_dsl_workflow.rst` — Step 2 lists 11 file
    paths under the **old** `packages/editor/src/main/...` tree. Those files
    do not exist in the v3 layout; the active source tree is
    `packages/library/lib/...`.

No occurrences of `model.elements`, `model.relationships`, `ApollonEditor`,
`@besser/wme@2.5`, `importPatch`, `subscribeToModelChangePatches`,
`nextRender`, or webpack-as-the-editor-build were found in any user-facing
RST page.

### Top 3 stalest docs

1. **`contributing/diagram_dsl_workflow.rst`** — 11 stale `packages/editor/...`
   path references (all in “Step 2: Add the Diagram Type in the Editor
   Package”). This is the only RST page with stale post-migration content.
2. *(no second stale doc — everything else is clean.)*
3. *(no third stale doc.)*

The remaining ~149 RST files were verified clean against all six checks.

---

## Per-file results

### Top-level pages

| File | Result | Notes |
|---|---|---|
| `docs/source/about.rst` | ✅ clean | No editor / shape refs. |
| `docs/source/ai_assistant_guide.rst` | ✅ clean | Workflow doc; only references `besser/utilities/web_modeling_editor/frontend` as the submodule mount path (still correct). |
| `docs/source/api.rst` | ✅ clean | Pure toctree. |
| `docs/source/buml_language.rst` | ✅ clean | Metamodel description; no editor coupling. |
| `docs/source/besser_action_language.rst` | ✅ clean | |
| `docs/source/contributor_guide.rst` | ✅ clean | Lines 70-91, 166-190 describe the submodule + `npm run dev` + Vite dev server on `:8080` — already up to date with v3 (Vite, not webpack). |
| `docs/source/examples.rst` | ✅ clean | |
| `docs/source/generators.rst` | ✅ clean | Generator catalog; no editor coupling. |
| `docs/source/index.rst` | ✅ clean | |
| `docs/source/installation.rst` | ✅ clean | |
| `docs/source/releases.rst` | ✅ clean | (Toctree only.) |
| `docs/source/troubleshooting.rst` | ✅ clean | |
| `docs/source/utilities.rst` | ✅ clean | |
| `docs/source/web_editor.rst` | ✅ clean | Mentions Apollon **only** in a footer note as historical context (line 134-135: *“based on a fork of the Apollon project”*). That is a true historical statement, not a stale API reference. The page does not document `model.elements`/`model.relationships` or the old class. |
| `docs/source/web_editor_backend.rst` | ✅ clean | NN section at line 49-50 talks about an NN diagram JSON whose `elements`/`relationships` describe layers. This refers to the **JSON payload shape** the backend receives, not the deprecated `model.elements` / `model.relationships` v3 client-side shape — and that JSON wire shape is still what the v4 frontend serializes. No change needed. |

### `docs/source/contributing/`

| File | Result | Notes |
|---|---|---|
| `contributing/index.rst` | ✅ clean | |
| `contributing/areas.rst` | ✅ clean | |
| `contributing/create_dsl.rst` | ✅ clean | Line 42 references `packages/webapp/src/main/features/project/ADDING_NEW_DIAGRAM_TYPE.md` — webapp path, still correct in v3. |
| `contributing/create_generator.rst` | ✅ clean | |
| `contributing/diagram_dsl_workflow.rst` | ❌ **needs update** | Step 2 lists 11 paths under `packages/editor/src/main/...`. See patch below. |

### `docs/source/buml_language/**` (15 files)

All ✅ clean. These describe the metamodel (structural, state machine, GUI,
agent, NN, quantum, OCL, deployment, feature_model, object) and are
editor-agnostic.

### `docs/source/generators/**` (22 files)

All ✅ clean. Generator-specific docs (Django, FastAPI/Backend, React, Flutter,
Pydantic, Java, Python, JSONSchema, RDF, Terraform, SQL, SQLAlchemy, Qiskit,
PyTorch, TensorFlow, BAF, agent_personalization, full_web_app, build_generator,
django/django_admin_panel, django/django_ui_components, rest_api). No editor
coupling.

### `docs/source/api/**` (24 files)

All ✅ clean. Auto-doc style references to Python modules under `besser.*`.

### `docs/source/utilities/**` (2 files)

All ✅ clean. (`buml_code_builder.rst`, `serializer.rst`.)

### `docs/source/besser_action_language/**` (2 files)

All ✅ clean.

### `docs/source/examples/**` (10 files)

All ✅ clean. Code-based examples; no editor API coupling.

### `docs/source/releases/**` (60+ files)

**Skipped — historical changelogs.** Per the audit scope, release notes are
allowed to reference whatever shipped at the time (and `releases/v2/v2.5.0.rst`
correctly documents the `@besser/wme@2.5.0` release).

### `docs/source/migrations/**`

**Skipped — migration artifacts** (`uml-v4-shape.md`, `parity-audit-wave2.md`,
`parity-final/*.md`, `api-surface-diff.md`, `webapp-cutover-checklist.md`,
`backend-smoke-result.md`, plus prior `analysis/A*.md`).

---

## Detail — the one stale file

### `docs/source/contributing/diagram_dsl_workflow.rst`

#### Stale references (Step 2, lines 51–75)

| Line | Stale reference | Replacement |
|---|---|---|
| 51 | "These files live in the WME repo under ``packages/editor``" | "These files live in the WME repo under ``packages/library`` (the v3 monorepo split: the editor used to live in ``packages/editor`` but the active diagramming engine is now ``packages/library``)." |
| 55 | ``packages/editor/src/main/packages/diagram-type.ts`` | ``packages/library/lib/types/DiagramType.ts`` |
| 56 | ``packages/editor/src/main/uml-element-type.ts`` | ``packages/library/lib/nodes/types.ts`` |
| 57 | ``packages/editor/src/main/uml-relationship-type.ts`` | ``packages/library/lib/edges/types.tsx`` |
| 61 | "Create ``packages/editor/src/main/packages/<your-diagram>/``" | "Create node renderers under ``packages/library/lib/nodes/<yourDiagram>/`` and edge renderers under ``packages/library/lib/edges/edgeTypes/`` (one diagram type may need both)." |
| 66 | ``packages/editor/src/main/packages/components.ts`` | (removed — React Flow registers nodes/edges via ``nodeTypes`` / ``edgeTypes`` in ``packages/library/lib/besser-editor.tsx``; there is no longer a single component map. Mention ``packages/library/lib/nodes/types.ts`` and ``packages/library/lib/edges/types.tsx`` instead.) |
| 67 | ``packages/editor/src/main/packages/uml-elements.ts`` | ``packages/library/lib/nodes/types.ts`` (node-type registration) |
| 68 | ``packages/editor/src/main/packages/uml-relationships.ts`` | ``packages/library/lib/edges/types.tsx`` (edge-type registration) |
| 69 | ``packages/editor/src/main/packages/compose-preview.ts`` | ``packages/library/lib/components/Sidebar.tsx`` (palette previews now live with the sidebar component) |
| 70 | ``packages/editor/src/main/packages/popups.ts`` | ``packages/library/lib/components/inspectors/<yourDiagram>/`` (per-diagram inspector folders, e.g. ``classDiagram/ClassEditPanel.tsx``, ``agentDiagram/...``) |
| 71 | ``packages/editor/src/main/components/create-pane/create-pane.tsx`` | (removed — diagram-type selection is in the **webapp**, not the library: see ``packages/webapp/src/main/features/project/...``. The library no longer owns the “create pane”.) |
| 75 | ``packages/editor/src/main/i18n/en.json`` | The library was de-i18n’d in v3. User-facing strings live with the React components in ``packages/library/lib/components/`` (no separate i18n bundle). If localisation is needed, raise the string to the **webapp** layer instead. |

#### Other observations on this file (not stale, but worth flagging)

- **Line 17**: *“Diagram type already exists in the editor package
  (``packages/editor``)?”* — should read ``packages/library`` to match v3.
- **Line 192** (Common Pitfalls): *“The webapp sidebar shows the diagram but
  the editor package does not know it.”* — replace **editor package** with
  **library package** (`packages/library`). Functionally the same warning.
- The Step 2 *Tip* at line 77 (“Follow patterns from existing diagram packages
  e.g. ``uml-class-diagram``”) is also stale — there is no longer a
  ``uml-class-diagram`` *package*. Suggest pointing at
  ``packages/library/lib/nodes/classDiagram/`` and
  ``packages/library/lib/edges/edgeTypes/ClassDiagramEdge.tsx`` as the
  reference pattern.

---

## Suggested patch (unified diff)

> **Apply manually** — A6 is read-only on `docs/source/*.rst`. This is the
> recommended replacement for the single stale file.

```diff
--- a/docs/source/contributing/diagram_dsl_workflow.rst
+++ b/docs/source/contributing/diagram_dsl_workflow.rst
@@ -14,11 +14,11 @@
 Decision Tree
 -------------

-1. **Diagram type already exists in the editor package** (``packages/editor``)?
+1. **Diagram type already exists in the library package** (``packages/library``)?

-   * Yes: only do the **webapp wiring** (sidebar, project model, import/export labels).
-   * No: you must add **editor package support** + **webapp wiring** + **backend processing**.
+   * Yes: only do the **webapp wiring** (sidebar, project model, import/export labels).
+   * No: you must add **library package support** + **webapp wiring** + **backend processing**.

 2. **New DSL / new semantics**?

@@ -29,7 +29,7 @@

 * **WME repo (frontend, diagram engine)**

-  - Editor package (diagram types, elements, rendering, palette, property panels)
+  - Library package (diagram types, nodes, edges, rendering, palette, inspectors)
   - Webapp (project model, sidebar, import/export, UI)

 * **BESSER repo (backend, BUML)**
@@ -45,38 +45,40 @@
 3. Add or update tests under ``tests/BUML`` and ``tests/BUML_invalid``.
 4. Update docs under ``docs/source/buml_language``.

-Step 2: Add the Diagram Type in the Editor Package (WME repo)
--------------------------------------------------------------
+Step 2: Add the Diagram Type in the Library Package (WME repo)
+--------------------------------------------------------------

-These files live in the WME repo under ``packages/editor``:
+These files live in the WME repo under ``packages/library`` (the v3 monorepo
+split — the active diagramming engine, published as ``@besser/wme`` v3.x):

-1. **Register diagram and element types**
+1. **Register diagram, node, and edge types**

-   * ``packages/editor/src/main/packages/diagram-type.ts``
-   * ``packages/editor/src/main/uml-element-type.ts``
-   * ``packages/editor/src/main/uml-relationship-type.ts`` (if you add new relationships)
+   * ``packages/library/lib/types/DiagramType.ts`` — diagram-type enum
+   * ``packages/library/lib/nodes/types.ts`` — React Flow node-type registry
+   * ``packages/library/lib/edges/types.tsx`` — React Flow edge-type registry

-2. **Create a new diagram package folder**
+2. **Create node and edge folders for the diagram**

-   * Create ``packages/editor/src/main/packages/<your-diagram>/``
-   * Add element classes (model), React components (rendering), and palette previews.
+   * Add node renderers under ``packages/library/lib/nodes/<yourDiagram>/``.
+   * Add edge renderers under ``packages/library/lib/edges/edgeTypes/``
+     (one ``<YourDiagram>Edge.tsx`` per relationship category).
+   * Add inspectors under ``packages/library/lib/components/inspectors/<yourDiagram>/``.

-3. **Wire registries**
+3. **Wire the React Flow registries**

-   * ``packages/editor/src/main/packages/components.ts`` (component map)
-   * ``packages/editor/src/main/packages/uml-elements.ts`` (element map)
-   * ``packages/editor/src/main/packages/uml-relationships.ts`` (relationship map)
-   * ``packages/editor/src/main/packages/compose-preview.ts`` (palette previews)
-   * ``packages/editor/src/main/packages/popups.ts`` (property panels)
-   * ``packages/editor/src/main/components/create-pane/create-pane.tsx`` (diagram type selection)
+   * ``packages/library/lib/nodes/types.ts`` — register node component
+   * ``packages/library/lib/edges/types.tsx`` — register edge component
+   * ``packages/library/lib/components/Sidebar.tsx`` — palette previews
+   * ``packages/library/lib/components/inspectors/<yourDiagram>/`` —
+     property panels (one ``*EditPanel.tsx`` per editable element)
+   * ``packages/library/lib/besser-editor.tsx`` — top-level editor wires
+     ``nodeTypes`` / ``edgeTypes`` into React Flow.
+
+   Diagram-type *selection* (the “create new diagram” UI) lives in the
+   **webapp**, not the library — see Step 3.

-4. **Translations**
-
-   * ``packages/editor/src/main/i18n/en.json`` (and other locales if needed)
-
-Tip: Follow patterns from existing diagram packages (e.g. ``uml-class-diagram``).
+Tip: Follow patterns from an existing diagram, e.g.
+``packages/library/lib/nodes/classDiagram/`` +
+``packages/library/lib/edges/edgeTypes/ClassDiagramEdge.tsx`` +
+``packages/library/lib/components/inspectors/classDiagram/``.

 Step 3: Wire the Diagram into the Webapp (WME repo)
 ---------------------------------------------------
@@ -187,9 +189,9 @@
 Common Pitfalls
 ---------------

-* The webapp sidebar shows the diagram but the editor package does not know it.
-* JSON type strings in the backend do not match the editor package.
+* The webapp sidebar shows the diagram but the library package does not know it.
+* JSON type strings in the backend do not match the library package.
 * Backend is not running on ``http://localhost:9000/besser_api`` when you test.
 * Existing projects do not include the new diagram slot (create a new project).
```

---

## Cross-checks performed

The following greps were run across `docs/source/**/*.rst` (excluding
`releases/` and `migrations/`):

| Pattern | Hits | File(s) |
|---|---|---|
| `model\.elements` | 0 | — |
| `model\.relationships` | 0 | — |
| `ApollonEditor` | 0 | — |
| `@besser/wme` | 0 | — |
| `packages/editor/src` | **11** | `contributing/diagram_dsl_workflow.rst` (lines 55–75) |
| `webpack` (as editor build) | 0 | — |
| `importPatch` | 0 | — |
| `subscribeToModelChangePatches` | 0 | — |
| `nextRender` | 0 | — |
| `umlModel`, `UMLModel`, `Apollon ` (loose) | 1 | `web_editor.rst:135` — historical *“based on a fork of the Apollon project”* note. **Not stale** — true statement of origin. |
| `v2.5`, `2.5.0` | 0 in active docs (only in `releases/v2/v2.5.*.rst`, which is the actual release note for that version) | — |

The web-editor backend's NN-section mention of "elements/relationships" at
`web_editor_backend.rst:49-50` was reviewed and confirmed to describe the
**JSON wire shape** posted to `/validate-diagram` and `/export-buml`, which is
unchanged by the v3→v4 client-side migration. Not flagged.

---

## Recommendation

Apply the unified diff above to `contributing/diagram_dsl_workflow.rst` in a
follow-up `docs:` commit. No other RST files require changes after the
v2.5→v3.0 / Apollon→Besser / webpack→Vite migration.

— A6
