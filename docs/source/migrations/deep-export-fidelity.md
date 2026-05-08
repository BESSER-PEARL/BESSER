# Deep Export / Import Fidelity Audit

Audited: 2026-05-08.
Branch: `claude/refine-local-plan-sS9Zv`.
Frontend submodule: `besser/utilities/web_modeling_editor/frontend` (current pointer).

Code paths audited:

- `BesserEditor.exportAsSVG` / `BesserEditor.exportModelAsSvg` — `packages/library/lib/besser-editor.tsx:259-382`
- SVG builder `getSVG` / `filterRenderedElements` / `getRenderedDiagramBounds` — `packages/library/lib/utils/exportUtils.ts`
- Webapp export hooks — `packages/webapp/src/main/features/export/{useExportSvg,useExportPng,useExportJson,useExportBuml,useExportProjectJSON,useExportProjectBUML}.ts`
- Webapp import hooks — `packages/webapp/src/main/features/import/{useImportDiagram,useImportDiagramPicture,useBumlToDiagram}.ts`
- Backend conversion — `besser/utilities/web_modeling_editor/backend/routers/conversion_router.py` (`/export-buml`, `/get-json-model`, `/get-json-model-from-image`)
- BUML→JSON converters return `version: "4.0.0"` (verified: `class_diagram_converter.py:526`, `agent_diagram_converter.py:681,694`, `nn_diagram_converter.py:474`, `object_diagram_converter.py:278`, `project_converter.py:56`).

---

## Top-line verdict

**PARTIAL.** JSON round-trips cleanly (the wire shape matches `editor.model` byte-for-byte, modulo the new `id` assigned on import). SVG and PNG are best-effort renderings — they are NOT round-trip imports (no SVG/PNG → diagram path exists) and the SVG is *not* a fully-standalone document: it lacks `<?xml ... ?>` and `<!DOCTYPE>` declarations and depends on host CSS variables in its default `web` mode. PNG rasterizes via `canvas.drawImage(svgBlob)` at `1.5×` of the bounds clip — captures the *full diagram* (since the export renders into an off-DOM 4000×4000 container, not the user's viewport), so zoom level is irrelevant; however it cannot read CSS-variable colours from the embedding page when consumed by the browser image-decoder, so anything left in `web` mode renders with the default fallbacks instead of the user's theme. BUML export is a one-way lossy transform (no .py → identical-JSON round-trip). The JSON import has **no v3 detection branch** in `useImportDiagram` / `useImportDiagramToProject` — both call `isUMLModel(diagram.model)`, which returns `false` for v3 models (they have `elements`/`relationships`, not `nodes`/`edges`), so a stored-disk v3 diagram JSON will be rejected with "Invalid diagram: missing model or type information" rather than migrated. Migration only happens at the project-storage layer (`migrateProjectToV5` in `shared/types/project.ts`), not at the per-file import boundary.

---

## Section A — Per-format export → re-import round-trip table

| Format | Export entry point | Re-import entry point | Round-trip identical? | Notes |
|---|---|---|---|---|
| **JSON (single diagram)** | `useExportJSON` (`useExportJson.ts:9-22`) writes `{ ...diagram, model: editor.model }` | `useImportDiagram` (`useImportDiagram.ts:24-35`) parses, assigns new `uuid()` to `id`, stores | **YES**, modulo `id` (always rotated by import) and `lastUpdate` (not refreshed in this path; reused as-is) | The wrapping `ProjectDiagram` object is preserved (title, description, references). Only mutation is `diagram.id = uuid()`. Round-trip is byte-clean for the `model` field itself. |
| **JSON (single → project import)** | same `useExportJSON` | `useImportDiagramToProject` (`useImportDiagram.ts:69-170`) | **NO** — `id`, `title` (echoed but new ref), `lastUpdate` (set to now), and `description` (defaulted if missing) all get rewritten. The `model` itself round-trips identically | Used by File → Import to current project. |
| **JSON (whole project)** | `exportProjectAsJson` → `buildProjectExportEnvelope` (`projectExportUtils.ts:99-109`) wraps in `{ project, exportedAt, version: '2.0.0' }` envelope and runs `diagramHasContent` filter | `projectImport.ts` (separate path; not in this audit set) | **NO** — empty diagrams are dropped (filter at `projectExportUtils.ts:30,60`), `exportedAt` always changes, and `name` is normalized via `normalizeProjectName`. Diagrams that pass the filter round-trip cleanly | This is what GitHub deploy / "Export project" shares. |
| **BUML (.py)** | `useExportBUML` POSTs `{title, model, generator: 'buml', referenceDiagramData?}` to `/export-buml` (`useExportBuml.ts:24-48`); backend dispatches on `model.type` and writes Python via `*_to_code()` builders | `useBumlToDiagram` POSTs the .py to `/get-json-model`; backend re-parses Python and returns JSON via the `buml_to_json` converters | **NO** — Python serialization drops layout (node positions, sizes, edge bend points, viewport / zoom), drops `references` (per-diagram cross-refs), drops `description`, drops free-form `interactive` / `assessments` fields, and may drop OCL parse-errors that don't survive the AST round-trip. Class names get safe-Python-identifier rewrites via `safe_var_name()`. The returned JSON does carry `version: "4.0.0"` and a synthetic `id` | Worst-fidelity supported format. Use only for code-handoff, not for archival. |
| **BUML (whole project)** | `useExportProjectBUML` → `/export-project-as-buml` | `/get-project-json-model` | **NO** — same losses as single-diagram BUML, plus inter-diagram references resolved by ID lookup and re-emitted as code references | Project-level loss aggregates per-diagram loss. |
| **SVG** | `useExportSVG` → `editor.exportAsSVG()` → `BesserEditor.exportModelAsSvg` (`besser-editor.tsx:259-374`) → `getSVG` (`exportUtils.ts:64-220`) → returns `{svg, clip}`; webapp blobs `besserSVG.svg` to `image/svg+xml` | **none** — there is no SVG → diagram importer | **N/A — one-way** | Exported SVG is not a round-trip artifact. See §B for fidelity issues vs. the canvas. |
| **PNG** | `useExportPNG` → calls `editor.exportAsSVG()` then `convertRenderedSVGToPNG` (`useExportPng.ts:25-73`); rasterizes via `Image` + `canvas` at `scale = 1.5` | **none** — there is no PNG → diagram importer | **N/A — one-way** | Inherits all SVG fidelity issues plus rasterization artefacts. |
| **Image (PNG/JPEG → diagram)** | n/a (no editor → image-as-input path) | `useImportDiagramPictureFromImage` POSTs the image to `/get-json-model-from-image` (`useImportDiagramPicture.ts:38-41`); backend uses OpenAI to emit a v4 ClassDiagram JSON; if there is an existing ClassDiagram with content, it is sent as `existing_model` so the backend merges instead of replacing | **N/A — one-way (LLM-based)** | Output shape is `data.title` / `data.model` (v4 with `nodes` / `edges`); `useImportDiagramPicture.ts:51` validates only `data.model.type` exists. Returned model is *not* validated against `isUMLModel`. |

---

## Section B — Per-step audit details

### 1. `BesserEditor.exportAsSVG()` — does it produce a standalone SVG?

**No, not in default `web` mode.** `getSVG` returns `mainSVG.outerHTML` (`exportUtils.ts:219`). Composition:

- The output **does** carry `xmlns="http://www.w3.org/2000/svg"` (`exportUtils.ts:81`), `viewBox`, `width`, `height`, `shape-rendering="geometricPrecision"`.
- The output **does not** prepend `<?xml version="1.0" encoding="UTF-8"?>` or `<!DOCTYPE svg ...>`. Browsers and most modern SVG consumers tolerate this when `xmlns` is present, but strict validators (xmllint default mode, some legacy tooling) will reject it.
- An inline `<style>` element is embedded once with the font-family stack only:
  ```css
  text { font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif; }
  ```
  No fonts are *embedded* — the recipient must have Inter (or fall through the stack to system fonts). PowerPoint on Windows that does not have Inter installed will substitute, which the team has already accepted via the fallback chain (per the `svgFontStyles` doc comment).
- Default `svgMode === "web"` (`exportUtils.ts:74`) **keeps `var(--besser-*)` CSS variables unresolved** in the output. Opening the SVG by itself in a browser will resolve to the root `--besser-*` variables on the host page if they exist, otherwise fall to the v4 `CSS_VARIABLE_FALLBACKS`. **This means an SVG exported in `web` mode is not standalone for CSS-variable-driven colours.** Only `svgMode === "compat"` resolves variables, inlines styles, drops `pointer-events`, replaces `currentColor`, and substitutes `text-decoration="underline"` with explicit `<line>` siblings (`exportUtils.ts:211-217, 1348-1388`).

**Webapp consumer (`useExportSVG`) does NOT pass `svgMode: 'compat'`** (`useExportSvg.ts:10`). Default `web` mode is what every SVG download from the editor produces. This is a usability regression unless the file is reopened in a similarly-themed browser context.

### 2. `useExportSvg.ts` packaging

The webapp wraps the SVG string in a `File` of type `image/svg+xml` and triggers download via `useFileDownload`. **No XML prolog is added at the webapp layer either.** The `<style>` block is inline (good), but as noted, CSS variable references are not resolved in default mode.

### 3. PNG export — does it capture non-visible parts?

**Yes, indirectly.** `BesserEditor.exportModelAsSvg` does **not** rasterize the user's live canvas. It mounts a *fresh* React tree into an off-DOM 4000×4000 container with a fresh `diagramStore` / `metadataStore` seeded from `model.nodes` / `model.edges` (`besser-editor.tsx:263-313`), waits for fonts and a 150 ms + double-rAF frame for layout (`besser-editor.tsx:336-349`), then computes bounds via `getRenderedDiagramBounds` over that off-DOM container. Therefore:

- The user's current zoom/pan is irrelevant.
- A full, scrollable diagram is rendered into 4000×4000 and clipped to its visible content + 60 px margin (`besser-editor.tsx:355-361`).
- Diagrams larger than the container would be truncated (4000 px effective limit). No guard, no warning.
- The PNG path takes that SVG, instantiates an `Image`, draws to a `canvas` scaled at `1.5×` (`useExportPng.ts:39-41,55-56`), and outputs PNG blob. Optional white background (`whiteBackground` flag) is the only setting; transparent is the default.
- `canvas.drawImage` with an `image/svg+xml` blob is subject to browser-side rasterization. CSS variables on the *embedding page* are NOT visible to the image decoder — only the inline `<style>` and any explicit attribute values are honoured. Combined with #1, **PNGs in `web` mode lose the user's theme and render with `CSS_VARIABLE_FALLBACKS`**.

### 4. JSON export wrapping

`useExportJSON` clones the active `ProjectDiagram` and overwrites `model` with `editor.model` (`useExportJson.ts:12`). The `editor.model` getter (`besser-editor.tsx:469-483`) emits:
```
{ id, version: "4.0.0", title, type, nodes, edges, assessments, interactive? }
```
Wrap is correct; this is the canonical v4 shape. JSON.stringify uses 2-space indent. No keys are dropped or renamed. **Round-trip clean, modulo `id` (rotated by importer).**

### 5. BUML export — does the v4 shape go through unmodified?

Backend reads `input_data.model` (a `dict`, not validated against `isUMLModel`) and dispatches on `model.type` (`conversion_router.py:297-298`). Each branch calls a `process_*_diagram(json_data)` parser (e.g. `process_class_diagram` is documented as *"Reads the v4 wire shape (`{nodes, edges}`) natively"* — `class_diagram_processor.py:1-6`) which constructs the BUML metamodel and dumps it via `*_to_code()`. The Python source returned is one-way; structure is preserved at the metamodel level (classes, attributes, methods, multiplicities, OCL, generalizations, associations) but layout/positioning is not. Confirmed by reading `class_diagram_processor.py:128-234` — `node_data`/`node_bounds` extract layout info but it is dropped on serialization to Python source.

The **`generator: 'buml'` field in the request body is ignored** by `/export-buml` — the endpoint always emits BUML Python (it lives at the BUML-specific endpoint). It's a harmless dead field per the JSON schema check; the field is kept in the frontend for symmetry with `/generate-output`.

### 6. JSON import — v3 vs. v4 detection

**`useImportDiagram.ts:33-43`:**
```
diagram = JSON.parse(fileContent);
diagram.id = uuid();
...
if (!isUMLModel(diagram.model)) {
  throw new Error('Invalid diagram: missing model or type information');
}
```
`isUMLModel` (`project.ts:738-750`) **only accepts v4** (it checks for `Array.isArray(nodes)` and `Array.isArray(edges)`). v3 models fail this guard.

There is a sibling helper `isV3UMLModel` (`project.ts:757-761`) that detects `elements`/`relationships`, and the project-storage migrator at `migrateProjectToV5` calls `migrateUMLModelV3ToV4` per-bucket (`project.ts:638-646`), but **`useImportDiagram` does not invoke `migrateUMLModelV3ToV4`**. The same gap exists in `useImportDiagramToProject` (`useImportDiagram.ts:91`). A single-file v3 JSON downloaded under the old editor will be rejected at import. (Non-UML models — `GrapesJSProjectData`, `QuantumCircuitData` — are also rejected here because `isUMLModel` is the gate, but those are routed through other importers in practice.)

### 7. Image import — v4 shape?

`useImportDiagramPicture.ts:51-53` checks only `data && data.model && data.model.type`. The backend `/get-json-model-from-image` is wired through OpenAI; the response shape is set by `services/reverse_engineering/` which ultimately uses the same `buml_to_json/class_diagram_converter.py` (which returns `version: "4.0.0"`). **No v4 regressions confirmed at the call boundary.** The image hook does NOT call `isUMLModel(data.model)`, so a malformed v3-shape model coming back from the backend would slip through and be saved into the project. Defensive validation here would mirror the JSON path.

### 8. Round-trip a real export — Identical model out?

| Format | Identical model out? | What changes |
|---|---|---|
| JSON single → JSON single | YES (model byte-identical; wrapper `id` rotates) | `id`, possibly `lastUpdate` |
| Project JSON envelope → Project import | YES per-diagram-with-content | Empty diagrams dropped, `name` normalized, `exportedAt` regenerated |
| BUML → re-imported via `/get-json-model` | NO | Layout (positions/sizes/edge bends), `interactive`, `assessments`, `references`, `description`, IDs (regenerated from class names), arbitrary node-level metadata (e.g. `data.color` if outside the BUML schema) |
| SVG / PNG | N/A | One-way render; no import path |
| Image-as-input | N/A | One-way LLM transform; output is an editor-friendly v4 ClassDiagram only |

---

## Section C — Top format degradations

1. **SVG default mode is not theme-portable.** `useExportSVG` calls `exportAsSVG()` with no options, so `svgMode` defaults to `"web"`, leaving `var(--besser-*)` references unresolved in the file. Re-opening the SVG in a different host (Inkscape, PowerPoint, raw browser tab without the besser CSS) renders with `CSS_VARIABLE_FALLBACKS` colours, not the user's selected theme. **Fix:** webapp should default to `svgMode: 'compat'` for downloads (or expose a toggle in the export dialog). Reference: `useExportSvg.ts:10`, `exportUtils.ts:74`.

2. **PNG inherits the same theme drift, plus rasterization artefacts.** `convertRenderedSVGToPNG` rasterizes a `web`-mode SVG blob through `Image`+`canvas`. CSS variables on the embedding page are NOT visible to the image decoder, so PNGs lose theme colours unless `compat` mode is also routed through. The 4000×4000 off-DOM render container is a *hard ceiling* — diagrams wider than 4000 px (large class diagrams, large NN architectures) get cropped silently. **Fix:** force `svgMode: 'compat'` before rasterizing; warn on bounds exceeding 4000 px.

3. **JSON import does not migrate v3 single-file diagrams.** `useImportDiagram` and `useImportDiagramToProject` both gate on `isUMLModel` (v4-only). Users who exported a single-diagram JSON under the old editor cannot re-import it without manually moving the data into a project's localStorage where `migrateProjectToV5` would catch it. **Fix:** call `isV3UMLModel(model)` after `JSON.parse`, run `migrateUMLModelV3ToV4(model, model.type)`, then re-validate with `isUMLModel`. Reference: `useImportDiagram.ts:33-42, 91-92`; `migrate-uml-v3-to-v4.ts`.

4. **BUML round-trip drops layout, `references`, `assessments`, `interactive`.** This is by design (Python source is structural, not visual), but it is not signposted in the export UI. A user exporting BUML for backup will lose all positioning. The toast says only "BUML export completed successfully" (`useExportBuml.ts:73`). **Fix:** annotate the "Export BUML" button with a lossy-export warning, or store layout in a sidecar `.json.layout` file. Cross-diagram `references` (per-diagram cross-refs by ID) are also dropped because BUML emits global references resolved from the class graph.

5. **Image import skips schema validation.** `useImportDiagramPictureFromImage` only checks `data.model.type` is present (`useImportDiagramPicture.ts:51-53`). It does not run `isUMLModel(data.model)`. If the OpenAI prompt drifts or the backend changes the shape, the importer happily saves a malformed model into the project's localStorage. The next read will fail in the editor with a less actionable error. **Fix:** add `if (!isUMLModel(data.model)) throw new Error(...)` to the image import path, mirroring `useImportDiagram`.

### Secondary observations (not in the top 5)

- SVG output lacks an XML prolog (`<?xml ... ?>`) and DOCTYPE. Most consumers tolerate this, but strict validators do not. `mainSVG.outerHTML` (`exportUtils.ts:219`) is a single string concatenation — adding `'<?xml version="1.0" encoding="UTF-8"?>\n'` upfront is a one-line change.
- BUML export sends a `generator: 'buml'` field (`useExportBuml.ts:34`) that the backend ignores. Harmless but confusing in network logs; consider dropping or having the backend echo back which generator ran.
- The PNG file `Content-Disposition` is set client-side with the diagram title, but the SVG path uses the same. If the title contains characters like `/`, the download filename is OS-dependent. Sanitize via `normalizeProjectName` (already used by the project-JSON path).
- The off-DOM render in `exportModelAsSvg` waits for `document.fonts.ready` (`besser-editor.tsx:336-338`), which is best-effort. On browsers without `document.fonts`, fall-through is an immediate render — text wrapping decisions can drift from the user's on-screen render. Acknowledged in the code comment but worth keeping in mind.

---

## Notes on what was *not* changed

This audit is read-only per the task contract. No source files were modified.
