# Backend Smoke: WebApp Generation E2E (post SA-6.2)

Reference: parent at `1a7be4c`, library (frontend submodule) at `503660a`.
Branch: `claude/refine-local-plan-sS9Zv`.

## Pytest status

```
$ python -m pytest tests/utilities/web_modeling_editor/
407 passed, 5 skipped in 3.25s
```

No failures. Skipped tests are the pre-existing 5 skips, unrelated to v4 work.

## Smoke script results

- **Smoke script**: `tests/utilities/web_modeling_editor/backend/smoke_webapp_generation.py`
- **Fixture**: `tests/fixtures/v4/webapp_smoke_project.json` (v4 ProjectInput with 1 ClassDiagram = `Book`/`Author` + 1..* `wrote` association, 1 GUINoCodeDiagram = `Home` page with heading + Create-Book button bound to `n-book`).
- **Run command**: `python tests/utilities/web_modeling_editor/backend/smoke_webapp_generation.py`
- **Result**: **PASS** (both direct invocation and HTTP path).

### Direct path (process_class_diagram + process_gui_diagram + WebAppGenerator)

| Metric | Value |
|---|---|
| Output dir size | 36 files / 138.0 KiB |
| Backend python files parsing `ast.parse` | 3 / 3 |
| Frontend web files (.tsx/.jsx/.ts/.js/.html) non-empty | 22 / 22 |
| `docker-compose.yml` at root | present |
| ClassDiagram → DomainModel | `['Book', 'Author']` |
| GUINoCodeDiagram → GUIModel | `modules=1, screens=1, screen_names=['body']` |

### HTTP path (mimics `useGenerateCode` → `POST /besser_api/generate-output-from-project`)

The script also exercises the FastAPI app via `httpx.AsyncClient` + `ASGITransport`,
posting the same v4 ProjectInput JSON the way `buildProjectPayloadForBackend` ships it:

```
POST /besser_api/generate-output-from-project
HTTP status=200, content-type='application/zip', body_bytes=51068
```

The router successfully hit `_handle_web_app_project_generation`, resolved the
referenced ClassDiagram via per-diagram `references`, processed both diagrams
through the v4 converters, and streamed back a real ZIP.

### Failure root cause (if any)

None. Generator and HTTP path both succeed end-to-end on a clean v4 input.

## v3 leaks surfaced

None. SA-6.1's `gui_processors/utils.py:get_element_by_id` already accepts the
v4 `nodes` shape; the rest of the GUI processor never touches the wire shape
(it operates on the Python `DomainModel` and the GrapesJS `pages/frames`
component tree, neither of which changed in v4).

The `class_model` JSON the GUI processor receives is `class_diagram.model`
(the v4 dict with `nodes/edges`) — `get_element_by_id` correctly walks `nodes`
to resolve `entity-class-id="n-book"`.

## Risks for SA-7

1. **Frontend GUI editor still emits GrapesJS-shaped pages, not React-Flow nodes/edges.**
   `process_gui_diagram` consumes GrapesJS (`pages[].frames[].component.components[]`).
   If SA-7 plans to migrate the GUI editor to React-Flow, the entire
   `gui_processors/` package needs a v4-style rewrite — the smoke confirms the
   current path works, but it does not de-risk a GUI-side wire-shape change.

2. **GUI screen naming falls back to wrapper `tagName`.**
   The smoke run produced `screen_names=['body']` because the fixture's wrapper
   has no explicit name/id. Real frontend exports usually carry a page name; if
   SA-7 changes how page metadata is shaped this fallback may surprise users.
   Watch for screens showing up named `body`, `wrapper`, etc.

3. **Generator stdout pollution.**
   `BackendGenerator` prints `"No configuration docker file." / "Code generated…"`
   to stdout from inside the request handler. Harmless for smoke, but if SA-7
   adds structured logging or runs gunicorn with JSON logs, these prints will
   sit outside the request-id middleware. Worth routing through the
   `request_logging` middleware logger.

4. **Active-diagram resolution is reference-driven, not by `currentDiagramIndices`.**
   `_handle_web_app_project_generation` resolves `ClassDiagram` via the GUI's
   per-diagram `references["ClassDiagram"]`, not the global active index. The
   smoke fixture sets both. Frontend code that ships ProjectInput without
   per-diagram `references` after SA-7 would fall back to the global active
   index — verify `useGenerateCode`'s payload always sets `references`.

## Smoke script source

See `tests/utilities/web_modeling_editor/backend/smoke_webapp_generation.py` —
the script:

1. Loads `tests/fixtures/v4/webapp_smoke_project.json` and parses it through
   `ProjectInput` (Pydantic).
2. Mirrors `_handle_web_app_project_generation`: resolves the active GUI
   diagram, finds the referenced ClassDiagram, calls `process_class_diagram`
   then `process_gui_diagram`, then `WebAppGenerator(...).generate()`.
3. Asserts directory is non-empty, backend `.py` files parse via `ast.parse`,
   frontend web files are non-empty, `docker-compose.yml` exists.
4. Posts the raw fixture JSON to `/besser_api/generate-output-from-project`
   via `httpx.AsyncClient + ASGITransport` and asserts a 200 ZIP response.
