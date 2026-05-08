# A10 — Backend deploy + GUI generator (v4 + GrapesJS)

Wave: final-analysis. Branch: `claude/refine-local-plan-sS9Zv`.

Scope: confirm `/besser_api/github/deploy-webapp` (post-`fa4e4d1`) and
`/besser_api/generate-output-from-project` (WebApp generator, GUINoCodeDiagram)
work end-to-end with v4 class diagrams + GrapesJS GUI diagrams.

> Naming clarification: the local `/besser_api/deploy-app` route in
> `routers/deployment_router.py` is **Django-only Docker Compose**, not the
> GitHub deploy. Public deploy from the editor goes through
> `services/deployment/github_deploy_api.py::deploy_webapp_to_github`
> (route `POST /besser_api/github/deploy-webapp`). The fa4e4d1 fix targets
> that file. Both routes are confirmed below.

## 1. Code paths inspected

| File | Lines | Purpose |
| --- | --- | --- |
| `besser/utilities/web_modeling_editor/backend/services/deployment/github_deploy_api.py` | 112-587 (deploy entry), 199 + 970 (v4 guards from fa4e4d1) | GitHub deploy: classifies target → builds tempdir → calls `WebAppGenerator` → pushes to GitHub |
| `besser/utilities/web_modeling_editor/backend/routers/generation_router.py` | 402-602 + 1026-1038 | `/generate-output-from-project` → `_handle_web_app_project_generation` → `_generate_web_app` |
| `besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/gui_processors/processor.py` | 57-651 | GrapesJS pages/frames → `GUIModel` (intentionally v3 GrapesJS shape, not Apollon `nodes`/`edges`) |
| `besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/gui_processors/utils.py` | 132-163 | `get_element_by_id`: v4-aware lookup over class diagram nodes (fa4e4d1) |
| `besser/utilities/web_modeling_editor/backend/models/project.py` | 5-77 | `ProjectInput` + `get_active_diagram` / `get_referenced_diagram` |
| `besser/generators/web_app/web_app_generator.py` | 30-172 | Composes ReactGenerator (frontend) + BackendGenerator (FastAPI backend) + Docker files + optional BAFGenerator agents |
| `besser/utilities/web_modeling_editor/backend/routers/deployment_router.py` | 70-134 | Local `/deploy-app` (Django-only, separate concern) |

## 2. Deploy path trace — `/besser_api/github/deploy-webapp` (webapp branch)

Fixture: `tests/fixtures/v4/deploy_project.json` (newly added, mirrors the
request body the frontend sends — `diagrams`, `currentDiagramIndices`,
`deploy_config`, `settings`).

Trace (in-process, stops just before any GitHub HTTP call):

```
[trace] active ClassDiagram id: cd-1
[trace] active GUINoCodeDiagram id: gui-1
[trace] gui references: {'ClassDiagram': 'cd-1'}
[trace] cd model has nodes: True
[trace] domain classes: ['Book', 'Author']
[trace] gui modules: 1 screens: ['body']
[trace] generator produced backend/: True frontend/: True docker-compose: True
[trace] total files: 36
[trace] deploy path would now call: github.create_repository / push_directory_to_repo
[trace] PASS
```

Step-by-step (lines refer to `github_deploy_api.py`):

1. `body.get('diagrams')`, `body.get('currentDiagramIndices')` extracted at
   170-171.
2. `_get_active('ClassDiagram')` (174-179) returns `diagrams['ClassDiagram'][0]`
   — `cd-1`, since `currentDiagramIndices['ClassDiagram'] = 0`.
3. `_get_active('GUINoCodeDiagram')` returns `gui-1`.
4. Webapp branch (373-394): non-empty guards `class_diagram_data.get('model')`
   and `gui_diagram_data.get('model')` pass. **No v3-keyed guard remains on
   this branch** — the agent branch's `model.get('nodes')` guard at line 199
   is the only one of its kind, and it's correct.
5. `process_class_diagram(class_diagram_data)` parses `model.nodes` /
   `model.edges` → `DomainModel{Book, Author}` (the processor is v4-native;
   see `class_diagram_processor.py:128` taking `nodes: list[dict]`).
6. `process_gui_diagram(gui.model, cd.model, buml_model)` walks
   `pages -> frames -> component (wrapper) -> components` (GrapesJS shape) →
   `GUIModel` with one `Module` containing one `Screen`. When a button binds
   `entity-class-id: n-book`, `get_element_by_id(class_model, 'n-book')` walks
   `class_model.get('nodes')` (utils.py:151) and resolves the v4 node — this
   is the fa4e4d1 fix.
7. Empty `AgentDiagram` list → `agent_models=[], agent_configs={}`.
8. `WebAppGenerator(buml, gui, output_dir=tmp, agent_models=[], agent_configs={}).generate()`
   emits:
   - `backend/main_api.py`, `backend/pydantic_classes.py`, `backend/Dockerfile`
     (FastAPI; **not Django** — task description was inaccurate on this point).
   - `frontend/` (22 React/TS/HTML/CSS sources, ReactGenerator-generated).
   - `docker-compose.yml` at root.
9. `_add_deployment_configs` writes `render.yaml` + env files.
10. `domain_model_to_code` / `gui_model_to_code` write `buml/*.py` round-trip
    artefacts.
11. **Stop point reached** (would now call `github.create_repository` →
    `update_repository` → `push_directory_to_repo`). All inputs are v4-clean.

Status: **WORKS** for the webapp branch.

Agent branch (lines 190-371): the v4 guard `model.get('nodes')` at
github_deploy_api.py:199 correctly rejects empty agent diagrams. The B-UML
export loop at 970 uses `model.get('nodes')` likewise. Both confirmed by
re-reading the post-fa4e4d1 source.

## 3. `/generate-output-from-project` parity

Same fixture (loaded as `ProjectInput`). The smoke script
`tests/utilities/web_modeling_editor/backend/smoke_webapp_generation.py`
exercises this path via `httpx.ASGITransport`:

```
[smoke] HTTP status=200 content-type='application/zip' body_bytes=51068
[smoke] PASS
```

`_handle_web_app_project_generation` (generation_router.py:546-602) does the
right thing:

1. `project.get_active_diagram('GUINoCodeDiagram')` — uses
   `currentDiagramIndices` fallback (project.py:36-43).
2. `project.get_referenced_diagram(gui, 'ClassDiagram')` — uses the
   per-diagram `references["ClassDiagram"]: "cd-1"` (project.py:45-77).
   String IDs are stable across reorder/delete; integer fallback also
   handled. **No global `currentDiagramIndices` is needed for the link** —
   the per-diagram reference wins (correct invariant).
3. `process_class_diagram(class_diagram.model_dump())` then
   `process_gui_diagram(gui.model, class.model, buml_model)` — same
   processors as the deploy path.
4. `_generate_web_app(buml, gui, generator_class=WebAppGenerator, ...)`
   awaits in a thread, ZIPs `temp_dir`, streams it back.

Status: **WORKS**.

## 4. Active-ClassDiagram resolution

| Mechanism | Where | Behavior on the fixture | Verdict |
| --- | --- | --- | --- |
| Per-diagram `gui.references["ClassDiagram"] = "cd-1"` | `ProjectInput.get_referenced_diagram` | Resolves to `cd-1` directly by ID | works |
| Global `currentDiagramIndices["ClassDiagram"] = 0` | Same method, fallback | Used only if `references` is missing or unresolvable | works |
| Deploy handler `_get_active("ClassDiagram")` | `github_deploy_api.py:173-179` | Uses `currentDiagramIndices` (deploy body has no per-diagram references field exposed at this layer) | works, **but see Issue D-1** |

The deploy body's `_get_active` ignores `from_diagram.references`, but
that's acceptable for deploy — there's only one ClassDiagram in real
deploy bodies. If multi-ClassDiagram projects ship, this becomes inferior
to the project-router path.

## 5. WebApp generator output validity

End-to-end smoke (`tests/fixtures/v4/webapp_smoke_project.json`):

- 36 files, 138 KiB.
- `backend/`: 3 Python files, all parse cleanly via `ast.parse` (FastAPI
  app + Pydantic models + Dockerfile).
- `frontend/`: 22 non-empty TS/TSX/HTML/CSS files (React app generated by
  `ReactGenerator`).
- `docker-compose.yml` at root (Postgres + backend + frontend services).
- `buml/domain_model.py`, `buml/gui_model.py` round-trip artefacts emitted
  by the deploy path (not part of the smoke ZIP, but exercised by the
  trace script above).

Status: **VALID**.

## 6. Issues

| ID | Severity | File / Site | Description | Recommendation |
| --- | --- | --- | --- | --- |
| D-1 | low | `github_deploy_api.py:173-179` (`_get_active`) | Helper duplicates `ProjectInput.get_active_diagram` and ignores per-diagram `references`. Currently fine because deploy bodies carry one ClassDiagram in practice; would silently pick the wrong class diagram if a future deploy body carries multiple. | Reuse `ProjectInput` (parse `body` into a `ProjectInput` first) so deploy and generate use the same active/referenced resolution. Also gives Pydantic validation for free. |
| D-2 | informational | `routers/deployment_router.py:70-134` (`/deploy-app`) | Django-only Docker Compose deploy. Not part of the GitHub deploy path. Naming collides with the public "Deploy" UX in the editor — easy to confuse during reviews (this task initially conflated them). | Rename to `/deploy-django` or fold into `/github/deploy-webapp` with a `target=django` knob. No correctness impact today. |
| D-3 | informational | `besser/generators/web_app/web_app_generator.py:101` (`BackendGenerator`) | Task description called it "Django from class". The WebApp generator emits **FastAPI** (`backend/main_api.py`) plus React. Keep the doc terminology aligned with reality so future deploy tickets don't expect a Django bundle here. | Update task templates / docstrings if any still say "Django". |
| D-4 | low | `services/converters/json_to_buml/gui_processors/utils.py:163` | `get_element_by_id` returns `None` silently when a button binds to a missing class id. Currently logged nowhere. With v4, the id contract changed (UUIDs from frontend), so a typo silently disables the button's CRUD wiring. | Add a one-time `logger.warning` when an `entity-class-id` doesn't resolve. |

No **critical** issues found. The fa4e4d1 fix closes the previously-reported
v3 leakage in the deploy guards; the webapp branch never had a v3-keyed guard
so it kept working through the cutover.

## 7. Verification commands (reproducible)

```bash
# E2E WebApp generation (project endpoint + in-process)
python tests/utilities/web_modeling_editor/backend/smoke_webapp_generation.py

# Deploy path trace (stops at github.create_repository)
python -c "
import json, tempfile, os
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram, process_gui_diagram,
)
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
body = json.load(open('tests/fixtures/v4/deploy_project.json'))
diagrams = body['diagrams']; current = body['currentDiagramIndices']
def active(t):
    v = diagrams.get(t); idx = current.get(t, 0)
    return v[idx] if isinstance(v, list) else (v or {})
cd = active('ClassDiagram'); gui = active('GUINoCodeDiagram')
buml = process_class_diagram(cd)
gui_m = process_gui_diagram(gui['model'], cd['model'], buml)
g = get_generator_info('web_app').generator_class(buml, gui_m, output_dir=tempfile.mkdtemp())
g.generate()
print('OK')
"
```

## 8. Verdict

- `/besser_api/github/deploy-webapp` (webapp branch): **green** on v4.
- `/besser_api/github/deploy-webapp` (agent branch): **green** on v4 (covered
  by fa4e4d1; `model.get('nodes')` guards in place at lines 199 and 970).
- `/besser_api/generate-output-from-project` (`generator=web_app`): **green**
  on v4 — both in-process and HTTP smoke pass.
- Active ClassDiagram resolution: **correct** via
  `ProjectInput.get_referenced_diagram` (per-diagram `references` first,
  global `currentDiagramIndices` as fallback).
- WebApp bundle: backend (FastAPI) + frontend (React) + docker-compose.yml,
  all valid; Python files `ast.parse`-clean; React/TS files non-empty.
