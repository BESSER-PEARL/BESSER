# Backend Generator Refactoring Plan

claude --resume "backend-generator-refactoring"
claude --resume "backend-generator-refactoring"

## Goal

Refactor the backend generator to produce a **layered FastAPI project structure** instead of 3 flat files.

### Current Output

```
backend/
├── main_api.py          # Everything: routes, middleware, DB setup, BAL helpers
├── sql_alchemy.py       # All ORM models in one file
├── pydantic_classes.py  # All schemas in one file
└── requirements.txt
```

### Target Output

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # App factory, middleware, startup
│   ├── database.py           # Engine, session, Base
│   ├── config.py             # Settings, env vars
│   ├── models/               # SQLAlchemy models (1 per entity)
│   │   ├── __init__.py
│   │   └── {entity}.py
│   ├── schemas/              # Pydantic schemas (1 per entity)
│   │   ├── __init__.py
│   │   └── {entity}.py
│   ├── routers/              # Route handlers (1 per entity)
│   │   ├── __init__.py
│   │   └── {entity}.py
│   └── crud/                 # DB operations (1 per entity)
│       ├── __init__.py
│       └── {entity}.py
├── requirements.txt
└── Dockerfile
```

---

## What Needs to Change

### Must Change (Core)

| # | File | What Changes |
|---|------|-------------|
| 1 | `rest_api/templates/backend_fast_api_template.py.j2` | **Heaviest lift.** Split this single massive template into: `main.py.j2`, `database.py.j2`, `config.py.j2`, `router.py.j2` (per-class), `crud.py.j2` (per-class). Currently contains routes, middleware, exception handling, BAL helpers, and DB setup all in one. |
| 2 | `sql_alchemy/templates/sql_alchemy_template.py.j2` | Split into per-class model templates. Shared pieces (Base, enums, association tables) go into a base template. |
| 3 | `pydantic_classes/templates/pydantic_classes_template.py.j2` | Split into per-class schema templates. Shared pieces (enums, base config) go into a base template. |
| 4 | `rest_api/rest_api_generator.py` | `generate()` must create directory structure and render templates per-class in a loop, instead of rendering once into a single file. |
| 5 | `sql_alchemy/sql_alchemy_generator.py` | Same: loop over classes, write individual files into `models/`. |
| 6 | `pydantic_classes/pydantic_classes_generator.py` | Same: loop over classes, write individual files into `schemas/`. |
| 7 | `backend/docker_files.py` | Dockerfile `COPY` commands currently reference 3 flat files by name. Must copy the new `app/` directory structure instead. |

### Likely Needs Updating

| # | File | What Changes |
|---|------|-------------|
| 8 | `backend/backend_generator.py` | Orchestrator may need to coordinate shared output (e.g. `database.py` generated once, not by each sub-generator). |
| 9 | `web_app/templates/backend.Dockerfile.j2` | Same Docker issue — used by WebAppGenerator. |
| 10 | `web_app/templates/docker-compose.yml.j2` | Startup command changes (e.g. `uvicorn app.main:app` instead of `python main_api.py`). |

### Tests

| # | File | What Changes |
|---|------|-------------|
| 11 | `tests/generators/` | Any tests asserting on generated file names or content will break and need updating. |

---

## What Does NOT Change

- **Metamodel** (`besser/BUML/metamodel/`) — untouched
- **Frontend** (React generator) — untouched
- **Converters** (json_to_buml, buml_to_json) — untouched
- **Non-backend REST API template** (`fast_api_template.py.j2`) — standalone mode stays as-is
- **Web modeling editor backend** — that's the BESSER editor itself, not generated code
- **Generation logic** — the metamodel traversal and data preparation stays the same

---

## Key Decisions to Make

1. **Do sub-generators produce per-class files directly, or does BackendGenerator orchestrate the split?**
   - Option A: Each sub-generator (Pydantic, SQLAlchemy, REST API) learns to output per-class files when `backend=True`
   - Option B: BackendGenerator calls sub-generators as-is, then post-processes/splits the output
   - **Recommendation:** Option A — cleaner, each generator owns its output format

2. **How to handle cross-class dependencies in per-class files?**
   - Association tables reference multiple classes
   - Inheritance requires parent models to be importable
   - Enums may be shared across classes
   - **Recommendation:** Shared `models/__init__.py` re-exports all models, `models/base.py` holds Base + enums + association tables

3. **Should the non-backend mode (`backend=False`) also get the new structure?**
   - Currently `backend=False` generates a simple `rest_api.py` for quick prototyping
   - **Recommendation:** Leave standalone mode as-is — it serves a different use case (quick scripts)

4. **How to handle BAL (BESSER Action Language) helpers?**
   - Currently inlined in `main_api.py`
   - **Recommendation:** Move to `app/bal.py` or `app/utils/bal.py`

---

## Template Splitting Strategy

### From `backend_fast_api_template.py.j2` (the big one)

| New Template | Contains |
|-------------|----------|
| `main.py.j2` | FastAPI app creation, CORS middleware, logging middleware, exception handlers, startup/shutdown, `uvicorn.run()` |
| `database.py.j2` | Engine creation, `SessionLocal`, `get_db()` dependency, `Base` class, `init_db()` |
| `config.py.j2` | Settings class (DATABASE_URL, port, CORS origins) |
| `router.py.j2` | Per-entity: all CRUD endpoints (GET, POST, PUT, DELETE), search, pagination, relationship endpoints, method endpoints |
| `crud.py.j2` | Per-entity: DB query functions (create, read, update, delete, bulk ops, relationship management) |
| `bal.py.j2` | BAL standard library functions (`BAL_size`, `BAL_is_empty`, `BAL_add`, etc.) |

### From `sql_alchemy_template.py.j2`

| New Template | Contains |
|-------------|----------|
| `model_base.py.j2` | `Base` class, enum definitions, association tables |
| `model.py.j2` | Per-entity: ORM class with mapped columns, relationships, FK constraints |

### From `pydantic_classes_template.py.j2`

| New Template | Contains |
|-------------|----------|
| `schema_base.py.j2` | Shared enums, base config |
| `schema.py.j2` | Per-entity: Pydantic model + Create variant, field validators |

---

## Customizable API Metadata (Issue #476)

As part of this refactoring, we will also address [#476](https://github.com/BESSER-PEARL/BESSER/issues/476) — the FastAPI app metadata (title, description, version) is currently hardcoded in the template.

### Current State

In `backend_fast_api_template.py.j2`:
```python
app = FastAPI(
    title="{{name}} API",
    description="Auto-generated REST API with full CRUD operations...",  # hardcoded
    version="1.0.0",                                                     # hardcoded
)
```

The root endpoint also duplicates the hardcoded `version`:
```python
@app.get("/")
def root():
    return {"name": "{{name}} API", "version": "1.0.0", ...}
```

### Solution

Add optional keyword arguments to `RESTAPIGenerator` and `BackendGenerator` with sensible defaults:

```python
RESTAPIGenerator(
    model=my_model,
    api_title="My Custom API",          # default: "{model.name} API"
    api_description="My service",        # default: "Auto-generated REST API..."
    api_version="2.1.0",                 # default: "1.0.0"
)
```

In the refactored structure, these values will live in the generated `config.py`, making them easy to find and modify post-generation as well.

### Files Affected

| File | Change |
|------|--------|
| `rest_api/rest_api_generator.py` | Add `api_title`, `api_description`, `api_version` params, pass to template context |
| `backend/backend_generator.py` | Accept and forward the same 3 params to `RESTAPIGenerator` |
| `backend_fast_api_template.py.j2` (or new `main.py.j2` + `config.py.j2`) | Use template variables instead of hardcoded strings |
| `fast_api_template.py.j2` | Same for non-backend mode |

---

## Estimated Scope

- **Templates:** ~6 new templates replacing 3 existing ones
- **Generator code:** 3 generators need loop-based rendering + API metadata params
- **Docker:** 2-3 Dockerfile/compose templates updated
- **Tests:** Existing backend generator tests need updating
- **No metamodel changes, no converter changes, no frontend changes**
