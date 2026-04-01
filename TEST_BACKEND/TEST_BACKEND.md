# Backend Generator Refactoring — Summary of Changes

## Overview

Refactored the backend generator from producing **3 flat files** (`main_api.py`, `sql_alchemy.py`, `pydantic_classes.py`) into a **layered FastAPI project structure** with per-entity files. Also addressed GitHub issues [#476](https://github.com/BESSER-PEARL/BESSER/issues/476) (configurable API metadata) and [#477](https://github.com/BESSER-PEARL/BESSER/issues/477) (extension documentation).

### Before

```
output_backend/
├── main_api.py          # Everything in one file
├── sql_alchemy.py       # All ORM models in one file
├── pydantic_classes.py  # All schemas in one file
└── requirements.txt
```

### After

```
output_backend/
├── requirements.txt
├── .env.example
└── app/
    ├── __init__.py
    ├── main.py           # FastAPI app, middleware, system endpoints
    ├── config.py          # API_TITLE, API_DESCRIPTION, API_VERSION, DATABASE_URL, PORT
    ├── database.py        # SQLAlchemy engine, session, get_db()
    ├── bal.py             # BESSER Action Language helpers
    ├── models/            # SQLAlchemy ORM models (per-entity)
    │   ├── __init__.py    # Association tables, imports, relationship attachments
    │   ├── _base.py       # Base(DeclarativeBase)
    │   ├── _enums.py      # All enum definitions
    │   └── {entity}.py    # One file per entity
    ├── schemas/           # Pydantic validation schemas (per-entity)
    │   ├── __init__.py    # Re-exports, model_rebuild()
    │   └── {entity}.py    # One file per entity
    └── routers/           # FastAPI routers (per-entity)
        ├── __init__.py    # Collects all routers into all_routers list
        └── {entity}.py    # CRUD endpoints per entity
```

---

## Architecture: Sub-Generator Orchestration

The key design decision: **BackendGenerator orchestrates the 3 sub-generators** rather than bypassing them. Each sub-generator has two modes:

```
SQLAlchemyGenerator
├── .generate()           → flat file:    sql_alchemy.py     (standalone use)
└── .generate_layered()   → per-entity:   app/models/         (called by BackendGenerator)

PydanticGenerator
├── .generate()           → flat file:    pydantic_classes.py (standalone use)
└── .generate_layered()   → per-entity:   app/schemas/        (called by BackendGenerator)

RESTAPIGenerator
├── .generate()           → flat file:    rest_api.py         (standalone use)
└── .generate_layered()   → per-entity:   app/routers/        (called by BackendGenerator)
```

**BackendGenerator.generate()** calls all three `.generate_layered()` methods, then renders its own glue files (main.py, config.py, database.py, bal.py, requirements.txt, .env.example).

### Shared Templates (No Duplication)

Each sub-generator's templates are shared between flat and layered modes via Jinja2 macro files:

| Generator | Shared macros file | What it contains |
|-----------|-------------------|-----------------|
| SQLAlchemy | `helpers.py.j2` | `render_attributes()`, `render_relationship()` — type mapping, FK columns, nullable |
| Pydantic | `pydantic_helpers.py.j2` | `render_class_body()` — attributes, relationships, OCL validators |
| REST API | *(not shared)* | Flat template removed; only layered templates remain |

Fix a bug in `helpers.py.j2` → fixed for both standalone `sql_alchemy.py` AND layered `app/models/{entity}.py`.

---

## Files Changed

### New Files Created

| File | Purpose |
|------|---------|
| `besser/generators/backend/templates/app_init.py.j2` | `app/__init__.py` template |
| `besser/generators/backend/templates/config.py.j2` | `app/config.py` template (API metadata, DB URL, port) |
| `besser/generators/backend/templates/database.py.j2` | `app/database.py` template (engine, session, get_db) |
| `besser/generators/backend/templates/bal.py.j2` | `app/bal.py` template (12 BAL async helpers) |
| `besser/generators/backend/templates/main.py.j2` | `app/main.py` template (FastAPI app, middleware, endpoints) |
| `besser/generators/backend/templates/requirements.txt.j2` | `requirements.txt` template |
| `besser/generators/backend/templates/env.example.j2` | `.env.example` template |
| `besser/generators/sql_alchemy/templates/models_base.py.j2` | `app/models/_base.py` template |
| `besser/generators/sql_alchemy/templates/models_enums.py.j2` | `app/models/_enums.py` template |
| `besser/generators/sql_alchemy/templates/model_entity.py.j2` | `app/models/{entity}.py` template |
| `besser/generators/sql_alchemy/templates/model_asso_entity.py.j2` | `app/models/{asso_class}.py` template |
| `besser/generators/sql_alchemy/templates/models_init.py.j2` | `app/models/__init__.py` template |
| `besser/generators/pydantic_classes/templates/pydantic_helpers.py.j2` | Shared macro: class body rendering |
| `besser/generators/pydantic_classes/templates/schema_entity.py.j2` | `app/schemas/{entity}.py` template |
| `besser/generators/pydantic_classes/templates/schemas_init.py.j2` | `app/schemas/__init__.py` template |
| `besser/generators/rest_api/templates/router_entity.py.j2` | `app/routers/{entity}.py` template |
| `besser/generators/rest_api/templates/routers_init.py.j2` | `app/routers/__init__.py` template |

### Files Modified

| File | Change |
|------|--------|
| `besser/generators/structural_utils.py` | Added `used_enums_for_class()` shared helper |
| `besser/generators/backend/backend_generator.py` | Rewritten to orchestrate sub-generators + render glue files. Added `api_title`, `api_description`, `api_version` params (#476) |
| `besser/generators/sql_alchemy/sql_alchemy_generator.py` | Added `generate_layered(app_dir)`, `_create_env()`, `_build_layered_context()` |
| `besser/generators/pydantic_classes/pydantic_classes_generator.py` | Added `generate_layered(app_dir)`, `_create_env()`. Updated flat template to use shared `pydantic_helpers.py.j2` |
| `besser/generators/pydantic_classes/templates/pydantic_classes_template.py.j2` | Refactored to import and use `pydantic_helpers.py.j2` macros instead of inline logic |
| `besser/generators/rest_api/rest_api_generator.py` | Added `generate_layered(app_dir)`, `_create_layered_env()`. Removed dead `backend` and `port` params. Removed `if self.backend:` branch |
| `besser/generators/backend/docker_files.py` | Updated `COPY app/ ./app/` and `CMD uvicorn app.main:app` |
| `besser/generators/web_app/templates/backend.Dockerfile.j2` | Updated `COPY app/ ./app/` and `CMD uvicorn app.main:app` |
| `besser/utilities/web_modeling_editor/backend/services/deployment/github_service.py` | `uvicorn main_api:app` → `uvicorn app.main:app` |
| `besser/utilities/web_modeling_editor/backend/services/deployment/github_deploy_api.py` | `uvicorn main_api:app` → `uvicorn app.main:app` |
| `tests/generators/backend/test_backend.py` | Updated all assertions for layered file structure |
| `tests/generators/backend/test_backend_full_uml.py` | Updated all assertions for layered file structure |

### Files Deleted

| File | Reason |
|------|--------|
| `besser/generators/rest_api/templates/backend_fast_api_template.py.j2` | Dead code — 1281-line flat backend template, replaced by `router_entity.py.j2` |

### Documentation Updated

| File | Change |
|------|--------|
| `docs/source/generators/backend.rst` | New layered structure diagram, `.env.example`, **extension guide** (custom routes, auth, middleware, business logic, Docker) |
| `docs/source/generators/rest_api.rst` | Removed `backend` and `port` params, simplified output description |
| `docs/source/generators/alchemy.rst` | Added "Layered Output (Backend Integration)" section |
| `docs/source/generators/pydantic.rst` | Added "Layered Output (Backend Integration)" section |
| `docs/source/generators/full_web_app.rst` | Updated directory tree and startup command |
| `docs/source/examples/backend_example.rst` | Rewritten with per-entity file examples |
| `docs/source/examples/terraform_example.rst` | Updated output description |

---

## Bug Fixes (in templates)

| Bug | Fix | File |
|-----|-----|------|
| `polymorphic_on` re-declared on mid-hierarchy classes (e.g. Student extends Person, also has GraduateStudent) | Added check: only root class (no non-concrete parents) gets `polymorphic_on` | `sql_alchemy/templates/model_entity.py.j2` |
| Self-association M:N `insert().values()` had duplicate keyword `student_id=..., student_id=...` | Use actual column names (`mentors=..., mentees=...`) from association end names | `rest_api/templates/router_entity.py.j2` |
| Schema imports concatenated on one line (`from app.schemas.person import PersonCreatefrom app.models._enums import AcademicRank`) | Replaced `{%-` with `{%` for parent/enum import loops in schema template | `pydantic_classes/templates/schema_entity.py.j2` |
| `EnrollmentCreate` (AssociationClass schema) missing from `schemas/__init__.py` | Added `asso_classes` to template context and init template | `pydantic_classes/pydantic_classes_generator.py` + `schemas_init.py.j2` |
| Generated `main.py` couldn't run from outside `backend/` directory (`ModuleNotFoundError: No module named 'app'`) | Added `sys.path.insert()` at top of `main.py` to resolve `app` package from any working directory | `backend/templates/main.py.j2` |

---

## Issue #476 — Configurable API Metadata

| Requirement | Status |
|-------------|--------|
| Layered FastAPI project structure | Done |
| `api_title`, `api_description`, `api_version` params on BackendGenerator | Done |
| Generated `config.py` with API metadata + `DATABASE_URL` + `PORT` | Done |
| `.env.example` for easy configuration | Done |
| Values configurable via environment variables (`os.getenv()`) | Done |

## Issue #477 — Extending Generated Code

| Requirement | Status |
|-------------|--------|
| Structure makes extension natural (separate files per entity) | Done |
| Doc: adding custom routes | Done |
| Doc: adding authentication / JWT middleware | Done |
| Doc: adding custom business logic (services layer) | Done |
| Doc: Docker deployment with `.env` | Done |
| Doc: re-generation workflow (custom files preserved) | Done |

---

## Future Improvements

### API Endpoint Design

The current endpoint structure works but could better follow REST conventions in a future iteration:

| Current | Improvement | Reason |
|---------|-------------|--------|
| `/book/` (singular + trailing slash) | `/books` (plural, no trailing slash) | REST convention: collections are plural |
| `GET /book/count/` separate endpoint | Return count in pagination response body | Fewer endpoints, count is metadata of a list |
| `GET /book/paginated/` separate endpoint | `GET /books?skip=0&limit=20` on the main GET | Pagination should be query params on the collection |
| `GET /book/search/` separate endpoint | `GET /books?title=foo&author=bar` on the main GET | Filtering/search = query params on the collection |
| Only `PUT` for updates | Add `PATCH /{id}` for partial updates | PUT requires all fields; PATCH allows updating just one field |

The main simplification: merge **3 GET endpoints** (list, paginated, search) into **1**:

```
# Current (3 endpoints):
GET /book/                              → all items
GET /book/paginated/?skip=0&limit=100   → paginated
GET /book/search/?name=foo              → filtered

# Proposed (1 endpoint):
GET /books?skip=0&limit=20&name=foo&detailed=true
```

### Pre-existing Template Bugs (Custom String PKs)

When a class uses a custom string primary key (e.g. `email: str` with `is_id=True`), the generated FK columns are incorrectly typed as `Mapped[int]` instead of `Mapped[str]`. This affects:

- Child class FK to parent in joined-table inheritance (`Student.id` referencing `Person.email`)
- Relationship FK columns (`Professor.department_id` referencing `Department.code`)
- Association table columns

This only manifests with custom string PKs — the default auto-generated `id: int` PK works fine. Fix requires updating `model_entity.py.j2` and the flat `sql_alchemy_template.py.j2` to resolve the parent PK type dynamically.

---

## Test Results

All **88 tests** pass across all generator packages:

- `tests/generators/backend/` — 72 tests (including OCL integration)
- `tests/generators/sqlalchemy/` — 11 tests
- `tests/generators/pydantic/` — 3 tests
- `tests/generators/rest_api/` — 2 tests
