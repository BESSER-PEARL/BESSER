"""Workflow (full-pipeline, output-verifying) tests for the two demo scenarios:

  * "only a database"    -> SQL + SQLAlchemy generators
  * "database + backend" -> FastAPI backend generator

Unlike the UNIT tests (which mock isolated functions, e.g. tests/generators/llm/
test_openai_provider.py), these run the REAL generator on a domain model
end-to-end and assert the *produced code* has the expected structure — the
regression net for "does generation actually work and stay complete".

This is the OFFLINE layer: deterministic, no network / no LLM — runs in CI.
The LIVE layer (the same two scenarios driven over the deployed HTTP endpoint)
lives in tests/live/test_generation_workflows_live.py.

The ``library_book_author_model`` fixture (tests/conftest.py) provides Library,
Book and Author with a 1..*/0..* association and an N:M association.
"""
import importlib.util
import sys

from besser.generators.backend import BackendGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _load_module(path, name):
    """Import a generated .py file as a module (side-effect free — table
    creation in the generated SQLAlchemy file is guarded by ``__main__``)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ======================================================================
# Scenario: "only a database"
# ======================================================================

class TestOnlyDatabaseWorkflow:
    """User says "generate a database" -> a SQL schema / SQLAlchemy models for
    the domain model, one table per class."""

    def test_sql_schema_has_a_table_per_class(self, library_book_author_model, tmp_path):
        out = tmp_path / "sql"
        out.mkdir()
        SQLGenerator(
            model=library_book_author_model,
            output_dir=str(out), sql_dialect="sqlite",
        ).generate()

        sql_file = out / "tables_sqlite.sql"
        assert sql_file.is_file(), "SQL generator did not write tables_sqlite.sql"
        ddl = sql_file.read_text(encoding="utf-8").lower()
        # A CREATE TABLE per domain class (class names are lowercased).
        for table in ("library", "book", "author"):
            assert f"create table {table}" in ddl, f"missing CREATE TABLE for {table}"
        # >= 3 class tables (plus the N:M join table).
        assert ddl.count("create table") >= 3, f"too few tables:\n{ddl}"

    def test_sqlalchemy_models_import_and_build_their_tables(self, library_book_author_model, tmp_path):
        out = tmp_path / "sa"
        out.mkdir()
        SQLAlchemyGenerator(
            model=library_book_author_model, output_dir=str(out),
        ).generate(dbms="sqlite")

        sa_file = out / "sql_alchemy.py"
        assert sa_file.is_file(), "SQLAlchemy generator did not write sql_alchemy.py"
        code = sa_file.read_text(encoding="utf-8")
        assert "class Base(DeclarativeBase):" in code
        for cls in ("Library", "Book", "Author"):
            assert f"class {cls}(Base):" in code, f"missing ORM class {cls}"

        # The generated module must import cleanly AND actually create its
        # tables against a real (in-memory) engine — proves it's runnable code,
        # not just plausible-looking text.
        module = _load_module(sa_file, "wf_sql_alchemy")
        from sqlalchemy import create_engine, inspect
        engine = create_engine("sqlite:///:memory:")
        module.Base.metadata.create_all(engine)
        tables = set(inspect(engine).get_table_names())
        assert {"library", "book", "author"} <= tables, f"tables built: {tables}"


# ======================================================================
# Scenario: "database + backend" (FastAPI)
# ======================================================================

class TestDatabaseAndBackendWorkflow:
    """User says "generate the backend" -> a runnable FastAPI app with
    persistence and CRUD routes per class."""

    def test_fastapi_backend_has_app_and_crud_per_class(self, library_book_author_model, tmp_path):
        out = tmp_path / "backend"
        out.mkdir()
        BackendGenerator(model=library_book_author_model, output_dir=str(out)).generate()

        # Core files of a FastAPI backend.
        for fname in ("main_api.py", "database.py", "pydantic_classes.py",
                      "sql_alchemy.py", "requirements.txt"):
            assert (out / fname).is_file(), f"backend missing {fname}"

        main_api = (out / "main_api.py").read_text(encoding="utf-8")
        assert "FastAPI(" in main_api, "main_api.py is not a FastAPI app"
        assert "def get_db(" in (out / "database.py").read_text(encoding="utf-8")

        # One CRUD router per class, wired into the app, exposing GET-all.
        for cls in ("library", "book", "author"):
            router = out / "routers" / f"{cls}.py"
            assert router.is_file(), f"missing router module for {cls}"
            rcode = router.read_text(encoding="utf-8")
            assert f'@router.get("/{cls}/"' in rcode, f"{cls} router missing GET-all route"
            assert f"from routers import {cls} as {cls}_router" in main_api, \
                f"{cls} router not imported in main_api.py"
            assert f"app.include_router({cls}_router.router)" in main_api, \
                f"{cls} router not wired into the app"

        # A Pydantic create-schema per class.
        pyd = (out / "pydantic_classes.py").read_text(encoding="utf-8")
        for cls in ("Library", "Book", "Author"):
            assert f"class {cls}Create(BaseModel):" in pyd, f"missing {cls}Create schema"
