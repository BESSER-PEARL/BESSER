"""Tests for db_reverse service and the database reverse-engineering endpoints.

The fixtures use SQLite (file-based) so the tests run anywhere without needing
a live PostgreSQL/MySQL/Oracle/MSSQL server.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.BUML.metamodel.structural.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    IntegerType,
    StringType,
)
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import (
    build_engine,
    db_to_domain_model,
    list_schemas_and_tables,
)

BASE_URL = "http://testserver"


def _run(coro):
    return asyncio.run(coro)


def _make_sqlite_db(path: str) -> None:
    """Create a small SQLite DB with two tables and an FK relationship."""
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE author (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            );
            CREATE TABLE book (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                pages INTEGER,
                rating REAL,
                published BOOLEAN,
                author_id INTEGER NOT NULL,
                FOREIGN KEY (author_id) REFERENCES author(id)
            );
            """
        )
        con.commit()
    finally:
        con.close()


@pytest.fixture
def sqlite_db(tmp_path):
    db_path = str(tmp_path / "fixture.sqlite")
    _make_sqlite_db(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------

def test_list_schemas_and_tables_for_sqlite(sqlite_db):
    engine = build_engine(dialect="sqlite", database=sqlite_db)
    try:
        schemas = list_schemas_and_tables(engine)
    finally:
        engine.dispose()

    # SQLAlchemy reports SQLite's implicit schema as "main".
    assert "main" in schemas
    assert sorted(schemas["main"]) == ["author", "book"]


def test_db_to_domain_model_classes_and_attributes(sqlite_db):
    engine = build_engine(dialect="sqlite", database=sqlite_db)
    try:
        model, warnings = db_to_domain_model(engine)
    finally:
        engine.dispose()

    assert isinstance(model, DomainModel)
    class_names = {c.name for c in model.types if isinstance(c, Class)}
    assert class_names == {"author", "book"}
    assert warnings == []

    by_name = {c.name: c for c in model.types if isinstance(c, Class)}
    author = by_name["author"]
    book = by_name["book"]

    author_attrs = {p.name: p for p in author.attributes}
    assert "id" in author_attrs and author_attrs["id"].is_id is True
    assert author_attrs["id"].type is IntegerType
    assert author_attrs["name"].type is StringType
    assert author_attrs["name"].is_optional is False  # NOT NULL
    assert author_attrs["email"].is_optional is True  # nullable

    book_attrs = {p.name: p for p in book.attributes}
    # author_id is modeled as an association, so it should NOT be an attribute.
    assert "author_id" not in book_attrs
    assert book_attrs["title"].type is StringType
    assert book_attrs["pages"].type is IntegerType


def test_db_to_domain_model_creates_fk_association(sqlite_db):
    engine = build_engine(dialect="sqlite", database=sqlite_db)
    try:
        model, _warnings = db_to_domain_model(engine)
    finally:
        engine.dispose()

    assocs = [a for a in model.associations if isinstance(a, BinaryAssociation)]
    assert len(assocs) == 1
    assoc = assocs[0]
    assert assoc.name == "book_author_id_to_author"

    end_types = {next(iter(e.type.name) if False else e.type.name for e in assoc.ends)}
    type_names = {e.type.name for e in assoc.ends}
    assert type_names == {"author", "book"}

    # Source side (FK column NOT NULL) should have lower bound 1.
    by_target = {e.type.name: e for e in assoc.ends}
    assert by_target["author"].multiplicity.min == 1
    assert by_target["author"].multiplicity.max == 1


def test_db_to_domain_model_respects_selection(sqlite_db):
    engine = build_engine(dialect="sqlite", database=sqlite_db)
    try:
        model, warnings = db_to_domain_model(engine, {"main": ["author"]})
    finally:
        engine.dispose()

    class_names = {c.name for c in model.types if isinstance(c, Class)}
    assert class_names == {"author"}
    # FK target ("author") is selected in this case, but book is not, so
    # the only association direction (book → author) cannot exist.
    assert all(not isinstance(a, BinaryAssociation) for a in model.associations) or \
        len([a for a in model.associations if isinstance(a, BinaryAssociation)]) == 0
    # No warnings expected when we just don't include the source table.
    assert warnings == []


def test_build_engine_rejects_unknown_dialect():
    with pytest.raises(ValueError, match="Unsupported dialect"):
        build_engine(dialect="cassandra", database="foo")


def test_build_engine_requires_database_for_structured_form():
    with pytest.raises(ValueError, match="database is required"):
        build_engine(dialect="postgresql", host="x", username="u", password="p")


def test_build_engine_accepts_raw_url(sqlite_db):
    engine = build_engine(raw_url=f"sqlite:///{sqlite_db}")
    try:
        schemas = list_schemas_and_tables(engine)
    finally:
        engine.dispose()
    assert "main" in schemas


# ---------------------------------------------------------------------------
# Bridge / junction table tests
# ---------------------------------------------------------------------------

def _make_library_db(path: str) -> None:
    """Library/Book/Author with two pure junction tables (M:N)."""
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE library (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE book (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                pages INTEGER
            );
            CREATE TABLE author (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE library_books (
                library_id INTEGER NOT NULL REFERENCES library(id),
                book_id INTEGER NOT NULL REFERENCES book(id),
                PRIMARY KEY (library_id, book_id)
            );
            CREATE TABLE book_authors (
                book_id INTEGER NOT NULL REFERENCES book(id),
                author_id INTEGER NOT NULL REFERENCES author(id),
                PRIMARY KEY (book_id, author_id)
            );
            """
        )
        con.commit()
    finally:
        con.close()


@pytest.fixture
def library_db(tmp_path):
    db_path = str(tmp_path / "library.sqlite")
    _make_library_db(db_path)
    return db_path


def test_pure_bridge_tables_become_m_to_n_associations(library_db):
    engine = build_engine(dialect="sqlite", database=library_db)
    try:
        model, warnings = db_to_domain_model(engine)
    finally:
        engine.dispose()

    class_names = {c.name for c in model.types if isinstance(c, Class)}
    # Bridge tables must NOT appear as classes.
    assert class_names == {"library", "book", "author"}
    assert warnings == []

    assocs = [a for a in model.associations if isinstance(a, BinaryAssociation)]
    # 2 M:N associations only — no FK 1:N pseudo-associations.
    assert len(assocs) == 2

    by_targets = {frozenset(e.type.name for e in a.ends): a for a in assocs}
    assert frozenset({"library", "book"}) in by_targets
    assert frozenset({"book", "author"}) in by_targets

    # Both ends should be (0, *) for an M:N relationship.
    for assoc in assocs:
        for end in assoc.ends:
            assert end.multiplicity.min == 0
            assert end.multiplicity.max > 1


def _make_assoc_class_db(path: str) -> None:
    """Junction with an extra payload column — should NOT collapse."""
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE student (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE course (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL
            );
            CREATE TABLE enrollment (
                student_id INTEGER NOT NULL REFERENCES student(id),
                course_id INTEGER NOT NULL REFERENCES course(id),
                grade TEXT,
                PRIMARY KEY (student_id, course_id)
            );
            """
        )
        con.commit()
    finally:
        con.close()


def test_junction_with_payload_stays_a_class(tmp_path):
    db_path = str(tmp_path / "enroll.sqlite")
    _make_assoc_class_db(db_path)
    engine = build_engine(dialect="sqlite", database=db_path)
    try:
        model, warnings = db_to_domain_model(engine)
    finally:
        engine.dispose()

    class_names = {c.name for c in model.types if isinstance(c, Class)}
    # `enrollment` carries a `grade` payload column → keep it as a Class.
    assert class_names == {"student", "course", "enrollment"}
    assert warnings == []

    # Two 1:N associations: enrollment → student and enrollment → course.
    assocs = [a for a in model.associations if isinstance(a, BinaryAssociation)]
    assert len(assocs) == 2
    pair_names = [
        frozenset(e.type.name for e in a.ends) for a in assocs
    ]
    assert frozenset({"enrollment", "student"}) in pair_names
    assert frozenset({"enrollment", "course"}) in pair_names


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

async def _post_json(url: str, json):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        return await ac.post(url, json=json)


async def _post_files(url: str, files):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        return await ac.post(url, files=files)


def test_db_introspect_endpoint_with_raw_url(sqlite_db):
    payload = {"connection": {"raw_url": f"sqlite:///{sqlite_db}"}}
    resp = _run(_post_json("/besser_api/db-introspect", payload))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "main" in data["schemas"]
    assert sorted(data["schemas"]["main"]) == ["author", "book"]


def test_db_to_domain_model_endpoint_with_raw_url(sqlite_db):
    payload = {
        "connection": {"raw_url": f"sqlite:///{sqlite_db}"},
        "selection": {"main": ["author", "book"]},
    }
    resp = _run(_post_json("/besser_api/db-to-domain-model", payload))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["diagramType"] == "ClassDiagram"
    elements = data["model"]["elements"]
    class_names = {
        el["name"] for el in elements.values() if el.get("type") == "Class"
    }
    assert class_names == {"author", "book"}
    rel_types = [r["type"] for r in data["model"]["relationships"].values()]
    assert any(t.startswith("Class") for t in rel_types)


def test_db_introspect_endpoint_rejects_unknown_dialect():
    # Pydantic Literal validation rejects unsupported dialects with 422
    # before the request reaches the service layer.
    payload = {"connection": {"dialect": "cassandra", "database": "x"}}
    resp = _run(_post_json("/besser_api/db-introspect", payload))
    assert resp.status_code == 422


def test_db_introspect_endpoint_requires_dialect_or_url():
    payload = {"connection": {"host": "localhost"}}
    resp = _run(_post_json("/besser_api/db-introspect", payload))
    assert resp.status_code == 400


def test_db_upload_sqlite_then_introspect_with_token(tmp_path):
    db_path = str(tmp_path / "upload.sqlite")
    _make_sqlite_db(db_path)
    with open(db_path, "rb") as fh:
        content = fh.read()

    files = [("file", ("fixture.sqlite", content, "application/octet-stream"))]
    upload = _run(_post_files("/besser_api/db-upload-sqlite", files))
    assert upload.status_code == 200, upload.text
    token = upload.json()["database_token"]
    assert token

    payload = {"connection": {"dialect": "sqlite", "database_token": token}}
    introspect = _run(_post_json("/besser_api/db-introspect", payload))
    assert introspect.status_code == 200, introspect.text
    assert sorted(introspect.json()["schemas"]["main"]) == ["author", "book"]


def test_db_upload_sqlite_rejects_non_sqlite_bytes():
    files = [("file", ("not_a_db.sqlite", b"hello world", "application/octet-stream"))]
    resp = _run(_post_files("/besser_api/db-upload-sqlite", files))
    assert resp.status_code == 400
    assert "SQLite" in resp.text


def test_db_upload_sqlite_rejects_unknown_extension():
    files = [("file", ("bad.txt", b"SQLite format 3\x00...", "text/plain"))]
    resp = _run(_post_files("/besser_api/db-upload-sqlite", files))
    assert resp.status_code == 415


def test_db_introspect_endpoint_rejects_bad_token():
    payload = {"connection": {"dialect": "sqlite", "database_token": "../../etc"}}
    resp = _run(_post_json("/besser_api/db-introspect", payload))
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
