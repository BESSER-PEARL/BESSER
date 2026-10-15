"""Scaffold map, FastAPI family: anchors over REAL BackendGenerator output.

The map is computed from the files on disk, so these tests generate the
scaffold into a temporary workspace (as the agent's ``_gen_fastapi_backend``
does) and assert anchors by id, never by absolute line numbers: line numbers
shift with every model and template change, ids must not.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import re
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    Class,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    Generalization,
    IntegerType,
    Method,
    MethodImplementationType,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)
from besser.spec_driven_agent.planning.scaffold_map import fastapi as fastapi_map
from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_CLASS,
    KIND_INSERT,
    KIND_ROUTE,
    KIND_STUB,
    WorkspaceMap,
)
from besser.spec_driven_agent.planning.scaffold_map.render import render_locate, render_overview
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE"]
CLASS_NAMES = ["Person", "Library", "Book", "Author"]


# ---------------------------------------------------------------------------
# Fixtures: a library model with an enum, a generalization, two associations,
# a CODE method and two body-less methods (HTTP 501 stubs).
# ---------------------------------------------------------------------------

def _library_model() -> DomainModel:
    genre = Enumeration(name="Genre", literals={
        EnumerationLiteral(name="FICTION"), EnumerationLiteral(name="SCIENCE"),
    })
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    library = Class(name="Library", attributes={
        Property(name="name", type=StringType), Property(name="address", type=StringType),
    })
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType), Property(name="pages", type=IntegerType),
        Property(name="release", type=DateType), Property(name="genre", type=genre),
    })
    author = Class(name="Author", attributes={Property(name="email", type=StringType)})
    book.add_method(Method(name="applyDiscount", parameters=[Parameter(name="percent", type=FloatType)],
                           type=FloatType))
    book.add_method(Method(name="describe", parameters=[], type=StringType,
                           implementation_type=MethodImplementationType.CODE,
                           code="def describe(self):\n    return f'{self.title}'\n"))
    author.add_method(Method(name="notify", parameters=[], type=None))
    located_in = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    written_by = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
    publishes = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
    return DomainModel(
        name="Library_Model",
        types={genre, person, library, book, author},
        associations={
            BinaryAssociation(name="lib_book", ends={located_in, has}),
            BinaryAssociation(name="book_author", ends={written_by, publishes}),
        },
        generalizations={Generalization(general=person, specific=author)},
    )


def _generate(model: DomainModel, out: Path) -> None:
    """Exactly how tool_executor._gen_fastapi_backend calls the generator."""
    from besser.generators.backend import BackendGenerator

    BackendGenerator(model=model, output_dir=str(out), http_methods=list(HTTP_METHODS)).generate()
    assert (out / "main_api.py").is_file()


@pytest.fixture(scope="module")
def library_model() -> DomainModel:
    return _library_model()


@pytest.fixture(scope="module")
def generated(tmp_path_factory, library_model) -> Path:
    """A pristine generated workspace; tests that edit files copy it first."""
    ws = tmp_path_factory.mktemp("fastapi_ws")
    _generate(library_model, ws / "backend")
    return ws


@pytest.fixture
def workspace(generated, tmp_path) -> Path:
    ws = tmp_path / "ws"
    shutil.copytree(generated, ws)
    return ws


def _map(ws: Path, class_names=CLASS_NAMES):
    maps = fastapi_map.extract(SourceIndex(str(ws)), list(class_names))
    return maps


def _only(ws: Path, class_names=CLASS_NAMES):
    maps = _map(ws, class_names)
    assert len(maps) == 1, [m.root for m in maps]
    return maps[0]


def _anchors(smap) -> dict:
    return {a.id: a for a in smap.all_anchors()}


def _assert_quotes_valid(ws: Path, anchors) -> None:
    for anchor in anchors:
        lines = (ws / anchor.file).read_text(encoding="utf-8").splitlines()
        assert 1 <= anchor.start_line <= anchor.end_line <= max(1, len(lines)), anchor
        if not anchor.quote:
            continue
        assert lines.count(anchor.quote) == 1, (anchor.id, anchor.quote)
        # modify_file matches old_text as a substring: unique in the whole text.
        assert (ws / anchor.file).read_text(encoding="utf-8").count(anchor.quote) == 1, (anchor.id, anchor.quote)
        at = lines.index(anchor.quote) + 1
        assert anchor.start_line <= at <= anchor.end_line, (anchor.id, at, anchor.start_line, anchor.end_line)


# ---------------------------------------------------------------------------
# The generated modular backend
# ---------------------------------------------------------------------------

def test_generated_backend_is_mapped_per_entity(generated):
    smap = _only(generated)
    assert smap.family == "fastapi"
    assert smap.root == "backend"
    assert smap.title == "FastAPI backend (modular routers)"
    assert sorted(smap.entities) == sorted(CLASS_NAMES)
    anchors = _anchors(smap)

    # App-level anchors of main_api.py, database.py and requirements.txt.
    for anchor_id in ("app", "router_imports_end", "include_routers", "include_routers_end", "middleware",
                      "exception_handlers", "main_guard", "init_db", "get_db", "requirements_end"):
        assert f"fastapi:{anchor_id}" in anchors, anchor_id
    assert anchors["fastapi:include_routers"].kind == KIND_BLOCK
    assert anchors["fastapi:include_routers_end"].kind == KIND_INSERT
    assert anchors["fastapi:include_routers_end"].quote == "app.include_router(author_methods_router.router)"
    assert anchors["fastapi:router_imports_end"].quote.startswith("from routers import ")
    assert anchors["fastapi:app"].quote == "app = FastAPI("
    assert anchors["fastapi:init_db"].file == "backend/database.py"
    assert "sql_alchemy.py" in anchors["fastapi:init_db"].hint     # the module-level engine is not the app's
    assert "NO trailing newline" in anchors["fastapi:requirements_end"].hint
    assert "CORSMiddleware" in anchors["fastapi:middleware"].label

    # Per entity: ORM, schema, router routes (decorator-inclusive), insert point.
    book = smap.entities["Book"]
    ids = {a.id for a in book.anchors}
    assert {"fastapi:orm:book", "fastapi:orm_add_column:book", "fastapi:orm_relationships:book",
            "fastapi:schema:book:BookCreate", "fastapi:router:book", "fastapi:methods_router:book",
            "fastapi:insert_static:book", "fastapi:ctor:book:create_book",
            "fastapi:setattr:book:update_book"} <= ids
    # Book's bulk create refuses every payload (required relationships): no constructor to extend there.
    assert "fastapi:ctor:book:bulk_create_book" not in ids
    assert "fastapi:ctor:author:bulk_create_author" in anchors
    for fn in ("get_all_book", "get_count_book", "get_paginated_book", "search_book", "get_book", "create_book",
               "bulk_create_book", "bulk_delete_book", "update_book", "delete_book", "add_writtenBy_to_book",
               "remove_writtenBy_from_book", "get_writtenBy_of_book", "execute_book_applyDiscount",
               "execute_book_describe"):
        anchor = anchors[f"fastapi:route:book:{fn}"]
        assert anchor.kind == KIND_ROUTE and anchor.entity == "Book"
        lines = (generated / anchor.file).read_text(encoding="utf-8").splitlines()
        assert lines[anchor.start_line - 1].lstrip().startswith("@router."), fn     # decorator-inclusive
    get_book = anchors["fastapi:route:book:get_book"]
    assert get_book.label == "GET /book/{book_id}/ -> get_book"
    assert get_book.quote.startswith("async def get_book(")
    assert anchors["fastapi:orm:book"].quote == "class Book(Base):"
    assert anchors["fastapi:schema:book:BookCreate"].kind == KIND_CLASS
    assert "genre" in anchors["fastapi:schema:book:BookCreate"].label
    # Relationship routes are attributed by their "<Class> Relationships" tag.
    assert anchors["fastapi:route:author:add_publishes_to_author"].entity == "Author"
    assert not smap.warnings
    assert all(not e.missing for e in smap.entities.values())


def test_every_quote_is_unique_and_inside_its_span(generated):
    smap = _only(generated)
    _assert_quotes_valid(generated, smap.all_anchors())
    ids = [a.id for a in smap.all_anchors()]
    assert len(ids) == len(set(ids))
    assert not [i for i in ids if re.search(r":\d+$", i)]      # no collision suffix on generated code


def test_method_stubs_carry_the_body_contract(generated):
    anchors = _anchors(_only(generated))
    stub = anchors["fastapi:stub:book:applyDiscount"]
    assert stub.kind == KIND_STUB
    assert stub.status == "HTTP 501 stub"
    # The identity comment '# Book.applyDiscount' is also the start of '# Book.applyDiscount: load
    # the instance', so it is no modify_file anchor; the quote is the stub's first unique line.
    text = (generated / stub.file).read_text(encoding="utf-8")
    assert stub.quote and text.count(stub.quote) == 1
    assert "(params or {}).get('percent')" in stub.hint
    assert "_book_object" in stub.hint
    assert "compare `.name`" in stub.hint                       # Book uses the Genre enum
    lines = (generated / stub.file).read_text(encoding="utf-8").splitlines()
    assert "status_code=501" in "\n".join(lines[stub.start_line - 1:stub.end_line])
    notify = anchors["fastapi:stub:author:notify"]
    assert "empty body {}" in notify.hint
    assert "compare `.name`" not in notify.hint                 # Author has no enum column
    # A CODE method is implemented: a route, but no stub.
    assert "fastapi:stub:book:describe" not in anchors
    assert anchors["fastapi:route:book:execute_book_applyDiscount"].status == "stub (HTTP 501)"


def test_static_route_insert_point_and_inheritance(generated):
    smap = _only(generated)
    anchors = _anchors(smap)
    insert = anchors["fastapi:insert_static:book"]
    assert insert.kind == KIND_INSERT
    assert insert.quote.startswith('@router.get("/book/{book_id}/"')
    assert insert.start_line == anchors["fastapi:route:book:get_book"].start_line
    assert "BEFORE" in insert.hint and "422" in insert.hint
    # One physical line holds every ORM constructor kwarg in create_book.
    ctor = anchors["fastapi:ctor:book:create_book"]
    assert "pages=book_data.pages" in ctor.quote and "title=book_data.title" in ctor.quote
    assert "ONE physical line" in ctor.hint
    # The setattr anchor is a direct statement of update_book, not the FK branch.
    assert anchors["fastapi:setattr:book:update_book"].quote.strip().startswith("setattr(db_book, 'title'")
    # Inheritance: Author re-emits Person's fields; the field recipe says so.
    assert "Person" in anchors["fastapi:orm:author"].hint
    assert "extends Person" in smap.entities["Author"].summary
    field_recipe = next(r for r in smap.recipes if r.intent == "Add a field")
    assert "subclass routers re-emit inherited fields" in " ".join(field_recipe.steps)


def test_recipes_name_existing_anchor_ids(generated):
    smap = _only(generated)
    ids = {a.id for a in smap.all_anchors()}
    intents = {r.intent for r in smap.recipes}
    assert {"Add an endpoint for an entity", "Implement a model method", "Add a field", "Add a new entity",
            "Add auth"} <= intents
    # Seeding and packages are demoted to the app-level summary (dropped first under budget pressure).
    assert any("seed data" in line and "`fastapi:include_routers_end`" in line for line in smap.summary)
    assert any("`fastapi:requirements_end`" in line and "newline" in line for line in smap.summary)
    text = " ".join(" ".join(r.steps) for r in smap.recipes) + " " + " ".join(smap.summary)
    assert not re.search(r"\bline \d+", text)
    for ref in re.findall(r"`(fastapi:[^`]+)`", text):
        concrete = (ref.replace("<entity>", "book").replace("<method>", "applyDiscount"))
        assert concrete in ids, ref
    for entity in smap.entities.values():
        assert entity.summary and not any(re.search(r"\d+-\d+|line \d", s) for s in entity.summary)


def test_model_class_without_code_is_missing(generated):
    smap = _only(generated, CLASS_NAMES + ["Loan"])
    assert smap.entities["Loan"].missing == ["ORM class", "Create schema", "router"]
    assert not smap.entities["Loan"].anchors
    assert "MISSING: ORM class, Create schema, router" in render_overview(WorkspaceMap(maps=[smap]))
    # Never claim missing pieces without a domain model.
    bare = _only(generated, [])
    assert "Loan" not in bare.entities
    assert all(not e.missing for e in bare.entities.values())
    assert sorted(bare.entities) == sorted(CLASS_NAMES)           # named after the ORM classes


def test_map_is_deterministic(generated):
    first = [dataclasses.asdict(m) for m in _map(generated)]
    second = [dataclasses.asdict(m) for m in _map(generated)]
    assert first == second


# ---------------------------------------------------------------------------
# Pitfalls detected in edited code
# ---------------------------------------------------------------------------

def test_dead_app_route_unmounted_router_and_captured_static_route(workspace):
    backend = workspace / "backend"
    main_api = backend / "main_api.py"
    main_api.write_text(main_api.read_text(encoding="utf-8") + (
        '\n\n@app.get("/book/")\n'
        "def list_books_again(database: Session = Depends(get_db)):\n"
        "    return []\n"), encoding="utf-8")
    (backend / "routers" / "reports.py").write_text(
        "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
        '@router.get("/reports/occupancy/")\n'
        "def occupancy():\n"
        "    return {}\n", encoding="utf-8")
    book = backend / "routers" / "book.py"
    book.write_text(book.read_text(encoding="utf-8") + (
        '\n\n@router.get("/book/stats/", response_model=None, tags=["Book"])\n'
        "def book_stats(database: Session = Depends(get_db)):\n"
        "    return {}\n"), encoding="utf-8")

    smap = _only(workspace)
    warnings = "\n".join(smap.warnings)
    assert "backend/main_api.py list_books_again (GET /book/) is dead code" in warnings
    assert "get_all_book" in warnings
    assert "backend/routers/reports.py: APIRouter `router` is never mounted" in warnings
    assert "book_stats (GET /book/stats/) is unreachable" in warnings and "422" in warnings
    anchors = _anchors(smap)
    assert anchors["fastapi:route:book:list_books_again"].status == "never served"
    assert anchors["fastapi:route:book:book_stats"].status == "never served"
    assert anchors["fastapi:route:occupancy"].status == "not mounted (404)"
    assert anchors["fastapi:router:reports"].label.endswith("(NOT MOUNTED: its routes 404)")
    _assert_quotes_valid(workspace, smap.all_anchors())
    # The insert point still names the route the static path must precede.
    assert anchors["fastapi:insert_static:book"].quote.startswith('@router.get("/book/{book_id}/"')
    # locate_code finds the dead routes by the hint's wording.
    found = render_locate(WorkspaceMap(maps=[smap]), query="never served")
    assert found["count"] == 2


def test_app_decorator_inside_a_router_file_is_flagged(workspace):
    book = workspace / "backend" / "routers" / "book.py"
    book.write_text(book.read_text(encoding="utf-8") + (
        '\n\n@app.get("/book/export/")\n'
        "def export_books():\n"
        "    return []\n"), encoding="utf-8")
    smap = _only(workspace)
    assert any("export_books is decorated with @app" in w for w in smap.warnings)


def test_string_primary_key_capture_is_a_misleading_404(tmp_path):
    seat = Class(name="Seat")
    seat.attributes = {Property(name="code", type=StringType, is_id=True), Property(name="row", type=IntegerType)}
    _generate(DomainModel(name="Venue", types={seat}), tmp_path / "backend")
    router = tmp_path / "backend" / "routers" / "seat.py"
    router.write_text(router.read_text(encoding="utf-8") + (
        '\n\n@router.get("/seat/stats/", response_model=None, tags=["Seat"])\n'
        "def seat_stats():\n"
        "    return {}\n"), encoding="utf-8")
    smap = _only(tmp_path, ["Seat"])
    insert = _anchors(smap)["fastapi:insert_static:seat"]
    assert "misleading 404" in insert.hint and "422" not in insert.hint
    assert any("seat_stats" in w and "misleading 404" in w for w in smap.warnings)


def test_malformed_files_are_skipped_not_fatal(workspace):
    book = workspace / "backend" / "routers" / "book.py"
    book.write_text(book.read_text(encoding="utf-8") + "\ndef broken(:\n", encoding="utf-8")
    index = SourceIndex(str(workspace))
    maps = fastapi_map.extract(index, CLASS_NAMES + ["Loan"])
    assert len(maps) == 1
    smap = maps[0]
    assert "backend/routers/book.py" in index.syntax_errors
    ids = {a.id for a in smap.all_anchors()}
    assert "fastapi:route:book:get_book" not in ids
    assert "fastapi:route:library:get_library" in ids
    assert "fastapi:stub:book:applyDiscount" in ids                 # book_methods.py still parses
    # The broken file mentions Book: never claim its router is missing.
    assert not smap.entities["Book"].missing
    assert smap.entities["Loan"].missing == ["ORM class", "Create schema", "router"]


def test_broken_app_file_still_maps_the_routers(workspace):
    main_api = workspace / "backend" / "main_api.py"
    main_api.write_text(main_api.read_text(encoding="utf-8") + "\nif True\n", encoding="utf-8")
    smap = _only(workspace)
    ids = {a.id for a in smap.all_anchors()}
    assert "fastapi:route:book:get_book" in ids
    assert "fastapi:include_routers" not in ids
    assert not smap.warnings                                      # mounts unknown: no "never mounted" claims


# ---------------------------------------------------------------------------
# Layouts: several roots, single-file app, hand-written routers with prefixes
# ---------------------------------------------------------------------------

def test_two_roots_get_unique_ids(generated, tmp_path, library_model):
    ws = tmp_path / "ws"
    shutil.copytree(generated, ws)
    _generate(library_model, ws / "web_app" / "backend")         # the web_app layout's backend half
    maps = _map(ws)
    assert [m.root for m in maps] == ["backend", "web_app/backend"]
    ids = [a.id for m in maps for a in m.all_anchors()]
    assert len(ids) == len(set(ids))
    assert "fastapi@backend:route:book:get_book" in ids
    assert "fastapi@web_app_backend:route:book:get_book" in ids
    web = maps[1]
    assert all(a.file.startswith("web_app/backend/") for a in web.all_anchors())
    assert any("`fastapi@backend:insert_static:<entity>`" in " ".join(r.steps) for r in maps[0].recipes)
    # The second root's recipes are the first's modulo ids and paths: said once, pointed to.
    assert [r.intent for r in web.recipes] == ["Same recipes as `backend/`"]
    assert "`fastapi@web_app_backend:…`" in web.recipes[0].steps[0]
    overview = render_overview(WorkspaceMap(maps=maps))
    assert len(overview) <= 4000
    first, second = overview.split("— `web_app/backend/`")
    for section in (first, second):
        assert "Per entity:" in section and "501 stubs: applyDiscount" in section


def test_single_file_rest_api_layout(tmp_path, library_model):
    from besser.generators.rest_api import RESTAPIGenerator

    RESTAPIGenerator(model=library_model, output_dir=str(tmp_path / "rest_api"),
                     http_methods=list(HTTP_METHODS)).generate()
    smap = _only(tmp_path)
    assert smap.root == "rest_api"
    assert smap.title == "FastAPI app (single file, @app routes)"
    anchors = _anchors(smap)
    routes = [a for a in anchors.values() if a.kind == KIND_ROUTE and a.entity == "Book"]
    assert {a.label.split(" -> ")[0] for a in routes} >= {"GET /book/", "GET /book/{id}/", "POST /book/"}
    assert anchors["fastapi:insert_static:book"].file == "rest_api/rest_api.py"
    assert "fastapi:schema:book:Book" in anchors
    recipes = " ".join(" ".join(r.steps) for r in smap.recipes)
    assert "@app.<verb>" in recipes and "@router" not in recipes
    assert any("in-memory" in line for line in smap.summary)
    _assert_quotes_valid(tmp_path, smap.all_anchors())          # duplicate function names stay unique ids
    assert len(anchors) == len(smap.all_anchors())


def test_handwritten_app_with_prefixes_and_relative_imports(tmp_path):
    app_dir = tmp_path / "server" / "app"
    (app_dir / "routers").mkdir(parents=True)
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "routers" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from .routers import books\n\n"
        "app = FastAPI()\n"
        'app.include_router(books.router, prefix="/api")\n', encoding="utf-8")
    (app_dir / "routers" / "books.py").write_text(
        "from fastapi import APIRouter\n\n"
        'router = APIRouter(prefix="/books", tags=["books"])\n\n\n'
        '@router.get("/")\n'
        "def list_books():\n"
        "    return []\n\n\n"
        '@router.get("/{book_id}")\n'
        "def read_book(book_id: int):\n"
        "    return {}\n\n\n"
        '@router.get("/featured")\n'
        "def featured_books():\n"
        "    return []\n", encoding="utf-8")
    (app_dir / "database.py").write_text(
        "from sqlalchemy import create_engine\n"
        "from sqlalchemy.orm import declarative_base, sessionmaker\n\n"
        'engine = create_engine("sqlite:///./app.db")\n'
        "SessionLocal = sessionmaker(bind=engine)\n"
        "Base = declarative_base()\n\n\n"
        "def get_db():\n"
        "    db = SessionLocal()\n"
        "    try:\n"
        "        yield db\n"
        "    finally:\n"
        "        db.close()\n", encoding="utf-8")
    (app_dir / "models.py").write_text(
        "from sqlalchemy import Column, Integer, String\n"
        "from .database import Base\n\n\n"
        "class Book(Base):\n"
        '    __tablename__ = "books"\n'
        "    id = Column(Integer, primary_key=True)\n"
        "    title = Column(String)\n", encoding="utf-8")
    (app_dir / "schemas.py").write_text(
        "from pydantic import BaseModel\n\n\n"
        "class BookBase(BaseModel):\n"
        "    title: str\n\n\n"
        "class BookCreate(BookBase):\n"
        "    pass\n", encoding="utf-8")
    smap = _only(tmp_path, ["Book", "Member"])
    assert smap.root == "server/app"
    anchors = _anchors(smap)
    assert anchors["fastapi:route:book:read_book"].label == "GET /api/books/{book_id} -> read_book"
    assert anchors["fastapi:route:book:read_book"].entity == "Book"            # plural path segment
    assert anchors["fastapi:route:book:featured_books"].status == "never served"
    assert any("featured_books" in w and "422" in w for w in smap.warnings)
    assert "fastapi:include_routers_end" in anchors
    assert anchors["fastapi:router_imports_end"].quote == "from .routers import books"
    assert anchors["fastapi:orm:book"].file == "server/app/models.py"
    assert "id, title" in anchors["fastapi:orm:book"].label
    assert {"fastapi:schema:book:BookBase", "fastapi:schema:book:BookCreate"} <= set(anchors)
    assert anchors["fastapi:init_db"].quote == 'engine = create_engine("sqlite:///./app.db")'
    assert anchors["fastapi:get_db"].file == "server/app/database.py"
    assert not smap.entities["Book"].missing
    assert smap.entities["Member"].missing == ["ORM class", "Create schema", "router"]
    _assert_quotes_valid(tmp_path, smap.all_anchors())


def test_custom_entity_router_and_captured_app_route(workspace):
    backend = workspace / "backend"
    (backend / "routers" / "book_reports.py").write_text(
        "from fastapi import APIRouter, Depends\n"
        "from database import get_db\n\n"
        "router = APIRouter()\n\n\n"
        '@router.get("/book/{book_id}/report/", tags=["Book"])\n'
        "def book_report(book_id: int):\n"
        "    return {}\n", encoding="utf-8")
    main_api = backend / "main_api.py"
    text = main_api.read_text(encoding="utf-8")
    text = text.replace("from routers import author_methods as author_methods_router\n",
                        "from routers import author_methods as author_methods_router\n"
                        "from routers import book_reports as book_reports_router\n", 1)
    text = text.replace("app.include_router(author_methods_router.router)\n",
                        "app.include_router(author_methods_router.router)\n"
                        "app.include_router(book_reports_router.router)\n", 1)
    text += '\n\n@app.get("/book/overdue/")\ndef overdue_books():\n    return []\n'
    main_api.write_text(text, encoding="utf-8")

    smap = _only(workspace)
    anchors = _anchors(smap)
    assert anchors["fastapi:router:book"].file == "backend/routers/book.py"       # the CRUD router keeps it
    assert anchors["fastapi:router:book:book_reports"].file == "backend/routers/book_reports.py"
    assert anchors["fastapi:route:book:book_report"].status is None               # mounted, reachable
    assert anchors["fastapi:include_routers_end"].quote == "app.include_router(book_reports_router.router)"
    assert anchors["fastapi:router_imports_end"].quote == "from routers import book_reports as book_reports_router"
    assert any("overdue_books" in w and "move it into backend/routers/book.py above get_book" in w
               for w in smap.warnings)
    assert any("custom GET /book/{book_id}/report/" in line for line in smap.entities["Book"].summary)
    _assert_quotes_valid(workspace, smap.all_anchors())


# ---------------------------------------------------------------------------
# Mounts the module-level walk cannot place, decorator stacks, several roots
# ---------------------------------------------------------------------------

_REPORTS_ROUTER = (
    "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
    '@router.get("/reports/")\n'
    "def reports():\n"
    "    return []\n"
)


def _include_block(ws: Path) -> tuple[str, list[str]]:
    """The generated run of ``app.include_router(<x>.router)`` lines and the router names, in order."""
    text = (ws / "backend" / "main_api.py").read_text(encoding="utf-8")
    block = re.search(r"(?m)(?:^app\.include_router\(\w+\.router\)\n)+", text)
    assert block is not None
    return block.group(0), re.findall(r"include_router\((\w+)\.router\)", block.group(0))


def _edit_main(ws: Path, old: str, new: str) -> None:
    main_api = ws / "backend" / "main_api.py"
    text = main_api.read_text(encoding="utf-8")
    old, new = old.replace("<INCLUDES>", _include_block(ws)[0]), new.replace("<INCLUDES>", _include_block(ws)[0])
    assert old in text
    main_api.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_include_router_loop_is_resolved_in_order(workspace):
    routers = _include_block(workspace)[1]
    assert len(routers) >= 6
    _edit_main(workspace, "<INCLUDES>",
               f"for _r in ({', '.join(routers[:3])},\n           {', '.join(routers[3:])}):\n"
               "    app.include_router(_r.router)\n")
    (workspace / "backend" / "routers" / "reports.py").write_text(_REPORTS_ROUTER, encoding="utf-8")
    smap = _only(workspace)
    anchors = _anchors(smap)
    # Every looped router is mounted; the one left out is still reported, since the loop resolved.
    assert all(not e.missing for e in smap.entities.values())
    assert anchors["fastapi:route:book:get_book"].status is None
    assert [w for w in smap.warnings if "never mounted" in w] == [
        w for w in smap.warnings if w.startswith("backend/routers/reports.py: APIRouter `router` is never mounted")]
    assert len(smap.warnings) == 1
    # The insertion point is the loop itself, not its indented body line.
    end = anchors["fastapi:include_routers_end"]
    assert end.quote.startswith("for _r in (") and "tuple" in end.hint
    _assert_quotes_valid(workspace, smap.all_anchors())


@pytest.mark.parametrize("old, new", [
    # Mounted inside an if block.
    ("app.include_router(author_router.router)\n", "if True:\n    app.include_router(author_router.router)\n"),
    # An optional router mounted in a try block.
    ("<INCLUDES>", "<INCLUDES>" + "try:\n    from routers import reports as reports_router\n"
                            "    app.include_router(reports_router.router)\nexcept ImportError:\n    pass\n"),
    # Mounted by a helper function on a parameter.
    ("<INCLUDES>", "<INCLUDES>" + "\n\ndef _register_extra_routers(application):\n"
                            "    from routers import reports as reports_router\n"
                            "    application.include_router(reports_router.router)\n\n\n"
                            "_register_extra_routers(app)\n"),
], ids=["if", "try", "helper"])
def test_mounts_the_walk_cannot_place_suppress_never_mounted_claims(workspace, old, new):
    (workspace / "backend" / "routers" / "reports.py").write_text(_REPORTS_ROUTER, encoding="utf-8")
    _edit_main(workspace, old, new)
    smap = _only(workspace, CLASS_NAMES)
    assert not [w for w in smap.warnings if "never mounted" in w]
    assert all("include_router mount" not in e.missing for e in smap.entities.values())
    assert not [a for a in smap.all_anchors() if a.status == "not mounted (404)"]
    assert "NOT MOUNTED" not in _anchors(smap)["fastapi:router:author"].label


def test_insert_point_sits_above_the_whole_decorator_stack(workspace):
    book = workspace / "backend" / "routers" / "book.py"
    text = book.read_text(encoding="utf-8")
    decorator = '@router.get("/book/{book_id}/", response_model=None, tags=["Book"])'
    assert decorator in text
    book.write_text(text.replace(decorator, f'@audit_log("read")\n{decorator}', 1), encoding="utf-8")
    anchors = _anchors(_only(workspace))
    insert, route = anchors["fastapi:insert_static:book"], anchors["fastapi:route:book:get_book"]
    assert insert.start_line == route.start_line
    assert insert.quote == '@audit_log("read")'               # new routes never land between the decorators
    assert "BEFORE this line" in insert.hint
    _assert_quotes_valid(workspace, [insert, route])


def test_secondary_root_claims_nothing_the_main_backend_serves(workspace):
    (workspace / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n"
        '@app.get("/book/")\n'
        "def proxy_books():\n"
        "    return []\n", encoding="utf-8")
    maps = {m.root: m for m in _map(workspace, CLASS_NAMES + ["Loan"])}
    assert set(maps) == {"", "backend"}
    proxy, backend = maps[""], maps["backend"]
    assert sorted(proxy.entities) == ["Book"]                     # no "no code found" lines for the others
    assert all(not e.missing for e in proxy.entities.values())
    # A class absent everywhere is reported once, by the root that serves the model.
    assert backend.entities["Loan"].missing == ["ORM class", "Create schema", "router"]
    assert all(not e.missing for name, e in backend.entities.items() if name != "Loan")
    ids = [a.id for m in maps.values() for a in m.all_anchors()]
    assert len(ids) == len(set(ids))


def test_single_file_schemas_named_like_classes_with_suffixes(tmp_path):
    from besser.generators.rest_api import RESTAPIGenerator

    survey = Class(name="Survey", attributes={Property(name="title", type=StringType)})
    response = Class(name="SurveyResponse", attributes={Property(name="answer", type=StringType)})
    knowledge = Class(name="KnowledgeBase", attributes={Property(name="topic", type=StringType)})
    check_in = Class(name="CheckIn", attributes={Property(name="at", type=StringType)})
    model = DomainModel(name="Feedback", types={survey, response, knowledge, check_in}, associations={
        BinaryAssociation(name="answers", ends={
            Property(name="survey", type=survey, multiplicity=Multiplicity(1, 1)),
            Property(name="responses", type=response, multiplicity=Multiplicity(0, "*"))})})
    RESTAPIGenerator(model=model, output_dir=str(tmp_path / "rest_api"), http_methods=list(HTTP_METHODS)).generate()
    names = ["Survey", "SurveyResponse", "KnowledgeBase", "CheckIn"]
    smap = _only(tmp_path, names)
    assert sorted(smap.entities) == sorted(names)
    assert all(not e.missing for e in smap.entities.values())
    schemas = {a.entity: a.id for a in smap.all_anchors() if ":schema:" in a.id}
    assert schemas == {
        "Survey": "fastapi:schema:survey:Survey",
        "SurveyResponse": "fastapi:schema:surveyresponse:SurveyResponse",
        "KnowledgeBase": "fastapi:schema:knowledgebase:KnowledgeBase",
        "CheckIn": "fastapi:schema:checkin:CheckIn",
    }
    # Without a domain model the route paths still name the right entity.
    assert sorted(_only(tmp_path, []).entities) == sorted(names)


# ---------------------------------------------------------------------------
# Association class, inherited relationships, read-only class
# ---------------------------------------------------------------------------

def _shop_model() -> DomainModel:
    trip = Class(name="Trip", attributes={Property(name="id", type=IntegerType, is_id=True),
                                          Property(name="reference", type=StringType)})
    seat = Class(name="Seat", attributes={Property(name="code", type=StringType, is_id=True),
                                          Property(name="label", type=StringType)})
    tag = Class(name="Tag", attributes={Property(name="label", type=StringType)})
    trip_seat = BinaryAssociation(name="trip_seat", ends={
        Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*"))})
    reservation = AssociationClass(name="Reservation", attributes={Property(name="price", type=FloatType)},
                                   association=trip_seat)
    trip_tag = BinaryAssociation(name="trip_tag", ends={
        Property(name="tagged_trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*"))})
    customer = Class(name="Customer", attributes={Property(name="name", type=StringType)})
    purchase = Class(name="Purchase", attributes={Property(name="total", type=FloatType)})
    vip = Class(name="Vip", attributes={Property(name="tier", type=IntegerType)})
    places = BinaryAssociation(name="places", ends={
        Property(name="buyer", type=customer, multiplicity=Multiplicity(1, 1)),
        Property(name="purchases", type=purchase, multiplicity=Multiplicity(0, "*"))})
    archive = Class(name="Archive", attributes={Property(name="label", type=StringType)}, is_read_only=True)
    return DomainModel(name="Shop", types={trip, seat, tag, reservation, customer, purchase, vip, archive},
                       associations={trip_seat, trip_tag, places},
                       generalizations={Generalization(general=customer, specific=vip)})


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    model = _shop_model()
    ws = tmp_path_factory.mktemp("shop_ws")
    _generate(model, ws / "backend")
    return ws, [c.name for c in model.get_classes()]


def test_interleaved_relationships_are_split_per_owner(shop):
    ws, names = shop
    smap = _only(ws, names)
    assert not smap.warnings and all(not e.missing for e in smap.entities.values())
    relationship_line = re.compile(r"^(\w+)\.(\w+)\s*(?::[^=]+)?=\s*relationship\(")
    roles: dict[str, set[str]] = {}
    for anchor in smap.all_anchors():
        if ":orm_relationships:" not in anchor.id:
            continue
        lines = (ws / anchor.file).read_text(encoding="utf-8").splitlines()
        owner = anchor.label.split("relationships of ", 1)[1].split(":", 1)[0]
        for line in lines[anchor.start_line - 1:anchor.end_line]:
            m = relationship_line.match(line)
            if m:
                assert m.group(1) == owner, (anchor.id, line)          # no other entity's lines inside
                roles.setdefault(owner, set()).add(m.group(2))
    # Reservation's lines interleave with Trip's and Seat's: every run is still covered.
    assert roles["Trip"] == {"tags", "reservations", "seats"}
    assert roles["Seat"] == {"reservations", "trips"}
    assert roles["Reservation"] == {"seats", "trips"}
    ids = [a.id for a in smap.all_anchors()]
    assert "fastapi:orm_relationships:trip" in ids
    assert any(i.startswith("fastapi:orm_relationships:trip:") for i in ids)   # a later run, named by its role
    assert not [i for i in ids if re.search(r":\d+$", i)]
    _assert_quotes_valid(ws, smap.all_anchors())


def test_schema_hints_name_the_handlers_that_use_them(shop):
    ws, names = shop
    anchors = _anchors(_only(ws, names))
    archive = anchors["fastapi:schema:archive:ArchiveCreate"].hint            # read-only: no PUT handler
    assert "request body of create_archive" in archive and "update_archive" not in archive
    assert "request body of" in anchors["fastapi:schema:seat:SeatCreate"].hint
    assert "update_seat" in anchors["fastapi:schema:seat:SeatCreate"].hint
    link = anchors["fastapi:schema:reservation:ReservationLinkCreate"].hint     # only a nested payload
    assert "nested in" in link and "TripCreate.seats" in link and "request body" not in link
    assert not [a for a in anchors.values() if "POST and PUT" in a.hint]


def test_inherited_link_routes_are_links_not_custom_routes(shop):
    ws, names = shop
    smap = _only(ws, names)
    vip = smap.entities["Vip"].summary
    assert vip[0] == "extends Customer"
    assert any(part.startswith("links purchases") for part in vip), vip
    assert not [part for part in vip if part.startswith("custom")]


# ---------------------------------------------------------------------------
# The overview budget: entity lines stay short so they survive
# ---------------------------------------------------------------------------

_BIG_NAMES = ("Customer", "Supplier", "Employee", "Purchase", "OrderLine", "Product", "Category", "Invoice",
              "Payment", "Shipment", "Warehouse", "Address", "Review", "Coupon", "Cart")


def _big_model() -> DomainModel:
    """15 classes with two body-less methods each (HTTP 501 stubs), an abstract base and links."""
    status = Enumeration(name="Status", literals={EnumerationLiteral(name="OPEN"), EnumerationLiteral(name="DONE")})
    party = Class(name="Party", attributes={Property(name="name", type=StringType)}, is_abstract=True)
    by_name = {}
    for name in _BIG_NAMES:
        cls = Class(name=name, attributes={Property(name=f"{name.lower()}_code", type=StringType),
                                           Property(name="status", type=status),
                                           Property(name="amount", type=FloatType)})
        cls.add_method(Method(name="recompute", parameters=[Parameter(name="factor", type=FloatType)],
                              type=FloatType))
        cls.add_method(Method(name="summary", parameters=[], type=StringType))
        by_name[name] = cls
    associations = set()
    for n, (a, b, many) in enumerate([("Purchase", "OrderLine", False), ("Product", "Category", True),
                                      ("Customer", "Purchase", False), ("Invoice", "Payment", False),
                                      ("Cart", "Product", True), ("Product", "Review", False)]):
        one = Multiplicity(0, "*") if many else Multiplicity(1, 1)
        associations.add(BinaryAssociation(name=f"link{n}", ends={
            Property(name=f"{a.lower()}_{n}", type=by_name[a], multiplicity=one),
            Property(name=f"{b.lower()}s_{n}", type=by_name[b], multiplicity=Multiplicity(0, "*"))}))
    return DomainModel(name="Shop", types={party, status, *by_name.values()}, associations=associations,
                       generalizations={Generalization(general=party, specific=by_name["Customer"]),
                                        Generalization(general=by_name["Customer"], specific=by_name["Employee"])})


def test_large_model_keeps_its_entity_lines_in_the_overview(tmp_path):
    model = _big_model()
    _generate(model, tmp_path / "backend")
    names = [c.name for c in model.get_classes()]
    smap = _only(tmp_path, names)
    overview = render_overview(WorkspaceMap(maps=[smap]))
    assert len(overview) <= 4000
    assert "Per entity:" in overview
    for name in names:
        line = next(line for line in overview.splitlines() if line.startswith(f"- {name}: "))
        # Conventional router paths are stated once by the recipes, not on every entity line.
        assert "backend/routers/" not in line and len(line) < 160, line
    assert "- Payment: 501 stubs: recompute, summary" in overview
    # Inherited link routes are links, not "custom"; the inherited bulk-delete guard is noted before them.
    assert "- Employee: extends Customer; bulk delete refused (409); links " in overview
    assert "Where new code goes:" in overview and "`fastapi:insert_static:<entity>`" in overview


# ---------------------------------------------------------------------------
# Generated routes that refuse every payload
# ---------------------------------------------------------------------------

def test_bulk_routes_that_refuse_every_payload_are_not_claimed_as_working(generated):
    """Book's bulk create (required relationships) answers 422 and Author's and Library's bulk
    delete answer 409 for any non-empty payload: the map must not list them as working routes."""
    smap = _only(generated)
    anchors = _anchors(smap)
    assert anchors["fastapi:route:book:bulk_create_book"].status == "refuses every payload: HTTP 422"
    assert "`items`" in anchors["fastapi:route:book:bulk_create_book"].hint
    assert anchors["fastapi:route:author:bulk_delete_author"].status == "refuses every payload: HTTP 409"
    assert anchors["fastapi:route:library:bulk_delete_library"].status == "refuses every payload: HTTP 409"
    # Served bulk routes, including a guard that refuses only items carrying relationships.
    for fn in ("book:bulk_delete_book", "author:bulk_create_author", "person:bulk_create_person",
               "person:bulk_delete_person"):
        assert anchors[f"fastapi:route:{fn}"].status is None, fn
    assert "bulk create refused (422)" in smap.entities["Book"].summary
    assert "bulk delete refused (409)" in smap.entities["Author"].summary
    assert not [p for p in smap.entities["Person"].summary if "refused" in p]
    # The shared route sentence outlives the entity lines in the overview: it names the exceptions.
    recipes = {r.intent: " ".join(r.steps) for r in smap.recipes}
    assert recipes["Add an endpoint for an entity"].startswith(
        "Each entity has CRUD, count, paginated, search, bulk create/delete (unless noted; refusing every "
        "payload: bulk create for Book; bulk delete for Author, Library). ")
    assert "(+ bulk_create_<entity> unless refused)" in recipes["Add a field"]
    overview = render_overview(WorkspaceMap(maps=[smap]))
    assert "- Book: bulk create refused (422); links writtenBy" in overview


_GUARDED_ROUTER = """from fastapi import APIRouter, HTTPException, status
from models import {cls}

router = APIRouter()
log = print


@router.get("/{low}/")
def get_all_{low}():
    return []


@router.post("/{low}/")
def create_{low}(data: dict):
    db_item = {cls}({field}=data["{field}"])
    return db_item


@router.post("/{low}/bulk/")
def bulk_create_{low}(items: list):
    \"\"\"Create many {cls} rows.\"\"\"
{create_guard}
    for data in items:
        db_item = {cls}({field}=data["{field}"])
    return {{}}


@router.delete("/{low}/bulk/")
def bulk_delete_{low}(ids: list):
{delete_guard}
    return {{}}


@router.get("/{low}/{{item_id}}/")
def get_{low}(item_id: int):
    return {{}}


@router.put("/{low}/{{item_id}}/")
def update_{low}(item_id: int, data: dict):
    return {{}}


@router.delete("/{low}/{{item_id}}/")
def delete_{low}(item_id: int):
    return {{}}
"""


def test_refused_routes_in_a_handwritten_app(tmp_path):
    root = tmp_path / "api"
    (root / "routers").mkdir(parents=True)
    (root / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from routers import book, shelf\n\n"
        "app = FastAPI()\n"
        "app.include_router(book.router)\n"
        "app.include_router(shelf.router)\n", encoding="utf-8")
    (root / "models.py").write_text(
        "from sqlalchemy import Column, Integer, String\n"
        "from sqlalchemy.orm import declarative_base\n\n"
        "Base = declarative_base()\n\n\n"
        "class Book(Base):\n"
        '    __tablename__ = "book"\n'
        "    id = Column(Integer, primary_key=True)\n"
        "    title = Column(String)\n\n\n"
        "class Shelf(Base):\n"
        '    __tablename__ = "shelf"\n'
        "    id = Column(Integer, primary_key=True)\n"
        "    label = Column(String)\n", encoding="utf-8")
    (root / "routers" / "book.py").write_text(_GUARDED_ROUTER.format(
        cls="Book", low="book", field="title",
        create_guard=("    if len(items) > 0:\n"
                      '        log("refused")\n'
                      '        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "use POST /book/")'),
        delete_guard='    if ids:\n        raise HTTPException(status_code=409, detail="one at a time")'),
        encoding="utf-8")
    (root / "routers" / "shelf.py").write_text(_GUARDED_ROUTER.format(
        cls="Shelf", low="shelf", field="label",
        create_guard='    if items:\n        raise HTTPException(status_code=422, detail="no")',
        # Refuses only the EMPTY payload: bulk delete works.
        delete_guard='    if not ids:\n        raise HTTPException(status_code=400, detail="empty")'),
        encoding="utf-8")
    smap = _only(tmp_path, ["Book", "Shelf"])
    anchors = _anchors(smap)
    assert anchors["fastapi:route:book:bulk_create_book"].status == "refuses every payload: HTTP 422"
    assert anchors["fastapi:route:book:bulk_delete_book"].status == "refuses every payload: HTTP 409"
    assert anchors["fastapi:route:shelf:bulk_create_shelf"].status == "refuses every payload: HTTP 422"
    assert anchors["fastapi:route:shelf:bulk_delete_shelf"].status is None
    assert smap.entities["Book"].summary == ["bulk create/delete refused (422/409)"]
    assert smap.entities["Shelf"].summary == ["bulk create refused (422)"]
    # No reachable bulk constructor anywhere: neither an anchor nor a recipe step points at one.
    assert "fastapi:ctor:book:create_book" in anchors
    assert not [i for i in anchors if ":ctor:" in i and ":bulk_create_" in i]
    recipes = {r.intent: " ".join(r.steps) for r in smap.recipes}
    assert recipes["Add an endpoint for an entity"].startswith(
        "Each entity has CRUD, bulk create/delete (unless noted; refusing every payload: "
        "bulk create for Book, Shelf; bulk delete for Book). ")
    assert "bulk_create" not in recipes["Add a field"]
    _assert_quotes_valid(tmp_path, smap.all_anchors())


@pytest.mark.parametrize("body, expected", [
    ('    """Doc."""\n    if items:\n        raise HTTPException(status_code=422, detail="x")\n    return 1\n',
     ("422", "items")),
    ("    if len(items):\n        raise HTTPException(409)\n", ("409", "items")),
    ("    if len(items) != 0:\n        raise fastapi.HTTPException(status_code=status.HTTP_409_CONFLICT)\n",
     ("409", "items")),
    ("    if items:\n        raise HTTPException(status_code=code)\n", ("error", "items")),
    # Not "every non-empty payload":
    ("    if not items:\n        raise HTTPException(status_code=400)\n", None),
    ("    if any(item.links for item in items):\n        raise HTTPException(status_code=422)\n", None),
    ("    total = 0\n    if items:\n        raise HTTPException(status_code=422)\n", None),
    ("    if items:\n        raise HTTPException(status_code=422)\n    else:\n        pass\n", None),
    ("    if items:\n        return []\n        raise HTTPException(status_code=422)\n", None),
    ("    if other:\n        raise HTTPException(status_code=422)\n", None),
    ("    if items:\n        raise ValueError('x')\n", None),
])
def test_refusal_guard_shapes(body, expected):
    fn = ast.parse(f"def handler(items: list, database=None):\n{body}").body[0]
    assert fastapi_map._refusal(fn) == expected


def test_refusal_wording_helpers():
    assert fastapi_map._refusal_note({"bulk delete": "409", "bulk create": "422"}) == \
        "bulk create/delete refused (422/409)"
    assert fastapi_map._refusal_note({"bulk delete": "409"}) == "bulk delete refused (409)"
    assert fastapi_map._merge_bulk({"bulk create": ["Book"], "bulk delete": ["Book"]}) == \
        {"bulk create/delete": ["Book"]}
    split = {"bulk create": ["Book"], "bulk delete": ["Author", "Book"]}
    assert fastapi_map._merge_bulk(split) == split


# ---------------------------------------------------------------------------
# Regex safety: no catastrophic backtracking on long adversarial lines
# ---------------------------------------------------------------------------

def test_docstring_and_tag_parsing_stay_linear_on_pathological_lines():
    """``\\s*:\\s*([...]+?)\\s*$`` (a lazy group of spaces between two ``\\s*``) took minutes on a
    20k line of spaces, and ``re.sub(r"\\s+(Relationships|Methods)$")`` retried ``\\s+`` from
    every position of a whitespace run; both now run in linear time."""
    doc = "Parameters:\n    - percent: float\n    - tags: list[str] | None\n\nReturns: the price"
    assert fastapi_map._declared_params(doc) == [("percent", "float"), ("tags", "list[str] | None")]
    assert fastapi_map._declared_params("Parameters:\n    - percent:   \n") == []
    assert fastapi_map._tag_entity("Book Relationships") == "Book"
    assert fastapi_map._tag_entity(" Book \t Methods ") == "Book"
    assert fastapi_map._tag_entity("Methods") == "Methods"
    assert fastapi_map._tag_entity("BookMethods") == "BookMethods"
    n = 20000
    for line in ("- a:" + " " * n + "!", "- a: x" + " x" * (n // 2) + "!", "-" + " " * n + "a:" + "|" * n + "\x00"):
        start = time.perf_counter()
        fastapi_map._declared_params("Parameters:\n" + line)
        assert time.perf_counter() - start < 0.2, line[:12]
    for tag in ("a" + " " * n + "!", " \n" * (n // 2) + "x", "x" + "\t" * n + "Methods!"):
        start = time.perf_counter()
        fastapi_map._tag_entity(tag)
        assert time.perf_counter() - start < 0.2, tag[:12]


# ---------------------------------------------------------------------------
# End to end through the package entry points
# ---------------------------------------------------------------------------

@pytest.fixture
def sibling_extractors(monkeypatch):
    """Let build_workspace_map run even before the other families' extractors exist."""
    import besser.spec_driven_agent.planning.scaffold_map as package

    for name in ("django", "react", "generic"):
        full = f"{package.__name__}.{name}"
        try:
            importlib.import_module(full)
        except ImportError:
            stub = types.ModuleType(full)
            stub.extract = lambda index, class_names, **kwargs: []
            monkeypatch.setitem(sys.modules, full, stub)
            monkeypatch.setattr(package, name, stub, raising=False)


def test_end_to_end_build_overview_and_locate(generated, library_model, sibling_extractors):
    from besser.spec_driven_agent.planning.scaffold_map import build_workspace_map, locate, scaffold_overview

    ws = build_workspace_map(str(generated), library_model)
    fastapi_maps = [m for m in ws.maps if m.family == "fastapi"]
    assert len(fastapi_maps) == 1 and fastapi_maps[0].root == "backend"
    assert not ws.broken

    overview = scaffold_overview(str(generated), library_model)
    assert "FastAPI backend (modular routers)" in overview
    assert "fastapi:insert_static:<entity>" in overview
    assert len(overview) <= 4000
    assert not re.search(r"\bline \d+|:\d+-\d+", overview.replace("read_file offset", ""))

    answer = locate(str(generated), entity="Book", domain_model=library_model)
    assert "[fastapi:stub:book:applyDiscount]" in answer["locations"]
    stub = _anchors(_only(generated))["fastapi:stub:book:applyDiscount"]
    assert f"quote: {stub.quote}\n" in answer["locations"]          # verbatim, indentation kept
    assert "read_file offset=" in answer["locations"]

    by_query = locate(str(generated), query="fastapi:include_routers_end")
    assert by_query["count"] == 1
    assert "app.include_router(author_methods_router.router)" in by_query["locations"]

    by_path = locate(str(generated), path="backend/routers/book_methods.py")
    assert "execute_book_applyDiscount" in by_path["locations"]

    first = locate(str(generated))
    assert "Book" in first["entities"]
