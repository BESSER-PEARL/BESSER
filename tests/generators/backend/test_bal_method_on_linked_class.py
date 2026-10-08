"""End-to-end: BAL methods work on classes linked by associations.

``this.<attr> = v`` renders as ``update_<class>(id, <Class>Create(...))``, which
passed the ORM relationship objects (``writer = inst.writer``) where the schema
takes ids, so every mutating method on a linked class returned 500
"Input should be a valid integer". Navigating ``this.writer`` also returned
``get_author``'s response wrapper, not the Author, and ``this.writer.m()`` called
``execute_book_m`` and crashed the type checker.
"""

import asyncio
import importlib
import os
import sys

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Method, MethodImplementationType,
    Multiplicity, Property, StringType,
)
from besser.generators.backend.backend_generator import BackendGenerator

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from httpx._transports.asgi import ASGITransport  # noqa: E402


def _method(owner, name, code, returns=None):
    method = Method(name=name, parameters=[], type=returns,
                    implementation_type=MethodImplementationType.BAL)
    method.code = code
    owner.add_method(method)


def _library_model() -> DomainModel:
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType), Property(name="status", type=StringType)})
    author = Class(name="Author", attributes={
        Property(name="name", type=StringType), Property(name="bio", type=StringType)})
    tag = Class(name="Tag", attributes={Property(name="label", type=StringType)})
    _method(book, "markRead", 'def markRead() -> nothing {\n this.status = "read";\n}')
    _method(book, "renameWriter", 'def renameWriter() -> nothing {\n this.writer.name = "Renamed";\n}')
    _method(book, "writerName", "def writerName() -> str {\n return this.writer.name;\n}", StringType)
    _method(book, "retireWriter", "def retireWriter() -> nothing {\n this.writer.retire();\n}")
    _method(author, "retire", 'def retire() -> nothing {\n this.bio = "retired";\n}')
    writes = BinaryAssociation(name="Writes", ends={
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
        Property(name="writer", type=author, multiplicity=Multiplicity(1, 1))})
    tagged = BinaryAssociation(name="Tagged", ends={
        Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
        Property(name="tagged", type=book, multiplicity=Multiplicity(0, "*"))})
    return DomainModel(name="Library", types={book, author, tag}, associations={writes, tagged})


GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib", "routers",
)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    generated_backend = tmp_path_factory.mktemp("backend_bal_linked")
    BackendGenerator(model=_library_model(), output_dir=str(generated_backend)).generate()

    def _is_generated(name):
        return name in GENERATED_MODULES or name.startswith("routers.")

    saved_modules = {name: sys.modules.pop(name) for name in list(sys.modules) if _is_generated(name)}
    saved_cwd = os.getcwd()
    saved_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{(generated_backend / 'test_api.db').as_posix()}"
    sys.path.insert(0, str(generated_backend))
    os.chdir(generated_backend)
    try:
        importlib.invalidate_caches()
        yield importlib.import_module("main_api").app
    finally:
        os.chdir(saved_cwd)
        sys.path.remove(str(generated_backend))
        for name in list(sys.modules):
            if _is_generated(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = saved_database_url


def _request(app, method, url, **kwargs):
    async def _send():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(_send())


@pytest.fixture()
def linked(app):
    author_id = _request(app, "POST", "/author/", json={"name": "Ada", "bio": "-"}).json()["author"]["id"]
    tag_id = _request(app, "POST", "/tag/", json={"label": "classic"}).json()["tag"]["id"]
    created = _request(app, "POST", "/book/", json={
        "title": "Notes", "status": "new", "writer": author_id, "tags": [tag_id]})
    assert created.status_code == 200, created.text
    return created.json()["book"]["id"], author_id, tag_id


def _call(app, path):
    response = _request(app, "POST", path, json={})
    assert response.status_code == 200, response.text
    return response.json()


def test_mutating_method_keeps_the_links(app, linked):
    book_id, author_id, tag_id = linked
    _call(app, f"/book/{book_id}/methods/markRead/")
    book = _request(app, "GET", f"/book/{book_id}/").json()
    assert book["book"]["status"] == "read"
    assert book["book"]["writer_id"] == author_id
    assert book["tag_ids"] == [tag_id]


def test_mutating_method_on_the_one_side(app, linked):
    book_id, author_id, _ = linked
    _call(app, f"/author/{author_id}/methods/retire/")
    author = _request(app, "GET", f"/author/{author_id}/").json()
    assert author["author"]["bio"] == "retired"
    assert author["books_ids"] == [book_id]


def test_navigation_reads_and_writes_the_linked_record(app, linked):
    book_id, author_id, _ = linked
    assert _call(app, f"/book/{book_id}/methods/writerName/")["result"] == "Ada"
    _call(app, f"/book/{book_id}/methods/renameWriter/")
    assert _request(app, "GET", f"/author/{author_id}/").json()["author"]["name"] == "Renamed"


def test_calls_a_method_of_the_linked_class(app, linked):
    book_id, author_id, _ = linked
    _call(app, f"/book/{book_id}/methods/retireWriter/")
    assert _request(app, "GET", f"/author/{author_id}/").json()["author"]["bio"] == "retired"
