"""Nested creation from a WebApp table's "Add" dialog (issue #383).

Library 1 <>-- * Book is a composition: every Book needs its Library, so an
existing Book cannot simply be picked when a Library is created (it already
belongs to another one). The generated create form now lets the user fill in
new Books alongside the Library. The frontend creates the Library, then each
Book through ``POST /book/`` with ``library`` set to the new Library's key,
and deletes what it created if one of them is refused.

The tests check the generated wiring and, on the generated backend, the
contract that flow relies on: the create responses carry the new keys, a
part is created against its parent, and the rollback order (parts, then the
parent) is accepted.
"""
import asyncio
import importlib
import json
import os
import sys

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import Table
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.web_app import WebAppGenerator


def _end(name, cls, low, high, composite=False):
    return Property(name=name, type=cls, multiplicity=Multiplicity(low, high), is_composite=composite)


def _library_model():
    library = Class(name="Library", attributes={Property(name="name", type=StringType)})
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType),
        Property(name="pages", type=IntegerType),
    })
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    shelf = Class(name="Shelf", attributes={Property(name="label", type=StringType)})
    holds = BinaryAssociation(name="Library_Book", ends={
        _end("library", library, 1, 1, composite=True), _end("books", book, 0, "*"),
    })
    writes = BinaryAssociation(name="Book_Author", ends={
        _end("books", book, 0, "*"), _end("authors", author, 1, "*"),
    })
    # A plain 1:N whose children may exist without a parent: picking is enough
    shelves = BinaryAssociation(name="Library_Shelf", ends={
        _end("library", library, 0, 1), _end("shelves", shelf, 0, "*"),
    })
    return DomainModel(name="Lib", types={library, book, author, shelf},
                       associations={holds, writes, shelves})


def _gui(domain):
    screens = set()
    for index, cls in enumerate(sorted(domain.get_classes(), key=lambda c: c.name)):
        table = Table(name=f"{cls.name}_table", data_binding=DataBinding(domain_concept=cls),
                      component_id=f"t{cls.name}")
        screens.add(Screen(name=cls.name, description=cls.name, view_elements={table},
                           is_main_page=index == 0))
    return GUIModel(name="G", package="", versionCode="1", versionName="1", description="",
                    modules={Module(name="M", screens=screens)})


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("web_app_nested")
    domain = _library_model()
    WebAppGenerator(domain, _gui(domain), output_dir=str(output_dir)).generate()
    return output_dir


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return handle.read()


def _form_columns(generated, page):
    source = _read(str(generated), "frontend", "src", "pages", f"{page}.tsx")
    options = source[source.index("options={") + len("options={"):]
    return {column.get("path") or column["field"]: column
            for column in json.JSONDecoder().raw_decode(options)[0]["formColumns"]}


def test_composition_parts_can_be_created_with_the_whole(generated):
    books = _form_columns(generated, "Library")["books"]
    nested = books["nested_create"]

    assert nested["entity"] == "Book"
    assert nested["endpoint"] == "/book/"
    assert nested["back_reference"] == "library"
    assert nested["composition"] is True
    assert (nested["parent_response_key"], nested["parent_key_field"]) == ("library", "id")
    assert (nested["response_key"], nested["key_field"]) == ("book", "id")
    fields = {column.get("path") or column["field"]: column for column in nested["form_columns"]}
    # The part's own form, minus the back-reference that the parent fills in
    assert set(fields) == {"title", "pages", "authors"}
    assert fields["authors"]["column_type"] == "lookup" and fields["authors"]["required"] is True
    # No nesting inside a nested form
    assert "nested_create" not in fields["authors"]


def test_no_nested_creation_where_existing_records_can_be_picked(generated):
    library = _form_columns(generated, "Library")
    assert "nested_create" not in library["shelves"]  # optional parent: pick existing shelves
    book = _form_columns(generated, "Book")
    assert "nested_create" not in book["authors"]  # N:M: an author is not owned by one book
    author = _form_columns(generated, "Author")
    assert "nested_create" not in author["books"]


def test_table_component_wires_the_nested_create_flow(generated):
    table_dir = (str(generated), "frontend", "src", "components", "table")
    component = _read(*table_dir, "TableComponent.tsx")
    section = _read(*table_dir, "NestedCreateSection.tsx")

    assert 'from "./NestedCreateSection"' in component
    assert "normalizeNestedCreate((col as any).nested_create" in component
    assert "<NestedCreateSection" in component
    # Parent first, then its parts, undone on failure
    submit = component[component.index("createResponse = await axios.post(url, processedValues)"):]
    submit = submit[:submit.index("} else if (modalMode === 'edit') {")]
    assert submit.index("entityFromResponse(createResponse?.data") < submit.index("createNestedRecords(")
    assert "rollbackCreatedRecords(created)" in submit
    # A required relationship is satisfied by new records as well as picked ones
    assert "nestedDrafts[col.field]?.length" in component
    assert "missingNestedFields(col.nestedCreate" in component

    assert "payload[meta.backReference] = parentKey" in section
    assert "await axios.delete(record.url)" in section
    assert "[...created].reverse()" in section


GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib", "routers",
)


@pytest.fixture(scope="module")
def app(generated):
    backend = generated / "backend"
    saved_modules = {name: sys.modules.pop(name) for name in list(sys.modules)
                     if name in GENERATED_MODULES or name.startswith("routers.")}
    saved_cwd = os.getcwd()
    saved_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{(backend / 'nested.db').as_posix()}"
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        importlib.invalidate_caches()
        yield importlib.import_module("main_api").app
    finally:
        os.chdir(saved_cwd)
        sys.path.remove(str(backend))
        for name in list(sys.modules):
            if name in GENERATED_MODULES or name.startswith("routers."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = saved_database_url


def _request(app, method, url, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send())


def _entity(data, key):
    """entityFromResponse of NestedCreateSection.tsx."""
    wrapped = data.get(key) if isinstance(data, dict) else None
    return wrapped if isinstance(wrapped, dict) else data


def test_backend_accepts_the_nested_create_sequence(app):
    author = _request(app, "POST", "/author/", json={"name": "Ada"})
    assert author.status_code == 200, author.text
    author_id = _entity(author.json(), "author")["id"]

    # 1. the parent, 2. each part pointing back at it
    library = _request(app, "POST", "/library/", json={"name": "Central"})
    assert library.status_code == 200, library.text
    library_id = _entity(library.json(), "library")["id"]
    book = _request(app, "POST", "/book/", json={
        "title": "Notes", "pages": 12, "authors": [author_id], "library": library_id,
    })
    assert book.status_code == 200, book.text
    book_id = _entity(book.json(), "book")["id"]
    detailed = _request(app, "GET", f"/library/{library_id}/")
    assert detailed.status_code == 200, detailed.text

    # A refused part (no author) triggers the rollback: parts first, then the parent
    refused = _request(app, "POST", "/book/", json={
        "title": "Draft", "pages": 1, "authors": [], "library": library_id,
    })
    assert refused.status_code >= 400
    assert _request(app, "DELETE", f"/library/{library_id}/").status_code == 409  # still holds a book
    assert _request(app, "DELETE", f"/book/{book_id}/").status_code == 200
    assert _request(app, "DELETE", f"/library/{library_id}/").status_code == 200
    assert _request(app, "GET", f"/library/{library_id}/").status_code == 404
