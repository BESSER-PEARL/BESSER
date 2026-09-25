"""router.py.j2 must not teach the LLM a `.value` idiom for enum attributes.

Observed on Spec-Driven Agent runs: the scaffold wrote enum
attributes as ``field=payload.field.value`` at every constructor/setattr call
(write direction, where SQLAlchemy accepts either form). That was the ONLY enum
idiom anywhere in the generated repo, and the model repeatedly copied
it verbatim into hand-authored READ-direction comparisons it had to write for
modeled methods, e.g. ``if booking.physicalStatus != Status.ARRIVED.value:`` --
which is always True, because a loaded SQLAlchemy Enum column holds the member,
never the string, so ``member != member.value`` never matches.

The fix writes ``.name`` instead of ``.value`` at the same 9 sites. Not merely
"equally harmless": this generator always defines enum literals as
``LITERAL = "LITERAL"`` (see sql_alchemy_template.py.j2 / pydantic_classes_template.py.j2),
so `.name` and `.value` are always the same string here, AND SQLAlchemy's
``Enum(PyEnum)`` binds/persists by the member's **name** by default (only by
value if the column type is built with ``values_callable``, which this
generator does not set) -- so `.name` is the semantically correct write, not
just a differently-spelled workaround.

`.value` was also, unexpectedly, load-bearing for a SEPARATE, pre-existing bug:
pydantic_classes.py and sql_alchemy.py each independently emit their own
``class SomeEnum(...)`` for every Enumeration (two distinct Python classes with
the same name, never unified). A bare enum member from the pydantic-generated
class does not compare equal to the sql_alchemy-generated class's members, so
passing it straight into `Enum(SqlAlchemySideClass)`'s bind raises
``LookupError: '...' is not among the defined enum values``. `.value` (and
`.name`) sidestep this because SQLAlchemy matches a plain string by content,
not by class identity. That duplication is a separate defect in the
pydantic_classes/ and sql_alchemy/ generators (out of scope here -- this file
may only touch router.py.j2, router_methods.py.j2 and tests/generators/backend/).
"""
import asyncio
import importlib
import os
import sys

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.generators.backend import BackendGenerator


def test_enum_attribute_writes_use_name_not_value(tmp_path, library_model_with_enum):
    """Fails against the pre-fix template: it writes ``author_data.member.value``
    at all three sites below (create, bulk create, update) and never writes
    ``.name``, so every assertion here fails on the unmodified scaffold."""
    BackendGenerator(model=library_model_with_enum, output_dir=str(tmp_path)).generate()
    src = (tmp_path / "routers" / "author.py").read_text(encoding="utf-8")

    # NOT ".value" not in src -- association-table code legitimately contains
    # "bookauthor_relation.insert().values(...)", where ".value" is a substring
    # of ".values(". Check the enum attribute's own accessor specifically.
    assert "member.value" not in src, (
        "author.py still writes the enum attribute with .value -- the write-side "
        "idiom the LLM copies into always-true/false READ-direction comparisons"
    )
    assert "member=author_data.member.name" in src, "create_author must construct with .name"
    assert "member=item_data.member.name" in src, "bulk_create_author must construct with .name"
    assert "setattr(db_author, 'member', author_data.member.name)" in src, (
        "update_author must setattr with .name"
    )


# --- Live round trip through the real generated REST API ---------------------
# Mirrors the generated_backend/app fixture pattern in test_backend_assoc_class.py:
# import the generated app with importlib, saving/restoring sys.path, sys.modules,
# cwd and DATABASE_URL so this doesn't leak into other tests in the same process.

GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib",
    "routers", "routers.author", "routers.book", "routers.library",
)


@pytest.fixture
def generated_backend(tmp_path_factory, library_model_with_enum):
    # library_model_with_enum is function-scoped (tests/generators/conftest.py),
    # so this and `app` below must be too.
    output_dir = tmp_path_factory.mktemp("backend_enum_roundtrip")
    BackendGenerator(model=library_model_with_enum, output_dir=str(output_dir)).generate()
    return output_dir


@pytest.fixture
def app(generated_backend):
    saved_modules = {
        name: sys.modules.pop(name) for name in GENERATED_MODULES if name in sys.modules
    }
    saved_cwd = os.getcwd()
    saved_database_url = os.environ.get("DATABASE_URL")

    database_path = (generated_backend / "test_enum_roundtrip.db").as_posix()
    os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"
    sys.path.insert(0, str(generated_backend))
    os.chdir(generated_backend)
    try:
        importlib.invalidate_caches()
        main_api = importlib.import_module("main_api")
        yield main_api.app
    finally:
        os.chdir(saved_cwd)
        sys.path.remove(str(generated_backend))
        for name in list(sys.modules):
            if name in GENERATED_MODULES or name.startswith("routers"):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = saved_database_url


def request(app, method, url, **kwargs):
    async def _send():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_send())


def test_enum_field_round_trips_through_create_get_update(app):
    """Does NOT fail against the pre-fix (``.value``) template -- SQLAlchemy
    accepts a plain string on either write path, so this is a non-regression
    safety net for the fix, not a reproduction of the defect. It does fail
    against a naive fix that writes the bare member instead of `.name`/`.value`
    (see the module docstring): that raises a 500 from the duplicate-enum-class
    LookupError on the very first POST below."""
    r = request(app, "POST", "/author/", json={"email": "a@example.com", "member": "STUDENT"})
    assert r.status_code == 200, r.text
    body = r.json()["author"]
    author_id = body["id"]
    assert body["member"] == "STUDENT"

    r = request(app, "GET", f"/author/{author_id}/")
    assert r.status_code == 200, r.text
    assert r.json()["author"]["member"] == "STUDENT"

    r = request(app, "PUT", f"/author/{author_id}/", json={"email": "a@example.com", "member": "ADULT"})
    assert r.status_code == 200, r.text
    put_body = r.json()
    assert put_body.get("member", put_body.get("author", {}).get("member")) == "ADULT"

    r = request(app, "GET", f"/author/{author_id}/")
    assert r.json()["author"]["member"] == "ADULT"

    # The raw stored column, independent of the API's own (de)serialization.
    import sql_alchemy
    from sqlalchemy import create_engine, text
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        raw = conn.execute(
            text("SELECT member FROM author WHERE id = :id"), {"id": author_id}
        ).scalar_one()
    assert raw == "ADULT"


def test_bulk_create_with_enum_field(app):
    r = request(app, "POST", "/author/bulk/", json=[
        {"email": "b@example.com", "member": "CHILD"},
        {"email": "c@example.com", "member": "SENIOR"},
    ])
    assert r.status_code == 200, r.text
    assert r.json()["created_count"] == 2
