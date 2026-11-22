"""End-to-end: a BAL method that assigns to ``this.<attr>`` works in the generated backend.

The BAL translator renders ``this.status = ...`` as ``await update_paper(...)``.
Method endpoints live in ``routers/paper_methods.py`` while ``update_paper`` lives
in ``routers/paper.py``, and the methods module never imported its own class's
CRUD functions, so every mutating BAL method returned
500 ``name 'update_paper' is not defined``.
"""

import asyncio
import importlib
import os
import sys

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, MethodImplementationType, Property, StringType,
)
from besser.generators.backend.backend_generator import BackendGenerator
from besser.generators.backend.api_generator import cross_router_calls

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from httpx._transports.asgi import ASGITransport  # noqa: E402


def _paper_model() -> DomainModel:
    paper = Class(name="Paper", attributes={
        Property(name="title", type=StringType),
        Property(name="status", type=StringType),
    })
    publish = Method(name="publish", parameters=[], type=None,
                     implementation_type=MethodImplementationType.BAL)
    publish.code = """def publish() -> nothing {
        this.status = "published";
    }"""
    paper.add_method(publish)
    label = Method(name="label", parameters=[], type=StringType,
                   implementation_type=MethodImplementationType.BAL)
    label.code = """def label() -> str {
        return this.title;
    }"""
    paper.add_method(label)
    return DomainModel(name="Papers", types={paper})


GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib", "routers",
)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    generated_backend = tmp_path_factory.mktemp("backend_bal_update")
    BackendGenerator(model=_paper_model(), output_dir=str(generated_backend)).generate()

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


def test_mutating_bal_method_updates_the_record(app):
    created = _request(app, "POST", "/paper/", json={"title": "BESSER", "status": "draft"})
    assert created.status_code == 200, created.text
    paper_id = created.json()["id"]

    response = _request(app, "POST", f"/paper/{paper_id}/methods/publish/", json={})
    assert response.status_code == 200, response.text
    assert _request(app, "GET", f"/paper/{paper_id}/").json()["paper"]["status"] == "published"


def test_read_only_bal_method_still_works(app):
    paper_id = _request(app, "POST", "/paper/", json={"title": "Read me", "status": "draft"}).json()["id"]

    response = _request(app, "POST", f"/paper/{paper_id}/methods/label/", json={})
    assert response.status_code == 200, response.text
    assert "Read me" in response.text


def test_own_class_imports_point_at_the_right_module():
    """CRUD functions come from ``routers.<class>``; method endpoints from ``routers.<class>_methods``,
    except the current class's own, which share the calling module."""
    code = (
        "await update_paper(inst_to_update.id, PaperCreate(x=1), database)\n"
        "(await execute_paper_publish(_paper_object.id, {}, database))\n"
        "(await execute_review_close(r.id, {}, database))\n"
    )
    assert cross_router_calls(code, "Paper", ["Paper", "Review"]) == [
        ("paper", "update_paper"),
        ("review_methods", "execute_review_close"),
    ]
