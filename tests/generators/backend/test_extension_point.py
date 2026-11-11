"""``main_api.py`` loads hand-written code from ``extensions.py``, so it never goes into a generated file.

``before_routers(app)`` adds routes that win over the generated ones with the same path;
``setup(app)`` adds the rest after them. Without the file the app is the generated one.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.backend import BackendGenerator

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

EXTENSIONS = textwrap.dedent("""
    from fastapi import APIRouter

    router = APIRouter()


    @router.get("/book/search/")
    def search():
        return {"custom": True}


    def before_routers(app):
        app.include_router(router)


    def setup(app):
        @app.get("/hello")
        def hello():
            return {"hello": "world"}
""")

CLIENT_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    with TestClient(main_api.app) as client:
        print(json.dumps({
            "search": client.get("/book/search/").json(),
            "hello": client.get("/hello").status_code,
            "create": client.post("/book/", json={"title": "Dune"}).status_code,
        }))
""")


def _run(out) -> dict:
    run = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(out),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.fixture
def backend(tmp_path):
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    BackendGenerator(model=DomainModel(name="Shelf", types={book}), output_dir=str(tmp_path)).generate()
    return tmp_path


def test_extensions_override_a_generated_route_and_add_their_own(backend):
    (backend / "extensions.py").write_text(EXTENSIONS, encoding="utf-8")

    assert _run(backend) == {"search": {"custom": True}, "hello": 200, "create": 200}


def test_without_extensions_the_app_is_the_generated_one(backend):
    result = _run(backend)

    assert result["search"] != {"custom": True} and result["hello"] == 404 and result["create"] == 200


def test_the_port_comes_from_the_environment(backend):
    assert 'port=int(os.environ.get("PORT", 8000))' in (backend / "main_api.py").read_text()
