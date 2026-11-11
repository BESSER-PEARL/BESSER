"""Code generated from the editor's bundled Library template must run.

``fixtures/library_template_class_diagram.json`` is the ClassDiagram of the
web editor's "Library (Full Stack)" template
(``packages/webapp/src/main/templates/pattern/project/library_full_stack.json``
in BESSER-Web-Modeling-Editor): Library / Book / Author, a ``Genre`` enum, an
OCL invariant on ``Book.pages``, ``Book.decrease_stock`` and
``Library.cheapest_book_by(author: Author)``.
"""
import ast
import json
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

from besser.generators.backend import BackendGenerator
from besser.generators.django import DjangoGenerator
from besser.generators.supabase import SupabaseGenerator
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "library_template_class_diagram.json"


def _library_model():
    return process_class_diagram(json.loads(FIXTURE.read_text(encoding="utf-8")))


def _assert_every_python_file_parses(root: pathlib.Path):
    files = sorted(root.rglob("*.py"))
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_django_project_parses(tmp_path):
    """The OCL clean() of Book used to end in ``raise ValidationError(errors)``
    with ``def decrease_stock`` glued onto the same line."""
    DjangoGenerator(_library_model(), project_name="library_project", app_name="library_app",
                    output_dir=str(tmp_path)).generate()
    _assert_every_python_file_parses(tmp_path)


CLIENT_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    book = {"title": "Dune", "pages": 412, "price": 9.5, "release": "1965-08-01", "stock": 3}
    with TestClient(main_api.app) as client:
        status = {}
        author = client.post("/author/", json={"name": "Herbert", "birth": "1920-10-08"})
        status["create author"] = author.status_code
        author_id = author.json()["author"]["id"]
        library = client.post("/library/", json={"name": "City", "web_page": "w", "address": "a",
                                                 "telephone": "1"})
        status["create library"] = library.status_code
        library_id = library.json()["library"]["id"]
        links = {"authors": [author_id], "library": [library_id]}
        created = client.post("/book/", json={**book, **links, "genre": GENRE})
        status["create book"] = created.status_code
        book_id = created.json()["book"]["id"]
        status["update book"] = client.put(f"/book/{book_id}/", json={**book, **links, "genre": GENRE}).status_code
        method = client.post(f"/library/{library_id}/methods/cheapest_book_by/",
                             json={"params": {"author": author_id}})
        status["cheapest_book_by"] = method.status_code
        print(json.dumps({"status": status, "method": method.json()}))
""")


def _run_backend(model, out: pathlib.Path, genre):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    BackendGenerator(model, output_dir=str(out)).generate()
    _assert_every_python_file_parses(out)
    run = subprocess.run([sys.executable, "-c", f"GENRE = {genre!r}\n" + CLIENT_SCRIPT], cwd=str(out),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_backend_method_with_class_parameter_runs(tmp_path):
    """routers/library_methods.py called get_author without importing it, so
    cheapest_book_by answered 500 (NameError)."""
    result = _run_backend(_library_model(), tmp_path, "Fantasy")
    failed = {name: code for name, code in result["status"].items() if code >= 300}
    assert not failed, failed
    assert result["method"]["result"] == "Dune"


def test_backend_optional_enum_left_empty(tmp_path):
    """An optional enum left empty arrived as None and the router read
    ``None.name``: create and update answered 500."""
    model = _library_model()
    genre = next(a for a in model.get_class_by_name("Book").attributes if a.name == "genre")
    genre.is_optional = True
    result = _run_backend(model, tmp_path, None)
    failed = {name: code for name, code in result["status"].items() if code >= 300}
    assert not failed, failed


def test_supabase_references_only_existing_id_columns(tmp_path):
    """No Library class sets is_id, so the tables had no ``id`` column while
    every junction column ``REFERENCES <table>(id)``."""
    SupabaseGenerator(_library_model(), output_dir=str(tmp_path)).generate()
    sql = next(tmp_path.glob("*.sql")).read_text(encoding="utf-8")
    tables = dict(re.findall(r'CREATE TABLE IF NOT EXISTS public\."(\w+)" \((.*?)\n\);', sql, re.S))
    referenced = set(re.findall(r'REFERENCES public\."(\w+)"\(id\)', sql))
    assert referenced
    for table in referenced:
        assert '"id" UUID PRIMARY KEY' in tables[table], table
