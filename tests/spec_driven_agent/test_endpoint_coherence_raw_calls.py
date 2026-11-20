"""A raw fetch or axios call to a route the generated typed client covers is a warning.

The client returns the plain record and throws one ApiError; a hand-written call to
the same route re-derives the wrappers and the error shape and drifts from the
contract on the next model change. A route the client does not cover (one the
agent added) and an app without a generated client are left alone.
"""
import json
import pathlib

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.backend import BackendGenerator
from besser.generators.react import ReactGenerator
from besser.generators.web_app import WebAppGenerator
from besser.spec_driven_agent.validation.endpoint_coherence import collect_endpoint_coherence_issues
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)

TESTS = pathlib.Path(__file__).parents[1]
PAGE = (
    'import axios from "axios";\n'
    "export async function load() {\n"
    '  const rows = await fetch("http://localhost:8000/customer/").then((r) => r.json());\n'
    "  await axios.put(`/customer/${rows[0].id}/`, { name: \"x\" });\n"
    "  await axios.get(`/customer/${rows[0].id}/`);\n"
    '  return fetch("/report/summary/");\n'
    "}\n"
)


def _model():
    return DomainModel(name="Shop", types={Class(name="Customer", attributes={Property(name="name", type=StringType)})})


def _app(tmp_path, client=True):
    model = _model()
    BackendGenerator(model, output_dir=str(tmp_path / "web_app" / "backend")).generate()
    (tmp_path / "web_app" / "backend" / "routers" / "report.py").write_text(
        "from fastapi import APIRouter\nrouter = APIRouter()\n"
        '@router.get("/report/summary/")\ndef summary(): ...\n', encoding="utf-8")
    if client:
        ReactGenerator(model, None, output_dir=str(tmp_path / "web_app" / "frontend")).generate_api_client()
    page = tmp_path / "web_app" / "frontend" / "src" / "pages" / "Customers.tsx"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(PAGE, encoding="utf-8")
    return str(tmp_path)


def test_a_raw_call_the_client_covers_names_the_client_function(tmp_path):
    issues = collect_endpoint_coherence_issues(_app(tmp_path))

    raw = [issue for issue in issues if "generated API client has as" in issue]
    assert len(raw) == 3, issues
    assert "line 3 calls GET /customer/ with fetch" in raw[0] and "customerApi.list(" in raw[0]
    assert "line 4 calls PUT /customer/{param}/ with axios" in raw[1] and "customerApi.update(" in raw[1]
    # An interpolated segment is the record's id, not the literal count/ or search/ route.
    assert "line 5 calls GET /customer/{param}/ with axios" in raw[2] and "customerApi.get(" in raw[2]
    # A route the agent added is not in the client: nothing to say about it.
    assert not any("/report/summary/" in issue for issue in issues)


def test_without_a_generated_client_raw_calls_are_not_reported(tmp_path):
    """A FastAPI backend writes api_contract.json too; its hand-written frontend has no client to use."""
    issues = collect_endpoint_coherence_issues(_app(tmp_path, client=False))

    assert not [issue for issue in issues if "generated API client has as" in issue]


def test_the_generated_web_app_itself_raises_none(tmp_path):
    class_json = json.loads((TESTS / "generators" / "fixtures" / "library_template_class_diagram.json")
                            .read_text(encoding="utf-8"))
    domain = process_class_diagram(class_json)
    gui_json = json.loads((TESTS / "utilities" / "web_modeling_editor" / "backend" / "fixtures"
                           / "library_template_gui_diagram.json").read_text(encoding="utf-8"))
    gui = process_gui_diagram(gui_json["model"], class_json["model"], domain)
    WebAppGenerator(domain, gui, output_dir=str(tmp_path / "web_app")).generate()

    assert collect_endpoint_coherence_issues(str(tmp_path)) == []
