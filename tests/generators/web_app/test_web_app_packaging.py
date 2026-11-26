"""What a generated web app ships besides its pages.

The output carried leftovers of the project template it grew from (a test
with no test runner, an unused stylesheet, a lint configuration without the
linter), an agent chat widget in apps without an agent, a link to an icon that
does not exist, and no README telling how to run it.
"""
import asyncio
import io
import json
import os
import zipfile

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.gui.dashboard import AgentComponent
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.web_app import WebAppGenerator
from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.generation import generate_from_project


def _models(*extra_elements):
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    domain = DomainModel(name="Catalogue", types={book})
    screen = Screen(name="Home", description="Home", is_main_page=True,
                    view_elements={Text(name="welcome", content="Welcome"), *extra_elements})
    gui = GUIModel(name="CatalogueGUI", package="", versionCode="1", versionName="1.0",
                   modules={Module(name="M", screens={screen})}, description="")
    return domain, gui


def _generate(tmp_path, *extra_elements, **options):
    domain, gui = _models(*extra_elements)
    WebAppGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path), **options).generate()
    return tmp_path


def _read(*parts):
    with open(os.path.join(*map(str, parts)), encoding="utf-8") as handle:
        return handle.read()


def test_the_app_has_a_readme_that_tells_how_to_run_it(tmp_path):
    out = _generate(tmp_path, app_name="Book_Catalogue")
    readme = _read(out, "README.md")

    assert readme.startswith("# Book Catalogue\n")
    assert "docker compose up --build" in readme
    assert "pip install -r requirements.txt" in readme and "python main_api.py" in readme
    assert "npm install" in readme and "npm run dev" in readme
    # The commands match what the generator ships.
    assert "uvicorn.run(app" in _read(out, "backend", "main_api.py")
    assert '"dev": "vite"' in _read(out, "frontend", "package.json")
    assert "agents/" not in readme


def test_the_api_is_titled_after_the_app(tmp_path):
    out = _generate(tmp_path, app_name="Book_Catalogue")

    assert "title='Book Catalogue API'," in _read(out, "backend", "main_api.py")


def test_without_an_app_name_the_readme_is_titled_after_the_model(tmp_path):
    out = _generate(tmp_path)

    assert _read(out, "README.md").startswith("# Catalogue\n")
    assert "title='Catalogue API'," in _read(out, "backend", "main_api.py")
    assert "<title>BESSER App</title>" in _read(out, "frontend", "index.html")


def test_the_browser_tab_shows_the_app_name(tmp_path):
    out = _generate(tmp_path, app_name="Books & <More>")
    index = _read(out, "frontend", "index.html")

    assert "<title>Books &amp; &lt;More&gt;</title>" in index
    assert "favicon.ico" not in index and '<link rel="icon" href="data:," />' in index
    assert _read(out, "frontend", "README.md").startswith("# Books & <More> frontend\n")


def test_no_file_of_an_unused_toolchain_is_shipped(tmp_path):
    out = _generate(tmp_path)
    src = os.path.join(str(out), "frontend", "src")

    for leftover in ("App.test.tsx", "setupTests.ts", "reportWebVitals.ts", "App.css"):
        assert not os.path.exists(os.path.join(src, leftover)), leftover
    assert "reportWebVitals" not in _read(src, "index.tsx")
    package = json.loads(_read(out, "frontend", "package.json"))
    assert "eslintConfig" not in package
    unused = [name for name in package["dependencies"]
              if name.startswith("@testing-library/") or name in ("@types/jest", "web-vitals")]
    assert unused == []
    # Every import of the shipped sources still resolves to a shipped file or a dependency.
    assert "import './index.css';" in _read(src, "index.tsx") and os.path.exists(os.path.join(src, "index.css"))


def test_the_agent_widget_ships_only_with_an_agent_component(tmp_path):
    plain = _generate(tmp_path / "plain")
    widget = os.path.join("frontend", "src", "components", "AgentComponent.tsx")
    image = os.path.join("frontend", "public", "img", "agent_back.png")

    assert not os.path.exists(plain / widget) and not os.path.exists(plain / image)
    assert "VITE_AGENT_URL" not in _read(plain, "frontend", "README.md")

    chat = AgentComponent(name="helper", agent_name="Helper")
    with_agent = _generate(tmp_path / "agent", chat)

    assert os.path.exists(with_agent / widget) and os.path.exists(with_agent / image)
    assert "VITE_AGENT_URL=ws://localhost:8765" in _read(with_agent, "frontend", "README.md")
    assert 'import { AgentComponent } from "../components/AgentComponent";' in _read(
        with_agent, "frontend", "src", "pages", "Home.tsx")


@pytest.fixture
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_the_editor_passes_the_project_name(_isolate_cwd):
    class_model = {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 800, "height": 600},
                   "interactive": {"elements": {}, "relationships": {}}, "relationships": {}, "assessments": {},
                   "elements": {"c1": {"id": "c1", "name": "Book", "type": "Class", "owner": None,
                                       "bounds": {"x": 0, "y": 0, "width": 200, "height": 100},
                                       "attributes": [], "methods": []}}}
    gui_model = {"pages": [{"id": "home", "name": "Home", "frames": [{"component": {
        "type": "wrapper", "components": [{"tagName": "p", "content": "Welcome"}]}}]}], "styles": []}
    project = ProjectInput(
        id="p1", type="Project", name="Reading_List", createdAt="2026-01-01T00:00:00Z",
        currentDiagramType="GUINoCodeDiagram",
        diagrams={
            "ClassDiagram": [{"id": "d1", "title": "Class Diagram", "model": class_model}],
            "GUINoCodeDiagram": [{"id": "d2", "title": "GUI Diagram", "model": gui_model}],
        },
        settings={"generator": "web_app", "config": {}},
    )

    generation = asyncio.run(generate_from_project(project))

    with zipfile.ZipFile(io.BytesIO(generation.output.content.getvalue())) as archive:
        assert "<title>Reading List</title>" in archive.read("frontend/index.html").decode("utf-8")
        assert archive.read("README.md").decode("utf-8").startswith("# Reading List\n")
        assert "title='Reading List API'," in archive.read("backend/main_api.py").decode("utf-8")
