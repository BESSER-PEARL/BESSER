"""A page keeps its route through export -> re-import.

The JSON -> B-UML converter uses the route stored on a page; the B-UML -> JSON
converter must write it back, or a re-imported page falls back to the route
derived from its name and links to the old route open the main page.
"""
import os
import tempfile

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import Class, DomainModel, Metadata
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)


def _roundtrip(gui_json):
    gui_model = process_gui_diagram(gui_json, {}, None)
    path = os.path.join(tempfile.mkdtemp(prefix="besser_"), "project.py")
    domain = DomainModel(name="D", types={Class(name="X")})
    project_to_code(Project(name="p", models=[domain, gui_model], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        entry = project_to_json(handle.read())["diagrams"]["GUINoCodeDiagram"]
    return (entry[0] if isinstance(entry, list) else entry)["model"]


def _page(page_id, name, **page):
    return {"id": page_id, "name": name, **page, "frames": [{"component": {"type": "wrapper"}}]}


def test_page_routes_come_back():
    exported = _roundtrip({"pages": [
        _page("page-products", "Products", route_path="/shop"),
        _page("page-reports-a", "Reports"),
        _page("page-reports-b", "Reports"),
        _page("page-orders", "Заказы"),
    ]})

    assert {page["id"]: page.get("route_path") for page in exported["pages"]} == {
        "page-products": "/shop", "page-reports-a": "/reports", "page-reports-b": "/reports-2",
        "page-orders": "/заказы",
    }
    reimported = process_gui_diagram(exported, {}, None)
    assert {screen.component_id: screen.route_path for screen in next(iter(reimported.modules)).screens} == {
        "page-products": "/shop", "page-reports-a": "/reports", "page-reports-b": "/reports-2",
        "page-orders": "/заказы",
    }
