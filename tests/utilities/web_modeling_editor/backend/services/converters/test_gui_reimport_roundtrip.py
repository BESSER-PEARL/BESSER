"""Export B-UML -> re-import must give back the GUI, not an empty project.

Measured on the AI-authored hotel and tasks designs: every page came back
empty. A generic element (an ``<br>``, an empty widget slot) was exported
without its ``display_order``, and sorting it next to its ordered siblings
compared ``None`` with a number; the converter swallowed the error and
returned an empty project. A Basic CRUD project failed too: a table's lookup
column was exported as an inline ``next(...)`` the safe loader refuses.
"""

import os
import tempfile

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import FieldColumn, LookupColumn, Table
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Metadata,
    Multiplicity,
    Property,
    StringType,
)
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)


def _roundtrip(domain_model, gui_model):
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    project_to_code(Project(name="p", models=[domain_model, gui_model], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        result = project_to_json(handle.read())
    entry = result["diagrams"]["GUINoCodeDiagram"]
    return (entry[0] if isinstance(entry, list) else entry)["model"]


def _components(page):
    found = []

    def walk(component):
        found.append(component)
        for child in component.get("components") or []:
            if child.get("type") != "textnode":
                walk(child)

    for child in page["frames"][0]["component"].get("components") or []:
        walk(child)
    return found


def test_a_design_with_a_generic_element_reimports_with_its_content():
    # Shape of the AI hotel design: an ordered heading next to a bare <br>.
    gui_json = {"pages": [{"id": "home", "name": "Home", "frames": [{"component": {
        "type": "wrapper",
        "components": [{"tagName": "section", "attributes": {"id": "hero"}, "components": [
            {"tagName": "h1", "components": [{"type": "textnode", "content": "Welcome"}]},
            {"tagName": "br"},
            {"tagName": "p", "components": [{"type": "textnode", "content": "Book a room"}]},
        ]}],
    }}]}]}
    gui_model = process_gui_diagram(gui_json, {}, None)

    model = _roundtrip(DomainModel(name="Hotel", types={Class(name="Room")}), gui_model)

    assert [page["name"] for page in model["pages"]] == ["Home"]
    components = _components(model["pages"][0])
    tags = [component.get("tagName") for component in components]
    assert tags == ["section", "h1", "br", "p"]


def test_a_table_with_a_lookup_column_reimports():
    name = Property(name="name", type=StringType)
    title = Property(name="title", type=StringType)
    author = Class(name="Author", attributes={name})
    book = Class(name="Book", attributes={title})
    writes = BinaryAssociation(name="writes", ends={
        Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
    })
    domain = DomainModel(name="Library", types={author, book}, associations={writes})
    author_end = next(end for end in writes.ends if end.name == "author")
    table = Table(
        name="books_table",
        label="Books",
        columns=[FieldColumn(label="Title", field=title), LookupColumn(label="Author", path=author_end, field=name)],
        data_binding=DataBinding(domain_concept=book),
        component_id="books-table",
        display_order=0,
    )
    screen = Screen(name="Books", description="Books", view_elements={table}, is_main_page=True)
    gui = GUIModel(name="ui", package="", versionCode="1", versionName="1", description="",
                   modules={Module(name="M", screens={screen})})

    model = _roundtrip(domain, gui)

    tables = [c for c in _components(model["pages"][0]) if c.get("type") == "table"]
    assert len(tables) == 1
    lookup = [c for c in tables[0]["attributes"]["columns"] if c.get("columnType") == "lookup"]
    assert lookup and lookup[0]["lookupPath"] == "author"
