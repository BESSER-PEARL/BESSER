"""
Tests for single-diagram fallback in ``project_to_json``.

A plain BUML file that defines just a ``DomainModel`` (or an Agent, StateMachine,
GUI, or NN model) and does not wrap it in a ``Project(...)`` with a
``models=[...]`` declaration should still be importable through the Project
Import flow — the converter wraps it into a one-diagram project rather than
failing with ``No models defined in 'models=[...]'``.

The diagram models inside the project envelope follow the v4 wire shape
(``{version: "4.0.0", nodes, edges}``); GUI models keep their bespoke
``{pages}`` shape.
"""

from pathlib import Path

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
    _detect_single_diagram_type,
)


SUPPLEMENT_CLASS_MODEL = '''\
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, BinaryAssociation,
    StringType, IntegerType,
)

user_id = Property(name="id", type=StringType, is_id=True)
user_name = Property(name="name", type=StringType)
user = Class(name="User", attributes={user_id, user_name})

item_id = Property(name="id", type=StringType, is_id=True)
item_qty = Property(name="qty", type=IntegerType)
item = Class(name="Item", attributes={item_id, item_qty})

end_user = Property(name="user", type=user, multiplicity=Multiplicity(1, 1))
end_items = Property(name="items", type=item, multiplicity=Multiplicity(0, "*"))
user_items = BinaryAssociation(name="user_items", ends={end_user, end_items})

supplement_model = DomainModel(
    name="SupplementAssistant",
    types={user, item},
    associations={user_items},
)
'''


def _class_names(model: dict) -> set:
    return {
        node["data"]["name"]
        for node in model["nodes"]
        if node.get("type") == "class"
    }


def test_detects_class_diagram_from_plain_domain_model():
    assert _detect_single_diagram_type(SUPPLEMENT_CLASS_MODEL) == "ClassDiagram"


def test_returns_unknown_when_no_known_keywords_present():
    assert _detect_single_diagram_type("# nothing relevant\nx = 1\n") == ""


def test_project_to_json_wraps_single_class_diagram():
    """A plain DomainModel file (no Project / models=[...]) becomes a 1-diagram project."""
    project = project_to_json(SUPPLEMENT_CLASS_MODEL)

    # Top-level project envelope is well-formed for the frontend importer.
    assert project["type"] == "Project"
    assert project["schemaVersion"] == 3
    assert isinstance(project["id"], str) and project["id"]
    assert project["currentDiagramType"] == "ClassDiagram"

    diagrams = project["diagrams"]
    # All seven diagram types are present so the frontend has a slot per type.
    for diagram_type in (
        "ClassDiagram", "ObjectDiagram", "AgentDiagram",
        "StateMachineDiagram", "GUINoCodeDiagram", "QuantumCircuitDiagram",
        "NNDiagram",
    ):
        assert diagram_type in diagrams
        assert isinstance(diagrams[diagram_type], list)
        assert len(diagrams[diagram_type]) == 1

    # The class diagram is actually populated with the User and Item classes,
    # emitted in the canonical v4 wire shape.
    class_model = diagrams["ClassDiagram"][0]["model"]
    assert class_model["version"] == "4.0.0"
    assert _class_names(class_model) == {"User", "Item"}

    # The other diagrams are empty v4 placeholders (no nodes / no edges).
    for other_type in (
        "ObjectDiagram", "AgentDiagram", "StateMachineDiagram",
        "QuantumCircuitDiagram", "NNDiagram",
    ):
        other_model = diagrams[other_type][0]["model"]
        assert other_model.get("nodes") == []
        assert other_model.get("edges") == []

    gui_model = diagrams["GUINoCodeDiagram"][0]["model"]
    assert gui_model.get("pages") == []


def test_project_to_json_unknown_file_raises():
    with pytest.raises(ValueError, match="No models defined"):
        project_to_json("x = 1\ny = 2\n")


PROJECT_WITHOUT_SECTION_HEADERS = '''\
from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, DomainModel, Multiplicity,
    StringType, IntegerType,
)

book = Class(name="Book")
book_title = Property(name="title", type=StringType)
book.attributes = {book_title}

author = Class(name="Author")
author_name = Property(name="name", type=StringType)
author.attributes = {author_name}

book_author = BinaryAssociation(
    name="book_author",
    ends={
        Property(name="book", type=book, multiplicity=Multiplicity(1, 1)),
        Property(name="author", type=author, multiplicity=Multiplicity(1, 9999)),
    },
)

library_model = DomainModel(
    name="Library",
    types={book, author},
    associations={book_author},
)

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="A small library project.")
project = Project(
    name="Library_Project",
    models=[library_model],
    owner="Tester",
    metadata=metadata,
)
'''


def test_project_to_json_handles_missing_section_headers():
    """Projects with `models=[...]` but no `# STRUCTURAL MODEL #` headers
    must still extract the embedded DomainModel — previously this returned
    a project where every diagram was empty because the section extractor
    found no headers to anchor on."""
    project = project_to_json(PROJECT_WITHOUT_SECTION_HEADERS)

    # Project metadata from the Project(...) wrapper is preserved.
    assert project["name"] == "Library_Project"
    assert project["owner"] == "Tester"
    assert project["description"] == "A small library project."

    class_model = project["diagrams"]["ClassDiagram"][0]["model"]
    assert class_model["version"] == "4.0.0"
    assert _class_names(class_model) == {"Book", "Author"}
    # The BinaryAssociation should round-trip into the v4 edges list.
    assert len(class_model["edges"]) == 1


ECORE_PROJECT = (Path(__file__).parent / "data" / "ecore_algorithme_project.py").read_text(encoding="utf-8")

BOOK_END = 'Property(name="book", type=book, multiplicity=Multiplicity(1, 1)),'
AUTHOR_END = 'Property(name="author", type=author, multiplicity=Multiplicity(1, 9999)),'


def _edit(content, *replacements):
    for original, replacement in replacements:
        assert original in content
        content = content.replace(original, replacement)
    return content


# Composition (author end composite) whose owner end, book, is not navigable.
NON_NAVIGABLE_OWNER = _edit(
    PROJECT_WITHOUT_SECTION_HEADERS,
    (BOOK_END, BOOK_END.replace("1, 1)", "1, 1), is_navigable=False")),
    (AUTHOR_END, AUTHOR_END.replace("1, 9999)", "1, 9999), is_composite=True")),
)

# Plain association with no navigable end: nothing to repair it into.
NO_NAVIGABLE_END = _edit(
    PROJECT_WITHOUT_SECTION_HEADERS,
    (BOOK_END, BOOK_END.replace("1, 1)", "1, 1), is_navigable=False")),
    (AUTHOR_END, AUTHOR_END.replace("1, 9999)", "1, 9999), is_navigable=False")),
)


def _class_diagram(project):
    model = project["diagrams"]["ClassDiagram"][0]["model"]
    names = {el["name"] for el in model["elements"].values() if el.get("type") in ("Class", "AbstractClass")}
    return names, model["relationships"]


def test_an_ecore_converter_project_imports_with_its_compositions_repaired():
    """The file that motivated v8.7.1: it set-unions its types, aliases ``Project``
    and leaves both compositions' owner ends non-navigable."""
    warnings = []

    project = project_to_json(ECORE_PROJECT, warnings)

    names, relationships = _class_diagram(project)
    assert names == {"PortSortie", "Port", "Documentation", "AlgorithmeElement", "Algorithme",
                     "PortEntree", "Ressource"}
    assert sorted(r["type"] for r in relationships.values()) == [
        "ClassBidirectional", "ClassComposition", "ClassComposition",
        "ClassInheritance", "ClassInheritance", "ClassInheritance", "ClassInheritance",
    ]
    assert project["name"] == "algorithme"
    assert len(warnings) == 2
    assert all("must be navigable" in w and "made it navigable" in w for w in warnings)
    assert any("inverse_Algorithme_algorithmeElements" in w for w in warnings)
    assert any("inverse_Ressource_ports" in w for w in warnings)


@pytest.mark.parametrize("content", [
    _edit(PROJECT_WITHOUT_SECTION_HEADERS, ("types={book, author},", "types={book} | {author},")),
    _edit(PROJECT_WITHOUT_SECTION_HEADERS,
          ("import Project\n", "import Project as BUMLProject\n"),
          ("project = Project(", "project = BUMLProject(")),
], ids=["set union", "aliased Project import"])
def test_valid_python_from_other_tools_imports_without_warnings(content):
    warnings = []

    names, relationships = _class_diagram(project_to_json(content, warnings))

    assert names == {"Book", "Author"}
    assert len(relationships) == 1
    assert warnings == []


def test_a_composition_with_a_non_navigable_owner_end_is_repaired_and_reported():
    warnings = []

    names, relationships = _class_diagram(project_to_json(NON_NAVIGABLE_OWNER, warnings))

    assert names == {"Book", "Author"}
    [relationship] = relationships.values()
    assert relationship["type"] == "ClassComposition"
    assert warnings == [
        "Association 'book_author': the non-composite end 'book' (Book) of a composition must be "
        "navigable; made it navigable."
    ]


def test_project_without_section_headers_reports_why_its_model_cannot_be_read():
    with pytest.raises(ValueError, match="at least one end must be navigable"):
        project_to_json(NO_NAVIGABLE_END)


@pytest.mark.parametrize("endpoint", ["get-project-json-model", "get-json-model"])
def test_import_endpoints_return_the_parse_error(client, endpoint):
    response = client.post(
        f"/besser_api/{endpoint}",
        files={"buml_file": ("library.py", NO_NAVIGABLE_END.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 400
    assert "at least one end must be navigable" in response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["get-project-json-model", "get-json-model"])
def test_import_endpoints_return_what_was_repaired(client, endpoint):
    response = client.post(
        f"/besser_api/{endpoint}",
        files={"buml_file": ("library.py", NON_NAVIGABLE_OWNER.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["warnings"] == [{
        "code": "buml-import-repaired",
        "message": "Association 'book_author': the non-composite end 'book' (Book) of a composition must be "
                   "navigable; made it navigable.",
    }]


def test_project_to_json_handles_main_guard():
    """Files with an ``if __name__ == '__main__':`` block should still import.

    The exec sandbox runs the file as if it were imported (not executed as a
    script), so the guarded block is skipped — mirroring normal Python import
    semantics. Previously this raised ``NameError: name '__name__' is not defined``.
    """
    content_with_main_guard = SUPPLEMENT_CLASS_MODEL + (
        '\nif __name__ == "__main__":\n'
        '    raise RuntimeError("main block should not execute during import")\n'
    )

    project = project_to_json(content_with_main_guard)
    assert project["currentDiagramType"] == "ClassDiagram"
    class_model = project["diagrams"]["ClassDiagram"][0]["model"]
    assert _class_names(class_model) == {"User", "Item"}
