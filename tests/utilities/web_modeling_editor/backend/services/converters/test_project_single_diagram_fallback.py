"""
Tests for single-diagram fallback in ``project_to_json``.

A plain BUML file that defines just a ``DomainModel`` (or an Agent, StateMachine,
GUI, or NN model) and does not wrap it in a ``Project(...)`` with a
``models=[...]`` declaration should still be importable through the Project
Import flow — the converter wraps it into a one-diagram project rather than
failing with ``No models defined in 'models=[...]'``.
"""

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

    # The class diagram is actually populated with the User and Item classes.
    class_entry = diagrams["ClassDiagram"][0]
    elements = class_entry["model"]["elements"]
    class_names = {el["name"] for el in elements.values() if el.get("type") == "Class"}
    assert class_names == {"User", "Item"}

    # The other diagrams are empty placeholders (no elements / no pages).
    for other_type in (
        "ObjectDiagram", "AgentDiagram", "StateMachineDiagram",
        "QuantumCircuitDiagram", "NNDiagram",
    ):
        other_model = diagrams[other_type][0]["model"]
        assert other_model.get("elements") == {}
        assert other_model.get("relationships") == {}

    gui_model = diagrams["GUINoCodeDiagram"][0]["model"]
    assert gui_model.get("pages") == []


def test_project_to_json_unknown_file_raises():
    with pytest.raises(ValueError, match="No models defined"):
        project_to_json("x = 1\ny = 2\n")


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
    elements = project["diagrams"]["ClassDiagram"][0]["model"]["elements"]
    class_names = {el["name"] for el in elements.values() if el.get("type") == "Class"}
    assert class_names == {"User", "Item"}
