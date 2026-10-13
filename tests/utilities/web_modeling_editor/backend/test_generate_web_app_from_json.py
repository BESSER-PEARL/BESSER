"""The standalone web-app generator has to read the export format the editor writes."""

from besser.utilities.web_modeling_editor.backend.tools.generate_web_app_from_json import (
    _get_diagram,
)


def test_active_diagram_is_taken_from_a_list_of_diagrams():
    """A project keeps a list per diagram type and names the active one.

    Reading the list as if it were the diagram made the tool fail on every current
    export with "'list' object has no attribute 'get'".
    """
    diagrams = {"ClassDiagram": [{"title": "first"}, {"title": "second"}]}

    assert _get_diagram(diagrams, "ClassDiagram", {"ClassDiagram": 1}) == {"title": "second"}
    # No index recorded, or one pointing outside the list: fall back to the first.
    assert _get_diagram(diagrams, "ClassDiagram", {}) == {"title": "first"}
    assert _get_diagram(diagrams, "ClassDiagram", {"ClassDiagram": 7}) == {"title": "first"}


def test_a_single_diagram_per_type_is_still_accepted():
    """Older exports stored the diagram directly instead of a one-element list."""
    diagrams = {"ClassDiagram": {"title": "only"}}

    assert _get_diagram(diagrams, "ClassDiagram", {}) == {"title": "only"}
    assert _get_diagram(diagrams, "ClassDiagram") == {"title": "only"}


def test_missing_and_empty_diagram_types_read_as_absent():
    assert _get_diagram({}, "GUINoCodeDiagram", {}) is None
    assert _get_diagram({"GUINoCodeDiagram": []}, "GUINoCodeDiagram", {}) is None
