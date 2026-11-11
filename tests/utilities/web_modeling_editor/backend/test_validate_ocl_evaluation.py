"""/validate-diagram checks OCL invariants on object diagrams for syntax only unless
BESSER_ENABLE_OCL_EVALUATION is on, and the evaluator then accepts only value expressions."""

import pytest

from besser.utilities.web_modeling_editor.backend.routers.validation_router import OCL_SYNTAX_ONLY_WARNING
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    _evaluate_value_expression,
)

INVARIANT = "context Book inv NotX: self.title <> 'x'"


def _object_diagram(title, constraint=INVARIANT):
    """One Book object with ``title`` set, against a Book class carrying ``constraint``."""
    return {
        "title": "Books",
        "model": {
            "type": "ObjectDiagram",
            "elements": {
                "o1": {"type": "ObjectName", "name": "b1", "classId": "c1", "className": "Book",
                       "attributes": ["oa1"], "methods": []},
                "oa1": {"type": "ObjectAttribute", "name": f"+ title: str = {title}", "owner": "o1"},
            },
            "relationships": {},
            "referenceDiagramData": {
                "elements": {
                    "c1": {"type": "Class", "name": "Book", "attributes": ["a1"], "methods": []},
                    "a1": {"type": "ClassAttribute", "name": "+ title: str", "owner": "c1"},
                    "k1": {"type": "ClassOCLConstraint", "constraint": constraint},
                },
                "relationships": {},
            },
        },
    }


def _file_creating_title(path):
    """A title whose text, evaluated as Python by the evaluator, would create ``path``."""
    return 'a" + str(open(r"%s", "w").close()) + "' % path


@pytest.fixture
def validate(client):
    def post(payload, url="/besser_api/validate-diagram"):
        response = client.post(url, json=payload)
        assert response.status_code == 200
        return response.json()
    return post


def test_evaluation_is_off_by_default(route_settings):
    assert route_settings.ocl_evaluation_enabled is False


@pytest.mark.parametrize("url", ["/besser_api/validate-diagram", "/besser_api/check-ocl"])
def test_does_not_evaluate_invariants_by_default(validate, route_settings, url):
    data = validate(_object_diagram("x"), url)  # violates NotX if evaluated

    assert data["isValid"] is True
    assert data["invalid_constraints"] == []
    [entry] = data["valid_constraints"]
    assert "Syntax valid; not evaluated against the object diagram: evaluation is disabled" in entry
    assert OCL_SYNTAX_ONLY_WARNING in data["warnings"]


def test_does_not_run_object_values_by_default(validate, route_settings, tmp_path):
    marker = tmp_path / "marker"

    validate(_object_diagram(_file_creating_title(marker)))

    assert not marker.exists()


def test_syntax_errors_are_still_reported(validate, route_settings):
    data = validate(_object_diagram("x", constraint="context Book inv Broken: self.title <> "))

    # As on a class diagram: the parse error is a warning naming the constraint.
    [warning] = data["warnings"]
    assert warning.startswith("Warning: Invalid OCL syntax in 'context Book inv Broken")
    assert "operand on the right is missing" in warning


def test_the_setting_enables_evaluation(validate, route_settings):
    route_settings.ocl_evaluation_enabled = True

    violated = validate(_object_diagram("x"))
    satisfied = validate(_object_diagram("y"))

    [violation] = violated["invalid_constraints"]
    assert "Constraint violation" in violation
    assert "Evaluates to: True" in satisfied["valid_constraints"][0]
    assert OCL_SYNTAX_ONLY_WARNING not in violated["warnings"] + satisfied["warnings"]


def test_enabled_evaluation_does_not_run_object_values(validate, route_settings, tmp_path):
    route_settings.ocl_evaluation_enabled = True
    marker = tmp_path / "marker"

    data = validate(_object_diagram(_file_creating_title(marker)))

    assert not marker.exists()
    assert data["invalid_constraints"] == []
    assert "not evaluated against the object diagram" in data["valid_constraints"][0]


@pytest.mark.parametrize("source, expected", [
    ('"a" != "b" and (3 > 2 or False)', True),
    ("[ \"x1\" ] in [ [ \"x1\" ] ]", True),
    ("not(1 + 2 * 3 == 7)", False),
    ("-4 // 3 <= 10 % 4 ^ 1", True),
])
def test_value_expressions_are_evaluated(source, expected):
    assert _evaluate_value_expression(source) is expected


@pytest.mark.parametrize("source", [
    "len('abc') == 3",
    "().__class__",
    "x == 1",
    "'abc'[0] == 'a'",
    "2 ** 8 > 1",
    "(lambda: 1)()",
])
def test_other_expressions_are_refused(source):
    with pytest.raises(ValueError):
        _evaluate_value_expression(source)
