"""An editor project with several diagrams of a type: the current one is read unless ``--use`` names another."""

import copy

import pytest


def _choice_notes(output: dict) -> list[str]:
    """The notes about which model was used; generate also notes the record it writes."""
    return [note for note in output["notes"] if "--use" in note]


def _class_diagram(diagram_id: str, title: str, class_name: str) -> dict:
    return {
        "id": diagram_id,
        "title": title,
        "model": {
            "type": "ClassDiagram",
            "elements": {
                f"class-{diagram_id}": {
                    "type": "Class", "name": class_name, "attributes": [f"attr-{diagram_id}"], "methods": [],
                },
                f"attr-{diagram_id}": {"type": "Attribute", "name": "+ name: str"},
            },
            "relationships": {},
        },
    }


EMPTY_CLASS_DIAGRAM = {
    "id": "empty", "title": "Empty", "model": {"type": "ClassDiagram", "elements": {}, "relationships": {}},
}

# Two class diagrams; the editor had the second one (Shop) open.
TWO_CLASS_DIAGRAMS = {
    "project": {
        "id": "project-1",
        "type": "Project",
        "name": "store",
        "createdAt": "2026-11-10T00:00:00Z",
        "currentDiagramType": "ClassDiagram",
        "currentDiagramIndices": {"ClassDiagram": 1},
        "diagrams": {"ClassDiagram": [
            _class_diagram("d1", "Library", "Book"),
            _class_diagram("d2", "Shop", "Product"),
        ]},
    },
}


def _project(indices=None, diagrams=None) -> dict:
    project = copy.deepcopy(TWO_CLASS_DIAGRAMS)
    if indices is None:
        del project["project"]["currentDiagramIndices"]
    else:
        project["project"]["currentDiagramIndices"] = indices
    if diagrams is not None:
        project["project"]["diagrams"]["ClassDiagram"] = diagrams
    return project


def _generated_classes(tmp_path, output: str) -> str:
    return (tmp_path / output / "classes.py").read_text(encoding="utf-8")


def test_generate_reads_the_current_diagram_and_says_so(besser, write, tmp_path):
    project = write("store.json", TWO_CLASS_DIAGRAMS)

    run = besser("generate", "python", project, "-o", "out")

    assert run.returncode == 0, run.stderr
    assert ("note: store.json holds 2 domain models (Library, Shop); using Shop, the project's current "
            "ClassDiagram (--use NAME picks another)") in run.stderr
    assert "class Product" in _generated_classes(tmp_path, "out")


def test_generate_json_records_the_model_used(besser, write):
    project = write("store.json", TWO_CLASS_DIAGRAMS)

    output = besser("generate", "pydantic", project, "-o", "out", "--json").json

    assert output["ok"] is True
    assert output["used"] == {"domain": "Shop"}
    assert len(_choice_notes(output)) == 1 and "using Shop" in _choice_notes(output)[0]


def test_use_overrides_the_current_diagram(besser, write, tmp_path):
    project = write("store.json", TWO_CLASS_DIAGRAMS)

    run = besser("generate", "python", project, "-o", "out", "--use", "Library", "--json")

    assert run.returncode == 0, run.stderr
    assert run.json["used"] == {"domain": "Library"}
    assert _choice_notes(run.json) == []
    assert "class Book" in _generated_classes(tmp_path, "out")


def test_project_without_a_marker_asks_for_use(besser, write):
    project = write("store.json", _project(indices=None))

    run = besser("generate", "python", project, "-o", "out")

    assert run.returncode == 2
    assert ("error: store.json holds 2 domain models (Library, Shop) and the 'python' generator reads one: "
            "name it with --use NAME") in run.stderr
    assert "note:" not in run.stderr


@pytest.mark.parametrize(
    ("indices", "diagrams", "reason"),
    [
        ({"ClassDiagram": 5}, None, "its currentDiagramIndices.ClassDiagram is 5 and it holds 2 ClassDiagram"),
        (
            {"ClassDiagram": 2},
            [_class_diagram("d1", "Library", "Book"), _class_diagram("d2", "Shop", "Product"), EMPTY_CLASS_DIAGRAM],
            "its currentDiagramIndices.ClassDiagram is 2, an empty diagram ('Empty')",
        ),
    ],
    ids=["out-of-range", "empty-diagram"],
)
def test_a_marker_that_names_no_model_is_reported(besser, write, indices, diagrams, reason):
    project = write("store.json", _project(indices=indices, diagrams=diagrams))

    run = besser("generate", "python", project, "-o", "out")
    info = besser("info", project)

    assert run.returncode == 2
    assert "holds 2 domain models (Library, Shop)" in run.stderr
    assert f"the project names no current one: {reason}" in run.stderr
    assert info.returncode == 0
    assert "current in the editor" not in info.stdout
    assert reason in info.stderr


def test_a_marker_for_a_single_diagram_adds_no_note(besser, write):
    project = write("store.json", _project(
        indices={"ClassDiagram": 0}, diagrams=[_class_diagram("d1", "Library", "Book")],
    ))

    output = besser("generate", "python", project, "-o", "out", "--json").json

    assert output["ok"] is True
    assert _choice_notes(output) == []
    assert output["used"] == {"domain": "Library"}


def test_same_titles_are_told_apart_by_the_marker(besser, write, tmp_path):
    project = write("store.json", _project(
        indices={"ClassDiagram": 1},
        diagrams=[_class_diagram("d1", "Model", "Book"), _class_diagram("d2", "Model", "Product")],
    ))

    run = besser("generate", "python", project, "-o", "out")

    assert run.returncode == 0, run.stderr
    assert "class Product" in _generated_classes(tmp_path, "out")


def test_info_marks_the_current_model(besser, write):
    project = write("store.json", TWO_CLASS_DIAGRAMS)

    text = besser("info", project)
    output = besser("info", project, "--json").json

    assert "  Shop (DomainModel) (current in the editor): 1 class" in text.stdout
    assert "  Library (DomainModel): 1 class" in text.stdout
    assert {model["name"]: model["current"] for model in output["models"]} == {"Library": False, "Shop": True}
