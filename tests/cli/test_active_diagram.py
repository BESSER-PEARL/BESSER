"""An editor project with several diagrams of a type: the current one is read unless ``--use`` names another."""

import copy
import json

import pytest


def _choice_notes(output: dict) -> list[str]:
    """The notes about which model was used; generate also notes the record it writes."""
    return [note for note in output["notes"] if "; using " in note]


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
        ({"ClassDiagram": 5}, None, "its currentDiagramIndices.ClassDiagram is 5 and it holds 2 ClassDiagram diagrams"),
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


def _info_current(besser, project: str) -> dict:
    return {model["name"]: model["current"] for model in besser("info", project, "--json").json["models"]}


def test_a_project_without_a_marker_says_so_in_info_as_generate_does(besser, write):
    project = write("store.json", _project(indices=None))

    info = besser("info", project)
    run = besser("generate", "python", project, "-o", "out")

    assert info.returncode == 0
    reason = "its currentDiagramIndices has no ClassDiagram entry"
    assert (f"note: store.json holds 2 domain models and names no current one: {reason}; "
            "generators need --use NAME\n") in info.stderr
    assert f"(the project names no current one: {reason})" in run.stderr


def test_an_empty_diagram_does_not_shift_the_current_one(besser, write, tmp_path):
    project = write("store.json", _project(
        indices={"ClassDiagram": 2},
        diagrams=[
            EMPTY_CLASS_DIAGRAM, _class_diagram("d1", "Library", "Book"), _class_diagram("d2", "Shop", "Product"),
        ],
    ))

    run = besser("generate", "python", project, "-o", "out")

    assert run.returncode == 0, run.stderr
    assert "using Shop" in run.stderr
    assert "class Product" in _generated_classes(tmp_path, "out")
    assert _info_current(besser, project) == {"Library": False, "Shop": True}


def test_diagrams_sharing_an_id_name_no_current_model(besser, write):
    # The converter caches a class diagram's model by id: both diagrams come out as Library.
    project = write("store.json", _project(
        indices={"ClassDiagram": 1},
        diagrams=[_class_diagram("d1", "Library", "Book"), _class_diagram("d1", "Shop", "Product")],
    ))

    run = besser("generate", "python", project, "-o", "out")
    info = besser("info", project)

    assert run.returncode == 2
    reason = "two of its ClassDiagram diagrams share the id 'd1'"
    assert f"(the project names no current one: {reason})" in run.stderr
    assert "current in the editor" not in info.stdout
    assert reason in info.stderr


def test_use_with_a_name_two_diagrams_share_takes_the_current_one(besser, write, tmp_path):
    diagrams = [_class_diagram("d1", "Model", "Book"), _class_diagram("d2", "Model", "Product")]
    project = write("store.json", _project(indices={"ClassDiagram": 1}, diagrams=diagrams))

    run = besser("generate", "python", project, "-o", "out", "--use", "Model")

    assert run.returncode == 0, run.stderr
    assert ("note: store.json holds 2 domain models named Model (ClassDiagram 0 (id 'd1'); ClassDiagram 1 "
            "(id 'd2')); using ClassDiagram 1 (id 'd2'), the project's current ClassDiagram\n") in run.stderr
    assert "class Product" in _generated_classes(tmp_path, "out")


def test_use_with_a_name_two_diagrams_share_and_no_marker_is_ambiguous(besser, write):
    diagrams = [_class_diagram("d1", "Model", "Book"), _class_diagram("d2", "Model", "Product")]
    project = write("store.json", _project(indices=None, diagrams=diagrams))

    run = besser("generate", "python", project, "-o", "out", "--use", "Model")

    assert run.returncode == 2
    assert run.stderr.startswith(
        "error: --use Model is ambiguous: store.json holds 2 domain models of that name "
        "(ClassDiagram 0 (id 'd1'); ClassDiagram 1 (id 'd2')) and the project names no current one to choose"
    )
    assert "name it with --use NAME" not in run.stderr


# A domain model and an object model of it, to convert to an editor project.
BOOKS_AND_OBJECTS = '''
from besser.BUML.metamodel.object import AttributeLink, DataValue, Object, ObjectModel
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType

title = Property(name="title", type=StringType)
book = Class(name="Book", attributes={title})
books = DomainModel(name="Books", types={book})
novel = Object(name="novel", classifier=book, slots=[
    AttributeLink(attribute=title, value=DataValue(classifier=StringType, value="Novel")),
])
shelf = ObjectModel(name="Shelf", objects={novel})
'''

USER_DIAGRAM = {"id": "user-1", "title": "Alice", "model": {
    "version": "3.0.0", "type": "UserDiagram", "relationships": {},
    "elements": {"u": {"id": "u", "name": "user_1", "type": "UserModelName", "owner": None, "attributes": [],
                       "methods": [], "className": "User", "bounds": {"x": 0, "y": 0, "width": 1, "height": 1}}},
}}


@pytest.fixture
def objects_project(besser, write, tmp_path):
    """An editor project with two object diagrams (First, Second) and a user diagram (Alice)."""
    assert besser("convert", write("books.py", BOOKS_AND_OBJECTS), "-o", "books.json").returncode == 0
    export = json.loads((tmp_path / "books.json").read_text(encoding="utf-8"))
    project = export.get("project", export)
    first = project["diagrams"]["ObjectDiagram"][0]
    second = {**copy.deepcopy(first), "id": "object-2", "title": "Second"}
    first["title"] = "First"
    project["diagrams"]["ObjectDiagram"] = [first, second]
    project["diagrams"]["UserDiagram"] = [USER_DIAGRAM]
    return export


@pytest.mark.parametrize(("current_type", "current"), [("ObjectDiagram", "Second"), ("UserDiagram", "Alice")])
def test_object_and_user_diagrams_are_told_apart_by_the_current_diagram_type(
        besser, write, objects_project, current_type, current):
    project = objects_project.get("project", objects_project)
    project["currentDiagramIndices"] = {"ObjectDiagram": 1, "UserDiagram": 0}
    project["currentDiagramType"] = current_type

    marked = _info_current(besser, write("project.json", objects_project))

    assert {name for name, is_current in marked.items() if is_current} == {current}


def test_object_and_user_markers_with_another_current_type_name_none(besser, write, objects_project):
    project = objects_project.get("project", objects_project)
    project["currentDiagramIndices"] = {"ObjectDiagram": 1, "UserDiagram": 0}
    project["currentDiagramType"] = "ClassDiagram"

    info = besser("info", write("project.json", objects_project))

    assert "current in the editor" not in info.stdout
    assert ("holds 3 object models and names no current one: it marks a current ObjectDiagram and UserDiagram "
            "and its currentDiagramType is neither") in info.stderr


def test_the_current_diagram_of_several_types_at_once(besser, write, objects_project):
    project = objects_project.get("project", objects_project)
    # The books' class diagram second: the object diagrams, which name no class diagram, read the current one.
    project["diagrams"]["ClassDiagram"].insert(0, _class_diagram("d2", "Shop", "Product"))
    project["diagrams"]["UserDiagram"] = []
    project["currentDiagramIndices"] = {"ClassDiagram": 1, "ObjectDiagram": 0}

    marked = _info_current(besser, write("project.json", objects_project))

    assert marked == {"Shop": False, "Class_Diagram": True, "First": True, "Second": False}
