"""``besser convert``: B-UML ``.py`` <-> editor JSON, through the existing converters."""

import json

from besser.cli.loading import load_models
from tests.cli.conftest import CLASS_DIAGRAM, PROJECT_EXPORT


def _class_names(path) -> list[str]:
    [loaded] = load_models(path).of_kind("domain")
    return sorted(cls.name for cls in loaded.model.get_classes())


def test_a_buml_file_becomes_an_editor_project(besser, model, tmp_path):
    run = besser("convert", model, "-o", "project.json")

    assert run.returncode == 0
    assert run.stdout == "Converted library.py (1 model) to project.json\n"
    export = json.loads((tmp_path / "project.json").read_text())
    assert set(export) == {"project", "exportedAt", "version"}
    [diagram] = export["project"]["diagrams"]["ClassDiagram"]
    names = {element["name"] for element in diagram["model"]["elements"].values() if element["type"] == "Class"}
    assert names == {"Library", "Book"}
    assert len(diagram["model"]["relationships"]) == 1


def test_an_editor_project_becomes_a_buml_file(besser, write, tmp_path):
    run = besser("convert", write("project.json", PROJECT_EXPORT), "--output", "models/project.py", "--json")

    assert run.returncode == 0
    assert run.json == {
        "ok": True, "command": "convert", "errors": [], "warnings": [],
        "files": ["models/project.py"], "input": "project.json", "output": "models/project.py",
    }
    assert _class_names(tmp_path / "models" / "project.py") == ["Book"]


def test_a_single_editor_diagram_becomes_a_buml_file(besser, write, tmp_path):
    assert besser("convert", write("diagram.json", CLASS_DIAGRAM), "-o", "diagram.py").returncode == 0
    assert _class_names(tmp_path / "diagram.py") == ["Book"]


def test_a_model_survives_the_round_trip(besser, model, tmp_path):
    assert besser("convert", model, "-o", "project.json").returncode == 0
    assert besser("convert", "project.json", "-o", "back.py").returncode == 0

    assert _class_names(tmp_path / "back.py") == _class_names(tmp_path / model) == ["Book", "Library"]
    [back] = load_models(tmp_path / "back.py").of_kind("domain")
    assert [association.name for association in back.model.associations] == ["holds"]
    assert besser("validate", "back.py", "--strict").returncode == 0


def test_the_converted_project_is_one_the_cli_reads_back(besser, model):
    besser("convert", model, "-o", "project.json")

    assert besser("validate", "project.json").returncode == 0
    assert besser("generate", "python", "project.json", "-o", "out").returncode == 0


def test_two_files_of_the_same_type_is_a_usage_error(besser, model):
    run = besser("convert", model, "-o", "copy.py")

    assert run.returncode == 2
    assert "got library.py -> copy.py" in run.stderr


def test_an_unknown_output_type_is_a_usage_error(besser, model):
    assert besser("convert", model, "-o", "model.xmi").returncode == 2


def test_an_input_that_does_not_load_is_a_load_error_and_writes_nothing(besser, write, tmp_path):
    run = besser("convert", write("crash.py", "1 / 0\n"), "-o", "project.json", "--json")

    assert run.returncode == 2
    assert run.json["ok"] is False and run.json["files"] == []
    assert not (tmp_path / "project.json").exists()
