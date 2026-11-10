"""``besser convert``: B-UML ``.py`` <-> editor JSON, through the existing converters."""

import json

from besser.cli.loading import load_models
from tests.cli.conftest import CLASS_DIAGRAM, LIBRARY_MODEL, PROJECT_EXPORT


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
        "ok": True, "command": "convert", "exit_code": 0, "errors": [], "warnings": [], "notes": [],
        "files": ["models/project.py"], "input": "project.json", "output": "models/project.py",
        "dry_run": False,
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


def test_a_buml_script_on_standard_input_becomes_an_editor_project(besser, tmp_path):
    run = besser("convert", "-", "-o", "project.json", stdin=LIBRARY_MODEL)

    assert run.returncode == 0
    assert run.stdout == "Converted <stdin> (1 model) to project.json\n"
    assert _class_names(tmp_path / "project.json") == ["Book", "Library"]
    assert json.loads((tmp_path / "project.json").read_text())["project"]["name"] == "model"


def test_an_editor_export_on_standard_input_becomes_a_buml_file(besser, tmp_path):
    run = besser("convert", "-", "-o", "library.py", stdin=json.dumps(PROJECT_EXPORT))

    assert run.returncode == 0
    assert _class_names(tmp_path / "library.py") == ["Book"]


def test_standard_input_of_the_type_of_the_output_is_a_usage_error(besser, tmp_path):
    run = besser("convert", "-", "-o", "copy.py", stdin=LIBRARY_MODEL)

    assert run.returncode == 2
    assert "got a B-UML script on standard input -> copy.py" in run.stderr
    assert not (tmp_path / "copy.py").exists()


# ── --dry-run ───────────────────────────────────────────────────────────────


def test_dry_run_says_it_would_create_the_target_and_writes_nothing(besser, model, tmp_path):
    run = besser("convert", model, "-o", "out/project.json", "--dry-run")
    as_json = besser("convert", model, "-o", "out/project.json", "--dry-run", "--json")

    assert run.returncode == 0
    assert run.stdout == "Dry run: nothing was written.\nConverting library.py (1 model) would create out/project.json\n"
    assert as_json.json["dry_run"] is True and as_json.json["files"] == []
    assert as_json.json["would_write"] == {"out/project.json": "new"}
    assert not (tmp_path / "out").exists()


def test_dry_run_says_it_would_overwrite_the_target_and_leaves_it(besser, model, write, tmp_path):
    write("project.json", {"mine": True})

    run = besser("convert", model, "-o", "project.json", "--dry-run", "--json")

    assert run.returncode == 0 and run.json["would_write"] == {"project.json": "changed"}
    assert json.loads((tmp_path / "project.json").read_text()) == {"mine": True}


def test_dry_run_says_when_the_target_would_not_change(besser, write, tmp_path):
    besser("convert", write("project.json", PROJECT_EXPORT), "-o", "library.py")
    before = (tmp_path / "library.py").read_bytes()

    run = besser("convert", "project.json", "-o", "library.py", "--dry-run")

    assert run.returncode == 0
    assert run.stdout.endswith("would leave library.py as it is: it holds the same\n")
    assert (tmp_path / "library.py").read_bytes() == before


def test_dry_run_of_an_input_that_does_not_load_fails_as_a_real_run_does(besser, write, tmp_path):
    run = besser("convert", write("crash.py", "1 / 0\n"), "-o", "project.json", "--dry-run", "--json")

    assert run.returncode == 2 and run.json["dry_run"] is True
    assert not (tmp_path / "project.json").exists()


def test_a_dry_run_predicts_the_run(besser, model, write):
    for source, target in ((model, "out/project.json"), (write("project.json", PROJECT_EXPORT), "out/library.py")):
        dry = besser("convert", source, "-o", target, "--dry-run", "--json").json
        real = besser("convert", source, "-o", target, "--json").json

        assert dry["exit_code"] == real["exit_code"] == 0
        assert list(dry["would_write"]) == real["files"] == [target]


# ── -o ──────────────────────────────────────────────────────────────────────


def test_an_output_under_a_file_is_a_usage_error(besser, model, write, tmp_path):
    write("out", "mine\n")

    for dry_run in ([], ["--dry-run"]):
        run = besser("convert", model, "-o", "out/project.json", *dry_run)

        assert run.returncode == 2
        assert run.stderr == "error: out/project.json cannot be created: out is a file\n"
    assert (tmp_path / "out").read_text() == "mine\n"


def test_an_output_that_is_a_directory_is_a_usage_error(besser, model, tmp_path):
    (tmp_path / "project.json").mkdir()

    run = besser("convert", model, "-o", "project.json")

    assert run.returncode == 2 and run.stderr == "error: project.json is a directory: -o takes a file\n"


# ── --quiet and --verbose ───────────────────────────────────────────────────


def test_quiet_prints_nothing_but_errors(besser, model, write):
    done = besser("convert", model, "-o", "project.json", "--quiet")
    failed = besser("convert", write("crash.py", "1 / 0\n"), "-o", "other.json", "--quiet")

    assert done.returncode == 0 and done.stdout == "" and done.stderr == ""
    assert failed.returncode == 2 and failed.stdout == ""
    assert failed.stderr == "error: crash.py: line 1: ZeroDivisionError: division by zero\n"


def test_verbose_adds_the_traceback_of_a_failure(besser, write):
    run = besser("convert", write("crash.py", "1 / 0\n"), "-o", "project.json", "--verbose")

    assert run.returncode == 2 and run.stdout == ""
    assert "Traceback (most recent call last):" in run.stderr and "ZeroDivisionError" in run.stderr
