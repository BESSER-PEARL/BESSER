"""``besser generate``: one generator, run on a model file into a directory."""

import copy

from tests.cli.conftest import (
    CLASS_DIAGRAM,
    INVALID_MODEL,
    LIBRARY_MODEL,
    PROJECT_EXPORT,
    TWO_DOMAIN_MODELS,
    WARNING_MODEL,
)


def test_it_writes_the_files_and_prints_them(besser, model, tmp_path):
    run = besser("generate", "python", model, "-o", "out")

    assert run.returncode == 0
    assert run.stdout == "Generated 1 file in out with the 'python' generator:\n  out/classes.py\n"
    assert "class Book" in (tmp_path / "out" / "classes.py").read_text()


def test_json_lists_the_files_under_the_output_directory_as_given(besser, model, tmp_path):
    run = besser("generate", "pydantic", model, "--output", "generated/models", "--json")

    assert run.returncode == 0
    assert run.json == {
        "ok": True, "command": "generate", "errors": [], "warnings": [],
        "files": ["generated/models/pydantic_classes.py"],
        "generator": "pydantic", "model": model, "output_dir": "generated/models",
    }
    assert (tmp_path / "generated" / "models" / "pydantic_classes.py").is_file()


def test_a_project_generator_lists_every_file(besser, model, tmp_path):
    result = besser("generate", "backend", model, "-o", "api", "--json").json

    assert result["ok"] is True
    assert {"api/main_api.py", "api/sql_alchemy.py", "api/routers/book.py"} <= set(result["files"])
    assert all((tmp_path / path).is_file() for path in result["files"])


def test_an_option_reaches_the_generator(besser, model, tmp_path):
    run = besser("generate", "sqlalchemy", model, "-o", "out", "--option", "dbms=postgresql")

    assert run.returncode == 0
    assert "postgresql" in (tmp_path / "out" / "sql_alchemy.py").read_text()


def test_options_are_read_as_their_schema_type(besser, model, tmp_path):
    run = besser(
        "generate", "rest_api", model, "-o", "out",
        "--option", "http_methods=GET, POST", "--json",
    )
    rest_api = (tmp_path / "out" / "rest_api.py").read_text()

    assert run.returncode == 0
    assert "@app.get(" in rest_api and "@app.post(" in rest_api and "@app.delete(" not in rest_api


def test_it_generates_from_an_editor_project(besser, write, tmp_path):
    run = besser("generate", "sql", write("project.json", PROJECT_EXPORT), "-o", "out", "--json")

    assert run.returncode == 0
    assert run.json["files"] == ["out/tables_sqlite.sql"]
    assert "CREATE TABLE book" in (tmp_path / "out" / "tables_sqlite.sql").read_text()


def test_it_generates_from_a_single_editor_diagram(besser, write, tmp_path):
    run = besser("generate", "python", write("diagram.json", CLASS_DIAGRAM), "-o", "out")

    assert run.returncode == 0
    assert "class Book" in (tmp_path / "out" / "classes.py").read_text()


def test_a_model_with_validation_errors_is_not_generated_from(besser, write, tmp_path):
    run = besser("generate", "python", write("broken.py", INVALID_MODEL), "-o", "out", "--json")

    assert run.returncode == 1
    assert run.json["ok"] is False and run.json["files"] == []
    assert run.json["errors"][0].startswith("broken_model: Attribute 'title' of class 'Book' has type 'Text'")
    assert run.json["errors"][-1] == "nothing was generated: the model has validation errors"
    assert not (tmp_path / "out").exists()


def test_validation_warnings_are_reported_and_do_not_stop_it(besser, write):
    run = besser("generate", "python", write("shelf.py", WARNING_MODEL), "-o", "out", "--json")

    assert run.returncode == 0
    assert run.json["files"] == ["out/classes.py"] and "Shelf" in run.json["warnings"][0]


def test_an_unknown_generator_is_a_usage_error_naming_the_generators(besser, model):
    run = besser("generate", "cobol", model, "-o", "out")

    assert run.returncode == 2
    assert "error: unknown generator 'cobol'; available generators: python, java, " in run.stderr


def test_an_unknown_option_is_a_usage_error_naming_the_options(besser, model, tmp_path):
    run = besser("generate", "sqlalchemy", model, "-o", "out", "--option", "dmbs=postgresql", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == ["unknown option(s) for the 'sqlalchemy' generator: dmbs (it takes dbms)"]
    assert not (tmp_path / "out").exists()


def test_an_option_value_outside_its_choices_is_a_usage_error(besser, model):
    run = besser("generate", "sqlalchemy", model, "-o", "out", "--option", "dbms=db2")

    assert run.returncode == 2
    assert "dbms must be one of sqlite, postgresql, mysql, mssql, mariadb, oracle, got 'db2'" in run.stderr


def test_an_option_value_of_the_wrong_type_is_a_usage_error(besser, model):
    run = besser("generate", "pydantic", model, "-o", "out", "--option", "backend=maybe")

    assert run.returncode == 2
    assert "error: backend must be true or false, got 'maybe'" in run.stderr


def test_an_option_without_a_value_is_a_usage_error(besser, model):
    run = besser("generate", "pydantic", model, "-o", "out", "--option", "backend")

    assert run.returncode == 2
    assert "error: --option takes name=value, got 'backend'" in run.stderr


def test_a_model_kind_the_file_does_not_hold_is_a_usage_error(besser, model):
    run = besser("generate", "react", model, "-o", "out")

    assert run.returncode == 2
    assert "error: the 'react' generator reads a gui model and library.py holds none (it holds: domain)" in run.stderr


def test_several_models_of_the_kind_it_reads_is_a_usage_error(besser, write):
    run = besser("generate", "python", write("two.py", TWO_DOMAIN_MODELS), "-o", "out")

    assert run.returncode == 2
    assert ("two.py holds 2 domain models (library_model, draft) and the 'python' generator reads one: "
            "name it with --use NAME") in run.stderr


def test_use_picks_a_model_by_the_name_it_is_bound_to(besser, write, tmp_path):
    run = besser("generate", "python", write("two.py", TWO_DOMAIN_MODELS), "-o", "out", "--use", "draft")
    classes = (tmp_path / "out" / "classes.py").read_text()

    assert run.returncode == 0
    assert "class Idea" in classes and "class Book" not in classes


def test_use_picks_a_model_by_its_own_name(besser, write, tmp_path):
    run = besser("generate", "python", write("two.py", TWO_DOMAIN_MODELS), "-o", "out", "--use", "LibraryModel")
    classes = (tmp_path / "out" / "classes.py").read_text()

    assert run.returncode == 0
    assert "class Book" in classes and "class Idea" not in classes


def test_use_picks_a_diagram_of_an_editor_project_by_its_title(besser, write, tmp_path):
    project = copy.deepcopy(PROJECT_EXPORT)
    notes = copy.deepcopy(CLASS_DIAGRAM)
    notes["title"] = "Notes"
    notes["model"]["elements"]["class-1"]["name"] = "Note"
    project["project"]["diagrams"]["ClassDiagram"].append({"id": "diagram-2", **notes})
    write("project.json", project)

    refused = besser("generate", "python", "project.json", "-o", "out")
    run = besser("generate", "python", "project.json", "-o", "out", "--use", "Notes")
    classes = (tmp_path / "out" / "classes.py").read_text()

    assert refused.returncode == 2 and "project.json holds 2 domain models (Library, Notes)" in refused.stderr
    assert run.returncode == 0
    assert "class Note" in classes and "class Book" not in classes


def test_use_with_a_name_the_file_does_not_hold_is_a_usage_error(besser, write, tmp_path):
    run = besser("generate", "python", write("two.py", TWO_DOMAIN_MODELS), "-o", "out", "--use", "sketch")

    assert run.returncode == 2
    assert "error: two.py holds no model named sketch (it holds: library_model, draft)" in run.stderr
    assert not (tmp_path / "out").exists()


def test_the_model_is_read_from_standard_input(besser, tmp_path):
    run = besser("generate", "python", "-", "-o", "out", "--json", stdin=LIBRARY_MODEL)

    assert run.returncode == 0
    assert run.json["files"] == ["out/classes.py"] and run.json["model"] == "-"
    assert "class Book" in (tmp_path / "out" / "classes.py").read_text()


def test_an_output_directory_that_is_not_empty_is_refused(besser, model, write, tmp_path):
    write("out/classes.py", "# my own work\n")
    run = besser("generate", "python", model, "-o", "out", "--json")

    assert run.returncode == 2
    assert run.json["files"] == []
    assert run.json["errors"] == [
        "out is not empty and generating overwrites what is there: pass --force to overwrite, "
        "or keep hand edits with 'besser regenerate' (see 'besser regenerate --help')"
    ]
    assert (tmp_path / "out" / "classes.py").read_text() == "# my own work\n"


def test_a_second_run_into_the_same_directory_is_refused(besser, model):
    assert besser("generate", "python", model, "-o", "out").returncode == 0

    run = besser("generate", "python", model, "-o", "out")

    assert run.returncode == 2 and run.stdout == ""
    assert "error: out is not empty" in run.stderr


def test_an_empty_output_directory_is_written_to(besser, model, tmp_path):
    (tmp_path / "out").mkdir()

    assert besser("generate", "python", model, "-o", "out").returncode == 0


def test_force_overwrites_a_directory_that_is_not_empty(besser, model, write, tmp_path):
    write("out/classes.py", "# my own work\n")
    write("out/notes.txt", "kept\n")
    run = besser("generate", "python", model, "-o", "out", "--force", "--json")

    assert run.returncode == 0 and run.json["files"] == ["out/classes.py"]
    assert "class Book" in (tmp_path / "out" / "classes.py").read_text()
    assert (tmp_path / "out" / "notes.txt").read_text() == "kept\n"


def test_a_model_file_that_does_not_load_is_a_usage_error(besser, write):
    run = besser("generate", "python", write("crash.py", "raise RuntimeError('no')\n"), "-o", "out", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == ["crash.py: line 1: RuntimeError: no"]


def test_the_output_directory_is_required(besser, model):
    run = besser("generate", "python", model)

    assert run.returncode == 2
    assert "the following arguments are required: -o/--output" in run.stderr
