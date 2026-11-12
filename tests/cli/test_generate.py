"""``besser generate``: one generator, run on a model file into a directory."""

import copy
import json
import shlex

from tests.cli.conftest import (
    CLASS_DIAGRAM,
    INVALID_MODEL,
    LIBRARY_MODEL,
    LIBRARY_MODEL_WITH_ISBN,
    PROJECT_EXPORT,
    TWO_DOMAIN_MODELS,
    WARNING_MODEL,
)

RECORD = ".besser_generated.json"


def test_it_writes_the_files_and_prints_them(besser, model, tmp_path):
    run = besser("generate", "python", model, "-o", "out")

    assert run.returncode == 0
    assert run.stdout == "Generated 1 file in out with the 'python' generator:\n  out/classes.py\n"
    assert "class Book" in (tmp_path / "out" / "classes.py").read_text()


def test_json_lists_the_files_under_the_output_directory_as_given(besser, model, tmp_path):
    run = besser("generate", "pydantic", model, "--output", "generated/models", "--json")

    assert run.returncode == 0
    assert run.json == {
        "ok": True, "command": "generate", "exit_code": 0, "errors": [], "warnings": [],
        "notes": ["recorded the output in generated/models/.besser_generated.json: 'besser regenerate pydantic "
                  f"{model} -o generated/models' brings it up to a changed model and keeps your edits"],
        "files": ["generated/models/pydantic_classes.py"],
        "generator": "pydantic", "model": model, "output_dir": "generated/models", "dry_run": False,
        "recorded": True, "used": {"domain": "library_model"},
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
        "or keep hand edits with 'besser regenerate' (a folder generated without a record needs "
        "'besser regenerate python library.py -o out --adopt' first; see 'besser regenerate --help')"
    ]
    assert (tmp_path / "out" / "classes.py").read_text() == "# my own work\n"


def test_a_generator_without_regenerate_is_not_pointed_to_regenerate(besser, model, write):
    write("out/notes.txt", "mine\n")
    run = besser("generate", "supabase", model, "-o", "out", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == [
        "out is not empty and generating overwrites what is there: pass --force to overwrite (the 'supabase' "
        "generator does not support 'besser regenerate', which keeps hand edits)"
    ]


def test_a_second_run_into_the_same_directory_is_refused(besser, model):
    assert besser("generate", "python", model, "-o", "out").returncode == 0

    run = besser("generate", "python", model, "-o", "out")

    assert run.returncode == 2 and run.stdout == ""
    assert "error: out holds generated code and the record of it (.besser_generated.json)" in run.stderr


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


# ── the record 'besser regenerate' merges against ──────────────────────────


def _tree(folder) -> dict:
    """Every file under ``folder``, by path, with its bytes: what a dry run must leave as it was."""
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def test_it_records_the_output_of_a_generator_that_supports_regenerate(besser, model, tmp_path):
    run = besser("generate", "python", model, "-o", "out", "--json")

    record = json.loads((tmp_path / "out" / RECORD).read_text())
    assert run.json["recorded"] is True and run.json["files"] == ["out/classes.py"]
    assert record["generator"] == "PythonGenerator"
    assert record["files"] == {"classes.py": (tmp_path / "out" / "classes.py").read_text()}
    assert run.json["notes"] == [
        "recorded the output in out/.besser_generated.json: 'besser regenerate python library.py -o out' "
        "brings it up to a changed model and keeps your edits"
    ]


def test_recording_does_not_need_pygit2(besser, model, write, tmp_path):
    # 'python -m' puts the working directory first on sys.path, so this module stands in for pygit2.
    write("pygit2.py", "raise ImportError('hidden by the test')\n")

    run = besser("generate", "python", model, "-o", "out", "--json")
    again = besser("regenerate", "python", model, "-o", "out")

    assert run.returncode == 0 and run.json["recorded"] is True
    assert (tmp_path / "out" / RECORD).is_file()
    # Nothing was edited, so there is nothing to merge.
    assert again.returncode == 0, again.stderr


def test_the_record_is_the_same_bytes_for_the_same_model(besser, model, tmp_path):
    besser("generate", "backend", model, "-o", "first")
    besser("generate", "backend", model, "-o", "second")

    assert (tmp_path / "first" / RECORD).read_bytes() == (tmp_path / "second" / RECORD).read_bytes()


def test_force_writes_a_fresh_record_of_what_it_wrote(besser, model, write, tmp_path):
    besser("generate", "python", model, "-o", "out")
    write("library.py", LIBRARY_MODEL_WITH_ISBN)

    run = besser("generate", "python", model, "-o", "out", "--force", "--json")

    record = json.loads((tmp_path / "out" / RECORD).read_text())
    assert run.returncode == 0 and run.json["recorded"] is True and run.json["warnings"] == []
    assert record["files"]["classes.py"] == (tmp_path / "out" / "classes.py").read_text()
    assert "isbn" in record["files"]["classes.py"]


def test_a_generator_without_regenerate_support_writes_no_record(besser, model, tmp_path):
    run = besser("generate", "supabase", model, "-o", "out", "--json")

    assert run.returncode == 0
    assert run.json["recorded"] is False and run.json["notes"] == []
    assert not (tmp_path / "out" / RECORD).exists()


# ── --dry-run ───────────────────────────────────────────────────────────────


def test_dry_run_lists_what_it_would_write_and_writes_nothing(besser, model, tmp_path):
    run = besser("generate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.returncode == 0
    # The record is bookkeeping, not output: "recorded" says the run would write it.
    assert run.json["dry_run"] is True and run.json["recorded"] is True and run.json["files"] == []
    assert run.json["would_write"] == {"out/classes.py": "new"}
    assert run.json["notes"] == [
        "would record the output in out/.besser_generated.json: 'besser regenerate python library.py -o out' "
        "would then bring it up to a changed model and keep your edits"
    ]
    assert not (tmp_path / "out").exists()


def test_dry_run_says_so_first(besser, model):
    run = besser("generate", "python", model, "-o", "out", "--dry-run")

    assert run.stdout == (
        "Dry run: nothing was written.\n"
        "The 'python' generator would write 1 file in out (1 new):\n"
        "  new  out/classes.py\n"
    )


def test_dry_run_over_generated_code_tells_changed_from_unchanged(besser, model, write, tmp_path):
    besser("generate", "python", model, "-o", "out")
    before = _tree(tmp_path / "out")

    same = besser("generate", "python", model, "-o", "out", "--force", "--dry-run", "--json")
    write("library.py", LIBRARY_MODEL_WITH_ISBN)
    changed = besser("generate", "python", model, "-o", "out", "--force", "--dry-run", "--json")

    assert same.json["would_write"] == {"out/classes.py": "unchanged"}
    assert changed.json["would_write"] == {"out/classes.py": "changed"}
    assert _tree(tmp_path / "out") == before


def test_dry_run_still_refuses_a_directory_that_is_not_empty(besser, model, write, tmp_path):
    write("out/classes.py", "# my own work\n")

    run = besser("generate", "python", model, "-o", "out", "--dry-run", "--json")

    assert run.returncode == 2 and run.json["dry_run"] is True
    assert run.json["errors"][0].startswith("out is not empty and generating overwrites what is there")
    assert _tree(tmp_path / "out") == {"classes.py": b"# my own work\n"}


def test_dry_run_of_a_generator_without_regenerate_support_lists_no_record(besser, model, tmp_path):
    run = besser("generate", "supabase", model, "-o", "out", "--dry-run", "--json")

    [path] = run.json["would_write"]
    assert run.returncode == 0 and path.endswith("_librarymodel.sql")
    assert not (tmp_path / "out").exists()


# ── the regenerate command it names ─────────────────────────────────────────


def _named_command(message: str) -> list[str]:
    """The ``besser regenerate`` command quoted in ``message``, as its arguments."""
    start = message.index("besser regenerate")
    quote = message[start - 1]
    return shlex.split(message[start:message.index(quote, start)])


def test_the_regenerate_command_it_notes_reads_the_same_options_and_model(besser, write, tmp_path):
    two = write("two.py", TWO_DOMAIN_MODELS)
    run = besser("generate", "sqlalchemy", two, "-o", "out", "--option", "dbms=postgresql",
                 "--use", "library_model", "--json")
    [note] = run.json["notes"]

    assert note == (
        "recorded the output in out/.besser_generated.json: 'besser regenerate sqlalchemy two.py -o out "
        "--option dbms=postgresql --use library_model' brings it up to a changed model and keeps your edits"
    )
    # Run as shown, it regenerates what generate wrote: nothing changes.
    again = besser(*_named_command(note)[1:])
    assert again.returncode == 0 and again.stdout == "out is up to date with two.py: no file changed.\n"


def test_the_regenerate_command_quotes_an_option_value_for_the_shell(besser, model):
    run = besser("generate", "rest_api", model, "-o", "out", "--option", "http_methods=GET, POST", "--json")

    [note] = run.json["notes"]
    assert "\"besser regenerate rest_api library.py -o out --option 'http_methods=GET, POST'\"" in note
    assert _named_command(note)[-2:] == ["--option", "http_methods=GET, POST"]


def test_the_refusal_of_a_recorded_directory_names_the_options_given(besser, model):
    besser("generate", "sqlalchemy", model, "-o", "out", "--option", "dbms=postgresql")

    run = besser("generate", "sqlalchemy", model, "-o", "out", "--option", "dbms=postgresql")

    assert run.returncode == 2
    assert "'besser regenerate sqlalchemy library.py -o out --option dbms=postgresql' brings it up" in run.stderr


def test_without_pygit2_the_note_says_what_regenerate_needs(besser, model, write):
    # 'python -m' puts the working directory first on sys.path, so this module stands in for pygit2.
    write("pygit2.py", "raise ImportError('hidden by the test')\n")

    [note] = besser("generate", "python", model, "-o", "out", "--json").json["notes"]

    assert note.endswith('keeps your edits (needs pip install "besser[regen]")')


# ── what the dry run predicts ───────────────────────────────────────────────


def test_a_dry_run_predicts_the_run(besser, model, write):
    for args in (["python", model, "-o", "out"], ["backend", model, "-o", "api"]):
        dry = besser("generate", *args, "--dry-run", "--json").json
        real = besser("generate", *args, "--json").json

        assert dry["exit_code"] == real["exit_code"] == 0
        assert sorted(dry["would_write"]) == real["files"]
        assert dry["recorded"] is real["recorded"] is True

    write("library.py", LIBRARY_MODEL_WITH_ISBN)
    dry = besser("generate", "python", model, "-o", "out", "--force", "--dry-run", "--json").json
    real = besser("generate", "python", model, "-o", "out", "--force", "--json").json
    assert dry["would_write"] == {"out/classes.py": "changed"} and real["files"] == ["out/classes.py"]


# ── -o ──────────────────────────────────────────────────────────────────────


def test_an_output_that_is_a_file_is_refused_before_the_run(besser, model, write, tmp_path):
    write("out", "mine\n")

    for dry_run in ([], ["--dry-run"]):
        run = besser("generate", "python", model, "-o", "out", *dry_run, "--json")

        assert run.returncode == 2
        assert run.json["errors"] == ["out is a file: -o takes a directory"]
    assert (tmp_path / "out").read_text() == "mine\n"


def test_an_output_under_a_file_is_refused(besser, model, write):
    write("out", "mine\n")

    run = besser("generate", "python", model, "-o", "out/app", "--dry-run")

    assert run.returncode == 2 and run.stderr == "error: out/app cannot be created: out is a file\n"


# ── an editor project's diagrams ────────────────────────────────────────────


def _project_with_a_dangling_reference() -> dict:
    """The library project and a component diagram whose component realizes a class it does not hold."""
    project = copy.deepcopy(PROJECT_EXPORT)
    project["project"]["diagrams"]["ComponentDiagram"] = [{
        "id": "diagram-2", "title": "Components",
        "model": {"type": "ComponentDiagram", "relationships": {}, "elements": {
            "component-1": {"id": "component-1", "name": "Catalog", "type": "Component", "owner": None,
                            "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                            "realizes": ["missing-class"]},
        }},
    }]
    return project


def test_a_project_with_dangling_references_between_diagrams_is_not_generated_from(besser, write, tmp_path):
    run = besser("generate", "python", write("project.json", _project_with_a_dangling_reference()), "-o", "out",
                 "--json")

    assert run.returncode == 1
    assert run.json["errors"] == [
        "Component 'Catalog' realizes unknown Class id 'missing-class'.",
        "nothing was generated: the project's diagrams reference elements or diagrams it does not hold",
    ]
    assert not (tmp_path / "out").exists()


def test_a_name_in_the_model_cannot_make_a_generator_write_outside_its_folder(besser, write, tmp_path):
    assert besser("new", "agent.json", "--kind", "agent").returncode == 0
    project = json.loads((tmp_path / "agent.json").read_text())
    project["project"]["diagrams"]["AgentDiagram"][0]["title"] = "../outside/pwnagent"
    write("agent.json", project)
    (tmp_path / "outside").mkdir()

    run = besser("generate", "agent", "agent.json", "-o", "out", "--json")

    assert run.returncode != 0
    assert "leads out of the output folder" in run.json["errors"][0]
    assert list((tmp_path / "outside").iterdir()) == []
