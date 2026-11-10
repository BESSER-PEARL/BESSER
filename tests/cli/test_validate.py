"""``besser validate``: exit status, report and JSON output."""

import json

from tests.cli.conftest import CLASS_DIAGRAM, INVALID_MODEL, LIBRARY_MODEL, PROJECT_EXPORT, WARNING_MODEL


def test_a_valid_model_exits_0_and_names_its_models(besser, model):
    run = besser("validate", model)

    assert run.returncode == 0
    assert "library_model (DomainModel): ok" in run.stdout
    assert run.stdout.rstrip().endswith("OK: 0 errors, 0 warnings")


def test_json_output_has_the_stable_keys(besser, model):
    result = besser("validate", model, "--json").json

    assert {"ok", "errors", "warnings", "files"} <= set(result)
    assert result["ok"] is True and result["command"] == "validate"
    assert result["errors"] == [] and result["warnings"] == [] and result["files"] == []
    assert result["model_files"] == [model]
    assert result["models"] == [{
        "file": model, "name": "library_model", "kind": "domain", "type": "DomainModel",
        "validated": True, "errors": [], "warnings": [],
    }]
    assert result["results"] == [
        {"file": model, "ok": True, "exit_code": 0, "errors": [], "warnings": [], "models": result["models"]},
    ]
    assert result["skipped"] == []


def test_validation_errors_exit_1(besser, write):
    run = besser("validate", write("broken.py", INVALID_MODEL))

    assert run.returncode == 1
    assert "broken_model (DomainModel): 1 error, 0 warnings" in run.stdout
    assert "error: Attribute 'title' of class 'Book' has type 'Text'" in run.stdout
    assert run.stdout.rstrip().endswith("FAILED: 1 error, 0 warnings")


def test_validation_errors_in_json(besser, write):
    run = besser("--json", "validate", write("broken.py", INVALID_MODEL))

    assert run.returncode == 1
    assert run.json["ok"] is False
    [error] = run.json["errors"]
    assert error.startswith("broken_model: Attribute 'title' of class 'Book' has type 'Text'")
    assert run.json["models"][0]["errors"] == [error.removeprefix("broken_model: ")]


def test_warnings_do_not_fail(besser, write):
    run = besser("validate", write("shelf.py", WARNING_MODEL), "--json")

    assert run.returncode == 0 and run.json["ok"] is True
    assert len(run.json["warnings"]) == 1 and "Shelf" in run.json["warnings"][0]


def test_strict_fails_on_warnings(besser, write):
    run = besser("validate", write("shelf.py", WARNING_MODEL), "--strict")

    assert run.returncode == 1
    assert run.stdout.rstrip().endswith("FAILED: 0 errors, 1 warning (--strict)")
    assert besser("validate", "shelf.py", "--strict", "--json").json["ok"] is False


def test_strict_passes_a_model_without_warnings(besser, model):
    assert besser("validate", model, "--strict").returncode == 0


def test_a_file_that_raises_is_a_load_error_naming_the_line(besser, write):
    run = besser("validate", write("crash.py", "x = 1\ny = x / 0\n"))

    assert run.returncode == 2
    assert run.stdout == ""
    assert "error: crash.py: line 2: ZeroDivisionError: division by zero" in run.stderr


def test_a_syntax_error_is_a_load_error(besser, write):
    run = besser("validate", write("typo.py", "model = (\n"), "--json")

    assert run.returncode == 2
    assert run.json["ok"] is False and "SyntaxError" in run.json["errors"][0]


def test_a_missing_file_is_a_load_error(besser):
    run = besser("validate", "nowhere.py", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == ["nowhere.py: no such file"]


def test_a_file_of_another_type_is_a_load_error(besser, write):
    run = besser("validate", write("model.yaml", "a: 1\n"))

    assert run.returncode == 2
    assert "expected a B-UML .py file or an editor .json export" in run.stderr


def test_a_file_without_a_model_is_a_load_error(besser, write):
    run = besser("validate", write("empty.py", "x = 1\n"))

    assert run.returncode == 2
    assert "empty.py: no B-UML model found" in run.stderr


def test_what_the_model_file_prints_goes_to_standard_error(besser, write):
    run = besser("validate", write("chatty.py", LIBRARY_MODEL + 'print("building the model")\n'), "--json")

    assert run.json["ok"] is True
    assert "building the model" in run.stderr


def test_the_main_block_of_a_model_file_does_not_run(besser, write, tmp_path):
    source = LIBRARY_MODEL + '''
if __name__ == "__main__":
    open("ran_main.txt", "w").close()
'''
    assert besser("validate", write("script.py", source)).returncode == 0
    assert not (tmp_path / "ran_main.txt").exists()


def test_a_module_beside_the_model_file_is_not_imported(besser, write):
    write("models/helper_beside_the_model.py", "raise SystemExit('the helper was executed')\n")
    run = besser("validate", write("models/uses_helper.py", "import helper_beside_the_model\n" + LIBRARY_MODEL))

    assert run.returncode == 2
    assert "ModuleNotFoundError" in run.stderr and "the helper was executed" not in run.stderr


def test_a_project_in_the_file_says_which_models_it_holds(besser, write):
    source = LIBRARY_MODEL + '''
from besser.BUML.metamodel.project import Project

draft = DomainModel(name="Draft", types={Class(name="Idea")})
project = Project(name="library", models=[library_model])
'''
    result = besser("validate", write("project.py", source), "--json").json

    assert [entry["name"] for entry in result["models"]] == ["library_model"]


def test_every_model_of_a_file_without_a_project_is_validated(besser, write):
    source = LIBRARY_MODEL + 'draft = DomainModel(name="Draft", types={Class(name="Idea")})\n'
    result = besser("validate", write("two.py", source), "--json").json

    assert [entry["name"] for entry in result["models"]] == ["library_model", "draft"]
    assert result["ok"] is True and any(warning.startswith("draft: ") for warning in result["warnings"])


def test_an_editor_project_export_is_validated(besser, write):
    run = besser("validate", write("project.json", PROJECT_EXPORT), "--json")

    assert run.returncode == 0
    assert [(entry["kind"], entry["type"]) for entry in run.json["models"]] == [("domain", "DomainModel")]


def test_a_single_editor_diagram_is_validated(besser, write):
    run = besser("validate", write("diagram.json", CLASS_DIAGRAM))

    assert run.returncode == 0
    assert "(DomainModel): ok" in run.stdout


def test_a_file_that_is_not_json_is_a_load_error(besser, write):
    run = besser("validate", write("broken.json", "{not json"))

    assert run.returncode == 2
    assert "broken.json: not a JSON file" in run.stderr


def test_json_that_is_not_an_editor_export_is_a_load_error(besser, write):
    run = besser("validate", write("other.json", {"name": "something else"}))

    assert run.returncode == 2
    assert "not an editor export" in run.stderr


def test_an_editor_diagram_the_converter_refuses_is_a_load_error(besser, write):
    diagram = {"title": "Bad", "model": {"type": "ClassDiagram", "relationships": {}, "elements": {
        "class-1": {"type": "Class", "name": "", "attributes": [], "methods": []},
    }}}
    run = besser("validate", write("bad.json", diagram), "--json")

    assert run.returncode == 2
    assert run.json["errors"][0].startswith("bad.json: cannot convert to B-UML: ")


def test_several_files_are_validated_in_one_run(besser, model, write):
    run = besser("validate", model, write("shelf.py", WARNING_MODEL))

    assert run.returncode == 0
    assert "library.py: 1 model\n  library_model (DomainModel): ok\n" in run.stdout
    assert "shelf.py: 1 model\n  library_model (DomainModel): 0 errors, 1 warning\n" in run.stdout
    assert run.stdout.rstrip().endswith("OK: 0 errors, 1 warning")


def test_several_files_in_json_say_which_file_each_model_and_message_is_from(besser, model, write):
    run = besser("validate", model, write("broken.py", INVALID_MODEL), "--json")

    assert run.returncode == 1
    assert run.json["model_files"] == ["library.py", "broken.py"]
    assert [(report["file"], report["name"]) for report in run.json["models"]] == [
        ("library.py", "library_model"), ("broken.py", "broken_model"),
    ]
    [error] = run.json["errors"]
    assert error.startswith("broken.py: broken_model: Attribute 'title' of class 'Book' has type 'Text'")


def test_one_file_with_errors_fails_the_run(besser, model, write):
    run = besser("validate", write("broken.py", INVALID_MODEL), model)

    assert run.returncode == 1
    assert "library_model (DomainModel): ok" in run.stdout
    assert run.stdout.rstrip().endswith("FAILED: 1 error, 0 warnings")


def test_a_file_that_does_not_load_exits_2_and_the_others_are_still_validated(besser, model, write):
    run = besser("validate", "missing.py", model, write("broken.py", INVALID_MODEL))

    assert run.returncode == 2
    # Listed in its place among the files checked, and not again on standard error.
    assert run.stdout.startswith("missing.py: does not load: no such file\nlibrary.py: 1 model\n")
    assert run.stderr == ""
    assert "library_model (DomainModel): ok" in run.stdout
    assert "broken_model (DomainModel): 1 error, 0 warnings" in run.stdout
    assert run.stdout.rstrip().endswith("FAILED: 2 errors, 0 warnings")

    result = besser("validate", "missing.py", model, "--json").json
    assert result["ok"] is False and result["errors"] == ["missing.py: no such file"]
    assert [report["file"] for report in result["models"]] == ["library.py"]


def test_a_python_model_is_read_from_standard_input(besser):
    run = besser("validate", "-", "--json", stdin=LIBRARY_MODEL)

    assert run.returncode == 0
    assert run.json["models"] == [{
        "file": "-", "name": "library_model", "kind": "domain", "type": "DomainModel",
        "validated": True, "errors": [], "warnings": [],
    }]
    assert "<stdin>: 1 model" in besser("validate", "-", stdin=LIBRARY_MODEL).stdout


def test_an_editor_export_is_read_from_standard_input(besser):
    run = besser("validate", "-", stdin=json.dumps(PROJECT_EXPORT))

    assert run.returncode == 0
    assert "(DomainModel): ok" in run.stdout


def test_a_model_on_standard_input_that_raises_names_the_line(besser):
    run = besser("validate", "-", stdin="x = 1\nraise RuntimeError('no')\n")

    assert run.returncode == 2
    assert run.stderr == "error: <stdin>: line 2: RuntimeError: no\n"


def test_empty_standard_input_is_a_load_error(besser):
    run = besser("validate", "-")

    assert run.returncode == 2
    assert run.stderr == "error: <stdin>: no B-UML model found\n"


def test_the_main_block_of_a_model_on_standard_input_does_not_run(besser):
    source = LIBRARY_MODEL + 'if __name__ == "__main__":\n    raise SystemExit("ran")\n'

    assert besser("validate", "-", stdin=source).returncode == 0
