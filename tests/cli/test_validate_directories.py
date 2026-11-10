"""``besser validate`` on directories: what is searched, what is skipped, how the results add up."""

import os

import pytest

from besser.cli.discovery import (
    SKIP_LINKED_DIRECTORY,
    SKIP_NO_MODEL,
    SKIP_NOT_A_MODEL,
    SKIP_NOT_JSON,
    SKIP_OUTSIDE,
    SKIP_PY,
)
from tests.cli.conftest import CLASS_DIAGRAM, INVALID_MODEL, LIBRARY_MODEL, PROJECT_EXPORT, WARNING_MODEL

# Executing this file leaves a marker behind, which proves whether it was loaded.
MARKING_MODEL = LIBRARY_MODEL + '\nopen("executed.marker", "w").close()\n'
# An editor diagram the loader refuses: it has no type.
UNTYPED_DIAGRAM = {"title": "Untyped", "model": {"elements": {}, "relationships": {}}}


def skipped(result) -> dict:
    return {entry["file"]: entry["reason"] for entry in result["skipped"]}


def test_a_directory_of_json_models_is_checked(besser, write):
    write("models/library.json", PROJECT_EXPORT)
    write("models/book.json", CLASS_DIAGRAM)
    run = besser("validate", "models")

    assert run.returncode == 0
    assert "models/book.json: 1 model\n" in run.stdout and "models/library.json: 1 model\n" in run.stdout
    assert run.stdout.rstrip().endswith("OK: 0 errors, 0 warnings")


def test_files_are_checked_in_sorted_order_and_nested_directories_are_searched(besser, write):
    write("models/b.json", CLASS_DIAGRAM)
    write("models/a.json", CLASS_DIAGRAM)
    write("models/sub/deeper/c.json", PROJECT_EXPORT)
    write("models/sub/d.json", CLASS_DIAGRAM)
    result = besser("validate", "models", "--json").json

    assert result["model_files"] == ["models/a.json", "models/b.json", "models/sub/d.json", "models/sub/deeper/c.json"]


def test_python_files_found_are_skipped_and_not_executed_without_exec_py(besser, write, tmp_path):
    write("models/library.json", PROJECT_EXPORT)
    write("models/shop.py", MARKING_MODEL)
    run = besser("validate", "models", "--json")

    assert run.returncode == 0
    assert not (tmp_path / "executed.marker").exists()
    assert run.json["model_files"] == ["models/library.json"]
    assert skipped(run.json) == {"models/shop.py": SKIP_PY}
    assert any("--exec-py" in note for note in run.json["notes"])

    text = besser("validate", "models")
    assert f"skipped: models/shop.py: {SKIP_PY}" in text.stdout
    assert "note: 1 .py file found in a directory skipped" in text.stderr


def test_exec_py_loads_the_python_files_found(besser, write, tmp_path):
    write("models/library.json", PROJECT_EXPORT)
    write("models/shop.py", MARKING_MODEL)
    run = besser("validate", "models", "--exec-py", "--json")

    assert run.returncode == 0
    assert (tmp_path / "executed.marker").exists()
    assert run.json["model_files"] == ["models/library.json", "models/shop.py"]
    assert run.json["skipped"] == [] and run.json["notes"] == []


def test_a_python_file_named_on_the_command_line_is_executed_without_exec_py(besser, write, tmp_path):
    write("models/library.json", PROJECT_EXPORT)
    model = write("shop.py", MARKING_MODEL)

    assert besser("validate", "models", model).returncode == 0
    assert (tmp_path / "executed.marker").exists()


def test_a_python_file_found_without_a_model_is_skipped(besser, write):
    write("models/library.py", LIBRARY_MODEL)
    write("models/helpers.py", "VALUE = 1\n")
    run = besser("validate", "models", "--exec-py", "--json")

    assert run.returncode == 0
    assert run.json["model_files"] == ["models/library.py"]
    assert skipped(run.json) == {"models/helpers.py": SKIP_NO_MODEL}


def test_json_files_that_are_not_editor_exports_are_skipped(besser, write):
    write("app/model.json", CLASS_DIAGRAM)
    write("app/package.json", {"name": "app", "version": "1.0.0"})
    write("app/tsconfig.json", '{\n  // comments: not JSON\n  "compilerOptions": {}\n}\n')
    run = besser("validate", "app", "--json")

    assert run.returncode == 0
    assert run.json["model_files"] == ["app/model.json"]
    assert skipped(run.json) == {"app/package.json": SKIP_NOT_A_MODEL, "app/tsconfig.json": SKIP_NOT_JSON}


def test_an_editor_export_found_that_does_not_load_fails_the_run(besser, write):
    write("models/good.json", CLASS_DIAGRAM)
    write("models/untyped.json", UNTYPED_DIAGRAM)
    run = besser("validate", "models", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == ["models/untyped.json: the diagram has no type (model.type)"]
    assert [(entry["file"], entry["exit_code"]) for entry in run.json["results"]] == [
        ("models/good.json", 0), ("models/untyped.json", 2),
    ]


def test_hidden_and_tool_directories_are_not_searched(besser, write):
    write("project/model.json", CLASS_DIAGRAM)
    write("project/.git/config.json", PROJECT_EXPORT)
    write("project/node_modules/pkg/project.json", PROJECT_EXPORT)
    write("project/venv/lib/project.json", PROJECT_EXPORT)
    write("project/__pycache__/project.json", PROJECT_EXPORT)

    result = besser("validate", "project", "--json").json
    assert result["model_files"] == ["project/model.json"] and result["skipped"] == []


def test_the_current_directory_names_files_without_a_prefix(besser, write):
    write("model.json", CLASS_DIAGRAM)

    assert besser("validate", ".", "--json").json["model_files"] == ["model.json"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_symbolic_links_out_of_the_directory_are_not_followed(besser, write, tmp_path):
    write("outside/secret.json", PROJECT_EXPORT)
    write("outside/more/other.json", PROJECT_EXPORT)
    write("models/library.json", CLASS_DIAGRAM)
    write("models/inner/real.json", CLASS_DIAGRAM)
    try:
        (tmp_path / "models/link.json").symlink_to(tmp_path / "outside/secret.json")
        (tmp_path / "models/linked_dir").symlink_to(tmp_path / "outside/more", target_is_directory=True)
        (tmp_path / "models/alias.json").symlink_to(tmp_path / "models/inner/real.json")
    except OSError:
        pytest.skip("cannot create symbolic links here")
    run = besser("validate", "models", "--json")

    assert run.returncode == 0
    # A link to a file inside the directory counts, once: the file it points to is the same.
    assert run.json["model_files"] == ["models/alias.json", "models/library.json"]
    assert skipped(run.json) == {
        "models/link.json": SKIP_OUTSIDE, "models/linked_dir": SKIP_LINKED_DIRECTORY,
    }


def test_a_file_named_and_found_in_a_directory_is_checked_once(besser, write):
    write("models/library.json", CLASS_DIAGRAM)

    assert besser("validate", "models/library.json", "models", "--json").json["model_files"] == [
        "models/library.json",
    ]


def test_explicit_files_and_directories_mix(besser, write, model):
    write("models/library.json", PROJECT_EXPORT)
    run = besser("validate", model, "models", "--json")

    assert run.returncode == 0
    assert run.json["model_files"] == ["library.py", "models/library.json"]
    assert [report["file"] for report in run.json["models"]] == ["library.py", "models/library.json"]


def test_an_empty_directory_is_a_usage_error(besser, tmp_path):
    (tmp_path / "empty").mkdir()
    run = besser("validate", "empty")

    assert run.returncode == 2
    assert run.stderr == "error: nothing to validate: no model file found in empty\n"


def test_a_directory_of_python_files_only_says_to_pass_exec_py(besser, write):
    write("models/library.py", LIBRARY_MODEL)
    run = besser("validate", "models", "--json")

    assert run.returncode == 2
    assert run.json["errors"] == [
        "nothing to validate: no model file found in models (1 .py file skipped; pass --exec-py to load them)"
    ]
    assert skipped(run.json) == {"models/library.py": SKIP_PY}


@pytest.mark.parametrize(("files", "exit_code"), [
    ({"a.py": LIBRARY_MODEL, "b.py": WARNING_MODEL}, 0),
    ({"a.py": LIBRARY_MODEL, "b.py": INVALID_MODEL}, 1),
    ({"a.py": INVALID_MODEL, "b.json": UNTYPED_DIAGRAM}, 2),
    ({"a.py": LIBRARY_MODEL, "b.py": "raise RuntimeError('broken')\n"}, 2),
])
def test_the_exit_status_is_the_worst_files(besser, write, files, exit_code):
    for name, content in files.items():
        write(f"models/{name}", content)
    run = besser("validate", "models", "--exec-py", "--json")

    assert run.returncode == exit_code == run.json["exit_code"]
    assert run.json["ok"] is (exit_code == 0)
    assert max(entry["exit_code"] for entry in run.json["results"]) == exit_code


def test_the_json_output_has_one_result_per_file(besser, write):
    write("models/good.py", LIBRARY_MODEL)
    write("models/broken.py", INVALID_MODEL)
    write("models/notes.json", {"todo": []})
    result = besser("validate", "models", "--exec-py", "--json").json

    broken, good = result["results"]
    assert set(good) == {"file", "ok", "exit_code", "errors", "warnings", "models"}
    assert (good["file"], good["ok"], good["exit_code"], good["errors"]) == ("models/good.py", True, 0, [])
    assert [report["name"] for report in good["models"]] == ["library_model"]
    assert (broken["file"], broken["ok"], broken["exit_code"]) == ("models/broken.py", False, 1)
    [error] = broken["errors"]
    assert error.startswith("broken_model: Attribute 'title' of class 'Book' has type 'Text'")
    # The run's messages name the file, as with several files.
    assert result["errors"] == [f"models/broken.py: {error}"]
    assert result["models"] == broken["models"] + good["models"]
    assert result["skipped"] == [{"file": "models/notes.json", "reason": SKIP_NOT_A_MODEL}]


def test_strict_fails_a_directory_with_warnings_in_one_file(besser, write):
    write("models/good.py", LIBRARY_MODEL)
    write("models/shelf.py", WARNING_MODEL)

    assert besser("validate", "models", "--exec-py").returncode == 0
    run = besser("validate", "models", "--exec-py", "--strict", "--json")
    assert run.returncode == 1
    assert [(entry["file"], entry["exit_code"]) for entry in run.json["results"]] == [
        ("models/good.py", 0), ("models/shelf.py", 1),
    ]
    assert run.json["warnings"][0].startswith("models/shelf.py: library_model: ")
