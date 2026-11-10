"""``besser new``: a starter model file the other commands accept as it is."""


def test_it_writes_a_model_that_validates_without_warnings(besser, tmp_path):
    run = besser("new", "model.py")

    assert run.returncode == 0
    assert run.stdout.startswith("Wrote a starter model to model.py. Next:\n  besser validate model.py\n")
    assert "besser validate model.py" in (tmp_path / "model.py").read_text()
    assert besser("validate", "model.py", "--strict").returncode == 0


def test_the_starter_model_is_one_the_generators_run_on(besser, tmp_path):
    besser("new", "model.py")
    run = besser("generate", "backend", "model.py", "-o", "app", "--json")

    assert run.returncode == 0 and run.json["warnings"] == []
    assert "app/main_api.py" in run.json["files"]


def test_json_lists_the_file(besser):
    result = besser("new", "models/shop.py", "--json").json

    assert result == {
        "ok": True, "command": "new", "exit_code": 0, "errors": [], "warnings": [], "notes": [], "files": ["models/shop.py"],
        "model": "models/shop.py",
    }


def test_an_existing_file_is_kept(besser, write, tmp_path):
    write("model.py", "mine = 1\n")
    run = besser("new", "model.py")

    assert run.returncode == 2
    assert run.stderr == "error: model.py exists: pass --force to overwrite it\n"
    assert (tmp_path / "model.py").read_text() == "mine = 1\n"


def test_force_overwrites_an_existing_file(besser, write, tmp_path):
    write("model.py", "mine = 1\n")

    assert besser("new", "model.py", "--force").returncode == 0
    assert "DomainModel(" in (tmp_path / "model.py").read_text()


def test_a_name_that_is_not_a_python_file_is_a_usage_error(besser, tmp_path):
    run = besser("new", "model.json")

    assert run.returncode == 2
    assert run.stderr == "error: a B-UML model is a .py file; got model.json\n"
    assert not (tmp_path / "model.json").exists()
