"""The API tests a ``besser init`` project gets: generated with the backend, run by ``besser apply``.

The point is what a coding agent no longer writes: the tests of the generated create, read, update and
delete routes, and the fixture its own tests start from.
"""

import importlib.util
import re
import subprocess
import sys

import pytest

from besser.cli.starters import starter_source

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("pytest", "fastapi", "httpx")),
    reason="the generated API tests need pytest, fastapi and httpx",
)

# One line, ending with what pytest said: "tests: 3 passed in 1.92s".
_APPLY_OK = re.compile(r"^OK: 1 output, 0 failed: \[backend\] .*; tests: (\d+) passed in [\d.]+s\n$")

USER_TEST = '''\
def test_a_library_is_listed_after_it_is_created(client):
    body = client.post("/library/", json={"name": "City", "address": "Main St"}).json()
    library = body.get("library", body)
    assert [item["id"] for item in client.get("/library/").json()] == [library["id"]]
'''


def _passed(run) -> int:
    match = _APPLY_OK.match(run.stdout)
    assert match, run.stdout + run.stderr
    return int(match.group(1))


def test_init_generates_the_api_tests_and_apply_runs_them(besser, tmp_path):
    init = besser("init", output=None)
    applied = besser("apply", output=None)

    config = (tmp_path / "besser.toml").read_text()
    assert 'test = "python -m pytest -q"' in config.splitlines()[1]
    assert "options = { tests = true, uniform_responses = true }" in config
    assert init.stdout.startswith("Set up .: wrote model.py, besser.toml, conftest.py, AGENTS.md")
    assert (tmp_path / "conftest.py").read_text().endswith("from backend.tests.conftest import client  # noqa: E402,F401\n")
    # The starter's Library, Book and Author: written by nobody but besser.
    assert applied.returncode == 0 and _passed(applied) == 3
    assert {path.name for path in (tmp_path / "backend" / "tests").glob("*.py")} == {
        "conftest.py", "test_generated_api.py"}


def test_a_test_of_the_project_uses_the_client_fixture_with_no_setup(besser, write, tmp_path):
    besser("init")
    write("tests/test_library.py", USER_TEST)
    applied = besser("apply", output=None)
    plain = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                           cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False)

    assert _passed(applied) == 4
    assert plain.returncode == 0 and "1 passed" in plain.stdout, plain.stdout + plain.stderr


def test_the_tests_follow_the_model_through_add_rename_and_remove(besser, tmp_path):
    besser("init")
    besser("apply")
    tests = tmp_path / "backend" / "tests" / "test_generated_api.py"

    assert besser("model", "add-class", "Shelf", "label:str", "capacity:int").returncode == 0
    added = besser("apply", output=None)
    assert _passed(added) == 4 and "def test_shelf(client):" in tests.read_text()

    assert besser("model", "rename", "Shelf", "Rack").returncode == 0
    renamed = besser("apply", output=None)
    assert _passed(renamed) == 4
    assert "def test_rack(client):" in tests.read_text() and "shelf" not in tests.read_text()

    assert besser("model", "remove", "Rack").returncode == 0
    removed = besser("apply", output=None)
    assert _passed(removed) == 3 and "rack" not in tests.read_text()


def test_init_for_a_gui_project_tests_the_web_app_s_backend(besser, tmp_path):
    besser("init", "--kind", "gui")
    applied = besser("apply", output=None)

    assert "options = { tests = true, uniform_responses = true }" in (tmp_path / "besser.toml").read_text()
    assert "from web_app.backend.tests.conftest import client" in (tmp_path / "conftest.py").read_text()
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assert re.search(r"; tests: 2 passed in [\d.]+s$", applied.stdout.strip()), applied.stdout


@pytest.mark.parametrize("kind", ["domain", "gui", "project"])
def test_the_generated_tests_pass_on_each_starter(kind, besser, write, tmp_path):
    write("model.py", starter_source(kind, "model.py"))
    generated = besser("generate", "backend", "model.py", "-o", "api", "--option", "tests=true")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "api"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False)

    assert generated.returncode == 0, generated.stderr
    assert run.returncode == 0 and " passed" in run.stdout and "skipped" not in run.stdout, run.stdout


# Runs 'besser init' with pytest and fastapi unimportable, as on a fresh install without them.
_INIT_WITHOUT_TEST_PACKAGES = (
    "import sys; sys.modules['pytest'] = sys.modules['fastapi'] = None; "
    "from besser.cli.main import main; sys.exit(main(['init']))"
)


def test_init_without_the_test_packages_leaves_test_commented_and_says_what_to_install(besser, tmp_path):
    init = subprocess.run([sys.executable, "-c", _INIT_WITHOUT_TEST_PACKAGES], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120, check=False)
    applied = besser("apply", output=None)

    assert init.returncode == 0, init.stderr
    assert "to run the generated API tests: pip install pytest fastapi, then uncomment 'test' in besser.toml" in (
        init.stdout)
    config = (tmp_path / "besser.toml").read_text()
    assert config.splitlines()[1].startswith('# test = "python -m pytest -q"')
    assert "options = { tests = true, uniform_responses = true }" in config
    # The first apply builds and does not fail on a package the user did not choose.
    assert applied.returncode == 0
    assert applied.stdout.rstrip("\n").endswith("; tests: none (set 'test' in besser.toml)")


def test_the_project_conftest_says_what_to_run_before_the_first_build(besser, tmp_path):
    besser("init")
    before = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=300, check=False)

    assert before.returncode != 0
    assert ("backend/tests/conftest.py (the client fixture) is written by 'besser apply' from the output "
            "'backend' of besser.toml (option tests = true): run it first") in before.stdout + before.stderr
    assert _passed(besser("apply", output=None)) == 3


def test_init_in_a_project_whose_backend_has_no_tests_writes_no_conftest_and_says_how(besser, write, tmp_path):
    """A conftest.py importing a fixture the besser.toml in place does not generate breaks every test."""
    write("model.py", starter_source("domain", "model.py"))
    write("besser.toml", 'model = "model.py"\ntest = "python -m pytest -q"\n\n'
                         '[output.backend]\ngenerator = "backend"\ndir = "backend"\n')
    write("tests/test_mine.py", "def test_mine():\n    assert True\n")

    init = besser("init", "--json")
    applied = besser("apply", output=None)

    assert not (tmp_path / "conftest.py").exists()
    assert "no conftest.py: no output of besser.toml generates the API tests and their client fixture " \
           "(options = { tests = true } on a backend or web_app output)" in init.json["notes"]
    assert applied.returncode == 0 and "tests: 1 passed" in applied.stdout, applied.stdout + applied.stderr


def test_init_points_the_conftest_at_the_dir_of_besser_toml(besser, write, tmp_path):
    write("besser.toml", 'model = "model.py"\ntest = "python -m pytest -q"\n\n[output.api]\ngenerator = "backend"\n'
                         'dir = "api"\noptions = { tests = true }\n')
    besser("init")

    assert (tmp_path / "conftest.py").read_text().endswith("from api.tests.conftest import client  # noqa: E402,F401\n")
    applied = besser("apply", output=None)
    assert applied.returncode == 0 and "tests: 3 passed" in applied.stdout, applied.stdout + applied.stderr


def test_init_keeps_a_conftest_and_says_what_it_lacks(besser, write, tmp_path):
    besser("init")
    again = besser("init", "--json")
    write("conftest.py", "import pytest\n")
    mine = besser("init", "--json")

    assert "kept conftest.py: it exists" in again.json["notes"]
    assert ("kept conftest.py: it exists; for the generated client fixture add "
            "'from backend.tests.conftest import client' to it") in mine.json["notes"]
