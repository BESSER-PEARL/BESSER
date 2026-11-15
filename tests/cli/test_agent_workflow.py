"""BESSER's defaults for coding agents: terse output off a terminal, the agent files, init, guide and apply."""

import importlib.util
import json
import re
import sys
import time

import pytest

from besser.cli.agent_files import GUI_GUIDE, GUIDE, guide_text

needs_pygit2 = pytest.mark.skipif(importlib.util.find_spec("pygit2") is None,
                                  reason='regenerate merges with pygit2: pip install "besser[regen]"')

# What 'apply' runs: a test command that passes, or fails after printing a line, without a shell.
PASSING_TEST = f'test = "{sys.executable} -c \\"print(\'3 passed\')\\""\n'
FAILING_TEST = f'test = "{sys.executable} -c \\"print(\'1 failed\'); raise SystemExit(3)\\""\n'


def _config(write, test: str = "") -> None:
    write("besser.toml", f'model = "library.py"\n{test}\n[output.api]\ngenerator = "backend"\ndir = "api"\n')


# ── terse output ────────────────────────────────────────────────────────────


def test_a_command_that_writes_files_prints_one_line_off_a_terminal(besser, model):
    run = besser("generate", "backend", model, "-o", "api", output=None)

    assert run.returncode == 0
    assert run.stdout == "Generated 9 files in api with the 'backend' generator.\n"
    assert run.stderr == ""  # no note about the record, nothing the generator printed


def test_a_failure_keeps_its_report_off_a_terminal(besser, write):
    write("broken.py", "from besser.BUML.metamodel.structural import DomainModel\nm = DomainModel(name='')\n")
    run = besser("validate", "broken.py", output=None)

    assert run.returncode == 2 and "broken.py" in run.stderr


def test_a_command_run_for_its_output_prints_all_of_it(besser, model):
    run = besser("info", model, output=None)

    assert "class Book: title:str, pages:int" in run.stdout


def test_besser_output_full_prints_the_report_and_an_unknown_value_is_a_usage_error(besser, model):
    full = besser("generate", "backend", model, "-o", "api", output="full")
    unknown = besser("generators", output="loud")

    assert "  api/main_api.py" in full.stdout and "note: " in full.stderr
    assert unknown.returncode == 2
    assert unknown.stderr.startswith("error: BESSER_OUTPUT must be 'full' or 'terse', not 'loud'")


def test_json_is_complete_off_a_terminal(besser, model):
    result = besser("generate", "backend", model, "-o", "api", "--json", output=None).json

    assert "api/main_api.py" in result["files"] and result["notes"]


# ── new, init and the agent files ───────────────────────────────────────────


def test_new_writes_the_guide_and_the_ignore_rules_beside_the_model_of_a_project(besser, write, tmp_path):
    _config(write)
    run = besser("new", "model.py", output=None)

    assert run.stdout == ("Wrote a starter domain model to model.py, and the agent guide: AGENTS.md, CLAUDE.md, "
                          ".ignore.\n")
    # The guide of this project: what 'besser guide' prints in it.
    assert besser("guide", output=None).stdout in (tmp_path / "AGENTS.md").read_text()
    assert (tmp_path / "CLAUDE.md").read_text() == "@AGENTS.md\n"
    assert "**/.besser_generated.json" in (tmp_path / ".ignore").read_text()


def test_new_without_agent_files_writes_only_the_model(besser, write, tmp_path):
    _config(write)
    run = besser("new", "model.py", "--no-agent-files", "--json")

    assert run.json["files"] == ["model.py"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["besser.toml", "model.py"]


def test_new_outside_a_project_writes_no_guide_for_commands_that_would_fail(besser, tmp_path):
    run = besser("new", "crm.py", "--json")
    edit = besser("model", "add-class", "Invoice", "number:str")

    assert run.returncode == 0 and run.json["files"] == ["crm.py"]
    assert run.json["notes"] == ["no agent guide: . has no besser.toml for it to describe ('besser init' sets one up)"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["crm.py"]
    assert edit.returncode == 2
    assert edit.stderr == "error: model.py: no such file; name the model file with -f FILE\n"


@pytest.mark.parametrize("kind", ["state_machine", "feature"])
def test_init_of_a_kind_without_a_build_writes_no_guide_about_apply(besser, kind, tmp_path):
    run = besser("init", "--kind", kind, "--json")

    assert run.returncode == 0 and run.json["files"] == ["model.py"]
    assert not (tmp_path / "AGENTS.md").exists() and not (tmp_path / "besser.toml").exists()
    assert "no agent guide: . has no besser.toml for it to describe" in run.json["notes"][-1]


def test_what_people_wrote_in_the_agent_files_is_kept(besser, write, tmp_path):
    _config(write)
    write("AGENTS.md", "# Shop\n\nRun the shop with make.\n")
    write("CLAUDE.md", "Be brief.\n")
    write(".ignore", "dist/\n")
    besser("new", "model.py")
    again = besser("new", "other.py", "--json")

    agents = (tmp_path / "AGENTS.md").read_text()
    assert agents.startswith("# Shop\n\nRun the shop with make.\n\n<!-- besser-guide:begin -->")
    assert agents.count("besser-guide:begin") == 1
    assert (tmp_path / "CLAUDE.md").read_text() == "Be brief.\n\n@AGENTS.md\n"
    assert (tmp_path / ".ignore").read_text().startswith("dist/\n\n# besser:begin\n")
    assert again.json["files"] == ["other.py"]  # the guide was up to date


def test_init_sets_up_a_project_that_apply_builds(besser, tmp_path):
    run = besser("init", "shop", output=None)
    applied = besser("apply", "-c", "shop/besser.toml", output=None)

    assert run.returncode == 0
    assert run.stdout == ("Set up shop: wrote shop/model.py, shop/besser.toml, shop/conftest.py, shop/AGENTS.md, "
                          "shop/CLAUDE.md, shop/.ignore.\n")
    assert (tmp_path / "shop" / ".ignore").read_text() == (
        "# besser:begin\n# Generated by BESSER (besser.toml): a black box, see AGENTS.md.\n"
        "/backend/*\n!/backend/extensions.py\n**/.besser_generated.json\n# besser:end\n")
    assert applied.returncode == 0
    # The generated API tests run (tests/cli/test_api_tests.py has the rest of that story).
    assert re.fullmatch(r"OK: 1 output, 0 failed: \[backend\] Generated 12 files in shop/backend with the "
                        r"'backend' generator; tests: 3 passed in [\d.]+s\n", applied.stdout), applied.stdout
    assert (tmp_path / "shop" / "backend" / "main_api.py").is_file()


@pytest.mark.parametrize("kind, generator", [("domain", "backend"), ("gui", "web_app")])
def test_init_turns_on_uniform_responses(besser, tmp_path, kind, generator):
    assert besser("init", "--kind", kind).returncode == 0

    assert (tmp_path / "besser.toml").read_text().endswith(
        f'[output.{generator}]\ngenerator = "{generator}"\ndir = "{generator}"\n'
        "options = { tests = true, uniform_responses = true }\n")


# A class of the starter model without a to-many end answers POST and PUT like GET, in the new project.
UNIFORM_ANSWERS_TEST = """\
def test_answers(client):
    deal = client.post("/deal/", json={"amount": 2}).json()
    assert deal == {"deal": {"id": deal["deal"]["id"], "amount": 2.0}}
    deal_id = deal["deal"]["id"]
    assert client.put(f"/deal/{deal_id}/", json={"amount": 3}).json() == client.get(f"/deal/{deal_id}/").json()
    library = client.post("/library/", json={"name": "City", "address": "Main St"}).json()
    assert set(library) == {"library", "books_ids"}
"""


def test_apply_on_a_new_project_builds_a_backend_whose_post_answers_like_get(besser, write, tmp_path):
    pytest.importorskip("fastapi.testclient")
    besser("init")
    assert besser("model", "add-class", "Deal", "amount:float").returncode == 0
    config = (tmp_path / "besser.toml").read_text()
    write("besser.toml", config.replace('test = "python -m pytest -q"',
                                        f'test = "{sys.executable} -m pytest -q -p no:cacheprovider"'))
    # The project's conftest.py, written by init, gives this test the generated `client` fixture.
    write("tests/test_answers.py", UNIFORM_ANSWERS_TEST)

    run = besser("apply", output=None)

    assert run.returncode == 0, run.stdout + run.stderr
    # The generated test of each class (Library, Book, Author, Deal) and the one written here.
    assert "tests: 5 passed" in run.stdout


def test_init_keeps_a_model_and_a_configuration_that_exist(besser, write, tmp_path):
    write("model.py", "mine = 1\n")
    run = besser("init", "--json")

    assert run.json["files"] == ["besser.toml", "conftest.py", "AGENTS.md", "CLAUDE.md", ".ignore"]
    assert run.json["notes"] == ["kept model.py: it exists"]
    assert (tmp_path / "model.py").read_text() == "mine = 1\n"


def test_guide_prints_the_rules_in_little_space(besser):
    run = besser("guide", output=None)

    assert run.stdout == GUIDE
    # Re-read on every turn of every session, so it stays small.
    assert len(GUIDE.encode("utf-8")) < 4200 and len(GUIDE.splitlines()) <= 60


def test_the_guide_has_the_lines_on_pages_only_in_a_project_that_builds_a_gui(besser, tmp_path):
    assert besser("init", "--kind", "gui").returncode == 0
    run = besser("guide", output=None)

    # The project's own file for hand-written backend code is named by its path. Before: "extensions.py
    # beside main_api.py", and a small model wrote it in the project folder, where nothing loads it.
    guide = guide_text(gui=True).replace("`extensions.py` beside `main_api.py`", "`web_app/backend/extensions.py`")
    assert "`web_app/backend/extensions.py`, with" in guide
    assert run.stdout == guide and GUI_GUIDE in run.stdout and "add-screen" not in GUIDE
    assert guide in (tmp_path / "AGENTS.md").read_text()
    # What a project with a web_app output pays for them, besides the guide's budget above.
    assert len(GUI_GUIDE.encode("utf-8")) < 1600 and len(GUI_GUIDE.splitlines()) <= 17


# What the guide says the generated backend answers, as tests that use the guide's own conftest.py.
GUIDE_ANSWERS_TEST = """\
def test_answers(client):
    library = client.post("/library/", json={"name": "City", "address": "Main St"}).json()
    assert set(library) == {"library", "books_ids"}  # one-to-many: "<end>_ids"
    author = client.post("/author/", json={"name": "Ada", "email": "ada@x.org"}).json()["author"]
    library_id = library["library"]["id"]
    book = {"title": "T", "pages": 3, "release": "2024-01-31", "library": library_id}
    assert client.post("/book/", json=book).status_code == 422  # a required end missing
    assert client.post("/book/", json={**book, "authors": []}).status_code == 400
    assert client.post("/book/", json={**book, "authors": [999]}).status_code == 404  # an unknown id in a list
    assert client.post("/book/", json={**book, "library": 999, "authors": [author["id"]]}).status_code == 400
    created = client.post("/book/", json={**book, "authors": [author["id"]]}).json()
    assert set(created) == {"book", "author_ids"}  # many-to-many: "<other class>_ids"
    assert created["book"]["library_id"] == library_id  # a to-one end: "<end>_id"
    deal = client.post("/deal/", json={"amount": 2}).json()  # no to-many end: the object
    assert deal == {"id": deal["id"], "amount": 2.0}
    assert client.put(f"/deal/{deal['id']}/", json={"amount": 3}).json()["amount"] == 3.0
    assert client.get(f"/deal/{deal['id']}/").json() == {"deal": {"id": deal["id"], "amount": 3.0}}
    assert client.get("/deal/").json() == [{"id": deal["id"], "amount": 3.0}]
    for body, name in ((deal, "deal"), (created, "book")):  # the guide's way to read either answer
        assert body.get(name, body)["id"] > 0
"""


def test_the_guides_test_fixture_runs_and_its_answers_hold_on_a_generated_backend(besser, write, tmp_path):
    pytest.importorskip("fastapi.testclient")
    besser("init")
    (tmp_path / "conftest.py").unlink()  # init's fixture, from the generated tests this besser.toml turns off
    assert besser("model", "add-class", "Deal", "amount:float").returncode == 0
    write("besser.toml", f'model = "model.py"\ntest = "{sys.executable} -m pytest -q -p no:cacheprovider"\n\n'
                         '[output.backend]\ngenerator = "backend"\ndir = "backend"\n')
    # With both options off the project reads the guide's other lines: a fixture to write, two answer shapes.
    guide = besser("guide", output=None).stdout
    plain = guide_text(tests=False, uniform_responses=False)
    # The guide of a project names the extension file by its path there.
    assert guide == plain.replace("`extensions.py` beside `main_api.py`", "`backend/extensions.py`") != GUIDE
    assert 'read it as `body.get("x", body)`' in guide and "generated tests" not in guide
    write("tests/conftest.py", guide.split("```python\n")[1].split("```")[0])
    write("tests/test_answers.py", GUIDE_ANSWERS_TEST)

    run = besser("apply", output=None)

    assert run.returncode == 0, run.stdout + run.stderr
    assert "tests: 1 passed" in run.stdout
    assert "400 when\nempty); an unknown id is 400 (404 in a list)" in GUIDE  # what the test above checks


def test_a_build_keeps_the_guide_in_agents_md_in_step_with_the_options(besser, write, tmp_path):
    """AGENTS.md is what an agent reads: turning the generated tests off must not leave it telling the agent
    to use their fixture."""
    besser("init")
    (tmp_path / "conftest.py").unlink()
    config = (tmp_path / "besser.toml").read_text()
    write("besser.toml", config.replace("tests = true, uniform_responses = true", "tests = false"))

    run = besser("build", "--json")

    agents = (tmp_path / "AGENTS.md").read_text()
    expected = guide_text(tests=False, uniform_responses=False).replace(
        "`extensions.py` beside `main_api.py`", "`backend/extensions.py`")
    assert expected in agents and GUIDE not in agents
    assert "updated AGENTS.md: the guide follows the outputs of besser.toml" in run.json["notes"]
    assert besser("guide", output=None).stdout in agents


def test_the_guide_of_a_new_project_is_the_short_one(besser, tmp_path):
    """'besser init' turns the generated tests and the uniform answers on, so its guide needs neither the
    fixture recipe nor the two answer shapes."""
    assert besser("init").returncode == 0

    guide = besser("guide", output=None).stdout

    # GUIDE, with the project's file for hand-written backend code named by its path.
    assert guide == GUIDE.replace("`extensions.py` beside `main_api.py`", "`backend/extensions.py`")
    assert guide in (tmp_path / "AGENTS.md").read_text()
    assert "`GET /x/{x_id}/`, `POST` and `PUT` answer" in guide and "```python" not in guide
    assert "generated tests of every" in guide and "body.get" not in guide


@needs_pygit2
def test_build_keeps_the_ignore_rules_in_step_with_the_outputs(besser, model, write, tmp_path):
    besser("init")  # its besser.toml builds 'backend'; the rules name that folder
    _config(write)
    build = besser("build", "--json")

    assert ".ignore" in build.json["files"]
    assert "/api/*\n!/api/extensions.py\n" in (tmp_path / ".ignore").read_text()
    assert ".ignore" not in besser("build", "--json").json["files"]


@needs_pygit2
def test_build_leaves_an_ignore_file_without_a_besser_block_alone(besser, model, write, tmp_path):
    _config(write)
    write(".ignore", "dist/\n")
    besser("build")

    assert (tmp_path / ".ignore").read_text() == "dist/\n"


# ── apply ───────────────────────────────────────────────────────────────────


@needs_pygit2
def test_apply_builds_and_runs_the_tests_in_one_line(besser, model, write):
    _config(write, PASSING_TEST)
    run = besser("apply", output=None)
    again = besser("apply", "--json")

    assert run.returncode == 0
    assert run.stdout == ("OK: 1 output, 0 failed: [api] Generated 9 files in api with the 'backend' generator; "
                          "tests: 3 passed\n")
    assert again.json["command"] == "apply" and again.json["test"]["exit_code"] == 0
    assert again.json["test"]["output_tail"] == ["3 passed"]


@needs_pygit2
def test_apply_prints_the_end_of_the_output_of_tests_that_fail(besser, model, write):
    _config(write, FAILING_TEST)
    run = besser("apply", output=None)

    assert run.returncode == 1
    assert run.stdout.startswith("1 failed\n")
    assert run.stdout.splitlines()[-1].startswith("FAILED: tests (exit status 3); build OK")
    assert run.stderr == ""  # the FAILED line says it once; --json has it in errors


@needs_pygit2
def test_apply_says_once_that_the_tests_failed(besser, model, write):
    _config(write, FAILING_TEST)
    run = besser("apply", output=None)
    as_json = besser("apply", "--json")

    assert run.stderr == "" and run.stdout.count("exit status 3") == 1
    assert as_json.json["errors"][0].startswith("the tests failed (exit status 3): ")


@needs_pygit2
def test_apply_prints_only_the_failing_tests_of_pytest(besser, model, write):
    write("test_app.py", "def test_ok():\n    pass\n\n\ndef test_broken():\n    assert 1 == 2\n")
    _config(write, f'test = "{sys.executable} -m pytest -q -p no:cacheprovider test_app.py"\n')
    run = besser("apply", output=None)

    assert run.returncode == 1
    assert run.stdout.splitlines()[0] == "FAILED test_app.py::test_broken - assert 1 == 2"
    assert "def test_broken" not in run.stdout  # not the traceback
    assert run.stdout.splitlines()[1].startswith("1 failed, 1 passed")


@needs_pygit2
def test_apply_prints_why_a_unittest_test_failed_in_one_call(besser, model, write):
    write("tests/__init__.py", "")
    write("tests/test_app.py", """
        import unittest

        class AppTest(unittest.TestCase):
            def test_ok(self):
                pass

            def test_broken(self):
                self.assertEqual({"a": 1}["a"], 2)

            def test_raises(self):
                raise KeyError("missing")

            def test_raises_while_handling(self):
                try:
                    {}["k"]
                except KeyError:
                    raise ValueError("bad value")
    """)
    _config(write, 'test = "python -m unittest discover -s tests -t ."\n')
    run = besser("apply", output=None)

    assert run.returncode == 1
    assert run.stdout.splitlines()[:7] == [
        "ERROR: test_raises (tests.test_app.AppTest.test_raises)",
        "    KeyError: 'missing'",
        "ERROR: test_raises_while_handling (tests.test_app.AppTest.test_raises_while_handling)",
        "    ValueError: bad value",  # what the test raised, not the exception it was handling
        "FAIL: test_broken (tests.test_app.AppTest.test_broken)",
        "    AssertionError: 1 != 2",
        "FAILED (failures=1, errors=2)",
    ]


def test_apply_runs_no_test_after_a_build_that_fails(besser, write):
    write("library.py", "raise ValueError('broken')\n")
    _config(write, FAILING_TEST)
    run = besser("apply", "--json")

    assert run.returncode == 2 and run.json["test"] is None
    assert "ValueError: broken" in run.json["errors"][0]


def test_apply_whose_test_command_cannot_run_still_reports_what_the_build_wrote(besser, model, write):
    _config(write, 'test = "no-such-test-runner -q"\n')
    run = besser("apply", "--json")
    text = besser("apply", output=None)

    assert run.returncode == 1 and run.json["exit_code"] == 1
    assert run.json["errors"] == ["cannot run the test command 'no-such-test-runner -q': "
                                  "[Errno 2] No such file or directory: 'no-such-test-runner'"]
    assert "api/main_api.py" in run.json["files"]  # written by the build: listed, as build lists them
    assert run.json["files"] == run.json["outputs"][0]["files"]
    assert text.stdout.startswith("FAILED: tests not run; build OK: 1 output, 0 failed: [api] ")


# A test command that starts a server (a child process that runs on) and then waits on it.
HANGING_TEST = """
    import subprocess, sys, time
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open("server.pid", "w") as pid_file:
        pid_file.write(str(server.pid))
    print("server started", flush=True)
    time.sleep(120)
"""


def _running(pid: int) -> bool:
    """Whether the process ``pid`` runs: it exists and is not a zombie waiting to be reaped."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rpartition(")")[2].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc to see the processes left")
def test_apply_stops_tests_that_run_past_test_timeout_and_every_process_they_started(besser, model, write,
                                                                                      tmp_path):
    write("hang.py", HANGING_TEST)
    _config(write, 'test = "python hang.py"\ntest_timeout = 1\n')
    started = time.monotonic()
    run = besser("apply", "--json")
    elapsed = time.monotonic() - started
    server = int((tmp_path / "server.pid").read_text())
    (tmp_path / "server.pid").unlink()
    text = besser("apply", output=None)

    assert elapsed < 60  # not the two minutes the test command and its server sleep
    assert not _running(server)
    assert run.returncode == 1
    assert run.json["errors"] == ["the tests timed out after 1 s (test_timeout in besser.toml) and were stopped, "
                                  "with the processes they started: python hang.py"]
    assert run.json["test"] == {"command": "python hang.py", "exit_code": None, "timed_out": True,
                                "output_tail": ["server started"], "failing": []}
    assert "api/main_api.py" in run.json["files"]  # the build's report is kept
    assert text.returncode == 1 and text.stderr == ""
    assert text.stdout.startswith("FAILED: the tests timed out after 1 s (test_timeout in besser.toml)")
    assert len(text.stdout.splitlines()) == 1
    assert not _running(int((tmp_path / "server.pid").read_text()))


def test_apply_without_a_test_command_says_so_in_its_one_line(besser, model, write):
    _config(write)
    run = besser("apply", output=None)
    as_json = besser("apply", "--json")

    assert run.returncode == 0
    assert run.stdout.rstrip("\n").endswith("; tests: none (set 'test' in besser.toml)")
    assert as_json.json["notes"][-1] == "no tests run: besser.toml has no 'test' command"


@needs_pygit2
def test_apply_that_fails_never_reports_ok(besser, model, write):
    _config(write, FAILING_TEST)
    run = besser("apply", output=None)

    last = run.stdout.splitlines()[-1]
    assert run.returncode == 1
    assert last.startswith("FAILED: tests (exit status 3)") and "OK" not in last.split(";")[0]


@needs_pygit2
def test_apply_shows_the_cause_of_a_pytest_collection_error(besser, model, write):
    _config(write, 'test = "python -m pytest -q -p no:cacheprovider tests"\n')
    write("helper.py", "def broken(:\n    pass\n")
    write("tests/test_a.py", """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import helper

        def test_a():
            assert helper
    """)
    run = besser("apply", "--json", output=None)
    text = besser("apply", output=None)

    assert run.returncode == 1 and run.json["test"]["failing"] == ["tests/test_a.py"]
    assert "helper.py\", line 1" in text.stdout and "SyntaxError" in text.stdout
    assert text.stdout.splitlines()[-1].startswith("FAILED: tests (exit status 2)")


@needs_pygit2
def test_apply_runs_the_tests_without_besser_s_standard_input(besser, model, write):
    _config(write, f'test = "python -c \\"import sys; print(repr(sys.stdin.read()) + \' passed\')\\""\n')
    run = besser("apply", output=None, stdin="secret")

    assert run.returncode == 0 and run.stdout.rstrip("\n").endswith("; tests: '' passed")


@needs_pygit2
def test_apply_reports_no_collected_test_as_none(besser, model, write):
    write("tests/__init__.py", "")
    _config(write, 'test = "python -m pytest -q -p no:cacheprovider tests"\n')
    run = besser("apply", output=None)

    assert run.returncode == 0
    assert run.stdout.rstrip("\n").endswith("; tests: none collected")


# ── what the agent reads instead of the code ────────────────────────────────


def test_info_shows_optional_attributes_defaults_constraints_and_routes(besser, write):
    write("shop.py", """
        from besser.BUML.metamodel.structural import Class, Constraint, DomainModel, IntegerType, Property, StringType
        price = Property(name="price", type=IntegerType)
        product = Class(name="Product", attributes={
            Property(name="name", type=StringType), price,
            Property(name="code", type=StringType, is_optional=True, default_value="X")})
        positive = Constraint(name="positive", context=product, language="OCL",
                              expression="context Product inv positive: self.price > 0")
        shop = DomainModel(name="Shop", types={product}, constraints={positive})
    """)
    text = besser("info", "shop.py").stdout
    product = besser("info", "shop.py", "--json").json["models"][0]
    code = next(attribute for attribute in product["classes"][0]["attributes"] if attribute["name"] == "code")

    assert "price:int" in text and "code:str?='X'" in text
    assert "constraint positive on Product: self.price > 0" in text
    assert "backend routes per class X: POST|GET /x/, GET|PUT|DELETE /x/{x_id}/" in text
    assert product["backend_endpoints"]["Product"][2] == "GET /product/{product_id}/"
    assert code == {"name": "code", "type": "str", "optional": True, "default": "X", "derived": False}


@needs_pygit2
def test_generated_files_say_they_are_generated_and_are_listed_in_the_record(besser, model, tmp_path):
    besser("generate", "backend", model, "-o", "api")

    first_line = (tmp_path / "api" / "main_api.py").read_text().splitlines()[0]
    assert first_line == ("# Generated by BESSER: edit the model, not this file. Custom code goes in "
                          "extensions.py.")
    assert (tmp_path / "api" / "requirements.txt").read_text().startswith("fastapi")  # no comment syntax known
    record = json.loads((tmp_path / "api" / ".besser_generated.json").read_text())
    assert "main_api.py" in record["files"]


@needs_pygit2
def test_adopting_output_generated_before_the_header_finds_no_edit(besser, model, tmp_path):
    besser("generate", "python", model, "-o", "out")
    (tmp_path / "out" / ".besser_generated.json").unlink()
    classes = tmp_path / "out" / "classes.py"
    classes.write_text("".join(classes.read_text().splitlines(keepends=True)[1:]))  # as an older BESSER wrote it
    run = besser("regenerate", "python", model, "-o", "out", "--adopt", "--json")

    assert run.returncode == 0 and run.json["differs_from_model"] == []


def test_the_guides_test_fixture_runs_in_a_gui_project_without_generated_tests(besser, write, tmp_path):
    """The fixture's backend folder is the output's dir for a backend, its backend/ for a web_app."""
    pytest.importorskip("fastapi.testclient")
    besser("init", "--kind", "gui")
    (tmp_path / "conftest.py").unlink()
    config = (tmp_path / "besser.toml").read_text().replace("tests = true", "tests = false")
    write("besser.toml", config.replace('test = "python -m pytest -q"',
                                        f'test = "{sys.executable} -m pytest -q -p no:cacheprovider"'))
    guide = besser("guide", output=None).stdout
    assert "(`backend`: the output's `dir`; of a\n  `web_app`, `<dir>/backend`)" in guide
    fixture = guide.split("```python\n")[1].split("```")[0]
    write("tests/conftest.py", fixture.replace("/../backend", "/../web_app/backend"))
    write("tests/test_listed.py", "def test_listed(client):\n    assert client.get('/book/').status_code == 200\n")

    run = besser("apply", output=None)

    assert run.returncode == 0 and "tests: 1 passed" in run.stdout, run.stdout + run.stderr
