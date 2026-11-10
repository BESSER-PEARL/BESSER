"""``run_generator``: the calling contract the web editor backend and the ``besser`` CLI share."""

import importlib.util
import inspect

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.registry import GENERATORS, get_generator
from besser.generators.regeneration import RECORD_FILENAME, RegenerationError
from besser.generators.runner import (
    GeneratorInputError,
    build_generator,
    check_options,
    preview_generator,
    regenerate_generator,
    run_generator,
)

# What a generator declares in ``GeneratorSpec.models``.
MODEL_KINDS = {"domain", "gui", "object", "agent", "quantum", "nn", "bpmn", "deployment", "uml_deployment"}


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _gui_model(main_pages: int = 1) -> GUIModel:
    screens = {
        Screen(name=f"Screen{i}", description="", x_dpi="x", y_dpi="y", screen_size="Small",
               view_elements=set(), is_main_page=i < main_pages)
        for i in range(2)
    }
    return GUIModel(name="shop", package="com.example.shop", versionCode="1", versionName="1.0",
                    description="", screenCompatibility=True, modules={Module(name="main", screens=screens)})


def test_it_returns_the_files_the_generator_wrote(simple_library_book_model, tmp_path):
    files = run_generator("python", {"domain": simple_library_book_model}, tmp_path / "out")

    assert files == ["classes.py"]
    assert "class Book" in (tmp_path / "out" / "classes.py").read_text()


def test_files_of_a_project_generator_are_relative_to_the_output_directory(simple_library_book_model, tmp_path):
    files = run_generator("backend", {"domain": simple_library_book_model}, tmp_path)

    assert {"main_api.py", "sql_alchemy.py", "pydantic_classes.py"} <= set(files)
    assert any(path.startswith("routers/") for path in files)
    assert all((tmp_path / path).is_file() for path in files)


def test_only_the_files_of_this_run_are_returned(simple_library_book_model, tmp_path):
    (tmp_path / "notes.txt").write_text("mine")
    run_generator("sql", {"domain": simple_library_book_model}, tmp_path)

    assert run_generator("python", {"domain": simple_library_book_model}, tmp_path) == ["classes.py"]


def test_an_option_is_a_keyword_argument_of_the_constructor(simple_library_book_model, tmp_path):
    models = {"domain": simple_library_book_model}
    [sqlite] = run_generator("sql", models, tmp_path / "sqlite")
    [postgresql] = run_generator("sql", models, tmp_path / "postgresql", {"sql_dialect": "postgresql"})

    assert (tmp_path / "sqlite" / sqlite).read_text() != (tmp_path / "postgresql" / postgresql).read_text()


def test_the_schema_default_of_an_option_is_filled_in():
    assert check_options(get_generator("django"), {"app_name": "books"}) == {
        "project_name": "myproject", "app_name": "books", "containerization": False,
    }


def test_an_option_without_a_schema_default_is_left_to_the_generator():
    assert check_options(get_generator("qiskit"), {}) == {}


def test_an_unknown_generator_names_the_available_ones(simple_library_book_model, tmp_path):
    with pytest.raises(GeneratorInputError, match="Unknown generator 'cobol'.*python"):
        run_generator("cobol", {"domain": simple_library_book_model}, tmp_path)


def test_a_missing_model_is_named_before_anything_is_written(simple_library_book_model, tmp_path):
    with pytest.raises(GeneratorInputError, match="reads a domain and a gui model; missing: gui"):
        run_generator("react", {"domain": simple_library_book_model}, tmp_path / "out")

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("generator, options, message", [
    ("python", {"dbms": "sqlite"}, r"unknown option\(s\) for the 'python' generator: dbms \(it takes no options\)"),
    ("sqlalchemy", {"dmbs": "sqlite"}, r"unknown option\(s\) for the 'sqlalchemy' generator: dmbs \(it takes dbms\)"),
    ("sqlalchemy", {"dbms": "db2"}, "dbms must be one of sqlite, postgresql, .*, got 'db2'"),
    ("pydantic", {"backend": "yes"}, "backend must be true or false, got str"),
    ("qiskit", {"shots": True}, "shots must be an integer, got bool"),
    ("django", {"app_name": 3}, "app_name must be a string, got int"),
    ("backend", {"http_methods": "GET"}, "http_methods must be a list, got str"),
    ("backend", {"http_methods": ["GET", "FETCH"]}, "each value of http_methods must be one of GET, .*, got 'FETCH'"),
])
def test_an_invalid_option_is_refused(simple_library_book_model, tmp_path, generator, options, message):
    with pytest.raises(GeneratorInputError, match=message):
        run_generator(generator, {"domain": simple_library_book_model, "quantum": object()}, tmp_path / "out", options)

    assert not (tmp_path / "out").exists()


def test_every_invalid_option_is_named_at_once():
    with pytest.raises(GeneratorInputError, match="unknown option.*colour.*; backend must be true or false"):
        check_options(get_generator("pydantic"), {"colour": "red", "backend": 1})


def test_a_value_the_constructor_refuses_is_an_input_error(simple_library_book_model, tmp_path):
    with pytest.raises(GeneratorInputError, match="package"):
        run_generator("spring", {"domain": simple_library_book_model}, tmp_path, {"package_name": "not a package"})


def test_build_generator_builds_without_running(simple_library_book_model, tmp_path):
    generator = build_generator(
        "sqlalchemy", {"domain": simple_library_book_model}, tmp_path / "out", {"dbms": "postgresql"},
    )

    assert type(generator).__name__ == "SQLAlchemyGenerator" and generator.dbms == "postgresql"
    assert generator.output_dir == str(tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(importlib.util.find_spec("pygit2") is None, reason="regenerate() needs pygit2 (besser[regen])")
def test_regenerate_generator_merges_into_the_folder_and_keeps_an_edit(simple_library_book_model, tmp_path):
    models, classes = {"domain": simple_library_book_model}, tmp_path / "out" / "classes.py"
    first = regenerate_generator("python", models, tmp_path / "out")
    classes.write_text("# mine\n" + classes.read_text())

    second = regenerate_generator("python", models, tmp_path / "out")

    assert first.written == ["classes.py"] and (tmp_path / "out" / RECORD_FILENAME).is_file()
    assert second.written == [] and second.conflicted == []
    assert classes.read_text().startswith("# mine\n")


def test_regenerate_generator_refuses_a_generator_that_does_not_take_part(simple_library_book_model, tmp_path):
    with pytest.raises(RegenerationError, match="SupabaseGenerator cannot regenerate"):
        regenerate_generator("supabase", {"domain": simple_library_book_model}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_regenerate_generator_checks_the_input_as_run_generator_does(simple_library_book_model, tmp_path):
    with pytest.raises(GeneratorInputError, match="unknown option"):
        regenerate_generator("python", {"domain": simple_library_book_model}, tmp_path / "out", {"dbms": "x"})


@pytest.mark.skipif(importlib.util.find_spec("pygit2") is None, reason="regenerate() needs pygit2 (besser[regen])")
def test_a_dry_run_regeneration_reports_and_writes_nothing(simple_library_book_model, tmp_path):
    report = regenerate_generator("python", {"domain": simple_library_book_model}, tmp_path / "out", dry_run=True)

    assert report.written == ["classes.py"] and not (tmp_path / "out").exists()


def test_run_generator_with_record_writes_the_record_regenerate_merges_against(simple_library_book_model, tmp_path):
    files = run_generator("python", {"domain": simple_library_book_model}, tmp_path / "out", record=True)

    assert files == ["classes.py"] and (tmp_path / "out" / RECORD_FILENAME).is_file()


def test_preview_generator_says_what_a_run_would_write(simple_library_book_model, tmp_path):
    models = {"domain": simple_library_book_model}
    run_generator("python", models, tmp_path / "out")

    plan = preview_generator("python", models, tmp_path / "out", record=True)

    assert plan == {RECORD_FILENAME: "new", "classes.py": "unchanged"}
    assert not (tmp_path / "out" / RECORD_FILENAME).exists()


@pytest.mark.parametrize("spec", GENERATORS, ids=lambda spec: spec.id)
def test_supports_regenerate_is_what_the_generator_class_says(spec):
    assert spec.supports_regenerate is spec.load_class().regenerable


def test_flutter_gets_the_main_page_of_the_gui_model(tmp_path):
    item = Class(name="Item", attributes={Property(name="name", type=StringType)})
    models = {"domain": DomainModel(name="Shop", types={item}), "gui": _gui_model()}

    assert run_generator("flutter", models, tmp_path) == ["main.dart", "pubspec.yaml", "sql_helper.dart"]


@pytest.mark.parametrize("main_pages", [0, 2])
def test_flutter_refuses_a_gui_model_without_exactly_one_main_page(simple_library_book_model, tmp_path, main_pages):
    models = {"domain": simple_library_book_model, "gui": _gui_model(main_pages)}

    with pytest.raises(GeneratorInputError, match=f"exactly one main page.*this one has {main_pages}"):
        run_generator("flutter", models, tmp_path)


@pytest.mark.parametrize("spec", GENERATORS, ids=lambda spec: spec.id)
def test_the_registry_entry_describes_the_constructor(spec):
    """``models`` are the constructor's leading arguments and every option is one of its keywords."""
    parameters = list(inspect.signature(spec.load_class().__init__).parameters.values())[1:]
    names = [parameter.name for parameter in parameters]

    assert spec.models and set(spec.models) <= MODEL_KINDS
    assert len(names) >= len(spec.models) and "output_dir" in names[len(spec.models):]
    if spec.id != "forecaster":  # its options are the fields of the ForecastConfig it takes
        assert set(spec.options) <= set(names[len(spec.models):])
    for option in spec.options.values():
        assert option["type"] in {"boolean", "integer", "string", "array"}


@pytest.mark.parametrize("spec", [spec for spec in GENERATORS if spec.agent_tool], ids=lambda spec: spec.id)
def test_the_agent_tool_agrees_with_the_registry_entry(spec):
    assert spec.agent_tool.requires == frozenset(spec.models)
    assert spec.agent_tool.parameters.items() <= spec.options.items()
