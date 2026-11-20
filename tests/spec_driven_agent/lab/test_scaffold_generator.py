"""``scaffold_generator`` writes a new generator that runs and keeps the registry's pinned tests true.

Each test copies the repository files the scaffold edits into ``tmp_path`` (with LF and
with CRLF endings, as a Windows checkout has them), scaffolds there, then loads what was
written: the copied registry, the generated package, its test and the regeneration builder.
"""

import ast
import configparser
import glob
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import besser.generators
from besser.generators import GeneratorInterface
from besser.spec_driven_agent.lab.scaffolds import ScaffoldError, scaffold_generator, scaffold_generator_tool
from besser.spec_driven_agent.lab.scaffolds.generator import TEMPLATES_DIR
from besser.spec_driven_agent.lab.tooling import LabToolExecutor
from tests.conftest import library_book_author

REPO_ROOT = Path(__file__).resolve().parents[3]
EDITED = (
    "besser/generators/registry.py",
    "tests/generators/test_generator_registry.py",
    "tests/generators/snapshots/web_editor_generators.json",
    "tests/generators/regenerable_builders.py",
    "docs/source/generators.rst",
)
NAME = "class_list"
CLASS = "ClassListGenerator"
CREATED = (
    "besser/generators/class_list/__init__.py",
    "besser/generators/class_list/class_list_generator.py",
    "besser/generators/class_list/templates/class_list.txt.j2",
    "tests/generators/class_list/test_class_list_generator.py",
    "docs/source/generators/class_list.rst",
)


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def repo(request, tmp_path):
    for path in EDITED:
        text = (REPO_ROOT / path).read_bytes().decode("utf-8").replace("\r\n", "\n")
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.replace("\n", request.param).encode("utf-8"))
    return tmp_path


@pytest.fixture
def importable(repo, monkeypatch):
    """Makes ``besser.generators.<name>`` of ``repo`` importable; forgets it afterwards."""
    monkeypatch.setattr(besser.generators, "__path__", [*besser.generators.__path__, str(repo / "besser/generators")])
    yield
    package = f"besser.generators.{NAME}"
    for module in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[module]
    besser.generators.__dict__.pop(NAME, None)


def _load(path: Path, module_name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _literal(path: Path, name: str):
    """The value of the module-level ``name = <literal>`` in ``path``, without running the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets))


def _files(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_writes_every_part_a_generator_needs_and_keeps_line_endings(repo):
    newline = b"\r\n" if b"\r\n" in (repo / EDITED[0]).read_bytes() else b"\n"

    written = scaffold_generator(str(repo), NAME)

    assert written == sorted([*CREATED, *EDITED])
    for path in EDITED:
        content = (repo / path).read_bytes()
        assert content.count(b"\n") == content.count(newline), path


def test_the_generator_registers_and_runs_and_its_own_test_passes(repo, importable, monkeypatch, tmp_path_factory):
    scaffold_generator(str(repo), NAME)

    registry = _load(repo / "besser/generators/registry.py", "lab_scaffold_registry", monkeypatch)
    spec = next(s for s in registry.GENERATORS if s.id == NAME)
    generator_class = spec.load_class()
    assert generator_class.__name__ == CLASS and issubclass(generator_class, GeneratorInterface)
    assert spec.output_type == "file" and spec.download_filename == "class_list.txt"

    out = tmp_path_factory.mktemp("out")
    generator = generator_class(library_book_author(), output_dir=str(out))
    generator.generate()
    assert (out / "class_list.txt").read_text(encoding="utf-8").startswith("Author\n  email: str\n  name: str\nBook\n")
    assert set(generator.traced_files) == {"class_list.txt"}

    generated_test = _load(repo / CREATED[3], "lab_scaffold_generated_test", monkeypatch)
    generated_test.test_writes_every_class_with_its_attributes(library_book_author(), tmp_path_factory.mktemp("t"))


def test_the_pinned_lists_and_snapshot_match_the_new_registry(repo, importable, monkeypatch):
    scaffold_generator(str(repo), NAME)

    registry = _load(repo / "besser/generators/registry.py", "lab_scaffold_registry", monkeypatch)
    ids = [spec.id for spec in registry.GENERATORS]
    real_ids = _literal(REPO_ROOT / "tests/generators/test_generator_registry.py", "WEB_EDITOR_IDS")
    assert ids == real_ids + [NAME]
    registry_test = repo / "tests/generators/test_generator_registry.py"
    assert _literal(registry_test, "WEB_EDITOR_IDS") == ids
    agent_ids = {spec.id for spec in registry.GENERATORS if spec.agent_tool}
    assert _literal(registry_test, "WEB_EDITOR_ONLY") == set(ids) - agent_ids

    snapshot = json.loads((repo / "tests/generators/snapshots/web_editor_generators.json").read_text(encoding="utf-8"))
    assert snapshot["supported_generators"] == ids
    spec = next(s for s in registry.GENERATORS if s.id == NAME)
    cls = spec.load_class()
    entry = snapshot["generators"][NAME]
    assert entry.keys() == snapshot["generators"]["rdf"].keys()
    assert entry == {"generator_class": f"{cls.__module__}.{cls.__qualname__}",
                     **{key: getattr(spec, key) for key in entry if key != "generator_class"}}


def test_the_regeneration_builder_covers_it_and_a_second_run_changes_nothing(repo, importable, monkeypatch, tmp_path):
    pytest.importorskip("pygit2")
    scaffold_generator(str(repo), NAME)

    builders = _load(repo / "tests/generators/regenerable_builders.py", "lab_scaffold_builders", monkeypatch).BUILDERS
    registry = _load(repo / "besser/generators/registry.py", "lab_scaffold_registry", monkeypatch)
    excluded = _literal(REPO_ROOT / "tests/generators/test_regenerable_generators.py", "NOT_REGENERABLE")
    assert set(excluded) | {*builders, "django"} == {spec.id for spec in registry.GENERATORS}

    out = str(tmp_path / "regen")
    builders[NAME](out).regenerate()
    report = builders[NAME](out).regenerate()
    assert not any(getattr(report, key) for key in ("written", "merged", "conflicted", "deleted",
                                                    "kept_removed_from_model", "not_restored", "not_merged"))


def test_the_docs_page_is_in_a_toctree_and_the_choosing_table(repo):
    scaffold_generator(str(repo), NAME)

    page = (repo / "docs/source/generators/class_list.rst").read_text(encoding="utf-8").splitlines()
    assert page[0] == "Class List Generator" and page[1] == "=" * len(page[0])
    doc = (repo / "docs/source/generators.rst").read_text(encoding="utf-8")
    choosing = doc.split("Choosing a Generator\n", 1)[1].split("\nNone of the above", 1)[0]
    assert choosing.rstrip().endswith(
        "   * - **Class List**\n     - Structural\n     - .txt\n"
        "     - You need one ``.txt`` file listing your model's classes and attributes")
    frameworks = doc.split("Frameworks & Languages\n", 1)[1].split("\nData & API", 1)[0]
    assert frameworks.rstrip().endswith("   generators/alloy\n   generators/class_list")


def test_generated_python_passes_the_repository_lint(repo):
    if shutil.which("ruff") is None and importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff is not installed")
    scaffold_generator(str(repo), NAME)

    python_files = [str(repo / path) for path in (*CREATED, *EDITED) if path.endswith(".py")]
    result = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache", "--config",
                             str(REPO_ROOT / "pyproject.toml"), *python_files],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name, kwargs, reason", [
    ("python", {}, "already registered"),
    ("my-gen", {}, "not a valid generator name"),
    ("MyGen", {}, "not a valid generator name"),
    ("class", {}, "not a valid generator name"),
    (NAME, {"class_name": "Not A Class"}, "not a valid Python class name"),
    (NAME, {"file_extension": "txt"}, "not a file extension"),
    ("registry", {}, "besser/generators/registry.py already exists"),
])
def test_refusals_leave_the_checkout_unchanged(repo, name, kwargs, reason):
    before = _files(repo)

    with pytest.raises(ScaffoldError, match=reason):
        scaffold_generator(str(repo), name, **kwargs)

    assert _files(repo) == before


def test_an_existing_package_folder_is_refused(repo):
    (repo / "besser/generators/class_list").mkdir()

    with pytest.raises(ScaffoldError, match="besser/generators/class_list already exists"):
        scaffold_generator(str(repo), NAME)


def test_the_tool_reports_what_it_wrote_and_answers_a_refusal_with_its_reason(repo):
    tool = scaffold_generator_tool(str(repo))
    assert tool.schema["name"] == "scaffold_generator" and tool.writes
    assert tool.schema["input_schema"]["required"] == ["name"]

    text = tool.handler({"name": "graph_dump", "class_name": "GraphDumper", "file_extension": ".dot"})

    assert "- besser/generators/graph_dump/graph_dump_generator.py" in text
    assert "- besser/generators/registry.py" in text
    assert 'class_path="besser.generators.graph_dump:GraphDumper"' in (
        repo / "besser/generators/registry.py").read_text(encoding="utf-8")
    # A refusal reaches the model as an error result, not as a success it would read past.
    executor = LabToolExecutor(str(repo), allow_shell=False, extra_tools=[tool])
    try:
        text, is_error = executor.execute("scaffold_generator", {"name": "graph_dump"})
    finally:
        executor.close()
    assert is_error
    assert text.startswith("scaffold_generator refused: Generator 'graph_dump' cannot be created")


def test_the_templates_ship_in_the_wheel():
    """``setup.cfg``'s package data for every package (key ``*``) matches each scaffold template."""
    config = configparser.ConfigParser()
    config.read(REPO_ROOT / "setup.cfg", encoding="utf-8")
    patterns = [p.strip() for p in config["options.package_data"]["*"].split(",")]
    package_dir = TEMPLATES_DIR.parent
    shipped = {Path(f) for p in patterns for f in glob.glob(str(package_dir / p), recursive=True)}
    templates = set(TEMPLATES_DIR.iterdir())
    assert templates and templates <= shipped
