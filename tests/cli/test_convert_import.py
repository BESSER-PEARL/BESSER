"""``besser convert`` on other notations: PlantUML, draw.io, CSV, RDF, SDMX and Python code become models."""

import gzip
import importlib.util
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
_DRAWIO = _TESTS / "BUML" / "notations" / "structuralDrawIO" / "test.drawio"
_TURTLE = _TESTS / "BUML" / "notations" / "kg_to_buml" / "kg_example.ttl"
_SDMX = _TESTS / "utilities" / "sdmx_to_buml" / "fixtures" / "ecb_EXR.xml.gz"

PLANTUML = """\
@startuml
class Library {
  + name: str
}
class Book {
  + title: str
  + pages: int
}
Library "1" -- "0..*" Book : holds
@enduml
"""

needs_backend = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None,
                                   reason="this import lives in the editor backend, which needs its packages")


def _classes(besser, model):
    domain = next(entry for entry in besser("info", model, "--json").json["models"] if entry["kind"] == "domain")
    return sorted(cls["name"] for cls in domain["classes"])


def test_a_plantuml_class_diagram_becomes_a_buml_file(besser, write):
    run = besser("convert", write("library.plantuml", PLANTUML), "-o", "library.py", "--json")

    assert run.returncode == 0, run.stdout
    assert run.json["models"] == [{"name": "DomainModel", "kind": "domain", "type": "DomainModel"}]
    assert _classes(besser, "library.py") == ["Book", "Library"]
    assert besser("validate", "library.py").returncode == 0


@needs_backend
def test_a_plantuml_class_diagram_becomes_an_editor_project(besser, write):
    assert besser("convert", write("library.puml", PLANTUML), "-o", "library.json").returncode == 0
    assert _classes(besser, "library.json") == ["Book", "Library"]


def test_a_plantuml_syntax_error_names_the_line_and_writes_nothing(besser, write, tmp_path):
    run = besser("convert", write("broken.plantuml", "@startuml\nclass {{{ nope\n"), "-o", "broken.py")

    assert run.returncode == 2
    assert run.stderr.startswith("error: broken.plantuml: line 2, column 7: ")
    assert not (tmp_path / "broken.py").exists()


def test_a_drawio_class_diagram_becomes_a_buml_file(besser):
    assert besser("convert", _DRAWIO, "-o", "drawn.py").returncode == 0
    assert "Book" in _classes(besser, "drawn.py")


@needs_backend
def test_csv_files_become_classes_linked_by_their_id_columns(besser, write):
    write("author.csv", "id,name\n1,Ada\n2,Bo\n")
    write("book.csv", "id,title,pages,author_id\n1,Notes,120,1\n2,Poems,80,2\n")

    run = besser("convert", "author.csv", "book.csv", "-o", "books.py", "--json")
    domain = besser("info", "books.py", "--json").json["models"][0]

    assert run.returncode == 0, run.stdout
    assert run.json["inputs"] == ["author.csv", "book.csv"]
    # The importer names a class after its file.
    assert sorted(cls["name"] for cls in domain["classes"]) == ["author", "book"]
    assert len(domain["associations"]) == 1


def test_an_rdf_file_becomes_a_knowledge_graph_and_goes_back_to_rdf(besser, tmp_path):
    imported = besser("convert", _TURTLE, "-o", "graph.py", "--json")
    exported = besser("convert", "graph.py", "-o", "graph.ttl")

    assert imported.returncode == 0, imported.stdout
    assert imported.json["models"][0]["kind"] == "kg"
    assert exported.returncode == 0
    assert "foaf:Person" in (tmp_path / "graph.ttl").read_text()


def test_class_diagram_derives_a_domain_model_from_a_knowledge_graph(besser):
    run = besser("convert", _TURTLE, "-o", "classes.py", "--class-diagram", "--json")

    assert run.returncode == 0, run.stdout
    assert run.json["models"][0]["kind"] == "domain"
    assert _classes(besser, "classes.py") == ["Organization", "Person", "Place"]


def test_class_diagram_on_a_file_without_a_knowledge_graph_is_a_usage_error(besser, model):
    run = besser("convert", model, "-o", "classes.py", "--class-diagram")

    assert run.returncode == 2
    assert "--class-diagram derives a class diagram from a knowledge graph and library.py holds none" in run.stderr


def test_rdf_is_written_from_a_knowledge_graph_only(besser, model, tmp_path):
    run = besser("convert", model, "-o", "library.ttl")

    assert run.returncode == 2
    assert "library.ttl is RDF, which is written from one knowledge graph, and library.py holds none" in run.stderr
    assert not (tmp_path / "library.ttl").exists()


def test_an_sdmx_structure_message_is_named_with_from(besser, tmp_path):
    (tmp_path / "exr.xml").write_bytes(gzip.decompress(_SDMX.read_bytes()))

    unnamed = besser("convert", "exr.xml", "-o", "exr.py")
    named = besser("convert", "exr.xml", "-o", "exr.py", "--from", "sdmx")

    assert unnamed.returncode == 2 and "--from (plantuml, drawio, csv, rdf, sdmx, code)" in unnamed.stderr
    assert named.returncode == 0, named.stderr
    assert besser("validate", "exr.py").returncode == 0


@needs_backend
def test_python_data_classes_are_read_without_running_them(besser, write, tmp_path):
    write("app/models.py", """\
        from dataclasses import dataclass

        open("ran.txt", "w").write("the code ran")

        @dataclass
        class Author:
            name: str

        @dataclass
        class Book:
            title: str
            pages: int
        """)

    undeclared = besser("convert", "app", "-o", "app.py")
    run = besser("convert", "app", "-o", "app.py", "--from", "code")

    assert undeclared.returncode == 2
    assert undeclared.stderr == "error: app is a directory: name what it holds with --from (code)\n"
    assert run.returncode == 0, run.stderr
    assert _classes(besser, "app.py") == ["Author", "Book"]
    assert not (tmp_path / "ran.txt").exists()


def test_inputs_convert_cannot_read_are_usage_errors(besser, write, model):
    write("notes.txt", "class Book")
    write("other.plantuml", PLANTUML)

    unknown = besser("convert", "notes.txt", "-o", "notes.py")
    missing = besser("convert", "nowhere.plantuml", "-o", "nowhere.py")
    several = besser("convert", "other.plantuml", "other.plantuml", "-o", "two.py")
    target = besser("convert", model, "-o", "library.yaml")

    assert unknown.returncode == 2 and unknown.stderr.startswith("error: notes.txt: convert reads a B-UML .py file")
    assert missing.returncode == 2 and missing.stderr == "error: nowhere.plantuml: no such file\n"
    assert several.returncode == 2
    assert several.stderr == "error: convert reads one input; only these formats take several files: csv\n"
    assert target.returncode == 2 and "got library.yaml" in target.stderr


def test_an_import_that_holds_no_class_is_refused(besser, write):
    run = besser("convert", write("empty.plantuml", "@startuml\n@enduml\n"), "-o", "empty.py")

    assert run.returncode == 2
    assert run.stderr == "error: empty.plantuml: no model found: it holds nothing the plantuml import reads\n"


@needs_backend
def test_a_lone_quantum_circuit_survives_the_editor_json(besser):
    besser("new", "bell.py", "--kind", "quantum")

    assert besser("convert", "bell.py", "-o", "bell.json").returncode == 0
    back = besser("info", "bell.json", "--json").json["models"]
    assert [(model["kind"], model["counts"]["operations"]) for model in back] == [("quantum", 4)]
