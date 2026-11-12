"""``besser schema``: the formats the command reads and writes, described from the code."""

import importlib.util
import json

import pytest

from besser.cli.config import load_config
from besser.cli.loading import MODEL_KINDS
from besser.cli.schema import COMMAND_KEYS, result_schema
from tests.cli.conftest import INVALID_MODEL

needs_backend = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None,
                                   reason="editor JSON needs the backend's packages")


def test_without_a_topic_it_lists_the_topics(besser):
    run = besser("schema", "--json")
    topics = [entry["topic"] for entry in run.json["topics"]]

    assert run.returncode == 0
    assert topics == [*(kind for kind, _ in MODEL_KINDS), "project", "result", "config"]
    assert "besser schema TOPIC" in besser("schema").stdout


@pytest.mark.parametrize("kind, root", [(kind, root.__name__) for kind, root in MODEL_KINDS])
def test_a_kind_is_described_by_its_classes_and_their_arguments(besser, kind, root):
    description = besser("schema", kind, "--json").json["description"]
    classes = {cls["name"]: cls for cls in description["classes"]}

    assert description["root_class"] == root and root in classes
    assert classes[root]["signature"].startswith(f"{root}(")
    # A signature names classes as the import line does, not by their module path.
    assert "besser." not in json.dumps(description["classes"])


def test_the_domain_description_gives_what_a_model_is_written_with(besser):
    run = besser("schema", "domain")
    description = besser("schema", "domain", "--json").json["description"]
    property_arguments = {arg["name"]: arg for arg in
                          next(cls for cls in description["classes"] if cls["name"] == "Property")["parameters"]}

    assert run.stdout.startswith("domain: a DomainModel, from besser.BUML.metamodel.structural\n")
    assert "besser new model.py --kind domain" in run.stdout
    assert property_arguments["name"] == {"name": "name", "required": True, "type": "str"}
    assert property_arguments["is_id"]["default"] == "False"
    assert "StringType" in [value["name"] for value in description["values"]]
    assert description["editor_diagram"] == "ClassDiagram" and "sqlalchemy" in description["generators"]


def test_an_unknown_topic_is_a_usage_error(besser):
    run = besser("schema", "petri_net", "--json")

    assert run.returncode == 2 and run.json["command"] == "schema"
    assert "invalid choice: 'petri_net'" in run.json["errors"][0]


@needs_backend
def test_project_is_the_json_schema_the_editor_validates_with(besser):
    result = besser("schema", "project", "--json").json

    assert {"diagrams", "currentDiagramIndices"} <= set(result["schema"]["properties"])
    assert [cls["name"] for cls in result["description"]["classes"]] == ["Project"]


def test_config_is_an_example_that_build_reads(besser, tmp_path):
    example = besser("schema", "config").stdout
    (tmp_path / "besser.toml").write_text(example, encoding="utf-8")

    outputs = load_config(str(tmp_path / "besser.toml"))

    assert [(output.name, output.generator) for output in outputs] == [("api", "backend"), ("db", "sqlalchemy")]
    assert outputs[0].options == {"http_methods": ["GET", "POST"]} and outputs[1].use == ["library_model"]


def test_every_command_prints_an_object_the_result_schema_accepts(besser, write, model):
    jsonschema = pytest.importorskip("jsonschema")
    schema = besser("schema", "result", "--json").json["schema"]
    write("broken.py", INVALID_MODEL)
    write("besser.toml", 'model = "library.py"\n[output.code]\ngenerator = "python"\ndir = "built"\n')
    runs = [
        ("--version",), ("new", "starter.py"), ("new", "--list"), ("schema",), ("schema", "domain"),
        ("validate", model, "broken.py"), ("info", model), ("info", model, "--class", "Book"), ("generators",),
        ("model", "-f", model, "add-attribute", "Book", "isbn:str"), ("model", "-f", model, "remove", "Nope"),
        ("generate", "python", model, "-o", "out"), ("generate", "python", model, "-o", "dry", "--dry-run"),
        ("regenerate", "python", model, "-o", "out", "--dry-run"), ("build",), ("build", "--dry-run"),
        ("convert", "library.plantuml", "-o", "x.py"), ("completion", "bash"), ("nonsense",), ("generate",),
    ]

    assert schema == result_schema()
    for arguments in runs:
        result = besser(*arguments, "--json").json
        jsonschema.validate(result, schema)
        documented = set(schema["properties"]) | set(COMMAND_KEYS.get(result["command"], {}))
        assert set(result) <= documented, f"{arguments}: undocumented keys {set(result) - documented}"
