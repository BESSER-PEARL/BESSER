"""Round trips through the v4 converters keep what the editor saved.

Each test records a loss or duplication found by round-tripping editor
templates and fixtures: JSON -> B-UML -> JSON, and project JSON -> project
``.py`` -> project JSON.
"""

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


def _node(node_id, node_type, data):
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "width": 160, "height": 100, "data": data}


def _class_diagram(nodes, edges=()):
    return {"title": "Library", "model": {"version": "4.0.0", "type": "ClassDiagram",
                                         "nodes": list(nodes), "edges": list(edges)}}


def test_class_description_is_not_doubled_by_a_round_trip():
    diagram = _class_diagram([_node("book", "class", {
        "name": "Book", "description": "A printed work", "attributes": [], "methods": [],
    })])
    for _ in range(2):
        exported = class_buml_to_json(process_class_diagram(diagram))
        diagram = {"title": "Library", "model": exported}
        # Where the class inspector reads it, and nowhere else.
        [book] = [n for n in exported["nodes"] if n["type"] == "class"]
        assert book["data"].get("description") == "A printed work"
        assert not [n for n in exported["nodes"] if n["type"] == "comment"]
    model = process_class_diagram(diagram)
    assert model.get_class_by_name("Book").metadata.description == "A printed work"


def test_method_contracts_are_not_duplicated_by_a_buml_file_round_trip(tmp_path):
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )

    renew = {"id": "renew", "name": "renew", "visibility": "public", "returnType": "any",
             "parameters": [{"id": "p", "name": "days", "parameterType": "int"}]}
    diagram = _class_diagram([
        _node("book", "class", {"name": "Book", "attributes": [], "methods": [renew]}),
        _node("pre", "ClassOCLConstraint",
              {"expression": "context Book::renew(days: int) pre: days > 0"}),
        _node("inv", "ClassOCLConstraint",
              {"expression": "context Book inv named: true"}),
    ])
    model = process_class_diagram(diagram)
    [method] = model.get_class_by_name("Book").methods
    assert len(method.pre) == 1 and len(model.constraints) == 1

    path = tmp_path / "model.py"
    domain_model_to_code(model, str(path))
    reloaded = parse_buml_content(path.read_text(encoding="utf-8"))
    assert {c.name for c in reloaded.constraints} == {"named"}
    exported = class_buml_to_json(reloaded)
    boxes = [n for n in exported["nodes"] if n["type"] == "ClassOCLConstraint"]
    assert len(boxes) == 2


def _fixture_model(name):
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[5] / "fixtures" / "v4" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["model"]


def _project_round_trip(diagrams, tmp_path, description="Round trip", name="RoundTrip"):
    import copy
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
        json_to_buml_project,
    )
    project = {"id": "p", "type": "Project", "name": name, "description": description,
               "owner": "tester", "createdAt": "2026-01-01T00:00:00Z",
               "diagrams": copy.deepcopy(diagrams)}
    out = tmp_path / "project.py"
    project_to_code(json_to_buml_project(ProjectInput(**project)), str(out))
    code = out.read_text(encoding="utf-8")
    return code, project_to_json(code)


def test_several_diagrams_of_one_type_import_once_each(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent, sm, cd = (_fixture_model(n) for n in
                     ("agent_diagram_basic", "state_machine_basic", "class_diagram_basic"))
    diagrams = {
        "AgentDiagram": [{"id": "a1", "title": "AgentOne", "model": agent},
                         {"id": "a2", "title": "AgentTwo", "model": agent}],
        "StateMachineDiagram": [{"id": "s1", "title": "SmOne", "model": sm},
                                {"id": "s2", "title": "SmTwo", "model": sm}],
        "ClassDiagram": [{"id": "c1", "title": "LibA", "model": cd},
                         {"id": "c2", "title": "LibB", "model": cd}],
    }
    _, imported = _project_round_trip(diagrams, tmp_path)
    for diagram_type in diagrams:
        entries = imported["diagrams"][diagram_type]
        assert len(entries) == 2, (diagram_type, [e["title"] for e in entries])
        assert all(e["model"]["nodes"] for e in entries), diagram_type


def _cross_reference_project():
    """A component realizes a class and joins a BPMN process; an object
    instantiates the same class by its node id."""
    bpmn = {"version": "4.0.0", "type": "BPMNDiagram", "edges": [
        {"id": "f1", "type": "BPMNSequenceFlow", "source": "s", "target": "e",
         "data": {"name": "", "points": []}},
    ], "nodes": [
        _node("s", "bpmnStartEvent", {"name": "start", "eventType": "default"}),
        _node("e", "bpmnEndEvent", {"name": "end", "eventType": "default"}),
    ]}
    component = {"version": "4.0.0", "type": "ComponentDiagram", "edges": [], "nodes": [
        _node("cmp", "component", {"name": "Catalogue", "isComponentHeaderShown": True,
                                   "realizes": ["n-book"], "processModelRefs": ["bp-1"]}),
    ]}
    objects = {"version": "4.0.0", "type": "ObjectDiagram", "edges": [], "nodes": [
        _node("o1", "objectName", {"name": "dune", "classId": "n-book", "methods": [],
                                   "attributes": []}),
    ]}
    return {
        "ClassDiagram": [{"id": "cd-1", "title": "Library",
                          "model": _fixture_model("class_diagram_basic")}],
        "ObjectDiagram": [{"id": "od-1", "title": "Shelf", "model": objects,
                           "references": {"ClassDiagram": "cd-1"}}],
        "BPMN": [{"id": "bp-1", "title": "Lending", "model": bpmn}],
        "ComponentDiagram": [{"id": "cmp-1", "title": "Services", "model": component}],
    }


def test_project_import_keeps_diagram_and_class_ids(tmp_path, monkeypatch):
    from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
        json_to_buml_project,
    )
    monkeypatch.chdir(tmp_path)
    diagrams = _cross_reference_project()
    _, imported = _project_round_trip(diagrams, tmp_path)

    for diagram_type, expected in (("ClassDiagram", "cd-1"), ("ObjectDiagram", "od-1"),
                                   ("BPMN", "bp-1"), ("ComponentDiagram", "cmp-1")):
        assert [d["id"] for d in imported["diagrams"][diagram_type]] == [expected], diagram_type
    class_ids = {n["data"]["name"]: n["id"]
                 for n in imported["diagrams"]["ClassDiagram"][0]["model"]["nodes"]
                 if n["type"] == "class"}
    assert class_ids["Book"] == "n-book"
    [obj] = imported["diagrams"]["ObjectDiagram"][0]["model"]["nodes"]
    assert obj["data"]["classId"] == "n-book"

    # The imported project exports again: its cross-diagram references resolve.
    project = dict(imported, name="RoundTrip")
    reexported = json_to_buml_project(ProjectInput(**project))
    assert reexported._cross_diagram_errors["errors"] == []


def test_agent_default_llm_survives_a_project_round_trip(tmp_path, monkeypatch):
    from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
        json_to_buml_project,
    )
    from tests.utilities.web_modeling_editor.backend.services.converters.test_agent_converter_multi_llm import (
        _multi_llm_payload,
    )
    monkeypatch.chdir(tmp_path)
    payload = _multi_llm_payload()
    diagrams = {"AgentDiagram": [{"id": "ag-1", "title": "Helper", "model": payload["model"],
                                  "config": payload["config"]}]}
    _, imported = _project_round_trip(diagrams, tmp_path)

    [agent_entry] = imported["diagrams"]["AgentDiagram"]
    # The editor reads the default LLM from the diagram's config envelope.
    assert (agent_entry.get("config") or {}).get("default_llm_name") == "big"
    agent = next(m for m in json_to_buml_project(ProjectInput(**dict(imported, name="RT"))).models
                 if type(m).__name__ == "Agent")
    assert agent.default_llm_name == "big"


def test_agent_default_llm_is_read_from_the_model_config_too():
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
        process_agent_diagram,
    )
    from tests.utilities.web_modeling_editor.backend.services.converters.test_agent_converter_multi_llm import (
        _multi_llm_payload,
    )
    payload = _multi_llm_payload()
    payload["model"]["config"] = payload.pop("config")
    assert process_agent_diagram(payload).default_llm_name == "big"


def test_nn_metrics_and_sub_nn_reference_use_the_editor_formats():
    from besser.BUML.metamodel.nn import NN, Configuration, LinearLayer
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
        nn_model_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
        process_nn_diagram,
    )
    classifier = NN(name="classifier")
    classifier.add_layer(LinearLayer(name="head", in_features=8, out_features=2))
    net = NN(name="net")
    net.add_layer(LinearLayer(name="body", in_features=4, out_features=8))
    net.add_sub_nn(classifier)
    net.add_configuration(Configuration(
        batch_size=8, epochs=1, learning_rate=0.01, optimizer="adam",
        loss_function="crossentropy", metrics=["f1-score", "accuracy"],
    ))

    out = nn_model_to_json(net)
    [config] = [n for n in out["nodes"] if n["type"] == "Configuration"]
    # The editor's metrics multiselect parses the bare bracketed form.
    assert config["data"]["attributes"]["metrics"] == "[f1-score, accuracy]"
    [ref] = [n for n in out["nodes"] if n["type"] == "NNReference"]
    container_ids = {n["data"]["name"]: n["id"] for n in out["nodes"] if n["type"] == "NNContainer"}
    # referenceTarget is the referenced container's node id, as the editor stores it.
    assert ref["data"]["referenceTarget"] == container_ids["classifier"]

    rebuilt = process_nn_diagram({"title": "net", "model": out})
    assert rebuilt.configuration.metrics == ["f1-score", "accuracy"]
    assert [m.name for m in rebuilt.modules if isinstance(m, NN)] == ["classifier"]


def test_project_description_comes_from_the_project_definition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    classes = _class_diagram([
        _node("book", "class", {"name": "Book", "attributes": [], "methods": []}),
        _node("note", "comment", {"name": "A printed work"}),
    ], [{"id": "l", "type": "CommentLink", "source": "note", "target": "book", "data": {"points": []}}])
    diagrams = {"ClassDiagram": [{"id": "cd-1", "title": "Library", "model": classes["model"]}]}
    _, imported = _project_round_trip(diagrams, tmp_path, description='Lending "desk" app')
    assert imported["description"] == 'Lending "desk" app'


def test_a_class_named_project_survives_a_buml_file_round_trip(tmp_path):
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )
    diagram = _class_diagram([_node("p", "class", {"name": "Project", "methods": [], "attributes": [
        {"id": "a", "name": "title", "attributeType": "str", "visibility": "public"},
    ]})])
    path = tmp_path / "model.py"
    domain_model_to_code(process_class_diagram(diagram), str(path))
    reloaded = parse_buml_content(path.read_text(encoding="utf-8"))
    assert [a.name for a in reloaded.get_class_by_name("Project").attributes] == ["title"]


def test_a_project_name_with_spaces_exports_and_comes_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    classes = _class_diagram([_node("book", "class", {"name": "Book", "attributes": [], "methods": []})])
    diagrams = {"ClassDiagram": [{"id": "cd-1", "title": "Library", "model": classes["model"]}]}
    code, imported = _project_round_trip(diagrams, tmp_path, name="My Library-App")
    # B-UML names cannot hold spaces or hyphens; the editor's name rides alongside.
    assert 'name="My_Library_App"' in code
    assert imported["name"] == "My Library-App"


def test_diagram_titles_survive_a_project_round_trip(tmp_path, monkeypatch):
    import copy
    monkeypatch.chdir(tmp_path)
    bpmn = _cross_reference_project()["BPMN"][0]["model"]
    diagrams = {
        "AgentDiagram": [{"id": "ag-1", "title": "Front desk bot",
                          "model": _fixture_model("agent_diagram_basic")}],
        "BPMN": [{"id": "bp-1", "title": "Lending", "model": bpmn},
                 {"id": "bp-2", "title": "Returns \u2014 late", "model": copy.deepcopy(bpmn)}],
    }
    _, imported = _project_round_trip(diagrams, tmp_path)
    assert [d["title"] for d in imported["diagrams"]["AgentDiagram"]] == ["Front desk bot"]
    assert [d["title"] for d in imported["diagrams"]["BPMN"]] == ["Lending", "Returns \u2014 late"]


def test_object_diagrams_import_against_their_referenced_class_diagram(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def classes(node_id, name, attr):
        return _class_diagram([_node(node_id, "class", {"name": name, "methods": [], "attributes": [
            {"id": f"{node_id}-{attr}", "name": attr, "attributeType": "str", "visibility": "public"}]})])["model"]

    def objects(class_id, class_name, attr):
        return {"version": "4.0.0", "type": "ObjectDiagram", "edges": [], "nodes": [
            _node(f"o-{class_id}", "objectName", {
                "name": f"{class_name.lower()}1", "className": class_name, "classId": class_id,
                "methods": [], "attributes": [
                    {"id": "r", "name": attr, "attributeId": f"{class_id}-{attr}", "value": "v"}]})]}

    diagrams = {
        "ClassDiagram": [{"id": "cd-a", "title": "Books", "model": classes("ca", "Book", "title")},
                         {"id": "cd-b", "title": "Cars", "model": classes("cb", "Car", "plate")}],
        # Listed in the opposite order, so neither can rely on the first class diagram.
        "ObjectDiagram": [{"id": "od-b", "title": "Fleet", "model": objects("cb", "Car", "plate"),
                           "references": {"ClassDiagram": "cd-b"}},
                          {"id": "od-a", "title": "Shelf", "model": objects("ca", "Book", "title"),
                           "references": {"ClassDiagram": "cd-a"}}],
    }
    _, imported = _project_round_trip(diagrams, tmp_path)
    by_id = {d["id"]: d for d in imported["diagrams"]["ObjectDiagram"]}
    for od_id, cd_id, class_id in (("od-b", "cd-b", "cb"), ("od-a", "cd-a", "ca")):
        assert by_id[od_id].get("references") == {"ClassDiagram": cd_id}
        assert [n["data"]["classId"] for n in by_id[od_id]["model"]["nodes"]] == [class_id]
