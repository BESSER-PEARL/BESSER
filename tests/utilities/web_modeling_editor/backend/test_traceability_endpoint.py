"""``POST /besser_api/traceability/generate-output`` and the trace file converters it uses."""

import base64
import copy

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters import (
    class_buml_to_json,
    process_trace,
    trace_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

URL = "/besser_api/traceability/generate-output"


@pytest.fixture
def payload(library_book_author_model):
    return {
        "id": "diagram-1",
        "title": "Library",
        "model": class_buml_to_json(library_book_author_model),
        "generator": "sqlalchemy",
        "config": {"dbms": "sqlite"},
        "actorName": "BESSER User",
    }


@pytest.fixture
def traced(client, payload):
    response = client.post(URL, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_file_is_the_plain_download(client, payload, traced):
    plain = client.post("/besser_api/generate-output", json={k: v for k, v in payload.items() if k != "actorName"})
    assert plain.status_code == 200
    assert base64.b64decode(traced["contentBase64"]) == plain.content
    assert traced["fileName"] == "sql_alchemy.py"


def test_trace_links_each_class_to_its_definition_line(payload, traced):
    trace = traced["trace"]
    lines = base64.b64decode(traced["contentBase64"]).decode("utf-8").split("\n")
    artifacts = {a["id"]: a for a in trace["artifacts"]}
    class_ids = {e["id"]: e["name"] for e in payload["model"]["elements"].values() if e["type"] == "Class"}
    linked = {}
    for link in trace["traceLinks"]:
        source, target = artifacts[link["sourceId"]], artifacts[link["targetId"]]
        if source.get("elementId") in class_ids and link["evidence"][0]["rule"].endswith("#class"):
            linked[class_ids[source["elementId"]]] = lines[target["startLine"] - 1]
    assert set(linked) == {"Library", "Book", "Author"}
    for name, line in linked.items():
        assert line.startswith(f"class {name}(")


def test_trace_records_actor_tool_and_model(traced):
    trace = traced["trace"]
    assert [(a["type"], a["name"]) for a in trace["actors"]] == [("Person", "BESSER User")]
    assert [(t["type"], t["name"]) for t in trace["tools"]] == [("CodeGenerator", "sqlalchemy")]
    (operation,) = trace["operations"]
    assert operation["type"] == "CodeGeneration" and operation["status"] == "SUCCEEDED"
    assert "changes" not in operation
    model = next(a for a in trace["artifacts"] if a.get("elementType") == "DomainModel")
    assert model["elementId"] == "diagram-1" and model["id"] in operation["consumes"]
    file = next(a for a in trace["artifacts"] if a["type"] == "CodeArtifact" and "startLine" not in a)
    assert file["content"] == base64.b64decode(traced["contentBase64"]).decode("utf-8")


def test_blank_actor_is_anonymous(client, payload):
    response = client.post(URL, json={**payload, "actorName": "   "})
    assert response.json()["trace"]["actors"][0]["name"] == "anonymous"


def test_untraced_generator_is_rejected(client, payload):
    response = client.post(URL, json={**payload, "generator": "django"})
    assert response.status_code == 400
    assert "does not record traces" in response.json()["detail"]


def test_diagram_id_is_required(client, payload):
    response = client.post(URL, json={**payload, "id": None})
    assert response.status_code == 400


def test_trace_file_round_trips(traced):
    trace = traced["trace"]
    assert trace_buml_to_json(process_trace(trace)) == trace


def test_trace_with_edits_round_trips(traced):
    trace = copy.deepcopy(traced["trace"])
    class_artifact = next(a for a in trace["artifacts"] if a.get("elementType") == "Class")
    trace["projectId"] = "project-1"
    trace["operations"].append({
        "id": "op-edit", "type": "Modeling", "status": "SUCCEEDED",
        "startedAt": "2026-10-29T10:00:00.000Z", "endedAt": "2026-10-29T10:05:00.000Z",
        "performedBy": [trace["actors"][0]["id"]], "uses": [], "consumes": [], "produces": [],
        "changes": [{"id": "ch-1", "kind": "UPDATED", "artifactId": class_artifact["id"], "feature": "name",
                     "oldValue": "Libro", "newValue": "Book", "timestamp": "2026-10-29T10:01:00.000Z"}],
        "logs": [],
    })
    trace["actors"].append({"id": "assistant", "type": "Agent", "name": "BESSER modeling assistant",
                            "timestamp": "2026-10-29T09:59:00.000Z"})
    trace["artifacts"].append({"id": "prompt-1", "type": "PromptArtifact", "name": "Library",
                               "timestamp": "2026-10-29T09:58:00.000Z", "content": "Books have a title.",
                               "contentHash": "h"})
    trace["operations"].append({
        "id": "op-chat", "type": "Conversation", "status": "SUCCEEDED", "startedAt": "2026-10-29T09:58:00.000Z",
        "performedBy": [trace["actors"][0]["id"], "assistant"], "uses": [], "consumes": ["prompt-1"],
        "produces": [], "logs": [],
    })
    assert trace_buml_to_json(process_trace(trace)) == trace


@pytest.mark.parametrize("corrupt, message", [
    (lambda t: t["operations"][0].update(type="CODE_GENERATION"), "unknown type"),
    (lambda t: t["operations"][0].update(changes=[{"id": "c"}]), "only a Modeling has changes"),
    (lambda t: t.update(schemaVersion=99), "schemaVersion"),
    (lambda t: t["traceLinks"][0].update(sourceId="missing"), "unknown artifact"),
    (lambda t: t["operations"][0].update(startedAt="yesterday"), "ISO 8601"),
    (lambda t: t["operations"][0].update(status="DONE"), "must be one of"),
    (lambda t: t["artifacts"][0].pop("name"), "missing 'name'"),
    (lambda t: t["operations"][0].update(performedBy=[]), "at least one Actor"),
])
def test_invalid_trace_is_a_conversion_error(traced, corrupt, message):
    trace = copy.deepcopy(traced["trace"])
    corrupt(trace)
    with pytest.raises(ConversionError, match=message):
        process_trace(trace)
