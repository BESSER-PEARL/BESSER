"""/get-json-model with ``existing_model``: a re-imported B-UML file keeps the diagram's layout (issue #774)."""

import json

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code

URL = "/besser_api/get-json-model"

# An agent written with keyword arguments, as people (and LLMs) write B-UML by hand.
# The agent parser reads the exporter's positional form only.
KEYWORD_STYLE_AGENT = '''from besser.BUML.metamodel.state_machine.agent import Agent
agent = Agent(name="kw")
hello = agent.new_intent(name="hello", training_sentences=["hi"])
s0 = agent.new_state(name="initial", initial=True)
s1 = agent.new_state(name="next")
s0.when_intent_matched(hello).go_to(s1)
'''


def agent_source(tmp_path, with_farewell):
    agent = Agent("Greeter")
    hello = agent.new_intent("hello_intent", ["hi"])
    initial = agent.new_state("initial", initial=True)
    greeting = agent.new_state("greeting")
    initial.when_intent_matched(hello).go_to(greeting)
    if with_farewell:
        greeting.go_to(agent.new_state("farewell"))
    path = tmp_path / "agent.py"
    agent_model_to_code(agent, str(path))
    return path.read_text()


def post(client, source, existing_model=None):
    data = {"existing_model": existing_model} if existing_model is not None else None
    response = client.post(URL, files={"buml_file": ("agent.py", source.encode(), "text/plain")}, data=data)
    assert response.status_code == 200, response.text
    return response.json()


def test_reimport_keeps_the_layout_of_the_existing_diagram(client, tmp_path):
    first = post(client, agent_source(tmp_path, with_farewell=False))
    existing = first["model"]
    for index, node in enumerate(existing["nodes"]):
        node["position"] = {"x": 500 + 300 * index, "y": 400}

    body = post(client, agent_source(tmp_path, with_farewell=True), json.dumps(existing))

    nodes = {node["data"]["name"]: node for node in body["model"]["nodes"]}
    for old in existing["nodes"]:
        assert (nodes[old["data"]["name"]]["id"], nodes[old["data"]["name"]]["position"]) == \
            (old["id"], old["position"])
    codes = [warning["code"] for warning in body["warnings"]]
    assert codes == ["layout-adopted", "element-added"]
    added = body["warnings"][1]
    assert (added["name"], added["placedNextTo"], added["elementId"]) == \
        ("farewell", "greeting", nodes["farewell"]["id"])
    assert body["diagramType"] == body["model"]["type"] == "AgentDiagram"


def test_without_existing_model_the_response_is_unchanged(client, tmp_path):
    body = post(client, agent_source(tmp_path, with_farewell=False))
    assert body["warnings"] is None
    assert [node["position"] for node in body["model"]["nodes"]] == [{"x": -280, "y": -80}, {"x": 210, "y": -80}]


def test_unusable_existing_model_imports_with_a_warning(client, tmp_path):
    body = post(client, agent_source(tmp_path, with_farewell=False), "{not json")
    assert [warning["code"] for warning in body["warnings"]] == ["layout-not-adopted"]
    assert len(body["model"]["nodes"]) == 2


def test_partly_read_agent_says_so(client):
    body = post(client, KEYWORD_STYLE_AGENT)
    assert body["model"]["nodes"] == []
    assert [warning["code"] for warning in body["warnings"]] == ["buml-import-partial"]
    assert "could be read only up to an unsupported statement" in body["warnings"][0]["message"]
