"""The agent (BAF) ZIP carries the element-level traces of its generator to the traced endpoint.

The converted model is replaced by a small one whose elements, its body included, have ids (the editor's
agent processor gives a state's body none: it is built from several editor actions), so each kind of
region can be checked line by line; the rest is the real path (``generate_agent_output``, packaging, the
trace builder). ``test_traced_generation_all.py`` runs the editor's own agent diagram.
"""

import base64
import io
import json
import pathlib
import zipfile

import pytest
from unittest.mock import patch

from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply, WebSocketPlatform
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.generators.agents import agent_personalization
from besser.utilities.web_modeling_editor.backend.services.generation import agents as agent_generation

TRACED = "/besser_api/traceability/generate-output"
FIXTURES = pathlib.Path(__file__).parents[3] / "generators" / "fixtures"
AGENT_DIAGRAM = {"id": "agent-diagram",
                 **json.loads((FIXTURES / "library_template_agent_diagram.json").read_text("utf-8"))}


def _agent_with_ids() -> Agent:
    agent = Agent("library_agent")
    agent.platforms.append(WebSocketPlatform())
    hello = agent.new_intent("hello", ["hi", "hello"])
    initial = agent.new_state("initial", initial=True)
    chat = agent.new_state("chat")
    initial.set_body(Body("initial_body", actions=[AgentReply("Welcome!")]))
    initial.when_intent_matched(hello).go_to(chat)
    hello.id, initial.id, chat.id = "intent-hello", "state-initial", "state-chat"
    initial.body.id, initial.transitions[0].id = "body-initial", "transition-hello"
    agent.id = "agent-model"
    return agent


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_generation, "process_agent_diagram", lambda _data: _agent_with_ids())


def _payload(**extra):
    return {**AGENT_DIAGRAM, "generator": "agent", "config": None, "actorName": "Ada", **extra}


def _fragments(trace, element_ids):
    """{element id: [(rule, path, start, end)]} of the code fragments each listed element is linked to."""
    artifacts = {a["id"]: a for a in trace["artifacts"]}
    found = {}
    for link in trace["traceLinks"]:
        source, target = artifacts[link["sourceId"]], artifacts[link["targetId"]]
        if source.get("elementId") in element_ids and "startLine" in target:
            found.setdefault(source["elementId"], []).append(
                (link["evidence"][0]["rule"], target["path"], target["startLine"], target["endLine"]))
    return found


def test_the_traced_endpoint_returns_the_element_regions_of_the_agent_files(client):
    response = client.post(TRACED, json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(body["contentBase64"]))) as archive:
        lines = archive.read("library_agent.py").decode("utf-8").split("\n")
        assert {"agent_model.py", "config.yaml", "readme.txt"} <= set(archive.namelist())

    fragments = _fragments(body["trace"], {"intent-hello", "state-initial", "state-chat", "body-initial",
                                           "transition-hello"})
    def line_of(element_id, rule):
        (_, path, start, end), = [f for f in fragments[element_id] if f[0].endswith(f"#{rule}")]
        assert path == "library_agent.py" and start <= end
        return lines[start - 1]

    assert line_of("intent-hello", "intent").startswith("hello = agent.new_intent('hello'")
    assert line_of("state-initial", "state") == "initial = agent.new_state('initial', initial=True)"
    assert line_of("state-chat", "state") == "chat = agent.new_state('chat')"
    assert line_of("body-initial", "body") == "def initial_body(session: Session):"
    assert line_of("transition-hello", "transition") == "initial.when_intent_matched(hello).go_to(chat)"
    # The config and the readme are linked to the agent as a whole.
    artifacts = {a["id"]: a for a in body["trace"]["artifacts"]}
    whole_file = {artifacts[link["targetId"]]["path"] for link in body["trace"]["traceLinks"]
                  if artifacts[link["sourceId"]].get("elementId") == "agent-model"
                  and "startLine" not in artifacts[link["targetId"]]
                  and not link["evidence"][0]["rule"].startswith("generator:")}
    assert {"config.yaml", "readme.txt", "library_agent.py"} <= whole_file


def test_the_delivery_is_the_one_the_plain_endpoint_makes(client):
    traced = client.post(TRACED, json=_payload()).json()
    plain = client.post("/besser_api/generate-output", json={k: v for k, v in _payload().items() if k != "actorName"})

    def entries(data):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}

    assert entries(base64.b64decode(traced["contentBase64"])) == entries(plain.content)


def test_the_output_of_a_zip_with_traces_has_the_paths_of_its_entries():
    generators = []
    zip_buffer, name = agent_generation.generate_agent_files(_agent_with_ids(), None, generators=generators)
    output = agent_generation._with_traces(agent_generation.agent_zip_output(zip_buffer, name), generators)
    with zipfile.ZipFile(zip_buffer) as archive:
        names = set(archive.namelist())
    assert output.traced_files and set(output.traced_files) <= names
    assert "library_agent.py" in output.traced_files


def test_without_exactly_one_generator_the_output_stays_file_level():
    output = agent_generation.agent_zip_output(io.BytesIO(), "x.zip")
    assert agent_generation._with_traces(output, []) is output


def test_the_regions_of_a_file_an_llm_wrote_are_not_delivered_as_template_output():
    generators = []
    with patch.object(agent_personalization, "call_openai_chat",
                      side_effect=lambda *_a, **_k: "\n".join(f"{n}. rewritten" for n in range(1, 30))):
        zip_buffer, name = agent_generation.generate_agent_files(
            _agent_with_ids(), {"agentStyle": "formal", "openaiApiKey": "k"}, generators=generators)
    (generator,) = generators
    assert "library_agent.py" in generator.ai_files
    output = agent_generation._with_traces(agent_generation.agent_zip_output(zip_buffer, name), generators)
    assert "library_agent.py" not in output.traced_files
    assert "config.yaml" in output.traced_files


def test_the_traced_endpoint_records_what_the_personalization_llm_wrote_as_its_work(client):
    """A file the personalization LLM rewrote is linked by AI evidence of that LLM, not by a rule."""
    rewritten = "\n".join(f"{n}. rewritten" for n in range(1, 30))
    with patch.object(agent_personalization, "call_openai_chat", side_effect=lambda *_a, **_k: rewritten) as llm:
        response = client.post(TRACED, json=_payload(config={"agentStyle": "formal", "openaiApiKey": "k"}))
    assert response.status_code == 200, response.text
    assert llm.called
    trace = response.json()["trace"]
    artifacts = {a["id"]: a for a in trace["artifacts"]}
    root = next(a for a in trace["artifacts"] if a.get("elementId") == "agent-diagram")
    evidence_of = {artifacts[link["targetId"]]["path"]: link["evidence"][0] for link in trace["traceLinks"]
                   if link["sourceId"] == root["id"]}
    for path in ("library_agent.py", "personalized_agent_model.py"):
        assert evidence_of[path]["type"] == "AIEvidence"
        assert evidence_of[path]["algorithm"] == agent_personalization.DEFAULT_OPENAI_MODEL
    assert evidence_of["config.yaml"] == {**evidence_of["config.yaml"], "type": "RuleEvidence",
                                          "rule": "generator:agent"}
    # The LLM is a tool of the run, called after the generator, with the provider that served it.
    tools = {tool["id"]: tool for tool in trace["tools"]}
    (operation,) = trace["operations"]
    called = [tools[call["toolId"]] for call in sorted(operation["toolCalls"], key=lambda c: c["sequenceNumber"])]
    assert [(t["type"], t["name"], t.get("provider")) for t in called] == [
        ("CodeGenerator", "agent", None), ("LLM", agent_personalization.DEFAULT_OPENAI_MODEL, "openai")]
    # No region of the rewritten file is presented as template output.
    assert not any(a.get("path") == "library_agent.py" and "startLine" in a for a in trace["artifacts"])
