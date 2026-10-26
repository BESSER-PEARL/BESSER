"""
A state body that mixes a code action with other replies survives the B-UML
round trip (v4 JSON -> Agent -> generated .py -> v4 JSON).

The exporter used to write only the code action of such a body, so
re-importing the .py lost the other replies.
"""

import os
import tempfile

from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.agent_diagram_converter import (
    agent_buml_to_json,
)

CODE = "def log_visit(session):\n    session.set('visited', True)"


def _node(node_id, type_, data):
    return {"id": node_id, "type": type_, "position": {"x": 0, "y": 0},
            "width": 160, "height": 80, "data": data}


def _roundtrip(bodies):
    payload = {"title": "MixedBody", "model": {
        "version": "4.0.0", "type": "AgentDiagram",
        "nodes": [
            _node("init", "StateInitialNode", {"name": ""}),
            _node("s", "AgentState", {"name": "greet", "replyType": "text", "bodies": bodies}),
        ],
        "edges": [{"id": "e", "type": "AgentStateTransitionInit", "source": "init", "target": "s",
                   "data": {"params": {}, "points": []}}],
    }}
    agent = process_agent_diagram(payload)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "agent.py")
        agent_model_to_code(agent, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    model = agent_buml_to_json(content)
    state = next(n for n in model["nodes"] if n["type"] == "AgentState")
    return [(row.get("replyType"), row.get("code") or row.get("name")) for row in state["data"]["bodies"]]


def test_code_mixed_with_text_keeps_every_reply_in_order():
    rows = _roundtrip([
        {"id": "r1", "name": "Hello!", "replyType": "text"},
        {"id": "r2", "name": "log_visit", "code": CODE, "replyType": "code"},
        {"id": "r3", "name": "Bye!", "replyType": "text"},
    ])
    assert [kind for kind, _ in rows] == ["text", "code", "text"]
    assert rows[0][1] == "Hello!" and rows[2][1] == "Bye!"
    assert "def log_visit(session):" in rows[1][1]


def test_a_code_only_body_still_round_trips():
    rows = _roundtrip([{"id": "r1", "name": "log_visit", "code": CODE, "replyType": "code"}])
    assert [kind for kind, _ in rows] == ["code"]
    assert "def log_visit(session):" in rows[0][1]
