"""
Roundtrip tests for the agent reasoning primitives (Wave 2).

v4 JSON (``{nodes, edges}``) -> ``process_agent_diagram`` -> Agent metamodel
-> ``agent_model_to_code`` -> ``agent_buml_to_json`` -> v4 JSON.

Covers ``AgentTool``, ``AgentSkill``, ``AgentWorkspace`` and reasoning
states — the four reasoning primitives the React Flow frontend ships on
the AgentDiagram palette (Capabilities / Reasoning sections). Shapes
mirror the frontend's ``NodeProps`` definitions.

A reasoning state is the canonical v4 folded shape: ``type: "AgentState"``
with ``data.stateType == "reasoning"``. The legacy ``type:
"AgentReasoningState"`` node (pre-fold) is covered separately below as a
back-compat read case.
"""

import os
import tempfile

import pytest

from besser.BUML.metamodel.state_machine.agent import ReasoningState
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.agent_diagram_converter import (
    agent_buml_to_json,
)


TOOL_CODE = "def ping():\n    return 'pong'\n"


def _node(node_id, type_, data, x=0, y=0, width=160, height=80):
    return {
        "id": node_id,
        "type": type_,
        "position": {"x": x, "y": y},
        "width": width,
        "height": height,
        "data": data,
    }


def _reasoning_payload():
    """A v4 agent diagram with one of each reasoning primitive.

    The initial-node edge targets the reasoning state, exercising the
    "reasoning state can be the initial state" path.

    ``llm_name`` is empty — the editor's "(use default)" choice. A
    non-empty name must reference a registered LLM (``AgentLLM``
    definitions arrive with the multi-LLM wave); see
    ``test_unknown_llm_name_fails_validation``.
    """
    nodes = [
        _node("init-1", "StateInitialNode", {"name": ""}, width=45, height=45),
        _node(
            "rs-1",
            "AgentState",
            {
                "name": "reason",
                "stateType": "reasoning",
                "llm_name": "",
                "max_steps": 15,
                "enable_task_planning": True,
                "stream_steps": False,
                "system_prompt": "Be concise.",
                "fallback_message": "Sorry, something broke.",
            },
            width=200,
        ),
        _node(
            "as-1",
            "AgentState",
            {
                "name": "farewell",
                "bodies": [
                    {"id": "b-1", "name": "Bye!", "replyType": "text"},
                ],
            },
        ),
        _node(
            "tool-1",
            "AgentTool",
            {"name": "ping", "description": "Ping the server.", "code": TOOL_CODE},
        ),
        _node(
            "skill-1",
            "AgentSkill",
            {
                "name": "GreetByName",
                "content": "Always greet the user by name.",
                "description": "Greeting playbook.",
            },
        ),
        _node(
            "ws-1",
            "AgentWorkspace",
            {
                "name": "cinema",
                "path": "/tmp/cinema",
                "description": "Cinema files.",
                "writable": False,
                "max_read_bytes": 50_000,
            },
        ),
    ]
    edges = [
        {
            "id": "e-init",
            "source": "init-1",
            "target": "rs-1",
            "type": "AgentStateTransitionInit",
            "data": {"points": []},
        },
        {
            "id": "e-1",
            "source": "rs-1",
            "target": "as-1",
            "type": "AgentStateTransition",
            "data": {
                "transitionType": "predefined",
                "predefined": {"predefinedType": "when_no_intent_matched"},
            },
        },
    ]
    return {
        "title": "reasoning_demo",
        "model": {
            "version": "4.0.0",
            "type": "AgentDiagram",
            "nodes": nodes,
            "edges": edges,
        },
    }


@pytest.fixture
def reasoning_agent():
    return process_agent_diagram(_reasoning_payload())


class TestProcessReasoningPrimitives:
    """v4 JSON -> Agent metamodel."""

    def test_tool_registered(self, reasoning_agent):
        assert len(reasoning_agent.tools) == 1
        tool = reasoning_agent.tools[0]
        assert tool.name == "ping"
        assert tool.description == "Ping the server."
        assert tool.code == TOOL_CODE

    def test_skill_registered(self, reasoning_agent):
        assert len(reasoning_agent.skills) == 1
        skill = reasoning_agent.skills[0]
        assert skill.name == "GreetByName"
        assert skill.content == "Always greet the user by name."
        assert skill.description == "Greeting playbook."

    def test_workspace_registered(self, reasoning_agent):
        assert len(reasoning_agent.workspaces) == 1
        ws = reasoning_agent.workspaces[0]
        assert ws.name == "cinema"
        assert ws.path == "/tmp/cinema"
        assert ws.description == "Cinema files."
        assert ws.writable is False
        assert ws.max_read_bytes == 50_000

    def test_reasoning_state_is_initial_with_fields(self, reasoning_agent):
        reasoning_states = [
            s for s in reasoning_agent.states if isinstance(s, ReasoningState)
        ]
        assert len(reasoning_states) == 1
        rs = reasoning_states[0]
        assert rs.name == "reason"
        assert rs.initial is True
        # Empty llm_name maps to None — "resolve the agent default at
        # codegen time".
        assert rs.llm is None
        assert rs.max_steps == 15
        assert rs.enable_task_planning is True
        assert rs.stream_steps is False
        assert rs.system_prompt == "Be concise."
        assert rs.fallback_message == "Sorry, something broke."

    def test_unknown_llm_name_fails_validation(self):
        """A free-entry llm_name that no registered LLM matches must fail
        loudly (metamodel ``_validate_llm_references``) instead of silently
        generating an agent that cannot resolve its LLM. ``AgentLLM``
        diagram definitions arrive with the multi-LLM wave."""
        payload = _reasoning_payload()
        rs_node = next(
            n for n in payload["model"]["nodes"]
            if n["type"] == "AgentState" and n["data"].get("stateType") == "reasoning"
        )
        rs_node["data"]["llm_name"] = "gpt-4o-mini"
        with pytest.raises(ValueError, match="not.*registered|new_llm"):
            process_agent_diagram(payload)

    def test_transition_from_reasoning_state(self, reasoning_agent):
        rs = next(s for s in reasoning_agent.states if isinstance(s, ReasoningState))
        assert len(rs.transitions) == 1
        assert rs.transitions[0].dest.name == "farewell"

    def test_duplicate_primitives_are_skipped(self):
        payload = _reasoning_payload()
        # Duplicate every primitive node under fresh ids.
        dupes = []
        for n in payload["model"]["nodes"]:
            if n["type"] in ("AgentTool", "AgentSkill", "AgentWorkspace"):
                dupe = dict(n)
                dupe["id"] = n["id"] + "-dupe"
                dupes.append(dupe)
        payload["model"]["nodes"].extend(dupes)
        agent = process_agent_diagram(payload)
        assert len(agent.tools) == 1
        assert len(agent.skills) == 1
        assert len(agent.workspaces) == 1


class TestLegacyReasoningStateBackCompat:
    """Pre-fold diagrams used a dedicated ``AgentReasoningState`` node
    type. The processor must keep reading that shape for back-compat even
    though the converter no longer emits it — canonical v4 output folds
    reasoning states into ``type: "AgentState"`` + ``data.stateType ==
    "reasoning"`` (see ``TestReasoningRoundTrip`` above)."""

    def _legacy_payload(self):
        nodes = [
            _node("init-1", "StateInitialNode", {"name": ""}, width=45, height=45),
            _node(
                "rs-1",
                "AgentReasoningState",
                {
                    "name": "reason",
                    "llm_name": "",
                    "max_steps": 15,
                    "enable_task_planning": True,
                    "stream_steps": False,
                    "system_prompt": "Be concise.",
                    "fallback_message": "Sorry, something broke.",
                },
                width=200,
            ),
        ]
        edges = [
            {
                "id": "e-init",
                "source": "init-1",
                "target": "rs-1",
                "type": "AgentStateTransitionInit",
                "data": {"points": []},
            },
        ]
        return {
            "title": "legacy_reasoning_demo",
            "model": {
                "version": "4.0.0",
                "type": "AgentDiagram",
                "nodes": nodes,
                "edges": edges,
            },
        }

    def test_legacy_agent_reasoning_state_type_still_imports(self):
        agent = process_agent_diagram(self._legacy_payload())
        reasoning_states = [s for s in agent.states if isinstance(s, ReasoningState)]
        assert len(reasoning_states) == 1
        rs = reasoning_states[0]
        assert rs.name == "reason"
        assert rs.initial is True
        assert rs.llm is None
        assert rs.max_steps == 15
        assert rs.enable_task_planning is True
        assert rs.stream_steps is False
        assert rs.system_prompt == "Be concise."
        assert rs.fallback_message == "Sorry, something broke."


class TestReasoningRoundTrip:
    """Agent metamodel -> builder code -> v4 JSON."""

    @pytest.fixture
    def reemitted(self, reasoning_agent):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.py")
            agent_model_to_code(reasoning_agent, path)
            with open(path, encoding="utf-8") as f:
                code = f.read()
        return agent_buml_to_json(code)

    def _nodes_by_type(self, payload, node_type):
        return [n for n in payload.get("nodes", []) if n.get("type") == node_type]

    def _reasoning_nodes(self, payload):
        """AgentState nodes folded into a reasoning state (``data.stateType
        == "reasoning"``) — the canonical v4 emission shape."""
        return [
            n for n in self._nodes_by_type(payload, "AgentState")
            if (n.get("data") or {}).get("stateType") == "reasoning"
        ]

    def test_tool_node_reemitted(self, reemitted):
        tools = self._nodes_by_type(reemitted, "AgentTool")
        assert len(tools) == 1
        data = tools[0]["data"]
        assert data["name"] == "ping"
        assert data["description"] == "Ping the server."
        assert data["code"] == TOOL_CODE

    def test_skill_node_reemitted(self, reemitted):
        skills = self._nodes_by_type(reemitted, "AgentSkill")
        assert len(skills) == 1
        data = skills[0]["data"]
        assert data["name"] == "GreetByName"
        assert data["content"] == "Always greet the user by name."
        assert data["description"] == "Greeting playbook."

    def test_workspace_node_reemitted(self, reemitted):
        workspaces = self._nodes_by_type(reemitted, "AgentWorkspace")
        assert len(workspaces) == 1
        data = workspaces[0]["data"]
        assert data["name"] == "cinema"
        assert data["path"] == "/tmp/cinema"
        assert data["description"] == "Cinema files."
        assert data["writable"] is False
        assert data["max_read_bytes"] == 50_000

    def test_reasoning_state_node_reemitted(self, reemitted):
        rs_nodes = self._reasoning_nodes(reemitted)
        assert len(rs_nodes) == 1
        data = rs_nodes[0]["data"]
        assert data["stateType"] == "reasoning"
        assert data["name"] == "reason"
        assert data["llm_name"] == ""
        assert data["max_steps"] == 15
        assert data["enable_task_planning"] is True
        assert data["stream_steps"] is False
        assert data["system_prompt"] == "Be concise."
        assert data["fallback_message"] == "Sorry, something broke."

    def test_initial_edge_targets_reasoning_state(self, reemitted):
        rs_node = self._reasoning_nodes(reemitted)[0]
        init_node = self._nodes_by_type(reemitted, "StateInitialNode")[0]
        init_edges = [
            e for e in reemitted.get("edges", [])
            if e.get("type") == "AgentStateTransitionInit"
        ]
        assert len(init_edges) == 1
        assert init_edges[0]["source"] == init_node["id"]
        assert init_edges[0]["target"] == rs_node["id"]

    def test_transition_reemitted_between_states(self, reemitted):
        rs_node = self._reasoning_nodes(reemitted)[0]
        as_node = next(
            n for n in self._nodes_by_type(reemitted, "AgentState")
            if n["data"]["name"] == "farewell"
        )
        transitions = [
            e for e in reemitted.get("edges", [])
            if e.get("type") == "AgentStateTransition"
        ]
        assert len(transitions) == 1
        assert transitions[0]["source"] == rs_node["id"]
        assert transitions[0]["target"] == as_node["id"]
        assert (
            transitions[0]["data"]["predefined"]["predefinedType"]
            == "when_no_intent_matched"
        )
