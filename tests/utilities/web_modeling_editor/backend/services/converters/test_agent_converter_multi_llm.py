"""
Multi-LLM round-trip tests for the agent converters (Wave 2).

Two directions are exercised:

1. v4 JSON (``{nodes, edges}`` with ``AgentLLM`` data-only nodes plus a
   ``config.default_llm_name`` envelope) -> ``process_agent_diagram``
   -> Agent metamodel (``agent.llms`` / ``default_llm_name`` /
   per-element ``llm_name``).
2. Agent metamodel -> generated B-UML code (``agent_model_to_code``)
   -> ``agent_buml_to_json`` -> v4 JSON, asserting the AgentLLM nodes,
   the ``config.default_llm_name`` payload, and the per-element
   ``llm_name`` passthrough all survive.
"""

import os
import tempfile

import pytest

from besser.BUML.metamodel.state_machine.agent import Agent, ReasoningState
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.agent_diagram_converter import (
    agent_buml_to_json,
)


def _node(node_id, type_, data, x=0, y=0, width=160, height=80):
    return {
        "id": node_id,
        "type": type_,
        "position": {"x": x, "y": y},
        "width": width,
        "height": height,
        "data": data,
    }


def _multi_llm_payload():
    """A v4 agent diagram with two LLM definitions, an LLM-reply body and
    a reasoning state each bound to one of them, a RAG element bound by
    name, and ``config.default_llm_name`` pointing at the second LLM."""
    nodes = [
        _node("init-1", "StateInitialNode", {"name": ""}, width=45, height=45),
        _node(
            "llm-1",
            "AgentLLM",
            {
                "name": "fast",
                "provider": "openai",
                "parameters": {"model": "gpt-4o-mini", "temperature": 0.2},
                "num_previous_messages": 3,
                "global_context": "Be terse.",
            },
            width=200,
            height=90,
        ),
        _node(
            "llm-2",
            "AgentLLM",
            {
                "name": "big",
                "provider": "openai",
                "parameters": {"model": "gpt-4o"},
                "num_previous_messages": 1,
                "global_context": "",
            },
            width=200,
            height=90,
        ),
        _node(
            "rs-1",
            "AgentReasoningState",
            {
                "name": "reason",
                "llm_name": "fast",
                "max_steps": 5,
                "enable_task_planning": True,
                "stream_steps": True,
                "system_prompt": "Think.",
                "fallback_message": "",
            },
            width=200,
        ),
        _node(
            "as-1",
            "AgentState",
            {
                "name": "answer",
                "bodies": [
                    {
                        "id": "b-1",
                        "name": "AI response",
                        "replyType": "llm",
                        "llm_name": "big",
                    },
                ],
            },
        ),
        _node(
            "rag-1",
            "AgentRagElement",
            {"name": "manuals", "llm_name": "big"},
            width=120,
            height=110,
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
        "title": "multi_llm_demo",
        "model": {
            "version": "4.0.0",
            "type": "AgentDiagram",
            "nodes": nodes,
            "edges": edges,
        },
        "config": {"default_llm_name": "big"},
    }


@pytest.fixture
def multi_llm_agent():
    return process_agent_diagram(_multi_llm_payload())


class TestProcessMultiLlm:
    """v4 JSON -> Agent metamodel."""

    def test_llms_registered(self, multi_llm_agent):
        assert {llm.name for llm in multi_llm_agent.llms} == {"fast", "big"}
        fast = next(llm for llm in multi_llm_agent.llms if llm.name == "fast")
        assert fast.parameters == {"model": "gpt-4o-mini", "temperature": 0.2}
        assert fast.num_previous_messages == 3
        assert fast.global_context == "Be terse."

    def test_default_llm_applied_from_config(self, multi_llm_agent):
        assert multi_llm_agent.default_llm_name == "big"

    def test_default_falls_back_to_first_registered(self):
        payload = _multi_llm_payload()
        payload["config"] = {}
        agent = process_agent_diagram(payload)
        assert agent.default_llm_name == "fast"

    def test_unknown_default_llm_is_ignored(self):
        payload = _multi_llm_payload()
        payload["config"] = {"default_llm_name": "missing"}
        agent = process_agent_diagram(payload)
        # Auto-default (first registered) stays in place.
        assert agent.default_llm_name == "fast"

    def test_duplicate_llm_names_are_skipped(self):
        payload = _multi_llm_payload()
        dupe = dict(next(
            n for n in payload["model"]["nodes"] if n["type"] == "AgentLLM"
        ))
        dupe["id"] = "llm-dupe"
        payload["model"]["nodes"].append(dupe)
        agent = process_agent_diagram(payload)
        assert len(agent.llms) == 2

    def test_reasoning_state_bound_to_named_llm(self, multi_llm_agent):
        rs = next(
            s for s in multi_llm_agent.states if isinstance(s, ReasoningState)
        )
        assert rs.llm == "fast"
        assert rs.max_steps == 5

    def test_rag_bound_to_named_llm(self, multi_llm_agent):
        rags = list(multi_llm_agent.rags)
        assert len(rags) == 1
        assert rags[0].llm_name == "big"

    def test_llm_reply_body_keeps_llm_name(self, multi_llm_agent):
        answer = next(s for s in multi_llm_agent.states if s.name == "answer")
        llm_replies = [
            a for a in answer.body.actions if type(a).__name__ == "LLMReply"
        ]
        assert len(llm_replies) == 1
        assert llm_replies[0].llm_name == "big"


class TestMultiLlmRoundTrip:
    """Agent metamodel -> builder code -> v4 JSON."""

    @pytest.fixture
    def reemitted(self, multi_llm_agent):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.py")
            agent_model_to_code(multi_llm_agent, path)
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

    def test_agent_llm_nodes_reemitted(self, reemitted):
        llm_nodes = self._nodes_by_type(reemitted, "AgentLLM")
        assert {n["data"]["name"] for n in llm_nodes} == {"fast", "big"}
        fast = next(n for n in llm_nodes if n["data"]["name"] == "fast")
        assert fast["data"]["provider"] == "openai"
        assert fast["data"]["parameters"] == {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
        }
        assert fast["data"]["num_previous_messages"] == 3
        assert fast["data"]["global_context"] == "Be terse."

    def test_default_llm_name_reemitted_in_config(self, reemitted):
        assert reemitted.get("config", {}).get("default_llm_name") == "big"

    def test_reasoning_state_llm_name_reemitted(self, reemitted):
        rs = self._reasoning_nodes(reemitted)[0]
        assert rs["data"]["stateType"] == "reasoning"
        assert rs["data"]["llm_name"] == "fast"

    def test_rag_llm_name_reemitted(self, reemitted):
        rag = self._nodes_by_type(reemitted, "AgentRagElement")[0]
        assert rag["data"]["name"] == "manuals"
        assert rag["data"]["llm_name"] == "big"

    def test_llm_reply_body_llm_name_reemitted(self, reemitted):
        answer = next(
            n for n in self._nodes_by_type(reemitted, "AgentState")
            if n["data"]["name"] == "answer"
        )
        llm_rows = [
            row for row in answer["data"].get("bodies", [])
            if row.get("replyType") == "llm"
        ]
        assert len(llm_rows) == 1
        assert llm_rows[0]["llm_name"] == "big"

    def test_full_circle_restores_llm_registry(self, reemitted):
        restored = process_agent_diagram(
            {"model": reemitted, "config": reemitted.get("config", {})}
        )
        assert {llm.name for llm in restored.llms} == {"fast", "big"}
        assert restored.default_llm_name == "big"
        rs = next(s for s in restored.states if isinstance(s, ReasoningState))
        assert rs.llm == "fast"


def test_builder_roundtrip_from_metamodel(tmp_path):
    """Port of the development backend's
    ``test_agent_roundtrip_preserves_multi_llm_and_reasoning`` — Agent
    built directly via the metamodel API survives builder -> JSON ->
    processor with both LLMs and the designated default intact."""
    agent = Agent("RoundTripAgent")
    agent.new_llm(name="fast", provider="openai", parameters={"model": "gpt-4o-mini"})
    agent.new_llm(name="big", provider="openai", parameters={"model": "gpt-4o"})
    agent.set_default_llm("big")
    initial = agent.new_state("initial", initial=True)
    reason = agent.new_reasoning_state(
        name="reason", llm="fast", max_steps=5, system_prompt="Think."
    )
    initial.go_to(reason)

    code_path = os.path.join(str(tmp_path), "agent.py")
    agent_model_to_code(agent, code_path)
    with open(code_path, encoding="utf-8") as handle:
        content = handle.read()

    json_model = agent_buml_to_json(content)
    restored = process_agent_diagram(
        {"model": json_model, "config": json_model.get("config", {})}
    )

    assert {llm.name for llm in restored.llms} == {"fast", "big"}
    assert restored.default_llm_name == "big"
    reasoning_states = [
        s for s in restored.states if isinstance(s, ReasoningState)
    ]
    assert len(reasoning_states) == 1
    assert reasoning_states[0].llm == "fast"
    assert reasoning_states[0].max_steps == 5
