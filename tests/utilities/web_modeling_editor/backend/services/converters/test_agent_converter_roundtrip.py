"""Round-trip tests for the agent converters covering the multi-LLM and
reasoning-extension surface added in 7.8.0.

Direction exercised: Agent (B-UML) -> generated B-UML code -> JSON (buml_to_json)
-> Agent (json_to_buml). The new fields (multiple LLMs + designated default,
per-reasoning-state LLM, tools/skills/workspaces) must survive the trip.
"""
import os

from besser.BUML.metamodel.state_machine.agent import Agent, RAGReply, RAGTextSplitter, RAGVectorStore, ReasoningState
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    agent_buml_to_json,
    process_agent_diagram,
)


def _build_reasoning_agent() -> Agent:
    agent = Agent("RoundTripAgent")
    agent.new_llm(name="fast", provider="openai", parameters={"model": "gpt-4o-mini"})
    agent.new_llm(name="big", provider="openai", parameters={"model": "gpt-4o"})
    agent.set_default_llm("big")
    agent.new_tool(name="lookup", description="Look things up", code="def lookup(x):\n    return x")
    agent.new_skill(name="greet", content="Be friendly", description="greeting skill")
    agent.new_workspace(
        name="docs", path="/tmp/docs", description="doc store", writable=True, max_read_bytes=1000
    )
    vector_store = RAGVectorStore(
        embedding_provider="openai",
        embedding_parameters={"api_key_property": "nlp.OPENAI_API_KEY"},
        persist_directory="vector_store/docs",
    )
    splitter = RAGTextSplitter(
        splitter_type="recursive_character",
        chunk_size=1000,
        chunk_overlap=100,
    )
    agent.new_rag(
        name="docs_rag",
        vector_store=vector_store,
        splitter=splitter,
        llm_name="fast",
        llm_prompt="Answer only using the docs corpus.",
        k=6,
        num_previous_messages=3,
    )
    initial = agent.new_state("initial", initial=True)
    initial.set_body(Body("initial_body", actions=[RAGReply("docs_rag", prompt="Use only cited docs.")]))
    reason = agent.new_reasoning_state(name="reason", llm="fast", max_steps=5, system_prompt="Think.")
    initial.go_to(reason)
    return agent


def test_agent_roundtrip_preserves_multi_llm_and_reasoning(tmp_path):
    agent = _build_reasoning_agent()
    code_path = os.path.join(str(tmp_path), "agent.py")
    agent_model_to_code(agent, code_path)
    with open(code_path, encoding="utf-8") as handle:
        content = handle.read()

    json_model = agent_buml_to_json(content)

    # The reasoning state must be emitted in the canonical shape the editor can
    # deserialize: an AgentState carrying stateType "reasoning". The legacy
    # AgentReasoningState element type was removed from the editor registry, so
    # emitting it would crash on import.
    json_elements = list(json_model.get("elements", {}).values())
    reasoning_elements = [el for el in json_elements if el.get("stateType") == "reasoning"]
    assert len(reasoning_elements) == 1
    assert reasoning_elements[0]["type"] == "AgentState"
    assert all(el.get("type") != "AgentReasoningState" for el in json_elements)

    # process_agent_diagram consumes the frontend envelope: the apollon model
    # under "model" and the diagram config surfaced at the top level.
    restored = process_agent_diagram({"model": json_model, "config": json_model.get("config", {})})

    # Multi-LLM: both LLMs and the designated default survive.
    assert {llm.name for llm in restored.llms} == {"fast", "big"}
    assert restored.default_llm_name == "big"

    # Reasoning state: present, bound to its LLM by name, config preserved.
    reasoning_states = [s for s in restored.states if isinstance(s, ReasoningState)]
    assert len(reasoning_states) == 1
    assert reasoning_states[0].llm == "fast"
    assert reasoning_states[0].max_steps == 5

    # Reasoning primitives survive with their distinguishing fields.
    assert {t.name for t in restored.tools} == {"lookup"}
    assert {s.name for s in restored.skills} == {"greet"}
    workspaces = {w.name: w for w in restored.workspaces}
    assert set(workspaces) == {"docs"}
    assert workspaces["docs"].path == "/tmp/docs"
    assert workspaces["docs"].writable is True

    # RAG configuration survives with newly surfaced arguments.
    rags = {r.name: r for r in restored.rags}
    assert set(rags) == {"docs_rag"}
    assert rags["docs_rag"].llm_name == "fast"
    assert rags["docs_rag"].llm_prompt == "Answer only using the docs corpus."
    assert rags["docs_rag"].k == 6
    assert rags["docs_rag"].num_previous_messages == 3

    # RAGReply action prompt survives the converters.
    initial_state = next(s for s in restored.states if s.name == "initial")
    rag_actions = [a for a in (initial_state.body.actions if initial_state.body else []) if isinstance(a, RAGReply)]
    assert len(rag_actions) == 1
    assert rag_actions[0].rag_db_name == "docs_rag"
    assert rag_actions[0].prompt == "Use only cited docs."

