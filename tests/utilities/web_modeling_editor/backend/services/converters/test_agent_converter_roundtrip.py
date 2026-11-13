"""Round-trip tests for the agent converters covering the multi-LLM and
reasoning-extension surface, plus the ten new provider families added in 7.9.0.

Direction exercised: Agent (B-UML) -> generated B-UML code -> JSON (buml_to_json)
-> Agent (json_to_buml). The new fields (multiple LLMs + designated default,
per-reasoning-state LLM, tools/skills/workspaces) must survive the trip.

The agent live-testing surface (session variables, stored / silent replies, custom
input prompts, GUI replies and agent GUIs, form submissions, GUI events, multi-MIME
file transitions) is exercised in both directions at the end of the module.
"""
import copy
import os

import pytest

from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    FileTypeMatcher,
    FormSubmitMatcher,
    GUIEvent,
    GUIReplyAction,
    RAGReply,
    RAGTextSplitter,
    RAGVectorStore,
    ReasoningState,
)
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    agent_buml_to_json,
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _serialize_gui_model,
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
        use_hybrid_rag=True,
        bm25_weight=0.7,
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

    # The reasoning state must be emitted in the canonical v4 shape the editor can
    # load: an AgentState node carrying data.stateType "reasoning". The legacy
    # AgentReasoningState type is no longer registered by the editor.
    json_nodes = json_model["nodes"]
    reasoning_nodes = [n for n in json_nodes if n["data"].get("stateType") == "reasoning"]
    assert len(reasoning_nodes) == 1
    assert reasoning_nodes[0]["type"] == "AgentState"
    assert all(n.get("type") != "AgentReasoningState" for n in json_nodes)
    # Components never become canvas nodes.
    component_types = {c["type"] for c in json_model["components"].values()}
    assert {"AgentLLM", "AgentTool", "AgentSkill", "AgentWorkspace", "AgentRagElement"} <= component_types
    assert not component_types & {n["type"] for n in json_nodes}

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
    # Hybrid retrieval settings survive BUML -> JSON -> BUML
    assert rags["docs_rag"].use_hybrid_rag is True
    assert rags["docs_rag"].bm25_weight == 0.7

    # RAGReply action prompt survives the converters.
    initial_state = next(s for s in restored.states if s.name == "initial")
    rag_actions = [a for a in (initial_state.body.actions if initial_state.body else []) if isinstance(a, RAGReply)]
    assert len(rag_actions) == 1
    assert rag_actions[0].rag_db_name == "docs_rag"
    assert rag_actions[0].prompt == "Use only cited docs."


# ---------------------------------------------------------------------------
# New LLM provider round-trip — parametrized over all 10 new families
# ---------------------------------------------------------------------------

_NEW_PROVIDERS = [
    "mistral",
    "deepseek",
    "google",
    "meta",
    "anthropic",
    "qwen",
    "xai",
    "groq",
    "together",
    "openrouter",
]

# Representative model names per provider (used as the 'model' parameter so the
# generated code is realistic, not just a bare empty-parameters instantiation).
_PROVIDER_MODEL = {
    "mistral": "mistral-small-latest",
    "deepseek": "deepseek-chat",
    "google": "gemini-2.5-flash",
    "meta": "Llama-3.3-70B-Instruct",
    "anthropic": "claude-sonnet-4-5",
    "qwen": "qwen-plus",
    "xai": "grok-3-mini",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openrouter": "openai/gpt-4o",
}

# Exact class names as they appear in generated imports (non-trivial capitalisation).
_PROVIDER_CLASS = {
    "mistral": "LLMMistral",
    "deepseek": "LLMDeepSeek",
    "google": "LLMGoogle",
    "meta": "LLMMeta",
    "anthropic": "LLMAnthropic",
    "qwen": "LLMQwen",
    "xai": "LLMxAI",
    "groq": "LLMGroq",
    "together": "LLMTogether",
    "openrouter": "LLMOpenRouter",
}


@pytest.mark.parametrize("provider", _NEW_PROVIDERS)
def test_new_provider_roundtrip_preserves_provider_key(provider, tmp_path):
    """Agent (B-UML) → code → JSON → Agent: provider key and model survive intact.

    This catches missing or misnamed entries in _LLM_PROVIDERS, _direct_llm_class_to_provider,
    and the generator import block — all of which would break the round-trip silently.
    """
    model = _PROVIDER_MODEL[provider]
    agent = Agent(f"agent_{provider}")
    agent.new_llm(name="main_llm", provider=provider, parameters={"model": model})

    code_path = os.path.join(str(tmp_path), "agent.py")
    from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        agent_buml_to_json,
        process_agent_diagram,
    )

    agent_model_to_code(agent, code_path)
    with open(code_path, encoding="utf-8") as fh:
        code = fh.read()

    # The generated code must import the correct provider class.
    expected_class = _PROVIDER_CLASS[provider]
    assert expected_class in code, (
        f"Generated agent.py for provider '{provider}' missing import of '{expected_class}'"
    )

    json_model = agent_buml_to_json(code)
    restored = process_agent_diagram({"model": json_model, "config": json_model.get("config", {})})

    # Exactly one LLM must survive with its original name.
    assert len(list(restored.llms)) == 1
    llm = next(iter(restored.llms))
    assert llm.name == "main_llm"

    # The restored object must be the *correct* LLMWrapper subclass. Without this,
    # a wrong entry in _direct_llm_class_to_provider (e.g. LLMMistral -> "deepseek")
    # would still satisfy the name/parameter assertions below and pass silently.
    assert type(llm).__name__ == expected_class, (
        f"provider '{provider}' round-tripped to {type(llm).__name__}, expected {expected_class}"
    )

    # The model parameter must be preserved through the round-trip.
    assert llm.parameters.get("model") == model, (
        f"model parameter lost for provider '{provider}': "
        f"expected {model!r}, got {llm.parameters.get('model')!r}"
    )


def test_agent_roundtrip_preserves_ollama_llm_and_embedding(tmp_path):
    """Ollama LLM provider and Ollama RAG embedding survive the converter
    round-trip (Agent -> B-UML code -> JSON -> Agent), including base_url/model."""
    from besser.BUML.metamodel.state_machine.agent import LLMOllama

    agent = Agent("OllamaAgent")
    agent.new_llm(
        name="local",
        provider="ollama",
        parameters={"base_url": "http://localhost:11434", "model": "llama3"},
    )
    agent.set_default_llm("local")
    vector_store = RAGVectorStore(
        embedding_provider="ollama",
        embedding_parameters={"base_url": "http://localhost:11434", "model": "nomic-embed-text"},
        persist_directory="vector_store/local",
    )
    splitter = RAGTextSplitter(splitter_type="recursive_character", chunk_size=1000, chunk_overlap=100)
    agent.new_rag(
        name="local_rag",
        vector_store=vector_store,
        splitter=splitter,
        llm_name="local",
        llm_prompt="Answer only using the docs corpus.",
        k=4,
        num_previous_messages=0,
    )
    initial = agent.new_state("initial", initial=True)
    initial.set_body(Body("initial_body", actions=[RAGReply("local_rag", prompt="Use only cited docs.")]))

    code_path = os.path.join(str(tmp_path), "agent.py")
    agent_model_to_code(agent, code_path)
    with open(code_path, encoding="utf-8") as handle:
        content = handle.read()

    json_model = agent_buml_to_json(content)
    restored = process_agent_diagram({"model": json_model, "config": json_model.get("config", {})})

    # The Ollama LLM survives as an LLMOllama carrying its base_url + model.
    llms = {llm.name: llm for llm in restored.llms}
    assert "local" in llms
    assert isinstance(llms["local"], LLMOllama)
    assert llms["local"].parameters.get("base_url") == "http://localhost:11434"
    assert llms["local"].parameters.get("model") == "llama3"

    # The Ollama RAG embedding provider + its parameters survive the trip.
    rags = {r.name: r for r in restored.rags}
    assert set(rags) == {"local_rag"}
    vector = rags["local_rag"].vector_store
    assert vector.embedding_provider == "ollama"
    assert vector.embedding_parameters.get("base_url") == "http://localhost:11434"
    assert vector.embedding_parameters.get("model") == "nomic-embed-text"


# ---------------------------------------------------------------------------
# Agent live-testing surface. Both directions are exercised:
#   JSON -> B-UML -> code -> JSON           (editor fields survive)
#   JSON -> B-UML -> code -> JSON -> B-UML  (metamodel objects are identical)
#   B-UML -> code -> exec                   (the emitted module rebuilds the agent)
# ---------------------------------------------------------------------------

# Text with backslashes and both quote kinds: must be stored verbatim in the
# metamodel and survive every direction unchanged (no double escaping).
_TRICKY = "It's C:\\path\\to 'x' \"y\" \\n end\\"

_SIGNUP_GUI = {
    "pages": [{
        "id": "page-signup", "name": "Signup",
        "frames": [{"component": {
            "type": "wrapper", "attributes": {"id": "wrapper-signup"},
            "components": [
                {"type": "text", "tagName": "h2", "attributes": {"id": "title-1"},
                 "components": [{"type": "textnode", "content": "Tell us about you"}]},
                {"type": "input", "tagName": "input",
                 "attributes": {"id": "email-1", "type": "email", "name": "email", "data-field-name": "email"}},
            ],
        }}],
    }],
    "styles": [],
}


def _action(action_id, action_type, **fields):
    """A v4 AgentState action row (inline in ``data.bodies`` / ``data.fallbackBodies``)."""
    return {"id": action_id, "name": fields.pop("name", action_type), "actionType": action_type, **fields}


def _transition(edge_id, source, target, **data):
    return {"id": edge_id, "type": "AgentStateTransition", "source": source, "target": target,
            "sourceHandle": "Right", "targetHandle": "Left", "data": {"points": [], **data}}


def _state(node_id, name, actions, fallback=(), initial=False):
    return {"id": node_id, "type": "AgentState", "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
            "data": {"name": name, "initial": initial, "stateType": "standard",
                     "bodies": list(actions), "fallbackBodies": list(fallback)}}


def _live_testing_components():
    return {
        "llm-1": {"id": "llm-1", "type": "AgentLLM", "name": "fast", "provider": "openai",
                  "parameters": {"model": "gpt-4o-mini"}},
        "int-1": {"id": "int-1", "type": "AgentIntent", "name": "greet", "bodies": ["int-1-b"]},
        "int-1-b": {"id": "int-1-b", "type": "AgentIntentBody", "name": _TRICKY, "owner": "int-1"},
        "rag-1": {"id": "rag-1", "type": "AgentRagElement", "name": "docs", "llm_name": "fast"},
        "tool-1": {"id": "tool-1", "type": "AgentTool", "name": "lookup", "description": "d",
                   "code": "def lookup(x):\n    return x"},
        "skill-1": {"id": "skill-1", "type": "AgentSkill", "name": "tone", "content": "Be nice"},
        "ws-1": {"id": "ws-1", "type": "AgentWorkspace", "name": "notes", "path": "/tmp/notes"},
        "gui-1": {"id": "gui-1", "type": "AgentGUI", "name": "signup", "gui_id": "signup", "persist": False,
                  "width": "420px", "is_form": True, "guiModel": _SIGNUP_GUI},
    }


def _live_testing_nodes():
    return [
        _state("s-ask", "ask", [
            _action("a-text", "TextReplyAction", name=_TRICKY, useSessionVars=True),
            _action("a-gui", "GUIReplyAction", guiId="signup"),
            _action("a-md", "WebSocketReplyMarkdownAction", ws_message="**{name}**", useSessionVars=True),
            _action("a-html", "WebSocketReplyHTMLAction", ws_message="<b>{name}</b>", useSessionVars=True),
            _action("a-speech", "WebSocketReplySpeechAction", ws_message="Hi {name}",
                    ws_audio_speed=1.5, useSessionVars=True),
        ], fallback=[_action("a-fb", "GUIReplyAction", guiId="signup")], initial=True),
        _state("s-think", "think", [
            _action("a-llm", "LLMReplyAction", system_message=_TRICKY, llm_name="fast",
                    inputPromptMode="custom", customInputPrompt="Summarise {email}",
                    customInputPromptUseSessionVars=True, systemPromptUseSessionVars=True,
                    storeInSession="summary", sendReply=False),
            _action("a-chat", "LLMChatAction", system_message="Chat about {summary}", llm_name="fast",
                    systemPromptUseSessionVars=True, storeInSession="chat", sendReply=False),
            _action("a-rag", "RAGReplyAction", ragDatabaseName="docs", prompt="Cite {summary}",
                    inputPromptMode="custom", customInputPrompt="Find {email}",
                    customInputPromptUseSessionVars=True, promptUseSessionVars=True,
                    storeInSession="rag_answer", sendReply=False),
            _action("a-db", "DBAction", dbQueryMode="llm_query", llm_name="fast",
                    inputPromptMode="custom", customInputPrompt="Rows for {email}",
                    customInputPromptUseSessionVars=True, storeInSession="rows", sendReply=False),
            _action("a-crawl", "WebCrawlLLMAction", initial_url="https://example.org",
                    system_message_prefix="Site of {email}", systemMessagePrefixUseSessionVars=True,
                    llm_name="fast", storeInSession="crawl", sendReply=False),
        ]),
        _state("s-files", "files", [_action("a-done", "TextReplyAction", name="Files received")]),
    ]


def _live_testing_edges():
    return [
        _transition("r-form", "s-ask", "s-think", transitionType="predefined",
                    predefined={"predefinedType": "when_form_submitted", "formGuiId": "signup"}),
        _transition("r-event", "s-think", "s-files", transitionType="custom",
                    custom={"event": "GUIEvent", "condition": [], "guiEventGuiId": "signup"}),
        _transition("r-files", "s-files", "s-ask", transitionType="predefined",
                    predefined={"predefinedType": "when_file_received", "fileType": "pdf, csv, image/png"}),
        _transition("r-intent", "s-ask", "s-files", transitionType="predefined",
                    predefined={"predefinedType": "when_intent_matched", "intentName": "greet"}),
    ]


def _component_nodes(components):
    """Legacy v4 layout: components rendered as canvas nodes (pre-Components-page models).

    Intents carry their training sentences inline on ``data.training_phrases``.
    """
    nodes = []
    for component in components.values():
        if component["type"] == "AgentIntentBody":
            continue
        data = {k: v for k, v in component.items() if k not in ("id", "type", "owner")}
        if component["type"] == "AgentIntent":
            data["training_phrases"] = [
                {"id": body_id, "name": components[body_id]["name"]} for body_id in data.pop("bodies")
            ]
        nodes.append({"id": component["id"], "type": component["type"], "position": {"x": 0, "y": 0},
                      "width": 160, "height": 100, "data": data})
    return nodes


def _live_testing_diagram(layout: str = "components"):
    """The live-testing diagram in the current layout (``model.components``), the legacy v4
    layout (components as canvas ``nodes``) or the legacy top-level ``agentComponents``."""
    nodes = _live_testing_nodes()
    model = {"version": "4.0.0", "type": "AgentDiagram", "nodes": nodes, "edges": _live_testing_edges()}
    diagram = {"title": "LiveAgent", "model": model}
    if layout == "components":
        model["components"] = _live_testing_components()
    elif layout == "nodes":
        nodes.extend(_component_nodes(_live_testing_components()))
    else:
        diagram["agentComponents"] = _live_testing_components()
    return diagram


def _state_rows(diagram_model, state_name, key="bodies"):
    state = next(n for n in diagram_model["nodes"]
                 if n["type"] == "AgentState" and n["data"]["name"] == state_name)
    return state["data"][key]


def _to_code(agent, tmp_path, name="agent.py"):
    code_path = os.path.join(str(tmp_path), name)
    agent_model_to_code(agent, code_path)
    with open(code_path, encoding="utf-8") as handle:
        return handle.read()


def _strip_generated_ids(value):
    """Drop the frame ids the GUI serializer regenerates on every export."""
    if isinstance(value, dict):
        return {k: _strip_generated_ids(v) for k, v in value.items()
                if not (k == "id" and isinstance(v, str) and v.startswith("frame-"))}
    if isinstance(value, list):
        return [_strip_generated_ids(v) for v in value]
    return value


def _condition_signature(condition):
    if isinstance(condition, FileTypeMatcher):
        return "FileTypeMatcher", tuple(sorted(condition.allowed_types))
    if isinstance(condition, FormSubmitMatcher):
        return "FormSubmitMatcher", condition.form_id
    return type(condition).__name__, condition.name


def _agent_signature(agent):
    """Comparable view of everything the live-testing surface adds to an agent."""
    states = {}
    for state in agent.states:
        transitions = sorted((
            (
                type(t.event).__name__ if t.event else None,
                t.event.message_id if isinstance(t.event, GUIEvent) else None,
                tuple(_condition_signature(c) for c in t.conditions or []),
                t.dest.name,
            )
            for t in state.transitions
        ), key=repr)
        states[state.name] = {
            "body": [repr(a) for a in state.body.actions] if state.body else [],
            "fallback": [repr(a) for a in state.fallback_body.actions] if state.fallback_body else [],
            "transitions": transitions,
        }
    return {
        "states": states,
        "intents": {i.name: list(i.training_sentences) for i in agent.intents},
        "llms": sorted(llm.name for llm in agent.llms),
        "rags": sorted(r.name for r in agent.rags),
        "tools": sorted(t.name for t in agent.tools),
        "skills": sorted(s.name for s in agent.skills),
        "workspaces": sorted(w.name for w in agent.workspaces),
        "guis": {gui_id: _strip_generated_ids(_serialize_gui_model(gui))
                 for gui_id, gui in agent.gui_models.items()},
    }


def _elements_by_type(json_model, element_type):
    """Components of ``element_type``; asserts none leaked onto the canvas."""
    assert all(n["type"] != element_type for n in json_model["nodes"])
    return [c for c in json_model["components"].values() if c.get("type") == element_type]


def _state_actions(json_model, state_name, key="bodies"):
    return {row["actionType"]: row for row in _state_rows(json_model, state_name, key)}


@pytest.mark.parametrize("layout", ["components", "nodes", "agentComponents"])
def test_live_testing_diagram_loads_in_every_layout(layout):
    agent = process_agent_diagram(_live_testing_diagram(layout))
    assert {llm.name for llm in agent.llms} == {"fast"}
    assert {i.name for i in agent.intents} == {"greet"}
    assert {r.name for r in agent.rags} == {"docs"}
    assert {t.name for t in agent.tools} == {"lookup"}
    assert {s.name for s in agent.skills} == {"tone"}
    assert {w.name for w in agent.workspaces} == {"notes"}
    assert set(agent.gui_models) == {"signup"}
    assert _agent_signature(agent) == _agent_signature(process_agent_diagram(_live_testing_diagram()))


def test_live_testing_json_to_buml_values():
    agent = process_agent_diagram(_live_testing_diagram())
    ask = next(s for s in agent.states if s.name == "ask")
    text, gui_reply = ask.body.actions[0], ask.body.actions[1]
    # Stored verbatim: no escaping in the metamodel.
    assert text.message == _TRICKY
    assert text.use_session_vars is True
    assert isinstance(gui_reply, GUIReplyAction)
    assert (gui_reply.gui_id, gui_reply.persist, gui_reply.width, gui_reply.is_form) == (
        "signup", False, "420px", True
    )
    assert agent.intents[0].training_sentences == [_TRICKY]
    think = next(s for s in agent.states if s.name == "think")
    llm = think.body.actions[0]
    assert llm.prompt == _TRICKY
    assert (llm.input_prompt_mode, llm.custom_input_prompt) == ("custom", "Summarise {email}")
    assert llm.custom_input_prompt_use_session_vars and llm.system_prompt_use_session_vars
    assert (llm.store_in_session, llm.send_reply) == ("summary", False)


def test_live_testing_json_roundtrip_preserves_editor_fields(tmp_path):
    """JSON -> B-UML -> code -> JSON keeps every live-testing field of the editor JSON."""
    source = _live_testing_diagram()
    json_model = agent_buml_to_json(_to_code(process_agent_diagram(copy.deepcopy(source)), tmp_path))

    for state_name in ("ask", "think"):
        actions = _state_actions(json_model, state_name)
        for expected in _state_rows(source["model"], state_name):
            produced = actions[expected["actionType"]]
            ignored = {"id"} if expected["actionType"] == "TextReplyAction" else {"id", "name"}
            for key, value in expected.items():
                if key not in ignored:
                    assert produced[key] == value, (state_name, expected["actionType"], key)
    fallback = _state_actions(json_model, "ask", key="fallbackBodies")
    assert fallback["GUIReplyAction"]["guiId"] == "signup"

    guis = _elements_by_type(json_model, "AgentGUI")
    assert len(guis) == 1
    gui = guis[0]
    assert (gui["gui_id"], gui["persist"], gui["width"], gui["is_form"]) == ("signup", False, "420px", True)
    assert gui["guiModel"]["pages"][0]["name"] == "Signup"
    assert gui["id"] in json_model["components"]

    sentences = _elements_by_type(json_model, "AgentIntentBody")
    assert [s["name"] for s in sentences] == [_TRICKY]

    transitions = [e["data"] for e in json_model["edges"] if e["type"] == "AgentStateTransition"]
    predefined = {r["predefined"]["predefinedType"]: r["predefined"] for r in transitions
                  if r["transitionType"] == "predefined"}
    assert predefined["when_form_submitted"]["formGuiId"] == "signup"
    assert predefined["when_file_received"]["fileType"] == "pdf, csv, png"
    custom = [r["custom"] for r in transitions if r["transitionType"] == "custom"]
    # The initial state is a data flag, not a StateInitialNode marker.
    initial = [n["data"]["name"] for n in json_model["nodes"] if n["data"].get("initial")]
    assert initial == ["ask"]
    assert custom == [{"event": "GUIEvent", "condition": [], "guiEventGuiId": "signup"}]


def test_live_testing_buml_roundtrip_is_identity(tmp_path):
    """JSON -> B-UML -> code -> JSON -> B-UML yields the same metamodel objects, and
    once exported the code is a fixed point of export -> import -> export."""
    def reimport(code):
        json_model = agent_buml_to_json(code)
        return process_agent_diagram({"title": "LiveAgent", "model": json_model,
                                      "config": json_model.get("config", {})})

    first = process_agent_diagram(_live_testing_diagram())
    second = reimport(_to_code(first, tmp_path))
    assert _agent_signature(second) == _agent_signature(first)
    # The GUI converter normalises hand-written GrapesJS JSON (e.g. screen names) on
    # its first pass, so code identity is checked from the first export onwards.
    exported = _to_code(second, tmp_path, "agent_second.py")
    assert _to_code(reimport(exported), tmp_path, "agent_third.py") == exported


def test_live_testing_code_exec_roundtrip(tmp_path):
    """The emitted agent module executes and rebuilds the same agent, GUI models included."""
    first = process_agent_diagram(_live_testing_diagram())
    code = _to_code(first, tmp_path)
    namespace = {}
    exec(compile(code, "agent.py", "exec"), namespace)
    restored = namespace["agent"]
    assert restored.validate(raise_exception=False)["success"]
    assert _agent_signature(restored) == _agent_signature(first)


def test_gui_reply_without_agent_gui_is_rejected():
    diagram = _live_testing_diagram()
    del diagram["model"]["components"]["gui-1"]
    with pytest.raises(ValueError, match="GUIReplyAction references GUI 'signup'"):
        process_agent_diagram(diagram)


def test_undesigned_agent_gui_becomes_empty_gui_model():
    diagram = _live_testing_diagram()
    diagram["model"]["components"]["gui-1"]["guiModel"] = None
    agent = process_agent_diagram(diagram)
    assert set(agent.gui_models) == {"signup"}
    assert all(not module.screens for module in agent.gui_models["signup"].modules)


def test_custom_input_mode_without_prompt_is_rejected():
    diagram = _live_testing_diagram()
    next(row for row in _state_rows(diagram["model"], "think") if row["id"] == "a-llm")["customInputPrompt"] = ""
    with pytest.raises(ValueError, match="requires a non-empty custom_input_prompt"):
        process_agent_diagram(diagram)


def test_project_export_import_keeps_agent_gui_inside_agent_diagram(tmp_path):
    """Project export (domain model + agent with GUI) re-imports: the agent GUI stays an
    AgentGUI component and is not mistaken for a GUINoCodeDiagram section."""
    from besser.BUML.metamodel.project import Project
    from besser.BUML.metamodel.structural import Class, DomainModel, Metadata
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )

    agent = process_agent_diagram(_live_testing_diagram())
    domain_model = DomainModel(name="Library", types={Class(name="Book")}, associations=set())
    project = Project(name="LiveProject", models=[domain_model, agent], owner="tester",
                      metadata=Metadata(description="Domain + agent"))
    file_path = os.path.join(str(tmp_path), "project.py")
    project_to_code(project, file_path)
    with open(file_path, encoding="utf-8") as handle:
        project_json = project_to_json(handle.read())

    diagrams = project_json["diagrams"]
    assert diagrams["GUINoCodeDiagram"][0]["model"]["pages"] == []
    agent_json = diagrams["AgentDiagram"][0]["model"]
    guis = _elements_by_type(agent_json, "AgentGUI")
    assert [g["gui_id"] for g in guis] == ["signup"]
    restored = process_agent_diagram({"title": "LiveAgent", "model": agent_json})
    assert _agent_signature(restored) == _agent_signature(agent)


def test_rag_embedding_and_hybrid_settings_roundtrip(tmp_path):
    """RAG embedding (Ollama base_url/model) and hybrid retrieval settings are read from the
    AgentRagElement component (and from a legacy RAG node's data) and emitted back on the
    component by buml_to_json."""
    rag = {"id": "rag-1", "type": "AgentRagElement", "name": "local_docs", "llm_name": "",
           "embedding_provider": "ollama", "embedding_base_url": "http://ollama:11434",
           "embedding_model": "mxbai-embed-large", "use_hybrid_rag": True, "bm25_weight": 0.3,
           "k": 5, "num_previous_messages": 2, "llm_prompt": "Use the docs."}
    state = _state("s-1", "answer", [_action("a-rag", "RAGReplyAction", ragDatabaseName="local_docs")],
                   initial=True)
    as_component = {"title": "RagAgent", "model": {"nodes": [state], "edges": [], "components": {"rag-1": rag}}}
    as_legacy_node = {"title": "RagAgent", "model": {"nodes": [state] + _component_nodes({"rag-1": rag}),
                                                     "edges": []}}
    for diagram in (as_component, as_legacy_node):
        agent = process_agent_diagram(copy.deepcopy(diagram))
        (restored_rag,) = agent.rags
        assert restored_rag.vector_store.embedding_provider == "ollama"
        assert restored_rag.vector_store.embedding_parameters == {
            "base_url": "http://ollama:11434", "model": "mxbai-embed-large"}
        assert (restored_rag.use_hybrid_rag, restored_rag.bm25_weight) == (True, 0.3)
        assert (restored_rag.k, restored_rag.num_previous_messages) == (5, 2)

    json_model = agent_buml_to_json(_to_code(process_agent_diagram(copy.deepcopy(as_component)), tmp_path))
    (emitted,) = _elements_by_type(json_model, "AgentRagElement")
    assert "bounds" not in emitted and "position" not in emitted
    for key in ("embedding_provider", "embedding_base_url", "embedding_model", "use_hybrid_rag",
                "bm25_weight", "k", "num_previous_messages", "llm_prompt"):
        assert emitted[key] == rag[key], key
