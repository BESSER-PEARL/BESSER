"""BAFGenerator output for agent-testing features.

Covers session-variable interpolation, ``store_in_session`` / ``send_reply``,
custom input prompts, GUI replies (``guis/`` package), GUI and form-submit
transitions, the ``use_ui`` flag under ``test_mode`` and workspace folders.
Every test also checks that each generated ``.py`` file compiles.
"""

import ast
import os

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.state_machine.agent import (
    Agent, AgentReply, GUIEvent, GUIReplyAction, LLMReply, WebSocketPlatform, WebSocketReplyMarkdown,
)
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.generators.agents.baf_generator import (
    BAFGenerator, GenerationMode, collect_gui_modules, extract_braced_vars, workspace_rel_dir,
)


def _generate(agent: Agent, output_dir, test_mode: bool = False, config: dict = None) -> str:
    """Generate ``agent`` into ``output_dir``, compile every emitted .py file and return the agent source."""
    BAFGenerator(
        model=agent,
        output_dir=str(output_dir),
        config=config,
        generation_mode=GenerationMode.CODE_ONLY,
        test_mode=test_mode,
    ).generate()
    for root, _dirs, files in os.walk(str(output_dir)):
        for file_name in files:
            if file_name.endswith(".py"):
                path = os.path.join(root, file_name)
                with open(path, encoding="utf-8") as f:
                    compile(f.read(), path, "exec")
    with open(os.path.join(str(output_dir), f"{agent.name}.py"), encoding="utf-8") as f:
        return f.read()


def _function_source(code: str, name: str) -> str:
    """Return the source of the top-level function ``name`` in ``code``."""
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(code, node)
    raise AssertionError(f"function {name!r} not found in generated code")


def _agent_with_body(actions, fallback_actions=None) -> Agent:
    agent = Agent("feature_agent")
    agent.platforms.append(WebSocketPlatform())
    agent.new_llm(name="gpt-4o", provider="openai", parameters={})
    initial = agent.new_state(name="initial", initial=True)
    done = agent.new_state(name="done")
    initial.set_body(Body("initial_body", actions=actions))
    if fallback_actions is not None:
        initial.set_fallback_body(Body("initial_fallback", actions=fallback_actions))
    initial.go_to(done)
    return agent


def _gui_model(name: str = "order_gui") -> GUIModel:
    text = Text(name="title", content="Place your order")
    screen = Screen(name="order_screen", description="", view_elements={text}, is_main_page=True)
    module = Module(name="order_module", screens={screen})
    return GUIModel(name=name, package="", versionCode="1", versionName="1.0", description="",
                    modules={module})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_extract_braced_vars_is_unique_and_ordered():
    assert extract_braced_vars("{b} and {a} and {b} {not-a-var}") == ["b", "a"]
    assert extract_braced_vars(None) == []


def test_workspace_rel_dir():
    assert workspace_rel_dir("/tmp/sandbox", "Sandbox") == "tmp/sandbox"
    assert workspace_rel_dir("C:\\data\\ws", "ws") == "data/ws"
    assert workspace_rel_dir("   ", "My Space") == "my_space"
    with pytest.raises(ValueError, match="Workspace 'evil'"):
        workspace_rel_dir("../outside", "evil")


# ---------------------------------------------------------------------------
# Session variables, store_in_session, send_reply, custom input prompt
# ---------------------------------------------------------------------------

def test_session_var_interpolation_in_body_and_fallback(tmp_path):
    message = "Hi {name}, you said {user_message}"
    agent = _agent_with_body(
        [AgentReply(message, use_session_vars=True)],
        fallback_actions=[WebSocketReplyMarkdown("**{name}**", use_session_vars=True)],
    )
    code = _generate(agent, tmp_path)

    body = _function_source(code, "initial_body")
    assert "reply_text = 'Hi {name}, you said {user_message}'" in body
    assert "reply_text = reply_text.replace('{name}', str(session.get('name') or ''))" in body
    assert "reply_text = reply_text.replace('{user_message}', str(session.event.message or ''))" in body
    assert "session.get('user_message')" not in body
    assert "session.reply(reply_text)" in body

    fallback = _function_source(code, "initial_fallback")
    assert "_ws_msg = _ws_msg.replace('{name}', str(session.get('name') or ''))" in fallback
    assert "platform.reply_markdown(session, _ws_msg)" in fallback


def test_message_without_session_vars_is_a_literal(tmp_path):
    code = _generate(_agent_with_body([AgentReply("Hi {name}")]), tmp_path)
    body = _function_source(code, "initial_body")
    assert "reply_text = 'Hi {name}'" in body
    assert ".replace(" not in body


def test_store_in_session_and_send_reply_false(tmp_path):
    agent = _agent_with_body([
        LLMReply(prompt="Summarize", store_in_session="summary", send_reply=False),
        LLMReply(prompt="Answer", store_in_session="answer"),
    ])
    body = _function_source(_generate(agent, tmp_path), "initial_body")
    silent, sent = body.split("# Action 2")
    assert 'session.set("summary", message)' in silent
    assert "session.reply(message)" not in silent
    assert 'session.set("answer", message)' in sent
    assert "session.reply(message)" in sent


def test_custom_input_prompt_with_session_vars(tmp_path):
    agent = _agent_with_body([
        LLMReply(prompt="Tone: {tone}", llm_name="gpt-4o", input_prompt_mode="custom",
                 custom_input_prompt="Question about {topic}: {user_message}",
                 custom_input_prompt_use_session_vars=True, system_prompt_use_session_vars=True),
    ])
    body = _function_source(_generate(agent, tmp_path), "initial_body")
    assert "_llm_input = 'Question about {topic}: {user_message}'" in body
    assert "_llm_input = _llm_input.replace('{topic}', str(session.get('topic') or ''))" in body
    assert "_llm_sys = _llm_sys.replace('{tone}', str(session.get('tone') or ''))" in body
    assert "message = gpt_4o.predict(message=_llm_input, session=session, system_message=_llm_sys)" in body


def test_custom_input_prompt_without_session_vars_is_not_interpolated(tmp_path):
    agent = _agent_with_body([
        LLMReply(input_prompt_mode="custom", custom_input_prompt="Literal {topic}"),
    ])
    body = _function_source(_generate(agent, tmp_path), "initial_body")
    assert "_llm_input = 'Literal {topic}'" in body
    assert "_llm_input.replace(" not in body
    assert "message = default_llm.predict(message=_llm_input, session=session)" in body


# ---------------------------------------------------------------------------
# GUI replies
# ---------------------------------------------------------------------------

def test_gui_reply_generates_guis_package(tmp_path):
    agent = _agent_with_body(
        [GUIReplyAction(gui_id="order-form", width="80%", is_form=True)],
        fallback_actions=[GUIReplyAction(gui_id="order-form")],
    )
    agent.add_gui_model("order-form", _gui_model())
    code = _generate(agent, tmp_path)

    assert code.count("from guis.order_form import gui as order_form") == 1
    assert "platform.reply_gui(session, order_form)" in _function_source(code, "initial_body")
    assert "platform.reply_gui(session, order_form)" in _function_source(code, "initial_fallback")

    guis_dir = tmp_path / "guis"
    assert sorted(os.listdir(guis_dir)) == ["__init__.py", "order_form.py"]
    gui_code = (guis_dir / "order_form.py").read_text(encoding="utf-8")
    assert "gui_model = GUIModel(" in gui_code
    assert 'name="order_gui"' in gui_code
    assert "from baf.core.gui.agent_gui import AgentGUI" in gui_code
    assert 'gui_id="order-form",' in gui_code
    assert "persist=True," in gui_code
    assert 'width="80%",' in gui_code


def test_gui_reply_without_registered_gui_model_raises(tmp_path):
    agent = _agent_with_body([GUIReplyAction(gui_id="missing")])
    with pytest.raises(ValueError, match="'missing'"):
        _generate(agent, tmp_path)


def test_gui_ids_with_same_module_name_raise():
    agent = _agent_with_body([GUIReplyAction(gui_id="order-form"), GUIReplyAction(gui_id="order form")])
    with pytest.raises(ValueError, match="guis/order_form.py"):
        collect_gui_modules(agent)


def test_no_gui_reply_generates_no_guis_package(tmp_path):
    code = _generate(_agent_with_body([AgentReply("hello")]), tmp_path)
    assert "GUI IMPORTS" not in code
    assert not (tmp_path / "guis").exists()


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def test_gui_event_and_form_submit_transitions(tmp_path):
    agent = Agent("gui_flow")
    agent.platforms.append(WebSocketPlatform())
    ask = agent.new_state(name="ask", initial=True)
    clicked = agent.new_state(name="clicked")
    submitted = agent.new_state(name="submitted")
    ask.set_body(Body("ask_body", actions=[GUIReplyAction(gui_id="order-form", is_form=True)]))
    agent.add_gui_model("order-form", _gui_model())
    ask.when_event(GUIEvent(message_id="order-form")).go_to(clicked)
    ask.when_form_submitted(form_id="order-form").go_to(submitted)
    clicked.when_form_submitted().go_to(submitted)
    code = _generate(agent, tmp_path)
    assert 'ask.when_event(GUIEvent(message_id="order-form")).go_to(clicked)' in code
    assert 'ask.when_form_submitted(form_id="order-form").go_to(submitted)' in code
    assert "clicked.when_form_submitted().go_to(submitted)" in code


def test_file_types_are_emitted_as_python_literals(tmp_path):
    agent = _agent_with_body([AgentReply("send a file")])
    initial = agent.states[0]
    initial.when_file_received(["pdf", "it's"]).go_to(agent.states[1])
    code = _generate(agent, tmp_path)
    call = next(line for line in code.splitlines() if line.startswith("initial.when_file_received("))
    types_literal = call[len("initial.when_file_received("):call.index(").go_to(done)")]
    assert ast.literal_eval(types_literal) == ["pdf", "it's"]


# ---------------------------------------------------------------------------
# use_ui and test_mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_mode, expected", [(False, "True"), (True, "False")])
def test_websocket_platform_use_ui_follows_test_mode(tmp_path, test_mode, expected):
    code = _generate(_agent_with_body([AgentReply("hi")]), tmp_path, test_mode=test_mode)
    platform_lines = [line for line in code.splitlines() if "use_websocket_platform(" in line]
    assert platform_lines
    assert all(line == f"platform = agent.use_websocket_platform(use_ui={expected})" for line in platform_lines)


@pytest.mark.parametrize("test_mode, expected", [(False, "True"), (True, "False")])
def test_streamlit_platform_use_ui_follows_test_mode(tmp_path, test_mode, expected):
    agent = Agent("streamlit_agent")
    agent.new_state(name="initial", initial=True).set_body(Body("initial_body", actions=[AgentReply("hi")]))
    code = _generate(agent, tmp_path, test_mode=test_mode, config={"agentPlatform": "streamlit"})
    assert f"platform = agent.use_websocket_platform(use_ui={expected})" in code
    assert f"use_ui={'False' if expected == 'True' else 'True'}" not in code


def test_test_mode_creates_workspace_dirs_and_rejects_escaping_paths(tmp_path):
    agent = _agent_with_body([AgentReply("hi")])
    agent.new_workspace(name="docs", path="/srv/docs")
    agent.new_workspace(name="ScratchSpace", path="")
    _generate(agent, tmp_path / "ok", test_mode=True)
    assert (tmp_path / "ok" / "srv" / "docs").is_dir()
    assert (tmp_path / "ok" / "scratchspace").is_dir()

    _generate(agent, tmp_path / "no_test_mode")
    assert not (tmp_path / "no_test_mode" / "srv").exists()

    agent.new_workspace(name="evil", path="../../outside")
    with pytest.raises(ValueError, match="Workspace 'evil'"):
        _generate(agent, tmp_path / "bad", test_mode=True)
