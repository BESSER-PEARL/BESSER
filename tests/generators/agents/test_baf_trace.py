"""BAFGenerator records which lines of the agent script, tools file and other files come from which agent
element, and which files an LLM personalization step produced."""

import copy
import os
from unittest.mock import patch

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.state_machine.agent import (
    Agent, AgentReply, GUIReplyAction, LLMReply, Skill, Tool, WebSocketPlatform,
)
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.generators.agents import agent_personalization
from besser.generators.agents.agent_personalization import configure_agent, personalization_uses_llm
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode
from besser.utilities.generation_trace import trace_source


def _agent(with_ids: bool) -> Agent:
    def ident(element, value):
        if with_ids:
            element.id = value
        return element

    agent = ident(Agent("trace_agent"), "agent-1")
    agent.platforms.append(WebSocketPlatform())
    agent.new_llm(name="gpt-4o", provider="openai", parameters={})
    agent.new_entity("Color", entries={"red": ["scarlet"]}, description="colors")
    ident(agent.entities[0], "entity-1")
    hello = ident(agent.new_intent("hello", ["hi", "hello there"], description="greeting"), "intent-hello")
    bye = ident(agent.new_intent("bye", ["bye"]), "intent-bye")
    initial = ident(agent.new_state("initial", initial=True), "state-initial")
    chat = ident(agent.new_state("chat"), "state-chat")
    done = ident(agent.new_state("done"), "state-done")
    initial.set_body(ident(Body("initial_body", actions=[AgentReply("Welcome!")]), "body-initial"))
    chat.set_body(ident(Body("chat_body", actions=[LLMReply(), AgentReply("It's a chat")]), "body-chat"))
    chat.set_fallback_body(ident(Body("chat_fallback", actions=[AgentReply("Sorry")]), "fallback-chat"))
    done.set_body(Body("done_body", actions=[GUIReplyAction(gui_id="g1")]))  # a body without id
    initial.when_intent_matched(hello).go_to(chat)
    chat.when_intent_matched(bye).go_to(done)
    chat.when_no_intent_matched().go_to(chat)
    ident(initial.transitions[0], "transition-hello")
    ident(chat.transitions[0], "transition-bye")
    ident(chat.transitions[1], "transition-fallback")
    agent.add_tool(ident(Tool("add", "adds", "def add(a, b):\n    return a + b\n"), "tool-add"))
    agent.add_tool(ident(Tool("sub", "subs", "def sub(a, b):\n    return a - b"), "tool-sub"))
    agent.add_skill(ident(Skill("playbook", "# Playbook\nDo things", "desc"), "skill-1"))
    gui = GUIModel(name="g", package="", versionCode="1", versionName="1.0", description="", modules={
        Module(name="m", screens={Screen(name="s", description="", view_elements={Text(name="t", content="Hi")},
                                         is_main_page=True)})})
    agent.add_gui_model("g1", ident(gui, "gui-1"))
    return agent


def _generate(agent, out, **options):
    generator = BAFGenerator(agent, output_dir=str(out), generation_mode=GenerationMode.CODE_ONLY, **options)
    generator.generate()
    files = {}
    for root, _, names in os.walk(str(out)):
        for name in names:
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", newline="") as f:
                files[os.path.relpath(path, str(out)).replace(os.sep, "/")] = f.read()
    return generator, files


def _regions(generator, path):
    return {(r.element_id, r.rule.split("#")[1]): r for r in generator.traced_files[path].regions}


def _text(files, path, region):
    return files[path].split("\n")[region.start_line - 1:region.end_line]


def test_each_written_file_is_traced_under_the_path_it_was_written_to(tmp_path):
    generator, files = _generate(_agent(True), tmp_path)
    assert set(generator.traced_files) == {"trace_agent.py", "config.yaml", "readme.txt", "tools.py",
                                           "skills/playbook.md", "guis/g1.py"}
    assert set(generator.traced_files) <= set(files)
    sources = {path: {(s.element.element_id, s.rule) for s in trace.sources}
               for path, trace in generator.traced_files.items()}
    assert sources["trace_agent.py"] == {("agent-1", "baf_agent_template.py.j2#agent")}
    assert sources["config.yaml"] == {("agent-1", "baf_config_template.py.j2#config")}
    assert sources["readme.txt"] == {("agent-1", "readme.txt.j2#readme")}
    assert sources["skills/playbook.md"] == {("skill-1", "skill.md#skill")}
    assert sources["guis/g1.py"] == {("gui-1", "agent_gui.py.j2#gui")}
    assert not generator.ai_files


def test_the_user_supplied_config_yaml_is_traced_to_the_agent(tmp_path):
    generator, _ = _generate(_agent(True), tmp_path, config_yaml="a: 1\n")
    assert [s.element.element_id for s in generator.traced_files["config.yaml"].sources] == ["agent-1"]


def test_agent_script_regions_contain_the_code_of_their_element(tmp_path):
    generator, files = _generate(_agent(True), tmp_path)
    path = "trace_agent.py"
    regions = _regions(generator, path)

    assert _text(files, path, regions[("intent-hello", "intent")])[0].startswith("hello = agent.new_intent('hello'")
    assert "'hello there'," in "\n".join(_text(files, path, regions[("intent-hello", "intent")]))
    assert _text(files, path, regions[("intent-bye", "intent")])[0].startswith("bye = agent.new_intent('bye'")
    assert _text(files, path, regions[("entity-1", "entity")])[0].startswith("Color = agent.new_entity('Color'")
    assert _text(files, path, regions[("state-chat", "state")]) == ["chat = agent.new_state('chat')"]
    assert _text(files, path, regions[("state-initial", "state")])[0].startswith("initial = agent.new_state('initial'")

    behavior = regions[("state-chat", "state_behavior")]
    assert _text(files, path, behavior)[0] == "# chat"
    body = regions[("body-chat", "body")]
    assert _text(files, path, body)[0] == "def chat_body(session: Session):"
    assert _text(files, path, body)[-1] == "chat.set_body(chat_body)"
    fallback = regions[("fallback-chat", "fallback_body")]
    assert _text(files, path, fallback)[0] == "def chat_fallback(session: Session):"
    assert _text(files, path, fallback)[-1] == "chat.set_fallback_body(chat_fallback)"
    bye = regions[("transition-bye", "transition")]
    assert _text(files, path, bye) == ["chat.when_intent_matched(bye).go_to(done)"]
    assert _text(files, path, regions[("transition-fallback", "transition")]) == [
        "chat.when_no_intent_matched().go_to(chat)"]
    # Bodies and transitions lie inside the behaviour of their state, which is not the state's creation line.
    for inner in (body, fallback, bye, regions[("transition-fallback", "transition")]):
        assert behavior.start_line < inner.start_line <= inner.end_line <= behavior.end_line
    assert regions[("state-chat", "state")].end_line < behavior.start_line
    assert _text(files, path, regions[("transition-hello", "transition")]) == [
        "initial.when_intent_matched(hello).go_to(chat)"]


def test_the_tools_file_has_one_region_per_tool_and_the_bytes_of_the_untraced_writer(tmp_path):
    agent = _agent(True)
    generator, files = _generate(agent, tmp_path)
    regions = _regions(generator, "tools.py")
    assert _text(files, "tools.py", regions[("tool-add", "tool")]) == ["def add(a, b):", "    return a + b"]
    assert _text(files, "tools.py", regions[("tool-sub", "tool")]) == ["def sub(a, b):", "    return a - b"]
    # What the generator wrote before it was traced: the header, then each tool's code and a blank line.
    expected = "# Auto-generated tool definitions\n\n"
    for tool in agent.tools:
        expected += tool.code + ("" if tool.code.endswith("\n") else "\n") + "\n"
    assert files["tools.py"] == expected


def test_elements_without_ids_render_untraced_and_the_output_is_unchanged(tmp_path):
    traced, traced_files = _generate(_agent(True), tmp_path / "traced")
    untraced, untraced_files = _generate(_agent(False), tmp_path / "untraced")
    assert traced_files == untraced_files
    assert not any("\x02" in text or "\x03" in text for text in traced_files.values())
    assert all(trace.regions == () and trace.sources == () for trace in untraced.traced_files.values())
    assert set(untraced.traced_files) == set(traced.traced_files)
    # The body of the state without an id is among the output but has no region.
    regions = _regions(traced, "trace_agent.py")
    assert not any(rule == "body" and element_id.startswith("body-done") for element_id, rule in regions)
    assert "def done_body(session: Session):" in traced_files["trace_agent.py"]


def test_the_trace_of_a_personalized_agent_covers_the_profile_variants(tmp_path):
    base = _agent(True)
    mapping = [{"name": "kid", "agent_model": copy.deepcopy(base), "configuration": {}}]
    generator, files = _generate(base, tmp_path, config={"personalizationMapping": mapping})
    regions = [r for r in generator.traced_files["trace_agent.py"].regions if r.element_id == "state-chat"]
    texts = {files["trace_agent.py"].split("\n")[r.start_line - 1] for r in regions}
    assert "chat = agent.new_state('chat')" in texts
    assert "chat_kid = agent.new_state('chat_kid')" in texts
    assert not generator.ai_files  # no text was rewritten by an LLM


def _numbered(*_args, **_kwargs):
    return "\n".join(f"{n}. rewritten {n}" for n in range(1, 40))


def test_files_an_llm_step_wrote_are_listed_as_ai_files(tmp_path):
    config = {"agentStyle": "formal", "openaiApiKey": "k"}
    with patch.object(agent_personalization, "call_openai_chat", side_effect=_numbered) as chat:
        generator = BAFGenerator(_agent(True), output_dir=str(tmp_path), config=config,
                                 openai_api_key="k", generation_mode=GenerationMode.FULL)
        generator.generate()
    assert chat.call_count > 0
    assert generator.ai_files == {"trace_agent.py", "personalized_agent_model.py", "personalized_agent_model.json"}
    for path in generator.ai_files:
        assert (tmp_path / path).is_file()
    assert "trace_agent.py" in generator.traced_files  # its regions are still recorded


def test_a_personalization_that_changes_no_text_lists_no_ai_files(tmp_path):
    with patch.object(agent_personalization, "call_openai_chat", side_effect=_numbered) as chat:
        generator = BAFGenerator(_agent(True), output_dir=str(tmp_path), config={"avatar": "robot"},
                                 generation_mode=GenerationMode.FULL)
        generator.generate()
    assert chat.call_count == 0
    assert (tmp_path / "personalized_agent_model.py").is_file()
    assert generator.ai_files == set()


@pytest.mark.parametrize("config", [
    {}, {"avatar": "robot"}, {"agentStyle": "original"}, {"agentStyle": "formal"}, {"agentLanguage": "none"},
    {"agentLanguage": "original"}, {"agentLanguage": "french"}, {"languageComplexity": "original"},
    {"languageComplexity": "simple"}, {"sentenceLength": "original"}, {"sentenceLength": "concise"},
    {"presentation": {"agentStyle": "formal"}}, {"presentation": {"agentStyle": "original"}},
])
def test_personalization_uses_llm_matches_the_calls_configure_agent_makes(config):
    agent = _agent(False)
    with patch.object(agent_personalization, "call_openai_chat", side_effect=_numbered) as chat:
        configure_agent(agent, dict(config), openai_api_key="k")
    assert personalization_uses_llm(config) == (chat.call_count > 0)


def test_a_user_profile_model_makes_the_personalization_use_an_llm():
    assert personalization_uses_llm({"userProfileModel": {"name": "p"}})
    assert not personalization_uses_llm({"userProfileModel": "p"})
    assert not personalization_uses_llm(None)


def test_trace_source_of_an_agent_without_an_id_is_none():
    assert trace_source(_agent(False), "x") is None
