"""The label shown with option buttons: editor JSON -> B-UML -> code -> JSON, the generated
agent and personalization."""
import copy
import os
from unittest.mock import patch

from besser.BUML.metamodel.state_machine.agent import WebSocketReplyOptions
from besser.generators.agents.agent_personalization import configure_agent
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    agent_buml_to_json,
    process_agent_diagram,
)


def _diagram(label):
    elements = {
        "init": {"id": "init", "type": "StateInitialNode", "name": ""},
        "s-ask": {"id": "s-ask", "type": "AgentState", "name": "ask", "actions": ["a-opt"]},
        "s-done": {"id": "s-done", "type": "AgentState", "name": "done", "actions": []},
        "a-opt": {"id": "a-opt", "type": "AgentStateBody", "name": "Reply Options",
                  "actionType": "WebSocketReplyOptionsAction", "ws_options": "Oui\nNon",
                  **({"ws_options_label": label} if label is not None else {})},
    }
    relationships = {
        "r-init": {"id": "r-init", "type": "AgentStateTransitionInit", "source": {"element": "init"},
                   "target": {"element": "s-ask"}},
        "r-next": {"id": "r-next", "type": "AgentStateTransition", "source": {"element": "s-ask"},
                   "target": {"element": "s-done"}, "transitionType": "predefined",
                   "predefined": {"predefinedType": "auto"}},
    }
    return {"title": "Quiz", "model": {"elements": elements, "relationships": relationships}}


def _options_action(agent):
    return next(a for a in agent.states[0].body.actions if isinstance(a, WebSocketReplyOptions))


def test_label_is_read_from_the_editor_json():
    assert _options_action(process_agent_diagram(_diagram("Choisissez"))).label == "Choisissez"
    assert _options_action(process_agent_diagram(_diagram("  "))).label is None
    assert _options_action(process_agent_diagram(_diagram(None))).label is None


def test_label_survives_code_and_json_roundtrip(tmp_path):
    agent = process_agent_diagram(_diagram("Choisissez"))
    code_path = os.path.join(str(tmp_path), "agent.py")
    agent_model_to_code(agent, code_path)
    with open(code_path, encoding="utf-8") as handle:
        json_model = agent_buml_to_json(handle.read())

    options = next(el for el in json_model["elements"].values() if el.get("actionType") == "WebSocketReplyOptionsAction")
    assert options["ws_options"] == "Oui\nNon"
    assert options["ws_options_label"] == "Choisissez"


def _generated(agent, tmp_path):
    BAFGenerator(agent, output_dir=str(tmp_path), generation_mode=GenerationMode.CODE_ONLY).generate()
    return (tmp_path / f"{agent.name}.py").read_text(encoding="utf-8")


def test_generated_agent_passes_the_label_to_reply_options(tmp_path):
    code = _generated(process_agent_diagram(_diagram("Choisissez l'option")), tmp_path)

    assert "platform.reply_options(session, ['Oui', 'Non'], label='Choisissez l\\'option')" in code


def test_generated_agent_without_a_label_keeps_the_old_call(tmp_path):
    code = _generated(process_agent_diagram(_diagram(None)), tmp_path)

    assert "platform.reply_options(session, ['Oui', 'Non'])" in code


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_label_is_personalized_but_the_buttons_are_not_restyled(mock_chat):
    mock_chat.side_effect = lambda system_prompt, user_prompt, **kw: "\n".join(
        f"{i}. {line.split('. ', 1)[1].upper()}"
        for i, line in enumerate((l for l in user_prompt.splitlines() if l[:1].isdigit()), 1)
    )
    agent = process_agent_diagram(copy.deepcopy(_diagram("Choose")))

    configure_agent(agent, {"agentStyle": "formal", "openaiApiKey": "k"})

    options = _options_action(agent)
    assert options.label == "CHOOSE"
    assert options.options == ["Oui", "Non"]
