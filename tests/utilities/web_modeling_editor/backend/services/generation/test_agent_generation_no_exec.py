"""Agent generation and transform never run the user's code in the backend process.

The agent model carries user-written Python (a state's code action, a tool's code). It is
emitted verbatim into ``agent_model.py`` and must only ever be shipped, never imported here.
Each payload below writes a marker file at definition time, which is when an import of the
serialized model would run it.
"""

import asyncio
import io
import zipfile

from besser.BUML.metamodel.state_machine.agent import Agent, WebSocketPlatform
from besser.BUML.metamodel.state_machine.state_machine import Body, CustomCodeAction
from besser.utilities.web_modeling_editor.backend.models import DiagramInput
from besser.utilities.web_modeling_editor.backend.routers import conversion_router
from besser.utilities.web_modeling_editor.backend.services.generation.agents import generate_agent_files
from besser.utilities.web_modeling_editor.backend.services.utils import agent_generation_utils


def _agent_with_user_code(action_marker, tool_marker) -> Agent:
    agent = Agent("marker_agent")
    agent.platforms.append(WebSocketPlatform())
    greet = agent.new_state(name="greet", initial=True)
    # A default argument is evaluated when the ``def`` runs, not when the action is called.
    greet.set_body(Body("greet_body", actions=[CustomCodeAction(source=(
        f"def greet_action(session, _m=open({str(action_marker)!r}, 'w').write('ran')):\n"
        "    session.reply('hi')\n"
    ))]))
    # Tool code is emitted at module level, so any statement in it runs on import.
    agent.new_tool(
        name="marker_tool",
        description="Writes a marker when its module is imported.",
        code=f"open({str(tool_marker)!r}, 'w').write('ran')\ndef marker_tool():\n    return 'ok'\n",
    )
    return agent


def test_generate_agent_files_does_not_run_user_code(tmp_path):
    action_marker, tool_marker = tmp_path / "action_ran", tmp_path / "tool_ran"
    agent = _agent_with_user_code(action_marker, tool_marker)

    zip_buffer, _ = generate_agent_files(agent, None)

    assert not action_marker.exists()
    assert not tool_marker.exists()
    # The user's code is still delivered, as source, in the shipped model.
    with zipfile.ZipFile(io.BytesIO(zip_buffer.getvalue())) as zf:
        shipped = zf.read("agent_model.py").decode()
    assert "def greet_action(" in shipped
    assert "def marker_tool(" in shipped


def test_multi_language_generation_does_not_run_user_code(tmp_path, monkeypatch):
    action_marker, tool_marker = tmp_path / "action_ran", tmp_path / "tool_ran"
    monkeypatch.setattr(
        agent_generation_utils, "process_agent_diagram",
        lambda _json: _agent_with_user_code(action_marker, tool_marker),
    )
    # A target equal to the source needs no translation, hence no LLM call.
    languages = {"source": "english", "target": ["english"]}

    asyncio.run(agent_generation_utils.handle_multi_language_generation(
        {"model": {}}, {"languages": languages}, languages, generate_agent_files_fn=None,
    ))

    assert not action_marker.exists()
    assert not tool_marker.exists()


def test_transform_agent_model_json_does_not_run_user_code(tmp_path, monkeypatch):
    action_marker, tool_marker = tmp_path / "action_ran", tmp_path / "tool_ran"
    monkeypatch.setattr(
        conversion_router, "process_agent_diagram",
        lambda _json: _agent_with_user_code(action_marker, tool_marker),
    )

    result = asyncio.run(conversion_router.transform_agent_model_json(
        DiagramInput(title="marker", model={}, config={"style": "formal"}),
    ))

    assert result["diagramType"] == "AgentDiagram"
    assert not action_marker.exists()
    assert not tool_marker.exists()
