"""Render blueprints install PyTorch whenever the generated agent needs it.

An agent with no intent classifier set and no LLM gets BAF's classical (PyTorch)
classifier from the BAF generator, without ``intentRecognitionTechnology`` being
``classical`` in its config.
"""
import json
import pathlib

from besser.BUML.metamodel.state_machine.agent import LLMOpenAI
from besser.generators.agents.baf_generator import BAFGenerator
from besser.generators.web_app.web_app_generator import agent_slug
from besser.utilities.web_modeling_editor.backend.routers.github_deploy_api import (
    _add_chatbot_deployment_configs,
    _add_deployment_configs,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)

FIXTURE = (pathlib.Path(__file__).parents[3] / "generators" / "fixtures"
           / "library_template_agent_diagram.json")
TORCH_INSTALL = "pip install torch==2.6.0+cpu"


def _library_agent(with_llm=False):
    agent = process_agent_diagram(json.loads(FIXTURE.read_text(encoding="utf-8")))
    if with_llm:
        LLMOpenAI(agent=agent, name="gpt-4o-mini", parameters={})
    return agent


def _chatbot_blueprint(tmp_path, agent):
    BAFGenerator(agent, output_dir=str(tmp_path), config={}).generate()
    _add_chatbot_deployment_configs(str(tmp_path), "library_agent", agent_model=agent,
                                    agent_config={}, service_suffix="a1b2c3")
    return (tmp_path / "render.yaml").read_text(encoding="utf-8")


def _web_app_blueprint(tmp_path, agent):
    BAFGenerator(agent, output_dir=str(tmp_path / "agents" / agent_slug(agent.name)), config={}).generate()
    (tmp_path / "frontend").mkdir()  # the blueprint also writes the frontend's .env.production
    _add_deployment_configs(str(tmp_path), "library_app", agent_models=[agent],
                            agent_configs={agent.name: {}}, service_suffix="a1b2c3")
    return (tmp_path / "render.yaml").read_text(encoding="utf-8")


def test_chatbot_without_llm_installs_torch(tmp_path):
    assert TORCH_INSTALL in _chatbot_blueprint(tmp_path, _library_agent())


def test_chatbot_with_llm_does_not_install_torch(tmp_path):
    assert TORCH_INSTALL not in _chatbot_blueprint(tmp_path, _library_agent(with_llm=True))


def test_web_app_agent_without_llm_installs_torch(tmp_path):
    assert TORCH_INSTALL in _web_app_blueprint(tmp_path, _library_agent())


def test_web_app_agent_with_llm_does_not_install_torch(tmp_path):
    assert TORCH_INSTALL not in _web_app_blueprint(tmp_path, _library_agent(with_llm=True))
