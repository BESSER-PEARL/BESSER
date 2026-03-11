import os
import pytest
from besser.BUML.metamodel.state_machine.agent import (
    Agent, Intent, WebSocketPlatform
)
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode


@pytest.fixture
def agent_model():
    """Create a minimal agent model with two states and a transition."""
    agent = Agent("test_agent")

    # Add a platform
    ws_platform = WebSocketPlatform()
    agent.platforms.append(ws_platform)

    # Define intents
    greet_intent = Intent(
        name="greet",
        training_sentences=["hello", "hi", "hey"],
    )
    agent.intents.append(greet_intent)

    # Create states using agent.new_state() so they are registered
    initial_state = agent.new_state(name="initial", initial=True)
    greeting_state = agent.new_state(name="greeting")

    # Add transition: initial -> greeting on greet intent
    initial_state.when_intent_matched(greet_intent).go_to(greeting_state)

    return agent


def test_baf_generator_instantiation(agent_model):
    """Test that the BAFGenerator can be instantiated."""
    generator = BAFGenerator(model=agent_model)
    assert generator is not None
    assert generator.generation_mode == GenerationMode.FULL


def test_baf_generator_code_only_mode(agent_model):
    """Test that CODE_ONLY generation mode is accepted."""
    generator = BAFGenerator(
        model=agent_model,
        generation_mode=GenerationMode.CODE_ONLY,
    )
    assert generator.generation_mode == GenerationMode.CODE_ONLY


def test_baf_generator_string_mode(agent_model):
    """Test that string generation mode is normalized correctly."""
    generator = BAFGenerator(
        model=agent_model,
        generation_mode="code_only",
    )
    assert generator.generation_mode == GenerationMode.CODE_ONLY


def test_baf_generator_generate(agent_model, tmpdir):
    """Test that generate() runs without errors and produces output."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    # The generator should create a Python file named after the agent
    agent_file = os.path.join(str(output_dir), f"{agent_model.name}.py")
    assert os.path.isfile(agent_file), f"{agent_model.name}.py should be generated"
    assert os.path.getsize(agent_file) > 0, "Generated agent file should not be empty"


def test_baf_generator_creates_config(agent_model, tmpdir):
    """Test that generate() creates a config.ini file."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    config_file = os.path.join(str(output_dir), "config.ini")
    assert os.path.isfile(config_file), "config.ini should be generated"


def test_baf_generator_creates_readme(agent_model, tmpdir):
    """Test that generate() creates a readme.txt file."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    readme_file = os.path.join(str(output_dir), "readme.txt")
    assert os.path.isfile(readme_file), "readme.txt should be generated"


def test_baf_generator_output_content(agent_model, tmpdir):
    """Test that generated agent code contains expected content."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    agent_file = os.path.join(str(output_dir), f"{agent_model.name}.py")
    with open(agent_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "test_agent" in code, "Generated code should reference the agent name"
