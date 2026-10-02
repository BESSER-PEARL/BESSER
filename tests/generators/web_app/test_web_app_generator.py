import os
import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, StringType, IntegerType,
    BinaryAssociation, Multiplicity
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.generators.web_app import WebAppGenerator


@pytest.fixture
def domain_model():
    """Create a minimal domain model for testing."""
    name_prop = Property(name="name", type=StringType)
    email_prop = Property(name="email", type=StringType)
    user = Class(name="User", attributes={name_prop, email_prop})

    title_prop = Property(name="title", type=StringType)
    item = Class(name="Item", attributes={title_prop})

    user_end = Property(name="user_end", type=user, multiplicity=Multiplicity(1, 1))
    item_end = Property(name="item_end", type=item, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="UserItem", ends={user_end, item_end})

    model = DomainModel(
        name="TestModel",
        types={user, item},
        associations={assoc},
    )
    return model


@pytest.fixture
def gui_model():
    """Create a minimal GUI model for testing."""
    text1 = Text(name="title_text", content="Welcome")
    screen1 = Screen(
        name="Dashboard",
        description="Main dashboard",
        view_elements={text1},
        is_main_page=True,
    )
    module1 = Module(name="AppModule", screens={screen1})
    gui = GUIModel(
        name="TestApp",
        package="com.test.app",
        versionCode="1",
        versionName="1.0",
        modules={module1},
        description="Test web application",
    )
    return gui


def test_web_app_generator_instantiation(domain_model, gui_model):
    """Test that the WebAppGenerator can be instantiated."""
    generator = WebAppGenerator(model=domain_model, gui_model=gui_model)
    assert generator is not None
    assert generator.gui_model is gui_model
    assert generator.agent_model is None


def test_web_app_generator_generate(domain_model, gui_model, tmpdir):
    """Test that generate() runs without errors and produces output."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    # Verify the output directory has content
    generated_files = []
    for root, dirs, files in os.walk(str(output_dir)):
        for f in files:
            generated_files.append(os.path.join(root, f))

    assert len(generated_files) > 0, "WebAppGenerator should produce output files"


def test_web_app_generator_creates_frontend_and_backend(domain_model, gui_model, tmpdir):
    """Test that the generator creates both frontend and backend directories."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    frontend_dir = os.path.join(str(output_dir), "frontend")
    backend_dir = os.path.join(str(output_dir), "backend")

    assert os.path.isdir(frontend_dir), "WebAppGenerator should create a frontend directory"
    assert os.path.isdir(backend_dir), "WebAppGenerator should create a backend directory"


def test_web_app_generator_creates_docker_compose(domain_model, gui_model, tmpdir):
    """Test that generate() creates a docker-compose.yml."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    docker_compose = os.path.join(str(output_dir), "docker-compose.yml")
    assert os.path.isfile(docker_compose), "docker-compose.yml should be generated"

    with open(docker_compose, "r", encoding="utf-8") as f:
        content = f.read()

    assert len(content) > 0, "docker-compose.yml should not be empty"


def test_web_app_generator_creates_dockerfiles(domain_model, gui_model, tmpdir):
    """Test that Dockerfiles are generated for frontend and backend."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    frontend_dockerfile = os.path.join(str(output_dir), "frontend", "Dockerfile")
    backend_dockerfile = os.path.join(str(output_dir), "backend", "Dockerfile")

    assert os.path.isfile(frontend_dockerfile), "Frontend Dockerfile should be generated"
    assert os.path.isfile(backend_dockerfile), "Backend Dockerfile should be generated"


def _host_binding_agent(name):
    from besser.BUML.metamodel.state_machine.agent import Agent, WebSocketPlatform
    agent = Agent(name)
    agent.platforms.append(WebSocketPlatform())
    agent.new_state(name="initial", initial=True)
    return agent


def _run_dockerfile_config_patch(agent_dir):
    """Run the agent Dockerfile's config.yaml RUN step in ``agent_dir``; return the result."""
    import re
    import subprocess
    import sys
    import yaml

    with open(os.path.join(agent_dir, "Dockerfile"), encoding="utf-8") as f:
        match = re.search(r'^RUN python -c "(.*config\.yaml.*)"$', f.read(), re.MULTILINE)
    assert match, "agent Dockerfile has no config.yaml patch step"
    subprocess.run([sys.executable, "-c", match.group(1)], cwd=agent_dir, check=True)
    with open(os.path.join(agent_dir, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


# Editor-shaped config.yaml (AgentConfigYamlEditor defaults bind localhost).
_EDITOR_AGENT_YAML = """agent:
  check_transitions_delay: 5

platforms:
  websocket:
    host: localhost
    port: 8765
    streamlit:
      host: localhost
      port: 5000
"""


@pytest.mark.parametrize("config_yaml", [None, _EDITOR_AGENT_YAML], ids=["template", "editor_yaml"])
def test_compose_agent_binds_all_interfaces(domain_model, gui_model, tmpdir, config_yaml):
    """A localhost-bound agent is unreachable through the compose port mapping."""
    pytest.importorskip("yaml")
    output_dir = str(tmpdir.mkdir("output"))
    agent = _host_binding_agent("helper")
    WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=output_dir,
        agent_models=[agent],
        agent_config_yamls={"helper": config_yaml} if config_yaml else None,
    ).generate()

    cfg = _run_dockerfile_config_patch(os.path.join(output_dir, "agents", "helper"))
    ws = cfg["platforms"]["websocket"]
    assert ws["host"] == "0.0.0.0"
    assert ws["streamlit"]["host"] == "0.0.0.0"
    assert ws["port"] == 8765


def test_standalone_agent_config_stays_on_localhost(tmpdir):
    """Outside docker-compose the agent keeps binding localhost."""
    yaml = pytest.importorskip("yaml")
    from besser.generators.agents.baf_generator import BAFGenerator

    output_dir = str(tmpdir.mkdir("output"))
    BAFGenerator(_host_binding_agent("helper"), output_dir=output_dir).generate()
    with open(os.path.join(output_dir, "config.yaml"), encoding="utf-8") as f:
        ws = yaml.safe_load(f)["platforms"]["websocket"]
    assert ws["host"] == "localhost"
    assert ws["streamlit"]["host"] == "localhost"
