"""Tests for multi-agent support in WebAppGenerator.

Covers:
    - Multi-agent: each agent lands under ``agents/<slug>/`` with its own Dockerfile.
    - docker-compose.yml emits one service block per agent with distinct ports.
    - Backward compatibility with the deprecated ``agent_model=``/``agent_config=`` scalars.
    - ``agent_slug`` normalization (spaces, casing, punctuation).

BAFGenerator is patched so tests don't depend on its runtime behavior — we only
verify the WebAppGenerator wiring around it.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, StringType,
    BinaryAssociation, Multiplicity,
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.generators.web_app import WebAppGenerator
from besser.generators.web_app.web_app_generator import agent_slug


def _make_domain_model():
    name_prop = Property(name="name", type=StringType)
    user = Class(name="User", attributes={name_prop})
    item = Class(name="Item", attributes={Property(name="title", type=StringType)})
    user_end = Property(name="user_end", type=user, multiplicity=Multiplicity(1, 1))
    item_end = Property(name="item_end", type=item, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="UserItem", ends={user_end, item_end})
    return DomainModel(name="TestModel", types={user, item}, associations={assoc})


def _make_gui_model():
    text1 = Text(name="title_text", content="Welcome")
    screen1 = Screen(
        name="Dashboard",
        description="Main dashboard",
        view_elements={text1},
        is_main_page=True,
    )
    module1 = Module(name="AppModule", screens={screen1})
    return GUIModel(
        name="TestApp",
        package="com.test.app",
        versionCode="1",
        versionName="1.0",
        modules={module1},
        description="Test web application",
    )


def _fake_agent(name):
    """Minimal Agent-shaped object.

    We can't instantiate ``besser.BUML.metamodel.state_machine.agent.Agent``
    cleanly in a unit test (it drags in its own validation), and anyway we
    only need ``.name`` for WebAppGenerator's slug + template logic —
    BAFGenerator is patched in these tests.
    """
    return SimpleNamespace(name=name)


@pytest.fixture
def domain_model():
    return _make_domain_model()


@pytest.fixture
def gui_model():
    return _make_gui_model()


@pytest.fixture
def mock_baf_generator():
    """Patch BAFGenerator so tests don't depend on its runtime behavior.

    The mock still creates a file inside the agent's output_dir so
    ``os.walk`` checks for non-empty agent directories pass.
    """
    with patch("besser.generators.agents.baf_generator.BAFGenerator") as mock:
        def _fake_generate(self):
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "agent.py"), "w") as f:
                f.write("# generated agent stub\n")

        instance = MagicMock()
        instance.generate.side_effect = lambda: _fake_generate(instance)
        # Captures the ctor args per call so we can inspect in assertions
        def _side_effect(agent, output_dir=None, config=None):
            inst = MagicMock()
            inst.output_dir = output_dir
            inst.config = config
            inst.agent = agent
            inst.generate.side_effect = lambda: _fake_generate(inst)
            return inst
        mock.side_effect = _side_effect
        yield mock


def test_agent_slug_normalizes_names():
    assert agent_slug("Alpha") == "alpha"
    assert agent_slug("My Agent") == "my_agent"
    assert agent_slug("Support-Bot 2") == "support_bot_2"
    assert agent_slug(_fake_agent("Beta")) == "beta"
    assert agent_slug("") == "agent"


def test_multi_agent_creates_one_dir_per_agent(domain_model, gui_model, tmpdir, mock_baf_generator):
    alpha = _fake_agent("Alpha")
    beta = _fake_agent("Beta")
    output_dir = tmpdir.mkdir("output")

    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
        agent_models=[alpha, beta],
    )
    generator.generate()

    assert os.path.isdir(os.path.join(str(output_dir), "agents", "alpha"))
    assert os.path.isdir(os.path.join(str(output_dir), "agents", "beta"))
    assert os.path.isfile(os.path.join(str(output_dir), "agents", "alpha", "Dockerfile"))
    assert os.path.isfile(os.path.join(str(output_dir), "agents", "beta", "Dockerfile"))

    # Legacy single-agent ``agent/`` directory must NOT be created anymore.
    assert not os.path.isdir(os.path.join(str(output_dir), "agent"))

    assert mock_baf_generator.call_count == 2


def test_multi_agent_docker_compose_contains_all_services(domain_model, gui_model, tmpdir, mock_baf_generator):
    alpha = _fake_agent("Alpha")
    beta = _fake_agent("Beta")
    output_dir = tmpdir.mkdir("output")

    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
        agent_models=[alpha, beta],
    )
    generator.generate()

    with open(os.path.join(str(output_dir), "docker-compose.yml"), encoding="utf-8") as f:
        compose = f.read()

    assert "alpha_agent:" in compose
    assert "beta_agent:" in compose
    # Port offsets: first agent at 8765/5000, second at 8766/5001
    assert "8765:8765" in compose
    assert "8766:8765" in compose
    assert "5000:5000" in compose
    assert "5001:5000" in compose
    # Build contexts must match the agents/<slug>/ layout
    assert "./agents/alpha" in compose
    assert "./agents/beta" in compose
    # JSON-shaped per-agent URL map so the React AgentComponent can route by agent name
    assert '"Alpha":"ws://localhost:8765"' in compose
    assert '"Beta":"ws://localhost:8766"' in compose


def test_docker_compose_agent_urls_parse_as_yaml(domain_model, gui_model, tmpdir, mock_baf_generator):
    """The rendered compose file must be valid YAML and the JSON map must parse."""
    import yaml

    alpha = _fake_agent("Alpha")
    beta = _fake_agent("Beta")
    output_dir = tmpdir.mkdir("output")

    WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
        agent_models=[alpha, beta],
    ).generate()

    with open(os.path.join(str(output_dir), "docker-compose.yml"), encoding="utf-8") as f:
        compose_doc = yaml.safe_load(f)

    import json
    frontend_args = compose_doc["services"]["frontend"]["build"]["args"]
    url_map = json.loads(frontend_args["VITE_AGENT_URLS"])
    assert url_map == {
        "Alpha": "ws://localhost:8765",
        "Beta": "ws://localhost:8766",
    }


def test_agent_config_passed_through_per_agent(domain_model, gui_model, tmpdir, mock_baf_generator):
    alpha = _fake_agent("Alpha")
    beta = _fake_agent("Beta")
    configs = {
        "Alpha": {"intentRecognitionTechnology": "classical"},
        "Beta": {"intentRecognitionTechnology": "llm"},
    }
    output_dir = tmpdir.mkdir("output")

    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
        agent_models=[alpha, beta],
        agent_configs=configs,
    )
    generator.generate()

    call_configs = {call.args[0].name: call.kwargs.get("config") for call in mock_baf_generator.call_args_list}
    assert call_configs["Alpha"] == {"intentRecognitionTechnology": "classical"}
    assert call_configs["Beta"] == {"intentRecognitionTechnology": "llm"}


def test_scalar_agent_model_backcompat(domain_model, gui_model, tmpdir, mock_baf_generator):
    """Deprecated ``agent_model=`` / ``agent_config=`` still work."""
    alpha = _fake_agent("Alpha")
    output_dir = tmpdir.mkdir("output")

    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
        agent_model=alpha,
        agent_config={"intentRecognitionTechnology": "llm"},
    )
    # Deprecated properties return the scalar form
    assert generator.agent_model is alpha
    assert generator.agent_config == {"intentRecognitionTechnology": "llm"}
    assert generator.agent_models == [alpha]

    generator.generate()

    # Output should live under agents/alpha/ (not legacy agent/)
    assert os.path.isdir(os.path.join(str(output_dir), "agents", "alpha"))
    assert not os.path.isdir(os.path.join(str(output_dir), "agent"))


def test_no_agents_skips_agents_directory(domain_model, gui_model, tmpdir, mock_baf_generator):
    """When no agents are passed, no agents/ directory is created and no BAF call."""
    output_dir = tmpdir.mkdir("output")

    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    assert not os.path.isdir(os.path.join(str(output_dir), "agents"))
    assert not os.path.isdir(os.path.join(str(output_dir), "agent"))
    mock_baf_generator.assert_not_called()


def test_underscored_agent_name_produces_underscored_slug(domain_model, gui_model, tmpdir, mock_baf_generator):
    """Underscores in the BUML Agent name survive into the filesystem slug.

    Render hostnames rewrite underscores to dashes, so the GitHub deploy path
    keeps a separate ``host_slug``. This test just pins the filesystem layout
    that the generator itself is responsible for.
    """
    support = _fake_agent("Support_Bot")
    output_dir = tmpdir.mkdir("output")
    WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
        agent_models=[support],
    ).generate()

    assert os.path.isdir(os.path.join(str(output_dir), "agents", "support_bot"))
    with open(os.path.join(str(output_dir), "docker-compose.yml"), encoding="utf-8") as f:
        compose = f.read()
    assert "support_bot_agent:" in compose
    assert "./agents/support_bot" in compose


def test_docker_compose_without_agents_has_no_agent_services(domain_model, gui_model, tmpdir, mock_baf_generator):
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    with open(os.path.join(str(output_dir), "docker-compose.yml"), encoding="utf-8") as f:
        compose = f.read()

    assert "_agent:" not in compose
    assert "VITE_AGENT_URL" not in compose
