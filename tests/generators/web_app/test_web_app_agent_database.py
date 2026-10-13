"""The agent of a generated web app queries the web app's database (issue #530).

A ``DBReply`` with ``db_selection_type='default'`` queries ``db.sql.db1`` of the
agent's ``config.yaml``. ``WebAppGenerator`` points ``db1`` at the backend's SQLite
file and mounts the backend's ``sqlite_data`` volume into every agent container;
standalone agent generation keeps the ``YOUR-DB-*`` placeholders.
"""

import os

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.state_machine.agent import Agent, DBReply, WebSocketPlatform
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.agents.baf_generator import BAFGenerator, link_default_database
from besser.generators.web_app import WebAppGenerator

yaml = pytest.importorskip("yaml")

MODEL_NAME = "Library"
AGENT_DB = {"dialect": "sqlite", "file": f"/app/webapp_data/{MODEL_NAME}.db"}
PLACEHOLDER_DB1 = {
    "dialect": "postgresql",
    "host": "YOUR-DB-HOST",
    "port": 5432,
    "database": "YOUR-DB-NAME",
    "username": "YOUR-DB-USERNAME",
    "password": "YOUR-DB-PASSWORD",
}


def _domain_model():
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    return DomainModel(name=MODEL_NAME, types={book})


def _gui_model():
    screen = Screen(name="Home", description="Home", view_elements={Text(name="t", content="Hi")},
                    is_main_page=True)
    return GUIModel(name="LibraryApp", package="org.library", versionCode="1", versionName="1.0",
                    modules={Module(name="Main", screens={screen})}, description="Library")


def _db_agent(name="helper"):
    agent = Agent(name)
    agent.platforms.append(WebSocketPlatform())
    state = agent.new_state(name="initial", initial=True)
    state.set_body(Body("initial_body", actions=[
        DBReply(db_query_mode="sql", db_operation="select", db_sql_query="SELECT title FROM book"),
    ]))
    return agent


def _sql_dbs(config):
    """``db.sql`` as ``{name: entry}``, the way BAF flattens it to ``db.sql.<name>.<key>``."""
    sql = config["db"]["sql"]
    if isinstance(sql, dict):
        return sql
    return {key: value for item in sql for key, value in item.items()}


def _generate(tmpdir, **kwargs):
    output_dir = str(tmpdir.mkdir("output"))
    WebAppGenerator(model=_domain_model(), gui_model=_gui_model(), output_dir=output_dir,
                    agent_models=[_db_agent()], **kwargs).generate()
    with open(os.path.join(output_dir, "docker-compose.yml"), encoding="utf-8") as f:
        compose = yaml.safe_load(f)
    agent_dir = os.path.join(output_dir, "agents", "helper")
    with open(os.path.join(agent_dir, "config.yaml"), encoding="utf-8") as f:
        config_text = f.read()
    return compose, config_text, agent_dir


def _assert_linked(compose, config):
    backend = compose["services"]["backend"]
    agent = compose["services"]["helper_agent"]
    # The backend runs in /app and keeps its database at ./data/<model name>.db.
    assert f"DATABASE_URL=sqlite:///./data/{MODEL_NAME}.db" in backend["environment"]
    assert "sqlite_data:/app/data" in backend["volumes"]
    # The agent sees the same volume, and db1 names the backend's file inside it.
    assert "sqlite_data:/app/webapp_data" in agent["volumes"]
    assert "helper_agent_data:/app/data" in agent["volumes"]
    assert _sql_dbs(config)["db1"] == AGENT_DB


def test_template_config_points_db1_at_webapp_database(tmpdir):
    compose, config_text, agent_dir = _generate(tmpdir)
    _assert_linked(compose, yaml.safe_load(config_text))
    assert "YOUR-DB-HOST" not in config_text.split("sql:", 1)[1].split("- db2:", 1)[0]

    with open(os.path.join(agent_dir, "helper.py"), encoding="utf-8") as f:
        agent_code = f.read()
    assert "db_name='db1'" in agent_code


def test_link_survives_dockerfile_config_patch(tmpdir):
    """The agent Dockerfile rewrites config.yaml at build time; db1 must survive it."""
    import re
    import subprocess
    import sys

    _, _, agent_dir = _generate(tmpdir)
    with open(os.path.join(agent_dir, "Dockerfile"), encoding="utf-8") as f:
        match = re.search(r'^RUN python -c "(.*config\.yaml.*)"$', f.read(), re.MULTILINE)
    assert match
    subprocess.run([sys.executable, "-c", match.group(1)], cwd=agent_dir, check=True)
    with open(os.path.join(agent_dir, "config.yaml"), encoding="utf-8") as f:
        assert _sql_dbs(yaml.safe_load(f))["db1"] == AGENT_DB


_EDITOR_YAML_PLACEHOLDER = """agent:
  check_transitions_delay: 5
# business data
db:
  sql:
    - db1:
        dialect: postgresql
        host: YOUR-DB-HOST
        port: 5432
        database: YOUR-DB-NAME
        username: YOUR-DB-USERNAME
        password: YOUR-DB-PASSWORD
    - db2:
        dialect: postgresql
        host: YOUR-DB-HOST
        port: 5432
"""


def test_editor_yaml_placeholder_db1_is_linked(tmpdir):
    compose, config_text, _ = _generate(tmpdir, agent_config_yamls={"helper": _EDITOR_YAML_PLACEHOLDER})
    config = yaml.safe_load(config_text)
    _assert_linked(compose, config)
    assert "# business data" in config_text  # rewritten in place, comments kept
    assert _sql_dbs(config)["db2"]["host"] == "YOUR-DB-HOST"


def test_editor_yaml_with_own_db1_is_kept(tmpdir):
    own = _EDITOR_YAML_PLACEHOLDER.replace("YOUR-DB-HOST", "db.example.org", 1) \
        .replace("YOUR-DB-NAME", "shop", 1).replace("YOUR-DB-USERNAME", "u", 1) \
        .replace("YOUR-DB-PASSWORD", "p", 1)
    _, config_text, _ = _generate(tmpdir, agent_config_yamls={"helper": own})
    assert config_text == own


def test_editor_yaml_without_db_section_gets_db1(tmpdir):
    editor_yaml = "agent:\n  check_transitions_delay: 5\n"
    compose, config_text, _ = _generate(tmpdir, agent_config_yamls={"helper": editor_yaml})
    config = yaml.safe_load(config_text)
    _assert_linked(compose, config)
    assert config["agent"]["check_transitions_delay"] == 5


def test_link_can_be_disabled(tmpdir):
    compose, config_text, _ = _generate(tmpdir, link_agent_database=False)
    agent = compose["services"]["helper_agent"]
    assert agent["volumes"] == ["helper_agent_data:/app/data"]
    assert _sql_dbs(yaml.safe_load(config_text))["db1"] == PLACEHOLDER_DB1


def test_compose_is_deterministic(tmpdir):
    first = _generate(tmpdir.mkdir("a"))
    second = _generate(tmpdir.mkdir("b"))
    assert first[:2] == second[:2]


def test_standalone_agent_keeps_placeholders(tmpdir):
    output_dir = str(tmpdir.mkdir("output"))
    BAFGenerator(_db_agent(), output_dir=output_dir).generate()
    with open(os.path.join(output_dir, "config.yaml"), encoding="utf-8") as f:
        config_text = f.read()
    assert "sqlite" not in config_text
    assert "webapp_data" not in config_text
    assert _sql_dbs(yaml.safe_load(config_text))["db1"] == PLACEHOLDER_DB1


def test_standalone_agent_with_editor_yaml_is_unchanged(tmpdir):
    output_dir = str(tmpdir.mkdir("output"))
    BAFGenerator(_db_agent(), output_dir=output_dir, config_yaml=_EDITOR_YAML_PLACEHOLDER).generate()
    with open(os.path.join(output_dir, "config.yaml"), encoding="utf-8") as f:
        assert f.read() == _EDITOR_YAML_PLACEHOLDER


@pytest.mark.parametrize("text", ["", "[1, 2]\n", "a: [unclosed\n"])
def test_link_default_database_edge_cases(text):
    result = link_default_database(text, AGENT_DB)
    if text == "":
        assert _sql_dbs(yaml.safe_load(result))["db1"] == AGENT_DB
    else:
        assert result == text  # not a mapping / not YAML: left alone


def test_link_default_database_mapping_form():
    text = "db:\n  sql:\n    db1:\n      dialect: postgresql\n      host: YOUR-DB-HOST\n"
    assert _sql_dbs(yaml.safe_load(link_default_database(text, AGENT_DB)))["db1"] == AGENT_DB


_DECOY_YAML = (
    "# my agent\n"
    "backups:\n"
    "  - db1:\n"
    "      keep: 7  # days\n"
    "db:\n"
    "  monitoring:\n"
    "  - db1:\n"
    "      enabled: true\n"
    "  sql:\n"
    "  # the default database\n"
    "  - db1:\n"
    "      dialect: postgresql\n"
    "      host: YOUR-DB-HOST\n"
    "  - analytics:\n"
    "      dialect: sqlite\n"
    "      file: a.db\n"
)


def test_link_default_database_skips_db1_items_outside_db_sql():
    """Only ``db.sql``'s ``- db1:`` is the agent's database; an earlier
    ``- db1:`` under another key must be left as it is."""
    result = link_default_database(_DECOY_YAML, AGENT_DB)
    config = yaml.safe_load(result)
    assert _sql_dbs(config)["db1"] == AGENT_DB
    assert _sql_dbs(config)["analytics"] == {"dialect": "sqlite", "file": "a.db"}
    assert config["backups"] == [{"db1": {"keep": 7}}]
    assert config["db"]["monitoring"] == [{"db1": {"enabled": True}}]
    # Rewritten in place: comments survive.
    assert "# my agent\n" in result and "# days" in result and "# the default database" in result


def test_link_default_database_falls_back_when_db1_is_not_a_block_item():
    """No ``- db1:`` block under ``db.sql`` to rewrite in place: parse/update/dump."""
    text = "other:\n  - db1:\n      x: 1\ndb:\n  sql: [{db1: {host: YOUR-DB-HOST}}]\n"
    config = yaml.safe_load(link_default_database(text, AGENT_DB))
    assert _sql_dbs(config)["db1"] == AGENT_DB
    assert config["other"] == [{"db1": {"x": 1}}]
