"""The lab agent's command line, against a real lab working copy and a scripted model."""

import json
import os
import shutil
import subprocess
import sys

import pytest

import besser
from besser.spec_driven_agent.lab import cli
from besser.spec_driven_agent.lab.contracts import FRONTEND_SUBMODULE_PATH
from tests.spec_driven_agent.lab.conftest import end_turn, tool_use
from tests.spec_driven_agent.lab.test_session import PLAN, git, init_repo


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    base = tmp_path_factory.mktemp("lab_cli_template")
    editor_origin = str(base / "editor-origin")
    init_repo(editor_origin, "develop", {"packages/webapp/src/app.ts": "export const app = 1;\n"})
    besser = str(base / "root" / "besser")
    init_repo(besser, "development", {"README.md": "BESSER\n"})
    git(besser, "submodule", "add", "-q", editor_origin, FRONTEND_SUBMODULE_PATH)
    git(besser, "commit", "-q", "-m", "add the editor")
    init_repo(str(base / "root" / "modeling-agent"), "develop", {"src/agent.py": "AGENT = 1\n"})
    return base / "root"


@pytest.fixture
def root(template, tmp_path):
    target = tmp_path / "root"
    shutil.copytree(template, target)
    return str(target)


@pytest.fixture
def factory(monkeypatch, scripted_client):
    """Replaces the LLM client factory; ``factory.client`` answers, ``factory.calls`` records the arguments."""
    class Factory:
        def __init__(self):
            self.calls = []
            self.client = None

        def __call__(self, provider, api_key=None, model=None):
            self.calls.append((provider, api_key, model))
            return self.client

        def script(self, responses):
            self.client = scripted_client(responses)

    made = Factory()
    monkeypatch.setattr(cli, "create_llm_client", made)
    return made


def test_a_message_plans_the_change_and_writes_the_outcome_as_json(root, factory, tmp_path, capsys):
    factory.script([tool_use("submit_plan", PLAN)])
    out = tmp_path / "outcome.json"

    code = cli.main(["--root", root, "--user", "Jordi", "--message", "Greet the user", "--json", str(out)])

    assert code == 0
    assert factory.calls == [("anthropic", None, None)]
    outcome = json.loads(out.read_text(encoding="utf-8"))
    assert outcome["step"] == "plan" and outcome["stop_reason"] == "completed"
    assert outcome["plan"]["title"] == "Add a greeting" and outcome["plan"]["kind"] == "agent"
    assert outcome["change"]["branches"]["modeling_agent"] == "lab/jordi/add-a-greeting"
    captured = capsys.readouterr()
    assert "Title: Add a greeting" in captured.out
    assert "[plan] Planning the change" in captured.err and "submit_plan" in captured.err


def test_provider_model_and_cost_cap_come_from_the_arguments(root, factory, tmp_path):
    factory.script([tool_use("submit_plan", PLAN)])
    out = tmp_path / "outcome.json"

    code = cli.main(["--root", root, "--user", "Jordi", "--message", "Greet", "--provider", "openai",
                     "--model", "gpt-test", "--max-cost", "0", "--json", str(out)])

    assert factory.calls == [("openai", None, "gpt-test")]
    # A zero cap stops before the first model call.
    assert factory.client.calls == []
    assert code == 1 and json.loads(out.read_text(encoding="utf-8"))["stop_reason"] == "cost_cap"


def test_a_build_without_a_change_is_reported_not_raised(root, factory, capsys):
    factory.script([end_turn("never sent")])

    code = cli.main(["--root", root, "--user", "Jordi", "--message", "Build", "--action", "build"])

    assert code == 1 and "no active change" in capsys.readouterr().err


def test_a_missing_key_is_reported(root, monkeypatch, capsys):
    def no_key(provider, api_key=None, model=None):
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN.")
    monkeypatch.setattr(cli, "create_llm_client", no_key)

    code = cli.main(["--root", root, "--user", "Jordi", "--message", "Greet"])

    assert code == 1 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_the_package_runs_as_a_module():
    checkout = os.path.dirname(os.path.dirname(besser.__file__))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [checkout, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-m", "besser.spec_driven_agent.lab", "--help"],
                            capture_output=True, text=True, env=env, timeout=120)

    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout and "--message" in result.stdout
