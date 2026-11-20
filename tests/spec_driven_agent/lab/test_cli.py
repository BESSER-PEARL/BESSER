"""The lab agent's command line, against a real lab layout (``conftest.lab_layout``) and a scripted model."""

import json
import os
import subprocess
import sys

import pytest

import besser
from besser.spec_driven_agent.lab import cli
from tests.spec_driven_agent.lab.conftest import end_turn, tool_use
from tests.spec_driven_agent.lab.test_session import PLAN


@pytest.fixture
def folders(lab_layout):
    """``--root``, ``--state-dir`` and ``--git-root`` of the layout."""
    return ["--root", lab_layout.root, "--state-dir", lab_layout.state_dir, "--git-root", lab_layout.git_root]


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


def test_a_message_without_a_change_makes_a_draft_plans_it_and_writes_the_outcome(lab_layout, folders, factory,
                                                                                   tmp_path, capsys):
    factory.script([tool_use("submit_plan", PLAN)])
    out = tmp_path / "outcome.json"

    code = cli.main(["Greet the user", *folders, "--login", "jordi", "--json", str(out)])

    assert code == 0
    assert factory.calls == [("anthropic", None, None)]
    captured = capsys.readouterr()
    first_line, rest = captured.out.split("\n", 1)
    change_id = first_line.removeprefix("Change ")
    outcome = json.loads(out.read_text(encoding="utf-8"))
    assert outcome["step"] == "plan" and outcome["stop_reason"] == "completed"
    assert outcome["plan"]["title"] == "Add a greeting" and outcome["plan"]["kind"] == "agent"
    assert outcome["change"]["id"] == change_id and outcome["change"]["status"] == "open"
    assert outcome["change"]["branches"]["modeling_agent"] == "lab/jordi/add-a-greeting"
    assert lab_layout.workspace().get_change(change_id).owner == "jordi"
    assert "Title: Add a greeting" in rest
    assert "[plan] Planning the change" in captured.err and "submit_plan" in captured.err


def test_a_message_about_a_named_change_continues_it(lab_layout, folders, factory, capsys):
    draft = lab_layout.workspace().create_draft(owner="jordi")
    factory.script([end_turn("Which agent?"), end_turn("Which agent do you mean?")])

    code = cli.main(["Make the agent greet", *folders, "--change", draft.id, "--login", "ana", "--action", "plan"])

    assert code == 1  # no plan
    out = capsys.readouterr().out
    assert out.startswith(f"Change {draft.id}\n") and "Which agent do you mean?" in out
    assert "ana: Make the agent greet" in lab_layout.workspace().notes(draft)
    assert len(lab_layout.workspace().changes()) == 1


def test_provider_model_and_cost_cap_come_from_the_arguments(folders, factory, tmp_path):
    factory.script([tool_use("submit_plan", PLAN)])
    out = tmp_path / "outcome.json"

    code = cli.main(["Greet", *folders, "--login", "jordi", "--provider", "openai", "--model", "gpt-test",
                     "--max-cost", "0", "--json", str(out)])

    assert factory.calls == [("openai", None, "gpt-test")]
    # A zero cap stops before the first model call.
    assert factory.client.calls == []
    assert code == 1 and json.loads(out.read_text(encoding="utf-8"))["stop_reason"] == "cost_cap"


def test_a_build_of_a_draft_and_an_unknown_change_are_reported_not_raised(lab_layout, folders, factory, capsys):
    factory.script([end_turn("never sent")])

    code = cli.main(["Build", *folders, "--login", "jordi", "--action", "build"])

    assert code == 1 and "is a draft" in capsys.readouterr().err
    code = cli.main(["Build", *folders, "--login", "jordi", "--change", "0" * 12])
    assert code == 1 and "no change 000000000000" in capsys.readouterr().err
    assert factory.client.calls == []


def test_a_missing_key_is_reported_and_makes_no_draft(lab_layout, folders, monkeypatch, capsys):
    def no_key(provider, api_key=None, model=None):
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN.")
    monkeypatch.setattr(cli, "create_llm_client", no_key)

    code = cli.main(["Greet", *folders, "--login", "jordi"])

    assert code == 1 and "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert lab_layout.workspace().changes() == []


def test_the_folders_and_the_login_are_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["Greet", "--root", "r", "--git-root", "g", "--login", "jordi"])
    args = cli.build_parser().parse_args(["Greet", "--root", "r", "--state-dir", "s", "--git-root", "g",
                                          "--login", "jordi"])
    assert (args.message, args.change, args.action) == ("Greet", None, "auto")


def test_the_package_runs_as_a_module():
    checkout = os.path.dirname(os.path.dirname(besser.__file__))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [checkout, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-m", "besser.spec_driven_agent.lab", "--help"],
                            capture_output=True, text=True, env=env, timeout=120)

    assert result.returncode == 0, result.stderr
    for option in ("--root", "--state-dir", "--git-root", "--change", "--login", "MESSAGE"):
        assert option in result.stdout
