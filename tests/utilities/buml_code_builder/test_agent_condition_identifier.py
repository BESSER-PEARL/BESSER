"""A custom condition's name is not always a Python identifier.

``agent_model_builder`` used ``condition.name`` verbatim as the variable it
assigns the ``Condition`` to and passes to ``when_condition``. ``NamedElement``
accepts ``is.ready`` or a keyword like ``if``, and the export then failed to
parse. Only the identifier is sanitised; the ``Condition('<name>')`` string the
file emits is unchanged.
"""
import os
import runpy

import pytest

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.state_machine.state_machine import Condition
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import agent_buml_to_json

SOURCE = "def is_ready(session, params):\n    return True\n"


def _export(condition_name, directory):
    agent = Agent("CondAgent")
    start = agent.new_state("start", initial=True)
    done = agent.new_state("done")
    start.when_condition(Condition(condition_name, source=SOURCE)).go_to(done)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), "agent.py")
    agent_model_to_code(agent, path)
    with open(path, encoding="utf-8") as handle:
        return path, handle.read()


def _custom_conditions(json_model):
    return [r["custom"]["condition"] for r in json_model["relationships"].values()
            if r.get("transitionType") == "custom"]


@pytest.mark.parametrize("condition_name", ["is.ready", "if"])
def test_a_non_identifier_condition_name_exports_and_reimports(condition_name, tmp_path):
    path, code = _export(condition_name, tmp_path / "odd")

    # Run from the file: Condition(callable=...) reads the function's source.
    start = next(s for s in runpy.run_path(path)["agent"].states if s.name == "start")
    assert [c.name for t in start.transitions for c in t.conditions] == ["is_ready"]

    baseline = _custom_conditions(agent_buml_to_json(_export("ready", tmp_path / "base")[1]))
    assert _custom_conditions(agent_buml_to_json(code)) == baseline
    assert "def is_ready" in baseline[0][0]


def test_a_condition_without_code_exports_and_reloads(tmp_path):
    # Used to emit ``Condition('x_callable', callable=x_callable)`` with
    # ``x_callable`` never defined: a NameError on load.
    agent = Agent("CondAgent")
    start = agent.new_state("start", initial=True)
    done = agent.new_state("done")
    start.when_condition(Condition("is_ready")).go_to(done)
    path = os.path.join(str(tmp_path), "agent.py")
    agent_model_to_code(agent, path)

    start = next(s for s in runpy.run_path(path)["agent"].states if s.name == "start")
    [condition] = [c for t in start.transitions for c in t.conditions]
    assert condition.name == "is_ready"
    assert condition.code is None
