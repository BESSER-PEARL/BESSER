"""An agent with no intent classifier set must still match its training sentences.

The default used to be ``LLMIntentClassifierConfiguration(llm_name="")`` even when
the agent had no LLM. BAF's LLM classifier then looks up ``_llms[""]``, swallows
the KeyError and predicts nothing, so every message, training sentences included,
got the fallback reply. Without an LLM the generator now emits BAF's classical
classifier.

``fixtures/library_template_agent_diagram.json`` is the AgentDiagram of the web
editor's "Library (Full Stack)" template
(``packages/webapp/src/main/templates/pattern/project/library_full_stack.json``
in BESSER-Web-Modeling-Editor): intents with training sentences, no LLM.
"""
import importlib.util
import json
import logging
import pathlib
import subprocess
import sys
import textwrap

import pytest

from besser.BUML.metamodel.state_machine.agent import LLMOpenAI
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "library_template_agent_diagram.json"


def _library_agent():
    return process_agent_diagram(json.loads(FIXTURE.read_text(encoding="utf-8")))


def _generate(agent, output_dir, config=None):
    BAFGenerator(agent, output_dir=str(output_dir), config=config,
                 generation_mode=GenerationMode.CODE_ONLY).generate()
    return (output_dir / f"{agent.name}.py").read_text(encoding="utf-8")


def test_library_agent_without_llm_uses_classical_classifier(tmp_path, caplog):
    agent = _library_agent()
    assert not agent.llms and any(intent.training_sentences for intent in agent.intents)

    with caplog.at_level(logging.WARNING, logger="besser.generators.agents.baf_generator"):
        code = _generate(agent, tmp_path)

    assert "ic_config = SimpleIntentClassifierConfiguration()" in code
    assert "LLMIntentClassifierConfiguration(" not in code
    assert 'llm_name=""' not in code
    assert any("classical intent classifier" in record.getMessage() for record in caplog.records)
    assert "needs PyTorch" in (tmp_path / "readme.txt").read_text(encoding="utf-8")


def test_agent_with_llm_keeps_llm_classifier(tmp_path, caplog):
    agent = _library_agent()
    LLMOpenAI(agent=agent, name="gpt-4o-mini", parameters={})

    with caplog.at_level(logging.WARNING, logger="besser.generators.agents.baf_generator"):
        code = _generate(agent, tmp_path)

    assert 'ic_config = LLMIntentClassifierConfiguration(\n    llm_name="gpt-4o-mini",' in code
    assert "SimpleIntentClassifierConfiguration()" not in code
    assert not any("classical intent classifier" in record.getMessage() for record in caplog.records)
    assert "needs PyTorch" not in (tmp_path / "readme.txt").read_text(encoding="utf-8")


def test_llm_named_in_config_keeps_llm_classifier(tmp_path):
    code = _generate(_library_agent(), tmp_path, config={"llm": {"name": "gpt-4o"}})

    assert 'ic_config = LLMIntentClassifierConfiguration(\n    llm_name="gpt-4o",' in code
    assert "SimpleIntentClassifierConfiguration()" not in code


def test_explicit_classical_choice_is_not_the_fallback(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="besser.generators.agents.baf_generator"):
        code = _generate(_library_agent(), tmp_path, config={"intentRecognitionTechnology": "classical"})

    assert "framework='pytorch'" in code
    assert "SimpleIntentClassifierConfiguration()" not in code
    assert not any("classical intent classifier" in record.getMessage() for record in caplog.records)


HEADLESS_SCRIPT = textwrap.dedent("""
    import importlib, json, os, sys
    os.chdir(sys.argv[1])
    sys.path.insert(0, sys.argv[1])
    agent = importlib.import_module(sys.argv[2]).agent
    agent.train()
    engine = agent.nlp_engine
    results = []
    for state, classifier in engine._intent_classifiers.items():
        for intent in state.intents:
            for sentence in intent.training_sentences:
                best = engine.get_best_intent_prediction(classifier.predict(sentence))
                results.append([sentence, intent.name, best.intent.name if best else None])
    print("RESULTS " + json.dumps(results))
""")


@pytest.mark.skipif(
    importlib.util.find_spec("baf") is None or importlib.util.find_spec("torch") is None,
    reason="running a generated agent needs besser-agentic-framework and torch",
)
def test_generated_library_agent_matches_its_training_sentences(tmp_path):
    agent = _library_agent()
    _generate(agent, tmp_path)

    completed = subprocess.run(
        [sys.executable, "-c", HEADLESS_SCRIPT, str(tmp_path), agent.name],
        capture_output=True, text=True, timeout=600, check=True,
    )

    line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULTS "))
    results = json.loads(line[len("RESULTS "):])
    assert results
    assert [(sentence, expected) for sentence, expected, _ in results] == \
        [(sentence, predicted) for sentence, _, predicted in results]
