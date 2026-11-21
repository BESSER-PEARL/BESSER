"""test_api, and the prompt text naming it, exist only where a FastAPI backend does.

test_api drives one generated FastAPI service (``main_api.py`` next to
``sql_alchemy.py``). It was offered to every run, the largest tool schema of a
state-machine run, and the repair prompt named it on every stack. The shell
runbook told the model to create records and call actions with ``req``, whose
results are kept nowhere, and never named test_api, whose scenarios are retained
and replayed as evidence.
"""
from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.runbook import runbook_section
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.prompt_budget import FIXTURES, phase2_request
from tests.spec_driven_agent.shelf_app import generate_backend, shelf_model


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"
    usage = _Usage()


def _names(tools):
    return [tool["name"] for tool in tools]


def test_a_run_without_a_backend_is_not_offered_test_api(tmp_path):
    system, tools = phase2_request(FIXTURES["state_machine"](), tmp_path)
    assert "test_api" not in _names(tools)
    assert "test_api" not in system


def test_a_generated_backend_keeps_test_api(tmp_path):
    system, tools = phase2_request(FIXTURES["shelf"](), tmp_path)
    assert "test_api" in _names(tools)
    assert _names(tools).index("test_api") == _names(tools).index("validate_app") - 1
    assert "Use test_api" in system


def test_the_repair_prompt_names_test_api_only_with_a_backend(tmp_path):
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    orch._offer_test_api()
    assert "test_api" not in orch._phase3_fix_system_prompt([]) and "test_api" not in _names(orch.tools)

    generate_backend(tmp_path, shelf_model())
    orch._offer_test_api()
    assert "test_api" in orch._phase3_fix_system_prompt([]) and "test_api" in _names(orch.tools)


def test_the_runbook_creates_and_acts_through_test_api(tmp_path):
    generate_backend(tmp_path, shelf_model())
    procedure = runbook_section(str(tmp_path)).split("### The procedure", 1)[1]
    create, action = (next(line for line in procedure.splitlines() if marker in line)
                      for marker in ("Create one of every entity", "Call every action"))
    assert "test_api" in create and "test_api" in action
