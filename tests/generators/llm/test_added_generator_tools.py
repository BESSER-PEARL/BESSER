"""The six generators added to the Spec-Driven Agent toolset (supabase,
json_object, baf, bpmn, pytorch, tensorflow) must be exposed ONLY when the
matching model is loaded, and must register without importing torch/tensorflow
(both optional and absent in the hosted backend image).
"""
from besser.generators.llm.tools import (
    GENERATOR_TOOLS,
    _TOOL_MODEL_REQUIREMENTS,
    get_tools_for,
)
from besser.generators.llm.tool_executor import ToolExecutor


def _names(**flags):
    return {t["name"] for t in get_tools_for(**flags)}


def test_all_generator_tools_have_requirement_and_handler():
    for tool in GENERATOR_TOOLS:
        assert tool["name"] in _TOOL_MODEL_REQUIREMENTS, f"{tool['name']} missing requirement"
        assert tool["name"] in ToolExecutor._handlers, f"{tool['name']} missing handler"


def test_supabase_available_with_domain():
    assert "generate_supabase" in _names(has_domain_model=True)


def test_json_object_gated_on_object_model():
    assert "generate_json_object" not in _names(has_domain_model=True)
    assert "generate_json_object" in _names(has_domain_model=True, has_object_model=True)


def test_baf_gated_on_agent_model():
    assert "generate_baf" not in _names(has_domain_model=True)
    assert "generate_baf" in _names(has_agent_model=True)


def test_bpmn_gated_on_bpmn_model():
    assert "generate_bpmn" not in _names(has_domain_model=True)
    assert "generate_bpmn" in _names(has_bpmn_model=True)


def test_pytorch_and_tf_gated_on_nn_model():
    base = _names(has_domain_model=True)
    assert "generate_pytorch" not in base
    assert "generate_tensorflow" not in base
    nn = _names(has_nn_model=True)
    assert "generate_pytorch" in nn
    assert "generate_tensorflow" in nn


def test_registration_does_not_require_torch_or_tensorflow():
    # Importing the tool modules + listing the tools must not import the optional
    # NN deps. If torch/tf were imported at registration time, importing this
    # test module (which imports the tools) would already have failed on a
    # backend image without them. Assert the NN tools are still registered.
    names = {t["name"] for t in GENERATOR_TOOLS}
    assert {"generate_pytorch", "generate_tensorflow"}.issubset(names)


def test_nn_handlers_degrade_when_dep_missing():
    # With no NN model, the handler returns a clean tool error (never raises).
    ex = ToolExecutor(workspace=".")
    for name in ("generate_pytorch", "generate_tensorflow"):
        result = ToolExecutor._handlers[name](ex, {})
        assert isinstance(result, dict)
        assert "error" in result
