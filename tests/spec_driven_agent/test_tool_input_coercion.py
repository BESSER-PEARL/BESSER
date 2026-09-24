"""Stringified array/object tool arguments are decoded on every provider path.

Live 2026-09-24: Sonnet 5 sent ``submit_requirements`` with the array as a JSON
string in 16 of 30 fresh calls (Haiku 4.5: 0 of 10), so the requirements
ledger returned nothing. The same slip on ``generate_fastapi_backend`` turns
``http_methods`` into characters and generates a backend with no endpoints.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic.types as at
import pytest

from besser.spec_driven_agent.agent.tools import get_tools_for
from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient, _openai_response_to_common
from besser.spec_driven_agent.providers.tool_input import coerce_to_schema, normalize_tool_blocks

REQUIREMENTS_TOOL = {
    "name": "submit_requirements",
    "input_schema": {"type": "object", "properties": {"requirements": {
        "type": "array",
        "items": {"type": "object", "properties": {"text": {"type": "string"}, "kind": {"type": "string"}}},
    }}},
}
# The shape Sonnet 5 actually sent: the whole input again, as a string.
SONNET_ANSWER = {"requirements": (
    '{"requirements":[\n'
    '{"kind":"uniqueness","text":"Each person has a unique identifying number."},\n'
    '{"kind":"validation","text":"An email address must have the usual shape."}\n]}'
)}
DECODED = [{"kind": "uniqueness", "text": "Each person has a unique identifying number."},
           {"kind": "validation", "text": "An email address must have the usual shape."}]
ALL_TOOLS = get_tools_for(has_domain_model=True, has_gui_model=True, has_agent_model=True,
                          has_state_machines=True, has_quantum_circuit=True, has_object_model=True,
                          has_bpmn_model=True, has_nn_model=True, allow_shell=True)


def _tool(name):
    return next(t for t in ALL_TOOLS if t["name"] == name)


# ------------------------------------------------------------------ decoding


@pytest.mark.parametrize("value", [SONNET_ANSWER["requirements"], json.dumps(DECODED)])
def test_a_stringified_array_is_decoded(value):
    assert coerce_to_schema({"requirements": value}, REQUIREMENTS_TOOL["input_schema"]) == {"requirements": DECODED}


def test_nested_stringified_values_are_decoded_at_every_depth():
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"tags": {"type": "array"}, "meta": {"type": "object"}}}}}}
    value = {"items": json.dumps([{"tags": '["a","b"]', "meta": '{"k": 1}'}])}
    assert coerce_to_schema(value, schema) == {"items": [{"tags": ["a", "b"], "meta": {"k": 1}}]}


def test_the_real_tools_that_take_arrays_get_real_arrays():
    methods = coerce_to_schema({"http_methods": '["GET", "POST"]'}, _tool("generate_fastapi_backend")["input_schema"])
    assert methods["http_methods"] == ["GET", "POST"]
    ids = coerce_to_schema({"action": "done", "ids": "[1, 2]"}, _tool("task_list")["input_schema"])
    assert ids["ids"] == [1, 2]
    status = _tool("test_api")["input_schema"]
    request = {"requests": json.dumps([{"method": "GET", "path": "/x", "expected_status": "[200, 201]"}])}
    assert coerce_to_schema(request, status)["requests"][0]["expected_status"] == [200, 201]


# ------------------------------------------------------------- never decoded


@pytest.mark.parametrize("value", [
    "not json", '"just a string"', "42", '{"other": []}', '{"requirements": [], "extra": 1}', "[1, 2",
])
def test_a_string_that_does_not_decode_to_the_declared_shape_is_left_alone(value):
    assert coerce_to_schema({"requirements": value}, REQUIREMENTS_TOOL["input_schema"]) == {"requirements": value}


def test_an_object_field_does_not_accept_an_array_and_vice_versa():
    schema = {"type": "object", "properties": {"obj": {"type": "object"}, "arr": {"type": "array"}}}
    value = {"obj": "[1]", "arr": '{"a": 1}'}
    assert coerce_to_schema(value, schema) == value


def test_a_field_that_allows_a_string_keeps_its_string():
    schema = {"type": "object", "properties": {"x": {"type": ["string", "array"]}, "y": {"anyOf": [
        {"type": "string"}, {"type": "array"}]}}}
    assert coerce_to_schema({"x": "[1]", "y": "[1]"}, schema) == {"x": "[1]", "y": "[1]"}


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t["name"])
def test_no_string_argument_of_any_real_tool_is_ever_decoded(tool):
    """File content, paths, commands and code that look like JSON must reach
    the executor byte for byte."""
    properties = tool["input_schema"].get("properties", {})
    for name, schema in properties.items():
        if "string" not in json.dumps(schema):
            continue
        if schema.get("type") not in ("string", None) and "anyOf" not in schema:
            continue
        for text in ('["a"]', '{"a": 1}', "[1, 2]"):
            assert coerce_to_schema({name: text}, tool["input_schema"]) == {name: text}, name


def test_valid_input_is_returned_unchanged():
    value = {"requirements": DECODED}
    assert coerce_to_schema(value, REQUIREMENTS_TOOL["input_schema"]) == value
    assert coerce_to_schema({"unknown": "[1]"}, REQUIREMENTS_TOOL["input_schema"]) == {"unknown": "[1]"}
    assert coerce_to_schema("[1]", None) == "[1]"


# ------------------------------------------------------------ every provider


def _anthropic_block(inp):
    return at.ToolUseBlock(type="tool_use", id="toolu_1", name="submit_requirements", input=inp)


def test_anthropic_chat_decodes_the_sonnet_answer():
    client = ClaudeLLMClient(api_key="sk-ant-test", model="claude-sonnet-5")
    client._client = MagicMock()
    client._client.messages.create.return_value = SimpleNamespace(
        stop_reason="tool_use", content=[_anthropic_block(dict(SONNET_ANSWER))],
        usage=at.Usage(input_tokens=1, output_tokens=1))
    result = client.chat(system="s", messages=[{"role": "user", "content": "x"}], tools=[REQUIREMENTS_TOOL],
                         force_tool="submit_requirements")
    assert result["content"][0].input == {"requirements": DECODED}


def test_anthropic_stream_decodes_the_sonnet_answer():
    client = ClaudeLLMClient(api_key="sk-ant-test", model="claude-sonnet-5")
    client._client = MagicMock()
    final = SimpleNamespace(stop_reason="tool_use", content=[_anthropic_block(dict(SONNET_ANSWER))],
                            usage=at.Usage(input_tokens=1, output_tokens=1))
    client._client.messages.stream.return_value.__enter__.return_value = SimpleNamespace(
        text_stream=iter([]), get_final_message=lambda: final)
    done = list(client.chat_stream(system="s", messages=[], tools=[REQUIREMENTS_TOOL]))[-1]
    assert done["content"][0].input == {"requirements": DECODED}


def _openai_tool_call():
    return SimpleNamespace(id="call_1", function=SimpleNamespace(
        name="submit_requirements", arguments=json.dumps(SONNET_ANSWER)))


def test_openai_compatible_chat_decodes_a_nested_string():
    """OpenAI, Mistral, Nebius, the free tier and Qwen all return through
    ``_openai_response_to_common``; its json.loads only decodes one level."""
    message = SimpleNamespace(content=None, tool_calls=[_openai_tool_call()])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])
    result = _openai_response_to_common(response, [REQUIREMENTS_TOOL])
    assert result["content"][0].input == {"requirements": DECODED}


def test_openai_compatible_stream_decodes_a_nested_string():
    from tests.spec_driven_agent.test_openai_provider import _make_openai_provider
    provider = _make_openai_provider()
    call = _openai_tool_call()
    chunks = [SimpleNamespace(usage=None, choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(index=0, id=call.id, function=call.function)]),
        finish_reason="tool_calls")])]
    provider._client.chat.completions.create.return_value = iter(chunks)
    done = list(provider.chat_stream(system="s", messages=[], tools=[REQUIREMENTS_TOOL]))[-1]
    assert done["content"][0].input == {"requirements": DECODED}


def test_blocks_for_unknown_tools_and_text_blocks_pass_through():
    text = at.TextBlock(type="text", text='["a"]')
    other = at.ToolUseBlock(type="tool_use", id="t", name="not_offered", input={"x": "[1]"})
    assert normalize_tool_blocks([text, other], [REQUIREMENTS_TOOL]) == [text, other]
    assert other.input == {"x": "[1]"}
