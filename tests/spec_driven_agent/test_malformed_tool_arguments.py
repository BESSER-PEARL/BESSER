"""A tool call whose JSON arguments don't parse must reach the executor as an
explicit parse error, never as an empty argument dict."""
from types import SimpleNamespace

from besser.spec_driven_agent.agent.tools import INVALID_ARGUMENTS_KEY
from besser.spec_driven_agent.providers.llm_client import _openai_response_to_common


def _response(arguments: str):
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name="write_file", arguments=arguments))
    message = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])


def test_unparseable_arguments_carry_the_parse_error():
    result = _openai_response_to_common(_response('{"path": "a.py", "content": "x'))
    block = next(b for b in result["content"] if b.type == "tool_use")
    assert INVALID_ARGUMENTS_KEY in block.input
    assert "JSONDecodeError" in block.input[INVALID_ARGUMENTS_KEY]


def test_valid_arguments_are_unchanged():
    result = _openai_response_to_common(_response('{"path": "a.py", "content": "x"}'))
    block = next(b for b in result["content"] if b.type == "tool_use")
    assert block.input == {"path": "a.py", "content": "x"}
