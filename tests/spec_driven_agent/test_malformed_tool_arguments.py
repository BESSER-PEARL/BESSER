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


def test_a_tool_call_with_no_name_is_still_answered(tmp_path):
    """An OpenAI-compatible server can return ``finish_reason: tool_calls``
    with an empty function name. The loop skipped such a call, so the next
    request carried a tool call with no result - which both providers reject,
    ending the phase as an API error."""
    from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    class Client:
        model = "mock"

        def __init__(self):
            self.usage = UsageTracker("mock")
            self.requests: list[list[dict]] = []

        def chat(self, system, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return {"stop_reason": "tool_use", "content": [
                    SimpleNamespace(type="tool_use", id="call_x", name="", input={})]}
            return {"stop_reason": "end_turn", "content": [SimpleNamespace(type="text", text="done")]}

    item = Class(name="Item")
    item.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    client = Client()
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=DomainModel(name="Shop", types={item}),
        output_dir=str(tmp_path), use_streaming=False,
        enable_tracing=False, enable_checkpointing=False, enable_requirements_ledger=False,
    )
    orchestrator._resume_messages = [{"role": "user", "content": "Build it"}]  # no planning call
    orchestrator._run_phase2("Build it")

    answer = client.requests[1][-1]["content"]
    assert [block["tool_use_id"] for block in answer] == ["call_x"]
    assert "Unknown tool" in answer[0]["content"]
