"""Findings of the 2026-09-24 prompt audit, each pinned to what the code does.

The prompts serve Sonnet 5, gpt-5.6, Qwen3-Coder and mistral-small alike, so
these fixes remove contradictions and wrong contracts, not guidance a weaker
model relies on.
"""

import fnmatch
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic.types as at

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.agent.tools import get_tools_for
from besser.spec_driven_agent.planning import gap_analyzer
from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient

TOOLS = {t["name"]: t for t in get_tools_for(has_domain_model=True, allow_shell=True)}


def _model() -> DomainModel:
    booking = Class(name="Booking")
    booking.attributes = {Property(name="id", type=StringType, is_id=True)}
    return DomainModel(name="Hotel", types={booking})


def _prompt(domain_model):
    return build_system_prompt(domain_model, None, None, inventory="", instructions="Build it.", max_turns=10)


def test_the_gap_prompt_no_longer_asks_for_delete_tasks():
    """The system prompt forbids deleting the scaffold; the user prompt still
    asked for 'explicit delete tasks', which devstral acted on (48177d465)."""
    prompt = gap_analyzer._build_user_prompt("req", "generate_web_app", "{}", "inv").lower()
    assert "delete tasks" not in prompt
    assert "read_file/write_file" not in gap_analyzer._SYSTEM_PROMPT


def test_model_query_tools_are_named_only_when_they_are_offered():
    offered_without_model = {t["name"] for t in get_tools_for(has_domain_model=False, has_gui_model=True)}
    assert "query_class" not in offered_without_model
    assert "query_class(name)" not in _prompt(None)
    assert "query_class(name)" in _prompt(_model())


def test_the_search_glob_contract_matches_the_executor():
    """The executor matches the glob against the bare file name, so the old
    '**/*.ts' example could never match anything."""
    assert fnmatch.fnmatch("app.ts", "**/*.ts") is False
    glob_doc = TOOLS["search_in_files"]["input_schema"]["properties"]["file_glob"]["description"]
    assert "**/" not in glob_doc and "NAME only" in glob_doc
    doc = TOOLS["search_in_files"]["description"]
    for fact in ("case-insensitive", "50 matches", "200 chars", "node_modules"):
        assert fact in doc


def test_run_command_documents_truncation_and_the_skipped_result():
    doc = TOOLS["run_command"]["description"]
    for fact in ("full_output_path", "skipped=true", "120-second", "refused"):
        assert fact in doc


def test_check_syntax_says_what_it_does_not_check():
    assert "validate_app" in TOOLS["check_syntax"]["description"]
    assert "neither file" in TOOLS["install_dependencies"]["description"]


def _claude(model="claude-sonnet-5"):
    client = ClaudeLLMClient(api_key="sk-ant-test", model=model)
    client._client = MagicMock()
    client._client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn", content=[at.TextBlock(type="text", text="ok")],
        usage=at.Usage(input_tokens=1, output_tokens=1))
    return client


def test_a_forced_tool_call_turns_thinking_off_and_a_free_call_does_not():
    """Bedrock (PIA) takes a forced tool_choice only with thinking disabled."""
    client = _claude()
    tool = {"name": "t", "input_schema": {"type": "object", "properties": {}}}
    client.chat(system="s", messages=[{"role": "user", "content": "x"}], tools=[tool], force_tool="t")
    forced = client._client.messages.create.call_args.kwargs
    assert forced["tool_choice"] == {"type": "tool", "name": "t"}
    assert forced["thinking"] == {"type": "disabled"}
    client.chat(system="s", messages=[{"role": "user", "content": "x"}], tools=[tool])
    assert "thinking" not in client._client.messages.create.call_args.kwargs

