"""The rules part of the Phase 2 system prompt is a cache entry shared across runs.

Live r11-web1 (Sonnet 5.5): the first Phase 2 call wrote 7,618 tokens to the
cache while only the tool schemas (5,624) were read from a sibling run. The
system prompt went out as one block with one breakpoint, so its static rules
were written again by every run, and the domain model and data contract sat in
front of the ``## Variable context`` marker, so the "stable" part differed per
project anyway.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic.types as at
import pytest

from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.planning.target import WEB_APP, Target
from besser.spec_driven_agent.providers import llm_client
from besser.spec_driven_agent.providers.llm_client import (
    SYSTEM_CACHE_SPLIT, ClaudeLLMClient, OpenAIProvider,
)
from tests.spec_driven_agent.message_validity import ANTHROPIC_MAX_CACHE_BREAKPOINTS, count_cache_breakpoints
from tests.spec_driven_agent.prompt_budget import conference_model
from tests.spec_driven_agent.shelf_app import shelf_model

SYSTEM = "You are a developer.\n\n## Rules\n\n1. Keep it scoped." + SYSTEM_CACHE_SPLIT + " (per-run)\n\nBuild a shelf."
TOOL = {"name": "read_file", "description": "Read.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
HISTORY = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "text"}]},
]


def _claude(*responses):
    client = ClaudeLLMClient(api_key="sk-ant-test", model="claude-sonnet-5-5")
    client._client = MagicMock()
    client._client.messages.create.side_effect = list(responses)
    return client


def _text_reply():
    return SimpleNamespace(stop_reason="end_turn", content=[at.TextBlock(type="text", text="done")],
                           usage=at.Usage(input_tokens=1, output_tokens=1))


def _tool_reply(name="read_file"):
    return SimpleNamespace(stop_reason="tool_use",
                           content=[at.ToolUseBlock(type="tool_use", id="toolu_9", name=name, input={})],
                           usage=at.Usage(input_tokens=1, output_tokens=1))


# ----------------------------------------------------------------------
# The Anthropic request
# ----------------------------------------------------------------------

def test_the_system_prompt_is_two_blocks_split_at_the_marker():
    blocks = llm_client._with_cache_control(SYSTEM)
    assert [block["text"] for block in blocks] == SYSTEM.split(SYSTEM_CACHE_SPLIT, 1)[0:1] + [
        SYSTEM_CACHE_SPLIT + SYSTEM.split(SYSTEM_CACHE_SPLIT, 1)[1]]
    assert "".join(block["text"] for block in blocks) == SYSTEM
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in blocks)


@pytest.mark.parametrize("system", ["Route this request.", SYSTEM_CACHE_SPLIT + " only a tail"])
def test_a_system_prompt_without_a_stable_part_stays_one_block(system):
    """Planning prompts have no marker; an empty first block would be a 400."""
    assert llm_client._with_cache_control(system) == [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def test_a_tool_loop_request_stays_within_the_breakpoint_limit():
    client = _claude(_text_reply())
    client.chat(system=SYSTEM, messages=HISTORY, tools=[TOOL])
    (request,) = [call.kwargs for call in client._client.messages.create.call_args_list]
    assert len(request["system"]) == 2
    assert count_cache_breakpoints(request) == ANTHROPIC_MAX_CACHE_BREAKPOINTS


def test_a_forced_call_on_a_model_without_forced_tool_choice_stays_within_the_limit():
    """The forced-via-auto path marks the history itself before appending the instruction."""
    client = _claude(_tool_reply("read_file"))
    client.chat(system=SYSTEM, messages=HISTORY, tools=[TOOL], force_tool="read_file")
    (request,) = [call.kwargs for call in client._client.messages.create.call_args_list]
    assert request["tool_choice"] == {"type": "auto"}
    assert count_cache_breakpoints(request) <= ANTHROPIC_MAX_CACHE_BREAKPOINTS


def test_a_streamed_request_stays_within_the_limit():
    client = ClaudeLLMClient(api_key="sk-ant-test", model="claude-sonnet-4-6")
    client._client = MagicMock()
    stream = client._client.messages.stream.return_value.__enter__.return_value
    stream.text_stream = iter(())
    stream.get_final_message.return_value = _text_reply()
    list(client.chat_stream(system=SYSTEM, messages=HISTORY, tools=[TOOL]))
    request = client._client.messages.stream.call_args.kwargs
    assert len(request["system"]) == 2
    assert count_cache_breakpoints(request) <= ANTHROPIC_MAX_CACHE_BREAKPOINTS


# ----------------------------------------------------------------------
# Other providers get the prompt unchanged
# ----------------------------------------------------------------------

@pytest.mark.parametrize("base_url", [None, "https://gateway.example.org/v1"])
def test_an_openai_compatible_request_carries_the_prompt_as_one_string(base_url):
    module = MagicMock()
    with patch.dict("sys.modules", {"openai": module}):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o", base_url=base_url)
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    provider.chat(system=SYSTEM, messages=[{"role": "user", "content": "go"}], tools=[TOOL])
    messages = provider._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM}


# ----------------------------------------------------------------------
# The prompt: everything before the marker is the same for any project
# ----------------------------------------------------------------------

def _prompt(model, instructions, inventory, manifest, design, requirements, gaps, issues, description):
    return build_system_prompt(
        domain_model=model, gui_model=None, agent_model=None, inventory=inventory,
        instructions=instructions, max_turns=40, scoped_issues=issues, gap_tasks=gaps,
        primary_kind="class", endpoint_manifest=manifest, requirements=requirements,
        design_system=design, target=Target(WEB_APP, description, True), routed_stack="")


@pytest.mark.parametrize("design", [False, True])
def test_two_projects_share_the_stable_block_byte_for_byte(design):
    shelf = _prompt(
        shelf_model(isbn=True, author=True), "I want a library app with books and authors, built with React.",
        "Generator `generate_fastapi_backend` produced 9 files:\n  - main_api.py",
        "GET /book/", "\n## Design system (from the GUI model)\n\nUse `ds-card`.\n" if design else "",
        "R1. A book has a title.", ["Create the React frontend"], [], "library web app")
    conference = _prompt(
        conference_model(), "Build a Django-style conference portal with tickets and a dashboard.",
        "Generator `generate_fastapi_backend` produced 40 files:\n  - main_api.py\n  - routers/ticket.py",
        "POST /ticket/", "\n## Design system (from the GUI model)\n\nUse `ds-panel`.\n" if design else "",
        "", ["Add a dashboard", "Sell tickets"], ["Dockerfile is missing"], "conference portal")
    stable, _ = shelf.split(SYSTEM_CACHE_SPLIT, 1)
    assert stable == conference.split(SYSTEM_CACHE_SPLIT, 1)[0]
    for per_project in ("## Domain Model", "## Data contract", "Idiomatic conventions", "ds-card"):
        assert per_project not in stable
    for kept in ("## Domain Model", "## Data contract", "## User request"):
        assert kept in shelf.split(SYSTEM_CACHE_SPLIT, 1)[1]
