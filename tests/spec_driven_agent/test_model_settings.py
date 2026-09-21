"""Per-model inference settings: the registry, and the request it produces.

The gap these cover: the OpenAI-compatible request path -- the one every
Nebius/keyless Qwen run goes through -- sent no sampling parameters at all and
inherited whatever the provider defaulted to. These assert the registry's
values, that they reach the wire in the right shape, and (the part most likely
to be broken by a later edit) that a model with no row is left exactly alone.

The "left alone" tests are the important half: adding a row for one model must
never change the request for another, and gpt-5.6 measures 12/12 on the current
no-sampling configuration.
"""

from unittest.mock import MagicMock

import pytest

from besser.spec_driven_agent.providers import model_settings
from besser.spec_driven_agent.providers.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ClaudeLLMClient,
    MistralProvider,
    NebiusProvider,
    OpenAIProvider,
    _needs_reasoning_none_for_tools,
)
from besser.spec_driven_agent.providers.model_settings import (
    max_output_tokens,
    reasoning_effort_for_tools,
    sampling_kwargs,
    settings_for,
)

NEBIUS_QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _captured_kwargs(provider) -> dict:
    """Run one ``chat`` turn against a mocked SDK and return the request."""
    response = MagicMock()
    response.usage = None
    response.model = provider.model
    choice = MagicMock()
    choice.message.content = "hi"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response.choices = [choice]

    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = response
    provider.chat(system="s", messages=[{"role": "user", "content": "u"}], tools=[])
    return provider._client.chat.completions.create.call_args.kwargs


# ======================================================================
# The registry
# ======================================================================

class TestRegistryValues:

    def test_nebius_qwen_carries_the_model_card_values(self):
        # Qwen3-30B-A3B-Instruct-2507 generation_config.json ships
        # temperature 0.7 / top_p 0.8 / top_k 20; the card's Best Practices
        # section adds MinP=0. Aider ships the same four for Qwen3.
        settings = settings_for(NEBIUS_QWEN)
        assert settings.sampling == {"temperature": 0.7, "top_p": 0.8}
        assert settings.extra_body == {"top_k": 20, "min_p": 0.0}

    def test_qwen3_coder_sends_only_protocol_parameters(self):
        # Same family, different card: qwen3-coder recommends
        # repetition_penalty 1.05 rather than min_p. It is withheld along with
        # top_k because this row serves the keyless tier and the self-hosted
        # Ollama box, neither verified to tolerate a non-protocol body key.
        settings = settings_for("qwen3-coder:30b")
        assert settings.sampling == {"temperature": 0.7, "top_p": 0.8}
        assert settings.extra_body == {}

    def test_gpt_5_6_carries_no_sampling_only_the_tool_quirk(self):
        settings = settings_for("gpt-5.6-terra")
        assert settings.sampling == {}
        assert settings.extra_body == {}
        assert settings.reasoning_effort_for_tools == "none"

    @pytest.mark.parametrize("model", [
        "gpt-4o", "claude-sonnet-4-6", "mistral-large-latest",
        "deepseek-v4-pro", "llama-3.3-70b", "", None,
    ])
    def test_unknown_models_get_empty_settings(self, model):
        assert sampling_kwargs(model) == {}
        assert reasoning_effort_for_tools(model) is None


class TestMarkerDisambiguation:
    """The two Qwen rows must not catch each other's ids."""

    def test_nebius_id_does_not_match_the_coder_row(self):
        assert settings_for(NEBIUS_QWEN).extra_body == {"top_k": 20, "min_p": 0.0}

    @pytest.mark.parametrize("model", [
        "qwen3-coder:30b",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "qwen3-coder-30b-a3b-instruct",
    ])
    def test_coder_ids_do_not_pick_up_min_p(self, model):
        # min_p rides in extra_body, which is the field these endpoints are
        # unverified for. Leaking the Nebius row onto them is the failure.
        assert settings_for(model).extra_body == {}

    def test_matching_is_case_insensitive(self):
        assert settings_for(NEBIUS_QWEN.upper()) is settings_for(NEBIUS_QWEN.lower())

    def test_a_longer_marker_wins_over_a_shorter_one(self):
        # Guards the ordering bug class the pricing table warns about
        # (gpt-5.5 being swallowed by the gpt-5 substring).
        markers = [marker for marker, _ in model_settings._REGISTRY]
        assert markers == sorted(markers, key=len, reverse=True)


class TestNoThinkingKnob:
    """Both Qwen rows are the non-thinking -2507 variants.

    Each card states the model "supports only non-thinking mode and does not
    generate <think></think> blocks", and probing Nebius with an explicit
    step-by-step prompt returned no <think> tag and no reasoning_content. A
    knob for a mode the model does not have would be dead config that reads
    as a supported feature.
    """

    def test_no_thinking_parameter_is_configured_anywhere(self):
        for _, settings in model_settings._REGISTRY:
            keys = {*settings.sampling, *settings.extra_body}
            assert not {"enable_thinking", "thinking", "chat_template_kwargs"} & keys

    def test_qwen_rows_request_no_reasoning_effort(self):
        for model in (NEBIUS_QWEN, "qwen3-coder:30b"):
            assert reasoning_effort_for_tools(model) is None


# ======================================================================
# What reaches the wire
# ======================================================================

class TestSamplingReachesTheRequest:

    def test_nebius_request_carries_sampling_and_extra_body(self):
        # FAILS before this change: the request carried only model,
        # max_tokens and messages.
        kwargs = _captured_kwargs(NebiusProvider(api_key="k", model=NEBIUS_QWEN))
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.8
        assert kwargs["extra_body"] == {"top_k": 20, "min_p": 0.0}

    def test_top_k_and_min_p_never_ride_as_top_level_kwargs(self):
        # The openai SDK raises TypeError on an unknown kwarg (verified
        # against 2.36.0), so a non-protocol parameter sent at top level
        # would fail every call rather than be ignored.
        kwargs = _captured_kwargs(NebiusProvider(api_key="k", model=NEBIUS_QWEN))
        assert "top_k" not in kwargs
        assert "min_p" not in kwargs

    def test_keyless_qwen_coder_request_carries_temperature_and_top_p(self):
        # FAILS before this change.
        provider = OpenAIProvider(
            api_key="free", model="qwen3-coder:30b",
            base_url="https://free.example/v1",
        )
        kwargs = _captured_kwargs(provider)
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.8
        assert "extra_body" not in kwargs

    def test_sampling_follows_the_planning_model_override(self):
        # A planning turn calls a different model; the settings must key on
        # the model actually being called.
        provider = OpenAIProvider(api_key="k", model="gpt-4o")
        response = MagicMock()
        response.usage = None
        response.model = NEBIUS_QWEN
        choice = MagicMock()
        choice.message.content = "hi"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response.choices = [choice]
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = response

        provider.chat(system="s", messages=[{"role": "user", "content": "u"}],
                      tools=[], model_override=NEBIUS_QWEN)
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7

    def test_sampling_also_applies_on_the_streaming_path(self):
        # FAILS before this change. The streaming path is the one Phase 2
        # actually uses, so omitting it here would ship a table that only
        # affects non-streaming calls.
        provider = NebiusProvider(api_key="k", model=NEBIUS_QWEN)
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = iter(())
        list(provider.chat_stream(system="s",
                                  messages=[{"role": "user", "content": "u"}],
                                  tools=[]))
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["extra_body"] == {"top_k": 20, "min_p": 0.0}


class TestUnknownModelsAreUntouched:
    """A no-op for anything without a row -- the safety property."""

    @pytest.mark.parametrize("model", ["gpt-4o", "mistral-large-latest"])
    def test_request_carries_no_sampling_keys(self, model):
        kwargs = _captured_kwargs(OpenAIProvider(api_key="k", model=model))
        for key in ("temperature", "top_p", "extra_body", "top_k", "min_p"):
            assert key not in kwargs

    def test_gpt_5_6_gets_no_temperature(self):
        # OpenAI reasoning models reject temperature/top_p, and this model is
        # the one at 12/12. Nothing here may change its request.
        kwargs = _captured_kwargs(OpenAIProvider(api_key="k", model="gpt-5.6-terra"))
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "extra_body" not in kwargs


# ======================================================================
# Consolidated per-model logic
# ======================================================================

class TestReasoningEffortMigration:
    """Behaviour preserved after moving the model half into the registry."""

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_gpt_5_6_still_needs_reasoning_none_on_openai(self, model):
        assert _needs_reasoning_none_for_tools(model, None) is True
        assert _needs_reasoning_none_for_tools(model, "https://api.openai.com/v1") is True

    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5", "gpt-4o", NEBIUS_QWEN])
    def test_other_models_still_do_not(self, model):
        assert _needs_reasoning_none_for_tools(model, None) is False

    def test_a_gateway_still_suppresses_the_flag(self):
        # The keyless tier 400s on reasoning_effort="none" -- it is not in the
        # gateway's own enum.
        assert _needs_reasoning_none_for_tools(
            "gpt-5.6-luna", "https://api.commandcode.ai/v1") is False

    def test_the_value_sent_comes_from_the_registry(self):
        provider = OpenAIProvider(api_key="k", model="gpt-5.6-terra",
                                  base_url="https://api.openai.com/v1")
        response = MagicMock()
        response.usage = None
        response.model = "gpt-5.6-terra"
        choice = MagicMock()
        choice.message.content = "hi"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response.choices = [choice]
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = response
        provider.chat(
            system="s", messages=[{"role": "user", "content": "u"}],
            tools=[{"name": "t", "description": "d",
                    "input_schema": {"type": "object", "properties": {}}}],
        )
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == "none"


class TestMaxTokensSingleSource:
    """The four identical DEFAULT_MAX_TOKENS class attributes now share one
    constant, so the ceiling is raised in one place rather than four."""

    def test_every_provider_reads_the_shared_default(self):
        for provider_cls in (ClaudeLLMClient, OpenAIProvider,
                             MistralProvider, NebiusProvider):
            assert provider_cls.DEFAULT_MAX_TOKENS is DEFAULT_MAX_OUTPUT_TOKENS

    def test_the_value_is_unchanged(self):
        assert DEFAULT_MAX_OUTPUT_TOKENS == 16384

    def test_no_row_overrides_it_today(self):
        # Instruct-2507's card recommends a 16,384 output length while the
        # from-scratch path raises the live ceiling to 32,768. That is left
        # alone deliberately: lowering it trades a soft recommendation for a
        # hard truncation risk, unmeasured. This test is the tripwire if
        # someone adds an override without evidence.
        for _, settings in model_settings._REGISTRY:
            assert settings.max_output_tokens is None
        assert max_output_tokens(NEBIUS_QWEN) == DEFAULT_MAX_OUTPUT_TOKENS

    def test_an_explicit_max_tokens_still_wins(self):
        assert NebiusProvider(api_key="k", max_tokens=4096).max_tokens == 4096


class TestContextWindowsStayInCompaction:
    """Windows are deliberately NOT duplicated here.

    ``compaction.effective_threshold`` resolves them through a four-source
    precedence chain (measured self-hosted deployment > provider catalog >
    advertised family window > default); a flat copy in the registry would
    lose that ordering and give two places to edit.
    """

    def test_registry_holds_no_window_values(self):
        assert not hasattr(model_settings.ModelSettings(), "context_window")

    def test_nebius_qwen_window_still_resolves_from_compaction(self):
        from besser.spec_driven_agent.agent import compaction

        # 262,144 native (Nebius console + the model card), so the threshold
        # must stay far above the 80k flat default rather than be clamped by
        # the self-hosted qwen3 row -- which is why namespaced ids skip the
        # measured table.
        assert compaction.effective_threshold(NEBIUS_QWEN) > compaction.COMPACT_TOKEN_THRESHOLD
