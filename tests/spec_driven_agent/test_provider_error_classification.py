"""Provider errors are classified by status / exception type, not by digits in the text.

Review 2026-09-28 findings:

* a 400 whose message mentions "140133 tokens" contained "401" and became an
  InvalidApiKeyError; "prompt is too long: 205000 tokens" contained "500" and
  was retried five times;
* the OpenAI SDK's timeout reads "Request timed out." (no "timeout"), so it
  was neither retried nor sent down the fallback chain;
* the SDKs' own ``max_retries=2`` stacked on our retry loop;
* pinned Mistral ids (``codestral-2508``) matched the free-family list and
  were priced at $0 on Mistral's paid API, switching the cost cap off;
* a fallback model kept being billed at the primary's rate.
"""

import itertools
from types import SimpleNamespace

import httpx
import openai
import pytest

from besser.spec_driven_agent.errors import InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.providers import llm_client
from besser.spec_driven_agent.providers.llm_client import (
    ClaudeLLMClient,
    MistralProvider,
    NebiusProvider,
    OpenAIProvider,
    UsageTracker,
    _is_auth_error,
    _is_rate_limit,
    _is_retryable,
    _openai_messages_to_api,
    _openai_stop_reason,
)

_REQUEST = httpx.Request("POST", "https://api.example.invalid/v1/chat/completions")


def _status_error(cls, status: int, message: str):
    """An SDK status error, worded the way the SDK words it."""
    if not message.startswith("Error code:"):
        message = f"Error code: {status} - {message}"
    return cls(message, response=httpx.Response(status, request=_REQUEST), body=None)


TOKENS_400 = ("Error code: 400 - {'error': {'message': 'This model's maximum context "
              "length is 128000 tokens. However, you requested 140133 tokens.'}}")
TOO_LONG_400 = ("Error code: 400 - {'type': 'error', 'error': {'type': "
                "'invalid_request_error', 'message': 'prompt is too long: 205000 tokens "
                "> 200000 maximum'}}")


class TestStatusBeatsDigitsInText:
    @pytest.mark.parametrize("message", [TOKENS_400, TOO_LONG_400])
    def test_a_400_with_token_counts_is_neither_auth_nor_retryable(self, message):
        for error in (Exception(message),
                      _status_error(openai.BadRequestError, 400, message)):
            assert _is_auth_error(error) is False
            assert _is_retryable(error) is False
            assert _is_rate_limit(error) is False

    def test_real_auth_errors_are_still_auth(self):
        assert _is_auth_error(_status_error(openai.AuthenticationError, 401, "bad key"))
        assert _is_auth_error(_status_error(openai.PermissionDeniedError, 403, "denied"))
        assert _is_auth_error(Exception("Error code: 401 - Incorrect API key provided"))
        assert _is_auth_error(Exception("invalid x-api-key"))

    def test_real_transient_statuses_are_still_retryable(self):
        assert _is_retryable(_status_error(openai.InternalServerError, 503, "down"))
        assert _is_retryable(Exception("Error code: 502 - bad gateway"))
        assert _is_rate_limit(_status_error(openai.RateLimitError, 429, "slow down"))


class TestTransportErrors:
    def test_sdk_timeout_and_connection_errors_are_retryable(self):
        assert str(openai.APITimeoutError(request=_REQUEST)) == "Request timed out."
        assert _is_retryable(openai.APITimeoutError(request=_REQUEST))
        assert _is_retryable(openai.APIConnectionError(request=_REQUEST))
        anthropic = pytest.importorskip("anthropic")
        assert _is_retryable(anthropic.APITimeoutError(request=_REQUEST))


class _RaisingCompletions:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self._exc


class _FakeClient:
    def __init__(self, exc):
        self.chat = type("C", (), {})()
        self.chat.completions = _RaisingCompletions(exc)


def _provider(exc, monkeypatch):
    p = OpenAIProvider(api_key="sk-test", model="gpt-4o",
                       fallback=("http://127.0.0.1:9/v1", "tok", "qwen3-coder:30b"))
    p._client = _FakeClient(exc)
    sleeps: list = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    fallbacks: list = []
    monkeypatch.setattr(p, "_activate_fallback",
                        lambda err, past_endpoint=None: (fallbacks.append(err), False)[1])
    return p, sleeps, fallbacks


def _chat(p):
    return p.chat("sys", [{"role": "user", "content": "hi"}], tools=[])


def _each_attempt_takes(monkeypatch, seconds: float) -> None:
    """Every request lasts ``seconds`` on the provider's clock (read at its
    start and at its failure); sleeps keep whatever the test patched in."""
    ticks = itertools.count(step=seconds)
    monkeypatch.setattr(llm_client, "time", SimpleNamespace(
        sleep=llm_client.time.sleep, monotonic=lambda: next(ticks)))


class TestOpenAIChatPaths:
    def test_timeout_goes_straight_to_the_fallback(self, monkeypatch):
        """Each timeout already cost 300s; with a fallback, don't re-pay it."""
        p, sleeps, fallbacks = _provider(openai.APITimeoutError(request=_REQUEST), monkeypatch)
        _each_attempt_takes(monkeypatch, 300)
        with pytest.raises(UpstreamLLMError):
            _chat(p)
        assert p._client.chat.completions.calls == 1
        assert sleeps == []
        assert len(fallbacks) == 1

    @pytest.mark.parametrize("exc", [openai.APITimeoutError(request=_REQUEST),
                                     openai.APIConnectionError(request=_REQUEST)])
    def test_without_a_fallback_a_slow_transport_failure_is_retried_twice(self, monkeypatch, exc):
        p = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        p._client = _FakeClient(exc)
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
        _each_attempt_takes(monkeypatch, 300)
        with pytest.raises(UpstreamLLMError):
            _chat(p)
        assert p._client.chat.completions.calls == llm_client._MAX_TRANSPORT_RETRIES + 1 == 3

    def test_claude_timeout_is_retried_twice(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")
        client = ClaudeLLMClient(api_key="sk-ant-x")
        messages = _RaisingCompletions(anthropic.APITimeoutError(request=_REQUEST))
        client._client = type("C", (), {"messages": messages})()
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
        _each_attempt_takes(monkeypatch, 300)
        with pytest.raises(UpstreamLLMError):
            client.chat("sys", [{"role": "user", "content": "hi"}], tools=[])
        assert messages.calls == 3

    def test_a_fast_transport_failure_gets_the_full_retry_budget(self, monkeypatch):
        """A refused connection costs milliseconds, not a 300s timeout."""
        p, sleeps, fallbacks = _provider(openai.APIConnectionError(request=_REQUEST), monkeypatch)
        with pytest.raises(UpstreamLLMError):
            _chat(p)
        assert p._client.chat.completions.calls == p._max_retries + 1 == 5
        assert len(sleeps) == 4
        assert len(fallbacks) == 1

    def test_token_count_400_is_not_an_invalid_key(self, monkeypatch):
        p, sleeps, _ = _provider(_status_error(openai.BadRequestError, 400, TOKENS_400),
                                 monkeypatch)
        with pytest.raises(UpstreamLLMError):
            _chat(p)
        assert p._client.chat.completions.calls == 1
        assert sleeps == []

    def test_insufficient_quota_429_still_fails_fast(self, monkeypatch):
        quota = _status_error(openai.RateLimitError, 429,
                              "Error code: 429 - insufficient_quota: You exceeded "
                              "your current quota")
        p, sleeps, fallbacks = _provider(quota, monkeypatch)
        with pytest.raises(UpstreamLLMError):
            _chat(p)
        assert p._client.chat.completions.calls == 1
        assert sleeps == []
        assert len(fallbacks) == 1

    def test_real_401_is_still_an_invalid_key(self, monkeypatch):
        p, _, _ = _provider(_status_error(openai.AuthenticationError, 401, "bad"), monkeypatch)
        with pytest.raises(InvalidApiKeyError):
            _chat(p)


def _reply(model):
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        model=model, choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, prompt_tokens_details=None))


def _stream(model):
    delta = SimpleNamespace(content="ok", tool_calls=None)
    return iter([SimpleNamespace(model=model, usage=None,
                                 choices=[SimpleNamespace(delta=delta, finish_reason="stop")])])


def _gateway(monkeypatch, fail):
    """Every client the provider builds, its fallbacks' too, answers through
    ``fail(model)``: an error to raise, or None to reply. Returns the models
    requested, in order."""
    requested: list = []

    def create(**kwargs):
        requested.append(kwargs["model"])
        error = fail(kwargs["model"])
        if error is not None:
            raise error
        return _stream(kwargs["model"]) if kwargs.get("stream") else _reply(kwargs["model"])

    completions = SimpleNamespace(create=create)
    monkeypatch.setattr(llm_client, "openai_client",
                        lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    return requested


def _chained_provider():
    return OpenAIProvider(api_key="sk-test", model="primary-model",
                          base_url="https://gateway.example.invalid/v1",
                          fallback=[("https://gateway.example.invalid/v1", "tok", "alt-model"),
                                    ("http://127.0.0.1:9/v1", "tok2", "last-resort-model")])


def _once(error):
    """Fail the first request with ``error``, then answer."""
    seen: list = []
    return lambda model: (seen.append(model), error if len(seen) == 1 else None)[1]


class TestOneBlipDoesNotSwitchTheRun:
    """A run on a tier with a fallback chain stays on its model for the rest of
    the run once it switches, so one connection error must not switch it."""

    @pytest.mark.parametrize("error", [openai.APIConnectionError(request=_REQUEST),
                                       ConnectionError("Connection error.")])
    def test_one_connection_error_is_retried_on_the_primary(self, monkeypatch, error):
        requested = _gateway(monkeypatch, _once(error))
        p = _chained_provider()
        _chat(p)
        assert requested == ["primary-model", "primary-model"]
        assert p.model == "primary-model" and p.fallback_reason is None

    def test_one_connection_error_on_a_stream_is_retried_on_the_primary(self, monkeypatch):
        requested = _gateway(monkeypatch, _once(openai.APIConnectionError(request=_REQUEST)))
        p = _chained_provider()
        events = list(p.chat_stream("sys", [{"role": "user", "content": "hi"}], tools=[]))
        assert events[-1]["type"] == "message_done"
        assert requested == ["primary-model", "primary-model"]
        assert p.model == "primary-model"

    def test_a_timeout_that_took_the_full_sdk_timeout_still_moves_on(self, monkeypatch):
        requested = _gateway(monkeypatch, _once(openai.APITimeoutError(request=_REQUEST)))
        p = _chained_provider()
        _each_attempt_takes(monkeypatch, 300)
        _chat(p)
        assert requested == ["primary-model", "alt-model"]
        assert p.model == "alt-model"

    def test_an_endpoint_that_stays_down_still_moves_on(self, monkeypatch):
        requested = _gateway(monkeypatch, lambda model: (
            openai.APIConnectionError(request=_REQUEST) if model == "primary-model" else None))
        p = _chained_provider()
        _chat(p)
        assert requested == ["primary-model"] * 5 + ["alt-model"]
        assert p.model == "alt-model"


class TestPlanningCallLeavesThePrimaryAlone:
    """A model_override (planning) call that fails is the caller's to retry on
    the primary; it never switches the primary down the fallback chain."""

    @pytest.mark.parametrize("error", [
        _status_error(openai.NotFoundError, 404, "model_not_found: planner-model"),
        _status_error(openai.InternalServerError, 503, "overloaded"),
        _status_error(openai.AuthenticationError, 401, "bad key"),
        openai.APITimeoutError(request=_REQUEST),
    ])
    def test_a_failed_planning_call_raises_and_keeps_the_primary(self, monkeypatch, error):
        requested = _gateway(monkeypatch, lambda model: error if model == "planner-model" else None)
        p = _chained_provider()
        _each_attempt_takes(monkeypatch, 300)
        with pytest.raises((UpstreamLLMError, InvalidApiKeyError)):
            p.chat("sys", [{"role": "user", "content": "hi"}], tools=[],
                   model_override="planner-model")
        assert set(requested) == {"planner-model"}
        assert p.model == "primary-model" and p._fallback_step == 0
        assert p.fallback_reason is None
        _chat(p)
        assert requested[-1] == "primary-model"

    def test_a_rejected_tool_choice_names_the_model_that_rejected_it(self, monkeypatch):
        rejected = _status_error(openai.BadRequestError, 400,
                                 "tool_choice of type 'function' is not supported")
        _gateway(monkeypatch, lambda model: rejected if model == "planner-model" else None)
        p = _chained_provider()
        with pytest.raises(UpstreamLLMError, match="planner-model"):
            p.chat("sys", [{"role": "user", "content": "hi"}], tools=[],
                   model_override="planner-model")


class TestSdkRetriesDisabled:
    def test_every_client_has_sdk_retries_off(self):
        pytest.importorskip("anthropic")
        assert ClaudeLLMClient(api_key="sk-ant-x")._client.max_retries == 0
        assert MistralProvider(api_key="x")._client.max_retries == 0
        p = OpenAIProvider(api_key="sk-x",
                           fallback=("http://127.0.0.1:9/v1", "tok", "qwen3-coder:30b"))
        assert p._client.max_retries == 0
        assert p._activate_fallback(Exception("Error code: 503 - down")) is True
        assert p._client.max_retries == 0


class TestBilledRoutePricing:
    @pytest.mark.parametrize("model", [
        "codestral-2508", "devstral-medium-2507", "devstral-small-2507",
        "mistral-small-2506",
    ])
    def test_pinned_mistral_ids_are_not_free_on_mistrals_api(self, model):
        pricing = MistralProvider(api_key="x", model=model).usage.pricing
        assert pricing["input"] > 0 and pricing["output"] > 0

    def test_published_rate_is_used_for_a_pinned_id(self):
        pricing = MistralProvider(api_key="x", model="codestral-2508").usage.pricing
        assert pricing["input"] == pytest.approx(0.3)
        assert pricing["output"] == pytest.approx(0.9)

    def test_an_unpublished_billed_id_is_not_zero(self):
        pricing = MistralProvider(api_key="x", model="codestral-2699").usage.pricing
        assert pricing["input"] > 0

    def test_nebius_is_billed(self):
        assert NebiusProvider(api_key="x", model="deepseek-coder").usage.pricing["input"] > 0

    def test_a_self_hosted_endpoint_stays_free(self):
        p = OpenAIProvider(api_key="x", model="codestral-2508",
                           base_url="http://localhost:11434/v1")
        assert p.usage.pricing["input"] == 0

    def test_id_shape_rule_unchanged_without_a_route(self):
        assert UsageTracker("codestral:22b").pricing["input"] == 0


class _Usage:
    def __init__(self, out):
        self.input_tokens = 0
        self.output_tokens = out
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class TestFallbackPricing:
    def test_fallback_tokens_are_billed_at_the_fallback_rate(self, monkeypatch):
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: object())
        p = OpenAIProvider(api_key="sponsored", model="openai/gpt-5.5",
                           base_url="https://gateway.example.invalid/v1",
                           fallback=("http://127.0.0.1:9/v1", "tok", "qwen3-coder:30b"))
        p.usage.record(_Usage(1_000_000))
        spent = p.usage.estimated_cost
        assert spent > 0
        assert p._activate_fallback(Exception("Error code: 503 - down")) is True
        p.usage.record(_Usage(1_000_000))
        assert p.usage.estimated_cost == pytest.approx(spent)


class TestOpenAITranslation:
    def test_content_filter_is_a_refusal(self):
        assert _openai_stop_reason("content_filter", False) == "refusal"

    def test_user_text_blocks_are_sent_as_text(self):
        api = _openai_messages_to_api("sys", [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "Now fix the tests."},
        ]}])
        assert api[-1] == {"role": "user", "content": "Now fix the tests."}


@pytest.mark.parametrize("status", [529, 520, 522, 524])
def test_every_5xx_is_retryable(status):
    # With the SDKs' own retry off (max_retries=0), Anthropic's 529 overload and
    # a Cloudflare tunnel's 52x must still be retried and reach the fallback.
    from besser.spec_driven_agent.providers.llm_client import _is_retryable

    class _Err(Exception):
        status_code = status

    assert _is_retryable(_Err(f"Error code: {status}"))


def test_client_errors_stay_non_retryable():
    from besser.spec_driven_agent.providers.llm_client import _is_retryable

    class _Err(Exception):
        status_code = 400

    assert not _is_retryable(_Err("Error code: 400 - prompt is too long: 205000 tokens"))
