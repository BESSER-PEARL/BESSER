"""An exhausted account balance or an out-of-plan model must reach the fallback.

Measured on prod 2026-10-21: the Command Code account behind the free tier ran
out of credits. Every paid model answered ``400 "You have insufficient credits
to make this request"``, which is neither retryable nor "model unavailable", so
runs died mid-customization ("OpenAI API streaming failed") instead of moving
to the configured qwen fallback. Models outside the plan answered
``403 MODEL_NOT_IN_PLAN``, which the 403 rule read as a rejected API key.
"""

import httpx
import openai
import pytest

from besser.spec_driven_agent.errors import InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.providers import llm_client
from besser.spec_driven_agent.providers.llm_client import (
    OpenAIProvider,
    _is_auth_error,
    _is_quota_exhausted,
)

_REQUEST = httpx.Request("POST", "https://api.commandcode.ai/provider/v1/chat/completions")

CREDITS_400 = (
    "Error code: 400 - {'error': {'message': 'You have insufficient credits to make "
    "this request. Please purchase more credits to continue using the service.', "
    "'type': 'invalid_request_error', 'code': 'BAD_REQUEST'}}"
)
NOT_IN_PLAN_403 = (
    "Error code: 403 - {'error': {'message': 'MODEL_NOT_IN_PLAN: GPT-5.4 Mini available "
    "in Pro and above plans or extra on demand usage'}}"
)


def _status_error(cls, status: int, message: str):
    return cls(message, response=httpx.Response(status, request=_REQUEST), body=None)


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
    p = OpenAIProvider(api_key="sk-test", model="moonshotai/Kimi-K3",
                       fallback=("http://127.0.0.1:9/v1", "tok", "qwen3-coder:30b"))
    p._client = _FakeClient(exc)
    sleeps: list = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    fallbacks: list = []
    monkeypatch.setattr(p, "_activate_fallback",
                        lambda err, past_endpoint=None: (fallbacks.append(err), False)[1])
    return p, sleeps, fallbacks


_MESSAGES = [{"role": "user", "content": "hi"}]


class TestClassification:
    def test_insufficient_credits_is_an_exhausted_quota(self):
        assert _is_quota_exhausted(_status_error(openai.BadRequestError, 400, CREDITS_400))

    def test_model_not_in_plan_is_not_a_rejected_key(self):
        assert not _is_auth_error(_status_error(openai.PermissionDeniedError, 403, NOT_IN_PLAN_403))

    def test_a_plain_403_is_still_a_rejected_key(self):
        assert _is_auth_error(_status_error(openai.PermissionDeniedError, 403, "denied"))


class TestReachesTheFallback:
    @pytest.mark.parametrize("exc", [
        _status_error(openai.BadRequestError, 400, CREDITS_400),
        _status_error(openai.PermissionDeniedError, 403, NOT_IN_PLAN_403),
    ])
    def test_chat(self, monkeypatch, exc):
        p, sleeps, fallbacks = _provider(exc, monkeypatch)
        with pytest.raises(UpstreamLLMError):
            p.chat("sys", _MESSAGES, tools=[])
        assert p._client.chat.completions.calls == 1
        assert sleeps == []
        assert len(fallbacks) == 1

    @pytest.mark.parametrize("exc", [
        _status_error(openai.BadRequestError, 400, CREDITS_400),
        _status_error(openai.PermissionDeniedError, 403, NOT_IN_PLAN_403),
    ])
    def test_chat_stream(self, monkeypatch, exc):
        p, sleeps, fallbacks = _provider(exc, monkeypatch)
        with pytest.raises(UpstreamLLMError):
            list(p.chat_stream("sys", _MESSAGES, tools=[]))
        assert p._client.chat.completions.calls == 1
        assert sleeps == []
        assert len(fallbacks) == 1

    def test_a_users_rejected_key_raises_invalid_key(self, monkeypatch):
        """A user's own key has no fallback: a 401 says so instead of switching models."""
        p = OpenAIProvider(api_key="sk-user", model="gpt-5.6-terra")
        p._client = _FakeClient(_status_error(openai.AuthenticationError, 401, "bad"))
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
        with pytest.raises(InvalidApiKeyError):
            p.chat("sys", _MESSAGES, tools=[])


def test_the_fallback_records_quota_exhausted_for_credits(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: object())
    p = OpenAIProvider(api_key="sk-test", model="moonshotai/Kimi-K3",
                       fallback=("http://127.0.0.1:9/v1", "tok", "qwen3-coder:30b"))
    assert p._activate_fallback(_status_error(openai.BadRequestError, 400, CREDITS_400))
    assert p.fallback_reason == "quota_exhausted"
    assert p.model == "qwen3-coder:30b"


# Measured on experimental 2026-10-29: the free tier's Command Code token
# expired. Every call answered 401 "Your API key has expired", which raised
# InvalidApiKeyError before the fallback check, so runs failed although the
# self-hosted fallback was configured and up.
EXPIRED_401 = (
    "Error code: 401 - {'error': {'message': 'Your API key has expired. Log in again to get a "
    "new key.', 'type': 'authentication_error'}}"
)


@pytest.mark.parametrize("stream", [False, True])
def test_an_expired_server_key_falls_back_past_its_own_endpoint(monkeypatch, stream):
    primary = "https://api.commandcode.ai/provider/v1"
    p = OpenAIProvider(
        api_key="free", model="moonshotai/Kimi-K3", base_url=primary,
        fallback=[(primary, "same-token", "poolside/laguna-s-2.1-free"),
                  ("https://ollama.example/v1", "own-token", "qwen3-coder:30b")],
    )
    p._client = _FakeClient(_status_error(openai.AuthenticationError, 401, EXPIRED_401))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    switched = []

    def activate(err, past_endpoint=None):
        switched.append(past_endpoint)
        return OpenAIProvider._activate_fallback.__get__(p)(err, past_endpoint)

    monkeypatch.setattr(p, "_activate_fallback", activate)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeClient(RuntimeError("fallback reached")))

    with pytest.raises(Exception) as raised:
        list(p.chat_stream("sys", _MESSAGES, tools=[])) if stream else p.chat("sys", _MESSAGES, tools=[])

    assert not isinstance(raised.value, InvalidApiKeyError)
    assert switched[0] == primary
    # The alt model on the rejected endpoint is skipped: same token, same 401.
    assert p.model == "qwen3-coder:30b"
    assert p.fallback_reason == "key_rejected"


# Measured on prod 2026-10-30: Command Code ended the free period of two ``:free``
# models in the chain. Each answered this 403, which read as a rejected key and
# skipped every other model on the endpoint, so runs went to the self-hosted
# fallback although beam :free and laguna-s-2.1-free on the same key still served.
WINDOW_ENDED_403 = (
    "Error code: 403 - The free Ornith 1.5 122B window has ended. Run /model and pick "
    "Ornith 1.5 122B to keep going.\nhttps://commandcode.ai/models"
)


def test_an_ended_free_window_is_not_a_rejected_key():
    assert not _is_auth_error(_status_error(openai.PermissionDeniedError, 403, WINDOW_ENDED_403))


@pytest.mark.parametrize("stream", [False, True])
def test_an_ended_free_window_moves_to_the_next_model_on_the_same_endpoint(monkeypatch, stream):
    primary = "https://api.commandcode.ai/provider/v1"
    p = OpenAIProvider(
        api_key="free", model="ornith/ornith-1.5-122b:free", base_url=primary,
        fallback=[(primary, "same-token", "reflection/beam:free"),
                  ("https://ollama.example/v1", "own-token", "qwen3-coder:30b")],
    )
    p._client = _FakeClient(_status_error(openai.PermissionDeniedError, 403, WINDOW_ENDED_403))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeClient(RuntimeError("fallback reached")))

    with pytest.raises(Exception) as raised:
        list(p.chat_stream("sys", _MESSAGES, tools=[])) if stream else p.chat("sys", _MESSAGES, tools=[])

    assert not isinstance(raised.value, InvalidApiKeyError)
    assert p.model == "reflection/beam:free"
    assert p.fallback_reason == "unavailable"
