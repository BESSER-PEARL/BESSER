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
                        lambda err: (fallbacks.append(err), False)[1])
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

    def test_a_real_401_still_raises_invalid_key(self, monkeypatch):
        p, _, fallbacks = _provider(_status_error(openai.AuthenticationError, 401, "bad"), monkeypatch)
        with pytest.raises(InvalidApiKeyError):
            p.chat("sys", _MESSAGES, tools=[])
        assert fallbacks == []


def test_the_fallback_records_quota_exhausted_for_credits(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: object())
    p = OpenAIProvider(api_key="sk-test", model="moonshotai/Kimi-K3",
                       fallback=("http://127.0.0.1:9/v1", "tok", "qwen3-coder:30b"))
    assert p._activate_fallback(_status_error(openai.BadRequestError, 400, CREDITS_400))
    assert p.fallback_reason == "quota_exhausted"
    assert p.model == "qwen3-coder:30b"
