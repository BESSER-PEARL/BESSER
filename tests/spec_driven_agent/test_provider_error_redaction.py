"""A provider error that echoes the run's key never reaches a log line or an error message.

Gateways in front of OpenAI-compatible and Anthropic endpoints may echo the
request (headers included) in an error body, which the SDK puts in the
exception text. The agent's clients log that text on every retry and wrap it in
the ``UpstreamLLMError`` / ``InvalidApiKeyError`` the orchestrator logs again.
"""

import logging

import httpx
import openai
import pytest

from besser.spec_driven_agent.errors import InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.providers import llm_client
from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient, OpenAIProvider

# No known token prefix: only the literal match can catch it.
_KEY = "gw-key-0123456789abcdef0123456789"
_REQUEST = httpx.Request("POST", "https://gateway.example.invalid/v1/chat/completions")


def _echo(cls, status):
    return cls(f"Error code: {status} - upstream failed for request "
               f"{{'headers': {{'Authorization': 'Bearer {_KEY}'}}}}",
               response=httpx.Response(status, request=_REQUEST), body=None)


class _Raising:
    def __init__(self, exc):
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc

    def stream(self, **kwargs):
        raise self._exc


def _openai(exc):
    provider = OpenAIProvider(api_key=_KEY, model="gpt-4o")
    provider._client = type("C", (), {})()
    provider._client.chat = type("Chat", (), {})()
    provider._client.chat.completions = _Raising(exc)
    return provider


def _claude(exc):
    client = ClaudeLLMClient(api_key=_KEY)
    client._client = type("C", (), {"messages": _Raising(exc)})()
    return client


def _calls(provider):
    yield lambda: provider.chat("sys", [{"role": "user", "content": "hi"}], tools=[])
    yield lambda: list(provider.chat_stream("sys", [{"role": "user", "content": "hi"}], tools=[]))


@pytest.mark.parametrize("make", [_openai, _claude])
@pytest.mark.parametrize("status, error_type, raised", [
    (503, "InternalServerError", UpstreamLLMError),
    (401, "AuthenticationError", InvalidApiKeyError),
])
def test_the_key_is_redacted_from_logs_and_errors(make, status, error_type, raised,
                                                  monkeypatch, caplog):
    sdk = openai if make is _openai else pytest.importorskip("anthropic")
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    provider = make(_echo(getattr(sdk, error_type), status))
    caplog.set_level(logging.DEBUG, logger=llm_client.__name__)

    for call in _calls(provider):
        with pytest.raises(raised) as excinfo:
            call()
        assert _KEY not in str(excinfo.value)
        assert "Error code" in str(excinfo.value)  # the provider's reason is kept

    assert _KEY not in caplog.text
