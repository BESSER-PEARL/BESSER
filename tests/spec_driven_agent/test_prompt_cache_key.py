"""A per-run prompt cache key goes to OpenAI's own API only.

OpenAI routes requests that share a ``prompt_cache_key`` to the same prompt
cache, so a run's turns reuse its cached prefix instead of landing on a server
that never saw it. The field is OpenAI's: a strict OpenAI-compatible gateway
may answer an unknown body field with HTTP 400, so every other endpoint gets
requests without it.
"""

import pytest

from besser.spec_driven_agent.providers.llm_client import OpenAIProvider
from tests.spec_driven_agent.run_transport import FakeOpenAI, ScriptedModel


def _client(monkeypatch, base_url):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = OpenAIProvider(api_key="sk-test", model="gpt-6-sol", base_url=base_url, max_retries=0)
    fake = FakeOpenAI(ScriptedModel(".", [], []), "gpt-6-sol")
    client._client = fake
    return client, fake


def _keys(fake) -> list:
    return [(request.kwargs.get("extra_body") or {}).get("prompt_cache_key") for request in fake.requests]


@pytest.mark.parametrize("base_url", [None, "https://api.openai.com/v1"])
def test_every_request_of_a_run_carries_one_cache_key_on_openai(monkeypatch, base_url):
    client, fake = _client(monkeypatch, base_url)
    client.chat(system="You build apps.", messages=[{"role": "user", "content": "hi"}], tools=[])
    list(client.chat_stream(system="You build apps.", messages=[{"role": "user", "content": "hi"}], tools=[]))
    keys = _keys(fake)
    assert len(keys) == 2 and keys[0] and keys[0] == keys[1]

    other, other_fake = _client(monkeypatch, base_url)
    other.chat(system="You build apps.", messages=[{"role": "user", "content": "hi"}], tools=[])
    assert _keys(other_fake)[0] != keys[0], "two runs share one cache key"


@pytest.mark.parametrize("base_url", ["https://gateway.example.test/v1", "http://localhost:11434/v1"])
def test_no_other_endpoint_gets_the_field(monkeypatch, base_url):
    client, fake = _client(monkeypatch, base_url)
    client.chat(system="You build apps.", messages=[{"role": "user", "content": "hi"}], tools=[])
    list(client.chat_stream(system="You build apps.", messages=[{"role": "user", "content": "hi"}], tools=[]))
    assert all("extra_body" not in request.kwargs for request in fake.requests)
