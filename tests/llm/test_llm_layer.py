"""The shared LLM layer, ``besser.llm``: retries, SDK clients, old import paths.

The error contract of :func:`post_openai_chat` is pinned in
``tests/utilities/test_llm_utils.py``; this file covers what the layer added.
Retries run against a real local HTTP server, so the shared policy
(:mod:`besser.llm.retry`) is exercised, not mocked; only the waits are recorded.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import besser.llm as llm
from besser.llm import LLMUpstreamError, anthropic_client, openai_client, post_openai_chat


class _Server:
    """Answers each POST with the next ``(status, body[, headers])`` and records the requests."""

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server's naming
                length = int(self.headers.get("Content-Length", 0))
                server.requests.append(json.loads(self.rfile.read(length)))
                status, body, *headers = server.answers.pop(0) if server.answers else (500, {})
                data = json.dumps(body).encode()
                self.send_response(status)
                for name, value in (headers[0] if headers else {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(request):
    srv = _Server(request.param)
    yield srv
    srv.close()


_OK = (200, {"choices": [{"message": {"content": "hello"}}]})
_BUSY = (503, {"error": {"message": "overloaded"}})


@pytest.mark.parametrize("server", [[_BUSY, _OK]], indirect=True)
def test_a_retryable_status_is_retried_when_retries_are_asked_for(server):
    assert post_openai_chat("k", {"model": "m"}, endpoint=server.url, max_retries=2) == "hello"
    assert server.requests == [{"model": "m"}, {"model": "m"}]


@pytest.mark.parametrize("server", [[_BUSY, _OK]], indirect=True)
def test_one_attempt_by_default(server):
    with pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat("k", {"model": "m"}, endpoint=server.url)
    assert caught.value.status == 503
    assert len(server.requests) == 1


@pytest.mark.parametrize("server", [[_BUSY, _BUSY, _BUSY, _OK]], indirect=True)
def test_exhausted_retries_surface_the_last_status(server):
    with pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat("k", {"model": "m"}, endpoint=server.url, max_retries=2)
    assert caught.value.status == 503
    assert len(server.requests) == 3


@pytest.mark.parametrize("server", [[(401, {"error": {"message": "bad key"}}), _OK]], indirect=True)
def test_a_client_error_is_not_retried(server):
    with pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat("k", {"model": "m"}, endpoint=server.url, max_retries=2)
    assert caught.value.status == 401
    assert len(server.requests) == 1


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    return waits


_SLOW_DOWN = (429, {"error": {"message": "slow down"}}, {"Retry-After": "120"})


@pytest.mark.parametrize("server", [[_BUSY, _SLOW_DOWN, _OK]], indirect=True)
def test_retries_wait_as_the_shared_policy_says(server, sleeps):
    # The first retry waits too, and a server's Retry-After is capped.
    assert post_openai_chat("k", {"model": "m"}, endpoint=server.url, max_retries=2) == "hello"
    assert sleeps == [0.5, 30.0]


_NO_QUOTA = (429, {"error": {"message": "You exceeded your current quota.", "code": "insufficient_quota"}})


@pytest.mark.parametrize("server", [[_NO_QUOTA, _OK]], indirect=True)
def test_a_spent_quota_is_not_retried(server, sleeps):
    with pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat("k", {"model": "m"}, endpoint=server.url, max_retries=2)
    assert caught.value.status == 429
    assert len(server.requests) == 1
    assert sleeps == []


def test_the_agent_client_uses_the_same_policy():
    from besser.llm import retry
    from besser.spec_driven_agent.providers import llm_client

    assert llm_client._backoff_seconds is retry.backoff_seconds
    assert llm_client._is_quota_exhausted is retry.is_quota_exhausted
    assert llm_client._status_code is retry.status_code


def test_a_connection_failure_after_retries_is_an_upstream_error():
    srv = _Server([])
    url = srv.url
    srv.close()  # nothing listens on the port any more
    with pytest.raises(LLMUpstreamError, match="OpenAI request failed"):
        post_openai_chat("k", {}, endpoint=url, max_retries=1)


def _fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_sdk_clients_get_the_callers_arguments_unchanged(monkeypatch):
    seen = {}
    _fake_module(monkeypatch, "openai", OpenAI=lambda **kw: seen.setdefault("openai", kw))
    _fake_module(monkeypatch, "anthropic", Anthropic=lambda **kw: seen.setdefault("anthropic", kw))

    openai_client(api_key="k", timeout=5, max_retries=0)
    anthropic_client(api_key="k", base_url="https://gw")

    assert seen == {
        "openai": {"api_key": "k", "timeout": 5, "max_retries": 0},
        "anthropic": {"api_key": "k", "base_url": "https://gw"},
    }


@pytest.mark.parametrize("build, package", [(openai_client, "openai"), (anthropic_client, "anthropic")])
def test_a_missing_sdk_names_the_extra(monkeypatch, build, package):
    monkeypatch.setitem(sys.modules, package, None)  # makes the import fail
    with pytest.raises(ImportError, match=r"pip install besser\[agents\]") as caught:
        build(api_key="k")
    assert f"pip install {package}" in str(caught.value)


_OLD_NAMES = [
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_REQUEST_TIMEOUT",
    "LLMUpstreamError",
    "OPENAI_CHAT_ENDPOINT",
    "call_openai_chat_json",
    "clean_json_response",
    "extract_json",
    "parse_json_safely",
    "post_openai_chat",
]


def test_the_old_import_path_re_exports_the_same_objects():
    from besser.utilities import llm_utils

    assert sorted(llm_utils.__all__) == _OLD_NAMES
    for name in _OLD_NAMES:
        assert getattr(llm_utils, name) is getattr(llm, name), name

