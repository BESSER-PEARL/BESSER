"""Tests for the shared OpenAI transport and its callers."""

from __future__ import annotations

import json

import pytest
import requests

from besser.utilities.llm_utils import (
    DEFAULT_REQUEST_TIMEOUT,
    LLMUpstreamError,
    call_openai_chat_json,
    post_openai_chat,
)

KEY = "sk-live-SECRET123"


class _Resp:
    def __init__(self, payload, status=200, text=None):
        self._payload, self.status_code = payload, status
        self.ok = 200 <= status < 300
        self.text = text if text is not None else (
            "<not json>" if payload is ValueError else json.dumps(payload)
        )

    def json(self):
        if self._payload is ValueError:
            raise ValueError("not json")
        return self._payload


def _completion(content):
    return {"choices": [{"message": {"content": content}}]}


def test_returns_the_assistant_content_and_always_sends_a_timeout():
    seen = {}

    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen.update(headers=headers, json=json, timeout=timeout)
        return _Resp(_completion("hello"))

    assert call_openai_chat_json(f" {KEY} ", "m", "sys", "usr", http_post=post) == "hello"
    assert seen["timeout"] == DEFAULT_REQUEST_TIMEOUT
    assert seen["headers"]["Authorization"] == f"Bearer {KEY}"      # stripped
    assert seen["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["json"]["temperature"] == 0.0


def test_empty_or_missing_content_is_returned_not_raised():
    for body in ({"choices": []}, {"choices": [{}]}, {}, _completion(None)):
        assert post_openai_chat(KEY, {}, http_post=lambda *a, **k: _Resp(body)) == ""


@pytest.mark.parametrize(
    "response, expected",
    [
        (_Resp({}, 401, text=f"Incorrect API key provided: {KEY}"), "HTTP 401"),
        (_Resp({"error": {"message": f"bad {KEY}"}}), "OpenAI API error."),
        (_Resp(ValueError), "not valid JSON"),
    ],
)
def test_failures_raise_a_sanitised_error(response, expected, caplog):
    import logging

    with caplog.at_level(logging.WARNING), pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat(KEY, {}, http_post=lambda *a, **k: response)

    assert expected in str(caught.value)
    assert KEY not in str(caught.value)
    assert KEY not in caplog.text                      # redacted in the log too


def test_transport_errors_name_the_class_not_the_url():
    def post(*a, **k):
        raise requests.ConnectionError("HTTPSConnectionPool(host='api.openai.com') max retries")

    with pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat(KEY, {}, http_post=post)

    assert "ConnectionError" in str(caught.value)
    assert "api.openai.com" not in str(caught.value)


def test_status_is_available_to_callers():
    with pytest.raises(LLMUpstreamError) as caught:
        post_openai_chat(KEY, {}, http_post=lambda *a, **k: _Resp({}, 429))

    assert caught.value.status == 429


# --- image_to_plantuml: the third caller, which used to have no timeout ------------


def _png(tmp_path):
    path = tmp_path / "d.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return str(path)


def test_image_to_plantuml_keeps_its_vision_payload_and_now_has_a_timeout(tmp_path, monkeypatch):
    from besser.utilities.image_to_buml import image_to_plantuml

    seen = {}

    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen.update(json=json, timeout=timeout)
        return _Resp(_completion("noise @startuml\nclass A\n@enduml trailing"))

    monkeypatch.setattr("requests.post", post)

    assert image_to_plantuml(_png(tmp_path), KEY) == "@startuml\nclass A\n@enduml"
    assert seen["timeout"] == DEFAULT_REQUEST_TIMEOUT                 # was: none at all
    assert seen["json"]["max_tokens"] == 500                          # payload unchanged
    assert "max_completion_tokens" not in seen["json"]
    parts = seen["json"]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]


def test_image_to_plantuml_reports_and_returns_none_on_an_api_error(tmp_path, monkeypatch, capsys):
    from besser.utilities.image_to_buml import image_to_plantuml

    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _Resp({"error": {"message": f"Incorrect API key {KEY}"}}, 401),
    )

    assert image_to_plantuml(_png(tmp_path), KEY) is None
    out = capsys.readouterr().out
    assert "An error took place" in out
    assert KEY not in out
