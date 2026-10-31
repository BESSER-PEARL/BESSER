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


# ---------------------------------------------------------------------------
# extract_json: one JSON extractor for LLM replies (review 2026-10-30, R4).
# The copies it replaced sliced from the first opening bracket to the LAST
# closing one, so a bracket in the prose around the JSON failed the parse.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind, expected", [
    ('[{"task": "Add search"}]', list, [{"task": "Add search"}]),
    ('```json\n[{"task": "Add search"}]\n```', list, [{"task": "Add search"}]),
    ('[{"task": "Add search"}]\n\nNote: optional items are marked [opt].', list, [{"task": "Add search"}]),
    ('Tasks [2 total]:\n[{"task": "A"}, {"task": "B"}]', list, [{"task": "A"}, {"task": "B"}]),
    ('{"requirements": [{"id": 1}]}\nUse {id} to reference one.', dict, {"requirements": [{"id": 1}]}),
    ('Format {id, text}:\n{"requirements": [{"id": 1}]}', dict, {"requirements": [{"id": 1}]}),
    ('{"a": "a } in a string"}', dict, {"a": "a } in a string"}),
])
def test_extract_json_ignores_brackets_in_surrounding_prose(text, kind, expected):
    from besser.utilities.llm_utils import extract_json
    assert extract_json(text, kind) == expected


@pytest.mark.parametrize("text, kind", [
    ("no json here", dict),
    ('[{"task": "Add search"},]', list),  # trailing comma: not JSON
    ('{"a": 1}', list),
    (None, dict),
])
def test_extract_json_returns_none_without_a_value_of_that_kind(text, kind):
    from besser.utilities.llm_utils import extract_json
    assert extract_json(text, kind) is None


def test_spec_driven_callers_use_the_shared_extractor():
    """The gap analyzer and the requirements ledger read prose replies the same way."""
    import types

    from besser.spec_driven_agent.planning.gap_analyzer import _parse_task_array
    from besser.spec_driven_agent.planning.requirements_ledger import _tool_input

    assert _parse_task_array('["A"]\nOptional items are marked [opt].') == ["A"]
    reply = {"content": [types.SimpleNamespace(type="text", text='{"r": [1]}\nUse {id}.')]}
    assert _tool_input(reply, "record_requirements") == {"r": [1]}


# Review of #705: a reply cut off by max_tokens returned an inner fragment of
# it, and a "[]" or a "[1]" citation in prose won over the real list.
TRUNCATED = ('{"agentLanguage": "fr", "presentation": {"style": "formal", "avatar": null}, '
             '"modality": {"voice": tr')


@pytest.mark.parametrize("text, kind, expected", [
    (TRUNCATED, dict, None),
    ('[{"task": "A"}, {"task": "B"}', list, None),
    ('Nothing would be [] if complete, but missing:\n```json\n["Add search"]\n```', list, ["Add search"]),
    ('Per requirement [1], tasks:\n["Add search", "Add export"]', list, ["Add search", "Add export"]),
    ("No tasks needed, the list is []. Done.", list, None),
    ("```json\n[]\n```", list, []),
    ("[]", list, []),
    ('{"outer": {"inner": 1}} and {"x": 2}', dict, {"outer": {"inner": 1}}),
])
def test_extract_json_top_level_values_only(text, kind, expected):
    from besser.utilities.llm_utils import extract_json
    assert extract_json(text, kind) == expected


def test_a_truncated_recommendation_is_not_presented_as_the_answer():
    from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_recommendation_utils import (
        extract_json_object,
    )
    with pytest.raises(ValueError):
        extract_json_object(TRUNCATED)
