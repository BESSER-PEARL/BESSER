"""The mockup-to-B-UML steps reach OpenAI through ``besser.llm`` only.

They used to post to api.openai.com themselves (12 raw ``requests`` calls with
no timeout, retried by ``retrying``, which no install provided, so the packages
failed to import).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import besser.BUML as buml
from besser.BUML.notations.mockup_to_structural.utilities import openai_request
from besser.llm import LLMUpstreamError

_NETWORK_MODULES = ("requests", "httpx", "aiohttp", "urllib.request", "http.client", "openai", "anthropic")


def test_no_notation_module_talks_http_itself():
    offenders = []
    for path in Path(buml.__file__).parent.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"api\.openai\.com", source):
            offenders.append(f"{path}: api.openai.com")
        for node in ast.walk(ast.parse(source)):
            names = (
                [alias.name for alias in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            offenders += [
                f"{path}: {name}" for name in names
                if any(name == m or name.startswith(m + ".") for m in _NETWORK_MODULES)
            ]
    assert offenders == []


@pytest.fixture
def chat(monkeypatch):
    """Stands in for ``post_openai_chat``: records the calls, answers ``reply``."""
    state = SimpleNamespace(calls=[], reply="the reply")

    def fake(api_key, payload, **kwargs):
        state.calls.append((api_key, payload, kwargs))
        return state.reply() if callable(state.reply) else state.reply

    monkeypatch.setattr(openai_request, "post_openai_chat", fake)
    return state


def test_a_step_keeps_its_request_and_retries_three_attempts(chat):
    from besser.BUML.notations.mockup_to_buml.besser_integration.multiple_pages.gui_model_generation import (
        gpt4_refactor,
    )

    assert gpt4_refactor("sk-k", "code") == "the reply"
    [(api_key, payload, kwargs)] = chat.calls
    assert api_key == "sk-k"
    assert kwargs == {"max_retries": 2}
    assert {k: payload[k] for k in ("model", "max_tokens", "temperature")} == {
        "model": "gpt-4o", "max_tokens": 4096, "temperature": 0.0,
    }
    assert payload["messages"][1]["content"][1] == {"type": "text", "text": "code"}


def test_a_structural_step_goes_the_same_way(chat):
    from besser.BUML.notations.mockup_to_structural.besser_integration.one_page.plantuml_generation import (
        gpt4_self_improvement_plantuml_code,
    )

    assert gpt4_self_improvement_plantuml_code("sk-k", "@startuml\n@enduml") == "the reply"
    assert chat.calls[0][1]["model"] == "gpt-4o"


def test_an_empty_reply_is_none(chat):
    chat.reply = ""
    assert openai_request.request_completion("k", {}) is None


def test_an_http_error_is_none_as_before(chat):
    def fail():
        raise LLMUpstreamError("OpenAI API call failed: HTTP 401", status=401)

    chat.reply = fail
    assert openai_request.request_completion("k", {}) is None


def test_no_answer_at_all_still_raises(chat):
    def fail():
        raise LLMUpstreamError("OpenAI request failed (ConnectionError).")

    chat.reply = fail
    with pytest.raises(LLMUpstreamError):
        openai_request.request_completion("k", {})
