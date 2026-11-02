"""BESSER's LLM access layer.

It works from a base install and imports none of its callers (only the
secret redaction leaf of :mod:`besser.utilities`), so the notations, the
utilities, the generators, the Spec-Driven Agent and the editor backend all
build on it (enforced by import-linter, see ``pyproject.toml``).

* :mod:`besser.llm.openai_chat`: OpenAI chat completions over HTTP (text, JSON
  and vision payloads), with a timeout, optional SDK-style retries, and errors
  that never carry the provider's response body.
* :mod:`besser.llm.sdk`: clients of the official ``openai`` and ``anthropic``
  SDKs, imported lazily (the ``agents`` extra).
* :mod:`besser.llm.replies`: reading the JSON out of a model's reply.
* :mod:`besser.llm.errors`: :class:`LLMUpstreamError`, what a failed call raises.
"""

from besser.llm.errors import LLMUpstreamError
from besser.llm.openai_chat import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    OPENAI_CHAT_ENDPOINT,
    call_openai_chat_json,
    post_openai_chat,
)
from besser.llm.replies import clean_json_response, extract_json, parse_json_safely
from besser.llm.sdk import anthropic_client, openai_client

__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_REQUEST_TIMEOUT",
    "LLMUpstreamError",
    "OPENAI_CHAT_ENDPOINT",
    "anthropic_client",
    "call_openai_chat_json",
    "clean_json_response",
    "extract_json",
    "openai_client",
    "parse_json_safely",
    "post_openai_chat",
]
