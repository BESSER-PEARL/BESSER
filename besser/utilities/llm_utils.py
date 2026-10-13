"""Shared helpers for LLM integrations across BESSER utilities.

Centralised here so both the deterministic ``kg_to_buml`` LLM call and
the new KG-cleanup LLM flow can reuse the same response-cleaning logic, and
so every OpenAI chat request in the codebase goes through one place that
enforces a timeout and never echoes the upstream response body to a caller.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

#: OpenAI chat-completions endpoint used by the helpers below.
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

#: Model used when a caller does not name one. Defined here, in the lowest layer
#: that talks to OpenAI, so the notation code, the legacy KG converter and the
#: backend all share one value and cannot drift apart.
DEFAULT_OPENAI_MODEL = "gpt-4o"

#: Seconds before an upstream request is abandoned. Callers that run inside a
#: web request should additionally keep the call off the event loop.
DEFAULT_REQUEST_TIMEOUT = 120

# How much of an upstream error body is written to the server log.
_LOGGED_BODY_CHARS = 500


class LLMUpstreamError(RuntimeError):
    """The LLM provider failed, or answered with something unusable.

    The message is safe to show to an end user: it names the failure and, for
    an HTTP error, the status code, but **never** the provider's response body
    (which can echo request fragments such as part of the API key). The body is
    logged on the server instead.

    Subclasses :class:`RuntimeError` so existing ``except RuntimeError``
    handlers keep working.
    """

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def clean_json_response(response: str) -> str:
    """Strip markdown fences from a JSON response produced by an LLM."""
    json_text = response.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    elif json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    return json_text.strip()


def parse_json_safely(json_text: str) -> Optional[Any]:
    """Parse JSON, returning ``None`` on decode errors instead of raising."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def call_openai_chat_json(
    api_key: str,
    model: str,
    system_msg: str,
    user_msg: str,
    *,
    http_post: Optional[Callable[..., Any]] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    endpoint: str = OPENAI_CHAT_ENDPOINT,
    temperature: float = 0.0,
    max_completion_tokens: int = 4000,
) -> str:
    """Send one chat request and return the assistant message content.

    Args:
        api_key: Bearer token. It is stripped, and redacted from anything logged.
        model: Model name.
        system_msg: System prompt.
        user_msg: User prompt.
        http_post: Transport override for tests. ``None`` looks up
            :func:`requests.post` **at call time**, so monkeypatching it works.
        timeout: Seconds before the request is abandoned. Always applied.

    Returns:
        The content string, which may be empty. Deciding whether an empty
        completion is an error is the caller's business.

    Raises:
        LLMUpstreamError: on a transport failure, a non-2xx status, a body that
            is not JSON, or a JSON ``error`` object. See the class docstring
            for what the message does and does not contain.
    """
    post = http_post if http_post is not None else requests.post
    key = api_key.strip()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }

    try:
        response = post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        # The exception text can carry the URL; the class name is enough here.
        logger.warning("OpenAI request failed: %s", _redact(str(exc), key))
        raise LLMUpstreamError(f"OpenAI request failed ({type(exc).__name__}).") from exc

    if not getattr(response, "ok", False):
        status = getattr(response, "status_code", None)
        body = _redact(str(getattr(response, "text", "")), key)[:_LOGGED_BODY_CHARS]
        logger.warning("OpenAI API call failed: HTTP %s %s", status, body)
        raise LLMUpstreamError(f"OpenAI API call failed: HTTP {status}", status=status)

    try:
        response_json = response.json()
    except ValueError as exc:
        raise LLMUpstreamError("OpenAI response was not valid JSON.") from exc

    if isinstance(response_json, dict) and "error" in response_json:
        error = response_json.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        logger.warning("OpenAI API error: %s", _redact(str(message), key)[:_LOGGED_BODY_CHARS])
        raise LLMUpstreamError("OpenAI API error.")

    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return content if isinstance(content, str) else ""
