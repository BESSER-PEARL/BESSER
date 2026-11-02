"""Shared helpers for LLM integrations across BESSER utilities.

Centralised here so the LLM KG conversion (``besser.utilities.kg_llm_to_buml``)
and the KG-cleanup LLM flow (``besser.BUML.notations.kg_to_buml.llm_cleanup``)
share the same response-cleaning logic, and so every OpenAI chat request in the
codebase goes through one place that enforces a timeout and never echoes the
upstream response body to a caller.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

#: OpenAI chat-completions endpoint used by the helpers below.
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

#: Model used when a caller does not name one. Defined here, in the lowest layer
#: that talks to OpenAI, so the notation code, the LLM KG converter and the
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


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*\s*\n(.*?)\n?```", re.DOTALL)


def _closes(text: str, start: int) -> bool:
    """Whether the bracket at ``start`` is closed before the text ends
    (JSON strings respected). An unclosed one is a reply cut off mid-value."""
    depth, in_string, escaped = 0, False, False
    for char in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return True
    return False


def extract_json(text: str, kind: type = dict) -> Optional[Any]:
    """The JSON value of type ``kind`` (``dict`` or ``list``) an LLM reply holds.

    The whole reply or a fenced block is the answer when it parses, even an
    empty one. Otherwise only top-level values in the prose are considered
    (each is decoded with ``JSONDecoder.raw_decode`` and skipped past, so
    the values nested inside it are never candidates), empty ones are not an
    answer, and the longest wins over a citation such as ``[1]``. A value
    that is still open when the text ends means the reply was truncated:
    ``None``, never a fragment of it.
    """
    if not isinstance(text, str):
        return None
    for block in [text, *_FENCE_RE.findall(text)]:
        try:
            value = json.loads(block)
        except ValueError:
            continue
        return value if isinstance(value, kind) else None
    decoder = json.JSONDecoder()
    best, best_span, index = None, 0, 0
    while True:
        index = min((i for i in (text.find("{", index), text.find("[", index)) if i != -1), default=-1)
        if index == -1:
            return best
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            if not _closes(text, index):
                return None
            index += 1
            continue
        if isinstance(value, kind) and value and end - index > best_span:
            best, best_span = value, end - index
        index = end


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def post_openai_chat(
    api_key: str,
    payload: dict,
    *,
    http_post: Optional[Callable[..., Any]] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    endpoint: str = OPENAI_CHAT_ENDPOINT,
) -> str:
    """POST a chat-completions ``payload`` and return the assistant content.

    The transport half of every OpenAI chat call in the codebase: it applies a
    timeout, turns every failure into :class:`LLMUpstreamError`, and never puts
    the provider's response body in an exception message. The payload is the
    caller's, untouched, because it differs by use (a text prompt, or a vision
    request with image parts and ``max_tokens``).

    Args:
        api_key: Bearer token. It is stripped, and redacted from anything logged.
        payload: The complete request body.
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
    """Send one system+user text prompt and return the assistant content.

    A thin payload builder over :func:`post_openai_chat`; see there for the
    error contract.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    return post_openai_chat(
        api_key, payload, http_post=http_post, timeout=timeout, endpoint=endpoint,
    )
