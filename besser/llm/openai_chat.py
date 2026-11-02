"""OpenAI chat completions over HTTP, from a base install.

Every OpenAI chat request outside the Spec-Driven Agent goes through
:func:`post_openai_chat`: it enforces a timeout, never echoes the upstream
response body to a caller, and retries (when asked) with the policy every
LLM call in BESSER shares, :mod:`besser.llm.retry`. It uses ``requests`` (a base dependency) rather than the ``openai`` SDK, which
is the optional ``agents`` extra.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import requests

from besser.llm.errors import LLMUpstreamError
from besser.llm.retry import backoff_seconds, is_quota_exhausted, is_retryable_status
from besser.utilities.secret_redaction import redact_literal

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


def post_openai_chat(
    api_key: str,
    payload: dict,
    *,
    http_post: Optional[Callable[..., Any]] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    endpoint: str = OPENAI_CHAT_ENDPOINT,
    max_retries: int = 0,
) -> str:
    """POST a chat-completions ``payload`` and return the assistant content.

    The transport half of every OpenAI chat call outside the Spec-Driven Agent:
    it applies a timeout, turns every failure into :class:`LLMUpstreamError`,
    and never puts the provider's response body in an exception message. The
    payload is the caller's, untouched, because it differs by use (a text
    prompt, or a vision request with image parts and ``max_tokens``).

    Args:
        api_key: Bearer token. It is stripped, and redacted from anything logged.
        payload: The complete request body.
        http_post: Transport override for tests. ``None`` looks up
            :func:`requests.post` **at call time**, so monkeypatching it works.
        timeout: Seconds before each attempt is abandoned. Always applied.
        max_retries: Retries after the first attempt, on a connection error, a
            timeout or a retryable status (408, 409, 429, 5xx), waiting as
            :func:`besser.llm.retry.backoff_seconds` says (``Retry-After``
            honoured, capped). A spent quota (``insufficient_quota``) is not
            retried. ``0`` (the default) makes one attempt.

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

    for attempt in range(max_retries + 1):
        try:
            response = post(endpoint, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            if attempt < max_retries and isinstance(exc, (requests.ConnectionError, requests.Timeout)):
                time.sleep(backoff_seconds(exc, attempt))
                continue
            # The exception text can carry the URL; the class name is enough here.
            logger.warning("OpenAI request failed: %s", redact_literal(str(exc), key))
            raise LLMUpstreamError(f"OpenAI request failed ({type(exc).__name__}).") from exc
        status = getattr(response, "status_code", None)
        if getattr(response, "ok", False) or attempt == max_retries or not isinstance(status, int):
            break
        # The shared policy reads a provider error: status, headers and message.
        failure = requests.HTTPError(f"HTTP {status} {getattr(response, 'text', '')}", response=response)
        if not is_retryable_status(status) or is_quota_exhausted(failure):
            break
        time.sleep(backoff_seconds(failure, attempt))

    if not getattr(response, "ok", False):
        status = getattr(response, "status_code", None)
        body = redact_literal(str(getattr(response, "text", "")), key)[:_LOGGED_BODY_CHARS]
        logger.warning("OpenAI API call failed: HTTP %s %s", status, body)
        raise LLMUpstreamError(f"OpenAI API call failed: HTTP {status}", status=status)

    try:
        response_json = response.json()
    except ValueError as exc:
        raise LLMUpstreamError("OpenAI response was not valid JSON.") from exc

    if isinstance(response_json, dict) and "error" in response_json:
        error = response_json.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        logger.warning("OpenAI API error: %s", redact_literal(str(message), key)[:_LOGGED_BODY_CHARS])
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
