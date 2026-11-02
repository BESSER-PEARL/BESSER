"""OpenAI chat completions over HTTP, from a base install.

Every OpenAI chat request outside the Spec-Driven Agent goes through
:func:`post_openai_chat`: it enforces a timeout, never echoes the upstream
response body to a caller, and retries (when asked) with the policy of the
official SDKs, applied by ``urllib3``'s :class:`~urllib3.util.retry.Retry`.
It uses ``requests`` (a base dependency) rather than the ``openai`` SDK, which
is the optional ``agents`` extra.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from besser.llm.errors import LLMUpstreamError
from besser.utilities.secret_redaction import mask_secret

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

# What the openai SDK retries: connection errors and these statuses, with
# exponential backoff and Retry-After honoured.
_RETRY_STATUSES = (408, 409, 429, 500, 502, 503, 504)
_RETRY_BACKOFF_SECONDS = 0.5


def _retrying_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=_RETRY_BACKOFF_SECONDS,
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=None,  # POST included, as the SDK does
        raise_on_status=False,  # the last response is mapped to an error below
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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
        timeout: Seconds before the request is abandoned. Always applied.
        max_retries: Retries after the first attempt, on a connection error or
            a retryable status (408, 409, 429, 5xx). ``0`` (the default) makes
            one attempt. Ignored when ``http_post`` is given.

    Returns:
        The content string, which may be empty. Deciding whether an empty
        completion is an error is the caller's business.

    Raises:
        LLMUpstreamError: on a transport failure, a non-2xx status, a body that
            is not JSON, or a JSON ``error`` object. See the class docstring
            for what the message does and does not contain.
    """
    session = _retrying_session(max_retries) if http_post is None and max_retries > 0 else None
    if http_post is not None:
        post = http_post
    else:
        post = session.post if session is not None else requests.post
    key = api_key.strip()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    try:
        response = post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        # The exception text can carry the URL; the class name is enough here.
        logger.warning("OpenAI request failed: %s", mask_secret(str(exc), key))
        raise LLMUpstreamError(f"OpenAI request failed ({type(exc).__name__}).") from exc
    finally:
        if session is not None:
            session.close()

    if not getattr(response, "ok", False):
        status = getattr(response, "status_code", None)
        body = mask_secret(str(getattr(response, "text", "")), key)[:_LOGGED_BODY_CHARS]
        logger.warning("OpenAI API call failed: HTTP %s %s", status, body)
        raise LLMUpstreamError(f"OpenAI API call failed: HTTP {status}", status=status)

    try:
        response_json = response.json()
    except ValueError as exc:
        raise LLMUpstreamError("OpenAI response was not valid JSON.") from exc

    if isinstance(response_json, dict) and "error" in response_json:
        error = response_json.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        logger.warning("OpenAI API error: %s", mask_secret(str(message), key)[:_LOGGED_BODY_CHARS])
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
