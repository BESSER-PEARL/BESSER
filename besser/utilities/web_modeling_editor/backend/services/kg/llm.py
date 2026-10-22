"""Reaching the user's own LLM from the KG cleanup endpoints.

``kg_to_buml.llm_cleanup`` asks for one thing: a ``(system, user) -> text``
callable (its ``Completer``). This module builds it for the provider the user's
key belongs to.

* **OpenAI** keeps the dedicated chat transport in ``besser.utilities.llm_utils``
  (timeout, every failure raised as ``LLMUpstreamError``).
* **Anthropic, Mistral and Nebius** go through the Spec-Driven Agent's provider
  clients (``create_llm_client``), so their SDK handling and model quirks are
  shared rather than re-implemented. Their retry loop is switched off
  (``max_retries=0``): a retry with backoff could otherwise start after the
  request's deadline, in a worker thread nobody waits for any more.

Whatever the provider, a failure leaves here as ``LLMUpstreamError`` with a fixed
message. The provider clients build their exception text from the SDK's, which
can echo request fragments such as part of the key, so that text is logged
(key redacted, truncated) and never returned to the client.

Every KG LLM request has **one** time budget (:class:`RequestDeadline`), however
many model calls it makes: each call is a single attempt whose timeout is the
time that is left, a call that would start with too little left is refused, and
:func:`run_with_deadline` stops waiting when the budget is spent. A model call
already in flight then ends by its own timeout, which never reaches past the
deadline, and nothing is attempted after it.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, List, TypeVar

from besser.BUML.notations.kg_to_buml import Completer
from besser.utilities.llm_utils import LLMUpstreamError, call_openai_chat_json
from besser.utilities.web_modeling_editor.backend.models import KGLLMRequest
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    UpstreamServiceError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Matches the OpenAI path's ``max_completion_tokens`` for these prompts.
_MAX_OUTPUT_TOKENS = 4000
_LOGGED_CHARS = 500
# A model call is not started with less time than this left in the budget.
_MIN_CALL_SECONDS = 5.0
_PROVIDER_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "mistral": "Mistral",
    "nebius": "Nebius",
}

_T = TypeVar("_T")


@dataclass(frozen=True)
class RequestDeadline:
    """The single time budget of one KG LLM request."""

    budget_seconds: float
    expires_at: float

    @classmethod
    def start(cls, budget_seconds: float) -> "RequestDeadline":
        """A deadline ``budget_seconds`` from now."""
        return cls(budget_seconds=budget_seconds, expires_at=time.monotonic() + budget_seconds)

    def exceeded_message(self) -> str:
        """The user-facing message for a request that ran out of time."""
        return f"The LLM did not finish within the {self.budget_seconds:g}-second limit for this request."

    def remaining(self) -> float:
        """Seconds left for the next model call.

        Raises:
            LLMUpstreamError: too little time is left to start another call.
        """
        left = self.expires_at - time.monotonic()
        if left < _MIN_CALL_SECONDS:
            raise LLMUpstreamError(self.exceeded_message())
        return left


async def run_with_deadline(deadline: RequestDeadline, func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run blocking ``func`` off the event loop and stop waiting at the deadline.

    The worker thread cannot be cancelled. It finishes its current model call
    (whose own timeout is at most the budget) and then fails fast, because every
    further call made through a completer built for ``deadline`` is refused.

    Raises:
        UpstreamServiceError: the deadline passed first (HTTP 502).
    """
    timeout = max(deadline.expires_at - time.monotonic(), 0.0)
    try:
        return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise UpstreamServiceError(deadline.exceeded_message()) from exc


def build_completer(request: KGLLMRequest, deadline: RequestDeadline) -> Completer:
    """Return the ``(system, user) -> text`` callable for ``request``'s provider and key.

    Each call it makes is given the time left in ``deadline`` as its timeout.

    Raises:
        ValidationError: the key is blank (HTTP 400).
    """
    api_key = request.resolved_api_key()
    if not api_key.strip():
        raise ValidationError("An API key is required.")
    model = request.resolved_model()

    if request.provider == "openai":
        def complete_openai(system_msg: str, user_msg: str) -> str:
            return call_openai_chat_json(
                api_key, model, system_msg, user_msg,
                max_completion_tokens=_MAX_OUTPUT_TOKENS, timeout=deadline.remaining(),
            )

        return complete_openai
    return _provider_completer(request.provider, api_key.strip(), model, deadline)


def _provider_completer(provider: str, api_key: str, model: str, deadline: RequestDeadline) -> Completer:
    # Imported here: the provider SDKs are heavy, and OpenAI never needs them.
    from besser.spec_driven_agent.providers.llm_client import create_llm_client

    name = _PROVIDER_NAMES[provider]

    def new_client(timeout: float) -> Any:
        # One client per call: the SDK timeout is fixed when the client is built,
        # and each call may only use what is left of the request's budget.
        try:
            return create_llm_client(
                provider=provider, api_key=api_key, model=model,
                max_tokens=_MAX_OUTPUT_TOKENS, timeout=timeout,
                # One attempt per call: the deadline, not the client, bounds the request.
                max_retries=0,
            )
        except ImportError as exc:
            # The server lacks the provider's SDK: a deployment problem, not the user's.
            logger.warning("%s client unavailable: %s", name, exc)
            raise LLMUpstreamError(f"{name} is not available on this server.") from exc
        except ValueError as exc:
            logger.warning("%s client could not be created: %s", name, _redact(str(exc), api_key))
            raise LLMUpstreamError(f"{name} could not be reached with this key.") from exc

    def complete(system_msg: str, user_msg: str) -> str:
        client = new_client(deadline.remaining())
        try:
            result = client.chat(
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
            )
        except Exception as exc:  # noqa: BLE001 - every provider failure maps to the same 502
            logger.warning(
                "%s call failed (%s): %s", name, type(exc).__name__, _redact(str(exc), api_key),
            )
            raise LLMUpstreamError(f"{name} API call failed.") from exc
        return _text_of(result.get("content") or [])

    return complete


def _text_of(blocks: List[object]) -> str:
    """The concatenated text blocks of a provider reply (tool blocks are ignored)."""
    return "".join(
        block.text for block in blocks
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    )


def _redact(text: str, secret: str) -> str:
    return (text.replace(secret, "***") if secret else text)[:_LOGGED_CHARS]
