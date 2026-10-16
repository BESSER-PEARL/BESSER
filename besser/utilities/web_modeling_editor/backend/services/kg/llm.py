"""Reaching the user's own LLM from the KG cleanup endpoints.

``kg_to_buml.llm_cleanup`` asks for one thing: a ``(system, user) -> text``
callable (its ``Completer``). This module builds it for the provider the user's
key belongs to.

* **OpenAI** keeps the dedicated chat transport in ``besser.utilities.llm_utils``
  (timeout, every failure raised as ``LLMUpstreamError``).
* **Anthropic, Mistral and Nebius** go through the Spec-Driven Agent's provider
  clients (``create_llm_client``), so their SDK handling, retries and model
  quirks are shared rather than re-implemented.

Whatever the provider, a failure leaves here as ``LLMUpstreamError`` with a fixed
message. The provider clients build their exception text from the SDK's, which
can echo request fragments such as part of the key, so that text is logged
(key redacted, truncated) and never returned to the client.
"""

import logging
from typing import List

from besser.BUML.notations.kg_to_buml import Completer
from besser.utilities.llm_utils import LLMUpstreamError, call_openai_chat_json
from besser.utilities.web_modeling_editor.backend.models import KGLLMRequest
from besser.utilities.web_modeling_editor.backend.services.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Matches the OpenAI path's ``max_completion_tokens`` for these prompts.
_MAX_OUTPUT_TOKENS = 4000
_LOGGED_CHARS = 500
_PROVIDER_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "mistral": "Mistral",
    "nebius": "Nebius",
}


def build_completer(request: KGLLMRequest) -> Completer:
    """Return the ``(system, user) -> text`` callable for ``request``'s provider and key.

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
                api_key, model, system_msg, user_msg, max_completion_tokens=_MAX_OUTPUT_TOKENS,
            )

        return complete_openai
    return _provider_completer(request.provider, api_key.strip(), model)


def _provider_completer(provider: str, api_key: str, model: str) -> Completer:
    # Imported here: the provider SDKs are heavy, and OpenAI never needs them.
    from besser.spec_driven_agent.providers.llm_client import create_llm_client

    name = _PROVIDER_NAMES[provider]
    try:
        client = create_llm_client(
            provider=provider, api_key=api_key, model=model, max_tokens=_MAX_OUTPUT_TOKENS,
        )
    except ImportError as exc:
        # The server lacks the provider's SDK: a deployment problem, not the user's.
        logger.warning("%s client unavailable: %s", name, exc)
        raise LLMUpstreamError(f"{name} is not available on this server.") from exc
    except ValueError as exc:
        logger.warning("%s client could not be created: %s", name, _redact(str(exc), api_key))
        raise LLMUpstreamError(f"{name} could not be reached with this key.") from exc

    def complete(system_msg: str, user_msg: str) -> str:
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
