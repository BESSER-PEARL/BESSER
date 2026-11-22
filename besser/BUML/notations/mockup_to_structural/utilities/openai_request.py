"""The OpenAI request every mockup-to-B-UML step makes, through :mod:`besser.llm`."""

from typing import Optional

from besser.llm import LLMUpstreamError, post_openai_chat

# Each step makes up to three attempts, as it did before.
_MAX_RETRIES = 2


def request_completion(api_key: str, payload: dict) -> Optional[str]:
    """The reply text of the chat-completions ``payload``, or ``None``.

    ``None`` when OpenAI answered with an HTTP error status or without content;
    the step's caller reports that and carries on. Any other failure (no answer,
    a timeout, an unreadable body) raises :class:`~besser.llm.errors.LLMUpstreamError`.
    """
    try:
        return post_openai_chat(api_key, payload, max_retries=_MAX_RETRIES) or None
    except LLMUpstreamError as exc:
        if exc.status is None:
            raise
        return None
