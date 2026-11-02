"""The error every LLM call in BESSER surfaces to its caller."""

from __future__ import annotations

from typing import Optional


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
