"""Clients of the official ``openai`` and ``anthropic`` SDKs.

The SDKs are the optional ``agents`` extra, so they are imported when a client
is built, not when this module is. The keyword arguments go to the SDK
unchanged: each caller keeps its own timeout, retry count and base URL.
"""

from __future__ import annotations

from typing import Any


def _missing(package: str) -> ImportError:
    return ImportError(
        f"The '{package}' package is required for this LLM call. Install it with: "
        f"pip install besser[agents] (or pip install {package})"
    )


def openai_client(**kwargs: Any) -> Any:
    """An ``openai.OpenAI`` client built with ``kwargs``.

    Raises:
        ImportError: the ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise _missing("openai") from None
    return OpenAI(**kwargs)


def anthropic_client(**kwargs: Any) -> Any:
    """An ``anthropic.Anthropic`` client built with ``kwargs``.

    Raises:
        ImportError: the ``anthropic`` package is not installed.
    """
    try:
        import anthropic
    except ImportError:
        raise _missing("anthropic") from None
    return anthropic.Anthropic(**kwargs)
