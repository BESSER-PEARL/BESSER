"""
LLM client adapter for the BESSER augmented generator.

Wraps the Anthropic SDK for tool-use conversations with Claude.
The ``anthropic`` package is an optional dependency — importing this
module without it installed raises a clear error message.

Supports custom base URLs and auth tokens for enterprise/gateway setups::

    # PowerShell
    $env:ANTHROPIC_BASE_URL = "https://gateway.example.com"
    $env:ANTHROPIC_AUTH_TOKEN = "your-token"

    # Bash
    export ANTHROPIC_BASE_URL="https://gateway.example.com"
    export ANTHROPIC_AUTH_TOKEN="your-token"
"""

import os
from typing import Any


def _resolve_api_key(
    api_key: str | None = None,
    config: dict | None = None,
) -> str:
    """
    Resolve the Anthropic API key / auth token from multiple sources.

    Priority order:
    1. Direct ``api_key`` parameter
    2. Config dict keys: ``api_key``, ``anthropic_api_key``, ``auth_token``
    3. Environment variables: ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``

    Raises:
        ValueError: If no key/token is found.
    """
    if api_key:
        return api_key
    if config:
        for key in ("api_key", "anthropic_api_key", "auth_token",
                     "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            if config.get(key):
                return config[key]
    # Check environment variables
    for env_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(env_var)
        if value:
            return value
    raise ValueError(
        "No Anthropic API key found. Provide it as api_key parameter, "
        "in config dict, or set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN "
        "environment variable."
    )


def _resolve_base_url(
    base_url: str | None = None,
    config: dict | None = None,
) -> str | None:
    """
    Resolve a custom Anthropic base URL (for enterprise gateways).

    Priority order:
    1. Direct ``base_url`` parameter
    2. Config dict key: ``base_url``
    3. Environment variable: ``ANTHROPIC_BASE_URL``
    4. None (use Anthropic default: https://api.anthropic.com)
    """
    if base_url:
        return base_url
    if config:
        if config.get("base_url"):
            return config["base_url"]
    return os.environ.get("ANTHROPIC_BASE_URL")


class ClaudeLLMClient:
    """
    Anthropic Claude client for tool-use conversations.

    Supports custom base URLs for enterprise gateways. Set
    ``ANTHROPIC_BASE_URL`` env var or pass ``base_url`` parameter.

    Args:
        api_key: Anthropic API key or auth token.
        model: Claude model ID (default: claude-sonnet-4-20250514).
        max_tokens: Maximum tokens per response.
        base_url: Custom API base URL (for gateways/proxies).
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 16384

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for LLM generation. "
                "Install it with: pip install anthropic"
            ) from None

        # Build client kwargs
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if resolved_base:
            client_kwargs["base_url"] = resolved_base

        self._client = anthropic.Anthropic(**client_kwargs)
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """
        Send a message with tools and return the response.

        Args:
            system: System prompt with model context.
            messages: Conversation history.
            tools: Tool definitions (Anthropic format).

        Returns:
            Dict with ``stop_reason`` and ``content`` (list of blocks).

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            error_msg = str(e)
            raise RuntimeError(f"Claude API call failed: {error_msg}") from None

        return {
            "stop_reason": response.stop_reason,
            "content": response.content,
        }
