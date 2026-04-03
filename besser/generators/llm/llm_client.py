"""
LLM client adapter for the BESSER augmented generator.

Features:
- **Provider abstraction** — ``LLMProvider`` interface with Anthropic and OpenAI backends
- Anthropic Claude API with tool-use support and prompt caching
- OpenAI-compatible API (GPT, etc.) with automatic tool format translation
- **Cost tracking** — tracks input/output/cache tokens and estimates USD cost
- **Retry with backoff** — retries on 429/5xx/timeouts (3 attempts, exponential backoff)
- Custom base URL support for enterprise gateways
- Streaming support for real-time text output
- **Factory function** — ``create_llm_client()`` to instantiate the right provider
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ======================================================================
# Cost tracking (inspired by claw-code/usage.rs)
# ======================================================================

# Pricing per million tokens (as of 2025)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "haiku": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.1},
    "sonnet": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3},
    "opus": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5},
    # OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0, "cache_write": 0, "cache_read": 0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cache_write": 0, "cache_read": 0},
    "gpt-5": {"input": 5.0, "output": 20.0, "cache_write": 0, "cache_read": 0},
    "o3": {"input": 10.0, "output": 40.0, "cache_write": 0, "cache_read": 0},
    "o3-mini": {"input": 1.1, "output": 4.4, "cache_write": 0, "cache_read": 0},
}


def _get_pricing(model_id: str) -> dict[str, float]:
    """Get pricing tier based on model ID."""
    model_lower = model_id.lower()
    # Anthropic tiers
    for tier in ("haiku", "sonnet", "opus"):
        if tier in model_lower:
            return _MODEL_PRICING[tier]
    # OpenAI exact matches (longest first to avoid partial matches)
    for key in ("gpt-4o-mini", "gpt-4o", "gpt-5", "o3-mini", "o3"):
        if key in model_lower:
            return _MODEL_PRICING[key]
    return _MODEL_PRICING["sonnet"]  # default


class UsageTracker:
    """Tracks token usage and estimated cost across all API calls."""

    def __init__(self, model: str):
        self.model = model
        self.pricing = _get_pricing(model)
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.api_calls = 0

    def record(self, usage) -> None:
        """Record usage from an API response."""
        if usage is None:
            return
        self.api_calls += 1
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_creation_tokens += cw
        self.cache_read_tokens += cr
        logger.debug("API call #%d: in=%d out=%d cache_w=%d cache_r=%d",
                     self.api_calls, inp, out, cw, cr)
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD."""
        p = self.pricing
        return (
            (self.input_tokens * p["input"] / 1_000_000)
            + (self.output_tokens * p["output"] / 1_000_000)
            + (self.cache_creation_tokens * p["cache_write"] / 1_000_000)
            + (self.cache_read_tokens * p["cache_read"] / 1_000_000)
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for the recipe."""
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
            "model": self.model,
        }

    def __str__(self) -> str:
        return (
            f"Tokens: {self.total_tokens:,} "
            f"(in={self.input_tokens:,}, out={self.output_tokens:,}, "
            f"cache_read={self.cache_read_tokens:,}, cache_write={self.cache_creation_tokens:,}) "
            f"| Cost: ${self.estimated_cost:.4f}"
        )


# ======================================================================
# API key / base URL resolution
# ======================================================================

def _resolve_api_key(
    api_key: str | None = None,
    config: dict | None = None,
) -> str:
    if api_key:
        return api_key
    if config:
        for key in ("api_key", "anthropic_api_key", "auth_token",
                     "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            if config.get(key):
                return config[key]
    for env_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(env_var)
        if value:
            return value
    raise ValueError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN."
    )


def _resolve_base_url(
    base_url: str | None = None,
    config: dict | None = None,
) -> str | None:
    if base_url:
        return base_url
    if config and config.get("base_url"):
        return config["base_url"]
    return os.environ.get("ANTHROPIC_BASE_URL")


# ======================================================================
# Provider interface
# ======================================================================

class LLMProvider(ABC):
    """
    Base interface for LLM providers.

    All providers must support synchronous chat with tool-use and
    streaming chat.  They must also expose a ``model`` property and a
    ``usage`` tracker so that the orchestrator can monitor cost.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model identifier."""

    @property
    @abstractmethod
    def usage(self) -> UsageTracker:
        """Return the usage tracker instance."""

    @abstractmethod
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        """
        Send a message with tools.

        Returns a dict with keys:
        - ``stop_reason``: ``"end_turn"`` or ``"tool_use"``
        - ``content``: list of content blocks (text or tool_use)
        """

    @abstractmethod
    def chat_stream(self, system: str, messages: list[dict], tools: list[dict]):
        """
        Stream a message with tools, yielding event dicts.

        Each event has a ``type`` key:
        - ``"text_delta"`` with ``"text"`` — incremental text
        - ``"message_done"`` with ``"stop_reason"`` and ``"content"``
        """


# ======================================================================
# Retry logic (inspired by claw-code/claw_provider.rs)
# ======================================================================

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_MAX_RETRIES = 2  # total 3 attempts
_INITIAL_BACKOFF = 0.3  # seconds
_MAX_BACKOFF = 3.0  # seconds


def _is_retryable(error: Exception) -> bool:
    """Check if an API error is worth retrying."""
    error_str = str(error)
    # Rate limit or server errors
    for code in ("429", "500", "502", "503", "504", "timeout", "connection"):
        if code in error_str.lower():
            return True
    # Auth errors are NOT retryable
    if "401" in error_str or "403" in error_str:
        return False
    return False


# ======================================================================
# Client
# ======================================================================

class ClaudeLLMClient(LLMProvider):
    """
    Anthropic Claude client with retry, cost tracking, and prompt caching.
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

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if resolved_base:
            client_kwargs["base_url"] = resolved_base

        self._client = anthropic.Anthropic(**client_kwargs)
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self._usage = UsageTracker(self._model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def usage(self) -> UsageTracker:
        return self._usage

    @usage.setter
    def usage(self, value: UsageTracker) -> None:
        self._usage = value

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Send a message with tools. Retries on transient errors."""
        last_error = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=_with_cache_control(system),
                    messages=messages,
                    tools=_with_tool_cache(tools),
                )
                # Track usage
                self.usage.record(response.usage)
                return {
                    "stop_reason": response.stop_reason,
                    "content": response.content,
                }
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    backoff = min(_INITIAL_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES + 1, backoff, e,
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"Claude API call failed: {e}") from None

        raise RuntimeError(f"Claude API call failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None

    def chat_stream(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ):
        """Stream a message with tools, yielding events."""
        last_error = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=_with_cache_control(system),
                    messages=messages,
                    tools=_with_tool_cache(tools),
                ) as stream:
                    for text in stream.text_stream:
                        yield {"type": "text_delta", "text": text}

                    response = stream.get_final_message()
                    self.usage.record(response.usage)
                    yield {
                        "type": "message_done",
                        "stop_reason": response.stop_reason,
                        "content": _clean_content_blocks(response.content),
                    }
                return  # Success — exit retry loop

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    backoff = min(_INITIAL_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
                    logger.warning("Stream failed (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"Claude API streaming failed: {e}") from None

        raise RuntimeError(f"Streaming failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None


# ======================================================================
# Helpers
# ======================================================================

def _clean_content_blocks(content: list) -> list:
    """Strip gateway-added metadata from content blocks."""
    import anthropic.types as at
    cleaned = []
    for block in content:
        if hasattr(block, "type"):
            if block.type == "text":
                cleaned.append(at.TextBlock(type="text", text=block.text))
            elif block.type == "tool_use":
                cleaned.append(at.ToolUseBlock(
                    type="tool_use", id=block.id,
                    name=block.name, input=block.input,
                ))
            else:
                cleaned.append(block)
        else:
            cleaned.append(block)
    return cleaned


def _with_cache_control(system: str) -> list[dict]:
    """Wrap system prompt for prompt caching."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _with_tool_cache(tools: list[dict]) -> list[dict]:
    """Add cache_control to last tool definition."""
    if not tools:
        return tools
    cached = [dict(t) for t in tools]
    cached[-1] = {**cached[-1], "cache_control": {"type": "ephemeral"}}
    return cached


# ======================================================================
# OpenAI tool format translation
# ======================================================================

def _openai_max_tokens_key(model: str) -> str:
    """Return the correct parameter name for max tokens.

    GPT-5+, o3, o1 models require ``max_completion_tokens``.
    Older models use ``max_tokens``.
    """
    m = model.lower()
    if any(k in m for k in ("gpt-5", "o3", "o1", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """
    Convert Anthropic tool definitions to OpenAI function-calling format.

    Anthropic format::

        {"name": "...", "description": "...", "input_schema": {...}}

    OpenAI format::

        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    converted = []
    for tool in tools:
        func: dict[str, Any] = {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        schema = tool.get("input_schema", {})
        if schema:
            func["parameters"] = schema
        converted.append({"type": "function", "function": func})
    return converted


def _openai_messages_to_api(system: str, messages: list[dict]) -> list[dict]:
    """
    Build the OpenAI messages list.

    The system prompt goes in ``messages[0]`` with ``role: "system"``.
    Anthropic-style tool_result blocks are converted to OpenAI ``role: "tool"``
    messages.
    """
    api_messages: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and isinstance(content, list):
            # Anthropic sends tool results as a list of dicts in a user message
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    })
                else:
                    api_messages.append({"role": "user", "content": str(block)})
        elif role == "assistant" and isinstance(content, list):
            # Convert Anthropic content blocks to OpenAI assistant message
            text_parts = []
            tool_calls = []
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        import json as _json
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": _json.dumps(block.input),
                            },
                        })
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        import json as _json
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": _json.dumps(block.get("input", {})),
                            },
                        })
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "\n".join(text_parts)
            else:
                assistant_msg["content"] = None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            api_messages.append(assistant_msg)
        else:
            api_messages.append({"role": role, "content": content})

    return api_messages


# ======================================================================
# Common content block types (provider-agnostic)
# ======================================================================

class _TextBlock:
    """Lightweight text content block for provider-agnostic responses."""

    __slots__ = ("type", "text")

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Lightweight tool-use content block for provider-agnostic responses."""

    __slots__ = ("type", "id", "name", "input")

    def __init__(self, tool_id: str, name: str, arguments: dict):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = arguments


def _openai_response_to_common(response) -> dict[str, Any]:
    """
    Convert an OpenAI chat completion response to the common format
    used by the orchestrator.

    Returns a dict with ``stop_reason`` and ``content`` list.
    """
    choice = response.choices[0]
    message = choice.message

    content: list[Any] = []

    # Text content
    if message.content:
        content.append(_TextBlock(text=message.content))

    # Tool calls
    if message.tool_calls:
        import json as _json
        for tc in message.tool_calls:
            try:
                arguments = _json.loads(tc.function.arguments)
            except (ValueError, TypeError):
                arguments = {}
            content.append(_ToolUseBlock(
                tool_id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            ))

    # Map OpenAI finish_reason to our stop_reason
    finish_reason = choice.finish_reason
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "stop":
        stop_reason = "end_turn"
    else:
        stop_reason = finish_reason or "end_turn"

    return {"stop_reason": stop_reason, "content": content}


# ======================================================================
# OpenAI provider
# ======================================================================

class _OpenAIUsageAdapter:
    """Adapter so ``UsageTracker.record()`` works with OpenAI usage objects."""

    def __init__(self, usage):
        self.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        # OpenAI does not report cache tokens in the same way
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible provider (GPT, o3, etc.).

    Wraps the ``openai`` package (lazy-imported). Automatically translates
    tool definitions from Anthropic format to OpenAI function-calling format
    and converts responses back to the common format used by the orchestrator.

    Args:
        api_key: OpenAI API key (or set ``OPENAI_API_KEY`` env var).
        model: Model identifier (default: ``gpt-4o``).
        max_tokens: Maximum output tokens per response.
        base_url: Custom API base URL for proxies/gateways.
    """

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_MAX_TOKENS = 16384

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from None

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base:
            client_kwargs["base_url"] = resolved_base

        self._client = OpenAI(**client_kwargs)
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self._usage = UsageTracker(self._model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def usage(self) -> UsageTracker:
        return self._usage

    @usage.setter
    def usage(self, value: UsageTracker) -> None:
        self._usage = value

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Send a message with tools. Retries on transient errors."""
        last_error = None
        api_messages = _openai_messages_to_api(system, messages)
        openai_tools = _anthropic_tools_to_openai(tools) if tools else None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                # Newer models (GPT-5+, o3+) use max_completion_tokens
                # instead of max_tokens
                tok_key = _openai_max_tokens_key(self._model)
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    tok_key: self._max_tokens,
                    "messages": api_messages,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools

                response = self._client.chat.completions.create(**kwargs)

                # Track usage
                if response.usage:
                    self._usage.record(_OpenAIUsageAdapter(response.usage))

                return _openai_response_to_common(response)

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    backoff = min(_INITIAL_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
                    logger.warning(
                        "OpenAI API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES + 1, backoff, e,
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"OpenAI API call failed: {e}") from None

        raise RuntimeError(f"OpenAI API call failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None

    def chat_stream(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ):
        """Stream a message with tools, yielding events."""
        last_error = None
        api_messages = _openai_messages_to_api(system, messages)
        openai_tools = _anthropic_tools_to_openai(tools) if tools else None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                tok_key = _openai_max_tokens_key(self._model)
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    tok_key: self._max_tokens,
                    "messages": api_messages,
                    "stream": True,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools

                collected_text = ""
                tool_calls_accum: dict[int, dict] = {}
                finish_reason = None

                stream = self._client.chat.completions.create(**kwargs)
                for chunk in stream:
                    if not chunk.choices:
                        # Usage-only chunk at end of stream
                        if chunk.usage:
                            self._usage.record(_OpenAIUsageAdapter(chunk.usage))
                        continue

                    delta = chunk.choices[0].delta
                    fr = chunk.choices[0].finish_reason

                    if fr:
                        finish_reason = fr

                    # Text content
                    if delta and delta.content:
                        collected_text += delta.content
                        yield {"type": "text_delta", "text": delta.content}

                    # Accumulate tool calls across chunks
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_accum:
                                tool_calls_accum[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.id:
                                tool_calls_accum[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_accum[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_accum[idx]["arguments"] += tc_delta.function.arguments

                # Build final content blocks
                content: list[Any] = []
                if collected_text:
                    content.append(_TextBlock(text=collected_text))
                for _idx in sorted(tool_calls_accum.keys()):
                    tc_data = tool_calls_accum[_idx]
                    import json as _json
                    try:
                        arguments = _json.loads(tc_data["arguments"])
                    except (ValueError, TypeError):
                        arguments = {}
                    content.append(_ToolUseBlock(
                        tool_id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments,
                    ))

                # Map finish_reason
                if finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "stop":
                    stop_reason = "end_turn"
                else:
                    stop_reason = finish_reason or "end_turn"

                yield {
                    "type": "message_done",
                    "stop_reason": stop_reason,
                    "content": content,
                }
                return  # Success — exit retry loop

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    backoff = min(_INITIAL_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
                    logger.warning("OpenAI stream failed (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"OpenAI API streaming failed: {e}") from None

        raise RuntimeError(f"OpenAI streaming failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None


# ======================================================================
# OpenAI API key resolution
# ======================================================================

def _resolve_openai_api_key(
    api_key: str | None = None,
    config: dict | None = None,
) -> str:
    if api_key:
        return api_key
    if config:
        for key in ("api_key", "openai_api_key", "OPENAI_API_KEY"):
            if config.get(key):
                return config[key]
    for env_var in ("OPENAI_API_KEY",):
        value = os.environ.get(env_var)
        if value:
            return value
    raise ValueError(
        "No OpenAI API key found. Set OPENAI_API_KEY or pass api_key explicitly."
    )


# ======================================================================
# Factory
# ======================================================================

def create_llm_client(
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Create an LLM client for the specified provider.

    Args:
        provider: ``"anthropic"`` (default) or ``"openai"``.
        api_key: API key. If not provided, resolved from environment variables.
        model: Model identifier. Defaults to provider-specific default.
        base_url: Custom API base URL.
        **kwargs: Additional keyword arguments passed to the provider constructor
            (e.g. ``max_tokens``).

    Returns:
        An ``LLMProvider`` instance.

    Raises:
        ValueError: If the provider is unknown or the API key cannot be resolved.
        ImportError: If the required SDK package is not installed.
    """
    if provider == "anthropic":
        resolved_key = _resolve_api_key(api_key=api_key)
        resolved_base = _resolve_base_url(base_url=base_url)
        return ClaudeLLMClient(
            api_key=resolved_key, model=model,
            base_url=resolved_base, **kwargs,
        )
    elif provider == "openai":
        resolved_key = _resolve_openai_api_key(api_key=api_key)
        return OpenAIProvider(
            api_key=resolved_key, model=model,
            base_url=base_url, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Supported providers: 'anthropic', 'openai'."
        )
