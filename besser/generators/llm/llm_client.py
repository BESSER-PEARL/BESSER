"""
LLM client adapter for the BESSER augmented generator.

Features:
- **Provider abstraction** — ``LLMProvider`` interface with Anthropic, OpenAI, and Mistral backends
- Anthropic Claude API with tool-use support and prompt caching
- OpenAI-compatible API (GPT, Mistral, etc.) with automatic tool format translation
- **Cost tracking** — tracks input/output/cache tokens and estimates USD cost
- **Retry with backoff** — retries on 429/5xx/timeouts (up to 5 attempts; honors
  the server ``Retry-After`` header; a longer backoff ceiling for rate limits)
- Custom base URL support for enterprise gateways
- Streaming support for real-time text output
- **Factory function** — ``create_llm_client()`` to instantiate the right provider
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from besser.generators.llm.errors import InvalidApiKeyError, UpstreamLLMError

logger = logging.getLogger(__name__)

# Total per-request timeout passed to the provider SDKs. Without it both
# SDKs fall back to their own defaults (up to 10 minutes), so a hung
# provider call can stall a turn for that long with no signal. 300s is
# generous enough for a full 16k-token streaming turn while still
# bounding the worst case.
_DEFAULT_SDK_TIMEOUT_SECONDS = float(
    os.environ.get("BESSER_LLM_CALL_TIMEOUT_SECONDS", "300")
)


# ======================================================================
# Cost tracking (inspired by claw-code/usage.rs)
# ======================================================================

# Pricing per million tokens (refreshed early 2026).
# Sources: OpenAI public pricing page + OpenRouter / pricepertoken.com
# corroboration for gpt-5.5. Anthropic rates are stable from 2025.
# ``cache_read`` is OpenAI's prompt-caching discount (~10% of input);
# ``cache_write`` is 0 for OpenAI since they don't charge a write
# premium like Anthropic does.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "haiku":  {"input": 1.0,  "output": 5.0,  "cache_write": 1.25,  "cache_read": 0.1},
    "sonnet": {"input": 3.0,  "output": 15.0, "cache_write": 3.75,  "cache_read": 0.3},
    "opus":   {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5},
    # OpenAI — early-2026 public rates
    "gpt-4o-mini": {"input": 0.15, "output": 0.6,  "cache_write": 0, "cache_read": 0.075},
    "gpt-4o":      {"input": 2.5,  "output": 10.0, "cache_write": 0, "cache_read": 1.25},
    "gpt-5.5":     {"input": 5.0,  "output": 30.0, "cache_write": 0, "cache_read": 0.5},
    "gpt-5":       {"input": 1.25, "output": 10.0, "cache_write": 0, "cache_read": 0.125},
    "o3-mini":     {"input": 1.1,  "output": 4.4,  "cache_write": 0, "cache_read": 0},
    "o3":          {"input": 10.0, "output": 40.0, "cache_write": 0, "cache_read": 0},
    # Mistral — public rates as of mid-2026. ``mistral-large-latest``
    # currently bills at $2 / $6 per 1M (input / output). Mistral has no
    # separate cache-write premium; prompt-caching reads, when used, are
    # billed at the input rate (no published discount), so cache_read is
    # set equal to input rather than a fictional cheaper tier.
    # VERIFY: the ``-latest`` alias may route to Mistral Large 3
    # ($0.50 / $1.50). Re-check https://mistral.ai/pricing/ if exactness
    # matters; $2 / $6 is the conservative (higher) of the two.
    "mistral-large": {"input": 2.0, "output": 6.0, "cache_write": 0, "cache_read": 2.0},
}


def _get_pricing(model_id: str) -> dict[str, float]:
    """Get pricing tier based on model ID.

    Order matters in the OpenAI loop — longer / more-specific keys
    must come first so e.g. ``gpt-5.5`` doesn't get mis-matched to
    ``gpt-5`` (which is a substring) and silently billed at the
    cheaper tier. Falls back to ``gpt-4o`` so an unknown model gets a
    reasonable middle-tier rate.
    """
    model_lower = model_id.lower()
    # Anthropic tiers — unambiguous identifiers so order doesn't matter.
    for tier in ("haiku", "sonnet", "opus"):
        if tier in model_lower:
            return _MODEL_PRICING[tier]
    # Mistral tiers — match before the OpenAI fallback so a Mistral
    # model never silently bills at the ``gpt-4o`` default rate.
    for key in ("mistral-large",):
        if key in model_lower:
            return _MODEL_PRICING[key]
    # OpenAI: ``gpt-5.5`` must precede ``gpt-5`` (substring conflict).
    for key in ("gpt-4o-mini", "gpt-4o", "gpt-5.5", "gpt-5", "o3-mini", "o3"):
        if key in model_lower:
            return _MODEL_PRICING[key]
    return _MODEL_PRICING["gpt-4o"]  # default


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
        # Cost corrections that the token counters can't express: the
        # (negative) delta for calls billed at a cheaper override model,
        # and cost seeded from a checkpoint on resume. ``estimated_cost``
        # adds this on top of the token-derived amount.
        self._extra_cost_usd = 0.0
        # The actual model name OpenAI / Anthropic returned in the
        # response. May differ from ``self.model`` (the requested name)
        # if the provider aliases server-side. Captured once and held;
        # logged a warning if it changes mid-run.
        self.served_model: Optional[str] = None

    def set_served_model(self, name: str | None) -> None:
        """Record the model name the provider actually served.

        Idempotent: silently keeps the first non-empty value. Logs a
        warning if the served model changes mid-run (rare, but would
        indicate a routing change we want to know about).
        """
        if not name:
            return
        if self.served_model is None:
            self.served_model = name
            logger.info("UsageTracker: served model = %s (requested %s)", name, self.model)
        elif self.served_model != name:
            logger.warning(
                "UsageTracker: served model changed mid-run: %s → %s",
                self.served_model, name,
            )
            self.served_model = name

    def record(self, usage, model: str | None = None) -> None:
        """Record usage from an API response.

        Args:
            usage: The provider usage object.
            model: When the call was made with a per-call model override
                (e.g. a cheap planning model), pass that model so the
                call is billed at its own pricing instead of the
                tracker's primary tier. The token counters still
                accumulate normally; only the cost is corrected.
        """
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
        if model and model != self.model:
            override = _get_pricing(model)
            primary = self.pricing
            delta = (
                inp * (override["input"] - primary["input"])
                + out * (override["output"] - primary["output"])
                + cw * (override["cache_write"] - primary["cache_write"])
                + cr * (override["cache_read"] - primary["cache_read"])
            ) / 1_000_000
            self._extra_cost_usd += delta
        logger.debug("API call #%d: in=%d out=%d cache_w=%d cache_r=%d",
                     self.api_calls, inp, out, cw, cr)

    def seed_cost(self, usd: float) -> None:
        """Add already-spent cost (e.g. from a checkpoint on resume).

        Without this, a crash-resume cycle restarts the tracker at $0
        and the cost cap only covers post-resume spend — a resumed run
        could legally spend up to twice the user's ``max_cost_usd``.
        """
        if usd and usd > 0:
            self._extra_cost_usd += float(usd)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD (token-derived + corrections/seed)."""
        p = self.pricing
        return (
            (self.input_tokens * p["input"] / 1_000_000)
            + (self.output_tokens * p["output"] / 1_000_000)
            + (self.cache_creation_tokens * p["cache_write"] / 1_000_000)
            + (self.cache_read_tokens * p["cache_read"] / 1_000_000)
            + self._extra_cost_usd
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
    raise InvalidApiKeyError(
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

    @property
    def planning_model(self) -> str | None:
        """Cheap sibling model for one-shot planning calls, or ``None``.

        ``None`` means "use the primary model". Overridable per deploy
        via the ``BESSER_LLM_PLANNING_MODEL`` env var (set it to
        ``primary`` to disable cheap routing — important for gateways
        where the cheap sibling may not be available).
        """
        return None

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        force_tool: str | None = None,
        model_override: str | None = None,
    ) -> dict:
        """
        Send a message with tools.

        Args:
            force_tool: Name of a tool the model MUST call (translated
                to the provider's ``tool_choice`` mechanism). Makes
                planning calls structured-by-construction instead of
                free-text-parsed.
            model_override: Use this model for this single call instead
                of the configured one (billed at its own pricing).

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
_MAX_RETRIES = 4  # total 5 attempts (bumped from 3 so a brief provider
                  # rate-limit doesn't kill a multi-turn customization run)
_INITIAL_BACKOFF = 0.5  # seconds
_MAX_BACKOFF = 5.0  # seconds — cap for ordinary (5xx/timeout) retries
# Rate limits (429) need a longer ceiling: some providers (notably
# Mistral) throttle hard and only recover after several seconds. A
# sub-second backoff just burns the retry budget and aborts the run.
_RATELIMIT_MAX_BACKOFF = 30.0  # seconds


def _is_retryable(error: Exception) -> bool:
    """Check if an API error is worth retrying."""
    error_str = str(error).lower()
    # Auth errors are NEVER retryable (a longer wait won't fix a bad key).
    if "401" in error_str or "403" in error_str:
        return False
    # Rate limit or transient server/network errors
    for marker in ("429", "rate limit", "rate_limit", "rate-limited",
                   "ratelimited", "500", "502", "503", "504",
                   "timeout", "connection"):
        if marker in error_str:
            return True
    return False


def _is_rate_limit(error: Exception) -> bool:
    """True if the error is a provider rate-limit (HTTP 429)."""
    s = str(error).lower()
    return (
        "429" in s
        or "rate limit" in s
        or "rate_limit" in s
        or "rate-limited" in s
        or "ratelimited" in s
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Extract a server-provided ``Retry-After`` hint (seconds), if any.

    The OpenAI/Mistral and Anthropic SDK error objects carry the HTTP
    response, whose headers may include ``retry-after`` (delta-seconds).
    Returns ``None`` when no usable hint is present.
    """
    resp = getattr(error, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
        try:
            val = headers.get(key)
        except Exception:
            val = None
        if not val:
            continue
        try:
            secs = float(val)
        except (TypeError, ValueError):
            continue
        if secs >= 0:
            return secs
    return None


def _backoff_seconds(error: Exception, attempt: int) -> float:
    """Seconds to wait before the next retry.

    Honors a server-provided ``Retry-After`` (capped), otherwise uses
    exponential backoff. Rate-limit (429) responses get a higher ceiling
    so a throttled provider isn't abandoned after a sub-second wait.
    """
    retry_after = _retry_after_seconds(error)
    rate_limited = _is_rate_limit(error)
    if retry_after is not None:
        return min(retry_after, _RATELIMIT_MAX_BACKOFF)
    cap = _RATELIMIT_MAX_BACKOFF if rate_limited else _MAX_BACKOFF
    return min(_INITIAL_BACKOFF * (2 ** attempt), cap)


def _is_auth_error(error: Exception) -> bool:
    """Heuristic: did the provider reject our credentials?"""
    error_str = str(error).lower()
    return any(
        marker in error_str
        for marker in ("401", "403", "authentication", "invalid x-api-key",
                       "invalid api key", "incorrect api key")
    )


# ======================================================================
# Client
# ======================================================================

class ClaudeLLMClient(LLMProvider):
    """
    Anthropic Claude client with retry, cost tracking, and prompt caching.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 16384

    PLANNING_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for LLM generation. "
                "Install it with: pip install anthropic"
            ) from None

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout if timeout is not None else _DEFAULT_SDK_TIMEOUT_SECONDS,
        }
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

    @property
    def planning_model(self) -> str | None:
        env = os.environ.get("BESSER_LLM_PLANNING_MODEL")
        if env:
            return None if env.lower() == "primary" else env
        if "haiku" in self._model.lower():
            return None  # already on the cheap tier
        return self.PLANNING_MODEL

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        force_tool: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Send a message with tools. Retries on transient errors."""
        last_error = None
        effective_model = model_override or self._model

        for attempt in range(_MAX_RETRIES + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": effective_model,
                    "max_tokens": self._max_tokens,
                    "system": _with_cache_control(system),
                    "messages": messages,
                    "tools": _with_tool_cache(tools),
                }
                if force_tool:
                    request_kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
                response = self._client.messages.create(**request_kwargs)
                # Track usage (billed at the effective model's pricing)
                self.usage.record(response.usage, model=model_override)
                return {
                    "stop_reason": response.stop_reason,
                    "content": response.content,
                }
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    backoff = _backoff_seconds(e, attempt)
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES + 1, backoff, e,
                    )
                    time.sleep(backoff)
                    continue
                if _is_auth_error(e):
                    raise InvalidApiKeyError(f"Claude API rejected the key: {e}") from None
                raise UpstreamLLMError(f"Claude API call failed: {e}") from None

        raise UpstreamLLMError(f"Claude API call failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None

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
                    backoff = _backoff_seconds(e, attempt)
                    logger.warning("Stream failed (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(backoff)
                    continue
                if _is_auth_error(e):
                    raise InvalidApiKeyError(f"Claude API rejected the key: {e}") from None
                raise UpstreamLLMError(f"Claude API streaming failed: {e}") from None

        raise UpstreamLLMError(f"Streaming failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None


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
    """Adapter so ``UsageTracker.record()`` works with OpenAI usage objects.

    Surfaces OpenAI's prompt-caching fields (``cached_tokens`` under
    ``prompt_tokens_details``) into the same shape Anthropic exposes,
    so the same ``UsageTracker`` cost formula works for both. The
    ``input_tokens`` value is reported NET of cached tokens — the
    cached portion is billed separately at the ``cache_read`` rate.
    Without this split, cached tokens get billed at the full input
    rate, inflating cost by ~10×.
    """

    def __init__(self, usage):
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        # Net "fresh" input tokens (those NOT served from cache). We
        # don't let cached exceed prompt; defensive against an API
        # quirk where the two could disagree.
        self.input_tokens = max(0, prompt - cached)
        self.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        # OpenAI has no separate cache-write step (writes happen
        # implicitly as a side-effect of a regular call); the discount
        # only applies on subsequent reads. So cache_creation = 0,
        # cache_read = cached_tokens.
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = cached


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
    PLANNING_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from None

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout if timeout is not None else _DEFAULT_SDK_TIMEOUT_SECONDS,
        }
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

    @property
    def planning_model(self) -> str | None:
        env = os.environ.get("BESSER_LLM_PLANNING_MODEL")
        if env:
            return None if env.lower() == "primary" else env
        if any(tier in self._model.lower() for tier in ("mini", "nano")):
            return None  # already on a cheap tier
        return self.PLANNING_MODEL

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        force_tool: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Send a message with tools. Retries on transient errors."""
        last_error = None
        api_messages = _openai_messages_to_api(system, messages)
        openai_tools = _anthropic_tools_to_openai(tools) if tools else None
        effective_model = model_override or self._model

        for attempt in range(_MAX_RETRIES + 1):
            try:
                # Newer models (GPT-5+, o3+) use max_completion_tokens
                # instead of max_tokens
                tok_key = _openai_max_tokens_key(effective_model)
                kwargs: dict[str, Any] = {
                    "model": effective_model,
                    tok_key: self._max_tokens,
                    "messages": api_messages,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools
                    if force_tool:
                        kwargs["tool_choice"] = {
                            "type": "function",
                            "function": {"name": force_tool},
                        }

                response = self._client.chat.completions.create(**kwargs)

                # Track usage + the actual served model (audit signal:
                # the requested model name may be aliased server-side).
                # Planning calls on the cheap override model are excluded
                # from served-model tracking — otherwise every run logs a
                # spurious "served model changed mid-run" warning when the
                # primary call follows a planning call.
                if response.usage:
                    self._usage.record(
                        _OpenAIUsageAdapter(response.usage), model=model_override
                    )
                if not model_override:
                    self._usage.set_served_model(getattr(response, "model", None))

                return _openai_response_to_common(response)

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES and _is_retryable(e):
                    backoff = _backoff_seconds(e, attempt)
                    logger.warning(
                        "OpenAI API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES + 1, backoff, e,
                    )
                    time.sleep(backoff)
                    continue
                if _is_auth_error(e):
                    raise InvalidApiKeyError(f"OpenAI API rejected the key: {e}") from None
                raise UpstreamLLMError(f"OpenAI API call failed: {e}") from None

        raise UpstreamLLMError(f"OpenAI API call failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None

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
                    # OpenAI omits the usage object from streaming
                    # responses by default. Without this flag, every
                    # chunk has ``chunk.usage is None`` and the
                    # ``UsageTracker.record(...)`` call below is never
                    # reached — cost stays frozen at whatever the
                    # non-streaming Phase 1 / gap-analyzer calls
                    # already recorded. Requesting the usage chunk
                    # restores per-turn cost tracking.
                    "stream_options": {"include_usage": True},
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools

                collected_text = ""
                tool_calls_accum: dict[int, dict] = {}
                finish_reason = None

                stream = self._client.chat.completions.create(**kwargs)
                try:
                    for chunk in stream:
                        # The served model name appears on every chunk
                        # in a stream; capture once.
                        self._usage.set_served_model(getattr(chunk, "model", None))
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
                finally:
                    # Ensure the underlying HTTP connection is released even
                    # if iteration raises. The OpenAI SDK stream object
                    # exposes ``close()``; fall back silently if missing.
                    close_fn = getattr(stream, "close", None)
                    if callable(close_fn):
                        try:
                            close_fn()
                        except Exception:
                            pass

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
                    backoff = _backoff_seconds(e, attempt)
                    logger.warning("OpenAI stream failed (attempt %d), retrying: %s", attempt + 1, e)
                    time.sleep(backoff)
                    continue
                if _is_auth_error(e):
                    raise InvalidApiKeyError(f"OpenAI API rejected the key: {e}") from None
                raise UpstreamLLMError(f"OpenAI API streaming failed: {e}") from None

        raise UpstreamLLMError(f"OpenAI streaming failed after {_MAX_RETRIES + 1} attempts: {last_error}") from None


# ======================================================================
# Mistral provider
# ======================================================================

class MistralProvider(OpenAIProvider):
    """
    Mistral provider via Mistral's OpenAI-compatible API.

    Mistral exposes an OpenAI-compatible Chat Completions endpoint at
    ``https://api.mistral.ai/v1`` that speaks the same ``tools`` /
    ``tool_calls`` / ``tool_choice`` function-calling protocol the
    orchestrator's ReAct loop relies on. We therefore reuse the entire
    ``OpenAIProvider`` request/response machinery (tool translation,
    streaming accumulation, usage tracking) and only override the
    defaults and the base-URL resolution — no separate ``mistralai``
    dependency is required.

    Mistral uses standard ``temperature`` + ``max_tokens`` parameters,
    so it goes through the normal (non-gpt-5) parameter branch:
    ``_openai_max_tokens_key`` returns ``max_tokens`` for Mistral model
    names because none of them contain ``gpt-5`` / ``o1`` / ``o3`` /
    ``o4``.

    Args:
        api_key: Mistral API key (or set ``MISTRAL_API_KEY`` env var).
        model: Model identifier (default: ``mistral-large-latest``).
        max_tokens: Maximum output tokens per response.
        base_url: Custom API base URL (defaults to the Mistral endpoint;
            falls back to ``MISTRAL_BASE_URL`` if set).
    """

    DEFAULT_MODEL = "mistral-large-latest"
    DEFAULT_MAX_TOKENS = 16384
    # Mistral's small model is a cheap sibling suitable for one-shot
    # planning calls. Like the other providers this is overridable via
    # ``BESSER_LLM_PLANNING_MODEL`` (set to ``primary`` to disable).
    PLANNING_MODEL = "mistral-small-latest"
    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        # Resolve the base URL here so the inherited OpenAI client is
        # pointed at Mistral. Precedence: explicit arg → MISTRAL_BASE_URL
        # env → the Mistral default endpoint. We never fall back to
        # OPENAI_BASE_URL for a Mistral client.
        resolved_base = (
            base_url
            or os.environ.get("MISTRAL_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        super().__init__(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            base_url=resolved_base,
            timeout=timeout,
        )

    @property
    def planning_model(self) -> str | None:
        env = os.environ.get("BESSER_LLM_PLANNING_MODEL")
        if env:
            return None if env.lower() == "primary" else env
        if "small" in self._model.lower():
            return None  # already on a cheap tier
        return self.PLANNING_MODEL


# ======================================================================
# Provider defaults (single source — the web runner and the config
# endpoint import this instead of keeping their own copies in sync)
# ======================================================================

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": ClaudeLLMClient.DEFAULT_MODEL,
    "openai": OpenAIProvider.DEFAULT_MODEL,
    "mistral": MistralProvider.DEFAULT_MODEL,
}


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
    raise InvalidApiKeyError(
        "No OpenAI API key found. Set OPENAI_API_KEY or pass api_key explicitly."
    )


def _resolve_mistral_api_key(
    api_key: str | None = None,
    config: dict | None = None,
) -> str:
    if api_key:
        return api_key
    if config:
        for key in ("api_key", "mistral_api_key", "MISTRAL_API_KEY"):
            if config.get(key):
                return config[key]
    for env_var in ("MISTRAL_API_KEY",):
        value = os.environ.get(env_var)
        if value:
            return value
    raise InvalidApiKeyError(
        "No Mistral API key found. Set MISTRAL_API_KEY or pass api_key explicitly."
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
        provider: ``"anthropic"`` (default), ``"openai"``, or ``"mistral"``.
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
    elif provider == "mistral":
        resolved_key = _resolve_mistral_api_key(api_key=api_key)
        return MistralProvider(
            api_key=resolved_key, model=model,
            base_url=base_url, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Supported providers: 'anthropic', 'openai', 'mistral'."
        )
