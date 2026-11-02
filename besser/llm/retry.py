"""The retry policy every LLM call in BESSER shares.

Which statuses are worth retrying, how long to wait, and which 429s are a
spent quota that no retry will fix. The functions read an error the way the
provider SDKs raise it: an HTTP status on the error or on its ``response``,
the response's headers, and the provider's message in ``str(error)``.
"""

from __future__ import annotations

import re

#: Statuses retried besides every 5xx.
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

INITIAL_BACKOFF = 0.5  # seconds
MAX_BACKOFF = 5.0  # seconds, cap for ordinary (5xx/timeout) retries
# Rate limits (429) need a longer ceiling: some providers (notably Mistral)
# throttle hard and only recover after several seconds. The same value caps a
# server's Retry-After.
RATELIMIT_MAX_BACKOFF = 30.0  # seconds

# A status in the text only counts next to a status word: a bare "401" or
# "500" also occurs inside token counts ("140133 tokens", "205000 tokens").
_STATUS_IN_TEXT_RE = re.compile(
    r"\b(?:error code|status(?: code)?|http(?:/[\d.]+)?)\s*[:=]?\s*([1-5]\d\d)\b",
    re.IGNORECASE,
)


def status_code(error: Exception) -> int | None:
    """HTTP status of a provider error: SDK attribute first, then the text."""
    for source in (error, getattr(error, "response", None)):
        code = getattr(source, "status_code", None)
        if isinstance(code, int):
            return code
    match = _STATUS_IN_TEXT_RE.search(str(error))
    return int(match.group(1)) if match else None


def is_retryable_status(status: int) -> bool:
    """Every 5xx, as the SDKs' own retry did (529 and a tunnel's 52x are transient), and 408/409/429."""
    return status in RETRYABLE_STATUS_CODES or status >= 500


def is_rate_limit(error: Exception) -> bool:
    """True if the error is a provider rate-limit (HTTP 429)."""
    status = status_code(error)
    if status is not None:
        return status == 429
    s = str(error).lower()
    return (
        "rate limit" in s
        or "rate_limit" in s
        or "rate-limited" in s
        or "ratelimited" in s
    )


def is_quota_exhausted(error: Exception) -> bool:
    """True when a 429 is a DAILY/period quota exhaustion, not a transient throttle.

    E.g. Command Code's free LongCat returns ``429 "You've used all 100 free LongCat 2.0 requests for today. Your
    quota resets at …"``, and OpenAI ``429 insufficient_quota``. That will not recover inside a retry window, so the
    5-attempt backoff (0.5s → 30s cap) just burns ~15-40s per call before the
    fallback finally engages. Classify it so callers skip retries and fall back
    (or fail) immediately — and can tell the user *why*.
    """
    s = str(error).lower()
    # An exhausted prepaid balance (Command Code: 400 "insufficient credits").
    if "insufficient credits" in s or "purchase more credits" in s:
        return True
    if "429" not in s and "quota" not in s and "limit" not in s:
        return False
    return any(marker in s for marker in (
        "used all", "requests for today", "quota resets", "daily limit",
        "daily quota", "per day", "insufficient_quota", "quota exceeded",
        "exceeded your current quota",
    ))


def retry_after_seconds(error: Exception) -> float | None:
    """Extract a server-provided ``Retry-After`` hint (seconds), if any.

    The OpenAI/Mistral and Anthropic SDK error objects (and ``requests``'
    ``HTTPError``) carry the HTTP response, whose headers may include
    ``retry-after`` (delta-seconds). Returns ``None`` when no usable hint is present.
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


def backoff_seconds(error: Exception, attempt: int) -> float:
    """Seconds to wait before the next retry.

    Honors a server-provided ``Retry-After`` (capped), otherwise uses
    exponential backoff. Rate-limit (429) responses get a higher ceiling
    so a throttled provider isn't abandoned after a sub-second wait.
    """
    retry_after = retry_after_seconds(error)
    rate_limited = is_rate_limit(error)
    if retry_after is not None:
        return min(retry_after, RATELIMIT_MAX_BACKOFF)
    cap = RATELIMIT_MAX_BACKOFF if rate_limited else MAX_BACKOFF
    return min(INITIAL_BACKOFF * (2 ** attempt), cap)
