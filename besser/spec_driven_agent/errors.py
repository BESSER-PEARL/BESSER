"""Typed exceptions for the Spec-Driven Agent pipeline.

The web runner maps worker exceptions to SSE error codes by these types, so
unrelated failures sharing a built-in type (e.g. a resume fingerprint
mismatch vs. an invalid API key) are not mislabeled. Each class subclasses
the built-in it specialises, so ``except ValueError`` / ``except
RuntimeError`` call sites keep working.
"""


class InvalidApiKeyError(ValueError):
    """The provider rejected the API key (or none could be resolved)."""


class UpstreamLLMError(RuntimeError):
    """The provider API failed after retries (5xx, rate limit, timeout)."""


class CheckpointMismatchError(ValueError):
    """A resume was attempted against a checkpoint for a different project."""


class EmptyInstructionsError(ValueError):
    """``run()`` was called with blank instructions."""
