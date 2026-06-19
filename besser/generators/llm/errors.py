"""Typed exceptions for the LLM generator pipeline.

The web runner maps worker exceptions to SSE error codes. Before this
module existed it keyed on built-in types (``ValueError`` →
``INVALID_KEY``, ``RuntimeError`` → ``UPSTREAM_LLM``), which mislabeled
unrelated failures — e.g. a resume fingerprint mismatch surfaced as an
invalid API key. Each class below subclasses the built-in it used to be
raised as, so any existing ``except ValueError`` / ``except
RuntimeError`` call sites keep working while the runner can now match
on the specific type first.
"""


class InvalidApiKeyError(ValueError):
    """The provider rejected the API key (or none could be resolved)."""


class UpstreamLLMError(RuntimeError):
    """The provider API failed after retries (5xx, rate limit, timeout)."""


class CheckpointMismatchError(ValueError):
    """A resume was attempted against a checkpoint for a different project."""


class EmptyInstructionsError(ValueError):
    """``run()`` was called with blank instructions."""
