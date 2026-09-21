"""Shared input-size contract for user specifications, not derived context."""

# Roughly 16k tokens for typical English, while remaining explicitly bounded.
# Model metadata, inventories and harness-added ledgers have separate budgets.
MAX_SPECIFICATION_CHARS = 64_000


def validate_specification(instructions: str) -> str:
    """Return accepted input unchanged; never silently discard requirements."""
    if len(instructions) > MAX_SPECIFICATION_CHARS:
        raise ValueError(
            f"instructions must contain at most {MAX_SPECIFICATION_CHARS:,} characters "
            f"(received {len(instructions):,}); the specification was not truncated."
        )
    return instructions
