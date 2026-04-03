"""
Context compaction for LLM conversation history.

When the conversation grows too large for the context window, older messages
are summarized and replaced with a compact representation that preserves
the essential information (what tools were called, what files exist).
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Context compaction config
COMPACT_TOKEN_THRESHOLD = 80_000
COMPACT_PRESERVE_RECENT = 6


def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate the token count of a message list (~4 chars per token)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("content", "") or block.get("text", "")
                    total += len(str(text)) // 4
                elif hasattr(block, "text"):
                    total += len(block.text) // 4
                elif hasattr(block, "input"):
                    total += len(str(block.input)) // 4
    return total


def maybe_compact(
    messages: list[dict],
    tool_calls_log: list[dict],
    output_dir: str,
    threshold: int = COMPACT_TOKEN_THRESHOLD,
    preserve_recent: int = COMPACT_PRESERVE_RECENT,
) -> tuple[list[dict], bool]:
    """
    Compact conversation history if it exceeds the token threshold.

    Args:
        messages: The current conversation messages.
        tool_calls_log: Log of tool calls made so far.
        output_dir: Path to the output directory (for file listing).
        threshold: Token threshold above which compaction triggers.
        preserve_recent: Number of recent messages to preserve verbatim.

    Returns:
        A tuple of (compacted_messages, did_compact).
    """
    est_tokens = _estimate_tokens(messages)
    if est_tokens < threshold or len(messages) <= preserve_recent:
        return messages, False

    logger.info(
        "Compacting: ~%d tokens -> preserving last %d messages",
        est_tokens, preserve_recent,
    )

    to_summarize = messages[:-preserve_recent]
    to_preserve = messages[-preserve_recent:]
    summary = _summarize_messages(to_summarize, tool_calls_log, output_dir)

    compacted = [
        {"role": "user", "content": f"[Earlier work summarized]\n\n{summary}\n\nContinue."},
        {"role": "assistant", "content": [{"type": "text", "text": "Continuing."}]},
    ]
    compacted.extend(to_preserve)
    return compacted, True


def _summarize_messages(
    messages: list[dict],
    tool_calls_log: list[dict],
    output_dir: str,
) -> str:
    """Build a compact summary of earlier conversation messages."""
    lines = [f"Earlier: {len(messages)} messages"]
    if tool_calls_log:
        tools: dict[str, int] = {}
        for tc in tool_calls_log:
            tools[tc["tool"]] = tools.get(tc["tool"], 0) + 1
        lines.append(f"Tools: {', '.join(f'{k}({v}x)' for k, v in sorted(tools.items()))}")
    try:
        files = []
        for root, _, fnames in os.walk(output_dir):
            for f in fnames:
                if not f.startswith(".besser_"):
                    files.append(
                        os.path.relpath(os.path.join(root, f), output_dir).replace("\\", "/")
                    )
        if files:
            lines.append(f"Files: {', '.join(sorted(files)[:25])}")
    except Exception:
        pass
    return "\n".join(lines)
