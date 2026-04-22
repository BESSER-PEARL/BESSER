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


# Real-tokenizer singleton. We prefer ``tiktoken`` because it's widely
# available and its BPE is close enough to Anthropic's tokenizer for the
# ``is the context over threshold?`` decision. If it's not installed, we
# fall back to the chars/4 heuristic (known to under-count code by
# ~30%, so the threshold trips slightly late — acceptable).
_TOKENIZER: Any = None
_TOKENIZER_LOADED = False


def _get_tokenizer():
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER
    _TOKENIZER_LOADED = True
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    """Count tokens via tiktoken if available, else chars/4 heuristic."""
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, disallowed_special=()))
        except Exception:
            # Tokenizer can fail on weird control characters; fall back.
            pass
    return len(text) // 4


def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count of a message list, preferring a real tokenizer."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("content", "") or block.get("text", "")
                    total += _count_tokens(str(text))
                elif hasattr(block, "text"):
                    total += _count_tokens(block.text)
                elif hasattr(block, "input"):
                    total += _count_tokens(str(block.input))
    return total


def maybe_compact(
    messages: list[dict],
    tool_calls_log: list[dict],
    output_dir: str,
    threshold: int = COMPACT_TOKEN_THRESHOLD,
    preserve_recent: int = COMPACT_PRESERVE_RECENT,
    domain_model: Any | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    state_machines: list[Any] | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
    primary_kind: str | None = None,
) -> tuple[list[dict], bool]:
    """
    Compact conversation history if it exceeds the token threshold.

    Args:
        messages: The current conversation messages.
        tool_calls_log: Log of tool calls made so far.
        output_dir: Path to the output directory (for file listing).
        threshold: Token threshold above which compaction triggers.
        preserve_recent: Number of recent messages to preserve verbatim.
        domain_model: Optional BUML DomainModel. When provided, a minimal
            recap (class names + association summary) is preserved in the
            summary so the LLM doesn't have to re-discover structure from
            file contents after compaction.

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
    summary = _summarize_messages(
        to_summarize, tool_calls_log, output_dir,
        domain_model=domain_model,
        gui_model=gui_model,
        agent_model=agent_model,
        state_machines=state_machines,
        object_model=object_model,
        quantum_circuit=quantum_circuit,
        primary_kind=primary_kind,
    )

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
    domain_model: Any | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    state_machines: list[Any] | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
    primary_kind: str | None = None,
) -> str:
    """Build a compact summary of earlier conversation messages."""
    lines = [f"Earlier: {len(messages)} messages"]

    if primary_kind:
        lines.append(f"Primary model: {primary_kind}")

    # Keep a minimal model recap so the LLM still knows what entities and
    # associations exist after the JSON was pruned from the context. A full
    # model JSON can be KB; this is ~hundreds of bytes.
    recap = _compact_model_recap(
        domain_model=domain_model,
        gui_model=gui_model,
        agent_model=agent_model,
        state_machines=state_machines,
        object_model=object_model,
        quantum_circuit=quantum_circuit,
    )
    if recap:
        lines.append(recap)

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


def _compact_model_recap(
    domain_model: Any | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    state_machines: list[Any] | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
) -> str:
    """One-line summary of the loaded BUML models for compaction context.

    Each model contributes a short recap if present. Nothing is serialized
    in full — the goal is to preserve just enough structure (names,
    counts) for the LLM to reason about entities after compaction drops
    the original JSON.
    """
    parts: list[str] = []

    if domain_model is not None:
        try:
            class_names = sorted(
                c.name for c in domain_model.get_classes() if getattr(c, "name", None)
            )
        except Exception:
            class_names = []
        try:
            enum_names = sorted(
                e.name for e in domain_model.get_enumerations() if getattr(e, "name", None)
            )
        except Exception:
            enum_names = []
        try:
            assoc_count = len(list(getattr(domain_model, "associations", []) or []))
        except Exception:
            assoc_count = 0
        if class_names:
            parts.append(f"Classes ({len(class_names)}): {', '.join(class_names)}")
        if enum_names:
            parts.append(f"Enums: {', '.join(enum_names)}")
        if assoc_count:
            parts.append(f"Associations: {assoc_count}")

    if state_machines:
        sm_names = [getattr(sm, "name", None) or "unnamed" for sm in state_machines]
        parts.append(f"State machines ({len(sm_names)}): {', '.join(sm_names)}")

    if agent_model is not None:
        parts.append("Agent present")

    if gui_model is not None:
        try:
            screen_count = sum(
                len(m.screens or []) for m in (gui_model.modules or [])
            )
        except Exception:
            screen_count = 0
        parts.append(f"GUI ({screen_count} screens)" if screen_count else "GUI present")

    if object_model is not None:
        parts.append("Object instances present")

    if quantum_circuit is not None:
        parts.append("Quantum circuit present")

    if not parts:
        return ""
    return "Model recap → " + " | ".join(parts)
