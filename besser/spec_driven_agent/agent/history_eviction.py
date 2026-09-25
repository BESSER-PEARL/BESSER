"""
Context eviction for the agent loop's message history.

The Phase-2 tool loop re-sends the ENTIRE growing message array to the model on
every turn. The dominant growth term is stale FILE CONTENT:

* the body a ``write_file`` / ``modify_file`` tool_use carried in its ``input``
  (the whole file the model wrote), and
* the body a ``read_file`` tool_result returned (up to ``MAX_FILE_READ`` chars).

Once the model has moved past those turns, the bodies are dead weight — the file
lives on disk and can be re-read on demand (the compaction summary already tells
the model to "re-read before editing"). Re-sending them every turn is what makes
token cost grow ~quadratically with turn count.

This module replaces those stale bodies in OLDER messages with short stubs so the
per-turn context stops re-sending them. Crucially it:

* never REMOVES a block, so every ``tool_use`` keeps its paired ``tool_result``
  (both providers reject an orphaned pair);
* never touches the most recent ``preserve_recent`` messages (the model's active
  working set);
* is pure — it never mutates the caller's message/block objects; it returns a
  new list, building fresh representations only for the blocks it elides and
  keeping every other block by reference.

Content blocks come in two shapes depending on provider/turn: normalized dicts
(the tool_result blocks this repo builds) and SDK objects (Anthropic tool_use
blocks). The helpers below read both; elided blocks are always emitted as plain
dicts, which both provider clients accept as message-content input.

SAFETY: this rewrites the message array the provider re-serializes each turn.
It is gated OFF by default (see ``orchestrator``); enable it only after a live
verification run on each provider you actually use.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# File-writing tools whose tool_use ``input`` carries a large body we can elide.
_FILE_WRITE_TOOLS = frozenset({"write_file", "modify_file", "replace_file_lines"})
# Input keys that hold a file body (never elide ``path`` — the model needs it).
_BODY_INPUT_KEYS = frozenset({"content", "new_content", "file_text", "text", "new_str", "old_text", "new_text"})

# Don't bother eliding anything smaller than this — the stub itself costs tokens,
# and small results (a write ack, a short file) aren't the problem.
DEFAULT_MIN_BODY_CHARS = 600
# Keep the most recent N messages fully intact (the active working set). Matches
# compaction's COMPACT_PRESERVE_RECENT so the two policies agree.
DEFAULT_PRESERVE_RECENT = 6


def _block_type(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def without_rejected_edit_drafts(messages: list[dict]) -> list[dict]:
    """A request-only projection: keep errors/current code, omit refused drafts.

    Replaying an invalid candidate beside each fresh read anchored Qwen to that
    candidate, even after changing editor tools. Do not make the model copy it
    again. Preserve IDs, ordering, successful edits and the original trace/history.
    Nothing is inferred applied; tool results retain their failure and guidance.
    """
    index = _index_tool_uses(messages)
    rejected = set()
    for message in messages:
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            bid = _block_get(block, "tool_use_id")
            if _block_type(block) != "tool_result" or index.get(bid, {}).get("name") not in _FILE_WRITE_TOOLS:
                continue
            try:
                result = json.loads(_block_get(block, "content"))
            except (TypeError, ValueError):
                continue
            if isinstance(result, dict) and result.get("error"):
                rejected.add(bid)
    if not rejected:
        return messages
    projected = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            projected.append(message)
            continue
        blocks = []
        for block in content:
            if _block_type(block) == "tool_use" and _block_get(block, "id") in rejected:
                args = _block_get(block, "input") or {}
                if not isinstance(args, dict):
                    blocks.append(block)
                    continue
                blocks.append({"type": "tool_use", "id": _block_get(block, "id"),
                    "name": _block_get(block, "name"), "input": {
                        key: "[Rejected draft omitted; NOT applied. Read current source and write a corrected replacement.]"
                        if key in _BODY_INPUT_KEYS else value for key, value in args.items()}})
            elif _block_type(block) == "tool_result" and _block_get(block, "tool_use_id") in rejected:
                result = json.loads(_block_get(block, "content"))
                result.pop("would_write", None)
                blocks.append({"type": "tool_result", "tool_use_id": _block_get(block, "tool_use_id"),
                               "content": json.dumps(result), "is_error": True})
            else:
                blocks.append(block)
        projected.append({**message, "content": blocks})
    return projected


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _index_tool_uses(messages: list[dict]) -> dict[str, dict[str, Any]]:
    """Map every ``tool_use`` id -> {name, path} across all messages.

    Lets a ``tool_result`` (which carries only ``tool_use_id``) be attributed to
    the tool that produced it, so we elide only read_file results — not, say, a
    validation summary the model still needs.
    """
    index: dict[str, dict[str, Any]] = {}
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if _block_type(block) != "tool_use":
                continue
            bid = _block_get(block, "id")
            if not isinstance(bid, str):
                continue
            inp = _block_get(block, "input") or {}
            path = inp.get("path") if isinstance(inp, dict) else None
            index[bid] = {"name": _block_get(block, "name"), "path": path}
    return index


def _elide_tool_use(block: Any, min_body_chars: int) -> tuple[Any, bool]:
    """If ``block`` is a file-write tool_use with a large body, return a dict
    copy with the body replaced by a stub. Otherwise return (block, False)."""
    if _block_type(block) != "tool_use":
        return block, False
    name = _block_get(block, "name")
    if name not in _FILE_WRITE_TOOLS:
        return block, False
    inp = _block_get(block, "input")
    if not isinstance(inp, dict):
        return block, False
    path = inp.get("path")
    new_input: dict[str, Any] = {}
    elided = False
    for key, val in inp.items():
        if (
            key in _BODY_INPUT_KEYS
            and isinstance(val, str)
            and len(val) >= min_body_chars
        ):
            new_input[key] = (
                f"<elided: {len(val)} chars from an earlier edit request for "
                f"{path or 'this file'}; this does not mean it was applied — "
                f"check its tool result and call read_file for current contents>"
            )
            elided = True
        else:
            new_input[key] = val
    if not elided:
        return block, False
    return (
        {
            "type": "tool_use",
            "id": _block_get(block, "id"),
            "name": name,
            "input": new_input,
        },
        True,
    )


def _elide_tool_result(
    block: Any, tool_index: dict[str, dict[str, Any]], min_body_chars: int
) -> tuple[Any, bool]:
    """If ``block`` is a large read_file tool_result, return a dict copy with the
    body replaced by a stub. Otherwise return (block, False)."""
    if _block_type(block) != "tool_result":
        return block, False
    bid = _block_get(block, "tool_use_id")
    info = tool_index.get(bid) if isinstance(bid, str) else None
    if not info or info.get("name") != "read_file":
        return block, False
    content = _block_get(block, "content")
    body = _stringify(content)
    if len(body) < min_body_chars:
        return block, False
    path = info.get("path")
    stub = json.dumps(
        {
            "note": (
                f"read_file result for {path or 'this file'} "
                f"({len(body)} chars) elided to save context — call read_file "
                f"again if you need its current contents."
            )
        }
    )
    new_block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": bid,
        "content": stub,
    }
    # Preserve an is_error flag if the original carried one.
    is_error = _block_get(block, "is_error")
    if is_error is not None:
        new_block["is_error"] = is_error
    return new_block, True


def evict_stale_file_bodies(
    messages: list[dict],
    *,
    preserve_recent: int = DEFAULT_PRESERVE_RECENT,
    min_body_chars: int = DEFAULT_MIN_BODY_CHARS,
) -> tuple[list[dict], int]:
    """Return ``(new_messages, elided_count)`` with stale file bodies stubbed.

    Elides ``write_file``/``modify_file`` bodies and ``read_file`` results in all
    but the last ``preserve_recent`` messages. Never removes a block (pairing is
    preserved) and never mutates the input; unchanged blocks are shared by
    reference. ``elided_count`` is how many blocks were stubbed (0 → nothing
    changed and the original list is returned unchanged).
    """
    if not isinstance(messages, list) or len(messages) <= preserve_recent:
        return messages, 0

    tool_index = _index_tool_uses(messages)
    cutoff = len(messages) - preserve_recent
    elided = 0
    out: list[dict] = []

    for i, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if i >= cutoff or not isinstance(content, list):
            out.append(msg)
            continue

        new_content: list[Any] = []
        changed = False
        for block in content:
            btype = _block_type(block)
            if btype == "tool_use":
                nb, did = _elide_tool_use(block, min_body_chars)
            elif btype == "tool_result":
                nb, did = _elide_tool_result(block, tool_index, min_body_chars)
            else:
                nb, did = block, False
            if did:
                elided += 1
                changed = True
            new_content.append(nb)

        if changed:
            out.append({**msg, "content": new_content})
        else:
            out.append(msg)

    if elided == 0:
        return messages, 0
    logger.info("History eviction: stubbed %d stale file body/bodies", elided)
    return out, elided
