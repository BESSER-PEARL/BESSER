"""What each provider accepts as a conversation, as two checkers.

``anthropic_violations`` checks a Messages API request (``system``, ``tools``,
``messages``) as ``ClaudeLLMClient._create`` builds it; ``openai_violations``
checks the ``messages`` list ``_openai_messages_to_api`` builds for Chat
Completions. Each returns a list of human-readable violations, empty when the
provider would accept the array. Content blocks are read in every shape the
agent's history holds: plain dicts, Anthropic SDK models and the client's own
``_TextBlock`` / ``_ToolUseBlock``.

The rules are the documented request constraints the agent's history must meet,
not a model of everything either API validates.
"""

from __future__ import annotations

import json
from typing import Any

# Breakpoints the Messages API accepts in one request (tools + system + messages).
ANTHROPIC_MAX_CACHE_BREAKPOINTS = 4

_THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})
_ANTHROPIC_BLOCK_TYPES = frozenset({"text", "tool_use", "tool_result"}) | _THINKING_TYPES


def block_type(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _blocks(content: Any) -> list:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content) if isinstance(content, list) else []


def _tool_use_ids(message: dict) -> list:
    return [block_get(b, "id") for b in _blocks(message.get("content"))
            if block_type(b) == "tool_use"]


def _tool_result_ids(message: dict) -> list:
    return [block_get(b, "tool_use_id") for b in _blocks(message.get("content"))
            if block_type(b) == "tool_result"]


def count_cache_breakpoints(request: dict) -> int:
    """``cache_control`` markers across tools, system and every message block."""
    count = 0
    for tool in request.get("tools") or []:
        count += bool(block_get(tool, "cache_control"))
    system = request.get("system")
    for block in system if isinstance(system, list) else []:
        count += bool(block_get(block, "cache_control"))
    for message in request.get("messages") or []:
        for block in _blocks(message.get("content")):
            count += bool(block_get(block, "cache_control"))
    return count


def anthropic_violations(request: dict) -> list[str]:
    """Why the Messages API would reject ``request``; empty when it would not."""
    errors: list[str] = []
    messages = request.get("messages") or []
    if not messages:
        return ["messages: empty"]
    if messages[0].get("role") != "user":
        errors.append(f"messages.0: first role is {messages[0].get('role')!r}, not 'user'")

    seen_tool_use_ids: set = set()
    for index, message in enumerate(messages):
        where = f"messages.{index}"
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant"):
            errors.append(f"{where}: role {role!r}")
            continue
        if isinstance(content, str):
            if not content.strip():
                errors.append(f"{where}: empty string content")
            continue
        if not isinstance(content, list) or not content:
            errors.append(f"{where}: empty {role} content")
            continue
        for position, block in enumerate(content):
            btype = block_type(block)
            at = f"{where}.content.{position}"
            if btype not in _ANTHROPIC_BLOCK_TYPES:
                errors.append(f"{at}: unknown block type {btype!r} ({type(block).__name__})")
                continue
            if btype == "text" and not str(block_get(block, "text") or "").strip():
                errors.append(f"{at}: empty text block")
            if btype in _THINKING_TYPES and block_get(block, "cache_control"):
                errors.append(f"{at}: cache_control on a thinking block")
            if btype == "tool_use":
                if role != "assistant":
                    errors.append(f"{at}: tool_use in a {role} message")
                bid = block_get(block, "id")
                if not isinstance(bid, str) or not bid:
                    errors.append(f"{at}: tool_use without an id")
                elif bid in seen_tool_use_ids:
                    errors.append(f"{at}: duplicate tool_use id {bid!r}")
                seen_tool_use_ids.add(bid)
                if not block_get(block, "name"):
                    errors.append(f"{at}: tool_use without a name")
                if not isinstance(block_get(block, "input"), dict):
                    errors.append(f"{at}: tool_use input is not an object")
            if btype == "tool_result":
                if role != "user":
                    errors.append(f"{at}: tool_result in a {role} message")
                if not isinstance(block_get(block, "content", ""), (str, list)):
                    errors.append(f"{at}: tool_result content is neither text nor blocks")
        types = [block_type(b) for b in content]
        if role == "assistant":
            first_other = next((i for i, t in enumerate(types) if t not in _THINKING_TYPES), len(types))
            if any(t in _THINKING_TYPES for t in types[first_other:]):
                errors.append(f"{where}: thinking block after a non-thinking block")
        if "tool_result" in types:
            last_result = max(i for i, t in enumerate(types) if t == "tool_result")
            if any(t != "tool_result" for t in types[:last_result]):
                errors.append(f"{where}: a non-tool_result block precedes a tool_result")

    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            ids = [i for i in _tool_use_ids(message) if i]
            if not ids:
                continue
            following = messages[index + 1] if index + 1 < len(messages) else None
            answered = _tool_result_ids(following) if following and following.get("role") == "user" else []
            for bid in ids:
                hits = answered.count(bid)
                if hits != 1:
                    errors.append(
                        f"messages.{index}: tool_use {bid!r} has {hits} tool_result(s) in the next message")
        elif message.get("role") == "user":
            results = _tool_result_ids(message)
            if not results:
                continue
            previous = messages[index - 1] if index else None
            issued = (_tool_use_ids(previous)
                      if previous is not None and previous.get("role") == "assistant" else [])
            for bid in results:
                if bid not in issued:
                    errors.append(
                        f"messages.{index}: tool_result {bid!r} does not answer the directly preceding "
                        "assistant message")

    breakpoints = count_cache_breakpoints(request)
    if breakpoints > ANTHROPIC_MAX_CACHE_BREAKPOINTS:
        errors.append(f"{breakpoints} cache_control breakpoints (max {ANTHROPIC_MAX_CACHE_BREAKPOINTS})")
    return errors


def openai_violations(api_messages: list[dict]) -> list[str]:
    """Why Chat Completions would reject ``api_messages``; empty when it would not."""
    errors: list[str] = []
    pending: set = set()
    issued: set = set()
    for index, message in enumerate(api_messages):
        where = f"messages.{index}"
        if not isinstance(message, dict):
            errors.append(f"{where}: not an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in issued:
                errors.append(f"{where}: tool message {call_id!r} answers no call of the preceding assistant")
            elif call_id not in pending:
                errors.append(f"{where}: tool call {call_id!r} answered twice")
            pending.discard(call_id)
            if not isinstance(content, (str, list)):
                errors.append(f"{where}: tool content is {type(content).__name__}")
            continue
        if pending:
            errors.append(f"{where}: {role} message before tool call(s) {sorted(pending)} were answered")
            pending = set()
        issued = set()
        if role in ("system", "user"):
            if not isinstance(content, (str, list)):
                errors.append(f"{where}: {role} content is {type(content).__name__}")
            elif isinstance(content, list) and any(
                    not isinstance(part, dict) or part.get("type") != "text" for part in content):
                errors.append(f"{where}: {role} content part that is not text")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if content is None and not calls:
                errors.append(f"{where}: assistant with null content and no tool calls")
            if content is not None and not isinstance(content, str):
                errors.append(f"{where}: assistant content is {type(content).__name__}")
            for call in calls:
                call_id = call.get("id")
                function = call.get("function") or {}
                if not isinstance(call_id, str) or not call_id:
                    errors.append(f"{where}: tool call without an id")
                if call_id in issued:
                    errors.append(f"{where}: duplicate tool call id {call_id!r}")
                if not function.get("name"):
                    errors.append(f"{where}: tool call {call_id!r} without a function name")
                try:
                    json.loads(function.get("arguments"))
                except (TypeError, ValueError):
                    errors.append(f"{where}: tool call {call_id!r} arguments are not JSON")
                issued.add(call_id)
            pending = set(issued)
        else:
            errors.append(f"{where}: role {role!r}")
    if pending:
        errors.append(f"tool call(s) {sorted(pending)} never answered")
    return errors
