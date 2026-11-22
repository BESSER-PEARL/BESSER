"""Reading the JSON out of a model's reply."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def clean_json_response(response: str) -> str:
    """Strip markdown fences from a JSON response produced by an LLM."""
    json_text = response.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    elif json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    return json_text.strip()


def parse_json_safely(json_text: str) -> Optional[Any]:
    """Parse JSON, returning ``None`` on decode errors instead of raising."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*\s*\n(.*?)\n?```", re.DOTALL)


def _closes(text: str, start: int) -> bool:
    """Whether the bracket at ``start`` is closed before the text ends
    (JSON strings respected). An unclosed one is a reply cut off mid-value."""
    depth, in_string, escaped = 0, False, False
    for char in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return True
    return False


def extract_json(text: str, kind: type = dict) -> Optional[Any]:
    """The JSON value of type ``kind`` (``dict`` or ``list``) an LLM reply holds.

    The whole reply or a fenced block is the answer when it parses, even an
    empty one. Otherwise only top-level values in the prose are considered
    (each is decoded with ``JSONDecoder.raw_decode`` and skipped past, so
    the values nested inside it are never candidates), empty ones are not an
    answer, and the longest wins over a citation such as ``[1]``. A value
    that is still open when the text ends means the reply was truncated:
    ``None``, never a fragment of it.
    """
    if not isinstance(text, str):
        return None
    for block in [text, *_FENCE_RE.findall(text)]:
        try:
            value = json.loads(block)
        except ValueError:
            continue
        return value if isinstance(value, kind) else None
    decoder = json.JSONDecoder()
    best, best_span, index = None, 0, 0
    while True:
        index = min((i for i in (text.find("{", index), text.find("[", index)) if i != -1), default=-1)
        if index == -1:
            return best
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            if not _closes(text, index):
                return None
            index += 1
            continue
        if isinstance(value, kind) and value and end - index > best_span:
            best, best_span = value, end - index
        index = end
