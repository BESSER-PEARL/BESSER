"""Markdown-safe splitting for agent personalization.

The personalization LLM calls work on a numbered list of single-line texts, which
cannot carry multi-line Markdown: line breaks, list markers and code blocks would be
mangled. So a Markdown reply is split into its *structure* (list markers, heading
hashes, quote markers, blank lines, code blocks, tables), which never goes to the LLM
and is put back unchanged, and single-line *units* of text, which do.

Inline formatting (bold, links, code spans, ...) stays inside a unit, so the LLM sees
whole sentences. :func:`same_inline_markup` then checks that a rewritten unit kept
every inline marker, link target, code span and ``{placeholder}``; the caller keeps
the previous text when it did not, so formatting is never lost.
"""

import re
from collections import Counter

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# Block prefix kept out of the text: quote markers, then a heading, bullet (with an
# optional task checkbox) or ordered-list marker.
_PREFIX_RE = re.compile(
    r"^(?P<prefix>\s*(?:>\s*)*(?:#{1,6}\s+|[-*+]\s+(?:\[[ xX]\]\s+)?|\d+[.)]\s+)?)"
    r"(?P<text>.*?)(?P<tail>\s*)$"
)
_LIST_MARKER_RE = re.compile(r"^\s*(?:>\s*)*(?:[-*+]|\d+[.)])\s")
_INLINE_RE = re.compile(
    r"`[^`]*`"                       # inline code span, contents included
    r"|!?\[|\]\([^)\s]*\)"           # link / image brackets and targets
    r"|https?://[^\s)>\]]+"          # bare URLs
    r"|</?[A-Za-z][^>]*>"            # inline HTML
    r"|\{[^{}]*\}"                   # {session_variable} placeholders
    r"|\*\*|__|~~|\*"                # bold, strike, emphasis
    r"|(?<!\w)_|_(?!\w)"             # emphasis underscores (not snake_case)
)


def _is_structure_only(line: str) -> bool:
    """Lines that are left exactly as they are: blank, rules, table rows, indented code."""
    stripped = line.strip()
    if not any(char.isalpha() for char in stripped):
        return True
    if stripped.startswith("|"):
        return True
    return line.startswith(("    ", "\t")) and not _LIST_MARKER_RE.match(line)


def split_markdown(text: str) -> tuple[list, list[str]]:
    """Split ``text`` into ``(parts, units)``.

    ``parts`` is the message skeleton: a string is kept verbatim, an ``int`` is the
    index of the unit that goes there. ``units`` are the single-line texts to
    personalize. :func:`join_markdown` is the inverse.
    """
    parts: list = []
    units: list[str] = []
    in_fence = None
    lines = (text or "").split("\n")
    for index, line in enumerate(lines):
        newline = "\n" if index < len(lines) - 1 else ""
        fence = _FENCE_RE.match(line)
        if in_fence is not None:
            if fence and fence.group(1)[0] == in_fence[0] and len(fence.group(1)) >= len(in_fence):
                in_fence = None
            parts.append(line + newline)
            continue
        if fence:
            in_fence = fence.group(1)
            parts.append(line + newline)
            continue
        if _is_structure_only(line):
            parts.append(line + newline)
            continue
        match = _PREFIX_RE.match(line)
        parts.append(match.group("prefix"))
        parts.append(len(units))
        units.append(match.group("text"))
        parts.append(match.group("tail") + newline)
    return parts, units


def join_markdown(parts: list, units: list[str]) -> str:
    """Rebuild a message from its skeleton and the (personalized) units."""
    return "".join(units[part] if isinstance(part, int) else part for part in parts)


def same_inline_markup(before: str, after: str) -> bool:
    """True if ``after`` is a single line that kept all of ``before``'s inline markup."""
    if not isinstance(after, str) or not after.strip() or "\n" in after:
        return False
    return Counter(_INLINE_RE.findall(before)) == Counter(_INLINE_RE.findall(after))
