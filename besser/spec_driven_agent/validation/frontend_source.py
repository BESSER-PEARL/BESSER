"""Structural syntax check for TS/TSX/JS/JSX, with no Node toolchain.

``_new_syntax_error`` refuses a Python edit that would not compile; this module
gives frontend files the same protection, so an edit (``modify_file`` /
``replace_file_lines``) that leaves a TSX file unparseable is refused rather
than written, even when ``tsc`` is not available downstream.

Two checks, both deliberately conservative - this runs as a REFUSAL, and a
false refusal costs the model a turn:

``structure_error`` walks the source tracking comments, quoted strings,
template literals (including nested ``${}``), regex literals and JSX, and
reports only an outright delimiter fault in ``{}``/``[]``. Parentheses are not
tracked: prose inside JSX text carries unbalanced ones and delimiter damage
never shows up only there.

``json_container_faults`` parses the ``attr={{...}}`` containers that are
written as strict JSON - the generated table config, where such breaks
typically land. A container using JS object syntax (bare identifier keys,
shorthand properties) never was JSON and is skipped; a trailing comma is legal
TS and is not a defect.

Calibrated to zero findings on thousands of generated and real-world files
that esbuild parses, and a finding on every known broken one. Both callers
apply it as a regression test - ``before`` clean, ``after`` not - so a file
the scanner cannot read is never protected rather than falsely refused.
"""

from __future__ import annotations

import json
import os
import re

SUPPORTED_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

# A ``/`` may open a regex literal only in operand position. ``}`` and ``>``
# are deliberately absent: after them ``/`` is overwhelmingly JSX (``/>``,
# ``</tag>``), and admitting them cost real false positives on the BESSER
# frontend. ``=>`` is matched separately for the same reason.
_REGEX_PRECEDERS = frozenset("(,=:[!&|?;")
_REGEX_KEYWORDS = frozenset(
    ("return", "typeof", "case", "in", "of", "do", "else", "yield", "await", "throw")
)


def supports(rel_path: str) -> bool:
    return os.path.splitext(rel_path.lower())[1] in SUPPORTED_EXTENSIONS


def structure_error(source: str) -> tuple[str, int] | None:
    """``(message, line)`` for the first outright delimiter fault, else ``None``."""
    stack: list[tuple[str, int]] = []
    index, size, line = 0, len(source), 1
    while index < size:
        char = source[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        if char == "/" and index + 1 < size:
            if source[index + 1] == "/":
                stop = source.find("\n", index)
                index = size if stop < 0 else stop
                continue
            if source[index + 1] == "*":
                stop = source.find("*/", index + 2)
                if stop < 0:
                    return ("unterminated block comment", line)
                line += source.count("\n", index, stop)
                index = stop + 2
                continue
            if _regex_may_start(source, index):
                stop = _regex_end(source, index)
                if stop is not None:
                    index = stop
                    continue
            index += 1
            continue
        if char == "`":
            stop, newlines, error = _template_end(source, index)
            if error:
                return (error, line)
            line += newlines
            index = stop
            continue
        if char in "'\"":
            # A quoted string never spans a line in JS. A quote that does not
            # close on its own line is an apostrophe in JSX text, not a string.
            stop = _string_end(source, index, char)
            index = stop if stop is not None else index + 1
            continue
        if char in "{[":
            stack.append((char, line))
            index += 1
            continue
        if char in "}]":
            expected = "{" if char == "}" else "["
            if not stack:
                return (f'unbalanced "{char}" - nothing is open here', line)
            opened, opened_line = stack.pop()
            if opened != expected:
                return (f'"{char}" closes a "{opened}" opened on line {opened_line}', line)
            index += 1
            continue
        index += 1
    if stack:
        opened, opened_line = stack[-1]
        return (f'"{opened}" opened on line {opened_line} is never closed', opened_line)
    return None


_ATTRIBUTE_OBJECT = re.compile(r"(\w+)=\{\{")
_IDENTIFIER_KEY = re.compile(r"(?:^|[\{\[,])\s*[A-Za-z_$][\w$]*\s*:")
_QUOTED_FIRST_KEY = re.compile(r"^\{\s*(?:\"|\})")
_TRAILING_COMMA = re.compile(r",(\s*[\]\}])")


def json_container_faults(source: str) -> list[tuple[int, str, str]]:
    """``(line, attribute, reason)`` for JSON-written ``attr={{...}}`` that no longer parses."""
    faults: list[tuple[int, str, str]] = []
    for match in _ATTRIBUTE_OBJECT.finditer(source):
        opened = match.end() - 2
        closed = _match_brace(source, opened)
        if closed is None:
            continue
        inner = source[opened + 1:closed].strip()
        if not (inner.startswith("{") and inner.endswith("}")):
            continue
        if not _QUOTED_FIRST_KEY.match(inner) or _IDENTIFIER_KEY.search(_blank_quoted(inner)):
            continue
        relaxed = inner
        while True:
            shortened = _TRAILING_COMMA.sub(r"\1", relaxed)
            if shortened == relaxed:
                break
            relaxed = shortened
        try:
            json.loads(relaxed)
        except ValueError as exc:
            faults.append((source.count("\n", 0, match.start()) + 1, match.group(1), str(exc)[:90]))
    return faults


def new_syntax_error(before: str, after: str) -> tuple[str, int] | None:
    """``(message, line)`` when ``after`` is structurally broken and ``before`` was not."""
    if structure_error(before) is not None:
        return None
    broke = structure_error(after)
    if broke:
        return broke
    was = len(json_container_faults(before))
    faults = json_container_faults(after)
    if len(faults) > was:
        line, attribute, reason = faults[0]
        return (f'the JSON in {attribute}={{{{...}}}} no longer parses ({reason})', line)
    return None


# ----------------------------------------------------------------------
# Lexical helpers
# ----------------------------------------------------------------------

def _string_end(source: str, index: int, quote: str) -> int | None:
    cursor = index + 1
    while cursor < len(source):
        char = source[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "\n":
            return None
        if char == quote:
            return cursor + 1
        cursor += 1
    return None


def _previous_significant(source: str, index: int) -> str:
    cursor = index - 1
    while cursor >= 0 and source[cursor] in " \t\r\n":
        cursor -= 1
    return source[cursor] if cursor >= 0 else ""


def _previous_word(source: str, index: int) -> str:
    cursor = index - 1
    while cursor >= 0 and source[cursor] in " \t\r\n":
        cursor -= 1
    start = cursor
    while start >= 0 and (source[start].isalnum() or source[start] == "_"):
        start -= 1
    return source[start + 1:cursor + 1]


def _regex_may_start(source: str, index: int) -> bool:
    if _previous_significant(source, index) in _REGEX_PRECEDERS:
        return True
    cursor = index - 1
    while cursor >= 0 and source[cursor] in " \t\r\n":
        cursor -= 1
    if source[max(0, cursor - 1):cursor + 1] == "=>":
        return True
    return _previous_word(source, index) in _REGEX_KEYWORDS


def _regex_end(source: str, index: int) -> int | None:
    """Index after a regex literal that closes on its own line, else ``None``."""
    cursor = index + 1
    in_class = False
    while cursor < len(source):
        char = source[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "\n":
            return None
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        elif char == "/" and not in_class:
            cursor += 1
            while cursor < len(source) and source[cursor].isalpha():
                cursor += 1
            return cursor
        cursor += 1
    return None


def _template_end(source: str, index: int) -> tuple[int, int, str | None]:
    """``(index_after, newlines, error)`` for the template literal at ``index``."""
    cursor = index + 1
    newlines = 0
    while cursor < len(source):
        char = source[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "\n":
            newlines += 1
            cursor += 1
            continue
        if char == "`":
            return cursor + 1, newlines, None
        if char == "$" and cursor + 1 < len(source) and source[cursor + 1] == "{":
            depth = 1
            cursor += 2
            while cursor < len(source) and depth:
                inner = source[cursor]
                if inner == "\\":
                    cursor += 2
                    continue
                if inner == "\n":
                    newlines += 1
                elif inner == "`":
                    stop, nested, error = _template_end(source, cursor)
                    if error:
                        return cursor, newlines, error
                    newlines += nested
                    cursor = stop
                    continue
                elif inner in "'\"":
                    stop = _string_end(source, cursor, inner)
                    if stop is not None:
                        cursor = stop
                        continue
                elif inner == "/" and cursor + 1 < len(source):
                    if source[cursor + 1] == "/":
                        stop = source.find("\n", cursor)
                        cursor = len(source) if stop < 0 else stop
                        continue
                    if source[cursor + 1] == "*":
                        stop = source.find("*/", cursor + 2)
                        if stop < 0:
                            return cursor, newlines, "unterminated block comment"
                        newlines += source.count("\n", cursor, stop)
                        cursor = stop + 2
                        continue
                    if _regex_may_start(source, cursor):
                        stop = _regex_end(source, cursor)
                        if stop is not None:
                            cursor = stop
                            continue
                elif inner == "{":
                    depth += 1
                elif inner == "}":
                    depth -= 1
                cursor += 1
            continue
        cursor += 1
    return cursor, newlines, "unterminated template literal"


def _match_brace(source: str, index: int) -> int | None:
    """Index of the ``}`` matching ``source[index] == "{"``, or ``None``."""
    depth = 0
    size = len(source)
    while index < size:
        char = source[index]
        if char == "/" and index + 1 < size:
            if source[index + 1] == "/":
                stop = source.find("\n", index)
                index = size if stop < 0 else stop
                continue
            if source[index + 1] == "*":
                stop = source.find("*/", index + 2)
                if stop < 0:
                    return None
                index = stop + 2
                continue
        if char == "`":
            stop, _, error = _template_end(source, index)
            if error:
                return None
            index = stop
            continue
        if char in "'\"":
            stop = _string_end(source, index, char)
            index = stop if stop is not None else index + 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _blank_quoted(text: str) -> str:
    """``text`` with the contents of every quoted run replaced by spaces."""
    out: list[str] = []
    index, size = 0, len(text)
    while index < size:
        char = text[index]
        if char in "'\"":
            stop = _string_end(text, index, char)
            if stop is not None:
                out.append(" " * (stop - index))
                index = stop
                continue
        if char == "`":
            stop, _, error = _template_end(text, index)
            if error:
                stop = size
            out.append(" " * (stop - index))
            index = stop
            continue
        out.append(char)
        index += 1
    return "".join(out)
