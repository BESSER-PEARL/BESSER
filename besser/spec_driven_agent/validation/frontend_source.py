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
typically land. Parts holding anything but JSON tokens (identifier keys or
values, template literals, operators, comments) are JavaScript, not JSON, and
are not judged; the object and array literals nested in them still are. A
trailing comma is legal TS and is not a defect.

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
# Words a quoted string may follow (``from 'x'``, ``T extends 'a'``, ``x as 'b'``).
_STRING_KEYWORDS = _REGEX_KEYWORDS | frozenset(
    ("from", "import", "extends", "as", "satisfies", "default", "void", "delete", "keyof")
)


def supports(rel_path: str) -> bool:
    return os.path.splitext(rel_path.lower())[1] in SUPPORTED_EXTENSIONS


def structure_error(source: str) -> tuple[str, int] | None:
    """``(message, line)`` for the first outright delimiter fault, else ``None``."""
    stack: list[tuple[str, int]] = []
    index, size, line = 0, len(source), 1
    last = -1  # index of the last code character, comments and whitespace skipped
    while index < size:
        char = source[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        if char in " \t\r":
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
            if _regex_may_follow(source, last):  # ``last`` skips comments
                stop = _regex_end(source, index)
                if stop is not None:
                    index, last = stop, stop - 1
                    continue
            index, last = index + 1, index
            continue
        if char == "`":
            stop, newlines, error = _template_end(source, index)
            if error:
                return (error, line)
            line += newlines
            index, last = stop, stop - 1
            continue
        if char in "'\"":
            # A quoted string never spans a line in JS. A quote that does not
            # close on its own line, or that follows a word or a closer, is an
            # apostrophe in JSX text, not a string.
            stop = _quoted_end(source, index, last)
            index, last = (stop, stop - 1) if stop is not None else (index + 1, index)
            continue
        if char in "{[":
            stack.append((char, line))
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if not stack:
                return (f'unbalanced "{char}" - nothing is open here', line)
            opened, opened_line = stack.pop()
            if opened != expected:
                return (f'"{char}" closes a "{opened}" opened on line {opened_line}', line)
        index, last = index + 1, index
    if stack:
        opened, opened_line = stack[-1]
        return (f'"{opened}" opened on line {opened_line} is never closed', opened_line)
    return None


_ATTRIBUTE_OBJECT = re.compile(r"(\w+)=\{\{")
_QUOTED_FIRST_KEY = re.compile(r"^\{\s*(?:\"|\})")
_TRAILING_COMMA = re.compile(r",(\s*[\]\}])")
# Whitespace, punctuation, a string with JSON escapes, a JSON number or literal.
_JSON_TOKEN = re.compile(
    r'[ \t\r\n]+|[{}\[\]:,]|"(?:[^"\\\n]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w$.])|(?:true|false|null)(?![\w$])")


def json_container_faults(source: str) -> list[tuple[int, str, str]]:
    """``(line, attribute, reason)`` for JSON-written ``attr={{...}}`` that no longer parses."""
    faults: list[tuple[int, str, str]] = []
    for match in _ATTRIBUTE_OBJECT.finditer(source):
        opened = match.end() - 2
        closed = match_bracket(source, opened)
        if closed is None:
            continue
        inner = source[opened + 1:closed].strip()
        if not (inner.startswith("{") and inner.endswith("}")):
            continue
        if not _QUOTED_FIRST_KEY.match(inner) or match_bracket(inner, 0) != len(inner) - 1:
            continue  # not JSON-written, or a comma expression ``{...}, {...}``
        if _in_comment(source, match.start()):
            continue
        fault = _json_fault(inner)
        if fault:
            message, pos = fault
            line = inner.count("\n", 0, pos) + 1
            column = pos - inner.rfind("\n", 0, pos)
            reason = f"{message}: line {line} column {column} (char {pos})"[:90]
            faults.append((source.count("\n", 0, match.start()) + 1, match.group(1), reason))
    return faults


def _json_fault(literal: str) -> tuple[str, int] | None:
    """``(message, offset)`` where the JSON-written parts of an object or array literal fail.

    A literal that also holds JavaScript is judged with each nested literal
    standing in as one value, once the nested literals themselves are judged,
    and its comments as whitespace.
    """
    reduced = literal
    if not _written_as_json(literal):
        for start, stop, nested in _nested_spans(literal):
            if nested:
                fault = _json_fault(literal[start:stop + 1])
                if fault:
                    return fault[0], start + fault[1]
            # Same length and line breaks, so the reason's position still holds.
            blank = re.sub(r"[^\n]", " ", literal[start + 1:stop + 1])
            reduced = reduced[:start] + ("0" if nested else " ") + blank + reduced[stop + 1:]
        if not _written_as_json(reduced):
            return None
    relaxed = reduced
    while True:
        shortened = _TRAILING_COMMA.sub(r"\1", relaxed)
        if shortened == relaxed:
            break
        relaxed = shortened
    try:
        json.loads(relaxed, strict=False)
    except json.JSONDecodeError as exc:
        return exc.msg, exc.pos
    return None


def _nested_spans(literal: str) -> list[tuple[int, int, bool]]:
    """``(start, stop, nested)`` of the comments and nested literals directly in ``literal``.

    A nested object or array literal counts only as a value - after ``:`` in
    an object, after ``[`` or ``,`` in an array (or a ``{`` after a closer
    there), outside parentheses. The scan stops at ``<`` or a ``/`` that is not
    a comment, which may open JSX or a regex.
    """
    found: list[tuple[int, int, bool]] = []
    parent = previous = literal[0]
    depth, index, size = 0, 1, len(literal) - 1
    while index < size:
        char = literal[index]
        if char in " \t\r\n":
            index += 1
            continue
        if char == "/" and literal[index + 1] in "/*":
            closer = "\n" if literal[index + 1] == "/" else "*/"
            stop = literal.find(closer, index + 2)
            if stop < 0:
                break
            found.append((index, stop - 1 if closer == "\n" else stop + 1, False))
            index = stop + len(closer)
            continue
        if char in "</":
            break
        if char in "'\"`":
            if char == "`":
                stop, _, error = _template_end(literal, index)
            else:
                stop, error = _string_end(literal, index, char), None
            if stop is None or error:
                break
            index, previous = stop, char
            continue
        if char in "{[":
            stop = match_bracket(literal, index)
            if stop is None:
                break
            # In an array a ``{`` after a closer is a value too: no JS reads ``} {``.
            in_array = previous in "[," or (char == "{" and previous in "}]")
            if depth == 0 and (previous == ":" if parent == "{" else in_array):
                found.append((index, stop, True))
            index, previous = stop + 1, literal[stop]
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        index, previous = index + 1, char
    return found


def _in_comment(source: str, index: int) -> bool:
    """``index`` may sit in a comment: a ``//`` before it on its line, or an open ``/*``."""
    line_start = source.rfind("\n", 0, index) + 1
    return "//" in source[line_start:index] or source.rfind("/*", 0, index) > source.rfind("*/", 0, index)


def _written_as_json(inner: str) -> bool:
    """``inner`` holds only JSON tokens, so a JSON parse failure is a JS one too.

    Any other token is a JavaScript value this check cannot judge. JS also reads
    some JSON-token runs that JSON rejects: ``[1][0]``, ``1 -2``, a numeric,
    computed or ``true``/``false``/``null`` key and an array hole. So a dropped
    comma that JS reads as indexing or subtraction (``[[1] [2]]``, ``[10 -2]``)
    is valid and passes.
    """
    stack: list[str] = []
    previous = ""
    index = 0
    while index < len(inner):
        match = _JSON_TOKEN.match(inner, index)
        if match is None:
            # A string left open on its line is broken JS too; else a JS-only escape.
            return inner[index] == '"' and _string_end(inner, index, '"') is None
        token, index = match.group(), match.end()
        if token.isspace():
            continue
        if previous not in ("", "{", "[", ":", ",") and token[0] in "[-":
            return False  # ``[1][0]``, ``1 -2``
        key_position = stack[-1:] == ["{"] and previous in ("{", ",")
        if key_position and (token == "[" or token[0] in "-0123456789tfn"):
            return False  # computed, numeric or keyword key
        if stack[-1:] == ["["] and token == "," and previous in ("[", ","):
            return False  # array hole
        if token in ("{", "["):
            stack.append(token)
        elif token in ("}", "]") and stack:
            stack.pop()
        previous = token
    return True


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


def _previous_word(source: str, index: int) -> str:
    cursor = index - 1
    while cursor >= 0 and source[cursor] in " \t\r\n":
        cursor -= 1
    start = cursor
    while start >= 0 and (source[start].isalnum() or source[start] == "_"):
        start -= 1
    return source[start + 1:cursor + 1]


def _regex_may_start(source: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and source[cursor] in " \t\r\n":
        cursor -= 1
    return _regex_may_follow(source, cursor)


def _regex_may_follow(source: str, last: int) -> bool:
    """A ``/`` after the code character at ``last`` may open a regex literal."""
    if last < 0:
        return False
    if source[last] in _REGEX_PRECEDERS or source[max(0, last - 1):last + 1] == "=>":
        return True
    return _previous_word(source, last + 1) in _REGEX_KEYWORDS


def _quoted_end(source: str, index: int, last: int) -> int | None:
    """Index after the string opening at ``index``, or ``None`` when it is prose.

    ``last`` is the index of the previous code character. After a non-keyword
    word or a closer the quote is usually an apostrophe in JSX text
    (``don't``, ``{name}'s``), so it is not paired with a later quote across
    a bracket - that hid the brace in ``don't {x === 'a'}``. Minified code
    does put strings there (``for(...)"margin"``), so a bracket-free one still
    counts.
    """
    stop = _string_end(source, index, source[index])
    if stop is None or last < 0:
        return stop
    previous = source[last]
    prose_position = previous in ")]}" or (
        (previous.isalnum() or previous == "_")
        and _previous_word(source, last + 1) not in _STRING_KEYWORDS)
    if prose_position and any(c in "{}[]" for c in source[index + 1:stop - 1]):
        return None
    return stop


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


def match_bracket(source: str, index: int) -> int | None:
    """Index of the ``}`` / ``]`` matching the ``{`` / ``[`` at ``index``, or ``None``.

    Only the opener's own kind is counted; strings, templates and comments are skipped.
    """
    opener = source[index]
    closer = "}" if opener == "{" else "]"
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
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None
