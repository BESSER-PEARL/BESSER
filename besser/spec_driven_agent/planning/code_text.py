"""Source text without its comments and string literals, in any language.

Pygments lexes the file by its name, so the same rule holds for Python,
Kotlin, Swift, Dart, C#, JavaScript and the rest: a reply such as "Task
added!", a training sentence or a "# must be unique" comment is not code.
"""
from __future__ import annotations

from functools import lru_cache

from pygments.lexer import Lexer
from pygments.lexers import get_lexer_for_filename
from pygments.token import Comment, String
from pygments.util import ClassNotFound


@lru_cache(maxsize=64)
def _lexer_for(suffix: str) -> Lexer | None:
    try:
        return get_lexer_for_filename(f"x{suffix}", stripnl=False, ensurenl=False)
    except ClassNotFound:
        return None


def code_only(text: str, filename: str) -> str:
    """``text`` with comments and strings removed, keeping every line break.

    Returns ``text`` unchanged for a file type Pygments does not know.
    """
    suffix = filename[filename.rfind("."):] if "." in filename else ""
    lexer = _lexer_for(suffix.lower())
    if lexer is None:
        return text
    return "".join("\n" * value.count("\n") if token in Comment or token in String else value
                   for token, value in lexer.get_tokens(text))
