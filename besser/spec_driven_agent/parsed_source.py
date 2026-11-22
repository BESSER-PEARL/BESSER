"""One parse per distinct source text, shared across the validators.

Many call sites across ``validation/`` and ``planning/`` each parse the
generated tree independently, and ``_collect_validation_issues`` runs once per
Phase 3 fix attempt, so without sharing the same unchanged source is parsed
many times per run.

Keyed on the source text, not the path: two callers reading the same file get
one parse, and a file rewritten between attempts re-parses because its text
differs. Hashing is two orders of magnitude cheaper than parsing, so the
lookup pays for itself even on a miss.

**Callers must treat the tree as read-only.** Nothing in this package mutates
an AST (``ast.unparse`` calls only render a node back to text), which is
what makes sharing safe. A future transformer must parse its own copy.
"""
from __future__ import annotations

import ast
from functools import lru_cache

# ~300 KB per tree for a typical generated file, so this bounds the cache at
# about 20 MB. Generated apps run 15-60 Python files; 64 holds a whole app plus the
# scaffold without evicting between validators in the same pass.
_MAX_CACHED_TREES = 64


# ast.parse's own placeholder when no filename was supplied.
_ANONYMOUS = "<unknown>"


@lru_cache(maxsize=_MAX_CACHED_TREES)
def _parse(source: str, mode: str) -> ast.AST:
    return ast.parse(source, mode=mode)


def parse_source(source: str, filename: str | None = None,
                 mode: str = "exec") -> ast.AST:
    """Parsed *source*, reusing an earlier parse of identical text.

    ``mode`` IS part of the key: ``mode="eval"`` yields an ``ast.Expression``
    where the default yields an ``ast.Module``, so sharing across modes would
    hand a caller the wrong node type. Callers parsing a type annotation use
    the eval mode.

    ``filename`` is NOT part of the key. It never reaches the tree, only the
    ``SyntaxError`` message, so two callers reading the same text under
    different names still share one parse and the name is attached to the
    error below -- a report still says which file failed.

    Raises ``SyntaxError`` exactly as ``ast.parse`` does -- callers that treat
    unparseable generated code as a finding keep working unchanged. A failing
    parse is not cached, so a file repaired between attempts is retried.
    """
    try:
        return _parse(source, mode)
    except SyntaxError as exc:
        # ast.parse fills in "<unknown>" when given no filename, so an
        # emptiness check never fires; replace the placeholder too.
        if filename and exc.filename in (None, "", _ANONYMOUS):
            exc.filename = filename
        raise


def cache_info():
    """Hits/misses, for tests asserting the redundancy is actually gone."""
    return _parse.cache_info()


def clear_cache() -> None:
    """Drop every cached tree. For tests, and for a run that has finished."""
    _parse.cache_clear()
