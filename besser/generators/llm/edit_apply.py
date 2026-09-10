"""Flexible ``old_text`` -> ``new_text`` application for ``ToolExecutor._modify_file``.

Ported from (github.com/Aider-AI/aider, ``aider/coders/editblock_coder.py``
@ 5dc9490, Apache-2.0 — retain this notice). Aider's own benchmark work is the
evidence base: a lenient apply ladder plus a *diagnostic* failure message is what
turns "old_text not found" from a multi-turn flail into a one-round-trip fix.

The ladder, most literal first — every tier is exact-by-construction, never
similarity-scored:

1. exact line-window match (caller handles the plain ``in`` case first)
2. uniform leading-whitespace correction — the model reproduced the block at the
   wrong indent level, which is the single most common miss
3. drop one spurious leading blank line (aider issue #25)

Two of aider's tiers are deliberately NOT ported:

* Its ``SequenceMatcher`` similarity tier (0.8 ratio) — aider disabled it in
  v0.11.2 because silently applying an 80%-similar edit is worse than a clean
  failure with a good hint. We agree, and we have the hint (below).
* Its ``...`` elision tier — in aider ``...`` is prose-level edit-block syntax,
  but ``...`` on its own line is a *valid Python statement* (Protocol/stub
  bodies, ``def f(): ...``). Treating it as an elision marker risks a spurious
  match in exactly the code we generate, for little gain in a JSON tool arg.

``find_similar_lines`` is the other half of the technique: on a miss, show the
model the closest actual lines so its retry can copy them verbatim.
"""

from __future__ import annotations

from difflib import SequenceMatcher


def _prep(text: str) -> tuple[str, list[str]]:
    """Normalize to a trailing newline and split keeping line endings."""
    if text and not text.endswith("\n"):
        text += "\n"
    return text, text.splitlines(keepends=True)


def _perfect_replace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    """Exact line-window match. First occurrence only (same as ``replace(.., 1)``)."""
    part_tup = tuple(part_lines)
    n = len(part_lines)
    if not n:
        return None
    for i in range(len(whole_lines) - n + 1):
        if tuple(whole_lines[i:i + n]) == part_tup:
            return "".join(whole_lines[:i] + replace_lines + whole_lines[i + n:])
    return None


def _match_but_for_leading_whitespace(
    whole_lines: list[str], part_lines: list[str]
) -> str | None:
    """Return the single common indent prefix if the window matches modulo indent.

    Requires (a) every line equal after ``lstrip()`` and (b) exactly ONE distinct
    extra-indent prefix across the non-blank lines — so a block that is uniformly
    shifted matches, while an inconsistently mangled one is refused.
    """
    n = len(whole_lines)
    if n != len(part_lines):
        return None
    if not all(whole_lines[i].lstrip() == part_lines[i].lstrip() for i in range(n)):
        return None
    prefixes = {
        whole_lines[i][: len(whole_lines[i]) - len(part_lines[i])]
        for i in range(n)
        if whole_lines[i].strip()
    }
    return prefixes.pop() if len(prefixes) == 1 else None


def _replace_with_missing_leading_whitespace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    """Whitespace-flexible tier.

    First remove any indent the model applied uniformly to BOTH old_text and
    new_text (so their relative indentation is preserved), then slide a window
    and re-apply the file's own indent to the replacement. Blank lines are never
    re-indented — that would add trailing whitespace.
    """
    leading = [len(p) - len(p.lstrip()) for p in part_lines if p.strip()]
    leading += [len(p) - len(p.lstrip()) for p in replace_lines if p.strip()]
    if leading and min(leading):
        k = min(leading)
        part_lines = [p[k:] if p.strip() else p for p in part_lines]
        replace_lines = [p[k:] if p.strip() else p for p in replace_lines]

    n = len(part_lines)
    if not n:
        return None
    for i in range(len(whole_lines) - n + 1):
        add = _match_but_for_leading_whitespace(whole_lines[i:i + n], part_lines)
        if add is None:
            continue
        fixed = [add + r if r.strip() else r for r in replace_lines]
        return "".join(whole_lines[:i] + fixed + whole_lines[i + n:])
    return None


def _perfect_or_whitespace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    return _perfect_replace(whole_lines, part_lines, replace_lines) or (
        _replace_with_missing_leading_whitespace(whole_lines, part_lines, replace_lines)
    )


def replace_most_similar_chunk(whole: str, part: str, replace: str) -> str | None:
    """Return the edited file text, or ``None`` when no tier matched.

    Pure: never touches disk, never mutates its arguments.
    """
    whole, whole_lines = _prep(whole)
    part, part_lines = _prep(part)
    replace, replace_lines = _prep(replace)

    res = _perfect_or_whitespace(whole_lines, part_lines, replace_lines)
    if res is not None:
        return res

    # Models sometimes prepend a blank line to the block (aider issue #25).
    if len(part_lines) > 2 and not part_lines[0].strip():
        res = _perfect_or_whitespace(whole_lines, part_lines[1:], replace_lines)
        if res is not None:
            return res

    return None


def find_similar_lines(
    search: str, content: str, threshold: float = 0.6, pad: int = 5
) -> str:
    """Closest line-window in *content* to *search*, as a "did you mean" hint.

    Returns "" when nothing clears *threshold* — a bad hint is worse than none.
    Diagnostic output only; it never drives an edit.

    Divergence from aider (deliberate): aider scores windows with
    ``SequenceMatcher`` over LISTS OF LINES, which compares whole lines for
    *equality*. That works for its multi-line prose edit blocks, but ours arrive
    as a JSON ``old_text`` that is frequently 1-3 lines — there, a single
    mistyped character makes a line unequal and a 2-line block scores 0.5,
    below any useful threshold, so the hint would vanish exactly when it is most
    needed. We score CHARACTER-level similarity of the joined window instead,
    with ``quick_ratio`` as an O(n) prefilter so a large file stays cheap.
    """
    search_lines = search.splitlines()
    content_lines = content.splitlines()
    if not search_lines or not content_lines:
        return ""

    n = len(search_lines)
    search_text = "\n".join(search_lines)
    best_ratio = 0.0
    best_i = -1
    matcher = SequenceMatcher(None, search_text, "", autojunk=False)
    for i in range(len(content_lines) - n + 1):
        matcher.set_seq2("\n".join(content_lines[i:i + n]))
        # quick_ratio is a cheap upper bound: if it cannot beat the incumbent,
        # the exact ratio cannot either.
        if matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i

    if best_i < 0 or best_ratio < threshold:
        return ""

    best = content_lines[best_i:best_i + n]
    # An exactly-anchored window is already the answer; otherwise pad for context.
    if best and best[0] == search_lines[0] and best[-1] == search_lines[-1]:
        return "\n".join(best)
    lo = max(0, best_i - pad)
    hi = min(len(content_lines), best_i + n + pad)
    return "\n".join(content_lines[lo:hi])
