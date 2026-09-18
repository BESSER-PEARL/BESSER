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
4. blank-line runs compared by presence, not count - every non-blank line
   still exact. Generated scaffolds carry runs of 3-7 blank lines (Jinja
   whitespace) that models collapse when quoting; measured 2026-09-17 on a
   fresh FastAPI scaffold, a quote spanning a run missed tiers 1-3 in 23/25
   windows of routers/bill.py - the largest single cause of "old_text not
   found" on text the model had just read.
5. a uniform line-number prefix (``  12| ``, ``12:``, ``12<tab>``) on EVERY
   non-blank quoted line is stripped - the quote was copied from numbered
   output (modify_file's own post-edit snippet, ``cat -n``).

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

import re

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


def _collapse_blank_runs(lines: list[str]) -> tuple[list[str], list[int]]:
    """``lines`` with each run of blank lines reduced to one ``"\\n"``, plus
    the original index of every kept line."""
    kept: list[str] = []
    index: list[int] = []
    prev_blank = False
    for i, line in enumerate(lines):
        blank = not line.strip()
        if blank and prev_blank:
            continue
        kept.append("\n" if blank else line)
        index.append(i)
        prev_blank = blank
    return kept, index


def _replace_with_collapsed_blank_runs(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    """Tier 4: forgive only the COUNT of blank lines in a run. Every
    non-blank line must match exactly and every blank the model quoted must
    exist; the match maps back to the original span, so the run's extra
    blank lines are consumed with it."""
    if not any(not line.strip() for line in part_lines):
        return None
    whole_c, index = _collapse_blank_runs(whole_lines)
    part_c, _ = _collapse_blank_runs(part_lines)
    n = len(part_c)
    for j in range(len(whole_c) - n + 1):
        if whole_c[j:j + n] != part_c:
            continue
        start = index[j]
        end = index[j + n] if j + n < len(index) else len(whole_lines)
        return "".join(whole_lines[:start] + replace_lines + whole_lines[end:])
    return None


_NUMBERED = re.compile(r"^\s*\d+(?:\||:|\t) ?")


def _strip_line_numbers(lines: list[str]) -> list[str] | None:
    """Tier 5: drop a uniform line-number prefix when every non-blank line
    carries one; ``None`` when the block is not numbered."""
    content = [ln for ln in lines if ln.strip()]
    if not content or not all(_NUMBERED.match(ln) for ln in content):
        return None
    return [_NUMBERED.sub("", ln, count=1) if ln.strip() else ln for ln in lines]


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

    res = _replace_with_collapsed_blank_runs(whole_lines, part_lines, replace_lines)
    if res is not None:
        return res

    stripped = _strip_line_numbers(part_lines)
    if stripped is not None and stripped != part_lines:
        # new_text copied from the same numbered output loses its prefixes too.
        stripped_replace = _strip_line_numbers(replace_lines) or replace_lines
        return replace_most_similar_chunk(whole, "".join(stripped), "".join(stripped_replace))
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


# ---------------------------------------------------------------------------
# Elided quotes
# ---------------------------------------------------------------------------

# Phrases that mark a deliberate omission, from Gemini CLI's
# omissionPlaceholderDetector (github.com/google-gemini/gemini-cli, Apache-2.0).
_OMISSION_PHRASES = (
    "rest of",
    "remaining",
    "unchanged",
    "same as before",
    "as above",
    "and so on",
    "etc.",
    "existing code",
    "previous code",
    "no changes",
)

# A CODE line that trails off: "contact_id:..." / "contact = rel...".
# Requires the dots glued to a non-space, non-dot character, which is what
# separates an abbreviation from legitimate Python: `Field(...)`, `Query(...)`,
# a Protocol stub `def f() -> int: ...`, or numpy `a[..., 0]` all have the dots
# preceded by a space, an opening bracket or a comma. A GLUED colon does count:
# `contact_id:...` is the live 2026-09-18 shape, while the stub writes `: ...`.
_GLUED_ELLIPSIS_RE = re.compile(r"[^\s.,(\[{=]\.\.\.\s*$", re.MULTILINE)

# A line that is ONLY "..." means "the rest is unchanged"; aider treats the
# same shape as an elision marker. Legal Python as a stub body, so it costs 2
# false positives across besser/ -- worth it: an edit carrying one spliced a
# second `try:` into an open one and left a module that fails to import.
_BARE_ELLIPSIS_RE = re.compile(r"^[ 	]*\.\.\.[ 	]*$", re.MULTILINE)


def find_elision(text: str) -> tuple[int, str] | None:
    """First line of *text* that abbreviates rather than quotes, or None.

    Returns ``(line_number, line)`` 1-based. Deliberately narrow: calibrated
    over 2,077 files / 400,777 lines of this repo, it flags one prose docstring
    and nothing else. A comment merely STARTING with "..." is not an elision --
    that is this codebase's own continuation style.
    """
    for n, line in enumerate(text.splitlines(), 1):
        if _GLUED_ELLIPSIS_RE.search(line) or _BARE_ELLIPSIS_RE.match(line):
            return n, line
        if "..." in line:
            low = line.lower()
            if any(p in low for p in _OMISSION_PHRASES):
                return n, line
    return None


def elided_lines(text: str) -> set[str]:
    """Every elided line in *text*, stripped, for comparing two sides of an edit.

    ``find_elision`` answers "does this abbreviate?"; a rewrite that preserves
    an ellipsis the file already had is legitimate, so the guard needs to know
    *which* lines rather than merely whether any exist.
    """
    found = set()
    for line in text.splitlines():
        if (_GLUED_ELLIPSIS_RE.search(line) or _BARE_ELLIPSIS_RE.match(line)
                or ("..." in line and any(p in line.lower() for p in _OMISSION_PHRASES))):
            found.add(line.strip())
    return found
