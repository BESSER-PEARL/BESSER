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

Tiers 6-7 port one comparator and one boundary rule from sst/opencode's
``edit.ts`` (github.com/sst/opencode, MIT) that aider does not have - found by
direct comparison against it, 2026-09-19:

6. leading AND trailing whitespace forgiven per line, not just leading -
   opencode's ``LineTrimmedReplacer``. Tier 2's ``lstrip()`` leaves a quoted
   line that differs from the file ONLY in trailing whitespace unmatched at
   every tier; Jinja-generated scaffold lines often carry it.
7. the spurious-blank-line tolerance of tier 3 is symmetric - a model pads
   the TRAILING end of the block, or both, just as often as the leading end
   that aider issue #25 covers - opencode's ``TrimmedBoundaryReplacer``.

Tier 8 is ours, calibrated 2026-09-20 over the 411 refused ``old_text``
values of 197 completed runs:

8. a wrong indent on the quote's FIRST line alone is forgiven, lines 2..n
   matching exactly. Of the 128 Qwen refusals tiers 1-7 cannot apply, 20 have
   a window matching modulo whitespace; 18 are this one shape - a decorator
   quoted at indent 4 above a body at 0, because the model reconstructs the
   opening line from a prior instead of copying it - 2 are irregular, and NONE
   is a uniform shift, so tier 2 never sees them. gpt-5.6 produces the shape
   0 times in 116 refusals. The tier rescues 21 (the 18, plus 3 mixed-numbered
   quotes whose unnumbered first line kept the model's own indent), each
   byte-identical to what an exact apply of the same edit produces, and 17 of
   the 18 Python ones would NOT parse if the replacement's first line were
   written verbatim - which is why the prefix comes off new_text too, and why
   a replacement that does not carry it is refused rather than guessed at.
   Over 4.7M line windows of besser/ and 4,712 generated app files a quote
   built this way never landed on a window other than its own; 283k ambiguous
   ones are refused by require_unique. One run resent the same six edits from
   turn 19 to turn 38 and landed none of them.

   7 of the 21 still leave a file that does not parse, unchanged by this tier:
   those quotes stop one line short of a docstring's closing triple quote, a
   separate defect that the same-turn write diagnostics name and the
   "old_text not found" message could not.

   The tier originally forgave OVER-indentation only, on the principle that
   it should never add indent it invented. Widened 2026-09-20 to the signed
   difference: of the three quotes left in the 79-case reconstructed-state
   replay set that a window matches modulo whitespace, two are the mirror
   shape - a JSX tag quoted one space short of where it sits, a def quoted a
   level short of its body - and tier 2/6 cannot take either, because the
   shift is not uniform and their single-prefix rule rejects the window. The
   whitespace restored is the FILE's own, sliced off the line being matched,
   never invented. The third stays refused: its replacement is authored a
   level shallower than the body it would replace, which is the guard in
   ``_first_line_replacement``. Re-calibrated over 300 generated app files /
   36,357 sampled windows in both directions: zero landings on a window other
   than the quote's own.

One tier that looks obvious is deliberately absent. 11 refused quotes spell a
regex with a doubled backslash where the file has one, and it is tempting to
un-double and retry. BESSER's own pydantic generator emits the same regex
twice one line apart - raw in the check, re-escaped inside the error message -
so a quote spanning both carries BOTH conventions and no whole-quote
un-doubling is right; replayed against the three live cases it rescues none.
An un-doubled ``new_text`` would also write a DIFFERENT regex into a file that
still parses, which is the silent-corruption failure this module refuses
elsewhere. ``describe_escape_mismatch`` names the mistake instead.

Three tiers are deliberately NOT ported. Opencode's
``WhitespaceNormalizedReplacer`` collapses runs of internal whitespace, which
also collapses them inside a string literal: quoting ``BANNER = "Room 101"``
against ``BANNER = "Room    101"`` matches, and the replacement then rewrites
the literal the model never saw. Measured against 34 refused ``old_text``
values from eleven live Qwen runs, it rescued none of them, so it carries a
real corruption risk for no observed gain here. And two of aider's own:

* Its ``SequenceMatcher`` similarity tier (0.8 ratio) — aider disabled it in
  v0.11.2 because silently applying an 80%-similar edit is worse than a clean
  failure with a good hint. We agree, and we have the hint (below).
* Its ``...`` elision tier — in aider ``...`` is prose-level edit-block syntax,
  but ``...`` on its own line is a *valid Python statement* (Protocol/stub
  bodies, ``def f(): ...``). Treating it as an elision marker risks a spurious
  match in exactly the code we generate, for little gain in a JSON tool arg.

``find_similar_lines`` is the other half of the technique: on a miss, show the
model the closest actual lines so its retry can copy them verbatim.
``locate_anchored_span`` goes further where it safely can: it returns the LINE
RANGE the quote brackets, so the executor hands back a pre-filled
``replace_file_lines`` instead of a second guess. Measured over the 70 refused
``old_text`` values from 21 live runs that the ladder above still cannot apply,
it locates 31 (44%), and 20 of the 28 whose file is unchanged since the refusal
(71%). One of the 31 closes on a later duplicate of the last anchor line - a
span that starts right and stops short. Tolerable only because it never
applies anything.
"""

from __future__ import annotations

import re

from difflib import SequenceMatcher


class AmbiguousEdit(ValueError):
    """More than one eligible window matched at the same apply tier."""


def _prep(text: str) -> tuple[str, list[str]]:
    """Normalize to a trailing newline and split keeping line endings."""
    if text and not text.endswith("\n"):
        text += "\n"
    return text, text.splitlines(keepends=True)


def _perfect_replace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str],
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
) -> str | None:
    """Exact line-window match. First occurrence only (same as ``replace(.., 1)``)."""
    part_tup = tuple(part_lines)
    n = len(part_lines)
    if not n:
        return None
    result = None
    for i in range(len(whole_lines) - n + 1):
        if (tuple(whole_lines[i:i + n]) == part_tup
                and not _protected_window(whole_lines, i, i + n, protected_spans)):
            if result is not None:
                raise AmbiguousEdit("Multiple exact line windows")
            result = "".join(whole_lines[:i] + replace_lines + whole_lines[i + n:])
            if not require_unique:
                return result
    return result


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
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str],
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
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
    result = None
    for i in range(len(whole_lines) - n + 1):
        add = _match_but_for_leading_whitespace(whole_lines[i:i + n], part_lines)
        if add is None:
            continue
        if _protected_window(whole_lines, i, i + n, protected_spans):
            continue
        if result is not None:
            raise AmbiguousEdit("Multiple indentation-corrected windows")
        fixed = [add + r if r.strip() else r for r in replace_lines]
        result = "".join(whole_lines[:i] + fixed + whole_lines[i + n:])
        if not require_unique:
            return result
    return result


def _uniform_indent_prefix(whole_lines: list[str], part_lines: list[str]) -> str | None:
    """The single common extra-indent prefix ``whole_lines`` carries over
    ``part_lines``, given a match already known to be equal up to whitespace.

    Shared by the trim tier below - same rule as
    ``_match_but_for_leading_whitespace``: exactly one uniform prefix, widening
    only (the file has never LESS indent than the quote), else ``None``.
    """
    prefixes = set()
    for w, p in zip(whole_lines, part_lines):
        if not w.strip():
            continue
        delta = (len(w) - len(w.lstrip())) - (len(p) - len(p.lstrip()))
        if delta < 0:
            return None
        prefixes.add(w[:delta])
    return prefixes.pop() if len(prefixes) == 1 else None


def _match_but_for_trim(whole_lines: list[str], part_lines: list[str]) -> str | None:
    """Like ``_match_but_for_leading_whitespace``, but a full ``.strip()`` on
    both sides instead of ``lstrip()`` only - opencode's ``LineTrimmedReplacer``
    (MIT). A quoted line differing from the file ONLY in trailing whitespace
    (Jinja-generated scaffolds carry it) matches nothing upstream: tier 2's
    ``lstrip()`` leaves the trailing difference in place."""
    n = len(whole_lines)
    if n != len(part_lines):
        return None
    if not all(whole_lines[i].strip() == part_lines[i].strip() for i in range(n)):
        return None
    return _uniform_indent_prefix(whole_lines, part_lines)


def _first_line_indent_delta(
    whole_lines: list[str], part_lines: list[str]
) -> int | None:
    """Signed leading-whitespace difference on the quote's FIRST line alone.

    Tier 8. Lines 2..n must match the window under tier 6's rule with no
    indent widening at all; only then is the first line matched after
    absorbing a whitespace difference the rest of the quote does not have.
    Positive means the quote is over-indented against the file, negative
    under-indented. ``None`` when the shape does not apply.
    """
    n = len(whole_lines)
    if n != len(part_lines) or n < 2:
        return None
    w0, p0 = whole_lines[0], part_lines[0]
    if not p0.strip() or w0.strip() != p0.strip():
        return None
    delta = (len(p0) - len(p0.lstrip())) - (len(w0) - len(w0.lstrip()))
    if delta == 0:
        return None
    if _match_but_for_trim(whole_lines[1:], part_lines[1:]) != "":
        return None
    return delta


def _base_indent(lines: list[str]) -> int | None:
    """Smallest leading-whitespace width over the non-blank lines, or None."""
    widths = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    return min(widths) if widths else None


def _first_line_replacement(
    delta: int, whole_first: str, part_lines: list[str], replace_lines: list[str],
) -> list[str] | None:
    """``replace_lines`` with its first line re-aligned to the file, or None.

    Both directions require the model to have repeated its own mistake in
    ``new_text``; a replacement that disagrees with the quote is refused
    rather than guessed at, because an edit landing at the wrong indent is
    worse than one that misses.
    """
    if not replace_lines or not replace_lines[0].strip():
        return None
    if delta > 0:
        # Over-indented. The prefix comes off new_text too: 17 of the 18
        # Python cases this tier was calibrated on would not parse if the
        # replacement's first line were written verbatim.
        if replace_lines[0][:delta].strip():
            return None
        return [replace_lines[0][delta:]] + replace_lines[1:]
    # Under-indented: the file carries -delta MORE whitespace than the quote.
    add = whole_first[: -delta]
    if add.strip():
        return None
    # (a) the replacement's opening line must sit where the quote's did, and
    if (len(replace_lines[0]) - len(replace_lines[0].lstrip())
            != len(part_lines[0]) - len(part_lines[0].lstrip())):
        return None
    # (b) its body must not sit SHALLOWER than the body the quote matched -
    #     that is a replacement authored against a different indent level, and
    #     shifting only its first line would splice it in broken.
    part_body, replace_body = _base_indent(part_lines[1:]), _base_indent(replace_lines[1:])
    if part_body is not None and replace_body is not None and replace_body < part_body:
        return None
    return [add + replace_lines[0]] + replace_lines[1:]


def _replace_with_first_line_indent(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str],
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
) -> str | None:
    """Tier 8: forgive a wrong indent on the quote's first line only.

    The model reconstructs the opening line from a prior instead of copying it
    - a decorator quoted at indent 4 above a body at 0, because "decorated
    functions are methods and methods are indented", or a JSX tag quoted one
    space short of where it sits. Both directions occur; see
    ``_first_line_replacement`` for what the replacement must satisfy.
    """
    n = len(part_lines)
    if n < 2 or not replace_lines or not replace_lines[0].strip():
        return None
    result = None
    for i in range(len(whole_lines) - n + 1):
        delta = _first_line_indent_delta(whole_lines[i:i + n], part_lines)
        if delta is None:
            continue
        fixed = _first_line_replacement(delta, whole_lines[i], part_lines, replace_lines)
        if fixed is None:
            continue
        if _protected_window(whole_lines, i, i + n, protected_spans):
            continue
        if result is not None:
            raise AmbiguousEdit("Multiple first-line-indent-corrected windows")
        result = "".join(whole_lines[:i] + fixed + whole_lines[i + n:])
        if not require_unique:
            return result
    return result


def _replace_with_normalized_lines(
    matcher, whole_lines: list[str], part_lines: list[str], replace_lines: list[str],
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
    ambiguous_msg: str = "Multiple normalized windows",
) -> str | None:
    """Slide a window, testing each with ``matcher`` (one of the two
    functions above), and re-indent the replacement by its returned prefix."""
    n = len(part_lines)
    if not n:
        return None
    result = None
    for i in range(len(whole_lines) - n + 1):
        add = matcher(whole_lines[i:i + n], part_lines)
        if add is None:
            continue
        if _protected_window(whole_lines, i, i + n, protected_spans):
            continue
        if result is not None:
            raise AmbiguousEdit(ambiguous_msg)
        fixed = [add + r if r.strip() else r for r in replace_lines]
        result = "".join(whole_lines[:i] + fixed + whole_lines[i + n:])
        if not require_unique:
            return result
    return result


def _perfect_or_whitespace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str],
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
) -> str | None:
    result = _perfect_replace(whole_lines, part_lines, replace_lines, protected_spans, require_unique)
    if result is not None:
        return result
    result = _replace_with_missing_leading_whitespace(
        whole_lines, part_lines, replace_lines, protected_spans, require_unique,
    )
    if result is not None:
        return result
    # Tier 6: leading AND trailing whitespace forgiven per line.
    result = _replace_with_normalized_lines(
        _match_but_for_trim, whole_lines, part_lines, replace_lines, protected_spans, require_unique,
        "Multiple trim-normalized windows",
    )
    return result


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
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str],
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
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
    result = None
    for j in range(len(whole_c) - n + 1):
        if whole_c[j:j + n] != part_c:
            continue
        start = index[j]
        end = index[j + n] if j + n < len(index) else len(whole_lines)
        if _protected_window(whole_lines, start, end, protected_spans):
            continue
        if result is not None:
            raise AmbiguousEdit("Multiple blank-run-normalized windows")
        result = "".join(whole_lines[:start] + replace_lines + whole_lines[end:])
        if not require_unique:
            return result
    return result


_NUMBERED = re.compile(r"^\s*(\d+)(?:\||:|\t) ?")
# read_file's own ``NNN| `` form. Only this one is trusted on a MIXED block:
# ``1: "one"`` in a dict literal matches the general pattern.
_PIPE_NUMBERED = re.compile(r"^\s*(\d+)\| ?")


def _strip_line_numbers(lines: list[str]) -> list[str] | None:
    """Tier 5: drop a line-number prefix the quote was copied with; ``None``
    when the block is not numbered.

    Every non-blank line numbered is the safe case. Mixed numbering is real
    too - 11 inputs across 23 runs carried prefixes on all but one line, and
    one of them was WRITTEN, baking ``NNN| `` into a shipped Booking.tsx. On a
    mixed block only read_file's ``NNN| `` form counts, only as a majority of
    at least two lines, and only with rising numbers. Calibrated over
    2.27M line windows of besser/ and 23 generated apps: zero matches in text
    that was not already numbered output.

    "Rising" was "strictly rising" until 2026-09-20. Qwen transcribes the
    gutter by hand and sometimes drops a digit - live Booking.tsx turn 13
    quoted ``228| 229| 30| 231|``, twice in 105 pairs - and an all-or-nothing
    rule threw away the other 104 good prefixes and refused a 107-line quote
    over two typos. A gutter still has to RISE, just not perfectly: a
    MAJORITY of the steps must climb. The real guard was never this rule but
    the one above it - most content lines carrying ``NNN| `` at all, which
    unnumbered source does not do.
    Measured over the whole corpus, 181 Qwen payloads carry a gutter and this
    tier already stripped 178 of them; that run was one of the 3 it did not.
    """
    content = [ln for ln in lines if ln.strip()]
    if not content:
        return None
    if all(_NUMBERED.match(ln) for ln in content):
        return [_NUMBERED.sub("", ln, count=1) if ln.strip() else ln for ln in lines]
    marked = [(i, m) for i, ln in enumerate(lines) if (m := _PIPE_NUMBERED.match(ln))]
    if len(marked) < 2 or len(marked) * 2 < len(content):
        return None
    numbers = [int(m.group(1)) for _, m in marked]
    pairs = len(numbers) - 1
    rising = sum(1 for a, b in zip(numbers, numbers[1:]) if b > a)
    # A MAJORITY of the steps must rise. Two numbers still have to rise
    # outright (1 pair: 0 rising fails, 1 passes), so nothing that used to be
    # accepted on a short block is loosened, and exactly-half (10 3 11 2 12)
    # is still refused.
    if pairs and rising * 2 <= pairs:
        return None
    numbered = {i for i, _ in marked}
    return [_PIPE_NUMBERED.sub("", ln, count=1) if i in numbered else ln
            for i, ln in enumerate(lines)]


def _protected_window(lines, start, end, protected_spans) -> bool:
    """Whether a line window is inside an already-completed replacement."""
    if not protected_spans:
        return False
    offset = sum(map(len, lines[:start]))
    # _prep adds a synthetic final newline. Ignore only that line terminator
    # when comparing boundaries, just as the apply ladder does.
    stop = offset + len("".join(lines[start:end]).rstrip("\r\n"))
    return any(lo <= offset and stop <= hi for lo, hi in protected_spans)


def replacement_spans(whole: str, replacement: str) -> tuple[tuple[int, int], ...]:
    """Locate completed replacement regions, not just replacement text anywhere.

    Used to exclude *enclosed* search anchors from replay. No edit history is
    needed, so this works after resume and leaves unrelated pending sites
    editable. Only exact text, uniform indentation and copied line numbers
    qualify; similarity scoring and arbitrary whitespace stripping do not.
    """
    if not replacement.strip():
        return ()
    _, lines = _prep(whole)
    _, quoted = _prep(replacement)
    stripped = _strip_line_numbers(quoted)
    if stripped is not None:
        replacement = "".join(stripped)
        quoted = stripped
    spans = set()
    start = whole.find(replacement)
    while start >= 0:
        spans.add((start, start + len(replacement)))
        start = whole.find(replacement, start + 1)
    leading = [len(line) - len(line.lstrip()) for line in quoted if line.strip()]
    indent = min(leading, default=0)
    unindented = [line[indent:] if line.strip() else line for line in quoted]
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    n = len(quoted)
    for i in range(len(lines) - n + 1):
        if _match_but_for_leading_whitespace(lines[i:i + n], unindented) is not None:
            spans.add((offsets[i], min(offsets[i + n], len(whole))))
    return tuple(sorted(spans))


def replace_most_similar_chunk(
    whole: str, part: str, replace: str,
    protected_spans: tuple[tuple[int, int], ...] = (),
    require_unique: bool = False,
) -> str | None:
    """Return the edited file text, or ``None`` when no tier matched.

    Pure: never touches disk, never mutates its arguments.

    With require_unique, ambiguity is rejected at every tier, following the
    candidate/uniqueness contract of OpenCode's edit.ts (MIT). We deliberately
    retain BESSER's conservative matching tiers, not its similarity fallbacks.
    """
    whole, whole_lines = _prep(whole)
    part, part_lines = _prep(part)
    replace, replace_lines = _prep(replace)

    res = _perfect_or_whitespace(whole_lines, part_lines, replace_lines, protected_spans, require_unique)
    if res is not None:
        return res

    # Models sometimes prepend a blank line to the block (aider issue #25).
    if len(part_lines) > 2 and not part_lines[0].strip():
        res = _perfect_or_whitespace(whole_lines, part_lines[1:], replace_lines, protected_spans, require_unique)
        if res is not None:
            return res

    # ...or append one, or both (opencode's TrimmedBoundaryReplacer, MIT).
    if len(part_lines) > 2 and not part_lines[-1].strip():
        res = _perfect_or_whitespace(whole_lines, part_lines[:-1], replace_lines, protected_spans, require_unique)
        if res is not None:
            return res

    if len(part_lines) > 2 and not part_lines[0].strip() and not part_lines[-1].strip():
        res = _perfect_or_whitespace(whole_lines, part_lines[1:-1], replace_lines, protected_spans, require_unique)
        if res is not None:
            return res

    res = _replace_with_collapsed_blank_runs(whole_lines, part_lines, replace_lines, protected_spans, require_unique)
    if res is not None:
        return res

    res = _replace_with_first_line_indent(
        whole_lines, part_lines, replace_lines, protected_spans, require_unique,
    )
    if res is not None:
        return res

    stripped = _strip_line_numbers(part_lines)
    if stripped is not None and stripped != part_lines:
        # new_text copied from the same numbered output loses its prefixes too.
        stripped_replace = _strip_line_numbers(replace_lines) or replace_lines
        return replace_most_similar_chunk(
            whole, "".join(stripped), "".join(stripped_replace), protected_spans, require_unique,
        )
    return None


def locate_chunk(whole: str, part: str) -> int | None:
    """1-based line where the lines of ``part`` appear consecutively in
    ``whole``, each line's surrounding whitespace aside, else ``None``.

    For the already-applied checks in ``_modify_file``. Those were byte-exact
    and went blind after every tier-2 apply: run 4efe04ff (2026-09-18) landed
    a block quoted at indent 4 in a file that holds it at 12, re-sent the same
    call and was told "old_text not found". Diagnostic only, never an edit, so
    it is looser than the ladder: a model's copy of a block differs from the
    file's mostly in indentation.
    """
    whole_lines = [line.strip() for line in whole.split("\n")]
    quoted = part.strip("\n").split("\n")
    part_lines = [line.strip() for line in (_strip_line_numbers(quoted) or quoted)]
    n = len(part_lines)
    for i in range(len(whole_lines) - n + 1):
        if whole_lines[i:i + n] == part_lines:
            return i + 1
    return None


def locate_anchored_span(whole: str, part: str) -> tuple[int, int] | None:
    """1-based inclusive line span of ``whole`` that ``part`` brackets, else None.

    The candidate rule of sst/opencode's ``BlockAnchorReplacer``
    (``tool/edit.ts``, MIT): ``part``'s first and last non-empty lines must
    equal candidate lines exactly after ``.strip()``, and the candidate's line
    count must be within 25% of ``part``'s. Two candidates return None rather
    than a guess - opencode scores them by similarity, we refuse.

    A LOCATOR, never an applier. The span is handed to the model so it can
    author the replacement itself against ``replace_file_lines``; see the
    module docstring for why similarity-driven *application* stays out.
    """
    part_lines = part.split("\n")
    part_lines = _strip_line_numbers(part_lines) or part_lines
    filled = [i for i, line in enumerate(part_lines) if line.strip()]
    if not filled:
        return None
    # Compare against the quote's core: boundary blank lines are padding the
    # model added (tiers 3/7), not part of the block it meant to name.
    core = part_lines[filled[0]:filled[-1] + 1]
    n = len(core)
    first, last = core[0].strip(), core[-1].strip()

    whole_lines = whole.split("\n")
    lo = max(1, (3 * n + 3) // 4)   # ceil(0.75n)
    hi = (5 * n) // 4               # floor(1.25n)
    found: tuple[int, int] | None = None
    for i, line in enumerate(whole_lines):
        if line.strip() != first:
            continue
        for j in range(i + lo - 1, min(i + hi, len(whole_lines))):
            if whole_lines[j].strip() != last:
                continue
            if found is not None:
                return None
            found = (i + 1, j + 1)
    return found


_ESCAPE_HINT = (
    "old_text escapes its backslashes twice: it has {q} where line {n} of the "
    "file has {f}. The JSON encoder already escapes them, so copy the line "
    "exactly as read_file printed it - one backslash, not two."
)


def describe_escape_mismatch(whole: str, part: str) -> str | None:
    """Name a doubled-backslash quote that the file spells with one, or None.

    A DIAGNOSTIC, not a tier. Applying the correction is unsafe here in a way
    the other tiers are not: BESSER's pydantic generator emits the same regex
    twice one line apart, raw in the check and re-escaped in the message ::

        if not (re.fullmatch(r'^[^\\s@]+@...', v) is not None):
            raise ValueError("email must match '^[^\\\\s@]+@...'")

    so a quote spanning both carries BOTH conventions and no whole-quote
    un-doubling can be right. Worse, an un-doubled ``new_text`` would write
    ``r'[^\\\\s@]'`` - valid Python, valid regex, and a different regex from
    the one the model read. 11 refused quotes across the 2026-09-20 corpus
    carry the shape and a global un-double rescues none of them, so the model
    gets told what it did instead.
    """
    if "\\\\" not in part:
        return None
    file_lines = whole.splitlines()
    stripped = {ln.strip(): n for n, ln in enumerate(file_lines, 1)}
    for q in part.splitlines():
        qs = q.strip()
        if "\\\\" not in qs or qs in stripped:
            continue
        n = stripped.get(qs.replace("\\\\", "\\"))
        if n:
            return _ESCAPE_HINT.format(q=qs[:80], n=n, f=file_lines[n - 1].strip()[:80])
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
