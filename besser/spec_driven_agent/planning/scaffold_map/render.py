"""Text renderings of a :class:`WorkspaceMap`.

* :func:`render_overview` — line-free, budgeted; goes into the system prompt and
  the gap analyzer, which are built once per run, so it names files, symbols,
  routes and anchor ids but never line numbers.
* :func:`render_locate` — the ``locate_code`` tool's answer: current line
  ranges, ``read_file`` offsets, a verbatim quote to anchor ``modify_file`` on,
  and who owns each region. Built from disk on every call.
"""

from __future__ import annotations

import posixpath
import re
from typing import Callable, Optional

from besser.spec_driven_agent.planning.scaffold_map.model import (
    Anchor,
    EntityMap,
    Recipe,
    ScaffoldMap,
    WorkspaceMap,
)

OVERVIEW_FOOTER = (
    "Line numbers are deliberately absent here: they change as you edit. "
    "Call locate_code (entity=, path= or query=) for the current line range, "
    "read_file offset and a verbatim line to anchor modify_file on."
)


def _derived_entity_summary(entity: EntityMap) -> list[str]:
    by_file: dict[str, list[str]] = {}
    for anchor in entity.anchors:
        by_file.setdefault(anchor.file, []).append(anchor.label)
    return [f"{path}: {', '.join(labels)}" for path, labels in by_file.items()]


def _entity_line(entity: EntityMap, compact: bool) -> str:
    if compact:
        files = sorted({a.file for a in entity.anchors})
        line = f"- {entity.name}: {', '.join(files) if files else 'no code found'}"
    else:
        parts = entity.summary or _derived_entity_summary(entity)
        line = f"- {entity.name}: " + ("; ".join(parts) if parts else "no code found")
    if entity.missing:
        line += f" — MISSING: {', '.join(entity.missing)}"
    return line


def _entities_collapsed(m: ScaffoldMap) -> str:
    """One line naming every entity; the pieces still missing stay visible."""
    names = ", ".join(m.entities)
    missing = [f"{e.name} ({', '.join(e.missing)})" for e in m.entities.values() if e.missing]
    line = f"Entities ({len(m.entities)}): {names}"
    if missing:
        line += f"\nMISSING on disk: {'; '.join(missing)}"
    return line


def _clip_line(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _clip_recipe(recipe: Recipe, limit: int) -> str:
    """A recipe line of at most ``limit`` chars, cut between steps, never inside an anchor id."""
    steps = list(recipe.steps)
    text = f"- {recipe.intent}: " + " ".join(steps)
    while len(text) > limit and len(steps) > 1:
        steps.pop()
        text = f"- {recipe.intent}: " + " ".join(steps) + " …"
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 1)].rsplit(" ", 1)[0]
    if cut.count("`") % 2:
        cut = cut[: cut.rfind("`")].rstrip()
    return cut + "…"


# Detail levels of one map's overview, most detailed first. Each drops or
# shortens the part that matters least for "where does new code go":
#   0  everything
#   1  without the named-anchor list, the do-not-edit list and the file roles
#   2  entity lines reduced to their files
#   3  entities collapsed to one line of names (missing pieces kept)
#   4  recipes and pitfalls shortened to one clipped line each, at most 4 pitfalls
_LEVELS = 5
# "line 12" in any position of a syntax error ("... on line 368", "SyntaxError line 5: ...").
_BROKEN_LINE_RE = re.compile(r",? (?:on )?line \d+")
_MAX_BROKEN_SHOWN = 10
# Below this many characters a map is named, not rendered.
_MIN_MAP_SHARE = 160


def _render_map(m: ScaffoldMap, level: int) -> str:
    where = f"`{m.root}/`" if m.root else "the workspace root"
    parts = [f"### {m.title} — {where}"]
    if m.summary and level < 1:
        parts.append("\n".join(f"- {line}" for line in m.summary))
    if m.entities:
        if level >= 3:
            parts.append(_entities_collapsed(m))
        else:
            parts.append("\n".join(["Per entity:"] + [_entity_line(e, compact=level >= 2)
                                                       for e in m.entities.values()]))
    if m.warnings:
        warnings = m.warnings if level < 4 else m.warnings[:4]
        lines = [f"- {_clip_line(w, 200) if level >= 4 else w}" for w in warnings]
        if len(warnings) < len(m.warnings):
            lines.append(f"- … {len(m.warnings) - len(warnings)} more (locate_code lists them all)")
        parts.append("Pitfalls found in this code:\n" + "\n".join(lines))
    if m.recipes:
        lines = ["Where new code goes:"]
        for recipe in m.recipes:
            text = f"- {recipe.intent}: " + " ".join(recipe.steps)
            lines.append(_clip_recipe(recipe, 220) if level >= 4 else text)
        parts.append("\n".join(lines))
    if level < 1:
        app_anchors = [a for a in m.anchors if a.kind != "file"]
        if app_anchors:
            parts.append("\n".join(["Named anchors (pass the id as locate_code query= for lines):"]
                                   + [f"- `{a.id}` — {a.file}: {a.label}" for a in app_anchors]))
        if m.do_not_edit:
            parts.append("Shared runtime files — configure them through props, do not edit: "
                         + ", ".join(m.do_not_edit))
    return "\n\n".join(parts)


def _hard_cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 60)].rsplit("\n", 1)[0]
    return f"{cut}\n… (cut to fit; locate_code has the rest)"


def _allocate(maps: list[ScaffoldMap], full: list[str], available: int) -> list[int]:
    """Character share per map (see :func:`render_overview`)."""
    gap = 2  # the blank line between maps
    floor = [len(_render_map(m, _LEVELS - 1)) for m in maps]
    if sum(f + gap for f in floor) > available:
        # Not even the compact renderings fit. Water-fill: the smallest maps
        # get their whole compact rendering first, the large ones share the
        # rest, so a big backend cannot push a small frontend out entirely.
        shares = [0] * len(maps)
        left = available
        order = sorted(range(len(maps)), key=lambda i: floor[i])
        for n, i in enumerate(order):
            fair = left // (len(order) - n)
            shares[i] = min(floor[i] + gap, fair) - gap
            left -= shares[i] + gap
        return shares
    shares = list(floor)
    spare = available - sum(f + gap for f in floor)
    wants = [max(0, len(t) - f) for t, f in zip(full, floor)]
    # Hand out the spare room in proportion to each map's extra detail, capped
    # at what it can use; repeat so room a map cannot use flows to the others.
    for _ in range(len(maps)):
        open_ = [i for i, w in enumerate(wants) if w > 0]
        if spare <= 0 or not open_:
            break
        weight = sum(wants[i] for i in open_)
        given = 0
        for i in open_:
            extra = min(wants[i], int(spare * wants[i] / weight))
            shares[i] += extra
            wants[i] -= extra
            given += extra
        spare -= given
        if given == 0:
            break
    return shares


def render_overview(ws: WorkspaceMap, max_chars: int = 4000) -> str:
    """Line-free overview for the system prompt / planner, at most ``max_chars``.

    Every map first gets the room its most compact rendering needs; the rest
    of the budget is shared out in proportion to how much more detail each
    map has, so one large map cannot starve the others. Each map is then
    rendered at the most detailed level that fits its share, and only cut on
    a line boundary past the last level. Unparseable files and the footer are
    always kept.
    """
    if not ws.maps and not ws.broken:
        return ""
    head = ""
    if ws.broken:
        shown = ws.broken[:6]
        # The error's line number is left to locate_code / check_syntax, like
        # every other line number: it moves as soon as the file is edited.
        lines = [f"- {_clip_line(_BROKEN_LINE_RE.sub('', b), 200)}" for b in shown]
        if len(ws.broken) > len(shown):
            lines.append(f"- … {len(ws.broken) - len(shown)} more")
        head = "Files that do not parse — fix these first:\n" + "\n".join(lines) + "\n\n"
    if ws.notes:
        head += "Coverage: " + _clip_line("; ".join(ws.notes), 240) + "\n\n"
    fixed = len(head) + len(OVERVIEW_FOOTER) + 2
    available = max(0, max_chars - fixed)

    full = [_render_map(m, 0) for m in ws.maps]
    rendered: list[str] = []
    if sum(len(t) + 2 for t in full) <= available:
        rendered = full
    else:
        shares = _allocate(ws.maps, full, available)
        levels: dict[int, int] = {}
        texts: dict[int, str] = {}
        skipped: list[str] = []
        for i, (m, share) in enumerate(zip(ws.maps, shares)):
            if share < _MIN_MAP_SHARE and len(_render_map(m, _LEVELS - 1)) > share:
                # Too little room to say anything useful: name the root instead.
                skipped.append(m.root or "(workspace root)")
                continue
            for level in range(_LEVELS):
                texts[i], levels[i] = _render_map(m, level), level
                if len(texts[i]) <= share:
                    break
            if len(texts[i]) > share:
                texts[i] = _hard_cut(texts[i], share)
        # A map that dropped a level for a few characters leaves its share
        # partly unused: spend the leftover on upgrading maps, most cut first.
        left = available - sum(len(t) + 2 for t in texts.values())
        if skipped:
            left -= 80 + sum(len(r) + 2 for r in skipped)
        # Levels are not strictly decreasing in size (compact entity lines can
        # be longer than dropped anchor lists), so try every richer level.
        upgraded = True
        while upgraded:
            upgraded = False
            for i in sorted(levels, key=lambda k: -levels[k]):
                for level in range(levels[i]):
                    better = _render_map(ws.maps[i], level)
                    if len(better) - len(texts[i]) <= left:
                        left -= len(better) - len(texts[i])
                        texts[i], levels[i] = better, level
                        upgraded = True
                        break
        rendered = [texts[i] for i in sorted(texts)]
        if skipped:
            rendered.append(f"Also mapped (ask locate_code with path=): {', '.join(skipped)}")
    body = "\n\n".join(t for t in rendered if t)
    text = f"{head}{body}\n\n{OVERVIEW_FOOTER}" if body else f"{head}{OVERVIEW_FOOTER}"
    if len(text) > max_chars:
        text = _hard_cut(f"{head}{body}", max(0, max_chars - len(OVERVIEW_FOOTER) - 2)) + f"\n\n{OVERVIEW_FOOTER}"
    return text[:max_chars]


# Words that say nothing about WHERE ("where do I add a field to Book?").
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "to", "in", "into", "of", "for", "do", "does", "i", "we", "my", "where", "how",
    "what", "which", "should", "can", "is", "it", "on", "at", "and", "or", "with", "add", "new",
    "go", "goes", "put", "code", "implement", "make", "create", "write", "change", "file",
})


def _words(text: str) -> set[str]:
    """Lower-case identifier words: ``execute_book_applyDiscount`` -> execute, book, apply, discount."""
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[a-z0-9]+", split.lower()))


def _haystack(anchor: Anchor) -> str:
    return " ".join(filter(None, (anchor.id, anchor.label, anchor.file, anchor.quote, anchor.hint,
                                  anchor.entity, anchor.status)))


def _match_query(anchors: list[Anchor], query: str) -> tuple[list[Anchor], str]:
    """Anchors matching ``query``, best first, and how they matched.

    An anchor id (or id prefix) wins outright; then anchors containing every
    meaningful word of the query as a whole word; then anchors ranked by how
    many of those words they contain, so a question in plain words still
    lands somewhere useful.
    """
    q = query.strip()
    by_id = [a for a in anchors if a.id.lower() == q.lower()] or \
        [a for a in anchors if a.id.lower().startswith(q.lower())]
    if by_id:
        return by_id, "id"
    words = [w for w in _words(q) if w not in _QUERY_STOPWORDS] or list(_words(q))
    if not words:
        return [], "none"
    scored = []
    for order, anchor in enumerate(anchors):
        hay = _words(_haystack(anchor))
        score = sum(1 for w in words if w in hay)
        if score:
            scored.append((-score, order, anchor))
    if not scored:
        return [], "none"
    scored.sort()
    if -scored[0][0] == len(words):
        return [a for score, _, a in scored if -score == len(words)], "all words"
    return [a for _, _, a in scored], "ranked by matched words"


def _render_anchor(anchor: Anchor, owner: Optional[str]) -> str:
    span = (f"line {anchor.start_line}" if anchor.start_line == anchor.end_line
            else f"lines {anchor.start_line}-{anchor.end_line}")
    count = anchor.end_line - anchor.start_line + 1
    lines = [
        f"[{anchor.id}] {anchor.label}"
        + (f" ({anchor.status})" if anchor.status else ""),
        f"  {anchor.file} {span} (read_file offset={anchor.start_line - 1} limit={count})"
        + (f" · owner: {owner}" if owner else ""),
    ]
    if anchor.quote:
        # Verbatim, indentation included: everything after "quote: " is the
        # text to pass as modify_file's old_text.
        lines.append(f"  quote: {anchor.quote}")
    if anchor.hint:
        lines.append(f"  {anchor.hint}")
    return "\n".join(lines)


def _normalise_path(path: Optional[str]) -> Optional[str]:
    """A workspace-relative '/'-path for a filter, or None when no path was given.

    Accepts what read_file accepts: './x', 'x/', 'a//b', 'a/../b', backslashes.
    An absolute path is expected to be made relative by the caller.
    """
    if path is None or not path.strip():
        return None
    p = path.strip().replace("\\", "/")
    p = posixpath.normpath(p)
    p = p.lstrip("/")
    return "" if p in (".", "") else p


def render_locate(
    ws: WorkspaceMap,
    *,
    entity: Optional[str] = None,
    path: Optional[str] = None,
    query: Optional[str] = None,
    owner_of: Optional[Callable[[str, int, int], Optional[str]]] = None,
    max_chars: int = 7000,
) -> dict:
    """The ``locate_code`` answer as a JSON-able dict.

    Filters combine (AND). With no filter, returns the app-level anchors, the
    entity names and the pitfalls, so a first call shows what to ask for next.
    """
    entity = (entity or "").strip() or None
    path = _normalise_path(path)
    query = (query or "").strip() or None
    anchors = ws.all_anchors()
    filtered = bool(entity or path or query)
    match_mode = ""
    if entity:
        wanted = entity.lower()
        anchors = [a for a in anchors if (a.entity or "").lower() == wanted]
    if path is not None:
        roots = [m for m in ws.maps if m.root == path]
        if roots:
            # A map's own root: that map, not the maps nested below it.
            own = {id(a) for m in roots for a in m.all_anchors()}
            anchors = [a for a in anchors if id(a) in own]
        elif path:
            anchors = [a for a in anchors if a.file == path or a.file.startswith(path + "/")]
    if query:
        anchors, match_mode = _match_query(anchors, query)
    if not filtered:
        anchors = [a for m in ws.maps for a in m.anchors]

    blocks: list[str] = []
    used = 0
    shown = 0
    for anchor in anchors:
        owner = owner_of(anchor.file, anchor.start_line, anchor.end_line) if owner_of else None
        block = _render_anchor(anchor, owner)
        if used + len(block) > max_chars and shown:
            break
        blocks.append(block)
        used += len(block) + 1
        shown += 1

    result: dict = {"locations": "\n".join(blocks) if blocks else "", "count": len(anchors)}
    if match_mode == "ranked by matched words":
        result["match"] = "no place matches every word; best partial matches first"
    if shown < len(anchors):
        result["truncated"] = (
            f"showing {shown} of {len(anchors)}; narrow with entity=, path= or query="
        )
    if not anchors:
        known = ws.entity_names()
        result["locations"] = "no mapped location matches"
        if known:
            result["entities"] = known
    if not filtered:
        result["entities"] = ws.entity_names()
        warnings = [w for m in ws.maps for w in m.warnings]
        if warnings:
            result["pitfalls"] = warnings
    broken = [b for b in ws.broken if path is None or not path or b.startswith(path)]
    if broken:
        result["broken"] = broken[:_MAX_BROKEN_SHOWN] + (
            [f"… {len(broken) - _MAX_BROKEN_SHOWN} more"] if len(broken) > _MAX_BROKEN_SHOWN else [])
    missing = {e.name: e.missing for m in ws.maps for e in m.entities.values() if e.missing}
    if missing and (not filtered or entity):
        result["missing"] = {k: v for k, v in missing.items()
                             if not entity or k.lower() == entity.strip().lower()}
    if not ws.maps:
        result["locations"] = "no scaffold structure recognised in this workspace"
    return result
