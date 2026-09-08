"""Reported-failure parsing for FIX/MODIFY runs.

When a user pastes a traceback or reports a broken endpoint and asks for a
fix, that reported failure IS the run's success criterion. This module
turns the user's words into a concrete, matchable :class:`ReportedTarget`
so the orchestrator can:

* feed the concrete defect into the customize loop as an explicit
  instruction (not only the user's paraphrase);
* promote the tool's OWN structural findings that match the target from
  warning to blocker, so the fix loop is driven to resolve them;
* gate the run's success on the target actually being addressed.

Scope: this is the backend generator's LOCAL detector. It mirrors the
*spirit* of the modeling-agent's ``_looks_like_fix_request`` /
``_FIX_INTENT_RE`` (a separate codebase) rather than importing it. It is
used only on the ``modify()`` path — from-scratch ``run()`` never touches
it, so first-generation behaviour is unchanged.

Design rule (shared with ``contract_checks``): **precision over recall**.
A target we cannot ground in concrete vocabulary falls back to a soft
target, which frames the loop honestly without inventing a specific
defect to chase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Strong, unambiguous fix / error vocabulary — mirrors the modeling-agent's
# ``_FIX_INTENT_RE`` in spirit (fix/error/traceback/exception/bug/broken/
# crash/fails/not working/regenerate) plus a bare HTTP 4xx/5xx status code.
_FIX_INTENT_RE = re.compile(
    r"\b(?:fix|fixes|fixed|fixing|"
    r"error|errors|traceback|stack\s?trace|exception|"
    r"bug|bugs|broken|crash(?:es|ed|ing)?|"
    r"fails?|failing|failed|"
    r"not\s+working|does(?:n['’]?t| not)\s+work|"
    r"regressed?|returns?\s+(?:a\s+)?[45]\d{2}|"
    r"update\s+the\s+code|re-?generate|re-?run)\b"
    r"|\b[45]\d{2}\b"
    # A bare exception-class token (``NameError``, ``ValidationError``) — a
    # pasted traceback often names the class without the standalone word
    # "error". The suffix must attach to a longer identifier, so the plain
    # word "error"/"exception" is handled by the alternation above, not here.
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b",
    re.IGNORECASE,
)

# HTTP method + path, e.g. ``POST /createWatchlist`` or ``GET /watchlists/{id}``.
_HTTP_METHOD_PATH_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\b\s*(/[\w\-/{}.:]*)",
    re.IGNORECASE,
)
# Bare HTTP status (already gated to 4xx/5xx by the intent regex).
_STATUS_RE = re.compile(r"\b([45]\d{2})\b")
# camelCase handler, e.g. ``createWatchlist`` / ``getWatchlistItems``.
_HANDLER_CAMEL_RE = re.compile(
    r"\b(create|get|update|delete|list|post|put|patch|add|remove|fetch|save)"
    r"([A-Z][A-Za-z0-9]+)\b"
)
# snake_case handler, e.g. ``create_watchlist``.
_HANDLER_SNAKE_RE = re.compile(
    r"\b(?:create|get|update|delete|list|post|put|patch|add|remove|fetch|save)"
    r"_([a-z][a-z0-9_]+)\b"
)
# Python / JS exception class, e.g. ``NameError`` / ``ValidationError``.
_EXCEPTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Warning))\b")
# A source file reference, e.g. ``forecast.py`` / ``src/App.tsx``.
_SOURCE_FILE_RE = re.compile(r"\b([\w./\\-]+\.(?:py|js|jsx|ts|tsx))\b")

# Verb prefixes stripped off a handler tail to recover the entity noun.
_VERB_PREFIXES = (
    "create", "get", "update", "delete", "list", "post", "put",
    "patch", "add", "remove", "fetch", "save",
)


@dataclass(frozen=True)
class ReportedTarget:
    """A user-reported failure, parsed into matchable facts.

    ``kind`` records the strongest signal found (priority: http > exception
    > entity > soft). Every other field is best-effort and may be empty —
    ``entities`` is the one the matcher keys on, so a target with no
    entities behaves like a soft target for matching purposes even if
    ``kind`` is more specific.
    """

    raw: str
    kind: str  # "http" | "exception" | "entity" | "soft"
    descriptor: str
    status: str | None = None
    method: str | None = None
    path: str | None = None
    exception: str | None = None
    source_file: str | None = None
    entities: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_soft(self) -> bool:
        return self.kind == "soft" or not self.entities


def looks_like_fix_request(instructions: str) -> bool:
    """True when the instructions carry fix / error / failure vocabulary."""
    return bool(_FIX_INTENT_RE.search(instructions or ""))


def _forms(name: str) -> set[str]:
    """Lowercase spellings a generated app plausibly uses for an entity.

    Mirrors ``acceptance._entity_forms`` so a target derived here matches
    the acceptance-matrix wording (``entity Watchlist``) it is compared to.
    """
    low = name.lower()
    forms = {low, f"{low}s"}
    if low.endswith("y"):
        forms.add(f"{low[:-1]}ies")
    if low.endswith("s"):
        forms.add(f"{low}es")
    if low.endswith("ies"):
        forms.add(f"{low[:-3]}y")
    return {f for f in forms if f}


def _singular_candidates(token: str) -> set[str]:
    """A token plus a couple of naive de-pluralisations."""
    low = re.sub(r"[^a-z0-9]", "", token.lower())
    out = {low} if low else set()
    if low.endswith("ies"):
        out.add(low[:-3] + "y")
    if low.endswith("es"):
        out.add(low[:-2])
    if low.endswith("s") and len(low) > 1:
        out.add(low[:-1])
    return {c for c in out if c}


def _canonical_class_names(domain_model) -> list[str]:
    if domain_model is None:
        return []
    try:
        return [
            c.name for c in domain_model.get_classes()
            if getattr(c, "name", None)
        ]
    except Exception:
        return []


def _derive_entities(
    candidates: set[str], class_names: list[str],
) -> tuple[str, ...]:
    """Map raw candidate nouns onto canonical model class names.

    A candidate that matches a class name (via singular/plural forms) is
    canonicalised to that class name (so it lines up with the acceptance
    matrix, which is keyed on class names). Candidates that match no class
    are still kept — the reported failure may name a feature that never
    became a class, and matching on the raw noun is better than nothing.
    """
    class_by_form: dict[str, str] = {}
    for cls in class_names:
        for form in _forms(cls):
            class_by_form[form] = cls

    # Class-matched entities lead (they line up with the acceptance matrix
    # and read best in the descriptor); unmatched raw nouns trail.
    matched: list[str] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for cand in sorted(candidates):
        matched_class = None
        for sc in _singular_candidates(cand):
            if sc in class_by_form:
                matched_class = class_by_form[sc]
                break
        chosen = matched_class or cand
        key = chosen.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        (matched if matched_class else unmatched).append(chosen)
    return tuple(matched + unmatched)


def parse_reported_target(
    instructions: str, domain_model=None,
) -> ReportedTarget | None:
    """Parse a user-reported failure into a :class:`ReportedTarget`.

    Returns ``None`` when the instructions are not a fix request. Always
    returns a target (possibly soft) when they are — the caller decides
    what to do with a soft target.
    """
    text = (instructions or "").strip()
    if not looks_like_fix_request(text):
        return None

    class_names = _canonical_class_names(domain_model)

    status = None
    m_status = _STATUS_RE.search(text)
    if m_status:
        status = m_status.group(1)

    method = path = None
    m_http = _HTTP_METHOD_PATH_RE.search(text)
    if m_http:
        method = m_http.group(1).upper()
        path = m_http.group(2) or None

    exception = None
    m_exc = _EXCEPTION_RE.search(text)
    if m_exc:
        exception = m_exc.group(1)

    source_file = None
    m_file = _SOURCE_FILE_RE.search(text)
    if m_file:
        source_file = m_file.group(1).replace("\\", "/")

    # Collect candidate entity nouns from handler names, the endpoint path,
    # and any class name mentioned verbatim.
    candidates: set[str] = set()
    for m in _HANDLER_CAMEL_RE.finditer(text):
        candidates.add(m.group(2))
    for m in _HANDLER_SNAKE_RE.finditer(text):
        candidates.add(m.group(1))
    if path:
        for seg in path.split("/"):
            seg = seg.strip("{}")
            if seg and not seg.isdigit():
                candidates.add(seg)
    low_text = text.lower()
    for cls in class_names:
        if any(form in low_text for form in _forms(cls)):
            candidates.add(cls)

    entities = _derive_entities(candidates, class_names)

    # Strongest-signal-first classification.
    if status or path or method:
        kind = "http"
    elif exception:
        kind = "exception"
    elif entities:
        kind = "entity"
    else:
        kind = "soft"

    descriptor = _build_descriptor(
        kind, method, path, status, exception, source_file, entities,
    )

    return ReportedTarget(
        raw=text[:500],
        kind=kind,
        descriptor=descriptor,
        status=status,
        method=method,
        path=path,
        exception=exception,
        source_file=source_file,
        entities=entities,
    )


def _build_descriptor(
    kind, method, path, status, exception, source_file, entities,
) -> str:
    """A short human phrase naming the reported failure, for run messages."""
    if kind == "http":
        parts = []
        if method and path:
            parts.append(f"{method} {path}")
        elif path:
            parts.append(path)
        elif method:
            parts.append(f"a {method} request")
        else:
            parts.append("the reported request")
        if status:
            parts.append(f"returning HTTP {status}")
        entity_note = (
            f" (entity {entities[0]})" if entities else ""
        )
        return " ".join(parts) + entity_note
    if kind == "exception":
        where = f" in {source_file}" if source_file else ""
        return f"{exception}{where}"
    if kind == "entity":
        return f"the {entities[0]} feature"
    return "the runtime failure you reported"


def finding_matches_target(
    finding_message: str, target: ReportedTarget,
) -> bool:
    """True when a validator finding is about the reported target.

    Keys on the target's entities: the acceptance matrix reports
    ``entity Watchlist — ...`` and the data-contract lint references
    ``routers/book.py`` / ``Book.id``, so an entity-form substring match
    connects the reported failure to the tool's own structural finding.

    A soft target (no entities) matches nothing — we never promote a
    finding we cannot attribute to what the user actually reported.
    """
    if target is None or not target.entities:
        return False
    low = (finding_message or "").lower()
    if not low:
        return False
    for entity in target.entities:
        for form in _forms(entity):
            if form and form in low:
                return True
    return False
