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

Scope: this is the backend generator's LOCAL detector, used only on the
``modify()`` path — from-scratch ``run()`` never touches it. Unlike the
modeling-agent's ``_FIX_INTENT_RE`` (which only picks confirmation copy), it
gates the run, so it needs evidence of a failure, not failure vocabulary.

Design rule (shared with ``contract_checks``): **precision over recall**.
A target we cannot ground in concrete vocabulary falls back to a soft
target, which frames the loop honestly without inventing a specific
defect to chase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from besser.spec_driven_agent.planning.requirements_ledger import original_request


# The modeling-agent sends its own summary first (machine prose such as "show
# clear error messages", "return 404 for unknown tasks"), then the user's words
# (``original_request``), each later message under this heading. Only the
# user's newest message can report a failure.
_LATER_REQUEST_MARKER = "## Subsequent user request, verbatim"

# Evidence that the user reports a failure of the existing app. Feature wording
# is not evidence: "show a validation error", "a fixed daily rate", "return
# 404 for unknown ids", "regenerate the app" (#659).
_FAILURE_EVIDENCE_RE = re.compile(
    "|".join((
        # A pasted traceback, stack trace or tool error.
        r"traceback \(most recent call last\)|\bstack\s?trace\b|\bFile \"[^\"]+\", line \d+",
        r"\bnpm (?:ERR!|error)|\bHTTP/\d(?:\.\d)? [45]\d\d\b|\berror\s*:\s*\S",
        # The app misbehaving.
        r"\b(?:does|do|did|is|are|was)(?:n['’]?t| not)\s+(?:work|load|start|run|open|compile|build|save|respond)",
        r"\bwon['’]?t\s+(?:work|load|start|run|open|compile|build|save|respond)",
        r"\bnot\s+working\b|\bcrash(?:es|ed|ing)\b|\bnothing happens\b|\binternal server error\b",
        r"\bblank (?:white )?(?:page|screen)\b|\bwhite screen\b|\bsame error\b|\b(?:error|[45]\d\d)\s+again\b",
        r"\b(?:is|are|was|were|seems|looks|still|['’]s)\s+(?:completely\s+|totally\s+|now\s+)?broken\b",
        r"\b(?:it|this|build|tests?|install|request|import|migration|login|app|server|backend"
        r"|frontend|page|call|endpoint|compilation|deploy(?:ment)?|command)\s+"
        r"(?:still\s+|keeps\s+|always\s+|now\s+|is\s+|are\s+)?(?:fails|failed|failing)\b",
        r"\bfail(?:s|ed|ing)?\s+with\b",
        # The user meeting an error, or asking for a fix.
        r"\b(?:i|we)\s+(?:get|got|keep getting|am getting|are getting|see|saw|hit|ran into|run into"
        r"|receive|received)\s+(?:an?\s+|this\s+|the\s+|that\s+|some\s+|following\s+)*(?:\w+\s+){0,2}errors?\b",
        r"\bgetting\s+(?:an?\s+|this\s+|the\s+)?(?:\w+\s+)?errors?\b",
        # Not the CSS sense: "fix the header to the top".
        r"\bfix(?:ing)?\s+(?:it|this|that|these|those|them|the|my|our|all)\b(?!(?:\s+\w+){1,3}\s+(?:to|at)\b)",
        r"\b(?:please|pls|can you|could you)\s+fix\b",
        r"\bthere(?:['’]s| is| are)\s+(?:an?\s+|some\s+)?bugs?\b|\bbuggy\b",
        r"\b[45]\d\ds\b",
    )),
    re.IGNORECASE,
)
# A pasted exception class (``NameError``, ``ValidationError``): case-sensitive,
# and not when the request specifies it ("raise a ValueError if ...").
_EXCEPTION_EVIDENCE_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception)\b")
# A status code the app returned ("POST /x returns 400", "a 500 error"), not
# one it should return ("Return 404 for unknown tasks", "a custom 404 page").
_STATUS_REPORT_RE = re.compile(
    r"\b(?:returns|returned|returning|gives|gave|got|get|getting|responds?\s+with)\s+"
    r"(?:an?\s+|http\s+|status\s+)*[45]\d\d\b"
    r"(?!\s+(?:if|for|when(?!\s+I\b)|unless|on\s+(?:unknown|missing|invalid)))"
    r"|\b[45]\d\d (?:error|bad request|internal server error|not found|unprocessable)"
    r"|\b(?:error|errors|status)\s+(?:code\s+)?[45]\d\d\b",
    re.IGNORECASE,
)
# Words that make the next match a specification ("raise a ValueError if",
# "should return 404", "log every 500 error"), not a report.
_SPECIFIED_BEFORE_RE = re.compile(
    r"\b(?:raise|throw|return|should|must|shall|will|to|every|each|any|all|custom)\s+(?:an?\s+)?$",
    re.IGNORECASE,
)


def _reported(pattern: re.Pattern, text: str) -> bool:
    return any(
        not _SPECIFIED_BEFORE_RE.search(text[max(0, m.start() - 20):m.start()])
        for m in pattern.finditer(text)
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


def users_latest_words(instructions: str) -> str:
    """The user's newest message inside the modeling-agent's framing."""
    return original_request(instructions).rsplit(_LATER_REQUEST_MARKER, 1)[-1]


def looks_like_fix_request(instructions: str) -> bool:
    """True when the user's newest message reports a failure of the app."""
    text = users_latest_words(instructions)
    return bool(
        _FAILURE_EVIDENCE_RE.search(text)
        or _reported(_EXCEPTION_EVIDENCE_RE, text)
        or _reported(_STATUS_REPORT_RE, text)
    )


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


def _mentions_form(lowered_text: str, form: str) -> bool:
    """``form`` as a whole word in already-lowercased text."""
    return bool(re.search(
        rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", lowered_text
    ))


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
    if not looks_like_fix_request(instructions):
        return None
    text = users_latest_words(instructions).strip()

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
        if any(_mentions_form(low_text, form) for form in _forms(cls)):
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

    Each form must match as a whole word: as a bare substring ``Book``
    would match inside ``booking`` and point the fix loop at the wrong
    entity. ``_`` and punctuation count as boundaries, so ``book_id`` and
    ``routers/book.py`` still match.
    """
    if target is None or not target.entities:
        return False
    low = (finding_message or "").lower()
    if not low:
        return False
    for entity in target.entities:
        for form in _forms(entity):
            if form and _mentions_form(low, form):
                return True
    return False
