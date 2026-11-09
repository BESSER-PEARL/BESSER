"""Reported-failure parsing for FIX/MODIFY runs.

When a user pastes a traceback or reports a broken endpoint and asks for a
fix, that reported failure IS the run's success criterion. Whether the
newest message reports a failure, and which entity it is about, is decided
by the modify run's routing call (``is_failure_report`` / ``reported_entity``
on ``choose_generator``). This module turns that report into a concrete,
matchable :class:`ReportedTarget` so the orchestrator can:

* feed the concrete defect into the customize loop as an explicit
  instruction (not only the user's paraphrase);
* promote the tool's OWN structural findings that match the target from
  warning to blocker, so the fix loop is driven to resolve them;
* gate the run's success on the target actually being addressed.

Scope: used only on the ``modify()`` path; from-scratch ``run()`` never
touches it.

Design rule (shared with ``contract_checks``): **precision over recall**.
Entities are model classes, handler nouns (``createWatchlist``) or the
routing call's ``reported_entity``, never bare URL segments such as ``api``
or ``v1``. A target with no entity is soft: it frames the loop honestly
without inventing a defect to chase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from besser.spec_driven_agent.planning.requirements_ledger import latest_user_message

# A reported HTTP status; the last one in the message is the observed one
# ("unknown ids should return 404, but POST /loans returns 500").
_STATUS_RE = re.compile(r"(?<![/\w])([45]\d\d)\b")


# HTTP method + path, e.g. ``POST /createWatchlist`` or ``GET /watchlists/{id}``.
_HTTP_METHOD_PATH_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\b\s*(/[\w\-/{}.:]*)",
    re.IGNORECASE,
)
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


def _class_entities(candidates: set[str], class_names: list[str]) -> list[str]:
    """The model classes the candidate nouns name (singular/plural forms), sorted.

    Canonical class names line up with the acceptance matrix, which is keyed
    on them.
    """
    class_by_form: dict[str, str] = {}
    for cls in class_names:
        for form in _forms(cls):
            class_by_form[form] = cls
    matched: list[str] = []
    for cand in sorted(candidates):
        cls = next((class_by_form[sc] for sc in _singular_candidates(cand) if sc in class_by_form), None)
        if cls and cls not in matched:
            matched.append(cls)
    return matched


def parse_reported_target(
    instructions: str, domain_model=None, reported_entity: str = "",
) -> ReportedTarget:
    """Parse a reported failure into a :class:`ReportedTarget`.

    Call it only for a failure report (the routing call's
    ``is_failure_report``); ``reported_entity`` is that call's answer. Always
    returns a target, possibly soft.
    """
    text = latest_user_message(instructions)
    class_names = _canonical_class_names(domain_model)
    statuses = _STATUS_RE.findall(text)
    status = statuses[-1] if statuses else None

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

    # A handler name names its entity (``createWatchlist``) even outside the
    # model; a path segment or a word counts only when it names a model class.
    handler_nouns = {m.group(2) for m in _HANDLER_CAMEL_RE.finditer(text)}
    handler_nouns |= {m.group(1) for m in _HANDLER_SNAKE_RE.finditer(text)}
    candidates = set(handler_nouns)
    if path:
        candidates |= {seg.strip("{}") for seg in path.split("/") if seg.strip("{}")}
    low_text = text.lower()
    candidates |= {cls for cls in class_names
                   if any(_mentions_form(low_text, form) for form in _forms(cls))}
    if reported_entity.strip():
        candidates.add(reported_entity.strip())

    classes = _class_entities(candidates, class_names)
    # Unmatched nouns the user or the routing call named, after the classes.
    named = [n for n in sorted(handler_nouns) + [reported_entity.strip()]
             if n and not _class_entities({n}, class_names)]
    entities = tuple(dict.fromkeys(classes + named))

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
