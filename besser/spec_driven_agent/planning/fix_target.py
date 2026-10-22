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

import logging
import re
from dataclasses import dataclass, field

from besser.spec_driven_agent.planning.requirements_ledger import latest_user_message

logger = logging.getLogger(__name__)

_I = re.IGNORECASE
_STATUS = r"[45]\d\d"
# After a status code: a condition or audience makes it a specification
# ("returns 404 for unknown ids", "returns 403 to guests"), except "for an
# existing / a valid ..." which describes the failing input.
_NOT_SPECIFIED_AFTER = (
    r"(?!(?:\s+(?:conflict|not found|bad request|unauthori[sz]ed|forbidden|unprocessable(?: entity)?))?"
    r"\s+(?:if|unless|whenever|to\b|when(?!\s+(?:I|we)\b|\s+\w+ing\b)|on\s+(?:unknown|missing|invalid|bad|duplicate)"
    r"|for(?!\s+(?:an?\s+|the\s+|my\s+)?(?:existing|valid|correct)\b)))"
)
# Subjects that "fail" in a report ("the build fails", "npm run dev fails").
_FAILING_SUBJECT = (
    r"(?:it|this|build|tests?|install|request|import|migration|login|app|server|backend|frontend|page"
    r"|call|endpoint|compilation|deploy(?:ment)?|command|dev|start|up)"
)

# Evidence that the user's newest message reports a failure of the existing
# app, as (name, pattern, guarded). Feature wording is not evidence: "show a
# validation error", "a fixed daily rate", "return 404 for unknown ids",
# "regenerate the app" (#659). A guarded rule does not count right after a
# word that makes it a specification (``_SPECIFIED_BEFORE_RE``).
_EVIDENCE_RULES: tuple[tuple[str, re.Pattern, bool], ...] = (
    ("traceback", re.compile(
        r"traceback \(most recent call last\)|\bstack\s?trace\b|\bFile \"[^\"]+\", line \d+", _I), False),
    ("tool log line", re.compile(
        rf"\bnpm (?:ERR!|error\b)|\bHTTP/\d(?:\.\d)? {_STATUS}\b|\bERR_[A-Z_]{{4,}}\b|\bECONNREFUSED\b"), False),
    ("request log line", re.compile(
        rf"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/\S*\s*(?:->|→|=>|:|-)?\s*{_STATUS}\b", _I), False),
    ("error line", re.compile(r"\berror\s*:\s*\S", _I), True),
    # Case-sensitive: a class name such as ``NameError`` or ``ValidationError``.
    ("exception class", re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception)\b"), True),
    ("status returned", re.compile(
        r"\b(?:returns|returned|returning|gives|gave|got|get|getting|says|said|shows|showed"
        r"|respond(?:s|ed)?\s+with)\s+"
        rf"(?:an?\s+|http\s+|status\s+(?:code\s+|of\s+)?)*{_STATUS}\b{_NOT_SPECIFIED_AFTER}", _I), True),
    ("status code", re.compile(
        rf"\b{_STATUS} (?:error|bad request|internal server error|not found|unprocessable)"
        rf"|\b(?:error|errors|status)\s+(?:code\s+)?{_STATUS}\b{_NOT_SPECIFIED_AFTER}", _I), True),
    ("status plural", re.compile(rf"\b{_STATUS}s\b", _I), False),
    ("status now", re.compile(
        rf"\b{_STATUS}\s+(?:now|instead)\b|\b{_STATUS}\s+(?:page|error)\s+(?:shows up|appears|pops up)\b", _I),
     True),
    ("status after a failure", re.compile(
        r"\b(?:fail(?:s|ed|ing)?|crash(?:es|ed|ing)?|errors?\s+out)\s+with\s+"
        rf"(?:an?\s+|http\s+|status\s+(?:code\s+)?)*{_STATUS}\b", _I), False),
    ("repeated failure", re.compile(
        rf"\bstill\s+(?:an?\s+|getting\s+(?:an?\s+)?)?(?:{_STATUS}|errors?)\b|\bsame error\b"
        rf"|\b(?:error|{_STATUS})\s+again\b", _I), False),
    ("does not work", re.compile(
        r"\b(?:does|do|did|is|are|was)(?:n['’]?t| not)\s+(?:work|load|start|run|open|compile|build|save|respond)"
        r"|\bwon['’]?t\s+(?:work|load|start|run|open|compile|build|save|respond)|\bnot\s+working\b"
        r"|\bdoes nothing\b|\b(?:does|did)(?:n['’]?t| not)\s+do\s+anything\b"
        r"|\bcan(?:['’]?t|not)\s+(?:reach|connect to)\s+the\s+(?:api|backend|server|database)\b", _I), False),
    ("crash", re.compile(
        r"\bcrash(?:es|ed|ing)\b|\bnothing happen(?:s|ed)\b|\binternal server error\b|\berrors?\s+out\b"
        r"|\b(?:app|server|backend|frontend|page|it)\s+crash\b(?!-)"
        r"|\bfreez(?:es|ing)\b|\bfroze\b|\bdisappeared\b|\bmiscalculat\w*", _I), False),
    # Guarded: "if an error occurs, show a toast".
    ("error occurs", re.compile(
        r"\b(?:an?\s+)?(?:error|exception)s?\s+(?:occurs|occurred|appears|appeared|pops up|popped up|shows up"
        r"|showed up|happens|happened|is thrown|was thrown)\b"
        r"|(?:^|[.!?]\s+)(?:an?\s+)?(?:error|exception)\s+when\s+(?:I\b|we\b|\w+ing\b)", _I), True),
    ("stuck loading", re.compile(
        r"\b(?:keeps|kept)\s+(?:loading|spinning|restarting|crashing|failing|reloading)\b"
        r"|\b(?:loading|spinning)\s+forever\b", _I), False),
    ("regression", re.compile(
        r"(?:n['’]t|\bnot|\bnever)\b[^.!?\n]{0,40}\banymore\b"
        r"|\bno longer\s+(?:works?|loads?|saves?|shows?|starts?|appears?)\b"
        r"|\bsince (?:your|the) last change\b|\bthat['’]?s (?:wrong|not right)\b", _I), False),
    # Guarded: "if the list is empty, show a placeholder".
    ("empty view", re.compile(
        r"\b(?:page|list|table|dropdown|column|form|chart|dashboard)\s+(?:is|are|loads?|stays|comes up)\s+"
        r"(?:still\s+|now\s+|completely\s+|always\s+)?empty\b", _I), True),
    ("broken value", re.compile(
        r"\b(?:shows?|showing|says|displays?|prints?)\s+(?:up\s+)?(?:as\s+)?['\"]?"
        r"(?:NaN|undefined|Invalid Date|\[object Object\])|:\s*NaN\b"), True),
    ("blank page", re.compile(
        r"\bblank (?:white )?(?:page|screen)\b|\bwhite screen\b"
        r"|\b(?:page|screen|app|frontend|it|dashboard)\s+(?:is|stays|remains)\s+(?:just\s+|still\s+)?blank\b",
        _I), False),
    ("broken", re.compile(
        r"\b(?:is|are|was|were|seems|looks|still|['’]s)\s+(?:completely\s+|totally\s+|now\s+)?broken\b"
        r"|\b(?:you|it|this|that)\s+(?:just\s+)?broke\b|\b(?:app|server|backend|frontend|page|login|site)\s+broken\b",
        _I), False),
    # Guarded: "whenever a request fails, show a toast".
    ("fails", re.compile(
        rf"\b{_FAILING_SUBJECT}\s+(?:still\s+|keeps\s+|always\s+|now\s+|is\s+|are\s+)?(?:fails|failed|failing)\b"
        r"|\b(?:creating|saving|deleting|updating|loading|editing|adding|opening|logging in|signing up)\b"
        r"(?:\s+\w+){0,4}?\s+(?:fails|failed)\b"
        r"|\bfailed to (?:fetch|compile)\b", _I), True),
    ("throws", re.compile(
        r"\b(?:throws|threw|raises|raised)\s+(?:an?\s+)?(?:exception|error)\b"
        r"(?!\s+(?:if|when(?!\s+I\b)|whenever|unless|for|on)\b)", _I), False),
    ("cors error", re.compile(r"\bCORS\s+(?:policy\s+)?errors?\b|\bblocked by CORS policy\b", _I), True),
    ("user meets an error", re.compile(
        r"\b(?:i|we)\s+(?:get|got|keep getting|am getting|are getting|see|saw|hit|ran into|run into"
        r"|receive|received)\s+(?:an?\s+|this\s+|the\s+|that\s+|some\s+|following\s+)*(?:\w+\s+){0,2}errors?\b"
        r"|\bgetting\s+(?:an?\s+|this\s+|the\s+)?(?:\w+\s+)?errors?\b", _I), True),
    # Guarded ("let the manager fix the price"), and not the CSS sense ("fix
    # the header to the top").
    ("fix request", re.compile(
        r"\bfix(?:ing)?\s+(?:exactly\s+|only\s+)?(?:it|this|that|these|those|them|the|my|our|all)\b"
        r"(?!(?:\s+\w+){1,3}\s+(?:to|at)\b)", _I), True),
    ("asks for a fix", re.compile(r"\b(?:please|pls|can you|could you|you to)\s+fix\b", _I), False),
    ("bug", re.compile(
        r"\bthere(?:['’]s| is| are)\s+(?:an?\s+|some\s+)?bugs?\b|\bbuggy\b"
        r"|\b(?:that|it|this)(?:['’]s| is)\s+(?:an?\s+)?bug\b", _I), False),
)
# Words that make the next match a specification ("raise a ValueError if",
# "should return 404", "use status code 409", "let the manager fix the price",
# a user story "As an admin, I see errors ...").
_SPECIFIED_BEFORE_RE = re.compile(
    r"\b(?:raise|throw|return|should|must|shall|will|to|every|each|any|all|custom|use|add|avoid|prevent"
    r"|handle|catch|no|show|display|if|when|whenever|unless|can|may)\s+(?:an?\s+|the\s+)?$"
    r"|\b(?:let|lets|allow|allows)\s+(?:\w+\s+){1,3}$|\bas an?\s+\w+(?:\s+\w+)?,?\s+$",
    _I,
)
_SPECIFIED_BEFORE_WINDOW = 30  # chars before a match that may hold the specifying words
_STATUS_IN_MATCH_RE = re.compile(rf"\b({_STATUS})")


def _evidence(text: str) -> list[tuple[str, re.Match]]:
    """Every (rule name, match) that reports a failure, in text order."""
    found = []
    for name, pattern, guarded in _EVIDENCE_RULES:
        for m in pattern.finditer(text):
            before = text[max(0, m.start() - _SPECIFIED_BEFORE_WINDOW):m.start()]
            if not (guarded and _SPECIFIED_BEFORE_RE.search(before)):
                found.append((name, m))
    return sorted(found, key=lambda item: item[1].start())


def _reported_status(evidence: list[tuple[str, re.Match]]) -> str | None:
    """The first status code inside the evidence, never one the text only specifies."""
    for _name, m in evidence:
        status = _STATUS_IN_MATCH_RE.search(m.group(0))
        if status:
            return status.group(1)
    return None


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


def looks_like_fix_request(instructions: str) -> bool:
    """True when the user's newest message reports a failure of the app."""
    return bool(_logged_evidence(latest_user_message(instructions)))


def _logged_evidence(text: str) -> list[tuple[str, re.Match]]:
    evidence = _evidence(text)
    if evidence:
        name, m = evidence[0]
        logger.debug("Fix request: rule %r matched %r", name, m.group(0))
    return evidence


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
    text = latest_user_message(instructions)
    evidence = _logged_evidence(text)
    if not evidence:
        return None

    class_names = _canonical_class_names(domain_model)
    status = _reported_status(evidence)

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
