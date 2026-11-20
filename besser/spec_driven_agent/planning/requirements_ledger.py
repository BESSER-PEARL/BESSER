"""Requirements ledger: what the user asked for, checked against the app.

The gap analyser plans from the spec, but a plan is not a verification: a run
can plan a business rule, ship without it, and still pass every static gate.

Two LLM calls close that gap. Before Phase 3 the user's verbatim request is
turned once into atomic, testable requirements. Each Phase 3 pass then asks
the model to judge every requirement against the generated code and to cite
where it is implemented; the harness re-checks the citation, so a claim that
points at nothing is reported as unverified, never as implemented. Missing
requirements become ``requirement:`` blockers for the fix loop, and every
verdict lands in the recipe so the reviewer sees what was NOT built.
"""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import logging
import os
import posixpath
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple
import re
import time

from pygments.token import Comment, String

from besser.BUML.metamodel.traceability import RequirementKind
from besser.spec_driven_agent.planning.code_text import _lexer_for, code_only
from besser.spec_driven_agent.planning.gap_analyzer import _chat_supports_kwargs, _is_real_provider
from besser.spec_driven_agent.planning.requested_routes import _ANY_VERB as _ROUTE_ANY_VERB
from besser.spec_driven_agent.planning.requested_routes import _PARAM_RE as _ROUTE_PARAM_RE
from besser.spec_driven_agent.planning.requested_routes import _VERBS as _ROUTE_VERBS
from besser.spec_driven_agent.planning.requested_routes import _literal as _route_literal
from besser.spec_driven_agent.planning.requested_routes import _methods as _route_methods
from besser.spec_driven_agent.planning.specification import MAX_REQUIREMENTS, validate_specification
from besser.spec_driven_agent.planning.target import ENTITY_CRUD_KIND
from besser.spec_driven_agent.providers.tool_input import coerce_to_schema
from besser.spec_driven_agent.state.agent_trace import matching_lines
from besser.spec_driven_agent.state.generator_output import GeneratorOutputError, load_generator_output
from besser.spec_driven_agent.validation.write_diagnostics import (
    python_structural_diagnostics, workspace_uses_sqlite,
)
from besser.llm import extract_json
from besser.utilities.workspace_fs import walk_plain

logger = logging.getLogger(__name__)

# The modeling agent appends the user's own text under this heading (see its
# smart_generation_handler.py); everything before it is an LLM summary.
VERBATIM_MARKER = "## The user's original request, verbatim"
# Each later user message follows this heading (modeling-agent
# generation_handler.py), inside the original request.
LATER_REQUEST_MARKER = "## Subsequent user request, verbatim"

_EXTRACTION_INCOMPLETE = (
    "Requirement extraction reached its item limit; coverage of the full original "
    "specification is unverified. Complete extraction and audit the remaining "
    "specification before declaring the application complete. This is a "
    "verification obligation, not an application feature."
)
_STATUSES = ("implemented", "partial", "missing", "unverified")

# The kinds of a requirement the editor sends that the judge checks against the
# code. An entity existing (data) is the model, already generated; the stack
# (technical) and qualities such as performance cannot be judged from source.
JUDGED_KINDS = frozenset({
    RequirementKind.UNIQUENESS.value, RequirementKind.VALIDATION.value, RequirementKind.RULE.value,
    RequirementKind.COMPUTED.value, RequirementKind.TRANSITION.value, RequirementKind.ACTION.value,
    RequirementKind.UI.value, RequirementKind.INTEGRATION.value,
})

# What the judge reads: the schemas and the ORM first (validators, unique
# columns), then the backend code the model wrote or changed (where rules are
# enforced), then generator output it never changed, then the pages. A
# mid-sized app compacts to ~150k chars (~38k tokens); a router with nested
# creation is ~20k on its own, so the per-file cap must hold one.
_DIGEST_MAX_TOTAL_CHARS = 200_000
_DIGEST_MAX_FILE_CHARS = 24_000
# Share of the source budget kept for non-Python source (the screens): a model's
# generated CRUD routers alone can fill the budget and hide every UI file.
_DIGEST_UI_SHARE = 0.3

# For these kinds an "implemented" verdict must cite a line that enforces
# something; otherwise the judge marks uniqueness or capacity rules implemented
# by citing a column or a relationship that enforces nothing.
_ENFORCEMENT_TOKENS = (
    "unique", "primary_key", "raise", "validat", "assert",
    "httpexception", "valueerror", "min_length", "max_length", "pattern",
    "field(", "constr", "check(", "ge=", "le=", "gt=", "lt=",
)
# The extractor still pads the ledger with "a person must have a first name"
# items and may label them 'rule'; a column IS that requirement. The
# enforcing-line demand applies to a 'rule' only when its text states a
# constraint.
# Whole words with their inflections: as substrings "less" matched "lesson"
# and "format" matched "information". A word the requirement lists as a value
# ("one of: positive, mixed, negative") still states the rule that the field
# takes only those values; an enumeration that has them enforces it.
_CONSTRAINT_WORDS = re.compile(
    r"\b(?:unique(?:ly|ness)?|must\s+not|may\s+not|may\s+never|cannot|can\s+not|never|at\s+least|at\s+most"
    r"|no\s+more\s+than|exceed(?:s|ed|ing)?|only\s+(?:if|when|after)|(?:in)?valid(?:ly|ity|at\w+)?"
    r"|format(?:s|ted|ting)?|shap(?:e|es|ed|ing)|(?:mis)?match(?:es|ed|ing)?|overlap(?:s|ped|ping)?|before|after"
    r"|greater|less(?:er)?|between|positive(?:ly)?|non-negative|refus(?:e|es|ed|ing|al)|reject(?:s|ed|ing|ion)?)\b",
    re.IGNORECASE,
)


def _states_constraint(text: str) -> bool:
    return bool(_CONSTRAINT_WORDS.search(text))


def _demands_enforcement(kind: str, text: str) -> bool:
    return kind == "uniqueness" or (kind in ("validation", "rule") and _states_constraint(text))


_READ_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s?")
# A judge abridges a run of code as "a ... b", lists lines as "a; b", and cites
# two places as "x.py: a and y.py: b". Each part must still be verbatim source.
# An ellipsis may carry the comma of the arguments it left out: "f(..., x)".
_ABRIDGED_RE = re.compile(r"\s?(?:\.\.\.|\u2026),?(?=[\s)\]}]|$)|;?\s+and\s+([\w./\\-]+\.\w{1,5}):\s|;\s+(?:and\s+)?")
# "a ... y.py: b" switches file without an "and"; only a judged file's path counts.
_FILE_SWITCH_RE = re.compile(r"(?<!\S)([\w./\\-]+\.\w{1,5}):\s")
# "<Plot region={r} /> which calls ...": an explanation after a closed statement
# or element runs to the next cited file and is not matched.
_EXPLANATION_RE = re.compile(r"(?:(?<=[);\]])|(?<=/>))\s+(?:which|that)\s.*?(?=;?\s+and\s+[\w./\\-]+\.\w{1,5}:\s|$)", re.S)
# A part shorter than this ("pass", "}") identifies no code; it is left out.
_MIN_PART_QUOTE = 8
# Evidence longer than this is cut before matching (the judge is asked for one line).
_MAX_EVIDENCE_CHARS = 2000
# Citations a verdict may give: behaviour that spans a few lines or files.
_MAX_CITATIONS = 6
# A formatted statement, or a def followed by its first body line, spans lines;
# the judge quotes it on one. Lines a quote may span: a Kotlin/Compose call
# puts one argument per line.
_MAX_QUOTE_LINES = 25
# Characters a multi-line quote may leave off its last line: a return
# annotation and colon ("-> List[Row]:"), not a statement's substance.
_MAX_UNQUOTED_TAIL = 24
_BLOCK_PUNCTUATION = frozenset("{(:;,[")


def _normalise_quote(text: str, escapes: bool = True) -> str:
    """The judge's quote, made comparable to file text: literal escape
    sequences become spaces (unless ``escapes`` is off, for a quote of code
    such as ``"\\n".join(...)``), single quotes become double quotes (a judge may
    rewrite ``Mapped_["Booking"]`` as ``Mapped_['Booking']``), and
    only the first quoted line counts - later lines are often abridged."""
    if escapes:
        text = text.replace("\\n", "\n").replace("\\t", " ")
    first = text.strip().split("\n", 1)[0]
    # read_file numbers what the model sees; no real source line starts with
    # "175| ", so such a prefix is always a copied artifact.
    first = _READ_NUMBER_PREFIX_RE.sub("", first, count=1)
    return _collapse(first.replace("'", '"'))


def _normalise_line(line: str) -> str:
    """A source line in a quote's form. Its escapes stay as written: turning
    ``"\\n".join(...)`` into a line break cut the line at that string."""
    return _collapse(line.replace("'", '"'))


def _citation_parts(raw: str, path: str, files: dict[str, str] | None = None) -> list[tuple[str, str, bool]]:
    """(path, quote, cut) for each part of an abridged or listed citation.

    ``cut`` when an ellipsis right after the part marks the rest of its line
    as left out; "and other.py: b", or "other.py: b" naming one of ``files``,
    moves the following parts to that file.
    """
    parts: list[tuple[str, str, bool]] = []
    position = 0
    raw = _EXPLANATION_RE.sub("", raw)
    bounds = [(m.start(), m.end(), m.group(1), m.group().strip().startswith(("...", "\u2026")))
              for m in _ABRIDGED_RE.finditer(raw)] + [(len(raw), len(raw), None, False)]
    for start, end, next_path, cut in bounds:
        segment, offset = raw[position:start], 0
        for switch in _FILE_SWITCH_RE.finditer(segment) if files else ():
            if len(_resolve_cited_path(switch.group(1), "", files, {})) == 1:
                quote = _evidence_quote(segment[offset:switch.start()])
                if len(quote) >= _MIN_PART_QUOTE:
                    parts.append((path, quote, False))
                path, offset = switch.group(1), switch.end()
        quote = _evidence_quote(segment[offset:])
        if len(quote) >= _MIN_PART_QUOTE:
            parts.append((path, quote, cut))
        path = next_path or path
        position = end
    return parts


_TRAILING_COMMA_RE = re.compile(r",(?=[)\]}])")


# A line that opens with a prefixed string piece (``f"{b} ..."``) continuing the last line's.
_PREFIXED_PIECE_RE = re.compile(r'[fFrRbBuU]{1,2}"')


@lru_cache(maxsize=65536)
def _joined_strings(text: str) -> str:
    """Adjacent string literals ``"a " "b"`` read as the one string ``"a b"``, and
    a formatter's trailing comma before a closing bracket is dropped."""
    return _TRAILING_COMMA_RE.sub("", text.replace('""', ""))


_CALL_RE = re.compile(r"[\w.$]+\(.*\);?")
# Shorter than this, a call such as "f(x)" is too generic to identify code.
_MIN_CALL_QUOTE = 12
# Shortest visible start of a line the judge cut with an ellipsis.
_MIN_CUT_QUOTE = 20
# Shortest run copied from inside one line that identifies it on its own
# ("onClick = { selected = if (on) a - b else a + b }" quotes the assignment).
_MIN_INLINE_QUOTE = 30


class _Source(NamedTuple):
    """One cited file, read once per verification."""
    path: str
    lines: list[str]
    normalised: list[str]
    code: list[str]  # normalised lines without comments and strings (Pygments)
    tree: ast.AST | None
    diagnostics: list[dict]
    comments: set[int]
    bare: list[str]  # normalised lines without comments; strings kept
    # normalised lines with each string and comment character replaced by _MASK
    # (same length); empty when the file could not be lexed line for line
    masked: list[str]


_MASK = "\uffff"  # a noncharacter: never in source text


def _masked_lines(source: str, path: str, normalised: list[str]) -> list[str]:
    """``normalised``, with the characters of strings and comments replaced by ``_MASK``."""
    lexer = _lexer_for(Path(path).suffix.lower())
    if lexer is None:
        return normalised
    masked = "".join(re.sub(r"\S", _MASK, value) if token in Comment or token in String else value
                     for token, value in lexer.get_tokens(source))
    lines = [_normalise_line(line) for line in masked.splitlines()]
    return lines if [len(line) for line in lines] == [len(line) for line in normalised] else []


def _in_code(quoted: str, source: _Source, index: int, min_code: int = 1) -> bool:
    """The quote is verbatim in the line, holds at least ``min_code`` characters of its
    code (not of its strings or comment), and starts and ends outside a string or comment.

    Verbatim, strings included: removing strings alone makes CheckConstraint("a > 0")
    and CheckConstraint("b > 0") the same text. The quote itself is not lexed:
    collapsed, ``<th className="num">`` lexed differently from its line (r12-p21).
    A string's text with its closing ``")`` is not code (review: ``...stored.";``).
    """
    line = source.normalised[index]
    if not quoted:
        return False
    if not source.masked:
        code = _collapse(code_only(quoted, source.path))
        return len(code) >= min_code and quoted in line and code in source.code[index]
    mask = source.masked[index]
    at = line.find(quoted)
    while at >= 0:
        end = at + len(quoted)
        inside = (at > 0 and mask[at - 1] == mask[at] == _MASK) or (
            end < len(mask) and mask[end - 1] == mask[end] == _MASK)
        if not inside and len(mask[at:end].replace(_MASK, "")) >= min_code:
            return True
        at = line.find(quoted, at + 1)
    return False


def _string_literal_at(quoted: str, source: _Source, index: int) -> bool:
    """The quote is exactly one or more whole string literals of the line, quotes included."""
    if not source.masked or len(quoted) < 3 or quoted[0] not in "\"`" or quoted[-1] not in "\"`":
        return False
    line, mask = source.normalised[index], source.masked[index]
    at = line.find(quoted)
    while at >= 0:
        end = at + len(quoted)
        if (not mask[at:end].strip(_MASK) and (at == 0 or mask[at - 1] != _MASK)
                and (end == len(mask) or mask[end] != _MASK)):
            return True
        at = line.find(quoted, at + 1)
    return False


def _is_complete_call(quoted: str) -> bool:
    """A whole call expression (balanced, closed), whitespace already removed."""
    return (len(quoted) >= _MIN_CALL_QUOTE and bool(_CALL_RE.fullmatch(quoted))
            and quoted.count("(") == quoted.count(")") and quoted.count("{") == quoted.count("}"))


def _quote_spans(quoted: str, source: _Source, cut: bool = False) -> list[tuple[int, int]]:
    """1-based (first, last) source line spans whose text is the quote, read as
    written or else without comments (a judge leaves out a "# noqa")."""
    spans = list(_spans_in(quoted, source, source.normalised, cut))
    if not spans and source.bare != source.normalised:
        spans = list(_spans_in(quoted, source, source.bare, cut))
    return spans


def _spans_in(quoted: str, source: _Source, normalised: list[str], cut: bool):
    target = _joined_strings(quoted)
    # Counted on the line's own code, not on the quote lexed alone.
    min_code = 1 if _is_complete_call(quoted) else _MIN_INLINE_QUOTE
    for start, first in enumerate(normalised):
        # A line may end in the block it opens ("Card(...) {", "def f():"), which
        # the judge leaves off; nothing else may be missing. A complete call
        # may also be quoted from inside its line ("onPressed: () => save(x),"),
        # and a line's start when an ellipsis says the rest was left out.
        if (first == quoted
                or (first.startswith(quoted) and set(first[len(quoted):]) <= _BLOCK_PUNCTUATION)
                or (cut and len(quoted) >= _MIN_CUT_QUOTE and first.startswith(quoted))
                or _in_code(quoted, source, start, min_code)):
            yield start + 1, start + 1
            continue
        # A first line ending in a string piece ends in its closing quote; the next line's
        # piece joins it (the join below then needs the whole quote).
        if not first or not target.startswith(_joined_strings(first).rstrip('"')):
            continue
        joined = first
        for end in range(start + 1, min(start + _MAX_QUOTE_LINES, len(normalised))):
            earlier = _joined_strings(joined)
            piece = _PREFIXED_PIECE_RE.match(normalised[end]) if joined.endswith('"') else None
            # A formatter's split ``"... " f"{b} ..."`` reads as one string: the seam is
            # removed only here, where one line's piece ends and the next one's starts.
            joined = joined[:-1] + normalised[end][piece.end():] if piece else joined + normalised[end]
            span_text = _joined_strings(joined)
            # The quote may stop inside its last line (a signature quoted without
            # its "-> T:", anywhere when an ellipsis marks the cut), never inside an earlier one.
            if (span_text.startswith(target) and len(target) > len(earlier.rstrip('",'))
                    and (cut or len(span_text) - len(target) <= _MAX_UNQUOTED_TAIL)):
                yield start + 1, end + 1
                break
            # A string piece's closing quote, or a trailing comma, waits for what
            # follows on the next line (its partner string, the closing bracket).
            if not target.startswith(span_text.rstrip('",')):
                break


def _evidence_quote(text: str, escapes: bool = True) -> str:
    """Keep a decorated function's identity instead of a shared registration line.

    The judge sometimes quotes a block despite the one-line contract. Taking
    only ``@model_validator(mode='after')`` confused a no-op capacity validator
    with a real date validator elsewhere in the same file.
    """
    lines = (text.replace("\\n", "\n") if escapes else text).strip().splitlines()
    if lines and lines[0].lstrip().startswith("@"):
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("@"):
                continue
            return _normalise_quote(line, escapes) if stripped.startswith(("def ", "async def ")) else ""
        return ""
    return _normalise_quote(text, escapes)


# BESSER's generated BAL runtime library: large and never app behaviour. Listed
# as omitted, so the judge knows it was not shown.
_DIGEST_SKIP_NAMES = frozenset({"bal_stdlib.py"})
_DIGEST_SKIP_DIRS = frozenset({
    "node_modules", "dist", "build", "__pycache__", "venv", "tests", "test",
    "__tests__", "fixtures", "coverage", "verification", "vendor",
})
_ARTIFACT_ROOT_DIRS = frozenset({"reports", "logs"})
# Third-party code copied into the app: it would spend the digest's frontend share.
_MINIFIED_NAME_RE = re.compile(r"\.min\.[cm]?js$", re.I)
# Every language the agent is expected to produce, because this one set feeds
# BOTH the digest the judge reads and the citation checker. A missing language
# gives the judge an empty digest on that stack, so every requirement comes
# back missing or unverified - both BLOCKERS - and the fix budget is spent
# chasing citations against files this cannot open.
#
# Derived from the stacks stack_metadata.py targets (_IDIOM_KEYWORDS and
# _GENERIC_IDIOM_KEYWORDS), so the two lists move together.
# Widening cannot inflate cost: _DIGEST_MAX_TOTAL_CHARS caps the digest and
# _digest_priority orders what fills it.
_SOURCE_SUFFIXES = frozenset({
    # Python / JS / TS / SQL
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".sql",
    # JVM (Spring Boot, Java or Kotlin)
    ".java", ".kt", ".kts",
    # the from-scratch backends
    ".rs", ".go", ".cs", ".fs", ".rb", ".php", ".ex", ".exs", ".dart", ".swift",
    # single-file component frontends
    ".vue", ".svelte", ".astro",
    # server-rendered templates that carry behaviour
    ".razor", ".cshtml", ".erb", ".blade.php",
    # native UI markup that holds a desktop app's controls: WPF/WinUI, Avalonia, Qt
    ".xaml", ".axaml", ".qml",
})


def _is_judged_source(path: Path) -> bool:
    """Application source, plus Android's screens (``res/layout*/*.xml``); other
    XML (``pom.xml``, the manifest) carries no behaviour."""
    suffix = path.suffix.lower()
    return suffix in _SOURCE_SUFFIXES or (suffix == ".xml" and path.parent.name.startswith("layout"))
_BEHAVIOURAL_KINDS = frozenset({"computed", "transition", "action"})
_TEST_SOURCE_NAME = re.compile(r"(?:\.(?:test|spec)\.[^.]+$|^test_.+\.py$|_test\.py$)", re.I)

_EXTRACT_SYSTEM_PROMPT = (
    "You turn a user's application request into the list of atomic, testable "
    "requirements a tester would check in the running application.\n"
    "Rules:\n"
    "- One behaviour per item, in the user's own words where possible.\n"
    "- List only behaviour the request itself states. Never add what an app "
    "of that kind usually has: no validations, refusals, endpoints, screens "
    "or display details the text does not name.\n"
    "- A vague request states little; return few items or an empty list. 'I "
    "want a todo app' states no testable requirement: return [].\n"
    "- Where stated, include: validations and the shape a value must have; uniqueness; "
    "business rules and limits; values that are computed rather than typed "
    "in; state transitions and the conditions under which an action must be "
    "refused; every named action and what it reports back; screens or "
    "interactions the user explicitly asked for.\n"
    "- NOT a requirement: that an entity exists or has a field ('a room has "
    "a description', 'a bill carries the date it was issued'). Keep such a "
    "statement only for the rule it carries: 'identified by its room number' "
    "is the uniqueness of that number, not the existence of the field.\n"
    "- kind: 'uniqueness' for identifiers and 'must be unique'; 'validation' "
    "for the shape a value must have; 'rule' for limits and cross-entity "
    "rules; 'computed' for values worked out rather than typed in; "
    "'transition' for state changes and when they are refused; 'action' for "
    "named operations; 'ui' for screens and interactions.\n"
    "- Skip the technology stack and instructions to the assistant.\n"
    "- Broad goals such as 'functional for any user' or 'no errors' do not add "
    "unstated authentication, authorization, user roles, integrations, or exhaustive "
    "accessibility/certification requirements. Extract the requested behavior, "
    "not a wish list of additional product features.\n"
    f"- At most {MAX_REQUIREMENTS} items. Never invent a requirement the "
    "user did not state."
)

_SUBMIT_REQUIREMENTS_TOOL = {
    "name": "submit_requirements",
    "description": "Submit the atomic requirements extracted from the request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "maxItems": MAX_REQUIREMENTS,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["validation", "uniqueness", "rule",
                                     "computed", "transition", "action", "ui"],
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        "required": ["requirements"],
    },
}
_EXTRACTED_KINDS = frozenset(
    _SUBMIT_REQUIREMENTS_TOOL["input_schema"]["properties"]["requirements"]["items"]["properties"]["kind"]["enum"])

# The judge call (unlike the much smaller extraction call) intermittently
# returns nothing usable within a second - an exception or empty completion,
# not a timeout, and not explained by digest size. One bounded retry is worth
# its cost against an intermittent failure.
_JUDGE_MAX_ATTEMPTS = 2
_JUDGE_RETRY_BACKOFF_SECONDS = 2.0
# Revisions on which the judge may return nothing before a run stops asking:
# its failure is a blocker no code edit fixes, so the fix loop's edits would
# otherwise buy a fresh full-price round every pass. Matches extraction's cap.
JUDGE_FAILED_ROUNDS_PER_RUN = 2

_JUDGE_SYSTEM_PROMPT = (
    "You audit generated application code against a numbered list of "
    "requirements. Decide from the CODE ONLY.\n"
    "When supplied, the original user specification is the scope authority; "
    "the numbered list is an index, not permission to invent additional requirements.\n"
    "- Judge only the stated requirement. Do not expand broad usability or "
    "error-free goals into unstated authentication, authorization, role-specific "
    "workflows, integrations, or exhaustive accessibility/certification work. "
    "Report a concrete defect in requested functionality, not an imagined feature.\n"
    "- Judge observable behavior, not one preferred implementation strategy. "
    "Enforcement at any effective layer is sufficient: for example, a database "
    "unique constraint with graceful rejection/rollback needs no redundant route "
    "duplicate query. For a partial/missing verdict, identify a publicly reachable "
    "operation and the outcome that violates THIS requirement. Arbitrary internal "
    "ORM writes, direct database tampering or a different requirement's defect "
    "are not counterexamples unless the original specification puts them in scope. "
    "Releasing/deactivating an allocation can be implemented by a state change "
    "that removes it from availability checks while retaining historical records. "
    "Do not infer physical deletion, loss of history, or removal of other required "
    "relationships from words such as 'release' alone.\n"
    "- For cross-entity invariants, inspect every relevant mutation path: create, "
    "update, delete, relationship changes and actions, including shared enforcing "
    "services or database constraints. One guarded route does not prove that other "
    "routes cannot bypass the rule. Cite a concrete uncovered path for a partial verdict.\n"
    "- Judge runtime application behavior from application code. A stale or unused "
    "scaffold test is not evidence that the running app is broken; configured test "
    "execution and actual test failures are a separate verification concern.\n"
    "- implemented: the code enforces or performs the behaviour. The evidence "
    "is a list of citations, each a file's path as listed and ONE source line "
    "copied exactly as it appears there, for example {\"path\": "
    "\"web_app/backend/routers/booking.py\", \"line\": \"raise "
    "HTTPException(status_code=400, detail='at least one room is required')\"}: "
    "the line that enforces or performs it: a unique=True, a validator, an "
    "if/raise in a router, the assignment that performs a transition. Behaviour "
    "spanning several lines or files gets one citation per line; explanations "
    "go in the note, never in a citation. A statement written over several lines "
    "is cited as ONE of its physical lines exactly as written (the line that does "
    "the work, or its first line), never rejoined. A "
    "column, relationship or field EXISTING is not evidence that a rule "
    "about it is enforced; a comment, docstring or TODO is not an "
    "implementation. A modeled action whose body raises HTTP 501 or "
    "NotImplementedError is not implemented. An enum or required input field "
    "does not implement a default, transition or computation. A validator that "
    "only returns its input does not enforce a rule. Overwritten declarations, "
    "nonexistent enum members and invalid database constraints do not count.\n"
    "- partial: some of it is there; quote what is there and say in the "
    "note what is missing. Missing excerpts are NOT a partial implementation: "
    "use unverified and inspection_paths when the relevant code was not shown.\n"
    "- missing: nothing in the code does it; say in the note what you looked "
    "for.\n"
    "- unverified: the supplied excerpts do not establish implementation or "
    "absence. Digest coverage markers explicitly identify truncated/omitted "
    "source; do not infer that behavior is absent from code you were not shown. "
    "Name the relevant file or mutation path that needs inspection and put its "
    "source path in inspection_paths. A code review cannot establish executed "
    "browser/build acceptance; do not call that absence of runtime evidence a "
    "code defect. Those checks are tracked separately by the harness.\n"
    "- For action/transition/computed requirements cite an executable body "
    "statement, not just a route decorator, enum class or ordinary column. "
    "A declarative default is suitable for an INITIAL-state requirement only. "
    "If citation-verification feedback is supplied, inspect the actual behavior "
    "and correct the rejected citation; never invent a code change just to "
    "satisfy the citation checker.\n"
    "Judge every requirement. Do not assume a framework enforces something "
    "the code does not show."
)

_SUBMIT_VERDICTS_TOOL = {
    "name": "submit_verdicts",
    "description": "Submit one verdict per requirement.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "status": {"type": "string", "enum": list(_STATUSES)},
                        "evidence": {
                            "type": "array", "maxItems": _MAX_CITATIONS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "maxLength": 500},
                                    "line": {"type": "string", "maxLength": _MAX_EVIDENCE_CHARS},
                                },
                                "required": ["path", "line"],
                            },
                            "description": (
                                "One citation per source line that enforces or performs the "
                                "requirement: the file's path as listed in the code and one "
                                "line copied exactly from it (of a multi-line statement, one "
                                "physical line, never the statement rejoined)."
                            ),
                        },
                        "note": {"type": "string", "description": "Explanation; never in a citation."},
                        "inspection_paths": {
                            "type": "array", "maxItems": 5,
                            "items": {"type": "string", "maxLength": 500},
                            "description": "Actual source paths needing focused inspection when excerpts are insufficient.",
                        },
                    },
                    "required": ["id", "status"],
                },
            },
        },
        "required": ["verdicts"],
    },
}


def original_request(instructions: str) -> str:
    """The user's own text: after the verbatim marker when the modeling agent
    appended it, otherwise the whole instructions."""
    text = instructions or ""
    idx = text.find(VERBATIM_MARKER)
    if idx < 0:
        return text.strip()
    tail = text[idx + len(VERBATIM_MARKER):].strip()
    head, sep, rest = tail.partition("\n\n")
    if sep and head.startswith("This is the authority"):
        return rest.strip()
    return tail


def assistant_notes(instructions: str) -> str:
    """The modeling agent's notes before the verbatim request; "" when there is no marker."""
    text = instructions or ""
    idx = text.find(VERBATIM_MARKER)
    return text[:idx].strip() if idx > 0 else ""


def latest_user_message(instructions: str) -> str:
    """The user's newest message: the text after the last later-request
    heading of :func:`original_request`, else the whole original request."""
    text = original_request(instructions).rsplit(LATER_REQUEST_MARKER, 1)[-1].strip()
    head, sep, rest = text.partition("\n\n")
    if sep and head.startswith("This later request takes precedence"):
        return rest.strip()
    return text


def _call_with_tool(llm_client, system: str, prompt: str, tool: dict, *,
                    use_planning_model: bool = True,
                    can_call: Callable[[], bool] | None = None) -> dict | None:
    """One forced tool call (planning model first, primary on error); the
    tool's input dict, or None when the call failed or returned nothing.

    ``use_planning_model=False`` goes straight to the primary model - a
    retry's second attempt should not repeat the same cheap-sibling call
    that just failed or came back empty; on a provider with no cheap
    sibling (``planning_model`` is already ``None``) this changes nothing.
    ``can_call`` is asked before the primary fallback, itself a paid call.
    """
    structured = _chat_supports_kwargs(llm_client, "force_tool", "model_override")
    planning_model = (
        getattr(llm_client, "planning_model", None) if structured and use_planning_model else None
    )
    messages = [{"role": "user", "content": prompt}]

    def _chat(model_override):
        if structured:
            return llm_client.chat(
                system=system, messages=messages, tools=[tool],
                force_tool=tool["name"], model_override=model_override,
            )
        return llm_client.chat(system=system, messages=messages, tools=[tool])

    try:
        response = _chat(planning_model)
    except Exception:
        if not planning_model:
            logger.warning("Requirements ledger: %s call failed", tool["name"], exc_info=True)
            return None
        if can_call is not None and not can_call():
            logger.warning("Requirements ledger: %s failed on the planning model; the primary "
                           "fallback is not sent (stopped or budget exhausted)", tool["name"], exc_info=True)
            return None
        try:
            response = _chat(None)
        except Exception:
            logger.warning("Requirements ledger: %s call failed", tool["name"], exc_info=True)
            return None
    payload = _tool_input(response, tool["name"])
    if payload is None:
        logger.warning("Requirements ledger: %s returned no usable tool input", tool["name"])
        return None
    # The client already coerced tool_use input; this also covers the prose fallback.
    return coerce_to_schema(payload, tool["input_schema"])


def _tool_input(response: dict, tool_name: str) -> dict | None:
    for block in response.get("content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            payload = getattr(block, "input", None)
            return payload if isinstance(payload, dict) else None
    # A client without forced tools answers in prose: take the first JSON object.
    text = "".join(
        getattr(b, "text", "") for b in response.get("content", []) or []
        if getattr(b, "type", None) == "text"
    )
    return extract_json(text, dict)


def extract_requirements(instructions: str, llm_client, *,
                         can_call: Callable[[], bool] | None = None) -> list[dict] | None:
    """``[{"id": 1, "text": ..., "kind": ...}, ...]`` from the user's request,
    or None when no real provider is available or the call failed.
    ``can_call`` as in :func:`_call_with_tool`."""
    validate_specification(instructions)
    if not _is_real_provider(llm_client):
        return None
    request = original_request(instructions)
    if not request:
        return None
    payload = _call_with_tool(
        llm_client, _EXTRACT_SYSTEM_PROMPT,
        f"## The user's request\n\n{request}", _SUBMIT_REQUIREMENTS_TOOL, can_call=can_call,
    )
    if not payload:
        return None
    extracted = payload.get("requirements")
    if not isinstance(extracted, list):
        logger.warning("Requirements ledger: 'requirements' came back as %s, not a list",
                       type(extracted).__name__)
        return None
    requirements: list[dict] = []
    for item in extracted:
        # A bare string is a requirement's text; a null or non-string text is
        # none ("None" was judged and reported as a blocker).
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip():
            continue
        kind = item.get("kind") if isinstance(item, dict) else ""
        # No client enforces the enum, and "verification" is the harness's own
        # undischargeable gate: an unknown kind is no kind.
        requirements.append({"id": len(requirements) + 1, "text": text.strip(),
                             "kind": kind if kind in _EXTRACTED_KINDS else ""})
    # Preserve every returned item, even if a provider ignores maxItems. A full
    # response may still have omitted the specification's tail; neither slicing
    # it nor treating a saturated extraction as complete is safe.
    if len(extracted) >= MAX_REQUIREMENTS:
        requirements.append(_incomplete_extraction_item(len(requirements) + 1))
    return requirements


def _incomplete_extraction_item(rid: int) -> dict:
    return {"id": rid, "text": _EXTRACTION_INCOMPLETE, "kind": "verification"}


def stated_requirements_ledger(stated: list[dict], instructions: str) -> list[dict] | None:
    """The ledger built from the requirements the editor sent, or None when they
    do not cover the request and it must be extracted instead.

    ``stated`` items are ``{"id", "text", "kind"}``: the modeling assistant
    extracted them from the user's request, and ``id`` is their artifact id in
    the project trace. Each ledger item keeps it as ``ref``, so a verdict names
    the requirement the trace links to model elements. Only the
    :data:`JUDGED_KINDS` are judged.

    The assistant extracts from the user's first request only. A request with
    later messages (``LATER_REQUEST_MARKER``) states requirements the list
    lacks, so the whole request is extracted instead.
    """
    if LATER_REQUEST_MARKER in original_request(instructions):
        logger.warning(
            "Requirements ledger: the request has later messages that the %d requirements "
            "from the editor do not cover; extracting from the whole request instead", len(stated),
        )
        return None
    judged = [item for item in stated if item["kind"] in JUDGED_KINDS]
    requirements = [{"id": rid, "text": item["text"], "kind": item["kind"], "ref": item["id"]}
                    for rid, item in enumerate(judged, start=1)]
    # The assistant's extraction stops at the same cap as ours: a full list may
    # have left the specification's tail out.
    if len(stated) >= MAX_REQUIREMENTS:
        requirements.append(_incomplete_extraction_item(len(requirements) + 1))
    return requirements


def render_requirements(requirements: list[dict] | None) -> str:
    """``R1. text`` lines for the planner prompt, the Phase 2 system prompt and
    the judge, so every party names a requirement the same way."""
    return "\n".join(f"R{r['id']}. {r['text']}" for r in requirements or [])


def _digest_priority(rel: str, shares: dict[str, float | None] | None = None) -> tuple[int, float, str]:
    """Digest order: schemas, backend code by the share of it the agent wrote, untouched
    generator output, pages and components, the rest. With no stored generator output
    (``shares`` empty) routers come before the other Python modules."""
    if rel.endswith(("pydantic_classes.py", "sql_alchemy.py")):
        return (0, 0.0, rel)
    router = "/routers/" in rel or rel.startswith("routers/")
    if router or rel.endswith(".py"):
        if not shares:
            return (1 if router else 2, 0.0, rel)
        # A file the generator did not write is wholly the agent's; an unknown share is edited.
        share = shares.get(rel, 1.0)
        share = _UNKNOWN_SHARE if share is None else share
        return (3 if share == 0 else 1, -share, rel)
    if "/pages/" in f"/{rel}" or "/components/" in f"/{rel}":
        return (4, 0.0, rel)
    return (5, 0.0, rel)


def _generator_baselines(output_dir: str, files: dict[str, str]) -> dict[str, str]:
    """The stored generator output of each judged file it holds; empty when there is no store."""
    for out_dir in sorted({rel.split("/", 1)[0] for rel in files if "/" in rel}):
        try:
            stored = load_generator_output(output_dir, out_dir)
        except GeneratorOutputError:
            continue
        if stored is None:
            return {}
        return {rel: content.decode("utf-8", "replace") for rel, content in stored.files.items() if rel in files}
    return {}


# Over this many lines a file's share is not computed: a line diff of a repetitive file
# grows fast (2,000 lines with a 3- and a 5-line period: 14 s). Generated files run to ~750.
_MAX_SHARE_DIFF_LINES = 1_500
# Where a file whose share is not computed ranks among the edited ones.
_UNKNOWN_SHARE = 0.5


@lru_cache(maxsize=256)
def _written_share(current: str, generated: str) -> float | None:
    """The share of ``current``'s lines that are not in ``generated`` (0 when untouched);
    None (edited, share unknown) for a file over ``_MAX_SHARE_DIFF_LINES``. Cached: the
    judge's cache key asks for every file's share on each validation."""
    if current == generated:
        return 0.0
    lines, old = current.splitlines(), generated.splitlines()
    if max(len(lines), len(old)) > _MAX_SHARE_DIFF_LINES:
        return None
    kept = sum(size for _, _, size in matching_lines(old, lines))
    return round((len(lines) - kept) / len(lines), 4) if lines else 0.0


def _written_shares(output_dir: str, files: dict[str, str],
                    baselines: dict[str, str] | None = None) -> dict[str, float | None]:
    """For each judged file the stored generator output holds, the share of its lines
    the agent wrote (0 for untouched); empty when there is no store."""
    shares: dict[str, float | None] = {}
    for rel, generated in (_generator_baselines(output_dir, files) if baselines is None else baselines).items():
        try:
            current = Path(files[rel]).read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        shares[rel] = _written_share(current, generated)
    return shares


# An edited generator file under this share is shown as its changed regions (r12-p81:
# seven routers 1-16% edited took 130K of the digest and pushed the rules module out).
_REGION_VIEW_MAX_SHARE = 0.5
# Unchanged lines kept around each changed one, so a changed line keeps its route or def.
_REGION_CONTEXT_LINES = 3


def _changed_regions(compacted: str, generated: str) -> str:
    """``compacted`` (a :func:`_compact` file) without its runs of lines unchanged from
    ``generated`` beyond ``_REGION_CONTEXT_LINES`` of a change; each run left out is
    marked, and so is each place where generator lines were removed."""
    lines = compacted.splitlines()
    changed = [True] * len(lines)
    removed: dict[int, int] = {}  # index in ``lines`` -> generator lines removed just before it
    old_end = 0
    for old_start, start, size in matching_lines(_compact(generated).splitlines(), lines):
        if old_start > old_end:
            removed[start] = removed.get(start, 0) + old_start - old_end
        changed[start:start + size] = [False] * size
        old_end = old_start + size
    shown = [False] * len(lines)
    windows = [(i, i + 1) for i, flag in enumerate(changed) if flag] + [(i, i) for i in removed]
    for first, last in windows:  # a change's lines, or the gap a removal left
        low, high = max(0, first - _REGION_CONTEXT_LINES), last + _REGION_CONTEXT_LINES
        shown[low:high] = [True] * len(shown[low:high])
    parts: list[str] = []
    skipped = 0
    for index in range(len(lines) + 1):
        keep = index < len(lines) and shown[index]
        if skipped and (keep or index in removed or index == len(lines)):
            parts.append(f"[{skipped} unchanged generator lines left out]")
            skipped = 0
        if index in removed:
            parts.append(f"[{removed[index]} generator lines removed here]")
        if index == len(lines):
            break
        if keep:
            parts.append(lines[index])
        else:
            skipped += 1
    return "\n".join(parts)


def _source_files(output_dir: str) -> dict[str, str]:
    """Only application source, never run logs/recipes, vendored or minified
    code or an escaped symlink. ``public/`` stays: an Express app serves its own
    browser code from there."""
    base = Path(output_dir).resolve()
    files: dict[str, str] = {}
    for root, dirs, names in walk_plain(base):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _DIGEST_SKIP_DIRS
                   and not (Path(root) == base and d in _ARTIFACT_ROOT_DIRS)]
        for name in names:
            source = Path(root, name)
            if (name.startswith(".") or not _is_judged_source(source) or _TEST_SOURCE_NAME.search(name)
                    or _MINIFIED_NAME_RE.search(name)):
                continue
            try:
                if not source.resolve().is_relative_to(base) or not source.is_file():
                    continue
            except OSError:
                continue
            files[source.relative_to(base).as_posix()] = str(source)
    return files


def judged_source_fingerprint(output_dir: str) -> str:
    """Hash of every file the judge and the citation check read: path and raw bytes.

    Raw bytes, not the digest: the digest is clipped (an edit past its budget
    still reaches the citation repair) and its comment stripping is a size
    heuristic, so a comment-only edit re-judges.
    """
    fingerprint = hashlib.sha256()
    files = _source_files(output_dir)
    # How much of each generated file the agent wrote orders the digest.
    fingerprint.update(json.dumps(sorted(_written_shares(output_dir, files).items())).encode("utf-8"))
    for rel, full in sorted(files.items()):
        try:
            content = Path(full).read_bytes()
        except OSError:
            content = b"<unreadable>"
        fingerprint.update(f"{rel}\0{len(content)}\0".encode("utf-8"))
        fingerprint.update(content)
    return fingerprint.hexdigest()


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def _class_homes(files: dict[str, str], cache: dict) -> dict[str, set[str]]:
    """Class name -> the source files defining it, built once per run."""
    if "classes" not in cache:
        homes: dict[str, set[str]] = {}
        for full in files.values():
            if not full.endswith(".py"):
                continue
            try:
                with open(full, "r", encoding="utf-8-sig", errors="ignore") as handle:
                    tree = ast.parse(handle.read())
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    homes.setdefault(node.name, set()).add(full)
        cache["classes"] = homes
    return cache["classes"]


def _compact(content: str) -> str:
    """Drop blank and comment-only lines; the judge reads code, not prose."""
    kept = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept)


def build_app_digest(output_dir: str, *, focus_paths: list[str] | None = None) -> str:
    """Bounded source with explicit coverage gaps and optional focused excerpts.

    Citation repair reads the cited files beyond the ordinary per-file cap,
    plus the schema files, and lists every other file as omitted: re-judging
    a few requirements against the whole app cost as much as the full
    judgment. With no resolvable focus path it falls back to the whole app.
    """
    return build_app_digest_and_gaps(output_dir, focus_paths=focus_paths)[0]


# Files a focused digest shows whole, beside the schema files.
_MAX_FOCUS_FILES = 10


def focus_paths_in_turn(path_lists: list[list[str]]) -> list[str]:
    """The paths of several requirements taken one from each in turn, without repeats, so
    the paths of a later requirement are not crowded out by an earlier one's (r13-p81)."""
    lists = [list(dict.fromkeys(path for path in paths if isinstance(path, str) and path.strip()))
             for paths in path_lists]
    ordered: list[str] = []
    for depth in range(max((len(paths) for paths in lists), default=0)):
        ordered += [paths[depth] for paths in lists if depth < len(paths) and paths[depth] not in ordered]
    return ordered


def resolved_focus_paths(files: dict[str, str], focus_paths: list | None) -> list[str]:
    """The first ten distinct judged files the focus paths name, each unambiguously (a path
    or a suffix). Capped after resolving: ten copies of one path left the rest out (r13-p81)."""
    resolved: list[str] = []
    for path in focus_paths or []:
        if len(resolved) >= _MAX_FOCUS_FILES:
            break
        if not isinstance(path, str):
            continue
        path = path.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        candidates = [path] if path in files else [rel for rel in files if rel.endswith("/" + path)]
        if path and len(candidates) == 1 and candidates[0] not in resolved:
            resolved.append(candidates[0])
    return resolved


def build_app_digest_and_gaps(output_dir: str, *,
                              focus_paths: list[str] | None = None) -> tuple[str, frozenset[str]]:
    """:func:`build_app_digest`, and the source files it left out or cut for the budget.

    The gaps are what the judge could not read in full: a verdict about them is
    not a fair one. Files never shown (``_DIGEST_SKIP_NAMES``) and the files a
    focused digest leaves out on purpose are not gaps.
    """
    digest, gaps, _shown = build_app_digest_details(output_dir, focus_paths=focus_paths)
    return digest, gaps


def build_app_digest_details(output_dir: str, *, focus_paths: list[str] | None = None
                             ) -> tuple[str, frozenset[str], frozenset[str]]:
    """:func:`build_app_digest_and_gaps`, and the files the digest shows (a focused one
    always shows the schema files too)."""
    files = _source_files(output_dir)
    baselines = _generator_baselines(output_dir, files)
    shares = _written_shares(output_dir, files, baselines)
    focused = set(resolved_focus_paths(files, focus_paths))
    if focus_paths and not focused:
        logger.warning("Requirements ledger: no cited path resolved for citation repair; "
                       "the repair reads the whole app")
    footer_budget = min(6000, max(256, _DIGEST_MAX_TOTAL_CHARS // 10), _DIGEST_MAX_TOTAL_CHARS)
    source_budget = _DIGEST_MAX_TOTAL_CHARS - footer_budget
    entries = []
    regions: set[str] = set()
    contents: list[tuple[str, str]] = []
    unreadable = []
    omitted_by_focus: list[str] = []
    for rel, full in files.items():
        if focused and rel not in focused and _digest_priority(rel)[0] != 0:
            omitted_by_focus.append(rel)
            continue
        if Path(full).name in _DIGEST_SKIP_NAMES:
            unreadable.append(rel)
            continue
        try:
            with open(full, "r", encoding="utf-8-sig", errors="ignore") as fh:
                content = _compact(fh.read())
        except OSError:
            unreadable.append(rel)
            continue
        if not content:  # an empty __init__.py: nothing to show
            continue
        contents.append((rel, content))
    # Only when the whole app does not fit: shown whole, generated code a requirement
    # relies on (a CRUD route) stays citable.
    whole = sum(len(rel) + min(len(content), _DIGEST_MAX_FILE_CHARS) + 7 for rel, content in contents)
    for rel, content in contents:
        if (whole > source_budget and rel not in focused and rel in baselines
                and _digest_priority(rel)[0] != 0 and shares[rel] is not None
                and 0 < shares[rel] < _REGION_VIEW_MAX_SHARE):
            content = _changed_regions(content, baselines[rel])
            regions.add(rel)
        header = f"### {rel}\n"
        # The cited files share the budget: one large file must not crowd out the next.
        cap = (max(0, source_budget // len(focused) - len(header) - 200) if rel in focused
               else _DIGEST_MAX_FILE_CHARS)
        truncated = len(content) > cap
        chunk = header + content[:cap]
        if truncated:
            chunk += (f"\n[DIGEST TRUNCATED: {rel}; showing {cap} of {len(content)} code characters. "
                      "Do not infer absence from omitted code.]\n")
        entries.append(((0 if rel in focused else 1, *_digest_priority(rel, shares)), rel, chunk, truncated))
    entries.sort(key=lambda e: e[0])
    reserved: set[str] = set()
    total = 0
    if not focused:  # pages and components first, then the rest of the non-Python source
        ui_budget = int(source_budget * _DIGEST_UI_SHARE)
        for key, rel, chunk, _ in entries:
            if key[1] >= 4 and total + len(chunk) + 2 <= ui_budget:
                reserved.add(rel)
                total += len(chunk) + 2
    parts: list[str] = []
    omitted = unreadable + omitted_by_focus
    truncated_paths = []
    over_budget = []
    shown: list[str] = []
    for _, rel, chunk, truncated in entries:
        if rel not in reserved:
            if total + len(chunk) + 2 > source_budget:
                omitted.append(rel)
                over_budget.append(rel)
                continue
            total += len(chunk) + 2
        parts.append(chunk)
        shown.append(rel)
        if truncated or rel in regions:
            truncated_paths.append(rel)
    footer = (
        "\n\n## Digest coverage (not application source)\n"
        f"Files shown: {len(parts)}; truncated excerpts: {len(truncated_paths)}; "
        f"omitted/unreadable files: {len(omitted)}.\n"
        "Omitted or truncated code is unknown, not evidence of absent behavior.\n"
        + ("This is a focused excerpt: only the cited files and the schema files are shown; "
           "the rest of the application is omitted, not absent.\n" if omitted_by_focus else "")
        + ("Truncated files: " + ", ".join(truncated_paths) + "\n" if truncated_paths else "")
        + ("Of these, shown as their edits only (unchanged generator lines left out, and lines the agent "
           "removed, are marked where they were): "
           + ", ".join(sorted(regions & set(truncated_paths))) + "\n" if regions & set(truncated_paths) else "")
        + ("Omitted files: " + ", ".join(omitted) + "\n" if omitted else "")
    )
    if len(footer) > footer_budget:
        footer = footer[:max(0, footer_budget - 42)] + "\n[Coverage path listing also truncated.]\n"
    return "\n\n".join(parts) + footer[:footer_budget], frozenset(over_budget + truncated_paths), frozenset(shown)


def named_digest_gaps(verdict: dict, gaps: frozenset[str]) -> list[str]:
    """The digest gaps a verdict's inspection paths, citations or note name.

    A gap counts when its path appears in the note, when a named path is it or a
    suffix of it (``risk_service.py`` for ``backend/risk_service.py``), or when its
    file name appears in the note and no other gap has that name.
    """
    named = [path.strip().replace("\\", "/").removeprefix("./")
             for path in [*verdict.get("inspection_paths", []), *cited_paths(verdict.get("evidence"))]
             if isinstance(path, str) and path.strip()]
    note = str(verdict.get("note") or "")
    names = [Path(rel).name for rel in gaps]
    return [rel for rel in sorted(gaps)
            if rel in note or any(rel == path or rel.endswith("/" + path) for path in named)
            or (names.count(Path(rel).name) == 1
                and re.search(r"(?<![\w.-])" + re.escape(Path(rel).name) + r"(?![\w-])", note))]


# The listing names requirements "R1. text"; a judge may echo "R1" or "1.".
_VERDICT_ID_RE = re.compile(r"\s*[Rr#]?(\d+)\.?\s*")


def _verdict_id(value) -> int | None:
    """A verdict's requirement id, or None. ``int()`` alone rejected "R1" and
    truncated 1.5 and True to requirement 1."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    match = _VERDICT_ID_RE.fullmatch(value) if isinstance(value, str) else None
    return int(match.group(1)) if match else None


def _parse_evidence(value) -> str | list[dict]:
    """``[{"path", "line"}, ...]`` from the tool's citations; a string (a text
    reply, or a citation sent as "path: line") keeps the older string form."""
    if not isinstance(value, list):
        return str(value or "").strip()
    citations = []
    for citation in value:
        if isinstance(citation, str):
            path, _, line = citation.partition(":")
            citation = {"path": path, "line": line}
        if isinstance(citation, dict):
            citations.append({"path": str(citation.get("path") or "").strip(),
                              "line": str(citation.get("line") or "").strip()})
    return citations


def render_evidence(evidence) -> str:
    """A verdict's evidence as ``path: line``, citations joined by " and "; a
    string (the older stored form) as it is."""
    if isinstance(evidence, list):
        return " and ".join(f"{c.get('path', '')}: {c.get('line', '')}" for c in evidence if isinstance(c, dict))
    return str(evidence or "")


def cited_paths(evidence) -> list[str]:
    """The source paths a verdict's evidence names."""
    if isinstance(evidence, list):
        return [str(c.get("path") or "").strip() for c in evidence if isinstance(c, dict)]
    return [str(evidence or "").split(":", 1)[0].strip()]


def _verdict_identity(rid: int, requirement: dict) -> dict:
    """What a verdict repeats of its requirement: id, text, kind and, for a
    requirement the editor sent, its trace artifact id (``ref``)."""
    identity = {"id": rid, "text": requirement["text"], "kind": requirement.get("kind", "")}
    if requirement.get("ref"):
        identity["ref"] = requirement["ref"]
    return identity


def _judge_once(llm_client, judge_prompt: str, by_id: dict[int, dict], *,
                use_planning_model: bool, can_call: Callable[[], bool] | None = None) -> dict[int, dict]:
    """One judge call parsed into ``{id: verdict}``; empty when the call
    raised, came back with no tool call, or parsed to nothing usable."""
    payload = _call_with_tool(
        llm_client, _JUDGE_SYSTEM_PROMPT, judge_prompt, _SUBMIT_VERDICTS_TOOL,
        use_planning_model=use_planning_model, can_call=can_call,
    )
    verdicts: dict[int, dict] = {}
    for item in (payload or {}).get("verdicts") or []:
        if not isinstance(item, dict):
            continue
        rid = _verdict_id(item.get("id"))
        if rid not in by_id:
            continue
        status = str(item.get("status", "")).lower()
        verdicts[rid] = {
            **_verdict_identity(rid, by_id[rid]),
            "status": status if status in _STATUSES else "unverified",
            "evidence": _parse_evidence(item.get("evidence")),
            "note": str(item.get("note") or "").strip(),
            "inspection_paths": [path[:500] for path in item.get("inspection_paths", [])
                                 if isinstance(path, str)][:5]
            if isinstance(item.get("inspection_paths"), list) else [],
        }
    return verdicts


def judge_coverage(requirements: list[dict], digest: str, llm_client, *,
                   original_spec: str | None = None,
                   previous_verdicts: list[dict] | None = None,
                   can_call: Callable[[], bool] | None = None) -> list[dict] | None:
    """One verdict per requirement, or None when the call failed.

    ``can_call`` (the run's stop and spend check) is asked before every paid
    call: an attempt, the retry and the primary fallback each carry the digest.
    """
    if not requirements or not _is_real_provider(llm_client):
        return None
    application_requirements = [r for r in requirements if r.get("kind") != "verification"]
    listing = "\n".join(
        f"R{r['id']}. {r['text']} [verification kind: {r.get('kind', '')}]"
        for r in application_requirements
    )
    authority = ("## Original user specification (scope authority)\n\n"
                 + original_request(original_spec) + "\n\n") if original_spec else ""
    requested_ids = {r["id"] for r in application_requirements}
    feedback = [{"id": item.get("id"),
                 "rejected_evidence": render_evidence(item.get("evidence"))[:1500],
                 "verification_feedback": str(item.get("note") or "")[:1500]}
                for item in previous_verdicts or []
                if isinstance(item, dict) and item.get("id") in requested_ids][:MAX_REQUIREMENTS]
    repair_context = ("## Previous citation verification (diagnostic data, not application code)\n\n"
                      + json.dumps(feedback, ensure_ascii=True)[:16000] + "\n\n") if feedback else ""
    by_id = {r["id"]: r for r in requirements}
    verdicts: dict[int, dict] = {}
    if application_requirements:
        judge_prompt = (authority + repair_context + f"## Requirements\n\n{listing}"
                        f"\n\n## Generated code\n\n{digest}")
        # The first attempt uses the planning model first; a
        # retry forces the primary model straight away (see _call_with_tool)
        # instead of repeating the exact call that just failed or came back
        # empty. Bounded to _JUDGE_MAX_ATTEMPTS: every attempt is a full,
        # paid call carrying the whole digest.
        for attempt in range(_JUDGE_MAX_ATTEMPTS):
            if can_call is not None and not can_call():
                logger.warning("Requirements ledger: judge attempt %d/%d not sent (stopped or "
                               "budget exhausted)", attempt + 1, _JUDGE_MAX_ATTEMPTS)
                break
            verdicts = _judge_once(llm_client, judge_prompt, by_id,
                                   use_planning_model=attempt == 0, can_call=can_call)
            if verdicts or attempt + 1 >= _JUDGE_MAX_ATTEMPTS:
                break
            logger.warning(
                "Requirements ledger: the judge returned no usable verdict on "
                "attempt %d/%d (model %s); retrying",
                attempt + 1, _JUDGE_MAX_ATTEMPTS, getattr(llm_client, "model", "?"),
            )
            time.sleep(_JUDGE_RETRY_BACKOFF_SECONDS)
    # Nothing usable came back at all, even after the retry above. That is a
    # failed check, not one unverified verdict per requirement, which would
    # bury the real blockers under one per requirement. The caller has a single
    # finding for this, and it must still fire once retries are exhausted.
    if not verdicts and application_requirements:
        logger.warning(
            "Requirements ledger: the judge returned no usable verdict for any "
            "of %d requirements after %d attempt(s) (model %s)",
            len(by_id), _JUDGE_MAX_ATTEMPTS, getattr(llm_client, "model", "?"),
        )
        return None
    # A requirement the judge skipped is not implemented until shown otherwise.
    for rid, req in by_id.items():
        if req.get("kind") == "verification":
            # No code citation (or model claim) can discharge incomplete
            # extraction. Only completing the extraction can remove this gate.
            verdicts[rid] = {
                **_verdict_identity(rid, req),
                "status": "unverified", "evidence": "",
                "note": "requirement extraction completeness has not been established",
            }
            continue
        verdicts.setdefault(rid, {
            **_verdict_identity(rid, req),
            "status": "unverified", "evidence": "", "note": "no verdict returned",
        })
    return [verdicts[rid] for rid in sorted(verdicts)]


def _collapse(text: str) -> str:
    """Whitespace-free: the code tokens must match verbatim, their spacing
    need not (``firstName: str`` and ``firstName:str`` are the same line)."""
    return "".join(text.split())


def _unimplemented_raise(node: ast.AST) -> bool:
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exc = node.exc
    target = exc.func if isinstance(exc, ast.Call) else exc
    name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
    if name == "NotImplementedError":
        return True
    return (
        name == "HTTPException" and isinstance(exc, ast.Call)
        and any(k.arg == "status_code" and isinstance(k.value, ast.Constant)
                and k.value.value == 501 for k in exc.keywords)
    )


def _stub_body(body: list[ast.stmt]) -> bool:
    """Recognise proven placeholders, not every action with an error branch.

    The generated methods load a row and then unconditionally raise 501
    inside try/except. A conditional unsupported-case response in an otherwise
    implemented action must not be mistaken for that scaffold.
    """
    executable = [s for s in body if not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
        and isinstance(s.value.value, str)
    )]
    if not executable or all(
        isinstance(s, ast.Pass) or (
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
            and s.value.value is Ellipsis
        ) for s in executable
    ):
        return True
    # A return in an earlier conditional can be the successful path. Prefer
    # inconclusive evidence to falsely rejecting such a real implementation.
    if any(isinstance(n, ast.Return) for statement in executable for n in ast.walk(statement)):
        return False
    for stmt in executable:
        if _unimplemented_raise(stmt):
            return True
        if (isinstance(stmt, ast.Try) and _stub_body(stmt.body)
                and any(_unimplemented_raise(n) for n in ast.walk(stmt))):
            # An exception handler may implement a fallback. Only treat the
            # wrapper as a stub when every handler finishes by re-raising or
            # failing, with no successful return or finally override.
            handlers_fail = all(
                handler.body and isinstance(handler.body[-1], ast.Raise)
                and not any(isinstance(n, ast.Return) for n in ast.walk(handler))
                for handler in stmt.handlers
            )
            if handlers_fail and not any(
                isinstance(n, ast.Return) for final in stmt.finalbody for n in ast.walk(final)
            ):
                return True
    return False


def _statement_start(node: ast.stmt) -> int:
    """Decorators belong to their definition, although AST lineno excludes them."""
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _noop_validator(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Recognise a proven pass-through, not unknown helpers or real predicates."""
    validator_decorators = {"validator", "root_validator", "field_validator", "model_validator"}
    decorated = any(
        getattr(target, "id", getattr(target, "attr", "")) in validator_decorators
        for decorator in function.decorator_list
        for target in [decorator.func if isinstance(decorator, ast.Call) else decorator]
    )
    if not decorated:
        return False
    arguments = {argument.arg for argument in (
        function.args.posonlyargs + function.args.args + function.args.kwonlyargs
    )}
    for statement in function.body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        ):
            continue
        if isinstance(statement, ast.Return) and (
            statement.value is None or isinstance(statement.value, ast.Constant)
            or isinstance(statement.value, ast.Name) and statement.value.id in arguments
        ):
            continue
        # Assignments, calls, conditions and other operations may implement the
        # rule. Leave their meaning to the judge instead of guessing from names.
        return False
    return True


# A JSX comment is written as an expression: {/* note */}. A brace on its own
# line ("catch (e) {\n /* ignore */\n}") opens a JS block, not a JSX comment.
_JSX_COMMENT_RE = re.compile(r"\{[ \t]*(/\*(?:(?!\*/)[\s\S])*\*/)[ \t]*\}")


def _blank_jsx_comment_braces(text: str) -> str:
    """``{/* note */}`` as `` /* note */ ``: every offset and line break kept."""
    return _JSX_COMMENT_RE.sub(lambda match: " " + match.group()[1:-1] + " ", text)


def _comment_only_lines(source: str, suffix: str) -> set[int]:
    """Recognise comments in their source language, not by a shared prefix.

    Python **kwargs / *args, JS generators and --decrements are executable.
    Mask C-style/SQL comments while skipping quoted strings, so interior lines
    of block comments cannot masquerade as code citations either.
    """
    if suffix in (".js", ".jsx", ".tsx"):
        source = _blank_jsx_comment_braces(source)
    lines = source.splitlines()
    if suffix == ".py":
        return {i for i, line in enumerate(lines, 1) if line.lstrip().startswith("#")}
    line_comment = "--" if suffix == ".sql" else "//"
    tokens = re.compile(
        r"/\*[\s\S]*?(?:\*/|\Z)|" + re.escape(line_comment) + r"[^\n]*|"
        r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`'''
    )
    masked = list(source)
    for match in tokens.finditer(source):
        if match.group().startswith(("/*", line_comment)):
            for index in range(match.start(), match.end()):
                if masked[index] not in "\r\n":
                    masked[index] = " "
    return {i for i, (original, visible) in enumerate(zip(lines, "".join(masked).splitlines()), 1)
            if original.strip() and not visible.strip()}


def _python_citation(tree: ast.AST | None, lineno: int, kind: str, text: str) -> tuple[ast.stmt | None, str]:
    """Check the cited statement's role; this is not a semantic proof."""
    if tree is None:
        return None, "the cited Python source does not parse"
    enclosing = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)
                 and _statement_start(n) <= lineno <= n.end_lineno]
    if not enclosing:
        return None, "the cited line is not an executable statement"
    statement = min(enclosing, key=lambda n: n.end_lineno - _statement_start(n))
    if isinstance(statement, (ast.Import, ast.ImportFrom, ast.Pass)) or (
        isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
    ):
        return statement, "the cited line is a comment, docstring, import or placeholder"
    functions = [n for n in enclosing if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if kind in {"validation", "rule", "uniqueness", "implementation"} and any(_noop_validator(fn) for fn in functions):
        return statement, "the cited validator is a no-op: it does not check or enforce a rule"
    if kind == "implementation":
        # Checklist evidence records existing source, not behavioral acceptance.
        # Real framework declarations are valid implementation evidence; known
        # stubs are not. The requirements ledger still verifies the actual rule.
        body = min(functions, key=lambda n: n.end_lineno - n.lineno).body if functions else (
            statement.body if isinstance(statement, ast.ClassDef) else None)
        if _unimplemented_raise(statement) or body is not None and _stub_body(body):
            return statement, "the cited implementation is an unimplemented scaffold"
    if kind in _BEHAVIOURAL_KINDS:
        if not functions:
            # Real declarative defaults / database expressions can implement
            # these behaviours; a bare enum or required input cannot.
            value = getattr(statement, "value", None)
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Call):
                if (kind == "transition" and any(word in text.lower() for word in (
                        "start", "begin", "initial", "default"))
                        and any(k.arg in {"default", "default_factory", "server_default"}
                                for k in value.keywords)):
                    return statement, ""
                if kind == "computed" and any(
                    isinstance(n, ast.Call) and (
                        getattr(n.func, "id", "") == "Computed"
                        or getattr(n.func, "attr", "") == "Computed"
                    ) for n in ast.walk(value)
                ):
                    return statement, ""
            return statement, "the cited declaration does not perform the required behaviour"
        function = min(functions, key=lambda n: n.end_lineno - n.lineno)
        if _stub_body(function.body) or _unimplemented_raise(statement):
            return statement, "the cited action body is an unimplemented scaffold"
    return statement, ""


def _without_comments(text: str, filename: str) -> str:
    """``text`` without its comments (Pygments), strings kept, every line break kept;
    a JSX ``{/* note */}`` goes with its braces."""
    suffix = Path(filename).suffix.lower()
    lexer = _lexer_for(suffix)
    if lexer is None:
        return text
    if suffix in (".js", ".jsx", ".tsx"):
        text = _blank_jsx_comment_braces(text)
    return "".join("\n" * value.count("\n") if token in Comment else value
                   for token, value in lexer.get_tokens(text))


def _load_source(candidate: str, parsed: dict, sqlite: bool) -> _Source:
    """One cited file, read and prepared once per verification."""
    if candidate not in parsed:
        try:
            with open(candidate, "r", encoding="utf-8-sig", errors="ignore") as fh:
                source = fh.read()
        except OSError:
            parsed[candidate] = _Source(candidate, [], [], [], None, [], set(), [], [])
            return parsed[candidate]
        tree = None
        if candidate.endswith(".py"):
            try:
                tree = ast.parse(source)
            except (SyntaxError, RecursionError):  # too deeply nested to parse: read as text only
                pass
        lines = source.splitlines()
        normalised = [_normalise_line(line) for line in lines]
        code = [_normalise_line(line) for line in code_only(source, candidate).splitlines()]
        bare = [_normalise_line(line) for line in _without_comments(source, candidate).splitlines()]
        parsed[candidate] = _Source(
            candidate, lines, normalised, code if len(code) == len(lines) else normalised, tree,
            python_structural_diagnostics(tree, sqlite=sqlite) if tree else [],
            _comment_only_lines(source, Path(candidate).suffix.lower()),
            bare if len(bare) == len(lines) else normalised,
            _masked_lines(source, candidate, normalised),
        )
    return parsed[candidate]


def _resolve_cited_path(path: str, quoted: str, files: dict[str, str], quote_cache: dict) -> list[str]:
    """The source files a citation's path can name (one, when unambiguous)."""
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    candidates = [] if not path or path.startswith("/") or ".." in path.split("/") else (
        [files[path]] if path in files else
        [full for rel, full in files.items() if rel.endswith("/" + path)]
    )
    # The judge is asked for ``<path>:<line>`` and sometimes answers with
    # the CLASS the line belongs to - ``Person: id: Mapped_[int] =
    # mapped_column(Integer_, primary_key=True)``. Accept the class as a
    # locator only when the tree defines exactly one class by that name,
    # so a placeholder such as the literal word "path" stays unverified.
    if not candidates and quoted and _IDENTIFIER_RE.match(path):
        homes = _class_homes(files, quote_cache).get(path, ())
        if len(homes) == 1:
            candidates = list(homes)
    return candidates


def _span_failure(item: dict, source: _Source, first_line: int, lineno: int, quote: str) -> str:
    """Why a located span does not implement the requirement; "" when it does.

    A span's checks apply to its last line: the body statement of a
    def-and-body quote, the same statement for a formatted call.
    """
    if first_line in source.comments or lineno in source.comments:
        return "the cited line is a comment, not an implementation"
    statement = None
    if source.path.endswith(".py"):
        statement, reason = _python_citation(source.tree, lineno, item.get("kind", ""), item.get("text", ""))
        if reason:
            return reason
        invalid = next((d for d in source.diagnostics if (
            d["line"] <= lineno <= d.get("end_line", d["line"])
            or statement is not None and _statement_start(statement) <= d["line"]
            <= d.get("end_line", d["line"]) <= statement.end_lineno
        )), None)
        if invalid:
            return "the cited implementation is invalid: " + invalid["message"]
    kind, text = item.get("kind", ""), item.get("text", "")
    if (_demands_enforcement(kind, text)
            and not isinstance(statement, (ast.If, ast.Raise, ast.Assert))
            and not source.lines[first_line - 1].lstrip().startswith(("if ", "if(", "if\t", "else if "))
            and not any(tok in quote.lower() for tok in _ENFORCEMENT_TOKENS)
            and not (kind != "uniqueness" and _only_lists_cited_enumeration(text, statement, source.tree))):
        return "the cited line declares data; it does not demonstrate enforcement"
    return ""


_MEMBER_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# A list of values: "a, b or c", "a/b", "a and b".
_LIST_SEPARATOR = r"(?:,\s*(?:(?:or|and)\b)?|/|\b(?:or|and)\b)"
_NEXT_LISTED_RE = re.compile(
    rf"\s*{_LIST_SEPARATOR}\s*(?P<item>[\w'-]+(?:\s+[\w'-]+){{0,2}}?)(?=\s*(?:[,/.;:)]|\b(?:or|and)\b|$))", re.I)
_PREVIOUS_LISTED_RE = re.compile(rf"(?P<item>[\w'-]+)\s*{_LIST_SEPARATOR}\s*$", re.I)
# Words that start a clause, not a value: "positive or negative, never anything else".
_NOT_A_VALUE = frozenset({"a", "an", "the", "its", "their", "each", "every", "is", "are", "be", "must", "may",
                          "can", "should", "will", "not", "never", "no", "any", "only"})


def _names_pattern(name: str) -> str:
    # IN_PROGRESS and InProgress both read "in progress", "in-progress" or "inprogress".
    return r"[\s_-]*".join(map(re.escape, _MEMBER_WORD_RE.findall(name)))


def _member_pattern(member: ast.Assign) -> re.Pattern | None:
    """A member as the requirement may write it: its name, or its string value (``P = "positive"``)."""
    names = [target.id for target in member.targets if isinstance(target, ast.Name)]
    value = member.value.value if isinstance(member.value, ast.Constant) else None
    names += [value] if isinstance(value, str) else []
    spelled = [_names_pattern(name) for name in names if len(name.strip()) > 1 and _MEMBER_WORD_RE.search(name)]
    return re.compile(rf"\b(?:{'|'.join(spelled)})\b", re.I) if spelled else None


def _lists_only_members(text: str, spans: list[tuple[int, int]]) -> bool:
    """Every value listed beside a member ("a, b or c") is a member too."""
    starts, ends = {start for start, _ in spans}, {end for _, end in spans}
    for start, end in spans:
        following = _NEXT_LISTED_RE.match(text, end)
        if (following and following.start("item") not in starts
                and following.group("item").split()[0].lower() not in _NOT_A_VALUE):
            return False
        preceding = _PREVIOUS_LISTED_RE.search(text[:start])
        if (preceding and preceding.end("item") not in ends
                and preceding.group("item").lower() not in _NOT_A_VALUE):
            return False
    return True


def _field_named(text: str, field: str) -> bool:
    """Every word of a field's name is a word of the requirement, singular or plural."""
    words = _MEMBER_WORD_RE.findall(field)
    return bool(words) and all(re.search(rf"\b{re.escape(word)}(?:e?s)?\b", text, re.I) for word in words)


def _only_lists_cited_enumeration(text: str, statement: ast.stmt | None, tree: ast.AST | None) -> bool:
    """The requirement's constraint words are values of the enumeration cited.

    The cited Python statement defines, belongs to or types a field with one
    enumeration of the same file. A field of a class of that file is of that
    enumeration and is the one the requirement names (the cited field, when one
    is cited). The requirement names every value of the enumeration, lists no
    other value beside them, and states no constraint once they are taken out.
    The type then refuses any other value, which is the enforcement. "Must not
    go from closed back to open" names the values too, and still needs its check."""
    if statement is None or tree is None:
        return False
    enumerations = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        and any(getattr(base, "id", getattr(base, "attr", "")).endswith("Enum") for base in node.bases)}
    cited = [node for node in enumerations.values() if node.lineno <= statement.lineno <= node.end_lineno] or [
        enumerations[name] for name in {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}
        if name in enumerations]
    if len(cited) != 1:
        return False
    enumeration = cited[0]
    fields = [field for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name not in enumerations
              for field in node.body if isinstance(field, (ast.AnnAssign, ast.Assign))
              and any(isinstance(n, ast.Name) and n.id == enumeration.name for n in ast.walk(field))]
    if statement in fields:
        fields = [statement]
    if not any(isinstance(target, ast.Name) and _field_named(text, target.id)
               for field in fields for target in getattr(field, "targets", None) or [field.target]):
        return False
    members = [_member_pattern(member) for member in enumeration.body if isinstance(member, ast.Assign)]
    if len(members) < 2 or None in members or not all(member.search(text) for member in members):
        return False
    spans = sorted(match.span() for member in members for match in member.finditer(text))
    if not _lists_only_members(text, spans):
        return False
    for member in members:
        text = member.sub(" ", text)
    return not _states_constraint(text)


_UNMATCHED_NOTE_CHARS = 80


def _as_written(raw: str, fragment: str, limit: int = _UNMATCHED_NOTE_CHARS) -> str:
    """``fragment`` (a whitespace-free quote part) as the judge wrote it in
    ``raw``, clipped, so a re-judge can see which part to correct."""
    kept = [i for i, char in enumerate(raw) if not char.isspace()]
    at = "".join(raw[i] for i in kept).replace("'", '"').find(fragment) if fragment else -1
    text = " ".join(raw[kept[at]:kept[at + len(fragment) - 1] + 1].split()) if at >= 0 else fragment
    return text if len(text) <= limit else text[:limit - 3] + "..."


# "sets x", "renders a plot": a word that starts no statement, then another word.
_PROSE_START_RE = re.compile(r"([a-z]+) +[A-Za-z_]")
_STATEMENT_WORDS = frozenset(keyword.kwlist) | {
    "const", "let", "var", "function", "new", "await", "async", "export", "default", "typeof",
    "throw", "catch", "case", "switch", "yield", "delete", "void", "of", "do", "val", "fun",
}
# Of a quote's prose, enough to recognise it.
_PROSE_NOTE_CHARS = 40


def _reads_as_description(text: str) -> bool:
    match = _PROSE_START_RE.match(text.strip())
    return bool(match) and match.group(1) not in _STATEMENT_WORDS


def _where_the_quote_leaves(part: str, source: _Source, path: str, raw: str, cache: dict) -> tuple[str, bool]:
    """For a quote part that matched no span: the source lines it follows from
    their start and what the file has next, or that it reads as prose; ""
    when it follows no line. The flag says a near match was found: the cited
    code is there, the quote of it is malformed."""
    lines = source.normalised
    # Verbatim inside a line, in its code: the line is there, the quote is only part of it.
    # A whole message literal ("Room already booked") is this file's text at any length; a
    # one-word one ("ok", "admin") is in almost any file, so it needs the minimum, like code.
    message = any(char.isspace() for char in _as_written(raw, part, len(raw) + 3))
    inside = next((i for i, line in enumerate(lines) if line.find(part) > 0 and (
        (_string_literal_at(part, source, i) and (message or len(part) >= _MIN_CUT_QUOTE))
        or (len(part) >= _MIN_CUT_QUOTE and _in_code(part, source, i)))), None)
    if inside is not None:
        return f"the part is inside {path} line {inside + 1}; cite that whole line", True
    best = (0, 0, 0, 0)  # characters followed, first line, line where it leaves, whole-line characters
    for start, first in enumerate(lines):
        if not first or not part.startswith(first[:_MIN_CUT_QUOTE]):
            continue
        whole, index = 0, start
        while index < min(start + _MAX_QUOTE_LINES, len(lines)) and part.startswith(lines[index], whole):
            whole, index = whole + len(lines[index]), index + 1
        followed = whole + (len(os.path.commonprefix([part[whole:], lines[index]])) if index < len(lines) else 0)
        if followed > best[0]:
            best = (followed, start, index, whole)
    followed, start, index, whole = best
    if followed < _MIN_CUT_QUOTE or followed >= len(part):
        closest = _closest_statement(part, source, path, raw, cache) if followed < _MIN_CUT_QUOTE else ""
        if closest:
            return closest, True
        prose = _as_written(raw, part, _PROSE_NOTE_CHARS)
        return (f"`{prose}` reads as a description, not code; put it in the note"
                if _reads_as_description(prose) else ""), False
    if index == len(lines):
        return "", False
    text = source.lines[index].strip()
    fits = len(text) <= _UNMATCHED_NOTE_CHARS
    next_line = f"`{text if fits else text[:_UNMATCHED_NOTE_CHARS - 3] + '...'}`"
    # A line too long to show whole is shown where the quote leaves it.
    differs = f"reads {next_line}" if fits else _divergence(text, followed - whole, raw, part[followed:])
    if index == start:
        return f"the part differs from {path} line {index + 1}" + (
            f": {next_line}" if fits else f", which {differs}"), True
    matched = f"the part matches {path} " + (f"lines {start + 1}-{index}" if index > start + 1 else f"line {start + 1}")
    rest = part[whole:]
    resumes = next((k for k in range(index + 1, min(index + 4, len(lines)))
                    if lines[k] and rest.startswith(lines[k][:_MIN_CUT_QUOTE])), None)
    if resumes is not None:
        skipped = f"line {index + 1}" if resumes == index + 1 else f"lines {index + 1}-{resumes}"
        return f"{matched}, then skips {skipped}: {next_line}", True
    prose = _as_written(raw, rest, _PROSE_NOTE_CHARS)
    if _reads_as_description(prose):
        return f"{matched}, then `{prose}` reads as a description; put it in the note", True
    return f"{matched}, then line {index + 1} {differs}", True


# Characters shown on each side of the point where a quote and the file part.
_DIVERGENCE_CONTEXT_CHARS = 24


def _divergence(text: str, offset: int, raw: str, rest: str) -> str:
    """``text`` and the quote side by side where they part: ``offset`` whitespace-free
    characters into ``text``; ``rest`` is the whitespace-free quote from that point."""
    kept = [i for i, char in enumerate(text) if not char.isspace()]
    at = kept[offset] if offset < len(kept) else len(text)
    lead = "..." if at > _DIVERGENCE_CONTEXT_CHARS else ""
    before = text[max(0, at - _DIVERGENCE_CONTEXT_CHARS):at]
    after = text[at:at + _DIVERGENCE_CONTEXT_CHARS] + ("..." if len(text) > at + _DIVERGENCE_CONTEXT_CHARS else "")
    quote = _as_written(raw, rest, _DIVERGENCE_CONTEXT_CHARS + 3)
    return f"has `{lead}{before}{after}` where your quote has `{lead}{before}{quote}`"


def _unparsed_statements(source: _Source, cache: dict) -> list[tuple[int, int, str, str]]:
    """(first line, last line, text on one line, normalised text) of each multi-line
    statement spanning fewer than ``_MAX_QUOTE_LINES`` lines, unparsed once per file
    per verification. A one-line statement is compared line by line instead."""
    key = ("unparsed", source.path)
    if key not in cache:
        statements = []
        for node in ast.walk(source.tree) if source.tree is not None else ():
            if (isinstance(node, ast.stmt) and node.end_lineno > node.lineno
                    and node.end_lineno - _statement_start(node) < _MAX_QUOTE_LINES):
                try:
                    text = " ".join(ast.unparse(node).split())
                except RecursionError:  # too deeply nested to unparse: no candidate
                    continue
                statements.append((_statement_start(node), node.end_lineno, text, _normalise_line(text)))
        cache[key] = statements
    return cache[key]


def _closest_statement(part: str, source: _Source, path: str, raw: str, cache: dict) -> str:
    """For a Python quote that starts no line (``patterns = (`` starts no quote): the
    multi-line statement it follows furthest and where it leaves it; "" when none is close."""
    best: tuple[int, tuple[int, int, str, str] | None] = (0, None)
    for statement in _unparsed_statements(source, cache):
        common = len(os.path.commonprefix([part, statement[3]]))
        if common >= best[0]:  # ast.walk goes outside in: the innermost wins a tie
            best = (common, statement)
    common, statement = best
    if statement is None or common < _MIN_CUT_QUOTE or common >= len(statement[3]):
        return ""
    first, last, text, _ = statement
    return (f"the closest statement is {path} lines {first}-{last}, which "
            + _divergence(text, common, raw, part[common:]))


def _collapsed_text(full: str, cache: dict) -> str:
    """A judged file's whole text in a quote's form (no whitespace, double quotes), read once."""
    key = ("collapsed", full)
    if key not in cache:
        try:
            with open(full, "r", encoding="utf-8-sig", errors="ignore") as handle:
                cache[key] = _collapse(handle.read().replace("'", '"'))
        except OSError:
            cache[key] = ""
    return cache[key]


def _found_elsewhere(part: str, cited: str | None, files: dict[str, str], parsed: dict,
                     quote_cache: dict, sqlite: bool) -> str:
    """For a line that is not in the file the citation names: where it is when exactly
    one other judged file holds it verbatim, "" otherwise. Never relocates the citation."""
    hits = []
    for rel, full in files.items():
        if full == cited or part not in _collapsed_text(full, quote_cache):
            continue
        spans = _quote_spans(part, _load_source(full, parsed, sqlite))
        if spans:
            hits.append(f"{rel} line {spans[0][0]}")
            if len(hits) > 1:
                return ""
    return (f"the line is in {hits[0]}, not in the cited file; the citation must name the file the line is in"
            if hits else "")


_NOT_FOUND = "the quoted executable line was not found in the cited source: "


def _not_found(part: str, source: _Source | None, path: str, raw: str, files: dict[str, str], parsed: dict,
               quote_cache: dict, sqlite: bool) -> tuple[list, str, bool]:
    """The failure of a part found in no span, and whether the cited code exists: the
    line verbatim in another judged file, or a near match in the cited one."""
    elsewhere = _found_elsewhere(part, source.path if source else None, files, parsed, quote_cache, sqlite)
    if elsewhere:
        return [], _NOT_FOUND + elsewhere, True
    where, near = _where_the_quote_leaves(part, source, path, raw, quote_cache) if source else ("", False)
    return [], _NOT_FOUND + (where or f"no source line matches the part `{_as_written(raw, part)}`"), near


def _quoted_statements(raw: str) -> list[list[ast.stmt]]:
    """The quote read as Python: its escapes as written (source such as ``"\\n".join``),
    then as line breaks; each reading that parses to at least one statement."""
    readings = []
    for text in dict.fromkeys((raw, raw.replace("\\n", "\n"))):
        try:
            body = ast.parse(textwrap.dedent(text).strip()).body
        except (SyntaxError, ValueError, RecursionError):
            continue
        if body:
            readings.append(body)
    return readings


def _statement_index(source: _Source, cache: dict) -> dict[str, list[tuple[list[ast.stmt], int]]]:
    """Each statement of a parsed file by its syntax tree without positions, with its
    block and position there. Built once per file per verification, from the tree
    :func:`_load_source` parsed; only statements spanning fewer than
    ``_MAX_QUOTE_LINES`` lines are indexed, which bounds the work on a large file."""
    key = ("statements", source.path)
    if key not in cache:
        index: dict[str, list[tuple[list[ast.stmt], int]]] = {}
        for node in ast.walk(source.tree) if source.tree is not None else ():
            for _, block in ast.iter_fields(node):
                if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                    for position, statement in enumerate(block):
                        if statement.end_lineno - _statement_start(statement) < _MAX_QUOTE_LINES:
                            try:
                                index.setdefault(ast.dump(statement), []).append((block, position))
                            except RecursionError:  # too deeply nested to compare: never matched
                                continue
        cache[key] = index
    return cache[key]


def _syntax_spans(raw: str, source: _Source, cache: dict) -> list[tuple[int, int]]:
    """1-based spans of the statement, or run of consecutive statements, whose syntax
    tree is the quote's: the same code whatever its layout (a formatter's wrapping
    parentheses, a string split into adjacent literals, a call or a small function on
    one line). One changed token changes the tree."""
    index = _statement_index(source, cache)
    for statements in _quoted_statements(raw) if index else ():
        spans = []
        try:
            dumps = [ast.dump(statement) for statement in statements]
            for block, position in index.get(dumps[0], ()):
                run = block[position:position + len(dumps)]
                first, last = _statement_start(run[0]), run[-1].end_lineno
                if (len(run) == len(dumps) and last - first < _MAX_QUOTE_LINES
                        and all(ast.dump(statement) == dump for statement, dump in zip(run[1:], dumps[1:]))):
                    spans.append((first, last))
        except RecursionError:  # too deeply nested to compare: no match
            continue
        if spans:
            return spans
    return []


# Statements whose header a judge quotes alone ("if a or b:"); defs and classes match as text.
_HEADER_STATEMENTS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)
_BLOCK_FIELDS = ("body", "orelse", "handlers", "finalbody")


def _header_dump(node: ast.stmt) -> str:
    """The syntax tree of a compound statement without its blocks: its header."""
    header = type(node)(**{field: [] if field in _BLOCK_FIELDS else getattr(node, field)
                           for field in node._fields})
    return ast.dump(header)


def _header_end(node: ast.stmt, source: _Source) -> int:
    """The line holding a compound statement's closing ``:``: the last line before its body
    with code on it (a comment between the header and the body is not the header)."""
    before_body = range(node.lineno, max(node.lineno, node.body[0].lineno - 1) + 1)
    return max((line for line in before_body if source.code[line - 1].strip()), default=node.lineno)


def _header_spans(raw: str, source: _Source) -> list[tuple[int, int]]:
    """1-based spans of the compound-statement headers whose syntax tree is the quote's,
    a header such as ``if a or b:`` quoted on one line while a formatter wrapped it."""
    text = raw.replace("\\n", "\n").strip()
    if source.tree is None or not text.endswith(":"):
        return []
    try:
        body = ast.parse(textwrap.dedent(text) + " pass").body
        if len(body) != 1 or not isinstance(body[0], _HEADER_STATEMENTS):
            return []
        wanted = _header_dump(body[0])
        return [(node.lineno, _header_end(node, source)) for node in ast.walk(source.tree)
                if isinstance(node, type(body[0])) and _header_dump(node) == wanted]
    except (SyntaxError, ValueError, RecursionError):
        return []


# Where a def collapsed onto its decorator's line starts: "@router.get(...) def f(...):".
_COLLAPSED_DEF_RE = re.compile(r"\s+(?=(?:async\s+)?def\s+\w+\s*\()")
_DECORATOR_ONLY = "the citation must identify an implemented statement, not only a decorator"


def _decorator_dumps(text: str) -> list[str] | None:
    """The syntax trees of ``text``'s decorators when it is decorators and nothing else."""
    try:
        body = ast.parse(textwrap.dedent(text).strip() + "\ndef _decorated(): pass").body
        if len(body) != 1 or not getattr(body[0], "decorator_list", None):
            return None
        return [ast.dump(decorator) for decorator in body[0].decorator_list]
    except (SyntaxError, ValueError, RecursionError):
        return None


def _decorated(source: _Source, cache: dict) -> list[tuple[ast.stmt, set[str]]]:
    """Each def and class of a parsed file with the syntax trees of its decorators."""
    key = ("decorated", source.path)
    if key not in cache:
        decorated = []
        for node in ast.walk(source.tree) if source.tree is not None else ():
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
                try:
                    decorated.append((node, {ast.dump(decorator) for decorator in node.decorator_list}))
                except RecursionError:  # too deeply nested to compare: never matched
                    continue
        cache[key] = decorated
    return cache[key]


def _annotates(source: _Source, end: int, starts: set[int]) -> bool:
    """An annotation ending on line ``end`` sits on a line in ``starts``: only blank
    lines or other annotations lie between."""
    return any(start > end and all(not line.strip() or line.strip().startswith("@")
                                   for line in source.lines[end:start - 1]) for start in starts)


def _locate_decorator(path: str, raw_quote: str, files: dict[str, str], parsed: dict,
                      quote_cache: dict, sqlite: bool) -> tuple[list, str, bool]:
    """A quote that starts with decorators (annotations): located in the file it names
    but proving nothing, so another citation must prove the role. What follows them in
    the quote (the next line, or a def collapsed onto the decorator's line) is located
    as a part of its own, and the decorators must be on that definition."""
    lines = raw_quote.replace("\\n", "\n").strip().splitlines()
    # The decorators: the most leading lines that parse as Python decorators, else the first line.
    count = next((k for k in range(len(lines), 0, -1) if _decorator_dumps("\n".join(lines[:k])) is not None), 1)
    head, tail = "\n".join(lines[:count]), "\n".join(lines[count:]).strip()
    if not tail:
        for split in _COLLAPSED_DEF_RE.finditer(head):
            if _decorator_dumps(head[:split.start()]) is not None:
                head, tail = head[:split.start()], head[split.end():]
                break
    candidates = _resolve_cited_path(path, "", files, quote_cache)
    if len(candidates) != 1:
        return [], "the cited source path is missing, excluded or ambiguous", False
    source = _load_source(candidates[0], parsed, sqlite)
    quoted = _normalise_quote(head)
    decorators = _decorator_dumps(head) if source.tree is not None else None
    if decorators is not None:
        homes = [node for node, dumps in _decorated(source, quote_cache) if set(decorators) <= dumps]
        spans = []
    else:  # not Python (a Java annotation), or not parseable: one line, verbatim
        homes, spans = [], _quote_spans(quoted, source) if "\n" not in head else []
    if not homes and not spans:
        return _not_found(quoted, source, path.strip(), raw_quote, files, parsed, quote_cache, sqlite)
    located = [(source, quoted, [])]
    if tail:
        rest, failure, malformed = _locate_citation(path, tail, files, parsed, quote_cache, sqlite)
        if failure:
            return [], failure, malformed
        starts = {first for _, _, part_spans in rest for first, _ in part_spans}
        on_it = (any(_statement_start(node) <= start <= node.lineno for node in homes for start in starts)
                 or any(_annotates(source, end, starts) for _, end in spans))
        if not on_it:
            return [], (_NOT_FOUND + f"the decorator `{_as_written(raw_quote, quoted)}` is in {path.strip()}, "
                        "but not on the definition quoted with it"), True
        located += rest
    return located, "", False


def _locate_citation(path: str, raw_quote: str, files: dict[str, str], parsed: dict,
                     quote_cache: dict, sqlite: bool) -> tuple[list, str, bool]:
    """The ``(source, part, spans)`` of one ``path: quote`` citation, or why it is
    not source: every part must be code in the file it names, one a whole line.
    The flag of a failure says the cited code exists (a near match in the file, or
    the line in another file): the citation is malformed, not invented."""
    quoted = _evidence_quote(raw_quote)
    if not quoted:
        if raw_quote.replace("\\n", "\n").strip().startswith("@"):
            return _locate_decorator(path, raw_quote, files, parsed, quote_cache, sqlite)
        return [], _DECORATOR_ONLY, False
    candidates = _resolve_cited_path(path, quoted, files, quote_cache)
    if len(candidates) != 1:
        elsewhere = "" if candidates else _found_elsewhere(quoted, None, files, parsed, quote_cache, sqlite)
        return [], "the cited source path is missing, excluded or ambiguous" + (
            f": {elsewhere}" if elsewhere else ""), bool(elsewhere)
    source = _load_source(candidates[0], parsed, sqlite)
    spans = list(_quote_spans(quoted, source))
    literal = _evidence_quote(raw_quote, escapes=False)
    if not spans and literal and literal != quoted:
        # The quote may copy a source escape, e.g. "\n".join(...).
        spans = list(_quote_spans(literal, source))
        quoted = literal if spans else quoted
    located = [(source, quoted, spans)]
    unmatched, unmatched_in = quoted, (source, path)
    if not spans:
        located = []
        literals: list[str] = []
        for part_path, part, cut in _citation_parts(raw_quote, path, files):
            homes = _resolve_cited_path(part_path, part, files, quote_cache)
            part_source = _load_source(homes[0], parsed, sqlite) if len(homes) == 1 else None
            spans = list(_quote_spans(part, part_source, cut)) if part_source else []
            # A part need not be a whole statement ("{ save(n); done = true }"
            # splits at its ";"), but it must be code in that file, or a whole string
            # literal of it beside another part that is ("if end < start: ... "msg"").
            lines = range(len(part_source.lines)) if part_source else range(0)
            if not spans and not any(_in_code(part, part_source, i) for i in lines):
                if any(_string_literal_at(part, part_source, i) for i in lines):
                    literals.append(part)
                    continue
                located, unmatched, unmatched_in = [], part, (part_source, part_path)
                break
            located.append((part_source, part, spans))
        if literals and not located and unmatched_in == (source, path):
            # Only string literals: real text of the file, but no code proves anything.
            unmatched = literals[0]
    if any(spans for _, _, spans in located):
        return located, "", False
    syntax = _syntax_spans(raw_quote, source, quote_cache) or _header_spans(raw_quote, source)
    if syntax:
        # The role check reads the matched source without comments, never the quote:
        # a comment the judge wrote into it is not code.
        return [(source, "".join(source.bare[first - 1:last]), [(first, last)]) for first, last in syntax], "", False
    return _not_found(unmatched, unmatched_in[0], unmatched_in[1].strip(), raw_quote,
                      files, parsed, quote_cache, sqlite)


def verify_evidence(verdicts: list[dict], output_dir: str, imports: ImportIndex | None = None) -> list[dict]:
    """Re-check every 'implemented' citation against the output tree.

    The evidence is a list of ``{"path", "line"}`` citations, or the older
    ``path: quote`` string. The cited application source must exist (an
    unambiguous path suffix is accepted) and contain the quoted code. Every
    citation, and every part of an abridged or listed quote ("a ... b",
    "a; b", "x.py: a and y.py: b"), must be found in the file it names, and one
    found line must enforce or perform the requirement; each is checked on its
    own text, so an invented part can never supply the proof. Never search a different
    file or the run's own recipe to rescue a bad citation. A declaration alone
    cannot prove enforcement or behaviour. Failed evidence is ``unverified``:
    it does not prove that no implementation exists elsewhere. A verified
    verdict records ``cited_file_hashes`` ({path: sha256}) of the files it cites
    and ``dependency_file_hashes`` of the files those rest on (:meth:`ImportIndex.dependencies`).
    ``imports`` is an index of this same tree a caller reuses across calls; one is built
    on the first verified verdict otherwise.
    """
    files = _source_files(output_dir)
    rel_of = {full: rel for rel, full in files.items()}
    sqlite = workspace_uses_sqlite(output_dir)
    parsed: dict[str, _Source] = {}
    quote_cache: dict = {}
    checked: list[dict] = []
    for verdict in verdicts:
        item = dict(verdict)
        for key in (*VERDICT_FINGERPRINT_KEYS, EVIDENCE_UNLOCATED):
            item.pop(key, None)
        if item.get("kind") == "verification":
            item.update(status="unverified", evidence="",
                        note="requirement extraction completeness has not been established")
            checked.append(item)
            continue
        if item.get("status") != "implemented":
            checked.append(item)
            continue
        evidence = item.get("evidence")
        structured = isinstance(evidence, list)
        citations = [(str(c.get("path") or ""), str(c.get("line") or "")[:_MAX_EVIDENCE_CHARS])
                     for c in evidence if isinstance(c, dict)] if structured else [
            (evidence or "")[:_MAX_EVIDENCE_CHARS].partition(":")[::2]]
        # An empty citation is no citation, not one of absent code (r13-p81 R8).
        citations = [(path, quote) for path, quote in citations if quote.replace("\\n", "").strip()]
        located, reason = [], "" if citations else "no source line is cited"
        for index, (path, raw_quote) in enumerate(citations, 1):
            found, failure, malformed = _locate_citation(path, raw_quote, files, parsed, quote_cache, sqlite)
            if failure:
                # The first failure is the reason; every failure decides whether the cited code exists.
                reason = reason or (f"citation {index} ({path.strip()}): {failure}" if structured else failure)
                item[EVIDENCE_UNLOCATED] = CITATION_MALFORMED if malformed and item.get(
                    EVIDENCE_UNLOCATED, CITATION_MALFORMED) == CITATION_MALFORMED else CITATION_ABSENT
                if item[EVIDENCE_UNLOCATED] == CITATION_ABSENT:
                    break
                continue
            located += [(index, path, *part) for part in found]
        if not reason:
            # A part found only inside a line proves no role: one whole line must.
            reason = (_DECORATOR_ONLY if located and all(part.startswith("@") for _, _, _, part, _ in located)
                      else "no cited part is a whole source line")
            for index, path, part_source, part, spans in located:
                if not spans:
                    continue
                for first_line, lineno in spans:
                    reason = _span_failure(item, part_source, first_line, lineno, part)
                    if not reason:
                        break
                if not reason:
                    break
                if structured and spans:
                    reason = f"citation {index} ({path.strip()}): {reason}"
        if reason:
            item["status"] = "unverified"
            item["note"] = " ".join(f"{item.get('note', '')} [{reason}]".split())
        else:
            cited = {rel_of[part_source.path] for _, _, part_source, _, _ in located}
            if imports is None:
                imports = ImportIndex(output_dir, files)
            item["cited_file_hashes"] = {rel: cited_file_hash(files[rel]) for rel in sorted(cited)}
            item["dependency_file_hashes"] = {rel: cited_file_hash(files[rel])
                                              for rel in sorted(imports.dependencies(cited))}
        checked.append(item)
    return checked


VERDICT_FINGERPRINT_KEYS = ("cited_file_hashes", "dependency_file_hashes")
# Set on a verdict with a citation not found in the file it names. CITATION_MALFORMED:
# every such citation's code exists (a near match in the file, or the line verbatim in
# one other file), so only the judge can correct it. CITATION_ABSENT: some cited code
# exists nowhere, so implementing it may be the fix.
EVIDENCE_UNLOCATED = "evidence_unlocated"
CITATION_MALFORMED = "near_match"
CITATION_ABSENT = "absent"

_JS_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".vue", ".svelte", ".astro"})
# ``from './x'``, ``import './x'``, ``import('./x')``, ``require('./x')``: relative specifiers only.
_JS_IMPORT_RE = re.compile(r"""(?:\bfrom|\bimport|\brequire)\s*\(?\s*["'](\.{1,2}/[^"'\n]+)["']""")
_JS_RESOLVE_SUFFIXES = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".vue", ".svelte",
                        "/index.ts", "/index.tsx", "/index.js", "/index.jsx")


def _python_import_stems(rel: str, tree: ast.AST) -> set[str]:
    """Workspace paths (no suffix) a module's imports can name. An absolute
    import resolves against every enclosing folder: a service runs from its own."""
    folder = posixpath.dirname(rel)
    ancestors = _ancestors(folder)
    stems: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            stems |= {posixpath.join(base, alias.name.replace(".", "/")) for alias in node.names for base in ancestors}
        elif isinstance(node, ast.ImportFrom):
            bases = ancestors
            if node.level:
                base = folder
                for _ in range(node.level - 1):
                    base = posixpath.dirname(base)
                bases = [base]
            module = (node.module or "").replace(".", "/")
            for base in bases:
                package = posixpath.join(base, module) if module else base
                if module:
                    stems.add(package)
                stems |= {posixpath.join(package, alias.name) for alias in node.names if alias.name != "*"}
    return stems


def _local_imports(rel: str, text: str, files) -> set[str]:
    """The judged files ``rel`` imports directly (Python, or relative JS/TS)."""
    suffix = Path(rel).suffix.lower()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return set()
        candidates = {stem + tail for stem in _python_import_stems(rel, tree) for tail in (".py", "/__init__.py")}
    elif suffix in _JS_SUFFIXES:
        folder = posixpath.dirname(rel)
        candidates = {posixpath.normpath(posixpath.join(folder, spec)) + tail
                      for spec in _JS_IMPORT_RE.findall(text) for tail in _JS_RESOLVE_SUFFIXES}
    else:
        return set()
    return {path for path in candidates if path in files and path != rel}


def _module_name(rel: str) -> str:
    """The name an import of ``rel`` spells: its stem, or its folder for a package/index file."""
    stem = posixpath.splitext(posixpath.basename(rel))[0]
    return posixpath.basename(posixpath.dirname(rel)) if stem in ("__init__", "index") else stem


class ImportIndex:
    """Direct local imports between the judged files of one tree, read and parsed on demand.

    A virtualenv the agent made (any folder holding ``pyvenv.cfg``, as ``ToolExecutor._list_dir``
    recognises one) is not the app: its files are neither imports nor importers.
    """

    def __init__(self, output_dir: str, files: dict[str, str] | None = None) -> None:
        self._output_dir = output_dir
        self._judged = files  # walked on first use when not given
        self._app: dict[str, str] | None = None
        self._texts: dict[str, str] = {}
        self._imports: dict[str, set[str]] = {}
        self._importers: dict[str, set[str]] = {}

    @property
    def _files(self) -> dict[str, str]:
        if self._app is None:
            files = self._judged if self._judged is not None else _source_files(self._output_dir)
            folders = {parent for rel in files for parent in _ancestors(posixpath.dirname(rel))}
            venvs = {folder for folder in folders
                     if folder and os.path.isfile(os.path.join(self._output_dir, folder, "pyvenv.cfg"))}
            self._app = {rel: full for rel, full in files.items()
                         if not venvs.intersection(_ancestors(posixpath.dirname(rel)))}
        return self._app

    def _text(self, rel: str) -> str:
        if rel not in self._texts:
            try:
                self._texts[rel] = Path(self._files[rel]).read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                self._texts[rel] = ""
        return self._texts[rel]

    def imports(self, rel: str) -> set[str]:
        if rel not in self._imports:
            self._imports[rel] = _local_imports(rel, self._text(rel), self._files) if rel in self._files else set()
        return self._imports[rel]

    def importers(self, rel: str) -> set[str]:
        """Every file importing ``rel``; only files that spell its name are parsed."""
        if rel not in self._importers:
            name = _module_name(rel)
            self._importers[rel] = {other for other in self._files if other != rel
                                    and name in self._text(other) and rel in self.imports(other)}
        return self._importers[rel]

    def dependencies(self, cited: set[str]) -> set[str]:
        """What a cited file's behaviour rests on beyond its own bytes: its direct
        local imports, and the entry files mounting it (a file importing a cited
        file that no judged file imports). One level only."""
        imported = set().union(*(self.imports(rel) for rel in cited))
        entries = {other for rel in cited for other in self.importers(rel) if not self.importers(other)}
        return (imported | entries) - cited


def _ancestors(folder: str) -> list[str]:
    """``folder`` and every folder above it, up to the workspace root ("")."""
    chain = [folder]
    while chain[-1]:
        chain.append(posixpath.dirname(chain[-1]))
    return chain


# Cited code no harness check runs (no screen is rendered); any other cited file is backend code.
_FRONTEND_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".html", ".css")


def _path_segments(path: str) -> tuple[str, ...]:
    """A route or a called path as segments, each parameter or numeric id as ``{}``."""
    path = _ROUTE_PARAM_RE.sub("{}", str(path).split("?", 1)[0])
    return tuple("{}" if part.isdigit() else part for part in path.split("/") if part)


def _declared_routes(full: str) -> set[tuple[frozenset[str] | None, tuple[str, ...], int, int]]:
    """The routes a Python file's decorators and ``add_api_route`` calls declare: (methods or
    None for any, segments, first line, last line of the handler or call), under the literal
    ``prefix=`` of the router they are on. A
    router whose prefix is not a literal declares nothing that can be matched. Prefixes
    added where the router is included are not known: calls are matched by suffix."""
    try:
        tree = ast.parse(Path(full).read_text(encoding="utf-8-sig", errors="ignore"))
    except (OSError, SyntaxError, ValueError, RecursionError):
        return set()
    prefixes: dict[str, str | None] = {}  # router name -> its literal prefix, None when not literal
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            prefix = next((kw.value for kw in node.value.keywords if kw.arg == "prefix"), None)
            for target in getattr(node, "targets", None) or [node.target]:
                if isinstance(target, ast.Name) and prefix is not None:
                    prefixes[target.id] = _route_literal(prefix)
    routes = set()
    for node in ast.walk(tree):
        calls = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = [d for d in node.decorator_list if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                     and d.func.attr in _ROUTE_VERBS | _ROUTE_ANY_VERB]
            lines = (_statement_start(node), node.end_lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_api_route":
            calls = [node]
            lines = (node.lineno, node.end_lineno)
        for call in calls:
            route = _route_literal(call.args[0] if call.args else next(
                (kw.value for kw in call.keywords if kw.arg == "path"), None))
            receiver = getattr(call.func.value, "id", None)
            if route is None or (receiver in prefixes and prefixes[receiver] is None):
                continue
            prefix = prefixes.get(receiver) or ""
            routes.add((_route_methods(call, call.func.attr), _path_segments(f"{prefix}/{route}"), *lines))
    return routes


def _route_called(routes: set, method: str, path: str) -> bool:
    """A call ends with one of ``routes`` (the router prefix is not known), method included.
    A route of parameters only (``/{id}``) proves nothing about which file served it."""
    called = _path_segments(path)
    for methods, segments, *_lines in routes:
        if (not any(part != "{}" for part in segments) or len(segments) > len(called)
                or (methods is not None and method.upper() not in methods)):
            continue
        tail = called[len(called) - len(segments):]
        if all(want == "{}" or want == got for want, got in zip(segments, tail)):
            return True
    return False


# Why no screen check ran, by the state of the frontend build on the final tree.
_SCREEN_REASONS = {"passed": "the frontend builds", "failed": "the frontend build failed",
                   "not_run": "the frontend build check did not run", "none": "no frontend"}


def _cited_lines(full: str, quotes: list[str]) -> set[int]:
    """The 1-based lines of a cited file the quotes locate on."""
    source = _load_source(full, {}, False)
    return {line for quote in quotes for first, last in _quote_spans(_evidence_quote(quote), source)
            for line in range(first, last + 1)}


def with_verification_levels(verdicts: list[dict], output_dir: str, *, frontend_build: str,
                             exercised: set[tuple[str, str]], workflows_passed: int) -> list[dict]:
    """Each implemented verdict with what besides its citation checked it, the API half and
    the screens half reported apart.

    The API half is exercised when a passing ``test_api`` call (``exercised``: method, path)
    hit the route a cited line is on (its decorator or handler), or, when no cited line is
    on a route, any route of a cited backend file; that call is named. No harness check
    renders a screen, so a cited frontend file is never exercised (``frontend_build``:
    passed, failed, not_run or none says why). ``verification``: ``verified`` when every
    cited half was exercised, ``api_verified`` when the API half was and screens are cited,
    ``unverified`` otherwise. Never a blocker.
    """
    files = _source_files(output_dir)
    routes_cache: dict[str, set] = {}
    leveled = []
    for item in verdicts:
        if item.get("status") != "implemented":
            leveled.append(item)
            continue
        evidence = item.get("evidence")
        cited = ([(str(c.get("path") or "").strip().replace("\\", "/"), str(c.get("line") or ""))
                  for c in evidence if isinstance(c, dict)] if isinstance(evidence, list)
                 else [(path.strip().replace("\\", "/"), quote) for path, quote in [str(evidence or "").partition(":")[::2]]])
        cited = [(path, quote) for path, quote in cited if path]
        screens = any(path.lower().endswith(_FRONTEND_SUFFIXES) for path, _ in cited)
        backend = [(path, quote) for path, quote in cited if not path.lower().endswith(_FRONTEND_SUFFIXES)]
        halves, api_exercised = [], False
        if backend:
            on_cited, in_files = set(), set()
            for path, quote in backend:
                full = files.get(path) or next((v for k, v in files.items() if k.endswith("/" + path)), None)
                if full and full.endswith(".py"):
                    routes = routes_cache.setdefault(full, _declared_routes(full))
                    lines = _cited_lines(full, [quote])
                    in_files |= routes
                    on_cited |= {route for route in routes if any(route[2] <= line <= route[3] for line in lines)}
            routes, which = (on_cited, "the cited route") if on_cited else (in_files, "a route of the cited file")
            call = next((f"{method} {path}" for method, path in sorted(exercised) if _route_called(routes, method, path)),
                        None)
            api_exercised = call is not None
            halves.append(f"API: a passing workflow called {call}, {which}" if call
                          else "API: a workflow passed but called no cited route" if workflows_passed
                          else "API: no workflow passed")
        if screens:
            halves.append(f"screens: not run ({_SCREEN_REASONS.get(frontend_build, frontend_build)})")
        level = ("verified" if api_exercised and not screens else "api_verified" if api_exercised else "unverified")
        note = ("Cited in the source; " if level == "verified" else "Found in the source; ") + "; ".join(
            halves or ["nothing cited"])
        leveled.append({**item, "verification": level, "verification_note": note})
    return leveled
def without_fingerprints(verdicts: list[dict]) -> list[dict]:
    """The verdicts without the file hashes only a kept verdict's check reads."""
    return [{key: value for key, value in item.items() if key not in VERDICT_FINGERPRINT_KEYS}
            for item in verdicts]


def cited_file_hash(full_path: str) -> str:
    """sha256 of a file's raw bytes ("" when unreadable, which never matches)."""
    try:
        return hashlib.sha256(Path(full_path).read_bytes()).hexdigest()
    except OSError:
        return ""


def scope_to_output(verdicts: list[dict], has_frontend: bool) -> list[dict]:
    """Mark unresolved UI requirements out of scope on an output with no frontend.

    A backend-only generator cannot satisfy or show a screen requirement, so
    judging one against backend source can only return a blocker the fix loop
    has no files to resolve.
    """
    if has_frontend:
        return verdicts
    return [dict(v, status="out_of_scope", evidence="",
                 note="UI requirement; this output has no frontend")
            if v.get("kind") == "ui" and v.get("status") != "implemented" else v
            for v in verdicts]


def ledger_issues(verdicts: list[dict]) -> list[str]:
    """Unresolved requirements block completion without claiming they are absent."""
    issues: list[str] = []
    for v in verdicts:
        label = f"R{v['id']} — {v['text']}"
        note = f" ({v['note']})" if v.get("note") else ""
        status = v.get("status")
        if status == "out_of_scope":
            # Unclassified prefix: a warning, never a blocker.
            issues.append(f"requirement out of scope: {label}{note}")
            continue
        if v.get("kind") == "verification":
            issues.append(f"requirement unverified: {label}{note}")
            continue
        if v.get("kind") == ENTITY_CRUD_KIND:
            if status != "implemented":
                # Unclassified prefix: a warning, like the web acceptance matrix.
                issues.append(f"entity coverage: {v['text']} ({status}){note}")
            continue
        if status == "missing":
            issues.append(f"requirement: {label} is not implemented{note}")
        elif status == "partial":
            issues.append(f"requirement partial: {label}{note}; inspect and complete the missing behaviour")
        elif status == "unverified":
            issues.append(
                f"requirement unverified: {label} — the cited evidence "
                f"'{render_evidence(v.get('evidence'))}' does not verify it{note}; implement the behaviour only if it is really absent: "
                "the citation is re-judged automatically"
            )
    return issues
