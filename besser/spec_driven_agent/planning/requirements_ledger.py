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
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple
import re
import time

from pygments.token import Comment

from besser.spec_driven_agent.planning.code_text import _lexer_for, code_only
from besser.spec_driven_agent.planning.gap_analyzer import _chat_supports_kwargs, _is_real_provider
from besser.spec_driven_agent.planning.specification import validate_specification
from besser.spec_driven_agent.planning.target import ENTITY_CRUD_KIND
from besser.spec_driven_agent.providers.tool_input import coerce_to_schema
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

_MAX_REQUIREMENTS = 128
_EXTRACTION_INCOMPLETE = (
    "Requirement extraction reached its item limit; coverage of the full original "
    "specification is unverified. Complete extraction and audit the remaining "
    "specification before declaring the application complete. This is a "
    "verification obligation, not an application feature."
)
_STATUSES = ("implemented", "partial", "missing", "unverified")

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
_CONSTRAINT_WORDS = (
    "unique", "must not", "may not", "may never", "cannot", "can not",
    "never", "at least", "at most", "no more than", "exceed", "only if",
    "only when", "only after", "valid", "format", "shape", "match",
    "overlap", "before", "after", "greater", "less", "between", "positive",
    "non-negative", "refuse", "reject",
)


def _demands_enforcement(kind: str, text: str) -> bool:
    if kind == "uniqueness":
        return True
    low = text.lower()
    return kind in ("validation", "rule") and any(w in low for w in _CONSTRAINT_WORDS)


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


def _in_code(quoted: str, source: _Source, index: int) -> bool:
    """The quote is verbatim in the line, and not inside its strings or comment.

    Both tests are needed: removing strings alone makes CheckConstraint("a > 0")
    and CheckConstraint("b > 0") the same text.
    """
    return (quoted in source.normalised[index]
            and _collapse(code_only(quoted, source.path)) in source.code[index])


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
    inline = _is_complete_call(quoted) or len(_collapse(code_only(quoted, source.path))) >= _MIN_INLINE_QUOTE
    for start, first in enumerate(normalised):
        # A line may end in the block it opens ("Card(...) {", "def f():"), which
        # the judge leaves off; nothing else may be missing. A complete call
        # may also be quoted from inside its line ("onPressed: () => save(x),"),
        # and a line's start when an ellipsis says the rest was left out.
        if (first == quoted
                or (first.startswith(quoted) and set(first[len(quoted):]) <= _BLOCK_PUNCTUATION)
                or (cut and len(quoted) >= _MIN_CUT_QUOTE and first.startswith(quoted))
                or (inline and _in_code(quoted, source, start))):
            yield start + 1, start + 1
            continue
        if not first or not target.startswith(_joined_strings(first)):
            continue
        joined = first
        for end in range(start + 1, min(start + _MAX_QUOTE_LINES, len(normalised))):
            earlier = _joined_strings(joined)
            joined += normalised[end]
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
    f"- At most {_MAX_REQUIREMENTS} items. Never invent a requirement the "
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
                "maxItems": _MAX_REQUIREMENTS,
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
    "go in the note, never in a citation. A "
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
                                "line copied exactly from it."
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


def extract_requirements(instructions: str, llm_client) -> list[dict] | None:
    """``[{"id": 1, "text": ..., "kind": ...}, ...]`` from the user's request,
    or None when no real provider is available or the call failed."""
    validate_specification(instructions)
    if not _is_real_provider(llm_client):
        return None
    request = original_request(instructions)
    if not request:
        return None
    payload = _call_with_tool(
        llm_client, _EXTRACT_SYSTEM_PROMPT,
        f"## The user's request\n\n{request}", _SUBMIT_REQUIREMENTS_TOOL,
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
    if len(extracted) >= _MAX_REQUIREMENTS:
        requirements.append({
            "id": len(requirements) + 1, "text": _EXTRACTION_INCOMPLETE,
            "kind": "verification",
        })
    return requirements


def render_requirements(requirements: list[dict] | None) -> str:
    """``R1. text`` lines for the planner prompt, the Phase 2 system prompt and
    the judge, so every party names a requirement the same way."""
    return "\n".join(f"R{r['id']}. {r['text']}" for r in requirements or [])


def _digest_priority(rel: str, untouched: frozenset[str] = frozenset()) -> tuple[int, str]:
    if rel.endswith(("pydantic_classes.py", "sql_alchemy.py")):
        return (0, rel)
    router = "/routers/" in rel or rel.startswith("routers/")
    if router or rel.endswith(".py"):
        # A CRUD router as generated holds no hand-written logic; a service module does.
        return (3 if rel in untouched else 1 if router else 2, rel)
    if "/pages/" in f"/{rel}" or "/components/" in f"/{rel}":
        return (4, rel)
    return (5, rel)


def _untouched_generator_files(output_dir: str, files: dict[str, str]) -> frozenset[str]:
    """The judged files still byte-identical to the stored generator output."""
    for out_dir in sorted({rel.split("/", 1)[0] for rel in files if "/" in rel}):
        try:
            stored = load_generator_output(output_dir, out_dir)
        except GeneratorOutputError:
            continue
        if stored is None:
            return frozenset()
        untouched = set()
        for rel, content in stored.files.items():
            try:
                if rel in files and Path(files[rel]).read_bytes() == content:
                    untouched.add(rel)
            except OSError:
                continue
        return frozenset(untouched)
    return frozenset()


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
    # Which files are untouched generator output orders the digest.
    fingerprint.update(json.dumps(sorted(_untouched_generator_files(output_dir, files))).encode("utf-8"))
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
    files = _source_files(output_dir)
    untouched = _untouched_generator_files(output_dir, files)
    focused = set()
    for path in (focus_paths or [])[:10]:
        if not isinstance(path, str):
            continue
        path = path.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        candidates = [path] if path in files else [rel for rel in files if rel.endswith("/" + path)]
        if path and len(candidates) == 1:
            focused.add(candidates[0])
    if focus_paths and not focused:
        logger.warning("Requirements ledger: no cited path resolved for citation repair; "
                       "the repair reads the whole app")
    footer_budget = min(6000, max(256, _DIGEST_MAX_TOTAL_CHARS // 10), _DIGEST_MAX_TOTAL_CHARS)
    source_budget = _DIGEST_MAX_TOTAL_CHARS - footer_budget
    entries = []
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
        header = f"### {rel}\n"
        cap = max(0, source_budget - len(header) - 200) if rel in focused else _DIGEST_MAX_FILE_CHARS
        truncated = len(content) > cap
        chunk = header + content[:cap]
        if truncated:
            chunk += (f"\n[DIGEST TRUNCATED: {rel}; showing {cap} of {len(content)} code characters. "
                      "Do not infer absence from omitted code.]\n")
        entries.append(((0 if rel in focused else 1, *_digest_priority(rel, untouched)), rel, chunk, truncated))
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
    for _, rel, chunk, truncated in entries:
        if rel not in reserved:
            if total + len(chunk) + 2 > source_budget:
                omitted.append(rel)
                continue
            total += len(chunk) + 2
        parts.append(chunk)
        if truncated:
            truncated_paths.append(rel)
    footer = (
        "\n\n## Digest coverage (not application source)\n"
        f"Files shown: {len(parts)}; truncated excerpts: {len(truncated_paths)}; "
        f"omitted/unreadable files: {len(omitted)}.\n"
        "Omitted or truncated code is unknown, not evidence of absent behavior.\n"
        + ("This is a focused excerpt: only the cited files and the schema files are shown; "
           "the rest of the application is omitted, not absent.\n" if omitted_by_focus else "")
        + ("Truncated files: " + ", ".join(truncated_paths) + "\n" if truncated_paths else "")
        + ("Omitted files: " + ", ".join(omitted) + "\n" if omitted else "")
    )
    if len(footer) > footer_budget:
        footer = footer[:max(0, footer_budget - 42)] + "\n[Coverage path listing also truncated.]\n"
    return "\n\n".join(parts) + footer[:footer_budget]


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
            "id": rid,
            "text": by_id[rid]["text"],
            "kind": by_id[rid].get("kind", ""),
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
                if isinstance(item, dict) and item.get("id") in requested_ids][:_MAX_REQUIREMENTS]
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
                "id": rid, "text": req["text"], "kind": "verification",
                "status": "unverified", "evidence": "",
                "note": "requirement extraction completeness has not been established",
            }
            continue
        verdicts.setdefault(rid, {
            "id": rid, "text": req["text"], "kind": req.get("kind", ""),
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
            parsed[candidate] = _Source(candidate, [], [], [], None, [], set(), [])
            return parsed[candidate]
        tree = None
        if candidate.endswith(".py"):
            try:
                tree = ast.parse(source)
            except SyntaxError:
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
    if (_demands_enforcement(item.get("kind", ""), item.get("text", ""))
            and not isinstance(statement, (ast.If, ast.Raise, ast.Assert))
            and not source.lines[first_line - 1].lstrip().startswith(("if ", "if(", "if\t", "else if "))
            and not any(tok in quote.lower() for tok in _ENFORCEMENT_TOKENS)):
        return "the cited line declares data; it does not demonstrate enforcement"
    return ""


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


def _where_the_quote_leaves(part: str, source: _Source, path: str, raw: str) -> str:
    """For a quote part that matched no span: the source lines it follows from
    their start and what the file has next, or that it reads as prose; ""
    when it follows no line."""
    lines = source.normalised
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
        prose = _as_written(raw, part, _PROSE_NOTE_CHARS)
        return f"`{prose}` reads as a description, not code; put it in the note" if _reads_as_description(prose) else ""
    if index == len(lines):
        return ""
    text = source.lines[index].strip()
    next_line = f"`{text if len(text) <= _UNMATCHED_NOTE_CHARS else text[:_UNMATCHED_NOTE_CHARS - 3] + '...'}`"
    if index == start:
        return f"the part differs from {path} line {index + 1}: {next_line}"
    matched = f"the part matches {path} " + (f"lines {start + 1}-{index}" if index > start + 1 else f"line {start + 1}")
    rest = part[whole:]
    resumes = next((k for k in range(index + 1, min(index + 4, len(lines)))
                    if lines[k] and rest.startswith(lines[k][:_MIN_CUT_QUOTE])), None)
    if resumes is not None:
        skipped = f"line {index + 1}" if resumes == index + 1 else f"lines {index + 1}-{resumes}"
        return f"{matched}, then skips {skipped}: {next_line}"
    prose = _as_written(raw, rest, _PROSE_NOTE_CHARS)
    if _reads_as_description(prose):
        return f"{matched}, then `{prose}` reads as a description; put it in the note"
    return f"{matched}, then line {index + 1} reads {next_line}"


def _locate_citation(path: str, raw_quote: str, files: dict[str, str], parsed: dict,
                     quote_cache: dict, sqlite: bool) -> tuple[list, str]:
    """The ``(source, part, spans)`` of one ``path: quote`` citation, or why it is
    not source: every part must be code in the file it names, one a whole line."""
    quoted = _evidence_quote(raw_quote)
    if not quoted:
        return [], "the citation must identify an implemented statement, not only a decorator"
    candidates = _resolve_cited_path(path, quoted, files, quote_cache)
    if len(candidates) != 1:
        return [], "the cited source path is missing, excluded or ambiguous"
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
        for part_path, part, cut in _citation_parts(raw_quote, path, files):
            homes = _resolve_cited_path(part_path, part, files, quote_cache)
            part_source = _load_source(homes[0], parsed, sqlite) if len(homes) == 1 else None
            spans = list(_quote_spans(part, part_source, cut)) if part_source else []
            # A part need not be a whole statement ("{ save(n); done = true }"
            # splits at its ";"), but it must be code in that file.
            if not spans and not (part_source and any(
                    _in_code(part, part_source, i) for i in range(len(part_source.lines)))):
                located, unmatched, unmatched_in = [], part, (part_source, part_path)
                break
            located.append((part_source, part, spans))
    if any(spans for _, _, spans in located):
        return located, ""
    where = _where_the_quote_leaves(unmatched, unmatched_in[0], unmatched_in[1].strip(), raw_quote) if (
        unmatched_in[0]) else ""
    return [], ("the quoted executable line was not found in the cited source: "
                + (where or f"no source line matches the part `{_as_written(raw_quote, unmatched)}`"))


def verify_evidence(verdicts: list[dict], output_dir: str) -> list[dict]:
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
    it does not prove that no implementation exists elsewhere.
    """
    files = _source_files(output_dir)
    sqlite = workspace_uses_sqlite(output_dir)
    parsed: dict[str, _Source] = {}
    quote_cache: dict = {}
    checked: list[dict] = []
    for verdict in verdicts:
        item = dict(verdict)
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
        located, reason = [], "" if citations else "no source line is cited"
        for index, (path, raw_quote) in enumerate(citations, 1):
            found, failure = _locate_citation(path, raw_quote, files, parsed, quote_cache, sqlite)
            if failure:
                reason = f"citation {index} ({path.strip()}): {failure}" if structured else failure
                break
            located += [(index, path, *part) for part in found]
        if not reason:
            # A part found only inside a line proves no role: one whole line must.
            reason = "no cited part is a whole source line"
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
        checked.append(item)
    return checked


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
                f"'{render_evidence(v.get('evidence'))}' does not verify it{note}; inspect the actual source "
                "and correct the citation, implementing behaviour only if it is absent"
            )
    return issues
