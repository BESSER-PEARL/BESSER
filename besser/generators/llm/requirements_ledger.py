"""Requirements ledger: what the user asked for, checked against the app.

The gap analyser plans from the spec, but a plan is not a verification. Run
19h35 (2026-09-18, hotel, Qwen3-30B) planned the guest-capacity rule twice and
shipped without it, and never planned the unique room number or the extra
charges at all; every static gate said the run succeeded. A coding agent that
reads the spec directly (the opencode build of the same spec) got all three.

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
import json
import logging
import os
from pathlib import Path
import re

from besser.generators.llm.gap_analyzer import _chat_supports_kwargs, _is_real_provider
from besser.generators.llm.specification import validate_specification
from besser.generators.llm.write_diagnostics import (
    python_structural_diagnostics, workspace_uses_sqlite,
)

logger = logging.getLogger(__name__)

# The modeling agent appends the user's own text under this heading (see
# smart_generation_handler.py); everything before it is an LLM summary.
VERBATIM_MARKER = "## The user's original request, verbatim"

_MAX_REQUIREMENTS = 128
_EXTRACTION_INCOMPLETE = (
    "Requirement extraction reached its item limit; coverage of the full original "
    "specification is unverified. Complete extraction and audit the remaining "
    "specification before declaring the application complete. This is a "
    "verification obligation, not an application feature."
)
_STATUSES = ("implemented", "partial", "missing", "unverified")

# What the judge reads: the schemas and the ORM first (validators, unique
# columns), then the routers (where rules are enforced), then the pages. The
# whole 19h35 hotel app compacts to ~150k chars (~38k tokens); a router with
# nested creation is ~20k on its own, so the per-file cap must hold one.
_DIGEST_MAX_TOTAL_CHARS = 200_000
_DIGEST_MAX_FILE_CHARS = 24_000

# For these kinds an "implemented" verdict must cite a line that enforces
# something. On the 19h35 app the judge marked the unique room number, the
# at-least-one-room rule and the guest-capacity rule implemented by citing a
# column or a relationship; none of the three is enforced anywhere.
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


def _normalise_quote(text: str) -> str:
    """The judge's quote, made comparable to file text: literal escape
    sequences become spaces, single quotes become double quotes (the live
    judge rewrote ``Mapped_["Booking"]`` as ``Mapped_['Booking']``), and
    only the first quoted line counts - later lines are often abridged."""
    text = text.replace("\\n", "\n").replace("\\t", " ")
    first = text.strip().split("\n", 1)[0]
    # read_file numbers what the model sees; no real source line starts with
    # "175| ", so such a prefix is always a copied artifact.
    first = _READ_NUMBER_PREFIX_RE.sub("", first, count=1)
    return _collapse(first.replace("'", '"'))


def _evidence_quote(text: str) -> str:
    """Keep a decorated function's identity instead of a shared registration line.

    The judge sometimes quotes a block despite the one-line contract. Taking
    only ``@model_validator(mode='after')`` confused a no-op capacity validator
    with a real date validator elsewhere in the same file.
    """
    lines = text.replace("\\n", "\n").strip().splitlines()
    if lines and lines[0].lstrip().startswith("@"):
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("@"):
                continue
            return _normalise_quote(line) if stripped.startswith(("def ", "async def ")) else ""
        return ""
    return _normalise_quote(text)


_DIGEST_SKIP_NAMES = frozenset({"bal_stdlib.py", "database.py", "__init__.py"})
_DIGEST_SKIP_DIRS = frozenset({
    "node_modules", "dist", "build", "__pycache__", "venv", "tests", "test",
    "__tests__", "fixtures", "coverage", "verification",
})
_ARTIFACT_ROOT_DIRS = frozenset({"reports", "logs"})
_SOURCE_SUFFIXES = frozenset({".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".sql"})
_BEHAVIOURAL_KINDS = frozenset({"computed", "transition", "action"})
_TEST_SOURCE_NAME = re.compile(r"(?:\.(?:test|spec)\.[^.]+$|^test_.+\.py$|_test\.py$)", re.I)

_EXTRACT_SYSTEM_PROMPT = (
    "You turn a user's application request into the list of atomic, testable "
    "requirements a tester would check in the running application.\n"
    "Rules:\n"
    "- One behaviour per item, in the user's own words where possible.\n"
    "- Include: validations and the shape a value must have; uniqueness; "
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
    "is the file's relative path, a colon, and ONE line copied verbatim from "
    "that file, for example: web_app/backend/routers/booking.py: raise "
    "HTTPException(status_code=400, detail=\"at least one room is required\") "
    "- the line that enforces or performs it: a unique=True, a validator, an "
    "if/raise in a router, the assignment that performs a transition. A "
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
                            "type": "string",
                            "description": (
                                "The file's relative path, a colon, and one "
                                "line copied verbatim from it - the line that "
                                "enforces or performs the requirement, e.g. "
                                "web_app/backend/routers/room.py: raise "
                                "HTTPException(status_code=409, detail=...)"
                            ),
                        },
                        "note": {"type": "string"},
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


def _call_with_tool(llm_client, system: str, prompt: str, tool: dict) -> dict | None:
    """One forced tool call (planning model first, primary on error); the
    tool's input dict, or None when the call failed or returned nothing."""
    structured = _chat_supports_kwargs(llm_client, "force_tool", "model_override")
    planning_model = getattr(llm_client, "planning_model", None) if structured else None
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
        try:
            response = _chat(None)
        except Exception:
            logger.warning("Requirements ledger: %s call failed", tool["name"], exc_info=True)
            return None
    return _tool_input(response, tool["name"])


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
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
        return None
    requirements: list[dict] = []
    for item in extracted:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if not text:
            continue
        kind = str(item.get("kind", "")) if isinstance(item, dict) else ""
        requirements.append({"id": len(requirements) + 1, "text": text, "kind": kind})
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


def _digest_priority(rel: str) -> tuple[int, str]:
    if rel.endswith(("pydantic_classes.py", "sql_alchemy.py")):
        return (0, rel)
    if "/routers/" in rel or rel.startswith("routers/"):
        return (1, rel)
    if rel.endswith(".py"):
        return (2, rel)
    if "/pages/" in f"/{rel}":
        return (3, rel)
    return (4, rel)


def _source_files(output_dir: str) -> dict[str, str]:
    """Only application source, never run logs/recipes or an escaped symlink."""
    base = Path(output_dir).resolve()
    files: dict[str, str] = {}
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _DIGEST_SKIP_DIRS
                   and not (Path(root) == base and d in _ARTIFACT_ROOT_DIRS)]
        for name in names:
            source = Path(root, name)
            if (name.startswith(".") or source.suffix.lower() not in _SOURCE_SUFFIXES
                    or _TEST_SOURCE_NAME.search(name)):
                continue
            try:
                if not source.resolve().is_relative_to(base) or not source.is_file():
                    continue
            except OSError:
                continue
            files[source.relative_to(base).as_posix()] = str(source)
    return files


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

    Citation repair can prioritize a few cited files and read them beyond the
    ordinary per-file cap, while keeping the same total prompt budget.
    """
    files = _source_files(output_dir)
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
    footer_budget = min(6000, max(256, _DIGEST_MAX_TOTAL_CHARS // 10), _DIGEST_MAX_TOTAL_CHARS)
    source_budget = _DIGEST_MAX_TOTAL_CHARS - footer_budget
    entries = []
    unreadable = []
    for rel, full in files.items():
        if Path(full).name in _DIGEST_SKIP_NAMES:
            continue
        try:
            with open(full, "r", encoding="utf-8-sig", errors="ignore") as fh:
                content = _compact(fh.read())
        except OSError:
            unreadable.append(rel)
            continue
        header = f"### {rel}\n"
        cap = max(0, source_budget - len(header) - 200) if rel in focused else _DIGEST_MAX_FILE_CHARS
        truncated = len(content) > cap
        chunk = header + content[:cap]
        if truncated:
            chunk += (f"\n[DIGEST TRUNCATED: {rel}; showing {cap} of {len(content)} code characters. "
                      "Do not infer absence from omitted code.]\n")
        entries.append(((0 if rel in focused else 1, *_digest_priority(rel)), rel, chunk, truncated))
    entries.sort(key=lambda e: e[0])
    parts: list[str] = []
    total = 0
    omitted = list(unreadable)
    truncated_paths = []
    for _, rel, chunk, truncated in entries:
        if total + len(chunk) + 2 > source_budget:
            omitted.append(rel)
            continue
        parts.append(chunk)
        total += len(chunk) + 2
        if truncated:
            truncated_paths.append(rel)
    footer = (
        "\n\n## Digest coverage (not application source)\n"
        f"Files shown: {len(parts)}; truncated excerpts: {len(truncated_paths)}; "
        f"omitted/unreadable files: {len(omitted)}.\n"
        "Omitted or truncated code is unknown, not evidence of absent behavior.\n"
        + ("Truncated files: " + ", ".join(truncated_paths) + "\n" if truncated_paths else "")
        + ("Omitted files: " + ", ".join(omitted) + "\n" if omitted else "")
    )
    if len(footer) > footer_budget:
        footer = footer[:max(0, footer_budget - 42)] + "\n[Coverage path listing also truncated.]\n"
    return "\n\n".join(parts) + footer[:footer_budget]


def judge_coverage(requirements: list[dict], digest: str, llm_client, *,
                   original_spec: str | None = None,
                   previous_verdicts: list[dict] | None = None) -> list[dict] | None:
    """One verdict per requirement, or None when the call failed."""
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
                 "rejected_evidence": str(item.get("evidence") or "")[:1500],
                 "verification_feedback": str(item.get("note") or "")[:1500]}
                for item in previous_verdicts or []
                if isinstance(item, dict) and item.get("id") in requested_ids][:_MAX_REQUIREMENTS]
    repair_context = ("## Previous citation verification (diagnostic data, not application code)\n\n"
                      + json.dumps(feedback, ensure_ascii=True)[:16000] + "\n\n") if feedback else ""
    payload = _call_with_tool(
        llm_client, _JUDGE_SYSTEM_PROMPT,
        authority + repair_context + f"## Requirements\n\n{listing}\n\n## Generated code\n\n{digest}",
        _SUBMIT_VERDICTS_TOOL,
    ) if application_requirements else {"verdicts": []}
    if not payload:
        return None
    by_id = {r["id"]: r for r in requirements}
    verdicts: dict[int, dict] = {}
    for item in payload.get("verdicts") or []:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if rid not in by_id:
            continue
        status = str(item.get("status", "")).lower()
        verdicts[rid] = {
            "id": rid,
            "text": by_id[rid]["text"],
            "kind": by_id[rid].get("kind", ""),
            "status": status if status in _STATUSES else "unverified",
            "evidence": str(item.get("evidence") or "").strip(),
            "note": str(item.get("note") or "").strip(),
            "inspection_paths": [path[:500] for path in item.get("inspection_paths", [])
                                 if isinstance(path, str)][:5]
            if isinstance(item.get("inspection_paths"), list) else [],
        }
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


def _comment_only_lines(source: str, suffix: str) -> set[int]:
    """Recognise comments in their source language, not by a shared prefix.

    Python **kwargs / *args, JS generators and --decrements are executable.
    Mask C-style/SQL comments while skipping quoted strings, so interior lines
    of block comments cannot masquerade as code citations either.
    """
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


def verify_evidence(verdicts: list[dict], output_dir: str) -> list[dict]:
    """Re-check every 'implemented' citation against the output tree.

    The cited application source must exist (an unambiguous path suffix is
    accepted) and contain the quoted executable line. Never search a different
    file or the run's own recipe to rescue a bad citation. A declaration alone
    cannot prove enforcement or behaviour. Failed evidence is ``unverified``:
    it does not prove that no implementation exists elsewhere.
    """
    files = _source_files(output_dir)
    sqlite = workspace_uses_sqlite(output_dir)
    parsed: dict[str, tuple[list[str], ast.AST | None, list[dict], set[int]]] = {}
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
        path, _, quoted = (item.get("evidence") or "").partition(":")
        path = path.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        quoted = _evidence_quote(quoted)
        candidates = [] if not path or path.startswith("/") or ".." in path.split("/") else (
            [files[path]] if path in files else
            [full for rel, full in files.items() if rel.endswith("/" + path)]
        )
        reason = "the cited source path is missing, excluded or ambiguous"
        if not quoted:
            reason = "the citation must identify an implemented statement, not only a decorator"
        if quoted and len(candidates) == 1:
            candidate = candidates[0]
            try:
                if candidate not in parsed:
                    with open(candidate, "r", encoding="utf-8-sig", errors="ignore") as fh:
                        source = fh.read()
                    tree = None
                    if candidate.endswith(".py"):
                        try:
                            tree = ast.parse(source)
                        except SyntaxError:
                            pass
                    diagnostics = python_structural_diagnostics(tree, sqlite=sqlite) if tree else []
                    parsed[candidate] = (source.splitlines(), tree, diagnostics,
                                         _comment_only_lines(source, Path(candidate).suffix.lower()))
            except OSError:
                parsed[candidate] = ([], None, [], set())
            lines, tree, diagnostics, comments = parsed[candidate]
            reason = "the quoted executable line was not found in the cited source"
            for lineno, line in enumerate(lines, 1):
                if quoted != _normalise_quote(line):
                    continue
                if lineno in comments:
                    reason = "the cited line is a comment, not an implementation"
                    continue
                statement = None
                if candidate.endswith(".py"):
                    statement, reason = _python_citation(
                        tree, lineno, item.get("kind", ""), item.get("text", ""),
                    )
                    if reason:
                        continue
                    invalid = next((d for d in diagnostics if (
                        d["line"] <= lineno <= d.get("end_line", d["line"])
                        or statement is not None and _statement_start(statement) <= d["line"]
                        <= d.get("end_line", d["line"]) <= statement.end_lineno
                    )), None)
                    if invalid:
                        reason = "the cited implementation is invalid: " + invalid["message"]
                        continue
                if (_demands_enforcement(item.get("kind", ""), item.get("text", ""))
                        and not isinstance(statement, (ast.If, ast.Raise, ast.Assert))
                        and not line.lstrip().startswith(("if ", "if(", "if\t", "else if "))
                        and not any(tok in quoted.lower() for tok in _ENFORCEMENT_TOKENS)):
                    reason = "the cited line declares data; it does not demonstrate enforcement"
                    continue
                reason = ""
                break
        if reason:
            item["status"] = "unverified"
            item["note"] = " ".join(f"{item.get('note', '')} [{reason}]".split())
        checked.append(item)
    return checked


def ledger_issues(verdicts: list[dict]) -> list[str]:
    """Unresolved requirements block completion without claiming they are absent."""
    issues: list[str] = []
    for v in verdicts:
        label = f"R{v['id']} — {v['text']}"
        note = f" ({v['note']})" if v.get("note") else ""
        status = v.get("status")
        if v.get("kind") == "verification":
            issues.append(f"requirement unverified: {label}{note}")
            continue
        if status == "missing":
            issues.append(f"requirement: {label} is not implemented{note}")
        elif status == "partial":
            issues.append(f"requirement partial: {label}{note}; inspect and complete the missing behaviour")
        elif status == "unverified":
            issues.append(
                f"requirement unverified: {label} — the cited evidence "
                f"'{v.get('evidence', '')}' does not verify it{note}; inspect the actual source "
                "and correct the citation, implementing behaviour only if it is absent"
            )
    return issues
