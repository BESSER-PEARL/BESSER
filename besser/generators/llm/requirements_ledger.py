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

import json
import logging
import os

from besser.generators.llm.gap_analyzer import _chat_supports_kwargs, _is_real_provider

logger = logging.getLogger(__name__)

# The modeling agent appends the user's own text under this heading (see
# smart_generation_handler.py); everything before it is an LLM summary.
VERBATIM_MARKER = "## The user's original request, verbatim"

_MAX_REQUIREMENTS = 40
_STATUSES = ("implemented", "partial", "missing")

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
    "unique", "primary_key", "raise", "if ", "validat", "assert",
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


def _normalise_quote(text: str) -> str:
    """The judge's quote, made comparable to file text: literal escape
    sequences become spaces, single quotes become double quotes (the live
    judge rewrote ``Mapped_["Booking"]`` as ``Mapped_['Booking']``), and
    only the first quoted line counts - later lines are often abridged."""
    text = text.replace("\\n", "\n").replace("\\t", " ")
    first = text.strip().split("\n", 1)[0]
    return _collapse(first.replace("'", '"'))
_DIGEST_SKIP_NAMES = frozenset({"bal_stdlib.py", "database.py", "__init__.py"})
_DIGEST_SKIP_DIRS = frozenset({"node_modules", "dist", "build", ".besser_snapshot", "__pycache__"})

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
    "- implemented: the code enforces or performs the behaviour. The evidence "
    "is the file's relative path, a colon, and ONE line copied verbatim from "
    "that file, for example: web_app/backend/routers/booking.py: raise "
    "HTTPException(status_code=400, detail=\"at least one room is required\") "
    "- the line that enforces or performs it: a unique=True, a validator, an "
    "if/raise in a router, the assignment that performs a transition. A "
    "column, relationship or field EXISTING is not evidence that a rule "
    "about it is enforced; a comment, docstring or TODO is not an "
    "implementation. If you cannot quote an enforcing line, the verdict is "
    "missing.\n"
    "- partial: some of it is there; quote what is there and say in the "
    "note what is missing.\n"
    "- missing: nothing in the code does it; say in the note what you looked "
    "for.\n"
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
    requirements: list[dict] = []
    for item in payload.get("requirements") or []:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if not text:
            continue
        kind = str(item.get("kind", "")) if isinstance(item, dict) else ""
        requirements.append({"id": len(requirements) + 1, "text": text, "kind": kind})
        if len(requirements) >= _MAX_REQUIREMENTS:
            break
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
    return (3, rel)


def _compact(content: str) -> str:
    """Drop blank and comment-only lines; the judge reads code, not prose."""
    kept = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept)


def build_app_digest(output_dir: str) -> str:
    """The backend and the frontend pages, routers first, bounded in size."""
    entries: list[tuple[tuple[int, str], str]] = []
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in _DIGEST_SKIP_DIRS]
        for fname in files:
            rel = os.path.relpath(os.path.join(root, fname), output_dir).replace("\\", "/")
            if fname in _DIGEST_SKIP_NAMES or rel.startswith(".besser_"):
                continue
            is_backend = fname.endswith(".py")
            is_page = fname.endswith((".tsx", ".jsx")) and "/pages/" in f"/{rel}"
            if not (is_backend or is_page):
                continue
            try:
                with open(os.path.join(root, fname), "r", encoding="utf-8", errors="ignore") as fh:
                    content = _compact(fh.read())
            except OSError:
                continue
            entries.append((_digest_priority(rel), f"### {rel}\n{content[:_DIGEST_MAX_FILE_CHARS]}"))
    entries.sort(key=lambda e: e[0])
    parts: list[str] = []
    total = 0
    for _, chunk in entries:
        if total + len(chunk) > _DIGEST_MAX_TOTAL_CHARS:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def judge_coverage(requirements: list[dict], digest: str, llm_client) -> list[dict] | None:
    """One verdict per requirement, or None when the call failed."""
    if not requirements or not _is_real_provider(llm_client):
        return None
    listing = render_requirements(requirements)
    payload = _call_with_tool(
        llm_client, _JUDGE_SYSTEM_PROMPT,
        f"## Requirements\n\n{listing}\n\n## Generated code\n\n{digest}",
        _SUBMIT_VERDICTS_TOOL,
    )
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
            "status": status if status in _STATUSES else "missing",
            "evidence": str(item.get("evidence") or "").strip(),
            "note": str(item.get("note") or "").strip(),
        }
    # A requirement the judge skipped is not implemented until shown otherwise.
    for rid, req in by_id.items():
        verdicts.setdefault(rid, {
            "id": rid, "text": req["text"], "kind": req.get("kind", ""),
            "status": "missing", "evidence": "", "note": "no verdict returned",
        })
    return [verdicts[rid] for rid in sorted(verdicts)]


def _collapse(text: str) -> str:
    """Whitespace-free: the code tokens must match verbatim, their spacing
    need not (``firstName: str`` and ``firstName:str`` are the same line)."""
    return "".join(text.split())


def verify_evidence(verdicts: list[dict], output_dir: str) -> list[dict]:
    """Re-check every 'implemented' citation against the output tree.

    The cited file must exist (a path suffix is accepted, the judge sees
    relative paths) and the quoted line must occur in it, whitespace aside;
    otherwise the verdict is ``unverified``. For uniqueness, validation and
    rule requirements the quoted line must also be an enforcing construct:
    when the judge can only quote a column or a relationship, that is the
    signature of a rule nobody enforces, and the verdict becomes ``missing``.
    """
    files: dict[str, str] = {}
    for root, dirs, names in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in _DIGEST_SKIP_DIRS]
        for name in names:
            full = os.path.join(root, name)
            files[os.path.relpath(full, output_dir).replace("\\", "/")] = full
    checked: list[dict] = []
    for verdict in verdicts:
        item = dict(verdict)
        if item.get("status") != "implemented":
            checked.append(item)
            continue
        path, _, quoted = (item.get("evidence") or "").partition(":")
        path = path.strip().replace("\\", "/").lstrip("./")
        quoted = _normalise_quote(quoted)
        match = files.get(path) or next(
            (full for rel, full in files.items() if path and rel.endswith("/" + path)), None,
        )
        # The quote is the evidence; the path only says where to look. When it
        # is missing or wrong (the live judge wrote the literal word "path"),
        # a verbatim line that exists somewhere in the output still counts.
        candidates = [match] if match is not None else list(files.values())
        found = False
        for candidate in candidates if quoted else []:
            try:
                with open(candidate, "r", encoding="utf-8", errors="ignore") as fh:
                    found = quoted in _collapse(fh.read().replace("'", '"'))
            except OSError:
                found = False
            if found:
                break
        if not found:
            item["status"] = "unverified"
        elif (_demands_enforcement(item.get("kind", ""), item.get("text", ""))
              and not any(tok in quoted.lower() for tok in _ENFORCEMENT_TOKENS)):
            item["status"] = "missing"
            item["note"] = " ".join(
                f"{item.get('note', '')} [the cited line declares data; "
                "nothing in it enforces the rule]".split()
            )
        checked.append(item)
    return checked


def ledger_issues(verdicts: list[dict]) -> list[str]:
    """Phase 3 issue strings: ``requirement:`` (blocker) for missing ones,
    warnings for partial and unverified."""
    issues: list[str] = []
    for v in verdicts:
        label = f"R{v['id']} — {v['text']}"
        note = f" ({v['note']})" if v.get("note") else ""
        status = v.get("status")
        if status == "missing":
            issues.append(f"requirement: {label} is not implemented{note}")
        elif status == "partial":
            issues.append(f"requirement (partial): {label}{note}")
        elif status == "unverified":
            issues.append(
                f"requirement (unverified): {label} — the cited evidence "
                f"'{v.get('evidence', '')}' was not found in the output"
            )
    return issues
