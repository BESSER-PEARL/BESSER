"""Validation findings and severity policy shared by the agent and probes."""

import re as _re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ValidationIssue:
    """A single Phase 3 validator finding, tagged by severity.

    severity:
      - ``blocker``: an execution, implementation or acceptance defect
        requiring repair (including missing behavior or invalid evidence),
        or explicitly authorized dependency setup needed for verification.
      - ``warning``: advisory findings or checks whose outcome is unknown.
        Explicit ``validation unverified:`` warnings prevent verified
        completion but do not justify spending source-repair turns.
      - ``style``: cosmetic / preference (unused imports, line length,
        formatting). Never blocks a release.

    The auto-fix loop (when enabled) only consumes ``blocker`` issues.
    Other findings are reported in the recipe and logs without automatic repair.
    """

    severity: Literal["blocker", "warning", "style"]
    message: str

    def __str__(self) -> str:  # for legacy log formatting
        return self.message


def _hard_blockers(issues: list) -> list:
    """Blockers from deterministic checks - everything but the requirements
    ledger's ``requirement:`` verdicts, which an LLM judge produces."""
    return [i for i in issues if not i.message.startswith((
        "requirement:", "requirement unverified:",
        "requirement partial:", "task unverified:",
    ))]


def _check_did_not_run(tool: str, reason: str) -> str:
    """A validation note saying a check was SKIPPED, not that it passed.

    A collector returning ``[]`` on a timeout is indistinguishable from a clean
    result, so the run reports "0 blockers" having verified nothing. The wording
    carries no rule code on purpose, so ``_classify_issue`` keeps it a warning:
    we can't prove the code is broken, only that we did not look.
    """
    return (
        f"validation: {tool} did not run ({reason}) - its checks were SKIPPED, "
        f"so this result does not cover them"
    )


def required_check_unverified(check: str, reason: str) -> str:
    """A required check is unknown, not a source defect for the repair agent."""
    return (
        f"validation unverified: {check} did not run completely ({reason}) - "
        "required verification was SKIPPED; generated output is not verified complete"
    )


def required_dependency_setup(check: str, project: str) -> str:
    """An authorized, recoverable verification prerequisite, not a code defect."""
    return (
        f"verification setup: {check} requires installed project dependencies. "
        f"Toolchain and shell tools are enabled and npm is available. Use "
        f"install_dependencies with working_dir={project!r} to install the declared "
        "dependencies, then call validate_app to run the required checks. "
        "Do not edit application code solely to satisfy this verification prerequisite."
    )


def is_completion_issue(issue: ValidationIssue) -> bool:
    """Missing required verification blocks completion, but not source repair."""
    return (getattr(issue, "severity", None) == "blocker"
            or str(getattr(issue, "message", "")).startswith("validation unverified:"))

_RUFF_STYLE_CODES = frozenset({
    "F401", "F841",               # genuinely cosmetic: unused import / variable
    # F811 (redefinition) is deliberately NOT here; see the blocker branch below.
    "E501",                       # line too long
    "W291", "W292", "W293", "W391",  # whitespace
    "E302", "E303", "E305", "E261", "E262", "E266",  # blank lines / comments
    "I001",                       # import order
})
# F811 joins the undefined-name codes: a redefinition means the later name
# silently wins — the ORM `User` shadowed by the Pydantic `User` and then
# queried through the wrong one. All 4 hits across a 10-app live batch were
# real defects (2026-09-11).
_RUFF_BLOCKER_CODES = frozenset({"F811", "F821", "F822", "F823"})
_RUFF_LINE_RE = _re.compile(r"\b([EWFCNI]\d{2,4})\b")


def _classify_issue(message: str) -> ValidationIssue:
    """Map a raw issue string to a ``ValidationIssue`` with severity.

    Heuristics keyed on the prefix our validators produce so the
    classification is stable as new validators are added.
    """
    text = message.strip()
    lower = text.lower()

    if lower.startswith("validation unverified:"):
        return ValidationIssue("warning", text)
    if lower.startswith("verification setup:"):
        return ValidationIssue("blocker", text)

    # Hard blockers — these prevent the generated app from running.
    if lower.startswith("syntax error in"):
        return ValidationIssue("blocker", text)
    if lower.startswith("dependency conflict in"):
        return ValidationIssue("blocker", text)
    if "but it doesn't exist" in lower:
        # e.g. "Dockerfile references requirements.txt but it doesn't exist"
        return ValidationIssue("blocker", text)
    # Frontend contract: correctness defects that leave the LLM-authored
    # UI visibly broken (blank on load, a form that can't submit). Scoped
    # to correctness, NOT scope — we never demand a feature the model
    # didn't build; we only require that what it DID build actually works.
    if lower.startswith("frontend contract:"):
        return ValidationIssue("blocker", text)
    # Data contract: the generated code disagrees with the domain model's
    # declared id types / server-owned fields, or fakes success for an
    # unimplemented method. High-precision patterns only (see
    # contract_checks.py); the fuzzy ones carry an "(advisory)" prefix
    # and fall through to the default warning below.
    if lower.startswith("data contract:"):
        return ValidationIssue("blocker", text)

    # An import naming a module the app does not ship is fatal at startup, and
    # ruff is structurally blind to it (a star import excuses every name rather
    # than flagging it). See _unresolvable_local_imports.
    if lower.startswith("missing module:"):
        return ValidationIssue("blocker", text)

    # The ORM module failed to import or to configure its mappers in the import
    # smoke check: every request that touches the database is a 500.
    if lower.startswith((
        "mapper config:", "application startup:", "python contract:",
        "runtime unverified:", "api scenario:",
    )):
        return ValidationIssue("blocker", text)

    # Missing behavior and incomplete requirement verification both prevent
    # verified completion; unknown evidence is not proof of absent behavior.
    if lower.startswith((
        "requirement:", "requirement partial:", "requirement unverified:",
        "action contract:", "task unverified:",
    )):
        return ValidationIssue("blocker", text)

    # The F821 ruff cannot emit under a star import; a NameError on the first
    # request that reaches the line. See _star_import_undefined_names.
    if lower.startswith("undefined name:"):
        return ValidationIssue("blocker", text)

    # The domain model describes an aggregate no client can create. Legal UML
    # (DomainModel.validate only warns), fatal here: generating a CRUD API is
    # exactly the intent this breaks. Live 2026-09-18 — Booking required a
    # ReservedRoom id and ReservedRoom required a Booking id, so the shipped
    # app served 69 paths and could not create either.
    if lower.startswith("model contract:"):
        return ValidationIssue("blocker", text)

    # The runtime probe observed a server/persistence failure. Guessed input
    # rejected with 4xx is inconclusive, not proof that all valid inputs fail;
    # those reports use the separate create-unverified path.
    #
    # ``action call:`` is the action-endpoint twin of ``create contract:`` -
    # constructibility.py observed a 500 from a handler it invoked. It was in
    # no prefix list here, so it fell through to the default warning and the
    # blocker-only fix loop never consumed it.
    if lower.startswith(("create contract:", "action call:")):
        return ValidationIssue("blocker", text)

    # Ruff: classify by rule code. F821 (undefined name) is a BLOCKER:
    # it means the backend imports crash on `uvicorn` even though
    # ast.parse was clean — the classic "ships green, boots dead" bug.
    if text.startswith("ruff:"):
        # The "+N more issues truncated" note is a count, not a defect; it
        # must never inflate the blocker/warning totals the fix loop gates on.
        if text.startswith("ruff: (+"):
            return ValidationIssue("style", text)
        match = _RUFF_LINE_RE.search(text)
        if match and match.group(1) in _RUFF_STYLE_CODES:
            return ValidationIssue("style", text)
        if match and match.group(1) in _RUFF_BLOCKER_CODES:
            return ValidationIssue("blocker", text)
        return ValidationIssue("warning", text)

    # Per-project toolchain failures (tsc / cargo / kotlinc) are
    # blockers: they mean the artifact does not compile on its own
    # toolchain, which is the per-project compile-pass criterion the
    # bench checks. The Phase 3 fix loop must drive these to zero.
    # ``tsc info`` / ``cargo info`` etc. (informational lines our
    # collectors emit when the binary is missing or the project has
    # no errors) are NOT prefixed this way — only real error lines
    # land here.
    if (text.startswith("tsc [")
            or text.startswith("frontend build [")
            or text.startswith("cargo [")
            or text.startswith("kotlinc [")):
        return ValidationIssue("blocker", text)

    # Legacy ``tsc `` (no bracket) prefix — kept as a soft warning so
    # any caller that constructs strings outside the collector path
    # doesn't trip the fix loop unexpectedly.
    if text.startswith("tsc "):
        return ValidationIssue("warning", text)

    # Unknown shape → conservative default: warning.
    return ValidationIssue("warning", text)
