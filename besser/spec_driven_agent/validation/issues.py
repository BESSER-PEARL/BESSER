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
      - ``style``: a count, not a defect: the ``(+N more ... truncated)`` notes of
        ruff and the compilers. Never blocks a release. The lint selects no cosmetic rule.

    The auto-fix loop (when enabled) only consumes ``blocker`` issues.
    Other findings are reported in the recipe and logs without automatic repair.
    """

    severity: Literal["blocker", "warning", "style"]
    message: str

    def __str__(self) -> str:  # for legacy log formatting
        return self.message


def _hard_blockers(issues: list) -> list:
    """Blockers from deterministic checks that OBSERVED a failure.

    Excluded: the requirements ledger's ``requirement:`` verdicts, which an
    LLM judge produces, and the ``unverified`` family, whose own text says a
    guessed request being refused "does not prove this endpoint is broken".
    They stay blockers - an unverified app is not a verified one - but they
    may not rank one tree above another or force a rollback, because we do
    not know that they describe a defect (e.g. an app correctly refusing to
    create an ABSTRACT entity).
    """
    return [i for i in issues if not i.message.startswith((
        "requirement:", "requirement unverified:",
        "requirement partial:", "task unverified:", "task incomplete:",
        "runtime unverified:", "create unverified:",
    ))]


# Findings that mean "we could not check", never "the app is broken". Each is
# an UNKNOWN: the run did not establish the behaviour either way. They belong in
# the ledger's could-not-verify bucket, which the UI already renders separately.
_UNKNOWN_PREFIXES = (
    "requirement unverified:", "requirement partial:", "task unverified:",
    "runtime unverified:", "create unverified:", "action unverified:",
    "validation unverified:", "verification setup:", "api scenario:",
    "requirement:",
)


def unresolved_defects(issues: list) -> list:
    """Completion issues that describe a DEFECT, not an unknown.

    The whole unverified family classifies as ``blocker`` severity, but an
    app whose only open items are could-not-verify checks is not broken.
    "Incomplete" is reserved for output that does not work: a tree that will
    not parse, an app that cannot boot or create a record, a requirement the
    code demonstrably does not implement. An unknown keeps its place in the
    ledger and in ``blockerCount``; it does not set the headline.
    """
    return [
        i for i in issues
        if not str(getattr(i, "message", i)).startswith(_UNKNOWN_PREFIXES)
    ]


# Requirement blockers by status rank: a requirement whose rank rises is progress,
# one whose rank falls is a new blocker (``requirement_status``). Equal ranks are
# neither: a stub that turns "missing" into "unsupported" settles nothing.
REQUIREMENT_STATUSES = {
    "requirement:": 0,
    "requirement unsupported:": 0,
    "requirement partial:": 1,
}
_REQUIREMENT_ID_RE = _re.compile(r"\s*R(\d+)\b")


def requirement_status(message: str) -> tuple[int, int] | None:
    """``(requirement id, status rank)`` of a requirement blocker, else None."""
    for prefix, rank in REQUIREMENT_STATUSES.items():
        if message.startswith(prefix):
            match = _REQUIREMENT_ID_RE.match(message, len(prefix))
            return (int(match.group(1)), rank) if match else None
    return None


# Phase 3's note when the requirement judge was not run on a tree.
JUDGMENT_DEFERRED = "validation: business-requirement judgment deferred"
# Findings that say the requirement judge did not run on a tree: deferred, failed
# (it returned no verdicts) or stopped (deadline or budget).
JUDGE_DID_NOT_RUN = (
    JUDGMENT_DEFERRED,
    "validation unverified: the requirements judge",
    "requirement unverified: not checked",
)

# The check family of a finding, and what stops that family running ("dependent"
# findings are excused per entity from the probe's records: see Phase 3's
# ``_newly_reachable``). A finding
# whose family could not run on the tree before an edit is newly visible, not
# newly broken (``could_not_run``). A family in ``toolchain`` / ``build`` is
# per project (``[P]`` in the message), in ``file`` per file; a finding of no
# family is always counted. One row per message prefix (lower case).
FINDING_FAMILIES = (
    ("application startup:", ("runtime",)),
    ("mapper config:", ("runtime",)),
    ("create contract:", ("runtime", "dependent")),
    ("action call:", ("runtime", "dependent")),
    ("api scenario:", ("runtime",)),
    ("requirement", ("judge",)),
    ("task ", ("judge",)),
    ("tsc [", ("toolchain",)),
    ("frontend build [", ("toolchain", "build")),
    ("ruff:", ("file",)),
    ("python contract:", ("file",)),
)
HIDDEN_BY = (
    ("missing module:", "runtime"),
    ("application startup:", "runtime"),
    ("mapper config:", "runtime"),
    *((note, "judge") for note in JUDGE_DID_NOT_RUN),
    ("verification setup: tsc [", "toolchain"),
    ("verification setup: frontend build [", "toolchain"),
    ("validation unverified: tsc [", "toolchain"),
    ("validation unverified: frontend build [", "toolchain"),
    ("tsc [", "build"),
    ("syntax error in ", "file"),
)
_PROJECT_RE = _re.compile(r"\[([^\]]+)\]")
# The first path a finding names with a line: ``x.py:3``, ``x.py line 3``, ``x.tsx(3,5)``.
_FILE_RE = _re.compile(r"([\w./\-]+\.\w+)(?::\d| line \d|\(\d)")


def _family_key(family: str, message: str) -> tuple[str, str] | None:
    """``(family, parameter)``; None when a per-project or per-file family names none."""
    if family in ("toolchain", "build"):
        match = _PROJECT_RE.search(message)
        return (family, match.group(1)) if match else None
    if family == "file":
        match = _FILE_RE.search(message)
        return (family, match.group(1).replace("\\", "/").removeprefix("./")) if match else None
    return family, ""


def finding_families(message: str) -> set[tuple[str, str]]:
    """The check families ``message`` belongs to (``FINDING_FAMILIES``)."""
    lower = message.lower()
    return {key for prefix, families in FINDING_FAMILIES if lower.startswith(prefix)
            for key in (_family_key(family, message) for family in families) if key}


def hidden_families(issues: list) -> set[tuple[str, str]]:
    """The check families that could not run on the tree ``issues`` (warnings included) describe."""
    hidden = set()
    for issue in issues:
        message = str(getattr(issue, "message", issue))
        lower = message.lower()
        for prefix, family in HIDDEN_BY:
            key = _family_key(family, message) if lower.startswith(prefix) else None
            if key:
                hidden.add(key)
    return hidden


def judge_ran(issues: list) -> bool:
    """Whether the requirement judge ran on the tree ``issues`` (warnings included) describe."""
    return not any(str(getattr(i, "message", i)).startswith(JUDGE_DID_NOT_RUN) for i in issues)


def could_not_run(message: str, hidden: set[tuple[str, str]]) -> bool:
    """Whether a check family of ``message`` is in ``hidden``; files match by path suffix."""
    for family, param in finding_families(message):
        if (family, param) in hidden:
            return True
        if family == "file" and any(kind == "file" and (other.endswith("/" + param) or param.endswith("/" + other))
                                    for kind, other in hidden):
            return True
    return False


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


# pip's signature of an index it could not reach. It then reports "No matching
# distribution found" / ResolutionImpossible, which reads as a conflict.
_PIP_NETWORK_MARKERS = (
    "CERTIFICATE_VERIFY_FAILED", "SSLError", "ProxyError", "NewConnectionError",
    "Failed to establish a new connection", "ConnectTimeoutError",
    "ReadTimeoutError", "Temporary failure in name resolution", "getaddrinfo failed",
)


def dependency_check_issue(rel: str, stderr: str) -> str | None:
    """The finding for a failed ``pip install --dry-run``, or ``None``.

    A network/TLS failure is an unknown, not a conflict: behind a
    TLS-inspecting proxy every run got a false "Dependency conflict" blocker.
    The quoted lines skip pip's ``[notice]`` upgrade nag, which used to fill
    the three-line excerpt and hide the actual error.
    """
    lines = [line.strip() for line in (stderr or "").splitlines()
             if line.strip() and not line.lstrip().startswith("[notice]")]
    network = next((line for line in lines
                    if any(marker in line for marker in _PIP_NETWORK_MARKERS)), None)
    if network:
        return (f"validation: could not verify dependencies in {rel}: network/TLS error - "
                f"pip could not reach the package index ({network[:300]}). This is not a "
                "dependency conflict; the check was skipped.")
    errors = [line for line in lines if "WARNING" not in line]
    if not errors:
        return None
    return f"Dependency conflict in {rel}:\n" + "\n".join(errors[-3:])


def required_check_unverified(check: str, reason: str) -> str:
    """A required check is unknown, not a source defect for the repair agent."""
    return (
        f"validation unverified: {check} did not run completely ({reason}) - "
        "required verification was SKIPPED; generated output is not verified complete"
    )


# A runtime check (import smoke check, boot-and-create probe, test_api replay)
# that did not run: no sandbox, a timeout, a harness dependency missing. Its
# ``validation unverified:`` prefix keeps it out of the blocker-only fix loop;
# the runtime gate keys on this prefix to keep its verdict "unverified".
RUNTIME_CHECK_UNVERIFIED = "validation unverified: runtime check"
_DID_NOT_RUN_NOTE_RE = _re.compile(
    r"^validation: (?P<check>.+?) did not run \((?P<reason>.*)\) - its checks were SKIPPED", _re.S)


def runtime_check_unverified(check: str, reason: str) -> str:
    """A runtime check that did not run: an unknown no source edit can settle."""
    return (
        f"{RUNTIME_CHECK_UNVERIFIED} ({check}) did not run ({reason}) - no source edit "
        "can settle this, so it is not sent to repair; the application is not runtime verified"
    )


def runtime_note_unverified(note: str) -> str:
    """``runtime_check_unverified`` for a ``_check_did_not_run`` note."""
    match = _DID_NOT_RUN_NOTE_RE.match(note)
    if match is None:
        return runtime_check_unverified("runtime probe", note.removeprefix("validation: "))
    return runtime_check_unverified(match["check"], match["reason"])


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

# F811 joins the undefined-name codes: a redefinition means the later name
# silently wins — the ORM `User` shadowed by the Pydantic `User` and then
# queried through the wrong one. In practice these hits are real defects.
_RUFF_BLOCKER_CODES = frozenset({"F811", "F821", "F822", "F823"})
_RUFF_LINE_RE = _re.compile(r"\b([EWFCNI]\d{2,4})\b")
# ``ruff: (+N more issues truncated)``, ``tsc [<project>]: (+N more errors truncated)``,
# and the same for cargo / kotlinc (see toolchain.py).
_TRUNCATION_NOTE_RE = _re.compile(r"^(?:ruff|(?:tsc|cargo|kotlinc) \[[^\]]*\]): \(\+\d+ more ")


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
    # Create-form checks (frontend_forms.py). Blockers: shapes that send every
    # create to the wrong URL or method, or leave the form unreachable.
    # ``form field type:`` stays a warning: a text box for a number still saves
    # when the user types digits, and calibration flagged it on working apps.
    if lower.startswith(("api url:", "form create target:", "form route:",
                         "form json textarea:")):
        return ValidationIssue("blocker", text)
    if lower.startswith("form field type:"):
        return ValidationIssue("warning", text)
    # Reserved for ui_probe.py screen findings; a warning until their precision is measured.
    if lower.startswith("screen:"):
        return ValidationIssue("warning", text)
    # A project agent the web app leaves out because no screen uses it: a notice, never repaired.
    if lower.startswith("agent left out:"):
        return ValidationIssue("warning", text)
    # A frontend call whose JSON body or response use contradicts its route's
    # handler (api_call_shapes.py): the data never reaches the handler, every
    # call is a 422, or the screen reads keys the response does not have.
    if lower.startswith("api call:"):
        return ValidationIssue("blocker", text)
    # Data contract: the generated code disagrees with the domain model's
    # declared id types / server-owned fields, or fakes success for an
    # unimplemented method. High-precision patterns only (see
    # contract_checks.py); the fuzzy ones carry an "(advisory)" prefix
    # and fall through to the default warning below.
    if lower.startswith("data contract:"):
        return ValidationIssue("blocker", text)
    # The backend's ORM / schemas disagree with a model change made since the
    # previous run (model_conformance.py): a removed attribute still defined,
    # an added one missing, a stale type or optionality.
    if lower.startswith("model conformance:"):
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

    # A checklist item nothing settled is reported, not repaired; one whose
    # deterministic check fails is missing work.
    if lower.startswith("task unverified:"):
        return ValidationIssue("warning", text)

    # Missing behavior and incomplete requirement verification both prevent
    # verified completion; unknown evidence is not proof of absent behavior.
    if lower.startswith((
        "requirement:", "requirement partial:", "requirement unverified:",
        "action contract:", "task incomplete:",
    )):
        return ValidationIssue("blocker", text)

    # The domain model describes an aggregate no client can create. Legal UML
    # (DomainModel.validate only warns), fatal here: generating a CRUD API is
    # exactly the intent this breaks (e.g. Booking requires a ReservedRoom id
    # and ReservedRoom requires a Booking id, so neither can be created).
    if lower.startswith("model contract:"):
        return ValidationIssue("blocker", text)

    # The runtime probe observed a server/persistence failure. Guessed input
    # rejected with 4xx is inconclusive, not proof that all valid inputs fail;
    # those reports use the separate create-unverified path.
    #
    # ``action call:`` is the action-endpoint twin of ``create contract:`` -
    # constructibility.py observed a 500 from a handler it invoked. It must be
    # a blocker, or the blocker-only fix loop never consumes it.
    if lower.startswith(("create contract:", "action call:")):
        return ValidationIssue("blocker", text)

    # A collector's "+N more ... truncated" note is a count, not a defect; it
    # must never inflate the blocker/warning totals the fix loop gates on.
    if _TRUNCATION_NOTE_RE.match(text):
        return ValidationIssue("style", text)

    # Ruff: classify by rule code. F821 (undefined name) is a BLOCKER:
    # it means the backend imports crash on `uvicorn` even though
    # ast.parse was clean — the classic "ships green, boots dead" bug.
    if text.startswith("ruff:"):
        match = _RUFF_LINE_RE.search(text)
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
