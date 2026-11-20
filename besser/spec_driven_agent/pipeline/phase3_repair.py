"""Phase 3: validate the delivered tree, repair it, and keep only repairs that help.

Each fix attempt targets the editable blockers (``severity == "blocker"``) and
may edit only the files they name, their neighbours and new files
(``repair_scope``). No tool is forced. An attempt that edits nothing ends the
loop; one that edits is kept only if it retires a blocker it faced without
adding an observed failure or blockers (``_attempt_not_kept``), else the tree
from before it is restored. An attempt that had retired or changed a blocker
gets one more attempt from that tree, told what broke; otherwise, or after a
second revert in a row, the loop ends.

Snapshot/rollback lives here too because it is the same decision: the snapshot
is re-taken after every kept attempt, so it always holds the tree the next
attempt starts from.

Mixed into ``LLMOrchestrator``; all run state is reached through ``self``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import Counter

from besser.spec_driven_agent.agent.edit_apply import gutter_line
from besser.spec_driven_agent.agent.history_eviction import (
    without_rejected_edit_drafts,
)
from besser.spec_driven_agent.errors import DeadlineReached, InvalidApiKeyError
from besser.spec_driven_agent.planning import requirements_ledger as _requirements_ledger
from besser.spec_driven_agent.planning.action_inventory import (
    collect_action_endpoints,
    format_action_inventory,
)
from besser.spec_driven_agent.agent.prompt_builder import DROPPED_RULES_NOTE, build_inventory
from besser.spec_driven_agent.agent.tools import with_scope_widening
from besser.spec_driven_agent.planning.model_rules import DROPPED
from besser.spec_driven_agent.planning.mutation_inventory import build_mutation_manifest, route_serves
from besser.spec_driven_agent.model_serializer import serialize_domain_model, to_prompt_json
from besser.spec_driven_agent.pipeline.constants import (
    _MAX_TOOLCHAIN_FIX_ITERATIONS,
    _PHASE3_CREATE_EVIDENCE_REMINDER,
    _PHASE3_FIX_TURNS,
    _ROLLBACK_DISCARD_DIR,
    _ROLLBACK_PRESERVED,
    _RUNTIME_FAILED,
    _RUNTIME_OK,
    _RUNTIME_FATAL_SOURCE_PREFIXES,
    _RUNTIME_OBSERVED_FAILURE_PREFIXES,
    _SNAPSHOT_IGNORED_DIRS,
    _SNAPSHOT_STAGING_DIR,
    _WRITE_TOOLS_ON_RECORD,
    _tool_use_blocks,
)
from besser.spec_driven_agent.pipeline.repair_scope import edit_scope, named_locations
from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.state.tracing import (
    EVENT_EDIT_SCOPE,
    EVENT_PHASE_ENTER,
    EVENT_PHASE_EXIT,
    EVENT_ROLLBACK,
    EVENT_SNAPSHOT,
    EVENT_TURN_END,
    EVENT_TURN_START,
    EVENT_VALIDATION_ISSUE,
)
from besser.spec_driven_agent.validation.api_probe import confirmed_create_paths
from besser.spec_driven_agent.validation.issues import (
    PHASE3_SKIPPED,
    RUNTIME_CHECK_UNVERIFIED,
    ValidationIssue,
    _classify_issue,
    _hard_blockers,
    could_not_run,
    hidden_families,
    is_completion_issue,
    judge_ran,
    required_check_unverified,
    requirement_status,
    unresolved_defects,
)
from besser.spec_driven_agent.validation.toolchain import (
    _build_toolchain_reminder,
    _toolchain_commands_for,
)
from besser.utilities.workspace_fs import copytree_plain, is_plain_entry, walk_plain

logger = logging.getLogger(__name__)


# Blockers one fix prompt lists. One missing import was reported as 105; a
# round repairs a few causes and the next validation lists what is left.
_FIX_PROMPT_MAX_BLOCKERS = 30
# A longer blocker is cut in the fix prompt. Tracebacks ran to 3,300 chars; a
# create contract with its fix site and remedy runs to about 1,000, kept whole.
# The tail survives the cut: a traceback ends with the error itself.
_FIX_PROMPT_BLOCKER_CHARS = 1_200
_FIX_PROMPT_BLOCKER_TAIL_CHARS = 300
# Same shape ``_apply_scenario_evidence`` matches: (backend, path).
_CREATE_UNVERIFIED = re.compile(r"create unverified: (.+?): POST (\S+) -")
# Paths the fix prompt lists: the files the run has edited, and an attempt's edit scope.
_FIX_PROMPT_MAX_EDITED = 40
_FIX_PROMPT_MAX_SCOPE_FILES = 40
# Masked when attempts compare findings: an edit moves the lines and columns of the
# errors it leaves. Other numbers (``/e2/`` and ``/e3/``) tell findings apart.
_LINE_NUMBERS_RE = re.compile(r"line \d+|:\d+(?::\d+)?|\(\d+,\d+\)")
# The route a runtime finding names: ``create contract: backend: POST /order/ - ...``.
_POST_ROUTE_RE = re.compile(r": POST (\S+) - ")
# What a runtime finding tests: its kind, backend and route. Its text is the error seen there.
_RUNTIME_TARGET_RE = re.compile(
    r"(?:runtime unverified: )?(create contract|action call|create unverified|action unverified): (\S+): POST (\S+) - ")
# The probe's generated input, quoted in a runtime finding: not part of the error.
_PROBE_INPUT_RE = re.compile(r"attempted input: .*?(?=\. Th(?:is|ese) )", re.S)
# An object's memory address in an exception text (``<Order at 0x7f...>``): new in every process.
_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
# Characters of the current text of the files the blockers name, pasted into a fix prompt.
_FIX_PROMPT_FILE_CHARS = 24_000


def clip_finding(text: str) -> str:
    """*text* cut to ``_FIX_PROMPT_BLOCKER_CHARS``, keeping its head and its tail."""
    if len(text) <= _FIX_PROMPT_BLOCKER_CHARS:
        return text
    head = _FIX_PROMPT_BLOCKER_CHARS - _FIX_PROMPT_BLOCKER_TAIL_CHARS
    cut = len(text) - _FIX_PROMPT_BLOCKER_CHARS
    return f"{text[:head]} [... {cut} chars cut ...] {text[-_FIX_PROMPT_BLOCKER_TAIL_CHARS:]}"


def _finding_key(message: str):
    """What a finding tests, as attempts compare findings: a runtime finding by its kind,
    backend and route, any other by its message with line numbers masked (an edit moves
    the lines of the errors it leaves)."""
    target = _RUNTIME_TARGET_RE.match(message)
    return target.groups() if target else _LINE_NUMBERS_RE.sub("#", message)


def _finding_text(message: str) -> str:
    """The error a finding reports, with line numbers, memory addresses and the probe's input masked."""
    text = _PROBE_INPUT_RE.sub("attempted input: #", message)
    return _LINE_NUMBERS_RE.sub("#", _ADDRESS_RE.sub("0x#", text))


def _blocker_kind(message: str) -> str:
    """The blocker's prefix, e.g. ``tsc [frontend]`` or ``Syntax error in``."""
    head = message.split(":", 1)[0]
    words = head.split()
    return " ".join(words[:3]) if len(words) > 3 else head


# Blockers about source that does not parse, compile or import. Any other
# blocker - an observed failure, a contract, a requirement, or a kind added
# later - keeps the behaviour sections in the fix prompt.
_SOURCE_DEFECT_PREFIXES = _RUNTIME_FATAL_SOURCE_PREFIXES + (
    "ruff:", "tsc [", "cargo [", "kotlinc [", "frontend build [", "dependency conflict in",
)

# A requirement text at least this long is cited by id wherever a blocker
# quotes it; a shorter one could match unrelated text and saves little.
_CITED_REQUIREMENT_MIN_CHARS = 60


def _cite_ledger_requirements(message: str, requirements: list[dict]) -> str:
    """Name a requirement the ledger already quotes by its id, not its text again.

    A conversion-recovery requirement runs to ~900 chars and was pasted three
    times per prompt: the ledger, its ``requirement:`` blocker and its
    ``task incomplete:`` blocker. The ledger keeps the one full copy.
    """
    for requirement in requirements:
        rid, text = f"R{requirement['id']}", requirement["text"]
        message = message.replace(f"{rid} — {text}", rid)
        if len(text) >= _CITED_REQUIREMENT_MIN_CHARS:
            message = message.replace(text, f"{rid} (text in the requirements ledger)")
    return message


def _located(message: str, found: list[tuple[str, int | None]]) -> str:
    """`` [in: a.py line 3, b.tsx]``: the workspace files a blocker names that it
    does not already spell as workspace paths (a project-relative or partial path)."""
    places = [f"{rel} line {line}" if line else rel for rel, line in found]
    places = [place for place in places if place not in message]
    return " [in: " + ", ".join(places) + "]" if places else ""


def _fix_prompt_blocker_lines(ranked: list[ValidationIssue], requirements: list[dict],
                              named: dict[str, list] | None = None) -> list[str]:
    """The fix prompt's blocker bullets, in priority order, capped in count and length,
    each with the workspace files it names (``named``, by message).

    Only the prompt is capped; Phase 3 records and reports every blocker.
    """
    named = named or {}
    lines = []
    for issue in ranked[:_FIX_PROMPT_MAX_BLOCKERS]:
        lines.append(f"- {clip_finding(_cite_ledger_requirements(issue.message, requirements))}"
                     f"{_located(issue.message, named.get(issue.message, []))}")
    rest = ranked[_FIX_PROMPT_MAX_BLOCKERS:]
    if rest:
        kinds = Counter(_blocker_kind(issue.message) for issue in rest)
        summary = ", ".join(f"{kind} x{count}" for kind, count in kinds.most_common())
        lines.append(f"- (+{len(rest)} more blockers of these kinds: {summary}; "
                     "the next validation lists them again)")
    return lines


def _no_edit_exit_reason(attempt: int, calls: list[dict]) -> str:
    """Name the edit-less round that ended the loop by what it actually did.

    Only a round with no tool call at all is a replay. A round of other tool
    calls stops the loop too, but earlier rounds may well have edited, so the
    label must not read as though the repair never touched the editor; a round
    whose edits were all refused says so.
    """
    if not calls:
        return f"replay (attempt {attempt} called no tool)"
    counts: dict[str, int] = {}
    for entry in calls:
        counts[entry["tool"]] = counts.get(entry["tool"], 0) + 1
    failed = sum(1 for entry in calls if not entry.get("success"))
    plural = "s" if len(calls) != 1 else ""
    tools = ", ".join(f"{tool} x{n}" for tool, n in counts.items())
    if any(entry["tool"] in _WRITE_TOOLS_ON_RECORD for entry in calls):
        return (f"no successful edit (attempt {attempt} made {len(calls)} tool call{plural}, "
                f"{failed} failed: {tools})")
    failed_note = f", {failed} failed" if failed else ""
    return (f"no edit attempted (attempt {attempt} made {len(calls)} tool call{plural}, "
            f"none an edit{failed_note}: {tools})")


def counted(count: int, noun: str) -> str:
    """``1 blocker`` / ``3 blockers``: a count as the run card's progress line says it."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _fixing_blockers(blockers: list[ValidationIssue], round_issues: list[ValidationIssue]) -> str:
    """A fix attempt's step: what it faces, by count, never by internal ids.

    The round's blocker count (``FindingsEvent``) also counts checks that could
    not run; no edit settles those, so the attempt says it leaves them.
    """
    not_run = sum(1 for issue in round_issues if issue.severity != "blocker" and is_completion_issue(issue))
    about_requirements = sum(1 for issue in blockers if issue.message.startswith("requirement "))
    notes = ([f"{about_requirements} about requirements"] if about_requirements else []) + (
        [f"{not_run} {'is a check' if not_run == 1 else 'are checks'} that could not run"] if not_run else [])
    head = (f"fixing {len(blockers)} of {counted(len(blockers) + not_run, 'blocker')}" if not_run
            else f"fixing {counted(len(blockers), 'blocker')}")
    return head + (f" ({'; '.join(notes)})" if notes else "")


def _edited_paths(calls: list[dict]) -> list[str]:
    """Workspace paths the successful, not rolled-back write calls in ``calls`` touched, last write last."""
    paths: dict[str, None] = {}
    for call in calls:
        path = (call.get("input") or {}).get("path") if isinstance(call.get("input"), dict) else None
        if (call.get("tool") in _WRITE_TOOLS_ON_RECORD and call.get("success")
                and not call.get("rolled_back") and isinstance(path, str)):
            paths.pop(path, None)
            paths[path] = None
    return list(paths)


class Phase3RepairMixin:
    """_run_phase3_validation and friends; see the module docstring."""

    #: When the run's first Phase 3 validation started (``time.monotonic``): the run record's phase timing.
    phase3_started_at: float | None = None
    #: What the last kept fix attempt did, for the next attempt's prompt.
    _repair_last_outcome: str | None = None

    def _complete_repair_if_verified(self) -> None:
        if self._phase2_stop_reason == "validation_required":
            self._phase2_exited_cleanly = True
            self._phase2_stop_reason = "completed"

    def _phase3_stop_requested(self, *, check_turn_budget: bool = True) -> str | None:
        """One stop gate for repair turns and the validation calls between them.

        ``check_turn_budget=False`` asks for validation only, which needs time
        before the cap for its checks; a repair turn needs time for an LLM call.
        """
        reason = None
        if self.deadline.checks_out_of_time():
            reason = "runtime budget exhausted"
        elif self.max_cost_usd is not None and self.client.usage.estimated_cost >= self.max_cost_usd:
            reason = "cost budget exhausted"
        elif self._phase2_quota_exhausted:
            reason = "provider quota exhausted in Phase 2"
        elif check_turn_budget and self.total_turns >= self.max_turns:
            # No further edit turn, but final verification of the last accepted
            # tool batch may still run. Do not poison that check as cancelled.
            return "turn budget exhausted"
        elif self.deadline.cancelled:
            reason = "cancellation requested"
        elif self._phase3_interrupted:
            reason = "repair interrupted before verification completed"
            if self._phase3_interrupt_detail:
                reason = f"{reason} ({self._phase3_interrupt_detail})"
        elif check_turn_budget and self.deadline.out_of_time():
            # Like the turn budget: no further repair turn, but the attempt
            # so far is still checked.
            return "runtime budget exhausted"
        if reason:
            self._phase3_interrupted = True
        return reason

    def _repair_obligations_revision(self) -> str:
        """Test corrections and checklist changes are progress without source edits,
        and so are requirement citations ``validate_app`` accepted: a fix attempt
        that moves one is checked again.

        The scenario KEY deliberately does NOT move it: the model may pass its
        own ``scenario_id``, so re-registering identical requests under a new
        name would mint a new entry. Only distinct scenario CONTENT counts, so
        a rename is not a new obligation.
        """
        tasks = self.executor.task_snapshot()
        scenarios = sorted(
            json.dumps(record["scenario"], sort_keys=True)
            for record in self._api_scenarios.values()
        )
        # ``validate_app`` citations change a verdict's status in place.
        verdicts = sorted(json.dumps([verdict.get("id"), verdict.get("status")], default=str)
                          for verdict in self._requirement_verdicts)
        payload = {"tasks": tasks, "scenarios": sorted(set(scenarios)), "verdicts": verdicts}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _run_phase3_validation(self) -> None:
        """Validate, repair, then apply the runtime exit gate.

        The gate runs on EVERY exit from the repair cycle - clean, stalled,
        budget-exhausted or crashed - so a dead app cannot leave through any
        of them quietly. See ``_apply_runtime_gate``.
        """
        if self.phase3_started_at is None:
            self.phase3_started_at = time.monotonic()
        # Announced before the first check, which can take minutes (npm, tsc), so
        # that time is counted as validation, not as the phase before it.
        self._emit_progress(self.total_turns, "__validating__", "")
        try:
            self._run_phase3_repair_cycle()
        finally:
            self._apply_runtime_gate()
        self._report_validation_step("Packaging the result")

    def _report_validation_step(self, message: str) -> None:
        """Name what Phase 3 is doing now on the run card: one check or LLM call can take minutes."""
        self._emit_progress(self.total_turns, "__validation_step__", message)

    def _report_round_step(self, step: str) -> None:
        """A step of the current validation round, e.g. "First check: building the frontend"."""
        self._report_validation_step(f"{self._validation_round}: {step}")

    def _begin_validation_round(self, label: str) -> None:
        self._validation_round = label
        self._report_round_step("checking the code for errors")

    def _report_findings(self, attempt: int, issues: list[ValidationIssue], restored: bool = False) -> None:
        """Hand one validation round to ``on_findings`` (the run card's findings-and-repairs view).

        Best-effort: a failing listener never stops the repair loop.
        """
        if not self.on_findings:
            return
        try:
            self.on_findings(attempt, list(issues), restored=restored)
        except Exception:
            logger.warning("Phase 3: reporting findings of attempt %d failed", attempt, exc_info=True)

    def _run_phase3_repair_cycle(self) -> None:
        """
        Lightweight validation of generated output. If issues found,
        give the LLM a few turns to fix them.

        Checks (no network, no Docker, instant unless the toolchain
        runs — tsc / cargo / kotlinc each have their own timeout):
        - Python syntax on all .py files
        - Dockerfiles reference files that exist
        - package.json exists if Dockerfile uses npm
        - npm ci -> npm install (common LLM mistake)
        - ``ruff`` lint (if installed)
        - the app booted in a subprocess: mappers configure, and every
          entity can be created through its own create endpoint
        - ``tsc --noEmit`` on every tsconfig (if tsc installed)
        - ``cargo check`` on every Cargo.toml (if cargo installed)
        - ``kotlinc`` on every Kotlin source root (if kotlinc installed)

        Per-project failures feed a repair/recheck loop bounded by the
        remaining turn, cost and runtime budgets. An attempt that writes
        nothing, one unchanged/repeated source state, or two rounds that edit
        the tree without improving its score stop retries; unresolved blockers
        remain explicitly incomplete, never accepted as verified output.
        """
        # The final allowed editing turn still deserves validation. The turn
        # cap prevents another repair interaction, not local checks/final review.
        stop_reason = self._phase3_stop_requested(check_turn_budget=False)
        if stop_reason:
            logger.warning("Skipping Phase 3 -- %s", stop_reason)
            self._phase3_exit_reason = f"skipped: {stop_reason}"
            # Not confirmed: nothing was checked, so nothing was seen to fail.
            self._validation_issues.append(_classify_issue(
                f"{PHASE3_SKIPPED}: {stop_reason}. "
                "The current application has not completed final verification."
            ))
            if self._checkpoint_phase == "phase3":
                self._save_phase3_checkpoint()
            return

        # A resumed repair has run attempts already: its first sweep is not the run's first check.
        self._begin_validation_round(
            "Check after resuming" if self._repair_progress.get("attempts_run") else "First check")
        issues = self._collect_validation_issues()

        if not issues:
            logger.info("Phase 3: Validation passed -- no issues found")
            self._phase3_exit_reason = "validation clean"
            self._validation_issues = self._with_model_contract([])
            self._complete_repair_if_verified()
            if self._checkpoint_phase == "phase3":
                self._save_phase3_checkpoint()
            return

        # Always record everything in the recipe — severity decides what
        # gets fixed automatically.
        blockers_before = [i for i in issues if i.severity == "blocker"]
        warnings_before = [i for i in issues if i.severity == "warning"]
        styles_before = [i for i in issues if i.severity == "style"]
        logger.warning(
            "Phase 3: Found %d issues (%d blocker / %d warning / %d style)",
            len(issues), len(blockers_before), len(warnings_before), len(styles_before),
        )
        for issue in issues:
            logger.warning("  [%s] %s", issue.severity, issue.message)
        self._validation_issues = self._with_model_contract(issues)
        if not blockers_before:
            self._complete_repair_if_verified()
        if self._checkpoint_phase == "phase3":
            self._save_phase3_checkpoint()

        self._emit_progress(
            self.total_turns,
            "validation",
            f"{len(blockers_before)} blockers / {len(issues)} total",
        )
        self._report_findings(0, issues)

        # Auto-fix is opt-in (default off). Industry pattern: report by
        # default, fix on request. Avoid the LLM running ``npm install`` —
        # post-install hooks execute arbitrary code from chosen packages.
        if not self.auto_fix_issues:
            logger.info(
                "Phase 3: auto_fix_issues=False — issues recorded, no LLM fix loop."
            )
            self._phase3_exit_reason = "auto-fix disabled"
            return

        # Auto-fix only consumes BLOCKER issues. Style notes (truncation
        # counts) and soft warnings (tsc type hints) are
        # left as-is; they don't justify burning LLM turns.
        if not blockers_before:
            self._phase3_exit_reason = "no blocker-class issues"
            logger.info(
                "Phase 3: auto_fix_issues=True but no blocker-class issues — "
                "skipping LLM fix loop. %d non-blocker issue(s) recorded.",
                len(warnings_before) + len(styles_before),
            )
            return

        # Validation itself rewrites files (npm ci -> npm install, dependency
        # pins, the vite config). Re-snapshot so the rollback target is the
        # tree ``issues`` describe, not the one before those rewrites.
        if self._create_snapshot():  # a failed one keeps the previous target, and is no trace event
            self._trace.write(EVENT_SNAPSHOT, before_phase="phase3_fix_loop")

        # An attempt that edits the tree is kept only if it moves it forward
        # (``_attempt_not_kept``); otherwise the tree before it is restored and
        # the loop ends, after one more attempt from that tree when the reverted
        # one had made progress. A kept tree is the next attempt's rollback target. An
        # attempt that edits nothing continues the loop only if its evidence
        # retired a blocker, or once after every write it tried was refused.
        current_issues = list(issues)
        current_blockers = blockers_before
        current_score = self._phase3_tree_score(blockers_before, issues)
        seen = {self._blocker_state_key(blockers_before)}
        # Set when the tree an attempt left was replaced: an interrupted
        # attempt's unvalidated edits, or an attempt that was not kept.
        restored = False
        # The notes of reverted attempts a retry followed: later findings replace them otherwise.
        rollback_notes: list[ValidationIssue] = []
        last_issues = list(issues)
        attempts_run = self._repair_progress.get("attempts_run", 0)
        self._repair_last_outcome = None
        retried_refusals = retried_revert = False

        def checkpoint_progress() -> None:
            self._repair_progress = {"attempts_run": attempts_run}
            if self._may_save_phase3_checkpoint():
                self._save_phase3_checkpoint()

        max_attempts = max(_MAX_TOOLCHAIN_FIX_ITERATIONS, self.max_turns - self.total_turns)
        # Which rule ends this loop, for the trace: report-only.
        exit_reason = "attempt cap"
        for _ in range(max_attempts):
            stop_reason = self._phase3_stop_requested()
            if stop_reason:
                logger.warning("Phase 3: %s; preserving unresolved findings", stop_reason)
                exit_reason = stop_reason
                break
            if self._only_unlocated_citations_remain(current_blockers):
                logger.warning("Phase 3: every blocker is a requirement citation the judge could not "
                               "locate after its citation-repair round; no fix attempt can change it")
                exit_reason = "only unlocated requirement citations remain"
                break
            is_first_attempt = attempts_run == 0
            attempts_run += 1
            self._validation_round = f"Fix attempt {attempts_run}"
            self._report_round_step(_fixing_blockers(current_blockers, last_issues))
            self._trace.write(
                EVENT_PHASE_ENTER,
                phase="phase3_fix_attempt",
                attempt=attempts_run,
                blockers=len(current_blockers),
            )
            revision_before = self._workspace_revision()
            obligations_before = self._repair_obligations_revision()
            installs_before = self._install_state()
            probe_before = (getattr(self, "_runtime_probe_cache", None), getattr(self, "_runtime_probe_facts", None))
            booted_before = self._booted(current_blockers, revision_before)
            checkpoint_progress()
            log_before = len(self.tool_calls_log)
            try:
                edits = self._invoke_phase3_fix_loop(current_blockers, is_first_attempt)
            except Exception as exc:
                # e.g. an auth failure, which must propagate - but not leave
                # the attempt's unverified edits behind as the deliverable.
                if revision_before != self._workspace_revision():
                    self._discard_unverified_attempt(
                        f"{type(exc).__name__}", current_issues, attempts_run)
                raise
            round_calls = self.tool_calls_log[log_before:]
            source_changed = revision_before != self._workspace_revision()
            # Installed dependencies are outside the revision, yet decide what tsc and a build resolve.
            obligations_changed = (obligations_before != self._repair_obligations_revision()
                                   or installs_before != self._install_state()
                                   or any(call["tool"] in ("install_dependencies", "run_command")
                                          and call.get("success") for call in round_calls))
            checkpoint_progress()
            if not edits:
                logger.warning(
                    "Phase 3: attempt %d ended with no successful edit (verification changed=%s)",
                    attempts_run, obligations_changed,
                )

            # A stop during a repair is not permission to issue another paid
            # coverage judgment. Preserve last-known findings and the checkpoint.
            mid_attempt_stop = self._phase3_stop_requested(check_turn_budget=False)
            if mid_attempt_stop:
                exit_reason = mid_attempt_stop
                # This attempt's edits were never validated. Ship the tree
                # that was, not an unverified one.
                if source_changed and self._discard_unverified_attempt(mid_attempt_stop, current_issues, attempts_run):
                    restored = True
                break
            if not source_changed and not obligations_changed:
                # Nothing moved on disk or in the evidence: a re-check would find the same.
                self._trace.write(
                    EVENT_PHASE_EXIT, phase="phase3_fix_attempt", attempt=attempts_run,
                    blockers_remaining=len(current_blockers), source_changed=False,
                    successful_writes=edits or 0,
                )
                refused = self._refused_writes(round_calls)
                if refused and not retried_refusals:
                    # Every write was refused: the next attempt is told why (measured: after
                    # such a round the next one wrote 57% of the time and cut blockers 20%).
                    retried_refusals = True
                    self._repair_last_outcome = (
                        f"Fix attempt {attempts_run} changed nothing: every write it tried was refused. "
                        + " ".join(refused))
                    logger.warning("Phase 3: attempt %d had every write refused; one more attempt", attempts_run)
                    continue
                logger.warning("Phase 3: attempt %d changed nothing; ending the fix loop", attempts_run)
                exit_reason = _no_edit_exit_reason(attempts_run, round_calls)
                break
            retried_refusals = False
            # Re-validate: re-running the checks is the only evidence that
            # the repair actually compiles.
            self._begin_validation_round(f"Re-check after fix attempt {attempts_run}")
            issues_after = self._collect_validation_issues()
            blockers_after = [i for i in issues_after if i.severity == "blocker"]
            if (self._phase3_tree_score(blockers_after, issues_after)[1] > current_score[1]
                    and not self._phase3_stop_requested(check_turn_budget=False)):
                # One probe that did not run can be a slow machine, not a hung
                # startup: measure the same tree once more before ranking it.
                logger.warning("Phase 3: attempt %d: the runtime probe did not measure a backend it "
                               "measured before; measuring the same tree again.", attempts_run)
                self._runtime_probe_cache = None
                issues_after = self._collect_validation_issues()
                blockers_after = [i for i in issues_after if i.severity == "blocker"]
            score_after = self._phase3_tree_score(blockers_after, issues_after)
            self._trace.write(
                EVENT_PHASE_EXIT,
                phase="phase3_fix_attempt",
                attempt=attempts_run,
                blockers_remaining=len(blockers_after),
                source_changed=source_changed,
                successful_writes=edits or 0,
            )

            hidden = self._hidden_before(current_issues, probe_before[1], revision_before)
            facts_after = getattr(self, "_runtime_probe_facts", None)
            not_kept = self._attempt_not_kept(
                current_blockers, blockers_after, hidden, judge_ran(issues_after),
                booted_before, self._booted(blockers_after, self._workspace_revision()),
                current_score, score_after,
                before_facts=probe_before[1][1] if probe_before[1] and probe_before[1][0] == revision_before else None,
                after_facts=facts_after[1] if facts_after and facts_after[0] == self._workspace_revision() else None)
            if not_kept and source_changed:
                progressed = (self._retired(current_blockers, blockers_after, judge_ran(issues_after))
                              or self._still_failing(current_blockers, blockers_after))
                restored = self._revert_attempt(attempts_run, not_kept, issues_after, probe_before)
                if restored and progressed and not retried_revert:
                    # It moved some blockers but broke something: one more attempt from the
                    # restored tree, told what broke. A second revert in a row ends the loop.
                    retried_revert = True
                    rollback_notes.append(self._validation_issues[-1])  # the note _revert_attempt adds
                    self._repair_last_outcome = (
                        f"Fix attempt {attempts_run} was rolled back: {not_kept}. The tree from before it is "
                        "restored, so the blockers below are the ones it faced. Repair them without causing that.")
                    logger.warning("Phase 3: attempt %d was reverted after progress; one more attempt", attempts_run)
                    continue
                exit_reason = (f"attempt {attempts_run} reverted: {not_kept}" if restored
                               else f"attempt {attempts_run} not kept, and its tree could not be restored: {not_kept}")
                if not restored:
                    last_issues = issues_after
                    self._report_findings(attempts_run, issues_after)
                break
            # These findings describe the tree that ships; an edit here is a tree newer than any restored one.
            restored = retried_revert = False
            if source_changed:
                self._phase3_rolled_back = False
            last_issues = issues_after
            self._validation_issues = self._with_model_contract(issues_after) + rollback_notes
            self._report_findings(attempts_run, issues_after)
            if not blockers_after:
                logger.info(
                    "Phase 3: All blockers fixed after %d attempt(s) "
                    "(%d non-blocker remain).",
                    attempts_run, len(issues_after),
                )
                self._complete_repair_if_verified()
                checkpoint_progress()
                self._phase3_exit_reason = "all blockers fixed"
                return
            if not_kept:
                # Evidence alone moved (nothing to revert), and it did not move the tree forward.
                exit_reason = f"{_no_edit_exit_reason(attempts_run, round_calls)}: {not_kept}"
                break
            retired = self._retired(current_blockers, blockers_after, judge_ran(issues_after))
            # Kept (or evidence that retired a blocker): the next attempt starts
            # from, and can only be rolled back to, this tree.
            if source_changed and not self._create_snapshot():
                exit_reason = f"attempt {attempts_run} kept, but its tree could not be snapshotted"
                break
            if source_changed:
                self._trace.write(EVENT_SNAPSHOT, before_phase="phase3_kept", attempt=attempts_run,
                                  score=list(score_after))
            state = self._blocker_state_key(blockers_after)
            if state in seen:
                exit_reason = "repeated blocker set"
                break
            seen.add(state)
            self._repair_last_outcome = self._attempt_outcome(
                attempts_run, retired, self._still_failing(current_blockers, blockers_after),
                len(blockers_after), round_calls, source_changed)
            current_issues, current_blockers, current_score = list(issues_after), blockers_after, score_after
            checkpoint_progress()

        self._phase3_exit_reason = exit_reason
        if restored:
            self._validation_issues += [note for note in rollback_notes if note not in self._validation_issues]
            last_issues = list(self._validation_issues)
            self._report_findings(attempts_run, last_issues, restored=True)
        else:
            self._validation_issues = self._with_model_contract(last_issues) + rollback_notes
        checkpoint_progress()
        remaining_blockers = [
            i for i in last_issues if i.severity == "blocker"
        ]
        if remaining_blockers:
            logger.warning(
                "Phase 3: %d blocker(s) remain after %d attempt(s); "
                "preserving partial output, not verified completion.",
                len(remaining_blockers), attempts_run,
            )
            for issue in last_issues:
                logger.warning("  [%s] %s", issue.severity, issue.message)

    @staticmethod
    def _blocker_key(issue: ValidationIssue):
        """What blocks, as attempts compare: a requirement by id (its status is compared
        apart, see ``_retired``), anything else by what it tests (``_finding_key``)."""
        requirement = requirement_status(issue.message)
        return ("requirement", requirement[0]) if requirement else _finding_key(issue.message)

    def _blocker_state_key(self, blockers: list[ValidationIssue]) -> frozenset:
        """A blocker set as the repeated-state check compares it, statuses and runtime errors included."""
        return frozenset(Counter((self._blocker_key(issue), requirement_status(issue.message),
                                  _finding_text(issue.message) if _RUNTIME_TARGET_RE.match(issue.message) else None)
                                 for issue in blockers).items())

    @staticmethod
    def _still_failing(before: list[ValidationIssue], after: list[ValidationIssue]) -> list[str]:
        """The runtime blockers of ``after`` that test what a blocker of ``before`` tested, now
        with another error: neither retired nor added."""
        errors: dict = {}
        for issue in before:
            if _RUNTIME_TARGET_RE.match(issue.message):
                errors.setdefault(_finding_key(issue.message), set()).add(_finding_text(issue.message))
        return [issue.message for issue in after
                if _finding_key(issue.message) in errors
                and _finding_text(issue.message) not in errors[_finding_key(issue.message)]]

    @staticmethod
    def _removed_entities(before_facts: list | None, after_facts: list | None) -> list[str]:
        """The entities the probe measured on a backend that booted before an attempt and
        does not find on that backend, booted, after it: compared by entity name."""
        if not before_facts or not after_facts:
            return []
        after = {backend.get("backend"): set(backend.get("entities") or {})
                 for backend in after_facts if backend.get("boot") == "ok"}
        return [f"{name} (POST {entry.get('path')})"
                for backend in before_facts
                if backend.get("boot") == "ok" and backend.get("backend") in after
                for name, entry in (backend.get("entities") or {}).items()
                if name not in after[backend.get("backend")]]

    def _retired(self, before: list[ValidationIssue], after: list[ValidationIssue],
                 judged_after: bool = True) -> list[str]:
        """The blockers of ``before`` an attempt retired: gone from ``after``, or a requirement
        whose status rose (missing -> partial is progress, though it still blocks). A
        requirement that left the blockers counts only if the judge ran on ``after``."""
        remaining = Counter(self._blocker_key(issue) for issue in after)
        statuses = {r[0]: r[1] for r in map(requirement_status, (i.message for i in after)) if r}
        retired = []
        for issue in before:
            key, requirement = self._blocker_key(issue), requirement_status(issue.message)
            if remaining[key] > 0 and not (requirement and statuses.get(requirement[0], -1) > requirement[1]):
                remaining[key] -= 1
                continue
            if requirement and not judged_after and requirement[0] not in statuses:
                continue  # left only because no judge looked
            retired.append(issue.message)
        return retired

    def _booted(self, blockers: list[ValidationIssue], revision: str) -> bool:
        """Whether the tree starts: the probe's facts for ``revision`` when it measured
        one, else no startup-class blocker."""
        facts = getattr(self, "_runtime_probe_facts", None)
        if facts and facts[0] == revision and facts[1]:
            return all(backend.get("boot") == "ok" for backend in facts[1])
        return not self._startup_blockers(blockers)

    def _hidden_before(self, issues: list[ValidationIssue], facts, revision: str) -> set:
        """The check families that could not run on the tree before an attempt: from its
        findings (``hidden_families``) and the probe's facts for it. Where the probe
        measured that tree, its boot verdict alone says whether runtime checks ran.
        Creates and actions that needed a record are excused per entity instead
        (``_newly_reachable``)."""
        hidden = hidden_families(issues)
        if facts and facts[0] == revision and facts[1]:
            hidden.discard(("runtime", ""))
            if any(backend.get("boot") != "ok" for backend in facts[1]):
                hidden.add(("runtime", ""))
        return hidden

    @staticmethod
    def _newly_reachable(before_facts: list | None, after_facts: list | None):
        """``(finding, entity)`` predicates: whether a ``create contract:`` / ``action call:``
        finding, or a probed entity, belongs to an entity the probe could not create on the
        tree before the attempt. The route's entity comes from the probe's records of the
        tree after it; without records of both trees nothing is excused."""
        if not before_facts or not after_facts:
            return (lambda message: False), (lambda entity: False)
        created = {name for backend in before_facts for name, entry in (backend.get("entities") or {}).items()
                   if entry.get("verdict") == "created"}
        entity_of = {}
        for backend in after_facts:
            for name, entry in (backend.get("entities") or {}).items():
                entity_of[("create contract:", entry.get("path"))] = name
            for call in [*(backend.get("action_calls") or []), *(backend.get("actions") or [])]:
                entity_of[("action call:", call.get("route"))] = call.get("entity")

        def finding(message: str) -> bool:
            prefix = next((p for p in ("create contract:", "action call:") if message.startswith(p)), None)
            match = _POST_ROUTE_RE.search(message)
            entity = entity_of.get((prefix, match.group(1))) if prefix and match else None
            return entity is not None and entity not in created

        return finding, (lambda entity: entity is not None and entity not in created)

    def _attempt_not_kept(self, before: list[ValidationIssue], after: list[ValidationIssue],
                          hidden: set, judged_after: bool, booted_before: bool, booted_after: bool,
                          score_before: tuple, score_after: tuple, *, before_facts: list | None = None,
                          after_facts: list | None = None) -> str | None:
        """Why an attempt is not kept (or, with no edit, does not continue the loop), or None.

        Never kept when the probe no longer finds an entity it measured before
        (``_removed_entities``): a removed create is failing, not fixed. Else kept
        only if (1) it retired a blocker it faced (``_retired``), or a runtime
        blocker now fails with another error (``_still_failing``); (2) every
        observed failure it added (``unresolved_defects``, compared by what it
        tests) belongs to a check
        family that could not run before (``hidden``): fixing the front failure
        of a layer shows what it hid; (3) the blocker count, leaving out the
        added findings of those families, is no higher, a requirement whose
        status fell counting as one more; (4) a tree that started still starts;
        (5) where runtime checks ran before, no more entities fail to be created,
        and no more actions fail, leaving out those of entities the probe could
        not create before (``_newly_reachable``).
        """
        reachable_finding, reachable_entity = self._newly_reachable(before_facts, after_facts)

        def excused(message: str) -> bool:
            return could_not_run(message, hidden) or reachable_finding(message)

        removed = self._removed_entities(before_facts, after_facts)
        if removed:
            return f"the probe no longer finds what it tested before: {', '.join(removed)[:300]}"
        if not self._retired(before, after, judged_after) and not self._still_failing(before, after):
            return f"it retired none of the {counted(len(before), 'blocker')} it faced"
        # A requirement verdict is a judgment, never an observed failure.
        observed_before = Counter(_finding_key(i.message) for i in unresolved_defects(before)
                                  if not requirement_status(i.message))
        added = [i.message for i in unresolved_defects(after)
                 if not requirement_status(i.message) and not excused(i.message)
                 and _finding_key(i.message) not in observed_before]
        if added:
            return f"it added an observed failure: {added[0][:300]}"
        new_keys = Counter(self._blocker_key(i) for i in after) - Counter(self._blocker_key(i) for i in before)
        newly_visible = 0
        for issue in after:
            key = self._blocker_key(issue)
            if new_keys[key] > 0 and excused(issue.message):
                new_keys[key] -= 1
                newly_visible += 1
        statuses = {r[0]: r[1] for r in map(requirement_status, (i.message for i in before)) if r}
        fell = sum(1 for r in map(requirement_status, (i.message for i in after))
                   if r and r[0] in statuses and r[1] < statuses[r[0]])
        count = len(after) - newly_visible + fell
        if count > len(before):
            return f"it left more blockers ({len(before)} -> {count})"
        if booted_before and not booted_after:
            return "the application no longer starts"
        failing_actions = score_after[3] if not after_facts else sum(
            1 for backend in after_facts for call in backend.get("action_calls") or []
            if call.get("verdict") != "effective" and not reachable_entity(call.get("entity")))
        if ("runtime", "") not in hidden and (score_after[2] > score_before[2]
                                              or failing_actions > score_before[3]):
            return (f"more entities or actions fail at runtime (entities {score_before[2]} -> {score_after[2]}, "
                    f"actions {score_before[3]} -> {failing_actions})")
        return None

    def _install_state(self) -> tuple:
        """The dependency-install markers of every frontend project (``node_modules`` is outside
        the workspace revision)."""
        from besser.spec_driven_agent.validation.frontend_build import install_markers

        projects = sorted(rel.rpartition("/")[0] for rel in self._workspace_file_list()
                          if rel.rpartition("/")[2] == "package.json")
        return tuple(install_markers(os.path.join(self.output_dir, project)) for project in projects)

    def _refused_writes(self, calls: list[dict]) -> list[str]:
        """What refused each write of an attempt whose every write was refused; empty
        when it wrote nothing or one write landed."""
        writes = [call for call in calls if call["tool"] in _WRITE_TOOLS_ON_RECORD]
        if not writes or any(call.get("success") for call in writes):
            return []
        # The newest write refusals on record (the record keeps the last few).
        failures = [item for item in self._recent_tool_failures
                    if item.get("tool") in _WRITE_TOOLS_ON_RECORD][-len(writes):] or [
                       {"tool": call["tool"], "path": (call.get("input") or {}).get("path"), "error": "refused"}
                       for call in writes]
        return [f"{item.get('tool')} on {item.get('path')}: {str(item.get('error', ''))[:200]}"
                for item in failures[-5:]]

    def _attempt_outcome(self, attempt: int, retired: list[str], still_failing: list[str], remaining: int,
                         calls: list[dict], source_changed: bool) -> str:
        """What a kept attempt did, for the next attempt's prompt."""
        edited = _edited_paths(calls)
        what = f"edited {', '.join(edited)}" if source_changed and edited else "settled evidence without an edit"
        return (f"Fix attempt {attempt} was kept. It {what} and retired "
                f"{counted(len(retired), 'blocker')}"
                + (": " + "; ".join(message[:160] for message in retired[:5]) if retired else "")
                + (f" (+{len(retired) - 5} more)" if len(retired) > 5 else "")
                + "".join(f". Still failing, now: {message[:300]}" for message in still_failing[:5])
                + f". {counted(remaining, 'blocker')} remain, listed below.")

    def _only_unlocated_citations_remain(self, blockers: list[ValidationIssue]) -> bool:
        """Every blocker is a requirement whose cited code exists but whose citation the
        verifier could not locate (malformed), and the judge already had its
        citation-repair round on this source.

        A fix attempt cannot clear such a blocker: on unchanged source the next validation
        re-verifies the same cached verdict (run r9-p81 spent 3 turns that way), and the fix
        agent may cite a line only in place of a fragment that names one
        (``CITED_FRAGMENT_LINES``). A citation of code that exists nowhere keeps its
        attempt: implementing the behaviour may be the fix.
        """
        unlocated = {issue for verdict in self._requirement_verdicts
                     if verdict.get(_requirements_ledger.EVIDENCE_UNLOCATED) == _requirements_ledger.CITATION_MALFORMED
                     and not verdict.get(_requirements_ledger.CITED_FRAGMENT_LINES)
                     for issue in _requirements_ledger.ledger_issues([verdict])}
        if not blockers or any(issue.message not in unlocated for issue in blockers):
            return False
        return self._requirement_cache_key(self._requirements_to_judge()) in self._requirement_evidence_retries

    @classmethod
    def _startup_blockers(cls, issues: list[ValidationIssue]) -> set[str]:
        """The startup-class blocker messages present in ``issues``."""
        return {i.message for i in issues
                if i.message.lower().startswith(cls._STARTUP_BLOCKER_PREFIXES)}

    def _phase3_tree_score(self, issues: list[ValidationIssue],
                           all_issues: list[ValidationIssue] | None = None) -> tuple[int, int, int, int, int]:
        """Rank one tree. LOWER is better; compare lexicographically.

        ``(boot broken, backends not measured, entities not confirmed created,
        actions not confirmed effective, hard blockers)`` - the runtime
        evidence, with the blocker count last because it is the weakest
        signal: it correlates only at noise level with whether the delivered
        app works. Boot dominates on purpose, so a tree that starts can never
        be discarded for one that does not.

        A backend the probe ran on THIS tree but could not measure (it timed
        out, could not launch, lacked a dependency) is counted next: a hung
        startup yields one ``did not run`` finding and nothing else, and
        without this component that tree would outrank every measured one.

        The next two come from the probe's own per-entity and per-action
        records where it measured THIS tree. Counting issue strings instead
        would re-derive facts already held and count what was rendered: an
        action the probe silently confirmed effective and one it never reached
        would score the same. Strings remain the fallback for a tree
        no probe measured (no backend, or the probe did not run), counted over
        ``all_issues`` when given: an unconfirmed create may be only a warning.
        """
        boot_broken = int(bool(self._startup_blockers(issues)))
        facts = getattr(self, "_runtime_probe_facts", None)
        measured = None
        unmeasured = 0
        if facts and facts[0] == self._workspace_revision():
            entities = actions = 0
            for backend in facts[1]:
                if backend.get("boot") != "ok":
                    unmeasured += 1
                    continue
                measured = True
                for entry in (backend.get("entities") or {}).values():
                    entities += int(entry.get("verdict") != "created")
                for call in backend.get("action_calls") or []:
                    actions += int(call.get("verdict") != "effective")
        if measured is None:
            entities = actions = 0
            for issue in issues if all_issues is None else all_issues:
                message = issue.message.lower()
                if "create contract:" in message or "create unverified:" in message:
                    entities += 1
                elif "action call:" in message or "action unverified:" in message:
                    actions += 1
        return boot_broken, unmeasured, entities, actions, len(_hard_blockers(issues))

    def _runtime_gate_finding(self) -> str | None:
        """The gate's refusal text, or None when the app is clear to ship.

        Detection alone is not enough: a fatal ``mapper config:`` blocker can
        sit in top-priority position across attempts and still ship. So this
        is a gate, not a finding - when it returns
        text the run cannot report itself complete, and the text names the
        runtime failure rather than burying it among forty lint lines.

        Silent when there is nothing to boot (a Qiskit or BAF run has no
        backend to probe), and silent when the probe booted the app and every
        entity create and modelled action came back settled.

        Also silent on an UNSETTLED result: a guessed fixture that a business
        rule legitimately refuses (e.g. a correct 409) is the normal answer
        from a correct app, so an app can pass acceptance while its create
        routes come back ``create unverified:``. Those
        routes stay reported through their own ``runtime unverified:`` finding;
        they are not grounds for the gate to refuse a working application.

        An app nothing started on this tree (no probe boot, no workflow that
        booted it or passed; the probe timed out, a dependency was missing, or
        no check ran) has no runtime evidence at all. That is not a failure:
        the finding is ``validation unverified:`` and the run reports it as not
        verified (``verification_gaps``), never as incomplete.
        """
        if not self._probeable_backends():
            return None
        verdict = self._runtime_probe_verdict
        if (verdict is not None and self._runtime_verdict_revision is not None
                and self._runtime_verdict_revision != self._workspace_revision()):
            # Measured on a tree that is no longer the one being shipped.
            verdict = None
        if verdict == _RUNTIME_OK:
            return None
        observed = [i.message for i in self._validation_issues
                    if i.message.lower().startswith(_RUNTIME_OBSERVED_FAILURE_PREFIXES)]
        fatal_source = [i.message for i in self._validation_issues
                        if i.message.lower().startswith(_RUNTIME_FATAL_SOURCE_PREFIXES)]
        if (verdict in (_RUNTIME_FAILED, None)) and (observed or fatal_source):
            what = ("the probe ran it and watched it fail" if observed
                    else "its own source cannot survive a request")
            return (
                f"{self._RUNTIME_GATE_PREFIX} the delivered application is not "
                f"proven to accept a record - {what}, so this run is NOT complete: "
                + "; ".join(sorted(observed or fatal_source)[:3])[:1200]
            )
        # Said once: a runtime check that did not run already says why nothing started it.
        if self._never_started() and not any(
                i.message.startswith(RUNTIME_CHECK_UNVERIFIED) and "(api scenario " not in i.message
                for i in self._validation_issues):
            return (
                f"validation unverified: {self._RUNTIME_GATE_PREFIX} the delivered application has no "
                "runtime evidence at all - nothing started it on this source revision (no "
                "boot-and-create probe verdict, no test_api workflow that booted it), so it is "
                "not verified. An unrun check is not a passing check."
            )
        return None

    def _apply_runtime_gate(self) -> None:
        """Record the gate's verdict as a blocker so completion is refused.

        Called on every Phase 3 exit, including the budget-exhausted one: the
        honest outcome when the money runs out with the gate still red is an
        incomplete run naming the runtime failure, never a silent pass.
        """
        try:
            finding = self._runtime_gate_finding()
        except Exception as exc:
            # Fail closed: a gate that could not decide is not a pass.
            logger.warning("Runtime gate evaluation failed; completion refused", exc_info=True)
            unverified = _classify_issue(required_check_unverified(
                "the runtime gate", f"it crashed: {type(exc).__name__}: {exc}"))
            if all(i.message != unverified.message for i in self._validation_issues):
                self._validation_issues.append(unverified)
                self._trace.write(
                    EVENT_VALIDATION_ISSUE, phase="phase3_runtime_gate",
                    message=unverified.message[:1000],
                )
            return
        if not finding:
            return
        if any(i.message == finding or i.message.startswith(self._RUNTIME_GATE_PREFIX)
               for i in self._validation_issues):
            return
        logger.warning("Phase 3 runtime gate: %s", finding[:300])
        self._validation_issues.append(_classify_issue(finding))
        self._trace.write(
            EVENT_VALIDATION_ISSUE, phase="phase3_runtime_gate", message=finding[:1000],
        )

    def _discard_unverified_attempt(
        self, reason: str, kept_issues: list[ValidationIssue], attempt: int,
    ) -> bool:
        """Restore the last kept tree over an attempt that was never validated.

        An interrupted repair (provider error, cost or runtime cap) leaves its
        last edits on disk with no validation behind them. The snapshot holds
        the last tree the loop kept, and ``kept_issues`` are that tree's
        findings, so nothing needs re-validating - which a stop must not pay
        for anyway.
        """
        logger.warning(
            "Phase 3: attempt %d was interrupted (%s) before its edits were "
            "validated; restoring the last kept tree.", attempt, reason,
        )
        if not self._restore_snapshot():
            logger.error("Phase 3: the last kept tree could not be restored; the "
                         "interrupted attempt's unvalidated edits remain.")
            return False
        self._phase3_rolled_back = True
        reopened = self.executor.reopen_unverifiable_tasks()
        self._validation_issues = self._with_model_contract(kept_issues) + [_classify_issue(
            f"validation: Phase 3 repair attempt {attempt} was interrupted ({reason}) "
            "before its edits were validated; they were discarded and the last "
            "validated tree ships."
            + (f" {len(reopened)} checklist item(s) verified during that attempt are "
               "open again." if reopened else "")
        )]
        self._trace.write(EVENT_ROLLBACK, phase="phase3_interrupted",
                          attempt=attempt, reason=reason)
        return True

    def _revert_attempt(self, attempt: int, reason: str, attempt_issues: list[ValidationIssue],
                        probe_before: tuple) -> bool:
        """Restore the tree from before ``attempt``, which was not kept, and check it again.

        The code is reverted, the findings are not: what only the discarded
        tree showed is recorded, so a real defect it exposed stays visible.
        The runtime probe's result for the restored tree (``probe_before``,
        tagged with its revision) is reused rather than measured again.
        """
        logger.warning("Phase 3: fix attempt %d is not kept (%s); restoring the tree from before it.",
                       attempt, reason)
        if not self._restore_snapshot():
            logger.error("Phase 3: the tree from before attempt %d could not be restored; "
                         "keeping the attempt's tree and reporting it as it stands.", attempt)
            return False
        self._phase3_rolled_back = True
        # The tree went back; the checklist did not. Anything whose verifier
        # no longer passes was undone by this restore and must stop reporting
        # itself complete.
        reopened = self.executor.reopen_unverifiable_tasks()
        self._runtime_probe_cache, self._runtime_probe_facts = probe_before
        self._begin_validation_round("Check of the restored earlier version")
        restored = self._collect_validation_issues()
        attempt_blockers = [i for i in attempt_issues if i.severity == "blocker"]
        only_in_discarded = sorted({i.message for i in attempt_blockers} - {i.message for i in restored})
        listed = only_in_discarded[:10]
        more = (f" (+{len(only_in_discarded) - len(listed)} more not listed)"
                if len(only_in_discarded) > len(listed) else "")
        seen_only = ("; ".join(listed) + more if listed
                     else "none - every finding in the discarded tree is also in this one")
        undone = (f" The restore also undid completed work: {len(reopened)} checklist "
                  f"item(s) verified during the attempt are open again." if reopened else "")
        self._validation_issues = self._with_model_contract(restored) + [_classify_issue(
            f"validation: Phase 3 fix attempt {attempt} was rolled back - {reason} - to the tree "
            f"from before it.{undone} Findings seen only in the discarded tree (they may "
            "still be real): " + seen_only
        )]
        restored_blockers = [i for i in restored if i.severity == "blocker"]
        self._trace.write(
            EVENT_ROLLBACK, phase="phase3", attempt=attempt, reason=reason,
            hard_blockers_after_attempt=len(_hard_blockers(attempt_blockers)),
            hard_blockers_after_rollback=len(_hard_blockers(restored_blockers)),
            total_blockers_after_rollback=len(restored_blockers),
        )
        return True

    def _invoke_phase3_fix_loop(
        self,
        blockers: list[ValidationIssue],
        is_first_attempt: bool,
    ) -> int:
        """Run one LLM fix attempt against ``blockers``; return the number
        of successful write-tool calls it made.

        The write tools accept only the files the blockers name, their
        neighbours (``repair_scope.edit_scope``) and new files while it runs;
        a blocker that names no file leaves the attempt open.
        """
        if not blockers:
            return 0
        files = self._workspace_file_list()
        named = {issue.message: named_locations(issue.message, files) for issue in blockers}
        # A module missing from the dependencies may be fixed in their manifest.
        manifests_for = frozenset(rel for issue in blockers if issue.message.startswith("missing module:")
                                  for rel, _ in named[issue.message])
        scope = (edit_scope({rel for found in named.values() for rel, _ in found}, self.output_dir, files,
                            manifests_for)
                 if all(named.values()) else None)
        self.executor.set_edit_scope(scope)
        self._trace.write(EVENT_EDIT_SCOPE, allowed=sorted(scope) if scope is not None else None)
        try:
            return self._phase3_fix_turns(blockers, is_first_attempt, named, scope)
        finally:
            self.executor.set_edit_scope(None)

    def _phase3_fix_turns(self, blockers: list[ValidationIssue], is_first_attempt: bool,
                          named: dict[str, list], scope: set[str] | None) -> int:
        """The turns of one fix attempt; see ``_invoke_phase3_fix_loop``.

        A fresh conversation whose prompt carries the files the run has
        edited, the previous attempt's outcome and, within a character budget,
        the current text of the files its blockers name. It ends when the model
        ends its turn or the per-attempt turn budget is spent; no tool is
        forced. Tool calls run through ``_execute_tool_blocks`` so they are
        recorded like Phase 2's.

        The prompt lists the current blockers in priority order, capped
        in count and length (``_fix_prompt_blocker_lines``) — including any ``tsc [...]:`` / ``cargo [...]:`` /
        ``kotlinc [...]:`` lines. When toolchain blockers are present,
        a high-salience reminder is appended instructing the LLM to
        re-run the toolchain via ``run_command`` after each edit, so
        the model verifies its own fixes instead of stopping at the
        first plausible-looking change.
        """
        # Citation/coverage obligations and model-authored tests can be wrong.
        # A verified existing implementation or corrected test need not mutate
        # production code.
        verification_only = all(issue.message.startswith((
            "requirement unverified:", "api scenario:",
            "runtime unverified:", "create unverified:", "verification setup:",
        )) for issue in blockers)

        # Only the toolchain half is needed: it drives the re-run reminder
        # below. Every blocker, toolchain or not, is in the prompt's list.
        toolchain_blockers = [
            i for i in blockers
            if i.message.startswith(("tsc [", "cargo [", "kotlinc ["))
        ]

        # The run-constant sections go in ``system``; this message holds only
        # what depends on the current tree and blockers.
        requirements = self._requirements_for_validation()
        system = self._phase3_fix_system_prompt(requirements)
        prompt_parts: list[str] = [self._dependencies_note, ""] if self._dependencies_note else []
        prompt_parts += [
            "## Current files and symbols (use these paths; do not invent models/ or schemas/ folders)",
            build_inventory(self.output_dir, self.domain_model, self._generator_used or "existing workspace"),
            "",
        ]
        # The action handlers and the model-wide mutation manifest describe
        # behaviour (a third of a measured prompt); a round that only has to
        # make the source compile or import does not need them.
        if not all(i.message.lower().startswith(_SOURCE_DEFECT_PREFIXES) for i in blockers):
            prompt_parts.extend([
                "## Actual action handlers",
                format_action_inventory(collect_action_endpoints(self.output_dir)), "",
            ])
            mutation_manifest = build_mutation_manifest(
                self.output_dir, scenario_records=self._api_scenarios.values(),
                current_revision=self._workspace_revision(),
            )
            if mutation_manifest:
                prompt_parts.extend([mutation_manifest, ""])
        if self._recent_tool_failures:
            prompt_parts.extend([
                "## Recent rejected operations (do not repeat unchanged requests)",
                json.dumps(self._recent_tool_failures, ensure_ascii=False), "",
            ])
        edited = _edited_paths(self.tool_calls_log)
        if edited:
            shown = edited[-_FIX_PROMPT_MAX_EDITED:]
            prompt_parts.extend([
                "## Files this run has edited (most recent last)",
                ", ".join(shown) + (f" (+{len(edited) - len(shown)} earlier)" if len(edited) > len(shown) else ""),
                "",
            ])
        if self._repair_last_outcome:
            prompt_parts.extend(["## Previous fix attempt", self._repair_last_outcome, ""])
        if is_first_attempt:
            prompt_parts.append(
                "Post-generation validation found these unresolved issues. "
                "Distinguish observed execution failures from unverified evidence "
                "and model judgments; resolve each against the original specification:"
            )
        else:
            prompt_parts.append("These blockers remain after the previous fix attempt:")

        prompt_parts.append("")
        prompt_parts.append(
            "Repair in dependency order: first import/database startup, then "
            "schema/router mismatches and failed creates, then business behavior. "
            "Removing client-writable derived fields also requires implementing "
            "their server defaults/computation and updating every route/form that "
            "uses them. Do not delete business rules just to make startup pass. "
            "Call validate_app after each coherent change for fresh diagnostics. "
            "A model-authored API assertion is not the specification: inspect a "
            "failed scenario with test_api(action='get', scenario_id=...) before "
            "changing code. If its expectation contradicts the original request, "
            "correct that same scenario with a specification-grounded correction_reason. "
            "Do not change correct behavior to satisfy a mistaken assertion. "
            "Do not invent authentication, new roles, or destructive data deletion "
            "from a generic usability or release requirement."
        )
        prompt_parts.extend(_fix_prompt_blocker_lines(
            sorted(blockers, key=self._repair_priority), requirements, named))

        # A fresh conversation: nothing read in an earlier round is in it.
        self.executor.forget_shown_results()
        seeded = {} if verification_only else self._seed_named_files(
            [rel for issue in sorted(blockers, key=self._repair_priority) for rel, _ in named.get(issue.message, [])])
        if seeded:
            prompt_parts.extend(["", "The files these blockers name, as read_file returns them now (each read_id "
                                     "is valid for replace_file_lines; quote old_text without the line-number "
                                     "gutter):"])
            prompt_parts.extend(seeded.values())
        # A named file that did not fit the budget above gets the lines around its blocker instead.
        excerpts = self._excerpts_for(blockers, shown=set(seeded))
        if excerpts:
            prompt_parts.append("")
            prompt_parts.append(
                "The offending lines, verbatim from disk. Quote old_text from "
                "here exactly — never abbreviate with '...':"
            )
            prompt_parts.extend(excerpts)

        prompt_parts.append("")
        prompt_parts.append(
            "Resolve authorized dependency setup by installing with run_command, then "
            "validate_app; resolve missing evidence using current exact citations "
            "or test_api. Do not manufacture source changes to "
            "satisfy bookkeeping; make an edit only if behavior is actually missing. "
            "The harness will recheck evidence when this attempt ends."
            if verification_only else
            "Fix actual code defects with modify_file, replace_file_lines, or write_file. "
            "After repeated text matching failures, read the target block and use "
            "replace_file_lines with its read_id and inclusive line numbers instead of "
            "quoting old_text again. Reading first "
            "is fine, but you are not done until defects are repaired and verified. "
            "An evidence/citation error alone does not justify changing correct code."
        )
        if scope is not None:
            listed = sorted(scope)
            shown = listed[:_FIX_PROMPT_MAX_SCOPE_FILES]
            prompt_parts.extend(["", (
                "This attempt may edit the files these blockers name and the files those import "
                f"or that import them: {', '.join(shown)}"
                + (f" (+{len(listed) - len(shown)} more)" if len(listed) > len(shown) else "")
                + ". It may also create new files. To edit any other file, add needed_for to that "
                "call, naming the blocker the file is needed for.")])

        # When the blockers include toolchain errors and shell tools are on,
        # instruct the LLM to drive the toolchain itself with run_command — that's the
        # only way to know whether a fix actually compiles. We do this
        # as additional text in the same user turn (vs. a separate
        # message) so the LLM sees the request as part of the brief.
        if toolchain_blockers and not self.allow_shell_tools:
            cmds = self._toolchain_commands_for(toolchain_blockers)
            prompt_parts.append("")
            prompt_parts.append(
                "Shell tools are unavailable in this run, so do not call run_command; "
                "the harness re-runs these checks when this attempt ends:"
            )
            prompt_parts.append("\n".join(f"  - {c}" for c in cmds))
            prompt_parts.append(
                "Fix every reported compile / type error before ending the attempt."
            )
        elif toolchain_blockers:
            cmds = self._toolchain_commands_for(toolchain_blockers)
            cmd_lines = "\n".join(f"  - {c}" for c in cmds)
            prompt_parts.append("")
            prompt_parts.append(
                "After each edit, re-run the relevant toolchain check "
                "using run_command to confirm the error is gone:"
            )
            prompt_parts.append(cmd_lines)
            prompt_parts.append(
                "Keep iterating (edit -> re-run) until the toolchain "
                "reports zero errors. Do not declare done while any "
                "compile / type error is still reported."
            )

        fix_prompt = "\n".join(prompt_parts)
        messages: list[dict] = [{"role": "user", "content": fix_prompt}]

        # Inject a high-salience reminder as a separate user message
        # right after the prompt. Matches the pattern used by the
        # per-file modify-loop guard: a <system-reminder>-tagged block
        # the LLM sees at response-time, not buried inside the prompt.
        if toolchain_blockers:
            reminder = self._build_toolchain_reminder(toolchain_blockers)
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": reminder}],
            })

        edits = 0                 # successful write-tool calls this attempt
        # needed_for is offered only while the attempt's edits are limited.
        tools = with_scope_widening(self.tools) if scope is not None else self.tools
        unconfirmed_at_reminder: set | None = None
        truncations = 0           # "emit less" retries after a max_tokens stop
        turn_cap = _PHASE3_FIX_TURNS
        turn = 0
        while True:
            stop_reason = self._phase3_stop_requested()
            if stop_reason:
                logger.warning("Phase 3: %s", stop_reason)
                return edits
            if turn >= turn_cap:
                return edits
            turn += 1
            self.total_turns += 1
            self._trace.write(EVENT_TURN_START, turn=self.total_turns, phase="phase3_fix")
            # Same compaction as Phase 2; the blockers ride in its work state.
            messages = self._maybe_compact(messages)
            request_messages = without_rejected_edit_drafts(messages)
            try:
                response = self.client.chat(
                    system=system, messages=request_messages, tools=tools,
                )
            except InvalidApiKeyError:
                # Same rule as Phase 2: an auth failure must PROPAGATE so the
                # runner reports INVALID_KEY. Swallowed here it would read as
                # "repair interrupted", which tells the user nothing about the
                # one thing they can fix.
                raise
            except Exception as exc:
                # Surface the failure instead of silently exiting the fix
                # loop — callers and logs need to see why validation bailed.
                # No retry here on purpose: the client already spent its own
                # 5-attempt backoff (and its outage-fallback model switch)
                # before the exception reached us.
                logger.warning(
                    "Phase 3: LLM call failed on fix turn %d, aborting fix loop: %s",
                    turn, exc,
                )
                if isinstance(exc, DeadlineReached) and not exc.cancelled:
                    return edits  # no time for another turn; the attempt so far is still checked
                self._phase3_interrupted = True
                self._phase3_interrupt_detail = (str(exc) if isinstance(exc, DeadlineReached)
                                                 else f"provider call failed: {type(exc).__name__}")
                return edits
            # Repair calls can trigger the outage fallback too, as in Phase 2.
            self._notify_model_switch()
            self._trace_cost_update(self.total_turns, phase="phase3_fix")
            self._trace.write(EVENT_TURN_END, turn=self.total_turns, phase="phase3_fix")
            # A stop can arrive while a provider request is in flight. Do not
            # apply its returned mutations after cancellation. The final allowed
            # turn may still execute its tools; it is not a new provider call.
            if self._phase3_stop_requested(check_turn_budget=False):
                return edits
            if response["stop_reason"] == "end_turn":
                # Listed creates still unread get another turn while each
                # reminder confirms at least one; otherwise the attempt ends.
                unconfirmed = self._unconfirmed_creates(blockers)
                if unconfirmed and (unconfirmed_at_reminder is None
                                    or unconfirmed < unconfirmed_at_reminder):
                    unconfirmed_at_reminder = unconfirmed
                    turn_cap = max(turn_cap, turn + 1)
                    if response["content"]:
                        messages.append({"role": "assistant", "content": response["content"]})
                    paths = ", ".join(f"POST {path}" for _, path in sorted(unconfirmed))
                    messages.append({"role": "user", "content": [{
                        "type": "text", "text": _PHASE3_CREATE_EVIDENCE_REMINDER.format(paths=paths)}]})
                    continue
                return edits
            if (response["stop_reason"] in ("max_tokens", "length")
                    and truncations < self._MAX_TRUNCATION_RETRIES):
                # Recoverable, as in Phase 2: drop the cut-off turn, ask for less.
                truncations += 1
                logger.warning("Phase 3: output truncated on fix turn %d; retry %d/%d",
                               turn, truncations, self._MAX_TRUNCATION_RETRIES)
                messages.append(self._truncation_retry_prompt())
                continue
            if response["stop_reason"] != "tool_use":
                # Without this the loop appends nothing and re-sends an
                # identical request until the cap. Phase 2 already breaks here.
                logger.warning(
                    "Phase 3: unexpected stop_reason %r on fix turn %d — "
                    "stopping the fix loop instead of re-sending the same "
                    "request", response["stop_reason"], turn,
                )
                self._phase3_interrupted = True
                self._phase3_interrupt_detail = (
                    f"unexpected stop_reason {response['stop_reason']!r}")
                return edits
            messages.append({"role": "assistant", "content": response["content"]})
            tool_blocks = _tool_use_blocks(response["content"])
            # Through the Phase 2 path, so the trace, recipe and sidecar see
            # these calls; the raw executor would leave no record of them.
            logged_before = len(self.tool_calls_log)
            tool_results = self._execute_tool_blocks(tool_blocks, self.total_turns - 1)
            edits += sum(
                1 for entry in self.tool_calls_log[logged_before:]
                if entry["tool"] in _WRITE_TOOLS_ON_RECORD and entry["success"]
            )
            for widening in self.executor.scope_widenings:
                logger.info("Phase 3: %s widened the attempt's files to %s: %s",
                            widening["tool"], widening["path"], widening["needed_for"])
                self._trace.write(EVENT_EDIT_SCOPE, widened=widening["path"], tool=widening["tool"],
                                  needed_for=widening["needed_for"])
            self.executor.scope_widenings.clear()
            messages.append({"role": "user", "content": tool_results})
            # The same streak/repeat guards Phase 2 and the fix cycle get.
            # Without them the bounded repair loop would run on _is_stuck
            # alone, on a tighter budget than either.
            if self._apply_edit_loop_guards(messages, where="phase 3 repair"):
                # Falling out of the loop would return ``None``, so an attempt
                # that DID write would report zero writes to the outer cycle,
                # the trace and the log.
                return edits
            if self._may_save_phase3_checkpoint():
                self._save_phase3_checkpoint()

    def _unconfirmed_creates(self, blockers: list[ValidationIssue]) -> set[tuple[str, str]]:
        """Listed ``create unverified`` (backend, path) pairs no current scenario read back."""
        revision = self._workspace_revision()
        confirmed = {(record["report"].get("backend"), path)
                     for record in self._api_scenarios.values()
                     if record["revision"] == revision
                     for path in confirmed_create_paths(record["report"])}
        listed = (_CREATE_UNVERIFIED.search(issue.message) for issue in blockers)
        return {(match.group(1), match.group(2)) for match in listed
                if match and not any(backend == match.group(1) and route_serves(match.group(2), path)
                                     for backend, path in confirmed)}

    def _may_save_phase3_checkpoint(self) -> bool:
        """Whether a repair checkpoint may be written now.

        It holds no conversation, so it must not replace the checkpoint of a
        Phase 2 that stopped early (provider error, turn cap, edit loop): that
        run stays resumable as Phase 2.
        """
        return (self._checkpoint_phase == "phase3" or self._phase2_exited_cleanly
                or self._phase2_stop_reason == "validation_required")

    def _phase3_fix_system_prompt(self, requirements: list[dict]) -> str:
        """The fix prompt's run-constant part: instructions, request, handoff, model, ledger.

        Byte-identical in every round of a run, so the provider reads it from
        its prompt cache instead of re-writing it with each round's blockers.
        The handoff is set by Phase 2 only, so it is constant here too.
        """
        parts = [
            "You are fixing validation errors in generated code. "
            "Fix each issue concisely. Call validate_app to recheck startup and "
            "data entry after a coherent repair. When shell tools are unavailable, "
            "validate_app and test_api are the supported verification tools. Use test_api "
            "for specification-based workflow assertions and invalid inputs. When shell tools "
            "are available and the report contains "
            "toolchain errors (tsc / cargo / kotlinc), verify each fix by "
            "re-running that toolchain with run_command; the diff alone does "
            "not show that it compiles.", "",
        ]
        if self._instructions:
            parts.extend([
                "## Original request (the authority for required behavior)",
                self._instructions, "",
            ])
        if self._phase2_inspection_handoff:
            parts.extend(["## Prior inspection handoff", self._phase2_inspection_handoff, ""])
        if self.domain_model is not None:
            serialized = serialize_domain_model(self.domain_model)
            parts.extend([
                "## Domain model and conversion losses",
                "conversion_issues are NOT implemented constraints. Recover their "
                "intended behavior using actual relationship names; the original "
                "request takes precedence over conflicting model expressions.",
                *([DROPPED_RULES_NOTE] if DROPPED in serialized else []),
                to_prompt_json(serialized), "",
            ])
        if requirements:
            parts.extend([
                "## Requirements to verify (including conversion recovery)",
                _requirements_ledger.render_requirements(requirements),
            ])
        return "\n".join(parts).rstrip("\n")

    def _seed_named_files(self, paths: list[str]) -> dict[str, str]:
        """``{path: prompt block}``: the files the blockers name, in order, read through
        ``read_file`` so a re-read in this conversation is answered with a stub. A file
        that does not fit what is left of ``_FIX_PROMPT_FILE_CHARS`` is left out unread:
        a read records the file as seen."""
        seeded: dict[str, str] = {}
        budget = _FIX_PROMPT_FILE_CHARS
        for rel in dict.fromkeys(paths):
            path = os.path.join(self.output_dir, rel)
            try:
                if os.path.getsize(path) > budget:
                    continue
                with open(path, encoding="utf-8") as handle:
                    lines = handle.read().split("\n")
            except (OSError, UnicodeDecodeError):
                continue
            # As read_file numbers it.
            if sum(len(gutter_line(n, line)) + 1 for n, line in enumerate(lines, 1)) - 1 > budget:
                continue
            result = self.executor.execute_typed("read_file", {"path": rel})
            payload = result.payload if result.succeeded else {}
            content = payload.get("content")
            if not content:
                continue
            budget -= len(content)
            seeded[rel] = "\n".join(["", f"{rel} (read_id {payload.get('read_id')}, whole file):",
                                     "```", content, "```"])
        return seeded

    def _excerpts_for(
        self, blockers: list[ValidationIssue], context: int = 5, limit: int = 3,
        shown: set[str] = frozenset(),
    ) -> list[str]:
        """Numbered source windows around each blocker that names file+line, except in ``shown`` files."""
        out: list[str] = []
        seen: set[tuple[str, int]] = set()
        for issue in blockers:
            match = self._FILE_LINE_RE.search(issue.message)
            if not match:
                continue
            rel, line_no = match.group(1), int(match.group(2))
            if (rel, line_no) in seen or len(seen) >= limit or rel in shown:
                continue
            path = os.path.realpath(os.path.join(self.output_dir, rel.replace("/", os.sep)))
            workspace = os.path.realpath(self.output_dir)
            try:
                if os.path.commonpath([workspace, path]) != workspace:
                    continue
            except ValueError:
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    lines = fh.read().splitlines()
            except (OSError, UnicodeError):
                continue
            if not 1 <= line_no <= len(lines):
                continue
            seen.add((rel, line_no))
            lo = max(0, line_no - 1 - context)
            hi = min(len(lines), line_no + context)
            body = chr(10).join(
                f"{n + 1:>5}| {lines[n]}" for n in range(lo, hi)
            )
            header = f"{rel} (lines {lo + 1}-{hi}):"
            fence = "```"
            out.append(chr(10).join(["", header, fence, body, fence]))
        return out

    def _toolchain_commands_for(self, toolchain_blockers: list[ValidationIssue]) -> list[str]:
        """Delegate to the toolchain module."""
        return _toolchain_commands_for(toolchain_blockers, shell_tools=self.allow_shell_tools)

    def _build_toolchain_reminder(self, toolchain_blockers: list[ValidationIssue]) -> str:
        """Delegate to the toolchain module."""
        return _build_toolchain_reminder(toolchain_blockers, shell_tools=self.allow_shell_tools)

    @staticmethod
    def _repair_priority(issue: ValidationIssue) -> tuple[int, str]:
        message = issue.message.lower()
        # ``create contract:`` / ``action call:`` are the runtime gate's own
        # observed failures - the app booted and still could not take a record.
        # They belong beside the startup classes, not below a lint finding.
        if message.startswith((
            "syntax", "python contract:", "mapper config:", "application startup:",
            "missing module:", "create contract:", "action call:",
        )):
            return 0, message
        if message.startswith(("data contract:", "runtime unverified:")):
            return 1, message
        if message.startswith(("requirement", "task incomplete:")):
            return 3, message
        return 2, message

    def _create_snapshot(self) -> bool:
        """Snapshot the output directory as the rollback target; True on success.

        Taken before Phase 3, again once entry validation has applied its own
        rewrites, and after every kept repair attempt, so the snapshot always
        holds the tree the next attempt starts from.

        Built in a staging directory and swapped in by rename: deleting the
        existing snapshot first and copying after leaves the run with no
        rollback target at all if the copy dies partway, and this runs
        several times per Phase 3.
        """
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        staging_path = os.path.join(self.output_dir, _SNAPSHOT_STAGING_DIR)
        try:
            if os.path.exists(staging_path):
                shutil.rmtree(staging_path)

            # Copy everything except the snapshot dir and the run bookkeeping
            # a rollback must never revert (see _ROLLBACK_PRESERVED).
            for item in os.listdir(self.output_dir):
                src = os.path.join(self.output_dir, item)
                # A planted link would be copied as its target's contents.
                if item in _ROLLBACK_PRESERVED or not is_plain_entry(src):
                    continue
                dst = os.path.join(staging_path, item)
                if os.path.isdir(src):
                    copytree_plain(src, dst, _SNAPSHOT_IGNORED_DIRS)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            os.makedirs(staging_path, exist_ok=True)

            if os.path.exists(snapshot_path):
                shutil.rmtree(snapshot_path)
            os.replace(staging_path, snapshot_path)
            # The calls whose edits the snapshot holds; a restore discards the later ones.
            self._snapshot_calls = len(self.tool_calls_log)
            self._snapshot_task_writes = self.executor.task_writes()
            logger.info("Snapshot created at %s", snapshot_path)
            return True
        except Exception as e:
            logger.warning("Failed to create snapshot; the previous rollback target stays: %s", e)
            shutil.rmtree(staging_path, ignore_errors=True)
            return False

    def _restore_snapshot(self) -> bool:
        """Restore the output directory from the post-Phase-1 snapshot.

        Returns True only if the workspace now holds the snapshot's content.

        The current tree is *moved* aside and only discarded once the restore
        succeeds; if anything fails it is moved back. Deleting first and copying
        after leaves a half-erased workspace with no way back when the copy dies
        partway (full disk, locked file), which the run then packages as the
        deliverable.
        """
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        if not os.path.isdir(snapshot_path):
            logger.warning("No snapshot to restore from")
            return False

        discard_path = os.path.join(self.output_dir, _ROLLBACK_DISCARD_DIR)
        moved: list[tuple[str, str]] = []
        try:
            if os.path.exists(discard_path):
                shutil.rmtree(discard_path)
            os.makedirs(discard_path, exist_ok=True)

            # Park the current tree instead of deleting it. Same filesystem,
            # so os.replace is a rename and costs nothing.
            for item in os.listdir(self.output_dir):
                if item in _ROLLBACK_PRESERVED:
                    continue
                src = os.path.join(self.output_dir, item)
                dst = os.path.join(discard_path, item)
                os.replace(src, dst)
                moved.append((src, dst))

            for item in os.listdir(snapshot_path):
                src = os.path.join(snapshot_path, item)
                dst = os.path.join(self.output_dir, item)
                if not is_plain_entry(src):
                    continue
                if os.path.isdir(src):
                    copytree_plain(src, dst)
                else:
                    shutil.copy2(src, dst)
        except Exception as e:
            logger.error("Rollback failed, reverting to the pre-rollback tree: %s", e)
            try:
                # Undo the partial restore, then put the parked tree back.
                for _src, dst in moved:
                    landed = os.path.join(self.output_dir, os.path.basename(dst))
                    if os.path.isdir(landed):
                        shutil.rmtree(landed, ignore_errors=True)
                    elif os.path.isfile(landed):
                        os.remove(landed)
                for src, dst in moved:
                    os.replace(dst, src)
                shutil.rmtree(discard_path, ignore_errors=True)
            except Exception as revert_exc:
                # Both directions failed. Say so loudly and leave the parked
                # copy in place — it is the only intact tree left.
                logger.error(
                    "Could not revert the rollback either; the Phase 2 output "
                    "is preserved under %s: %s", _ROLLBACK_DISCARD_DIR, revert_exc,
                )
            return False

        self._carry_over_ignored_dirs(discard_path)
        shutil.rmtree(discard_path, ignore_errors=True)
        if self._snapshot_calls is not None:
            for call in self.tool_calls_log[self._snapshot_calls:]:
                call["rolled_back"] = True
            self.executor.restore_task_writes(self._snapshot_task_writes)
        logger.info("Restored from snapshot")
        return True

    def _carry_over_ignored_dirs(self, discard_path: str) -> None:
        """Move installed dependencies (``_SNAPSHOT_IGNORED_DIRS``) from the
        parked tree back into the restored one.

        The snapshot skips them to stay small, so without this a rollback
        leaves the app with no ``node_modules`` / ``.venv`` and the next
        validation reports missing dependencies. Only restored into a
        directory that still exists; failures are logged, never raised.
        """
        for root, dirs, _files in walk_plain(discard_path):
            for name in [d for d in dirs if d in _SNAPSHOT_IGNORED_DIRS]:
                dirs.remove(name)
                rel = os.path.relpath(os.path.join(root, name), discard_path)
                target = os.path.join(self.output_dir, rel)
                if os.path.isdir(os.path.dirname(target)) and not os.path.exists(target):
                    try:
                        os.replace(os.path.join(root, name), target)
                    except OSError as exc:
                        logger.warning("Could not keep %s across the rollback: %s", rel, exc)

    def _remove_snapshot(self) -> None:
        """Clean up the snapshot directory (and any staging left by a failure)."""
        for name in (_SNAPSHOT_DIR, _SNAPSHOT_STAGING_DIR):
            path = os.path.join(self.output_dir, name)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass

