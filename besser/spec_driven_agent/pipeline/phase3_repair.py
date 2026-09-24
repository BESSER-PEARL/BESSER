"""Phase 3: validate the delivered tree, repair it, and keep the best version.

The repair loop's guards are the interesting part and every number in them is
mined from recorded runs rather than chosen: 69% of 393 runs ended on a stall
guard, leaving 6,774 turns and $514 of declared budget unspent, and a round of
REJECTED edits is byte-identical on disk to a barren one -- which is why a
rejected round gets a second attempt (57% wrote source next round) while a
read-only replay ends immediately (0 of 4 ever wrote again).

Snapshot/rollback lives here too because it is the same decision: the snapshot
is re-taken on every strictly better tree and the BEST is restored, not the
last, ranked runtime-first. Run 673hzu0z walked 6-10-7-2-9-11-9-2-6-6-6 and
shipped 6; 10 of 22 runs with a repair loop ended worse than a state they had
already reached.

Mixed into ``LLMOrchestrator``; all run state is reached through ``self``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time

from besser.spec_driven_agent.agent.history_eviction import (
    without_rejected_edit_drafts,
)
from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.planning import requirements_ledger as _requirements_ledger
from besser.spec_driven_agent.planning.action_inventory import (
    collect_action_endpoints,
    format_action_inventory,
)
from besser.spec_driven_agent.agent.prompt_builder import build_inventory
from besser.spec_driven_agent.planning.mutation_inventory import build_mutation_manifest
from besser.spec_driven_agent.model_serializer import serialize_domain_model
from besser.spec_driven_agent.pipeline.constants import (
    _MAX_TOOLCHAIN_FIX_ITERATIONS,
    _PHASE3_FIX_TURNS,
    _PHASE3_NO_EDIT_REMINDER,
    _PHASE3_NO_PROGRESS_ROUNDS,
    _PHASE3_PLATEAU_ROUNDS,
    _ROLLBACK_DISCARD_DIR,
    _ROLLBACK_PRESERVED,
    _RUNTIME_FAILED,
    _RUNTIME_OK,
    _RUNTIME_FATAL_SOURCE_PREFIXES,
    _RUNTIME_OBSERVED_FAILURE_PREFIXES,
    _SNAPSHOT_IGNORED_DIRS,
    _SNAPSHOT_STAGING_DIR,
    _WRITE_TOOLS_ON_RECORD,
)
from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.state.tracing import (
    EVENT_PHASE_ENTER,
    EVENT_PHASE_EXIT,
    EVENT_ROLLBACK,
    EVENT_SNAPSHOT,
    EVENT_VALIDATION_ISSUE,
)
from besser.spec_driven_agent.validation.issues import (
    ValidationIssue,
    _classify_issue,
    _hard_blockers,
)
from besser.spec_driven_agent.validation.toolchain import (
    _build_toolchain_reminder,
    _toolchain_commands_for,
)

logger = logging.getLogger(__name__)


class Phase3RepairMixin:
    """_run_phase3_validation and friends; see the module docstring."""

    def _complete_repair_if_verified(self) -> None:
        if self._phase2_stop_reason == "validation_required":
            self._phase2_exited_cleanly = True
            self._phase2_stop_reason = "completed"

    def _phase3_stop_requested(self, *, check_turn_budget: bool = True) -> str | None:
        """One stop gate for repair turns and the validation calls between them."""
        reason = None
        if self._start_time is not None and time.monotonic() - self._start_time > self.max_runtime_seconds:
            reason = "runtime budget exhausted"
        elif self.max_cost_usd is not None and self.client.usage.estimated_cost >= self.max_cost_usd:
            reason = "cost budget exhausted"
        elif check_turn_budget and self.total_turns >= self.max_turns:
            # No further edit turn, but final verification of the last accepted
            # tool batch may still run. Do not poison that check as cancelled.
            return "turn budget exhausted"
        elif self._should_continue is not None and not self._should_continue():
            reason = "cancellation requested"
        elif self._phase3_interrupted:
            reason = "repair interrupted before verification completed"
            if self._phase3_interrupt_detail:
                reason = f"{reason} ({self._phase3_interrupt_detail})"
        if reason:
            self._phase3_interrupted = True
        return reason

    def _repair_obligations_revision(self) -> str:
        """Test corrections and checklist evidence are progress without source edits.

        Two things deliberately do NOT move it, because both moved it without
        anything being discharged and so reset the no-progress streak:

        * ``attempts`` in the task snapshot - a REFUSED ``task_list(done=...)``
          increments it, so failing to close an item read as closing one;
        * the scenario KEY - the model may pass its own ``scenario_id``, so
          re-registering identical requests under a new name minted a new
          entry. Only distinct scenario CONTENT counts, so a rename is not a
          new obligation. 84 rounds across 65 runs survived on these two.
        """
        tasks = [{k: v for k, v in task.items() if k != "attempts"}
                 for task in self.executor.task_snapshot()]
        scenarios = sorted(
            json.dumps(record["scenario"], sort_keys=True)
            for record in self._api_scenarios.values()
        )
        payload = {"tasks": tasks, "scenarios": sorted(set(scenarios))}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _run_phase3_validation(self) -> None:
        """Validate, repair, then apply the runtime exit gate.

        The gate runs on EVERY exit from the repair cycle - clean, stalled,
        budget-exhausted or crashed - because those are exactly the paths a
        dead app used to leave through quietly. See ``_apply_runtime_gate``.
        """
        try:
            self._run_phase3_repair_cycle()
        finally:
            self._apply_runtime_gate()

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
        remain explicitly incomplete, never accepted as verified output. This
        closes the gap where Phase 3 used to surface tsc errors as
        warnings (no fix attempt) and never invoked cargo / kotlinc
        at all, leaving the per-project compile-pass at 0/n for TS /
        Rust / Kotlin runs.
        """
        # The final allowed editing turn still deserves validation. The turn
        # cap prevents another repair interaction, not local checks/final review.
        stop_reason = self._phase3_stop_requested(check_turn_budget=False)
        if stop_reason:
            logger.warning("Skipping Phase 3 -- %s", stop_reason)
            self._phase3_exit_reason = f"skipped: {stop_reason}"
            self._validation_issues.append(ValidationIssue(
                "blocker", f"requirement unverified: Phase 3 validation did not run: {stop_reason}. "
                "The current application has not completed final verification.",
            ))
            if self._checkpoint_phase == "phase3":
                self._save_phase3_checkpoint()
            return

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

        if self.on_progress:
            self.on_progress(
                self.total_turns,
                "validation",
                f"{len(blockers_before)} blockers / {len(issues)} total",
            )

        # Auto-fix is opt-in (default off). Industry pattern: report by
        # default, fix on request. Avoid the LLM running ``npm install`` —
        # post-install hooks execute arbitrary code from chosen packages.
        if not self.auto_fix_issues:
            logger.info(
                "Phase 3: auto_fix_issues=False — issues recorded, no LLM fix loop."
            )
            self._phase3_exit_reason = "auto-fix disabled"
            return

        # Auto-fix only consumes BLOCKER issues. Style warnings (unused
        # imports, line length) and soft warnings (tsc type hints) are
        # left as-is; they don't justify burning LLM turns.
        if not blockers_before:
            self._phase3_exit_reason = "no blocker-class issues"
            logger.info(
                "Phase 3: auto_fix_issues=True but no blocker-class issues — "
                "skipping LLM fix loop. %d non-blocker issue(s) recorded.",
                len(warnings_before) + len(styles_before),
            )
            return

        # Repair within the remaining budgets. Fixing an upstream failure can
        # expose downstream failures, so blocker counts are not a progress
        # metric. One unchanged/repeated source state stops an unproductive loop.
        current_blockers = blockers_before
        prev_blocker_count = len(blockers_before)
        # Only a repair that actually wrote something can be rolled back.
        source_ever_changed = False
        last_issues = list(issues)
        # Best tree seen so far, and the snapshot that holds it. The snapshot
        # starts as the Phase 3 entry tree; every strictly better tree replaces
        # it, so the restore at the end returns the BEST state reached rather
        # than the last. Run 673hzu0z walked 6-10-7-2-9-11-9-2-6-6-6 and shipped
        # 6; 10 of the 22 runs with a repair loop ended worse than a state they
        # had already reached.
        best_issues = list(blockers_before)
        best_score = prev_score = self._phase3_tree_score(blockers_before)
        progress = self._repair_progress
        attempts_run = progress.get("attempts_run", 0)
        last_validated_revision = self._workspace_revision()
        last_obligations_revision = self._repair_obligations_revision()
        same_validated_state = (
            progress.get("last_validated_revision") == last_validated_revision
            and progress.get("last_obligations_revision") == last_obligations_revision
        )
        no_progress_streak = progress.get("no_progress_streak", 0) if same_validated_state else 0
        # Rounds that edited the tree without improving its score. Not
        # checkpointed: a resume re-derives the score from the tree it finds.
        plateau_streak = 0
        seen_states = {
            (source, obligations, tuple(messages))
            for source, obligations, messages in progress.get("seen_states", [])
        }

        def checkpoint_progress() -> None:
            self._repair_progress = {
                "attempts_run": attempts_run,
                "no_progress_streak": no_progress_streak,
                "seen_states": [
                    [source, obligations, list(messages)]
                    for source, obligations, messages in sorted(seen_states)
                ],
                "last_validated_revision": last_validated_revision,
                "last_obligations_revision": last_obligations_revision,
            }
            self._save_phase3_checkpoint()

        max_attempts = max(_MAX_TOOLCHAIN_FIX_ITERATIONS, self.max_turns - self.total_turns)
        # Which guard ends this loop, for the trace. Reconstructing it from
        # phase_enter/phase_exit pairs after the fact is guesswork, and the
        # guard histogram is the only way to tell a tight guard from a weak
        # model. Report-only: nothing reads it back as a decision input.
        exit_reason = "attempt cap"
        for _ in range(max_attempts):
            stop_reason = self._phase3_stop_requested()
            if stop_reason:
                logger.warning("Phase 3: %s; preserving unresolved findings", stop_reason)
                exit_reason = stop_reason
                break
            is_first_attempt = attempts_run == 0
            attempts_run += 1
            self._trace.write(
                EVENT_PHASE_ENTER,
                phase="phase3_fix_attempt",
                attempt=attempts_run,
                blockers=len(current_blockers),
            )
            revision_before = self._workspace_revision()
            obligations_before = self._repair_obligations_revision()
            checkpoint_progress()
            log_before = len(self.tool_calls_log)
            edits = self._invoke_phase3_fix_loop(current_blockers, is_first_attempt)
            # Writes the attempt REACHED FOR, successful or not. A rejected
            # edit is not nothing: the rejection is fed back into the next
            # attempt's prompt as a recent-failure, so that attempt is not the
            # same request again. See the replay test below.
            attempted_writes = sum(
                1 for entry in self.tool_calls_log[log_before:]
                if entry["tool"] in _WRITE_TOOLS_ON_RECORD
            )
            source_changed = revision_before != self._workspace_revision()
            source_ever_changed = source_ever_changed or source_changed
            obligations_changed = obligations_before != self._repair_obligations_revision()
            checkpoint_progress()
            if not edits:
                # The one line that was missing from run 7f918e11's log: the
                # attempt burned its turns and changed nothing.
                logger.warning(
                    "Phase 3: attempt %d ended with no successful edit (verification changed=%s)",
                    attempts_run, obligations_changed,
                )

            # A stop during a repair is not permission to issue another paid
            # coverage judgment. Preserve last-known findings and the checkpoint.
            mid_attempt_stop = self._phase3_stop_requested(check_turn_budget=False)
            if mid_attempt_stop:
                exit_reason = mid_attempt_stop
                break
            # Re-validate. The bench's per-project compile-pass score
            # only cares about a clean toolchain, so re-running these
            # is what actually drives the metric.
            issues_after = self._collect_validation_issues()
            last_issues = issues_after
            self._validation_issues = self._with_model_contract(issues_after)
            last_validated_revision = self._workspace_revision()
            last_obligations_revision = self._repair_obligations_revision()
            blockers_after = [i for i in issues_after if i.severity == "blocker"]
            self._trace.write(
                EVENT_PHASE_EXIT,
                phase="phase3_fix_attempt",
                attempt=attempts_run,
                blockers_remaining=len(blockers_after),
                source_changed=source_changed,
                successful_writes=edits or 0,
            )

            if not blockers_after:
                logger.info(
                    "Phase 3: All blockers fixed after %d attempt(s) "
                    "(%d non-blocker remain).",
                    attempts_run, len(issues_after),
                )
                self._validation_issues = self._with_model_contract(issues_after)
                self._complete_repair_if_verified()
                checkpoint_progress()
                self._phase3_exit_reason = "all blockers fixed"
                return

            # Keep the best tree, not the last one. Strictly better only, so a
            # flat round never overwrites the snapshot it would restore.
            score_after = self._phase3_tree_score(blockers_after)
            if score_after < best_score:
                logger.info(
                    "Phase 3: attempt %d is the best tree so far %s -> %s; "
                    "re-snapshotting it as the rollback target.",
                    attempts_run, best_score, score_after,
                )
                self._create_snapshot()
                self._trace.write(
                    EVENT_SNAPSHOT, before_phase="phase3_best",
                    attempt=attempts_run, score=list(score_after),
                )
                best_issues, best_score = list(blockers_after), score_after

            # Fixing one import can expose several previously unreachable CRUD
            # errors. A larger count is not evidence of regression. Continue on
            # new source states; only unchanged/repeated states count as stalls.
            state = (
                last_validated_revision, last_obligations_revision,
                tuple(i.message for i in _hard_blockers(blockers_after)),
            )
            # A round can move a run without writing a byte of source. Verifying
            # a checklist item or correcting a scenario discharges a blocker on
            # its own - that is exactly what ``_repair_obligations_revision``
            # measures - and a re-validation can clear findings an earlier
            # round's edits had already fixed. Run 053ydac9 (2026-09-20,
            # gpt-5.6-terra) spent its last round on test_api / read_file /
            # task_list, took 10 blockers to 4, wrote no source, and was stopped
            # as "this attempt cannot have moved anything" still holding 86 of
            # its 120 turns and $4.13 of its $5. So the question a stall guard
            # has to ask is not "did it write" but "did anything measurably
            # move": a better tree score, a changed source tree, or a
            # discharged obligation. Zero writes is no longer its own stop -
            # it is one no-progress round like any other.
            improved = score_after < prev_score
            # An attempt that changed nothing AND never reached for the editor
            # leaves the next prompt identical to this one, so the next round
            # really would be this round again: end it here, on the first
            # occurrence. An attempt whose edits were all REJECTED is the
            # opposite case - it is the commonest way a weak model spends a
            # round (a third of Qwen3-30B's edit calls fail), the rejections
            # reach the next prompt, and across the pre-guard corpus the round
            # after one wrote source 57% of the time and cut the blocker count
            # 20% (n=30). Those get the second round; at ~8 turns an attempt
            # that is ~1.2 turns per run, against the ~10 turns per run that
            # granting it to every barren round would cost.
            replay = edits == 0 and not source_changed and attempted_writes == 0
            if not improved and (
                (not source_changed and not obligations_changed)
                or state in seen_states
            ):
                no_progress_streak += 1
                budget_left = (
                    self.max_cost_usd is None
                    or self.client.usage.estimated_cost < self.max_cost_usd
                )
                if (replay or no_progress_streak >= _PHASE3_NO_PROGRESS_ROUNDS
                        or not budget_left):
                    logger.warning(
                        "Phase 3: Attempt %d made no progress (%d -> %d "
                        "blockers); ending fix loop (%d consecutive "
                        "no-progress round(s), writes=%s, attempted_writes=%s, "
                        "budget_left=%s).",
                        attempts_run, prev_blocker_count, len(blockers_after),
                        no_progress_streak, edits, attempted_writes, budget_left,
                    )
                    exit_reason = (
                        "cost budget exhausted" if not budget_left
                        else "replay (attempt never reached for the editor)" if replay
                        else "no-progress streak")
                    break
                logger.info(
                    "Phase 3: Attempt %d made no progress (%d -> %d "
                    "blockers); retrying once more (budget remains).",
                    attempts_run, prev_blocker_count, len(blockers_after),
                )
                # Re-attempt against the current state on the next round.
                current_blockers = blockers_after
                checkpoint_progress()
                continue

            # The tree changed but the score did not improve ON THE PREVIOUS
            # ROUND. ``state`` above can never catch this: it keys on a content
            # hash, so any edit at all - including a different useless one each
            # round - reads as a new state. Run mbzbzhq9 ran six attempts that
            # each ended at exactly 13 blockers, hit the 120-turn cap and
            # shipped a dead app.
            #
            # Measured against the previous round rather than the best ever, on
            # purpose: fixing one import legitimately exposes the errors behind
            # it, and 11 -> 45 -> 40 -> 35 is a repair converging, not a stall.
            # Oscillation (673hzu0z: 6-10-7-2-9-11-9-2-6-6-6) is the best-tree
            # snapshot's problem, not this guard's.
            if score_after >= prev_score:
                plateau_streak += 1
                if plateau_streak >= _PHASE3_PLATEAU_ROUNDS:
                    logger.warning(
                        "Phase 3: attempt %d is consecutive no-improvement round "
                        "%d (score %s, best %s); ending fix loop.",
                        attempts_run, plateau_streak, score_after, best_score,
                    )
                    exit_reason = "plateau"
                    break
            else:
                plateau_streak = 0

            # Progress this round: reset the stall counter and keep going.
            seen_states.add(state)
            no_progress_streak = 0
            prev_blocker_count = len(blockers_after)
            prev_score = score_after
            current_blockers = blockers_after
            checkpoint_progress()

        # We get here either by ending the loop early (no progress)
        # or by exhausting the attempt cap. Record whatever the final
        # state is so the recipe surfaces it.
        self._phase3_exit_reason = exit_reason
        if self._rollback_phase3_if_worse(best_issues, last_issues, source_ever_changed,
                                          entry_score=best_score):
            last_issues = list(self._validation_issues)
        else:
            self._validation_issues = self._with_model_contract(last_issues)
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

    @classmethod
    def _startup_blockers(cls, issues: list[ValidationIssue]) -> set[str]:
        """The startup-class blocker messages present in ``issues``."""
        return {i.message for i in issues
                if i.message.lower().startswith(cls._STARTUP_BLOCKER_PREFIXES)}

    def _phase3_tree_score(self, issues: list[ValidationIssue]) -> tuple[int, int, int, int]:
        """Rank one tree. LOWER is better; compare lexicographically.

        ``(boot broken, entities not confirmed created, actions not confirmed
        effective, hard blockers)`` - the runtime evidence, with the blocker
        count last because it is the weakest signal there is: across the 23
        runs of 2026-09-19 it correlated +0.21 with whether the delivered app
        worked, i.e. the wrong sign at noise magnitude. Boot dominates on
        purpose, so a tree that starts can never be discarded for one that
        does not.

        The middle two come from the probe's own per-entity and per-action
        records where it measured THIS tree. Counting issue strings instead
        was a re-derivation of facts we already had, and it counts what was
        rendered: an action the probe silently confirmed effective and one it
        never reached scored the same. Strings remain the fallback for a tree
        no probe measured (no backend, or the probe did not run).
        """
        boot_broken = int(bool(self._startup_blockers(issues)))
        facts = getattr(self, "_runtime_probe_facts", None)
        measured = None
        if facts and facts[0] == self._workspace_revision():
            entities = actions = 0
            for backend in facts[1]:
                if backend.get("boot") != "ok":
                    continue
                measured = True
                for entry in (backend.get("entities") or {}).values():
                    entities += int(entry.get("verdict") != "created")
                for call in backend.get("action_calls") or []:
                    actions += int(call.get("verdict") != "effective")
        if measured is None:
            entities = actions = 0
            for issue in issues:
                message = issue.message.lower()
                if "create contract:" in message or "create unverified:" in message:
                    entities += 1
                elif "action call:" in message or "action unverified:" in message:
                    actions += 1
        return boot_broken, entities, actions, len(_hard_blockers(issues))

    def _runtime_gate_finding(self) -> str | None:
        """The gate's refusal text, or None when the app is clear to ship.

        Detection alone is provably not enough: run mbzbzhq9 held its fatal
        ``mapper config:`` blocker in top-priority position for six attempts
        and shipped anyway. So this is a gate, not a finding - when it returns
        text the run cannot report itself complete, and the text names the
        runtime failure rather than burying it among forty lint lines.

        Silent when there is nothing to boot (a Qiskit or BAF run has no
        backend to probe), and silent when the probe booted the app and every
        entity create and modelled action came back settled.

        Also silent on an UNSETTLED result, and that restraint is measured. A
        guessed fixture a business rule legitimately refuses is the normal
        answer from a correct app: rescoring the 2026-09-19 batch against a
        corrected acceptance probe (``verification/rescore_corrected_probe.json``)
        shows 308z4wo2 passing 11/11 while every one of its create routes came
        back ``create unverified:``, and dp3trml9 - the accepted artifact -
        answering the probe's ReservedRoom payload with a correct 409. Those
        routes stay reported through their own ``runtime unverified:`` blocker;
        they are not grounds for the gate to refuse a working application.
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
        if verdict is None and not any(
            i.message.lower().startswith("runtime unverified:")
            for i in self._validation_issues
        ):
            return (
                f"{self._RUNTIME_GATE_PREFIX} the delivered application has no runtime "
                "evidence at all - the boot-and-create probe did not produce a "
                "verdict for this source revision and no test_api workflow was run, "
                "so this run is NOT complete. An unrun check is not a passing check."
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
        except Exception:
            logger.debug("Runtime gate evaluation failed", exc_info=True)
            return
        if not finding:
            return
        if any(i.message.startswith(self._RUNTIME_GATE_PREFIX)
               for i in self._validation_issues):
            return
        logger.warning("Phase 3 runtime gate refused completion: %s", finding[:300])
        self._validation_issues.append(ValidationIssue("blocker", finding))
        self._trace.write(
            EVENT_VALIDATION_ISSUE, phase="phase3_runtime_gate", message=finding[:1000],
        )

    def _rollback_phase3_if_worse(
        self, entry_blockers: list[ValidationIssue],
        final_issues: list[ValidationIssue],
        source_changed: bool,
        entry_score: tuple[int, int, int, int] | None = None,
    ) -> bool:
        """Ship the pre-Phase-3 tree when repair ended worse than it began.

        Three conditions, and all are needed. The repair must have actually
        written something: a blocker that appears while nothing was edited is
        newly-exposed truth or judge variance, and rolling back would hide it
        (there would also be nothing to undo). A rising count *during* the
        loop is expected, since fixing an import exposes the errors behind it,
        so only the final state counts. And only hard blockers count, because
        two judge passes on one app returned 12 then 22 missing requirements.

        Run trilraak entered Phase 3 with 11 blockers; attempt 2 wrote six
        files, added an association table using ``Table`` without importing
        it, took the count to 45, and the run shipped that: ``sql_alchemy.py``
        no longer imported, so every router that star-imports it was dead.
        ``_restore_snapshot`` existed and was unit-tested, but nothing in
        production ever called it.

        The code is reverted, the findings are not: what the repaired tree
        revealed is recorded, so a real defect a partial fix exposed does not
        become invisible again.
        """
        if not source_changed:
            return False
        final_blockers = [i for i in final_issues if i.severity == "blocker"]
        entry_hard = len(_hard_blockers(entry_blockers))
        final_hard = len(_hard_blockers(final_blockers))
        # Ranked on runtime evidence first (boot, entities created, actions
        # callable) and only then on the hard count - see _phase3_tree_score.
        # ``entry_blockers`` is the BEST state reached, which is what the
        # snapshot now holds, not necessarily the Phase 3 entry state.
        #
        # The score MUST be the one computed while that tree was the tree on
        # disk. _phase3_tree_score reads ``_runtime_probe_facts``, and by the
        # time we get here those facts describe the FINAL tree: recomputing
        # the entry score here gave both trees the same middle two components
        # (entities not created, actions not effective), they cancelled, and
        # the comparison collapsed to (boot, hard count) - the component this
        # ranking exists to demote. A repair that broke three entity creates
        # and removed one hard blocker then scored as an improvement and
        # shipped. Callers inside the repair loop pass ``best_score``;
        # ``None`` means "no probe has measured a different tree since", which
        # only holds for a direct call.
        if entry_score is None:
            entry_score = self._phase3_tree_score(entry_blockers)
        final_score = self._phase3_tree_score(final_blockers)
        # A count cannot see a TRADE. Run mbzbzhq9 held 13 blockers flat across
        # six attempts while swapping a hard blocker for a broken ORM mapper,
        # so "not more than we started with" was true and the run shipped an
        # app whose every endpoint returned 500. Introducing a blocker that
        # stops the app starting is never an acceptable trade, at any count.
        broke_startup = (self._startup_blockers(final_blockers)
                         - self._startup_blockers(entry_blockers))
        if final_score <= entry_score and not broke_startup:
            return False
        if broke_startup:
            logger.warning(
                "Phase 3 introduced %d blocker(s) that stop the app starting: %s",
                len(broke_startup), "; ".join(sorted(broke_startup))[:300],
            )
        logger.warning(
            "Phase 3 ended worse than the best tree it reached (%s -> %s; "
            "%d -> %d hard blockers); restoring that tree.",
            entry_score, final_score, entry_hard, final_hard,
        )
        if not self._restore_snapshot():
            logger.error(
                "Phase 3 regressed but the snapshot could not be restored; "
                "keeping the repaired tree and reporting it as it stands.",
            )
            self._validation_issues = self._with_model_contract(final_issues)
            return False
        self._phase3_rolled_back = True
        # The tree went back; the checklist did not. Anything whose verifier
        # no longer passes was undone by this restore and must stop reporting
        # itself complete.
        reopened = self.executor.reopen_unverifiable_tasks()
        if reopened:
            logger.warning(
                "Phase 3 rollback discarded the implementation of %d checklist "
                "item(s), now reopened: %s", len(reopened), reopened,
            )
        restored = self._collect_validation_issues()
        # Only what the DISCARDED tree added. Printing the whole final set
        # listed findings that describe the tree that just shipped - the one
        # text a human reads to decide whether the rollback hid a real defect,
        # naming defects it did not hide.
        kept = {i.message for i in entry_blockers}
        only_in_discarded = sorted({i.message for i in final_blockers} - kept)
        discarded = only_in_discarded[:10]
        more = (f" (+{len(only_in_discarded) - len(discarded)} more not listed)"
                if len(only_in_discarded) > len(discarded) else "")
        seen_only = ("; ".join(discarded) + more if discarded
                     else "none - every finding in the discarded tree is also in this one")
        undone = (f" The restore also undid completed work: {len(reopened)} checklist "
                  f"item(s) verified during the repair are open again." if reopened else "")
        self._validation_issues = self._with_model_contract(restored) + [_classify_issue(
            "validation: the Phase 3 repair was rolled back - it ended with "
            f"{final_hard} hard blockers against {entry_hard} on entry, so the "
            f"pre-repair output is what ships.{undone} Findings seen only in the "
            "discarded tree (they may still be real): " + seen_only
        )]
        restored_blockers = [i for i in restored if i.severity == "blocker"]
        self._trace.write(
            EVENT_ROLLBACK, phase="phase3",
            # All three are HARD counts so they compare; the totals include
            # ledger/checklist verdicts the decision deliberately ignores.
            hard_blockers_on_entry=entry_hard,
            hard_blockers_after_repair=final_hard,
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

        Factored out of ``_run_phase3_validation`` so the outer
        toolchain-fix iteration cap can call it more than once. Each
        invocation builds a fresh prompt (so the LLM doesn't see
        stale context from a previous attempt) and exits when the LLM
        emits ``end_turn`` or hits the per-attempt turn budget - except
        that an attempt with no successful edit is re-prompted once,
        with ``modify_file`` forced where the client supports
        ``tool_choice`` and a reminder in the message either way.
        Tool calls run through ``_execute_tool_blocks`` so they are
        recorded like Phase 2's.

        The prompt always reproduces the current blocker list verbatim
        — including any ``tsc [...]:`` / ``cargo [...]:`` /
        ``kotlinc [...]:`` lines. When toolchain blockers are present,
        a high-salience reminder is appended instructing the LLM to
        re-run the toolchain via ``run_command`` after each edit, so
        the model verifies its own fixes instead of stopping at the
        first plausible-looking change.
        """
        if not blockers:
            return 0

        # Citation/coverage obligations and model-authored tests can be wrong.
        # A verified existing implementation or corrected test need not mutate
        # production code. Keep forced edits only for concrete code defects.
        verification_only = all(issue.message.startswith((
            "requirement unverified:", "task unverified:", "api scenario:",
            "runtime unverified:", "create unverified:", "verification setup:",
        )) for issue in blockers)

        # Only the toolchain half is needed: it drives the re-run reminder
        # below. Every blocker, toolchain or not, is listed in the prompt.
        toolchain_blockers = [
            i for i in blockers
            if i.message.startswith(("tsc [", "cargo [", "kotlinc ["))
        ]

        prompt_parts: list[str] = []
        if self._instructions:
            prompt_parts.extend([
                "## Original request (the authority for required behavior)",
                self._instructions, "",
            ])
        if self._phase2_inspection_handoff:
            prompt_parts.extend(["## Prior inspection handoff", self._phase2_inspection_handoff, ""])
        if self.domain_model is not None:
            prompt_parts.extend([
                "## Domain model and conversion losses",
                "conversion_issues are NOT implemented constraints. Recover their "
                "intended behavior using actual relationship names; the original "
                "request takes precedence over conflicting model expressions.",
                json.dumps(serialize_domain_model(self.domain_model), ensure_ascii=False), "",
            ])
        requirements = self._requirements_for_validation()
        if requirements:
            prompt_parts.extend([
                "## Requirements to verify (including conversion recovery)",
                _requirements_ledger.render_requirements(requirements), "",
            ])
        prompt_parts.extend([
            "## Current files and symbols (use these paths; do not invent models/ or schemas/ folders)",
            build_inventory(self.output_dir, self.domain_model, self._generator_used or "existing workspace"),
            "", "## Actual action handlers",
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
        if is_first_attempt:
            prompt_parts.append(
                "Post-generation validation found these unresolved issues. "
                "Distinguish observed execution failures from unverified evidence "
                "and model judgments; resolve each against the original specification:"
            )
        else:
            prompt_parts.append(
                "After your previous fixes, these BLOCKER issues "
                "still remain. Fix every one:"
            )

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
        prompt_parts.extend(f"- {i.message}" for i in sorted(blockers, key=self._repair_priority))

        # Show the offending lines. Measured 2026-09-18 on the model that had
        # just failed here: with only "file line N" it spends a turn on
        # read_file (6/6); with the excerpt it calls modify_file immediately
        # (6/6). A BOUNDED window is the point -- SWE-agent's ablation scores
        # a 100-line window above the whole file (18.0 vs 12.7 on SWE-bench
        # Lite), so this never pastes an entire file.
        excerpts = self._excerpts_for(blockers)
        if excerpts:
            prompt_parts.append("")
            prompt_parts.append(
                "The offending lines, verbatim from disk. Quote old_text from "
                "here exactly — never abbreviate with '...':"
            )
            prompt_parts.extend(excerpts)

        prompt_parts.append("")
        prompt_parts.append(
            "Resolve authorized dependency setup through install_dependencies, then "
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
            "An evidence/citation error alone does not justify changing correct code. "
            "Do NOT touch anything unrelated."
        )

        # When the blockers include toolchain errors, instruct the LLM
        # to drive the toolchain itself with run_command — that's the
        # only way to know whether a fix actually compiles, and it's
        # the bench's per-project compile-pass criterion. We do this
        # as additional text in the same user turn (vs. a separate
        # message) so the LLM sees the request as part of the brief.
        if toolchain_blockers:
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
        system = (
            "You are fixing validation errors in generated code. "
            "Fix each issue concisely. Call validate_app to recheck startup and "
            "data entry after a coherent repair. When shell tools are unavailable, "
            "validate_app and test_api are the supported verification tools. Use test_api "
            "for specification-based workflow assertions and invalid inputs. When shell tools "
            "are available and the report contains "
            "toolchain errors (tsc / cargo / kotlinc), verify each fix by "
            "re-running that toolchain with run_command; the diff alone does "
            "not show that it compiles."
        )
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
        nudged = False            # the one re-prompt an edit-less attempt gets
        force_next: str | None = None
        turn_cap = _PHASE3_FIX_TURNS
        turn = 0
        while True:
            stop_reason = self._phase3_stop_requested()
            if stop_reason:
                logger.warning("Phase 3: %s", stop_reason)
                return edits
            if turn >= turn_cap:
                if edits or nudged or verification_only:
                    return edits
                # Read until the cap and wrote nothing: one more turn, and
                # it has to be the edit.
                nudged, force_next = True, "modify_file"
                turn_cap += 1
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": _PHASE3_NO_EDIT_REMINDER}],
                })
            turn += 1
            self.total_turns += 1
            # Recovery is shared with Phase 2. Do not force the failing text
            # strategy again after the executor requested a fresh read/range edit.
            force = self._force_tool_next or force_next
            self._force_tool_next, force_next = None, None
            request_messages = without_rejected_edit_drafts(messages)
            try:
                if force and self._client_supports_structured_chat():
                    response = self.client.chat(
                        system=system, messages=request_messages, tools=self.tools,
                        force_tool=force,
                    )
                else:
                    response = self.client.chat(
                        system=system, messages=request_messages, tools=self.tools,
                    )
            except InvalidApiKeyError:
                # Same rule as Phase 2: an auth failure must PROPAGATE so the
                # runner reports INVALID_KEY. Swallowed here it read as
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
                self._phase3_interrupted = True
                self._phase3_interrupt_detail = f"provider call failed: {type(exc).__name__}"
                return edits
            # A stop can arrive while a provider request is in flight. Do not
            # apply its returned mutations after cancellation. The final allowed
            # turn may still execute its tools; it is not a new provider call.
            if self._phase3_stop_requested(check_turn_budget=False):
                return edits
            if response["stop_reason"] == "end_turn":
                if edits or nudged or verification_only:
                    return edits
                # Ended in prose with nothing written. Say so once, force the
                # edit where tool_choice is honoured, and let the reminder
                # carry it where the gateway ignores tool_choice.
                nudged, force_next = True, "modify_file"
                if response["content"]:
                    messages.append({"role": "assistant", "content": response["content"]})
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": _PHASE3_NO_EDIT_REMINDER}],
                })
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
            tool_blocks = [
                block for block in response["content"]
                if hasattr(block, "type") and block.type == "tool_use"
                and getattr(block, "name", None)
            ]
            # Through the Phase 2 path, so the trace, recipe and sidecar see
            # these calls; run 7f918e11's ten turns went through the raw
            # executor and left no record of what they were.
            logged_before = len(self.tool_calls_log)
            tool_results = self._execute_tool_blocks(tool_blocks, self.total_turns - 1)
            edits += sum(
                1 for entry in self.tool_calls_log[logged_before:]
                if entry["tool"] in _WRITE_TOOLS_ON_RECORD and entry["success"]
            )
            messages.append({"role": "user", "content": tool_results})
            # The same streak/repeat guards Phase 2 and the fix cycle get.
            # Omitting them here left the bounded repair loop running on
            # _is_stuck alone, on a tighter budget than either.
            if self._apply_edit_loop_guards(messages, where="phase 3 repair"):
                # Falling out of the loop without this returned ``None``, so
                # an attempt that DID write reported zero writes to the outer
                # cycle, to the trace and to the log line that says the
                # attempt changed nothing.
                return edits
            self._save_phase3_checkpoint()

    def _excerpts_for(
        self, blockers: list[ValidationIssue], context: int = 5, limit: int = 3
    ) -> list[str]:
        """Numbered source windows around each blocker that names file+line."""
        out: list[str] = []
        seen: set[tuple[str, int]] = set()
        for issue in blockers:
            match = self._FILE_LINE_RE.search(issue.message)
            if not match:
                continue
            rel, line_no = match.group(1), int(match.group(2))
            if (rel, line_no) in seen or len(seen) >= limit:
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
        return _toolchain_commands_for(toolchain_blockers)

    def _build_toolchain_reminder(self, toolchain_blockers: list[ValidationIssue]) -> str:
        """Delegate to the toolchain module."""
        return _build_toolchain_reminder(toolchain_blockers)

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
        if message.startswith(("data contract:", "undefined name:", "runtime unverified:")):
            return 1, message
        if message.startswith(("requirement", "task unverified:")):
            return 3, message
        return 2, message

    def _create_snapshot(self) -> None:
        """Snapshot the output directory as the rollback target.

        Taken once before Phase 3 and again after any repair attempt that
        reaches a strictly better tree, so the snapshot always holds the BEST
        state the run has reached rather than its first one.

        Built in a staging directory and swapped in by rename: deleting the
        existing snapshot first and copying after leaves the run with no
        rollback target at all if the copy dies partway, and this now runs
        several times per Phase 3 rather than once.
        """
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        staging_path = os.path.join(self.output_dir, _SNAPSHOT_STAGING_DIR)
        try:
            if os.path.exists(staging_path):
                shutil.rmtree(staging_path)

            # Copy everything except the snapshot dir and the run bookkeeping
            # a rollback must never revert (see _ROLLBACK_PRESERVED).
            for item in os.listdir(self.output_dir):
                if item in _ROLLBACK_PRESERVED:
                    continue
                src = os.path.join(self.output_dir, item)
                dst = os.path.join(staging_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                        *_SNAPSHOT_IGNORED_DIRS))
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            os.makedirs(staging_path, exist_ok=True)

            if os.path.exists(snapshot_path):
                shutil.rmtree(snapshot_path)
            os.replace(staging_path, snapshot_path)
            logger.info("Snapshot created at %s", snapshot_path)
        except Exception as e:
            logger.warning("Failed to create snapshot: %s", e)
            shutil.rmtree(staging_path, ignore_errors=True)

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
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
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

        shutil.rmtree(discard_path, ignore_errors=True)
        logger.info("Restored from snapshot")
        return True

    def _remove_snapshot(self) -> None:
        """Clean up the snapshot directory (and any staging left by a failure)."""
        for name in (_SNAPSHOT_DIR, _SNAPSHOT_STAGING_DIR):
            path = os.path.join(self.output_dir, name)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass

