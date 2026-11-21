"""The modify run: re-enter a finished workspace and change it in place.

A modify differs from a fresh run in what it must NOT do. The tree already
holds a working app, so Phase 1 is skipped, the recipe's generator files are
seeded back into context rather than regenerated, and a fix-target request is
scoped to the findings it names instead of the whole issue list. The model
deltas exist because a modify often implies classes the original diagram never
had, and Phase 2's inventory is only as good as the model behind it.

Mixed into ``LLMOrchestrator``; all run state is reached through ``self``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from collections import Counter

from besser.spec_driven_agent.agent.prompt_builder import build_inventory
from besser.spec_driven_agent.agent.tool_executor import ends_shell_session
from besser.spec_driven_agent.errors import EmptyInstructionsError
from besser.spec_driven_agent.execution.deadline import under_deadline
from besser.spec_driven_agent.pipeline.phase3_repair import _RUNTIME_TARGET_RE, _finding_text
from besser.spec_driven_agent.pipeline.regeneration import MAX_REASON_CHARS, regenerate
from besser.spec_driven_agent.planning.model_change import (
    FULL_MODEL,
    compare_with_base,
    model_change_from_dict,
)
from besser.spec_driven_agent.planning.model_rules import (
    is_dropped_by_model_change,
    mark_rules_broken_by_model_change,
)
from besser.spec_driven_agent.planning.requirements_ledger import (
    latest_user_message,
    ledger_issues,
    original_request,
    previous_run_verdicts,
    source_hashes,
)
from besser.spec_driven_agent.planning.user_request import user_request
from besser.spec_driven_agent.planning.specification import validate_specification
from besser.spec_driven_agent.planning.target import (
    MODIFY_RULES,
    STACK_RULES,
    TARGET_RULES,
    Target,
)
from besser.spec_driven_agent.providers.llm_client import MODIFY_MAX_TOKENS
from besser.spec_driven_agent.run_recap import (
    NO_REQUEST,
    load_recipe,
    MAX_RUNS,
    RECORD_NOT_INSTRUCTIONS,
    next_run_notes,
    one_line,
    recorded_requests,
    repo_notes,
)
from besser.spec_driven_agent.state.api_workflows import load_workflows, workflow_entries
from besser.spec_driven_agent.state.trace_model import generator_output_record
from besser.spec_driven_agent.state.tracing import (
    EVENT_PHASE_ENTER,
    EVENT_PHASE_EXIT,
    EVENT_REGENERATION,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_SNAPSHOT,
    EVENT_VALIDATION_ISSUE,
)
from besser.spec_driven_agent.validation.issues import (
    OPEN_SINCE_PREVIOUS,
    REGRESSION_TO_CONFIRM,
    ValidationIssue,
    _check_did_not_run,
    _classify_issue,
    could_not_run,
    hidden_families,
    is_completion_issue,
    is_not_confirmed,
    unresolved_defects,
)
from besser.spec_driven_agent.validation.model_conformance import model_conformance_issues
from besser.spec_driven_agent.validation.user_wording import user_sentence
from besser.utilities.workspace_fs import open_plain

logger = logging.getLogger(__name__)


# Never promoted to the reported failure on a fix run, and never held against it by the
# gate. A user's report can corroborate an OBSERVATION of the running app (a guessed create
# or action the app refused: ``create``/``action``/``runtime unverified:``), never "no
# conclusion about the code" (a citation not located), "the check did not run", or a
# checklist item nothing closed. The first entry is ``_FIX_TARGET_PREFIX``. Leaving the
# screen check out is a policy choice: its findings stay warnings until calibrated.
_NEVER_PROMOTED = (
    "reported-failure: ",
    "requirement unverified:",
    "validation unverified:",
    "task unverified:",
    "screen:",
    "screen warning:",
    "screens: not run",
)

# How a seed run's forwarded blocker opens this run's issues.
_SEED_FORWARD_PREFIX = "Unresolved from the previous run: "
# A blocker of these kinds that the previous version did not have is a regression to confirm:
# the app no longer starts, creates a record, or builds.
_REGRESSION_PREFIXES = (
    "application startup:", "mapper config:", "missing module:",
    "create contract:", "read contract:", "action call:", "tsc [", "frontend build [",
)


class ModifyRunMixin:
    """modify and friends; see the module docstring."""

    @ends_shell_session
    @under_deadline
    def modify(self, instructions: str) -> str:
        """Edit a seeded workspace in place instead of rebuilding it.

        Modelled on ``run()`` (NOT ``resume()``): there is no checkpoint
        load, no fingerprint gate, and no crash-recovery replay. The runner
        has already copied a previous run's generated files into
        ``self.output_dir`` (stripping that run's checkpoint + snapshot but
        KEEPING its ``.besser_recipe.json``). This method:

          * SKIPS Phase 1 entirely — the deterministic generator would
            overwrite the customised files the user wants to keep.
          * Re-derives the inventory + generator-file tags from the seed
            so Phase 2 sees the real on-disk state.
          * Drives Phase 2 with ``modify_mode=True`` so the system prompt
            biases the LLM toward the smallest surgical change.
          * Runs Phase 1.5 validation, Phase 3 validation, and the recipe
            save exactly like ``run()``; drops the checkpoint on clean exit.

        The user may also have edited the model between runs;
        ``self.domain_model`` reflects that. The inventory and gap analyzer
        surface the model's classes vs. the files on disk, so Phase 2
        authors any deltas via write_file / modify_file (no scaffold merge).

        Returns the path to ``self.output_dir``.
        """
        if not instructions or not instructions.strip():
            raise EmptyInstructionsError("Instructions cannot be empty")
        validate_specification(instructions)
        self._instructions = instructions

        self._start_time = time.monotonic()
        # The previous version, before this run's model sync and regeneration change it.
        self._seed_source_hashes = source_hashes(self.output_dir)
        self._previous_verdicts = self._load_previous_verdicts()
        # Before the model-sync below adds classes: a resume re-checks it
        # against the model in the request.
        self._project_fingerprint = self._compute_project_fingerprint(instructions)
        self._modify_mode = True
        # Edit-first guardrail: in a modify run, whole-file rewrites of
        # existing files are rejected until targeted edits were tried —
        # rewrites are where modify-run regressions come from.
        self.executor.enable_modify_guard()

        # Give the modify/fix path the wider per-call output ceiling: a
        # single-turn file rewrite here overruns the client's default cap
        # and truncates mid-file. Set once, before Phase 2. Cost/runtime
        # caps are untouched; only the response-size ceiling is raised.
        self._apply_modify_budget()

        # Re-hydrate generator-file tags + the seed's generator name from
        # the copied recipe BEFORE building the inventory (which needs a
        # generator name for its framing line).
        self._seed_generator_files_from_recipe()
        self._seed_workflows = load_workflows(self.output_dir)
        # One routing call: does the request change the target, does it report a failure.
        self._route_modify_request()
        # Frame the run as editing an existing project. When the seed came
        # from a deterministic generator we adopt that name (accurate — the
        # base + prior LLM edits descend from it) so gap analysis, the
        # scaffold-snapshot inlining, and the saved recipe all line up.
        self._generator_used = self._seed_generator_used
        self.executor.set_scaffold_family(
            self._scaffold_family(), self._instructions)

        # How the user's model relates to the one the seed app was built from,
        # read before the model-sync below adds classes of its own.
        self._model_change = compare_with_base(self.output_dir, self.domain_model)
        if self._model_change is not None and not self._model_change.base:
            logger.warning(
                "The base run's recipe records only class names (it predates "
                "user_model): verdict %s, no model change list for this run",
                self._model_change.verdict)

        # -- Model-sync: derive + apply class-diagram deltas implied by the
        # instruction BEFORE building the inventory, so a genuinely new
        # domain entity (e.g. a ``User`` class for "add authentication")
        # flows into Phase 2's inventory AND the updated model reaches the
        # push path. Fully guarded (see the method): an empty/failed/bad
        # delta leaves the run proceeding unchanged — no model
        # change, no crash. Scoped to modify() only; run()/_run_phase1
        # never invoke it, so from-scratch output is byte-identical.
        self._derive_and_apply_model_deltas(instructions)
        self._mark_rules_broken_by_model_change()
        # Before the inventory, so Phase 2 sees the regenerated files.
        self._regenerate_generated_files()

        self._inventory = build_inventory(
            self.output_dir,
            self.domain_model,
            self._seed_generator_used or "existing project",
        )
        # Session memory: open the run knowing what was asked and changed
        # in every previous run on this app (recipe history), instead of
        # rediscovering it from file contents.
        session_recap = self._render_seed_history()
        if session_recap:
            self._inventory = f"{self._inventory}\n{session_recap}"
        notes = repo_notes(self.output_dir)
        if notes:
            self._inventory = f"{self._inventory}\n\n{notes}"
        model_note = self._render_model_change()
        regeneration_note = self._regeneration.note() if self._regeneration else ""
        if model_note:
            self._inventory = (
                f"{self._inventory}\n\n## Model change since the last run\n{model_note}")
        if regeneration_note:
            self._inventory = f"{self._inventory}\n\n## Regenerated files\n{regeneration_note}"

        # -- Fix/modify success gate: parse the user-reported failure ------
        # A modify run seeded by a reported error (traceback / broken
        # endpoint) makes that failure the run's success criterion. Detect
        # + parse it once here; every downstream branch is gated on
        # ``_is_fix_run`` so run()/resume() stay byte-identical.
        self._detect_fix_target(instructions)

        self._trace.write(
            EVENT_RUN_START,
            mode="modify",
            # bounded: trace display only, never a decision input
            instructions=original_request(instructions)[:500],
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
            max_turns=self.max_turns,
            fix_run=self._is_fix_run,
            fix_target=(self._fix_target.descriptor if self._fix_target else None),
            model_change=(self._model_change.to_dict() if self._model_change else None),
            model_sync_error=self._model_sync_error,
        )

        # -- Phase 0: the model itself ------------------------------------
        # Runs after the model-sync deltas above, so it checks the model this
        # run will actually build against. A user who edited the diagram
        # between runs can introduce a mandatory creation cycle here.
        self._collect_model_contract_issues()

        # -- Phase 1.5: Validate the seeded output (no Phase 1 run) --------
        phase1_issues = self._validate_phase1_output()
        for issue in phase1_issues:
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="phase1_5", message=issue)

        # -- Phase 2: LLM edits the seeded files in place ------------------
        # Forward the seed run's unresolved blockers: they ride along
        # with the fresh Phase 1.5 findings so the modify run pays down
        # the known debt instead of preserving it forever.
        if self._seed_unresolved_issues:
            logger.info(
                "Forwarding %d unresolved blocker(s) from the seed run",
                len(self._seed_unresolved_issues),
            )
            for issue in self._seed_unresolved_issues:
                self._trace.write(
                    EVENT_VALIDATION_ISSUE, phase="seed_forward", message=issue,
                )
        self._trace.write(EVENT_PHASE_ENTER, phase="phase2_modify")
        self._run_phase2(instructions, extra_issues=self._modify_scoped_issues(phase1_issues))
        self._trace.write(
            EVENT_PHASE_EXIT, phase="phase2_modify", turns=self.total_turns,
            stop_reason=self._phase2_stop_reason,
            stop_detail=self._phase2_stop_detail,
        )

        # -- Snapshot BEFORE Phase 3 (preserves all Phase 2 edits) --------
        self._create_snapshot()
        self._trace.write(EVENT_SNAPSHOT, before_phase="phase3")

        # -- Phase 3: Validate & fix --------------------------------------
        if self._phase2_exited_cleanly or self._phase2_stop_reason == "validation_required":
            self._save_phase3_checkpoint()
        self._trace.write(EVENT_PHASE_ENTER, phase="phase3")
        self._run_phase3_validation()
        self._settle_open_setup()
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="phase3",
            unresolved_blockers=sum(1 for i in self._validation_issues if is_completion_issue(i)),
            not_confirmed=sum(1 for i in self._validation_issues if is_not_confirmed(i)),
            exit_reason=self._phase3_exit_reason,
        )

        # -- Against the previous version, and the fix/modify success gate --
        self._modify_end_checks()

        elapsed = time.monotonic() - self._start_time
        logger.info(
            "LLM modify finished: %d turns, %.1fs, %d tool calls, "
            "seed_generator=%s, compactions=%d",
            self.total_turns, elapsed, len(self.tool_calls_log),
            self._seed_generator_used or "none", self._compaction_count,
        )
        logger.info("Cost: %s", self.client.usage)

        self._save_recipe(instructions, elapsed)
        self._remove_snapshot()

        self._finish_checkpoint()

        self._trace.write(
            EVENT_RUN_END,
            mode="modify",
            elapsed_seconds=round(elapsed, 2),
            total_turns=self.total_turns,
            estimated_cost_usd=float(self.client.usage.estimated_cost),
            validation_issues=len(self._validation_issues),
        )
        return self.output_dir

    def _modify_scoped_issues(self, phase1_issues: list[str]) -> list[str]:
        """The issues a modify run's Phase 2 opens with; its resume rebuilds them."""
        # The tool's OWN structural findings that match the reported target,
        # as explicit fix instructions, so the LLM acts on the concrete defect
        # (e.g. "the Watchlist create form is not wired") instead of only the
        # user's paraphrase. Empty on a non-fix modify run.
        fix_seed_issues = self._fix_run_scoped_issues()
        for issue in fix_seed_issues:
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="fix_target_seed", message=issue)
        # The model change leads the issues, after the reported failure's
        # headline in a fix run. A resume has no regeneration note: the
        # restored inventory carries it.
        regeneration_note = self._regeneration.note() if self._regeneration else ""
        leading = [note for note in (self._render_model_change(), regeneration_note) if note]
        if leading and fix_seed_issues:
            leading = fix_seed_issues[:1] + leading
            fix_seed_issues = fix_seed_issues[1:]
        return leading + phase1_issues + self._seed_unresolved_issues + fix_seed_issues

    def _modify_opening_issues(self, issues: list[str]) -> list[str]:
        """Phase 2's opening issues of a modify run, before validate_app adds its blockers on
        the tree Phase 2 starts from: each fact once.

        The checks validate_app runs, unclipped (the probe's result is cached for its call):
        a forwarded seed blocker they find again is left out. On a fresh run (not a resume)
        they and the seed recipe's findings are the previous version's (``_seed_baseline``).
        """
        opening = [*self._collect_execution_issues(), *self._static_frontend_issues()]
        if self._seed_baseline is None and self._resume_messages is None:
            self._seed_baseline = [*opening, *self._seed_recorded_issues]
        current = {self._seed_identity(issue) for issue in opening
                   if _classify_issue(issue).severity == "blocker"}
        return list(dict.fromkeys(
            issue for issue in issues
            if not (issue.startswith(_SEED_FORWARD_PREFIX)
                    and self._seed_identity(issue[len(_SEED_FORWARD_PREFIX):]) in current)))

    def _seed_identity(self, message: str) -> tuple:
        """What a finding tests, as Phase 3 attempts compare them (``_blocker_key``), and for a
        runtime finding its error too: a new error on the same route is a new finding."""
        return (self._blocker_key(ValidationIssue("blocker", message)),
                _finding_text(message) if _RUNTIME_TARGET_RE.match(message) else None)

    def _seed_defects(self) -> list[str]:
        """The previous version's observed defects: blockers of ``_seed_baseline`` that are no unknown."""
        return unresolved_defects([message for message in self._seed_baseline or []
                                   if _classify_issue(message).severity == "blocker"])

    def _label_seed_findings(self, issues: list[ValidationIssue]) -> list[ValidationIssue]:
        """A modify run's defect the previous version already had, matched by identity (not
        text), becomes ``open since the previous version:``: listed, not a blocker, not
        repaired. A finding that matches a fix run's reported failure keeps its severity."""
        old = Counter(self._seed_identity(message) for message in self._seed_defects())
        if not self._modify_mode or not old:
            return issues
        from besser.spec_driven_agent.planning.fix_target import finding_matches_target

        labelled = []
        for issue in issues:
            if (is_completion_issue(issue) and unresolved_defects([issue])
                    and not issue.message.startswith(self._FIX_TARGET_PREFIX)
                    and not (self._is_fix_run and self._fix_target is not None
                             and finding_matches_target(issue.message, self._fix_target))):
                key = self._seed_identity(issue.message)
                if old[key] > 0:
                    old[key] -= 1
                    labelled.append(ValidationIssue("warning", OPEN_SINCE_PREVIOUS + issue.message))
                    continue
            labelled.append(issue)
        return labelled

    def _note_seed_regressions(self) -> None:
        """The hard blockers (no start, no create, no build) the run ends with that the
        previous version did not have, where that check could run on it: regressions to
        confirm (``_seed_regressions``)."""
        if self._seed_baseline is None:
            return
        old = {self._seed_identity(message) for message in self._seed_defects()}
        hidden = hidden_families(self._seed_baseline)
        self._seed_regressions = [
            issue.message for issue in self._validation_issues
            if is_completion_issue(issue) and issue.message.lower().startswith(_REGRESSION_PREFIXES)
            and self._seed_identity(issue.message) not in old and not could_not_run(issue.message, hidden)]
        for message in self._seed_regressions:
            logger.warning("Modify run: new since the previous version, a regression to confirm: %s",
                           message[:300])
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="seed_regression", message=message[:1000])

    def _replay_seed_workflows(self) -> None:
        """Run each workflow that passed on the previous version once on the delivered tree,
        unless this run kept one under its id. One that fails now is a regression to confirm,
        never a blocker: the change may have meant it. Those that pass, and those no check
        could run, are kept for the next run (``_carried_seed_workflows``)."""
        if not self._seed_workflows:
            return
        from besser.spec_driven_agent.validation.api_probe import DID_NOT_RUN_BOOTS, probe_api_scenario

        own = {record.get("scenario_id") for record in self._api_scenarios.values()}
        for workflow in self._seed_workflows:
            if workflow["scenario_id"] in own or workflow["status"] != "passed":
                continue
            if not self.enable_import_smoke_check or not self._checks_allowed():
                self._carried_seed_workflows.append(workflow)
                continue
            scenario = workflow["scenario"]
            report = probe_api_scenario(self.output_dir, scenario["requests"], backend=scenario.get("backend"))
            if report.get("status") == "passed" or report.get("boot") in DID_NOT_RUN_BOOTS:
                self._carried_seed_workflows.append(workflow)
                continue
            detail = report.get("error") or json.dumps({
                "assertion_failures": report.get("assertion_failures", []),
                "request_errors": [{"path": response.get("path"), "error": response["error"]}
                                   for response in report.get("responses", []) if response.get("error")],
            })
            message = (f"{REGRESSION_TO_CONFIRM}api scenario {workflow['scenario_id']} passed on the previous "
                       f"version and fails now: {detail[:1500]}")
            logger.warning("Modify run: %s", message[:300])
            self._validation_issues.append(ValidationIssue("warning", message))
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="seed_workflow_replay", message=message[:1000])

    def _api_workflows_to_keep(self) -> list[dict]:
        """This run's retained workflows, then the previous version's it carried."""
        own = workflow_entries(self._api_scenarios.values())
        ids = {entry["scenario_id"] for entry in own}
        return own + [workflow for workflow in self._carried_seed_workflows if workflow["scenario_id"] not in ids]

    def _modify_end_checks(self) -> None:
        """After Phase 3 of a modify run: compare with the previous version, then the fix gate."""
        if not self._modify_mode:
            return
        self._replay_seed_workflows()
        self._note_seed_regressions()
        # After Phase 3, decide whether the reported failure is plausibly
        # addressed. A matching finding still present marks the run
        # incomplete with an honest, target-specific message instead of a
        # clean "success / 0 blockers".
        self._evaluate_fix_target_gate()

    def _render_model_change(self) -> str:
        """Every model change since the seed run, so the edit follows the new
        model instead of keeping old and new elements side by side."""
        change = getattr(self, "_model_change", None)
        # An older recipe lists the run's classes, model-sync additions included:
        # a class missing from the editor may never have been the user's.
        if change is None or not change.changes:
            return ""
        return (
            "The class model changed since the last run. Model change: "
            + "; ".join(change.changes)
            + ". Update the code (generated and hand-written files) to this model "
            "change. The new model wins: do not bring back removed or renamed elements."
        )

    def _collect_model_conformance_issues(self) -> list[str]:
        """Phase 3: the FastAPI backend's ORM and schemas against the model change."""
        change = getattr(self, "_model_change", None)
        if not self._modify_mode or change is None or not change.changes:
            return []
        try:
            issues = model_conformance_issues(self.output_dir, change.base, change.new)
        except Exception as exc:
            logger.warning("Model conformance check failed", exc_info=True)
            return [_check_did_not_run(
                "the model conformance check", f"it crashed: {type(exc).__name__}")]
        if issues is None:
            logger.warning("Model conformance check skipped: no FastAPI backend in the app")
            return [_check_did_not_run(
                "the model conformance check",
                "the app has no FastAPI backend (sql_alchemy.py / pydantic_classes.py)")]
        return issues

    def _regenerate_generated_files(self) -> None:
        """Bring the generated files up to the new model, merging into edited ones (#659).

        Only when the model changed: with the same model it would at most bring in a
        newer generator's output, which the modify request did not ask for.
        """
        change = getattr(self, "_model_change", None)
        if not ((change is not None and change.changes) or self._model_sync_classes):
            return
        generator = self._seed_generator_used
        if not generator:
            # Nothing to regenerate: the app was written without a BESSER generator.
            logger.info("Model changed; the app has no generator output to regenerate")
            self._trace.write(EVENT_REGENERATION, generator=None, skipped="the app has no generator output")
            return
        started = time.time()
        try:
            regeneration, reason = regenerate(
                self.output_dir, generator, self.domain_model, self._seed_output_files,
                change.class_renames if change else (), self._previous_domain_model, self.executor.app_name)
        except Exception as exc:
            # The run goes on as before: the agent applies the whole model change.
            logger.warning("Regenerating generated files failed", exc_info=True)
            regeneration, reason = None, f"it failed: {type(exc).__name__}: {exc}"[:MAX_REASON_CHARS]
        if regeneration is None:
            logger.warning("Generated files not regenerated for the model change: %s", reason)
            self._trace.write(EVENT_REGENERATION, generator=generator, skipped=reason)
            return
        self._regeneration = regeneration
        # A moved file counts once, under its new path.
        self._regenerated_files = sorted({
            *regeneration.written, *regeneration.merged, *regeneration.removed,
            *(new for _, new in regeneration.moved)})
        moved_from = [old for old, _ in regeneration.moved]
        self.executor.update_generator_files(
            regeneration.written + [new for _, new in regeneration.moved], regeneration.removed + moved_from)
        # Regenerated files hold the generator's text again; a moved file keeps its edits.
        self._seed_llm_modified.difference_update(regeneration.written)
        self._seed_llm_modified.update(new for old, new in regeneration.moved if old in self._seed_llm_modified)
        self._seed_llm_modified.difference_update(moved_from)
        self._trace.write(
            EVENT_REGENERATION, generator=generator, method=regeneration.method,
            replaced=regeneration.replaced, added=regeneration.added, merged=regeneration.merged,
            moved=[list(pair) for pair in regeneration.moved], removed=regeneration.removed,
            kept_in_use=regeneration.kept_in_use, to_update=[u.path for u in regeneration.to_update],
            not_restored=regeneration.not_restored, not_checked=regeneration.not_checked,
            upgraded_from=regeneration.upgraded_from, base_rebuilt=regeneration.base_rebuilt,
            merge_warning=regeneration.merge_warning or None)
        # The regenerated and merged files, as a generator call of the run trace
        # (file level: a merged file's lines are no longer the template's).
        regenerated = regeneration.written + regeneration.merged + [new for _, new in regeneration.moved]
        if regenerated:
            self._record_generator_output(generator_output_record(
                self.output_dir, self.output_dir, regenerated, {}, generator, started))

    def _render_seed_history(self) -> str:
        """Format the seed recipe's session history for the inventory."""
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        history = self._load_recipe_history(recipe_path)
        if not history:
            return ""
        recipe = load_recipe(self.output_dir)
        lines = ["\nPrevious work on this app (oldest first):", RECORD_NOT_INSTRUCTIONS]
        if len(history) > MAX_RUNS:
            lines.append(f"({len(history) - MAX_RUNS} earlier runs are not shown.)")
            history = history[-MAX_RUNS:]
        requests = recorded_requests(history, recipe)
        for i, (entry, request) in enumerate(zip(history, requests), 1):
            request = request or NO_REQUEST
            # The recipe can come from any repository: nothing of it reaches the prompt as it is.
            mode = entry.get("mode") if entry.get("mode") in ("create", "modify") else "run"
            files = entry.get("files_touched")
            files = [one_line(f) for f in files if isinstance(f, str)] if isinstance(files, list) else []
            file_note = ""
            if files:
                shown = ", ".join(files[:8])
                more = f" (+{len(files) - 8} more)" if len(files) > 8 else ""
                file_note = f" — touched: {shown}{more}"
            lines.append(f"  {i}. [{mode}] \"{request}\"{file_note}")
        lines.append(
            "Those changes are in the code: keep them working unless the new request says otherwise."
        )
        notes = next_run_notes(recipe)
        if notes:
            lines.append(notes)
        return "\n".join(lines)

    def _load_previous_verdicts(self) -> list[dict] | None:
        """The seed recipe's requirement verdicts on the seeded tree (see
        ``previous_run_verdicts``), and its request; None without a ledger or a recorded one.
        Best-effort: the recipe can come from a repository anyone edited."""
        recipe = load_recipe(self.output_dir)
        recorded = recipe.get("requirements") if isinstance(recipe, dict) else None
        if not self.enable_requirements_ledger or not isinstance(recorded, list) or not recorded:
            return None
        try:
            verdicts = previous_run_verdicts(recorded, self.output_dir)
        except Exception:
            logger.warning("Seed recipe requirements unreadable; judging the whole request", exc_info=True)
            return None
        instructions = recipe.get("instructions")
        self._previous_instructions = instructions if isinstance(instructions, str) else None
        return verdicts or None

    def _seed_generator_files_from_recipe(self) -> None:
        """Pre-load generator-file tags from a seeded run's recipe.

        ``modify()`` runs against ``output_dir`` copied from a previous
        run, and that copy KEEPS the previous ``.besser_recipe.json`` whose
        ``output_files`` entries are tagged ``source: generator|llm``. We
        replay the ``generator`` tags into
        ``self.executor._generator_files`` so the write-tool guardrail
        still protects deterministically-generated files, and
        ``_save_recipe`` re-tags them ``generator`` for the new run. Also
        records ``generator_used`` so the caller can frame the run.

        Best-effort: a missing / unreadable / malformed recipe just leaves
        every file tagged ``llm`` (harmless — the guardrail relaxes and the
        LLM can still edit anything).
        """
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        if not os.path.isfile(recipe_path):
            return
        try:
            with open_plain(recipe_path, "r", root=self.output_dir, encoding="utf-8") as fh:
                recipe = json.load(fh)
        except Exception:
            logger.debug(
                "Seed recipe unreadable; treating all seeded files as llm",
                exc_info=True,
            )
            return
        if not isinstance(recipe, dict):
            return
        self._seed_generator_used = recipe.get("generator_used")
        self._seed_target = self.target = Target.from_payload(recipe.get("target"))
        converted = (recipe.get("model") or {}).get("converted_rules")
        if isinstance(converted, list):
            self._seed_converted_rules = set(converted)
            self._seed_dropped_rules = [
                i for i in recipe.get("model_conversion_issues") or [] if is_dropped_by_model_change(i)]
        # Seed-issue forwarding: defects the seed run could not fix are
        # THIS run's opening tasks. Blockers only — warnings/style would
        # bloat the prompt with ruff noise and dilute the real debt. Not
        # forwarded either: unknowns (``unresolved_defects``), the model
        # contract (Phase 0 checks the current model) and the runtime gate's
        # verdict (it quotes findings forwarded on their own).
        try:
            recorded = [i for i in recipe.get("validation_issues", [])
                        if isinstance(i, dict) and isinstance(i.get("message"), str) and i["message"]]
            # A defect an earlier modify run found open since its previous version is still one.
            self._seed_recorded_issues = [i["message"].removeprefix(OPEN_SINCE_PREVIOUS) for i in recorded]
            seed_blockers = [
                ValidationIssue("blocker", i["message"].removeprefix(OPEN_SINCE_PREVIOUS))
                for i in recorded
                if i.get("severity") == "blocker" or i["message"].startswith(OPEN_SINCE_PREVIOUS)
            ]
            forwarded = [
                i for i in unresolved_defects(seed_blockers)
                if not i.message.startswith(("model contract:", self._RUNTIME_GATE_PREFIX))
            ]
            if len(forwarded) < len(seed_blockers):
                logger.info(
                    "Seed run: %d of %d blocker(s) not forwarded (unknowns, model contract, runtime gate)",
                    len(seed_blockers) - len(forwarded), len(seed_blockers))
            self._seed_unresolved_issues = [f"{_SEED_FORWARD_PREFIX}{i.message}" for i in forwarded]
        except Exception:
            logger.debug("Seed recipe validation_issues malformed", exc_info=True)
        try:
            self._seed_output_files = {
                entry["path"] for entry in recipe.get("output_files", [])
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)}
            for entry in recipe.get("output_files", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("source") == "generator"
                    and isinstance(entry.get("path"), str)
                ):
                    self.executor._generator_files.add(entry["path"])
                    # Edited by an earlier run: this run's tool calls alone would forget it.
                    if entry.get("llm_modified"):
                        self._seed_llm_modified.add(entry["path"])
        except Exception:
            logger.debug("Seed recipe output_files malformed", exc_info=True)

    def _route_modify_request(self) -> None:
        """Decide what this modify run changes with one forced ``choose_generator`` call.

        The answer says whether the request turns the app into another kind of
        software (a new target, built next to the existing project) and whether
        the newest message reports a failure (``_detect_fix_target``). A failed
        call keeps the seed's target and leaves the run ungated, with a warning.
        """
        seed = self._seed_target
        routing = None
        if self._can_route():
            with self._traced_llm_call("routing") as call:
                routing = self._select_generator_structured(self._modify_routing_prompt(seed), [], modify=True)
                call["outcome"] = "failed" if routing is None or routing.target is None else "ok"
        if routing is None or routing.target is None:
            logger.warning(
                "Modify run: the routing call failed; keeping the existing target (%s) "
                "and treating the request as a change, not a failure report",
                seed.description if seed else "none recorded")
            return
        self._modify_routing = routing
        if seed is None:
            self._record_target(routing.target)
        elif routing.changes_target:
            self._previous_target = seed
            self._record_target(routing.target)

    def _modify_routing_prompt(self, seed: Target | None) -> str:
        instructions = self._instructions
        existing = (
            f"The existing app was built as: {seed.description or seed.kind} ({seed.kind}).\n"
            if seed is not None else "The existing app's target was not recorded.\n"
        )
        try:
            classes = sorted(c.name for c in self.domain_model.get_classes()) if self.domain_model else []
        except Exception:
            classes = []
        return (
            f"Conversation so far (context only): {user_request(instructions)}\n\n"
            f"Newest user message: {latest_user_message(instructions)}\n\n"
            f"{TARGET_RULES}\n{STACK_RULES}\n{MODIFY_RULES}\n\n"
            f"{existing}Model classes: {', '.join(classes) or 'none'}\n"
            "The generator is already settled: answer generator=NONE."
        )

    def _mark_rules_broken_by_model_change(self) -> None:
        """Mark rejected model rules whose element this model change removed or renamed.

        Needs the previous run's converted-rule fingerprints and full model (an
        older recipe's class names cannot show an attribute was removed);
        without either, every rejected rule keeps the normal recovery path.
        """
        change = getattr(self, "_model_change", None)
        converted = getattr(self, "_seed_converted_rules", None)
        if change is None or change.basis != FULL_MODEL or converted is None:
            return
        mark_rules_broken_by_model_change(
            getattr(self.domain_model, "conversion_issues", None), change,
            converted_before=converted, dropped_before=self._seed_dropped_rules,
        )

    def _detect_fix_target(self, instructions: str) -> None:
        """MODIFY-only: turn a reported failure into the run's success target.

        Sets ``self._fix_target`` and ``self._is_fix_run`` from the routing
        call's ``is_failure_report`` / ``reported_entity``. A feature request,
        or a failed routing call, leaves the run ungated. NEVER invoked from
        run()/resume(); from-scratch generation is untouched.
        """
        from besser.spec_driven_agent.planning.fix_target import parse_reported_target
        routing = self._modify_routing
        self._fix_target = (
            parse_reported_target(instructions, self.domain_model, routing.reported_entity)
            if routing is not None and routing.is_failure_report else None
        )
        self._is_fix_run = self._fix_target is not None
        if self._is_fix_run:
            logger.info(
                "Fix/modify run: reported target = %s (kind=%s, entities=%s)",
                self._fix_target.descriptor,
                self._fix_target.kind,
                ", ".join(self._fix_target.entities) or "none",
            )

    def _modify_checkpoint_state(self) -> dict | None:
        """What a resume needs to continue this run as a modify run; ``None`` for a create run."""
        if not self._modify_mode:
            return None
        change = getattr(self, "_model_change", None)
        converted = getattr(self, "_seed_converted_rules", None)
        return {
            "seed_generator_used": self._seed_generator_used,
            "seed_llm_modified": sorted(self._seed_llm_modified),
            # The seed recipe's findings. The run's own recipe may replace it before a resume.
            "seed_unresolved_issues": self._seed_unresolved_issues,
            "seed_baseline": self._seed_baseline,
            "seed_converted_rules": sorted(converted) if converted is not None else None,
            "seed_dropped_rules": getattr(self, "_seed_dropped_rules", []),
            "fix_target": dataclasses.asdict(self._fix_target) if self._fix_target else None,
            "previous_target": self._previous_target.to_dict() if self._previous_target else None,
            "model_change": (
                {**change.to_dict(), "base": change.base, "new": change.new} if change else None),
            # Model sync is an LLM call: its result is replayed, not asked again.
            "model_sync_classes": self._model_sync_classes,
            "updated_project_export": self._updated_project_export,
            "model_sync_error": self._model_sync_error,
            "seed_source_hashes": self._seed_source_hashes,
            "regenerated_files": self._regenerated_files,
            "previous_verdicts": self._previous_verdicts,
            "previous_instructions": self._previous_instructions,
            "previous_ids": sorted(self._previous_ids),
            "follow_up_continues": self._follow_up_continues,
            "replaced_ids": sorted(self._replaced_ids),
        }

    def _restore_modify_state(self, state: dict) -> None:
        """Continue a checkpointed modify run as one (the inverse of ``_modify_checkpoint_state``)."""
        from besser.spec_driven_agent.planning.fix_target import ReportedTarget
        self._modify_mode = True
        self.executor.enable_modify_guard()
        self._seed_generator_used = state.get("seed_generator_used")
        self._seed_llm_modified.update(
            p for p in state.get("seed_llm_modified") or [] if isinstance(p, str))
        fix_target = state.get("fix_target")
        self._fix_target = (
            ReportedTarget(**{**fix_target, "entities": tuple(fix_target.get("entities") or ())})
            if isinstance(fix_target, dict) else None)
        self._is_fix_run = self._fix_target is not None
        self._previous_target = Target.from_payload(state.get("previous_target"))
        self._model_change = model_change_from_dict(state.get("model_change"))
        self._seed_unresolved_issues = [
            i for i in state.get("seed_unresolved_issues") or [] if isinstance(i, str)]
        baseline = state.get("seed_baseline")
        self._seed_baseline = [i for i in baseline if isinstance(i, str)] if isinstance(baseline, list) else None
        # Still the seed's file: a run writes its own when it saves its recipe.
        self._seed_workflows = load_workflows(self.output_dir)
        converted = state.get("seed_converted_rules")
        self._seed_converted_rules = set(converted) if isinstance(converted, list) else None
        self._seed_dropped_rules = [
            i for i in state.get("seed_dropped_rules") or [] if isinstance(i, dict)]
        self._mark_rules_broken_by_model_change()
        classes = state.get("model_sync_classes")
        if isinstance(classes, list) and classes and self.domain_model is not None:
            self._model_sync_classes = classes
            self._apply_new_classes(classes)
        export = state.get("updated_project_export")
        self._updated_project_export = export if isinstance(export, dict) else None
        self._model_sync_error = state.get("model_sync_error")
        hashes = state.get("seed_source_hashes")
        self._seed_source_hashes = hashes if isinstance(hashes, dict) else None
        self._regenerated_files = [
            p for p in state.get("regenerated_files") or [] if isinstance(p, str)]
        verdicts = state.get("previous_verdicts")
        self._previous_verdicts = verdicts if isinstance(verdicts, list) and verdicts else None
        previous = state.get("previous_instructions")
        self._previous_instructions = previous if isinstance(previous, str) else None
        self._previous_ids = {rid for rid in state.get("previous_ids") or [] if type(rid) is int}
        self._follow_up_continues = state.get("follow_up_continues") is True
        self._replaced_ids = {rid for rid in state.get("replaced_ids") or [] if type(rid) is int}

    def _acceptance_seed_issues(self) -> list[str]:
        """Acceptance-matrix findings on the current (seed) workspace."""
        try:
            from besser.spec_driven_agent.validation.acceptance import (
                build_acceptance_matrix,
                matrix_issues,
            )
            matrix = build_acceptance_matrix(self.output_dir, self.domain_model,
                                             gui_model=self.gui_model)
            return matrix_issues(matrix)
        except Exception:
            logger.debug("Seed acceptance computation failed", exc_info=True)
            return []

    def _fix_run_scoped_issues(self) -> list[str]:
        """Explicit Phase-2 fix instructions derived from the reported target.

        Always leads with a headline naming the reported failure (so the
        LLM knows the success criterion even when nothing structural
        matched), then appends the tool's OWN structural findings on the
        SEED workspace that match the target (acceptance matrix +
        data-contract lint). Empty on a non-fix run and on run().
        """
        if not self._is_fix_run or self._fix_target is None:
            return []
        from besser.spec_driven_agent.planning.fix_target import finding_matches_target

        issues: list[str] = [
            "The user reported this failure and it must be resolved in this "
            f"run: {self._fix_target.descriptor}. Reproduce the cause, fix "
            "it, and confirm the reported request/flow now succeeds; do not "
            "end the run while it is unresolved."
        ]
        raw = list(self._acceptance_seed_issues()) + list(
            self._collect_data_contract_issues()
        )
        for finding in raw:
            if finding_matches_target(finding, self._fix_target):
                issues.append(
                    "Concrete defect behind the reported failure — "
                    f"{finding}. Repair this so the reported request succeeds."
                )
        return issues

    def _promote_fix_target_findings(
        self, issues: list[ValidationIssue],
    ) -> list[ValidationIssue]:
        """Promote target-matching findings from warning to blocker.

        Scoped to a fix run only; a no-op otherwise (so from-scratch
        Phase 3 classification is identical). A promoted finding is
        rewritten with a stable prefix + the target descriptor so it reads
        honestly in the recipe and the end-of-run gate can identify it.
        Then, on any modify run, the previous version's defects are
        labelled (``_label_seed_findings``).
        """
        if not self._is_fix_run or self._fix_target is None:
            return self._label_seed_findings(issues)
        from besser.spec_driven_agent.planning.fix_target import finding_matches_target

        earlier = self._earlier_requirement_findings()
        promoted: list[ValidationIssue] = []
        for issue in issues:
            # A refusal the probe observed is corroborated when it matches what the user reported.
            if (
                issue.severity != "blocker"
                and issue.message not in earlier
                and not issue.message.startswith(_NEVER_PROMOTED)
                and finding_matches_target(issue.message, self._fix_target)
            ):
                promoted.append(ValidationIssue(
                    "blocker",
                    f"{self._FIX_TARGET_PREFIX}{issue.message} "
                    f"(matches the reported failure: {self._fix_target.descriptor})",
                ))
            else:
                promoted.append(issue)
        return self._label_seed_findings(promoted)

    def _earlier_requirement_findings(self) -> frozenset[str]:
        """The ledger's findings on requirements of earlier runs: they judge those requests and
        name their entities, so they say nothing about the failure reported now."""
        return frozenset(message for verdict in self._requirement_verdicts if verdict["id"] in self._previous_ids
                         for message in ledger_issues([verdict]))

    def _evaluate_fix_target_gate(self) -> None:
        """Decide whether the reported failure is plausibly addressed.

        Sets ``_fix_target_resolved`` / ``_fix_target_message`` from the
        FINAL Phase-3 issue list. A blocker that matches the target (either
        one we promoted or a pre-existing data-contract blocker about the
        same entity) means the run must NOT claim a clean success. With no
        such finding it is ``True`` only when a check exercised the target on
        the delivered tree (``_fix_target_exercised``), else ``None``.

        A soft target (no entities we could ground) is left as ``None`` —
        we neither confirm nor deny a specific defect, rather than
        over-claiming either way.
        """
        if not self._is_fix_run or self._fix_target is None:
            return
        if self._fix_target.is_soft:
            self._fix_target_resolved = None
            logger.info(
                "Fix run: soft target %s — no structural signal to verify",
                self._fix_target.descriptor,
            )
            return
        from besser.spec_driven_agent.planning.fix_target import finding_matches_target

        earlier = self._earlier_requirement_findings()
        # A not-confirmed observation that matches the report is the failure persisting too.
        unresolved = [
            i for i in self._validation_issues
            if (is_completion_issue(i) or (is_not_confirmed(i) and not i.message.startswith(_NEVER_PROMOTED)))
            and i.message not in earlier
            and (
                i.message.startswith(self._FIX_TARGET_PREFIX)
                or finding_matches_target(i.message, self._fix_target)
            )
        ]
        if unresolved:
            self._fix_target_resolved = False
            detail = unresolved[0].message
            if detail.startswith(self._FIX_TARGET_PREFIX):
                detail = detail[len(self._FIX_TARGET_PREFIX):]
            self._fix_target_message = (
                "I changed the app, but could not confirm the reported "
                f"failure is fixed ({self._fix_target.descriptor}). "
                f"Outstanding issue: {user_sentence(detail)}"
            )
            logger.warning(
                "Fix run: reported target NOT confirmed fixed — %s",
                self._fix_target.descriptor,
            )
        else:
            # No finding left is not evidence: resolved only when a check exercised the target.
            self._fix_target_evidence = self._fix_target_exercised()
            self._fix_target_resolved = True if self._fix_target_evidence else None
            logger.info(
                "Fix run: no finding matches the reported target %s; %s",
                self._fix_target.descriptor,
                self._fix_target_evidence or "no check exercised it on the delivered tree",
            )
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="fix_target_gate",
            resolved=self._fix_target_resolved,
            target=self._fix_target.descriptor,
            evidence=self._fix_target_evidence,
        )

    def _fix_target_exercised(self) -> str | None:
        """The check that created a record of a target entity on the delivered tree, or None:
        the boot-and-create probe, or a ``test_api`` workflow that created and read back a
        record on the probe's route for it."""
        from besser.spec_driven_agent.planning.mutation_inventory import route_serves
        from besser.spec_driven_agent.validation.api_probe import confirmed_create_paths

        revision = self._workspace_revision()
        facts = getattr(self, "_runtime_probe_facts", None)
        routes = []
        for backend in (facts[1] if facts and facts[0] == revision else None) or []:
            for name, entry in (backend.get("entities") or {}).items():
                if name not in self._fix_target.entities:
                    continue
                if entry.get("verdict") == "created":
                    return f"the boot-and-create probe created {name} on the delivered tree"
                routes.append((backend.get("backend"), entry.get("path"), name))
        for record in self._api_scenarios.values():
            report = record["report"]
            if record["revision"] != revision:
                continue
            for path in confirmed_create_paths(report):
                for backend, route, name in routes:
                    if backend == report.get("backend") and isinstance(route, str) and route_serves(route, path):
                        return (f"test_api workflow {record.get('scenario_id')} created {name} "
                                "on the delivered tree")
        return None

    @staticmethod
    def _is_valid_model_name(name) -> bool:
        """Cheap pre-check mirroring the metamodel ``NamedElement`` rules.

        The name setter rejects None / empty / whitespace / spaces /
        hyphens; we filter those here so a bad LLM name is skipped
        silently instead of forcing a ValueError through the try/except.
        """
        return (
            isinstance(name, str)
            and name.strip() != ""
            and " " not in name
            and "-" not in name
        )

    def _resolve_primitive_type(self, type_str):
        """Map an LLM-supplied type string to a ``PrimitiveDataType``."""
        from besser.BUML.metamodel.structural import PrimitiveDataType

        key = type_str.strip().lower() if isinstance(type_str, str) else ""
        return PrimitiveDataType(self._PRIMITIVE_TYPE_ALIASES.get(key, "str"))

    def _derive_and_apply_model_deltas(self, instructions: str) -> None:
        """MODIFY-only: sync the domain model with the modification intent.

        Asks the orchestrator's LLM for genuinely-new domain entities
        implied by ``instructions`` (e.g. "add authentication" → a ``User``
        class), applies them to ``self.domain_model`` IN PLACE, and
        re-serialises an updated project export onto
        ``self._updated_project_export`` so the GitHub push writes
        ``buml/diagrams.json`` + ``buml/*.py`` from the UPDATED model.

        Fully guarded — this is the load-bearing safety contract:
          * Skipped entirely when there is no class diagram to sync
            (``self.domain_model is None``) or the client is a test/mock
            double (same ``_client`` gate the generator selector uses), so
            unit tests driving the full ``modify()`` loop with a scripted
            client are unaffected.
          * Any failure (LLM error, malformed delta, serialisation error)
            is logged as a warning: ``modify()`` proceeds exactly as without
            it — no model change beyond what already applied, no crash. A
            failed LLM call is also kept in ``_model_sync_error`` for the
            trace and the recipe. An empty result is the common, expected
            case.

        NEVER invoked from ``run()`` / ``resume()`` / ``_run_phase1`` — the
        from-scratch path is untouched and byte-identical.
        """
        # Class-diagram only: nothing to sync without a domain model.
        if self.domain_model is None:
            return
        # Skip mock/duck-typed clients that don't look like a real provider
        # (mirrors _select_generator_with_llm's gate). Keeps the full
        # modify() loop deterministic under a scripted test client.
        if not hasattr(self.client, "_client"):
            return
        if not self._client_supports_structured_chat():
            return

        try:
            with self._traced_llm_call("model_deltas"):
                new_classes = self._request_model_deltas(instructions)
        except Exception as exc:
            logger.warning(
                "modify: model-sync LLM call failed; proceeding without "
                "model sync: %s", exc, exc_info=True,
            )
            self._model_sync_error = str(exc)[:500]
            return

        if not new_classes:
            return

        added = self._apply_new_classes(new_classes)
        if added == 0:
            return
        self._model_sync_classes = new_classes

        logger.info("modify: model-sync added %d new class(es)", added)

        # Re-serialise the (now-mutated) model and slot it into the run's
        # original project export for the push path. A failure here still
        # leaves the domain-model mutation in place (it already improved
        # Phase 2's inventory) — only the push export falls back.
        try:
            from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
                class_buml_to_json,
            )

            updated_class_json = class_buml_to_json(self.domain_model)
            self._updated_project_export = self._build_updated_project_export(
                updated_class_json
            )
        except Exception:
            logger.warning(
                "modify: failed to serialise updated model; the push will "
                "fall back to the request's projectExport", exc_info=True,
            )
            self._updated_project_export = None

    def _request_model_deltas(self, instructions: str) -> list[dict]:
        """Structured LLM call returning ``new_classes`` implied by the edit.

        Uses the same forced-tool structured-prediction infra as
        ``_select_generator_structured``. Returns a (possibly empty) list
        of ``{"name": str, "attributes": [{"name": str, "type": str}]}``.
        """
        existing = sorted(c.name for c in self.domain_model.get_classes())
        delta_tool = {
            "name": "derive_model_deltas",
            "description": (
                "Report genuinely-new domain entities the underlying data "
                "MODEL should gain because of a modification request."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "new_classes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "attributes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                        },
                                        "required": ["name", "type"],
                                    },
                                },
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["new_classes"],
            },
        }
        prompt = (
            "An existing app is being MODIFIED. The change is the user's newest message:\n\n"
            f"'{latest_user_message(instructions)}'\n\n"
            f"The conversation so far (context only):\n\n{user_request(instructions)}\n\n"
            "The app's current domain model has these classes: "
            f"{', '.join(existing) if existing else '(none)'}.\n\n"
            "List ONLY genuinely NEW domain entities the MODEL should gain "
            "because of this change — e.g. adding authentication implies a "
            "User/Account entity with username/password/role. Do NOT repeat "
            "entities that already exist. Return an EMPTY list if the change "
            "is purely code-level (styling, responsiveness, routing, copy, "
            "config, performance). An empty list is the common, expected "
            "answer."
        )
        planning_model = getattr(self.client, "planning_model", None)
        # The user's key may not reach the planning model: retry on the
        # primary, as the generator selection does.
        for model_override in dict.fromkeys([planning_model, None]):
            try:
                response = self.client.chat(
                    system=(
                        "You extract new domain-model entities implied by a code "
                        "modification. Call derive_model_deltas with your answer."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    tools=[delta_tool],
                    force_tool="derive_model_deltas",
                    model_override=model_override,
                )
                break
            except Exception as exc:
                if model_override is None:
                    raise
                logger.warning(
                    "modify: model-sync call on planning model %s failed (%s); "
                    "retrying on the primary model", model_override, exc,
                )
        for block in response.get("content", []):
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_use":
                continue
            payload = getattr(block, "input", None) or (
                block.get("input") if isinstance(block, dict) else None
            )
            classes = (payload or {}).get("new_classes", [])
            if isinstance(classes, list):
                return classes
        return []

    def _apply_new_classes(self, new_classes: list[dict]) -> int:
        """Apply ``new_classes`` to ``self.domain_model`` in place.

        Returns the number of classes actually added. Duplicates (name
        collides with an existing type) and invalid names are skipped
        silently; the model set setter also raises on duplicates, which
        the per-class try/except absorbs.
        """
        from besser.BUML.metamodel.structural import Class, Property

        # Existing type names (classes + enums + primitives) — adding a
        # type whose name already exists raises in the ``types`` setter.
        existing_type_names = {t.name for t in self.domain_model.types}
        added = 0
        for spec in new_classes:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not self._is_valid_model_name(name):
                continue
            if name in existing_type_names:
                continue  # skip duplicates silently
            try:
                new_cls = Class(name=name)
                for attr in spec.get("attributes") or []:
                    if not isinstance(attr, dict):
                        continue
                    attr_name = attr.get("name")
                    if not self._is_valid_model_name(attr_name):
                        continue
                    prop_type = self._resolve_primitive_type(attr.get("type"))
                    try:
                        new_cls.add_attribute(
                            Property(name=attr_name, type=prop_type)
                        )
                    except Exception:
                        # Duplicate attribute name, etc. — skip that attr.
                        continue
                self.domain_model.add_type(new_cls)
            except Exception:
                logger.debug(
                    "modify: skipped invalid model delta %r", name, exc_info=True,
                )
                continue
            existing_type_names.add(name)
            added += 1
        return added

    def _build_updated_project_export(self, updated_class_json: dict):
        """Slot ``updated_class_json`` into a copy of the run's export.

        Replaces the active ``ClassDiagram`` entry's ``model`` with the
        re-serialised class diagram. Returns ``None`` (push falls back to
        the request's projectExport) when there is no source export or it
        has no ClassDiagram entry to update.
        """
        import copy

        source = self._source_project_export
        if not isinstance(source, dict):
            return None
        diagrams = source.get("diagrams")
        if not isinstance(diagrams, dict):
            return None
        class_entries = diagrams.get("ClassDiagram")
        if not isinstance(class_entries, list) or not class_entries:
            return None

        # Resolve the active index the same way ProjectInput.get_active_diagram
        # does (currentDiagramIndices, clamped into range).
        idx = 0
        indices = source.get("currentDiagramIndices")
        if isinstance(indices, dict):
            maybe_idx = indices.get("ClassDiagram")
            if isinstance(maybe_idx, int):
                idx = maybe_idx
        idx = min(max(idx, 0), len(class_entries) - 1)

        export = copy.deepcopy(source)
        entry = export["diagrams"]["ClassDiagram"][idx]
        if not isinstance(entry, dict):
            return None
        entry["model"] = updated_class_json
        return export

    def _apply_modify_budget(self) -> None:
        """Raise only the output-token ceiling for modify/fix runs.

        Called once at the start of ``modify()`` (before Phase 2). A
        modify/fix run frequently rewrites a whole existing file in a
        single ``write_file`` turn -- a targeted edit that still touches
        most of the file, or a smaller model electing a full rewrite over
        a surgical patch -- which is exactly the large-single-response
        case that overruns the client's default per-call output cap and
        truncates mid-file (see the ``stop_reason in ("max_tokens",
        "length")`` handling in ``_run_customization_loop``).

        Mirrors ``_apply_adaptive_budget`` (raises only the per-call
        output limit; the caller-authorised cost and runtime caps are
        never touched), but is keyed on the modify path rather than
        ``self._generator_used`` -- a modify run adopts the seed's
        generator name, so that "no scaffold ran" signal isn't available
        here. NEVER invoked from ``run()`` / ``resume()``, so
        first-generation output sizing (scaffolded stays at the client
        default; pure from-scratch uses ``_apply_adaptive_budget``) is
        unchanged.
        """
        try:
            current_max_tokens = self.client.max_tokens
        except (AttributeError, NotImplementedError):
            # Defensive: a test double / older client without the
            # max_tokens property. Don't fail the run over telemetry.
            current_max_tokens = None
        if current_max_tokens is not None and current_max_tokens < MODIFY_MAX_TOKENS:
            logger.info(
                "Adaptive response sizing: raising output-token limit %d -> %d for "
                "modify/fix run",
                current_max_tokens, MODIFY_MAX_TOKENS,
            )
            self.client.max_tokens = MODIFY_MAX_TOKENS
            self._adaptive_budget_applied = True

