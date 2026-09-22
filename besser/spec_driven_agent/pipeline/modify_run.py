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

import json
import logging
import os
import time

from besser.spec_driven_agent.agent.prompt_builder import build_inventory
from besser.spec_driven_agent.errors import EmptyInstructionsError
from besser.spec_driven_agent.planning.user_request import user_request
from besser.spec_driven_agent.planning.specification import validate_specification
from besser.spec_driven_agent.providers.llm_client import MODIFY_MAX_TOKENS
from besser.spec_driven_agent.state.tracing import (
    EVENT_PHASE_ENTER,
    EVENT_PHASE_EXIT,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_SNAPSHOT,
    EVENT_VALIDATION_ISSUE,
)
from besser.spec_driven_agent.validation.issues import ValidationIssue

logger = logging.getLogger(__name__)


class ModifyRunMixin:
    """modify and friends; see the module docstring."""

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
        ``self.domain_model`` reflects that. For the MVP this is a pure
        code-edit — the inventory and gap analyzer already surface the
        model's classes vs. the files on disk, so Phase 2 authors any
        deltas via write_file / modify_file (no scaffold merge yet).

        Returns the path to ``self.output_dir``.
        """
        if not instructions or not instructions.strip():
            raise EmptyInstructionsError("Instructions cannot be empty")
        validate_specification(instructions)
        self._instructions = instructions

        self._start_time = time.monotonic()
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
        # Frame the run as editing an existing project. When the seed came
        # from a deterministic generator we adopt that name (accurate — the
        # base + prior LLM edits descend from it) so gap analysis, the
        # scaffold-snapshot inlining, and the saved recipe all line up.
        self._generator_used = self._seed_generator_used
        self.executor.set_scaffold_family(
            self._scaffold_family(), self._instructions)

        # -- Model-sync: derive + apply class-diagram deltas implied by the
        # instruction BEFORE building the inventory, so a genuinely new
        # domain entity (e.g. a ``User`` class for "add authentication")
        # flows into Phase 2's inventory AND the updated model reaches the
        # push path. Fully guarded (see the method): an empty/failed/bad
        # delta leaves the run proceeding EXACTLY as before — no model
        # change, no crash. Scoped to modify() only; run()/_run_phase1
        # never invoke it, so from-scratch output is byte-identical.
        self._derive_and_apply_model_deltas(instructions)

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
            instructions=instructions[:500],
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
            max_turns=self.max_turns,
            fix_run=self._is_fix_run,
            fix_target=(self._fix_target.descriptor if self._fix_target else None),
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
        # Forward the seed run's unresolved blockers (D1): they ride along
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
        # Feed the tool's OWN structural findings that match the reported
        # target into Phase 2 as explicit fix instructions, so the LLM acts
        # on the concrete defect (e.g. "the Watchlist create form is not
        # wired") instead of only the user's paraphrase. Empty on a
        # non-fix modify run and on run()/resume().
        fix_seed_issues = self._fix_run_scoped_issues()
        if fix_seed_issues:
            for issue in fix_seed_issues:
                self._trace.write(
                    EVENT_VALIDATION_ISSUE, phase="fix_target_seed", message=issue,
                )
        self._trace.write(EVENT_PHASE_ENTER, phase="phase2_modify")
        self._run_phase2(
            instructions,
            extra_issues=(
                phase1_issues + self._seed_unresolved_issues + fix_seed_issues
            ),
        )
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
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="phase3",
            unresolved_blockers=sum(
                1 for i in self._validation_issues if i.severity == "blocker"
            ),
            exit_reason=self._phase3_exit_reason,
        )

        # -- Fix/modify success gate --------------------------------------
        # After Phase 3, decide whether the reported failure is plausibly
        # addressed. A matching finding still present marks the run
        # incomplete with an honest, target-specific message instead of a
        # clean "success / 0 blockers".
        self._evaluate_fix_target_gate()

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

    def _render_seed_history(self) -> str:
        """Format the seed recipe's session history for the inventory."""
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        history = self._load_recipe_history(recipe_path)
        if not history:
            return ""
        lines = ["\nPrevious work on this app (oldest first):"]
        for i, entry in enumerate(history, 1):
            request = (entry.get("instructions") or "").strip().replace("\n", " ")
            mode = entry.get("mode") or "run"
            files = entry.get("files_touched") or []
            file_note = ""
            if files:
                shown = ", ".join(files[:8])
                more = f" (+{len(files) - 8} more)" if len(files) > 8 else ""
                file_note = f" — touched: {shown}{more}"
            lines.append(f"  {i}. [{mode}] \"{request}\"{file_note}")
        lines.append(
            "Respect this history: those changes are deliberate and must "
            "survive your edits unless the new request says otherwise."
        )
        return "\n".join(lines)

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
            with open(recipe_path, "r", encoding="utf-8") as fh:
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
        # Seed-issue forwarding: blockers the seed run could not fix are
        # THIS run's opening tasks. Blockers only — warnings/style would
        # bloat the prompt with ruff noise and dilute the real debt.
        try:
            self._seed_unresolved_issues = [
                f"Unresolved from the previous run: {i.get('message', '')}"
                for i in recipe.get("validation_issues", [])
                if isinstance(i, dict)
                and i.get("severity") == "blocker"
                and i.get("message")
            ]
        except Exception:
            logger.debug("Seed recipe validation_issues malformed", exc_info=True)
        try:
            for entry in recipe.get("output_files", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("source") == "generator"
                    and isinstance(entry.get("path"), str)
                ):
                    self.executor._generator_files.add(entry["path"])
        except Exception:
            logger.debug("Seed recipe output_files malformed", exc_info=True)

    def _detect_fix_target(self, instructions: str) -> None:
        """MODIFY-only: parse the user-reported failure into a target.

        Sets ``self._fix_target`` and ``self._is_fix_run``. Best-effort — a
        parse failure (or a modify run with no fix/error vocabulary) leaves
        the run ungated, so a plain feature-add modify behaves exactly as
        before. NEVER invoked from run()/resume(); from-scratch generation
        is untouched.
        """
        try:
            from besser.spec_driven_agent.planning.fix_target import parse_reported_target
            self._fix_target = parse_reported_target(
                instructions, self.domain_model,
            )
        except Exception:
            logger.debug("Fix-target parse failed", exc_info=True)
            self._fix_target = None
        self._is_fix_run = self._fix_target is not None
        if self._is_fix_run:
            logger.info(
                "Fix/modify run: reported target = %s (kind=%s, entities=%s)",
                self._fix_target.descriptor,
                self._fix_target.kind,
                ", ".join(self._fix_target.entities) or "none",
            )

    def _acceptance_seed_issues(self) -> list[str]:
        """Acceptance-matrix findings on the current (seed) workspace."""
        try:
            from besser.spec_driven_agent.validation.acceptance import (
                build_acceptance_matrix,
                matrix_issues,
            )
            matrix = build_acceptance_matrix(self.output_dir, self.domain_model)
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
        data-contract lint). Empty on a non-fix run and on run()/resume().
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
        """
        if not self._is_fix_run or self._fix_target is None:
            return issues
        from besser.spec_driven_agent.planning.fix_target import finding_matches_target

        promoted: list[ValidationIssue] = []
        for issue in issues:
            if (
                issue.severity != "blocker"
                and not issue.message.startswith(self._FIX_TARGET_PREFIX)
                and finding_matches_target(issue.message, self._fix_target)
            ):
                promoted.append(ValidationIssue(
                    "blocker",
                    f"{self._FIX_TARGET_PREFIX}{issue.message} "
                    f"(matches the reported failure: {self._fix_target.descriptor})",
                ))
            else:
                promoted.append(issue)
        return promoted

    def _evaluate_fix_target_gate(self) -> None:
        """Decide whether the reported failure is plausibly addressed.

        Sets ``_fix_target_resolved`` / ``_fix_target_message`` from the
        FINAL Phase-3 issue list. A blocker that matches the target (either
        one we promoted or a pre-existing data-contract blocker about the
        same entity) means the run must NOT claim a clean success.

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

        unresolved = [
            i for i in self._validation_issues
            if i.severity == "blocker"
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
                f"Outstanding issue: {detail}"
            )
            logger.warning(
                "Fix run: reported target NOT confirmed fixed — %s",
                self._fix_target.descriptor,
            )
        else:
            self._fix_target_resolved = True
            logger.info(
                "Fix run: reported target plausibly addressed — %s",
                self._fix_target.descriptor,
            )
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="fix_target_gate",
            resolved=self._fix_target_resolved,
            target=self._fix_target.descriptor,
        )

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
            is swallowed: ``modify()`` proceeds EXACTLY as it does today —
            no model change beyond what already applied, no crash. An empty
            result is the common, expected case.

        NEVER invoked from ``run()`` / ``resume()`` / ``_run_phase1`` — the
        from-scratch path is untouched and byte-identical.
        """
        # Class-diagram-only MVP: nothing to sync without a domain model.
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
            new_classes = self._request_model_deltas(instructions)
        except Exception:
            logger.debug(
                "modify: model-delta LLM call failed; proceeding without "
                "model sync", exc_info=True,
            )
            return

        if not new_classes:
            return

        added = self._apply_new_classes(new_classes)
        if added == 0:
            return

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
        request = user_request(instructions)
        prompt = (
            "An existing app is being MODIFIED with this instruction:\n\n"
            f"'{request}'\n\n"
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
        response = self.client.chat(
            system=(
                "You extract new domain-model entities implied by a code "
                "modification. Call derive_model_deltas with your answer."
            ),
            messages=[{"role": "user", "content": prompt}],
            tools=[delta_tool],
            force_tool="derive_model_deltas",
            model_override=planning_model,
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

