"""The Phase 2 rules must not prescribe the failures seen live (2026-09-17).

Read back from a real prompt, three rules were instructing the behaviour the
run was blamed for: "three or more changes -> one write_file" (whole-file
rewrites of scaffold code), "every project must ship package.json ... ship a
README" (the rewritten package.json and the unrequested README), and "don't
re-read files, remember what's in them" (quoting a stale copy after an edit
or a compaction).
"""

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.prompt_builder import build_system_prompt


def _model() -> DomainModel:
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="M", types={user})


def _prompt(scaffold: bool) -> str:
    if scaffold:
        inventory = (
            "Generator `generate_fastapi_backend` produced 2 files:\n"
            "- backend/main_api.py (2,000 bytes)\n- backend/database.py (900 bytes)"
        )
        snapshot = "### `backend/database.py`\n\n```\nx = 1\n```\n"
    else:
        inventory = "Generator `None` produced 0 files:"
        snapshot = ""
    return build_system_prompt(
        _model(), None, None, inventory=inventory, instructions="Build it",
        max_turns=10, scaffold_snapshot=snapshot,
    )


class TestRulesDoNotPrescribeTheFailures:

    def test_no_rule_prefers_a_whole_file_rewrite(self):
        """A rewrite may be ESCALATED TO after refusals; it must never be
        prescribed by a count of SUCCESSFUL edits, nor on an unread file.

        7cb06829 deleted "three or more changes -> one write_file" because it
        counted *successful* edits: three good edits to one router became a
        whole-file rewrite. The 2026-09-20 edit-ladder change puts a
        whole-file rewrite back in these rules, but keyed on two *refused*
        edits on one path - the same trigger class the guard was re-pointed
        at, and the executor pops that counter on every landed edit (pinned
        behaviourally by
        test_repeat_rejection_escalation.test_successful_edits_never_escalate_to_a_rewrite).

        The read-first constraint survives the rewording, so this test moved
        with it rather than dropping: "Never rewrite a file from memory"
        became "Do not rewrite a file you have not read this run", which is
        the wording ``ToolExecutor._write_file`` actually enforces through
        its ``_known_paths`` check. The phrases are matched against a
        whitespace-flattened prompt because the new rule wraps over lines.
        """
        for scaffold in (True, False):
            prompt = _prompt(scaffold)
            flat = " ".join(prompt.split())
            # Never again: a rewrite ordered by a count of edits that WORKED.
            assert "three or more changes" not in prompt
            assert "is cheaper than a chain" not in prompt
            for banned in ("after three edits", "after three changes",
                           "three good edits", "after two successful edits",
                           "after three successful edits"):
                assert banned not in flat.lower(), banned
            # The escalation that is allowed, and the only trigger for it.
            assert "After two refused edits on one file, switch strategy:" in flat
            assert "read_file` the WHOLE file, then `write_file` it back" in flat
            # Read-first: the rewrite is of a file just read, never of memory.
            assert "Do not rewrite a file you have not read this run." in flat
            assert ("Use `write_file` for new files, and to replace any file "
                    "you have just read in full") in flat

    def test_scaffold_runs_do_not_recreate_project_files_or_add_a_readme(self):
        prompt = _prompt(True)
        assert "Ship a brief `README.md`" not in prompt
        assert "never recreate or rewrite" in prompt

    def test_from_scratch_runs_still_ship_the_standard_project_file(self):
        prompt = _prompt(False)
        assert "`pyproject.toml` for Python" in prompt
        assert "never recreate or rewrite" not in prompt

    def test_re_read_after_editing_or_compaction(self):
        prompt = _prompt(True)
        assert "Remember what's in them" not in prompt
        assert "read it again before quoting" in prompt

    def test_out_of_scope_checklist_items_are_dropped_not_marked_done(self):
        """The live run marked "Add authentication" done with nothing built; the
        checklist paragraph told it to."""
        prompt = build_system_prompt(
            _model(), None, None, inventory="Generator `x` produced 1 files:\n- a.py (1 bytes)",
            instructions="Build it", max_turns=10, gap_tasks=["Add authentication"],
        )
        assert "mark it done with a brief reason" not in prompt
        assert "task_list(action='drop', id=N, reason=...)" in prompt
        assert "never mark undone work done" in prompt
