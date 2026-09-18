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
        for scaffold in (True, False):
            prompt = _prompt(scaffold)
            assert "three or more changes" not in prompt
            assert "is cheaper than a chain" not in prompt
            assert "Never rewrite a file from memory" in prompt

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
