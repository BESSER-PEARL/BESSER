"""Executor bookkeeping shared by the tool calls of one turn.

The orchestrator runs a turn's independent tool calls on a thread pool, so
one call's ``_clear_rejections`` (after every successful write) iterated
``_rejected_edits`` while another call's ``_note_rejection`` inserted into it:
"dictionary changed size during iteration", raised AFTER the write was on
disk, so the model was told an applied edit had failed. Each test forces the
interleaving at the exact point instead of relying on timing.
"""
import threading

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

# How long the interleaved call may take; it blocks for all of it once the
# bookkeeping is locked.
_WAIT = 0.5


def _interleave(call):
    """Start ``call`` on another thread and give it ``_WAIT`` to finish."""
    worker = threading.Thread(target=call)
    worker.start()
    worker.join(_WAIT)
    return worker


class _OnIterate(dict):
    """A dict that runs ``hook`` once, after yielding its first key."""

    def __init__(self, *args, hook):
        super().__init__(*args)
        self.hook = hook
        self.workers = []

    def __iter__(self):
        for key in super().__iter__():
            yield key
            if not self.workers:
                self.workers.append(None)
                self.workers[0] = _interleave(self.hook)


class _OnDelete(dict):
    def __init__(self, *args, hook):
        super().__init__(*args)
        self.hook = hook
        self.workers = []

    def __delitem__(self, key):
        if not self.workers:
            self.workers.append(None)
            self.workers[0] = _interleave(self.hook)
        super().__delitem__(key)


def test_clearing_rejections_while_another_call_records_one(tmp_path):
    ex = ToolExecutor(workspace=str(tmp_path))
    ex._note_rejection("a.py", "old", "new")
    ex._note_rejection("c.py", "old", "new")
    ex._rejected_edits = _OnIterate(
        ex._rejected_edits, hook=lambda: ex._note_rejection("b.py", "x", "y"))

    ex._clear_rejections("a.py")

    for worker in ex._rejected_edits.workers:
        worker.join()
    assert set(ex._rejected_edits) == {("c.py", "old", "new"), ("b.py", "x", "y")}


def test_two_reads_evicting_the_oldest_view_at_once(tmp_path):
    ex = ToolExecutor(workspace=str(tmp_path))
    for n in range(128):
        ex._mint_read_view(str(tmp_path / "a.py"), "x", 1, n + 1)
    ex._read_views = _OnDelete(
        ex._read_views, hook=lambda: ex._mint_read_view(str(tmp_path / "b.py"), "y", 1, 1))

    ex._mint_read_view(str(tmp_path / "c.py"), "z", 1, 1)

    for worker in ex._read_views.workers:
        worker.join()
    assert len(ex._read_views) == 128


def test_reading_the_miss_count_while_another_path_misses(tmp_path):
    ex = ToolExecutor(workspace=str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    class _ItemsOnIterate(_OnIterate):
        def items(self):
            return ((key, self[key]) for key in self)

    ex._failed_modifies = _ItemsOnIterate(
        {"z.py": 1, "a.py": 2}, hook=lambda: ex._failed_modifies.__setitem__("b.py", 1))

    assert ex.consecutive_modify_misses("a.py") == 2
