"""Two reads of one turn evicting the oldest shown-read record at once.

A turn's tool calls run on a thread pool. ``_read_file_once`` evicted ``_shown_reads``
outside ``_bookkeeping_lock``, so a second read could delete the same oldest key first
and the first read then raised ``KeyError`` after the file had been read. The interleaving
is forced at the eviction instead of relying on timing.
"""

import threading

from besser.spec_driven_agent.agent.tool_executor import _SHOWN_READS_MAX, ToolExecutor

# How long the interleaved read may take; it blocks for all of it once the eviction is locked.
_WAIT = 3.0


class _OnDelete(dict):
    """A dict that starts ``hook`` on another thread at its first deletion."""

    def __init__(self, *args, hook):
        super().__init__(*args)
        self.hook = hook
        self.workers: list[threading.Thread] = []

    def __delitem__(self, key):
        if not self.workers:
            worker = threading.Thread(target=self.hook)
            self.workers.append(worker)
            worker.start()
            worker.join(_WAIT)
        super().__delitem__(key)


def test_two_reads_evicting_the_oldest_shown_read_at_once(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path))
    filler = {(f"old{n}.py", 0, None): ("digest", "0:0", 1, 1) for n in range(_SHOWN_READS_MAX)}
    ex._shown_reads = _OnDelete(filler, hook=lambda: ex._read_file_once({"path": "b.py"}))

    result = ex._read_file_once({"path": "a.py"})

    for worker in ex._shown_reads.workers:
        worker.join(_WAIT)
        assert not worker.is_alive()
    assert "a = 1" in result["content"]
    assert len(ex._shown_reads) == _SHOWN_READS_MAX
    assert {("a.py", 0, None), ("b.py", 0, None)} <= set(ex._shown_reads)
