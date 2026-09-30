"""Concurrent trace writes must never tear a line.

Run qwen1's ``.besser_trace.jsonl`` holds a fragment line,
``ocks_in_turn": 6}}``, written during a turn with six parallel tool calls:
each record went out as two text-mode writes (the JSON, then the newline)
through its own append handle, and Windows emulates append with seek + write,
so concurrent records overwrote and interleaved each other.
"""
import json
import threading

from besser.spec_driven_agent.state.tracing import TraceWriter

THREADS = 12
RECORDS = 60


def test_parallel_writers_leave_only_whole_lines(tmp_path):
    writer = TraceWriter(str(tmp_path), run_id="r")
    # Larger than the 8 KiB text buffer, so one record needs several writes.
    detail = "x" * 20_000
    start = threading.Barrier(THREADS)

    def work(worker):
        start.wait()
        for n in range(RECORDS):
            writer.write("tool_call", worker=worker, n=n, detail=detail, blocks_in_turn=6)

    threads = [threading.Thread(target=work, args=(w,)) for w in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with open(writer.path, "rb") as fh:
        lines = fh.read().split(b"\n")
    assert lines[-1] == b"", "the file must end on a complete line"
    records = []
    for number, line in enumerate(lines[:-1], 1):
        try:
            records.append(json.loads(line))
        except ValueError:
            raise AssertionError(f"line {number} is torn: {line[:80]!r}...") from None
    assert len(records) == THREADS * RECORDS
    assert {(r["payload"]["worker"], r["payload"]["n"]) for r in records} == {
        (w, n) for w in range(THREADS) for n in range(RECORDS)
    }
