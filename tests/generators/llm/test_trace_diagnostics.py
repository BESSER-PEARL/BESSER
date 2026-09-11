"""The trace must record WHY a tool failed, not just that it did.

Written after a post-mortem that couldn't be completed: a run on 2026-09-11
burned 11 of its 80 turns on failed ``modify_file`` calls, and the trace held
only ``status: error`` with no reason. The single error message recoverable came
from the checkpoint, because compaction had already discarded the rest. The
question "is the harness misbehaving, or is the model?" was unanswerable from a
finished run — which is the one question a trace exists to answer.
"""

import json

from besser.generators.llm.orchestrator import LLMOrchestrator as O


# ------------------------------------------------------------- extraction


def test_the_failure_reason_is_captured():
    result = json.dumps({
        "error": "old_text not found in frontend/src/api.js. File has 150 "
                 "lines, 3458 chars. Make sure old_text matches exactly.",
    })
    diag = O._trace_diagnostics(result)
    assert "old_text not found" in diag["error"]


def test_the_matched_edit_tier_is_captured():
    """Which ladder tier matched tells us whether models are hitting anchors
    exactly or leaning on the flexible whitespace path."""
    diag = O._trace_diagnostics(json.dumps({
        "status": "modified", "path": "app.py", "matched_by": "flexible",
    }))
    assert diag["matched_by"] == "flexible"


def test_recovery_hints_are_captured():
    diag = O._trace_diagnostics(json.dumps({
        "error": "old_text not found",
        "did_you_mean": "def compute():\n    return 42",
        "advice": "read_file the region first",
        "note": "new_text is ALREADY present",
    }))
    assert set(("error", "did_you_mean", "advice", "note")) <= set(diag)


def test_per_write_diagnostics_are_counted_not_copied():
    """The bodies are already in the tool_result the model saw; the trace only
    needs to know how many fired, so a noisy write stays greppable."""
    diag = O._trace_diagnostics(json.dumps({
        "status": "written",
        "diagnostics": [{"source": "pyflakes", "message": f"undefined name x{i}"}
                        for i in range(7)],
    }))
    assert diag["diagnostics_count"] == 7
    assert "diagnostics" not in diag


# ------------------------------------------------------------- robustness


def test_long_values_are_truncated():
    """``did_you_mean`` can carry a whole file excerpt — a trace line must stay
    readable and the file must not balloon."""
    diag = O._trace_diagnostics(json.dumps({"did_you_mean": "x" * 5000}))
    assert len(diag["did_you_mean"]) <= O._TRACE_DIAG_MAX_CHARS + 20
    assert diag["did_you_mean"].endswith("[truncated]")


def test_a_clean_success_adds_nothing():
    assert O._trace_diagnostics(json.dumps({"status": "written", "path": "a.py"})) == {}


def test_malformed_results_never_raise():
    """A broken tool result must not break the run it is describing."""
    for bad in ("", "not json", "null", "[1,2,3]", '"a string"', "{oops"):
        assert O._trace_diagnostics(bad) == {}


def test_non_string_values_are_stringified():
    diag = O._trace_diagnostics(json.dumps({"error": {"code": 42}}))
    assert "42" in diag["error"]


def test_empty_values_are_dropped():
    """An empty error is not a failure reason — don't pollute the trace."""
    assert O._trace_diagnostics(json.dumps({"error": "", "note": None})) == {}
