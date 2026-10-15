"""The scaffold-map A/B harness (tests/live/scaffold_map_ab.py) works end to end offline.

The harness itself makes paid LLM calls; its ``--dry-run`` swaps in an LLM that
ends every turn, so the pipeline, the functional probes, the metrics and the
summary run without network. On the bare scaffold every requested feature must
be judged missing, in both arms: a probe that passed there would count work the
agent never did.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[1] / "live" / "scaffold_map_ab.py"


@pytest.fixture(scope="module")
def ab():
    spec = importlib.util.spec_from_file_location("scaffold_map_ab", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_a_dry_run_judges_the_bare_scaffold_in_both_arms(ab, tmp_path, monkeypatch):
    monkeypatch.delenv("BESSER_LLM_SCAFFOLD_MAP", raising=False)
    assert ab.main(["--dry-run", "--repeats", "1", "--scenarios", "booking", "--out", str(tmp_path)]) == 0

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((tmp_path / "runs").glob("*.json"))]
    assert {r["arm"] for r in records} == {"map", "baseline"}
    for record in records:
        assert record["error"] is None
        assert record["checks"]["boots"] is True
        assert record["checks_passed"] == 0 and record["all_passed"] is False
        assert record["metrics"]["turns"] is not None
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## booking" in summary and "| check: cancel_route | 0/1 | 0/1 |" in summary
    # The switch is restored after the runs.
    assert "BESSER_LLM_SCAFFOLD_MAP" not in os.environ


def test_the_summary_compares_the_arms(ab):
    def record(arm, passed, cost, failed_edits):
        return {"scenario": "library", "arm": arm, "error": None, "all_passed": passed,
                "checks": {"boots": True, "stats_route": passed},
                "metrics": {"cost_usd": cost, "failed_edits": failed_edits, "placement_pitfalls": []}}

    text = ab.summarize([record("map", True, 1.0, 0), record("map", True, 2.0, 1),
                         record("baseline", False, 3.0, 4)])
    assert "| all checks pass | 2/2 | 0/1 |" in text
    assert "| check: stats_route | 2/2 | 0/1 |" in text
    assert "| mean cost_usd | 1.5 | 3.0 |" in text
    assert "| mean failed_edits | 0.5 | 4 |" in text


def test_a_missing_backend_is_a_failed_check_not_a_crash(ab, tmp_path):
    assert ab.run_checks(ab.SCENARIOS["library"], tmp_path) == {"boots": False, "error": "no main_api.py"}
