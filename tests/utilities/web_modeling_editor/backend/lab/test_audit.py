"""The lab's audit log: one JSON-lines file per UTC month, the ledger behind the monthly caps."""

import json
import os
import threading
from datetime import datetime, timezone

import pytest

from besser.utilities.web_modeling_editor.backend.services.lab.audit import AuditEntry, AuditLog

NOV_15 = datetime(2026, 11, 15, 12, tzinfo=timezone.utc).timestamp()
NOV_30_LATE = datetime(2026, 11, 30, 23, 59, 59, tzinfo=timezone.utc).timestamp()
DEC_1 = datetime(2026, 12, 1, 0, 0, 1, tzinfo=timezone.utc).timestamp()


class Clock:
    """A clock the test moves."""

    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _run(login: str, usd: float, at: float, **fields) -> AuditEntry:
    return AuditEntry(at=at, login=login, action="run", change_id="a1b2c3d4e5f6", run_id="r" * 32,
                      step="build", cost_usd=usd, stop_reason="completed", model="m", seconds=12.0, **fields)


def _lines(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_an_entry_is_one_line_in_its_utc_month_file(tmp_path):
    log = AuditLog(str(tmp_path), clock=Clock(NOV_15))

    log.append(_run("ana", 1.82, NOV_15))

    assert _lines(tmp_path / "audit" / "2026-11.jsonl") == [{
        "v": 1, "at": NOV_15, "login": "ana", "action": "run", "changeId": "a1b2c3d4e5f6", "runId": "r" * 32,
        "step": "build", "costUsd": 1.82, "stopReason": "completed", "model": "m", "seconds": 12.0, "detail": "",
        "ownKey": False,
    }]


def test_entries_read_back_a_month(tmp_path):
    log = AuditLog(str(tmp_path), clock=Clock(NOV_15))
    created = AuditEntry(at=NOV_15, login="jordi", action="change_created", change_id="a1b2c3d4e5f6")
    log.append(created)
    log.append(_run("ana", 0.5, NOV_15))

    assert log.entries("2026-11") == [created, _run("ana", 0.5, NOV_15)]
    assert log.entries("2026-10") == []


@pytest.mark.parametrize("month", ["../x", "2026-13", "2026-1", "", "2026-11/../../etc"])
def test_a_malformed_month_is_refused(tmp_path, month):
    with pytest.raises(ValueError):
        AuditLog(str(tmp_path)).entries(month)


def test_spent_sums_this_months_runs_per_login_and_for_the_instance(tmp_path):
    log = AuditLog(str(tmp_path), clock=Clock(NOV_15))
    log.append(_run("ana", 1.0, NOV_15))
    log.append(_run("jordi", 2.5, NOV_15))
    log.append(_run("ana", 0.25, NOV_15))
    # Not a run: no cost counts, even if one were set.
    log.append(AuditEntry(at=NOV_15, login="ana", action="run_refused", detail="user_cap"))

    assert log.spent("ana") == pytest.approx(1.25)
    assert log.spent("jordi") == pytest.approx(2.5)
    assert log.spent("nobody") == 0.0
    assert log.spent() == pytest.approx(3.75)


def test_this_months_totals_are_loaded_at_start(tmp_path):
    AuditLog(str(tmp_path), clock=Clock(NOV_15)).append(_run("ana", 4.0, NOV_15))

    restarted = AuditLog(str(tmp_path), clock=Clock(NOV_15))

    assert restarted.spent("ana") == pytest.approx(4.0)
    assert restarted.spent() == pytest.approx(4.0)


def test_a_month_boundary_resets_the_meter(tmp_path):
    clock = Clock(NOV_30_LATE)
    log = AuditLog(str(tmp_path), clock=clock)
    log.append(_run("ana", 3.0, NOV_30_LATE))
    assert log.spent("ana") == pytest.approx(3.0)
    assert log.current_month() == "2026-11"

    clock.now = DEC_1

    assert log.current_month() == "2026-12"
    assert log.spent("ana") == 0.0
    assert log.spent() == 0.0
    log.append(_run("ana", 1.0, DEC_1))
    assert log.spent("ana") == pytest.approx(1.0)
    assert len(log.entries("2026-11")) == 1 and len(log.entries("2026-12")) == 1


def test_a_torn_line_is_skipped_with_a_warning(tmp_path, caplog):
    log = AuditLog(str(tmp_path), clock=Clock(NOV_15))
    log.append(_run("ana", 1.0, NOV_15))
    path = tmp_path / "audit" / "2026-11.jsonl"
    with open(path, "ab") as handle:
        handle.write(b'{"v":1,"at":1763,"login":"ana","act')   # a crash in the middle of a write

    with caplog.at_level("WARNING"):
        restarted = AuditLog(str(tmp_path), clock=Clock(NOV_15))
        assert restarted.spent("ana") == pytest.approx(1.0)
        assert [e.login for e in restarted.entries("2026-11")] == ["ana"]
    assert any("2026-11.jsonl" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)

    # The next entry starts on a line of its own, so only the torn line is lost.
    restarted.append(_run("jordi", 2.0, NOV_15))
    assert [e.login for e in restarted.entries("2026-11")] == ["ana", "jordi"]
    assert AuditLog(str(tmp_path), clock=Clock(NOV_15)).spent() == pytest.approx(3.0)


def test_concurrent_appends_keep_every_line_whole(tmp_path):
    log = AuditLog(str(tmp_path), clock=Clock(NOV_15))
    barrier = threading.Barrier(8)

    def append_many(index: int) -> None:
        barrier.wait()
        for _ in range(25):
            log.append(_run(f"user{index}", 0.01, NOV_15, detail="x" * 2000))

    threads = [threading.Thread(target=append_many, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(_lines(tmp_path / "audit" / "2026-11.jsonl")) == 200
    assert log.spent() == pytest.approx(2.0)


def test_the_file_is_opened_for_append(tmp_path, monkeypatch):
    opened = []
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda path, flags, *a, **kw: opened.append(flags) or real_open(path, flags, *a, **kw))

    AuditLog(str(tmp_path), clock=Clock(NOV_15)).append(_run("ana", 1.0, NOV_15))

    assert opened and all(flags & os.O_APPEND for flags in opened)
