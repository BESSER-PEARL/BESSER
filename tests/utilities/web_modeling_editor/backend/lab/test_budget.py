"""The lab's monthly caps: a run reserves its cost cap, so runs at once cannot overshoot a cap."""

import threading
import time
from datetime import datetime, timezone

import pytest

from besser.utilities.web_modeling_editor.backend.models.lab import LabMeter, LabMeters
from besser.utilities.web_modeling_editor.backend.services.lab.audit import AuditEntry, AuditLog
from besser.utilities.web_modeling_editor.backend.services.lab.budget import MIN_RUN_BUDGET_USD, LabBudget
from besser.utilities.web_modeling_editor.backend.services.lab.service import LabRefusal

NOV_15 = datetime(2026, 11, 15, 12, tzinfo=timezone.utc).timestamp()
DEC_1 = datetime(2026, 12, 1, 0, 0, 1, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock(NOV_15)


@pytest.fixture
def audit(tmp_path, clock):
    return AuditLog(str(tmp_path), clock=clock)


def _spend(audit: AuditLog, login: str, usd: float, at: float = NOV_15) -> None:
    audit.append(AuditEntry(at=at, login=login, action="run", change_id="a1b2c3d4e5f6", run_id="r" * 32,
                            step="build", cost_usd=usd, stop_reason="completed"))


def _budget(audit, per_run=5.0, user=75.0, instance=300.0) -> LabBudget:
    return LabBudget(audit, per_run_usd=per_run, user_monthly_usd=user, instance_monthly_usd=instance)


def test_the_minimum_is_a_quarter_dollar():
    assert MIN_RUN_BUDGET_USD == 0.25


def test_a_run_reserves_the_per_run_cap_when_both_months_have_room(audit):
    assert _budget(audit).reserve("ana") == 5.0


def test_a_run_reserves_what_is_left_of_the_users_month(audit):
    _spend(audit, "ana", 72.0)

    assert _budget(audit).reserve("ana") == pytest.approx(3.0)


def test_a_run_reserves_what_is_left_of_the_instances_month(audit):
    _spend(audit, "jordi", 298.0)

    assert _budget(audit, user=1000.0).reserve("ana") == pytest.approx(2.0)


def test_a_user_under_the_minimum_is_refused_user_cap(audit):
    _spend(audit, "ana", 74.9)

    with pytest.raises(LabRefusal) as refused:
        _budget(audit).reserve("ana")

    assert refused.value.code == "user_cap"
    assert refused.value.status == 409
    assert "$74.90 of your $75.00" in refused.value.message
    assert "1 December" in refused.value.message
    # Someone else still runs.
    assert _budget(audit).reserve("jordi") == 5.0


def test_an_instance_under_the_minimum_is_refused_instance_cap(audit):
    _spend(audit, "jordi", 299.9)

    with pytest.raises(LabRefusal) as refused:
        _budget(audit, user=1000.0).reserve("ana")

    assert refused.value.code == "instance_cap"
    assert refused.value.status == 409
    assert "$300.00" in refused.value.message


def test_reservations_count_against_the_caps_until_settled(audit):
    budget = _budget(audit, per_run=5.0, user=10.0)

    first = budget.reserve("ana")
    second = budget.reserve("ana")
    with pytest.raises(LabRefusal) as refused:
        budget.reserve("ana")
    assert (first, second, refused.value.code) == (5.0, 5.0, "user_cap")
    assert "held by your runs in progress" in refused.value.message

    # The run ends: its cost is in the audit log first, then the reservation is released.
    _spend(audit, "ana", 1.5)
    budget.settle("ana", first, 1.5)

    assert budget.reserve("ana") == pytest.approx(3.5)


def test_concurrent_reservations_respect_cap(tmp_path, clock):
    class SlowAudit(AuditLog):
        """Reads its totals slowly, so a reservation that is not atomic overshoots."""

        def spent(self, login=None):
            value = super().spent(login)
            time.sleep(0.01)
            return value

    budget = _budget(SlowAudit(str(tmp_path), clock=clock), per_run=5.0, user=12.0, instance=300.0)
    barrier = threading.Barrier(8)
    granted, refused = [], []

    def reserve() -> None:
        barrier.wait()
        try:
            granted.append(budget.reserve("ana"))
        except LabRefusal as refusal:
            refused.append(refusal.code)

    threads = [threading.Thread(target=reserve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(granted) <= 12.0
    assert sorted(granted) == [2.0, 5.0, 5.0]
    assert refused == ["user_cap"] * 5


def test_settle_warns_when_a_run_cost_more_than_it_reserved(audit, caplog):
    budget = _budget(audit)
    reserved = budget.reserve("ana")

    with caplog.at_level("WARNING"):
        budget.settle("ana", reserved, reserved + 0.4)

    assert any("ana" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)


def test_meters_show_spent_reserved_and_caps(audit):
    _spend(audit, "ana", 2.0)
    _spend(audit, "jordi", 3.0)
    budget = _budget(audit)
    budget.reserve("ana")
    budget.reserve("jordi")

    assert budget.meters("ana") == LabMeters(
        month="2026-11",
        user=LabMeter(spentUsd=2.0, reservedUsd=5.0, capUsd=75.0),
        instance=LabMeter(spentUsd=5.0, reservedUsd=10.0, capUsd=300.0),
    )


def test_a_new_month_resets_the_meter(audit, clock):
    _spend(audit, "ana", 75.0)
    budget = _budget(audit)
    with pytest.raises(LabRefusal):
        budget.reserve("ana")

    clock.now = DEC_1

    assert budget.reserve("ana") == 5.0
    assert budget.meters("ana").month == "2026-12"
    assert budget.meters("ana").user.spentUsd == 0.0
