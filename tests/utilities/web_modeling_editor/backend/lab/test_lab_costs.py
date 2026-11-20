"""What runs cost: per change (``runs.jsonl``, D2 decision 17) and per month in the audit log (D4 decisions 14-15),
which outlives a dropped change."""

from __future__ import annotations

import pytest

from besser.utilities.web_modeling_editor.backend.models.lab import LabRunRecord
from tests.utilities.web_modeling_editor.backend.lab.test_lab_service import (  # noqa: F401 (lab_env: a fixture)
    lab_env,
    make_service,
    record,
    run,
    user,
)


def test_run_costs_sum_per_change(lab_env):  # noqa: F811 - lab_env is the imported fixture
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    other = service.workspace.create_draft(owner="ben")
    first = run(service, change.id, "ana", usd=1.25, step="plan")
    second = run(service, change.id, "ben", usd=0.5)
    run(service, other.id, "ben", usd=2.0)

    detail = service.change_detail(change.id)

    assert (detail.cost.usd, detail.cost.turns, detail.cost.runs, detail.cost.elapsedSeconds) == (1.75, 6, 2, 120.0)
    assert [(r.runId, r.login, r.usd) for r in detail.runs] == [(first, "ana", 1.25), (second, "ben", 0.5)]
    assert all(isinstance(r, LabRunRecord) for r in detail.runs)
    assert [(item.runId, item.costUsd) for item in detail.thread] == [(first, 1.25), (second, 0.5)]
    assert detail.requestedBy == ["ana", "ben"]
    listed = {summary.changeId: summary.cost.usd for summary in service.changes().changes}
    assert listed == {change.id: 1.75, other.id: 2.0}


def test_the_audit_month_and_the_meters_follow_the_runs(lab_env):  # noqa: F811
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    run(service, change.id, "ana", usd=1.25)
    ticket = service.begin_run(change.id, "r2", "ben")

    held = service.budget.meters("ben")
    assert (held.user.spentUsd, held.user.reservedUsd) == (0.0, ticket.cost_cap_usd)
    assert (held.instance.spentUsd, held.instance.reservedUsd) == (1.25, ticket.cost_cap_usd)

    service.end_run(ticket, status="done", record=record("r2", "ben", usd=0.5), done=None, error=None)

    audit = service.audit_view(None)
    assert audit.month == service.audit.current_month()
    assert audit.totalsByLogin == {"ana": 1.25, "ben": 0.5}
    runs = [entry for entry in audit.entries if entry.action == "run"]
    assert [(e.login, e.changeId, e.costUsd, e.step, e.stopReason) for e in runs] == [
        ("ana", change.id, 1.25, "build", "completed"), ("ben", change.id, 0.5, "build", "completed")]
    meters = service.budget.meters("ben")
    assert (meters.user.spentUsd, meters.user.reservedUsd) == (0.5, 0.0)
    assert (meters.instance.spentUsd, meters.instance.reservedUsd) == (1.75, 0.0)
    assert service.home(user("ana")).meters.user.spentUsd == 1.25


def test_a_dropped_change_keeps_its_cost_in_the_audit(lab_env):  # noqa: F811
    service = make_service(lab_env.workspace())
    change = service.workspace.create_draft(owner="ana")
    run(service, change.id, "ana", usd=3.0)

    service.drop(user("ana"), change.id)

    assert service.changes().changes == []
    audit = service.audit_view(None)
    assert audit.totalsByLogin == {"ana": 3.0}
    assert [entry.action for entry in audit.entries if entry.changeId == change.id] == ["run", "change_dropped"]
    assert service.budget.meters("ana").user.spentUsd == 3.0


def test_an_audit_month_that_is_not_a_month_is_refused(lab_env):  # noqa: F811
    with pytest.raises(ValueError):
        make_service(lab_env.workspace()).audit_view("2026-13")

