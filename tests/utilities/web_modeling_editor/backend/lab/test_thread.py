"""A change's thread: messages, runs, pull requests and notices in ``changes/<id>/thread.jsonl``."""

import json

import pytest

from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabChangeCost,
    LabDone,
    LabPlan,
    LabPullRequest,
    LabRunRecord,
    LabThreadMessage,
    LabThreadNotice,
    LabThreadPullRequests,
    LabThreadRun,
)
from besser.utilities.web_modeling_editor.backend.services.lab.thread import (
    MAX_THREAD_MODEL_BYTES,
    ChangeThread,
)

CHANGE = "a1b2c3d4e5f6"
RUN_1 = "1" * 32
RUN_2 = "2" * 32
RUN_3 = "3" * 32


@pytest.fixture
def thread(tmp_path):
    return ChangeThread(str(tmp_path), CHANGE)


def _path(tmp_path):
    return tmp_path / "changes" / CHANGE / "thread.jsonl"


def _lines(tmp_path) -> list[dict]:
    with open(_path(tmp_path), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _done(step: str = "plan") -> LabDone:
    plan = LabPlan(kind="other", title="Note element", summary="s", steps=["a"], repos=["frontend"], questions=[])
    return LabDone(step=step, changeId=CHANGE, title="Note element", kind="other", repoChanges=[], checks=[],
                   checklist=[], summary="Planned.", stopReason="completed",
                   cost=LabChangeCost(usd=0.4, turns=3, elapsedSeconds=20.0, runs=1),
                   plan=plan if step == "plan" else None, modelProposal=None)


def _record(run_id: str, usd: float, step: str = "plan", started_at: float = 1.0) -> LabRunRecord:
    return LabRunRecord(runId=run_id, step=step, login="ana", model="m", usd=usd, turns=3, elapsedSeconds=20.0,
                        stopReason="completed", startedAt=started_at)


def test_a_new_change_has_an_empty_thread(thread):
    assert thread.items({}) == []


@pytest.mark.parametrize("change_id", ["../../etc", "A1B2C3D4E5F6", "a1b2", ""])
def test_a_malformed_change_id_is_refused(tmp_path, change_id):
    with pytest.raises(ValueError):
        ChangeThread(str(tmp_path), change_id)


def test_each_kind_of_line_is_written_as_the_design_shows(tmp_path, thread):
    message_id = thread.add_message(login="ana", text="Add a Note element", action="auto", model={"a": 1})
    thread.run_started(RUN_1, "ana")
    thread.run_finished(RUN_1, "ana", status="failed", lab=None, error={"code": "api_error", "message": "x"})
    pr = LabPullRequest(repo="besser", number=812, url="https://github.com/o/r/pull/812", branch="lab/a/b",
                        base="development")
    thread.add_pull_requests("jordi", [pr], "Checks are running.")
    thread.add_notice("ana", "Updated from development: 14 commits, no conflicts.")

    lines = _lines(tmp_path)
    assert all(line["v"] == 1 and isinstance(line["at"], float) for line in lines)
    assert [{k: v for k, v in line.items() if k not in ("v", "at")} for line in lines] == [
        {"type": "message", "id": message_id, "login": "ana", "text": "Add a Note element", "action": "auto",
         "model": {"a": 1}},
        {"type": "run", "runId": RUN_1, "login": "ana", "status": "running"},
        {"type": "run", "runId": RUN_1, "login": "ana", "status": "failed", "lab": None,
         "error": {"code": "api_error", "message": "x"}},
        {"type": "pull_requests", "login": "jordi", "pullRequests": [pr.model_dump(mode="json")],
         "checksNote": "Checks are running."},
        {"type": "notice", "login": "ana", "text": "Updated from development: 14 commits, no conflicts."},
    ]
    assert len(message_id) == 12 and int(message_id, 16) >= 0


def test_items_parse_every_kind(thread):
    thread.add_message(login="ana", text="hi", action="plan", model=None)
    thread.add_pull_requests("jordi", [], "")
    thread.add_notice("ana", "n")

    items = thread.items({})

    assert [type(item) for item in items] == [LabThreadMessage, LabThreadPullRequests, LabThreadNotice]
    assert items[0].text == "hi" and items[0].action == "plan" and items[0].model is None


def test_run_folding_the_last_line_of_a_run_wins_in_the_place_of_its_first(thread):
    thread.add_message(login="ana", text="plan it", action="auto", model=None)
    thread.run_started(RUN_1, "ana")
    thread.add_notice("jordi", "meanwhile")
    thread.run_finished(RUN_1, "ana", status="done", lab=_done(), error=None)

    items = thread.items({})

    assert [item.type for item in items] == ["message", "run", "notice"]
    run = items[1]
    assert isinstance(run, LabThreadRun)
    assert run.status == "done"
    assert run.lab == _done()
    assert run.error is None


def test_a_running_run_has_no_cost(thread):
    thread.run_started(RUN_1, "ana")

    (run,) = thread.items({})

    assert run.status == "running" and run.lab is None and run.costUsd is None and run.planVersion is None


def test_items_are_joined_with_run_cost_and_plan_version(thread):
    thread.run_started(RUN_1, "ana")
    thread.run_finished(RUN_1, "ana", status="done", lab=_done("plan"), error=None)
    thread.run_started(RUN_2, "ana")
    thread.run_finished(RUN_2, "ana", status="done", lab=_done("plan"), error=None)
    thread.run_started(RUN_3, "jordi")
    thread.run_finished(RUN_3, "jordi", status="done", lab=_done("build"), error=None)
    records = {RUN_1: _record(RUN_1, 0.4), RUN_2: _record(RUN_2, 0.6), RUN_3: _record(RUN_3, 1.82, step="build")}

    runs = thread.items(records)

    assert [(r.costUsd, r.planVersion) for r in runs] == [(0.4, 1), (0.6, 2), (1.82, None)]


def test_a_plan_run_that_did_not_finish_is_not_a_plan_version(thread):
    thread.run_started(RUN_1, "ana")
    thread.run_finished(RUN_1, "ana", status="cancelled", lab=None, error=None)
    thread.run_started(RUN_2, "ana")
    thread.run_finished(RUN_2, "ana", status="done", lab=_done("plan"), error=None)

    runs = thread.items({RUN_1: _record(RUN_1, 0.1), RUN_2: _record(RUN_2, 0.3)})

    assert [(r.status, r.costUsd, r.planVersion) for r in runs] == [("cancelled", 0.1, None), ("done", 0.3, 1)]


def test_items_keep_the_newest_up_to_the_limit_with_plan_versions_counted_from_the_start(thread):
    thread.run_started(RUN_1, "ana")
    thread.run_finished(RUN_1, "ana", status="done", lab=_done("plan"), error=None)
    for index in range(5):
        thread.add_notice("ana", f"n{index}")
    thread.run_started(RUN_2, "ana")
    thread.run_finished(RUN_2, "ana", status="done", lab=_done("plan"), error=None)

    items = thread.items({RUN_1: _record(RUN_1, 0.4), RUN_2: _record(RUN_2, 0.6)}, limit=3)

    assert [getattr(i, "text", None) for i in items] == ["n3", "n4", None]
    assert items[-1].planVersion == 2


def test_the_model_cap_drops_a_larger_class_diagram_with_a_notice(tmp_path, thread, caplog):
    assert MAX_THREAD_MODEL_BYTES == 512 * 1024
    big = {"elements": "x" * MAX_THREAD_MODEL_BYTES}

    with caplog.at_level("WARNING"):
        thread.add_message(login="ana", text="with a big model", action="auto", model=big)

    message, notice = thread.items({})
    assert message.text == "with a big model" and message.model is None
    assert isinstance(notice, LabThreadNotice) and "512 KB" in notice.text and notice.login == "ana"
    assert len(_lines(tmp_path)) == 2
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_a_class_diagram_within_the_cap_is_kept(thread):
    model = {"elements": "x" * 1000}

    thread.add_message(login="ana", text="t", action="auto", model=model)

    (message,) = thread.items({})
    assert message.model == model


def test_a_torn_last_line_is_skipped_with_a_warning_and_the_next_line_stays_whole(tmp_path, thread, caplog):
    thread.add_notice("ana", "before")
    with open(_path(tmp_path), "ab") as handle:
        handle.write(b'{"v":1,"type":"notice","at":1.0,"login":"ana","te')   # a crash mid-write

    with caplog.at_level("WARNING"):
        assert [i.text for i in thread.items({})] == ["before"]
    assert any("thread.jsonl" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)

    thread.add_notice("ana", "after")
    assert [i.text for i in thread.items({})] == ["before", "after"]


def test_a_line_of_an_unknown_version_or_shape_is_skipped_with_a_warning(tmp_path, thread, caplog):
    thread.add_notice("ana", "kept")
    with open(_path(tmp_path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"v": 2, "type": "notice", "at": 1.0, "login": "a", "text": "future"}) + "\n")
        handle.write(json.dumps({"v": 1, "type": "poll", "at": 1.0, "login": "a"}) + "\n")

    with caplog.at_level("WARNING"):
        assert [i.text for i in thread.items({})] == ["kept"]
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 2


def test_run_finished_refuses_running_as_an_end(thread):
    with pytest.raises(ValueError):
        thread.run_finished(RUN_1, "ana", status="running", lab=None, error=None)
