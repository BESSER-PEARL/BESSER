"""The runtime probe reads back what it created.

A backend whose every GET handler raised was reported as verified and
completed: the probe counted a 2xx create as the whole result and never read
the record back, so nothing blocked on reads that every screen depends on.
After each create the probe now GETs the record's item route and the
collection route OpenAPI declares. A 5xx there is an observed failure; any
other answer to the item read leaves the create unverified, because an
owner-scoped read or a list that needs a query parameter can refuse correctly.
"""
from __future__ import annotations

import re

import pytest

from besser.spec_driven_agent.pipeline.constants import _RUNTIME_FAILED, _runtime_verdict
from besser.spec_driven_agent.pipeline.phase3_repair import Phase3RepairMixin, _finding_key
from besser.spec_driven_agent.validation.constructibility import (
    READ_PREFIX,
    UNVERIFIED_PREFIX,
    _issues_from_report,
    collect_constructibility_report,
)
from besser.spec_driven_agent.validation.issues import (
    _classify_issue,
    finding_families,
    is_completion_issue,
    unresolved_defects,
)
from besser.spec_driven_agent.validation.user_wording import user_sentence

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _scaffold(tmp_path) -> str:
    """A generator-fresh backend with one entity, Order. Returns the workspace root."""
    from besser.BUML.metamodel.structural import Class, DomainModel, IntegerType, Property, StringType
    from besser.generators.backend import BackendGenerator

    order = Class(name="Order", attributes={
        Property(name="reference", type=StringType), Property(name="quantity", type=IntegerType)})
    BackendGenerator(model=DomainModel(name="Shop", types={order}),
                     output_dir=str(tmp_path / "app" / "backend")).generate()
    return str(tmp_path)


def _break(tmp_path, handler: str, statement: str) -> None:
    """Make the generated GET handler ``handler`` run ``statement`` first."""
    router = tmp_path / "app" / "backend" / "routers" / "order.py"
    text = router.read_text(encoding="utf-8")
    head = re.search(rf"(?:async )?def {handler}\(.*\n", text).group(0)
    router.write_text(text.replace(head, head + f"    {statement}\n", 1), encoding="utf-8")


def test_a_backend_whose_reads_all_fail_is_an_observed_failure(tmp_path):
    workspace = _scaffold(tmp_path)
    _break(tmp_path, "get_order", 'raise RuntimeError("reads fail")')
    _break(tmp_path, "get_all_order", 'raise RuntimeError("reads fail")')

    report = collect_constructibility_report(workspace)

    [backend] = report["backends"]
    assert backend["entities"]["Order"]["verdict"] == "created"
    reads = [i for i in report["issues"] if i.startswith(READ_PREFIX)]
    assert sorted(re.search(r"GET (\S+) -", i).group(1) for i in reads) == ["/order/", "/order/{order_id}/"]
    for issue in reads:
        assert "RuntimeError: reads fail" in issue
        assert "Fix site: get_order" in issue or "Fix site: get_all_order" in issue
        assert is_completion_issue(_classify_issue(issue))
        assert unresolved_defects([issue]) == [issue]  # the run is incomplete, not "completed"
    assert _runtime_verdict(report["issues"]) == _RUNTIME_FAILED


def test_a_healthy_scaffold_reads_back_without_a_finding(tmp_path):
    report = collect_constructibility_report(_scaffold(tmp_path))
    assert report["issues"] == []
    assert "reads" not in report["backends"][0]["entities"]["Order"]


def test_an_item_read_that_refuses_leaves_the_create_unverified(tmp_path):
    workspace = _scaffold(tmp_path)
    _break(tmp_path, "get_order", 'raise HTTPException(status_code=404, detail="not yours")')

    [issue] = collect_constructibility_report(workspace)["issues"]

    assert issue.startswith(f"{UNVERIFIED_PREFIX} app/backend: POST /order/ - the Order it created could not "
                            "be read back: GET /order/{order_id}/ answered 404")
    assert not is_completion_issue(_classify_issue(issue))
    assert user_sentence(issue) == ("Creating order (POST /order/) worked, but the record could not be read back "
                                    "in the test run.")


def test_a_list_that_refuses_without_a_server_error_is_no_finding(tmp_path):
    workspace = _scaffold(tmp_path)
    _break(tmp_path, "get_all_order", 'raise HTTPException(status_code=422, detail="give a filter")')
    assert collect_constructibility_report(workspace)["issues"] == []


def _read_report() -> dict:
    return {"boot": "ok", "entities": {"Order": {
        "path": "/order/", "verdict": "created", "attempts": [{"status": 201, "outcome": "created"}],
        "reads": [{"route": "/order/", "status": 500, "body": "boom"}]}}}


def test_the_repair_loop_tracks_a_read_failure_by_its_route():
    [issue] = _issues_from_report(_read_report(), "b")
    assert _finding_key(issue) == ("read contract", "b", "/order/")
    assert finding_families(issue) == {("runtime", ""), ("dependent", "")}
    assert user_sentence(issue) == "Reading order (GET /order/) failed with a server error in the test run."


def test_a_read_failure_of_an_entity_first_created_by_the_attempt_is_newly_visible():
    """Fixing a create shows the reads it hid; the keep rule must not revert that attempt."""
    before = [{"boot": "ok", "entities": {"Order": {"path": "/order/", "verdict": "crashed"}}}]
    after = [{"backend": "b", **_read_report()}]
    [issue] = _issues_from_report(_read_report(), "b")
    finding, _entity = Phase3RepairMixin._newly_reachable(before, after)
    assert finding(issue)
    created_before = [{"boot": "ok", "entities": {"Order": {"path": "/order/", "verdict": "created"}}}]
    assert not Phase3RepairMixin._newly_reachable(created_before, after)[0](issue)
