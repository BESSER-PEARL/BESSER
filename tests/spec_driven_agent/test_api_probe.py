"""Declarative workflow feedback runs only on disposable app/database copies."""
import hashlib
import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.validation import api_probe

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")

FIXTURE = Path(__file__).parent / "fixtures" / "run_9a6063ed"
ROOM = {"description": "Scenario", "roomNumber": "101", "standardPrice": 100, "capacity": 2}


def copied_backend(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return tmp_path / "web_app" / "backend"


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_real_sequence_references_and_failed_business_assertion_are_not_false_success(tmp_path, monkeypatch):
    copied_backend(tmp_path)
    before = source_hashes(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    resolved = api_probe._resolve_references({"id": "{{0.room.id}}", "ids": ["{{0.room.id}}"]}, [{"room": {"id": 7}}])
    assert resolved == {"id": 7, "ids": [7]} and type(resolved["id"]) is int
    assert api_probe._resolve_references("{{0}}", [None]) is None
    assert not api_probe._same_literal({"paid": True}, {"paid": 1})
    report = api_probe.probe_api_scenario(str(tmp_path), [
        {"method": "POST", "path": "/room/", "json": ROOM, "expected_status": [200, 201], "expected_fields": {"room.capacity": 2}},
        {"method": "PUT", "path": "/room/{{0.room.id}}/", "json": dict(ROOM, capacity=3, standardPrice="{{0.room.standardPrice}}"),
         "expected_fields": {"room.capacity": 99}},
        {"method": "GET", "path": "/room/{{0.room.id}}/", "expected_fields": {"room.capacity": 3}},
        {"method": "GET", "path": "/room/{{1.no_such_field}}/"},
        {"method": "GET", "path": "/room/count/", "expected_fields": {"count": 1}},
    ])
    assert report["boot"] == "ok", report
    assert report["status"] == "failed"
    assert [entry["index"] for entry in report["assertion_failures"]] == [1, 3]
    assert report["responses"][2]["failures"] == report["responses"][4]["failures"] == []
    assert report["responses"][1]["json"]["room"]["standardPrice"] == 100
    assert source_hashes(tmp_path) == before
    # A new invocation starts with a fresh DB, not the rows from the first run.
    clean = api_probe.probe_api_scenario(str(tmp_path), [{"method": "GET", "path": "/room/count/", "expected_fields": {"count": 0}}])
    assert clean["status"] == "passed", clean
    assert api_probe.confirmed_create_paths(clean) == set(), "a health/count read is not creation evidence"
    # This scenario's POST created a Room (201) and request 2 read that same id
    # back. Requests 1 and 3 failed on a deliberately wrong expected value and an
    # unresolvable reference -- neither says anything about whether Room can be
    # created, and the create check re-derives its verdict from the raw
    # responses rather than trusting any assertion in the scenario.
    #
    # Gating this on the scenario's overall verdict made the function inert:
    # tools.py tells the model to "Test happy paths AND invalid input/state
    # transitions", and across 784 recorded scenarios from 177 runs, 583 of the
    # 784 (74% of create-bearing ones) carry at least one request the model
    # deliberately expects to fail. Following our own instruction emptied this
    # set, so "create unverified:" could not be retired and was promoted to a
    # blocker against apps that demonstrably created the record.
    assert api_probe.confirmed_create_paths(report) == {"/room"}, "a peer failure discarded proof the app created and read back a Room"
    persisted = api_probe.probe_api_scenario(str(tmp_path), [
        {"method": "POST", "path": "/room/", "json": ROOM},
        {"method": "GET", "path": "/room/{{0.room.id}}/"},
    ])
    assert persisted["status"] == "passed", persisted
    assert api_probe.confirmed_create_paths(persisted) == {"/room"}
    no_read = dict(persisted, responses=persisted["responses"][:1])
    assert api_probe.confirmed_create_paths(no_read) == set()
    unrelated = copy.deepcopy(persisted)
    unrelated["responses"][1]["path"] = "/health/"
    assert api_probe.confirmed_create_paths(unrelated) == set()
    assert source_hashes(tmp_path) == before


def test_invalid_requests_and_external_paths_never_launch_a_child(tmp_path, monkeypatch):
    monkeypatch.setattr(api_probe.subprocess, "run", lambda *a, **k: pytest.fail("invalid input launched a child"))
    invalid = [[], [{"method": "GET", "path": "/"}] * 21,
               [{"method": "GET", "path": "https://example.com/"}],
               [{"method": "GET", "path": "//example.com/"}],
               [{"method": "GET", "path": "/%2fexample.com/"}],
               [{"method": "GET", "path": "/../secret"}],
               [{"method": "GET", "path": "/", "headers": {"Authorization": "secret"}}],
               [{"method": "RUN", "path": "/"}],
               [{"method": "GET", "path": "/", "expected_status": True}],
               [{"method": "GET", "path": "/", "expected_fields": []}]]
    for requests in invalid:
        report = api_probe.probe_api_scenario(str(tmp_path), requests)
        assert report["status"] == "error" and report["boot"] == "not_started", report


def test_boot_errors_and_external_side_effects_are_reported_without_mutating_original(tmp_path):
    backend = copied_backend(tmp_path)
    marker = tmp_path / "untouched.txt"
    marker.write_text("keep", encoding="utf-8")
    boot_cases = [
        "raise RuntimeError('broken startup')\n",
        f"open({str(marker)!r}, 'w').write('changed')\n",
        f"import sqlite3\nsqlite3.connect({str(tmp_path / 'outside.db')!r})\n",
        "import socket\nsocket.create_connection(('example.com', 443))\n",
    ]
    for source in boot_cases:
        (backend / "main_api.py").write_text(source, encoding="utf-8")
        before = source_hashes(tmp_path)
        report = api_probe.probe_api_scenario(str(tmp_path), [{"method": "GET", "path": "/"}])
        assert report["status"] == "error" and report["boot"] == "boot_error", report
        assert source_hashes(tmp_path) == before
    assert marker.read_text(encoding="utf-8") == "keep"
    (backend / "main_api.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/crash/')\ndef crash():\n    raise ValueError('broken workflow')\n",
        encoding="utf-8",
    )
    before = source_hashes(tmp_path)
    failed = api_probe.probe_api_scenario(str(tmp_path), [{"method": "GET", "path": "/crash/"}])
    assert failed["boot"] == "ok" and failed["status"] == "failed", failed
    response = failed["responses"][0]
    assert response["status"] == 500 and response["error"]["type"] == "ValueError"
    assert response["error"]["locations"][-1] == {"path": "main_api.py", "line": 5, "function": "crash"}
    assert source_hashes(tmp_path) == before


def test_backend_selection_and_timeout_are_explicit_errors(tmp_path, monkeypatch):
    backend = copied_backend(tmp_path)
    shutil.copytree(backend, tmp_path / "another_backend")
    scenario = [{"method": "GET", "path": "/room/count/"}]
    ambiguous = api_probe.probe_api_scenario(str(tmp_path), scenario)
    assert ambiguous["status"] == "error" and len(ambiguous["backends"]) == 2
    assert api_probe.probe_api_scenario(str(tmp_path), scenario, backend="../elsewhere")["status"] == "error"
    before = source_hashes(tmp_path)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] <= 60
        assert kwargs["env"].get("OPENAI_API_KEY") is None
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(api_probe.subprocess, "run", timeout)
    report = api_probe.probe_api_scenario(str(tmp_path), scenario, backend="web_app/backend")
    assert report["status"] == "error" and report["boot"] == "timeout", report
    assert source_hashes(tmp_path) == before


def scenario_orchestrator(tmp_path, **kwargs):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.providers.llm_client import UsageTracker
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

    client = SimpleNamespace(model="mock-model", usage=UsageTracker("mock-model"))
    return LLMOrchestrator(llm_client=client, domain_model=DomainModel(name="Probe", types={Class(name="Room")}),
                           output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
                           enable_requirements_ledger=False, enable_toolchain_validation=False, **kwargs)


def test_orchestration_retains_workflows_reruns_changed_source_and_never_evicts_a_failure(tmp_path, monkeypatch):
    from besser.spec_driven_agent.pipeline.orchestrator import _classify_issue

    source = tmp_path / "state.py"
    source.write_text("ready = False\n", encoding="utf-8")
    calls = []

    def probe(output_dir, requests, *, backend=None):
        calls.append(copy.deepcopy(requests))
        if not requests:
            return api_probe._error("invalid scenario")
        passed = "True" in source.read_text(encoding="utf-8")
        return {"status": "passed" if passed else "failed", "boot": "ok", "responses": [],
                "assertion_failures": [] if passed else [{"index": 0, "message": "business assertion failed"}]}

    monkeypatch.setattr(api_probe, "probe_api_scenario", probe)
    orchestrator = scenario_orchestrator(tmp_path)
    args = {"requests": [{"method": "GET", "path": "/state/", "expected_fields": {"ready": True}}]}
    first = json.loads(orchestrator.executor.execute("test_api", args))
    assert first["status"] == "failed" and first["scenario_id"]
    assert len(orchestrator._api_scenarios) == 1
    orchestrator._test_api({"requests": [{"expected_fields": {"ready": True}, "path": "/state/", "method": "GET"}]})
    assert len(orchestrator._api_scenarios) == 1, "canonical JSON ordering must deduplicate scenarios"
    count = len(calls)
    issues = orchestrator._collect_api_scenario_issues()
    assert len(calls) == count, "unchanged source must not rerun retained workflows"
    assert len(issues) == 1 and issues[0].startswith("api scenario:")
    assert _classify_issue(issues[0]).severity == "blocker"

    source.write_text("ready = True\n", encoding="utf-8")
    assert orchestrator._collect_api_scenario_issues() == []
    assert len(calls) == count + 1
    source.write_text("ready = False\n", encoding="utf-8")
    assert orchestrator._collect_api_scenario_issues()
    for number in range(9):
        orchestrator._test_api(dict(args, scenario_id=f"workflow-{number}"))
    assert len(orchestrator._api_scenarios) == 10
    before = copy.deepcopy(orchestrator._api_scenarios)
    assert "error" in orchestrator._test_api(dict(args, scenario_id="eleventh"))
    assert orchestrator._api_scenarios == before, "the cap must not evict a failed workflow"

    orchestrator.enable_import_smoke_check = False
    source.write_text("ready = True\n", encoding="utf-8")
    count = len(calls)
    assert orchestrator._collect_api_scenario_issues(), "disabled runtime refresh cannot clear stored failures"
    assert len(calls) == count
    assert "error" in json.loads(orchestrator.executor.execute("test_api", args))
    fresh = scenario_orchestrator(tmp_path, enable_import_smoke_check=False)
    assert "error" in json.loads(fresh.executor.execute("test_api", args))
    assert not fresh._api_scenarios and len(calls) == count

    # A valid persisted create/read can discharge only its exact guessed-probe
    # unknown, never an actual crash or a different backend/route.
    from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
    from besser.spec_driven_agent.validation import constructibility
    unknown = "create unverified: web_app/backend: POST /room/ - guessed fixture was rejected"
    monkeypatch.setattr(orchestrator_module, "_import_smoke_issues", lambda _root: [])
    monkeypatch.setattr(constructibility, "collect_constructibility_report",
                        lambda _root, _model=None: {"issues": [unknown], "backends": []})
    coverage = scenario_orchestrator(tmp_path)
    assert any(unknown in issue for issue in coverage._collect_execution_issues())
    proof = {"status": "passed", "boot": "ok", "backend": "web_app/backend", "responses": [
        {"method": "POST", "path": "/room/", "status": 200, "json": {"room": {"id": 0, "name": "sample"}}},
        {"method": "GET", "path": "/room/0/", "status": 200, "json": {"room": {"id": 0, "name": "sample"}}},
    ]}
    coverage._api_scenarios["proof"] = {"scenario": args, "revision": coverage._workspace_revision(), "report": proof}
    assert coverage._collect_execution_issues() == []
    proof["backend"] = "another/backend"
    assert any(unknown in issue for issue in coverage._collect_execution_issues())
    proof["backend"] = "web_app/backend"
    monkeypatch.setattr(constructibility, "collect_constructibility_report",
                        lambda _root, _model=None: {"issues": [
                            "create contract: web_app/backend: POST /room/ - observed 500"],
                            "backends": []})
    # The boot probe is now cached per source revision - it costs up to 90s a
    # backend and Phase 3 re-validates after every fix attempt. Swapping the
    # probe's answer without touching a byte of source is something only a
    # test can do, so the test drops the cache the way a real edit would.
    coverage._runtime_probe_cache = None
    assert any("observed 500" in issue for issue in coverage._collect_execution_issues())


def test_orchestration_corrects_named_tests_explicitly_and_waits_for_batched_writes(tmp_path, monkeypatch):
    calls = []

    def probe(output_dir, requests, *, backend=None):
        calls.append(copy.deepcopy(requests))
        if requests == []:
            return api_probe._error("invalid scenario")
        passed = requests[0].get("expected_fields") == {"ready": True}
        if requests[0]["path"] == "/after-write/":
            assert (tmp_path / "ready.py").read_text(encoding="utf-8") == "ready = True\n"
            passed = True
        return {"status": "passed" if passed else "failed", "boot": "ok", "responses": [],
                "assertion_failures": [] if passed else [{"index": 0, "message": "wrong JSON path"}]}

    monkeypatch.setattr(api_probe, "probe_api_scenario", probe)
    orchestrator = scenario_orchestrator(tmp_path)
    wrong = {"requests": [{"method": "GET", "path": "/state/", "expected_fields": {"wrong.path": True}}]}
    first = orchestrator._test_api(wrong)
    scenario_id = first["scenario_id"]  # Even initially unnamed mistakes remain correctable.
    original = copy.deepcopy(next(iter(orchestrator._api_scenarios.values()))["scenario"])
    wrong["requests"][0]["expected_fields"] = {"ready": True}
    assert next(iter(orchestrator._api_scenarios.values()))["scenario"] == original, "retained assertions must not alias tool arguments"
    corrected = dict(wrong, scenario_id=scenario_id)
    count = len(calls)
    assert "error" in orchestrator._test_api(corrected)
    assert len(calls) == count, "unexplained correction must not run or replace the old scenario"
    before = copy.deepcopy(orchestrator._api_scenarios)
    listing = orchestrator._test_api({"action": "list"})
    assert listing["scenarios"][0]["scenario_id"] == scenario_id
    inspected = orchestrator._test_api({"action": "get", "scenario_id": scenario_id})
    assert inspected["requests"] == original["requests"] and inspected["current_revision"]
    inspected["requests"][0]["expected_fields"] = {"ready": True}
    assert orchestrator._api_scenarios == before and len(calls) == count
    issue = orchestrator._collect_api_scenario_issues()[0]
    assert scenario_id in issue and "action='get'" in issue and "original specification" in issue
    assert "error" in orchestrator._test_api({"action": "get", "scenario_id": "missing"})
    assert "error" in orchestrator._test_api({"action": "delete", "scenario_id": scenario_id})
    assert "error" in orchestrator._test_api({"scenario_id": "unknown-without-requests"})
    assert orchestrator._test_api({"scenario_id": scenario_id})["status"] == "failed"
    assert calls[-1] == original["requests"]
    orchestrator.enable_import_smoke_check = False
    assert orchestrator._test_api({"action": "get", "scenario_id": scenario_id})["requests"] == original["requests"]
    assert orchestrator._test_api({"action": "list"})["scenarios"]
    assert "error" in orchestrator._test_api({"scenario_id": scenario_id})
    orchestrator.enable_import_smoke_check = True
    report = orchestrator._test_api(dict(corrected, correction_reason="OpenAPI/actual response uses ready, not wrong.path"))
    assert report["status"] == "passed" and "correction" in report
    assert len(orchestrator._api_scenarios) == 1
    assert orchestrator._collect_api_scenario_issues() == []
    orchestrator._test_api({"scenario_id": "invalid", "requests": []})
    assert len(orchestrator._api_scenarios) == 1, "malformed scenarios must not become permanent obligations"

    orchestrator.executor.set_tasks([{"text": "Write the ready flag", "verify": lambda: (tmp_path / "ready.py").is_file()}])
    blocks = [
        SimpleNamespace(type="tool_use", id="done", name="task_list", input={"action": "done", "id": 1}),
        SimpleNamespace(type="tool_use", id="api", name="test_api", input={
            "scenario_id": "post-write", "requests": [{"method": "GET", "path": "/after-write/"}]}),
        SimpleNamespace(type="tool_use", id="write", name="write_file", input={"path": "ready.py", "content": "ready = True\n"}),
    ]
    results = orchestrator._execute_tool_blocks(blocks, turn=1)
    assert [result["tool_use_id"] for result in results] == ["done", "api", "write"]
    assert orchestrator.executor.task_snapshot()[0]["done"], "task verification must see the completed write batch"
    assert json.loads(results[1]["content"])["status"] == "passed"

    # Crash recovery keeps the accepted definitions, never a cached green report.
    from besser.spec_driven_agent.state.checkpoint import compute_fingerprint, load_checkpoint, save_checkpoint, restore_api_scenarios
    instructions = "Verify the room workflow"
    orchestrator._project_fingerprint = compute_fingerprint(
        instructions, orchestrator.primary_kind, domain_model=orchestrator.domain_model)
    orchestrator._checkpointing_enabled = True
    orchestrator._save_checkpoint_for_turn(1, [], instructions)
    checkpoint = load_checkpoint(str(tmp_path))
    assert len(checkpoint.api_scenarios) == 2
    assert "report" not in checkpoint.api_scenarios[0]
    assert checkpoint.api_scenarios[0]["correction_history"]
    checkpoint.api_scenarios[0]["report"] = {"status": "passed", "boot": "ok"}
    save_checkpoint(str(tmp_path), checkpoint)
    with pytest.raises(ValueError):
        restore_api_scenarios(checkpoint.api_scenarios * 6)
    malformed = copy.deepcopy(checkpoint.api_scenarios)
    malformed[0]["scenario"]["requests"][0]["path"] = "https://example.com"
    with pytest.raises(ValueError):
        restore_api_scenarios(malformed)

    resumed = scenario_orchestrator(tmp_path)
    monkeypatch.setattr(resumed, "_apply_adaptive_budget", lambda: None)
    def stop_before_llm(*args, **kwargs):
        raise RuntimeError("checkpoint restored; stop before provider call")
    monkeypatch.setattr(resumed, "_run_phase2", stop_before_llm)
    with pytest.raises(RuntimeError, match="checkpoint restored"):
        resumed.resume(instructions)
    assert all(record["revision"] is None and record["report"]["status"] == "unverified"
               for record in resumed._api_scenarios.values())
    count = len(calls)
    assert resumed._collect_api_scenario_issues() == []
    assert len(calls) == count + 2
    resumed._save_recipe(instructions, elapsed=1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert len(recipe["api_scenarios"]) == 2
    assert recipe["api_scenarios"][0]["report"]["status"] == "passed"
    assert recipe["api_scenarios"][0]["scenario"] == original | {"requests": corrected["requests"]}
