"""A run's recipe says what kind of run it was, and its log says when it started and ended.

A shared output folder could not say whether its run edited a previous result
or rebuilt from scratch because that result had expired, which key tier ran
it, or what stopped it; and the worker logged no line when a run started or
ended, so a user's runs could only be found from their download URLs.
"""
import asyncio
import json
import logging
import os
import time
from types import SimpleNamespace

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven import test_modify_seed as seed
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _build_request, _make_base_dir, _parse,
)

KEY = "sk-ant-test-NEVER-LEAK-1234567890"
REQUEST = "Add a dark theme"


class _Orchestrator(seed._StubOrchestrator):
    """Counts its own work like the real one, and writes the recipe a run writes."""

    stop_reason = "completed"
    runtime_cap = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = kwargs.get("max_turns")
        self.max_runtime_seconds = type(self).runtime_cap
        self._phase2_stop_reason = type(self).stop_reason
        self._phase2_exited_cleanly = type(self).stop_reason == "completed"

    def _finish(self, method):
        time.sleep(0.05)  # past a time limit of a few milliseconds, at any clock resolution
        self.phase3_started_at = time.monotonic()
        out = super()._finish(method)
        self.total_turns = 4
        self._generator_used = "generate_fastapi_backend"
        self.target = SimpleNamespace(kind="backend")
        self.tool_calls_log = [{"tool": "write_file", "input": {"path": "a.py"}},
                               {"tool": "modify_file", "input": {"path": "b.py"}},
                               {"tool": "modify_file", "input": {"path": "a.py"}}]
        recipe_path = os.path.join(out, ".besser_recipe.json")
        recipe = json.load(open(recipe_path, encoding="utf-8")) if os.path.isfile(recipe_path) else {}
        recipe.update({"instructions": REQUEST, "llm_model": "claude-sonnet-4-5"})
        with open(recipe_path, "w", encoding="utf-8") as fh:
            json.dump(recipe, fh)
        return out


class _Usage(seed._FakeUsage):
    def summary(self):
        return {"estimated_cost_usd": round(self.estimated_cost, 4), "model": "claude-sonnet-4-5"}


class _Client(seed._FakeClient):
    model = "claude-sonnet-4-5"

    def __init__(self):
        self.usage = _Usage()


@pytest.fixture(autouse=True)
def _clean():
    seed._CALLS.clear()
    asyncio.run(seed._clear_registry())
    _Orchestrator.stop_reason, _Orchestrator.runtime_cap = "completed", 60
    yield
    asyncio.run(seed._clear_registry())


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _Orchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _Client())


def _run(runner):
    frames = asyncio.run(seed._collect_frames(runner))
    parsed = [_parse(frame) for frame in frames]
    assert parsed[-1]["event"] == "done", parsed[-1]
    return parsed[-1]["recipe"]["run"]


def test_a_generate_run_records_a_fresh_project(orchestrator):
    run = _run(SmartGenerationRunner(_build_request(), runtime=current_runtime()))

    assert (run["mode"], run["mode_reason"], run["base_run_id"]) == ("generate", "fresh project", None)
    assert run["key_tier"] == "own key" and run["models"] == ["anthropic/claude-sonnet-4-5"]
    assert (run["target"], run["target_decided_by"], run["generator"]) == ("backend", "not decided",
                                                                           "generate_fastapi_backend")
    assert run["stopped_by"] == "finished" and run["outcome"] == "completed"
    assert run["counts"] == {"turns": 4, "tool_calls": 3, "files_written": 1, "files_edited": 1, "blockers": 0}
    assert run["limits"]["time_seconds"] == 60 and run["request_chars"] == len(REQUEST)
    assert run["estimated_cost_usd"] == 0.01 and run["besser_version"]


def test_a_modify_run_records_the_edited_base(tmp_path, orchestrator):
    base_dir, base_id = _make_base_dir(tmp_path), "a" * 32
    asyncio.run(current_runtime().registry.put(base_id, SmartRunEntry(
        file_path=os.path.join(base_dir, "app", "main.py"), file_name="main.py", is_zip=False,
        temp_dir=base_dir, created_at=time.time())))

    run = _run(SmartGenerationRunner(_build_request(mode="modify", base_run_id=base_id),
                                     runtime=current_runtime(), base_run_id=base_id, mode="modify"))

    assert (run["mode"], run["mode_reason"], run["base_run_id"]) == ("modify", "edit of a previous result", base_id)


def test_an_expired_base_records_the_rebuild_and_the_warning(orchestrator):
    missing = "f" * 32
    run = _run(SmartGenerationRunner(_build_request(mode="modify", base_run_id=missing),
                                     runtime=current_runtime(), base_run_id=missing, mode="modify"))

    assert run["mode"] == "generate"
    assert run["mode_reason"] == "previous result expired, rebuilt from scratch"
    assert run["base_run_id"] == missing
    assert any("expired" in warning for warning in run["warnings"])


def test_a_run_past_its_time_limit_records_what_stopped_it(orchestrator):
    _Orchestrator.runtime_cap = 0.01
    run = _run(SmartGenerationRunner(_build_request(), runtime=current_runtime()))

    assert run["stopped_by"] == "time limit" and run["outcome"] == "budget_stop"
    assert run["stop_message"] == "The run reached its time limit and was stopped before it finished."


def test_a_cancelled_run_that_left_output_records_the_cancel(orchestrator):
    _Orchestrator.stop_reason = "cancelled"
    run = _run(SmartGenerationRunner(_build_request(), runtime=current_runtime()))

    assert run["stopped_by"] == "cancelled" and run["outcome"] == "cancelled"
    assert run["stop_message"] == "The run was cancelled before it finished."


def test_the_section_is_small_and_holds_no_key_path_or_request(orchestrator):
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    run = _run(runner)
    text = json.dumps(run)

    assert len(text) < 4096
    assert KEY not in text and "sk-" not in text and REQUEST not in text
    assert runner.temp_dir not in text and os.path.dirname(runner.temp_dir) not in text
    assert list(run)[:3] == ["mode", "mode_reason", "base_run_id"]


def test_a_free_run_publishes_its_route_not_the_served_model(monkeypatch, orchestrator):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven import run_record

    section = run_record.run_section(run_record.RunFacts(), None, provider="free",
                                     models=["vendor-x/private-model:free"], request_chars=3)
    assert section["key_tier"] == "free" and section["models"] == ["free"]


def test_the_log_says_when_a_run_starts_and_ends_without_its_request(orchestrator, caplog):
    caplog.set_level(logging.INFO, logger=runner_module.__name__)
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    _run(runner)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("spec-driven run ")]
    start = next(line for line in lines if line.startswith("spec-driven run start"))
    end = next(line for line in lines if line.startswith("spec-driven run end"))
    assert f"run_id={runner.run_id}" in start and "mode=generate" in start and "key_tier=own_key" in start
    assert "model=anthropic/claude-sonnet-4-5" in start
    assert f"run_id={runner.run_id}" in end and "outcome=completed" in end and "archive_written=true" in end
    assert "estimated_cost_usd=0.0100" in end and "blockers=0" in end
    for line in (start, end):
        assert REQUEST not in line and KEY not in line


_HOST = "llm.gateway.example-corp.internal"
_LEAKY = (f"Error code: 404 - model not found at https://{_HOST}/v1/chat/completions org-Zx9Q81 "
          "while reading /srv/runs/besser_llm_0f/backend/main.py")


class _FailingModify(_Orchestrator):
    """The provider fails after files exist: the recipe on disk is still the seeded base's."""

    def modify(self, instructions):
        self.client.usage.estimated_cost = 0.07
        self.total_turns, self.tool_calls_log = 2, [{"tool": "write_file", "input": {"path": "new.py"}}]
        with open(os.path.join(self.output_dir, "NEW.md"), "w", encoding="utf-8") as fh:
            fh.write("x\n")
        raise RuntimeError(_LEAKY)


def _base_with_recipe(tmp_path):
    base_dir, base_id = _make_base_dir(tmp_path), "b" * 32
    recipe_path = os.path.join(base_dir, ".besser_recipe.json")
    recipe = json.load(open(recipe_path, encoding="utf-8"))
    recipe.update({"turns": 40, "tool_calls_count": 200, "usage": {"estimated_cost_usd": 2.34},
                   "tool_calls": [{"tool": "write_file", "input": {"path": f"f{i}.py"}} for i in range(30)]})
    json.dump(recipe, open(recipe_path, "w", encoding="utf-8"))
    asyncio.run(current_runtime().registry.put(base_id, SmartRunEntry(
        file_path=os.path.join(base_dir, "app", "main.py"), file_name="main.py", is_zip=False,
        temp_dir=base_dir, created_at=time.time())))
    return base_id


def test_a_modify_run_that_fails_records_its_own_facts_and_no_error_text(tmp_path, monkeypatch, orchestrator):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FailingModify)
    base_id = _base_with_recipe(tmp_path)
    frames = asyncio.run(seed._collect_frames(SmartGenerationRunner(
        _build_request(mode="modify", base_run_id=base_id), runtime=current_runtime(),
        base_run_id=base_id, mode="modify")))
    recipe = _parse(frames[-1])["recipe"]
    run = recipe["run"]

    assert (run["stopped_by"], run["outcome"]) == ("provider error", "failed")
    assert run["counts"]["turns"] == 2 and run["counts"]["tool_calls"] == 1 and run["counts"]["files_written"] == 1
    assert run["estimated_cost_usd"] == recipe["usage"]["estimated_cost_usd"] == 0.07
    text = json.dumps(run)
    for leaked in (_HOST, "org-Zx9Q81", "/srv/runs", "404"):
        assert leaked not in text, leaked
    assert run["stop_message"] == "The model provider failed, so the output may be incomplete."


def test_a_run_that_fails_before_any_recipe_still_logs_its_end(monkeypatch, caplog):
    class _Crashing(_Orchestrator):
        def run(self, instructions):
            raise RuntimeError("provider down")

    caplog.set_level(logging.INFO, logger=runner_module.__name__)
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _Crashing)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _Client())
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    asyncio.run(seed._collect_frames(runner))

    ends = [r.getMessage() for r in caplog.records if r.getMessage().startswith("spec-driven run end")]
    assert len(ends) == 1 and f"run_id={runner.run_id}" in ends[0] and "outcome=failed" in ends[0]
    assert "archive_written=false" in ends[0]


def test_validation_is_timed_from_where_it_starts():
    from besser.utilities.web_modeling_editor.backend.services.spec_driven import run_record

    facts = run_record.RunFacts()
    facts.phase_started = {"customize": 100.0, "validate": 160.0}  # Phase 3 reported 60 s after it began
    orchestrator = SimpleNamespace(phase3_started_at=130.0)
    section = run_record.run_section(facts, orchestrator, provider="openai", models=[], request_chars=1)
    assert section["phase_seconds"]["customise"] == 30.0
