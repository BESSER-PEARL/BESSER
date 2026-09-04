"""Tests for the pilot-experiment telemetry system.

Covers the collector gating (master switch + participant), the JSONL
store schema, the three-way deterministic/LLM file split, the report
endpoint's auth, the runner's in-process ``run_summary`` emission, and
the fail-open request-field sanitization (telemetry must never fail a
generation run).
"""

import asyncio
import json
import os
from types import SimpleNamespace

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    telemetry,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _FakeOrchestrator,
    _FailingOrchestrator,
    _build_request,
    _clear_registry,
    _collect_frames,
    _parse_frame,
)


BASE_URL = "http://testserver"


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def telemetry_env(monkeypatch, tmp_path):
    """Enable telemetry with a per-test store directory."""
    monkeypatch.setenv("BESSER_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("BESSER_TELEMETRY_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def telemetry_disabled(monkeypatch, tmp_path):
    """Telemetry explicitly OFF, but with a store dir so a leak would show."""
    monkeypatch.delenv("BESSER_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setenv("BESSER_TELEMETRY_DIR", str(tmp_path))
    return tmp_path


def _read_events(directory, session):
    path = os.path.join(str(directory), f"{session}.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def _post_event(body: dict) -> httpx.Response:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        return await ac.post("/besser_api/telemetry/event", json=body)


async def _get_report(headers=None, params=None) -> httpx.Response:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        return await ac.get(
            "/besser_api/telemetry/report",
            headers=headers or {},
            params=params or {},
        )


def _event_body(**overrides) -> dict:
    body = {
        "session": "sess-abc-123",
        "participant": "P3",
        "kind": "prompt",
        "payload": {"text": "build a library app", "action_taken": "created_model"},
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------
# record_event / gating (unit level)
# ---------------------------------------------------------------------


class TestRecordEventGating:
    def test_disabled_writes_nothing(self, telemetry_disabled):
        telemetry.record_event("s1", "P1", "prompt", {"text": "hi"})
        assert os.listdir(str(telemetry_disabled)) == []

    def test_enabled_writes_line_with_schema(self, telemetry_env):
        telemetry.record_event("s1", "P1", "prompt", {"text": "hi"})
        events = _read_events(telemetry_env, "s1")
        assert len(events) == 1
        event = events[0]
        assert set(event.keys()) == {"at", "session", "participant", "kind", "payload"}
        assert event["session"] == "s1"
        assert event["participant"] == "P1"
        assert event["kind"] == "prompt"
        assert event["payload"] == {"text": "hi"}
        # ISO-8601 UTC timestamp
        assert "T" in event["at"]

    def test_empty_participant_dropped(self, telemetry_env):
        telemetry.record_event("s1", "", "prompt", {"text": "hi"})
        telemetry.record_event("s1", None, "prompt", {"text": "hi"})
        assert os.listdir(str(telemetry_env)) == []

    def test_invalid_session_dropped(self, telemetry_env):
        telemetry.record_event("bad session!", "P1", "prompt", {})
        telemetry.record_event("../../etc/passwd", "P1", "prompt", {})
        assert os.listdir(str(telemetry_env)) == []

    def test_unknown_kind_dropped(self, telemetry_env):
        telemetry.record_event("s1", "P1", "surveillance", {})
        assert os.listdir(str(telemetry_env)) == []

    def test_run_summary_kind_accepted_in_process(self, telemetry_env):
        telemetry.record_event("s1", "P1", "run_summary", {"outcome": "success"})
        events = _read_events(telemetry_env, "s1")
        assert len(events) == 1
        assert events[0]["kind"] == "run_summary"

    def test_never_raises_on_bad_payload(self, telemetry_env):
        # A non-serializable payload must not raise into the caller.
        telemetry.record_event("s1", "P1", "prompt", {"obj": object()})
        events = _read_events(telemetry_env, "s1")
        assert len(events) == 1  # written with default=str fallback

    def test_never_raises_on_unusable_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BESSER_TELEMETRY_ENABLED", "1")
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        monkeypatch.setenv("BESSER_TELEMETRY_DIR", str(blocker / "nested"))
        telemetry.record_event("s1", "P1", "prompt", {})  # must not raise


# ---------------------------------------------------------------------
# compute_file_split / llm_touched_paths (unit level)
# ---------------------------------------------------------------------


class TestFileSplit:
    def _build_tree(self, root):
        files = {
            "app/gen_untouched.py": "# generator\n",
            "app/gen_modified.py": "# generator, later edited\n",
            "app/llm_new.py": "# llm\n",
            ".besser_recipe.json": "{}",
            "besser_smart_abc.zip": "zipbytes",
            "node_modules/pkg/index.js": "x",
        }
        for rel, content in files.items():
            full = os.path.join(str(root), rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)

    def test_three_way_split(self, tmp_path):
        self._build_tree(tmp_path)
        split = telemetry.compute_file_split(
            str(tmp_path),
            generator_files={"app/gen_untouched.py", "app/gen_modified.py"},
            llm_touched={"app/gen_modified.py", "app/llm_new.py"},
            excluded_patterns=runner_module._EXCLUDED_OUTPUT_DIRS,
        )
        assert split["generator_untouched"] == 1
        assert split["generator_llm_modified"] == 1
        assert split["llm_authored"] == 1
        assert split["total"] == 3  # recipe, zip, node_modules all excluded
        assert split["generator_untouched_pct"] == pytest.approx(33.3)
        assert split["llm_authored_pct"] == pytest.approx(33.3)

    def test_empty_dir_returns_zeros(self):
        split = telemetry.compute_file_split("", None, None)
        assert split["total"] == 0
        assert split["generator_untouched_pct"] == 0.0

    def test_llm_touched_paths_filters_and_normalizes(self):
        log = [
            {"tool": "write_file", "input": {"path": "a\\b.py"}, "success": True},
            {"tool": "modify_file", "input": {"path": "c.py"}, "success": True},
            # Guardrail rejection — changed nothing on disk.
            {"tool": "write_file", "input": {"path": "rejected.py"}, "success": False},
            {"tool": "read_file", "input": {"path": "ignored.py"}, "success": True},
            "not-a-dict",
        ]
        assert telemetry.llm_touched_paths(log) == {"a/b.py", "c.py"}


# ---------------------------------------------------------------------
# POST /besser_api/telemetry/event (HTTP collector)
# ---------------------------------------------------------------------


class TestCollectorEndpoint:
    def test_disabled_returns_204_and_writes_nothing(self, telemetry_disabled):
        response = asyncio.run(_post_event(_event_body()))
        assert response.status_code == 204
        assert response.content == b""
        assert os.listdir(str(telemetry_disabled)) == []

    def test_enabled_returns_204_and_writes_file(self, telemetry_env):
        response = asyncio.run(_post_event(_event_body()))
        assert response.status_code == 204
        events = _read_events(telemetry_env, "sess-abc-123")
        assert len(events) == 1
        assert events[0]["kind"] == "prompt"
        assert events[0]["participant"] == "P3"

    def test_invalid_session_rejected_422(self, telemetry_env):
        response = asyncio.run(_post_event(_event_body(session="bad session!")))
        assert response.status_code == 422
        response = asyncio.run(_post_event(_event_body(session="x" * 65)))
        assert response.status_code == 422

    def test_invalid_participant_rejected_422(self, telemetry_env):
        response = asyncio.run(_post_event(_event_body(participant="P3 ok?")))
        assert response.status_code == 422
        response = asyncio.run(_post_event(_event_body(participant="x" * 17)))
        assert response.status_code == 422

    def test_invalid_kind_rejected_422(self, telemetry_env):
        response = asyncio.run(_post_event(_event_body(kind="telepathy")))
        assert response.status_code == 422
        # run_summary is in-process only — the HTTP collector refuses it.
        response = asyncio.run(_post_event(_event_body(kind="run_summary")))
        assert response.status_code == 422

    def test_oversized_payload_rejected_422(self, telemetry_env):
        big = {"blob": "x" * (9 * 1024)}
        response = asyncio.run(_post_event(_event_body(payload=big)))
        assert response.status_code == 422
        assert os.listdir(str(telemetry_env)) == []

    def test_all_valid_kinds_accepted(self, telemetry_env):
        for kind in ("prompt", "agent_action", "delivery", "friction"):
            response = asyncio.run(_post_event(_event_body(kind=kind)))
            assert response.status_code == 204
        events = _read_events(telemetry_env, "sess-abc-123")
        assert [e["kind"] for e in events] == [
            "prompt", "agent_action", "delivery", "friction",
        ]


# ---------------------------------------------------------------------
# GET /besser_api/telemetry/report (auth + rendering)
# ---------------------------------------------------------------------


def _seed_report_data():
    telemetry.record_event("s1", "P1", "prompt", {"text": "make an app"})
    telemetry.record_event("s1", "P1", "delivery", {"action": "download"})
    telemetry.record_event("s1", "P1", "run_summary", {
        "outcome": "success", "duration_seconds": 120.0, "turns": 10,
        "tokens": 5000, "blockers_found": 2, "blockers_remaining": 0,
        "file_split": {
            "generator_untouched": 6, "generator_llm_modified": 2,
            "llm_authored": 2, "total": 10,
        },
    })
    telemetry.record_event("s2", "P2", "prompt", {"text": "another app"})
    telemetry.record_event("s2", "P2", "friction", {"signal": "run_cancelled"})
    telemetry.record_event("s2", "P2", "run_summary", {
        "outcome": "failed:UPSTREAM_LLM", "duration_seconds": 30.0,
        "turns": 2, "tokens": 800,
        "file_split": {
            "generator_untouched": 0, "generator_llm_modified": 0,
            "llm_authored": 0, "total": 0,
        },
    })


class TestReportEndpoint:
    TOKEN = "pilot-admin-token-123"

    @pytest.fixture
    def report_env(self, telemetry_env, monkeypatch):
        monkeypatch.setenv("BESSER_TELEMETRY_ADMIN_TOKEN", self.TOKEN)
        _seed_report_data()
        return telemetry_env

    def test_404_when_no_admin_token_configured(self, telemetry_env, monkeypatch):
        monkeypatch.delenv("BESSER_TELEMETRY_ADMIN_TOKEN", raising=False)
        response = asyncio.run(_get_report(headers={"X-Telemetry-Token": "anything"}))
        assert response.status_code == 404

    def test_403_when_token_missing(self, report_env):
        response = asyncio.run(_get_report())
        assert response.status_code == 403

    def test_403_when_token_wrong(self, report_env):
        response = asyncio.run(_get_report(headers={"X-Telemetry-Token": "wrong"}))
        assert response.status_code == 403

    def test_markdown_report(self, report_env):
        response = asyncio.run(
            _get_report(headers={"X-Telemetry-Token": self.TOKEN})
        )
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        text = response.text
        assert "## Participant P1" in text
        assert "## Participant P2" in text
        assert "## Overall" in text
        assert "1 success / 0 incomplete / 0 failed" in text
        assert "0 success / 0 incomplete / 1 failed" in text
        # P1's aggregate split percentages
        assert "60.0%" in text
        assert "Delivery actions: download 1" in text

    def test_csv_report(self, report_env):
        response = asyncio.run(_get_report(
            headers={"X-Telemetry-Token": self.TOKEN},
            params={"format": "csv"},
        ))
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        lines = response.text.strip().splitlines()
        assert lines[0].startswith("participant,sessions,prompts")
        assert len(lines) == 3  # header + P1 + P2
        assert lines[1].startswith("P1,")
        assert lines[2].startswith("P2,")

    def test_participant_filter(self, report_env):
        response = asyncio.run(_get_report(
            headers={"X-Telemetry-Token": self.TOKEN},
            params={"participant": "P1"},
        ))
        assert response.status_code == 200
        assert "## Participant P1" in response.text
        assert "## Participant P2" not in response.text


# ---------------------------------------------------------------------
# Runner run_summary emission (in-process)
# ---------------------------------------------------------------------


class _UsageWithTokens:
    def __init__(self):
        self.estimated_cost = 0.0
        self.total_tokens = 4321
        self.served_model = "claude-sonnet-4-5"


class _ClientWithTokens:
    def __init__(self):
        self.usage = _UsageWithTokens()


class _SplitOrchestrator(_FakeOrchestrator):
    """Writes a mixed generator/LLM tree and exposes the real orchestrator's
    file-tagging surfaces (executor._generator_files, tool_calls_log)."""

    def run(self, instructions: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        for rel in (
            "app/gen_untouched.py", "app/gen_modified.py", "app/llm_new.py",
        ):
            full = os.path.join(self.output_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write("# content\n")
        with open(
            os.path.join(self.output_dir, ".besser_recipe.json"),
            "w", encoding="utf-8",
        ) as fh:
            json.dump({"instructions": instructions}, fh)

        self.executor = SimpleNamespace(
            _generator_files={"app/gen_untouched.py", "app/gen_modified.py"},
        )
        self.tool_calls_log = [
            {"turn": 1, "tool": "modify_file",
             "input": {"path": "app/gen_modified.py"}, "success": True},
            {"turn": 2, "tool": "write_file",
             "input": {"path": "app/llm_new.py"}, "success": True},
            {"turn": 3, "tool": "write_file",
             "input": {"path": "app/rejected.py"}, "success": False},
        ]
        self._validation_issues = [
            SimpleNamespace(severity="blocker", message="import error"),
            SimpleNamespace(severity="warning", message="style"),
        ]
        # blockerCount on the done event keys off a CLEAN phase-2 exit.
        self._phase2_exited_cleanly = True
        self.total_turns = 5
        if self.on_progress:
            self.on_progress(5, "validation", "2 blockers / 4 total")
        self.client.usage.estimated_cost = 0.05
        return self.output_dir


@pytest.fixture
def split_orchestrator(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _SplitOrchestrator)
    monkeypatch.setattr(
        runner_module, "create_llm_client", lambda **_: _ClientWithTokens()
    )


@pytest.fixture
def failing_orchestrator(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FailingOrchestrator)
    monkeypatch.setattr(
        runner_module, "create_llm_client", lambda **_: _ClientWithTokens()
    )


def _telemetry_request(**overrides):
    defaults = dict(
        telemetry_session="pilot-sess-1",
        telemetry_participant="P3",
    )
    defaults.update(overrides)
    return _build_request(**defaults)


class TestRunSummaryEmission:
    def test_successful_run_records_summary_with_split(
        self, split_orchestrator, telemetry_env
    ):
        runner = SmartGenerationRunner(_telemetry_request())
        frames = asyncio.run(_collect_frames(runner))

        events = _read_events(telemetry_env, "pilot-sess-1")
        summaries = [e for e in events if e["kind"] == "run_summary"]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["participant"] == "P3"
        payload = summary["payload"]
        assert payload["outcome"] == "incomplete"  # 1 unfixed blocker remains
        assert payload["provider"] == "anthropic"
        assert payload["mode"] == "generate"
        assert payload["run_id"] == runner.run_id
        assert payload["model_requested"] == "claude-sonnet-4-5"
        assert payload["model_final"] == "claude-sonnet-4-5"
        assert payload["model_switched"] is False
        assert payload["turns"] == 5
        assert payload["tokens"] == 4321
        assert payload["files_produced"] == 3
        assert payload["blockers_found"] == 2
        assert payload["blockers_remaining"] == 1
        assert payload["duration_seconds"] >= 0
        split = payload["file_split"]
        assert split["generator_untouched"] == 1
        assert split["generator_llm_modified"] == 1
        assert split["llm_authored"] == 1
        assert split["total"] == 3

        # The done event carries the same split.
        done = [
            p for p in (_parse_frame(f) for f in frames) if p["event"] == "done"
        ]
        assert len(done) == 1
        assert done[0]["fileSplit"] == split

    def test_failed_run_records_failure_summary(
        self, failing_orchestrator, telemetry_env
    ):
        runner = SmartGenerationRunner(_telemetry_request())
        asyncio.run(_collect_frames(runner))

        events = _read_events(telemetry_env, "pilot-sess-1")
        summaries = [e for e in events if e["kind"] == "run_summary"]
        assert len(summaries) == 1
        payload = summaries[0]["payload"]
        assert payload["outcome"] == "failed:UPSTREAM_LLM"
        assert payload["files_produced"] == 0
        assert payload["file_split"]["total"] == 0

    def test_no_summary_without_telemetry_fields(
        self, split_orchestrator, telemetry_env
    ):
        runner = SmartGenerationRunner(_build_request())
        frames = asyncio.run(_collect_frames(runner))
        assert os.listdir(str(telemetry_env)) == []
        # The run itself still succeeds.
        assert _parse_frame(frames[-1])["event"] == "done"

    def test_no_summary_when_disabled(
        self, split_orchestrator, telemetry_disabled
    ):
        runner = SmartGenerationRunner(_telemetry_request())
        frames = asyncio.run(_collect_frames(runner))
        assert os.listdir(str(telemetry_disabled)) == []
        assert _parse_frame(frames[-1])["event"] == "done"

    def test_summary_failure_cannot_break_the_run(
        self, split_orchestrator, telemetry_env, monkeypatch
    ):
        def _boom(*args, **kwargs):
            raise RuntimeError("telemetry store exploded")

        monkeypatch.setattr(runner_module.telemetry, "record_event", _boom)
        runner = SmartGenerationRunner(_telemetry_request())
        frames = asyncio.run(_collect_frames(runner))
        assert _parse_frame(frames[-1])["event"] == "done"


# ---------------------------------------------------------------------
# Request-field sanitization (a run must never fail over telemetry)
# ---------------------------------------------------------------------


class TestRequestFieldSanitization:
    def test_valid_fields_preserved(self):
        request = _build_request(
            telemetry_session="pilot-sess_1", telemetry_participant="P12",
        )
        assert request.telemetry_session == "pilot-sess_1"
        assert request.telemetry_participant == "P12"

    def test_invalid_session_nulled_not_rejected(self):
        request = _build_request(
            telemetry_session="bad session!", telemetry_participant="P3",
        )
        assert request.telemetry_session is None
        assert request.telemetry_participant == "P3"  # independent validation

    def test_invalid_participant_nulled_not_rejected(self):
        request = _build_request(
            telemetry_session="ok-session",
            telemetry_participant="way-too-long-participant-label",
        )
        assert request.telemetry_session == "ok-session"
        assert request.telemetry_participant is None

    def test_non_string_values_nulled(self):
        request = _build_request(
            telemetry_session=12345, telemetry_participant={"P": 3},
        )
        assert request.telemetry_session is None
        assert request.telemetry_participant is None

    def test_overlong_session_nulled(self):
        request = _build_request(
            telemetry_session="x" * 65, telemetry_participant="P1",
        )
        assert request.telemetry_session is None

    def test_absent_fields_default_to_none(self):
        request = _build_request()
        assert request.telemetry_session is None
        assert request.telemetry_participant is None
