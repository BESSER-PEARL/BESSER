"""Tests for the smart-generation SSE event schema and framer."""

import json

import pytest

from besser.utilities.web_modeling_editor.backend.services.smart_generation.sse_events import (
    CostEvent,
    DoneEvent,
    ErrorEvent,
    PhaseEvent,
    StartEvent,
    TextDeltaEvent,
    ToolCallEvent,
    format_sse,
)


class TestFormatSse:
    """format_sse must produce exactly `event: <name>\\ndata: <json>\\n\\n`."""

    def test_start_event_framing(self):
        event = StartEvent(
            runId="abc123",
            provider="anthropic",
            llmModel="claude-sonnet-4-5",
            maxCost=1.0,
            maxRuntime=600,
        )
        frame = format_sse(event)
        assert isinstance(frame, bytes)
        text = frame.decode("utf-8")
        assert text.startswith("event: start\n")
        assert "\ndata: " in text
        assert text.endswith("\n\n")
        # Exactly one terminating blank line
        assert text.count("\n\n") == 1

    def test_data_line_is_valid_json_with_event_field(self):
        event = PhaseEvent(phase="generate", message="running django_generator")
        frame = format_sse(event).decode("utf-8")
        data_line = [l for l in frame.splitlines() if l.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: "):])
        assert payload["event"] == "phase"
        assert payload["phase"] == "generate"
        assert payload["message"] == "running django_generator"

    def test_text_delta_escapes_newlines(self):
        """A delta containing literal newlines must not break SSE framing."""
        event = TextDeltaEvent(delta="line one\nline two\r\nline three")
        frame = format_sse(event).decode("utf-8")
        # Still exactly one `\n\n` terminator
        assert frame.count("\n\n") == 1
        # The payload must round-trip
        data_line = [l for l in frame.splitlines() if l.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: "):])
        assert payload["delta"] == "line one\nline two\r\nline three"

    @pytest.mark.parametrize(
        "event",
        [
            ToolCallEvent(turn=3, tool="write_file", status="executing"),
            CostEvent(usd=0.0342, turns=7, elapsedSeconds=23.41),
            DoneEvent(
                downloadUrl="/besser_api/download-smart/abc",
                fileName="besser_smart_abc.zip",
                isZip=True,
                recipe={"usage": {"input_tokens": 100}},
            ),
            ErrorEvent(code="INVALID_KEY", message="No API key"),
            ErrorEvent(code="COST_CAP", message="Cost cap reached ($1.01 > $1.00)"),
        ],
    )
    def test_all_event_types_round_trip(self, event):
        """Every event type should round-trip through format_sse + JSON."""
        frame = format_sse(event).decode("utf-8")
        assert frame.startswith(f"event: {event.event}\n")
        data_line = [l for l in frame.splitlines() if l.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: "):])
        assert payload["event"] == event.event


class TestEventValidation:
    """Pydantic should reject malformed events at construction time."""

    def test_phase_rejects_unknown_phase(self):
        with pytest.raises(Exception):
            PhaseEvent(phase="nonsense", message="oops")  # type: ignore[arg-type]

    def test_error_rejects_unknown_code(self):
        with pytest.raises(Exception):
            ErrorEvent(code="FROBNICATED", message="oops")  # type: ignore[arg-type]

    def test_events_do_not_expose_api_key_field(self):
        """None of the event models should have an api_key field at all."""
        for cls in (StartEvent, PhaseEvent, TextDeltaEvent, ToolCallEvent,
                    CostEvent, DoneEvent, ErrorEvent):
            assert "api_key" not in cls.model_fields
            assert "apiKey" not in cls.model_fields
