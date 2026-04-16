"""Mechanical proof that the user's API key never reaches any log record.

Capture every LogRecord produced by every logger during a full
``/besser_api/smart-generate`` request and assert that the literal
API-key string does not appear in any record's formatted message.

Also sanity-check that the request *is* actually being logged (path +
status) so we're not just passing because logging is silent.
"""

import asyncio
import json
import logging

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.smart_generation import runner as runner_module
from tests.utilities.web_modeling_editor.backend.smart_generation.test_runner import (
    _FakeOrchestrator,
    _FakeClient,
    _clear_registry,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_smart_generation_router import (
    _build_project_body,
)


BASE_URL = "http://testserver"
LEAK_CANARY = "sk-ant-LEAK-CANARY-DO-NOT-LOG-xyzzy-0123456789abcdef"


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def all_text(self) -> str:
        parts = []
        for rec in self.records:
            try:
                parts.append(rec.getMessage())
            except Exception:
                parts.append(str(rec.msg))
            if rec.exc_info:
                import traceback
                parts.append("".join(traceback.format_exception(*rec.exc_info)))
            parts.append(repr(rec.args))
        return "\n".join(parts)


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def capture_all_logs():
    handler = _CaptureHandler()
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    # Also make sure the middleware and backend loggers propagate.
    touched = []
    for name in (
        "besser",
        "besser.utilities.web_modeling_editor.backend",
        "besser.utilities.web_modeling_editor.backend.middleware.request_logging",
        "besser.utilities.web_modeling_editor.backend.services.smart_generation.runner",
        "besser.utilities.web_modeling_editor.backend.routers.smart_generation_router",
    ):
        lg = logging.getLogger(name)
        touched.append((lg, lg.level, lg.propagate))
        lg.setLevel(logging.DEBUG)
        lg.propagate = True

    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        for lg, level, propagate in touched:
            lg.setLevel(level)
            lg.propagate = propagate


@pytest.fixture
def stub_backend(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())


class TestApiKeyNeverInLogs:
    def test_happy_path_request_does_not_leak_key(self, stub_backend, capture_all_logs):
        body = _build_project_body(api_key=LEAK_CANARY)
        transport = ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                async with ac.stream(
                    "POST",
                    "/besser_api/smart-generate",
                    json=body,
                ) as response:
                    async for _ in response.aiter_bytes():
                        pass
                    return response.status_code

        status = asyncio.run(_run())
        assert status == 200

        log_text = capture_all_logs.all_text()
        assert LEAK_CANARY not in log_text, (
            "API key leaked into captured log output. "
            f"Occurrences found; first 200 chars: {log_text[:200]!r}"
        )

    def test_log_capture_actually_captures_something(self, stub_backend, capture_all_logs):
        """Sanity: make sure the fixture is wired such that at least
        one log record is produced for a request — otherwise the
        leak test could pass trivially because no logs happened at all.
        """
        body = _build_project_body()
        transport = ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                async with ac.stream(
                    "POST",
                    "/besser_api/smart-generate",
                    json=body,
                ) as response:
                    async for _ in response.aiter_bytes():
                        pass

        asyncio.run(_run())
        log_text = capture_all_logs.all_text()
        # The request_logging middleware emits a structured "request completed"
        # line for every response, so we expect the path to appear.
        assert "/besser_api/smart-generate" in log_text, (
            f"Expected the request path in captured logs, got: {log_text[:500]!r}"
        )

    def test_invalid_key_error_path_does_not_leak_key(self, monkeypatch, capture_all_logs):
        """Even on the error path, the literal key must not reach logs."""
        monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)

        def _reject(**kwargs):
            # Simulate real provider error. The plain-text message
            # from besser.generators.llm.llm_client does NOT echo the
            # actual key — it just says "No Anthropic API key found...".
            raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY.")

        monkeypatch.setattr(runner_module, "create_llm_client", _reject)

        body = _build_project_body(api_key=LEAK_CANARY + "-invalid")
        transport = ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                async with ac.stream(
                    "POST",
                    "/besser_api/smart-generate",
                    json=body,
                ) as response:
                    async for _ in response.aiter_bytes():
                        pass

        asyncio.run(_run())
        log_text = capture_all_logs.all_text()
        assert LEAK_CANARY not in log_text
