"""Shell access is a deployment decision, and it must be visible as one.

``BESSER_LLM_ENABLE_SHELL_TOOLS`` is read once at import into a module
constant and threaded to the orchestrator by the runner. That is what makes
the hosted gate real: no request body can reach it, so a stranger on
editor.besser-pearl.org cannot turn arbitrary shell on for their own run.
``test_shell_tool_gate.py`` and ``test_shell_tool_defaults.py`` pin the gate
itself; these pin the two things that make the *local* opt-in workable.

1. An operator who sets the variable on an on-prem install can confirm from
   outside the process that it took effect — ``/spec-driven/config`` already
   reports every other feature flag and the docs point clients at it, but it
   omitted this one, so the only way to check was to read container env or
   provoke the agent into running a command.
2. The request body genuinely cannot carry it.
"""

from __future__ import annotations

import asyncio

import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.constants import constants as C
from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
    SmartGenerateRequest,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_spec_driven_router import (
    _build_project_body,
)

BASE_URL = "http://testserver"


def _config() -> dict:
    async def _go() -> httpx.Response:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.get("/besser_api/spec-driven/config")

    return asyncio.run(_go()).json()


# ----------------------------------------------------------------------
# The flag is observable
# ----------------------------------------------------------------------


def test_the_hosted_deploy_reports_shell_tools_off():
    assert _config()["features"]["shell_tools_enabled"] is False


def test_a_local_deploy_that_opted_in_reports_it_on(monkeypatch):
    """Otherwise 'did my env var take?' has no answer short of an RCE probe."""
    monkeypatch.setattr(C, "LLM_ENABLE_SHELL_TOOLS", True)

    assert _config()["features"]["shell_tools_enabled"] is True


# ----------------------------------------------------------------------
# ...and only from the process environment
# ----------------------------------------------------------------------


def test_the_request_body_has_no_shell_field():
    fields = set(SmartGenerateRequest.model_fields)
    assert not [f for f in fields if "shell" in f or "command" in f]


def test_an_extra_body_field_cannot_turn_shell_on():
    """Pydantic ignores unknown keys here; assert that rather than assume it."""
    request = SmartGenerateRequest.model_validate(_build_project_body(
        allow_shell_tools=True,
        features={"shell_tools_enabled": True},
    ))

    assert not hasattr(request, "allow_shell_tools")
    assert "allow_shell_tools" not in request.model_dump()
