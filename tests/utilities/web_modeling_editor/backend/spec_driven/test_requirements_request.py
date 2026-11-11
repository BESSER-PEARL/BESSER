"""``SmartGenerateRequest.requirements``: the modeling assistant's requirements reach the run.

The editor sends the requirements it recorded in the project trace; the run
builds its requirements ledger from them (see
``tests/spec_driven_agent/test_stated_requirements.py``).
"""

import asyncio

import pytest
from pydantic import ValidationError

from besser.spec_driven_agent.planning.specification import MAX_REQUIREMENT_CHARS, MAX_REQUIREMENTS
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import SmartGenerationRunner
from tests.utilities.web_modeling_editor.backend.spec_driven.test_done_verdict import _cleanup
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _FakeClient,
    _StubOrchestrator,
    _build_request,
    _collect_frames,
)

REQUIREMENT = {"id": "call-1:R2", "text": "A room is identified by its number", "kind": "uniqueness"}


def test_requirements_are_optional():
    assert _build_request().requirements is None


@pytest.mark.parametrize("requirements, message", [
    ([dict(REQUIREMENT, kind="feature")], "kind"),
    ([REQUIREMENT, dict(REQUIREMENT)], "unique ids"),
    ([dict(REQUIREMENT, text="")], "text"),
    ([dict(REQUIREMENT, text="x" * (MAX_REQUIREMENT_CHARS + 1))], "text"),
    ([dict(REQUIREMENT, id=f"r{i}") for i in range(MAX_REQUIREMENTS + 1)], "requirements"),
])
def test_malformed_requirements_are_rejected(requirements, message):
    with pytest.raises(ValidationError, match=message):
        _build_request(requirements=requirements)


def test_the_runner_hands_the_requirements_to_the_orchestrator(monkeypatch):
    observed = []

    class CaptureRequirements(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            observed.append(kwargs["stated_requirements"])
            self._validation_issues = []

    monkeypatch.setattr(runner_module, "create_llm_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(runner_module, "LLMOrchestrator", CaptureRequirements)
    try:
        asyncio.run(_collect_frames(SmartGenerationRunner(_build_request(requirements=[REQUIREMENT]))))
        asyncio.run(_collect_frames(SmartGenerationRunner(_build_request())))
    finally:
        _cleanup()
    assert observed == [[REQUIREMENT], None]
