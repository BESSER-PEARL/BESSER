"""Phase 3 reports checks it could not run instead of passing or retrying them.

The endpoint-coherence check and the acceptance matrix swallowed their own
exceptions at debug level, so a crash was indistinguishable from "no findings".
A Phase 2 quota stop still let Phase 3 start its provider calls.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue
from besser.spec_driven_agent.providers.llm_client import UsageTracker


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")


def _boom(*_args, **_kwargs):
    raise RuntimeError("validator bug")


@pytest.mark.parametrize("target, check", [
    ("besser.spec_driven_agent.validation.endpoint_coherence.collect_endpoint_coherence_issues",
     "the endpoint coherence check"),
    ("besser.spec_driven_agent.validation.acceptance.build_acceptance_matrix",
     "the acceptance matrix"),
])
def test_a_crashing_validator_is_reported_as_not_run(
        simple_library_book_model, tmp_path, monkeypatch, target, check):
    monkeypatch.setattr(target, _boom)
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
        enable_toolchain_validation=False,
    )

    notes = [i for i in orch._collect_validation_issues()
             if i.message.startswith(f"validation: {check} did not run")]

    assert len(notes) == 1, [i.message for i in orch._collect_validation_issues()]
    assert "RuntimeError" in notes[0].message
    assert _classify_issue(notes[0].message).severity == "warning"


def test_phase3_does_not_start_after_a_phase2_quota_stop(simple_library_book_model, tmp_path):
    """An exhausted quota fails every later call the same way, so Phase 3's
    validation and repair calls could only burn time on it."""
    class _QuotaClient(_Client):
        def chat(self, **kwargs):
            raise RuntimeError(
                "OpenAI API call failed: Error code: 429 - You exceeded your current "
                "quota (insufficient_quota)")

    orch = LLMOrchestrator(
        llm_client=_QuotaClient(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
        enable_toolchain_validation=False, auto_fix_issues=True,
    )
    orch.run("Build a library app")

    assert orch._phase2_stop_reason == "api_error"
    assert orch._phase3_exit_reason == "skipped: provider quota exhausted in Phase 2"
