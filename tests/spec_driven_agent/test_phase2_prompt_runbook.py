"""The Phase 2 prompt carries the runtime-verification runbook with shell on.

``build_system_prompt`` renders it only when given ``allow_shell`` and
``output_dir``, and the orchestrator passed neither, so no Phase 2 run was
ever shown the procedure the probe script exists for.
"""
import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

RUNBOOK = "## Runtime verification"


class _Client:
    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no LLM call expected
        raise AssertionError("no LLM call expected")


@pytest.mark.parametrize("shell", [True, False])
def test_the_runbook_follows_the_shell_setting(tmp_path, shell):
    (tmp_path / "main_api.py").write_text("app = None\n", encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Hotel", types={Class(name="Booking")}),
        output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False, allow_shell_tools=shell,
    )

    prompt = orch._build_system_prompt("add a booking total")

    assert (RUNBOOK in prompt) is shell
    assert (tmp_path / ".besser_probe.py").exists() is shell
