"""#666: once the app exists, the whole-application generators are gone.

Re-running ``generate_fastapi_backend`` / ``generate_web_app`` after Phase 1
regenerates the app over every Phase 2 edit. Before the fix the primaries
stayed in the Phase 2 toolset and the executor ran them; only the prompt
("already generated") stood in the way.
"""
import json
import os

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.constants import _APPLICATION_PRIMARY_TOOLS
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):  # pragma: no cover - not called
        raise AssertionError("must not call the LLM")


def _orchestrator(tmp_path) -> LLMOrchestrator:
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
    return LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=DomainModel(name="Library", types={book}),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )


def _names(tools):
    return {t.get("name") for t in tools if isinstance(t, dict)}


def _scaffold_then_edit(orch, tmp_path):
    """Phase 1 runs the primary; Phase 2 then edits a generated file."""
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"
    orch._generator_used = "generate_fastapi_backend"
    main = os.path.join(str(tmp_path), "backend", "main_api.py")
    with open(main, "a", encoding="utf-8") as fh:
        fh.write("\n# custom route added in Phase 2\n")
    with open(main, encoding="utf-8") as fh:
        return main, fh.read()


def test_primary_rerun_after_phase1_leaves_edits_unchanged(tmp_path):
    orch = _orchestrator(tmp_path)
    main, edited = _scaffold_then_edit(orch, tmp_path)

    orch._drop_redundant_generator_tools()  # Phase 2 entry
    result = json.loads(orch.executor.execute("generate_fastapi_backend", {}))

    assert "error" in result and "start a fresh run" in result["error"]
    with open(main, encoding="utf-8") as fh:
        assert fh.read() == edited


def test_primaries_leave_the_phase2_toolset(tmp_path):
    orch = _orchestrator(tmp_path)
    orch._generator_used = "generate_fastapi_backend"
    orch._drop_redundant_generator_tools()
    assert not (_names(orch.tools) & _APPLICATION_PRIMARY_TOOLS)
    assert "write_file" in _names(orch.tools)


def test_modify_run_locks_primaries_whatever_built_the_seed(tmp_path):
    orch = _orchestrator(tmp_path)
    orch._modify_mode = True
    orch._generator_used = None  # an LLM-authored seed app
    (tmp_path / "backend").mkdir()
    orch._drop_redundant_generator_tools()
    assert not (_names(orch.tools) & _APPLICATION_PRIMARY_TOOLS)
    assert "start a fresh run" in json.loads(orch.executor.execute("generate_web_app", {}))["error"]


def test_phase1_generator_call_is_not_refused(tmp_path):
    orch = _orchestrator(tmp_path)
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"


def test_no_scaffold_keeps_primaries(tmp_path):
    # Phase 1 failed or picked none: a primary in Phase 2 overwrites nothing.
    orch = _orchestrator(tmp_path)
    orch._generator_used = None
    orch._drop_redundant_generator_tools()
    assert "generate_fastapi_backend" in _names(orch.tools)


def test_modify_run_with_a_class_model_seed_locks_primaries(tmp_path):
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"model": {"classes": ["Book"]}}))
    orch = _orchestrator(tmp_path)
    orch._modify_mode = True
    orch._drop_redundant_generator_tools()
    assert not (_names(orch.tools) & _APPLICATION_PRIMARY_TOOLS)


def test_modify_run_on_a_seed_without_an_app_keeps_primaries(tmp_path):
    # An agent-only or state-machine seed: nothing a primary would overwrite.
    (tmp_path / ".besser_recipe.json").write_text(
        json.dumps({"generator_used": "generate_baf", "model": {"primary_kind": "agent"}}))
    (tmp_path / "agent").mkdir()
    orch = _orchestrator(tmp_path)
    orch._modify_mode = True
    orch._generator_used = "generate_baf"
    orch._drop_redundant_generator_tools()
    assert "generate_fastapi_backend" in _names(orch.tools)
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"
