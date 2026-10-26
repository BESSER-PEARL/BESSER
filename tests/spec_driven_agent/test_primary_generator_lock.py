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
from besser.spec_driven_agent.state.checkpoint import compute_fingerprint, load_checkpoint


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):  # pragma: no cover - not called
        raise AssertionError("must not call the LLM")


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        self.__dict__.update(kwargs)


class _ScriptedClient:
    """Calls ``generate_fastapi_backend`` once, then ends the turn."""

    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")
        self.tool_results = []
        self._called = False

    def chat(self, system, messages, tools):
        for message in messages:
            if isinstance(message.get("content"), list):
                self.tool_results += [b for b in message["content"]
                                      if isinstance(b, dict) and b.get("type") == "tool_result"]
        if self._called:
            return {"stop_reason": "end_turn", "content": [_Block("text", text="Done")]}
        self._called = True
        return {"stop_reason": "tool_use", "content": [
            _Block("tool_use", id="g1", name="generate_fastapi_backend", input={})]}


def _model():
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
    return DomainModel(name="Library", types={book})


def _orchestrator(tmp_path, client=None, checkpointing=False) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client or _MockClient(),
        domain_model=_model(),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=checkpointing,
        use_streaming=False,
    )


def _names(tools):
    return {t.get("name") for t in tools if isinstance(t, dict)}


def _refused(orch, tool):
    result = json.loads(orch.executor.execute(tool, {}))
    return "start a fresh run" in result.get("error", "")


def _modify_run(orch):
    """The steps of ``modify()`` that decide the lock, then Phase 2 entry."""
    orch._modify_mode = True
    orch._seed_generator_files_from_recipe()
    orch._generator_used = orch._seed_generator_used
    orch._drop_redundant_generator_tools()
    orch._lock_application_generators()


def _write_recipe(tmp_path, generator_used, **model):
    (tmp_path / ".besser_recipe.json").write_text(
        json.dumps({"generator_used": generator_used, "model": model}))


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

    orch._lock_application_generators()  # Phase 2 entry
    result = json.loads(orch.executor.execute("generate_fastapi_backend", {}))

    assert "error" in result and "start a fresh run" in result["error"]
    with open(main, encoding="utf-8") as fh:
        assert fh.read() == edited


def test_primaries_leave_the_phase2_toolset(tmp_path):
    orch = _orchestrator(tmp_path)
    orch._generator_used = "generate_fastapi_backend"
    orch._lock_application_generators()
    assert not (_names(orch.tools) & _APPLICATION_PRIMARY_TOOLS)
    assert "write_file" in _names(orch.tools)


def test_phase1_generator_call_is_not_refused(tmp_path):
    orch = _orchestrator(tmp_path)
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"


def test_no_scaffold_keeps_primaries(tmp_path):
    # Phase 1 failed or picked none: a primary in Phase 2 overwrites nothing.
    orch = _orchestrator(tmp_path)
    orch._generator_used = None
    orch._lock_application_generators()
    assert "generate_fastapi_backend" in _names(orch.tools)


def test_modify_run_on_a_primary_built_seed_locks_all_primaries(tmp_path):
    _write_recipe(tmp_path, "generate_web_app", classes=["Book"])
    (tmp_path / "web_app").mkdir()
    orch = _orchestrator(tmp_path)
    _modify_run(orch)
    assert not (_names(orch.tools) & _APPLICATION_PRIMARY_TOOLS)
    assert _refused(orch, "generate_django")


def test_modify_run_locks_the_primary_whose_folder_an_llm_seed_has(tmp_path):
    (tmp_path / "backend").mkdir()
    main = tmp_path / "backend" / "main_api.py"
    main.write_text("# hand-written\n")
    orch = _orchestrator(tmp_path)
    _modify_run(orch)

    assert _refused(orch, "generate_fastapi_backend")
    assert main.read_text() == "# hand-written\n"
    # generate_django writes django/, which this seed does not have.
    assert "generate_django" in _names(orch.tools)


def test_modify_run_on_a_seed_without_an_app_keeps_primaries(tmp_path):
    # An agent-only or state-machine seed: nothing a primary would overwrite.
    _write_recipe(tmp_path, "generate_baf", primary_kind="agent")
    (tmp_path / "agent").mkdir()
    orch = _orchestrator(tmp_path)
    _modify_run(orch)
    assert "generate_fastapi_backend" in _names(orch.tools)
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"


def test_agent_seed_with_a_class_diagram_keeps_primaries(tmp_path):
    _write_recipe(tmp_path, "generate_baf", primary_kind="agent", classes=["Customer"])
    (tmp_path / "agent").mkdir()
    orch = _orchestrator(tmp_path)
    offered = _names(orch.tools) & _APPLICATION_PRIMARY_TOOLS
    _modify_run(orch)
    assert offered and offered <= _names(orch.tools)


def test_python_classes_seed_can_add_a_backend(tmp_path):
    _write_recipe(tmp_path, "generate_python_classes", primary_kind="class", classes=["Book"])
    (tmp_path / "python").mkdir()
    orch = _orchestrator(tmp_path)
    _modify_run(orch)
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"


def test_frontend_only_seed_can_add_a_backend(tmp_path):
    (tmp_path / "frontend").mkdir()
    orch = _orchestrator(tmp_path)
    _modify_run(orch)
    assert json.loads(orch.executor.execute("generate_fastapi_backend", {}))["status"] == "ok"


def test_resumed_modify_run_keeps_the_lock(tmp_path):
    instructions = "Add a search endpoint"
    (tmp_path / "backend").mkdir()
    main = tmp_path / "backend" / "main_api.py"
    main.write_text("# hand-written\n")
    crashed = _orchestrator(tmp_path, checkpointing=True)
    crashed._project_fingerprint = compute_fingerprint(
        instructions=instructions, primary_kind=crashed.primary_kind, domain_model=crashed.domain_model)
    _modify_run(crashed)
    crashed._save_checkpoint_for_turn(1, [{"role": "user", "content": instructions}], instructions)
    assert load_checkpoint(str(tmp_path)).locked_generators == ["generate_fastapi_backend"]

    client = _ScriptedClient()
    resumed = _orchestrator(tmp_path, client=client, checkpointing=True)
    assert resumed._modify_mode is False  # resume() does not restore it
    resumed.resume(instructions)

    assert any("start a fresh run" in str(r.get("content")) for r in client.tool_results)
    assert main.read_text() == "# hand-written\n"
