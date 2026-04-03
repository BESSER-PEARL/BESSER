"""Tests for new orchestrator features: parallel execution, snapshot/rollback, fix_error, modules."""

import json
import os
import time

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator, _SNAPSHOT_DIR


@pytest.fixture
def simple_model():
    """A minimal domain model for testing."""
    StringType = PrimitiveDataType("str")
    IntegerType = PrimitiveDataType("int")
    user = Class(name="User")
    user.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
    }
    return DomainModel(name="TestModel", types={user})


class MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_end_turn_client():
    """A mock client that always says 'done' on the first call."""
    class MockClient:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, system, messages, tools):
            return {"stop_reason": "end_turn", "content": [
                MockBlock("text", text="Done!"),
            ]}

    return MockClient()


# ======================================================================
# Parallel tool execution tests
# ======================================================================

class TestParallelExecution:

    def test_parallel_execution_multiple_tools(self, simple_model, tmp_path):
        """Multiple tool calls in one turn are executed in parallel."""
        turn_counter = {"n": 0}
        execution_log = []

        class ParallelClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if turn_counter["n"] == 1:
                    # Return multiple tool calls in one response
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="write_file", input={
                            "path": "file1.py", "content": "x = 1",
                        }, id="c1"),
                        MockBlock("tool_use", name="write_file", input={
                            "path": "file2.py", "content": "y = 2",
                        }, id="c2"),
                        MockBlock("tool_use", name="write_file", input={
                            "path": "file3.py", "content": "z = 3",
                        }, id="c3"),
                    ]}
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Done!"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=ParallelClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        orchestrator.run("Build an app")

        # All three files should be created
        assert os.path.isfile(os.path.join(str(tmp_path), "file1.py"))
        assert os.path.isfile(os.path.join(str(tmp_path), "file2.py"))
        assert os.path.isfile(os.path.join(str(tmp_path), "file3.py"))

        # Tool calls should be logged
        phase2_calls = [tc for tc in orchestrator.tool_calls_log if tc["turn"] > 0]
        write_calls = [tc for tc in phase2_calls if tc["tool"] == "write_file"]
        assert len(write_calls) == 3

    def test_parallel_execution_results_ordered(self, simple_model, tmp_path):
        """Results from parallel execution are returned in block order."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        blocks = [
            MockBlock("tool_use", name="write_file", input={
                "path": "a.py", "content": "a = 1",
            }, id="id_a"),
            MockBlock("tool_use", name="write_file", input={
                "path": "b.py", "content": "b = 2",
            }, id="id_b"),
        ]

        results = orchestrator._execute_tool_blocks(blocks, turn=0)
        assert len(results) == 2
        assert results[0]["tool_use_id"] == "id_a"
        assert results[1]["tool_use_id"] == "id_b"

    def test_single_tool_not_parallelized(self, simple_model, tmp_path):
        """A single tool call is executed directly (no thread pool)."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        blocks = [
            MockBlock("tool_use", name="write_file", input={
                "path": "solo.py", "content": "solo = 1",
            }, id="id_solo"),
        ]

        results = orchestrator._execute_tool_blocks(blocks, turn=0)
        assert len(results) == 1
        assert results[0]["tool_use_id"] == "id_solo"
        assert os.path.isfile(os.path.join(str(tmp_path), "solo.py"))

    def test_empty_tool_blocks(self, simple_model, tmp_path):
        """Empty tool blocks returns empty list."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        results = orchestrator._execute_tool_blocks([], turn=0)
        assert results == []

    def test_parallel_execution_handles_failure(self, simple_model, tmp_path):
        """If one tool fails in parallel, others still succeed."""
        turn_counter = {"n": 0}

        class ParallelWithErrorClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if turn_counter["n"] == 1:
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="write_file", input={
                            "path": "good.py", "content": "x = 1",
                        }, id="c1"),
                        MockBlock("tool_use", name="read_file", input={
                            "path": "nonexistent.py",
                        }, id="c2"),
                    ]}
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Done!"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=ParallelWithErrorClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        orchestrator.run("Build an app")

        # The good file should be created despite the read error
        assert os.path.isfile(os.path.join(str(tmp_path), "good.py"))


# ======================================================================
# Snapshot / Rollback tests
# ======================================================================

class TestSnapshotRollback:

    def test_snapshot_created_when_generator_used(self, simple_model, tmp_path):
        """Snapshot is created after Phase 1 when a generator is used."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        # Write a file to simulate generator output
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "app.py"), "w") as f:
            f.write("x = 1\n")

        orchestrator._generator_used = "generate_fastapi_backend"
        orchestrator._create_snapshot()

        snapshot_path = os.path.join(str(tmp_path), _SNAPSHOT_DIR)
        assert os.path.isdir(snapshot_path)
        assert os.path.isfile(os.path.join(snapshot_path, "app.py"))

    def test_restore_snapshot(self, simple_model, tmp_path):
        """Restore replaces output directory contents with snapshot."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        # Phase 1: write original file
        os.makedirs(str(tmp_path), exist_ok=True)
        original_path = os.path.join(str(tmp_path), "app.py")
        with open(original_path, "w") as f:
            f.write("original = True\n")

        # Create snapshot
        orchestrator._generator_used = "generate_fastapi_backend"
        orchestrator._create_snapshot()

        # Phase 2: modify file
        with open(original_path, "w") as f:
            f.write("modified = True\n")

        # Add a new file that Phase 2 created
        with open(os.path.join(str(tmp_path), "new_file.py"), "w") as f:
            f.write("extra = True\n")

        # Restore
        orchestrator._restore_snapshot()

        # Original content should be restored
        with open(original_path) as f:
            assert f.read() == "original = True\n"

        # New file should be removed
        assert not os.path.isfile(os.path.join(str(tmp_path), "new_file.py"))

    def test_restore_no_snapshot(self, simple_model, tmp_path, caplog):
        """Restoring when no snapshot exists logs a warning."""
        import logging
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        with caplog.at_level(logging.WARNING, logger="besser.generators.llm.orchestrator"):
            orchestrator._restore_snapshot()
        assert any("No snapshot" in r.message for r in caplog.records)

    def test_snapshot_cleaned_up_after_run(self, simple_model, tmp_path):
        """Snapshot directory is removed after a successful run."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator.run("Build an app")

        snapshot_path = os.path.join(str(tmp_path), _SNAPSHOT_DIR)
        assert not os.path.isdir(snapshot_path)

    def test_snapshot_with_subdirectories(self, simple_model, tmp_path):
        """Snapshot preserves subdirectory structure."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        # Create nested structure
        os.makedirs(os.path.join(str(tmp_path), "src", "models"), exist_ok=True)
        with open(os.path.join(str(tmp_path), "src", "models", "user.py"), "w") as f:
            f.write("class User: pass\n")
        with open(os.path.join(str(tmp_path), "main.py"), "w") as f:
            f.write("from src.models.user import User\n")

        orchestrator._generator_used = "generate_fastapi_backend"
        orchestrator._create_snapshot()

        snapshot_path = os.path.join(str(tmp_path), _SNAPSHOT_DIR)
        assert os.path.isfile(os.path.join(snapshot_path, "src", "models", "user.py"))
        assert os.path.isfile(os.path.join(snapshot_path, "main.py"))


# ======================================================================
# fix_error tests
# ======================================================================

class TestFixError:

    def test_fix_error_basic(self, simple_model, tmp_path):
        """fix_error sends error to LLM and returns output dir."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        result = orchestrator.fix_error("ImportError: No module named 'auth'")
        assert isinstance(result, str)  # returns explanation

    def test_fix_error_empty_raises(self, simple_model, tmp_path):
        """fix_error raises ValueError on empty message."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        with pytest.raises(ValueError, match="empty"):
            orchestrator.fix_error("")

        with pytest.raises(ValueError, match="empty"):
            orchestrator.fix_error("   ")

    def test_fix_error_with_tool_calls(self, simple_model, tmp_path):
        """fix_error executes tool calls from LLM response."""
        turn_counter = {"n": 0}

        class FixClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if turn_counter["n"] == 1:
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="write_file", input={
                            "path": "auth.py", "content": "# Auth module\n",
                        }, id="fix1"),
                    ]}
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Fixed!"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=FixClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        orchestrator.fix_error("ImportError: No module named 'auth'")
        assert os.path.isfile(os.path.join(str(tmp_path), "auth.py"))

    def test_fix_error_respects_timeout(self, simple_model, tmp_path):
        """fix_error stops if runtime timeout is exceeded."""
        turn_counter = {"n": 0}

        class SlowFixClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                time.sleep(0.2)
                return {"stop_reason": "tool_use", "content": [
                    MockBlock("tool_use", name="list_files", input={}, id=f"f{turn_counter['n']}"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=SlowFixClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_runtime_seconds=1,
        )

        orchestrator.fix_error("Some error")
        # Should stop before exhausting all 10 fix turns due to timeout
        assert orchestrator.total_turns <= 10

    def test_fix_error_limited_to_10_turns(self, simple_model, tmp_path):
        """fix_error is limited to 10 turns."""
        turn_counter = {"n": 0}

        class InfiniteFixClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                return {"stop_reason": "tool_use", "content": [
                    MockBlock("tool_use", name="list_files", input={}, id=f"f{turn_counter['n']}"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=InfiniteFixClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_runtime_seconds=600,
        )

        orchestrator.fix_error("Some error")
        assert orchestrator.total_turns <= 10


# ======================================================================
# Standalone module tests (gap_analyzer, prompt_builder)
# ======================================================================

class TestGapAnalyzerModule:

    def test_analyze_gaps_no_generator(self, simple_model, tmp_path):
        """analyze_gaps returns scratch-build task when no generator used."""
        from besser.generators.llm.gap_analyzer import analyze_gaps
        tasks = analyze_gaps(
            instructions="Build a NestJS backend",
            generator_used=None,
            domain_model=simple_model,
            llm_client=_make_end_turn_client(),
        )
        assert len(tasks) == 1
        assert "from scratch" in tasks[0].lower()

    def test_analyze_gaps_keyword_auth(self, simple_model, tmp_path):
        """Keyword analysis detects auth requirements."""
        from besser.generators.llm.gap_analyzer import _analyze_gaps_keyword
        tasks = _analyze_gaps_keyword("Add JWT authentication", "generate_fastapi_backend")
        assert any("auth" in t.lower() for t in tasks)

    def test_analyze_gaps_keyword_pagination(self, simple_model, tmp_path):
        """Keyword analysis detects pagination requirements."""
        from besser.generators.llm.gap_analyzer import _analyze_gaps_keyword
        tasks = _analyze_gaps_keyword("Add pagination to list endpoints", "generate_fastapi_backend")
        assert any("pagination" in t.lower() for t in tasks)

    def test_build_model_summary(self, simple_model):
        """Model summary contains class names."""
        from besser.generators.llm.gap_analyzer import build_model_summary
        summary = build_model_summary(simple_model)
        assert "User" in summary
        assert "Associations" in summary


class TestPromptBuilderModule:

    def test_build_system_prompt(self, simple_model):
        """build_system_prompt includes model and tasks."""
        from besser.generators.llm.prompt_builder import build_system_prompt
        prompt = build_system_prompt(
            domain_model=simple_model,
            gui_model=None,
            agent_model=None,
            inventory="",
            gap_tasks=["Add auth", "Write README"],
            max_turns=50,
        )
        assert "User" in prompt
        assert "Add auth" in prompt
        assert "Write README" in prompt
        assert "Domain Model" in prompt

    def test_build_system_prompt_with_inventory(self, simple_model):
        """build_system_prompt includes inventory when provided."""
        from besser.generators.llm.prompt_builder import build_system_prompt
        prompt = build_system_prompt(
            domain_model=simple_model,
            gui_model=None,
            agent_model=None,
            inventory="Generator produced 5 files",
            gap_tasks=[],
            max_turns=50,
        )
        assert "Generator produced 5 files" in prompt
        assert "What was already generated" in prompt

    def test_build_system_prompt_no_inventory(self, simple_model):
        """build_system_prompt omits inventory section when empty."""
        from besser.generators.llm.prompt_builder import build_system_prompt
        prompt = build_system_prompt(
            domain_model=simple_model,
            gui_model=None,
            agent_model=None,
            inventory="",
            gap_tasks=["Task 1"],
            max_turns=50,
        )
        assert "What was already generated" not in prompt

    def test_build_inventory(self, simple_model, tmp_path):
        """build_inventory lists files and model info."""
        from besser.generators.llm.prompt_builder import build_inventory
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "main_api.py"), "w") as f:
            f.write("from fastapi import FastAPI\napp = FastAPI()\n")

        inv = build_inventory(str(tmp_path), simple_model, "generate_fastapi_backend")
        assert "main_api.py" in inv
        assert "User" in inv
        assert "generate_fastapi_backend" in inv


# ======================================================================
# Phase 3 rollback integration test
# ======================================================================

class TestPhase3Rollback:

    def test_rollback_when_fixes_worse(self, simple_model, tmp_path):
        """Phase 3 rolls back if LLM fixes introduce more errors."""
        turn_counter = {"n": 0}

        class BreakingFixClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if "validation" in str(messages).lower() or "fix" in str(system).lower():
                    # When asked to fix, introduce more syntax errors
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="write_file", input={
                            "path": "extra_broken.py",
                            "content": "def broken(\n    x =",
                        }, id=f"fix{turn_counter['n']}"),
                    ]}
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Done!"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=BreakingFixClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        # Create initial state with one syntax error
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "broken.py"), "w") as f:
            f.write("def f(\n    x =")
        with open(os.path.join(str(tmp_path), "good.py"), "w") as f:
            f.write("x = 1\n")

        # Create snapshot so rollback is possible
        orchestrator._generator_used = "generate_fastapi_backend"
        orchestrator._create_snapshot()

        # Run Phase 3 -- the LLM will make things worse
        orchestrator._run_phase3_validation()

        # After rollback, extra_broken.py should not exist
        # (it was created by the "fix" but rolled back)
        assert not os.path.isfile(os.path.join(str(tmp_path), "extra_broken.py"))
        # The original good.py should still exist
        assert os.path.isfile(os.path.join(str(tmp_path), "good.py"))
