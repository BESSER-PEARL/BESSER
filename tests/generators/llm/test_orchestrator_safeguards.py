"""Tests for orchestrator safeguards: cost cap, runtime timeout, phase 1 validation, LLM gap analysis."""

import json
import os
import time

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator


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
# Cost cap tests
# ======================================================================

class TestCostCap:

    def test_cost_cap_stops_generation(self, simple_model, tmp_path):
        """Generation stops when estimated cost exceeds max_cost_usd."""
        turn_counter = {"n": 0}

        class ExpensiveClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                # Simulate expensive API calls by inflating usage
                self.usage.input_tokens += 500_000
                self.usage.output_tokens += 100_000
                self.usage.api_calls += 1
                # Always request a tool call so loop continues
                return {"stop_reason": "tool_use", "content": [
                    MockBlock("tool_use", name="list_files", input={}, id=f"c{turn_counter['n']}"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=ExpensiveClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_turns=50,
            max_cost_usd=0.01,  # Very low cap to trigger quickly
        )

        orchestrator.run("Build an app")

        # Should have stopped early due to cost cap (not exhausted all 50 turns)
        assert orchestrator.total_turns < 50
        # Verify cost exceeded the cap
        assert orchestrator.client.usage.estimated_cost > 0.01

    def test_cost_cap_default_is_5(self, simple_model, tmp_path):
        """Default max_cost_usd is 5.0."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        assert orchestrator.max_cost_usd == 5.0

    def test_cost_cap_in_recipe(self, simple_model, tmp_path):
        """Cost cap appears in the saved recipe."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_cost_usd=3.5,
        )
        orchestrator.run("Build an app")

        recipe_path = os.path.join(str(tmp_path), ".besser_recipe.json")
        with open(recipe_path) as f:
            recipe = json.load(f)
        assert recipe["max_cost_usd"] == 3.5

    def test_cost_warning_at_80_percent(self, simple_model, tmp_path, caplog):
        """Warning fires at 80% of cost cap."""
        import logging
        turn_counter = {"n": 0}

        class ModerateClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                # Push cost past 80% on first call, then stop
                if turn_counter["n"] == 1:
                    # Sonnet pricing: input=$3/M, output=$15/M
                    # 100k output tokens = $1.50
                    self.usage.output_tokens += 100_000
                    self.usage.api_calls += 1
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="list_files", input={}, id="c1"),
                    ]}
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Done"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=ModerateClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_cost_usd=1.0,  # 80% = $0.80
        )

        with caplog.at_level(logging.WARNING, logger="besser.generators.llm.orchestrator"):
            orchestrator.run("Build an app")

        # Check that 80% warning was logged
        assert any("80%" in record.message for record in caplog.records)


# ======================================================================
# Runtime timeout tests
# ======================================================================

class TestRuntimeTimeout:

    def test_runtime_timeout_stops_generation(self, simple_model, tmp_path):
        """Generation stops when runtime exceeds max_runtime_seconds."""
        turn_counter = {"n": 0}

        class SlowClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                # Simulate slow work
                time.sleep(0.1)
                return {"stop_reason": "tool_use", "content": [
                    MockBlock("tool_use", name="list_files", input={}, id=f"c{turn_counter['n']}"),
                ]}

        orchestrator = LLMOrchestrator(
            llm_client=SlowClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_turns=100,
            max_runtime_seconds=1,  # 1 second — very short
        )

        orchestrator.run("Build an app")

        # Should have stopped early due to timeout (not all 100 turns)
        assert orchestrator.total_turns < 100

    def test_runtime_timeout_default_is_1200(self, simple_model, tmp_path):
        """Default max_runtime_seconds is 1200 (20 minutes)."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        assert orchestrator.max_runtime_seconds == 1200

    def test_runtime_timeout_in_recipe(self, simple_model, tmp_path):
        """Runtime timeout appears in the saved recipe."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_runtime_seconds=600,
        )
        orchestrator.run("Build an app")

        recipe_path = os.path.join(str(tmp_path), ".besser_recipe.json")
        with open(recipe_path) as f:
            recipe = json.load(f)
        assert recipe["max_runtime_seconds"] == 600

    def test_phase3_skipped_on_timeout(self, simple_model, tmp_path, caplog):
        """Phase 3 is skipped when runtime budget is exhausted."""
        import logging

        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
            max_runtime_seconds=0,  # Already expired
        )
        # Set start time in the past so timeout is hit
        orchestrator._start_time = time.monotonic() - 100

        # Write a file with syntax error to trigger Phase 3
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "bad.py"), "w") as f:
            f.write("def f(\n    x =")

        with caplog.at_level(logging.WARNING, logger="besser.generators.llm.orchestrator"):
            orchestrator._run_phase3_validation()

        assert any("Skipping Phase 3" in record.message for record in caplog.records)


# ======================================================================
# Phase 1 validation tests
# ======================================================================

class TestPhase1Validation:

    def test_detects_python_syntax_errors(self, simple_model, tmp_path):
        """Phase 1 validation catches Python syntax errors."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        # Write a file with a syntax error
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "broken.py"), "w") as f:
            f.write("def f(\n    x =")

        issues = orchestrator._validate_phase1_output()
        assert len(issues) >= 1
        assert any("syntax error" in i.lower() or "Syntax error" in i for i in issues)

    def test_detects_missing_requirements_txt(self, simple_model, tmp_path):
        """Phase 1 validation catches Dockerfile referencing missing requirements.txt."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "Dockerfile"), "w") as f:
            f.write("FROM python:3.11\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n")

        issues = orchestrator._validate_phase1_output()
        assert len(issues) >= 1
        assert any("requirements.txt" in i for i in issues)

    def test_detects_missing_package_json(self, simple_model, tmp_path):
        """Phase 1 validation catches Dockerfile referencing missing package.json."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "Dockerfile"), "w") as f:
            f.write("FROM node:18\nCOPY package.json .\nRUN npm install\n")

        issues = orchestrator._validate_phase1_output()
        assert len(issues) >= 1
        assert any("package.json" in i for i in issues)

    def test_no_issues_when_files_exist(self, simple_model, tmp_path):
        """No issues reported when referenced files exist."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "Dockerfile"), "w") as f:
            f.write("FROM python:3.11\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n")
        with open(os.path.join(str(tmp_path), "requirements.txt"), "w") as f:
            f.write("fastapi\n")
        with open(os.path.join(str(tmp_path), "app.py"), "w") as f:
            f.write("x = 1\n")

        issues = orchestrator._validate_phase1_output()
        assert len(issues) == 0

    def test_valid_python_passes(self, simple_model, tmp_path):
        """Valid Python files pass validation."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "good.py"), "w") as f:
            f.write("def hello():\n    return 'world'\n")

        issues = orchestrator._validate_phase1_output()
        assert len(issues) == 0

    def test_phase1_issues_added_to_gap_tasks(self, simple_model, tmp_path):
        """Phase 1 validation issues are passed as extra gap tasks to Phase 2."""
        turn_counter = {"n": 0}

        class InspectingClient:
            """Client that captures the system prompt to verify tasks."""
            model = "mock-model"
            usage = UsageTracker("mock-model")
            captured_system = None

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                if self.captured_system is None:
                    self.captured_system = system
                return {"stop_reason": "end_turn", "content": [
                    MockBlock("text", text="Done!"),
                ]}

        client = InspectingClient()
        orchestrator = LLMOrchestrator(
            llm_client=client,
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        # Write a file with syntax error
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "broken.py"), "w") as f:
            f.write("def f(\n    x =")

        orchestrator.run("Build an app")

        # The system prompt should contain the syntax error issue
        assert client.captured_system is not None
        assert "syntax error" in client.captured_system.lower() or "Syntax error" in client.captured_system


# ======================================================================
# LLM-based gap analysis tests
# ======================================================================

class TestLLMGapAnalysis:

    def test_keyword_fallback_when_generator_used(self, simple_model, tmp_path):
        """Keyword fallback works when LLM gap analysis fails."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator._generator_used = "generate_fastapi_backend"

        # _analyze_gaps_with_llm will fail (no real API key) and fall back to keywords
        tasks = orchestrator._analyze_gaps("Build a FastAPI backend with JWT auth and PostgreSQL")

        # Should get keyword-based tasks (auth + db + docker + readme)
        task_text = " ".join(tasks).lower()
        assert "auth" in task_text
        assert "readme" in task_text

    def test_keyword_fallback_no_generator(self, simple_model, tmp_path):
        """When no generator is used, returns scratch-build task."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator._generator_used = None

        tasks = orchestrator._analyze_gaps("Build a NestJS backend")
        assert len(tasks) == 1
        assert "from scratch" in tasks[0].lower()

    def test_analyze_gaps_keyword_auth(self, simple_model, tmp_path):
        """Keyword analysis detects auth requirements."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator._generator_used = "generate_fastapi_backend"

        tasks = orchestrator._analyze_gaps_keyword("Add JWT authentication")
        assert any("auth" in t.lower() for t in tasks)

    def test_analyze_gaps_keyword_docker(self, simple_model, tmp_path):
        """Keyword analysis detects Docker requirements."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator._generator_used = "generate_fastapi_backend"

        tasks = orchestrator._analyze_gaps_keyword("Deploy with Docker")
        assert any("docker" in t.lower() for t in tasks)

    def test_analyze_gaps_keyword_always_readme(self, simple_model, tmp_path):
        """Keyword analysis always includes README task."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator._generator_used = "generate_fastapi_backend"

        tasks = orchestrator._analyze_gaps_keyword("Build a simple backend")
        assert any("readme" in t.lower() for t in tasks)

    def test_build_model_summary(self, simple_model, tmp_path):
        """Model summary contains class names."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        summary = orchestrator._build_model_summary()
        assert "User" in summary

    def test_analyze_gaps_with_llm_returns_none_on_failure(self, simple_model, tmp_path):
        """LLM gap analysis returns None when API call fails."""
        orchestrator = LLMOrchestrator(
            llm_client=_make_end_turn_client(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )
        orchestrator._generator_used = "generate_fastapi_backend"

        # This will fail because there's no real API key
        result = orchestrator._analyze_gaps_with_llm("Build an app")
        assert result is None
