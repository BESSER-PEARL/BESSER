"""Tests for the LLM tool executor."""

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.tool_executor import ToolExecutor


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


@pytest.fixture
def executor(tmp_path, simple_model):
    return ToolExecutor(workspace=str(tmp_path), domain_model=simple_model)


def _call(executor, tool, args=None):
    """Execute a tool and return parsed JSON result."""
    return json.loads(executor.execute(tool, args or {}))


# ======================================================================
# File tools
# ======================================================================

class TestFileTools:

    def test_write_and_read(self, executor):
        _call(executor, "write_file", {"path": "test.py", "content": "x = 1"})
        result = _call(executor, "read_file", {"path": "test.py"})
        assert result["content"] == "x = 1"

    def test_write_creates_subdirs(self, executor):
        _call(executor, "write_file", {"path": "deep/dir/file.py", "content": "y = 2"})
        result = _call(executor, "read_file", {"path": "deep/dir/file.py"})
        assert result["content"] == "y = 2"

    def test_modify_file(self, executor):
        _call(executor, "write_file", {"path": "app.py", "content": "name = 'old'"})
        result = _call(executor, "modify_file", {"path": "app.py", "old_text": "'old'", "new_text": "'new'"})
        assert result["status"] == "modified"
        assert _call(executor, "read_file", {"path": "app.py"})["content"] == "name = 'new'"

    def test_modify_not_found(self, executor):
        assert "error" in _call(executor, "modify_file", {"path": "nope.py", "old_text": "x", "new_text": "y"})

    def test_modify_old_text_missing(self, executor):
        _call(executor, "write_file", {"path": "a.py", "content": "x = 1"})
        result = _call(executor, "modify_file", {"path": "a.py", "old_text": "NOPE", "new_text": "y"})
        assert "error" in result and "not found" in result["error"]

    def test_list_files(self, executor):
        _call(executor, "write_file", {"path": "a.py", "content": "1"})
        _call(executor, "write_file", {"path": "sub/b.py", "content": "2"})
        paths = {f["path"] for f in _call(executor, "list_files")["files"]}
        assert "a.py" in paths and "sub/b.py" in paths

    def test_read_missing(self, executor):
        assert "error" in _call(executor, "read_file", {"path": "nope.py"})

    def test_path_traversal_blocked(self, executor):
        assert "error" in _call(executor, "read_file", {"path": "../../etc/passwd"})

    def test_search_in_files(self, executor):
        _call(executor, "write_file", {"path": "one.py", "content": "def hello():\n    pass"})
        _call(executor, "write_file", {"path": "two.py", "content": "def world():\n    pass"})
        result = _call(executor, "search_in_files", {"pattern": "def hello"})
        assert result["total"] == 1
        assert result["matches"][0]["file"] == "one.py"

    def test_search_with_glob(self, executor):
        _call(executor, "write_file", {"path": "a.py", "content": "hello"})
        _call(executor, "write_file", {"path": "b.txt", "content": "hello"})
        result = _call(executor, "search_in_files", {"pattern": "hello", "file_glob": "*.py"})
        assert result["total"] == 1


# ======================================================================
# Execution tools
# ======================================================================

class TestExecutionTools:

    def test_run_command_success(self, executor):
        result = _call(executor, "run_command", {"command": "python -c \"print('hello')\" "})
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_run_command_failure(self, executor):
        result = _call(executor, "run_command", {"command": "python -c \"raise ValueError('boom')\" "})
        assert result["success"] is False
        assert "boom" in result["stderr"]

    def test_run_command_in_subdir(self, executor):
        _call(executor, "write_file", {"path": "subdir/test.py", "content": "print('sub')"})
        result = _call(executor, "run_command", {"command": "python test.py", "working_dir": "subdir"})
        assert result["success"] is True
        assert "sub" in result["stdout"]

    def test_run_command_timeout(self, executor):
        """Commands that exceed timeout return an error."""
        import besser.generators.llm.tool_executor as mod
        old_timeout = mod.COMMAND_TIMEOUT
        mod.COMMAND_TIMEOUT = 1  # 1 second
        try:
            result = _call(executor, "run_command", {"command": "python -c \"import time; time.sleep(10)\" "})
            assert "error" in result and "timed out" in result["error"]
        finally:
            mod.COMMAND_TIMEOUT = old_timeout

    def test_install_dependencies_no_files(self, executor):
        result = _call(executor, "install_dependencies", {})
        assert "error" in result

    def test_install_dependencies_custom_command(self, executor):
        result = _call(executor, "install_dependencies", {"command": "python --version"})
        assert result["exit_code"] == 0


# ======================================================================
# Validation tools
# ======================================================================

class TestValidationTools:

    def test_check_syntax_valid(self, executor):
        _call(executor, "write_file", {"path": "good.py", "content": "x = 1\ny = x + 2"})
        assert _call(executor, "check_syntax", {"path": "good.py"})["status"] == "ok"

    def test_check_syntax_invalid(self, executor):
        _call(executor, "write_file", {"path": "bad.py", "content": "def f(\n    x ="})
        result = _call(executor, "check_syntax", {"path": "bad.py"})
        assert "error" in result and "Syntax error" in result["error"]


# ======================================================================
# Generator tools
# ======================================================================

class TestGeneratorTools:

    def test_generate_pydantic(self, executor):
        result = _call(executor, "generate_pydantic")
        assert result["status"] == "ok"
        assert any("pydantic" in f["path"] for f in result["files"])

    def test_generate_python_classes(self, executor):
        result = _call(executor, "generate_python_classes")
        assert result["status"] == "ok"

    def test_generate_sql(self, executor):
        result = _call(executor, "generate_sql")
        assert result["status"] == "ok"
        assert any(f["path"].endswith(".sql") for f in result["files"])

    def test_generate_sqlalchemy(self, executor):
        assert _call(executor, "generate_sqlalchemy", {"dbms": "sqlite"})["status"] == "ok"

    def test_generate_json_schema(self, executor):
        assert _call(executor, "generate_json_schema")["status"] == "ok"

    def test_generate_react_fails_without_gui(self, executor):
        result = _call(executor, "generate_react")
        assert "error" in result and "GUI" in result["error"]

    def test_unknown_tool(self, executor):
        assert "error" in _call(executor, "nonexistent_tool")


# ======================================================================
# Orchestrator integration (mock LLM)
# ======================================================================

class TestOrchestratorIntegration:

    def test_generate_test_fix_loop(self, simple_model, tmp_path):
        """Simulate the full generate → test → fix loop with a mock LLM."""

        class MockBlock:
            def __init__(self, block_type, **kwargs):
                self.type = block_type
                for k, v in kwargs.items():
                    setattr(self, k, v)

        turn_counter = {"n": 0}

        class MockClient:
            model = "mock-model"

            def chat(self, system, messages, tools):
                turn_counter["n"] += 1
                n = turn_counter["n"]

                if n == 1:
                    # Step 1: Generate backend
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="generate_pydantic", input={}, id="c1"),
                    ]}
                elif n == 2:
                    # Step 2: Write a test file
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="write_file", input={
                            "path": "test_app.py",
                            "content": "import sys\nprint('tests pass')\nsys.exit(0)",
                        }, id="c2"),
                    ]}
                elif n == 3:
                    # Step 3: Run the test
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="run_command", input={
                            "command": "python test_app.py",
                        }, id="c3"),
                    ]}
                elif n == 4:
                    # Step 4: Write README
                    return {"stop_reason": "tool_use", "content": [
                        MockBlock("tool_use", name="write_file", input={
                            "path": "README.md",
                            "content": "# App\nRun: python test_app.py\n",
                        }, id="c4"),
                    ]}
                else:
                    return {"stop_reason": "end_turn", "content": [
                        MockBlock("text", text="Done!"),
                    ]}

        from besser.generators.llm.orchestrator import LLMOrchestrator

        orchestrator = LLMOrchestrator(
            llm_client=MockClient(),
            domain_model=simple_model,
            output_dir=str(tmp_path),
        )

        progress_log = []
        orchestrator.on_progress = lambda turn, tool, status: progress_log.append((turn, tool))

        result_dir = orchestrator.run("Build me an app")

        # Verify outputs
        assert os.path.isdir(os.path.join(result_dir, "pydantic"))
        assert os.path.isfile(os.path.join(result_dir, "test_app.py"))
        assert os.path.isfile(os.path.join(result_dir, "README.md"))
        assert os.path.isfile(os.path.join(result_dir, ".besser_recipe.json"))

        # Verify progress callbacks fired
        assert len(progress_log) >= 4

        # Verify recipe has metadata
        with open(os.path.join(result_dir, ".besser_recipe.json")) as f:
            recipe = json.load(f)
        assert recipe["model_name"] == "TestModel"
        assert recipe["turns"] == 5
        assert len(recipe["tool_calls"]) == 4
