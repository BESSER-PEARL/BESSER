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
from besser.spec_driven_agent.tool_executor import ToolExecutor


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


@pytest.fixture
def shell_executor(tmp_path, simple_model):
    """An executor that opted into the shell tools.

    Since 2026-09-14 ``allow_shell`` defaults to False and ``execute_typed``
    refuses run_command / install_dependencies outright, so the tests that
    exercise the shell implementations must ask for it explicitly. Using the
    plain ``executor`` fixture here would make them pass on the refusal
    instead of on the behaviour under test.
    """
    return ToolExecutor(
        workspace=str(tmp_path), domain_model=simple_model, allow_shell=True
    )


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
        assert result["content"] == "   1| x = 1"

    def test_write_creates_subdirs(self, executor):
        _call(executor, "write_file", {"path": "deep/dir/file.py", "content": "y = 2"})
        result = _call(executor, "read_file", {"path": "deep/dir/file.py"})
        assert result["content"] == "   1| y = 2"

    def test_modify_file(self, executor):
        _call(executor, "write_file", {"path": "app.py", "content": "name = 'old'"})
        result = _call(executor, "modify_file", {"path": "app.py", "old_text": "'old'", "new_text": "'new'"})
        assert result["status"] == "modified"
        assert _call(executor, "read_file", {"path": "app.py"})["content"] == "   1| name = 'new'"

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

    def test_read_file_with_offset_limit(self, executor):
        """Line-based pagination: read specific line range."""
        content = "\n".join(f"line {i}" for i in range(100))
        _call(executor, "write_file", {"path": "big.py", "content": content})
        result = _call(executor, "read_file", {"path": "big.py", "offset": 10, "limit": 5})
        assert result["start_line"] == 11  # 1-indexed
        assert result["end_line"] == 15
        assert result["total_lines"] == 100
        assert result["lines_read"] == 5
        assert "line 10" in result["content"]
        assert "line 14" in result["content"]
        assert "line 15" not in result["content"]

    def test_read_file_large_reports_size_without_urging_pagination(self, executor):
        """A large file still fits in MAX_FILE_READ, so report its size only.

        The old unconditional "Large file. Use offset/limit to read specific
        sections." hint is what taught run trilraak to crawl a 598-line router
        in 10-line windows - 36 reads, a third of its turn budget. Pagination
        advice now appears only when a read is actually truncated.
        """
        content = "\n".join(f"x = {i}" for i in range(300))
        _call(executor, "write_file", {"path": "large.py", "content": content})
        result = _call(executor, "read_file", {"path": "large.py"})
        assert result["total_lines"] == 300
        assert "hint" not in result

    def test_read_file_numbers_lines_from_one(self, executor):
        """Numbered in the post-edit echo's format, so a quote copied out of a
        read_file result is quotable straight back into modify_file."""
        _call(executor, "write_file", {"path": "n.py", "content": "a = 1\nb = 2\nc = 3"})
        result = _call(executor, "read_file", {"path": "n.py"})
        assert result["content"] == "   1| a = 1\n   2| b = 2\n   3| c = 3"

    def test_read_file_numbers_lines_from_the_offset(self, executor):
        """offset=10 starts at line 11: the file's own numbers, not the slice's."""
        content = "\n".join(f"line {i}" for i in range(100))
        _call(executor, "write_file", {"path": "off.py", "content": content})
        result = _call(executor, "read_file", {"path": "off.py", "offset": 10, "limit": 3})
        assert result["content"].split("\n") == [
            "  11| line 10", "  12| line 11", "  13| line 12",
        ]

    def test_truncation_marker_is_not_numbered_like_a_file_line(self, executor):
        """The marker is appended after numbering, so it cannot be mistaken for
        (or quoted back as) a line of the file."""
        content = "\n".join("x" * 200 for _ in range(1000))
        _call(executor, "write_file", {"path": "huge.py", "content": content})
        result = _call(executor, "read_file", {"path": "huge.py"})
        marker = result["content"].rsplit("\n", 1)[-1]
        assert marker.startswith("... [truncated"), marker

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
# Path safety — Windows extended-path prefix regression
# ======================================================================

class TestPathNormalization:
    """Pin the fix for the Windows ``\\\\?\\`` extended-path prefix bug.

    On Windows with long-path support enabled, ``os.path.realpath`` can
    return paths with the extended-length ``\\\\?\\`` prefix for
    resolved child paths but NOT for the workspace root (because the
    root exists at init time and the child doesn't). A naive
    ``startswith`` check then fails for legitimate subdirectory writes
    like ``config/puma.rb`` or ``app/models/author.rb``, which is how
    a Rails run ended up corrupted with "Path traversal blocked"
    errors for every subdirectory file.
    """

    def test_strips_windows_extended_path_prefix(self):
        from besser.spec_driven_agent.tool_executor import _normalize_path_for_comparison

        # ``\\?\C:\...`` → ``C:\...``
        assert (
            _normalize_path_for_comparison("\\\\?\\C:\\Users\\test\\file.txt")
            == "C:\\Users\\test\\file.txt"
        )

    def test_strips_unc_extended_path_prefix(self):
        from besser.spec_driven_agent.tool_executor import _normalize_path_for_comparison

        # ``\\?\UNC\server\share\...`` → ``\\server\share\...``
        assert (
            _normalize_path_for_comparison("\\\\?\\UNC\\server\\share\\file.txt")
            == "\\\\server\\share\\file.txt"
        )

    def test_passes_regular_paths_through(self):
        from besser.spec_driven_agent.tool_executor import _normalize_path_for_comparison

        assert _normalize_path_for_comparison("/home/test/file.txt") == "/home/test/file.txt"
        assert (
            _normalize_path_for_comparison("C:\\Users\\test\\file.txt")
            == "C:\\Users\\test\\file.txt"
        )

    def test_subdirectory_write_works_after_fix(self, executor, tmp_path):
        """Regression: writing ``config/puma.rb`` (subdir of workspace)
        must succeed, not raise "Path traversal blocked". This is the
        exact pattern that was broken in the Rails run."""
        _call(executor, "write_file", {
            "path": "config/puma.rb",
            "content": "# puma config",
        })
        result = _call(executor, "read_file", {"path": "config/puma.rb"})
        assert result["content"] == "   1| # puma config"
        assert (tmp_path / "config" / "puma.rb").exists()

    def test_deeply_nested_write_works(self, executor, tmp_path):
        """Another variant: deeply nested paths like
        ``db/migrate/20240101_create_books.rb`` must work."""
        _call(executor, "write_file", {
            "path": "db/migrate/20240101_create_books.rb",
            "content": "class CreateBooks; end",
        })
        assert (tmp_path / "db" / "migrate" / "20240101_create_books.rb").exists()

    def test_actual_traversal_still_blocked(self, executor):
        """The fix must NOT weaken security: paths escaping the
        workspace still raise the traversal error."""
        result = _call(executor, "write_file", {
            "path": "../escape.txt",
            "content": "nope",
        })
        assert "error" in result
        assert "traversal" in result["error"].lower() or "Path" in result["error"]

    def test_absolute_path_outside_workspace_blocked(self, executor):
        """An absolute path to somewhere else on disk must also be blocked."""
        import tempfile
        outside = tempfile.gettempdir() + os.sep + "not_the_workspace.txt"
        result = _call(executor, "write_file", {
            "path": outside,
            "content": "nope",
        })
        assert "error" in result

    def test_sibling_workspace_prefix_not_matched(self, tmp_path, simple_model):
        """Regression for the classic startswith trap: a workspace at
        ``/tmp/besser_llm_abc`` must NOT allow writes to a sibling
        ``/tmp/besser_llm_abc_DIFFERENT/evil.txt`` — that path starts
        with the workspace string but isn't actually inside it.
        """
        ws = tmp_path / "besser_llm_abc"
        ws.mkdir()
        sibling_evil = tmp_path / "besser_llm_abc_DIFFERENT" / "evil.txt"
        executor_local = ToolExecutor(workspace=str(ws), domain_model=simple_model)
        result = _call(executor_local, "write_file", {
            "path": str(sibling_evil),
            "content": "nope",
        })
        assert "error" in result


# ======================================================================
# Execution tools
# ======================================================================

class TestExecutionTools:

    def test_run_command_success(self, shell_executor):
        result = _call(shell_executor, "run_command", {"command": "python -c \"print('hello')\" "})
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_run_command_failure(self, shell_executor):
        result = _call(shell_executor, "run_command", {"command": "python -c \"raise ValueError('boom')\" "})
        assert result["success"] is False
        assert "boom" in result["stderr"]

    def test_run_command_in_subdir(self, shell_executor):
        _call(shell_executor, "write_file", {"path": "subdir/test.py", "content": "print('sub')"})
        result = _call(shell_executor, "run_command", {"command": "python test.py", "working_dir": "subdir"})
        assert result["success"] is True
        assert "sub" in result["stdout"]

    def test_run_command_timeout(self, shell_executor):
        """Commands that exceed timeout return an error."""
        import besser.spec_driven_agent.tool_executor as mod
        old_timeout = mod.COMMAND_TIMEOUT
        mod.COMMAND_TIMEOUT = 1  # 1 second
        try:
            result = _call(shell_executor, "run_command", {"command": "python -c \"import time; time.sleep(10)\" "})
            assert "error" in result and "timed out" in result["error"]
        finally:
            mod.COMMAND_TIMEOUT = old_timeout

    def test_install_dependencies_no_files(self, shell_executor):
        result = _call(shell_executor, "install_dependencies", {})
        assert "error" in result

    def test_install_dependencies_custom_command(self, shell_executor):
        result = _call(shell_executor, "install_dependencies", {"command": "python --version"})
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

        from besser.spec_driven_agent.llm_client import UsageTracker

        class MockClient:
            model = "mock-model"
            usage = UsageTracker("mock-model")

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

        from besser.spec_driven_agent.orchestrator import LLMOrchestrator

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
        assert recipe["model"]["name"] == "TestModel"
        # Phase 1 adds generate_fastapi_backend (turn 0) + mock LLM adds 4 tool calls
        assert recipe["turns"] >= 4
        assert len(recipe["tool_calls"]) >= 4


# ======================================================================
# Model query tools
# ======================================================================


@pytest.fixture
def library_executor(tmp_path):
    """Executor with a richer model for query tests.

    Two classes (Book extends Item, abstract Item), one constraint, one
    method, a couple of attributes — enough to exercise every predicate
    branch in ``list_classes_with``.
    """
    from besser.BUML.metamodel.structural import (
        Constraint,
        Generalization,
        Method,
    )

    String = PrimitiveDataType("str")
    Int = PrimitiveDataType("int")

    item = Class(name="Item", is_abstract=True)
    item.attributes = {
        Property(name="id", type=Int, is_id=True),
        Property(name="title", type=String),
    }
    item.methods = {Method(name="describe", type=String)}

    book = Class(name="Book")
    book.attributes = {Property(name="isbn", type=String)}

    Generalization(general=item, specific=book)

    constraint = Constraint(
        name="non_empty_title",
        context=item,
        expression="self.title.size() > 0",
        language="OCL",
    )

    model = DomainModel(
        name="Library",
        types={item, book},
        constraints={constraint},
    )
    return ToolExecutor(workspace=str(tmp_path), domain_model=model)


class TestModelQueryTools:

    def test_query_class_returns_full_definition(self, library_executor):
        result = _call(library_executor, "query_class", {"name": "Item"})
        assert result["name"] == "Item"
        assert result["is_abstract"] is True
        attr_names = {a["name"] for a in result["attributes"]}
        assert {"id", "title"} <= attr_names
        method_names = {m["name"] for m in result["methods"]}
        assert "describe" in method_names

    def test_query_class_unknown_returns_error(self, library_executor):
        result = _call(library_executor, "query_class", {"name": "DoesNotExist"})
        assert "error" in result
        assert "DoesNotExist" in result["error"]

    def test_list_classes_is_abstract(self, library_executor):
        result = _call(library_executor, "list_classes_with", {"predicate": "is_abstract"})
        assert result["matches"] == ["Item"]
        assert result["count"] == 1

    def test_list_classes_is_root(self, library_executor):
        """``is_root`` = classes with no parents."""
        result = _call(library_executor, "list_classes_with", {"predicate": "is_root"})
        # Item is abstract root. Book extends Item, so it is not root.
        assert "Item" in result["matches"]
        assert "Book" not in result["matches"]

    def test_list_classes_has_constraint(self, library_executor):
        result = _call(
            library_executor, "list_classes_with", {"predicate": "has_constraint"}
        )
        assert result["matches"] == ["Item"]

    def test_list_classes_has_attribute(self, library_executor):
        result = _call(
            library_executor, "list_classes_with", {"predicate": "has_attribute:isbn"}
        )
        assert result["matches"] == ["Book"]

    def test_list_classes_extends(self, library_executor):
        result = _call(
            library_executor, "list_classes_with", {"predicate": "extends:Item"}
        )
        assert result["matches"] == ["Book"]

    def test_list_classes_unknown_predicate(self, library_executor):
        result = _call(
            library_executor, "list_classes_with", {"predicate": "quantum_entangled"}
        )
        assert "error" in result
        assert "Unknown predicate" in result["error"]

    def test_list_classes_predicate_requires_value(self, library_executor):
        result = _call(
            library_executor, "list_classes_with", {"predicate": "has_attribute"}
        )
        assert "error" in result
        assert "requires a value" in result["error"]

    def test_get_constraints_for_class(self, library_executor):
        result = _call(
            library_executor, "get_constraints_for", {"class_name": "Item"}
        )
        assert result["class"] == "Item"
        assert result["count"] == 1
        assert result["constraints"][0]["name"] == "non_empty_title"
        assert "title.size" in result["constraints"][0]["expression"]

    def test_get_constraints_for_class_without_any(self, library_executor):
        result = _call(
            library_executor, "get_constraints_for", {"class_name": "Book"}
        )
        assert result["class"] == "Book"
        assert result["count"] == 0
        assert result["constraints"] == []

    def test_get_constraints_for_unknown_class(self, library_executor):
        result = _call(
            library_executor, "get_constraints_for", {"class_name": "Nope"}
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# The Windows extended-path prefix must not escape _safe_path / _safe_cwd.
#
# self.workspace is stored prefix-free (__init__ normalises it), but both
# resolvers used to return the RAW os.path.realpath. On Windows realpath keeps
# the "\?\" extended prefix when the underlying _getfinalpathname fails
# transiently -- which is what "[WinError 1450] Insufficient system resources"
# is. Every downstream os.path.relpath(path, self.workspace) then raised
#
#     ValueError: path is on mount '\?\C:', start on mount 'C:'
#
# including the one in execute_typed that EVERY file tool passes through. Seen
# 12 times across 9 runs, always on write_file creating the first file in a new
# directory (frontend/index.html, frontend/src/api.js, ...), i.e. when the
# parent does not exist yet and the resolve is most likely to fail. Needing the
# transient is why nobody could reproduce it on demand; these tests inject it.
# ---------------------------------------------------------------------------
_EXT_PREFIX = "\\\\?\\"  # the literal four characters \ \ ? \


class TestExtendedPathPrefixNeverEscapes:
    """Each test fails against the pre-fix `return full` / `return cwd`."""

    @staticmethod
    def _flaky_realpath(monkeypatch, workspace):
        """realpath() as Windows behaves when the resolve fails transiently."""
        import ntpath

        real = os.path.realpath

        def _realpath(path, *args, **kwargs):
            resolved = real(path, *args, **kwargs)
            # The workspace root resolves cleanly (it exists); the child being
            # created does not, and that is the one that comes back extended.
            if resolved != workspace and ntpath.normcase(resolved).startswith(
                ntpath.normcase(workspace),
            ):
                return _EXT_PREFIX + resolved
            return resolved

        monkeypatch.setattr(os.path, "realpath", _realpath)

    def test_safe_path_returns_a_prefix_free_path(self, tmp_path, monkeypatch):
        from besser.spec_driven_agent.tool_executor import ToolExecutor

        ex = ToolExecutor(workspace=str(tmp_path))
        self._flaky_realpath(monkeypatch, ex.workspace)

        resolved = ex._safe_path("frontend/index.html")
        assert not resolved.startswith(_EXT_PREFIX), resolved
        # The whole point: this is what blew up on every file tool.
        assert os.path.relpath(resolved, ex.workspace).replace("\\", "/") == (
            "frontend/index.html"
        )

    def test_safe_cwd_returns_a_prefix_free_path(self, tmp_path, monkeypatch):
        from besser.spec_driven_agent.tool_executor import ToolExecutor

        ex = ToolExecutor(workspace=str(tmp_path))
        self._flaky_realpath(monkeypatch, ex.workspace)

        resolved = ex._safe_cwd("backend")
        assert not resolved.startswith(_EXT_PREFIX), resolved
        assert os.path.isdir(resolved)

    def test_write_file_survives_the_first_file_in_a_new_directory(self, tmp_path, monkeypatch):
        """The exact live shape: write_file creating a file whose parent does
        not exist yet, with the resolve failing transiently."""
        from besser.spec_driven_agent.tool_executor import ToolExecutor

        ex = ToolExecutor(workspace=str(tmp_path))
        self._flaky_realpath(monkeypatch, ex.workspace)

        result = _call(ex, "write_file", {
            "path": "frontend/src/pages/StudentList.jsx",
            "content": "export default function StudentList() { return null; }\n",
        })
        assert "error" not in result, result
        assert (tmp_path / "frontend" / "src" / "pages" / "StudentList.jsx").is_file()

    def test_traversal_is_still_blocked_when_the_prefix_is_present(self, tmp_path, monkeypatch):
        """Normalising the return value must not soften the containment check."""
        from besser.spec_driven_agent.tool_executor import ToolExecutor

        ex = ToolExecutor(workspace=str(tmp_path))
        self._flaky_realpath(monkeypatch, ex.workspace)

        with pytest.raises(ValueError):
            ex._safe_path("../../../etc/passwd")
        with pytest.raises(ValueError):
            ex._safe_cwd("../../../")
