"""``list_files`` must not list installed dependencies.

A recorded hotel run: Phase 1 installed the scaffold frontend's npm
packages (10,304 files under ``web_app/frontend/node_modules`` next to 134
project files). The model's first ``list_files`` returned all of them, the
prompt jumped from 41k to 229k tokens, turn 2 exceeded the model's 262k
context (HTTP 400) and Phase 2 ended as ``api_error`` before any feature code.
"""

from besser.spec_driven_agent.agent.tool_executor import _LIST_FILES_MAX, ToolExecutor


def _touch(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _paths(result):
    return {line.rsplit(" ", 1)[0] for line in result["files"].splitlines() if not line.startswith("...")}


def test_dependency_and_cache_dirs_are_not_listed(tmp_path):
    for rel in (
        "web_app/backend/main_api.py",
        "web_app/frontend/package.json",
        "web_app/frontend/src/App.tsx",
    ):
        _touch(tmp_path, rel)
    for rel in (
        "web_app/frontend/node_modules/react/index.js",
        "web_app/frontend/node_modules/.bin/vite",
        "web_app/backend/.venv/lib/site.py",
        "web_app/backend/__pycache__/main_api.cpython-311.pyc",
        "web_app/frontend/dist/index.html",
        ".git/HEAD",
    ):
        _touch(tmp_path, rel)

    listed = _paths(ToolExecutor(workspace=str(tmp_path))._list_files({}))

    assert listed == {
        "web_app/backend/main_api.py",
        "web_app/frontend/package.json",
        "web_app/frontend/src/App.tsx",
    }


def test_a_virtualenv_under_any_name_is_not_listed_but_named(tmp_path):
    """A recorded run's venv was ``env/``: 3,000 files, 232k chars in one result."""
    _touch(tmp_path, "backend/main.py")
    _touch(tmp_path, "env/pyvenv.cfg")
    for i in range(50):
        _touch(tmp_path, f"env/lib/python3.12/site-packages/pkg/mod_{i}.py")

    result = ToolExecutor(workspace=str(tmp_path))._list_files({})

    assert result["files"] == "backend/main.py 1"
    assert result["total"] == 1
    assert "env/" in result["not_listed"]


def test_a_large_listing_is_capped_source_first_with_counts_for_the_rest(tmp_path):
    for i in range(10):
        _touch(tmp_path, f"app/src/view_{i}.tsx")
    for i in range(_LIST_FILES_MAX + 40):
        _touch(tmp_path, f"app/assets/img_{i:04}.png")

    result = ToolExecutor(workspace=str(tmp_path))._list_files({})

    lines = result["files"].splitlines()
    assert len(lines) == _LIST_FILES_MAX + 1
    assert {f"app/src/view_{i}.tsx" for i in range(10)} <= _paths(result)
    assert result["truncated"] is True and result["total"] == _LIST_FILES_MAX + 50
    assert lines[-1].startswith("... 50 more files not listed, by directory: app/assets/ (50)")
