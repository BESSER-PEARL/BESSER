"""``list_files`` must not list installed dependencies.

Live hotel run (2026-09-23): Phase 1 installed the scaffold frontend's npm
packages (10,304 files under ``web_app/frontend/node_modules`` next to 134
project files). The model's first ``list_files`` returned all of them, the
prompt jumped from 41k to 229k tokens, turn 2 exceeded the model's 262k
context (HTTP 400) and Phase 2 ended as ``api_error`` before any feature code.
"""

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


def _touch(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


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

    listed = {f["path"] for f in ToolExecutor(workspace=str(tmp_path))._list_files({})["files"]}

    assert listed == {
        "web_app/backend/main_api.py",
        "web_app/frontend/package.json",
        "web_app/frontend/src/App.tsx",
    }
