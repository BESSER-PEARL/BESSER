"""``search_in_files`` spent its 50-match budget on a lock file.

Pattern "react" with a package-lock.json present: 50 matches, all in the lock
file, 9,494 chars, no source hit. And a ``file_glob`` with a '/' returned an
empty result that read as "pattern absent".
"""

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


def _put(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def executor(tmp_path):
    lock = "{\n" + "".join(f'  "node_modules/react-{i}": {{"version": "18.0.{i}"}},\n' for i in range(400)) + "}\n"
    _put(tmp_path, "frontend/package-lock.json", lock)
    _put(tmp_path, "frontend/pnpm-lock.yaml", "react: 18\n")
    _put(tmp_path, "frontend/public/app.min.js", "var react=1;\n")
    _put(tmp_path, "frontend/src/App.tsx.map", '{"sources": ["react"]}\n')
    _put(tmp_path, "frontend/src/App.tsx", "import React from 'react';\n")
    return ToolExecutor(workspace=str(tmp_path))


def test_lock_files_bundles_and_maps_do_not_take_the_match_budget(executor):
    result = executor._search_in_files({"pattern": "react"})

    assert [m["file"] for m in result["matches"]] == ["frontend/src/App.tsx"], result
    assert "lock files" in result["not_searched"]


def test_a_glob_naming_a_lock_file_still_searches_it(executor):
    result = executor._search_in_files({"pattern": "react-399", "file_glob": "package-lock.json"})

    assert [m["file"] for m in result["matches"]] == ["frontend/package-lock.json"], result


def test_a_clean_search_reports_nothing_skipped(executor):
    result = executor._search_in_files({"pattern": "React", "file_glob": "*.tsx"})

    assert len(result["matches"]) == 1 and "not_searched" not in result


@pytest.mark.parametrize("glob", ["src/*.tsx", "**/*.ts", "src\\*.tsx"])
def test_a_glob_with_a_path_is_an_error_not_an_empty_result(executor, glob):
    result = executor._search_in_files({"pattern": "react", "file_glob": glob})

    assert "matches" not in result
    assert "NAMES only" in result["error"]
