"""Deleting a file is decided by what the delete does, for every model.

The executor refused to delete a generated file only when the model id looked
free or self-hosted (``is_free_local_model``, a pricing helper), so the same
request was refused on ``name:tag`` ids and allowed on ``vendor/name`` ids. The
only delete in the recorded runs was legitimate (a planned task removing a
generated page after its import and route were taken out), and the guard
would have refused it on a "free" id. A delete now always runs, and its result
names the files that still import what was deleted.
"""
from __future__ import annotations

import inspect
import json
import os

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import FILE_TOOLS


def _workspace(tmp_path, files: dict[str, str]) -> ToolExecutor:
    for rel, text in files.items():
        full = os.path.join(str(tmp_path), rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(text)
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.update_generator_files(list(files), [])
    return executor


def _delete(executor, rel) -> dict:
    return json.loads(executor.execute("delete_file", {"path": rel}))


BACKEND = {
    "backend/main_api.py": "from fastapi import FastAPI\nfrom sql_alchemy import *\n\napp = FastAPI()\n",
    "backend/sql_alchemy.py": "Base = object\n",
    "backend/notes.py": "NOTE = 1\n",
}
FRONTEND = {
    "frontend/package.json": json.dumps({"dependencies": {"react": "18.3.1"}}),
    "frontend/src/App.tsx": ("import React from 'react';\nimport OrderList from './pages/OrderList';\n\n"
                             "export default function App() { return <OrderList />; }\n"),
    "frontend/src/pages/OrderList.tsx": ("import React from 'react';\n\n"
                                         "export default function OrderList() { return <div />; }\n"),
}


def test_the_delete_policy_is_not_a_model_setting():
    assert "protect_scaffold" not in inspect.signature(ToolExecutor).parameters


def test_a_generated_module_still_imported_is_deleted_and_its_importers_named(tmp_path):
    executor = _workspace(tmp_path, BACKEND)

    result = _delete(executor, "backend/sql_alchemy.py")

    assert result["status"] == "deleted", result
    assert not os.path.exists(tmp_path / "backend" / "sql_alchemy.py")
    assert any("backend/main_api.py" in line for line in result["still_imported"]), result


def test_a_generated_page_still_imported_is_named_too(tmp_path):
    executor = _workspace(tmp_path, FRONTEND)

    result = _delete(executor, "frontend/src/pages/OrderList.tsx")

    assert result["status"] == "deleted", result
    assert any("frontend/src/App.tsx" in line for line in result["still_imported"]), result


def test_a_page_removed_after_its_import_is_deleted_without_a_note(tmp_path):
    executor = _workspace(tmp_path, {**FRONTEND, "frontend/src/App.tsx": (
        "import React from 'react';\n\nexport default function App() { return <div />; }\n")})

    result = _delete(executor, "frontend/src/pages/OrderList.tsx")

    assert result == {"status": "deleted", "path": "frontend/src/pages/OrderList.tsx"}


def test_a_breakage_that_was_there_before_the_delete_is_not_blamed_on_it(tmp_path):
    executor = _workspace(tmp_path, {**BACKEND, "backend/extra.py": "import missing_module\n"})

    result = _delete(executor, "backend/notes.py")

    assert "still_imported" not in result, result


def test_delete_file_description_states_the_contract():
    description = next(t["description"] for t in FILE_TOOLS if t["name"] == "delete_file").lower()
    assert "protected run" not in description
    assert "generator left behind" not in description
    assert "still_imported" in description and "top-level module" in description
