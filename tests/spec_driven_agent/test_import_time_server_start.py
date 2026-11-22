"""A backend entry module that starts its server while being imported.

``uvicorn.run(app)`` outside the main guard never returns, so the boot probe
times out on the import. A timeout is an unknown (the host may just be slow)
and is not sent to repair, which left this common, one-line defect unrepaired.
The static check names the line, so the fix loop can act on it.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.python_imports import _import_time_server_starts

_APP = "from fastapi import FastAPI\nimport uvicorn\n\napp = FastAPI()\n\n"


def _backend(tmp_path, source: str, name: str = "main_api.py") -> str:
    folder = tmp_path / "backend"
    folder.mkdir()
    (folder / name).write_text(source, encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("tail", [
    "uvicorn.run(app, host='0.0.0.0', port=8000)\n",
    "if True:\n    uvicorn.run(app)\n",
    "try:\n    uvicorn.run(app)\nexcept KeyboardInterrupt:\n    pass\n",
    "if __name__ == '__main__':\n    pass\nelse:\n    uvicorn.run(app)\n",
])
def test_a_server_started_at_import_time_is_a_startup_blocker(tmp_path, tail):
    issues = _import_time_server_starts(_backend(tmp_path, _APP + tail))

    assert len(issues) == 1
    assert issues[0].startswith("application startup: backend/main_api.py line ")
    assert _classify_issue(issues[0]).severity == "blocker"


@pytest.mark.parametrize("tail", [
    "if __name__ == '__main__':\n    uvicorn.run(app)\n",
    "if '__main__' == __name__:\n    uvicorn.run(app)\n",
    "def serve():\n    uvicorn.run(app)\n",
    "class Runner:\n    def go(self):\n        uvicorn.run(app)\n",
    "config = uvicorn.Config(app)\n",
])
def test_a_guarded_or_deferred_start_is_not_reported(tmp_path, tail):
    assert _import_time_server_starts(_backend(tmp_path, _APP + tail)) == []


def test_a_launcher_script_may_start_the_server_at_its_top_level(tmp_path):
    """``python run.py`` executes the file; nothing imports it."""
    source = "import uvicorn\nuvicorn.run('main_api:app')\n"
    assert _import_time_server_starts(_backend(tmp_path, source, name="run.py")) == []


def test_an_unparseable_entry_module_is_left_to_the_syntax_check(tmp_path):
    assert _import_time_server_starts(_backend(tmp_path, "def broken(:\n")) == []
