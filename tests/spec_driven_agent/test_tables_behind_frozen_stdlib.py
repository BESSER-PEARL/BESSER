"""A handler calling a frozen stdlib function (``os.getenv``) still has readable tables.

Live run r10-p81: its register route reads ``os.getenv`` on the way to creating the
AppUser. ``os`` is a frozen module, so ``getenv``'s file is ``<frozen os>``; made
absolute against the app copy it looked like app code, ``inspect.getsource``
failed and the whole handler became unknowable. The probe then could not tell
what register creates.
"""
from __future__ import annotations

import pytest

from tests.spec_driven_agent.test_constructibility_create_behind_other_table import _DIRECT, _note_app

from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_a_handler_reading_os_getenv_is_probed_as_its_table(tmp_path):
    body = "import os\n    os.getenv('NOTE_PREFIX')\n    " + _DIRECT.format(extra="")
    report = collect_constructibility_report(_note_app(tmp_path, body))

    backend = report["backends"][0]
    assert backend["skipped_non_entity"] == [], backend
    assert backend["entities"]["NoteRow"]["verdict"] == "created"


def test_a_handler_calling_a_function_with_no_source_file_is_probed_as_its_table(tmp_path):
    """The same for a ``<string>`` code object (``eval``, generated wrappers): no file to read."""
    root = _note_app(tmp_path, "_helper(1)\n    " + _DIRECT.format(extra=""))
    main = tmp_path / "web_app" / "backend" / "main_api.py"
    main.write_text("_helper = eval('lambda x: x')\n" + main.read_text(encoding="utf-8"), encoding="utf-8")

    backend = collect_constructibility_report(root)["backends"][0]

    assert backend["skipped_non_entity"] == [], backend
    assert backend["entities"]["NoteRow"]["verdict"] == "created"
