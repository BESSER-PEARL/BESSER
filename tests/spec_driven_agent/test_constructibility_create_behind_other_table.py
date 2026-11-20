"""A ``<X>Create`` POST whose table is not named ``<X>`` is still probed.

Review probe on PR 788: ORM class ``NoteRow`` behind ``POST /note/`` with body
``NoteCreate`` went to ``skipped_non_entity``, which only the stats read, so a create
that crashed gave no finding at all; before the skip it was ``create unverified``.
Only a body with no table behind its handler (a newsletter signup) is skipped.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.constructibility import (
    PREFIX, UNVERIFIED_PREFIX, collect_constructibility_report,
)

from tests.spec_driven_agent.test_constructibility_probe import _scaffold

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

_NOTE_TABLE = '''
class NoteRow(Base):
    __tablename__ = "note"
    id: Mapped_[int] = mapped_column(primary_key=True)
    text: Mapped_[str] = mapped_column(String_(100))

'''

_NOTE_ROUTE = '''

from pydantic import BaseModel as _NoteBase
import note_service


class NoteCreate(_NoteBase):
    text: str


@app.post("/note/")
def create_note(payload: NoteCreate, database: Session = Depends(get_db)):
    {body}
'''

_DIRECT = "row = NoteRow(text=payload.text{extra})\n    " \
          "database.add(row)\n    database.commit()\n    return {{'id': row.id, 'text': row.text}}"
_VIA_SERVICE = "return note_service.add_note(database, payload.text)"

_SERVICE = '''
from sql_alchemy import NoteRow


def add_note(database, text):
    row = NoteRow(text=text, author=1)  # NoteRow has no ``author``: TypeError
    database.add(row)
    database.commit()
    return {"id": row.id, "text": row.text}
'''


def _note_app(tmp_path, body: str) -> str:
    workspace = _scaffold(tmp_path)
    backend = tmp_path / "web_app" / "backend"
    orm = backend / "sql_alchemy.py"
    text = orm.read_text(encoding="utf-8")
    orm.write_text(text.replace("# Database connection", _NOTE_TABLE + "# Database connection", 1), encoding="utf-8")
    (backend / "note_service.py").write_text(_SERVICE, encoding="utf-8")
    main = backend / "main_api.py"
    main.write_text(main.read_text(encoding="utf-8") + _NOTE_ROUTE.format(body=body), encoding="utf-8")
    return workspace


def test_a_crashing_create_into_a_differently_named_table_is_reported(tmp_path):
    report = collect_constructibility_report(_note_app(tmp_path, _DIRECT.format(extra=", author=1")))
    [finding] = [issue for issue in report["issues"] if "POST /note/" in issue]
    assert finding.startswith(f"{PREFIX} web_app/backend: POST /note/ - observed a server/persistence failure"), finding
    assert "creating NoteRow" in finding and "Fix site: create_note in web_app/backend/main_api.py" in finding
    assert report["backends"][0]["skipped_non_entity"] == []


def test_a_crash_inside_a_service_the_handler_calls_is_reported(tmp_path):
    report = collect_constructibility_report(_note_app(tmp_path, _VIA_SERVICE))
    assert [issue for issue in report["issues"] if issue.startswith(PREFIX) and "POST /note/" in issue], report


def test_a_create_writing_two_unnamed_tables_is_reported_unverified(tmp_path):
    workspace = _note_app(tmp_path, _DIRECT.format(extra="") + "\n\n\ndef _audit(database):\n"
                                                                 "    database.add(NoteLogRow(text='created'))")
    orm = tmp_path / "web_app" / "backend" / "sql_alchemy.py"
    log_table = _NOTE_TABLE.replace("NoteRow", "NoteLogRow").replace('"note"', '"note_log"')
    orm.write_text(orm.read_text(encoding="utf-8").replace("# Database connection", log_table + "# Database connection",
                                                           1), encoding="utf-8")
    main = tmp_path / "web_app" / "backend" / "main_api.py"
    main.write_text(main.read_text(encoding="utf-8").replace(
        "    database.add(row)", "    database.add(row)\n    _audit(database)", 1), encoding="utf-8")

    report = collect_constructibility_report(workspace)

    [finding] = [issue for issue in report["issues"] if "POST /note/" in issue]
    assert finding.startswith(f"{UNVERIFIED_PREFIX} web_app/backend: POST /note/ - its body NoteCreate"), finding
    assert "NoteLogRow, NoteRow" in finding and "GET that record by its id" in finding


def test_a_working_create_into_a_differently_named_table_is_verified(tmp_path):
    report = collect_constructibility_report(_note_app(tmp_path, _DIRECT.format(extra="")))
    assert report["issues"] == [], report["issues"]
    assert report["backends"][0]["entities"]["NoteRow"]["verdict"] == "created"


_LOGIN = """
    user = database.query(NoteRow).filter(NoteRow.text == payload.text).first()
    if user is None:
        raise HTTPException(status_code=401, detail="bad credentials")
    return {'token': 't-' + str(user.id)}"""


def test_a_login_body_that_only_reads_a_table_is_skipped(tmp_path):
    """Review probe, round 6: ``POST /sessions/`` (body ``SessionCreate``) only reads its user
    table, was probed as that table and reported ``create unverified`` - a blocker no edit clears."""
    workspace = _note_app(tmp_path, _LOGIN.strip())
    main = tmp_path / "web_app" / "backend" / "main_api.py"
    main.write_text(main.read_text(encoding="utf-8").replace("NoteCreate", "SessionCreate")
                    .replace('"/note/"', '"/sessions/"'), encoding="utf-8")

    report = collect_constructibility_report(workspace)

    assert report["issues"] == [], report["issues"]
    assert report["backends"][0]["skipped_non_entity"] == ["Session"]


_RAISING_MODULE = """
def __getattr__(name):
    raise KeyError(f"setting {name} is not configured")
"""


def test_a_handler_lookup_that_raises_leaves_the_rest_of_the_probe_running(tmp_path):
    """Review probe, round 6: a module whose ``__getattr__`` raises KeyError took the whole
    child down, so every entity of the app came back ``probe_error``."""
    workspace = _note_app(tmp_path, "note_service.clip(payload.text)\n    " + _DIRECT.format(extra=""))
    (tmp_path / "web_app" / "backend" / "note_service.py").write_text(_RAISING_MODULE, encoding="utf-8")

    report = collect_constructibility_report(workspace)

    [backend] = report["backends"]
    assert backend["boot"] == "ok", backend
    assert backend["skipped_non_entity"] == ["Note"]
    assert {"Room", "Booking"} <= set(backend["entities"])
