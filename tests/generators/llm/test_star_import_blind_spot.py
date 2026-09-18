"""An undefined name behind ``from x import *`` must be reported at write time.

Live run 0c537a4e (2026-09-18, Nebius Qwen/Qwen3-30B-A3B-Instruct-2507): five
modify_file edits to routers/booking_methods.py each landed a body reading
``booking_id`` in an endpoint that has no such parameter, and one called
``computeAmountOwed(db_booking.id, database)``, a helper defined nowhere. All
five results came back ``status: modified`` with no ``diagnostics`` key, so
the model learned of the missing helper only later and spent turns 21-50
(13 identical modify_file misses, 16 re-reads) trying to edit its
definition into existence.

pyflakes is why nothing fired: once a module scope carries a star import,
every unresolved load is reported as ``ImportStarUsage`` ("may be undefined,
or defined from star imports") instead of ``UndefinedName``
(pyflakes/checker.py, ``handleNodeLoad``), and write_diagnostics collected
only the latter. Every scaffold router opens with ``from pydantic_classes
import *`` / ``from sql_alchemy import *`` / ``from bal_stdlib import *``, so
every router was blind. Those modules are ours: what they export can be read
off their AST, and a star-usage name none of them exports is undefined.

The fixture is that run's backend, verbatim.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from besser.generators.llm.write_diagnostics import diagnose_written_content

pytest.importorskip("pyflakes")

FIXTURE = Path(__file__).parent / "fixtures" / "run_0c537a4e"
ROUTER = "web_app/backend/routers/booking_methods.py"


def _names(findings):
    return {f["message"].split("'")[1] for f in findings if f["code"] == "UndefinedName"}


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return tmp_path


# -- the live file -------------------------------------------------------
def test_run_0c537a4e_router_reports_the_two_real_undefined_names(workspace):
    content = (workspace / ROUTER).read_text(encoding="utf-8")
    findings = diagnose_written_content(ROUTER, content, workspace=str(workspace))
    assert _names(findings) == {"booking_id", "computeAmountOwed"}, findings
    assert {f["line"] for f in findings if "booking_id" in f["message"]} == {37, 86, 134, 184, 239}
    assert [f["line"] for f in findings if "computeAmountOwed" in f["message"]] == [193]


def test_names_the_star_modules_that_were_searched(workspace):
    content = (workspace / ROUTER).read_text(encoding="utf-8")
    finding = next(f for f in diagnose_written_content(ROUTER, content, workspace=str(workspace))
                   if "computeAmountOwed" in f["message"])
    for module in ("sql_alchemy", "pydantic_classes", "bal_stdlib"):
        assert module in finding["message"]


def test_the_executor_reports_it_in_the_same_turn(workspace):
    """The result the model reads back must carry the finding."""
    from besser.generators.llm.tool_executor import ToolExecutor

    executor = ToolExecutor(workspace=str(workspace))
    content = (workspace / ROUTER).read_text(encoding="utf-8")
    executor.execute("read_file", {"path": ROUTER})
    result = json.loads(executor.execute("write_file", {"path": ROUTER, "content": content}))
    assert result["status"] == "written"
    assert "booking_id" in json.dumps(result.get("diagnostics", [])), result


# -- calibration: names the star modules DO export stay silent -------------
def test_star_exported_names_are_not_flagged(workspace):
    """``Booking``, ``Bill`` and ``dt_date`` come from sql_alchemy; they are
    the bulk of pyflakes' star-usage noise on this file and must be absent."""
    content = (workspace / ROUTER).read_text(encoding="utf-8")
    names = _names(diagnose_written_content(ROUTER, content, workspace=str(workspace)))
    assert not names & {"Booking", "Bill", "dt_date", "HTTPException", "Session"}


def test_a_healthy_generated_scaffold_yields_nothing(tmp_path):
    """Known-good output first: a fresh BackendGenerator scaffold, every file."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DateType, DomainModel, FloatType, IntegerType,
        Method, Multiplicity, Property,
    )
    from besser.generators.backend import BackendGenerator

    room = Class(name="Room", attributes={Property(name="roomNumber", type=IntegerType)})
    booking = Class(
        name="Booking",
        attributes={Property(name="arrivalDate", type=DateType)},
        methods={Method(name="computeAmountOwed", type=FloatType),
                 Method(name="cancel")},
    )
    link = BinaryAssociation(name="room_booking", ends={
        Property(name="room", type=room, multiplicity=Multiplicity(1, 1)),
        Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*")),
    })
    model = DomainModel(name="Hotel", types={room, booking}, associations={link})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=model, output_dir=str(backend)).generate()

    noise = []
    for root, _dirs, files in os.walk(backend):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, tmp_path).replace(os.sep, "/")
            with open(path, encoding="utf-8") as handle:
                noise += diagnose_written_content(rel, handle.read(), workspace=str(tmp_path))
    assert (backend / "routers" / "booking_methods.py").is_file()
    assert noise == []


# -- resolution rules ------------------------------------------------------
def test_a_module_we_did_not_generate_is_not_judged(tmp_path):
    """``from fastapi import *`` cannot be read off disk: stay silent rather
    than condemn every FastAPI name."""
    source = "from fastapi import *\nrouter = APIRouter()\n"
    (tmp_path / "r.py").write_text(source, encoding="utf-8")
    assert diagnose_written_content("r.py", source, workspace=str(tmp_path)) == []


def test_the_service_folder_is_the_import_root(tmp_path):
    """routers/x.py resolves ``sql_alchemy`` from backend/, its service's cwd."""
    (tmp_path / "backend" / "routers").mkdir(parents=True)
    (tmp_path / "backend" / "sql_alchemy.py").write_text("class Booking: pass\n", encoding="utf-8")
    source = "from sql_alchemy import *\nx = Booking()\ny = Guest()\n"
    findings = diagnose_written_content("backend/routers/x.py", source, workspace=str(tmp_path))
    assert _names(findings) == {"Guest"}


def test_dunder_all_and_underscore_names_follow_python(tmp_path):
    (tmp_path / "m.py").write_text(
        "__all__ = ['public']\ndef public(): pass\ndef hidden(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "n.py").write_text("def shown(): pass\ndef _private(): pass\n", encoding="utf-8")
    source = "from m import *\nfrom n import *\npublic(); hidden(); shown(); _private()\n"
    findings = diagnose_written_content("x.py", source, workspace=str(tmp_path))
    assert _names(findings) == {"hidden", "_private"}


def test_star_exports_are_transitive(tmp_path):
    (tmp_path / "inner.py").write_text("def deep(): pass\n", encoding="utf-8")
    (tmp_path / "outer.py").write_text("from inner import *\n", encoding="utf-8")
    source = "from outer import *\ndeep(); gone()\n"
    findings = diagnose_written_content("x.py", source, workspace=str(tmp_path))
    assert _names(findings) == {"gone"}


def test_without_a_workspace_star_usage_is_still_not_judged():
    """Library callers that pass no workspace keep today's behaviour."""
    assert diagnose_written_content("r.py", "from m import *\nx = y\n") == []
