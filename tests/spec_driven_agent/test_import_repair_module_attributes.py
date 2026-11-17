"""A missing stdlib module name is imported as that module, never from a module that happens to hold it.

Live run r12-p81, turn 34: ``re.findall(...)`` without ``import re`` was "repaired"
by adding ``re`` to ``from typing import List, Optional``. ``typing.re`` is a
deprecated alias module, so the file imported cleanly and then raised
``AttributeError: type object 'typing.re' has no attribute 'findall'``; the
model spent a turn undoing it. ``os`` mapped to ``uuid`` the same way
(``uuid`` imports ``os``). A name that is itself a stdlib module now becomes
``import <name>`` when every use reads an attribute of it; a re-export a
module's ``__all__`` leaves out is not an export.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.repair.import_repair import _exporting_modules, repair_missing_imports


@pytest.mark.parametrize("name, line", [
    ("re", "return bool(re.findall(r'[a-z]+', text))"),
    ("os", "return os.path.join(text, 'x')"),
    ("json", "return json.dumps(text)"),
])
def test_a_missing_stdlib_module_is_imported_as_itself(tmp_path, name, line):
    source = f"from typing import List, Optional\n\n\ndef f(text: Optional[str]):\n    {line}\n"

    repaired, notes = repair_missing_imports("backend/x.py", source, root=str(tmp_path))

    assert f"import {name}\n" in repaired, repaired
    assert "from typing import List, Optional\n" in repaired, repaired
    assert notes == [f"added 'import {name}'"]
    namespace: dict = {}
    exec(compile(repaired, "x.py", "exec"), namespace)
    namespace["f"]("abc")


@pytest.mark.parametrize("name", ["re", "os", "sys", "collections", "io", "functools"])
def test_no_module_exports_a_stdlib_module_it_merely_imported(name):
    assert _exporting_modules(name) == []


def test_a_stdlib_name_used_as_a_value_is_left_alone():
    source = "def f():\n    return json\n"
    assert repair_missing_imports("x.py", source) == (source, [])


def test_a_module_the_package_merely_holds_is_never_imported():
    """Replay of r10-p81 ``routers/risk_analysis.py`` without its ``import risk_engine as engine``:
    the old repair added ``from sqlalchemy import engine``, a module of the wrong package.
    (An explicit re-export such as ``fastapi.status`` is covered in test_import_repair_stdlib_guard.py.)"""
    assert _exporting_modules("engine") == []
    source = "def f():\n    return engine.score(1)\n"
    assert repair_missing_imports("x.py", source) == (source, [])


def test_select_stays_the_sqlalchemy_function():
    # ``select`` is also a stdlib module; a name some candidate exports is never re-read as one.
    source = "def f(User):\n    return select(User)\n"
    repaired, _ = repair_missing_imports("x.py", source)
    assert "from sqlalchemy import select\n" in repaired, repaired
