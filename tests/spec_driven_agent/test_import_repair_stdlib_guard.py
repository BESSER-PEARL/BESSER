"""``import <stdlib module>`` only for a real module use; a framework's own re-export still resolves.

Review of round 10 (``imp.py``): the stdlib rule fired for any undefined VARIABLE read
as ``name.attr`` whose name is also a stdlib module (``email.lower()``,
``token.split``, ``profile.bio``, ``code.strip()``, ``string.upper()``). The file then
imported cleanly, ruff's F821 disappeared and the call failed at runtime. A local
app module (``calendar.build_schedule``) got the stdlib import too. And dropping every
module attribute lost ``from fastapi import status`` (``reexport.py``).
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.repair.import_repair import _exporting_modules, repair_missing_imports

_VARIABLES = {
    "token": "def me(authorization):\n    return token.split('.')[0]\n",
    "email": "def welcome(user):\n    return {'to': email.lower(), 'name': user.name}\n",
    "profile": "def show(user):\n    return {'bio': profile.bio}\n",
    "code": "def check(otp):\n    return code.strip() == otp\n",
    "string": "def up(s):\n    return string.upper()\n",
    "this": "def f():\n    return this.s\n",
}


@pytest.mark.parametrize("name", sorted(_VARIABLES))
def test_an_undefined_variable_named_like_a_stdlib_module_is_left_alone(tmp_path, name):
    source = _VARIABLES[name]
    assert repair_missing_imports("backend/x.py", source, root=str(tmp_path)) == (source, [])


def test_a_local_module_of_the_same_name_is_not_replaced_by_the_stdlib_one(tmp_path):
    # Beside the file, so ``json`` imports the app's module (where shadowing counts: see
    # test_import_repair_local_shadow.py).
    (tmp_path / "backend").mkdir(parents=True)
    (tmp_path / "backend" / "json.py").write_text("def loads(x):\n    return x\n", encoding="utf-8")
    source = "def plan(text):\n    return json.loads(text)\n"
    assert repair_missing_imports("backend/x.py", source, root=str(tmp_path)) == (source, [])


def test_a_function_the_stdlib_module_lacks_gets_no_import(tmp_path):
    # The reviewer's ``calendar.build_schedule``: an app's services/calendar.py was meant.
    source = "def plan(day):\n    return calendar.build_schedule(day)\n"
    assert repair_missing_imports("backend/x.py", source, root=str(tmp_path)) == (source, [])


def test_a_real_stdlib_use_is_still_imported(tmp_path):
    repaired, notes = repair_missing_imports("backend/x.py", "def f(t):\n    return re.findall('a', t)\n",
                                             root=str(tmp_path))
    assert notes == ["added 'import re'"] and "import re\n" in repaired


def test_without_a_workspace_root_no_stdlib_import_is_added():
    source = "def f(t):\n    return re.findall('a', t)\n"
    assert repair_missing_imports("backend/x.py", source) == (source, [])


def test_an_explicit_framework_reexport_resolves():
    assert _exporting_modules("status") == ["fastapi"]
    source = ("from fastapi import HTTPException\n\n\n"
              "def f():\n    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)\n")
    repaired, notes = repair_missing_imports("backend/x.py", source)
    assert "from fastapi import HTTPException, status\n" in repaired, repaired
    assert notes


@pytest.mark.parametrize("name", ["re", "os", "engine"])
def test_a_module_merely_held_is_still_rejected(name):
    assert _exporting_modules(name) == []


def test_a_reexported_module_lacking_the_used_attribute_is_rejected():
    # ``sqlalchemy.orm`` re-exports its own ``exc``, which has no IntegrityError.
    source = "def f(e):\n    return isinstance(e, exc.IntegrityError)\n"
    assert repair_missing_imports("backend/x.py", source) == (source, [])
