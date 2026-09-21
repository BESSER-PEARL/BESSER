"""``NAME.attr`` (a call or a plain attribute) where ``NAME`` is provided only
by a star import and the real object behind it does not have ``attr``.

Live run _abcgx9s: a generated hotel app's ``routers/booking_methods.py``
(``from pydantic_classes import *``) wrote

    billNumber=f"BIL-{booking_id}-{int(time.time())}",

``pydantic_classes.py`` does ``from datetime import datetime, date, time``,
so the star import binds ``time`` to the *class* ``datetime.time``, not the
``time`` module. ``time.time()`` raises ``AttributeError: type object
'datetime.time' has no attribute 'time'`` on the first request that reaches
it - confirmed live at
verification/spec-iterations/Qwen-Qwen3-30B-A3B-Instruct-2507-_abcgx9s/app/web_app/backend/routers/booking_methods.py:45.
``time`` is genuinely bound, so pyflakes and ruff both pass, and the file
``ast.parse``s cleanly too: the same "ships green, boots dead" class as an
undefined name behind a star import, one layer over.
"""
from __future__ import annotations

import textwrap

from besser.spec_driven_agent.write_diagnostics import (
    diagnose_written_content, python_structural_diagnostics,
)

ROUTER_PATH = "backend/routers/booking_methods.py"
PYDANTIC_PATH = "backend/pydantic_classes.py"

# The exact live shape: pydantic_classes.py shadows the `time` module with
# the `datetime.time` CLASS via `from datetime import ..., time`.
PYDANTIC_CLASSES = textwrap.dedent("""\
    from datetime import datetime, date, time
    from pydantic import BaseModel


    class Bill(BaseModel):
        billNumber: str
        totalAmountDue: float
    """)


def _findings(tmp_path, orm_source, router_source, orm_path=PYDANTIC_PATH, router_path=ROUTER_PATH):
    (tmp_path / orm_path.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
    (tmp_path / orm_path).write_text(orm_source, encoding="utf-8")
    return diagnose_written_content(router_path, router_source, workspace=str(tmp_path))


def _misuse_findings(findings):
    return [f for f in findings if f["code"] == "star-import-attribute-misuse"]


# -- required case: the exact live shape is reported ------------------------
def test_live_shape_time_time_call_shadowed_by_datetime_time_is_reported(tmp_path):
    """FAILS before the fix: write_diagnostics had no notion of what object a
    star import actually binds a name to, only whether the name is bound at
    all (pyflakes' ImportStarUsage / this module's UndefinedName). 'time' IS
    bound - just to the wrong object - so neither one fired, and this is a
    live AttributeError at request time, not an undefined name.

    The adjacent 'datetime.now().date()' line is included deliberately: it
    is the SAME kind of shadowing ('datetime' is also rebound to a class by
    this import), but it is valid code (the class has a .now() classmethod),
    and must not be flagged.
    """
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def produce_bill(booking_id, _booking_object, total_amount):
            _bill = Bill(
                billNumber=f"BIL-{booking_id}-{int(time.time())}",
                booking=_booking_object,
                issuedDate=datetime.now().date(),
                totalAmountDue=total_amount,
                settled=False,
            )
            return _bill
        """)
    findings = _misuse_findings(_findings(tmp_path, PYDANTIC_CLASSES, router))
    assert len(findings) == 1, findings
    finding = findings[0]
    assert finding["line"] == 6
    assert "time.time" in finding["message"]
    assert "datetime.time" in finding["message"]
    assert "pydantic_classes" in finding["message"]
    # The valid, adjacent datetime.now() use must not itself be reported.
    assert all(f["line"] != 8 for f in findings)


def test_plain_attribute_access_without_a_call_is_also_reported(tmp_path):
    """FAILS before the fix, same root cause as above. The task's shape is
    explicitly 'call or plain attribute' - this is the non-call half."""
    router = textwrap.dedent("""\
        from pydantic_classes import *

        _time_fn = time.time
        """)
    findings = _misuse_findings(_findings(tmp_path, PYDANTIC_CLASSES, router))
    assert len(findings) == 1, findings
    assert findings[0]["line"] == 3


def test_missing_attribute_on_a_genuinely_star_imported_module_is_reported(tmp_path):
    """FAILS before the fix. Not the datetime/time shape at all - proves the
    rule is general ('the bound object does not have that attribute'), not
    special-cased to 'time'. pydantic_classes.py re-exports the real 'os'
    module via a plain 'import os'; 'os' genuinely has no such attribute."""
    orm = PYDANTIC_CLASSES + "import os\n"
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def f():
            return os.definitely_not_a_real_stdlib_attr()
        """)
    findings = _misuse_findings(_findings(tmp_path, orm, router))
    assert len(findings) == 1, findings
    assert "os.definitely_not_a_real_stdlib_attr" in findings[0]["message"]
    assert "the module 'os'" in findings[0]["message"]


# -- required case: a real module whose attribute genuinely exists ----------
def test_real_module_reexported_with_an_existing_attribute_is_not_flagged(tmp_path):
    """Passes both before and after the fix - the bar the detector must not
    cross. 'import os' re-exported via the star import, then a genuinely
    valid os.path.join(...) call."""
    orm = PYDANTIC_CLASSES + "import os\n"
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def f():
            return os.path.join("a", "b")
        """)
    assert _misuse_findings(_findings(tmp_path, orm, router)) == []


# -- required case: the file's own binding shadows the star import ----------
def test_a_direct_local_import_shadows_the_star_import(tmp_path):
    """Passes both before and after the fix. The router imports the real
    'time' module itself; that import, not pydantic_classes.py's star
    export, is what 'time' resolves to here."""
    router = textwrap.dedent("""\
        import time

        from pydantic_classes import *


        def produce_bill(booking_id):
            return f"BIL-{booking_id}-{int(time.time())}"
        """)
    assert _misuse_findings(_findings(tmp_path, PYDANTIC_CLASSES, router)) == []


# -- required case: an attribute on an instance, not the imported object ----
def test_an_attribute_on_a_local_parameter_is_never_considered(tmp_path):
    """Passes both before and after the fix. '_booking_object' is a plain
    function parameter, never bound by any import; '.physicalStatus' is an
    attribute on that instance. Structurally outside what this detector
    inspects: it only ever looks at NAME.attr where NAME is itself one of
    the star-imported bindings, never a variable that merely holds one."""
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def register_arrival(_booking_object):
            return _booking_object.physicalStatus
        """)
    assert _misuse_findings(_findings(tmp_path, PYDANTIC_CLASSES, router)) == []


def test_a_local_reassignment_shadows_the_star_import(tmp_path):
    """Passes both before and after the fix. 'time' is reassigned to some
    other object inside the function; Python's own scoping makes every use
    of 'time' in that function the local variable, not the star import -
    an instance sharing the star-imported name, not the imported object."""
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def f():
            time = _make_timer()
            return time.time()
        """)
    assert _misuse_findings(_findings(tmp_path, PYDANTIC_CLASSES, router)) == []


# -- required case: a builtin name is never treated as a check target -------
def test_a_name_that_is_also_a_builtin_is_never_flagged(tmp_path):
    """Passes both before and after the fix. pydantic_classes.py binds
    'type' (a builtin) to the real 're' module via 'import re as type'; even
    though 're' genuinely lacks the attribute used here, 'type' is excluded
    purely for being a builtin name, per spec."""
    orm = PYDANTIC_CLASSES + "import re as type\n"
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def f():
            return type.definitely_not_a_real_attr()
        """)
    assert _misuse_findings(_findings(tmp_path, orm, router)) == []


# -- required case: an unresolvable star module reports nothing -------------
def test_unresolvable_star_module_reports_nothing(tmp_path):
    """No pydantic_classes.py on disk at all: the module cannot be resolved,
    so nothing is reported rather than guessed at. Passes both before and
    after the fix; included as the contract's explicit boundary."""
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def produce_bill(booking_id):
            return f"BIL-{booking_id}-{int(time.time())}"
        """)
    (tmp_path / "backend" / "routers").mkdir(parents=True)
    findings = diagnose_written_content(ROUTER_PATH, router, workspace=str(tmp_path))
    assert _misuse_findings(findings) == []


# -- precision: __all__ can exclude a resolvable name from the star import --
def test_dunder_all_excludes_a_resolvable_name_from_the_star_import(tmp_path):
    """Passes both before and after the fix. pydantic_classes.py's __all__
    omits 'time', so 'from pydantic_classes import *' never actually binds
    it in the router at all - a plain undefined name (already covered by the
    existing pyflakes-backed check), not this detector's shape. Locks in
    that the __all__ intersection this detector applies does not over-fire
    on a name it could otherwise resolve but that is not actually exported.
    """
    orm = textwrap.dedent("""\
        __all__ = ["Bill"]
        from datetime import time
        from pydantic import BaseModel


        class Bill(BaseModel):
            billNumber: str
        """)
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def f():
            return time.time()
        """)
    findings = _findings(tmp_path, orm, router)
    assert _misuse_findings(findings) == []
    # The name is genuinely undefined here - the existing detector, not this
    # one, is expected to be the one that catches it.
    assert any(f["code"] == "UndefinedName" for f in findings), findings


# -- precision: a name rebound after its import is not trusted --------------
def test_a_name_reassigned_after_its_import_in_the_star_module_is_not_trusted(tmp_path):
    """Passes both before and after the fix. pydantic_classes.py imports
    'time' from datetime and then reassigns it to something else entirely
    before the module finishes executing; the import-based resolution can no
    longer prove what object the star import actually hands out, so nothing
    is reported rather than guessed at."""
    orm = textwrap.dedent("""\
        from datetime import time
        from pydantic import BaseModel


        class Bill(BaseModel):
            billNumber: str


        def _compute_default_time():
            return 0


        time = _compute_default_time()
        """)
    router = textwrap.dedent("""\
        from pydantic_classes import *


        def f():
            return time.time()
        """)
    assert _misuse_findings(_findings(tmp_path, orm, router)) == []


# -- backward compatibility: the pre-existing 2-argument call still works ---
def test_python_structural_diagnostics_keeps_its_old_two_argument_call_working():
    """requirements_ledger.py calls python_structural_diagnostics(tree,
    sqlite=sqlite) with no rel_path/workspace; the new parameters must stay
    optional and default to skipping this new check too, not raise."""
    import ast

    tree = ast.parse(
        "from pydantic_classes import *\n"
        "def f():\n"
        "    return time.time()\n"
    )
    findings = python_structural_diagnostics(tree, sqlite=False)
    assert _misuse_findings(findings) == []
