"""Which guarded and declared imports may still be a missing module.

A ``try: import x / except ImportError`` is an optional import: the app
starts without ``x``. A broad ``except Exception`` around an import of a
module the app does not ship more often hides a defect (a forgotten local
file), so it is reported, as a warning rather than a ``missing module:``
blocker. A declared distribution excuses an import only when the import is its
module: ``auth-lib`` does not provide the app's own ``auth``.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.python_imports import _unresolvable_local_imports


def _write(root, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("handler", [
    "except Exception:\n    settings = None\n",
    "except BaseException:\n    settings = None\n",
    "except:\n    settings = None\n",
    "except (ValueError, Exception) as exc:\n    print(exc)\n",
])
def test_a_broad_handler_around_a_missing_local_module_is_a_warning(tmp_path, handler):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/main.py", "try:\n    from config import settings\n" + handler)
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and "'config'" in problems[0], problems
    assert not problems[0].startswith("missing module:")
    assert _classify_issue(problems[0]).severity == "warning"


def test_a_suppress_exception_around_a_missing_module_is_a_warning(tmp_path):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/main.py",
           "import contextlib\nwith contextlib.suppress(Exception):\n    import config\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and _classify_issue(problems[0]).severity == "warning", problems


@pytest.mark.parametrize("source", [
    "try:\n    import sentry_sdk\nexcept Exception:\n    sentry_sdk = None\n",
    "try:\n    import uvloop\nexcept Exception:\n    uvloop = None\n",
    "try:\n    from config import settings\nexcept Exception:\n    settings = None\n",
])
def test_a_broad_handler_around_a_resolvable_module_stays_silent(tmp_path, source):
    """A known or declared package, or a module the app ships, is never reported."""
    _write(tmp_path, "requirements.txt", "fastapi\nuvloop\n")
    _write(tmp_path, "backend/config.py", "settings = {}\n")
    _write(tmp_path, "backend/main.py", source)
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_the_fallback_in_a_broad_handler_is_not_reported(tmp_path):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/main.py",
           "try:\n    import orjson as json\nexcept Exception:\n    import json_fallback as json\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_an_import_error_handler_stays_exempt(tmp_path):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/main.py", "try:\n    import vendor_sdk\nexcept ImportError:\n    vendor_sdk = None\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


@pytest.mark.parametrize("dist,module", [
    ("auth-lib", "auth"),
    ("models-common", "models"),
    ("python-config-tools", "config"),
])
def test_a_distribution_named_after_a_module_the_app_ships_does_not_excuse_it(tmp_path, dist, module):
    """The app's own ``worker/auth.py`` proves ``auth`` is local; ``auth-lib`` does not provide it."""
    _write(tmp_path, "requirements.txt", f"fastapi\n{dist}==1.0\n")
    _write(tmp_path, f"worker/{module}.py", "x = 1\n")
    _write(tmp_path, "backend/main.py", f"from {module} import x\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and problems[0].startswith("missing module:") and f"'{module}'" in problems[0]


def test_a_distribution_named_exactly_after_a_shipped_module_still_excuses_it(tmp_path):
    """Only the name-prefix rule yields to local evidence, not an exact distribution name."""
    _write(tmp_path, "requirements.txt", "fastapi\nauth==1.0\n")
    _write(tmp_path, "worker/auth.py", "x = 1\n")
    _write(tmp_path, "backend/main.py", "from auth import x\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


@pytest.mark.parametrize("handler", ["except Exception:\n        text = None\n", "except:\n        pass\n"])
def test_a_broad_handler_around_an_import_inside_a_function_is_an_optional_import(tmp_path, handler):
    """A deferred import under a catch-all is the app's optional fast path with a fallback after it:
    it never runs at startup and the handler is the designed failure."""
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/extract.py",
           "def read(raw):\n    text = None\n    try:\n        import vendor_parser\n"
           "        text = vendor_parser.parse(raw)\n    " + handler
           + "    return text or raw.decode()\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_an_unguarded_import_inside_a_function_is_still_reported(tmp_path):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/extract.py", "def read(raw):\n    import vendor_parser\n    return vendor_parser.parse(raw)\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and problems[0].startswith("missing module:"), problems
