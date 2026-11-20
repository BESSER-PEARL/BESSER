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


@pytest.mark.parametrize("handler", [
    "except Exception as exc:\n        return JSONResponse({'detail': str(exc)}, status_code=500)\n",
    "except Exception:\n        logger.exception('export failed')\n        raise\n",
])
def test_a_handler_that_returns_or_raises_does_not_excuse_a_forgotten_module(tmp_path, handler):
    """Review of d5b411a20: an endpoint that answers 500 from its catch-all has no fallback,
    so a module that was never written fails every call and must still be reported."""
    _write(tmp_path, "backend/requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/main.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    _write(tmp_path, "backend/routers/reports.py",
           "def export_report(report_id: int):\n    try:\n        from report_builder import build_pdf\n"
           "        return build_pdf(report_id)\n    " + handler)
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and "report_builder" in problems[0], problems


@pytest.mark.parametrize("body", [
    "    try:\n        import vendor_parser\n    except Exception:\n        print('parser unavailable')\n"
    "    return vendor_parser.parse(raw)\n",
    "    try:\n        import vendor_parser\n    except Exception:\n        return vendor_parser.fallback(raw)\n",
    "    with suppress(Exception):\n        from vendor_parser import parse\n    return parse(raw)\n",
], ids=["after-try", "in-handler", "after-suppress"])
def test_a_broad_guarded_import_used_outside_its_guard_is_still_reported(tmp_path, body):
    """Used where the import may have failed, the name raises NameError on every call: not optional."""
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/extract.py", "from contextlib import suppress\n\n\ndef read(raw):\n" + body)
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and problems[0].startswith("possibly missing module:"), problems


@pytest.mark.parametrize("body", [
    "    with suppress(Exception):\n        from vendor_parser import parse\n        return parse(raw)\n"
    "    return raw.decode()\n",
    "    try:\n        import vendor_parser as vp\n    except Exception:\n        pass\n"
    "    else:\n        return vp.parse(raw)\n    return raw.decode()\n",
], ids=["in-suppress", "in-else"])
def test_a_broad_guarded_import_used_only_under_its_guard_is_optional(tmp_path, body):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/extract.py", "from contextlib import suppress\n\n\ndef read(raw):\n" + body)
    assert _unresolvable_local_imports(str(tmp_path)) == []


@pytest.mark.parametrize("body", [
    "    try:\n        import vendor_json as json\n    except Exception:\n        import json\n    return json.dumps(raw)\n",
    "    try:\n        import vendor_parser\n    except Exception:\n        vendor_parser = None\n"
    "    if vendor_parser:\n        return vendor_parser.parse(raw)\n    return raw\n",
    "    vp = None\n    with suppress(Exception):\n        import vendor_parser as vp\n    return vp.parse(raw) if vp else raw\n",
    "    parse = str\n    raw = parse(raw)\n    try:\n        from vendor_parser import parse\n        raw = parse(raw)\n"
    "    except Exception:\n        pass\n    return parse(raw)\n",
    "    try:\n        import vendor_parser\n        raw = vendor_parser.parse(raw)\n    except Exception:\n        pass\n"
    "    def show(vendor_parser):\n        return vendor_parser\n    return show(raw)\n",
    "    try:\n        import vendor_parser\n        raw = vendor_parser.parse(raw)\n    except Exception:\n        pass\n"
    "    vendor_parser = object()\n    return vendor_parser\n",
], ids=["handler-imports-a-fallback", "handler-assigns-none", "bound-before-the-guard",
        "local-bound-before-the-guard", "nested-function-parameter", "rebound-after-the-guard"])
def test_a_broad_guarded_import_with_a_fallback_binding_is_optional(tmp_path, body):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/extract.py", "from contextlib import suppress\n\n\ndef read(raw):\n" + body)
    assert _unresolvable_local_imports(str(tmp_path)) == []


@pytest.mark.parametrize("body", [
    "    try:\n        import vendor_parser\n        raw = vendor_parser.parse(raw)\n    except Exception:\n        pass\n"
    "    finally:\n        print(vendor_parser)\n    return raw\n",
    "    try:\n        import vendor_parser\n    except Exception:\n        pass\n    later = lambda: vendor_parser.parse(raw)\n"
    "    return later()\n",
    "    try:\n        import vendor_pkg.sub\n    except Exception:\n        pass\n    return vendor_pkg.sub.go(raw)\n",
    "    try:\n        from vendor_parser import first, second\n        raw = first(raw)\n    except Exception:\n        pass\n"
    "    return second(raw)\n",
    # A flag set in the handler is not followed: the read is reported (a warning).
    "    try:\n        import vendor_parser\n        found = True\n    except Exception:\n        found = False\n"
    "    if found:\n        return vendor_parser.parse(raw)\n    return raw\n",
], ids=["read-in-finally", "read-in-a-closure", "dotted-import-read-as-its-root", "second-name-read-after",
        "flag-pattern"])
def test_a_broad_guarded_import_read_where_it_may_have_failed_is_reported(tmp_path, body):
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "backend/extract.py", "from contextlib import suppress\n\n\ndef read(raw):\n" + body)
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1 and problems[0].startswith("possibly missing module:"), problems
