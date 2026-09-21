"""Whether the generated Python can actually resolve its own imports.

Three checks that need no model and no LLM: an import of a local module that
does not exist (``missing module:``), a name used behind ``import *`` that
nothing provides (``undefined name:``), and an ORM module that fails to
import or to configure its mappers when actually run (``mapper config:``).
The last one is the only check here that executes anything, in a subprocess
with an allowlisted environment.

The allowlist tables decide which import roots count as satisfied: the stdlib
half is taken from the interpreter rather than hand-maintained, and the
external half is what the generators are known to emit.

Split out of ``orchestrator.py``. Every function here was already
module-level and took a path - none of them ever touched run state.
"""

from __future__ import annotations

import ast as _ast
import os
import re as _re
import sys

from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.execution.process import _safe_subprocess_env
from besser.spec_driven_agent.validation.issues import _check_did_not_run
from besser.spec_driven_agent.validation.python_source import _python_files
from besser.spec_driven_agent.parsed_source import parse_source


# The stdlib half of the allowlist, taken from the interpreter rather than
# hand-maintained: the hand-written set was missing argparse, sqlite3, glob,
# zipfile and ~180 others, each of which would be reported as "this app
# cannot start".
_STDLIB_ROOTS = frozenset(getattr(sys, "stdlib_module_names", ())) | {
    "typing_extensions",  # not stdlib, but ubiquitous and always installed
}

# Third-party roots a generated Python app legitimately imports. This is a
# fallback: the authoritative source is the app's own manifest (see
# ``_declared_dependency_roots``). Kept because a generator can emit an
# import before the manifest entry, and a missing manifest should not turn
# every framework import into a blocker.
#
# Every non-FastAPI family used to be absent here, which is why a 2-class
# Django app whose requirements.txt declares Django was reported with 10
# blockers and "cannot start" (2026-09-17). All 20 generators were affected.
_EXTERNAL_IMPORT_ROOTS = frozenset({
    # FastAPI / Starlette
    "fastapi", "pydantic", "pydantic_settings", "pydantic_core", "sqlalchemy",
    "starlette", "uvicorn", "alembic", "sqlmodel", "databases", "anyio",
    "sniffio", "greenlet", "mako",
    # Django
    "django", "rest_framework", "corsheaders", "django_filters", "drf_yasg",
    "drf_spectacular", "django_extensions", "storages", "environ",
    # Flask
    "flask", "flask_sqlalchemy", "flask_cors", "flask_migrate",
    "flask_login", "flask_jwt_extended", "flask_restful", "werkzeug",
    "jinja2", "markupsafe", "itsdangerous", "click", "marshmallow",
    # Drivers / infra
    "psycopg", "psycopg2", "pymysql", "mysql", "aiosqlite", "asyncpg",
    "pymongo", "motor", "redis", "celery", "boto3", "botocore", "gunicorn",
    # HTTP / auth / serialisation
    "httpx", "requests", "aiohttp", "websockets", "urllib3", "certifi",
    "idna", "charset_normalizer", "jose", "jwt", "passlib", "bcrypt",
    "argon2", "cryptography", "nacl", "dotenv", "multipart",
    "email_validator", "yaml", "toml", "orjson", "ujson", "jsonschema",
    "dateutil", "pytz", "attr", "attrs",
    # Common utility packages
    "loguru", "structlog", "rich", "typer", "tenacity", "cachetools",
    "slugify", "phonenumbers", "validators", "shortuuid", "ulid", "qrcode",
    "PIL", "openpyxl", "reportlab", "markdown", "bleach", "babel",
    "numpy", "pandas", "openai", "anthropic", "stripe", "sentry_sdk",
    "prometheus_client", "apscheduler",
    # Test tooling
    "pytest", "pytest_asyncio", "faker", "factory", "freezegun", "responses",
    "httpretty", "hypothesis",
})

# Distribution name → import root, for the cases where they differ. The
# default rule (lowercase, '-' → '_') handles everything not listed.
_DIST_TO_IMPORT_ROOT = {
    "djangorestframework": "rest_framework",
    "django-cors-headers": "corsheaders",
    "django-environ": "environ",
    "python-jose": "jose",
    "pyjwt": "jwt",
    "python-dotenv": "dotenv",
    "python-multipart": "multipart",
    "python-slugify": "slugify",
    "psycopg2-binary": "psycopg2",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "sentry-sdk": "sentry_sdk",
    "mysqlclient": "MySQLdb",
    "python-docx": "docx",
    "attrs": "attr",
}

_REQUIREMENT_NAME_RE = _re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _declared_dependency_roots(output_dir: str) -> set[str]:
    """Import roots the app's own manifest declares.

    Reading the manifest is what makes this check work for all 20
    generators instead of only the stack whose packages someone
    remembered to hardcode. A dependency the app declares is a
    dependency the app has.
    """
    roots: set[str] = set()

    def _add(dist: str) -> None:
        dist = dist.strip().strip('"\'').lower()
        if not dist:
            return
        roots.add(_DIST_TO_IMPORT_ROOT.get(dist, dist.replace("-", "_")))

    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [
            d for d in dirs
            if d not in ("node_modules", _SNAPSHOT_DIR, "__pycache__", ".git")
        ]
        for name in files:
            low = name.lower()
            path = os.path.join(root, name)
            try:
                if low.startswith("requirements") and low.endswith(".txt"):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            line = line.split("#", 1)[0].strip()
                            if not line or line.startswith("-"):
                                continue
                            match = _REQUIREMENT_NAME_RE.match(line)
                            if match:
                                _add(match.group(1))
                elif low in ("pyproject.toml", "pipfile", "setup.py", "setup.cfg"):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                    # Deliberately loose: any quoted requirement-looking
                    # token. A false extra allowlist entry costs nothing;
                    # a missed one costs a spurious blocker.
                    for token in _re.findall(r"[\"']([A-Za-z0-9._-]{2,})"
                                             r"[><=!~\s\"']", text):
                        _add(token)
            except OSError:
                continue
    return roots




def _unresolvable_local_imports(output_dir: str) -> list[str]:
    """Local imports that name a module the app does not ship where it is used.

    Ruff cannot find these: ``from sql_alchemy import *`` makes it report
    "unable to detect undefined names", which EXCUSES every name instead of
    flagging it, so a missing MODULE is invisible to F821. Live 2026-09-11: an
    app imported ``sql_alchemy`` and ``pydantic_classes`` with neither file
    present and still reported "0 blockers / 23 total", status=done.

    Resolution rules, each learned from a false positive against a working app:

    * A service runs with its OWN folder as cwd (``uvicorn main_api:app`` from
      ``backend/``), so that folder is importable from everything beneath it.
      Resolving only against the file's own directory condemned six imports.
    * Any directory holding ``.py`` files is importable - Python 3 implicit
      namespace packages need no ``__init__.py``.
    * The module existing in SOME OTHER directory does not count: a copy under
      ``pydantic/`` is not reachable from ``backend/``.
    """
    try:
        py_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(output_dir)
            for name in files
            if name.endswith(".py")
            if not any(part in ("node_modules", _SNAPSHOT_DIR, "__pycache__")
                       for part in root.split(os.sep))
        ]
    except OSError:
        return []

    provided: dict[str, set[str]] = {}
    for path in py_files:
        provided.setdefault(os.path.dirname(path), set()).add(
            os.path.splitext(os.path.basename(path))[0]
        )
    package_dirs = set(provided)

    # A directory holding .py files is an importable package FROM ITS
    # PARENT, not only as a sibling of the importing file. Without this,
    # `from core.config import settings` in backend/routers/x.py was
    # reported missing even though backend/core/ exists and backend/ is
    # the service's cwd.
    for directory in package_dirs:
        parent = os.path.dirname(directory)
        if parent and parent != directory:
            provided.setdefault(parent, set()).add(os.path.basename(directory))

    declared = _declared_dependency_roots(output_dir)

    problems: list[str] = []
    for path in py_files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tree = parse_source(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue  # syntax errors are reported by their own check

        # Everything importable from this file: its own folder plus every
        # ancestor folder up to the app root (the service's cwd).
        reachable: set[str] = set()
        probe = os.path.dirname(path)
        while True:
            reachable |= provided.get(probe, set())
            if os.path.normpath(probe) == os.path.normpath(output_dir):
                break
            parent = os.path.dirname(probe)
            if not parent or parent == probe:
                break
            probe = parent

        roots: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                if node.level:            # explicit relative import
                    continue
                if node.module:
                    roots.append(node.module.split(".")[0])
            elif isinstance(node, _ast.Import):
                roots.extend(alias.name.split(".")[0] for alias in node.names)

        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        for root_name in sorted(set(roots)):
            if (root_name in _EXTERNAL_IMPORT_ROOTS
                    or root_name in _STDLIB_ROOTS
                    or root_name in declared
                    or root_name in reachable):
                continue
            if any(os.path.basename(d) == root_name and os.path.dirname(d) ==
                   os.path.dirname(path) for d in package_dirs):
                continue
            problems.append(
                f"missing module: {rel} imports '{root_name}', which the app "
                f"does not provide where it is used - this app cannot start"
            )
    return problems


_STAR_IMPORT_RE = _re.compile(r"^\s*from\s+\S+\s+import\s+\*", _re.MULTILINE)


def _star_import_undefined_names(output_dir: str) -> list[str]:
    """``undefined name:`` blockers for a name a star-importing module uses but
    neither defines nor receives from the modules it star-imports.

    ruff cannot report these: with ``from x import *`` in scope every
    unresolved name is F405 ("may be undefined"), never F821, so the delivered
    app of 2026-09-18 shipped six NameError routes as "0 blockers". Files
    without a star import are left to ruff, which reports them as F821
    already. Same resolver as the write-time check the model was shown.
    """
    import importlib

    try:
        importlib.import_module("pyflakes.checker")
    except ImportError:
        return [_check_did_not_run(
            "the star-import name check", "pyflakes is not installed",
        )]
    from besser.spec_driven_agent.validation.write_diagnostics import diagnose_written_content

    issues: list[str] = []
    for path in _python_files(output_dir):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not _STAR_IMPORT_RE.search(content):
            continue
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        for finding in diagnose_written_content(rel, content, workspace=output_dir):
            if finding.get("code") != "UndefinedName":
                continue
            detail = finding["message"].removeprefix("undefined name ")
            issues.append(f"undefined name: {rel} line {finding.get('line')}: {detail}")
    return issues


# The generated ORM module is imported and its mappers configured in a
# subprocess. SQLAlchemy resolves the strings in relationship() lazily, on the
# first query, so ast.parse and ruff both passed the 2026-09-18 live run
# 52befadf while every database request returned 500: a class-body
# ``relationship("Guest", secondary="booking_guest", ...)`` named a table that
# does not exist. Import-time NameErrors surface here as well - the class ruff
# excuses under a star import. ~0.5s per module, no database, server or
# network: the template keeps create_all under __main__.
_IMPORT_SMOKE_TIMEOUT_SECONDS = 30
_TRACEBACK_FRAME_RE = _re.compile(r'^\s*File "(.+?)", line (\d+)', _re.M)
_QUOTED_NAME_RE = _re.compile(r"'([^']+)'")


def _import_smoke_issues(output_dir: str) -> list[str]:
    """``mapper config:`` blockers for ORM modules that fail to import or
    configure; ``_check_did_not_run`` notes when the check itself could not."""
    import subprocess

    issues: list[str] = []
    for path in _python_files(output_dir):
        if os.path.basename(path) != "sql_alchemy.py":
            continue
        folder = os.path.dirname(path)
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        modules = ["sql_alchemy"]
        if os.path.isfile(os.path.join(folder, "pydantic_classes.py")):
            modules.append("pydantic_classes")
        code = (
            f"import {', '.join(modules)}\n"
            "from sqlalchemy.orm import configure_mappers\n"
            "configure_mappers()\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                timeout=_IMPORT_SMOKE_TIMEOUT_SECONDS,
                cwd=folder, env=_safe_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            issues.append(_check_did_not_run(
                "the import smoke check",
                f"timed out after {_IMPORT_SMOKE_TIMEOUT_SECONDS}s on {rel}",
            ))
            continue
        except OSError as exc:
            issues.append(_check_did_not_run(
                "the import smoke check", f"could not be launched: {exc}",
            ))
            continue
        if result.returncode == 0:
            continue
        stderr = result.stderr or ""
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
        error = (lines[-1] if lines else "exited non-zero with no error output")[:400]
        if error.startswith("ModuleNotFoundError:"):
            # The harness interpreter is not the app's venv: a third-party
            # module missing HERE says nothing about the app. A missing local
            # module is _unresolvable_local_imports' finding.
            issues.append(_check_did_not_run(
                "the import smoke check", f"{rel}: {error}",
            ))
            continue
        where = _import_smoke_location(output_dir, folder, rel, stderr, error)
        issues.append(f"mapper config: {where}: {error}")
    return issues


def _import_smoke_location(output_dir, folder, rel, stderr, error) -> str:
    """``<file> line N`` when it can be found, else ``<file>``.

    An import-time error's traceback ends in the app's own file. A mapper
    configuration error's ends inside sqlalchemy, so the quoted name from the
    message (``'booking_guest'``) is looked up in the module instead.
    """
    real_folder = os.path.normcase(os.path.realpath(folder))
    located = None
    for m in _TRACEBACK_FRAME_RE.finditer(stderr):
        frame = os.path.realpath(os.path.join(folder, m.group(1)))
        inside = os.path.normcase(frame).startswith(real_folder + os.sep)
        if inside and os.path.isfile(frame):
            located = (frame, int(m.group(2)))
    if located is not None:
        frame, line_no = located
        frame_rel = os.path.relpath(
            frame, os.path.realpath(output_dir)
        ).replace("\\", "/")
        return f"{frame_rel} line {line_no}"
    try:
        with open(os.path.join(folder, "sql_alchemy.py"), "r",
                  encoding="utf-8", errors="ignore") as fh:
            module_lines = fh.read().splitlines()
    except OSError:
        return rel
    for name in _QUOTED_NAME_RE.findall(error):
        token = _re.compile(rf"\b{_re.escape(name)}\b")
        for idx, text in enumerate(module_lines, 1):
            if token.search(text):
                return f"{rel} line {idx}"
    return rel
