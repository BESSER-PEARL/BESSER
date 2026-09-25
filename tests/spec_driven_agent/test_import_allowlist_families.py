"""``missing module:`` must not fire on a stack that isn't FastAPI.

Observed live: a 2-class Django app whose ``requirements.txt``
declares ``Django>=5.0`` was reported with 10 blocker-level "this app
cannot start" issues, because ``django`` was simply absent from a
hardcoded allowlist that only ever listed the FastAPI ecosystem. The same
hole affects all 20 generators — the allowlist also had no ``flask``, and
no ``argparse``/``sqlite3``/``glob`` from the stdlib.

``missing module:`` is classified as a blocker, so every one of these
drives the Phase-3 fix loop against imports that were already correct.

The fix has three parts, one test each below: stdlib comes from the
interpreter, framework families are listed, and — the part that
generalises — the app's own manifest is read.
"""

from __future__ import annotations

import textwrap

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import (
    _declared_dependency_roots,
    _unresolvable_local_imports,
)


def _write(tmp_path, rel: str, body: str) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


def test_django_app_reports_no_missing_modules(tmp_path):
    _write(tmp_path, "requirements.txt", """
        Django>=5.0
        djangorestframework==3.15.1
        django-cors-headers
        psycopg2-binary
    """)
    _write(tmp_path, "hotel/settings.py", """
        from django.conf import settings
        from corsheaders.defaults import default_headers
        import os
    """)
    _write(tmp_path, "hotel/models.py", """
        from django.db import models
        from rest_framework import serializers
    """)
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_flask_app_reports_no_missing_modules(tmp_path):
    _write(tmp_path, "requirements.txt", "Flask\nFlask-SQLAlchemy\npython-dotenv\n")
    _write(tmp_path, "app.py", """
        from flask import Flask
        from flask_sqlalchemy import SQLAlchemy
        from dotenv import load_dotenv
    """)
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_stdlib_modules_the_old_allowlist_omitted(tmp_path):
    """Each of these was a blocker before the stdlib set came from sys."""
    _write(tmp_path, "main.py", """
        import argparse
        import sqlite3
        import glob
        import zipfile
        import pickle
        import calendar
        import mimetypes
        import ipaddress
    """)
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_manifest_covers_a_package_nobody_hardcoded(tmp_path):
    """The generalising case: an import root in no hand-written list, but
    declared by the app, is the app's dependency."""
    _write(tmp_path, "requirements.txt", "some-exotic-lib==1.2.3\n")
    _write(tmp_path, "main.py", "import some_exotic_lib\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_pyproject_dependencies_are_read(tmp_path):
    _write(tmp_path, "pyproject.toml", """
        [project]
        name = "demo"
        dependencies = ["litestar>=2.0", "msgspec"]
    """)
    _write(tmp_path, "main.py", "import litestar\nimport msgspec\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_subpackage_reachable_from_service_cwd(tmp_path):
    """`import core` where core/ sits beside the service's entry point.

    The old sibling-only rule required the package to be a sibling of the
    *importing file*, so a router one level down could not see it."""
    _write(tmp_path, "backend/main.py", "from core.config import settings\n")
    _write(tmp_path, "backend/core/config.py", "settings = {}\n")
    _write(tmp_path, "backend/routers/booking.py", "from core.config import settings\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


# ----------------------------------------------------------------------
# The check must still catch what it exists to catch
# ----------------------------------------------------------------------


def test_genuinely_missing_local_module_is_still_reported(tmp_path):
    """The live defect this validator was written for."""
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "main_api.py", "from sql_alchemy import *\nfrom fastapi import FastAPI\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1
    assert "sql_alchemy" in problems[0]


def test_undeclared_third_party_is_still_reported(tmp_path):
    """A package the app neither ships nor declares stays a blocker."""
    _write(tmp_path, "requirements.txt", "fastapi\n")
    _write(tmp_path, "main.py", "import nonexistent_vendor_sdk\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1
    assert "nonexistent_vendor_sdk" in problems[0]


def test_module_in_an_unreachable_sibling_directory_is_still_reported(tmp_path):
    """A copy under pydantic/ is not reachable from backend/."""
    _write(tmp_path, "backend/main.py", "import helpers\n")
    _write(tmp_path, "pydantic_out/helpers.py", "x = 1\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert len(problems) == 1
    assert "helpers" in problems[0]


# ----------------------------------------------------------------------
# Manifest parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist,expected",
    [
        ("djangorestframework", "rest_framework"),
        ("django-cors-headers", "corsheaders"),
        ("python-jose", "jose"),
        ("PyJWT", "jwt"),
        ("python-dotenv", "dotenv"),
        ("Pillow", "PIL"),
        ("psycopg2-binary", "psycopg2"),
        ("some-new-thing", "some_new_thing"),
    ],
)
def test_distribution_name_maps_to_import_root(tmp_path, dist, expected):
    _write(tmp_path, "requirements.txt", f"{dist}==1.0\n")
    assert expected in _declared_dependency_roots(str(tmp_path))


def test_requirements_comments_and_flags_are_skipped(tmp_path):
    _write(tmp_path, "requirements.txt", """
        # a comment
        --index-url https://example.invalid/simple
        -r other.txt
        fastapi==0.141.0  # inline comment
    """)
    roots = _declared_dependency_roots(str(tmp_path))
    assert "fastapi" in roots
    assert not any(r.startswith("#") or r.startswith("-") for r in roots)
