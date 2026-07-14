"""LIVE workflow tests: the two demo scenarios ("only a database",
"database + backend") driven over the DEPLOYED backend HTTP endpoint
(POST /besser_api/generate-output), asserting the REAL produced code — the
deploy-gate companion to the offline tests/workflows/test_generation_workflows.py.

Also covers code-gen robustness edge cases (empty model / malformed payload must
degrade gracefully — a clean 4xx, never a 500 / crash / hang).

Skipped by default (needs a running backend + the ``requests`` package). Enable::

    RUN_LIVE_BACKEND_TESTS=1 python -m pytest tests/live/test_generation_workflows_live.py

Point at a different host with BACKEND_URL (default: the experimental deploy).
"""
import io
import os
import zipfile

import pytest

BACKEND_URL = os.environ.get(
    "BACKEND_URL", "https://experimental.besser-pearl.org/besser_api"
).rstrip("/")

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_BACKEND_TESTS"),
    reason="live backend test — set RUN_LIVE_BACKEND_TESTS=1 to run",
)

# Author 1..* --- 0..* Book (frontend JSON shape, per test_api_integration.py).
_MODEL = {
    "type": "ClassDiagram",
    "elements": {
        "class-1": {"type": "Class", "name": "Author", "attributes": ["attr-1"], "methods": []},
        "attr-1": {"type": "Attribute", "name": "+ name: str"},
        "class-2": {"type": "Class", "name": "Book", "attributes": ["attr-2"], "methods": []},
        "attr-2": {"type": "Attribute", "name": "+ title: str"},
    },
    "relationships": {
        "rel-1": {
            "type": "ClassBidirectional",
            "source": {"element": "class-1", "multiplicity": "1..*", "role": "authors"},
            "target": {"element": "class-2", "multiplicity": "0..*", "role": "books"},
        },
    },
}


def _post(payload, timeout=120):
    requests = pytest.importorskip("requests")
    return requests.post(f"{BACKEND_URL}/generate-output", json=payload, timeout=timeout)


def _generate(generator, config=None, model=None):
    payload = {"title": "LibraryModel", "model": model or _MODEL, "generator": generator}
    if config:
        payload["config"] = config
    return _post(payload)


# ======================================================================
# Scenario: "only a database"
# ======================================================================

class TestOnlyDatabaseLive:

    def test_sql(self):
        r = _generate("sql", {"dialect": "sqlite"})
        assert r.status_code == 200, r.text[:300]
        assert "tables.sql" in r.headers.get("content-disposition", "")
        low = r.text.lower()
        assert "create table" in low
        assert "author" in low and "book" in low

    def test_sqlalchemy(self):
        r = _generate("sqlalchemy", {"dbms": "sqlite"})
        assert r.status_code == 200, r.text[:300]
        assert "sql_alchemy.py" in r.headers.get("content-disposition", "")
        assert "class Author(Base):" in r.text and "class Book(Base):" in r.text


# ======================================================================
# Scenario: "database + backend" (FastAPI)
# ======================================================================

class TestDatabaseAndBackendLive:

    def test_fastapi_backend_zip(self):
        r = _generate("backend")
        assert r.status_code == 200, r.text[:300]
        assert "application/zip" in r.headers.get("content-type", "")
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(n.endswith("main_api.py") for n in names), names
        assert any(n.endswith("database.py") for n in names), names
        assert any(n.endswith("routers/author.py") or n.endswith("author.py")
                   for n in names), names
        main_api = zf.read(
            next(n for n in names if n.endswith("main_api.py"))
        ).decode("utf-8")
        assert "FastAPI(" in main_api


# ======================================================================
# Robustness: bad input must degrade gracefully (never a 500 / crash / hang)
# ======================================================================

class TestGenerationRobustnessLive:

    def test_empty_model_does_not_500(self):
        empty = {"type": "ClassDiagram", "elements": {}, "relationships": {}}
        r = _generate("python", model=empty)
        assert r.status_code != 500, f"empty model 500'd: {r.text[:200]}"

    def test_malformed_payload_returns_4xx(self):
        # Missing the required 'model' field.
        r = _post({"title": "X", "generator": "python"}, timeout=60)
        assert r.status_code in (400, 422), \
            f"expected 4xx for malformed payload, got {r.status_code}: {r.text[:200]}"

    def test_invalid_sql_dialect_does_not_500(self):
        r = _generate("sql", {"dialect": "totally-not-a-dialect"})
        assert r.status_code != 500, f"invalid dialect 500'd: {r.text[:200]}"
