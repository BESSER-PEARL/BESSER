"""Tests for the model-derived acceptance matrix (report-only)."""

import types

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.acceptance import build_acceptance_matrix, matrix_issues
from besser.generators.llm.orchestrator import _classify_issue

StringType = PrimitiveDataType("str")


def _model(*names) -> DomainModel:
    classes = set()
    for n in names:
        c = Class(name=n)
        c.attributes = {Property(name="name", type=StringType)}
        classes.add(c)
    return DomainModel(name="Test", types=classes)


def _workspace(tmp_path, files: dict):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


ROUTER = (
    "from fastapi import APIRouter\n"
    'router = APIRouter(prefix="/books", tags=["books"])\n'
    '@router.get("/")\n'
    "def list_books():\n    return []\n"
    '@router.post("/")\n'
    "def create_book(payload: dict):\n    return payload\n"
)

PAGE = (
    "import api from '../api';\n"
    "export default function Books() {\n"
    "  const submit = (b) => api.post('/books', b);\n"
    "  return <div>Books</div>;\n"
    "}\n"
)


def test_full_coverage_entity_is_all_green(tmp_path):
    _workspace(tmp_path, {
        "backend/routers/book.py": ROUTER,
        "frontend/src/pages/Books.tsx": PAGE,
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("Book"))
    assert matrix == {"Book": {"route": True, "page": True, "create": True}}
    assert matrix_issues(matrix) == []


def test_missing_entity_is_reported_not_blocked(tmp_path):
    _workspace(tmp_path, {
        "backend/routers/book.py": ROUTER,
        "frontend/src/pages/Books.tsx": PAGE,
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("Book", "Author"))
    assert matrix["Author"] == {"route": False, "page": False, "create": False}
    issues = matrix_issues(matrix)
    assert len(issues) == 1
    assert issues[0].startswith("acceptance: entity Author")
    assert "no backend REST route" in issues[0]
    # Advisory: classifies as warning, never blocker.
    assert _classify_issue(issues[0]).severity == "warning"


def test_page_without_post_is_flagged_create_missing(tmp_path):
    _workspace(tmp_path, {
        "backend/routers/book.py": ROUTER,
        "frontend/src/pages/Books.tsx": "export default () => <div>Books list</div>;",
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("Book"))
    assert matrix["Book"] == {"route": True, "page": True, "create": False}
    issues = matrix_issues(matrix)
    assert "create form not wired" in issues[0]


def test_plural_y_form_is_matched(tmp_path):
    _workspace(tmp_path, {
        "backend/routers/category.py": (
            'from fastapi import APIRouter\n'
            'router = APIRouter(prefix="/categories")\n'
            '@router.get("/")\ndef list_categories():\n    return []\n'
        ),
        "frontend/src/pages/Categories.tsx": (
            "import api from '../api';\n"
            "export const save = (c) => api.post('/categories', c);\n"
        ),
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("Category"))
    assert matrix == {"Category": {"route": True, "page": True, "create": True}}


def test_no_domain_model_returns_none(tmp_path):
    assert build_acceptance_matrix(str(tmp_path), None) is None
    assert matrix_issues(None) == []
