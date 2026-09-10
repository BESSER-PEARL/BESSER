"""Frontend HTTP calls must resolve against the generated backend manifest."""

from besser.generators.llm.endpoint_coherence import collect_endpoint_coherence_issues


def _seed(tmp_path, rel_path: str, content: str) -> None:
    path = tmp_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _backend(tmp_path) -> None:
    _seed(tmp_path, "backend/routers/books.py", (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        '@router.get("/book/")\n'
        "def list_books(): ...\n"
        '@router.post("/book/")\n'
        "def create_book(): ...\n"
        '@router.get("/book/{book_id}/")\n'
        "def get_book(): ...\n"
    ))


def test_exact_fetch_and_template_parameter_match(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.ts", (
        "export const all = () => fetch(`${API_URL}/book/`);\n"
        "export const one = (id) => fetch(`${API_URL}/book/${id}/`);\n"
        "export const add = (x) => fetch('/book/', { method: 'POST', body: x });\n"
    ))
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []


def test_pluralized_path_is_reported_with_file_and_line(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.ts", (
        "export const ok = true;\n"
        "export const all = () => fetch('/books');\n"
    ))
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1
    assert "frontend/src/api.ts line 2" in issues[0]
    assert "GET /books" in issues[0]


def test_wrong_method_is_reported(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.ts", "axios.delete('/book/')\n")
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1
    assert "path exists only for GET, POST" in issues[0]


def test_axios_config_form_is_checked(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.ts", (
        "axios({ url: '/api/book/', method: 'post', data: book })\n"
    ))
    assert "POST /api/book/" in collect_endpoint_coherence_issues(str(tmp_path))[0]


def test_external_and_static_asset_fetches_are_ignored(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.ts", (
        "fetch('https://status.example.com/v1/health')\n"
        "fetch('/fixtures/books.json')\n"
    ))
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []


def test_no_backend_manifest_means_not_applicable(tmp_path):
    _seed(tmp_path, "frontend/src/api.ts", "fetch('/books')\n")
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []
