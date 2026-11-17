"""Frontend HTTP calls must resolve against the generated backend manifest."""

from besser.spec_driven_agent.validation.endpoint_coherence import collect_endpoint_coherence_issues


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


# The shared client's wrapper (Sonnet 5.5, a hotel app and a todo app): its
# options are a variable, so the URL used to run on past its closing backtick
# to the next one and was reported as `GET /{param}`, opts); } catch ...`.
WRAPPER = (
    "async function request(method, path, body) {\n"
    "  let res;\n"
    "  try {\n"
    "    res = await fetch(`${BASE}${path}`, opts);\n"
    "  } catch (e) {\n"
    "    throw new Error('Cannot reach the server. Is the backend running on ' + BASE + '?');\n"
    "  }\n"
    "  return res.json();\n"
    "}\n"
    "export const api = {\n"
    "  list: (entity) => request('GET', `/${entity}/`),\n"
    "};\n"
)


def test_a_wrapper_with_variable_options_is_not_a_call(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.js", WRAPPER)
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []


def test_a_literal_path_after_an_interpolated_base_is_still_reported(tmp_path):
    _backend(tmp_path)
    _seed(tmp_path, "frontend/src/api.js", WRAPPER + "export const all = () => fetch(`${BASE}/books/`);\n")
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "line 13 calls GET /books/," in issues[0]


def test_a_router_prefix_is_part_of_the_route(tmp_path):
    _seed(tmp_path, "backend/routers/books.py", (
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix='/api')\n"
        '@router.get("/book/")\n'
        "def list_books(): ...\n"
    ))
    _seed(tmp_path, "frontend/src/api.js", (
        "export const all = () => fetch(`${BASE}/api/book/`);\n"
        "export const bad = () => fetch(`${BASE}/book/`);\n"
    ))
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "line 2 calls GET /book/," in issues[0]
