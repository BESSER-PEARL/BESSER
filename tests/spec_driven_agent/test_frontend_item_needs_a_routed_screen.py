"""The frontend checklist item needs an app that routes to a screen, not entity metadata.

Live runs r10-p81 and r11-p81: the item closed at the first write turn, with only
``frontend/package.json`` and ``frontend/src/entities.js`` on disk. That module names
every entity, so the acceptance matrix marked each class ``page: True`` before any screen
existed. The verifier now also requires ``frontend_contract``'s home-route signal: a
route that is an index route, serves "/" or has a computed path.
"""
from unittest import mock

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import Target
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

ENTITIES = ("export const ENTITIES = [\n  { key: 'author', label: 'Authors' },\n"
            "  { key: 'book', label: 'Books' },\n  { key: 'library', label: 'Libraries' },\n];\n")
MAIN = ("import React from 'react';\nimport ReactDOM from 'react-dom/client';\n"
        "import { BrowserRouter } from 'react-router-dom';\nimport App from './App.jsx';\n"
        "ReactDOM.createRoot(document.getElementById('root')).render(<BrowserRouter><App /></BrowserRouter>);\n")
APP = ("import React from 'react';\nimport { Routes, Route } from 'react-router-dom';\n"
       "import BookList from './pages/BookList.jsx';\n"
       "export default function App() {\n  return (<Routes><Route path=\"/\" element={<BookList />} /></Routes>);\n}\n")
PAGE = "import React from 'react';\nexport default function BookList() { return <h1>Books</h1>; }\n"


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _verify(tmp_path):
    generate_backend(tmp_path, shelf_model(author=True))
    orch = LLMOrchestrator(llm_client=mock.MagicMock(), domain_model=shelf_model(author=True),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    orch._generator_used = GENERATOR
    orch.target = Target(kind="web_app", description="library web app", has_user_interface=True)
    return orch, orch._frontend_checklist_task()["verify"]


def test_package_json_and_entities_js_do_not_close_the_item(tmp_path):
    orch, verify = _verify(tmp_path)
    _write(tmp_path, "frontend/package.json", '{"name": "library"}')
    _write(tmp_path, "frontend/src/entities.js", ENTITIES)
    assert all(orch._routed_entity_pages().values()), "the matrix alone reads every class as a page"
    assert not verify()
    _write(tmp_path, "frontend/src/main.jsx", MAIN)  # a router with no route yet
    assert not verify()


def test_a_routed_entity_screen_closes_it(tmp_path):
    _, verify = _verify(tmp_path)
    _write(tmp_path, "frontend/package.json", '{"name": "library"}')
    _write(tmp_path, "frontend/src/main.jsx", MAIN)
    _write(tmp_path, "frontend/src/App.jsx", APP)
    _write(tmp_path, "frontend/src/pages/BookList.jsx", PAGE)
    assert verify()


def test_a_route_inside_a_comment_or_string_does_not_count(tmp_path):
    _, verify = _verify(tmp_path)
    _write(tmp_path, "frontend/package.json", '{"name": "library"}')
    _write(tmp_path, "frontend/src/entities.js", ENTITIES + '// <Route path="/" element={<Home />} />\n'
                                                            'export const HINT = \'<Route index />\';\n')
    assert not verify()
