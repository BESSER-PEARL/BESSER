"""Frontend-contract signals count only where they are code.

A ``<Route path={HOME}>`` or ``<Navigate>`` inside a string, a template literal
or a comment silenced the blank-on-load blocker for a router that has no home
route, and ``document.createElement(`` in a button handler read as an API
write, so a form that cannot save passed as saved.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.frontend_contract import collect_frontend_contract_issues

APP = ("import { BrowserRouter, Routes, Route } from 'react-router-dom';\n"
       "export default function App() {\n"
       "  return <BrowserRouter><Routes><Route path=\"/books\" element={<Books />} /></Routes></BrowserRouter>;\n"
       "}\n")


def _issues(tmp_path, files: dict[str, str]) -> list[str]:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return collect_frontend_contract_issues(str(tmp_path))


def _blank_on_load(issues: list[str]) -> bool:
    return any("no home route" in issue for issue in issues)


@pytest.mark.parametrize("extra", [
    "export const SNIPPET = `<Route path={HOME} element={<Home />} />`;\n",
    "export const SNIPPET = '<Route index element={<Home />} />';\n",
    "export const SNIPPET = '<Route path=\"/\" />';\n",
    "export const DOC = \"use { path: '/' } for the home page\";\n",
    "// <Route path={ROUTES.HOME} element={<Home />} />\n",
    "/* <Navigate to=\"/books\" /> */\n",
    "const doc = '<Navigate to=\"/books\" />';\n",
])
def test_a_home_route_in_a_string_or_comment_does_not_silence_the_blocker(tmp_path, extra):
    assert _blank_on_load(_issues(tmp_path, {"frontend/src/App.jsx": APP + extra}))


@pytest.mark.parametrize("home", [
    "<Route path=\"/\" element={<Home />} />",
    "<Route index element={<Home />} />",
    "<Route path={ROUTES.HOME} element={<Home />} />",
    "<Route path=\"*\" element={<Navigate to=\"/books\" />} />",
])
def test_a_real_home_route_still_counts(tmp_path, home):
    app = APP.replace("</Routes>", home + "</Routes>") + "const note = <p>Don't forget the books</p>;\n"
    assert not _blank_on_load(_issues(tmp_path, {"frontend/src/App.jsx": app}))


def test_a_data_router_home_route_still_counts(tmp_path):
    app = ("import { createBrowserRouter } from 'react-router-dom';\n"
           "export const router = createBrowserRouter([{ path: \"/\", element: <Home /> }]);\n")
    assert not _blank_on_load(_issues(tmp_path, {"frontend/src/router.js": app}))


def test_a_route_only_inside_a_template_literal_is_not_a_router(tmp_path):
    """No real router: a template that mentions ``<Route`` must not raise blank-on-load."""
    page = "export const SNIPPET = `<Route path=\"/books\" />`;\nexport default () => <h1>Books</h1>;\n"
    assert not _blank_on_load(_issues(tmp_path, {"frontend/src/Help.jsx": page}))


_DEAD_FORM = ("export default function F() {\n"
              "  const save = () => { BODY };\n"
              "  return <form onSubmit={(e) => { e.preventDefault(); }}>"
              "<button onClick={save}>Save</button></form>;\n}\n")


def _dead_form(issues: list[str]) -> bool:
    return any("onSubmit does nothing" in issue for issue in issues)


@pytest.mark.parametrize("body", [
    "const a = document.createElement('a'); a.click();",
    "const url = URL.createObjectURL(blob); window.open(url);",
    "const copy = Object.create(proto);",
])
def test_a_dom_or_builtin_create_call_is_not_a_save(tmp_path, body):
    assert _dead_form(_issues(tmp_path, {"frontend/src/F.jsx": _DEAD_FORM.replace("BODY", body)}))


@pytest.mark.parametrize("body", [
    "api.createBook(form);",
    "bookService.update(id, form);",
    "fetch('/book/', { method: 'POST', body });",
])
def test_an_api_write_in_the_button_handler_still_saves(tmp_path, body):
    assert not _dead_form(_issues(tmp_path, {"frontend/src/F.jsx": _DEAD_FORM.replace("BODY", body)}))
