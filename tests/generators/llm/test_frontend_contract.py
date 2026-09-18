"""Tests for the Phase-3 frontend-contract validator.

The check is intentionally conservative: it flags only two correctness
defects that leave an LLM-authored React frontend visibly broken --
blank-on-load (router with no home route) and a form that cannot save
(no-op onSubmit, or forms with no HTTP write calls anywhere). Scope
choices (delete button, nav, styling) are NEVER enforced here.

Precision matters more than recall: a false blocker would burn auto-fix
turns and can mark a good run incomplete. These tests pin down the
no-false-positive cases (good CRUD app, router-less single-page app,
real named handlers) as tightly as the true-positive cases.
"""
import os
import types

from besser.generators.llm.orchestrator import LLMOrchestrator, _classify_issue


def _run(tmp_path, files: dict) -> list[str]:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    shim = types.SimpleNamespace(output_dir=str(tmp_path))
    return LLMOrchestrator._collect_frontend_contract_issues(shim)


# --------------------------------------------------------------------------- #
# No false positives
# --------------------------------------------------------------------------- #
def test_good_crud_frontend_is_clean(tmp_path):
    files = {
        "frontend/src/App.js": (
            'import React from "react";\n'
            'import { Routes, Route, Navigate } from "react-router-dom";\n'
            "import AuthorList from './AuthorList';\n"
            "function App(){ return (<Routes>\n"
            '  <Route path="/" element={<AuthorList/>} />\n'
            '  <Route path="/authors" element={<AuthorList/>} />\n'
            "</Routes>);}\n"
            "export default App;\n"
        ),
        "frontend/src/AuthorForm.js": (
            'import React from "react";\n'
            "export default function AuthorForm(){\n"
            "  const onSubmit = async (e) => { e.preventDefault();\n"
            '    await fetch("/authors", {method:"POST", body: JSON.stringify(x)}); load(); };\n'
            "  return (<form onSubmit={onSubmit}><input/></form>);\n"
            "}\n"
        ),
    }
    assert _run(tmp_path, files) == []


def test_single_page_app_without_router_is_not_flagged(tmp_path):
    # No <Route> / react-router => blank-on-load is not a routing concern.
    files = {
        "frontend/src/App.js": (
            'import React from "react";\n'
            "function App(){ const s = async (e) => { e.preventDefault();\n"
            '    await fetch("/a", {method:"POST"}); };\n'
            "  return (<div><form onSubmit={s}><input/></form></div>); }\n"
            "export default App;\n"
        )
    }
    assert _run(tmp_path, files) == []


def test_named_handler_is_not_flagged_as_dead(tmp_path):
    files = {
        "frontend/src/App.js": (
            'import {Routes,Route,Navigate} from "react-router-dom";\n'
            'function App(){return <Routes><Route path="/" element={<X/>}/></Routes>;}\n'
            "function F(){ const s = async (e) => { e.preventDefault();\n"
            '   await fetch("/x",{method:"PUT"}); }; return <form onSubmit={s}/>; }\n'
        )
    }
    assert _run(tmp_path, files) == []


def test_non_react_js_is_ignored(tmp_path):
    # A stray JS config with no react / no JSX must not be treated as frontend.
    files = {"scripts/build.js": "module.exports = { port: 3000 };\n"}
    assert _run(tmp_path, files) == []


# --------------------------------------------------------------------------- #
# True positives
# --------------------------------------------------------------------------- #
def test_blank_on_load_router_without_home_route(tmp_path):
    files = {
        "frontend/src/App.js": (
            'import { Routes, Route } from "react-router-dom";\n'
            "function App(){ return (<Routes>\n"
            '  <Route path="/authors" element={<A/>} />\n'
            '  <Route path="/books" element={<B/>} />\n'
            "</Routes>);}\n"
        )
    }
    issues = _run(tmp_path, files)
    assert len(issues) == 1
    assert issues[0].startswith("frontend contract:")
    assert "blank on first load" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_home_route_via_navigate_redirect_is_ok(tmp_path):
    files = {
        "frontend/src/App.js": (
            'import { Routes, Route, Navigate } from "react-router-dom";\n'
            "function App(){ return (<Routes>\n"
            '  <Route path="/authors" element={<A/>} />\n'
            '  <Route path="/" element={<Navigate to="/authors" replace/>} />\n'
            "</Routes>);}\n"
        )
    }
    assert _run(tmp_path, files) == []


def test_dead_form_noop_onsubmit(tmp_path):
    # A sibling issues a real POST, so check 2b (no writes at all) stays silent
    # and we isolate the no-op-handler check 2a.
    files = {
        "frontend/src/api.js": (
            'export const create = (x) => fetch("/authors", {method:"POST", body: x});\n'
        ),
        "frontend/src/Form.js": (
            'import React from "react";\n'
            "export default function Form(){ return <form onSubmit={() => {}}><input/></form>; }\n"
        ),
    }
    issues = _run(tmp_path, files)
    assert len(issues) == 1
    assert "onSubmit does" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_dead_form_prevent_default_only(tmp_path):
    files = {
        "frontend/src/Form.js": (
            "export default function Form(){\n"
            "  return <form onSubmit={(e) => e.preventDefault()}><input/></form>; }\n"
        )
    }
    issues = _run(tmp_path, files)
    assert any("onSubmit does" in i for i in issues)


def test_forms_present_but_no_http_write(tmp_path):
    files = {
        "frontend/src/App.js": (
            'import {Routes,Route,Navigate} from "react-router-dom";\n'
            'function App(){return <Routes><Route path="/" element={<X/>}/></Routes>;}\n'
            "function F(){ const s = (e) => { e.preventDefault(); setName(''); };\n"
            "  return <form onSubmit={s}><input/></form>; }\n"
        )
    }
    issues = _run(tmp_path, files)
    assert any("no HTTP" in i for i in issues)
    assert all(_classify_issue(i).severity == "blocker" for i in issues)


def test_classify_frontend_contract_prefix_is_blocker():
    v = _classify_issue("frontend contract: something is broken on load.")
    assert v.severity == "blocker"
