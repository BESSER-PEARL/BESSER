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
import json
import os
import types

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue
from besser.spec_driven_agent.validation.frontend_schema import (
    collect_frontend_schema_diagnostics,
    collect_frontend_schema_issues,
)


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


def _schema_form(tmp_path, *, schema=None, columns=None, prefix=""):
    schema = schema or (
        "from pydantic import BaseModel, Field\n"
        "class Identity(BaseModel):\n    reference: str\n"
        "class RentalInput(Identity):\n"
        "    name: str = Field(alias='displayName')\n    owner: int\n"
    )
    columns = columns if columns is not None else [
        {"field": "reference", "required": True},
        {"field": "displayName", "required": True},
        {"field": "ownerLabel", "path": "owner", "column_type": "lookup", "required": True},
    ]
    options = {"actionButtons": True, "columns": [{"field": "computedTotal", "required": True}], "formColumns": columns}
    binding = {"entity": "Rental", "endpoint": prefix + "/rental/"}
    files = {
        "backend/models.py": schema + "\nraise RuntimeError('generated code must not be imported')\n",
        "backend/routes.py": (
            "from fastapi import APIRouter\nfrom models import RentalInput as Payload\n"
            f"router = APIRouter(prefix={prefix!r})\n"
            "@router.post('/rental/')\ndef create(payload: Payload): pass\n"
            "@router.put('/rental/{reference}/')\ndef update(reference: str, payload: Payload): pass\n"
        ),
        "frontend/src/Rental.tsx": (
            "export default () => <TableBlock options={" + json.dumps(options)
            + "} dataBinding={" + json.dumps(binding) + "} />;\n"
        ),
    }
    for relative, content in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return files


def test_form_schema_flags_writable_removed_fields_but_not_display_columns(tmp_path):
    _schema_form(tmp_path, columns=[
        {"field": "reference", "required": True},
        {"field": "computedTotal", "required": True},
        {"field": "settled", "required": False, "readOnly": True},
    ])
    findings = collect_frontend_schema_diagnostics(tmp_path)
    assert len(findings) == 2
    assert all(f["path"] == "frontend/src/Rental.tsx" and f["line"] == 1 for f in findings)
    assert "required editable field 'computedTotal'" in findings[0]["message"]
    # TableBlock ignores this flag: it must not conceal a still-editable field.
    assert "editable field 'settled'" in findings[1]["message"]
    assert all("create (RentalInput" in f["message"] and "update (RentalInput" in f["message"] for f in findings)
    assert all(_classify_issue(issue).severity == "blocker" for issue in collect_frontend_schema_issues(tmp_path))


def test_form_schema_preserves_inherited_alias_lookup_and_display_contracts(tmp_path):
    _schema_form(tmp_path, prefix="/api")
    assert collect_frontend_schema_diagnostics(tmp_path) == []


@pytest.mark.parametrize("schema", [
    "from pydantic import BaseModel, ConfigDict\nclass RentalInput(BaseModel):\n    model_config = ConfigDict(extra='allow')\n",
    "from pydantic import BaseModel\nclass RentalInput(BaseModel):\n    model_config = {'extra': 'allow'}\n",
    "from pydantic import BaseModel\nclass RentalInput(BaseModel):\n    class Config:\n        extra = 'allow'\n",
    "from pydantic import BaseModel, model_validator\nclass RentalInput(BaseModel):\n    @model_validator(mode='before')\n    def legacy(cls, value): return value\n",
    "from external_package import RequestBase\nclass RentalInput(RequestBase):\n    reference: str\n",
    "from pydantic import BaseModel, ConfigDict\nclass RentalInput(BaseModel):\n    model_config = ConfigDict(alias_generator=make_alias)\n",
])
def test_form_schema_does_not_guess_dynamic_or_extensible_requests(tmp_path, schema):
    _schema_form(tmp_path, schema=schema)
    assert collect_frontend_schema_diagnostics(tmp_path) == []


def test_form_schema_schema_write_checks_consumers_and_frontend_overlay_can_repair(tmp_path):
    files = _schema_form(tmp_path)
    removed = files["backend/models.py"].replace("    owner: int\n", "")
    findings = collect_frontend_schema_diagnostics(tmp_path, "backend/models.py", removed)
    assert len(findings) == 1 and "field 'owner'" in findings[0]["message"]
    assert collect_frontend_schema_diagnostics(tmp_path) == []  # overlay never mutates sources
    (tmp_path / "backend/models.py").write_text(removed, encoding="utf-8")
    repaired = files["frontend/src/Rental.tsx"].replace(
        ', {"field": "ownerLabel", "path": "owner", "column_type": "lookup", "required": true}', ""
    )
    assert collect_frontend_schema_diagnostics(tmp_path, "frontend/src/Rental.tsx", repaired) == []


def test_form_schema_does_not_confuse_multiple_bindings_or_inactive_forms(tmp_path):
    files = _schema_form(tmp_path)
    page = tmp_path / "frontend/src/Rental.tsx"
    # Valid other table; same unknown field is harmless on display-only tables.
    inactive = files["frontend/src/Rental.tsx"].replace('"actionButtons": true', '"actionButtons": false').replace('"reference"', '"unknown"')
    page.write_text(files["frontend/src/Rental.tsx"] + inactive, encoding="utf-8")
    assert collect_frontend_schema_diagnostics(tmp_path) == []
    page.write_text(files["frontend/src/Rental.tsx"] + inactive.replace('"actionButtons": false', '"actionButtons": true'), encoding="utf-8")
    findings = collect_frontend_schema_diagnostics(tmp_path)
    assert len(findings) == 1 and findings[0]["line"] == 2


@pytest.mark.parametrize("form_metadata", ["missing", "empty", "null", "filtered_empty", "snake_case"])
def test_form_schema_matches_runtime_empty_form_fallback_and_metadata_alias(tmp_path, form_metadata):
    _schema_form(tmp_path)
    options = {"action-buttons": True, "columns": ["computedTotal"]}
    if form_metadata == "empty":
        options["formColumns"] = []
    elif form_metadata == "null":
        options["formColumns"] = None
    elif form_metadata == "filtered_empty":
        options["formColumns"] = [None, ""]
    elif form_metadata == "snake_case":
        options["form_columns"] = ["reference", "computedTotal"]
    source = '<TableBlock options={' + json.dumps(options) + '} dataBinding={{"endpoint": "/rental/"}} />'
    findings = collect_frontend_schema_diagnostics(tmp_path, "frontend/src/Rental.tsx", source)
    assert len(findings) == 1 and "field 'computedTotal'" in findings[0]["message"]
    # Omitting actionButtons uses the renderer's false default, not a form.
    source = source.replace('"action-buttons": true, ', "")
    assert collect_frontend_schema_diagnostics(tmp_path, "frontend/src/Rental.tsx", source) == []


def test_form_schema_ignores_commented_or_quoted_example_components(tmp_path):
    files = _schema_form(tmp_path)
    example = files["frontend/src/Rental.tsx"].replace('"reference"', '"unknown"')
    for wrapped in ("/* " + example + " */", "// " + example, "const example = `" + example + "`;"):
        source = wrapped + "\n" + files["frontend/src/Rental.tsx"]
        assert collect_frontend_schema_diagnostics(tmp_path, "frontend/src/Rental.tsx", source) == []
