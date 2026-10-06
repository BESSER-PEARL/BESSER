"""Create forms the backend probes cannot see.

``validate_app``, ``test_api`` and the runbook only drive HTTP, so a UI whose
create forms never save scored as working. Replaying the 40 hill-climb apps
(spec-gen-luna baseline + v1) in a browser, 16 saved no record through any
form. The shapes below are taken from those apps; each rule has a negative
case from an app whose forms did save, and the blocker rules fired on none of
the 24 working apps.
"""

from __future__ import annotations

import json

import pytest

from besser.spec_driven_agent.validation.frontend_forms import collect_frontend_form_issues
from besser.spec_driven_agent.validation.issues import _classify_issue

_SCHEMAS = '''
from datetime import date
from typing import List, Optional
from pydantic import BaseModel

class ExpenseCreate(BaseModel):
    amount: float
    date: date
    description: str
    category: int

class CategoryCreate(BaseModel):
    name: str
    description: str

class PersonCreate(BaseModel):
    name: str

class EmployeeCreate(PersonCreate):
    hired: Optional[date] = None
    tags: Optional[List[int]] = None
'''

_API = (
    "const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';\n"
    "export async function api(path, options = {}) {\n"
    "  const r = await fetch(`${BASE}${path}`, options); return r.json(); }\n"
    "export const list = (path) => api(`${path}/`);\n"
    "export const create = (path, body) => api(`${path}/`, {method: 'POST', body: JSON.stringify(body)});\n"
)


def _app(tmp_path, files: dict[str, str]) -> str:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "pydantic_classes.py").write_text(_SCHEMAS, encoding="utf-8")
    front = tmp_path / "frontend"
    (front / "src").mkdir(parents=True)
    (front / "package.json").write_text(json.dumps({"dependencies": {"react": "^18"}}), encoding="utf-8")
    for rel, text in files.items():
        (front / "src" / rel).write_text(text, encoding="utf-8")
    return str(tmp_path)


def _findings(tmp_path, files, prefix):
    return [i for i in collect_frontend_form_issues(_app(tmp_path, files)) if i.startswith(prefix)]


# ---- T3b: BASE + path with resource paths lacking the leading "/" -------

def test_resource_path_without_leading_slash_is_reported(tmp_path):
    """baseline p5-expense-tracker rep1: requests went to
    http://localhost:8000expense/ and every page listed nothing."""
    [issue] = _findings(tmp_path, {
        "api.js": _API,
        "App.jsx": "const entities={Expense:{path:'expense'},Category:{path:'category'}};\n"
                   "export default function App(){return <div/>}\n",
    }, "api url:")
    assert "frontend/src/App.jsx line 1" in issue
    assert "'expense', 'category'" in issue
    assert "http://localhost:8000expense/" in issue


def test_resource_path_with_leading_slash_is_quiet(tmp_path):
    assert not _findings(tmp_path, {
        "api.js": _API,
        "App.jsx": "const entities={Expense:{path:'/expense'}};\n"
                   "export default function App(){return <div/>}\n",
    }, "api url:")


def test_slashless_path_is_quiet_when_the_caller_adds_the_slash(tmp_path):
    """p2-stock-tracker: ``api(`/${entity}/`)`` - the value never starts the URL."""
    api = _API.replace("api(`${path}/`)", "api(`/${path}/`)").replace(
        "api(`${path}/`, {", "api(`/${path}/`, {")
    assert not _findings(tmp_path, {
        "api.js": api,
        "App.jsx": "const entities={Expense:{path:'expense'}};\n"
                   "export default function App(){return <div/>}\n",
    }, "api url:")


# ---- T2a: Create shares the edit handler keyed on 'new' ------------------

def test_new_placeholder_that_selects_update_is_reported(tmp_path):
    """baseline p11-permits rep0: ``?edit=new`` then ``if(editId) update(...)``
    sent PUT /permit/new/."""
    [issue] = _findings(tmp_path, {"EntityList.jsx": (
        "export default function L(){const editId=params.get('edit');\n"
        "async function save(e){if(editId)await api.update(entity,editId,data);"
        "else await api.create(entity,data)}\n"
        "return <h2>{editId==='new'?'Create':'Edit'}</h2>}\n")}, "form create target:")
    assert "line 2" in issue and "PUT /<entity>/new/" in issue


def test_new_placeholder_compared_explicitly_is_quiet(tmp_path):
    """v1 p1-hotel-bookings rep0: ``if(editing==='new') create(...)``."""
    assert not _findings(tmp_path, {"EntityList.jsx": (
        "export default function L(){function start(row){setEditing(row?.id||'new')}\n"
        "async function submit(e){if(editing==='new')await create(entity,body);"
        "else await update(entity,editing,body)}return <form onSubmit={submit}/>}\n")},
        "form create target:")


# ---- T2b: a create link to an undeclared route ---------------------------

_ROUTES = ("<Routes><Route path=\"/\" element={<Home/>}/>"
           "{kinds.map(k=><Route key={k} path={'/'+k} element={<List kind={k}/>}/>)}</Routes>")


def test_create_link_to_undeclared_route_is_reported(tmp_path):
    """baseline p8-risk-awareness rep1: Create linked to /<kind>/new, which
    no route declares."""
    [issue] = _findings(tmp_path, {"App.jsx": (
        "function List({kind}){return <Link className=\"button\" to={`/${kind}/new`}>Create</Link>}\n"
        f"export default function App(){{return {_ROUTES}}}\n")}, "form route:")
    assert "'Create' link goes to /${...}/new" in issue


def test_create_link_to_declared_route_is_quiet(tmp_path):
    routes = _ROUTES.replace("</Routes>", "<Route path=\"/:kind/new\" element={<Form/>}/></Routes>")
    assert not _findings(tmp_path, {"App.jsx": (
        "function List({kind}){return <Link to={`/${kind}/new`}>Create</Link>}\n"
        f"export default function App(){{return {routes}}}\n")}, "form route:")


def test_unknown_route_paths_disable_the_route_rule(tmp_path):
    """A route path held in a variable could be anything; never guess."""
    assert not _findings(tmp_path, {"App.jsx": (
        "function List({kind}){return <Link to={`/${kind}/new`}>Create</Link>}\n"
        "export default function App(){return <Routes>{rs.map(r=><Route path={r.path}/>)}</Routes>}\n")},
        "form route:")


# ---- T2c: raw JSON textarea ---------------------------------------------

def test_json_textarea_form_is_reported(tmp_path):
    """baseline p1-hotel-bookings rep0: the only create form was a JSON box."""
    [issue] = _findings(tmp_path, {"App.jsx": (
        "export default function E(){async function submit(e){await save(entity,JSON.parse(json))}\n"
        "return <form onSubmit={submit}><textarea rows=\"3\" value={json} "
        "onChange={e=>setJson(e.target.value)}/></form>}\n")}, "form json textarea:")
    assert "line 2" in issue


def test_textarea_for_a_text_field_is_quiet(tmp_path):
    assert not _findings(tmp_path, {"App.jsx": (
        "export default function E(){return <form><textarea value={form.description} "
        "onChange={e=>set(e.target.value)}/></form>}\n")}, "form json textarea:")


# ---- T1: text inputs for typed schema fields -----------------------------

def test_typeless_generic_input_over_typed_fields_is_reported(tmp_path):
    """baseline hotel-spec-wrong-summary rep0: every field rendered as an
    <input> with no type, so 'capacity: int' got text and the API said 422."""
    [issue] = _findings(tmp_path, {"EntityList.jsx": (
        "const fields={expense:['amount','date','description','category'],category:['name','description']};\n"
        "export default function L({entity}){return <form>{fields[entity].map(f=>"
        "<input key={f} value={form[f]??''} onChange={e=>set(f,e.target.value)}/>)}</form>}\n")},
        "form field type:")
    assert "Expense form" in issue
    assert "amount (number), date (date), category (number)" in issue
    assert "Category" not in issue  # all-string schema: nothing to report


def test_inherited_typed_field_is_resolved(tmp_path):
    [issue] = _findings(tmp_path, {"L.jsx": (
        "const f=['name','hired'];\n"
        "export default function L(){return f.map(k=><input value={form[k]}/>)}\n")},
        "form field type:")
    assert "Employee form" in issue and "hired (date)" in issue


def test_named_text_input_for_a_number_field_is_reported(tmp_path):
    [issue] = _findings(tmp_path, {"Expense.jsx": (
        "export default function E(){return <form><input type=\"text\" value={form.amount}/></form>}\n")},
        "form field type:")
    assert "'amount'" in issue and 'type="number"' in issue


@pytest.mark.parametrize("control", [
    "<input type=\"number\" value={form.amount}/>",
    "<input type={t} value={form[k]}/>",          # typed per field: not decidable here
    "<input value={form.description}/>",          # a str field
    "<input value={form.name}/>",                 # str in every schema that has it
])
def test_typed_or_undecidable_controls_are_quiet(tmp_path, control):
    files = {"E.jsx": f"const f=['amount','date'];\nexport default function E(){{return <form>{control}</form>}}\n"}
    assert not _findings(tmp_path, files, "form field type:")


# ---- severities ----------------------------------------------------------

@pytest.mark.parametrize("prefix, severity", [
    ("api url:", "blocker"),
    ("form create target:", "blocker"),
    ("form route:", "blocker"),
    ("form json textarea:", "blocker"),
    ("form field type:", "warning"),
])
def test_finding_severities(prefix, severity):
    assert _classify_issue(f"{prefix} frontend/src/App.jsx line 1: x").severity == severity


def test_phase_3_surfaces_form_findings_with_their_severity(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    class _Client:
        model = "mock-model"
        max_tokens = 4096
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):  # pragma: no cover
            raise AssertionError("no LLM call expected")

    workspace = _app(tmp_path, {
        "api.js": _API,
        "App.jsx": "import React from 'react';\n"
                   "const entities={Expense:{path:'expense'}};\n"
                   "const fields=['amount','date','description','category'];\n"
                   "export default function App(){return fields.map(f=><input value={form[f]}/>)}\n",
    })
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Ledger", types={Class(name="Expense")}),
        output_dir=workspace, enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    issues = orch._collect_validation_issues()

    assert [i for i in issues if i.message.startswith("api url:") and i.severity == "blocker"]
    assert [i for i in issues if i.message.startswith("form field type:") and i.severity == "warning"]
