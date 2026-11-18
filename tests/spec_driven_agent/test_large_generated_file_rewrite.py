"""A full rewrite of a large generated file is judged by what it keeps, not by a ritual.

Over 200 lines, ``write_file`` on a generator file was refused until two
``modify_file`` attempts had been made, whatever the content: a page the
request asked to redesign was generated in full, thrown away, and generated
again after two edits - which then unlocked even a rewrite dropping
everything. The risk the rule stands for is lost scaffold code (classes,
routes, components); a rewrite that keeps every top-level definition is
accepted at once, and one that drops some is refused naming them.
"""
import json
import os

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

ROUTER = "".join(
    f"@router.get('/order/{n}')\ndef read_order_{n}():\n    return {{'n': {n}}}\n\n\n"
    for n in range(45)
)
PAGE = (
    'import React from "react";\n\n'
    "const OrderList: React.FC = () => {\n  return (\n    <div>\n"
    + "".join(f'      <p className="row-{n}">Order {n}</p>\n' for n in range(220))
    + "    </div>\n  );\n};\n\nexport default OrderList;\n"
)


def _generated(tmp_path, rel, content):
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(content)
    executor = ToolExecutor(workspace=str(tmp_path))
    executor._generator_files.add(rel)
    executor._read_file({"path": rel})
    return executor


def _write(executor, rel, content):
    return json.loads(executor.execute("write_file", {"path": rel, "content": content}))


def test_a_rewrite_keeping_every_definition_is_accepted_at_once(tmp_path):
    rel = "backend/routers/order.py"
    executor = _generated(tmp_path, rel, "from fastapi import APIRouter\nrouter = APIRouter()\n\n" + ROUTER)
    rewrite = ("from fastapi import APIRouter, HTTPException\nrouter = APIRouter()\n\n"
               + ROUTER.replace("return {'n': 3}", "raise HTTPException(404)"))
    assert _write(executor, rel, rewrite).get("status") == "written"


def test_a_redesigned_page_that_keeps_its_component_is_accepted(tmp_path):
    rel = "frontend/src/pages/OrderList.tsx"
    executor = _generated(tmp_path, rel, PAGE)
    redesigned = ('import React, { useState } from "react";\n\n'
                  "const OrderList: React.FC = () => {\n  const [page] = useState(0);\n"
                  "  return <section>{page}</section>;\n};\n\nexport default OrderList;\n")
    assert _write(executor, rel, redesigned).get("status") == "written"


def test_a_rewrite_dropping_definitions_is_refused_and_names_them(tmp_path):
    rel = "backend/routers/order.py"
    executor = _generated(tmp_path, rel, "from fastapi import APIRouter\nrouter = APIRouter()\n\n" + ROUTER)
    truncated = "from fastapi import APIRouter\nrouter = APIRouter()\n\n" + ROUTER.split("@router.get('/order/40')")[0]
    refused = _write(executor, rel, truncated)
    assert "error" in refused
    for n in range(40, 45):
        assert f"read_order_{n}" in refused["error"]
    assert "read_order_39" not in refused["error"]


def test_a_file_whose_definitions_cannot_be_listed_keeps_the_old_rule(tmp_path):
    rel = "frontend/src/styles/theme.css"
    executor = _generated(tmp_path, rel, "".join(f".row-{n} {{ color: red; }}\n" for n in range(250)))
    refused = _write(executor, rel, ".row-0 { color: blue; }\n")
    assert "after two modify_file attempts" in refused.get("error", ""), refused
