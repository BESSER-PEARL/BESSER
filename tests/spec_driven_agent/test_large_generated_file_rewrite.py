"""A full rewrite of a large generated file is judged by what it keeps.

Over 200 lines, ``write_file`` on a generator file was refused until two
``modify_file`` attempts had been made, whatever the content: a page the
request asked to redesign was generated in full, thrown away, and generated
again after two edits, and any two edit calls, refused ones included, then
let a rewrite drop code. A rewrite that keeps its code is now accepted at once,
and one that loses code is refused however many edits came before: one that
shrinks the file or abbreviates it, drops a top-level declaration, or has
fewer default exports, routes, re-exports, decorators or module-level calls.
"""
import json
import os
import re

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

ROW_LINES = [f'      <p className="row-{n}">Order {n}</p>\n' for n in range(220)]
ROWS = "".join(ROW_LINES)
PAGE = ('import React from "react";\n\nconst OrderList: React.FC = () => {\n  return (\n    <div>\n'
        + ROWS + "    </div>\n  );\n};\n\nexport default OrderList;\n")
APP = ('import { Routes, Route } from "react-router-dom";\nimport OrderList from "./pages/OrderList";\n\n'
       "function App() {\n  return (\n    <Routes>\n"
       + "".join(f'      <Route path="/p{n}" element={{<OrderList />}} />\n' for n in range(210))
       + "    </Routes>\n  );\n}\n\nexport default App;\n")
ROUTES = "".join(
    f'@app.get("/order{n}/")\ndef get_order_{n}():\n    x = {n}\n    return {{"n": x}}\n\n\n' for n in range(40))
MAIN = ("from fastapi import FastAPI\nfrom routers import order_router\n\napp = FastAPI()\n\n"
        "app.add_middleware(object)\napp.include_router(order_router.router)\n\n" + ROUTES
        + 'if __name__ == "__main__":\n    run(app)\n')
ORM = "Base = declarative_base()\n\n" + "".join(
    f"class Model{n}(Base):\n    __tablename__ = 'm{n}'\n    a = 1\n    b = 2\n\n\n" for n in range(40))
BARREL = "".join(f'export {{ Thing{n} }} from "./thing{n}";\n' for n in range(210))

PAGE_PATH = "web_app/frontend/src/pages/OrderList.tsx"


def _generated(tmp_path, rel, content):
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(content)
    executor = ToolExecutor(workspace=str(tmp_path))
    executor._generator_files.add(rel)
    executor._read_file({"path": rel})
    return executor


def _write(tmp_path, rel, before, after):
    executor = _generated(tmp_path, rel, before)
    return json.loads(executor.execute("write_file", {"path": rel, "content": after}))


def test_a_redesigned_page_that_keeps_its_code_is_accepted_at_once(tmp_path):
    redesigned = PAGE.replace('import React from "react";', 'import React, { useState } from "react";').replace(
        "  return (\n", "  const [page] = useState(0);\n  return (\n").replace("<p className", "<li className")
    redesigned = redesigned.replace("</p>", "</li>")
    assert _write(tmp_path, PAGE_PATH, PAGE, redesigned).get("status") == "written"


def test_a_page_rewrite_dropping_a_declaration_is_refused_and_names_it(tmp_path):
    before = PAGE.replace("export default OrderList;", "export const OrderBadge = () => null;\n\nexport default OrderList;")
    refused = _write(tmp_path, PAGE_PATH, before, PAGE)
    assert "OrderBadge" in refused.get("error", ""), refused


ELIDED_PAGE = PAGE.replace(ROWS, "      {/* ... existing rows unchanged ... */}\n" + "".join(ROW_LINES[:180]))
NO_DEFAULT = PAGE.replace("\nexport default OrderList;\n", "\nexport { OrderList };\n")
IN_COMMENT = PAGE.replace("const OrderList: React.FC", "/*\nconst OrderList = old\n*/\nconst Other: React.FC").replace(
    "export default OrderList;", "export default Other;")
PADDED_STUB = 'import React from "react";\n\nconst OrderList: React.FC = () => null;\n' + "\n" * 240 + \
    "export default OrderList;\n"
PLACEHOLDER = PAGE.replace("".join(ROW_LINES[150:]), "      {/* remaining rows unchanged */}\n")


@pytest.mark.parametrize("rel,before,after", [
    pytest.param(PAGE_PATH, PAGE, PAGE.replace(ROWS, "".join(ROW_LINES[:70])), id="shrinks-below-60-percent"),
    pytest.param(PAGE_PATH, PAGE, PADDED_STUB, id="gutted-and-padded-with-blank-lines"),
    pytest.param(PAGE_PATH, PAGE, PLACEHOLDER, id="comment-placeholder-without-dots"),
    pytest.param(PAGE_PATH, PAGE, ELIDED_PAGE, id="jsx-elided-to-a-comment"),
    pytest.param(PAGE_PATH, PAGE, NO_DEFAULT, id="export-default-dropped"),
    pytest.param(PAGE_PATH, PAGE, IN_COMMENT, id="name-kept-only-in-a-comment"),
    pytest.param("web_app/frontend/src/App.tsx", APP,
                 APP.replace('      <Route path="/p7" element={<OrderList />} />\n', ""), id="a-route-dropped"),
    pytest.param("web_app/frontend/src/index.ts", BARREL, BARREL.replace('export { Thing3 } from "./thing3";\n', ""),
                 id="barrel-re-export-dropped"),
    pytest.param("web_app/backend/main_api.py", MAIN, MAIN.replace("app.include_router(order_router.router)\n", ""),
                 id="py-include-router-dropped"),
    pytest.param("web_app/backend/main_api.py", MAIN, MAIN.replace("@app.get(", "# @app.get("),
                 id="py-route-decorators-commented-out"),
    pytest.param("web_app/backend/main_api.py", MAIN,
                 re.sub(r"(def get_order_\d+\(\):\n)    x = \d+\n", r"\1    ...  # unchanged\n", MAIN),
                 id="py-bodies-elided"),
    pytest.param("web_app/backend/sql_alchemy.py", ORM,
                 re.sub(r"(class Model\d+\(Base\):\n)(    .*\n)+", r"\1    pass\n", ORM), id="py-orm-classes-stubbed"),
])
def test_a_rewrite_that_loses_scaffold_is_refused(tmp_path, rel, before, after):
    executor = _generated(tmp_path, rel, before)
    for junk in ({"old_text": "  ", "new_text": "x"}, {"old_text": "zz_absent", "new_text": "y"}):
        executor.execute("modify_file", {"path": rel, **junk})
    refused = json.loads(executor.execute("write_file", {"path": rel, "content": after}))
    assert refused.get("rejection_kind") == "rewrite_loss", refused
    with open(os.path.join(str(tmp_path), rel), encoding="utf-8") as fh:
        assert fh.read() == before


def test_a_rewrite_of_a_file_with_no_declarations_that_keeps_its_lines_is_accepted(tmp_path):
    rel = "web_app/frontend/src/styles/theme.css"
    css = "".join(f".row-{n} {{ color: red; }}\n" for n in range(250))
    assert _write(tmp_path, rel, css, css.replace("red", "blue")).get("status") == "written"


def test_a_rewrite_that_keeps_a_backend_module_is_accepted(tmp_path):
    rewritten = MAIN.replace('return {"n": x}', 'return {"n": x, "ok": True}')
    assert _write(tmp_path, "web_app/backend/main_api.py", MAIN, rewritten).get("status") == "written"


def test_declaration_names_are_not_keywords():
    from besser.spec_driven_agent.agent.tool_executor import _TOP_LEVEL_DECL_RE

    source = "export default class extends Base {}\nexport const enum Status { A }\nexport async function load() {}\n"
    assert _TOP_LEVEL_DECL_RE.findall(source) == ["Status", "load"]


def test_a_new_comment_with_an_omission_word_in_a_rewrite_that_keeps_every_line_is_accepted(tmp_path):
    commented = PAGE.replace("  return (\n", "  // Count the remaining rows before showing the footer\n  return (\n")
    assert _write(tmp_path, PAGE_PATH, PAGE, commented).get("status") == "written"


def test_a_placeholder_refusal_says_how_many_lines_it_replaces(tmp_path):
    error = _write(tmp_path, PAGE_PATH, PAGE, PLACEHOLDER).get("error", "")
    assert "replaces 70 lines" in error and "remaining rows unchanged" in error, error


NEW_SECTION = "".join(f'      <label className="filter-{n}">Filter {n}</label>\n' for n in range(60))
PLACEHOLDER_TAIL = "      {/* remaining rows unchanged */}\n"


@pytest.mark.parametrize("filler", [
    pytest.param(NEW_SECTION, id="net-longer-with-a-new-section"),
    pytest.param("".join(f"      {{/* note {n} */}}\n" for n in range(50)), id="padded-with-comments"),
])
def test_a_placeholder_that_replaces_rows_is_refused_however_long_the_rewrite(tmp_path, filler):
    after = PAGE.replace("".join(ROW_LINES[170:]), PLACEHOLDER_TAIL + filler)
    error = _write(tmp_path, PAGE_PATH, PAGE, after).get("error", "")
    assert "replaces 50 lines" in error and "remaining rows unchanged" in error, error


def test_a_redesign_with_a_new_comment_using_an_omission_word_is_accepted(tmp_path):
    redesigned = PAGE.replace("  return (\n", "  // Count the remaining rows before showing the footer\n  return (\n")
    redesigned = redesigned.replace("<p className", "<li className").replace("</p>", "</li>")
    assert _write(tmp_path, PAGE_PATH, PAGE, redesigned).get("status") == "written"


SECTION = "".join(f'      <div className="card">\n        <span>{{order.total}}</span>\n      </div>\n' for _ in range(7))
SECTIONS = PAGE.replace("    </div>\n  );", "    </div>\n" + "".join(
    f'    <section className="s{n}">\n{SECTION}    </section>\n' for n in range(3)) + "  );")


def test_a_placeholder_for_a_repeated_section_is_refused(tmp_path):
    """The replaced lines still appear in the other sections, but fewer times than before."""
    after = SECTIONS.replace('    <section className="s2">\n' + SECTION,
                             '    <section className="s2">\n      {/* cards same as above */}\n')
    error = _write(tmp_path, PAGE_PATH, SECTIONS, after).get("error", "")
    assert "placeholder comment" in error and "cards same as above" in error, error


def test_a_placeholder_added_apart_from_the_rows_it_drops_is_refused(tmp_path):
    after = PAGE.replace("    <div>\n", "    <div>\n" + PLACEHOLDER_TAIL, 1).replace("".join(ROW_LINES[170:]), "")
    error = _write(tmp_path, PAGE_PATH, PAGE, after).get("error", "")
    assert "removes 50 lines" in error and "remaining rows unchanged" in error, error
