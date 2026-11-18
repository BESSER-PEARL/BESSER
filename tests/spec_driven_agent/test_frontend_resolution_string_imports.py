"""A ``from`` or ``import`` inside a string or JSX text is not an import.

Live run r9-p111, ``frontend/src/entities.js:52``::

    { label: 'Valid until – from', field: 'validUntil', kind: 'dateFrom' }

was read as an import of the package ``', field: '``. A string that ends in
"from" followed by a "." (``'Valid from'.toUpperCase()``) or JSX text such as
``Data loaded from './seed/permits.json'`` became a missing-file
``frontend contract:`` blocker that no edit to the import graph can clear.
"""

from __future__ import annotations

import json

import pytest

from besser.spec_driven_agent.validation.frontend_resolution import (
    collect_frontend_resolution_issues,
)

_MANIFEST = {"dependencies": {"react": "^18.3.1"},
             "devDependencies": {"vite": "^5.4.8", "@vitejs/plugin-react": "^4.3.1"}}
_VITE_CONFIG = ("import { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n"
                "export default defineConfig({ plugins: [react()] });\n")


def _issues(tmp_path, files: dict[str, str]) -> list[str]:
    project = tmp_path / "frontend"
    (project / "src").mkdir(parents=True)
    (project / "package.json").write_text(json.dumps(_MANIFEST), encoding="utf-8")
    (project / "vite.config.js").write_text(_VITE_CONFIG, encoding="utf-8")
    for rel, text in files.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return collect_frontend_resolution_issues(str(tmp_path))


def test_the_r9_p111_label_is_not_a_package_import(tmp_path):
    source = ("export const FIELDS = [\n"
              "  { label: 'Valid until – from', field: 'validUntil', kind: 'dateFrom' },\n"
              "  { label: 'Valid until – to', field: 'validUntil', kind: 'dateTo' },\n"
              "];\n")
    assert _issues(tmp_path, {"src/entities.js": source}) == []


@pytest.mark.parametrize("rel,source", [
    # These two were missing-file blockers.
    ("src/labels.js", "export const L = { a: 'Valid from'.toUpperCase(), b: 'x' };\n"),
    ("src/Seed.jsx", "export default function A() {\n"
                     "  return <p>Data loaded from './seed/permits.json' at start</p>;\n}\n"),
    ("src/Seed.js", "export default function A() {\n"
                    "  return <p>Data loaded from './seed/permits.json' at start</p>;\n}\n"),
    ("src/copy.js", "export const L = [{ label: 'Copy from', path: './templates' }];\n"),
    ("src/help.js", "export const H = { help: 'Bulk import ', key: 'k' };\n"),
    ("src/template.js", "export const t = (x) => `from '${x}'`;\n"),
    ("src/About.jsx", "export default function A() {\n"
                      "  return <p>Imported from 'Digichecks' ontology</p>;\n}\n"),
])
def test_text_inside_strings_and_jsx_is_not_an_import(tmp_path, rel, source):
    assert _issues(tmp_path, {rel: source}) == []


_SHAPES = {
    "multi-line clause": ("src/App.jsx", "import React, {\n  useState,\n  useEffect as ue,\n}\n"
                                         "  from './Gone';\nexport default React;\n"),
    "export star": ("src/index.js", "export * from './Gone';\n"),
    "export star as": ("src/index.js", "export * as gone from './Gone';\n"),
    "export named": ("src/index.js", "export { default as Gone, type X } from './Gone';\n"),
    "import type": ("src/types.ts", "import type { X } from './Gone';\nexport type Y = X;\n"),
    "side effect": ("src/main.js", "import './Gone.css';\n"),
    "dynamic import": ("src/App.jsx", "const Page = lazy(() => import(/* chunk */ './Gone'));\n"),
    "require": ("src/legacy.cjs", "const gone = require('./Gone');\nmodule.exports = gone;\n"),
    "query suffix": ("src/Icon.js", "import raw from './Gone.svg?raw';\nexport default raw;\n"),
    "after a statement": ("src/main.js", "'use client'; import x from './Gone';\n"),
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_every_real_import_shape_is_still_read(tmp_path, shape):
    rel, source = _SHAPES[shape]
    issues = _issues(tmp_path, {rel: source})
    assert len(issues) == 1 and issues[0].startswith("frontend contract:"), issues
    assert "imports './Gone" in issues[0]


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_every_real_import_shape_resolves_when_the_file_exists(tmp_path, shape):
    rel, source = _SHAPES[shape]
    targets = {f"src/Gone{ext}": "export default 1;\n" for ext in (".js", ".css", ".svg")}
    assert _issues(tmp_path, {rel: source, **targets}) == []


def test_an_undeclared_package_is_still_reported(tmp_path):
    [issue] = _issues(tmp_path, {"src/api.js": "import axios from 'axios';\nexport default axios;\n"})
    assert issue.startswith("frontend dependency:") and "'axios'" in issue


def test_a_runtime_built_path_is_not_an_import(tmp_path):
    source = "export const load = (name) => import('./pages/' + name);\n"
    assert _issues(tmp_path, {"src/routes.js": source}) == []


def test_an_apostrophe_in_jsx_text_does_not_hide_a_later_import(tmp_path):
    """The lexer opens a string at "Don't" and reads the code after it as
    string content; the file falls back to the specifier regex."""
    source = ("function A() { return <p>Don't worry</p>; }\n"
              "const B = lazy(() => import('./Gone'));\n"
              "function C() { return <p>it's</p>; }\n")
    [issue] = _issues(tmp_path, {"src/App.js": source})
    assert "imports './Gone'" in issue


@pytest.mark.parametrize("rel,source", [
    # Unterminated: the template literal runs to the end of the file.
    ("src/Keys.jsx", "import a from './a';\nconst H = () => <p>Press the ` key</p>;\n"
                     "const P = React.lazy(() => import('./Gone'));\n"),
    ("src/Keys.js", "import a from './a';\nconst H = () => <p>Press the ` key</p>;\n"
                    "import P from './Gone';\nexport default P;\n"),
    # Two lone backticks close a template literal around real code.
    ("src/Pair.jsx", "const H = () => <p>Press ` here</p>;\nimport P from './Gone';\n"
                     "const K = () => <p>or ` there</p>;\n"),
])
def test_a_lone_backtick_in_jsx_text_does_not_hide_a_later_import(tmp_path, rel, source):
    """The lexer opens a template literal at the backtick and reads the code
    after it as template text; the file falls back to the specifier regex."""
    issues = _issues(tmp_path, {rel: source, "src/a.js": "export default 1;\n"})
    assert [i for i in issues if "imports './Gone'" in i], issues


def test_a_quoted_import_name_keeps_its_specifier(tmp_path):
    """ES2022 string export names: ``import { "a-b" as ab } from "./x"``."""
    [issue] = _issues(tmp_path, {"src/names.js": 'import { "a-b" as ab } from "./Gone";\nexport default ab;\n'})
    assert "imports './Gone'" in issue


def test_a_template_literal_with_text_stays_quiet(tmp_path):
    source = "export const msg = (n) => `Imported from '${n}'\nimport the rest later`;\n"
    assert _issues(tmp_path, {"src/msg.js": source}) == []
