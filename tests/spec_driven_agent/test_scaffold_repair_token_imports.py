"""Scaffold repair reads imports from code tokens, like the resolution check.

``scaffold_repair._bare_imports`` used a raw-text regex, so ``from 'axios'`` in
JSX text or a commented-out import declared ``axios`` in package.json for an app
that never imports it. The validator already reads specifiers from tokens
(``frontend_resolution.import_specifiers``); both now read the same thing.
"""

from __future__ import annotations

import json

from besser.spec_driven_agent.repair.scaffold_repair import _bare_imports, ensure_frontend_scaffold

_MANIFEST = {"scripts": {"dev": "vite", "build": "vite build"},
             "dependencies": {"react": "^18.3.1", "react-dom": "^18.3.1"},
             "devDependencies": {"vite": "^5.4.8", "@vitejs/plugin-react": "^4.3.1"}}
_APP = ("import React from 'react';\n"
        "// import axios from 'axios';\n"
        "export default function App() {\n"
        "  return <p>Rows come from 'react-router-dom' examples</p>;\n"
        "}\n")


def _frontend(tmp_path):
    project = tmp_path / "frontend"
    (project / "src").mkdir(parents=True)
    (project / "package.json").write_text(json.dumps(_MANIFEST), encoding="utf-8")
    (project / "src" / "App.jsx").write_text(_APP, encoding="utf-8")
    return project


def test_text_and_comments_are_not_bare_imports(tmp_path):
    assert _bare_imports(str(_frontend(tmp_path))) == {"react"}


def test_repair_declares_no_package_the_code_does_not_import(tmp_path):
    project = _frontend(tmp_path)

    ensure_frontend_scaffold(str(tmp_path))

    manifest = json.loads((project / "package.json").read_text(encoding="utf-8"))
    assert "axios" not in manifest["dependencies"]
    assert "react-router-dom" not in manifest["dependencies"]
