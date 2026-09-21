"""Deterministic scaffold repair for an LLM-authored React frontend.

The class-diagram path has no GUI model, so ``generate_web_app`` is never
offered (``tools.py`` requires ``{domain, gui}``) and the frontend is written
file-by-file by the LLM. The orchestrator's checklist names package.json /
index.html / main.jsx / App.jsx / api.js / pages and nothing else, so the
build configuration around them is left to chance.

Live evidence, 2026-09-21. Run ``gpt-5.6-terra-053ydac9`` (in
``verification/spec-iterations/``) was replayed in Chrome twice from the same
source files. Untouched, it rendered nothing and the console showed

    ReferenceError: React is not defined  at App (src/App.jsx:15:3)

because App.jsx uses JSX without importing React and the project has no vite
config, so Vite falls back to esbuild's classic transform. With
``ensure_frontend_scaffold`` applied and nothing else changed, the same app
rendered its home page and its /book page with the create form, and the
console was clean. Across all 192 recorded runs: 192 had no vite config, 96
had a JSX file with no React import, 185 hardcoded the API base, 20 had
``"build": "vite"``, and none had a Dockerfile or a compose file.

Each test below fails against the pre-fix tree, where the function does not
exist and none of these files are ever written.
"""
import json
import os

from besser.spec_driven_agent.scaffold_repair import ensure_frontend_scaffold


# The exact shape the frontend checklist produces, reduced to what matters.
_PACKAGE_JSON = {
    "scripts": {"dev": "vite", "build": "vite build"},
    "dependencies": {"react": "^18.3.1", "react-dom": "^18.3.1",
                     "react-router-dom": "^6.26.2"},
    "devDependencies": {"vite": "^5.4.8"},
    "type": "module",
}

# App.jsx renders JSX and never imports React - the blank-page defect.
_APP_JSX = (
    "import {BrowserRouter, Route, Routes} from 'react-router-dom';\n"
    "export default function App(){return <BrowserRouter>"
    "<Routes><Route path='/' element={<p>hi</p>}/></Routes></BrowserRouter>}\n"
)

_API_JS = (
    "const BASE_URL = 'http://localhost:8000';\n"
    "export const list = (e) => fetch(`${BASE_URL}/${e}/`).then(r => r.json());\n"
)

_MAIN_API_PY = (
    "from fastapi import FastAPI\n"
    "app = FastAPI()\n"
    'if __name__ == "__main__":\n'
    "    import uvicorn\n"
    '    uvicorn.run(app, host="0.0.0.0", port=8000)\n'
)


def _flat_app(root, *, api_js=_API_JS, package_json=None, with_backend=True):
    """The flat backend/ + frontend/ layout a class-diagram run produces."""
    frontend = root / "frontend"
    (frontend / "src").mkdir(parents=True)
    (frontend / "package.json").write_text(
        json.dumps(package_json or _PACKAGE_JSON), encoding="utf-8")
    (frontend / "index.html").write_text(
        '<div id="root"></div><script type="module" src="/src/main.jsx"></script>',
        encoding="utf-8")
    (frontend / "src" / "main.jsx").write_text(
        "import React from 'react';\nimport App from './App.jsx';\n", encoding="utf-8")
    (frontend / "src" / "App.jsx").write_text(_APP_JSX, encoding="utf-8")
    (frontend / "src" / "api.js").write_text(api_js, encoding="utf-8")
    if with_backend:
        backend = root / "backend"
        backend.mkdir()
        (backend / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
        (backend / "main_api.py").write_text(_MAIN_API_PY, encoding="utf-8")
    return frontend


def test_writes_vite_config_with_plugin_react(tmp_path):
    """The whole blank-page failure: no config means the classic transform."""
    frontend = _flat_app(tmp_path)
    assert ensure_frontend_scaffold(str(tmp_path))

    config = (frontend / "vite.config.mjs").read_text(encoding="utf-8")
    assert "@vitejs/plugin-react" in config
    assert "react()" in config

    manifest = json.loads((frontend / "package.json").read_text(encoding="utf-8"))
    # Declared as well as imported: nothing installs it otherwise, and it is
    # also what validation/frontend_resolution.py keys the automatic-runtime
    # check on.
    assert "@vitejs/plugin-react" in manifest["devDependencies"]


def test_repaired_project_passes_the_frontend_resolution_check(tmp_path):
    """The two checks must agree, or the fix loop burns turns on a fixed app."""
    from besser.spec_driven_agent.validation.frontend_resolution import (
        collect_frontend_resolution_issues,
    )
    _flat_app(tmp_path)
    before = collect_frontend_resolution_issues(str(tmp_path))
    assert any("React" in issue for issue in before), before

    ensure_frontend_scaffold(str(tmp_path))
    assert not [issue for issue in collect_frontend_resolution_issues(str(tmp_path))
                if "React" in issue]


def test_api_base_reads_vite_api_url(tmp_path):
    """A hardcoded base can only be changed by editing delivered source."""
    frontend = _flat_app(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))
    api = (frontend / "src" / "api.js").read_text(encoding="utf-8")
    assert api.startswith(
        "const BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000');")


def test_api_base_rewrite_handles_axios_and_skips_interpolations(tmp_path):
    frontend = _flat_app(tmp_path, api_js=(
        "import axios from 'axios';\n"
        "export const api = axios.create({baseURL: 'http://localhost:8000'});\n"
        "export const one = (id) => fetch(`http://localhost:8000/order/${id}/`);\n"
    ))
    ensure_frontend_scaffold(str(tmp_path))
    api = (frontend / "src" / "api.js").read_text(encoding="utf-8")
    assert "baseURL: (import.meta.env.VITE_API_URL || 'http://localhost:8000')" in api
    # Rewriting a template literal would drop the ${id}, so it is left alone.
    assert "fetch(`http://localhost:8000/order/${id}/`)" in api


def test_declares_an_imported_but_missing_package(tmp_path):
    """Live: run mz30st3s was blank because axios was imported, not declared."""
    frontend = _flat_app(tmp_path, api_js="import axios from 'axios';\nexport default axios;\n")
    ensure_frontend_scaffold(str(tmp_path))
    manifest = json.loads((frontend / "package.json").read_text(encoding="utf-8"))
    assert "axios" in manifest["dependencies"]


def test_fixes_build_script_that_starts_a_dev_server(tmp_path):
    package_json = dict(_PACKAGE_JSON, scripts={"dev": "vite", "build": "vite"})
    frontend = _flat_app(tmp_path, package_json=package_json)
    ensure_frontend_scaffold(str(tmp_path))
    manifest = json.loads((frontend / "package.json").read_text(encoding="utf-8"))
    assert manifest["scripts"]["build"] == "vite build"


def test_writes_compose_dockerfiles_and_readme(tmp_path):
    frontend = _flat_app(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))

    compose = (tmp_path / "docker-compose.yml").read_text(encoding="utf-8")
    assert "context: ./backend" in compose and "context: ./frontend" in compose
    # Vite inlines VITE_* at build time, so the image needs it as a build arg.
    assert "VITE_API_URL: http://localhost:8000" in compose

    assert 'CMD ["python", "main_api.py"]' in (
        tmp_path / "backend" / "Dockerfile").read_text(encoding="utf-8")
    # -s serves index.html for unknown paths, which client-side routes need.
    assert 'CMD ["serve", "-s", "dist", "-l", "3000"]' in (
        frontend / "Dockerfile").read_text(encoding="utf-8")
    assert "docker compose up --build" in (
        tmp_path / "README.md").read_text(encoding="utf-8")


def test_keeps_an_existing_deployment_and_config(tmp_path):
    """A project that brought its own setup keeps it; we are the fallback."""
    frontend = _flat_app(tmp_path)
    (frontend / "vite.config.ts").write_text(
        "import react from '@vitejs/plugin-react';\n"
        "export default {plugins: [react()], build: {outDir: 'build'}};\n",
        encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

    ensure_frontend_scaffold(str(tmp_path))
    assert not os.path.exists(frontend / "vite.config.mjs")
    assert (tmp_path / "docker-compose.yml").read_text(encoding="utf-8") == "services: {}\n"
    assert not os.path.exists(frontend / "Dockerfile")


def test_is_idempotent(tmp_path):
    _flat_app(tmp_path)
    assert ensure_frontend_scaffold(str(tmp_path))
    assert ensure_frontend_scaffold(str(tmp_path)) == []


def test_leaves_a_non_react_project_alone(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"express": "^4"}}), encoding="utf-8")
    assert ensure_frontend_scaffold(str(tmp_path)) == []
    assert not os.path.exists(tmp_path / "vite.config.mjs")
