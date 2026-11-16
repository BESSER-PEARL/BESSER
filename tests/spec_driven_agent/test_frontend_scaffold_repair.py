"""Deterministic scaffold repair for an LLM-authored React frontend.

The class-diagram path has no GUI model, so ``generate_web_app`` is never
offered (``tools.py`` requires ``{domain, gui}``) and the frontend is written
file-by-file by the LLM. The orchestrator's checklist names package.json /
index.html / main.jsx / App.jsx / api.js / pages and nothing else, so the
build configuration around them is left to chance.

Live evidence: a recorded gpt-5.6-terra run was replayed in Chrome twice from
the same source files. Untouched, it rendered nothing and the console showed

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
import fnmatch
import json
import os
import re

import pytest

from besser.spec_driven_agent.repair.scaffold_repair import ensure_frontend_scaffold


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


def test_readme_is_titled_after_the_model_not_the_run_folder(tmp_path):
    """Run workspaces are temp folders (besser_llm_<run id>_...); the title
    comes from the backend's FastAPI(title="<Name> API")."""
    run_dir = tmp_path / "besser_llm_example_run_folder"
    run_dir.mkdir()
    _flat_app(run_dir)
    (run_dir / "backend" / "main_api.py").write_text(
        _MAIN_API_PY.replace("FastAPI()", 'FastAPI(\n    title="Todo API",\n)'), encoding="utf-8")
    ensure_frontend_scaffold(str(run_dir))
    first = (run_dir / "README.md").read_text(encoding="utf-8").splitlines()[0]
    assert first == "# Todo"


def _run_dir_app(tmp_path, title_literal):
    """A run workspace whose backend declares ``FastAPI(title=<title_literal>)``."""
    run_dir = tmp_path / "besser_llm_example_run_folder"
    run_dir.mkdir()
    _flat_app(run_dir)
    (run_dir / "backend" / "main_api.py").write_text(
        _MAIN_API_PY.replace("FastAPI()", f"FastAPI(\n    title={title_literal},\n)"), encoding="utf-8")
    return run_dir


def _readme_title(run_dir, **kwargs):
    ensure_frontend_scaffold(str(run_dir), **kwargs)
    return (run_dir / "README.md").read_text(encoding="utf-8").splitlines()[0]


@pytest.mark.parametrize("model_name", ["Library", "Author's_Shelf", 'The "Big" Library'])
def test_readme_title_reads_the_title_the_backend_generator_writes(tmp_path, model_name):
    """The generator emits the title with repr(), so the quote character
    follows the text: single quotes unless the name holds one."""
    assert _readme_title(_run_dir_app(tmp_path, repr(f"{model_name} API"))) == (
        "# " + model_name.replace("_", " "))


def test_readme_title_matches_the_generated_backend(tmp_path, library_book_author_model):
    from besser.generators.backend import BackendGenerator

    run_dir = tmp_path / "besser_llm_example_run_folder"
    _flat_app(run_dir, with_backend=False)
    BackendGenerator(library_book_author_model, output_dir=str(run_dir / "backend")).generate()
    assert _readme_title(run_dir) == "# " + library_book_author_model.name.replace("_", " ")


def test_readme_title_prefers_the_project_name(tmp_path):
    run_dir = _run_dir_app(tmp_path, "'Library_Model API'")
    assert _readme_title(run_dir, app_name="City_Library") == "# City Library"


@pytest.mark.parametrize("default_name", ["Class_Diagram", "Class_Diagram_2", "Class Diagram"])
def test_readme_title_is_never_a_default_diagram_tab_name(tmp_path, default_name):
    run_dir = _run_dir_app(tmp_path, repr(f"{default_name} API"))
    assert _readme_title(run_dir) == "# Generated app"


def test_readme_title_skips_a_default_tab_name_for_the_project_name(tmp_path):
    run_dir = _run_dir_app(tmp_path, "'Class_Diagram API'")
    assert _readme_title(run_dir, app_name="City_Library") == "# City Library"


@pytest.mark.parametrize("project_name", ["New_Project", "New Project", "new_project"])
def test_readme_title_skips_the_editors_default_project_name(tmp_path, project_name):
    """Every project starts as "New_Project" (the editor's ProjectHubDialog and
    useProject); the model it holds has the real name."""
    run_dir = _run_dir_app(tmp_path, "'Library API'")
    assert _readme_title(run_dir, app_name=project_name) == "# Library"


def test_readme_title_is_cut_to_a_heading(tmp_path):
    run_dir = _run_dir_app(tmp_path, "'Library API'")
    title = _readme_title(run_dir, app_name="Warehouse " * 40)
    assert title == "# " + ("Warehouse " * 8).strip()


def test_readme_never_uses_a_run_folder_name(tmp_path):
    run_dir = tmp_path / "besser_llm_example_run_folder"
    run_dir.mkdir()
    _flat_app(run_dir)
    ensure_frontend_scaffold(str(run_dir))
    first = (run_dir / "README.md").read_text(encoding="utf-8").splitlines()[0]
    assert first == "# Generated app"


def test_readme_keeps_a_user_chosen_folder_name(tmp_path):
    app_dir = tmp_path / "library_site"
    app_dir.mkdir()
    _flat_app(app_dir)
    ensure_frontend_scaffold(str(app_dir))
    first = (app_dir / "README.md").read_text(encoding="utf-8").splitlines()[0]
    assert first == "# library site"


def _readme(root):
    ensure_frontend_scaffold(str(root))
    return (root / "README.md").read_text(encoding="utf-8")


def _shell_blocks(readme):
    return [block.strip().splitlines() for block in re.findall(r"```bash\n(.*?)```", readme, re.DOTALL)]


def test_readme_commands_each_start_from_the_app_folder(tmp_path):
    """A block that starts a server never returns, so the frontend's steps
    cannot follow the backend's in one block, relative to its folder."""
    _flat_app(tmp_path)
    readme = _readme(tmp_path)

    blocks = [block for block in _shell_blocks(readme) if block[0].startswith("cd ")]
    assert [block[0] for block in blocks] == ["cd backend", "cd frontend"]
    for block in blocks:
        assert (tmp_path / block[0].split()[1]).is_dir()
        assert sum(line.startswith("cd ") or "&& cd " in line for line in block) == 1
    assert blocks[0][-1] == "python main_api.py"
    assert blocks[1][-1] == "npm run dev"
    assert "two terminals" in readme


_PRINTED = "the address `npm run dev` prints"


@pytest.mark.parametrize("vite_config, dev_script, address", [
    (None, "vite", "http://localhost:3000"),                   # the config this repair writes
    ("export default {plugins: [], server: { port: 5173 }};\n", "vite", "http://localhost:5173"),
    ("export default {plugins: []};\n", "vite", "http://localhost:5173"),          # Vite's default
    ("export default {server: {port: 5173}};\n", "vite --host 0.0.0.0 --port 4000", "http://localhost:4000"),
    # The server's own port, not the one of its hot-reload socket or of a proxy target.
    ("export default {server: {hmr: {port: 24678}, port: 5190}};\n", "vite", "http://localhost:5190"),
    ("export default {server: {proxy: {'/api': {target: 'http://x', port: 9}}, port: 5180}};\n", "vite",
     "http://localhost:5180"),
    ("export default {server: {hmr: {port: 24678}}};\n", "vite", "http://localhost:5173"),
    ("export default {server: {\n  // port: 3000 was taken\n  port: 5200,\n}};\n", "vite", "http://localhost:5200"),
    # Not readable without running the config: say where the address appears.
    ("const p = 4321;\nexport default {server: {port: p}};\n", "vite", _PRINTED),
    ("export default {server: {port: Number(process.env.PORT) || 5173}};\n", "vite", _PRINTED),
    ("export default {preview: {port: 4173}};\n", "vite", _PRINTED),
    ("export default {plugins: []};\n", "vite --host --port=$PORT", _PRINTED),
])
def test_readme_names_the_dev_server_address(tmp_path, vite_config, dev_script, address):
    package_json = dict(_PACKAGE_JSON, scripts={"dev": dev_script, "build": "vite build"})
    frontend = _flat_app(tmp_path, package_json=package_json)
    if vite_config:
        (frontend / "vite.config.js").write_text(vite_config, encoding="utf-8")
    assert f"The web app, on {address}:" in _readme(tmp_path)


def test_readme_starts_a_backend_that_does_not_run_itself_with_uvicorn(tmp_path):
    _flat_app(tmp_path)
    (tmp_path / "backend" / "main_api.py").unlink()
    (tmp_path / "backend" / "server.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
    readme = _readme(tmp_path)

    assert "python -m uvicorn server:app --port 8000" in readme
    assert "python server.py" not in readme and "main_api" not in readme


def test_readme_moves_the_api_port_without_editing_source(tmp_path):
    """The generated entry point hard-codes port 8000; uvicorn takes another."""
    _flat_app(tmp_path)
    readme = _readme(tmp_path)

    assert "python -m uvicorn main_api:app --port 8080" in readme
    assert "frontend/.env.local" in readme
    assert "VITE_API_URL=http://localhost:8080" in readme


def _docker_ignores(patterns, rel):
    """Docker's rule for the pattern shapes used: ``**/x`` matches a path
    component at any depth, a bare ``x`` the first one; a matched folder
    takes everything under it."""
    parts = rel.split("/")
    return any(
        any(fnmatch.fnmatchcase(part, pattern[3:]) for part in parts) if pattern.startswith("**/")
        else fnmatch.fnmatchcase(parts[0], pattern)
        for pattern in patterns
    )


@pytest.mark.parametrize("service", ["backend", "frontend"])
def test_each_dockerfile_has_a_dockerignore_for_what_a_local_run_leaves(tmp_path, service):
    """Both Dockerfiles ``COPY . .``: without it a host node_modules, virtualenv
    or database file is copied into the image built after a local run."""
    _flat_app(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))

    assert "COPY . ." in (tmp_path / service / "Dockerfile").read_text(encoding="utf-8")
    patterns = (tmp_path / service / ".dockerignore").read_text(encoding="utf-8").split()
    for leftover in ("node_modules/react/index.js", "data/app.db", ".venv/lib/site.py",
                     "venv/pyvenv.cfg", "routers/__pycache__/order.cpython-311.pyc",
                     "dist/index.html", ".env", ".env.local", "server.log"):
        assert _docker_ignores(patterns, leftover), leftover
    for source in ("main_api.py", "database.py", "routers/order.py", "requirements.txt",
                   "package.json", "package-lock.json", "index.html", "src/App.jsx",
                   "src/environment.js", "vite.config.mjs"):
        assert not _docker_ignores(patterns, source), source


def test_keeps_an_existing_dockerignore(tmp_path):
    _flat_app(tmp_path)
    (tmp_path / "backend" / ".dockerignore").write_text("tests\n", encoding="utf-8")
    ensure_frontend_scaffold(str(tmp_path))
    assert (tmp_path / "backend" / ".dockerignore").read_text(encoding="utf-8") == "tests\n"


def test_frontend_dockerfile_builds_on_node_22(tmp_path):
    """npm 10.8.2 (Node 20's) crashes installing a vite 8 tree with "Cannot
    read properties of null (reading 'edgesOut')"; Node 22's npm installs it."""
    frontend = _flat_app(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))

    first = (frontend / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    assert first == "FROM node:22-alpine"
