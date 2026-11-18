"""Every output says how to run it.

The two-service writer of ``ensure_frontend_scaffold`` writes a README only for
a FastAPI backend beside a Vite frontend with no Docker files. A backend on its
own, or an app started by one Compose file, reached the user with no word on
how to start it. The commands asserted here are read back from the output.
"""
import re

from besser.spec_driven_agent.repair.scaffold_repair import ensure_frontend_scaffold, ensure_run_instructions
from tests.spec_driven_agent.test_frontend_scaffold_repair import _MAIN_API_PY, _flat_app


def _backend(root, entry="main_api.py", source=_MAIN_API_PY):
    backend = root / "backend"
    backend.mkdir(parents=True)
    (backend / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    (backend / entry).write_text(source, encoding="utf-8")
    return backend


def _commands(readme):
    return [line for block in re.findall(r"```bash\n(.*?)```", readme, re.DOTALL)
            for line in block.strip().splitlines()]


def test_a_backend_on_its_own_gets_a_readme_with_its_commands(tmp_path):
    backend = _backend(tmp_path)

    assert ensure_run_instructions(str(tmp_path), app_name="Order Desk") == ["README.md"]
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    commands = _commands(readme)

    assert readme.splitlines()[0] == "# Order Desk"
    assert commands[0] == "cd backend" and (tmp_path / "backend").is_dir()
    assert "python -m venv .venv" in commands
    assert "pip install -r requirements.txt" in commands and (backend / "requirements.txt").is_file()
    assert commands[-1] == "python main_api.py"
    assert "http://localhost:8000/docs" in readme
    assert "python -m uvicorn main_api:app --port 8080" in readme


def test_the_port_is_the_one_the_entry_file_serves_on(tmp_path):
    _backend(tmp_path, source=_MAIN_API_PY.replace("port=8000", "port=8100"))
    ensure_run_instructions(str(tmp_path))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert _commands(readme)[-1] == "python main_api.py"
    assert "http://localhost:8100/docs" in readme and "localhost:8000" not in readme


def test_an_entry_that_does_not_run_itself_is_started_with_uvicorn(tmp_path):
    _backend(tmp_path, entry="server.py", source="from fastapi import FastAPI\napp = FastAPI()\n")
    ensure_run_instructions(str(tmp_path))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert _commands(readme)[-1] == "python -m uvicorn server:app --port 8000"
    assert "http://localhost:8000/docs" in readme


def test_a_backend_at_the_output_root_needs_no_cd(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / "main_api.py").write_text(_MAIN_API_PY, encoding="utf-8")
    ensure_run_instructions(str(tmp_path))
    commands = _commands((tmp_path / "README.md").read_text(encoding="utf-8"))

    assert not [line for line in commands if line.startswith("cd ")]
    assert commands[-1] == "python main_api.py"


def test_an_existing_readme_is_kept(tmp_path):
    _backend(tmp_path)
    (tmp_path / "readme.md").write_text("mine\n", encoding="utf-8")

    assert ensure_run_instructions(str(tmp_path)) == []
    assert not (tmp_path / "README.md").exists() or (tmp_path / "README.md").read_text(encoding="utf-8") == "mine\n"


def test_the_two_service_readme_is_unchanged(tmp_path):
    _flat_app(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))
    before = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert ensure_run_instructions(str(tmp_path)) == []
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == before


def test_a_frontend_the_two_service_writer_skipped_gets_no_backend_only_readme(tmp_path):
    """Instructions for half an app would read as the whole of it."""
    frontend = _flat_app(tmp_path)
    (frontend / "Dockerfile").write_text("FROM node:22\n", encoding="utf-8")

    assert ensure_run_instructions(str(tmp_path)) == []


_COMPOSE = """\
services:
  frontend:
    build:
      context: ./frontend
      args:
        VITE_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
  backend:
    build: ./backend
    ports:
      - "127.0.0.1:8000:8000"
  worker:
    build: ./worker
    ports:
      - "9000"
"""


def test_an_app_started_by_one_compose_file_gets_its_commands_and_addresses(tmp_path):
    app = tmp_path / "web_app"
    _backend(app)
    (app / "frontend").mkdir()
    (app / "docker-compose.yml").write_text(_COMPOSE, encoding="utf-8")

    assert ensure_run_instructions(str(tmp_path), app_name="Book Shelf") == ["README.md"]
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert readme.splitlines()[0] == "# Book Shelf"
    assert _commands(readme) == ["cd web_app", "docker compose up --build"]
    assert "| frontend | http://localhost:3000 |" in readme
    assert "| backend | http://localhost:8000 |" in readme
    assert "worker" not in readme  # publishes no fixed port
    assert "`web_app/docker-compose.yml`" in readme and "VITE_API_URL" in readme


def test_a_later_run_rewrites_the_readme_besser_wrote(tmp_path):
    """Run 1 delivers a backend; a modify run adds a frontend. The README must describe both."""
    _backend(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))
    assert "A FastAPI backend (`backend/`):" in (tmp_path / "README.md").read_text(encoding="utf-8")

    _flat_app(tmp_path, with_backend=False)
    ensure_frontend_scaffold(str(tmp_path))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "and a React + Vite frontend (`frontend/`)" in readme
    assert "docker compose up --build" in readme and "cd frontend" in _commands(readme)
    ensure_frontend_scaffold(str(tmp_path))
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == readme


def test_a_readme_without_besser_s_line_is_never_touched(tmp_path):
    _backend(tmp_path)
    ensure_frontend_scaffold(str(tmp_path))
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    own = "\n".join(line for line in text.splitlines() if "<!--" not in line) + "\nmy notes\n"
    (tmp_path / "README.md").write_text(own, encoding="utf-8")

    _flat_app(tmp_path, with_backend=False)
    ensure_frontend_scaffold(str(tmp_path))
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == own


def test_a_backend_beside_another_frontend_app_gets_no_half_readme(tmp_path):
    _backend(tmp_path)
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text('{"dependencies": {"next": "14"}}', encoding="utf-8")

    assert ensure_run_instructions(str(tmp_path)) == []
    assert not (tmp_path / "README.md").exists()


def test_an_entry_module_python_cannot_import_gets_no_readme(tmp_path):
    _backend(tmp_path, entry="main-api.py", source="from fastapi import FastAPI\napp = FastAPI()\n")
    assert ensure_run_instructions(str(tmp_path)) == []


_COMPOSE_WITH_STORES = """\
services:
  web:
    build: ./web
    ports:
      - "${WEB_PORT:-3000}:80"
  api:
    build: ./backend
    ports: ["8000:8000"]
  db:
    image: postgres:16
    ports: ["5432:5432"]
  cache:
    image: redis:7
    ports: ["6379:6379"]
"""


def test_the_compose_table_lists_the_app_s_own_services(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(_COMPOSE_WITH_STORES, encoding="utf-8")
    ensure_run_instructions(str(tmp_path))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "| web | http://localhost:3000 |" in readme and "| api | http://localhost:8000 |" in readme
    assert "5432" not in readme and "6379" not in readme


def test_a_compose_file_compose_cannot_read_gets_no_readme(tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: [web, api\n", encoding="utf-8")
    assert ensure_run_instructions(str(tmp_path)) == []
