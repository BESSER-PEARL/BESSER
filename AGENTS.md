# Repository Guidelines

BESSER is the backend for the B-UML SDK and the Web Modeling Editor. This repository focuses on Python metamodels, generators, and services; the UI lives in a submodule.

## Project Structure & Module Organization
- `besser/BUML/`: B-UML metamodel, notations, and validation.
- `besser/generators/`: code generators and templates (Django, SQLAlchemy, Flutter, etc.).
- `besser/utilities/`: shared helpers; editor backend lives in `besser/utilities/web_modeling_editor/backend`.
- `besser/utilities/web_modeling_editor/frontend/`: embedded frontend submodule; prefer changes in its upstream repo.
- `tests/`: pytest suite (`tests/BUML`, `tests/generators`, and utilities).
- `docs/source/`: Sphinx documentation and contributor guidance.
- `generated/`: generated artifacts; treat as build output.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` and `pip install -e .`: install the backend in editable mode (Python 3.10+).
- `pip install -r docs/requirements.txt`: documentation build dependencies.
- `python -m pytest`: run the full test suite.
- `python -m pytest tests/generators -k sqlalchemy`: run targeted tests.
- `python tests/BUML/metamodel/structural/library/library.py`: run a small BUML example.
- `docker-compose up --build`: run the full stack locally via Docker.

## Coding Style & Naming Conventions
- PEP 8 with 4-space indentation and a 120-character line limit (pylint config in `pyproject.toml`).
- Use type hints for public APIs, keep docstrings current, and prefer shared helpers in `besser/utilities` over duplication.
- Naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants.
- Imports ordered standard library, third-party, then local modules.

## Testing Guidelines
- Use pytest; place tests under `tests/` and name files `test_*.py`.
- Add or update tests for behavioral changes, especially in metamodels and generators.

## Commit & Pull Request Guidelines
- Recent history uses Conventional Commits-style prefixes like `feat:` and `fix:`; keep subjects short and imperative.
- Use topic branches (for example, `feature/add-generator`), keep commits scoped, and stay up to date with `main`.
- Fill in the PR template with a clear summary and tests run; link issues and include screenshots for frontend changes.

## Documentation & Configuration
- Update `docs/source/` alongside user-facing changes; build locally with `cd docs && make html` (or `docs\\make.bat html` on Windows).
- Use `.env.example` as a template for local configuration; do not commit secrets.
- If using AI tooling, follow `AI_INSTRUCTIONS.md` (mirrored in `.cursorrules` and `.github/copilot-instructions.md`).

## AI Assistant & Architecture Notes
- This repository is backend-first; only touch the frontend submodule when necessary and keep its tooling consistent.
- Keep API, generator, and metamodel changes in sync with docs and tests.
- Favor deterministic generators and reusable utilities to keep outputs stable.
