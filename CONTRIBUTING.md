# Contributing to this Repository

First of all — **thank you** for your interest in contributing to BESSER! ❤️
There are several ways you can help, beyond writing code. This document explains how
to get started, how we collaborate, and what we expect from every contribution.

---

## 📋 Before You Start

- Read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).
- Familiarize yourself with the [project governance model](GOVERNANCE.md).
- Skim the [README](README.md) and the
  [Read the Docs site](https://besser.readthedocs.io/) for background and examples.

If you will work with an AI pair-programming assistant, also review the
dedicated [AI assistant guide](docs/source/ai_assistant_guide.rst).

---

## 🧠 Asking Questions

Have a question? Feel free to [open an issue](https://github.com/BESSER-PEARL/BESSER/issues/new/choose)
or start a discussion on an existing ticket. Please write **clear and concise**
messages—the more detail you provide, the easier it is for us to help.

---

## 🐞 Reporting Issues or Requesting Features

Found a bug? Have a feature request? We want to hear from you! Follow these steps
to help us handle your report effectively:

1. **Search for existing issues.** Your topic may already be tracked in the
   [issue list](https://github.com/BESSER-PEARL/BESSER/issues).
2. **Describe one problem per issue.** Include clear reproduction steps,
   screenshots, stack traces, links to relevant models, and the expected vs.
   actual behavior.
3. **Explain the impact.** Knowing why the change matters helps prioritize the
   work.

---

## 🍴 Setting Up Your Development Environment

1. **Fork** this repository to your own GitHub account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/BESSER.git
   cd BESSER
   ```
3. **Create and activate a virtual environment.** Use your preferred tool
   (`python -m venv`, `conda`, etc.).
4. **Install project dependencies and configure the environment:**
   ```bash
   python setup_environment.py
   ```
   The script installs core dependencies and ensures the `PYTHONPATH` points to
   the local repository. Re-run it whenever you open a new shell or IDE session.
5. **Install documentation extras (optional but recommended):**
   ```bash
   pip install -r docs/requirements.txt
   ```

If you prefer to manage the environment manually, use `pip install -r
requirements.txt` followed by `pip install -e .` to work with an editable
installation.

---

## 🗂️ Repository Tour

- `besser/` – Core Python packages for the B-UML metamodel, generators,
  utilities, and the backend services that power the web modeling editor.
- `docs/` – The Sphinx documentation that powers
  [besser.readthedocs.io](https://besser.readthedocs.io/). Any user-facing
  guidance lives here.
- `tests/` – Automated tests for the metamodel, generators, and utilities.
- `start.sh` / `start.bat` – Helper scripts for local demos.
- `docker-compose.yml` – Container definition for running the full stack
  locally.

The [Contributor Guide](docs/source/contributor_guide.rst) provides an extended
overview of how the pieces fit together.

---

## 💻 BESSER-Specific Guidelines

### Code style and tooling

- Follow the formatting, naming, and AI-assistant rules defined in
  `.github/copilot-instructions.md` (also available as `.cursorrules` for Cursor IDE).
- Write idiomatic Python (PEP 8), add type hints where they improve clarity, and
  prefer descriptive naming.
- Keep functions, modules, and classes focused. Share repeated logic through
  helpers in `besser/utilities` where appropriate.

### Working in core packages

- **Generators:** Place new generators in `besser/generators/`, inheriting from
  [`GeneratorInterface`](besser/generators/generator_interface.py).
- **B-UML metamodel:** Extend the metamodel under `besser/BUML/metamodel/`. Review
  existing packages (e.g., `structural`, `state_machine`) to match patterns.
- **Notations:** Add new textual or graphical notations in
  `besser/BUML/notations/`. Existing PlantUML grammars are good references.

### Backend architecture at a glance

- **Metamodel (`besser/BUML/`)** – Defines the domain-specific language that
  describes applications. Keep entity, state machine, and notation updates
  backward compatible whenever possible.
- **Generators (`besser/generators/`)** – Transform models into deployable
  artifacts (e.g., Django projects, SQLAlchemy schemas). Extend
  `GeneratorInterface` and reuse existing templates/utilities when adding a new
  target.
- **Utilities (`besser/utilities/`)** – Home to shared helpers, CLI tooling, and
  the services backing the web modeling editor. Prefer augmenting these modules
  over creating bespoke helpers inside generators.

### Related projects

The web modeling editor frontend lives in the
[BESSER_WME_standalone](https://github.com/BESSER-PEARL/BESSER_WME_standalone)
repository (included here as a submodule under
`besser/utilities/web_modeling_editor/`). Contribute UI changes or new widgets
there. This repository focuses on the backend API, metamodel, generators, and
utilities that the editor consumes.

---

## ✅ Testing Your Changes

- Run unit tests with `pytest` before submitting a pull request:
  ```bash
  python -m pytest
  ```
- Use the `-k` flag to target specific modules while iterating:
  ```bash
  python -m pytest tests/BUML/metamodel/structural -k library
  ```
- Add or update tests alongside code changes to demonstrate the expected
  behavior.

---

## 🧾 Documentation Updates

- Keep documentation in sync with code. Most user-facing docs live in
  `docs/source/` and are written in reStructuredText.
- Build the docs locally to verify formatting:
  ```bash
  cd docs
  make html  # or: sphinx-build -b html source build/html
  ```
- Screenshots or diagrams should go in `docs/source/img/` or
  `docs/source/_static/`.

---

## 🔀 Creating Pull Requests

1. Create a topic branch off `main` that describes your change, e.g.,
   `feature/add-django-generator`.
2. Make logically grouped commits with meaningful messages.
3. Ensure your branch is up to date with `main` before opening the PR.
4. Fill in the pull request template, summarizing the change, tests run, and any
   follow-up work.
5. Be responsive to review feedback—review is a collaborative conversation.

All pull requests are reviewed according to our project governance rules and
must pass automated checks before merging.

---

## 🧭 Governance

All contributions are reviewed by project maintainers following the rules in our
[GOVERNANCE.md](GOVERNANCE.md). This ensures fair and transparent
decision-making.

---

## 💖 Thank You!

Your contributions—big or small—make this project better. Thank you for your
time, energy, and passion for open source.
