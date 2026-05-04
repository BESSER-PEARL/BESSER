# Contributing to BESSER

First — **thank you** for your interest in contributing to BESSER! ❤️

BESSER is a community-driven, low-code platform for model-driven software
engineering. Contributions of every size are welcome: code, documentation,
examples, bug reports, design feedback, and ideas. This guide explains how to
get set up, how we collaborate, and what we expect from every contribution.

---

## 📑 Table of Contents

- [Contributing to BESSER](#contributing-to-besser)
  - [📑 Table of Contents](#-table-of-contents)
  - [📋 Before You Start](#-before-you-start)
  - [🤝 Ways to Contribute](#-ways-to-contribute)
  - [🧠 Asking Questions](#-asking-questions)
  - [🐞 Reporting Issues \& Requesting Features](#-reporting-issues--requesting-features)
  - [🍴 Setting Up Your Development Environment](#-setting-up-your-development-environment)
    - [Prerequisites](#prerequisites)
    - [Fork, clone, and initialize](#fork-clone-and-initialize)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install dependencies](#install-dependencies)
    - [Verify the install](#verify-the-install)
    - [(Optional) Run the full local stack](#optional-run-the-full-local-stack)
  - [🗂️ Repository Tour](#️-repository-tour)
  - [💻 Development Guidelines](#-development-guidelines)
    - [Code style](#code-style)
    - [Linting](#linting)
    - [Working in core packages](#working-in-core-packages)
    - [Validation](#validation)
  - [✅ Testing Your Changes](#-testing-your-changes)
  - [🧾 Documentation Updates](#-documentation-updates)
  - [🌿 Branching \& Commits](#-branching--commits)
    - [Branch naming](#branch-naming)
    - [Commit conventions](#commit-conventions)
  - [🔀 Creating Pull Requests](#-creating-pull-requests)
    - [Target branch](#target-branch)
    - [Step-by-step](#step-by-step)
    - [Writing a good PR description](#writing-a-good-pr-description)
      - [Tips](#tips)
      - [Example (good)](#example-good)
    - [Review \& merging](#review--merging)
  - [🔗 Cross-Repo Changes (Frontend Submodule)](#-cross-repo-changes-frontend-submodule)
  - [🧭 Governance](#-governance)
  - [💖 Thank You!](#-thank-you)

---

## 📋 Before You Start

- Read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).
- Familiarize yourself with the [project governance model](GOVERNANCE.md).
- Skim the [README](README.md) and the
  [Read the Docs site](https://besser.readthedocs.io/) for background and
  examples.
- If you plan to use an AI pair-programming assistant, also review the
  [AI assistant guide](docs/source/ai_assistant_guide.rst).

---

## 🤝 Ways to Contribute

You don't need to be a Python expert to help. Useful contributions include:

- **Code** — new generators, metamodel improvements, bug fixes, performance
  work, refactors.
- **Documentation** — clarifying existing pages, fixing typos, writing
  tutorials, adding examples to `docs/source/`.
- **Tests** — improving coverage, adding regression tests, hardening the
  metamodel and converters.
- **Issue triage** — reproducing reported bugs, narrowing down causes,
  suggesting workarounds.
- **Feedback** — sharing how you use BESSER, what is missing, what is
  confusing. Open a discussion or issue.
- **Examples** — contributing new sample models or use cases to
  [BESSER-examples](https://github.com/BESSER-PEARL/BESSER-examples).

If you are looking for a starting point, browse issues labeled
[`good first issue`](https://github.com/BESSER-PEARL/BESSER/labels/good%20first%20issue)
or [`help wanted`](https://github.com/BESSER-PEARL/BESSER/labels/help%20wanted).

---

## 🧠 Asking Questions

Have a question?

- For usage or design questions, [open an issue](https://github.com/BESSER-PEARL/BESSER/issues/new/choose)
  or comment on an existing ticket.
- For private inquiries, email **info@besser-pearl.org**.

Please write **clear, concise** messages — the more detail you provide
(version, OS, expected vs. actual behavior), the easier it is to help.

---

## 🐞 Reporting Issues & Requesting Features

Found a bug or have a feature idea? We want to hear it. To make your report
actionable:

1. **Search first.** Check the [issue list](https://github.com/BESSER-PEARL/BESSER/issues)
   to avoid duplicates.
2. **One topic per issue.** Mixing concerns slows triage.
3. **Include reproduction details:**
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment (OS, Python version, BESSER version, browser if WME-related)
   - Stack traces, screenshots, or links to the model that triggers the issue
4. **Explain the impact.** Knowing *why* the change matters helps prioritize.

For feature requests, describe the use case, not just the implementation —
that opens room for the best design.

---

## 🍴 Setting Up Your Development Environment

### Prerequisites

- **Python 3.10, 3.11, or 3.12** (CI tests against all three).
- **Git**, including submodule support.
- **Node.js 20+** *(only if you plan to run the web modeling editor frontend
  locally — matches the frontend repo's CI)*.

### Fork, clone, and initialize

1. **Fork** this repository to your GitHub account.
2. **Clone** your fork (with submodules so the frontend is available):

   ```bash
   git clone --recurse-submodules https://github.com/<your-username>/BESSER.git
   cd BESSER
   ```

   If you already cloned without `--recurse-submodules`, run:

   ```bash
   git submodule update --init --recursive
   ```

3. **Add the upstream remote** so you can keep your fork in sync:

   ```bash
   git remote add upstream https://github.com/BESSER-PEARL/BESSER.git
   ```

### Create a virtual environment

```bash
python -m venv venv

# Activate it
source venv/bin/activate           # Linux/macOS
venv\Scripts\activate              # Windows (cmd / PowerShell)
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install -e .                   # editable install of the BESSER package
```

Optional but recommended:

```bash
pip install -r docs/requirements.txt   # Sphinx + theme for building the docs
```

Some test modules require extra packages that are **not** in
`requirements.txt`: `torch`, `tensorflow` (for `tests/generators/nn/`), and
`openpyxl` (for the spreadsheet importer test). CI installs them; locally,
install them only if you intend to touch those areas.

### Verify the install

```bash
python tests/BUML/metamodel/structural/library/library.py
python -m pytest -k library
```

If both run cleanly, you are ready to go.

### (Optional) Run the full local stack

```bash
docker compose up --build
```

This brings up the backend API and the web modeling editor for end-to-end
testing.

---

## 🗂️ Repository Tour

| Path | Purpose |
| ---- | ------- |
| `besser/BUML/metamodel/` | Core B-UML metamodel: structural, state machine, GUI, agents, quantum, etc. |
| `besser/BUML/notations/` | Concrete-syntax parsers (PlantUML, OCL, mockup-to-BUML, …). |
| `besser/generators/` | Code generators (Django, FastAPI, SQLAlchemy, React, Flutter, …). |
| `besser/utilities/` | Shared helpers, CLI tooling, and the web modeling editor backend. |
| `besser/utilities/web_modeling_editor/frontend/` | Frontend submodule (separate repo). |
| `docs/` | Sphinx documentation published to [besser.readthedocs.io](https://besser.readthedocs.io/). |
| `tests/` | Test suite mirroring the source layout. |
| `start.sh` / `start.bat` | Helper scripts for local demos. |
| `docker-compose.yml` | Full local stack. |

For a deeper architectural overview, see the
[Contributor Guide](docs/source/contributor_guide.rst).

---

## 💻 Development Guidelines

### Code style

- Follow **PEP 8** with 4-space indentation and a **120-character** line limit
  (configured in `pyproject.toml`).
- Use `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_CASE` for constants.
- Add type hints for public APIs and write descriptive docstrings.
- Order imports as: standard library → third-party → local.
- Additional formatting, naming, and AI-assistant rules are in
  [`.github/copilot-instructions.md`](.github/copilot-instructions.md) (also
  available as `.cursorrules` for Cursor users).

### Linting

We use **Ruff** for linting (run automatically in CI):

```bash
pip install ruff
ruff check .
```

Fix lint warnings before opening a PR. CI will fail on lint errors.

### Working in core packages

- **Generators (`besser/generators/`)** — inherit from
  [`GeneratorInterface`](besser/generators/generator_interface.py); place
  Jinja2 templates in `generators/<name>/templates/`; register the generator
  in `besser/utilities/web_modeling_editor/backend/config/generators.py` if it
  should be exposed via the web editor.
- **Metamodel (`besser/BUML/metamodel/`)** — keep changes backward-compatible
  where possible; mirror existing patterns (private fields with validating
  setters, `Element → NamedElement → …` hierarchy).
- **Notations (`besser/BUML/notations/`)** — use ANTLR grammars + listener
  pattern, mirroring the existing PlantUML and OCL parsers.
- **Backend (`besser/utilities/web_modeling_editor/backend/`)** — the FastAPI
  app uses a modular router architecture (routers, middleware, services,
  models). Prefer extending an existing module over creating bespoke helpers.
- **Bidirectional converters** — if you support a feature in `json_to_buml/`,
  add the symmetric path in `buml_to_json/`. Round-trips must be lossless.

### Validation

Three validation layers run together; respect each in the right place:

1. **Construction** — setter checks in metamodel classes (fail fast).
2. **Metamodel** — `.validate()` returns `{success, errors, warnings}`.
3. **Constraints** — OCL constraints evaluated against the model.

Collect validation errors instead of throwing on the first one, so users see
the full picture.

---

## ✅ Testing Your Changes

Run the full suite before pushing:

```bash
python -m pytest
```

While iterating, target the area you are changing:

```bash
python -m pytest tests/generators -k sqlalchemy
python -m pytest tests/BUML/metamodel/structural -k library
```

If you do **not** have `torch`, `tensorflow`, or `openpyxl` installed locally,
skip the modules that import them so collection does not fail:

```bash
python -m pytest tests/ \
  --ignore=tests/generators/nn \
  --ignore=tests/utilities/web_modeling_editor/backend/test_spreadsheet_import.py
```

Guidelines:

- **Add or update tests** alongside any behavior change.
- **Reuse fixtures** from `tests/conftest.py` (e.g.,
  `library_book_author_model`, `employee_self_assoc_model`,
  `player_team_domain_model`) and `tests/generators/conftest.py` instead of
  duplicating models.
- **Test round-trips** for converters — `JSON → BUML → JSON` must be the
  identity.
- **Validate behavior**, not implementation details. Assertions on observable
  outputs age better than assertions on internal structure.

---

## 🧾 Documentation Updates

Documentation lives in `docs/source/` and is written in reStructuredText.
Keep it in sync with code: a new generator, endpoint, metamodel concept, or
workflow should be reflected in the docs.

Build locally to check formatting:

```bash
cd docs
make html                # Linux/macOS
make.bat html            # Windows
# or:
sphinx-build -b html source build/html
```

Open `docs/build/html/index.html` to preview.

Common pages to update:

- `buml_language.rst` — metamodel additions
- `generators.rst` — new generators
- `web_editor.rst` — backend API changes
- `utilities.rst` — new utilities
- `contributor_guide.rst` / `ai_assistant_guide.rst` — workflow changes

Place images in `docs/source/img/` or `docs/source/_static/`.

---

## 🌿 Branching & Commits

### Branch naming

Create a topic branch off **`development`** (the integration branch). Use a
prefix that matches the type of change:

| Prefix       | Use for                                          | Example                              |
| ------------ | ------------------------------------------------ | ------------------------------------ |
| `feature/`   | New features or capabilities                     | `feature/add-django-generator`       |
| `fix/`       | Bug fixes                                        | `fix/ocl-parser-null-handling`       |
| `docs/`      | Documentation-only changes                       | `docs/update-contributor-guide`      |
| `refactor/`  | Code restructuring without behavior changes      | `refactor/split-backend-routers`     |
| `test/`      | Test additions or improvements                   | `test/state-machine-validation`      |
| `chore/`     | Tooling, CI, dependency bumps, repo maintenance  | `chore/bump-fastapi`                 |

Use **lowercase, hyphen-separated, descriptive** names. Avoid personal
prefixes (`arm/...`) and avoid umbrella branches that bundle unrelated
changes.

### Commit conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/). Use
one of these prefixes in your commit subject:

- `feat:` — a new feature
- `fix:` — a bug fix
- `docs:` — documentation-only changes
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or fixing tests
- `chore:` — tooling, CI, dependency, or repo-maintenance changes

Keep subjects short, imperative, and lowercase, e.g.:

```
feat: add quantum circuit generator
fix: handle null cardinality in OCL parser
docs: clarify branch naming in CONTRIBUTING
```

Group related work into logically scoped commits — reviewers should be able
to follow the reasoning commit by commit.

---

## 🔀 Creating Pull Requests

### Target branch

> **Open all pull requests against `development`, not `master`.**
>
> The `master` branch tracks released versions. The `development` branch is
> where features are integrated and validated before release. Maintainers
> periodically merge `development` into `master` as part of the release
> process — contributors should not target `master` directly.

When opening the PR on GitHub, the **base branch** dropdown defaults to
`master`. **Change it to `development`** before clicking *Create pull
request*:

```
base repository: BESSER-PEARL/BESSER     base: development   ←  change this!
head repository: <your-fork>/BESSER      compare: feature/your-change
```

If you forget, no harm done — just edit the base branch on the existing PR
(GitHub allows changing it after creation). Maintainers will also flag it.

### Step-by-step

1. **Sync your fork and branch off `development`:**

   ```bash
   git checkout development
   git pull upstream development
   git checkout -b feature/your-change
   ```

2. **Commit your work** using the [Conventional Commits](#commit-conventions)
   prefixes. Make focused commits — reviewers should be able to follow your
   reasoning commit by commit.

3. **Run quality checks locally** before pushing:

   ```bash
   python -m pytest                # tests
   ruff check .                    # lint
   cd docs && make html && cd ..   # if you touched the docs
   ```

4. **Push your branch:**

   ```bash
   git push -u origin feature/your-change
   ```

5. **Open the PR on GitHub** with `development` as the base branch (see
   above). If the work is in progress and you want early feedback, open it
   as a **draft PR** — maintainers will review when you mark it ready.

6. **Rebase or merge `development`** into your branch before requesting
   final review, so the PR reflects an up-to-date base:

   ```bash
   git fetch upstream
   git rebase upstream/development
   git push --force-with-lease
   ```

7. **Respond to review feedback.** Review is a collaborative conversation,
   not a gate. Push fixup commits during review and let the maintainer
   squash on merge if appropriate.

### Writing a good PR description

A great PR description answers four questions: **What changed? Why? How was
it tested? What should reviewers pay attention to?** Use this template (the
PR template in this repo follows the same structure):

```markdown
## What

A 1–3 sentence summary of the change. Stay concrete — name the files,
generators, endpoints, or metamodel concepts you touched.

## Why

The motivation. Link the issue (`Closes #123`), or describe the bug,
limitation, or use case driving the change. Reviewers should not have to
guess the intent.

## How

A brief explanation of the approach, *especially* for non-obvious design
choices. Mention alternatives you considered and why you discarded them.
Skip this section for trivial fixes.

## Testing

- Unit tests added/updated: `tests/path/to/test_thing.py`
- Manual checks: e.g., "ran `python tests/.../library.py` and inspected
  output", or "tested in the web editor: created a class diagram with
  inheritance, exported it, re-imported, round-trip preserved"
- Edge cases considered: …

## Screenshots / Recordings

(For UI or output changes. Drag-and-drop images or GIFs into the PR.)

## Follow-ups / Known limitations

Anything intentionally out of scope for this PR, or known gaps to address
later. Linking a follow-up issue is even better.

## Checklist

- [ ] Targeted `development` (not `master`)
- [ ] Tests added or updated
- [ ] Docs updated (`docs/source/...`) if user-facing behavior changed
- [ ] Lint passes (`ruff check .`)
- [ ] Frontend submodule pointer updated (if a cross-repo change)
```

#### Tips

- **Title matters.** Use a Conventional Commits-style title that mirrors
  your main commit, e.g. `feat: add quantum circuit generator`. Reviewers
  scan PR lists by title.
- **Be specific, not generic.** "Fix bug" is not a PR description. "Fix
  null-pointer in OCL parser when constraint references an unset
  attribute" is.
- **Show, don't tell.** For UI work, include before/after screenshots. For
  generator work, paste a snippet of the generated output. For metamodel
  work, paste the model that exercises the change.
- **Call out anything risky.** Backward-incompatible changes, schema
  migrations, performance trade-offs, dependencies bumped — flag them
  explicitly so reviewers don't miss them.
- **Keep it focused.** One PR = one logical change. If you find yourself
  writing "this PR also …", split it.

#### Example (good)

> **Title:** `feat: add JSONSchema generator with array & enum support`
>
> **What.** Adds a new `JSONSchemaGenerator` under
> `besser/generators/jsonschema/`, registered in the backend generator
> registry. Supports primitive types, enums, arrays (via multiplicity),
> and nested associations.
>
> **Why.** Closes #412. Several users asked for JSON Schema export to
> drive validation in non-Python services.
>
> **How.** Walks the `DomainModel`, emitting one schema file per `Class`.
> Reuses the existing primitive-type mapping from `SQLGenerator` (extracted
> into `besser/utilities/type_mapping.py`). Templates use Jinja2,
> consistent with other generators.
>
> **Testing.** Added `tests/generators/jsonschema/` with three fixtures
> (library, enums, self-association). Round-trip-validated outputs against
> `jsonschema` Python package. Manually exported from the web editor.
>
> **Follow-ups.** OneOf/AnyOf for inheritance is out of scope — tracked in
> #418.

### Review & merging

- All PRs must pass automated checks (tests on Python 3.10/3.11/3.12, Ruff
  lint, frontend lint+build, CodeQL).
- At least one maintainer review is required, per the
  [governance rules](GOVERNANCE.md).
- Maintainers choose the merge strategy (squash, rebase, or merge commit)
  based on the change.

---

## 🔗 Cross-Repo Changes (Frontend Submodule)

The web modeling editor frontend lives in
[BESSER-Web-Modeling-Editor](https://github.com/BESSER-PEARL/BESSER-Web-Modeling-Editor)
and is included here as a submodule under
`besser/utilities/web_modeling_editor/frontend`.

When a change touches **both** repositories (e.g., a new diagram type, an API
contract change):

1. Implement and commit the **frontend** change in the WME repository,
   targeting its `develop` branch.
2. Implement and commit the **backend** change here, targeting `development`.
3. Update the submodule pointer in this repo to the new WME commit:

   ```bash
   cd besser/utilities/web_modeling_editor/frontend
   git pull origin develop
   cd ../../../..
   git add besser/utilities/web_modeling_editor/frontend
   git commit -m "chore: bump frontend submodule"
   ```

4. Link the two PRs in their descriptions and note the intended **merge
   order**.

For a concrete walk-through, see
[Adding a New Diagram Type](https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html)
and `docs/source/contributing/diagram_dsl_workflow.rst`.

---

## 🧭 Governance

All contributions are reviewed by project maintainers following the rules in
[GOVERNANCE.md](GOVERNANCE.md). This ensures fair, transparent
decision-making. If a discussion stalls, maintainers will help unblock it.

---

## 💖 Thank You!

Your contributions — big or small — make BESSER better. Thank you for your
time, energy, and passion for open source.

If anything in this guide is unclear or out of date, that itself is a great
first contribution: open a `docs/` PR and propose an improvement.
