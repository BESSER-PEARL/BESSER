# GitHub Copilot Instructions for the BESSER Backend Repository

These instructions tailor Copilot suggestions to the structure of the `BESSER`
repository, which implements the **backend** for the BESSER Web Modeling Editor
and the core B-UML SDK. The web editor's frontend lives in
[`BESSER_WME_standalone`](https://github.com/BESSER-PEARL/BESSER_WME_standalone);
avoid authoring UI code here unless you are touching the embedded submodule
under `besser/utilities/web_modeling_editor/`.

## Orientation
- Prioritize changes in the Python packages located in `besser/`:
  - `BUML/` contains the metamodel, textual/graphical notations, and the validation logic consumed by generators.
  - `generators/` contains backend code generators (Django, SQLAlchemy, Flutter, etc.).
  - `utilities/` hosts shared helpers (templating, configuration, CLI tools) that back both the SDK and the editor services.
- When enhancing the web modeling editor backend, reuse these utilities rather than duplicating logic in ad-hoc modules.
- Keep documentation in sync—many backend changes require updates under `docs/source/` or in `CONTRIBUTING.md`.

## General Coding Guidelines
- Write clear, maintainable, and well-documented code.
- Follow the conventions already established in the touched package.
- Use descriptive names and add or update docstrings for public APIs.
- Prefer type annotations for function signatures where practical.
- Avoid duplication—promote helpers into `besser.utilities` when multiple generators or services share behavior.
- Cover new behavior with tests in `tests/`, especially around metamodel semantics and generator output.
- Ensure all code passes linting, formatting, and pytest before committing.

## Python Code Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) with 4-space indentation and
  a 120-character soft line limit.
- Use [PEP 484](https://peps.python.org/pep-0484/) type hints to clarify
  interfaces between metamodel elements, generators, and utilities.
- Organize imports as standard library, third-party, then local modules.
- Avoid wildcard imports and implicit re-exports.
- Use f-strings for formatting, snake_case for functions/variables, PascalCase for classes, and UPPER_CASE for module constants.
- Keep docstrings (PEP 257) up to date, including examples for DSL usage when relevant.

## TypeScript / Frontend Touchpoints
- TypeScript should rarely appear in this repository. If you must edit the editor submodule under `besser/utilities/web_modeling_editor/`, follow the conventions upstream in `BESSER_WME_standalone`:
  - Enable strict compiler options.
  - Prefer `interface` for object shapes, `const` for immutable bindings, and explicit return types.
  - Run the frontend toolchain in that submodule (`npm install`, `npm test`) and keep linting/formatting consistent with its configuration.
- All other UI changes belong in the dedicated frontend repository.

## Documentation Expectations
- Update reStructuredText files under `docs/source/` when backend behavior or
  contributor workflows change. Build locally with `cd docs && make html`.
- Keep `README.md`, `CONTRIBUTING.md`, and the contributor/AI-assistant guides
  aligned with new features or tooling.
- Document environment setup steps any time dependencies or workflows change
  (e.g., new generator prerequisites, updated CLI flags).

---

_Following these guidelines helps Copilot propose changes that respect the
architecture of the BESSER backend and keeps human and AI contributions
consistent._
