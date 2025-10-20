## Repository-specific Guidance
- This repo houses the backend for the BESSER Web Modeling Editor and the B-UML SDK.
- Focus changes on the Python packages in `besser/` (`BUML/`, `generators/`, `utilities/`) and shared tooling under `tests/` and `docs/`.
- Reuse helpers in `besser.utilities` before introducing new modules.
- Keep backend documentation aligned across `README.md`, `CONTRIBUTING.md`, and `docs/source/`.
- Frontend work belongs in the `BESSER_WME_standalone` repository; only touch the editor submodule if specifically requested.
- Write clear, maintainable code with docstrings and type hints where they clarify APIs.
- Add or update tests when changing metamodel rules, generators, or utilities, and run `python -m pytest`.
- Ensure code passes formatters/linters configured in the project before committing.
