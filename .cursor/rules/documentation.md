## Documentation Guidelines
- Documentation lives under `docs/source/` (reStructuredText) and must describe backend features, generators, and contributor workflows.
- Mirror updates in `CONTRIBUTING.md` or `README.md` when changing onboarding steps.
- Reference modules using Sphinx roles like `:mod:` and `:class:` to help readers find backend entry points.
- Place shared images in `docs/source/_static/` or `docs/source/img/`.
- Run `cd docs && make html` before submitting documentation changes to catch syntax issues.
