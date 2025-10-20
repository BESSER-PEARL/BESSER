## Python Guidelines
- Follow PEP 8 with 4-space indentation and keep lines within ~120 characters.
- Use type hints to describe interactions between metamodel elements, generators, and utilities.
- Organize imports (stdlib, third-party, local) and avoid wildcard imports.
- Prefer dataclasses or attrs-style patterns already present in the package you touch.
- Add docstrings for public APIs and include examples for B-UML usage when helpful.
- Favor small, composable functions; promote reusable logic to `besser.utilities`.
- Update or add pytest coverage in `tests/` whenever behavior changes.
