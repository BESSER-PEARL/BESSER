# GitHub Copilot Instructions for BESSER Repository

## General Guidelines
- Write clear, maintainable, and well-documented code.
- Follow the existing code style and conventions in each submodule.
- Use descriptive variable, function, and class names.
- Add or update docstrings for all public classes and methods.
- Prefer type annotations for all function signatures (Python, TypeScript, etc.).
- Avoid code duplication; refactor or reuse existing utilities where possible.
- Write unit tests for new features and bug fixes.
- Ensure all code passes linting and formatting checks before committing.

## Python Code Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) for code style.
- Use [PEP 484](https://peps.python.org/pep-0484/) type hints.
- Use 4 spaces per indentation level.
- Limit lines to 120 characters.
- Use docstrings for all public modules, classes, and functions (PEP 257).
- Use f-strings for string formatting.
- Import standard libraries first, then third-party, then local modules.
- Avoid wildcard imports.
- Use explicit relative imports within packages.
- Use snake_case for variables and functions, PascalCase for classes, and UPPER_CASE for constants.

## TypeScript Best Practices
- Use [TypeScript strict mode](https://www.typescriptlang.org/tsconfig#strict) and enable all recommended compiler options.
- Prefer `interface` over `type` for object shapes.
- Use `readonly` for properties that should not be reassigned.
- Always specify explicit return types for functions and methods.
- Avoid using `any`; prefer specific types or generics.
- Use `const` for variables that are not reassigned, otherwise use `let`.
- Prefer arrow functions for callbacks and function expressions.
- Use single quotes for strings.
- Use template literals for string interpolation.
- Organize imports: external modules first, then internal modules.
- Use consistent file and folder naming (kebab-case or camelCase).
- Write JSDoc comments for all exported functions, classes, and interfaces.
- Avoid side effects in modules; prefer pure functions.
- Use ESLint and Prettier for linting and formatting.

## Documentation
- Update or add docstrings and comments for all new or modified code.
- Update the `README.md` and other relevant documentation when introducing new features or breaking changes.

---

_These instructions are intended to help maintain a high-quality, consistent codebase and leverage Copilot effectively for this repository._
