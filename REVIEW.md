# Review instructions

These rules apply to automated pull request review. Project conventions live in `CLAUDE.md`; read it as the
reference for every finding below.

## What this review enforces

A change must fit the codebase as it already is. Grade it against BESSER's existing patterns, not a generic
best-practice rubric.

- **Follow the existing convention.** When the codebase already has a way to do something, a new but equivalent
  way is a finding: `@handle_endpoint_errors` plus the exceptions in `services/exceptions.py`, the
  `GeneratorInfo` registry, the `tests/conftest.py` fixtures, `_escape_python_string` / `safe_var_name` in
  `buml_code_builder/common.py`, Jinja2 templates under `generators/<name>/templates/`, `constants/constants.py`.
- **No papering over.** Flag a fix that hides a symptom instead of resolving the cause:
  - `except Exception: pass` or a bare `except:` that swallows an error
  - `isinstance(x, dict)` fallbacks next to a metamodel type, or `hasattr` / `getattr(..., default)` guards on
    metamodel attributes
  - a duplicated helper instead of an import of the existing one, or a copy-pasted sibling file that should
    have been a parameter
  - a value hardcoded that already lives in `constants/constants.py` or an env var
  - a test made to pass by loosening its assertion
  - a BESSER exception (`ConversionError`, `ValidationError`, ...) re-raised as a plain `HTTPException`
  - `# TODO` / `# workaround` without a tracked issue
- **Prefer deletion.** If a new code path does the job of an old one that is still there, ask why both exist.
- **Cite the canonical example.** A convention finding names the file and line that shows the correct pattern
  (for example "use `@handle_endpoint_errors` as in `routers/conversion_router.py`"). If you cannot point to
  one, do not post the finding.

## What Important means here

Important is for findings that break behavior or a contract:

- incorrect logic, crashes, data loss, security issues
- a converter feature added in only one direction (`json_to_buml` without `buml_to_json`, or the reverse)
- a generator that does not implement `GeneratorInterface` or is missing from `SUPPORTED_GENERATORS` /
  `get_filename_for_generator`
- a removed or renamed public metamodel attribute without a documented migration
- user-controlled strings emitted into generated Python without `_escape_python_string` / `safe_var_name`
- `shell=True` with user input, path traversal on upload/import/export, secrets in the diff
- a release PR whose `setup.cfg` version does not match the newest `docs/source/releases/` notes file
- a load-bearing convention break from the list above

Missing tests or docs for a public-surface change are Important. Cosmetic convention breaks, naming, and
style are Nit at most.

## Always check

- **Metamodel** (`besser/BUML/metamodel/`): private attributes with validating setters; new `NamedElement`
  subclasses keep the `name` validation; `validate()` collects errors and returns them, it does not raise;
  "many" is `UNLIMITED_MAX_MULTIPLICITY`, not a new magic number.
- **Converters** (`services/converters/`): both directions updated; a round-trip test (JSON to BUML to JSON is
  identity); cross-diagram references resolve by ID, not by name; the old single-diagram project format still
  migrates.
- **Generators** (`besser/generators/`): Jinja2 templates for non-trivial output; deterministic output (no
  timestamps or random names); temp dirs use a `besser_*` prefix from `constants.py` and are cleaned up;
  optional heavy deps (torch, tensorflow) imported in `try` and registered only when importable.
- **Notations** (`besser/BUML/notations/`): grammar and listener change together; parse errors are collected.
- **Backend** (`besser/utilities/web_modeling_editor/backend/`): request/response models in `models/`, not
  inline in routers; new temp-dir prefixes registered in `services/cleanup.py`; no new CORS wildcard; OAuth
  state never reused.
- **Contract changes**: a changed endpoint or response shape names the matching frontend change in
  `shared/api/` of BESSER-Web-Modeling-Editor.
- **Tests**: reuse the `tests/conftest.py` fixtures instead of rebuilding the same model; generator tests
  assert both structure (names, endpoints) and content (the logic is in the output).
- **Docs**: public-surface changes update `docs/source/` as listed under "Documentation Sync" in `CLAUDE.md`.
- **Python version**: `python_requires >= 3.11`; CI runs 3.11 and 3.12. Do not flag 3.11 syntax.

## Do not report

- Generated ANTLR files under `besser/BUML/notations/` (`*Lexer.py`, `*Parser.py`, `*Listener.py`,
  `*.interp`, `*.tokens`) unless the matching `.g4` is missing from the diff. Their `F401` / `E711`
  suppressions are expected.
- The frontend submodule pointer (`besser/utilities/web_modeling_editor/frontend`). It is a SHA bump; the
  frontend is reviewed in its own repository.
- Anything CI's ruff command already enforces (see "Linting" in `CLAUDE.md`).
- Test files that shadow source file names; `--import-mode=importlib` makes that intentional.
- Generic advice such as "consider adding validation" or "consider more tests" without a concrete path.

## Cap the nits

Report at most five Nits per review. Prefer the ones that point at a reusable existing helper.
