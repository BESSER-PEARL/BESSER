"""The repositories' own CI checks, run on what a lab change touched.

:func:`checks_for` picks the checks the CI workflows of the three repositories
would run (``.github/workflows/ci.yml`` of BESSER and of the editor,
``tests.yml`` of the modeling agent), narrowed to the touched areas;
``full=True`` gives each touched repository's whole CI instead.
:func:`run_checks` runs them and :func:`format_results` turns the results into
what the model reads.

Tests of the touched areas (BESSER): a file maps to the test files directly in
the mirror of its Python package under ``tests/`` (with the renames of
:data:`_TEST_DIR_RENAMES`). A package without such tests falls back to its
nearest ancestor that has some, but never to a top-level area such as
``tests/spec_driven_agent``: the routers' tests sit flat in
``tests/utilities/web_modeling_editor/backend``. A module with neither maps to
the test files of its top-level area that import it (``tests/spec_driven_agent``
is flat). A module the change adds maps only to ``test_<name>.py`` in its
package's mirror and to the tests of its area that import it. A file that maps
to nothing is reported in a skipped "No tests found" result. The editor package
(``packages/editor``) has no test runner, so its files get a skipped result
saying that its tests go in ``packages/webapp``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from functools import partial
from typing import Callable, Iterable, Mapping, Optional, Sequence

from besser.spec_driven_agent.execution.process import (
    FLOOD_NOTE,
    MAX_CAPTURE_BYTES,
    _captured_size,
    _decode,
    _kill_tree,
)
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    CheckResult,
    CheckSpec,
    ExtraTool,
    LabCallbacks,
    RepoChange,
    FileChange,
    RepoSpec,
)
from besser.utilities.subprocess_env import safe_subprocess_env

logger = logging.getLogger(__name__)

Runner = Callable[[CheckSpec, str], tuple[int, str]]

#: Exit code of a runner whose executable is not installed ("command not found").
MISSING_TOOL = 127
#: Exit code of a runner whose command ran past the spec's timeout.
TIMED_OUT = 124
#: Exit code of a runner whose command was killed because the run was stopped (the shell's code for Ctrl-C).
STOPPED = 130
_STOPPED_SUMMARY = "Stopped: the run was stopped before this check finished"
#: How often a running check looks for a timeout, a flood of output or a stop (``run_bounded``'s interval).
_POLL_SECONDS = 0.05

#: Characters of a check's output kept in its result.
OUTPUT_TAIL_CHARS = 6_000
#: What :func:`format_results` shows of one failure's output, and of all of them.
_FAILURE_TAIL_CHARS = 4_000
_FORMAT_BUDGET_CHARS = 20_000

_TESTS_TIMEOUT = 1_800.0
_FULL_SUITE_TIMEOUT = 3_600.0

#: Source package (relative to ``besser/``) -> its test directories (relative to ``tests/``).
_TEST_DIR_RENAMES: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {
    ("generators", "java_classes"): (("generators", "java"),),
    ("generators", "pydantic_classes"): (("generators", "pydantic"),),
    ("generators", "python_classes"): (("generators", "python"),),
    ("generators", "sql_alchemy"): (("generators", "sqlalchemy"),),
    ("generators", "json"): (("generators", "json_object"), ("generators", "json_schema")),
    ("generators", "nn", "forecaster"): (("generators", "forecaster"),),
}

#: Tests a change to the generator registry always runs.
_REGISTRY_TESTS = (
    "tests/generators/test_generator_registry.py",
    "tests/generators/test_regenerable_generators.py",
    "tests/spec_driven_agent/test_added_generator_tools.py",
)

_EDITOR_SCHEMA = "packages/webapp/openapi/besser-openapi.json"
_EDITOR_API_TYPES = "packages/webapp/src/main/shared/api/generated/besser-openapi.d.ts"
# The editor CI's "OpenAPI types match the committed schema" step, without rewriting the d.ts: the types
# openapi-typescript generates from the schema JSON must equal the committed file.
_API_TYPES_SCRIPT = (
    "const {execFileSync}=require('child_process'),fs=require('fs'),path=require('path');"
    "const cli=path.join(path.dirname(require.resolve('openapi-typescript/package.json')),'bin','cli.js');"
    "const file='src/main/shared/api/generated/besser-openapi.d.ts';"
    "const want=execFileSync(process.execPath,[cli,'openapi/besser-openapi.json'],"
    "{encoding:'utf8',stdio:['ignore','pipe','inherit'],maxBuffer:1<<28});"
    "if(fs.readFileSync(file,'utf8')!==want){console.error(file+' is stale: run npm run api:types "
    "--workspace=webapp');process.exit(1)}"
    "console.log(file+' matches openapi/besser-openapi.json');"
)

_WEBAPP = "packages/webapp/"
_WEBAPP_TEST = re.compile(r"\.(test|spec)\.tsx?$")
_SCRIPT_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

_AUTODOC = re.compile(r"^\s*\.\.\s+auto(?:module|class|function|exception|data|attribute|method)::\s*([\w.]+)",
                      re.MULTILINE)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_COUNT_LINE = re.compile(r"\b(passed|failed|errors?|broken)\b", re.IGNORECASE)
_MISSING_MODULE = re.compile(r"No module named '?([A-Za-z_][\w]*)")
#: Python modules that are the check tool itself: missing, the check could not run.
_TOOL_MODULES = frozenset({"pytest", "sphinx", "sphinx_immaterial", "ruff", "importlinter"})
_ADDED = "added"
_DELETED = "deleted"
#: Scripts folder of a checkout's own environment (``python -m venv venv``).
_VENV_SCRIPTS = os.path.join("venv", "Scripts") if os.name == "nt" else os.path.join("venv", "bin")


def checks_for(changes: Sequence[RepoChange], repos: Mapping[str, RepoSpec], *,
               full: bool = False) -> list[CheckSpec]:
    """The CI checks of the touched repositories, narrowed to the changed files unless ``full``.

    A spec with an empty ``argv`` is not run: :func:`run_checks` reports it as
    skipped, with its title as the reason (used for "No tests found for ...").
    """
    by_repo = {change.repo: change for change in changes if change.files}
    frontend = by_repo.get(REPO_FRONTEND)
    editor_schema = frontend is not None and any(f.path == _EDITOR_SCHEMA for f in frontend.files)
    specs: list[CheckSpec] = []
    for key in REPO_ORDER:
        change = by_repo.get(key)
        if change is None:
            continue
        root = repos[key].path
        if key == REPO_BESSER:
            specs += _full_besser(root) if full else _besser_checks(change, root, editor_schema)
        elif key == REPO_FRONTEND:
            specs += _full_frontend() if full else _frontend_checks(change, root)
        elif key == REPO_MODELING_AGENT:
            specs += _agent_checks(change, full, root)
    # The editor's schema can change without a BESSER change; the check runs in BESSER.
    if editor_schema and not full and REPO_BESSER not in by_repo and REPO_BESSER in repos:
        specs.append(_schema_check(repos[REPO_BESSER].path))
    return specs


def run_checks(specs: Sequence[CheckSpec], repos: Mapping[str, RepoSpec], *,
               runner: Runner | None = None, callbacks: LabCallbacks | None = None) -> list[CheckResult]:
    """Run ``specs`` in order and return one result per spec run.

    ``runner(spec, cwd)`` returns ``(exit code, output)``; the default runs the
    command and ends it once ``callbacks.should_continue()`` is False. Exit code
    :data:`MISSING_TOOL`, or a missing check tool module, gives a skipped result, and
    :data:`STOPPED` a skipped "Stopped" one. No check starts once the run is stopped.
    """
    callbacks = callbacks or LabCallbacks()
    run = runner or partial(default_runner, should_continue=callbacks.should_continue)
    results: list[CheckResult] = []
    for spec in specs:
        if not callbacks.should_continue():
            break
        if not spec.argv:
            results.append(CheckResult(spec_id=spec.id, repo=spec.repo, title=spec.title,
                                       passed=False, skipped=True, summary=spec.title))
            continue
        callbacks.on_phase("check", f"{spec.title} ({spec.repo})")
        started = time.monotonic()
        code, output = run(spec, os.path.normpath(os.path.join(repos[spec.repo].path, spec.cwd)))
        result = _result(spec, code, _ANSI.sub("", output), time.monotonic() - started)
        if result.skipped:
            logger.warning("Lab check %s skipped: %s", spec.id, result.summary)
        results.append(result)
    return results


def default_runner(spec: CheckSpec, cwd: str, *,
                   should_continue: Optional[Callable[[], bool]] = None) -> tuple[int, str]:
    """Run ``spec.argv`` in ``cwd`` with a scrubbed environment; stdout and stderr combined.

    The command and every process it started are killed at the spec's timeout (:data:`TIMED_OUT`)
    and as soon as ``should_continue()`` is False (:data:`STOPPED`).
    """
    env = safe_subprocess_env()
    executable = shutil.which(spec.argv[0], path=env.get("PATH"))
    if executable is None:
        return MISSING_TOOL, f"{spec.argv[0]} is not installed (not found on PATH)"
    try:
        return _run_stoppable([executable, *spec.argv[1:]], cwd=cwd, env=env, timeout=spec.timeout_seconds,
                              should_continue=should_continue or (lambda: True))
    except OSError as exc:
        return 1, f"Could not start {spec.argv[0]} in {cwd}: {exc}"


def _run_stoppable(argv: list[str], *, cwd: str, env: dict[str, str], timeout: float,
                   should_continue: Callable[[], bool]) -> tuple[int, str]:
    """``run_bounded``'s handling (own process group, output in temp files, the whole tree killed on a
    timeout or a flood of output), plus a stop: the tree is also killed once ``should_continue()`` is False.

    ``run_bounded`` has no stop hook; once it has one, this is that call.
    """
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, **group)
        deadline = time.monotonic() + timeout
        code: Optional[int] = None
        note = ""
        while True:
            try:
                code = proc.wait(timeout=min(_POLL_SECONDS, max(deadline - time.monotonic(), 0)))
                break
            except subprocess.TimeoutExpired:
                pass
            if not should_continue():
                code, note = STOPPED, "\nStopped: the run was stopped"
            elif max(_captured_size(out), _captured_size(err)) > MAX_CAPTURE_BYTES:
                note = FLOOD_NOTE
            elif time.monotonic() >= deadline:
                code, note = TIMED_OUT, f"\nTimed out after {timeout:g} s"
            else:
                continue
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if code is None:  # a flood: the killed command's own exit code
                code = proc.returncode if proc.returncode is not None else 1
            break
        return code, _decode(out) + _decode(err) + note


def format_results(results: Sequence[CheckResult]) -> str:
    """The results as the model reads them: counts, then failures with their output, skips, passes."""
    if not results:
        return "No checks ran: nothing changed that a check covers."
    failed = [r for r in results if not r.passed and not r.skipped]
    skipped = [r for r in results if r.skipped]
    passed = [r for r in results if r.passed]
    lines = [f"Checks: {len(failed)} failed, {len(skipped)} skipped, {len(passed)} passed."]
    tail_chars = min(_FAILURE_TAIL_CHARS, _FORMAT_BUDGET_CHARS // max(len(failed), 1))
    for result in failed:
        lines.append(f"\nFAILED {result.spec_id}: {result.title} - {result.summary}")
        tail = result.output_tail.strip()
        if len(tail) > tail_chars:
            tail = "[... earlier output cut]\n" + tail[-tail_chars:]
        if tail:
            lines.append(tail)
    if skipped:
        lines.append("")
    lines += [f"SKIPPED {r.spec_id}: {r.title} - {r.summary}" for r in skipped]
    lines += [f"passed {r.spec_id}: {r.title} - {r.summary}" for r in passed]
    return "\n".join(lines)


def run_checks_tool(get_changes: Callable[[], list[RepoChange]], repos: Mapping[str, RepoSpec], *,
                    runner: Runner | None = None, callbacks: LabCallbacks | None = None,
                    on_results: Callable[[list[CheckResult]], None] | None = None) -> ExtraTool:
    """The ``run_checks`` tool: the checks of what the change touched so far, formatted for the model.

    ``runner`` runs each check as in :func:`run_checks` (a caller bounds it by the run's time left).
    ``callbacks`` stream each check's phase and stop the checks once the run is cancelled.
    ``on_results`` receives the results of every call, those of a stopped call included.
    """
    callbacks = callbacks or LabCallbacks()

    def handler(arguments: dict) -> str:
        results = run_checks(checks_for(get_changes(), repos), repos, runner=runner, callbacks=callbacks)
        if on_results is not None:
            on_results(results)
        if not callbacks.should_continue():
            return "The run was stopped; the checks did not finish."
        return format_results(results)

    schema = {
        "name": "run_checks",
        "description": (
            "Run the repositories' own CI checks (tests of the touched areas, lint, import layers, type "
            "check, docs, i18n, API schema) on the files changed so far. Returns failures first with the "
            "end of their output. Run it after a change and fix what fails."
        ),
        "input_schema": {"type": "object", "properties": {}},
    }
    return ExtraTool(schema=schema, handler=handler, writes=True)


# --- BESSER ---------------------------------------------------------------


def _besser_checks(change: RepoChange, root: str, editor_schema: bool) -> list[CheckSpec]:
    tests, unmapped = _besser_tests(change, root)
    python_files = [f for f in change.files if f.path.startswith("besser/") and f.path.endswith(".py")]
    lint_files = [f.path for f in python_files if f.status != _DELETED]
    specs: list[CheckSpec] = []
    if lint_files:
        specs.append(CheckSpec(id="besser:ruff", repo=REPO_BESSER, title="Ruff on the changed files",
                               argv=(_python_tool("ruff", root), "check", *lint_files), timeout_seconds=300,
                               kind="lint"))
    if python_files:
        specs.append(_imports_check(root))
    if tests:
        specs.append(CheckSpec(id="besser:tests", repo=REPO_BESSER, title="Tests of the touched areas",
                               argv=(_python(root), "-m", "pytest", *tests, "-q", "--tb=short",
                                     "-p", "no:cacheprovider"),
                               timeout_seconds=_TESTS_TIMEOUT))
    if unmapped:
        specs.append(_no_tests(REPO_BESSER, unmapped))
    if any(f.path.startswith("docs/") for f in change.files) or _touches_documented_module(python_files, root):
        specs.append(_docs_check(root))
    # Any module the app imports can define a schema component (lab contracts, the traceability metamodel);
    # BESSER's CI runs the check on every pull request.
    if editor_schema or python_files:
        specs.append(_schema_check(root))
    return specs


def _full_besser(root: str) -> list[CheckSpec]:
    return [
        CheckSpec(id="besser:ruff", repo=REPO_BESSER, title="Ruff",
                  argv=(_python_tool("ruff", root), "check", "besser/"), timeout_seconds=300, kind="lint"),
        _imports_check(root),
        CheckSpec(id="besser:tests", repo=REPO_BESSER, title="Test suite",
                  argv=(_python(root), "-m", "pytest", "tests", "-q", "--tb=short", "-p", "no:cacheprovider"),
                  timeout_seconds=_FULL_SUITE_TIMEOUT),
        _docs_check(root),
        _schema_check(root),
    ]


def _imports_check(root: str) -> CheckSpec:
    return CheckSpec(id="besser:imports", repo=REPO_BESSER, title="Import layers",
                     argv=(_python_tool("lint-imports", root),), timeout_seconds=300, kind="lint")


def _docs_check(root: str) -> CheckSpec:
    return CheckSpec(id="besser:docs", repo=REPO_BESSER, title="Documentation build",
                     argv=(_python(root), "docs/check_docs.py", "--offline"), kind="docs")


def _schema_check(root: str) -> CheckSpec:
    return CheckSpec(id="besser:openapi", repo=REPO_BESSER, title="Editor API schema matches the backend",
                     argv=(_python(root), ".github/scripts/check_openapi.py"), timeout_seconds=300, kind="schema")


def _besser_tests(change: RepoChange, root: str) -> tuple[list[str], list[str]]:
    """(test files to run, changed files that map to no tests)."""
    tests: set[str] = set()
    unmapped: list[str] = []
    for file in change.files:
        parts = tuple(file.path.split("/"))
        if parts[0] == "tests":
            if _is_python_test(parts[-1]):
                if file.status != _DELETED and os.path.isfile(os.path.join(root, *parts)):
                    tests.add(file.path)
                continue
            found = _nearest_tests(root, parts[1:-1])
        elif parts[0] == "besser" and len(parts) > 1:
            package = _package_parts(root, parts)
            if file.status == _ADDED and parts[-1].endswith(".py") and parts[-1] != "__init__.py":
                found = _new_module_tests(root, package, file.path)
            else:
                found = _nearest_tests(root, package, renames=True) or _importing_tests(root, file.path)
            if file.path == "besser/generators/registry.py":
                found += [path for path in _REGISTRY_TESTS if os.path.isfile(os.path.join(root, path))]
        else:
            continue
        if found:
            tests.update(found)
        else:
            unmapped.append(file.path)
    return sorted(tests), unmapped


def _package_parts(root: str, parts: tuple[str, ...]) -> tuple[str, ...]:
    """The innermost Python package holding ``besser/...`` file ``parts``, relative to ``besser/``."""
    directory = parts[1:-1]
    while directory and not os.path.isfile(os.path.join(root, "besser", *directory, "__init__.py")):
        directory = directory[:-1]
    return directory if directory else parts[1:-1]


def _nearest_tests(root: str, parts: tuple[str, ...], *, renames: bool = False) -> list[str]:
    """Test files directly in ``tests/<parts>`` or, failing that, its nearest ancestor with any.

    The walk stops at depth 2 (``tests/generators/django``), never reaching a
    top-level area it did not start in. With ``renames``, ``parts`` is a source
    package relative to ``besser/``, mapped through :data:`_TEST_DIR_RENAMES`.
    """
    lowest = max(min(len(parts), 2), 1)
    while len(parts) >= lowest:
        found: list[str] = []
        for mirror in (_mirrors(parts) if renames else (parts,)):
            directory = os.path.join(root, "tests", *mirror)
            if os.path.isdir(directory):
                found += ["/".join(("tests", *mirror, name)) for name in sorted(os.listdir(directory))
                          if _is_python_test(name) and os.path.isfile(os.path.join(directory, name))]
        if found:
            return found
        parts = parts[:-1]
    return []


def _importing_tests(root: str, path: str) -> list[str]:
    """Test files of ``path``'s top-level area (``tests/<area>/``, recursively) that import its module."""
    parts = path.split("/")
    if not path.endswith(".py") or len(parts) < 3:
        return []
    module = _module_name(path)
    parent, _, name = module.rpartition(".")
    imports = re.compile(rf"\b{re.escape(module)}\b"
                         rf"|\bfrom\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(name)}\b")
    found = []
    for directory, _, names in os.walk(os.path.join(root, "tests", parts[1])):
        for test_name in names:
            if _is_python_test(test_name):
                test_path = os.path.join(directory, test_name)
                with open(test_path, encoding="utf-8", errors="replace") as handle:
                    if imports.search(handle.read()):
                        found.append(os.path.relpath(test_path, root).replace(os.sep, "/"))
    return sorted(found)


def _new_module_tests(root: str, package: tuple[str, ...], path: str) -> list[str]:
    """Tests of a module the change adds: ``test_<name>.py`` in its package's mirror, and those importing it.

    A test that existed before the module cannot use it except through a changed module, whose own
    tests that module's mapping selects; the package's other tests are left out.
    """
    name = "test_" + path.rsplit("/", 1)[-1]
    found = {"/".join(("tests", *mirror, name)) for mirror in _mirrors(package)
             if os.path.isfile(os.path.join(root, "tests", *mirror, name))}
    return sorted(found.union(_importing_tests(root, path)))


def _mirrors(package: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    for length in range(len(package), 0, -1):
        targets = _TEST_DIR_RENAMES.get(package[:length])
        if targets:
            return tuple(target + package[length:] for target in targets)
    return (package,)


def _is_python_test(name: str) -> bool:
    return name.startswith("test_") and name.endswith(".py")


def _touches_documented_module(files: Iterable[FileChange], root: str) -> bool:
    modules = {_module_name(f.path) for f in files}
    if not modules:
        return False
    targets: set[str] = set()
    for directory, _, names in os.walk(os.path.join(root, "docs", "source")):
        for name in names:
            if name.endswith(".rst"):
                with open(os.path.join(directory, name), encoding="utf-8", errors="replace") as handle:
                    targets.update(_AUTODOC.findall(handle.read()))
    # ``.. autoclass:: pkg.mod.Name`` documents ``pkg.mod`` too.
    return any(t in modules or t.rsplit(".", 1)[0] in modules for t in targets)


def _module_name(path: str) -> str:
    dotted = path[:-len(".py")].replace("/", ".")
    return dotted[:-len(".__init__")] if dotted.endswith(".__init__") else dotted


# --- Editor ---------------------------------------------------------------


def _frontend_checks(change: RepoChange, root: str) -> list[CheckSpec]:
    tests: set[str] = set()
    unmapped: list[str] = []
    editor_files: list[str] = []
    for file in change.files:
        path = file.path
        if path.startswith(_WEBAPP + "src/"):
            if _WEBAPP_TEST.search(path):
                if file.status != _DELETED:
                    tests.add(path[len(_WEBAPP):])
                continue
            found = _webapp_tests(root, path)
            if found:
                tests.update(found)
            elif path.endswith(_SCRIPT_EXTENSIONS):
                unmapped.append(path)
        elif path.startswith("packages/editor/src/") and path.endswith(_SCRIPT_EXTENSIONS):
            editor_files.append(path)
    paths = [f.path for f in change.files]
    specs: list[CheckSpec] = []
    if any(p.startswith((_WEBAPP, "packages/server/")) and p.endswith(_SCRIPT_EXTENSIONS) for p in paths):
        specs.append(_lint_check())
    if any(p.startswith((_WEBAPP, "packages/editor/")) and _type_checked(p) for p in paths):
        specs.append(_typecheck())
    if tests:
        specs.append(CheckSpec(id="frontend:tests", repo=REPO_FRONTEND, title="Tests of the touched areas",
                               argv=("npx", "vitest", "run", *sorted(tests)), cwd="packages/webapp",
                               timeout_seconds=_TESTS_TIMEOUT))
    if any(p.startswith("packages/server/") for p in paths):
        specs.append(_server_tests())
    if unmapped:
        specs.append(_no_tests(REPO_FRONTEND, unmapped))
    if editor_files:
        # packages/editor has no test script or config, and the editor CI runs only webapp and server tests;
        # the webapp's vitest resolves @besser/wme to the editor's source.
        specs.append(CheckSpec(id="frontend:editor-tests", repo=REPO_FRONTEND, argv=(), title=(
            f"No tests run for {_listing(editor_files)}: the editor package has no test runner and CI runs "
            "none of its tests; tests of editor code go in packages/webapp")))
    if any(p.startswith("packages/i18n/") for p in paths):
        specs.append(_i18n_check())
    if _EDITOR_SCHEMA in paths or _EDITOR_API_TYPES in paths:
        specs.append(_api_types_check())
    return specs


def _type_checked(path: str) -> bool:
    return path.endswith((".ts", ".tsx")) or os.path.basename(path).startswith("tsconfig")


def _webapp_tests(root: str, path: str) -> list[str]:
    """Test files of a webapp source file's area, relative to ``packages/webapp``.

    ``src/main/features/<x>/...`` and ``src/main/shared/<y>/...`` map to every test
    under that folder; any other file to the tests next to it and in its ``__tests__``.
    """
    relative = path[len(_WEBAPP):].split("/")
    if len(relative) > 4 and relative[:2] == ["src", "main"] and relative[2] in ("features", "shared"):
        area = os.path.join(root, "packages", "webapp", *relative[:4])
        found = []
        for directory, _, names in os.walk(area):
            found += [os.path.join(directory, n) for n in names if _WEBAPP_TEST.search(n)]
    else:
        directory = os.path.join(root, "packages", "webapp", *relative[:-1])
        found = [os.path.join(d, n) for d in (directory, os.path.join(directory, "__tests__")) if os.path.isdir(d)
                 for n in os.listdir(d) if _WEBAPP_TEST.search(n)]
    webapp = os.path.join(root, "packages", "webapp")
    return sorted(os.path.relpath(f, webapp).replace(os.sep, "/") for f in found)


def _full_frontend() -> list[CheckSpec]:
    return [
        _lint_check(),
        _typecheck(),
        CheckSpec(id="frontend:tests", repo=REPO_FRONTEND, title="Webapp test suite",
                  argv=("npm", "test", "--workspace=webapp"), timeout_seconds=_FULL_SUITE_TIMEOUT),
        _server_tests(),
        _i18n_check(),
        _api_types_check(),
    ]


def _lint_check() -> CheckSpec:
    return CheckSpec(id="frontend:lint", repo=REPO_FRONTEND, title="ESLint (webapp and server)",
                     argv=("npm", "run", "lint"), timeout_seconds=600, kind="lint")


def _typecheck() -> CheckSpec:
    # scripts/typecheck.mjs fails above its own error baseline, as CI counts it.
    return CheckSpec(id="frontend:types", repo=REPO_FRONTEND, title="Type check (no errors above the baseline)",
                     argv=("npm", "run", "typecheck"), kind="types")


def _server_tests() -> CheckSpec:
    return CheckSpec(id="frontend:server-tests", repo=REPO_FRONTEND, title="Server tests",
                     argv=("npm", "test", "--workspace=server"), timeout_seconds=600)


def _i18n_check() -> CheckSpec:
    return CheckSpec(id="frontend:i18n", repo=REPO_FRONTEND, title="Translation keys",
                     argv=("node", "scripts/i18n-check.mjs"), timeout_seconds=120, kind="i18n")


def _api_types_check() -> CheckSpec:
    return CheckSpec(id="frontend:api-types", repo=REPO_FRONTEND, title="OpenAPI types match the committed schema",
                     argv=("node", "-e", _API_TYPES_SCRIPT), cwd="packages/webapp", timeout_seconds=120,
                     kind="schema")


# --- Modeling agent -------------------------------------------------------


def _agent_checks(change: RepoChange, full: bool, root: str) -> list[CheckSpec]:
    # The agent's tests are flat (tests/test_*.py), so any change that can affect them runs the whole suite.
    relevant = full or any(not f.path.startswith("docs/") and not f.path.endswith(".md") for f in change.files)
    specs: list[CheckSpec] = []
    if full or any(f.path.endswith(".py") for f in change.files):
        specs.append(CheckSpec(id="modeling_agent:ruff", repo=REPO_MODELING_AGENT, title="Ruff",
                               argv=(_python_tool("ruff", root), "check", "src", "patches", "modeling_agent.py"),
                               timeout_seconds=300, kind="lint"))
    if full or any(f.path.startswith("src/") and f.path.endswith(".py") for f in change.files):
        specs.append(CheckSpec(id="modeling_agent:imports", repo=REPO_MODELING_AGENT, title="Import layers",
                               argv=(_python_tool("lint-imports", root),), timeout_seconds=300, kind="lint"))
    if relevant:
        specs.append(CheckSpec(id="modeling_agent:tests", repo=REPO_MODELING_AGENT, title="Test suite",
                               argv=(_python(root), "-m", "pytest", "tests", "-q", "--tb=short",
                                     "--ignore=tests/live", "-p", "no:cacheprovider"),
                               timeout_seconds=_TESTS_TIMEOUT))
    return specs


# --- Running --------------------------------------------------------------


def _no_tests(repo: str, paths: list[str]) -> CheckSpec:
    return CheckSpec(id=f"{repo}:tests-missing", repo=repo, title=f"No tests found for {_listing(paths)}", argv=())


def _listing(paths: list[str]) -> str:
    return ", ".join(paths[:5]) + (f" and {len(paths) - 5} more" if len(paths) > 5 else "")


def _result(spec: CheckSpec, code: int, output: str, duration: float) -> CheckResult:
    tail = output[-OUTPUT_TAIL_CHARS:]
    if code == STOPPED:
        return CheckResult(spec_id=spec.id, repo=spec.repo, title=spec.title, passed=False, skipped=True,
                           summary=_STOPPED_SUMMARY, output_tail=tail, duration_seconds=duration)
    missing = _missing_tool(code, output)
    if missing:
        return CheckResult(spec_id=spec.id, repo=spec.repo, title=spec.title, passed=False, skipped=True,
                           summary=missing, output_tail=tail, duration_seconds=duration)
    return CheckResult(spec_id=spec.id, repo=spec.repo, title=spec.title, passed=code == 0,
                       summary=_summary(code, output), output_tail=tail, duration_seconds=duration)


def _missing_tool(code: int, output: str) -> Optional[str]:
    if code == MISSING_TOOL:
        return _last_line(output) or "the check's tool is not installed"
    if code != 0:
        match = _MISSING_MODULE.search(output)
        if match and match.group(1) in _TOOL_MODULES:
            return f"{match.group(1)} is not installed"
    return None


def _summary(code: int, output: str) -> str:
    if code == TIMED_OUT:
        return _last_line(output)
    lines = [line.strip().strip("=").strip() for line in output.splitlines()]
    lines = [line for line in lines if line]
    for line in reversed(lines):
        if _COUNT_LINE.search(line):
            return line[:200]
    if lines:
        return lines[-1][:200]
    return "passed" if code == 0 else f"exit code {code}"


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def _python(root: str) -> str:
    """The checkout's own venv interpreter, else this Python.

    The lab's own Python has the lab app's BESSER and lacks the modeling agent's packages; a checkout's
    ``venv/`` (as its CLAUDE.md sets it up) imports that checkout with its own dependencies.
    """
    for candidate in ("python", "python.exe"):
        path = os.path.join(root, _VENV_SCRIPTS, candidate)
        if os.path.isfile(path):
            return path
    return sys.executable


def _python_tool(name: str, root: str) -> str:
    """The console script ``name`` of the checkout's venv, else of this Python, else ``name`` for a PATH lookup."""
    for scripts in (os.path.join(root, _VENV_SCRIPTS), sysconfig.get_path("scripts")):
        for candidate in (name, name + ".exe"):
            if os.path.isfile(os.path.join(scripts, candidate)):
                return os.path.join(scripts, candidate)
    return name
