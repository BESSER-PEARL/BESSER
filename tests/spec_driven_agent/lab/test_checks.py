"""Tests for the lab's CI checks: which checks a change gets, running them, and what the model reads."""

import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from besser.spec_driven_agent.lab.checks import (
    MISSING_TOOL,
    TIMED_OUT,
    checks_for,
    default_runner,
    format_results,
    run_checks,
    run_checks_tool,
)
from besser.spec_driven_agent.lab.contracts import (
    CheckResult,
    CheckSpec,
    FileChange,
    LabCallbacks,
    RepoChange,
    RepoSpec,
)

REAL_BESSER = Path(__file__).resolve().parents[3]


def _write(root: Path, *paths: str) -> None:
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")


def _repos(**paths) -> dict[str, RepoSpec]:
    return {key: RepoSpec(key=key, path=str(path), base_branch="development", github_owner="o", github_name=key)
            for key, path in paths.items()}


def _change(repo: str, *paths: str, status: str = "modified") -> RepoChange:
    return RepoChange(repo=repo, branch="lab/u/x", base="development",
                      files=tuple(FileChange(path=p, status=status) for p in paths))


def _by_id(specs: list[CheckSpec]) -> dict[str, CheckSpec]:
    return {spec.id: spec for spec in specs}


def _pytest_files(spec: CheckSpec) -> list[str]:
    return [arg for arg in spec.argv if arg.startswith("tests")]


def _besser_tree(root: Path) -> Path:
    _write(
        root,
        "besser/__init__.py", "besser/generators/__init__.py", "besser/generators/registry.py",
        "besser/generators/django/__init__.py", "besser/generators/django/django_generator.py",
        "besser/generators/django/templates/models.py.j2",
        "besser/generators/json/__init__.py", "besser/generators/json/json_schema_generator.py",
        "besser/spec_driven_agent/__init__.py", "besser/spec_driven_agent/agent/__init__.py",
        "besser/spec_driven_agent/agent/tool_executor.py",
        "besser/utilities/__init__.py", "besser/utilities/web_modeling_editor/__init__.py",
        "besser/utilities/web_modeling_editor/backend/__init__.py",
        "besser/utilities/web_modeling_editor/backend/routers/__init__.py",
        "besser/utilities/web_modeling_editor/backend/routers/lab_router.py",
        "tests/generators/test_generator_registry.py", "tests/generators/test_regenerable_generators.py",
        "tests/generators/django/test_django.py", "tests/generators/django/helpers.py",
        "tests/generators/django/fixtures/model.json",
        "tests/generators/json_object/test_object.py", "tests/generators/json_schema/test_schema.py",
        "tests/spec_driven_agent/test_added_generator_tools.py", "tests/spec_driven_agent/test_other.py",
        "tests/utilities/web_modeling_editor/backend/test_lab_router.py",
        "docs/source/index.rst",
    )
    (root / "docs/source/generators.rst").write_text(
        "Django\n======\n\n.. automodule:: besser.generators.django.django_generator\n   :members:\n",
        encoding="utf-8")
    return root


# --- Which checks: BESSER -------------------------------------------------


def test_a_generator_change_runs_its_own_tests_ruff_imports_and_docs(tmp_path):
    root = _besser_tree(tmp_path)
    specs = _by_id(checks_for([_change("besser", "besser/generators/django/django_generator.py")],
                              _repos(besser=root)))

    assert _pytest_files(specs["besser:tests"]) == ["tests/generators/django/test_django.py"]
    assert specs["besser:tests"].argv[:3] == (sys.executable, "-m", "pytest")
    assert specs["besser:tests"].argv[-4:] == ("-q", "--tb=short", "-p", "no:cacheprovider")
    assert specs["besser:ruff"].argv[1:] == ("check", "besser/generators/django/django_generator.py")
    assert os.path.basename(specs["besser:imports"].argv[0]).startswith("lint-imports")
    # The module is documented with automodule, so its docstrings are in the docs build.
    assert specs["besser:docs"].argv[1:] == ("docs/check_docs.py", "--offline")
    # BESSER's CI runs the schema check on every pull request.
    assert "besser:openapi" in specs


def test_a_checkout_with_its_own_venv_is_checked_with_that_venv(tmp_path):
    # The lab's own Python has the lab app's BESSER and not the modeling agent's packages.
    scripts = ("venv", "Scripts") if os.name == "nt" else ("venv", "bin")
    exe = ".exe" if os.name == "nt" else ""
    besser = _besser_tree(tmp_path / "besser")
    agent = tmp_path / "modeling-agent"
    _write(besser, f"{'/'.join(scripts)}/python{exe}", f"{'/'.join(scripts)}/ruff{exe}",
           f"{'/'.join(scripts)}/lint-imports{exe}")
    _write(agent, f"{'/'.join(scripts)}/python{exe}", "src/agent.py")

    specs = _by_id(checks_for([_change("besser", "besser/generators/django/django_generator.py",
                                       "besser/utilities/web_modeling_editor/backend/routers/lab_router.py"),
                               _change("modeling_agent", "src/agent.py")],
                              _repos(besser=besser, modeling_agent=agent)))

    besser_python = str(besser.joinpath(*scripts, f"python{exe}"))
    assert specs["besser:tests"].argv[0] == besser_python
    assert specs["besser:docs"].argv[0] == besser_python
    assert specs["besser:openapi"].argv[0] == besser_python
    assert specs["besser:ruff"].argv[0] == str(besser.joinpath(*scripts, f"ruff{exe}"))
    assert specs["besser:imports"].argv[0] == str(besser.joinpath(*scripts, f"lint-imports{exe}"))
    assert specs["modeling_agent:tests"].argv[0] == str(agent.joinpath(*scripts, f"python{exe}"))
    # No ruff in the agent's venv: the lab's own one.
    assert specs["modeling_agent:ruff"].argv[0] != str(agent.joinpath(*scripts, f"ruff{exe}"))


def test_a_template_change_maps_to_its_package_tests_without_python_lint(tmp_path):
    root = _besser_tree(tmp_path)
    specs = _by_id(checks_for([_change("besser", "besser/generators/django/templates/models.py.j2")],
                              _repos(besser=root)))

    assert _pytest_files(specs["besser:tests"]) == ["tests/generators/django/test_django.py"]
    assert set(specs) == {"besser:tests"}


def test_renamed_test_directories_are_followed(tmp_path):
    root = _besser_tree(tmp_path)
    specs = _by_id(checks_for([_change("besser", "besser/generators/json/json_schema_generator.py")],
                              _repos(besser=root)))

    assert _pytest_files(specs["besser:tests"]) == [
        "tests/generators/json_object/test_object.py", "tests/generators/json_schema/test_schema.py"]


def test_a_registry_change_always_adds_the_registry_tests(tmp_path):
    root = _besser_tree(tmp_path)
    specs = _by_id(checks_for([_change("besser", "besser/generators/registry.py")], _repos(besser=root)))

    files = _pytest_files(specs["besser:tests"])
    assert "tests/spec_driven_agent/test_added_generator_tools.py" in files
    assert "tests/generators/test_generator_registry.py" in files
    assert "tests/generators/test_regenerable_generators.py" in files
    # Only the generators' own top-level tests, not every generator's folder.
    assert "tests/generators/django/test_django.py" not in files
    assert "tests/spec_driven_agent/test_other.py" not in files


def test_a_router_change_falls_back_to_the_backend_tests_and_checks_the_api_schema(tmp_path):
    root = _besser_tree(tmp_path)
    specs = _by_id(checks_for(
        [_change("besser", "besser/utilities/web_modeling_editor/backend/routers/lab_router.py")],
        _repos(besser=root)))

    assert _pytest_files(specs["besser:tests"]) == ["tests/utilities/web_modeling_editor/backend/test_lab_router.py"]
    assert specs["besser:openapi"].argv[1:] == (".github/scripts/check_openapi.py",)


@pytest.mark.parametrize("path", [
    "besser/spec_driven_agent/lab/contracts.py",                # ChangeKind is a schema component
    "besser/BUML/metamodel/traceability/traceability.py",       # so is RequirementKind
])
def test_a_module_outside_routers_and_models_still_checks_the_api_schema(tmp_path, path):
    root = _besser_tree(tmp_path)
    assert "besser:openapi" in _by_id(checks_for([_change("besser", path)], _repos(besser=root)))


def test_an_agent_module_without_a_mirror_maps_to_the_flat_tests_that_import_it(tmp_path):
    root = _besser_tree(tmp_path)
    (root / "tests/spec_driven_agent/test_tools.py").write_text(
        "from besser.spec_driven_agent.agent.tools import get_tools_for\n", encoding="utf-8")
    (root / "tests/spec_driven_agent/test_executor.py").write_text(
        "from besser.spec_driven_agent.agent import (\n    compaction, tool_executor)\n"
        "from besser.spec_driven_agent.agent import tool_executor\n", encoding="utf-8")
    (root / "tests/spec_driven_agent/test_tools_extra.py").write_text(
        "import besser.spec_driven_agent.agent.tools_extra\n", encoding="utf-8")

    def tests_for(path):
        return _pytest_files(_by_id(checks_for([_change("besser", path)], _repos(besser=root)))["besser:tests"])

    assert tests_for("besser/spec_driven_agent/agent/tools.py") == ["tests/spec_driven_agent/test_tools.py"]
    assert tests_for("besser/spec_driven_agent/agent/tool_executor.py") == ["tests/spec_driven_agent/test_executor.py"]


def test_a_file_without_tests_is_reported_not_dropped(tmp_path):
    root = _besser_tree(tmp_path)
    specs = checks_for([_change("besser", "besser/spec_driven_agent/agent/tool_executor.py")], _repos(besser=root))
    by_id = _by_id(specs)

    # The walk never climbs into the whole tests/spec_driven_agent area.
    assert "besser:tests" not in by_id
    missing = by_id["besser:tests-missing"]
    assert missing.argv == ()
    assert missing.title == "No tests found for besser/spec_driven_agent/agent/tool_executor.py"

    results = run_checks(specs, _repos(besser=root), runner=lambda spec, cwd: (0, "All checks passed!"))
    reported = next(r for r in results if r.spec_id == "besser:tests-missing")
    assert reported.skipped and not reported.passed
    assert "tool_executor.py" in reported.summary


def _utilities_tree(root: Path) -> Path:
    _write(root, "besser/__init__.py", "besser/utilities/__init__.py", "besser/utilities/old.py",
           "tests/utilities/test_old.py", "tests/utilities/test_other.py", "tests/utilities/test_model_stats.py")
    (root / "tests/utilities/test_uses.py").write_text("from besser.utilities import model_stats, old\n",
                                                       encoding="utf-8")
    return root


def test_a_new_module_runs_its_own_test_and_the_tests_importing_it_not_its_whole_package(tmp_path):
    # Live run B: a new besser/utilities/model_stats.py ran all 19 utilities test files. Tests that existed
    # before the module cannot use it except through a changed module, whose own tests are selected.
    root = _utilities_tree(tmp_path)
    repos = _repos(besser=root)

    def tests_for(path, status):
        specs = _by_id(checks_for([_change("besser", path, status=status)], repos))
        return _pytest_files(specs["besser:tests"]) if "besser:tests" in specs else specs

    assert tests_for("besser/utilities/model_stats.py", "added") == [
        "tests/utilities/test_model_stats.py", "tests/utilities/test_uses.py"]
    # A changed module can break any test of its package, so it keeps the package's tests.
    assert tests_for("besser/utilities/old.py", "modified") == [
        "tests/utilities/test_model_stats.py", "tests/utilities/test_old.py", "tests/utilities/test_other.py",
        "tests/utilities/test_uses.py"]
    lonely = tests_for("besser/utilities/lonely.py", "added")
    assert lonely["besser:tests-missing"].title == "No tests found for besser/utilities/lonely.py"


def test_changed_tests_run_themselves_and_deleted_ones_do_not(tmp_path):
    root = _besser_tree(tmp_path)
    changes = [RepoChange(repo="besser", branch="lab/u/x", base="development", files=(
        FileChange("tests/spec_driven_agent/test_other.py", "added"),
        FileChange("tests/spec_driven_agent/test_removed.py", "deleted"),
        FileChange("tests/generators/django/fixtures/model.json", "modified"),
    ))]
    specs = _by_id(checks_for(changes, _repos(besser=root)))

    assert _pytest_files(specs["besser:tests"]) == [
        "tests/generators/django/test_django.py", "tests/spec_driven_agent/test_other.py"]
    # Tests are not under besser/, so no ruff or import check.
    assert set(specs) == {"besser:tests"}


def test_ruff_skips_deleted_files_but_imports_still_run(tmp_path):
    root = _besser_tree(tmp_path)
    specs = _by_id(checks_for([_change("besser", "besser/generators/django/old.py", status="deleted")],
                              _repos(besser=root)))

    assert "besser:ruff" not in specs
    assert "besser:imports" in specs


def test_docs_gate_runs_for_docs_and_documented_modules_only(tmp_path):
    root = _besser_tree(tmp_path)
    repos = _repos(besser=root)

    assert "besser:docs" in _by_id(checks_for([_change("besser", "docs/source/index.rst")], repos))
    undocumented = checks_for([_change("besser", "besser/spec_driven_agent/agent/tool_executor.py")], repos)
    assert "besser:docs" not in _by_id(undocumented)


def test_the_real_besser_tree_maps_as_ci_lays_it_out():
    """Calibration against the working copy itself: the mapping must find the tests that exist."""
    repos = _repos(besser=REAL_BESSER)

    django = _by_id(checks_for([_change("besser", "besser/generators/django/django_generator.py")], repos))
    assert _pytest_files(django["besser:tests"])
    assert all(p.startswith("tests/generators/django/") for p in _pytest_files(django["besser:tests"]))

    json_files = _pytest_files(_by_id(checks_for(
        [_change("besser", "besser/generators/json/json_schema_generator.py")], repos))["besser:tests"])
    assert any(p.startswith("tests/generators/json_object/") for p in json_files)
    assert any(p.startswith("tests/generators/json_schema/") for p in json_files)

    registry = _pytest_files(_by_id(checks_for([_change("besser", "besser/generators/registry.py")],
                                               repos))["besser:tests"])
    assert {"tests/generators/test_generator_registry.py", "tests/generators/test_regenerable_generators.py",
            "tests/spec_driven_agent/test_added_generator_tools.py"} <= set(registry)
    for path in registry:
        assert (REAL_BESSER / path).is_file()

    # tests/spec_driven_agent is flat: the agent's tool surface maps to the tests importing it.
    tools = _pytest_files(_by_id(checks_for([_change("besser", "besser/spec_driven_agent/agent/tools.py")],
                                            repos))["besser:tests"])
    assert "tests/spec_driven_agent/test_added_generator_tools.py" in tools

    # A new utility nothing imports yet has no tests, rather than all of tests/utilities.
    new = _by_id(checks_for([_change("besser", "besser/utilities/model_stats.py", status="added")], repos))
    assert "besser:tests" not in new and "besser:tests-missing" in new


# --- Which checks: editor and modeling agent ------------------------------


def _frontend_tree(root: Path) -> Path:
    _write(
        root,
        "packages/webapp/src/main/features/lab/LabPage.tsx",
        "packages/webapp/src/main/features/lab/__tests__/LabPage.test.tsx",
        "packages/webapp/src/main/features/lab/hooks/__tests__/useLab.test.ts",
        "packages/webapp/src/main/features/other/__tests__/other.test.ts",
        "packages/webapp/src/main/shared/perspectives.ts",
        "packages/webapp/src/main/shared/__tests__/perspectives.test.ts",
        "packages/webapp/src/main/shared/api/__tests__/lab.test.ts",
        "packages/webapp/src/components/chatbot-kit/ui/lab-run-result.tsx",
        "packages/webapp/src/components/chatbot-kit/ui/__tests__/chat-message.lab.test.tsx",
        "packages/webapp/src/components/chatbot-kit/hooks/__tests__/hook.test.ts",
    )
    return root


def test_an_editor_feature_change_runs_that_feature_tests_lint_and_types(tmp_path):
    root = _frontend_tree(tmp_path)
    specs = _by_id(checks_for([_change("frontend", "packages/webapp/src/main/features/lab/LabPage.tsx")],
                              _repos(frontend=root)))

    tests = specs["frontend:tests"]
    assert tests.cwd == "packages/webapp"
    assert tests.argv == ("npx", "vitest", "run", "src/main/features/lab/__tests__/LabPage.test.tsx",
                          "src/main/features/lab/hooks/__tests__/useLab.test.ts")
    assert specs["frontend:lint"].argv == ("npm", "run", "lint")
    assert specs["frontend:types"].argv == ("npm", "run", "typecheck")
    assert "frontend:i18n" not in specs


def test_other_editor_files_map_to_the_tests_next_to_them(tmp_path):
    root = _frontend_tree(tmp_path)
    specs = _by_id(checks_for([_change("frontend", "packages/webapp/src/components/chatbot-kit/ui/lab-run-result.tsx",
                                       "packages/webapp/src/main/shared/perspectives.ts")],
                              _repos(frontend=root)))

    assert specs["frontend:tests"].argv[3:] == (
        "src/components/chatbot-kit/ui/__tests__/chat-message.lab.test.tsx",
        "src/main/shared/__tests__/perspectives.test.ts")


def test_translations_get_the_i18n_check_and_editor_package_files_are_reported(tmp_path):
    root = _frontend_tree(tmp_path)
    specs = _by_id(checks_for([_change("frontend", "packages/i18n/en/webapp.json",
                                       "packages/editor/src/main/packages/components.ts")],
                              _repos(frontend=root)))

    assert specs["frontend:i18n"].argv == ("node", "scripts/i18n-check.mjs")
    assert specs["frontend:types"].argv == ("npm", "run", "typecheck")
    assert specs["frontend:editor-tests"].title == (
        "No tests run for packages/editor/src/main/packages/components.ts: the editor package has no test "
        "runner and CI runs none of its tests; tests of editor code go in packages/webapp")
    assert specs["frontend:editor-tests"].argv == ()
    assert "frontend:lint" not in specs


def test_a_test_file_of_the_editor_package_is_reported_as_never_run(tmp_path):
    # packages/editor/src/tests/ holds jest-era tests that no script or CI job runs.
    root = _frontend_tree(tmp_path)
    specs = _by_id(checks_for([_change("frontend", "packages/editor/src/tests/unit/merge-test.ts")],
                              _repos(frontend=root)))

    assert specs["frontend:editor-tests"].title.startswith(
        "No tests run for packages/editor/src/tests/unit/merge-test.ts: the editor package has no test runner")
    assert "frontend:tests" not in specs and "frontend:tests-missing" not in specs


def test_an_editor_schema_change_runs_the_api_schema_check_in_besser(tmp_path):
    besser = _besser_tree(tmp_path / "besser")
    frontend = _frontend_tree(tmp_path / "frontend")
    specs = _by_id(checks_for([_change("frontend", "packages/webapp/openapi/besser-openapi.json")],
                              _repos(besser=besser, frontend=frontend)))

    assert specs["besser:openapi"].repo == "besser"
    # The editor CI also checks that the generated d.ts matches the schema JSON.
    assert specs["frontend:api-types"].cwd == "packages/webapp"
    assert specs["frontend:api-types"].argv[:2] == ("node", "-e")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("committed, passed", [("export type Paths = 1;\n", True), ("stale\n", False)])
def test_the_api_types_check_compares_the_generated_types_with_the_committed_file(tmp_path, committed, passed):
    webapp = tmp_path / "packages/webapp"
    files = {
        # A stand-in openapi-typescript CLI that prints the types of the schema it is given.
        "node_modules/openapi-typescript/package.json": '{"name": "openapi-typescript"}',
        "node_modules/openapi-typescript/bin/cli.js": (
            "process.stdout.write(require('fs').readFileSync(process.argv[2], 'utf8') === '{}' ? "
            "'export type Paths = 1;\\n' : 'other');\n"),
        "openapi/besser-openapi.json": "{}",
        "src/main/shared/api/generated/besser-openapi.d.ts": committed,
    }
    for path, text in files.items():
        _write(webapp, path)
        (webapp / path).write_bytes(text.encode("utf-8"))
    specs = _by_id(checks_for([_change("frontend", "packages/webapp/openapi/besser-openapi.json")],
                              _repos(frontend=tmp_path)))

    [result] = run_checks([specs["frontend:api-types"]], _repos(frontend=tmp_path))

    assert result.passed is passed and not result.skipped
    assert ("matches" if passed else "is stale") in result.output_tail


def test_a_modeling_agent_change_runs_its_suite_and_ruff_as_ci(tmp_path):
    specs = _by_id(checks_for([_change("modeling_agent", "src/app/agent_setup.py")],
                              _repos(modeling_agent=tmp_path)))

    assert specs["modeling_agent:tests"].argv[1:] == (
        "-m", "pytest", "tests", "-q", "--tb=short", "--ignore=tests/live", "-p", "no:cacheprovider")
    assert specs["modeling_agent:ruff"].argv[1:] == ("check", "src", "patches", "modeling_agent.py")
    # The agent's CI lint job also runs lint-imports (contracts in its pyproject.toml).
    assert os.path.basename(specs["modeling_agent:imports"].argv[0]).startswith("lint-imports")
    assert len(specs["modeling_agent:imports"].argv) == 1
    assert checks_for([_change("modeling_agent", "docs/source/index.md")], _repos(modeling_agent=tmp_path)) == []
    tests_only = _by_id(checks_for([_change("modeling_agent", "tests/test_x.py")], _repos(modeling_agent=tmp_path)))
    assert "modeling_agent:imports" not in tests_only


def test_full_runs_each_touched_repository_whole_ci(tmp_path):
    repos = _repos(besser=tmp_path, frontend=tmp_path, modeling_agent=tmp_path)
    specs = checks_for([_change("frontend", "README.md"), _change("besser", "setup.cfg")], repos, full=True)

    ids = [spec.id for spec in specs]
    # Editor first: the pull requests open in that order.
    assert ids == ["frontend:lint", "frontend:types", "frontend:tests", "frontend:server-tests", "frontend:i18n",
                   "frontend:api-types", "besser:ruff", "besser:imports", "besser:tests", "besser:docs",
                   "besser:openapi"]
    by_id = _by_id(specs)
    assert by_id["besser:tests"].argv[1:4] == ("-m", "pytest", "tests")
    assert by_id["besser:ruff"].argv[1:] == ("check", "besser/")
    assert by_id["frontend:tests"].argv == ("npm", "test", "--workspace=webapp")


def test_no_changes_no_checks(tmp_path):
    assert checks_for([_change("besser")], _repos(besser=tmp_path)) == []


# --- Running and reporting ------------------------------------------------


def _spec(spec_id: str = "besser:tests", argv=("tool",), cwd: str = ".", timeout: float = 30) -> CheckSpec:
    return CheckSpec(id=spec_id, repo="besser", title="Tests of the touched areas", argv=argv, cwd=cwd,
                     timeout_seconds=timeout)


def test_run_checks_reads_results_from_the_runner(tmp_path):
    seen = []

    def runner(spec, cwd):
        seen.append((spec.id, cwd))
        if spec.id == "besser:tests":
            return 1, "\x1b[31mFAILED tests/test_a.py::test_x\x1b[0m\n1 failed, 2 passed in 0.5s\n"
        return 0, "All checks passed!\n"

    phases = []
    results = run_checks([_spec(cwd="docs"), _spec("besser:ruff")], _repos(besser=tmp_path), runner=runner,
                         callbacks=LabCallbacks(on_phase=lambda phase, message: phases.append(phase)))

    assert seen == [("besser:tests", str(tmp_path / "docs")), ("besser:ruff", str(tmp_path))]
    failed, passed = results
    assert not failed.passed and not failed.skipped
    assert failed.summary == "1 failed, 2 passed in 0.5s"
    assert "\x1b" not in failed.output_tail
    assert passed.passed and passed.summary == "All checks passed!"
    assert phases == ["check", "check"]


def test_a_missing_tool_is_a_skip_that_names_it(tmp_path):
    results = run_checks(
        [_spec("besser:imports"), _spec("besser:docs")], _repos(besser=tmp_path),
        runner=lambda spec, cwd: (MISSING_TOOL, "lint-imports is not installed (not found on PATH)")
        if spec.id == "besser:imports"
        else (1, "Extension error:\nCould not import extension sphinx_immaterial "
                 "(exception: No module named 'sphinx_immaterial')"))

    assert [(r.skipped, r.passed) for r in results] == [(True, False), (True, False)]
    assert results[0].summary == "lint-imports is not installed (not found on PATH)"
    assert results[1].summary == "sphinx_immaterial is not installed"


def test_a_missing_module_of_the_code_under_test_is_a_failure(tmp_path):
    [result] = run_checks([_spec()], _repos(besser=tmp_path),
                          runner=lambda spec, cwd: (2, "ModuleNotFoundError: No module named 'besser_typo'\n"
                                                       "1 error in 0.1s"))

    assert not result.passed and not result.skipped


def test_run_checks_stops_when_the_run_is_cancelled(tmp_path):
    calls = []
    answers = iter([True, False])
    results = run_checks([_spec("a"), _spec("b")], _repos(besser=tmp_path),
                         runner=lambda spec, cwd: calls.append(spec.id) or (0, "ok"),
                         callbacks=LabCallbacks(should_continue=lambda: next(answers)))

    assert calls == ["a"]
    assert [r.spec_id for r in results] == ["a"]


def test_default_runner_runs_the_command_with_combined_output(tmp_path):
    code = "import sys; print('3 passed in 0.1s'); print('warn', file=sys.stderr)"
    [result] = run_checks([_spec(argv=(sys.executable, "-c", code))], _repos(besser=tmp_path))

    assert result.passed
    assert result.summary == "3 passed in 0.1s"
    assert "warn" in result.output_tail


def test_default_runner_hides_the_server_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("BESSER_LAB_LLM_API_KEY", "sk-should-not-leak")
    code = "import os; print(os.environ.get('BESSER_LAB_LLM_API_KEY', 'absent'))"

    exit_code, output = default_runner(_spec(argv=(sys.executable, "-c", code)), str(tmp_path))

    assert exit_code == 0
    assert output.strip() == "absent"


# The lab backend's secrets, as its environment holds them (names from settings.py and the deploy's lab.env).
_PLANTED_SECRETS = {
    "BESSER_LAB_LLM_API_KEY": "planted-llm-key",
    "ANTHROPIC_API_KEY": "planted-anthropic-key",
    "GITHUB_CLIENT_SECRET": "planted-oauth-secret",
    "GITHUB_CLIENT_ID": "planted-oauth-client-id",
    "BESSER_SESSION_SECRET": "planted-session-secret",
    "GITHUB_TOKEN": "planted-user-token",
    "BESSER_TELEMETRY_ADMIN_TOKEN": "planted-admin-token",
}


def test_a_check_cannot_print_the_lab_secrets_from_its_environment(tmp_path, monkeypatch):
    """Model-written code runs inside the checks: dumping its environment must show none of the secrets."""
    for name, value in _PLANTED_SECRETS.items():
        monkeypatch.setenv(name, value)
    dump = "import os; print('\\n'.join(f'{k}={v}' for k, v in os.environ.items())); print('1 passed')"

    [result] = run_checks([_spec(argv=(sys.executable, "-c", dump))], _repos(besser=tmp_path))

    assert result.passed
    assert "PATH=" in result.output_tail.upper()
    assert [value for value in _PLANTED_SECRETS.values() if value in result.output_tail] == []


def test_default_runner_reports_a_missing_executable(tmp_path):
    exit_code, output = default_runner(_spec(argv=("besser-lab-no-such-tool",)), str(tmp_path))

    assert exit_code == MISSING_TOOL
    assert "besser-lab-no-such-tool is not installed" in output


def test_default_runner_stops_a_check_at_its_timeout(tmp_path):
    spec = _spec(argv=(sys.executable, "-c", "import time; time.sleep(30)"), timeout=1)

    exit_code, output = default_runner(spec, str(tmp_path))
    [result] = run_checks([spec], _repos(besser=tmp_path))

    assert exit_code == TIMED_OUT
    assert not result.passed and result.summary == "Timed out after 1 s"


def test_a_stop_ends_a_running_check_and_the_processes_it_started(tmp_path):
    # Live run C (D3): a stop during pytest waited for pytest to end; a test check may run for 30 minutes.
    beat = tmp_path / "beat"
    child = (f"import pathlib, time\np = pathlib.Path({str(beat)!r})\n"
             "while True:\n    p.write_text(str(time.time()))\n    time.sleep(0.1)\n")
    parent = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}])\ntime.sleep(60)\n"
    started = time.monotonic()
    stop_at = started + 3

    [result] = run_checks([_spec(argv=(sys.executable, "-c", parent), timeout=30)], _repos(besser=tmp_path),
                          callbacks=LabCallbacks(should_continue=lambda: time.monotonic() < stop_at))

    assert time.monotonic() - started < 15
    assert result.skipped and not result.passed
    assert result.summary == "Stopped: the run was stopped before this check finished"
    # The process the check started is gone too: its heartbeat no longer moves.
    assert beat.is_file()
    time.sleep(0.5)
    last = beat.read_text()
    time.sleep(1)
    assert beat.read_text() == last


def test_format_results_puts_failures_first_and_bounds_their_output():
    results = [
        CheckResult(spec_id="besser:ruff", repo="besser", title="Ruff", passed=True, summary="All checks passed!"),
        CheckResult(spec_id="besser:tests-missing", repo="besser", title="No tests found for x.py", passed=False,
                    skipped=True, summary="No tests found for x.py"),
        CheckResult(spec_id="besser:tests", repo="besser", title="Tests", passed=False, summary="1 failed",
                    output_tail="x" * 50_000 + "\nAssertionError: the real reason"),
    ]

    text = format_results(results)

    assert text.startswith("Checks: 1 failed, 1 skipped, 1 passed.")
    assert text.index("FAILED besser:tests") < text.index("SKIPPED besser:tests-missing") < text.index("passed besser:ruff")
    assert "AssertionError: the real reason" in text
    assert "[... earlier output cut]" in text
    assert len(text) < 5_000


def test_format_results_without_checks():
    assert format_results([]) == "No checks ran: nothing changed that a check covers."


def test_the_tool_runs_the_checks_of_the_current_changes(tmp_path):
    """End to end with the default runner: a modeling-agent-shaped repository with one passing test."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    changes = [[], [_change("modeling_agent", "tests/test_ok.py")]]
    tool = run_checks_tool(lambda: changes.pop(0), _repos(modeling_agent=tmp_path))

    assert tool.schema["name"] == "run_checks"
    assert tool.writes
    assert tool.handler({}) == "No checks ran: nothing changed that a check covers."
    text = tool.handler({})
    assert "passed modeling_agent:tests: Test suite - 1 passed" in text


def test_the_tool_streams_its_checks_and_stops_when_the_run_is_cancelled(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    changes = [_change("modeling_agent", "tests/test_ok.py")]
    phases, running = [], [True]
    callbacks = LabCallbacks(on_phase=lambda phase, message: phases.append(phase),
                             should_continue=lambda: running[0])
    tool = run_checks_tool(lambda: changes, _repos(modeling_agent=tmp_path), callbacks=callbacks)

    assert "1 passed" in tool.handler({})
    streamed = len(phases)
    assert streamed and set(phases) == {"check"}

    running[0] = False
    assert tool.handler({}) == "The run was stopped; the checks did not finish."
    assert len(phases) == streamed


def test_the_tool_runs_its_checks_with_the_runner_it_was_given(tmp_path):
    # The session passes a runner bounded by the run's time left; the tool must not fall back to the default.
    changes = [_change("modeling_agent", "tests/test_ok.py")]
    ran = []

    def runner(spec, cwd):
        ran.append(spec.id)
        return 0, "7 passed"

    repos = _repos(modeling_agent=tmp_path)
    tool = run_checks_tool(lambda: changes, repos, runner=runner)

    assert "passed modeling_agent:tests: Test suite - 7 passed" in tool.handler({})
    assert ran == [spec.id for spec in checks_for(changes, repos)] and "modeling_agent:tests" in ran
