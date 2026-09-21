"""tsc must actually read the source when node_modules is absent.

Run 36e9c8a6 (2026-09-18) shipped a frontend that could not render: App.tsx
used <Link> 8 times without importing it, and rendered <PersonPage /> and six
siblings against imports bound to bare names (`import Person from ...`). 23
TS2304 errors, first render threw, #root stayed empty on every route.

Phase 3 passed it. A `"types": ["vite/client"]` entry naming an uninstalled
package makes tsc abort at CONFIG resolution -- it emits one TS2688 and
type-checks zero files. The missing-deps demotion then correctly marked that
single line advisory, and the run reported a clean frontend.

Calibration: promoting TS2304 costs 0 false positives across 1,291 known-good
files (the editor and webapp packages of BESSER's own frontend), and found a
genuine committed break there (<OnboardingChecklist>, never defined).
"""
import json
import os

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


@pytest.fixture
def orchestrator(tmp_path):
    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.output_dir = str(tmp_path)
    return orch


def _project(tmp_path, with_deps=False):
    # The real generated scaffold's options, verbatim -- es5 / node are what
    # TypeScript 7 removed.
    (tmp_path / "tsconfig.json").write_text(
        json.dumps({"compilerOptions": {
            "target": "es5", "moduleResolution": "node",
            "types": ["vite/client"]}}), encoding="utf-8")
    if with_deps:
        (tmp_path / "node_modules").mkdir()
    return str(tmp_path)


def test_a_probe_config_is_used_when_deps_are_missing(orchestrator, tmp_path):
    arg, cleanup = orchestrator._tsc_project_arg(_project(tmp_path), deps_installed=False)
    try:
        assert arg == LLMOrchestrator._TSC_PROBE_NAME
        assert (tmp_path / LLMOrchestrator._TSC_PROBE_NAME).exists()
    finally:
        cleanup()


def test_the_probe_clears_types_so_tsc_does_not_abort(orchestrator, tmp_path):
    arg, cleanup = orchestrator._tsc_project_arg(_project(tmp_path), deps_installed=False)
    try:
        body = json.loads((tmp_path / arg).read_text(encoding="utf-8"))
        assert body["compilerOptions"]["types"] == []
        assert body["extends"] == "./tsconfig.json"
    finally:
        cleanup()


def test_the_probe_overrides_options_typescript_7_removed(orchestrator, tmp_path):
    """`target: es5` / `moduleResolution: node` are TS5108 on tsc 7 -- the same
    zero-files-checked abort the probe exists to prevent. Shipped once without
    this and the probe was inert in production (run 773b8549)."""
    arg, cleanup = orchestrator._tsc_project_arg(_project(tmp_path), deps_installed=False)
    try:
        opts = json.loads((tmp_path / arg).read_text(encoding="utf-8"))["compilerOptions"]
        assert opts["target"] not in ("es5", "ES5")
        assert opts["moduleResolution"] not in ("node", "node10")
    finally:
        cleanup()


def test_the_probe_is_removed_afterwards(orchestrator, tmp_path):
    _, cleanup = orchestrator._tsc_project_arg(_project(tmp_path), deps_installed=False)
    cleanup()
    assert not (tmp_path / LLMOrchestrator._TSC_PROBE_NAME).exists()
    assert not any("besser-probe" in f for f in os.listdir(tmp_path))


def test_an_installed_project_is_checked_normally(orchestrator, tmp_path):
    arg, cleanup = orchestrator._tsc_project_arg(_project(tmp_path, True), deps_installed=True)
    cleanup()
    assert arg == "."
    assert not (tmp_path / LLMOrchestrator._TSC_PROBE_NAME).exists()


SHIPPED = "src/App.tsx(23,18): error TS2304: Cannot find name 'Link'."
PAGE = "src/App.tsx(94,20): error TS2304: Cannot find name 'PersonPage'."


def test_the_shipped_defect_survives_the_missing_deps_demotion():
    issues = []
    kept = LLMOrchestrator._demote_tsc_without_deps(
        LLMOrchestrator, [SHIPPED, PAGE], "frontend", issues)
    assert kept == [SHIPPED, PAGE], "both are real regardless of node_modules"


def test_cascade_errors_are_still_demoted():
    issues = []
    cascade = [
        "error TS2688: Cannot find type definition file for 'vite/client'.",
        "src/App.tsx(1,19): error TS2307: Cannot find module 'react'.",
        "src/x.tsx(4,9): error TS7026: JSX element implicitly has type 'any'.",
    ]
    assert LLMOrchestrator._demote_tsc_without_deps(
        LLMOrchestrator, cascade, "frontend", issues) == []
    assert any("advisory" in i for i in issues)


@pytest.mark.parametrize("name", ["expect", "describe", "vi", "process", "React", "NodeJS"])
def test_globals_a_package_would_have_provided_are_not_blockers(name):
    line = f"src/App.tsx(8,3): error TS2304: Cannot find name '{name}'."
    assert not LLMOrchestrator._is_real_undefined_name(line)


@pytest.mark.parametrize("path", [
    "src/App.test.tsx", "src/App.spec.ts", "src/__tests__/App.tsx",
    "src/Button.stories.tsx",
])
def test_test_files_are_skipped(path):
    line = f"{path}(8,3): error TS2304: Cannot find name 'renderWithRouter'."
    assert not LLMOrchestrator._is_real_undefined_name(line)


def test_a_component_missing_from_a_real_source_file_is_a_blocker():
    assert LLMOrchestrator._is_real_undefined_name(PAGE)


def test_repeated_names_collapse_so_the_cap_shows_distinct_problems():
    """8 'Link' lines used to fill the 10-line cap and hide 7 other names."""
    lines = [f"src/App.tsx({n},18): error TS2304: Cannot find name 'Link'."
             for n in range(20, 68, 3)]
    lines += [f"src/App.tsx(9{n},20): error TS2304: Cannot find name '{p}Page'."
              for n, p in enumerate(["Person", "Employee", "Guest", "Room",
                                     "Booking", "Reservedroom", "Bill"])]
    collapsed = LLMOrchestrator._collapse_repeated_names(lines)
    names = {ln.split("'")[1] for ln in collapsed}
    assert len(collapsed) == 8
    assert "Link" in names and "BillPage" in names


def test_non_ts2304_lines_pass_through_the_collapse_untouched():
    other = ["src/a.ts(1,1): error TS2307: Cannot find module './b'.",
             "src/a.ts(2,1): error TS2307: Cannot find module './c'."]
    assert LLMOrchestrator._collapse_repeated_names(other) == other
