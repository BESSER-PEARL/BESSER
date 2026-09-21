"""A kotlinc compile that never finished must not read as "compiled clean".

``except (subprocess.TimeoutExpired, OSError): continue`` made a module whose
180s compile timed out byte-identical, in the report, to one that compiled with
zero errors. The run then declared "0 blockers" having compiled nothing -- what
this module's own docstring forbids. ruff, tsc and cargo all report their
timeout through ``_check_did_not_run``; kotlinc was the sole collector that
swallowed it.

It stays a WARNING, not a blocker: a check that did not run is unknown, not
proof of a defect. The point is that the unknown is visible.
"""
import os
import shutil
import subprocess

import pytest

from besser.spec_driven_agent.validation import toolchain as tc


@pytest.fixture
def kotlin_module(tmp_path):
    """The layout _collect_kotlinc_issues looks for: Gradle file + src/main/kotlin."""
    mod = tmp_path / "svc"
    src = mod / "src" / "main" / "kotlin"
    src.mkdir(parents=True)
    (mod / "build.gradle.kts").write_text('plugins { kotlin("jvm") }\n')
    (src / "App.kt").write_text("fun main() {}\n")
    return str(tmp_path)


@pytest.fixture
def kotlinc_on_path(monkeypatch):
    """`import shutil as _shutil` is function-local, so patch the module itself."""
    real = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: "/tools/kotlinc" if "kotlinc" in name else real(name, *a, **k),
    )


def _raise(exc):
    def _run(*_a, **_k):
        raise exc
    return _run


@pytest.mark.parametrize("exc, fragment", [
    (subprocess.TimeoutExpired("kotlinc", 180), "timed out after 180s"),
    (OSError("kotlinc vanished"), "could not be launched"),
])
def test_a_compile_that_never_ran_is_reported(
    monkeypatch, kotlin_module, kotlinc_on_path, exc, fragment
):
    """The regression: both arms used to `continue` with an empty issue list."""
    monkeypatch.setattr(subprocess, "run", _raise(exc))

    issues = tc._collect_kotlinc_issues(kotlin_module)

    assert len(issues) == 1, issues
    assert "kotlinc [svc/src/main/kotlin]" in issues[0]
    assert fragment in issues[0]
    assert "SKIPPED" in issues[0]


def test_the_report_names_the_module_that_did_not_compile(
    monkeypatch, tmp_path, kotlinc_on_path
):
    """One module timing out must not implicate, or excuse, its peers."""
    for name in ("alpha", "beta"):
        src = tmp_path / name / "src" / "main" / "kotlin"
        src.mkdir(parents=True)
        (tmp_path / name / "build.gradle.kts").write_text("plugins { }\n")
        (src / "App.kt").write_text("fun main() {}\n")

    monkeypatch.setattr(
        subprocess, "run", _raise(subprocess.TimeoutExpired("kotlinc", 180)))

    issues = tc._collect_kotlinc_issues(str(tmp_path))

    assert len(issues) == 2
    assert {"alpha", "beta"} == {i.split("[")[1].split("/")[0] for i in issues}


def test_it_stays_a_warning_not_a_blocker(monkeypatch, kotlin_module, kotlinc_on_path):
    """`kotlinc errors` are blockers; "we did not look" must not be promoted to one.

    A rollback ranked on an unknown would discard a working tree.
    """
    from besser.spec_driven_agent.validation.issues import _classify_issue

    monkeypatch.setattr(
        subprocess, "run", _raise(subprocess.TimeoutExpired("kotlinc", 180)))
    issue = tc._collect_kotlinc_issues(kotlin_module)[0]

    assert _classify_issue(issue).severity == "warning"


def test_no_kotlin_module_means_nothing_to_report(tmp_path, kotlinc_on_path):
    """The soft-skip is deliberate: not every project is Kotlin."""
    (tmp_path / "main.py").write_text("print(1)\n")
    assert tc._collect_kotlinc_issues(str(tmp_path)) == []


def test_kotlinc_absent_stays_silent(monkeypatch, kotlin_module):
    """Documented soft-skip: the bench host either has kotlinc or it doesn't."""
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    assert tc._collect_kotlinc_issues(kotlin_module) == []


def test_a_successful_compile_still_reports_its_errors(
    monkeypatch, kotlin_module, kotlinc_on_path
):
    """The happy path must be untouched: real diagnostics still come through."""
    class _Result:
        returncode = 1
        stdout = ""
        stderr = "App.kt:1:5: error: unresolved reference: foo\n"

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Result())

    issues = tc._collect_kotlinc_issues(kotlin_module)

    assert any("unresolved reference" in i for i in issues), issues
    assert not any("SKIPPED" in i for i in issues), issues
