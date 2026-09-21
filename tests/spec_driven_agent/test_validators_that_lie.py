"""Two validators that reported the opposite of the truth.

Both were found by comparing a validation verdict against the app it
judged, on the 2026-09-17 hotel run:

  1. The missing-frontend blocker never fired on "build a web application",
     because ``\\b(web ?app|...)\\b`` cannot match when "app" continues into
     "lication". 9 of 10 sweep runs shipped a backend with no UI at all and
     reported success.

  2. ``tsc`` ran without ``node_modules``, so every package import was
     unresolvable. The run reported "3 blocker-level issue(s) remain — the
     app may not run as-is" about a frontend that boots and renders; the
     headline error was ``TS2688: Cannot find type definition file for
     'vite/client'``.

Until both were fixed no sweep number could be believed in either
direction — one under-reported, the other over-reported.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue
from besser.spec_driven_agent.providers.llm_client import UsageTracker


# ----------------------------------------------------------------------
# 1. The missing-frontend ask
# ----------------------------------------------------------------------

ASKS_FOR_A_UI = [
    # The exact phrasing that slipped through, and its neighbours.
    "build a web application to manage a small hotel",
    "a web applications suite",
    "build a webapp",
    "build a web app",
    "build a web-app",
    "i need a website for my shop",
    "with a front-end in react",
    "a front end please",
    "just a simple ui",
    "the user interface should be clean",
    "an admin dashboard",
    "a customer portal",
    "a single page application",
    "a web interface for staff",
]

DOES_NOT_ASK_FOR_A_UI = [
    "a rest api and nothing else",
    "store the data in postgres",
    "a command line tool",
    "model the hotel domain",
    # The hotel domain word that must NOT be read as "single page app".
    "the hotel has a spa, a gym and a restaurant",
]


@pytest.mark.parametrize("text", ASKS_FOR_A_UI)
def test_ui_request_is_recognised(text):
    assert LLMOrchestrator._WEBAPP_ASK_RE.search(text), text


@pytest.mark.parametrize("text", DOES_NOT_ASK_FOR_A_UI)
def test_non_ui_request_is_not_flagged(text):
    assert not LLMOrchestrator._WEBAPP_ASK_RE.search(text), text


def test_web_application_spelled_out_is_the_regression():
    """Pinning the specific miss, so a future tightening can't undo it."""
    assert LLMOrchestrator._WEBAPP_ASK_RE.search("build a web application")


# ----------------------------------------------------------------------
# 2. tsc without node_modules
# ----------------------------------------------------------------------


class _MockClient:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):  # pragma: no cover - unused
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(tmp_path):
    class _SM:
        name = "DummySM"

    return LLMOrchestrator(
        llm_client=_MockClient(), state_machines=[_SM()], output_dir=str(tmp_path)
    )


# Verbatim from the live run's tsc output, plus one genuine break.
UNINSTALLED_TREE_OUTPUT = [
    "error TS2688: Cannot find type definition file for 'vite/client'.",
    "src/App.tsx(1,19): error TS2307: Cannot find module 'react' or its corresponding type declarations.",
    "src/App.tsx(2,26): error TS2307: Cannot find module 'react-router-dom' or its corresponding type declarations.",
    "src/main.tsx(1,27): error TS2304: Cannot find name 'React'.",
    "src/pages/Booking.tsx(4,8): error TS7026: JSX element implicitly has type 'any'.",
    "src/App.tsx(3,17): error TS2307: Cannot find module './components/HotelNav' or its corresponding type declarations.",
]


def test_package_noise_is_demoted_and_the_real_break_survives(orch):
    issues: list[str] = []
    real = orch._demote_tsc_without_deps(UNINSTALLED_TREE_OUTPUT, "frontend", issues)

    # The only error that is true whether or not deps are installed: a
    # relative import naming a file the run never wrote.
    assert len(real) == 1
    assert "./components/HotelNav" in real[0]

    advisory = [i for i in issues if i.startswith("tsc-advisory [frontend]:")]
    assert len(advisory) == 6  # 5 shown + the explanatory summary line
    assert any("node_modules installed" in i for i in advisory)


def test_demoted_lines_are_warnings_not_blockers():
    """The classification is what actually stops the fix loop burning its
    budget on imports that are already correct."""
    assert _classify_issue(
        "tsc-advisory [frontend]: error TS2688: Cannot find type definition "
        "file for 'vite/client'."
    ).severity == "warning"
    # A real compile error, with deps present, stays a blocker.
    assert _classify_issue(
        "tsc [frontend]: src/pages/Room.tsx(21,9): error TS2322: Type "
        "'string' is not assignable to type 'never'."
    ).severity == "blocker"


@pytest.mark.parametrize("deps_installed", [False, True])
def test_collector_demotes_only_when_deps_are_absent(
    orch, tmp_path, monkeypatch, deps_installed
):
    """End to end through ``_collect_tsc_issues``: the same tsc output is a
    blocker on an installed tree and advisory on an uninstalled one."""
    import shutil
    import subprocess
    orch.enable_toolchain_validation = True

    project = tmp_path / "frontend"
    project.mkdir()
    (project / "tsconfig.json").write_text("{}", encoding="utf-8")
    if deps_installed:
        (project / "node_modules").mkdir()

    monkeypatch.setattr(
        shutil, "which", lambda name: "tsc" if name == "tsc" else None
    )

    class _Result:
        returncode = 2
        stdout = "\n".join(UNINSTALLED_TREE_OUTPUT)
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())

    issues = orch._collect_tsc_issues()
    severities = [_classify_issue(i).severity for i in issues]

    if deps_installed:
        assert "blocker" in severities
        assert not any(i.startswith("tsc-advisory") for i in issues)
    else:
        # Exactly one blocker survives: the relative import that is a real
        # break either way. Everything else is advisory.
        assert severities.count("blocker") == 1
        blocker = issues[severities.index("blocker")]
        assert "./components/HotelNav" in blocker
        assert any(i.startswith("tsc-advisory") for i in issues)
