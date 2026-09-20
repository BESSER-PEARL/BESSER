"""Phase 3 must look at the one thing the user actually sees.

A browser sweep of the recorded corpus on 2026-09-21 drove seven generated
apps in Chrome. Four rendered a blank page, and three of those four carried a
PERFECT probe score - because the probe boots the backend and drives HTTP and
never renders a page, so nothing in the pipeline was looking:

    library/terra  s7a5e4er   9/9 FULL   white screen, React is not defined
    inventory/qwen mjpuzh5s  10/10 FULL  white screen, ./pages/OrderView.jsx
    library/qwen   mz30st3s   9/9 FULL   white screen, ./pages/FineList.jsx
    grading/terra  v9ib6cct   5/5 FULL   white screen, React is not defined

``frontend_resolution.py`` decides all three shapes from the files on disk -
no install, no bundler, no shell - and existed for a day with no caller. This
file pins the wiring.

Calibration, stated rather than absorbed: the check fires on 24 apps the
corpus labels WORKING. That label comes from an HTTP probe that never renders
a page; four of those 24 were opened in Chrome and every one is blank. So the
24 are newly-visible real defects, not false positives, and a frontend check
tuned to agree with the HTTP label would be tuned to stay quiet about dead
UIs. ``React is not defined`` is also a RUNTIME error in a bundle that builds
cleanly, so ``frontend_build.py`` cannot substitute for the JSX rule even
where shell tools are enabled - 21 of the 24 are that shape.
"""

from __future__ import annotations

import json

import pytest

from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.validation.frontend_resolution import (
    collect_frontend_resolution_issues,
)


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        from besser.generators.llm.llm_client import UsageTracker
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


def _frontend(tmp_path, *, manifest: dict, files: dict, vite_config: str | None = None):
    project = tmp_path / "frontend"
    (project / "src" / "pages").mkdir(parents=True)
    (project / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    if vite_config is not None:
        (project / "vite.config.js").write_text(vite_config, encoding="utf-8")
    for rel, text in files.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


_MANIFEST = {"dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"}}


def test_an_import_of_a_file_that_was_never_written_is_a_blocker(tmp_path):
    """Run mjpuzh5s: ``App.jsx`` imports ``./pages/OrderView.jsx`` and the
    model never wrote it. Vite cannot resolve it, the module graph fails and
    the page never mounts - a white screen, on a 10/10 probe score."""
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        "src/App.jsx": "import React from 'react';\n"
                       "import OrderView from './pages/OrderView.jsx';\n"
                       "export default function App() { return <OrderView />; }\n",
    })

    [issue] = collect_frontend_resolution_issues(workspace)

    assert issue.startswith("frontend contract:")
    assert "./pages/OrderView.jsx" in issue


def test_jsx_without_react_and_without_an_automatic_runtime_is_a_blocker(tmp_path):
    """Run s7a5e4er: no ``vite.config``, no ``@vitejs/plugin-react``, so the
    classic transform emits ``React.createElement`` and the page dies on
    mount. The bundle BUILDS - this is the case no build check can catch."""
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        "src/App.jsx": "export default function App() { return <div>hi</div>; }\n",
    })

    [issue] = collect_frontend_resolution_issues(workspace)

    assert issue.startswith("frontend contract:")
    assert "React is not defined" in issue


def test_the_automatic_runtime_carve_out_keeps_a_working_app_quiet(tmp_path):
    """The same file is correct under ``@vitejs/plugin-react``. Nothing here
    may refuse an app that works."""
    workspace = _frontend(
        tmp_path,
        manifest={"dependencies": _MANIFEST["dependencies"],
                  "devDependencies": {"@vitejs/plugin-react": "^4.0.0"}},
        vite_config="import react from '@vitejs/plugin-react';\nexport default {};\n",
        files={"src/App.jsx": "export default function App() { return <div>hi</div>; }\n"},
    )

    assert collect_frontend_resolution_issues(workspace) == []


def test_an_undeclared_package_is_reported_but_does_not_drive_the_loop(tmp_path):
    """Run mz30st3s imported ``axios`` with no entry in ``package.json`` and
    rendered nothing. It is reported - but as a warning, because a bundler
    can still satisfy an undeclared package (hoisting, a workspace link, an
    implicit peer) and this is the one rule of the three that a filesystem
    cannot settle on its own."""
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        "src/api.js": "import axios from 'axios';\nexport default axios;\n",
    })

    [issue] = collect_frontend_resolution_issues(workspace)

    assert issue.startswith("frontend dependency:")
    assert "axios" in issue


@pytest.mark.parametrize("case", ["missing_module", "jsx_without_react"])
def test_phase_3_validation_reports_a_dead_frontend_as_a_blocker(tmp_path, case):
    """The wiring, which is the whole point: the module existed and nothing
    called it, so every one of these shipped as a verified success."""
    files = {
        "missing_module": {
            "src/App.jsx": "import React from 'react';\n"
                           "import Missing from './pages/Missing.jsx';\n"
                           "export default function App() { return <Missing />; }\n",
        },
        "jsx_without_react": {
            "src/App.jsx": "export default function App() { return <div>hi</div>; }\n",
        },
    }[case]
    from besser.BUML.metamodel.structural import Class, DomainModel

    workspace = _frontend(tmp_path, manifest=_MANIFEST, files=files)
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
        output_dir=workspace,
        enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )

    blockers = [i.message for i in orch._collect_validation_issues()
                if i.severity == "blocker"]

    assert [b for b in blockers if b.startswith("frontend contract:")
            and "App.jsx" in b], blockers


def test_the_deterministic_repair_runs_before_the_detector(tmp_path):
    """A class-only frontend never gets a Vite config from the model: across
    192 recorded class-only runs, 192 had none and 96 also had a JSX file
    with no React import - a guaranteed blank page.

    ``ensure_frontend_scaffold`` writes that config (with
    ``@vitejs/plugin-react``, which is exactly what makes the missing import
    harmless), and it has to run BEFORE the detector above, or the fix loop
    pays LLM turns to repair something the harness settles for free. Running
    it inside ``_collect_validation_issues`` rather than at packaging time
    also means a config a Phase 3 edit deleted comes back.
    """
    from besser.BUML.metamodel.structural import Class, DomainModel

    workspace = _frontend(
        tmp_path,
        manifest={"dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"},
                  "devDependencies": {"vite": "^5.0.0"},
                  "scripts": {"dev": "vite"}},
        files={"src/App.jsx": "export default function App() { return <div>hi</div>; }\n"},
    )
    # Exactly the shape the detector calls a blank page, before anything runs.
    assert [i for i in collect_frontend_resolution_issues(workspace)
            if "React is not defined" in i]

    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
        output_dir=workspace,
        enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    blockers = [i.message for i in orch._collect_validation_issues()
                if i.severity == "blocker"]

    assert (tmp_path / "frontend" / "vite.config.mjs").is_file(), (
        "the deterministic repair did not run inside the validation pass"
    )
    assert not [b for b in blockers if "React is not defined" in b], blockers
