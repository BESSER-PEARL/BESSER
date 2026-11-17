"""Phase 3 must look at the one thing the user actually sees.

A browser sweep of the recorded corpus drove seven generated
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

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation.frontend_resolution import (
    collect_frontend_resolution_issues,
)


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        from besser.spec_driven_agent.providers.llm_client import UsageTracker
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
        vite_config="import react from '@vitejs/plugin-react';\n"
                    "export default { plugins: [react()] };\n",
        files={"src/App.jsx": "export default function App() { return <div>hi</div>; }\n"},
    )

    assert collect_frontend_resolution_issues(workspace) == []


def test_a_react_plugin_that_is_declared_but_not_registered_is_still_a_blocker(tmp_path):
    """baseline p11-permits rep1: the model dropped ``plugins: [react()]`` from
    the scaffold's vite.config but kept its comment naming
    ``@vitejs/plugin-react``, and package.json still declared it. Both used to
    count as an automatic runtime; every page died with 'React is not
    defined'."""
    workspace = _frontend(
        tmp_path,
        manifest={"dependencies": dict(_MANIFEST["dependencies"], **{"@vitejs/plugin-react": "^4.3.4"})},
        vite_config=("import { defineConfig } from 'vite';\n"
                     "// @vitejs/plugin-react enables the automatic JSX runtime.\n"
                     "export default defineConfig({ server: { port: 3000 } });\n"),
        files={"src/pages/AgentList.jsx":
               "export default function AgentList(){return <div>agents</div>}\n"},
    )

    [issue] = [i for i in collect_frontend_resolution_issues(workspace)
               if i.startswith("frontend contract:")]

    assert "AgentList.jsx" in issue and "plugins" in issue


_SVGR_MANIFEST = {"dependencies": _MANIFEST["dependencies"],
                  "devDependencies": {"@vitejs/plugin-react": "^4.3.0", "vite-plugin-svgr": "^4.2.0"}}
_SVGR_IMPORTS = "import react from '@vitejs/plugin-react';\nimport svgr from 'vite-plugin-svgr';\n"


def test_a_bracket_in_an_earlier_plugins_options_does_not_hide_react(tmp_path):
    """The ``]`` of ``include: [...]`` ended the old ``plugins: [...]`` match,
    so a registered react() counted as missing and every JSX file was a
    blocker."""
    workspace = _frontend(
        tmp_path, manifest=_SVGR_MANIFEST,
        vite_config=_SVGR_IMPORTS + "export default { plugins: [svgr({ include: ['**/*.svg?react'] }), react()] };\n",
        files={"src/App.jsx": "export default function App() { return <div>hi</div>; }\n"},
    )

    assert collect_frontend_resolution_issues(workspace) == []


def test_react_imported_but_not_in_plugins_is_one_finding_naming_every_file(tmp_path):
    workspace = _frontend(
        tmp_path, manifest=_SVGR_MANIFEST,
        vite_config=_SVGR_IMPORTS + "export default { plugins: [svgr({ include: ['**/*.svg?react'] })] };\n"
                                    "export const unused = [react()];\n",
        files={"src/App.jsx": "export default function App() { return <div>hi</div>; }\n",
               "src/main.jsx": "import App from './App';\nexport const root = <App />;\n"},
    )

    [issue] = collect_frontend_resolution_issues(workspace)

    assert issue.startswith("frontend contract:") and "React is not defined" in issue
    assert "src/App.jsx" in issue and "src/main.jsx" in issue


def test_a_vite_query_suffix_resolves_to_the_file(tmp_path):
    """Vite strips ``?react``, ``?raw``, ``?url``, ``?inline``, ``?worker``."""
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        "src/logo.svg": "<svg/>", "src/help.md": "# help", "src/a.css": "a{}",
        "src/worker.js": "self.onmessage = () => {};\n",
        "src/App.jsx": "import React from 'react';\n"
                       "import Logo from './logo.svg?react';\n"
                       "import help from './help.md?raw';\n"
                       "import logoUrl from './logo.svg?url';\n"
                       "import css from './a.css?inline';\n"
                       "import W from './worker?worker';\n"
                       "export default function App() { return <Logo />; }\n",
    })

    assert collect_frontend_resolution_issues(workspace) == []


def test_a_missing_file_with_a_query_suffix_is_still_reported(tmp_path):
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        "src/App.jsx": "import React from 'react';\nimport Logo from './gone.svg?react';\n"
                       "export default function App() { return <Logo />; }\n",
    })

    [issue] = collect_frontend_resolution_issues(workspace)

    assert "'./gone.svg?react'" in issue


@pytest.mark.parametrize("target", ["src/types.d.ts", "src/types/index.d.ts", "src/types.mts",
                                    "src/types.cts"])
def test_a_type_declaration_or_mts_module_resolves(tmp_path, target):
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        target: "export interface Book { id: number }\n",
        "src/api.ts": "import type { Book } from './types';\nexport const b: Book[] = [];\n",
    })

    assert collect_frontend_resolution_issues(workspace) == []


@pytest.mark.parametrize("spec,target", [("./util.mjs", "src/util.mts"), ("./util.cjs", "src/util.cts")])
def test_a_js_specifier_resolves_to_its_typescript_source(tmp_path, spec, target):
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        target: "export const x = 1;\n",
        "src/api.ts": f"import {{ x }} from '{spec}';\nexport const y = x;\n",
    })

    assert collect_frontend_resolution_issues(workspace) == []


def test_a_computed_dynamic_import_is_not_a_missing_file(tmp_path):
    """``'./pages/' + name`` is a prefix of a path built at runtime."""
    workspace = _frontend(tmp_path, manifest=_MANIFEST, files={
        "src/pages/A.jsx": "import React from 'react';\nexport default () => <p />;\n",
        "src/App.jsx": "import React from 'react';\n"
                       "export const load = (n) => import('./pages/' + n + '.jsx');\n"
                       "export const req = (n) => require('./pages/' +n);\n",
    })

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


# Written by the model in run p5-expense-tracker rep0 (one of 8 of 24 eval runs
# with "latest" versions). npm 10.8.2, the version in the worker and grader
# images, then failed: "Cannot read properties of null (reading 'edgesOut')".
_LATEST_MANIFEST = {
    "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
    "dependencies": {"react": "latest", "react-dom": "latest",
                     "react-router-dom": "latest", "vite": "latest"},
    "devDependencies": {"@vitejs/plugin-react": "^4.3.4"},
}


def test_a_dependency_without_a_version_range_is_a_blocker_naming_it(tmp_path):
    workspace = _frontend(tmp_path, manifest=_LATEST_MANIFEST, files={})

    issues = collect_frontend_resolution_issues(workspace)

    assert len(issues) == 4, issues
    assert all(i.startswith("frontend contract: frontend/package.json dependencies ")
               for i in issues)
    assert {"'react'", "'react-dom'", "'react-router-dom'", "'vite'"} == {
        i.split()[4] for i in issues}


@pytest.mark.parametrize("version", ["latest", "*", "x", "X", "", " latest "])
def test_every_unpinned_spelling_is_flagged(tmp_path, version):
    workspace = _frontend(tmp_path, manifest={"devDependencies": {"vite": version}}, files={})

    [issue] = collect_frontend_resolution_issues(workspace)

    assert issue.startswith("frontend contract: frontend/package.json devDependencies 'vite'")


@pytest.mark.parametrize("version", ["^5.4.8", "~5.4.0", "5.x", ">=5 <6", "5.4.8",
                                     "npm:vite@^5.4.8", "file:../vite"])
def test_a_real_range_stays_quiet(tmp_path, version):
    workspace = _frontend(
        tmp_path,
        # "*" is a normal peer range, so peerDependencies are never flagged.
        manifest={"devDependencies": {"vite": version}, "peerDependencies": {"react": "*"}},
        files={})

    assert collect_frontend_resolution_issues(workspace) == []


def test_phase_3_reports_a_latest_version_as_a_blocker(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel

    workspace = _frontend(tmp_path, manifest=_LATEST_MANIFEST, files={})
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
        output_dir=workspace,
        enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )

    blockers = [i.message for i in orch._collect_validation_issues()
                if i.severity == "blocker"]

    assert [b for b in blockers if "package.json dependencies 'vite'" in b], blockers
