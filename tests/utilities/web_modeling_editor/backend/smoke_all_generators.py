"""All-generators smoke test (A2 wave).

Iterates over every entry in the backend's generator registry and exercises
``generate(model, output_dir)`` on a representative v4 fixture. For each
generator the script prints PASS/FAIL/SKIP plus output sizing.

This is **not** a pytest. Run directly:

    python tests/utilities/web_modeling_editor/backend/smoke_all_generators.py

Exits 0 if no PASS=>FAIL regressions were observed (FAILs are still printed,
but the goal of this script is to surface the matrix, not to gate CI).
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# Import the FastAPI app first to trigger the full router/service init in the
# canonical order. Importing ``config.generators`` first leads to a circular
# import via deployment/github_deploy_api -> config.
from besser.utilities.web_modeling_editor.backend.backend import app  # noqa: E402,F401
from besser.utilities.web_modeling_editor.backend.config.generators import (  # noqa: E402
    SUPPORTED_GENERATORS,
    GeneratorInfo,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (  # noqa: E402
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_diagram_processor import (  # noqa: E402
    process_gui_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (  # noqa: E402
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.object_diagram_processor import (  # noqa: E402
    process_object_diagram,
)


FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "v4"
CLASS_FIXTURE = FIXTURES_DIR / "class_diagram_basic.json"
WEBAPP_FIXTURE = FIXTURES_DIR / "webapp_smoke_project.json"
AGENT_FIXTURE = FIXTURES_DIR / "agent_diagram_basic.json"
OBJECT_FIXTURE = FIXTURES_DIR / "object_diagram_basic.json"


@dataclass
class SmokeResult:
    name: str
    required: str
    status: str  # "PASS", "FAIL", "SKIP"
    output_size: int = 0
    file_count: int = 0
    notes: str = ""
    detail: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Fixture loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_class_model():
    payload = _load_json(CLASS_FIXTURE)
    return process_class_diagram(payload)


def load_webapp_models():
    """Returns (domain_model, gui_model) from the webapp project fixture."""
    raw = _load_json(WEBAPP_FIXTURE)
    class_diag = raw["diagrams"]["ClassDiagram"][0]
    gui_diag = raw["diagrams"]["GUINoCodeDiagram"][0]
    domain = process_class_diagram(class_diag)
    gui = process_gui_diagram(gui_diag["model"], class_diag["model"], domain)
    return domain, gui


def load_agent_model():
    payload = _load_json(AGENT_FIXTURE)
    return process_agent_diagram(payload)


def load_object_model():
    """Build a small ObjectModel from the v4 object diagram fixture."""
    payload = _load_json(OBJECT_FIXTURE)
    # The processor needs an explicit reference DomainModel built from the
    # embedded referenceDiagramData (mirrors the router's behavior).
    ref = payload["model"].get("referenceDiagramData") or {}
    ref_payload = {"title": payload.get("title", "Ref"), "model": ref}
    domain = process_class_diagram(ref_payload)
    obj_payload = {"title": payload.get("title"), "model": payload["model"]}
    return process_object_diagram(obj_payload, domain)


def build_minimal_quantum_circuit():
    """Construct a tiny QuantumCircuit programmatically (one H + measure)."""
    from besser.BUML.metamodel.quantum.quantum import (
        QuantumCircuit,
        HadamardGate,
        Measurement,
        ClassicalRegister,
    )

    qc = QuantumCircuit(name="SmokeCircuit", qubits=1)
    qc.add_creg(ClassicalRegister("c", 1))
    qc.add_operation(HadamardGate(target_qubit=0))
    qc.add_operation(Measurement(target_qubit=0, output_bit=0))
    return qc


def build_minimal_nn_model():
    """Build an NNModel with a tiny MLP for the NN generators."""
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
        process_nn_diagram,
    )

    payload = _load_json(FIXTURES_DIR / "nn_diagram_basic.json")
    return process_nn_diagram(payload)


def build_minimal_deployment_model():
    """Construct a DeploymentModel for Terraform.

    This is heavy: Terraform expects PublicCluster instances with a
    ``config_file`` it can read. Without one, we mark Terraform SKIP because
    the smoke harness has no CI fixture for it.
    """
    raise RuntimeError("Terraform deployment fixture not provided")


# ---------------------------------------------------------------------------
# Generator runners — each returns (success, notes, detail)
# ---------------------------------------------------------------------------


def _walk_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            out.append(os.path.join(dirpath, name))
    return out


def _python_parse_ok(out_dir: str) -> tuple[int, int]:
    py_files = [p for p in _walk_files(out_dir) if p.endswith(".py")]
    ok = 0
    for path in py_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                ast.parse(fh.read(), filename=path)
            ok += 1
        except SyntaxError:
            pass
    return ok, len(py_files)


def _output_stats(out_dir: str) -> tuple[int, int]:
    files = _walk_files(out_dir)
    total = sum(os.path.getsize(p) for p in files if os.path.isfile(p))
    return total, len(files)


def run_class_based(name: str, info: GeneratorInfo, domain_model) -> SmokeResult:
    cls = info.generator_class
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            kwargs: Dict[str, Any] = {"output_dir": tmp}
            if name == "django":
                # Django generator shells out to ``django-admin``. If the CLI
                # is not installed in the runner's PATH the generator can
                # never produce output — surface this as SKIP rather than a
                # bogus FAIL.
                import shutil as _sh
                if not _sh.which("django-admin"):
                    return SmokeResult(
                        name, info.category, "SKIP", 0, 0,
                        "django-admin CLI not installed",
                    )
                inst = cls(
                    model=domain_model,
                    project_name="smoke_proj",
                    app_name="smoke_app",
                    output_dir=tmp,
                )
                # Django shells out to django-admin and uses CWD; chdir.
                cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    inst.generate()
                finally:
                    os.chdir(cwd)
            else:
                inst = cls(domain_model, **kwargs)
                inst.generate()
            size, count = _output_stats(tmp)
            if count == 0:
                return SmokeResult(name, info.category, "FAIL", size, count, "empty output")
            ok, total_py = _python_parse_ok(tmp)
            note = f"py={ok}/{total_py}" if total_py else "no .py files"
            return SmokeResult(name, info.category, "PASS", size, count, note)
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name,
                info.category,
                "FAIL",
                0,
                0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_webapp(name: str, info: GeneratorInfo) -> SmokeResult:
    domain, gui = load_webapp_models()
    cls = info.generator_class
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(model=domain, gui_model=gui, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            if count == 0:
                return SmokeResult(name, info.category, "FAIL", size, count, "empty output")
            ok, total_py = _python_parse_ok(tmp)
            return SmokeResult(
                name,
                info.category,
                "PASS",
                size,
                count,
                f"py={ok}/{total_py}",
            )
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name,
                info.category,
                "FAIL",
                0,
                0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_react(name: str, info: GeneratorInfo) -> SmokeResult:
    domain, gui = load_webapp_models()
    cls = info.generator_class
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(domain, gui, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            return SmokeResult(name, info.category, "PASS" if count else "FAIL",
                               size, count, "" if count else "empty output")
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name,
                info.category,
                "FAIL",
                0,
                0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_flutter(name: str, info: GeneratorInfo) -> SmokeResult:
    # FlutterGenerator(model, gui_model, main_page, module=None, output_dir).
    # Use the webapp fixture's GUI model and the first screen as the main page.
    cls = info.generator_class
    try:
        domain, gui = load_webapp_models()
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            name, info.category, "FAIL", 0, 0,
            f"webapp fixture parse failed: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )

    modules = list(getattr(gui, "modules", []) or [])
    main_module = modules[0] if modules else None
    screens = []
    for mod in modules:
        screens.extend(getattr(mod, "screens", []) or [])
    if not screens:
        return SmokeResult(name, info.category, "FAIL", 0, 0,
                           "no GUI screens parsed from fixture")
    main_page = screens[0]

    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(model=domain, gui_model=gui, main_page=main_page,
                module=main_module, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            return SmokeResult(name, info.category, "PASS" if count else "FAIL",
                               size, count, "" if count else "empty output")
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name,
                info.category,
                "FAIL",
                0,
                0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_agent(name: str, info: GeneratorInfo) -> SmokeResult:
    cls = info.generator_class
    try:
        agent_model = load_agent_model()
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            name, info.category, "FAIL", 0, 0,
            f"agent fixture parse failed: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(agent_model, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            ok, total_py = _python_parse_ok(tmp)
            return SmokeResult(
                name, info.category,
                "PASS" if count else "FAIL",
                size, count,
                f"py={ok}/{total_py}" if total_py else ("" if count else "empty output"),
            )
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name, info.category, "FAIL", 0, 0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_qiskit(name: str, info: GeneratorInfo) -> SmokeResult:
    cls = info.generator_class
    try:
        qc = build_minimal_quantum_circuit()
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            name, info.category, "FAIL", 0, 0,
            f"quantum fixture build failed: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(qc, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            ok, total_py = _python_parse_ok(tmp)
            return SmokeResult(
                name, info.category,
                "PASS" if count and ok > 0 else "FAIL",
                size, count,
                f"py={ok}/{total_py}",
            )
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name, info.category, "FAIL", 0, 0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_jsonobject(name: str, info: GeneratorInfo) -> SmokeResult:
    cls = info.generator_class
    try:
        obj_model = load_object_model()
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            name, info.category, "FAIL", 0, 0,
            f"object fixture parse failed: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(obj_model, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            return SmokeResult(name, info.category, "PASS" if count else "FAIL",
                               size, count, "" if count else "empty output")
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name, info.category, "FAIL", 0, 0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_nn(name: str, info: GeneratorInfo) -> SmokeResult:
    cls = info.generator_class
    if cls is None:
        return SmokeResult(name, info.category, "SKIP", 0, 0, "dependency not installed")
    try:
        nn_model = build_minimal_nn_model()
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            name, info.category, "FAIL", 0, 0,
            f"NN fixture build failed: {type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )
    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmp:
        try:
            cls(nn_model, output_dir=tmp).generate()
            size, count = _output_stats(tmp)
            ok, total_py = _python_parse_ok(tmp)
            return SmokeResult(
                name, info.category,
                "PASS" if count else "FAIL",
                size, count,
                f"py={ok}/{total_py}",
            )
        except Exception as exc:  # noqa: BLE001
            return SmokeResult(
                name, info.category, "FAIL", 0, 0,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )


def run_terraform(name: str, info: GeneratorInfo) -> SmokeResult:
    return SmokeResult(
        name, info.category, "SKIP", 0, 0,
        "no v4 deployment fixture (DeploymentModel + .conf required)",
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


# Map generator name -> runner function. Default runner is class-based.
RUNNERS: Dict[str, Callable[[str, GeneratorInfo], SmokeResult]] = {
    "web_app": run_webapp,
    "react": run_react,
    "flutter": run_flutter,
    "agent": run_agent,
    "qiskit": run_qiskit,
    "jsonobject": run_jsonobject,
    "pytorch": run_nn,
    "tensorflow": run_nn,
    "terraform": run_terraform,
}


def main() -> int:
    print(f"[smoke] registry contains {len(SUPPORTED_GENERATORS)} generators")

    # Lazy-load class-based domain model once.
    domain_model: Optional[Any] = None
    try:
        domain_model = load_class_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FATAL: failed to load class fixture: {exc!r}", file=sys.stderr)
        return 2

    results: List[SmokeResult] = []

    for gen_name, info in SUPPORTED_GENERATORS.items():
        runner = RUNNERS.get(gen_name)
        try:
            if runner is not None:
                res = runner(gen_name, info)
            else:
                # Default: class-diagram-based generator.
                res = run_class_based(gen_name, info, domain_model)
        except Exception as exc:  # noqa: BLE001
            res = SmokeResult(
                gen_name,
                info.category,
                "FAIL",
                0,
                0,
                f"runner crash: {type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )
        results.append(res)
        print(
            f"[smoke] {res.status:5s} {gen_name:14s} cat={info.category:18s} "
            f"size={res.output_size:>9d}B files={res.file_count:>4d}  {res.notes}"
        )
        if res.status == "FAIL" and res.detail:
            for line in res.detail.splitlines()[-3:]:
                print(f"[smoke]     | {line}")

    # ----- summary -----
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("")
    print(f"[smoke] SUMMARY: {counts}")

    # markdown table for inclusion in the analysis doc
    print("")
    print("| Generator | Required diagrams | Status | Output size | Notes |")
    print("|-----------|-------------------|--------|-------------|-------|")
    for r in results:
        # Pretty-print sizes
        size = (
            f"{r.output_size}B"
            if r.output_size < 1024
            else (f"{r.output_size/1024:.1f}KiB" if r.output_size < 1024 * 1024
                  else f"{r.output_size/(1024*1024):.2f}MiB")
        )
        notes = r.notes.replace("|", "\\|")
        print(f"| `{r.name}` | {r.required} | {r.status} | {size} ({r.file_count} files) | {notes} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
