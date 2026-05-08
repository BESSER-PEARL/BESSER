"""Backend smoke test: WebApp generation E2E from a v4 ProjectInput fixture.

This is NOT a pytest. It's a runnable script that exercises the same path the
frontend's `useGenerateCode` hook will hit after the Phase 7 v4 cutover:
ProjectInput JSON -> process_class_diagram + process_gui_diagram -> WebAppGenerator.

Run with:

    python tests/utilities/web_modeling_editor/backend/smoke_webapp_generation.py

Exits 0 on success, non-zero on failure. Prints a structured report to stdout.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from besser.utilities.web_modeling_editor.backend.models import (  # noqa: E402
    DiagramInput,
)
from besser.utilities.web_modeling_editor.backend.models.project import (  # noqa: E402
    ProjectInput,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (  # noqa: E402
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_diagram_processor import (  # noqa: E402
    process_gui_diagram,
)
from besser.generators.web_app import WebAppGenerator  # noqa: E402


FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "v4" / "webapp_smoke_project.json"


def _load_project() -> ProjectInput:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return ProjectInput(**raw)


def _walk_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            out.append(os.path.join(dirpath, name))
    return out


def _human_bytes(total: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if total < 1024.0:
            return f"{total:.1f} {unit}"
        total = int(total / 1024)
    return f"{total} TiB"


def main() -> int:
    print(f"[smoke] loading fixture: {FIXTURE_PATH}")
    project = _load_project()

    # Mirror the router's project -> WebApp pipeline in
    # _handle_web_app_project_generation (generation_router.py).
    gui_diagram: DiagramInput | None = project.get_active_diagram("GUINoCodeDiagram")
    if gui_diagram is None:
        print("[smoke] FAIL: no GUINoCodeDiagram in fixture", file=sys.stderr)
        return 2

    class_diagram: DiagramInput | None = project.get_referenced_diagram(
        gui_diagram, "ClassDiagram"
    )
    if class_diagram is None:
        print("[smoke] FAIL: no referenced ClassDiagram", file=sys.stderr)
        return 2

    print("[smoke] processing ClassDiagram (v4 nodes/edges) -> DomainModel ...")
    domain_model = process_class_diagram(class_diagram.model_dump())
    classes = list(getattr(domain_model, "get_classes", lambda: [])())
    print(f"[smoke]   classes: {[c.name for c in classes]}")
    if len(classes) < 2:
        print(
            "[smoke] FAIL: expected at least 2 classes from ClassDiagram fixture",
            file=sys.stderr,
        )
        return 2

    print("[smoke] processing GUINoCodeDiagram -> GUIModel ...")
    gui_model = process_gui_diagram(
        gui_diagram.model, class_diagram.model, domain_model
    )
    modules = list(getattr(gui_model, "modules", []) or [])
    screens = []
    for mod in modules:
        screens.extend(getattr(mod, "screens", []) or [])
    print(
        f"[smoke]   modules={len(modules)} screens={len(screens)} "
        f"screen_names={[getattr(s, 'name', '?') for s in screens]}"
    )
    if not screens:
        print("[smoke] FAIL: no screens parsed from GUI fixture", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="besser_smoke_") as tmpdir:
        print(f"[smoke] running WebAppGenerator -> {tmpdir}")
        gen = WebAppGenerator(
            model=domain_model, gui_model=gui_model, output_dir=tmpdir
        )
        try:
            gen.generate()
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] FAIL: generator raised: {exc!r}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 3

        files = _walk_files(tmpdir)
        total_bytes = sum(os.path.getsize(p) for p in files if os.path.isfile(p))
        print(f"[smoke] generated {len(files)} files / {_human_bytes(total_bytes)}")
        if not files:
            print("[smoke] FAIL: output directory is empty", file=sys.stderr)
            return 3

        backend_dir = os.path.join(tmpdir, "backend")
        frontend_dir = os.path.join(tmpdir, "frontend")
        if not os.path.isdir(backend_dir):
            print("[smoke] FAIL: no backend/ subdir produced", file=sys.stderr)
            return 3
        if not os.path.isdir(frontend_dir):
            print("[smoke] FAIL: no frontend/ subdir produced", file=sys.stderr)
            return 3

        # Parse every Python file in backend/ with ast — at least one must
        # parse cleanly, and any failure is reported as a smoke failure.
        py_files = [
            p
            for p in _walk_files(backend_dir)
            if p.endswith(".py") and os.path.isfile(p)
        ]
        py_ok = 0
        py_failures: list[tuple[str, str]] = []
        for path in py_files:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    src = fh.read()
                ast.parse(src, filename=path)
                py_ok += 1
            except SyntaxError as exc:
                py_failures.append((path, f"{exc.lineno}: {exc.msg}"))
        print(f"[smoke]   backend python files: {py_ok}/{len(py_files)} parse OK")
        for path, why in py_failures[:5]:
            print(f"[smoke]     ! {path}: {why}")
        if py_files and py_ok == 0:
            print(
                "[smoke] FAIL: no backend python file parsed cleanly",
                file=sys.stderr,
            )
            return 3
        if py_failures:
            print(
                f"[smoke] WARN: {len(py_failures)} python file(s) failed ast.parse",
                file=sys.stderr,
            )

        # Frontend: check at least one TSX/JSX/TS file is non-empty.
        web_exts = (".tsx", ".jsx", ".ts", ".js", ".html")
        web_files = [
            p
            for p in _walk_files(frontend_dir)
            if p.endswith(web_exts) and os.path.isfile(p)
        ]
        non_empty = sum(1 for p in web_files if os.path.getsize(p) > 0)
        print(
            f"[smoke]   frontend web files: {non_empty}/{len(web_files)} non-empty"
        )
        if not web_files or non_empty == 0:
            print(
                "[smoke] FAIL: no non-empty frontend web sources produced",
                file=sys.stderr,
            )
            return 3

        # docker-compose.yml at root
        compose = os.path.join(tmpdir, "docker-compose.yml")
        if not os.path.isfile(compose):
            print(
                "[smoke] WARN: docker-compose.yml missing at output root",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # HTTP path: hit /generate-output-from-project the way useGenerateCode
    # does (POST JSON body with the full ProjectInput + settings.generator).
    # ------------------------------------------------------------------
    print("[smoke] exercising HTTP /generate-output-from-project ...")
    try:
        import asyncio

        import httpx
        from httpx._transports.asgi import ASGITransport

        from besser.utilities.web_modeling_editor.backend.backend import app

        with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        # useGenerateCode sets settings.generator and a normalized name; the
        # fixture already has settings.generator='web_app'.

        async def _post() -> httpx.Response:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=60.0
            ) as ac:
                return await ac.post(
                    "/besser_api/generate-output-from-project", json=payload
                )

        resp = asyncio.run(_post())
        ctype = resp.headers.get("content-type", "")
        body_len = len(resp.content)
        print(
            f"[smoke]   HTTP status={resp.status_code} "
            f"content-type={ctype!r} body_bytes={body_len}"
        )
        if resp.status_code != 200:
            preview = resp.text[:500] if resp.text else "<empty>"
            print(f"[smoke] FAIL: HTTP non-200 — body: {preview}", file=sys.stderr)
            return 4
        if "zip" not in ctype.lower():
            print(
                f"[smoke] WARN: expected zip media-type, got {ctype!r}",
                file=sys.stderr,
            )
        if body_len < 1024:
            print(
                f"[smoke] FAIL: HTTP zip too small ({body_len}B)",
                file=sys.stderr,
            )
            return 4
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAIL: HTTP exercise raised: {exc!r}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 4

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
