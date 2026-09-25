"""Static correctness checks over the LLM-authored frontend.

Two defect classes, both blocker-worthy because they leave the UI visibly
broken whatever the request was: a router with no home route (the app opens
blank) and a submit handler that does nothing (the form is dead). Plus a
method button wired to another entity's table, which sends the wrong id.

Both checks read only ``output_dir`` and depend on ``frontend_bindings``.
"""

from __future__ import annotations

import os
import re as _re

from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.validation.frontend_bindings import literal_component_props


def _method_button_source_issues(output_dir: str) -> list[str]:
    """A method button whose id comes from a table of another entity.

    E.g. a Bill page's ``registerPayment`` button copied into Booking.tsx and
    rebound to the Booking table posts
    ``/bill/<booking id>/methods/registerPayment/``. The generated
    ``TableBlock`` names its entity in ``dataBinding``, so the mismatch is a
    one-file check; a button whose table is not on the page is left alone.
    """
    issues: list[str] = []
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", "build")]
        for fname in files:
            if not fname.endswith((".tsx", ".jsx")):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
            if rel.startswith(_SNAPSHOT_DIR):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue
            tables: dict[str, str] = {}
            buttons: list[tuple[int, dict]] = []
            for start, component, props in literal_component_props(content):
                if component == "TableBlock":
                    binding = props.get("dataBinding")
                    entity = binding.get("entity") if isinstance(binding, dict) else None
                    tid = props.get("id")
                    if isinstance(tid, str) and isinstance(entity, str):
                        tables[tid] = entity
                else:
                    buttons.append((start, props))
            for start, props in buttons:
                endpoint = props.get("endpoint")
                source = props.get("instanceSourceTableId")
                if not isinstance(endpoint, str) or not isinstance(source, str):
                    continue
                route_entity = endpoint.strip("/").split("/")[0].lower()
                table_entity = tables.get(source)
                if not route_entity or table_entity is None or table_entity.lower() == route_entity:
                    continue
                label = props.get("label") or endpoint
                line_no = content.count("\n", 0, start) + 1
                issues.append(
                    f"frontend contract: {rel} line {line_no}: method button "
                    f"'{label}' posts to "
                    f"/{route_entity}/ but takes its id from table "
                    f"'{source}', which lists {table_entity} rows - the "
                    f"request would carry a {table_entity} id where a "
                    f"{route_entity} id is required. Bind it to a {route_entity} "
                    f"table (or move it to the {route_entity} page)."
                )
    return issues


def collect_frontend_contract_issues(output_dir: str) -> list[str]:
    """High-precision correctness checks on the (LLM-authored) frontend.

    Only two defect classes are reported, both BLOCKER-worthy because
    they leave the UI visibly broken regardless of scope:

      1. Blank-on-load: a React Router config with no home ("/") route.
      2. Dead form: a <form> whose onSubmit is a no-op, or a UI with
         forms but no HTTP write calls at all -- it cannot save.

    Deliberately conservative: named submit handlers and single-page
    (router-less) apps are NOT flagged, to avoid false blockers that
    would trigger needless auto-fix turns or mark good runs incomplete.
    Scope choices (delete button, nav bar, styling) stay guidance in the
    Phase-2 prompt, never enforced here.
    """
    issues: list[str] = []
    frontend_files: list[tuple[str, str]] = []  # (rel, content)

    for root, dirs, files in os.walk(output_dir):
        # Prune noisy / irrelevant trees in place.
        dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", "build")]
        for fname in files:
            if not fname.endswith((".js", ".jsx", ".ts", ".tsx")):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
            if rel.startswith(_SNAPSHOT_DIR):
                continue
            try:
                if os.path.getsize(fpath) > 1_000_000:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            low = content.lower()
            # Treat a file as frontend if it lives in a frontend/src tree
            # (where generated apps put components AND the react-less
            # services/api.js layer) or it clearly contains React/JSX.
            # This keeps the HTTP-write scan from missing api.js while
            # still ignoring stray backend/config JS.
            rel_low = rel.lower()
            is_frontend = (
                "frontend/" in rel_low
                or rel_low.startswith("src/")
                or "/src/" in rel_low
                or "react" in low
                or "<" in content
            )
            if not is_frontend:
                continue
            frontend_files.append((rel, content))

    if not frontend_files:
        return issues

    blob = "\n".join(c for _, c in frontend_files)

    # ---- Check 1: blank-on-load (router present, but no home route) ----
    uses_router = bool(
        _re.search(r"<Route\b", blob)
        or "react-router" in blob.lower()
        or "createbrowserrouter" in blob.lower()
    )
    if uses_router:
        has_home = bool(
            _re.search(r"""path\s*[=:]\s*['"]/['"]""", blob)  # path="/" or path: "/"
            or _re.search(r"<Route\s+index\b", blob)          # <Route index .../>
            or _re.search(r"\bindex\s*:\s*true\b", blob)       # data-router index: true
            or _re.search(r"<Navigate\b", blob)                # redirect to a default
        )
        if not has_home:
            route_file = next(
                (rel for rel, c in frontend_files
                 if _re.search(r"<Route\b", c) or "createbrowserrouter" in c.lower()),
                frontend_files[0][0],
            )
            issues.append(
                f"frontend contract: {route_file} defines routes but no home "
                f'route for "/", so the app renders blank on first load. Add a '
                f'route for path "/" (a home/index page or a <Navigate to="..."/> '
                f"redirect to the primary list page)."
            )

    # ---- Check 2a: no-op / preventDefault-only onSubmit handlers ----
    noop_onsubmit = _re.compile(
        r"onSubmit\s*=\s*\{\s*(?:async\s*)?"
        r"\(?\s*\w*\s*\)?\s*=>\s*"
        r"\{\s*(?:[\w.]*\.preventDefault\(\)\s*;?\s*)?\}"  # {} or { e.preventDefault(); }
        r"\s*\}"
    )
    onsubmit_prevent_only = _re.compile(
        r"onSubmit\s*=\s*\{\s*\(?\s*\w+\s*\)?\s*=>\s*[\w.]*\.preventDefault\(\)\s*\}"
    )
    for rel, content in frontend_files:
        if noop_onsubmit.search(content) or onsubmit_prevent_only.search(content):
            issues.append(
                f"frontend contract: {rel} has a form whose onSubmit does "
                f"nothing (an empty or preventDefault-only handler), so the form "
                f"cannot save. Wire onSubmit to call the backend (POST to create, "
                f"PUT to update) and refresh the list on success."
            )

    # ---- Check 2b: forms exist but the frontend never writes to the API ----
    has_form = bool(
        _re.search(r"<form\b", blob, _re.IGNORECASE) or "onsubmit" in blob.lower()
    )
    has_http_write = bool(
        _re.search(r"\bfetch\s*\(", blob)
        or "axios" in blob.lower()
        or _re.search(r"\.(post|put|patch)\s*\(", blob)
        or _re.search(r"""method\s*:\s*['"](post|put|patch)['"]""", blob, _re.IGNORECASE)
        or "xmlhttprequest" in blob.lower()
    )
    if has_form and not has_http_write:
        issues.append(
            "frontend contract: the frontend renders forms but makes no HTTP "
            "write calls (no fetch/axios POST/PUT anywhere), so nothing can be "
            "saved to the backend. Add API calls that POST/PUT form data to the "
            "REST endpoints and reload the affected list on success."
        )

    return issues
