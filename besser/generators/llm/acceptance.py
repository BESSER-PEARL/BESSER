"""Model-derived acceptance matrix for generated apps.

For every class in the domain model, three statically-checkable facts
decide whether the entity actually made it into the app:

* ``route``  — the backend exposes REST routes for it (checked against
  the same static route parse the endpoint manifest uses);
* ``page``   — some frontend file is about it (name or content match);
* ``create`` — a frontend file about it issues a POST (a create form
  that actually submits somewhere).

The matrix is deliberately REPORT-ONLY (warnings + a recipe field, never
blockers): a GUI-model-driven run may legitimately scope the UI to a
subset of entities, so "entity X has no page" is a visibility signal for
the checklist and the recipe, not a defect the fix loop must burn turns
on. Blocker-level enforcement stays with ``contract_checks``.
"""

from __future__ import annotations

import os
import re

_FRONTEND_EXTS = (".js", ".jsx", ".ts", ".tsx")
_SKIP_DIRS = ("node_modules", "dist", "build", "__pycache__")

# A POST issued from frontend code — axios/api `.post(`, fetch with
# method POST, or a generated api-layer helper.
_POST_RE = re.compile(r"\.post\s*\(|method\s*:\s*['\"]POST['\"]", re.IGNORECASE)


def _entity_forms(name: str) -> list[str]:
    """Lowercase spellings a generated app plausibly uses for a class."""
    low = name.lower()
    forms = {low, f"{low}s"}
    if low.endswith("y"):
        forms.add(f"{low[:-1]}ies")
    if low.endswith("s"):
        forms.add(f"{low}es")
    return sorted(forms, key=len, reverse=True)


def _mentions(text: str, forms: list[str]) -> bool:
    low = text.lower()
    return any(f in low for f in forms)


def build_acceptance_matrix(
    output_dir: str,
    domain_model,
    endpoint_manifest: str = "",
    snapshot_dir: str = ".besser_snapshot",
) -> dict[str, dict[str, bool]] | None:
    """Compute {class name: {route, page, create}} for the workspace.

    ``endpoint_manifest`` is the statically parsed route listing (from
    ``prompt_builder.build_endpoint_manifest``); pass it in when the
    caller already built one, else it is rebuilt here.

    Returns None when there is no domain model / no classes.
    """
    if domain_model is None:
        return None
    try:
        classes = [c.name for c in domain_model.get_classes() if getattr(c, "name", None)]
    except Exception:
        return None
    if not classes:
        return None

    if not endpoint_manifest:
        try:
            from besser.generators.llm.prompt_builder import build_endpoint_manifest
            endpoint_manifest = build_endpoint_manifest(output_dir) or ""
        except Exception:
            endpoint_manifest = ""
    manifest_low = endpoint_manifest.lower()

    # Collect frontend files once: (rel path, content).
    frontend_files: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith(_FRONTEND_EXTS):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
            if rel.startswith(snapshot_dir) or rel.startswith(".besser_"):
                continue
            try:
                if os.path.getsize(fpath) > 1_000_000:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    frontend_files.append((rel, f.read()))
            except Exception:
                continue

    matrix: dict[str, dict[str, bool]] = {}
    for cls in sorted(classes):
        forms = _entity_forms(cls)
        route = any(f in manifest_low for f in forms)
        page = False
        create = False
        for rel, content in frontend_files:
            about = _mentions(rel.rsplit("/", 1)[-1], forms) or _mentions(content, forms)
            if not about:
                continue
            page = True
            if _POST_RE.search(content):
                create = True
                break
        matrix[cls] = {"route": route, "page": page, "create": create}
    return matrix


def matrix_issues(matrix: dict[str, dict[str, bool]] | None) -> list[str]:
    """Render missing matrix cells as advisory issue strings."""
    if not matrix:
        return []
    issues: list[str] = []
    for cls, cells in matrix.items():
        missing = [k for k in ("route", "page", "create") if not cells.get(k)]
        if not missing:
            continue
        detail = {
            "route": "no backend REST route",
            "page": "no frontend page/component references it",
            "create": "no frontend POST for it (create form not wired)",
        }
        issues.append(
            "acceptance: entity "
            + cls
            + " — "
            + "; ".join(detail[m] for m in missing)
        )
    return issues
