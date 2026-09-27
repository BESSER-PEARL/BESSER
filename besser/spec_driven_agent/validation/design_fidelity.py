"""Design-regression check: the frontend must keep the GUI design it started with.

The React generator carries a GUI model's stylesheet into ``src/design.css``
and renders every designed screen from it. Live runs showed the agent throwing
that away: a ``write_file`` over a designed page kept 37 of 358 lines, dropping
the nav, the layout wrappers and five generated ``MethodButton``s. The code
still compiled and every other validator stayed green.

A per-page fingerprint of the Phase-1 frontend (design classes used, nav or
header present, generated components rendered, inline styles, hex colours,
bare form controls) is saved to :data:`BASELINE_FILENAME` before the agent
starts, and compared with the final tree. Losing the design is a
``design regression:`` blocker; adding off-design markup is a
``design drift:`` warning. Without a design stylesheet there is no baseline
and the check is silent.
"""

from __future__ import annotations

import json
import os
import re
import shutil

_BASELINE_PREFIX = ".besser_design_"
BASELINE_FILENAME = _BASELINE_PREFIX + "baseline.json"
DESIGN_CSS = "design.css"

# Components the React generator renders from GUI-model widgets. Each one is
# wired to the backend already, so hand-written markup in its place is a loss.
GENERATED_COMPONENTS = (
    "MethodButton", "CrudButton", "TableBlock", "FormBlock", "ChartBlock",
    "MetricCardBlock", "DataListBlock", "MapBlock", "AlertBlock",
)
# A page is "designed" when it uses at least this many design classes.
MIN_DESIGN_CLASSES = 3
# Calibrated on six live runs (see the docs): pages edited in place lost 0% of
# their design classes, the two rewritten ones 71% and 84%.
MAX_CLASS_LOSS = 0.4

_SKIP_DIRS = {"node_modules", "dist", "build", ".next", ".git"}
_SOURCE_EXT = (".tsx", ".jsx", ".ts", ".js")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_CSS_CLASS = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
_CSS_VAR = re.compile(r"(--[\w-]+)\s*:")
_CLASS_ATTR = re.compile(r"\bclassName\s*=\s*")
_CLASS_TOKEN = re.compile(r"-?[A-Za-z_][\w-]*")
_STRING_LITERAL = re.compile(r"\"([^\"\n]*)\"|'([^'\n]*)'|`([^`]*)`")
_INLINE_STYLE = re.compile(r"\bstyle=\{\{")
_HEX = re.compile(r"(?<![\w&])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_LANDMARK = re.compile(r"<(nav|header)\b")
_CONTROL = re.compile(r"<(input|select|textarea|button|label)\b")
_COMPONENT = re.compile(r"<(" + "|".join(GENERATED_COMPONENTS) + r")\b")
_ENDPOINT = re.compile(r"\bendpoint=\"([^\"]+)\"")
_DESIGN_IMPORT = re.compile(r"""import\s+['"][^'"]*\bdesign\.css['"]""")


# ---------------------------------------------------------------- stylesheet

def design_classes(css: str) -> set[str]:
    """Class names the stylesheet defines selectors for (``@media`` included)."""
    css = _CSS_COMMENT.sub("", css or "")
    classes: set[str] = set()
    for selector in re.findall(r"([^{}]+)\{", css):
        if selector.strip().startswith("@"):
            continue
        classes.update(_CSS_CLASS.findall(selector))
    return classes


def design_variables(css: str) -> list[str]:
    """Custom properties the stylesheet declares, in declaration order."""
    seen: dict[str, None] = {}
    for name in _CSS_VAR.findall(_CSS_COMMENT.sub("", css or "")):
        seen.setdefault(name, None)
    return list(seen)


def find_design_css(output_dir: str) -> str | None:
    """Relative path of the generated ``src/design.css``, or ``None``."""
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".besser_"))
        if DESIGN_CSS in files and os.path.basename(root) == "src":
            return os.path.relpath(os.path.join(root, DESIGN_CSS), output_dir).replace("\\", "/")
    return None


# ---------------------------------------------------------------- JSX scan

def _tag_end(source: str, start: int) -> int:
    """Index just past the ``>`` closing the JSX tag opened at ``start``."""
    depth, quote, i = 0, None, start
    while i < len(source):
        ch = source[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ">" and depth <= 0:
            return i + 1
        i += 1
    return len(source)


def _class_literals(tag: str) -> list[str] | None:
    """Tokens of the tag's ``className``; ``None`` when it has no literal to read."""
    match = _CLASS_ATTR.search(tag)
    if not match:
        return []
    rest = tag[match.end():]
    if rest[:1] in "\"'":
        end = rest.find(rest[0], 1)
        literals = [rest[1:end]]
    elif rest[:1] == "{":
        expression, depth = rest, 0
        for k, ch in enumerate(rest):
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if depth == 0:
                expression = rest[:k + 1]
                break
        literals = [a or b or c for a, b, c in _STRING_LITERAL.findall(expression)]
        if not literals:
            return None
    else:
        return None
    tokens: list[str] = []
    for literal in literals:
        tokens.extend(_CLASS_TOKEN.findall(re.sub(r"\$\{[^}]*\}", " ", literal)))
    return tokens


def page_fingerprint(source: str, classes: set[str]) -> dict:
    """What a page renders of the design, as comparable counts and sets."""
    used: set[str] = set()
    for match in _CLASS_ATTR.finditer(source):
        start = source.rfind("<", 0, match.start())
        tokens = _class_literals(source[start:_tag_end(source, start)]) or []
        used.update(t for t in tokens if t in classes)
    bare = 0
    for match in _CONTROL.finditer(source):
        tokens = _class_literals(source[match.start():_tag_end(source, match.start())])
        if tokens is not None and not any(t in classes for t in tokens):
            bare += 1
    components: dict[str, int] = {}
    for name in _COMPONENT.findall(source):
        components[name] = components.get(name, 0) + 1
    return {
        "design_classes": sorted(used),
        "landmarks": sorted(set(_LANDMARK.findall(source))),
        "components": components,
        "inline_styles": len(_INLINE_STYLE.findall(source)),
        "hex_colors": len(_HEX.findall(source)),
        "bare_controls": bare,
    }


def _frontend_sources(src_dir: str) -> dict[str, str]:
    """``{path relative to src: text}`` for the frontend's script files."""
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".besser_")]
        for name in files:
            if name.endswith(_SOURCE_EXT):
                path = os.path.join(root, name)
                try:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        out[os.path.relpath(path, src_dir).replace("\\", "/")] = fh.read()
                except OSError:
                    continue
    return out


def _is_page(rel: str) -> bool:
    return rel.startswith("pages/") and rel.endswith((".tsx", ".jsx"))


def _component_counts(sources: dict[str, str]) -> dict[str, int]:
    """Generated-component render counts over the whole tree.

    Tree-wide so moving a component into a new child file is not a loss; the
    component's own definition file is skipped.
    """
    counts: dict[str, int] = {}
    for rel, text in sources.items():
        stem = os.path.splitext(os.path.basename(rel))[0]
        for name in _COMPONENT.findall(text):
            if name != stem:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _endpoint_counts(sources: dict[str, str], endpoints) -> dict[str, int]:
    """Occurrences of each method-button endpoint literal anywhere in the tree."""
    blob = "\n".join(sources.values())
    return {e: blob.count(f'"{e}"') + blob.count(f"'{e}'") for e in endpoints}


# ---------------------------------------------------------------- baseline

def capture_design_baseline(output_dir: str) -> dict | None:
    """Fingerprint the frontend's design as it stands; ``None`` without one."""
    css_rel = find_design_css(output_dir)
    if css_rel is None:
        return None
    try:
        with open(os.path.join(output_dir, css_rel), encoding="utf-8", errors="replace") as fh:
            css = fh.read()
    except OSError:
        return None
    classes = design_classes(css)
    if not classes:
        return None
    src_dir = os.path.dirname(os.path.join(output_dir, css_rel))
    src_rel = os.path.dirname(css_rel)
    sources = _frontend_sources(src_dir)
    endpoints = sorted({e for text in sources.values() for e in _ENDPOINT.findall(text)})
    pages = {}
    for rel, text in sources.items():
        if _is_page(rel):
            fingerprint = page_fingerprint(text, classes)
            if len(fingerprint["design_classes"]) >= MIN_DESIGN_CLASSES:
                pages[f"{src_rel}/{rel}"] = fingerprint
    return {
        "design_css": css_rel,
        "classes": sorted(classes),
        "imported": any(_DESIGN_IMPORT.search(t) for t in sources.values()),
        "components": _component_counts(sources),
        "endpoints": _endpoint_counts(sources, endpoints),
        "shared_landmarks": sorted(
            rel for rel, text in sources.items()
            if not _is_page(rel) and _LANDMARK.search(text)
        ),
        "pages": pages,
    }


def save_design_baseline(output_dir: str) -> dict | None:
    """Capture and persist the baseline, replacing any earlier one.

    The designed pages and design.css are also copied verbatim (as ``.txt``,
    so no source scanner reads them), which is what makes a regression
    repairable: the finding names the copy to restore from.
    """
    for name in os.listdir(output_dir):
        if name.startswith(_BASELINE_PREFIX):
            os.remove(os.path.join(output_dir, name))
    baseline = capture_design_baseline(output_dir)
    if baseline is None:
        return None
    copies = {}
    for index, rel in enumerate([baseline["design_css"], *sorted(baseline["pages"])]):
        name = f"{_BASELINE_PREFIX}{index}_{os.path.basename(rel)}.txt"
        shutil.copyfile(os.path.join(output_dir, rel), os.path.join(output_dir, name))
        copies[rel] = name
    baseline["copies"] = copies
    with open(os.path.join(output_dir, BASELINE_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=1)
    return baseline


def load_design_baseline(output_dir: str) -> dict | None:
    try:
        with open(os.path.join(output_dir, BASELINE_FILENAME), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- compare

def collect_design_fidelity_issues(output_dir: str, baseline: dict | None = None) -> list[str]:
    """Findings for design lost since the baseline; ``[]`` without a baseline."""
    if baseline is None:
        baseline = load_design_baseline(output_dir)
    if not baseline:
        return []
    copies = baseline.get("copies", {})

    def restore_hint(rel: str) -> str:
        return f" Its Phase-1 version is saved at {copies[rel]}." if rel in copies else ""

    css_rel = baseline["design_css"]
    base_classes = set(baseline["classes"])
    issues: list[str] = []
    try:
        with open(os.path.join(output_dir, css_rel), encoding="utf-8", errors="replace") as fh:
            classes = design_classes(fh.read())
    except OSError:
        classes = set()
    kept = len(classes & base_classes)
    if kept * 2 < len(base_classes):
        issues.append(
            f"design regression: {css_rel} was deleted or emptied - it defined "
            f"{len(base_classes)} design classes and now defines {kept} of them. "
            f"Restore it: every designed page is styled by it.{restore_hint(css_rel)}"
        )
        classes |= base_classes  # judge the pages against the design they had
    src_rel = os.path.dirname(css_rel)
    sources = _frontend_sources(os.path.join(output_dir, src_rel))
    if baseline.get("imported") and not any(_DESIGN_IMPORT.search(t) for t in sources.values()):
        issues.append(
            f"design regression: nothing imports {css_rel} any more, so the app "
            "renders without the GUI design. Restore `import './design.css';` "
            "in the entry file (index.tsx / main.tsx)."
        )

    shared_landmarks = {
        rel for rel, text in sources.items() if not _is_page(rel) and _LANDMARK.search(text)
    }
    new_shared_nav = bool(shared_landmarks - set(baseline.get("shared_landmarks", [])))
    counts = _component_counts(sources)
    lost_components = {
        name for name, before in baseline.get("components", {}).items()
        if counts.get(name, 0) < before
    }
    # MethodButtons folded into one rendered from a list of their endpoints
    # are not a loss: every endpoint literal is still there as often as before.
    base_endpoints = baseline.get("endpoints", {})
    now = _endpoint_counts(sources, base_endpoints)
    if base_endpoints and all(now[e] >= n for e, n in base_endpoints.items()):
        lost_components.discard("MethodButton")

    for page_rel, before in sorted(baseline["pages"].items()):
        text = sources.get(page_rel[len(src_rel) + 1:] if src_rel else page_rel)
        if text is None:
            issues.append(
                f"design regression: designed page {page_rel} was deleted. Restore "
                f"it and edit it in place.{restore_hint(page_rel)}"
            )
            continue
        after = page_fingerprint(text, classes)
        had = set(before["design_classes"])
        lost = sorted(had - set(after["design_classes"]))
        problems = []
        gone = sorted(set(before["landmarks"]) - set(after["landmarks"]))
        if gone and not new_shared_nav:
            problems.append("its " + " and ".join(f"<{tag}>" for tag in gone)
                            + (" are" if len(gone) > 1 else " is") + " gone")
        if len(lost) > MAX_CLASS_LOSS * len(had):
            shown = ", ".join(lost[:8]) + (", ..." if len(lost) > 8 else "")
            problems.append(f"{len(lost)} of its {len(had)} design classes are gone ({shown})")
        removed = [
            f"{n - after['components'].get(name, 0)} {name}"
            for name, n in sorted(before["components"].items())
            if name in lost_components and after["components"].get(name, 0) < n
        ]
        if removed:
            problems.append("generated components were removed (" + ", ".join(removed) + ")")
        if problems:
            issues.append(
                f"design regression: {page_rel}: " + "; ".join(problems) + ". The page "
                "was rebuilt instead of edited: restore its designed markup and generated "
                "components, then wire the behaviour inside them with modify_file / "
                f"replace_file_lines.{restore_hint(page_rel)}"
            )
        added = [
            f"+{after[key] - before[key]} {what}"
            for key, what in (("inline_styles", "inline style={{...}} object(s)"),
                              ("hex_colors", "hard-coded hex colour(s)"),
                              ("bare_controls", "form control(s) without a design class"))
            if after[key] > before[key]
        ]
        if added:
            issues.append(
                f"design drift: {page_rel}: " + ", ".join(added) + ". Use the "
                "design.css classes (e.g. its field/input/button classes) and "
                "var(--ds-*) variables instead."
            )
    return issues
