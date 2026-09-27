"""The "design system" prompt section for runs whose GUI model carries a design.

Rendered only when the generated app ships ``src/design.css`` (the GUI model's
stylesheet, emitted by the React generator) or, failing that, when the GUI model
has a non-empty stylesheet. It names the real classes and variables of that
stylesheet so new markup reuses them, and tells the agent to edit designed
pages in place. ``validation/design_fidelity.py`` checks the same rules.
"""

from __future__ import annotations

import os
import re

from besser.spec_driven_agent.validation.design_fidelity import (
    design_variables,
    find_design_css,
)

# Role of a class, from the first of its hyphen-separated parts that names one:
# ``ds-kpi-label`` is a card part, ``rail-link`` belongs to the navigation.
_ROLES = (
    ("Layout", {"page", "shell", "container", "section", "grid", "layout", "wrap",
                "stack", "head", "hero", "band"}),
    ("Navigation", {"nav", "rail", "brand", "menu", "tabs", "footer", "top"}),
    ("Cards", {"card", "kpi", "metric", "panel", "tile", "summary", "confirmation"}),
    ("Forms", {"field", "label", "input", "form", "filter", "select", "search"}),
    ("Buttons", {"btn", "button", "action", "link"}),
    ("Tables", {"table"}),
    ("Status", {"badge", "pill", "notice", "alert", "note", "check", "status"}),
    ("Text", {"heading", "title", "text", "muted", "value", "price", "stamp"}),
)
_ROLE_OF = {word: role for role, words in _ROLES for word in words}
_MAX_PER_ROLE = 5
_MAX_VARIABLES = 16
# The fallback embeds the stylesheet itself (there is no file to read).
_MAX_INLINE_CSS = 16_000


def _class_vocabulary(css: str) -> tuple[list[str], dict[str, list[str]]]:
    """Base classes in declaration order, and the modifiers each one takes.

    ``.hotel-btn.outline`` makes ``outline`` a modifier of ``hotel-btn``; a
    class that never appears first in a compound selector is only a modifier.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    firsts: dict[str, None] = {}
    modifiers: dict[str, dict[str, None]] = {}
    for selector in re.findall(r"([^{}]+)\{", css):
        if selector.strip().startswith("@"):
            continue
        for compound in re.findall(r"((?:\.-?[A-Za-z_][\w-]*)+)", selector):
            names = re.findall(r"\.(-?[A-Za-z_][\w-]*)", compound)
            firsts.setdefault(names[0], None)
            for name in names[1:]:
                modifiers.setdefault(names[0], {}).setdefault(name, None)
    return list(firsts), {k: list(v) for k, v in modifiers.items()}


def _vocabulary_lines(css: str) -> list[str]:
    classes, modifiers = _class_vocabulary(css)
    variables = [v for v in design_variables(css) if v.startswith("--ds-")] or design_variables(css)
    lines = []
    if variables:
        shown = variables[:_MAX_VARIABLES]
        more = f" (+{len(variables) - len(shown)} more)" if len(variables) > len(shown) else ""
        if all(v.startswith("--ds-") for v in shown):
            lines.append("- `--ds-*`: " + ", ".join(v[5:] for v in shown) + more)
        else:
            lines.append("- Variables: " + ", ".join(f"`{v}`" for v in shown) + more)
    grouped: dict[str, list[str]] = {role: [] for role, _ in _ROLES}
    listed = 0
    for name in classes:
        role = next((_ROLE_OF[p] for p in name.split("-") if p in _ROLE_OF), None)
        if role is None or len(grouped[role]) >= _MAX_PER_ROLE:
            continue
        mods = modifiers.get(name)
        grouped[role].append(f"`{name}`" + (f" (+{', '.join(mods)})" if mods else ""))
        listed += 1
    lines.extend(f"- {role}: " + ", ".join(names) for role, names in grouped.items() if names)
    if len(classes) > listed:
        lines.append(f"- ...and {len(classes) - listed} more classes in the stylesheet.")
    return lines


def design_system_section(output_dir: str | None, gui_model=None) -> str:
    """The prompt section, or ``""`` when the run has no GUI design."""
    css_rel = find_design_css(output_dir) if output_dir else None
    css = ""
    if css_rel:
        try:
            with open(os.path.join(output_dir, css_rel), encoding="utf-8", errors="replace") as fh:
                css = fh.read()
        except OSError:
            pass
    if not css.strip():
        css_rel = None
        css = getattr(gui_model, "stylesheet", "") or ""
    classes, _ = _class_vocabulary(css)
    if not classes:
        return ""
    has = set(classes)

    if css_rel:
        source = f"`{css_rel}` is the design's source of truth, imported by the entry file."
        embedded = ""
    else:
        source = (
            "The GUI model carries the design stylesheet below and the scaffold has no "
            "copy: add it unchanged as `src/design.css`, imported by the frontend entry file."
        )
        embedded = f"\n```css\n{css[:_MAX_INLINE_CSS]}\n```\n"
    field = (
        "a field is `ds-field` > `ds-label` + `ds-input`"
        if {"ds-field", "ds-label", "ds-input"} <= has else "fields use the Forms classes"
    )
    button = (
        "a button `ds-btn` / `ds-btn-primary`" if {"ds-btn", "ds-btn-primary"} <= has
        else "buttons the Buttons classes"
    )
    spacing = (
        "Spacing from `--ds-space-*`"
        if any(v.startswith("--ds-space") for v in design_variables(css)) else "One spacing scale"
    )
    vocabulary = "\n".join(_vocabulary_lines(css))
    return f"""
## Design system (from the GUI model)

{source} Where Rules 2, 13 or 15 disagree, this section wins.
{embedded}
Classes and variables (real names - reuse them, never invent parallel ones):
{vocabulary}

Keep the design:
- Never edit design.css; a genuinely new rule goes in `src/design-overrides.css`,
  imported after it, using `var(--ds-*)`.
- Edit designed pages in place (`modify_file` / `replace_file_lines`); never
  `write_file` over an existing page, even after refused edits.
- Keep each page's nav/header, layout wrappers and generated components
  (`TableBlock`, `MethodButton`, `CrudButton`, `FormBlock`, charts, metric cards):
  they are wired to the backend. Wire behaviour inside the existing JSX; never
  swap them for hand-written fetch markup.
- New markup uses the design classes: {field}, {button}.
  No `style={{{{...}}}}` objects, no hex colours.

Design quality:
- {spacing}; align to the page's grid. One page heading, then
  section headings and muted secondary text.
- Loading, empty and error states on every data view, styled with the design classes.
- Every control labelled (visible label or `aria-label`); one primary button per form.
- A new screen copies an existing page's shell and card pattern.
"""
