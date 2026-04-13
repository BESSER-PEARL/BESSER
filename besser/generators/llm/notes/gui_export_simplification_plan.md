# Simplify GUI Model Export

## Context
The `gui_model_builder.py` generates very verbose Python exports (3189 lines for a single project). The main bloat sources:
1. Every component gets `color_palette="default"` even though it's the default — 220 occurrences in the example file, 33 are _only_ `Color(color_palette="default")` (completely meaningless)
2. Duplicate styling serializers: `_write_styling_object` (150 lines, multi-line) does the same thing as `_build_styling_expr` (inline)
3. Empty `Size()`, `Position()`, `Color()` objects still emitted by `_write_styling_object`

## What We Already Did
- Commented out `style_entries` from the export (`gui_model_builder.py`) — no generator uses it
- Commented out `style_entries` population in `json_to_buml/processor.py` — only needed for editor round-trip, TODO: rebuild from per-component Styling in buml_to_json later

## Changes

### File: `besser/utilities/buml_code_builder/gui_model_builder.py`

### Step 1: Filter default values in `_build_styling_expr`
- Skip `color_palette` when value is `"default"`
- This alone removes noise from all 220 Styling expressions and eliminates 33 entirely empty Color blocks

### Step 2: Remove `_write_styling_object` (lines 1003-1152) and `_write_styling` (lines 1155-1160)
- Replace the 2 call sites to use `_build_styling_expr` instead:
  - **Screen styling** (line 291-292): Use `_build_styling_expr` + write as assignment
  - **Chart series styling** (line 1221-1224): Use `_build_styling_expr` inline

### Step 3: Keep `_write_layout`
`_write_layout` is used in 3 other places (screen, container, generic component) — only remove its call from `_write_styling_object`.

## What We DON'T Remove
The audit confirmed that the React/WebApp generators use ALL of these:

| Property | Used by generator? | Keep in export? |
|---|---|---|
| `component_id` | YES — React element ID, CSS selectors | YES |
| `display_order` | YES — sorting children | YES |
| `custom_attributes` | YES — spread into React props | YES |
| `css_classes` | YES — className on JSX | YES |
| `tag_name` | YES — HTML tag selection | YES |
| All `Styling` fields | YES — full CSS extraction | YES |
| `style_entries` | NO — not used by any generator | Already removed |

## Expected Impact
- ~160 lines of duplicate code removed (`_write_styling_object` + `_write_styling`)
- Exported files significantly shorter (fewer empty objects, no `color_palette="default"` noise)
- No functional change — generators receive the same data

## Verification
- Run: `python -m pytest tests/ -k gui` to check existing tests
- Manually test with the LLM generator example if possible
