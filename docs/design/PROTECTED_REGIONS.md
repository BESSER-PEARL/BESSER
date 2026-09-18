# Round-trip preservation for BESSER generators

Status: design draft, no implementation
Author: BESSER core
Date: 2026-05-19

## 1. The current state

BESSER's generators are **single-shot, write-only**. Each `generate()`
call renders Jinja templates from a `DomainModel` and writes a fresh
file tree to `output_dir`. No generator looks at what was there before.

Concrete evidence:

- `besser/generators/generator_interface.py::build_generation_dir`
  only does `os.makedirs(..., exist_ok=True)` — it makes the directory
  but doesn't read from it. Existing files are clobbered by the
  `open(file_path, mode="w")` in each per-generator `generate()`.
- `besser/generators/python_classes/python_classes_generator.py`
  opens `classes.py` with `mode="w"` and writes the rendered template
  unconditionally.
- `besser/generators/rest_api/rest_api_generator.py` does the same
  for `main_api.py`, `rest_api.py`, `requirements.txt`.
- `besser/generators/python_classes/templates/python_classes_template.py.j2`
  emits class bodies, getters/setters, and method stubs straight from
  `class.methods`. The only escape hatch is `{% if method.code %}` —
  but that code lives in the **model**, not in the generated file.
  Editing the rendered Python and regenerating discards the edit.
- The LLM orchestrator
  (`besser/generators/llm/orchestrator.py::_phase1_generate`, around
  line 558) calls the deterministic generator into `self.output_dir`
  and then runs the gap-analyzer / Phase 2 LLM loop on top.
  Snapshot/rollback (`_create_snapshot`, `_restore_snapshot`,
  `.besser_snapshot/`) exists only for *within-run* recovery if Phase
  3 auto-fix regresses the build — it is deleted at run end (line
  ~431, 529). Nothing survives across two independent `generate()`
  invocations.

`vibe-bench`'s evolution probe quantifies the consequence: across a
6-step chain, BESSER preserved 25–59% of step-1 files vs naive's
75–100% (`PROBE_evolution.md` §2). A developer who customises one
method between regenerations loses the customisation on the next
model edit.

## 2. Approach comparison

### (A) Protected regions / regions of escape

**Shape in BESSER.** Add paired markers around every block a user
might reasonably edit, in every Jinja template:

```jinja
    def {{ method.name }}(...):
        # BESSER-REGION-BEGIN: {{ class.name }}.{{ method.name }}.body
        {{ method_body.infer_body(class, method) }}
        # BESSER-REGION-END: {{ class.name }}.{{ method.name }}.body
```

Mechanism. A new module `besser/generators/_regions.py` would provide
`capture_regions(output_dir) -> dict[id, text]` and
`reinject_regions(output_dir, captured)`. Each generator's
`generate()` becomes: capture → render → reinject. The orchestrator
would do the same wrap at the directory level. Region IDs need to be
stable — qualified `module.Class.member.slot` from the model graph,
not file-line positions.

Files touched: every `*.j2` in `besser/generators/*/templates/` gets
markers around (a) method bodies, (b) custom-import blocks, (c) the
"safe to edit" footer of each module; each generator's `generate()`
gets the capture/reinject pair (~10 generators); one new shared
module for the region engine; orchestrator gets one wrapper call.

**Effort:** ~3 weeks. Bulk of cost is template surgery across all
generators plus stabilising the region-ID strategy.

**Unlocks.** Users edit method bodies between regenerations and keep
those edits. Imports and helper blocks survive. Regenerate-from-spec
loses its destructive sting.

**Risks / limitations.**
- Markers leak into the generated output (some teams hate this; can
  be hidden with language-appropriate comments — fine for Python,
  uglier for JSON, impossible inside JSX expressions).
- Rename in the model = new region ID = lost edit. Step 3 of the
  evolution probe ("rename Tag → Category") is exactly the case
  this approach handles badly.
- Doesn't cover content **outside** any region (e.g. a new helper
  function the user adds at module level). Need a "free zone"
  convention for top-of-file / bottom-of-file user code.
- Cross-file user edits (new files entirely) need separate handling.

### (B) Generation Gap pattern

**Shape in BESSER.** For each `Class` in `python_classes` /
`pydantic_classes` / `java_classes`: emit two files,
`<Class>Base.py` (regenerated freely) and `<Class>.py` containing
`class <Class>(<Class>Base): pass`. The user-facing subclass is
generated **once** and never overwritten on subsequent regens.

Files touched: split `python_classes_template.py.j2` into a `_base`
template and a stub-subclass template; same for
`pydantic_classes/templates/`, `java_classes/templates/`,
`sql_alchemy/templates/`. Generator `generate()` methods learn to
check `os.path.exists(subclass_path)` before writing. Imports
elsewhere (REST API, Django) need to switch from `<Class>` to import
both the base and the subclass.

**Effort:** ~3 weeks for the OO generators alone; another ~3-5 weeks
to figure out a passable story for FastAPI route modules and React
components (these aren't class-shaped). Realistic: **5-8 weeks** for
broad coverage.

**Unlocks.** Cleanest separation: regenerated code lives in one file,
user code in another, no markers in the middle. Works well in
languages with cheap subclassing (Python, Java).

**Risks / limitations.**
- Non-OO generators (React TSX components, FastAPI route handlers,
  SQL DDL, Terraform HCL, Flutter, JSON schemas) don't have a
  subclass-shaped escape hatch. Need a per-language strategy
  (companion file? wrapper module?) which fragments the design.
- Doubles the file count for class-shaped generators. The bench
  scenarios already complain about file sprawl (BESSER fastapi step
  6: 24 vs naive 12).
- Cross-cutting refactors (extract superclass, step 6 of the probe)
  ripple through both the `Base` and the subclass and require
  thinking about whether the subclass's hand-edits still make sense
  against a new parent.
- Generated APIs (FastAPI endpoints) still consume the **subclass**
  type. If the user accidentally breaks the subclass-base contract,
  the whole generated stack stops compiling.

### (C) Three-way merge

**Shape in BESSER.** Treat `output_dir` as a git working tree. The
orchestrator (entry point: `LLMOrchestrator.run` /
`_phase1_generate`) records three references on every successful
generation:

- `BASE`: the last clean generation (auto-committed to a
  `.besser-history` git branch in `output_dir`)
- `USER`: whatever is currently in `output_dir` (the user's edited
  state)
- `THEIRS`: the freshly generated output, computed in a sibling
  temp dir

Then a `git merge-file --merge=union` (or full three-way merge per
file) is run between `BASE`, `THEIRS`, and `USER`. Conflicts are
either auto-resolved (LLM-driven) or surfaced as conflict markers
the user resolves in their editor.

Files touched: a new module `besser/generators/_merge.py` invoking
git via subprocess; orchestrator wraps Phase 1's generator call;
optional dependency check for git binary. Generators themselves
don't change.

**Effort:** ~3 weeks for a working v1 (git plumbing, temp-dir
shuffling, basic conflict tests). Another 2-3 weeks of iteration to
make conflict output palatable. **~6 weeks for a shippable
implementation.**

**Unlocks.** Handles **any** generator output (no per-template
work), tolerates added/removed files, gives the user a familiar
conflict-resolution UI (git markers / `git mergetool`).

**Risks / limitations.**
- Requires git installed wherever BESSER runs (CI, Docker image, web
  editor backend). The web editor's per-request temp dirs are not
  git repos today.
- Spec-level renames (Tag → Category) don't merge cleanly in git
  because the file is renamed and the content is rewritten;
  three-way merge has known weakness on rename-plus-edit.
- The first regeneration into a previously-untouched directory has
  no `BASE` — needs a bootstrap path.
- Web-editor UX: how does a browser user resolve a conflict marker?
  Either we drop merge for the web editor (CLI/SDK only) or we
  build a conflict-resolution UI.

## 3. Recommendation

**Implement (A) Protected Regions first.**

Reasoning:

1. **Smallest blast radius.** Each generator opts in independently
   by adding markers to its templates. Generators that haven't been
   converted continue to behave exactly as today. The OO generators
   (`python_classes`, `pydantic_classes`, `java_classes`) cover the
   common case and can ship in a single release.
2. **Cleanest fit with the current architecture.** BESSER already
   has a single canonical "user-extensible slot" per method via
   `method.code` (rendered at line 114 of
   `python_classes_template.py.j2`). Protected regions formalises
   that mechanism and extends it to the *generated file* instead of
   only the *model*. No git dependency, no doubled file count, no
   per-language escape strategy.
3. **Compatibility with the LLM orchestrator.** Phase 2's LLM
   writes can target protected regions too — the LLM already has
   `write_file` / `read_file` tools and just needs to learn the
   marker convention. The existing `.besser_snapshot/` directory
   provides a free safety net during the iteration.
4. **Effort proportional to value.** Three weeks is the right
   investment for a feature this critical. Generation Gap doubles
   the file count and still doesn't cover React/FastAPI. Three-way
   merge demands git everywhere and conflict-resolution UX.

Approach (C) remains attractive as a **second** layer (regions
handle the common case; three-way merge handles structural
refactors like rename and add-superclass). Approach (B) is
specifically *not* recommended — its weaknesses bite hardest in the
non-OO generators that are growing fastest in BESSER (React,
Flutter, Terraform).

## 4. Concrete next steps for the recommendation

- **Week 1 — Region engine + Python classes.** Add
  `besser/generators/_regions.py` with `capture_regions` and
  `reinject_regions`. Decide region-ID schema (proposed:
  `<file_relpath>::<qualified_name>::<slot>` where slot ∈
  `{body, imports, class_footer, module_footer}`). Add markers to
  `python_classes/templates/python_classes_template.py.j2`. Wire
  `PythonGenerator.generate()` to capture-then-reinject. Add one
  unit test in `tests/generators/python_classes/` that round-trips
  a hand-edited method body.
- **Week 2 — Orchestrator integration + Pydantic/REST.** Extend the
  region wrapper to `pydantic_classes` and `rest_api` (FastAPI)
  templates. Update `LLMOrchestrator._phase1_generate` so its
  generator call also runs through the capture/reinject pair. Add
  the LLM tool surface a `region_info` query so Phase 2 doesn't
  overwrite user regions blindly.
- **Week 3 — Evolution-probe validation.** Re-run the
  `vibe-bench` evolution probe with a tiny artificial
  customisation (one method body edit) injected between steps.
  Measure the regression on the
  "files-in-step-1-preserved-at-step-N" metric. Target: bring
  BESSER from 25-59% retention to ≥90% on chains where the
  customisations are inside marked regions. Iterate on region
  granularity if numbers fall short.
- **Week 4 (buffer / docs).** Document the markers in
  `docs/source/generators.rst`. Add a "How to extend generated
  code" page. Decide whether the markers are visible (`#
  BESSER-REGION-BEGIN`) or hidden (`# region:body` style
  understood by an IDE plugin only).

## 5. Open questions

1. **Region-ID stability under rename.** If the user renames `Tag
   → Category` in the model, the regions in `Tag.format_for_ui`
   become `Category.format_for_ui` — old captures no longer match.
   Do we (a) match by name only, (b) follow renames via the model's
   internal IDs, or (c) accept the loss and document it?
2. **Free zones vs strict regions.** Should we let the user add
   *new* methods between markers (free-zone semantics), or only
   edit content **inside** markers? Strict is simpler to implement
   and reason about; free-zone is more useful in practice.
3. **Marker visibility in the rendered output.** Visible (clearly
   labelled comments) is honest but ugly. Invisible (only an IDE
   plugin shows them) is prettier but harder to discover. Which
   does the web editor's user persona prefer?
4. **Non-comment file types.** JSON schemas, OpenAPI specs, SQL
   migrations — none of these tolerate `// BESSER-REGION` markers
   cleanly. Sidecar `.regions.json` files? Skip those generators
   for v1?
5. **Web-editor backend behaviour.** The
   `/generate-output-from-project` endpoint streams a ZIP from a
   fresh temp dir per request. Regions only make sense if there's
   continuity between two requests. Is the web editor in scope for
   v1, or is this strictly a CLI/SDK feature first?
