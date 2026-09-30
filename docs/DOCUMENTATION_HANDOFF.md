# Documentation maintenance handoff

Updated 2026-09-30 for the coordinated v8 documentation rebuild. This is a
handoff for the next maintainer or coding agent. Check the current branches,
versions and deployed editor before repeating these steps.

## Repositories and starting point

| Project | Integration branch used | Documentation |
| --- | --- | --- |
| BESSER | `development` | `docs/source/` |
| Web Modeling Editor (WME) | `develop` | `besser/utilities/web_modeling_editor/frontend/docs/source/` |
| modeling-agent | `develop` | Its own repository's `docs/source/` |

WME is a Git submodule of BESSER and has its own commits, CI and Read the Docs
configuration. The agent is a separate repository. Start by reading any current
`AGENTS.md` instructions and checking `git status`, branch names and remotes in
each repository. Keep existing work. Fetch current PR metadata and compare its
head with the checkout before claiming to review a release.

The original review artifacts in this workspace are under
`reviews/release-v8-2026-09-30/`; the local site preview is under
`reviews/docs-preview/`. Those are workspace artifacts, not the authoritative
documentation sources. The reusable capture tool lives with WME docs under
`docs/tools/capture_editor_screenshots.py` and is included with the
documentation changes; it does not depend on the
workspace review scripts.

## What changed and why

* All three sites now use **Sphinx-Immaterial**, the Material theme fork.
  Configuration lives in `docs/source/conf.py`; presentation lives in
  `docs/source/_static/docs.css`. The three stylesheets share the same rules.
* BESSER navigation separates start guides, task guides, concepts, reference,
  and operations. WME separates using the editor from building and maintaining
  it. Agent docs separate service setup, usage, reference and contribution.
* Existing technical reference pages and established URLs were retained.
  Introductions, installation steps and tutorials were rewritten; redundant
  editor explanations were replaced by links to the editor guide.
* Release highlights were shortened. Detailed breaking changes and deployment
  instructions moved to `source/releases/v8/migration.rst` in BESSER.
* A runnable Book example lives in `examples/documentation/first_model.py`.
  The WME first-project tutorial covers Book, Author, their association,
  validation, generation and backup.
* Read the Docs and CI use Python 3.12 for documentation and fail on warnings.
  The portable `docs/check_docs.py` replaces a shell gate that could hide a
  failed Sphinx build. WME and agent have their own documentation workflows.

Preserve this structure when adding features: a start guide should finish one
small task, a task guide should use exact UI labels, reference should contain
complete options, and operations should explain deployment and migration.
Keep prerequisites and the expected result visible. Avoid copying long
explanations between the three sites.

## Theme details to retain

* Light accent: `hsl(193, 45%, 40%)`; dark accent: `hsl(195, 52%, 64%)`.
  These match the editor's brand variables.
* Light mode uses a white header and `besser_logo_light.png`. Dark mode uses
  a navy header and `besser_logo_dark.png`. The mobile drawer needs the same
  contrast treatment as the header.
* `font: False` avoids theme font downloads. Segoe UI and Cascadia Code are
  configured through CSS, with browser fallbacks.
* Sphinx **8.2.3** and Sphinx-Immaterial **0.13.9** are pinned in all three
  requirements files. Sphinx 9.1 produced HTML successfully but its changed
  search stopword format broke the browser search interface. An upgrade must
  verify actual search results, not just a successful build.
* Do not restore `sphinx_copybutton` or `sphinx_paramlinks` without checking
  compatibility: Material already supplies related behaviour and those
  extensions conflicted during this rebuild.
* Main API docs use Napoleon custom `Attributes`/`Methods` sections and
  `napoleon_use_param=False` to avoid indexing properties twice. Keep API
  descriptions when resolving warnings; do not hide all reference members.

Change shared CSS consistently across the sites and inspect desktop, phone,
dark mode, mobile navigation and API pages after a theme change.

## Build and preview

Use an isolated Python 3.12 environment. From the parent workspace containing
`BESSER/` and `modeling-agent/`, the PowerShell setup is:

```powershell
py -3.12 -m venv .venv-docs
.\.venv-docs\Scripts\python.exe -m pip install -r BESSER/requirements.txt
.\.venv-docs\Scripts\python.exe -m pip install -r BESSER/docs/requirements.txt
.\.venv-docs\Scripts\python.exe -m pip install -r BESSER/besser/utilities/web_modeling_editor/frontend/docs/requirements.txt
.\.venv-docs\Scripts\python.exe -m pip install -r modeling-agent/docs/requirements.txt
```

Do not recreate an existing environment blindly. Main BESSER needs runtime
dependencies because Sphinx imports its Python APIs; WME and agent docs mostly
need documentation tooling. Keep documentation tooling separate from the
application environments.

Build all three sites:

```powershell
.\.venv-docs\Scripts\python.exe BESSER/docs/check_docs.py --output reviews/docs-preview/BESSER/html
.\.venv-docs\Scripts\python.exe BESSER/besser/utilities/web_modeling_editor/frontend/docs/check_docs.py --output reviews/docs-preview/WME/html
.\.venv-docs\Scripts\python.exe modeling-agent/docs/check_docs.py --output reviews/docs-preview/modeling-agent/html
python -m http.server 8021 --bind 127.0.0.1 --directory reviews/docs-preview
```

Open `http://localhost:8021/BESSER/html/`, `/WME/html/`, and
`/modeling-agent/html/`. If the review portal exists, its root links to all
three. Alternatively, from any individual repo run `python docs/check_docs.py`
and serve `docs/build/html` on a free local port.

`--offline` skips external intersphinx inventories; it does not check hosted
documentation links. The checker writes `warnings.log` beside the HTML folder,
rejects all warnings/errors, builds a fresh environment and removes the old
generated search index before rebuilding. Do not reuse HTML from a different
theme as evidence that the current theme works.

## Updating screenshots

Read WME's `docs/SCREENSHOT_HANDOFF.md` for the runnable capture workflow.
Screenshots are real browser captures from
`https://experimental.besser-pearl.org/`, not generated images or mocked chat
responses. The browser has its own fresh profile and a small My Library project.
The capture script verifies model validation, downloads Python classes and
exports a JSON backup. `--assistant` explicitly enables one real modeling
request using the default hosted assistant.

WME screenshots live in `docs/source/images/wme/v8/`. BESSER's browser guide
has copies of the welcome screen, project dialog and Book properties under
`docs/source/images/editor/`. When refreshing those shared images, copy the
same reviewed files to both repositories. Use a new release folder when the
interface changes; keep historical assets that existing pages reference.

Embed each screenshot next to the action it explains using a figure, meaningful
alt text and a short caption:

```rst
.. figure:: ../images/wme/v8/export-project.png
   :width: 700
   :alt: Export Project dialog with Class Diagram selected and Export as JSON

   Choose **Export as JSON** to save a project backup.
```

Use the correct relative path for the page. Prefer focused screenshots of
dialogs, menus or properties; crop at browser capture time to remove empty
canvas space. Do not paint over the UI, fabricate successful results, or edit
generated HTML. Confirm that figures load in the actual generated guide, not
only in the preview portal. Keep text instructions usable without the image.

## Verify before handing over

1. Build every changed site with zero warnings; run `git diff --check` in its
   repository. Review the actual diff and new assets.
2. Run new command examples. For the Python tutorial, use a temporary output
   directory and import the generated Book class to check an instance.
3. Audit local HTML targets, anchors and assets. In this original workspace,
   `python reviews/release-v8-2026-09-30/audit_html.py` writes the results.
4. Open actual guides at desktop and phone widths (tested at 1440px and 390px).
   Check image loading, readable captions, code/table scrolling and no page
   overflow. Test search, light/dark switching and opening/closing the mobile
   menu. Check browser JavaScript errors.
5. In this workspace, `browser_review.py --docs` and
   `verify_embedded_screenshots.py` under the review folder automate those
   checks. Future workspaces can reproduce them with Playwright or manual
   browser review; the capture script remains in WME's repository.
6. Check external and cross-project links against the published sites after
   publication. A local link audit cannot prove those new public URLs exist.

Record what was actually tested. A documentation review and selected editor
flows are not an exhaustive code/security audit or proof of production rollout.
The release review also found missing React assets in the original BESSER wheel
and an editor export version of `1.0.0`; check whether those have since been
resolved rather than assuming that documentation fixes addressed them.

## Commit and publish

The rebuild was prepared and verified locally. Integration-branch commits do
not publish it to Read the Docs. Before committing further changes, review the
current status in all repositories. Commit WME first;
then point BESSER's submodule at that real WME commit. Commit agent docs in its
own repo. Preview/build artifacts and downloaded example files belong outside
the source tree; include only reviewed source documentation, images and tools.

Update release PR descriptions to link to the separate migration guide. Follow
the project's current merge and publication workflow. After merging, verify
Read the Docs builds, version selection, theme assets, search and cross-project
links on the hosted sites. Keep the production worker deployment gate separate
from documentation publication.

## Verification at this handoff

* All three sites built with zero warnings during the rebuild; BESSER and WME
  were rebuilt again after the final screenshot and maintenance-guide changes.
* The final audit found no broken local targets across 335 HTML pages and
  38,076 local link/asset references.
* All seven editor captures were refreshed through the reusable script.
  Its complete `--assistant` workflow passed, including model validation,
  Python download, JSON backup and validation after the assistant edit.
* Figures in the BESSER browser guide and WME's first-project, project,
  generation and assistant guides load on desktop and 390px phone layouts,
  with no page overflow or JavaScript errors.
* These results describe the local documentation builds. Read the Docs
  publication and release packaging fixes remain separate steps.
